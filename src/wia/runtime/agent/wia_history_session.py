"""Switch WIA Phase 2 to a fresh diagnostic native conversation.

Run CPU checks: python -B -m pytest -q -p no:cacheprovider tests/test_wia_history_session.py
The caller owns the physical environment and evaluator; this wrapper never resets either.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from numbers import Real
from pathlib import Path
import shutil

from core.agent.decision import ModelError
from wia.runtime.agent.wia_history_projection import project_phase2_input


def _configuration(session):
    """Compare declared settings, excluding the intentional history condition."""
    assembler = session.assembler
    prompt = deepcopy(assembler.provenance())
    prompt.pop('wia_history_condition', None)
    executable = str(session.executable)
    executable = shutil.which(executable) or executable
    return {
        'public_task': deepcopy(session.public_task),
        'episode_configuration': deepcopy(session.episode_configuration),
        'model': session.model, 'reasoning_effort': session.reasoning_effort,
        'reasoning_summary': session.reasoning_summary, 'timeout_s': session.timeout_s,
        'executable': executable, 'snapshot_cache': session.snapshot_cache,
        'prompt_root': str(assembler.root), 'prompt_plugins': dict(assembler.plugins),
        'task_mode': assembler.task_mode, 'prompt_configuration': prompt,
        'qa_protocol': session.qa_protocol.provenance() if session.qa_protocol is not None else None,
    }


def _usage_total(*records):
    """Sum numeric usage fields; raw per-phase usage is retained separately."""
    total = {}
    for key in set().union(*(record.keys() for record in records)):
        values = [record[key] for record in records if key in record]
        if all(isinstance(value, Mapping) for value in values):
            total[key] = _usage_total(*values)
        elif all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            total[key] = sum(values)
    return total


class WiaHistorySession:
    """Keep Phase 1 intact and verify a distinct unopened Phase-2 child before inference."""

    def __init__(self, *, phase1_session, phase2_factory, condition, output, oracle_facts=None):
        if condition not in ('history_removed', 'history_oracle'):
            raise ValueError('WIA history session requires a diagnostic history condition.')
        if condition == 'history_oracle':
            if not isinstance(oracle_facts, str) or not oracle_facts.strip():
                raise ValueError('history_oracle requires nonempty caller-supplied facts.')
        elif oracle_facts is not None:
            raise ValueError('history_removed forbids oracle facts.')
        if not callable(phase2_factory):
            raise ValueError('Phase-2 factory must be callable without arguments.')
        if phase1_session.resume_state is not None or getattr(phase1_session, 'resumed_at', None) is not None:
            raise ValueError('WIA history diagnostic resume is unsupported.')
        if getattr(phase1_session.assembler, 'wia_history_condition', 'full') != 'full':
            raise ValueError('Phase 1 must retain the full ordinary prompt condition.')
        self.phase1_session = phase1_session
        self.phase2_session = None
        self.phase2_factory = phase2_factory
        self.wia_history_condition = condition
        self.output = Path(output).resolve()
        self.oracle_facts = oracle_facts
        self.resume_state = None
        self.closed = False
        self._reference_configuration = _configuration(phase1_session)
        self._configuration_verified = False
        self._observation_configuration = None
        self._phase = None
        self._last_action_step = None
        self._last_decision_step = None
        self.phase2_cutoff_action_step = None
        self.phase2_cutoff_decision_step = None
        self._phase1_close_attempted = False
        self._transition_attempted = False
        self._distinct_ids_verified = False
        self._phase1_id = self._phase2_id = None
        self._failure = None
        self._deciding = False
        self._output_logger = self._input_logger = None
        self._prepared = {}
        self._attempted_decisions = set()
        self._audits = []

    @property
    def qa_protocol(self):
        return self.phase1_session.qa_protocol

    @property
    def snapshot_cache(self):
        return self.phase1_session.snapshot_cache

    @property
    def turn_count(self):
        return self.phase1_session.turn_count + (self.phase2_session.turn_count if self.phase2_session else 0)

    @property
    def usage(self):
        return _usage_total(self.phase1_session.usage, self.phase2_session.usage if self.phase2_session else {})

    @property
    def session_id(self):
        return (self.phase2_session if self.phase2_session is not None else self.phase1_session).session_id

    def description(self):
        return {**self.phase1_session.description(), 'session': 'wia_phase_split_native_diagnostic',
                'wia_history_condition': self.wia_history_condition, 'resume_supported': False,
                'phase2_fresh_native_thread_required': True}

    def attach_output_logger(self, logger):
        self._output_logger = logger
        self.phase1_session.attach_output_logger(logger)
        if self.phase2_session is not None:
            self.phase2_session.attach_output_logger(logger)

    def attach_input_logger(self, logger):
        self._input_logger = logger
        self.phase1_session.attach_input_logger(logger)
        if self.phase2_session is not None:
            self.phase2_session.attach_input_logger(logger)

    def _save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'wia_history_session.json').write_text(
            json.dumps(self.provenance(), indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def _audit_event(self, event, audit):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / 'wia_history_request_audits.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': event, **audit}, allow_nan=False) + '\n')
        self._save()

    def _fail(self, message):
        self._failure = self._failure or message
        self._save()
        raise ModelError(message)

    def _guard(self):
        if self.closed:
            raise ModelError('An ended WIA history diagnostic cannot resume.')
        if self._failure is not None:
            raise ModelError('WIA history diagnostic failed closed: ' + self._failure)

    def _boundary(self, request):
        self._guard()
        context = request.action_context or {}
        survey = context.get('temporal_survey') or {}
        explicit, header = context.get('temporal_phase'), survey.get('current_phase')
        if explicit is not None and header is not None and explicit != header:
            self._fail('Conflicting WIA temporal phase declarations.')
        phase = explicit or header
        if phase not in ('phase_1', 'phase_2'):
            self._fail('WIA history diagnostic requires an explicit temporal phase.')
        if self._phase == 'phase_2' and phase != 'phase_2':
            self._fail('WIA temporal phase reversal is forbidden.')
        step = request.observation.action_step
        if request.sensor_inputs.action_step != step:
            self._fail('Canonical and sensor action steps must match for WIA history projection.')
        if self._last_action_step is not None and step < self._last_action_step:
            self._fail('WIA native action step regressed; resume or reset is unsupported.')
        decision = request.decision_step
        if decision is not None:
            if type(decision) is not int or decision < 0:
                self._fail('WIA decision step must be a nonnegative integer.')
            if self._last_decision_step is not None and decision < self._last_decision_step:
                self._fail('WIA decision step regressed.')
            self._last_decision_step = decision
        settings = request.observation.observation_config.to_dict()
        if request.sensor_inputs.observation_config.to_dict() != settings:
            self._fail('Canonical and sensor observation settings differ.')
        if self._observation_configuration is not None and settings != self._observation_configuration:
            self._fail('WIA observation settings changed across the history intervention.')
        self._observation_configuration = settings
        self._phase, self._last_action_step = phase, step
        return phase

    def _create_phase2(self, request):
        self.phase2_cutoff_action_step = request.observation.action_step
        self.phase2_cutoff_decision_step = request.decision_step
        try:
            self.phase2_session = self.phase2_factory()
        except Exception as error:
            self._failure = 'Phase-2 factory failed: ' + type(error).__name__
            self._save()
            raise
        child = self.phase2_session
        if child is self.phase1_session:
            self._fail('Phase 2 requires a separate native session object.')
        if (child.session_id is not None or child.turn_count != 0 or child.resume_state is not None
                or child.closed or getattr(child, 'resumed_at', None) is not None
                or getattr(child, '_temporary', None) is not None or getattr(child, '_opening', False)):
            self._fail('Phase-2 native session must be unopened, unused and non-resumed.')
        if getattr(child.assembler, 'wia_history_condition', None) != self.wia_history_condition:
            self._fail('Phase-2 prompt assembler has the wrong WIA history condition.')
        if (_configuration(self.phase1_session) != self._reference_configuration
                or _configuration(child) != self._reference_configuration):
            self._fail('Phase-2 native session changed model, public task, sensor or prompt configuration.')
        self._configuration_verified = True
        child.attach_output_logger(self._output_logger)
        child.attach_input_logger(self._input_logger)

    def prepare_request(self, request):
        phase = self._boundary(request)
        cached = self._prepared.get(id(request))
        if cached is not None and cached[0] is request:
            return request
        if phase == 'phase_1':
            return self.phase1_session.prepare_request(request)
        if self.phase2_session is None:
            self._create_phase2(request)
        count = max(0, request.observation.action_step - self.phase2_cutoff_action_step)
        history = tuple(request.observation.action_history[-count:]) if count else ()
        feedback = request.observation.last_action_feedback if count else None
        key = (request.observation.action_step, request.decision_step)
        retry = key in self._attempted_decisions
        context = dict(request.action_context or {}, temporal_phase='phase_2')
        projected, audit = project_phase2_input(replace(request, action_context=context),
            condition=self.wia_history_condition, phase2_action_history=history,
            phase2_last_action_feedback=feedback, oracle_facts=self.oracle_facts)
        if retry and request.feedback is not None:
            projected.action_context['wia_history_intervention']['phase2_retry_feedback_allowed'] = True
            projected = replace(projected, feedback=request.feedback)
        prepared = self.phase2_session.prepare_request(projected)
        audit.update(request_index=len(self._audits) + 1,
            cutoff_action_step=self.phase2_cutoff_action_step,
            cutoff_decision_step=self.phase2_cutoff_decision_step,
            source_action_step=request.observation.action_step,
            source_sensor_action_step=request.sensor_inputs.action_step,
            source_decision_step=request.decision_step,
            source_history_entries=len(request.observation.action_history),
            postcutoff_native_actions=count, available_phase2_history_entries=len(history),
            phase2_history_truncated=count > len(history),
            history_selection_rule='last_min(native_action_step_minus_cutoff, available_history_entries)',
            phase2_history=[entry.to_dict() for entry in history],
            phase2_last_action_feedback=feedback.to_dict() if feedback is not None else None,
            repeated_phase2_decision=retry, retry_feedback_preserved=prepared.feedback is not None,
            diagnostic_prompt_assembler_verified=True, decision_attempted=False,
            decision_returned=False, attempt_count=0)
        self._audits.append(audit)
        self._prepared[id(prepared)] = (prepared, audit, key)
        self._audit_event('prepared', audit)
        return prepared

    def prepare_force_choice(self, request):
        if self.qa_protocol is None:
            raise ModelError('WIA session has no QA force-choice protocol.')
        phase = self._boundary(request)
        if phase == 'phase_1':
            return self.phase1_session.prepare_force_choice(request)
        prepared = self.prepare_request(request)
        forced = replace(prepared, assembled_prompt=prepared.structured_prompt() + '\n\n'
                         + self.qa_protocol.force_choice_prompt())
        _, audit, key = self._prepared[id(prepared)]
        self._prepared[id(forced)] = (forced, audit, key)
        return forced

    async def _close_phase1(self):
        if not self._phase1_close_attempted:
            self._phase1_close_attempted = True
            await self.phase1_session.close()

    async def _open_phase2(self, request):
        if self._distinct_ids_verified:
            if self.phase2_session.session_id != self._phase2_id:
                self._fail('Phase-2 native thread ID changed after verification.')
            if (self.phase2_session.resume_state is not None or self.phase2_session.closed
                    or getattr(self.phase2_session, 'resumed_at', None) is not None
                    or _configuration(self.phase2_session) != self._reference_configuration):
                self._fail('Phase-2 native session configuration or lifecycle changed after verification.')
            return
        if self._transition_attempted:
            self._fail('A failed WIA thread transition cannot be reopened or resumed.')
        self._transition_attempted = True
        child = self.phase2_session
        self._phase1_id = self.phase1_session.session_id
        if not isinstance(self._phase1_id, str) or not self._phase1_id.strip():
            self._fail('Phase 1 has no native thread ID to distinguish from Phase 2.')
        if (child.session_id is not None or child.turn_count != 0 or child.resume_state is not None
                or getattr(child, 'resumed_at', None) is not None):
            self._fail('Phase-2 thread was opened, used or resumed before the verified transition.')
        if (_configuration(child) != self._reference_configuration
                or _configuration(self.phase1_session) != self._reference_configuration):
            self._fail('WIA native session configuration changed before thread creation.')
        await self._close_phase1()
        returned_id = await child.open(request)
        identifier = child.session_id
        if (not isinstance(identifier, str) or not identifier.strip() or returned_id != identifier
                or identifier == self._phase1_id or child.turn_count != 0
                or child.resume_state is not None or getattr(child, 'resumed_at', None) is not None):
            self._fail('Phase-2 open did not establish a distinct unused native thread before inference.')
        self._phase2_id = identifier
        self._distinct_ids_verified = True
        self._save()

    async def decide(self, request):
        self._guard()
        if self._deciding:
            self._fail('WIA native decisions must execute sequentially.')
        phase = self._boundary(request)
        if phase == 'phase_1':
            return await self.phase1_session.decide(request)
        cached = self._prepared.get(id(request))
        prepared = request if cached is not None and cached[0] is request else self.prepare_request(request)
        _, audit, key = self._prepared[id(prepared)]
        self._deciding = True
        try:
            await self._open_phase2(prepared)
            self._attempted_decisions.add(key)
            audit.update(decision_attempted=True, attempt_count=audit['attempt_count'] + 1,
                         native_context_reset_verified=True, distinct_native_thread_ids_verified=True)
            self._audit_event('decision_attempted', audit)
            reply = await self.phase2_session.decide(prepared)
            audit['decision_returned'] = True
            self._audit_event('decision_returned', audit)
            return reply
        except BaseException as error:
            if not self._distinct_ids_verified:
                self._failure = self._failure or 'WIA native thread transition failed: ' + type(error).__name__
            self._save()
            raise
        finally:
            self._deciding = False

    def provenance(self):
        first = self.phase1_session.provenance()
        second = self.phase2_session.provenance() if self.phase2_session is not None else None
        from core.agent.api_accounting import combine_accounting
        accounts = [p['request_accounting'] for p in (first, second) if p and p.get('request_accounting') is not None]
        return {
            'request_accounting': combine_accounting(accounts) if accounts else None,
            'schema': 'wia_history_session_v1', 'wia_history_condition': self.wia_history_condition,
            'resume_supported': False, 'episode_id': first.get('episode_id'),
            'model_name': self.phase1_session.model, 'reasoning_effort': self.phase1_session.reasoning_effort,
            'turns': self.turn_count, 'usage': self.usage,
            'usage_aggregation': 'sum of numeric fields; unmodified per-phase usage retained',
            'usage_by_phase': {'phase_1': deepcopy(self.phase1_session.usage),
                               'phase_2': deepcopy(self.phase2_session.usage) if self.phase2_session else {}},
            'phase_sessions': {'phase_1': first, 'phase_2': second},
            'cutoff_action_step': self.phase2_cutoff_action_step,
            'cutoff_decision_step': self.phase2_cutoff_decision_step,
            'model_invariants': {'reference': deepcopy(self._reference_configuration),
                                 'verified': self._configuration_verified},
            'observation_configuration': deepcopy(self._observation_configuration),
            'distinct_native_thread_ids_verified': self._distinct_ids_verified,
            'phase_1_native_thread_id': self._phase1_id,
            'phase_2_native_thread_id': self._phase2_id,
            'request_audits': deepcopy(self._audits),
            'delivered_history_audits': deepcopy([audit for audit in self._audits if audit['decision_attempted']]),
            'oracle_text_sha256': hashlib.sha256(self.oracle_facts.encode('utf-8')).hexdigest()
                if self.oracle_facts is not None else None,
            'oracle_factual_accuracy_verified': False, 'evaluator_state_modified': False,
            'failure': self._failure, 'closed': self.closed,
        }

    async def close(self):
        if self.closed:
            return
        try:
            try:
                await self._close_phase1()
            finally:
                if self.phase2_session is not None and self.phase2_session is not self.phase1_session:
                    await self.phase2_session.close()
        finally:
            self.closed = True
            self._save()

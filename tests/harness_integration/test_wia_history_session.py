"""Exercise native history-session ordering with CPU-only fake sessions.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_history_session.py
No model, simulator, native runtime or network is opened by these tests.
"""
import asyncio
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from core.agent.decision import DecisionInput, ModelError, ModelReply
from wia.runtime.agent.prompts import PromptAssembler
from wia.runtime.agent.wia_history_session import WiaHistorySession
from core.observations import (ActionHistoryEntry, AgentActionFeedback, AgentObservation,
                               InitialOverview, PublicTaskEvent)
from core.robot_cameras import ADDITIONAL_RGB_NAMES, ROBOT_CAMERA_LABELS
from core.sensor_observations import SensorReadings


STAMP = '2026-10-23T12:00:00+00:00'
PUBLIC_TASK = {'task_id': 'WIA-fixture', 'task_version': 1, 'objective': 'Survey and compare.',
               'success_criteria': 'Answer the public question.', 'environment': {}, 'constraints': []}
CONFIGURATION = {'episode_id': 'episode', 'max_steps': 20, 'world_mode': 'pause',
                 'world_advances_during_inference': False,
                 'observation_condition': 'current', 'action_mode': 'fixed'}
OLD_HISTORY = (ActionHistoryEntry('BASE_LEFT', 'completed'), ActionHistoryEntry('RELEASE', 'completed'))


def observation(step=8, *, history=OLD_HISTORY, feedback=None, snapshot=None):
    rgb = np.full((2, 3, 3), 40, dtype=np.uint8)
    overview = InitialOverview(rgb + 1, 'episode', 'reset', 0, 1, STAMP,
        (-2., -2., 2., 2.), (1., 0.), (0., 1.), (0., 0., 0.), 0.)
    return AgentObservation('Survey and compare.', rgb, rgb + 2, history,
        step + 2 if snapshot is None else snapshot, STAMP, step, True,
        last_action_feedback=feedback or AgentActionFeedback('RELEASE', 'completed', True, {}, {}),
        sensors=SensorReadings(head_depth_m=np.ones((2, 3), np.float32)),
        additional_rgb={name: rgb + index + 3 for index, name in enumerate(ADDITIONAL_RGB_NAMES)},
        initial_overview=overview, initial_overview_status='available',
        public_task_events=(PublicTaskEvent('event', 'history_followup', 'OLD_EVENT_CANARY', 0, 1),))


def request(phase='phase_2', step=8, decision=9, **changes):
    obs = observation(step)
    value = DecisionInput(obs, ('BASE_FWD', 'MV_DOWN', 'GRASP', 'RELEASE', 'DONE'),
        feedback='OLD_REJECTION_CANARY', action_context={'temporal_phase': phase,
            'working_arm': 'left', 'temporal_feedback': 'OLD_TEMPORAL_CANARY',
            'recovery': {'details': 'OLD_RECOVERY_CANARY'}},
        public_memory=({'note': 'OLD_MEMORY_CANARY'},),
        previous_phase_observation=observation(3),
        sensor_observation=replace(obs, snapshot_id=obs.snapshot_id + 1),
        assembled_prompt='OLD_PROMPT_CANARY', resident_prompt='OLD_RESIDENT_CANARY',
        prompt_phase='old', steps_remaining=10, decision_step=decision)
    return replace(value, **changes)


class FakeSession:
    def __init__(self, label, events, *, condition='full', new_id=None):
        self.label, self.events = label, events
        self.public_task, self.episode_configuration = dict(PUBLIC_TASK), dict(CONFIGURATION)
        self.model, self.reasoning_effort, self.reasoning_summary = 'fake-model', 'high', 'concise'
        self.timeout_s, self.executable, self.snapshot_cache = 30, '/tmp/wia-fake-codex', True
        self.assembler = PromptAssembler(wia_history_condition=condition)
        self.qa_protocol = None
        self.session_id = self.resume_state = self.resumed_at = None
        self.turn_count, self.usage, self.closed = 0, {}, False
        self.new_id = new_id or label + '-thread'
        self.prepared, self.decided = [], []
        self.input_logger = self.output_logger = None
        self.close_error = None

    def prepare_request(self, value):
        self.events.append(self.label + '.prepare')
        self.prepared.append(value)
        if self.assembler.wia_history_condition == 'full':
            return replace(value, assembled_prompt='Unchanged Phase-1 fixture prompt.')
        return self.assembler.build(value, self.public_task, self.episode_configuration,
                                    first_turn=self.turn_count == 0)

    async def open(self, value):
        self.events.append(self.label + '.open')
        assert self.session_id is None
        self.session_id = self.new_id
        return self.session_id

    async def decide(self, value):
        if self.session_id is None:
            await self.open(value)
        self.events.append(self.label + '.decide')
        self.decided.append(value)
        self.turn_count += 1
        self.usage = {'inputTokens': 7 * self.turn_count, 'outputTokens': 2 * self.turn_count,
                      'details': {'cachedTokens': self.turn_count}}
        return ModelReply('fixture reply', {'session': self.session_id})

    async def close(self):
        self.events.append(self.label + '.close')
        self.closed = True
        if self.close_error:
            raise self.close_error

    def attach_output_logger(self, logger):
        self.output_logger = logger

    def attach_input_logger(self, logger):
        self.input_logger = logger

    def description(self):
        return {'backend': 'fake', 'model': self.model}

    def provenance(self):
        return {'episode_id': 'episode', 'codex_session_id': self.session_id,
                'turns': self.turn_count, 'usage': dict(self.usage), 'closed': self.closed}


def sessions(tmp_path, *, condition='history_removed', facts=None):
    events = []
    first = FakeSession('phase1', events)
    second = FakeSession('phase2', events, condition=condition)
    made = []

    def factory():
        made.append(True)
        return second

    wrapper = WiaHistorySession(phase1_session=first, phase2_factory=factory,
        condition=condition, output=tmp_path, oracle_facts=facts)
    return wrapper, first, second, events, made


def run(coro):
    return asyncio.run(coro)


def phase1_turn(wrapper):
    value = wrapper.prepare_request(request('phase_1', step=7, decision=8))
    return run(wrapper.decide(value))


def test_phase1_passes_the_exact_request_and_prepared_request_unchanged(tmp_path):
    wrapper, first, second, events, made = sessions(tmp_path)
    source = request('phase_1', step=7, decision=8)
    source = replace(source, action_context={'temporal_survey': {'current_phase': 'phase_1'}},
                     memory_mode='model_notes')
    prepared = wrapper.prepare_request(source)
    assert first.prepared == [source]
    assert prepared.observation is source.observation
    assert prepared.feedback == source.feedback and prepared.public_memory == source.public_memory
    assert prepared.previous_phase_observation is source.previous_phase_observation
    assert prepared.memory_mode == source.memory_mode
    reply = run(wrapper.decide(prepared))
    assert first.decided == [prepared] and reply.metadata['session'] == 'phase1-thread'
    assert not made and second.turn_count == 0
    assert events == ['phase1.prepare', 'phase1.open', 'phase1.decide']


def test_first_phase2_opens_and_checks_a_distinct_thread_before_deciding(tmp_path):
    wrapper, first, second, events, made = sessions(tmp_path)
    phase1_turn(wrapper)
    prepared = wrapper.prepare_request(request())
    assert made == [True] and second.session_id is None
    assert wrapper.provenance()['distinct_native_thread_ids_verified'] is False
    assert not first.closed
    events.clear()
    run(wrapper.decide(prepared))
    assert events == ['phase1.close', 'phase2.open', 'phase2.decide']
    assert second.decided[0] is prepared
    assert wrapper.turn_count == 2
    assert wrapper.usage == {'inputTokens': 14, 'outputTokens': 4, 'details': {'cachedTokens': 2}}
    provenance = wrapper.provenance()
    assert provenance['distinct_native_thread_ids_verified'] is True
    assert provenance['phase_sessions']['phase_1']['codex_session_id'] == 'phase1-thread'
    assert provenance['phase_sessions']['phase_2']['codex_session_id'] == 'phase2-thread'
    assert provenance['request_audits'][0]['decision_attempted'] is True
    assert provenance['request_audits'][0]['native_context_reset_verified'] is True
    assert provenance['request_audits'][0]['no_history_condition_verified'] is False
    assert json.loads((tmp_path / 'wia_history_session.json').read_text()) == provenance


def test_phase2_boundary_removes_all_explicit_history_but_preserves_current_sensors(tmp_path):
    wrapper, _, second, _, _ = sessions(tmp_path)
    source = request()
    projected = wrapper.prepare_request(source)
    assert projected.previous_phase_observation is None
    assert projected.public_memory == () and projected.memory_mode == 'none'
    assert projected.feedback is None
    assert 'OLD_' not in projected.structured_prompt()
    assert 'OLD_' not in json.dumps(projected.action_context)
    for original, current in ((source.observation, projected.observation),
                              (source.sensor_inputs, projected.sensor_inputs)):
        assert current.initial_overview is None and current.public_task_events == ()
        assert current.action_history == () and current.last_action_feedback is None
        assert (current.snapshot_id, current.action_step) == (original.snapshot_id, original.action_step)
        for name in ROBOT_CAMERA_LABELS:
            assert current.rgb_images()[name].tobytes() == original.rgb_images()[name].tobytes()
        assert current.sensors.head_depth_m.tobytes() == original.sensors.head_depth_m.tobytes()
    assert source.observation.initial_overview is not None
    assert source.previous_phase_observation is not None
    assert second.prepared[0].assembled_prompt is None
    assert wrapper.phase2_cutoff_action_step == 8
    audit = wrapper.provenance()['request_audits'][0]
    assert audit['source_history_entries'] == 2 and audit['available_phase2_history_entries'] == 0
    assert audit['cutoff_action_step'] == 8 and not audit['decision_attempted']


def test_later_phase2_history_and_measured_feedback_are_kept_without_reprojection(tmp_path):
    wrapper, _, second, _, _ = sessions(tmp_path)
    phase1_turn(wrapper)
    run(wrapper.decide(wrapper.prepare_request(request())))
    entry = ActionHistoryEntry('BASE_FWD', 'completed', distance_m=.1)
    feedback = AgentActionFeedback('BASE_FWD', 'completed', True, {}, {})
    obs = observation(9, history=OLD_HISTORY + (entry,), feedback=feedback)
    raw = request(step=9, decision=10, observation=obs, sensor_observation=replace(obs, snapshot_id=12))
    prepared = wrapper.prepare_request(raw)
    for current in (prepared.observation, prepared.sensor_inputs):
        assert current.action_history == (entry,) and current.last_action_feedback is feedback
    assert prepared.feedback is None
    count = len(second.prepared)
    assert wrapper.prepare_request(prepared) is prepared
    run(wrapper.decide(prepared))
    assert second.decided[-1] is prepared and len(second.prepared) == count
    assert wrapper.provenance()['request_audits'][-1]['postcutoff_native_actions'] == 1


def test_retry_feedback_requires_an_actual_attempt_of_the_same_phase2_decision(tmp_path):
    wrapper, _, second, _, _ = sessions(tmp_path)
    phase1_turn(wrapper)
    first = wrapper.prepare_request(request())
    same_not_attempted = wrapper.prepare_request(request(feedback='Not yet a real retry.'))
    assert same_not_attempted.feedback is None
    run(wrapper.decide(first))
    retried = wrapper.prepare_request(request(feedback='Invalid command magnitude.'))
    assert retried.feedback == 'Invalid command magnitude.'
    assert retried.action_context['wia_history_intervention']['phase2_retry_feedback_allowed'] is True
    assert 'recovery' not in retried.action_context and 'temporal_feedback' not in retried.action_context
    assert retried.observation.last_action_feedback is None
    run(wrapper.decide(retried))
    assert second.decided[-1] is retried
    next_decision = wrapper.prepare_request(request(decision=10, feedback='OLD_REJECTION_CANARY'))
    assert next_decision.feedback is None


def test_oracle_facts_are_verbatim_and_not_certified_as_factually_correct(tmp_path):
    facts = '  Before the intervention, the blue cup was on the left.\nOnly this fact is supplied.  '
    wrapper, _, _, _, _ = sessions(tmp_path, condition='history_oracle', facts=facts)
    prepared = wrapper.prepare_request(request())
    assert prepared.action_context['diagnostic_historical_facts'] == facts
    provenance = wrapper.provenance()
    assert provenance['oracle_text_sha256'] == hashlib.sha256(facts.encode()).hexdigest()
    assert provenance['oracle_factual_accuracy_verified'] is False


@pytest.mark.parametrize('field,value', [
    ('model', 'other-model'), ('reasoning_effort', 'low'), ('reasoning_summary', 'none'),
    ('timeout_s', 40), ('executable', '/tmp/other-fake-codex'), ('snapshot_cache', False),
    ('public_task', {**PUBLIC_TASK, 'objective': 'Changed objective.'}),
    ('episode_configuration', {**CONFIGURATION, 'max_steps': 21}),
])
def test_phase2_configuration_differences_fail_before_a_thread_is_opened(tmp_path, field, value):
    wrapper, _, second, events, _ = sessions(tmp_path)
    setattr(second, field, value)
    with pytest.raises(ModelError, match='configuration'):
        wrapper.prepare_request(request())
    assert 'phase2.open' not in events and not second.decided
    assert wrapper.provenance()['model_invariants']['verified'] is False


@pytest.mark.parametrize('change', ['id', 'used', 'resumed', 'wrong_assembler'])
def test_phase2_factory_must_produce_an_unopened_nonresumed_matching_diagnostic(tmp_path, change):
    wrapper, _, second, events, _ = sessions(tmp_path)
    if change == 'id':
        second.session_id = 'existing-thread'
    elif change == 'used':
        second.turn_count = 1
    elif change == 'resumed':
        second.resume_state = {'existing': True}
    else:
        second.assembler = PromptAssembler()
    with pytest.raises(ModelError):
        wrapper.prepare_request(request())
    assert 'phase2.open' not in events and not second.decided


@pytest.mark.parametrize('new_id', ['phase1-thread', '', None])
def test_same_or_missing_native_thread_id_never_reaches_phase2_decide(tmp_path, new_id):
    wrapper, _, second, events, _ = sessions(tmp_path)
    phase1_turn(wrapper)
    second.new_id = new_id
    prepared = wrapper.prepare_request(request())
    with pytest.raises(ModelError, match='distinct unused'):
        run(wrapper.decide(prepared))
    assert 'phase2.open' in events and not second.decided and second.turn_count == 0
    assert wrapper.provenance()['distinct_native_thread_ids_verified'] is False
    with pytest.raises(ModelError, match='failed closed'):
        run(wrapper.decide(prepared))
    assert events.count('phase2.open') == 1


def test_phase1_resume_and_factory_reuse_are_rejected(tmp_path):
    wrapper, first, _, _, _ = sessions(tmp_path)
    first.resume_state = {'old': True}
    with pytest.raises(ValueError, match='resume is unsupported'):
        WiaHistorySession(phase1_session=first, phase2_factory=lambda: first,
                          condition='history_removed', output=tmp_path)
    first.resume_state = None
    wrapper.phase2_factory = lambda: first
    with pytest.raises(ModelError, match='separate native session'):
        wrapper.prepare_request(request())


@pytest.mark.parametrize('change', ['phase_reversal', 'action_regression', 'decision_regression', 'sensor_step'])
def test_phase_and_step_regressions_and_mixed_sensor_boundaries_fail_closed(tmp_path, change):
    wrapper, _, _, _, _ = sessions(tmp_path)
    wrapper.prepare_request(request())
    if change == 'phase_reversal':
        invalid = request('phase_1')
    elif change == 'action_regression':
        invalid = request(step=7)
    elif change == 'decision_regression':
        invalid = request(decision=8)
    else:
        invalid = request(sensor_observation=observation(9))
    with pytest.raises(ModelError):
        wrapper.prepare_request(invalid)
    with pytest.raises(ModelError, match='failed closed'):
        wrapper.prepare_request(request())


def test_native_identity_and_configuration_are_rechecked_after_prepare_before_open(tmp_path):
    wrapper, _, second, events, _ = sessions(tmp_path)
    phase1_turn(wrapper)
    prepared = wrapper.prepare_request(request())
    second.session_id = 'externally-opened-thread'
    with pytest.raises(ModelError, match='before the verified transition'):
        run(wrapper.decide(prepared))
    assert 'phase2.open' not in events and not second.decided


def test_model_configuration_cannot_change_after_the_distinct_thread_check(tmp_path):
    wrapper, _, second, _, _ = sessions(tmp_path)
    phase1_turn(wrapper)
    prepared = wrapper.prepare_request(request())
    run(wrapper.decide(prepared))
    second.model = 'changed-after-first-turn'
    with pytest.raises(ModelError, match='after verification'):
        run(wrapper.decide(prepared))
    assert second.turn_count == 1


def test_close_both_phases_once_and_forward_loggers_including_late_child_creation(tmp_path):
    wrapper, first, second, events, _ = sessions(tmp_path)
    sink = lambda value: value
    wrapper.attach_input_logger(sink)
    wrapper.attach_output_logger(sink)
    phase1_turn(wrapper)
    prepared = wrapper.prepare_request(request())
    assert first.input_logger is second.input_logger is sink
    assert first.output_logger is second.output_logger is sink
    run(wrapper.decide(prepared))
    run(wrapper.close())
    run(wrapper.close())
    assert events.count('phase1.close') == events.count('phase2.close') == 1
    assert wrapper.closed and first.closed and second.closed
    wrapper.attach_input_logger(None)
    wrapper.attach_output_logger(None)
    assert first.input_logger is second.input_logger is None
    assert wrapper.description()['resume_supported'] is False
    assert wrapper.snapshot_cache is True and wrapper.qa_protocol is None


def test_close_still_closes_child_if_phase1_cleanup_raises(tmp_path):
    wrapper, first, second, events, _ = sessions(tmp_path)
    wrapper.prepare_request(request())
    first.close_error = RuntimeError('fixture cleanup failure')
    with pytest.raises(RuntimeError, match='cleanup failure'):
        run(wrapper.close())
    assert events[-2:] == ['phase1.close', 'phase2.close'] and second.closed
    assert wrapper.closed


def test_bounded_native_history_suffix_remains_phase2_only_and_records_truncation(tmp_path):
    wrapper, _, _, _, _ = sessions(tmp_path)
    wrapper.prepare_request(request())
    later = observation(23, history=(ActionHistoryEntry('BASE_FWD', 'completed'),))
    prepared = wrapper.prepare_request(request(step=23, decision=24, observation=later,
                                               sensor_observation=None))
    assert prepared.observation.action_history == later.action_history
    audit = wrapper.provenance()['request_audits'][-1]
    assert audit['postcutoff_native_actions'] == 15
    assert audit['available_phase2_history_entries'] == 1 and audit['phase2_history_truncated']

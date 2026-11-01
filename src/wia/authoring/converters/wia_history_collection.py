"""Native two-phase historical multi-object collection evaluator.

Run CPU checks: python -m pytest tests/test_wia_history_collection.py
Dispatch through temporal_understanding_agent_observation using operation
``history_multi_retrieval_v1``; native readers never issue simulator actions.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from wia.authoring.io import write_json
from wia.tasks.completion import progress
from wia.tasks.temporal_policy import informative_temporal_rows
from wia.tasks.wia_history_collection import (
    public_collection_query, seeded_collection_query, validate_collection_parameters)
from .temporal_understanding_agent_observation import (
    NativePlacementProgress, native_protected_inventory, native_temporal_placement,
    protected_inventory_preserved, qualify_history_retrieval)

READ_ERRORS = (ValueError, AttributeError, RuntimeError, KeyError, TypeError)
EMPTY_GRIPPER = {'holding': False, 'constraint_active': False,
                 'release_pending': False, 'object_name': None}


def _valid_gripper(grasp):
    return (isinstance(grasp, dict) and 'object_name' in grasp
        and all(type(grasp.get(key)) is bool for key in ('holding', 'constraint_active', 'release_pending'))
        and (grasp['object_name'] is None or isinstance(grasp['object_name'], str)))


def _empty_gripper(grasp):
    return _valid_gripper(grasp) and all(grasp.get(key) == value for key, value in EMPTY_GRIPPER.items())


def qualify_collection(session, executor, config, bound):
    """Retain native inventory/support qualification for the explicit new operation."""
    try:
        validate_collection_parameters(config['parameters'])
    except READ_ERRORS as error:
        return {'accepted': False, 'reason': 'invalid_history_collection_contract', 'error': str(error)}
    return qualify_history_retrieval(session, executor, config, bound)


class HistoryCollectionEvaluator:
    """Freeze verified first encounters; credit each requested native instance once."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        validate_collection_parameters(self.parameters)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history, self.seen, self.ties = [], set(), []
        self.frozen, self.query, self.public_query = None, None, None
        self.selected_keys, self.placements = (), {}
        self.active, self.visual_available = False, False
        self.constraint_violation, self.protection_available = False, True
        self.grasp_available, self.last_gripper = True, None
        self.delivery_status, self.physical_available = {}, False
        self.answer = None
        self.last_boundary = None

    def _protected(self):
        return native_protected_inventory(self.session, self.config['bindings'], self.config['binding_keys'])

    def _placement(self, key):
        return native_temporal_placement(self.session, self.controller, self.config,
            key, self.parameters['destination_key'])

    def begin_episode(self, observation):
        self.initial = self._protected()
        self.active = True
        return self.record(observation)

    def _preserved(self):
        try:
            allowed = self.selected_keys if self.frozen is not None else ()
            if not protected_inventory_preserved(self.initial, self._protected(), except_keys=allowed):
                self.constraint_violation = True
        except READ_ERRORS:
            self.protection_available = False

    def _observe(self, observation):
        evidence = informative_temporal_rows(self.session, observation, self.config['bindings'], self.parameters)
        self.visual_available = evidence['available']
        if self.frozen is None and evidence['available']:
            fresh = set(evidence['objects']).intersection(self.parameters['target_keys']) - self.seen
            # Storage order inside a snapshot has no chronological meaning. Query
            # validation rejects same-category ties, including across repeat reads.
            for key in sorted(fresh):
                self.history.append({'key': key, 'snapshot_id': observation.snapshot_id,
                    'action_step': observation.action_step, 'evidence': deepcopy(evidence['objects'][key])})
                self.seen.add(key)
            groups = {}
            for row in self.history:
                boundary = (row['snapshot_id'], self.parameters['target_categories'][row['key']])
                groups.setdefault(boundary, []).append(row['key'])
            self.ties = [{'snapshot_id': snapshot, 'category': category, 'targets': keys}
                for (snapshot, category), keys in groups.items() if len(keys) > 1]
        self.last_boundary = (observation.snapshot_id, observation.action_step)
        return evidence

    def _read_grasp(self):
        try:
            grasp = self.controller.debug_gripper_state()
            if not _valid_gripper(grasp):
                raise ValueError('Native grasp telemetry unavailable.')
            self.last_gripper = deepcopy(grasp)
            if self.frozen is None:
                if not _empty_gripper(grasp):
                    self.constraint_violation = True
            elif grasp['holding'] or grasp['constraint_active'] or grasp['object_name'] is not None:
                allowed = {self.config['bindings'][key] for key in self.selected_keys}
                if grasp['object_name'] not in allowed:
                    self.constraint_violation = True
            return grasp
        except READ_ERRORS:
            self.grasp_available = False
            self.last_gripper = None
            return None

    def complete_survey(self, observation):
        if not self.active:
            raise RuntimeError('Survey completion requires an active episode.')
        if self.frozen is not None:
            raise ValueError('The encounter history has already been frozen.')
        self._observe(observation)
        self._preserved()
        self._read_grasp()
        if not self.visual_available or self.seen != set(self.parameters['target_keys']):
            raise ValueError('The survey lacks informative encounters for every target.')
        if self.constraint_violation or not self.protection_available or not self.grasp_available:
            raise ValueError('Survey preservation and empty gripper are not verified.')
        query = seeded_collection_query(self.history, self.parameters)
        public_query = public_collection_query(query, self.parameters['destination_label'])
        events = getattr(self.session, 'public_task_events', None)
        if events is None:
            events = []
            self.session.public_task_events = events
        if not isinstance(events, list):
            raise ValueError('Public follow-up event transport must be a list.')
        self.frozen = tuple(deepcopy(self.history))
        self.query, self.public_query = query, public_query
        self.selected_keys = tuple(query['selected_keys'])
        self.placements = {key: NativePlacementProgress(self.parameters) for key in self.selected_keys}
        events.append({'event_id': 'history_multi_followup_' + str(len(events) + 1),
            # Use the declared PublicTaskEvent transport envelope. Requested
            # categories/counts and the historical rule are fully in its text.
            'kind': 'history_followup', 'text': public_query['text'],
            'action_step': observation.action_step, 'snapshot_id': observation.snapshot_id})
        return {'survey_complete': True, 'followup_question': public_query['text'],
                'query': deepcopy(public_query)}

    def report_answer(self, answer, observation):
        if str(answer).strip() == 'SURVEY_COMPLETE':
            return self.complete_survey(observation)
        if self.frozen is None:
            raise ValueError('The historical collection question is unavailable before survey completion.')
        # Retain optional text for auditing; no text can confer physical credit.
        self.answer = str(answer).strip()

    def _update(self, *, count_step=False):
        self._preserved()
        actual_grasp = self._read_grasp()
        if self.frozen is None:
            return
        self.delivery_status = {}
        available = actual_grasp is not None
        selected_names = {self.config['bindings'][key] for key in self.selected_keys}
        for key, tracker in self.placements.items():
            try:
                measured = self._placement(key)
            except READ_ERRORS as error:
                measured = {'measurement_error': str(error)}
            measured = deepcopy(measured)
            grasp = measured.get('gripper')
            if not _valid_gripper(grasp) or grasp != actual_grasp:
                measured['gripper'] = None
            else:
                name = self.config['bindings'][key]
                if (grasp['object_name'] != name and grasp['object_name'] in selected_names
                        and grasp['holding'] is True and grasp['constraint_active'] is True
                        and grasp['release_pending'] is False):
                    # This target is released while another requested target is
                    # held. Preserve its own dwell using target-relative release.
                    measured['actual_gripper'] = deepcopy(grasp)
                    measured['gripper'] = deepcopy(EMPTY_GRIPPER)
                    measured['release_evidence_scope'] = 'target_released_other_requested_item_held'
            item_available, placed = tracker.update(measured, self.config['bindings'][key], count_step=count_step)
            available &= item_available and tracker.selection_available
            self.delivery_status[key] = bool(tracker.selected and placed)
        self.physical_available = bool(available)

    def sample_step(self):
        if self.active:
            self._update(count_step=True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        visual = self._observe(observation)
        self._update(count_step=False)
        queried = self.frozen is not None
        identities = bool(queried and all(tracker.selected for tracker in self.placements.values()))
        delivered = bool(queried and all(self.delivery_status.values()) and _empty_gripper(self.last_gripper))
        identity_available = self.grasp_available and all(
            tracker.selection_available for tracker in self.placements.values())
        items = [
            {'id': 'survey', 'completed': self.seen == set(self.parameters['target_keys']), 'kind': 'checkpoint',
             'evaluation_validity': 'valid' if self.visual_available or queried else 'unavailable'},
            {'id': 'delayed_history_query', 'completed': queried, 'kind': 'checkpoint', 'evaluation_validity': 'valid'},
            {'id': 'requested_native_identities', 'completed': identities, 'kind': 'checkpoint',
             'evaluation_validity': 'valid' if identity_available else 'unavailable'},
            {'id': 'all_requested_items_delivered', 'completed': delivered, 'kind': 'terminal_condition',
             'evaluation_validity': 'valid' if not queried or self.physical_available else 'unavailable',
             'reason': 'query_not_opened' if not queried else None}]
        available = self.protection_available and self.grasp_available and all(
            item['evaluation_validity'] == 'valid' for item in items)
        result = {'schema': 'history_multi_retrieval_v1', 'completion_items': items,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'evaluation_validity': 'complete' if available else 'partial',
            'task_success': bool(available and all(item['completed'] for item in items) and not self.constraint_violation),
            'constraints_satisfied': not self.constraint_violation if self.protection_available and self.grasp_available else None,
            'encounter_history': deepcopy(self.history), 'ambiguous_first_encounters': deepcopy(self.ties),
            'frozen_history': deepcopy(self.frozen), 'private_query': deepcopy(self.query),
            'public_query': deepcopy(self.public_query), 'history_selected_bindings': list(self.selected_keys),
            'reported_answer': self.answer, 'visual_evidence': visual, 'native_gripper': self.last_gripper,
            'observed_target_count': len(self.seen), 'declared_target_count': len(self.parameters['target_keys']),
            'requested_count': len(self.selected_keys), 'delivered_count': sum(self.delivery_status.values()),
            'per_item': {key: {'native_identity_verified': tracker.selected,
                'currently_delivered': self.delivery_status.get(key, False),
                'release_stable_steps': tracker.stable_steps, 'native_placement': tracker.last}
                for key, tracker in self.placements.items()}}
        result.update(progress(result))
        write_json(self.output / 'state.json', result)
        return result

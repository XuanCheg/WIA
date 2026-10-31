"""Pending canonical Mapping worlds, with reset-bound native qualification.

Run: python -B -m pytest -q tests/test_task_edit2_mapping_world.py
Native integration supplies two read-only session providers taking the private
contract: mapping_joint_world_identity(contract) and
mapping_joint_world_qualification(contract). The latter may instead be a dict.
Neither this module nor its construction hook changes native actors or doors.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from wia.authoring.converters.composite_task import CompositeEvaluator
from wia.authoring.io import write_json

MODULE = 'wia.authoring.converters.task_edit2_mapping_world'
REVISION = 'task_edit2_mapping_world_20261018_v1'
MEASUREMENTS = {
    'geometry': ('loaded_rooms', 'object_aabbs', 'boundary_identity'),
    'inventory': ('native_objects', 'source_inventory_evidence'),
    'doors': ('native_door_states', 'retention_evidence', 'source_policy_reconciliation'),
    'start': ('robot_pose', 'snapshot_id', 'support_and_clearance_evidence'),
    'controller': ('configuration', 'public_control_trial_evidence'),
    'mounted_cameras': ('camera_ids', 'snapshot_id', 'rgb_identities'),
    'initial_overview': ('snapshot_id', 'rgb_identity', 'reset_id'),
}
GATES = ('full_building_loaded', 'native_furniture_and_doors_retained',
         'common_start_qualified', 'public_controller_qualified',
         'all_eight_and_initial_global_delivered', 'source_relations_revalidated')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def configure_scene(config, parameters):
    """Request the complete source building without replacing or adding assets."""
    contract = parameters['mapping_joint_world_contract']
    if contract['revision'] != REVISION or contract['room'] != 'full_scene':
        raise ValueError('Canonical Mapping initialization requires a full-building contract.')
    native = config.setdefault('scene', {})
    removed = {}
    for key in ('load_room_instances', 'load_room_types', 'load_object_categories',
                'not_load_object_categories'):
        if key in native:
            removed[key] = native.pop(key)
    return {'full_building_requested': True, 'cleared_filters': removed,
            'assets_added': [], 'native_qualification': 'pending'}


def _present(value):
    return value is not None and value is not False and value != '' and value != {} and value != []


def joint_world_qualification(session, controller, config):
    """Reject absent/stale proof; compare its measurements to a live read-only provider.

    The provider must return episode_id/reset_id/seed, scene, and measurements.
    Measurements include full native geometry/inventory and original door states,
    settled initial start evidence, controller configuration, and delivered frozen
    eight-camera/Global identities. A receipt additionally binds contract_sha256,
    exact source_members, source_relations, explicit gates, and budget_calibration.
    This API supplies no graph, room, doorway, or answer truth of its own.
    """
    try:
        from wia.tasks.release_policy import fixed_release_evaluation
        if fixed_release_evaluation(session, config):
            return {'accepted': True, 'evaluation_validity': 'complete', 'reason': None,
                    'policy': 'fixed_v1_definition', 'physical_checks_run': False,
                    'contract_sha256': digest(config['parameters']['mapping_joint_world_contract'])}
        contract = config['parameters']['mapping_joint_world_contract']
        if contract['revision'] != REVISION:
            raise ValueError('Unknown canonical-world revision.')
        actual_members = sorted((m['source_identity'], m['source_sha256'])
            for m in config['parameters']['members'])
        if actual_members != sorted((m['source_identity'], m['source_sha256']) for m in contract['source_members']):
            raise ValueError('Evaluator members differ from the canonical-world source contract.')
        provider = getattr(session, 'mapping_joint_world_identity', None)
        receipt = getattr(session, 'mapping_joint_world_qualification', None)
        if not callable(provider) or receipt is None:
            raise ValueError('Native joint-world measurement provider or receipt is unavailable.')
        live = deepcopy(provider(deepcopy(contract)))
        proof = deepcopy(receipt(deepcopy(contract)) if callable(receipt) else receipt)
        reset = session.initial_overview_identity
        if not isinstance(reset, dict) or any(not _present(reset.get(k)) for k in ('episode_id', 'reset_id')):
            raise ValueError('The actual initialized episode/reset identity is unavailable.')
        for key in ('episode_id', 'reset_id', 'seed'):
            if key not in reset or live.get(key) != reset[key] or proof.get(key) != reset[key]:
                raise ValueError('Joint-world proof has a different ' + key + '.')
        if reset['seed'] != contract['seed']:
            raise ValueError('Native initialization seed differs from the declared common start.')
        if live.get('scene') != contract['scene'] or proof.get('scene') != contract['scene']:
            raise ValueError('Native scene does not match the canonical building.')
        if proof.get('contract_sha256') != digest(contract):
            raise ValueError('Joint-world contract identity differs.')
        expected = sorted(contract['source_members'], key=lambda x: x['source_identity'])
        supplied = proof.get('source_members')
        if not isinstance(supplied, list) or sorted(supplied, key=lambda x: x['source_identity']) != expected:
            raise ValueError('Native proof has different source membership or source hashes.')
        measured = live.get('measurements', {})
        for domain, fields in MEASUREMENTS.items():
            row = measured.get(domain)
            if not isinstance(row, dict) or any(not _present(row.get(k)) for k in fields):
                # Empty native_door_states is legitimate when a measured inventory
                # proves that the complete native building has no door objects.
                if not (domain == 'doors' and isinstance(row, dict)
                        and row.get('native_door_states') == []
                        and row.get('native_door_count') == 0
                        and all(_present(row.get(k)) for k in fields[1:])):
                    raise ValueError('Incomplete native measurement: ' + domain)
        cameras = measured['mounted_cameras']
        if (len(cameras['camera_ids']) != 8 or len(set(cameras['camera_ids'])) != 8
                or set(cameras['rgb_identities']) != set(cameras['camera_ids'])
                or any(not _present(v) for v in cameras['rgb_identities'].values())):
            raise ValueError('Exactly eight identified delivered mounted RGB views are required.')
        if measured['initial_overview']['reset_id'] != reset['reset_id']:
            raise ValueError('Initial Global belongs to a different reset.')
        if measured['initial_overview']['snapshot_id'] != cameras['snapshot_id']:
            raise ValueError('Initial Global and all eight views must share the initialized observation boundary.')
        if proof.get('measurements') != measured:
            raise ValueError('Native joint-world measurements changed or differ from the receipt.')
        if any(proof.get('gates', {}).get(key) is not True for key in GATES):
            raise ValueError('Native joint-world gates are incomplete.')
        relations = proof.get('source_relations', [])
        if (len(relations) != len(expected)
                or sorted((r.get('source_identity'), r.get('source_sha256')) for r in relations)
                != sorted((r['source_identity'], r['source_sha256']) for r in expected)
                or any(r.get('verified') is not True or not _present(r.get('native_evidence')) for r in relations)):
            raise ValueError('Every original source relation needs its own native revalidation evidence.')
        budget = proof.get('budget_calibration', {})
        if (budget.get('verified') is not True or budget.get('budget_sha256') != digest(contract['episode_budget'])
                or not _present(budget.get('native_trial_evidence'))):
            raise ValueError('The shared episode budget has no matching native calibration.')
        return {'accepted': True, 'evaluation_validity': 'complete', 'reason': None,
                'contract_sha256': digest(contract), 'native_receipt_sha256': digest(proof)}
    except Exception as error:
        return {'accepted': False, 'evaluation_validity': 'unavailable',
                'reason': f'{type(error).__name__}: {error}'}


def qualify_operation(session, controller, config, bound):
    from wia.authoring.converters.composite_task import qualify_operation as qualify_children
    result = joint_world_qualification(session, controller, config)
    if not result['accepted']:
        return result
    children = qualify_children(session, controller, config, bound)
    result['members'] = children['members']
    result['accepted'] = children['all_members_qualified']
    if not result['accepted']:
        result.update(evaluation_validity='unavailable', reason='An original child native qualifier is unavailable.')
    return result


class MappingJointWorldEvaluator(CompositeEvaluator):
    """Use the existing answer sheet and child reducers, gated on exact native proof."""
    def __init__(self, session, controller, config, output):
        runtime = deepcopy(config)
        self.original_config = deepcopy(config)
        # Only this private runtime copy uses CompositeEvaluator's explicit
        # compatibility extension. Persisted source configs remain byte-for-byte.
        for member in runtime['parameters']['members']:
            member['evaluation']['parameters']['mapping_composition_contract'] = {
                'compatibility_verified': False}
        super().__init__(session, controller, runtime, output)

    def record(self, observation, *, action=None):
        proof = joint_world_qualification(self.session, self.controller, self.original_config)
        for member in self.members:
            member['evaluation']['parameters']['mapping_composition_contract'] = {
                'compatibility_verified': proof['accepted'],
                'joint_contract_sha256': digest(self.original_config['parameters']['mapping_joint_world_contract'])}
        row = super().record(observation, action=action)
        row['mapping_joint_world_qualification'] = proof
        if not proof['accepted']:
            row.update(task_success=False, evaluation_validity='unavailable',
                answer_evaluation_validity='unavailable', answer_accuracy_percent=None,
                score_percent=None, completion_score=None, obligation_score_percent=None,
                diagnostic_answer_lower_bound_percent=None,
                failure_stage='mapping_joint_world_qualification', failure_reason=proof['reason'])
            for item in row.get('completion_items') or []:
                item['evaluation_validity'] = 'unavailable'
            for report in row.get('per_subclass', {}).values():
                report.update(answer_accuracy_percent=None, answer_evaluation_validity='unavailable',
                              diagnostic_answer_lower_bound_percent=None)
            for question in row.get('semantic_questions', []):
                question['evaluation_validity'] = 'unavailable'
            physical = row.get('physical_obligations')
            if physical:
                physical.update(evaluation_validity='unavailable', score_percent=None, completion_score=None)
                for item in physical.get('completion_items') or []:
                    item['evaluation_validity'] = 'unavailable'
        write_json(self.output / 'state.json', row)
        return row


def create_evaluator(session, controller, config, output):
    return MappingJointWorldEvaluator(session, controller, config, output)

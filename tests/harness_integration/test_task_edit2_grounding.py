"""CPU contracts for public informative inspection before original-object retrieval.

Run: python -B -m pytest -q tests/test_task_edit2_grounding.py
Fixtures use explicit private feature receipts; no mask area implies recognition.
No simulator, GPU, renderer, source edit, queue or Review mutation is performed.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from wia.authoring.conversion import ROOT, SourceInstance, ConversionContext
from wia.authoring.converters import partial_occlusion as partial
from wia.authoring.converters import perceptual_grounding_view_hallucination as view
from core.robot_cameras import ROBOT_CAMERA_LABELS
from wia.native.esi_category_evaluation import create_category_operation_evaluator, CategoryEvaluationEnvelope
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from wia.runtime.commands.build_task_edit2_grounding_candidates import build_candidates


class Actor:
    def __init__(self, name):
        self.name, self.category, self.model = name, 'fixture_category', 'fixture_model'
        self.position, self.quaternion = np.array([1., 0., .7]), np.array([0., 0., 0., 1.])
        self.scale, self.materials = np.ones(3), []
        self.fixed_base, self.visual_only = False, False
        self.prim_path = '/World/' + name

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()


@pytest.fixture
def world(tmp_path, monkeypatch):
    actors = {key: Actor(key) for key in ('target', 'occluder', 'destination', 'support', 'occluder_rest')}
    state = {'feature': False, 'audited': True, 'global_easy': False, 'global_review': True,
             'sequence': 0, 'head': [0., 0., 1.], 'history': [], 'step': 0,
             'gripper': {'holding': False, 'constraint_active': False, 'release_pending': False, 'object_name': None},
             'destination_relation': False, 'rest_relation': False, 'linear_speed_m_s': 0.,
             'angular_speed_rad_s': 0., 'robot_withdrawn': True, 'measurement_errors': {}}
    global_context = {'task_id': 'task', 'task_sha256': 'task_sha', 'source_identity': 'fixture/source',
        'source_sha256': 'source_sha', 'scene_id': 'scene', 'scene_sha256': 'scene_sha',
        'episode_id': 'episode', 'reset_id': 'reset', 'seed': 0, 'rgb_sha256': 'frozen_sha'}
    receipt = {'schema': 'native_frozen_initial_input_audit_v1', 'audit_id': 'frozen_audit',
        'context': global_context, 'target_identity_audits': {'target': {
            'native_name': 'target', 'informative': False, 'feature_ids': []}}}
    session = NS(scene_name='scene', room_name='room', episode_id='episode', native_reset_id='reset',
        scene_camera_position=[0., 0., 1.],
        scene_camera_quaternion=[0., 0., 0., 1.],
        category_interaction_setup={'source_reference_camera_translation_m': [0., 0., 0.]},
        env=NS(scene=NS(objects=actors, object_registry=lambda field, name: actors.get(name))),
        visual_evidence_frozen_input_receipts=[receipt])

    def observation():
        return NS(snapshot_id=state['step'] + 1, action_step=state['step'], task_text='fixture',
            rgb_images=lambda: {**{name: np.zeros((4, 4, 3), np.uint8) for name in ROBOT_CAMERA_LABELS},
                               'initial_overview_rgb': np.zeros((4, 4, 3), np.uint8)})

    def visual(observation, bindings, parameters):
        context = dict(global_context, snapshot_id=observation.snapshot_id, action_step=observation.action_step)
        rows = {}
        for camera in ROBOT_CAMERA_LABELS:
            informative = state['feature'] and camera == 'agentview_rgb'
            rows[camera] = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                'camera': camera, 'native_name': 'target', 'native_rgb_matches_delivered': True,
                'identity_synchronized': True, 'audited': state['audited'], 'informative': informative,
                'feature_audit_id': 'exact_feature_audit' if informative else None,
                'visible_pixels': 1, 'minimum_extent_px': 1,
                'informative_feature_evidence': {'semantic_identity_verified': True, 'view_kind': 'direct',
                    'feature_ids': ['diagnostic_handle']} if informative else {},
                'audit_context': dict(context, camera=camera, native_identity={'name': 'target'},
                    robot_pose={'position_m': [state['head'][0], 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.]})}
        return {'schema': 'native_visual_evidence_v1', 'recognition_inferred_from_mask_size': False,
            'measurements': {'target': {'views': rows}}, 'cameras': {'agentview_rgb': {
                'camera_pose': {'position_m': list(state['head']), 'quaternion_xyzw': [0., 0., 0., 1.]}}},
            'native_capture_context': context,
            'native_public_control_context': {'available': True, 'state_setter_used': False,
                                              'control_sequence': state['sequence']},
            'public_action_history': list(state['history']),
            'frozen_views': {'initial_overview_rgb': {'initial_input_reviewed': state['global_review'],
                'audit_id': 'frozen_audit', 'audit_context': global_context}}}

    session.task_visual_evidence = visual

    def measure(session, controller, config, target, destination):
        result = {key: deepcopy(state[key]) for key in ('gripper', 'linear_speed_m_s',
            'angular_speed_rad_s', 'robot_withdrawn', 'measurement_errors')}
        result['destination_relation'] = state['rest_relation'] if target.name == 'occluder' else state['destination_relation']
        return result

    monkeypatch.setattr(partial, '_placement_measurements', measure)

    def create(recipe='viewpoint'):
        contract = {'recipe': recipe, 'source_identity': 'fixture/source',
                    'position_tolerance_m': .02, 'orientation_tolerance_deg': 2.,
                    'dependency_group': ['target', 'occluder'],
                    'source_reference': {'position_m': [0., 0., 1.], 'quaternion_xyzw': [0., 0., 0., 1.]},
                    'source_actors': {key: {'name': key, 'category': 'fixture_category', 'model': 'fixture_model',
                        'position': [1., 0., .7], 'orientation': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}
                        for key in ('target', 'occluder')}}
        config = {'type': 'category_operation', 'module': partial.MODULE, 'operation': partial.GROUNDING_OPERATION,
            'scene': 'scene', 'room': 'room', 'binding_keys': list(actors),
            'bindings': {key: key for key in actors if key != 'occluder_rest' or recipe == 'move_occluder'},
            'parameters': {'grounding_contract': contract, 'expected_identity': 'lamp', 'placement_relation': 'on_top',
                'release_stability_steps': 3, 'maximum_release_speed_m_s': .02,
                'maximum_release_angular_speed_rad_s': .1}}
        evaluator = create_category_operation_evaluator(session, NS(), config, tmp_path / recipe)
        return evaluator

    def move(token='BASE_LEFT', *, feature=True, head_changed=True):
        state['step'] += 1
        state['sequence'] += 1
        state['history'].append({'action': token, 'result_summary': 'completed'})
        state['feature'] = feature
        if head_changed:
            state['head'][2 if token.startswith('TRUNK') else 0] += .10
        return observation()

    return NS(state=state, actors=actors, session=session, receipt=receipt, create=create,
              observation=observation, move=move, output=tmp_path)


@pytest.mark.parametrize('motion', [None, 'MV_UP', 'BASE_LEFT_no_change'])
def test_initial_view_tiny_mask_and_nonviewpoint_motion_never_earn_credit(world, motion):
    evaluator = world.create()
    evaluator.begin_episode(world.observation())
    if motion:
        boundary = world.move('BASE_LEFT' if motion.endswith('no_change') else motion,
                              head_changed=not motion.endswith('no_change'))
        evaluator.record(boundary, action='BASE_LEFT' if motion.endswith('no_change') else motion)
    evaluator.report_answer('lamp', world.observation())
    row = evaluator.record(world.observation())
    assert not row['completion_items'][0]['completed'] and not row['completion_items'][1]['completed']
    assert not row['task_success'] and row['answer_accuracy_percent'] == 0.


@pytest.mark.parametrize('token', ['BASE_LEFT', 'TRUNK_UP'])
def test_new_diagnostic_feature_after_successful_public_viewpoint_is_accepted(world, token):
    evaluator = world.create()
    start = evaluator.begin_episode(world.observation())
    assert start['recognition_difficulty'] == 'initially_ambiguous_audited'
    boundary = world.move(token)
    row = evaluator.record(boundary, action=token)
    assert row['completion_items'][0]['completed'] and row['completion_items'][0]['evidence']['feature_ids']
    evaluator.report_answer('lamp', boundary)
    assert evaluator.record(boundary)['answer_accuracy_percent'] == 100.


def test_missing_exact_feature_receipt_remains_unavailable_after_motion(world):
    evaluator = world.create()
    evaluator.begin_episode(world.observation())
    boundary = world.move()
    world.state['audited'] = False
    row = evaluator.record(boundary, action='BASE_LEFT')
    evaluator.report_answer('lamp', boundary)
    row = evaluator.record(boundary)
    assert row['answer_accuracy_percent'] is None
    assert row['completion_items'][0]['evaluation_validity'] == 'unavailable'


def test_global_target_audit_is_required_and_initial_shortcut_is_labeled_easy(world):
    world.receipt['target_identity_audits']['target'].update(informative=True, feature_ids=['visible_global_shape'])
    evaluator = world.create()
    row = evaluator.begin_episode(world.observation())
    assert row['recognition_difficulty'] == 'easy_initial_identity_available'
    # Frozen Global is initial context, never a post-movement reveal.
    evaluator.report_answer('lamp', world.observation())
    assert not evaluator.record(world.observation())['completion_items'][0]['completed']
    world.receipt['target_identity_audits'] = {}
    evaluator = world.create()
    row = evaluator.begin_episode(world.observation())
    assert row['recognition_difficulty'] == 'unavailable' and row['answer_accuracy_percent'] is None


def identify(world, evaluator):
    evaluator.begin_episode(world.observation())
    boundary = world.move()
    evaluator.record(boundary, action='BASE_LEFT')
    evaluator.report_answer('lamp', boundary)
    return boundary


def grasp(world, evaluator, name='target'):
    world.state['gripper'].update(holding=True, constraint_active=True, object_name=name)
    evaluator.sample_step()


def release(world, evaluator):
    world.state['gripper'].update(holding=False, constraint_active=False, object_name=None)
    world.state['destination_relation'] = True
    for _ in range(3):
        evaluator.sample_step()
    return evaluator.record(world.observation())


def test_same_object_grasp_after_identification_and_current_terminal_revocation(world):
    evaluator = world.create()
    identify(world, evaluator)
    grasp(world, evaluator)
    assert release(world, evaluator)['task_success']
    world.state['destination_relation'] = False
    row = evaluator.record(world.observation())
    assert not row['task_success'] and row['answer_accuracy_percent'] == 100.
    assert row['completion_items'][2]['completed'] and not row['completion_items'][3]['completed']


def test_wrong_confuser_cannot_replace_original_target_selection(world):
    evaluator = world.create()
    identify(world, evaluator)
    grasp(world, evaluator, name='unrelated_confuser')
    row = release(world, evaluator)
    assert not row['completion_items'][2]['completed'] and not row['task_success']


def test_grasp_before_informative_answer_cannot_be_repaired_by_later_guess(world):
    evaluator = world.create()
    evaluator.begin_episode(world.observation())
    evaluator.report_answer('lamp', world.observation())
    grasp(world, evaluator)
    boundary = world.move()
    evaluator.record(boundary, action='BASE_LEFT')
    evaluator.report_answer('lamp', boundary)
    row = release(world, evaluator)
    assert 'target_grasped_before_informative_identification' in row['grounding_faults']
    assert not row['task_success'] and not row['completion_items'][2]['completed']


def test_target_pose_and_occluder_are_protected_during_viewpoint_inspection(world):
    evaluator = world.create()
    evaluator.begin_episode(world.observation())
    world.actors['occluder'].position[0] += .10
    evaluator.sample_step()
    row = evaluator.record(world.observation())
    assert 'protected_occluder_moved' in row['grounding_faults']
    world.actors['target'].position[2] += .10
    evaluator.sample_step()
    assert 'target_moved_before_identification' in evaluator.record(world.observation())['grounding_faults']


def test_permitted_occluder_recipe_requires_real_grasp_stable_rest_and_new_reveal(world):
    evaluator = world.create('move_occluder')
    evaluator.begin_episode(world.observation())
    grasp(world, evaluator, 'occluder')
    world.actors['occluder'].position[0] += .20
    world.state['gripper'].update(holding=False, constraint_active=False, object_name=None)
    world.state['rest_relation'] = True
    for _ in range(3):
        evaluator.sample_step()
    boundary = world.move('RELEASE')
    evaluator.record(boundary, action='RELEASE')
    evaluator.report_answer('lamp', boundary)
    grasp(world, evaluator)
    assert release(world, evaluator)['task_success']
    world.state['rest_relation'] = False
    assert not evaluator.record(boundary)['task_success']


def test_factory_envelope_preserves_boundary_methods_and_semantic_physical_items(world):
    native = world.create()
    task = NS(verify_unchanged=lambda: None, definition=NS(id='task', repair={}, status='pending'),
              task_text='fixture', file_sha256='task_sha', source_sha256='source_sha')
    envelope = CategoryEvaluationEnvelope(native, task, world.output / 'envelope')
    row = envelope.begin_episode(world.observation())
    assert row['snapshot_id'] == 1 and row['action_step'] == 0
    assert len(row['native_result']['completion_items']) == 4
    boundary = world.move()
    envelope.record(boundary, action='BASE_LEFT')
    envelope.report_answer('lamp', boundary)
    assert envelope.last_native['answer_accuracy_percent'] == 100.
    assert all(callable(getattr(native, name)) for name in ('begin_episode', 'record', 'sample_step', 'report_answer'))


def test_common_translation_and_source_reference_are_shared_initial_gates(world):
    evaluator = world.create()
    world.actors['target'].position[0] += .20
    row = evaluator.begin_episode(world.observation())
    assert not row['initial_source_geometry']['accepted']
    assert row['answer_accuracy_percent'] is None
    world.actors['occluder'].position[0] += .20
    world.session.scene_camera_position[0] += .20
    world.session.category_interaction_setup['source_reference_camera_translation_m'] = [.20, 0., 0.]
    evaluator = world.create()
    assert evaluator.begin_episode(world.observation())['initial_source_geometry']['accepted']


def test_new_feature_from_wrong_reset_never_satisfies_inspection(world):
    evaluator = world.create()
    evaluator.begin_episode(world.observation())
    boundary = world.move()
    provider = world.session.task_visual_evidence

    def stale(*args):
        result = provider(*args)
        for row in result['measurements']['target']['views'].values():
            row['audit_context']['reset_id'] = 'different_reset'
        return result

    world.session.task_visual_evidence = stale
    row = evaluator.record(boundary, action='BASE_LEFT')
    assert not row['completion_items'][0]['completed']
    assert row['completion_items'][0]['evaluation_validity'] == 'unavailable'


def source(identity):
    big, small = Path(identity).parts[:2]
    return SourceInstance(ROOT / 'ESI/dataset/json_clean' / identity, identity, big, small)


def destination():
    return {'binding': {'name': 'delivery_fixture', 'category': 'desk'},
            'public_label': 'the destination desk', 'relation': 'on_top'}


def test_occluder_manipulation_proposal_requires_explicit_distinct_resting_support(tmp_path):
    selected = source(partial.FEEDBACK_SOURCE)
    config = {'task_edit2_grounding': True, 'placement_destination': destination(), 'grounding_recipe': 'move_occluder'}
    with pytest.raises(ValueError, match='resting-support contract'):
        partial.PartialOcclusionConverter().convert(selected, ConversionContext(tmp_path, config=config))
    config['occluder_resting_support'] = {'binding': {'name': 'rest_fixture', 'category': 'desk'},
        'public_label': 'the separate blocker resting desk', 'relation': 'on_top'}
    result = partial.PartialOcclusionConverter().convert(selected, ConversionContext(tmp_path, config=config))
    assert result['grounding_contract']['occluder_motion_permitted']
    assert not result['grounding_contract']['resting_support_contract']['swept_space_verified']
    assert result['task_definition']['private']['bindings']['occluder_rest']['name'] == 'rest_fixture'


def test_actual_current_core_eight_derivatives_validate_and_never_move_building_support(tmp_path):
    manifest = build_candidates(tmp_path)
    assert manifest['core_grounding_source_count'] == 8 and manifest['counts'] == {'generated': 8}
    for record in manifest['records']:
        task = yaml.safe_load(Path(record['task_file']).read_text())
        scene = yaml.safe_load(Path(record['scene_file']).read_text())
        task_definition(task)
        SceneDefinition.model_validate(scene)
        p = task['private']['evaluation']['parameters']
        assert task['private']['evaluation']['operation'] == partial.GROUNDING_OPERATION
        assert p['grounding_revision'] == partial.GROUNDING_REVISION
        assert not any(patch.get('fixed_base') is False for patch in task['scene']['variants']['adapted']['patches'])
        policy = record['interaction_policy']
        assert 'support' not in policy['group_bindings']
        assert policy['source_camera_policy'] == 'translate_with_group'
        if 'View Hallucination' in record['source_identity'] and task['private']['bindings']['support'].get('category') == 'table_lamp':
            assert policy['relocation_policy'] == 'preserve'
        assert 'expected_identity' not in json.dumps(task['public'])


def test_old_conversion_remains_opt_in_and_shared_readers_remain_exported(tmp_path):
    selected = source(partial.FEEDBACK_SOURCE)
    result = partial.PartialOcclusionConverter().convert(selected, ConversionContext(tmp_path,
        config={'placement_destination': destination()}))
    assert result['task_definition']['private']['evaluation']['operation'] == partial.IDENTIFY_PLACE
    assert 'grounding_contract' not in result['task_definition']['private']['evaluation']['parameters']
    assert callable(partial.native_visual_evidence) and callable(partial._placement_measurements)

"""Behavior CPU checks for whole-apparatus inclined-plane contracts.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_inclined.py.
Native-shaped contacts exercise converter/evaluator behavior, not robot feasibility.
"""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import physical_dynamics_inclined_plane as inclined
from wia.tasks.inclined_apparatus import (source_apparatus_proposal, materialize_apparatus,
    required_support_dimensions, support_envelope_fit, common_translation, REVISION)
from wia.tasks.interaction_setup import prepare_definitions
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from harness_integration.test_annotation_manipulation_categories import NativeObject, native_types, put_on, EMPTY_GRIPPER

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
CORE = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')


def source_payload():
    return {'slope': {'aabb_min': [-.2, -.15, 0.], 'aabb_max': [.2, .15, .1]},
        'objects': {'task_obj': {'pos_on_floor': [0., .35, .05], 'pos_init': [0., 0., .15], 'pos_final': [.25, 0., .05],
            'aabb_min_init': [-.05, -.05, .1], 'aabb_max_init': [.05, .05, .2],
            'aabb_min_final': [.2, -.05, 0.], 'aabb_max_final': [.3, .05, .1]}}}


def test_proposal_requires_measured_native_pickup_and_full_diameter():
    proposal = source_apparatus_proposal(source_payload())
    assert proposal['required_bounds_m'] is None and proposal['pickup_bounds_m'] is None
    assert not required_support_dimensions(proposal)['available']
    measured = materialize_apparatus(proposal, pickup_bounds=([-.05, .3, 0.], [.05, .4, .1]))
    diameter = np.sqrt(.02)
    assert measured['horizontal_diameter_margin_m'] == pytest.approx(diameter)
    assert measured['required_bounds_m'][0][0] <= -.05 - diameter
    assert measured['required_bounds_m'][1][0] >= .3 + diameter
    assert measured['required_bounds_m'][1][1] >= .4
    assert not measured['one_second_runoff_native_validated']
    assert proposal['required_bounds_m'] is None
    with pytest.raises(ValueError, match='shrink'):
        materialize_apparatus(proposal, pickup_bounds=([-.05, .3, 0.], [.05, .4, .1]), target_horizontal_diameter_m=.01)


def test_full_runoff_support_fit_and_size_limit_are_not_clamped():
    proposal = source_apparatus_proposal(source_payload())
    envelope = materialize_apparatus(proposal, pickup_bounds=([-.05, .3, 0.], [.05, .4, .1]))
    assert support_envelope_fit(envelope, ([-.8, -.6, .65], [.8, .6, .7]), [0, 0, .7])['accepted']
    assert not support_envelope_fit(envelope, ([-.2, -.15, .65], [.2, .15, .7]), [0, 0, .7])['accepted']
    assert not support_envelope_fit(envelope, ([-.8, -.6, .65], [.8, .6, .7]), [0, 0, .5])['accepted']
    wide = copy.deepcopy(source_payload())
    wide['objects']['task_obj']['pos_final'][0] = 3.
    wide['objects']['task_obj']['aabb_min_final'][0] = 2.95
    wide['objects']['task_obj']['aabb_max_final'][0] = 3.05
    huge = materialize_apparatus(source_apparatus_proposal(wide), pickup_bounds=([-.05, .3, 0.], [.05, .4, .1]))
    assert required_support_dimensions(huge)['dimensions_xy_m'][0] > 3.
    assert not support_envelope_fit(huge, ([-10, -10, .65], [10, 10, .7]), [0, 0, .7])['accepted']


def test_common_transform_requires_target_ramp_and_source_camera():
    setup = {'accepted': True, 'relocated': True, 'support_prim_path': '/table', 'group_translation_m': [1, 2, .7],
        'object_translations_m': {'target': [1, 2, .7], 'slope': [1, 2, .7]},
        'source_reference_camera_translation_m': [1, 2, .7]}
    assert np.allclose(common_translation(setup, target_name='target', slope_name='slope'), [1, 2, .7])
    setup['object_translations_m']['slope'][0] = 0.
    with pytest.raises(ValueError, match='common'):
        common_translation(setup, target_name='target', slope_name='slope')
    setup['object_translations_m']['slope'][0] = 1.
    setup['source_reference_camera_translation_m'] = [0, 0, 0]
    with pytest.raises(ValueError, match='camera'):
        common_translation(setup, target_name='target', slope_name='slope')


@pytest.fixture
def world(native_types, monkeypatch, tmp_path):
    floor = NativeObject('floor', [0, 0, -.05], (10., 10., .1), category='floors')
    deck = NativeObject('table', [0, 0, .35], (1.6, 1.2, .7), category='table')
    deck.fixed_base = True
    slope = NativeObject('slope', [0, 0, .05], (.4, .3, .1))
    slope.scale = np.array([.4, .3, .1]); slope.fixed_base = True
    target = NativeObject('task_obj', [0, .35, .05])
    by_name = {obj.name: obj for obj in (floor, deck, slope, target)}
    session = SimpleNamespace(scene_name='source_scene', room_name='room_0',
        scene_camera_position=[1., 0., .4], scene_camera_quaternion=[0., 0., 0., 1.],
        env=SimpleNamespace(scene=SimpleNamespace(objects=by_name, object_registry=lambda field, name: by_name.get(name))))
    parameters = {'task_revision': REVISION, 'slope_position': slope.position.tolist(),
        'slope_quaternion': slope.quaternion.tolist(), 'slope_scale': slope.scale.tolist(),
        'floor_z_m': 0., 'target_floor_position': target.position.tolist(), 'static_friction': 1., 'dynamic_friction': .5,
        'source_identity': 'test_source', 'source_sha256': 'test_hash', 'target_identity': {'category': target.category, 'model': target.model},
        'apparatus_proposal': source_apparatus_proposal(source_payload())}
    monkeypatch.setattr(inclined, '_friction', lambda obj: [{'mesh': obj.name, 'static_friction': 1., 'dynamic_friction': .5}])
    monkeypatch.setattr(inclined, '_robot_state', lambda session: {'in_original_room': True, 'floor_contacts': [['floor']], 'obstacle_contacts': []})
    receipt = inclined.measure_source_apparatus(session, target, slope, parameters)
    delta = np.array([0, 0, .7])
    target.position += delta; slope.position += delta
    session.scene_camera_position = (np.asarray(session.scene_camera_position) + delta).tolist()
    put_on(target, deck, native_types)
    put_on(slope, deck, native_types)
    session.category_interaction_setup = {'accepted': True, 'relocated': True, 'object_names': [target.name, slope.name],
        'support_name': deck.name, 'support_prim_path': deck.prim_path, 'support_surface_z_m': .7,
        'group_translation_m': delta.tolist(), 'object_translations_m': {obj.name: delta.tolist() for obj in (target, slope)},
        'source_reference_camera_translation_m': delta.tolist(), 'inclined_apparatus': receipt}
    receipt['support_reference'] = inclined._pose(deck)
    # Fixed-fixed PhysX omits contacts; shared helper performs native surface rays.
    # This fixture only supplies that fixed-ramp measurement, leaving target contact logic real.
    original_touch, original_top = inclined.touching_task_support, inclined.on_top_task_support
    monkeypatch.setattr(inclined, 'touching_task_support', lambda sess, obj, floor: True if obj is slope else original_touch(sess, obj, floor))
    monkeypatch.setattr(inclined, 'on_top_task_support', lambda sess, obj, floor: True if obj is slope else original_top(sess, obj, floor))
    controller = SimpleNamespace(debug_gripper_state=lambda: dict(EMPTY_GRIPPER),
        geometric_rearrangement_evidence=lambda: {'available': True, 'state_setter_used': False, 'sample_id': 0, 'control_sequence': 0})
    config = {'type': 'category_operation', 'module': inclined.MODULE, 'operation': inclined.APPARATUS_OPERATION,
        'scene': session.scene_name, 'room': session.room_name, 'bindings': {'target': target.name, 'slope': slope.name, 'floor': floor.name},
        'parameters': parameters}
    return SimpleNamespace(session=session, controller=controller, config=config, parameters=parameters,
        bound={'target': target, 'slope': slope, 'floor': floor}, target=target, slope=slope, floor=floor,
        deck=deck, states=native_types, output=tmp_path)


def test_actual_qualifier_reads_exact_accepted_apparatus(world):
    result = inclined.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert result['accepted'], result
    assert result['native_floor_top_m'] == 0. and result['apparatus_trial_surface_z_m'] == .7
    assert not result['real_robot_placement_qualified']
    assert not result['one_second_runoff_native_validated']
    assert result['initial_target_reference_m'] == pytest.approx([0, .35, .75])


@pytest.mark.parametrize('failure', ['missing_receipt', 'target_only', 'camera', 'stale_pose', 'ramp_angle', 'scale', 'mass', 'material', 'small_deck'])
def test_qualifier_rejects_stale_and_changed_apparatus(world, failure, monkeypatch):
    setup = world.session.category_interaction_setup
    if failure == 'missing_receipt':
        setup.pop('inclined_apparatus')
    elif failure == 'target_only':
        setup['object_translations_m']['slope'] = [0, 0, 0]
    elif failure == 'camera':
        setup['source_reference_camera_translation_m'] = [0, 0, 0]
    elif failure == 'stale_pose':
        world.target.position[0] += .05
    elif failure == 'ramp_angle':
        world.slope.quaternion = np.array([0, .01, 0, .99995])
    elif failure == 'scale':
        world.target.scale *= .5
    elif failure == 'mass':
        world.target.mass = 2.
    elif failure == 'material':
        monkeypatch.setattr(inclined, '_friction', lambda obj: [{'mesh': obj.name, 'static_friction': 2., 'dynamic_friction': .5}])
    else:
        world.deck.extent[:2] = [.4, .3]
    assert not inclined.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']


def trial_row(world):
    apparatus = inclined._resolve_apparatus(world.session, world.parameters, world.target, world.slope)
    return inclined._trial_state(world.target, world.slope, world.floor,
        {**world.parameters, 'apparatus_translation_m': [0, 0, .7]}, apparatus=apparatus)[0]


def test_real_deck_contact_is_runoff_and_original_floor_is_escape(world):
    row = trial_row(world)
    assert row['touching_apparatus_deck'] and not row['escaped_apparatus']
    assert row['interfering_contacts'] == []
    world.target.position = np.array([.25, 0, .05])
    put_on(world.target, world.floor, world.states, z=0.)
    row = trial_row(world)
    assert row['touching_original_floor'] and row['escaped_apparatus']
    assert row['interfering_contacts']


def test_same_table_contact_stays_interference_in_legacy(world):
    parameters = {**world.parameters, 'task_revision': inclined.TASK_REVISION}
    row, _ = inclined._trial_state(world.target, world.slope, world.floor, parameters)
    assert row['interfering_contacts'] and not row['slope_unchanged']
    assert 'escaped_apparatus' not in row


def test_actual_evaluator_factory_trace_and_broken_terminal(world, monkeypatch):
    time = {'now': 0.}
    monkeypatch.setattr(inclined, '_native_time', lambda: {'time_s': time['now'], 'physics_dt_s': 1/120, 'sim_step_dt_s': 1/30})
    evaluator = inclined.create_evaluator(world.session, world.controller, world.config, world.output)
    observation = SimpleNamespace(snapshot_id='native_fixture', action_step=0)
    evaluator.begin_episode(observation)
    assert not evaluator.last['task_success']
    world.target.position[2] += .1
    world.controller.debug_gripper_state = lambda: {'holding': True, 'object_name': world.target.name, 'constraint_active': True}
    time['now'] += .02; evaluator.sample_step()
    world.target.position = np.array([0., 0., .85])
    put_on(world.target, world.slope, world.states, z=.8)
    time['now'] += .02; evaluator.sample_step()
    world.controller.debug_gripper_state = lambda: dict(EMPTY_GRIPPER)
    time['now'] += .02; evaluator.sample_step()
    assert evaluator.release_seen
    for _ in range(51):
        time['now'] += .02; evaluator.sample_step()
    assert evaluator.last['task_success'], evaluator.last
    world.target.position = np.array([.25, 0, .75])
    put_on(world.target, world.deck, world.states, z=.7)
    time['now'] += .02
    value = evaluator.record(observation, action=None)
    assert value['task_success'] and value['measured_outcome'] == 'intended_runoff_on_selected_deck'
    assert value['schema'] == 'private_whole_inclined_apparatus_trial_v1'
    world.target.position[2] = .05
    put_on(world.target, world.floor, world.states, z=0.)
    time['now'] += .02
    assert not evaluator._update()['task_success']
    assert evaluator.last['measured_outcome'] == 'escaped_apparatus_to_original_floor'


def test_all_actual_core_conversion_preserves_source_and_versioned_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(inclined, '_robot_start', lambda *args: None)
    members = [row for row in json.loads(CORE.read_text())['tasks'] if row.get('subcategory') == 'Inclined Plane']
    assert len(members) == 4
    for member in members:
        identity = member['source']; path = ROOT / 'ESI/dataset/json_clean' / identity
        source = SourceInstance(path, identity, 'Physical Dynamics', 'Inclined Plane')
        before = path.read_bytes()
        result = inclined.CategoryConverter().convert(source, ConversionContext(tmp_path, config={'task_edit2_inclined': True}))
        task, scene = result['task_definition'], result['scene_definition']
        prepare_definitions(source, scene, task, result['interaction_policy'])
        task_definition(task); SceneDefinition.model_validate(scene)
        parameters = task['private']['evaluation']['parameters']
        assert task['private']['evaluation']['operation'] == inclined.APPARATUS_OPERATION
        assert parameters['slope_position'] == source.payload['slope']['position']
        assert parameters['slope_quaternion'] == source.payload['slope_quaternion']
        assert parameters['slope_scale'] == source.payload['slope_scale']
        assert parameters['static_friction'] == source.payload['static_friction']
        assert parameters['dynamic_friction'] == source.payload['dynamic_friction']
        assert scene['scene']['variants']['original']['patches'] == []
        assert parameters['apparatus_proposal']['pickup_bounds_m'] is None
        assert result['interaction_policy']['group_bindings'] == ['target', 'slope']
        assert result['interaction_policy']['source_camera_policy'] == 'translate_with_group'
        assert path.read_bytes() == before
        legacy = inclined.CategoryConverter().convert(source, ConversionContext(tmp_path))
        assert legacy['task_definition']['private']['evaluation']['operation'] == inclined.OPERATION
        assert legacy['task_revision'] == inclined.TASK_REVISION


def test_missing_native_control_history_is_unavailable(world, monkeypatch):
    del world.controller.geometric_rearrangement_evidence
    result = inclined.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert not result['accepted'] and not result['checks']['native_public_control_history_available']
    monkeypatch.setattr(inclined, '_native_time', lambda: {'time_s': 0., 'physics_dt_s': 1/120, 'sim_step_dt_s': 1/30})
    evaluator = inclined.create_evaluator(world.session, world.controller, world.config, world.output)
    result = evaluator.begin_episode(SimpleNamespace(snapshot_id='start', action_step=0))
    result = evaluator.record(SimpleNamespace(snapshot_id='start', action_step=0), action=None)
    assert result['evaluation_validity'] == 'unavailable' and not result['task_success']


def test_setter_violation_latches_after_native_evidence_recovers(world, monkeypatch):
    evidence = {'available': True, 'state_setter_used': False}
    world.controller.geometric_rearrangement_evidence = lambda: dict(evidence)
    monkeypatch.setattr(inclined, '_native_time', lambda: {'time_s': 0., 'physics_dt_s': 1/120, 'sim_step_dt_s': 1/30})
    evaluator = inclined.create_evaluator(world.session, world.controller, world.config, world.output)
    evaluator.begin_episode(SimpleNamespace(snapshot_id='start', action_step=0))
    evidence['state_setter_used'] = True
    evaluator.sample_step()
    evidence['state_setter_used'] = False
    assert not evaluator._update()['criteria']['no_episode_simulator_setters']
    assert not evaluator.last['task_success']


def test_native_table_side_contact_is_interference_not_runoff(world):
    world.target.contacts[0].normal = np.array([1., 0., 0.])
    row = trial_row(world)
    assert not row['touching_apparatus_deck'] and row['interfering_contacts']


def test_full_selected_deck_is_safe_beyond_minimum_envelope(world):
    world.target.position = np.array([.65, 0., .75])
    put_on(world.target, world.deck, world.states, z=.7)
    row = trial_row(world)
    assert row['within_selected_deck'] and not row['within_required_envelope']
    assert not row['escaped_apparatus']


def test_actual_category_envelope_retains_new_trial_validity(world, monkeypatch):
    from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope
    monkeypatch.setattr(inclined, '_native_time', lambda: {'time_s': 0., 'physics_dt_s': 1/120, 'sim_step_dt_s': 1/30})
    monkeypatch.setattr('wia.tasks.background_policy.record_background_movements', lambda session: [])
    evaluator = inclined.create_evaluator(world.session, world.controller, world.config, world.output / 'native')
    task = SimpleNamespace(verify_unchanged=lambda: None, task_text='Perform the physical trial.',
        definition=SimpleNamespace(id='inclined', repair={}, status='pending',
            private=SimpleNamespace(evaluation=SimpleNamespace(model_dump=lambda: world.config))),
        file_sha256='task', source_sha256='source')
    envelope = CategoryEvaluationEnvelope(evaluator, task, world.output / 'envelope')
    observation = SimpleNamespace(snapshot_id='start', action_step=0)
    envelope.begin_episode(observation)
    value = envelope.record(observation, action=None)
    assert envelope.failure is None
    assert value['native_schema'] == 'private_whole_inclined_apparatus_trial_v1'
    assert not value['task_success']


def test_actual_source_reference_camera_change_invalidates_receipt(world):
    world.session.scene_camera_position[0] += .1
    result = inclined.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert not result['accepted']
    assert 'camera' in result['exception_summary']


def test_stale_source_receipt_is_not_reused(world):
    world.session.category_interaction_setup['inclined_apparatus']['source_sha256'] = 'other_source'
    result = inclined.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert not result['accepted'] and 'source-bound' in result['exception_summary']

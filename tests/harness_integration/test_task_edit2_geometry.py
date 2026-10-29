"""Check TaskEdit2 seeded geometry and causal evaluation with native-shaped fixtures.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
tests/test_task_edit2_geometry.py tests/test_core_revision_manipulation.py
These CPU checks do not certify native support, grasp access or executed actions.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext
from wia.authoring.converters import spatial_relations_geometric_configuration as geometry
from harness_integration.test_annotation_manipulation_categories import source_for
from harness_integration.test_spatial_relations_tabletop import world_factory


def layout(shape='equilateral', seed=11, offsets=None):
    return geometry.seeded_initial_layout(object_keys=list(geometry.VERTICES),
        aabb_half_extents_xy=[[.1, .1]] * 3,
        root_to_aabb_center_xy=offsets or [[0., 0.]] * 3,
        root_z_m=[.1] * 3, support_aabb_xy=[[-2., -2.], [2., 2.]],
        target_shape=shape, seed=seed, source_identity='scene/room/source.json',
        minimum_clearance_m=.1)


@pytest.mark.parametrize('shape', ['equilateral', 'isosceles', 'random triangle'])
def test_seeded_layout_is_deterministic_supported_nonoverlapping_and_initially_wrong(shape):
    first = layout(shape)
    assert first == layout(shape)
    assert first['initial_positions'] != layout(shape, seed=12)['initial_positions']
    assert geometry.initial_shape_is_wrong(first['initial_aabb_centers_xy'], shape)
    assert not geometry.shape_geometry(first['initial_aabb_centers_xy'], shape)['satisfied']
    assert min(first['horizontal_clearances_m']) >= .1
    centers = np.asarray(first['initial_aabb_centers_xy'])
    assert np.all(centers - .1 >= -2.) and np.all(centers + .1 <= 2.)


def test_root_offsets_preserve_the_aabb_center_reference_convention():
    first = layout()
    offsets = [[.07, -.02], [-.03, .05], [.01, .04]]
    shifted = layout(offsets=offsets)
    assert shifted['initial_aabb_centers_xy'] == first['initial_aabb_centers_xy']
    np.testing.assert_allclose(np.asarray(shifted['initial_positions'])[:, :2] + offsets,
        first['initial_aabb_centers_xy'])


def test_inadequate_support_is_rejected():
    inputs = layout()['inputs']
    inputs['support_aabb_xy'] = [[0., 0.], [.1, .1]]
    with pytest.raises(ValueError, match='Insufficient support'):
        geometry.seeded_initial_layout(**inputs)


def test_new_generation_declares_seeded_rearrangement_and_retains_explicit_legacy(tmp_path):
    converter, source = source_for(geometry)
    result = converter.convert(source, ConversionContext(tmp_path, seed=17))
    task = result['task_definition']
    parameters = task['private']['evaluation']['parameters']
    assert parameters['task_revision'] == geometry.TASK_EDIT2_REVISION
    assert parameters['initial_layout']['seed'] == 17
    assert parameters['target_shape'] == source.raw['answer']
    assert parameters['source_positions'] == [obj['position'] for obj in source.payload['objects']]
    assert 'already correct' not in task['public']['objective']
    old = converter.convert(source, ConversionContext(tmp_path,
        config={'task_revision': geometry.TASK_REVISION}))
    assert 'already correct' in old['task_definition']['public']['objective']
    assert 'initial_layout' not in old['task_definition']['private']['evaluation']['parameters']


@pytest.fixture
def revised_world(world_factory):
    world = world_factory(geometry, adapted=False)
    receipt = layout()
    for prop, position in zip(world.props, receipt['initial_positions']):
        prop.position[:] = position
        prop.scale = np.ones(3)
    world.config['operation'] = geometry.ARRANGEMENT_OPERATION
    world.config['parameters'].update(task_revision=geometry.TASK_EDIT2_REVISION,
        initial_layout={'version': geometry.LAYOUT_VERSION, 'seed': 11,
            'source_identity': 'scene/room/source.json'},
        object_keys=list(geometry.VERTICES), source_scales=[1., 1., 1.],
        target_shape='equilateral', release_stability_steps=15, minimum_rearrangement_m=.035)
    world.session.category_interaction_setup = {'accepted': True,
        'geometric_initial_layout': {**receipt, 'accepted': True}}
    world.evidence = {'available': True, 'sample_id': 0, 'control_sequence': 0,
        'control_active': False, 'controlled_object_names': [], 'grasped_object_names': [],
        'contact_object_names': [], 'state_setter_used': False}
    world.controller.geometric_rearrangement_evidence = lambda: copy.deepcopy(world.evidence)
    world.observation = SimpleNamespace(snapshot_id='s0', action_step=0)
    return world


def test_qualifier_accepts_revised_pose_and_rejects_stale_or_mismatched_pose(revised_world):
    world = revised_world
    result = geometry.qualify_arrangement(world.session, world.controller, world.config, world.bound)
    assert result['accepted'], result
    assert result['checks']['initial_target_false_with_margin']
    for prop, source in zip(world.props, world.source):
        prop.position[:] = source
    assert not geometry.qualify_arrangement(world.session, world.controller, world.config, world.bound)['accepted']


@pytest.mark.parametrize('field,value', [('seed', 999), ('source_identity', 'stale.json'), ('initial_positions', [[0., 0., .1]] * 3)])
def test_qualifier_rejects_mismatched_recipe(revised_world, field, value):
    world = revised_world
    world.session.category_interaction_setup['geometric_initial_layout'][field] = value
    assert not geometry.qualify_arrangement(world.session, world.controller, world.config, world.bound)['accepted']


def advance(world, evaluator, steps=1):
    for _ in range(steps):
        world.evidence['sample_id'] += 1
        evaluator.sample_step()


def target(world):
    for prop, xy in zip(world.props, [[-.5, -.3], [.5, -.3], [0., -.3 + np.sqrt(3.) / 2]]):
        prop.position[:2] = xy


def test_untouched_initial_world_cannot_succeed(revised_world, tmp_path):
    world = revised_world
    evaluator = geometry.ShapeArrangementEvaluator(world.session, world.controller, world.config, tmp_path)
    evaluator.begin_episode(world.observation)
    advance(world, evaluator, 20)
    result = evaluator.record(world.observation)
    assert result['evaluation_validity'] == 'complete'
    assert not result['task_success']
    assert not result['completion_items'][1]['completed']


def test_actual_robot_motion_then_stable_released_target_succeeds_and_broken_terminal_fails(revised_world, tmp_path):
    world = revised_world
    evaluator = geometry.ShapeArrangementEvaluator(world.session, world.controller, world.config, tmp_path)
    evaluator.begin_episode(world.observation)
    world.evidence.update(control_sequence=1, control_active=True,
        controlled_object_names=[prop.name for prop in world.props],
        grasped_object_names=[world.props[0].name],
        contact_object_names=[prop.name for prop in world.props[1:]])
    target(world)
    advance(world, evaluator)
    world.evidence.update(control_active=False, grasped_object_names=[], contact_object_names=[])
    advance(world, evaluator, 15)
    result = evaluator.record(world.observation)
    assert result['task_success'], result
    assert result['completion_score'] == 1.
    world.props[2].position[:2] += [.4, .3]
    result = evaluator.record(world.observation)
    assert not result['task_success']
    assert result['completion_items'][0]['completed']
    assert result['completion_items'][1]['completed']
    assert not result['completion_items'][2]['completed']


@pytest.mark.parametrize('claimed_control,state_setter', [(False, False), (True, True)])
def test_placement_or_simulator_teleport_is_not_robot_movement(revised_world, tmp_path, claimed_control, state_setter):
    world = revised_world
    evaluator = geometry.ShapeArrangementEvaluator(world.session, world.controller, world.config, tmp_path)
    evaluator.begin_episode(world.observation)
    target(world)
    world.evidence.update(control_sequence=1, control_active=claimed_control,
        controlled_object_names=[prop.name for prop in world.props],
        grasped_object_names=[prop.name for prop in world.props], state_setter_used=state_setter)
    advance(world, evaluator)
    world.evidence['control_active'] = False
    advance(world, evaluator, 16)
    assert not evaluator.record(world.observation)['task_success']
    # A subsequent grasp with no displacement cannot credit the preceding placement.
    world.evidence.update(control_active=True, control_sequence=2)
    advance(world, evaluator, 16)
    assert not evaluator.record(world.observation)['completion_items'][1]['completed']


def test_missing_native_causal_evidence_stays_unavailable(revised_world, tmp_path):
    world = revised_world
    del world.controller.geometric_rearrangement_evidence
    evaluator = geometry.ShapeArrangementEvaluator(world.session, world.controller, world.config, tmp_path)
    result = evaluator.begin_episode(world.observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['completion_score'] is None
    assert not result['task_success']


def test_repeated_native_sample_ids_cannot_fill_release_dwell(revised_world, tmp_path):
    world = revised_world
    evaluator = geometry.ShapeArrangementEvaluator(world.session, world.controller, world.config, tmp_path)
    evaluator.begin_episode(world.observation)
    world.evidence.update(control_sequence=1, control_active=True,
        controlled_object_names=[world.props[0].name], grasped_object_names=[world.props[0].name])
    target(world)
    advance(world, evaluator)
    for _ in range(20):
        evaluator.sample_step()
    assert evaluator.stable_steps == 1
    assert not evaluator.record(world.observation)['task_success']

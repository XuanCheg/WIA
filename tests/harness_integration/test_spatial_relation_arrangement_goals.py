"""Check a goal-only Linear Alignment change with CPU-shaped native interfaces.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_spatial_relation_arrangement_goals.py
These fixtures measure goal logic without running physics or certifying grasps.
"""
import copy
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import spatial_relations_linear_alignment as alignment
from wia.tasks.loader import task_definition
from harness_integration.test_spatial_relations_tabletop import world_factory


MODULES = [alignment]


def target_xy(module):
    return np.array([[.6, 1.8], [1., 1.8], [1.4, 1.8]])


def wrong_xy():
    return np.array([[.7, 1.7], [1.2, 1.7], [.95, 1.7 + math.sqrt(3) * .25]])


def place(world, xy):
    for obj, position in zip(world.props, xy):
        obj.position[:2] = position
        obj.contact.position[:2] = position.tolist()


def initialize(world_factory, module, *, correct=True):
    world = world_factory(module)
    world.config['operation'] = module.ARRANGEMENT_OPERATION
    xy = target_xy(module) if correct else wrong_xy()
    place(world, xy)
    delta = np.array([1., 2., .6])
    world.config['parameters']['source_positions'] = [(obj.position - delta).tolist() for obj in world.props]
    world.session.category_interaction_setup['object_translations_m'] = {
        obj.name: delta.tolist() for obj in world.props}
    world.actor_calls = []

    def forbidden(*args, **kwargs):
        world.actor_calls.append((args, kwargs))
        raise AssertionError('A read-only evaluator attempted an actor or simulator call.')

    world.session.env.step = forbidden
    world.session.pipeline.og.sim.step = forbidden
    world.session.robot.set_position_orientation = forbidden
    world.controller.execute = forbidden
    world.controller.grasp = forbidden
    for obj in world.props:
        obj.set_position_orientation = forbidden
        obj.set_linear_velocity = forbidden
    return world


@pytest.mark.parametrize('module', MODULES)
def test_already_satisfied_initial_state_succeeds_without_motion(world_factory, module, tmp_path):
    world = initialize(world_factory, module)
    initial_positions = [obj.position.copy() for obj in world.props]
    evaluator = module.create_evaluator(world.session, world.controller, world.config, tmp_path)
    observation = SimpleNamespace(snapshot_id='initial', action_step=0)
    initial = evaluator.begin_episode(observation)
    assert initial['task_success'], initial
    assert initial['action'] is None
    assert initial['stability_evidence'] == 'current_boundary_contact_and_velocity_only'
    assert evaluator.record(observation, action=None)['task_success']
    assert evaluator.sample_step()['task_success']
    assert world.actor_calls == []
    assert world.session.pipeline.og.sim.current_time == 0.
    for obj, position in zip(world.props, initial_positions):
        np.testing.assert_array_equal(obj.position, position)


@pytest.mark.parametrize('module', MODULES)
def test_wrong_initial_geometry_can_pass_after_measured_placement(world_factory, module, tmp_path):
    world = initialize(world_factory, module, correct=False)
    evaluator = module.create_evaluator(world.session, world.controller, world.config, tmp_path)
    observation = SimpleNamespace(snapshot_id='initial', action_step=0)
    assert not evaluator.begin_episode(observation)['task_success']
    place(world, target_xy(module)[::-1])
    obj = world.props[0]
    obj.get_position_orientation = lambda: (obj.position.copy(), np.array([0., 0., 1., 0.]))
    result = evaluator.record(SimpleNamespace(snapshot_id='placed', action_step=1), action='measured placement')
    assert result['task_success'], result
    assert world.actor_calls == []
    assert world.session.pipeline.og.sim.current_time == 0.


@pytest.mark.parametrize('module', MODULES)
def test_private_source_annotation_cannot_change_native_goal(world_factory, module, tmp_path):
    world = initialize(world_factory, module)
    evaluator = module.create_evaluator(world.session, world.controller, world.config, tmp_path)
    observation = SimpleNamespace(snapshot_id='initial', action_step=0)
    assert evaluator.begin_episode(observation)['task_success']
    for answer in ('yes', 'no'):
        world.config['parameters']['source_answer_private'] = answer
        assert evaluator.record(observation, action=None)['task_success']
    assert world.actor_calls == []


@pytest.mark.parametrize('module', MODULES)
@pytest.mark.parametrize('failure', ['wrong_geometry', 'floating', 'not_on_top', 'moving', 'rotating',
    'held_name', 'holding', 'constraint', 'robot_tilt', 'robot_contact', 'visual_only',
    'coincident_centres', 'invalid_bounds'])
def test_current_state_revokes_success_when_terminal_conditions_fail(world_factory, module, failure, tmp_path):
    world = initialize(world_factory, module)
    evaluator = module.create_evaluator(world.session, world.controller, world.config, tmp_path)
    observation = SimpleNamespace(snapshot_id='initial', action_step=0)
    assert evaluator.begin_episode(observation)['task_success']
    obj = world.props[0]
    if failure == 'wrong_geometry':
        place(world, wrong_xy())
    elif failure == 'floating':
        obj.touching = False
        obj.position[2] += .1
    elif failure == 'not_on_top':
        obj.on_top = False
    elif failure == 'moving':
        obj.get_linear_velocity = lambda: np.array([.031, 0., 0.])
    elif failure == 'rotating':
        obj.get_angular_velocity = lambda: np.array([0., 0., .101])
    elif failure in ('held_name', 'holding', 'constraint'):
        state = world.controller.debug_gripper_state()
        state[{'held_name': 'object_name', 'holding': 'holding', 'constraint': 'constraint_active'}[failure]] = (
            obj.name if failure == 'held_name' else True)
        world.controller.debug_gripper_state = lambda: state
    elif failure == 'robot_tilt':
        pose = world.controller.pose()
        pose['quaternion_xyzw'] = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
        world.controller.pose = lambda: pose
    elif failure == 'robot_contact':
        world.robot_contacts[0].body1 = world.table.prim_path + '/side'
    elif failure == 'visual_only':
        obj.visual_only = True
    elif failure == 'coincident_centres':
        obj.position[:2] = world.props[1].position[:2]
    elif failure == 'invalid_bounds':
        obj.position[0] = float('nan')
    result = evaluator.record(SimpleNamespace(snapshot_id='invalid', action_step=1), action=None)
    assert not result['task_success'], result
    assert world.actor_calls == []


def test_line_tolerance_uses_longest_baseline_and_both_caps():
    for span in (.4, 2.):
        tolerance = min(.02, .02 * span)
        good = [[0., 0.], [span / 2, tolerance], [span, 0.]]
        bad = [[0., 0.], [span / 2, tolerance + 1e-6], [span, 0.]]
        assert alignment._relation_geometry(good)['target_relation_satisfied']
        assert not alignment._relation_geometry(bad)['target_relation_satisfied']


def test_existing_fixed_flags_do_not_force_physics_edits(world_factory, tmp_path, monkeypatch):
    world = initialize(world_factory, alignment)
    sampling = ModuleType('omnigibson.utils.sampling_utils')

    def raytest(start, end, *, only_closest):
        low, high = world.table.aabb
        if all(low[axis] <= start[axis] <= high[axis] for axis in (0, 1)) and end[2] <= high[2] <= start[2]:
            return [{'hit': True, 'rigidBody': world.table.prim_path + '/top',
                     'normal': [0., 0., 1.], 'position': [*start[:2], float(high[2])]}]
        return []

    sampling.raytest = raytest
    monkeypatch.setitem(sys.modules, 'omnigibson.utils', ModuleType('omnigibson.utils'))
    monkeypatch.setitem(sys.modules, sampling.__name__, sampling)
    for obj in world.props:
        obj.fixed_base = True
    evaluator = alignment.create_evaluator(world.session, world.controller, world.config, tmp_path)
    assert evaluator.begin_episode(SimpleNamespace(snapshot_id='initial', action_step=0))['task_success']
    assert all(obj.fixed_base for obj in world.props)
    assert world.actor_calls == []
    world.props[0].position[2] += .1
    assert not evaluator.record(SimpleNamespace(snapshot_id='floating', action_step=1), action=None)['task_success']
    assert world.actor_calls == []


def source_fixture(tmp_path, module, answer):
    categories = ['bottle', 'mug', 'vase']
    source_category = module.CategoryConverter.category
    path = tmp_path.joinpath(*source_category, 'scene', 'room_0', 'instance.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'floor_name': 'floor', 'objects': {
        'obj' + str(i + 1): {'name': 'obj' + str(i + 1), 'category': category, 'model': 'model_' + str(i),
            'position': [.3 * i, 0., .1], 'quaternion_xyzw': [0., 0., 0., 1.], 'scale': [.2, .2, .2]}
        for i, category in enumerate(categories)},
        'camera_poses': [{'position': [0., -1., .4], 'quaternion_xyzw': [0., 0., 0., 1.]}]}
    question = 'Are the bottle, mug and vase arranged in a straight line?'
    raw = {'big_task': source_category[0], 'small_task': source_category[1], 'scene': 'scene', 'room': 'room_0',
        'runner_task': 'line', 'question': question,
        'answer': answer, 'metadata_json': json.dumps(payload)}
    path.write_text(json.dumps(raw))
    return SourceInstance(path, str(path.relative_to(tmp_path)), *source_category)


def test_conversion_changes_goal_and_keeps_existing_scene_configuration(tmp_path, monkeypatch):
    calls = []
    robot = {'position': [0., -1.2, 0.], 'yaw_deg': 90., 'operation': alignment.OPERATIONS[0]}

    def existing_route_screen(*args):
        calls.append(args)
        return robot, {'source_map_route_preserved': True}

    monkeypatch.setattr(alignment, '_route_screen', existing_route_screen)
    results = []
    for i, answer in enumerate(('yes', 'no')):
        source = source_fixture(tmp_path / str(i), alignment, answer)
        original = source.path.read_bytes()
        converter = alignment.CategoryConverter()
        assert converter.inspect(source)['eligible']
        result = converter.convert(source, ConversionContext(tmp_path / 'outputs'))
        task = result['task_definition']
        task_definition(task)
        assert task['private']['evaluation']['operation'] == alignment.ARRANGEMENT_OPERATION
        assert task['requirements']['capabilities'] == [
            'base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']
        assert 'any object order' in task['public']['objective']
        assert 'without moving anything' in task['public']['objective']
        assert all(label in task['public']['objective'] for label in ('bottle', 'mug', 'vase'))
        assert task['public']['environment'] == alignment._public(source.raw, alignment.OPERATIONS[0])['environment']
        expected_patches = [{'operation': 'robot_pose', 'position': [0., -1.2, 0.],
                             'yaw_deg': 90., 'head_pitch_deg': 45.}]
        for key in alignment.OBJECTS:
            expected_patches.extend({'operation': 'object_physics', 'phase': phase, 'binding': key,
                                     'visual_only': False}
                                    for phase in ('before_construction', 'after_postprocessing'))
        assert task['scene']['variants']['adapted']['patches'] == expected_patches
        assert result['scene_definition']['scene'] == task['scene']
        assert task['scene']['variants']['original']['patches'] == []
        assert source.path.read_bytes() == original
        results.append(result)
    assert len(calls) == 2
    assert results[0]['task_definition']['public'] == results[1]['task_definition']['public']
    assert results[0]['task_definition']['private']['evaluation'] == results[1]['task_definition']['private']['evaluation']


def test_goal_overlay_is_idempotent_and_changes_only_authorized_task_fields(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, alignment, 'yes')
    monkeypatch.setattr(alignment, '_route_screen', lambda *args: ({
        'position': [0., -1.2, 0.], 'yaw_deg': 90., 'operation': alignment.OPERATIONS[0]}, {}))
    task = alignment.CategoryConverter().convert(source, ConversionContext(tmp_path / 'outputs'))['task_definition']
    task['version'] = 1
    task['public'] = alignment._public(source.raw, alignment.OPERATIONS[0])
    task['requirements']['capabilities'] = ['base_translation', 'base_rotation']
    task['private']['evaluation']['operation'] = alignment.OPERATIONS[0]
    original = copy.deepcopy(task)
    revised = alignment.revise_goal(task)
    assert task == original
    assert revised['version'] == 2
    assert alignment.revise_goal(revised) == revised
    assert revised['scene'] is not task['scene']
    assert revised['private']['evaluation']['parameters'] == original['private']['evaluation']['parameters']
    authorized = copy.deepcopy(revised)
    authorized['version'] = original['version']
    for key in ('objective', 'success_criteria', 'constraints'):
        authorized['public'][key] = original['public'][key]
    authorized['requirements']['capabilities'] = original['requirements']['capabilities']
    authorized['private']['evaluation']['operation'] = original['private']['evaluation']['operation']
    assert authorized == original
    task_definition(revised)


def test_existing_route_failure_diagnostic_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(alignment, '_route_screen', lambda *args: (None, {
        'failure_reason': 'no_connected_source_faithful_survey_stations'}))
    source = source_fixture(tmp_path, alignment, 'yes')
    result = alignment.CategoryConverter().convert(source, ConversionContext(tmp_path / 'outputs'))
    assert result['status'] == 'conversion_failed'
    assert result['failure_reason'] == 'not_convertible_under_L1_L2:no_connected_source_faithful_survey_stations'
    assert result['task_definition'] is None


@pytest.mark.parametrize('module', MODULES)
def test_legacy_survey_operations_remain_callable(world_factory, module, tmp_path):
    world = world_factory(module)
    assert module.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']
    evaluator = module.create_evaluator(world.session, world.controller, world.config, tmp_path)
    expected = alignment.AlignmentSurveyEvaluator
    assert isinstance(evaluator, expected)

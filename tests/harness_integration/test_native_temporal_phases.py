"""Check continuous measured temporal events without a simulator or model.

Run: behavior/bin/python -m pytest -q tests/test_native_temporal_phases.py
Fixtures expose native-shaped state and mutations; they do not certify rendering.
"""
from copy import deepcopy
from types import SimpleNamespace as NS
import threading

import numpy as np
import pytest

from wia.native.esi_temporal_phases import (MODULE, OPERATION, TemporalChangeEvaluator,
    TemporalPhaseController, _expected_answer, qualify_operation)


class Object:
    def __init__(self, name, category, position, *, model='original', extent=(.1, .1, .1)):
        self.name, self.category, self.model = name, category, model
        self.position = np.array(position, dtype=float)
        self.quaternion = np.array([0., 0., 0., 1.])
        self.extent, self.scale = np.array(extent), np.ones(3)
        self.fixed_base, self.visual_only, self.visible = True, False, True
        self.prim_path = '/World/' + name
        self.pose_writes = 0

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def set_position_orientation(self, position, orientation):
        self.position, self.quaternion = np.array(position), np.array(orientation)
        self.pose_writes += 1

    def keep_still(self):
        pass


@pytest.fixture
def world():
    objects = {}
    floor = Object('floor', 'floors', [0., 0., -.05], extent=(20., 20., .1))
    objects['floor'] = floor
    bindings, source, phases, relations = {'floor': {'name': 'floor', 'category': 'floors'}}, {}, {
        'image1': [], 'image2': []}, {}
    for index, category in enumerate(('syrup', 'baseball', 'cumin')):
        key = f'container_{index:02d}'
        box = Object('box_' + str(index), 'box', [index, 1., .5], extent=(.4, .4, .4))
        objects[key] = box
        bindings[key] = {'name': box.name, 'category': box.category, 'model': box.model}
        source[key] = {'position': box.position.tolist(), 'quaternion_xyzw': box.quaternion.tolist()}
        for phase in ('image1', 'image2'):
            if index == 2 and phase == 'image2':
                continue
            content = f'{phase}_content_{index:02d}'
            position = [index, 1., .5] if phase == 'image1' else [1000. + index, 1000., 120.]
            obj = Object(content, category, position)
            objects[content] = obj
            bindings[content] = {'name': obj.name, 'category': obj.category, 'model': obj.model}
            source[content] = {'position': [index, 1., .5], 'quaternion_xyzw': [0., 0., 0., 1.],
                               'bbox_size_m': [.1, .1, .1]}
            phases[phase].append(content)
            relations[content] = key
    robot = Object('robot', 'robot', [0., 0., .4])
    robot.sensors = {}
    robot.get_joint_positions = lambda: np.array([.1, .2, .3])
    robot.contact_list = lambda: [NS(body0=robot.prim_path + '/wheel', body1=floor.prim_path,
        normal=np.array([0., 0., 1.]), position=np.array([0., 0., 0.]))]
    scene_objects = [*objects.values(), robot]
    scene = NS(objects=scene_objects, object_registry=lambda kind, name: next(
        (obj for obj in scene_objects if obj.name == name), None))
    session = NS(env=NS(scene=scene), robot=robot, scene_name='house', room_name='room_0',
                 _step_camera_data={'stale': True}, invalidate_camera_cache=lambda: None)
    parameters = {'bindings': bindings, 'source_objects': source, 'phase_keys': phases,
        'container_keys': [f'container_{i:02d}' for i in range(3)], 'relations': relations}
    controller = TemporalPhaseController(session, parameters)
    session.temporal_phase_controller = controller
    executor = NS(debug_gripper_state=lambda: {'object_name': None, 'holding': False, 'constraint_active': False})
    config = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
        'scene': 'house', 'room': 'room_0', 'binding_keys': list(bindings),
        'bindings': {key: row['name'] for key, row in bindings.items()},
        'parameters': {'source_task_type': 'current_state_reasoning', 'original_answer': 'yellow box',
            'public_options': ['brown box', 'red box', 'yellow box'],
            'box_labels': {f'container_{i:02d}': label for i, label in enumerate(('brown', 'red', 'yellow'))}}}
    return NS(session=session, controller=controller, executor=executor, config=config, objects=objects)


def test_world_snapshot_skips_jointless_native_entities_but_tracks_doors(world):
    floor = world.objects['floor']
    floor.n_joints = 0

    def forbidden_joint_read():
        raise AssertionError('Jointless native entities cannot read joint positions.')

    floor.get_joint_positions = forbidden_joint_read
    door = Object('door', 'door', [2., 0., 1.])
    door.n_joints = 1
    door.get_joint_positions = lambda: np.array([0.])
    world.session.env.scene.objects.append(door)
    before = world.controller._unchanged_world()
    assert 'floor' in before['poses']
    assert 'floor' not in before['object_joints']
    assert before['object_joints']['door'] == [0.]
    door.get_joint_positions = lambda: np.array([.2])
    assert world.controller._unchanged_world() != before


def test_survey_reads_authored_camera_mount_and_records_real_changes(world, tmp_path):
    mount = np.array([.1, .2, .3])
    quaternion = NS(GetImaginary=lambda: [0., 0., 0.], GetReal=lambda: 1.)
    sensor = NS(get_attribute=lambda name: mount if name == 'xformOp:translate' else quaternion,
                get_position_orientation=lambda **_: (mount + 1e-6, [0., 0., 0., 1.]))
    world.session.robot.sensors = {'head': sensor}
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    observation = NS(snapshot_id=1, action_step=0)
    evaluator.begin_episode(observation)
    sensor.get_position_orientation = lambda **_: (mount - 1e-6, [0., 0., 0., -1.])
    evaluator.sample_step()
    assert not world.controller.survey_violation
    mount[0] += .001
    evaluator.sample_step()
    evaluator.sample_step()
    assert not world.controller.survey_violation
    assert not evaluator.survey_violations
    assert 'camera_mount:head' in evaluator.survey_environment_warnings
    evaluator.prepare_phase_transition(observation)


def test_survey_records_background_motion_without_cancelling(world, tmp_path):
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    evaluator.begin_episode(NS(snapshot_id=1, action_step=0))
    world.objects['floor'].position[0] += .01
    evaluator.sample_step()
    assert 'object_pose:floor' in evaluator.survey_environment_warnings
    assert not world.controller.survey_violation


@pytest.mark.parametrize('restriction', [None, 'critical', 'missing_policy',
                                        'dependency_error', 'contact', 'contact_error'])
def test_background_drift_and_missing_safety_do_not_cancel_phase_transition(world, tmp_path, restriction):
    door = Object('door', 'door', [2., 0., 1.])
    door.n_joints = 1
    joints = np.array([.7591391])
    door.get_joint_positions = lambda: joints.copy()
    world.session.env.scene.objects.append(door)
    world.session.background_movement_policy = {
        'protected': ['door'] if restriction == 'critical' else [],
        'measurement_errors': ['unresolved support'] if restriction == 'dependency_error' else []}
    if restriction == 'missing_policy':
        world.session.background_movement_policy = None
    world.session._navigation_cache = {'stale': True}
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    observation = NS(snapshot_id=1, action_step=0)
    evaluator.begin_episode(observation)
    if restriction == 'contact':
        contacts = world.session.robot.contact_list
        world.session.robot.contact_list = lambda: contacts() + [NS(
            body0='/World/robot/hand', body1='/World/another_object/link')]
        evaluator.sample_step()
        world.session.robot.contact_list = contacts
    elif restriction == 'contact_error':
        contacts = world.session.robot.contact_list
        def unavailable():
            raise RuntimeError('Contact read failed')
        world.session.robot.contact_list = unavailable
        evaluator.sample_step()
        world.session.robot.contact_list = contacts
    joints[0] += .0061134
    evaluator.sample_step()
    assert not world.controller.survey_violation
    assert 'object_joints:door' in evaluator.survey_environment_warnings
    assert (tmp_path / 'survey_environment_warnings.json').exists()
    if restriction is None:
        assert world.session.background_static_map_stale
        assert not world.session._navigation_cache
    evaluator.prepare_phase_transition(observation)
    world.controller.advance()
    result = evaluator.record(NS(snapshot_id=2, action_step=0), action=None)
    assert result['phase'] == 'phase_2'
    assert result['transition']['world_unchanged']
    assert 'phase_1_contact_history' not in result
    assert door.pose_writes == 0
    assert joints[0] == pytest.approx(.7652525)


def test_same_world_transition_and_answer_require_two_observations(world, tmp_path):
    c = world.controller
    assert qualify_operation(world.session, world.executor, world.config, {})['accepted']
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first = NS(snapshot_id=1, action_step=0)
    assert not evaluator.begin_episode(first)['task_success']
    robot = world.session.robot
    robot_initial = robot.get_position_orientation()
    objects = {key: id(obj) for key, obj in world.objects.items()}
    unchanged_props = [world.objects['image1_content_00'], world.objects['image1_content_01']]
    evaluator.prepare_phase_transition(first)
    c.advance()
    assert c.phase == 'phase_2' and c.transition['physics_steps'] == 0
    assert c.transition['world_unchanged'] and robot.pose_writes == 0
    assert all(obj.pose_writes == 0 for obj in unchanged_props)
    assert objects == {key: id(obj) for key, obj in world.objects.items()}
    for before, after in zip(robot_initial, robot.get_position_orientation()):
        np.testing.assert_array_equal(before, after)
    assert world.session._step_camera_data is None
    with pytest.raises(ValueError, match='Phase 2 observation'):
        evaluator.report_answer('yellow box', first)
    second = NS(snapshot_id=2, action_step=0)
    assert not evaluator.record(second, action=None)['task_success']
    evaluator.report_answer('yellow box', second)
    result = evaluator.record(second, action=None)
    assert result['task_success'] and result['measured_expected_answer'] == 'yellow box'
    assert result['measured_changes']['container_02'] == 'remove'
    assert not world.objects['image1_content_02'].visible
    assert not world.objects['image2_content_00'].visible


def test_wrong_answer_and_source_disagreement_never_receive_credit(world, tmp_path):
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first, second = NS(snapshot_id=10, action_step=3), NS(snapshot_id=11, action_step=3)
    evaluator.begin_episode(first)
    evaluator.prepare_phase_transition(first)
    world.controller.advance()
    evaluator.record(second, action=None)
    evaluator.report_answer('red box', second)
    assert not evaluator.record(second, action=None)['task_success']
    world.config['parameters']['original_answer'] = 'red box'
    assert not evaluator.record(second, action=None)['task_success']
    evaluator.report_answer('yellow box', second)
    assert not evaluator.record(second, action=None)['task_success']


@pytest.mark.parametrize('teleport', [False, True])
def test_new_operational_goal_requires_sampled_phase_2_motion_before_answer(world, tmp_path, teleport):
    clock = NS(current_time=0.)
    world.session.pipeline = NS(og=NS(sim=clock))
    world.executor.dt = .1
    world.executor.config = NS(max_translation_speed_m_s=1., translation_tolerance_m=.01)
    world.config['parameters'].update(phase_2_navigation_required=True,
        phase_2_minimum_base_travel_m=.25, phase_2_container_visit_radius_m=1.)
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first, second = NS(snapshot_id=1, action_step=0), NS(snapshot_id=2, action_step=0)
    evaluator.begin_episode(first)
    evaluator.prepare_phase_transition(first)
    world.controller.advance()
    evaluator.record(second, action=None)
    evaluator.report_answer('yellow box', second)
    assert not evaluator.record(second, action=None)['task_success']
    if teleport:
        world.session.robot.position[1] += .3
        result = evaluator.record(second, action=None)
        assert not result['task_success']
        assert 'native_motion_outside_control_step_sampling' in result['phase_2_native_navigation']['faults']
        return
    for _ in range(3):
        world.session.robot.position[1] += .1
        clock.current_time += .1
        evaluator.sample_step()
    clock.current_time += .1
    evaluator.sample_step()
    third = NS(snapshot_id=3, action_step=1)
    assert not evaluator.record(third, action='MOVE_FORWARD')['task_success']
    evaluator.report_answer('yellow box', third)
    result = evaluator.record(third, action=None)
    assert result['task_success']
    assert result['phase_2_native_navigation']['travel_m'] == pytest.approx(.3)


def test_passive_content_motion_is_recorded_at_phase_boundary(world, tmp_path):
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first = NS(snapshot_id=1, action_step=0)
    evaluator.begin_episode(first)
    world.objects['image1_content_02'].position += [.1, 0., 0.]
    evaluator.prepare_phase_transition(first)
    assert not world.controller.survey_violation
    assert evaluator.survey_environment_warnings
    assert evaluator.phase_1['native_state']['objects']['image1_content_02']['position'][0] == pytest.approx(2.1)
    assert world.controller.phase == 'phase_1' and world.controller.transition is None


def test_unprepared_repeated_stale_and_nonowner_events_do_not_change_world(world, tmp_path):
    c = world.controller
    with pytest.raises(ValueError):
        c.advance()
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    first = NS(snapshot_id=1, action_step=0)
    evaluator.begin_episode(first)
    with pytest.raises(ValueError):
        evaluator.prepare_phase_transition(NS(snapshot_id=0, action_step=0))
    evaluator.prepare_phase_transition(first)
    errors = []
    def worker():
        try:
            c.advance()
        except RuntimeError as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(errors) == 1 and c.phase == 'phase_1'
    c.advance()
    saved = deepcopy(c.transition)
    with pytest.raises(ValueError):
        c.advance()
    assert c.transition == saved and c.phase == 'phase_2'


@pytest.mark.parametrize('a,b,change', [(None, None, 'no_change'),
    (None, {'category': 'x', 'model': 'a'}, 'add'),
    ({'category': 'x', 'model': 'a'}, None, 'remove'),
    ({'category': 'x', 'model': 'a'}, {'category': 'y', 'model': 'b'}, 'replace')])
def test_original_question_types_follow_measured_content_identities(a, b, change):
    first, second = {'container_00': a}, {'container_00': b}
    p = {'source_task_type': 'change_identification', 'target_box_index': 0}
    assert _expected_answer(p, first, second)[0] == change
    p['source_task_type'] = 'change_detection'
    assert _expected_answer(p, first, second)[0] == ('no' if change == 'no_change' else 'yes')


def test_navigation_contact_does_not_cancel_or_block_transition(world, tmp_path):
    evaluator = TemporalChangeEvaluator(world.session, world.executor, world.config, tmp_path)
    observation = NS(snapshot_id=1, action_step=0)
    evaluator.begin_episode(observation)
    original = world.session.robot.contact_list
    world.session.robot.contact_list = lambda: original() + [NS(
        body0='/World/robot/base', body1='/World/table/base_link')]
    evaluator.sample_step()
    assert not world.controller.survey_violation
    assert 'native_check:robot_clear' not in evaluator.survey_environment_warnings
    evaluator.prepare_phase_transition(observation)
    world.controller.advance()
    assert world.controller.phase == 'phase_2'


@pytest.mark.parametrize('action', ['MV_FWD', 'MV_UP', 'ROTATE_CW', 'GRASP',
                                   'RELEASE', 'RIGHT_MV_FWD', 'RIGHT_GRASP'])
def test_survey_rejects_arm_and_hand_commands(action):
    from wia.tasks.temporal_policy import survey_action_allowed
    assert not survey_action_allowed(action)


@pytest.mark.parametrize('action', ['BASE_FWD', 'BASE_TURN_LEFT', 'TRUNK_UP', 'DONE'])
def test_survey_allows_navigation_and_torso(action):
    from wia.tasks.temporal_policy import survey_action_allowed
    assert survey_action_allowed(action)

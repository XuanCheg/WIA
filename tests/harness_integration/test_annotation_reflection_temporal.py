"""Check explicit historical reflection pickup and observation contracts.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_annotation_reflection_temporal.py
Fixtures provide CPU native-shaped contact, grasp, AABB and clock readbacks.
No GPU, native simulation, model, renderer or physical success is claimed.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core.action_interface import BackendCapabilities
from wia.authoring.conversion import ROOT, ConversionContext, SourceInstance
from wia.authoring.converters import specular_reflection_reflection_authoring as authoring
from wia.authoring.converters import specular_reflection_correspondence as correspondence
from wia.authoring.converters import specular_reflection_spatial_relations as spatial
from wia.authoring.converters import temporal_understanding_agent_observation as observation_patrol


class Object:
    def __init__(self, name, category, position, extent, *, model='original_model', visual_only=False, fixed_base=False):
        self.name, self.category, self.model = name, category, model
        self.position, self.extent = np.array(position, float), np.array(extent, float)
        self.quaternion, self.scale = np.array([0., 0., 0., 1.]), np.ones(3)
        self.prim_path = '/World/' + name
        self.visual_only, self.fixed_base = visual_only, fixed_base
        self.speed, self.support_path, self.blocked = 0., '/World/floor', False
        self.robot = None

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def get_linear_velocity(self):
        return np.array([0., 0., self.speed])

    def get_angular_velocity(self):
        return np.zeros(3)

    def contact_list(self):
        held = self.robot is not None and self.robot._ag_obj_in_hand['left'] is self
        if held:
            return [NS(body0=self.prim_path + '/base_link', body1='/World/robot/hand_left',
                position=self.position.tolist(), normal=[0., 0., 1.])]
        return [NS(body0=self.prim_path + '/base_link', body1=self.support_path + '/top',
            position=[*self.position[:2], float(self.aabb[0][2])], normal=[0., 0., 1.])]

    def set_position_orientation(self, *args, **kwargs):
        pytest.fail('An evaluator must never move a native object.')


class R1(Object):
    def __init__(self):
        super().__init__('robot', 'robot', [0., 0., .4], [.3, .3, .8], fixed_base=True)
        self.base_footprint_link_name = 'base_link'
        self.links = {'base_link': self}
        self.finger_links = {'left': [NS(prim_path='/World/robot/finger_left'), NS(prim_path='/World/robot/finger_right')]}
        self.eef_links = {'left': NS(prim_path='/World/robot/hand_left')}
        self._ag_obj_in_hand, self._ag_obj_constraints = {'left': None}, {'left': None}
        self.reading = None
        self.floor_missing, self.bad_contact = False, False

    def is_grasping(self, arm):
        return self.reading if self.reading is not None else (1 if self._ag_obj_in_hand[arm] is not None else -1)

    def contact_list(self):
        result = [] if self.floor_missing else [NS(body0='/World/robot/base_link', body1='/World/floor/base_link',
            position=[0., 0., 0.], normal=[0., 0., 1.])]
        if self.bad_contact:
            result.append(NS(body0='/World/robot/base_link', body1='/World/wall', position=[0., 0., .4], normal=[1., 0., 0.]))
        if self._ag_obj_in_hand['left'] is not None:
            result.append(NS(body0='/World/robot/hand_left', body1=self._ag_obj_in_hand['left'].prim_path + '/base_link',
                position=[1., 0., .2], normal=[0., 0., 1.]))
        return result


def lift_world(tmp_path, *, supported_height=0.):
    robot = R1()
    prop = Object('original_prop', 'bottle', [1., 0., .1 + supported_height], [.1, .1, .2])
    prop.robot = robot
    mirror = Object('original_mirror', 'mirror', [2., 2., 1.], [.1, .5, 1.], visual_only=True)
    floor = Object('floor', 'floors', [0., 0., -.05], [10., 10., .1], fixed_base=True)
    objects = {obj.name: obj for obj in (prop, mirror, floor)}
    scene = NS(objects=objects, object_registry=lambda kind, name: objects.get(name),
        seg_map=NS(get_room_instance_by_point=lambda xy: 'room_0'))
    session = NS(scene_name='fixture', room_name='room_0', robot=robot, env=NS(scene=scene),
        pipeline=NS(og=NS(sim=NS(current_time=0.))))
    if supported_height:
        prop.support_path = '/World/table'
        session.category_interaction_setup = {'accepted': True, 'relocated': True, 'object_names': [prop.name],
            'support_prim_path': prop.support_path, 'group_translation_m': [0., 0., supported_height],
            'object_translations_m': {prop.name: [0., 0., supported_height]}}
    controller = NS(capabilities=BackendCapabilities(base_translation=True, base_rotation=True,
        ee_translation=True, ee_rotation=True, gripper=True, trunk_height=True), dt=.1,
        config=NS(max_translation_speed_m_s=.5, translation_tolerance_m=.01,
            max_rotation_speed_deg_s=90., rotation_tolerance_deg=1.),
        pose=lambda: {'position': robot.position.tolist(), 'quaternion_xyzw': robot.quaternion.tolist()},
        ee_pose=lambda: {'position': [1., 0., .1]}, debug_gripper_state=lambda: authoring._native_grasp(robot, 'left'))
    p = {'original_prop': {'name': prop.name, 'category': prop.category, 'model': prop.model},
        'expected_optical_poses': {'prop': {'position': [1., 0., .1], 'quaternion_xyzw': [0., 0., 0., 1.]},
            'mirror': {'position': mirror.position.tolist(), 'quaternion_xyzw': [0., 0., 0., 1.]}},
        'proposed_start_xy': [0., 0.], 'start_tolerance_m': .12, 'start_heading_tolerance_deg': 3.,
        'minimum_lift_m': .05, 'working_arm': 'left', 'stability_steps': 3, 'minimum_up_dot': .95,
        'floor_z_m': 0., 'floor_height_tolerance_m': .005, 'minimum_floor_normal_z': .5,
        'asset_position_tolerance_m': .02, 'asset_orientation_tolerance_deg': 2., 'mirror_clearance_m': .03}
    config = {'type': 'category_operation', 'module': authoring.MODULE, 'operation': authoring.LIFT_OPERATION,
        'scene': 'fixture', 'room': 'room_0', 'binding_keys': ['prop', 'mirror', 'floor'],
        'bindings': {'prop': prop.name, 'mirror': mirror.name, 'floor': floor.name}, 'parameters': p}
    evaluator = authoring.create_evaluator(session, controller, config, tmp_path)
    observation = NS(snapshot_id=1, action_step=0, task_text='Find the bottle and lift it.')
    return NS(session=session, robot=robot, prop=prop, mirror=mirror, controller=controller,
        config=config, evaluator=evaluator, observation=observation,
        bound={'prop': prop, 'mirror': mirror, 'floor': floor})


def sample(world, *, lift=0., holding=True):
    world.robot._ag_obj_in_hand['left'] = world.prop if holding else None
    world.robot._ag_obj_constraints['left'] = object() if holding else None
    world.prop.position[2] += lift
    world.prop.speed = .1 if lift else 0.
    world.session.pipeline.og.sim.current_time += world.controller.dt
    world.evaluator.sample_step()


def retrieval_world(tmp_path, *, legacy=False, supported_height=0.):
    world = lift_world(tmp_path, supported_height=supported_height)
    world.config = {'type': 'category_operation', 'module': observation_patrol.MODULE,
        'operation': observation_patrol.OPERATION if legacy else observation_patrol.RETRIEVAL_OPERATION,
        'scene': 'fixture', 'room': 'room_0', 'binding_keys': ['prop'], 'bindings': {'prop': world.prop.name},
        'parameters': {'start_xy': [0., 0.], 'start_yaw_rad': 0., 'visit_radius_m': 2.,
            'departure_m': observation_patrol.DEPARTURE_M, 'return_m': observation_patrol.RETURN_M,
            'minimum_travel_m': observation_patrol.MINIMUM_TRAVEL_M, 'stability_steps': observation_patrol.STABILITY_STEPS,
            'source_objects': {'prop': {'name': world.prop.name, 'category': world.prop.category,
                'model': world.prop.model, 'position': [1., 0., .1]}},
            'minimum_lift_m': .05, 'working_arm': 'left', 'lift_stability_steps': 3}}
    world.evaluator = observation_patrol.create_evaluator(world.session, world.controller, world.config, tmp_path)
    return world


def patrol(world, *, retrieve):
    world.evaluator.begin_episode(world.observation)
    for point in np.linspace(.01, .6, 60):
        world.robot.position[0] = point
        sample(world, holding=False)
    if retrieve:
        for _ in range(6):
            sample(world, lift=.01)
        for _ in range(3):
            sample(world)
    for point in np.linspace(.59, 0., 60):
        world.robot.position[0] = point
        sample(world, holding=retrieve)
    for _ in range(16):
        sample(world, holding=retrieve)
    return world.evaluator.record(world.observation, action='DONE')


@pytest.mark.parametrize('supported_height', [0., .5])
def test_observation_retrieval_needs_original_native_lift_and_real_patrol(tmp_path, supported_height):
    world = retrieval_world(tmp_path, supported_height=supported_height)
    result = patrol(world, retrieve=True)
    assert result['task_success'], result['faults']
    assert result['native_visit_count'] == result['native_retrieval_count'] == 1
    assert result['retrieved']['prop']['native_name'] == world.prop.name
    assert result['retrieved']['prop']['lowest_point_lift_m'] == pytest.approx(.06)
    assert result['bag_or_containment_implemented'] is False
    assert world.session.pipeline.og.sim.current_time > 0
    rows = [json.loads(line) for line in (tmp_path / 'trajectory.jsonl').read_text().splitlines()]
    assert rows[-1]['retrieved'] == result['retrieved']


def test_observation_patrol_without_pickup_cannot_satisfy_new_recipe(tmp_path):
    result = patrol(retrieval_world(tmp_path), retrieve=False)
    assert not result['task_success']
    assert result['native_visit_count'] == 1
    assert not result['criteria']['encountered_original_prop_grasped_and_lifted']


def test_observation_start_repair_requires_actual_matching_native_evidence(tmp_path):
    world = retrieval_world(tmp_path)
    world.robot.position[0] = .3
    _, checks = observation_patrol._initial_checks(world.session, world.config)
    assert not checks['source_start_xy']
    world.session.category_interaction_setup = {'accepted': True,
        'spawn_grid': {'requested_source_region': 'room_0'},
        'robot_spawn': {'checks': {'floor_supported': True, 'collision_free': True},
            'robot_position_m': world.robot.position.tolist(),
            'robot_quaternion_xyzw': world.robot.quaternion.tolist()}}
    _, checks = observation_patrol._initial_checks(world.session, world.config)
    assert all(checks.values())
    world.robot.position[0] += .2
    _, checks = observation_patrol._initial_checks(world.session, world.config)
    assert not checks['source_start_xy']


@pytest.mark.parametrize('module', [authoring, observation_patrol])
def test_legacy_manipulation_public_payload_excludes_original_private_qa(module, tmp_path):
    source = sources(module)[0]
    raw = deepcopy(source.raw)
    payload = deepcopy(source.payload)
    raw['answer'] = '918273645'
    if module is observation_patrol:
        # Preserve the source answer/count consistency required before conversion.
        payload['true_count'] = int(raw['answer'])
    surrogate = NS(raw=raw, payload=payload, path=source.path,
        relative_path=source.relative_path, big_task=source.big_task, small_task=source.small_task)
    legacy_config = ({'reflection_authoring_operation': authoring.LIFT_OPERATION}
                     if module is authoring else {'legacy_agent_observation_patrol': True,
                         'agent_observation_operation': observation_patrol.RETRIEVAL_OPERATION})
    result = module.CategoryConverter().convert(surrogate, ConversionContext(tmp_path, config=legacy_config))
    public = json.dumps(result['task_definition']['public'])
    assert raw['answer'] not in public
    assert 'expected_optical_poses' not in public and 'source_objects' not in public


def test_historical_observation_patrol_remains_callable_without_pickup(tmp_path):
    result = patrol(retrieval_world(tmp_path, legacy=True), retrieve=False)
    assert result['task_success']
    assert result['schema'] == 'private_agent_observation_patrol_v1'


@pytest.mark.parametrize('fault', ['unheld_move', 'floor', 'wrong_native_identity', 'no_constraint', 'unsampled_move'])
def test_observation_retrieval_rejects_false_or_unsafe_physics(tmp_path, fault):
    world = retrieval_world(tmp_path)
    world.evaluator.begin_episode(world.observation)
    if fault in ('unheld_move', 'unsampled_move'):
        world.prop.position[0] += .2
    elif fault == 'floor':
        world.robot.floor_missing = True
    else:
        world.robot._ag_obj_in_hand['left'] = world.prop if fault == 'no_constraint' else Object('wrong', 'bottle', [1., 0., .1], [.1, .1, .2])
        world.robot._ag_obj_constraints['left'] = None if fault == 'no_constraint' else object()
        world.prop.position[2] += .1
    if fault != 'unsampled_move':
        world.session.pipeline.og.sim.current_time += .1
        world.evaluator.sample_step()
    result = world.evaluator.record(world.observation, action='DONE')
    assert not result['task_success']
    assert not result['retrieved']
    assert result['faults']


def test_legacy_observation_all_sources_keep_original_counts_and_diagnostics(tmp_path):
    root = ROOT / 'ESI/dataset/json_clean'
    paths = sorted(root.joinpath(*observation_patrol.CATEGORY).rglob('*.json'))
    assert len(paths) == 133
    converter = observation_patrol.CategoryConverter()
    generated, diagnostic, bindings, height_repairs = 0, 0, 0, 0
    for path in paths:
        before = path.read_bytes()
        source = SourceInstance(path, str(path.relative_to(root)), *observation_patrol.CATEGORY)
        result = converter.convert(source, ConversionContext(tmp_path,
            config={'legacy_agent_observation_patrol': True,
                    'agent_observation_operation': observation_patrol.RETRIEVAL_OPERATION}))
        objects = source.payload['objects_meta']
        bindings += len(objects)
        assert result['private_source_evidence']['object_count'] == len(objects)
        if result['private_source_evidence']['implausibly_below_floor_bindings']:
            height_repairs += 1
            assert 'original_bad_height_pending_native_support_repair' in result['warning_codes']
            assert 'original_pose_and_qa_reference_changed' in result['warning_codes']
            legacy = converter.convert(source, ConversionContext(tmp_path,
                config={'legacy_agent_observation_patrol': True,
                        'agent_observation_operation': observation_patrol.OPERATION}))
            assert legacy['status'] == 'conversion_failed' and legacy['diagnostic_only']
            assert legacy['failure_stage'] == 'source_geometry'
        if result['status'] != 'generated':
            diagnostic += 1
            assert result['diagnostic_only'] and result['failure_stage'] == 'source_geometry'
        else:
            generated += 1
            assert result['operation_recipe'] == observation_patrol.RETRIEVAL_OPERATION
            assert set(result['interaction_policy']['group_bindings']) == {obj['_key'] for obj in objects}
            assert result['interaction_policy']['support_search'] == 'measured_surfaces'
            assert result['interaction_policy']['allow_fallback_table'] is True
            assert result['scene_definition']['scene']['variants']['original']['patches'] == []
            assert 'bag' not in json.dumps(result['task_definition']['public']).lower()
        assert path.read_bytes() == before
    assert (generated, diagnostic, bindings, height_repairs) == (133, 0, 253, 13)


@pytest.mark.parametrize('height', (0., .5))
def test_actual_original_prop_hold_and_lowest_aabb_lift_uses_initialized_reference(tmp_path, height):
    world = lift_world(tmp_path, supported_height=height)
    initial_prop = world.prop.position.copy()
    initial_mirror = world.mirror.position.copy()
    qualified = authoring.qualify_operation(world.session, world.controller, world.config, world.bound)
    assert qualified['accepted'] and not qualified['reflected_visibility_qualified']
    np.testing.assert_array_equal(world.prop.position, initial_prop)
    world.evaluator.begin_episode(world.observation)
    assert not world.evaluator.record(world.observation, action=None)['task_success']
    for _ in range(6):
        sample(world, lift=.01)
    for _ in range(3):
        sample(world)
    record = world.evaluator.record(world.observation, action='MV_UP')
    assert record['task_success'] and record['lift_m'] >= .05
    assert record['initial_target']['aabb_min'][2] == pytest.approx(height)
    assert record['native_state']['native_grasp']['object_name'] == world.prop.name
    assert not record['native_initial_reflection_only_verified']
    assert not record['original_qa_used_for_success']
    np.testing.assert_array_equal(world.mirror.position, initial_mirror)


@pytest.mark.parametrize('failure', ('no_hold', 'wrong_object', 'no_constraint', 'short_lift', 'no_control_samples'))
def test_height_or_claim_without_real_correct_grasp_cannot_succeed(tmp_path, failure):
    world = lift_world(tmp_path)
    world.evaluator.begin_episode(world.observation)
    if failure == 'no_control_samples':
        world.robot._ag_obj_in_hand['left'] = world.prop
        world.robot._ag_obj_constraints['left'] = object()
        world.prop.position[2] += .06
    else:
        for _ in range(6):
            sample(world, lift=.005 if failure == 'short_lift' else .01, holding=failure != 'no_hold')
        for _ in range(3):
            sample(world, holding=failure != 'no_hold')
        if failure == 'wrong_object':
            world.robot._ag_obj_in_hand['left'] = Object('other_object', 'bottle', [1., 0., .2], [.1, .1, .2])
        elif failure == 'no_constraint':
            world.robot._ag_obj_constraints['left'] = None
    record = world.evaluator.record(world.observation, action=None)
    assert not record['task_success']


@pytest.mark.parametrize('failure', ('floor', 'wall', 'mirror', 'scale', 'clock', 'base_teleport'))
def test_native_safety_and_optical_anchor_failures_latch(tmp_path, failure):
    world = lift_world(tmp_path)
    world.evaluator.begin_episode(world.observation)
    if failure == 'floor':
        world.robot.floor_missing = True
    elif failure == 'wall':
        world.robot.bad_contact = True
    elif failure == 'mirror':
        world.mirror.position[0] += .1
    elif failure == 'scale':
        world.prop.scale[0] *= .5
    elif failure == 'base_teleport':
        world.robot.position[0] += 1.
    if failure == 'clock':
        world.evaluator.sample_step()
    else:
        sample(world)
    assert world.evaluator.faults
    assert not world.evaluator.record(world.observation, action=None)['task_success']


def test_unaccepted_or_xy_support_changes_cannot_become_initialized_reference(tmp_path):
    world = lift_world(tmp_path, supported_height=.5)
    world.session.category_interaction_setup['accepted'] = False
    assert not authoring.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']
    world.session.category_interaction_setup['accepted'] = True
    world.session.category_interaction_setup['object_translations_m'][world.prop.name][0] = .2
    assert not authoring.qualify_operation(world.session, world.controller, world.config, world.bound)['accepted']


def sources(module):
    category = module.CategoryConverter.category
    corpus = ROOT / 'ESI/dataset/json_clean'
    return [SourceInstance(path, str(path.relative_to(corpus)), *category)
        for path in sorted(corpus.joinpath(*category).rglob('*.json'))]


def test_legacy_authoring_all_sources_have_answer_independent_pickup_without_public_optical_hints(tmp_path):
    converter = authoring.CategoryConverter()
    originals = sources(authoring)
    assert len(originals) == 99
    for source in originals:
        before = source.path.read_bytes()
        result = converter.convert(source, ConversionContext(tmp_path,
            config={'reflection_authoring_operation': authoring.LIFT_OPERATION}))
        task = result['task_definition']
        public = json.dumps(task['public']).lower()
        assert 'mirror' not in public and 'reflection' not in public and 'original question' not in public
        assert task['private']['evaluation']['operation'] == authoring.LIFT_OPERATION
        assert task['private']['evaluation']['parameters']['minimum_lift_m'] == .05
        assert result['interaction_policy']['group_bindings'] == ['prop']
        assert result['interaction_policy']['relocation_policy'] == 'independent_groups'
        assert result['interaction_policy']['placement_groups'] == [['prop']]
        assert result['interaction_policy']['source_camera_policy'] == 'preserve'
        assert result['interaction_policy']['support_search'] == 'measured_surfaces'
        assert result['interaction_policy']['allow_fallback_table'] is True
        assert result['private_source_evidence']['source_qa_remains_original_reference_only']
        assert 'gripper' in task['requirements']['capabilities']
        patches = task['scene']['variants']['adapted']['patches']
        assert [patch['operation'] for patch in patches] == ['robot_pose', 'object_physics']
        assert patches[-1]['binding'] == 'prop' and patches[-1]['phase'] == 'after_postprocessing'
        assert task['scene']['variants']['original']['patches'] == []
        assert 'native_initial_reflection_only_unverified' in result['warning_codes']
        assert source.path.read_bytes() == before


@pytest.mark.parametrize('module, count', ((correspondence, 88), (spatial, 113)))
def test_legacy_other_reflection_classes_keep_navigation_and_original_optical_poses(module, count, tmp_path):
    originals = sources(module)
    assert len(originals) == count
    legacy_config = ({'correspondence_operation': correspondence.OPERATIONS[0]}
                     if module is correspondence else {'spatial_relations_operation': spatial.OPERATIONS[2]})
    result = module.CategoryConverter().convert(originals[0], ConversionContext(tmp_path, config=legacy_config))
    task = result['task_definition']
    assert task['requirements']['capabilities'] == ['base_translation', 'base_rotation']
    assert task['private']['evaluation']['operation'] in module.OPERATIONS
    assert all(patch['operation'] == 'robot_pose' for patch in task['scene']['variants']['adapted']['patches'])
    vector = np.array((1., 2., 3.), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    assert module._array(vector).tolist() == [1., 2., 3.]


def test_historical_authoring_survey_operation_remains_callable(tmp_path):
    result = authoring.CategoryConverter().convert(sources(authoring)[0],
        ConversionContext(tmp_path, config={'reflection_authoring_operation': authoring.OPERATION}))
    assert result['task_definition']['private']['evaluation']['operation'] == authoring.OPERATION
    assert result['task_definition']['requirements']['capabilities'] == ['base_translation', 'base_rotation']
    assert 'mirror survey' in result['task_definition']['public']['objective']

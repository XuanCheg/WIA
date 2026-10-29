"""Check read-only rigid adapter protocols without starting OmniGibson.

Run: python -m pytest -q tests/test_wia_rigid_native_adapter.py
tests/test_rigid_insertion_scoring.py
Fixtures exercise native API contracts; they do not qualify simulator physics.
"""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import pytest
import torch

from wia.authoring.converters import rigid_partial_insertion as module


class SurfaceMesh:
    def __init__(self, owner, points, faces=None):
        self.owner = owner
        self.points = torch.as_tensor(points, dtype=torch.float64)
        self.faces = None if faces is None else torch.as_tensor(faces, dtype=torch.long)
        self.collision_enabled = True
        self.contains = lambda points: points[:, 2] <= 0

    @property
    def mesh_face_centroids(self):
        return self.points[self.faces].mean(dim=1)

    def transform_local_points_to_world(self, points):
        return points * torch.as_tensor(self.owner.scale) + torch.as_tensor(self.owner.position)

    def check_points_in_volume(self, points):
        return self.contains(points)


class Volume:
    def __init__(self):
        self.is_meta_link = True
        self.meta_link_type = 'fillable'
        self.prim_path = '/World/container/cavity'
        self.visual_meshes = {'volume': object()}
        self.collision_meshes = {}
        self.visual_bounds_reads = 0

    @property
    def aabb(self):
        raise AssertionError('Meta-link collision bounds are not cavity bounds.')

    @property
    def visual_aabb(self):
        self.visual_bounds_reads += 1
        return np.array([-1., -1., 0.]), np.array([1., 1., 1.])

    def check_points_in_volume(self, points):
        low, high = self.visual_aabb
        return ((points >= torch.as_tensor(low)) & (points <= torch.as_tensor(high))).all(dim=1)


class NativeObject:
    def __init__(self, name, *, bounds=None, position=(0., 0., 0.), fixed=False):
        self.name, self.prim_path = name, '/World/' + name
        self.position, self.quaternion = np.array(position, dtype=float), np.array([0., 0., 0., 1.])
        self.scale = np.ones(3)
        self.bounds = bounds
        self.fixed_base, self.visual_only = fixed, False
        self.linear, self.angular = np.zeros(3), np.zeros(3)
        self.contacts, self.touching, self.touching_calls = [], {}, []
        self.scene = NS(idx=7)
        self.links = {}
        self.states = {}

    @property
    def aabb(self):
        if self.bounds is not None:
            return tuple(np.array(value, dtype=float) + self.position for value in self.bounds)
        mesh = self.links['body'].collision_meshes['mesh']
        points = mesh.transform_local_points_to_world(mesh.points).numpy()
        return points.min(axis=0), points.max(axis=0)

    def get_position_orientation(self):
        return self.position, self.quaternion

    def get_linear_velocity(self):
        return self.linear

    def get_angular_velocity(self):
        return self.angular

    def contact_list(self):
        return list(self.contacts)


def contact(obj, other, z, *, normal=(0., 0., 1.)):
    return NS(body0=obj.prim_path + '/body', body1=other.prim_path + '/body',
              position=np.array([obj.position[0], obj.position[1], z]), normal=np.array(normal))


@pytest.fixture
def native(monkeypatch, tmp_path):
    og, states, macros, utils, usd, sampling = (ModuleType(name) for name in
        ('omnigibson', 'omnigibson.object_states', 'omnigibson.macros', 'omnigibson.utils',
         'omnigibson.utils.usd_utils', 'omnigibson.utils.sampling_utils'))
    states.Touching = object()
    macros.macros = NS(object_states=NS(contains=NS(CONTAINER_META_LINK_TYPES=['fillable', 'openfillable'])))
    for name, value in ((og.__name__, og), (states.__name__, states), (macros.__name__, macros),
                        (utils.__name__, utils), (usd.__name__, usd), (sampling.__name__, sampling)):
        monkeypatch.setitem(sys.modules, name, value)
    target = NativeObject('item', position=(2., 0., .2))
    mesh = SurfaceMesh(target, [[-.1, -.1, 0.], [.1, -.1, .1], [0., .1, 0.]], [[0, 1, 2]])
    target.links = {'body': NS(is_meta_link=False, prim_path=target.prim_path + '/body', collision_meshes={'mesh': mesh})}
    container = NativeObject('container', bounds=([-1., -1., -.1], [1., 1., 1.]))
    solid = SurfaceMesh(container, [[0., 0., 0.]])
    volume = Volume()
    container.links = {'cavity': volume,
                       'body': NS(is_meta_link=False, prim_path=container.prim_path + '/body', collision_meshes={'mesh': solid})}
    pickup = NativeObject('pickup', bounds=([1.5, -.5, 0.], [2.5, .5, .2]), fixed=True)
    table = NativeObject('table', bounds=([-2., -2., -1.], [2., 2., -.1]), fixed=True)
    objects = {obj.name: obj for obj in (target, container, pickup, table)}
    for obj in objects.values():
        def touching(other, obj=obj):
            obj.touching_calls.append(other.name)
            return obj.touching.get(other.name, False)
        obj.states[states.Touching] = NS(get_value=touching)
    readings, grasp_calls = {'left': -1, 'right': -1}, []
    def is_grasping(arm):
        grasp_calls.append(arm)
        return readings[arm]
    far = NS(aabb=(np.array([-3., -3., 2.]), np.array([-2.8, -2.8, 2.2])))
    robot = NS(arm_names=('left', 'right'), prim_path='/World/robot',
        _ag_obj_in_hand={'left': None, 'right': None}, _ag_obj_constraints={'left': None, 'right': None},
        _ag_release_counter={'left': None, 'right': None},
        joint_positions=np.array([.04, .04, .025, .025]),
        control_limits={'position': (np.zeros(4), np.array([.04, .04, .05, .05]))},
        controllers={'gripper_left': NS(dof_idx=np.array([0, 1])), 'gripper_right': NS(dof_idx=np.array([2, 3]))},
        finger_joint_names={'left': ('left0', 'left1'), 'right': ('right0', 'right1')},
        joints={name: NS(joint_type='prismatic') for name in ('left0', 'left1', 'right0', 'right1')},
        eef_links={'left': far, 'right': far}, finger_links={'left': [far, far], 'right': [far, far]},
        is_grasping=is_grasping)
    robot.get_joint_positions = lambda: robot.joint_positions
    control_evidence = {'available': True, 'state_setter_used': False}
    def forbidden_debug():
        raise AssertionError('The working-arm-only debug snapshot must not define aggregate release.')
    controller = NS(working_arm='left', gripper_config=NS(joint_tolerance_m=.001),
                    geometric_rearrangement_evidence=lambda: dict(control_evidence), debug_gripper_state=forbidden_debug)
    clock = NS(current_time=0.)
    scene = NS(objects=objects, object_registry=lambda kind, name: objects.get(name))
    session = NS(robot=robot, env=NS(scene=scene), pipeline=NS(og=NS(sim=clock)))
    separation, contact_calls, ray_calls = [0.], [], []
    def contact_data(scene_index, **kwargs):
        contact_calls.append((scene_index, kwargs))
        return [[None, None, None, None, None, value] for value in separation]
    usd.RigidContactAPI = NS(get_contact_data=contact_data)
    surfaces = {pickup.prim_path: .2, table.prim_path: -.1}
    def ray(start, end, *, only_closest):
        ray_calls.append((start, end, only_closest))
        hits = []
        for obj in (pickup, table):
            low, high = obj.aabb
            z = surfaces[obj.prim_path]
            if np.all(np.array(start[:2]) >= low[:2]) and np.all(np.array(start[:2]) <= high[:2]) and end[2] <= z <= start[2]:
                hits.append({'hit': True, 'rigidBody': obj.prim_path + '/body', 'position': [*start[:2], z],
                             'normal': [0., 0., 1.], 'distance': start[2] - z})
        return hits
    sampling.raytest = ray
    config = {'bindings': {name: name for name in objects}, 'parameters': {
        'scorer_version': module.REVISION, 'targets': ['item'], 'containers': ['container'],
        'expected_containers': {'item': 'container'}, 'minimum_lift_m': .05,
        'stability_duration_s': .5, 'maximum_linear_speed_m_s': .02,
        'maximum_angular_speed_rad_s': .1, 'max_sample_gap_s': None,
        'maximum_container_rotation_rad': .05, 'scale_relative_tolerance': .001,
        'container_pose_tolerance_m': .01, 'withdrawal_clearance_m': .05,
        'interior_margin_m': .01, 'penetration_tolerance_m': .001,
        'expected_answers': {'item': 'medium container'}, 'target_labels': {'item': 'item'},
        'initial_supports': {'item': 'pickup', 'container': 'table'},
        'maximum_initial_linear_speed_m_s': .02, 'maximum_initial_angular_speed_rad_s': .2,
        'initial_support_tolerance_m': .015}}
    evaluator = module.RigidPartialInsertionEvaluator(session, controller, config, tmp_path)
    return NS(target=target, container=container, pickup=pickup, table=table, objects=objects,
              mesh=mesh, solid=solid, volume=volume, robot=robot, controller=controller,
              readings=readings, grasp_calls=grasp_calls, control_evidence=control_evidence,
              clock=clock, session=session, config=config, evaluator=evaluator,
              observation=NS(snapshot_id='cpu-protocol', action_step=0), contact_calls=contact_calls,
              ray_calls=ray_calls, sampling=sampling, separation=separation, surfaces=surfaces)


def step(native, time):
    native.clock.current_time = time
    native.evaluator.sample_step()
    return native.evaluator.scorer.result()


def pickup_and_release(native, *, qualify=True):
    if qualify:
        supported_initial_state(native)
    native.evaluator.begin_episode(native.observation)
    native.target.contacts = []
    native.robot._ag_obj_in_hand['left'] = native.target
    native.robot._ag_obj_constraints['left'] = object()
    native.robot.joint_positions[:2] = 0
    native.readings['left'] = 1
    step(native, 1)
    native.target.position[2] = .3
    step(native, 2)
    native.robot._ag_obj_in_hand['left'] = None
    native.robot._ag_obj_constraints['left'] = None
    native.robot.joint_positions[:2] = .04
    native.readings['left'] = -1
    native.target.position[:] = [0., 0., .2]
    native.target.touching['container'] = True
    step(native, 3)
    return step(native, 3.5)


def supported_initial_state(native):
    for obj, support in ((native.target, native.pickup), (native.container, native.table)):
        obj.touching[support.name] = True
        obj.contacts = [contact(obj, support, float(obj.aabb[0][2]))]


def test_two_arm_readback_allows_inactive_half_open_empty_hand(native):
    result = pickup_and_release(native)
    assert result['physical_success'] is True
    grasp = native.evaluator.last_sample['grasp']
    assert grasp['gripper_open'] is True
    assert grasp['all_grippers_open'] is False
    assert grasp['arms']['right']['joint_positions_m'] == [.025, .025]
    assert grasp['arms']['right']['open_joint_limits_m'] == [.05, .05]
    assert set(native.grasp_calls) == {'left', 'right'}
    assert native.target.touching_calls[-1] == 'container'
    assert native.contact_calls[-1][0] == 7
    assert native.contact_calls[-1][1]['row_prim_paths'] == ['/World/item/body']


@pytest.mark.parametrize('initial_problem', ['unsupported', 'unavailable'])
def test_initial_qualification_failure_cannot_be_repaired_by_later_reducer_success(
        native, monkeypatch, initial_problem):
    if initial_problem == 'unavailable':
        supported_initial_state(native)
        def no_initial_ray(*args, **kwargs):
            raise RuntimeError('Initial native support surface unavailable.')
        monkeypatch.setattr(native.sampling, 'raytest', no_initial_ray)
    assert pickup_and_release(native, qualify=False)['physical_success'] is True
    original = deepcopy(native.evaluator.initial_qualification)
    assert original['accepted'] is False
    def forbidden_requalification(*args, **kwargs):
        raise AssertionError('Initial qualification must remain fixed after manipulation.')
    monkeypatch.setattr(module, 'qualify_operation', forbidden_requalification)
    native.evaluator.answer = {'item': 'medium container'}
    receipt = native.evaluator.record(native.observation)
    assert receipt['initial_qualification'] == original
    assert receipt['evaluation_validity'] == 'unavailable'
    assert receipt['task_success'] is receipt['physical_success'] is False
    assert receipt['completion_score'] is receipt['score_percent'] is None
    assert receipt['diagnostic_reducer_physical_success'] is True
    assert receipt['verified_targets'] == ['item']
    assert receipt['physical_scorer_schema'] == 'rigid_insertion_progress_v1'
    assert receipt['physical_scorer_version'] == 'rigid_insertion_progress_v1'
    assert receipt['schema'] == module.REVISION


@pytest.mark.parametrize('mode', ['hold', 'pending', 'closed_working', 'unknown'])
def test_other_arm_attachments_and_working_release_are_current_final_gates(native, mode):
    assert pickup_and_release(native)['physical_success']
    if mode == 'hold':
        native.robot._ag_obj_in_hand['right'] = NS(name='other')
        native.robot._ag_obj_constraints['right'] = object()
        native.readings['right'] = 1
    elif mode == 'pending':
        native.robot._ag_obj_in_hand['right'] = native.target
        native.robot._ag_release_counter['right'] = 0
    elif mode == 'closed_working':
        native.robot.joint_positions[:2] = 0
    else:
        native.readings['right'] = 0
    result = step(native, 4)
    assert result['physical_success'] is False
    assert result['verified_targets'] == ['item']
    assert result['constraints_satisfied'] is True
    assert native.evaluator.last_sample['grasp']['arms']['right']


def test_multiple_native_identities_never_create_a_single_pickup_event(native):
    native.robot._ag_obj_in_hand.update(left=native.target, right=NS(name='other'))
    native.robot._ag_obj_constraints.update(left=object(), right=object())
    native.readings.update(left=1, right=1)
    grasp = module._aggregate_grasp(native.robot, native.controller)
    assert grasp['available'] is False
    assert grasp['object_name'] == 'item'
    assert grasp['arms']['right']['object_name'] == 'other'
    assert any('Multiple native held identities' in reason for reason in grasp['measurement_errors'])


def test_native_open_limits_are_read_independently_for_each_arm(native):
    native.robot.control_limits['position'][1][:] = [.06, .06, .07, .07]
    native.robot.joint_positions[:] = [.04, .04, .07, .07]
    grasp = module._aggregate_grasp(native.robot, native.controller)
    assert grasp['available'] is True
    assert grasp['gripper_open'] is False
    assert grasp['arms']['right']['gripper_open'] is True
    assert grasp['arms']['left']['open_joint_limits_m'] == [.06, .06]


def test_measurement_does_not_change_native_state(native):
    native.evaluator.begin_episode(native.observation)
    positions = native.robot.joint_positions.copy()
    scales = {key: obj.scale.copy() for key, obj in native.objects.items()}
    poses = {key: (obj.position.copy(), obj.quaternion.copy()) for key, obj in native.objects.items()}
    attachments = dict(native.robot._ag_obj_in_hand)
    timestamp = native.clock.current_time
    measured = native.evaluator._measure()
    np.testing.assert_array_equal(native.robot.joint_positions, positions)
    for key, obj in native.objects.items():
        np.testing.assert_array_equal(obj.scale, scales[key])
        np.testing.assert_array_equal(obj.position, poses[key][0])
        np.testing.assert_array_equal(obj.quaternion, poses[key][1])
    assert native.robot._ag_obj_in_hand == attachments
    assert native.clock.current_time == timestamp
    json.dumps(measured, allow_nan=False)


@pytest.mark.parametrize('key', ['target', 'container'])
def test_scale_changes_latch_constraint_failure_even_after_restoration(native, key):
    assert pickup_and_release(native)['physical_success']
    obj = getattr(native, key)
    obj.scale[0] = 1.1
    result = step(native, 4)
    assert native.evaluator.last_sample['control_valid'] is False
    assert result['constraints_satisfied'] is False
    assert not result['physical_success']
    obj.scale[:] = 1
    assert step(native, 5)['constraints_satisfied'] is False
    assert native.evaluator.last_sample['control_valid'] is True


@pytest.mark.parametrize('mode', ['position', 'rotation'])
def test_container_pose_tolerances_are_explicit_and_latch(native, mode):
    assert pickup_and_release(native)['physical_success']
    if mode == 'position':
        native.container.position[0] = .02
    else:
        angle = .1
        native.container.quaternion[:] = [0., 0., np.sin(angle / 2), np.cos(angle / 2)]
    result = step(native, 4)
    assert result['constraints_satisfied'] is False
    assert native.evaluator.last_sample['protected_geometry']['objects']['container']['pose_unchanged'] is False


def test_quaternion_sign_and_norm_do_not_create_rotation_failure(native):
    native.evaluator.begin_episode(native.observation)
    native.container.quaternion[:] = [0., 0., 0., -2.]
    measured = native.evaluator._measure()
    assert measured['control_valid'] is True
    assert measured['protected_geometry']['objects']['container']['rotation_change_rad'] == 0


def test_missing_scale_is_unavailable_instead_of_a_scale_violation(native):
    assert pickup_and_release(native)['physical_success']
    del native.container.scale
    result = step(native, 4)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['constraints_satisfied'] is None
    assert not result['physical_success']
    assert native.evaluator.last_sample['protected_geometry']['measurement_errors']


def test_partial_read_failure_does_not_hide_scale_violation_or_reuse_old_sample(native):
    assert pickup_and_release(native)['physical_success']
    native.container.scale[0] = 1.1
    def unavailable_velocity():
        raise RuntimeError('Native velocity unavailable.')
    native.target.get_linear_velocity = unavailable_velocity
    result = step(native, 4)
    assert result['constraints_satisfied'] is False
    assert result['evaluation_validity'] == 'unavailable'
    assert native.evaluator.last_sample['native_time_s'] == 4
    assert 'targets' not in native.evaluator.last_sample
    assert 'Native velocity unavailable' in native.evaluator.measurement_error


def test_missing_public_control_evidence_is_not_an_explicit_violation(native):
    native.evaluator.begin_episode(native.observation)
    del native.control_evidence['state_setter_used']
    result = step(native, 1)
    assert result['constraints_satisfied'] is None
    assert result['evaluation_validity'] == 'unavailable'


def test_unavailable_control_accessor_does_not_erase_independent_scale_violation(native):
    assert pickup_and_release(native)['physical_success']
    native.container.scale[0] = 1.1
    def unavailable_control():
        raise RuntimeError('Native control evidence unavailable.')
    native.controller.geometric_rearrangement_evidence = unavailable_control
    result = step(native, 4)
    assert result['constraints_satisfied'] is False
    assert result['evaluation_validity'] == 'unavailable'
    assert native.evaluator.last_sample['protected_geometry']['unchanged'] is False


def test_real_surface_triangle_centroid_can_establish_partial_interior(native):
    native.target.position[:] = 0
    native.mesh.points = torch.tensor([[-2., 0., .5], [2., 0., .5], [0., 2., .5]], dtype=torch.float64)
    evidence = module.partial_interior_evidence(native.target, native.container, .01)
    assert evidence['interior_valid'] is True
    assert evidence['partial_inside'] is True
    assert evidence['surface_sample_count'] == 4
    assert evidence['positive_witness_count'] == 1
    np.testing.assert_allclose(evidence['witness_world_m'], [[0., 2 / 3, .5]])


def test_top_rest_uses_visual_cavity_bounds_and_cannot_be_insertion(native):
    native.target.position[:] = [0., 0., 1.1]
    evidence = module.partial_interior_evidence(native.target, native.container, .01)
    assert evidence['interior_valid'] is True
    assert evidence['partial_inside'] is False
    assert evidence['positive_witness_count'] == 0
    assert native.volume.visual_bounds_reads > 0


def test_overlapping_boxes_without_surface_witness_remain_inconclusive(native):
    native.target.position[:] = 0
    native.mesh.points = torch.tensor([[-2., -2., .5], [2., -2., .5], [0., 2., .5]], dtype=torch.float64)
    native.mesh.faces = None
    evidence = module.partial_interior_evidence(native.target, native.container, .01)
    assert evidence['partial_inside'] is None
    assert evidence['positive_witness_count'] == 0
    assert evidence['reason']


@pytest.mark.parametrize('mode', ['missing_volume', 'missing_solids', 'inside_solid'])
def test_missing_geometry_or_container_solid_points_never_certify_interior(native, mode):
    native.target.position[:] = [0., 0., .2]
    if mode == 'missing_volume':
        del native.container.links['cavity']
    elif mode == 'missing_solids':
        native.solid.collision_enabled = False
    else:
        native.solid.contains = lambda points: torch.ones(len(points), dtype=torch.bool)
    evidence = module.partial_interior_evidence(native.target, native.container, .01)
    assert evidence['partial_inside'] is not True
    if mode != 'inside_solid':
        assert evidence['interior_valid'] is None


def test_disabled_target_collision_geometry_does_not_supply_surface_witness(native):
    native.mesh.collision_enabled = False
    with pytest.raises(ValueError, match='surface points are unavailable'):
        module.partial_interior_evidence(native.target, native.container, .01)


def test_missing_touching_keeps_true_interior_but_blocks_physical_success(native):
    native.evaluator.begin_episode(native.observation)
    native.target.position[:] = [0., 0., .2]
    native.target.touching['container'] = None
    measured = native.evaluator._measure()['targets']['item']['containers']['container']
    assert measured['partial_inside'] is True
    assert measured['touching'] is None
    assert 'Touching evidence is unavailable' in measured['contact_error']


def test_actual_contact_geometry_penetration_blocks_insertion(native):
    native.separation[:] = [-.01]
    result = pickup_and_release(native)
    relation = native.evaluator.last_sample['targets']['item']['containers']['container']
    assert relation['partial_inside'] is True
    assert relation['touching'] is True
    assert relation['penetration_free'] is False
    assert relation['contact_geometry']['separation_source'] == 'native_physx_contact_data'
    assert not result['physical_success']


def test_declared_initial_supports_require_native_contact_and_upward_surface(native):
    supported_initial_state(native)
    result = module.qualify_operation(native.session, native.controller, native.config, native.objects)
    assert result['accepted'] is True
    assert result['evaluation_validity'] == 'complete'
    assert result['initial_supports']['item']['support_binding'] == 'pickup'
    assert result['initial_supports']['container']['support_binding'] == 'table'
    assert result['initial_grasp']['arms']['right']['gripper_open'] is False
    assert result['initial_grasp']['gripper_open'] is True
    assert len(native.ray_calls) == 2
    assert all(call[2] is False for call in native.ray_calls)
    assert not result['complete_physical_execution_verified']
    assert native.clock.current_time == 0


def test_qualifier_accepts_action_executor_wrapping_native_controller(native):
    supported_initial_state(native)
    result = module.qualify_operation(native.session, NS(controller=native.controller),
                                      native.config, native.objects)
    assert result['accepted'] is True
    assert result['initial_grasp']['working_arm'] == 'left'


@pytest.mark.parametrize('mode', ['no_contact', 'no_touching', 'side_contact', 'vertical_gap',
                                  'fast_linear', 'fast_angular', 'robot_contact', 'wrong_support'])
def test_initial_support_and_stability_failures_cannot_be_certified_by_bounds(native, mode):
    supported_initial_state(native)
    if mode == 'no_contact':
        native.target.contacts = []
    elif mode == 'no_touching':
        native.target.touching['pickup'] = False
    elif mode == 'side_contact':
        native.target.contacts[0].normal[:] = [1., 0., 0.]
    elif mode == 'vertical_gap':
        native.target.position[2] += .03
    elif mode == 'fast_linear':
        native.target.linear[0] = .03
    elif mode == 'fast_angular':
        native.target.angular[0] = .3
    elif mode == 'robot_contact':
        native.target.contacts.append(NS(body0='/World/item/body', body1='/World/robot/right_finger'))
    else:
        native.config['parameters']['initial_supports']['item'] = 'table'
    result = module.qualify_operation(native.session, native.controller, native.config, native.objects)
    assert result['accepted'] is False
    assert result['initial_supports']['item']['passed'] is False


@pytest.mark.parametrize('mode', ['no_normal', 'unknown_touching', 'ray_unavailable'])
def test_missing_initial_support_measurements_are_explicitly_unavailable(native, mode):
    supported_initial_state(native)
    if mode == 'no_normal':
        del native.target.contacts[0].normal
    elif mode == 'unknown_touching':
        native.target.touching['pickup'] = None
    else:
        def no_ray(*args, **kwargs):
            raise RuntimeError('Native ray unavailable.')
        native.sampling.raytest = no_ray
    result = module.qualify_operation(native.session, native.controller, native.config, native.objects)
    assert result['accepted'] is False
    assert result['evaluation_validity'] == 'unavailable'
    assert result['initial_supports']['item']['available'] is False


def test_initial_support_rays_do_not_use_a_furniture_aabb_top_as_surface(native):
    supported_initial_state(native)
    native.table.bounds = ([-2., -2., -1.], [2., 2., 5.])
    result = module.qualify_operation(native.session, native.controller, native.config, native.objects)
    assert result['accepted'] is True
    witness = result['initial_supports']['container']['support_witnesses'][0]
    assert witness['surface_position_m'][2] == -.1


@pytest.mark.parametrize('key', ['maximum_container_rotation_rad', 'scale_relative_tolerance'])
def test_new_geometry_protection_thresholds_have_no_silent_default(native, key, tmp_path):
    config = deepcopy(native.config)
    del config['parameters'][key]
    with pytest.raises(KeyError):
        module.RigidPartialInsertionEvaluator(native.session, native.controller, config, tmp_path)

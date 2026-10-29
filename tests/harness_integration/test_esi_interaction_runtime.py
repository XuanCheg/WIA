"""Exercise initialization, current contacts and rollback with CPU native-shaped APIs.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_esi_interaction_runtime.py --basetemp=/tmp/esi-interaction-runtime-fixtures
No simulator, CUDA, renderer, model or robot action is started. Synthetic contact
and state fixtures do not establish a native grasp or a real collision-free IK path.
"""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.tasks import interaction_setup as setup


def test_native_structured_contact_vectors_preserve_xyz_and_reject_nonfinite():
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    vector = np.array((1., 2., 3.), dtype=dtype)
    assert setup.array(vector).tolist() == [1., 2., 3.]
    assert setup.array(np.array([(1., 2., 3.), (4., 5., 6.)], dtype=dtype)).tolist() == [
        [1., 2., 3.], [4., 5., 6.]]
    vector['z'] = np.nan
    with pytest.raises(ValueError, match='finite'):
        setup.array(vector)


class Object:
    """Minimal movable native object with pose-dependent measured AABBs."""
    def __init__(self, world, name, category, position, extent, *, fixed=False):
        self.world, self.name, self.category = world, name, category
        self.model = name + '_original_model'
        self.prim_path = '/World/' + name
        self.position = np.array(position, dtype=float)
        self.quaternion = np.array([0., 0., 0., 1.])
        self.extent = np.array(extent, dtype=float)
        self.scale = np.ones(3)
        self.visual_only, self.fixed_base = False, fixed
        self.in_rooms = ['room_0']
        self.states = {}
        self.setter_calls = 0

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def set_position_orientation(self, position=None, orientation=None):
        self.setter_calls += 1
        if position is not None:
            self.position = np.array(position, dtype=float)
        if orientation is not None:
            self.quaternion = np.array(orientation, dtype=float)
        if self.name == 'robot' and position is not None:
            self.world.placements += 1
            self.world.steps_this_placement = 0
            self.world.cache_at_placement.append(deepcopy(self.world.session.camera_cache))

    def keep_still(self):
        pass

    def contact_list(self):
        if self.name == 'robot':
            return self.world.robot_contacts()
        if self.name not in ('target', 'companion', 'second_target', 'second_reference'):
            return []
        if not self.world.on_table(self):
            return []
        if self.world.support_contact == 'missing':
            return []
        normal = [1., 0., 0.] if self.world.support_contact == 'side' else [0., 0., 1.]
        z = .95 if self.world.support_contact == 'wrong_height' else .7
        return [SimpleNamespace(body0=self.prim_path + '/base_link', body1='/World/table/top',
                                normal=normal, position=[*self.position[:2], z])]


class Wheel:
    def __init__(self, robot, offset_y):
        self.robot, self.offset_y = robot, offset_y

    @property
    def aabb(self):
        point = self.robot.position + [0., self.offset_y, -.35]
        return point - [.05, .05, .05], point + [.05, .05, .05]


class Simulation:
    def __init__(self, world):
        self.world = world
        self.dump_count = self.load_count = self.step_count = 0

    def dump_state(self, serialized=False):
        assert not serialized
        self.dump_count += 1
        return {obj.name: (obj.position.copy(), obj.quaternion.copy()) for obj in self.world.objects}

    def load_state(self, state, serialized=False):
        assert not serialized
        self.load_count += 1
        for obj in self.world.objects:
            obj.position, obj.quaternion = (value.copy() for value in state[obj.name])

    def step(self):
        self.step_count += 1
        world = self.world
        world.steps_this_placement += 1
        # The native wheel AABB starts at root_z - 0.4. Initial placement uses
        # a 5 mm contact offset; one synthetic physics step settles onto z=0.
        world.robot.position[2] = .4
        world.session.camera_cache['last_physics_pose'] = world.robot.position.tolist()
        if world.steps_this_placement == 1:
            if world.disturbance == 'background':
                world.background.position[0] += .10
            elif world.disturbance == 'relative_group':
                world.companion.position[0] += .10

    def render(self, *args, **kwargs):
        raise AssertionError('A renderer must never be called by CPU fixtures.')


class World:
    def __init__(self, *, small_table=False, robot_contact='valid', support_contact='valid', disturbance=None):
        self.robot_contact, self.support_contact, self.disturbance = robot_contact, support_contact, disturbance
        self.placements = self.steps_this_placement = 0
        self.cache_at_placement = []
        self.cache_invalidations = self.pose_invalidations = 0
        self.forbidden_calls = []
        self.target = Object(self, 'target', 'apple', [0., 0., .10], [.12, .12, .20])
        self.companion = Object(self, 'companion', 'mug', [.30, 0., .10], [.12, .12, .20])
        self.table = Object(self, 'table', 'coffee_table', [1.50, 0., .35],
                            [.15, .15, .70] if small_table else [1.40, 1.20, .70], fixed=True)
        self.floor = Object(self, 'floor', 'floors', [0., 0., -.10], [8., 8., .20], fixed=True)
        self.chair = Object(self, 'chair', 'chair', [-2., 2., .50], [.5, .5, 1.], fixed=True)
        self.background = Object(self, 'background', 'book', [-1., 1., .20], [.2, .2, .4])
        self.robot = Object(self, 'robot', 'robot', [0., -1., .40], [.6, .6, 1.])
        self.robot.floor_touching_base_link_names = ['left_wheel', 'right_wheel']
        self.robot.links = {'left_wheel': Wheel(self.robot, -.2), 'right_wheel': Wheel(self.robot, .2)}
        self.robot.arm_names = ['left']
        self.robot.controllers = {'gripper_left': SimpleNamespace(dof_idx=np.array([0, 1]))}
        self.robot.control_limits = {'position': (np.zeros(2), np.array([.05, .05]))}
        self.objects = [self.target, self.companion, self.table, self.floor, self.chair, self.background, self.robot]
        scene = SimpleNamespace(objects=self.objects, object_registry=lambda field, name:
                                next((obj for obj in self.objects if obj.name == name), None))
        self.sim = Simulation(self)
        self.session = SimpleNamespace(room_name='room_0', env=SimpleNamespace(scene=scene, robots=[self.robot]),
            pipeline=SimpleNamespace(og=SimpleNamespace(sim=self.sim)), camera_cache={'initial': 'cached'},
            scene_camera_position=[4., -3., 1.2], postprocess={'viewer_camera': {'position': [3., -2., 1.3]}},
            invalidate_camera_cache=self.invalidate_camera_cache, render=self.forbidden,
            model_call=self.forbidden, execute_action=self.forbidden)
        self.bindings = {'target': SimpleNamespace(name='target'), 'companion': SimpleNamespace(name='companion')}
        self.policy = {'mode': 'manipulation', 'group_bindings': ['target', 'companion'],
            'manipulation_bindings': ['target'], 'relocation_policy': 'rigid_group', 'source_region': 'room_0',
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85}
        self.original = self.sim.dump_state()
        self.sim.dump_count = 0
        self.original_camera = deepcopy(self.session.scene_camera_position)
        self.original_viewer = deepcopy(self.session.postprocess['viewer_camera']['position'])
        self.models = {obj.name: obj.model for obj in self.objects}
        self.extents = {obj.name: obj.extent.copy() for obj in self.objects}

    def forbidden(self, *args, **kwargs):
        self.forbidden_calls.append((args, kwargs))
        raise AssertionError('No model, renderer or robot action command is allowed.')

    def invalidate_camera_cache(self):
        self.cache_invalidations += 1
        self.session.camera_cache = {}

    def invalidate_pose_cache(self):
        self.pose_invalidations += 1

    def on_table(self, obj):
        low, high = obj.aabb
        table_low, table_high = self.table.aabb
        return bool(np.all(low[:2] >= table_low[:2]) and np.all(high[:2] <= table_high[:2]) and
                    abs(low[2] - table_high[2]) <= .01)

    def robot_contacts(self):
        mode = self.robot_contact
        if mode == 'missing_floor':
            return []
        contacts = [SimpleNamespace(body0='/World/robot/left_wheel', body1='/World/floor/base_link',
                                   normal=[0., 0., 1.], position=[*self.robot.position[:2], 0.])]
        if mode == 'chair' or (mode == 'first_chair' and self.placements == 1):
            contacts.append(SimpleNamespace(body0='/World/robot/body', body1='/World/chair/base_link',
                                            normal=[1., 0., 0.], position=[0., 0., .5]))
        return contacts

    def raytest(self, start, end, only_closest=True):
        low, high = self.table.aabb
        inside = np.all(np.array(start[:2]) >= low[:2]) and np.all(np.array(start[:2]) <= high[:2])
        hit = {'hit': bool(inside), 'rigidBody': '/World/table/top', 'normal': [0., 0., 1.],
               'position': [*start[:2], high[2]]}
        return hit if only_closest else [hit]

    def run(self):
        return setup.initialize_interaction(self.session, self.bindings, self.policy)

    def assert_restored(self):
        for obj in self.objects:
            assert np.array_equal(obj.position, self.original[obj.name][0]), obj.name
            assert np.array_equal(obj.quaternion, self.original[obj.name][1]), obj.name
            assert obj.model == self.models[obj.name]
            assert np.array_equal(obj.extent, self.extents[obj.name])
        assert self.session.scene_camera_position == self.original_camera
        assert self.session.postprocess['viewer_camera']['position'] == self.original_viewer
        assert self.forbidden_calls == []


@pytest.fixture
def world_factory(monkeypatch):
    def make(**kwargs):
        world = World(**kwargs)
        touching, on_top = type('Touching', (), {}), type('OnTop', (), {})
        states = ModuleType('omnigibson.object_states')
        states.Touching, states.OnTop = touching, on_top
        for obj in (world.target, world.companion):
            obj.states[touching] = SimpleNamespace(get_value=lambda other, obj=obj:
                bool(other is world.table and world.on_table(obj) and world.support_contact != 'missing'))
            obj.states[on_top] = SimpleNamespace(get_value=lambda other, obj=obj:
                bool(other is world.table and world.on_table(obj)))
        sampling = ModuleType('omnigibson.utils.sampling_utils')
        sampling.raytest = world.raytest
        usd = ModuleType('omnigibson.utils.usd_utils')
        usd.PoseAPI = SimpleNamespace(invalidate=world.invalidate_pose_cache)
        helpers = ModuleType('wia.tasks.support_sampling')
        helpers.same_body = lambda root, body: body == root or body.startswith(root + '/')
        helpers.support_edge_points = (lambda low, high, margin, stances, initial, maximum_distance:
            (np.array([[(low[0] + high[0]) / 2, (low[1] + high[1]) / 2]]), {}))
        helpers.upward_layers = lambda hits, path, minimum, maximum: sorted({float(hit['position'][2])
            for hit in hits if hit['hit'] and helpers.same_body(path, hit['rigidBody']) and
            hit['normal'][2] > .95 and minimum <= hit['position'][2] <= maximum})
        torch = ModuleType('torch')
        torch.float32 = np.float32
        torch.tensor = lambda values, dtype=None: np.array(values, dtype=dtype)
        for name, module in {'torch': torch, 'omnigibson': ModuleType('omnigibson'),
                'omnigibson.object_states': states, 'omnigibson.utils': ModuleType('omnigibson.utils'),
                'omnigibson.utils.sampling_utils': sampling, 'omnigibson.utils.usd_utils': usd,
                'wia.tasks.support_sampling': helpers}.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(setup, '_navigation_cells', lambda session, robot, region, excluded:
            (np.array([[1.50, -.65], [1.60, -.65]]), 0., {'requested_source_region': region,
                'candidate_cell_count': 2, 'floor_z_m': 0., 'measurement_fixture': True}))
        return world
    return make


@pytest.mark.parametrize('search', ['bounded_group', 'measured_surfaces'])
def test_common_group_translation_uses_existing_support_and_wheel_floor_datum(world_factory, search):
    world = world_factory()
    world.policy['support_search'] = search
    original_relative = world.companion.position - world.target.position
    evidence = world.run()
    assert evidence['accepted'] and evidence['relocated']
    assert evidence['support_name'] == 'table'
    delta = np.array(evidence['group_translation_m'])
    assert delta == pytest.approx([1.35, 0., .705])
    for obj in (world.target, world.companion):
        assert obj.position == pytest.approx(world.original[obj.name][0] + delta)
        assert np.array_equal(obj.quaternion, world.original[obj.name][1])
        assert np.array_equal(obj.extent, world.extents[obj.name])
        assert obj.model == world.models[obj.name]
    assert world.companion.position - world.target.position == pytest.approx(original_relative)
    assert world.robot.position[2] == pytest.approx(.4)
    assert world.robot.links['left_wheel'].aabb[0][2] == pytest.approx(0.)
    assert evidence['robot_spawn']['checks']['floor_supported']
    assert evidence['robot_spawn']['checks']['collision_free']
    assert world.session.scene_camera_position == pytest.approx(np.array(world.original_camera) + delta)
    assert world.session.postprocess['viewer_camera']['position'] == pytest.approx(np.array(world.original_viewer) + delta)
    assert world.session.camera_cache == {}
    assert not evidence['physical_grasp_verified'] and not evidence['ik_verified']
    assert world.forbidden_calls == []


def test_no_fitting_support_rejects_and_restores_exact_world(world_factory):
    world = world_factory(small_table=True)
    evidence = world.run()
    assert not evidence['accepted'] and not evidence['relocated']
    assert 'No existing same-room support fits' in evidence['reason']
    assert world.sim.step_count == 0
    world.assert_restored()
    assert world.session.camera_cache == {}
    assert world.pose_invalidations == world.sim.load_count > 0


def test_rejected_layout_keeps_a_repaired_safe_robot_without_moving_source_objects(world_factory):
    world = world_factory(small_table=True)
    world.robot.position[2] = 50.
    evidence = world.run()
    assert not evidence['accepted'] and not evidence['relocated']
    assert evidence['robot_spawn_repaired'] is True
    assert evidence['robot_spawn']['task_layout_qualified'] is False
    assert evidence['robot_spawn']['checks']['original_nonrobot_geometry_preserved']
    assert world.robot.position[2] == pytest.approx(.4)
    for obj in world.objects:
        if obj is not world.robot:
            assert np.array_equal(obj.position, world.original[obj.name][0])


@pytest.mark.parametrize('contact_mode', ['chair', 'missing_floor'])
def test_chair_contact_or_missing_floor_contact_rejects_and_restores(world_factory, contact_mode):
    world = world_factory(robot_contact=contact_mode)
    evidence = world.run()
    assert not evidence['accepted'] and not evidence['relocated']
    assert len(evidence['attempts']) == 2
    if contact_mode == 'chair':
        assert all(not attempt['checks']['collision_free'] for attempt in evidence['attempts'])
    else:
        assert all(not attempt['checks']['floor_supported'] for attempt in evidence['attempts'])
    world.assert_restored()
    assert world.session.camera_cache == {}
    assert world.pose_invalidations == world.sim.load_count


@pytest.mark.parametrize('support_mode', ['missing', 'side', 'wrong_height'])
def test_aabb_fit_without_current_upward_surface_contact_cannot_pass(world_factory, support_mode):
    world = world_factory(support_contact=support_mode)
    evidence = world.run()
    assert not evidence['accepted'] and not evidence['relocated']
    assert evidence['attempts']
    assert all(attempt['checks']['collision_free'] and attempt['checks']['floor_supported']
               for attempt in evidence['attempts'])
    assert all(next(iter(attempt['height_screen'].values()))['within_height_window']
               for attempt in evidence['attempts'])
    world.assert_restored()


def test_preserve_policy_with_invalid_height_rejects_before_geometry_edits(world_factory):
    world = world_factory()
    world.policy['relocation_policy'] = 'preserve'
    evidence = world.run()
    assert not evidence['accepted'] and 'height is outside' in evidence['reason']
    assert world.sim.dump_count == 1 and world.sim.load_count == 1 and world.sim.step_count == 0
    assert all(obj.setter_calls == 0 for obj in world.objects)
    assert world.session.camera_cache == {}
    world.assert_restored()


@pytest.mark.parametrize('disturbance', ['background', 'relative_group'])
def test_settling_cannot_move_background_or_change_relative_group_geometry(world_factory, disturbance):
    world = world_factory(disturbance=disturbance)
    evidence = world.run()
    assert not evidence['accepted']
    check = 'background_geometry_preserved' if disturbance == 'background' else 'task_group_geometry_preserved'
    assert all(attempt['checks']['collision_free'] and attempt['checks']['floor_supported']
               and not attempt['checks'][check] for attempt in evidence['attempts'])
    world.assert_restored()


def test_attempt_restoration_invalidates_pose_and_camera_state_before_retry(world_factory):
    world = world_factory(robot_contact='first_chair')
    evidence = world.run()
    assert evidence['accepted'] and len(evidence['attempts']) == 2
    assert not evidence['attempts'][0]['checks']['collision_free']
    assert evidence['attempts'][1]['checks']['collision_free']
    assert world.placements == 2
    assert world.cache_at_placement == [{}, {}]
    assert world.session.camera_cache == {}
    assert world.pose_invalidations == world.sim.load_count
    delta = np.array(evidence['group_translation_m'])
    assert world.session.scene_camera_position == pytest.approx(np.array(world.original_camera) + delta)
    assert world.session.postprocess['viewer_camera']['position'] == pytest.approx(np.array(world.original_viewer) + delta)
    assert world.background.position == pytest.approx(world.original['background'][0])
    assert world.forbidden_calls == []


def test_comparison_pairs_fit_table_without_preserving_interpair_floor_spacing(world_factory):
    world = world_factory()
    second = Object(world, 'second_target', 'apple', [-1.4, 0., .10], [.12, .12, .20])
    reference = Object(world, 'second_reference', 'mug', [-1.1, 0., .10], [.12, .12, .20])
    world.objects.extend([second, reference])
    for obj in (second, reference):
        obj.states = {key: SimpleNamespace(get_value=lambda other, obj=obj: world.on_table(obj))
                      for key in world.target.states}
    world.bindings.update({obj.name: SimpleNamespace(name=obj.name) for obj in (second, reference)})
    world.policy.update(relocation_policy='comparison_pairs',
        group_bindings=['target', 'companion', 'second_target', 'second_reference'],
        manipulation_bindings=['target', 'second_target'],
        placement_groups=[['target', 'companion'], ['second_target', 'second_reference']])
    evidence = world.run()
    assert evidence['accepted'], evidence
    assert evidence['relocated'] and evidence['group_translation_m'] is None
    assert all(world.on_table(obj) for obj in (world.target, world.companion, second, reference))
    delta = evidence['object_translations_m']
    assert delta['target'] == delta['companion']
    assert delta['second_target'] == delta['second_reference']
    assert delta['target'] != delta['second_target']
    checks = evidence['robot_spawn']['checks']
    assert checks['comparison_pair_geometry_preserved'] and checks['comparison_reference_association_preserved']
    assert checks['floor_supported'] and checks['collision_free']
    assert all(evidence['robot_spawn']['per_target_candidate_stances'].values())
    assert not evidence['physical_grasp_verified']
    assert 'comparison_pair_layout_adapted_requires_review' in evidence['warning_codes']
    for obj in (world.target, world.companion, second, reference):
        expected = world.original.get(obj.name, (np.array([-1.4 if obj is second else -1.1, 0., .10]),))[0]
        assert setup.reference_position(world.session, obj.name, expected) == pytest.approx(obj.position)
    assert world.forbidden_calls == []


def test_navigation_formation_moves_to_table_and_scales_only_root_spacing(world_factory):
    world = world_factory()
    world.target.position[0], world.companion.position[0] = -1., 1.
    relative = world.companion.position - world.target.position
    extents = [obj.extent.copy() for obj in (world.target, world.companion)]
    world.policy.update(mode='navigation', manipulation_bindings=[], relocation_policy='relation_group')
    evidence = world.run()
    assert evidence['accepted'], evidence
    assert evidence['relocated'] and 0 < evidence['layout_scale_xy'] < 1
    assert world.on_table(world.target) and world.on_table(world.companion)
    assert world.companion.position - world.target.position == pytest.approx(
        relative * [evidence['layout_scale_xy'], evidence['layout_scale_xy'], 1.])
    assert all(np.array_equal(obj.extent, extent) for obj, extent in zip((world.target, world.companion), extents))
    assert all(row['within_height_window'] for row in evidence['height_screen_after'].values())
    checks = evidence['robot_spawn']['checks']
    assert checks['relation_root_similarity_preserved'] and checks['native_task_support_contact']
    assert checks['floor_supported'] and checks['collision_free']
    assert 'layout_uniform_position_scale_requires_review' in evidence['warning_codes']
    assert world.forbidden_calls == []


def test_each_lazy_surface_proposal_reads_restored_source_geometry(world_factory, monkeypatch):
    world = world_factory(robot_contact='chair')
    world.policy['support_search'] = 'measured_surfaces'
    raytest = world.raytest
    def baseline_surface_query(*args, **kwargs):
        assert world.target.position == pytest.approx(world.original['target'][0])
        assert world.companion.position == pytest.approx(world.original['companion'][0])
        return raytest(*args, **kwargs)
    monkeypatch.setattr(sys.modules['omnigibson.utils.sampling_utils'], 'raytest', baseline_surface_query)
    evidence = world.run()
    assert not evidence['accepted'] and len(evidence['attempts']) > 2
    assert all(not attempt['checks']['collision_free'] for attempt in evidence['attempts'] if 'checks' in attempt)
    world.assert_restored()


def test_table_placement_may_require_later_base_approach_without_claiming_reach(world_factory):
    world = world_factory()
    world.policy.update(support_search='measured_surfaces', maximum_stance_distance_m=.01)
    evidence = world.run()
    assert evidence['accepted'] and evidence['relocated']
    assert world.on_table(world.target) and world.on_table(world.companion)
    assert 'initial_base_approach_required' in evidence['warning_codes']
    screen = evidence['robot_spawn']['per_target_stance_screen']['target']
    assert not screen['within_preferred_radius'] and not screen['arm_reach_verified']
    assert evidence['robot_spawn']['checks']['collision_free']
    assert not evidence['physical_grasp_verified'] and not evidence['ik_verified']

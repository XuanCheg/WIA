"""Exercise measured Search start initialization without a GPU or scene writes.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_search_start.py
Fixtures use the actual local-floor contact reader and the existing placement
helper against a small native-style physics adapter; no service is started.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.tasks import search_start as start
from wia.tasks import native_floor_contacts as contacts


class Actor:
    def __init__(self, name, category, position, extent, *, fixed=True, visual=False):
        self.name, self.category, self.model = name, category, 'native_' + category
        self.prim_path = '/World/' + name
        self.position, self.orientation = np.array(position, dtype=float), np.array([0., 0., 0., 1.])
        self.extent, self.scale = np.array(extent, dtype=float), np.ones(3)
        self.fixed_base, self.visual_only, self.collision_enabled = fixed, visual, True
        self.joints = {}

    def get_position_orientation(self, **kwargs):
        return self.position.copy(), self.orientation.copy()

    @property
    def aabb(self):
        return self.position - self.extent / 2, self.position + self.extent / 2


class Wheel:
    def __init__(self, robot, name, offset):
        self.robot, self.prim_path, self.offset = robot, robot.prim_path + '/' + name, np.array(offset)

    @property
    def aabb(self):
        centre = self.robot.position + start._rotation(self.robot.orientation) @ self.offset
        return centre - [.06, .035, .05], centre + [.06, .035, .05]


class NoOpController:
    motor_type = 'position'
    use_delta_commands = False
    command_input_limits = None

    def __init__(self, action):
        self.action = np.array(action)

    def compute_no_op_action(self, control):
        return self.action.copy()


class Robot(Actor):
    def __init__(self, fixture):
        super().__init__('robot', 'agent', [0., 0., .1], [.6, .6, 1.], fixed=False)
        self.model, self.fixture = 'R1', fixture
        self.floor_touching_base_link_names = ['wheel_left', 'wheel_right']
        self.links = {name: Wheel(self, name, offset) for name, offset in
            [('wheel_left', [0., .22, -.45]), ('wheel_right', [0., -.22, -.45])]}
        self.sensors = {'head': NS(get_position_orientation=lambda **kwargs: (np.zeros(3), np.array([0., 0., 0., 1.])))}
        self.controllers = {'base': NoOpController([0., 0., 0.]), 'arm': NoOpController([.7])}
        self.controller_action_idx = {'base': np.array([0, 1, 2]), 'arm': np.array([3])}
        self.action_dim, self.pose_sets, self.keep_still_calls = 4, 0, 0

    def get_joint_positions(self):
        return np.array([.7])

    def get_control_dict(self):
        return {'joint_position': self.get_joint_positions()}

    def set_position_orientation(self, position=None, orientation=None):
        if position is not None:
            self.position = start.array(position).copy()
        if orientation is not None:
            self.orientation = start.array(orientation).copy()
        self.pose_sets += 1

    def keep_still(self):
        self.keep_still_calls += 1

    def contact_list(self):
        fixture = self.fixture
        if self.position[0] <= 1. or abs(self.position[2] - .5 - fixture.floor_z) > .001:
            return []
        rows = [NS(body0=wheel.prim_path, body1='/World/floor/collision',
            position=np.array([self.position[0], self.position[1], fixture.floor_z]), normal=np.array([0., 0., 1.]))
            for wheel in self.links.values()]
        if fixture.foreign_contact:
            rows.append(NS(body0=self.links['wheel_left'].prim_path, body1='/World/wall/collision',
                position=np.array([*self.position[:2], fixture.floor_z]), normal=np.array([1., 0., 0.])))
        return rows


class Session(NS):
    @property
    def observation_source(self):
        raise AssertionError('Search initialization must not invoke the lazy observation rig property.')


def bind_runtime(session, original, runtime=None):
    """Install the exact receipt/binding consumed by the real layout resolver."""
    runtime = deepcopy(original if runtime is None else runtime)
    context = {'original_world_digest': start.digest(original), 'reset_id': session.native_reset_id}
    session._search_world_contract = runtime
    session._search_world_contract_binding = {**context, 'runtime_layout_digest': start.digest(runtime)}
    session._search_table_layout_initialization = {'context': context, 'accepted': True,
        'runtime_layout_digest': start.digest(runtime), 'groups': [], 'background_modified': False}


@pytest.fixture
def native(tmp_path, monkeypatch):
    fixture = NS(floor_z=.0055916, foreign_contact=False, ray_missing=False,
        ray_foreign=False, hold_drift=0., sim_callback=None, actions=[], rays=[], navigation_calls=[])
    robot = Robot(fixture)
    floor = Actor('floor', 'floors', [3., 2.5, fixture.floor_z - .05], [4., 5., .1])
    wall = Actor('wall', 'walls', [5., 2., 1.], [.1, 4., 2.])
    table = Actor('table', 'breakfast_table', [3., 2., .73], [.8, 1., .05])
    source_actor = Actor('source_prop', 'ball', [3., 2., .9], [.1, .1, .1], visual=True)
    dynamic = Actor('dynamic_native', 'chair', [4., 4., .5], [.4, .4, 1.], fixed=False)
    objects = {obj.name: obj for obj in [robot, floor, wall, table, source_actor, dynamic]}
    identity = 'Enumerative Perception/Category Ambiguity/building/room_first/q.json'
    source_path = tmp_path / 'sources' / identity
    source_path.parent.mkdir(parents=True)
    source_path.write_text('{"scene":"building","room":"room_first"}')
    world = {'scene_model': 'building', 'required_rooms': ['room_first', 'room_other'],
        'joint_inventory': [{'source_identity': identity, 'native': {'name': source_actor.name,
            'category': source_actor.category, 'model': source_actor.model}}],
        'source_provenance': [{'source_identity': identity,
            'source_sha256': __import__('hashlib').sha256(source_path.read_bytes()).hexdigest(),
            'source_payload': {'room': 'room_first'}}],
        'search_groups': [{'id': 'group_1', 'source_identity': identity, 'room': 'Public first room'}],
        'curated_layout': {'tables': [{'name': table.name, 'room_instance': 'room_first', 'centre_xy_m': [3., 2.]}]}}
    task = NS(definition=NS(id='search_task', source=NS(json_path=identity), private=NS(evaluation=NS(
        module='wia.authoring.converters.enumerative_perception_search', parameters={'world_contract': deepcopy(world)}))),
        verify_unchanged=lambda: None, source_path=source_path, file_sha256='task_hash',
        source_sha256='source_hash', seed=7)
    room_query = lambda xy: 'room_first' if float(xy[0]) > 1. else 'room_other'
    scene = NS(objects=objects, scene_model='building', object_registry=lambda key, name: objects.get(name),
        seg_map=NS(get_room_instance_by_point=room_query))
    sim = NS(current_time=0.)
    def step():
        sim.current_time += .02
        if robot.position[0] > 1.:
            robot.position[2] = fixture.floor_z + .5
        if fixture.sim_callback:
            fixture.sim_callback()
    sim.step = step
    def hold(actions):
        fixture.actions.append(deepcopy(actions))
        robot.position[0] += fixture.hold_drift
        step()
    session = Session(robot=robot, env=NS(scene=scene, robots=[robot], step=hold),
        pipeline=NS(og=NS(sim=sim)), scene_name='building', loaded_task=task,
        initial_overview_identity={'episode_id': 'pre_episode', 'reset_id': 'native_reset', 'seed': 7},
        native_reset_id='native_reset', _observation_source=None,
        category_interaction_setup={'sentinel': 'must not be modified'},
        invalidate_camera_cache=lambda: None)
    bind_runtime(session, world)
    initial = {}
    for obj in (source_actor, table):
        p, q = obj.get_position_orientation()
        initial[obj.name] = {'position': p.tolist(), 'orientation': q.tolist(),
            'scale': obj.scale.tolist(), 'category': obj.category, 'model': obj.model,
            'fixed_base': obj.fixed_base, 'visual_only': obj.visual_only}
    session._search_initial_native = initial
    def navigation(session_arg, robot_arg, room, excluded):
        fixture.navigation_calls.append((room, excluded))
        return np.array(fixture.cells), 0., {'candidate_cell_count': len(fixture.cells), 'requested_source_region': room}
    fixture.cells = [[1.7, 2.5], [2., 2.]]
    def ray(begin, end, *, only_closest=False):
        fixture.rays.append((list(begin), list(end)))
        if fixture.ray_missing or begin[0] <= 1.:
            return []
        result = [{'hit': True, 'rigidBody': '/World/floor/collision',
                   'position': [begin[0], begin[1], fixture.floor_z], 'normal': [0., 0., 1.]}]
        if fixture.ray_foreign:
            result.append({'hit': True, 'rigidBody': '/World/obstruction/collision',
                           'position': [begin[0], begin[1], fixture.floor_z + .02], 'normal': [0., 0., 1.]})
        return result
    monkeypatch.setattr(start, '_navigation_cells', navigation)
    monkeypatch.setattr(contacts, 'native_vertical_ray', ray)
    fixture.session, fixture.world, fixture.robot = session, world, robot
    fixture.floor, fixture.wall, fixture.source_actor, fixture.dynamic = floor, wall, source_actor, dynamic
    return fixture


def test_successful_measured_same_room_start_uses_local_floor_and_complete_pilot_hold(native):
    row = start.initialize_search_start(native.session, native.world)
    assert row is native.session.search_start_initialization and row['accepted'] is True
    assert native.navigation_calls == [('room_first', set())]
    assert row['attempts'][0]['proposed_xy_m'] == [2., 2.]
    attempt = row['attempts'][0]
    rays = attempt['wheel_floor_rays']
    assert rays['measured_floor_z_m'] == pytest.approx(native.floor_z)
    assert rays['nominal_floor_offset_m'] == pytest.approx(.0055916)
    assert len(rays['samples']) == 10 and len(row['wheel_footprints']) == 2
    assert attempt['stability']['completed_hold_steps'] == 30
    assert attempt['stability']['simulated_duration_s'] == pytest.approx(.6)
    assert len(attempt['hold_samples']) == 31
    assert all(sample['floor_contact_count'] == 2 and sample['contacts']['floor_supported']
               for sample in attempt['hold_samples'])
    assert len(native.actions) == 30
    assert all(np.allclose(action['robot'], [0., 0., 0., .7]) for action in native.actions)
    assert row['invariants']['preserved']
    assert row['policy']['provenance'] == 'task_edit2_search_native_start_pilot_20261018_v1_root_selected'
    assert row['route_or_manipulation_certificate'] is False
    assert row['provenance']['original_world_digest'] == start.digest(native.world)
    assert row['provenance']['runtime_layout_digest'] == start.digest(native.world)
    assert native.session.category_interaction_setup == {'sentinel': 'must not be modified'}
    assert not hasattr(native.session, 'search_world_qualification')
    assert {'loaded_routes', 'mounted_camera_audit', 'semantic_access'} <= set(row['pending_gates'])


@pytest.mark.parametrize('failure', ['no_floor', 'hole', 'foreign_ray', 'foreign_contact'])
def test_no_native_floor_holes_foreign_shape_or_contact_never_accept(native, failure):
    if failure == 'no_floor':
        native.session.env.scene.objects.pop('floor')
    elif failure == 'hole':
        native.ray_missing = True
    elif failure == 'foreign_ray':
        native.ray_foreign = True
    else:
        native.foreign_contact = True
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and row['reason']
    if failure in ('no_floor', 'hole', 'foreign_ray'):
        assert row['started_placements'] == 0 and native.robot.pose_sets == 0
    else:
        assert all(attempt['hold_samples'][0]['contacts']['collision_free'] is False for attempt in row['attempts'])
        assert not native.actions


def test_cap_is_explicit_and_no_other_room_fallback_or_infeasibility_claim(native):
    native.ray_missing = True
    native.cells = [[1.5 + index * .02, 2.] for index in range(40)]
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and len(row['attempts']) == 32
    assert row['search_coverage']['same_room_candidate_count'] == 40
    assert row['search_coverage']['started_fraction'] == .8
    assert row['search_coverage']['other_room_fallback_used'] is False
    assert native.navigation_calls == [('room_first', set())]
    assert 'does not prove infeasibility' in row['reason']


def test_actual_source_room_is_enforced_even_if_navigation_adapter_returns_other_room(native, monkeypatch):
    native.cells = [[.5, 2.]]
    def floor_ray(begin, end, **kwargs):
        return [{'hit': True, 'rigidBody': '/World/floor/collision',
                 'position': [begin[0], begin[1], native.floor_z], 'normal': [0., 0., 1.]}]
    monkeypatch.setattr(contacts, 'native_vertical_ray', floor_ray)
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and row['started_placements'] == 0
    assert row['selection']['room'] == 'room_first'
    assert row['attempts'][0]['reason'] == 'wheel_floor_rays_or_exact_source_room_rejected'


@pytest.mark.parametrize('activity', ['observation', 'action', 'public_sequence'])
def test_after_observation_or_public_control_refuses_without_native_changes(native, activity):
    if activity == 'observation':
        native.session._observation_source = NS(initial_observation=object(), action_step=0)
    elif activity == 'action':
        native.session._observation_source = NS(action_step=1)
    else:
        native.session.controller = NS(_public_action_sequence=1)
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and 'forbidden' in row['reason']
    assert native.robot.pose_sets == 0 and not native.navigation_calls


@pytest.mark.parametrize('stale', ['contract', 'reset_seed', 'source_actor'])
def test_stale_world_or_source_actor_refuses(native, stale):
    if stale == 'contract':
        native.session._search_world_contract['required_rooms'] = ['unrelated']
    elif stale == 'reset_seed':
        native.session.initial_overview_identity['seed'] = 8
    else:
        native.source_actor.position[0] += .1
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and native.robot.pose_sets == 0


@pytest.mark.parametrize('target', ['source_actor', 'wall'])
def test_protected_pose_change_aborts_and_is_not_reversed_or_hidden(native, target):
    obj = getattr(native, target)
    original = obj.position.copy()
    changed = False
    def move():
        nonlocal changed
        if not changed:
            obj.position[0] += .01
            changed = True
    native.sim_callback = move
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and row['invariants']['preserved'] is False
    assert any(obj.name in reason for reason in row['invariants']['violations'])
    assert obj.position[0] == pytest.approx(original[0] + .01)
    assert row['source_actor_or_background_state_restore_used'] is False


def test_dynamic_background_settling_is_reported_without_restore(native):
    original = native.dynamic.position.copy()
    native.sim_callback = lambda: native.dynamic.position.__setitem__(0, native.dynamic.position[0] + .0001)
    row = start.initialize_search_start(native.session, native.world)
    assert row['accepted'] and row['invariants']['preserved']
    assert native.dynamic.position[0] > original[0]
    settling = row['invariants']['native_dynamic_settling'][native.dynamic.name]
    assert settling['restored_or_hidden'] is False


def test_passive_drift_rejects_and_safe_original_robot_pose_only_can_be_restored(native):
    native.robot.position = np.array([2., 1.5, native.floor_z + .5])
    original = native.robot.position.copy()
    native.hold_drift = .002
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted']
    assert row['attempts'][0]['stability']['maximum_translation_m'] > .01
    assert row['restoration']['performed'] is True
    assert np.array_equal(native.robot.position, original)
    assert row['restoration']['post_restore_qualification'].startswith('unavailable')


def test_world_reset_change_during_initialization_is_explicit_failure(native):
    native.sim_callback = lambda: native.session.initial_overview_identity.update(reset_id='changed')
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and 'reset' in row['reason']
    assert row['restoration']['performed'] is False


def test_private_relocated_runtime_uses_actual_table_and_keeps_original_evaluator_immutable(native):
    original = deepcopy(native.world)
    runtime = deepcopy(original)
    runtime['curated_layout']['tables'][0]['centre_xy_m'] = [4., 2.]
    runtime['curated_layout']['runtime_layout_revision'] = 'native_measured_translation'
    table = native.session.env.scene.object_registry('name', 'table')
    table.position[0] += 1.
    native.source_actor.position[0] += 1.
    runtime['joint_inventory'][0]['native']['position'] = native.source_actor.position.tolist()
    for obj in (table, native.source_actor):
        native.session._search_initial_native[obj.name]['position'] = obj.position.tolist()
    bind_runtime(native.session, original, runtime)
    native.cells = [[2., 2.], [3.5, 2.]]
    row = start.initialize_search_start(native.session, original)
    assert row['accepted']
    assert row['selection']['target_centre_m'][:2] == [4., 2.]
    assert row['attempts'][0]['proposed_xy_m'] == [3.5, 2.]
    assert row['provenance']['original_world_digest'] == start.digest(original)
    assert row['provenance']['runtime_layout_digest'] == start.digest(runtime)
    assert row['provenance']['original_world_digest'] != row['provenance']['runtime_layout_digest']
    assert original == native.world
    assert native.session.loaded_task.definition.private.evaluation.parameters['world_contract'] == original


@pytest.mark.parametrize('stale', ['passed_original', 'loaded_original', 'runtime', 'runtime_digest',
    'reset', 'receipt_reset', 'receipt_missing', 'binding_missing', 'both_missing', 'receipt_rejected'])
def test_exact_original_runtime_and_receipt_binding_are_mandatory(native, stale):
    original = deepcopy(native.world)
    if stale == 'passed_original':
        original['required_rooms'].append('unrelated')
    elif stale == 'loaded_original':
        native.session.loaded_task.definition.private.evaluation.parameters['world_contract']['required_rooms'].append('unrelated')
    elif stale == 'runtime':
        native.session._search_world_contract['curated_layout']['tables'][0]['centre_xy_m'][0] += 1.
    elif stale == 'runtime_digest':
        native.session._search_world_contract_binding['runtime_layout_digest'] = 'stale'
    elif stale == 'reset':
        native.session.native_reset_id = 'new_reset'
    elif stale == 'receipt_reset':
        native.session._search_table_layout_initialization['context']['reset_id'] = 'stale'
    elif stale == 'receipt_missing':
        del native.session._search_table_layout_initialization
    elif stale == 'binding_missing':
        del native.session._search_world_contract_binding
    elif stale == 'both_missing':
        del native.session._search_world_contract
        del native.session._search_world_contract_binding
    else:
        native.session._search_table_layout_initialization['accepted'] = False
    row = start.initialize_search_start(native.session, original)
    assert not row['accepted'] and row['reason']
    assert native.robot.pose_sets == 0 and not native.navigation_calls


@pytest.mark.parametrize('changed', ['runtime', 'reset', 'receipt'])
def test_layout_binding_is_revalidated_during_native_hold(native, changed):
    def mutate():
        if changed == 'runtime':
            native.session._search_world_contract['curated_layout']['tables'][0]['centre_xy_m'][0] += .001
        elif changed == 'reset':
            native.session.native_reset_id = 'new_reset'
        else:
            native.session._search_table_layout_initialization['accepted'] = False
    native.sim_callback = mutate
    row = start.initialize_search_start(native.session, native.world)
    assert not row['accepted'] and row['reason']
    assert row['started_placements'] == 1 and row['restoration']['performed'] is False

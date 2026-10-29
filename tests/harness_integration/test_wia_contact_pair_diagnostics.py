"""Check pair-local native contact reads with CPU-only public-getter mocks.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_contact_pair_diagnostics.py
No native view, simulator, GPU workload or actor state is initialized or changed.
"""
from copy import deepcopy
import json
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.runtime.sim.wia_contact_diagnostics import measure_contact_pair


class Native:
    def __init__(self, name, path):
        self.name, self.prim_path = name, path
        self.sleep_threshold, self.is_asleep = .001, False

    @property
    def aabb(self):
        raise AssertionError('Contact diagnostics must not infer contact from AABB overlap.')

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('Contact diagnostics must not set native state.')

    def wake(self):
        raise AssertionError('Contact diagnostics must not wake native actors.')


class ContactAPI:
    def __init__(self, first, second):
        a, b = first.prim_path, second.prim_path
        self.scenes = {a: 7, b: 7}
        self.rows, self.columns = {a: 1, b: 2}, {a: 2, b: 0}
        self.impulses = {(a, b): np.array([3., 4., 0.]), (b, a): np.array([0., 0., 2.])}
        self.contacts = {(a, b): [self.sample(a, b, -.003), self.sample(a, b, -.001)],
                         (b, a): [self.sample(b, a, .0004)]}
        counts, starts = np.zeros((3, 3), int), np.zeros((3, 3), int)
        counts[1, 0], starts[1, 0] = 2, 1
        counts[2, 2], starts[2, 2] = 1, 5
        self._CONTACT_DATA = {7: (np.arange(8.).reshape(8, 1), np.zeros((8, 3)), np.ones((8, 3)),
            np.array([-99., -.003, -.001, 99., 99., .0004, 99., 99.]), counts, starts)}
        self.calls = []

    @staticmethod
    def sample(row, column, separation):
        return row, column, np.array([.5]), np.array([1., 2., 3.]), np.array([0., 0., 1.]), np.array([separation])

    def get_scene_idx(self, path):
        self.calls.append(('scene', path))
        return self.scenes[path]

    def get_body_row_idx(self, path):
        self.calls.append(('row', path))
        return self.scenes[path], self.rows[path]

    def get_body_col_idx(self, path):
        self.calls.append(('column', path))
        return self.scenes[path], self.columns[path]

    def get_impulses(self, rows, columns):
        assert len(rows) == len(columns) == 1
        self.calls.append(('impulses', tuple(rows), tuple(columns)))
        return self.impulses[rows[0], columns[0]].reshape(1, 1, 3)

    def get_contact_data(self, scene, row_prim_paths=None, column_prim_paths=None):
        assert scene == 7 and len(row_prim_paths) == len(column_prim_paths) == 1
        self.calls.append(('contact_data', tuple(row_prim_paths), tuple(column_prim_paths)))
        return self.contacts[row_prim_paths[0], column_prim_paths[0]]

    def get_max_contact_data_count(self):
        return 256

    def initialize_view(self, *args, **kwargs):
        raise AssertionError('Diagnostics must never initialize or recreate views.')

    def get_all_impulses(self, *args, **kwargs):
        raise AssertionError('The helper must request pair-local impulses through the public subset getter.')


@pytest.fixture
def native(monkeypatch):
    first, second = Native('target', '/World/target'), Native('container', '/World/container')
    a, b = Native('target_base', '/World/target/base_link'), Native('container_base', '/World/container/base_link')
    first.links, second.links = {'base_link': a}, {'base_link': b}
    api = ContactAPI(a, b)
    touching = object()
    first.states = {touching: NS(get_value=lambda other: other is second)}
    second.states = {touching: NS(get_value=lambda other: False)}
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.usd_utils', NS(RigidContactAPI=api))
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', NS(Touching=touching))
    return NS(first=first, second=second, a=a, b=b, api=api)


def pair(result, direction='first_to_second'):
    return result['directions'][direction]['link_pairs'][0]


def test_forward_reverse_paths_indices_impulses_samples_raw_pair_and_sleep_getters(native):
    before = deepcopy(native.api._CONTACT_DATA)
    result = measure_contact_pair(native.first, native.second)
    assert result['read_only'] is True and result['physics_steps'] == result['actor_mutations'] == 0
    assert result['first']['object_prim_path'] == '/World/target'
    assert result['second']['object_prim_path'] == '/World/container'
    forward, reverse = pair(result), pair(result, 'second_to_first')
    assert (forward['scene_idx'], forward['row_index'], forward['column_index']) == (7, 1, 0)
    assert (reverse['scene_idx'], reverse['row_index'], reverse['column_index']) == (7, 2, 2)
    assert forward['row_link_prim_path'] == reverse['column_link_prim_path'] == native.a.prim_path
    assert forward['column_link_prim_path'] == reverse['row_link_prim_path'] == native.b.prim_path
    assert forward['impulse']['norm_native'] == 5. and reverse['impulse']['norm_native'] == 2.
    assert forward['contact_data']['sample_count'] == 2
    assert forward['contact_data']['minimum_separation_m'] == -.003
    assert forward['contact_data']['separations_m'] == [-.003, -.001]
    assert reverse['contact_data']['separations_m'] == [.0004]
    assert result['directions']['first_to_second']['native_touching']['value'] is True
    assert result['directions']['second_to_first']['native_touching']['value'] is False
    raw = forward['raw_contact_data']
    assert raw['contact_counts'] == 2 and raw['start_indices'] == 1
    assert raw['configured_buffer_capacity']['value'] == 256
    assert raw['buffer_lengths'] == {'forces': 8, 'points': 8, 'normals': 8, 'separations': 8}
    assert raw['pair_separations_m'] == [-.003, -.001]
    assert reverse['raw_contact_data']['start_indices'] == 5
    assert -99. not in raw['pair_separations_m'] and 99. not in raw['pair_separations_m']
    for inventory in (result['first'], result['second']):
        for getter in (inventory['sleep_getters'], inventory['links'][0]['sleep_getters']):
            assert getter['sleep_threshold'] == {'available': True, 'native_getter': 'sleep_threshold', 'value': .001}
            assert getter['is_asleep'] == {'available': True, 'native_getter': 'is_asleep', 'value': False}
    for actual, expected in zip(native.api._CONTACT_DATA[7], before[7]):
        np.testing.assert_array_equal(actual, expected)
    contact_calls = [call for call in native.api.calls if call[0] == 'contact_data']
    assert contact_calls == [('contact_data', (native.a.prim_path,), (native.b.prim_path,)),
                             ('contact_data', (native.b.prim_path,), (native.a.prim_path,))]
    json.dumps(result, allow_nan=False)


def test_no_contact_samples_and_unpopulated_raw_cache_do_not_invent_separation(native):
    native.api._CONTACT_DATA.clear()
    native.api.contacts = {key: [] for key in native.api.contacts}
    native.api.impulses = {key: np.zeros(3) for key in native.api.impulses}
    result = measure_contact_pair(native.first, native.second)
    for direction in result['directions'].values():
        measured = direction['link_pairs'][0]
        assert measured['available'] is True
        assert measured['impulse']['norm_native'] == 0.
        assert measured['contact_data']['available'] is True and measured['contact_data']['sample_count'] == 0
        assert measured['contact_data']['minimum_separation_m'] is None
        assert measured['contact_data']['separations_m'] == []
        assert measured['raw_contact_data']['available'] is False
        assert measured['raw_contact_data']['reason'] == 'native_contact_data_cache_not_populated'
    assert result['directions']['first_to_second']['native_touching']['value'] is True
    assert native.api._CONTACT_DATA == {}
    json.dumps(result, allow_nan=False)


def test_positive_impulse_and_touching_do_not_fabricate_samples_when_raw_count_is_zero(native):
    native.api.contacts[native.a.prim_path, native.b.prim_path] = []
    native.api._CONTACT_DATA[7][4][1, 0] = 0
    measured = pair(measure_contact_pair(native.first, native.second))
    assert measured['impulse']['norm_native'] == 5.
    assert measured['contact_data']['sample_count'] == 0
    assert measured['contact_data']['minimum_separation_m'] is None
    assert measured['raw_contact_data']['contact_counts'] == 0
    assert measured['raw_contact_data']['pair_minimum_separation_m'] is None


def test_missing_forward_row_index_leaves_reverse_measurements_available(native):
    native.api.rows.pop(native.a.prim_path)
    result = measure_contact_pair(native.first, native.second)
    assert pair(result)['available'] is False
    assert pair(result)['contact_data']['reason'] == 'native_pair_indices_unavailable'
    assert pair(result)['scene_idx'] == 7 and pair(result)['column_index'] == 0
    assert pair(result)['row_index'] is None
    assert pair(result, 'second_to_first')['available'] is True
    assert 'KeyError' in result['first']['links'][0]['indices']['row']['error']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['native_error', 'malformed_sample', 'foreign_pair', 'nonfinite_separation'])
def test_per_direction_contact_errors_are_explicit_and_preserve_other_evidence(native, failure):
    key = native.a.prim_path, native.b.prim_path
    if failure == 'native_error':
        previous = native.api.get_contact_data

        def get_data(scene, row_prim_paths=None, column_prim_paths=None):
            if row_prim_paths == [native.a.prim_path]: raise RuntimeError('Native contact buffer read failed.')
            return previous(scene, row_prim_paths, column_prim_paths)

        native.api.get_contact_data = get_data
    elif failure == 'malformed_sample': native.api.contacts[key] = [('missing',)]
    elif failure == 'foreign_pair': native.api.contacts[key] = [native.api.sample('/World/unrelated', key[1], -.1)]
    else: native.api.contacts[key] = [native.api.sample(*key, float('nan'))]
    result = measure_contact_pair(native.first, native.second)
    forward = pair(result)
    assert forward['available'] is False and forward['contact_data']['available'] is False
    assert forward['contact_data']['minimum_separation_m'] is None
    assert forward['impulse']['norm_native'] == 5.
    assert forward['raw_contact_data']['contact_counts'] == 2
    assert pair(result, 'second_to_first')['available'] is True
    assert '/World/unrelated' not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('failure', ['overflow', 'negative_start', 'malformed_raw', 'nonfinite_raw'])
def test_raw_pair_buffer_failures_do_not_change_public_contact_sample_measurements(native, failure):
    if failure == 'overflow': native.api._CONTACT_DATA[7][5][1, 0] = 255
    elif failure == 'negative_start': native.api._CONTACT_DATA[7][5][1, 0] = -1
    elif failure == 'malformed_raw': native.api._CONTACT_DATA[7] = ('incomplete',)
    else: native.api._CONTACT_DATA[7][3][1] = float('nan')
    result = measure_contact_pair(native.first, native.second)
    forward = pair(result)
    assert forward['contact_data']['sample_count'] == 2
    raw = forward['raw_contact_data']
    if failure in ('overflow', 'negative_start'):
        assert raw['available'] is True and raw['pair_slice_available'] is False
        assert raw['pair_separations_m'] is raw['pair_minimum_separation_m'] is None
        assert raw['buffer_lengths']['forces'] == 8
    else:
        assert raw['available'] is False and 'error' in raw
    json.dumps(result, allow_nan=False)


def test_missing_or_failed_sleep_getters_are_unavailable_without_substitute_values(native):
    del native.first.sleep_threshold

    class FailedSleep(Native):
        @property
        def is_asleep(self):
            raise RuntimeError('Native sleep state unavailable.')

    del native.a.is_asleep
    native.a.__class__ = FailedSleep
    result = measure_contact_pair(native.first, native.second)
    threshold = result['first']['sleep_getters']['sleep_threshold']
    asleep = result['first']['links'][0]['sleep_getters']['is_asleep']
    assert threshold['available'] is asleep['available'] is False
    assert 'AttributeError' in threshold['error'] and 'RuntimeError' in asleep['error']
    assert 'value' not in threshold and 'value' not in asleep
    assert pair(result)['available'] is True


@pytest.mark.parametrize('failure', ['shape', 'nan', 'native_error'])
def test_impulse_errors_do_not_replace_valid_contact_data(native, failure):
    if failure == 'shape':
        previous = native.api.get_impulses
        native.api.get_impulses = lambda rows, columns: np.zeros(3) if rows == [native.a.prim_path] else previous(rows, columns)
    elif failure == 'nan': native.api.impulses[native.a.prim_path, native.b.prim_path][0] = np.nan
    else:
        previous = native.api.get_impulses

        def get_impulses(rows, columns):
            if rows == [native.a.prim_path]: raise RuntimeError('Native impulse read failed.')
            return previous(rows, columns)

        native.api.get_impulses = get_impulses
    result = measure_contact_pair(native.first, native.second)
    assert pair(result)['impulse']['available'] is False
    assert pair(result)['contact_data']['sample_count'] == 2
    assert pair(result, 'second_to_first')['available'] is True
    json.dumps(result, allow_nan=False)


def test_no_native_link_inventory_is_explicitly_unavailable(native):
    native.first.links = {}
    result = measure_contact_pair(native.first, native.second)
    assert result['first']['errors']
    assert all(direction['available'] is False and direction['link_pairs'] == []
               for direction in result['directions'].values())
    json.dumps(result, allow_nan=False)


def test_cross_scene_pair_reports_indices_without_querying_cross_scene_contacts(native):
    native.api.scenes[native.b.prim_path] = 8
    result = measure_contact_pair(native.first, native.second)
    for direction in result['directions'].values():
        measured = direction['link_pairs'][0]
        assert measured['available'] is False
        assert measured['scene_idx'] != measured['column_scene_idx']
        assert measured['contact_data']['reason'] == 'native_pair_scenes_differ'
    assert not any(call[0] in ('impulses', 'contact_data') for call in native.api.calls)
    json.dumps(result, allow_nan=False)


def test_missing_native_api_is_reported_without_initializing_a_replacement(native, monkeypatch):
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.usd_utils', NS())
    result = measure_contact_pair(native.first, native.second)
    assert 'ImportError' in result['contact_api_error']
    assert all(direction['available'] is False for direction in result['directions'].values())
    assert result['first']['sleep_getters']['is_asleep']['available'] is True
    assert native.api.calls == []
    json.dumps(result, allow_nan=False)


@pytest.fixture
def robot(native):
    native.first.arm_names = ['left', 'right']
    native.first.finger_links = {arm: [native.a] for arm in native.first.arm_names}
    native.a.get_position_orientation = lambda: (np.array([1., 2., 3.]),
        np.array([0., 0., np.sqrt(.5), np.sqrt(.5)]))
    native.a.collision_boundary_points_world = np.array([[.9, 1.8, 2.9], [1.1, 2.2, 3.1]])
    native.first.get_eef_pose = lambda arm: (np.array([10., 20., 30.]) + (arm == 'right'), np.array([0., 0., 0., 1.]))
    native.first.eef_to_fingertip_lengths = {
        'left': {native.a.name: .08}, 'right': {native.a.name: .09}}
    native.first._ag_obj_in_hand = {'left': None, 'right': native.second}
    native.first.is_grasping = lambda arm: -1 if arm == 'left' else 1
    native.first.assisted_grasp_start_points = {arm: [
        NS(link_name='base_link', position=np.array([.1, 0., 0.])),
        NS(link_name='base_link', position=np.array([0., .2, 0.]))] for arm in native.first.arm_names}
    native.first.assisted_grasp_end_points = {arm: [
        NS(link_name='base_link', position=np.array([-.1, 0., 0.])),
        NS(link_name='base_link', position=np.array([0., -.2, 0.]))] for arm in native.first.arm_names}

    def forbidden(*args, **kwargs):
        raise AssertionError('Grasp geometry must not cast rays, mutate grasp state or advance physics.')

    native.first._find_gripper_raycast_collisions = forbidden
    native.first.step = native.first.apply_action = native.first.keep_still = forbidden
    return native


def test_robot_grasp_geometry_reports_each_arm_pose_fingers_and_actual_ray_endpoint_math(robot):
    before = deepcopy(robot.a.collision_boundary_points_world)
    held_before = dict(robot.first._ag_obj_in_hand)
    points_before = {side: {arm: [point.position.copy() for point in mapping[arm]]
        for arm in robot.first.arm_names} for side, mapping in (
            ('start', robot.first.assisted_grasp_start_points), ('end', robot.first.assisted_grasp_end_points))}
    result = measure_contact_pair(robot.first, robot.second)
    geometry = result['robot_grasp_geometry']['first']
    assert geometry['available'] is True and geometry['arm_names_source'] == 'arm_names'
    assert set(geometry['arms']) == {'left', 'right'}
    for arm, row in geometry['arms'].items():
        assert row['available'] is True
        assert row['eef_pose']['position_m'] == ([10., 20., 30.] if arm == 'left' else [11., 21., 31.])
        assert row['eef_pose']['quaternion_xyzw'] == [0., 0., 0., 1.]
        assert row['eef_to_fingertip_lengths']['value'] == {robot.a.name: .08 if arm == 'left' else .09}
        finger, = row['finger_links']['links']
        assert finger['link_prim_path'] == robot.a.prim_path
        assert finger['pose']['position_m'] == [1., 2., 3.]
        assert finger['collision_boundary_aabb']['aabb_m'] == [[.9, 1.8, 2.9], [1.1, 2.2, 3.1]]
        assert finger['collision_boundary_aabb']['point_count'] == 2
        assert row['ag_obj_in_hand']['object_name'] == (None if arm == 'left' else robot.second.name)
        assert row['is_grasping']['native_value'] == (-1 if arm == 'left' else 1)
        rays = row['grasp_ray_endpoints']
        assert rays['available'] is True and rays['ray_count'] == 4 and rays['end_epsilon_m'] == 1e-8
        np.testing.assert_allclose(rays['start'][0]['world_position_m'], [1., 2.1, 3.], atol=1e-15)
        np.testing.assert_allclose(rays['start'][1]['world_position_m'], [.8, 2., 3.], atol=1e-15)
        np.testing.assert_allclose(rays['end'][0]['world_position_m'], np.array([1., 1.9, 3.]) + 1e-8, atol=1e-15)
        np.testing.assert_allclose(rays['end'][1]['world_position_m'], np.array([1.2, 2., 3.]) + 1e-8, atol=1e-15)
        assert [(ray['start_index'], ray['end_index']) for ray in rays['rays']] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert result['robot_grasp_geometry']['second']['available'] is False
    assert result['robot_grasp_geometry']['second']['reason'] == 'native_robot_grasp_interface_unavailable'
    np.testing.assert_array_equal(robot.a.collision_boundary_points_world, before)
    assert robot.first._ag_obj_in_hand == held_before
    for side, mapping in (('start', robot.first.assisted_grasp_start_points), ('end', robot.first.assisted_grasp_end_points)):
        for arm, points in mapping.items():
            for point, original in zip(points, points_before[side][arm]):
                np.testing.assert_array_equal(point.position, original)
    assert result['physics_steps'] == result['actor_mutations'] == 0
    json.dumps(result, allow_nan=False)


def test_robot_geometry_is_discovered_when_robot_is_second_argument(robot):
    result = measure_contact_pair(robot.second, robot.first)
    assert result['robot_grasp_geometry']['second']['available'] is True
    assert result['robot_grasp_geometry']['first']['available'] is False
    assert pair(result)['impulse']['norm_native'] == 2.


@pytest.mark.parametrize('source', ['arms', 'finger_links'])
def test_robot_arm_enumeration_uses_only_available_declared_native_sources(robot, source):
    del robot.first.arm_names
    if source == 'arms': robot.first.arms = ['left', 'right']
    result = measure_contact_pair(robot.first, robot.second)
    geometry = result['robot_grasp_geometry']['first']
    assert geometry['available'] is True and geometry['arm_names_source'] == source
    assert geometry['interface_getters']['arm_names']['available'] is False


@pytest.mark.parametrize('failure', ['missing_eef', 'missing_lengths', 'missing_held', 'missing_endpoint', 'malformed_endpoint', 'grasp_getter_error'])
def test_optional_native_grasp_failures_are_explicit_and_do_not_change_contact_evidence(robot, failure):
    if failure == 'missing_eef':
        del robot.first.get_eef_pose
        field = 'eef_pose'
    elif failure == 'missing_lengths':
        robot.first.eef_to_fingertip_lengths.pop('left')
        field = 'eef_to_fingertip_lengths'
    elif failure == 'missing_held':
        robot.first._ag_obj_in_hand.pop('left')
        field = 'ag_obj_in_hand'
    elif failure == 'missing_endpoint':
        robot.first.assisted_grasp_end_points['left'] = []
        field = 'grasp_ray_endpoints'
    elif failure == 'malformed_endpoint':
        robot.first.assisted_grasp_start_points['left'][0].position[0] = np.nan
        field = 'grasp_ray_endpoints'
    else:
        def failed(arm):
            raise RuntimeError('Native grasp state unavailable.')
        robot.first.is_grasping = failed
        field = 'is_grasping'
    result = measure_contact_pair(robot.first, robot.second)
    row = result['robot_grasp_geometry']['first']['arms']['left']
    assert row['available'] is False and row[field]['available'] is False and 'error' in row[field]
    assert pair(result)['contact_data']['sample_count'] == 2
    assert pair(result)['impulse']['norm_native'] == 5.
    json.dumps(result, allow_nan=False)


def test_unavailable_finger_collision_boundary_keeps_native_finger_pose_without_aabb_fallback(robot):
    del robot.a.collision_boundary_points_world
    result = measure_contact_pair(robot.first, robot.second)
    row = result['robot_grasp_geometry']['first']['arms']['left']
    assert row['available'] is False
    finger, = row['finger_links']['links']
    assert finger['pose']['available'] is True
    assert finger['collision_boundary_aabb']['available'] is False
    assert 'aabb_m' not in finger['collision_boundary_aabb']
    assert row['grasp_ray_endpoints']['available'] is True


def test_partial_robot_interface_without_arm_declarations_is_unavailable_with_reason(native):
    native.first.get_eef_pose = lambda arm: ([0., 0., 0.], [0., 0., 0., 1.])
    result = measure_contact_pair(native.first, native.second)
    geometry = result['robot_grasp_geometry']['first']
    assert geometry['available'] is False and geometry['reason'] == 'native_robot_arm_names_unavailable'
    assert geometry['interface_getters']['get_eef_pose']['available'] is True


def test_native_unknown_grasp_state_stays_integer_zero_without_truth_value_conversion(robot):
    robot.first.is_grasping = lambda arm: 0
    result = measure_contact_pair(robot.first, robot.second)
    for row in result['robot_grasp_geometry']['first']['arms'].values():
        assert row['is_grasping'] == {'available': True, 'native_getter': 'is_grasping', 'native_value': 0}
    json.dumps(result, allow_nan=False)


@pytest.fixture
def direct_view(native):
    counts, starts = np.zeros((3, 3), int), np.zeros((3, 3), int)
    counts[1, 0], starts[1, 0] = 3, 2
    counts[2, 2], starts[2, 2] = 2, 8
    data = (np.zeros((12, 1)), np.zeros((12, 3)), np.ones((12, 3)),
        np.array([99., 99., -.02, -.01, .0002, 99., 99., 99., -.007, .0005, 99., 99.]), counts, starts)
    calls = []

    def get_contact_data(*, dt):
        calls.append(dt)
        return data

    view = NS(get_contact_data=get_contact_data)
    native.api._CONTACT_VIEW = {7: view}
    return NS(native=native, data=data, calls=calls, view=view)


def direct_pair(result, direction='first_to_second'):
    return next(row for row in result['direct_native_contact_view']['scenes']['7']['link_pairs']
                if row['direction'] == direction)


def test_direct_existing_view_is_read_once_with_correct_forward_reverse_slices_and_untouched_cache(direct_view):
    fixture, native = direct_view, direct_view.native
    cache_identity, cache_tuple = native.api._CONTACT_DATA, native.api._CONTACT_DATA[7]
    view_mapping = native.api._CONTACT_VIEW
    cached_before, direct_before = deepcopy(cache_tuple), deepcopy(fixture.data)
    result = measure_contact_pair(native.first, native.second)
    evidence = result['direct_native_contact_view']
    scene = evidence['scenes']['7']
    assert evidence['available'] is scene['available'] is True
    assert evidence['public_contact_cache_written_by_helper'] is False
    assert evidence['empty_samples_establish_clearance'] is False
    assert fixture.calls == [1.0] and scene['direct_getter_calls'] == 1
    assert native.api._CONTACT_VIEW is view_mapping and view_mapping[7] is fixture.view
    assert native.api._CONTACT_DATA is cache_identity and cache_identity[7] is cache_tuple
    for actual, expected in zip(cache_tuple, cached_before):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(fixture.data, direct_before):
        np.testing.assert_array_equal(actual, expected)
    forward, reverse = direct_pair(result), direct_pair(result, 'second_to_first')
    assert (forward['row_index'], forward['column_index'], forward['contact_counts'], forward['start_indices']) == (1, 0, 3, 2)
    assert (reverse['row_index'], reverse['column_index'], reverse['contact_counts'], reverse['start_indices']) == (2, 2, 2, 8)
    assert forward['pair_separations_m'] == [-.02, -.01, .0002]
    assert forward['pair_minimum_separation_m'] == -.02
    assert reverse['pair_separations_m'] == [-.007, .0005]
    assert forward['buffer_lengths'] == {'forces': 12, 'points': 12, 'normals': 12, 'separations': 12}
    assert forward['configured_buffer_capacity']['value'] == 256
    assert pair(result)['raw_contact_data']['pair_separations_m'] == [-.003, -.001]
    assert 99. not in forward['pair_separations_m']
    json.dumps(result, allow_nan=False)


def test_zero_public_impulses_do_not_gate_direct_view_or_populate_public_cache(direct_view):
    native = direct_view.native
    native.api._CONTACT_DATA.clear()
    native.api.contacts = {key: [] for key in native.api.contacts}
    native.api.impulses = {key: np.zeros(3) for key in native.api.impulses}
    result = measure_contact_pair(native.first, native.second)
    assert pair(result)['impulse']['norm_native'] == 0.
    assert pair(result)['contact_data']['sample_count'] == 0
    assert pair(result)['raw_contact_data']['available'] is False
    assert direct_pair(result)['pair_separations_m'] == [-.02, -.01, .0002]
    assert direct_view.calls == [1.0] and native.api._CONTACT_DATA == {}


@pytest.mark.parametrize('failure', ['overflow', 'negative_start', 'malformed_buffers', 'nonfinite_separation', 'native_error'])
def test_direct_view_truncation_or_errors_remain_separate_from_public_contact_evidence(direct_view, failure):
    fixture, native = direct_view, direct_view.native
    if failure == 'overflow': fixture.data[5][1, 0] = 11
    elif failure == 'negative_start': fixture.data[5][1, 0] = -1
    elif failure == 'malformed_buffers': fixture.view.get_contact_data = lambda dt: ('incomplete',)
    elif failure == 'nonfinite_separation': fixture.data[3][2] = np.nan
    else:
        def failed(dt):
            fixture.calls.append(dt)
            raise RuntimeError('Native view buffer getter failed.')
        fixture.view.get_contact_data = failed
    result = measure_contact_pair(native.first, native.second)
    scene = result['direct_native_contact_view']['scenes']['7']
    assert scene['direct_getter_calls'] == 1
    row = direct_pair(result)
    if failure in ('overflow', 'negative_start'):
        assert row['available'] is True and row['pair_slice_available'] is False
        assert row['pair_separations_m'] is row['pair_minimum_separation_m'] is None
        assert direct_pair(result, 'second_to_first')['pair_separations_m'] == [-.007, .0005]
    else:
        assert row['available'] is False
    assert pair(result)['contact_data']['sample_count'] == 2
    assert pair(result)['raw_contact_data']['pair_separations_m'] == [-.003, -.001]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('missing', ['mapping', 'entry', 'none'])
def test_missing_existing_native_view_is_unavailable_without_creation_or_fallback(direct_view, missing):
    fixture, native = direct_view, direct_view.native
    if missing == 'mapping': del native.api._CONTACT_VIEW
    elif missing == 'entry': native.api._CONTACT_VIEW.clear()
    else: native.api._CONTACT_VIEW[7] = None
    before = deepcopy(native.api._CONTACT_DATA)
    result = measure_contact_pair(native.first, native.second)
    scene = result['direct_native_contact_view']['scenes']['7']
    assert scene['available'] is False and scene['reason'] == 'existing_native_contact_view_missing'
    assert scene['direct_getter_calls'] == 0 and fixture.calls == []
    assert direct_pair(result)['available'] is False
    for actual, expected in zip(native.api._CONTACT_DATA[7], before[7]):
        np.testing.assert_array_equal(actual, expected)


def test_empty_direct_pair_data_never_claims_clearance(direct_view):
    direct_view.data[4][:] = 0
    result = measure_contact_pair(direct_view.native.first, direct_view.native.second)
    for row in result['direct_native_contact_view']['scenes']['7']['link_pairs']:
        assert row['pair_slice_available'] is True and row['pair_sample_count'] == 0
        assert row['pair_separations_m'] == [] and row['pair_minimum_separation_m'] is None
    assert result['direct_native_contact_view']['empty_samples_establish_clearance'] is False


def test_missing_direction_indices_do_not_block_the_other_direct_slice(direct_view):
    native = direct_view.native
    native.api.rows.pop(native.a.prim_path)
    result = measure_contact_pair(native.first, native.second)
    evidence = result['direct_native_contact_view']
    assert evidence['available'] is False
    assert len(evidence['unavailable_pairs']) == 1
    assert evidence['unavailable_pairs'][0]['direction'] == 'first_to_second'
    assert direct_pair(result, 'second_to_first')['pair_separations_m'] == [-.007, .0005]
    assert direct_view.calls == [1.0]

"""CPU protocol tests for read-only measured rectangular stance proposals.

Run: python -B -m pytest -q tests/test_task_edit2_native_stance_proposals.py
Synthetic native maps/AABBs test proposal geometry; no simulator or GPU runs.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.tasks.interaction_setup import _navigation_cells
from wia.tasks.native_stance_proposals import propose_native_stances


class NativeMap:
    map_resolution = .1
    map_size = 21
    floor_heights = [0.]

    def __init__(self):
        self.floor_map = [np.full((21, 21), 255, dtype=np.uint8)]
        self.room_ins_map = np.ones((21, 21), dtype=np.uint8)
        self.room_ins_name_to_ins_id = {'bathroom_1': 1, 'other_room': 2}

    def map_to_world(self, cells):
        return np.flip((np.asarray(cells)-10)*.1, axis=-1).copy()

    def world_to_map(self, points):
        # Deliberately reproduce the installed API's batch-axis reversal.
        return np.flip(np.asarray(points)/.1+10).astype(int).copy()

    def get_room_instance_by_point(self, point):
        cell = self.world_to_map(point).reshape(2)
        if np.any(cell < 0) or np.any(cell >= 21):
            return None
        return {1: 'bathroom_1', 2: 'other_room'}[int(self.room_ins_map[tuple(cell)])]


class NativeFloat32Map(NativeMap):
    """Installed OG-shaped map with large corner coordinates and float32 output."""
    map_size = 862

    def __init__(self, offset=(0., 0.), distortion=None):
        self.floor_map = [np.zeros((862, 862), dtype=np.uint8)]
        self.floor_map[0][426:437, 423:440] = 255
        self.offset, self.distortion = np.asarray(offset, dtype=np.float32), distortion

    def map_to_world(self, cells):
        import torch
        cells = torch.as_tensor(cells, dtype=torch.float32)
        scale = .101 if self.distortion == 'scale' else self.map_resolution
        result = torch.flip((cells-self.map_size/2)*scale, dims=(-1,))
        if self.distortion == 'nonaffine':
            span = self.map_size-1
            result[..., 0] += .04*cells[..., 0]*(span-cells[..., 0])/(span*span)
        elif self.distortion == 'degenerate':
            result[..., 1] = result[..., 0]
        return result + torch.as_tensor(self.offset)

    def world_to_map(self, points):
        return np.flip((np.asarray(points)-self.offset)/self.map_resolution+self.map_size/2).astype(int).copy()

    def get_room_instance_by_point(self, point):
        cell = self.world_to_map(point).reshape(2)
        return 'bathroom_1' if np.all(cell >= 0) and np.all(cell < self.map_size) else None


class Actor:
    def __init__(self, name, category, low, high, visual=False):
        self.name, self.category, self.model = name, category, 'native_model'
        self.prim_path = '/World/' + name
        self.aabb = (np.array(low, dtype=float), np.array(high, dtype=float))
        self.visual_only = visual

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError('Proposal code must never set poses.')


class Robot(Actor):
    def __init__(self):
        super().__init__('robot', 'agent', [-.4, -.1, .1], [.4, .1, 1.2])
        self.position, self.quaternion = np.array([0., 0., .2]), np.array([0., 0., 0., 1.])
        self.base_footprint_link_name = 'chassis'
        self.links = {'chassis': NS(prim_path='/World/robot/chassis',
            aabb=(np.array([-.4, -.1, .1]), np.array([.4, .1, .3])))}

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()


class Session(NS):
    @property
    def observation_source(self):
        raise AssertionError('Proposal code must not access observation state.')


@pytest.fixture
def native():
    def forbidden(*args, **kwargs):
        raise AssertionError('Proposal code must not render, step or use controls.')
    native_map, robot = NativeMap(), Robot()
    floor = Actor('floor', 'floors', [-1.1, -1.1, -.1], [1.1, 1.1, 0.])
    scene = NS(scene_model='building', trav_map=native_map, seg_map=native_map, objects=[robot, floor])
    task = NS(definition=NS(id='revised_task'), source_path='/source/original.json', source_sha256='source_hash',
              file_sha256='task_hash', seed=12)
    session = Session(env=NS(scene=scene, step=forbidden), native_reset_id='reset_a', episode_id='episode_a',
        loaded_task=task, pipeline=NS(og=NS(sim=NS(step=forbidden, render=forbidden))))
    return NS(session=session, robot=robot, native_map=native_map, scene=scene)


def propose(native, *, region='bathroom_1', excluded=(), target=(3., 0.), limit=32):
    return propose_native_stances(native.session, native.robot, region, set(excluded), target,
                                 candidate_limit=limit)


def test_narrow_aisle_circle_fails_but_measured_rectangle_proposes(native):
    native.native_map.floor_map[0][:8] = 0
    native.native_map.floor_map[0][13:] = 0
    with pytest.raises(ValueError, match='No conservative'):
        _navigation_cells(native.session, native.robot, 'bathroom_1', set())
    xy, floor, receipt = propose(native)
    assert receipt['available'] and len(xy) > 0 and floor == 0.
    assert receipt['counts']['all_raw_free_cells'] == 105
    assert receipt['counts']['post_foreign_AABB_cells'] > 0
    assert receipt['proposal_only'] and receipt['qualification'] is None
    assert receipt['gap_padding_m'] == 0. and receipt['scale_changed'] is False
    assert receipt['provenance']['reset_id'] == 'reset_a'
    assert receipt['provenance']['source_sha256'] == 'source_hash'
    assert receipt['native_chassis']['foreign_AABB_screen_absolute_z_interval_m'] == [.1, .3]
    assert receipt['raw_floor']['nominal_floor_z_m'] == 0.
    assert all(row['accepted'] for row in receipt['proposals'] for row in row['footprint']['sampled_cells'])
    json.dumps(receipt, allow_nan=False)


def test_exact_original_room_and_entire_footprint_are_enforced(native):
    native.native_map.room_ins_map[:, 11:] = 2
    xy, _, receipt = propose(native, target=(.9, 0.))
    assert receipt['available'] and len(xy)
    assert receipt['counts']['exact_room_raw_cells'] == 231
    assert receipt['counts']['all_raw_free_cells'] == 441
    assert receipt['other_room_fallback_used'] is False
    for row in receipt['proposals']:
        assert native.native_map.get_room_instance_by_point(row['world_xy_m']) == 'bathroom_1'
        assert set(row['footprint']['rectangle_corner_rooms']) == {'bathroom_1'}
        assert {cell['room'] for cell in row['footprint']['sampled_cells']} == {'bathroom_1'}
    no_room, _, absent = propose(native, region='absent_room')
    assert no_room.shape == (0, 2) and absent['counts']['exact_room_raw_cells'] == 0
    full, _, unavailable = propose(native, region='full_scene')
    assert full.shape == (0, 2) and not unavailable['available']


def test_current_yaw_and_root_to_chassis_offset_survive_candidate_rotation(native):
    native.robot.position[:2] = [.2, -.3]
    native.robot.quaternion = np.array([0., 0., np.sin(np.pi/4), np.cos(np.pi/4)])
    native.robot.links['chassis'].aabb = (np.array([.25, -.7, .1]), np.array([.45, .1, .3]))
    xy, _, receipt = propose(native, target=(3., 0.), limit=1)
    assert len(xy) == 1
    row = receipt['proposals'][0]
    polygon = np.array(row['rectangle_xy_m'])
    angle = row['yaw_rad'] - receipt['native_chassis']['root_yaw_rad']
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    original = np.array([[.25, -.7], [.45, -.7], [.45, .1], [.25, .1]])
    expected = (original-native.robot.position[:2]) @ rotation.T + xy[0]
    assert np.allclose(polygon, expected)
    assert not np.allclose(polygon.mean(0), xy[0])
    assert receipt['native_chassis']['root_to_AABB_centre_xy_m'] == pytest.approx([.15, 0.])


@pytest.mark.parametrize('category,visual', [('table', False), ('walls', False), ('task_prop', True)])
def test_foreign_geometry_is_retained_unless_its_exact_name_is_excluded(native, category, visual):
    blocker = Actor('foreign', category, [-2., -2., .15], [2., 2., .25], visual=visual)
    native.scene.objects.append(blocker)
    xy, _, receipt = propose(native)
    assert receipt['available'] and not len(xy)
    assert receipt['counts']['post_footprint_cells'] > 0
    assert receipt['counts']['post_foreign_AABB_cells'] == 0
    assert receipt['height_overlapping_foreign_inventory'][0]['name'] == 'foreign'
    allowed, _, explicit = propose(native, excluded={'foreign'})
    assert len(allowed) and explicit['excluded_actor_names'] == ['foreign']


def test_actual_chassis_height_does_not_treat_high_furniture_as_low_obstacle(native):
    native.scene.objects.append(Actor('high_shelf', 'shelf', [-2., -2., .8], [2., 2., 1.]))
    xy, _, receipt = propose(native)
    assert len(xy) and receipt['height_overlapping_foreign_inventory'] == []
    assert 'arm_clearance' in receipt['pending_gates']


def test_raster_checks_interior_hole_even_when_rectangle_corners_are_free(native):
    native.native_map.floor_map[0][10, 10] = 0
    _, _, receipt = propose(native, target=(0., 0.), limit=441)
    assert receipt['available']
    for row in receipt['proposals']:
        assert all(sample['map_cell'] != [10, 10] for sample in row['footprint']['sampled_cells'])
    assert receipt['counts']['post_footprint_cells'] < receipt['counts']['exact_room_raw_cells']


@pytest.mark.parametrize('missing', ['map', 'floor_height', 'floor_object', 'chassis', 'pose', 'nonfinite'])
def test_missing_native_measurements_never_promote_candidates(native, missing):
    if missing == 'map':
        del native.scene.trav_map
    elif missing == 'floor_height':
        native.native_map.floor_heights = []
    elif missing == 'floor_object':
        native.scene.objects = [native.robot]
    elif missing == 'chassis':
        native.robot.links = {}
    elif missing == 'pose':
        native.robot.get_position_orientation = lambda: (np.zeros(2), np.array([0., 0., 0., 1.]))
    else:
        native.robot.links['chassis'].aabb[0][0] = np.nan
    xy, _, receipt = propose(native)
    assert xy.shape == (0, 2) and not receipt['available'] and receipt['reason']
    assert receipt['qualification'] is None and receipt['proposals'] == []


def test_complete_screen_counts_and_deterministic_cap_are_explicit(native):
    first, floor, receipt = propose(native, limit=3)
    second, same_floor, again = propose(native, limit=3)
    assert len(first) == 3 and np.array_equal(first, second) and floor == same_floor
    assert receipt == again and receipt['candidate_truncation_applied']
    assert receipt['counts']['post_foreign_AABB_cells'] > 3
    assert receipt['counts']['returned_cells'] == 3
    distances = np.linalg.norm(first-[3., 0.], axis=1)
    assert np.all(np.diff(distances) >= 0)
    assert receipt['previous_circle_reason'] is None


def test_zero_state_mutation_no_render_physics_observation_or_setters(native):
    old_floor = native.native_map.floor_map[0].copy()
    old_rooms = native.native_map.room_ins_map.copy()
    old_pose = native.robot.get_position_orientation()
    old_aabbs = deepcopy([obj.aabb for obj in native.scene.objects])
    old_attributes = set(vars(native.session))
    for _ in range(2):
        assert propose(native, limit=1)[2]['available']
    assert np.array_equal(old_floor, native.native_map.floor_map[0])
    assert np.array_equal(old_rooms, native.native_map.room_ins_map)
    assert all(np.array_equal(a, b) for a, b in zip(old_pose, native.robot.get_position_orientation()))
    assert all(np.array_equal(a, b) for previous, obj in zip(old_aabbs, native.scene.objects)
               for a, b in zip(previous, obj.aabb))
    assert set(vars(native.session)) == old_attributes


@pytest.mark.parametrize('offset', [(0., 0.), (1000., -2000.)])
def test_long_native_spans_handle_actual_862_float32_map_and_record_numeric_proof(native, offset):
    large = NativeFloat32Map(offset=offset)
    native.scene.trav_map = native.scene.seg_map = large
    if offset == (0., 0.):
        short = np.asarray(large.map_to_world([[0, 0], [1, 0], [0, 1]]), dtype=float)
        assert not np.allclose(np.linalg.norm(short[1:]-short[0], axis=1), .1, rtol=1e-5, atol=0.)
    xy, _, receipt = propose(native, target=np.array(offset)+[3., 0.], limit=3)
    assert receipt['available'] and len(xy) == 3
    proof = receipt['raw_floor']['affine_basis_measurement']
    assert proof['native_output_dtype'] == 'float32'
    assert proof['sample_indices'] == [[0, 0], [861, 0], [0, 861], [431, 431]]
    assert proof['axis_cell_spans'] == [861, 861]
    assert proof['resolution_rtol'] == 1e-5 and proof['resolution_atol_m'] == 0.
    assert proof['axes_nondegenerate'] and proof['scale_matches_declared_resolution']
    check = proof['central_affine_check']
    assert check['consistent']
    assert np.all(np.array(check['absolute_residual_m']) <= check['coordinate_numeric_bound_m'])
    assert np.all(np.array(check['coordinate_numeric_bound_m']) > 0)
    assert receipt['gap_padding_m'] == 0. and receipt['qualification'] is None
    assert receipt['counts']['all_raw_free_cells'] == 187
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize('distortion', ['scale', 'nonaffine', 'degenerate'])
def test_wrong_native_scale_or_nonaffine_conversion_still_rejects(native, distortion):
    large = NativeFloat32Map(distortion=distortion)
    native.scene.trav_map = native.scene.seg_map = large
    xy, _, receipt = propose(native)
    assert xy.shape == (0, 2) and not receipt['available']
    proof = receipt['raw_floor']['affine_basis_measurement']
    assert proof['available'] is False
    if distortion == 'scale':
        assert not proof['scale_matches_declared_resolution']
    elif distortion == 'nonaffine':
        assert proof['scale_matches_declared_resolution']
        assert not proof['central_affine_check']['consistent']
    else:
        assert not proof['axes_nondegenerate']
    assert receipt['proposals'] == [] and receipt['qualification'] is None

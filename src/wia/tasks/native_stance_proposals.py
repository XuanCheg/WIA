"""Read-only TaskEdit2 pilot XY stance proposals from measured chassis bounds.

Run: python -B -m pytest -q tests/test_task_edit2_native_stance_proposals.py
Call propose_native_stances only under the explicit revised TaskEdit2 policy
after the legacy circle proposal screen fails. Returned cells are proposals,
never native collision, floor-contact, stability, arm or access certificates.
"""
from __future__ import annotations

from itertools import product
import hashlib
import math

import numpy as np

from .interaction_setup import _objects, array


def _native_points(trav, indices):
    import torch
    result = trav.map_to_world(torch.as_tensor(np.asarray(indices, dtype=int)))
    if hasattr(result, 'detach'):
        result = result.detach().cpu().numpy()
    return np.asarray(result)


def _points(trav, indices):
    result = array(_native_points(trav, indices))
    if result.shape != np.asarray(indices).shape:
        raise ValueError('Native map_to_world returned an unexpected shape.')
    return result


def _affine_basis(trav, shape, resolution):
    """Measure long native spans; check conversion error without a physical margin."""
    spans = np.array(shape, dtype=int) - 1
    centre = np.array(shape, dtype=int) // 2
    indices = np.array([[0, 0], [spans[0], 0], [0, spans[1]], centre])
    native = _native_points(trav, indices)
    points = array(native)
    if points.shape != (4, 2) or native.dtype.kind != 'f':
        raise ValueError('Native map basis samples must be floating XY coordinates.')
    basis = (points[1:3]-points[0]) / spans[:, None]
    weights = np.array([1.-centre[0]/spans[0]-centre[1]/spans[1],
                        centre[0]/spans[0], centre[1]/spans[1]])
    reconstructed = weights @ points[:3]
    # Each returned coordinate receives one ULP of its actual output dtype.
    # Weighted endpoint ULPs plus central ULP and float64 evaluation roundoff
    # bound only the affine conversion check, never footprint/collision geometry.
    ulps = np.asarray(np.spacing(np.abs(native)), dtype=float)
    evaluation_roundoff = (np.abs(weights) @ np.abs(points[:3])) * np.finfo(float).eps * 4
    bound = np.abs(weights) @ ulps[:3] + ulps[3] + evaluation_roundoff
    residual = np.abs(points[3]-reconstructed)
    norms = np.linalg.norm(basis, axis=1)
    scale_matches = bool(np.allclose(norms, resolution, rtol=1e-5, atol=0.))
    nondegenerate = bool(abs(np.linalg.det(basis)) > 0)
    consistent = bool(np.all(residual <= bound))
    evidence = {'method': 'native_in_map_long_axis_spans_divided_by_exact_cell_spans',
        'native_output_dtype': str(native.dtype), 'sample_indices': indices.tolist(),
        'native_sample_world_xy_m': points.tolist(), 'axis_cell_spans': spans.tolist(),
        'map_cell_basis_xy_m': basis.tolist(), 'basis_norms_m': norms.tolist(),
        'declared_resolution_m': resolution, 'resolution_rtol': 1e-5, 'resolution_atol_m': 0.,
        'scale_matches_declared_resolution': scale_matches, 'axes_nondegenerate': nondegenerate,
        'central_affine_check': {'sample_index': centre.tolist(), 'endpoint_weights': weights.tolist(),
            'reconstructed_world_xy_m': reconstructed.tolist(), 'absolute_residual_m': residual.tolist(),
            'native_sample_coordinate_ulps_m': ulps.tolist(),
            'float64_evaluation_roundoff_bound_m': evaluation_roundoff.tolist(),
            'coordinate_numeric_bound_m': bound.tolist(), 'consistent': consistent,
            'bound_derivation': 'sum(abs(endpoint weights)*endpoint output ULP) + central output ULP + four-operation float64 roundoff',
            'scope': 'map_to_world numerical conversion consistency only; no physical clearance allowance'}}
    evidence['available'] = scale_matches and nondegenerate and consistent
    return basis, evidence


def _yaw(quaternion):
    quaternion = array(quaternion)
    if quaternion.shape != (4,) or np.linalg.norm(quaternion) <= 0:
        raise ValueError('Actual native root quaternion is unavailable.')
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return float(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def _rotation(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def _overlap(first, second):
    """Closed rectangle intersection by separating axes; no padding or shrinkage."""
    for polygon in (first, second):
        for edge in np.roll(polygon, -1, axis=0) - polygon:
            axis = np.array([-edge[1], edge[0]])
            if np.linalg.norm(axis) == 0:
                raise ValueError('Degenerate native rectangle.')
            a, b = first @ axis, second @ axis
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def _rectangle(low, high):
    return np.array([[low[0], low[1]], [high[0], low[1]],
                     [high[0], high[1]], [low[0], high[1]]])


def _provenance(session):
    state = vars(session)
    task = state.get('loaded_task') or getattr(state.get('scene_adapter'), 'task', None)
    definition = getattr(task, 'definition', None)
    return {'reset_id': getattr(session, 'native_reset_id', None),
        'episode_id': getattr(session, 'episode_id', None), 'seed': getattr(task, 'seed', None),
        'scene_model': getattr(session.env.scene, 'scene_model', None),
        'task_id': getattr(definition, 'id', None), 'task_sha256': getattr(task, 'file_sha256', None),
        'source_path': str(task.source_path) if getattr(task, 'source_path', None) is not None else None,
        'source_sha256': getattr(task, 'source_sha256', None)}


def _footprint(trav, free, query, region, polygon, basis):
    # The installed world_to_map reverses a batch dimension; use scalar calls.
    indices = np.stack([array(trav.world_to_map(point)) for point in polygon])
    if indices.shape != (4, 2) or not np.isfinite(indices).all():
        raise ValueError('Native world_to_map is unavailable for footprint corners.')
    begin = np.floor(indices.min(0)).astype(int) - 1
    end = np.ceil(indices.max(0)).astype(int) + 1
    cells = np.array(list(product(range(begin[0], end[0]+1), range(begin[1], end[1]+1))))
    centres = _points(trav, cells)
    corners = np.array([(-basis[0]-basis[1])/2, (basis[0]-basis[1])/2,
                        (basis[0]+basis[1])/2, (-basis[0]+basis[1])/2])
    sampled = []
    for cell, centre in zip(cells, centres):
        if not _overlap(polygon, centre + corners):
            continue
        valid = bool(np.all(cell >= 0) and np.all(cell < free.shape))
        room = query(centre)
        clear = bool(valid and free[tuple(cell)] and room == region)
        sampled.append({'map_cell': cell.tolist(), 'world_xy_m': centre.tolist(),
                        'room': room, 'raw_free': bool(valid and free[tuple(cell)]), 'accepted': clear})
    corner_rooms = [query(point) for point in polygon]
    return {'accepted': bool(sampled and all(row['accepted'] for row in sampled)
                            and all(room == region for room in corner_rooms)),
        'method': 'all_native_map_cell_footprints_intersecting_rotated_chassis_rectangle',
        'sampled_cells': sampled, 'rectangle_corner_rooms': corner_rooms}


def propose_native_stances(session, robot, region, excluded, target_xy, *, candidate_limit=32):
    """Return (world_xy[N,2], nominal_floor_z|None, private proposal receipt).

    No state changes or observation access occur. Missing measurements return
    an empty array and explicit unavailable receipt. A zero result never proves
    infeasibility. The current chassis absolute Z range screens foreign AABBs;
    final placement must independently establish actual height and native safety.
    """
    receipt = {'schema': 'task_edit2_native_stance_proposals_20261018_v1', 'available': False,
        'proposal_only': True, 'qualification': None, 'requested_source_region': region,
        'excluded_actor_names': sorted(excluded), 'candidate_limit': candidate_limit,
        'previous_circle_reason': None, 'proposals': [], 'candidate_truncation_applied': False,
        'pending_gates': ['native_placement', 'local_floor_rays', 'native_floor_contacts',
            'upright', 'passive_stability', 'arm_clearance', 'public_access'],
        'other_room_fallback_used': False, 'gap_padding_m': 0., 'scale_changed': False,
        'limitations': ['Current rotated chassis AABB is conservative geometry, not an exact collision hull.',
            'Current absolute chassis height is a proposal screen; no candidate elevation is inferred.',
            'No result or a truncated result does not establish room or task infeasibility.']}
    nominal = None
    try:
        if not isinstance(region, str) or not region or region == 'full_scene':
            raise ValueError('An exact requested source room is required; no whole-building fallback.')
        if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit < 1:
            raise ValueError('candidate_limit must be a positive integer.')
        target = array(target_xy)
        if target.shape != (2,):
            raise ValueError('A finite explicit target XY is required.')
        scene = session.env.scene
        receipt['provenance'] = _provenance(session)
        trav, query = scene.trav_map, scene.seg_map.get_room_instance_by_point
        free = array(trav.floor_map[0]) == 255
        if free.ndim != 2 or min(free.shape) < 2:
            raise ValueError('Actual raw floor_map[0] must be a two-dimensional native map.')
        nominal = float(trav.floor_heights[0])
        resolution = float(trav.map_resolution)
        if not math.isfinite(nominal) or not math.isfinite(resolution) or resolution <= 0:
            raise ValueError('Native nominal floor height/resolution unavailable.')
        receipt['raw_floor'] = {'floor_index': 0, 'nominal_floor_z_m': nominal,
            'map_resolution_m': resolution, 'shape': list(free.shape),
            'map_sha256': hashlib.sha256(np.ascontiguousarray(array(trav.floor_map[0])).tobytes()).hexdigest()}
        objects = _objects(scene)
        floors = [obj for obj in objects if obj.category in ('floor', 'floors') and not obj.visual_only]
        if not floors:
            raise ValueError('Actual loaded collidable native floor inventory unavailable.')
        receipt['native_floor_inventory'] = [{'name': obj.name, 'prim_path': obj.prim_path,
            'aabb_m': [array(value).tolist() for value in obj.aabb]} for obj in floors]
        position, quaternion = (array(value) for value in robot.get_position_orientation())
        if position.shape != (3,):
            raise ValueError('Actual native robot root position unavailable.')
        current_yaw = _yaw(quaternion)
        name = robot.base_footprint_link_name
        link = robot.links[name]
        low, high = (array(value) for value in link.aabb)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
            raise ValueError('Actual native chassis link AABB unavailable.')
        corners = _rectangle(low[:2], high[:2])
        local = (corners - position[:2]) @ _rotation(current_yaw)
        receipt['native_chassis'] = {'link_name': name, 'link_prim_path': link.prim_path,
            'aabb_m': [low.tolist(), high.tolist()], 'extent_m': (high-low).tolist(),
            'root_position_m': position.tolist(), 'root_quaternion_xyzw': quaternion.tolist(),
            'root_yaw_rad': current_yaw, 'root_relative_xy_corners_m': local.tolist(),
            'root_to_AABB_centre_xy_m': ((low[:2]+high[:2])/2-position[:2]).tolist(),
            'foreign_AABB_screen_absolute_z_interval_m': [float(low[2]), float(high[2])]}
        basis, basis_evidence = _affine_basis(trav, free.shape, resolution)
        receipt['raw_floor']['affine_basis_measurement'] = basis_evidence
        if not basis_evidence['available']:
            raise ValueError('Native map cell basis does not match declared resolution or affine conversion consistency.')
        receipt['raw_floor']['map_cell_basis_xy_m'] = basis.tolist()
        raw = np.argwhere(free)
        world = _points(trav, raw)
        rooms = [query(point) for point in world]
        required = [i for i, room in enumerate(rooms) if room == region]
        counts = {'all_raw_free_cells': len(raw), 'exact_room_raw_cells': len(required),
            'post_footprint_cells': 0, 'post_foreign_AABB_cells': 0, 'returned_cells': 0}
        receipt['counts'] = counts
        foreign = []
        floor_ids = {id(obj) for obj in floors}
        for obj in objects:
            if obj is robot or obj.name in excluded or id(obj) in floor_ids:
                continue
            lower, upper = (array(value) for value in obj.aabb)
            if lower.shape != (3,) or upper.shape != (3,) or np.any(upper < lower):
                raise ValueError('Native foreign object AABB unavailable: ' + obj.name)
            if upper[2] < low[2] or lower[2] > high[2]:
                continue
            foreign.append({'name': obj.name, 'category': obj.category,
                'visual_only': bool(obj.visual_only), 'aabb_m': [lower.tolist(), upper.tolist()]})
        receipt['height_overlapping_foreign_inventory'] = foreign
        order = sorted(required, key=lambda i: (float(np.linalg.norm(world[i]-target)),
            float(world[i][0]), float(world[i][1]), int(raw[i][0]), int(raw[i][1])))
        passed = []
        for index in order:
            xy = world[index]
            delta = target-xy
            yaw = float(math.atan2(delta[1], delta[0])) if np.linalg.norm(delta) > .001 else 0.
            rectangle = local @ _rotation(yaw).T + xy
            footprint = _footprint(trav, free, query, region, rectangle, basis)
            if not footprint['accepted']:
                continue
            counts['post_footprint_cells'] += 1
            overlaps = [row['name'] for row in foreign if _overlap(rectangle,
                _rectangle(array(row['aabb_m'][0])[:2], array(row['aabb_m'][1])[:2]))]
            if overlaps:
                continue
            counts['post_foreign_AABB_cells'] += 1
            passed.append({'world_xy_m': xy.tolist(), 'yaw_rad': yaw, 'raw_map_cell': raw[index].tolist(),
                'rectangle_xy_m': rectangle.tolist(), 'footprint': footprint,
                'foreign_AABB_overlaps': [], 'proposal_only': True})
        receipt['candidate_truncation_applied'] = len(passed) > candidate_limit
        receipt['proposals'] = passed[:candidate_limit]
        counts['returned_cells'] = len(receipt['proposals'])
        receipt.update(available=True, reason=None if passed else 'No rectangle proposal passed; infeasibility is not established.')
        return np.array([row['world_xy_m'] for row in receipt['proposals']], dtype=float).reshape(-1, 2), nominal, receipt
    except Exception as error:
        receipt.update(available=False, reason=type(error).__name__ + ': ' + str(error), proposals=[])
        return np.empty((0, 2), dtype=float), nominal, receipt

"""Read-only support-space evidence, without choosing new footprint thresholds.

Run CPU checks: python -B -m pytest -q tests/test_task_edit2_support_diagnostics.py.
Use support_space_diagnostic(session, robot, region, excluded) for complete raw
floor points in one native room. Without a room, explicit bounds_xy_m=(low,high)
are required; unbounded full-building coordinate payloads are never emitted.
No function renders, steps, moves actors or certifies qualification.
"""
from __future__ import annotations

import numpy as np

SCHEMA = 'task_edit2_support_space_diagnostic_20261018_v1'


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in 'biuf' or not np.isfinite(result).all():
        raise ValueError('Native map/geometry measurement is unavailable or nonfinite.')
    return result


def _world(native_map, cells):
    import torch
    return _array(native_map.map_to_world(torch.as_tensor(cells, dtype=torch.float32))).astype(float)


def _map(native_map, points):
    import torch
    return _array(native_map.world_to_map(torch.as_tensor(points, dtype=torch.float32))).astype(int)


def _bounds(bounds):
    bounds = _array(bounds).astype(float)
    if bounds.shape != (2, 2) or np.any(bounds[1] <= bounds[0]):
        raise ValueError('Diagnostic XY bounds must be finite increasing (low_xy, high_xy).')
    return bounds


def _room_bounds(scene, region):
    segmentation = scene.seg_map
    label = segmentation.room_ins_name_to_ins_id.get(region)
    if label is None:
        raise ValueError('Requested region is absent from the native segmentation map.')
    labels = _array(segmentation.room_ins_map)
    cells = np.argwhere(labels == label)
    if not len(cells):
        raise ValueError('Requested native region has no segmentation cells.')
    corners = _world(segmentation, np.stack([cells.min(axis=0), cells.max(axis=0)]))
    # Include the whole boundary cells before the authoritative native per-point
    # room query. This bounds enumeration; it is not a clearance adjustment.
    resolution = float(segmentation.map_resolution)
    return np.stack([corners.min(axis=0) - resolution, corners.max(axis=0) + resolution])


def _raw_floor(session, region, bounds_xy_m):
    scene = session.env.scene
    trav = scene.trav_map
    free = _array(trav.floor_map[0]) == 255
    if free.ndim != 2:
        raise ValueError('Native floor traversability must be a two-dimensional map.')
    heights = _array(trav.floor_heights).astype(float)
    resolution = float(trav.map_resolution)
    if heights.ndim != 1 or not len(heights) or resolution <= 0 or not np.isfinite(resolution):
        raise ValueError('Native floor heights or map resolution are unavailable.')
    requested_room = bool(region and region != 'full_scene')
    if not requested_room and bounds_xy_m is None:
        raise ValueError('Explicit diagnostic XY bounds are required without a requested room.')
    domain = _room_bounds(scene, region) if requested_room else _bounds(bounds_xy_m)
    if bounds_xy_m is not None and requested_room:
        explicit = _bounds(bounds_xy_m)
        domain = np.stack([np.maximum(domain[0], explicit[0]), np.minimum(domain[1], explicit[1])])
        if np.any(domain[1] <= domain[0]):
            raise ValueError('Explicit diagnostic bounds do not intersect the requested native room.')
    corners = _map(trav, domain)
    begin = np.maximum(corners.min(axis=0) - 1, 0)
    end = np.minimum(corners.max(axis=0) + 2, free.shape)
    cells = np.argwhere(free[begin[0]:end[0], begin[1]:end[1]]) + begin
    points = _world(trav, cells) if len(cells) else np.empty((0, 2))
    points = points[np.all((points >= domain[0]) & (points <= domain[1]), axis=1)]
    rooms = [scene.seg_map.get_room_instance_by_point(point) for point in points]
    if requested_room:
        keep = np.array([name == region for name in rooms], dtype=bool)
        points = points[keep]
        rooms = [name for name, kept in zip(rooms, keep) if kept]
    return {'available': True, 'floor_index': 0, 'map_resolution_m': resolution,
        'native_floor_heights_m': heights.tolist(), 'requested_region': region,
        'enumeration_bounds_xy_m': domain.tolist(), 'complete_requested_room': requested_room and bounds_xy_m is None,
        'raw_free_cell_count': len(points), 'world_xy_m': points.tolist(), 'queried_rooms': rooms,
        'point_room_alignment': 'queried_rooms[i] is the actual native room query for world_xy_m[i]',
        'candidate_truncation_applied': False}, domain


def _objects(session, region, excluded, domain):
    objects = session.env.scene.objects
    objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
    rows = []
    for obj in objects:
        row = {'name': getattr(obj, 'name', None), 'category': getattr(obj, 'category', None),
            'model': getattr(obj, 'model', None), 'excluded_by_argument': getattr(obj, 'name', None) in excluded}
        try:
            low, high = [_array(value).astype(float) for value in obj.aabb]
            if low.shape != (3,) or high.shape != (3,) or np.any(high < low):
                raise ValueError('Native object AABB is invalid.')
            if domain is None:
                # Do not accidentally dump an entire building when raw scope
                # failed. Named room membership remains explicit evidence.
                if not region or region == 'full_scene' or region not in (getattr(obj, 'in_rooms', ()) or ()):
                    continue
            elif not np.all((low[:2] <= domain[1]) & (high[:2] >= domain[0])):
                continue
            centre = (low + high) / 2
            row.update(available=True, aabb_m=[low.tolist(), high.tolist()],
                visual_only=bool(obj.visual_only), visible=bool(obj.visible), fixed_base=bool(obj.fixed_base),
                native_in_rooms=list(getattr(obj, 'in_rooms', ()) or ()),
                queried_centre_room=session.env.scene.seg_map.get_room_instance_by_point(centre[:2]))
        except Exception as error:
            row.update(available=False, reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    return {'available': True, 'rows': rows, 'scope': 'native AABBs intersecting diagnostic XY domain; unreadable objects retained'}


def support_space_diagnostic(session, robot, region, excluded, *, bounds_xy_m=None):
    """Collect raw room cells, native geometry and the unchanged circle screen.

    This is a snapshot diagnostic, not a support proposal or a qualification.
    Explicit bounds can restrict a room snapshot; its completeness flag then
    remains false. With no room, the existing navigation API cannot safely
    return a bounded circle screen, so that reading stays unavailable.
    """
    excluded = set(excluded)
    value = {'schema': SCHEMA, 'requested_region': region, 'excluded_object_names': sorted(excluded),
        'native_reset_id': getattr(session, 'native_reset_id', None)}
    domain = None
    try:
        value['native_time_s'] = float(session.pipeline.og.sim.current_time)
    except Exception as error:
        value['native_time_s'] = None
        value['native_time_reason'] = f'{type(error).__name__}: {error}'
    try:
        value['raw_floor'], domain = _raw_floor(session, region, bounds_xy_m)
    except Exception as error:
        value['raw_floor'] = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    try:
        value['native_objects'] = _objects(session, region, excluded, domain)
    except Exception as error:
        value['native_objects'] = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    try:
        from .support_sampling import chassis_footprint
        name, extent, radius = chassis_footprint(robot)
        low, high = [_array(part).tolist() for part in robot.links[name].aabb]
        value['native_chassis'] = {'available': True, 'link_name': name, 'aabb_m': [low, high],
            'extent_m': _array(extent).tolist(), 'existing_circumscribed_screen_radius_m': float(radius),
            'robot_aabb_m': [_array(part).tolist() for part in robot.aabb]}
    except Exception as error:
        value['native_chassis'] = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    try:
        if not region or region == 'full_scene':
            raise ValueError('Existing navigation API has no bounded circle-screen domain without a requested room.')
        if not value['raw_floor']['available']:
            raise ValueError('Native raw room-map evidence is unavailable; circle screen was not run.')
        from .interaction_setup import _navigation_cells
        points, height, grid = _navigation_cells(session, robot, region, excluded)
        points = _array(points).astype(float)
        rooms = [session.env.scene.seg_map.get_room_instance_by_point(point) for point in points]
        if bounds_xy_m is not None:
            bounds = _bounds(bounds_xy_m)
            keep = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
            points = points[keep]
            rooms = [room for room, kept in zip(rooms, keep) if kept]
        value['existing_circle_screen'] = {'available': True, 'world_xy_m': points.tolist(),
            'queried_rooms': rooms, 'candidate_count': len(points), 'floor_z_m': float(height),
            'native_screen_details': grid}
    except Exception as error:
        value['existing_circle_screen'] = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    return value

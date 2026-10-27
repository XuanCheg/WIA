"""Read native headroom rays for explicit revised tabletop placement proposals.

Run: ``python -B -m pytest -q tests/test_task_edit2_support_headroom.py
tests/test_task_edit2_tabletop.py`` in the behavior environment. Call on the
simulator owner/main thread before a placement trial. This sampled rejection
screen never certifies full collision clearance, fit, stability, grasp or paths.
No actor, sensor, observation, physics or render state is changed.
"""
from __future__ import annotations

from collections.abc import Mapping
import threading

import numpy as np

SCHEMA = 'projected_native_headroom_proposal_v1'
PROBE_ENDPOINT_EXCLUSION_M = .002
ASSET_BOTTOM_ENDPOINT_INSET_M = .001


def native_headroom_ray(start, end, *, only_closest=False):
    """Use the existing read-only OG all-hit collision ray wrapper."""
    from .native_floor_contacts import native_vertical_ray
    return native_vertical_ray(start, end, only_closest=only_closest)


def _body_under(body, root):
    return body == root or body.startswith(root + '/')


def _vector(value):
    from .interaction_setup import array
    result = array(value).copy()
    if result.shape != (3,):
        raise ValueError('Native/projected geometry requires finite XYZ vectors.')
    return result


def _bounds(value):
    low, high = (_vector(part) for part in value)
    if np.any(high <= low) or not np.isfinite(high - low).all():
        raise ValueError('Native/projected AABB requires positive finite extents.')
    return low, high


def _actor(obj):
    low, high = _bounds(obj.aabb)
    return {'name': str(obj.name), 'prim_path': str(obj.prim_path),
        'category': str(obj.category), 'model': str(obj.model) if getattr(obj, 'model', None) is not None else None,
        'native_bounds_m_provenance': [low.tolist(), high.tolist()]}


def measure_projected_headroom(session, support, projected_bounds, *, surface_z_m, excluded_names=()):
    """Screen exact world AABBs ``{actual_native_name: (low_xyz, high_xyz)}``.

    Nine rays per object include XY corners, edge midpoints and centre. They run
    from highZ+.002 down to max(deckZ+.002, lowZ+.001). A nonignored native hit
    strictly inside that ray interval and within the object's closed vertical
    volume is a definite obstruction, regardless of its surface normal.
    Only exact roots resolved from explicitly excluded loaded actor names and
    the actual ``env.robots[0]`` root are ignored, including their descendants.
    The selected support can never be excluded; its own shelves/doors count.
    """
    result = {'schema': SCHEMA, 'available': False, 'accepted': False,
        'scope': 'sampled_proposal_rejection_only_not_full_collision_fit_grasp_or_path_certificate',
        'surface_z_m': None, 'support': None, 'provenance': None, 'actors': {},
        'excluded_actor_roots': [], 'actual_robot_root': None, 'objects': {}, 'obstructions': [],
        'sample_coverage': {'pattern': '3x3_xy_including_four_corners_edge_midpoints_and_centre',
            'rays_per_object': 9, 'expected_ray_count': None, 'completed_ray_count': 0,
            'probe_endpoint_exclusion_m': PROBE_ENDPOINT_EXCLUSION_M,
            'asset_bottom_endpoint_inset_m': ASSET_BOTTOM_ENDPOINT_INSET_M},
        'reason': None}
    try:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Native headroom reads require the simulator owner/main thread.')
        from .interaction_setup import _objects
        from .native_floor_contacts import _provenance
        result['provenance'] = _provenance(session)
        surface = float(surface_z_m)
        if not np.isfinite(surface):
            raise ValueError('Measured support surface is nonfinite.')
        result['surface_z_m'] = surface
        if not isinstance(projected_bounds, Mapping) or not projected_bounds:
            raise ValueError('Projected bounds require a nonempty native-name mapping.')
        objects = _objects(session.env.scene)
        by_name = {}
        for obj in objects:
            by_name.setdefault(str(obj.name), []).append(obj)
        def resolve(name):
            matches = by_name.get(name, [])
            if len(matches) != 1:
                raise ValueError('Native actor name must resolve exactly once: ' + name)
            return matches[0]
        if resolve(str(support.name)) is not support:
            raise ValueError('Selected support is not the actual loaded native object.')
        result['support'] = _actor(support)
        if not result['support']['prim_path'].startswith('/'):
            raise ValueError('Selected native support prim path is unavailable.')
        names = [excluded_names] if isinstance(excluded_names, str) else sorted(set(excluded_names))
        ignored_roots = []
        for name in names:
            obj = resolve(str(name))
            root = str(obj.prim_path)
            support_root = result['support']['prim_path']
            if obj is support or _body_under(root, support_root) or _body_under(support_root, root):
                raise ValueError('Selected support and its parents/parts must not be excluded.')
            if not root.startswith('/'):
                raise ValueError('Excluded native actor prim path is unavailable.')
            result['excluded_actor_roots'].append({'name': str(obj.name), 'prim_path': root})
            ignored_roots.append(root)
        robot = session.env.robots[0]
        robot_root = str(robot.prim_path)
        if not robot_root.startswith('/'):
            raise ValueError('Actual native robot prim path is unavailable.')
        if _body_under(result['support']['prim_path'], robot_root):
            raise ValueError('Selected support cannot be inside the ignored native robot root.')
        result['actual_robot_root'] = robot_root
        ignored_roots.append(robot_root)
        boxes = {}
        for name, value in projected_bounds.items():
            if not isinstance(name, str):
                raise ValueError('Projected bound keys must be actual native object names.')
            result['actors'][name] = _actor(resolve(name))
            boxes[name] = _bounds(value)
        result['sample_coverage']['expected_ray_count'] = 9 * len(boxes)
        for name, (low, high) in boxes.items():
            start_z = float(high[2] + PROBE_ENDPOINT_EXCLUSION_M)
            end_z = float(max(surface + PROBE_ENDPOINT_EXCLUSION_M, low[2] + ASSET_BOTTOM_ENDPOINT_INSET_M))
            if not start_z > end_z or not np.isfinite([start_z, end_z]).all():
                raise ValueError('Projected object has no valid vertical probe interval: ' + name)
            object_row = {'projected_bounds_m': [low.tolist(), high.tolist()], 'samples': []}
            result['objects'][name] = object_row
            for x in np.linspace(low[0], high[0], 3):
                for y in np.linspace(low[1], high[1], 3):
                    start, end = [float(x), float(y), start_z], [float(x), float(y), end_z]
                    sample = {'xy_m': [float(x), float(y)], 'ray_start_m': start, 'ray_end_m': end,
                        'hits': [], 'obstruction': False}
                    object_row['samples'].append(sample)
                    hits = native_headroom_ray(start, end, only_closest=False)
                    if not isinstance(hits, (list, tuple)):
                        raise TypeError('Native all-hit ray API did not return a hit list.')
                    for hit in hits:
                        row = {'hit': bool(hit.get('hit'))}
                        if row['hit']:
                            body = str(hit['rigidBody'])
                            if not body.startswith('/'):
                                raise ValueError('Native ray hit rigid body path is unavailable.')
                            point, normal = _vector(hit['position']), _vector(hit['normal'])
                            ignored = next((root for root in ignored_roots if _body_under(body, root)), None)
                            in_interval = bool(end_z < point[2] < start_z and surface < point[2] and low[2] <= point[2] <= high[2])
                            blocked = bool(ignored is None and in_interval)
                            row.update(body=body, collision=str(hit['collision']) if hit.get('collision') is not None else None,
                                position_m=point.tolist(), normal=normal.tolist(), ignored_root=ignored,
                                within_projected_vertical_probe=in_interval, obstruction=blocked)
                            if blocked:
                                sample['obstruction'] = True
                                result['obstructions'].append({'object_name': name, 'sample_xy_m': [float(x), float(y)], **row})
                        sample['hits'].append(row)
                    result['sample_coverage']['completed_ray_count'] += 1
        result.update(available=True, accepted=not result['obstructions'],
            reason='native_sampled_headroom_obstruction' if result['obstructions'] else 'no_sampled_headroom_obstruction')
    except Exception as error:
        result.update(available=False, accepted=False, reason='native_headroom_measurement_unavailable',
            exception_type=type(error).__name__, error=f'{type(error).__name__}: {error}')
    return result

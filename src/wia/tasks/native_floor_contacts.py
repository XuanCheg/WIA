"""Measure wheel support against actual local native floor surfaces, read only.

Run CPU tests with ``python -B -m pytest -q
tests/test_task_edit2_native_floor_contacts.py`` in the behavior environment.
Call on the simulator owner thread. Nominal map heights and floor AABBs are
provenance only: support requires a native wheel contact and same-body ray hit.
No observations, physics, rendering, clocks or poses are changed.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def native_vertical_ray(start, end, *, only_closest=False):
    """Read installed OG collision rays; exposed for CPU protocol tests."""
    from omnigibson.utils.sampling_utils import raytest
    return raytest(start, end, only_closest=only_closest)


def _error(error):
    return type(error).__name__ + ': ' + str(error)


def _provenance(session):
    from wia.tasks.loader import LoadedScene, LoadedTask, LoadedRigidTask
    loaded = getattr(session, 'loaded_task', None)
    adapted = getattr(getattr(session, 'scene_adapter', None), 'task', None)
    task = loaded if isinstance(loaded, (LoadedTask, LoadedRigidTask)) else (
        adapted if isinstance(adapted, (LoadedTask, LoadedRigidTask)) else None)
    scene = adapted if isinstance(adapted, LoadedScene) else loaded if isinstance(loaded, LoadedScene) else None
    result = {'task_id': str(task.definition.id) if task is not None else None,
        'task_sha256': str(task.file_sha256) if task is not None else None,
        'scene_id': str(scene.definition.id) if scene is not None else None,
        'scene_sha256': str(scene.file_sha256) if scene is not None else None,
        'source_sha256': str((task or scene).source_sha256) if task is not None or scene is not None else None,
        'source_path': str((task or scene).source_path) if task is not None or scene is not None else None,
        'seed': int((task or scene).seed) if task is not None or scene is not None else None,
        'reset_id': getattr(session, 'native_reset_id', None),
        'episode_id': getattr(session, 'episode_id', None)}
    try:
        time = float(session.pipeline.og.sim.current_time)
        if not np.isfinite(time):
            raise ValueError('Non-finite native clock.')
        result['native_time_s'] = time
    except Exception as error:
        result['native_time_s'] = None
        result['native_time_error'] = _error(error)
    for key in ('reset_id', 'episode_id'):
        if result[key] is not None:
            result[key] = str(result[key])
    return result


def _vector(value, array):
    value = array(value)
    if value.shape != (3,):
        raise ValueError('Native vector must have exactly three finite coordinates.')
    return value


def _ray_evidence(position, body, root, robot_root, array, same_body):
    start, end = position.copy(), position.copy()
    start[2] += .05
    end[2] -= .05
    result = {'ray_start_m': start.tolist(), 'ray_end_m': end.tolist(), 'ray_hits': [],
        'floor_verified': None, 'local_surface_z_m': None, 'contact_surface_offset_m': None}
    try:
        hits = native_vertical_ray(start.tolist(), end.tolist(), only_closest=False)
        if not isinstance(hits, (list, tuple)):
            raise TypeError('All-hit native ray API did not return a hit list.')
        candidates = []
        for hit in hits:
            row = {'hit': bool(hit.get('hit'))}
            if row['hit']:
                hit_body = str(hit['rigidBody'])
                hit_position = _vector(hit['position'], array)
                hit_normal = _vector(hit['normal'], array)
                row.update(body=hit_body, collision=str(hit['collision']) if hit.get('collision') is not None else None,
                    position_m=hit_position.tolist(), normal=hit_normal.tolist(),
                    robot_self_hit=same_body(robot_root, hit_body),
                    same_contact_floor_body=hit_body == body,
                    same_contact_floor_root=same_body(root, hit_body))
                # The root establishes loaded identity; the hit must resolve the
                # actual contacted rigid body, not a sibling floor or furniture.
                if hit_body == body and same_body(root, hit_body) and hit_normal[2] >= .95:
                    candidates.append(hit_position)
            result['ray_hits'].append(row)
        if not candidates:
            result.update(floor_verified=False, reason='same_contact_floor_upward_surface_missing')
            return result
        surface = min(candidates, key=lambda point: abs(float(point[2] - position[2])))
        offset = float(position[2] - surface[2])
        result.update(local_surface_z_m=float(surface[2]), contact_surface_offset_m=offset,
            floor_verified=bool(abs(offset) <= .005), reason='verified_same_body_native_floor_surface'
            if abs(offset) <= .005 else 'native_contact_local_surface_mismatch')
    except Exception as error:
        result.update(floor_verified=None, reason='native_floor_ray_unavailable', error=_error(error))
    return result


def measure_native_floor_contacts(session, robot, floor_paths, *, nominal_floor_z=None):
    """Return three-valued native wheel-support and external-collision evidence.

    ``floor_supported`` / ``collision_free`` are bool or None (unavailable).
    ``measurement_available`` means both conclusions are measured, including a
    measured failure. Callers must require all three fields to be True for a
    positive gate. Any unavailable floor-contact ray prevents positive support;
    any contradictory floor-contact ray yields False. At least one exact native
    floor-touching wheel link contact is required; torso/self contact is not it.
    """
    # Lazy reuse avoids importing interaction_setup while its callers initialize.
    from .interaction_setup import array, _contact_parts, _objects
    from .support_sampling import same_body

    result = {'schema': 'native_floor_contacts_v1', 'floor_supported': None,
        'collision_free': None, 'measurement_available': False, 'floor_contact_positions_m': [],
        'unexpected_contact_bodies': [], 'contacts': [], 'floor_inventory': [],
        'wheel_link_paths': [], 'supported_wheel_link_paths': [],
        'nominal_floor_z_m': None, 'provenance': _provenance(session),
        'tolerances': {'minimum_abs_contact_normal_z': .5, 'minimum_floor_ray_normal_z': .95,
            'contact_surface_tolerance_m': .005, 'ray_half_span_m': .05}, 'errors': []}
    if nominal_floor_z is not None:
        try:
            nominal = float(nominal_floor_z)
            if not np.isfinite(nominal):
                raise ValueError('Non-finite nominal floor height.')
            result['nominal_floor_z_m'] = nominal
        except Exception as error:
            result['errors'].append('Nominal provenance: ' + _error(error))
    roots = [str(path) for path in ((floor_paths,) if isinstance(floor_paths, str) else floor_paths)]
    result['requested_floor_roots'] = roots
    try:
        robot_root = str(robot.prim_path)
        if not robot_root.startswith('/'):
            raise ValueError('Native robot prim path unavailable.')
        result['robot_prim_path'] = robot_root
        objects = _objects(session.env.scene)
        verified_roots = set()
        for root in roots:
            matches = [obj for obj in objects if str(obj.prim_path) == root]
            if len(matches) != 1:
                raise ValueError('Supplied floor root does not uniquely identify a loaded native object: ' + root)
            obj = matches[0]
            row = {'name': str(obj.name), 'prim_path': root, 'category': str(obj.category),
                'visual_only': bool(obj.visual_only), 'collision_reference': 'native same-body ray required'}
            try:
                row['aabb_m_provenance_only'] = [_vector(value, array).tolist() for value in obj.aabb]
            except Exception as error:
                row['aabb_error'] = _error(error)
            result['floor_inventory'].append(row)
            if obj.category not in ('floor', 'floors') or obj.visual_only:
                raise ValueError('Supplied root is not a collidable loaded floor: ' + root)
            verified_roots.add(root)
        if not verified_roots:
            raise ValueError('No actual loaded floor roots supplied.')
        floor_inventory_available = True
    except Exception as error:
        result['errors'].append('Floor identity: ' + _error(error))
        floor_inventory_available = False
        verified_roots = set()
        robot_root = str(getattr(robot, 'prim_path', ''))
    try:
        names = robot.floor_touching_base_link_names
        if not names:
            raise ValueError('Native floor-touching wheel names unavailable.')
        if not isinstance(robot.links, Mapping):
            raise TypeError('Native robot.links is not a mapping.')
        wheel_paths = {str(robot.links[name].prim_path) for name in names}
        if any(not same_body(robot_root, path) or path == robot_root for path in wheel_paths):
            raise ValueError('Wheel prim paths are not exact installed robot link paths.')
        result['wheel_link_paths'] = sorted(wheel_paths)
        wheels_available = True
    except Exception as error:
        result['errors'].append('Wheel identity: ' + _error(error))
        wheel_paths, wheels_available = set(), False
    try:
        contacts = list(robot.contact_list())
    except Exception as error:
        result.update(reason='native_contact_api_unavailable')
        result['errors'].append(_error(error))
        return result
    uncertain_collision = not floor_inventory_available
    uncertain_support = not floor_inventory_available or not wheels_available
    contradictory_floor, supported = False, set()
    for contact in contacts:
        bodies = _contact_parts(contact)
        row = {'bodies': list(bodies), 'floor_verified': None}
        try:
            position, normal = _vector(contact.position, array), _vector(contact.normal, array)
            row.update(position_m=position.tolist(), normal=normal.tolist())
        except Exception as error:
            row['contact_geometry_error'] = _error(error)
        result['contacts'].append(row)
        owns = [same_body(robot_root, body) for body in bodies] if robot_root else [False, False]
        if all(owns):
            row.update(classification='self', reason='robot_self_contact')
            continue
        if not all(body.startswith('/') for body in bodies) or not any(owns):
            row.update(classification='unavailable', reason='native_contact_bodies_not_robot_external_pair')
            uncertain_collision = uncertain_support = True
            continue
        external = [body for body, own in zip(bodies, owns) if not own]
        row.update(robot_body=next(body for body, own in zip(bodies, owns) if own), external_bodies=external)
        floor_roots = [root for root in roots if all(same_body(root, body) for body in external)]
        if not floor_roots:
            row.update(classification='unexpected_external', reason='external_body_not_supplied_floor')
            result['unexpected_contact_bodies'].extend(external)
            continue
        root = max(floor_roots, key=len)
        row.update(classification='floor', floor_root=root)
        wheel = row['robot_body'] if row['robot_body'] in wheel_paths else None
        row['wheel_link_path'] = wheel
        if root not in verified_roots:
            row['reason'] = 'loaded_collidable_floor_identity_unavailable'
            uncertain_support = True
            continue
        try:
            if 'contact_geometry_error' in row:
                raise ValueError(row['contact_geometry_error'])
            position, normal = array(row['position_m']), array(row['normal'])
            row.update(nominal_floor_offset_m=float(position[2] - result['nominal_floor_z_m'])
                if result['nominal_floor_z_m'] is not None else None)
        except Exception as error:
            row.update(reason='native_contact_geometry_unavailable', error=_error(error))
            uncertain_support = True
            continue
        row.update(_ray_evidence(position, external[0], root, robot_root, array, same_body))
        if abs(float(normal[2])) < .5:
            row.update(floor_verified=False, reason='native_contact_normal_not_supporting')
        if row['floor_verified'] is None:
            uncertain_support = True
        elif not row['floor_verified']:
            contradictory_floor = True
        elif wheel:
            supported.add(wheel)
            result['floor_contact_positions_m'].append(position.tolist())
        else:
            row['reason'] = 'floor_contact_is_not_exact_native_wheel_body'
    result['unexpected_contact_bodies'] = sorted(set(result['unexpected_contact_bodies']))
    result['supported_wheel_link_paths'] = sorted(supported)
    result['collision_free'] = False if result['unexpected_contact_bodies'] else None if uncertain_collision else True
    result['floor_supported'] = False if contradictory_floor else None if uncertain_support else bool(supported)
    result['measurement_available'] = result['floor_supported'] is not None and result['collision_free'] is not None
    result['reason'] = ('unexpected_nonfloor_contact' if result['unexpected_contact_bodies'] else
        'contradictory_native_floor_geometry' if contradictory_floor else
        'native_support_measurement_unavailable' if not result['measurement_available'] else
        'verified_native_wheel_floor_support' if result['floor_supported'] else
        'no_verified_native_wheel_floor_contact')
    return result

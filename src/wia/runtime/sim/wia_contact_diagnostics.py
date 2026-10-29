"""Read pair-local native contact diagnostics without changing physics.

Run CPU checks: python -m pytest tests/test_wia_contact_pair_diagnostics.py
In an already initialized OmniGibson session, call measure_contact_pair(a, b).
Only existing public contact/state getters and existing cache reads are used.
Empty contact samples, including buffer truncation, never imply geometric
separation or absence of penetration. No view is initialized or recreated.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value):
    array = _array(value)
    if array.size != 1 or array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError('Native measurement must be one finite numeric scalar.')
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError('Native scalar cannot be represented as a finite JSON number.')
    return scalar


def _integer(value, *, nonnegative=False):
    scalar = _scalar(value)
    if scalar != int(scalar) or (nonnegative and scalar < 0):
        raise ValueError('Native index/count must be an integer in its declared range.')
    return int(scalar)


def _numeric(value, *, size=None):
    array = _array(value)
    if array.dtype.kind not in 'iuf' or not np.isfinite(array).all() or (size is not None and array.size != size):
        raise ValueError('Native sample has invalid numeric geometry.')
    converted = array.astype(float)
    if not np.isfinite(converted).all():
        raise ValueError('Native sample cannot be represented as finite JSON numbers.')
    return converted.tolist()


def _error(error, **fields):
    return {'available': False, **fields, 'error': f'{type(error).__name__}: {error}'}


def _sleep_getters(native):
    """Read the actual installed property getters; never call a sleep backend."""
    result = {}
    for name in ('sleep_threshold', 'is_asleep'):
        try:
            value = getattr(native, name)
            if name == 'is_asleep':
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError('Native is_asleep getter did not return a boolean.')
                value = bool(value)
            else:
                value = _scalar(value)
            result[name] = {'available': True, 'native_getter': name, 'value': value}
        except Exception as error:
            result[name] = _error(error, native_getter=name)
    return result


def _link_indices(api, path):
    result = {}
    for label, getter in (('scene', 'get_scene_idx'), ('row', 'get_body_row_idx'), ('column', 'get_body_col_idx')):
        try:
            if api is None:
                raise RuntimeError('Native RigidContactAPI is unavailable.')
            value = getattr(api, getter)(path)
            if label == 'scene':
                result[label] = {'available': True, 'scene_idx': _integer(value, nonnegative=True)}
            else:
                scene, index = value
                result[label] = {'available': True, 'scene_idx': _integer(scene, nonnegative=True),
                                 'index': _integer(index, nonnegative=True)}
        except Exception as error:
            result[label] = _error(error, native_getter=getter)
    return result


def _object_inventory(native, api):
    result = {'object_name': None, 'object_prim_path': None, 'links': [], 'errors': [],
              'sleep_getters': _sleep_getters(native)}
    for output, attribute in (('object_name', 'name'), ('object_prim_path', 'prim_path')):
        try:
            value = getattr(native, attribute)
            if not isinstance(value, str) or not value:
                raise ValueError('Native object identity must be a nonempty string.')
            result[output] = value
        except Exception as error:
            result['errors'].append(f'{attribute}: {type(error).__name__}: {error}')
    try:
        links = native.links
        if not isinstance(links, Mapping) or not links:
            raise ValueError('Native object has no readable link mapping.')
        for name, link in links.items():
            path = link.prim_path
            if not isinstance(path, str) or not path:
                raise ValueError('Native link prim path is unavailable.')
            result['links'].append({'link_name': str(name), 'link_prim_path': path,
                'indices': _link_indices(api, path), 'sleep_getters': _sleep_getters(link)})
    except Exception as error:
        result['errors'].append(f'links: {type(error).__name__}: {error}')
    return result


def _native_touching(first, second):
    try:
        from omnigibson.object_states import Touching
        value = first.states[Touching].get_value(second)
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('Native Touching getter did not return a boolean.')
        return {'available': True, 'native_getter': 'states[Touching].get_value', 'value': bool(value)}
    except Exception as error:
        return _error(error, native_getter='states[Touching].get_value')


def _contact_samples(api, scene, row_path, column_path):
    try:
        contacts = api.get_contact_data(scene, row_prim_paths=[row_path], column_prim_paths=[column_path])
        samples = []
        for sample in contacts:
            row, column, force, point, normal, separation = sample
            if row != row_path or column != column_path:
                raise ValueError('Native getter returned a contact outside the requested link pair.')
            if _array(force).size not in (1, 3):
                raise ValueError('Native pair force sample must be a scalar or three-vector.')
            samples.append({'row_link_prim_path': row_path, 'column_link_prim_path': column_path,
                'force_native': _numeric(force), 'point_m': _numeric(point, size=3),
                'normal': _numeric(normal, size=3), 'separation_m': _scalar(separation)})
        separations = [sample['separation_m'] for sample in samples]
        return {'available': True, 'sample_count': len(samples), 'separations_m': separations,
            'minimum_separation_m': min(separations) if separations else None, 'samples': samples}
    except Exception as error:
        return _error(error, sample_count=None, separations_m=None, minimum_separation_m=None)


def _contact_buffer_capacity(api):
    try:
        return {'available': True, 'value': _integer(api.get_max_contact_data_count(), nonnegative=True),
                'native_getter': 'RigidContactAPI.get_max_contact_data_count'}
    except Exception as error:
        return _error(error, native_getter='RigidContactAPI.get_max_contact_data_count')


def _pair_contact_buffers(data, row, column, capacity):
    """Validate identical pair-local slices for cached and direct native buffers."""
    try:
        forces, points, normals, separations, counts, starts = data
        buffers = {'forces': forces, 'points': points, 'normals': normals, 'separations': separations}
        lengths = {key: len(value) for key, value in buffers.items()}
        count = _integer(counts[row, column], nonnegative=True)
        start = _integer(starts[row, column])
        result = {'available': True, 'contact_counts': count, 'start_indices': start,
            'row_index': row, 'column_index': column, 'configured_buffer_capacity': capacity,
            'buffer_lengths': lengths}
        if count == 0:
            result.update(pair_slice_available=True, pair_sample_count=0, pair_separations_m=[],
                          pair_minimum_separation_m=None)
        elif start < 0 or any(start + count > length for length in lengths.values()):
            result.update(pair_slice_available=False, pair_sample_count=None, pair_separations_m=None,
                pair_minimum_separation_m=None, slice_error='Native pair indices exceed an existing contact-data buffer.')
        else:
            values = [_scalar(value) for value in separations[start:start + count]]
            result.update(pair_slice_available=True, pair_sample_count=len(values),
                pair_separations_m=values, pair_minimum_separation_m=min(values))
        return result
    except Exception as error:
        return _error(error, configured_buffer_capacity=capacity)


def _raw_pair_cache(api, scene, row, column):
    """Inspect only this pair's raw indices and slice in an existing cache."""
    capacity = _contact_buffer_capacity(api)
    try:
        cache = api._CONTACT_DATA
        if scene not in cache:
            return {'available': False, 'reason': 'native_contact_data_cache_not_populated',
                    'configured_buffer_capacity': capacity}
        return _pair_contact_buffers(cache[scene], row, column, capacity)
    except Exception as error:
        return _error(error, configured_buffer_capacity=capacity)


def _direct_native_contact_view(api, directions):
    """Read each existing scene view once, without replacing the public cache."""
    result = {'available': False,
        'native_getter': 'RigidContactAPI._CONTACT_VIEW[scene_idx].get_contact_data(dt=1.0)',
        'public_contact_cache_written_by_helper': False, 'empty_samples_establish_clearance': False,
        'scenes': {}, 'unavailable_pairs': []}
    requested = {}
    for direction, measured in directions.items():
        for pair in measured['link_pairs']:
            identity = {'direction': direction, 'row_link_prim_path': pair['row_link_prim_path'],
                'column_link_prim_path': pair['column_link_prim_path'],
                'scene_idx': pair.get('scene_idx'), 'row_index': pair.get('row_index'),
                'column_index': pair.get('column_index')}
            if (identity['scene_idx'] is None or identity['row_index'] is None or identity['column_index'] is None
                    or pair.get('column_scene_idx', identity['scene_idx']) != identity['scene_idx']):
                result['unavailable_pairs'].append({**identity, 'available': False,
                    'reason': 'native_pair_indices_or_common_scene_unavailable'})
                continue
            requested.setdefault(identity['scene_idx'], []).append(identity)
    for scene, pairs in requested.items():
        entry = {'scene_idx': scene, 'direct_getter_calls': 0, 'link_pairs': []}
        result['scenes'][str(scene)] = entry
        try:
            views = getattr(api, '_CONTACT_VIEW', None)
            if views is None or scene not in views or views[scene] is None:
                entry.update(available=False, reason='existing_native_contact_view_missing')
                continue
            view = views[scene]
            getter = view.get_contact_data
            entry['direct_getter_calls'] = 1
            data = getter(dt=1.0)
            capacity = _contact_buffer_capacity(api)
            entry['link_pairs'] = [{**identity, **_pair_contact_buffers(
                data, identity['row_index'], identity['column_index'], capacity)} for identity in pairs]
            entry['available'] = all(pair['available'] for pair in entry['link_pairs'])
        except Exception as error:
            entry.update(_error(error))
        finally:
            if not entry.get('available') and not entry['link_pairs']:
                entry['link_pairs'] = [{**identity, 'available': False,
                    'reason': entry.get('reason', 'existing_native_contact_view_read_failed')} for identity in pairs]
    result['available'] = (bool(result['scenes']) and not result['unavailable_pairs']
        and all(entry['available'] for entry in result['scenes'].values()))
    if not requested:
        result['reason'] = 'no_requested_link_pair_has_usable_same_scene_indices'
    return result


def _direction_pair(api, row_link, column_link):
    result = {'row_link_prim_path': row_link['link_prim_path'],
              'column_link_prim_path': column_link['link_prim_path']}
    row, column = row_link['indices']['row'], column_link['indices']['column']
    if not row['available'] or not column['available']:
        result.update(available=False, scene_idx=row.get('scene_idx', column.get('scene_idx')),
            row_index=row.get('index'), column_index=column.get('index'),
            indices={'row': row, 'column': column},
            impulse={'available': False, 'reason': 'native_pair_indices_unavailable'},
            contact_data={'available': False, 'reason': 'native_pair_indices_unavailable'},
            raw_contact_data={'available': False, 'reason': 'native_pair_indices_unavailable'})
        return result
    scene, row_index, column_index = row['scene_idx'], row['index'], column['index']
    result.update(scene_idx=scene, row_index=row_index, column_index=column_index)
    if scene != column['scene_idx']:
        result.update(available=False, column_scene_idx=column['scene_idx'],
            error='Native row and column links belong to different scenes.',
            impulse={'available': False, 'reason': 'native_pair_scenes_differ'},
            contact_data={'available': False, 'reason': 'native_pair_scenes_differ'},
            raw_contact_data={'available': False, 'reason': 'native_pair_scenes_differ'})
        return result
    try:
        impulse = _array(api.get_impulses([row_link['link_prim_path']], [column_link['link_prim_path']]))
        if impulse.shape != (1, 1, 3) or not np.isfinite(impulse).all():
            raise ValueError('Native pair impulse must have finite shape (1, 1, 3).')
        vector = impulse[0, 0].astype(float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm):
            raise ValueError('Native pair impulse norm is not finite.')
        result['impulse'] = {'available': True, 'vector_native': vector.tolist(),
                             'norm_native': norm}
    except Exception as error:
        result['impulse'] = _error(error)
    result['contact_data'] = _contact_samples(api, scene, row_link['link_prim_path'], column_link['link_prim_path'])
    result['raw_contact_data'] = _raw_pair_cache(api, scene, row_index, column_index)
    result['available'] = result['impulse']['available'] and result['contact_data']['available']
    return result


def _geometry_read(getter, source):
    try:
        return {'available': True, 'native_getter': source, **getter()}
    except Exception as error:
        return _error(error, native_getter=source)


def _pose_values(values):
    position, quaternion = values
    position, quaternion = _array(position), _array(quaternion)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError('Native grasp pose requires position (3,) and XYZW quaternion (4,).')
    pose = {'position_m': _numeric(position), 'quaternion_xyzw': _numeric(quaternion)}
    norm = np.linalg.norm(quaternion.astype(float))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError('Native grasp pose has an invalid quaternion norm.')
    return pose


def _collision_boundary_bounds(link):
    points = _array(link.collision_boundary_points_world)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('Native collision boundary points require nonempty (N, 3) geometry.')
    _numeric(points)
    return {'aabb_m': [points.min(axis=0).astype(float).tolist(), points.max(axis=0).astype(float).tolist()],
            'point_count': len(points)}


def _finger_geometry(robot, arm):
    links = robot.finger_links[arm]
    if not links:
        raise ValueError('Native arm has no finger links.')
    rows = []
    for link in links:
        if any(not isinstance(value, str) or not value for value in (link.name, link.prim_path)):
            raise ValueError('Native finger identity is unavailable.')
        rows.append({'link_name': link.name, 'link_prim_path': link.prim_path,
            'pose': _geometry_read(lambda: _pose_values(link.get_position_orientation()), 'get_position_orientation'),
            'collision_boundary_aabb': _geometry_read(lambda: _collision_boundary_bounds(link),
                                                     'collision_boundary_points_world')})
    return {'links': rows, 'all_finger_geometry_available': all(
        row['pose']['available'] and row['collision_boundary_aabb']['available'] for row in rows)}


def _grasp_ray_endpoints(robot, arm):
    """Reproduce native endpoint math only, with no native raycast invocation."""
    sides = {}
    for side, mapping in (('start', robot.assisted_grasp_start_points), ('end', robot.assisted_grasp_end_points)):
        points = mapping[arm]
        if not points:
            raise ValueError('Native assisted-grasp endpoint definitions are unavailable.')
        sides[side] = []
        for point in points:
            if not isinstance(point.link_name, str) or not point.link_name:
                raise ValueError('Native assisted-grasp endpoint link name is unavailable.')
            link = robot.links[point.link_name]
            if not isinstance(link.prim_path, str) or not link.prim_path:
                raise ValueError('Native assisted-grasp endpoint link prim path is unavailable.')
            pose = _pose_values(link.get_position_orientation())
            local = _array(point.position)
            if local.shape != (3,):
                raise ValueError('Native assisted-grasp local endpoint requires shape (3,).')
            local = np.asarray(_numeric(local), dtype=float)
            quaternion = np.asarray(pose['quaternion_xyzw'], dtype=float)
            quaternion = quaternion / np.linalg.norm(quaternion)
            vector, scalar = quaternion[:3], quaternion[3]
            rotated = (2 * np.dot(vector, local) * vector
                + (scalar*scalar - np.dot(vector, vector)) * local + 2 * scalar * np.cross(vector, local))
            world = np.asarray(pose['position_m']) + rotated
            if side == 'end':
                world += 1e-8
            sides[side].append({'link_name': point.link_name, 'link_prim_path': str(link.prim_path),
                'local_position_m': local.tolist(), 'link_pose': pose, 'world_position_m': _numeric(world)})
    rays = [{'start_index': start_index, 'end_index': end_index,
             'start_world_m': start['world_position_m'], 'end_world_m': end['world_position_m']}
            for end_index, end in enumerate(sides['end']) for start_index, start in enumerate(sides['start'])]
    return {**sides, 'rays': rays, 'ray_count': len(rays), 'end_epsilon_m': 1e-8,
        'native_transform_source': 'ManipulationRobot._find_gripper_raycast_collisions; link-local quaternion '
            'rotation and translation, end epsilon 1e-8; endpoint math only, no raycast.'}


def _robot_grasp_geometry(robot):
    """Optionally expose native robot geometry without changing grasp state."""
    getters, values = {}, {}
    for name in ('arm_names', 'arms', 'finger_links', 'get_eef_pose'):
        try:
            values[name] = getattr(robot, name)
            getters[name] = {'available': True, 'native_getter': name}
        except Exception as error:
            getters[name] = _error(error, native_getter=name)
    if not values:
        return {'available': False, 'reason': 'native_robot_grasp_interface_unavailable', 'interface_getters': getters}
    arms, source = None, None
    for name in ('arm_names', 'arms', 'finger_links'):
        value = values.get(name)
        candidates = list(value) if isinstance(value, (Mapping, list, tuple)) else []
        if candidates and all(isinstance(arm, str) and arm for arm in candidates) and len(set(candidates)) == len(candidates):
            arms, source = candidates, name
            break
    if not arms:
        return {'available': False, 'reason': 'native_robot_arm_names_unavailable', 'interface_getters': getters}
    result = {'available': True, 'arm_names_source': source, 'interface_getters': getters, 'arms': {}}
    for arm in arms:
        def fingertips():
            lengths = robot.eef_to_fingertip_lengths[arm]
            if not isinstance(lengths, Mapping) or not lengths:
                raise ValueError('Native per-finger EEF distances are unavailable.')
            return {'value': {str(name): _scalar(value) for name, value in lengths.items()}}

        def held_object():
            held = robot._ag_obj_in_hand[arm]
            name = held.name if held is not None else None
            if held is not None and (not isinstance(name, str) or not name):
                raise ValueError('Native held-object name is unavailable.')
            return {'object_name': name}

        row = {
            'eef_pose': _geometry_read(lambda: _pose_values(robot.get_eef_pose(arm=arm)), 'get_eef_pose'),
            'eef_to_fingertip_lengths': _geometry_read(fingertips, 'eef_to_fingertip_lengths'),
            'finger_links': _geometry_read(lambda: _finger_geometry(robot, arm), 'finger_links'),
            'ag_obj_in_hand': _geometry_read(held_object, '_ag_obj_in_hand'),
            'is_grasping': _geometry_read(lambda: {'native_value': _integer(robot.is_grasping(arm=arm))}, 'is_grasping'),
            'grasp_ray_endpoints': _geometry_read(lambda: _grasp_ray_endpoints(robot, arm),
                                                  'assisted_grasp_start_points/assisted_grasp_end_points')}
        row['available'] = all(field['available'] for field in row.values()) and row['finger_links'].get(
            'all_finger_geometry_available', False)
        result['arms'][arm] = row
        result['available'] &= row['available']
    return result


def measure_contact_pair(first, second):
    """Return JSON-safe read-only forward/reverse native pair measurements.

    Each direction reports all declared native link pairs independently. Native
    Touching, impulses, returned contact samples and raw cache availability stay
    separate; none constitutes a scoring or penetration verdict. Public getters
    may populate their normal caches. Existing views/caches are never reset.
    Optional robot_grasp_geometry records native per-arm geometry and endpoint
    transforms only; its rays are not cast and do not establish grasp success.
    direct_native_contact_view reads each existing native view once separately
    from public cached getters, with no write to the public contact-data cache.
    """
    result = {'schema': 'wia_native_contact_pair_diagnostics_v1', 'read_only': True,
              'physics_steps': 0, 'actor_mutations': 0, 'directions': {}}
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI
        api = RigidContactAPI
    except Exception as error:
        api = None
        result['contact_api_error'] = f'{type(error).__name__}: {error}'
    first_inventory, second_inventory = _object_inventory(first, api), _object_inventory(second, api)
    result.update(first=first_inventory, second=second_inventory)
    result['robot_grasp_geometry'] = {'first': _robot_grasp_geometry(first), 'second': _robot_grasp_geometry(second)}
    for name, row_object, column_object, row_inventory, column_inventory in (
            ('first_to_second', first, second, first_inventory, second_inventory),
            ('second_to_first', second, first, second_inventory, first_inventory)):
        pairs = [_direction_pair(api, row, column) for row in row_inventory['links'] for column in column_inventory['links']]
        result['directions'][name] = {'row_object_prim_path': row_inventory['object_prim_path'],
            'column_object_prim_path': column_inventory['object_prim_path'],
            'native_touching': _native_touching(row_object, column_object), 'link_pairs': pairs,
            'available': bool(pairs) and not row_inventory['errors'] and not column_inventory['errors']
                         and all(pair['available'] for pair in pairs)}
    result['direct_native_contact_view'] = _direct_native_contact_view(api, result['directions'])
    return result

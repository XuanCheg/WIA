"""Read initial native camera mounts without capturing or changing the scene.

Run CPU checks with ``python -B -m pytest -q
tests/test_task_edit2_initial_camera_diagnostics.py`` in the behavior environment.
The caller invokes :func:`measure_initial_camera_mounts` on the simulator owner
thread. USD matrices are authored USD transforms at Default time, reported
separately from OG's runtime world/parent poses; they are not assumed identical.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _json(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        values = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [_json(item) for item in values]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('Non-finite native measurement.')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported measurement value: ' + type(value).__name__)


def _read(callback, source):
    try:
        value = callback()
        if value is None:
            raise ValueError('Native field is unavailable (None).')
        return {'available': True, 'value': _json(value), 'source': source}
    except Exception as error:
        return {'available': False, 'source': source,
            'error': type(error).__name__ + ': ' + str(error)}


def _pose(obj, frame):
    position, quaternion = obj.get_position_orientation(frame=frame)
    position, quaternion = np.asarray(_json(position), dtype=float), np.asarray(_json(quaternion), dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError('Expected finite native position XYZ and quaternion XYZW.')
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('Native quaternion has zero norm.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _usd_transform(prim, frame):
    # This is OG's installed USD API, not a synthesized robot-link transform.
    import omnigibson.lazy as lazy
    xform = lazy.pxr.UsdGeom.Xformable(prim)
    if frame == 'world':
        matrix = xform.ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
    else:
        matrix = xform.GetLocalTransformation(lazy.pxr.Usd.TimeCode.Default())
    return {'matrix4x4': [[float(matrix[i][j]) for j in range(4)] for i in range(4)],
        'convention': 'USD Gf.Matrix4d row-vector transform', 'time': 'Usd.TimeCode.Default'}


def _node(obj):
    return {'prim_path': _read(lambda: str(obj.prim_path), 'native prim_path'),
        'world_pose': _read(lambda: _pose(obj, 'world'), 'OG get_position_orientation(frame=world)'),
        'local_to_parent_pose': _read(lambda: _pose(obj, 'parent'), 'OG get_position_orientation(frame=parent)')}


def _provenance(session):
    from wia.tasks.loader import LoadedScene, LoadedTask, LoadedRigidTask
    explicit = getattr(session, 'loaded_task', None)
    adapter = getattr(getattr(session, 'scene_adapter', None), 'task', None)
    task = explicit if isinstance(explicit, (LoadedTask, LoadedRigidTask)) else (
        adapter if isinstance(adapter, (LoadedTask, LoadedRigidTask)) else None)
    scene = adapter if isinstance(adapter, LoadedScene) else explicit if isinstance(explicit, LoadedScene) else None

    def record(owner):
        if owner is None:
            return {'available': False, 'error': 'No role-verified loaded definition available.'}
        return {key: _read(callback, 'loaded definition') for key, callback in {
            'id': lambda: owner.definition.id, 'version': lambda: owner.definition.version,
            'file': lambda: owner.path, 'file_sha256': lambda: owner.file_sha256,
            'source_file': lambda: owner.source_path, 'source_sha256': lambda: owner.source_sha256,
            'seed': lambda: owner.seed, 'variant': lambda: owner.variant,
            'scene_name': lambda: owner.scene_name, 'room_name': lambda: owner.room_name}.items()}

    return {'task': record(task), 'scene': record(scene),
        'reset_id': _read(lambda: session.native_reset_id, 'session.native_reset_id'),
        'episode_id': _read(lambda: session.episode_id, 'session.episode_id')}


def _forward(sensor, world_pose):
    if sensor.prim.GetTypeName() != 'Camera':
        raise ValueError('USD Camera type is not established; no camera-axis convention applied.')
    if not world_pose['available']:
        raise ValueError('Native world camera pose unavailable.')
    q = np.asarray(world_pose['value']['quaternion_xyzw'], dtype=float)
    q /= np.linalg.norm(q)
    axis = np.asarray([0., 0., -1.])
    forward = axis + 2 * q[3] * np.cross(q[:3], axis) + 2 * np.cross(q[:3], np.cross(q[:3], axis))
    return {'vector_world': forward.tolist(), 'convention': 'USD Camera local -Z; native world quaternion XYZW'}


def measure_initial_camera_mounts(session, stage):
    """Return read-only, JSON-safe measurements of actual ``env.robots[0]``.

    No sensor observations, intrinsics, rendering, stepping or setters are used.
    Each field carries availability/error; an installed sensor with failed reads
    stays an installed sensor. Missing registries/expected head are separate.
    Eye/head selection uses actual link names only, and makes no mount claim.
    """
    result = {'schema': 'initial_camera_mount_diagnostic_v1', 'stage': str(stage),
        'provenance': _provenance(session), 'sensors': [], 'head_eye_links': []}
    try:
        robot = session.env.robots[0]
        if robot is None:
            raise ValueError('env.robots[0] is None.')
    except Exception as error:
        result['robot'] = {'available': False, 'error': type(error).__name__ + ': ' + str(error)}
        result['sensor_inventory'] = {'available': False, 'error': 'Actual robot unavailable.'}
        return result
    result['robot'] = {'available': True, 'source': 'session.env.robots[0]', **_node(robot),
        'name': _read(lambda: robot.name, 'native robot.name')}
    try:
        links = robot.links
        if not isinstance(links, Mapping):
            raise TypeError('Native robot.links is not a mapping.')
        result['link_inventory'] = {'available': True, 'names': [str(name) for name in links]}
    except Exception as error:
        links = {}
        result['link_inventory'] = {'available': False, 'error': type(error).__name__ + ': ' + str(error)}
    for name, link in links.items():
        if any(part in str(name).lower() for part in ('head', 'eye')):
            result['head_eye_links'].append({'name': str(name), 'selection': 'actual link name contains head or eye', **_node(link)})
    try:
        sensors = robot.sensors
        if not isinstance(sensors, Mapping):
            raise TypeError('Native robot.sensors is not a mapping.')
        result['sensor_inventory'] = {'available': True, 'installed_names': [str(name) for name in sensors], 'count': len(sensors)}
    except Exception as error:
        result['sensor_inventory'] = {'available': False, 'error': type(error).__name__ + ': ' + str(error)}
        return result
    requested = getattr(session, 'camera_name', None)
    result['requested_head_sensor'] = {'name': requested, 'status': 'unspecified' if requested is None else
        'installed' if requested in sensors else 'missing'}
    for name, sensor in sensors.items():
        row = {'registry_name': str(name), 'status': 'installed', **_node(sensor),
            'name': _read(lambda: sensor.name, 'native sensor.name'),
            'usd_type': _read(lambda: sensor.prim.GetTypeName(), 'USD prim.GetTypeName')}
        parent = None
        try:
            parent = sensor.prim.GetParent()
            if not parent.IsValid():
                raise ValueError('USD parent prim is invalid.')
            row['parent_prim_path'] = _read(lambda: str(parent.GetPath()), 'USD prim.GetParent().GetPath')
            row['parent_usd_world_transform'] = _read(lambda: _usd_transform(parent, 'world'), 'USD Xformable')
            row['parent_usd_local_transform'] = _read(lambda: _usd_transform(parent, 'parent'), 'USD Xformable')
            matches = []
            for link_name, link in links.items():
                if str(link.prim_path) == str(parent.GetPath()):
                    matches.append({'name': str(link_name), **_node(link)})
            row['actual_parent_links'] = matches
        except Exception as error:
            row['parent_measurement_error'] = type(error).__name__ + ': ' + str(error)
        row['usd_world_transform'] = _read(lambda: _usd_transform(sensor.prim, 'world'), 'USD Xformable')
        row['usd_local_transform'] = _read(lambda: _usd_transform(sensor.prim, 'parent'), 'USD Xformable')
        row['world_camera_forward'] = _read(lambda: _forward(sensor, row['world_pose']), 'declared USD Camera -Z convention')
        row['properties'] = {field: _read(lambda field=field: getattr(sensor, field), 'native sensor.' + field)
            for field in ('image_width', 'image_height', 'modalities', 'focal_length', 'horizontal_aperture', 'clipping_range')}
        for field, attribute in (('vertical_aperture', 'verticalAperture'), ('projection', 'projection')):
            row['properties'][field] = _read(lambda attribute=attribute: sensor.prim.GetAttribute(attribute).Get(),
                'USD prim attribute ' + attribute)
        result['sensors'].append(row)
    return result

"""Read exact named native object links before source settling, without writes.

Run: python -B -m pytest -q tests/test_task_edit2_initial_link_diagnostics.py
Call measure_initial_object_links(session, names, stage) on the simulator owner
thread. Runtime OG/link getters and authored USD Default-time values are separate
evidence. No simulator import, stepping, rendering or schema creation occurs at
module import. This report diagnoses measurements; it never certifies safety.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from wia.runtime.sim.esi_initial_camera_diagnostics import _provenance, _usd_transform

DESCENDANT_CAP = 256
ATTRIBUTE_CAP = 128


class NonFiniteMeasurement(ValueError):
    """An actual returned numeric measurement contains NaN or infinity."""


def _value(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return _value(value.tolist())
    if isinstance(value, np.generic):
        return _value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise NonFiniteMeasurement('Native numeric value is non-finite.')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # USD Gf quaternion ordering is explicitly distinct from OG XYZW poses.
    if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
        return {'real': _value(float(value.GetReal())), 'imaginary_xyz': _value(list(value.GetImaginary()))}
    if hasattr(value, '__iter__'):
        return _value(list(value))
    raise TypeError('Unsupported native measurement type: ' + type(value).__name__)


def _read(callback, source):
    try:
        value = callback()
        if value is None:
            raise LookupError('Native field is unavailable (None).')
        return {'available': True, 'status': 'available', 'value': _value(value), 'source': source,
                'numeric_validation': 'all_numeric_leaves_finite'}
    except Exception as error:
        return {'available': False, 'status': 'nonfinite' if isinstance(error, NonFiniteMeasurement) else 'unavailable',
                'error_type': type(error).__name__, 'error': str(error), 'source': source}


def _unavailable(reason, source):
    return {'available': False, 'status': 'unavailable', 'error_type': 'UnsupportedNativeMeasurement',
            'error': reason, 'source': source}


def _validated_pose(position, quaternion):
    position, quaternion = np.asarray(_value(position), dtype=float), np.asarray(_value(quaternion), dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0:
        raise ValueError('Expected native position XYZ and nonzero quaternion XYZW.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _pose(node, frame):
    classes = {base.__name__ for base in type(node).__mro__}
    if frame == 'parent' and classes & {'RigidKinematicPrim', 'RigidDynamicPrim'}:
        # These installed overrides do not support parent frame; the kinematic
        # override even returns WORLD for it. Reuse XFormPrim's installed reader.
        import omnigibson.lazy as lazy
        position, quaternion_wxyz = lazy.isaacsim.core.utils.xforms.get_local_pose(node.prim_path)
        quaternion = np.asarray(_value(quaternion_wxyz), dtype=float)
        if quaternion.shape != (4,):
            raise ValueError('Expected Isaac local quaternion WXYZ.')
        return _validated_pose(position, quaternion[[1, 2, 3, 0]])
    if frame == 'world' and 'RigidKinematicPrim' in classes:
        # Reading the existing cache avoids populating/changing it. A missing
        # cache is unavailable; authored USD values remain separately reported.
        cache = vars(node).get('_kinematic_pose_cache', {})
        if 'world' not in cache:
            raise LookupError('Existing OG kinematic world-pose cache unavailable; no live PhysX pose inferred.')
        return _validated_pose(*cache['world'])
    return _validated_pose(*node.get_position_orientation(frame=frame))


def _aabb(node):
    bounds = np.asarray(_value(node.aabb), dtype=float)
    if bounds.shape != (2, 3) or np.any(bounds[1] < bounds[0]):
        raise ValueError('Expected native increasing AABB bounds [low_xyz, high_xyz].')
    return bounds.tolist()


def _node(node):
    classes = {base.__name__ for base in type(node).__mro__}
    kinematic_stub = 'RigidKinematicPrim' in classes
    rigid = bool(classes & {'RigidKinematicPrim', 'RigidDynamicPrim'})
    row = {'native_class': type(node).__name__,
        'prim_path': _read(lambda: str(node.prim_path), 'actual native prim_path'),
        'world_pose': _read(lambda: _pose(node, 'world'), 'existing OG kinematic world-pose cache, NOT live PhysX'
            if kinematic_stub else 'OG get_position_orientation(frame=world) getter'),
        'local_to_parent_pose': _read(lambda: _pose(node, 'parent'), 'installed Isaac xforms.get_local_pose, WXYZ converted to XYZW'
            if rigid else 'OG get_position_orientation(frame=parent) getter'),
        'scale': _read(lambda: node.scale, 'OG scale getter'),
        'aabb_m': _read(lambda: _aabb(node), 'OG aabb getter')}
    row['kinematic_compatibility_stub_detected'] = kinematic_stub
    for name in ('mass', 'center_of_mass'):
        source = 'OG ' + name + ' property'
        row[name] = (_unavailable('Installed RigidKinematicPrim property is a compatibility zero, not a physical measurement.', source)
            if kinematic_stub else _read(lambda name=name: getattr(node, name), source))
    row['mass']['semantics'] = 'OG dynamic property may fall back to analytical volume*density; raw runtime view is separate.'
    # Reuse an already-existing native view only. Never construct or initialize one.
    row['runtime_rigid_view'] = {name: _read(
        lambda method=method: getattr(node, '_rigid_prim_view').__getattribute__(method)(),
        'existing native _rigid_prim_view.' + method + '()')
        for name, method in (('world_poses_raw', 'get_world_poses'), ('masses', 'get_masses'),
            ('centres_of_mass', 'get_coms'), ('inertias', 'get_inertias'))}
    return row


def _attribute(attribute):
    # Attribute.Get() with no time argument reads USD Default, never PhysX state.
    result = _read(lambda: attribute.Get(), 'USD attribute.Get(), authored/composed Default time, not runtime PhysX')
    result['authored_value_opinion'] = _read(lambda: attribute.HasAuthoredValueOpinion(), 'USD HasAuthoredValueOpinion')
    return result


def _prim(prim):
    row = {'path': str(prim.GetPath()), 'type': _read(prim.GetTypeName, 'USD prim.GetTypeName'),
        'applied_schemas': _read(prim.GetAppliedSchemas, 'USD prim.GetAppliedSchemas, read only'),
        'custom_data': _read(prim.GetCustomData, 'USD prim.GetCustomData'),
        'metalink_name_marker': 'meta' in str(prim.GetName()).lower(),
        'usd_world_transform_default': _read(lambda: _usd_transform(prim, 'world'), 'USD Xformable Default-time transform, not runtime OG'),
        'usd_local_transform_default': _read(lambda: _usd_transform(prim, 'parent'), 'USD Xformable Default-time transform, not runtime OG'),
        'attributes_default': {}, 'joint_relationships': {}}
    required = ['physics:rigidBodyEnabled', 'physics:kinematicEnabled', 'physics:mass',
        'physics:diagonalInertia', 'physics:centerOfMass', 'physics:collisionEnabled',
        'physics:lowerLimit', 'physics:upperLimit', 'physics:localPos0', 'physics:localPos1',
        'physics:localRot0', 'physics:localRot1']
    try:
        names = [str(attribute.GetName()) for attribute in prim.GetAttributes()]
        selected = sorted(set(required + [name for name in names if name.startswith(('xformOp:', 'physx', 'physics:'))
            or any(marker in name.lower() for marker in ('meta', 'toggle', 'joint'))]))
        row['attribute_inventory'] = {'available': True, 'selected_count': len(selected),
            'cap': ATTRIBUTE_CAP, 'truncated': len(selected) > ATTRIBUTE_CAP}
        for name in selected[:ATTRIBUTE_CAP]:
            try:
                row['attributes_default'][name] = _attribute(prim.GetAttribute(name))
            except Exception as error:
                row['attributes_default'][name] = _read(lambda error=error: (_ for _ in ()).throw(error), 'USD GetAttribute')
    except Exception as error:
        row['attribute_inventory'] = _read(lambda: (_ for _ in ()).throw(error), 'USD prim.GetAttributes')
    for name in ('physics:body0', 'physics:body1'):
        row['joint_relationships'][name] = _read(
            lambda name=name: [str(path) for path in prim.GetRelationship(name).GetTargets()],
            'USD joint relationship.GetTargets, authored linkage')
    return row


def _descendants(root):
    root_path = str(root.GetPath())
    pending, result = [root], []
    while pending and len(result) < DESCENDANT_CAP:
        prim = pending.pop()
        path = str(prim.GetPath())
        if path != root_path and not path.startswith(root_path + '/'):
            raise ValueError('USD descendant escaped exact requested object root.')
        if not prim.IsValid():
            raise ValueError('Invalid USD descendant: ' + path)
        result.append(_prim(prim))
        children = sorted(prim.GetChildren(), key=lambda child: str(child.GetPath()), reverse=True)
        pending.extend(children)
    return {'available': True, 'includes_root': True, 'cap': DESCENDANT_CAP,
        'reported_count': len(result), 'truncated': bool(pending),
        'total_count': None if pending else len(result), 'prims': result}


def measure_initial_object_links(session, names, stage):
    """Return bounded, JSON-safe diagnostics for exact native registry names.

    A failed link remains in the inventory with typed unavailable/nonfinite
    fields. Missing or stale objects are explicit; no substitute is selected.
    The caller chooses when to measure; ``stage`` is a label, not a USD stage.
    """
    requested = [names] if isinstance(names, str) else list(names)
    result = {'schema': 'initial_object_link_diagnostic_v1', 'stage': str(stage),
        'requested_names': requested, 'provenance': _provenance(session),
        'native_time_s': _read(lambda: float(session.pipeline.og.sim.current_time), 'native simulator current_time'),
        'scope': 'explicit native object names only', 'descendant_cap_per_object': DESCENDANT_CAP,
        'safety_qualification': None, 'objects': []}
    for name in dict.fromkeys(requested):
        row = {'requested_name': name, 'status': 'unavailable', 'links': [], 'joints': []}
        result['objects'].append(row)
        try:
            if not isinstance(name, str) or not name or '/' in name:
                raise ValueError('An exact native registry object name is required, not a path or pattern.')
            obj = session.env.scene.object_registry('name', name)
            if obj is None:
                raise LookupError('Exact named native object is absent.')
            if obj.name != name or not obj.prim.IsValid() or str(obj.prim.GetPath()) != str(obj.prim_path):
                raise ValueError('Native registry object/root is stale or mismatched.')
            row.update(status='resolved', **_node(obj))
            row['identity'] = {field: _read(lambda field=field: getattr(obj, field), 'native object.'+field)
                for field in ('name', 'category', 'model', 'fixed_base', 'visual_only')}
            for field in ('links', 'joints'):
                try:
                    registry = getattr(obj, field)
                    if not isinstance(registry, Mapping):
                        raise TypeError('Actual native '+field+' registry is not a mapping.')
                    row[field+'_inventory'] = {'available': True, 'actual_count': len(registry),
                        'cap': DESCENDANT_CAP, 'truncated': len(registry) > DESCENDANT_CAP}
                    for key, node in list(registry.items())[:DESCENDANT_CAP]:
                        record = {'registry_name': str(key), **_node(node)} if field == 'links' else {
                            'registry_name': str(key), 'prim_path': _read(lambda: str(node.prim_path), 'native joint prim_path'),
                            'state': _read(lambda: node.get_state(normalized=False), 'OG joint.get_state(normalized=False): position, velocity, effort')}
                        row[field].append(record)
                except Exception as error:
                    row[field+'_inventory'] = _read(lambda: (_ for _ in ()).throw(error), 'actual native '+field+' registry')
            row['usd_descendants'] = _descendants(obj.prim)
        except Exception as error:
            row['error_type'], row['error'] = type(error).__name__, str(error)
            if row['status'] == 'resolved':
                row['usd_descendants'] = {'available': False, 'error_type': type(error).__name__, 'error': str(error)}
    return result

"""Apply an explicit awake-contact protocol only during native initialization.

Run CPU checks: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -B -m pytest -q -p no:cacheprovider tests/test_wia_contact_awake_initialization.py
Use initialize_scene in an after_postprocessing category_initialization patch.
Only selected dynamic links' sleep thresholds and one native wake call change;
the helper never steps physics or installs a runtime callback.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from numbers import Real

REVISION = 'wia_contact_awake_initialization_v1'


def _threshold(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Native sleep_threshold must be a finite numeric value.')
    return float(value)


def _asleep(link):
    value = link.is_asleep
    if type(value) is not bool:
        raise ValueError('Native is_asleep getter must return a boolean.')
    return value


def _publish(session, receipt):
    session.wia_contact_awake_initialization = receipt
    print(json.dumps({'wia_contact_awake_initialization': receipt}, separators=(',', ':'), allow_nan=False), flush=True)


def initialize_scene(session, parameters):
    """Validate every selected native link before changing its sleep protocol."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Awake-contact protocol is initialization-only.')
    if getattr(session, 'wia_contact_awake_initialization', None) is not None:
        raise ValueError('Awake-contact initialization has already been attempted in this session.')
    expected = parameters.get('expected_scene')
    names = parameters.get('object_names')
    if (parameters.get('revision') != REVISION or not isinstance(expected, str) or not expected.strip()
            or getattr(session, 'scene_name', None) != expected or session.env.scene.scene_model != expected):
        raise ValueError('Awake-contact initialization requires its revision and exact actual session/scene identity.')
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('object_names must be an explicit unique nonempty list of native object names.')
    if _threshold(parameters.get('sleep_threshold')) != 0.0:
        raise ValueError('Awake-contact initialization requires sleep_threshold exactly 0.0.')
    from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

    pending, seen_paths = [], set()
    for name in names:
        obj = session.env.scene.object_registry('name', name)
        if obj is None or getattr(obj, 'name', None) != name:
            raise ValueError('Named awake-contact native object is unavailable: ' + name)
        if getattr(obj, 'fixed_base', None) is not False or getattr(obj, 'visual_only', None) is not False:
            raise ValueError('Awake-contact objects must be non-fixed and collidable: ' + name)
        links = getattr(obj, 'links', None)
        if not isinstance(links, Mapping) or not links:
            raise ValueError('Awake-contact object has no native link mapping: ' + name)
        physical_count = 0
        for link_name, link in links.items():
            meta = link.is_meta_link
            if type(meta) is not bool:
                raise ValueError('Native meta-link identity is unavailable: ' + name)
            if meta:
                continue
            meshes = link.collision_meshes
            if not isinstance(meshes, Mapping):
                raise ValueError('Native collision geometry is unavailable: ' + name)
            if not meshes:
                continue
            if not isinstance(link, RigidDynamicPrim):
                raise ValueError('Every selected physical collision link must be a native RigidDynamicPrim: ' + name)
            if any(mesh.collision_enabled is not True for mesh in meshes.values()):
                raise ValueError('Selected native collision geometry must be enabled: ' + name)
            rigid, kinematic = link.get_attribute('physics:rigidBodyEnabled'), link.get_attribute('physics:kinematicEnabled')
            if rigid is not True or kinematic is not False:
                raise ValueError('Selected native links must have enabled rigid bodies and disabled kinematics: ' + name)
            path = link.prim_path
            if not isinstance(path, str) or not path or path in seen_paths:
                raise ValueError('Selected native collision link paths must be nonempty and unique.')
            if not callable(link.wake):
                raise ValueError('Native dynamic link wake method is unavailable: ' + name)
            seen_paths.add(path)
            physical_count += 1
            row = {'object_name': name, 'object_prim_path': obj.prim_path, 'link_name': str(link_name),
                'link_prim_path': path, 'native_link_type': type(link).__module__ + '.' + type(link).__name__,
                'rigid_body_enabled': True, 'kinematic_enabled': False, 'collision_mesh_count': len(meshes),
                'original_sleep_threshold': _threshold(link.sleep_threshold), 'original_is_asleep': _asleep(link),
                'requested_sleep_threshold': 0.0, 'sleep_threshold_set_calls': 0, 'wake_calls': 0,
                'readback_verified': False}
            pending.append((link, row))
        if not physical_count:
            raise ValueError('Awake-contact object has no non-meta physical collision links: ' + name)
    receipt = {'revision': REVISION, 'expected_scene': expected, 'object_names': list(names),
        'sleep_threshold': 0.0, 'initialization_only': True, 'runtime_wake_callbacks': False,
        'physics_steps': 0, 'status': 'applying', 'readback_verified': False,
        'native_validation_run': False, 'physical_success_verified': False,
        'links': [row for _, row in pending]}
    session.wia_contact_awake_initialization = receipt
    try:
        for link, row in pending:
            row['sleep_threshold_set_calls'] = 1
            link.sleep_threshold = 0.0
            row['wake_calls'] = 1
            link.wake()
            row['readback_sleep_threshold'] = _threshold(link.sleep_threshold)
            row['readback_is_asleep'] = _asleep(link)
            row['readback_verified'] = row['readback_sleep_threshold'] == 0.0 and row['readback_is_asleep'] is False
            if not row['readback_verified']:
                raise ValueError('Native awake-contact sleep threshold or awake-state readback mismatch: ' + row['link_prim_path'])
        receipt.update(status='applied', readback_verified=True)
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        _publish(session, receipt)
        raise
    _publish(session, receipt)
    return receipt

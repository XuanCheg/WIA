"""Bind native feasibility evidence to the actual reset state and controller.

Run: python src/wia/runtime/commands/check_esi_core_native.py --help
Set ESI_NATIVE_QUALIFICATION_MANIFEST to a reviewed release manifest for runtime
reuse. Missing, stale or mismatched evidence remains unavailable. This module
never moves a robot or an object and never infers feasibility from Agent failure.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from wia.tasks.interaction_setup import array
from wia.runtime.resources import runtime_source

ROOT = WIA_RUNTIME_ROOT
CONTROLLER_FILES = ('src/wia/runtime/sim/esi_base.py', 'src/wia/runtime/sim/esi_arm.py',
    'src/wia/runtime/sim/esi_rotation.py', 'src/wia/runtime/sim/esi_gripper.py', 'src/wia/runtime/sim/esi_actions.py',
    'core/action_compiler.py', 'src/wia/runtime/sim/esi_camera_geometry.py',
    'src/wia/runtime/sim/esi_native_qualification.py',
    'src/wia/runtime/sim/esi_trunk.py', 'src/wia/runtime/sim/esi_ik.py', 'src/wia/runtime/sim/esi_rotation_checks.py',
    'src/wia/runtime/sim/esi_omnigibson.py', 'src/wia/runtime/sim/esi_safety.py',
    'src/wia/runtime/sim/esi_robot_cameras.py', 'src/wia/runtime/sim/esi_reflection.py', 'configs/esi_base.json', 'configs/esi_arm.json',
    'configs/esi_robot_cameras.json', 'configs/esi_trunk.json')


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def condition_snapshot(session, controller, task):
    """Read a reproducible initial world, sensor rig and controller condition."""
    values = session.env.scene.objects
    objects = list(values.values()) if isinstance(values, dict) else list(values)
    rows, native_names = {}, {}
    for obj in objects:
        p, q = (array(v).tolist() for v in obj.get_position_orientation())
        key = '@robot' if obj is session.robot else obj.name
        native_names[key] = obj.name
        rows[key] = {'category': obj.category, 'model': getattr(obj, 'model', None),
            'position': p, 'quaternion_xyzw': q, 'scale': array(obj.scale).tolist(),
            'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only),
            'joints': {name: array(joint.get_state()[0]).tolist() for name, joint in obj.joints.items()}}
    registered_views = {sensor: view for view, sensor in getattr(session, 'robot_camera_registry', {}).items()}
    sensors = {}
    for name, sensor in session.robot.sensors.items():
        mount_path = sensor.prim_path.removeprefix(session.robot.prim_path)
        key = registered_views.get(name, 'mounted:' + mount_path)
        sensors[key] = {'mount_path': mount_path,
            'mount_pose': [array(v).tolist() for v in sensor.get_position_orientation(frame='parent')],
            'focal_length': float(sensor.focal_length), 'horizontal_aperture': float(sensor.horizontal_aperture),
            'image_size_px': [int(sensor.image_width), int(sensor.image_height)],
            'clipping_range_m': array(sensor.clipping_range).tolist(),
            'modalities': sorted(sensor.modalities)}
    return {'schema': 'native_initial_condition_v2', 'scene': session.scene_name,
        'room': session.room_name, 'source_sha256': task.source_sha256,
        'task_sha256': task.file_sha256, 'variant': task.variant, 'seed': task.seed,
        'controller_files': {name: digest_file(runtime_source(name)) for name in CONTROLLER_FILES if runtime_source(name).is_file()},
        'objects': rows, 'native_object_names': native_names,
        'robot': {'position': controller.pose()['position'],
            'quaternion_xyzw': controller.pose()['quaternion_xyzw'],
            'scale': array(session.robot.scale).tolist(),
            'joint_positions': array(session.robot.get_joint_positions()).tolist()},
        'sensors': sensors}


def compare_conditions(expected, current):
    """Allow reset numerical noise, never a different scene/scale/door state."""
    failures = []
    for key in ('schema', 'scene', 'room', 'source_sha256', 'task_sha256', 'variant', 'seed', 'controller_files'):
        if expected.get(key) != current.get(key):
            failures.append(key)
    def close(a, b, tolerance):
        try:
            x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            return x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x, y, atol=tolerance, rtol=0)
        except (ValueError, TypeError):
            return False
    def pose(a, b, name):
        if not close(a.get('position'), b.get('position'), .01):
            failures.append(name + ':position')
        try:
            q, r = (np.asarray(row['quaternion_xyzw'], dtype=float) for row in (a, b))
            dot = abs(float(np.dot(q, r))) / max(1e-12, float(np.linalg.norm(q) * np.linalg.norm(r)))
            if not np.isfinite(dot) or 2 * np.arccos(np.clip(dot, 0, 1)) > np.deg2rad(1):
                failures.append(name + ':orientation')
        except (KeyError, ValueError, TypeError):
            failures.append(name + ':orientation_unavailable')
        if not close(a.get('scale'), b.get('scale'), 1e-5):
            failures.append(name + ':scale')
    pose(expected.get('robot', {}), current.get('robot', {}), 'robot')
    if not close(expected.get('robot', {}).get('joint_positions'), current.get('robot', {}).get('joint_positions'), .02):
        failures.append('robot:joints')
    for name in set(expected.get('objects', {})) | set(current.get('objects', {})):
        a, b = expected.get('objects', {}).get(name), current.get('objects', {}).get(name)
        if a is None or b is None:
            failures.append(name + ':missing'); continue
        for key in ('category', 'model', 'fixed_base', 'visual_only'):
            if a.get(key) != b.get(key): failures.append(name + ':' + key)
        pose(a, b, name)
        if set(a['joints']) != set(b['joints']): failures.append(name + ':joint_inventory')
        for key in a['joints']:
            if not close(a['joints'][key], b['joints'].get(key), .02): failures.append(name + ':joint:' + key)
    if expected.get('sensors') != current.get('sensors'):
        # Sensor local transforms can include float roundoff but cannot float away
        # from their mounts. Full precision sensor metadata remains in both receipts.
        a, b = expected.get('sensors', {}), current.get('sensors', {})
        if set(a) != set(b): failures.append('sensor_inventory')
        for name in set(a) & set(b):
            for key in ('mount_path', 'image_size_px', 'modalities'):
                if a[name].get(key) != b[name].get(key): failures.append(name + ':' + key)
            for key in ('focal_length', 'horizontal_aperture'):
                if not close(a[name][key], b[name][key], 1e-5): failures.append(name + ':' + key)
            if not close(a[name].get('clipping_range_m'), b[name].get('clipping_range_m'), 1e-5):
                failures.append(name + ':clipping_range')
            if not close(a[name]['mount_pose'][0], b[name]['mount_pose'][0], 1e-5): failures.append(name + ':mount_position')
            q, r = a[name]['mount_pose'][1], b[name]['mount_pose'][1]
            if not (close(q, r, 1e-5) or close(q, -np.asarray(r), 1e-5)):
                failures.append(name + ':mount_orientation')
    return {'matched': not failures, 'mismatches': failures, 'position_tolerance_m': .01,
        'orientation_tolerance_deg': 1., 'joint_tolerance_rad': .02}


def apply_reviewed_qualification(session, controller, task, output):
    """Import reviewed, hashed native receipts only after actual reset matching."""
    selected = os.environ.get('ESI_NATIVE_QUALIFICATION_MANIFEST')
    if not selected:
        return {'accepted': False, 'reason': 'No reviewed native qualification manifest configured.'}
    path = Path(selected)
    try:
        manifest = json.loads(path.read_text())
        entry = manifest['tasks'][task.file_sha256]
        proofs = []
        for ref in entry['evidence']:
            evidence_path = Path(ref['path'])
            if digest_file(evidence_path) != ref['sha256']:
                raise ValueError('Native evidence hash mismatch.')
            proofs.append(json.loads(evidence_path.read_text()))
        if not proofs or entry.get('reviewed') is not True:
            raise ValueError('Native evidence has not been reviewed.')
        match = compare_conditions(entry['initial_condition'], condition_snapshot(session, controller, task))
        result = {'accepted': match['matched'], 'condition_match': match,
            'manifest': str(path), 'manifest_sha256': digest_file(path), 'evidence': entry['evidence']}
        if match['matched']:
            if entry.get('reflection'):
                # Current pose, scale, mount and source optics are independently
                # measured; only prior real mobility can be reused here.
                session.reflection_qualification.update(legal_mobility_verified=entry['reflection'].get('legal_mobility_verified') is True,
                    reviewed_evidence=result)
            if entry.get('mapping'):
                receipt = deepcopy(entry['mapping'])
                receipt.update(episode_id=session.episode_id, condition_identity_verified=True)
                session.mapping_qualification = {task.definition.source.json_path: receipt}
                session.door_qualification = {key: {**deepcopy(row), 'condition_matches_current_episode': True}
                    for key, row in entry.get('doors', {}).items()}
    except (OSError, KeyError, ValueError, TypeError) as error:
        result = {'accepted': False, 'reason': f'{type(error).__name__}: {error}', 'manifest': str(path)}
    from wia.runtime.scene_capture_queue import write_json
    write_json(Path(output) / 'native_qualification_reuse.json', result)
    return result

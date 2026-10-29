"""Audit two native Phase-2 captures without claiming memory-free equivalence.

Run: python -B src/wia/runtime/commands/audit_wia_temporal_pair.py --left-session PATH \
  --right-session PATH --left-label LABEL --right-label LABEL \
  --left-task PATH --right-task PATH --output PATH
Only explicit task files and capture-local artifacts are read; no simulator runs.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from core.robot_cameras import ROBOT_CAMERA_LABELS
from wia.tasks.loader import load_task

TOLERANCES = {
    'position_l2_m': 0.001, 'aabb_max_abs_m': 0.001,
    'quaternion_geodesic_rad': 0.001, 'scale_max_abs': 0.000001,
    'intrinsics_max_abs_pixels': 0.001, 'resolution': 'exact_integer_equality',
}


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8')), None
    except (OSError, ValueError) as error:
        return {}, f'{type(error).__name__}: {error}'


def _status(rows):
    values = [row['match'] for row in rows]
    return None if not values or None in values else all(values)


def _numeric(left, right, shape, tolerance, metric='max_abs'):
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if a.shape != shape or b.shape != shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Required finite native measurement has the wrong shape.')
        if metric == 'quaternion_geodesic':
            if min(np.linalg.norm(a), np.linalg.norm(b)) == 0:
                raise ValueError('Native quaternion must be nonzero.')
            dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
            delta = 2 * math.acos(min(1.0, dot))
        else:
            delta = float(np.linalg.norm(a - b)) if metric == 'l2' else float(np.max(np.abs(a - b)))
        return {'match': bool(delta <= tolerance), 'delta': delta,
                'tolerance': tolerance, 'metric': metric}
    except (TypeError, ValueError) as error:
        return {'match': None, 'delta': None, 'tolerance': tolerance,
                'metric': metric, 'reason': str(error)}


def _pose(left, right, position_key='position_m', quaternion_key='quaternion_xyzw'):
    return {
        'position_l2_m': _numeric(left.get(position_key), right.get(position_key), (3,),
                                  TOLERANCES['position_l2_m'], 'l2'),
        'quaternion_geodesic_rad': _numeric(left.get(quaternion_key), right.get(quaternion_key), (4,),
                                            TOLERANCES['quaternion_geodesic_rad'], 'quaternion_geodesic'),
    }


def _brief(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True)
    if len(encoded) <= 400:
        return value
    return {'type': type(value).__name__, 'serialized_characters': len(encoded),
            'sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest()}


def _field_comparison(left, right, prefix=''):
    """Report leaf equality while keeping encoded pixels and large values out."""
    rows = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            path = f'{prefix}.{key}' if prefix else key
            if key in ('png_base64', 'npy_base64', 'pixels'):
                rows.append({'field': path, 'equal': left.get(key) == right.get(key),
                             'values_omitted': 'encoded_pixel_or_numeric_array'})
            elif key not in left or key not in right:
                rows.append({'field': path, 'equal': False,
                             'left_present': key in left, 'right_present': key in right})
            else:
                rows.extend(_field_comparison(left[key], right[key], path))
    elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        if not left:
            rows.append({'field': prefix, 'equal': True})
        for index, (a, b) in enumerate(zip(left, right)):
            rows.extend(_field_comparison(a, b, f'{prefix}[{index}]'))
    else:
        equal = left == right
        rows.append({'field': prefix, 'equal': equal,
                     **({} if equal else {'left': _brief(left), 'right': _brief(right)})})
    return rows


def _selectors(loaded):
    p = loaded.definition.private.evaluation.parameters
    roles = {}
    for role, keys in (
            ('container', p['container_keys']), ('active_phase_2_content', p['phase_keys']['image2']),
            ('apparatus', [row['name'] for kind in ('primitives', 'dataset_objects')
                           for row in p['apparatus'].get(kind, [])]),
            ('public_spare', [row['binding_key'] for row in p['public_spare_inventory']])):
        for key in keys:
            roles.setdefault(key, []).append(role)
    return roles, [key for key in p['phase_keys']['image1'] if key not in roles]


def _binding_snapshot(loaded, state, key):
    binding = loaded.definition.private.bindings.get(key)
    if binding is None:
        return None, None
    return binding.name, state.get('objects', {}).get(binding.name)


def _load_image(capture, observation, name):
    info = observation.get('images', {}).get(name)
    if not isinstance(info, dict) or not info.get('path'):
        raise ValueError('Required capture-local PNG metadata is missing.')
    relative = Path(info['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Only capture-local relative image pointers are read.')
    path = (capture / relative).resolve()
    if not path.is_relative_to(capture.resolve()):
        raise ValueError('Image pointer is outside this capture; external evidence pointers are not followed.')
    data = path.read_bytes()
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB':
            raise ValueError('The actual saved image must be a uint8 RGB PNG.')
        pixels = np.array(image)
    return pixels, data


def _image_pair(left_capture, right_capture, left_obs, right_obs, name):
    try:
        a, a_png = _load_image(left_capture, left_obs, name)
        b, b_png = _load_image(right_capture, right_obs, name)
    except (OSError, ValueError, TypeError) as error:
        return {'match': None, 'exact_pixels_equal': None, 'mae_uint8_rgb': None,
                'differing_pixel_fraction': None, 'reason': f'{type(error).__name__}: {error}'}
    result = {'left_shape': list(a.shape), 'right_shape': list(b.shape),
              'png_bytes_equal': a_png == b_png, 'match': False, 'exact_pixels_equal': False,
              'mae_uint8_rgb': None, 'differing_pixel_fraction': None}
    if a.shape != b.shape:
        result['reason'] = 'Actual RGB image dimensions differ; per-pixel metrics are undefined.'
        return result
    difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
    exact = bool(np.array_equal(a, b))
    result.update(match=exact, exact_pixels_equal=exact,
                  mae_uint8_rgb=float(difference.mean()),
                  differing_pixel_fraction=float(np.any(difference != 0, axis=2).mean()))
    return result


def _overview_start(left, right):
    a = ((left.get('initial_overview') or {}).get('capture') or {})
    b = ((right.get('initial_overview') or {}).get('capture') or {})
    position = _numeric(a.get('robot_start_position_m'), b.get('robot_start_position_m'),
                        (3,), TOLERANCES['position_l2_m'], 'l2')
    try:
        difference = float(a['robot_start_yaw_rad']) - float(b['robot_start_yaw_rad'])
        if not math.isfinite(difference):
            raise ValueError('Overview start yaw is not finite.')
        delta = abs(math.atan2(math.sin(difference), math.cos(difference)))
        yaw = {'match': delta <= TOLERANCES['quaternion_geodesic_rad'],
               'delta': delta, 'tolerance': TOLERANCES['quaternion_geodesic_rad']}
    except (KeyError, TypeError, ValueError) as error:
        yaw = {'match': None, 'delta': None, 'reason': str(error)}
    return {'position_l2_m': position, 'wrapped_yaw_difference_rad': yaw}


def audit_pair(*, left_session, right_session, left_label, right_label, left_task, right_task):
    """Compare measured current geometry/RGB and report history-bearing inputs."""
    sessions = [Path(left_session).resolve(), Path(right_session).resolve()]
    labels = (left_label, right_label)
    if any(not label or Path(label).name != label or label in ('.', '..') for label in labels):
        raise ValueError('Snapshot labels must be single capture-directory names.')
    captures = [session / label for session, label in zip(sessions, labels)]
    states, observations, input_errors = [], [], []
    for side, session, label, capture in zip(('left', 'right'), sessions, labels, captures):
        state, error = _read_json(session / f'{label}_state.json')
        states.append(state)
        if error:
            input_errors.append({'side': side, 'artifact': 'native_state', 'reason': error})
        observation, error = _read_json(capture / 'observation.json')
        observations.append(observation)
        if error:
            input_errors.append({'side': side, 'artifact': 'public_observation', 'reason': error})
    loaded = [load_task(path, variant='adapted', run_mode='diagnostic') for path in (left_task, right_task)]
    selectors = [_selectors(task) for task in loaded]
    left_state, right_state = states
    geometry, missing = [], []
    object_rows = {}
    for key in sorted(set(selectors[0][0]) | set(selectors[1][0])):
        names, actual = zip(*[_binding_snapshot(task, state, key) for task, state in zip(loaded, states)])
        row = {'roles': {side: roles.get(key, []) for side, (roles, _) in zip(('left', 'right'), selectors)},
               'left_native_name': names[0], 'right_native_name': names[1]}
        if any(value is None for value in actual) or any(key not in selection[0] for selection in selectors):
            row.update(match=None, reason='A required active native binding or measured object is missing.')
            missing.append({'scope': 'active_object', 'binding_key': key, 'reason': row['reason']})
        else:
            checks = _pose(*actual)
            checks['aabb_max_abs_m'] = _numeric(actual[0].get('aabb_m'), actual[1].get('aabb_m'),
                                               (2, 3), TOLERANCES['aabb_max_abs_m'])
            checks['scale_max_abs'] = _numeric(actual[0].get('scale'), actual[1].get('scale'),
                                               (3,), TOLERANCES['scale_max_abs'])
            checks['category_equal'] = {'match': actual[0].get('category') == actual[1].get('category')}
            row.update(checks=checks, match=_status(checks.values()))
            geometry.extend({'scope': 'active_object', 'binding_key': key, 'field': field, **value}
                            for field, value in checks.items())
        object_rows[key] = row
    for key, row in object_rows.items():
        if row['match'] is None and 'checks' not in row:
            geometry.append({'scope': 'active_object', 'binding_key': key, 'match': None, 'reason': row['reason']})
    robot_checks = _pose(left_state.get('robot') or {}, right_state.get('robot') or {}, position_key='position')
    geometry.extend({'scope': 'current_robot', 'field': key, **row} for key, row in robot_checks.items())
    cameras = {}
    for name in ROBOT_CAMERA_LABELS:
        a, b = left_state.get('cameras', {}).get(name), right_state.get('cameras', {}).get(name)
        if not isinstance(a, dict) or not isinstance(b, dict):
            cameras[name] = {'match': None, 'reason': 'A required actual mounted-camera measurement is missing.'}
            missing.append({'scope': 'camera', 'name': name, 'reason': cameras[name]['reason']})
            geometry.append({'scope': 'camera', 'name': name, **cameras[name]})
            continue
        pose_a, pose_b = a.get('pose_world') or [], b.get('pose_world') or []
        a_pose = dict(zip(('position_m', 'quaternion_xyzw'), pose_a))
        b_pose = dict(zip(('position_m', 'quaternion_xyzw'), pose_b))
        checks = _pose(a_pose, b_pose)
        checks['intrinsics_max_abs_pixels'] = _numeric(a.get('intrinsics'), b.get('intrinsics'), (3, 3),
                                                      TOLERANCES['intrinsics_max_abs_pixels'])
        resolutions = (a.get('resolution'), b.get('resolution'))
        valid_sizes = all(isinstance(size, list) and len(size) == 2
                          and all(type(value) is int and value > 0 for value in size) for size in resolutions)
        checks['resolution_equal'] = {'match': resolutions[0] == resolutions[1] if valid_sizes else None,
                                       'left': resolutions[0], 'right': resolutions[1]}
        cameras[name] = {'match': _status(checks.values()), 'checks': checks}
        geometry.extend({'scope': 'camera', 'name': name, 'field': field, **row} for field, row in checks.items())
    phase_evidence = {side: {
        'phase': (state.get('temporal_initialization') or {}).get('phase'),
        'native_validity': (state.get('temporal_initialization') or {}).get('valid'),
        'actual_slots': (state.get('temporal_initialization') or {}).get('slots'),
    } for side, state in zip(('left', 'right'), states)}
    phase_two = all(row['phase'] == 'phase_2' for row in phase_evidence.values())
    if not phase_two:
        geometry.append({'scope': 'phase_evidence', 'match': None,
                         'reason': 'Both captures must explicitly measure temporal phase_2.'})
    rgb = {name: _image_pair(*captures, *observations, name) for name in ROBOT_CAMERA_LABELS}
    overview = {
        'history_bearing': True, 'excluded_from_current_state_matching': True,
        'rgb': _image_pair(*captures, *observations, 'initial_overview_rgb'),
        'robot_start': _overview_start(*observations),
        'metadata': _field_comparison(observations[0].get('initial_overview'),
                                      observations[1].get('initial_overview'), 'initial_overview'),
    }
    parked = {}
    for side, task, state, (_, keys) in zip(('left', 'right'), loaded, states, selectors):
        parked[side] = []
        temporal_objects = (state.get('temporal_initialization') or {}).get('objects', {})
        for key in keys:
            name, actual = _binding_snapshot(task, state, key)
            parked[side].append({'binding_key': key, 'native_name': name,
                'declared_identity': task.definition.private.bindings[key].model_dump(mode='json'),
                'actual_geometry': {field: actual.get(field) for field in (
                    'category', 'position_m', 'quaternion_xyzw', 'aabb_m', 'scale')} if actual else None,
                'native_temporal_properties': {field: temporal_objects.get(key, {}).get(field)
                                              for field in ('fixed_base', 'visual_only')},
                'visibility': 'not_directly_recorded_in_state',
                'native_inactive_check': (state.get('temporal_initialization') or {}).get(
                    'checks', {}).get('inactive:' + key),
                'possible_observation_leak': True})
    metadata = []
    metadata_fields = (set(observations[0]) | set(observations[1]) | {
        'action_history', 'snapshot_id', 'captured_at', 'action_step', 'public_task_events'})
    for field in sorted(metadata_fields - {'initial_overview'}):
        a_present, b_present = field in observations[0], field in observations[1]
        comparisons = _field_comparison(observations[0].get(field), observations[1].get(field), field)
        metadata.append({'field': field, 'left_present': a_present, 'right_present': b_present,
                         'equal': a_present == b_present and all(row['equal'] for row in comparisons),
                         'fields': comparisons})
    declared_start = [{name: [patch for patch in variant['patches'] if patch['operation'] == 'robot_pose']
                       for name, variant in task.definition.scene.model_dump(mode='json')['variants'].items()}
                      for task in loaded]
    public_rows = _field_comparison(loaded[0].public_payload, loaded[1].public_payload)
    flags = {
        'native_geometry_match': _status(geometry),
        'public_task_match': loaded[0].public_payload == loaded[1].public_payload,
        'current_rgb_exact_match': _status(rgb.values()),
        'no_history_condition_verified': False,
        'legal_reference_execution_verified': False,
    }
    current_flags = [flags[key] for key in ('native_geometry_match', 'public_task_match', 'current_rgb_exact_match')]
    status = 'inconclusive' if None in current_flags else 'current_state_mismatch' if not all(current_flags) else 'current_state_matches_only'
    maxima = {}
    for row in geometry:
        if row.get('delta') is not None:
            maxima[row['field']] = max(maxima.get(row['field'], 0.0), row['delta'])
    return {
        'schema': 'wia_temporal_native_pair_audit_v1', 'status': status, **flags,
        'inputs': {side: {'session': str(session), 'label': label, 'task_file': str(task.path)}
                   for side, session, label, task in zip(('left', 'right'), sessions, labels, loaded)},
        'tolerances': TOLERANCES, 'input_errors': input_errors,
        'native_phase_evidence': phase_evidence,
        'native_phase_evidence_comparison': _field_comparison(
            phase_evidence['left'], phase_evidence['right']),
        'native_geometry': {'active_objects': object_rows, 'current_robot': robot_checks,
                            'mounted_cameras': cameras, 'maximum_deltas': maxima,
                            'mismatches': [row for row in geometry if row['match'] is False],
                            'inconclusive_fields': [row for row in geometry if row['match'] is None],
                            'missing_required_evidence': missing},
        'declared_robot_initialization': {'equal': declared_start[0] == declared_start[1],
                                          'left': declared_start[0], 'right': declared_start[1]},
        'public_task': {'full_loader_payload_comparison': public_rows},
        'current_mounted_rgb': rgb,
        'frozen_initial_overview': overview,
        'parked_phase_1_objects': {'excluded_from_active_geometry_match': True, 'captures': parked,
                                  'observation_leak_excluded': False},
        'public_metadata': metadata,
        'potential_history_channels': ['action_history', 'snapshot_id', 'captured_at', 'action_step',
                                      'public_task_events', 'initial_overview', 'parked_phase_1_objects'],
        'limitations': [
            'Geometry comparison uses actual capture measurements, not declared source poses.',
            'Only task-selected active binding objects, current robot and eight mounted cameras enter geometry matching.',
            'Other furniture, articulated link state and unselected background objects are not certified equivalent.',
            'Exact current RGB equality and numeric differences are reported; no approximate-image pass threshold is defined.',
            'Low image MAE does not prove observational equivalence or exclude history shortcuts.',
            'Frozen overview pixels, overview-derived start records and public metadata remain history-bearing channels.',
            'This audit does not verify a memory-free condition or legal reference execution.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('left-session', 'right-session', 'left-task', 'right-task', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('left-label', 'right-label'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    report = audit_pair(**{key: value for key, value in vars(args).items() if key != 'output'})
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Wrote temporal pair audit: {output} ({report['status']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

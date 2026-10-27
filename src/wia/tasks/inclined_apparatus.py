"""Pure support-envelope contracts for the versioned whole inclined apparatus.

Run CPU checks: python -B -m pytest -q tests/test_task_edit2_inclined.py.
Root must materialize the proposal with native pre-translation pickup bounds
before support search. Neither a source endpoint nor this proposal certifies
one second of future motion, physical support, robot access or release.
"""
from __future__ import annotations

import copy
import numpy as np

REVISION = 'task_edit2_inclined_apparatus_20261018_v1'
NATIVE_RECEIPT = 'task_edit2_inclined_native_placement_20261018_v1'
MAXIMUM_DECK_DIMENSIONS_M = np.array([2., 1.4])


def _vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Expected a finite three-dimensional source/native vector.')
    return result


def _bounds(value):
    low, high = [_vector(v) for v in value]
    if np.any(high <= low):
        raise ValueError('Apparatus bounds must have positive dimensions.')
    return low, high


def source_apparatus_proposal(payload):
    """Preserve saved geometry; absent pickup bounds remain explicitly unavailable."""
    objects = payload['objects']
    objects = objects if isinstance(objects, dict) else {row['_key']: row for row in objects}
    obj, slope = objects['task_obj'], payload['slope']
    ramp = _bounds((slope['aabb_min'], slope['aabb_max']))
    initial = _bounds((obj['aabb_min_init'], obj['aabb_max_init']))
    final = _bounds((obj['aabb_min_final'], obj['aabb_max_final']))
    for key in ('pos_on_floor', 'pos_init', 'pos_final'):
        _vector(obj[key])
    return {'schema': REVISION, 'status': 'pending_native_pickup_measurement',
        'ramp_bounds_m': [value.tolist() for value in ramp],
        'trial_initial_bounds_m': [value.tolist() for value in initial],
        'trial_final_bounds_m': [value.tolist() for value in final],
        'source_positions_m': {key: list(obj[key]) for key in ('pos_on_floor', 'pos_init', 'pos_final')},
        'source_floor_z_m': float(slope['aabb_min'][2]), 'pickup_bounds_m': None,
        'horizontal_diameter_margin_m': None, 'required_bounds_m': None,
        'path_provenance': 'Saved source initial/final roots and AABBs; intermediate path is unavailable.',
        'source_observation_horizon': '30 source steps; not a one-second runoff certificate',
        'one_second_runoff_native_validated': False,
        'required_native_gates': ['original_pickup_contact_feasibility', 'complete_envelope_support',
            'native_robot_approach', 'real_grasp_lift', 'top_face_supported_release', 'one_second_safe_motion']}


def materialize_apparatus(proposal, *, pickup_bounds, target_horizontal_diameter_m=None):
    """Union actual pickup, saved ramp and endpoint sweep plus one full diameter.

    All bounds remain in the original pre-translation world frame. The diameter
    is measured from the initialized native target's horizontal AABB diagonal
    when not explicitly supplied; it is never reduced to fit a support.
    """
    if proposal.get('schema') != REVISION:
        raise ValueError('The apparatus proposal revision is unsupported.')
    pickup = _bounds(pickup_bounds)
    diameter = float(np.linalg.norm((pickup[1] - pickup[0])[:2]) if target_horizontal_diameter_m is None
        else target_horizontal_diameter_m)
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('A positive measured native horizontal diameter is required.')
    if diameter + 1e-9 < np.linalg.norm((pickup[1] - pickup[0])[:2]):
        raise ValueError('A supplied diameter may not shrink the native AABB diagonal margin.')
    initial, final = [_bounds(proposal[key]) for key in ('trial_initial_bounds_m', 'trial_final_bounds_m')]
    positions = proposal['source_positions_m']
    start, end = _vector(positions['pos_init']), _vector(positions['pos_final'])
    # Preserve asymmetric root-to-AABB offsets at both recorded endpoints.
    offset_low = np.minimum(initial[0] - start, final[0] - end)
    offset_high = np.maximum(initial[1] - start, final[1] - end)
    path_low, path_high = np.minimum(start, end) + offset_low, np.maximum(start, end) + offset_high
    path_low[:2] -= diameter
    path_high[:2] += diameter
    all_bounds = [pickup, _bounds(proposal['ramp_bounds_m']), initial, final, (path_low, path_high)]
    low = np.minimum.reduce([row[0] for row in all_bounds])
    high = np.maximum.reduce([row[1] for row in all_bounds])
    result = copy.deepcopy(proposal)
    result.update(status='measured_pickup_envelope_proposal', pickup_bounds_m=[row.tolist() for row in pickup],
        horizontal_diameter_margin_m=diameter, trial_path_with_margin_bounds_m=[path_low.tolist(), path_high.tolist()],
        required_bounds_m=[low.tolist(), high.tolist()])
    result['support_dimensions'] = required_support_dimensions(result)
    return result


def required_support_dimensions(envelope):
    if envelope.get('schema') != REVISION or envelope.get('required_bounds_m') is None:
        return {'available': False, 'reason': 'native_pickup_envelope_unavailable', 'dimensions_xy_m': None}
    low, high = _bounds(envelope['required_bounds_m'])
    dimensions = (high - low)[:2]
    available = bool(np.all(dimensions <= MAXIMUM_DECK_DIMENSIONS_M + 1e-9))
    return {'available': available, 'reason': None if available else 'apparatus_exceeds_existing_support_limit',
        'dimensions_xy_m': dimensions.tolist(), 'maximum_dimensions_xy_m': MAXIMUM_DECK_DIMENSIONS_M.tolist(),
        'clearance_reduced': False}


def support_envelope_fit(envelope, support_bounds, translation, *, support_surface_z_m=None):
    """Reject a physical deck that cannot contain the FULL translated proposal."""
    dimensions = required_support_dimensions(envelope)
    if not dimensions['available']:
        return {'accepted': False, 'reason': dimensions['reason'], 'dimensions': dimensions}
    low, high = _bounds(envelope['required_bounds_m'])
    deck_low, deck_high = _bounds(support_bounds)
    delta = _vector(translation)
    moved_low, moved_high = low + delta, high + delta
    surface = float(deck_high[2] if support_surface_z_m is None else support_surface_z_m)
    checks = {'full_envelope_inside_deck_xy': bool(np.all(moved_low[:2] >= deck_low[:2] - 1e-6) and
        np.all(moved_high[:2] <= deck_high[:2] + 1e-6)),
        'source_trial_floor_at_deck_surface': bool(abs(envelope['source_floor_z_m'] + delta[2] - surface) <= .015),
        'selected_surface_matches_deck': bool(abs(surface - deck_high[2]) <= .015)}
    return {'accepted': all(checks.values()), 'reason': None if all(checks.values()) else 'full_apparatus_envelope_does_not_fit_deck',
        'checks': checks, 'dimensions': dimensions, 'translated_required_bounds_m': [moved_low.tolist(), moved_high.tolist()],
        'translation_m': delta.tolist(), 'support_surface_z_m': surface,
        'safe_runoff_region_name': 'selected apparatus deck',
        'one_second_runoff_native_validated': False}


def common_translation(setup, *, target_name, slope_name):
    """Read only an accepted explicit target+ramp transform and same camera shift."""
    if not setup.get('accepted') or not setup.get('relocated') or not setup.get('support_prim_path'):
        raise ValueError('Accepted elevated whole-apparatus support is unavailable.')
    delta = _vector(setup.get('group_translation_m'))
    transforms = setup.get('object_translations_m', {})
    if any(name not in transforms or not np.allclose(_vector(transforms[name]), delta, atol=1e-6, rtol=0)
            for name in (target_name, slope_name)):
        raise ValueError('Target and ramp require one common XYZ translation.')
    if not np.allclose(_vector(setup.get('source_reference_camera_translation_m')), delta, atol=1e-6, rtol=0):
        raise ValueError('The source camera must share the apparatus translation.')
    return delta

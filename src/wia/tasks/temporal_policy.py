"""Bounded temporal survey policy; run pytest tests/test_task_repair.py."""
from __future__ import annotations

SURVEY_DEFAULTS = {'phase_1_mode': 'bounded_read_only_survey', 'phase_1_max_decisions': 20,
                  'phase_1_auto_transition': True, 'phase_1_allow_early_transition': True}


def survey_policy(parameters=None):
    result = {key: (parameters or {}).get(key, default) for key, default in SURVEY_DEFAULTS.items()}
    if (result['phase_1_mode'] != 'bounded_read_only_survey'
            or type(result['phase_1_max_decisions']) is not int or result['phase_1_max_decisions'] < 1
            or any(type(result[key]) is not bool for key in ('phase_1_auto_transition', 'phase_1_allow_early_transition'))):
        raise ValueError('Invalid bounded Phase-1 survey configuration.')
    return result


def survey_action_allowed(action):
    token = action if isinstance(action, str) else getattr(action, 'action', None)
    if isinstance(action, dict):
        token = action.get('action')
    from core.action_units import arm_primitive
    token = arm_primitive(token)
    return isinstance(token, str) and (token.startswith(('BASE_', 'MV_', 'ROTATE_', 'TRUNK_')) or token == 'DONE')


TASK_EDIT2_TEMPORAL_REVISION = 'task_edit2_temporal_20261018_v1'


def informative_temporal_rows(session, observation, bindings, parameters):
    """Read only synchronized informative receipts; missing adapters stay unavailable."""
    provider = getattr(session, 'task_visual_evidence', None)
    if not callable(provider):
        return {'available': False, 'objects': {}, 'error': 'task_visual_evidence adapter unavailable'}
    try:
        result = provider(observation, bindings, parameters)
        if not isinstance(result, dict):
            raise ValueError('Visual evidence must be a mapping.')
        if parameters.get('encounter_policy') == 'mounted_visibility_v1':
            result = mounted_visibility_rows(session, observation, result, bindings, parameters)
        delivered = set(observation.rgb_images())
        rows = {}
        for key, row in result.get('objects', {}).items():
            if key not in bindings or not isinstance(row, dict):
                continue
            if (row.get('snapshot_id') == observation.snapshot_id and row.get('camera') in delivered
                    and row.get('native_name') == bindings[key]
                    and row.get('native_rgb_matches_delivered') is True
                    and row.get('identity_synchronized') is True and row.get('informative') is True
                    and bool(row.get('informative_feature_evidence'))
                    and row.get('visit_verified') is True
                    and isinstance(row.get('visit_evidence'), dict)
                    and row['visit_evidence'].get('snapshot_id') == observation.snapshot_id
                    and row['visit_evidence'].get('legal_observation_pose') is True):
                rows[key] = row
        return {**result, 'objects': rows, 'available': result.get('available') is True}
    except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as error:
        return {'available': False, 'objects': {}, 'error': f'{type(error).__name__}: {error}'}


def mounted_visibility_rows(session, observation, result, bindings, parameters):
    """Record visual exposure, never claim semantic recognition from mask size."""
    import math
    from copy import deepcopy
    rows = {}
    available = bool(result.get('measurement_available'))
    control = result.get('native_public_control_context', {})
    legal = control.get('available') is True and control.get('state_setter_used') is False
    available &= control.get('available') is True
    for key, measurement in result.get('measurements', {}).items():
        if key not in bindings:
            continue
        for view, row in measurement.get('views', {}).items():
            if view == 'initial_overview_rgb' or view not in observation.rgb_images():
                continue
            context = row.get('audit_context', {})
            native = context.get('native_pose', {}).get('position_m')
            robot = context.get('robot_pose', {}).get('position_m')
            near = native is not None and robot is not None and math.dist(native[:2], robot[:2]) <= parameters.get(
                'encounter_max_distance_m', 2.)
            exposed = (row.get('identity_synchronized') is True and
                row.get('native_rgb_matches_delivered') is True and
                row.get('snapshot_id') == observation.snapshot_id and
                row.get('native_name') == bindings[key] and
                row.get('visible_pixels', 0) >= parameters.get('encounter_min_pixels', 144) and
                row.get('minimum_extent_px', 0) >= parameters.get('encounter_min_extent_px', 12))
            if legal and near and exposed:
                rows[key] = {**deepcopy(row), 'informative': True,
                    'informative_feature_evidence': {'kind': 'measured_visual_exposure',
                        'semantic_recognition_verified': False, 'visible_pixels': row['visible_pixels']},
                    'visit_verified': True, 'visit_evidence': {'snapshot_id': observation.snapshot_id,
                        'legal_observation_pose': True, 'maximum_distance_m': parameters.get('encounter_max_distance_m', 2.),
                        'policy': 'mounted_visibility_v1'}}
    return {**result, 'available': available, 'objects': rows,
        'semantic_recognition_verified': False, 'encounter_policy': 'mounted_visibility_v1'}

"""Reduce native spatial-tray evidence without advancing the simulator.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_spatial_tray.py
Call SpatialTrayProgress(parameters).begin(initial), update(sample), result().
All tray coordinates are explicitly world coordinates, and all thresholds are
provided by the authored design. A result read never advances stability time.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Real

import numpy as np

REVISION = 'wia_spatial_tray_v1'
SCORING_VERSION = 'wia_spatial_tray_progress_v2'
FINAL_STATE_REVISION = 'final_state_v1'
FINAL_STATE_SCORING_VERSION = 'wia_spatial_tray_final_state_progress_v1'
THRESHOLDS = (
    'minimum_lift_m', 'minimum_displacement_m', 'stability_duration_s',
    'maximum_linear_speed_m_s', 'maximum_angular_speed_rad_s', 'max_sample_gap_s',
    'fixture_position_tolerance_m', 'fixture_angle_tolerance_rad',
    'deck_height_tolerance_m', 'maximum_bottom_gap_m', 'penetration_tolerance_m',
    'minimum_separation_m',
)


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    return float(value) if np.isfinite(value) else None


def _vector(value, shape):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.shape != shape:
            return None
        result = raw.astype(float, copy=True)
        return result if np.isfinite(result).all() else None
    except (TypeError, ValueError, OverflowError):
        return None


def validate_parameters(parameters):
    """Validate the main-authored contract without inventing geometry or limits."""
    p = deepcopy(parameters)
    if p.get('revision') != REVISION:
        raise ValueError('An explicit wia_spatial_tray_v1 revision is required.')
    identity = p.get('scene_contract', {})
    if any(not isinstance(identity.get(key), str) or not identity[key].strip()
           for key in ('scene_model', 'room')):
        raise ValueError('An explicit scene_model and room contract is required.')
    targets = p.get('targets')
    if (not isinstance(targets, dict) or len(targets) not in (2, 3)
            or any(not isinstance(key, str) or not key or not isinstance(name, str) or not name
                   for key, name in targets.items()) or len(set(targets.values())) != len(targets)):
        raise ValueError('Two or three distinct exact target bindings are required.')
    supports = p.get('initial_supports', {})
    if (set(supports) != set(targets)
            or any(not isinstance(key, str) or not key for key in supports.values())):
        raise ValueError('Every target requires an explicit initial support binding.')
    tray = p.get('tray', {})
    if tray.get('mounting_mode') != 'fixed_fixture_on_native_table':
        raise ValueError('Declare the fixed_fixture_on_native_table mounting mode explicitly.')
    keys = [tray.get('deck_key'), tray.get('table_support_key'), *tray.get('rim_keys', [])]
    if (len(keys) < 3 or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys) or set(keys) & set(targets)
            or set(supports.values()) & {*targets, tray.get('deck_key'), *tray.get('rim_keys', [])}):
        raise ValueError('The tray requires distinct deck, table support and rim bindings.')
    bounds = _vector(tray.get('usable_world_xy_m'), (2, 2))
    if bounds is None or np.any(bounds[1] <= bounds[0]) or _number(tray.get('deck_top_world_z_m')) is None:
        raise ValueError('Explicit positive usable_world_xy_m bounds and deck_top_world_z_m are required.')
    scoring = p.get('scoring', {})
    revision = scoring.get('manipulation_revision')
    if 'manipulation_revision' in scoring and revision != FINAL_STATE_REVISION:
        raise ValueError('Unsupported spatial-tray manipulation_revision.')
    final_state = revision == FINAL_STATE_REVISION
    history_thresholds = ('minimum_lift_m', 'minimum_displacement_m') if final_state else ()
    if any(key not in scoring or scoring[key] is not None for key in history_thresholds):
        raise ValueError('Final-state spatial-tray history thresholds must be explicitly null.')
    if any(_number(scoring.get(key)) is None or scoring[key] <= 0
           for key in THRESHOLDS if key not in history_thresholds):
        raise ValueError('Every spatial-tray scoring threshold must be explicit, finite and positive.')
    # Existing shape classification is reused verbatim. These upper/lower limits
    # likewise prevent weakening the historical contact and release tolerances.
    if ((not final_state and scoring['minimum_lift_m'] < .05) or scoring['minimum_separation_m'] < .05
            or scoring['maximum_linear_speed_m_s'] > .02
            or scoring['maximum_angular_speed_rad_s'] > .1
            or scoring['penetration_tolerance_m'] > .001
            or scoring['maximum_bottom_gap_m'] > .015
            or scoring['deck_height_tolerance_m'] > .015
            or scoring['fixture_position_tolerance_m'] > .03
            or scoring['fixture_angle_tolerance_rad'] > np.deg2rad(15)):
        raise ValueError('The explicit design must not weaken existing contact/support tolerances.')
    goal = p.get('goal_relation', {})
    if goal.get('kind') == 'shape':
        if len(targets) != 3 or goal.get('shape') not in ('equilateral', 'isosceles', 'random triangle', 'collinear'):
            raise ValueError('Shape goals require three targets and an existing declared shape.')
    elif goal.get('kind') == 'contact':
        pair = goal.get('pair', [])
        if (len(pair) != 2 or len(set(pair)) != 2 or not set(pair).issubset(targets)
                or goal.get('relation') not in ('touching', 'separated')):
            raise ValueError('Contact goals require an explicit pair and touching or separated relation.')
    else:
        raise ValueError('An explicit shape or contact goal relation is required.')
    return p


class SpatialTrayProgress:
    """Reduce the declared manipulation contract and simultaneous terminal dwell."""

    def __init__(self, parameters):
        self.p = validate_parameters(parameters)
        self.scoring = self.p['scoring']
        self.final_state = self.scoring.get('manipulation_revision') == FINAL_STATE_REVISION
        self._begun = False

    def begin(self, sample):
        self._begun = True
        self._initial = deepcopy(sample)
        self._initial_errors = []
        self._violations = set()
        self._states = {key: {'acquired': False, 'moved': False, 'maximum_held_lift_m': None,
                              'maximum_displacement_m': None} for key in self.p['targets']}
        self._last_time = self._high_time = self._dwell_start = None
        self._stable_duration = 0.
        self._current = {}
        result = self._consume(sample, initial=True)
        self._initial_errors = list(result['measurement_errors'])
        if sample.get('initial_supports_valid') is not True:
            self._initial_errors.append('initial_separate_native_supports_unavailable_or_false')
        for key, row in sample.get('targets', {}).items():
            if key in self.p['targets'] and self._inside(row):
                self._initial_errors.append(key + ':initially_inside_destination')
        grasp = sample.get('grasp', {})
        if grasp.get('released') is not True or grasp.get('held_names') != []:
            self._initial_errors.append('initial_aggregate_hands_not_released')
        if self._initial_errors:
            self._current['available'] = False
        return self.result()

    def _inside(self, row):
        low, high = _vector(row.get('aabb_min'), (3,)), _vector(row.get('aabb_max'), (3,))
        bounds = np.asarray(self.p['tray']['usable_world_xy_m'])
        return bool(low is not None and high is not None and np.all(high > low)
                    and np.all(low[:2] >= bounds[0]) and np.all(high[:2] <= bounds[1]))

    def _relation(self, sample, rows):
        goal = self.p['goal_relation']
        if goal['kind'] == 'shape':
            from wia.authoring.converters.spatial_relations_geometric_configuration import shape_geometry
            centres = [(np.asarray(row['aabb_min'])[:2] + np.asarray(row['aabb_max'])[:2]) / 2
                       for row in rows.values()]
            return shape_geometry(centres, goal['shape'])
        geometry = deepcopy(sample.get('contact_geometry', {}))
        geometry['satisfied'] = bool(geometry.get('available') is True
            and geometry.get('relation') == goal['relation'] and geometry.get('penetrating') is False
            and (goal['relation'] != 'separated'
                 or (_number(geometry.get('minimum_separation_m')) is not None
                     and geometry['minimum_separation_m'] >= self.scoring['minimum_separation_m'])))
        return geometry

    def update(self, sample):
        if not self._begun:
            raise RuntimeError('Begin the spatial-tray episode before sampling.')
        return self._consume(sample, initial=False)

    def _consume(self, sample, *, initial):
        errors, rows = [], {}
        now = _number(sample.get('native_time_s'))
        gap = bool(now is not None and self._last_time is not None
                   and now - self._last_time > self.scoring['max_sample_gap_s'])
        if now is None:
            errors.append('native_time_unavailable')
        elif self._high_time is not None and now < self._high_time:
            self._violations.add('native_time_regressed')
        if now is not None:
            self._high_time = now if self._high_time is None else max(now, self._high_time)
        self._last_time = now
        for flag in ('control_valid', 'fixtures_valid', 'tray_valid'):
            if type(sample.get(flag)) is not bool:
                errors.append(flag + '_unavailable')
            elif sample[flag] is False:
                self._violations.add(flag + '_false')
        grasp = sample.get('grasp', {})
        held, released = grasp.get('held_names'), grasp.get('released')
        grasp_valid = (grasp.get('available') is True and isinstance(held, list)
                       and all(isinstance(name, str) and name for name in held)
                       and len(set(held)) == len(held) and type(released) is bool
                       and not (released and held))
        if not grasp_valid:
            errors.append('aggregate_grasp_unavailable_or_inconsistent')
            held, released = [], False
        checks = {}
        for key, name in self.p['targets'].items():
            row = deepcopy(sample.get('targets', {}).get(key, {}))
            rows[key] = row
            low, high = _vector(row.get('aabb_min'), (3,)), _vector(row.get('aabb_max'), (3,))
            position = _vector(row.get('position'), (3,))
            if (row.get('name') != name or low is None or high is None or position is None
                    or np.any(high <= low)):
                errors.append(key + ':native_identity_or_geometry_unavailable')
            scale_valid = True
            if self.final_state:
                scale = _vector(row.get('scale'), (3,))
                initial_scale = _vector(self._initial.get('targets', {}).get(key, {}).get('scale'), (3,))
                if scale is None or np.any(scale <= 0):
                    errors.append(key + ':native_scale_unavailable')
                    scale_valid = False
                elif initial_scale is None or np.any(initial_scale <= 0):
                    errors.append(key + ':initial_native_scale_unavailable')
                    scale_valid = False
                elif not np.array_equal(scale, initial_scale):
                    self._violations.add(key + ':asset_scale_changed')
                    scale_valid = False
            for field in ('linear_speed_m_s', 'angular_speed_rad_s'):
                if _number(row.get(field)) is None or row[field] < 0:
                    errors.append(key + ':' + field + '_unavailable')
            for field in ('dynamic_collidable', 'robot_contact', 'on_top_deck'):
                if type(row.get(field)) is not bool:
                    errors.append(key + ':' + field + '_unavailable')
            deck = row.get('deck_contact_geometry', {})
            if deck.get('available') is not True:
                errors.append(key + ':native_deck_contact_unavailable')
            if row.get('dynamic_collidable') is False:
                self._violations.add(key + ':target_not_dynamic_collidable')
            initial_row = self._initial.get('targets', {}).get(key, {})
            initial_position = _vector(initial_row.get('position'), (3,))
            initial_low = _vector(initial_row.get('aabb_min'), (3,))
            if not initial and position is not None and initial_position is not None:
                distance = float(np.linalg.norm(position - initial_position))
                state = self._states[key]
                previous = state['maximum_displacement_m']
                state['maximum_displacement_m'] = distance if previous is None else max(previous, distance)
                if not self.final_state:
                    state['moved'] |= distance >= self.scoring['minimum_displacement_m']
                if grasp_valid and name in held and low is not None and initial_low is not None:
                    lift = float(low[2] - initial_low[2])
                    previous = state['maximum_held_lift_m']
                    state['maximum_held_lift_m'] = lift if previous is None else max(previous, lift)
                    if not self.final_state:
                        state['acquired'] |= lift >= self.scoring['minimum_lift_m']
            gap_z = None if low is None else float(low[2] - self.p['tray']['deck_top_world_z_m'])
            checks[key] = {
                'wholly_inside_usable_xy': self._inside(row),
                'actual_nonpenetrating_deck_contact': bool(deck.get('available') is True
                    and deck.get('relation') == 'touching' and deck.get('penetrating') is False),
                'on_top_deck': row.get('on_top_deck') is True,
                'bottom_at_deck': gap_z is not None and abs(gap_z) <= self.scoring['maximum_bottom_gap_m'],
                'no_robot_support': row.get('robot_contact') is False,
                'dynamic_collidable': row.get('dynamic_collidable') is True,
                'stationary': bool(_number(row.get('linear_speed_m_s')) is not None
                    and _number(row.get('angular_speed_rad_s')) is not None
                    and row['linear_speed_m_s'] <= self.scoring['maximum_linear_speed_m_s']
                    and row['angular_speed_rad_s'] <= self.scoring['maximum_angular_speed_rad_s']),
            }
            if self.final_state:
                checks[key]['asset_scale_preserved'] = scale_valid
                checks[key]['original_identity_preserved'] = row.get('name') == name
        relation = {'available': False, 'satisfied': False}
        try:
            relation = self._relation(sample, rows)
            if self.p['goal_relation']['kind'] == 'contact' and relation.get('available') is not True:
                errors.append('native_pair_contact_unavailable')
        except (ValueError, KeyError, TypeError) as error:
            errors.append('declared_relation_unavailable:' + str(error))
        if gap:
            errors.append('native_sample_gap_exceeded')
        available = bool(not errors and not self._initial_errors)
        terminal = bool(available and not self._violations and released and not held
            and relation.get('satisfied') is True and all(all(value.values()) for value in checks.values())
            and (self.final_state or all(state['acquired'] and state['moved'] for state in self._states.values())))
        if not terminal or initial:
            self._dwell_start, self._stable_duration = None, 0.
        else:
            if self._dwell_start is None:
                self._dwell_start = now
            self._stable_duration = now - self._dwell_start
        self._current = {'available': available, 'measurement_errors': errors, 'native_time_s': now,
                         'terminal_checks': checks, 'declared_relation': relation,
                         'released_empty_hands': bool(grasp_valid and released and not held),
                         'terminal_valid': terminal, 'sample_gap_exceeded': gap}
        return self.result()

    def result(self):
        current = self._current
        stable = bool(current.get('terminal_valid') and self._stable_duration >= self.scoring['stability_duration_s'])
        first = ('targets_on_tray', bool(current.get('terminal_checks'))
                 and all(all(value for key, value in checks.items() if key != 'stationary')
                         for checks in current['terminal_checks'].values())) if self.final_state else (
                 'retrieve_all', all(state['acquired'] and state['moved'] for state in self._states.values()))
        checkpoints = [first,
                       ('declared_relation', current.get('declared_relation', {}).get('satisfied') is True),
                       ('stable_released_tray_placement', stable)]
        validity = 'invalid' if self._violations else 'complete' if current.get('available') else 'unavailable'
        score = sum(done for _, done in checkpoints) / len(checkpoints) if validity == 'complete' else None
        success = bool(validity == 'complete' and all(done for _, done in checkpoints))
        version = FINAL_STATE_SCORING_VERSION if self.final_state else SCORING_VERSION
        revision = {'manipulation_revision': FINAL_STATE_REVISION} if self.final_state else {}
        return deepcopy({'schema': version, 'scoring_version': version, **revision,
            'evaluation_validity': validity, 'task_success': success, 'physical_success': success,
            'completion_items': [{'id': key, 'completed': bool(done),
                'kind': 'historical_checkpoint' if key == 'retrieve_all' else 'terminal_condition'}
                for key, done in checkpoints], 'completed_goals': sum(done for _, done in checkpoints),
            'total_goals': len(checkpoints), 'completion_score': score,
            'score_percent': None if score is None else 100 * score,
            'per_target': self._states, 'stable_duration_s': self._stable_duration,
            'required_stability_duration_s': self.scoring['stability_duration_s'],
            'initial_measurement_errors': self._initial_errors,
            'constraint_violations': sorted(self._violations), **current})

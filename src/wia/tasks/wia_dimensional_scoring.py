"""Score measured dimensional manipulation without simulator imports or I/O.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_dimensional_scoring.py
Call DimensionalProgress(parameters).begin(initial_sample), update(native_sample),
and result(). Aperture tasks finalize only through declare_done().
Reading results never advances simulation time or stability dwell.
Aperture evidence is sampled passage, never a continuous-path certificate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from numbers import Real

import numpy as np

SCORING_VERSION = 'wia_dimensional_progress_v1'
APERTURE_SCORING_VERSION = 'wia_dimensional_progress_v2'
FINAL_STATE_SCORING_VERSION = 'wia_dimensional_final_state_v1'
_THRESHOLDS = (
    'minimum_lift_m', 'stability_duration_s', 'maximum_linear_speed_m_s',
    'maximum_angular_speed_rad_s', 'max_sample_gap_s', 'fixture_position_tolerance_m',
    'fixture_angle_tolerance_rad', 'dimension_relative_tolerance',
    'aperture_clearance_margin_m', 'minimum_assignment_gap_m',
)


def _number(value, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if np.isfinite(result) and (not nonnegative or result >= 0) else None


def _bool(value):
    return value if type(value) is bool else None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _vector(value, shape):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.shape != shape:
            return None
        result = raw.astype(float, copy=True)
        return result if np.isfinite(result).all() else None
    except (TypeError, ValueError, OverflowError):
        return None


class DimensionalProgress:
    """Measure actual pickup history and current released placement checkpoints.

    Parameters contain operation, candidates, ordered destinations, explicit
    scoring thresholds, and (for aperture tasks) native aperture bounds. Legacy
    unselected-position tolerances are inert. Proposal candidate dimensions are
    never used to infer selection, lift, assignment or passage.
    Explicit ``scoring.manipulation_revision='final_state_v1'`` with a null
    minimum lift removes pickup checkpoints; aperture passage is still sampled.
    Optional ``scoring.aperture_floor_contact_tolerance_m`` permits measured
    floor contact during passage; the default retains the original lower inset.
    """

    def __init__(self, parameters):
        p = deepcopy(dict(parameters))
        self.operation = p.get('operation')
        if self.operation not in ('select_pass_aperture', 'sort_intrinsic_dimension'):
            raise ValueError('Provide a supported dimensional operation.')
        candidates = p.get('candidates')
        if (not isinstance(candidates, Mapping) or not candidates
                or any(not isinstance(key, str) or not key or not isinstance(name, str) or not name
                       for key, name in candidates.items()) or len(set(candidates.values())) != len(candidates)):
            raise ValueError('Candidates must map distinct stable keys to distinct native object names.')
        self.candidates = dict(candidates)
        destinations = p.get('destinations')
        if (not isinstance(destinations, (list, tuple)) or not destinations
                or any(not isinstance(key, str) or not key for key in destinations)
                or len(set(destinations)) != len(destinations)):
            raise ValueError('Destinations must be distinct pad keys in ascending public order.')
        self.destinations = tuple(destinations)
        if self.operation == 'sort_intrinsic_dimension' and len(destinations) != len(candidates):
            raise ValueError('Sorting requires one destination for every original candidate.')
        self.scoring = deepcopy(dict(_mapping(p.get('scoring'))))
        self.manipulation_revision = self.scoring.get('manipulation_revision')
        if self.manipulation_revision not in (None, 'final_state_v1'):
            raise ValueError('Unsupported dimensional manipulation_revision.')
        if (self.manipulation_revision == 'final_state_v1'
                and ('minimum_lift_m' not in self.scoring or self.scoring['minimum_lift_m'] is not None)):
            raise ValueError('final_state_v1 requires minimum_lift_m to be explicitly None.')
        for key in _THRESHOLDS:
            if key == 'minimum_lift_m' and self.manipulation_revision == 'final_state_v1':
                continue
            value = _number(self.scoring.get(key), nonnegative=True)
            if value is None or (key in ('minimum_lift_m', 'stability_duration_s', 'max_sample_gap_s') and value == 0):
                raise ValueError(key + ' must be an explicit finite valid threshold.')
            self.scoring[key] = value
        if 'aperture_floor_contact_tolerance_m' in self.scoring:
            value = _number(self.scoring['aperture_floor_contact_tolerance_m'], nonnegative=True)
            if value is None:
                raise ValueError('aperture_floor_contact_tolerance_m must be an explicit finite nonnegative number.')
            self.scoring['aperture_floor_contact_tolerance_m'] = value
        self.aperture = None
        if self.operation == 'select_pass_aperture':
            self.aperture = {key: _number(_mapping(p.get('aperture')).get(key)) for key in
                ('entry_x_m', 'exit_x_m', 'y_min_m', 'y_max_m', 'z_min_m', 'z_max_m')}
            if any(value is None for value in self.aperture.values()):
                raise ValueError('Aperture tasks require all six finite measured native bounds.')
            a, margin = self.aperture, self.scoring['aperture_clearance_margin_m']
            if not (a['entry_x_m'] < a['exit_x_m'] and a['y_min_m'] + margin <= a['y_max_m'] - margin
                    and a['z_min_m'] + margin <= a['z_max_m'] - margin):
                raise ValueError('Measured native aperture bounds must define a nonempty inset opening.')
        self._reset()

    def _reset(self):
        self._begun = False
        self._initial_errors = []
        self._initial = {}
        self._assignment = None
        self._selected = None
        self._terminal_result = None
        self._done_declared = False
        self._pickup_enabled = False
        self._last_time = self._time_high_water = None
        self._violations = set()
        self._previous_vertices = {}
        self._states = {key: {
            'object_name': name, 'grasp_seen': False, 'lift_seen': False,
            'maximum_held_lift_m': None, 'currently_held': None,
            'expected_destination': self.destinations[0] if self.aperture is not None else None,
            'assignment_relations': None, 'assignment_satisfied': False,
            'dwell_started_at_s': None, 'stable_duration_s': 0., 'stable_placement': False,
            'measured': None, 'measurement_errors': ['episode_not_started'],
            'path': {'armed': False, 'slab_overlap_seen': False, 'completed': False,
                     'armed_at_s': None, 'last_whole_front_at_s': None,
                     'slab_overlap_at_s': None, 'completed_at_s': None,
                     'interruption_reason': None, 'completed_evidence': None},
        } for key, name in self.candidates.items()}
        self._current = {'available': False, 'native_time_s': None, 'final_released': None,
                         'sample_gap_exceeded': False, 'measurement_errors': ['episode_not_started']}

    def _target(self, key, value):
        row, errors = _mapping(value), []
        if row.get('object_name') != self.candidates[key]:
            errors.append('object_identity_unavailable_or_mismatched')
        dimensions = _vector(row.get('dimensions_m'), (3,))
        if dimensions is None or np.any(dimensions <= 0):
            dimensions = None
            errors.append('intrinsic_dimensions_unavailable')
        position, quaternion = _vector(row.get('position_m'), (3,)), _vector(row.get('quaternion_xyzw'), (4,))
        if position is None:
            errors.append('position_unavailable')
        if quaternion is None or not np.linalg.norm(quaternion):
            quaternion = None
            errors.append('quaternion_unavailable')
        try:
            vertices = np.asarray(row.get('vertices_m'))
            vertices = _vector(vertices, vertices.shape) if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) else None
        except (ValueError, TypeError):
            vertices = None
        if vertices is None:
            errors.append('collision_vertices_unavailable')
        values = {'dimensions_m': dimensions, 'position_m': position, 'quaternion_xyzw': quaternion,
                  'vertices_m': vertices}
        for field in ('lowest_z_m', 'linear_speed_m_s', 'angular_speed_rad_s'):
            values[field] = _number(row.get(field), nonnegative=field != 'lowest_z_m')
            if values[field] is None:
                errors.append(field + '_unavailable')
        for field in ('dynamic_collidable', 'robot_contact'):
            values[field] = _bool(row.get(field))
            if values[field] is None:
                errors.append(field + '_unavailable')
        values['destinations'] = {}
        for pad in self.destinations:
            relation = _mapping(_mapping(row.get('destinations')).get(pad))
            values['destinations'][pad] = {field: _bool(relation.get(field)) for field in
                                          ('touching', 'on_top', 'footprint_inside')}
            if any(value is None for value in values['destinations'][pad].values()):
                errors.append('destination:' + pad + ':native_relations_unavailable')
        return values, errors

    @staticmethod
    def _grasp(value):
        row = _mapping(value)
        names, released = row.get('held_names'), _bool(row.get('released'))
        available = (row.get('available') is True and isinstance(names, list)
            and all(isinstance(name, str) and name for name in names) and len(set(names)) == len(names)
            and released is not None and not (released and names))
        return (tuple(names), released) if available else (None, None)

    def begin(self, sample):
        """Freeze native initial dimensions, positions and pickup height references."""
        self._reset()
        self._begun = True
        sample = _mapping(sample)
        for key in self.candidates:
            row, errors = self._target(key, _mapping(sample.get('targets')).get(key))
            self._initial_errors.extend(key + ':' + error for error in errors)
            if row['dynamic_collidable'] is not True:
                self._initial_errors.append(key + ':initial_target_not_dynamic_collidable')
            self._initial[key] = {field: deepcopy(row[field]) for field in ('dimensions_m', 'position_m', 'lowest_z_m')}
            self._states[key].update(
                initial_dimensions_m=row['dimensions_m'].tolist() if row['dimensions_m'] is not None else None,
                initial_longest_dimension_m=float(max(row['dimensions_m'])) if row['dimensions_m'] is not None else None,
                initial_position_m=row['position_m'].tolist() if row['position_m'] is not None else None,
                baseline_lowest_z_m=row['lowest_z_m'])
            if (self.aperture is not None and row['vertices_m'] is not None
                    and not self._wholly_front(row['vertices_m'])):
                self._initial_errors.append(key + ':initial_body_not_wholly_near_aperture')
        if _number(sample.get('native_time_s')) is None:
            self._initial_errors.append('initial_native_time_unavailable')
        for flag in ('fixtures_valid', 'geometry_valid'):
            if sample.get(flag) is not True:
                self._initial_errors.append('initial_' + flag + '_unavailable_or_false')
        if self.operation == 'sort_intrinsic_dimension' and not self._initial_errors:
            ranked = sorted((float(max(row['dimensions_m'])), key) for key, row in self._initial.items())
            if any(right[0] == left[0] or right[0] - left[0] < self.scoring['minimum_assignment_gap_m']
                   for left, right in zip(ranked, ranked[1:])):
                self._initial_errors.append('initial_intrinsic_dimension_assignment_ambiguous')
            else:
                self._assignment = {key: pad for (_, key), pad in zip(ranked, self.destinations)}
                for key, pad in self._assignment.items():
                    self._states[key]['expected_destination'] = pad
        return self._consume(sample, initial=True)

    def update(self, sample):
        if self._terminal_result is not None:
            return self.result()
        if self._begun:
            return self._consume(_mapping(sample), initial=False)
        return self.result()

    def _reset_dwell(self, state):
        state.update(dwell_started_at_s=None, stable_duration_s=0., stable_placement=False)

    def _interrupt_path(self, key, reason):
        path = self._states[key]['path']
        path.update(armed=False, slab_overlap_seen=False, armed_at_s=None, slab_overlap_at_s=None,
                    interruption_reason=reason)
        self._previous_vertices.pop(key, None)

    def _wholly_front(self, vertices):
        return bool(np.max(vertices[:, 0]) < self.aperture['entry_x_m'] - self.scoring['aperture_clearance_margin_m'])

    def _update_path(self, key, vertices, now, *, duplicate):
        state, a = self._states[key], self.aperture
        path = state['path']
        if path['completed'] or duplicate:
            return
        front = self._wholly_front(vertices)
        if front:
            path['last_whole_front_at_s'] = now
        if self.manipulation_revision is None and not state['lift_seen']:
            return
        if not path['armed']:
            if front:
                path.update(armed=True, armed_at_s=now, interruption_reason=None)
                self._previous_vertices[key] = vertices.copy()
            return
        previous = self._previous_vertices[key]
        union = np.concatenate((previous, vertices), axis=0)
        union_reaches_slab = bool(np.min(union[:, 0]) <= a['exit_x_m'] and np.max(union[:, 0]) >= a['entry_x_m'])
        margin = self.scoring['aperture_clearance_margin_m']
        floor_minimum = (a['z_min_m'] - self.scoring['aperture_floor_contact_tolerance_m']
                         if 'aperture_floor_contact_tolerance_m' in self.scoring else a['z_min_m'] + margin)
        yz_clear = bool(np.all(union[:, 1] >= a['y_min_m'] + margin)
            and np.all(union[:, 1] <= a['y_max_m'] - margin)
            and np.all(union[:, 2] >= floor_minimum)
            and np.all(union[:, 2] <= a['z_max_m'] - margin))
        if union_reaches_slab and not yz_clear:
            self._interrupt_path(key, 'adjacent_sample_union_not_inside_inset_opening')
            return
        overlap = bool(np.min(vertices[:, 0]) <= a['exit_x_m'] and np.max(vertices[:, 0]) >= a['entry_x_m'])
        beyond = bool(np.min(vertices[:, 0]) > a['exit_x_m'] + margin)
        if overlap:
            path.update(slab_overlap_seen=True, slab_overlap_at_s=now)
        if beyond:
            if not path['slab_overlap_seen']:
                self._interrupt_path(key, 'beyond_without_measured_slab_overlap')
                return
            path.update(completed=True, completed_at_s=now, completed_evidence={
                'front_at_s': path['armed_at_s'], 'slab_overlap_at_s': path['slab_overlap_at_s'],
                'beyond_at_s': now, 'beyond_min_x_m': float(np.min(vertices[:, 0])),
                'sampled_passage': True, 'continuous_path_certified': False})
        elif front and path['slab_overlap_seen']:
            self._interrupt_path(key, 'returned_wholly_front_before_completion')
        self._previous_vertices[key] = vertices.copy()

    def _consume(self, sample, *, initial):
        errors = []
        now = _number(sample.get('native_time_s'))
        previous_time = self._last_time
        duplicate = now is not None and now == previous_time
        gap = bool(now is not None and previous_time is not None
                   and now - previous_time > self.scoring['max_sample_gap_s'])
        if now is None:
            errors.append('native_time_unavailable')
        elif self._time_high_water is not None and now < self._time_high_water:
            self._violations.add('native_time_regressed')
        if now is not None:
            self._time_high_water = now if self._time_high_water is None else max(now, self._time_high_water)
        self._last_time = now
        flags = {key: _bool(sample.get(key)) for key in ('control_valid', 'fixtures_valid', 'geometry_valid')}
        for flag, value in flags.items():
            if value is None:
                errors.append(flag + '_unavailable')
            elif value is False and (not initial or flag == 'control_valid'):
                self._violations.add(flag + '_false')
        held, released = self._grasp(sample.get('grasp'))
        if held is None:
            errors.append('grasp_unavailable_or_inconsistent')
        rows = {}
        for key, state in self._states.items():
            row, target_errors = self._target(key, _mapping(sample.get('targets')).get(key))
            rows[key] = row
            state['measurement_errors'] = target_errors
            errors.extend(key + ':' + reason for reason in target_errors)
            reference = self._initial[key]
            dimensions = row['dimensions_m']
            if dimensions is not None and reference['dimensions_m'] is not None:
                if np.any(np.abs(dimensions / reference['dimensions_m'] - 1.) > self.scoring['dimension_relative_tolerance']):
                    self._violations.add(key + ':intrinsic_dimensions_changed')
            if not initial and row['dynamic_collidable'] is False:
                self._violations.add(key + ':dynamic_collidable_false')
            state['currently_held'] = None if held is None else state['object_name'] in held
            state['measured'] = {field: value.tolist() if isinstance(value, np.ndarray) else deepcopy(value)
                                for field, value in row.items() if field != 'vertices_m'}
        available = not errors and not self._initial_errors
        trustworthy = bool(available and all(value is True for value in flags.values()) and not self._violations)
        if held is not None and released is True and not held:
            self._pickup_enabled = True
        if trustworthy and not initial and self._pickup_enabled:
            for key, state in self._states.items():
                if state['object_name'] in held:
                    state['grasp_seen'] = True
                    lift = rows[key]['lowest_z_m'] - self._initial[key]['lowest_z_m']
                    state['maximum_held_lift_m'] = lift if state['maximum_held_lift_m'] is None else max(lift, state['maximum_held_lift_m'])
                    if (self.manipulation_revision is None
                            and lift >= self.scoring['minimum_lift_m']):
                        state['lift_seen'] = True
        if gap:
            errors.append('native_sample_gap_exceeded')
        uninterrupted = bool(trustworthy and not self._violations and not gap)
        for key, state in self._states.items():
            row = rows[key]
            if not uninterrupted:
                self._reset_dwell(state)
                self._interrupt_path(key, 'native_sample_gap_exceeded' if gap else 'unavailable_or_invalid_sample')
            if self.aperture is not None and uninterrupted:
                self._update_path(key, row['vertices_m'], now, duplicate=duplicate)
            pad = state['expected_destination']
            relation = row['destinations'].get(pad) if pad is not None else None
            assigned = bool(relation is not None and all(value is True for value in relation.values()))
            state.update(assignment_relations=deepcopy(relation), assignment_satisfied=assigned)
            stable = bool(uninterrupted and assigned and released is True and held == ()
                          and row['robot_contact'] is False and row['dynamic_collidable'] is True
                          and row['linear_speed_m_s'] <= self.scoring['maximum_linear_speed_m_s']
                          and row['angular_speed_rad_s'] <= self.scoring['maximum_angular_speed_rad_s'])
            if not stable:
                self._reset_dwell(state)
            else:
                if state['dwell_started_at_s'] is None:
                    state['dwell_started_at_s'] = now
                state['stable_duration_s'] = now - state['dwell_started_at_s']
                state['stable_placement'] = state['stable_duration_s'] >= self.scoring['stability_duration_s']
        self._current = {'available': available and not gap, 'native_time_s': now, 'final_released': released,
                         'sample_gap_exceeded': gap, 'measurement_errors': errors,
                         'control_flags': flags, 'held_names': list(held) if held is not None else None}
        return self.result()

    def declare_done(self):
        """Latch one aperture verdict; sorting keeps its original live semantics."""
        if self.operation != 'select_pass_aperture':
            return self.result()
        if self._terminal_result is None:
            def rank(key):
                state = self._states[key]
                grasp = state['grasp_seen']
                lift = state['lift_seen']
                passage = state['path']['completed']
                placement = state['stable_placement']
                if self.manipulation_revision == 'final_state_v1':
                    return (passage and placement, passage, placement)
                return (grasp and lift and passage and placement, placement, passage, lift, grasp)

            if self.manipulation_revision == 'final_state_v1':
                attempts = [key for key in self.candidates if self._states[key]['path']['completed']
                            or self._states[key]['path']['slab_overlap_seen']
                            or self._states[key]['stable_placement']]
            else:
                attempts = [key for key in self.candidates if self._states[key]['grasp_seen']]
            self._selected = max(attempts, key=rank) if attempts else None
            self._done_declared = True
            self._terminal_result = self.result()
        return self.result()

    def result(self):
        """Return detached JSON-safe evidence; no reads advance any state."""
        if self._terminal_result is not None:
            return deepcopy(self._terminal_result)
        current = self._current
        aperture = self.operation == 'select_pass_aperture'
        if aperture:
            selected = self._states.get(self._selected)
            checkpoints = [('passage', bool(selected and selected['path']['completed'])),
                ('stable_placement', bool(selected and selected['stable_placement']))]
            if self.manipulation_revision is None:
                checkpoints = [('choice', bool(selected and selected['grasp_seen'])),
                    ('grasp_lift', bool(selected and selected['grasp_seen'] and selected['lift_seen']))] + checkpoints
        else:
            checkpoints = [('assignment', self._assignment is not None and all(state['assignment_satisfied'] for state in self._states.values())),
                ('stable_placement', all(state['stable_placement'] for state in self._states.values()))]
            if self.manipulation_revision is None:
                checkpoints = [('grasp_lift', all(state['grasp_seen'] and state['lift_seen']
                    for state in self._states.values()))] + checkpoints
        validity = 'invalid' if self._violations else 'complete' if current['available'] else 'unavailable'
        success = bool(validity == 'complete' and all(completed for _, completed in checkpoints))
        score = sum(completed for _, completed in checkpoints) / len(checkpoints) if validity == 'complete' else None
        items = [{'id': name, 'completed': bool(completed),
                  'evidence_scope': 'measured_grasp_only_not_feasibility' if name == 'choice' else
                      'sampled_passage_not_continuous_path_certificate' if name == 'passage' else 'measured_native_checkpoint'}
                 for name, completed in checkpoints]
        version = (FINAL_STATE_SCORING_VERSION if self.manipulation_revision == 'final_state_v1'
                   else APERTURE_SCORING_VERSION if aperture else SCORING_VERSION)
        result = {'schema': version, 'scoring_version': version,
            'task_success': success, 'physical_success': success, 'evaluation_validity': validity,
            'constraints_satisfied': False if self._violations else True if validity == 'complete' else None,
            'completion_items': items, 'completion_score': score, 'score_percent': None if score is None else score * 100,
            'completed_goals': sum(completed for _, completed in checkpoints), 'total_goals': len(checkpoints),
            'operation': self.operation, 'selected_candidate': self._selected,
            'expected_assignment': deepcopy(self._assignment), 'per_target': self._states,
            'initial_measurement_errors': self._initial_errors, 'constraint_violations': sorted(self._violations),
            'native_time_s': current['native_time_s'], 'final_released': current['final_released'],
            'sample_gap_exceeded': current['sample_gap_exceeded'],
            'measurement_errors': current['measurement_errors'], 'scoring': self.scoring,
            'continuous_path_certified': False, 'physical_feasibility_of_unselected_candidates_inferred': False}
        if self.manipulation_revision is not None:
            result['manipulation_revision'] = self.manipulation_revision
        if aperture:
            result['evaluation_finalized'] = self._done_declared
            result['native_evaluation_validity'] = validity
            if not self._done_declared:
                result.update(evaluation_validity='pending_done', task_success=False, physical_success=False,
                    constraints_satisfied=None, selected_candidate=None, completion_items=None,
                    completion_score=None, score_percent=None, completed_goals=None, total_goals=None)
        return deepcopy(result)

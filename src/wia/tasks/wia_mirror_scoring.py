"""Score native mirror-task evidence without simulator imports or I/O.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_mirror_scoring.py
Use MirrorProgress(parameters).begin(sample), update(sample), report_answer(text),
and result(). Reports and result reads never advance native time or dwell.
Geometric optical selection does not certify visibility or optical necessity.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from wia.tasks.wia_dimensional_scoring import _bool, _mapping, _number, _vector


SCORING_VERSION = 'wia_mirror_progress_v1'
FINAL_STATE_REVISION = 'final_state_v1'
FINAL_STATE_SCORING_VERSION = 'wia_mirror_final_state_progress_v1'
_COMPLETION = {
    'mirror_select_deliver': ('selection', 'grasp_lift', 'stable_placement'),
    'classify_appearance_deliver': ('appearance_report', 'selection', 'grasp_lift', 'stable_placement'),
    'mirror_reconfigure': ('selection', 'grasp_lift', 'reconfiguration', 'stable_placement'),
}
_FINAL_COMPLETION = {
    'mirror_select_deliver': ('selected_original_at_destination', 'stable_placement'),
    'classify_appearance_deliver': ('appearance_report', 'selected_original_at_destination', 'stable_placement'),
    'mirror_reconfigure': ('selected_original_at_destination', 'reconfiguration', 'stable_placement'),
}
_THRESHOLDS = (
    'minimum_lift_m', 'stability_duration_s', 'maximum_linear_speed_m_s',
    'maximum_angular_speed_rad_s', 'max_sample_gap_s', 'fixture_position_tolerance_m',
    'fixture_angle_tolerance_rad', 'dimension_relative_tolerance',
    'minimum_assignment_gap_m', 'unselected_position_tolerance_m', 'minimum_required_motion_m',
)
DESTINATION_REVISION = 'wia_mirror_v4'


def destination_contracts(parameters):
    """Validate explicit unscaled root-link-local bounds for v4 destinations."""
    p = parameters
    contracts = p.get('destination_contracts')
    keys = {p.get('table_key'), *_mapping(p.get('destinations')).values()}
    if (not isinstance(contracts, Mapping) or set(contracts) != keys
            or not keys.issubset(p.get('fixture_keys', []))):
        raise ValueError('V4 requires explicit contracts for every fixed destination binding.')
    for key, contract in contracts.items():
        if (not isinstance(key, str) or not key or not isinstance(contract, Mapping)
                or set(contract) != {'kind', 'usable_bounds_local_m'}
                or contract.get('kind') not in ('on_top', 'inside')):
            raise ValueError('Destination contracts require kind on_top or inside and local usable bounds.')
        bounds = _vector(contract.get('usable_bounds_local_m'), (2, 3))
        if bounds is None or np.any(bounds[1] <= bounds[0]):
            raise ValueError('Destination usable bounds must be finite with three positive extents.')
    return deepcopy(dict(contracts))


def _json_value(value):
    """Detach native diagnostic values without emitting nonfinite JSON scalars."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return _number(value)


class MirrorProgress:
    """Freeze optical identity and reduce the declared placement contract."""

    def __init__(self, parameters):
        p = deepcopy(dict(parameters))
        self.operation = p.get('operation')
        if self.operation not in _COMPLETION:
            raise ValueError('Provide a supported mirror operation.')
        candidates = p.get('candidates')
        if (not isinstance(candidates, Mapping) or not candidates
                or any(not isinstance(key, str) or not key or not isinstance(name, str) or not name
                       for key, name in candidates.items())
                or len(set(candidates.values())) != len(candidates)):
            raise ValueError('Candidates must map distinct stable keys to distinct native names.')
        self.candidates = dict(candidates)
        self._keys_by_name = {name: key for key, name in self.candidates.items()}
        self.appearance_mode = p.get('appearance_mode')
        if self.appearance_mode not in ('direct', 'reflected'):
            raise ValueError('Provide the declared direct or reflected appearance_mode.')
        self.destinations = dict(_mapping(p.get('destinations')))
        if any(not isinstance(self.destinations.get(key), str) or not self.destinations[key]
               for key in ('blue', 'yellow')):
            raise ValueError('Provide blue and yellow destination bindings.')
        self.bounded_destinations = p.get('revision') == DESTINATION_REVISION
        self.destination_contracts = destination_contracts(p) if self.bounded_destinations else {}
        self.delivery_destination, self.spatial_destination = 'blue', 'table'
        if self.bounded_destinations:
            self.delivery_destination = p.get('delivery_destination', 'blue')
            self.spatial_destination = p.get('spatial_destination', 'table')
            if (self.delivery_destination not in ('blue', 'yellow')
                    or self.spatial_destination not in ('table', 'blue', 'yellow')):
                raise ValueError('Unsupported delivery_destination or spatial_destination.')
        self.destination_kinds = ({'table': self.destination_contracts[p['table_key']]['kind'],
            **{label: self.destination_contracts[key]['kind'] for label, key in self.destinations.items()}}
            if self.bounded_destinations else {})
        self.scoring = deepcopy(dict(_mapping(p.get('scoring'))))
        revision = self.scoring.get('manipulation_revision')
        if 'manipulation_revision' in self.scoring and revision != FINAL_STATE_REVISION:
            raise ValueError('Unsupported mirror manipulation_revision.')
        self.final_state = revision == FINAL_STATE_REVISION
        self.completion = _FINAL_COMPLETION if self.final_state else _COMPLETION
        for key in _THRESHOLDS:
            if self.final_state and key in ('minimum_lift_m', 'minimum_required_motion_m'):
                if key not in self.scoring or self.scoring[key] is not None:
                    raise ValueError(key + ' must be explicitly null for final-state placement.')
                continue
            value = _number(self.scoring.get(key), nonnegative=True)
            if value is None or (key in ('minimum_lift_m', 'stability_duration_s', 'max_sample_gap_s') and value == 0):
                raise ValueError(key + ' must be an explicit finite valid threshold.')
            self.scoring[key] = value
        self.relation, self.distance_gap = None, None
        self.spatial = self.operation == 'mirror_reconfigure'
        self.swap = False
        if self.spatial:
            self.relation = p.get('relation')
            if self.relation not in ('selected_farther', 'selected_nearer', 'reverse_initial_distance_order'):
                raise ValueError('Provide a supported fixed-reference spatial relation.')
            self.distance_gap = _number(p.get('distance_gap_m'), nonnegative=True)
            self.swap = self.relation == 'reverse_initial_distance_order'
            if (len(candidates) != 2 or self.distance_gap is None or self.distance_gap == 0
                    or (not self.final_state and p.get('requires_both_objects_moved') is not self.swap)
                    or p.get('non_goal_initialization_required') is not True):
                raise ValueError('Spatial tasks require two originals and the declared motion/initial-goal gates.')
        declared = p.get('completion_items')
        if declared is not None and [item.get('id') for item in declared] != list(self.completion[self.operation]):
            raise ValueError('Completion item IDs must match the fixed mirror operation contract.')
        self._reset()

    def _reset(self):
        self._begun = False
        self._initial = {}
        self._initial_errors = []
        self._violations = set()
        self._selection = None
        self._optical = None
        self._anchor = None
        self._initial_order = None
        self._initial_distances = None
        self._committed = None
        self._pickup_enabled = False
        self._last_time = self._high_water = None
        self._latest_report = None
        self._report_frozen = False
        self._frozen_report = None
        self._report_frozen_at = None
        self._reports = []
        self._dwell_start = None
        self._stable_duration = 0.
        self._stable_placement = False
        self._states = {key: {
            'object_name': name, 'grasp_seen': False, 'lift_seen': False,
            'first_grasp_at_s': None, 'lift_at_s': None, 'maximum_held_lift_m': None,
            'currently_held': None, 'centroid_m': None, 'xy_displacement_m': None,
            'distance_to_frozen_anchor_m': None, 'placement_relations': None,
            'placement_satisfied': False, 'measured': None, 'measurement_errors': [],
        } for key, name in self.candidates.items()}
        self._current = {'available': False, 'native_time_s': None, 'held_names': None,
                         'final_released': None, 'measurement_errors': ['episode_not_started'],
                         'sample_gap_exceeded': False, 'reconfiguration_satisfied': False,
                         'unselected_objects_preserved': False}

    def _target(self, key, value):
        row, errors = _mapping(value), []
        identity_valid = row.get('object_name') == self.candidates[key]
        if not identity_valid:
            errors.append('object_identity_unavailable_or_mismatched')
        dimensions = _vector(row.get('dimensions_m'), (3,))
        if dimensions is None or np.any(dimensions <= 0):
            dimensions = None
            errors.append('intrinsic_dimensions_unavailable')
        values = {'dimensions_m': dimensions}
        for field, shape in (('position_m', (3,)), ('quaternion_xyzw', (4,))):
            values[field] = _vector(row.get(field), shape)
            if values[field] is None or (field == 'quaternion_xyzw' and not np.any(values[field])):
                values[field] = None
                errors.append(field + '_unavailable')
        try:
            vertices = np.asarray(row.get('vertices_m'))
            vertices = _vector(vertices, vertices.shape) if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) else None
        except (TypeError, ValueError):
            vertices = None
        values['vertices_m'] = vertices
        if vertices is None:
            errors.append('collision_vertices_unavailable')
        values['centroid_m'] = _json_value(row.get('centroid_m'))
        with np.errstate(over='ignore', invalid='ignore'):
            values['collision_centroid_m'] = None if vertices is None else _vector(vertices.mean(axis=0), (3,))
        if vertices is not None and values['collision_centroid_m'] is None:
            errors.append('collision_centroid_unavailable')
        for field in ('lowest_z_m', 'linear_speed_m_s', 'angular_speed_rad_s'):
            values[field] = _number(row.get(field), nonnegative=field != 'lowest_z_m')
            if values[field] is None:
                errors.append(field + '_unavailable')
        for field in ('dynamic_collidable', 'robot_contact'):
            values[field] = _bool(row.get(field))
            if values[field] is None:
                errors.append(field + '_unavailable')
        values['destinations'] = {}
        for pad in (self.spatial_destination,) if self.spatial else ('blue', 'yellow'):
            relation = _mapping(_mapping(row.get('destinations')).get(pad))
            kind = self.destination_kinds.get(pad, 'on_top')
            fields = ('touching', 'inside', 'vertices_inside') if kind == 'inside' else ('touching', 'on_top', 'footprint_inside')
            values['destinations'][pad] = {field: _bool(relation.get(field)) for field in fields}
            if self.bounded_destinations and relation.get('kind') != kind:
                errors.append('destination:' + pad + ':relation_kind_unavailable_or_mismatched')
            if any(value is None for value in values['destinations'][pad].values()):
                errors.append('destination:' + pad + ':native_relations_unavailable')
            if self.bounded_destinations:
                values['destinations'][pad]['kind'] = kind
        values['identity_valid'] = identity_valid
        return values, errors

    @staticmethod
    def _grasp(value):
        row = _mapping(value)
        held, released = row.get('held_names'), _bool(row.get('released'))
        available = (row.get('available') is True and isinstance(held, list)
            and all(isinstance(name, str) and name for name in held) and len(set(held)) == len(held)
            and released is not None and not (held and released))
        return (tuple(held), released) if available else (None, None)

    def _distances(self, rows):
        if self._anchor is None:
            return None
        if any(row['collision_centroid_m'] is None for row in rows.values()):
            return None
        with np.errstate(over='ignore', invalid='ignore'):
            distances = {key: _number(np.linalg.norm(row['collision_centroid_m'][:2] - self._anchor[:2]))
                         for key, row in rows.items()}
        return distances if all(value is not None for value in distances.values()) else None

    def _relation_satisfied(self, distances):
        if distances is None or self._selection is None:
            return False
        if self.swap:
            if self._initial_order is None:
                return False
            near, far = self._initial_order
            return bool(distances[near] - distances[far] >= self.distance_gap)
        other = next(key for key in self.candidates if key != self._selection)
        delta = distances[self._selection] - distances[other]
        return bool((delta if self.relation == 'selected_farther' else -delta) >= self.distance_gap)

    def begin(self, sample):
        """Reset the episode and freeze only measured initial optical/physical evidence."""
        self._reset()
        self._begun = True
        sample = _mapping(sample)
        optical = _mapping(sample.get('optical_selection'))
        self._optical = _json_value(optical)
        eye = _vector(optical.get('frozen_eye_m'), (3,))
        if (optical.get('available') is not True or optical.get('accepted') is not True
                or not isinstance(optical.get('selected_candidate'), str)
                or optical.get('selected_candidate') not in self.candidates or eye is None
                or optical.get('appearance_mode') not in ('direct', 'reflected')):
            self._initial_errors.append('initial_optical_selection_unavailable_or_ambiguous')
        else:
            self._selection = optical['selected_candidate']
            if optical['appearance_mode'] != self.appearance_mode:
                self._initial_errors.append('measured_appearance_mode_mismatches_configuration')
        rows = {}
        for key in self.candidates:
            row, errors = self._target(key, _mapping(sample.get('targets')).get(key))
            rows[key] = row
            self._initial_errors.extend(key + ':' + reason for reason in errors)
            if row['dynamic_collidable'] is not True:
                self._initial_errors.append(key + ':initial_target_not_dynamic_collidable')
            self._initial[key] = {field: deepcopy(row[field]) for field in
                ('dimensions_m', 'position_m', 'lowest_z_m', 'collision_centroid_m')}
        if _number(sample.get('native_time_s')) is None:
            self._initial_errors.append('initial_native_time_unavailable')
        if self.spatial:
            self._anchor = _vector(sample.get('anchor_centroid_m'), (3,))
            if self._anchor is None:
                self._initial_errors.append('initial_anchor_centroid_unavailable')
            self._initial_distances = self._distances(rows)
            if self._initial_distances is None:
                self._initial_errors.append('initial_fixed_anchor_distances_unavailable')
            if self.swap and self._initial_distances is not None:
                ranked = sorted((distance, key) for key, distance in self._initial_distances.items())
                gap = ranked[1][0] - ranked[0][0]
                if gap == 0 or gap < self.scoring['minimum_assignment_gap_m']:
                    self._initial_errors.append('initial_distance_order_ambiguous')
                else:
                    self._initial_order = [key for _, key in ranked]
            self._initial_goal_satisfied = self._relation_satisfied(self._initial_distances)
            if self._initial_goal_satisfied and not self.final_state:
                self._initial_errors.append('initial_spatial_goal_already_satisfied')
        return self._consume(sample, initial=True)

    def report_answer(self, text):
        """Record classification text without changing physics or stability time."""
        normalized = None
        if isinstance(text, str):
            value = text.strip()
            if value[:7].lower() == 'answer:':
                value = value[7:].strip()
            value = value.lower()
            if value in ('direct', 'reflected'):
                normalized = value
        self._reports.append({'text': text if isinstance(text, str) else None,
            'normalized': normalized, 'valid': normalized is not None,
            'after_first_grasp': (any(state['grasp_seen'] for state in self._states.values())
                                  if self.final_state else self._report_frozen),
            'native_time_s': self._current['native_time_s']})
        if normalized is not None and (self.final_state or not self._report_frozen):
            self._latest_report = normalized
        return self.result()

    def update(self, sample):
        return self._consume(_mapping(sample), initial=False) if self._begun else self.result()

    def _reset_dwell(self):
        self._dwell_start, self._stable_duration, self._stable_placement = None, 0., False

    def _consume(self, sample, *, initial):
        errors = []
        now = _number(sample.get('native_time_s'))
        increasing = now is not None and self._high_water is not None and now > self._high_water
        gap = bool(now is not None and self._last_time is not None
                   and now - self._last_time > self.scoring['max_sample_gap_s'])
        if now is None:
            errors.append('native_time_unavailable')
        elif self._high_water is not None and now < self._high_water:
            self._violations.add('native_time_regressed')
        if now is not None:
            self._high_water = now if self._high_water is None else max(now, self._high_water)
        self._last_time = now
        flags = {key: _bool(sample.get(key)) for key in ('control_valid', 'fixtures_valid', 'geometry_valid')}
        for key, value in flags.items():
            if value is None:
                errors.append(key + '_unavailable')
            elif value is False:
                self._violations.add(key + '_false')
        held, released = self._grasp(sample.get('grasp'))
        if held is None:
            errors.append('grasp_unavailable_or_inconsistent')
        rows, unselected_preserved = {}, True
        for key, state in self._states.items():
            row, target_errors = self._target(key, _mapping(sample.get('targets')).get(key))
            rows[key] = row
            state['measurement_errors'] = target_errors
            errors.extend(key + ':' + reason for reason in target_errors)
            reference = self._initial[key]
            dimensions = row['dimensions_m']
            if row['identity_valid'] and dimensions is not None and reference['dimensions_m'] is not None:
                if np.any(np.abs(dimensions / reference['dimensions_m'] - 1.) > self.scoring['dimension_relative_tolerance']):
                    self._violations.add(key + ':intrinsic_dimensions_changed')
            if row['dynamic_collidable'] is False:
                self._violations.add(key + ':dynamic_collidable_false')
            state['currently_held'] = None if held is None else state['object_name'] in held
            centroid = row['collision_centroid_m']
            state['centroid_m'] = None if centroid is None else centroid.tolist()
            with np.errstate(over='ignore', invalid='ignore'):
                state['xy_displacement_m'] = (None if centroid is None or reference['collision_centroid_m'] is None
                    else _number(np.linalg.norm(centroid[:2] - reference['collision_centroid_m'][:2])))
            if self.spatial and state['xy_displacement_m'] is None:
                errors.append(key + ':native_xy_displacement_unavailable')
            state['measured'] = _json_value({field: value for field, value in row.items() if field != 'vertices_m'})
            if (not self.swap and self._selection is not None and key != self._selection
                    and row['identity_valid'] and row['position_m'] is not None and reference['position_m'] is not None):
                preserved = bool(np.linalg.norm(row['position_m'] - reference['position_m'])
                                 <= self.scoring['unselected_position_tolerance_m'])
                unselected_preserved &= preserved
                if not preserved and not self.final_state:
                    self._violations.add(key + ':unselected_object_moved')
                if self.final_state:
                    state['unselected_position_preserved'] = preserved
        if self.spatial:
            anchor = _vector(sample.get('anchor_centroid_m'), (3,))
            if anchor is None:
                errors.append('anchor_centroid_unavailable')
            elif self._anchor is not None and np.linalg.norm(anchor - self._anchor) > self.scoring['fixture_position_tolerance_m']:
                self._violations.add('fixed_anchor_changed')
        if not initial and increasing and held is not None and self._pickup_enabled and held:
            candidate_keys = [self._keys_by_name[name] for name in held if name in self._keys_by_name]
            if candidate_keys and not self._report_frozen and not self.final_state:
                self._report_frozen = True
                self._frozen_report = self._latest_report
                self._report_frozen_at = now
            if len(candidate_keys) != len(held) and not self.final_state:
                self._violations.add('grasped_object_not_original_candidate')
            if not self.swap and candidate_keys and not self.final_state:
                if self._committed is None:
                    self._committed = candidate_keys[0] if len(candidate_keys) == 1 else None
                if len(candidate_keys) != 1 or any(
                        (self._selection is not None and key != self._selection) or key != self._committed
                        for key in candidate_keys):
                    self._violations.add('wrong_or_different_candidate_grasped')
            for key in candidate_keys:
                row, state = rows[key], self._states[key]
                state['grasp_seen'] = True
                if state['first_grasp_at_s'] is None:
                    state['first_grasp_at_s'] = now
                baseline = self._initial[key]['lowest_z_m']
                if row['identity_valid'] and row['lowest_z_m'] is not None and baseline is not None and flags['geometry_valid'] is True:
                    lift = _number(row['lowest_z_m'] - baseline)
                    if lift is None:
                        errors.append(key + ':held_lift_unavailable')
                    else:
                        prior = state['maximum_held_lift_m']
                        state['maximum_held_lift_m'] = lift if prior is None else max(prior, lift)
                        if not self.final_state and lift >= self.scoring['minimum_lift_m']:
                            state['lift_seen'] = True
                            if state['lift_at_s'] is None:
                                state['lift_at_s'] = now
        if held == () and released is True and now is not None:
            self._pickup_enabled = True
        distances = self._distances(rows) if self.spatial else None
        if self.spatial and distances is None:
            errors.append('fixed_anchor_distances_unavailable')
        for key, state in self._states.items():
            state['distance_to_frozen_anchor_m'] = None if distances is None else distances[key]
        relation_satisfied = self._relation_satisfied(distances) if self.spatial else False
        required = list(self.candidates) if self.spatial else [self._selection] if self._selection is not None else []
        motion_satisfied = (self.spatial and bool(required) and (self.final_state or all(
            self._states[key]['xy_displacement_m'] is not None
            and self._states[key]['xy_displacement_m'] >= self.scoring['minimum_required_motion_m']
            for key in (required if self.swap else [self._selection]))))
        if gap:
            errors.append('native_sample_gap_exceeded')
        available = not errors and not self._initial_errors
        uninterrupted = bool(available and not self._violations)
        destination = (self.spatial_destination if self.spatial else self.delivery_destination if self.operation == 'mirror_select_deliver'
                       or _mapping(self._optical).get('appearance_mode') == 'reflected' else 'yellow')
        stable = bool(uninterrupted and required and held == () and released is True
                      and (not self.final_state or unselected_preserved)
                      and (not self.spatial or (relation_satisfied and motion_satisfied)))
        for key, state in self._states.items():
            row = rows[key]
            relation = row['destinations'].get(destination) if key in required else None
            placed = bool(relation is not None and (not self.final_state or row['identity_valid'])
                          and all(value is True for field, value in relation.items() if field != 'kind'))
            state.update(placement_relations=deepcopy(relation), placement_satisfied=placed)
            if key in required:
                stable = bool(stable and placed and row['robot_contact'] is False
                    and row['dynamic_collidable'] is True
                    and row['linear_speed_m_s'] <= self.scoring['maximum_linear_speed_m_s']
                    and row['angular_speed_rad_s'] <= self.scoring['maximum_angular_speed_rad_s'])
        if not stable:
            self._reset_dwell()
        elif self._dwell_start is None:
            self._dwell_start = now
        elif increasing:
            self._stable_duration = now - self._dwell_start
            self._stable_placement = self._stable_duration >= self.scoring['stability_duration_s']
        self._current = {'available': available, 'native_time_s': now, 'held_names': None if held is None else list(held),
            'final_released': released, 'measurement_errors': errors, 'sample_gap_exceeded': gap,
            'reconfiguration_satisfied': bool(relation_satisfied and motion_satisfied),
            'distance_relation_satisfied': relation_satisfied,
            'required_motion_satisfied': None if self.final_state else bool(motion_satisfied),
            'unselected_objects_preserved': unselected_preserved,
            'destination': destination, 'distances_m': distances, 'control_flags': flags}
        return self.result()

    def result(self):
        """Return detached evidence without changing reports, event history or time."""
        required = list(self.candidates) if self.swap else [self._selection] if self._selection is not None else []
        selected = bool(required and (all(self._states[key]['placement_satisfied'] for key in required)
                        if self.final_state else all(self._states[key]['grasp_seen'] for key in required)
                        and (self.swap or self._committed == self._selection)))
        lifted = bool(required and all(self._states[key]['grasp_seen'] and self._states[key]['lift_seen'] for key in required))
        report_value = self._latest_report if self.final_state else self._frozen_report if self._report_frozen else None
        report_correct = bool(report_value is not None and report_value == _mapping(self._optical).get('appearance_mode'))
        stages = {'selection': selected, 'grasp_lift': lifted, 'appearance_report': report_correct,
                  'selected_original_at_destination': selected,
                  'reconfiguration': self._current['reconfiguration_satisfied'], 'stable_placement': self._stable_placement}
        items = [{'id': key, 'completed': bool(stages[key]), **({'kind': 'terminal_condition'} if self.final_state else {})}
                 for key in self.completion[self.operation]]
        validity = 'invalid' if self._violations else 'complete' if self._current['available'] else 'unavailable'
        physical = bool(validity == 'complete' and selected and (self.final_state or lifted) and self._stable_placement
                        and (not self.spatial or stages['reconfiguration']))
        success = bool(physical and (self.operation != 'classify_appearance_deliver' or report_correct))
        completed = sum(item['completed'] for item in items)
        score = completed / len(items) if validity == 'complete' else None
        version = FINAL_STATE_SCORING_VERSION if self.final_state else SCORING_VERSION
        revision = {'manipulation_revision': FINAL_STATE_REVISION} if self.final_state else {}
        report = ({'latest_valid': self._latest_report, 'acceptance_policy': 'latest_valid_final_report',
                   'correct': report_correct, 'history': self._reports} if self.final_state else {
                   'latest_valid_before_grasp': self._latest_report, 'frozen': self._report_frozen,
                   'frozen_value': self._frozen_report, 'frozen_at_s': self._report_frozen_at,
                   'correct': report_correct, 'history': self._reports})
        return deepcopy({'schema': version, 'scoring_version': version, **revision,
            'operation': self.operation, 'task_success': success, 'physical_success': physical,
            'evaluation_validity': validity,
            'constraints_satisfied': False if self._violations or (self.final_state
                and not self._current['unselected_objects_preserved']) else True if validity == 'complete' else None,
            'completion_items': items, 'completed_goals': completed, 'total_goals': len(items),
            'completion_score': score, 'score_percent': None if score is None else score * 100,
            'frozen_selection': self._optical, 'selected_candidate': self._selection,
            'committed_candidate': self._committed, 'frozen_anchor_centroid_m': _json_value(self._anchor),
            'initial_distance_order': self._initial_order, 'initial_distances_m': self._initial_distances,
            'report': report,
            'per_target': self._states, 'initial_measurement_errors': self._initial_errors,
            'constraint_violations': sorted(self._violations),
            'native_time_s': self._current['native_time_s'],
            'final_released': self._current['final_released'],
            'measurement_errors': self._current['measurement_errors'],
            'dwell_started_at_s': self._dwell_start, 'stable_duration_s': self._stable_duration,
            'stable_placement': self._stable_placement, 'current': self._current, 'scoring': self.scoring,
            'optical_visibility_verified': False, 'optical_necessity_verified': False,
            'complete_reference_execution_verified': False})

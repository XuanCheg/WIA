"""Native occupancy and explicit Agent arrival reports for Cognitive Mapping.

Run: python -m pytest tests/test_cognitive_mapping_arrival_reports.py
Category evaluators provide read-only native measurement and start qualification.
Only sample_step records control samples; observation/report reads never earn visits.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
import re

import numpy as np


ARRIVAL_INSTRUCTION = (
    "While physically and stably inside each required room, output a standalone line "
    "'ARRIVED: <room_instance>' in your world_observation text, replacing <room_instance> "
    "with the room instance you identify from your observations. Report each required room; "
    "a declaration alone does not establish a visit. For an ordered route, report each step "
    "after completing its predecessor, including repeated room names. Out-of-order visits "
    "do not erase completed steps; revisit and report the pending step to continue."
)


def completion_public_goal(public, subclass, *, ordered=False):
    """Pure migration for existing arrival-report goals; preserve source questions.

    Call with a current task's public mapping and subclass, then persist the
    returned mapping in a versioned task revision. Geometry and thresholds are
    deliberately outside this migration.
    """
    result = deepcopy(public)
    scopes = {
        'Connectivity': 'Scoring measures physical room reachability and arrival confirmation; direct adjacency is not tested.',
        'Traversable Passage': 'Scoring measures endpoint room visits and arrival confirmation along any legal route; no designated opening is tested.',
        'Regional Boundary': 'Scoring measures visits and arrival confirmation for the rooms defined by the queried anchors; visual discovery of the anchors and the source QA answer are not scored.',
        'Long-Term Navigation': 'Scoring measures physical room visits and arrival confirmation for the public route requirements; the source QA answer is not scored.',
    }
    scope = scopes[subclass]
    if scope not in result['success_criteria']:
        result['success_criteria'] += ' ' + scope
    scoring = ('Each accepted arrival report completes only the next required route step. '
               'Repeated room names are separate steps. Out-of-order visits preserve the completed prefix; '
               'return to and report the pending room to continue. Partial credit is the completed prefix divided by all required steps.'
               if ordered else 'Partial credit is the number of required rooms stably visited and reported divided by all required rooms. Repeated reports do not add credit.')
    if scoring not in result['success_criteria']:
        result['success_criteria'] += ' ' + scoring
    safety = 'All existing collision, floor support, uprightness and protected-object constraints remain effective throughout the episode.'
    if safety not in result['success_criteria']:
        result['success_criteria'] += ' ' + safety
    replacements = {
        'Keep room-defining furniture and the original building unchanged.':
            'Preserve the original building and task-critical room-defining anchors.',
        'Preserve the original floorplan, furniture, doorways and source reference frames.':
            'Preserve the original floorplan, doorway geometry, task-critical anchors and source reference frames.',
        'Preserve the building, obstacles, furniture and reference frames.':
            'Preserve the building, task-critical anchors and source reference frames.',
        'Keep region-defining furniture, dimensions and scene topology unchanged.':
            'Keep the queried anchors, their dimensions and poses, and scene topology unchanged.',
    }
    result['constraints'] = [replacements.get(value, value) for value in result.get('constraints', [])]
    furniture = ('Ordinary noncritical furniture that is natively movable may be moved with legal arm and gripper actions; '
                 'preserve task-critical anchors and supports, and retain all base collision guards.')
    if furniture not in result['constraints']:
        result['constraints'].append(furniture)
    return result


def native_array(value):
    """Normalize ordinary tensors and PhysX structured x/y/z contact coordinates."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.names and all(axis in array.dtype.names for axis in ('x', 'y', 'z')):
        array = np.stack([array[axis] for axis in ('x', 'y', 'z')], axis=-1)
    return np.asarray(array, dtype=float)


def public_navigation_route(question, start_region, target_region):
    """Extract only explicit, nonblank numbered room steps from public ESI text.

    Missing route blanks and turn answers are never reconstructed. A plain pair
    of endpoints, adjacency question, or shortest-path MCQ does not impose order.
    """
    prefix = re.match(r'^You are beginning in the .+?\. You want to navigate to the .+?\.', question)
    if prefix is None:
        return [], []
    steps = []
    patterns = (
        r'(?:^|\s)(\d+\. Go to the (.+?)\.)(?=\s+\d+\. | You have reached|$)',
        r'(?:^|\s)(\d+\. Choose the next region to explore\. '
        r'([a-zA-Z][a-zA-Z_ ]*\d+(?: \([^)]*\))?))(?=\s+\d+\. | You have reached|$)',
    )
    matches = sorted((match for pattern in patterns for match in re.finditer(pattern, question)),
                     key=lambda match: match.start())
    for match in matches:
        phrase, label = match.group(1), match.group(2).strip().rstrip('.')
        # Turn-choice prompts are a different instruction, not room names.
        if 'Choose the next turn direction' in label or 'please fill in' in label:
            continue
        label = label.split(' (', 1)[0]
        region = '_'.join(label.lower().split())
        if not re.fullmatch(r'[a-z][a-z0-9_]*_\d+', region):
            continue
        steps.append({'region': region, 'source_phrase': phrase})
    if not steps:
        return [], []
    evidence = [{'region': start_region, 'source_phrase': prefix.group(0)}]
    evidence.extend(steps)
    if evidence[-1]['region'] != target_region:
        evidence.append({'region': target_region, 'source_phrase': prefix.group(0)})
    # Consecutive duplicate labels describe one occupancy, not a forced exit.
    evidence = [step for index, step in enumerate(evidence)
                if index == 0 or step['region'] != evidence[index - 1]['region']]
    return [step['region'] for step in evidence], evidence


def visit_parameters(required_regions, *, required_order=None, route_evidence=None, question=None):
    """Declare the new set-based contract; ordered proof needs public provenance."""
    order = list(required_order or [])
    return {'required_regions': list(dict.fromkeys(required_regions)),
        'scoring_version': 'task_completion_v2',
        'required_order': order, 'arrival_report_required': True,
        'public_route_required': bool(order),
        'public_route_evidence': list(route_evidence or []),
        'source_public_question': question if order else None}


def settled_native_state(session, measured, *, minimum_up_dot=.95):
    """Add native chassis velocity/upright checks without altering the scene."""
    robot = session.robot
    chassis = robot.links[robot.base_footprint_link_name]
    linear, angular = (native_array(value) for value in (
        chassis.get_linear_velocity(), chassis.get_angular_velocity()))
    quaternion = native_array(measured['quaternion_xyzw'])
    if (linear.shape != (3,) or angular.shape != (3,) or not np.isfinite([linear, angular]).all()
            or quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) <= 0):
        raise ValueError('native_chassis_velocity_or_orientation_unavailable')
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    up_dot = float(1 - 2 * (x*x + y*y))
    return {**measured, 'linear_speed_m_s': float(np.linalg.norm(linear)),
        'angular_speed_rad_s': float(np.linalg.norm(angular)), 'up_dot': up_dot,
        'state_valid': bool(measured['state_valid'] and up_dot >= minimum_up_dot),
        'occupancy_valid': bool(measured.get('floor_contact', measured['state_valid'])
                                and up_dot >= minimum_up_dot)}


class NativeVisitReportEvaluator:
    """Read native samples and validate same-boundary Agent-authored claims.

    Subclasses implement _measure_native, _qualify_start, and optionally
    _additional_checks. Every return value and file here is private evaluation.
    No method creates Agent text, advances simulation, or moves any native body.
    """
    schema = 'private_cognitive_mapping_visit_reports_v2'

    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        p = self.parameters
        required = p.get('required_regions')
        answer_only = (p.get('actual_capability') == 'direct_adjacency_and_unreachability_answers'
                       and p.get('source_graph_reachable') is False)
        if not isinstance(required, list) or (not required and not answer_only) or not all(isinstance(r, str) and r for r in required):
            raise ValueError('native_required_room_set_missing')
        if p.get('arrival_report_required') is not (False if answer_only else True):
            raise ValueError('explicit_agent_arrival_reports_required')
        self.required = set(required)
        self.required_order = list(p.get('required_order', []))
        if self.required_order:
            evidence, question = p.get('public_route_evidence', []), p.get('source_public_question')
            if (p.get('public_route_required') is not True or not isinstance(question, str)
                    or not evidence or [e.get('region') for e in evidence] != self.required_order
                    or not set(self.required_order).issubset(self.required)
                    or any(not e.get('source_phrase') or e['source_phrase'] not in question for e in evidence)):
                raise ValueError('ordered_route_requires_exact_public_question_evidence')
        elif p.get('public_route_required'):
            raise ValueError('public_route_flag_without_order')
        self.stability_steps = int(p.get('stability_steps', p.get('stop_samples', 3)))
        if self.stability_steps < 1:
            raise ValueError('positive_native_stability_samples_required')
        self.maximum_speed = float(p.get('maximum_stop_speed_m_s', p.get('maximum_terminal_speed_m_s', .02)))
        self.maximum_angular_speed = float(p.get('maximum_stop_angular_speed_rad_s',
            p.get('maximum_terminal_angular_speed_rad_s', .1)))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _time(self):
        value = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(value):
            raise ValueError('native_simulation_clock_unavailable')
        return value

    def _additional_checks(self, measured):
        return {}

    def _task_faults(self):
        # Contact/pose safety history remains visible but does not erase factual
        # room visits. Measurement, continuity and task-invariant faults remain.
        return self.faults - {'native_floor_contact_or_pose_failed', 'native_nonfloor_contact'}

    def _occupancy_valid(self, measured):
        return measured.get('occupancy_valid', measured['state_valid'])

    def _settled(self, measured):
        return (measured['linear_speed_m_s'] <= self.maximum_speed
                and measured['angular_speed_rad_s'] <= self.maximum_angular_speed)

    def _append(self, filename, row):
        with (self.output / filename).open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def begin_episode(self, observation):
        self.start_checks = self._qualify_start()
        if not self.start_checks or not all(self.start_checks.values()):
            raise ValueError('native_visit_episode_start_rejected')
        self.episode += 1
        self.active = True
        self.visited, self.reported, self.visits, self.arrival_reports = set(), set(), [], []
        self.order_stage, self.samples, self.consecutive = 0, 0, 0
        self.completed_events, self.measurement_errors = {}, []
        self.stable_region, self.sample_region = None, None
        self.faults, self.travel_m = set(), 0.
        measured = self._measure_native()
        self.previous = np.asarray(measured['position'], dtype=float)
        self.previous_time = self._time()
        return self.record(observation, action=None)

    def _read_boundary(self):
        measured = self._measure_native()
        if measured.get('nonfloor_contacts'):
            self.faults.add('native_nonfloor_contact')
        if not self._occupancy_valid(measured):
            self.faults.add('native_floor_contact_or_pose_failed')
        displacement = float(np.linalg.norm(np.asarray(measured['position']) - self.previous))
        if displacement > self.controller.config.translation_tolerance_m:
            self.faults.add('unsampled_native_motion')
        checks = self._additional_checks(measured)
        if not all(checks.values()):
            self.faults.add('native_category_invariant_failed')
        if (not self._occupancy_valid(measured) or not self._settled(measured)
                or measured['chassis_region'] != self.sample_region):
            self.consecutive, self.stable_region = 0, None
        return measured, checks

    def sample_step(self):
        if not self.active:
            return
        try:
            measured, now = self._measure_native(), self._time()
            extra_checks = self._additional_checks(measured)
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
            self.faults.add('native_control_measurement_unavailable')
            self.measurement_errors.append({'stage': 'control_sample', 'reason': f'{type(error).__name__}: {error}'})
            self.consecutive, self.stable_region = 0, None
            return
        point = np.asarray(measured['position'], dtype=float)
        elapsed = now - self.previous_time
        distance = float(np.linalg.norm(point - self.previous))
        cfg = self.controller.config
        limit = cfg.max_translation_speed_m_s * max(elapsed, 0.) + cfg.translation_tolerance_m
        if 'maximum_step_displacement_m' in self.parameters:
            limit = min(limit, self.parameters['maximum_step_displacement_m'])
        if elapsed <= 0 or distance > limit:
            self.faults.add('motion_clock_or_continuity_failed')
        if measured.get('nonfloor_contacts'):
            self.faults.add('native_nonfloor_contact')
        if not self._occupancy_valid(measured):
            self.faults.add('native_floor_contact_or_pose_failed')
        if not all(extra_checks.values()):
            self.faults.add('native_category_invariant_failed')
        self.samples += 1
        self.travel_m += distance
        region = measured['chassis_region']
        eligible = (self._occupancy_valid(measured) and self._settled(measured)
                    and region is not None and not self._task_faults())
        if eligible:
            self.consecutive = self.consecutive + 1 if region == self.sample_region else 1
        else:
            self.consecutive = 0
        self.sample_region = region
        if self.consecutive >= self.stability_steps:
            if self.stable_region != region:
                self.visited.add(region)
                self.visits.append({'region': region, 'sample': self.samples,
                    'position': measured['position'], 'simulation_time_s': now})
            self.stable_region = region
        else:
            self.stable_region = None
        self.previous, self.previous_time = point, now
        self._append('physical_trajectory.jsonl', {'event': 'native_control_sample',
            'sample': self.samples, 'simulation_time_s': now, 'native_state': measured,
            'visited_regions': sorted(self.visited), 'order_stage': self.order_stage,
            'motion_faults': sorted(self.faults)})

    def report_arrival(self, room_name: str, observation):
        """Privately accept a claim only at its current native stable occupancy."""
        if not self.active:
            return {'accepted': False, 'reason': 'episode_not_active', 'claimed_room': room_name}
        measured, measurement_error = None, None
        try:
            measured, _ = self._read_boundary()
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
            self.faults.add('native_arrival_measurement_unavailable')
            measurement_error = f'{type(error).__name__}: {error}'
            self.measurement_errors.append({'stage': 'arrival_report', 'reason': measurement_error})
        if measurement_error is not None:
            reason = 'native_arrival_measurement_unavailable'
        elif not isinstance(room_name, str) or not room_name:
            reason = 'invalid_agent_room_claim'
        elif self._task_faults() or not self._occupancy_valid(measured):
            reason = 'native_trajectory_or_current_state_invalid'
        elif measured['chassis_region'] != room_name:
            reason = 'claim_does_not_match_current_native_room'
        elif room_name not in self.visited:
            reason = 'room_not_natively_visited'
        elif self.stable_region != room_name or self.consecutive < self.stability_steps or not self._settled(measured):
            reason = 'current_native_occupancy_not_stable'
        else:
            reason = None
            self.reported.add(room_name)
        row = {'event': 'agent_arrival_report', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'claimed_room': room_name, 'accepted': reason is None, 'reason': reason,
            'native_control_sample': self.samples}
        row['order_advanced'] = False
        if reason is None:
            if self.required_order:
                if (self.order_stage < len(self.required_order)
                        and room_name == self.required_order[self.order_stage]):
                    item_id = f'step:{self.order_stage}:{room_name}'
                    self.order_stage += 1
                    row['order_advanced'] = True
                    row['completion_item_id'] = item_id
                    self.completed_events[item_id] = dict(row)
            elif room_name in self.required:
                item_id = f'room:{room_name}'
                row['completion_item_id'] = item_id
                self.completed_events.setdefault(item_id, dict(row))
        if measurement_error is not None:
            row['exception_summary'] = measurement_error
        self.arrival_reports.append(row)
        self._append('trajectory.jsonl', {**row, 'native_state': measured})
        return row

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        try:
            measured, additional = self._read_boundary()
        except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError) as error:
            self.faults.add('native_record_measurement_unavailable')
            self.measurement_errors.append({'stage': 'record', 'reason': f'{type(error).__name__}: {error}'})
            measured, additional = None, {}
        checks = {'qualified_native_start': all(self.start_checks.values()),
            'required_native_room_set_visited': self.required.issubset(self.visited),
            'required_agent_room_set_reported': self.required.issubset(self.reported),
            'source_requested_order_proven': self.order_stage == len(self.required_order),
            'continuous_safe_native_trajectory': not self.faults,
            'task_trajectory_evidence_valid': not self._task_faults(),
            'current_native_state_valid': bool(measured is not None and measured['state_valid']), **additional}
        definitions = ([(f'step:{index}:{room}', room) for index, room in enumerate(self.required_order)]
                       if self.required_order else [(f'room:{room}', room) for room in sorted(self.required)])
        items = [{'id': item_id, 'room': room, 'completed': item_id in self.completed_events,
                  'evaluation_validity': 'valid' if not self.measurement_errors or item_id in self.completed_events else 'unavailable',
                  'kind': 'historical_checkpoint',
                  'evidence': self.completed_events.get(item_id)} for item_id, room in definitions]
        completed = sum(item['completed'] for item in items)
        validity = 'unavailable' if self.measurement_errors else 'complete'
        score = completed / len(items) if items and validity == 'complete' else None
        constraints_satisfied = bool(not self.faults and measured is not None
                                     and measured['state_valid'] and all(additional.values()))
        row = {'schema': self.schema, 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'operation': self.config['operation'],
            'task_success': bool(all(value for key, value in checks.items()
                if key not in ('continuous_safe_native_trajectory', 'current_native_state_valid'))),
            'checks': checks, 'native_state': measured,
            'scoring_version': 'task_completion_v2', 'evaluation_validity': validity,
            'completion_items': items, 'completed_goals': completed, 'total_goals': len(items),
            'completion_score': score, 'score_percent': None if score is None else 100. * score,
            'diagnostic_completion_lower_bound': completed / len(items) if items else None,
            'constraints_satisfied': constraints_satisfied,
            'continuous_constraints_satisfied': constraints_satisfied,
            'task_constraints_satisfied': bool(not self._task_faults() and all(additional.values())),
            'safety_affects_task_success': False,
            'measurement_errors': list(self.measurement_errors),
            'required_regions': sorted(self.required), 'visited_regions': sorted(self.visited),
            'reported_regions': sorted(self.reported), 'required_order': self.required_order,
            'order_stage': self.order_stage, 'measured_visits': self.visits,
            'arrival_reports': self.arrival_reports, 'native_control_samples': self.samples,
            'sampled_travel_m': self.travel_m, 'motion_faults': sorted(self.faults),
            'proof_requirements': {'stable_full_chassis_control_samples': self.stability_steps,
                'explicit_agent_arrival_reports': self.parameters['arrival_report_required'],
                'required_room_set': bool(self.required),
                'ordered_route': bool(self.required_order), 'terminal_room_required': False,
                'minimum_travel_required': False, 'source_route_evidence': self.parameters.get('public_route_evidence', [])},
            'initial_qualification': self.start_checks, 'configuration': self.config}
        (self.output / 'state.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        self._append('trajectory.jsonl', row)
        return row


TASK_REVISION = 'task_edit_20261018_v1'
REVISED_MAPPING_OPERATIONS = {
    'Connectivity': 'answer_direct_adjacency_with_exploration',
    'Long-Term Navigation': 'execute_supplied_ordered_route',
    'Regional Boundary': 'answer_observed_anchor_region_relation',
    'Traversable Passage': 'answer_shortest_route_via_region',
}


def source_door_policy(source):
    """Record generator/runtime mismatch without treating static maps as proof."""
    generator = {'Connectivity': 'connect', 'Long-Term Navigation': 'plan',
                 'Regional Boundary': 'region', 'Traversable Passage': 'connect'}[source.small_task]
    return {'version': 'mapping_source_runtime_doors_v1',
        'generator': f'ESI-Bench/src/dataset_generation/task_cognitive_map/batch_cognitivemap_{generator}.py',
        'source_policy': 'exclude_door_and_sliding_door_then_remove_door_named_objects',
        'source_trav_map_basename': source.payload.get('trav_map_basename'),
        'runtime_policy': 'retain_native_doors', 'reconciled': None,
        'physical_opening_and_crossing_verified': None}


def revise_mapping_result(source, context, result):
    """Revise generated goals only; retain the complete historical source QA."""
    if result.get('task_definition') is None:
        return result
    raw, payload = source.raw, source.payload
    q = payload['question_data']
    task = result['task_definition']
    evaluation = task['private']['evaluation']
    p = evaluation['parameters']
    old_operation = evaluation['operation']
    kind = source.small_task
    options = raw.get('options_json', q.get('options', []))
    options = json.loads(options) if isinstance(options, str) else options
    if q.get('question') != raw.get('question') or q.get('answer') != raw.get('answer'):
        raise ValueError('Mapping source QA annotations disagree.')
    p.update(task_revision=TASK_REVISION, source_path=source.relative_path,
        mapping_composition_contract=deepcopy(context.config.get('mapping_composition_contracts', {}).get(source.relative_path)),
        legacy_operation=old_operation, source_qa={'question': raw['question'], 'answer': raw['answer'],
            'options': deepcopy(options), 'scored': kind != 'Long-Term Navigation'},
        source_door_policy=source_door_policy(source),
        mechanism_contract={'invariants': ['original_building_and_region_boundaries',
            'original_task_anchors_and_reference', 'source_order_and_intentional_revisits'],
            'required_native_evidence': ['source_runtime_door_policy_reconciled', 'task_relevant_door_qualification'],
            'unavailable_is_not_agent_failure': True})
    evaluation['operation'] = REVISED_MAPPING_OPERATIONS[kind]
    task['public'] = {'objective': raw['question'],
        'success_criteria': 'Report a standalone ANSWER: <option> line in world_observation, choosing from: '
            + ', '.join(str(x) for x in options) + '.',
        'environment': 'Explore the preserved original building using the robot-mounted observations.',
        'constraints': ['Preserve original room boundaries, queried anchors and doorway geometry.',
            'Use legal robot controls; door opening requires actual physical interaction.']}
    public = task['public']
    if kind == 'Connectivity':
        p['actual_capability'] = 'direct_adjacency_and_region_exploration'
        p['relation_semantics'] = 'direct_connection_not_global_reachability'
        p['source_graph_reachable'] = q.get('verification', {}).get('graph_reachable')
        public['objective'] += ' Explore the two named regions and report each arrival. ' + ARRIVAL_INSTRUCTION
        public['success_criteria'] += ' Answer direct adjacency: a No answer does not mean the rooms are globally unreachable. Each distinct required room is counted once; no return tour is required.'
        if p['source_graph_reachable'] is False:
            p.update(actual_capability='direct_adjacency_and_unreachability_answers', required_regions=[],
                required_order=[], arrival_report_required=False, public_route_required=False,
                reachability_answer={'version': 'mapping_reachability_answer_v1', 'answer': 'unreachable',
                    'options': ['reachable', 'unreachable'], 'source': 'question_data.verification.graph_reachable',
                    'public_format': 'ANSWER: adjacency=<Yes|No>; reachability=<reachable|unreachable>'})
            p['mechanism_contract']['required_native_evidence'] += ['source_unreachability_verified',
                'native_graph_reachable_is_false', 'operable_doors_considered', 'reachability_evidence']
            public['objective'] = (raw['question'] + ' Separately determine whether any legal route connects '
                'the two named regions, accounting for doors that can be physically opened.')
            public['success_criteria'] = ('Submit one standalone line in world_observation: '
                'ANSWER: adjacency=<Yes|No>; reachability=<reachable|unreachable>. '
                'The two answers are scored independently. Direct adjacency and global reachability are distinct. '
                'No arrival or route execution is required. Failure to find a route or an initially closed door '
                'does not establish unreachability.')
    elif kind == 'Long-Term Navigation':
        route = list(q.get('region_sequence', []))
        if (not route or route[0] != p['start_region'] or route[-1] != p['target_region']
                or any(not isinstance(region, str) or not region for region in route)):
            raise ValueError('Complete source region sequence is required for supplied-order execution.')
        route_text = 'Follow this supplied room route in order: ' + ' -> '.join(route) + '.'
        p.update(visit_parameters(route, required_order=route,
            route_evidence=[{'region': region, 'source_phrase': route_text} for region in route], question=route_text))
        p['actual_capability'] = 'supplied_order_route_execution'
        p['source_qa']['scored'] = False
        public['objective'] = route_text + ' ' + ARRIVAL_INSTRUCTION
        public['success_criteria'] = ('Complete every route step in order, including intentional returns to earlier rooms. '
            'The first room and heading remain the original start reference. Completed prefixes survive later detours; '
            'a repeated declaration at an unchanged occupancy cannot replace a new required arrival.')
    elif kind == 'Regional Boundary':
        p['actual_capability'] = 'observed_anchor_region_relation'
        p['required_anchor_bindings'] = list(evaluation['binding_keys'])
        p['public_anchor_labels'] = {key: value.get('category', '').replace('_', ' ')
                                   for key, value in task['private']['bindings'].items()}
        public['objective'] += ' Inspect each queried anchor using its public description and relate it to the named regions.'
        public['success_criteria'] += ' Each queried anchor must be identified in delivered robot-camera observations. Room occupancy alone does not establish the anchor relation; no extra room tour is required.'
        p['mechanism_contract']['required_native_evidence'] += ['public_anchor_identity_and_delivered_rgb']
    elif kind == 'Traversable Passage':
        p['actual_capability'] = 'shortest_route_via_region'
        p['route_answer_order'] = [q['source_region'], q['correct_region'], q['target_region']]
        p.update(visit_parameters(p['route_answer_order']))
        p['queried_portal'] = None
        p['mechanism_contract']['required_native_evidence'] += ['native_route_graph_and_metric_agreement']
        public['objective'] += (' Choose the intermediate region, then execute a route from the named start through '
            'that region to the destination, reporting each arrival. ' + ARRIVAL_INSTRUCTION)
        public['success_criteria'] += ' Execute the start, selected intermediate region and destination in that order. This source asks about an intermediate region on a shortest route; it does not identify a particular opening.'
    result.update(operation=evaluation['operation'], rule_revision=TASK_REVISION,
        operation_recipe=evaluation['operation'], available_operation_recipes=[evaluation['operation'], old_operation],
        actual_capability=p['actual_capability'], source_mechanism_provenance=deepcopy(p['source_door_policy']),
        private_source_qa=deepcopy(p['source_qa']),
        change_summary=['Preserve source geometry and references while restoring the declared mapping capability.',
            'Separate semantic answers, physical progress, constraints and native evaluation validity.'],
        assumptions=[], limitations=['Source/runtime door-policy reconciliation and task-relevant physical door trials remain required.',
            'Unknown source-answer/native-graph agreement produces unavailable comparable results.'],
        next_step='Reconcile source/native relations and qualify the required native doors, public anchors and route.')
    result['warning_codes'] = sorted(set(result.get('warning_codes', []) + [
        'source_runtime_door_policy_difference', 'native_mapping_relation_and_door_qualification_pending']))
    if p.get('reachability_answer'):
        result['available_operation_recipes'] = [evaluation['operation']]
    return result


def legacy_mapping_config(config):
    legacy = deepcopy(config)
    legacy['operation'] = config['parameters']['legacy_operation']
    return legacy


def mapping_qualification(session, config):
    """Select trusted release definitions or reviewed episode-bound prerequisites."""
    p = config['parameters']
    task = getattr(session, 'loaded_task', None)
    if (getattr(task, 'release_v1', False)
            and task.definition.source.json_path == p['source_path']):
        # The loader already trusts fixed-v1 definitions. This is not evidence
        # of physical qualification or of actions completed in this episode.
        return {'accepted': True, 'checks': {'fixed_v1_definition': True},
            'policy': 'fixed_v1_definition', 'physical_checks_run': False,
            'receipt': {}, 'doors': {}, 'evaluation_validity': 'complete'}
    episode_id = getattr(session, 'episode_id', None)
    receipt = deepcopy(getattr(session, 'mapping_qualification', {}).get(p['source_path'], {}))
    checks = {'current_episode_bound': episode_id is not None and receipt.get('episode_id') == episode_id,
        'source_runtime_door_policy_reconciled': receipt.get('source_runtime_door_policy_reconciled'),
        'condition_identity_verified': receipt.get('condition_identity_verified')}
    if p['source_qa']['scored']:
        checks['source_answer_verified'] = receipt.get('source_answer_verified')
    if p['actual_capability'] == 'shortest_route_via_region':
        checks['route_graph_metric_verified'] = receipt.get('route_graph_metric_verified')
    if p.get('reachability_answer'):
        checks['source_unreachability_verified'] = receipt.get('source_unreachability_verified')
        checks['native_graph_reachable_is_false'] = receipt.get('native_graph_reachable') is False
        checks['operable_doors_considered'] = receipt.get('operable_doors_considered')
        evidence = receipt.get('reachability_evidence') or {}
        checks['reachability_evidence'] = (evidence.get('method') in (
            'native_graph_exhaustive_connectivity', 'native_configuration_space_certificate')
            and bool(evidence.get('provenance')))
    door_ids = receipt.get('relevant_door_ids')
    checks['task_relevant_door_inventory_available'] = isinstance(door_ids, list)
    doors = {}
    for door_id in door_ids or []:
        door = deepcopy(getattr(session, 'door_qualification', {}).get(door_id, {}))
        doors[door_id] = door
        state = door.get('initial_state')
        checks['door:' + door_id] = bool(state in ('absent', 'open', 'closed')
            and door.get('condition_matches_current_episode') is True
            and door.get('doorway_crossing_verified') is True
            and door.get('reset_restoration_verified') is True
            and door.get('public_controls_only') is True
            and (state != 'closed' or door.get('door_opening_verified') is True))
    return {'accepted': all(value is True for value in checks.values()), 'checks': checks,
        'receipt': receipt, 'doors': doors, 'evaluation_validity': 'complete'
        if all(value is True for value in checks.values()) else 'unavailable'}


def queried_portal_crossing(session, portal):
    """Validate current-episode evidence for an explicitly identified opening."""
    events = getattr(session, 'portal_crossings', None)
    if not isinstance(events, list):
        return {'completed': False, 'evaluation_validity': 'unavailable', 'evidence': None}
    episode_id = getattr(session, 'episode_id', None)
    for event in events:
        if (episode_id is None or event.get('episode_id') != episode_id
                or event.get('portal_id') != portal['portal_id']
                or (event.get('from_region'), event.get('to_region')) !=
                   (portal['from_region'], portal['to_region'])
                or event.get('through_opening_verified') is not True
                or event.get('public_controls_only') is not True):
            continue
        before, after = (native_array(event.get(key)) for key in ('before_position_m', 'after_position_m'))
        time = event.get('simulation_time_s')
        if (before.shape == (3,) and after.shape == (3,) and np.isfinite([before, after]).all()
                and isinstance(time, (int, float)) and math.isfinite(time)
                and np.linalg.norm(after - before) > 0):
            return {'completed': True, 'evaluation_validity': 'valid', 'evidence': deepcopy(event)}
    return {'completed': False, 'evaluation_validity': 'valid', 'evidence': None}


def qualify_revised_mapping(session, executor, config, bound, legacy_qualifier):
    result = legacy_qualifier(session, executor, legacy_mapping_config(config), bound)
    proof = mapping_qualification(session, config)
    result['mapping_qualification'] = proof
    result['accepted'] = bool(result.get('accepted') and proof['accepted'])
    result['evaluation_validity'] = 'complete' if result['accepted'] else 'unavailable'
    if not proof['accepted']:
        result['reason'] = 'source_runtime_mapping_or_door_qualification_unavailable'
    return result


class SourceAnswerMappingEvaluator:
    """Combine existing native visits with semantic outputs and episode evidence."""

    def __init__(self, session, controller, config, output, native):
        self.session, self.controller, self.config = session, controller, config
        self.parameters, self.native, self.output = config['parameters'], native, Path(output)

    def begin_episode(self, observation):
        self.answer, self.route_stage = None, 0
        self.reachability_response = None
        self.anchor_evidence, self.route_events = {}, []
        self.native.begin_episode(observation)
        return self.record(observation)

    def sample_step(self):
        self.native.sample_step()

    def report_answer(self, answer, observation):
        if self.parameters.get('reachability_answer'):
            text = str(answer).strip()
            text = re.sub(r'^ANSWER:\s*', '', text, flags=re.IGNORECASE)
            self.answer, self.reachability_response = None, None
            for field, value in re.findall(r'(?:^|;)\s*(adjacency|reachability)\s*=\s*([^;]+)', text, re.IGNORECASE):
                if field.casefold() == 'adjacency':
                    self.answer = value.strip()
                else:
                    self.reachability_response = value.strip()
        else:
            self.answer = str(answer)

    def report_arrival(self, room_name, observation):
        row = self.native.report_arrival(room_name, observation)
        route = self.parameters.get('route_answer_order', [])
        if row.get('accepted') and self.route_stage < len(route) and room_name == route[self.route_stage]:
            self.route_events.append(deepcopy(row))
            self.route_stage += 1
        return row

    def _observed_anchors(self):
        entries = getattr(self.session, 'mapping_anchor_observations', {}).get(self.parameters['source_path'])
        if not isinstance(entries, list):
            return False
        for row in entries:
            key = row.get('binding')
            if (key in self.parameters.get('required_anchor_bindings', [])
                    and row.get('episode_id') == getattr(self.session, 'episode_id', None)
                    and row.get('native_object_name') == self.config['bindings'][key]
                    and row.get('observable_identity_verified') is True
                    and row.get('public_label_association_verified') is True
                    and row.get('delivered_rgb_verified') is True and row.get('snapshot_id')):
                self.anchor_evidence.setdefault(key, deepcopy(row))
        return True

    def record(self, observation, *, action=None):
        row = self.native.record(observation, action=action)
        p = self.parameters
        proof = mapping_qualification(self.session, self.config)
        valid = proof['accepted'] and row['evaluation_validity'] == 'complete'
        correct = self.answer is not None and ' '.join(self.answer.casefold().split()) == ' '.join(str(p['source_qa']['answer']).casefold().split())
        if p['actual_capability'] == 'observed_anchor_region_relation':
            valid = self._observed_anchors() and valid
            items = [{'id': 'anchor:' + key, 'kind': 'historical_checkpoint',
                'completed': key in self.anchor_evidence, 'evaluation_validity': 'valid',
                'evidence': self.anchor_evidence.get(key)} for key in p['required_anchor_bindings']]
        elif p.get('route_answer_order'):
            items = [{'id': f'route_step:{index}:{room}', 'kind': 'historical_checkpoint',
                'completed': index < self.route_stage, 'evaluation_validity': 'valid'}
                for index, room in enumerate(p['route_answer_order'])]
        else:
            items = deepcopy(row['completion_items'])
        if p['source_qa']['scored']:
            items.append({'id': 'source_relation_answer', 'kind': 'answer_requirement',
                'completed': bool(correct), 'evaluation_validity': 'valid'})
        if p.get('reachability_answer'):
            reachable_correct = (self.reachability_response is not None
                and self.reachability_response.casefold() == p['reachability_answer']['answer'])
            items.append({'id': 'global_reachability_answer', 'kind': 'answer_requirement',
                'completed': reachable_correct, 'evaluation_validity': 'valid'})
        if p.get('queried_portal'):
            crossing = queried_portal_crossing(self.session, p['queried_portal'])
            items.append({'id': 'queried_portal_crossing', 'kind': 'historical_checkpoint', **crossing})
            valid = valid and crossing['evaluation_validity'] == 'valid'
        if not valid:
            for item in items:
                item['evaluation_validity'] = 'unavailable'
        if p.get('reachability_answer') and proof.get('policy') != 'fixed_v1_definition':
            semantic_checks = {'source_answer_verified', 'source_unreachability_verified',
                'native_graph_reachable_is_false', 'operable_doors_considered', 'reachability_evidence'}
            common_valid = (row['evaluation_validity'] == 'complete' and all(value is True
                for key, value in proof['checks'].items() if key not in semantic_checks))
            for item in items:
                keys = ({'source_answer_verified'} if item['id'] == 'source_relation_answer'
                        else semantic_checks - {'source_answer_verified'})
                item['evaluation_validity'] = 'valid' if common_valid and all(
                    proof['checks'].get(key) is True for key in keys) else 'unavailable'
        if p.get('reachability_answer'):
            row['reachability_response'] = self.reachability_response
            row['reachability_answer_correct'] = reachable_correct
        row.update(schema='private_revised_mapping_v1', operation=self.config['operation'],
            native_visit_task_success=row['task_success'], configuration=self.config,
            completion_items=items, evaluation_validity='complete' if valid else 'unavailable',
            answer=self.answer, answer_correct=bool(correct), mapping_qualification=proof,
            observed_anchor_evidence=deepcopy(self.anchor_evidence), route_answer_stage=self.route_stage,
            task_success=bool(valid and row['task_constraints_satisfied'] and all(item['completed'] for item in items)))
        from wia.tasks.completion import progress
        row.update(progress(row))
        (self.output / 'state.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        return row

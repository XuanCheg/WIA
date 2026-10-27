"""Convert passage MCQs to real room-to-room navigation with private native checks.

Run CategoryConverter().convert_all(sources, ConversionContext(batch_root)) using
wia.authoring.conversion on CPU. Native hooks are invoked by the existing
task runtime; they only read state and never issue robot or simulation actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

import json
import math
from pathlib import Path
import re

import numpy as np

from wia.tasks.background_policy import legal_background_contact

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.tasks.interaction_setup import verified_repaired_start
from wia.authoring.converters.cognitive_mapping_visit_reports import (
    ARRIVAL_INSTRUCTION, completion_public_goal, NativeVisitReportEvaluator, native_array, visit_parameters)


from wia.authoring.converters.cognitive_mapping_visit_reports import (
    REVISED_MAPPING_OPERATIONS, revise_mapping_result, legacy_mapping_config,
    qualify_revised_mapping, SourceAnswerMappingEvaluator)

MODULE = 'core.esi_converters.cognitive_mapping_traversable_passage'
REVISED_OPERATION = REVISED_MAPPING_OPERATIONS['Traversable Passage']
EXPLORATION_OPERATION = 'passage_exploration_arrivals'
LEGACY_OPERATIONS = ('passage_arrival', 'passage_round_trip')
OPERATIONS = (EXPLORATION_OPERATION, *LEGACY_OPERATIONS)
QUESTION = re.compile(r'^On the shortest path from (?P<start>.+?) to (?P<target>.+?), '
                      r'which intermediate room do you pass through among these options: .+\?$')
# The installed SegmentationMap loads floor_insseg_0 and defaults to floor Z=0.
# Environment.default_config.initial_pos_z_offset=.1 is drop clearance only.
INITIAL_CLEARANCE_M = 0.1


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value))


def _region_id(label):
    return '_'.join(label.split(' (', 1)[0].strip().lower().split())


def _source_data(source):
    raw, payload = source.raw, source.payload
    category = CategoryConverter.category
    if (source.big_task, source.small_task) != category or (raw.get('big_task'), raw.get('small_task')) != category:
        raise ValueError('source_category_mismatch')
    q = payload.get('question_data', {})
    match = QUESTION.fullmatch(raw.get('question', ''))
    if raw.get('room') != 'full_scene' or not raw.get('scene'):
        raise ValueError('original_full_scene_required')
    if match is None or q.get('case_type') != 'shortest_path_via_region':
        raise ValueError('unsupported_original_passage_question')
    names = [_region_id(match.group(key)) for key in ('start', 'target')]
    if names != [q.get('source_region'), q.get('target_region')] or names[0] == names[1]:
        raise ValueError('public_private_passage_endpoints_mismatch')
    camera = q.get('render', {}).get('initial_view', q.get('initial_view', {}))
    pose = camera.get('camera_pose', {})
    if (not _vector(pose.get('position'), 3) or not _vector(pose.get('quaternion_xyzw'), 4)
            or not any(pose['quaternion_xyzw']) or not _vector(pose.get('target'), 3)):
        raise ValueError('missing_original_initial_camera_reference')
    if camera.get('room_instance') != names[0]:
        raise ValueError('source_initial_view_region_mismatch')
    dx, dy = (pose['target'][i] - pose['position'][i] for i in (0, 1))
    if math.hypot(dx, dy) <= 1e-6:
        raise ValueError('source_initial_heading_undefined')
    return raw, q, match, pose, math.atan2(dy, dx)


def _operation(source, context):
    operation = context.config.get('traversable_passage_operation')
    if operation is not None:
        if operation != EXPLORATION_OPERATION:
            raise ValueError('New passage conversion requires unordered exploration; legacy recipes are read-only.')
        return operation
    return EXPLORATION_OPERATION


class CategoryConverter(ConverterBase):
    category = ('Cognitive Mapping', 'Traversable Passage')
    version = 'cognitive_mapping_traversable_passage_task_edit_20261018'

    def inspect(self, source):
        try:
            _source_data(source)
        except (ValueError, KeyError, TypeError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_endpoint_review_required']}
        return {'eligible': True, 'reason': 'Public source and destination support a real navigation operation.'}

    def convert_all(self, sources, context):
        # The shared batcher preserves failures, private QA, immutable revisions
        # and existing records. Stable ordering also makes manifests reproducible.
        return super().convert_all(sorted(sources, key=lambda source: source.relative_path), context)

    def convert(self, source, context):
        result = self._convert_legacy(source, context)
        if context.config.get('mapping_revision') == 'legacy' or context.config.get('traversable_passage_operation'):
            return result
        return revise_mapping_result(source, context, result)

    def _convert_legacy(self, source, context):
        raw, q, match, pose, yaw = _source_data(source)
        operation = _operation(source, context)
        start, target = match.group('start'), match.group('target')
        objective = (f'Explore the original passages and visit both {start} and {target}, in any order. '
            + ARRIVAL_INSTRUCTION)
        public = {'objective': objective,
            'success_criteria': 'Stably occupy and explicitly report each named room with the chassis '
                'fully inside, supported by the floor, upright and clear of furniture or walls. '
                'Extra rooms, alternate routes and backtracking are allowed; no return or terminal room is required.',
            'environment': 'The complete original building, room-defining furniture and doorways are retained.',
            'constraints': ['Use the existing guarded base controls and robot mounted observations.',
                'Preserve the building, obstacles, furniture and reference frames.',
                'Identify arrival room names from mounted observations; no room map is supplied.',
                'An already occupied required room can be reported after stable native samples.']}
        parameters = {'start_region': q['source_region'], 'target_region': q['target_region'],
            'start_xy': [float(x) for x in pose['position'][:2]], 'initial_yaw_rad': yaw,
            'floor_idx': 0, 'floor_z_m': 0.0, 'start_position_tolerance_m': 0.05,
            'start_heading_tolerance_deg': 2.0, 'floor_contact_tolerance_m': 0.005,
            'minimum_up_dot': 0.95, 'maximum_terminal_speed_m_s': 0.02,
            'maximum_terminal_angular_speed_rad_s': 0.1, 'stability_steps': 3}
        parameters.update(visit_parameters([q['source_region'], q['target_region']]))
        patches = [{'operation': 'robot_pose', 'phase': 'before_construction',
            'position': [*parameters['start_xy'], INITIAL_CLEARANCE_M],
            'yaw_deg': math.degrees(yaw), 'head_pitch_deg': 0.0}]
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': [], 'parameters': parameters}
        task = physical_task(source, context, public=completion_public_goal(public, self.category[1]), evaluation=evaluation, bindings={}, patches=patches,
            capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        task['defaults']['max_steps'] = int(context.config.get('traversable_passage_max_steps', 600))
        warnings = ['assumption_review_required', 'native_route_unverified', 'robot_start_unqualified',
                    'native_floor_layer_unverified', 'navigation_budget_unqualified',
                    'same_source_region_spawn_repair_requires_native_evidence']
        legacy_differs = q.get('via_region') != q.get('correct_region')
        if legacy_differs:
            warnings.append('legacy_via_region_differs_from_mcq_answer')
        options = raw.get('options_json', [])
        options = json.loads(options) if isinstance(options, str) else options
        qa_conflict = (q.get('question') != raw['question'] or q.get('answer') != raw.get('answer')
            or q.get('options') != options or _region_id(raw.get('answer', '')) != q.get('correct_region'))
        if qa_conflict:
            warnings.append('source_qa_annotation_conflict')
        return {'scene_definition': original_scene(source, context, patches=patches), 'task_definition': task,
            'interaction_policy': {'mode': 'navigation', 'group_bindings': [], 'manipulation_bindings': [],
                'relocation_policy': 'preserve', 'source_region': q['source_region'],
                'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                'maximum_stance_distance_m': .85},
            'status': 'generated', 'reuse_level': 'L2', 'operation_recipe': operation,
            'available_operation_recipes': [EXPLORATION_OPERATION],
            'change_summary': ['Keep all source assets, dimensions, poses, answers and original observer references.',
                'Initialize only the existing R1 at the original initial-view XY with source-derived heading.',
                'Add unordered native room visits with explicit same-room Agent arrival reports.'],
            'warning_codes': warnings,
            'assumptions': [{
                'doubt': 'A free-camera initial-view point does not establish a collision-free R1 stance or a traversable route.',
                'chosen_assumption': 'Use its exact XY and horizontal look direction as a reversible robot-only start. '
                    'The installed floor-0 convention plus 0.1 m drop clearance proposes the initial root Z; '
                    'actual native floor contact, room occupancy, heading and chassis state must qualify it.',
                'alternatives': ['Retain the empty original variant for diagnostic comparison.',
                    'Use a separately measured stance in the same source room if native qualification rejects this point.',
                    'Keep an impassable original route as a failed Review item without widening it.'],
                'scope': source.relative_path + '; only initial R1 pose and zero additional head pitch.',
                'risk': 'The robot may collide, lack floor support, occupy the wrong floor or see different views '
                    'from the source free camera. No native start or passage feasibility is claimed.'}, {
                'doubt': 'The MCQ shortest path was computed for source traversal, not for the actual R1 footprint.',
                'chosen_assumption': 'Require the set of two publicly named endpoint rooms with explicit Agent '
                    'arrival reports, allowing safe exploration and backtracking. No shortest path, hidden '
                    'intermediate answer, order or return is imposed. Retain original shortest-path QA privately.',
                'alternatives': ['Add robot-specific shortest-path efficiency only after native clearance qualification.',
                    'Retain a diagnostic item if native passages are inaccessible.'],
                'scope': 'The exploration recipe; 600 public decisions unless explicitly configured otherwise.',
                'risk': 'Arrival does not prove shortest-path optimality or the original intermediate-room answer. '
                    'The new decision budget is uncalibrated and may be insufficient for long routes.'}],
            'private_source_evidence': {'legacy_via_region': q.get('via_region'), 'mcq_correct_region': q.get('correct_region'),
                'legacy_differs_from_mcq': legacy_differs, 'qa_annotation_conflict': qa_conflict,
                'source_shortest_path_regions': q.get('shortest_path_regions'),
                'source_path_length_m': q.get('path_views', {}).get('path_length_m'),
                'source_verification_method': q.get('verification', {}).get('method'),
                'initial_pose_provenance': '/question_data/render/initial_view/camera_pose',
                'native_execution_verified': False},
            'limitations': ['CPU conversion and source traversal flags do not qualify robot initialization or navigation.',
                'The evaluator measures floor-0 room raster membership and a nine-point chassis AABB footprint; '
                    'this is conservative room occupancy, not a geometric proof of every swept arm surface.',
                'Native contacts and continuity are sampled after guarded controller steps, not every physics substep.',
                'Missing floor support or prohibited external contact invalidates the measured episode; legal noncritical furniture arm contacts remain permitted.',
                'Source paths, answers, maps, construction data and Human review images remain private.',
                'Runtime cost, route success rate, mounted sensor coverage and sufficient navigation budgets remain unmeasured.'],
            'attempted_repairs': ['Declarative robot-only start proposal; no native execution or asset changes.'],
            'next_step': 'Inspect original/adapted images, qualify the native start and floor, then execute '
                'guarded passage navigation; retain all capture or qualification failures in Review.'}


def _array(value):
    return native_array(value)


def _parameters(session, config):
    if (module_identity(config.get('module')) != MODULE or config.get('operation') not in OPERATIONS
            or config.get('scene') != session.scene_name or config.get('room') != session.room_name
            or session.room_name != 'full_scene'):
        raise ValueError('passage_operation_scene_contract_mismatch')
    p = config['parameters']
    if (not p.get('start_region') or not p.get('target_region') or p['start_region'] == p['target_region']
            or not _vector(p.get('start_xy'), 2) or not math.isfinite(p['initial_yaw_rad'])
            or p['floor_idx'] != 0 or p['floor_z_m'] != 0.0 or p['stability_steps'] < 1
            or not 0 < p['floor_contact_tolerance_m'] <= 0.005 or not 0.95 <= p['minimum_up_dot'] <= 1
            or not 0 < p['maximum_terminal_speed_m_s'] <= 0.02
            or not 0 < p['maximum_terminal_angular_speed_rad_s'] <= 0.1
            or (config['operation'] in LEGACY_OPERATIONS and p.get('minimum_travel_m', 0) < 0.25)):
        raise ValueError('invalid_private_passage_parameters')
    return p


def _region(segmentation, xy):
    # Use the installed map's tensor coordinate convention explicitly.
    import torch
    return segmentation.get_room_instance_by_point(torch.tensor(list(xy), dtype=torch.float32))


def measure_native_state(session, parameters):
    """Read measured robot pose, chassis bounds, room raster and native contacts."""
    robot, scene = session.robot, session.env.scene
    position, quaternion = (_array(value) for value in robot.get_position_orientation())
    if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) <= 0):
        raise ValueError('invalid_native_robot_pose')
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    up_dot = float(1 - 2 * (x*x + y*y))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    seg = scene.seg_map
    known = sorted(seg.room_ins_name_to_ins_id)
    floor_z = float(seg.floor_heights[parameters['floor_idx']])
    if not math.isfinite(floor_z):
        raise ValueError('native_floor_height_unavailable')
    chassis = robot.links[robot.base_footprint_link_name]
    low, high = (_array(value) for value in chassis.aabb)
    if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
            or np.any(high[:2] <= low[:2])):
        raise ValueError('native_chassis_footprint_unavailable')
    footprint = [_region(seg, [px, py]) for px in (low[0], (low[0]+high[0])/2, high[0])
                 for py in (low[1], (low[1]+high[1])/2, high[1])]
    root_region = _region(seg, position[:2])
    chassis_region = root_region if root_region in known and all(r == root_region for r in footprint) else None
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floor_paths = [obj.prim_path for obj in objects if getattr(obj, 'category', None) in ('floor', 'floors')]
    if not floor_paths:
        raise ValueError('native_floor_objects_unavailable')
    within = lambda body, path: body == path or body.startswith(path + '/')
    floor_contacts, nonfloor_contacts = [], []
    for contact in robot.contact_list():
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not within(body, robot.prim_path)]
        if not external:
            continue
        point, normal = _array(contact.position), _array(contact.normal)
        row = {'bodies': bodies,
            'position': point.tolist() if point.shape == (3,) and np.isfinite(point).all() else None,
            'normal': normal.tolist() if normal.shape == (3,) and np.isfinite(normal).all() else None}
        floors = [any(within(body, path) for path in floor_paths) for body in external]
        if all(floors):
            floor_contacts.append(row)
        elif not legal_background_contact(session, *bodies):
            nonfloor_contacts.append(row)
    floor_supported = bool(floor_contacts) and all(row['position'] is not None and row['normal'] is not None
        and abs(row['position'][2] - floor_z) <= parameters['floor_contact_tolerance_m']
        and abs(row['normal'][2]) >= 0.5 for row in floor_contacts)
    linear, angular = (_array(value) for value in (chassis.get_linear_velocity(), chassis.get_angular_velocity()))
    if linear.shape != (3,) or angular.shape != (3,) or not np.isfinite([linear, angular]).all():
        raise ValueError('native_chassis_velocity_unavailable')
    floor_matches = abs(floor_z - parameters['floor_z_m']) <= parameters['floor_contact_tolerance_m']
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'yaw_rad': yaw,
        'up_dot': up_dot, 'root_region': root_region, 'chassis_region': chassis_region,
        'footprint_regions': footprint, 'known_regions': known, 'native_floor_z_m': floor_z,
        'floor_layer_matches': floor_matches, 'floor_supported': floor_supported,
        'floor_contacts': floor_contacts, 'nonfloor_contacts': nonfloor_contacts,
        'linear_speed_m_s': float(np.linalg.norm(linear)), 'angular_speed_rad_s': float(np.linalg.norm(angular)),
        # New exploration permits supported unlabelled doorway transit. Such
        # cells never establish chassis_region or earn native room visits.
        'state_valid': bool((root_region in known or parameters.get('arrival_report_required') is True)
            and floor_matches and floor_supported
            and not nonfloor_contacts and up_dot >= parameters['minimum_up_dot'])}


def _start_checks(session, config):
    p = _parameters(session, config)
    measured = measure_native_state(session, p)
    heading_error = abs(math.degrees(math.atan2(math.sin(measured['yaw_rad'] - p['initial_yaw_rad']),
                                              math.cos(measured['yaw_rad'] - p['initial_yaw_rad']))))
    # Native R1 is a fixed-root articulation whose holonomic base joints move
    # the physical base_link. fixed_base=True is correct for this robot.
    checks = {'native_r1': type(session.robot).__name__ == 'R1' and not session.robot.visual_only,
        'native_holonomic_base': type(session.robot.controllers.get('base')).__name__ == 'HolonomicBaseJointController',
        'native_endpoint_regions_exist': all(p[key] in measured['known_regions'] for key in ('start_region', 'target_region')),
        'source_initial_point_maps_to_start': _region(session.env.scene.seg_map, p['start_xy']) == p['start_region'],
        'native_start_chassis_region': measured['chassis_region'] == p['start_region'],
        'native_start_position_matches': float(np.linalg.norm(np.array(measured['position'][:2]) - p['start_xy']))
            <= p['start_position_tolerance_m'],
        'native_start_heading_matches': heading_error <= p['start_heading_tolerance_deg'],
        'native_floor_contact_and_safe_pose': measured['state_valid']}
    checks['original_start_matches'] = bool(checks['source_initial_point_maps_to_start']
        and checks['native_start_position_matches'] and checks['native_start_heading_matches'])
    repair = verified_repaired_start(session, p['start_region'], np.asarray(measured['position']),
        np.asarray(measured['quaternion_xyzw']), position_tolerance_m=p['start_position_tolerance_m'],
        heading_tolerance_deg=p['start_heading_tolerance_deg'])
    checks['verified_repair_evidence'] = repair
    checks['start_contract_accepted'] = checks['original_start_matches'] or repair['accepted']
    checks['start_contract'] = ('original_exact_pose' if checks['original_start_matches'] else
        'verified_repaired_pose_same_source_region' if repair['accepted'] else 'rejected')
    return measured, checks


def _start_accepted(checks):
    required = ('native_r1', 'native_holonomic_base', 'native_endpoint_regions_exist',
                'native_start_chassis_region', 'native_floor_contact_and_safe_pose', 'start_contract_accepted')
    return all(checks[key] for key in required)


def qualify_operation(session, executor, config, bound):
    if config.get('operation') == REVISED_OPERATION:
        return qualify_revised_mapping(session, executor, config, bound, qualify_operation)
    """Measure initialization only; the complete route remains unqualified."""
    try:
        measured, checks = _start_checks(session, config)
        checks['guarded_navigation_capabilities'] = bool(executor.capabilities.base_translation
                                                        and executor.capabilities.base_rotation)
        accepted = _start_accepted(checks) and checks['guarded_navigation_capabilities']
        return {'accepted': accepted, 'checks': checks, 'native_state': measured, 'route_qualified': False,
                'reason': None if accepted else 'native_passage_start_precondition_failed'}
    except (ValueError, RuntimeError, KeyError, AttributeError, IndexError, AssertionError) as error:
        return {'accepted': False, 'checks': {}, 'route_qualified': False,
                'reason': f'native_passage_measurement_unavailable:{type(error).__name__}:{error}'}


class LegacyPassageEvaluator:
    """Measure real ordered visits without consulting source paths or QA answers."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = _parameters(session, config)
        self.required = [self.parameters['target_region']]
        if config['operation'] == 'passage_round_trip':
            self.required.append(self.parameters['start_region'])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _time(self):
        value = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(value):
            raise ValueError('native_simulation_clock_unavailable')
        return value

    def begin_episode(self, observation):
        measured, checks = _start_checks(self.session, self.config)
        if not _start_accepted(checks):
            raise ValueError('native_passage_episode_start_rejected')
        self.episode += 1
        self.active = True
        self.previous = np.asarray(measured['position'])
        self.previous_time = self._time()
        self.stage, self.consecutive, self.samples, self.terminal_consecutive = 0, 0, 0, 0
        self.travel_m, self.faults = 0.0, set()
        self.visits = [{'region': self.parameters['start_region'], 'sample': 0}]
        self.start_checks = checks
        return self.record(observation, action=None)

    def sample_step(self):
        if not self.active:
            return
        measured = measure_native_state(self.session, self.parameters)
        point, now = np.asarray(measured['position']), self._time()
        elapsed = now - self.previous_time
        distance = float(np.linalg.norm(point - self.previous))
        cfg = self.controller.config
        limit = cfg.max_translation_speed_m_s * max(elapsed, 0.0) + cfg.translation_tolerance_m
        if elapsed <= 0 or distance > limit:
            self.faults.add('motion_clock_or_continuity_failed')
        if not measured['state_valid']:
            self.faults.add('native_floor_contact_or_pose_failed')
        self.travel_m += distance
        self.samples += 1
        settled = (measured['linear_speed_m_s'] <= self.parameters['maximum_terminal_speed_m_s']
                   and measured['angular_speed_rad_s'] <= self.parameters['maximum_terminal_angular_speed_rad_s'])
        terminal_valid = measured['state_valid'] and settled and measured['chassis_region'] == self.required[-1]
        self.terminal_consecutive = self.terminal_consecutive + 1 if terminal_valid else 0
        if self.stage < len(self.required):
            at_next = (measured['state_valid'] and settled and not self.faults
                       and measured['chassis_region'] == self.required[self.stage])
            self.consecutive = self.consecutive + 1 if at_next else 0
            if self.consecutive >= self.parameters['stability_steps']:
                self.visits.append({'region': self.required[self.stage], 'sample': self.samples,
                                    'position': measured['position']})
                self.stage += 1
                self.consecutive = 0
        self.previous, self.previous_time = point, now
        with (self.output / 'physical_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'sample': self.samples, 'simulation_time_s': now,
                'native_state': measured, 'travel_m': self.travel_m, 'visit_stage': self.stage,
                'faults': sorted(self.faults)}, allow_nan=False) + '\n')

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = measure_native_state(self.session, self.parameters)
        if not measured['state_valid']:
            self.faults.add('native_floor_contact_or_pose_failed')
        if float(np.linalg.norm(np.asarray(measured['position']) - self.previous)) > self.controller.config.translation_tolerance_m:
            self.faults.add('unsampled_native_motion')
        criteria = {'qualified_start': _start_accepted(self.start_checks),
            'ordered_room_visits': self.stage == len(self.required),
            'positive_measured_travel': self.samples > 0 and self.travel_m >= self.parameters['minimum_travel_m'],
            'continuous_safe_motion': not self.faults,
            'native_terminal_state_valid': measured['state_valid'],
            'full_chassis_in_terminal_room': measured['chassis_region'] == self.required[-1],
            'terminal_stability_samples': self.terminal_consecutive >= self.parameters['stability_steps'],
            'terminal_at_rest': measured['linear_speed_m_s'] <= self.parameters['maximum_terminal_speed_m_s']
                and measured['angular_speed_rad_s'] <= self.parameters['maximum_terminal_angular_speed_rad_s']}
        row = {'schema': 'private_native_passage_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
            'task_success': bool(all(criteria.values())), 'criteria': criteria, 'native_state': measured,
            'physical_travel_m': self.travel_m, 'control_step_samples': self.samples,
            'required_native_room_visits': self.required, 'measured_visits': self.visits,
            'motion_faults': sorted(self.faults), 'initial_qualification': self.start_checks,
            'configuration': self.config}
        (self.output / 'state.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        return row


def create_evaluator(session, controller, config, output):
    if config.get('operation') == REVISED_OPERATION:
        native = create_evaluator(session, controller, legacy_mapping_config(config), output)
        return SourceAnswerMappingEvaluator(session, controller, config, output, native)
    evaluator = PassageEvaluator if config['operation'] == EXPLORATION_OPERATION else LegacyPassageEvaluator
    return evaluator(session, controller, config, output)


class PassageEvaluator(NativeVisitReportEvaluator):
    """Visit the public endpoint set, allowing free exploration and backtracking."""
    def __init__(self, session, controller, config, output):
        _parameters(session, config)
        super().__init__(session, controller, config, output)

    def _measure_native(self):
        return measure_native_state(self.session, self.parameters)

    def _qualify_start(self):
        measured, checks = _start_checks(self.session, self.config)
        self.start_evidence = checks
        return {'native_original_start_qualified': _start_accepted(checks),
            'native_required_regions_exist': self.required.issubset(measured['known_regions'])}

"""Convert Connectivity into native room exploration without changing its map.

Run CategoryConverter().convert_all(sources, ConversionContext(output_root))
for CPU conversion. The owned operation hooks run only during native episodes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from wia.tasks.background_policy import legal_background_contact

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.authoring.converters.cognitive_mapping_visit_reports import (
    ARRIVAL_INSTRUCTION, completion_public_goal, NativeVisitReportEvaluator, native_array, settled_native_state, visit_parameters)


from wia.authoring.converters.cognitive_mapping_visit_reports import (
    REVISED_MAPPING_OPERATIONS, revise_mapping_result, legacy_mapping_config,
    qualify_revised_mapping, SourceAnswerMappingEvaluator)

MODULE = 'wia.authoring.converters.cognitive_mapping_connectivity'
REVISED_OPERATION = REVISED_MAPPING_OPERATIONS['Connectivity']
SOURCE_EYE_HEIGHT_M = 1.35
INITIAL_CLEARANCE_M = .1
EXPLORATION_OPERATION = 'region_exploration_arrivals'
LEGACY_OPERATIONS = ('region_route_arrival', 'region_round_trip')
OPERATIONS = (EXPLORATION_OPERATION, *LEGACY_OPERATIONS)


def source_initial_pose(payload):
    """Propose the saved source-room XY/heading; native stance checks still decide.

    The original Connectivity generator places its camera 1.35 m above floor 0.
    This is an initialization request, never a camera-mount or world-asset edit.
    """
    question = payload.get('question_data', {})
    view = question.get('render', {}).get('initial_view', question.get('initial_view', {}))
    pose = view.get('camera_pose', {})
    position, target = pose.get('position'), pose.get('target')
    if (view.get('room_instance') != question.get('source_region')
            or not isinstance(position, (list, tuple)) or len(position) != 3
            or not isinstance(target, (list, tuple)) or len(target) != 3
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                       for x in [*position, *target])):
        raise ValueError('source_initial_room_pose_unavailable')
    dx, dy = target[0] - position[0], target[1] - position[1]
    if math.hypot(dx, dy) <= 1e-6:
        raise ValueError('source_initial_horizontal_heading_undefined')
    return {'operation': 'robot_pose', 'phase': 'before_construction',
        'position': [float(position[0]), float(position[1]),
                     float(position[2]) - SOURCE_EYE_HEIGHT_M + INITIAL_CLEARANCE_M],
        'yaw_deg': math.degrees(math.atan2(dy, dx)), 'head_pitch_deg': 0.0}


def public_regions(question):
    """Split the two public region labels outside descriptive parentheses."""
    suffix = ' adjacent, with a direct connection between them?'
    if not question.startswith('Are ') or not question.endswith(suffix):
        return None
    pair = question[4:-len(suffix)]
    depth = 0
    for index, character in enumerate(pair):
        depth += (character == '(') - (character == ')')
        if depth < 0:
            return None
        if depth == 0 and pair[index:index + 5] == ' and ':
            labels = (pair[:index], pair[index + 5:])
            names = tuple(label.split('(', 1)[0].strip().replace(' ', '_') for label in labels)
            return {'labels': labels, 'names': names} if all(names) and names[0] != names[1] else None
    return None


def public_goal(raw, operation):
    regions = public_regions(raw['question'])
    first, second = regions['labels']
    if operation == EXPLORATION_OPERATION:
        return {'objective': f'Explore the original passages and visit both {first} and {second}, '
                    'in any order. ' + ARRIVAL_INSTRUCTION,
            'success_criteria': 'Stably occupy and explicitly report each named room with the full '
                'chassis supported by the floor and clear of other objects. Extra rooms and backtracking '
                'are allowed. An already occupied required room may be reported after stable native samples.',
            'environment': 'The original full-scene floorplan, furniture and passages are retained.',
            'constraints': ['Use real guarded base motion and mounted observations.',
                'Keep room-defining furniture and the original building unchanged.',
                'Choose your route freely; no return trip or terminal room is prescribed.']}
    objective = f'Investigate the original passages. First enter {first}, then navigate into {second}.'
    terminal = second
    if operation == 'region_round_trip':
        objective += f' Return to {first} after visiting {second}.'
        terminal = first
    options = raw.get('options_json', [])
    options = json.loads(options) if isinstance(options, str) else options
    return {'objective': objective + ' Original question: ' + raw['question'],
        'success_criteria': f'The robot chassis enters the named regions in the requested order '
            f'and finishes fully inside {terminal}, supported by the floor and clear of other objects.',
        'environment': 'The original full-scene floorplan, furniture and passages are retained. '
            'Original answer choices: ' + '; '.join(options) + '.',
        'constraints': ['Use real base motion through the original passages.',
            'Intermediate rooms may be used; investigate direct adjacency separately.',
            'Keep room-defining furniture and the original building unchanged.']}


def _operation(source, context):
    configured = context.config.get('connectivity_operation')
    if configured is not None:
        if configured != EXPLORATION_OPERATION:
            raise ValueError('New Connectivity conversion requires unordered room exploration; legacy recipes are read-only.')
        return configured
    return EXPLORATION_OPERATION


def _unreachable_source_conflict(source):
    """A negative graph annotation cannot override contradictory source evidence."""
    q = source.payload.get('question_data', {})
    verification = q.get('verification', {})
    if verification.get('graph_reachable') is not False:
        return None
    if (q.get('directly_adjacent') is not False or q.get('answer_bool') is not False
            or source.raw.get('answer') != 'No' or q.get('answer') != 'No'
            or source.raw.get('question') != q.get('question')):
        return 'unreachable_source_contradicts_direct_adjacency_annotations'
    if len(q.get('graph_shortest_region_path') or []) >= 2:
        return 'unreachable_source_contains_a_region_path'
    if any(len(attempt.get('path_world_xy') or []) >= 2
           for attempt in verification.get('shortest_path_attempts', [])):
        return 'unreachable_source_contains_a_successful_path_trace'
    return None


class CategoryConverter(ConverterBase):
    category = ('Cognitive Mapping', 'Connectivity')
    version = 'connectivity_task_edit_20261018'

    def inspect(self, source):
        raw, payload = source.raw, source.payload
        if (source.big_task, source.small_task) != self.category or (
                raw.get('big_task'), raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'category_mismatch'}
        regions = public_regions(raw.get('question', ''))
        question = payload.get('question_data', {})
        if not regions or question.get('case_type') != 'pair_connectivity':
            return {'eligible': False, 'reason': 'public_region_pair_not_resolved'}
        if regions['names'] != (question.get('source_region'), question.get('target_region')):
            return {'eligible': False, 'reason': 'public_private_region_pair_disagreement'}
        if raw.get('room') != 'full_scene':
            return {'eligible': False, 'reason': 'original_full_scene_required'}
        conflict = _unreachable_source_conflict(source)
        if conflict:
            return {'eligible': True, 'reason': conflict,
                    'warning_codes': ['source_unreachability_annotation_conflict']}
        expected = 'Yes' if question.get('directly_adjacent') is True else 'No'
        if raw.get('answer') != expected or question.get('answer') != expected or question.get(
                'answer_bool') != question.get('directly_adjacent'):
            return {'eligible': False, 'reason': 'source_connectivity_annotation_disagreement',
                'warning_codes': ['source_annotation_conflict']}
        try:
            source_initial_pose(payload)
        except ValueError as error:
            return {'eligible': False, 'reason': str(error),
                'warning_codes': ['source_initial_room_pose_unavailable']}
        return {'eligible': True, 'reason': 'public_regions_support_an_original_scene_navigation_goal'}

    def sample(self, manifest, policy=None):
        selected = super().sample(manifest, policy)
        rows = {row['candidate_id']: row for row in manifest['records']}
        for operation in OPERATIONS:
            candidates = sorted(row['candidate_id'] for row in manifest['records'] if
                row.get('operation_recipe') == operation and row['status'] == 'generated')
            if candidates and selected and not set(candidates).intersection(selected):
                replaceable = [index for index, key in enumerate(selected) if rows[key]['status'] == 'generated']
                if replaceable:
                    selected[replaceable[-1]] = candidates[0]
        return selected

    def convert(self, source, context):
        conflict = _unreachable_source_conflict(source)
        if conflict:
            return {'scene_definition': original_scene(source, context), 'task_definition': None,
                'status': 'needs_human_decision', 'failure_reason': conflict,
                'diagnostic_stage': 'source_reachability_consistency', 'qa_preserved': True,
                'private_source_evidence': {'original_question': source.raw.get('question'),
                    'original_answer': source.raw.get('answer'), 'question_data': source.payload['question_data']},
                'warning_codes': ['source_unreachability_annotation_conflict'],
                'next_step': 'Reconcile contradictory source adjacency/path evidence before defining an executable goal.'}
        result = self._convert_legacy(source, context)
        if source.payload['question_data'].get('verification', {}).get('graph_reachable') is False:
            # Existing saved historical definitions retain their evaluator. A
            # newly generated unreachable pair must never acquire arrival goals.
            return revise_mapping_result(source, context, result)
        if context.config.get('mapping_revision') == 'legacy' or context.config.get('connectivity_operation'):
            return result
        return revise_mapping_result(source, context, result)

    def _convert_legacy(self, source, context):
        raw, payload = source.raw, source.payload
        question = payload['question_data']
        names = public_regions(raw['question'])['names']
        operation = _operation(source, context)
        patches = [source_initial_pose(payload)]
        parameters = {'source_region': names[0], 'target_region': names[1], 'stability_steps': 3,
            'floor_idx': 0, 'floor_contact_height_tolerance_m': 0.005,
            'minimum_floor_contact_vertical_normal': 0.5}
        parameters.update(visit_parameters(names))
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': [], 'parameters': parameters}
        task = physical_task(source, context, public=completion_public_goal(public_goal(raw, operation), self.category[1]), evaluation=evaluation,
            bindings={}, patches=patches, capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        task['defaults']['max_steps'] = int(context.config.get('connectivity_max_steps', 600))
        warnings = ['assumption_review_required', 'native_route_execution_unverified',
            'robot_initialization_unqualified', 'graph_path_advisory_only', 'native_floor_layer_unverified',
            'source_camera_xy_is_not_chassis_clearance', 'native_door_passability_unverified']
        assumptions = [{
            'doubt': 'A source camera path and graph connectivity do not establish R1 chassis '
                'clearance, floor support, safe initialization or executable passages.',
            'chosen_assumption': 'Retain the original full scene and request the saved source-room '
                'camera XY with its horizontal heading for the existing R1 before construction. '
                'Native stance search must still measure floor support and full-body clearance; '
                'the source camera point does not force an unsafe robot pose. Native qualification must resolve room names and read '
                'robot floor/contact state before an operation can run.',
            'alternatives': ['Qualify a source-anchored robot-only start using actual native floor '
                'measurements if the retained default start fails.', 'Keep blocked routes as '
                'diagnostic Review items without enlarging passages or changing furniture.'],
            'scope': source.relative_path,
            'risk': 'An original camera-accessible passage may be too narrow or unsafe for the '
                'actual R1. No route or initialization is qualified on CPU.',
        }, {
            'doubt': 'The original QA asks direct adjacency, whereas a useful physical visit '
                'operation may traverse intermediate rooms.',
            'chosen_assumption': 'Require the set of two publicly named rooms and explicit Agent '
                'arrival reports, in any order with exploration and backtracking allowed. Keep '
                'direct-adjacency QA private and distinct from physical navigation success.',
            'alternatives': ['Add a separately documented direct-crossing operation after native '
                'passage evidence.', 'Retain a scene-only diagnostic if no faithful robot route exists.'],
            'scope': 'This category; no ordered route is present in the public adjacency question.',
            'risk': 'Completing a route does not prove a direct doorway, and must not replace '
                'the original adjacency scorer.',
        }]
        return {'scene_definition': original_scene(source, context, patches=patches), 'task_definition': task,
            'interaction_policy': {'mode': 'navigation', 'group_bindings': [], 'manipulation_bindings': [],
                'relocation_policy': 'preserve', 'source_region': names[0],
                'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                'maximum_stance_distance_m': .85},
            'status': 'generated', 'reuse_level': 'L1', 'operation_recipe': operation,
            'available_operation_recipes': [EXPLORATION_OPERATION],
            'change_summary': ['Preserve the original full native floorplan, furniture, camera '
                'reference and source QA.', 'Request the saved source-room initial XY and horizontal '
                'heading before mandatory measured native stance screening.',
                'Add unordered stable native room visits and explicit '
                'same-room Agent arrival reports; no object binding is needed.'],
            'warning_codes': warnings, 'assumptions': assumptions,
            'private_navigation_advice': {'graph_shortest_region_path': question.get('graph_shortest_region_path'),
                'source_camera_path_length_m': question.get('path_views', {}).get('path_length_m'),
                'source_verification_method': question.get('verification', {}).get('method'),
                'advisory_only': True},
            'limitations': ['Native route, doorway clearance, robot initialization and sensor '
                'coverage remain unverified.', 'The floor-0 room raster requires measured native '
                'contact heights matching its declared floor frame; no height or layer is guessed.',
                'Room segmentation is private measurement evidence, '
                'not an Agent map or a proof of collision-free traversal.', 'Three consecutive '
                'native control samples establish room entry; they do not certify every swept '
                'body surface or the entire route.', 'The uniform 600-decision default is a new '
                'navigation budget proposal, not a measured or calibrated runtime guarantee. '
                'Wall-clock and action/controller limits retain existing values.'],
            'attempted_repairs': [], 'next_step': 'Review preserved room references and qualify '
                'native robot initialization, then execute guarded base navigation through original passages.'}


def _array(value):
    return native_array(value)


def _validate_config(session, config):
    if config.get('operation') not in OPERATIONS or (config.get('scene'), config.get('room')) != (
            session.scene_name, session.room_name) or session.room_name != 'full_scene':
        raise ValueError('Connectivity operation must match its original full native scene.')
    parameters = config['parameters']
    if not parameters.get('source_region') or not parameters.get('target_region') or parameters[
            'source_region'] == parameters['target_region'] or int(parameters.get('stability_steps', 0)) < 1:
        raise ValueError('Declare two distinct regions and positive native stability samples.')
    if parameters.get('floor_idx') != 0 or not 0 < parameters.get('floor_contact_height_tolerance_m', 0) <= 0.005 or not (
            0.5 <= parameters.get('minimum_floor_contact_vertical_normal', 0) <= 1):
        raise ValueError('Use the original floor-0 map with its conservative contact-layer boundary.')
    return parameters


def measure_native_state(session, pose=None, *, parameters):
    """Read actual robot/chassis, original room segmentation and native contacts."""
    robot, scene = session.robot, session.env.scene
    if pose is None:
        position, quaternion = robot.get_position_orientation()
    else:
        position, quaternion = pose['position'], pose['quaternion_xyzw']
    position, quaternion = _array(position), _array(quaternion)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(
            quaternion).all() or not np.linalg.norm(quaternion):
        raise ValueError('Native robot pose is invalid.')
    segmentation = scene.seg_map
    room_names = segmentation.room_ins_name_to_ins_id
    heights = getattr(segmentation, 'floor_heights', None)
    floor_idx = parameters['floor_idx']
    layer_known = (isinstance(heights, (list, tuple, np.ndarray)) and len(heights) > floor_idx
        and np.isfinite(heights[floor_idx]))
    floor_z = float(heights[floor_idx]) if layer_known else None
    root_region = segmentation.get_room_instance_by_point(position[:2])
    low, high = (_array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high[:2] <= low[:2]):
        raise ValueError('Native physical chassis footprint is unavailable.')
    footprint_regions = [segmentation.get_room_instance_by_point(np.array([x, y]))
        for x in (low[0], (low[0] + high[0]) / 2, high[0])
        for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floor_paths = [obj.prim_path for obj in objects if getattr(obj, 'category', None) in ('floor', 'floors')]
    contacts, nonfloor, floor_contacts = [], [], []
    for contact in robot.contact_list():
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not (body == robot.prim_path or body.startswith(robot.prim_path + '/'))]
        if not external:
            continue
        contact_position = _array(contact.position) if hasattr(contact, 'position') else np.empty(0)
        contact_normal = _array(contact.normal) if hasattr(contact, 'normal') else np.empty(0)
        row = {'body0': bodies[0], 'body1': bodies[1],
            'position': contact_position.tolist() if contact_position.shape == (3,) and np.isfinite(contact_position).all() else None,
            'normal': contact_normal.tolist() if contact_normal.shape == (3,) and np.isfinite(contact_normal).all() else None}
        contacts.append(row)
        is_floor = [bool('groundPlane' in body or any(body == root or body.startswith(root + '/')
            for root in floor_paths)) for body in external]
        if any(is_floor):
            floor_contacts.append(row)
        if not all(is_floor) and not legal_background_contact(session, *bodies):
            nonfloor.append(row)
    region = root_region if root_region is not None and all(room == root_region for room in footprint_regions) else None
    floor_contact = bool(layer_known and floor_contacts and all(row['position'] is not None
        and row['normal'] is not None and abs(row['position'][2] - floor_z) <= parameters[
            'floor_contact_height_tolerance_m'] and abs(row['normal'][2]) >= parameters[
            'minimum_floor_contact_vertical_normal'] for row in floor_contacts))
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'root_region': root_region, 'chassis_region': region, 'footprint_regions': footprint_regions,
        'known_regions': sorted(room_names), 'floor_contact': floor_contact,
        'floor_layer_known': bool(layer_known), 'segmentation_floor_idx': floor_idx,
        'segmentation_floor_z_m': floor_z, 'floor_contact_present': bool(floor_contacts),
        'native_floor_contacts': floor_contacts,
        'nonfloor_contacts': nonfloor, 'contacts': contacts,
        # Unlabelled doorway cells do not invalidate measured floor/contact safety.
        # Only a known region containing the sampled chassis footprint earns a visit.
        'state_valid': bool(floor_contact and not nonfloor),
        'visit_eligible': bool(region in room_names)}


def qualify_operation(session, executor, config, bound):
    if config.get('operation') == REVISED_OPERATION:
        return qualify_revised_mapping(session, executor, config, bound, qualify_operation)
    """Read preconditions only; never execute an action or move a native object."""
    checks = {}
    try:
        parameters = _validate_config(session, config)
        measured = measure_native_state(session, parameters=parameters)
        checks = {'native_regions_exist': all(parameters[key] in measured['known_regions']
            for key in ('source_region', 'target_region')),
            'base_motion_capabilities': bool(executor.capabilities.base_translation and executor.capabilities.base_rotation),
            'robot_in_original_room_map': measured['root_region'] in measured['known_regions'],
            'native_floor_layer_known': measured['floor_layer_known'],
            'native_floor_contact': measured['floor_contact'],
            'no_nonfloor_robot_contact': not measured['nonfloor_contacts']}
        accepted = all(checks.values())
        return {'accepted': bool(accepted), 'checks': checks,
            'reason': None if accepted else 'native_navigation_initialization_or_region_precondition_failed',
            'native_state': measured, 'route_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_operation_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}', 'route_qualified': False}


class LegacyConnectivityEvaluator:
    """Track physical ordered visits; source graphs and answers cannot score success."""
    def __init__(self, session, controller, config, output):
        parameters = _validate_config(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.parameters = parameters
        self.required = [parameters['source_region'], parameters['target_region']]
        if config['operation'] == 'region_round_trip':
            self.required.append(parameters['source_region'])
        self.stability_steps = int(parameters['stability_steps'])
        # Preserve the actual controller's movement bound; no new speed limit is imposed.
        self.maximum_sample_displacement = (controller.config.max_translation_speed_m_s * controller.dt
            + controller.config.translation_tolerance_m)
        if not math.isfinite(self.maximum_sample_displacement) or self.maximum_sample_displacement <= 0:
            raise ValueError('Native controller sampling bounds are unavailable.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _observe(self, *, count_step):
        measured = measure_native_state(self.session, self.controller.pose(), parameters=self.parameters)
        point = np.asarray(measured['position'][:2])
        if self.previous is not None:
            displacement = float(np.linalg.norm(point - self.previous))
            self.travel_m += displacement
            if displacement > self.maximum_sample_displacement:
                self.continuous = False
        self.previous = point
        if not measured['state_valid']:
            self.contact_safe = False
        eligible = (self.stage < len(self.required) and measured['state_valid']
            and measured['visit_eligible'] and self.continuous and self.contact_safe
            and measured['chassis_region'] == self.required[self.stage])
        if not eligible:
            # Observation-only boundaries also interrupt consecutive visit credit.
            self.consecutive = 0
        if count_step:
            self.sample_count += 1
            if eligible:
                self.consecutive += 1
                if self.consecutive >= self.stability_steps:
                    self.visits.append({'region': self.required[self.stage], 'sample': self.sample_count,
                        'position': measured['position']})
                    self.stage += 1
                    self.consecutive = 0
        self.last = measured
        return measured

    def begin_episode(self, observation):
        self.episode += 1
        self.active = True
        self.stage, self.consecutive, self.sample_count = 0, 0, 0
        self.visits, self.previous = [], None
        self.continuous, self.contact_safe, self.travel_m = True, True, 0.0
        self._observe(count_step=False)
        return self.record(observation, action=None)

    def sample_step(self):
        if self.active:
            self._observe(count_step=True)

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = self._observe(count_step=False)
        checks = {'ordered_native_visits': self.stage == len(self.required),
            'inside_terminal_region': measured['chassis_region'] == self.required[-1],
            'native_floor_contact': measured['floor_contact'], 'contact_safe_trajectory': self.contact_safe,
            'native_floor_layer_known': measured['floor_layer_known'],
            'continuous_sampled_motion': self.continuous, 'sampled_real_displacement': self.travel_m > 0,
            'native_control_samples': self.sample_count >= self.stability_steps * len(self.required)}
        value = {'schema': 'private_connectivity_operation_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'operation': self.config['operation'], 'task_success': bool(all(checks.values())),
            'checks': checks, 'native_state': measured, 'ordered_visits': self.visits,
            'required_regions': self.required, 'sample_count': self.sample_count,
            'sampled_travel_m': self.travel_m, 'maximum_sample_displacement_m': self.maximum_sample_displacement,
            'configuration': self.config}
        (self.output / 'state.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == REVISED_OPERATION:
        native = create_evaluator(session, controller, legacy_mapping_config(config), output)
        return SourceAnswerMappingEvaluator(session, controller, config, output, native)
    evaluator = ConnectivityEvaluator if config['operation'] == EXPLORATION_OPERATION else LegacyConnectivityEvaluator
    return evaluator(session, controller, config, output)


class ConnectivityEvaluator(NativeVisitReportEvaluator):
    """Unordered native room set plus explicit same-boundary Agent reports."""
    def __init__(self, session, controller, config, output):
        _validate_config(session, config)
        super().__init__(session, controller, config, output)

    def _measure_native(self):
        measured = measure_native_state(self.session, parameters=self.parameters)
        return settled_native_state(self.session, measured)

    def _qualify_start(self):
        measured = self._measure_native()
        regions = self.required or {self.parameters['source_region'], self.parameters['target_region']}
        return {'native_required_regions_exist': regions.issubset(measured['known_regions']),
            'native_initial_floor_contact_and_pose': measured['state_valid'],
            'native_initial_region_known': measured['root_region'] in measured['known_regions']}

"""Convert full-scene navigation questions and measure real R1 room arrival.

Run category conversion with ConverterBase.convert_all. Native qualification
and evaluation are called by wia.tasks.scene only after task approval.
The evaluator never moves the robot, objects, cameras or simulation clock.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

from wia.tasks.background_policy import legal_background_contact

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.tasks.interaction_setup import verified_repaired_start
from wia.authoring.converters.cognitive_mapping_connectivity import measure_native_state as measure_room_state
from wia.authoring.converters.cognitive_mapping_visit_reports import (
    ARRIVAL_INSTRUCTION, completion_public_goal, NativeVisitReportEvaluator, native_array, public_navigation_route,
    settled_native_state, visit_parameters)

from wia.authoring.converters.cognitive_mapping_visit_reports import (
    REVISED_MAPPING_OPERATIONS, revise_mapping_result, legacy_mapping_config,
    qualify_revised_mapping, SourceAnswerMappingEvaluator)

MODULE = 'wia.authoring.converters.cognitive_mapping_long_term_navigation'
REVISED_OPERATION = REVISED_MAPPING_OPERATIONS['Long-Term Navigation']
LEGACY_OPERATION = 'navigate_from_start_to_destination'
OPERATION = 'navigation_exploration_arrivals'
ENDPOINTS = re.compile(r'^You are beginning in the (?P<start>.+?)\. You want to navigate to the (?P<goal>.+?)\.', re.DOTALL)
# Installed Environment.default_config.initial_pos_z_offset is .1 m. This is
# collision-check initialization clearance, not a calibrated R1 base height.
INITIAL_CLEARANCE_M = .1


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value))


def _endpoint_id(text):
    text = text.split(' standing at the point shown in the image')[0].split(' (')[0]
    return '_'.join(text.lower().strip().split())


def _parameters(config):
    if config.get('operation') not in (OPERATION, LEGACY_OPERATION):
        raise ValueError('unsupported_navigation_operation')
    p = config['parameters']
    if (not _vector(p.get('start_xy'), 2) or not _vector(p.get('goal_xy'), 2)
            or not isinstance(p.get('start_region'), str) or not isinstance(p.get('target_region'), str)
            or p['start_region'] == p['target_region']):
        raise ValueError('invalid_navigation_endpoint_contract')
    return p


def _array(value):
    return native_array(value)


def _region(scene, xy):
    import torch
    seg = getattr(scene, 'seg_map', None)
    if seg is None:
        raise ValueError('native_room_segmentation_unavailable')
    return seg.get_room_instance_by_point(torch.tensor(list(xy), dtype=torch.float32))


def _floor_paths(scene):
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    paths = [obj.prim_path for obj in objects if getattr(obj, 'category', None) in ('floor', 'floors')]
    if not paths:
        raise ValueError('native_floor_objects_unavailable')
    return paths


def _contacts(robot, floor_paths, floor_z, session=None):
    contacts = []
    for contact in robot.contact_list():
        position, normal = getattr(contact, 'position', None), getattr(contact, 'normal', None)
        position = _array(position) if position is not None else np.empty(0)
        normal = _array(normal) if normal is not None else np.empty(0)
        contacts.append({'body0': contact.body0, 'body1': contact.body1,
                         'position': position.tolist() if position.shape == (3,) and np.isfinite(position).all() else None,
                         'normal': normal.tolist() if normal.shape == (3,) and np.isfinite(normal).all() else None})
    floor = lambda body: any(body == path or body.startswith(path + '/') for path in floor_paths)
    robot_body = lambda body: body == robot.prim_path or body.startswith(robot.prim_path + '/')
    floor_contacts = [c for c in contacts if (robot_body(c['body0']) and floor(c['body1'])) or
                      (robot_body(c['body1']) and floor(c['body0']))]
    supported = bool(floor_contacts) and all(c['position'] is not None and c['normal'] is not None
        and abs(c['position'][2] - floor_z) <= .005 and abs(c['normal'][2]) >= .5 for c in floor_contacts)
    nonfloor = [c for c in contacts if not floor(c['body0']) and not floor(c['body1'])
                and not legal_background_contact(session, c['body0'], c['body1'])]
    return {'floor_contact': supported, 'native_floor_contacts': floor_contacts,
            'nonfloor_contacts': nonfloor}


def _start_checks(session, config):
    p = _parameters(config)
    scene, robot = session.env.scene, session.robot
    position, quaternion = (_array(value) for value in robot.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all():
        raise ValueError('invalid_native_robot_pose')
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('invalid_native_robot_orientation')
    x, y, z, w = quaternion / norm
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    heading_error = abs(math.degrees(math.atan2(math.sin(yaw - p['initial_yaw_rad']),
                                               math.cos(yaw - p['initial_yaw_rad']))))
    actual_floor_z = float(scene.seg_map.floor_heights[p['floor_idx']])
    checks = {'scene_matches': session.scene_name == config['scene'] and session.room_name == 'full_scene',
              'native_r1': type(robot).__name__ == 'R1' and not robot.visual_only,
              'native_holonomic_base': type(robot.controllers.get('base')).__name__ == 'HolonomicBaseJointController',
              'native_fixed_base': bool(robot.fixed_base),
              'source_floor_matches_native_map': abs(actual_floor_z - p['floor_z']) <= .005,
              'source_start_maps_to_public_room': _region(scene, p['start_xy']) == p['start_region'],
              'source_goal_maps_to_public_room': _region(scene, p['goal_xy']) == p['target_region'],
              'robot_in_start_region': _region(scene, position[:2]) == p['start_region'],
              'start_position_error_m': float(np.linalg.norm(position[:2] - p['start_xy'])),
              'start_heading_error_deg': heading_error,
              'native_robot_position': position.tolist(), 'native_floor_z': actual_floor_z,
              **_contacts(robot, _floor_paths(scene), p['floor_z'], session)}
    checks['start_position_matches'] = checks['start_position_error_m'] <= p['start_position_tolerance_m']
    checks['start_heading_matches'] = heading_error <= p['start_heading_tolerance_deg']
    checks['no_nonfloor_contacts'] = not checks['nonfloor_contacts']
    checks['original_start_matches'] = bool(checks['source_start_maps_to_public_room']
        and checks['start_position_matches'] and checks['start_heading_matches'])
    repair = verified_repaired_start(session, p['start_region'], position, quaternion,
        position_tolerance_m=p['start_position_tolerance_m'], heading_tolerance_deg=p['start_heading_tolerance_deg'])
    checks['verified_repair_evidence'] = repair
    checks['start_contract_accepted'] = checks['original_start_matches'] or repair['accepted']
    checks['start_contract'] = ('original_exact_pose' if checks['original_start_matches'] else
        'verified_repaired_pose_same_source_region' if repair['accepted'] else 'rejected')
    required = ('scene_matches', 'native_r1', 'native_holonomic_base', 'source_floor_matches_native_map',
                'source_goal_maps_to_public_room', 'robot_in_start_region',
                'start_contract_accepted', 'floor_contact', 'no_nonfloor_contacts')
    failed = [key for key in required if not checks[key]]
    return checks, failed


def qualify_operation(session, executor, config, bound):
    if config.get('operation') == REVISED_OPERATION:
        return qualify_revised_mapping(session, executor, config, bound, qualify_operation)
    """Use the actual session robot; ActionExecutor has no native pose method."""
    try:
        checks, failed = _start_checks(session, config)
        checks['navigation_capabilities'] = bool(executor.capabilities.base_translation and executor.capabilities.base_rotation)
        if not checks['navigation_capabilities']:
            failed.append('navigation_capabilities')
        return {'accepted': not failed, 'checks': checks,
                'reason': None if not failed else 'native_navigation_start_rejected:' + ','.join(failed)}
    except (ValueError, RuntimeError, KeyError, AttributeError, IndexError, AssertionError) as error:
        return {'accepted': False, 'checks': {}, 'reason': f'unsupported_native_navigation_start:{error}'}


class LegacyNavigationEvaluator:
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = _parameters(config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.floor_paths = _floor_paths(session.env.scene)
        self.initial = None
        self.episode = 0

    def _time(self):
        return float(self.session.pipeline.og.sim.current_time)

    def begin_episode(self, observation):
        checks, failed = _start_checks(self.session, self.config)
        if failed:
            raise ValueError('native_navigation_start_rejected:' + ','.join(failed))
        self.episode += 1
        self.initial = _array(self.controller.pose()['position'])
        self.previous = self.initial.copy()
        self.previous_time = self._time()
        self.samples, self.travel_m = 0, 0.
        self.motion_faults = []
        self.visits = [self.parameters['start_region']]
        self.start_checks = checks
        return self.record(observation, action=None)

    def sample_step(self):
        if self.initial is None:
            return
        position = _array(self.controller.pose()['position'])
        now = self._time()
        dt = now - self.previous_time
        distance = float(np.linalg.norm(position[:2] - self.previous[:2]))
        native_config = self.controller.config
        limit = native_config.max_translation_speed_m_s * max(dt, 0.) + native_config.translation_tolerance_m
        if dt <= 0 or distance > limit or not np.isfinite(position).all():
            self.motion_faults.append('native_motion_clock_or_continuity_failed')
        contacts = _contacts(self.session.robot, self.floor_paths, self.parameters['floor_z'], self.session)
        if not contacts['floor_contact'] or contacts['nonfloor_contacts']:
            self.motion_faults.append('native_floor_support_or_contact_failed')
        region = _region(self.session.env.scene, position[:2])
        if region != self.visits[-1]:
            self.visits.append(region)
        self.travel_m += distance
        self.samples += 1
        self.previous, self.previous_time = position.copy(), now
        with (self.output / 'physical_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'sample': self.samples, 'simulation_time_s': now,
                                     'base_position': position.tolist(), 'native_region': region,
                                     'floor_contact': contacts['floor_contact'],
                                     'nonfloor_contacts': contacts['nonfloor_contacts']}, allow_nan=False) + '\n')

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        position = _array(self.controller.pose()['position'])
        unsampled_motion = float(np.linalg.norm(position[:2] - self.previous[:2]))
        if unsampled_motion > self.controller.config.translation_tolerance_m:
            self.motion_faults.append('unsampled_native_base_motion')
        region = _region(self.session.env.scene, position[:2])
        contacts = _contacts(self.session.robot, self.floor_paths, self.parameters['floor_z'], self.session)
        criteria = {'qualified_start_contract': bool(self.start_checks['start_contract_accepted']),
                    'positive_control_step_motion': self.samples > 0 and self.travel_m >= .05,
                    'continuous_measured_motion': not self.motion_faults,
                    'in_public_destination_region': region == self.parameters['target_region'],
                    'native_floor_supported': contacts['floor_contact'],
                    'no_nonfloor_contacts': not contacts['nonfloor_contacts']}
        row = {'schema': 'private_native_navigation_v1', 'episode': self.episode,
               'task_success': all(criteria.values()), 'criteria': criteria,
               'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
               'native_base_position': position.tolist(), 'native_region': region,
               'physical_travel_m': self.travel_m, 'control_step_samples': self.samples,
               'native_region_visits': self.visits, 'motion_faults': sorted(set(self.motion_faults)),
               'initial_qualification': self.start_checks, 'configuration': self.config}
        (self.output / 'state.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        return row


def create_evaluator(session, controller, config, output):
    if config.get('operation') == REVISED_OPERATION:
        native = create_evaluator(session, controller, legacy_mapping_config(config), output)
        return SourceAnswerMappingEvaluator(session, controller, config, output, native)
    evaluator = NavigationEvaluator if config['operation'] == OPERATION else LegacyNavigationEvaluator
    return evaluator(session, controller, config, output)


class NavigationEvaluator(NativeVisitReportEvaluator):
    """Require public room visits/reports, with order only from literal public steps."""
    def __init__(self, session, controller, config, output):
        _parameters(config)
        super().__init__(session, controller, config, output)

    def _measure_native(self):
        parameters = {**self.parameters, 'floor_contact_height_tolerance_m': .005,
            'minimum_floor_contact_vertical_normal': .5}
        measured = measure_room_state(self.session, parameters=parameters)
        return settled_native_state(self.session, measured)

    def _qualify_start(self):
        checks, failed = _start_checks(self.session, self.config)
        self.start_evidence = checks
        measured = self._measure_native()
        return {'native_original_start_qualified': not failed,
            'native_required_regions_exist': self.required.issubset(measured['known_regions']),
            'native_initial_chassis_and_pose': measured['state_valid']}


class CategoryConverter(ConverterBase):
    category = ('Cognitive Mapping', 'Long-Term Navigation')
    version = 'cognitive_mapping_long_term_navigation_task_edit_20261018'

    def _source_data(self, source):
        raw, payload = source.raw, source.payload
        q = payload['question_data']
        if (source.big_task, source.small_task) != self.category or (raw['big_task'], raw['small_task']) != self.category:
            raise ValueError('source_category_mismatch')
        match = ENDPOINTS.match(raw['question'])
        if not match or _endpoint_id(match.group('start')) != q['source_region'] or _endpoint_id(match.group('goal')) != q['target_region']:
            raise ValueError('public_private_navigation_endpoints_mismatch')
        if q['source_region'] == q['target_region'] or not _vector(q.get('start_xy'), 2) or not _vector(q.get('goal_xy'), 2):
            raise ValueError('missing_or_degenerate_original_navigation_endpoints')
        if (type(q.get('floor_idx')) is not int or q['floor_idx'] < 0 or not isinstance(q.get('floor_z'), (int, float))
                or not math.isfinite(q['floor_z']) or not math.isfinite(q.get('initial_yaw', float('nan')))):
            raise ValueError('missing_original_floor_or_heading')
        camera = q.get('render', {}).get('initial_view', q.get('initial_view', {})).get('camera_pose', {})
        if not _vector(camera.get('position'), 3) or not _vector(camera.get('quaternion_xyzw'), 4):
            raise ValueError('missing_original_reference_camera')
        return raw, payload, q, match

    def inspect(self, source):
        try:
            self._source_data(source)
        except (ValueError, KeyError, TypeError) as error:
            return {'eligible': False, 'reason': str(error), 'warning_codes': ['not_convertible_under_L1_L2']}
        return {'eligible': True, 'reason': 'Public endpoints have exact original construction bindings.'}

    def convert(self, source, context):
        result = self._convert_legacy(source, context)
        if context.config.get('mapping_revision') == 'legacy' or context.config.get('long_term_navigation_operation'):
            return result
        return revise_mapping_result(source, context, result)

    def _convert_legacy(self, source, context):
        raw, payload, q, match = self._source_data(source)
        patches = [{'operation': 'robot_pose', 'phase': 'before_construction',
                    'position': [*q['start_xy'], q['floor_z'] + INITIAL_CLEARANCE_M],
                    'yaw_deg': math.degrees(q['initial_yaw']), 'head_pitch_deg': 0.}]
        parameters = {'start_region': q['source_region'], 'target_region': q['target_region'],
                      'start_xy': q['start_xy'], 'goal_xy': q['goal_xy'], 'floor_idx': q['floor_idx'],
                      'floor_z': q['floor_z'], 'initial_yaw_rad': q['initial_yaw'],
                      'start_position_tolerance_m': .05, 'start_heading_tolerance_deg': 2.,
                      'stability_steps': 3}
        order, route_evidence = public_navigation_route(raw['question'], q['source_region'], q['target_region'])
        required_regions = order or [q['source_region'], q['target_region']]
        parameters.update(visit_parameters(required_regions, required_order=order,
            route_evidence=route_evidence, question=raw['question']))
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
                      'binding_keys': [], 'parameters': parameters}
        route_text = (' Preserve the order of the explicitly named public room steps: '
            + ' -> '.join(order) + '. Other rooms and backtracking are allowed between required steps.'
            if order else ' Visit the required rooms in any order; extra rooms and backtracking are allowed.')
        public = {'objective': raw['question'] + ' Physically visit these required rooms: '
                  + ', '.join(dict.fromkeys(required_regions)) + '.' + route_text + ' ' + ARRIVAL_INSTRUCTION,
                  'success_criteria': 'Stably occupy and explicitly report every required room with the full chassis '
                  'supported by the floor and clear of nonfloor contacts. Only explicit public room steps impose order; '
                  'missing steps are not filled from an answer. No additional return trip or terminal room is required.',
                  'environment': 'Explore the complete original building using the robot mounted sensors and bounded controls.',
                  'constraints': ['Use observed room cues and accumulated legitimate observations to navigate.',
                                  'Preserve the original floorplan, furniture, doorways and source reference frames.',
                                  'Use real bounded robot motion with all existing controller and collision guards.',
                                  'An already occupied required room may be reported after stable native control samples.']}
        return {'scene_definition': original_scene(source, context, bindings={}, patches=patches),
                'interaction_policy': {'mode': 'navigation', 'group_bindings': [], 'manipulation_bindings': [],
                    'relocation_policy': 'preserve', 'source_region': q['source_region'],
                    'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                    'maximum_stance_distance_m': .85},
                'task_definition': physical_task(source, context, public=completion_public_goal(public, self.category[1], ordered=bool(order)), evaluation=evaluation,
                    bindings={}, patches=patches, capabilities=['base_translation', 'base_rotation'],
                    qualification={'on_failure': 'skip', 'initial_visibility': 'none'}),
                'status': 'generated', 'reuse_level': 'L2',
                'change_summary': ['Preserve the original complete floorplan, native assets, source reference cameras and QA.',
                                   'Declare robot-only initialization at source start_xy/initial_yaw and source floor plus existing initialization clearance.',
                                   'Add native stable room-set occupancy and explicit Agent reports; retain only literal public room-step order.'],
                'warning_codes': ['assumption_review_required', 'native_route_unverified', 'robot_spawn_unqualified',
                                  'mounted_sensor_coverage_unverified', 'long_route_budget_unqualified',
                                  'same_source_region_spawn_repair_requires_native_evidence'],
                'assumptions': [{'doubt': 'The source planning point is not a qualified R1 footprint, and the installed '
                                'env_base.py Environment.default_config.initial_pos_z_offset=.1 m is a collision-test/drop '
                                'clearance, not a calibrated R1 origin height. Native body-root-to-floor validity is unverified.',
                                'chosen_assumption': 'Use this reversible robot-only initialization candidate and reject it '
                                'unless native start region/position/heading, floor and nonfloor contact checks pass.',
                                'alternatives': ['Retain the unpatched original variant as a source-only diagnostic.',
                                                 'Require a separately evidenced same-start-region robot stance.'],
                                'scope': 'Only robot initial position/heading; no world asset, route anchor or sensor/reference pose changes.',
                                'risk': 'A room-center planning cell or body-root-to-floor placement may collide or be '
                                'inaccessible for R1; head and other mounted views may '
                                'differ from source free-camera frames, and the unchanged decision budget may be insufficient.'}],
                'limitations': ['CPU conversion is not native initialization, route, doorway-clearance or destination qualification.',
                                'Private source path/goal_xy/answers remain construction or auxiliary QA evidence, never a route-success oracle.',
                                'Native start rejection retains a failed candidate; no runtime robot/object teleport or room shrink is allowed.',
                                'Existing episode defaults are retained; source path distances and graph steps do not certify a sufficient primitive budget.'],
                'private_navigation_evidence': {'source_start_region': q['source_region'], 'source_target_region': q['target_region'],
                    'source_task_type': q['task_type'], 'source_path_distance_m': q['path_distance'],
                    'source_region_sequence': q['region_sequence'], 'native_route_verified': False},
                'attempted_repairs': ['Reversible robot-only initialization declaration; native execution not run.'],
                'next_step': 'Qualify native start/floor/collision and destination mapping, capture legitimate mounted views, '
                             'then test continuous bounded R1 travel with a separately reviewed sufficient episode budget.'}

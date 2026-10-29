"""Convert Agent Observation into measured, embodied observation patrols.

Run on CPU from the repository root with the behavior environment's Python:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m
wia.authoring.converters.temporal_understanding_agent_observation --batch-root DIR

Native initialization restores only the existing R1's saved episode start.
Qualification/evaluation read native state and never issue robot actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.resources import ESI_ROOT

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from wia.resources import ROOT

from wia.authoring.conversion import (
    ROOT, ConversionContext, ConverterBase, SourceInstance, original_scene, physical_task,
)
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import sha256, write_json
from wia.tasks.interaction_setup import (
    array as native_array, reference_position, support_contact_paths, verified_repaired_start)

MODULE = 'wia.authoring.converters.temporal_understanding_agent_observation'
OPERATION = 'observe_objects_and_return'
RETRIEVAL_OPERATION = 'observe_retrieve_and_return'
CATEGORY = ('Temporal Understanding', 'Agent Observation')
DEPARTURE_M = 0.5
RETURN_M = 0.15
MINIMUM_TRAVEL_M = 0.8
STABILITY_STEPS = 15
POSE_TOLERANCE_M = 0.05
OBJECT_DRIFT_M = 0.03


def _completion_result(value, items, *, available=True):
    """Expose required physical obligations without averaging safety predicates."""
    completion_items = [{'id': key, 'completed': bool(done),
        'evaluation_validity': 'valid' if available else 'unavailable',
        'kind': kind, 'evidence': evidence} for key, done, kind, evidence in items]
    completed = sum(item['completed'] for item in completion_items)
    total = len(completion_items)
    score = completed / total if total and available else None
    value.update(scoring_version='task_completion_v2',
        evaluation_validity='complete' if available else 'unavailable',
        completion_items=completion_items, completed_goals=completed, total_goals=total,
        completion_score=score, score_percent=100. * score if score is not None else None)
    return value


def _vector(value, size):
    return isinstance(value, (tuple, list)) and len(value) == size and all(
        isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def _array(value):
    return native_array(value)


def _yaw(quaternion):
    q = _array(quaternion)
    if q.shape != (4,) or not np.linalg.norm(q):
        raise ValueError('Missing native or source orientation.')
    x, y, z, w = q / np.linalg.norm(q)
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def _source_data(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CATEGORY or (
            raw.get('big_task'), raw.get('small_task')) != CATEGORY:
        raise ValueError('source_category_mismatch')
    if not raw.get('scene') or not raw.get('room') or not payload.get('object_category'):
        raise ValueError('missing_scene_room_or_category')
    if not _vector(payload.get('start_xy'), 2) or not math.isfinite(payload.get('robot_start_z', math.nan)):
        raise ValueError('missing_source_episode_start')
    log = payload.get('step_log', [])
    views = payload.get('observer_poses', [])
    objects = payload.get('objects_meta', [])
    if not log or not _vector(log[0].get('robot_quat_xyzw'), 4):
        raise ValueError('missing_source_start_heading')
    _yaw(log[0]['robot_quat_xyzw'])
    if not views or any(not _vector(v.get('position'), 3) or not _vector(
            v.get('quaternion_xyzw'), 4) or not any(v['quaternion_xyzw']) for v in views):
        raise ValueError('missing_original_observer_reference')
    if not isinstance(objects, list) or not objects:
        raise ValueError('no_original_objects_for_observation_patrol')
    names = set()
    for obj in objects:
        name = obj.get('_key')
        if (not name or name in names or not obj.get('model') or
                obj.get('category') != payload['object_category'] or
                any(not _vector(obj.get(k), n) for k, n in (
                    ('position', 3), ('quaternion_xyzw', 4), ('aabb_min', 3), ('aabb_max', 3))) or
                not any(obj['quaternion_xyzw']) or any(
                    a >= b for a, b in zip(obj['aabb_min'], obj['aabb_max']))):
            raise ValueError('incomplete_original_object_binding_or_geometry')
        names.add(name)
    threshold = payload.get('proximity_thresh')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= RETURN_M:
        raise ValueError('invalid_source_proximity_threshold')
    if not _vector(payload.get('navigation', {}).get('final_xy'), 2):
        raise ValueError('missing_original_endpoint')
    if str(raw.get('answer')) != str(payload.get('true_count')):
        raise ValueError('original_count_answer_conflict')
    return raw, payload, objects


def initialize_scene(session, parameters):
    """Restore the saved actor start once, after endpoint-only source processing.

    The runtime may already bind the retained R1, but observations and guarded
    controls must not have started. No callbacks, actor replay or runtime pose
    setters are installed. The original external observer camera stays untouched.
    """
    state = vars(session)
    sources = [state.get(name) for name in ('_observation_source', 'observation_source')]
    observed = any(source is not None and (any(getattr(source, name, None) is not None for name in
        ('initial_observation', 'latest_snapshot', 'boundary_observation', 'initial_overview'))
        or getattr(source, 'history', ()) or getattr(source, 'action_step', 0)
        or getattr(source, 'snapshot_id', 0)) for source in sources)
    controllers = [state.get('controller'), state.get('task_controller'),
        getattr(state.get('task_evaluator'), 'controller', None),
        getattr(state.get('executor'), 'controller', None)]
    controllers.extend(getattr(getattr(source, 'controller_state_provider', None), '__self__', None)
                       for source in sources if source is not None)
    attached_controller = any(controller is not None and controller is not session for controller in controllers)
    acted = any(controller is not None and (getattr(controller, '_public_action_sequence', 0)
        or getattr(controller, '_public_action_context', None)) for controller in [session, *controllers])
    if state.get('_agent_observation_start_restored', False) or observed or attached_controller or acted:
        raise ValueError('Episode-start restoration is initialization-only.')
    if len(session.env.robots) != 1 or type(session.env.robots[0]).__name__ != 'R1':
        raise ValueError('Restore only the original single installed R1.')
    robot = session.env.robots[0]
    if state.get('robot') is not None and state['robot'] is not robot:
        raise ValueError('Episode-start restoration cannot use a foreign or stale bound robot.')
    payload = session.payload
    expected = [*payload['start_xy'], float(payload['robot_start_z'])]
    quaternion = payload['step_log'][0]['robot_quat_xyzw']
    if (parameters.get('restore') != 'original_episode_start' or
            (session.scene_name, session.room_name) != (parameters.get('scene'), parameters.get('room')) or
            parameters.get('position') != expected or parameters.get('quaternion_xyzw') != quaternion or
            sha256(session.question_path) != parameters.get('source_sha256')):
        raise ValueError('Original episode-start provenance mismatch.')
    import torch
    before = [_array(v).tolist() for v in robot.get_position_orientation()]
    robot.set_position_orientation(position=torch.tensor(expected, dtype=torch.float32),
                                   orientation=torch.tensor(quaternion, dtype=torch.float32))
    # Flush stale endpoint contacts and use the original builder's 30-step
    # initialization settling convention. This is not task execution/evidence.
    for _ in range(30):
        session.pipeline.og.sim.step()
    session._agent_observation_start_restored = True
    return {'source_provenance': parameters['provenance'], 'source_sha256': parameters['source_sha256'],
            'before_position_quaternion': before,
            'after_position_quaternion': [_array(v).tolist() for v in robot.get_position_orientation()],
            'settle_steps': 30, 'original_observer_camera_preserved': True,
            'actor_replay_restored': False, 'physical_success_evidence': False}


def _parameters(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') not in (OPERATION, RETRIEVAL_OPERATION) or (config.get('scene'), config.get('room')) != (
                session.scene_name, session.room_name)):
        raise ValueError('Operation must match the original native scene and room.')
    p = config['parameters']
    if (not _vector(p.get('start_xy'), 2) or not p.get('source_objects') or
            set(p['source_objects']) != set(config.get('binding_keys', [])) or
            p.get('departure_m') != DEPARTURE_M or p.get('return_m') != RETURN_M or
            p.get('minimum_travel_m') != MINIMUM_TRAVEL_M or p.get('stability_steps') != STABILITY_STEPS or
            not 0 < p.get('visit_radius_m', 0) <= 2):
        raise ValueError('Invalid physical patrol contract.')
    if config['operation'] == RETRIEVAL_OPERATION and (
            p.get('minimum_lift_m') != .05 or p.get('working_arm') != 'left' or
            p.get('lift_stability_steps') != 3):
        raise ValueError('Invalid physical retrieval contract.')
    return p


def _within(body, root):
    return body == root or body.startswith(root + '/')


def _native_contacts(obj, floor_paths, session=None):
    floor, blocked = [], []
    for contact in obj.contact_list():
        if session is not None and legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not _within(body, obj.prim_path)]
        if not external:
            continue
        row = {'body0': bodies[0], 'body1': bodies[1]}
        for key in ('position', 'normal'):
            value = _array(getattr(contact, key)) if hasattr(contact, key) else np.empty(0)
            row[key] = value.tolist() if value.shape == (3,) else None
        is_floor = [any(_within(body, path) for path in floor_paths) or
                    body.endswith('/groundPlane') or '/groundPlane/' in body for body in external]
        if all(is_floor):
            floor.append(row)
        else:
            blocked.append(row)
    supported = bool(floor) and all(row['position'] is not None and row['normal'] is not None and
                                  abs(row['normal'][2]) >= .5 for row in floor)
    return {'floor_supported': supported, 'floor_contacts': floor, 'nonfloor_contacts': blocked}


def _native_state(session, config):
    """Read measured bodies, original target geometry and floor contacts only."""
    p = _parameters(session, config)
    robot, scene = session.robot, session.env.scene
    if type(robot).__name__ != 'R1' or robot.visual_only:
        raise ValueError('An actual collidable installed R1 is required.')
    # Installed HolonomicBaseRobot deliberately has fixed_base=True. Its
    # articulated base_link moves; rejecting fixed_base would reject every R1.
    position, quaternion = (_array(v) for v in robot.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError('Native base pose unavailable.')
    _yaw(quaternion)
    q = quaternion / np.linalg.norm(quaternion)
    up_dot = float(1 - 2 * (q[0]**2 + q[1]**2))
    scene_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floor_paths = [obj.prim_path for obj in scene_objects if obj.category in ('floor', 'floors')]
    if not floor_paths:
        raise ValueError('Native floor identity unavailable.')
    contacts = _native_contacts(robot, floor_paths, session)
    measured_objects = {}
    for key, reference in p['source_objects'].items():
        obj = scene.object_registry('name', config['bindings'][key])
        if obj is None or obj.name != reference['name'] or obj.category != reference['category'] or obj.model != reference['model']:
            raise ValueError('Native original object binding mismatch: ' + key)
        point, _ = (_array(v) for v in obj.get_position_orientation())
        low, high = (_array(v) for v in obj.aabb)
        if any(v.shape != (3,) for v in (point, low, high)) or np.any(high <= low):
            raise ValueError('Native target geometry unavailable: ' + key)
        support = _native_contacts(obj, floor_paths)
        measured_objects[key] = {'name': obj.name, 'position': point.tolist(),
            'centre': ((low + high) / 2).tolist(), 'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
            'visual_only': bool(obj.visual_only), 'floor_supported': support['floor_supported'],
            'source_position_error_m': float(np.linalg.norm(point - reference['position'])),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'up_dot': up_dot,
        'linear_speed_m_s': float(np.linalg.norm(_array(robot.get_linear_velocity()))),
        'angular_speed_rad_s': float(np.linalg.norm(_array(robot.get_angular_velocity()))),
        'objects': measured_objects, **contacts}


def _initial_checks(session, config):
    retrieval = config['operation'] == RETRIEVAL_OPERATION
    measured = _retrieval_state(session, config) if retrieval else _native_state(session, config)
    p = config['parameters']
    heading_error = abs(math.atan2(math.sin(_yaw(measured['quaternion_xyzw']) - p['start_yaw_rad']),
                                  math.cos(_yaw(measured['quaternion_xyzw']) - p['start_yaw_rad'])))
    checks = {'source_start_xy': math.dist(measured['position'][:2], p['start_xy']) <= POSE_TOLERANCE_M,
        'source_start_heading': heading_error <= math.radians(3),
        'upright_robot': measured['up_dot'] >= .95, 'native_floor_support': measured['floor_supported'],
        'no_external_nonfloor_robot_contact': not measured['nonfloor_contacts'],
        'original_objects_collidable_supported_and_stable': all(not obj['visual_only'] and obj['floor_supported'] and
            obj['source_position_error_m'] <= POSE_TOLERANCE_M and obj['linear_speed_m_s'] <= .02 and
            obj['angular_speed_rad_s'] <= .1 for obj in measured['objects'].values())}
    if retrieval:
        repair = verified_repaired_start(session, session.room_name, measured['position'],
            measured['quaternion_xyzw'], position_tolerance_m=POSE_TOLERANCE_M, heading_tolerance_deg=3.)
        checks['source_start_xy'] = bool(checks['source_start_xy'] or repair['accepted'])
        checks['source_start_heading'] = bool(checks['source_start_heading'] or repair['accepted'])
        checks['original_objects_collidable_supported_and_stable'] = all(
            not obj['visual_only'] and not obj['fixed_base'] and obj['support_supported'] and
            not obj['forbidden_contacts'] and obj['accepted_reference_error_m'] <= POSE_TOLERANCE_M and
            obj['linear_speed_m_s'] <= .02 and obj['angular_speed_rad_s'] <= .1
            for obj in measured['objects'].values())
        checks['empty_native_gripper'] = bool(measured['native_grasp']['holding'] is False and
            measured['native_grasp']['object_name'] is None and not measured['native_grasp']['constraint_active'])
        measured['verified_repaired_start'] = repair
    return measured, checks


def qualify_operation(session, executor, config, bound):
    """Read preconditions; never call an executor action or change native state."""
    if config.get('operation') == 'history_multi_retrieval_v1':
        from .wia_history_collection import qualify_collection
        return qualify_collection(session, executor, config, bound)
    if config.get("operation") == "survey_history_retrieve_place":
        return qualify_history_retrieval(session, executor, config, bound)
    try:
        measured, checks = _initial_checks(session, config)
        checks['guarded_base_capabilities'] = bool(executor.capabilities.base_translation and executor.capabilities.base_rotation)
        if config['operation'] == RETRIEVAL_OPERATION:
            checks['guarded_manipulation_capabilities'] = bool(executor.capabilities.ee_translation and executor.capabilities.gripper)
        checks['native_bindings_resolved'] = set(bound) == set(config['binding_keys']) and all(
            obj is session.env.scene.object_registry('name', config['bindings'][key]) for key, obj in bound.items())
        failed = [key for key, value in checks.items() if not value]
        return {'accepted': not failed, 'checks': checks, 'native_state': measured,
                'reason': 'native_patrol_precondition_failed:' + ','.join(failed) if failed else None,
                'route_execution_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': {}, 'reason': 'native_patrol_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}'}


class ObservationPatrolEvaluator:
    """Measure actual visits and a subsequent safe return, without moving bodies."""
    def __init__(self, session, controller, config, output):
        self.parameters = _parameters(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial = None
        self.episode = 0

    def _time(self):
        return float(self.session.pipeline.og.sim.current_time)

    def _measure(self):
        return _native_state(self.session, self.config)

    def _record_details(self, value):
        return value

    def begin_episode(self, observation):
        measured, checks = _initial_checks(self.session, self.config)
        if not all(checks.values()):
            raise ValueError('Native patrol initialization failed: ' + ','.join(k for k, v in checks.items() if not v))
        self.episode += 1
        self.initial = measured
        self.start = np.array(measured['position'])
        self.previous = self.start.copy()
        self.previous_yaw = _yaw(measured['quaternion_xyzw'])
        self.previous_time = self._time()
        self.initial_checks = checks
        self.samples, self.travel_m, self.maximum_departure_m = 0, 0., 0.
        self.visits, self.faults = {}, set()
        self.visit_streaks = {key: 0 for key in measured['objects']}
        self.sample_motion_stopped = True
        self.stable_steps = 0
        return self.record(observation, action=None)

    def _safe(self, measured):
        preserved = all(math.dist(obj['position'], self.initial['objects'][key]['position']) <= OBJECT_DRIFT_M and
                        obj['floor_supported'] and not obj['visual_only'] and obj['linear_speed_m_s'] <= .02 and
                        obj['angular_speed_rad_s'] <= .1 for key, obj in measured['objects'].items())
        return bool(measured['floor_supported'] and not measured['nonfloor_contacts'] and
                    measured['up_dot'] >= .95 and preserved)

    def _terminal(self, measured):
        return {'all_original_objects_visited': len(self.visits) == len(self.initial['objects']),
            'actual_departure': self.maximum_departure_m >= DEPARTURE_M,
            'actual_travel': self.travel_m >= MINIMUM_TRAVEL_M,
            'returned_to_measured_start': math.dist(measured['position'][:2], self.start[:2]) <= RETURN_M,
            'measured_control_steps': self.samples > 0, 'continuous_safe_episode': not self.faults,
            'currently_safe': self._safe(measured),
            'stopped': self.sample_motion_stopped and measured['linear_speed_m_s'] <= .02 and measured['angular_speed_rad_s'] <= .1}

    def sample_step(self):
        if self.initial is None:
            return
        measured = self._measure()
        point, now = np.asarray(measured['position']), self._time()
        dt = now - self.previous_time
        displacement = float(np.linalg.norm(point - self.previous))
        yaw = _yaw(measured['quaternion_xyzw'])
        yaw_step = abs(math.atan2(math.sin(yaw - self.previous_yaw), math.cos(yaw - self.previous_yaw)))
        # Bound measured per-step motion by the existing controller contract.
        limit = self.controller.config.max_translation_speed_m_s * self.controller.dt + self.controller.config.translation_tolerance_m
        yaw_limit = math.radians(self.controller.config.max_rotation_speed_deg_s * self.controller.dt +
                                 self.controller.config.rotation_tolerance_deg)
        if dt <= 0 or dt > self.controller.dt * 1.5 or displacement > limit or yaw_step > yaw_limit:
            self.faults.add('unsampled_or_discontinuous_native_motion')
        self.sample_motion_stopped = bool(dt > 0 and displacement / dt <= .02 and yaw_step / dt <= .1)
        if not self._safe(measured):
            self.faults.add('native_contact_support_upright_or_object_preservation_failed')
        self.samples += 1
        self.travel_m += float(np.linalg.norm(point[:2] - self.previous[:2]))
        self.maximum_departure_m = max(self.maximum_departure_m, float(np.linalg.norm(point[:2] - self.start[:2])))
        if not self.faults and self.travel_m >= .1:
            for key, obj in measured['objects'].items():
                distance = math.dist(point[:2], obj['centre'][:2])
                self.visit_streaks[key] = self.visit_streaks[key] + 1 if distance <= self.parameters['visit_radius_m'] else 0
                if self.visit_streaks[key] >= 3 and key not in self.visits:
                    self.visits[key] = {'sample': self.samples, 'simulation_time_s': now,
                        'native_distance_m': distance, 'base_position': point.tolist(), 'object_centre': obj['centre']}
        criteria = self._terminal(measured)
        self.stable_steps = self.stable_steps + 1 if all(criteria.values()) else 0
        self.previous, self.previous_time = point.copy(), now
        self.previous_yaw = yaw
        with (self.output / 'physical_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'episode': self.episode, 'sample': self.samples, 'simulation_time_s': now,
                'native_state': measured, 'faults': sorted(self.faults)}, allow_nan=False) + '\n')

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        measured = self._measure()
        yaw_error = _yaw(measured['quaternion_xyzw']) - self.previous_yaw
        if math.dist(measured['position'], self.previous) > 1e-6 or abs(
                math.atan2(math.sin(yaw_error), math.cos(yaw_error))) > 1e-6:
            self.faults.add('native_motion_outside_control_step_sampling')
        if not self._safe(measured):
            self.faults.add('native_contact_support_upright_or_object_preservation_failed')
        criteria = self._terminal(measured)
        if not all(criteria.values()):
            self.stable_steps = 0
        value = {'schema': 'private_agent_observation_patrol_v1', 'episode': self.episode,
            'task_success': bool(all(criteria.values()) and self.stable_steps >= STABILITY_STEPS),
            'criteria': criteria, 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'native_state': measured, 'visits': self.visits,
            'physical_travel_m': self.travel_m, 'maximum_departure_m': self.maximum_departure_m,
            'control_samples': self.samples, 'stable_steps': self.stable_steps,
            'required_stability_steps': STABILITY_STEPS, 'faults': sorted(self.faults),
            'initial_qualification': self.initial_checks, 'configuration': self.config}
        items = [(f'{key}:geometric_visit', key in self.visits, 'historical_checkpoint', self.visits.get(key))
                 for key in sorted(self.initial['objects'])]
        if self.config.get('operation') == RETRIEVAL_OPERATION:
            items.append(('retrieve_at_least_one', bool(self.retrieved), 'historical_checkpoint', self.retrieved))
        items.append(('return_and_park', value['task_success'], 'terminal_condition',
            {'stable_steps': self.stable_steps, 'physical_travel_m': self.travel_m}))
        _completion_result(value, items)
        value['scored_capabilities'] = ['geometric_object_visits', 'safe_return'] + (
            ['native_grasp_and_lift'] if self.config.get('operation') == RETRIEVAL_OPERATION else [])
        value['visual_discovery_verified'] = False
        value = self._record_details(value)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def _retrieval_state(session, config):
    """Read actual original identities, grasp constraints and accepted supports."""
    from wia.authoring.converters.specular_reflection_reflection_authoring import _contacts, _native_grasp, _under
    measured = _native_state(session, config)
    robot, scene = session.robot, session.env.scene
    arm = config['parameters']['working_arm']
    objects = {key: scene.object_registry('name', name) for key, name in config['bindings'].items()}
    scene_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj.prim_path for obj in scene_objects if obj.category in ('floor', 'floors')]
    gripper_paths = [link.prim_path for link in list(robot.finger_links[arm]) + [robot.eef_links[arm]]]
    prop_paths = [obj.prim_path for obj in objects.values()]
    floor_contacts, forbidden, _ = _contacts(robot, floors,
        allowed_contact=lambda owned, external: all(any(_under(body, path) for path in gripper_paths) for body in owned)
            and all(any(_under(body, path) for path in prop_paths) for body in external)
            or all(legal_background_contact(session, a, b) for a in owned for b in external))
    measured.update(native_grasp=_native_grasp(robot, arm), floor_contacts=floor_contacts,
        floor_supported=bool(floor_contacts and all(abs(row['normal'][2]) >= .5 for row in floor_contacts)),
        nonfloor_contacts=forbidden)
    for key, obj in objects.items():
        support_paths = list(support_contact_paths(session, obj.name)) or floors
        contacts, blocked, gripper = _contacts(obj, support_paths,
            allowed_contact=lambda owned, external: all(any(_under(body, path) for path in gripper_paths) for body in external))
        reference = reference_position(session, obj.name, config['parameters']['source_objects'][key]['position'], root=True)
        measured['objects'][key].update(fixed_base=bool(obj.fixed_base),
            scale=_array(obj.scale).tolist(), support_supported=bool(contacts and all(abs(row['normal'][2]) >= .5 for row in contacts)),
            support_contacts=contacts, forbidden_contacts=blocked, gripper_contacts=gripper,
            accepted_reference_position=reference.tolist(),
            accepted_reference_error_m=float(np.linalg.norm(_array(measured['objects'][key]['position']) - reference)))
    return measured


class ObservationRetrievalEvaluator(ObservationPatrolEvaluator):
    """Keep real patrol evidence and require a physically grasped original prop."""
    def _measure(self):
        return _retrieval_state(self.session, self.config)

    def begin_episode(self, observation):
        measured = self._measure()
        self.grasped, self.retrieved, self.lift_streaks = set(), {}, {key: 0 for key in measured['objects']}
        self.object_previous = {key: _array(obj['position']) for key, obj in measured['objects'].items()}
        self.object_previous_speed = {key: obj['linear_speed_m_s'] for key, obj in measured['objects'].items()}
        self.object_time = self._time()
        return super().begin_episode(observation)

    def _held_key(self, measured):
        grasp = measured['native_grasp']
        if grasp['holding'] is not True or not grasp['constraint_active']:
            return None
        return next((key for key, obj in measured['objects'].items() if obj['name'] == grasp['object_name']), None)

    def _safe(self, measured):
        held = self._held_key(measured)
        preserved = True
        for key, obj in measured['objects'].items():
            original = self.initial['objects'][key]
            moved_legitimately = key == held or key in self.grasped
            preserved &= bool(not obj['visual_only'] and not obj['fixed_base'] and not obj['forbidden_contacts']
                and obj['scale'] == original['scale']
                and (moved_legitimately or math.dist(obj['position'], original['position']) <= OBJECT_DRIFT_M)
                and (key == held or (obj['support_supported'] and obj['linear_speed_m_s'] <= .02 and obj['angular_speed_rad_s'] <= .1)))
        return bool(measured['floor_supported'] and not measured['nonfloor_contacts'] and measured['up_dot'] >= .95 and preserved)

    def _terminal(self, measured):
        criteria = super()._terminal(measured)
        criteria['encountered_original_prop_grasped_and_lifted'] = bool(self.retrieved)
        held = self._held_key(measured)
        criteria['held_prop_stable_if_any'] = held is None or (
            measured['objects'][held]['linear_speed_m_s'] <= .02 and measured['objects'][held]['angular_speed_rad_s'] <= .1)
        return criteria

    def sample_step(self):
        if self.initial is None:
            return
        measured, now = self._measure(), self._time()
        elapsed, held = now - self.object_time, self._held_key(measured)
        for key, obj in measured['objects'].items():
            distance = float(np.linalg.norm(_array(obj['position']) - self.object_previous[key]))
            limit = max(obj['linear_speed_m_s'], self.object_previous_speed[key]) * max(elapsed, 0.) + OBJECT_DRIFT_M
            if elapsed <= 0 or distance > limit:
                self.faults.add('unsampled_or_discontinuous_original_prop_motion')
            # A native held identity is accepted only at this actual control
            # sample. It cannot retroactively authorize earlier moved props.
            if key == held and not self.faults:
                self.grasped.add(key)
            initial_low = self.initial['objects'][key]['aabb_min'][2]
            lifted = bool(key == held and not obj['support_contacts'] and
                obj['aabb_min'][2] - initial_low >= .05 - 1e-9 and
                obj['linear_speed_m_s'] <= .02 and obj['angular_speed_rad_s'] <= .1)
            self.lift_streaks[key] = self.lift_streaks[key] + 1 if lifted and not self.faults else 0
        super().sample_step()
        for key, count in self.lift_streaks.items():
            if count >= 3 and key in self.visits and not self.faults:
                self.retrieved.setdefault(key, {'sample': self.samples, 'simulation_time_s': now,
                    'native_name': measured['objects'][key]['name'],
                    'lowest_point_lift_m': measured['objects'][key]['aabb_min'][2] - self.initial['objects'][key]['aabb_min'][2],
                    'native_grasp': measured['native_grasp']})
        self.object_previous = {key: _array(obj['position']) for key, obj in measured['objects'].items()}
        self.object_previous_speed = {key: obj['linear_speed_m_s'] for key, obj in measured['objects'].items()}
        self.object_time = now

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        measured = self._measure()
        if any(np.linalg.norm(_array(obj['position']) - self.object_previous[key]) > 1e-6
                for key, obj in measured['objects'].items()):
            self.faults.add('original_prop_motion_outside_control_step_sampling')
        return super().record(observation, action=action)

    def _record_details(self, value):
        value.update(schema='private_agent_observation_retrieval_v1', retrieved=self.retrieved,
            physically_grasped_original_bindings=sorted(self.grasped), native_visit_count=len(self.visits),
            native_retrieval_count=len(self.retrieved), bag_or_containment_implemented=False)
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == 'history_multi_retrieval_v1':
        from .wia_history_collection import HistoryCollectionEvaluator
        return HistoryCollectionEvaluator(session, controller, config, output)
    if config.get('operation') == 'survey_history_retrieve_place':
        return HistoryRetrievalEvaluator(session, controller, config, output)
    if config.get('operation') == RETRIEVAL_OPERATION:
        return ObservationRetrievalEvaluator(session, controller, config, output)
    return ObservationPatrolEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = CATEGORY
    version = '3.1.0'

    def inspect(self, source):
        try:
            _source_data(source)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_structure_review_required']}
        return {'eligible': True, 'reason': 'Original bindings and episode-start provenance are available.'}

    def convert(self, source, context):
        if context.config.get('task_edit2_temporal') is True:
            return build_history_retrieval(source, context)
        if context.config.get('legacy_agent_observation_patrol'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import visual_discovery_task
        raw, payload, objects = _source_data(source)
        bindings = {obj['_key']: {'name': obj['_key'], 'category': obj['category'], 'model': obj['model']}
                    for obj in objects}
        category = payload['object_category'].replace('_', ' ')
        public = {'objective': f'Find all {len(objects)} distinct {category} objects in this scene using your own mounted camera observations.',
            'success_criteria': 'Every distinct target must appear in a delivered robot camera observation; '
                + 'repeated sightings count once and proximity without visible identity does not count.',
            'environment': f'The original scene contains {len(objects)} {category} objects at their source locations.',
            'constraints': ['Use the named category and visible appearance to track individual objects across views.',
                'Keep original target placement, scales and reference geometry. No return trip or pickup is required.',
                "This task measures your current visual search; it does not score another actor's historical observations."]}
        position = [float(value) for value in [*payload['start_xy'], payload['robot_start_z']]]
        quaternion = payload['step_log'][0]['robot_quat_xyzw']
        provenance = {'source_json': source.relative_path, 'position': '/start_xy + /robot_start_z',
            'orientation': '/step_log/0/robot_quat_xyzw'}
        patches = [{'operation': 'robot_pose', 'phase': 'before_construction', 'position': position,
            'yaw_deg': math.degrees(_yaw(quaternion)), 'head_pitch_deg': 0.},
            {'operation': 'category_initialization', 'phase': 'after_postprocessing', 'module': MODULE,
             'parameters': {'restore': 'original_episode_start', 'scene': raw['scene'], 'room': raw['room'],
                'position': position, 'quaternion_xyzw': quaternion, 'source_sha256': sha256(source.path),
                'provenance': provenance}}]
        result = visual_discovery_task(source, context, bindings=bindings, target_keys=list(bindings),
            public=public, patches=patches,
            preserved_relations=['original object positions and scales', 'source observer poses and actor history as provenance'],
            source_evidence={'source_objects': objects, 'source_observer_poses': payload['observer_poses'],
                'historical_answer': raw['answer'], 'original_actor_replay_restored': False})
        result['measured_capability'] = 'all_target_visual_search'
        result['capability_coverage_gaps'] = ['historical_actor_perspective_reasoning_not_implemented']
        result['warning_codes'] += ['source_observer_QA_provenance_only', 'native_fallen_target_check_required']
        return result

    def convert_legacy(self, source, context):
        raw, payload, objects = _source_data(source)
        retrieval = context.config.get('agent_observation_operation', RETRIEVAL_OPERATION) == RETRIEVAL_OPERATION
        bindings = {obj['_key']: {'name': obj['_key'], 'category': obj['category'], 'model': obj['model']} for obj in objects}
        warnings = ['assumption_review_required', 'native_patrol_unverified', 'original_actor_replay_not_restored']
        failed_placements = [obj['_key'] for obj in objects if obj.get('placed_on') != 'floor']
        if failed_placements:
            warnings.append('source_placement_failure_annotation')
        if payload['navigation'].get('forced_to_goal'):
            warnings.append('source_forced_endpoint_not_execution_evidence')
        below_floor = [obj['_key'] for obj in objects if obj['aabb_max'][2] < min(0., payload['robot_start_z']) - 2.]
        result = {'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': None,
            'status': 'conversion_failed', 'reuse_level': 'L1', 'warning_codes': warnings,
            'assumptions': [{'doubt': 'The original task describes another actor\'s past walk, while the active loader has one R1 and no replay.',
                'chosen_assumption': 'Add a separate embodied observation patrol over the original props, preserving original QA privately. '
                    'The controlled R1 observes its own actual visit history; no historical third-person frames are supplied to it.',
                'alternatives': ['Keep the empty original variant as source-only evidence.',
                    'Restore a separately controlled original actor and legitimate independent observer history in a future mechanism implementation.'],
                'scope': 'Whole category; physical patrol evaluation only.',
                'risk': 'Patrol completion does not validate the original historical count, original actor route or independent observer mechanism.'}],
            'private_source_evidence': {'source_sha256': sha256(source.path), 'observer_count': len(payload['observer_poses']),
                'object_count': len(objects), 'failed_placement_bindings': failed_placements,
                'implausibly_below_floor_bindings': below_floor,
                'source_forced_to_goal': payload['navigation'].get('forced_to_goal'),
                'original_actor_replay_restored': False, 'native_execution_verified': False},
            'limitations': ['CPU conversion and source navigation/placement flags do not qualify native physical execution.',
                'Original third-person actor replay is not implemented; the new task uses its own legitimate patrol history.',
                'Source poses, trajectory, observer images, answer and scorer state remain private.',
                'The physical predicate measures proximity visits; visual recognition of every object is auxiliary and unverified.',
                'Mounted visibility, whole-route clearance, runtime cost and task success rate remain unmeasured.'],
            'attempted_repairs': [], 'next_step': 'Inspect and qualify original/adapted native state, then execute the guarded patrol.'}
        if below_floor and not retrieval:
            result.update(scene_definition=None, failure_stage='source_geometry',
                failure_reason='not_convertible_under_L1_L2:original_objects_far_below_floor',
                diagnostic_only=True, change_summary=['Preserve original scene for source-only diagnostic Review; do not reposition fallen props.'],
                next_step='Review saved object heights and original support evidence; no supported L2 height reconstruction is currently known.')
            warnings.append('not_convertible_under_L1_L2')
            return result
        if below_floor:
            warnings.extend(['original_bad_height_pending_native_support_repair', 'original_pose_and_qa_reference_changed'])
            result['attempted_repairs'].append('Request measured independent support placement for finite original bad-height props without guessing coordinates.')
        position = [float(v) for v in [*payload['start_xy'], payload['robot_start_z']]]
        quaternion = payload['step_log'][0]['robot_quat_xyzw']
        yaw = _yaw(quaternion)
        provenance = {'source_json': source.relative_path, 'position': '/start_xy + /robot_start_z',
            'orientation': '/step_log/0/robot_quat_xyzw',
            'source_postprocess': 'agent_observation.postprocess_env: single R1 at navigation.final_xy'}
        patches = [{'operation': 'robot_pose', 'phase': 'before_construction', 'position': position,
                    'yaw_deg': math.degrees(yaw), 'head_pitch_deg': 0.},
            {'operation': 'category_initialization', 'phase': 'after_postprocessing', 'module': MODULE,
             'parameters': {'restore': 'original_episode_start', 'scene': raw['scene'], 'room': raw['room'],
                'position': position, 'quaternion_xyzw': quaternion, 'source_sha256': sha256(source.path), 'provenance': provenance}}]
        parameters = {'start_xy': [float(v) for v in payload['start_xy']], 'start_yaw_rad': yaw,
            'visit_radius_m': float(payload['proximity_thresh']), 'departure_m': DEPARTURE_M, 'return_m': RETURN_M,
            'minimum_travel_m': MINIMUM_TRAVEL_M, 'stability_steps': STABILITY_STEPS,
            'source_objects': {obj['_key']: {k: obj[k] for k in ('category', 'model', 'position')} |
                {'name': obj['_key']} for obj in objects}}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
                      'binding_keys': list(bindings), 'parameters': parameters}
        category = payload['object_category'].replace('_', ' ')
        public = {'objective': f'Make a geometric access patrol of the {category} task objects in this room. '
            'Approach every such object, remember which ones you have passed, then return to your starting place and stop.',
            'success_criteria': f'Pass within {payload["proximity_thresh"]:.2f} m horizontally of each task object centre using real base motion. '
                f'Leave your starting position by at least {DEPARTURE_M:.2f} m, travel at least {MINIMUM_TRAVEL_M:.2f} m in total, '
                f'and finish within {RETURN_M:.2f} m of where you began, stopped and stable. '
                'Keep the original objects in place and retain floor support without chassis contact with furniture, objects or walls.',
            'environment': 'The original room, original task-category props and original visual layout are retained.',
            'constraints': ['Use only robot mounted observations, your legitimate current episode history and existing guarded controls.',
                'Preserve task props and their supports. Native movable noncritical furniture may be moved using guarded arm/gripper actions; never move props to manufacture a visit.',
                'Use observations to identify task objects. Scoring confirms geometric visits, not visual discovery or counting.']}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches,
            capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        task['defaults']['max_steps'] = 300
        if retrieval:
            evaluation['operation'] = RETRIEVAL_OPERATION
            parameters.update(minimum_lift_m=.05, working_arm='left', lift_stability_steps=3)
            public['objective'] = f'Explore the {category} task objects in this room, visit every original task object, '
            public['objective'] += 'pick up at least one encountered object, and return safely to your starting place.'
            public['success_criteria'] += ' Pick up at least one original task object and hold it stably at least 0.05 m '
            public['success_criteria'] += 'above its initial lowest-point height. You may keep holding it when you return.'
            public['success_criteria'] = public['success_criteria'].replace(
                'Keep the original objects in place and retain floor support without chassis contact with furniture, objects or walls.',
                'Keep unpicked objects in place and retain floor support without chassis contact with furniture or walls. '
                'Only guarded gripper contact may acquire task objects.')
            public['constraints'][1] = 'Keep unpicked task objects and their supports fixed. Move picked objects and native movable noncritical furniture only with guarded physical grasping.'
            public['environment'] = 'The original room and task-category assets are retained; native support placement may adapt the task group.'
            task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches,
                capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper'],
                qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': True})
            task['defaults']['max_steps'] = 300
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
            task_definition=task, status='generated', reuse_level='L2', operation_recipe=OPERATION,
            change_summary=['Retain original assets, dimensions, prop poses, reference cameras and private QA.',
                'Restore only the existing R1 to the exact saved episode start during scene initialization.',
                'Add a real object-visit and return patrol with private native measured temporal evidence.'],
            attempted_repairs=['Declared reversible saved-start restoration after source endpoint postprocessing.'])
        if retrieval:
            result.update(operation_recipe=RETRIEVAL_OPERATION,
                interaction_policy={'mode': 'manipulation', 'group_bindings': list(bindings),
                    'manipulation_bindings': list(bindings), 'relocation_policy': 'independent_groups',
                    'placement_groups': [[key] for key in bindings],
                    'support_search': 'measured_surfaces', 'allow_fallback_table': True,
                    'source_camera_policy': 'preserve', 'source_region': raw['room'],
                    'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85,
                    'warning_codes': ['native_pickup_and_patrol_unverified', 'original_observer_replay_not_restored']})
            result['limitations'] = [line for line in result['limitations'] if 'physical predicate measures proximity' not in line]
            result['limitations'].append('Actual native grasp identity and sampled 5 cm lowest-AABB lift are required; no bag, containment or count-report mechanism is claimed.')
            result['change_summary'].append('Add actual pickup of at least one encountered original prop and accepted adapted-reference support checks.')
            result['change_summary'].append('Place each portable task prop independently on an accessible measured support when needed; preserve asset dimensions.')
            result['assumptions'].append({'doubt': 'Original floor placement or a dispersed rigid group can leave otherwise portable props unreachable.',
                'chosen_assumption': 'Use each original prop as its own placement group, prefer existing measured supports and allow the shared fallback support. '
                    'Measure visits and pickup against the accepted initialized arrangement; preserve original historical QA separately.',
                'alternatives': ['Keep a source placement only when the actual native height and guarded reach are suitable.',
                    'Retain a diagnostic when unchanged dimensions cannot fit a qualified support.'],
                'scope': 'New retrieval operations only; task supports, room structure and original assets/scales are preserved.',
                'risk': 'Accessible placement changes the adapted spatial context and does not reproduce the original actor history or saved count geometry.'})
            warnings.append('native_pickup_and_patrol_unverified')
            warnings.append('accessible_placement_changes_original_actor_context')
        result['assumptions'].append({'doubt': 'The saved actor start and source floor placements are not qualified R1 patrol geometry.',
            'chosen_assumption': 'Keep the source start XY, robot_start_z and first logged orientation as a reversible L2 proposal. '
                'Require native pose agreement, actual floor contacts, stable collidable props and all existing guards.',
            'alternatives': ['Retain the empty original endpoint-only variant.',
                'Use a separately measured same-room stance after a future evidence-based rule revision.'],
            'scope': 'Only existing R1 initialization and an uncalibrated 300-decision patrol budget; prop physics are unchanged.',
            'risk': 'Source placements may collide or drift; a half-metre excursion or complete patrol may be inaccessible. '
                'A failure annotation alone is not treated as proof; measured qualification can still reject any candidate.'})
        warnings.extend(['robot_start_unqualified', 'navigation_budget_unqualified'])
        return result

    def sample(self, manifest, policy=None):
        policy = policy or {}
        count = min(int(policy.get('count', 5)), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        selected = []
        prior = set(policy.get('previous_failures', []))
        for row in rows:
            if row['candidate_id'] in prior and len(selected) < count:
                selected.append(row)
        # Reserve a diagnostic, a source forced-endpoint annotation and a
        # placement-failure annotation when present, then diversify geometry.
        for predicate in (lambda r: r['status'] != 'generated',
                lambda r: r['status'] == 'generated' and r.get('private_source_evidence', {}).get('source_forced_to_goal'),
                lambda r: r['status'] == 'generated' and r.get('private_source_evidence', {}).get('failed_placement_bindings')):
            match = next((row for row in rows if predicate(row) and row not in selected), None)
            if match and len(selected) < count:
                selected.append(match)
        while len(selected) < count:
            seen_scenes = {row.get('scene_model') for row in selected}
            seen_counts = {row.get('private_source_evidence', {}).get('object_count') for row in selected}
            choices = [row for row in rows if row not in selected]
            selected.append(max(choices, key=lambda row: (
                row.get('private_source_evidence', {}).get('object_count') not in seen_counts,
                row.get('scene_model') not in seen_scenes, row['status'] == 'generated')))
        return [row['candidate_id'] for row in selected]

    def convert_all(self, sources, context):
        manifest = super().convert_all(sorted(sources, key=lambda source: source.relative_path), context)
        manifest['failure_distribution'] = dict(Counter(row['failure_reason'] for row in manifest['records'] if row['status'] != 'generated'))
        manifest['sample_ids'] = self.sample(manifest, {'count': 5, 'seed': context.seed})
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=ESI_ROOT / 'dataset/json_clean')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    paths = sorted(args.corpus.joinpath(*CATEGORY).rglob('*.json'))
    sources = [SourceInstance(path, str(path.relative_to(args.corpus)), *CATEGORY) for path in paths]
    manifest = CategoryConverter().convert_all(sources, ConversionContext(args.batch_root, args.seed))
    print(json.dumps({key: manifest[key] for key in ('source_count', 'counts', 'failure_distribution', 'sample_ids')}))


if __name__ == '__main__':
    main()


HISTORY_OPERATION = 'survey_history_retrieve_place'
HISTORY_INVENTORY_REVISION = 'task_edit2_core_history_three_targets_v1'
HISTORY_ADDED_ASSETS = {('jar_of_jam', 'vxjbjf'), ('box_of_sanitary_napkins', 'ykpeja')}


def history_target_inventory(source, config):
    """Copy original targets and validate the two explicitly approved additions.

    No source JSON, donor geometry or legacy prototype inventory is rewritten.
    Donor measurements establish geometry provenance, never task feasibility.
    """
    _, _, originals = _source_data(source)
    if config.get('history_inventory_revision') != HISTORY_INVENTORY_REVISION:
        return deepcopy(originals)
    additions = config.get('history_additional_targets')
    if len(originals) != 1 or not isinstance(additions, list) or len(additions) != 2:
        raise ValueError('Core history redesign requires one original and exactly two explicit additions.')
    plan = config.get('history_apparatus', {})
    labels = config.get('history_target_labels', {})
    objects = deepcopy(originals) + deepcopy(additions)
    names, identities = set(), set()
    apparatus = apparatus_objects(plan)
    native = {}
    for row in apparatus:
        if row['name'] in native:
            raise ValueError('History apparatus names must be unique.')
        native[row['name']] = row
    if any(obj['_key'] in native for obj in originals):
        raise ValueError('History apparatus cannot replace an original target asset.')
    if {(obj.get('category'),obj.get('model')) for obj in additions} != HISTORY_ADDED_ASSETS:
        raise ValueError('Core history additions must use the two explicitly approved asset identities.')
    for obj in objects:
        name = obj.get('_key')
        identity = (obj.get('category'),obj.get('model'))
        if (not isinstance(name,str) or not name.strip() or name in names or identity in identities
                or not all(isinstance(value,str) and value for value in identity)
                or any(not _vector(obj.get(key),size) for key,size in (
                    ('position',3),('quaternion_xyzw',4),('aabb_min',3),('aabb_max',3)))
                or not any(obj['quaternion_xyzw']) or np.any(np.array(obj['aabb_max']) <= obj['aabb_min'])):
            raise ValueError('History target identities and explicit geometry must be complete and distinct.')
        names.add(name); identities.add(identity)
        position = plan.get('source_target_positions', {}).get(name)
        support = plan.get('target_support_bindings', {}).get(name)
        if (not _vector(position,3) or not isinstance(labels.get(name),str) or not labels[name].strip()
                or support not in native or support in {row.get('_key') for row in objects}):
            raise ValueError('Every history target needs explicit initialized position, label and declared support.')
        support_row = native[support]
        if (not _vector(support_row.get('position'),3) or not _vector(support_row.get('scale'),3)
                or any(value <= 0 for value in support_row['scale'])
                or support_row.get('fixed_base') is not True or support_row.get('visual_only') is not False):
            raise ValueError('History target supports require explicit fixed collidable apparatus geometry.')
    for obj in additions:
        name = obj['_key']; row = native.get(name, {})
        expected = {'type':'DatasetObject','name':name,'category':obj['category'],'model':obj['model'],
            'position':obj['position'],'orientation':obj['quaternion_xyzw'],
            'scale':[1.,1.,1.],'fixed_base':False,'visual_only':False}
        if (obj.get('scale') != [1.,1.,1.] or any(row.get(key) != value for key,value in expected.items())
                or row.get('fixed_base') is not False or row.get('visual_only') is not False
                or plan['source_target_positions'][name] != obj['position']):
            raise ValueError('An added target must exactly match its dynamic collidable DatasetObject declaration.')
        provenance = obj.get('asset_provenance')
        if not isinstance(provenance,dict):
            raise ValueError('An added history asset requires explicit donor measurement provenance.')
        try:
            donor = Path(provenance['source_file'])
            if not donor.is_absolute():
                donor = ROOT / donor
            if not donor.is_absolute() or not donor.is_file() or sha256(donor) != provenance['source_sha256']:
                raise ValueError('Donor native geometry receipt file/hash mismatch.')
            receipt = json.loads(donor.read_text())
            def resolve(path):
                value = receipt
                for key in path.split('.'):
                    value = value[key]
                return value
            donor_name = provenance['source_object_name']
            if (provenance['geometry_json_path'] != 'interaction_setup.original_geometry.' + donor_name
                    or provenance['native_identity_json_path'] != 'objects.' + donor_name):
                raise ValueError('Donor paths must refer to one exact native source object.')
            geometry = resolve(provenance['geometry_json_path'])
            identity = resolve(provenance['native_identity_json_path'])
            if (identity['category'] != obj['category'] or provenance['native_category'] != obj['category']
                    or identity['scale'] != [1.,1.,1.] or provenance['native_scale'] != [1.,1.,1.]
                    or identity.get('model',obj['model']) != obj['model']):
                raise ValueError('Donor native category/scale or declared model identity mismatch.')
            low, high, point = (np.asarray(geometry[key],dtype=float) for key in
                ('aabb_min_m','aabb_max_m','position_m'))
            offset = point - low
            if (not np.allclose(np.asarray(obj['aabb_max'])-obj['aabb_min'],high-low,atol=1e-8,rtol=0)
                    or not np.allclose(np.asarray(obj['position'])-obj['aabb_min'],offset,atol=1e-8,rtol=0)
                    or not np.allclose(provenance['root_offset_from_aabb_min_m'],offset,atol=1e-8,rtol=0)
                    or obj['quaternion_xyzw'] != geometry['quaternion_xyzw']):
                raise ValueError('Added target must preserve measured donor dimensions, root offset and orientation.')
        except (KeyError,TypeError,OSError,json.JSONDecodeError) as error:
            raise ValueError('Added history asset donor provenance is incomplete: ' + str(error)) from error
    return objects


def apparatus_objects(plan):
    """Translate declared table/tray proposals into existing primitive constructors."""
    objects = []
    for item in plan.get('primitives', []):
        objects.append({'type': 'PrimitiveObject', 'name': item['name'],
            'category': item.get('category', 'breakfast_table'), 'primitive_type': 'Cube',
            'size': 1., 'scale': item['dimensions_m'], 'position': item['position'],
            'orientation': [0., 0., 0., 1.], 'fixed_base': True, 'visual_only': False,
            'rgba': item.get('rgba', [.45, .35, .22, 1.])})
    objects.extend(plan.get('dataset_objects', []))
    return objects


def configure_scene(config, parameters):
    """Create only the explicit versioned apparatus, before native construction."""
    from wia.tasks.temporal_policy import TASK_EDIT2_TEMPORAL_REVISION
    if parameters.get('task_revision') != TASK_EDIT2_TEMPORAL_REVISION:
        raise ValueError('History apparatus requires the explicit TaskEdit2 revision.')
    plan = parameters['apparatus']
    if config['scene']['scene_model'] != parameters['scene_model']:
        raise ValueError('Temporal apparatus belongs to a different world.')
    names = {row['name'] for row in config.get('objects', [])}
    additions = apparatus_objects(plan)
    if names.intersection(row['name'] for row in additions):
        raise ValueError('Temporal apparatus cannot overwrite existing source assets.')
    config.setdefault('objects', []).extend(additions)
    for name, position in plan.get('source_target_positions', {}).items():
        obj = next((x for x in config['objects'] if x['name'] == name), None)
        if obj is None:
            raise ValueError('A history target is absent from the source constructor.')
        obj.update(position=position, fixed_base=False, visual_only=False)
    return {'task_revision': parameters['task_revision'], 'apparatus_proposal': plan,
        'native_support_and_access_verified': False, 'source_assets_preserved': True}


def _table_primitives(prefix, xy, floor_z, surface_z, color):
    """Explicit proposed dimensions; native reach/fit are always separate gates."""
    rows = [{'name': prefix, 'position': [*xy, surface_z - .025],
             'dimensions_m': [.6, .5, .05], 'rgba': color}]
    for index, (dx, dy) in enumerate(((-.24, -.19), (-.24, .19), (.24, -.19), (.24, .19))):
        rows.append({'name': f'{prefix}_leg_{index}',
            'position': [xy[0] + dx, xy[1] + dy, (floor_z + surface_z - .05) / 2],
            'dimensions_m': [.04, .04, surface_z - .05 - floor_z]})
    return rows


def build_history_retrieval(source, context):
    """Version source-supported multiple targets into a history-dependent mission."""
    from wia.tasks.temporal_policy import TASK_EDIT2_TEMPORAL_REVISION
    raw, payload, originals = _source_data(source)
    objects = history_target_inventory(source, context.config)
    redesigned = context.config.get('history_inventory_revision') == HISTORY_INVENTORY_REVISION
    if len(objects) < 2:
        raise ValueError('History retrieval requires at least two source-supported targets.')
    labels = context.config.get('history_target_labels', {})
    route = context.config.get('history_public_route')
    apparatus = context.config.get('history_apparatus')
    destination = context.config.get('history_destination')
    if (not isinstance(route, str) or not route.strip() or not apparatus or not destination
            or any(not labels.get(obj['_key']) for obj in objects)):
        raise ValueError('History revision needs explicit route, target labels, apparatus and named destination.')
    if len(set(labels.values())) != len(objects):
        raise ValueError('History targets need distinguishable public landmark/appearance labels.')
    bindings = {obj['_key']: {'name': obj['_key'], 'category': obj['category'], 'model': obj['model']} for obj in objects}
    for row in apparatus_objects(apparatus):
        binding = {key: row[key] for key in ('name', 'category', 'model') if key in row}
        if row['name'] in bindings and bindings[row['name']] != binding:
            raise ValueError('History apparatus cannot overwrite a conflicting target binding.')
        bindings[row['name']] = binding
    destination_key = destination['binding_key']
    if destination_key not in bindings:
        raise ValueError('The named delivery surface must be constructed or explicitly bound.')
    targets = [obj['_key'] for obj in objects]
    p = {'task_revision': TASK_EDIT2_TEMPORAL_REVISION, 'requires_native_visual_discovery': True, 'target_keys': targets,
        'public_target_labels': {key: labels[key] for key in targets}, 'public_route': route,
        'followup_seed': context.seed, 'followup_rule': 'rank_selected_after_frozen_complete_unique_history',
        'destination_key': destination_key, 'destination_label': destination['public_label'],
        'placement_relation': destination.get('relation', 'on_top'), 'apparatus': apparatus,
        'source_objects': {obj['_key']: obj for obj in objects},
        'source_identity': source.relative_path, 'source_sha256': sha256(source.path),
        'release_stability_steps': 15, 'maximum_release_speed_m_s': .02,
        'maximum_release_angular_speed_rad_s': .1,
        'budget_status': 'pending_representative_native_route_and_retrieval',
        'native_visual_adapter': 'task_visual_evidence',
        'historical_count_answer_is_provenance_only': raw['answer']}
    if redesigned:
        p.update(history_inventory_revision=HISTORY_INVENTORY_REVISION,
            original_target_keys=[obj['_key'] for obj in originals],
            added_target_keys=[obj['_key'] for obj in objects[len(originals):]],
            original_source_objects=deepcopy(originals), original_source_target_count=len(originals),
            capability_redesign='One original source target plus two approved visually distinct added assets; '
                'the three-target inventory is a new history-memory capability, not the original source count.')
    init = {'task_revision': TASK_EDIT2_TEMPORAL_REVISION, 'scene_model': raw['scene'], 'apparatus': apparatus}
    position = [*payload['start_xy'], float(payload['robot_start_z'])]
    quaternion = payload['step_log'][0]['robot_quat_xyzw']
    patches = [{'operation': 'category_initialization', 'phase': 'before_construction', 'module': MODULE,
                'parameters': init},
        {'operation': 'robot_pose', 'position': position, 'yaw_deg': math.degrees(_yaw(quaternion)), 'head_pitch_deg': 0.},
        {'operation': 'category_initialization', 'phase': 'after_postprocessing', 'module': MODULE,
            'parameters': {'restore': 'original_episode_start', 'scene': raw['scene'], 'room': raw['room'],
                'position': position, 'quaternion_xyzw': quaternion, 'source_sha256': sha256(source.path),
                'provenance': {'source_json': source.relative_path, 'task_revision': TASK_EDIT2_TEMPORAL_REVISION}}}]
    public = {'objective': f'Survey all {len(targets)} distinct targets along this route: {route} '
        'Target inventory: ' + '; '.join(labels[k] for k in targets) + '. '
        'Remember their first-observed order. After observing every target from an informative legal viewpoint, '
        'submit ANSWER: SURVEY_COMPLETE. A reproducible rank question will then be revealed. '
        f'Identify that history-selected target, retrieve it, and place it on {destination["public_label"]}.',
        'success_criteria': 'Complete the informative survey without moving targets, resolve the revealed rank '
            'using your own first-encounter history, then grasp the same bound object and release it stably at '
            'the named destination. Submit the selected public target label as ANSWER: <label>. '
            'Simultaneous first encounters do not define an order and invalidate that order question.',
        'environment': 'One continuous native episode; no actor replay or historical count answer is supplied. '
            'The colored target stations and named delivery surface are explicit initialized apparatus.',
        'constraints': ['During the survey, preserve targets, furniture and apparatus; use legal mounted-view inspection.',
            'After the follow-up, manipulate only the selected target through public controls.',
            'No arbitrary travel distance, return trip or lift-height bonus is required.']}
    evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': HISTORY_OPERATION,
                  'binding_keys': list(bindings), 'parameters': p}
    task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches,
        capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper'])
    scene = original_scene(source, context, bindings=bindings, patches=patches)
    scene['runtime'] = {'requires_native_visual_discovery': True}
    result = {'task_definition': task, 'scene_definition': scene,
        'status': 'generated', 'execution_status': 'unavailable', 'operation_recipe': HISTORY_OPERATION,
        'native_gates': {key: 'pending' for key in ('source_asset_preservation', 'supported_reachable_stations',
            'stable_robot_start', 'informative_encounters', 'nonambiguous_history', 'grasp_and_stable_destination')},
        'source_identity': source.relative_path, 'final_selection': 'provisional',
        'interaction_policy': {'mode': 'manipulation', 'group_bindings': targets,
            'manipulation_bindings': targets, 'relocation_policy': 'independent_groups',
            'allow_fallback_table': False, 'source_region': raw['room']},
        'capability_change': 'Active first-encounter temporal memory and retrieval; historical count QA retained privately.'}
    if redesigned:
        result.update(history_inventory_revision=HISTORY_INVENTORY_REVISION,
            original_target_keys=list(p['original_target_keys']), added_target_keys=list(p['added_target_keys']),
            capability_change=p['capability_redesign'])
        result['interaction_policy'].update(placement_groups=[[key] for key in targets],
            support_preferences={key:apparatus['target_support_bindings'][key] for key in targets},
            support_search='measured_surfaces', source_camera_policy='preserve')
    return result


def native_temporal_placement(session, executor, config, target_key, destination_key):
    """Read native grasp, contact, destination and current robot support directly."""
    from .partial_occlusion import _placement_measurements
    scene = session.env.scene
    target = scene.object_registry('name', config['bindings'][target_key])
    destination = scene.object_registry('name', config['bindings'][destination_key])
    if target is None or destination is None:
        raise ValueError('Native target or destination binding unavailable.')
    measured = _placement_measurements(session, executor, config, target, destination)
    try:
        from wia.tasks.interaction_setup import _robot_contact_check
        objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
        floors = [obj for obj in objects if obj.category in ('floor', 'floors')]
        if not floors:
            raise ValueError('Native floor bodies unavailable.')
        floor_z = max(float(_array(obj.aabb[1])[2]) for obj in floors)
        base = _robot_contact_check(session.robot, [obj.prim_path for obj in floors], floor_z)
        quaternion = _array(session.robot.get_position_orientation()[1])
        quaternion = quaternion / np.linalg.norm(quaternion)
        measured['robot_terminal_safe'] = bool(base['floor_supported'] and base['collision_free']
            and 1 - 2 * (quaternion[0] ** 2 + quaternion[1] ** 2) >= .95)
        measured['robot_contact_evidence'] = base
        measured['target_dynamic_collidable'] = not target.fixed_base and not target.visual_only
    except (ValueError, AttributeError, RuntimeError, KeyError, TypeError) as error:
        measured['robot_terminal_safe'] = None
        measured['measurement_errors']['robot_support'] = f'{type(error).__name__}: {error}'
    return measured


class NativePlacementProgress:
    """Count real control samples; repeated score reads cannot create a dwell."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.selected = False
        self.selection_available = True
        self.stable_steps = 0
        self.last = None

    def update(self, measured, native_target_name, *, count_step=False):
        self.last = measured
        grasp = measured.get('gripper')
        if grasp is None:
            self.selection_available = False
        elif count_step and grasp.get('holding') is True and grasp.get('constraint_active') is True:
            if grasp.get('object_name') == native_target_name:
                self.selected = True
        released = isinstance(grasp, dict) and all(grasp.get(key) is False for key in
            ('holding', 'constraint_active', 'release_pending')) and grasp.get('object_name') is None
        available = all(measured.get(key) is not None for key in ('destination_relation', 'linear_speed_m_s',
            'angular_speed_rad_s', 'robot_withdrawn', 'robot_terminal_safe', 'target_dynamic_collidable')) and grasp is not None
        stable = bool(available and released and measured['destination_relation'] and measured['robot_withdrawn']
            and measured['robot_terminal_safe'] and measured['target_dynamic_collidable']
            and measured['linear_speed_m_s'] <= self.parameters['maximum_release_speed_m_s']
            and measured['angular_speed_rad_s'] <= self.parameters['maximum_release_angular_speed_rad_s'])
        if not stable:
            self.stable_steps = 0
        elif count_step:
            self.stable_steps += 1
        return available, bool(stable and self.stable_steps >= self.parameters['release_stability_steps'])


def native_protected_inventory(session, bindings, keys):
    """Capture exact native poses/scales for task items and their apparatus."""
    result = {}
    for key in keys:
        obj = session.env.scene.object_registry('name', bindings[key])
        if obj is None:
            raise ValueError('Protected temporal object unavailable: ' + key)
        result[key] = {'name': obj.name, 'category': obj.category, 'model': getattr(obj, 'model', None),
            'position': _array(obj.get_position_orientation()[0]).tolist(),
            'orientation': _array(obj.get_position_orientation()[1]).tolist(), 'scale': _array(obj.scale).tolist()}
    return result


def protected_inventory_preserved(before, after, *, except_keys=()):
    for key, row in before.items():
        if key in except_keys:
            continue
        current = after.get(key)
        if current is None or any(current[field] != row[field] for field in ('name', 'category', 'model', 'scale')):
            return False
        a, b = np.array(row['orientation']), np.array(current['orientation'])
        if (np.linalg.norm(np.array(row['position']) - current['position']) > .03
                or min(np.linalg.norm(a - b), np.linalg.norm(a + b)) > .05):
            return False
    return True


class HistoryRetrievalEvaluator:
    """Freeze actual unique encounters before selecting the seeded rank question."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters, self.output = config['parameters'], Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history, self.seen, self.ties = [], set(), []
        self.frozen, self.selected_key, self.question_rank = None, None, None
        self.answer, self.active, self.visual_available = None, False, False
        self.placement = NativePlacementProgress(self.parameters)
        self.constraint_violation = False
        self.protection_available = True
        self.last_boundary = None

    def _protected(self):
        return native_protected_inventory(self.session, self.config['bindings'], self.config['binding_keys'])

    def _placement(self):
        return native_temporal_placement(self.session, self.controller, self.config,
            self.selected_key, self.parameters['destination_key'])

    def begin_episode(self, observation):
        self.initial = self._protected()
        self.active = True
        return self.record(observation)

    def _preserved(self):
        try:
            current = self._protected()
            allowed = [self.selected_key] if self.frozen is not None else []
            if not protected_inventory_preserved(self.initial, current, except_keys=allowed):
                self.constraint_violation = True
        except (ValueError, AttributeError, RuntimeError, KeyError, TypeError):
            self.protection_available = False

    def _observe(self, observation):
        from wia.tasks.temporal_policy import informative_temporal_rows
        evidence = informative_temporal_rows(self.session, observation, self.config['bindings'], self.parameters)
        self.visual_available = evidence['available']
        if self.frozen is None and evidence['available']:
            fresh = sorted(set(evidence['objects']).intersection(self.parameters['target_keys']) - self.seen)
            if len(fresh) > 1:
                # All cameras belong to one snapshot; camera iteration order cannot define chronology.
                self.ties.append({'snapshot_id': observation.snapshot_id, 'targets': fresh})
            for key in fresh:
                self.history.append({'key': key, 'snapshot_id': observation.snapshot_id,
                    'action_step': observation.action_step, 'evidence': evidence['objects'][key]})
                self.seen.add(key)
        self.last_boundary = (observation.snapshot_id, observation.action_step)
        return evidence

    def complete_survey(self, observation):
        import random
        if self.frozen is not None:
            raise ValueError('The encounter history has already been frozen.')
        self._observe(observation)
        self._preserved()
        if not self.visual_available or self.seen != set(self.parameters['target_keys']):
            raise ValueError('The survey lacks informative encounters for every target.')
        if self.ties:
            raise ValueError('Simultaneous first encounters make the rank question ambiguous.')
        if self.constraint_violation or not self.protection_available:
            raise ValueError('Survey preservation is not verified.')
        self.frozen = tuple(row['key'] for row in self.history)
        self.question_rank = random.Random(self.parameters['followup_seed']).randrange(1, len(self.frozen) + 1)
        self.selected_key = self.frozen[self.question_rank - 1]
        question = (f'Survey complete. Which target was number {self.question_rank} in your first-observed order? '
            'Submit its public target label as ANSWER: <label>, retrieve that same target, and place it on '
            + self.parameters['destination_label'] + '.')
        if self.parameters.get('followup_rule') in ('before_reference_encounter', 'after_reference_encounter'):
            before = self.parameters['followup_rule'] == 'before_reference_encounter'
            reference_index = random.Random(self.parameters['followup_seed']).randrange(
                1 if before else 0, len(self.frozen) if before else len(self.frozen) - 1)
            selected_index = reference_index - 1 if before else reference_index + 1
            self.question_rank = selected_index + 1
            self.selected_key = self.frozen[selected_index]
            reference = self.parameters['public_target_labels'][self.frozen[reference_index]]
            question = ('Survey complete. For the next delivery, retrieve the item you first inspected immediately '
                + ('before ' if before else 'after ') + reference + '. Use your actual survey history. '
                'Submit its item label as ANSWER: <label>, then place that same item on '
                + self.parameters['destination_label'] + '.')
        events = getattr(self.session, 'public_task_events', None)
        if events is None:
            events = []
            self.session.public_task_events = events
        if not isinstance(events, list):
            raise ValueError('Public follow-up event transport must be a list.')
        events.append({'event_id': 'history_followup_' + str(len(events) + 1),
            'kind': 'history_followup', 'text': question,
            'action_step': observation.action_step, 'snapshot_id': observation.snapshot_id})
        self.followup_public_text = question
        return {'survey_complete': True, 'followup_question': question}

    def report_answer(self, answer, observation):
        if str(answer).strip() == 'SURVEY_COMPLETE':
            return self.complete_survey(observation)
        if self.frozen is None:
            raise ValueError('The rank question is unavailable before survey completion.')
        self.answer = str(answer).strip()

    def sample_step(self):
        if not self.active:
            return
        self._preserved()
        if self.frozen is None:
            try:
                grasp = self.controller.debug_gripper_state()
                if grasp.get('holding') is True or grasp.get('constraint_active') is True:
                    self.constraint_violation = True
                elif grasp.get('holding') is not False or grasp.get('constraint_active') is not False:
                    self.protection_available = False
            except (ValueError, AttributeError, RuntimeError, KeyError, TypeError):
                self.protection_available = False
        if self.frozen is not None:
            try:
                measured = self._placement()
            except (ValueError, AttributeError, RuntimeError, KeyError, TypeError) as error:
                measured = {'measurement_error': str(error)}
            self.placement.update(measured, self.config['bindings'][self.selected_key], count_step=True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        visual = self._observe(observation)
        self._preserved()
        physical_available, placed = False, False
        if self.frozen is not None:
            try:
                measured = self._placement()
            except (ValueError, AttributeError, RuntimeError, KeyError, TypeError) as error:
                measured = {'measurement_error': str(error)}
            physical_available, placed = self.placement.update(measured, self.config['bindings'][self.selected_key])
        correct = self.frozen is not None and self.answer == self.parameters['public_target_labels'][self.selected_key]
        items = [{'id': 'survey', 'completed': self.seen == set(self.parameters['target_keys']) and not self.ties,
                  'kind': 'checkpoint', 'evaluation_validity': 'valid' if self.visual_available else 'unavailable'},
            {'id': 'history_answer', 'completed': bool(correct), 'kind': 'semantic_answer', 'evaluation_validity': 'valid'},
            {'id': 'correct_history_bound_selection', 'completed': self.placement.selected,
             'kind': 'checkpoint', 'evaluation_validity': 'valid' if self.placement.selection_available else 'unavailable'},
            {'id': 'stable_destination', 'completed': placed, 'kind': 'terminal_condition',
             'evaluation_validity': 'valid' if physical_available else 'unavailable'}]
        available = all(item['evaluation_validity'] == 'valid' for item in items) and self.protection_available
        result = {'schema': 'task_edit2_history_retrieval_v1', 'completion_items': items,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'evaluation_validity': 'complete' if available else 'partial',
            'task_success': bool(available and all(item['completed'] for item in items) and not self.constraint_violation),
            'constraints_satisfied': not self.constraint_violation if self.protection_available else None,
            'encounter_history': self.history, 'ambiguous_first_encounters': self.ties,
            'frozen_history': self.frozen, 'followup_rank': self.question_rank,
            'history_selected_binding': self.selected_key, 'reported_answer': self.answer,
            'visual_evidence': visual, 'release_stable_steps': self.placement.stable_steps,
            'native_placement': self.placement.last}
        from wia.tasks.completion import progress
        result.update(progress(result))
        write_json(self.output / 'state.json', result)
        return result


def qualify_history_retrieval(session, executor, config, bound):
    """Reject missing native target support or named destination measurements."""
    try:
        from omnigibson.object_states import Touching
        references = native_protected_inventory(session, config['bindings'], config['binding_keys'])
        checks = {}
        for key in config['parameters']['target_keys']:
            obj = bound[key]
            checks['target_dynamic:' + key] = not obj.fixed_base and not obj.visual_only
            support_names = config['parameters']['apparatus']['target_support_bindings'][key]
            support = session.env.scene.object_registry('name', config['bindings'][support_names])
            checks['native_target_support:' + key] = bool(obj.states[Touching].get_value(support))
        measured = native_temporal_placement(session, executor, config, config['parameters']['target_keys'][0],
                                             config['parameters']['destination_key'])
        checks['stable_floor_supported_robot'] = measured.get('robot_terminal_safe') is True
        return {'accepted': all(checks.values()), 'checks': checks, 'native_inventory': references,
            'informative_history_and_retrieval_verified': False}
    except (ValueError, AttributeError, RuntimeError, KeyError, TypeError, ImportError) as error:
        return {'accepted': False, 'reason': 'native_history_preconditions_unavailable', 'error': str(error)}

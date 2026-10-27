"""Run one original Unobserved Change episode across two real scene states.

Use a temporal_understanding_two_phase category task through run_esi_robot.py.
For a native continuity check run scripts/check_esi_temporal_phases.py with an
existing pending scene. Only TemporalPhaseController.advance mutates content;
qualification and the evaluator read measured state without acting or stepping.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from copy import deepcopy
import json
from pathlib import Path
import threading

import numpy as np

from wia.authoring.io import write_json
from wia.tasks.interaction_setup import array, _robot_contact_check
from wia.authoring.converters.temporal_understanding_two_phase import physical_target_keys

MODULE = 'core.esi_converters.temporal_understanding_two_phase'
OPERATION = 'observe_two_phase_change'
PHYSICAL_OPERATION = 'compare_change_and_restore_or_collect'


def _owner():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Temporal scene changes require the simulator owner thread.')


def _pose(obj):
    position, quaternion = (array(value) for value in obj.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0:
        raise ValueError('Invalid native temporal pose.')
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _same_pose(first, second, tolerance=1e-6):
    a, b = np.asarray(first['quaternion_xyzw']), np.asarray(second['quaternion_xyzw'])
    return bool(np.max(np.abs(np.asarray(first['position']) - second['position'])) <= tolerance
                and min(np.max(np.abs(a - b)), np.max(np.abs(a + b))) <= tolerance)


def _visible(obj):
    return bool(obj.visible)


def _camera_mount(sensor):
    """Read the authored mount, not a world-to-parent floating-point reconstruction."""
    getter = getattr(sensor, 'get_attribute', None)
    if callable(getter):
        position = getter('xformOp:translate')
        orientation = getter('xformOp:orient')
        return [list(position), [*orientation.GetImaginary(), orientation.GetReal()]]
    position, quaternion = sensor.get_position_orientation(frame='parent')
    return [array(position).tolist(), array(quaternion).tolist()]


def _set_pose(obj, position, quaternion):
    # Native setters require torch tensors; these are CPU tensors, not CUDA work.
    import torch
    obj.set_position_orientation(position=torch.tensor(position, dtype=torch.float32),
                                 orientation=torch.tensor(quaternion, dtype=torch.float32))
    obj.keep_still()


def _container_floor_support(obj, floor):
    """Read real unchanged-floor contacts; content contact is never a substitute."""
    contacts = getattr(obj, 'contact_list', None)
    if not callable(contacts) or floor.visual_only:
        return []
    floor_z, rows = float(array(floor.aabb[1])[2]), []
    within = lambda body, root: body == root or body.startswith(root + '/')
    for contact in contacts():
        external = [str(body) for body in (contact.body0, contact.body1) if not within(str(body), obj.prim_path)]
        if not external or not all(within(body, floor.prim_path) for body in external):
            continue
        position, normal = array(contact.position), array(contact.normal)
        if (position.shape == normal.shape == (3,) and abs(position[2] - floor_z) <= .015
                and abs(normal[2]) >= .5):
            rows.append({'external_bodies': external, 'position': position.tolist(), 'normal': normal.tolist()})
    return rows


class TemporalPhaseController:
    """Apply the source's external change, preserving the same robot and world."""

    def __init__(self, session, parameters):
        _owner()
        self.session, self.parameters = session, deepcopy(parameters)
        self.phase = 'phase_1'
        self.prepared_boundary = None
        self.transition = None
        self.failed = False
        self.agent_observation_started = False
        self.interaction_rebase = None
        self.objects = {key: session.env.scene.object_registry('name', binding['name'])
                        for key, binding in parameters['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('Both original temporal phases must be preloaded.')
        self.contents = list(parameters['phase_keys']['image1']) + list(parameters['phase_keys']['image2'])
        self.slots = {}
        for phase in ('image1', 'image2'):
            self.slots[phase] = {key: None for key in parameters['container_keys']}
            for key in parameters['phase_keys'][phase]:
                container = parameters['relations'][key]
                if self.slots[phase][container] is not None:
                    raise ValueError('A temporal source slot has more than one content object.')
                self.slots[phase][container] = key
        # An unchanged source slot keeps its original actor and exact placement.
        for container, second in list(self.slots['image2'].items()):
            first = self.slots['image1'][container]
            if first and second and self._identity(first) == self._identity(second):
                self.slots['image2'][container] = first
        self.active = dict(self.slots['image1'])
        self.parking = {key: _pose(self.objects[key]) for key in self.contents}
        for index, key in enumerate(self.contents):
            if key in self.active.values():
                self.parking[key] = {'position': [1000. + 2.5 * index, 1000., 120.],
                                     'quaternion_xyzw': [0., 0., 0., 1.]}
            self._set_visible(self.objects[key], key in self.active.values())
        self.initial_scales = {key: array(obj.scale).tolist() for key, obj in self.objects.items()
                               if key != 'floor'}

    def _identity(self, key):
        binding = self.parameters['bindings'][key]
        return binding['category'], binding['model']

    def _set_visible(self, obj, visible):
        obj.visible = visible
        if self.parameters.get('survey_revision') == 'free_survey_10_v1':
            # Update visual leaves as well as their parent. RTX and instance
            # annotators must agree after a parent visibility transition.
            for link in obj.links.values():
                for mesh in link.visual_meshes.values():
                    mesh.visible = visible

    def rebase_interaction(self, evidence):
        """Accept measured initialization translations without moving any actor.

        Shared placement has already moved each container and its active first
        phase content. Future phase actors remain at their original parking
        poses; only their expected onstage poses receive the container delta.
        """
        _owner()
        if (self.phase != 'phase_1' or self.prepared_boundary is not None or self.transition is not None
                or self.failed or self.agent_observation_started or self.interaction_rebase is not None):
            raise ValueError('Temporal support rebasing is initialization-only and may run once.')
        if evidence.get('accepted') is not True:
            raise ValueError('Temporal support rebasing requires accepted native placement.')
        translations = evidence.get('object_translations_m', {})
        supports = evidence.get('support_prim_paths', {})
        original, rebased = self.parameters, deepcopy(self.parameters)
        before = self._unchanged_world()
        inactive = {key: deepcopy(self.parking[key]) for key in self.contents if key not in self.active.values()}
        deltas, support_evidence, floor_evidence, protrusions = {}, {}, {}, {}
        for container in original['container_keys']:
            obj = self.objects[container]
            delta = array(translations.get(obj.name, [0., 0., 0.]))
            if delta.shape != (3,):
                raise ValueError('Every temporal support translation must be a finite XYZ vector.')
            first = self.slots['image1'][container]
            direct = supports.get(obj.name)
            unchanged_floor = []
            if not direct and np.linalg.norm(delta) <= 1e-6:
                unchanged_floor = _container_floor_support(obj, self.objects['floor'])
            if original.get('accessible_support_rebase_required') and not (direct or unchanged_floor):
                raise ValueError('Every accessible container needs its own measured support; content support is insufficient.')
            deltas[container] = delta.tolist()
            support_evidence[container] = deepcopy(direct)
            floor_evidence[container] = unchanged_floor
            source = rebased['source_objects'][container]
            source['position'] = (array(source['position']) + delta).tolist()
            if 'bbox' in source:
                source['bbox'] = (array(source['bbox']) + delta).tolist()
            if not _same_pose(_pose(obj), source, .03):
                raise ValueError('Actual initialized container pose disagrees with its accepted translation.')
            for key in self.contents:
                if original['relations'][key] != container:
                    continue
                planned = rebased['source_objects'][key]
                planned['position'] = (array(planned['position']) + delta).tolist()
                if key == first:
                    member = array(translations.get(self.objects[key].name, [0., 0., 0.]))
                    if member.shape != (3,) or not np.allclose(member, delta, atol=1e-6, rtol=0):
                        raise ValueError('Active Phase-1 contents must retain their container-relative placement.')
                    if not _same_pose(_pose(self.objects[key]), planned, .04):
                        raise ValueError('Actual Phase-1 content disagrees with the accepted group translation.')
                    content_low = array(self.objects[key].aabb[0])
                    container_low = array(obj.aabb[0])
                    if content_low[2] < container_low[2] - .005:
                        protrusions[key] = {'content_aabb_min_m': content_low.tolist(),
                            'container_aabb_min_m': container_low.tolist(),
                            'below_container_aabb_m': float(container_low[2] - content_low[2]),
                            'mesh_or_support_collision_proven': False}
        if any(not _same_pose(_pose(self.objects[key]), pose) for key, pose in inactive.items()):
            raise ValueError('Inactive temporal actors must remain parked during support placement.')
        if before != self._unchanged_world():
            raise ValueError('Temporal rebasing must not mutate robot, furniture or camera mounts.')
        proof = {'accepted': True, 'phase': 'phase_1', 'container_translations_m': deltas,
            'container_support_prim_paths': support_evidence, 'unchanged_container_floor_contacts': floor_evidence,
            'source_content_below_container_bounds': protrusions,
            'warning_codes': ['source_content_support_clearance_unverified'] if protrusions else [],
            'support_surfaces_z_m': deepcopy(evidence.get('support_surfaces_z_m', {})),
            'inactive_actor_poses_unchanged': True, 'world_unchanged_by_callback': True,
            'actor_pose_writes': 0, 'physics_steps': 0, 'before_first_agent_observation': True}
        self.parameters, self.interaction_rebase = rebased, proof
        try:
            if not self.measure()['valid']:
                raise ValueError('Rebased temporal slots failed native initialization checks.')
        except BaseException:
            self.parameters, self.interaction_rebase = original, None
            raise
        return deepcopy(proof)

    def _unchanged_world(self):
        scene_objects = self.session.env.scene.objects
        scene_objects = list(scene_objects.values()) if isinstance(scene_objects, dict) else list(scene_objects)
        content_ids = {id(self.objects[key]) for key in self.contents}
        poses = {obj.name: _pose(obj) for obj in scene_objects if id(obj) not in content_ids}
        robot = getattr(self.session, 'robot', None)
        if robot is None:
            # Shared placement runs before EsiSession attaches its robot field.
            # Read the sole already constructed robot without assigning it or
            # changing the original environment/actor ordering.
            robots = getattr(self.session.env, 'robots', None)
            if robots is None or len(robots) != 1 or robots[0] is None:
                raise ValueError('Temporal initialization requires exactly one original environment robot.')
            robot = robots[0]
        object_joints = {obj.name: array(obj.get_joint_positions()).tolist() for obj in scene_objects
                         if id(obj) not in content_ids and obj is not robot
                         and getattr(obj, 'n_joints', 1) > 0
                         and callable(getattr(obj, 'get_joint_positions', None))}
        joints = getattr(robot, 'get_joint_positions', None)
        cameras = {}
        for name, sensor in getattr(robot, 'sensors', {}).items():
            if hasattr(sensor, 'get_position_orientation'):
                cameras[name] = _camera_mount(sensor)
        return {'poses': poses, 'robot': _pose(robot),
                'robot_joints': array(joints()).tolist() if callable(joints) else None,
                'camera_mounts': cameras, 'object_joints': object_joints}

    def measure(self, executor=None):
        """Read actual assets/locations; saved labels never establish a live state."""
        rows, checks = {}, {}
        p = self.parameters
        if p.get('accessible_support_rebase_required'):
            checks['accepted_accessible_support_rebase'] = self.interaction_rebase is not None
        for key, obj in self.objects.items():
            if key == 'floor':
                continue
            identity = p['bindings'][key]
            rows[key] = {**_pose(obj), 'name': obj.name, 'category': obj.category, 'model': getattr(obj, 'model', None),
                         'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only),
                         'scale': array(obj.scale).tolist()}
            checks['identity:' + key] = all(getattr(obj, field) == value for field, value in identity.items())
            checks['scale:' + key] = bool(np.allclose(rows[key]['scale'], self.initial_scales[key], atol=1e-6, rtol=0))
        for key in p['container_keys']:
            source = p['source_objects'][key]
            checks['container_layout:' + key] = _same_pose(rows[key], source, .03)
            checks['container_physics:' + key] = rows[key]['fixed_base'] and not rows[key]['visual_only']
        slots = {}
        for container, key in self.active.items():
            expected_key = self.slots['image1' if self.phase == 'phase_1' else 'image2'][container]
            if key is None:
                slots[container] = None
                checks['slot:' + container] = expected_key is None
                continue
            obj, row = self.objects[key], rows[key]
            source = p['source_objects'][key]
            physical_target = (p.get('task_edit2_physical') and self.phase == 'phase_2'
                and key in physical_target_keys(p))
            checks['slot:' + container] = (key == expected_key and _visible(obj)
                and not row['visual_only'] and (not row['fixed_base'] if physical_target else row['fixed_base'])
                and (physical_target or np.linalg.norm(np.asarray(row['position']) - source['position']) <= .04))
            low, high = (array(value) for value in obj.aabb)
            checks['content_dimensions:' + key] = bool(low.shape == high.shape == (3,)
                and np.all(high > low)
                and np.max(np.abs(high - low - source['bbox_size_m'])) <= .03)
            slots[container] = {'category': obj.category, 'model': obj.model}
        active = {key for key in self.active.values() if key is not None}
        for key in self.contents:
            if key not in active:
                row = rows[key]
                checks['inactive:' + key] = (not _visible(self.objects[key])
                    and (row['fixed_base'] or (p.get('task_edit2_physical') and key in physical_target_keys(p)))
                    and row['position'][0] > 900. and row['position'][1] > 900.)
        robot = None
        if executor is not None:
            floor = self.objects['floor']
            floor_z = float(array(floor.aabb[1])[2])
            robot = _robot_contact_check(self.session.robot, [floor.prim_path], floor_z)
            q = np.asarray(_pose(self.session.robot)['quaternion_xyzw'])
            q = q / np.linalg.norm(q)
            checks['robot_floor_support'] = robot['floor_supported']
            checks['robot_clear'] = robot['collision_free']
            checks['robot_upright'] = bool(1 - 2 * (q[0] ** 2 + q[1] ** 2) >= .9)
            gripper = executor.debug_gripper_state()
            if not (p.get('task_edit2_physical') and self.phase == 'phase_2'):
                checks['gripper_empty'] = (gripper.get('object_name') is None
                    and gripper.get('holding') is False and gripper.get('constraint_active') is False)
        checks = {key: bool(value) for key, value in checks.items()}
        return {'phase': self.phase, 'slots': slots, 'objects': rows,
                'robot_support': robot, 'checks': checks, 'valid': bool(all(checks.values()))}

    def advance(self):
        """One source-defined external event; no robot command or physics step."""
        _owner()
        if self.phase != 'phase_1' or self.prepared_boundary is None or self.failed:
            raise ValueError('The original temporal episode cannot advance at this boundary.')
        world = self._unchanged_world()
        before = {key: {'pose': _pose(self.objects[key]), 'visible': _visible(self.objects[key])}
                  for key in self.contents}
        prior_active = dict(self.active)
        try:
            desired = {key for key in self.slots['image2'].values() if key is not None}
            for key in self.contents:
                obj = self.objects[key]
                if key in desired:
                    if key not in prior_active.values():
                        source = self.parameters['source_objects'][key]
                        _set_pose(obj, source['position'], source['quaternion_xyzw'])
                    self._set_visible(obj, True)
                    if self.parameters.get('task_edit2_physical') and key in physical_target_keys(self.parameters):
                        if obj.fixed_base or not callable(getattr(obj, 'enable_gravity', None)):
                            raise ValueError('Phase-2 target lacks native dynamic gravity control.')
                        obj.enable_gravity()
                else:
                    self._set_visible(obj, False)
                    parked = self.parking[key]
                    _set_pose(obj, parked['position'], parked['quaternion_xyzw'])
            self.active = dict(self.slots['image2'])
            self.phase = 'phase_2'
            if self.parameters.get('survey_revision') == 'free_survey_10_v1':
                # Propagate kinematic / visibility edits without advancing time.
                # Reading tensor poses alone does not certify rendered updates.
                sim = self.session.pipeline.og.sim
                sim.pi.update_simulation(elapsedStep=0, currentTime=sim.current_time)
                from omnigibson.utils.usd_utils import PoseAPI
                # Native property edits (visibility / gravity) can resynchronize
                # a dynamic body's old authored transform. Submit the event's
                # poses after those edits have propagated, then read them back.
                for key in desired - set(prior_active.values()):
                    source = self.parameters['source_objects'][key]
                    obj = self.objects[key]
                    _set_pose(obj, source['position'], source['quaternion_xyzw'])
                    if not obj.kinematic_only:
                        # This native CPU-physics build updates tensor poses
                        # before USD/Fabric transforms. Publish the same rigid
                        # root pose to both representations, without stepping
                        # physics or editing any mesh / physical parameter.
                        from omnigibson.prims.xform_prim import XFormPrim
                        position, quaternion = obj.get_position_orientation()
                        XFormPrim.set_position_orientation(obj.root_link,
                            position=position, orientation=quaternion, frame='world')
                PoseAPI.invalidate()
                PoseAPI._refresh()
                actor_checks = {}
                for key in desired:
                    obj = self.objects[key]
                    expected = (before[key]['pose'] if key in prior_active.values()
                        else self.parameters['source_objects'][key])
                    low, high = (array(v) for v in obj.aabb)
                    actor_checks[key] = {'position_matches_event': _same_pose(_pose(obj), expected, .001),
                        'geometry_is_onstage': bool(np.max(np.abs([low, high])) < 900.),
                        'actual_pose': _pose(obj), 'aabb_m': [low.tolist(), high.tolist()]}
                if not all(row['position_matches_event'] and row['geometry_is_onstage']
                           for row in actor_checks.values()):
                    raise ValueError('Native temporal event placement failed: ' + json.dumps(actor_checks))
            measured = self.measure()
            after_world = self._unchanged_world()
            if not measured['valid'] or world != after_world:
                raise ValueError('Temporal change did not preserve its original native state contract.')
            self.transition = {'from_phase': 'phase_1', 'to_phase': 'phase_2',
                'prepared_boundary': list(self.prepared_boundary), 'world_before': world,
                'world_after': after_world, 'world_unchanged': True, 'native_after': measured,
                'physics_steps': 0, 'robot_actions': 0, 'source_defined_external_event': True}
            if self.parameters.get('survey_revision') == 'free_survey_10_v1':
                self.transition['active_actor_placement_checks'] = actor_checks
            invalidate = getattr(self.session, 'invalidate_camera_cache', None)
            if callable(invalidate):
                invalidate()
            if hasattr(self.session, '_step_camera_data'):
                self.session._step_camera_data = None
            return self.transition
        except BaseException:
            self.failed = True
            for key, saved in before.items():
                _set_pose(self.objects[key], saved['pose']['position'], saved['pose']['quaternion_xyzw'])
                self._set_visible(self.objects[key], saved['visible'])
            self.active, self.phase = prior_active, 'phase_1'
            raise


def install_temporal_phases(session, parameters):
    if hasattr(session, 'temporal_phase_controller'):
        raise ValueError('A temporal scene can be initialized only once.')
    session.temporal_phase_controller = TemporalPhaseController(session, parameters)
    return {'temporal_episode': True, 'phase': 'phase_1',
            'preloaded_phase_count': 2, 'source_defined_external_event': True}


def _configuration(session, config):
    if (module_identity(config.get('module')) != MODULE or config.get('operation') not in (OPERATION, PHYSICAL_OPERATION)
            or (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('Native temporal configuration does not match this episode.')
    controller = session.temporal_phase_controller
    if set(config['binding_keys']) != set(controller.objects):
        raise ValueError('Native temporal bindings are incomplete.')
    if any(config['bindings'][key] != obj.name for key, obj in controller.objects.items()):
        raise ValueError('Native temporal identities do not match the loaded source.')
    return controller


def qualify_operation(session, executor, config, bound):
    try:
        controller = _configuration(session, config)
        native_controller = getattr(executor, 'controller', executor)
        if native_controller is not executor and getattr(native_controller, 'session', None) is not session:
            raise ValueError('Temporal qualification received a controller from another native session.')
        measured = controller.measure(native_controller)
        accepted = controller.phase == 'phase_1' and measured['valid']
        return {'accepted': accepted, 'checks': measured['checks'], 'native_state': measured,
                'reason': None if accepted else 'native_temporal_preconditions_failed',
                'robot_reasoning_executed': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_temporal_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'robot_reasoning_executed': False}


def _expected_answer(parameters, first, second):
    changes = {}
    for key in first:
        a, b = first[key], second[key]
        changes[key] = ('no_change' if a == b else 'add' if a is None else
                        'remove' if b is None else 'replace')
    changed = [key for key, value in changes.items() if value != 'no_change']
    kind = parameters['source_task_type']
    if kind == 'change_detection':
        answer = 'yes' if changed else 'no'
    elif kind == 'change_identification':
        index = parameters.get('target_box_index')
        key = ('container_' + str(index).zfill(2)) if index is not None else next(iter(first))
        answer = changes[key]
    elif kind == 'current_state_reasoning' and len(changed) == 1:
        label = parameters['box_labels'][changed[0]]
        answer = label if label.endswith(' box') else label + ' box'
    else:
        raise ValueError('The measured temporal change does not define the original question.')
    return answer, changes


class TemporalChangeEvaluator:
    """Score an Agent answer only after two actual observation boundaries."""

    def __init__(self, session, executor, config, output):
        self.controller = _configuration(session, config)
        self.session, self.executor, self.config = session, executor, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode = 0
        self.phase_1 = self.phase_2 = self.answer = None
        self.last_boundary = None
        self.navigation = None
        self.survey_reference = None
        self.survey_violations = []
        self.survey_violation_evidence = {}
        self.survey_environment_warnings = {}
        self.survey_contact_bodies = set()
        self.survey_contact_history_verified = True

    def _survey_contacts(self, measured):
        """Latch all non-floor contacts, including unresolved external bodies."""
        support = measured.get('robot_support')
        if not isinstance(support, dict) or 'unexpected_contact_bodies' not in support:
            self.survey_contact_history_verified = False
            return
        self.survey_contact_bodies.update(support['unexpected_contact_bodies'])

    def _survey_drift(self, name, evidence, failures, key):
        policy = getattr(self.session, 'background_movement_policy', None)
        critical = {obj.name for obj in self.controller.objects.values()}
        if policy:
            critical.update(policy.get('protected', ()))
        if evidence['after'] is None:
            reason = 'native_object_state_missing'
        elif name in critical:
            reason = 'critical_task_dependency'
        elif (not policy or 'protected' not in policy or 'measurement_errors' not in policy
              or policy['measurement_errors'] or getattr(self.session, 'background_policy_error', None)):
            reason = 'dependency_measurement_unverified'
        elif not self.survey_contact_history_verified:
            reason = 'contact_history_unverified'
        elif self.survey_contact_bodies:
            # Conservatively retain protection even for indirect motion after
            # contact with another object; current contact absence is insufficient.
            reason = 'non_floor_robot_contact_during_survey'
        else:
            if key not in self.survey_environment_warnings:
                self.survey_environment_warnings[key] = dict(evidence,
                    reason='noncritical_drift_without_observed_robot_contact',
                    last_boundary=self.last_boundary)
                write_json(self.output / 'survey_environment_warnings.json', {
                    'phase': 'phase_1', 'warnings': self.survey_environment_warnings})
            self.session.background_static_map_stale = True
            for attribute in ('_navigation_cache', '_collision_cache', '_path_cache'):
                cache = getattr(self.session, attribute, None)
                if hasattr(cache, 'clear'):
                    cache.clear()
            return
        failures[key] = dict(evidence, reason=reason)

    def begin_episode(self, observation):
        if self.episode or self.controller.phase != 'phase_1':
            raise ValueError('A temporal episode must start once in Phase 1.')
        measured = self.controller.measure(self.executor)
        write_json(self.output / 'phase_1_initial_qualification.json', measured)
        if not measured['valid']:
            raise ValueError('The original Phase 1 failed native qualification.')
        self.survey_reference = self.controller._unchanged_world()
        self._survey_contacts(measured)
        self.controller.survey_violation = False
        self.controller.agent_observation_started = True
        self.episode = 1
        return self.record(observation, action=None)

    def prepare_phase_transition(self, observation):
        if (not self.episode or self.controller.phase != 'phase_1'
                or self.last_boundary != (observation.snapshot_id, observation.action_step)):
            raise ValueError('Phase 1 completion does not match the active observation.')
        if self.survey_violations:
            raise ValueError('Phase 1 read-only safety check failed; inspect private survey evidence.')
        measured = self.controller.measure(self.executor)
        if not measured['valid']:
            raise ValueError('The original Phase 1 was changed before its external event.')
        self.sample_step()
        if self.survey_violations:
            raise ValueError('Phase 1 read-only safety check failed; inspect private survey evidence.')
        self.phase_1 = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                        'native_state': measured}
        self.phase_1_robot = array(self.session.robot.get_position_orientation()[0]).copy()
        self.controller.prepared_boundary = self.last_boundary

    def report_answer(self, answer, observation):
        if (not self.episode or self.controller.phase != 'phase_2' or self.phase_2 is None
                or self.last_boundary != (observation.snapshot_id, observation.action_step)):
            raise ValueError('An answer requires a current Phase 2 observation.')
        self.answer = {'text': answer, 'snapshot_id': observation.snapshot_id,
                       'action_step': observation.action_step}

    def sample_step(self):
        """Check read-only survey invariants and controlled Phase-2 navigation."""
        if self.controller.phase == 'phase_1' and self.survey_reference is not None:
            current = self.controller._unchanged_world()
            original = self.survey_reference
            try:
                measured = self.controller.measure(self.executor)
            except Exception:
                self.survey_contact_history_verified = False
                raise
            self._survey_contacts(measured)
            # Robot pose/joints may change. Only noncritical background drift
            # with a verified, contact-free history is an environment warning.
            protected = {key: pose for key, pose in original['poses'].items()
                         if key != self.session.robot.name}
            failures = {}
            for name, pose in protected.items():
                actual = current['poses'].get(name)
                if actual is None or not _same_pose(pose, actual, .005):
                    self._survey_drift(name, {'before': pose, 'after': actual},
                                       failures, 'object_pose:' + name)
            for name, values in original.get('object_joints', {}).items():
                actual = current.get('object_joints', {}).get(name)
                if actual is None or not np.allclose(values, actual, atol=.005, rtol=0):
                    self._survey_drift(name, {'before': values, 'after': actual},
                                       failures, 'object_joints:' + name)
            for name in set(original['camera_mounts']) | set(current['camera_mounts']):
                before, after = original['camera_mounts'].get(name), current['camera_mounts'].get(name)
                if before != after:
                    failures['camera_mount:' + name] = {'before': before, 'after': after}
            for name, passed in measured['checks'].items():
                if not passed:
                    failures['native_check:' + name] = measured
            new_failures = {key: value for key, value in failures.items()
                            if key not in self.survey_violation_evidence}
            if failures:
                self.controller.survey_violation = True
            if new_failures:
                self.survey_violation_evidence.update(new_failures)
                self.survey_violations = list(self.survey_violation_evidence)
                write_json(self.output / 'survey_violation.json', {
                    'phase': 'phase_1', 'last_boundary': self.last_boundary,
                    'violations': self.survey_violation_evidence})
            return
        if (self.controller.phase != 'phase_2' or self.navigation is None
                or not self.config['parameters'].get('phase_2_navigation_required')):
            return
        nav = self.navigation
        position, quaternion = (array(value) for value in self.session.robot.get_position_orientation())
        now = float(self.session.pipeline.og.sim.current_time)
        dt = now - nav['previous_time_s']
        displacement = float(np.linalg.norm(position - np.asarray(nav['previous_position_m'])))
        control_dt = float(self.executor.dt)
        limit = self.executor.config.max_translation_speed_m_s * control_dt + self.executor.config.translation_tolerance_m
        if dt <= 0 or dt > control_dt * 1.5 or displacement > limit:
            nav['faults'].append('unsampled_or_discontinuous_native_motion')
        measured = self.controller.measure(self.executor)
        if not measured['valid']:
            nav['faults'].append('native_source_or_robot_safety_failed')
        nav['samples'] += 1
        nav['travel_m'] += float(np.linalg.norm(position[:2] - np.asarray(nav['previous_position_m'])[:2]))
        nav['maximum_departure_m'] = max(nav['maximum_departure_m'], float(np.linalg.norm(
            position[:2] - self.phase_1_robot[:2])))
        radius = self.config['parameters']['phase_2_container_visit_radius_m']
        for key in self.controller.parameters['container_keys']:
            centre = array(self.controller.objects[key].get_position_orientation()[0])
            if float(np.linalg.norm(position[:2] - centre[:2])) <= radius:
                nav['visited_containers'][key] = {'sample': nav['samples'], 'simulation_time_s': now,
                                                 'base_position_m': position.tolist()}
        nav.update(previous_position_m=position.tolist(), previous_time_s=now,
                   stopped=bool(dt > 0 and displacement / dt <= .02))

    def _navigation_checks(self):
        required = self.config['parameters'].get('phase_2_navigation_required')
        if not required:
            return {'phase_2_navigation_requirement_satisfied': True}
        nav = self.navigation
        if nav is None:
            return {'phase_2_navigation_requirement_satisfied': False}
        position = array(self.session.robot.get_position_orientation()[0])
        if np.linalg.norm(position - np.asarray(nav['previous_position_m'])) > 1e-5:
            nav['faults'].append('native_motion_outside_control_step_sampling')
        minimum = self.config['parameters']['phase_2_minimum_base_travel_m']
        return {'phase_2_controlled_native_steps': nav['samples'] > 0,
            'phase_2_real_base_travel': nav['travel_m'] >= minimum and nav['maximum_departure_m'] >= minimum,
            'phase_2_original_container_visited': bool(nav['visited_containers']),
            'phase_2_continuously_safe_sampled_motion': not nav['faults'],
            'phase_2_stopped': nav['stopped']}

    def record(self, observation, *, action=None):
        if not self.episode:
            return None
        self.last_boundary = (observation.snapshot_id, observation.action_step)
        expected, changes = None, None
        try:
            measured = self.controller.measure(self.executor)
            if self.controller.phase == 'phase_2' and self.phase_1 is not None:
                if self.navigation is None and self.config['parameters'].get('phase_2_navigation_required'):
                    self.navigation = {'samples': 0, 'travel_m': 0., 'maximum_departure_m': 0.,
                        'visited_containers': {}, 'faults': [], 'stopped': True,
                        'previous_position_m': array(self.session.robot.get_position_orientation()[0]).tolist(),
                        'previous_time_s': float(self.session.pipeline.og.sim.current_time)}
                if observation.snapshot_id > self.phase_1['snapshot_id']:
                    self.phase_2 = {'snapshot_id': observation.snapshot_id,
                                    'action_step': observation.action_step, 'native_state': measured}
                expected, changes = _expected_answer(self.config['parameters'],
                    self.phase_1['native_state']['slots'], measured['slots'])
            checks = {'phase_1_read_only_preserved': not self.survey_violations,
                'native_source_state_valid': measured['valid'],
                'phase_1_observed': self.phase_1 is not None,
                'source_event_preserved_world': bool(self.controller.transition and
                    self.controller.transition['world_unchanged'] and not self.controller.failed),
                'phase_2_observed_after_phase_1': self.phase_2 is not None,
                'answer_at_current_boundary': self.answer is not None and
                    (self.answer['snapshot_id'], self.answer['action_step']) == self.last_boundary,
                'answer_is_original_public_option': self.answer is not None and
                    self.answer['text'] in self.config['parameters']['public_options'],
                'measured_answer_matches_original_qa': expected is not None and
                    expected == self.config['parameters']['original_answer'],
                'agent_answer_correct': self.answer is not None and self.answer['text'] == expected}
            checks.update(self._navigation_checks())
            error = None
        except Exception as exception:
            measured, checks = None, {'native_measurements_available': False}
            error = f'{type(exception).__name__}: {exception}'
        result = {'schema': 'private_unobserved_two_phase_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'phase': self.controller.phase, 'task_success': bool(all(checks.values())),
            'criteria': checks, 'native_state': measured, 'phase_1': self.phase_1, 'phase_2': self.phase_2,
            'agent_answer': self.answer, 'measured_expected_answer': expected, 'measured_changes': changes,
            'transition': self.controller.transition, 'measurement_error': error,
            'phase_2_native_navigation': self.navigation,
            'phase_1_survey_violations': self.survey_violations,
            'phase_1_survey_violation_evidence': self.survey_violation_evidence,
            'phase_1_environment_warnings': self.survey_environment_warnings,
            'phase_1_contact_history': {'verified': self.survey_contact_history_verified,
                                      'non_floor_bodies': sorted(self.survey_contact_bodies)},
            'evaluation_validity': 'unavailable' if error else 'complete',
            'scoring_version': 'task_completion_v2',
            'observation_evidence': {'captured': True, 'delivered_to_agent': 'recorded_by_runner',
                'target_visibility': 'unverified', 'answerability': 'unverified'}}
        write_json(self.output / 'state.json', result)
        with (self.output / 'temporal_boundaries.jsonl').open('a') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def create_evaluator(session, executor, config, output):
    if config.get('operation') == PHYSICAL_OPERATION:
        return PhysicalTemporalChangeEvaluator(session, executor, config, output)
    return TemporalChangeEvaluator(session, executor, config, output)


class PhysicalTemporalChangeEvaluator(TemporalChangeEvaluator):
    """Keep read-only survey and hidden event, replacing obsolete distance gates."""
    def __init__(self, session, executor, config, output):
        from wia.authoring.converters.temporal_understanding_agent_observation import NativePlacementProgress
        super().__init__(session, executor, config, output)
        self.parameters = config['parameters']
        self.coverage = {'phase_1': set(), 'phase_2': set()}
        self.visual_available = {'phase_1': False, 'phase_2': False}
        self.placement = NativePlacementProgress(self.parameters)
        self.operations = self.parameters.get('physical_operations', [])
        self.placements = [NativePlacementProgress({**self.parameters, **row}) for row in self.operations]
        self.physical_reference = None
        self.physical_constraints = True
        self.physical_measurement_error = None
        self.phase_2_event = None

    def _inventory(self):
        from wia.authoring.converters.temporal_understanding_agent_observation import native_protected_inventory
        return native_protected_inventory(self.session, self.config['bindings'], self.config['binding_keys'])

    def _visual(self, observation):
        from wia.tasks.temporal_policy import informative_temporal_rows
        result = informative_temporal_rows(self.session, observation, self.config['bindings'], self.parameters)
        phase = self.controller.phase
        self.visual_available[phase] = result['available']
        if result['available']:
            self.coverage[phase].update(result['objects'])
        return result

    def _required_coverage(self, phase):
        # Empty interiors need an informative container receipt; no fictitious object identity.
        required = set(self.controller.parameters['container_keys'])
        if phase == 'phase_1':
            required.update(key for key in self.controller.slots['image1'].values() if key is not None)
        elif self.parameters['physical_branch'] != 'no_change':
            target = self.parameters.get('physical_target_key')
            if target:
                required.add(target)
            required.update(physical_target_keys(self.parameters))
        return required

    def prepare_phase_transition(self, observation):
        self._visual(observation)
        if (self.parameters.get('require_complete_phase_1_coverage', True) and (
                not self.visual_available['phase_1']
                or not self._required_coverage('phase_1').issubset(self.coverage['phase_1']))):
            raise ValueError('Phase 1 requires informative views of every comparison location and present content.')
        return super().prepare_phase_transition(observation)

    def _measure_placement(self):
        from wia.authoring.converters.temporal_understanding_agent_observation import native_temporal_placement
        return native_temporal_placement(self.session, self.executor, self.config,
            self.parameters['physical_target_key'], self.parameters['physical_destination_key'])

    def _physical_sample(self, *, count_step=False):
        from wia.authoring.converters.temporal_understanding_agent_observation import protected_inventory_preserved
        try:
            current = self._inventory()
            if self.physical_reference is None:
                self.physical_reference = current
            target = self.parameters.get('physical_target_key')
            if not protected_inventory_preserved(self.physical_reference, current, except_keys=physical_target_keys(self.parameters)):
                self.physical_constraints = False
            self.physical_measurement_error = None
            if self.parameters['physical_branch'] == 'no_change':
                grasp = self.executor.debug_gripper_state()
                if not isinstance(grasp.get('holding'), bool) or not isinstance(grasp.get('constraint_active'), bool):
                    raise ValueError('No-change requires measured empty native gripper.')
                if grasp['holding'] or grasp['constraint_active'] or grasp.get('object_name') is not None:
                    self.physical_constraints = False
                return True, True
            if self.operations:
                from wia.authoring.converters.temporal_understanding_agent_observation import native_temporal_placement
                results = []
                for operation, tracker in zip(self.operations, self.placements):
                    config = {**self.config, 'parameters': {**self.parameters, **operation}}
                    target = operation['physical_target_key']
                    measured = native_temporal_placement(self.session, self.executor, config,
                        target, operation['physical_destination_key'])
                    results.append(tracker.update(measured, self.config['bindings'][target], count_step=count_step))
                return all(row[0] for row in results), all(row[1] for row in results)
            measured = self._measure_placement()
            return self.placement.update(measured, self.config['bindings'][target], count_step=count_step)
        except (KeyError, ValueError, AttributeError, RuntimeError, TypeError, ImportError) as error:
            self.physical_measurement_error = f'{type(error).__name__}: {error}'
            return False, False

    def sample_step(self):
        if self.controller.phase == 'phase_1':
            return super().sample_step()
        if self.controller.phase == 'phase_2' and self.episode:
            self._physical_sample(count_step=True)

    def report_answer(self, answer, observation):
        # Preserve the current-boundary original option interface.
        return super().report_answer(answer, observation)

    def record(self, observation, *, action=None):
        if not self.episode:
            return None
        from wia.tasks.completion import progress
        self.last_boundary = (observation.snapshot_id, observation.action_step)
        visual = self._visual(observation)
        error, measured, expected, changes = None, None, None, None
        physical_available, placed = False, False
        try:
            measured = self.controller.measure(self.executor if self.parameters['physical_branch'] == 'no_change' else None)
            if self.controller.phase == 'phase_2' and self.phase_1 is not None:
                transition = self.controller.transition
                if transition is None or not transition.get('world_unchanged') or self.controller.failed:
                    raise ValueError('The measured hidden source event is unavailable.')
                if observation.snapshot_id <= self.phase_1['snapshot_id']:
                    raise ValueError('Phase 2 requires a fresh post-event mounted observation.')
                if self.phase_2_event is None:
                    self.phase_2_event = deepcopy(transition['native_after'])
                self.phase_2 = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
                expected, changes = _expected_answer(self.parameters,
                    self.phase_1['native_state']['slots'], self.phase_2_event['slots'])
                physical_available, placed = self._physical_sample()
        except (KeyError, ValueError, AttributeError, RuntimeError, TypeError) as exception:
            error = f'{type(exception).__name__}: {exception}'
        survey = (self.phase_1 is not None and not self.survey_violations
            and (not self.parameters.get('require_complete_phase_1_coverage', True)
                 or self._required_coverage('phase_1').issubset(self.coverage['phase_1'])))
        comparison = (self.phase_2 is not None and self._required_coverage('phase_2').issubset(self.coverage['phase_2']))
        answer = self.answer is not None and self.answer['text'] == expected
        truth_valid = expected is not None and expected == self.parameters['original_answer']
        event_branch_valid = (self.phase_2_event is not None and self.phase_1 is not None
            and measured_change_branch(self.phase_1['native_state']['slots'], self.phase_2_event['slots'])
                == self.parameters['physical_branch'])
        source_valid = measured is not None and measured.get('valid') is True and truth_valid and event_branch_valid
        phase1_validity = 'valid' if self.visual_available['phase_1'] else 'unavailable'
        phase2_validity = 'valid' if self.visual_available['phase_2'] and error is None else 'unavailable'
        items = [{'id': 'two_phase_informative_comparison', 'kind': 'checkpoint',
                  'completed': bool(survey and comparison), 'evaluation_validity': 'valid'
                  if phase1_validity == phase2_validity == 'valid' else 'unavailable'},
            {'id': 'change_answer', 'kind': 'semantic_answer', 'completed': bool(answer),
             'evaluation_validity': 'valid' if truth_valid else 'unavailable'}]
        if self.parameters['physical_branch'] != 'no_change':
            items += [{'id': 'correct_changed_object_selection', 'kind': 'checkpoint',
                'completed': all(t.selected for t in self.placements) if self.operations else self.placement.selected,
                'evaluation_validity': 'valid' if (all(t.selection_available for t in self.placements)
                    if self.operations else self.placement.selection_available) else 'unavailable'},
                {'id': 'stable_branch_destination', 'kind': 'terminal_condition', 'completed': placed,
                 'evaluation_validity': 'valid' if physical_available else 'unavailable'}]
        constraints = not self.survey_violations and self.physical_constraints
        available = bool(source_valid and error is None and self.physical_measurement_error is None
            and self.visual_available['phase_1'] and self.visual_available['phase_2']
            and all(item['evaluation_validity'] == 'valid' for item in items))
        result = {'schema': 'task_edit2_physical_temporal_v1', 'phase': self.controller.phase,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'physical_branch': self.parameters['physical_branch'], 'completion_items': items,
            'task_success': bool(available and constraints and all(item['completed'] for item in items)),
            'evaluation_validity': 'complete' if available else 'unavailable',
            'constraints_satisfied': bool(constraints), 'source_event_native_state': self.phase_2_event,
            'phase_1': self.phase_1, 'phase_2': self.phase_2, 'agent_answer': self.answer,
            'measured_expected_answer': expected, 'measured_changes': changes,
            'visual_coverage': {phase: sorted(keys) for phase, keys in self.coverage.items()},
            'visual_evidence': visual, 'native_state': measured,
            'native_placement': [t.last for t in self.placements] if self.operations else self.placement.last,
            'release_stable_steps': [t.stable_steps for t in self.placements] if self.operations else self.placement.stable_steps,
            'phase_1_survey_violations': self.survey_violations,
            'measurement_error': error or self.physical_measurement_error,
            'obsolete_distance_gates_applied': False, 'measured_branch_matches_definition': event_branch_valid}
        result.update(progress(result))
        write_json(self.output / 'state.json', result)
        return result


def measured_change_branch(first, second):
    """Require the physical obligation to match the actual native source event."""
    changed = [key for key in first if first[key] != second.get(key)]
    if not changed:
        return 'no_change'
    if len(changed) == 1:
        key = changed[0]
        return 'added' if first[key] is None else 'removed' if second[key] is None else 'replaced'
    removed = [key for key in changed if first[key] is not None and second[key] is None]
    added = [key for key in changed if first[key] is None and second[key] is not None]
    if len(removed) == len(added) == 1 and first[removed[0]] == second[added[0]]:
        return 'moved'
    return 'multiple'

"""Convert original inclined planes into real grasp, placement and release trials.

Run CPU batching from the repository root with the behavior environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Native hooks are read-only and run through the existing guarded robot runtime.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_contact, legal_background_grasp

import copy

from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.authoring.io import sha256, write_json
from wia.tasks.interaction_setup import (
    reference_position, support_contact_paths, touching_task_support, on_top_task_support,
)


MODULE = 'wia.authoring.converters.physical_dynamics_inclined_plane'
TASK_REVISION = 'task_edit_20261018_v1'
MECHANISM_CONTRACT = {'preserve': ['source_ramp_pose', 'ramp_scale', 'friction', 'trial_outcome_reference'], 'goal': 'native_supported_release_and_observed_motion', 'intended_motion': ['sliding', 'rolling', 'runoff'], 'safety_context': 'Intended post-release ramp motion is not automatically an accidental drop.'}
QUESTION = re.compile(r'Do you think the (.+?) will stay stable \(not slide\) when placed on the slope\?')
OPERATION = 'place_release_observe'
APPARATUS_OPERATION = 'place_release_observe_whole_apparatus_v1'
from wia.tasks.inclined_apparatus import (REVISION as APPARATUS_REVISION, NATIVE_RECEIPT,
    source_apparatus_proposal, materialize_apparatus, support_envelope_fit, common_translation)
MINIMUM_LIFT_M = .05
OBSERVATION_SECONDS = 1.
MINIMUM_TRACE_SAMPLES = 20
MAXIMUM_SAMPLE_GAP_S = .1
SLIDE_THRESHOLD_M = .03
FALL_HEIGHT_M = .05
MAXIMUM_LINEAR_SPEED = .02
MAXIMUM_ANGULAR_SPEED = .1
SPAWN_RADIUS_M = .4
MAP_RESOLUTION_M = .01


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite native or source measurement.')
    return value


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
        for item in value)


def _label(value):
    return ' '.join(str(value).replace('_', ' ').lower().split())


def _source_data(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('source_category_mismatch')
    if tuple(Path(source.relative_path).parts[-3:-1]) != (raw.get('scene'), raw.get('room')):
        raise ValueError('source_hierarchy_mismatch')
    match = QUESTION.fullmatch(raw.get('question', '').strip())
    obj = map_records(payload.get('objects')).get('task_obj', {})
    category, model = payload.get('obj_category') or obj.get('category'), payload.get('obj_model') or obj.get('model')
    if not category or not model or not match or _label(match[1]) != _label(category):
        raise ValueError('original_public_object_binding_unresolved')
    if obj.get('category') != category or obj.get('model') != model:
        raise ValueError('original_object_annotations_disagree')
    for field in ('pos_on_floor', 'pos_init', 'aabb_min_init', 'aabb_max_init'):
        if not _vector(obj.get(field), 3):
            raise ValueError('missing_original_object_' + field)
    if any(a >= b for a, b in zip(obj['aabb_min_init'], obj['aabb_max_init'])):
        raise ValueError('invalid_original_object_bounds')
    slope = payload.get('slope', {})
    if any(not _vector(slope.get(key), 3) for key in ('position', 'aabb_min', 'aabb_max')):
        raise ValueError('missing_original_slope_geometry')
    if not _vector(payload.get('slope_scale'), 3) or min(payload['slope_scale']) <= 0:
        raise ValueError('invalid_original_slope_dimensions')
    if not _vector(payload.get('slope_quaternion'), 4) or not any(payload['slope_quaternion']):
        raise ValueError('invalid_original_slope_quaternion')
    cameras = payload.get('camera_poses') or []
    if not cameras or any(not _vector(camera.get('position'), 3) or not _vector(
            camera.get('quaternion_xyzw'), 4) for camera in cameras):
        raise ValueError('missing_original_reference_camera')
    if not payload.get('floor_name'):
        raise ValueError('original_floor_binding_unresolved')
    for key in ('static_friction', 'dynamic_friction'):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid_original_' + key)
    # The generator sets every camera eye to floor_top + CAMERA_HEIGHT (0.40 m).
    floor_z = float(np.median([camera['position'][2] - .4 for camera in cameras]))
    if abs(floor_z - slope['aabb_min'][2]) > .005:
        raise ValueError('source_camera_floor_height_disagrees_with_slope_bottom')
    return raw, payload, obj, match[1], floor_z


@lru_cache(maxsize=1)
def _maps(asset_root, scene):
    """Read installed original maps without importing OmniGibson or starting GPU work."""
    import cv2
    cv2.setNumThreads(2)
    root = Path(asset_root)
    paths = [root / 'scenes' / scene / 'layout' / name for name in (
        'floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    categories = root / 'metadata/room_categories.txt'
    if not all(path.is_file() for path in [*paths, categories]):
        return None
    trav, instances, semantics = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(value is None for value in (trav, instances, semantics)) or not (
            trav.shape == instances.shape == semantics.shape and trav.shape[0] == trav.shape[1]):
        return None
    names = categories.read_text().splitlines()
    room_ids, counts = {}, Counter()
    # Match SegmentationMap's ascending instance-ID / semantic-category ordering.
    for instance in np.unique(instances):
        if instance == 0:
            continue
        row, col = np.argwhere(instances == instance)[0]
        semantic = int(semantics[row, col])
        if not 0 < semantic <= len(names):
            continue
        name = names[semantic - 1]
        room_ids[f'{name}_{counts[name]}'] = int(instance)
        counts[name] += 1
    return trav, instances, room_ids, {str(path): sha256(path) for path in [*paths, categories]}


def _robot_start(source, payload, obj, floor_z, context):
    data_root = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    asset_root = context.config.get('original_asset_root', str(data_root / 'behavior-1k-assets'))
    maps = _maps(str(asset_root), source.raw['scene'])
    if maps is None or source.raw['room'] not in maps[2]:
        return None
    trav, rooms, room_ids, hashes = maps
    slope_xy = np.asarray(payload['slope']['position'][:2])
    object_xy = np.asarray(obj['pos_on_floor'][:2])
    centre = (slope_xy + object_xy) / 2
    candidates = [(f'original_view_{index}', np.asarray(camera['position'][:2]))
                  for index, camera in enumerate(payload['camera_poses'])]
    # A bounded deterministic search around the original trial, never relocate assets.
    for radius in (.75, .9, 1.05, 1.2):
        for angle in range(0, 360, 15):
            radians = math.radians(angle)
            candidates.append((f'trial_ring_{radius}_{angle}', centre + radius * np.array(
                [math.cos(radians), math.sin(radians)])))
    pixel_radius = math.ceil(SPAWN_RADIUS_M / MAP_RESOLUTION_M)
    offsets = np.arange(-pixel_radius, pixel_radius + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= pixel_radius ** 2
    # Saved initial AABB encloses the same asset on the slope; use its XY diagonal
    # as a conservative object exclusion radius, not a footprint/reach certificate.
    extent = np.asarray(obj['aabb_max_init']) - obj['aabb_min_init']
    object_radius = float(np.linalg.norm(extent) / 2)
    slope_radius = float(np.linalg.norm(np.asarray(payload['slope_scale'])[:2]) / 2)
    valid = []
    for label, xy in candidates:
        row, col = (xy[::-1] / MAP_RESOLUTION_M + trav.shape[0] / 2).astype(int)
        if min(row, col) < pixel_radius or max(row, col) + pixel_radius >= trav.shape[0]:
            continue
        region = (slice(row - pixel_radius, row + pixel_radius + 1),
                  slice(col - pixel_radius, col + pixel_radius + 1))
        if not np.all(trav[region][disk] > 0) or not np.all(rooms[region][disk] == room_ids[source.raw['room']]):
            continue
        distances = [float(np.linalg.norm(xy - point)) for point in (slope_xy, object_xy)]
        if distances[0] < SPAWN_RADIUS_M + slope_radius + .05 or distances[1] < SPAWN_RADIUS_M + object_radius + .05:
            continue
        score = (max(distances), sum(distances), label)
        valid.append((score, xy, label, distances))
    if not valid:
        return None
    _, xy, label, distances = min(valid, key=lambda item: item[0])
    return {'position': [float(xy[0]), float(xy[1]), floor_z],
            'yaw_deg': math.degrees(math.atan2(centre[1] - xy[1], centre[0] - xy[0])),
            'selection': label, 'static_body_radius_m': SPAWN_RADIUS_M,
            'source_room_instance_id': room_ids[source.raw['room']],
            'distance_to_slope_and_floor_object_m': distances, 'map_sha256': hashes,
            'provenance': 'Original 1 cm floor_trav/insseg/semseg maps and room_categories; '
                'BaseMap.world_to_map; source slope.position, task_obj.pos_on_floor, '
                'slope_scale, object AABB and camera_poses. Camera eye_z minus generator CAMERA_HEIGHT=0.4.',
            'native_reach_and_collision_qualified': False}


def _configuration(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') not in (OPERATION, APPARATUS_OPERATION) or config.get('scene') != session.scene_name or
            config.get('room') != session.room_name or set(config.get('bindings', {})) != {'target', 'slope', 'floor'}):
        raise ValueError('Inclined plane operation requires its original scene and three native bindings.')
    if config['operation'] == APPARATUS_OPERATION and config.get('parameters', {}).get('task_revision') != APPARATUS_REVISION:
        raise ValueError('Whole-apparatus operation requires its explicit TaskEdit2 revision.')
    return config['parameters']


def _pose(obj):
    position, quaternion = [_array(value) for value in obj.get_position_orientation()]
    low, high = [_array(value) for value in obj.aabb]
    kinematic = bool(getattr(obj, 'kinematic_only', False))
    try:
        linear = float(np.linalg.norm(_array(obj.get_linear_velocity())))
        angular = float(np.linalg.norm(_array(obj.get_angular_velocity())))
        velocity_source = 'native_dynamic_velocity'
    except AttributeError:
        if not (kinematic and obj.fixed_base):
            raise
        # Installed RigidKinematicPrim has no velocity state. Its continuously
        # checked pose establishes preservation; do not invent measured zeros.
        linear = angular = None
        velocity_source = 'fixed_native_kinematic_body_without_dynamic_velocity_state'
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
            'scale': _array(obj.scale).tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'kinematic_only': kinematic, 'velocity_source': velocity_source,
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular}


def _static_or_slow(row):
    speed = row['linear_speed_m_s']
    return bool(speed <= MAXIMUM_LINEAR_SPEED if speed is not None else
        row.get('kinematic_only') is True and row.get('fixed_base') is True)


def _friction(obj):
    rows = []
    for link in obj.links.values():
        for mesh in link.collision_meshes.values():
            material = mesh.get_applied_physics_material()
            if material is None:
                raise ValueError('Source collision mesh has no measurable applied friction material.')
            values = [material.get_static_friction(), material.get_dynamic_friction()]
            if not all(value is not None and math.isfinite(float(value)) for value in values):
                raise ValueError('Native friction material is not measurable.')
            rows.append({'mesh': mesh.prim_path, 'static_friction': float(values[0]),
                         'dynamic_friction': float(values[1])})
    if not rows:
        raise ValueError('Source object has no native collision meshes.')
    return rows


def _slope_unchanged(row, parameters):
    revised = parameters.get('task_revision') == APPARATUS_REVISION
    delta = np.asarray(parameters.get('apparatus_translation_m', [0., 0., 0.])) if revised else np.zeros(3)
    angle = (Rotation.from_quat(row['quaternion_xyzw']).inv() * Rotation.from_quat(
        parameters['slope_quaternion'])).magnitude()
    return bool(not row['visual_only'] and row['fixed_base'] and
        np.linalg.norm(np.asarray(row['position']) - np.asarray(parameters['slope_position']) - delta) <= .005 and
        np.max(np.abs(np.asarray(row['scale']) - parameters['slope_scale'])) <= (1e-6 if revised else .001) and
        angle <= (1e-5 if revised else math.radians(1.)) and _static_or_slow(row))


def _robot_state(session):
    robot, scene = session.robot, session.env.scene
    low, high = [_array(value) for value in robot.links[robot.base_footprint_link_name].aabb]
    regions = [scene.seg_map.get_room_instance_by_point(np.array([x, y]))
               for x in (low[0], (low[0] + high[0]) / 2, high[0])
               for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj.prim_path for obj in objects if obj.category in ('floor', 'floors')]
    floor_contacts, obstacle_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        external = [str(body) for body in (contact.body0, contact.body1) if not (
            str(body) == robot.prim_path or str(body).startswith(robot.prim_path + '/'))]
        if not external:
            continue
        is_floor = all('groundPlane' in body or any(body == path or body.startswith(path + '/')
                                                  for path in floors) for body in external)
        (floor_contacts if is_floor else obstacle_contacts).append(external)
    return {'footprint_rooms': regions, 'in_original_room': all(room == session.room_name for room in regions),
            'floor_contacts': floor_contacts, 'obstacle_contacts': obstacle_contacts}


def _trial_state(target, slope, floor, parameters, apparatus=None):
    """Measure actual top-face contacts, floor contact and external interference."""
    from omnigibson.object_states import Touching
    row, slope_row = _pose(target), _pose(slope)
    rotation = Rotation.from_quat(slope_row['quaternion_xyzw'])
    half_size = np.asarray(parameters['slope_scale']) / 2
    centre = (np.asarray(row['aabb_min']) + row['aabb_max']) / 2
    local_centre = rotation.inv().apply(centre - slope_row['position'])
    top_contacts, interference = [], []
    for contact in target.contact_list():
        external = [str(body) for body in (contact.body0, contact.body1) if not (
            str(body) == target.prim_path or str(body).startswith(target.prim_path + '/'))]
        if not external:
            continue
        if all(body == slope.prim_path or body.startswith(slope.prim_path + '/') for body in external):
            position = rotation.inv().apply(_array(contact.position) - slope_row['position'])
            normal = rotation.inv().apply(_array(contact.normal))
            if (abs(position[0]) <= half_size[0] + .005 and abs(position[1]) <= half_size[1] + .005 and
                    abs(position[2] - half_size[2]) <= .01 and abs(normal[2]) >= .5):
                top_contacts.append(position.tolist())
        elif not all(body == floor.prim_path or body.startswith(floor.prim_path + '/') for body in external):
            interference.append(external)
    touching_slope = bool(target.states[Touching].get_value(slope))
    touching_floor = bool(target.states[Touching].get_value(floor))
    row.update(touching_slope=touching_slope, touching_floor=touching_floor,
               top_surface_contacts=top_contacts, interfering_contacts=interference,
               centre_in_slope_footprint=bool(np.all(np.abs(local_centre[:2]) <= half_size[:2])),
               top_supported=bool(touching_slope and top_contacts and not touching_floor and
                   np.all(np.abs(local_centre[:2]) <= half_size[:2]) and local_centre[2] > half_size[2]),
               slope_unchanged=_slope_unchanged(slope_row, parameters))
    if parameters.get('task_revision') == APPARATUS_REVISION:
        if apparatus is None:
            raise ValueError('Current selected apparatus support measurements are unavailable.')
        _measure_apparatus_trial(row, target, slope, floor, apparatus)
        row['slope_unchanged'] = bool(row['slope_unchanged'] and
            _materials_match(_friction(slope), apparatus['source_baseline']['slope']['materials']))
    return row, slope_row


def qualify_operation(session, executor, config, bound):
    """Read native preconditions; never move the robot or task assets."""
    if config.get('operation') == APPARATUS_OPERATION:
        return _qualify_apparatus(session, executor, config, bound)
    checks = {}
    try:
        parameters = _configuration(session, config)
        target, slope, floor = [bound[key] for key in ('target', 'slope', 'floor')]
        target_state, slope_state = _trial_state(target, slope, floor, parameters)
        robot = _robot_state(session)
        materials = {key: _friction(obj) for key, obj in (('target', target), ('slope', slope))}
        floor_top = float(_array(floor.aabb[1])[2])
        gripper = executor.debug_gripper_state()
        paths = support_contact_paths(session, target.name)
        adapted_support = bool(paths and on_top_task_support(session, target, floor) and
                               touching_task_support(session, target, floor))
        # A declared support may change the pickup start only. During the later
        # ramp release/observation, _trial_state still treats table contact as
        # interference and measures the unchanged original floor outcome.
        initial_interference = [bodies for bodies in target_state['interfering_contacts']
            if not (adapted_support and bodies and all(any(body == path or body.startswith(path + '/')
                for path in paths) for body in bodies))]
        checks = {'original_target_movable_collidable': not target.visual_only and not target.fixed_base,
                  'slope_pose_fixed_and_collidable': _slope_unchanged(slope_state, parameters),
                  'original_floor_collidable': not floor.visual_only,
                  'source_floor_height_preserved': abs(floor_top - parameters['floor_z_m']) <= .01,
                  'original_target_start_preserved':
                      bool(np.linalg.norm(np.asarray(target_state['position']) - reference_position(
                          session, target.name, parameters['target_floor_position'])) <= .05),
                  'initial_supported_by_task_surface_or_original_slope': bool(target_state['touching_floor'] or
                      target_state['top_supported'] or adapted_support) and not initial_interference,
                  'target_contact_reporting_available': any(link.contact_reporting_enabled for link in target.links.values()),
                  'target_stationary': target_state['linear_speed_m_s'] <= MAXIMUM_LINEAR_SPEED and
                      target_state['angular_speed_rad_s'] <= MAXIMUM_ANGULAR_SPEED,
                  'native_friction_matches_source': all(abs(row[key] - parameters[key]) <= 1e-4 for
                      rows in materials.values() for row in rows for key in ('static_friction', 'dynamic_friction')),
                  'initial_gripper_empty': gripper.get('object_name') is None and gripper.get('constraint_active') is False,
                  'robot_supported_in_source_room': robot['in_original_room'] and bool(robot['floor_contacts']) and
                      not robot['obstacle_contacts']}
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'reason': None if accepted else 'native_trial_precondition_failed',
                'checks': checks, 'native_target': target_state, 'native_slope': slope_state,
                'native_floor_top_m': floor_top, 'native_materials': materials, 'native_robot': robot,
                'initial_target_reference_m': reference_position(
                    session, target.name, parameters['target_floor_position']).tolist(),
                'initial_adapted_support_contact': adapted_support,
                'native_mass_kg': float(target.mass), 'real_robot_placement_qualified': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_trial_measurement_unavailable', 'checks': checks,
                'exception_summary': f'{type(error).__name__}: {error}', 'real_robot_placement_qualified': False}


def _native_time():
    import omnigibson as og
    result = {'time_s': float(og.sim.current_time), 'physics_dt_s': float(og.sim.get_physics_dt()),
              'sim_step_dt_s': float(og.sim.get_sim_step_dt())}
    if not all(math.isfinite(value) for value in result.values()) or min(
            result['physics_dt_s'], result['sim_step_dt_s']) <= 0:
        raise ValueError('Native clock and time increments must be measurable and positive.')
    return result


class InclinedPlaneEvaluator:
    """Measure a held lift, top-surface release and uninterrupted native time trace."""
    def __init__(self, session, controller, config, output):
        self.parameters = copy.deepcopy(_configuration(session, config))
        self.session, self.controller, self.config = session, controller, config
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('Original slope trial binding is missing.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.initial, self.last = 0, None, None

    def measure(self):
        target, slope, floor = [self.objects[key] for key in ('target', 'slope', 'floor')]
        apparatus = _resolve_apparatus(self.session, self.parameters, target, slope) if self.config['operation'] == APPARATUS_OPERATION else None
        parameters = {**self.parameters, 'apparatus_translation_m': apparatus['translation_m']} if apparatus else self.parameters
        row, _ = _trial_state(target, slope, floor, parameters, apparatus=apparatus) if apparatus else _trial_state(target, slope, floor, parameters)
        return row, self.controller.debug_gripper_state(), _native_time()

    def begin_episode(self, observation):
        self.episode += 1
        self.initial, gripper, self.initial_clock = self.measure()
        self.last_native_time, self.clock_valid = self.initial_clock['time_s'], True
        self.initially_empty = gripper.get('object_name') is None and gripper.get('constraint_active') is False
        bound = self.objects
        self.preconditions = qualify_operation(self.session, self.controller, self.config, bound)
        self.lifted, self.previous_held, self.pending_release = False, False, None
        self.previous_row = None
        self.trace, self.trace_valid = [], True
        self.release_seen = False
        self.wrong_object_grasped, self.slope_preserved = False, True
        self.apparatus_preserved = True
        self.target_physics_preserved = True
        self.control_evidence_available, self.no_episode_setters = True, True
        self.record(observation, action=None)

    def _update(self):
        if self.initial is None:
            return None
        row, gripper, clock = self.measure()
        now = clock['time_s']
        self.clock_valid = self.clock_valid and now >= self.last_native_time - 1e-7
        self.last_native_time = now
        held = (gripper.get('holding') is True and gripper.get('constraint_active') is True and
                gripper.get('object_name') == self.objects['target'].name)
        free = (gripper.get('object_name') is None and gripper.get('constraint_active') is False and
                gripper.get('gripper_open') is True and gripper.get('release_pending') is False)
        if (gripper.get('object_name') not in (None, self.objects['target'].name)
                and not legal_background_grasp(self.session, gripper.get('object_name'))):
            self.wrong_object_grasped = True
        self.slope_preserved = self.slope_preserved and row['slope_unchanged']
        if self.config['operation'] == APPARATUS_OPERATION:
            self.apparatus_preserved = self.apparatus_preserved and row['apparatus_support_preserved']
            self.target_physics_preserved = self.target_physics_preserved and row['source_target_physics_preserved']
            evidence = _trial_control_evidence(self.controller)
            self.control_evidence_available &= evidence['available']
            self.no_episode_setters &= evidence['available'] and evidence['state_setter_used'] is False
        lift = row['aabb_min'][2] - self.initial['aabb_min'][2]
        if held:
            self.lifted = self.lifted or lift >= MINIMUM_LIFT_M
            self.pending_release, self.trace, self.trace_valid = None, [], True
        elif self.previous_held:
            # The target must already be supported by the slope before detachment.
            # This excludes dropping beside the ramp or grazing its underside.
            self.pending_release = now if self.lifted and self.previous_row['top_supported'] else None
            self.trace, self.trace_valid = [], True
        if self.pending_release is not None and not self.trace and free:
            if now - self.pending_release <= .5 and row['top_supported'] and not row['interfering_contacts']:
                self.trace = [{'time_s': now, **row}]
            elif now - self.pending_release > .5:
                self.pending_release = None
        if self.trace and now > self.trace[-1]['time_s'] + 1e-7 and (
                self.trace[-1]['time_s'] - self.trace[0]['time_s'] < OBSERVATION_SECONDS - 1e-7):
            gap = now - self.trace[-1]['time_s']
            self.trace_valid = bool(self.trace_valid and free and not row['interfering_contacts'] and
                                    gap <= MAXIMUM_SAMPLE_GAP_S + 1e-7)
            self.trace.append({'time_s': now, **row})
        # Current native state must also remain free of robot or other-object contact.
        self.release_seen |= bool(self.trace)
        current_free = free and not row['interfering_contacts']
        elapsed = self.trace[-1]['time_s'] - self.trace[0]['time_s'] if self.trace else 0.
        drift = max((math.dist(sample['position'][:2], self.trace[0]['position'][:2]) for sample in self.trace), default=0.)
        floor_z = self.preconditions.get('native_floor_top_m', self.parameters['floor_z_m'])
        fallen = any(sample['aabb_min'][2] < floor_z + FALL_HEIGHT_M for sample in self.trace)
        recent = [sample for sample in self.trace if self.trace[-1]['time_s'] - sample['time_s'] <= .5 + 1e-7]
        stable = bool(recent and recent[-1]['time_s'] - recent[0]['time_s'] >= .45 and drift <= SLIDE_THRESHOLD_M and
            all(sample['top_supported'] and sample['linear_speed_m_s'] <= MAXIMUM_LINEAR_SPEED and
                sample['angular_speed_rad_s'] <= MAXIMUM_ANGULAR_SPEED for sample in recent))
        outcome = 'fallen_by_legacy_height_screen' if fallen else 'slid_or_rolled' if drift > SLIDE_THRESHOLD_M else (
            'stable_on_slope' if stable else 'unsettled_within_observation_window')
        criteria = {'native_initial_preconditions': self.preconditions['accepted'],
                    'initially_empty': self.initially_empty, 'native_held_lift_observed': self.lifted,
                    'released_after_top_surface_support': bool(self.trace),
                    'native_unheld_trace_complete': elapsed >= OBSERVATION_SECONDS - 1e-7 and
                        len(self.trace) >= MINIMUM_TRACE_SAMPLES,
                    'native_clock_monotonic': self.clock_valid,
                    'trace_uninterrupted_without_interference': self.trace_valid,
                    'target_currently_free_without_interference': bool(current_free),
                    'no_wrong_object_grasp': not self.wrong_object_grasped,
                    'source_slope_preserved': bool(self.slope_preserved),
                    'target_still_movable_collidable': not row['visual_only'] and not row['fixed_base']}
        apparatus_result = {}
        if self.config['operation'] == APPARATUS_OPERATION:
            safe = bool(self.trace and all(not sample['escaped_apparatus'] for sample in self.trace)
                and not row['escaped_apparatus'])
            criteria.update(complete_apparatus_support_preserved=self.apparatus_preserved,
                source_target_physics_preserved=self.target_physics_preserved,
                native_public_control_history_available=self.control_evidence_available,
                no_episode_simulator_setters=self.no_episode_setters,
                safe_trial_contained_on_apparatus=safe)
            if any(sample['touching_original_floor'] for sample in self.trace) or row['touching_original_floor']:
                outcome = 'escaped_apparatus_to_original_floor'
            elif not safe:
                outcome = 'escaped_selected_apparatus_deck' if row['escaped_apparatus'] else 'trial_not_yet_observed'
            elif row['touching_apparatus_deck'] and not row['touching_slope']:
                outcome = 'intended_runoff_on_selected_deck'
            elif drift > SLIDE_THRESHOLD_M:
                outcome = 'intended_slide_or_roll_on_apparatus'
            else:
                outcome = 'stable_on_slope' if stable else 'unsettled_within_apparatus'
            apparatus_result = {'apparatus_trial_safe': safe, 'apparatus_support_preserved': self.apparatus_preserved,
                'original_building_floor_is_trial_support': False}
        self.previous_held, self.previous_row = held, row
        self.last = {'native_target': row, 'native_clock': clock, 'gripper': gripper,
                     'lift_m': lift, 'criteria': criteria, 'trace': self.trace,
                     'trace_duration_s': elapsed, 'trace_samples': len(self.trace),
                     'maximum_xy_displacement_m': drift, 'legacy_fall_height_screen': fallen,
                     'released_stability_measured': stable, 'measured_outcome': outcome,
                     'task_success': bool(all(criteria.values()))}
        self.last.update(apparatus_result)
        return self.last

    def sample_step(self):
        self._update()

    def record(self, observation, *, action):
        value = self._update()
        if value is None:
            return None
        value = {**value, 'schema': 'private_whole_inclined_apparatus_trial_v1' if self.config['operation'] == APPARATUS_OPERATION else 'private_inclined_plane_trial_v1', 'episode': self.episode,
                 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                 'action': action, 'configuration': self.config,
                 'initial_native_target': self.initial, 'initial_native_preconditions': self.preconditions}
        available = bool(self.preconditions['accepted'] and self.clock_valid and
            (self.config['operation'] != APPARATUS_OPERATION or self.control_evidence_available))
        value.update(task_revision=APPARATUS_REVISION if self.config['operation'] == APPARATUS_OPERATION else TASK_REVISION, evaluation_validity='complete' if available else 'unavailable',
            intended_motion_context=MECHANISM_CONTRACT['intended_motion'],
            completion_items=[{'id': 'native_target_lift', 'kind': 'checkpoint', 'completed': self.lifted},
                {'id': 'supported_release', 'kind': 'checkpoint', 'completed': self.release_seen},
                {'id': 'uninterrupted_trial_observation', 'kind': 'checkpoint',
                 'completed': bool(value['criteria']['native_unheld_trace_complete'] and self.trace_valid)}])
        for item in value['completion_items']:
            item['evaluation_validity'] = 'valid' if available else 'unavailable'
        from wia.tasks.completion import progress
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return InclinedPlaneEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Physical Dynamics', 'Inclined Plane')
    version = 'task_edit_20261018_v1_1.4.0_accessible_task_repair_v2'

    def inspect(self, source):
        try:
            _source_data(source)
        except (ValueError, TypeError, KeyError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_contract_review_required']}
        return {'eligible': True, 'reason': 'Original ramp, floor, object and observer construction are available.'}

    def convert(self, source, context):
        raw, payload, obj, label, floor_z = _source_data(source)
        if context.config.get('task_edit2_inclined') is True:
            return _convert_apparatus(self, source, context)
        bindings = {'target': {'name': 'task_obj', 'category': obj['category'], 'model': obj['model']},
                    'slope': {'name': 'slope'}, 'floor': {'name': payload['floor_name']}}
        robot = _robot_start(source, payload, obj, floor_z, context)
        assumptions = [{
            'doubt': 'Original room geometry and floor-level R1 reach/grasp/placement are not physically qualified.',
            'chosen_assumption': 'Screen a source-room R1 start and place the pickup target on a measured accessible '
                'support, allowing its initial XY to change while preserving its size and orientation. Keep the '
                'original ramp pose, dimensions, angle, floor, friction and observers. Require native support and '
                'collision qualification and real guarded arm/gripper controls before physical scoring.',
            'alternatives': ['Keep a source-only diagnostic if no screened start exists or native reach fails.',
                             'Use guarded navigation to another original approach before grasping; do not move the ramp.'],
            'scope': source.relative_path,
            'risk': 'Map pixels and conservative 0.4 m disks do not certify the swept arms, gripper fit or floor reach. '
                'Original maps may differ from runtime segmentation resolution; native room checks take precedence.',
        }, {
            'doubt': 'Legacy timestep configuration and QA labels do not certify current native motion; friction '
                'application errors are swallowed by the active source builder.',
            'chosen_assumption': 'Preserve current loader physics settings and original friction; require measured '
                'mesh friction and native clock. Score a one-second real placement/release trial independently of '
                'its stable/sliding/falling outcome, retaining legacy QA privately.',
            'alternatives': ['Treat native material or time measurement failure as a qualification failure.',
                             'Request a separately controlled generator-protocol reproduction before comparing QA.'],
            'scope': 'All inclined-plane operation predicates; no change to original simulation frequency or mass.',
            'risk': 'The one-second window is an explicit new operation protocol and may disagree with legacy labels. '
                'AABB_min_z < floor_top + 0.05 m can classify low objects as fallen while still on the ramp; '
                'report that legacy screen separately from native contact and stability.',
        }]
        warnings = ['assumption_review_required', 'native_placement_unverified', 'native_floor_reach_unverified',
                    'native_friction_qualification_required', 'legacy_dynamics_protocol_not_reproduced',
                    'pickup_initial_xy_adaptation_requires_review']
        nominal_floor_root_inside_slope_xy = all(payload['slope']['aabb_min'][axis] <=
            obj['pos_on_floor'][axis] <= payload['slope']['aabb_max'][axis] for axis in (0, 1))
        if nominal_floor_root_inside_slope_xy:
            warnings.append('source_nominal_floor_start_overlaps_slope')
            assumptions.append({
                'doubt': 'The saved pos_on_floor root lies inside the original slope XY AABB; the field name '
                    'does not establish that this object is actually supported by the floor.',
                'chosen_assumption': 'Use the accepted initialized pickup reference on a measured accessible '
                    'support, allowing horizontal relocation away from the occupied ramp footprint. Require '
                    'a fresh held lift and real placement/release on the unchanged original slope.',
                'alternatives': ['Keep a diagnostic if no native supported pickup start qualifies.',
                                 'Seek original generator-frame evidence before interpreting the nominal floor phase.'],
                'scope': source.relative_path + ': initial support qualification only.',
                'risk': 'A source label or AABB overlap alone cannot identify support. The original geometry may '
                    'settle differently; no robot task is qualified until actual native contact is measured.',
            })
        expected_qa = 'no' if payload.get('ground_truth_slid') or payload.get('ground_truth_fallen') else 'yes'
        if raw.get('answer') != expected_qa:
            warnings.append('source_qa_outcome_annotation_conflict')
        result = {'scene_definition': original_scene(source, context, bindings=bindings),
                  'task_definition': None, 'reuse_level': 'L2', 'status': 'generated',
                  'warning_codes': warnings, 'assumptions': assumptions,
                  'change_summary': ['Retain the original ramp/object/floor, geometry, friction, observers and private QA.',
                                     'Add an independent real lift/place/release trial with native contact and timed motion evidence.'],
                  'private_source_evidence': {'robot_start': robot, 'original_slope_angle_deg': payload.get('slope_angle_deg'),
                      'original_answer': raw.get('answer'), 'original_outcome_flags': {key: payload.get(key) for key in (
                          'ground_truth_slid', 'ground_truth_fallen')}, 'source_floor_z_estimate_m': floor_z,
                      'source_nominal_floor_root_inside_slope_xy': nominal_floor_root_inside_slope_xy,
                      'native_dynamics_executed_by_converter': False},
                  'limitations': ['CPU generation and static map screens do not establish native load, reach, grasp or placement.',
                                  'One-second native release trace differs from an unverified legacy 30-step generator horizon.',
                                  'Legacy put_on_slope and original QA are auxiliary and never counted as robot success.'],
                  'attempted_repairs': [],
                  'next_step': 'Publish pending candidates and failures for Review; qualify measured native initialization '
                      'and execute only with existing guarded robot controls.'}
        if robot is None:
            warnings.append('source_start_native_measurement_required')
            assumptions[0]['chosen_assumption'] = ('Retain a pending task with no guessed robot pose. '
                'The bounded CPU map search is inconclusive; the shared native initializer must independently '
                'measure a floor-supported collision-free start before accepting any support adaptation.')
        patches = [{'operation': 'object_physics', 'binding': 'target', 'visual_only': False},
                   {'operation': 'object_physics', 'phase': 'after_postprocessing', 'binding': 'target', 'visual_only': False}]
        if robot is not None:
            patches.insert(0, {'operation': 'robot_pose', 'position': robot['position'], 'yaw_deg': robot['yaw_deg'],
                'head_pitch_deg': 45., 'heading_offsets_deg': [0.], 'jitter_xy_m': [0., 0.]})
        parameters = {'slope_position': payload['slope']['position'], 'slope_quaternion': payload['slope_quaternion'],
                      'slope_scale': payload['slope_scale'], 'target_floor_position': obj['pos_on_floor'],
                      'floor_z_m': floor_z, 'static_friction': payload['static_friction'],
                      'dynamic_friction': payload['dynamic_friction']}
        public = {'objective': f'Grasp the {label} at its initialized starting support, lift it clear of that support, '
                      'carry it to the original slope, '
                      'lower it onto the upper face until supported, then open the gripper and withdraw. '
                      'Leave it untouched for at least one second to observe whether it stays, slides or falls. '
                      'Original auxiliary question: ' + raw['question'],
                  'success_criteria': 'Complete a real grasp with at least 5 cm lift, establish support on the slope '
                      'before releasing, and record at least one second of uninterrupted unheld motion after release. '
                      'Keep the robot and other objects from touching it during observation. Stable, sliding and falling '
                      'outcomes all count as completed trials; the original yes/no answer is auxiliary.',
                  'environment': 'Original room, object, inclined plane, friction, scale and reference view are retained.',
                  'constraints': ['Use only existing guarded base, arm and gripper controls.',
                                  'Keep the original slope, its supports and task geometry unchanged. Noncritical native movable furniture may be moved using guarded arm and gripper actions.',
                                  'Do not use the legacy put_on_slope shortcut; physically carry and release the object.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
                      'binding_keys': ['target', 'slope', 'floor'], 'parameters': parameters}
        evaluation['parameters'].update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        result.update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
                      task_definition=physical_task(source, context, public=public, evaluation=evaluation,
                          bindings=bindings, patches=patches, capabilities=['base_translation', 'base_rotation',
                              'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']))
        result['change_summary'].append('Apply a reversible source-derived R1 pose and enable original target physics '
            'before construction and after source postprocessing; do not change fixed_base, scale or physics frequency.')
        result['change_summary'].append('Allow initialization-only XYZ translation of the pickup target to a '
            'measured accessible same-room support; initial target XY is not a ramp-trial invariant. '
            'Retain the fixed ramp, source floor, friction and post-release outcome reference unchanged.')
        return result

    def convert_all(self, sources, context):
        """Use shared immutable candidate identity, schemas and per-source failure capture."""
        return super().convert_all(sorted(sources, key=lambda source: source.relative_path), context)

    def sample(self, manifest, policy=None):
        """Stratify failures plus native asset size, angle, room and private legacy outcome."""
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        rows = manifest['records']
        chosen = []
        failures = sorted((row for row in rows if row['status'] != 'generated'), key=lambda row: row['candidate_id'])
        if failures:
            chosen.append(failures[0]['candidate_id'])
        previous = set(policy.get('previous_failures', ()))
        for row in sorted(rows, key=lambda row: row['candidate_id']):
            if row['candidate_id'] in previous and row['candidate_id'] not in chosen and len(chosen) < count:
                chosen.append(row['candidate_id'])
        generated = [row for row in rows if row['status'] == 'generated']
        # Selection uses private construction evidence; it is never an Agent sensor.
        if generated:
            angle = lambda row: row['private_source_evidence']['original_slope_angle_deg']
            for row in (min(generated, key=angle), max(generated, key=angle)):
                if row['candidate_id'] not in chosen and len(chosen) < count:
                    chosen.append(row['candidate_id'])
        base = super().sample(manifest, {**policy, 'count': len(rows)})
        by_id = {row['candidate_id']: row for row in rows}
        strata = {(by_id[key].get('scene_model'), by_id[key].get('question')) for key in chosen}
        for key in base:
            row = by_id[key]
            stratum = (row.get('scene_model'), row.get('question'))
            if len(chosen) < count and key not in chosen and stratum not in strata:
                chosen.append(key)
                strata.add(stratum)
        return (chosen + [key for key in base if key not in chosen])[:count]


def measure_source_apparatus(session, target, slope, parameters):
    """Read pre-translation source physics for root's native setup integration."""
    rows = {}
    for key, obj in (('target', target), ('slope', slope)):
        rows[key] = {**_pose(obj), 'name': obj.name, 'category': obj.category,
            'model': getattr(obj, 'model', None), 'mass_kg': float(obj.mass), 'materials': _friction(obj)}
    proposal = parameters['apparatus_proposal']
    envelope = materialize_apparatus(proposal, pickup_bounds=[rows['target']['aabb_min'], rows['target']['aabb_max']])
    return {'schema': NATIVE_RECEIPT, 'source_identity': parameters['source_identity'],
        'source_sha256': parameters['source_sha256'], 'source_baseline': rows, 'envelope': envelope,
        'source_camera_position_m': _array(session.scene_camera_position).tolist(),
        'source_camera_quaternion_xyzw': _array(session.scene_camera_quaternion).tolist()}


def _materials_match(current, expected):
    return (len(current) == len(expected) and bool(current) and all(
        first.get('mesh') == second.get('mesh') and all(abs(first[key] - second[key]) <= 1e-4
            for key in ('static_friction', 'dynamic_friction')) for first, second in zip(current, expected)))


def _resolve_apparatus(session, parameters, target, slope):
    """Resolve the exact accepted support; source original poses remain provenance."""
    setup = getattr(session, 'category_interaction_setup', {})
    delta = common_translation(setup, target_name=target.name, slope_name=slope.name)
    receipt = setup.get('inclined_apparatus', {})
    if (receipt.get('schema') != NATIVE_RECEIPT or receipt.get('source_identity') != parameters['source_identity'] or
            receipt.get('source_sha256') != parameters['source_sha256']):
        raise ValueError('Exact source-bound native apparatus physics receipt is unavailable.')
    if (not np.allclose(_array(session.scene_camera_position),
            _array(receipt['source_camera_position_m']) + delta, atol=1e-5, rtol=0) or
            not np.allclose(_array(session.scene_camera_quaternion), receipt['source_camera_quaternion_xyzw'], atol=1e-5, rtol=0)):
        raise ValueError('Current source reference camera does not preserve the accepted common translation.')
    baseline = receipt['source_baseline']
    if (set(baseline) != {'target', 'slope'} or baseline['target']['name'] != target.name or
            baseline['slope']['name'] != slope.name or
            any(baseline['target'][key] != parameters['target_identity'][key] for key in ('category', 'model'))):
        raise ValueError('Native source apparatus identity differs from the source task.')
    if (baseline['target']['mass_kg'] <= 0 or any(not math.isfinite(row['mass_kg']) or row['mass_kg'] < 0 or not row['materials']
            for row in baseline.values())):
        raise ValueError('Pre-translation native mass/material measurements are missing.')
    # Source-native initialization is measured independently from saved trial images.
    if (np.linalg.norm(np.asarray(baseline['target']['position']) - parameters['target_floor_position']) > .05 or
            not _slope_unchanged(baseline['slope'], parameters)):
        raise ValueError('Pre-translation native apparatus does not preserve original source poses.')
    expected = materialize_apparatus(parameters['apparatus_proposal'],
        pickup_bounds=[baseline['target']['aabb_min'], baseline['target']['aabb_max']])
    if receipt.get('envelope') != expected:
        raise ValueError('The required apparatus envelope differs from exact native/source measurements.')
    support = session.env.scene.object_registry('name', setup.get('support_name'))
    if support is None or support.prim_path != setup['support_prim_path'] or support.visual_only:
        raise ValueError('Selected real apparatus deck is not available.')
    if support is target or support is slope or support.category in ('floor', 'floors'):
        raise ValueError('The apparatus requires an elevated selected support, separate from the building floor.')
    bounds = [_array(v).tolist() for v in support.aabb]
    fit = support_envelope_fit(expected, bounds, delta, support_surface_z_m=setup.get('support_surface_z_m'))
    if not fit['accepted']:
        raise ValueError('Selected deck does not contain the complete required runoff proposal: ' + str(fit['reason']))
    # Root stores the post-placement selected support snapshot; compare it live.
    support_reference = receipt.get('support_reference')
    if not support_reference:
        raise ValueError('Post-placement native support reference is missing.')
    return {'support': support, 'translation_m': delta.tolist(), 'source_baseline': baseline,
        'support_reference': support_reference, 'envelope': expected, 'fit': fit,
        'surface_z_m': float(setup['support_surface_z_m'])}


def _inside_xy(row, bounds):
    low, high = np.asarray(bounds[0]), np.asarray(bounds[1])
    return bool(np.all(np.asarray(row['aabb_min'])[:2] >= low[:2] - .005) and
        np.all(np.asarray(row['aabb_max'])[:2] <= high[:2] + .005))


def _measure_apparatus_trial(row, target, slope, floor, apparatus):
    from omnigibson.object_states import Touching
    support, surface = apparatus['support'], apparatus['surface_z_m']
    deck = _pose(support)
    reference = apparatus['support_reference']
    preserved = all(np.allclose(deck[key], reference[key], atol=1e-5, rtol=0)
        for key in ('position', 'quaternion_xyzw', 'scale', 'aabb_min', 'aabb_max'))
    preserved = bool(preserved and deck['fixed_base'] == reference['fixed_base'] and
        not deck['visual_only'] and _static_or_slow(deck))
    deck_contacts, interference, original_floor_contacts = [], [], []
    for contact in target.contact_list():
        external = [str(body) for body in (contact.body0, contact.body1)
            if not (str(body) == target.prim_path or str(body).startswith(target.prim_path + '/'))]
        if not external:
            continue
        belongs = lambda obj: all(body == obj.prim_path or body.startswith(obj.prim_path + '/') for body in external)
        if belongs(slope):
            continue
        if belongs(floor):
            original_floor_contacts.append(external)
            interference.append(external)
        elif belongs(support):
            position, normal = _array(contact.position), _array(contact.normal)
            if abs(position[2] - surface) <= .015 and abs(normal[2]) >= .5:
                deck_contacts.append(position.tolist())
            else:
                interference.append(external)
        else:
            interference.append(external)
    touching_deck = bool(target.states[Touching].get_value(support) and deck_contacts)
    inside_deck = _inside_xy(row, [deck['aabb_min'], deck['aabb_max']])
    inside_envelope = _inside_xy(row, apparatus['fit']['translated_required_bounds_m'])
    original_floor = bool(row['touching_floor'] or original_floor_contacts)
    escaped = bool(original_floor or not inside_deck or row['aabb_max'][2] < surface - .005)
    baseline = apparatus['source_baseline']['target']
    physics = bool(np.allclose(row['scale'], baseline['scale'], atol=1e-6, rtol=0) and
        math.isclose(float(target.mass), baseline['mass_kg'], abs_tol=1e-6, rel_tol=0) and
        _materials_match(_friction(target), baseline['materials']))
    row.update(interfering_contacts=interference, touching_original_floor=original_floor,
        touching_apparatus_deck=touching_deck, apparatus_deck_contacts=deck_contacts,
        within_selected_deck=inside_deck, within_required_envelope=inside_envelope,
        escaped_apparatus=escaped, apparatus_support_preserved=preserved,
        source_target_physics_preserved=physics, apparatus_surface_z_m=surface)


def _qualify_apparatus(session, executor, config, bound):
    checks = {}
    try:
        parameters = _configuration(session, config)
        target, slope, floor = [bound[key] for key in ('target', 'slope', 'floor')]
        apparatus = _resolve_apparatus(session, parameters, target, slope)
        translated = {**parameters, 'apparatus_translation_m': apparatus['translation_m']}
        row, ramp = _trial_state(target, slope, floor, translated, apparatus=apparatus)
        baseline = apparatus['source_baseline']
        delta = np.asarray(apparatus['translation_m'])
        robot, gripper = _robot_state(session), executor.debug_gripper_state()
        control_evidence = _trial_control_evidence(executor)
        materials = {'target': _friction(target), 'slope': _friction(slope)}
        checks = {'native_public_control_history_available': control_evidence['available'],
            'no_episode_simulator_setters': control_evidence['available'] and control_evidence['state_setter_used'] is False,
            'native_common_translation_and_full_envelope': apparatus['fit']['accepted'],
            'original_target_movable_collidable': not row['visual_only'] and not row['fixed_base'],
            'source_pickup_pose_common_translation': bool(np.allclose(row['position'],
                np.asarray(baseline['target']['position']) + delta, atol=.005, rtol=0) and
                np.allclose(row['quaternion_xyzw'], baseline['target']['quaternion_xyzw'], atol=1e-5, rtol=0)),
            'slope_pose_fixed_and_collidable': row['slope_unchanged'],
            'source_target_physics_preserved': row['source_target_physics_preserved'],
            'source_friction_preserved': all(_materials_match(materials[key], baseline[key]['materials']) for key in materials) and
                all(abs(material[name] - parameters[name]) <= 1e-4 for rows in materials.values()
                    for material in rows for name in ('static_friction', 'dynamic_friction')),
            'selected_deck_preserved': row['apparatus_support_preserved'],
            'ramp_currently_supported_on_selected_deck': bool(touching_task_support(session, slope, floor) and on_top_task_support(session, slope, floor)),
            'pickup_currently_supported_on_apparatus': bool(row['top_supported'] or
                (row['touching_apparatus_deck'] and on_top_task_support(session, target, floor))),
            'no_initial_interference_or_escape': not row['interfering_contacts'] and not row['escaped_apparatus'],
            'original_floor_unchanged': not floor.visual_only and abs(float(_array(floor.aabb[1])[2]) - parameters['floor_z_m']) <= .01,
            'target_contact_reporting_available': any(link.contact_reporting_enabled for link in target.links.values()),
            'target_stationary': row['linear_speed_m_s'] <= MAXIMUM_LINEAR_SPEED and row['angular_speed_rad_s'] <= MAXIMUM_ANGULAR_SPEED,
            'initial_gripper_empty': gripper.get('object_name') is None and gripper.get('constraint_active') is False,
            'robot_supported_in_source_room': robot['in_original_room'] and bool(robot['floor_contacts']) and not robot['obstacle_contacts']}
        accepted = all(checks.values())
        return {'accepted': accepted, 'reason': None if accepted else 'native_whole_apparatus_precondition_failed',
            'checks': checks, 'native_target': row, 'native_slope': ramp, 'native_materials': materials,
            'native_robot': robot, 'native_mass_kg': float(target.mass),
            'native_floor_top_m': float(_array(floor.aabb[1])[2]), 'apparatus_trial_surface_z_m': apparatus['surface_z_m'],
            'apparatus_envelope': apparatus['fit'], 'source_baseline': baseline, 'native_control_evidence': control_evidence,
            'initial_target_reference_m': (np.asarray(baseline['target']['position']) + delta).tolist(),
            'real_robot_placement_qualified': False, 'one_second_runoff_native_validated': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_whole_apparatus_measurement_unavailable', 'checks': checks,
            'exception_summary': f'{type(error).__name__}: {error}', 'real_robot_placement_qualified': False}


def _convert_apparatus(converter, source, context):
    """Version only the authorized common-translation apparatus variant."""
    legacy_context = copy.copy(context)
    legacy_context.config = {**context.config, 'task_edit2_inclined': False}
    result = converter.convert(source, legacy_context)
    raw, payload, obj, label, floor_z = _source_data(source)
    proposal = source_apparatus_proposal(payload)
    task = result['task_definition']
    evaluation = task['private']['evaluation']
    parameters = evaluation['parameters']
    parameters.update(task_revision=APPARATUS_REVISION, apparatus_proposal=proposal,
        source_identity=source.relative_path, source_sha256=sha256(source.path),
        target_identity={'category': obj['category'], 'model': obj['model']},
        original_source_camera_poses=copy.deepcopy(payload['camera_poses']),
        source_target_reference={key: copy.deepcopy(obj[key]) for key in ('pos_on_floor', 'pos_init', 'pos_final',
            'aabb_min_init', 'aabb_max_init', 'aabb_min_final', 'aabb_max_final')})
    evaluation['operation'] = APPARATUS_OPERATION
    contract = {'preserve': ['target_identity_scale_mass_material', 'ramp_angle_quaternion_scale_friction',
        'target_pickup_to_ramp_common_translation', 'source_camera_common_translation'],
        'goal': 'native_supported_release_and_safe_apparatus_trial', 'intended_motion': ['sliding', 'rolling', 'safe_runoff_on_deck'],
        'safety_context': 'Escape from the selected apparatus onto the building floor is a failed safe trial.'}
    parameters['mechanism_contract'] = contract
    task['public'] = {'objective': f'Grasp the {label} from its initialized pickup on the elevated complete ramp apparatus, '
        'lift it clear, lower it onto the ramp upper face until supported, release it and withdraw. '
        'Observe one uninterrupted second of stable, sliding or rolling motion; use '
        'the full selected apparatus deck as the safe runoff region. Original auxiliary question: ' + raw['question'],
        'success_criteria': 'Perform a real grasp and at least 5 cm lift, supported top-face release, and one second '
        'without robot/other-object interference while the target remains on the complete apparatus. Intended slide, roll '
        'and runoff onto its safe deck may complete the trial. Escaping the deck or landing on the building floor fails. '
        'The auxiliary yes/no answer never substitutes for the physical trial.',
        'environment': 'The original target pickup and ramp move together onto one measured support with their source '
        'geometry, scale, mass, angle and materials retained. Source reference cameras share that translation. '
        'Robot approach, grasp access and safe runoff require native qualification.',
        'constraints': ['Use only guarded base, arm and gripper controls; do not use legacy put_on_slope or simulator setters.',
            'Preserve the complete ramp apparatus and the named safe runoff region.']}
    from wia.authoring.converters.interaction_profiles import policy_for
    policy = policy_for(source, task)
    policy.update(group_bindings=['target', 'slope'], manipulation_bindings=['target'], support_bindings=['target', 'slope'],
        relocation_policy='rigid_group', source_camera_policy='translate_with_group',
        layout_parameters={'inclined_apparatus': proposal},
        warning_codes=['native_whole_apparatus_support_pending', 'native_pickup_bounds_and_physics_required',
            'native_runoff_and_release_pending'])
    result.update(interaction_policy=policy, task_revision=APPARATUS_REVISION, mechanism_contract=contract,
        apparatus_proposal=proposal, assumptions=[{'doubt': 'Source records lack pickup AABB and native scale/mass/material measurements; '
            'saved trial endpoints do not certify a new one-second runoff horizon.',
            'chosen_assumption': 'Measure pre-translation native pickup and physics, preserve the complete target/ramp '
            'relationship with one XYZ translation, and require the entire measured support/runoff envelope. '
            'Keep unavailable when no support within existing limits fits; never shrink clearance or repack the pickup.',
            'alternatives': ['Retain a source diagnostic when native support, pickup contact or release cannot qualify.'],
            'scope': source.relative_path, 'risk': 'Native pickup can overlap the ramp; no physical access is promised.'}],
        warning_codes=['native_whole_apparatus_support_pending', 'native_pickup_contact_feasibility_pending',
            'source_30_steps_do_not_certify_one_second_runoff', 'original_default_target_physics_native_receipt_required'],
        change_summary=['Elevate target pickup and ramp by one common XYZ translation; keep the original building floor.',
            'Require complete deck/runoff envelope and separate safe intended motion from escape to the original floor.'],
        limitations=['Native source pickup bounds, exact scale/mass/materials and support fit are pending.',
            'Original source endpoints plus one measured diameter form a support proposal, not a one-second guarantee.',
            'No native loading, robot approach, grasp, release or runoff has been executed by this converter.'])
    return result


def _trial_control_evidence(controller):
    """Use root's existing native action callback; missing evidence is unavailable."""
    callback = getattr(controller, 'geometric_rearrangement_evidence', None)
    try:
        value = callback() if callable(callback) else {}
        available = value.get('available') is True and type(value.get('state_setter_used')) is bool
        return {'available': available, 'state_setter_used': value.get('state_setter_used'),
            'native_sample_id': value.get('sample_id'), 'public_control_sequence': value.get('control_sequence'),
            'reason': None if available else 'native_public_control_and_setter_evidence_unavailable'}
    except Exception as error:
        return {'available': False, 'state_setter_used': None,
            'reason': f'native_public_control_evidence_unavailable: {type(error).__name__}: {error}'}

"""Convert original three-object scenes into guarded, physically measured stacks.

Run CPU batching from the repository root with the behavior environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
The campaign owns native captures; this module never starts GPU work or actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_grasp

from collections import Counter
from functools import lru_cache
from itertools import permutations
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
    reference_position, touching_task_support, on_top_task_support, support_contact_paths,
)


MODULE = 'wia.authoring.converters.physical_dynamics_stacking_stability'
TASK_REVISION = 'task_edit_20261018_v1'
MECHANISM_CONTRACT = {'preserve': ['original_objects', 'asset_scales', 'native_grasp_release_history'], 'goal': 'released_tower_after_robot_withdrawal', 'stability_seconds': 1.0}
OPERATION = 'build_released_three_object_stack'
FINAL_STATE_REVISION = 'final_state_v1'
OBJECT_KEYS = ('object_0', 'object_1', 'object_2')
MINIMUM_LIFT_M = .05
STABILITY_SECONDS = 1.
MINIMUM_STABILITY_SAMPLES = 20
MAXIMUM_SAMPLE_GAP_S = .1
MAXIMUM_LINEAR_SPEED = .02
MAXIMUM_ANGULAR_SPEED = .1
MAXIMUM_DRIFT_M = .01
MAXIMUM_ROTATION_RAD = math.radians(3.)
MINIMUM_UP_DOT = .9
ROBOT_SCREEN_RADIUS_M = .45
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


def _source_data(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('source_category_mismatch')
    if tuple(Path(source.relative_path).parts[-3:-1]) != (raw.get('scene'), raw.get('room')):
        raise ValueError('source_scene_room_hierarchy_mismatch')
    if raw.get('runner_task') != 'stacking' or payload.get('n_objects') != 3:
        raise ValueError('unsupported_original_stacking_task')
    names, categories, models, scales = [payload.get(key) for key in (
        'object_names', 'object_categories', 'object_models', 'object_scales')]
    if any(not isinstance(value, list) or len(value) != 3 for value in (names, categories, models, scales)):
        raise ValueError('expected_three_original_assets')
    if any(not isinstance(value, str) or not value for value in [*names, *categories, *models]):
        raise ValueError('original_asset_identity_missing')
    if len(set(names)) != 3 or len(set(categories)) != 3:
        raise ValueError('original_object_identity_ambiguous')
    poses = map_records(payload.get('initial_poses'))
    for name, scale in zip(names, scales):
        if not _vector(scale, 3) or min(scale) <= 0:
            raise ValueError('invalid_original_scale:' + name)
        pose = poses.get(name, {})
        if any(not _vector(pose.get(key), size) for key, size in (
                ('position', 3), ('quaternion_xyzw', 4), ('aabb_min', 3), ('aabb_max', 3))):
            raise ValueError('missing_original_pose_or_bounds:' + name)
        if not any(pose['quaternion_xyzw']) or any(a >= b for a, b in zip(pose['aabb_min'], pose['aabb_max'])):
            raise ValueError('invalid_original_pose_or_bounds:' + name)
    if not payload.get('floor'):
        raise ValueError('missing_original_floor')
    cameras = map_records(payload.get('initial_cam_poses'))
    if not cameras or any(not _vector(camera.get('position'), 3) or not _vector(
            camera.get('quaternion_xyzw'), 4) for camera in cameras.values()):
        raise ValueError('missing_original_reference_observer')
    match = re.fullmatch(r'How should you stack (.+) to maximize stability\?', raw.get('question', '').strip())
    labels = [value.strip().replace(' ', '_') for value in match[1].split(',')] if match else []
    if sorted(labels) != sorted(categories):
        raise ValueError('public_source_categories_disagree')
    # Preserve the same fallback order set used by source get_gt_orders().
    by_name = dict(zip(names, categories))
    orders = payload.get('_ground_truth') or [
        [by_name.get(name) for name in trial.get('order', [])]
        for trial in payload.get('trials', []) if trial.get('all_stable')]
    if not orders or any(not isinstance(order, list) or sorted(order) != sorted(categories) for order in orders):
        raise ValueError('missing_or_invalid_legacy_accepted_orders')
    return raw, payload, poses, orders


@lru_cache(maxsize=2)
def _maps(asset_root, scene_name):
    """Read original 1 cm maps; match native Map and SegmentationMap indexing."""
    import cv2
    cv2.setNumThreads(2)
    root = Path(asset_root)
    paths = [root / 'scenes' / scene_name / 'layout' / name for name in (
        'floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    categories_path = root / 'metadata/room_categories.txt'
    if not all(path.is_file() for path in [*paths, categories_path]):
        return None
    trav, instances, semantics = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(value is None for value in (trav, instances, semantics)) or not (
            trav.shape == instances.shape == semantics.shape and trav.shape[0] == trav.shape[1]):
        return None
    categories = categories_path.read_text().splitlines()
    ids, counts = {}, Counter()
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        row, col = np.argwhere(instances == instance_id)[0]
        semantic_id = int(semantics[row, col])
        if not 0 < semantic_id <= len(categories):
            continue
        name = categories[semantic_id - 1]
        ids[f'{name}_{counts[name]}'] = int(instance_id)
        counts[name] += 1
    return trav, instances, ids, {str(path): sha256(path) for path in [*paths, categories_path]}


def _asset_root(context):
    data_root = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    return str(context.config.get('original_asset_root', data_root / 'behavior-1k-assets'))


def _robot_start(raw, payload, poses, context):
    maps = _maps(_asset_root(context), raw['scene'])
    evidence = {'map_asset_root': _asset_root(context), 'screen_radius_m': ROBOT_SCREEN_RADIUS_M,
                'map_resolution_m': MAP_RESOLUTION_M, 'maximum_search_radius_m': 1.5,
                'native_qualified': False}
    if maps is None or raw['room'] not in maps[2]:
        return None, {**evidence, 'failure_reason': 'original_maps_or_room_unavailable'}
    trav, rooms, room_ids, hashes = maps
    centre = np.mean([poses[name]['position'][:2] for name in payload['object_names']], axis=0)
    candidates = [(key, np.asarray(camera['position'][:2])) for key, camera in
                  map_records(payload['initial_cam_poses']).items() if camera.get('type') == 'side']
    for radius in (.75, .9, 1.05, 1.2, 1.5):
        for angle in range(0, 360, 15):
            radians = math.radians(angle)
            candidates.append((f'original_cluster_ring_{radius}_{angle}', centre + radius * np.array(
                [math.cos(radians), math.sin(radians)])))
    pixel_radius = math.ceil(ROBOT_SCREEN_RADIUS_M / MAP_RESOLUTION_M)
    offsets = np.arange(-pixel_radius, pixel_radius + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= pixel_radius ** 2
    rejections, valid = Counter(), []
    for label, xy in candidates:
        row, col = (xy[::-1] / MAP_RESOLUTION_M + trav.shape[0] / 2).astype(int)
        if min(row, col) < pixel_radius or max(row, col) + pixel_radius >= trav.shape[0]:
            rejections['map_bounds'] += 1
            continue
        region = (slice(row - pixel_radius, row + pixel_radius + 1),
                  slice(col - pixel_radius, col + pixel_radius + 1))
        if not np.all(trav[region][disk] > 0) or not np.all(rooms[region][disk] == room_ids[raw['room']]):
            rejections['room_or_furniture_clearance'] += 1
            continue
        clearances = [float(np.linalg.norm(xy - np.clip(xy, poses[name]['aabb_min'][:2],
                       poses[name]['aabb_max'][:2]))) for name in payload['object_names']]
        if min(clearances) < ROBOT_SCREEN_RADIUS_M + .05:
            rejections['original_object_clearance'] += 1
            continue
        distances = [float(np.linalg.norm(xy - poses[name]['position'][:2])) for name in payload['object_names']]
        valid.append(((max(distances), sum(distances), label), xy, label, clearances, distances))
    evidence.update(map_sha256=hashes, original_room_id=room_ids[raw['room']],
                    screened_count=len(candidates), feasible_count=len(valid), rejections=dict(rejections))
    if not valid:
        return None, {**evidence, 'failure_reason': 'no_collision_screened_original_room_start'}
    _, xy, label, clearances, distances = min(valid, key=lambda item: item[0])
    floor_z = float(np.median([poses[name]['aabb_min'][2] for name in payload['object_names']]))
    robot = {'position': [float(xy[0]), float(xy[1]), floor_z],
             'yaw_deg': math.degrees(math.atan2(centre[1] - xy[1], centre[0] - xy[0])),
             'selection': label, 'object_aabb_clearances_m': clearances,
             'object_centre_distances_m': distances}
    return robot, evidence


def _native_config(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') != OPERATION or config.get('scene') != session.scene_name or
            config.get('room') != session.room_name or set(config.get('bindings', {})) != {*OBJECT_KEYS, 'floor'}):
        raise ValueError('Stack operation requires its original scene and four native bindings.')
    parameters = config['parameters']
    if parameters.get('manipulation_revision') not in (None, FINAL_STATE_REVISION):
        raise ValueError('Unsupported tower manipulation revision.')
    return parameters


def _pose(obj):
    position, quaternion = [_array(value) for value in obj.get_position_orientation()]
    low, high = [_array(value) for value in obj.aabb]
    return {'name': obj.name, 'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(), 'scale': _array(obj.scale).tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'up_dot': float(Rotation.from_quat(quaternion).as_matrix()[2, 2]),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}


def _external_contacts(obj):
    return [[str(body) for body in (contact.body0, contact.body1) if not (
        str(body) == obj.prim_path or str(body).startswith(obj.prim_path + '/'))]
        for contact in obj.contact_list()]


def _belongs(body, obj):
    return body == obj.prim_path or body.startswith(obj.prim_path + '/')


def _gripper_state(session, controller):
    state = dict(controller.debug_gripper_state())
    robot = session.robot
    state['all_arm_held_names'] = {arm: obj.name if obj is not None else None
                                  for arm, obj in robot._ag_obj_in_hand.items()}
    state['all_arm_constraints'] = {arm: value is not None for arm, value in robot._ag_obj_constraints.items()}
    state['all_arm_release_pending'] = {arm: value is not None for arm, value in robot._ag_release_counter.items()}
    state['all_arms_free'] = bool(all(value is None for value in state['all_arm_held_names'].values()) and
                                 not any(state['all_arm_constraints'].values()) and
                                 not any(state['all_arm_release_pending'].values()))
    return state


def _robot_start_checks(session):
    robot, scene = session.robot, session.env.scene
    low, high = [_array(value) for value in robot.links[robot.base_footprint_link_name].aabb]
    rooms = [scene.seg_map.get_room_instance_by_point(np.array([x, y]))
             for x in (low[0], (low[0] + high[0]) / 2, high[0])
             for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj for obj in objects if obj.category in ('floor', 'floors')]
    external = [bodies for bodies in _external_contacts(robot) if bodies]
    floor_contacts = [bodies for bodies in external if all('groundPlane' in body or any(
        _belongs(body, floor) for floor in floors) for body in bodies)]
    return {'in_original_room': all(room == session.room_name for room in rooms),
            'has_floor_contact': bool(floor_contacts), 'no_obstacle_contact': len(external) == len(floor_contacts),
            'footprint_rooms': rooms, 'external_contacts': external}


def qualify_operation(session, executor, config, bound):
    """Measure initial support, identity, geometry and collision state without actions."""
    checks = {}
    try:
        from omnigibson.object_states import OnTop, Touching
        parameters = _native_config(session, config)
        floor = bound['floor']
        rows = {key: _pose(bound[key]) for key in OBJECT_KEYS}
        gripper, robot = _gripper_state(session, executor), _robot_start_checks(session)
        for key, row in rows.items():
            original = parameters['original_objects'][key]
            obj = bound[key]
            checks[key + '_movable_collidable'] = not row['visual_only'] and not row['fixed_base']
            checks[key + '_identity_scale_preserved'] = bool(obj.name == original['name'] and
                obj.category == original['category'] and obj.model == original['model'] and
                np.max(np.abs(np.asarray(row['scale']) - original['scale'])) <= 1e-5)
            checks[key + '_initial_pose_preserved'] = bool(
                np.linalg.norm(np.asarray(row['position']) -
                    reference_position(session, obj.name, original['position'])) <= .02 and
                (Rotation.from_quat(row['quaternion_xyzw']).inv() * Rotation.from_quat(
                    original['quaternion_xyzw'])).magnitude() <= math.radians(3.))
            row['touching_original_floor'] = bool(obj.states[Touching].get_value(floor))
            row['on_top_original_floor'] = bool(obj.states[OnTop].get_value(floor))
            checks[key + '_native_task_support'] = bool(on_top_task_support(session, obj, floor) and
                                                        touching_task_support(session, obj, floor))
            checks[key + '_stationary_upright'] = bool(row['up_dot'] >= MINIMUM_UP_DOT and
                row['linear_speed_m_s'] <= MAXIMUM_LINEAR_SPEED and row['angular_speed_rad_s'] <= MAXIMUM_ANGULAR_SPEED)
        checks.update(original_floor_collidable=not floor.visual_only,
                      initial_gripper_empty=gripper['all_arms_free'],
                      robot_supported_in_original_room=robot['in_original_room'] and robot['has_floor_contact'] and
                          robot['no_obstacle_contact'])
        checks['initial_objects_separate'] = all(not bound[a].states[Touching].get_value(bound[b])
                                                 for a, b in permutations(OBJECT_KEYS, 2))
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'reason': None if accepted else 'native_stack_precondition_failed',
                'checks': checks, 'native_objects': rows, 'native_robot': robot,
                'native_gripper': gripper, 'physical_stack_executed': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_stack_measurement_unavailable', 'checks': checks,
                'exception_summary': f'{type(error).__name__}: {error}', 'physical_stack_executed': False}


def _native_time():
    import omnigibson as og
    now, dt = float(og.sim.current_time), float(og.sim.get_sim_step_dt())
    if not math.isfinite(now) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Native clock and sampling dt are unavailable.')
    return now, dt


class StackOperationEvaluator:
    """Read real lift/release histories and independent consecutive stack evidence."""
    def __init__(self, session, controller, config, output):
        self.parameters = _native_config(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.bound.values()):
            raise ValueError('Original stack binding is missing.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.initial, self.last = 0, None, None

    def measure(self):
        from omnigibson.object_states import OnTop, Touching
        rows = {key: _pose(self.bound[key]) for key in OBJECT_KEYS}
        on_top, touching = {}, {}
        for upper in OBJECT_KEYS:
            obj = self.bound[upper]
            rows[upper]['touching_original_floor'] = bool(obj.states[Touching].get_value(self.bound['floor']))
            rows[upper]['on_top_original_floor'] = bool(obj.states[OnTop].get_value(self.bound['floor']))
            rows[upper]['external_contacts'] = [bodies for bodies in _external_contacts(obj) if bodies]
            for lower in (*OBJECT_KEYS, 'task_support'):
                if upper == lower:
                    continue
                pair = (upper, lower)
                if lower == 'task_support':
                    on_top[pair] = on_top_task_support(self.session, obj, self.bound['floor'])
                    touching[pair] = touching_task_support(self.session, obj, self.bound['floor'])
                else:
                    on_top[pair] = bool(obj.states[OnTop].get_value(self.bound[lower]))
                    touching[pair] = bool(obj.states[Touching].get_value(self.bound[lower]))
        return rows, on_top, touching, _gripper_state(self.session, self.controller), _native_time()

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = self.measure()[0]
        self.preconditions = qualify_operation(self.session, self.controller, self.config, self.bound)
        self.previous_held, self.previous_on_top, self.previous_touching = None, {}, {}
        self.current_lifted, self.lifted, self.released = False, set(), {}
        self.release_events, self.wrong_object_grasped = [], False
        self.trace, self.trace_order, self.last_time = [], None, None
        self.record(observation, action=None)

    def _terminal_orders(self, rows, on_top, touching, gripper):
        if not gripper['all_arms_free'] or gripper.get('gripper_open') is not True:
            return []
        if any(row['visual_only'] or row['fixed_base'] or row['up_dot'] < MINIMUM_UP_DOT or
               row['linear_speed_m_s'] > MAXIMUM_LINEAR_SPEED or row['angular_speed_rad_s'] > MAXIMUM_ANGULAR_SPEED
               for row in rows.values()):
            return []
        orders = []
        for bottom, middle, top in permutations(OBJECT_KEYS):
            pairs = ((bottom, 'task_support'), (middle, bottom), (top, middle))
            if not all(on_top[pair] and touching[pair] for pair in pairs):
                continue
            if touching[(middle, 'task_support')] or touching[(top, 'task_support')]:
                continue
            if (getattr(self, 'parameters', {}).get('manipulation_revision') != FINAL_STATE_REVISION and
                    not all(key in self.lifted and self.released.get(key) == lower
                       for key, lower in ((middle, bottom), (top, middle)))):
                continue
            # No robot, furniture, wall, floor support of an upper object, or
            # unrelated body may prop up the completed tower.
            allowed = {bottom: ('task_support', middle), middle: (bottom, top), top: (middle,)}
            support_paths = support_contact_paths(self.session, self.bound[bottom].name) or (
                self.bound['floor'].prim_path,)
            if any(not any((any(body == path or body.startswith(path + '/') for path in support_paths)
                           if other == 'task_support' else _belongs(body, self.bound[other]))
                           for other in allowed[key])
                   for key in OBJECT_KEYS for bodies in rows[key]['external_contacts'] for body in bodies):
                continue
            orders.append((bottom, middle, top))
        return orders

    def _update_final(self, *, advance):
        """Inspect final native support; only controller callbacks add dwell samples."""
        try:
            rows, on_top, touching, gripper, (now, dt) = self.measure()
        except Exception as error:
            self.trace, self.trace_order = [], None
            if self.last is None:
                raise
            self.last = {**self.last, 'measurement_error': f'{type(error).__name__}: {error}',
                'measured_order': None, 'stability_duration_s': 0., 'stability_samples': 0,
                'criteria': {**self.last['criteria'], 'native_measurement_available': False,
                    'native_contact_order_upright_stationary_unheld': False,
                    'continuous_native_stability_window': False}, 'task_success': False}
            return self.last
        if self.last_time is not None and now < self.last_time - 1e-9:
            raise ValueError('Native simulation clock moved backwards within the episode.')
        orders = self._terminal_orders(rows, on_top, touching, gripper)
        order = orders[0] if len(orders) == 1 else None
        gap = now - self.last_time if self.last_time is not None else 0.
        restart = order != self.trace_order or gap > MAXIMUM_SAMPLE_GAP_S + 1e-7
        if self.trace and not restart:
            first = self.trace[0]['objects']
            restart = any(np.linalg.norm(np.asarray(row['position']) - first[key]['position']) > MAXIMUM_DRIFT_M or
                (Rotation.from_quat(row['quaternion_xyzw']).inv() * Rotation.from_quat(
                    first[key]['quaternion_xyzw'])).magnitude() > MAXIMUM_ROTATION_RAD
                for key, row in rows.items())
        if order is None or restart:
            self.trace, self.trace_order = [], None
        if advance and (self.last_time is None or now > self.last_time + 1e-9):
            if order is not None:
                self.trace_order = order
                if not self.trace or self.trace[-1]['time_s'] - self.trace[0]['time_s'] < STABILITY_SECONDS - 1e-7:
                    self.trace.append({'time_s': now, 'objects': rows})
            self.last_time = now
        duration = self.trace[-1]['time_s'] - self.trace[0]['time_s'] if self.trace else 0.
        criteria = {'native_initial_preconditions': self.preconditions['accepted'],
            'native_measurement_available': True,
            'native_contact_order_upright_stationary_unheld': order is not None,
            'continuous_native_stability_window': duration >= STABILITY_SECONDS - 1e-7 and
                len(self.trace) >= MINIMUM_STABILITY_SAMPLES,
            'original_scales_preserved': all(np.max(np.abs(np.asarray(rows[key]['scale']) -
                self.parameters['original_objects'][key]['scale'])) <= 1e-5 for key in OBJECT_KEYS)}
        categories = [self.parameters['original_objects'][key]['category'] for key in order] if order else None
        self.last = {'native_objects': rows, 'native_gripper': gripper, 'time_s': now, 'sim_step_dt_s': dt,
            'native_relations': [{'upper': upper, 'lower': lower, 'on_top': value,
                'touching': touching[(upper, lower)]} for (upper, lower), value in on_top.items()],
            'measured_order': list(order) if order else None, 'measurement_error': None,
            'stability_duration_s': duration, 'stability_samples': len(self.trace), 'criteria': criteria,
            'auxiliary_original_order_match': categories in self.parameters['legacy_accepted_category_orders']
                if categories else None, 'task_success': bool(all(criteria.values()))}
        return self.last

    def _update(self, *, advance=True):
        if self.initial is None:
            return None
        if self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION:
            return self._update_final(advance=advance)
        rows, on_top, touching, gripper, (now, dt) = self.measure()
        # record() and the control callback can see the same native step.
        if self.last_time is not None and now <= self.last_time + 1e-9:
            if now < self.last_time - 1e-9:
                raise ValueError('Native simulation clock moved backwards within the episode.')
            return self.last
        held_name = gripper.get('object_name') if (gripper.get('holding') is True and
                    gripper.get('constraint_active') is True and gripper.get('release_pending') is False) else None
        held = next((key for key in OBJECT_KEYS if self.bound[key].name == held_name), None)
        if held_name is not None and held is None and not legal_background_grasp(self.session, held_name):
            self.wrong_object_grasped = True
        if held != self.previous_held:
            if self.previous_held is not None:
                released = self.previous_held
                supported_by = [lower for lower in OBJECT_KEYS if lower != released and
                                self.previous_on_top.get((released, lower)) and
                                self.previous_touching.get((released, lower))]
                if self.current_lifted and len(supported_by) == 1:
                    self.lifted.add(released)
                    self.released[released] = supported_by[0]
                    self.release_events.append({'time_s': now, 'upper': released, 'lower': supported_by[0],
                                                'native_held_lift_observed': True})
                else:
                    self.released.pop(released, None)
            self.current_lifted = False
            if held is not None:
                self.released.pop(held, None)
        if held is not None:
            self.current_lifted = bool(self.current_lifted or
                (rows[held]['aabb_min'][2] - self.initial[held]['aabb_min'][2] >= MINIMUM_LIFT_M and
                 not touching[(held, 'task_support')]))
        orders = self._terminal_orders(rows, on_top, touching, gripper)
        order = orders[0] if len(orders) == 1 else None
        gap = now - self.last_time if self.last_time is not None else 0.
        if order is None:
            self.trace, self.trace_order = [], None
        else:
            restart = order != self.trace_order or gap > MAXIMUM_SAMPLE_GAP_S + 1e-7
            if self.trace and not restart:
                first = self.trace[0]['objects']
                restart = any(np.linalg.norm(np.asarray(row['position']) - first[key]['position']) > MAXIMUM_DRIFT_M or
                    (Rotation.from_quat(row['quaternion_xyzw']).inv() * Rotation.from_quat(
                        first[key]['quaternion_xyzw'])).magnitude() > MAXIMUM_ROTATION_RAD
                    for key, row in rows.items())
            if restart:
                self.trace = []
            self.trace_order = order
            # Keep one full terminal window, continuing to check current state.
            if not self.trace or self.trace[-1]['time_s'] - self.trace[0]['time_s'] < STABILITY_SECONDS - 1e-7:
                self.trace.append({'time_s': now, 'objects': rows})
        duration = self.trace[-1]['time_s'] - self.trace[0]['time_s'] if self.trace else 0.
        criteria = {'native_initial_preconditions': self.preconditions['accepted'],
                    'two_upper_real_lift_and_supported_release_events': order is not None,
                    'native_contact_order_upright_stationary_unheld': order is not None,
                    'continuous_native_stability_window': duration >= STABILITY_SECONDS - 1e-7 and
                        len(self.trace) >= MINIMUM_STABILITY_SAMPLES,
                    'no_wrong_object_grasp': not self.wrong_object_grasped,
                    'original_scales_preserved': all(np.max(np.abs(np.asarray(rows[key]['scale']) -
                        self.parameters['original_objects'][key]['scale'])) <= 1e-5 for key in OBJECT_KEYS)}
        categories = [self.parameters['original_objects'][key]['category'] for key in order] if order else None
        self.last = {'native_objects': rows, 'native_gripper': gripper, 'time_s': now, 'sim_step_dt_s': dt,
                     'native_relations': [{'upper': upper, 'lower': lower, 'on_top': value,
                         'touching': touching[(upper, lower)]} for (upper, lower), value in on_top.items()],
                     'measured_order': list(order) if order else None, 'release_events': list(self.release_events),
                     'stability_duration_s': duration, 'stability_samples': len(self.trace), 'criteria': criteria,
                     'auxiliary_original_order_match': categories in self.parameters['legacy_accepted_category_orders']
                         if categories else None, 'task_success': bool(all(criteria.values()))}
        self.previous_held, self.previous_on_top, self.previous_touching = held, on_top, touching
        self.last_time = now
        return self.last

    def sample_step(self):
        self._update()

    def record(self, observation, *, action):
        final_state = self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION
        value = self._update(advance=not final_state)
        if value is None:
            return None
        record = {**value, 'schema': 'private_real_stack_v1', 'episode': self.episode,
                  'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                  'action': action, 'configuration': self.config, 'initial_native_objects': self.initial,
                  'initial_native_preconditions': self.preconditions, 'stability_trace': self.trace}
        available = bool(self.preconditions['accepted'])
        released = len({event['upper'] for event in self.release_events}) >= 2
        record.update(task_revision=TASK_REVISION, evaluation_validity='complete' if available else 'unavailable',
            completion_items=[{'id': 'two_supported_upper_releases', 'kind': 'checkpoint', 'completed': released,
                'evaluation_validity': 'valid' if available else 'unavailable'},
                {'id': 'withdrawn_stable_tower', 'kind': 'terminal_condition', 'completed': value['task_success'],
                 'evaluation_validity': 'valid' if available else 'unavailable'}])
        if final_state:
            available = available and value['criteria']['native_measurement_available']
            record.update(schema='private_final_stack_v1',
                task_revision=self.parameters.get('task_revision', FINAL_STATE_REVISION),
                manipulation_revision=FINAL_STATE_REVISION,
                measured_capability='released_final_tower',
                evaluation_validity='complete' if available else 'unavailable',
                completion_items=[{'id': 'withdrawn_stable_tower', 'kind': 'terminal_condition',
                    'completed': value['task_success'],
                    'evaluation_validity': 'valid' if available else 'unavailable'}])
        from wia.tasks.completion import progress
        record.update(progress(record))
        write_json(self.output / 'state.json', record)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        return record


def create_evaluator(session, controller, config, output):
    return StackOperationEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Physical Dynamics', 'Stacking & Stability')
    version = 'task_edit_20261018_v1_stacking_stability_v1.4_accessible_task_repair_v2'

    def inspect(self, source):
        try:
            _source_data(source)
            return {'eligible': True, 'reason': 'three_original_objects_and_private_legacy_orders_available'}
        except (ValueError, TypeError, KeyError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_only_diagnostic']}

    def convert(self, source, context):
        raw, payload, poses, orders = _source_data(source)
        bindings = {key: {'name': name, 'category': category, 'model': model}
                    for key, name, category, model in zip(OBJECT_KEYS, payload['object_names'],
                                                       payload['object_categories'], payload['object_models'])}
        bindings['floor'] = {'name': payload['floor']}
        originals = {key: {**bindings[key], 'scale': scale, **{field: poses[bindings[key]['name']][field]
                     for field in ('position', 'quaternion_xyzw', 'aabb_min', 'aabb_max')}}
                     for key, scale in zip(OBJECT_KEYS, payload['object_scales'])}
        robot, robot_evidence = _robot_start(raw, payload, poses, context)
        warnings = ['assumption_review_required', 'native_stack_execution_unverified',
                    'native_floor_reach_unverified', 'legacy_stability_not_contact_qualified']
        assumptions = [{
            'doubt': 'A static 45 cm circular footprint and original maps do not certify complete R1 body clearance, '
                'floor grasp reach, graspability, camera visibility or released stacking.',
            'chosen_assumption': 'Place each initially independent object on a measured accessible support, '
                'preserving its original identity, scale and orientation. Require a measured native floor start '
                'and real grasp, lift, placement, release and stable tower evidence.',
            'alternatives': ['Retain a diagnostic if no measured accessible placement qualifies.'],
            'scope': source.relative_path,
            'risk': 'Some original objects may be unreachable or ungraspable; generated candidates remain pending.'}, {
            'doubt': 'The legacy accepted-order set is based on scripted placement and weak AABB stability checks.',
            'chosen_assumption': 'Preserve its exact private QA scoring and report auxiliary order agreement; '
                'physical completion requires an independently measured stable tower in any order.',
            'alternatives': ['Restrict a future explicit task variant to the legacy order set if human Review requests it.',
                             'Keep unconfirmed source-selected orders as diagnostic-only goals.'],
            'scope': 'All 89 category sources and their original accepted-order sets.',
            'risk': 'A real stable order may differ from legacy labels; auxiliary QA disagreement must remain visible.'}, {
            'doubt': 'Source postprocessing moves the R1 eye sensor away from its native parent-frame mount.',
            'chosen_assumption': 'Restore the mount captured before postprocessing, retain source reference cameras, '
                'and apply the declared 45 degree robot head pitch.',
            'alternatives': ['Retain the original scene variant and its historical offset for diagnostic comparison.'],
            'scope': 'Adapted variants only.',
            'risk': 'Robot observations differ from legacy exploration; source observers and answers remain private and unchanged.'}]
        try:
            if sorted(json.loads(raw['answer'])) != sorted(orders):
                warnings.append('source_answer_trial_order_conflict')
        except (ValueError, TypeError):
            warnings.append('source_answer_format_unrecognized')
        result = {'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': None,
                  'status': 'generated', 'reuse_level': 'L2', 'warning_codes': warnings, 'assumptions': assumptions,
                  'private_source_evidence': {'original_objects': originals, 'legacy_accepted_category_orders': orders,
                      'legacy_trial_count': len(payload.get('trials', [])), 'robot_start': robot,
                      'robot_start_screen': robot_evidence, 'native_experiment_run': False},
                  'change_summary': ['Retain original three assets, scales, orientations, floor and private historical QA.',
                      'Place independent starting objects separately on measured accessible supports; initial inter-object spacing may change.',
                      'Add real robot lift/place/release goals; original QA is private auxiliary evidence.',
                      'Measure native support/contact, all-arm detachment, tilt, speeds and one-second stability.'],
                  'limitations': ['CPU conversion and original maps do not establish native scene loading, reach or physical success.',
                      'Legacy mock operations and historic native snapshots are not evidence of guarded stacking.',
                      'At least two upper objects must each have a measured held 5 cm lift and supported release.',
                      'Native state APIs and release timing still require native execution qualification.'],
                  'attempted_repairs': [],
                  'next_step': 'Publish all pending and diagnostic records to Review through the shared runner, '
                      'then qualify and execute guarded real stacking without legacy shortcuts.'}
        if robot is None:
            warnings.append('source_start_native_measurement_required')
            assumptions[0]['chosen_assumption'] = ('Retain a pending task without a guessed robot pose. '
                'Preserve the CPU screen failure as evidence and require the shared native initializer to '
                'measure an independent floor-supported collision-free start before qualification.')
        # Current pipeline already supplies R1 via r1_primitives.yaml; adding a
        # second robot would violate the installed template and source semantics.
        patches = [{'operation': 'restore_robot_head_mount'}]
        if robot is not None:
            patches.insert(0, {'operation': 'robot_pose', 'position': robot['position'], 'yaw_deg': robot['yaw_deg'],
                'heading_offsets_deg': [0.], 'jitter_xy_m': [0., 0.], 'head_pitch_deg': 45.})
        for key in OBJECT_KEYS:
            patches.extend([{'operation': 'object_physics', 'binding': key, 'visual_only': False, 'fixed_base': False},
                            {'operation': 'object_physics', 'phase': 'after_postprocessing',
                             'binding': key, 'visual_only': False}])
        labels = ', '.join(category.replace('_', ' ') for category in payload['object_categories'])
        public = {'objective': f'Build one stable three-object tower using the {labels}. Choose a bottom object '
                      'on the starting support surface. Grasp and lift each of the other two objects at least 5 cm, '
                      'place one on the bottom and the last on the middle, lower into support contact before '
                      'opening the gripper, then withdraw and leave the tower untouched for at least one second. '
                      'You may test another order by physically unstacking and rebuilding. '
                      'Original auxiliary question: ' + raw['question'],
                  'success_criteria': 'All three objects are upright and form a supported tower in any stable order. '
                      'The bottom touches the starting support surface; each upper touches and rests on its predecessor, '
                      'with no other support. Both upper objects have real grasp, lift and release evidence. '
                      'All robot hands are detached and withdrawn; the tower stays still for one continuous second. '
                      'The original answer is scored separately as auxiliary QA.',
                  'environment': 'The original room contains the three original-size objects on accessible starting supports.',
                  'constraints': ['Use only existing guarded base, arm, trunk, wrist and gripper controls.',
                                  'Keep task object sizes and task supports unchanged. Noncritical native movable furniture may be moved using guarded arm and gripper actions.',
                                  'Use real pick-and-place; legacy mock grasp and kinematic placement are excluded.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
                      'binding_keys': [*OBJECT_KEYS, 'floor'], 'parameters': {'original_objects': originals,
                          'legacy_accepted_category_orders': orders}}
        evaluation['parameters'].update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        result.update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
                      task_definition=physical_task(source, context, public=public, evaluation=evaluation,
                          bindings=bindings, patches=patches, capabilities=['base_translation', 'base_rotation',
                              'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']))
        result['interaction_policy'] = {
            'mode': 'manipulation', 'group_bindings': list(OBJECT_KEYS),
            'manipulation_bindings': list(OBJECT_KEYS),
            'relocation_policy': 'independent_groups',
            'placement_groups': [[key] for key in OBJECT_KEYS],
            'source_camera_policy': 'preserve', 'source_region': raw.get('room'),
            'support_search': 'measured_surfaces', 'allow_fallback_table': True,
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
            'maximum_stance_distance_m': .85,
            'warning_codes': ['independent_candidate_support_adaptation_requires_review',
                              'source_qa_reference_layout_changed', 'native_grasp_execution_unverified'],
        }
        result['change_summary'].append('Apply a reversible source-map R1 stance, restore its measured native '
            'head mount, and enable original object collisions before and after postprocessing; retain physics settings.')
        return result

    def convert_all(self, sources, context):
        """Keep immutable identities and shared exception capture for every source."""
        return super().convert_all(sorted(sources, key=lambda source: source.relative_path), context)

    def sample(self, manifest, policy=None):
        """Cover failures, accepted-order counts and original footprint extremes."""
        policy = policy or {}
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        count = min(policy.get('count', 5), len(rows))
        chosen = []
        def add(row):
            if row['candidate_id'] not in chosen and len(chosen) < count:
                chosen.append(row['candidate_id'])
        for row in rows:
            if row['status'] != 'generated':
                add(row)
                break
        for row in rows:
            if row['candidate_id'] in policy.get('previous_failures', ()):
                add(row)
        generated = [row for row in rows if row['status'] == 'generated']
        if generated:
            def area(row):
                objects = row['private_source_evidence']['original_objects'].values()
                return max(np.prod(np.asarray(obj['aabb_max'][:2]) - obj['aabb_min'][:2]) for obj in objects)
            add(min(generated, key=area))
            add(max(generated, key=area))
            for orders in (1, 2, 3):
                choices = [row for row in generated if len(row['private_source_evidence']['legacy_accepted_category_orders']) == orders]
                if choices:
                    add(choices[0])
        by_id = {row['candidate_id']: row for row in rows}
        for key in super().sample(manifest, {**policy, 'count': len(rows)}):
            add(by_id[key])
        return chosen


def render_cpu_map_evidence(record, context, output):
    """Plot private original maps and deterministic stance; never a native image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    evidence = record['private_source_evidence']
    maps = _maps(_asset_root(context), record['scene_model'])
    if maps is None:
        return None
    trav, rooms, room_ids, _ = maps
    objects = evidence['original_objects']
    centre = np.mean([obj['position'][:2] for obj in objects.values()], axis=0)
    half_size = 2.5
    low = np.floor((centre[::-1] - half_size) / MAP_RESOLUTION_M + trav.shape[0] / 2).astype(int)
    high = np.ceil((centre[::-1] + half_size) / MAP_RESOLUTION_M + trav.shape[0] / 2).astype(int)
    low, high = np.maximum(low, 0), np.minimum(high, trav.shape)
    crop = (slice(low[0], high[0]), slice(low[1], high[1]))
    region = rooms[crop] == room_ids[record['room']]
    image = np.zeros((*region.shape, 3), dtype=float)
    image[region] = [.35, .35, .35]
    image[region & (trav[crop] > 0)] = [.85, .85, .85]
    extent = [(low[1] - trav.shape[0] / 2) * MAP_RESOLUTION_M,
              (high[1] - trav.shape[0] / 2) * MAP_RESOLUTION_M,
              (high[0] - trav.shape[0] / 2) * MAP_RESOLUTION_M,
              (low[0] - trav.shape[0] / 2) * MAP_RESOLUTION_M]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, extent=extent, origin='upper')
    for index, obj in enumerate(objects.values()):
        lo, hi = np.asarray(obj['aabb_min']), np.asarray(obj['aabb_max'])
        ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], fill=False, color=f'C{index}', linewidth=2))
        ax.text(obj['position'][0], obj['position'][1], obj['category'].replace('_', ' '), color=f'C{index}', fontsize=8)
    robot = evidence['robot_start']
    if robot:
        xy = robot['position'][:2]
        ax.add_patch(Circle(xy, ROBOT_SCREEN_RADIUS_M, fill=False, color='red', linewidth=2))
        angle = math.radians(robot['yaw_deg'])
        ax.arrow(*xy, .3 * math.cos(angle), .3 * math.sin(angle), color='red', head_width=.08)
    ax.set_title('CPU SOURCE MAP DIAGNOSTIC — NOT A NATIVE CAPTURE\n' +
                 record['scene_model'] + ' / ' + record['room'] + '\n' + record['status'], fontsize=10)
    ax.set_xlabel('Original world X (m)'); ax.set_ylabel('Original world Y (m)')
    ax.set_aspect('equal'); fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    return str(output)

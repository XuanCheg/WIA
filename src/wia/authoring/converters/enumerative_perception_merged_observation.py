"""Convert compact counting clusters into pending native inventory operations.

Run on CPU with the behavior environment's Python from the repository root:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2. Native hooks
are called by the existing guarded task runtime; they never perform actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from collections import Counter
from functools import lru_cache
from itertools import combinations
import json
import math
import os
from pathlib import Path
import random
import re

import numpy as np

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import sha256, write_json
from wia.tasks.interaction_setup import reference_position, touching_task_support, on_top_task_support

MODULE = 'wia.authoring.converters.enumerative_perception_merged_observation'
QUESTION = re.compile(r'Considering both images, how many objects of category "([^"\n]+)" '
                      r'are present in this scene\?')
OPERATIONS = ('retrieve_return', 'inventory_return')
LIFT_M = .05
POSE_TOLERANCE_M = .03
PAIR_TOLERANCE_M = .03
ANGLE_TOLERANCE_DEG = 15.
LINEAR_SPEED_M_S = .02
ANGULAR_SPEED_RAD_S = .1
STABILITY_STEPS = 15


def _legal_background_grasp(session, controller, name):
    if name is None:
        return False
    obj = session.env.scene.object_registry('name', name)
    arm = getattr(controller, 'working_arm', 'left')
    eef = getattr(session.robot, 'eef_links', {}).get(arm)
    return bool(obj is not None and eef is not None and
                legal_background_contact(session, eef.prim_path, obj.prim_path))


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


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Native and source measurements must be finite.')
    return result


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def _render(source):
    return source.payload.get('question_data', {}).get('render', {})


def _camera(render):
    cameras = render.get('camera_poses') or []
    name = Path(render.get('image', '')).name
    if isinstance(cameras, dict):
        return cameras.get(name) or next(iter(cameras.values()), {})
    return next((row for row in cameras if row.get('_key') == name), cameras[0] if cameras else {})


@lru_cache(maxsize=2)
def _maps(asset_root, scene_model):
    """Read installed original 1 cm maps without importing the simulator."""
    import cv2
    cv2.setNumThreads(2)
    layout = Path(asset_root) / 'scenes' / scene_model / 'layout'
    paths = [layout / name for name in ('floor_trav_0.png', 'floor_insseg_0.png')]
    if not all(path.is_file() for path in paths):
        return None
    trav, rooms = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths)
    if trav is None or rooms is None or trav.shape != rooms.shape or trav.shape[0] != trav.shape[1]:
        return None
    return trav, rooms, {str(path): sha256(path) for path in paths}


def _map_clear(maps, xy, room_id):
    trav, rooms, _ = maps
    radius = 40
    # BaseMap.world_to_map: flip(xy / resolution + map_size / 2).int().
    row, col = [int(x / .01 + trav.shape[0] / 2.) for x in (xy[1], xy[0])]
    if min(row, col) < radius or row + radius >= trav.shape[0] or col + radius >= trav.shape[1]:
        return False
    offsets = np.arange(-radius, radius + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius ** 2
    area = (slice(row - radius, row + radius + 1), slice(col - radius, col + radius + 1))
    return bool(np.all(trav[area][disk] > 0) and np.all(rooms[area][disk] == room_id))


def _robot_start(source, context):
    """A private static proposal, not proof of native reach or clearance."""
    render = _render(source)
    bounds = render.get('room_bbox', {}).get('expanded_bbox_world_xy') or render.get(
        'room_bbox', {}).get('bbox_world_xy')
    camera = _camera(render)
    room_id = render.get('room_bbox', {}).get('room_id')
    if not _vector(bounds, 4) or min(bounds[2] - bounds[0], bounds[3] - bounds[1]) <= 1.3 or not _vector(
            camera.get('position'), 3) or type(room_id) is not int or room_id <= 0:
        return None
    data_root = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    asset_root = context.config.get('original_asset_root', str(data_root / 'behavior-1k-assets'))
    maps = _maps(str(asset_root), source.raw['scene'])
    if maps is None:
        return None
    eye = camera['position']
    proposals = [('primary_camera', eye[:2])]
    for view in render.get('multi_image_input', {}).get('views', []):
        point = view.get('camera_pose', {}).get('position')
        if view.get('is_traversable') is True and _vector(point, 3):
            proposals.append((view.get('view_id', 'source_corner'), point[:2]))
    items = render['resolved_objects']['targets']
    screened = []
    # Retain already-supported direct proposals. A bounded stencil is used only
    # after every direct source-observer proposal fails the same static screens.
    stencil = [(dx, dy) for dx in (-.5, -.25, 0., .25, .5) for dy in (-.5, -.25, 0., .25, .5)]
    for offsets in ([(0., 0.)], stencil):
        for label, original_xy in proposals:
            projected = [min(max(original_xy[i], bounds[i] + .65), bounds[i + 2] - .65) for i in (0, 1)]
            for offset in offsets:
                xy = [projected[i] + offset[i] for i in (0, 1)]
                if not all(bounds[i] + .65 <= xy[i] <= bounds[i + 2] - .65 for i in (0, 1)):
                    continue
                clearance = min(math.sqrt(sum(max(item['bbox'][0][i] - xy[i], xy[i] - item['bbox'][1][i],
                    0.) ** 2 for i in (0, 1))) for item in items)
                if clearance >= .65 and _map_clear(maps, xy, room_id):
                    screened.append((math.dist(xy, eye[:2]), label, xy, original_xy, clearance, offset))
        if screened:
            break
    if not screened:
        return None
    _, label, xy, original_xy, clearance, offset = min(screened)
    center = np.mean([item['position'] for item in items], axis=0)
    return {'position': [*xy, float(eye[2] - 1.25)],
            'yaw_deg': math.degrees(math.atan2(center[1] - xy[1], center[0] - xy[0])),
            'observer_id': label, 'original_observer_xy': list(original_xy),
            'observer_search_offset_xy_m': list(offset),
            'source_bbox_clearance_m': clearance, 'static_map_disk_radius_m': .4,
            'source_room_id': room_id, 'original_map_sha256': maps[2],
            'provenance': 'Original primary/corner observers, room_bbox and generated-object AABBs; '
                'installed floor_trav_0/floor_insseg_0 maps at 1 cm; BaseMap.world_to_map; '
                'batch_counting_merge._sample_primary_view_pose uses floor_top + 1.25 m.'}


def _geometry_audit(source):
    q = source.payload['question_data']
    items = _render(source)['resolved_objects']['targets']
    overlaps = []
    for a, b in combinations(items, 2):
        widths = [min(a['bbox'][1][i], b['bbox'][1][i]) - max(a['bbox'][0][i], b['bbox'][0][i])
                  for i in range(3)]
        if min(widths) > 0:
            overlaps.append({'names': [a['name'], b['name']], 'overlap_extents_m': widths})
    return {'resolved_target_count': len(items), 'source_aabb_overlaps': overlaps,
            'merge_patterns': sorted({entry.get('metadata', {}).get('merge_pattern', 'unspecified')
                for entry in q.get('case_metadata', {}).get('source_entries', [])}),
            'legacy_original_requested_category': q.get('count_object', {}).get('original_requested_category'),
            'saved_answer_matches_generated_count': str(source.raw.get('answer')) == str(len(items)),
            'source_camera_policy_preserved': True,
            'aabb_overlap_is_not_mesh_penetration_evidence': True}


def _snapshot(obj):
    """Read measured native geometry, physics flags and velocities only."""
    position, quaternion = (_array(x) for x in obj.get_position_orientation())
    low, high = (_array(x) for x in obj.aabb)
    if position.shape != (3,) or quaternion.shape != (4,) or low.shape != (3,) or high.shape != (3,) or (
            np.linalg.norm(quaternion) < 1e-6 or np.any(high <= low)):
        raise ValueError('Invalid measured native object geometry.')
    return {'name': obj.name, 'category': obj.category, 'model': obj.model,
            'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}


def _near(row, original):
    a, b = (_array(x['quaternion_xyzw']) for x in (row, original))
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    angle = math.degrees(2. * math.acos(min(1., dot)))
    return math.dist(row['position'], original['position']) <= POSE_TOLERANCE_M and angle <= ANGLE_TOLERANCE_DEG


def _layout_preserved(rows, originals):
    if not all(_near(row, originals[key]) for key, row in rows.items()):
        return False
    return all(np.linalg.norm((_array(rows[a]['position']) - _array(rows[b]['position'])) -
        (_array(originals[a]['position']) - _array(originals[b]['position']))) <= PAIR_TOLERANCE_M
        for a, b in combinations(sorted(rows), 2))


def _stationary(rows):
    return all(row['linear_speed_m_s'] <= LINEAR_SPEED_M_S and row['angular_speed_rad_s'] <=
               ANGULAR_SPEED_RAD_S for row in rows.values())


def _support(objects, floor, session):
    """Native Touching/OnTop chains permit the original stacked-cluster mechanism."""
    from omnigibson.object_states import OnTop, Touching
    edges = {key: [] for key in objects}
    floor_roots, task_support_roots = [], []
    for key, obj in objects.items():
        if bool(obj.states[Touching].get_value(floor)) and bool(obj.states[OnTop].get_value(floor)):
            floor_roots.append(key)
        if touching_task_support(session, obj, floor) and on_top_task_support(session, obj, floor):
            task_support_roots.append(key)
        for lower, support in objects.items():
            if key != lower and bool(obj.states[Touching].get_value(support)) and bool(
                    obj.states[OnTop].get_value(support)):
                edges[key].append(lower)
    grounded = set(task_support_roots)
    for _ in objects:
        grounded.update(key for key, lower in edges.items() if any(item in grounded for item in lower))
    return {'floor_roots': sorted(floor_roots), 'task_support_roots': sorted(task_support_roots),
            'target_support_edges': edges,
            'grounded_bindings': sorted(grounded), 'all_grounded': grounded == set(objects)}


def _robot_state(session):
    """Native chassis segmentation and contacts; no motion or contact mutation."""
    robot, scene = session.robot, session.env.scene
    position, _ = robot.get_position_orientation()
    low, high = (_array(x) for x in robot.links[robot.base_footprint_link_name].aabb)
    rooms = [scene.seg_map.get_room_instance_by_point(np.array([x, y]))
             for x in (low[0], (low[0] + high[0]) / 2, high[0])
             for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    native_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj.prim_path for obj in native_objects if obj.category in ('floor', 'floors')]
    floor_contacts, obstacle_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not (body == robot.prim_path or body.startswith(robot.prim_path + '/'))]
        if external:
            on_floor = all('groundPlane' in body or any(body == path or body.startswith(path + '/')
                for path in floors) for body in external)
            (floor_contacts if on_floor else obstacle_contacts).append(bodies)
    return {'position': _array(position).tolist(), 'footprint_rooms': rooms,
            'in_source_room': bool(rooms and all(room == session.room_name for room in rooms)),
            'floor_contacts': floor_contacts, 'obstacle_contacts': obstacle_contacts}


def _configuration(session, config):
    if config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or config.get('operation') not in OPERATIONS:
        raise ValueError('Use this owned native adapter and a declared operation recipe.')
    if (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name):
        raise ValueError('Operation configuration does not match the native source scene/room.')
    params = config['parameters']
    targets = params['target_keys']
    keys = [*targets, 'floor']
    if not targets or len(set(keys)) != len(keys) or set(config['binding_keys']) != set(keys) or (
            set(config['bindings']) != set(keys) or set(params['source_objects']) != set(targets) or
            len(set(config['bindings'].values())) != len(keys)):
        raise ValueError('Original target and floor bindings must be complete and unique.')
    return params


def qualify_operation(session, executor, config, bound):
    """Fail closed on measured initialization; never step, act or set poses."""
    checks = {}
    try:
        params = _configuration(session, config)
        objects = {key: bound[key] for key in params['target_keys']}
        rows = {key: _snapshot(obj) for key, obj in objects.items()}
        support, robot, gripper = _support(objects, bound['floor'], session), _robot_state(session), executor.debug_gripper_state()
        checks = {'floor_collidable': not bound['floor'].visual_only,
            'targets_movable_collidable': all(not row['visual_only'] and not row['fixed_base'] for row in rows.values()),
            'original_categories': all(row['category'] == params['target_category'] for row in rows.values()),
            'source_layout_and_pairs_preserved': _layout_preserved(rows, {key: {**original,
                'position': reference_position(session, objects[key].name, original['position'])}
                for key, original in params['source_objects'].items()}),
            'original_dimensions_preserved': all(np.allclose(_array(row['aabb_max']) - _array(row['aabb_min']),
                _array(params['source_objects'][key]['bbox'][1]) - _array(params['source_objects'][key]['bbox'][0]),
                rtol=.1, atol=.005) for key, row in rows.items()),
            'native_task_surface_connected_support': support['all_grounded'], 'targets_stationary': _stationary(rows),
            'initial_gripper_empty': gripper.get('object_name') is None and gripper.get('constraint_active') is False,
            'robot_in_source_room': robot['in_source_room'],
            'robot_supported_without_obstacle_contact': bool(robot['floor_contacts']) and not robot['obstacle_contacts']}
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_objects': rows, 'native_support': support,
            'native_robot': robot, 'reason': None if accepted else 'native_cluster_initialization_failed',
            'grasp_lift_return_execution_verified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_cluster_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'grasp_lift_return_execution_verified': False}


class ClusterEvaluator:
    """Measure distinct physical lift/release history and stable native return."""
    def __init__(self, session, controller, config, output):
        self.params = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.bound.values()):
            raise ValueError('A required original native binding is missing.')
        self.objects = {key: self.bound[key] for key in self.params['target_keys']}
        self.name_to_key = {obj.name: key for key, obj in self.objects.items()}
        self.initial, self.episode = None, 0

    def measure(self):
        return ({key: _snapshot(obj) for key, obj in self.objects.items()},
                _support(self.objects, self.bound['floor'], self.session), self.controller.debug_gripper_state())

    def begin_episode(self, observation):
        self.episode += 1
        self.initial_qualification = qualify_operation(self.session, self.controller, self.config, self.bound)
        self.initial, self.initial_support, _ = self.measure()
        self.lifted, self.returned = set(), set()
        self.return_stability = dict.fromkeys(self.objects, 0)
        self.active_target = None
        self.wrong_object_grasped = False
        self.stable_steps = 0
        self.record(observation, action=None)

    def _update(self, count_step=False):
        if self.initial is None:
            return None
        rows, support, gripper = self.measure()
        held_name = gripper.get('object_name')
        held = self.name_to_key.get(held_name) if gripper.get('holding') is True and gripper.get('constraint_active') is True else None
        if (held_name is not None and held_name not in self.name_to_key
                and not _legal_background_grasp(self.session, self.controller, held_name)):
            self.wrong_object_grasped = True
        lift = {key: row['aabb_min'][2] - self.initial[key]['aabb_min'][2] for key, row in rows.items()}
        if held is not None:
            self.active_target = held
            # A later grasp invalidates a previously recorded return until it is released again.
            self.returned.discard(held)
            if lift[held] >= LIFT_M:
                self.lifted.add(held)
        empty_open = (held_name is None and gripper.get('constraint_active') is False and gripper.get(
            'holding') is False and gripper.get('gripper_open') is True and gripper.get('release_pending') is False)
        if empty_open and self.active_target is not None:
            key = self.active_target
            if key in self.lifted and _near(rows[key], self.initial[key]) and key in support['grounded_bindings']:
                self.returned.add(key)
                self.active_target = None
        completed = self.lifted & self.returned
        operation_done = bool(completed) if self.config['operation'] == 'retrieve_return' else completed == set(self.objects)
        current_valid = all(not row['visual_only'] and not row['fixed_base'] for row in rows.values())
        source_layout_restored = _layout_preserved(rows, self.initial)
        # Restore every original target-on-target support edge, including stacked objects.
        support_restored = (support['all_grounded'] and set(self.initial_support['task_support_roots']).issubset(
            support['task_support_roots']) and all(set(lower).issubset(support['target_support_edges'][key])
            for key, lower in self.initial_support['target_support_edges'].items()))
        criteria = {'initial_native_qualification': self.initial_qualification['accepted'],
            'native_lift_and_return_history': bool(operation_done), 'no_wrong_object_grasp': not self.wrong_object_grasped,
            'targets_movable_collidable': current_valid, 'gripper_empty_open_released': empty_open,
            'native_layout_and_pairs_restored': source_layout_restored, 'native_support_restored': support_restored,
            'targets_stationary': _stationary(rows)}
        terminal = bool(all(criteria.values()))
        if count_step:
            self.stable_steps = self.stable_steps + 1 if terminal else 0
        elif not terminal:
            self.stable_steps = 0
        value = {'native_objects': rows, 'native_support': support, 'native_gripper': gripper,
            'native_lift_m': lift, 'lifted_bindings': sorted(self.lifted), 'returned_bindings': sorted(self.returned),
            'criteria': criteria, 'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS,
            'task_success': bool(terminal and self.stable_steps >= STABILITY_STEPS)}
        if self.config['operation'] == 'retrieve_return':
            items = [('retrieve_return', value['task_success'], 'terminal_condition',
                {'returned_bindings': sorted(completed), 'stable_steps': self.stable_steps})]
        else:
            items = []
            for key in sorted(self.objects):
                returned = (key in completed and _near(rows[key], self.initial[key]) and
                    key in support['grounded_bindings'] and _stationary({key: rows[key]}) and
                    set(self.initial_support['target_support_edges'][key]).issubset(support['target_support_edges'][key]))
                if count_step:
                    self.return_stability[key] = self.return_stability[key] + 1 if returned else 0
                elif not returned:
                    self.return_stability[key] = 0
                items.append((f'{key}:grasp_lift_return', self.return_stability[key] >= STABILITY_STEPS,
                    'terminal_condition', {'lifted': key in self.lifted, 'returned': key in self.returned,
                    'stable_steps': self.return_stability[key]}))
        return _completion_result(value, items)

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        value = self._update()
        if value is None:
            return None
        value.update(schema='private_merged_observation_operation_v1', episode=self.episode,
            snapshot_id=observation.snapshot_id, action_step=observation.action_step, action=action,
            configuration=self.config, initial_native_objects=self.initial, initial_native_support=self.initial_support,
            initial_qualification=self.initial_qualification)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return ClusterEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Enumerative Perception', 'Merged Observation')
    version = '2.0.0_task_edit'

    def inspect(self, source):
        raw, payload = source.raw, source.payload
        if (source.big_task, source.small_task) != self.category or (raw.get('big_task'), raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:category_mismatch',
                    'warning_codes': ['not_convertible_under_L1_L2']}
        q, render = payload.get('question_data', {}), _render(source)
        match = QUESTION.fullmatch(raw.get('question', '').strip())
        resolved = render.get('resolved_objects', {})
        items = resolved.get('targets') or []
        reason = None
        if not match or q.get('case') != 'observation_merged':
            reason = 'unsupported_original_counting_mechanism'
        elif not items or resolved.get('containers') or resolved.get('confusers'):
            reason = 'missing_targets_or_unimplemented_original_mechanism'
        elif match[1] != render.get('target_category') or any(item.get('category') != match[1] for item in items):
            reason = 'public_category_and_resolved_assets_disagree'
        elif not payload.get('floor_name') or len({item.get('name') for item in items}) != len(items):
            reason = 'missing_unique_original_target_or_floor'
        elif any(not all(isinstance(item.get(field), str) and item[field] for field in ('name', 'category', 'model')) or
                 not _vector(item.get('position'), 3) or not _vector(item.get('quaternion_xyzw'), 4) or
                 not any(item['quaternion_xyzw']) or not isinstance(item.get('bbox'), list) or len(item['bbox']) != 2 or
                 not all(_vector(row, 3) for row in item['bbox']) or any(item['bbox'][1][i] <= item['bbox'][0][i]
                 for i in range(3)) for item in items):
            reason = 'incomplete_original_geometry'
        return {'eligible': reason is None, 'reason': 'not_convertible_under_L1_L2:' + reason if reason else None,
                'warning_codes': ['not_convertible_under_L1_L2'] if reason else []}

    def convert(self, source, context):
        if context.config.get('legacy_merged_manipulation'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import visual_discovery_task
        render = _render(source)
        bindings = {'floor': {'name': source.payload['floor_name']}}
        targets, originals = [], {}
        for index, item in enumerate(sorted(render['resolved_objects']['targets'], key=lambda item: item['name'])):
            key = f'target_{index:02d}'
            targets.append(key)
            bindings[key] = {field: item[field] for field in ('name', 'category', 'model')}
            originals[key] = item
        category = render['target_category'].replace('_', ' ')
        public = {'objective': f'Find all {len(targets)} distinct {category} objects in the overlapping original cluster. '
                + 'Combine mounted-camera views to distinguish individual objects and avoid counting repeated sightings.',
            'success_criteria': 'Every original target appears in delivered native robot images at least once; '
                + 'one target pickup or repeated observations cannot replace complete inventory discovery.',
            'environment': 'Original cluster overlap, support order, object identities, dimensions and reference views remain unchanged.',
            'constraints': ['Preserve the source cluster and inspect it through legal robot viewpoints.',
                'The total is supplied. No arbitrary lift, return, travel distance or count-estimation answer is required.']}
        return visual_discovery_task(source, context, bindings=bindings, target_keys=targets, public=public,
            preserved_relations=['source cluster and overlap structure', 'cross-view distinct identities',
                'original support relationships and source observer'], source_evidence={'source_objects': originals,
                    'geometry_audit': _geometry_audit(source)})

    def convert_legacy(self, source, context):
        eligibility = self.inspect(source)
        if not eligibility['eligible']:
            return {'scene_definition': original_scene(source, context), 'task_definition': None,
                'status': 'conversion_failed', 'failure_stage': 'source_inspection',
                'failure_reason': eligibility['reason'], 'warning_codes': eligibility['warning_codes'],
                'diagnostic_source_only': True}
        operation = context.config.get('operation', 'retrieve_return')
        if operation not in OPERATIONS:
            raise ValueError('Supported operation recipes: retrieve_return, inventory_return.')
        audit, start = _geometry_audit(source), _robot_start(source, context)
        payload, render = source.payload, _render(source)
        category = QUESTION.fullmatch(source.raw['question'].strip())[1]
        bindings, source_objects = {'floor': {'name': payload['floor_name']}}, {}
        for index, item in enumerate(sorted(render['resolved_objects']['targets'], key=lambda item: item['name'])):
            key = f'target_{index:02d}'
            bindings[key] = {field: item[field] for field in ('name', 'category', 'model')}
            source_objects[key] = {field: item[field] for field in ('position', 'quaternion_xyzw', 'bbox')}
        patches = [] if start is None else [{'operation': 'robot_pose', 'position': start['position'],
                    'yaw_deg': start['yaw_deg'], 'head_pitch_deg': 45.}]
        for key in source_objects:
            patches.extend([{'operation': 'object_physics', 'phase': 'before_construction',
                             'binding': key, 'visual_only': False, 'fixed_base': False},
                {'operation': 'object_physics', 'phase': 'after_postprocessing', 'binding': key, 'visual_only': False}])
        quantifier = 'one accessible distinct' if operation == 'retrieve_return' else 'every distinct'
        objective = (f'Locate the compact {category} cluster using robot observations. '
            f'Physically retrieve {quantifier} original {category}: grasp and lift it at least 5 cm, then return and '
            'release it near its starting pose. Restore the compact cluster and any original stacking; finish with '
            'an empty open gripper.')
        completion = 'At least one original target' if operation == 'retrieve_return' else 'Every distinct original target'
        public = {'objective': objective,
            'success_criteria': completion + ' has been held by an active native grasp constraint, lifted 5 cm '
                'above its initial lowest point and released near its initial pose. All original objects end within '
                '3 cm and 15 degrees of their starting poses, pairwise displacement vectors within 3 cm, with '
                'episode-start surface/stack supports restored and speeds at most 0.02 m/s and 0.1 rad/s for 15 physics samples. '
                'The gripper is open, empty and fully released.',
            'environment': 'The original room, floor, original-scale compact objects and reference camera are retained.',
            'constraints': ['The source count question is retained as private provenance; counting and multiple-view coverage are not scored.',
                'Use robot-mounted sensors and the existing guarded base, arm, trunk and gripper controls.',
                'Do not grasp task objects of any other category; native movable noncritical furniture may be moved with guarded arm/gripper actions. Avoid disturbing unselected objects; restore any original stack.',
                'Human review images, source maps, object identities/poses and evaluator evidence are private, not Agent sensors.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': list(bindings), 'parameters': {'target_category': category,
                'target_keys': list(source_objects), 'source_objects': source_objects}}
        scene = original_scene(source, context, bindings=bindings, patches=patches)
        scene['runtime'] = {'post_controller_settle_steps': 30}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches,
            capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none'})
        task['defaults'].update(max_steps=180 if operation == 'retrieve_return' else 300, wall_timeout_s=14400.)
        warnings = ['assumption_review_required', 'source_visual_only_restored', 'source_cluster_aabb_overlap',
                    'native_physical_execution_unverified', 'source_merged_occlusion_adapted',
                    'independent_target_support_requires_native_review']
        if start is None:
            warnings.extend(['source_static_start_unverified', 'native_spawn_repair_pending'])
        if 'random_compact_stacked_cluster' in audit['merge_patterns']:
            warnings.append('source_stacked_view_adapted_for_accessibility')
        if start is not None and any(start['observer_search_offset_xy_m']):
            warnings.append('source_observer_local_stance_repair')
        legacy = audit['legacy_original_requested_category']
        if legacy and legacy != category:
            warnings.append('legacy_requested_category_differs_from_resolved_public_category')
        assumptions = [{'doubt': 'Overlapping visual-only AABBs do not establish mesh penetration, stable contact '
                'or original-scale floor-level grasp feasibility.',
            'chosen_assumption': 'Treat each original target as an independent placement group, enable its '
                'collisions and gravity without changing asset size, and initialize it on a measured accessible '
                'support. Qualify accepted per-object reference poses, native stationarity and support before '
                'executing. Keep all candidates pending and preserve the empty original variant.',
            'alternatives': ['Keep source-only diagnostics when collision restoration fails.',
                'Retain the original compact view only when its native support and access qualify.',
                'Select inventory_return after individual native retrieval/return becomes supported.'],
            'scope': 'The adapted physical variant of every source, including original stacks.',
            'risk': 'Independent accessible placement changes source compact/stacked occlusion. Original-view QA '
                'compatibility, reach, grasp, replacement and native contact remain unverified. Qualification '
                'still rejects invalid accepted-baseline poses, physics or support.'},
            {'doubt': 'Camera traversability annotations and 2D maps do not certify whole R1-body clearance.',
            'chosen_assumption': 'Project an original observer 0.65 m inside room bounds, retain at least 0.65 m '
                'generated-object AABB clearance and a free original-room 0.4 m map disk; independently check '
                'native chassis room membership, floor support and obstacle contacts. Only if all direct proposals '
                'fail, search a fixed 0.25 m stencil up to 0.5 m per axis around projected source observers. '
                'If no CPU proposal qualifies, omit robot_pose and require a measured Native start instead of inventing one.',
            'alternatives': ['Qualify the original native start.', 'Retain a diagnostic after failed native stance qualification.'],
            'scope': 'One deterministic initialization-only R1 pose per generated adapted variant.',
            'risk': '2D screens omit extended arms and height-dependent obstacles. They do not prove navigation or IK.'}]
        if legacy and legacy != category:
            assumptions.append({'doubt': 'Historical requested category differs from the current public question and assets.',
                'chosen_assumption': 'Use the current public category and resolved assets; retain historical metadata verbatim.',
                'alternatives': ['Review upstream normalization independently without replacing assets or answers.'],
                'scope': 'Sources with a differing legacy category.',
                'risk': 'The historical pre-normalization semantic category is not the category of the saved scene.'})
        audit['robot_start'] = start
        return {'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'operation_recipe': operation, 'available_operation_recipes': list(OPERATIONS), 'private_source_audit': audit,
            'interaction_policy': {'mode': 'manipulation', 'group_bindings': list(source_objects),
                'manipulation_bindings': list(source_objects), 'relocation_policy': 'independent_groups',
                'placement_groups': [[key] for key in source_objects], 'support_search': 'measured_surfaces',
                'allow_fallback_table': True, 'source_camera_policy': 'preserve',
                'source_region': source.raw['room'], 'minimum_grasp_height_m': .35,
                'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85,
                'warning_codes': ['native_grasp_ik_and_aperture_pending', 'source_merged_occlusion_adapted']},
            'change_summary': ['Preserve assets, dimensions, identities/count and source reference observer.',
                'Initialize every independently countable original target on its own measured accessible support group.',
                'Restore original target physics at construction and after the source visual-only reset.',
                'Add an evidenced initialization-only R1 stance and native grasp/lift/return/stability predicates.',
                'Preserve source QA privately and independently of physical terminal evidence.'],
            'warning_codes': warnings, 'assumptions': assumptions,
            'limitations': ['CPU source/map evidence and screenshots do not prove native reach, collision-free '
                'settling or successful manipulation.', 'AABB overlap is a risk annotation, not an automatic exclusion.',
                'Measured support chains admit original target stacks; unavailable native measurements fail closed.',
                'Operation decision/time budgets are unbenchmarked ceilings, not observed duration estimates.'],
            'attempted_repairs': ['Reversible robot stance proposal, independent original-target support placement '
                'and two-phase physics restoration without asset scale changes.'],
            'next_step': 'Publish every candidate/diagnostic to Review; measure native collision-enabled layout, support '
                'and body clearance before real guarded grasp/lift/return trials. Never promote screenshots to physical qualification.'}

    def convert_all(self, sources, context):
        """Use the shared idempotent writer; every source retains a diagnostic row."""
        sources = sorted(sources, key=lambda source: source.relative_path)
        manifest = super().convert_all(sources, context)
        manifest['failure_distribution'] = dict(Counter(row.get('failure_reason') or 'unspecified'
            for row in manifest['records'] if row['status'] != 'generated'))
        manifest['sample_ids'] = self.sample(manifest, {'count': 5, 'seed': context.seed})
        manifest['physical_execution_verified'] = False
        manifest['available_operation_recipes'] = list(OPERATIONS)
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Cover failures, counts and both stack patterns with deterministic diversity."""
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        random.Random(policy.get('seed', manifest.get('seed', 0))).shuffle(rows)
        selected, scenes, strata = [], set(), set()
        previous = set(policy.get('previous_failures', ()))
        def add(row):
            if row and row['candidate_id'] not in selected and len(selected) < count:
                selected.append(row['candidate_id'])
                scenes.add(row.get('scene_model'))
                audit = row.get('private_source_audit', {})
                strata.add((audit.get('resolved_target_count'), tuple(audit.get('merge_patterns', []))))
        for row in rows:
            if row['candidate_id'] in previous:
                add(row)
        add(next((row for row in rows if row['status'] != 'generated'), None))
        for cardinality in sorted({row.get('private_source_audit', {}).get('resolved_target_count')
                for row in rows if row.get('private_source_audit', {}).get('resolved_target_count') is not None}):
            candidates = [row for row in rows if row['status'] == 'generated' and row.get(
                'private_source_audit', {}).get('resolved_target_count') == cardinality]
            repair_covered = any(row['candidate_id'] in selected and
                'source_observer_local_stance_repair' in row['warning_codes'] for row in rows)
            candidates.sort(key=lambda row: (not repair_covered and
                'source_observer_local_stance_repair' not in row['warning_codes'], row.get('scene_model') in scenes))
            add(candidates[0] if candidates else None)
        for row in sorted(rows, key=lambda row: row.get('scene_model') in scenes):
            audit = row.get('private_source_audit', {})
            if (audit.get('resolved_target_count'), tuple(audit.get('merge_patterns', []))) not in strata:
                add(row)
        for row in rows:
            add(row)
        return selected

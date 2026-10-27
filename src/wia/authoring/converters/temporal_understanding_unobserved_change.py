"""Restore saved Unobserved Change snapshots and add measured robot operations.

Run from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
CPU conversion never imports the simulator. Initialization restores source
assets only once; qualification and evaluation only read native measurements.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.resources import ESI_ROOT

from collections import Counter
import copy
from functools import lru_cache
import importlib
import math
import os
from pathlib import Path
import sys

import numpy as np

from wia.authoring.conversion import ROOT, ConverterBase, map_records, original_scene, physical_task
from wia.authoring.io import sha256, write_json

MODULE = 'core.esi_converters.temporal_understanding_unobserved_change'
SOURCE_BUILDER = ESI_ROOT / 'src/active_explore/tasks/temporal_understanding/unobserved_change.py'
REPLAY = ESI_ROOT / 'src/passive_gt/unobserved_regeneration.py'
PRODUCER = ESI_ROOT / 'src/dataset_generation/task_counting/batch_counting_merge.py'
PHASES = ('image1', 'image2')
OPERATIONS = ('phase_inventory_return', 'phase_inventory_extract', 'box_inspection_route')
MINIMUM_LIFT_M = .05
EXTRACTION_CLEARANCE_M = .01
STABILITY_STEPS = 15


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Source/native measurements must be finite.')
    return result


def _vector(value, size):
    value = _array(value)
    if value.shape != (size,):
        raise ValueError('Source/native vector dimensions are unresolved.')
    return value


def _angle(a, b):
    a, b = _vector(a, 4), _vector(b, 4)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        raise ValueError('Source/native quaternion is unresolved.')
    return math.degrees(2 * math.acos(min(1., abs(float(np.dot(a, b) / norm)))))


@lru_cache(maxsize=1)
def _replay():
    """Reuse the original CPU-only parser/config helper, not its GPU renderer."""
    source_root = str(ESI_ROOT / 'src')
    added = source_root not in sys.path
    if added:
        sys.path.insert(0, source_root)
    try:
        return importlib.import_module('passive_gt.unobserved_regeneration')
    finally:
        if added:
            sys.path.remove(source_root)


def _layout(source):
    raw, payload = source.raw, source.payload
    question = payload['question_data']
    # The original parser expects direct phase fields. Cleaning sometimes stores
    # those fields as keyed records. Normalize a private copy; never alter JSON.
    boxes = copy.deepcopy(question.get('boxes') or [])
    for box in boxes:
        keyed = map_records(box.get('phase_content'))
        for field in ('phase1_content', 'phase2_content'):
            if field not in box:
                item = keyed.get(field)
                box[field] = item if item and item.get('category') and item.get('model') else None
    gt = copy.deepcopy(map_records(question['render']['gt_view']))
    for phase, entry in map_records(gt.get('image')).items():
        gt.setdefault(phase, entry)
    minimal = {'scene': raw['scene'], 'room': raw['room'], 'seed': payload.get('seed', 7),
               'question_index': payload.get('question_index', 0), 'task_type': payload.get('task_type'),
               'question_data': {'boxes': boxes, 'task_type': question.get('task_type'),
                                 'candidate_index': question.get('candidate_index', 0), 'render': {'gt_view': gt}}}
    plan = _replay().build_unobserved_replay_plan(minimal, validate_selector=False)
    if plan.task_type not in ('change_detection', 'change_identification', 'current_state_reasoning'):
        raise ValueError('Original temporal mechanism is unresolved.')
    if raw.get('answer') != question.get('answer'):
        raise ValueError('Source answer fields disagree; do not infer a replacement answer.')
    if not payload.get('floor_name'):
        raise ValueError('Original floor identity is unresolved.')
    bindings, keys, relations, expected = {}, {phase: [] for phase in PHASES}, {}, {}
    container_keys = []
    for box in plan.boxes:
        key = f'container_{box.box_index:02d}'
        container_keys.append(key)
        bindings[key] = {'name': box.container_name, 'category': box.container_category, 'model': box.container_model}
        expected[key] = {'position': list(box.container_position),
                         'quaternion_xyzw': list(box.container_orientation_xyzw),
                         'bbox': [list(box.container_bbox_min), list(box.container_bbox_max)]}
        for phase, field in zip(PHASES, ('phase1_content', 'phase2_content')):
            content = box.content_for_phase(phase)
            if content is None:
                continue
            content_key = f'{phase}_content_{box.box_index:02d}'
            keys[phase].append(content_key)
            bindings[content_key] = {'name': f'{_replay().CONTENT_NAME_PREFIX}{field}_{content.category}_{box.box_index}',
                                     'category': content.category, 'model': content.model}
            relations[content_key] = key
            expected[content_key] = {'position': ((np.array(box.container_bbox_min) +
                                                   np.array(box.container_bbox_max)) / 2).tolist(),
                                     'quaternion_xyzw': [0., 0., 0., 1.],
                                     'bbox_size_m': list(content.bbox_size_m)}
    bindings['floor'] = {'name': payload['floor_name'], 'category': 'floors'}
    if len({b['name'] for b in bindings.values()}) != len(bindings):
        raise ValueError('Original native identities must be distinct.')
    room = question['render']['room_bbox']
    bounds = _vector(room['expanded_bbox_world_xy'], 4)
    if np.any(bounds[2:] <= bounds[:2]):
        raise ValueError('Original room bounds are unresolved.')
    primary = map_records(question['render'].get('image')).get('image2', {})
    poses = map_records(primary.get('camera_poses'))
    camera = poses.get('image2.png') or next(iter(poses.values()), None)
    if camera is None:
        raise ValueError('Source primary observer pose is missing; GT closeups do not identify robot floor height.')
    _vector(camera['position'], 3)
    _angle(camera['quaternion_xyzw'], camera['quaternion_xyzw'])
    return {'raw': raw, 'payload': payload, 'plan': plan, 'replay_payload': minimal,
            'bindings': bindings, 'container_keys': container_keys, 'phase_keys': keys,
            'relations': relations, 'source_objects': expected, 'camera': camera,
            'bounds': bounds, 'room_id': int(room['room_id'])}


def _asset_root(context):
    default = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior')) / 'behavior-1k-assets'
    return Path(context.config.get('original_asset_root', str(default)))


@lru_cache(maxsize=4)
def _maps(asset_root, scene):
    import cv2
    cv2.setNumThreads(2)
    directory = Path(asset_root) / 'scenes' / scene / 'layout'
    paths = [directory / name for name in ('floor_trav_0.png', 'floor_insseg_0.png')]
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(im is None for im in images) or images[0].shape != images[1].shape:
        raise ValueError('Original full-furniture room/traversability maps are unavailable.')
    if images[0].shape[0] != images[0].shape[1]:
        raise ValueError('Original map frame is unresolved.')
    return *images, {str(path): sha256(path) for path in paths}


@lru_cache(maxsize=6)
def _clearance_map(asset_root, scene, room_id):
    """Cache an exact Euclidean map screen instead of repeated large erosion."""
    import cv2
    trav, rooms, hashes = _maps(asset_root, scene)
    valid = ((trav == 255) & (rooms == room_id)).astype(np.uint8)
    valid[[0, -1], :] = 0
    valid[:, [0, -1]] = 0
    distance = cv2.distanceTransform(valid, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    free = (distance > 40.0001).astype(np.uint8)
    _, components = cv2.connectedComponents(free, connectivity=8)
    return free, components, hashes


def _robot_start(layout, context):
    """Match the full-furniture builder; maps only propose, never qualify, R1."""
    free, components, hashes = _clearance_map(str(_asset_root(context)), layout['raw']['scene'], layout['room_id'])
    low, high = layout['bounds'][:2] + .45, layout['bounds'][2:] - .45
    camera = layout['camera']
    if np.any(high < low):
        return None
    proposed = [('source_primary_observer_projection', np.array(camera['position'][:2]))]
    proposed.extend(('original_room_grid', np.array([x, y])) for x in np.arange(low[0], high[0] + 1e-6, .15)
                    for y in np.arange(low[1], high[1] + 1e-6, .15))
    boxes = [_array(layout['source_objects'][key]['bbox']) for key in layout['container_keys']]
    candidates = []
    for label, xy in proposed:
        rc = (xy[::-1] / .01 + free.shape[0] / 2).astype(int)
        if np.any(rc < 0) or np.any(rc >= free.shape[0]) or not free[tuple(rc)]:
            continue
        if np.any(xy < low) or np.any(xy > high):
            continue
        if any(np.linalg.norm(xy - np.clip(xy, box[0, :2], box[1, :2])) < .55 for box in boxes):
            continue
        candidates.append((label, xy, rc))
    if not candidates:
        return None
    label, xy, rc = min(candidates, key=lambda row: (math.dist(row[1], camera['position'][:2]),
                                                   float(row[1][0]), float(row[1][1]), row[0]))
    component = int(components[tuple(rc)])
    connected = [p[1].tolist() for p in candidates if int(components[tuple(p[2])]) == component]
    look = np.mean([np.mean(box, axis=0) for box in boxes], axis=0)
    return {'position': [float(xy[0]), float(xy[1]), float(camera['position'][2] - 1.25)],
            'yaw_deg': math.degrees(math.atan2(look[1] - xy[1], look[0] - xy[0])),
            'source_observer_or_rule': label, 'original_map_sha256': hashes,
            'room_id': layout['room_id'], 'map_radius_m': .4, 'container_clearance_m': .55,
            'map_provenance': 'Installed TraversableMap._load_map: floor_trav_0.png for full-furniture scenes; '
                              'BaseMap.world_to_map uses reversed XY / 0.01 + image_size / 2.',
            'floor_height_provenance': 'Original primary camera = floor_top + AGENT_CAMERA_HEIGHT_M (1.25 m).',
            'connected_proposals_xy': connected, 'native_clearance_and_reach_qualified': False}


def _route_screen(layout, start):
    """Private geometry proposal for two distinct inspection positions."""
    for key in layout['container_keys']:
        box = _array(layout['source_objects'][key]['bbox'])
        center = box.mean(axis=0)[:2]
        points = [np.asarray(xy) for xy in start['connected_proposals_xy']
                  if .7 <= math.dist(xy, center) <= 1.8 and math.dist(xy, start['position'][:2]) >= .5]
        for i, first in enumerate(points):
            for second in points[i + 1:]:
                a, b = first - center, second - center
                angle = math.degrees(math.acos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.)))
                if math.dist(first, second) >= .65 and angle >= 60.:
                    return {'container': key, 'proposal_xy': [first.tolist(), second.tolist()],
                            'separation_m': math.dist(first, second), 'angle_deg': angle,
                            'native_visibility_and_path_qualified': False}
    return None


def configure_scene(config, parameters):
    """Reuse exact replay preload configs, with no mid-episode object placement."""
    if config['scene']['scene_model'] != parameters['scene_model'] or parameters['phase'] not in PHASES:
        raise ValueError('Source snapshot restoration belongs to a different scene/phase.')
    if not config.get('robots') or config['robots'][0].get('type') != 'R1':
        raise ValueError('The audited active pipeline must supply its installed R1 configuration.')
    plan = _replay().build_unobserved_replay_plan(parameters['replay_payload'], validate_selector=False)
    objects = _replay()._preloaded_object_configs(plan)
    active_names = {parameters['bindings'][key]['name'] for key in parameters['phase_keys'][parameters['phase']]}
    prior = {item.get('name') for item in config.setdefault('objects', [])}
    for item in objects:
        if item['name'] in prior:
            raise ValueError('Source restoration cannot overwrite an existing object.')
        key = next(key for key, binding in parameters['bindings'].items() if binding['name'] == item['name'])
        if item['name'] in active_names:
            item['position'] = parameters['source_objects'][key]['position']
        config['objects'].append(item)
    return {'source_provenance': parameters['provenance'], 'phase': parameters['phase'],
            'preloaded_objects': objects, 'source_generator_sha256': parameters['generator_sha256'],
            'replay_helper_sha256': parameters['replay_sha256'], 'mid_episode_pose_changes': False,
            'inactive_phase_policy': 'Original replay fixed visual-only offstage preload; never an Agent observation.'}


def initialize_scene(session, parameters):
    """Restore visible closed source lids once and enable private sensor evidence."""
    from omnigibson.object_states import Open
    if (session.scene_name, session.room_name) != (parameters['scene_model'], parameters['room']):
        raise ValueError('Initialization belongs to a different source scene/room.')
    if hasattr(session, '_unobserved_change_initialization'):
        raise ValueError('Original snapshot restoration runs only once at initialization.')
    evidence = []
    for key in parameters['container_keys']:
        binding = parameters['bindings'][key]
        obj = session.env.scene.object_registry('name', binding['name'])
        if obj is None or obj.category != binding['category'] or obj.model != binding['model']:
            raise ValueError('An exact original container could not be restored.')
        links = []
        if Open in obj.states:
            state = obj.states[Open]
            state.set_value(False)
            for joint in state.relevant_joints_info[1]:
                for link in obj.links.values():
                    if link.prim_path == joint.body1:
                        link.visible = True
                        links.append(link.prim_path)
            closed = not bool(state.get_value())
            if not closed:
                raise ValueError('Original container failed initialization-only closing.')
        else:
            closed = None
        evidence.append({'key': key, 'name': obj.name, 'closed': closed, 'restored_lid_links': links,
                         'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only)})
    sensors = []
    # Extra segmentation stays private. The Agent still receives its ordinary
    # RGB cameras, without names, source models, masks, poses or scorer state.
    for name, sensor in session.env.robots[0].sensors.items():
        if 'rgb' in sensor.modalities:
            if 'seg_instance' not in sensor.modalities:
                sensor.add_modality('seg_instance')
            sensors.append(name)
    session._unobserved_change_initialization = {'phase': parameters['phase'], 'containers': evidence}
    return {'source_provenance': parameters['provenance'], 'containers': evidence,
            'private_segmentation_sensors': sensors, 'phase': parameters['phase'],
            'human_framing_names': [parameters['bindings'][key]['name'] for key in
                parameters['container_keys'] + parameters['phase_keys'][parameters['phase']]],
            'original_reference_cameras_preserved': True, 'historical_agent_observations_created': False,
            'warning_codes': ['assumption_review_required', 'independent_phase_snapshots_only']}


def _snapshot(obj, *, dynamic=False):
    position, quaternion = obj.get_position_orientation()
    low, high = [_array(v) for v in obj.aabb]
    row = {'name': obj.name, 'category': obj.category, 'model': obj.model,
           'position': _array(position).tolist(), 'quaternion_xyzw': _array(quaternion).tolist(),
           'scale': _array(obj.scale).tolist(),
           'bbox': [low.tolist(), high.tolist()], 'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base)}
    if dynamic:
        row.update(linear_speed_m_s=float(np.linalg.norm(_array(obj.get_linear_velocity()))),
                   angular_speed_rad_s=float(np.linalg.norm(_array(obj.get_angular_velocity()))))
    return row


def _stationary(row):
    return row['linear_speed_m_s'] <= .02 and row['angular_speed_rad_s'] <= .1


def _near(row, original, tolerance=.04):
    return math.dist(row['position'], original['position']) <= tolerance and _angle(
        row['quaternion_xyzw'], original['quaternion_xyzw']) <= 10.


def _configuration(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') not in OPERATIONS or
            (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('Use the owned source snapshot operation in its original scene/room.')
    p = config['parameters']
    keys = p['container_keys'] + p['phase_keys']['image1'] + p['phase_keys']['image2'] + ['floor']
    if (len(set(keys)) != len(keys) or set(config['binding_keys']) != set(keys) or set(config['bindings']) != set(keys)
            or len(set(config['bindings'].values())) != len(keys)):
        raise ValueError('Complete distinct original snapshot bindings are required.')
    phase = session._unobserved_change_initialization['phase']
    if phase not in PHASES:
        raise ValueError('The original source snapshot was not initialized.')
    return p, phase


def _robot_state(session, controller, floor):
    pose = controller.pose()
    low, high = [_array(v) for v in session.robot.links[session.robot.base_footprint_link_name].aabb]
    regions = [session.env.scene.seg_map.get_room_instance_by_point(np.array([x, y]))
               for x in (low[0], (low[0] + high[0]) / 2, high[0])
               for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    scene_objects = session.env.scene.objects
    scene_objects = scene_objects.values() if isinstance(scene_objects, dict) else scene_objects
    floor_paths = [obj.prim_path for obj in scene_objects if obj.category in ('floor', 'floors')]
    contacts, obstacles = [], []
    for contact in session.robot.contact_list():
        for raw in (contact.body0, contact.body1):
            body = str(raw)
            if body == session.robot.prim_path or body.startswith(session.robot.prim_path + '/'):
                continue
            contacts.append(body)
            if not any(body == path or body.startswith(path + '/') for path in floor_paths):
                obstacles.append(body)
    return {**pose, 'in_original_room': all(r == session.room_name for r in regions),
            'supported': any(body == floor.prim_path or body.startswith(floor.prim_path + '/') for body in contacts),
            'nonfloor_contacts': obstacles}


def _measure(session, controller, config, objects):
    from omnigibson.object_states import Inside, Touching
    p, phase = _configuration(session, config)
    containers = {key: _snapshot(objects[key]) for key in p['container_keys']}
    active = {}
    for key in p['phase_keys'][phase]:
        row = _snapshot(objects[key], dynamic=True)
        row['inside'] = bool(objects[key].states[Inside].get_value(objects[p['relations'][key]]))
        row['touching_container'] = bool(objects[key].states[Touching].get_value(objects[p['relations'][key]]))
        active[key] = row
    inactive = {key: {'position': _array(objects[key].get_position_orientation()[0]).tolist(),
                      'fixed_base': bool(objects[key].fixed_base), 'visual_only': bool(objects[key].visual_only)}
                for other in PHASES if other != phase for key in p['phase_keys'][other]}
    return {'phase': phase, 'containers': containers, 'active_contents': active, 'inactive_contents': inactive,
            'robot': _robot_state(session, controller, objects['floor']), 'gripper': controller.debug_gripper_state()}


def _source_checks(config, measured, objects, *, check_initial_dimensions=True):
    p, active, robot = config['parameters'], measured['active_contents'], measured['robot']
    source = p['source_objects']
    checks = {'original_floor_collidable': not objects['floor'].visual_only,
            'original_native_identities': all(all(getattr(objects[key], field) == value for field, value in identity.items())
                                              for key, identity in p['identities'].items()),
            'original_containers_fixed_collidable': all(row['fixed_base'] and not row['visual_only']
                                                       for row in measured['containers'].values()),
            'original_container_root_layout': all(_near(row, source[key], .03)
                for key, row in measured['containers'].items()),
            'active_contents_movable_collidable': all(not row['fixed_base'] and not row['visual_only'] for row in active.values()),
            'inactive_contents_offstage_fixed_visual_only': all(row['fixed_base'] and row['visual_only'] and
                row['position'][0] > 900. and row['position'][1] > 900. for row in measured['inactive_contents'].values()),
            'robot_in_original_room': robot['in_original_room'], 'robot_supported': robot['supported']}
    if check_initial_dimensions:
        checks.update(original_closed_container_dimensions=all(
            np.max(np.abs(_array(row['bbox']) - _array(source[key]['bbox']))) <= .04
            for key, row in measured['containers'].items()),
            active_contents_original_dimensions=all(np.max(np.abs((_array(row['bbox'])[1] - _array(row['bbox'])[0]) -
                _array(source[key]['bbox_size_m']))) <= .03 for key, row in active.items()))
    return checks


def qualify_operation(session, executor, config, bound):
    """Only read measured native state; reaching/opening is not pre-certified."""
    try:
        measured = _measure(session, executor, config, bound)
        checks = _source_checks(config, measured, bound)
        checks.update(contents_initially_inside_supported_stationary=all(row['inside'] and row['touching_container'] and _stationary(row)
                                                               for row in measured['active_contents'].values()),
                      initial_gripper_empty=measured['gripper'].get('object_name') is None and
                                            measured['gripper'].get('constraint_active') is False,
                      robot_initially_clear=not measured['robot']['nonfloor_contacts'])
        return {'accepted': bool(all(checks.values())), 'checks': checks, 'native_state': measured,
                'reason': None if all(checks.values()) else 'native_snapshot_preconditions_failed',
                'reach_opening_grasp_route_and_terminal_success_qualified': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_snapshot_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'physical_execution_qualified': False}


class SnapshotOperationEvaluator:
    """Credit real extraction/return or a distinct native inspection route."""
    def __init__(self, session, controller, config, output):
        self.parameters, self.phase = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('An original native snapshot object is missing.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial, self.episode = None, 0

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = _measure(self.session, self.controller, self.config, self.objects)
        checks = _source_checks(self.config, self.initial, self.objects)
        gripper = self.initial['gripper']
        self.initial_valid = (all(checks.values()) and not self.initial['robot']['nonfloor_contacts'] and
            gripper.get('object_name') is None and gripper.get('constraint_active') is False and
            all(row['inside'] and row['touching_container'] and _stationary(row)
                for row in self.initial['active_contents'].values()))
        self.history_valid, self.wrong_grasp = True, False
        self.lifted, self.completed, self.visits = set(), set(), {}
        self.stable_steps, self.last_action_step = 0, observation.action_step
        self.initial_action_step = observation.action_step
        self.record(observation, action=None)

    def _update(self, count_step=False):
        if self.initial is None:
            return None
        measured = _measure(self.session, self.controller, self.config, self.objects)
        contents, gripper = measured['active_contents'], measured['gripper']
        checks = _source_checks(self.config, measured, self.objects, check_initial_dimensions=False)
        # Articulation and grasp rotation legitimately change world AABBs. Use
        # closed initial dimensions plus unchanged native scale/root evidence.
        checks['native_scales_unchanged'] = all(np.allclose(row['scale'], self.initial[group][key]['scale'], atol=1e-6, rtol=0.)
            for group in ('containers', 'active_contents') for key, row in measured[group].items())
        self.history_valid = self.history_valid and measured['phase'] == self.phase and all(checks.values())
        names = {row['name']: key for key, row in contents.items()}
        held = names.get(gripper.get('object_name')) if (gripper.get('holding') is True and
                                                        gripper.get('constraint_active') is True) else None
        if gripper.get('object_name') is not None and gripper['object_name'] not in names:
            self.wrong_grasp = True
        lifts = {key: row['bbox'][0][2] - self.initial['active_contents'][key]['bbox'][0][2]
                 for key, row in contents.items()}
        extracted = False
        if held:
            row = contents[held]
            container = self.initial['containers'][self.parameters['relations'][held]]
            extracted = (not row['inside'] and lifts[held] >= MINIMUM_LIFT_M and
                         row['bbox'][0][2] >= container['bbox'][1][2] + EXTRACTION_CLEARANCE_M)
            if extracted:
                self.lifted.add(held)
        empty_open = (gripper.get('object_name') is None and gripper.get('constraint_active') is False and
                      gripper.get('gripper_open') is True)
        all_returned = all(row['inside'] and row['touching_container'] and _stationary(row)
                           and _near(row, self.initial['active_contents'][key])
                           for key, row in contents.items())
        if empty_open:
            for key in self.lifted:
                row = contents[key]
                if row['inside'] and row['touching_container'] and _stationary(row) and _near(row, self.initial['active_contents'][key]):
                    self.completed.add(key)
        operation = self.config['operation']
        route_required = operation == 'box_inspection_route' or not contents
        route_complete = any(len(visits) >= 2 for visits in self.visits.values())
        at_start = math.dist(measured['robot']['position'][:2], self.initial['robot']['position'][:2]) <= .2
        other_contents_preserved = all(row['inside'] and row['touching_container'] and _stationary(row) and
            _near(row, self.initial['active_contents'][key]) for key, row in contents.items() if key != held)
        operated = (route_complete and at_start and empty_open and all_returned if route_required else
                    bool(held and extracted and other_contents_preserved) if operation == 'phase_inventory_extract' else
                    bool(self.completed) and empty_open and all_returned)
        # Navigation must be contact-free. Legitimate target/lid contacts during
        # manipulation are handled by the existing guarded controls.
        navigation_clear = not measured['robot']['nonfloor_contacts'] if route_required else True
        terminal = (self.initial_valid and self.history_valid and not self.wrong_grasp and navigation_clear and
                    operated and self.last_action_step > self.initial_action_step)
        if count_step:
            self.stable_steps = self.stable_steps + 1 if terminal else 0
        elif not terminal:
            self.stable_steps = 0
        return {'native_state': measured, 'held_binding': held, 'native_lift_m': lifts,
                'extracted_bindings': sorted(self.lifted), 'returned_bindings': sorted(self.completed),
                'inspection_visits': self.visits, 'criteria': {'initial_native_state_valid': self.initial_valid,
                    'source_native_history_valid': self.history_valid, 'no_wrong_grasp': not self.wrong_grasp,
                    'physical_operation_complete': bool(operated), 'guarded_action_history_present':
                        self.last_action_step > self.initial_action_step, 'navigation_clear': navigation_clear},
                'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS,
                'task_success': bool(terminal and self.stable_steps >= STABILITY_STEPS)}

    def _record_visible_visits(self, observation, measured):
        """Read existing private segmentation only for RGB delivered to the Agent."""
        if observation.action_step <= self.initial_action_step:
            return
        robot = measured['robot']
        if not robot['in_original_room'] or not robot['supported'] or robot['nonfloor_contacts']:
            return
        xy = _array(robot['position'])[:2]
        if math.dist(xy, self.initial['robot']['position'][:2]) < .5:
            return
        visible = set()
        for view, name in self.session.robot_camera_registry.items():
            delivered = observation.rgb_images().get(view)
            if delivered is None:
                continue
            sensor = self.session.robot.sensors[name]
            native, info = sensor.get_obs()
            if 'seg_instance' not in native or not np.array_equal(_array(native['rgb'])[..., :3], delivered):
                continue
            labels = info.get('seg_instance', {})
            for key, row in measured['containers'].items():
                ids = [int(label) for label, object_name in labels.items() if object_name == row['name']]
                mask = np.isin(_array(native['seg_instance']), ids)
                ys, xs = np.nonzero(mask)
                if len(xs) >= 64 and min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) >= 8:
                    visible.add(key)
        for key in sorted(visible):
            center = _array(measured['containers'][key]['bbox']).mean(axis=0)[:2]
            if not .7 <= math.dist(xy, center) <= 1.8:
                continue
            entries = self.visits.setdefault(key, [])
            if len(entries) >= 2:
                continue
            if entries:
                previous = _array(entries[0]['xy'])
                a, b = previous - center, xy - center
                angle = math.degrees(math.acos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.)))
                if math.dist(previous, xy) < .65 or angle < 60.:
                    continue
            entries.append({'xy': xy.tolist(), 'snapshot_id': observation.snapshot_id,
                            'action_step': observation.action_step, 'native_rgb_segmentation_match': True})

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        if action is not None:
            self.last_action_step = max(self.last_action_step, observation.action_step)
        measured = _measure(self.session, self.controller, self.config, self.objects)
        if self.config['operation'] == 'box_inspection_route' or not measured['active_contents']:
            self._record_visible_visits(observation, measured)
        value = self._update()
        value.update(schema='private_unobserved_snapshot_operation_v1', episode=self.episode,
                     snapshot_id=observation.snapshot_id, action_step=observation.action_step,
                     action=action, configuration=self.config, initial_native_state=self.initial)
        write_json(self.output / 'state.json', value)
        import json
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return SnapshotOperationEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Temporal Understanding', 'Unobserved Change')
    version = 'unobserved_change_v1.3'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category:
            return {'eligible': False, 'reason': 'source_category_mismatch'}
        try:
            layout = _layout(source)
            return {'eligible': True, 'operation_eligible': True, 'box_count': len(layout['container_keys']),
                    'phase_content_counts': {phase: len(keys) for phase, keys in layout['phase_keys'].items()},
                    'saved_view_audit': layout['plan'].audit_dict()}
        except (ValueError, KeyError, TypeError) as error:
            return {'eligible': True, 'operation_eligible': False, 'reason': 'not_convertible_under_L1_L2',
                    'diagnostic': f'{type(error).__name__}: {error}'}

    def convert(self, source, context):
        result = CategoryConverter.convert_existing(self, source, context)
        task = result.get('task_definition')
        if task is not None:
            from .partial_occlusion import TASK_REVISION
            parameters = task['private']['evaluation']['parameters']
            parameters['task_revision'] = TASK_REVISION
            parameters['mechanism_contract'] = {
                'preserved_relations': ['source temporal event', 'explicit historical saved-phase snapshot semantics',
                    'existing budgets and movement criteria'],
                'required_native_evidence': ['informative source view access in the selected saved phase',
                    'source snapshot geometry and mounted-camera observations'],
                'continuous_two_phase_episode': False,
                'movement_boundary_sensitivity': 'Existing thresholds retained; near-boundary outcomes require review.',
                'native_operation_verified': False}
            result.setdefault('warning_codes', []).append('both_phase_informative_access_pending')
            result['temporal_policy_unchanged'] = True
        return result

    def convert_existing(self, source, context):
        inspection = self.inspect(source)
        if not inspection.get('operation_eligible'):
            return self._diagnostic(source, context, 'source_layout', 'not_convertible_under_L1_L2',
                                    inspection.get('diagnostic', inspection.get('reason')))
        layout = _layout(source)
        continuous_accessibility = context.config.get('continuous_temporal_accessibility') is True
        phase = context.config.get('unobserved_initial_phase', 'image2')
        operation = context.config.get('unobserved_operation', OPERATIONS[0])
        if phase not in PHASES or operation not in OPERATIONS:
            raise ValueError('Unsupported original phase/operation recipe.')
        missing = [binding for key, binding in layout['bindings'].items() if key != 'floor' and not
                   (_asset_root(context) / 'objects' / binding['category'] / binding['model'] / 'usd').is_dir()]
        if missing:
            return self._diagnostic(source, context, 'asset_availability', 'original_asset_unavailable', missing,
                                    bindings=layout['bindings'])
        start = _robot_start(layout, context)
        if start is None:
            if not continuous_accessibility:
                return self._diagnostic(source, context, 'robot_start_screen', 'no_evidenced_r1_start_in_original_room',
                    'The conservative full-furniture map screen found no robot proposal; this does not prove L1/L2 impossibility.',
                    bindings=layout['bindings'])
            start = {'pending_shared_spawn_repair': True, 'connected_proposals_xy': [],
                'requested_source_region': layout['raw']['room'],
                'source_observer_or_rule': 'Native same-room spawn selection required; no CPU robot pose proposed.',
                'native_clearance_and_reach_qualified': False}
        route = None if start.get('pending_shared_spawn_repair') else _route_screen(layout, start)
        if not continuous_accessibility and (operation == 'box_inspection_route' or not layout['phase_keys'][phase]) and route is None:
            return self._diagnostic(source, context, 'inspection_route_screen', 'no_evidenced_two_view_inspection_route',
                'Connected map proposals do not support the declared two-view route; native evidence may justify a narrower rule.',
                bindings=layout['bindings'])
        initialization = {field: layout[field] for field in ('replay_payload', 'bindings', 'container_keys',
                          'phase_keys', 'relations', 'source_objects')}
        initialization.update(scene_model=layout['raw']['scene'], room=layout['raw']['room'],
            source_identity=source.relative_path, phase=phase, builder_sha256=sha256(SOURCE_BUILDER),
            generator_sha256=sha256(PRODUCER), replay_sha256=sha256(REPLAY),
            provenance='Original JSON boxes/GT cameras; passive_gt.unobserved_regeneration '
                       'build_unobserved_replay_plan and _preloaded_object_configs; saved AABB-center placement '
                       'from _place_preloaded_phase_content and producer _try_place_inside_container.')

        def patches_for(selected_phase):
            parameters = {**initialization, 'phase': selected_phase}
            patches = [{'operation': 'category_initialization', 'phase': 'before_construction',
                        'module': MODULE, 'parameters': parameters}]
            if not start.get('pending_shared_spawn_repair'):
                patches.append({'operation': 'robot_pose', 'position': start['position'], 'yaw_deg': start['yaw_deg'],
                    'head_pitch_deg': 35.})
            for when in ('before_construction', 'after_postprocessing'):
                patches.extend({'operation': 'object_physics', 'phase': when, 'binding': key, 'visual_only': False}
                               for key in layout['container_keys'])
                patches.extend({'operation': 'object_physics', 'phase': when, 'binding': key, 'visual_only': False,
                                **({'fixed_base': False} if when == 'before_construction' else {})}
                               for key in layout['phase_keys'][selected_phase])
            patches.append({'operation': 'category_initialization', 'phase': 'after_postprocessing',
                            'module': MODULE, 'parameters': parameters})
            return patches

        p = {field: layout[field] for field in ('container_keys', 'phase_keys', 'relations', 'source_objects')}
        p['identities'] = layout['bindings']
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(layout['bindings']), 'parameters': p}
        physical_goal = ('Keep a selected item grasped above its original box rim.' if operation == 'phase_inventory_extract' else
                         'Return the selected item to the same original box, release it and let it settle.')
        objective = ('Physically inspect the exteriors of the original boxes from two separated viewpoints and return to your starting place.'
                     if operation == 'box_inspection_route' else
                     'Inspect the boxes in this saved scene snapshot. Choose one currently present item, open its lid with '
                     'your hand if necessary, grasp the item and lift it completely above its original box. ' + physical_goal +
                     ' If all boxes are empty, instead inspect the exterior of one original box from two separated viewpoints and return to your starting place.')
        public = {'objective': objective,
            'success_criteria': ('For an item operation, lift at least 5 cm and clear the original rim by 1 cm while held. ' +
                physical_goal + ' A returned item must be inside and in native contact with its original box, '
                'stationary and near its starting placement, with an empty open gripper. '
                'For an inspection route, walk to two positions 0.7–1.8 m from the same box, '
                'at least 0.65 m and 60 degrees apart; each position must be at least 0.5 m from the start and provide '
                'a visible box in an enabled robot RGB camera. Return within 0.2 m of the start with an empty open gripper. '
                'Keep other contents in their original boxes and all box roots unchanged. Sustain the terminal state for 15 control samples.'),
            'environment': 'Original room, original container models and saved contents of one independently initialized phase. '
                           'There is no automatic temporal replay or supplied historical Agent image.',
            'constraints': ['Use existing guarded robot controls for navigation, lid contact, grasping and placement.',
                            'Preserve original assets, dimensions, original box placements and the observer reference camera.',
                            'The original before/after QA is private auxiliary evaluation and does not prove physical success.']}
        patches = patches_for(phase)
        scene = original_scene(source, context, bindings=layout['bindings'], patches=patches)
        task = physical_task(source, context, public=public, evaluation=evaluation,
                             bindings=layout['bindings'], patches=patches)
        for definition in (scene, task):
            for selected in PHASES:
                definition['scene']['variants']['saved_' + selected] = {'patches': patches_for(selected)}
        assumptions = [
            {'doubt': 'The active loader omits both phase scenes; the shared robot runner has no source-specific legitimate before/after history handoff.',
             'chosen_assumption': 'Restore both saved phase snapshots as independent initialization variants; default to phase 2. Define physical snapshot operations, not temporal QA success.',
             'alternatives': ['Integrate measured two-phase observation history in shared code before a temporal-memory task.', 'Use the original empty variant as a source-only diagnostic.'],
             'scope': 'All sources and both saved phase variants.',
             'risk': 'The operation does not evaluate unobserved-change memory or preserve one continuous original episode history.'},
            {'doubt': 'Original replay uses fixed visual-only objects, content at AABB centers, and temporary lid hiding for render views.',
             'chosen_assumption': 'Keep exact assets and source placements; restore visible closed original lids and fixed collidable container roots; make active contents movable with object_physics before construction and after postprocessing.',
             'alternatives': ['Keep original visual-only diagnostic reconstruction.', 'Use the navigation recipe if manipulation is unreachable.'],
             'scope': 'Adapted source snapshots only; original variant remains empty.',
             'risk': 'Gravity can settle a suspended render object, lids may prevent grasping, and floor-level objects may be unreachable. Native containment, size, support and clearance gates must pass.'},
            {'doubt': 'A source map and observer-derived floor height cannot prove R1 body clearance, camera visibility or reach.',
             'chosen_assumption': 'Use a deterministic robot-only pose screened on full-furniture room/traversability maps with a 0.4 m disk and 0.55 m container clearance; retain native qualification.',
             'alternatives': ['Select another native-qualified robot stance after Review.', 'Retain a diagnostic if no source-faithful stance is available.'],
             'scope': 'Robot initialization and optional navigation proposal only.',
             'risk': 'Map errors, closed doors, sensor coverage, controller drift and true 3D obstacles may reject the CPU proposal.'}]
        return {'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'change_summary': ['Restore original box and both content snapshots at initialization only.',
                'Enable original active contents and container collisions without resizing or relocation.',
                'Add a screened robot-only start and private native operation predicates; preserve empty original variant.'],
            'warning_codes': ['assumption_review_required', 'independent_phase_snapshots_only', 'native_operation_unqualified',
                              'original_render_physics_restored'] +
                (['source_map_start_pending_shared_native_repair'] if start.get('pending_shared_spawn_repair') else []),
            'assumptions': assumptions, 'qa_preserved': True,
            'operation_recipe': operation, 'initial_phase': phase, 'source_layout_evidence': inspection,
            'robot_initialization_evidence': start, 'inspection_route_proposal': route,
            'limitations': ['No native loading, reaching, opening, grasping, physical success or human approval is certified by CPU conversion.',
                'Historical QA reference images remain private; no new Agent phase history is fabricated.',
                'The original replay code documents a carton AABB PhysX crash on an earlier OG build. Current native safety of this property is unverified.'],
            'attempted_repairs': ['Rule-based original replay reconstruction, native collision restoration, source-map R1 proposal.'],
            'next_step': 'Capture all generated and diagnostic records through the shared runner; inspect original lids, native containment and guarded-controller readbacks before qualification.'}

    def _diagnostic(self, source, context, stage, reason, evidence, *, bindings=None):
        return {'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': None,
                'status': 'conversion_failed', 'failure_stage': stage, 'failure_reason': reason,
                'diagnostic_reason': evidence, 'warning_codes': ['source_only_diagnostic', 'native_operation_unqualified'],
                'limitations': ['No physical task generated. Source-only images remain diagnostic; failure must enter Review.'],
                'next_step': 'Review the exact source mechanism, original asset installation and native robot clearance; preserve source answers and assets.'}

    def convert_all(self, sources, context):
        """Use shared idempotent batching; retain every failed source for Review."""
        sources = sorted(sources, key=lambda source: source.relative_path)
        manifest = super().convert_all(sources, context)
        manifest.update(operation_counts=dict(Counter(row.get('operation_recipe', 'diagnostic') for row in manifest['records'])),
                        failure_distribution=dict(Counter(row['failure_reason'] for row in manifest['records'] if row['failure_reason'])),
                        physical_execution_certified=False)
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest


def source_change_recipe(layout, *, allow_multiple=False):
    """Read actual source event membership without inventing unsupported branches."""
    slots = {phase: {key: None for key in layout['container_keys']} for phase in PHASES}
    for phase in PHASES:
        for key in layout['phase_keys'][phase]:
            container = layout['relations'][key]
            if slots[phase][container] is not None:
                raise ValueError('Multiple source contents in one slot are unsupported.')
            slots[phase][container] = key
    identity = lambda key: None if key is None else tuple(layout['bindings'][key][x] for x in ('category', 'model'))
    first, second = slots['image1'], slots['image2']
    changed = [key for key in first if identity(first[key]) != identity(second[key])]
    if not changed:
        return {'branch': 'no_change', 'changed_containers': [], 'phase_1_slots': first, 'phase_2_slots': second}
    if len(changed) == 1:
        container = changed[0]
        branch = 'added' if first[container] is None else 'removed' if second[container] is None else 'replaced'
        return {'branch': branch, 'changed_containers': changed, 'source_container': container,
            'original_key': first[container], 'current_key': second[container],
            'phase_1_slots': first, 'phase_2_slots': second}
    removed = [key for key in changed if first[key] is not None and second[key] is None]
    added = [key for key in changed if first[key] is None and second[key] is not None]
    if len(removed) == len(added) == 1 and identity(first[removed[0]]) == identity(second[added[0]]):
        return {'branch': 'moved', 'changed_containers': changed, 'source_container': removed[0],
            'destination_container': added[0], 'original_key': first[removed[0]], 'current_key': second[added[0]],
            'phase_1_slots': first, 'phase_2_slots': second}
    if not allow_multiple:
        raise ValueError('Source has multiple ambiguous changes; retain independent source protocol.')
    return {'branch': 'multiple', 'changed_containers': changed,
        'phase_1_slots': first, 'phase_2_slots': second,
        'events': [{'branch': 'added' if first[key] is None else
                    'removed' if second[key] is None else 'replaced',
                    'source_container': key, 'original_key': first[key], 'current_key': second[key]}
                   for key in changed]}

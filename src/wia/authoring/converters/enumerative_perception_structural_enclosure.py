"""Convert original enclosures into private measured door-operation tasks.

Run on CPU with the behavior Python environment from the repository root:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Only initialization hooks restore original mechanisms. Qualification and
evaluation read native state; all actions use the existing guarded controls.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.resources import ESI_ROOT

import ast
from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from wia.authoring.conversion import ROOT, ConverterBase, map_records, original_scene, physical_task
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import sha256, write_json

MODULE = 'core.esi_converters.enumerative_perception_structural_enclosure'
OPERATIONS = ('inspect_all_enclosures', 'inspect_any_enclosure')
SOURCE_COMMON = ESI_ROOT / 'src/active_explore/tasks/enumerative_perception/common.py'
SOURCE_GENERATOR = ESI_ROOT / 'src/dataset_generation/task_counting/batch_counting_merge.py'
MINIMUM_ANGLE_RAD = math.radians(30.)
MINIMUM_SLIDE_M = .06
CLOSED_ANGLE_RAD = math.radians(5.)
CLOSED_SLIDE_M = .005
LAYOUT_TOLERANCE_M = .05
STABILITY_STEPS = 15


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
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Source/native measurements must be finite.')
    return value


def _vector(value, size):
    value = _array(value)
    if value.shape != (size,):
        raise ValueError('Source/native vector shape is unresolved.')
    return value


def _angle(first, second):
    first, second = _vector(first, 4), _vector(second, 4)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-8:
        raise ValueError('Source/native orientation is unresolved.')
    return math.degrees(2 * math.acos(min(1., abs(float(np.dot(first, second) / denominator)))))


def _near(row, original, tolerance=LAYOUT_TOLERANCE_M):
    return math.dist(row['position'], original['position']) <= tolerance and _angle(
        row['quaternion_xyzw'], original['quaternion_xyzw']) <= 5.


@lru_cache(maxsize=1)
def _source_helpers():
    """Compile only the source's pure helpers; importing OG would start a GPU."""
    names = {'normalize_text', 'extract_primary_camera_pose', 'initial_camera', 'build_env_objects'}
    tree = ast.parse(SOURCE_COMMON.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Audited original pure source helpers are unavailable.')
    namespace = {'np': np, 'Path': Path, 'Any': object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_COMMON), 'exec'), namespace)
    return namespace


def _layout(source):
    raw, payload = source.raw, source.payload
    if payload.get('task_type') != 'hidden_in_box':
        raise ValueError('The source enclosure mechanism is unresolved.')
    question, render = payload['question_data'], payload['question_data']['render']
    match = re.fullmatch(r'Considering both images, how many objects of category "([^"\n]+)" '
                         r'are present in this scene\?', raw['question'].strip())
    if not match or question.get('count_target') != match[1]:
        raise ValueError('Public question/category disagrees with the source task.')
    resolved = render.get('resolved_objects')
    if not resolved or not resolved.get('containers') or not resolved.get('targets'):
        raise ValueError('missing_resolved_enclosure_layout')
    if resolved.get('confusers'):
        raise ValueError('Unresolved enclosure confuser mechanism.')
    helpers = _source_helpers()
    native = helpers['build_env_objects'](payload)
    camera = helpers['initial_camera'](payload)[2]['camera_pose']
    _vector(camera['position'], 3)
    _angle(camera['quaternion_xyzw'], camera['quaternion_xyzw'])
    objects, bindings, expected = {}, {}, {}
    keys = {'containers': [], 'targets': []}
    for group, prefix in (('containers', 'container'), ('targets', 'content')):
        for index, item in enumerate(sorted(resolved[group], key=lambda item: item['name'])):
            key = f'{prefix}_{index:02d}'
            if not item.get('model') or item['name'] in objects:
                raise ValueError('Original object identities/models are unresolved.')
            if group == 'containers' and (item['category'] != 'microwave' or item.get('open_state') is not False):
                raise ValueError('This rule requires original initially closed microwaves.')
            if group == 'targets' and item['category'] != match[1]:
                raise ValueError('Original count category is unresolved.')
            _vector(item['position'], 3)
            _angle(item['quaternion_xyzw'], item['quaternion_xyzw'])
            low, high = [_vector(part, 3) for part in item['bbox']]
            if np.any(high <= low):
                raise ValueError('Original object dimensions are unresolved.')
            objects[item['name']] = item
            bindings[key] = {field: item[field] for field in ('name', 'category', 'model')}
            expected[key] = {field: item[field] for field in ('position', 'quaternion_xyzw', 'bbox')}
            keys[group].append(key)
    if {obj['name'] for obj in native} != set(objects) or len(native) != len(objects):
        raise ValueError('Original builder does not reconstruct every resolved object.')
    containers = {bindings[key]['name']: key for key in keys['containers']}
    relations = {key: containers[objects[bindings[key]['name']]['container_name']] for key in keys['targets']}
    if not payload.get('floor_name'):
        raise ValueError('Original floor identity is unresolved.')
    bindings['floor'] = {'name': payload['floor_name'], 'category': 'floors'}
    bounds = _vector(render['room_bbox']['expanded_bbox_world_xy'], 4)
    if np.any(bounds[2:] <= bounds[:2]):
        raise ValueError('Original room bounds are unresolved.')
    return {'raw': raw, 'payload': payload, 'render': render, 'camera': camera, 'objects': objects,
            'bindings': bindings, 'source_objects': expected, 'container_keys': keys['containers'],
            'content_keys': keys['targets'], 'relations': relations, 'bounds': bounds}


def _asset_root(context):
    data = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    return Path(context.config.get('original_asset_root', str(data / 'behavior-1k-assets')))


@lru_cache(maxsize=4)
def _maps(asset_root, scene_model):
    import cv2
    cv2.setNumThreads(2)
    directory = Path(asset_root) / 'scenes' / scene_model / 'layout'
    # This source case loads only floors/walls, not the original furniture.
    # The installed TraversableMap explicitly uses this map when objects are
    # absent. Generated container AABBs are screened independently below.
    paths = [directory / name for name in ('floor_trav_no_obj_0.png', 'floor_insseg_0.png')]
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(image is None for image in images) or images[0].shape != images[1].shape:
        raise ValueError('Original room/traversability maps are unavailable.')
    if images[0].shape[0] != images[0].shape[1]:
        raise ValueError('Original map frame is unresolved.')
    return *images, {str(path): sha256(path) for path in paths}


def _robot_start(layout, context):
    """Private map proposal only; native body clearance remains a separate gate."""
    trav, rooms, checksums = _maps(str(_asset_root(context)), layout['raw']['scene'])
    room_id = layout['render']['room_bbox']['room_id']
    radius, clearance = 40, .6
    delta = np.arange(-radius, radius + 1)
    disk = delta[:, None] ** 2 + delta[None, :] ** 2 <= radius ** 2
    low, high = layout['bounds'][:2] + clearance, layout['bounds'][2:] - clearance
    if np.any(high < low):
        return None
    camera = layout['camera']
    points = [('reference_projection', np.clip(camera['position'][:2], low, high))]
    for view in layout['render'].get('multi_image_input', {}).get('views', []):
        if view.get('is_traversable') is True:
            points.append((view['view_id'], np.clip(view['camera_pose']['position'][:2], low, high)))
    points.extend(('original_room_grid', np.array([x, y]))
                  for x in np.arange(low[0], high[0] + 1e-6, .2)
                  for y in np.arange(low[1], high[1] + 1e-6, .2))
    boxes = [_array(item['bbox']) for item in layout['objects'].values()]
    valid = []
    for label, xy in points:
        row, column = (int(v / .01 + trav.shape[0] / 2.) for v in (xy[1], xy[0]))
        if min(row, column) < radius or row + radius >= trav.shape[0] or column + radius >= trav.shape[1]:
            continue
        area = (slice(row - radius, row + radius + 1), slice(column - radius, column + radius + 1))
        if np.all(trav[area][disk] > 0) and np.all(rooms[area][disk] == room_id) and all(
                np.linalg.norm(xy - np.clip(xy, box[0, :2], box[1, :2])) >= clearance for box in boxes):
            valid.append((label, xy))
    if not valid:
        return None
    label, xy = min(valid, key=lambda row: (math.dist(row[1], camera['position'][:2]),
                                           float(row[1][0]), float(row[1][1]), row[0]))
    look = _vector(camera['look_target'], 3)
    return {'position': [float(xy[0]), float(xy[1]), float(camera['position'][2] - 1.25)],
            'yaw_deg': math.degrees(math.atan2(look[1] - xy[1], look[0] - xy[0])),
            'source_observer_or_rule': label, 'original_map_sha256': checksums,
            'room_id': room_id, 'body_map_radius_m': .4, 'generated_object_clearance_m': clearance,
            'map_selection_provenance': 'common.build_env_config loads floors/walls for hidden_in_box; '
                'installed TraversableMap._load_map uses floor_trav_no_obj_0.png when original furniture is absent.',
            'floor_height_provenance': 'batch_counting_merge._sample_primary_view_pose: floor_top + AGENT_CAMERA_HEIGHT_M (1.25 m)',
            'native_clearance_reach_and_opening_qualified': False}


def configure_scene(config, parameters):
    """Restore original fixed container roots before construction, never mid-run."""
    if config['scene']['scene_model'] != parameters['scene_model']:
        raise ValueError('Original enclosure restoration belongs to a different scene.')
    if not config.get('robots') or config['robots'][0].get('type') != 'R1':
        raise ValueError('The current original common builder must supply the installed R1.')
    if set(config['scene'].get('load_object_categories') or []) != {'floors', 'walls'}:
        raise ValueError('The no-furniture start proposal requires the audited floors/walls-only source builder.')
    evidence = []
    for binding in parameters['containers']:
        obj = next((item for item in config.get('objects', []) if item.get('name') == binding['name']), None)
        if obj is None or any(obj.get(field) != binding[field] for field in ('category', 'model')):
            raise ValueError('Container construction restoration must match the original asset.')
        previous = obj.get('fixed_base')
        obj['fixed_base'] = True
        evidence.append({'name': obj['name'], 'previous_fixed_base': previous, 'restored_fixed_base': True})
    inspection_contents = []
    for binding in parameters.get('inspection_contents', []):
        obj = next((item for item in config.get('objects', []) if item.get('name') == binding['name']), None)
        if obj is None or any(obj.get(field) != binding[field] for field in ('category', 'model')):
            raise ValueError('Inspection contents must preserve the original source asset.')
        obj.update(fixed_base=True, visual_only=False)
        inspection_contents.append({'name': obj['name'], 'fixed_base': True, 'visual_only': False})
    return {'source_provenance': parameters['provenance'], 'containers': evidence,
            'inspection_contents': inspection_contents,
            'builder_sha256': parameters['builder_sha256'], 'generator_sha256': parameters['generator_sha256']}


def initialize_scene(session, parameters):
    """Undo only the audited source reveal-link hiding during initialization."""
    from omnigibson.object_states import Open
    common = session.task_module.common
    evidence = []
    if session.scene_name != parameters['scene_model']:
        raise ValueError('Enclosure initialization belongs to a different scene.')
    for binding in parameters['containers']:
        obj = session.env.scene.object_registry('name', binding['name'])
        if obj is None or any(getattr(obj, field) != binding[field] for field in ('category', 'model')):
            raise ValueError('Reveal restoration must match the original native container.')
        links = common.find_container_reveal_links(obj)
        if not links:
            raise ValueError('Original source-hidden reveal links are unresolved.')
        restored = []
        for link in links:
            before = bool(link.visible)
            link.visible = True
            restored.append({'name': link.name, 'prim_path': link.prim_path,
                             'visible_before': before, 'visible_after': bool(link.visible)})
        evidence.append({'name': obj.name, 'links': restored,
                         'measured_closed': Open in obj.states and not bool(obj.states[Open].get_value()),
                         'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only)})
    session._structural_enclosure_initialization = evidence
    return {'source_provenance': parameters['provenance'], 'containers': evidence,
            'original_empty_variant_preserved': True, 'opening_tested': False}


def _snapshot(obj):
    position, quaternion = obj.get_position_orientation()
    low, high = [_array(value) for value in obj.aabb]
    return {'position': _array(position).tolist(), 'quaternion_xyzw': _array(quaternion).tolist(),
            'bbox': [low.tolist(), high.tolist()], 'visual_only': bool(obj.visual_only),
            'fixed_base': bool(obj.fixed_base)}


def _door_snapshot(obj, hand_paths):
    """Read Open's actual annotated joints and child-link contacts, not source flags."""
    from omnigibson.object_states import Open
    from omnigibson.utils.constants import JointType
    state = obj.states[Open]
    both_sides, joints, directions = state.relevant_joints_info
    rows = []
    for joint, direction in zip(joints, directions):
        if joint.joint_type == JointType.JOINT_REVOLUTE:
            kind, minimum, tolerance = 'revolute', MINIMUM_ANGLE_RAD, CLOSED_ANGLE_RAD
        elif joint.joint_type == JointType.JOINT_PRISMATIC:
            kind, minimum, tolerance = 'prismatic', MINIMUM_SLIDE_M, CLOSED_SLIDE_M
        else:
            raise ValueError('Native Open references an unsupported joint type.')
        link = next(link for link in obj.links.values() if link.prim_path == joint.body1)
        position, velocity = joint.get_state()[:2]
        if not math.isfinite(float(joint.lower_limit)) or not math.isfinite(float(joint.upper_limit)):
            raise ValueError('Native door limits must be measured and finite.')
        contacts = [{'body0': str(c.body0), 'body1': str(c.body1)} for c in link.contact_list()]
        handles = []
        for mesh in link.collision_meshes.values():
            name = str(mesh.prim.GetName())
            if 'handle' in name.lower() or 'handle' in str(link.prim_path).lower():
                handles.append({'mesh': str(mesh.prim.GetPath()),
                    'bbox': [_array(value).tolist() for value in mesh.aabb]})
        rows.append({'name': joint.name, 'position': float(_array(position).reshape(-1)[0]),
                     'velocity': float(_array(velocity).reshape(-1)[0]), 'kind': kind,
                     'lower_limit': float(joint.lower_limit), 'upper_limit': float(joint.upper_limit),
                     'open_direction': int(direction), 'minimum_travel': minimum,
                     'closed_tolerance': tolerance, 'driven': bool(joint.driven),
                     'door_link': link.prim_path, 'visible': bool(link.visible),
                     'collision_mesh_count': len(link.collision_meshes),
                     'contact_reporting': bool(link.contact_reporting_enabled), 'contacts': contacts,
                     'door_link_bbox': [_array(value).tolist() for value in link.aabb],
                     'handle_geometry': handles,
                     'hand_contact': any(c[field] in hand_paths for c in contacts for field in ('body0', 'body1'))})
    return {**_snapshot(obj), 'open': bool(state.get_value()), 'both_sides': bool(both_sides),
            'root_collision_mesh_count': len(obj.root_link.collision_meshes), 'joints': rows}


def _configuration(session, config):
    if config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or config.get('operation') not in OPERATIONS:
        raise ValueError('Use the owned enclosure operation adapter.')
    if (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name):
        raise ValueError('Enclosure evaluation belongs to a different scene/room.')
    p = config['parameters']
    keys = p['container_keys'] + p['content_keys'] + ['floor']
    if not p['container_keys'] or len(set(keys)) != len(keys) or set(config['binding_keys']) != set(keys):
        raise ValueError('Original enclosure bindings must be unique and complete.')
    if set(config['bindings']) != set(keys) or len(set(config['bindings'].values())) != len(keys):
        raise ValueError('Original native identities must be unique and complete.')
    if set(p['source_objects']) != set(keys) - {'floor'} or set(p['relations']) != set(p['content_keys']):
        raise ValueError('Original layout/containment evidence is incomplete.')
    if p.get('placement_groups'):
        members = [key for group in p['placement_groups'] for key in group]
        if (len(members) != len(set(members)) or set(members) != set(p['source_objects']) or
                any(not group or group[0] not in p['container_keys'] or any(
                    p['relations'].get(key) != group[0] for key in group[1:]) for group in p['placement_groups'])):
            raise ValueError('Each independent enclosure group must contain its original container and contents once.')
    return p


def _measure(session, controller, config, objects):
    from omnigibson.object_states import Inside
    robot, p = session.robot, config['parameters']
    arm = controller.working_arm
    hand_paths = {robot.eef_links[arm].prim_path, *(link.prim_path for link in robot.finger_links[arm])}
    doors = {key: _door_snapshot(objects[key], hand_paths) for key in p['container_keys']}
    contents = {key: {**_snapshot(objects[key]),
                     'inside_original_container': bool(objects[key].states[Inside].get_value(objects[p['relations'][key]]))}
                for key in p['content_keys']}
    position = _array(robot.get_position_orientation()[0])
    low, high = [_array(value) for value in robot.links[robot.base_footprint_link_name].aabb]
    regions = [session.env.scene.seg_map.get_room_instance_by_point(np.array([x, y]))
               for x in (low[0], (low[0] + high[0]) / 2, high[0])
               for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    floor_paths = {objects['floor'].prim_path}
    contacts = []
    for contact in robot.contact_list():
        if legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        for body in (str(contact.body0), str(contact.body1)):
            if not (body == robot.prim_path or body.startswith(robot.prim_path + '/')):
                contacts.append(body)
    on_floor = any(any(body.startswith(path + '/') for path in floor_paths) for body in contacts)
    obstacles = [body for body in contacts if not any(body.startswith(path + '/') for path in floor_paths)]
    return doors, contents, controller.debug_gripper_state(), {'position': position.tolist(),
        'in_original_room': all(region == session.room_name for region in regions),
        'supported_without_obstacle_contact': bool(on_floor and not obstacles), 'external_contacts': contacts}


def _interaction_heights(config, objects, measured):
    """Measure actual handles and contents; door-link centers remain only proxies."""
    floor_z = float(_array(objects['floor'].aabb[1])[2])

    def screen(bounds):
        low, high = (_vector(value, 3) for value in bounds)
        if np.any(high <= low):
            raise ValueError('Actual interaction geometry must have positive extents.')
        bottom, centre = float(low[2] - floor_z), float((low[2] + high[2]) / 2 - floor_z)
        return {'bottom_above_floor_m': bottom, 'centre_above_floor_m': centre,
            'within_height_window': .35 <= bottom and centre <= 1.10, 'reach_or_grasp_certified': False}

    doors, contents = measured[:2]
    containers = {}
    for key, row in doors.items():
        handles = [{**handle, **screen(handle['bbox'])}
                   for joint in row['joints'] for handle in joint.get('handle_geometry', [])]
        proxies = [screen(joint['door_link_bbox']) for joint in row['joints'] if joint.get('door_link_bbox')]
        containers[key] = {'actual_handles': handles, 'door_link_proxies': proxies,
            'handle_geometry_resolved': bool(handles),
            'known_handle_height_valid': not handles or any(handle['within_height_window'] for handle in handles)}
    return {'floor_top_m': floor_z, 'containers': containers,
            'contained_targets': {key: screen(row['bbox']) for key, row in contents.items()},
            'warning_codes': ['native_handle_reach_unverified'] + (
                ['handle_geometry_unresolved'] if any(not row['handle_geometry_resolved']
                    for row in containers.values()) else [])}


def _initial_checks(session, controller, config, objects, measured):
    from wia.tasks.interaction_setup import reference_position, touching_task_support, on_top_task_support
    doors, contents, gripper, robot = measured
    p = config['parameters']
    deltas = {key: reference_position(session, objects[key].name, row['position']) - _array(row['position'])
              for key, row in p['source_objects'].items()}
    groups = p.get('placement_groups') or [list(p['source_objects'])]
    translated_groups = all(all(np.linalg.norm(deltas[key] - deltas[group[0]]) <= 1e-6
        for key in group) for group in groups)
    legacy_vertical = bool(p.get('placement_groups') or (deltas and
        np.max(np.abs(next(iter(deltas.values()))[:2])) <= 1e-6))
    return {'original_floor_collidable': not objects['floor'].visual_only,
        'within_group_source_translations': bool(translated_groups and legacy_vertical),
        'container_upward_task_support': all(touching_task_support(session, objects[key], objects['floor']) and
            on_top_task_support(session, objects[key], objects['floor']) for key in p['container_keys']),
        'physical_inspection_contents': not p.get('placement_groups') or all(
            not objects[key].visual_only for key in p['content_keys']),
        'original_container_roots_fixed_collidable': all(row['fixed_base'] and not row['visual_only'] and
            row['root_collision_mesh_count'] > 0 for row in doors.values()),
        'original_doors_closed_visible_passive_measurable': all(not row['open'] and not row['both_sides'] and
            bool(row['joints']) and all(j['visible'] and not j['driven'] and j['collision_mesh_count'] > 0 and
            j['contact_reporting'] and abs(j['velocity']) <= .03 and
            (j['upper_limit'] - j['position'] if j['open_direction'] > 0 else
             j['position'] - j['lower_limit']) >= j['minimum_travel'] for j in row['joints'])
            for row in doors.values()),
        'source_layout_preserved': all(_near(row, {**p['source_objects'][key],
            'position': reference_position(session, objects[key].name, p['source_objects'][key]['position'])}) and
            np.max(np.abs((_array(row['bbox'])[1] - _array(row['bbox'])[0]) -
                          (_array(p['source_objects'][key]['bbox'])[1] - _array(p['source_objects'][key]['bbox'])[0]))) <= .05
            for key, row in {**doors, **contents}.items()),
        'original_contents_inside': all(row['inside_original_container'] for row in contents.values()),
        'initial_gripper_empty': gripper.get('object_name') is None and gripper.get('constraint_active') is False,
        'robot_in_original_room': robot['in_original_room'],
        'robot_supported_without_obstacle_contact': robot['supported_without_obstacle_contact']}


def qualify_operation(session, executor, config, bound):
    if config['operation'] == 'open_and_discover_contents':
        try:
            controller = getattr(executor, 'controller', executor)
            measured = _measure(session, controller, config, bound)
            checks = _initial_checks(session, controller, config, bound, measured)
            return {'accepted': all(checks.values()), 'checks': checks,
                'initial_native_inside': {key: row['inside_original_container'] for key, row in measured[1].items()},
                'interior_geometry_verified': False, 'door_opening_verified': False,
                'executed_operation_verified': False}
        except (ValueError, KeyError, AttributeError, RuntimeError) as error:
            return {'accepted': False, 'reason': f'native_enclosure_measurement_unavailable:{error}'}

    """Read-only precondition checks; no stepping, actions, pose or state setters."""
    try:
        _configuration(session, config)
        measured = _measure(session, executor, config, bound)
        checks = _initial_checks(session, executor, config, bound, measured)
        heights = _interaction_heights(config, bound, measured)
        known_handles = [row['known_handle_height_valid'] for row in heights['containers'].values()]
        checks['known_handle_height_window'] = (all(known_handles) if config['operation'] == OPERATIONS[0]
                                               else any(known_handles))
        checks['contained_target_height_window'] = all(row['within_height_window']
            for row in heights['contained_targets'].values())
        return {'accepted': bool(all(checks.values())), 'checks': checks,
                'reason': None if all(checks.values()) else 'native_enclosure_precondition_failed',
                'native_doors': measured[0], 'native_contents': measured[1], 'native_robot': measured[3],
                'native_interaction_heights': heights, 'warning_codes': heights['warning_codes'],
                'reach_handle_grasp_and_complete_door_cycle_qualified': False}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_enclosure_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}',
                'reach_handle_grasp_and_complete_door_cycle_qualified': False}


def _joint_cycle_progress():
    return {'previous': 0., 'phase': 'idle', 'hand_open_travel': 0.,
            'hand_close_travel': 0., 'opened': False, 'completed': False,
            'excursion_id': 0, 'qualified_excursion_id': None,
            'completed_excursion_id': None, 'opening_start': None,
            'qualified_opening_excursion': None, 'closing_start': None,
            'completed_cycle_evidence': None}


def _advance_joint_cycle(progress, joint, initial, *, enclosure_open, count_step):
    """Credit one opening stroke and its matching contact-driven closed return.

    Partial credit belongs to the current direction of a single excursion. A
    reversal before qualification discards opening credit; reopening discards
    the matching partial close. An actually completed cycle is retained apart
    from partial progress and cannot be manufactured by later oscillations.
    """
    excursion = (joint['position'] - initial['position']) * joint['open_direction']
    delta = excursion - progress['previous']
    hand_motion = bool(count_step and joint['hand_contact'])
    returned = abs(joint['position'] - initial['position']) <= joint['closed_tolerance']
    if delta > 0.:
        if progress['opened']:
            # The door has not returned closed, but this close was abandoned.
            progress.update(phase='opened', hand_close_travel=0., closing_start=None)
        else:
            if progress['phase'] != 'opening':
                progress.update(hand_open_travel=0., opening_start=progress['previous'])
                progress['excursion_id'] += 1
            progress['phase'] = 'opening'
            if hand_motion:
                progress['hand_open_travel'] += delta
            if (hand_motion and enclosure_open and excursion >= joint['minimum_travel']
                    and progress['hand_open_travel'] >= joint['minimum_travel']):
                progress.update(opened=True, phase='opened', hand_close_travel=0.,
                    qualified_excursion_id=progress['excursion_id'],
                    qualified_opening_excursion=excursion, closing_start=None)
    elif delta < 0.:
        if progress['opened']:
            if progress['phase'] != 'closing':
                progress.update(phase='closing', hand_close_travel=0.,
                                closing_start=progress['previous'])
            if hand_motion:
                progress['hand_close_travel'] += -delta
        else:
            # A partial opening reversed before an actual qualifying opening.
            progress.update(phase='idle', hand_open_travel=0., opening_start=None)
    progress['previous'] = excursion
    if returned and delta < 0.:
        if (progress['opened'] and progress['phase'] == 'closing' and hand_motion
                and progress['hand_close_travel'] >= joint['minimum_travel']):
            progress.update(completed=True, completed_excursion_id=progress['qualified_excursion_id'],
                completed_cycle_evidence={
                    'excursion_id': progress['qualified_excursion_id'],
                    'opening_start': progress['opening_start'],
                    'qualified_opening_excursion': progress['qualified_opening_excursion'],
                    'hand_open_travel': progress['hand_open_travel'],
                    'closing_start': progress['closing_start'],
                    'hand_close_travel': progress['hand_close_travel'],
                    'returned_excursion': excursion})
        # A return completes or abandons this excursion, never a later one.
        progress.update(phase='idle', opened=False, hand_open_travel=0., hand_close_travel=0.,
                        qualified_excursion_id=None, opening_start=None,
                        qualified_opening_excursion=None, closing_start=None)
    return returned


class EnclosureEvaluator:
    """Require measured hand-driven joint travel, opening, return and stability."""
    def __init__(self, session, controller, config, output):
        self.parameters = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('An original native enclosure binding is missing.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial, self.episode = None, 0

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = _measure(self.session, self.controller, self.config, self.objects)
        self.initial_qualification = qualify_operation(self.session, self.controller, self.config, self.objects)
        self.initial_valid = bool(self.initial_qualification['accepted'] and all(_initial_checks(
            self.session, self.controller, self.config, self.objects, self.initial).values()))
        self.progress = {key: {j['name']: _joint_cycle_progress() for j in row['joints']}
                         for key, row in self.initial[0].items()}
        self.layout_preserved, self.stable_steps = True, 0
        self.cycle_stability = dict.fromkeys(self.initial[0], 0)
        self.record(observation, action=None)

    def _update(self, count_step=False):
        if self.initial is None:
            return None
        doors, contents, gripper, robot = _measure(self.session, self.controller, self.config, self.objects)
        initial_doors, initial_contents = self.initial[:2]
        self.layout_preserved = self.layout_preserved and all(_near(row, initial_doors[key]) for key, row in doors.items()) and all(
            _near(row, initial_contents[key]) and row['inside_original_container'] for key, row in contents.items())
        cycles = []
        closed = True
        for key, row in doors.items():
            joint_closed, cycle = True, False
            for joint, initial in zip(row['joints'], initial_doors[key]['joints']):
                progress = self.progress[key][joint['name']]
                returned = _advance_joint_cycle(progress, joint, initial,
                    enclosure_open=row['open'], count_step=count_step)
                joint_closed = joint_closed and returned and abs(joint['velocity']) <= .03
                cycle = cycle or (progress['completed'] and returned)
            closed = closed and not row['open'] and joint_closed
            if cycle and not row['open'] and joint_closed:
                cycles.append(key)
        empty = gripper.get('object_name') is None and gripper.get('constraint_active') is False
        sufficient = len(cycles) == len(doors) if self.config['operation'] == 'inspect_all_enclosures' else bool(cycles)
        native_valid = all(row['fixed_base'] and not row['visual_only'] and all(j['visible'] and
            not j['driven'] and j['collision_mesh_count'] > 0 and j['contact_reporting'] for j in row['joints']) for row in doors.values())
        terminal = bool(self.initial_valid and self.layout_preserved and native_valid and sufficient and closed and
                        empty and robot['in_original_room'] and robot['supported_without_obstacle_contact'])
        if count_step:
            self.stable_steps = self.stable_steps + 1 if terminal else 0
        elif not terminal:
            self.stable_steps = 0
        value = {'native_doors': doors, 'native_contents': contents, 'gripper': gripper, 'native_robot': robot,
            'joint_progress': self.progress, 'completed_container_cycles': cycles,
            'criteria': {'initial_native_preconditions': self.initial_valid, 'layout_preserved': self.layout_preserved,
                'original_door_physics_and_visibility': native_valid, 'required_hand_driven_cycles': bool(sufficient),
                'all_original_doors_closed_stationary': bool(closed), 'gripper_empty': empty,
                'robot_in_original_room': robot['in_original_room'],
                'robot_supported_without_obstacle_contact': robot['supported_without_obstacle_contact']},
            'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS,
            'task_success': bool(terminal and self.stable_steps >= STABILITY_STEPS)}
        if self.config['operation'] == 'inspect_any_enclosure':
            items = [('open_close_any_enclosure', value['task_success'], 'terminal_condition',
                {'completed_container_cycles': cycles, 'stable_steps': self.stable_steps})]
        else:
            items = []
            for key in sorted(doors):
                if count_step:
                    self.cycle_stability[key] = self.cycle_stability[key] + 1 if key in cycles else 0
                elif key not in cycles:
                    self.cycle_stability[key] = 0
                items.append((f'{key}:open_close', self.cycle_stability[key] >= STABILITY_STEPS,
                    'terminal_condition', {'joint_progress': self.progress[key],
                    'closed_stable_steps': self.cycle_stability[key]}))
        return _completion_result(value, items)

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        value = self._update()
        if value is None:
            return None
        value.update(schema='private_structural_enclosure_v1', episode=self.episode,
                     snapshot_id=observation.snapshot_id, action_step=observation.action_step,
                     action=action, configuration=self.config, initial_native_doors=self.initial[0],
                     initial_native_contents=self.initial[1])
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config['operation'] == 'open_and_discover_contents':
        return OpenDiscoverEvaluator(session, controller, config, output)

    return EnclosureEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Enumerative Perception', 'Structural Enclosure')
    version = 'structural_enclosure_v2_task_edit'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category:
            return {'eligible': False, 'reason': 'source_category_mismatch'}
        try:
            layout = _layout(source)
            return {'eligible': True, 'operation_eligible': True,
                    'container_count': len(layout['container_keys']), 'content_count': len(layout['content_keys'])}
        except (ValueError, KeyError, TypeError) as error:
            # Keep source-only diagnostic overlays instead of dropping incomplete sources.
            return {'eligible': True, 'operation_eligible': False,
                    'reason': 'not_convertible_under_L1_L2', 'diagnostic': str(error)}

    def convert(self, source, context):
        if context.config.get('legacy_enclosure_cycle'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import TASK_REVISION
        result = self.convert_legacy(source, context)
        task = result.get('task_definition')
        if task is None:
            return result
        evaluation = task['private']['evaluation']
        parameters = evaluation['parameters']
        count = len(parameters['content_keys'])
        category = next(task['private']['bindings'][key]['category'] for key in parameters['content_keys']).replace('_', ' ')
        evaluation['operation'] = 'open_and_discover_contents'
        parameters.update(task_revision=TASK_REVISION, requires_native_visual_discovery=True,
            completion_item_ids=['open:' + key for key in parameters['container_keys']]
                + ['discovery:' + key for key in parameters['content_keys']],
            mechanism_contract={'preserved_relations': ['original container/content identities and scales',
                'original enclosure membership and room reference'],
                'required_native_evidence': ['initial native Inside plus separate interior geometry',
                    'Agent-accessible physical door opening', 'synchronized visible content discovery',
                    'enclosure roots and contents preserved'], 'native_operation_verified': False})
        task['public'] = {'objective': f'Open each original microwave enclosure through the robot controls and find all {count} distinct {category} objects inside them.',
            'success_criteria': 'Each original enclosure has a measured hand-driven opening, and every distinct content target '
                + 'appears in a delivered mounted-camera observation. Opening and content discovery are separate requirements.',
            'environment': 'Original enclosed objects, container membership and dimensions are retained.',
            'constraints': ['Leave the enclosure roots and contents in place.',
                'Use legal base/arm/gripper controls. Direct joint setters and automatic opening cannot establish success.',
                'Closing the doors again is not required.']}
        result['interaction_policy'].update(relocation_policy='preserve', allow_fallback_table=False)
        result['interaction_policy'].pop('placement_groups', None)
        result['interaction_policy']['warning_codes'] = ['native_handle_articulation_and_interior_pending']
        result.update(operation_recipe=evaluation['operation'], goal_predicate=evaluation,
            change_summary=['Preserve original enclosure membership, source scales and geometry.',
                'Score each legal opening separately from each visible content discovery; remove the closing obligation.'],
            assumptions=[], limitations=['Native Inside can use geometric approximations; independent interior verification remains a release gate.',
                'Contact-driven joint motion and visible contents require native traces through public controls.'])
        return result

    def convert_legacy(self, source, context):
        inspection = self.inspect(source)
        if not inspection.get('operation_eligible'):
            return {'scene_definition': original_scene(source, context), 'task_definition': None,
                    'status': 'conversion_failed', 'failure_stage': 'source_layout',
                    'failure_reason': 'not_convertible_under_L1_L2', 'diagnostic_reason': inspection.get('diagnostic'),
                    'warning_codes': ['unresolved_original_enclosure_layout', 'source_only_diagnostic'],
                    'limitations': ['No faithful physical goal can be bound without original resolved assets/poses.',
                                    'Source render.success is not physical evidence. Source-only capture remains diagnostic.'],
                    'next_step': 'Recover the original resolved enclosure layout from source artifacts; do not invent assets or placements.'}
        layout = _layout(source)
        operation = context.config.get('enclosure_operation', OPERATIONS[0])
        if operation not in OPERATIONS:
            raise ValueError('Unsupported enclosure operation recipe.')
        for obj in layout['objects'].values():
            if not (_asset_root(context) / 'objects' / obj['category'] / obj['model']).is_dir():
                return {'scene_definition': original_scene(source, context, bindings=layout['bindings']),
                    'task_definition': None, 'status': 'conversion_failed', 'failure_stage': 'asset_availability',
                    'failure_reason': 'original_asset_unavailable', 'warning_codes': ['source_only_diagnostic'],
                    'next_step': 'Restore the installed original asset; no replacement or new asset is authorized.'}
        start = _robot_start(layout, context)
        assumptions = [{'doubt': 'Native handle reach, contact reporting, articulated collision stability and opening feasibility have not been measured.',
            'chosen_assumption': 'Restore original container anchoring and visible closed doors, enable door collisions and retain a pending operation for native qualification.',
            'alternatives': ['Use the single-enclosure operation recipe.', 'Apply an evidence-based common container/content L2 relocation only after native diagnosis.',
                             'Retain a not_convertible_under_L1_L2 diagnostic when native evidence rules out every faithful operation.'],
            'scope': source.relative_path, 'risk': 'Floor-level door handles may be unreachable with R1, driven or missing contacts; CPU candidates do not prove success.'},
            {'doubt': 'The active source builder omits the generator fixed container roots and hides reveal links; restoring visible doors changes the adapted initial visibility.',
             'chosen_assumption': 'Restore fixed_base=True only before construction and the exact source-selected reveal links only at initialization; retain the empty original variant and private QA.',
             'alternatives': ['Keep the empty original variant for source-only diagnostic review.', 'Request native evidence if the source link heuristic selects an unrelated mesh.'],
             'scope': 'Original microwave containers; contents, count, dimensions, each container/content relative pose, reference camera and answers are unchanged.',
             'risk': 'Original hidden-door screenshots cannot prove visibility equivalence, articulation correctness or real opening in the adapted variant.'},
            {'doubt': 'A private 2D map screen and source camera floor-height formula do not prove three-dimensional R1 clearance or sensor coverage.',
             'chosen_assumption': 'Use the original no-furniture map matching the source floors/walls-only builder, a deterministic original-room robot-only start with a 0.4 m map disk and 0.6 m generated-object clearance; require native room/support/contact checks.',
             'alternatives': ['Choose another native-qualified robot-only start.', 'Keep a source diagnostic if no evidenced stance is available.'],
             'scope': 'Robot initialization only.', 'risk': 'Map resolution, original obstacles and R1 shape may cause a proposed start to fail native qualification.'}]
        if start is None:
            return {'scene_definition': original_scene(source, context, bindings=layout['bindings']),
                    'task_definition': None, 'status': 'needs_human_decision', 'failure_stage': 'robot_start_screen',
                    'failure_reason': 'no_evidenced_r1_start_in_original_room', 'reuse_level': 'L1',
                    'warning_codes': ['assumption_review_required', 'source_only_diagnostic'], 'assumptions': assumptions,
                    'source_layout_evidence': {'container_count': len(layout['container_keys']),
                        'content_count': len(layout['content_keys']), 'models': sorted({obj['model'] for obj in layout['objects'].values()})},
                    'limitations': ['The conservative source-map screen found no robot proposal; this is not proof that every L1/L2 physical operation is impossible.'],
                    'next_step': 'Inspect source-only images and native room/body clearance before selecting a reversible robot-only stance.'}
        initialization = {'scene_model': layout['raw']['scene'], 'source_identity': source.relative_path,
            'containers': [layout['bindings'][key] for key in layout['container_keys']],
            'inspection_contents': [layout['bindings'][key] for key in layout['content_keys']],
            'builder_sha256': sha256(SOURCE_COMMON), 'generator_sha256': sha256(SOURCE_GENERATOR),
            'provenance': 'Original JSON question_data.render.resolved_objects.containers; '
                'batch_counting_merge._spawn_hidden_box_container_at_pose (fixed_base=True); '
                '_force_container_lid_visible and _close_container_if_possible; '
                'enumerative_perception.common.hide_hidden_in_box_microwave_doors and postprocess_env.'}
        patches = [{'operation': 'category_initialization', 'phase': 'before_construction',
                    'module': MODULE, 'parameters': initialization},
                   {'operation': 'robot_pose', 'position': start['position'], 'yaw_deg': start['yaw_deg'], 'head_pitch_deg': 30.}]
        for phase in ('before_construction', 'after_postprocessing'):
            patches.extend({'operation': 'object_physics', 'phase': phase, 'binding': key, 'visual_only': False}
                           for key in layout['container_keys'])
            patches.extend({'operation': 'object_physics', 'phase': phase, 'binding': key, 'visual_only': False}
                           for key in layout['content_keys'])
        patches.append({'operation': 'category_initialization', 'phase': 'after_postprocessing',
                        'module': MODULE, 'parameters': initialization})
        parameters = {field: layout[field] for field in ('container_keys', 'content_keys', 'relations', 'source_objects')}
        groups = [[container] + [key for key in layout['content_keys'] if layout['relations'][key] == container]
                  for container in layout['container_keys']]
        parameters['placement_groups'] = groups
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(layout['bindings']), 'parameters': parameters}
        scope = 'each original microwave enclosure' if operation == OPERATIONS[0] else 'one original microwave enclosure of your choice'
        public = {'objective': f'Inspect {scope} by physically opening its door and closing it again. '
                    'Use the enclosed objects as visual inventory; keep them in their original enclosures. '
                    'The original counting question is auxiliary: ' + layout['raw']['question'],
            'success_criteria': 'Use hand contact to open each selected door through at least 30 degrees '
                '(or 0.06 m for a sliding door), then physically close it to its starting position. '
                'Finish with all doors closed, the gripper empty and the original contents and enclosure roots undisturbed.',
            'environment': 'Original indoor enclosure inventory with initially closed microwave doors. '
                'Each portable enclosure and its contents may be initialized together on a measured accessible support.',
            'constraints': ['Use robot observations to locate the enclosures; scoring measures door operation, not content recognition.',
                'Use existing guarded robot motions and gripper controls for all operations.',
                'Keep initialized containers and contents in place, preserving original dimensions and each container/content grouping.',
                'Do not teleport objects or command enclosure joints through a scripted state setter.']}
        return {'scene_definition': original_scene(source, context, bindings=layout['bindings'], patches=patches),
            'task_definition': physical_task(source, context, public=public, evaluation=evaluation,
                bindings=layout['bindings'], patches=patches, qualification={'on_failure': 'skip', 'initial_visibility': 'none'},
                capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'gripper', 'trunk_height']),
            'status': 'generated', 'reuse_level': 'L2',
            'interaction_policy': {'mode': 'articulation', 'group_bindings': list(layout['source_objects']),
                'manipulation_bindings': layout['container_keys'], 'relocation_policy': 'independent_groups',
                'support_bindings': layout['container_keys'],
                'placement_groups': groups, 'support_search': 'measured_surfaces', 'allow_fallback_table': True,
                'source_camera_policy': 'preserve', 'source_region': layout['raw']['room'],
                'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
                'maximum_stance_distance_m': .85,
                'warning_codes': ['native_handle_height_and_articulation_pending',
                    'independent_container_groups_native_support_unverified']},
            'change_summary': ['Preserve the empty original scene, source QA, assets, sizes, within-container poses and reference camera.',
                'Restore original fixed microwave roots before construction and source-hidden reveal-link visibility after postprocessing.',
                'Enable original articulated-container collisions before construction and again after source visual-only reset.',
                'Propose an evidenced robot-only start and add a private hand-driven opening/closing operation evaluator.',
                'Place each portable container/content group independently without changing its internal geometry.',
                'Restore fixed collidable inspection contents before construction; no target retrieval or held-object shortcut is claimed.'],
            'goal_predicate': evaluation, 'operation_recipe': operation, 'available_operation_recipes': list(OPERATIONS),
            'warning_codes': ['assumption_review_required', 'l2_original_enclosure_restoration',
                              'l2_robot_start_proposal', 'physical_qualification_pending',
                              'independent_container_groups_native_support_unverified', 'fixed_inspection_contents_no_grasp_claim'],
            'assumptions': assumptions, 'source_layout_evidence': {'container_count': len(layout['container_keys']),
                'content_count': len(layout['content_keys']), 'models': sorted({obj['model'] for obj in layout['objects'].values()}),
                'robot_start': start, 'native_constructor': 'Original pure common.build_env_objects',
                'builder_sha256': initialization['builder_sha256'], 'generator_sha256': initialization['generator_sha256']},
            'limitations': ['CPU generation is not native loading, reachable handle, grasp, navigation or physical qualification.',
                'Only native joint/hand-contact/containment/pose/stability evidence can prove the operation; QA is auxiliary.',
                'Contents remain fixed and collidable for inspection; retrieval would require a separately qualified dynamic-content goal.',
                'Restored visibility follows the source heuristic and must be compared with actual articulated door child links.',
                'Human reference images, private source maps/poses/counts and scorer state are not Agent sensor inputs.'],
            'attempted_repairs': ['Declared original fixed-root, reveal-link and collision restoration; no native trial has been run.'],
            'next_step': 'Publish every generated/diagnostic record for Review; inspect restored doors and qualify native handle reach/contact without changing source geometry.'}

    def convert_all(self, sources, context):
        """Use the shared deterministic batch/diagnostic contract for every source."""
        manifest = super().convert_all(sources, context)
        manifest['failure_distribution'] = dict(Counter(
            f"{row.get('failure_stage')}:{row.get('failure_reason')}" for row in manifest['records'] if row['status'] != 'generated'))
        manifest['operation_recipe_distribution'] = dict(Counter(row.get('operation_recipe', 'diagnostic_only') for row in manifest['records']))
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Prioritize distinct failures, counts and original microwave models."""
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        selected, strata = [], set()
        rows = sorted(manifest['records'], key=lambda row: (row['status'] == 'generated', row['candidate_id']))
        previous = set(policy.get('previous_failures', ()))
        for row in rows:
            if row['candidate_id'] in previous and len(selected) < count:
                selected.append(row['candidate_id'])
        for row in rows:
            layout = row.get('source_layout_evidence', {})
            stratum = ((row.get('failure_stage'), row.get('failure_reason')) if row['status'] != 'generated'
                       else ('generated', layout.get('content_count')))
            if stratum not in strata and row['candidate_id'] not in selected and len(selected) < count:
                selected.append(row['candidate_id'])
                strata.add(stratum)
        for candidate in super().sample(manifest, policy):
            if candidate not in selected and len(selected) < count:
                selected.append(candidate)
        return selected


class OpenDiscoverEvaluator:
    """Observe legal opening strokes independently from delivered content images."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial = None

    def begin_episode(self, observation):
        self.initial = _measure(self.session, self.controller, self.config, self.objects)
        self.progress = {key: {joint['name']: _joint_cycle_progress() for joint in row['joints']}
                         for key, row in self.initial[0].items()}
        self.opened, self.discovered = set(), {}
        self.measurement_error = None
        self.preserved = True
        self.initial_containment = all(row['inside_original_container'] for row in self.initial[1].values())
        self.initial_closed = all(not row['open'] for row in self.initial[0].values())
        return self.record(observation, action=None)

    def _update(self, count_step=False):
        try:
            doors, contents, gripper, robot = _measure(self.session, self.controller, self.config, self.objects)
            self.preserved &= all(_near(row, self.initial[0][key]) for key, row in doors.items())
            self.preserved &= all(_near(row, self.initial[1][key]) and row['inside_original_container']
                                  for key, row in contents.items())
            for key, row in doors.items():
                for joint, original in zip(row['joints'], self.initial[0][key]['joints']):
                    progress = self.progress[key][joint['name']]
                    _advance_joint_cycle(progress, joint, original, enclosure_open=row['open'], count_step=count_step)
                    valid_physics = bool(not joint['driven'] and joint['visible'] and joint['collision_mesh_count']
                                         and joint['contact_reporting'] and not row['visual_only'])
                    if count_step and self.initial_closed and valid_physics and progress['opened']:
                        self.opened.add(key)
            self.measurement_error = None
            return {'doors': doors, 'contents': contents, 'robot': robot}
        except (KeyError, ValueError, AttributeError, RuntimeError) as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            return None

    def sample_step(self):
        if self.initial is not None:
            self._update(count_step=True)
            collector = getattr(self.session, 'enclosure_evidence_collector', None)
            if collector is not None:
                collector.sample_step()

    def record(self, observation, *, action=None):
        if self.initial is None:
            return None
        from .partial_occlusion import native_visual_evidence
        from wia.tasks.completion import progress
        native = self._update()
        evidence = native_visual_evidence(self.session, observation,
            {key: self.objects[key] for key in self.parameters['content_keys']})
        curated_gates = None
        if self.parameters.get('curation_revision') == TASK_EDIT2_CURATION_REVISION:
            # New additions require a native informative-content receipt; the
            # legacy pixel gate remains diagnostic and cannot release this revision.
            evidence, curated_gates = task_edit2_enclosure_evidence(
                self.session, observation, self.config, evidence)
        for key, row in evidence['objects'].items():
            if row['visible']:
                self.discovered.setdefault(key, {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})
        items = [{'id': 'open:' + key, 'kind': 'checkpoint', 'completed': key in self.opened,
            'evaluation_validity': 'valid' if key in self.opened or native is not None else 'unavailable'}
            for key in self.parameters['container_keys']]
        items += [{'id': 'discovery:' + key, 'kind': 'checkpoint', 'completed': key in self.discovered,
            'evaluation_validity': 'valid' if key in self.discovered or evidence['complete_coverage'] else 'unavailable'}
            for key in self.parameters['content_keys']]
        result = {'schema': 'private_enclosure_open_discovery_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'completion_items': items,
            'task_success': bool(self.initial_containment and self.initial_closed and self.preserved and native is not None
                and all(item['completed'] and item['evaluation_validity'] == 'valid' for item in items)),
            'initial_native_containment': self.initial_containment, 'initial_doors_closed': self.initial_closed,
            'layout_preserved': bool(self.preserved),
            'interior_geometry_verified': False, 'joint_progress': self.progress, 'visual_evidence': evidence,
            'native': native, 'measurement_error': self.measurement_error}
        if not self.initial_containment or not self.initial_closed:
            result['evaluation_validity'] = 'unavailable'
        if curated_gates is not None:
            result['curated_native_gates'] = curated_gates
            result['interior_geometry_verified'] = curated_gates.get('independent_interior_geometry') is True
            qualified = all(curated_gates.get(gate) is True for gate in TASK_EDIT2_ENCLOSURE_GATES)
            result['task_success'] &= qualified
            if not qualified:
                result['evaluation_validity'] = 'unavailable'
        result.update(progress(result))
        write_json(self.output / 'state.json', result)
        return result


# Explicit TaskEdit2 additions; historical converters and the current pool stay intact.
TASK_EDIT2_CURATION_REVISION = 'task_edit2_enclosure_additions_20261018_v1'
TASK_EDIT2_ENCLOSURE_GATES = ('actual_containment', 'independent_interior_geometry',
    'public_control_contact_opening', 'post_open_informative_visibility',
    'access_to_every_content', 'stable_robot_start')
TASK_EDIT2_ADDITIONAL_SOURCES = (
    'Enumerative Perception/Structural Enclosure/Beechwood_1_int/playroom_0/q_000.json',
    'Enumerative Perception/Structural Enclosure/Ihlen_0_int/living_room_0/q_000.json',
    'Enumerative Perception/Structural Enclosure/house_double_floor_lower/kitchen_0/q_000.json',
)


def curate_task_edit2_additions(context):
    """Return exactly three fresh candidate definitions without native approval.

    The source geometries, approach proposals, asset mixes and occupied-container
    subsets differ. Source identity and raw content provenance remain private;
    no metadata claim establishes actual native containment or opening.
    """
    from wia.authoring.conversion import SourceInstance
    from wia.authoring.io import digest
    results = []
    for identity in TASK_EDIT2_ADDITIONAL_SOURCES:
        source = SourceInstance(ESI_ROOT / 'dataset/json_clean' / identity,
                                identity, 'Enumerative Perception', 'Structural Enclosure')
        result = CategoryConverter().convert(source, context)
        layout = _layout(source)
        provenance = {'source_identity': identity, 'source_sha256': sha256(source.path),
            'source_objects': layout['objects'], 'source_content_membership': layout['relations'],
            'source_camera': layout['camera'], 'source_question': source.raw['question'],
            'source_answer': source.raw['answer']}
        key = 'enclosure_addition_' + digest({'revision': TASK_EDIT2_CURATION_REVISION,
            'provenance': provenance, 'seed': context.seed})[:24]
        task, scene = result.get('task_definition'), result.get('scene_definition')
        if task:
            task['id'] = 'task_' + key
            parameters = task['private']['evaluation']['parameters']
            parameters['curation_revision'] = TASK_EDIT2_CURATION_REVISION
            parameters['curation_provenance'] = provenance
        if scene:
            scene['id'] = 'scene_' + key
        result.update(candidate_id=key, source_identity=identity,
            curation_revision=TASK_EDIT2_CURATION_REVISION, curation_provenance=provenance,
            addition_to_existing_pool=True, final_selection='provisional', execution_status='unavailable',
            native_gates={gate: 'pending' for gate in TASK_EDIT2_ENCLOSURE_GATES})
        results.append(result)
    return results


def task_edit2_enclosure_evidence(session, observation, config, legacy_evidence):
    """Gate only fresh curated additions on measured informative native receipts."""
    provider = getattr(session, 'enumerative_enclosure_evidence', None)
    try:
        native = provider(observation, config) if callable(provider) else {}
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as error:
        native = {'measurement_error': f'{type(error).__name__}: {error}'}
    rows = {}
    delivered = set(observation.rgb_images())
    for key in config['parameters']['content_keys']:
        row = native.get('objects', {}).get(key, {})
        rows[key] = {**row, 'visible': bool(row.get('informative') is True
            and row.get('identity_synchronized') is True
            and row.get('native_rgb_matches_delivered') is True
            and row.get('snapshot_id') == observation.snapshot_id
            and row.get('camera') in delivered
            and row.get('native_name') == config['bindings'][key]
            and row.get('informative_feature_evidence'))}
    gates = native.get('gates', {})
    return {'objects': rows, 'complete_coverage': native.get('available') is True,
        'legacy_pixel_diagnostics': legacy_evidence, 'native_receipts': native}, gates

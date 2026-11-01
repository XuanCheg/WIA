"""Measure ordered native room pickups and physical right-hand tray contents.

Run CPU checks: python -B -m pytest -q tests/test_wia_room_collection.py
Build explicit definitions with scripts/build_wia_room_collection.py. Scene
setters run only in initialization; episode evaluation only reads native state.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from wia.authoring.converters import hidden_object_collection as collection

MODULE = 'wia.authoring.converters.wia_room_collection'
REVISION = 'wia_room_collection_v1'
ORDERED_OPERATION = 'visit_rooms_and_collect_on_right_hand_tray'
SEARCH_OPERATION = 'search_and_collect_on_right_hand_tray'


def validate_layout(layout, operation):
    """Require named, authored inventory; never infer a replacement object."""
    if operation not in (ORDERED_OPERATION, SEARCH_OPERATION):
        raise ValueError('Unsupported room collection operation.')
    if layout.get('source_objects_policy') not in (None, 'replace_authored_source_objects'):
        raise ValueError('Unsupported explicit source object policy.')
    native, added = layout.get('native_inventory', []), layout.get('added_inventory', [])
    if not isinstance(native, list) or not isinstance(added, list):
        raise ValueError('Collection inventories must be explicit lists.')
    if added and operation == ORDERED_OPERATION and layout.get('allow_authored_navigation_inventory') is not True:
        raise ValueError('Ordered collection requires native inventory unless explicitly authorized.')
    rows = native + added
    names = [row['name'] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError('Collection requires distinct explicitly selected objects.')
    targets = []
    for row in rows:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ('name', 'category', 'model', 'room', 'public_name')):
            raise ValueError('Inventory requires exact native identity, room and public name.')
        if row.get('role') not in ('target', 'distractor'):
            raise ValueError('Inventory role must be target or distractor.')
        if row['role'] == 'target':
            if not isinstance(row.get('support'), str) or not row['support'].strip():
                raise ValueError('Each selected target requires an explicit support name.')
            targets.append(row)
        for key in ('support_surface_mode', 'support_alignment_mode'):
            if key in row and row[key] not in ('aabb_top', 'local_raycast'):
                raise ValueError(key + ' must be aabb_top or local_raycast.')
        if 'scale' in row and 'bounding_box' in row:
            raise ValueError('Specify scale or bounding_box, never both.')
        for key in ('scale', 'bounding_box'):
            if key in row:
                values = np.asarray(row[key], dtype=float)
                if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(key + ' must contain three positive finite dimensions.')
        if row in added and not {'position', 'orientation'}.issubset(row):
            raise ValueError('Added inventory requires explicitly authored poses.')
        if 'pickup_xy_policy' in row:
            tolerance = row.get('maximum_pickup_xy_drift_m')
            if (row['pickup_xy_policy'] != 'preserve_authored_v1' or row not in added
                    or 'pickup_edge_margin_m' in row or 'pickup_approach_xy_m' in row
                    or isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                    or not np.isfinite(tolerance) or not 0 < tolerance <= .015):
                raise ValueError('Preserved pickup XY requires added inventory, an exclusive authored pose policy and drift tolerance at most 0.015 m.')
            position = np.asarray(row['position'], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('Preserved pickup XY requires a finite authored position.')
    if not targets:
        raise ValueError('At least one selected target is required.')
    route = layout.get('route', [])
    if operation == ORDERED_OPERATION:
        if (not isinstance(route, list) or not route or len(set(route)) != len(route)
                or any(not isinstance(room, str) or not room for room in route)):
            raise ValueError('Ordered collection requires an explicit distinct room route.')
        if set(row['room'] for row in targets) != set(route):
            raise ValueError('Each declared route room needs a selected suitable object, and all targets need a route room.')
    elif route:
        raise ValueError('Search collection has no ordered room contract.')
    if layout['plate']['name'] in names:
        raise ValueError('Tray and inventory identities must be distinct.')
    if any(row['name'] == 'collection_initial_support' for row in rows):
        raise ValueError('Inventory must not reuse the initialization support identity.')
    return rows


def resolve_native_inventory(layout, snapshot):
    """Check selected identities against a supplied native scene snapshot."""
    if isinstance(snapshot, dict):
        snapshot = snapshot.get('objects', snapshot.get('objects_info', snapshot))
        # Native save_scene JSON separates constructors from runtime states.
        if isinstance(snapshot, dict) and isinstance(snapshot.get('init_info'), dict):
            snapshot = snapshot['init_info']
    if isinstance(snapshot, dict):
        rows = []
        for name, value in snapshot.items():
            if not isinstance(value, dict):
                continue
            item = dict(value.get('init_info', {}).get('args', value.get('args', value)))
            item.setdefault('name', name)
            rows.append(item)
    else:
        rows = snapshot
    if not isinstance(rows, list):
        raise ValueError('Native snapshot must contain named scene objects.')
    resolved = []
    for selected in layout.get('native_inventory', []):
        matches = [row for row in rows if row.get('name') == selected['name']]
        if len(matches) != 1 or any(matches[0].get(key) != selected[key] for key in ('category', 'model')):
            raise ValueError('Selected native object is missing or has a different identity: ' + selected['name'])
        actual = matches[0]
        rooms = actual.get('rooms', actual.get('in_rooms', []))
        rooms = [rooms] if isinstance(rooms, str) else rooms
        if not selected.get('relocation') and rooms and selected['room'] not in rooms:
            raise ValueError('Selected object is not native to its designated room: ' + selected['name'])
        if actual.get('fixed_base') is True or actual.get('visual_only') is True:
            raise ValueError('Selected native object must be movable and collidable: ' + selected['name'])
        resolved.append(deepcopy(actual))
    return resolved


def configure_scene(config, parameters):
    """Retain native scene objects and add only explicitly declared apparatus."""
    layout = parameters['layout']
    validate_layout(layout, parameters['operation'])
    if parameters['revision'] != REVISION or config['scene']['scene_model'] != layout['scene_model']:
        raise ValueError('Room collection cannot replace the declared native scene.')
    allowed = {'load_room_instances', 'load_room_types', 'load_object_categories', 'not_load_object_categories'}
    if set(layout.get('scene_config', {})) - allowed:
        raise ValueError('Only existing scene loading filters may be configured.')
    config['scene'].update(deepcopy(layout.get('scene_config', {})))
    replaced_source_objects = []
    if layout.get('source_objects_policy') == 'replace_authored_source_objects':
        replaced_source_objects = [row.get('name') for row in config.get('objects', [])]
        config['objects'] = []
    objects = config.setdefault('objects', [])
    existing_names = {row.get('name') for row in objects}
    authored = []
    for row in layout.get('added_inventory', []):
        authored.append(_added_object_spec(row))
    authored.extend(deepcopy(layout.get('support_objects', [])))
    plate = layout['plate']
    authored.append(dict(type='DatasetObject', name=plate['name'], category=plate['category'],
                         model=plate['model'], position=[100., 100., 1.], fixed_base=False, visual_only=False))
    authored.append(dict(type='PrimitiveObject', name='collection_initial_support', primitive_type='Cube',
                         scale=[.25, .36, .04], position=[100., 100., .8], fixed_base=True, visual_only=False))
    if len({row['name'] for row in authored}) != len(authored) or existing_names.intersection(row['name'] for row in authored):
        raise ValueError('Added collection apparatus must not overwrite existing objects.')
    objects.extend(authored)
    import yaml
    from wia.runtime.sim.esi_omnigibson import _import_pipeline
    pipeline = _import_pipeline(ESI_ROOT)
    robot = yaml.safe_load((Path(pipeline.og.example_config_path) / 'r1_primitives.yaml').read_text())['robots'][0]
    robot.update(deepcopy(layout['robot_start']), name='robot', grasping_mode='assisted', disable_grasp_handling=False)
    robot['reset_joint_pos'][6:10] = [.6981317007977318, -1.3962634015954636, -.6981317007977318, 0.]
    config['robots'] = [robot]
    return {'revision': REVISION, 'native_objects_retained': True,
            'native_inventory_spawned': False, 'added_inventory': [row['name'] for row in layout.get('added_inventory', [])],
            'source_objects_policy': layout.get('source_objects_policy', 'retain'),
            'replaced_authored_source_objects': replaced_source_objects,
            'native_layout_qualified': False}


def _added_object_spec(row):
    return dict(type='DatasetObject', fixed_base=False, visual_only=False,
                **{key: deepcopy(row[key]) for key in
                   ('name', 'category', 'model', 'position', 'orientation', 'scale', 'bounding_box') if key in row})


def postprocess_joint_world(session, parameters):
    if session.scene_name != parameters['layout']['scene_model']:
        raise ValueError('Collection world differs from its explicit scene.')
    return {'joint_world_postprocess': True, 'revision': REVISION, 'native_objects_retained': True,
            'source_mechanism_reused': False, 'physical_success_verified': False}


def _object(session, row):
    obj = session.env.scene.object_registry('name', row['name'])
    if (obj is None or any(getattr(obj, key, None) != row[key] for key in ('name', 'category', 'model'))
            or obj.fixed_base or obj.visual_only):
        raise ValueError('Exact movable collidable inventory unavailable: ' + row['name'])
    return obj


def _region(session, position):
    return session.env.scene.seg_map.get_room_instance_by_point(collection.array(position)[:2])


def _support_bounds(obj):
    bounds = np.asarray([collection.array(value) for value in obj.aabb])
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError('Collection support measurement requires finite ordered native AABBs: ' + obj.name)
    return bounds[0], bounds[1]


def _support_surface(obj, support, mode, stage):
    low, high = _support_bounds(support)
    if mode == 'aabb_top':
        return {'surface_z_m': float(high[2])}
    from wia.authoring.converters.active_layout import _support_ray_surface
    xy = np.mean(_support_bounds(obj), axis=0)[:2].tolist()
    return _support_ray_surface(obj, support, [*xy, float(high[2]) + .10],
                                [*xy, float(low[2]) - .10], stage)


def _support_evidence(obj, support, spec, touching):
    """Reduce native settled support measurements with the existing policy."""
    low, high = _support_bounds(obj)
    slo, shi = _support_bounds(support)
    mode = spec.get('support_surface_mode', 'aabb_top')
    surface = _support_surface(obj, support, mode, 'final_receipt')
    velocity = collection.array(obj.get_linear_velocity())
    if velocity.shape != (3,) or not np.isfinite(velocity).all() or not np.isfinite(surface['surface_z_m']):
        raise ValueError('Collection support measurement requires finite native velocity and surface height: ' + obj.name)
    receipt = {'support': support.name, 'touching': bool(touching),
               'within_footprint': bool(np.all(low[:2] >= slo[:2] - .003)
                                        and np.all(high[:2] <= shi[:2] + .003)),
               'speed_m_s': float(np.linalg.norm(velocity)),
               'bottom_gap_m': float(low[2] - surface['surface_z_m']),
               'aabb_m': [low.tolist(), high.tolist()], 'support_surface_mode': mode,
               'support_surface': surface, 'alignment_permitted': bool(spec.get('relocation') or spec.get('_added'))}
    if 'support_alignment_mode' in spec:
        receipt['support_alignment_mode'] = spec['support_alignment_mode']
    receipt['accepted'] = bool(receipt['touching'] and receipt['within_footprint']
                               and abs(receipt['bottom_gap_m']) < .015 and receipt['speed_m_s'] < .02)
    return receipt


def _robot_initialization(session, parameters):
    """Read floor, upright and collision evidence without advancing the world."""
    from wia.tasks.interaction_setup import (
        array as native_array, _body_under, _contact_parts, _objects, _upright_check)
    from wia.authoring.converters.wia_background import GROUND_CATEGORIES
    measured = {'floor_supported': False, 'upright': False, 'collision_free': False,
                'measurement_available': False, 'floor_z_m': None, 'floor_prim_paths': [],
                'floor_contact_positions_m': [], 'unexpected_contact_bodies': [], 'unexpected_contacts': [],
                'allowed_right_hand_plate_contacts': [], 'right_hand_plate_contact_paths': [],
                'right_hand_holds_declared_plate': False, 'evidence_scope': 'initialization_only'}
    try:
        robot, scene = session.robot, session.env.scene
        floor_z = float(scene.trav_map.floor_heights[0])
        if not np.isfinite(floor_z):
            raise ValueError('Native traversal floor height must be finite.')
        floors = [obj for obj in _objects(scene) if obj.category in GROUND_CATEGORIES
                  and not getattr(obj, 'visual_only', False)]
        floor_paths = [str(obj.prim_path) for obj in floors]
        if not floor_paths or any(not path.startswith('/') for path in floor_paths):
            raise ValueError('Real native floor object paths are unavailable.')
        measured.update(floor_z_m=floor_z, floor_prim_paths=sorted(set(floor_paths)))
        quaternion = native_array(robot.get_position_orientation()[1])
        if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or not np.any(quaternion):
            raise ValueError('Native robot upright quaternion must be finite and nonzero.')
        upright = _upright_check(quaternion)
        measured.update(upright)
        if (not upright['upright_measurement_available']
                or not np.isfinite(upright['upright_vertical_axis_dot'])):
            raise ValueError('Native upright measurement is unavailable.')
        plate_spec = parameters.get('plate', parameters['layout']['plate'])
        plate = scene.object_registry('name', plate_spec['name'])
        if plate is None or any(getattr(plate, key, None) != plate_spec[key] for key in ('name', 'category', 'model')):
            raise ValueError('Exact declared native tray is unavailable for robot contact classification.')
        held = (robot._ag_obj_in_hand.get('right') is plate and int(robot.is_grasping('right')) == 1)
        right_links = [robot.eef_links['right'], *robot.finger_links['right']]
        right_paths = [str(link.prim_path) for link in right_links]
        robot_path, plate_path = str(robot.prim_path), str(plate.prim_path)
        if (not robot_path.startswith('/') or not plate_path.startswith('/') or not right_paths
                or any(not path or not _body_under(path, robot_path) or path == robot_path for path in right_paths)):
            raise ValueError('Native robot, tray or right hand link paths are unavailable.')
        measured.update(right_hand_holds_declared_plate=held,
                        right_hand_plate_contact_paths=sorted(set(right_paths)))
        unexpected = set()
        for contact in robot.contact_list():
            bodies = _contact_parts(contact)
            robot_bodies = [body for body in bodies if _body_under(body, robot_path)]
            external = [body for body in bodies if not _body_under(body, robot_path)]
            if not external:
                continue
            if len(robot_bodies) != 1 or len(external) != 1 or not all(bodies):
                raise ValueError('Native robot contact must identify one robot body and one external body.')
            body, other = robot_bodies[0], external[0]
            if any(_body_under(other, root) for root in floor_paths):
                position = native_array(getattr(contact, 'position', None))
                normal = native_array(getattr(contact, 'normal', None))
                if (position.shape != (3,) or normal.shape != (3,)
                        or not np.isfinite([position, normal]).all()):
                    raise ValueError('Native floor contact position and normal must be finite three-vectors.')
                if abs(float(normal[2])) >= .5 and abs(float(position[2]) - floor_z) <= .005:
                    measured['floor_contact_positions_m'].append(position.tolist())
            elif (held and _body_under(other, plate_path)
                  and any(_body_under(body, path) for path in right_paths)):
                measured['allowed_right_hand_plate_contacts'].append({'robot_body': body, 'plate_body': other})
            else:
                unexpected.add(other)
                measured['unexpected_contacts'].append({'robot_body': body, 'external_body': other})
        measured.update(floor_supported=bool(measured['floor_contact_positions_m']),
                        unexpected_contact_bodies=sorted(unexpected), collision_free=not unexpected,
                        measurement_available=True)
    except (ValueError, TypeError, RuntimeError, AttributeError, KeyError, IndexError, OverflowError) as error:
        measured.update(floor_supported=False, upright=False, collision_free=False,
                        reason=('Robot initialization measurement unavailable: ' + type(error).__name__ + ': ' + str(error))[:640])
    return measured


def _right_gripper_failure_diagnostics(session):
    """Read assisted-grasp inputs without attempting or establishing a grasp."""
    from wia.tasks.interaction_setup import array as native_array
    result = {}

    def read(key, getter):
        try:
            result[key] = getter()
        except Exception as error:
            result[key] = {'error': (type(error).__name__ + ': ' + str(error))[:180]}

    def contacts():
        candidates, links = session.robot._find_gripper_contacts(arm='right')
        paths = sorted(str(path) for path in candidates)
        return {'candidate_prim_paths': [path[:112] for path in paths[:4]],
                'candidate_count': len(paths), 'contact_link_names': {
                    str(path)[:112]: [str(link).rsplit('/', 1)[-1][:48] for link in sorted(links[path], key=str)[:4]]
                    for path in sorted(links, key=str)[:4]}}

    def raycasts():
        paths = sorted(str(path) for path in session.robot._find_gripper_raycast_collisions(arm='right'))
        return {'prim_paths': [path[:112] for path in paths[:4]], 'count': len(paths)}

    def counter(name):
        value = getattr(session.robot, name)['right']
        if value is None:
            return None
        values = native_array(value)
        if values.shape != ():
            raise ValueError('Native grasp counter must be scalar.')
        return values.item()

    def control():
        values = native_array(session.robot.controllers['gripper_right'].control).reshape(-1)
        return {'values': values[:8].tolist(), 'count': int(values.size)}

    def pose(link):
        position, quaternion = (native_array(value) for value in link.get_position_orientation())
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError('Native link pose requires position and xyzw quaternion.')
        return {'prim_path': str(link.prim_path)[:112], 'position_m': position.tolist(),
                'quaternion_xyzw': quaternion.tolist()}

    read('gripper_contacts', contacts)
    read('gripper_raycast_collisions', raycasts)
    read('ag_grasp_counter_right', lambda: counter('_ag_grasp_counter'))
    read('ag_release_counter_right', lambda: counter('_ag_release_counter'))
    read('grasping_mode', lambda: str(session.robot.grasping_mode)[:48])
    read('gripper_right_control', control)
    read('right_eef_pose', lambda: pose(session.robot.eef_links['right']))
    try:
        fingers = session.robot.finger_links['right'][:4]
    except Exception as error:
        result['right_finger_poses'] = {'error': (type(error).__name__ + ': ' + str(error))[:180]}
    else:
        result['right_finger_poses'] = {}
        for index, link in enumerate(fingers):
            key = 'right_finger_' + str(index)
            read(key, lambda link=link: pose(link))
            result['right_finger_poses'][key] = result.pop(key)
    return result


def _compact_failure_diagnostic(value, list_limit, string_limit):
    """Keep diagnostic field names while bounding oversized native detail."""
    if isinstance(value, dict):
        return {key: _compact_failure_diagnostic(item, list_limit, string_limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_failure_diagnostic(item, list_limit, string_limit) for item in value[:list_limit]]
    return value[:string_limit] if isinstance(value, str) else value


def initialize_scene(session, parameters):
    """Align only declared native relocations, then initialize the held tray."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Collection setup is restricted to explicit initialization.')
    preview_only = parameters.get('unqualified_tray_preview') is True
    rows = validate_layout(parameters['layout'], parameters['operation'])
    relocations, alignments, edge_placements = [], [], []
    added_names = {row['name'] for row in parameters['layout'].get('added_inventory', [])}
    for row in rows:
        obj = _object(session, row)
        pose = row.get('relocation')
        if pose:
            if set(pose) != {'position', 'orientation'}:
                raise ValueError('Relocation requires only explicit position and orientation.')
            obj.set_position_orientation(position=pose['position'], orientation=pose['orientation'])
            obj.keep_still()
            relocations.append(row['name'])
    for row in rows:
        if not row.get('support'):
            continue
        obj = _object(session, row)
        support = session.env.scene.object_registry('name', row['support'])
        if support is None:
            raise ValueError('Declared collection support is missing: ' + row['support'])
        if row['name'] not in added_names and not row.get('relocation') and not getattr(session, '_release_edge_policy', False):
            continue
        if row.get('pickup_xy_policy') == 'preserve_authored_v1':
            # Concave furniture's outer AABB edge can be behind a partition.
            # Generic preload physics may already have displaced an added object.
            # Restore its declared pose before measuring the local support surface.
            previous_position, previous_orientation = (
                collection.array(value) for value in obj.get_position_orientation())
            obj.set_position_orientation(position=row['position'], orientation=row['orientation'])
            obj.keep_still()
            edge_placements.append({'object': obj.name, 'support': support.name,
                'policy': row['pickup_xy_policy'], 'authored_xy_m': row['position'][:2],
                'pre_alignment_position_m': previous_position.tolist(),
                'pre_alignment_orientation_xyzw': previous_orientation.tolist(),
                'authored_pose_restored_before_support_check': True,
                'physical_grasp_verified': False, 'route_connectivity_verified': False})
        elif row.get('pickup_edge_margin_m') is not None:
            from wia.tasks.pickup_edges import place_pickup_near_edge
            edge_placements.append(place_pickup_near_edge(session, obj, support, row['pickup_edge_margin_m'], row['room'], row.get('pickup_approach_xy_m')))
            if not hasattr(session, 'pickup_edge_placements'):
                session.pickup_edge_placements = []
            session.pickup_edge_placements.append(dict(edge_placements[-1], status='placed'))
        elif getattr(session, '_release_edge_policy', False):
            from wia.tasks.pickup_edges import apply_edge_policy
            edge_placements.extend(apply_edge_policy(session, [obj], supports={obj.name: support.name}))
        mode = row.get('support_alignment_mode', row.get('support_surface_mode', 'aabb_top'))
        surface = _support_surface(obj, support, mode, 'initial_alignment')
        position, quaternion = (collection.array(value) for value in obj.get_position_orientation())
        position[2] += surface['surface_z_m'] - float(_support_bounds(obj)[0][2]) + .003
        obj.set_position_orientation(position=position.tolist(), orientation=quaternion.tolist())
        obj.keep_still()
        alignments.append({'target': obj.name, 'support': support.name, 'mode': mode})
    from wia.runtime.sim.esi_collection_setup import initialize_collection_tray
    try:
        receipt = initialize_collection_tray(session, parameters)
    except ValueError as error:
        tray_receipt = getattr(session, 'hidden_collection_initialization', {})
        diagnostic = {key: tray_receipt.get(key) for key in (
            'accepted', 'stage', 'right_eef_residual_m', 'closure_steps', 'plate_start_position', 'pinch')}
        diagnostic['closure'] = tray_receipt.get('closure', [])[-2:]
        diagnostic['contacts'] = tray_receipt.get('contacts', [])[:12]
        diagnostic['robot_initialization'] = _robot_initialization(session, parameters)
        diagnostic['right_gripper_diagnostics'] = _right_gripper_failure_diagnostics(session)
        evidence = json.dumps(diagnostic, separators=(',', ':'), allow_nan=False)
        prefix = str(error) + '; tray_initialization_diagnostic='
        for list_limit, string_limit in ((8, 128), (4, 96), (2, 64), (1, 32)):
            if len(prefix) + len(evidence) <= 6000:
                break
            compact = _compact_failure_diagnostic(diagnostic, list_limit, string_limit)
            compact['diagnostic_compacted'] = True
            evidence = json.dumps(compact, separators=(',', ':'), allow_nan=False)
        message = prefix + evidence
        if len(message) > 6000:
            truncation = '...[truncated]'
            message = message[:6000 - len(truncation)] + truncation
        if not preview_only:
            raise ValueError(message) from error
        receipt = {**tray_receipt, 'accepted': False, 'stage': 'unqualified_tray_preview',
                   'initialization_preview_only': True, 'tray_initialization_error': message,
                   'tray_initialization_diagnostic': diagnostic}
    robot_initialization = _robot_initialization(session, parameters)
    from omnigibson.object_states import Touching
    inventory, supports = {}, {}
    for row in rows:
        obj = _object(session, row)
        position = collection.array(obj.get_position_orientation()[0])
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Collection inventory requires a finite native position: ' + row['name'])
        room = _region(session, position)
        inventory[row['name']] = {'native_room': room, 'required_room': row['room'],
                                 'room_matches': room == row['room'], 'position_m': position.tolist()}
        if row.get('pickup_xy_policy') == 'preserve_authored_v1':
            drift = float(np.linalg.norm(position[:2] - np.asarray(row['position'])[:2]))
            inventory[row['name']].update(authored_xy_drift_m=drift,
                authored_xy_preserved=drift <= row['maximum_pickup_xy_drift_m'])
        if row.get('support'):
            support = session.env.scene.object_registry('name', row['support'])
            spec = {**row, '_added': row['name'] in added_names}
            supports[row['name']] = _support_evidence(obj, support, spec, obj.states[Touching].get_value(support))
    accepted = bool(receipt['accepted'] and all(row['room_matches'] for row in inventory.values())
                    and all(row.get('authored_xy_preserved', True) for row in inventory.values())
                    and all(row['accepted'] for row in supports.values())
                    and not preview_only)
    result = {**receipt, 'accepted': accepted, 'inventory': inventory, 'explicit_relocations': relocations,
              'inventory_supports': supports, 'support_alignments': alignments,
              'pickup_edge_placements': edge_placements,
              'robot_initialization': robot_initialization,
              'native_inventory_spawned': False, 'complete_collection_verified': False,
              'native_layout_qualified': False}
    if preview_only:
        result.update(stage='unqualified_tray_preview', initialization_preview_only=True)
    if not accepted:
        failed_rooms = [name for name, row in inventory.items() if not row['room_matches']]
        failed_supports = [name + ':' + row['support'] for name, row in supports.items() if not row['accepted']]
        summary = {'parent_initialization_accepted': bool(receipt['accepted']),
                   'failed_robot_initialization_checks': [key for key in ('floor_supported', 'upright', 'collision_free')
                                                           if not robot_initialization[key]],
                   'robot_measurement_reason': robot_initialization.get('reason'),
                   'failed_inventory_rooms': [name[:80] for name in failed_rooms[:8]],
                   'failed_inventory_room_count': len(failed_rooms),
                   'failed_authored_pickup_xy': [name for name, row in inventory.items()
                                                if row.get('authored_xy_preserved') is False],
                   'failed_supports': [name[:80] for name in failed_supports[:8]],
                   'failed_support_count': len(failed_supports)}
        if preview_only:
            summary = {'initialization_preview_only': True,
                       'preview_reason': 'Unqualified scene preview only; this variant cannot qualify or score task success.',
                       **summary}
        result['reason'] = ('Collection initialization rejected: ' + json.dumps(summary, separators=(',', ':')))[:1800]
    session.wia_room_collection_initialization = result
    session.category_interaction_setup = result
    return result


def _state(session, parameters):
    state = collection._state(session, parameters)
    state['native_region'] = _region(session, session.robot.get_position_orientation()[0])
    # Requiring the chassis footprint inside one segmented room avoids doorway credit.
    robot = session.robot
    low, high = (collection.array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    footprint = [_region(session, [x, y]) for x in (low[0], (low[0] + high[0]) / 2, high[0])
                 for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    state['room_occupancy_valid'] = bool(state['native_region'] is not None
                                        and all(room == state['native_region'] for room in footprint))
    state['object_rooms'] = {row['name']: _region(session, _object(session, row).get_position_orientation()[0])
                             for row in parameters['layout'].get('native_inventory', []) + parameters['layout'].get('added_inventory', [])}
    return state


def qualify_operation(session, executor, config, bound):
    parameters = config['parameters']
    state = _state(session, parameters)
    receipt = getattr(session, 'wia_room_collection_initialization', {})
    checks = {'initialization_receipt': receipt.get('accepted') is True,
              'right_hand_holds_plate': state['right_holds_plate'], 'plate_level': state['plate_level'],
              'initial_plate_empty': not state['supported_objects'], 'left_hand_empty': state['left_empty'],
              'native_room_occupancy': state['room_occupancy_valid'],
              'left_arm_control': getattr(getattr(executor, 'controller', None), 'working_arm', None) == 'left'}
    return {'accepted': all(checks.values()), 'checks': checks, 'state': state}


class RoomCollectionEvaluator:
    """Keep room-route progress separate from valid pickups and tray contents."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.rows = validate_layout(self.parameters['layout'], config['operation'])
        self.targets = {row['name']: row for row in self.rows if row['role'] == 'target'}
        self.route = self.parameters['layout'].get('route', [])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.begin_episode(None)

    def begin_episode(self, observation):
        self.route_stage, self.acquisition_stage = 0, 0
        self.picked, self.wrong_pickups, self.previous_collected = set(), set(), set()
        self.visits, self.acquisitions, self.spills, self.faults = [], [], [], []
        self.previous_held = None
        self.clock_regression = None
        self.last_time = self.stable_since = self.state = None

    def _result(self, duration, dwell):
        result = collection.collection_result(self.state, list(self.targets), self.parameters.get('distractors', []),
                                              self.picked, stable_seconds=duration, required_dwell=dwell)
        result['checks'].update(declared_route_completed=self.route_stage == len(self.route),
                                ordered_room_acquisitions_completed=self.acquisition_stage == len(self.route),
                                native_measurements_valid=not self.faults,
                                initialization_receipt=getattr(self.session, 'wia_room_collection_initialization', {}).get('accepted') is True)
        result['task_success'] = all(result['checks'].values())
        return result

    def sample_step(self):
        state = _state(self.session, self.parameters)
        now = float(state['native_time_s'])
        if self.last_time is not None and now <= self.last_time:
            if now < self.last_time:
                if 'native_clock_regressed' not in self.faults:
                    self.faults.append('native_clock_regressed')
                self.clock_regression = {'previous_time_s': self.last_time, 'current_time_s': now}
                self.state = state
                self.stable_since = None
            return
        room, valid = state['native_region'], state['room_occupancy_valid']
        if (not self.visits or self.visits[-1]['room'] != room
                or self.visits[-1]['eligible'] != valid):
            self.visits.append({'room': room, 'eligible': valid, 'native_time_s': now})
        if valid and self.route_stage < len(self.route) and room == self.route[self.route_stage]:
            self.route_stage += 1
        held = state['left_holding']
        if held != self.previous_held and held in self.targets:
            target = self.targets[held]
            room_ok = valid and room == target['room'] and state['object_rooms'].get(held) == target['room']
            order_ok = not self.route or (self.acquisition_stage < len(self.route)
                                         and target['room'] == self.route[self.acquisition_stage])
            accepted = bool(room_ok and order_ok)
            self.acquisitions.append({'target': held, 'room': room, 'object_room': state['object_rooms'].get(held),
                                      'in_designated_room': bool(room_ok), 'correct_order': bool(order_ok),
                                      'accepted': accepted, 'native_time_s': now})
            if accepted:
                self.picked.add(held)
                if self.route and all(name in self.picked for name, row in self.targets.items() if row['room'] == target['room']):
                    self.acquisition_stage += 1
        if held in self.parameters.get('distractors', []):
            self.wrong_pickups.add(held)
        present = set(state['supported_objects']).intersection(self.targets)
        self.spills.extend({'target': name, 'native_time_s': now} for name in sorted(self.previous_collected - present))
        self.previous_collected, self.previous_held = present, held
        self.state, self.last_time = state, now
        if self._result(0., 0.)['task_success']:
            if self.stable_since is None:
                self.stable_since = now
        else:
            self.stable_since = None

    def record(self, observation, *, action=None):
        self.sample_step()
        duration = 0. if self.stable_since is None else self.last_time - self.stable_since
        result = self._result(duration, self.parameters['terminal_dwell_s'])
        initialization_accepted = result['checks']['initialization_receipt']
        validity = ('unavailable' if self.faults else
                    'complete' if initialization_accepted else 'invalid_initialization')
        result.update(schema='native_wia_room_collection_v1', snapshot_id=observation.snapshot_id,
                      action_step=observation.action_step, evaluation_validity=validity,
                      terminal_conditions_satisfied=result['task_success'], route=list(self.route),
                      route_stage=self.route_stage, acquisition_stage=self.acquisition_stage,
                      native_room_visits=self.visits, acquisitions=self.acquisitions,
                      validly_acquired_targets=sorted(self.picked), wrong_pickups=sorted(self.wrong_pickups),
                      spills=self.spills, native_state=self.state, configuration=self.config,
                      faults=list(self.faults), clock_regression=getattr(self, 'clock_regression', None),
                      completion_items=[{'id': 'collect:' + name, 'kind': 'current_physical_inventory',
                                         'completed': validity == 'complete' and name in result['collected_targets'],
                                         'evaluation_validity': 'valid' if validity == 'complete' else validity}
                                        for name in self.targets])
        (self.output / 'latest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    if config['operation'] not in (ORDERED_OPERATION, SEARCH_OPERATION) or config['parameters']['revision'] != REVISION:
        raise ValueError('Unsupported room collection contract.')
    return RoomCollectionEvaluator(session, controller, config, output)

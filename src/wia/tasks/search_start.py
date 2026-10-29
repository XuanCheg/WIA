"""Initialization-only measured R1 start for curated multi-room Search.

Run: python -B -m pytest -q tests/test_task_edit2_search_start.py
Call initialize_search_start(session, world) after owned source actors/tables are
placed and before any AgentObservation or public control. Only the robot root
pose and native neutral hold may change. Failure returns an explicit private
receipt; this helper never qualifies Search routes, cameras, semantics or access.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from copy import deepcopy
import hashlib
from itertools import product
import math
from pathlib import Path

import numpy as np

from wia.authoring.io import digest

from .interaction_setup import _navigation_cells, _place_robot, _robot_contact_check, _objects, array
from .support_sampling import same_body

POLICY = {'hold_steps': 30, 'max_translation_m': .01, 'max_rotation_deg': 1., 'upright_dot_min': .95,
    'provenance': 'task_edit2_search_native_start_pilot_20261018_v1_root_selected',
    'status': 'new_pilot_initialization_acceptance_screen_not_route_or_manipulation_calibration'}
LIMIT = 32


def _native_floor_ray(start, end):
    from .native_floor_contacts import native_vertical_ray
    return native_vertical_ray(start, end, only_closest=False)


def _pose(obj):
    position, quaternion = (array(value) for value in obj.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,) or np.linalg.norm(quaternion) < 1e-12:
        raise ValueError('Native pose unavailable.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _rotation(quaternion):
    x, y, z, w = array(quaternion) / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _angle(first, second):
    a, b = array(first), array(second)
    dot = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.rad2deg(2 * np.arccos(np.clip(dot, 0., 1.))))


def _time(session):
    try:
        value = float(session.pipeline.og.sim.current_time)
        return value if math.isfinite(value) else None
    except (AttributeError, TypeError, ValueError):
        return None


def _source(session):
    # Do not invoke ESISession.observation_source: that property creates a rig.
    state = vars(session)
    return state.get('_observation_source') or state.get('observation_source')


def _guard(session):
    source = _source(session)
    if source is not None and (any(getattr(source, name, None) is not None for name in
            ('initial_observation', 'latest_snapshot', 'boundary_observation'))
            or getattr(source, 'action_step', 0) or getattr(source, 'history', ())):
        raise ValueError('Search start initialization is forbidden after an observation or public action.')
    controllers = [vars(session).get('controller'), vars(session).get('task_controller'),
        getattr(getattr(source, 'controller_state_provider', None), '__self__', None),
        getattr(vars(session).get('task_evaluator'), 'controller', None)]
    for controller in [session, *controllers]:
        if controller is not None and (getattr(controller, '_public_action_sequence', 0)
                or (getattr(controller, '_public_action_context', {}) or {}).get('sequence', 0)):
            raise ValueError('Search start initialization is forbidden after public controls.')


def _context(session, original_world):
    _guard(session)
    task = vars(session).get('loaded_task') or getattr(vars(session).get('scene_adapter'), 'task', None)
    if task is None:
        raise ValueError('Actual loaded Search task provenance is unavailable.')
    task.verify_unchanged()
    evaluation = task.definition.private.evaluation
    if (module_identity(evaluation.module) != 'wia.authoring.converters.enumerative_perception_search'
            or digest(evaluation.parameters['world_contract']) != digest(original_world)
            or session.scene_name != original_world['scene_model']
            or session.env.scene.scene_model != original_world['scene_model']):
        raise ValueError('Stale or different loaded Search world.')
    from wia.authoring.converters.task_edit2_search_layout import resolve_runtime_world
    runtime_world, layout_receipt = resolve_runtime_world(session, original_world)
    if not isinstance(layout_receipt, dict) or layout_receipt.get('accepted') is not True:
        raise ValueError('Search start requires an accepted reset-bound native layout receipt, including unchanged layouts.')
    if runtime_world['scene_model'] != original_world['scene_model']:
        raise ValueError('Search runtime layout changed the original scene identity.')
    identity = dict(session.initial_overview_identity)
    if (identity.get('seed') != task.seed or any(not identity.get(k) for k in ('episode_id', 'reset_id'))
            or identity['reset_id'] != getattr(session, 'native_reset_id', None)):
        raise ValueError('Actual Search reset/seed identity unavailable or mismatched.')
    root = Path(task.source_path).resolve()
    for _ in Path(task.definition.source.json_path).parts:
        root = root.parent
    for source in original_world['source_provenance']:
        path = (root / source['source_identity']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != source['source_sha256']:
            raise ValueError('An exact Search source identity/hash changed.')
    return {**identity, 'original_world_digest': digest(original_world),
        'runtime_layout_digest': digest(runtime_world), 'runtime_layout_receipt_digest': digest(layout_receipt),
        'task_id': task.definition.id,
        'task_sha256': task.file_sha256, 'source_sha256': task.source_sha256,
        'source_path': str(task.source_path), 'scene': session.scene_name}, runtime_world


def _snapshot(session, actor_names):
    objects = {}
    for obj in _objects(session.env.scene):
        if obj is session.robot:
            continue
        objects[obj.name] = {**_pose(obj), 'category': obj.category, 'model': getattr(obj, 'model', None),
            'scale': array(obj.scale).tolist(), 'fixed_base': bool(obj.fixed_base),
            'visual_only': bool(obj.visual_only), 'collision_enabled': (bool(obj.collision_enabled)
                if getattr(obj, 'collision_enabled', None) is not None else None),
            'aabb_m': [array(value).tolist() for value in obj.aabb],
            'joints': {str(name): array(joint.get_state()[0]).tolist() for name, joint in obj.joints.items()},
            'protected': obj.name in actor_names or bool(obj.fixed_base)
                or obj.category in ('floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings'),
            'owned_source_actor': obj.name in actor_names}
    robot = session.robot
    return {'objects': objects, 'robot': {**_pose(robot), 'scale': array(robot.scale).tolist(),
        'fixed_base': bool(robot.fixed_base), 'visual_only': bool(robot.visual_only),
        'collision_enabled': bool(robot.collision_enabled) if getattr(robot, 'collision_enabled', None) is not None else None,
        'joint_positions': array(robot.get_joint_positions()).tolist(),
        'sensor_parent_mounts': {name: [array(value).tolist() for value in sensor.get_position_orientation(frame='parent')]
                                 for name, sensor in robot.sensors.items()}}}


def _invariants(before, after):
    failures, settling = [], {}
    if set(before['objects']) != set(after['objects']):
        failures.append('native_object_inventory_changed')
    for name in set(before['objects']) & set(after['objects']):
        first, last = before['objects'][name], after['objects'][name]
        for key in ('category', 'model', 'scale', 'fixed_base', 'visual_only', 'collision_enabled'):
            if first[key] != last[key]:
                failures.append(name + ':' + key)
        pose_changed = any(first[key] != last[key] for key in ('position_m', 'quaternion_xyzw'))
        if pose_changed and first['protected']:
            failures.append(name + ':protected_pose')
        if first['owned_source_actor'] and any(first[key] != last[key] for key in ('joints', 'aabb_m')):
            failures.append(name + ':owned_visual_geometry')
        if first['category'] in ('floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings') and first['aabb_m'] != last['aabb_m']:
            failures.append(name + ':structural_geometry')
        if pose_changed or first['joints'] != last['joints']:
            settling[name] = {'before': deepcopy(first), 'after': deepcopy(last),
                              'restored_or_hidden': False}
    for key in ('scale', 'fixed_base', 'visual_only', 'collision_enabled', 'sensor_parent_mounts'):
        if before['robot'][key] != after['robot'][key]:
            failures.append('robot:' + key)
    return {'preserved': not failures, 'violations': failures, 'native_dynamic_settling': settling,
            'background_state_restore_used': False}


def _target(session, world):
    groups, tables = world.get('search_groups', []), world.get('curated_layout', {}).get('tables', [])
    scene = session.env.scene
    if groups:
        first = groups[0]
        source = next(row for row in world['source_provenance'] if row['source_identity'] == first['source_identity'])
        room = source['source_payload']['room']
        same_room = [table for table in tables if table['room_instance'] == room]
        if same_room:
            table = same_room[0]
            obj = scene.object_registry('name', table['name'])
            if obj is None:
                raise ValueError('First source-room curated table is not loaded.')
            centre = np.mean(np.stack([array(value) for value in obj.aabb]), axis=0)
            method = 'first_source_group_first_loaded_same_room_curated_table_AABB_centre'
        else:
            actors = [scene.object_registry('name', item['native']['name']) for item in world['joint_inventory']
                      if item['source_identity'] == first['source_identity']]
            if not actors or any(obj is None for obj in actors):
                raise ValueError('First source-group native target centre unavailable.')
            centre = np.mean([np.mean(np.stack([array(value) for value in obj.aabb]), axis=0) for obj in actors], axis=0)
            method = 'first_source_group_loaded_native_actor_AABB_centroid'
        identity = first['source_identity']
    elif tables:
        table = tables[0]
        room = table['room_instance']
        obj = scene.object_registry('name', table['name'])
        if obj is None:
            raise ValueError('First curated table is not loaded.')
        centre = np.mean(np.stack([array(value) for value in obj.aabb]), axis=0)
        identity, method = None, 'first_explicit_curated_table_AABB_centre'
    else:
        raise ValueError('No explicit first source group or curated table.')
    if not isinstance(room, str) or not room or room == 'full_scene' or room not in world['required_rooms']:
        raise ValueError('Start requires an exact original source room, never full-building fallback.')
    if scene.seg_map.get_room_instance_by_point(centre[:2]) != room:
        raise ValueError('Loaded first group/table centre is outside its original source room.')
    return room, centre, {'method': method, 'source_identity': identity, 'room': room, 'target_centre_m': centre.tolist()}


def _wheel_offsets(robot):
    pose = _pose(robot)
    rotation = _rotation(pose['quaternion_xyzw'])
    origin = array(pose['position_m'])
    result = {}
    for name in getattr(robot, 'floor_touching_base_link_names', ()):
        link = robot.links[name]
        low, high = (array(value) for value in link.aabb)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
            raise ValueError('Actual native wheel link AABB unavailable: ' + name)
        corners = np.array(list(product(*zip(low, high))))
        local = (corners - origin) @ rotation
        result[name] = {'prim_path': link.prim_path, 'native_aabb_m': [low.tolist(), high.tolist()],
            'base_relative_AABB_corners_m': local.tolist()}
    if not result:
        raise ValueError('Native wheel footprint links are unavailable; robot AABB is not a substitute.')
    return result


def _wheel_floor(session, xy, target, offsets, floor_paths, nominal):
    direction = array(target)[:2] - xy
    yaw = math.atan2(float(direction[1]), float(direction[0])) if np.linalg.norm(direction) > .001 else 0.
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    samples, heights = [], []
    robot_path = session.robot.prim_path
    for name, wheel in offsets.items():
        projected = array(wheel['base_relative_AABB_corners_m'])[:, :2] @ rotation.T + xy
        low, high = projected.min(0), projected.max(0)
        points = [np.mean(projected, axis=0)] + [np.array(point) for point in product(*zip(low, high))]
        for point in points:
            row = {'wheel': name, 'wheel_link_path': wheel['prim_path'], 'xy_m': point.tolist(),
                   'ray_start_m': [*point.tolist(), nominal + .08], 'ray_end_m': [*point.tolist(), nominal - .08]}
            hits = _native_floor_ray(row['ray_start_m'], row['ray_end_m'])
            floor_hits, foreign = [], []
            for hit in hits:
                if not hit.get('hit'):
                    continue
                body = str(hit.get('rigidBody', ''))
                if same_body(robot_path, body):
                    continue
                measured = {'body': body, 'position_m': array(hit['position']).tolist(), 'normal': array(hit['normal']).tolist()}
                if any(same_body(root, body) for root in floor_paths):
                    if measured['normal'][2] >= .95:
                        floor_hits.append(measured)
                else:
                    foreign.append(measured)
            row['floor_hits'], row['foreign_obstructions'] = floor_hits, foreign
            row['source_room'] = session.env.scene.seg_map.get_room_instance_by_point(point)
            if floor_hits:
                selected = max(floor_hits, key=lambda hit: hit['position_m'][2])
                row['selected_floor'] = selected
                heights.append(selected['position_m'][2])
            row['accepted'] = bool(floor_hits and not foreign)
            samples.append(row)
    level = float(np.mean(heights)) if len(heights) == len(samples) else None
    spread = float(max(heights) - min(heights)) if heights else None
    return {'accepted': bool(samples and all(row['accepted'] for row in samples) and spread <= .005),
        'method': 'actual_native_wheel_link_AABB_offsets_projected_at_candidate_yaw',
        'measured_floor_z_m': level, 'nominal_floor_z_m': nominal,
        'nominal_floor_offset_m': level - nominal if level is not None else None,
        'sample_height_spread_m': spread, 'maximum_surface_spread_m': .005,
        'ray_proposal_half_span_m': .08, 'samples': samples}


def _neutral_action(robot):
    indices = {name: array(value).astype(int) for name, value in robot.controller_action_idx.items()}
    if not indices or not np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(robot.action_dim)):
        raise ValueError('Native R1 controller action indices do not cover the exact hold action.')
    action = np.empty(robot.action_dim, dtype=np.float32)
    control = robot.get_control_dict()
    for name, controller in robot.controllers.items():
        if (controller.motor_type != 'position' or controller.use_delta_commands
                or controller.command_input_limits is not None):
            raise ValueError('Search initialization requires the existing native position/no-op controllers.')
        action[indices[name]] = array(controller.compute_no_op_action(control))
    if not np.isfinite(action).all() or 'base' not in indices or np.any(action[indices['base']]):
        raise ValueError('Native neutral hold must have exactly zero base command.')
    return action


def initialize_search_start(session, original_world):
    """Resolve a private layout from ORIGINAL world; return initialization acceptance only.

    The caller must bind session.robot after source/table postprocessing and
    before the first observation. The immutable evaluator world is never edited.
    """
    receipt = {'schema': 'task_edit2_search_start_20261018_v1', 'accepted': False,
        'policy': deepcopy(POLICY), 'candidate_limit': LIMIT, 'attempts': [], 'started_placements': 0,
        'pending_gates': ['public_control_approach', 'loaded_routes', 'mounted_camera_audit', 'informative_view', 'semantic_access'],
        'route_or_manipulation_certificate': False, 'category_interaction_setup_modified': False,
        'source_actor_or_background_state_restore_used': False, 'joint_state_setters_used': False,
        'restoration': {'performed': False, 'reason': 'No robot placement has been started.'}}
    previous = vars(session).get('search_start_initialization')
    session.search_start_initialization = receipt
    before, provenance, neutral = None, None, None
    try:
        receipt['stage'] = 'initialization_context'
        if previous is not None:
            raise ValueError('Search start initialization has already been attempted in this session.')
        provenance, world = _context(session, original_world)
        receipt['provenance'] = provenance
        robot = session.robot
        if type(robot).__name__ != 'R1' and getattr(robot, 'model', None) != 'R1':
            raise ValueError('This initialization screen requires the actual native R1.')
        actor_names = {item['native']['name'] for item in world['joint_inventory']}
        for item in world['joint_inventory']:
            obj = session.env.scene.object_registry('name', item['native']['name'])
            if obj is None or any(getattr(obj, field, None) != item['native'][field] for field in ('category', 'model')):
                raise ValueError('An exact owned Search actor binding is unavailable.')
        room, target, target_receipt = _target(session, world)
        receipt['selection'] = target_receipt
        before = _snapshot(session, actor_names)
        receipt['before'] = deepcopy(before)
        initial_native = getattr(session, '_search_initial_native', {})
        if not actor_names.issubset(initial_native):
            raise ValueError('Owned Search source postprocessing baseline is unavailable.')
        for name, expected in initial_native.items():
            current = before['objects'].get(name)
            fields = {'position': 'position_m', 'orientation': 'quaternion_xyzw', 'scale': 'scale',
                'category': 'category', 'model': 'model', 'fixed_base': 'fixed_base', 'visual_only': 'visual_only'}
            if current is None or any(expected.get(key) != current[field] for key, field in fields.items()):
                raise ValueError('Owned source/table baseline is stale: ' + name)
        offsets = _wheel_offsets(robot)
        receipt['wheel_footprints'] = offsets
        neutral = _neutral_action(robot)
        receipt['neutral_hold_action'] = neutral.tolist()
        floors = [obj for obj in _objects(session.env.scene) if obj.category in ('floor', 'floors') and not obj.visual_only]
        if not floors:
            raise ValueError('No loaded collidable native floor bodies.')
        floor_paths = [obj.prim_path for obj in floors]
        from .native_floor_contacts import measure_native_floor_contacts
        contact_reader = vars(session).get('search_start_floor_contacts') or measure_native_floor_contacts
        receipt['stage'] = 'same_source_room_navigation_proposals'
        cells, nominal, grid = _navigation_cells(session, robot, room, set())
        receipt['navigation_grid'] = grid
        nominal = float(nominal)
        original_contact = contact_reader(session, robot, floor_paths, nominal_floor_z=nominal)
        receipt['original_contact_measurement'] = original_contact
        original_safe = (original_contact.get('measurement_available') is True
            and original_contact.get('floor_supported') is True and original_contact.get('collision_free') is True
            and _rotation(before['robot']['quaternion_xyzw'])[2, 2] >= POLICY['upright_dot_min'])
        order = sorted(range(len(cells)), key=lambda i: (float(np.linalg.norm(array(cells[i]) - target[:2])),
                                                         float(cells[i][0]), float(cells[i][1]), i))
        receipt['search_coverage'] = {'same_room_candidate_count': len(cells), 'started_candidates': 0,
            'started_fraction': 0., 'limit': LIMIT, 'other_room_fallback_used': False,
            'coverage_limitation': 'A capped unsuccessful search is not proof of room or task infeasibility.'}
        for index in order[:LIMIT]:
            if _context(session, original_world)[0] != provenance:
                raise ValueError('Reset/task/source/world identity changed during initialization.')
            xy = array(cells[index])
            attempt = {'candidate_index': index, 'proposed_xy_m': xy.tolist(), 'accepted': False}
            receipt['attempts'].append(attempt)
            receipt['search_coverage'].update(started_candidates=len(receipt['attempts']),
                started_fraction=len(receipt['attempts']) / len(cells) if len(cells) else 0.)
            rays = _wheel_floor(session, xy, target, offsets, floor_paths, nominal)
            attempt['wheel_floor_rays'] = rays
            exact_room = (session.env.scene.seg_map.get_room_instance_by_point(xy) == room
                          and all(row['source_room'] == room for row in rays['samples']))
            if not rays['accepted'] or not exact_room:
                attempt['reason'] = 'wheel_floor_rays_or_exact_source_room_rejected'
                continue
            receipt['stage'] = 'native_robot_placement_and_settle'
            receipt['started_placements'] += 1
            _place_robot(session, robot, xy, target, rays['measured_floor_z_m'])
            receipt['restoration'] = {'performed': False, 'reason': 'Initialization candidate has been placed.'}
            attempt['legacy_contact_diagnostic_at_measured_height'] = _robot_contact_check(
                robot, floor_paths, rays['measured_floor_z_m'])
            start = _pose(robot)
            begin_time = _time(session)
            samples = []
            for step in range(POLICY['hold_steps'] + 1):
                if step:
                    session.env.step({robot.name: neutral.copy()})
                if _context(session, original_world)[0] != provenance:
                    raise ValueError('Reset/task/source/world identity changed during neutral hold.')
                pose = _pose(robot)
                contacts = contact_reader(session, robot, floor_paths, nominal_floor_z=nominal)
                invariants = _invariants(before, _snapshot(session, actor_names))
                sample = {'step': step, 'native_time_s': _time(session), **pose,
                    'translation_from_hold_start_m': float(np.linalg.norm(array(pose['position_m']) - start['position_m'])),
                    'rotation_from_hold_start_deg': _angle(pose['quaternion_xyzw'], start['quaternion_xyzw']),
                    'upright_dot': float(_rotation(pose['quaternion_xyzw'])[2, 2]),
                    'floor_contact_count': len(contacts.get('floor_contact_positions_m', [])),
                    'contacts': contacts, 'invariants': invariants}
                samples.append(sample)
                if not invariants['preserved']:
                    attempt['hold_samples'] = samples
                    raise ValueError('Protected source actor/background/robot configuration changed during initialization.')
                if (contacts.get('measurement_available') is not True or contacts.get('floor_supported') is not True
                        or contacts.get('collision_free') is not True or sample['upright_dot'] < POLICY['upright_dot_min']):
                    break
            attempt['hold_samples'] = samples
            end_time = _time(session)
            attempt['stability'] = {'native_time_start_s': begin_time, 'native_time_end_s': end_time,
                'simulated_duration_s': end_time - begin_time if begin_time is not None and end_time is not None else None,
                'requested_hold_steps': POLICY['hold_steps'], 'completed_hold_steps': len(samples) - 1,
                'maximum_translation_m': max(sample['translation_from_hold_start_m'] for sample in samples),
                'maximum_rotation_deg': max(sample['rotation_from_hold_start_deg'] for sample in samples),
                'velocity_inferred': False, 'policy': deepcopy(POLICY)}
            supported = all(sample['contacts'].get('measurement_available') is True
                and sample['contacts'].get('floor_supported') is True
                and sample['contacts'].get('collision_free') is True
                and sample['upright_dot'] >= POLICY['upright_dot_min'] for sample in samples)
            attempt['accepted'] = bool(supported and len(samples) == POLICY['hold_steps'] + 1
                and attempt['stability']['maximum_translation_m'] <= POLICY['max_translation_m']
                and attempt['stability']['maximum_rotation_deg'] <= POLICY['max_rotation_deg'])
            if attempt['accepted']:
                receipt.update(accepted=True, stage='pilot_floor_supported_start_measured', reason=None,
                    after=_snapshot(session, actor_names), final_pose=_pose(robot))
                receipt['invariants'] = _invariants(before, receipt['after'])
                invalidate = getattr(session, 'invalidate_camera_cache', None)
                if callable(invalidate):
                    invalidate()
                return receipt
            attempt['reason'] = 'native_contact_upright_or_passive_stability_screen_rejected'
        receipt['reason'] = 'No accepted start in the tested same-room proposals; this does not prove infeasibility.'
    except Exception as error:
        receipt.update(accepted=False, reason=f'{type(error).__name__}: {error}')
        original_safe = False
    if before is not None:
        try:
            owned = {name for name, row in before['objects'].items() if row['owned_source_actor']}
            receipt['after'] = _snapshot(session, owned)
            receipt['invariants'] = _invariants(before, receipt['after'])
            if (receipt['started_placements'] and 'original_safe' in locals() and original_safe
                    and receipt['invariants']['preserved'] and provenance == _context(session, original_world)[0]):
                original = before['robot']
                session.robot.set_position_orientation(position=array(original['position_m']), orientation=array(original['quaternion_xyzw']))
                session.robot.keep_still()
                receipt['restoration'] = {'performed': True, 'scope': 'original_robot_root_pose_only',
                    'native_hold_action': neutral.tolist(), 'reason': 'Original native contact/upright state was safe before initialization.',
                    'post_restore_qualification': 'unavailable_no_episode_may_start_from_this_failed_receipt'}
                receipt['after'] = _snapshot(session, owned)
                receipt['invariants'] = _invariants(before, receipt['after'])
            elif receipt['started_placements']:
                receipt['restoration'] = {'performed': False,
                    'reason': 'Safe original pose/hold restoration was not established; initialization must stop.'}
        except Exception as error:
            receipt['cleanup_error'] = f'{type(error).__name__}: {error}'
            receipt['restoration'] = {'performed': False, 'reason': 'Robot-only restoration unavailable; initialization must stop.'}
    return receipt

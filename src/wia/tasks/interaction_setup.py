"""Screen and repair category interaction layouts during initialization only.

Run through scripts/run_esi_capture_worker.py after the user resumes rendering.
Pure geometry helpers can be checked with tests/test_esi_interaction_setup.py.
Height, aperture and chassis screens never certify an executed grasp or IK path.
"""
from __future__ import annotations

import copy
import math
import json

import numpy as np


REVISION = 'interaction_accessible_groups_v3'


def initialize_loaded_room_start(session):
    """Repair an out-of-loaded-world default spawn before source settling.

    A traversability proposal is accepted only after native floor/contact checks.
    This initialization repair neither changes objects nor certifies task access.
    """
    robot = session.env.robots[0]
    floors = [obj for obj in _objects(session.env.scene) if obj.category in ('floor', 'floors')]
    position, quaternion = (array(value) for value in robot.get_position_orientation())
    floor_bounds = [(array(obj.aabb[0]), array(obj.aabb[1])) for obj in floors]
    inside = any(np.all(position[:2] >= low[:2]) and np.all(position[:2] <= high[:2])
                 for low, high in floor_bounds)
    receipt = {'applicable': not inside, 'accepted': False,
        'original_position_m': position.tolist(), 'original_quaternion_xyzw': quaternion.tolist(),
        'loaded_floor_names': [obj.name for obj in floors], 'attempts': [],
        'task_access_verified': False}
    session.loaded_room_start = receipt
    if inside:
        receipt['reason'] = 'Original spawn lies over a loaded floor footprint; normal native qualification remains required.'
        return receipt
    region = session.room_name if session.room_name != 'full_scene' else None
    world, floor_z, grid = _navigation_cells(session, robot, region, set())
    receipt['spawn_grid'] = grid
    target = array(session.scene_camera_position)
    order = np.argsort(np.linalg.norm(world - target[:2], axis=1), kind='stable')
    baseline = session.pipeline.og.sim.dump_state(serialized=False)
    floors_paths = [obj.prim_path for obj in floors]
    for index in order[:32]:
        session.pipeline.og.sim.load_state(baseline, serialized=False)
        _place_robot(session, robot, world[index], target, floor_z)
        checks = _robot_contact_check(robot, floors_paths, floor_z)
        p = array(robot.get_position_orientation()[0])
        q = checks['upright_normalized_quaternion_xyzw']
        checks['loaded_floor_footprint'] = any(np.all(p[:2] >= low[:2]) and np.all(p[:2] <= high[:2])
                                             for low, high in floor_bounds)
        receipt['attempts'].append({'position_m': p.tolist(), 'quaternion_xyzw': q, 'checks': checks})
        if all(checks[key] for key in ('floor_supported', 'collision_free', 'upright', 'loaded_floor_footprint')):
            receipt.update(accepted=True, position_m=p.tolist(), quaternion_xyzw=q)
            session.invalidate_camera_cache()
            return receipt
    session.pipeline.og.sim.load_state(baseline, serialized=False)
    raise ValueError('No native stable start exists among the tested loaded-room proposals; initialization remains unavailable.')


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    raw = np.asarray(value)
    if raw.dtype.names:
        # PhysX contact/raycast vectors use structured x/y/z float records.
        # Preserve coordinate order for scalar records and batched vectors.
        if set(raw.dtype.names) != {'x', 'y', 'z'}:
            raise ValueError('Unsupported structured native interaction vector.')
        raw = np.stack([raw[axis] for axis in ('x', 'y', 'z')], axis=-1)
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Interaction geometry requires finite native measurements.')
    return result


def height_screen(bounds, floor_z, minimum=.35, maximum=1.10):
    """Screen support-side height and object centre relative to the actual floor."""
    rows = {}
    for name, (low, high) in bounds.items():
        low, high = array(low), array(high)
        if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
            raise ValueError('Native object AABBs must have positive finite extents.')
        bottom, centre = float(low[2] - floor_z), float((low[2] + high[2]) / 2 - floor_z)
        rows[name] = {'bottom_above_floor_m': bottom, 'centre_above_floor_m': centre,
            'within_height_window': minimum <= bottom and centre <= maximum,
            'height_is_grasp_certificate': False}
    return rows


def group_translation(bounds, support_bounds, xy, surface_z, *, margin=.03):
    """Fit an unchanged group on a measured support without scaling or repacking."""
    lows = np.stack([array(value[0]) for value in bounds.values()])
    highs = np.stack([array(value[1]) for value in bounds.values()])
    low, high = lows.min(axis=0), highs.max(axis=0)
    support_low, support_high = (array(value) for value in support_bounds)
    centre = (low + high) / 2
    delta = np.r_[array(xy) - centre[:2], float(surface_z) - low[2] + .005]
    if (np.any(low[:2] + delta[:2] < support_low[:2] + margin)
            or np.any(high[:2] + delta[:2] > support_high[:2] - margin)):
        raise ValueError('The unchanged task group does not fit this support.')
    return delta


def _independent_candidate_clearance(scene, bounds, policy):
    """Read previous native footprints without imposing a within-group gap."""
    minimum = policy.get('_independent_group_minimum_xy_gap_m')
    if minimum is None:
        return {'applicable': False, 'accepted': True, 'pairs': []}
    from .interaction_groups import independent_group_clearance
    groups = {policy['_independent_group_key']: bounds}
    for group_key, names in policy.get('_placed_independent_groups', {}).items():
        objects = [scene.object_registry('name', name) for name in names]
        if any(obj is None for obj in objects):
            return {'applicable': True, 'accepted': False, 'pairs': [],
                'minimum_gap_m': minimum, 'reason': 'A previously placed independent candidate is missing.',
                'group': group_key, 'objects': list(names)}
        groups[group_key] = {obj.name: obj.aabb for obj in objects}
    return independent_group_clearance(groups, minimum)


def prepare_definitions(source, scene, task, explicit=None):
    """Add the owned initialization policy to a new pending revision only."""
    if task is None or task.get('kind') != 'category_robot_task':
        return None
    parameters = task['private'].get('evaluation', {}).get('parameters', {})
    if scene is not None:
        if parameters.get('reflection_sensor_profile'):
            scene.setdefault('runtime', {})['reflection_sensor_profile'] = copy.deepcopy(parameters['reflection_sensor_profile'])
        if parameters.get('requires_native_visual_discovery'):
            scene.setdefault('runtime', {})['requires_native_visual_discovery'] = True
    if explicit is None:
        from wia.authoring.converters.interaction_profiles import SUPPORTED_CATEGORIES, policy_for
        if (source.big_task, source.small_task) not in SUPPORTED_CATEGORIES:
            return None
        explicit = policy_for(source, task)
    if explicit is None:
        return None
    explicit = copy.deepcopy(explicit)
    if explicit.get('layout_parameters', {}).get('inclined_apparatus'):
        explicit['layout_parameters']['inclined_source_parameters'] = copy.deepcopy(parameters)
    if explicit.get('layout_parameters', {}).get('deformable_cover'):
        explicit['layout_parameters']['deformable_source_parameters'] = copy.deepcopy(parameters)
    if (str(parameters.get('task_revision', '')).startswith('task_edit2_')
            and explicit.get('mode') in ('manipulation', 'articulation')
            and not parameters.get('reflection_sensor_profile')):
        from .loader import ROOT
        arm_configuration = json.loads((ROOT / 'configs/esi_arm.json').read_text())
        explicit['controller_initial_trunk_joint_positions_rad'] = arm_configuration[
            'initial_trunk_joint_positions_rad']
    if (parameters.get('task_revision') == 'task_edit2_20261018_v1'
            and parameters.get('initial_layout')):
        layout = parameters['initial_layout']
        explicit.update(relocation_policy='seeded_geometry',
            layout_parameters={'object_keys': parameters['object_keys'],
                'target_shape': parameters['target_shape'], 'seed': layout['seed'],
                'source_identity': layout['source_identity']})
    if explicit.get('mode') != 'navigation' and explicit.get('relocation_policy') != 'preserve':
        explicit.setdefault('support_search', 'measured_surfaces')
        explicit.setdefault('allow_fallback_table', True)
    from .schema import CategoryInteractionPatch
    patch = CategoryInteractionPatch.model_validate({
        **explicit, 'operation': 'prepare_category_interaction', 'phase': 'after_postprocessing'}).model_dump(mode='json')
    bindings = task['private']['bindings']
    if not set(patch['group_bindings']) <= set(bindings):
        raise ValueError('Interaction grouping must use existing private binding keys.')
    if not set(patch.get('support_preferences', {}).values()) <= set(bindings):
        raise ValueError('Preferred supports must use existing private binding keys.')
    for definition in (scene, task):
        if definition is None:
            continue
        variants = definition['scene']['variants']
        adapted = copy.deepcopy(variants.get('adapted', variants['original']))
        adapted['patches'] = [item for item in adapted['patches']
            if item['operation'] != 'prepare_category_interaction'] + [copy.deepcopy(patch)]
        variants['adapted'] = adapted
    return explicit


def _objects(scene):
    values = scene.objects
    return list(values.values()) if isinstance(values, dict) else list(values)


def _body_under(value, root):
    return str(value) == str(root) or str(value).startswith(str(root) + '/')


def _notify_interaction(session, stage, evidence):
    observer = getattr(session, '_interaction_setup_observer', None)
    if callable(observer):
        observer(stage, evidence)


def _contact_parts(contact):
    return tuple(str(getattr(contact, key, '')) for key in ('body0', 'body1'))


def support_contact_paths(session, object_name):
    evidence = getattr(session, 'category_interaction_setup', {})
    if evidence.get('accepted') and object_name in evidence.get('support_prim_paths', {}):
        value = evidence['support_prim_paths'][object_name]
        return (value,) if isinstance(value, str) else tuple(value)
    if (evidence.get('accepted') and evidence.get('relocated')
            and 'support_prim_path' in evidence and object_name in evidence.get('object_names', ())):
        return (evidence['support_prim_path'],)
    return ()


def reference_position(session, object_name, original_xyz, *, root=False):
    evidence = getattr(session, 'category_interaction_setup', {})
    position = array(original_xyz)
    adapted = evidence.get('adapted_geometry', {}).get(object_name, {}).get('position_m')
    if root and evidence.get('accepted') and adapted is not None:
        return array(adapted)
    if evidence.get('accepted') and object_name in evidence.get('object_translations_m', {}):
        position = position + array(evidence['object_translations_m'][object_name])
    elif support_contact_paths(session, object_name):
        delta = evidence.get('object_translations_m', {}).get(object_name)
        position = position + array(delta if delta is not None else evidence['group_translation_m'])
    return position


def verified_repaired_start(session, source_region, position, quaternion, *, position_tolerance_m,
                            heading_tolerance_deg):
    """Require the actual initialized pose and measured support in its source room."""
    evidence = getattr(session, 'category_interaction_setup', {}) or {}
    spawn = evidence.get('robot_spawn', {})
    recorded_position = spawn.get('robot_position_m')
    recorded_quaternion = spawn.get('robot_quaternion_xyzw')
    checks = {'setup_accepted': evidence.get('accepted') is True,
        'floor_supported': spawn.get('checks', {}).get('floor_supported') is True,
        'collision_free': spawn.get('checks', {}).get('collision_free') is True,
        'upright': (spawn.get('checks', {}).get('upright') is True
                    and _upright_check(quaternion)['upright']),
        'requested_source_region_matches': evidence.get('spawn_grid', {}).get('requested_source_region') == source_region,
        'recorded_position_matches': False, 'recorded_orientation_matches': False}
    if (isinstance(recorded_position, (list, tuple)) and len(recorded_position) == 3
            and isinstance(recorded_quaternion, (list, tuple)) and len(recorded_quaternion) == 4):
        recorded_position, recorded_quaternion = array(recorded_position), array(recorded_quaternion)
        position, quaternion = array(position), array(quaternion)
        norm, recorded_norm = np.linalg.norm(quaternion), np.linalg.norm(recorded_quaternion)
        checks['recorded_position_matches'] = bool(np.linalg.norm(position - recorded_position) <= position_tolerance_m)
        if norm > 0 and recorded_norm > 0:
            cosine = abs(float(np.dot(quaternion / norm, recorded_quaternion / recorded_norm)))
            error = math.degrees(2 * math.acos(float(np.clip(cosine, 0., 1.))))
            checks['recorded_orientation_matches'] = error <= heading_tolerance_deg
    return {'accepted': all(checks.values()), 'checks': checks}


def _support_object(session, obj, original_floor):
    paths = support_contact_paths(session, obj.name)
    if not paths:
        return original_floor
    return next((item for item in _objects(session.env.scene) if item.prim_path in paths), None)


def touching_task_support(session, obj, original_floor):
    """Read current support contact; historical initial contact is insufficient."""
    from omnigibson.object_states import Touching
    paths = support_contact_paths(session, obj.name)
    if not paths:
        return bool(obj.states[Touching].get_value(original_floor))
    support = _support_object(session, obj, original_floor)
    if support is None:
        return False
    setup = session.category_interaction_setup
    z = setup.get('support_surfaces_z_m', {}).get(obj.name, setup.get('support_surface_z_m'))
    if z is None:
        return False
    if getattr(obj, 'fixed_base', False):
        return _fixed_surface_contact(obj, support, z)
    if not obj.states[Touching].get_value(support):
        return False
    for contact in obj.contact_list():
        if not any(_body_under(body, path) for body in _contact_parts(contact) for path in paths):
            continue
        normal, position = getattr(contact, 'normal', None), getattr(contact, 'position', None)
        if normal is not None and position is not None:
            if abs(float(array(normal)[2])) >= .5 and abs(float(array(position)[2]) - z) <= .015:
                return True
    return False


def on_top_task_support(session, obj, original_floor):
    from omnigibson.object_states import OnTop
    support = _support_object(session, obj, original_floor)
    if support is not None and getattr(obj, 'fixed_base', False) and support_contact_paths(session, obj.name):
        setup = session.category_interaction_setup
        z = setup.get('support_surfaces_z_m', {}).get(obj.name, setup.get('support_surface_z_m'))
        return z is not None and _fixed_surface_contact(obj, support, z)
    return bool(support is not None and obj.states[OnTop].get_value(support))


def _fixed_surface_contact(obj, support, surface_z):
    """Measure current support geometry when PhysX omits fixed-fixed contacts."""
    from omnigibson.utils.sampling_utils import raytest
    from .support_sampling import same_body
    low, high = (array(value) for value in obj.aabb)
    if obj.visual_only or support.visual_only or abs(float(low[2]) - surface_z) > .02:
        return False
    for x in (low[0] + .001, high[0] - .001):
        for y in (low[1] + .001, high[1] - .001):
            hits = raytest([x, y, surface_z + .025], [x, y, surface_z - .025], only_closest=False)
            if not any(hit['hit'] and same_body(support.prim_path, hit['rigidBody'])
                       and float(array(hit['normal'])[2]) >= .95
                       and abs(float(array(hit['position'])[2]) - surface_z) <= .005 for hit in hits):
                return False
    return True


def _initial_support_contact(session, group, bounds):
    """Reject floating layouts even when their object heights already look valid."""
    from omnigibson.utils.sampling_utils import raytest
    roots = [obj.prim_path for obj in group.values()]
    bottom = min(box[0][2] for box in bounds.values()) if bounds else 0.
    evidence = {}
    for obj in group.values():
        low, high = bounds[obj.name]
        if low[2] > bottom + .03:
            continue
        if not obj.fixed_base:
            contact = any(abs(float(array(hit.normal)[2])) >= .5
                and abs(float(array(hit.position)[2]) - low[2]) <= .03
                and any(not any(_body_under(body, root) for root in roots) for body in _contact_parts(hit))
                for hit in obj.contact_list())
        else:
            contact = True
            for x in (low[0] + .001, high[0] - .001):
                for y in (low[1] + .001, high[1] - .001):
                    hits = raytest([x, y, low[2] + .025], [x, y, low[2] - .025], only_closest=False)
                    contact &= any(hit['hit'] and not any(_body_under(hit['rigidBody'], root) for root in roots)
                                   and float(array(hit['normal'])[2]) >= .95
                                   and abs(float(array(hit['position'])[2]) - low[2]) <= .015 for hit in hits)
        evidence[obj.name] = bool(contact)
    return evidence


def _navigation_cells(session, robot, region, excluded):
    import torch
    from .support_sampling import chassis_footprint, footprint_clearance
    scene = session.env.scene
    trav, segmentation = scene.trav_map, scene.seg_map
    free = array(trav.floor_map[0]) == 255
    if free.ndim != 2:
        raise ValueError('A measured floor traversability map is required.')
    floor_z = float(trav.floor_heights[0])
    _, extent, radius = chassis_footprint(robot)
    height = float(array(robot.aabb[1])[2] - array(robot.aabb[0])[2])
    for obj in _objects(scene):
        if (obj is robot or obj.name in excluded or obj.visual_only
                or obj.category in ('floors', 'floor', 'walls', 'wall', 'ceilings', 'ceiling')):
            continue
        low, high = (array(value) for value in obj.aabb)
        if high[2] <= floor_z + .04 or low[2] >= floor_z + height:
            continue
        corners = array(trav.world_to_map(np.stack([low[:2], high[:2]]))).astype(int)
        begin = np.maximum(corners.min(axis=0), 0)
        end = np.minimum(corners.max(axis=0) + 1, free.shape)
        free[begin[0]:end[0], begin[1]:end[1]] = False
    cells = np.argwhere(footprint_clearance(free, trav.map_resolution, radius))
    if not len(cells):
        raise ValueError('No conservative collision-free chassis cell exists.')
    world = array(trav.map_to_world(torch.as_tensor(cells)))
    if region and region != 'full_scene':
        room_id = segmentation.room_ins_name_to_ins_id.get(region)
        if room_id is None:
            raise ValueError('Requested source region is absent from the native room map.')
        indices = (world[:, ::-1] / segmentation.map_resolution + segmentation.map_size / 2).astype(int)
        valid = np.all((indices >= 0) & (indices < segmentation.map_size), axis=1)
        mask = np.zeros(len(world), dtype=bool)
        mask[valid] = array(segmentation.room_ins_map)[tuple(indices[valid].T)] == room_id
        world = world[mask]
    if not len(world):
        raise ValueError('No conservative stance exists in the requested original room.')
    return world, floor_z, {'chassis_extent_m': extent.tolist(), 'chassis_radius_m': radius,
        'floor_index': 0, 'floor_z_m': floor_z, 'requested_source_region': region,
        'candidate_cell_count': len(world), 'route_and_arm_clearance_verified': False}


def _upright_check(quaternion):
    """Measure local +Z against world +Z from a normalized native xyzw quaternion."""
    measured = {'upright': False, 'upright_measurement_available': False,
        'upright_vertical_axis_dot': None, 'upright_minimum_vertical_axis_dot': .95,
        'upright_quaternion_xyzw': None, 'upright_normalized_quaternion_xyzw': None}
    try:
        q = array(quaternion)
        if q.shape != (4,):
            raise ValueError('Native robot orientation must contain four finite xyzw values.')
        scale = float(np.max(np.abs(q)))
        if scale == 0:
            raise ValueError('Native robot orientation quaternion has zero norm.')
        unit = q / scale
        unit /= np.linalg.norm(unit)
        vertical = float(np.clip(1 - 2 * (unit[0] ** 2 + unit[1] ** 2), -1., 1.))
        measured.update(upright=vertical >= .95, upright_measurement_available=True,
            upright_vertical_axis_dot=vertical, upright_quaternion_xyzw=q.tolist(),
            upright_normalized_quaternion_xyzw=unit.tolist())
    except (ValueError, TypeError, RuntimeError, AttributeError) as error:
        measured['upright_measurement_error'] = f'{type(error).__name__}: {error}'
    return measured


def _robot_contact_check(robot, floor_paths, floor_z):
    floor_contacts, other = [], []
    for contact in robot.contact_list():
        bodies = _contact_parts(contact)
        external = [body for body in bodies if not _body_under(body, robot.prim_path)]
        if not external:
            continue
        if not all(any(_body_under(body, root) for root in floor_paths) for body in external):
            other.extend(external)
            continue
        normal, position = getattr(contact, 'normal', None), getattr(contact, 'position', None)
        if normal is not None and position is not None:
            if abs(float(array(normal)[2])) >= .5 and abs(float(array(position)[2]) - floor_z) <= .005:
                floor_contacts.append(array(position).tolist())
    try:
        upright = _upright_check(robot.get_position_orientation()[1])
    except (ValueError, TypeError, RuntimeError, AttributeError, IndexError) as error:
        upright = _upright_check(None)
        upright['upright_measurement_error'] = f'{type(error).__name__}: {error}'
    return {'floor_supported': bool(floor_contacts), 'floor_contact_positions_m': floor_contacts,
        'unexpected_contact_bodies': sorted(set(other)), 'collision_free': not other, **upright}


def _retained_source_stance_proposal(session, robot, proposed_world, floor_paths, floor_z, target, policy):
    """Retain one measured independent-group source cell without certifying access."""
    if (policy.get('support_search') != 'measured_surfaces' or not policy.get('_independent_group_key')
            or policy.get('relocation_policy') in ('shared_tabletop', 'seeded_geometry', 'comparison_pairs', 'relation_group')
            or policy.get('layout_parameters', {}).get('inclined_apparatus')):
        return None, None
    checks = _robot_contact_check(robot, floor_paths, floor_z)
    if not all(checks[key] for key in ('upright', 'floor_supported', 'collision_free')):
        return None, None
    try:
        position, quaternion = (array(value) for value in robot.get_position_orientation())
        cells, target = array(proposed_world), array(target)
        resolution = float(session.env.scene.trav_map.map_resolution)
        if (position.shape != (3,) or quaternion.shape != (4,) or target.shape != (3,)
                or cells.ndim != 2 or cells.shape[1] != 2 or not len(cells)
                or not np.isfinite(resolution) or resolution <= 0):
            return None, None
        distances = np.linalg.norm(cells - position[:2], axis=1)
        index = int(np.argmin(distances))
        if distances[index] > resolution:
            return None, None
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return None, None
    target_distance = float(np.linalg.norm(position[:2] - target[:2]))
    return index, {'source_position_m': position.tolist(), 'source_quaternion_xyzw': quaternion.tolist(),
        'original_native_checks': checks, 'proposed_world_index': index,
        'grid_position_xy_m': cells[index].tolist(), 'grid_match_distance_m': float(distances[index]),
        'measured_map_resolution_tolerance_m': resolution, 'target_position_m': target.tolist(),
        'target_distance_m': target_distance,
        'initial_approach_required': target_distance > policy['maximum_stance_distance_m'],
        'proposal_only': True, 'native_post_placement_acceptance_required': True,
        'arm_reach_verified': False, 'route_verified': False}


def _place_robot(session, robot, xy, target, floor_z):
    import torch
    position, _ = (array(value) for value in robot.get_position_orientation())
    direction = array(target)[:2] - array(xy)
    yaw = math.atan2(direction[1], direction[0]) if np.linalg.norm(direction) > .001 else 0.
    quaternion = [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]
    robot.set_position_orientation(orientation=torch.tensor(quaternion, dtype=torch.float32))
    wheel_names = getattr(robot, 'floor_touching_base_link_names', ())
    if not wheel_names:
        raise ValueError('Native floor-touching R1 wheel links are required.')
    bottom = min(float(array(robot.links[name].aabb[0])[2]) for name in wheel_names)
    position[:2] = xy
    position[2] += floor_z - bottom + .005
    robot.set_position_orientation(position=torch.tensor(position, dtype=torch.float32),
                                   orientation=torch.tensor(quaternion, dtype=torch.float32))
    robot.keep_still()
    for _ in range(30):
        session.pipeline.og.sim.step()


def _aperture_evidence(robot, bounds):
    openings = {}
    try:
        for arm in robot.arm_names:
            indices = array(robot.controllers[f'gripper_{arm}'].dof_idx).astype(int)
            openings[str(arm)] = float(array(robot.control_limits['position'][1])[indices].sum())
    except (AttributeError, KeyError, ValueError, TypeError):
        openings = {}
    maximum = max(openings.values()) if openings else None
    return {'native_finger_limit_openings_m': openings, 'objects': {
        name: {'aabb_extent_m': (array(high) - array(low)).tolist(),
            'minimum_aabb_extent_m': float((array(high) - array(low)).min()),
            'aperture_warning': maximum is None or float((array(high) - array(low)).min()) > maximum,
            'local_handle_geometry_and_ik_verified': False}
        for name, (low, high) in bounds.items()}, 'physical_grasp_verified': False,
        'scope': 'Native finger limits and global AABB only; local grasp affordances require execution.'}


def _pose_unchanged(obj, pose, translation, tolerance=.015):
    position, quaternion = (array(value) for value in obj.get_position_orientation())
    original_position, original_quaternion = pose
    rotation = abs(float(np.dot(quaternion / np.linalg.norm(quaternion),
                                original_quaternion / np.linalg.norm(original_quaternion))))
    angle = 2 * math.acos(min(1., rotation))
    return bool(np.linalg.norm(position - original_position - translation) <= tolerance and angle <= .035)


def inclined_support_layouts(bounds, support_bounds, surface_z, envelope, *, vertical_only=False):
    """Translate the real apparatus using its complete measured runoff envelope."""
    from .inclined_apparatus import support_envelope_fit
    low, high = (array(value) for value in support_bounds)
    required_low, required_high = (array(value) for value in envelope['required_bounds_m'])
    centre = (required_low[:2] + required_high[:2]) / 2
    for fx, fy in ((.5, .5), (.25, .25), (.75, .75), (.25, .75), (.75, .25)):
        xy = centre if vertical_only else low[:2] + [fx, fy] * (high[:2] - low[:2])
        delta = np.r_[xy - centre, surface_z - envelope['source_floor_z_m'] + .005]
        fit = support_envelope_fit(envelope, support_bounds, delta, support_surface_z_m=surface_z)
        if fit['accepted']:
            yield {'translations': {name: delta.copy() for name in bounds},
                'bounds': {name: (a + delta, b + delta) for name, (a, b) in bounds.items()},
                'inclined_envelope_fit': fit}


def _comparison_candidates(session, group, bounds, policy, floor_z, target, evidence):
    """Yield measured same-room comparison or relational table layouts lazily."""
    from omnigibson.utils.sampling_utils import raytest
    from .support_sampling import same_body, upward_layers
    from .comparison_support_layout import comparison_support_layouts
    from .relation_support_layout import relation_support_layouts
    scene = session.env.scene
    names = {obj.name for obj in group.values()}
    comparison = policy['relocation_policy'] == 'comparison_pairs'
    relational = policy['relocation_policy'] == 'relation_group'
    pairs = [[group[key].name for key in keys] for keys in policy.get('placement_groups', [])]
    positions = {obj.name: array(obj.get_position_orientation()[0]).copy() for obj in group.values()}
    from .interaction_supports import PREFIX
    supports = [obj for obj in _objects(scene) if obj.name not in names and not obj.visual_only
                and not obj.name.startswith(PREFIX)
                and any(word in obj.category for word in ('table', 'counter', 'desk', 'bench', 'shelf', 'cabinet', 'chair', 'stool'))]
    supports.sort(key=lambda obj: (float(np.linalg.norm(
        (array(obj.aabb[0])[:2] + array(obj.aabb[1])[:2]) / 2 - target[:2])), obj.name))
    preferences = set(policy.get('_preferred_support_names', ()))
    supports.sort(key=lambda obj: obj.name not in preferences)
    evidence['support_search'] = []
    for support in supports:
        trace = {'support': support.name, 'candidate_layouts': 0, 'rejections': {},
                 'surface_heights_m': []}
        evidence['support_search'].append(trace)
        def reject(reason):
            trace['rejections'][reason] = trace['rejections'].get(reason, 0) + 1
        rooms = getattr(support, 'in_rooms', ())
        region = policy.get('source_region')
        if region and rooms and region not in rooms:
            reject('different_source_room')
            continue
        low, high = (array(value) for value in support.aabb)
        trace['bounds_m'] = [low.tolist(), high.tolist()]
        heights = []
        for x in (low[0] * .75 + high[0] * .25, (low[0] + high[0]) / 2, low[0] * .25 + high[0] * .75):
            for y in (low[1] * .75 + high[1] * .25, (low[1] + high[1]) / 2, low[1] * .25 + high[1] * .75):
                hits = raytest([x, y, high[2] + .1], [x, y, low[2] - .1], only_closest=False)
                for z in upward_layers(hits, support.prim_path,
                        floor_z + policy['minimum_grasp_height_m'], floor_z + policy['maximum_grasp_height_m']):
                    if not any(abs(z - old) < .005 for old in heights):
                        heights.append(z)
        trace['surface_heights_m'] = heights
        if not heights:
            reject('no_measured_upward_surface_in_height_window')
        for z in heights:
            if policy.get('_inclined_envelope'):
                layouts = list(inclined_support_layouts(bounds, (low, high), z, policy['_inclined_envelope']))
            elif policy['relocation_policy'] in ('shared_tabletop', 'seeded_geometry'):
                from .shared_tabletop import tabletop_layouts
                layouts = tabletop_layouts(group, bounds, (low, high), z, policy)
            elif comparison:
                layouts = comparison_support_layouts(bounds, pairs, (low, high), z)
            elif relational:
                layouts = relation_support_layouts(bounds, positions, (low, high), z)
            else:
                layouts = []
                vertical = policy['relocation_policy'] == 'vertical_only'
                original_centre = (np.minimum.reduce([row[0] for row in bounds.values()])[:2]
                                   + np.maximum.reduce([row[1] for row in bounds.values()])[:2]) / 2
                for fx in (.5, .25, .75):
                    for fy in (.5, .25, .75):
                        if vertical and (fx != .5 or fy != .5):
                            continue
                        try:
                            delta = group_translation(bounds, (low, high),
                                original_centre if vertical else low[:2] + [fx, fy] * (high[:2] - low[:2]), z)
                        except ValueError:
                            continue
                        layouts.append({'translations': {name: delta for name in bounds},
                            'bounds': {name: (a + delta, b + delta) for name, (a, b) in bounds.items()}})
            if not layouts:
                reject('comparison_pairs_do_not_fit' if comparison else 'relation_formation_does_not_fit'
                       if relational else 'unchanged_rigid_group_does_not_fit')
            for layout in layouts:
                required_z = policy.get('_required_translation_z_m')
                if required_z is not None and any(abs(float(delta[2]) - required_z) > .005
                                                  for delta in layout['translations'].values()):
                    reject('surface_breaks_required_common_height_translation')
                    continue
                valid = True
                for pair in (pairs if comparison else [list(bounds)]):
                    bottom = min(bounds[name][0][2] for name in pair)
                    for name in pair:
                        if bounds[name][0][2] > bottom + .03:
                            continue
                        box_low, box_high = layout['bounds'][name]
                        for x in (box_low[0], box_high[0]):
                            for y in (box_low[1], box_high[1]):
                                hit = raytest([x, y, z + .02], [x, y, z - .02])
                                if (not hit['hit'] or not same_body(support.prim_path, hit['rigidBody'])
                                        or float(array(hit['normal'])[2]) < .95
                                        or abs(float(array(hit['position'])[2]) - z) > .005):
                                    valid = False
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    reject('pair_footprint_lacks_local_native_surface')
                    continue
                if policy['relocation_policy'] in ('shared_tabletop', 'seeded_geometry'):
                    from .support_headroom import measure_projected_headroom
                    headroom = measure_projected_headroom(session, support, layout['bounds'],
                        surface_z_m=z, excluded_names=names)
                    if not headroom['available'] or not headroom['accepted']:
                        reject('native_projected_headroom_unavailable' if not headroom['available']
                               else 'native_projected_headroom_obstructed')
                        trace.setdefault('first_headroom_rejection', headroom)
                        continue
                    layout['native_projected_headroom'] = headroom
                if layout.get('inclined_envelope_fit'):
                    # A bounding box alone cannot establish a continuous deck.
                    # Retain the explicit native surface samples separately.
                    a, b = layout['inclined_envelope_fit']['translated_required_bounds_m']
                    surface_samples = []
                    for x in np.linspace(a[0], b[0], max(3, int(np.ceil((b[0] - a[0]) / .10)) + 1)):
                        for y in np.linspace(a[1], b[1], max(3, int(np.ceil((b[1] - a[1]) / .10)) + 1)):
                            hit = raytest([x, y, z + .02], [x, y, z - .02])
                            ok = bool(hit['hit'] and same_body(support.prim_path, hit['rigidBody'])
                                and float(array(hit['normal'])[2]) >= .95)
                            surface_samples.append({'xy_m': [float(x), float(y)], 'accepted': ok})
                    if not all(row['accepted'] for row in surface_samples):
                        reject('apparatus_runoff_native_surface_samples_failed')
                        continue
                    layout['inclined_envelope_fit']['native_surface_samples'] = surface_samples
                trace['candidate_layouts'] += 1
                yield support, z, layout['translations'], layout


def _interaction_bounds(group, policy, floor_z=0.):
    """Use real handles and contents instead of the articulated body's centre."""
    if policy['mode'] != 'articulation':
        keys = policy['group_bindings'] if policy['relocation_policy'] == 'relation_group' else policy['manipulation_bindings']
        return {group[key].name: group[key].aabb for key in keys}
    result = {obj.name: obj.aabb for key, obj in group.items() if key not in policy['manipulation_bindings']}
    for key in policy['manipulation_bindings']:
        obj = group[key]
        # A movable appliance needs an accessible body support even when its
        # handle is already high enough or annotated handle meshes are absent.
        result[obj.name] = obj.aabb
        handles = {}
        for link in obj.links.values():
            for name, mesh in link.collision_meshes.items():
                if 'handle' in str(name).lower() or 'handle' in str(link.prim_path).lower():
                    handles[f'{obj.name}:{name}'] = mesh.aabb
        if handles:
            def cost(name):
                low, high = (array(value) for value in handles[name])
                bottom, centre = low[2] - floor_z, (low[2] + high[2]) / 2 - floor_z
                return max(0., policy['minimum_grasp_height_m'] - bottom,
                           centre - policy['maximum_grasp_height_m']), name
            selected = min(handles, key=cost)
            result[selected] = handles[selected]
    return result


def _repair_safe_spawn(session, robot, group, region, restore, evidence):
    """Keep a failed task diagnostic while repairing its unsafe robot start."""
    scene = session.env.scene
    if restore is None:
        return
    try:
        world, floor_z, grid = _navigation_cells(session, robot, region, {obj.name for obj in group.values()})
    except ValueError as error:
        evidence['source_room_spawn_error'] = str(error)
        try:
            world, floor_z, grid = _navigation_cells(session, robot, None, {obj.name for obj in group.values()})
            evidence['warning_codes'].append('spawn_in_adjacent_room_requires_route_review')
            grid.update(requested_source_region=region, measured_source_region=None)
        except ValueError as fallback_error:
            evidence['safe_spawn_repair_error'] = str(fallback_error)
            return
    floors = [obj for obj in _objects(scene) if obj.category in ('floors', 'floor')]
    floor_paths = [obj.prim_path for obj in floors]
    current = _robot_contact_check(robot, floor_paths, floor_z)
    evidence['original_robot_start_checks'] = current
    wheel_names = getattr(robot, 'floor_touching_base_link_names', ())
    wheel_bottom = min((float(array(robot.links[name].aabb[0])[2]) for name in wheel_names), default=float('inf'))
    if (current['floor_supported'] and current['collision_free'] and current['upright']
            and abs(wheel_bottom - floor_z) <= .015):
        evidence['original_robot_start_safe'] = True
        return
    unchanged = {obj.name: (obj, tuple(array(value).copy() for value in obj.get_position_orientation()))
                 for obj in _objects(scene) if obj is not robot}
    scene_centre = world.mean(axis=0)
    points = [array(obj.get_position_orientation()[0]) for obj in group.values()]
    points = [point for point in points if np.linalg.norm(point[:2] - scene_centre) < 30.]
    target = np.mean(points, axis=0) if points else np.r_[scene_centre, floor_z + .65]
    order = np.argsort(np.linalg.norm(world - target[:2], axis=1), kind='stable')
    for index in order[:16]:
        restore()
        _place_robot(session, robot, world[index], target, floor_z)
        checks = _robot_contact_check(robot, floor_paths, floor_z)
        checks['original_nonrobot_geometry_preserved'] = all(
            _pose_unchanged(obj, pose, np.zeros(3), .005) for obj, pose in unchanged.values())
        if (checks['floor_supported'] and checks['collision_free'] and checks['upright']
                and checks['original_nonrobot_geometry_preserved']):
            evidence.update(robot_spawn_repaired=True, spawn_grid=grid, robot_spawn={
                'robot_position_m': array(robot.get_position_orientation()[0]).tolist(),
                'robot_quaternion_xyzw': array(robot.get_position_orientation()[1]).tolist(), 'checks': checks,
                'task_layout_qualified': False})
            session.invalidate_camera_cache()
            return
    restore()
    evidence['safe_spawn_repair_error'] = 'No measured native floor-supported collision-free upright start passed.'


def initialize_interaction(session, bindings, policy):
    """Finalize one reversible repair; internal proposals never re-enter this boundary."""
    if getattr(session, '_category_interaction_initialized', False):
        raise RuntimeError('Category placement is initialization-only and cannot run twice.')
    session._category_interaction_initialized = True
    if getattr(session, 'reflection_sensor_profile', None):
        # Source head alignment owns the robot start. Generic nearest-object
        # stance search would change the meaning of the optical question.
        from core.sim.esi_reflection import align_initial_robot
        align_initial_robot(session)
        evidence = {'schema': REVISION, 'accepted': True, 'relocated': False,
            'group_translation_m': [0., 0., 0.], 'object_names': [],
            'reason': 'reflection_source_alignment_requires_separate_native_qualification',
            'native_physical_qualification': False, 'robot_sensors_modified': False,
            'warning_codes': ['reflection_scale_view_mobility_pending']}
        session.category_interaction_setup = evidence
        return evidence
    provisional = getattr(session, '_category_interaction_provisional', None)
    if provisional is not None and provisional['accepted']:
        session.category_interaction_setup = provisional
        provisional['provisional_adopted'] = True
        return provisional
    return _initialize_policy(session, bindings, policy)


def initialize_interaction_trial(session, bindings, policy):
    """Try a legacy owned initializer proposal before the single final boundary."""
    if (not getattr(session, '_category_initialization_active', False)
            or getattr(session, '_category_interaction_initialized', False)):
        raise RuntimeError('Placement trials require an active owned scene initializer before controls.')
    evidence = _initialize_policy(session, bindings, policy)
    session._category_interaction_provisional = evidence if evidence['accepted'] else None
    return evidence


def _initialize_policy(session, bindings, policy):
    if policy.get('mode') != 'navigation' and policy.get('support_search') == 'measured_surfaces':
        policy = dict(policy)
        policy.setdefault('_allow_dynamic_background_settling', True)
    if policy.get('placement_groups') and policy['relocation_policy'] in ('independent_groups', 'vertical_only'):
        from .interaction_groups import initialize_groups
        return initialize_groups(session, bindings, policy, _initialize_single)
    return _initialize_single(session, bindings, policy)


def _initialize_single(session, bindings, policy):
    """Measure one meaningful group and jointly search its support and safe approach."""
    evidence = {'schema': REVISION, 'accepted': False, 'relocated': False,
        'group_translation_m': [0., 0., 0.], 'object_names': [], 'attempts': [],
        'physical_grasp_verified': False, 'ik_verified': False,
        'task_process_checks': {name: {'status': 'unverified', 'evidence_kind': 'not_executed'}
            for name in ('route_to_target', 'base_turning_space', 'arm_approach', 'local_grasp',
                         'lift_clearance', 'transport_path', 'container_opening', 'door_sweep',
                         'release_and_retreat', 'required_return')},
        'warning_codes': list(policy.get('warning_codes', ())) + ['native_grasp_execution_unverified']}
    session.category_interaction_setup = evidence
    _notify_interaction(session, 'started', evidence)
    baseline = None
    restore = None
    anchors, robot, group, region = None, None, {}, policy.get('source_region')
    try:
        import torch
        from omnigibson.utils.sampling_utils import raytest
        from omnigibson.utils.usd_utils import PoseAPI
        from .support_sampling import same_body, support_edge_points, upward_layers
        scene, sim = session.env.scene, session.pipeline.og.sim
        if len(session.env.robots) != 1:
            raise ValueError('Interaction setup requires the original single R1.')
        robot = session.env.robots[0]
        from .interaction_supports import hide_supports, support_candidates, activate_support, PREFIX, NAMES
        baseline = sim.dump_state(serialized=False)
        def restore():
            sim.load_state(baseline, serialized=False)
            if anchors is not None:
                anchors.restore()
            hide_supports(session, slot=policy.get('_support_slot', 0))
            PoseAPI.invalidate()
            session.invalidate_camera_cache()
        group = {key: scene.object_registry('name', bindings[key].name) for key in policy['group_bindings']}
        if any(obj is None for obj in group.values()):
            raise ValueError('A declared original interaction group member is unavailable.')
        names = [obj.name for obj in group.values()]
        pending_names = set(policy.get('_pending_group_names', ()))
        evidence['object_names'] = names
        bounds = {obj.name: tuple(array(value).copy() for value in obj.aabb) for obj in group.values()}
        support_keys = policy.get('support_bindings') or policy['group_bindings']
        load_bearing = {key: group[key] for key in support_keys}
        placement_bounds = {obj.name: bounds[obj.name] for obj in load_bearing.values()}
        poses = {obj.name: tuple(array(value).copy() for value in obj.get_position_orientation()) for obj in group.values()}
        apparatus_receipt = None
        if policy.get('layout_parameters', {}).get('inclined_apparatus'):
            from wia.authoring.converters.physical_dynamics_inclined_plane import measure_source_apparatus
            loaded = getattr(session, 'loaded_task', None)
            parameters = (loaded.definition.private.evaluation.parameters if loaded is not None else
                policy['layout_parameters'].get('inclined_source_parameters'))
            if not parameters:
                raise ValueError('Whole-apparatus placement requires exact source-bound native parameters.')
            apparatus_receipt = measure_source_apparatus(session, group['target'], group['slope'], parameters)
            policy = {**policy, '_inclined_envelope': apparatus_receipt['envelope']}
            evidence['inclined_apparatus_proposal'] = copy.deepcopy(apparatus_receipt)
            if not apparatus_receipt['envelope']['support_dimensions']['available']:
                raise ValueError('The full native apparatus envelope exceeds the permitted support dimensions.')
        deformable_receipt = None
        deformable_parameters = None
        if policy.get('layout_parameters', {}).get('deformable_cover'):
            from .deformable_support import capture_deformable_source
            loaded = getattr(session, 'loaded_task', None)
            deformable_parameters = (loaded.definition.private.evaluation.parameters if loaded is not None else
                policy['layout_parameters'].get('deformable_source_parameters'))
            if not deformable_parameters:
                raise ValueError('Deformable support requires exact source-bound native parameters.')
            deformable_receipt = capture_deformable_source(session, group['item'], group['cloth'], deformable_parameters)
            evidence['deformable_source_measurement'] = copy.deepcopy(deformable_receipt)
            _notify_interaction(session, 'source_measured', evidence)
            if not deformable_receipt['accepted']:
                raise ValueError('Original native cloth mesh/physics/cover is unavailable: ' + str(deformable_receipt.get('reason')))
            points = array(deformable_receipt['points_world_m'])
            bounds[group['cloth'].name] = (points.min(axis=0), points.max(axis=0))
            # Pack the whole real mesh, while only the covered rigid item is a
            # required load-bearing floor/table contact for the group.
            placement_bounds = bounds.copy()
        evidence['original_geometry'] = {name: {'position_m': poses[name][0].tolist(),
            'quaternion_xyzw': poses[name][1].tolist(), 'aabb_min_m': box[0].tolist(),
            'aabb_max_m': box[1].tolist()} for name, box in bounds.items()}
        region = policy.get('source_region') or (session.room_name if session.room_name != 'full_scene' else None)
        try:
            world, floor_z, grid = _navigation_cells(session, robot, region, set(names) | pending_names)
        except ValueError as error:
            evidence['source_room_spawn_error'] = str(error)
            world, floor_z, grid = _navigation_cells(session, robot, None, set(names) | pending_names)
            grid.update(requested_source_region=region, measured_source_region=None)
            evidence['warning_codes'].append('spawn_in_adjacent_room_requires_route_review')
        evidence['spawn_grid'] = grid
        floors = [obj for obj in _objects(scene) if obj.category in ('floors', 'floor')]
        if not floors:
            raise ValueError('Native floor bodies are required to verify robot support.')
        scene_low = np.stack([array(obj.aabb[0]) for obj in floors]).min(axis=0)
        scene_high = np.stack([array(obj.aabb[1]) for obj in floors]).max(axis=0)
        outliers = [name for name, box in bounds.items() if
            np.any((box[0][:2] + box[1][:2]) / 2 < scene_low[:2] - .1)
            or np.any((box[0][:2] + box[1][:2]) / 2 > scene_high[:2] + .1)]
        evidence['objects_outside_native_floor_bounds'] = outliers
        if outliers:
            raise ValueError('An original task/reference object lies outside the measured native scene floor bounds.')
        if any(box[1][2] < scene_low[2] - .1 for box in bounds.values()):
            evidence['warning_codes'].append('original_target_below_floor_height_repair_required')
        from .interaction_anchors import InteractionAnchors
        anchors = InteractionAnchors(session, policy, group)
        targets = [group[key] for key in policy['manipulation_bindings']]
        evidence['target_physics'] = {obj.name: {'visual_only': bool(obj.visual_only),
            'fixed_base': bool(obj.fixed_base)} for obj in targets}
        if policy['mode'] != 'navigation' and any(obj.visual_only for obj in targets):
            raise ValueError('A physical interaction target has no enabled collision geometry.')
        if policy['mode'] == 'manipulation' and any(obj.visual_only or obj.fixed_base for obj in targets):
            raise ValueError('A required grasp target is visual-only or fixed-base.')
        relation_layout = policy['relocation_policy'] == 'relation_group'
        target_bounds = {name: tuple(array(value).copy() for value in box)
                         for name, box in _interaction_bounds(group, policy, floor_z).items()}
        evidence['height_screen_before'] = height_screen(target_bounds, floor_z,
            policy['minimum_grasp_height_m'], policy['maximum_grasp_height_m'])
        evidence['aperture_screen'] = _aperture_evidence(robot, target_bounds)
        impossible = [name for name, (low, high) in target_bounds.items()
                      if policy['minimum_grasp_height_m'] + float(high[2] - low[2]) / 2
                      > policy['maximum_grasp_height_m']]
        if impossible:
            evidence['height_window_impossible_without_source_geometry_change'] = impossible
            raise ValueError('An original interaction geometry is too tall for translation-only height repair.')
        low_height = any(
            not row['within_height_window'] for row in evidence['height_screen_before'].values())
        if policy['mode'] != 'navigation':
            evidence['initial_support_contacts'] = _initial_support_contact(session, load_bearing, placement_bounds)
            if not all(evidence['initial_support_contacts'].values()):
                low_height = True
                evidence['warning_codes'].append('unsupported_initial_object_requires_support_repair')
        if policy.get('_required_translation_z_m') is not None and abs(policy['_required_translation_z_m']) > 1e-7:
            low_height = True
        initial_clearance = _independent_candidate_clearance(scene, bounds, policy)
        if initial_clearance['applicable']:
            evidence['independent_group_clearance_before'] = initial_clearance
            if not initial_clearance['accepted']:
                # A supported thin board may still occupy another group's XY
                # footprint. Search a new layout even when its height is valid.
                low_height = True
                evidence['warning_codes'].append('independent_candidate_footprint_requires_repair')
        if policy['mode'] == 'articulation':
            evidence['warning_codes'].append('handle_geometry_requires_native_link_height_measurement')
        revised_tabletop = policy['relocation_policy'] in ('shared_tabletop', 'seeded_geometry')
        if revised_tabletop or apparatus_receipt is not None or deformable_receipt is not None:
            low_height = True
        pair_layout = policy['relocation_policy'] == 'comparison_pairs'
        if low_height and policy['relocation_policy'] not in ('rigid_group', 'comparison_pairs', 'relation_group', 'vertical_only', 'shared_tabletop', 'seeded_geometry'):
            raise ValueError('Original semantics are preserved but a manipulation height is outside the permitted window.')
        background = {obj.name: (obj, tuple(array(value).copy() for value in obj.get_position_orientation()))
            for obj in _objects(scene) if obj is not robot and obj.name not in names
            and obj.name not in pending_names and not obj.name.startswith(PREFIX) and not obj.visual_only}
        floor_paths = [obj.prim_path for obj in floors]
        initial = array(robot.get_position_orientation()[0])
        framing = target_bounds or bounds
        target = (np.mean([box[0] / 2 + box[1] / 2 for box in framing.values()], axis=0)
            if framing else np.r_[world.mean(axis=0), floor_z + .65])
        candidates = [(None, None, np.zeros(3))]
        structured_layout = pair_layout or relation_layout or revised_tabletop
        measured_layout = structured_layout or policy.get('support_search') == 'measured_surfaces'
        if low_height and measured_layout:
            candidates = _comparison_candidates(session, group, placement_bounds, policy, floor_z, target, evidence)
        elif low_height:
            candidates = []
            lows = np.stack([box[0] for box in bounds.values()])
            highs = np.stack([box[1] for box in bounds.values()])
            extent = highs.max(axis=0) - lows.min(axis=0)
            supports = [obj for obj in _objects(scene) if obj.name not in names and not obj.visual_only
                and any(word in obj.category for word in ('table', 'counter', 'desk', 'bench', 'shelf', 'cabinet'))]
            supports.sort(key=lambda obj: (float(np.linalg.norm((array(obj.aabb[0])[:2] + array(obj.aabb[1])[:2]) / 2 - target[:2])), obj.name))
            for support in supports[:16]:
                rooms = getattr(support, 'in_rooms', ())
                if region and rooms and region not in rooms:
                    continue
                support_bounds = tuple(array(value) for value in support.aabb)
                points, _ = support_edge_points(*support_bounds, extent[:2] / 2 + .03,
                    world, initial[:2], maximum_distance=policy['maximum_stance_distance_m'])
                for xy in points[:3]:
                    hits = raytest([*xy, support_bounds[1][2] + .1], [*xy, support_bounds[0][2] - .1], only_closest=False)
                    layers = upward_layers(hits, support.prim_path,
                        floor_z + policy['minimum_grasp_height_m'], floor_z + policy['maximum_grasp_height_m'])
                    for z in layers[:2]:
                        try:
                            delta = group_translation(bounds, support_bounds, xy, z)
                            # Verify the footprint of every lowest original member, not just its origin.
                            for low, high in bounds.values():
                                if low[2] > lows[:, 2].min() + .03:
                                    continue
                                for x in (low[0], high[0]):
                                    for y in (low[1], high[1]):
                                        px, py = np.array([x, y]) + delta[:2]
                                        hit = raytest([px, py, z + .02], [px, py, z - .02])
                                        if (not hit['hit'] or not same_body(support.prim_path, hit['rigidBody'])
                                                or float(array(hit['normal'])[2]) < .95
                                                or abs(float(array(hit['position'])[2]) - z) > .005):
                                            raise ValueError('The unchanged group footprint lacks local native support.')
                            candidates.append((support, z, delta))
                        except ValueError:
                            continue
            if not candidates:
                raise ValueError('No existing same-room support fits the unchanged group at a suitable height.')
        if low_height and policy.get('allow_fallback_table'):
            from itertools import chain
            candidates = chain(candidates, support_candidates(session, group, placement_bounds, policy,
                                                               world, floor_z, target, evidence))
            measured_layout = True
        _notify_interaction(session, 'candidate_generation', evidence)
        diagnostic_attempts = 0
        diagnostic_limit = getattr(session, 'diagnostic_interaction_attempt_limit', None)
        for candidate in (candidates if measured_layout else candidates[:12]):
            support, surface_z, delta = candidate[:3]
            layout = candidate[3] if len(candidate) > 3 else {}
            restore()
            proposed_world = world
            native_stance_fallback = None
            if layout.get('new_support'):
                table = layout['new_support']
                support = activate_support(session, table['centre_xy_m'], table['dimensions_xy_m'],
                                           surface_z, floor_z, slot=policy.get('_support_slot', 0))
                try:
                    proposed_world, _, _ = _navigation_cells(session, robot,
                        grid.get('measured_source_region', region), set(names))
                except ValueError as error:
                    evidence.setdefault('support_layout_rejections', []).append({
                        'support': support.name, 'new_support': copy.deepcopy(table),
                        'failure_stage': 'post_table_stance_candidate_generation', 'reason': str(error)})
                    if 'post_table_support_space_diagnostic' not in evidence:
                        from .support_diagnostics import support_space_diagnostic
                        evidence['post_table_support_space_diagnostic'] = support_space_diagnostic(
                            session, robot, grid.get('measured_source_region', region), set(names))
                    _notify_interaction(session, 'table_stance_rejected', evidence)
                    if apparatus_receipt is not None and getattr(session, 'diagnostic_native_stance_fallback', False):
                        from .native_stance_proposals import propose_native_stances
                        proposed_world, _, native_stance_fallback = propose_native_stances(session, robot,
                            grid.get('measured_source_region', region), set(names), table['centre_xy_m'])
                        native_stance_fallback['previous_circle_reason'] = str(error)
                        evidence.setdefault('diagnostic_native_stance_proposals', []).append(native_stance_fallback)
                    if not len(proposed_world) or native_stance_fallback is None:
                        restore()
                        continue
            deltas = delta if isinstance(delta, dict) else {obj.name: delta for obj in group.values()}
            if deltas:
                common = next(iter(deltas.values()))
                deltas = {obj.name: deltas.get(obj.name, common) for obj in group.values()}
            projected_clearance = _independent_candidate_clearance(scene,
                {name: (low + deltas[name], high + deltas[name]) for name, (low, high) in bounds.items()}, policy)
            if not projected_clearance['accepted']:
                evidence['attempts'].append({'support': support.name if support else None,
                    'failure_stage': 'projected_independent_group_clearance',
                    'checks': {'independent_groups_horizontal_clearance': False},
                    'independent_group_clearance': projected_clearance})
                restore()
                continue
            camera_delta = np.mean(list(deltas.values()), axis=0) if deltas else np.zeros(3)
            proposed_target = target + camera_delta
            target_points = {obj.name: (bounds[obj.name][0] + bounds[obj.name][1]) / 2 + deltas[obj.name]
                             for obj in targets}
            stance_candidates = {}
            stance_evidence = {}
            if measured_layout and targets and support is not None:
                for name, point in target_points.items():
                    distances = np.linalg.norm(proposed_world - point[:2], axis=1)
                    indices = np.flatnonzero(distances <= policy['maximum_stance_distance_m'])
                    nearby = bool(len(indices))
                    stance_evidence[name] = {'nearest_floor_stance_distance_m': float(distances.min()),
                        'within_preferred_radius': nearby, 'initial_approach_required': not nearby,
                        'arm_reach_verified': False}
                    if not nearby:
                        # An episode may start with a base approach. A conservative
                        # map radius is not proof that a table cannot be used.
                        indices = np.argsort(distances, kind='stable')[:4]
                    stance_candidates[name] = indices
                eligible = np.unique(np.concatenate(list(stance_candidates.values())))
            else:
                distances = np.linalg.norm(proposed_world - proposed_target[:2], axis=1)
                eligible = np.flatnonzero(distances <= policy['maximum_stance_distance_m']) if targets else np.arange(len(proposed_world))
                if targets and not len(eligible):
                    eligible = np.argsort(distances, kind='stable')[:4]
                    evidence['warning_codes'].append('initial_base_approach_required')
            retained_index, retained_source = _retained_source_stance_proposal(session, robot,
                proposed_world, floor_paths, floor_z, proposed_target, policy)
            if retained_index is not None:
                eligible = np.unique(np.append(eligible, retained_index))
                evidence['retained_valid_source_stance_proposal'] = retained_source
            if not len(eligible):
                continue
            stance_origin = proposed_target[:2] if relation_layout else initial[:2]
            order = eligible[np.argsort(np.linalg.norm(proposed_world[eligible] - stance_origin, axis=1), kind='stable')]
            for index in order[:4]:
                if diagnostic_limit is not None and diagnostic_attempts >= diagnostic_limit:
                    evidence['failure_stage'] = 'diagnostic_initialization_attempt_limit'
                    raise ValueError('Diagnostic initialization attempt limit reached; task feasibility remains unverified.')
                diagnostic_attempts += 1
                if diagnostic_limit is not None:
                    evidence['diagnostic_attempt_limit'] = diagnostic_limit
                    evidence['diagnostic_attempts_started'] = diagnostic_attempts
                _notify_interaction(session, 'attempt_started', evidence)
                restore()
                if layout.get('new_support'):
                    table = layout['new_support']
                    support = activate_support(session, table['centre_xy_m'], table['dimensions_xy_m'],
                                               surface_z, floor_z, slot=policy.get('_support_slot', 0))
                trial_deformable = None
                if deformable_receipt is not None:
                    from .deformable_support import capture_deformable_source
                    trial_deformable = capture_deformable_source(session, group['item'], group['cloth'], deformable_parameters)
                    if (not trial_deformable['accepted'] or not np.allclose(
                            trial_deformable.get('points_world_m', []), deformable_receipt['points_world_m'],
                            atol=1e-5, rtol=0)):
                        evidence['deformable_restore_failure'] = trial_deformable
                        continue
                if support is not None:
                    for obj in group.values():
                        position, quaternion = poses[obj.name]
                        obj.set_position_orientation(position=torch.tensor(position + deltas[obj.name], dtype=torch.float32),
                            orientation=torch.tensor(quaternion, dtype=torch.float32))
                        obj.keep_still()
                if trial_deformable is not None:
                    from .deformable_support import verify_initial_mesh_translation
                    PoseAPI.invalidate()
                    proof = verify_initial_mesh_translation(session, group['cloth'], trial_deformable,
                        deltas[group['cloth'].name])
                    trial_deformable['initial_mesh_translation'] = proof
                    if not proof['accepted']:
                        evidence['deformable_mesh_translation_failure'] = proof
                        continue
                anchor_evidence = anchors.apply(deltas)
                facing_target = (min(target_points.values(), key=lambda point: np.linalg.norm(point[:2] - proposed_world[index]))
                                 if measured_layout and target_points else proposed_target)
                _place_robot(session, robot, proposed_world[index], facing_target, floor_z)
                checks = _robot_contact_check(robot, floor_paths, floor_z)
                native_stance_contacts = None
                if native_stance_fallback is not None:
                    from .native_floor_contacts import measure_native_floor_contacts
                    native_stance_contacts = measure_native_floor_contacts(session, robot, floor_paths,
                        nominal_floor_z=floor_z)
                    checks['floor_supported'] = native_stance_contacts['floor_supported'] is True
                    checks['collision_free'] = native_stance_contacts['collision_free'] is True
                    checks['native_local_floor_measurement_available'] = native_stance_contacts['measurement_available']
                    actual_position = array(robot.get_position_orientation()[0])
                    checks['native_stance_proposal_position_retained'] = bool(
                        np.linalg.norm(actual_position[:2] - proposed_world[index]) <= .02)
                    checks['native_stance_upright'] = checks['upright']
                reference_evidence = anchors.measure_reference()
                checks['original_scene_reference_unchanged'] = True
                checks['task_group_geometry_preserved'] = all(_pose_unchanged(obj, poses[obj.name], deltas[obj.name])
                    for key, obj in group.items() if trial_deformable is None or key != 'cloth')
                if trial_deformable is not None:
                    from .deformable_support import measure_deformable_placement
                    placement = measure_deformable_placement(session, group['item'], group['cloth'], support,
                        trial_deformable, camera_delta, surface_z)
                    trial_deformable['placement'] = placement
                    checks['native_deformable_mesh_cover_and_support'] = placement['accepted']
                    checks['task_group_geometry_preserved'] &= placement['accepted']
                if pair_layout:
                    pairs = [[group[key] for key in keys] for keys in policy['placement_groups']]
                    centres_now = {obj.name: (array(obj.aabb[0]) + array(obj.aabb[1])) / 2
                                   for obj in group.values()}
                    checks['comparison_pair_geometry_preserved'] = all(np.linalg.norm(
                        (array(first.get_position_orientation()[0]) - array(second.get_position_orientation()[0]))
                        - (poses[first.name][0] - poses[second.name][0])) <= .025 for first, second in pairs)
                    checks['comparison_reference_association_preserved'] = all(
                        np.linalg.norm(centres_now[first.name] - centres_now[own.name]) + .02
                        < np.linalg.norm(centres_now[first.name] - centres_now[other.name])
                        for (first, own), (_, other) in (pairs, pairs[::-1]))
                if relation_layout:
                    members = list(group.values())
                    anchor = members[0]
                    scale = layout.get('layout_scale_xy', 1.)
                    checks['relation_root_similarity_preserved'] = all(np.linalg.norm(
                        array(obj.get_position_orientation()[0]) - array(anchor.get_position_orientation()[0])
                        - (poses[obj.name][0] - poses[anchor.name][0]) * [scale, scale, 1.]) <= .025
                        for obj in members)
                checks['background_geometry_preserved'] = all(_pose_unchanged(obj, pose, np.zeros(3), .005)
                    for obj, pose in background.values())
                background_changes = {}
                for name, (obj, pose) in background.items():
                    if not _pose_unchanged(obj, pose, np.zeros(3), .005):
                        background_changes[name] = {
                            'position_delta_m': (array(obj.get_position_orientation()[0]) - pose[0]).tolist(),
                            'fixed_base': bool(obj.fixed_base), 'category': obj.category}
                protected_categories = {'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings',
                                        'mirror', 'mirrors', 'door', 'window'}
                checks['fixed_background_geometry_preserved'] = all(
                    not row['fixed_base'] and row['category'] not in protected_categories
                    for row in background_changes.values())
                background_valid = checks['background_geometry_preserved'] or (
                    policy.get('_allow_dynamic_background_settling', False)
                    and checks['fixed_background_geometry_preserved'])
                if background_changes and not evidence.get('background_changes_reported'):
                    evidence['background_changes_reported'] = True
                    print('[esi] Native background pose changes: ' + json.dumps(background_changes), flush=True)
                prior = [scene.object_registry('name', name) for name in policy.get('_placed_group_names', ())]
                checks['independent_groups_do_not_interpenetrate'] = all(
                    not np.all(np.minimum(array(obj.aabb[1]), array(other.aabb[1]))
                               - np.maximum(array(obj.aabb[0]), array(other.aabb[0])) > .005)
                    for obj in group.values() for other in prior if other is not None)
                measured_clearance = _independent_candidate_clearance(scene,
                    {obj.name: obj.aabb for obj in group.values()}, policy)
                checks['independent_groups_horizontal_clearance'] = measured_clearance['accepted']
                after = height_screen(_interaction_bounds(group, policy, floor_z), floor_z,
                    policy['minimum_grasp_height_m'], policy['maximum_grasp_height_m'])
                attempt = {'support': support.name if support else None, 'translation_m': None if structured_layout else camera_delta.tolist(),
                    'robot_position_m': array(robot.get_position_orientation()[0]).tolist(), 'checks': checks,
                    'robot_quaternion_xyzw': array(robot.get_position_orientation()[1]).tolist(),
                    'height_screen': after}
                if retained_index is not None and index == retained_index:
                    attempt['retained_valid_source_stance_proposal'] = retained_source
                if native_stance_fallback is not None:
                    attempt['diagnostic_native_stance_proposal_filter'] = native_stance_fallback['schema']
                    attempt['native_stance_contact_measurement'] = native_stance_contacts
                if layout.get('native_projected_headroom') is not None:
                    attempt['native_projected_headroom'] = layout['native_projected_headroom']
                if not checks['task_group_geometry_preserved']:
                    attempt['measured_rigid_pose_deltas'] = {}
                    for key, obj in group.items():
                        if trial_deformable is not None and key == 'cloth':
                            continue
                        actual_position, actual_quaternion = (array(value) for value in obj.get_position_orientation())
                        original_position, original_quaternion = poses[obj.name]
                        cosine = abs(float(np.dot(actual_quaternion / np.linalg.norm(actual_quaternion),
                                                  original_quaternion / np.linalg.norm(original_quaternion))))
                        attempt['measured_rigid_pose_deltas'][obj.name] = {
                            'position_error_m': (actual_position - original_position - deltas[obj.name]).tolist(),
                            'rotation_error_rad': 2 * math.acos(min(1., cosine)),
                            'actual_position_m': actual_position.tolist(),
                            'expected_position_m': (original_position + deltas[obj.name]).tolist(),
                            'actual_quaternion_xyzw': actual_quaternion.tolist(),
                            'source_quaternion_xyzw': original_quaternion.tolist()}
                if trial_deformable is not None:
                    attempt['deformable_placement'] = {key: copy.deepcopy(placement.get(key)) for key in (
                        'available', 'accepted', 'checks', 'reason', 'native_cover',
                        'below_deck_particle_count', 'minimum_particle_z_m',
                        'native_item_support_contact_positions_m', 'selected_support')}
                    if not placement['accepted'] and 'deformable_first_failed_placement' not in evidence:
                        evidence['deformable_first_failed_placement'] = copy.deepcopy(placement)
                if measured_clearance['applicable']:
                    attempt['independent_group_clearance'] = measured_clearance
                if background_changes:
                    attempt['background_pose_changes'] = background_changes
                if measured_layout:
                    attempt['object_translations_m'] = {name: value.tolist() for name, value in deltas.items()}
                if relation_layout:
                    attempt['layout_scale_xy'] = layout.get('layout_scale_xy', 1.)
                if measured_layout:
                    attempt['per_target_candidate_stances'] = {name: [proposed_world[i].tolist() for i in indices[:4]]
                                                              for name, indices in stance_candidates.items()}
                    attempt['per_target_stance_screen'] = stance_evidence
                evidence['attempts'].append(attempt)
                _notify_interaction(session, 'attempt_finished', evidence)
                if (not checks['floor_supported'] or not checks['collision_free'] or not checks['upright']
                        or (native_stance_fallback is not None and not all(checks[key] for key in (
                            'native_local_floor_measurement_available', 'native_stance_proposal_position_retained',
                            'native_stance_upright')))
                        or not checks['task_group_geometry_preserved'] or not background_valid
                        or not checks['independent_groups_do_not_interpenetrate']
                        or not checks['independent_groups_horizontal_clearance']) or (
                        any(not row['within_height_window'] for row in after.values())) or (
                        pair_layout and (not checks['comparison_pair_geometry_preserved']
                                         or not checks['comparison_reference_association_preserved'])) or (
                        relation_layout and not checks['relation_root_similarity_preserved']):
                    continue
                if support is not None:
                    # Only verified contact at the physical support may replace the original floor reference.
                    evidence.update(accepted=True, relocated=True, support_prim_path=support.prim_path,
                        support_name=support.name, support_surface_z_m=float(surface_z),
                        group_translation_m=None if structured_layout else camera_delta.tolist(),
                        object_translations_m={name: value.tolist() for name, value in deltas.items()})
                    bottom_names = set()
                    subsets = ([[key] for key in support_keys] if revised_tabletop else
                               policy['placement_groups'] if pair_layout else [support_keys])
                    for keys in subsets:
                        members = [group[key] for key in keys]
                        bottom = min(bounds[obj.name][0][2] for obj in members)
                        bottom_names.update(obj.name for obj in members if bounds[obj.name][0][2] <= bottom + .03)
                    bottom_members = [obj for obj in group.values() if obj.name in bottom_names]
                    def supported(obj):
                        if obj.fixed_base:
                            low, high = (array(value) for value in obj.aabb)
                            for x in (low[0], high[0]):
                                for y in (low[1], high[1]):
                                    hits = raytest([x, y, surface_z + .02], [x, y, surface_z - .02], only_closest=False)
                                    if not any(hit['hit'] and same_body(support.prim_path, hit['rigidBody'])
                                               and float(array(hit['normal'])[2]) >= .95
                                               and abs(float(array(hit['position'])[2]) - surface_z) <= .005
                                               for hit in hits):
                                        return False
                            attempt['checks']['fixed_container_support_measured_by_native_surface'] = True
                            return abs(float(low[2]) - surface_z) <= .015
                        return touching_task_support(session, obj, None)
                    if not all(supported(obj) for obj in bottom_members):
                        attempt['checks']['native_task_support_contact'] = False
                        evidence.update(accepted=False, relocated=False)
                        continue
                    attempt['checks']['native_task_support_contact'] = True
                    if layout.get('geometric_initial_layout') is not None:
                        evidence['geometric_initial_layout'] = {**layout['geometric_initial_layout'], 'accepted': True}
                    if revised_tabletop:
                        evidence['single_shared_surface'] = True
                if trial_deformable is not None:
                    evidence['deformable_cover'] = trial_deformable
                if apparatus_receipt is not None:
                    from wia.authoring.converters.physical_dynamics_inclined_plane import _pose
                    from .inclined_apparatus import support_envelope_fit
                    fit = support_envelope_fit(apparatus_receipt['envelope'], support.aabb, camera_delta,
                        support_surface_z_m=surface_z)
                    if not fit['accepted']:
                        evidence.update(accepted=False, relocated=False)
                        attempt['checks']['complete_inclined_apparatus_envelope'] = False
                        continue
                    apparatus_receipt['support_reference'] = _pose(support)
                    apparatus_receipt['support_fit'] = layout.get('inclined_envelope_fit', fit)
                    evidence['inclined_apparatus'] = copy.deepcopy(apparatus_receipt)
                    attempt['checks']['complete_inclined_apparatus_envelope'] = True
                evidence.update(accepted=True, robot_spawn=attempt, height_screen_after=after,
                    adapted_geometry={obj.name: {'position_m': array(obj.get_position_orientation()[0]).tolist(),
                        'aabb_min_m': array(obj.aabb[0]).tolist(), 'aabb_max_m': array(obj.aabb[1]).tolist()} for obj in group.values()})
                if background_changes:
                    evidence['background_settling'] = background_changes
                    evidence['warning_codes'].append('dynamic_background_settled_during_initialization_requires_review')
                evidence.update(anchor_evidence, **reference_evidence)
                if layout.get('new_support'):
                    from .interaction_supports import support_names
                    evidence['added_support'] = {**layout['new_support'], 'asset_names': list(support_names(policy.get('_support_slot', 0))),
                        'authorization': 'User requested a simple table or chair when existing supports cannot be used.',
                        'original_asset_geometry_changed': False}
                    evidence['warning_codes'].append('user_authorized_simple_support_added')
                if (any(row['initial_approach_required'] for row in stance_evidence.values())
                        or (retained_index is not None and index == retained_index
                            and retained_source['initial_approach_required'])):
                    evidence['warning_codes'].extend(['initial_base_approach_required',
                        'conservative_stance_radius_exceeded', 'native_reach_and_route_unverified'])
                if support is not None:
                    if policy.get('source_camera_policy') != 'preserve':
                        session.scene_camera_position = (array(session.scene_camera_position) + camera_delta).tolist()
                        viewer = getattr(session, 'postprocess', {}).get('viewer_camera')
                        if isinstance(viewer, dict) and viewer.get('position') is not None:
                            viewer['position'] = (array(viewer['position']) + camera_delta).tolist()
                        evidence['source_reference_camera_translation_m'] = camera_delta.tolist()
                    else:
                        evidence['source_reference_camera_translation_m'] = [0., 0., 0.]
                    if pair_layout:
                        evidence['layout_policy'] = 'pair_preserving_tabletop_repacking'
                        evidence['warning_codes'].extend(['comparison_pair_layout_adapted_requires_review',
                                                         'per_target_route_and_grasp_execution_unverified'])
                    elif relation_layout:
                        evidence['layout_policy'] = 'uniform_root_xy_similarity_on_existing_support'
                        evidence['layout_scale_xy'] = layout['layout_scale_xy']
                        evidence['warning_codes'].extend(['layout_uniform_position_scale_requires_review',
                                                         'native_navigation_unverified'])
                session.invalidate_camera_cache()
                _notify_interaction(session, 'finished', evidence)
                return evidence
            # The next lazy proposal measures native surfaces. Previous rejected
            # trials must not leave props or the robot in those ray queries.
            restore()
        if not evidence['attempts']:
            evidence['failure_stage'] = 'support_candidate_generation'
            raise ValueError('No measured support placement candidate fits this task group; no robot placement was attempted.')
        evidence['failure_stage'] = 'native_support_and_robot_approach'
        raise ValueError('Measured support placements or robot approaches failed native height, support, upright, geometry or collision checks.')
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        if restore is not None:
            restore()
        evidence.update(accepted=False, relocated=False, reason=f'{type(error).__name__}: {error}')
        if robot is not None and policy.get('layout_parameters', {}).get('inclined_apparatus'):
            from .support_diagnostics import support_space_diagnostic
            evidence['restored_support_space_diagnostic'] = support_space_diagnostic(
                session, robot, region, {obj.name for obj in group.values() if obj is not None})
        evidence['failure_reasons'] = interaction_failure_reasons(evidence)
        evidence['warning_codes'].append('interaction_layout_requires_review')
        if robot is not None:
            try:
                _repair_safe_spawn(session, robot, {key: obj for key, obj in group.items() if obj is not None},
                                   region, restore, evidence)
            except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as spawn_error:
                evidence['safe_spawn_repair_error'] = f'{type(spawn_error).__name__}: {spawn_error}'
        _notify_interaction(session, 'failure', evidence)
        return evidence


def interaction_failure_reasons(evidence):
    """Separate measured contact/support faults from static search uncertainty."""
    reasons = []
    mapping = {'collision_free': 'native_collision', 'floor_supported': 'missing_floor_support',
        'upright': 'robot_upright_gate_failed',
        'background_geometry_preserved': 'background_restore_drift',
        'task_group_geometry_preserved': 'task_dependency_group_changed',
        'previous_groups_preserved': 'previous_task_group_changed',
        'independent_groups_horizontal_clearance': 'independent_group_clearance_failed'}
    for index, attempt in enumerate(evidence.get('attempts', [])):
        for key, code in mapping.items():
            if attempt.get('checks', {}).get(key) is False:
                reasons.append({'code': code, 'candidate_attempt': index,
                                'evidence_kind': 'native_measurement', 'check': key})
    message = ' '.join(str(evidence.get(key, '')) for key in
        ('reason', 'source_room_spawn_error', 'safe_spawn_repair_error'))
    if 'conservative' in message or 'map' in message:
        reasons.append({'code': 'conservative_map_screen_failed', 'evidence_kind': 'static_screen',
                        'task_feasibility': 'unverified', 'possible_prerequisites':
                        ['open_a_door', 'legally_move_noncritical_furniture'],
                        'prerequisites_verified': False})
    if not evidence.get('attempts'):
        reasons.append({'code': 'no_candidate_attempted', 'evidence_kind': 'search_result'})
    if not reasons:
        reasons.append({'code': 'measurement_or_precondition_unavailable',
                        'evidence_kind': 'unverified', 'reason': message})
    return reasons

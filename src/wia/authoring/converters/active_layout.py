"""Construct explicit native layouts for the October task redesign.

Run a generated scene with scripts/check_esi_core_native.py. All placement
setters are confined to initialization; evaluators and episodes only read state.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT

from copy import deepcopy
from pathlib import Path

import numpy as np


MATERIAL_BINDING_REVISION = 'construction_v1'
MATERIAL_TOLERANCE = 1e-6
# Installed PrimitiveObject._load creates one base_link for these mesh types.
MATERIAL_PRIMITIVES = {'Cone', 'Cube', 'Cylinder', 'Disk', 'Plane', 'Sphere', 'Torus'}


def _construction_material_binding(parameters):
    if 'material_binding_revision' not in parameters:
        return False
    if parameters['material_binding_revision'] != MATERIAL_BINDING_REVISION:
        raise ValueError('Use the explicit construction_v1 material binding revision or omit it for legacy binding.')
    return True


def _material_specs(layout):
    """Validate exact constructor materials without changing declared numbers."""
    objects = layout['objects']
    specs = {}
    for name, declared in layout.get('friction', {}).items():
        matches = [obj for obj in objects if obj.get('name') == name]
        if (len(matches) != 1 or matches[0].get('type') != 'PrimitiveObject'
                or matches[0].get('primitive_type') not in MATERIAL_PRIMITIVES
                or matches[0].get('prim_type', 0) != 0):
            raise ValueError('Construction materials require one explicitly named rigid PrimitiveObject with a supported base_link.')
        if (not isinstance(declared, dict) or not {'static_friction', 'dynamic_friction'}.issubset(declared)
                or set(declared) - {'static_friction', 'dynamic_friction', 'restitution'}):
            raise ValueError('Construction material declarations support only exact friction and restitution coefficients.')
        material = {'static_friction': declared['static_friction'], 'dynamic_friction': declared['dynamic_friction'],
                    'restitution': declared.get('restitution', .1)}
        for key, value in material.items():
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or value < 0 or (key == 'restitution' and value > 1)):
                raise ValueError('Construction material coefficients must be explicit finite valid numbers.')
        authored = matches[0].get('link_physics_materials')
        if authored is not None and authored != {} and authored != {'base_link': material}:
            raise ValueError('A preexisting link_physics_materials spec conflicts with the declared construction material.')
        specs[name] = material
    return specs


def _read_construction_materials(session, layout, specs):
    """Read installed collision-wrapper getters; never repair a missing binding."""
    receipt = {'revision': MATERIAL_BINDING_REVISION, 'phase': 'construction', 'readback_phase': 'initialization',
        'available': False, 'accepted': False, 'tolerance': MATERIAL_TOLERANCE, 'objects': {},
        'corrective_writes': 0, 'effective_solver_coefficients_verified': False,
        'solver_material_consumption_verified': False,
        'source': 'Existing collision mesh get_applied_physics_material wrapper getters; solver coefficients are unverified.'}
    expected_types = {obj['name']: obj['primitive_type'] for obj in layout['objects'] if obj['name'] in specs}
    errors = (AttributeError, ValueError, TypeError, RuntimeError, KeyError)
    for name, requested in specs.items():
        row = {'requested': deepcopy(requested), 'available': False, 'accepted': False, 'collision_materials': [], 'errors': []}
        receipt['objects'][name] = row
        try:
            obj = session.env.scene.object_registry('name', name)
            if (obj is None or obj.name != name or obj._primitive_type != expected_types[name]
                    or set(obj.links) != {'base_link'} or obj.root_link is not obj.links['base_link']):
                raise ValueError('The exact authored native primitive and single base_link are unavailable.')
            meshes = obj.links['base_link'].collision_meshes.values()
            for mesh in meshes:
                material_row = {'mesh': str(mesh.prim_path), 'available': False, 'accepted': False, 'errors': []}
                row['collision_materials'].append(material_row)
                try:
                    material = mesh.get_applied_physics_material()
                    if material is None:
                        raise ValueError('The native collision mesh has no applied physics material.')
                    material_row['material_prim_path'] = str(material.prim_path)
                    measured = {}
                    for key in requested:
                        value = getattr(material, 'get_' + key)()
                        if isinstance(value, (bool, np.bool_)) or value is None or not np.isfinite(float(value)):
                            raise ValueError('Native material coefficients must be finite measured numbers.')
                        measured[key] = float(value)
                    material_row.update(measured, available=True,
                        accepted=all(abs(measured[key] - requested[key]) <= MATERIAL_TOLERANCE for key in requested))
                except errors as error:
                    material_row['errors'].append(f'{type(error).__name__}: {error}')
            row['available'] = bool(row['collision_materials']) and all(item['available'] for item in row['collision_materials'])
            row['accepted'] = row['available'] and all(item['accepted'] for item in row['collision_materials'])
            if not row['collision_materials']:
                row['errors'].append('No native collision mesh is available for material readback.')
        except errors as error:
            row['errors'].append(f'{type(error).__name__}: {error}')
    receipt['available'] = all(row['available'] for row in receipt['objects'].values())
    receipt['accepted'] = receipt['available'] and all(row['accepted'] for row in receipt['objects'].values())
    return receipt


def configure_scene(config, parameters):
    layout = parameters['layout']
    if config['scene']['scene_model'] != layout['scene_model']:
        raise ValueError('Authored layout and native background must match.')
    construction_materials = _construction_material_binding(parameters)
    materials = _material_specs(layout) if construction_materials else {}
    config['scene'].update(load_room_instances=layout['rooms'], load_room_types=None,
        load_object_categories=None, not_load_object_categories=None)
    config['objects'] = deepcopy(layout['objects'])
    if construction_materials:
        for obj in config['objects']:
            if obj['name'] in materials:
                obj['link_physics_materials'] = {'base_link': deepcopy(materials[obj['name']])}
    import yaml
    from core.sim.esi_omnigibson import _import_pipeline
    pipeline = _import_pipeline(ESI_ROOT)
    robot = yaml.safe_load(
        (Path(pipeline.og.example_config_path) / 'r1_primitives.yaml').read_text())['robots'][0]
    robot.update(deepcopy(layout['robot_start']), name='robot',
        grasping_mode='assisted', disable_grasp_handling=False)
    robot['reset_joint_pos'][6:10] = [.6981317007977318, -1.3962634015954636,
        -.6981317007977318, 0.]
    config['robots'] = [robot]
    return {'revision': parameters['revision'], 'native_layout_qualified': False}


def postprocess_joint_world(session, parameters):
    session.room_name = parameters['layout']['rooms'][0]
    return {'joint_world_postprocess': True, 'revision': parameters['revision'],
        'source_background_preserved': True, 'authored_task_inventory': True}


def _support_ray_surface(obj, support, start, end, stage):
    """Measure the bound upward surface or fail with bounded private evidence."""
    import json
    from wia.tasks.interaction_setup import array
    from wia.tasks.scene_only import local_support_surface
    from omnigibson.utils.sampling_utils import raytest

    def finite_distance(value):
        try:
            measured = float(value)
            return measured if np.isfinite(measured) else None
        except (ValueError, TypeError):
            return None

    hits, exterior_hits = [], []
    try:
        hits = list(raytest(start, end, only_closest=False))
        exterior_hits = [hit for hit in hits if
            (distance := finite_distance(hit.get('distance'))) is not None and distance > 1e-6]
        selected = local_support_surface(support.prim_path, exterior_hits)
        point = array(selected['position']).tolist()
        return {'surface_z_m': float(point[2]), 'distance_m': float(selected['distance']),
                'ray_start_m': start, 'ray_end_m': end, 'hit_position_m': point}
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        def finite_vector(value):
            try:
                measured = array(value)
                return measured.tolist() if measured.shape == (3,) else None
            except (ValueError, RuntimeError, AttributeError, TypeError):
                return None

        summaries = [{'body': str(hit.get('rigidBody', ''))[:256],
                      'position': finite_vector(hit.get('position')),
                      'normal': finite_vector(hit.get('normal')),
                      'distance': finite_distance(hit.get('distance'))}
                     for hit in hits[:8]]
        diagnostic = {'stage': stage, 'object_name': obj.name, 'support_name': support.name,
            'support_prim_path': support.prim_path,
            'object_aabb_m': [array(value).tolist() for value in obj.aabb],
            'support_aabb_m': [array(value).tolist() for value in support.aabb],
            'ray_start_m': start, 'ray_end_m': end, 'hit_count': len(hits),
            'positive_finite_distance_hit_count': len(exterior_hits),
            'hit_summaries': summaries, 'hit_summaries_truncated': len(hits) > 8,
            'reason': str(error)[:512]}
        raise ValueError('Local support ray failed: ' + json.dumps(diagnostic, allow_nan=False)) from error


def _native_pose_vector(value, size, field):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all() or (size == 4 and not np.linalg.norm(result)):
        raise ValueError('Native furniture ' + field + ' must be finite and have its declared pose dimensions.')
    return result.tolist()


def _native_object_pose(obj):
    position, orientation = obj.get_position_orientation()
    return {'position': _native_pose_vector(position, 3, 'position'),
            'orientation': _native_pose_vector(orientation, 4, 'orientation')}


def _fixed_mount_geometry(obj):
    """Require measured fixed, collidable kinematic links for an explicit mount."""
    if getattr(obj, 'fixed_base', None) is not True or getattr(obj, 'visual_only', None) is not False:
        raise ValueError('fixed_mount requires a fixed_base=True, visual_only=False native object.')
    links = getattr(obj, 'links', None)
    root = getattr(obj, 'root_link', None)
    if not isinstance(links, dict) or not links or root is None or not any(root is link for link in links.values()):
        raise ValueError('fixed_mount requires native collision links and an owned physical root link.')
    evidence = []
    for name, link in links.items():
        meshes = getattr(link, 'collision_meshes', None)
        if not isinstance(meshes, dict):
            raise ValueError('fixed_mount native collision geometry is unavailable: ' + str(name))
        if not meshes:
            continue
        missing = object()
        kinematic = getattr(link, 'kinematic_only', missing)
        if kinematic is missing:
            from omnigibson.prims.rigid_kinematic_prim import RigidKinematicPrim
            if type(link) is not RigidKinematicPrim:
                raise ValueError('fixed_mount collision link is not an exact native RigidKinematicPrim: ' + str(name))
            source = 'exact_native_RigidKinematicPrim_type'
        elif type(kinematic) not in (bool, np.bool_) or not bool(kinematic):
            raise ValueError('fixed_mount requires kinematic_only=True for every physical collision link: ' + str(name))
        else:
            source = 'native_kinematic_only_boolean'
        enabled = [getattr(mesh, 'collision_enabled', None) for mesh in meshes.values()]
        if not all(type(value) in (bool, np.bool_) and bool(value) for value in enabled):
            raise ValueError('fixed_mount requires measured enabled native collision meshes: ' + str(name))
        evidence.append({'link_name': name, 'native_type': type(link).__module__ + '.' + type(link).__name__,
            'kinematic_evidence': source, 'collision_mesh_count': len(meshes),
            'all_collision_meshes_enabled': True})
    if not evidence:
        raise ValueError('fixed_mount requires nonempty physical collision geometry.')
    return {'fixed_base': True, 'visual_only': False, 'collision_links': evidence}


def _fixed_mount_snapshot(obj):
    """Read both object and physical root poses without a kinematic pose cache."""
    root = obj.root_link
    clear_cache = getattr(root, 'clear_kinematic_only_cache', None)
    if callable(clear_cache):
        clear_cache()
    return {'object_pose': _native_object_pose(obj), 'root_link_pose': _native_object_pose(root)}


def _fixed_mount_stability(before, after):
    """Compare actual aligned and settled poses; never fabricate static velocity."""
    evidence = {'before': before, 'after': after, 'settle_steps': 90,
        'maximum_position_drift_m': .002, 'maximum_orientation_drift_rad': .01}
    stable = True
    for field in ('object_pose', 'root_link_pose'):
        start, final = before[field], after[field]
        position_drift = float(np.linalg.norm(np.asarray(final['position']) - start['position']))
        first, second = np.asarray(start['orientation']), np.asarray(final['orientation'])
        dot = abs(float(np.dot(first / np.linalg.norm(first), second / np.linalg.norm(second))))
        angle = float(2 * np.arccos(np.clip(dot, 0., 1.)))
        evidence[field + '_position_drift_m'] = position_drift
        evidence[field + '_orientation_drift_rad'] = angle
        stable = stable and position_drift <= .002 and angle <= .01
    evidence['pose_stable'] = bool(stable)
    return evidence


def _initialize_native_object_poses(session, layout):
    """Apply explicit poses only to exact retained native objects at reset."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Native furniture positioning is initialization-only.')
    specs = layout.get('native_object_poses', [])
    if not isinstance(specs, list):
        raise ValueError('Native furniture poses must be an explicit list.')
    authored = {row['name'] for row in layout.get('objects', [])}
    seen, pending = set(), []
    for spec in specs:
        if (not isinstance(spec, dict) or set(spec) != {'name', 'category', 'model', 'position', 'orientation'}
                or any(not isinstance(spec[key], str) or not spec[key].strip() for key in ('name', 'category', 'model'))):
            raise ValueError('Native furniture poses require only exact name/category/model and pose fields.')
        name = spec['name']
        if name in seen or name in authored:
            raise ValueError('Native furniture pose names must be unique and exclude authored layout objects.')
        seen.add(name)
        obj = session.env.scene.object_registry('name', name)
        if (obj is None or obj is getattr(session, 'robot', None)
                or any(getattr(obj, key, None) != spec[key] for key in ('name', 'category', 'model'))):
            raise ValueError('Exact retained native furniture is unavailable: ' + name)
        requested = {'position': _native_pose_vector(spec['position'], 3, 'position'),
                     'orientation': _native_pose_vector(spec['orientation'], 4, 'orientation')}
        pending.append((obj, {'name': name, 'category': spec['category'], 'model': spec['model'],
                              'original_pose': _native_object_pose(obj), 'requested_pose': requested,
                              'evidence_scope': 'initialization_only'}))
    # All identities and poses are validated before changing any retained object.
    for obj, receipt in pending:
        obj.set_position_orientation(**deepcopy(receipt['requested_pose']))
        obj.keep_still()
    return [receipt for _, receipt in pending]


def _finalize_native_object_poses(session, receipts):
    """Read the retained objects after the existing initialization settling."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Native furniture receipts are initialization-only.')
    for receipt in receipts:
        obj = session.env.scene.object_registry('name', receipt['name'])
        receipt['final_pose'] = _native_object_pose(obj)


def _initial_gripper_state(session, layout):
    """Open explicitly empty native fingers only at the initialization boundary."""
    if 'initial_gripper_state' not in layout:
        return None
    if layout['initial_gripper_state'] != 'open':
        raise ValueError('initial_gripper_state supports only the explicit open initialization policy.')
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Initial gripper state is initialization-only.')
    import torch
    from wia.tasks.interaction_setup import array
    robot = session.robot
    arms = list(robot.arm_names)
    if set(arms) != {'left', 'right'}:
        raise ValueError('Initial gripper policy requires both native hands.')
    pending, indices = [], []
    native_positions = robot.get_joint_positions()
    before = array(native_positions).copy()
    lower, upper = (array(value) for value in robot.control_limits['position'])
    for arm in arms:
        if (robot._ag_obj_in_hand[arm] is not None or robot._ag_obj_constraints[arm] is not None
                or robot._ag_release_counter[arm] is not None or int(robot.is_grasping(arm)) != -1):
            raise ValueError('Initial open policy requires an empty native hand: ' + arm)
        name = 'gripper_' + arm
        controller = robot.controllers[name]
        dofs = array(controller.dof_idx).astype(int)
        fingers = robot.finger_joint_names[arm]
        if (type(controller).__name__ != 'JointController' or controller.motor_type != 'position'
                or controller.use_delta_commands or controller.command_input_limits is not None
                or controller.command_dim != 2 or dofs.shape != (2,) or len(fingers) != 2
                or any('prismatic' not in str(robot.joints[key].joint_type).lower() for key in fingers)
                or not np.isfinite(upper[dofs]).all() or np.any(upper[dofs] <= lower[dofs])):
            raise ValueError('Initial open policy requires measured absolute-position native finger limits.')
        indices.extend(dofs.tolist())
        pending.append((arm, name, dofs, upper[dofs].copy()))
    if len(set(indices)) != len(indices):
        raise ValueError('Native hands must own distinct finger DOFs.')
    other_states = {name: deepcopy(controller.dump_state()) for name, controller in robot.controllers.items()
                    if name not in {'gripper_' + arm for arm in arms}}
    dofs = np.asarray(indices, dtype=int)
    targets = np.concatenate([target for _, _, _, target in pending])
    native_device = native_positions.device if isinstance(native_positions, torch.Tensor) else 'cpu'
    native_dtype = native_positions.dtype if isinstance(native_positions, torch.Tensor) else torch.float32
    target_tensor = torch.tensor(targets, dtype=native_dtype, device=native_device)
    native_indices = torch.tensor(dofs, dtype=torch.long, device=native_device)
    # The native position setter resets all controller goals. Restore unrelated
    # native states immediately; only selected finger positions are written.
    try:
        robot.set_joint_positions(target_tensor, indices=native_indices, drive=False)
    finally:
        for name, state in other_states.items():
            robot.controllers[name].load_state(state)
    robot.set_joint_positions(target_tensor, indices=native_indices, drive=True)
    control = robot.get_control_dict()
    joint_position = control['joint_position']
    for _, name, hand_dofs, _ in pending:
        hand_indices = (torch.tensor(hand_dofs, dtype=torch.long, device=joint_position.device)
            if isinstance(joint_position, torch.Tensor) else hand_dofs)
        robot.controllers[name].update_goal(command=joint_position[hand_indices], control_dict=control)
    after = array(robot.get_joint_positions())
    other = np.ones(len(before), dtype=bool)
    other[dofs] = False
    receipt = {'state': 'open', 'scope': 'initialization_only', 'accepted': False,
        'nonfinger_joint_positions_unchanged_after_write': bool(np.array_equal(before[other], after[other])),
        'unrelated_native_controller_states_restored': True, 'hands': {}}
    for arm, name, hand_dofs, target in pending:
        receipt['hands'][arm] = {'dof_indices': hand_dofs.tolist(), 'before_m': before[hand_dofs].tolist(),
            'target_upper_limits_m': target.tolist(), 'after_write_m': after[hand_dofs].tolist(),
            'requested_native_motor_targets_m': target.tolist(), 'empty_before_write': True}
    if not receipt['nonfinger_joint_positions_unchanged_after_write']:
        raise ValueError('Initial finger write changed other joint positions.')
    session.initial_gripper_state_receipt = receipt
    return _finalize_initial_gripper_state(session, receipt)


def _finalize_initial_gripper_state(session, receipt):
    if receipt is None:
        return None
    from wia.tasks.interaction_setup import array
    robot = session.robot
    positions = array(robot.get_joint_positions())
    for arm, row in receipt['hands'].items():
        row['after_m'] = positions[row['dof_indices']].tolist()
        row['empty_after'] = bool(robot._ag_obj_in_hand[arm] is None
            and robot._ag_obj_constraints[arm] is None and robot._ag_release_counter[arm] is None
            and int(robot.is_grasping(arm)) == -1)
        row['accepted'] = bool(row['empty_after'] and np.max(np.abs(
            np.asarray(row['after_m']) - row['target_upper_limits_m'])) <= .001)
    receipt['accepted'] = all(row['accepted'] for row in receipt['hands'].values())
    if not receipt['accepted']:
        raise ValueError('Initial open fingers did not retain empty native hands at their measured upper limits.')
    return receipt


def initialize_scene(session, parameters):
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Authored layout positioning is initialization-only.')
    construction_materials = _construction_material_binding(parameters)
    import torch
    import omnigibson as og
    from wia.tasks.interaction_setup import array, _robot_contact_check
    from omnigibson.object_states import Touching
    layout = parameters['layout']
    gripper_receipt = _initial_gripper_state(session, layout)
    synchronize_poses = layout.get('synchronize_authored_poses_before_alignment', False)
    if type(synchronize_poses) is not bool:
        raise ValueError('synchronize_authored_poses_before_alignment must be a boolean.')
    if construction_materials:
        material_receipt = _read_construction_materials(session, layout, _material_specs(layout))
    else:
        for name, material in layout.get('friction', {}).items():
            import omnigibson.lazy as lazy
            obj = session.env.scene.object_registry('name', name)
            native_material = lazy.isaacsim.core.api.materials.PhysicsMaterial(
                prim_path=f'{obj.prim_path}/Looks/redesign_physics', name=name + '_redesign_physics',
                static_friction=material['static_friction'], dynamic_friction=material['dynamic_friction'],
                restitution=material.get('restitution', .1))
            for link in obj.links.values():
                for mesh in link.collision_meshes.values():
                    mesh.apply_physics_material(native_material)
    native_pose_receipts = _initialize_native_object_poses(session, layout)
    if synchronize_poses:
        pending, seen = [], set()
        for spec in layout.get('supported_items', []):
            name = spec['name']
            authored = [row for row in layout['objects'] if row.get('name') == name]
            obj = session.env.scene.object_registry('name', name)
            if (name in seen or len(authored) != 1 or obj is None or obj is session.robot
                    or getattr(obj, 'name', None) != name):
                raise ValueError('Authored pose synchronization requires unique explicitly supported native objects and authored poses.')
            seen.add(name)
            requested = {'position': _native_pose_vector(authored[0]['position'], 3, 'position'),
                'orientation': _native_pose_vector(authored[0].get('orientation', [0., 0., 0., 1.]), 4, 'orientation')}
            pending.append((obj, requested))
        for obj, requested in pending:
            obj.set_position_orientation(
                position=torch.tensor(requested['position'], dtype=torch.float32),
                orientation=torch.tensor(requested['orientation'], dtype=torch.float32))
            obj.keep_still()
        # Flush native physics/Fabric geometry once for the whole authored batch
        # before any local AABB or support ray is used for alignment.
        og.sim.step()
        pose_sync_receipt = {'additional_initialization_steps': 1,
            'evidence_scope': 'initialization_only', 'objects': {}}
        for obj, requested in pending:
            pose_sync_receipt['objects'][obj.name] = {'authored_pose': requested,
                'measured_pre_alignment_pose': _native_object_pose(obj),
                'measured_pre_alignment_aabb_m': [_native_pose_vector(value, 3, 'aabb') for value in obj.aabb]}
    fixed_mount_snapshots = {}
    for spec in layout.get('supported_items', []):
        obj = session.env.scene.object_registry('name', spec['name'])
        support = session.env.scene.object_registry('name', spec['support'])
        contract = spec.get('support_contract', 'dynamic_support')
        if contract not in ('dynamic_support', 'fixed_mount'):
            raise ValueError('Supported items require dynamic_support or fixed_mount support_contract.')
        if contract == 'fixed_mount':
            _fixed_mount_geometry(obj)
            if spec.get('support_surface_mode') != 'local_raycast':
                raise ValueError('fixed_mount requires native local_raycast support_surface_mode.')
        surface_mode = spec.get('support_surface_mode', 'aabb_top')
        if surface_mode not in ('aabb_top', 'local_raycast'):
            raise ValueError('Supported items require aabb_top or local_raycast support_surface_mode.')
        alignment_mode = spec.get('support_alignment_mode', surface_mode)
        if alignment_mode not in ('aabb_top', 'local_raycast'):
            raise ValueError('Supported items require aabb_top or local_raycast support_alignment_mode.')
        if not synchronize_poses:
            authored = next(row for row in layout['objects'] if row['name'] == obj.name)
            obj.set_position_orientation(
                position=torch.tensor(authored['position'], dtype=torch.float32),
                orientation=torch.tensor(authored.get('orientation', [0., 0., 0., 1.]), dtype=torch.float32))
        position, quaternion = obj.get_position_orientation()
        position = position.clone()
        surface_z = float(array(support.aabb[1])[2])
        if alignment_mode == 'local_raycast':
            low, high = (array(value) for value in obj.aabb)
            slo, shi = (array(value) for value in support.aabb)
            xy = ((low + high) / 2)[:2].tolist()
            surface_z = _support_ray_surface(obj, support,
                [*xy, float(shi[2]) + .10], [*xy, float(slo[2]) - .10], 'initial_alignment')['surface_z_m']
        position[2] += float(surface_z - array(obj.aabb[0])[2]) + .003
        obj.set_position_orientation(position=position, orientation=quaternion)
        obj.keep_still()
        if contract == 'fixed_mount':
            fixed_mount_snapshots[obj.name] = _fixed_mount_snapshot(obj)
    for _ in range(90):
        og.sim.step()
    _finalize_initial_gripper_state(session, gripper_receipt)
    _finalize_native_object_poses(session, native_pose_receipts)
    receipts = {}
    for spec in layout.get('supported_items', []):
        obj = session.env.scene.object_registry('name', spec['name'])
        support = session.env.scene.object_registry('name', spec['support'])
        low, high = (array(x) for x in obj.aabb)
        slo, shi = (array(x) for x in support.aabb)
        surface_mode = spec.get('support_surface_mode', 'aabb_top')
        surface_z = float(shi[2])
        if surface_mode == 'local_raycast':
            xy = ((low + high) / 2)[:2].tolist()
            surface = _support_ray_surface(obj, support,
                [*xy, float(shi[2]) + .10], [*xy, float(slo[2]) - .10], 'final_receipt')
            surface_z = surface['surface_z_m']
        fixed_mount = spec.get('support_contract') == 'fixed_mount'
        receipt = {'support': support.name, 'touching': bool(obj.states[Touching].get_value(support)),
            'within_footprint': bool(np.all(low[:2] >= slo[:2] - .003) and np.all(high[:2] <= shi[:2] + .003)),
            'speed_m_s': None if fixed_mount else float(np.linalg.norm(array(obj.get_linear_velocity()))),
            'bottom_gap_m': float(low[2] - surface_z), 'aabb_m': [low.tolist(), high.tolist()]}
        if surface_mode == 'local_raycast':
            receipt.update(support_surface_mode=surface_mode, support_surface_z_m=surface_z,
                           support_aabb_top_z_m=float(shi[2]),
                           support_ray_distance_m=surface['distance_m'],
                           support_ray_start_m=surface['ray_start_m'],
                           support_ray_end_m=surface['ray_end_m'],
                           support_hit_position_m=surface['hit_position_m'])
        if 'support_alignment_mode' in spec:
            receipt['support_alignment_mode'] = spec['support_alignment_mode']
        if fixed_mount:
            geometry = _fixed_mount_geometry(obj)
            stability = _fixed_mount_stability(fixed_mount_snapshots[obj.name], _fixed_mount_snapshot(obj))
            receipt.update(support_contract='fixed_mount', physical_contact_required=False,
                fixed_mount_geometry=geometry, fixed_mount_stability=stability,
                maximum_absolute_bottom_gap_m=.015, footprint_tolerance_m=.003)
            receipt['accepted'] = (stability['pose_stable'] and receipt['within_footprint']
                and abs(receipt['bottom_gap_m']) < .015)
        else:
            receipt['accepted'] = (receipt['touching'] and receipt['within_footprint'] and
                abs(receipt['bottom_gap_m']) < .015 and receipt['speed_m_s'] < .02)
        receipts[obj.name] = receipt
    floor = session.env.scene.object_registry('name', layout['floor_name'])
    if floor is None:
        from wia.tasks.interaction_setup import _objects
        position = array(session.robot.get_position_orientation()[0])
        floors = [obj for obj in _objects(session.env.scene) if obj.category == 'floors' and
            np.all(position[:2] >= array(obj.aabb[0])[:2]) and np.all(position[:2] <= array(obj.aabb[1])[:2])]
        if not floors:
            raise ValueError('No loaded native floor supports the declared robot start.')
        floor = min(floors, key=lambda obj: abs(float(array(obj.aabb[1])[2] - position[2])))
    robot = _robot_contact_check(session.robot, [floor.prim_path], float(array(floor.aabb[1])[2]))
    result = {'revision': parameters['revision'], 'accepted': bool(
        all(x['accepted'] for x in receipts.values()) and robot['floor_supported'] and robot['collision_free']),
        'supported_items': receipts, 'robot': robot,
        'validation_scope': 'Native initial support and floor contacts; route and manipulation remain unverified.'}
    if fixed_mount_snapshots:
        result['validation_scope'] = ('Native dynamic supports, explicitly fixed mount geometry and pose stability, '
            'and robot floor contacts. Fixed mounts do not certify gravity-supported contact. '
            'Route and manipulation remain unverified.')
    if construction_materials:
        result['material_binding_initialization'] = material_receipt
        result['accepted'] = result['accepted'] and material_receipt['accepted']
    if 'native_object_poses' in layout:
        result['native_object_poses'] = native_pose_receipts
    if synchronize_poses:
        result['authored_pose_synchronization'] = pose_sync_receipt
    if gripper_receipt is not None:
        result['initial_gripper_state'] = gripper_receipt
    session.category_interaction_setup = result
    return result

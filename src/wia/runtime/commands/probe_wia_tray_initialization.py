"""Reproduce production T07 tray initialization in a bounded native apparatus.

Run in the behavior environment with explicit CUDA_VISIBLE_DEVICES:
python -B src/wia/runtime/commands/probe_wia_tray_initialization.py --design-file CASE.json --output-dir DIR
The only asset is tray/gsxbym at unit scale. This diagnostic calls the production
initializer once, uses zero model calls, and never qualifies a full task.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.tasks.interaction_setup import array as _interaction_array

PLATE_NAME = 'collection_plate'
SUPPORT_NAME = 'collection_initial_support'
OFFSET_FIELD = 'tray_position_offset_base_m'
PROBE_ACTIONS = ('TRUNK_DOWN', 'TRUNK_UP', 'BASE_BACK', 'BASE_FWD')


def validate_commands(value):
    """Validate an explicitly authored short public-motion list before native load."""
    from core.action_compiler import MacroLimits, ParameterizedAction, PrimitiveResolution, compile_action
    from core.action_interface import BackendCapabilities
    if not isinstance(value, list) or not value:
        raise ValueError('The optional commands file must contain a nonempty JSON command list.')
    profile = json.loads((ROOT / 'configs/esi_primitives_fine.json').read_text())
    limits = MacroLimits(**json.loads((ROOT / 'configs/esi_parameterized_fine.json').read_text()))
    resolution = PrimitiveResolution(base_translation_m=profile['base']['translation_m'],
        trunk_translation_m=profile['trunk']['translation_m'])
    capabilities = BackendCapabilities(base_translation=True, trunk_height=True)
    commands = []
    for row in value:
        command = ParameterizedAction.from_payload(row, PROBE_ACTIONS)
        plan = compile_action(command, capabilities, resolution, limits)
        if abs(plan.quantization_error or 0.) > 1e-12:
            raise ValueError('Probe commands must be exactly represented by the existing fine profile.')
        commands.append(command.to_dict())
    return commands


def validate_design(value):
    """Accept only the main-authored apparatus settings; never choose a case."""
    required = {'robot_yaw_deg', 'right_eef_position_base_m', 'rim_axis',
                'rim_penetration_m', 'maximum_plate_tilt_deg'}
    if (not isinstance(value, dict) or not required.issubset(value)
            or set(value) - required - {OFFSET_FIELD, 'case_id', 'robot_position_world_m', 'apparatus_position_world_m'}):
        raise ValueError('Provide the five explicit tray settings and only supported optional fields.')
    for key in ('robot_yaw_deg', 'rim_penetration_m', 'maximum_plate_tilt_deg'):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number):
            raise ValueError(key + ' must be a finite number.')
    if value['rim_penetration_m'] <= 0 or not 0 < value['maximum_plate_tilt_deg'] < 90:
        raise ValueError('Penetration must be positive and tilt tolerance must be between zero and 90 degrees.')
    if value['rim_axis'] not in ('x', 'y'):
        raise ValueError('rim_axis must be x or y.')
    for key in ('right_eef_position_base_m', OFFSET_FIELD, 'robot_position_world_m', 'apparatus_position_world_m'):
        vector = value.get(key, [0., 0., 0.])
        array = np.asarray(vector)
        if (array.shape != (3,) or array.dtype.kind not in 'fiu'
                or any(isinstance(item, bool) for item in vector) or not np.isfinite(array).all()):
            raise ValueError(key + ' must have three finite numeric components.')
    if 'case_id' in value and (not isinstance(value['case_id'], str) or not value['case_id'].strip()):
        raise ValueError('case_id must be a nonempty string when supplied.')
    return deepcopy(value)


def robot_start_configuration(design):
    return {'name': 'robot', 'position': deepcopy(design.get('robot_position_world_m', [0., 0., .02])),
        'orientation': Rotation.from_euler('z', design['robot_yaw_deg'], degrees=True).as_quat().tolist(),
        'grasping_mode': 'assisted', 'disable_grasp_handling': False}


def apparatus_start_configuration(design):
    position = deepcopy(design.get('apparatus_position_world_m', [2., 0., 1.]))
    stand_position = [position[0], position[1], position[2] - .2]
    return [
        {'type': 'DatasetObject', 'name': PLATE_NAME, 'category': 'tray', 'model': 'gsxbym',
         'scale': [1., 1., 1.], 'position': position, 'fixed_base': False, 'visual_only': False},
        {'type': 'PrimitiveObject', 'name': SUPPORT_NAME, 'primitive_type': 'Cube',
         'scale': [.25, .36, .04], 'position': stand_position, 'fixed_base': True, 'visual_only': False}]


def production_parameters(design):
    return {'plate': {'name': PLATE_NAME, 'category': 'tray', 'model': 'gsxbym',
        'pinch_reference': 'eef_fingertip', 'right_eef_position_base_m': deepcopy(design['right_eef_position_base_m']),
        'rim_axis': design['rim_axis'], 'rim_penetration_m': design['rim_penetration_m']},
        'maximum_plate_tilt_deg': design['maximum_plate_tilt_deg']}


def _array(value):
    return _interaction_array(value)


def native_rgb_array(value):
    """Detach the camera buffer while retaining its native pixel dtype and range."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pose(obj):
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _bounds(points):
    values = _array(points)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError('Collision bounds require nonempty native N x 3 points.')
    return [values.min(axis=0).tolist(), values.max(axis=0).tolist()]


def ray_endpoints(robot, arm='right'):
    """Read the installed ray endpoint transform, including its end epsilon."""
    result = {}
    for side, points in (('start', robot.assisted_grasp_start_points[arm]),
                         ('end', robot.assisted_grasp_end_points[arm])):
        if not points:
            raise ValueError('Actual native assisted-grasp endpoints are unavailable.')
        result[side] = []
        for point in points:
            link = robot.links[point.link_name]
            position, quaternion = (_array(value) for value in link.get_position_orientation())
            local = _array(point.position)
            world = position + Rotation.from_quat(quaternion).apply(local)
            if side == 'end':
                world += 1e-8
            result[side].append({'link_name': point.link_name, 'local_position_m': local.tolist(),
                'link_pose': _pose(link), 'world_position_m': world.tolist()})
    result['native_transform_source'] = 'ManipulationRobot._find_gripper_raycast_collisions; link pose rotation/translation, end epsilon 1e-8.'
    return result


@contextmanager
def offset_apparatus(session, plate, stand, offset_base, records):
    """Offset each first authored placement, then preserve every native setter call."""
    original_plate, original_stand = plate.set_position_orientation, stand.set_position_orientation
    world_offset = None
    placed = set()

    def wrapped(original, name, *args, **kwargs):
        nonlocal world_offset
        if name in placed:
            return original(*args, **kwargs)
        if not session._category_initialization_active:
            raise RuntimeError('Apparatus offsets are initialization-only.')
        if args or kwargs.get('position') is None or kwargs.get('frame', 'world') != 'world':
            raise ValueError('Production offset wrapper requires an explicit keyword world position.')
        if world_offset is None:
            if name != PLATE_NAME:
                raise RuntimeError('The support must follow the production tray placement.')
            world_offset = Rotation.from_quat(_array(session.robot.get_position_orientation()[1])).apply(offset_base)
        requested = _array(kwargs['position'])
        actual = requested + world_offset
        record = {'object_name': name, 'offset_base_m': list(offset_base),
            'offset_world_m': world_offset.tolist(), 'requested_position_m': requested.tolist(),
            'applied_position_m': actual.tolist(), 'orientation_xyzw': kwargs.get('orientation'),
            'initialization_only': True, 'before_pose': _pose(plate if name == PLATE_NAME else stand)}
        records.append(record)
        # Native assisted grasp can restore a pose through this same instance
        # method. Mark before entering the setter so nested calls also pass through.
        placed.add(name)
        result = original(**{**kwargs, 'position': actual.tolist()})
        record['after_pose'] = _pose(plate if name == PLATE_NAME else stand)
        return result

    plate.set_position_orientation = lambda *args, **kwargs: wrapped(original_plate, PLATE_NAME, *args, **kwargs)
    try:
        stand.set_position_orientation = lambda *args, **kwargs: wrapped(original_stand, SUPPORT_NAME, *args, **kwargs)
        yield
    finally:
        plate.set_position_orientation = original_plate
        stand.set_position_orientation = original_stand


def raw_contacts(obj):
    values = []
    for contact in obj.contact_list():
        row = {'body0': str(contact.body0), 'body1': str(contact.body1)}
        for field in ('position', 'normal', 'impulse'):
            value = getattr(contact, field, None)
            row[field] = _array(value).tolist() if value is not None else None
        values.append(row)
    return values


def native_snapshot(session, plate, label):
    """Read native evidence without invoking setters or physics; retain errors."""
    robot = session.robot
    result = {'stage': label, 'errors': []}

    def read(key, function):
        try:
            result[key] = function()
        except Exception as error:
            result[key] = None
            result['errors'].append(key + ': ' + type(error).__name__ + ': ' + str(error))

    def geometry():
        root = plate.root_link
        root_pose = _pose(root)
        world, meshes = [], []
        for link in plate.links.values():
            if bool(link.is_meta_link):
                continue
            for mesh in link.collision_meshes.values():
                if not bool(mesh.collision_enabled):
                    continue
                points = _array(mesh.transform_local_points_to_world(mesh.points))
                world.append(points)
                meshes.append({'prim_path': str(mesh.prim_path), 'vertex_count': len(points),
                    'local_bounds_m': _bounds(mesh.points), 'world_bounds_m': _bounds(points)})
        all_points = np.concatenate(world)
        scale = _array(root.scale)
        if np.any(scale <= 0):
            raise ValueError('Measured native root-link scale must be positive.')
        local = Rotation.from_quat(root_pose['quaternion_xyzw']).inv().apply(
            all_points - root_pose['position_m']) / scale
        return {'object_pose': _pose(plate), 'object_aabb_m': [_array(value).tolist() for value in plate.aabb],
            'object_scale': _array(plate.scale).tolist(), 'root_pose': root_pose,
            'root_aabb_m': [_array(value).tolist() for value in root.aabb], 'root_scale': scale.tolist(),
            'collision_world_bounds_m': _bounds(all_points), 'collision_root_local_bounds_m': _bounds(local),
            'collision_meshes': meshes}

    def contacts_with_positions():
        candidates, mapping = robot._find_gripper_contacts(arm='right', return_contact_positions=True)
        return {'contacts': [{'prim_path': str(path), 'position_m': _array(point).tolist()}
                for path, point in sorted(candidates, key=lambda item: (str(item[0]), str(item[1])))],
            'robot_contact_links': {str(path): sorted(str(link) for link in links) for path, links in mapping.items()}}

    def grasp():
        indices = _array(robot.controllers['gripper_right'].dof_idx).astype(int)
        held = robot._ag_obj_in_hand['right']
        return {'object_name': getattr(held, 'name', None), 'constraint_active': robot._ag_obj_constraints['right'] is not None,
            'constraint_type': type(robot._ag_obj_constraints['right']).__name__,
            'is_grasping': int(robot.is_grasping('right')), 'release_pending': robot._ag_release_counter['right'] is not None,
            'finger_positions_m': _array(robot.get_joint_positions())[indices].tolist(),
            'finger_limits_m': [_array(value)[indices].tolist() for value in robot.control_limits['position']]}

    read('native_time_s', lambda: float(session.pipeline.og.sim.current_time))
    read('robot_pose', lambda: _pose(robot))
    read('tray_geometry', geometry)
    read('right_eef_pose', lambda: dict(zip(('position_m', 'quaternion_xyzw'),
        (_array(value).tolist() for value in robot.get_eef_pose('right')))))
    read('right_finger_poses', lambda: [{'prim_path': str(link.prim_path), 'pose': _pose(link),
        'aabb_m': [_array(value).tolist() for value in link.aabb]} for link in robot.finger_links['right']])
    read('ray_endpoints', lambda: ray_endpoints(robot))
    read('gripper_contacts', contacts_with_positions)
    read('gripper_raycast_collisions', lambda: sorted(str(path) for path in robot._find_gripper_raycast_collisions(arm='right')))
    read('grasp', grasp)
    read('robot_contacts', lambda: raw_contacts(robot))
    read('tray_contacts', lambda: raw_contacts(plate))
    return result


def native_motion_sample(session, plate):
    """Read raw contact, pose and native attachment evidence without sensors."""
    robot = session.robot
    quaternion = _array(plate.get_position_orientation()[1])
    up = float(Rotation.from_quat(quaternion).apply([0., 0., 1.])[2])
    attached = robot._ag_obj_in_hand['right']
    constraint = robot._ag_obj_constraints['right']
    releasing = robot._ag_release_counter['right']
    grasping = int(robot.is_grasping('right'))
    return {'native_time_s': float(session.pipeline.og.sim.current_time), 'robot_pose': _pose(robot),
        'tray_pose': _pose(plate), 'hand_poses': {arm: dict(zip(('position_m', 'quaternion_xyzw'),
            (_array(v).tolist() for v in robot.get_eef_pose(arm)))) for arm in robot.arm_names},
        'robot_contacts': raw_contacts(robot), 'tray_contacts': raw_contacts(plate),
        'plate_up_dot': up, 'plate_tilt_deg': float(np.degrees(np.arccos(np.clip(up, -1., 1.)))),
        'native_attachment': {'object_name': getattr(attached, 'name', None),
            'constraint_active': constraint is not None, 'release_pending': releasing is not None,
            'is_grasping': grasping},
        'attachment_retained': bool(attached is plate and constraint is not None and releasing is None and grasping == 1)}


def probe_session(env, robot, pipeline):
    from core.performance import PerformanceCounters
    session = SimpleNamespace(env=env, robot=robot, pipeline=pipeline,
        _category_initialization_active=True, _physical_step_generation=0,
        _action_execution_depth=0, _step_camera_data=None, _action_boundary_ready=False,
        performance=PerformanceCounters(), _monitor_public_motion=False, _probe_attachment_lost=False)
    def invalidate_camera_cache():
        session._step_camera_data = None
        session._action_boundary_ready = False
    session.invalidate_camera_cache = invalidate_camera_cache
    from wia.runtime.sim.esi_rendering import action_execution
    session.action_execution = lambda: action_execution(session)
    return session


def production_step_observer(session, plate, report, save):
    """Observe physical samples around the shared image-free control bridge."""
    support_seen_removed = False

    def step_control(action):
        nonlocal support_seen_removed
        from wia.runtime.sim.esi_rendering import step_without_observation
        result = step_without_observation(session, action)
        report['session_control_steps'] += 1
        receipt = getattr(session, 'hidden_collection_initialization', {})
        report['production_receipt'] = deepcopy(receipt)
        if receipt.get('stage') == 'closing_right_gripper':
            support_present = session.env.scene.object_registry('name', SUPPORT_NAME, None) is not None
            if support_present:
                index = report['closure_steps_observed']
                report['closure_steps_observed'] += 1
                report['stage'] = 'closing_right_gripper'
                if index % 5 == 0:
                    sample = native_motion_sample(session, plate)
                    sample['closure_step_index'] = index
                    report['physical_samples'].append(sample)
            elif not support_seen_removed:
                support_seen_removed = True
                report['support_removed'] = True
                report['stage'] = 'support_removed'
                report['physical_samples'].append(native_motion_sample(session, plate))
        if session._monitor_public_motion:
            sample = native_motion_sample(session, plate)
            sample['phase'] = 'public_motion'
            report['physical_samples'].append(sample)
            session._probe_attachment_lost |= not sample['attachment_retained']
        return result

    return step_control


def public_motion_probes(session, plate, commands, report, save):
    """Execute authored metric commands, retaining physical failures and stopping."""
    from core.action_compiler import MacroLimits
    from wia.runtime.sim.esi_actions import create_r1_executor
    from wia.runtime.sim.esi_arm import ArmControlConfig
    from wia.runtime.sim.esi_base import BaseControlConfig
    from wia.runtime.sim.esi_gripper import R1ManipulationController
    from wia.runtime.sim.esi_trunk import R1TrunkController, TrunkControlConfig
    session._category_initialization_active = False
    session._monitor_public_motion = True
    profile_file = ROOT / 'configs/esi_primitives_fine.json'
    limits_file = ROOT / 'configs/esi_parameterized_fine.json'
    profile = json.loads(profile_file.read_text())
    limits = MacroLimits(**json.loads(limits_file.read_text()))
    report['public_probe_configuration'] = {'primitive_profile_file': str(profile_file),
        'primitive_profile_sha256': hashlib.sha256(profile_file.read_bytes()).hexdigest(),
        'limits_file': str(limits_file), 'limits_sha256': hashlib.sha256(limits_file.read_bytes()).hexdigest(),
        'execution_strategy': 'metric_target', 'qualification_scope': 'Short unloaded tray retention probes only.'}
    report['stage'] = 'constructing_public_controller'
    report['public_motion_probes_passed'] = False
    save()
    controller = R1ManipulationController(session, replace(BaseControlConfig.load(), **profile['base']),
        arm_config=replace(ArmControlConfig.load(), **profile['arm']))
    controller.trunk_controller = R1TrunkController(controller,
        replace(TrunkControlConfig.load(ROOT / 'configs/esi_trunk.json'), **profile['trunk']))
    executor = create_r1_executor(controller)
    executor.macro_execution = 'metric_target'
    original_metric = executor._execute_metric
    def measured_metric(*args, **kwargs):
        try:
            return original_metric(*args, **kwargs)
        except Exception as error:
            report.setdefault('public_controller_errors', []).append(type(error).__name__ + ': ' + str(error))
            raise
    executor._execute_metric = measured_metric
    report['public_probe_configuration'].update(resolution=executor.primitive_resolution.to_dict(), limits=limits.to_dict())
    if session._probe_attachment_lost or not native_motion_sample(session, plate)['attachment_retained']:
        raise RuntimeError('Native tray attachment was lost during public-controller construction.')
    report['stage'] = 'post_initialization_public_motion'
    for command in commands:
        row = {'command': deepcopy(command), 'before': native_motion_sample(session, plate), 'result': None}
        report['public_commands'].append(row)
        started, sim_started = time.monotonic(), float(session.pipeline.og.sim.current_time)
        try:
            row['result'] = executor.execute_macro(command, resolution=executor.primitive_resolution,
                limits=limits, should_cancel=lambda: session._probe_attachment_lost)
        finally:
            row.update(wall_time_s=time.monotonic() - started,
                simulated_time_s=float(session.pipeline.og.sim.current_time) - sim_started,
                after=native_motion_sample(session, plate))
            child = (row['result'] or {}).get('child_results', [])
            row['height_measurements'] = [{k: r[k] for k in ('target_height_m', 'trunk_height_before_m',
                'trunk_height_after_m', 'signed_height_error_m', 'absolute_height_error_m') if k in r} for r in child]
            save()
        if session._probe_attachment_lost or not row['after']['attachment_retained']:
            raise RuntimeError('Native tray attachment was lost during the public command.')
        if row['result'].get('success') is not True:
            raise RuntimeError('Public command failed; see the unchanged native command result.')
    report['public_motion_probes_passed'] = True


def _dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(design, output, commands=None):
    report = {'schema': 'wia_tray_initialization_probe_v1', 'design': deepcopy(design),
        'accepted': False, 'stage': 'constructing', 'model_calls': 0, 'manual_attachment': False,
        'full_task_qualified': False, 'scope': 'Single production tray initialization in an empty native Scene.',
        'snapshots': [], 'offset_setter_calls': [], 'session_control_steps': 0,
        'closure_steps_observed': 0, 'production_return': None, 'production_receipt': None,
        'support_removed': False, 'elapsed_wall_s': 0., 'physical_samples': [], 'public_commands': [],
        'public_motion_probes_passed': None, 'initialization_accepted': False, 'authored_commands': deepcopy(commands),
        'full_carrying_feasibility_verified': False}
    began = time.monotonic()
    env = og = session = plate = None
    owns_app = False

    def save():
        report['elapsed_wall_s'] = time.monotonic() - began
        _dump(output / 'result.json', report)
        _dump(output / 'progress.json', {key: report[key] for key in (
            'stage', 'accepted', 'session_control_steps', 'closure_steps_observed', 'support_removed', 'elapsed_wall_s')})

    save()
    try:
        if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
            raise ValueError('An explicit CUDA_VISIBLE_DEVICES selection is required.')
        from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
        from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
        report['gpu'] = allocation_gpu(0, execution_mode='local', renderer_order='pci')
        install_gpu_selection(report['gpu'])
        save()
        from wia.runtime.sim.esi_omnigibson import _import_pipeline, DEFAULT_ESI_ROOT
        pipeline = _import_pipeline(DEFAULT_ESI_ROOT)
        og = pipeline.og
        report['app_present_after_pipeline_import'] = og.app is not None
        if og.app is not None:
            raise RuntimeError('Refuse to reuse or close an existing native application.')
        from omnigibson import simulator
        isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
        from wia.runtime.sim.esi_collection_setup import initialize_collection_tray
        import yaml
        config = yaml.safe_load((Path(og.example_config_path) / 'r1_primitives.yaml').read_text())
        config['scene'] = {'type': 'Scene'}
        config['robots'][0].update(robot_start_configuration(design))
        trunk_pose = json.loads((ROOT / 'configs/esi_arm.json').read_text())['initial_trunk_joint_positions_rad']
        config['robots'][0]['reset_joint_pos'][6:10] = trunk_pose
        config['objects'] = apparatus_start_configuration(design)
        report['environment_config'] = deepcopy(config)
        report['production_parameters'] = production_parameters(design)
        owns_app = True
        env = og.Environment(configs=config)
        robot = env.robots[0]
        plate = env.scene.object_registry('name', PLATE_NAME)
        stand = env.scene.object_registry('name', SUPPORT_NAME)
        session = probe_session(env, robot, pipeline)
        if not np.array_equal(_array(plate.scale), [1., 1., 1.]) or plate.category != 'tray' or plate.model != 'gsxbym':
            raise ValueError('The native tray identity or unit scale differs from the fixed diagnostic contract.')
        report['trunk_initialization'] = {'source_file': str(ROOT / 'configs/esi_arm.json'),
            'target_joint_positions_rad': trunk_pose, 'actual_joint_positions_rad': _array(robot.get_joint_positions())[
                _array(robot.controllers['trunk'].dof_idx).astype(int)].tolist(),
            'method': 'Production r1_primitives reset_joint_pos[6:10] before Environment construction.'}
        report['snapshots'].append(native_snapshot(session, plate, 'before_initializer'))

        session.step_control = production_step_observer(session, plate, report, save)
        report['stage'] = 'production_initializer'
        save()
        with offset_apparatus(session, plate, stand, design.get(OFFSET_FIELD, [0., 0., 0.]), report['offset_setter_calls']):
            report['production_return'] = deepcopy(initialize_collection_tray(session, report['production_parameters']))
        report['production_receipt'] = deepcopy(getattr(session, 'hidden_collection_initialization', None))
        report['support_removed'] = env.scene.object_registry('name', SUPPORT_NAME, None) is None
        report['accepted'] = bool(report['production_return'].get('accepted') is True and report['support_removed'])
        report['initialization_accepted'] = report['accepted']
        if commands:
            public_motion_probes(session, plate, commands, report, save)
        report['stage'] = 'complete'
    except Exception as error:
        report.update(stage='failed', accepted=False, error=type(error).__name__ + ': ' + str(error))
        if session is not None:
            report['production_receipt'] = deepcopy(getattr(session, 'hidden_collection_initialization', None))
    finally:
        if session is not None:
            session._category_initialization_active = False
            report['performance'] = session.performance.snapshot()
        if session is not None and plate is not None:
            report['snapshots'].append(native_snapshot(session, plate, 'final'))
            try:
                report['support_removed'] = env.scene.object_registry('name', SUPPORT_NAME, None) is None
                from core.record.images import save_png
                camera = og.sim.viewer_camera
                base_position, base_quaternion = (_array(value) for value in session.robot.get_position_orientation())
                rotation = Rotation.from_quat(base_quaternion)
                eye, target = (base_position + rotation.apply(value) for value in ([1.7, -1.7, 1.7], [.3, 0., .8]))
                z = eye - target
                z /= np.linalg.norm(z)
                x = np.cross([0., 0., 1.], z)
                x /= np.linalg.norm(x)
                camera.set_position_orientation(position=eye.tolist(),
                    orientation=Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z))).as_quat().tolist())
                for _ in range(3):
                    og.sim.render()
                save_png(output / 'final.png', native_rgb_array(camera.get_obs()[0]['rgb']))
                report['final_image'] = str(output / 'final.png')
            except Exception as error:
                report['final_image_error'] = type(error).__name__ + ': ' + str(error)
        save()
        if owns_app and (env is not None or og.app is not None):
            cleanup = []
            if env is not None:
                cleanup.append(('environment_close', env.close))
            if og.sim is not None:
                cleanup.extend((('simulator_stop', og.sim.stop), ('simulator_close', og.sim.close)))
            cleanup.append(('native_cleanup', og.cleanup))
            for label, function in cleanup:
                try:
                    function()
                except Exception as error:
                    report.setdefault('cleanup_errors', []).append(label + ': ' + type(error).__name__ + ': ' + str(error))
            save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--commands-file', type=Path, help='Optional authored JSON list of bounded public metric commands.')
    args = parser.parse_args(argv)
    raw = args.design_file.read_bytes()
    design = validate_design(json.loads(raw))
    commands_raw = args.commands_file.read_bytes() if args.commands_file else None
    commands = validate_commands(json.loads(commands_raw)) if commands_raw else None
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a fresh output directory to preserve existing evidence.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'input_design.json').write_bytes(raw)
    if commands_raw:
        (args.output_dir / 'input_commands.json').write_bytes(commands_raw)
    source = Path(__file__).read_bytes()
    (args.output_dir / 'probe_source.py').write_bytes(source)
    initializer = ROOT / 'src/wia/runtime/sim/esi_collection_setup.py'
    production_source = initializer.read_bytes()
    (args.output_dir / 'production_initializer_source.py').write_bytes(production_source)
    _dump(args.output_dir / 'code_provenance.json', {'script_file': str(Path(__file__).resolve()),
        'script_sha256': hashlib.sha256(source).hexdigest(), 'design_sha256': hashlib.sha256(raw).hexdigest(),
        'commands_sha256': hashlib.sha256(commands_raw).hexdigest() if commands_raw else None,
        'production_initializer_file': str(initializer), 'production_initializer_sha256': hashlib.sha256(production_source).hexdigest()})
    return 0 if run(design, args.output_dir, commands)['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

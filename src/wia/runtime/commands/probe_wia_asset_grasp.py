"""Probe one supplied asset feature with native R1 assisted grasp on a prepared bench.

Run in the behavior environment with explicit CUDA_VISIBLE_DEVICES:
python -B src/wia/runtime/commands/probe_wia_asset_grasp.py --design-file DESIGN.json --output-dir DIR
This controlled bench diagnostic uses no model and never qualifies a full task.
Only initialization places apparatus; episode motion uses existing controller macros.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
STAGES = ('GRASP', 'MV_UP', 'MV_RIGHT', 'MV_DOWN', 'RELEASE')
MINIMUM_LIFT_M = .05
MINIMUM_LATERAL_M = .05
DEFAULT_BENCH_DIMENSIONS_M = (.8, .8, .04)
STANCE_FIELDS = ('upright', 'floor_supported', 'collision_free')


def validate_design(value):
    """Require the main-selected identity, pose, feature and source provenance."""
    required = {'category', 'model', 'scale', 'quaternion_xyzw', 'grasp_point_local_m',
                'source_task_file', 'source_binding', 'case_id'}
    optional = {'initialization_actions', 'bench_dimensions_m'}
    if (not isinstance(value, dict) or not required.issubset(value)
            or not set(value).issubset(required | optional)):
        raise ValueError('Design requires the documented asset and provenance fields with only supported optional fields.')
    for field in ('category', 'model', 'source_task_file', 'source_binding', 'case_id'):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(field + ' must be an explicit nonempty string.')
    for field, size in (('scale', 3), ('quaternion_xyzw', 4), ('grasp_point_local_m', 3)):
        raw = np.asarray(value[field])
        if raw.shape != (size,) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
            raise ValueError(field + ' must be an explicit finite numeric vector.')
    if np.any(np.asarray(value['scale']) <= 0):
        raise ValueError('Scale must be positive.')
    if not np.isclose(np.linalg.norm(value['quaternion_xyzw']), 1., rtol=0., atol=1e-6):
        raise ValueError('Quaternion must already be normalized; the design is never silently changed.')
    if 'bench_dimensions_m' in value:
        dimensions = np.asarray(value['bench_dimensions_m'])
        if (dimensions.shape != (3,) or dimensions.dtype.kind not in 'fiu'
                or any(isinstance(item, bool) for item in value['bench_dimensions_m'])
                or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
            raise ValueError('bench_dimensions_m requires three finite positive numbers.')
    actions = value.get('initialization_actions', [])
    if not isinstance(actions, list):
        raise ValueError('initialization_actions must be an ordered list.')
    for action in actions:
        if (not isinstance(action, dict) or set(action) != {'action', 'distance_m'}
                or action['action'] not in ('MV_FWD', 'MV_LEFT')):
            raise ValueError('Initialization actions support only explicit MV_FWD or MV_LEFT distances.')
        distance = action['distance_m']
        if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not np.isfinite(distance) or not 0 < distance <= .25):
            raise ValueError('Initialization distance_m must be finite, positive and at most .25 m.')
    return deepcopy(value)


def native_holding(state, target_name):
    """Command completion alone is insufficient evidence of assisted attachment."""
    return bool(isinstance(target_name, str) and target_name and isinstance(state, dict)
                and state.get('object_name') == target_name
                and state.get('constraint_active') is True and state.get('is_grasping') == 1
                and state.get('release_pending') is False)


def empty_hand(state):
    return bool(isinstance(state, dict) and state.get('object_name') is None
                and state.get('constraint_active') is False and state.get('is_grasping') == -1
                and state.get('release_pending') is False and state.get('gripper_open') is True)


def stance_accepted(value):
    return isinstance(value, dict) and all(value.get(field) is True for field in STANCE_FIELDS)


def reduce_success(report):
    """Reduce only complete measured bench evidence; missing values cannot pass."""
    failures = []
    initial = report.get('initial', {})
    if (initial.get('touching') is not True or initial.get('robot_contact') is not False
            or not empty_hand(initial.get('grasp'))):
        failures.append('initial_supported_collision_free_state_unavailable_or_rejected')
    if initial.get('robot_bench_contact') is not False:
        failures.append('initial_robot_bench_contact_unavailable_or_rejected')
    if not stance_accepted(initial.get('robot_stance')):
        failures.append('initial_robot_stance_unavailable_or_rejected')
    if report.get('hold_loss') is not False:
        failures.append('holding_lost_or_unavailable')
    receipts = report.get('commands', [])
    if [row.get('action') for row in receipts] != list(STAGES):
        failures.append('required_command_sequence_incomplete')
    else:
        for row in receipts:
            if row.get('receipt', {}).get('completed') is not True:
                failures.append(row['action'] + ':controller_not_completed')
            if row['action'] != 'RELEASE' and not native_holding(row.get('measurement', {}).get('grasp'), report.get('target_name')):
                failures.append(row['action'] + ':native_holding_not_verified')
    for field, minimum in (('measured_lift_m', MINIMUM_LIFT_M), ('measured_lateral_displacement_m', MINIMUM_LATERAL_M)):
        value = report.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value < minimum):
            failures.append(field + ':minimum_not_verified')
    final = report.get('final', {})
    for field in ('touching', 'on_top', 'footprint_inside'):
        if final.get(field) is not True:
            failures.append('final:' + field + ':not_verified')
    if final.get('robot_contact') is not False or not empty_hand(final.get('grasp')):
        failures.append('final_released_empty_hand_not_verified')
    if not stance_accepted(final.get('robot_stance')):
        failures.append('final_robot_stance_unavailable_or_rejected')
    if report.get('final_hold_steps') != 60:
        failures.append('final_hold_incomplete')
    return {'accepted': not failures, 'failure_reasons': failures,
            'full_task_qualified': False, 'model_calls': 0,
            'qualification_scope': 'Prepared single-asset bench diagnostic only.'}


def _dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(design, output):
    """Own one isolated native environment and flush evidence at stage boundaries."""
    report = {'schema': 'wia_asset_grasp_probe_v1', 'design': deepcopy(design),
        'stage': 'constructing', 'accepted': False, 'full_task_qualified': False,
        'model_calls': 0, 'target_name': 'wia_probe_asset', 'manual_attachment': False,
        'qualification_scope': 'Prepared single-asset bench diagnostic only.',
        'thresholds': {'minimum_lift_m': MINIMUM_LIFT_M, 'minimum_lateral_displacement_m': MINIMUM_LATERAL_M,
                       'commanded_lift_m': .10, 'commanded_lateral_displacement_m': .10},
        'commands': [], 'initialization_commands': [], 'trajectory': [], 'screenshots': [], 'hold_loss': False,
        'initialization_action_count': 0, 'episode_action_count': 0,
        'initialization_control_steps': 0, 'episode_control_steps': 0,
        'final_hold_steps': 0,
        'step_count_scope': 'Observed controller env.step calls; excludes internal Environment construction.',
        'apparatus_setters_after_boundary': 0}
    save = lambda: _dump(output / 'result.json', report)
    save()
    env, og = None, None
    phase, monitor, current_stage = 'initialization', False, 'constructing'
    try:
        if not os.environ.get('CUDA_VISIBLE_DEVICES', '').strip():
            raise ValueError('Select an explicit CUDA_VISIBLE_DEVICES before launching this diagnostic.')
        from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
        from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
        gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
        install_gpu_selection(gpu)
        report['gpu'] = gpu
        save()
        from wia.runtime.sim.esi_omnigibson import _import_pipeline, DEFAULT_ESI_ROOT, _array
        pipeline = _import_pipeline(DEFAULT_ESI_ROOT)
        og = pipeline.og
        if og.app is not None:
            raise RuntimeError('Refuse to reuse or close a preexisting OmniGibson application.')
        from omnigibson import simulator
        isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
        from omnigibson.object_states import Touching, OnTop
        from wia.runtime.sim.esi_gripper import R1ManipulationController
        from core.record.images import save_png
        from wia.tasks.interaction_setup import _robot_contact_check
        import torch
        import yaml

        config = yaml.safe_load((Path(og.example_config_path) / 'r1_primitives.yaml').read_text())
        config['scene'] = {'type': 'Scene'}
        config['robots'][0].update(name='robot', position=[0., 0., .02],
            orientation=[0., 0., 0., 1.], grasping_mode='assisted', disable_grasp_handling=False)
        bench_dimensions = list(design.get('bench_dimensions_m', DEFAULT_BENCH_DIMENSIONS_M))
        config['objects'] = [
            {'type': 'DatasetObject', 'name': report['target_name'], 'category': design['category'],
             'model': design['model'], 'scale': design['scale'], 'orientation': design['quaternion_xyzw'],
             'position': [2., 0., 1.], 'fixed_base': False, 'visual_only': False},
            {'type': 'PrimitiveObject', 'name': 'wia_probe_bench', 'primitive_type': 'Cube', 'size': 1.,
             'scale': bench_dimensions, 'position': [2., 0., .8], 'fixed_base': True, 'visual_only': False}]
        report['environment_config'] = deepcopy(config)
        env = og.Environment(configs=config)
        robot = env.robots[0]
        obj = env.scene.object_registry('name', report['target_name'])
        bench = env.scene.object_registry('name', 'wia_probe_bench')
        session = SimpleNamespace(env=env, robot=robot, pipeline=pipeline)

        def step_control(action):
            result = env.step(action)
            report[phase + '_control_steps'] += 1
            return result

        session.step_control = step_control
        q = _array(robot.get_joint_positions()).copy()
        trunk = _array(robot.controllers['trunk'].dof_idx).astype(int)
        trunk_pose = json.loads((ROOT / 'configs/esi_arm.json').read_text())['initial_trunk_joint_positions_rad']
        q[trunk] = trunk_pose
        robot.reset_joint_pos = torch.tensor(q, dtype=torch.float32)
        robot.reset()
        controller = R1ManipulationController(session)
        report['initialization_action_count'] += 1
        opening = controller.execute('RELEASE')
        report['initial_release_receipt'] = opening
        if opening.get('completed') is not True:
            raise RuntimeError('Initialization RELEASE did not complete.')

        def grasp_state():
            state = controller.debug_gripper_state()
            state['is_grasping'] = int(robot.is_grasping('left'))
            return state

        def grasp_diagnostics():
            """Read native candidate gates without establishing any attachment."""
            try:
                contacts, fingers = robot._find_gripper_contacts(arm='left')
                rays = robot._find_gripper_raycast_collisions(arm='left')
                return {'contact_candidates': sorted(contacts), 'ray_candidates': sorted(rays),
                    'contact_and_ray_candidates': sorted(contacts.intersection(rays)),
                    'contacting_finger_links': {str(key): sorted(value) for key, value in fingers.items()},
                    'target_link_masses_kg': {name: float(link.mass) for name, link in obj.links.items()
                        if not bool(link.is_meta_link)}}
            except Exception as error:
                return {'measurement_error': repr(error)}

        def vertices():
            points = []
            for link in obj.links.values():
                if bool(link.is_meta_link):
                    continue
                for mesh in link.collision_meshes.values():
                    if bool(mesh.collision_enabled):
                        points.append(_array(mesh.transform_local_points_to_world(mesh.points)))
            if not points:
                raise RuntimeError('Actual object collision vertices are unavailable.')
            values = np.concatenate(points)
            if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise RuntimeError('Actual collision geometry is not finite N x 3.')
            return values

        def measurement():
            row = {'native_time_s': float(og.sim.current_time), 'grasp': grasp_state(),
                'position_m': _array(obj.get_position_orientation()[0]).tolist(),
                'quaternion_xyzw': _array(obj.get_position_orientation()[1]).tolist(),
                'aabb_m': [_array(value).tolist() for value in obj.aabb],
                'measurement_errors': []}
            try:
                base_position, base_quaternion = (_array(value) for value in robot.get_position_orientation())
                row['robot_base_pose'] = {'position_m': base_position.tolist(),
                    'quaternion_xyzw': base_quaternion.tolist()}
                row['robot_stance'] = _robot_contact_check(robot, ['/World/ground_plane'], 0.0)
            except Exception as error:
                message = 'robot_stance: ' + repr(error)
                row['robot_base_pose'] = None
                row['robot_stance'] = {**dict.fromkeys(STANCE_FIELDS), 'measurement_error': message}
                row['measurement_errors'].append(message)
            try:
                points = vertices()
                row.update(collision_lowest_z_m=float(points[:, 2].min()), collision_centroid_m=points.mean(axis=0).tolist())
            except Exception as error:
                row['measurement_errors'].append('collision_geometry: ' + repr(error))
            for field, state in (('touching', Touching), ('on_top', OnTop)):
                try:
                    value = obj.states[state].get_value(bench)
                    if not isinstance(value, (bool, np.bool_)):
                        raise ValueError('Native relation must be boolean.')
                    row[field] = bool(value)
                except Exception as error:
                    row[field] = None
                    row['measurement_errors'].append(field + ': ' + repr(error))
            try:
                contacts = list(obj.contact_list())
                row['contacts'] = [{'body0': str(contact.body0), 'body1': str(contact.body1)} for contact in contacts]
                row['robot_contact'] = any(str(body) == robot.prim_path or str(body).startswith(robot.prim_path + '/')
                    for contact in contacts for body in (contact.body0, contact.body1))
                robot_contacts = list(robot.contact_list())
                row['robot_contacts'] = [{'body0': str(contact.body0), 'body1': str(contact.body1)} for contact in robot_contacts]
                row['robot_bench_contact'] = any(str(body) == bench.prim_path or str(body).startswith(bench.prim_path + '/')
                    for contact in robot_contacts for body in (contact.body0, contact.body1))
                row['robot_contact'] |= any(str(body) == obj.prim_path or str(body).startswith(obj.prim_path + '/')
                    for contact in robot_contacts for body in (contact.body0, contact.body1))
            except Exception as error:
                row['robot_contact'] = None
                row['robot_bench_contact'] = None
                row['measurement_errors'].append('contacts: ' + repr(error))
            try:
                lo, hi = (_array(value) for value in bench.aabb)
                row['footprint_inside'] = bool(np.all(points[:, :2] >= lo[:2]) and np.all(points[:, :2] <= hi[:2]))
            except Exception as error:
                row['footprint_inside'] = None
                row['measurement_errors'].append('footprint: ' + repr(error))
            return row

        def observe_step():
            row = measurement()
            row.update(phase=phase, stage=current_stage)
            report['trajectory'].append(row)
            if monitor and not native_holding(row['grasp'], obj.name):
                report['hold_loss'] = True
            if len(report['trajectory']) % 10 == 0:
                save()

        controller.on_step = observe_step
        for action in design.get('initialization_actions', []):
            current_stage = report['stage'] = 'initialization_' + action['action']
            report['initialization_action_count'] += 1
            save()
            receipt = controller.execute_metric(action['action'], action['distance_m'])
            report['initialization_commands'].append({'action': action['action'],
                'distance_m': action['distance_m'], 'receipt': receipt})
            save()
            if receipt.get('completed') is not True:
                report.update(stage='initialization_rejected',
                    rejection=action['action'] + ':initialization_controller_not_completed')
                report.update(reduce_success(report))
                save()
                return report
        ray_points = {}
        for name, values in (('start', robot.assisted_grasp_start_points['left']),
                             ('end', robot.assisted_grasp_end_points['left'])):
            if not values:
                raise RuntimeError('Native left assisted-grasp ray endpoints are unavailable.')
            ray_points[name] = []
            for point in values:
                link_position, link_quaternion = (_array(value) for value in robot.links[point.link_name].get_position_orientation())
                world = link_position + Rotation.from_quat(link_quaternion).apply(_array(point.position))
                ray_points[name].append(world)
        starts, ends = np.asarray(ray_points['start']), np.asarray(ray_points['end']) + 1e-8
        midpoint = ((starts[:, None, :] + ends[None, :, :]) / 2).mean(axis=(0, 1))
        measured_scale = _array(obj.scale)
        if not np.allclose(measured_scale, design['scale'], rtol=1e-6, atol=1e-6):
            raise RuntimeError('Native asset scale does not match the supplied design.')
        offset = Rotation.from_quat(design['quaternion_xyzw']).apply(np.asarray(design['grasp_point_local_m']) * measured_scale)
        # Native mesh world transforms may still reflect the previous physics
        # sample immediately after a setter. Measure before moving, then rigidly
        # transform that cloud only to choose the initial bench placement.
        old_position, old_quaternion = (_array(value) for value in obj.get_position_orientation())
        local_points = Rotation.from_quat(old_quaternion).inv().apply(vertices() - old_position)
        requested_position = midpoint - offset
        points = Rotation.from_quat(design['quaternion_xyzw']).apply(local_points) + requested_position
        obj.set_position_orientation(position=requested_position.tolist(), orientation=design['quaternion_xyzw'])
        obj.keep_still()
        centre = (points.min(axis=0) + points.max(axis=0)) / 2
        centre[2] = points[:, 2].min() - .003 - bench_dimensions[2] / 2
        bench.set_position_orientation(position=centre.tolist(), orientation=[0., 0., 0., 1.])
        report['apparatus_initialization'] = {'ray_start_world_m': starts.tolist(), 'ray_end_world_m': ends.tolist(),
            'mean_ray_midpoint_world_m': midpoint.tolist(), 'placed_object_pose': [_array(v).tolist() for v in obj.get_position_orientation()],
            'bench_pose': [_array(v).tolist() for v in bench.get_position_orientation()],
            'bench_dimensions_m': bench_dimensions, 'deck_gap_m': .003,
            'bench_placement_geometry': 'Rigidly transformed pre-setter collision cloud; final support uses actual settled native evidence.',
            'ray_transform_source': 'Installed ManipulationRobot._find_gripper_raycast_collisions finger-link pose transform.',
            'placement_setters_complete': True}
        report['robot_measurements'] = {'scale': _array(robot.scale).tolist(),
            'aabb_m': [_array(value).tolist() for value in robot.aabb],
            'aabb_dimensions_m': (_array(robot.aabb[1]) - _array(robot.aabb[0])).tolist(),
            'joint_positions': _array(robot.get_joint_positions()).tolist(),
            'left_finger_limits_m': [_array(value)[controller.gripper_dofs].tolist() for value in robot.control_limits['position']],
            'jaw_state': grasp_state(), 'initial_trunk_target_rad': trunk_pose}
        # Every object, bench and direct robot setter is above this boundary.
        report['apparatus_setter_boundary_native_time_s'] = float(og.sim.current_time)
        current_stage = report['stage'] = 'settling'
        save()
        for _ in range(60):
            controller.step_base_debug()
        report['initial'] = measurement()
        report['initial']['native_grasp_diagnostics'] = grasp_diagnostics()
        report['initialization_settle_steps'] = 60

        camera = None
        try:
            camera = og.sim.viewer_camera
            eye, target = midpoint + [1.3, -1.4, .9], midpoint
            z = eye - target
            z /= np.linalg.norm(z)
            x = np.cross([0., 0., 1.], z)
            x /= np.linalg.norm(x)
            camera.set_position_orientation(position=eye.tolist(),
                orientation=Rotation.from_matrix(np.column_stack((x, np.cross(z, x), z))).as_quat().tolist())
            report['diagnostic_camera'] = {'position_m': eye.tolist(), 'look_at_m': target.tolist(), 'fixed': True}
        except Exception as error:
            report['camera_error'] = repr(error)

        def screenshot(stage):
            if camera is None:
                return
            try:
                for _ in range(3):
                    og.sim.render()
                path = output / (stage.lower() + '.png')
                save_png(path, _array(camera.get_obs()[0]['rgb']))
                report['screenshots'].append({'stage': stage, 'path': str(path)})
            except Exception as error:
                report['screenshots'].append({'stage': stage, 'error': repr(error)})

        screenshot('initial')
        if (report['initial'].get('touching') is not True or report['initial'].get('robot_contact') is not False
                or report['initial'].get('robot_bench_contact') is not False
                or not stance_accepted(report['initial'].get('robot_stance'))
                or not empty_hand(report['initial'].get('grasp'))):
            report.update(stage='initial_rejected', rejection='Initial support or robot collision gate failed.')
            report.update(reduce_success(report))
            save()
            return report
        phase = 'episode'
        report['episode_boundary_native_time_s'] = float(og.sim.current_time)
        for token in STAGES:
            current_stage = report['stage'] = token
            monitor = token in ('MV_UP', 'MV_RIGHT', 'MV_DOWN')
            if monitor and not native_holding(grasp_state(), obj.name):
                report['hold_loss'] = True
                report['rejection'] = token + ':native_holding_missing_before_motion'
                break
            report['episode_action_count'] += 1
            save()
            receipt = (controller.execute(token) if token in ('GRASP', 'RELEASE') else
                       controller.execute_metric(token, .10, should_cancel=lambda: report['hold_loss']))
            measured = measurement()
            measured['native_grasp_diagnostics'] = grasp_diagnostics()
            report['commands'].append({'action': token, 'receipt': receipt, 'measurement': measured})
            if token == 'MV_UP' and 'collision_lowest_z_m' in measured and 'collision_lowest_z_m' in report['initial']:
                report['measured_lift_m'] = measured['collision_lowest_z_m'] - report['initial']['collision_lowest_z_m']
            if token == 'MV_RIGHT' and 'collision_centroid_m' in measured:
                lifted = report['commands'][-2]['measurement'].get('collision_centroid_m')
                if lifted is not None:
                    report['measured_lateral_displacement_m'] = float(np.linalg.norm(np.asarray(measured['collision_centroid_m'])[:2] - np.asarray(lifted)[:2]))
            screenshot(token)
            save()
            if receipt.get('completed') is not True or report['hold_loss']:
                report['rejection'] = token + ':controller_failure_or_hold_loss'
                break
            if token != 'RELEASE' and not native_holding(measured['grasp'], obj.name):
                report['rejection'] = token + ':native_holding_not_verified'
                break
        monitor = False
        if len(report['commands']) == len(STAGES) and report['commands'][-1]['receipt'].get('completed') is True:
            current_stage = report['stage'] = 'final_hold'
            for _ in range(60):
                controller.step_base_debug()
                report['final_hold_steps'] += 1
        report['final'] = measurement()
        screenshot('final')
        report['stage'] = 'complete'
        report.update(reduce_success(report))
        save()
        return report
    except Exception as error:
        report.update(stage='failed', error=repr(error), accepted=False, full_task_qualified=False)
        save()
        raise
    finally:
        save()
        if env is not None:
            # env.close() is a no-op in the installed OG; this process owns its app.
            try:
                env.close()
                og.sim.stop()
                og.sim.close()
                og.cleanup()
            except Exception as error:
                report['cleanup_error'] = repr(error)
                save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.design_file.read_bytes()
    design = validate_design(json.loads(raw))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'result.json').exists() or (args.output_dir / 'input_design.json').exists():
        raise ValueError('Use a fresh output directory; existing diagnostic evidence is preserved.')
    (args.output_dir / 'input_design.json').write_bytes(raw)
    source = Path(__file__).read_bytes()
    (args.output_dir / 'probe_source.py').write_bytes(source)
    _dump(args.output_dir / 'code_provenance.json', {'script_sha256': hashlib.sha256(source).hexdigest(),
        'script_file': str(Path(__file__).resolve()), 'initialization_geometry_revision': 'pre_setter_cloud_transform_v2'})
    report = run(design, args.output_dir)
    return 0 if report['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Offline glass-box validation; run with the same arguments as run_esi_robot.py.

Source a start script in the retained interactive GPU shell. This private replay
uses exact geometry for validation only; measured model episodes restart physics.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main(argv=None):
    from core.action_compiler import ParameterizedAction
    from wia.native.esi_glass_box import hinge_angle_deg
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    from wia.runtime.commands.run_esi_robot import main as robot_main
    import torch
    import omnigibson.lazy as lazy

    args = sys.argv[1:] if argv is None else list(argv)
    geometry_only = '--geometry-only' in args
    keep_viewer = '--keep-viewer' in args
    args = [argument for argument in args if argument not in ('--geometry-only', '--keep-viewer')]
    trunk_rise = 0.
    if '--trunk-rise-m' in args:
        index = args.index('--trunk-rise-m')
        trunk_rise = float(args[index + 1])
        del args[index:index + 2]
    contact_replay = None
    if '--contact-replay-run' in args:
        index = args.index('--contact-replay-run')
        contact_replay = Path(args[index + 1])
        del args[index:index + 2]
    output = Path(args[args.index('--output-dir') + 1])
    original_web = web.run_esi_control

    def verify(session, metadata, controller, executor, **kwargs):
        source = session.observation_source
        evaluator = session.task_evaluator
        box, target = evaluator.box, evaluator.target
        hinge = box.joints['lid_hinge']
        report = {'schema': 'private_glass_box_validation_v1', 'model_calls': 0,
                  'private_oracle_replay': True, 'checks': {}, 'actions': []}
        checks = report['checks']

        def save():
            (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                default=lambda value: _array(value).tolist()) + '\n')

        def snapshot(name):
            session.invalidate_camera_cache()
            source.capture(action_boundary=True).observation.save(output / name)

        def move(action, magnitude=None):
            command = {'action': action}
            if magnitude is not None:
                command['angle_deg' if 'TURN' in action or 'ROTATE' in action else 'distance_m'] = magnitude
            result = executor.execute_macro(ParameterizedAction(**command), resolution=executor.primitive_resolution,
                                            limits=executor.macro_limits)
            report['actions'].append({'command': command, 'result': result,
                                      'door_angle_deg': hinge_angle_deg(box, 'lid_hinge'),
                                      'handle_angle_deg': (hinge_angle_deg(box, 'handle_hinge')
                                                           if 'handle_hinge' in box.joints else None),
                                      'attachment': dict(controller.robot._ag_obj_constraint_params[controller.working_arm])})
            save()
            print('[glass-box]', command, result.get('success'), result.get('reason'), flush=True)
            if not result['success']:
                # Match the existing Agent policy for a measured, stable stop.
                # The planner reads the resulting pose before choosing its
                # next command; the failed receipt is preserved verbatim.
                if result.get('reason') in ('target_not_reached', 'arm_step_limit') and result.get('stop_stability_verified'):
                    report.setdefault('recovered_stable_stops', []).append(len(report['actions']))
                    save()
                    return
                raise RuntimeError('Canonical manipulation stopped: ' + str(result.get('reason')))

        def world_to_base(point):
            pose = controller.pose()
            return Rotation.from_quat(pose['quaternion_xyzw']).inv().apply(np.array(point) - pose['position'])

        def align(point, axes=(0, 1, 2), maximum_step_m=.08, stop_open_angle_deg=None):
            for axis in axes:
                for _ in range(12):
                    if stop_open_angle_deg is not None and hinge_angle_deg(box, 'lid_hinge') >= stop_open_angle_deg:
                        return
                    desired = world_to_base(point)
                    delta = float(desired[axis] - controller.ee_pose()['base_position'][axis])
                    if abs(delta) < .003:
                        break
                    pairs = [('MV_FWD', 'MV_BACK'), ('MV_LEFT', 'MV_RIGHT'), ('MV_UP', 'MV_DOWN')]
                    move(pairs[axis][delta < 0], min(maximum_step_m, abs(delta)))
                else:
                    raise RuntimeError('Private alignment exceeded its bounded action budget.')

        try:
            observation = source.begin_episode(session.public_task_text)
            snapshot('closed_initial')
            from wia.runtime.commands.inspect_esi_handle_rendering import capture_handle_views
            capture_handle_views(session, output / 'closed_initial')
            initial = _array(target.get_position_orientation()[0]).copy()
            for _ in range(15):
                controller.step_base_debug()
            report['apple_settle_displacement_m'] = float(np.linalg.norm(_array(target.get_position_orientation()[0]) - initial))
            checks['apple_stable_inside_closed_box'] = report['apple_settle_displacement_m'] < .005
            checks['closed_hinge'] = abs(hinge_angle_deg(box, 'lid_hinge')) < 3.
            report['glass_materials'] = {mesh.prim.GetName():
                lazy.pxr.UsdShade.MaterialBindingAPI(mesh.prim).GetDirectBinding().GetMaterialPath().pathString
                for link in box.links.values() for mesh in link.visual_meshes.values()
                if mesh.prim.GetName().startswith('glass_')}
            checks['six_native_glass_panels'] = (len(report['glass_materials']) == 6 and
                all('OmniGlass' in path for path in report['glass_materials'].values()))
            report['collision_counts'] = {name: len(link.collision_meshes) for name, link in box.links.items()}
            checks['separate_collidable_base_and_lid'] = report['collision_counts'] in (
                {'base_link': 9, 'lid_link': 6}, {'base_link': 9, 'lid_link': 3, 'handle_link': 3})
            report['hinge_limits_rad'] = [float(hinge.lower_limit), float(hinge.upper_limit)]
            report['hinge_dynamics'] = {'maximum_velocity_rad_s': float(hinge.max_velocity),
                'friction': float(hinge.friction), 'stiffness': float(hinge.stiffness),
                'damping': float(hinge.damping), 'lid_mass_kg': float(box.links['lid_link'].mass)}
            if 'handle_hinge' in box.joints:
                handle_joint = box.joints['handle_hinge']
                report['handle_hinge_dynamics'] = {'driven': bool(handle_joint.driven),
                    'friction': float(handle_joint.friction), 'mass_kg': float(box.links['handle_link'].mass)}
                checks['passive_handle_hinge'] = not handle_joint.driven
            checks['passive_limited_hinge'] = bool(not hinge.driven and float(hinge.upper_limit) >= np.deg2rad(100))
            # Inspect the native stage rather than assuming authored local +Z
            # remains above the lid after articulated-object initialization.
            geometry = {}
            for mesh in (mesh for link in box.links.values() for mesh in link.visual_meshes.values()):
                label = mesh.prim.GetName()
                if label not in ('glass_lid', 'handle_bar', 'handle_left_post', 'handle_right_post'):
                    continue
                low, high = (_array(value) for value in mesh.aabb)
                geometry[label] = {'world_min': low.tolist(), 'world_max': high.tolist()}
            report['initial_lid_geometry'] = geometry
            report['initial_link_poses'] = {name: [_array(value).tolist() for value in
                link.get_position_orientation()] for name, link in box.links.items()}
            gap = geometry['handle_bar']['world_min'][2] - geometry['glass_lid']['world_max'][2]
            report['handle_grip_gap_m'] = float(gap)
            checks['raised_handle_with_grip_clearance'] = gap >= .075
            report['handle_width_m'] = float(geometry['handle_bar']['world_max'][0] -
                                             geometry['handle_bar']['world_min'][0])
            save()
            if contact_replay is not None:
                from wia.runtime.commands.verify_esi_contact_recovery import verify_contact_recovery
                recovery = verify_contact_recovery(session, controller, executor, output, contact_replay)
                report['contact_recovery_verified'] = recovery['verification_success']
                # Keep mechanism validation independent if the stochastic
                # contact replay does not reproduce the same failure category.
            # The initial working hand can intersect the lid's swept volume.
            # Retract it using the normal controller before the isolated probe.
            move('BASE_BACK', .20)
            # Isolated mechanism probe: torque, not pose-setting, must rotate the lid.
            report['torque_probe_effort_nm'] = 1.0
            angles = []
            for _ in range(180):
                hinge.set_effort(torch.tensor([1.0]))
                controller.step_base_debug()
                angles.append(hinge_angle_deg(box, 'lid_hinge'))
                if angles[-1] >= 109.:
                    break
            hinge.set_effort(torch.zeros(1))
            report['torque_probe_angles_deg'] = angles
            checks['door_opens_under_physics'] = max(angles) > 60. and max(angles) < 113.
            snapshot('mechanism_open')
            save()
            if not all(checks.values()):
                raise RuntimeError('Native scene or mechanism checks failed.')
            if geometry_only:
                return
            # Restore the door for a distinct robot-only physical replay.
            hinge.set_pos(torch.zeros(1), drive=False)
            hinge.set_vel(torch.zeros(1), drive=False)
            if 'handle_hinge' in box.joints:
                box.joints['handle_hinge'].set_pos(torch.zeros(1), drive=False)
                box.joints['handle_hinge'].set_vel(torch.zeros(1), drive=False)
            for _ in range(6):
                controller.step_base_debug()
            move('BASE_FWD', .20)
            source.begin_episode(session.public_task_text)
            if trunk_rise:
                move('TRUNK_UP', trunk_rise)
                report['private_trunk_rise_m'] = trunk_rise
            origin = _array(box.get_position_orientation()[0])
            handle = (np.array(geometry['handle_bar']['world_min']) +
                      np.array(geometry['handle_bar']['world_max'])) / 2
            pivot, orientation = box.links['lid_link'].get_position_orientation()
            pivot = _array(pivot)
            lever = handle - pivot
            joint_rotation = hinge.get_attribute('physics:localRot0')
            axis = Rotation.from_quat([*joint_rotation.GetImaginary(), joint_rotation.GetReal()]).apply([1., 0., 0.])
            report['initial_ee_world'] = controller.ee_pose()['world_position']
            move('RELEASE')
            # Approach above the handle before lowering to avoid the closed door.
            safe_z = max(handle[2] + .045, controller.ee_pose()['world_position'][2])
            align([*handle[:2], safe_z], axes=(2,))
            move('BASE_FWD', .05)
            align([*handle[:2], safe_z], axes=(0, 1))
            align(handle, axes=(2,))
            snapshot('handle_grasp_pose')
            move('GRASP')
            attachment = controller.robot._ag_obj_constraint_params[controller.working_arm]
            report['handle_grasp_attachment'] = dict(attachment)
            grasp_link = box.links.get('handle_link', box.links['lid_link'])
            checks['robot_grasps_door'] = attachment.get('ag_link_prim_path') == grasp_link.prim_path
            if not checks['robot_grasps_door']:
                raise RuntimeError('Robot did not attach to the door handle.')
            # The spherical constraint follows its contact point, not the rod
            # centre. Preserve the contact-to-EE offset with fixed EE rotation.
            contact = _array(attachment['contact_pos'])
            lever = contact - pivot
            reference_offset = np.array(controller.ee_pose()['world_position']) - contact
            report['opening_reference_offset_world_m'] = reference_offset.tolist()
            report['opening_maximum_cartesian_step_m'] = .020
            for theta in np.deg2rad([10, 20, 30, 40, 50, 60, 65]):
                point = pivot + Rotation.from_rotvec(axis * theta).apply(lever) + reference_offset
                align(point, axes=(2, 0, 1), maximum_step_m=.020, stop_open_angle_deg=60.)
                if hinge_angle_deg(box, 'lid_hinge') >= 60.:
                    break
            checks['robot_opens_door'] = hinge_angle_deg(box, 'lid_hinge') >= 60.
            move('RELEASE')
            snapshot('robot_door_open')
            apple = _array(target.get_position_orientation()[0]).copy()
            # Clear the highest part of the opened lid before moving toward
            # the apple. Use the existing trunk action to retain arm reach.
            move('TRUNK_UP', .03)
            safe_height = max(float(_array(box.links['lid_link'].aabb[1])[2]) + .04,
                              controller.ee_pose()['world_position'][2])
            align([*apple[:2], safe_height], axes=(2, 0, 1))
            align(apple, axes=(2,))
            move('GRASP')
            checks['robot_grasps_apple'] = controller.debug_gripper_state()['object_name'] == target.name
            for _ in range(3):
                move('MV_UP', .08)
            result = evaluator.record(source.agent_observation(), action=None)
            report['final_evaluation'] = result
            checks['robot_extracts_held_apple'] = result['task_success']
            snapshot('robot_extracted_apple')
        except Exception as error:
            report['error'] = str(error)
        finally:
            hinge.set_effort(torch.zeros(1))
            report['passed'] = all(checks.values()) and 'error' not in report
            save()
            print('[glass-box] validation passed:', report['passed'], flush=True)
            if keep_viewer:
                return original_web(session, metadata, controller, executor, **kwargs)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())

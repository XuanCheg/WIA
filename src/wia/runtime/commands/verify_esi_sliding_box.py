#!/usr/bin/env python3
"""Verify sliding-lid retrieval using normal robot macros and no model calls.

Run in the retained GPU shell with run_esi_robot.py arguments; --keep-viewer
retains the resulting Human-owned scene after the private technical validation.
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
    from core.feedback_semantics import CONTINUABLE_MOTION_ERRORS
    from wia.native.esi_glass_box import sliding_displacement_m
    from wia.runtime.sim.esi_omnigibson import _array
    import wia.runtime.sim.esi_web as web
    import omnigibson.lazy as lazy
    from wia.runtime.commands.inspect_esi_handle_rendering import capture_handle_views
    from wia.runtime.commands.run_esi_robot import main as robot_main

    args = list(sys.argv[1:] if argv is None else argv)
    keep_viewer = '--keep-viewer' in args
    args = [argument for argument in args if argument != '--keep-viewer']
    output = Path(args[args.index('--output-dir') + 1])
    original_web = web.run_esi_control

    def verify(session, metadata, controller, executor, **kwargs):
        source, evaluator = session.observation_source, session.task_evaluator
        box, target = evaluator.box, evaluator.target
        joint = box.joints['lid_slide']
        report = {'model_calls': 0, 'private_geometry_replay': True, 'checks': {}, 'actions': []}
        checks = report['checks']

        def save():
            (output / 'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False,
                default=lambda value: _array(value).tolist()) + '\n')

        def snapshot(name):
            session.invalidate_camera_cache()
            source.capture(action_boundary=True).observation.save(output / name)

        def move(action, distance=None):
            command = {'action': action}
            if distance is not None:
                command['distance_m'] = distance
            before = source.agent_observation()
            result = executor.execute_macro(ParameterizedAction(**command),
                resolution=executor.primitive_resolution, limits=executor.macro_limits)
            after = source.agent_observation()
            report['actions'].append({'command': command, 'result': result,
                'slide_displacement_m': sliding_displacement_m(box, 'lid_slide'),
                'attachment': dict(controller.robot._ag_obj_constraint_params[controller.working_arm]),
                'public_feedback': source.last_action_feedback.to_dict()})
            save()
            if after.action_step - before.action_step != 1 or after.snapshot_id - before.snapshot_id != 1:
                raise RuntimeError('One command must produce one canonical boundary.')
            print('[sliding-box]', command, result['reason'], report['actions'][-1]['slide_displacement_m'], flush=True)
            if not result['success'] and not (
                    result['reason'] in CONTINUABLE_MOTION_ERRORS
                    and source.last_action_feedback.motion_stopped is True):
                raise RuntimeError('Canonical motion stopped: ' + result['reason'])

        def align(point, axes=(0, 1, 2)):
            for axis in axes:
                for _ in range(12):
                    pose = controller.pose()
                    desired = Rotation.from_quat(pose['quaternion_xyzw']).inv().apply(np.array(point) - pose['position'])
                    delta = float(desired[axis] - controller.ee_pose()['base_position'][axis])
                    # Half of the 5 mm action quantum plus readback tolerance.
                    if abs(delta) < .003:
                        break
                    pairs = [('MV_FWD', 'MV_BACK'), ('MV_LEFT', 'MV_RIGHT'), ('MV_UP', 'MV_DOWN')]
                    move(pairs[axis][delta < 0], min(.08, abs(delta)))
                else:
                    raise RuntimeError('Private alignment exhausted its bounded budget.')

        try:
            source.begin_episode(session.public_task_text)
            snapshot('closed_initial')
            capture_handle_views(session, output / 'closed_initial')
            meshes = {mesh.prim.GetName(): mesh for link in box.links.values()
                      for mesh in link.visual_meshes.values()}
            materials = {name: lazy.pxr.UsdShade.MaterialBindingAPI(mesh.prim).GetDirectBinding().GetMaterialPath().pathString
                         for name, mesh in meshes.items() if name.startswith('glass_')}
            report['glass_materials'] = materials
            checks['six_native_glass_panels'] = len(materials) == 6 and all('OmniGlass' in path for path in materials.values())
            report['slide_limits_m'] = [float(joint.lower_limit), float(joint.upper_limit)]
            checks['passive_limited_linear_guide'] = not joint.driven and .299 <= joint.upper_limit <= .301
            checks['initially_closed'] = abs(sliding_displacement_m(box, 'lid_slide')) < .003
            checks['rigid_handle_without_folding_joint'] = set(box.links) == {'base_link', 'lid_link'} and set(box.joints) == {'lid_slide'}
            report['collision_counts'] = {name: len(link.collision_meshes) for name, link in box.links.items()}
            checks['collidable_box_and_lid'] = all(report['collision_counts'].values())
            low, high = (_array(value) for value in meshes['handle_bar'].aabb)
            report['handle_grip_gap_m'] = float(low[2] - _array(meshes['glass_lid'].aabb[1])[2])
            checks['raised_visible_grip_geometry'] = report['handle_grip_gap_m'] >= .075
            handle = (low + high) / 2
            save()
            if not all(checks.values()):
                raise RuntimeError('Native scene geometry checks failed.')
            # An ordinary base translation keeps the complete slide within
            # arm reach. This private plan is not supplied to measured agents.
            move('BASE_FWD', .12)
            move('RELEASE')
            safe_z = max(float(handle[2]) + .04, controller.ee_pose()['world_position'][2])
            align([*handle[:2], safe_z], axes=(2, 0, 1))
            align(handle, axes=(2,))
            snapshot('handle_grasp_pose')
            move('GRASP')
            attachment = controller.robot._ag_obj_constraint_params[controller.working_arm]
            checks['robot_grasps_sliding_lid'] = attachment.get('ag_link_prim_path') == box.links['lid_link'].prim_path
            if not checks['robot_grasps_sliding_lid']:
                raise RuntimeError('Robot did not acquire the rigid lid handle.')
            for _ in range(6):
                displacement = sliding_displacement_m(box, 'lid_slide')
                remaining = .280 - displacement
                if remaining < .002:
                    break
                move('MV_FWD', min(.07, remaining))
            checks['robot_opens_lid'] = sliding_displacement_m(box, 'lid_slide') >= .275
            if not checks['robot_opens_lid']:
                raise RuntimeError('Normal robot commands did not open the lid.')
            move('RELEASE')
            snapshot('robot_door_open')
            apple = _array(target.get_position_orientation()[0]).copy()
            align([*apple[:2], safe_z], axes=(2, 0, 1))
            align(apple, axes=(2,))
            snapshot('apple_grasp_pose')
            for _ in range(3):
                move('GRASP')
                if controller.debug_gripper_state()['object_name'] == target.name:
                    break
            checks['robot_grasps_apple'] = controller.debug_gripper_state()['object_name'] == target.name
            if not checks['robot_grasps_apple']:
                raise RuntimeError('Robot did not acquire the apple.')
            for _ in range(3):
                move('MV_UP', .075)
            result = evaluator.record(source.agent_observation(), action=None)
            report['final_evaluation'] = result
            checks['robot_extracts_held_apple'] = result['task_success']
            snapshot('robot_extracted_apple')
        except Exception as error:
            report['error'] = str(error)
        finally:
            report['passed'] = all(checks.values()) and 'error' not in report
            save()
            print('[sliding-box] validation passed:', report['passed'], flush=True)
            if keep_viewer:
                return original_web(session, metadata, controller, executor, **kwargs)

    web.run_esi_control = verify
    return robot_main(args)


if __name__ == '__main__':
    raise SystemExit(main())

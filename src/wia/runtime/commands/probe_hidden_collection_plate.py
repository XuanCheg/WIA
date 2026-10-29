"""Probe native right-hand plate initialization in a bounded empty apparatus.

Run with an explicit GPU UUID in the behavior environment:
python -B src/wia/runtime/commands/probe_hidden_collection_plate.py --model MODEL --output DIR
This diagnostic positions apparatus only before the episode and never creates
manual grasp constraints. It preserves the source dataset and robot dimensions.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
from pathlib import Path
import sys
from copy import deepcopy
from types import SimpleNamespace

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="tray")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rim", choices=("x", "y"), default="y")
    parser.add_argument("--penetration", type=float, default=.025)
    parser.add_argument("--pickup", action="store_true")
    parser.add_argument("--lower-plate", action="store_true")
    parser.add_argument("--grasp-height", type=float, default=.88)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
    from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
    gpu = allocation_gpu(0, execution_mode="local", renderer_order="pci")
    install_gpu_selection(gpu)
    from wia.runtime.sim.esi_omnigibson import _import_pipeline, DEFAULT_ESI_ROOT, _array
    pipeline = _import_pipeline(DEFAULT_ESI_ROOT)
    og = pipeline.og
    from omnigibson import simulator
    isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    import numpy as np
    import torch
    import yaml
    from scipy.spatial.transform import Rotation
    from wia.runtime.sim.esi_base import R1BaseController
    from wia.runtime.sim.esi_gripper import R1ManipulationController
    from core.record.images import save_png
    config = yaml.safe_load((Path(og.example_config_path) / "r1_primitives.yaml").read_text())
    config["scene"] = {"type": "Scene"}
    config["robots"][0].update(name="robot", position=[0., 0., .02],
        orientation=[0., 0., 0., 1.], grasping_mode="assisted", disable_grasp_handling=False)
    config["objects"] = [
        {"type": "DatasetObject", "name": "collection_plate", "category": args.category,
         "model": args.model, "position": [2., 0., 1.], "fixed_base": False, "visual_only": False},
        {"type": "PrimitiveObject", "name": "initial_plate_support", "primitive_type": "Cube",
         "size": 1., "scale": [.25, .36, .04], "position": [2., 0., .8],
         "fixed_base": True, "visual_only": False},
    ]
    if args.pickup:
        config["objects"].extend([
            {"type": "DatasetObject", "name": "pickup_hamburger", "category": "hamburger",
             "model": "fyswch", "position": [3., 0., 1.], "fixed_base": False, "visual_only": False},
            {"type": "PrimitiveObject", "name": "pickup_table", "primitive_type": "Cube",
             "scale": [.35, .35, .05], "position": [3., 0., .9],
             "fixed_base": True, "visual_only": False},
        ])
    report = {"gpu": gpu, "stage": "constructing", "accepted": False, "grasp_mode": "assisted",
        "manual_attachment": False, "parameters": vars(args) | {"output": str(args.output)}}

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        env = og.Environment(configs=config)
        robot = env.robots[0]
        session = SimpleNamespace(env=env, robot=robot, pipeline=pipeline)
        session.step_control = env.step
        plate = env.scene.object_registry("name", "collection_plate")
        stand = env.scene.object_registry("name", "initial_plate_support")
        q = _array(robot.get_joint_positions()).copy()
        trunk = _array(robot.controllers["trunk"].dof_idx).astype(int)
        q[trunk] = json.loads((ROOT / "configs/esi_arm.json").read_text())["initial_trunk_joint_positions_rad"]
        robot.reset_joint_pos = torch.tensor(q, dtype=torch.float32)
        robot.reset()
        if args.lower_plate:
            original_config = deepcopy(robot._controller_config)
            setup_config = deepcopy(original_config)
            setup_config["arm_right"] = deepcopy(robot._default_arm_ik_controller_configs["right"])
            setup_config["arm_right"].update(mode="absolute_pose", command_input_limits=None,
                command_output_limits=None, smoothing_filter_size=None, use_impedances=False)
            robot.reload_controllers(setup_config)
            control = robot.get_control_dict()
            action = np.empty(robot.action_dim, dtype=np.float32)
            for name, ctrl in robot.controllers.items():
                action[_array(robot.controller_action_idx[name]).astype(int)] = _array(
                    ctrl.compute_no_op_action(control))
            start = _array(control["eef_right_pos_relative"])
            quaternion = _array(control["eef_right_quat_relative"])
            idx = _array(robot.controller_action_idx["arm_right"]).astype(int)
            target = np.array([.4, -.25, .8])
            report["stage"] = "initializing_lower_right_arm"
            save()
            for fraction in np.linspace(0., 1., 180):
                action[idx] = np.r_[start + fraction * (target - start),
                    Rotation.from_quat(quaternion).as_rotvec()]
                env.step({robot.name: action})
            for _ in range(20):
                env.step({robot.name: action})
            report["lower_right_arm_pose"] = _array(
                robot.get_control_dict()["eef_right_pos_relative"]).tolist()
            robot.reload_controllers(original_config)
        controller = R1ManipulationController(session)
        hold = controller
        for _ in range(30):
            hold.step_base_debug()
        right_pos, right_quat = (_array(v) for v in robot.get_eef_pose("right"))
        links = robot.finger_links["right"]
        fingers = [(_array(v.aabb[0]) + _array(v.aabb[1])) / 2 for v in links]
        tip_length = float(np.mean(list(robot.eef_to_fingertip_lengths["right"].values())))
        pinch = right_pos + Rotation.from_quat(right_quat).apply([0., 0., tip_length - args.penetration])
        original_position = _array(plate.get_position_orientation()[0])
        low, high = (_array(v) - original_position for v in plate.aabb)
        axis = 0 if args.rim == "x" else 1
        rim = (low + high) / 2
        rim[axis] = low[axis] + .008
        rim[2] = high[2] - .008
        target = pinch - rim
        plate.set_position_orientation(position=target.tolist(), orientation=[0., 0., 0., 1.])
        plate.keep_still()
        support_centre = target + (low + high) / 2
        support_centre[2] = target[2] + low[2] - .02
        stand.set_position_orientation(position=support_centre.tolist())
        grip = robot.controllers["gripper_right"]
        indices = _array(grip.dof_idx).astype(int)
        lower = _array(robot.control_limits["position"][0])[indices]
        report.update(stage="closing_right_gripper", initial_right_eef=[right_pos.tolist(), right_quat.tolist()],
            finger_centres=[v.tolist() for v in fingers], pinch=pinch.tolist(),
            finger_joint_limits=[_array(v)[indices].tolist() for v in robot.control_limits["position"]],
            fingertip_lengths={k: float(v) for k, v in robot.eef_to_fingertip_lengths["right"].items()},
            plate_local_bounds=[low.tolist(), high.tolist()], initial_plate_position=target.tolist(),
            robot_joints=_array(robot.get_joint_positions()).tolist(),
            right_arm_dofs=_array(robot.controllers["arm_right"].dof_idx).astype(int).tolist(),
            closure=[])
        save()
        for step in range(120):
            now = _array(robot.get_joint_positions())[indices]
            hold.hold_action[hold.indices["gripper_right"]] = np.maximum(lower, now - .001)
            hold.step_base_debug()
            held = getattr(robot._ag_obj_in_hand["right"], "name", None)
            if step % 5 == 0:
                report["closure"].append({"step": step, "holding": held,
                    "finger_positions": now.tolist(), "plate_position": _array(plate.get_position_orientation()[0]).tolist()})
                save()
            if held == plate.name and int(robot.is_grasping("right")) == 1:
                report["native_grasp_acquired"] = True
                break
        report["native_grasp_acquired"] = report.get("native_grasp_acquired", False)
        if report["native_grasp_acquired"]:
            # Initialization-only apparatus removal, after contact-qualified grasp.
            env.scene.remove_object(stand)
            for _ in range(90):
                hold.step_base_debug()
            holding = getattr(robot._ag_obj_in_hand["right"], "name", None) == plate.name
            up = float(Rotation.from_quat(_array(plate.get_position_orientation()[1])).apply([0., 0., 1.])[2])
            report.update(retained_without_support=holding, plate_up_dot=up,
                accepted=bool(holding and up > np.cos(np.deg2rad(10.))))
            # Reconstruct exactly as the episode executor will after initialization.
            controller = R1ManipulationController(session)
            report["controller_reconstruction_retained_grasp"] = (
                getattr(robot._ag_obj_in_hand["right"], "name", None) == plate.name)
            report["carry_commands"] = []
            for token, magnitude in (("BASE_FWD", .2), ("BASE_LEFT", .2),
                                     ("BASE_TURN_LEFT", 20.)):
                command = controller.execute_metric(token, magnitude)
                report["carry_commands"].append(command)
                save()
                if not command["completed"]:
                    break
            report["after_carry_holding"] = (
                getattr(robot._ag_obj_in_hand["right"], "name", None) == plate.name)
            report["after_carry_up_dot"] = float(Rotation.from_quat(
                _array(plate.get_position_orientation()[1])).apply([0., 0., 1.])[2])
            if args.pickup:
                burger = env.scene.object_registry("name", "pickup_hamburger")
                table = env.scene.object_registry("name", "pickup_table")
                base_pos, base_quat = (_array(v) for v in robot.get_position_orientation())
                base_rotation = Rotation.from_quat(base_quat)
                surface = base_pos + base_rotation.apply([.55, .46, .85])
                offset = _array(burger.get_position_orientation()[0])[2] - _array(burger.aabb[0])[2]
                table.set_position_orientation(position=(surface - [0., 0., .025]).tolist(),
                    orientation=base_quat.tolist())
                burger.set_position_orientation(position=(surface + [0., 0., offset + .002]).tolist(),
                    orientation=base_quat.tolist())
                burger.keep_still()
                for _ in range(15):
                    controller.step_base_debug()
                report["pickup_initialization"] = {
                    "phase": "diagnostic_initialization_before_public_pickup",
                    "burger_pose": [_array(v).tolist() for v in burger.get_position_orientation()],
                    "surface": surface.tolist()}
                report["pickup_commands"] = []
                report["pickup_commands"].append(controller.set_gripper_debug(False))
                # Test an unobstructed vertical approach at native asset scale.
                goals = [[.52, .18, .98], [.55, .46, .98], [.55, .46, args.grasp_height]]
                for goal in goals:
                    for _ in range(30):
                        current = np.array(controller.ee_pose()["base_position"])
                        delta = np.array(goal) - current
                        if np.linalg.norm(delta) < .004:
                            break
                        delta *= min(1., .04 / np.linalg.norm(delta))
                        command = controller.move_ee_debug(*delta)
                        report["pickup_commands"].append(command)
                        save()
                        if not command["completed"]:
                            break
                    if not report["pickup_commands"][-1]["completed"]:
                        break
                # The installed native assisted-grasp window is longer than one
                # public gripper motor macro. Repeat legal GRASP holds; never
                # bypass its contact/raycast qualification or create a joint.
                for _ in range(8):
                    report["pickup_commands"].append(controller.set_gripper_debug(True))
                    save()
                    if getattr(robot._ag_obj_in_hand["left"], "name", None) == burger.name:
                        break
                report["left_holds_hamburger"] = (
                    getattr(robot._ag_obj_in_hand["left"], "name", None) == burger.name)
                if report["left_holds_hamburger"]:
                    report["pickup_commands"].append(controller.move_ee_debug(0., 0., .08))
                    report["deposit_commands"] = []
                    plate_pos, plate_quat = (_array(v) for v in plate.get_position_orientation())
                    eef_world = _array(robot.get_eef_pose("left")[0])
                    bottom_offset = eef_world[2] - _array(burger.aabb[0])[2]
                    desired_world = plate_pos + Rotation.from_quat(plate_quat).apply([0., 0., .007])
                    desired_world[2] += bottom_offset + .008
                    base_pos, base_quat = (_array(v) for v in robot.get_position_orientation())
                    desired_base = Rotation.from_quat(base_quat).inv().apply(desired_world - base_pos)
                    goals = [[*desired_base[:2], 1.0], desired_base.tolist()]
                    for goal in goals:
                        for _ in range(30):
                            delta = np.array(goal) - np.array(controller.ee_pose()["base_position"])
                            if np.linalg.norm(delta) < .004:
                                break
                            delta *= min(1., .04 / np.linalg.norm(delta))
                            command = controller.move_ee_debug(*delta)
                            report["deposit_commands"].append(command)
                            save()
                            if not command["completed"]:
                                break
                        if report["deposit_commands"] and not report["deposit_commands"][-1]["completed"]:
                            break
                    report["deposit_commands"].append(controller.set_gripper_debug(False))
                    for _ in range(45):
                        controller.step_base_debug()
                    report["burger_contacts"] = [
                        {"body0": str(c.body0), "body1": str(c.body1)} for c in burger.contact_list()]
                    report["deposited_on_plate"] = bool(
                        getattr(robot._ag_obj_in_hand["left"], "name", None) is None
                        and any(plate.prim_path in str(getattr(c, side)) for c in burger.contact_list()
                                for side in ("body0", "body1")))
                report["burger_final_pose"] = [_array(v).tolist() for v in burger.get_position_orientation()]
                save()
        report["final_plate_pose"] = [_array(v).tolist() for v in plate.get_position_orientation()]
        report["right_contacts"] = [{"body0": str(c.body0), "body1": str(c.body1)} for c in robot.contact_list()]
        camera = og.sim.viewer_camera
        eye, centre = np.array([1.7, -1.7, 1.7]), np.array([.3, 0., .8])
        z = eye - centre
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        camera.set_position_orientation(position=eye.tolist(),
            orientation=Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])).as_quat().tolist())
        for _ in range(8):
            og.sim.render()
        save_png(args.output / "apparatus.png", _array(camera.get_obs()[0]["rgb"]))
        report["stage"] = "complete"
    except Exception as error:
        report.update(stage="failed", error=repr(error))
        raise
    finally:
        save()
        og.shutdown()


if __name__ == "__main__":
    main()

"""Initialize a contact-qualified right-hand tray before an episode.

Run native diagnostics with src/wia/runtime/commands/probe_hidden_collection_plate.py.
This module is called only by an explicit collection scene initializer.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation

from wia.runtime.sim.esi_omnigibson import _array
from wia.runtime.sim.esi_base import R1BaseController


def initialize_collection_tray(session, parameters):
    if not getattr(session, "_category_initialization_active", False):
        raise RuntimeError("Tray setup is restricted to explicit scene initialization.")
    robot, env = session.robot, session.env
    plate_spec = parameters["plate"]
    if plate_spec.get("pinch_reference", "eef_fingertip") != "eef_fingertip":
        raise ValueError("Experimental tray ray anchors were not retained; reproduce those failures with their frozen capture code.")
    plate = env.scene.object_registry("name", plate_spec["name"])
    stand = env.scene.object_registry("name", "collection_initial_support")
    if plate is None or stand is None or plate.fixed_base or plate.visual_only:
        raise ValueError("Tray initialization requires its movable collidable apparatus.")
    if robot.grasping_mode != "assisted":
        raise ValueError("Collection uses native contact-qualified assisted grasp.")
    if not np.array_equal(_array(robot.controllers["trunk"].dof_idx), [6, 7, 8, 9]):
        raise ValueError("Installed R1 trunk ordering differs from the declared reset.")
    original = deepcopy(robot._controller_config)
    setup = deepcopy(original)
    setup["arm_right"] = deepcopy(robot._default_arm_ik_controller_configs["right"])
    setup["arm_right"].update(mode="absolute_pose", command_input_limits=None,
        command_output_limits=None, smoothing_filter_size=None, use_impedances=False)
    robot.reload_controllers(setup)
    control = robot.get_control_dict()
    action = np.empty(robot.action_dim, dtype=np.float32)
    for name, controller in robot.controllers.items():
        action[_array(robot.controller_action_idx[name]).astype(int)] = _array(
            controller.compute_no_op_action(control))
    start = _array(control["eef_right_pos_relative"])
    orientation = Rotation.from_quat(_array(control["eef_right_quat_relative"]))
    target = np.asarray(plate_spec["right_eef_position_base_m"], dtype=float)
    indices = _array(robot.controller_action_idx["arm_right"]).astype(int)
    step_control = getattr(session, "step_control", env.step)
    try:
        for fraction in np.linspace(0., 1., 180):
            action[indices] = np.r_[start + fraction * (target - start), orientation.as_rotvec()]
            step_control({robot.name: action})
        for _ in range(20):
            step_control({robot.name: action})
        residual = float(np.linalg.norm(
            _array(robot.get_control_dict()["eef_right_pos_relative"]) - target))
    finally:
        robot.reload_controllers(original)
    if residual > .005:
        raise ValueError("Right-hand initialization did not reach the declared pose.")
    hold = R1BaseController(session)
    for _ in range(30):
        hold.step_base_debug()
    eef_pos, eef_quat = (_array(v) for v in robot.get_eef_pose("right"))
    tip = np.mean([float(v) for v in robot.eef_to_fingertip_lengths["right"].values()])
    pinch = eef_pos + Rotation.from_quat(eef_quat).apply([0., 0., tip - plate_spec["rim_penetration_m"]])
    # Establish horizontal native geometry in the robot's initial yaw frame.
    _, base_quat = (_array(v) for v in robot.get_position_orientation())
    yaw = Rotation.from_quat(base_quat).as_euler("xyz")[2]
    tray_rotation = Rotation.from_euler("z", yaw)
    # Measure before teleportation: native AABB caches still describe the last
    # physics sample immediately after set_position_orientation.
    origin, asset_quaternion = (_array(v) for v in plate.get_position_orientation())
    points = np.concatenate([_array(link.collision_boundary_points_world)
        for link in plate.links.values() if link.collision_boundary_points_world is not None])
    local = Rotation.from_quat(asset_quaternion).inv().apply(points - origin)
    low, high = local.min(0), local.max(0)
    rim = (low + high) / 2
    axis = 0 if plate_spec["rim_axis"] == "x" else 1
    rim[axis] = low[axis] + .008
    rim[2] = high[2] - .008
    position = pinch - tray_rotation.apply(rim)
    plate.set_position_orientation(position=position.tolist(),
        orientation=tray_rotation.as_quat().tolist())
    plate.keep_still()
    stand_position = position + tray_rotation.apply((low + high) / 2)
    stand_position[2] = position[2] + low[2] - .02
    stand.set_position_orientation(position=stand_position.tolist(),
        orientation=tray_rotation.as_quat().tolist())
    receipt = {"accepted": False, "stage": "closing_right_gripper",
        "initial_right_eef": [eef_pos.tolist(), eef_quat.tolist()],
        "pinch": pinch.tolist(), "local_asset_bounds": [low.tolist(), high.tolist()],
        "plate_start_position": position.tolist(), "stand_position": stand_position.tolist(),
        "right_eef_residual_m": residual, "closure": []}
    session.hidden_collection_initialization = receipt
    grip_indices = _array(robot.controllers["gripper_right"].dof_idx).astype(int)
    minimum = _array(robot.control_limits["position"][0])[grip_indices]
    for step in range(120):
        hold.hold_action[hold.indices["gripper_right"]] = np.maximum(
            minimum, _array(robot.get_joint_positions())[grip_indices] - .001)
        hold.step_base_debug()
        if step % 5 == 0:
            receipt["closure"].append({"step": step,
                "plate_pose": [_array(v).tolist() for v in plate.get_position_orientation()],
                "finger_positions": _array(robot.get_joint_positions())[grip_indices].tolist(),
                "holding": getattr(robot._ag_obj_in_hand["right"], "name", None)})
        if getattr(robot._ag_obj_in_hand["right"], "name", None) == plate.name:
            break
    if getattr(robot._ag_obj_in_hand["right"], "name", None) != plate.name:
        receipt["contacts"] = [{"body0": str(c.body0), "body1": str(c.body1)}
                               for c in robot.contact_list()]
        raise ValueError("Contact-qualified right-hand tray grasp was not established.")
    env.scene.remove_object(stand)
    for _ in range(90):
        hold.step_base_debug()
    quaternion = _array(plate.get_position_orientation()[1])
    up = float(Rotation.from_quat(quaternion).apply([0., 0., 1.])[2])
    retained = (getattr(robot._ag_obj_in_hand["right"], "name", None) == plate.name
                and int(robot.is_grasping("right")) == 1)
    accepted = bool(retained and up >= np.cos(np.deg2rad(parameters["maximum_plate_tilt_deg"])))
    receipt.update({"accepted": accepted, "stage": "complete", "grasp_mode": "assisted", "manual_attachment": False,
        "right_eef_residual_m": residual, "closure_steps": step + 1,
        "retained_without_support": retained, "plate_up_dot": up,
        "plate_pose": [_array(v).tolist() for v in plate.get_position_orientation()],
        "native_layout_qualified": False})
    session.hidden_collection_initialization = receipt
    if not accepted:
        raise ValueError("Initialized tray did not remain grasped and level after support removal.")
    return receipt

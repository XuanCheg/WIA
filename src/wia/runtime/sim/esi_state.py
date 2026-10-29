"""Robot control state and explicit real-sensor adapters; no attachment projection."""
from __future__ import annotations

import numpy as np

from core.observations import ControllerState, RobotSensorFeedback
from wia.runtime.sim.esi_omnigibson import _array


def read_controller_state(session, *, controller=None, working_arm="left") -> ControllerState:
    from wia.runtime.sim.esi_base import _main_thread

    _main_thread()
    robot = session.robot
    if robot is None or getattr(session, "closed", False):
        raise RuntimeError("Robot readback requires an initialized open session.")

    def optional(name):
        getter = getattr(robot, name, None)
        if getter is None:
            return None
        value = getter()
        if value is None:
            return None
        value = _array(value)
        if not np.isfinite(value).all():
            raise RuntimeError(f"Nonfinite robot readback: {name}.")
        return value.tolist()

    if controller is not None:
        base = dict(controller.pose())
    else:
        position, quaternion = (_array(value) for value in robot.get_position_orientation())
        if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
                or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0):
            raise RuntimeError("Invalid native base pose.")
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        base = {"position": position.tolist(), "quaternion_xyzw": quaternion.tolist(),
                "yaw_rad": float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))}
    base["source"] = "simulator_ground_truth"
    positions = optional("get_joint_positions")
    ee_reader = getattr(controller, "ee_pose", None)
    ee = ee_reader() if ee_reader is not None else None
    control = robot.get_control_dict()
    position_key, quaternion_key = f"eef_{working_arm}_pos_relative", f"eef_{working_arm}_quat_relative"
    # The installed SDK returns CachedFunctions, whose __getitem__ is a lazy
    # named lookup, not Python's integer-indexed sequence/membership protocol.
    control_names = control.get_fcn_names() if hasattr(control, 'get_fcn_names') else control.keys()
    if ee is None and position_key in control_names and quaternion_key in control_names:
        world_position, world_quaternion = robot.get_eef_pose(working_arm)
        ee = {"base_position": _array(control[position_key]).tolist(),
              "base_quaternion_xyzw": _array(control[quaternion_key]).tolist(),
              "world_position": _array(world_position).tolist(),
              "world_quaternion_xyzw": _array(world_quaternion).tolist()}

    provider = getattr(session, "robot_sensor_feedback_provider", None)
    readings = provider() if provider is not None else RobotSensorFeedback()
    if type(readings) is not RobotSensorFeedback:
        raise TypeError("Real-sensor adapters must return RobotSensorFeedback.")
    grip_controller = robot.controllers.get(f"gripper_{working_arm}")
    fingers = getattr(robot, "finger_joint_names", {}).get(working_arm, ())
    width = closed = opened = None
    limits = getattr(robot, "control_limits", {})
    close_target = open_target = None
    if positions is not None and grip_controller is not None and len(fingers) == 2 and all(
            "prismatic" in str(robot.joints[name].joint_type).lower() for name in fingers):
        indices = _array(grip_controller.dof_idx).astype(int)
        measured = np.array(positions)[indices]
        width = float(np.sum(measured))
        if "position" in limits:
            close_target, open_target = [_array(value)[indices] for value in limits["position"]]
            tolerance = getattr(getattr(controller, "gripper_config", None), "joint_tolerance_m", 0.001)
            closed = bool(np.max(np.abs(measured-close_target)) <= tolerance)
            opened = bool(np.max(np.abs(measured-open_target)) <= tolerance)
    grip = {"gripper_width_m": width, "gripper_command_name": getattr(controller, "gripper_command", "HOLD"),
            "gripper_closed": closed, "gripper_open": opened,
            "gripper_joint_positions_m": measured.tolist() if width is not None else None}
    proprio_reader = getattr(controller, "proprioception", None)
    proprio = proprio_reader() if proprio_reader is not None else {}
    # The legacy controller dictionary includes assisted truth. Keep only named
    # robot quantities for internal diagnostics, then overwrite finger flags.
    proprio = {key: proprio[key] for key in ("frame", "source", "working_arm", "eef_pos", "eef_quaternion_xyzw",
                "eef_height_base_m", "arm_joint_positions_rad", "trunk_joint_positions_rad", "trunk_height_m",
                "trunk_control_height_m") if key in proprio}
    proprio.update(grip, frame="robot_base", source="joint_encoders_forward_kinematics", working_arm=working_arm,
                   eef_pos=ee["base_position"] if ee else None,
                   eef_quaternion_xyzw=ee["base_quaternion_xyzw"] if ee else None)
    last = getattr(controller, "_last_control_result", {}) or {}
    active = getattr(controller, "_active_control", {}) or {}
    status = active.get("execution_status", "idle")
    arm_target = getattr(controller, "_arm_target_pose", None)
    hold_action = getattr(controller, "hold_action", None)
    gripper_target = (hold_action[getattr(controller, "indices", {})[f"gripper_{working_arm}"]].tolist()
                      if hold_action is not None and f"gripper_{working_arm}" in controller.indices else None)
    diagnostic = getattr(controller, "ik_guard", None)
    return ControllerState(
        base_pose=base,
        base_velocity={"frame": "world", "source": "simulator_ground_truth", "linear_m_s": optional("get_linear_velocity"),
                       "angular_rad_s": optional("get_angular_velocity")}, ee_pose=ee,
        joint_state={"positions": positions, "velocities": optional("get_joint_velocities"),
                     "position_limits": [_array(value).tolist() for value in limits.get("position", ())],
                     "velocity_limits": [_array(value).tolist() for value in limits.get("velocity", ())]},
        gripper_state=grip,
        controller_status={"hold_action": hold_action.tolist() if hold_action is not None else None,
                           "working_arm": working_arm, "execution_status": status,
                           "action": active.get("action"),
                           "joint_hold": getattr(diagnostic, "hold", None) is not None},
        proprioception=proprio, sensor_feedback=readings,
        trunk_control=vars(controller.trunk_controller.config) if getattr(controller, "trunk_controller", None) is not None else {},
        base_control={"initial_pose": active.get("base_initial_pose", last.get("pose_before", last.get("base_pose_before"))),
                      "target_pose": active.get("base_target_pose"), "control_error": active.get("base_error"),
                      "execution_status": status, "feedback_source": "ideal_sim_controller"},
        arm_control={"initial_pose": active.get("arm_initial_pose", last.get("ee_pose_before")),
                     "target_pose": arm_target,
                     "translation_tolerance_m": getattr(getattr(controller, "arm_config", None), "translation_tolerance_m", None),
                     "orientation_tolerance_deg": getattr(getattr(controller, "arm_config", None), "orientation_tolerance_deg", None),
                     "rotation_tolerance_deg": getattr(getattr(controller, "rotation_config", None), "rotation_tolerance_deg", None),
                     "ik_state": diagnostic.diagnostics() if diagnostic is not None else None,
                     "execution_status": status},
        gripper_control={"requested_command": grip["gripper_command_name"], "target_joint_positions_m": gripper_target,
                         "close_joint_positions_m": close_target.tolist() if close_target is not None else None,
                         "open_joint_positions_m": open_target.tolist() if open_target is not None else None,
                         "measured_width_m": width, "force_feedback": readings.sensors.force_torque,
                         "execution_status": status},
    )

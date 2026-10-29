"""Calibrate fixed robot camera mounts, then use only measured joint FK per snapshot."""
from __future__ import annotations

import numpy as np

from core.camera_geometry import CameraGeometry, RobotKinematics, pose_matrix
from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import NativeCameraNotReady, _array


def native_intrinsics_when_ready(session, sensor):
    """Wait for actual camera parameters with pure renders; never invent calibration."""
    _main_thread()
    for attempt in range(8):
        k = _array(sensor.intrinsic_matrix).astype(float)
        valid = (k.shape == (3, 3) and np.isfinite(k).all() and k[0, 0] > 0
                 and k[1, 1] > 0 and np.allclose(k[2], [0, 0, 1]))
        if valid:
            return k
        print(f"[camera-geometry] native intrinsics not ready: attempt={attempt + 1} "
              f"sensor={sensor.name} values={k.tolist()}", flush=True)
        if attempt < 7:
            invalidate = getattr(session, "invalidate_camera_cache", None)
            if invalidate:
                invalidate()
            session.pipeline.og.sim.render()
    raise NativeCameraNotReady("Native camera intrinsics remained invalid after eight reads.")


def calibrated_robot_kinematics(robot, *, joint_names=None):
    """Robot-only FK using installed joint mounts and measured encoder axes."""
    kinematics = RobotKinematics(robot.urdf_path, robot.base_footprint_link_name)
    paths = {link.prim_path: name for name, link in robot.links.items()}
    urdf_joints = {item[0] for items in kinematics.children.values() for item in items}
    selected = urdf_joints if joint_names is None else set(joint_names)
    calibration = {}
    for name, joint in robot.joints.items():
        if name not in urdf_joints or name not in selected:
            continue
        parent, child = paths[joint.body0], paths[joint.body1]
        parent_joint = pose_matrix(_array(joint.local_position_0), _array(joint.local_orientation_0))
        child_joint = pose_matrix(_array(joint.local_position_1), _array(joint.local_orientation_1))
        axis = {"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}[joint.axis]
        kinematics.calibrate_joint(name, parent=parent, child=child,
            parent_joint=parent_joint, child_joint=child_joint, axis=axis)
        calibration[name] = {"parent": parent, "child": child,
            "parent_joint": parent_joint.tolist(), "child_joint": child_joint.tolist(), "axis": axis}
    if joint_names is not None and set(calibration) != selected:
        raise RuntimeError("Installed joint mounts are required for calibrated trunk height.")
    scale = _array(getattr(robot, 'scale', [1., 1., 1.])).astype(float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or not np.allclose(scale, scale[0]) or scale[0] <= 0:
        raise ValueError('Encoder camera geometry requires a positive uniform robot scale.')
    # USD joint-local offsets and URDF fixed offsets are expressed in unscaled
    # link coordinates. Measured prismatic encoders remain physical metres.
    for children in kinematics.children.values():
        for _, _, _, origin, _, post in children:
            origin[:3, 3] *= scale[0]
            post[:3, 3] *= scale[0]
    return kinematics, calibration


class ESICameraGeometry:
    def __init__(self, session, working_arm="left"):
        _main_thread()
        self.session, self.robot, self.working_arm = session, session.robot, working_arm
        self.kinematics, self.joint_calibration = calibrated_robot_kinematics(self.robot)
        known = {self.kinematics.root, *self.kinematics.parents}
        # Installed R1 USD adds fixed eyes/eef links absent from its URDF and
        # omits steerable wheel joints. Resolve only required encoder chains;
        # calibrate extra rigid attachments only after proving a FixedJoint path.
        self.fixed_graph = {}
        from pxr import UsdPhysics, Usd
        paths = {link.prim_path: name for name, link in self.robot.links.items()}
        for prim in Usd.PrimRange(self.robot.prim):
            if prim.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.FixedJoint(prim)
                pairs = (joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets())
                if all(len(x) == 1 and str(x[0]) in paths for x in pairs):
                    left, right = (paths[str(x[0])] for x in pairs)
                    self.fixed_graph.setdefault(left, set()).add(right)
                    self.fixed_graph.setdefault(right, set()).add(left)
        self.attachments = {}

        def attachment(name):
            if name not in self.attachments:
                anchor = name
                if name not in known:
                    pending, seen = [name], {name}
                    anchor = None
                    while pending and anchor is None:
                        for other in self.fixed_graph.get(pending.pop(0), ()):
                            if other in known:
                                anchor = other
                                break
                            if other not in seen:
                                seen.add(other)
                                pending.append(other)
                    if anchor is None:
                        raise RuntimeError(f"No verified rigid URDF attachment for robot link {name}.")
                a, b = (pose_matrix(*[_array(x) for x in self.robot.links[n].get_position_orientation()])
                        for n in (anchor, name))
                self.attachments[name] = (anchor, np.linalg.solve(a, b))
            return self.attachments[name]
        registry = getattr(session, "robot_camera_registry", None) or {
            "agentview_rgb": session.camera_name, "wrist_rgb": session.wrist_camera_name}
        self.mounts = {}
        for name in ("agentview_rgb", "wrist_rgb"):
            sensor = self.robot.sensors[registry[name]]
            candidates = [(link_name, link) for link_name, link in self.robot.links.items()
                          if sensor.prim_path.startswith(link.prim_path + "/")]
            if not candidates:
                raise RuntimeError("A model camera must have a physical robot-link mount.")
            link_name, link = max(candidates, key=lambda pair: len(pair[1].prim_path))
            attachment(link_name)
            # Establish only the constant mechanical mounting transform. World
            # quantities are discarded here and never enter the public payload.
            world_link = pose_matrix(*[_array(x) for x in link.get_position_orientation()])
            world_camera = pose_matrix(*[_array(x) for x in sensor.get_position_orientation()])
            optical = np.diag([1., -1., -1., 1.])  # USD camera -Z/+Y -> optical +Z/down.
            link_camera = np.linalg.solve(world_link, world_camera) @ optical
            k = native_intrinsics_when_ready(session, sensor)
            self.mounts[name] = (link_name, link_camera, k, (int(sensor.image_width), int(sensor.image_height)))
        self.grasp_points = []
        for points in (self.robot.assisted_grasp_start_points, self.robot.assisted_grasp_end_points):
            for point in points.get(working_arm, []) or []:
                attachment(point.link_name)
                self.grasp_points.append((point.link_name, _array(point.position).astype(float)
                    * _array(getattr(self.robot, 'scale', [1., 1., 1.]))))
        attachment(self.robot.eef_link_names[working_arm])

    def capture(self):
        _main_thread()
        measured = _array(self.robot.get_joint_positions())
        names = self.robot.dof_names_ordered
        if len(names) != len(measured):
            raise RuntimeError("Camera FK requires one measured encoder per robot joint.")
        frames = self.kinematics.frames(dict(zip(names, measured)),
            links={anchor for anchor, _ in self.attachments.values()})
        for name, (anchor, fixed) in self.attachments.items():
            frames[name] = frames[anchor] @ fixed
        eef = frames[self.robot.eef_link_names[self.working_arm]][:3, 3]
        points = [(frames[link] @ np.r_[position, 1])[:3] for link, position in self.grasp_points]
        grasp = np.mean(points, axis=0) if points else None
        return {name: CameraGeometry(name=name, image_size=size, intrinsics=k,
                    camera_to_base=frames[link] @ mount, control_point_base_m=eef,
                    grasp_center_base_m=grasp,
                    grasp_reference="mean_of_robot_finger_grasp_points" if points else "unavailable")
                for name, (link, mount, k, size) in self.mounts.items()}

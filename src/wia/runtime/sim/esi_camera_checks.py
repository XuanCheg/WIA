"""Native, private camera/FK/depth acceptance; never sent through MCP."""
from __future__ import annotations

import itertools
import json

import numpy as np

from core.camera_geometry import pose_matrix
from core.agent.image_inputs import image_inputs
from core.record.images import save_png
from wia.runtime.sim.esi_omnigibson import _array


def run_camera_geometry_checks(session, controller, executor, source, output):
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def measure(action=None, result=None):
        observation = source.capture(action_boundary=True).observation
        provider = source.camera_geometry_provider
        base = session.robot.links[session.robot.base_footprint_link_name]
        world_base = pose_matrix(*[_array(x) for x in base.get_position_orientation()])
        registry = session.robot_camera_registry
        record = {"action": action, "result": result, "action_step": observation.action_step, "views": {}}
        for name, geometry in observation.camera_geometry.items():
            sensor = session.robot.sensors[registry[name]]
            world_camera = pose_matrix(*[_array(x) for x in sensor.get_position_orientation()])
            native = np.linalg.solve(world_base, world_camera) @ np.diag([1., -1., -1., 1.])
            predicted = np.asarray(geometry.camera_to_base)
            relative = predicted[:3, :3].T @ native[:3, :3]
            angular_error = float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.))))
            translation_error = float(np.linalg.norm(predicted[:3, 3]-native[:3, 3]))
            control_error = float(np.linalg.norm(np.asarray(geometry.control_point_base_m)-controller.ee_pose()["base_position"]))
            native_grasp_points = []
            for points in (session.robot.assisted_grasp_start_points, session.robot.assisted_grasp_end_points):
                for point in points.get(provider.working_arm, []) or []:
                    link = session.robot.links[point.link_name]
                    world_link = pose_matrix(*[_array(x) for x in link.get_position_orientation()])
                    native_grasp_points.append(np.linalg.solve(world_base,
                        world_link @ np.r_[_array(point.position), 1])[:3])
            grasp_error = (float(np.linalg.norm(np.asarray(geometry.grasp_center_base_m)-np.mean(native_grasp_points, axis=0)))
                           if native_grasp_points and geometry.grasp_center_base_m is not None else None)
            view = {"translation_error_m": translation_error, "rotation_error_deg": angular_error,
                    "control_point_error_m": control_error, "grasp_reference_error_m": grasp_error,
                    "references": geometry.references(),
                    "calibration": geometry.to_dict(),
                    "passed": translation_error < 0.001 and angular_error < 0.1 and control_error < 0.001
                              and (grasp_error is None or grasp_error < 0.001)}
            # Independently compare target surface depth with its privately
            # projected world AABB; no target coordinates enter public geometry.
            checker = getattr(session, "private_visibility_checker", None)
            if checker is not None:
                from PIL import Image
                mask_files = sorted(checker.output.glob(f"initial-*-{'head' if name == 'agentview_rgb' else 'wrist'}-target-mask.png"))
                lower, upper = (_array(x) for x in checker.target.aabb)
                corners = np.array(list(itertools.product(*zip(lower, upper))))
                base_corners = (np.linalg.solve(world_base, np.c_[corners, np.ones(8)].T)).T
                z = (np.linalg.solve(predicted, base_corners.T)).T[:, 2]
                depth = getattr(observation.sensors, "head_depth_m" if name == "agentview_rgb" else "wrist_depth_m")
                if action is None and mask_files and depth is not None:
                    mask = np.asarray(Image.open(mask_files[-1]))[..., 0] > 0
                    values = np.asarray(depth).squeeze()[mask]
                    values = values[np.isfinite(values) & (values > 0)]
                    if len(values):
                        fraction = float(np.mean((values >= z.min()-0.005) & (values <= z.max()+0.005)))
                        view["private_target_depth_check"] = {"valid_pixels": len(values), "median_optical_z_m": float(np.median(values)),
                            "target_aabb_optical_z_range_m": [float(z.min()), float(z.max())],
                            "fraction_within_aabb": fraction, "passed": fraction >= 0.95}
                        view["passed"] &= fraction >= 0.95
            record["views"][name] = view
        # Mount calibration is constant; encode the measured encoder input as
        # evidence that subsequent poses use FK, rather than repeated world queries.
        record["joint_positions"] = dict(zip(session.robot.dof_names_ordered,
                                                _array(session.robot.get_joint_positions()).tolist()))
        record["mount_links"] = {name: value[0] for name, value in provider.mounts.items()}
        record["joint_mount_calibration"] = provider.joint_calibration
        record["passed"] = bool(record["views"] and all(v["passed"] for v in record["views"].values()))
        trunk = getattr(controller, "trunk_controller", None)
        if trunk is not None:
            torso = pose_matrix(*[_array(x) for x in session.robot.links["torso_link4"].get_position_orientation()])
            native_height = float(np.linalg.solve(world_base, torso)[2, 3])
            reading = observation.robot_state.trunk_height_m
            control_height = observation.robot_state.trunk_control_height_m
            record["trunk_height_check"] = {"calibrated_height_m": reading,
                "native_installed_height_m": native_height, "nominal_control_height_m": control_height,
                "height_error_m": abs(reading-native_height),
                "mount_correction_m": reading-control_height, "passed": abs(reading-native_height) < 0.001}
            record["passed"] &= record["trunk_height_check"]["passed"]
        index = len(records)
        for name, rgb, _, _ in image_inputs(observation):
            if name in ("agentview_rgb", "wrist_rgb", "head_depth_m", "wrist_depth_m"):
                save_png(output / f"{index:02d}-{action or 'initial'}-{name}.png", rgb)
        records.append(record)
        (output / "checks.json").write_text(json.dumps({"passed": all(r["passed"] for r in records),
                                                       "records": records}, indent=2, allow_nan=False)+"\n")

    measure()
    for action in ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN"):
        measure(action, executor.execute(action))
    if not all(r["passed"] for r in records):
        raise RuntimeError("Native camera geometry acceptance failed; see camera_geometry_checks/checks.json.")
    return records

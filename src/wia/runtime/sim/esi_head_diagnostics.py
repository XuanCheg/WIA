"""Inspect native R1 head occlusion without restarting or moving the robot.

Call diagnose_head_camera(session, output, trial_mounts=...) from a retained
native sampling snapshot. Trials change only the sensor mount and restore it.
RGB and segmentation evidence come directly from the simulator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.record.images import save_png
from wia.runtime.scene_review import write_json
from wia.runtime.sim.esi_omnigibson import _array


def diagnose_head_camera(session, output, *, trial_mounts=()):
    from omnigibson.utils.sampling_utils import raytest
    from omnigibson.utils.usd_utils import PoseAPI

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sensor = session.robot.sensors[session.camera_name]
    old_position, old_quaternion = (_array(value).copy() for value in
        sensor.get_position_orientation(frame="parent"))
    added = "seg_instance_id" not in sensor.modalities
    if added:
        sensor.add_modality("seg_instance_id")
    trials = []
    try:
        for index, mount in enumerate((old_position, *trial_mounts)):
            sensor.set_position_orientation(position=list(mount),
                orientation=old_quaternion.tolist(), frame="parent")
            PoseAPI.invalidate()
            for _ in range(9):
                session.pipeline.og.sim.render()
            data, info = sensor.get_obs()
            rgb = session._rgb_from_data(data, sensor.name)
            filename = output / f"head_{index:02d}.png"
            save_png(filename, rgb)
            segmentation = _array(data["seg_instance_id"])
            visible = [{"label": str(label), "pixels": int(np.count_nonzero(segmentation == int(key)))}
                for key, label in info["seg_instance_id"].items()]
            visible.sort(key=lambda item: item["pixels"], reverse=True)
            position, quaternion = (_array(value) for value in sensor.get_position_orientation())
            direction = Rotation.from_quat(quaternion).apply([0., 0., -1.])
            end = position + 3 * direction
            hits = raytest(position.tolist(), end.tolist(), only_closest=False)
            containing = []
            for name, link in session.robot.links.items():
                low, high = (_array(value) for value in link.aabb)
                if np.all(low <= position) and np.all(position <= high):
                    containing.append(name)
            trials.append({"mount_position_m": np.asarray(mount, dtype=float).tolist(), "mount_quaternion_xyzw": old_quaternion.tolist(),
                "world_position_m": position.tolist(), "world_quaternion_xyzw": quaternion.tolist(),
                "optical_direction_world": direction.tolist(), "containing_robot_link_aabbs": containing,
                "visible_instances": visible, "segmentation_pixel_count": int(segmentation.size), "rgb": str(filename),
                "centre_ray_hits": [{"body": item["rigidBody"], "distance_m": float(item["distance"])}
                    for item in hits if item.get("hit")]})
    finally:
        sensor.set_position_orientation(position=old_position.tolist(),
            orientation=old_quaternion.tolist(), frame="parent")
        if added:
            sensor.remove_modality("seg_instance_id")
        PoseAPI.invalidate()
        for _ in range(3):
            session.pipeline.og.sim.render()
        session.invalidate_camera_cache()
    write_json(output / "diagnostic.json", {"head_sensor": sensor.prim_path,
        "robot": session.robot.name, "robot_prim": session.robot.prim_path,
        "trials": trials, "original_mount_restored": True, "robot_actions": 0})
    return str(output / "diagnostic.json")

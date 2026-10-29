"""Human-only multi-angle cameras. No debug frame enters AgentObservation.

Run ``python -m pytest tests/test_esi_mapping_review.py`` for CPU framing checks.
"""
from __future__ import annotations

import cv2
import numpy as np


def camera_orientation(position, target):
    """USD optical -Z looks at target, +Y up; return xyzw quaternion."""
    from scipy.spatial.transform import Rotation

    backward = np.asarray(position, float) - np.asarray(target, float)
    if backward.shape != (3,) or not np.isfinite(backward).all() or np.linalg.norm(backward) < 1e-8:
        raise ValueError("Camera needs distinct finite position and look-at target.")
    backward /= np.linalg.norm(backward)
    up = np.array([0., 0., 1.])
    if abs(up @ backward) > 0.99:
        up = np.array([0., 1., 0.])
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return Rotation.from_matrix(np.column_stack((right, up, backward))).as_quat()


def depth_rgb(depth, minimum=0.1, maximum=5.0):
    """Fixed meter scale: near is warm, far cool, invalid pixels black."""
    depth = np.asarray(depth)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape, np.float32)
    normalized[valid] = np.clip((depth[valid]-minimum)/(maximum-minimum), 0, 1)
    color = cv2.applyColorMap(np.uint8(np.round((1-normalized)*255)), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    cv2.putText(rgb, f"DEPTH {minimum:g}-{maximum:g} m", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def object_camera_target(positions, base, *, include_robot=True):
    """Frame native task objects for human review, optionally including the R1."""
    target = np.mean(positions, axis=0)
    if include_robot and base is not None:
        target[:2] = (2*target[:2]+base[:2])/3
        target[2] = max(target[2], base[2]+0.65)
    return target


def configure_global_cameras(session, config, *, include_robot=True):
    from omnigibson.sensors import VisionSensor
    from wia.runtime.sim.esi_base import _main_thread
    from wia.runtime.sim.esi_omnigibson import _array
    from wia.runtime.sim.esi_floorplan_review import is_cognitive_mapping_session, mapping_camera_geometry, fit_mapping_camera

    _main_thread()
    if session.global_cameras:
        raise RuntimeError("Human global cameras are configured once per scene.")
    # Ground-truth object poses frame human debug cameras only; do not expose
    # this target, these poses or the sensor registry to the Agent projection.
    positions = []
    for name in getattr(session, "task_object_names", ()):
        obj = session.env.scene.object_registry("name", name)
        if (obj is not None and obj.category not in ('floor', 'floors', 'ceiling', 'ceilings')
                and bool(getattr(obj, 'visible', True))):
            positions.append(_array(obj.get_position_orientation()[0]))
    base = _array(session.robot.get_position_orientation()[0]) if session.robot is not None else None
    mapping_geometry = mapping_camera_geometry(session) if is_cognitive_mapping_session(session) else None
    if mapping_geometry is not None:
        low, high = np.asarray(mapping_geometry["bounds_world_xy_m"], float)
        target = np.array([*((low + high) / 2), mapping_geometry["floor_z_m"] + .8])
    elif positions:
        target = object_camera_target(positions, base, include_robot=include_robot)
    elif base is None:
        from scipy.spatial.transform import Rotation
        target = np.asarray(session.scene_camera_position) + Rotation.from_quat(
            session.scene_camera_quaternion).apply([0., 0., -2.])
    else:
        target = base + np.array([0., 0., 0.65])
    session.human_camera_framing = {
        "target_world_m": target.tolist(), "include_robot": include_robot,
        "task_object_count": len(positions),
    }
    if mapping_geometry is not None:
        session.human_camera_framing.update(strategy="whole_native_floor_bounds_v1",
                                           floor_geometry=mapping_geometry, views={})
    session.viewer_config = config
    for entry in config.global_views:
        position = target + entry.offset_m
        sensor = VisionSensor(relative_prim_path=f"/human_cameras/{entry.name}",
                              name=entry.name, modalities="rgb", image_width=config.camera_size,
                              image_height=config.camera_size, focal_length=config.focal_length)
        sensor.load(session.env.scene)
        sensor.initialize()
        if mapping_geometry is not None:
            fitted = fit_mapping_camera(mapping_geometry, entry.offset_m,
                horizontal_aperture=float(sensor.horizontal_aperture), focal_length=float(sensor.focal_length),
                vertical_aperture=float(sensor.get_attribute("verticalAperture")),
                image_width=int(sensor.image_width), image_height=int(sensor.image_height),
                overhead=entry.name == "global_overhead")
            position = np.asarray(fitted["position_world_m"])
            target = np.asarray(fitted["target_world_m"])
            # Sensor clippingRange affects only this human camera. Never hide
            # scene bodies or modify the mounted Agent cameras or physics.
            sensor.clipping_range = tuple(fitted["clipping_range_m"])
            session.human_camera_framing["views"][entry.name] = fitted
        sensor.set_position_orientation(position=position.tolist(),
                                        orientation=camera_orientation(position, target).tolist(), frame="world")
        session.global_cameras[entry.name] = sensor
    # Human depth visualization can be enabled independently of Agent switches.
    for name in (session.camera_name, session.wrist_camera_name) if session.robot is not None else ():
        sensor = session.robot.sensors[name]
        if "depth_linear" in sensor.all_modalities and "depth_linear" not in sensor.modalities:
            sensor.add_modality("depth_linear")
    for _ in range(3):
        session.pipeline.og.sim.render()

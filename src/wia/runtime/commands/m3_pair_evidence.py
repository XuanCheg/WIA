"""Native evidence for the two-source M3 timing diagnostic.

CPU import from the repository root:
  python -c 'from wia.runtime.commands.m3_pair_evidence import compare_evidence'
Only collect_evidence and capture_review require an initialized native session.
They import native runtime modules inside their functions and execute no physics
steps, robot actions, or model calls. Comparisons concern recorded geometry,
state, and cameras, without pixel equality or physical qualification claims.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import os
from pathlib import Path

import numpy as np


THRESHOLDS = {
    "translation_m": 0.001,
    "orientation_deg": 0.1,
    "scale_absolute": 1e-6,
    "aabb_m": 0.001,
    "robot_joint_position_rad": 1e-4,
    "task_linear_velocity_m_s": 1e-3,
    "task_angular_velocity_rad_s": 1e-2,
    "robot_joint_velocity_rad_s": 1e-3,
    "intrinsics_absolute": 1e-3,
}
VIEWS = ("head", "scene", "global_overhead")


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        from wia.runtime.sim.esi_omnigibson import _array
        return _json_safe(_array(value).tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"unavailable": "Native numeric value is not finite."}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unavailable": f"Unsupported native value type: {type(value).__name__}."}


def _read(getter):
    try:
        return _json_safe(getter())
    except Exception as error:
        return {"unavailable": f"{type(error).__name__}: {error}"}


def _pose(obj, frame="world"):
    result = _read(lambda: obj.get_position_orientation(frame=frame))
    if isinstance(result, list) and len(result) == 2:
        return result
    return result, result


def _camera_evidence(session, sensor):
    if sensor is None:
        return {"unavailable": "Requested native camera is absent."}
    from wia.runtime.sim.esi_camera_geometry import native_intrinsics_when_ready

    position, quaternion = _pose(sensor)
    local_position, local_quaternion = _pose(sensor, "parent")
    return {
        "name": sensor.name,
        "world_position": position,
        "world_quaternion_xyzw": quaternion,
        "local_position": local_position,
        "local_quaternion_xyzw": local_quaternion,
        "intrinsics": _read(lambda: native_intrinsics_when_ready(session, sensor)),
        "image_size": _read(lambda: [int(sensor.image_width), int(sensor.image_height)]),
    }


def collect_evidence(session, gpu):
    """Read actual native state after capture has initialized all three cameras."""
    from omnigibson.utils.usd_utils import PoseAPI
    from wia.runtime.scene_capture_adapters import gpu_evidence

    PoseAPI.invalidate()
    task_names = list(session.task_object_names)
    tasks = {}
    for name in task_names:
        obj = session.env.scene.object_registry("name", name)
        if obj is None:
            tasks[name] = {"unavailable": "Declared task object is absent."}
            continue
        position, quaternion = _pose(obj)
        aabb = _read(lambda: obj.aabb)
        lower, upper = aabb if isinstance(aabb, list) and len(aabb) == 2 else (aabb, aabb)
        tasks[name] = {
            "name": obj.name,
            "category": _read(lambda: obj.category),
            "model": _read(lambda: obj.model),
            "scale": _read(lambda: obj.scale),
            "position": position,
            "quaternion_xyzw": quaternion,
            "aabb_min": lower,
            "aabb_max": upper,
            "state": _read(lambda: obj.dump_state(serialized=False)),
            "linear_velocity": _read(obj.get_linear_velocity)
                if hasattr(obj, "get_linear_velocity") else {"unavailable": "No native velocity getter."},
            "angular_velocity": _read(obj.get_angular_velocity)
                if hasattr(obj, "get_angular_velocity") else {"unavailable": "No native velocity getter."},
        }
    robot = session.robot
    if robot is None:
        robot_record = {"unavailable": "Native robot is absent."}
        head = None
    else:
        position, quaternion = _pose(robot)
        robot_record = {
            "name": robot.name,
            "position": position,
            "quaternion_xyzw": quaternion,
            "joint_names": _read(lambda: list(robot.joints)),
            "joint_positions": _read(robot.get_joint_positions),
            "joint_velocities": _read(robot.get_joint_velocities),
        }
        head = robot.sensors.get(session.camera_name)
    excluded = set(task_names)
    excluded.update(item.name for item in session.env.robots)
    if robot is not None:
        excluded.add(robot.name)
    background = sorted(obj.name for obj in session.env.scene.objects if obj.name not in excluded)
    return _json_safe({
        "native_pid": os.getpid(),
        "scene": session.scene_name,
        "room": session.room_name,
        "source_sha256": hashlib.sha256(Path(session.question_path).read_bytes()).hexdigest(),
        "task_module": session.task_module.__name__,
        "task_object_names": task_names,
        "background_object_names": background,
        "task_objects": tasks,
        "robot": robot_record,
        "cameras": {
            "head": _camera_evidence(session, head),
            "scene": _camera_evidence(session, session.scene_camera),
            "global_overhead": _camera_evidence(session, getattr(session, "_m3_overhead_sensor", None)),
        },
        "gpu": gpu_evidence(session, gpu),
    })


def capture_review(session, output_dir):
    """Render exactly three views while preserving mounted and source transforms."""
    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.usd_utils import PoseAPI
    from core.record.images import save_png
    from wia.runtime.sim.esi_debug_cameras import camera_orientation
    from wia.runtime.sim.esi_omnigibson import NativeCameraNotReady, _array

    if session.robot is None:
        raise RuntimeError("The pair diagnostic requires the native head camera.")
    if session.scene_camera is None:
        session.configure_scene_camera()
    PoseAPI.invalidate()
    positions = []
    for name in session.task_object_names:
        obj = session.env.scene.object_registry("name", name)
        if obj is None:
            raise RuntimeError(f"Missing native task object: {name}.")
        positions.append(_array(obj.get_position_orientation()[0]))
    if not positions:
        raise RuntimeError("No native task objects are available to frame the overhead view.")
    centre = np.mean(positions, axis=0)
    position = centre + np.array([0., 0., 1.45])
    overhead = getattr(session, "_m3_overhead_sensor", None)
    if overhead is None:
        overhead = VisionSensor(
            relative_prim_path="/human_cameras/human_overhead", name="human_overhead",
            modalities="rgb", image_width=512, image_height=512, focal_length=12.,
        )
        overhead.load(session.env.scene)
        overhead.initialize()
        session._m3_overhead_sensor = overhead
    overhead.set_position_orientation(
        position=position.tolist(), orientation=camera_orientation(position, centre).tolist(),
        frame="world",
    )
    sensors = (session.robot.sensors[session.camera_name], session.scene_camera, overhead)
    images = None
    for attempt in range(8):
        PoseAPI.invalidate()
        for sensor in sensors:
            sensor.get_position_orientation()
        for _ in range(3):
            session.pipeline.og.sim.render()
        session.invalidate_camera_cache()
        try:
            raw = session.observe(include_scene=True)
            data, _ = overhead.get_obs()
            images = {
                "head": raw["head_rgb"], "scene": raw["scene_rgb"],
                "global_overhead": session._rgb_from_data(data, overhead.name),
            }
        except NativeCameraNotReady:
            images = None
        if attempt >= 2 and images is not None:
            break
    if images is None:
        raise RuntimeError("Three-view RGB capture remained unavailable after pure renders.")
    output = Path(output_dir).resolve()
    paths = {}
    for view in VIEWS:
        path = output / f"{view}.png"
        save_png(path, images[view])
        paths[view] = str(path)
    return paths


def _array_pair(left, right):
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return a, b


def _maximum_error(left, right):
    pair = _array_pair(left, right)
    return None if pair is None else float(np.max(np.abs(pair[0] - pair[1])))


def _translation_error(left, right):
    pair = _array_pair(left, right)
    return None if pair is None or pair[0].shape != (3,) else float(np.linalg.norm(pair[0] - pair[1]))


def _orientation_error(left, right):
    pair = _array_pair(left, right)
    if pair is None or pair[0].shape != (4,):
        return None
    a, b = pair
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return None
    cosine = np.clip(abs(float(np.dot(a / na, b / nb))), 0., 1.)
    return float(np.degrees(2. * np.arccos(cosine)))


def compare_evidence(warm, cold):
    """Compare native measurements with the thresholds declared for this pair."""
    checks, errors, state = {}, {}, {}

    def exact(name, left, right, *, sets=False):
        available = left is not None and right is not None
        if sets:
            available = isinstance(left, list) and isinstance(right, list)
            same = available and len(left) == len(set(left)) and len(right) == len(set(right)) and set(left) == set(right)
        else:
            available = available and not isinstance(left, dict) and not isinstance(right, dict)
            same = available and left == right
        checks[name] = bool(same)

    def numeric(name, left, right, threshold, measure=_maximum_error):
        error = measure(left, right)
        errors[name] = error
        checks[name] = error is not None and error <= threshold

    def pose(prefix, a, b, position="position", quaternion="quaternion_xyzw"):
        numeric(prefix + ".translation_m", a.get(position), b.get(position),
                THRESHOLDS["translation_m"], _translation_error)
        numeric(prefix + ".orientation_deg", a.get(quaternion), b.get(quaternion),
                THRESHOLDS["orientation_deg"], _orientation_error)

    for field in ("scene", "room", "source_sha256", "task_module"):
        exact(field, warm.get(field), cold.get(field))
    for field in ("task_object_names", "background_object_names"):
        exact(field, warm.get(field), cold.get(field), sets=True)
    a_tasks, b_tasks = warm.get("task_objects", {}), cold.get("task_objects", {})
    exact("task_object_record_names", list(a_tasks), list(b_tasks), sets=True)
    for name in sorted(set(a_tasks) | set(b_tasks)):
        a, b = a_tasks.get(name, {}), b_tasks.get(name, {})
        prefix = "task_objects." + name
        for field in ("name", "category", "model"):
            exact(prefix + "." + field, a.get(field), b.get(field))
        pose(prefix, a, b)
        numeric(prefix + ".scale", a.get("scale"), b.get("scale"), THRESHOLDS["scale_absolute"])
        for field in ("aabb_min", "aabb_max"):
            numeric(prefix + "." + field, a.get(field), b.get(field), THRESHOLDS["aabb_m"])
        for field, threshold in (
            ("linear_velocity", "task_linear_velocity_m_s"),
            ("angular_velocity", "task_angular_velocity_rad_s"),
        ):
            numeric(prefix + "." + field, a.get(field), b.get(field), THRESHOLDS[threshold])
        state[prefix + ".dump_state_available"] = (
            isinstance(a.get("state"), dict) and "unavailable" not in a["state"]
            and isinstance(b.get("state"), dict) and "unavailable" not in b["state"]
        )
        state[prefix + ".dump_state_exact_equal"] = (
            a["state"] == b["state"] if state[prefix + ".dump_state_available"] else None
        )
    a_robot, b_robot = warm.get("robot", {}), cold.get("robot", {})
    exact("robot.name", a_robot.get("name"), b_robot.get("name"))
    exact("robot.joint_names", a_robot.get("joint_names"), b_robot.get("joint_names"))
    pose("robot", a_robot, b_robot)
    numeric("robot.joint_positions", a_robot.get("joint_positions"), b_robot.get("joint_positions"),
            THRESHOLDS["robot_joint_position_rad"])
    numeric("robot.joint_velocities", a_robot.get("joint_velocities"), b_robot.get("joint_velocities"),
            THRESHOLDS["robot_joint_velocity_rad_s"])
    a_cameras, b_cameras = warm.get("cameras", {}), cold.get("cameras", {})
    exact("camera_names", list(a_cameras), list(b_cameras), sets=True)
    checks["required_camera_names"] = set(a_cameras) == set(b_cameras) == set(VIEWS)
    for name in VIEWS:
        a, b = a_cameras.get(name, {}), b_cameras.get(name, {})
        prefix = "cameras." + name
        for frame in ("world", "local"):
            pose(prefix + "." + frame, a, b, frame + "_position", frame + "_quaternion_xyzw")
        numeric(prefix + ".intrinsics", a.get("intrinsics"), b.get("intrinsics"),
                THRESHOLDS["intrinsics_absolute"])
        exact(prefix + ".image_size", a.get("image_size"), b.get("image_size"))
    return {
        "pass": bool(all(checks.values())), "thresholds": dict(THRESHOLDS),
        "checks": checks, "errors": errors, "state_diagnostics": state,
        "state_diagnostics_affect_pass": False,
        "scope": "Recorded native geometry, task velocities, robot joints, and cameras only.",
        "physical_qualification": False, "agent_reset_equivalence": False,
    }

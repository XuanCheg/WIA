"""Native wrist-yaw acceptance; privileged poses remain in debug artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import ROTATE_ATOMS
from core.record.images import StreamingVideoWriter, save_png

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import _array


def run_rotation_checks(session, controller, output: Path, *, direct=False):
    """Check both signs, one target, new orientation holds and camera updates."""
    _main_thread()
    from wia.runtime.sim.esi_actions import create_r1_executor

    executor = create_r1_executor(controller)
    cfg = controller.rotation_config
    if cfg is None:
        raise ValueError("Rotation acceptance requires enabled wrist yaw.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    videos = {view: StreamingVideoWriter(output / f"rotation_{view}.mp4", 6)
              for view in ("head", "wrist")}
    results, checks, coordinate_checks = [], {}, []
    previous_step = controller.on_step
    steps = 0
    completed = False
    initial = controller.ee_pose()

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise RuntimeError(f"Rotation acceptance failed: {name}")

    def capture():
        observation = session.observe(include_wrist=True)
        require("native_head_wrist_cameras", "wrist_rgb" in observation)
        for view, writer in videos.items():
            writer.append(observation[f"{view}_rgb"])

    def on_step():
        nonlocal steps
        steps += 1
        if steps % max(1, round(1 / (6 * controller.dt))) == 0:
            capture()
        if previous_step is not None:
            previous_step()

    def execute(token):
        if direct and token in ROTATE_ATOMS:
            result = controller.rotate_ee_debug("cw" if token == "ROTATE_CW" else "ccw", cfg.rotation_deg)
        else:
            result = executor.execute(token)
        results.append(result)
        require(f"action_{len(results)}_{token}", result["completed"])
        if token in ROTATE_ATOMS:
            expected = (Rotation.from_quat(result["ee_pose_before"]["base_quaternion_xyzw"])
                        * Rotation.from_rotvec([0, 0, np.deg2rad(cfg.rotation_deg)
                                               * (1 if token == "ROTATE_CW" else -1)]))
            actual = Rotation.from_quat(result["ee_pose_after"]["base_quaternion_xyzw"])
            error = float(np.rad2deg((expected.inv() * actual).magnitude()))
            require(f"fixed_target_{len(results)}", error <= cfg.rotation_tolerance_deg)
            base = Rotation.from_quat(result["base_pose_before"]["quaternion_xyzw"])
            expected_world = base * expected
            measured_world = Rotation.from_quat(result["ee_pose_after"]["world_quaternion_xyzw"])
            world_error = float(np.rad2deg((expected_world.inv() * measured_world).magnitude()))
            coordinate_checks.append({"base_yaw_deg": float(np.rad2deg(result["base_pose_before"]["yaw_rad"])),
                                      "action": token, "world_orientation_error_deg": world_error})
            require(f"world_frame_{len(results)}", world_error <= cfg.rotation_tolerance_deg)
        return result

    try:
        controller.on_step = on_step
        capture()
        for view in videos:
            save_png(output / f"before_rotation_{view}.png", session.observe(include_wrist=True)[f"{view}_rgb"])
        # Check the projected wrist-frame direction using the native camera
        # mount, rather than inferring a sign from an Euler yaw at pi roll.
        wrist = session.robot.sensors[session.wrist_camera_name]
        _, camera_quat = wrist.get_position_orientation()
        camera_rotation = Rotation.from_quat(_array(camera_quat))
        first = execute("ROTATE_CW")
        before_world = Rotation.from_quat(first["ee_pose_before"]["world_quaternion_xyzw"])
        after_world = Rotation.from_quat(first["ee_pose_after"]["world_quaternion_xyzw"])
        a, b = [camera_rotation.inv().apply(r.apply([1, 0, 0]))[:2] * [1, -1]
                for r in (before_world, after_world)]
        screen_turn = float(np.rad2deg(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))
        require("cw_matches_native_wrist_projection", screen_turn > 0)
        execute("ROTATE_CCW")
        # Accumulation occurs once per token, including across the quaternion
        # branch at 180 degrees. Translation and base macros preserve it.
        for _ in range(3):
            execute("ROTATE_CW")
        save_png(output / "after_cw_wrist.png", session.observe(include_wrist=True)["wrist_rgb"])
        for token in ("MV_UP", "BASE_BACK", "BASE_TURN_LEFT", "ROTATE_CCW",
                      "MV_DOWN", "BASE_TURN_RIGHT", "BASE_FWD"):
            execute(token)
        for _ in range(2):
            execute("ROTATE_CCW")
        # Large base sweeps have their own M3 scene/placement acceptance in
        # run_arm_checks; a pick/carry scene need not have clearance for them.
        # Cancel an in-progress native macro and verify idle steps keep its
        # measured pose instead of continuing toward the abandoned target.
        cancel_step = steps + 3
        cancelled = executor.execute("ROTATE_CW", should_cancel=lambda: steps >= cancel_step)
        results.append(cancelled)
        require("native_rotation_cancelled", cancelled["cancelled"] and not cancelled["completed"])
        stopped = controller.ee_pose()
        for _ in range(20):
            controller.step_base_debug()
        idle = controller.ee_pose()
        idle_angle = float(np.rad2deg((Rotation.from_quat(stopped["base_quaternion_xyzw"]).inv()
                                      * Rotation.from_quat(idle["base_quaternion_xyzw"])).magnitude()))
        # Only this direct diagnostic restores the small measured partial
        # angle; model-facing actions remain fixed-angle tokens.
        if cancelled["actual_deg"] > 0:
            restored = controller.rotate_ee_debug("ccw", cancelled["actual_deg"])
            results.append(restored)
            require("cancelled_rotation_restored", restored["completed"])
        final = controller.ee_pose()
        error = float(np.rad2deg((Rotation.from_quat(initial["base_quaternion_xyzw"]).inv()
                                  * Rotation.from_quat(final["base_quaternion_xyzw"])).magnitude()))
        require("opposite_rotations_restore_orientation", error <= 2 * cfg.rotation_tolerance_deg)
        require("cameras_updated_during_motion", all(writer.frame_count > 10 for writer in videos.values()))
        capture()
        completed = True
    finally:
        controller.on_step = previous_step
        for writer in videos.values():
            writer.close()
        report = {"completed": completed, "direct": direct, "checks": checks,
                  "results": results, "coordinate_checks": coordinate_checks,
                  "initial": initial, "final": controller.ee_pose(),
                  "videos": {view: {"path": str(writer.live_path), "frames": writer.frame_count}
                             for view, writer in videos.items()}}
        (output / "rotation_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report

"""Question-bound fixed-token M4 acceptance, with privileged debug logging only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from core.action_units import CANONICAL_ACTIONS
from core.record.images import StreamingVideoWriter, save_png

from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_omnigibson import SHOWHARNESS_ROOT, _array


def run_manipulation_checks(session, controller, output, demo, *, executor):
    """Run a manually authored script, never plan from object poses or move them."""
    _main_thread()
    if (session.metadata_path != (SHOWHARNESS_ROOT / demo["metadata"]).resolve()
            or session.question_index != demo["question_index"]):
        raise ValueError("Manipulation demonstration belongs to a different question.")
    stages = demo["stages"]
    expected = ("open", "base_approach", "arm_approach", "grasp", "lift", "carry", "lower", "release", "retreat")
    if tuple(stages) != expected or any(not isinstance(tokens, list) or not tokens or
        any(token not in CANONICAL_ACTIONS or not executor.action_allowed(token) for token in tokens)
        for tokens in stages.values()):
        raise ValueError("Demo requires its ordered stages and fixed BASE/MV/gripper tokens.")
    obj = session.env.scene.object_registry("name", demo["debug_target"])
    if obj is None or obj.visual_only or obj.fixed_base:
        raise ValueError("Demo requires an existing movable collidable rigid target.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fps = float(demo.get("video_fps", 6))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be finite and positive.")
    videos = {view: StreamingVideoWriter(output / f"{view}.mp4", fps) for view in ("head", "wrist")}
    previous_step, previous_result = controller.on_step, controller.on_result
    sample_steps = max(1, round(1 / (fps * controller.dt)))
    steps = 0
    phase = "initial"
    checkpoints, results, checks = {}, [], {}
    initial = None
    carry_offset_errors = []
    grasp_offset = None
    completed = False

    def snapshot():
        pos, quat = obj.get_position_orientation()
        bmin, bmax = obj.aabb
        return {"phase": phase, "control_step": steps,
                "base": controller.pose(), "eef": controller.ee_pose(),
                "gripper": controller.debug_gripper_state(),
                "debug_object": {"name": obj.name, "position": _array(pos).tolist(),
                                 "quaternion_xyzw": _array(quat).tolist(),
                                 "aabb_min": _array(bmin).tolist(), "aabb_max": _array(bmax).tolist()}}

    def offset(sample):
        ee = sample["eef"]
        return Rotation.from_quat(ee["world_quaternion_xyzw"]).inv().apply(
            np.asarray(sample["debug_object"]["position"]) - ee["world_position"])

    def capture():
        sample = snapshot()
        observation = session.observe(include_wrist=True)
        if "wrist_rgb" not in observation:
            raise RuntimeError("M4 demonstration requires the native wrist camera.")
        for view, writer in videos.items():
            writer.append(observation[f"{view}_rgb"])
        with (output / "debug_frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample) + "\n")
        if phase == "carry" and grasp_offset is not None:
            carry_offset_errors.append(float(np.linalg.norm(offset(sample) - grasp_offset)))

    def on_step():
        nonlocal steps
        steps += 1
        if steps % sample_steps == 0:
            capture()
        if previous_step is not None:
            previous_step()

    def on_result(result):
        result["debug_object_after"] = snapshot()["debug_object"]
        results.append(result)
        if previous_result is not None:
            previous_result(result)

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise RuntimeError(f"Manipulation acceptance failed: {name}")

    try:
        controller.on_step, controller.on_result = on_step, on_result
        initial = snapshot()
        require("initial_target_unheld", initial["gripper"]["object_name"] is None)
        capture()
        for phase, tokens in stages.items():
            for token in tokens:
                result = executor.execute(token)
                if not result["completed"]:
                    raise RuntimeError(f"Manipulation action failed: {token}: {result['reason']}")
            sample = checkpoints[phase] = snapshot()
            capture()
            save_png(output / f"{phase}_head.png", session.observe()["head_rgb"])
            if phase == "base_approach":
                require("before_grasp_target_stationary", np.linalg.norm(
                    np.asarray(sample["debug_object"]["position"]) - initial["debug_object"]["position"]) < 0.01)
                require("before_grasp_robot_moved", np.linalg.norm(
                    np.asarray(sample["base"]["position"]) - initial["base"]["position"]) > 0.15)
            elif phase == "grasp":
                require("grasp_holds_target", sample["gripper"]["holding"] is True and
                        sample["gripper"]["object_name"] == obj.name and sample["gripper"]["constraint_active"])
                grasp_offset = offset(sample)
            elif phase == "lift":
                require("lift_clears_support", sample["debug_object"]["aabb_min"][2]
                        > initial["debug_object"]["aabb_min"][2] + 0.05)
            elif phase == "carry":
                require("carry_remains_held", sample["gripper"]["holding"] is True and sample["gripper"]["object_name"] == obj.name)
                require("carry_moves_object", np.linalg.norm(
                    np.asarray(sample["debug_object"]["position"]) - checkpoints["lift"]["debug_object"]["position"]) > 0.15)
                require("carry_follows_ee", max(carry_offset_errors + [float(np.linalg.norm(offset(sample) - grasp_offset))]) < 0.02)
            elif phase == "release":
                require("release_detaches", sample["gripper"]["object_name"] is None and
                        not sample["gripper"]["constraint_active"] and sample["gripper"]["gripper_open"])
            elif phase == "lower":
                require("lower_near_support", abs(sample["debug_object"]["aabb_min"][2]
                        - initial["debug_object"]["aabb_min"][2]) < 0.02)
            elif phase == "retreat":
                for _ in range(round(1 / controller.dt)):
                    controller.step_base_debug()
                sample = checkpoints[phase] = snapshot()
                require("released_object_does_not_follow", np.linalg.norm(
                    np.asarray(sample["debug_object"]["position"]) - checkpoints["release"]["debug_object"]["position"]) < 0.015)
                require("placed_on_support", abs(sample["debug_object"]["aabb_min"][2]
                        - initial["debug_object"]["aabb_min"][2]) < 0.02)
        completed = True
    finally:
        controller.on_step, controller.on_result = previous_step, previous_result
        for writer in videos.values():
            writer.close()
        report = {"completed": completed, "grasp_mode": "assisted", "demo": demo,
                  "checks": checks, "checkpoints": checkpoints, "results": results,
                  "initial": initial, "max_carry_offset_error_m": max(carry_offset_errors, default=None),
                  "videos": {view: {"path": str(writer.live_path), "frames": writer.frame_count}
                             for view, writer in videos.items()}}
        (output / "manipulation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "action_results.json").write_text(json.dumps(results, indent=2) + "\n")
    return report

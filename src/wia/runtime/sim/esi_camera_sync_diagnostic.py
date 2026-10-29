"""Run a short native turn/camera diagnostic without a model.

Submit an ordinary resident episode request with camera_sync_diagnostic=true.
The resident service loads the normal task/robot/executor, runs four turns and
retains its Kit application after scene cleanup. Results are not benchmark scores.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

import numpy as np


def image_difference(first, second):
    difference = np.abs(np.asarray(first, float) - np.asarray(second, float))
    return {"mean_absolute_channel_difference_0_255": float(difference.mean()),
            "fraction_pixels_any_channel_change_gt_20": float((difference.max(axis=-1) > 20).mean())}


def depth_difference(first, second):
    a, b = np.asarray(first), np.asarray(second)
    finite = np.isfinite(a) & np.isfinite(b)
    difference = np.abs(a[finite] - b[finite])
    return {"identical": bool(np.array_equal(a, b, equal_nan=True)),
            "max_finite_absolute_difference_m": float(difference.max()) if difference.size else None,
            "fraction_finite_pixels_change_gt_1mm": float((difference > .001).mean()) if difference.size else None,
            "finite_masks_identical": bool(np.array_equal(np.isfinite(a), np.isfinite(b)))}


def run(native_argv, output):
    """Replace only the web loop, keeping ordinary native setup and teardown."""
    from core.action_compiler import ParameterizedAction
    from core.record.images import save_png
    import wia.runtime.sim.esi_web as esi_web
    from wia.runtime.sim.esi_camera_freshness import prepare_camera_freshness, check_camera_freshness
    from wia.runtime.commands.run_esi_robot import main

    output = Path(output)
    report = {"kind": "native_camera_sync_diagnostic", "model_requests": 0,
              "benchmark_result": False, "steps": [], "passed": False}

    def save_report():
        (output / "camera_sync_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def diagnostic(session, metadata, controller, executor, **kwargs):
        source = session.observation_source
        began = time.monotonic()
        report["native_boundary_graphics_iterations"] = session._native_boundary_graphics_iterations
        report["native_freshness_required"] = session._native_camera_freshness_required
        try:
            source.begin_episode(source.task_text)
            before = source.latest_snapshot.observation
            before.save(output / "camera_sync_initial")
            report["initial_freshness"] = deepcopy(session._native_camera_freshness)
            for index, (action, angle) in enumerate((
                    ("BASE_TURN_RIGHT", 20.), ("BASE_TURN_LEFT", 20.),
                    ("BASE_TURN_RIGHT", 40.), ("BASE_TURN_LEFT", 40.)), 1):
                pose_before = controller.pose()
                result = executor.execute_macro(ParameterizedAction(action=action, angle_deg=angle),
                    resolution=executor.primitive_resolution, limits=executor.macro_limits)
                observation = source.latest_snapshot.observation
                directory = output / f"camera_sync_step_{index:02d}"
                observation.save(directory / "canonical")
                pose_after = controller.pose()
                row = {"step": index, "command": {"action": action, "angle_deg": angle},
                       "pose_before": pose_before, "pose_after": pose_after,
                       "result": result, "freshness": deepcopy(session._native_camera_freshness),
                       "head_rgb_before_after": image_difference(before.agentview_rgb, observation.agentview_rgb),
                       "head_depth_before_after": depth_difference(before.sensors.head_depth_m, observation.sensors.head_depth_m),
                       "late_graphics_reads": []}
                generation = session._physical_step_generation
                targets = prepare_camera_freshness(session)
                for render_index in range(1, 4):
                    session.invalidate_camera_cache()
                    session.pipeline.og.sim.render()
                    raw = session.observe(include_wrist=True, include_sensors=True)
                    check_camera_freshness(session, targets)
                    late = {"additional_render": render_index,
                            "head_rgb_vs_canonical": image_difference(observation.agentview_rgb, raw["head_rgb"]),
                            "wrist_rgb_vs_canonical": image_difference(observation.wrist_rgb, raw["wrist_rgb"]),
                            "head_depth_vs_canonical": depth_difference(observation.sensors.head_depth_m, raw["head_depth_m"]),
                            "wrist_depth_vs_canonical": depth_difference(observation.sensors.wrist_depth_m, raw["wrist_depth_m"])}
                    row["late_graphics_reads"].append(late)
                    if render_index == 3:
                        save_png(directory / "late_head.png", raw["head_rgb"])
                        save_png(directory / "late_wrist.png", raw["wrist_rgb"])
                        np.save(directory / "late_head_depth_m.npy", raw["head_depth_m"])
                        np.save(directory / "late_wrist_depth_m.npy", raw["wrist_depth_m"])
                row["physical_generation_unchanged_during_late_reads"] = generation == session._physical_step_generation
                row["pose_after_late_reads"] = controller.pose()
                row["late_depth_stable"] = all(
                    late[name]["finite_masks_identical"]
                    and late[name]["max_finite_absolute_difference_m"] is not None
                    and late[name]["max_finite_absolute_difference_m"] <= .001
                    for late in row["late_graphics_reads"]
                    for name in ("head_depth_vs_canonical", "wrist_depth_vs_canonical"))
                report["steps"].append(row)
                save_report()
                print(f"[camera-sync] step={index} action={action} completed={result['completed']} "
                      f"late_depth_stable={row['late_depth_stable']}", flush=True)
                if not result["completed"]:
                    break
                before = observation
            report["passed"] = len(report["steps"]) == 4 and all(
                row["result"]["completed"] and row["freshness"]["status"] == "ready"
                and row["physical_generation_unchanged_during_late_reads"] and row["late_depth_stable"]
                and not row["head_depth_before_after"]["identical"] for row in report["steps"])
        except Exception:
            report["error"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            report["wall_time_s"] = time.monotonic() - began
            report["performance_counters"] = session.performance.snapshot()
            save_report()

    previous = esi_web.run_esi_control
    esi_web.run_esi_control = diagnostic
    try:
        return main(native_argv)
    finally:
        esi_web.run_esi_control = previous

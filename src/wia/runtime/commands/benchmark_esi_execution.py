#!/usr/bin/env python3
"""Seeded native performance replay or instrumented web Agent service.

Run inside a retained interactive allocation. --code-root can point to an
archived source tree, so the same harness measures unmodified baseline code.
Arguments after -- are forwarded to run_esi_robot.py.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=WIA_RUNTIME_ROOT)
    parser.add_argument("--mode", choices=("replay", "agent"), default="replay")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sequence-file", type=Path, help="Explicit scripted commands; default is the archived successful tomato trajectory.")
    options, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.path.insert(0, str(options.code_root.resolve()))
    import numpy as np
    import torch
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    from core.performance import PerformanceCounters, instrument_native_session, timed_method
    import wia.runtime.sim.esi_actions as adapters
    import wia.runtime.sim.esi_web as web
    from core.action_interface import ActionExecutor
    from core.action_compiler import ParameterizedAction
    from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
    from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
    from core.observations import AgentObservation
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.web.web_teleop.esi_recording import CompactVideoWriter
    from wia.runtime.commands.run_esi_robot import main as robot_main

    counters = PerformanceCounters()
    original_create, original_macro, original_web = adapters.create_r1_executor, ActionExecutor.execute_macro, web.run_esi_control
    original_save = AgentObservation.save
    original_session_close = ESISession.close
    output_path = Path(next(forwarded[i+1] for i, arg in enumerate(forwarded) if arg == "--output-dir"))
    output_path.mkdir(parents=True, exist_ok=True)
    import os
    (output_path / "sim_pid.txt").write_text(str(os.getpid())+'\n')
    def session_close(self):
        counters.save(output_path / "performance_counters.json")
        return original_session_close(self)
    ESISession.close = session_close
    def measured_save(self, *args, **kwargs):
        with counters.measure("observation_serialization_time"):
            return original_save(self, *args, **kwargs)
    AgentObservation.save = measured_save
    original_append, original_close = CompactVideoWriter.append, CompactVideoWriter.close
    def append(self, *args, **kwargs):
        with counters.measure("video_encode_wall_time"):
            return original_append(self, *args, **kwargs)
    def close(self, *args, **kwargs):
        with counters.measure("video_finalize_wall_time"):
            return original_close(self, *args, **kwargs)
    CompactVideoWriter.append, CompactVideoWriter.close = append, close
    def create(controller):
        controller.session.performance = counters
        instrument_native_session(controller.session)
        timed_method(controller.session.observation_source, "capture", counters, "observation_capture_inclusive_wall_time",
                     "full_observation_captures")
        return original_create(controller)
    adapters.create_r1_executor = create
    def macro(self, *args, **kwargs):
        before = counters.snapshot()
        started = time.perf_counter()
        result = original_macro(self, *args, **kwargs)
        if getattr(self, "performance", None) is not counters:
            counters.add("macro_wall_time", time.perf_counter()-started)
        result['performance'] = counters.difference(before)
        with (output_path / 'timing_macros.jsonl').open('a') as log:
            log.write(json.dumps({'command': result.get('command'), 'timings': result['performance'],
                                 'duration_s': result['duration_s'], 'execution': result['execution']})+'\n')
        return result
    ActionExecutor.execute_macro = macro

    def replay(session, metadata, controller, executor, **kwargs):
        output = Path(next(forwarded[i+1] for i, arg in enumerate(forwarded) if arg == "--output-dir"))
        source = session.observation_source
        backend = ESIControlBackend(metadata, executor, observation_source=source)
        viewer = ESIViewerBackend(backend, kwargs["recording_dir"], kwargs["viewer_config"]) if kwargs.get("viewer_config") else None
        previous_step, previous_capture = controller.on_step, source.on_capture
        source.include_scene = bool(metadata.get("scene_camera"))
        next_preview = 0.
        optimized = metadata.get("performance", {}).get("observation_pipeline") == "optimized"
        preview_fps = metadata.get("performance", {}).get("preview_fps", kwargs.get("capture_fps", 10.))
        def publish(snapshot):
            backend.publish(snapshot, controller_state=snapshot.controller_state)
            if viewer:
                viewer.publish(snapshot)
        source.on_capture = publish
        def on_step():
            nonlocal next_preview
            if previous_step:
                previous_step()
            now = time.monotonic()
            if preview_fps > 0 and now >= next_preview:
                if optimized:
                    preview = source.capture_preview()
                    backend.publish_preview(preview)
                    if viewer:
                        viewer.publish_preview(preview)
                else:
                    source.capture(action_boundary=False)
                next_preview = time.monotonic() + 1. / preview_fps
        controller.on_step = on_step
        source.begin_episode(source.task_text)
        if viewer and options.record:
            started_recording = viewer.recorder.start()
            if not started_recording.get("ok"):
                raise RuntimeError(str(started_recording))
        sequence = [{"action": "MV_DOWN", "distance_m": .08}] * 4 + [
            {"action": "GRASP"}, {"action": "MV_UP", "distance_m": .06}]
        if options.sequence_file:
            sequence = json.loads(options.sequence_file.read_text())
        rows = []
        started = time.perf_counter()
        def execute(command, label, **extra):
            before = counters.snapshot()
            boundary = source.action_step
            wall = time.perf_counter()
            result = executor.execute_macro(ParameterizedAction(**command),
                resolution=executor.primitive_resolution, limits=executor.macro_limits, **extra)
            row = {"label": label, "command": command, "wall_time_s": time.perf_counter()-wall,
                "result": result, "timings": counters.difference(before),
                "action_boundaries": source.action_step-boundary}
            rows.append(row)
            with (output / "timing_actions.jsonl").open("a") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print("[performance] " + json.dumps({"label": label, "wall_time_s": row["wall_time_s"],
                "completed": result["completed"], "reason": result.get("reason"),
                "control_steps": result.get("control_steps"), "actual": result["actual"]}), flush=True)
            return result
        try:
            for index, command in enumerate(sequence):
                result = execute(command, f"trajectory_{index+1}")
                if not result["success"]:
                    break
            evaluator = session.task_evaluator
            success = evaluator.record(source.agent_observation(), action=None)
            if optimized:
                observation = source.agent_observation()
                reuse = session.reuse_step_observations
                session.reuse_step_observations = False
                try:
                    raw = session.observe(include_wrist=True, include_sensors=True)
                finally:
                    session.reuse_step_observations = reuse
                mapping = {'agentview_rgb': 'head_rgb', 'wrist_rgb': 'wrist_rgb'}
                equal = {name: bool(np.array_equal(pixels, raw[mapping.get(name, name)]))
                         for name, pixels in observation.rgb_images().items() if pixels is not None}
                for name in ('head_depth_m', 'wrist_depth_m'):
                    depth = getattr(observation.sensors, name)
                    equal[name] = depth is not None and name in raw and bool(np.array_equal(depth, raw[name], equal_nan=True))
                (output / 'native_cache_alignment.json').write_text(json.dumps({
                    'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                    'same_render_no_physics_between_reads': True, 'channels_equal': equal,
                    'passed': all(equal.values())}, indent=2)+'\n')
                if not all(equal.values()):
                    raise RuntimeError('Cached native RGB/depth differs from direct reads of the same render.')
            checks = []
            if options.checks:
                probes = [("arm_1cm", {"action": "MV_UP", "distance_m": .01}),
                          ("arm_8cm", {"action": "MV_UP", "distance_m": .08}),
                          ("base_20cm", {"action": "BASE_FWD", "distance_m": .20}),
                          ("base_30deg", {"action": "BASE_TURN_LEFT", "angle_deg": 30.}),
                          ("trunk_2cm", {"action": "TRUNK_UP", "distance_m": .02})]
                for label, command in probes:
                    checks.append(execute(command, label))
            counters.add("total_episode_wall_time", time.perf_counter()-started)
            summary = {"seed": options.seed, "node": __import__("socket").gethostname(),
                "started_from": str(options.code_root), "metadata": metadata,
                "task_success": success["task_success"], "evaluation": success,
                "timing_semantics": "inclusive; categories overlap", "timings": counters.snapshot(),
                "actions": rows}
            (output / "performance_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        finally:
            controller.on_step, source.on_capture = previous_step, previous_capture
            if viewer:
                viewer.close()
            counters.save(output / "performance_counters.json")
    web.run_esi_control = replay if options.mode == "replay" else original_web
    return robot_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())

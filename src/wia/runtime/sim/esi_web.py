"""ESI owner-thread lifecycle: physics/sensors behind the canonical executor."""
from __future__ import annotations

import signal
import threading
import time
import queue
import json
from pathlib import Path

from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve

from wia.runtime.sim.esi_base import _main_thread


def run_esi_control(session, metadata: dict, controller, executor, *, host="127.0.0.1", port=8615,
                     capture_fps=5.0, duration_s=None, viewer_config=None, recording_dir=None,
                     preview_fps=None, observation_pipeline="optimized"):
    _main_thread()
    if (capture_fps <= 0 or (duration_s is not None and duration_s <= 0)):
        raise ValueError("Capture FPS and optional duration must be positive.")
    source = getattr(session, "observation_source", None)
    if source is None:
        from wia.runtime.sim.esi_observations import ESIObservationSource

        source = ESIObservationSource(session, task_text=metadata.get("question") or metadata["task"])
    backend = ESIControlBackend(metadata, executor, observation_source=source)
    viewer = None
    if viewer_config is not None:
        from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
        viewer = ESIViewerBackend(backend, recording_dir, viewer_config)
    server = serve(viewer or backend, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, name="esi-http", daemon=True)
    previous_signals = {}
    previous_step = controller.on_step
    previous_capture, previous_scene = source.on_capture, source.include_scene
    previous_boundary_hook = source.on_boundary_start
    source.on_boundary_start = lambda: backend.owner_phase(
        "capturing_observation" if backend.control_owner == "agent" else "human_idle")
    deadline = time.monotonic() + duration_s if duration_s is not None else None
    next_capture = 0.0
    from wia.runtime.sim.esi_rendering import RecordingCadence
    recording_cadence = RecordingCadence(metadata.get("performance", {}).get("recording_hz", 5.0))
    preview_fps = capture_fps if preview_fps is None else preview_fps
    human_preview = metadata.get("performance", {}).get("human_preview", True)
    previews = queue.Queue(maxsize=1)
    preview_stop = threading.Event()
    def preview_worker():
        while not preview_stop.is_set():
            try:
                preview = previews.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                # Order encoding and boundary publication; an old preview may
                # never overwrite the frames from a newer canonical boundary.
                with backend.frame_cond:
                    current = backend.agent_snapshot
                    if current is not None and preview.source_snapshot_id < current.snapshot_id:
                        continue
                    backend.publish_preview(preview)
                    if viewer:
                        viewer.publish_preview(preview)
            except Exception:
                counters = getattr(session, "performance", None)
                if counters:
                    counters.add("human_preview_dropped")
            finally:
                previews.task_done()
    preview_thread = threading.Thread(target=preview_worker, name="esi-preview-encode", daemon=True)

    def publish(snapshot):
        backend.publish(snapshot, controller_state=snapshot.controller_state)
        if viewer is not None and (human_preview or viewer.recording_enabled):
            viewer.publish(snapshot)

    source.on_capture = publish
    source.controller_state_provider = controller.controller_state
    source.include_scene = bool(metadata.get("scene_camera"))

    def capture(force=False):
        nonlocal next_capture
        now = time.monotonic()
        if force:
            source.capture(action_boundary=True)
        elif getattr(session, "action_executing", False):
            active = viewer is not None and viewer.recording_enabled and viewer.recorder.state()['recording']
            if recording_cadence.due(float(session.pipeline.og.sim.current_time) if active else 0., active=active):
                # Publish only to the human recorder; never mutate the Agent snapshot.
                viewer.publish_preview(source.capture_preview(recording=True))
                counters = getattr(session, "performance", None)
                if counters:
                    counters.add("intermediate_recording_capture_count")
            return
        elif human_preview and preview_fps > 0 and now >= next_capture:
            if observation_pipeline == "legacy":
                source.capture(action_boundary=not backend.busy and backend.control_owner != "agent")
            elif not previews.full():
                try:
                    previews.put_nowait(source.capture_preview())
                except (queue.Full, RuntimeError):
                    pass  # Readiness/load affects only the optional human preview.
            next_capture = time.monotonic() + 1.0 / preview_fps

    def on_step():
        backend.owner_sample()
        if previous_step is not None:
            previous_step()
        capture()

    def interrupt(signum, frame):
        # Kit may catch KeyboardInterrupt raised inside a native callback.
        # A persistent flag survives that callback and cancels on main after
        # the current physics step returns.
        backend.closing.set()

    def expired():
        return backend.closing.is_set() or (deadline is not None and time.monotonic() >= deadline)

    try:
        controller.on_step = on_step
        capture(force=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, interrupt)
        thread.start()
        if human_preview and observation_pipeline == "optimized":
            preview_thread.start()
        groups = ("base + both arms" if backend.capabilities.right_ee_translation else
                  "base + left arm" if backend.action_groups["arm"] else "base")
        if backend.action_groups["gripper"]:
            groups += " + grippers (assisted)" if backend.capabilities.right_gripper else " + gripper (assisted)"
        print(f"[esi-web] GUMI {groups} control: http://{host}:{server.server_port}", flush=True)
        print(f"[esi-web] action_groups: {backend.action_groups}; main-thread physics", flush=True)
        if metadata.get("standby_ready_path"):
            import os
            from datetime import datetime, timezone
            Path(metadata["standby_ready_path"]).write_text(json.dumps({
                "ready": True, "native_pid": os.getpid(), "port": server.server_port,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_id": source.agent_observation().snapshot_id,
                "physical_generation": getattr(session, "_physical_step_generation", None),
                "simulation_time_s": backend.simulation_time(),
                "native_boundary_graphics_iterations": getattr(session, "_native_boundary_graphics_iterations", None),
                "camera_freshness": getattr(session, "_native_camera_freshness", None),
                "standby_policy": "No idle physics while human-owned; Agent control restores the episode world policy."
            }, indent=2) + "\n")
        while not expired():
            started = time.monotonic()
            processed = backend.process_next(should_stop=expired)
            exploration = getattr(session, "exploration_evaluator", None)
            if exploration is not None and exploration.active and backend.control_owner == "human":
                exploration.finish_episode()
            if not processed:
                # Maintain the frozen joint / base-relative EE targets while idle.
                standby = getattr(session, "standby_until_agent", False) and backend.control_owner == "human"
                if standby:
                    backend.owner_phase("standby")
                if not standby and backend.should_advance_world():
                    controller.step_base_debug()
                    backend.owner_sample(idle_step=True)
                else:
                    backend.owner_sample()
            capture()
            time.sleep(max(0.0, controller.dt - (time.monotonic() - started)))
    except KeyboardInterrupt:
        backend.closing.set()
    finally:
        print("[esi-web] stopping control", flush=True)
        controller.on_step = previous_step
        source.on_capture, source.include_scene = previous_capture, previous_scene
        source.on_boundary_start = previous_boundary_hook
        if metadata.get("world_metrics_path"):
            Path(metadata["world_metrics_path"]).write_text(json.dumps(backend.world_policy.snapshot(), indent=2) + "\n")
        backend.shutdown()
        preview_stop.set()
        if preview_thread.is_alive():
            preview_thread.join(timeout=5)
        if viewer is not None:
            viewer.close()
        try:
            controller.stop()
        finally:
            if thread.is_alive():
                server.shutdown()
                thread.join(timeout=5)
            server.server_close()
            for sig, handler in previous_signals.items():
                signal.signal(sig, handler)

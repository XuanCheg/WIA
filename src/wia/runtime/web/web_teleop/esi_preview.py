"""Read-only ESI camera adapter for the existing GUMI UI.

The main thread renders and reads the simulator. HTTP threads only read cached
JPEGs/state; no request can submit a robot action, reset, or recording command.
"""
from __future__ import annotations

import signal
import threading
import time
from datetime import datetime, timezone

import cv2
from core.robot_cameras import RGB_TO_VIEW, ROBOT_CAMERA_LABELS
import numpy as np

from core.action_interface import BackendCapabilities
from core.observations import CameraSnapshot

from gumi.web_teleop.server import serve

READ_ONLY_MESSAGE = "ESI live cameras — controls disabled"


class ESIPreviewBackend:
    read_only = True
    dual_arm = False
    STEP_TOKENS = ()
    STREAM_VIEWS = ("agentview", *RGB_TO_VIEW.values())

    def __init__(self, metadata: dict, *, observation_source=None) -> None:
        self.frame_cond = threading.Condition()
        self.frames: dict[str, bytes] = {}
        self.frame_seq = 0
        self.frame_time: str | None = None
        self.metadata = metadata
        self.main_view = "head"
        self.observation_source = observation_source

    def agent_observation(self):
        if self.observation_source is None:
            raise RuntimeError("Canonical observation source is unavailable.")
        return self.observation_source.agent_observation()

    def publish(self, observation: CameraSnapshot | dict) -> None:
        """Encode a fresh observation supplied by the simulator's main thread."""
        snapshot = observation if isinstance(observation, CameraSnapshot) else None
        if snapshot is not None:
            observation = dict(snapshot.rgb_views(), **snapshot.observation.additional_rgb)
        frames = {}
        scene_available = "scene_rgb" in observation
        views = (("head" if scene_available or snapshot is not None else "agentview", "head_rgb"),
                 ("wrist", "wrist_rgb"), ("scene", "scene_rgb"),
                 *((view, name) for name, view in RGB_TO_VIEW.items() if name not in ("agentview_rgb", "wrist_rgb")))
        for view, key in views:
            if key not in observation:
                continue
            # Scene/debug cameras stay local. Both UI and model HTTP routes use
            # only the configured robot-mounted input views.
            if view == "scene":
                continue
            switch = f"{view if view != 'agentview' else 'head'}_rgb" if view in ("head", "wrist", "agentview") else "additional_robot_rgb"
            if snapshot is not None and not getattr(snapshot.observation.observation_config, switch):
                continue
            bgr = cv2.cvtColor(np.ascontiguousarray(observation[key]), cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError(f"Cannot encode {view} camera frame.")
            frames[view] = encoded.tobytes()
        if "head" in frames:
            frames["agentview"] = frames["head"]
        with self.frame_cond:
            self.frames = frames
            self.frame_seq = snapshot.observation.snapshot_id if snapshot is not None else self.frame_seq + 1
            self.frame_time = (snapshot.observation.captured_at if snapshot is not None else
                               datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
            self.frame_cond.notify_all()

    def publish_preview(self, preview):
        keys = {"head": "head_rgb", "wrist": "wrist_rgb", "scene": "scene_rgb"}
        keys.update({view: name for name, view in RGB_TO_VIEW.items() if name not in ('agentview_rgb', 'wrist_rgb')})
        self.publish({keys[name]: rgb for name, rgb in preview.frames.items() if name in keys})
        with self.frame_cond:
            if 'agentview' in self.frames and 'head' not in self.frames:
                self.frames['head'] = self.frames['agentview']

    def state(self) -> dict:
        with self.frame_cond:
            seq, frame_time, views = self.frame_seq, self.frame_time, sorted(self.frames)
        return {
            "read_only": True, "mode": "esi_preview", "message": f"{READ_ONLY_MESSAGE} · frame {seq}",
            "capabilities": BackendCapabilities().to_dict(),
            "task_evaluation_available": False,
            "task_text": self.metadata.get("question") or self.metadata["task"],
            "robot_name": self.metadata["robot_name"],
            "robot_type": self.metadata["robot_type"],
            "camera_names": {
                "agentview": self.metadata["camera_selected"],
                "head": self.metadata["camera_selected"],
                "wrist": self.metadata.get("wrist_camera_selected"),
            },
            "camera_labels": {"agentview": "R1 HEAD", **{RGB_TO_VIEW[name]: label for name, label in ROBOT_CAMERA_LABELS.items()}},
            "camera_views": [], "main_view": "head",
            "available_views": views, "frame_seq": seq, "frame_time": frame_time,
            "physics_paused": True,
            # Existing GUMI status fields; no manipulation/evaluation is performed.
            "recording": False, "rollout": None, "steps": 0, "rollouts_saved": 0,
            "last_token": "-", "gripper_closed": None, "next_gripper": None,
            "holding": None, "picked": None, "placed": None, "task_done": None,
            "can_stop": False, "require_task_done": False, "task_editable": False,
            "ee_pos": None, "sim": True, "dual_arm": False, "arms": [],
            "step_tokens": [], "aliases": {}, "history": [],
        }

    def submit(self, kind, arg=None, timeout=60.0) -> dict:
        return {"ok": False, "status": 405, "message": READ_ONLY_MESSAGE}


def run_preview(session, metadata: dict, *, host: str = "127.0.0.1", port: int = 8615,
                capture_fps: float = 5.0, duration_s: float | None = None,
                viewer_config=None, recording_dir=None, preview_fps=None, observation_pipeline="optimized") -> None:
    """Serve cached cameras while rendering on the caller's main thread."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("ESI preview must run on the main thread.")
    if capture_fps <= 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("Capture FPS and optional duration must be positive.")
    source = getattr(session, "observation_source", None)
    if source is None:
        from wia.runtime.sim.esi_observations import ESIObservationSource

        source = ESIObservationSource(session, task_text=metadata.get("question") or metadata["task"])
    backend = ESIPreviewBackend(metadata, observation_source=source)
    viewer = None
    if viewer_config is not None:
        from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
        viewer = ESIViewerBackend(backend, recording_dir, viewer_config)
    server = serve(viewer or backend, host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, name="esi-http", daemon=True)
    previous_signals = {}
    previous_capture, previous_scene = source.on_capture, source.include_scene
    def publish(snapshot):
        backend.publish(snapshot)
        if viewer is not None:
            viewer.publish(snapshot)

    source.on_capture = publish
    source.include_scene = bool(metadata.get("scene_camera"))
    preview_fps = capture_fps if preview_fps is None else preview_fps
    human_preview = metadata.get("performance", {}).get("human_preview", True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    try:
        # Pause physics after upstream initialization. Rendering refreshes actual
        # robot sensor buffers without env.step(), actions, or continued free fall.
        session.pipeline.og.sim.pause()
        source.capture(action_boundary=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_signals[sig] = signal.signal(sig, interrupt)
        thread.start()
        print(f"[esi-web] GUMI read-only preview: http://{host}:{server.server_port}", flush=True)
        print(f"[esi-web] cameras: {backend.state()['camera_names']}; physics paused", flush=True)
        print("[esi-web] Ctrl+C stops preview; robot movement is disabled", flush=True)
        deadline = time.monotonic() + duration_s if duration_s is not None else None
        while deadline is None or time.monotonic() < deadline:
            started = time.monotonic()
            if human_preview and preview_fps > 0:
                if observation_pipeline == "legacy":
                    source.capture(action_boundary=True)
                else:
                    invalidate = getattr(session, "invalidate_camera_cache", None)
                    if invalidate:
                        invalidate()
                    session.pipeline.og.sim.render()
                    preview = source.capture_preview()
                    backend.publish_preview(preview)
                    if viewer:
                        viewer.publish_preview(preview)
            delay = max(0.0, 1.0 / (preview_fps or 2.) - (time.monotonic() - started))
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
            time.sleep(delay)
    except KeyboardInterrupt:
        print("[esi-web] stopping preview", flush=True)
    finally:
        source.on_capture, source.include_scene = previous_capture, previous_scene
        if viewer is not None:
            viewer.close()
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)

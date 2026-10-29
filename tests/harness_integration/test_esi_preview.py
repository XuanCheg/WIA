"""Read-only HTTP boundary and simulator-thread checks without OmniGibson."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import requests

from wia.runtime.web.web_teleop.esi_preview import ESIPreviewBackend, run_preview
from gumi.web_teleop.server import serve

METADATA = {
    "task": "action_order_inference", "scene": "test_scene", "room": "test_room",
    "robot_name": "test_robot", "robot_type": "R1", "camera_selected": "test:eyes:Camera:0",
    "wrist_camera_selected": "test:left_eef_link:Camera:0",
}


@pytest.fixture
def preview_http():
    backend = ESIPreviewBackend(dict(METADATA))
    frame = np.full((12, 16, 3), [30, 90, 150], dtype=np.uint8)
    backend.publish({"head_rgb": frame, "wrist_rgb": frame})

    def unexpected_submit(*args, **kwargs):
        raise AssertionError("HTTP write reached backend submission")

    backend.submit = unexpected_submit
    server = serve(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield backend, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_preview_ui_and_real_jpeg_snapshots(preview_http):
    backend, url = preview_http
    assert 'data-read-only="true"' in requests.get(url, timeout=2).text
    state = requests.get(url + "/api/state", timeout=2).json()
    assert state["read_only"] and state["physics_paused"]
    assert state["step_tokens"] == []
    for view in ("agentview", "wrist"):
        response = requests.get(url + "/snapshot/" + view, timeout=2)
        assert response.status_code == 200
        decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (12, 16, 3)
    with requests.get(url + "/stream/agentview", timeout=2, stream=True) as response:
        assert "multipart/x-mixed-replace" in response.headers["Content-Type"]
        assert b"--mjpeg-frame" in next(response.iter_content(128))
    backend.publish({"head_rgb": np.zeros((12, 16, 3), dtype=np.uint8)})
    state = requests.get(url + "/api/state", timeout=2).json()
    assert state["frame_seq"] == 2 and state["available_views"] == ["agentview"]
    assert requests.get(url + "/snapshot/wrist", timeout=2).status_code == 404


def test_debug_scene_is_unavailable_over_public_snapshot_and_stream_routes(preview_http):
    backend, url = preview_http
    backend.metadata["scene_camera"] = {"name": "ESI-viewer"}
    head = np.full((12, 16, 3), [200, 30, 40], dtype=np.uint8)
    scene = np.full((24, 32, 3), [30, 200, 40], dtype=np.uint8)
    backend.publish({"head_rgb": head, "scene_rgb": scene, "wrist_rgb": head})
    state = requests.get(url + "/api/state", timeout=2).json()
    assert state["camera_views"] == [] and state["camera_labels"]["agentview"] == "R1 HEAD"
    assert state["camera_names"]["agentview"] == METADATA["camera_selected"]
    assert requests.get(url + "/snapshot/scene", timeout=2).status_code == 404
    assert requests.get(url + "/stream/scene", timeout=2).status_code == 404
    for view in ("agentview", "head", "wrist"):
        response = requests.get(url + "/snapshot/" + view, timeout=2)
        decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == head.shape
        with requests.get(url + "/stream/" + view, timeout=2, stream=True) as response:
            assert response.status_code == 200
            assert b"--mjpeg-frame" in next(response.iter_content(128))
    assert backend.state() == state



@pytest.mark.parametrize("command", ["step", "move", "gripper", "start", "stop", "cancel", "task", "reset"])
def test_http_writes_cannot_reach_simulator(preview_http, command):
    backend, url = preview_http
    response = requests.post(url + "/api/" + command,
                             json={"token": "MV_FWD", "task": "changed"}, timeout=2)
    assert response.status_code == 405
    assert not response.json()["ok"]
    assert requests.get(url + "/api/state", timeout=2).json()["task_text"] == METADATA["task"]
    assert backend.frame_seq == 1


def test_non_preview_backend_still_accepts_actions():
    submitted = []
    backend = SimpleNamespace(
        dual_arm=False, STEP_TOKENS=("MV_FWD",), state=lambda: {},
        submit=lambda kind, arg=None, timeout=60: submitted.append((kind, arg)) or {"ok": True},
    )
    server = serve(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert 'data-read-only="false"' in requests.get(url, timeout=2).text
        assert requests.post(url + "/api/step", json={"token": "MV_FWD"}, timeout=2).status_code == 200
        assert submitted == [("step", ["MV_FWD"])]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_renderer_and_camera_reads_stay_on_main_thread():
    calls = []
    owner = threading.get_ident()

    def record():
        calls.append(threading.get_ident())

    def observe(*, include_wrist, include_scene=False, include_sensors=False):
        assert include_wrist
        record()
        return {"head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "wrist_rgb": np.ones((8, 8, 3), dtype=np.uint8)}

    session = SimpleNamespace(
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(pause=record, render=record))),
        observe=observe,
    )
    run_preview(session, METADATA, port=0, capture_fps=20, duration_s=0.08)
    assert len(calls) >= 4 and set(calls) == {owner}
    errors = []

    def wrong_thread():
        try:
            run_preview(session, METADATA)
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join(timeout=2)
    assert errors and "main thread" in errors[0]
    assert ESIPreviewBackend(METADATA).submit("move", "MV_FWD")["status"] == 405

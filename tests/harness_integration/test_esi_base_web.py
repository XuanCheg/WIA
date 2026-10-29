"""Real HTTP capability boundaries, queue cancellation and main-thread ownership."""
import threading
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from core.action_interface import ActionExecutor, action_result
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.sim.esi_web import run_esi_control
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_base import make_controller
from harness_integration.test_esi_preview import METADATA


@pytest.fixture
def base_http():
    backend = ESIControlBackend(dict(METADATA), create_r1_executor(make_controller()[0]))
    backend.publish({"head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                     "wrist_rgb": np.full((8, 8, 3), 50, dtype=np.uint8)})
    server = serve(backend, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield backend, f"http://127.0.0.1:{server.server_port}"
    finally:
        backend.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_http_legacy_actions_and_routes_never_reach_submission(base_http):
    backend, url = base_http

    def unexpected(*args, **kwargs):
        raise AssertionError("Forbidden request reached submission")

    backend.submit = unexpected
    for route in ("start", "stop", "cancel", "gripper", "task", "reset"):
        assert requests.post(url + "/api/" + route, json={"task": "changed"}, timeout=2).status_code == 405
    for token in ("MV_FWD", "GRASP", "RELEASE", "RT_YAW_CW", "ROTATE_CCW", "STAY"):
        assert requests.post(url + "/api/move", json={"token": token}, timeout=2).status_code == 405
        assert requests.post(url + "/api/step", json={"token": token}, timeout=2).status_code in (400, 405)
    assert requests.post(url + "/api/step", json={"tokens": ["BASE_FWD", "GRASP"]}, timeout=2).status_code == 400
    html = requests.get(url, timeout=2).text
    assert 'data-base-control="true"' in html and 'data-read-only="false"' in html
    state = requests.get(url + "/api/state", timeout=2).json()
    assert state["action_groups"] == {"base": True, "arm": False, "gripper": False}
    assert state["task_text"] == METADATA["task"] and not state["recording"]
    for view in ("agentview", "wrist"):
        assert requests.get(url + "/snapshot/" + view, timeout=2).status_code == 200


def test_allowed_http_tokens_queue_and_execute_only_on_main_thread(base_http):
    backend, url = base_http
    replies = []
    callers = []

    def request():
        replies.append(requests.post(url + "/api/step", json={"command": "BASE_TURN_LEFT*2 BASE_FWD"}, timeout=3))

    def execute(token, **kwargs):
        callers.append(threading.get_ident())
        backend.publish({"head_rgb": np.full((8, 8, 3), len(callers), dtype=np.uint8)})
        return action_result(token, completed=True, timeout=False, requested={"distance_m": 0.25},
                             actual={"distance_m": 0.25}, message="completed", reason="completed")

    backend.executor = ActionExecutor(execute, capabilities=backend.capabilities)
    thread = threading.Thread(target=request)
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    assert backend.process_next()
    thread.join(timeout=3)
    assert replies and replies[0].status_code == 200
    assert set(callers) == {threading.get_ident()} and len(callers) == 3
    assert backend.state()["history"] == ["BASE_TURN_LEFT", "BASE_TURN_LEFT", "BASE_FWD"]
    assert backend.state()["frame_seq"] == 4 and not backend.busy


def test_backend_rechecks_capability_and_expired_queue_does_not_execute():
    backend = ESIControlBackend(dict(METADATA), create_r1_executor(make_controller()[0]))
    assert backend.submit("move", "MV_FWD")["status"] == 405
    assert backend.submit("step", ["BASE_FWD", "RELEASE"])["status"] == 405
    assert backend.commands.empty()
    reply = []
    thread = threading.Thread(target=lambda: reply.append(backend.submit("move", "BASE_FWD", timeout=0.02)))
    thread.start()
    thread.join(timeout=1)
    assert reply[0]["status"] == 504

    def unexpected(*args, **kwargs):
        raise AssertionError("Expired queued action executed")

    backend.executor = ActionExecutor(unexpected, capabilities=backend.capabilities)
    assert backend.process_next()
    assert not backend.busy and backend.steps == 0


def test_shutdown_signal_survives_simulator_callback_and_restores_handlers():
    from harness_integration.test_esi_base import make_controller

    controller, robot, env = make_controller()
    original_step = env.step
    previous_handler = signal.getsignal(signal.SIGTERM)
    swallowed = []

    def native_callback_step(action):
        original_step(action)
        if len(env.actions) == 1:
            # Reproduce Kit catching exceptions from its Python callbacks.
            try:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            except KeyboardInterrupt:
                swallowed.append(True)

    env.step = native_callback_step
    controller.session.pipeline = SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=lambda: None)))
    controller.session.observe = lambda **kwargs: {"head_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                                                  "wrist_rgb": np.ones((8, 8, 3), dtype=np.uint8)}
    run_esi_control(controller.session, dict(METADATA), controller, create_r1_executor(controller),
                    port=0, capture_fps=20, duration_s=0.15)
    assert not swallowed
    assert len(env.actions) == 1 + controller.config.stop_steps
    assert np.all(env.actions[-1][robot.controller_action_idx["base"]] == 0)
    assert signal.getsignal(signal.SIGTERM) == previous_handler

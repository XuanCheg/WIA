"""Combined base/arm HTTP capabilities and simulator-owner queue execution."""
import threading

import numpy as np
import pytest
import requests

from core.action_units import BASE_ATOMS, MOVE_ATOMS
from core.action_interface import ActionExecutor, action_result
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_arm import make_arm_controller
from harness_integration.test_esi_preview import METADATA


@pytest.fixture
def arm_http():
    metadata = dict(METADATA, arm_control={"working_arm": "left", "config": {"translation_m": 0.02}})
    backend = ESIControlBackend(metadata, create_r1_executor(make_arm_controller()[0]))
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


def test_gripper_and_rotation_are_rejected_before_submission(arm_http):
    backend, url = arm_http

    def unexpected(*args, **kwargs):
        raise AssertionError("Forbidden request reached submission")

    backend.submit = unexpected
    for route in ("gripper", "start", "stop", "cancel", "task", "reset"):
        assert requests.post(url + "/api/" + route, json={}, timeout=2).status_code == 405
    for token in ("GRASP", "RELEASE", "RT_YAW_CW", "ROTATE_CW", "STAY", "ARM_DEBUG"):
        for route in ("move", "step"):
            assert requests.post(url + "/api/" + route, json={"token": token}, timeout=2).status_code in (400, 405)
    response = requests.post(url + "/api/step", json={"command": "BASE_FWD MV_UP GRASP"}, timeout=2)
    assert response.status_code == 400  # Reject the entire mixed batch.
    html = requests.get(url, timeout=2).text
    assert 'data-base-control="true"' in html and 'data-arm-control="true"' in html
    state = requests.get(url + "/api/state", timeout=2).json()
    assert state["action_groups"] == {"base": True, "arm": True, "gripper": False}
    assert state["step_tokens"] == list(BASE_ATOMS + MOVE_ATOMS)
    assert state["working_arm"] == "left" and state["arm_config"]["translation_m"] == 0.02
    for view in ("agentview", "wrist"):
        assert requests.get(url + "/snapshot/" + view, timeout=2).status_code == 200


def test_interleaved_base_arm_http_batch_executes_on_main_and_updates_frames(arm_http):
    backend, url = arm_http
    replies, callers = [], []
    tokens = ["BASE_TURN_LEFT", "MV_FWD", "MV_LEFT", "MV_DOWN", "MV_UP"]

    def request():
        replies.append(requests.post(url + "/api/step", json={"tokens": tokens}, timeout=3))

    def execute(token, **kwargs):
        callers.append(threading.get_ident())
        backend.publish({"head_rgb": np.full((8, 8, 3), len(callers), dtype=np.uint8),
                         "wrist_rgb": np.full((8, 8, 3), len(callers) + 30, dtype=np.uint8)})
        return action_result(token, completed=True, timeout=False, requested={"delta_m": 0.02},
                             actual={"delta_m": 0.02}, message="completed", reason="completed",
                             ee_pose_after={"base_position": [0.5, 0.2, 1.2]})

    backend.executor = ActionExecutor(execute, capabilities=backend.capabilities)
    thread = threading.Thread(target=request)
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    assert backend.process_next()
    thread.join(timeout=3)
    assert replies and replies[0].status_code == 200
    assert set(callers) == {threading.get_ident()} and len(callers) == len(tokens)
    state = requests.get(url + "/api/state", timeout=2).json()
    assert state["history"] == tokens and state["frame_seq"] == 1 + len(tokens)
    # No controller snapshot was published in this transport mock. Do not fill
    # current proprioception from the previous raw diagnostic result.
    assert state["ee_pos"] is None and not state["busy"]


def test_backend_cannot_queue_parameterized_debug_or_gripper():
    backend = ESIControlBackend(dict(METADATA), create_r1_executor(make_arm_controller()[0]))
    for token in ("GRASP", "RELEASE", "ARM_DEBUG", "MV_FWD(0.4)", "RT_YAW_CW"):
        assert backend.submit("move", token)["status"] == 405
    assert backend.submit("step", ["MV_UP", "RELEASE"])["status"] == 405
    assert backend.commands.empty()

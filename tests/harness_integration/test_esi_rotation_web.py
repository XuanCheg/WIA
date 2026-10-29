"""Rotation capability enforcement and measured feedback through HTTP and MCP."""
import asyncio
import threading
from dataclasses import asdict, replace

import numpy as np
import pytest
import requests

from core.action_units import WRIST_TILT_ATOMS
from core.action_units import BASE_ATOMS, MOVE_ATOMS, GRIPPER_ATOMS, ROTATE_ATOMS
from core.action_interface import ActionExecutor
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.sim.esi_rotation import ROTATION_FRAME, RotationControlConfig
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_gripper import make_manipulation_controller
from harness_integration.test_esi_base import make_controller
from harness_integration.test_esi_preview import METADATA


@pytest.fixture
def rotation_http():
    c, r, env = make_manipulation_controller()
    c.rotation_config = RotationControlConfig.load()
    metadata = dict(METADATA, arm_control={"working_arm": "left", "config": asdict(c.arm_config)},
                    rotation_control={"config": asdict(c.rotation_config), "frame": ROTATION_FRAME})
    backend = ESIControlBackend(metadata, create_r1_executor(c))
    backend.publish({"head_rgb": np.zeros((8, 8, 3), np.uint8),
                     "wrist_rgb": np.full((8, 8, 3), 50, np.uint8)}, controller_state=c.controller_state())
    http = serve(backend, host='127.0.0.1', port=0)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    try:
        yield c, env, backend, f'http://127.0.0.1:{http.server_port}'
    finally:
        backend.shutdown()
        http.shutdown()
        http.server_close()
        worker.join(timeout=2)


def test_rotation_capability_admits_local_tilt_but_not_world_axis_or_encoded_arguments(rotation_http):
    c, env, backend, url = rotation_http
    state = requests.get(url + '/api/state', timeout=2).json()
    assert state['rotation_enabled'] and state['rotation_frame'] == ROTATION_FRAME
    assert state['action_groups'] == {'base': True, 'arm': True, 'gripper': True}
    assert state['step_tokens'] == list(BASE_ATOMS + MOVE_ATOMS + GRIPPER_ATOMS + ROTATE_ATOMS + WRIST_TILT_ATOMS)
    assert 'data-rotation-control="true"' in requests.get(url, timeout=2).text
    for token in ('ROTATE_CW(90)', 'RT_YAW_CW', 'RT_PITCH_FWD', 'ARM_ROTATE_DEBUG'):
        response = requests.post(url + '/api/step', json={'tokens': ['ROTATE_CW', token]}, timeout=2)
        assert response.status_code in (400, 405) and not backend.busy
    assert backend.commands.empty() and not env.actions
    base = ESIControlBackend(backend.metadata, create_r1_executor(make_controller()[0]))
    assert not base.rotation_enabled and not base.action_allowed('ROTATE_CW')


def test_http_rotations_and_empty_close_return_execution_receipts(rotation_http):
    c, env, backend, url = rotation_http
    replies = []
    tokens = ['ROTATE_CW', 'MV_UP', 'ROTATE_CCW', 'GRASP', 'ROTATE_CW']
    env.available_object = False
    worker = threading.Thread(target=lambda: replies.append(requests.post(
        url + '/api/step', json={'tokens': tokens}, timeout=5)))
    worker.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    assert backend.process_next()
    worker.join(timeout=5)
    data = replies[0].json()
    assert replies[0].status_code == 200 and data['ok']
    assert [r['action'] for r in data['results']] == tokens
    assert data['results'][0]['execution_status'] == 'completed' and data['results'][0]['estimated_motion']['angle_deg'] == pytest.approx(15, abs=1)
    assert data['results'][3]['target_reached'] is True
    assert 'empty_grasp' not in replies[0].text
    assert set(env.threads) == {threading.get_ident()}


def test_mcp_rotation_context_and_failure_feedback_exclude_privileged_fields(rotation_http):
    pytest.importorskip('mcp')
    from mcp import Client
    from wia.runtime.web.mcp_server import create_server
    from harness_integration.test_robot_mcp import queued_call
    c, env, backend, url = rotation_http

    async def run():
        async with Client(create_server(url)) as client:
            context = (await client.call_tool('robot_get_context', {})).structured_content['context']
            assert context['steps']['arm_rotation_deg'] == 15 and context['rotation_frame'] == ROTATION_FRAME
            assert all(token in context['available_actions'] for token in ROTATE_ATOMS)
            result = await queued_call(client, backend, ['ROTATE_CW', 'ROTATE_CCW'], c.execute)
            assert not result.is_error
            assert result.structured_content['results'][0]['sources']['estimated_motion'] == 'joint_encoders_forward_kinematics'
            assert 'world_position' not in result.content[0].text and 'debug_gripper_after' not in result.content[0].text
            assert 'private_test_object' not in result.content[0].text
            env.arm_response = 0
            failure = await queued_call(client, backend, ['ROTATE_CW', 'MV_UP'], c.execute)
            assert failure.is_error and failure.structured_content['reason'] == 'arm_motion_blocked'
            assert failure.structured_content['unexecuted_actions'] == ['MV_UP']
            # Preflight rejects rotation when removed from the running backend's
            # whitelist even if its arm capability remains enabled.
            backend.executor = ActionExecutor(c.execute, capabilities=replace(backend.capabilities, ee_rotation=False))
            disabled = await client.call_tool('robot_execute_actions', {'tokens': ['ROTATE_CW']})
            assert disabled.is_error and disabled.structured_content['reason'] == 'action_disabled'
            assert backend.commands.empty()

    asyncio.run(run())


def test_lost_grasp_during_wrist_rotation_stops_and_reports_native_hold_state(rotation_http):
    c, env, backend, url = rotation_http
    assert c.execute('GRASP')['completed']
    env.drop_on_step = len(env.actions) + 1
    result = c.execute('ROTATE_CW')
    assert not result['completed'] and result['reason'] == 'grasp_lost'
    assert result['gripper_after']['holding'] is False and result['actual_deg'] < 15

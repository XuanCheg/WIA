"""Full R1 capabilities and safe grasp feedback through MCP/HTTP/owner queue."""
import asyncio
import threading

import numpy as np
import pytest
import requests

from core.action_units import BASE_ATOMS, MOVE_ATOMS, GRIPPER_ATOMS
from wia.runtime.sim.esi_actions import create_r1_executor
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_gripper import make_manipulation_controller
from harness_integration.test_esi_preview import METADATA


@pytest.fixture
def manipulation_http():
    c, robot, env = make_manipulation_controller()
    metadata = dict(METADATA, grasp_mode='assisted', image_shape=[8, 8, 3],
                    arm_control={'working_arm':'left', 'config':vars(c.arm_config)},
                    base_control={'config':vars(c.config)})
    backend = ESIControlBackend(metadata, create_r1_executor(c))
    def publish():
        backend.publish({'head_rgb':np.zeros((8,8,3),np.uint8),
                         'wrist_rgb':np.ones((8,8,3),np.uint8)}, controller_state=c.controller_state())
    c.on_step = publish
    publish()
    server = serve(backend, host='127.0.0.1', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield c, env, backend, f'http://127.0.0.1:{server.server_port}'
    finally:
        backend.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_full_manual_capabilities_keep_rotation_and_parameterized_commands_disabled(manipulation_http):
    c, env, backend, url = manipulation_http
    state = requests.get(url+'/api/state',timeout=2).json()
    assert state['action_groups'] == {'base':True,'arm':True,'gripper':True}
    assert state['step_tokens'] == list(BASE_ATOMS+MOVE_ATOMS+GRIPPER_ATOMS)
    assert state['grasp_mode'] == 'assisted' and state['holding'] is None and state['holding_estimate'] is None
    assert 'data-gripper-control="true"' in requests.get(url,timeout=2).text
    for token in ('RT_YAW_CW','GRASP(0.1)','MV_DOWN(0.5)','STOP'):
        assert requests.post(url+'/api/step',json={'tokens':['GRASP',token]},timeout=2).status_code in (400,405)
    assert backend.commands.empty() and not env.actions


def test_mcp_full_mobile_grasp_loop_and_idempotence_exclude_object_identity(manipulation_http):
    pytest.importorskip('mcp')
    from mcp import Client
    from wia.runtime.web.mcp_server import create_server
    from harness_integration.test_robot_mcp import queued_call
    c, env, backend, url = manipulation_http
    async def run():
        async with Client(create_server(url)) as client:
            context = (await client.call_tool('robot_get_context',{})).structured_content['context']
            assert context['grasp_mode'] == 'assisted' and len(context['available_actions']) == 14
            result = await queued_call(client,backend,
                ['GRASP','GRASP','MV_UP','BASE_LEFT','RELEASE','RELEASE'],c.execute)
            data = result.structured_content
            assert not result.is_error and data['completed'] and len(data['results']) == 6
            assert data['results'][0]['execution_status'] == 'completed'
            assert data['results'][0]['target_reached'] is False
            assert data['results'][1]['execution_status'] == 'completed'
            assert data['results'][-1]['target_reached'] is True
            assert 'private_test_object' not in result.content[0].text
            assert 'object_name' not in result.content[0].text and 'debug_object' not in result.content[0].text
            state = (await client.call_tool('robot_get_state',{})).structured_content['state']
            assert state['robot_state']['holding_estimate'] is None
            assert state['task_progress'] is None and not state['busy']
    asyncio.run(run())


def test_empty_grasp_reports_motor_completion_without_attachment_truth(manipulation_http):
    pytest.importorskip('mcp')
    from mcp import Client
    from wia.runtime.web.mcp_server import create_server
    from harness_integration.test_robot_mcp import queued_call
    c, env, backend, url = manipulation_http
    env.available_object = False
    async def run():
        async with Client(create_server(url)) as client:
            result = await queued_call(client,backend,['GRASP','MV_UP','BASE_FWD'],c.execute)
            data = result.structured_content
            assert not result.is_error and data['reason'] == 'completed'
            assert data['unexecuted_actions'] == []
            assert 'empty_grasp' not in result.content[0].text
            assert data['results'][0]['target_reached'] is True
            assert 'holding' not in data['results'][0]['estimated_motion']
            assert backend.steps == 3
    asyncio.run(run())

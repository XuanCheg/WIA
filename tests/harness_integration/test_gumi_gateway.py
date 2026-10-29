"""Real shared HTTP/MCP listener, group bundles and Agent input switches."""
import asyncio
import socket
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import requests

pytest.importorskip('mcp')
from mcp import Client
import uvicorn

from core.observation_config import ObservationConfig
from core.observations import CameraSnapshot
from core.viewer_config import ViewerConfig
from wia.runtime.web.http_gateway import create_gateway
from wia.runtime.web.web_teleop.esi_preview import ESIPreviewBackend
from wia.runtime.web.web_teleop.esi_viewer import ESIViewerBackend
from gumi.web_teleop.server import serve
from harness_integration.test_esi_preview import METADATA
from harness_integration.test_esi_viewer import snapshot


@pytest.fixture
def shared_gateway(tmp_path):
    frame = snapshot()
    source = SimpleNamespace(agent_observation=lambda: frame.observation)
    robot = ESIPreviewBackend(dict(METADATA), observation_source=source)
    viewer = ESIViewerBackend(robot, tmp_path, ViewerConfig())
    robot.publish(frame)
    viewer.publish(frame)
    backend = serve(viewer, host='127.0.0.1', port=0)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    app = create_gateway(f'http://127.0.0.1:{backend.server_port}').streamable_http_app(json_response=True)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='error'))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    for _ in range(300):
        if server.started:
            break
        time.sleep(0.01)
    assert server.started
    try:
        yield f'http://127.0.0.1:{port}', viewer, source, robot
    finally:
        viewer.close()
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        backend.shutdown()
        backend.server_close()
        backend_thread.join(timeout=2)


def test_one_listener_serves_gumi_groups_mcp_observation_and_recording(shared_gateway):
    url, viewer, _, _ = shared_gateway
    page = requests.get(url+'/gumi')
    assert page.status_code == 200 and page.url.endswith('/gumi/')
    assert 'data-human-viewer="true"' in page.text
    state = requests.get(url+'/gumi/api/state').json()
    assert [len(group['views']) for group in state['camera_groups']] == [1, 6, 4]
    bundle = requests.get(url+'/gumi/api/camera-group/global').json()
    assert len(bundle['images']) == 6 and bundle['frame_seq'] == viewer.frame_seq
    assert requests.get(url+'/api/state').json()['camera_groups'] == state['camera_groups']

    async def run():
        async with Client(url+'/mcp') as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "robot_get_context", "robot_get_state", "robot_get_camera", "robot_get_observation",
                "robot_execute_actions", "robot_execute_parameterized"}
            observed = await client.call_tool('robot_get_observation', {})
            assert not observed.is_error and len([item for item in observed.content if item.type == 'image']) == 2
            allowed = (await client.call_tool('robot_get_state', {})).structured_content['state']
            assert allowed['available_cameras'] == ['head', 'wrist'] and allowed['task_progress'] is None
            assert 'camera_groups' not in allowed and 'camera_catalog' not in allowed
            invalid = await client.call_tool('robot_get_camera', {'view': 'global_front'})
            assert invalid.is_error
            native = await client.call_tool('robot_get_camera', {'view': 'head'})
            assert not native.is_error and native.content[1].type == 'image'
    asyncio.run(run())
    assert requests.post(url+'/gumi/api/start', json={}).json()['ok']
    result = requests.post(url+'/gumi/api/stop', json={}).json()
    assert result['ok'] and {video['view'] for video in result['recording']['videos']} == {'esi', 'global', 'robot'}
    video = result['recording']['videos'][0]
    downloaded = requests.get(url+'/gumi'+video['url'])
    assert downloaded.status_code == 200 and len(downloaded.content) == video['size_bytes']


def test_merged_gumi_does_not_bypass_disabled_agent_rgb(shared_gateway):
    url, viewer, source, robot = shared_gateway
    previous = snapshot()
    observation = replace(previous.observation, agentview_rgb=None, wrist_rgb=None,
                          observation_config=ObservationConfig(head_rgb=False, wrist_rgb=False))
    frame = CameraSnapshot(observation, scene_rgb=previous.scene_rgb,
                           head_rgb=previous.observation.agentview_rgb,
                           wrist_rgb=previous.observation.wrist_rgb, human_rgb=previous.human_rgb)
    source.agent_observation = lambda: observation
    robot.publish(frame)
    viewer.publish(frame)
    assert requests.get(url+'/gumi/snapshot/head').status_code == 200
    assert requests.get(url+'/gumi/api/agent/snapshot/head').status_code in (404, 503)

    async def run():
        async with Client(url+'/mcp') as client:
            state = (await client.call_tool('robot_get_state', {})).structured_content['state']
            assert state['available_cameras'] == []
            result = await client.call_tool('robot_get_observation', {})
            assert not result.is_error and not [item for item in result.content if item.type == 'image']
            camera = await client.call_tool('robot_get_camera', {'view': 'head'})
            assert camera.is_error
    asyncio.run(run())


def test_unavailable_scene_keeps_browser_page_and_fails_agent_api(shared_gateway, monkeypatch):
    url, _, _, _ = shared_gateway
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError('Scene stopped for a transition')
    monkeypatch.setattr('wia.runtime.web.http_gateway.requests.request', unavailable)
    page = requests.get(url + '/gumi/', timeout=5)
    assert page.status_code == 200 and page.headers['Content-Type'].startswith('text/html')
    assert 'Waiting for a scene' in page.text and "fetch('/gumi/api/state'" in page.text
    state = requests.get(url + '/gumi/api/agent/state', timeout=5)
    assert state.status_code == 503 and state.json()['ok'] is False

"""Verify optional RGB transport. Run: python -m pytest tests/harness_integration/test_optional_rgb.py."""
import asyncio
import base64
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from core.agent.codex_session import CodexEpisodeSession, dynamic_depth_tools
from core.agent.decision import DecisionInput, ModelError
from core.agent.image_inputs import (OPTIONAL_RGB_CAMERAS, RGB_TOOL, image_inputs,
    rgb_tool_result, rgb_followup)
from core.agent.api_sessions import ChatCompletionsEpisodeSession, ResponsesEpisodeSession
from core.agent.local_http_session import LocalHTTPEpisodeSession
from harness_integration.test_api_sessions import session_fixture, mock_http, chat_reply, responses_reply
from harness_integration.test_local_http_episode import local_episode


def assert_pixels(url, expected):
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))))
    np.testing.assert_array_equal(image, expected)


def test_first_and_later_views_and_exact_optional_pixels(tmp_path):
    env, session = local_episode(tmp_path)
    obs = env.source.agent_observation()
    request = DecisionInput(obs, ('DONE',))
    assert len(session.request_images(request)) == 9
    session.turn_count = 1
    assert [x[0] for x in session.request_images(request)] == [
        'agentview_rgb', 'wrist_rgb', 'right_wrist_rgb', 'chassis_rear_rgb']
    for camera in OPTIONAL_RGB_CAMERAS:
        data, url = rgb_tool_result(obs, {'camera': camera})
        assert data['snapshot_id'] == obs.snapshot_id and data['ok']
        assert_pixels(url, obs.rgb_images()[camera])
    for args in ({'camera': 'initial_overview_rgb'}, {'camera': 'global'},
                 {'camera': OPTIONAL_RGB_CAMERAS[0], 'snapshot_id': 0}, {}, None):
        data, url = rgb_tool_result(obs, args)
        assert not data['ok'] and url is None


@pytest.mark.parametrize('kind', [ChatCompletionsEpisodeSession, ResponsesEpisodeSession, LocalHTTPEpisodeSession])
def test_requested_rgb_enters_actual_model_payload(tmp_path, monkeypatch, kind):
    if kind is LocalHTTPEpisodeSession:
        env, session = local_episode(tmp_path)
    else:
        env, session = session_fixture(tmp_path, kind)
    camera = OPTIONAL_RGB_CAMERAS[0]
    args = json.dumps({'camera': camera})
    calls = [{'id': 'rgb1', 'type': 'function', 'function': {'name': RGB_TOOL, 'arguments': args}}]
    response_mode = kind is ResponsesEpisodeSession
    first = ({'status': 'completed', 'output': [{'type': 'function_call', 'name': RGB_TOOL,
        'arguments': args, 'call_id': 'rgb1'}]} if response_mode else
        {'choices': [{'message': {'role': 'assistant', 'content': None, 'tool_calls': calls}, 'finish_reason': 'tool_calls'}]})
    final = '{"command":{"action":"DONE"},"world_observation":"Visible workspace."}'
    seen = mock_http(monkeypatch, [first, (responses_reply if response_mode else chat_reply)(final)])
    obs = env.source.agent_observation()
    session.turn_count = 1
    request = session.prepare_request(DecisionInput(obs, ('DONE',)))
    asyncio.run(session.decide(request))
    key = 'input' if response_mode else 'messages'
    image_type = 'input_image' if response_mode else 'image_url'
    assert sum(p['type'] == image_type for p in seen[0]['payload'][key][0]['content']) == 4
    items = seen[1]['payload'][key]
    assert items[-2].get('role', items[-2].get('type')) in ('tool', 'function_call_output')
    image = items[-1]['content'][-1]
    url = image['image_url'] if response_mode else image['image_url']['url']
    assert_pixels(url, obs.rgb_images()[camera])
    assert session.sensor_ids == [obs.snapshot_id]


def test_native_rgb_is_authorized_without_depth_and_rejects_stale_turn(tmp_path):
    env, _ = local_episode(tmp_path)
    obs = env.source.agent_observation()
    agent = CodexEpisodeSession(public_task={}, episode_configuration={'episode_id': 'episode-A'}, output=tmp_path)
    agent.session_id, agent._turn_id = 'thread', 'turn'
    agent._current_request = DecisionInput(obs, ('DONE',))
    agent._has_depth = False
    assert [t['name'] for t in dynamic_depth_tools(False)] == [RGB_TOOL]
    params = {'threadId': 'thread', 'turnId': 'turn', 'tool': RGB_TOOL,
              'arguments': {'camera': OPTIONAL_RGB_CAMERAS[0]}, 'callId': 'rgb1'}
    async def run():
        agent._completion = asyncio.get_running_loop().create_future()
        result = await agent._tool_call(params)
        assert result['success']
        assert_pixels(result['contentItems'][1]['imageUrl'], obs.rgb_images()[OPTIONAL_RGB_CAMERAS[0]])
        with pytest.raises(ModelError):
            await agent._tool_call(dict(params, turnId='previous'))
        agent._completion.set_result(None)
        with pytest.raises(ModelError):
            await agent._tool_call(params)
    asyncio.run(run())


def test_sliding_window_preserves_rgb_with_its_decision(tmp_path):
    _, session = session_fixture(tmp_path, provider='vllm', sliding_window_tokens=10000)
    def turn(name):
        return [{'role': 'user', 'content': name}, {'role': 'assistant', 'tool_calls': []},
                {'role': 'tool', 'content': name}, rgb_followup({'snapshot_id': name}, 'data:image/png;base64,AA=='),
                {'role': 'assistant', 'content': 'decision'}]
    initial, old, current = [turn(name) for name in ('initial', 'old', 'current')]
    messages = initial + old + current
    session._drop_oldest_window_turn(messages, 'test')
    assert messages == initial + current
    with pytest.raises(ModelError):
        session._drop_oldest_window_turn(messages, 'test')


def test_mcp_rgb_binding_advances_freezes_and_closes(tmp_path):
    from dataclasses import replace
    from core.agent.depth_binding import EpisodeDepthBinding
    from core.agent.native_runtime import EpisodeDepthMCP
    env, _ = local_episode(tmp_path)
    initial = env.source.agent_observation()
    next_obs = replace(initial, snapshot_id=initial.snapshot_id + 1)
    binding = EpisodeDepthBinding(tmp_path / 'mcp', 'test')
    client = EpisodeDepthMCP(tmp_path / 'mcp', 'test')
    args = {'camera': OPTIONAL_RGB_CAMERAS[0]}
    async def run():
        await client.start()
        try:
            for sensor, canonical, condition in ((initial, initial.snapshot_id, 'current'),
                    (next_obs, next_obs.snapshot_id, 'current'), (initial, next_obs.snapshot_id, 'initial_view_only')):
                binding.publish(sensor, canonical_snapshot_id=canonical, decision_step=1, observation_condition=condition)
                result = await client.call(RGB_TOOL, args)
                assert result.structured_content['snapshot_id'] == sensor.snapshot_id
                assert result.structured_content['ok']
                assert_pixels('data:image/png;base64,' + result.content[1].data, sensor.rgb_images()[args['camera']])
            binding.close()
            result = await client.call(RGB_TOOL, args)
            assert result.is_error and not result.structured_content['ok']
        finally:
            await client.close()
    asyncio.run(run())

"""Validate the local WIA conversation boundary without a GPU.

Run: python -B -m pytest -q tests/test_local_http_episode.py
"""
import asyncio
import json

import pytest

from core.agent.local_http_session import LocalHTTPEpisodeSession
from wia.runtime.agent.prompts import PromptAssembler
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.task_examples import load_example_task
from harness_integration.test_task_edit2_initial_overview import overview_source
from harness_integration.test_zero_shot_agent import Environment


def local_episode(tmp_path):
    task = load_example_task('find_and_lift')
    env = Environment()
    env.source, env.rig, _, _ = overview_source()
    env.source.begin_episode(task.task_text)
    policy = dict(episode_id='episode-A', max_steps=2, world_mode='pause',
                  world_advances_during_inference=False, observation_condition='current', action_mode='fixed')
    agent = LocalHTTPEpisodeSession(assembler=PromptAssembler(), public_task=task.public_payload, episode_configuration=policy,
        output=tmp_path, model='official-adapter', base_url='http://127.0.0.1:8000/v1')
    return env, agent


def reply_mock(monkeypatch, replies):
    seen = []
    outputs = iter(replies)

    def post(url, *, json, **kwargs):
        seen.append(json)
        text = next(outputs)

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {'choices': [{'message': {'content': text,
                    'reasoning_content': 'PRIVATE_REASONING_CANARY'}}], 'usage': {'completion_tokens': 3}}

        return Response()

    monkeypatch.setattr('core.agent.backends.requests.post', post)
    return seen


def test_complete_public_history_and_current_images_survive_two_decisions(tmp_path, monkeypatch):
    env, agent = local_episode(tmp_path)
    seen = reply_mock(monkeypatch, [
        '{"command":{"action":"BASE_FWD"},"world_observation":"A visible workspace."}',
        '{"command":{"action":"DONE"},"world_observation":"The workspace remains visible."}'])
    result = asyncio.run(run_episode(env, agent, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'model_done'
    assert env.calls == ['BASE_FWD']
    assert [len(payload['messages']) for payload in seen] == [1, 3]
    assert seen[1]['messages'][0] == seen[0]['messages'][0]
    first, second = seen[0]['messages'][0]['content'], seen[1]['messages'][-1]['content']
    assert sum(item['type'] == 'image_url' for item in first) == 9
    assert sum(item['type'] == 'image_url' for item in second) == 4
    assert 'Public task' in first[0]['text']
    assert 'Current measured robot state' in second[0]['text']
    assert all('response_format' not in payload for payload in seen)
    assert 'PRIVATE_REASONING_CANARY' not in json.dumps(agent.history)
    assert agent.provenance()['canonical_observation_ids'][0] != agent.provenance()['canonical_observation_ids'][1]


def test_bare_policy_tokens_are_measured_invalid_without_robot_motion(tmp_path, monkeypatch):
    env, agent = local_episode(tmp_path)
    seen = reply_mock(monkeypatch, ['MV_FWD', 'MV_FWD', 'MV_FWD'])
    result = asyncio.run(run_episode(env, agent, tmp_path, max_steps=2))
    assert result['termination_reason'] == 'invalid_output'
    assert result['model_calls'] == 3 and env.calls == []
    assert agent.sensor_ids[0] == agent.sensor_ids[1] == agent.sensor_ids[2]
    assert [len(payload['messages']) for payload in seen] == [1, 3, 5]
    assert all(payload['max_tokens'] == 1024 and payload['temperature'] == 0 for payload in seen)
    assert all('response_format' not in payload for payload in seen)


def test_private_fields_cannot_enter_public_local_episode(tmp_path):
    env, agent = local_episode(tmp_path)
    agent.public_task['private'] = 'PRIVATE_TARGET_CANARY'
    from core.agent.decision import DecisionInput
    request = DecisionInput(env.source.agent_observation(), ('DONE',))
    with pytest.raises(ValueError, match='public task'):
        agent.prepare_request(request)

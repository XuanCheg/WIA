"""Keep optional monitoring failures separate from robot action outcomes.

Run: python -B -m pytest -q tests/test_world_metrics_timeout.py
"""
import asyncio
import json

import pytest
import requests

from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_zero_shot_agent import Agent, Environment


@pytest.mark.parametrize('failed_read', [1, 2])
def test_timing_read_timeout_does_not_discard_or_replay_a_decision(tmp_path, monkeypatch, failed_read):
    class MetricsEnvironment(Environment):
        url = 'http://native-fixture'
        world_metrics = MCPRobotEnvironment.world_metrics

    reads = []

    def get(*args, **kwargs):
        reads.append(args)
        if len(reads) == failed_read:
            raise requests.ReadTimeout('Fixture timing read expired.')

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {'world_mode': 'run', 'idle_controller_env_steps': 7,
                        'private_canary': 'PRIVATE_EVALUATOR_CANARY'}

        return Response()

    monkeypatch.setattr('wia.runtime.agent.mcp_environment.requests.get', get)
    env = MetricsEnvironment()
    agent = Agent('{"action":"BASE_FWD"}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(env, agent, tmp_path, max_steps=2))
    assert summary['termination_reason'] == 'model_done'
    assert env.calls == ['BASE_FWD'] and len(agent.inputs) == 2
    steps = [json.loads(line) for line in (tmp_path / 'steps.jsonl').read_text().splitlines()]
    attempt = steps[0]['attempts'][0]
    failed = attempt['world_before_model' if failed_read == 1 else 'world_after_model']
    assert failed == {'world_metrics_validity': 'unavailable', 'world_metrics_error': 'read_timeout'}
    assert 'PRIVATE_EVALUATOR_CANARY' not in (tmp_path / 'steps.jsonl').read_text()
    assert all('world_metrics_error' not in request.prompt() for request in agent.inputs)


def test_http_error_remains_a_transport_failure(monkeypatch):
    class Response:
        def raise_for_status(self):
            raise requests.HTTPError('Fixture HTTP error.')

    monkeypatch.setattr('wia.runtime.agent.mcp_environment.requests.get', lambda *a, **kw: Response())
    env = MCPRobotEnvironment('http://native-fixture', run_id='test', task_text='Test.')
    with pytest.raises(requests.HTTPError):
        asyncio.run(env.world_metrics())

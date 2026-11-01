"""Check the fixed policy through the real runner. Run: python -m pytest -q tests/test_fixed_validation.py."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from mcp import Client
import pytest

from wia.runtime.agent.fixed_validation import FixedPolicy
from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
from wia.runtime.agent.zero_shot import run_episode
from wia.runtime.web.mcp_server import create_server
from harness_integration.test_agent_control import control_http, pump
from harness_integration.test_parameterized_actions import CAPABILITIES, RESOLUTION, LIMITS


def test_fixed_sequence_uses_runner_and_mcp_without_model_calls(control_http, monkeypatch, tmp_path):
    backend, _, url, calls, _ = control_http
    backend.executor.capabilities = CAPABILITIES
    backend.executor.primitive_resolution, backend.executor.macro_limits = RESOLUTION, LIMITS
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client',
                        lambda *args, **kwargs: Client(create_server(url)))
    sequence = json.loads((Path(__file__).resolve().parents[1]/'configs/native_scene_validation.json').read_text())
    finished = asyncio.Event()

    async def run():
        owner = asyncio.create_task(pump(backend, finished))
        try:
            async with MCPRobotEnvironment(url, run_id='fixed-policy-test', task_text='Approach the table.',
                                           action_mode='parameterized') as env:
                return await run_episode(env, FixedPolicy(sequence), tmp_path/'episode',
                                         max_steps=12, action_mode='parameterized')
        finally:
            finished.set()
            await owner

    summary = asyncio.run(run())
    assert summary['termination_reason'] == 'model_done'
    assert summary['model_calls'] == 0
    assert summary['executed_actions'] == [row['action'] for row in sequence['actions']]
    assert backend.steps == 8 and backend.control_owner == 'human'


def test_temporal_policy_transitions_without_consuming_next_motion():
    policy = FixedPolicy({'actions': [{'action': 'MV_UP', 'distance_m': .02}]})
    request = SimpleNamespace(action_context={'temporal_phase': 'phase_1'})
    reply = asyncio.run(policy.decide(request))
    assert json.loads(reply.raw_text) == {'action': 'DONE'} and policy.index == 0
    assert reply.metadata['survey_complete'] is True and reply.metadata['answer'] is None
    request.action_context['temporal_phase'] = 'phase_2'
    assert json.loads(asyncio.run(policy.decide(request)).raw_text)['action'] == 'MV_UP'
    request.observation = SimpleNamespace(task_text='Original answer options: ["yes", "no"]')
    reply = asyncio.run(policy.decide(request))
    assert reply.metadata['answer'] == 'yes'


def test_fixed_confirmation_can_precede_done_in_another_decision():
    policy = FixedPolicy({'actions': [{'action': 'BASE_FWD', 'distance_m': .05}],
                          'answers': {'0': 'inside'}})
    request = SimpleNamespace(action_context={})
    first = asyncio.run(policy.decide(request))
    final = asyncio.run(policy.decide(request))
    assert first.metadata['answer'] == 'inside'
    assert json.loads(first.raw_text)['action'] == 'BASE_FWD'
    assert json.loads(final.raw_text)['action'] == 'DONE'
    assert final.metadata['world_observation'] == 'Fixed diagnostic sequence.'
    assert final.metadata['answer'] is None


@pytest.mark.parametrize('alias', ['t40', 't41', 't45', 't46', 't47', 't48'])
def test_current_physical_temporal_manifest_preserves_authored_survey_budget(alias):
    from wia.tasks.loader import load_task
    root = Path(__file__).resolve().parents[1]
    task = load_task(root/f'explore_to_achieve/wia_v1/release/tasks/{alias}.yaml',
                     variant='adapted', seed=0, run_mode='benchmark')
    parameters = task.definition.private.evaluation.parameters
    policy = task.manifest('temporal-budget-check')['temporal_policy']
    assert policy['phase_1_max_decisions'] == parameters['phase_1_max_decisions'] == 10

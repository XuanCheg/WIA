"""Check public scheduling time. Run: python -m pytest -q tests/test_simulation_clock.py."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
from wia.runtime.agent.prompts import PromptAssembler
from wia.runtime.web.mcp_server import create_server
from mcp import Client
from harness_integration.test_agent_control import control_http, pump
from harness_integration.test_milestone85_prompts import request_inputs


def test_m4_clock_matches_capture_survives_wait_and_resets_per_episode(control_http, monkeypatch):
    backend, _, url, _, rig = control_http
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client', lambda *args, **kwargs: Client(create_server(url)))
    rig.loaded_task = SimpleNamespace(definition=SimpleNamespace(private=SimpleNamespace(
        evaluation=SimpleNamespace(parameters={'prototype': 'M4'}))))
    sim = rig.pipeline.og.sim
    sim.current_time = 100.
    finished = asyncio.Event()
    prompt_task, prompt_request, prompt_policy = request_inputs()

    async def run():
        owner = asyncio.create_task(pump(backend, finished))
        try:
            async with MCPRobotEnvironment(url, run_id='clock-test', task_text='Approach the table.') as env:
                first = await env.observe()
                clock = dict(env.action_context['simulation_clock'])
                assert clock['elapsed_simulation_s'] == 0
                sim.current_time = 112.
                # Reads during model waits retain the camera's capture time.
                await env.observe()
                assert env.action_context['simulation_clock'] == clock
                await env.execute('BASE_FWD')
                second = await env.observe()
                clock = env.action_context['simulation_clock']
                assert clock['elapsed_simulation_s'] == 12.
                assert clock['snapshot_id'] == second.snapshot_id > first.snapshot_id
                assert clock['action_step'] == second.action_step == 1
                assert clock['captured_at'] == second.captured_at
                request = replace(prompt_request, action_context=env.action_context)
                rendered = PromptAssembler().build(request, prompt_task.public_payload, prompt_policy, first_turn=False).prompt()
                assert 'elapsed_simulation_s' in rendered and '12.0' in rendered
                assert 'not wall time or a live timer' in rendered
                def reject_survey(*args, **kwargs):
                    raise ValueError('Unmet survey prerequisite.')
                monkeypatch.setattr(backend.observation_source, 'report_answer', reject_survey)
                receipt = await env.report_survey(second)
                assert receipt['ok'] is False and backend.control_owner == 'agent'
                backend.observation_source.simulation_clock = dict(clock, snapshot_id=-1)
                with pytest.raises(RuntimeError, match='does not match'):
                    await env.observe()
            sim.current_time = 500.
            async with MCPRobotEnvironment(url, run_id='clock-next', task_text='Approach the table.') as env:
                await env.observe()
                assert env.action_context['simulation_clock']['elapsed_simulation_s'] == 0.
        finally:
            finished.set()
            await owner

    asyncio.run(run())

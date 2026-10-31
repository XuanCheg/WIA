"""Check cancellation resume without changing the original decision budget.

Run: python -B -m pytest -q tests/test_cancelled_resume_budget.py
"""
import asyncio
import json

import pytest
from mcp import Client

from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
from wia.runtime.agent.zero_shot import run_episode
from wia.tasks.loader import load_task
from wia.runtime.web.mcp_server import create_server
import wia.runtime.commands.run_esi_agent as runner
from harness_integration.task_examples import example_task_path
from harness_integration.test_agent_control import control_http, pump  # noqa: F401
from harness_integration.test_zero_shot_agent import Agent, Environment


class CancelAfterAction(Agent):
    async def decide(self, request):
        if self.inputs:
            self.inputs.append(request)
            raise asyncio.CancelledError()
        return await super().decide(request)


def test_cancelled_same_budget_resumes_cumulative_history_without_reset(control_http, monkeypatch, tmp_path):
    backend, _, url, calls, _ = control_http
    contract = {"episode_id": "retained-task", "action_mode": "fixed",
                "world_advances_during_inference": True}
    backend.metadata["episode_contract"] = contract
    monkeypatch.setattr("wia.runtime.agent.mcp_environment.Client", lambda *args, **kwargs: Client(create_server(url)))
    manifest = dict(contract, max_steps=4)
    original_manifest = dict(manifest)

    async def exercise():
        completed = asyncio.Event()
        worker = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id="retained-task",
                    task_text=backend.observation_source.task_text, episode_contract=contract) as environment:
                with pytest.raises(asyncio.CancelledError):
                    await run_episode(environment, CancelAfterAction('{"action":"BASE_FWD"}'),
                        tmp_path / "previous", max_steps=4, episode_manifest=manifest)
            first = json.loads((tmp_path / "previous/summary.json").read_text())
            assert first["termination_reason"] == "cancelled"
            assert first["model_calls"] is None
            assert first["agent_decisions"] == 1 and first["decision_calls"] == 2
            history = list(backend.history)

            def forbidden_reset(*args, **kwargs):
                pytest.fail("Cancellation resume must preserve the retained episode")

            monkeypatch.setattr(backend.observation_source, "begin_episode", forbidden_reset)
            agent = Agent('{"action":"BASE_FWD"}', '{"action":"DONE"}')
            async with MCPRobotEnvironment(url, run_id="retained-task",
                    task_text=backend.observation_source.task_text, episode_contract=contract,
                    resume_action_step=1) as environment:
                assert backend.steps == 1 and backend.history == history
                result = await run_episode(environment, agent, tmp_path / "resumed", max_steps=4,
                    resume_summary=first, expected_action_step=1, episode_manifest=manifest)
            assert result["termination_reason"] == "model_done"
            assert result["max_steps"] == first["max_steps"] == 4
            assert result["agent_decisions"] == 3 and result["decision_calls"] == 4
            assert result["resume_from_decision_calls"] == first["decision_calls"]
            assert result["started_at"] == first["started_at"]
            assert result["previous_active_wall_time_s"] == first["timings"]["total_episode_wall_time"]
            assert result["episode_manifest"] == first["episode_manifest"] == original_manifest
            assert result["budget_changes"] == [] and manifest == original_manifest
            assert result["simulator_action_step_offset"] == 0
            assert calls == ["BASE_FWD", "BASE_FWD"] and backend.steps == 2
            assert [request.decision_step for request in agent.inputs] == [2, 3]
            assert [request.steps_remaining for request in agent.inputs] == [3, 2]
            assert agent.inputs[0].observation.action_history[-1].action == "BASE_FWD"
            steps = [json.loads(line) for line in (tmp_path / "resumed/steps.jsonl").read_text().splitlines()]
            assert [row["step"] for row in steps] == [2, 3]
        finally:
            completed.set()
            await worker

    asyncio.run(exercise())
    assert backend.control_owner == "human"


@pytest.fixture
def cancelled_cli_case(tmp_path, monkeypatch):
    task = load_task(example_task_path("find_and_lift"), variant="visible_target_control", seed=0)
    manifest = task.manifest("retained-episode")
    manifest.update(max_steps=60, world_mode="run", world_advances_during_inference=True,
        observation_condition="current", action_mode="parameterized", model_name="gpt-6-astra",
        reasoning_effort="medium", wall_timeout_s=5400., agent_session="native_persistent",
        world_observation_log=True)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    previous = tmp_path / "previous"
    (previous / "step_0059/before").mkdir(parents=True)
    (previous / "step_0059/before/observation.json").write_text(json.dumps({"action_step": 58}))
    session = {"codex_session_id": "original-thread", "turns": 59,
               "model_sensor_snapshot_ids": [1, 2], "canonical_observation_ids": [1, 2],
               "native_context_compaction_events": [{"turn": 40}], "usage": {"input_tokens": 123}}
    summary = {"episode_manifest": manifest, "termination_reason": "cancelled", "agent_decisions": 58,
        "max_steps": 60, "native_session": session, "max_duration_s": 5400.,
        "agent": {"model": "gpt-6-astra", "reasoning_effort": "medium"},
        "action_mode": "parameterized", "world_mode": "run", "observation_condition": "current"}
    summary_path = previous / "summary.json"
    summary_path.write_text(json.dumps(summary))
    captured = {}

    def native(**kwargs):
        captured["native"] = kwargs
        return object()

    class RetainedEnvironment:
        def __init__(self, *args, **kwargs):
            captured["environment"] = kwargs

        async def __aenter__(self):
            raise RuntimeError("fixture stops before control acquisition")

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr("core.agent.codex_session.CodexEpisodeSession", native)
    monkeypatch.setattr(runner, "MCPRobotEnvironment", RetainedEnvironment)
    arguments = ["--task-file", str(task.path), "--variant", task.variant, "--seed", "0",
                 "--episode-manifest", str(manifest_path), "--resume-from", str(previous),
                 "--output-dir", str(tmp_path / "continued")]
    return arguments, summary, summary_path, manifest_path, captured


def test_cancelled_cli_preserves_native_thread_state_manifest_and_cap(cancelled_cli_case):
    arguments, summary, _, manifest_path, captured = cancelled_cli_case
    manifest_bytes = manifest_path.read_bytes()
    with pytest.raises(SystemExit) as error:
        runner.main(arguments)
    assert error.value.code == 1  # The CPU fixture stops before acquiring live control.
    assert captured["native"]["resume_state"] == summary["native_session"]
    assert captured["native"]["episode_configuration"]["max_steps"] == 60
    assert captured["environment"]["resume_action_step"] == 58
    assert captured["environment"]["episode_contract"]["max_steps"] == 60
    assert manifest_path.read_bytes() == manifest_bytes


def test_cancelled_cli_rejects_unauthorized_budget_extension(cancelled_cli_case):
    arguments, _, _, _, captured = cancelled_cli_case
    with pytest.raises(SystemExit) as error:
        runner.main(arguments + ["--max-steps", "120"])
    assert error.value.code == 2 and captured == {}


@pytest.mark.parametrize("reason", ["max_steps", "model_done", "environment_success", "timeout"])
def test_completed_done_and_timeout_cli_cannot_resume(cancelled_cli_case, reason):
    arguments, summary, summary_path, _, captured = cancelled_cli_case
    summary["termination_reason"] = reason
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(SystemExit) as error:
        runner.main(arguments)
    assert error.value.code == 2 and captured == {}


@pytest.mark.parametrize("reason,max_steps", [
    ("cancelled", 5), ("max_steps", 4), ("model_done", 4),
    ("environment_success", 4), ("timeout", 4),
])
def test_runner_still_rejects_new_cap_and_completed_episodes(tmp_path, reason, max_steps):
    environment, agent = Environment(), Agent()
    previous = {"termination_reason": reason, "max_steps": 4, "agent_decisions": 1}
    with pytest.raises(ValueError, match="Only an interrupted"):
        asyncio.run(run_episode(environment, agent, tmp_path, max_steps=max_steps, resume_summary=previous))
    assert agent.inputs == [] and environment.calls == []

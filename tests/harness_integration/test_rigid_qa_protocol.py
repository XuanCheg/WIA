"""Check shared containment semantics and native stopping.

Run .venv/bin/python -m pytest -q tests/test_rigid_qa_protocol.py.
"""
import asyncio
import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest

from core.agent.backends import structured_reply
from core.agent.decision import DecisionInput, ModelReply
from wia.runtime.agent.zero_shot import run_episode
from wia.tasks.rigid_qa import RigidContainmentQA
from harness_integration.test_tabletop_scenes import manifest
from harness_integration.test_zero_shot_agent import Environment


def test_local_cpu_protocol_matches_official_task_definitions():
    from wia.tasks.rigid_qa import PROTOCOL_PATH

    upstream = __import__('wia.resources', fromlist=['ROOT']).ROOT / "ESI/src/active_explore/tasks/physical_structure/rigid_containment.py"
    def definitions(path):
        result = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef):
                result[node.name] = ast.dump(node, include_attributes=False)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                result[node.targets[0].id] = ast.dump(node, include_attributes=False)
        return result
    original = definitions(upstream)
    for name, definition in definitions(PROTOCOL_PATH).items():
        assert definition == original[name], name


@pytest.fixture
def qa(tmp_path):
    scene, _ = manifest(tmp_path)
    return RigidContainmentQA(scene)


def test_source_rules_and_private_projection(qa):
    assert "any part of it is inside and it is stable" in qa.public_prompt
    assert "sitting completely on top" in qa.public_prompt
    assert "choose the smaller suitable one" in qa.public_prompt
    assert "prefer physical tests over camera movement" in qa.public_prompt
    assert "confidence >= 0.80" in qa.public_prompt
    assert "pick up <object>" not in qa.public_prompt
    serialized = json.dumps(qa.public_payload)
    for private in ("_ground_truth", "target_container", "bbox_inventory", "metadata_json", "initial_poses"):
        assert private not in serialized


@pytest.mark.parametrize("answer,confidence,action,step,expected", [
    ("medium container", .1, "DONE", 1, (True, "model_stop")),
    ("medium container", .8, "BASE_FWD", 1, (True, "confidence_threshold")),
    ("medium container", .79, "BASE_FWD", 1, (False, "")),
    ("unsure", .9, "BASE_FWD", 1, (False, "")),
    ("unsure", .1, "DONE", 1, (False, "")),
    ("unsure", .1, "DONE", 30, (True, "max_steps")),
    ("medium container", .1, "BASE_FWD", 30, (True, "max_steps")),
])
def test_exact_upstream_stopping_rules(qa, answer, confidence, action, step, expected):
    parsed = qa.parse({"answer": answer, "confidence": confidence, "reasoning": "Observed evidence."}, action)
    assert qa.should_stop(parsed, [], step, 30) == expected


def test_qa_transport_retains_answer_and_executes_only_command(qa):
    request = DecisionInput(Environment().source.agent_observation(), ("BASE_FWD", "DONE"),
                            world_observation_log=True, qa_response=True)
    payload = {"command": {"action": "BASE_FWD"}, "world_observation": "A box is visible.",
               "answer": "unsure", "confidence": .2, "reasoning": "The opening is occluded."}
    reply = structured_reply(request, json.dumps(payload), {})
    assert json.loads(reply.raw_text) == payload["command"]
    assert reply.metadata["qa_response"]["confidence"] == .2
    assert request.structured_schema["required"] == list(payload)


class QAAgent:
    def __init__(self, qa, replies):
        self.qa_protocol, self.replies, self.calls = qa, iter(replies), 0

    def description(self):
        return {"backend": "test"}

    async def decide(self, request):
        self.calls += 1
        output = next(self.replies)
        if isinstance(output, Exception):
            raise output
        action, answer, confidence = output
        return ModelReply(json.dumps({"action": action}), metadata={"qa_response": {
            "answer": answer, "confidence": confidence, "reasoning": "Observed evidence."}})

    def prepare_force_choice(self, request):
        return replace(request, assembled_prompt=self.qa_protocol.force_choice_prompt())


def test_threshold_stops_before_motion(qa, tmp_path):
    env = Environment()
    agent = QAAgent(qa, [("BASE_FWD", "medium container", .8)])
    result = asyncio.run(run_episode(env, agent, tmp_path / "episode", max_steps=3))
    assert result["qa_stop_reason"] == "confidence_threshold"
    assert result["qa_final_answer"] == "medium container"
    assert env.calls == [] and agent.calls == 1


def test_budget_skips_last_motion_and_forces_one_selection_in_same_agent(qa, tmp_path):
    env = Environment()
    agent = QAAgent(qa, [("DONE", "unsure", .1), ("BASE_FWD", "unsure", .1),
                         ("DONE", "medium container", .3)])
    result = asyncio.run(run_episode(env, agent, tmp_path / "episode", max_steps=2))
    assert result["qa_stop_reason"] == "max_steps"
    assert result["qa_final_answer"] == "medium container"
    assert result["qa_forced_choice"]["confidence"] == .3
    assert env.calls == [] and agent.calls == result["model_calls"] == 3


@pytest.fixture
def combined(tmp_path):
    scene, _ = manifest(tmp_path)
    return RigidContainmentQA(scene, goal="answer_and_place")


def test_combined_confidence_does_not_suppress_physical_action(combined, tmp_path):
    env = Environment()
    agent = QAAgent(combined, [("BASE_FWD", "medium container", .8),
                               ("DONE", "medium container", .8)])
    result = asyncio.run(run_episode(env, agent, tmp_path / "episode", max_steps=60))
    assert env.calls == ["BASE_FWD"]
    assert result["agent_decisions"] == 2 and result["model_calls"] == 2
    assert result["qa_stop_reason"] == "model_stop"
    assert result["task_success"] is None
    records = [json.loads(line) for line in (tmp_path / "episode/steps.jsonl").read_text().splitlines()]
    assert records[0]["qa_subtask_stop_reason"] == "confidence_threshold"
    assert "model_requested_action" not in records[0]


def test_combined_unsure_done_ends_attempt_and_forces_final_answer(combined, tmp_path):
    env = Environment()
    agent = QAAgent(combined, [("DONE", "unsure", .1), ("DONE", "large container", .3)])
    result = asyncio.run(run_episode(env, agent, tmp_path / "episode", max_steps=60))
    assert env.calls == [] and agent.calls == 2
    assert result["agent_decisions"] == 1
    assert result["qa_final_answer"] == "large container"
    assert result["qa_answer_step"] == 2
    assert result["qa_stop_reason"] == "model_stop"


def test_combined_budget_retains_final_nonmotion_decision(combined, tmp_path):
    env = Environment()
    agent = QAAgent(combined, [("BASE_FWD", "unsure", .1), ("BASE_FWD", "medium container", .4)])
    result = asyncio.run(run_episode(env, agent, tmp_path / "episode", max_steps=2))
    assert env.calls == ["BASE_FWD"] and agent.calls == 2
    assert result["qa_stop_reason"] == "max_steps"
    assert result["qa_final_answer"] == "medium container"


def test_combined_prompt_has_both_goals_and_no_qa_only_early_stop(combined):
    from wia.runtime.agent.prompts import PromptAssembler
    from harness_integration.test_milestone85_prompts import request_inputs
    _, request, policy = request_inputs()
    policy["max_steps"] = 60
    request = replace(request, observation=replace(request.observation, task_text=combined.public_payload["objective"]),
                      steps_remaining=60)
    assembler = PromptAssembler(task_mode="original_qa", qa_protocol=combined)
    prompt = assembler.build(request, combined.public_payload, policy, first_turn=True)
    assert "physically pick up" in prompt.prompt()
    assert "continue native robot actions" in prompt.prompt()
    assert "does not end robot execution" in prompt.prompt()
    assert "completion does not require physically placing" not in prompt.prompt()
    assert "stops BEFORE any proposed motion" not in prompt.prompt()
    assert "target_container" not in prompt.prompt()
    assert prompt.structured_schema["required"] == ["command", "world_observation", "answer", "confidence", "reasoning"]


def test_native_error_retains_code_without_retrying_a_terminated_turn(tmp_path):
    from core.agent.codex_session import CodexEpisodeSession
    from core.agent.decision import ModelError

    async def check():
        agent = CodexEpisodeSession(public_task={}, episode_configuration={"episode_id": "test"}, output=tmp_path)
        agent.session_id = "test-thread"
        agent._completion = asyncio.get_running_loop().create_future()
        error = {"message": "Connection failed.", "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 503}},
                 "additionalDetails": "PRIVATE_DIAGNOSTIC_CANARY"}
        params = {"threadId": "test-thread", "turnId": "test-turn", "error": error, "willRetry": True}
        agent._notification("error", params)
        assert not agent._completion.done()
        agent._notification("error", dict(params, willRetry=False))
        with pytest.raises(ModelError, match="responseStreamDisconnected"):
            await agent._completion
        assert len(agent.runtime_errors) == 2
        saved = (tmp_path / "native_runtime_errors.jsonl").read_text()
        assert "503" in saved and "PRIVATE_DIAGNOSTIC_CANARY" not in saved

    asyncio.run(check())


def test_resume_continues_budget_and_history_without_replaying_motion(combined, tmp_path):
    from core.agent.decision import ModelError
    env = Environment()
    interrupted = QAAgent(combined, [("BASE_FWD", "unsure", .2), ("BASE_FWD", "unsure", .2),
                                     ModelError("serverOverloaded")])
    first = asyncio.run(run_episode(env, interrupted, tmp_path / "first", max_steps=4))
    assert first["agent_decisions"] == 2 and first["model_calls"] == 3
    resumed = QAAgent(combined, [("BASE_FWD", "unsure", .3), ("DONE", "medium container", .4)])
    result = asyncio.run(run_episode(env, resumed, tmp_path / "resumed", max_steps=4,
                                    resume_summary=first, expected_action_step=2))
    assert env.calls == ["BASE_FWD"] * 3
    assert result["agent_decisions"] == 4 and result["model_calls"] == 5
    assert result["qa_final_answer"] == "medium container"
    assert [r["step"] for r in result["qa_history"]] == [1, 2, 3, 4]
    records = [json.loads(x) for x in (tmp_path / "resumed/steps.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [3, 4]
    assert result["previous_active_wall_time_s"] == first["timings"]["total_episode_wall_time"]


def test_native_resume_preserves_thread_history_and_uses_fresh_phase(combined, tmp_path):
    from core.agent.codex_session import CodexEpisodeSession
    from wia.runtime.agent.prompts import PromptAssembler
    from harness_integration.test_milestone85_prompts import request_inputs
    _, request, policy = request_inputs()
    policy["max_steps"] = 60
    assembler = PromptAssembler(task_mode="original_qa", qa_protocol=combined)
    state = dict(assembler.provenance(), episode_id=policy["episode_id"], model_name="gpt-6.1-sol",
                 reasoning_effort="high", turns=29, model_sensor_snapshot_ids=list(range(29)),
                 canonical_observation_ids=list(range(29)), native_context_compaction_events=[], usage={},
                 codex_session_id="original-thread")
    agent = CodexEpisodeSession(public_task=combined.public_payload, episode_configuration=policy,
                               output=tmp_path, model="gpt-6.1-sol", assembler=assembler, resume_state=state)
    request = replace(request, observation=replace(request.observation, task_text=combined.public_payload["objective"]),
                      steps_remaining=32)
    prepared = agent.prepare_request(request)
    assert agent.turn_count == 29
    assert prepared.prompt_phase == "decision_step"
    assert "Public task:" not in prepared.prompt() and "Remaining decision budget:\n32" in prepared.prompt()


def test_explicit_budget_extension_preserves_decisions_and_allows_new_motion(combined, tmp_path):
    from core.agent.decision import ModelError
    env = Environment()
    interrupted = QAAgent(combined, [("BASE_FWD", "unsure", .2), ModelError("Interrupted for budget update.")])
    first = asyncio.run(run_episode(env, interrupted, tmp_path / "first", max_steps=2))
    first["termination_reason"] = "cancelled"
    resumed = QAAgent(combined, [("BASE_FWD", "unsure", .2), ("BASE_FWD", "unsure", .3),
                                 ("DONE", "medium container", .4)])
    result = asyncio.run(run_episode(env, resumed, tmp_path / "extended", max_steps=4,
                                    resume_summary=first, expected_action_step=1, budget_extension=True))
    assert result["max_steps"] == 4 and result["agent_decisions"] == 4
    assert env.calls == ["BASE_FWD"] * 3
    assert [r["step"] for r in result["qa_history"]] == [1, 2, 3, 4]
    assert result["budget_changes"] == [{"previous_max_steps": 2, "max_steps": 4,
                                        "at_decision": 2, "reason": "user_requested_extension"}]


def test_budget_change_requires_explicit_extension(combined, tmp_path):
    from core.agent.decision import ModelError
    env = Environment()
    interrupted = QAAgent(combined, [("BASE_FWD", "unsure", .2), ModelError("serverOverloaded")])
    first = asyncio.run(run_episode(env, interrupted, tmp_path / "first", max_steps=2))
    for extend, budget in [(False, 4), (True, 2), (True, 1)]:
        with pytest.raises(ValueError, match="remaining budget"):
            asyncio.run(run_episode(env, QAAgent(combined, []), tmp_path / f"invalid_{extend}_{budget}",
                                    max_steps=budget, resume_summary=first, budget_extension=extend))


def test_resume_keeps_the_cumulative_active_time_limit(combined, tmp_path):
    from core.agent.decision import ModelError
    env = Environment()
    first = asyncio.run(run_episode(env, QAAgent(combined, [ModelError("serverOverloaded")]),
                                   tmp_path / "first", max_steps=2, max_duration_s=10))
    first["timings"]["total_episode_wall_time"] = 11
    result = asyncio.run(run_episode(env, QAAgent(combined, []), tmp_path / "resume",
                                    max_steps=2, max_duration_s=10, resume_summary=first))
    assert result["termination_reason"] == "timeout"
    assert result["agent_decisions"] == 0 and not env.calls

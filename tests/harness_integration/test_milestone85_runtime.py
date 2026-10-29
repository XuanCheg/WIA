"""Native adapter safety with deterministic RPC fixtures. Run pytest -q this_file.

These fixtures verify Python boundaries; the real Codex evidence is produced by
scripts/verify_milestone85_native.py rather than inferred from these tests.
"""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from core.agent.codex_session import CodexEpisodeSession
from core.agent.decision import ModelError
from core.agent.depth_binding import EpisodeDepthBinding
from core.agent.depth_inputs import depth_access
from wia.runtime.agent.zero_shot import run_episode
from harness_integration.test_zero_shot_agent import Environment
from harness_integration.test_milestone85_prompts import request_inputs
from harness_integration.test_agent_control import control_http  # noqa: F401
from harness_integration.task_examples import example_task_path


class RPCFixture:
    failure = None

    def __init__(self, agent, outputs):
        self.agent, self.outputs = agent, iter(outputs)
        self.calls = []

    async def request(self, method, params, **kwargs):
        self.calls.append((method, params))
        if method == "turn/start":
            turn = {"id": "turn-" + str(self.agent.turn_count), "status": "completed"}
            if self.outputs is not None:
                self.agent._notification("item/completed", {"threadId": self.agent.session_id,
                    "item": {"id": turn["id"], "type": "agentMessage", "text": next(self.outputs)}})
                self.agent._notification("turn/completed", {"turn": turn})
            return {"turn": turn}
        return {}

    async def close(self):
        pass


def fixture_session(tmp_path, monkeypatch, outputs, *, timeout=1):
    task, request, policy = request_inputs()
    agent = CodexEpisodeSession(public_task=task.public_payload, episode_configuration=policy,
                               output=tmp_path / "episode", timeout_s=timeout)
    opened = []

    async def open_once(request):
        opened.append(True)
        agent.root = tmp_path / "neutral-native"
        agent.root.mkdir()
        agent.session_id = "exact-native-session-id"
        agent._has_depth = depth_access(request.sensor_inputs)["available"]
        agent._binding = EpisodeDepthBinding(agent.root / "depth", agent.episode_id)
        agent._rpc = RPCFixture(agent, outputs or [])
        if outputs is None:
            agent._rpc.outputs = None

    monkeypatch.setattr(agent, "_open", open_once)
    return task, request, agent, opened


def test_native_adapter_retry_has_one_session_one_action_and_fresh_later_context(tmp_path, monkeypatch):
    from wia.runtime.agent.prompts import PromptAssembler
    from harness_integration.test_task_edit2_initial_overview import overview_source

    task, _, agent, opened = fixture_session(tmp_path, monkeypatch,
        [json.dumps({"command": {"action": action}, "world_observation": "Observed scene log " + str(index)})
         for index, action in enumerate(("UNSUPPORTED", "BASE_FWD", "DONE"))])
    agent.assembler = PromptAssembler()
    env = Environment()
    env.source, env.rig, _, _ = overview_source()
    env.rig.initial_overview_identity['episode_id'] = agent.episode_id
    env.source.begin_episode(task.task_text)
    summary = asyncio.run(run_episode(env, agent, tmp_path / "episode", success_provider=lambda: False))
    assert len(opened) == 1 and env.calls == ["BASE_FWD"]
    assert summary["termination_reason"] == "model_done" and summary["task_success"] is False
    calls = [params for method, params in agent._rpc.calls if method == "turn/start"]
    assert len(calls) == 3 and {params["threadId"] for params in calls} == {"exact-native-session-id"}
    assert "Public task" in calls[0]["input"][0]["text"]
    protection = agent.assembler.components["core/protection.md"]
    assert protection in calls[0]["input"][0]["text"]
    assert all(protection not in params["input"][0]["text"] for params in calls[1:])
    assert "Public task" not in calls[1]["input"][0]["text"]
    assert '"retry_feedback":' in calls[1]["input"][0]["text"]
    assert agent.sensor_ids[0] == agent.sensor_ids[1] and agent.sensor_ids[2] > agent.sensor_ids[1]
    assert agent.closed and json.loads((agent.root / "depth/active.json").read_text())["status"] == "closed"
    observations = [json.loads(line) for line in (tmp_path / "episode/world_observations.jsonl").read_text().splitlines()]
    assert len(observations) == 3
    assert all(row["used_by_controller"] is False and row["used_by_scorer"] is False for row in observations)
    assert all("Observed scene log" not in params["input"][0]["text"] for params in calls)
    assert not summary["public_memory"]


def test_input_manifests_match_native_images_across_retry_and_next_step(tmp_path, monkeypatch):
    from pathlib import Path
    from PIL import Image
    import numpy as np
    from harness_integration.test_task_edit2_initial_overview import overview_source

    task, _, agent, _ = fixture_session(tmp_path, monkeypatch,
        [json.dumps({'command': {'action': action}, 'world_observation': 'Visible workspace.'})
         for action in ('UNSUPPORTED', 'BASE_FWD', 'DONE')])
    env = Environment()
    env.source, env.rig, _, _ = overview_source()
    env.rig.initial_overview_identity['episode_id'] = agent.episode_id
    env.source.begin_episode(task.task_text)
    output = tmp_path / 'episode'
    result = asyncio.run(run_episode(env, agent, output, max_steps=2))
    assert result['termination_reason'] == 'model_done'
    calls = [params for method, params in agent._rpc.calls if method == 'turn/start']
    rows = [json.loads(line) for line in (output / 'model_inputs.jsonl').read_text().splitlines()
            if json.loads(line)['event'] == 'step_input']
    assert [len(row['images']) for row in rows] == [9, 4, 4]
    for row, call in zip(rows, calls, strict=True):
        paths = [Path(item['path']) for item in call['input'] if item['type'] == 'localImage']
        assert [image['name'] for image in row['images']] == [path.stem for path in paths]
        for image, path in zip(row['images'], paths, strict=True):
            assert np.array_equal(np.asarray(Image.open(output / image['path'])), np.asarray(Image.open(path)))
    for step in (1, 2):
        manifest = json.loads((output / f'step_{step:04d}' / 'model_inputs.json').read_text())
        assert len(manifest) == 4
        assert 'initial_overview_rgb' not in [image['name'] for image in manifest]


def test_native_adapter_timeout_interrupts_exact_turn_and_does_not_start_again(tmp_path, monkeypatch):
    _, _, agent, opened = fixture_session(tmp_path, monkeypatch, None, timeout=0.025)
    _, request, _ = request_inputs()
    async def exercise():
        with pytest.raises(ModelError, match="timed out"):
            await agent.decide(request)
        interrupts = [params for method, params in agent._rpc.calls if method == "turn/interrupt"]
        assert interrupts == [{"threadId": "exact-native-session-id", "turnId": "turn-1"}]
        await agent.close()
        with pytest.raises(ModelError, match="ended episode"):
            await agent.decide(request)
    asyncio.run(exercise())
    assert len(opened) == 1


@pytest.mark.parametrize("kind", ["commandExecution", "fileChange", "mcpToolCall", "collabAgentSpawn"])
def test_native_adapter_rejects_unauthorized_operation_events(kind, tmp_path, monkeypatch):
    _, _, agent, _ = fixture_session(tmp_path, monkeypatch, [])
    with pytest.raises(ModelError, match="unauthorized operation"):
        agent._notification("item/started", {"item": {"type": kind}})


def test_depth_callback_rejects_extra_paths_and_mismatched_snapshot_before_command(tmp_path, monkeypatch):
    _, _, agent, _ = fixture_session(tmp_path, monkeypatch, [])
    async def exercise():
        agent._has_depth = True
        agent.session_id = "episode-session"
        agent._completion = asyncio.get_running_loop().create_future()
        agent._current_binding = {"snapshot_id": 1, "captured_at": "time-a"}
        calls = []
        async def query(name, args):
            calls.append((name, args))
            return SimpleNamespace(structured_content={"ok": True, "snapshot_id": 2, "captured_at": "time-b"})
        agent._depth = SimpleNamespace(call=query)
        params = {"threadId": agent.session_id, "tool": "read_depth_pixels", "namespace": "functions",
                  "arguments": {"sensor": "head_depth_m", "pixels_uv": [[1, 1]], "path": "/private/ground_truth"}}
        response = await agent._tool_call(params)
        assert response["success"] is False and calls == []
        assert "/private" not in json.dumps(response)
        params["arguments"].pop("path")
        with pytest.raises(ModelError, match="unauthorized snapshot"):
            await agent._tool_call(params)
        params["threadId"] = "other-episode"
        with pytest.raises(ModelError, match="unauthorized tool or episode"):
            await agent._tool_call(params)
        agent._completion.cancel()
    asyncio.run(exercise())


def test_manifest_lease_cannot_change_identity_or_be_reused(request):
    backend, _, _, calls, _ = request.getfixturevalue("control_http")
    task_text = backend.observation_source.task_text
    backend.metadata["episode_contract"] = {"episode_id": "test-episode", "action_mode": "fixed",
                                            "world_advances_during_inference": True}
    assert backend.claim_control("other", task_text)["status"] == 409
    assert backend.claim_control("test-episode", task_text + "Changed")["status"] == 409
    backend._declarative_episode_claimed = True
    assert backend.claim_control("test-episode", task_text)["status"] == 409
    assert calls == [] and backend.commands.empty()


@pytest.fixture
def occupied_local_gpus(monkeypatch):
    """Make all local devices occupied and fail if a workload launch is attempted."""
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    requests = []
    def query(command, *args, **kwargs):
        if command[0] == "git" and command[-2:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        requests.append(command)
        assert command == ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                           "--format=csv,noheader,nounits"]
        return SimpleNamespace(stdout="\n".join(
            f"{index}, GPU-fixture-{index}, 90, 12000" for index in range(8)))
    monkeypatch.setattr(coordinator.subprocess, "run", query)
    return requests


@pytest.mark.parametrize("flag,expected", [([], None), (["--timeout", "none"], None), (["--timeout", "480"], 480.)])
def test_coordinator_preserves_default_explicit_and_unlimited_deadlines(flag, expected, tmp_path, occupied_local_gpus):
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    with pytest.raises(RuntimeError, match="No selected idle local GPU"):
        coordinator.main(["--task-file", str(example_task_path("find_and_lift")), "--variant", "visible_target_control",
                          "--max-steps", "60", "--output-dir", str(tmp_path), *flag])
    manifest = json.loads((tmp_path / "episode_manifest.json").read_text())
    assert manifest["wall_timeout_s"] == expected and manifest["max_steps"] == 60
    assert len(occupied_local_gpus) == 1  # No simulator, control lease or model invocation.


@pytest.mark.parametrize("filename", ["task_status.json", "scene_status.json"])
@pytest.mark.parametrize("status,reason", [("accepted_minimal_patch", None), ("skipped", "native_support_missing")])
def test_coordinator_reads_both_native_qualification_producers(filename, status, reason, tmp_path):
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    assert coordinator.read_native_qualification(tmp_path) is None
    payload = {"qualification_status": status, "reason": reason}
    (tmp_path / filename).write_text(json.dumps(payload))
    assert coordinator.read_native_qualification(tmp_path) == payload


def test_declarative_resume_budget_keeps_manifest_and_requires_extension():
    from wia.runtime.commands.run_esi_agent import declarative_resume_budget
    manifest = {"model_name": "gpt-6-astra", "reasoning_effort": "medium", "max_steps": 60,
                "wall_timeout_s": 5400, "action_mode": "parameterized", "world_mode": "run",
                "observation_condition": "current"}
    summary = {"episode_manifest": dict(manifest), "max_steps": 60,
               "agent": {"model": "gpt-6-astra", "reasoning_effort": "medium"},
               "max_duration_s": 5400, "action_mode": "parameterized", "world_mode": "run",
               "observation_condition": "current"}
    assert declarative_resume_budget(summary, manifest) == 60
    with pytest.raises(ValueError, match="--extend-max-steps"):
        declarative_resume_budget(summary, manifest, 120)
    assert declarative_resume_budget(summary, manifest, 120, extend=True) == 120
    assert manifest["max_steps"] == 60 and summary["max_steps"] == 60
    summary["max_steps"] = 120
    assert declarative_resume_budget(summary, manifest) == 120
    with pytest.raises(ValueError, match="exact original"):
        declarative_resume_budget(summary, dict(manifest, max_steps=120))
    summary["agent"]["model"] = "another-model"
    with pytest.raises(ValueError, match="original model"):
        declarative_resume_budget(summary, manifest)


def test_declarative_resume_cli_reuses_manifest_thread_and_boundary(tmp_path, monkeypatch):
    from wia.tasks.loader import load_task
    import wia.runtime.commands.run_esi_agent as runner
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
    session = {"codex_session_id": "original-thread"}
    summary = {"episode_manifest": manifest, "termination_reason": "model_error", "agent_decisions": 58,
        "max_steps": 60, "native_session": session, "max_duration_s": 5400.,
        "agent": {"model": "gpt-6-astra", "reasoning_effort": "medium"},
        "action_mode": "parameterized", "world_mode": "run", "observation_condition": "current"}
    (previous / "summary.json").write_text(json.dumps(summary))
    captured = {}
    def native(**kwargs):
        captured["native"] = kwargs
        return object()
    class Environment:
        def __init__(self, *args, **kwargs):
            captured["environment"] = kwargs
        async def __aenter__(self):
            raise RuntimeError("fixture stops before control acquisition")
        async def __aexit__(self, *args):
            pass
    monkeypatch.setattr("core.agent.codex_session.CodexEpisodeSession", native)
    monkeypatch.setattr(runner, "MCPRobotEnvironment", Environment)
    with pytest.raises(SystemExit) as error:
        runner.main(["--task-file", str(task.path), "--variant", task.variant, "--seed", "0",
            "--episode-manifest", str(manifest_path), "--resume-from", str(previous),
            "--extend-max-steps", "--max-steps", "120", "--output-dir", str(tmp_path / "continued")])
    assert error.value.code == 1
    assert captured["native"]["resume_state"] == session
    assert captured["native"]["episode_configuration"]["max_steps"] == 120
    assert captured["environment"]["resume_action_step"] == 58
    assert captured["environment"]["episode_contract"]["max_steps"] == 60
    assert json.loads(manifest_path.read_text()) == manifest


@pytest.mark.parametrize("owner,busy,completed,expected", [
    ("human", False, True, True), ("agent", False, True, False),
    ("human", True, True, False), ("human", False, False, False)])
def test_completed_simulator_retention_requires_released_idle_lease(owner, busy, completed, expected,
                                                                   tmp_path, monkeypatch):
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"control_owner": owner, "busy": busy})
    monkeypatch.setattr(coordinator.requests, "get", lambda *args, **kwargs: response)
    assert coordinator.retain_completed_simulator("http://fixture", tmp_path,
        requested=True, rollout_completed=completed) is expected
    assert not coordinator.retain_completed_simulator("http://fixture", tmp_path,
        requested=False, rollout_completed=completed)
    (tmp_path / "native_exit_code.txt").write_text("0\n")
    assert not coordinator.retain_completed_simulator("http://fixture", tmp_path,
        requested=True, rollout_completed=completed)


def test_live_human_preview_verification_preserves_canonical_boundary(tmp_path, monkeypatch):
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    import base64
    import io
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="JPEG")
    image = base64.b64encode(buffer.getvalue()).decode("ascii")
    frames = iter(("preview_1", "preview_2"))
    def get(url, **kwargs):
        data = {"observation_snapshot_id": 2, "steps": 0}
        if url.endswith("/robot"):
            seq = next(frames)
            data = {"frame_seq": seq, "frame_time": seq, "images": {"head": image, "wrist": image}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(coordinator.requests, "get", get)
    monkeypatch.setattr(coordinator.time, "sleep", lambda _: None)
    result = coordinator.verify_human_preview("http://fixture", tmp_path)
    assert result["status"] == "passed" and result["jpeg_image_sizes"] == {"head": [8, 8], "wrist": [8, 8]}
    assert result["first"]["canonical_snapshot_id"] == result["second"]["canonical_snapshot_id"] == 2


@pytest.mark.parametrize("flags,expected", [([], False), (["--record"], True),
                                          (["--norecord"], False), (["--no-record"], False)])
def test_coordinator_records_by_default_with_explicit_opt_out(flags, expected, tmp_path, monkeypatch, occupied_local_gpus):
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    original_parse = coordinator.argparse.ArgumentParser.parse_args
    captured = []
    def parse(parser, *args, **kwargs):
        parsed = original_parse(parser, *args, **kwargs)
        captured.append(parsed)
        return parsed
    monkeypatch.setattr(coordinator.argparse.ArgumentParser, "parse_args", parse)
    with pytest.raises(RuntimeError, match="No selected idle local GPU"):
        coordinator.main(["--task-file", str(example_task_path("find_and_lift")), "--output-dir", str(tmp_path), *flags])
    assert captured[0].record is expected
    assert len(occupied_local_gpus) == 1


def test_phase_transition_delivers_prior_and_current_images_in_one_existing_thread_turn(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from core.agent.image_inputs import image_inputs

    task, request, agent, opened = fixture_session(tmp_path, monkeypatch, [
        json.dumps({"command": {"action": "DONE"}, "world_observation": "Phase 1 view."}),
        json.dumps({"command": {"action": "DONE"}, "world_observation": "Phase 2 view."})])
    env = Environment()
    env.source.task_text = task.task_text
    prior = env.source.capture(action_boundary=True).observation
    current = env.source.capture(action_boundary=True).observation
    assert current.snapshot_id > prior.snapshot_id
    first = replace(request, observation=prior, decision_step=0)
    transitioned = replace(request, observation=current, decision_step=1,
                           previous_phase_observation=prior)

    async def exercise():
        await agent.decide(first)
        await agent.decide(transitioned)
        await agent.close()
    asyncio.run(exercise())

    turns = [params for method, params in agent._rpc.calls if method == "turn/start"]
    assert len(opened) == 1 and len(turns) == agent.turn_count == 2
    assert env.calls == []  # Delivering the transition does not execute a robot action.
    assert {turn["threadId"] for turn in turns} == {"exact-native-session-id"}
    assert [method for method, _ in agent._rpc.calls] == ["turn/start", "turn/start"]
    transition_inputs = turns[1]["input"]
    phase_labels = [(index, item["text"]) for index, item in enumerate(transition_inputs)
                    if item["type"] == "text" and "authorized observation bundle, snapshot" in item["text"]]
    assert [text.split(":", 1)[0] for _, text in phase_labels] == ["phase_1_final", "phase_2_initial"]
    assert f"snapshot {prior.snapshot_id}." in phase_labels[0][1]
    assert f"snapshot {current.snapshot_id}." in phase_labels[1][1]
    images = [(index, item) for index, item in enumerate(transition_inputs) if item["type"] == "localImage"]
    expected = [(phase, name, pixels) for phase, observation in
                [("phase_1_final", prior), ("phase_2_initial", current)]
                for name, pixels, _, transform in image_inputs(observation, include_initial_overview=False)]
    assert len(images) == len(expected)
    for (index, image), (phase, name, pixels) in zip(images, expected):
        assert image["path"].endswith(f"/{phase}_{name}.png")
        assert transition_inputs[index - 1]["text"].startswith(phase + ": ")
        assert np.array_equal(np.asarray(Image.open(image["path"])), pixels)
        if phase == "phase_1_final":
            assert phase_labels[0][0] < index < phase_labels[1][0]
        else:
            assert index > phase_labels[1][0]
    assert agent.sensor_ids == [prior.snapshot_id, current.snapshot_id]

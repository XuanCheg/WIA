"""CPU history integration boundaries. Run: PYTHONPATH=src pytest -q tests/test_candidate_history_integration.py.

No model, robot action, native simulator, render or HTTP service is executed.
"""

import asyncio
import base64
from copy import deepcopy
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from core.action_interface import ActionExecutor, BackendCapabilities
from core.agent.decision import DecisionInput
from core.agent.image_inputs import image_inputs
from core.observation_config import ObservationConfig
from core.observations import AgentObservation
from wia.runtime.agent.candidate_history import attach_history_images, decode_history
from wia.runtime.agent.prompts import PromptAssembler
from wia.runtime.agent.zero_shot import run_episode
from wia.runtime.sim.esi_observations import ESIObservationSource
from wia.runtime.web.web_teleop.esi_control import ESIControlBackend


def current_observation(task_text="Inspect the supplied views."):
    return AgentObservation(task_text=task_text, agentview_rgb=np.full((8, 8, 3), 90, dtype=np.uint8),
        wrist_rgb=np.full((8, 8, 3), 120, dtype=np.uint8), action_history=(), snapshot_id=3,
        captured_at="2026-10-30T00:00:01.000000+00:00", action_step=0, action_boundary=True)


def public_history():
    frames = []
    for index in range(2):
        buffer = BytesIO()
        Image.fromarray(np.full((8, 8, 3), 20 + index, dtype=np.uint8)).save(buffer, format="PNG")
        png = buffer.getvalue()
        frames.append({"index": index, "snapshot_id": index + 1,
            "captured_at": f"2026-10-30T00:00:00.{index:06d}+00:00", "simulation_time_s": index * .05,
            "camera": "agentview_rgb", "png_base64": base64.b64encode(png).decode("ascii"),
            "sha256": hashlib.sha256(png).hexdigest()})
    return {"schema": "wia_candidate_observation_history_v1", "frames": frames,
            "authored_intervention": True, "robot_actions": 0, "model_calls": 0}


@pytest.mark.parametrize("with_prelude", [False, True])
def test_episode_start_orders_frozen_overview_optional_prelude_and_baselines(with_prelude):
    calls = []
    initial, frozen, final = (SimpleNamespace(snapshot_id=i) for i in (1, 2, 3))
    session = SimpleNamespace(
        safety_observer=SimpleNamespace(begin_episode=lambda: calls.append("safety")),
        task_evaluator=SimpleNamespace(begin_episode=lambda obs: calls.append(("evaluator", obs))),
        exploration_evaluator=SimpleNamespace(begin_episode=lambda obs: calls.append(("exploration", obs))))
    source = ESIObservationSource(session, task_text="Existing task.")

    def capture(*, action_boundary):
        assert action_boundary
        calls.append("initial_capture")
        return SimpleNamespace(observation=initial)

    def overview(obs):
        assert obs is initial
        calls.append("frozen_overview")
        source.initial_observation = frozen
        return frozen

    source.capture, source._capture_initial_overview = capture, overview
    if with_prelude:
        def prelude(actual_source, obs):
            assert actual_source is source and obs is frozen
            calls.append("candidate_prelude")
            return final
        session.candidate_episode_prelude = prelude
    session.candidate_public_history_delivery = lambda: pytest.fail("Episode capture cannot acknowledge public delivery")
    returned = source.begin_episode("Current task.")
    expected = final if with_prelude else frozen
    assert returned is expected and source.initial_observation is expected
    assert calls == ["safety", "initial_capture", "frozen_overview",
                     *(["candidate_prelude"] if with_prelude else []),
                     ("evaluator", expected), ("exploration", expected)]
    assert source.action_step == 0 and source.history == ()


@pytest.mark.parametrize("with_history", [False, True])
def test_control_claim_exports_public_history_but_keeps_delivery_receipt_private(with_history):
    calls, responses = [], []
    payload = public_history()
    session = SimpleNamespace()
    source = SimpleNamespace(session=session, config=ObservationConfig(), task_text="Inspect the supplied views.")

    def begin(task_text):
        assert task_text == source.task_text
        calls.append("episode_ready")
        if with_history:
            session.candidate_public_history = payload

    def delivery():
        assert calls == ["episode_ready"] and not responses
        calls.append("delivery_acknowledged")
        return {"private_delivery_detail": "PRIVATE_DELIVERY_CANARY", "observed_tags_added": ["PRIVATE_OBJECT_ID"]}

    source.begin_episode = begin
    session.candidate_public_history_delivery = delivery
    executor = ActionExecutor(lambda *a, **k: pytest.fail("Control initialization must not execute a robot action"),
                              capabilities=BackendCapabilities(base_translation=True))
    backend = ESIControlBackend({}, executor, observation_source=source)
    worker = threading.Thread(target=lambda: responses.append(backend.claim_control(
        "cpu-history-claim", source.task_text, timeout=3, world_advances_during_inference=False)))
    worker.start()
    try:
        with backend.frame_cond:
            assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=2)
        assert backend.process_next()
        worker.join(timeout=3)
        assert not worker.is_alive() and len(responses) == 1
        response = responses[0]
        assert response["ok"] is True and response["scene_reset"] is False
        assert response.get("candidate_observation_history") == (payload if with_history else None)
        assert "PRIVATE_" not in json.dumps(response)
        expected_keys = {"ok", "message", "world_timing", "control_token", "run_id", "control_owner",
                         "scene_reset", "temporal_episode", "temporal_phase"}
        assert set(response) == expected_keys | ({"candidate_observation_history"} if with_history else set())
        if with_history:
            assert session.candidate_history_delivery_receipt["private_delivery_detail"] == "PRIVATE_DELIVERY_CANARY"
        else:
            assert not hasattr(session, "candidate_history_delivery_receipt")
        assert calls == ["episode_ready", *(["delivery_acknowledged"] if with_history else [])]
        assert backend.steps == backend.primitive_actions == backend.physics_steps == 0
    finally:
        worker.join(timeout=3)


@pytest.mark.parametrize("mode", ["no_history", "history", "error_after_attachment"])
def test_runner_attaches_after_observe_persists_once_and_restores_on_error(monkeypatch, tmp_path, mode):
    import wia.runtime.agent.candidate_history as history_module
    observation, calls, prepared = current_observation(), [], []
    payload = public_history()
    initial = {"scene_reset": False, **({"candidate_observation_history": deepcopy(payload)} if mode != "no_history" else {})}
    context = {"frame": {"forward": "+x"}}
    environment = SimpleNamespace(initialization=initial, action_context=deepcopy(context),
                                  capabilities=BackendCapabilities(base_translation=True))
    agent = SimpleNamespace(description=lambda: {"backend": "cpu-integration-mock"})
    original_images = lambda request: [("agentview_rgb", request.sensor_inputs.agentview_rgb, "Current head", "identity")]
    agent.request_images = original_images

    async def observe():
        assert agent.request_images is original_images
        calls.append("observe")
        return observation

    async def forbidden(*args, **kwargs):
        pytest.fail("Integration setup must not execute a model decision or robot action")

    def prepare(request):
        calls.append("prepare")
        assert request.observation is observation
        images = agent.request_images(request)
        prepared.append((request, images))
        if mode == "error_after_attachment":
            raise RuntimeError("CPU setup interruption after history attachment")
        return request

    actual_decode = history_module.decode_history
    def tracked_decode(value, current):
        assert calls == ["observe"]
        calls.append("decode")
        return actual_decode(value, current)

    monkeypatch.setattr(history_module, "decode_history", tracked_decode)
    environment.observe, environment.execute = observe, forbidden
    agent.prepare_request, agent.decide = prepare, forbidden
    output = tmp_path / mode
    summary = asyncio.run(run_episode(environment, agent, output, success_provider=lambda: True))
    assert agent.request_images is original_images
    assert summary["model_calls"] == summary["agent_decisions"] == 0
    assert summary["executed_actions"] == []
    request, images = prepared[0]
    if mode == "no_history":
        assert calls == ["observe", "prepare"]
        assert environment.initialization == {"scene_reset": False}
        assert environment.action_context == context
        assert list(request.action_context) == list(context)
        assert [name for name, *_ in images] == ["agentview_rgb"]
        assert not (output / "candidate_observation_history.json").exists()
        assert "candidate_history_frames" not in summary
    else:
        assert calls == ["observe", "decode", "prepare"]
        assert [name for name, *_ in images] == ["candidate_history_00_agentview_rgb", "candidate_history_01_agentview_rgb", "agentview_rgb"]
        for index, (_, pixels, _, transform) in enumerate(images[:2]):
            np.testing.assert_array_equal(pixels, np.full((8, 8, 3), 20 + index, dtype=np.uint8))
            assert transform == "identity"
        assert images[-1][1] is observation.agentview_rgb
        metadata = request.action_context["candidate_history_frames"]
        assert all(set(row) == {"index", "snapshot_id", "captured_at", "simulation_time_s", "camera", "sha256"} for row in metadata)
        assert "png_base64" not in json.dumps(request.action_context)
        history_file = output / "candidate_observation_history.json"
        assert json.loads(history_file.read_text()) == payload
        assert environment.initialization == {"scene_reset": False, "candidate_history_file": str(history_file)}
        # Large public image payloads have one durable owner, not duplicate
        # copies in the run manifest, summaries or initialization metadata.
        for path in output.glob("*.json"):
            if path != history_file:
                assert "png_base64" not in path.read_text(), path.name
    assert summary["termination_reason"] == ("runtime_error" if mode == "error_after_attachment" else "environment_success")


def test_prompt_chronology_matches_attached_images_without_changing_absent_history(monkeypatch):
    public = {"task_id": "cpu-history", "task_version": 1, "objective": "Inspect the supplied views.",
              "success_criteria": "Complete the physical task.", "environment": "Use supplied native images.", "constraints": []}
    assembler = PromptAssembler()
    observation = current_observation(assembler.task_text(public))
    request = DecisionInput(observation, ("DONE",), memory_mode="none", steps_remaining=2, decision_step=1)
    policy = {"episode_id": "cpu-history", "max_steps": 2, "world_mode": "pause",
              "world_advances_during_inference": False, "observation_condition": "current", "action_mode": "fixed"}
    captured = []
    render = assembler.render

    def capture_render(phase, slots):
        if phase == "episode_start":
            captured.append(json.loads(slots["INITIAL_OBSERVATION"].split("\n", 1)[1]))
        return render(phase, slots)

    monkeypatch.setattr(assembler, "render", capture_render)
    baseline = assembler.build(request, public, policy, first_turn=True)
    agent = SimpleNamespace(request_images=lambda current: image_inputs(current.sensor_inputs, include_initial_overview=True))
    restore_empty, metadata = attach_history_images(agent, decode_history(None, observation))
    assert metadata == []
    assert assembler.build(request, public, policy, first_turn=True).prompt() == baseline.prompt()
    restore_empty()
    restore, metadata = attach_history_images(agent, decode_history(public_history(), observation))
    try:
        with_history = replace(request, action_context={"candidate_history_frames": metadata})
        built = assembler.build(with_history, public, policy, first_turn=True)
        declared = captured[-1]["attached_images_in_order"]
        actual = agent.request_images(with_history)
        assert [row["name"] for row in declared] == [name for name, *_ in actual]
        assert [row["snapshot_id"] for row in declared[:2]] == [1, 2]
        assert all(row["snapshot_id"] == observation.snapshot_id for row in declared[2:])
        assert [row["sha256"] for row in declared[:2]] == [row["sha256"] for row in metadata]
        assert "png_base64" not in built.prompt() and "PRIVATE_" not in built.prompt()
        assert built.observation is observation and built.available_actions == request.available_actions
    finally:
        restore()

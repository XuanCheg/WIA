"""Exploration information boundary, note validation and world-phase contracts."""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from core.agent.decision import DecisionInput, unpack_memory_decision
from wia.runtime.agent.zero_shot import run_episode
from wia.runtime.sim.esi_explore_scenes import ExploreScene
from wia.native.esi_visibility import mask_metrics, passes_visibility
from wia.runtime.sim.esi_world_policy import WorldPolicy
from harness_integration.test_zero_shot_agent import Agent, Environment
from harness_integration.test_private_visibility import checker, Sensor
from core.observation_config import ObservationConfig
from core.robot_cameras import RGB_TO_VIEW


def test_note_is_carried_only_after_valid_command_and_associated_with_evidence(tmp_path):
    environment = Environment()
    agent = Agent('{"command":{"action":"BAD"},"memory_update":"unvalidated"}',
                  '{"command":{"action":"BASE_LEFT"},"memory_update":"Near table inspected."}',
                  '{"command":{"action":"DONE"},"memory_update":""}')
    summary = asyncio.run(run_episode(environment, agent, tmp_path, memory_mode="model_notes"))
    assert environment.calls == ["BASE_LEFT"]
    assert not agent.inputs[1].public_memory
    note = agent.inputs[2].public_memory[0]
    assert note == {"action_step": 0, "snapshot_id": agent.inputs[0].observation.snapshot_id,
                    "decision_step": 1, "text": "Near table inspected."}
    assert summary["public_memory"] == [note]
    assert agent.inputs[0].structured_schema["required"] == ["command", "memory_update"]


def test_invalid_note_does_not_invalidate_valid_motion(tmp_path):
    environment = Environment()
    agent = Agent(json.dumps({"command": {"action": "BASE_RIGHT"}, "memory_update": "x"*513}))
    summary = asyncio.run(run_episode(environment, agent, tmp_path, max_steps=1, memory_mode="model_notes"))
    assert environment.calls == ["BASE_RIGHT"]
    assert not summary["public_memory"]
    row = json.loads((tmp_path / "steps.jsonl").read_text())
    assert row["attempts"][0]["memory_validation_error"]


@pytest.mark.parametrize("raw", ['{"command":{"action":"BASE_FWD"},"memory_update":"ok","oracle":true}',
    '{"command":{"action":"BASE_FWD"},"command":{"action":"DONE"},"memory_update":"ok"}',
    '{"action":"BASE_FWD","memory_update":"ok"}'])
def test_malformed_envelopes_never_move(raw, tmp_path):
    environment = Environment()
    summary = asyncio.run(run_episode(environment, Agent(raw), tmp_path, max_invalid_retries=0,
                                      memory_mode="model_notes"))
    assert summary["termination_reason"] == "invalid_output" and environment.calls == []


def test_memory_is_bounded_and_contains_only_model_text(tmp_path):
    environment = Environment()
    agent = Agent(*[json.dumps({"command": {"action": "BASE_FWD"}, "memory_update": str(i)}) for i in range(6)])
    asyncio.run(run_episode(environment, agent, tmp_path, max_steps=6, memory_mode="model_notes"))
    assert [note["text"] for note in agent.inputs[-1].public_memory] == ["1", "2", "3", "4"]


def test_note_schema_does_not_alter_ordinary_command_contract():
    environment = Environment()
    request = DecisionInput(environment.source.agent_observation(), ("BASE_FWD", "DONE"))
    assert request.structured_schema == request.schema
    raw, note, error = unpack_memory_decision('{"command":{"action":"DONE"},"memory_update":12}')
    assert json.loads(raw) == {"action": "DONE"} and note is None and error


@pytest.mark.parametrize("advances", [True, False])
def test_policy_counts_wait_and_execution_in_distinct_simulation_phases(advances):
    policy = WorldPolicy()
    policy.begin(advances, 10.)
    policy.transition("capturing_observation", 10.)
    policy.transition("awaiting_model", 10.)
    assert policy.should_advance("agent") is advances
    if advances:
        for index in range(1, 4):
            policy.sample(10.+index*.1, idle_step=True)
    policy.transition("executing")
    assert policy.should_advance("agent")
    policy.transition("capturing_observation", policy.sim_time+.4)
    policy.transition("awaiting_model")
    state = policy.snapshot()
    assert state["simulation_time_during_model_wait_s"] == pytest.approx(.3 if advances else 0.)
    assert state["idle_controller_env_steps"] == (3 if advances else 0)
    assert state["action_execution_sim_time_s"] == pytest.approx(.4)
    policy.transition("human_idle")
    assert policy.should_advance("human")


def test_world_policy_requires_real_boolean():
    with pytest.raises(ValueError):
        WorldPolicy().begin("pause", 0.)


def test_few_pixels_are_not_zero_occlusion_or_recognition():
    rgb = np.zeros((32, 32, 3), np.uint8)
    mask = np.zeros((32, 32), bool)
    mask[12:14, 12:14] = True
    metrics = mask_metrics(mask, rgb)
    assert metrics["visible_pixels"] == 4
    assert not passes_visibility(metrics, minimum_pixels=64, minimum_extent_px=8)


@pytest.mark.parametrize("camera", list(RGB_TO_VIEW))
def test_hidden_gate_checks_every_enabled_camera(tmp_path, camera):
    check = checker(tmp_path, head_visible=False)
    check.config.update(camera_scope="enabled_agent", initial_condition="hidden_zero")
    check.session.agent_observation_config = ObservationConfig()
    check.session.robot_camera_registry = {name: name for name in RGB_TO_VIEW}
    check.session.robot.sensors = {name: Sensor(name == camera) for name in RGB_TO_VIEW}
    with pytest.raises(RuntimeError, match="Initial target visibility"):
        check.check_initial()
    record = json.loads((tmp_path / "trajectory.jsonl").read_text())
    assert len(record["views"]) == 8
    assert record["views"][RGB_TO_VIEW[camera]]["visible_pixels"] == 100
    assert all(sensor.modalities == {"rgb", "depth_linear"} for sensor in check.session.robot.sensors.values())


def test_hidden_gate_distinguishes_sliver_from_zero(tmp_path):
    check = checker(tmp_path, head_visible=False)
    check.config.update(initial_condition="hidden_zero")
    check.session.robot.sensors["head"].mask[8, 8] = 7
    with pytest.raises(RuntimeError):
        check.check_initial()
    check.config["initial_condition"] = "hidden_unrecognizable"
    record = check.check_initial()
    assert not record["all_views_zero_pixels"] and not record["any_recognizably_visible"]


def test_scene_configuration_is_deterministic_and_setup_is_guarded():
    root = __import__('wia.resources', fromlist=['ROOT']).ROOT
    path = root / "configs/esi_explore/single_table_occlusion.json"
    left, right = ExploreScene.load(path, seed=7), ExploreScene.load(path, seed=7)
    assert np.array_equal(left.target_position, right.target_position)
    assert left.occluders == right.occluders
    left.initialized = True
    with pytest.raises(RuntimeError):
        left.apply_to_env_config({})


def test_initial_view_only_keeps_initial_sensors_and_current_robot_feedback(tmp_path):
    environment = Environment()
    agent = Agent('{"action":"BASE_FWD"}', '{"action":"DONE"}')
    summary = asyncio.run(run_episode(environment, agent, tmp_path, observation_condition="initial_view_only"))
    first, second = agent.inputs
    assert first.sensor_inputs.snapshot_id == second.sensor_inputs.snapshot_id
    assert second.observation.snapshot_id > first.observation.snapshot_id
    assert second.observation.action_history[-1].action == "BASE_FWD"
    assert "initial snapshot" in second.prompt()
    assert summary["observation_condition"] == "initial_view_only"

"""Validate optional logging and action isolation. Run .venv/bin/python -m pytest -q this file."""
from dataclasses import replace
import json

import pytest

from core.agent.backends import structured_reply
from core.agent.decision import parse_action, ModelReply
from wia.runtime.agent.task_response import unpack_task_response
from wia.runtime.agent.prompts import PromptAssembler
from core.action_compiler import ParameterizedAction
from harness_integration.test_milestone85_prompts import request_inputs
from harness_integration.test_parameterized_actions import LIMITS, RESOLUTION


@pytest.mark.parametrize("mode", ["fixed", "parameterized"])
@pytest.mark.parametrize("enabled", [True, False])
def test_plugin_default_and_disabled_transport_preserve_commands(mode, enabled):
    task, request, policy = request_inputs()
    if mode == "parameterized":
        request = replace(request, action_mode=mode, action_context={"parameterized_actions": {
            "limits": LIMITS.__dict__, "resolution": RESOLUTION.__dict__}})
        policy["action_mode"] = mode
    assembler = PromptAssembler() if enabled else PromptAssembler(world_observation=False)
    request = assembler.build(request, task.public_payload, policy, first_turn=True)
    assert request.world_observation_log == enabled
    assert ("World observation logging plugin" in request.prompt()) == enabled
    assert assembler.provenance()["prompt_plugins"]["world_observation"] == enabled
    command = {"action": "GRASP"}
    payload = {"command": command, "answer": None, "arrived_rooms": [], "survey_complete": False}
    if enabled:
        payload["world_observation"] = "An object is visible near the fingers; attachment is uncertain."
    wire = json.dumps(payload)
    reply, fields, _ = unpack_task_response(request, ModelReply(wire))
    assert fields == {"answer": None, "arrived_rooms": [], "survey_complete": False}
    assert json.loads(reply.raw_text) == command
    assert ("world_observation" in reply.metadata) == enabled
    if enabled:
        assert reply.metadata["structured_output"]["raw_output"] == wire
        assert set(request.structured_schema["required"]) == {"command", "world_observation", "answer", "arrived_rooms", "survey_complete"}
        assert "world_observation" not in request.schema.get("properties", {})
    assert (ParameterizedAction.parse(reply.raw_text, request.available_actions).action if mode == "parameterized"
            else parse_action(reply.raw_text, request.available_actions)) == "GRASP"


def test_unknown_or_duplicate_envelope_fields_are_never_silently_executed():
    task, request, policy = request_inputs()
    request = PromptAssembler().build(request, task.public_payload, policy, first_turn=True)
    for raw in ('{"command":{"action":"DONE"},"world_observation":"visible","extra":1}',
                '{"command":{"action":"DONE"},"world_observation":"a","world_observation":"b"}'):
        reply = structured_reply(request, raw, {})
        assert "world_observation" not in reply.metadata
        with pytest.raises(ValueError):
            parse_action(reply.raw_text, request.available_actions)


def test_neutral_task_and_initial_prompt_do_not_hint_at_enclosure():
    from harness_integration.task_examples import load_example_task as load_task
    task = load_task("retrieve_apple_and_lift", variant="closed_sliding_box")
    _, request, policy = request_inputs()
    request = replace(request, observation=replace(request.observation, task_text=task.task_text))
    prompt = PromptAssembler().build(request, task.public_payload, policy, first_turn=True).prompt()
    for hint in ("glass", "box", "lid_hinge", "task_apple", "closed_glass_box", "hinged", "top door"):
        assert hint not in prompt.lower()

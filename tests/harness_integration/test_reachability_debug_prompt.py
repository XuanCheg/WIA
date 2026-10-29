"""Verify optional public blockage reports without changing action or privacy contracts.

Run: python -B -m pytest -q tests/test_reachability_debug_prompt.py
"""
import json

import pytest

from core.agent.backends import structured_reply
from wia.runtime.agent.prompts import PromptAssembler, REACHABILITY_DEBUG_COMPONENT
from harness_integration.test_milestone85_prompts import request_inputs


def test_disabled_debug_retains_default_prompts_and_provenance():
    task, request, policy = request_inputs()
    default = PromptAssembler()
    disabled = PromptAssembler(debug_reachability=False)
    for first in [True, False]:
        assert default.build(request, task.public_payload, policy, first_turn=first).prompt() == disabled.build(
            request, task.public_payload, policy, first_turn=first).prompt()
    assert default.provenance() == disabled.provenance()
    assert REACHABILITY_DEBUG_COMPONENT not in disabled.provenance()['prompt_component_hashes']


@pytest.mark.parametrize('first', [True, False])
def test_debug_prompt_keeps_public_action_and_observation_contracts(first):
    task, request, policy = request_inputs()
    baseline = PromptAssembler().build(request, task.public_payload, policy, first_turn=first)
    assembler = PromptAssembler(debug_reachability=True)
    prepared = assembler.build(request, task.public_payload, policy, first_turn=first)
    assert prepared.prompt().count('Reachability debug reporting is enabled') == 1
    for field in ['DEBUG_GOAL:', 'DEBUG_ITEM:', 'DEBUG_REASON:', 'DEBUG_EVIDENCE:']:
        assert field in prepared.prompt()
    assert 'uncertain inference' in prepared.prompt()
    assert 'not that the object or entire task is physically impossible' in prepared.prompt()
    assert prepared.schema == baseline.schema
    assert prepared.structured_schema == baseline.structured_schema
    assert set(prepared.structured_schema['properties']) == {'command', 'world_observation'}
    assert assembler.provenance()['prompt_plugins']['reachability_debug'] is True
    assert REACHABILITY_DEBUG_COMPONENT in assembler.provenance()['prompt_component_hashes']
    assert 'PRIVATE_TARGET_CANARY' not in prepared.prompt()
    report = ('The item remains on its support.\nDEBUG_GOAL: Pick up the visible target.\n'
              'DEBUG_ITEM: The visible target.\nDEBUG_REASON: Controller-reported arm_joint_limit.\n'
              'DEBUG_EVIDENCE: The previous command stopped; acquisition is unconfirmed.')
    reply = structured_reply(prepared, json.dumps({'command': {'action': 'GRASP'},
        'world_observation': report}), {})
    assert json.loads(reply.raw_text) == {'action': 'GRASP'}
    assert reply.metadata['world_observation'] == report


def test_debug_requires_world_observation():
    with pytest.raises(ValueError, match='requires world-observation'):
        PromptAssembler(world_observation=False, debug_reachability=True)

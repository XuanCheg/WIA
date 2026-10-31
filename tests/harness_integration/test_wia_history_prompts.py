"""Check explicit Phase-2 diagnostic prompts and no-turn session opening.

Run: PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_history_prompts.py
Only fake native transports are used; no model or simulator is launched.
"""
import asyncio
from dataclasses import replace
import json

import pytest

from core.agent.codex_session import CodexEpisodeSession
from core.agent.decision import ModelError
from wia.runtime.agent.prompts import PromptAssembler
from wia.runtime.agent.wia_history_projection import project_phase2_input
from core.observations import PublicTaskEvent
from harness_integration.test_wia_history_projection import request as history_request


def inputs():
    public = {'task_id': 'wia_test', 'task_version': 1, 'objective': 'Compare current contents.',
              'success_criteria': 'Perform the correct physical action.',
              'environment': 'Use the current robot sensors.', 'constraints': []}
    task_text = public['objective'] + '\n\nPhysical success criteria: ' + public['success_criteria']
    original = history_request()
    original = replace(original, observation=replace(original.observation, task_text=task_text),
                       sensor_observation=replace(original.sensor_observation, task_text=task_text))
    policy = {'episode_id': 'episode', 'max_steps': 20, 'world_mode': 'pause',
              'world_advances_during_inference': False, 'observation_condition': 'current',
              'action_mode': original.action_mode}
    return public, original, policy


def full_inputs():
    public, original, policy = inputs()
    context = {key: value for key, value in original.action_context.items()
               if key in ('frame', 'working_arm', 'steps', 'rotation_frame', 'trunk_height_range_m',
                          'trunk_height_range_reference', 'feedback_semantics', 'parameterized_actions')}
    return public, replace(original, memory_mode='none', public_memory=(), feedback=None,
                           previous_phase_observation=None, assembled_prompt=None, resident_prompt=None,
                           prompt_phase=None, action_context=context), policy


def test_default_full_prompt_and_provenance_remain_compatible():
    public, request, policy = full_inputs()
    default, explicit = PromptAssembler(), PromptAssembler(wia_history_condition='full')
    assert default.initial_overview_required is explicit.initial_overview_required is True
    assert default.provenance() == explicit.provenance()
    assert 'wia_history_condition' not in default.provenance()
    first = default.build(request, public, policy, first_turn=True)
    later = default.build(request, public, policy, first_turn=False)
    assert first.prompt() == explicit.build(request, public, policy, first_turn=True).prompt()
    assert later.prompt() == explicit.build(request, public, policy, first_turn=False).prompt()
    assert first.schema == later.schema == request.schema
    assert first.sensor_inputs is later.sensor_inputs is request.sensor_inputs
    assert 'Every episode starts with one initial global RGB overview' in first.resident_prompt


@pytest.mark.parametrize('condition', ['history_removed', 'history_oracle'])
def test_diagnostic_metadata_is_explicit_and_current_physical_inputs_are_preserved(condition):
    public, original, policy = inputs()
    facts = 'Earlier: the red jar was in the left box.\nNo answer is supplied.'
    request, _ = project_phase2_input(original, condition=condition,
                                      oracle_facts=facts if condition == 'history_oracle' else None)
    assembler = PromptAssembler(wia_history_condition=condition)
    assert assembler.initial_overview_required is False
    assert assembler.provenance()['wia_history_condition'] == condition
    first = assembler.build(request, public, policy, first_turn=True)
    later = assembler.build(request, public, policy, first_turn=False)
    for built in (first, later):
        text = built.prompt()
        assert '"initial_overview_required": false' in text
        assert '"required": false' in text
        assert '"native_capture_failure": false' in text
        assert '"wia_history_condition": "' + condition + '"' in text
        assert 'deliberately withheld' in text
        assert 'not a native capture failure' in text
        assert '"initial_overview_rgb"' not in text
        assert built.schema == request.schema
        assert built.available_actions == request.available_actions
        assert built.sensor_inputs is request.sensor_inputs
        assert '"raw_metric_depth"' in text and '"camera_geometry"' in text
        if condition == 'history_oracle':
            assert json.dumps(facts) in text
        else:
            assert 'diagnostic_historical_facts' not in text
    assert 'Every episode starts with one initial global RGB overview' not in first.resident_prompt
    assert 'All eight mounted views are captured; subsequent turns attach four' in first.resident_prompt
    assert 'offer the other four through read_rgb_camera' in first.resident_prompt
    assert original.observation.initial_overview is not None


@pytest.mark.parametrize('channel', ['overview', 'sensor_overview', 'previous_phase', 'public_memory',
                                    'phase', 'intervention', 'frozen', 'events', 'retry_feedback', 'context_history'])
def test_contaminated_diagnostic_inputs_fail_closed(channel):
    public, original, policy = inputs()
    request, _ = project_phase2_input(original, condition='history_removed')
    if channel == 'overview':
        request = replace(request, observation=original.observation)
    elif channel == 'sensor_overview':
        request = replace(request, sensor_observation=original.sensor_observation)
    elif channel == 'previous_phase':
        request = replace(request, previous_phase_observation=original.observation)
    elif channel == 'public_memory':
        request = replace(request, public_memory=({'note': 'old memory'},))
    elif channel == 'frozen':
        request = replace(request, observation_condition='initial_view_only')
    elif channel == 'events':
        event = PublicTaskEvent('old', 'history_followup', 'Old event.', 1, 2)
        request = replace(request, observation=replace(request.observation, public_task_events=(event,)))
    elif channel == 'retry_feedback':
        request = replace(request, feedback='Old rejection text.')
    else:
        context = dict(request.action_context)
        if channel == 'phase':
            context['temporal_phase'] = 'phase_1'
        elif channel == 'intervention':
            context.pop('wia_history_intervention')
        else:
            context['temporal_feedback'] = 'Old temporal event.'
        request = replace(request, action_context=context)
    with pytest.raises(ValueError):
        PromptAssembler(wia_history_condition='history_removed').build(request, public, policy, first_turn=True)


def test_oracle_conditions_and_physical_mode_are_guarded():
    public, original, policy = inputs()
    removed, _ = project_phase2_input(original, condition='history_removed')
    oracle, _ = project_phase2_input(original, condition='history_oracle', oracle_facts='Earlier fact.')
    with pytest.raises(ValueError):
        PromptAssembler(wia_history_condition='history_oracle').build(removed, public, policy, first_turn=True)
    with pytest.raises(ValueError):
        PromptAssembler(wia_history_condition='history_removed').build(oracle, public, policy, first_turn=True)
    with pytest.raises(ValueError):
        PromptAssembler(task_mode='original_qa', wia_history_condition='history_removed')
    with pytest.raises(ValueError):
        PromptAssembler(wia_history_condition='unknown')


def test_explicit_phase2_retry_feedback_is_retained_without_accepting_old_rejection_text():
    public, original, policy = inputs()
    request, _ = project_phase2_input(original, condition='history_removed')
    retry = replace(request, feedback='Return exactly one valid command JSON.')
    assembler = PromptAssembler(wia_history_condition='history_removed')
    with pytest.raises(ValueError, match='retry marker'):
        assembler.build(retry, public, policy, first_turn=False)
    context = dict(retry.action_context)
    context['wia_history_intervention'] = dict(context['wia_history_intervention'],
                                              phase2_retry_feedback_allowed=True)
    prepared = assembler.build(replace(retry, action_context=context), public, policy, first_turn=False)
    assert prepared.feedback == retry.feedback
    assert retry.feedback in prepared.prompt()
    assert 'OLD_REJECTION_CANARY' not in prepared.prompt()


def test_public_open_creates_one_thread_without_a_model_turn_then_decide_uses_it(tmp_path, monkeypatch):
    public, original, policy = inputs()
    request, _ = project_phase2_input(original, condition='history_removed')
    agent = CodexEpisodeSession(public_task=public, episode_configuration=policy,
        output=tmp_path, assembler=PromptAssembler(wia_history_condition='history_removed'))
    calls = []

    async def fake_open(prepared):
        calls.append('thread/start')
        assert prepared.assembled_prompt is not None
        agent.session_id = 'fresh-diagnostic-thread'

    async def fake_turn(prepared):
        calls.append('turn/start')
        return 'reply'

    monkeypatch.setattr(agent, '_open', fake_open)
    monkeypatch.setattr(agent, '_turn', fake_turn)

    async def exercise():
        assert await agent.open(request) == 'fresh-diagnostic-thread'
        assert calls == ['thread/start'] and agent.turn_count == 0
        assert agent.provenance()['initial_overview_required'] is False
        with pytest.raises(ModelError, match='already|exactly one'):
            await agent.open(request)
        assert await agent.decide(request) == 'reply'
        assert calls == ['thread/start', 'turn/start']
        agent.closed = True
        with pytest.raises(ModelError, match='ended|closed|exactly one'):
            await agent.open(request)
    asyncio.run(exercise())


def test_default_start_still_opens_then_performs_one_turn(tmp_path, monkeypatch):
    public, request, policy = full_inputs()
    agent = CodexEpisodeSession(public_task=public, episode_configuration=policy, output=tmp_path, assembler=PromptAssembler())
    calls = []

    async def fake_open(prepared):
        calls.append('open')
        agent.session_id = 'full-thread'

    async def fake_turn(prepared):
        calls.append('turn')
        return 'full-reply'

    monkeypatch.setattr(agent, '_open', fake_open)
    monkeypatch.setattr(agent, '_turn', fake_turn)
    assert asyncio.run(agent.start(request)) == 'full-reply'
    assert calls == ['open', 'turn']
    assert agent.provenance()['initial_overview_required'] is True

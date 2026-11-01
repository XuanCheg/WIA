"""Exercise structured declarations end to end. Run: python -m pytest -q tests/test_task_response.py."""
import asyncio
from dataclasses import replace
import json

import pytest

from core.agent.decision import ModelReply
from wia.resources import ROOT
from wia.runtime.agent.prompts import PromptAssembler
from wia.runtime.agent.task_response import unpack_task_response
from wia.runtime.agent.output_normalization import normalize_extra_fields
from wia.runtime.agent.zero_shot import run_episode
from wia.tasks.loader import load_task
from harness_integration.test_milestone85_prompts import request_inputs
from harness_integration.test_zero_shot_agent import Environment
from test_t52_confirmation import confirmed_world


def payload(**changes):
    return {'command': {'action': 'DONE'}, 'world_observation': 'Current observations.',
            'answer': None, 'arrived_rooms': [], 'survey_complete': False, **changes}


def prepared(alias='t52', *, declarations=False, world_observation=True):
    _, request, policy = request_inputs()
    task = load_task(ROOT/f'explore_to_achieve/wia_v1/release/tasks/{alias}.yaml', run_mode='benchmark')
    request = replace(request, observation=replace(request.observation, task_text=task.task_text),
                      action_context={'observation_declarations_available': declarations})
    return PromptAssembler(world_observation=world_observation).build(request, task.public_payload, policy, first_turn=True)


@pytest.mark.parametrize('alias,keys', [('t23', ['granola box', 'spray can', 'colored pencil']),
                                      ('t05', [f'goal_{i}' for i in range(1, 6)]),
                                      ('t06', [f'goal_{i}' for i in range(1, 6)])])
def test_mapping_answers_use_only_public_keys(alias, keys):
    request = prepared(alias)
    assert list(request.structured_schema['properties']['answer']['anyOf'][0]['properties']) == keys
    answer = dict.fromkeys(keys, 'unselected public option')
    reply, fields, _ = unpack_task_response(request, ModelReply(json.dumps(payload(answer=answer))))
    assert fields['answer'] == answer and json.loads(reply.raw_text) == {'action': 'DONE'}
    with pytest.raises(ValueError, match='map every public'):
        unpack_task_response(request, ModelReply(json.dumps(payload(answer='a single container'))))


@pytest.mark.parametrize('answer', [True, 3, [], '', {'private_target': 'inside'}])
def test_invalid_scalar_answer_rejected_before_dispatch(answer):
    with pytest.raises(ValueError):
        unpack_task_response(prepared(), ModelReply(json.dumps(payload(answer=answer))))


def test_declarations_survive_normalization_and_remain_in_trace():
    request = prepared('t09', declarations=True)
    claims = [{'label': 'chandelier', 'camera': 'agentview_rgb'}]
    value = payload(answer='kitchen_0', observed_objects=claims, ignored_extra='drop me')
    reply, fields, observed = unpack_task_response(request,
        normalize_extra_fields(request, ModelReply(json.dumps(value))))
    assert fields['answer'] == 'kitchen_0' and observed == claims
    trace = json.loads(reply.metadata['structured_output']['raw_output'])
    assert trace['answer'] == 'kitchen_0' and trace['observed_objects'] == claims
    assert 'ignored_extra' not in trace


def test_task_fields_work_without_observation_logging():
    request = prepared(world_observation=False)
    value = payload(answer='inside'); value.pop('world_observation')
    reply, fields, _ = unpack_task_response(request, ModelReply(json.dumps(value)))
    assert fields['answer'] == 'inside' and json.loads(reply.raw_text) == {'action': 'DONE'}


def test_malformed_survey_command_is_a_validation_error():
    request = prepared('t38')
    with pytest.raises(ValueError, match='command must be an object'):
        unpack_task_response(request, ModelReply(json.dumps(payload(command=None, survey_complete=True))))


class Policy:
    def __init__(self, *outputs, alias='t52'):
        self.outputs = iter(outputs)
        self.task = load_task(ROOT/f'explore_to_achieve/wia_v1/release/tasks/{alias}.yaml', run_mode='benchmark')
        self.count = 0

    def description(self):
        return {'backend': 'test'}

    def prepare_request(self, request):
        policy = {'episode_id': 'test', 'max_steps': 3, 'world_mode': 'pause',
                  'world_advances_during_inference': False, 'observation_condition': 'current', 'action_mode': 'fixed'}
        result = PromptAssembler().build(request, self.task.public_payload, policy, first_turn=self.count == 0)
        self.count += 1
        return result

    async def decide(self, request):
        return ModelReply(json.dumps(next(self.outputs)))


@pytest.mark.parametrize('explicit', [True, False])
def test_t52_separate_answer_then_done_and_text_only_no_credit(tmp_path, monkeypatch, explicit):
    wrapper, _, _ = confirmed_world(tmp_path, monkeypatch)
    env = Environment()
    async def answer(text, observation):
        wrapper.report_answer(text, observation)
        return {'ok': True}
    async def done(observation):
        wrapper.report_done(observation)
        return {'ok': True, 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    env.report_answer, env.report_done = answer, done
    first = payload(command={'action': 'BASE_FWD'}, answer='inside' if explicit else None,
                    world_observation='ANSWER: inside\nARRIVED: kitchen_0\nSURVEY_COMPLETE')
    policy = Policy(first, payload())
    result = asyncio.run(run_episode(env, policy, tmp_path/'episode', max_steps=3))
    assert result['termination_reason'] == 'model_done' and env.calls == ['BASE_FWD']
    assert wrapper.done_result['score_percent'] == (100. if explicit else 0.)
    assert wrapper.native.answer_history[0]['action_step'] == 0 if explicit else not wrapper.native.answer_history


def test_arrivals_use_field_and_history_survey_continues_without_motion(tmp_path):
    env = Environment(); arrivals = []; answers = []; done = []
    async def report_arrivals(rooms, observation):
        arrivals.extend(rooms); return {'ok': True}
    async def report_answer(text, observation):
        answers.append(text); return {'ok': True}
    async def report_done(observation):
        done.append(observation.action_step)
        return {'ok': True, 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    env.report_arrivals, env.report_answer, env.report_done = report_arrivals, report_answer, report_done
    policy = Policy(payload(survey_complete=True, arrived_rooms=['kitchen_0']), payload(), alias='t38')
    result = asyncio.run(run_episode(env, policy, tmp_path/'episode', max_steps=3))
    assert result['termination_reason'] == 'model_done' and result['decision_calls'] == 2
    assert answers == ['SURVEY_COMPLETE'] and arrivals == ['kitchen_0'] and done == [0]
    assert not env.calls


def test_survey_cannot_silently_execute_motion_or_finish_unrelated_task():
    with pytest.raises(ValueError, match='No survey'):
        unpack_task_response(prepared(), ModelReply(json.dumps(payload(survey_complete=True))))
    with pytest.raises(ValueError, match='command DONE'):
        unpack_task_response(prepared('t38'), ModelReply(json.dumps(payload(
            survey_complete=True, command={'action': 'GRASP'}))))


def test_rejected_survey_does_not_end_episode_or_publish_completion(tmp_path):
    env = Environment()
    done = []
    async def reject(observation):
        return {'ok': False, 'message': 'Answer declaration is unavailable at this boundary.'}
    async def report_done(observation):
        done.append(observation.action_step)
        return {'ok': True, 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    env.report_survey, env.report_done = reject, report_done
    policy = Policy(payload(survey_complete=True), payload(command={'action': 'BASE_FWD'}), payload(), alias='t38')
    result = asyncio.run(run_episode(env, policy, tmp_path/'episode', max_steps=3))
    assert result['termination_reason'] == 'model_done' and result['decision_calls'] == 3
    assert not result.get('history_collection_survey_complete')
    assert env.calls == ['BASE_FWD'] and done == [1]
    rows = [json.loads(line) for line in (tmp_path/'episode/steps.jsonl').read_text().splitlines()]
    assert rows[0]['survey_rejected'] and rows[0]['survey_declaration']['receipt']['ok'] is False


def test_unmet_survey_does_not_poison_native_evaluator(tmp_path):
    from types import SimpleNamespace
    from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope
    def reject(*args):
        raise ValueError('The survey lacks informative encounters for every target.')
    native = SimpleNamespace(report_answer=reject)
    wrapper = CategoryEvaluationEnvelope(native, SimpleNamespace(verify_unchanged=lambda: None), tmp_path)
    wrapper.episode = 1
    with pytest.raises(ValueError, match='lacks informative'):
        wrapper.report_answer('SURVEY_COMPLETE', None)
    assert wrapper.failure is None and wrapper.done_result is None


def test_survey_controls_cannot_bypass_their_field_through_answer():
    request = prepared('t38')
    assert request.structured_schema['properties']['answer']['type'] == 'null'
    with pytest.raises(ValueError, match='control marker'):
        unpack_task_response(request, ModelReply(json.dumps(payload(answer='SURVEY_COMPLETE'))))
    with pytest.raises(ValueError, match='until the survey is accepted'):
        unpack_task_response(request, ModelReply(json.dumps(payload(answer='a guessed target'))))

"""Run: python -m pytest -q tests/test_task_repair.py (CPU event replay only)."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import yaml

from wia.tasks.completion import progress
from wia.tasks.loader import load_task
from wia.tasks.temporal_policy import survey_action_allowed
from wia.runtime.agent.zero_shot import run_episode
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope, UnavailableEvaluator
from harness_integration.test_temporal_phase_transport import TemporalEnvironment, MetadataAgent, reply, OPTION


def test_bounded_survey_invalid_decisions_and_final_image_delivery(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    agent = MetadataAgent(reply('FORBIDDEN'), reply('BASE_FWD'), reply(text=f'ANSWER: {OPTION}'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path / 'episode', max_steps=4,
                                     success_provider=environment.success))
    assert summary['phase_1_decisions'] == 2
    assert summary['agent_decisions'] == summary['model_calls'] == 3
    assert summary['primitive_actions'] == 1
    assert environment.controller.advances == 1
    final = agent.inputs[-1]
    assert final.previous_phase_observation.action_step == 1
    assert final.observation.action_step == 1
    assert final.previous_phase_observation.snapshot_id < final.observation.snapshot_id
    transition = summary['temporal_transitions'][0]
    assert transition['reason'] == 'budget_exhausted' and transition['final_phase_1_delivered']
    assert json.loads(open(transition['final_phase_1_payload']).read())['snapshot_id'] == final.previous_phase_observation.snapshot_id


def test_survey_budget_incompatible_with_episode_is_explicit(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    with pytest.raises(ValueError, match='leave a Phase-2'):
        asyncio.run(run_episode(environment, MetadataAgent(), tmp_path / 'episode', max_steps=2))


def test_partial_validity_keeps_denominator_and_no_comparable_score():
    row = progress({'task_success': False, 'completion_items': [
        {'id': str(index), 'completed': index < 4, 'evaluation_validity': 'valid' if index < 4 else 'unavailable'}
        for index in range(5)]})
    assert row['completion_score'] is None and row['evaluation_validity'] == 'partial'
    assert row['total_goals'] == 5 and row['diagnostic_lower_bound'] == .8
    valid = progress({'task_success': False, 'completion_items': [
        {'id': str(index), 'completed': index < 4, 'evaluation_validity': 'valid'} for index in range(5)]})
    assert valid['score_percent'] == 80 and valid['completion_score'] == .8


@pytest.mark.parametrize('termination,countable', [('runtime_error', False), ('model_error', False),
                                                 ('cancelled', False), ('max_steps', True)])
def test_interrupted_execution_is_not_a_model_quality_failure(termination, countable):
    from wia.runtime.commands.run_esi_agent import finalize_evaluation_summary
    measured = {'evaluation_validity': 'complete', 'completion_score': 0., 'score_percent': 0.}
    summary = {'termination_reason': termination, 'task_success': False}
    finalize_evaluation_summary(summary, measured, 'diagnostic')
    assert summary['agent_failure_countable'] is countable
    assert summary['task_progress'] is measured
    if not countable:
        assert summary['task_success'] is None and summary['completion_score'] is None
        assert not summary['task_progress_is_final']
    else:
        assert summary['task_success'] is False and summary['completion_score'] == 0.


def test_pending_diagnostic_loading_keeps_review_state(tmp_path):
    from harness_integration.test_milestone85_tasks import task_yaml
    loaded, definition = task_yaml()
    definition.update(kind='category_robot_task', status='pending')
    path = tmp_path / 'task.yaml'
    path.write_text(yaml.safe_dump(definition))
    with pytest.raises(ValueError, match='pending'):
        load_task(path)
    task = load_task(path, run_mode='diagnostic')
    assert task.definition.status == 'pending' and task.manifest('test')['run_mode'] == 'diagnostic'


def test_unavailable_native_score_is_not_zero():
    evaluator = UnavailableEvaluator({'type': 'grasp_lift'}, 'binding', ValueError('missing target'))
    row = evaluator.record(NS(snapshot_id=1, action_step=0))
    assert row['completion_score'] is None and row['total_goals'] == 1
    assert row['failure_stage'] == 'binding' and row['evaluation_validity'] == 'unavailable'


@pytest.mark.parametrize('action,allowed', [('BASE_FWD', True), ('MV_UP', True), ('TRUNK_UP', True),
                                           ('GRASP', False), ('RELEASE', False), ('DONE', True)])
def test_phase_one_action_permissions(action, allowed):
    assert survey_action_allowed(action) is allowed


def test_temporal_resume_retains_budget_and_undelivered_final_observation(tmp_path):
    from core.agent.decision import ModelError
    class Interrupted(MetadataAgent):
        async def decide(self, request):
            if request.previous_phase_observation is not None:
                raise ModelError('Synthetic interrupted transport')
            return await super().decide(request)
    environment = TemporalEnvironment(tmp_path / 'private')
    first = asyncio.run(run_episode(environment, Interrupted(reply('BASE_FWD'), reply('BASE_FWD')),
        tmp_path / 'episode', max_steps=4, success_provider=environment.success))
    assert first['termination_reason'] == 'model_error'
    assert first['phase_1_decisions'] == 2 and environment.controller.advances == 1
    assert not first['temporal_transitions'][0]['final_phase_1_delivered']
    agent = MetadataAgent(reply(text=f'ANSWER: {OPTION}'))
    resumed = asyncio.run(run_episode(environment, agent, tmp_path / 'resume', max_steps=4,
        success_provider=environment.success, resume_summary=first))
    assert environment.controller.advances == 1 and resumed['phase_1_decisions'] == 2
    assert resumed['agent_decisions'] == 3 and resumed['model_calls'] == 4
    assert agent.inputs[0].previous_phase_observation.action_step == 2
    assert resumed['temporal_transitions'][0]['final_phase_1_delivered']


def test_background_policy_protects_support_and_only_allows_arm_contact():
    import numpy as np
    from wia.tasks.background_policy import install_background_policy, legal_background_contact, record_background_movements
    class Object:
        def __init__(self, name, category, z):
            self.name, self.category, self.prim_path = name, category, '/' + name
            self.fixed_base = self.visual_only = False
            self.position = np.array([0., 0., z])
            self.aabb = (self.position, self.position + [1., 1., .1])
        def get_position_orientation(self):
            return self.position, [0., 0., 0., 1.]
        def contact_list(self):
            return [NS(body0='/target/link', body1='/support/link')] if self.name == 'target' else []
    target, support, chair = Object('target', 'apple', 1.), Object('support', 'table', .9), Object('chair', 'chair', 0.)
    robot = NS(arm_link_names={'left': ['arm']}, finger_link_names={'left': []}, links={'arm': NS(prim_path='/robot/arm')})
    session = NS(env=NS(scene=NS(objects=[target, support, chair])), robot=robot, _path_cache={'old': True})
    task = NS(definition=NS(kind='category_robot_task', model_dump=lambda **_: {'private': {'bindings': ['target']}, 'scene': {}}))
    policy = install_background_policy(session, task)
    assert set(policy['allowed']) == {'chair'} and 'support' in policy['protected']
    assert legal_background_contact(session, '/robot/arm', '/chair/link')
    assert not legal_background_contact(session, '/robot/base', '/chair/link')
    assert not legal_background_contact(session, '/robot/arm', '/support/link')
    chair.position[0] += .1
    assert record_background_movements(session)[0]['name'] == 'chair'
    assert session.background_static_map_stale and not session._path_cache
    session.temporal_phase_controller = NS(phase='phase_1')
    assert not legal_background_contact(session, '/robot/arm', '/chair/link')


def test_source_conflict_never_counts_uncertified_selection_as_completed(tmp_path):
    from harness_integration.test_cognitive_arrival_transport import LoadedTask, NativeArrival, A
    task = LoadedTask()
    task.definition = deepcopy(task.definition)
    task.definition.repair = {'source_conflict': 'unresolved'}
    native = NativeArrival(required=(A,))
    envelope = CategoryEvaluationEnvelope(native, task, tmp_path)
    observation = NS(snapshot_id=1, action_step=0)
    envelope.begin_episode(observation)
    envelope.report_arrivals([A], observation)
    row = json.loads((tmp_path / 'state.json').read_text())
    assert row['native_result']['task_success'] is True
    assert row['task_success'] is False and row['completion_score'] is None
    assert row['progress']['diagnostic_lower_bound'] == 0

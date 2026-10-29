"""Run CPU terminal transport checks with python -m pytest tests/test_done_declaration.py."""
import asyncio
import json
import threading
from types import SimpleNamespace as NS

import pytest
import requests

from wia.runtime.agent.zero_shot import run_episode
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope, validate_episode_evaluation
from harness_integration.test_agent_control import claim, control_http
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_zero_shot_agent import Agent, Environment


PRIVATE = 'PRIVATE_OBJECT_ID_AND_CORRECTNESS'


class NativeDone:
    requires_done_declaration = True

    def __init__(self, *, fail=False):
        self.declarations, self.records, self.samples = [], 0, 0
        self.fail = fail

    def value(self, observation, *, finalized=False):
        return {'schema': 'terminal_native_fixture', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'task_success': finalized,
            'evaluation_validity': 'complete' if finalized else 'pending_done',
            'evaluation_finalized': finalized, 'private_object': PRIVATE,
            'completion_items': [{'id': 'terminal', 'completed': True}] if finalized else None,
            'completed_goals': 1 if finalized else None, 'total_goals': 1 if finalized else None,
            'completion_score': 1. if finalized else None, 'score_percent': 100. if finalized else None,
            'diagnostic_lower_bound': 1. if finalized else None}

    def begin_episode(self, observation):
        return self.record(observation, action=None)

    def record(self, observation, *, action):
        self.records += 1
        return self.value(observation)

    def sample_step(self):
        self.samples += 1

    def declare_done(self, observation):
        self.declarations.append(observation)
        if self.fail:
            raise RuntimeError(PRIVATE)
        return self.value(observation, finalized=True)


def envelope(tmp_path, *, fail=False):
    native = NativeDone(fail=fail)
    task = NS(definition=NS(id='terminal_fixture', status='pending', repair={},
        private=NS(evaluation=NS(model_dump=lambda: {}))), task_text='Complete the physical goal.',
        file_sha256='task-sha', source_sha256='source-sha', verify_unchanged=lambda: None)
    return CategoryEvaluationEnvelope(native, task, tmp_path), native


def post_on_owner(backend, url, body):
    responses = []
    thread = threading.Thread(target=lambda: responses.append(requests.post(url + '/api/agent/done',
        json=body, timeout=3)))
    thread.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=1)
    backend.process_next()
    thread.join(timeout=3)
    return responses[0]


def test_http_done_is_owned_nonmotion_fresh_private_and_idempotent(control_http):
    backend, _, url, calls, rig = control_http
    token = claim(backend, url)
    reports = []
    def report(observation):
        assert threading.current_thread() is threading.main_thread()
        reports.append(observation)
        return {'correct': True, 'private_object': PRIVATE}
    rig.task_evaluator = NS(requires_done_declaration=True, report_done=report)
    rig.pipeline.og.sim.current_time = 7.
    rig.pipeline.og.sim.step = lambda: pytest.fail('DONE must not step simulation')
    source = backend.observation_source
    before = source.agent_observation()
    body = {'snapshot_id': before.snapshot_id, 'action_step': before.action_step, 'control_token': token}
    cycle = rig.cycle
    response = post_on_owner(backend, url, body)
    assert response.status_code == 200
    receipt = response.json()
    assert set(receipt) == {'ok', 'message', 'snapshot_id', 'action_step'}
    assert receipt['snapshot_id'] > before.snapshot_id and receipt['action_step'] == before.action_step
    assert len(reports) == 1 and reports[0] is source.agent_observation()
    assert rig.cycle == cycle + 1 and rig.pipeline.og.sim.current_time == 7.
    assert backend.steps == source.action_step == 0 and calls == []
    assert PRIVATE not in response.text and 'correct' not in response.text and 'task_success' not in response.text
    rig.cycle += 10  # Later world state cannot replace the accepted terminal boundary.
    repeated = post_on_owner(backend, url, body)
    assert {key: repeated.json()[key] for key in ('ok', 'message', 'snapshot_id', 'action_step')} == {
        key: receipt[key] for key in ('ok', 'message', 'snapshot_id', 'action_step')}
    assert len(reports) == 1 and rig.cycle == cycle + 11


def test_done_rejects_token_extra_fields_and_stale_boundary(control_http):
    backend, _, url, _, rig = control_http
    token = claim(backend, url)
    source = backend.observation_source
    before = source.agent_observation()
    body = {'snapshot_id': before.snapshot_id, 'action_step': 0, 'control_token': token}
    assert requests.post(url+'/api/agent/done', json=dict(body, control_token='wrong')).status_code == 403
    assert requests.post(url+'/api/agent/done', json=dict(body, answer='PRIVATE')).status_code == 400
    cycle = rig.cycle
    stale = post_on_owner(backend, url, dict(body, snapshot_id=before.snapshot_id + 1))
    assert stale.status_code == 409 and rig.cycle == cycle
    assert not source._done_declaration


def test_source_done_unsupported_noop_and_owner_thread_guard():
    source, rig = source_and_rig()
    before = source.capture(action_boundary=True).observation
    rig.task_evaluator = NS(record=lambda *args, **kwargs: pytest.fail('Nonterminal scoring changed'))
    cycle = rig.cycle
    receipt = source.report_done(snapshot_id=before.snapshot_id, action_step=0)
    assert receipt['snapshot_id'] == before.snapshot_id and rig.cycle == cycle
    errors = []
    def wrong_owner():
        try:
            source.report_done(snapshot_id=before.snapshot_id, action_step=0)
        except RuntimeError as error:
            errors.append(str(error))
    thread = threading.Thread(target=wrong_owner)
    thread.start(); thread.join(timeout=2)
    assert errors and 'main thread' in errors[0]
    del rig.task_evaluator
    source.begin_episode('A new episode.')
    assert source._done_declaration is None


def test_envelope_keeps_pending_scores_null_and_freezes_first_done(tmp_path):
    wrapper, native = envelope(tmp_path)
    initial = NS(snapshot_id=1, action_step=0)
    pending = wrapper.begin_episode(initial)
    assert pending['progress']['status'] == pending['evaluation_validity'] == 'pending_done'
    assert pending['task_success'] is False and pending['evaluation_finalized'] is False
    assert all(pending['progress'][key] is None for key in (
        'completion_items', 'completed_goals', 'total_goals', 'completion_score', 'score_percent', 'diagnostic_lower_bound'))
    terminal_observation = NS(snapshot_id=2, action_step=0)
    terminal = wrapper.report_done(terminal_observation)
    assert terminal['task_success'] and terminal['evaluation_finalized']
    assert wrapper.report_done(NS(snapshot_id=9, action_step=0)) == terminal
    assert wrapper.record(NS(snapshot_id=10, action_step=1), action='BASE_FWD') == terminal
    wrapper.sample_step()
    assert len(native.declarations) == 1 and native.records == 1 and native.samples == 0
    assert json.loads((tmp_path/'state.json').read_text())['snapshot_id'] == 2
    wrapper.begin_episode(NS(snapshot_id=11, action_step=0))
    assert wrapper.done_result is None


def test_failed_native_done_is_explicit_and_cannot_be_rescored(tmp_path):
    wrapper, native = envelope(tmp_path, fail=True)
    wrapper.begin_episode(NS(snapshot_id=1, action_step=0))
    for snapshot in (2, 3):
        with pytest.raises(RuntimeError, match='did not finalize'):
            wrapper.report_done(NS(snapshot_id=snapshot, action_step=0))
    assert len(native.declarations) == 1 and wrapper.done_result['task_success'] is False


class DoneEnvironment(Environment):
    def __init__(self, *, fail_done=False):
        super().__init__()
        self.declarations, self.finalized, self.fail_done = [], False, fail_done

    async def report_done(self, observation):
        self.declarations.append(observation)
        if self.fail_done:
            raise RuntimeError('Terminal declaration rejected')
        after = self.source.capture(action_boundary=True).observation
        self.finalized = True
        return {'ok': True, 'message': 'DONE declaration recorded.',
                'snapshot_id': after.snapshot_id, 'action_step': after.action_step}


def test_runner_genuine_done_finalizes_before_success_read_and_saves_fresh_observation(tmp_path):
    env = DoneEnvironment()
    def success():
        return env.finalized
    result = asyncio.run(run_episode(env, Agent('{"action":"DONE"}'), tmp_path, success_provider=success))
    assert result['termination_reason'] == 'model_done' and result['task_success'] is True
    assert len(env.declarations) == 1 and not env.calls
    row = json.loads((tmp_path/'steps.jsonl').read_text().splitlines()[0])
    assert (tmp_path/'step_0001/done_declaration.json').exists()
    after = json.loads((tmp_path/row['next_observation']).read_text())
    assert after['snapshot_id'] == row['done_declaration']['receipt']['snapshot_id']
    assert after['action_step'] == 0


def test_runner_budget_is_not_done_and_pending_task_cannot_succeed_early(tmp_path):
    env = DoneEnvironment()
    result = asyncio.run(run_episode(env, Agent('{"action":"BASE_FWD"}'), tmp_path,
        max_steps=1, success_provider=lambda: env.finalized))
    assert result['termination_reason'] == 'max_steps' and not result['task_success']
    assert env.declarations == [] and env.calls == ['BASE_FWD']


def test_runner_declaration_error_does_not_claim_model_done_or_final_score(tmp_path):
    env = DoneEnvironment(fail_done=True)
    reads = []
    def success():
        reads.append(env.finalized)
        return False
    result = asyncio.run(run_episode(env, Agent('{"action":"DONE"}'), tmp_path,
        success_provider=success))
    assert result['termination_reason'] == 'runtime_error' and result['task_success'] is None
    assert 'Terminal declaration rejected' in result['done_declaration_error']
    assert reads == [False]


def test_runner_qa_unsure_synthetic_budget_and_forced_done_do_not_declare(tmp_path):
    from harness_integration.test_rigid_qa_protocol import QAAgent
    from harness_integration.test_tabletop_scenes import manifest
    from wia.tasks.rigid_qa import RigidContainmentQA
    scene, _ = manifest(tmp_path)
    qa = RigidContainmentQA(scene)
    env = DoneEnvironment()
    agent = QAAgent(qa, [('DONE', 'unsure', .1), ('BASE_FWD', 'unsure', .1),
                        ('DONE', 'medium container', .3)])
    result = asyncio.run(run_episode(env, agent, tmp_path/'episode', max_steps=2))
    assert env.declarations == [] and result['qa_stop_reason'] == 'max_steps'


def test_runner_temporal_phase_one_done_does_not_declare_terminal(tmp_path):
    from harness_integration.test_temporal_phase_transport import OPTION, SessionAgent, TemporalEnvironment, reply
    env = TemporalEnvironment(tmp_path/'private')
    phases = []
    async def report_done(observation):
        phases.append(env.temporal_phase)
        return {'ok': True, 'message': 'DONE declaration recorded.',
                'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    env.report_done = report_done
    result = asyncio.run(run_episode(env, SessionAgent(reply(text='PHASE1_COMPLETE'),
        reply(text=f'ANSWER: {OPTION}')), tmp_path/'episode', max_steps=4))
    assert result['termination_reason'] == 'model_done' and phases == ['phase_2']


@pytest.mark.parametrize('available', [False, True])
def test_mcp_done_learns_capability_and_preserves_legacy_servers(control_http, monkeypatch, available):
    from mcp import Client
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.web.mcp_server import create_server
    from harness_integration.test_agent_control import pump
    backend, _, url, calls, rig = control_http
    original = backend.state
    if not available:
        def legacy_state():
            state = original()
            state.pop('done_declaration_available')
            state.pop('requires_done_declaration')
            return state
        monkeypatch.setattr(backend, 'state', legacy_state)
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client', lambda *args, **kwargs: Client(create_server(url)))
    reports = []
    routes = []
    original_post = requests.post
    def post(url, *args, **kwargs):
        routes.append(url)
        return original_post(url, *args, **kwargs)
    monkeypatch.setattr(requests, 'post', post)
    async def run():
        finished = asyncio.Event()
        worker = asyncio.create_task(pump(backend, finished))
        try:
            async with MCPRobotEnvironment(url, run_id='terminal-test', task_text='Approach the table.') as env:
                assert env.done_declaration_available is available
                rig.task_evaluator = NS(requires_done_declaration=True,
                    report_done=lambda observation: reports.append(observation))
                observation = await env.observe()
                cycle = rig.cycle
                receipt = await env.report_done(observation)
                assert receipt['ok'] and rig.cycle == cycle + int(available)
                assert (receipt['snapshot_id'] > observation.snapshot_id) is available
                event = env._pending_input_events[-1]
                assert event['status'] == ('received' if available else 'local_noop')
                if not available:
                    assert 'does not advertise' in receipt['message']
        finally:
            finished.set()
            await worker
    asyncio.run(run())
    assert len(reports) == int(available) and not calls
    assert sum(route.endswith('/api/agent/done') for route in routes) == int(available)


def test_supported_done_failure_has_no_legacy_fallback():
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    env = MCPRobotEnvironment('http://localhost:8615', run_id='test', task_text='Finish.')
    env.done_declaration_available = True
    calls = []
    async def failed(path, body):
        calls.append(path)
        raise RuntimeError('Supported terminal endpoint failed')
    env._http = failed
    with pytest.raises(RuntimeError, match='terminal endpoint failed'):
        asyncio.run(env.report_done(NS(snapshot_id=1, action_step=0)))
    assert calls == ['/api/agent/done']


def test_canonical_validator_keeps_pending_unscored_and_terminal_results_boolean(tmp_path):
    wrapper, _ = envelope(tmp_path)
    wrapper.task.definition.kind = 'category_robot_task'
    observation = NS(snapshot_id=1, action_step=0)
    pending = wrapper.begin_episode(observation)
    assert validate_episode_evaluation(pending, observation, wrapper.task.task_text, wrapper.task) is None
    terminal = wrapper.report_done(observation)
    for success in (False, True):
        value = dict(terminal, task_success=success)
        assert validate_episode_evaluation(value, observation, wrapper.task.task_text, wrapper.task) is success
    # Nonterminal category tasks keep their existing boolean consumer contract.
    other_category = dict(pending)
    other_category.pop('requires_done_declaration')
    assert validate_episode_evaluation(other_category, observation, wrapper.task.task_text, wrapper.task) is False
    for success in (False, True):
        legacy = {'schema': 'private_grasp_lift_v1', 'task_success': success,
                  'task_text': wrapper.task.task_text, 'action_step': 0}
        assert validate_episode_evaluation(legacy, observation, wrapper.task.task_text) is success


@pytest.mark.parametrize('field,value', [('schema', 'other-schema'), ('task_id', 'other-task'),
    ('task_file_sha256', 'other-task-hash'), ('source_sha256', 'other-source-hash'),
    ('snapshot_id', 2), ('action_step', 1), ('episode', 2), ('task_text', 'Other goal.')])
def test_pending_validator_still_rejects_mismatched_native_identity(tmp_path, field, value):
    wrapper, _ = envelope(tmp_path)
    wrapper.task.definition.kind = 'category_robot_task'
    observation = NS(snapshot_id=1, action_step=0)
    pending = wrapper.begin_episode(observation)
    with pytest.raises(RuntimeError, match='current episode/action boundary'):
        validate_episode_evaluation(dict(pending, **{field: value}), observation,
                                    wrapper.task.task_text, wrapper.task)


def test_runner_budget_retains_canonical_pending_score_as_none_without_done(tmp_path):
    env = DoneEnvironment()
    wrapper, native = envelope(tmp_path/'private')
    wrapper.task.definition.kind = 'category_robot_task'
    wrapper.task.task_text = env.source.task_text
    wrapper.begin_episode(env.source.agent_observation())
    env.rig.task_evaluator = wrapper
    def success():
        data = json.loads((wrapper.output/'state.json').read_text())
        return validate_episode_evaluation(data, env.source.agent_observation(),
                                          wrapper.task.task_text, wrapper.task)
    result = asyncio.run(run_episode(env, Agent('{"action":"BASE_FWD"}'), tmp_path/'episode',
                                    max_steps=1, success_provider=success))
    assert result['termination_reason'] == 'max_steps' and result['task_success'] is None
    assert env.calls == ['BASE_FWD'] and not env.declarations and not native.declarations
    assert json.loads((tmp_path/'episode/summary.json').read_text())['task_success'] is None

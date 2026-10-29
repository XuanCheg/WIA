"""Validate one-episode temporal events using CPU native and camera fixtures.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_temporal_phase_transport.py
No GPU, native simulator, model, motor action, campaign or capture job is launched.
Only HTTP/MCP cases require the optional MCP SDK.
"""
import asyncio
from dataclasses import replace
import importlib.util
import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest
import requests

from core.agent.decision import ModelReply
from wia.runtime.agent.zero_shot import run_episode
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope, validate_episode_evaluation
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_cognitive_arrival_transport import MetadataAgent
from harness_integration.test_zero_shot_agent import Environment


HAS_MCP = importlib.util.find_spec('mcp') is not None
if HAS_MCP:
    from harness_integration.test_agent_control import claim, control_http, pump
else:
    @pytest.fixture
    def control_http():
        pytest.skip('Optional MCP SDK unavailable; pure temporal tests still run.')


PRIVATE = 'PRIVATE_TEMPORAL_BOX_ID_AND_CORRECTNESS'
OPTION = 'Option B'


class Task:
    definition = NS(kind='category_robot_task', id='temporal_transport_fixture')
    task_text = 'Inspect Phase 1, then compare Phase 2 using the earlier observations. Options: Option A, Option B.'
    file_sha256, source_sha256 = 'temporal-task-sha', 'temporal-source-sha'

    def __init__(self):
        self.checks = 0

    def verify_unchanged(self):
        self.checks += 1


class PhaseController:
    def __init__(self, rig, events):
        self.rig, self.events = rig, events
        self.phase, self.advances = 'phase_1', 0

    def advance(self):
        assert threading.current_thread() is threading.main_thread()
        assert self.phase == 'phase_1'
        self.events.append(('advance', self.phase))
        self.advances += 1
        self.phase = 'phase_2'


class NativeTemporal:
    """Private preparation/answer stub; phase mutation belongs to the controller."""
    def __init__(self, controller, events):
        self.controller, self.events = controller, events
        self.begins, self.samples = 0, 0
        self.prepared, self.answers = [], []
        self.allow_transition = True

    def begin_episode(self, observation):
        self.begins += 1
        return self.record(observation, action=None)

    def sample_step(self):
        self.samples += 1

    def prepare_phase_transition(self, observation):
        assert threading.current_thread() is threading.main_thread()
        if not self.allow_transition:
            raise ValueError(PRIVATE)
        assert self.controller.phase == 'phase_1'
        self.prepared.append((observation.snapshot_id, observation.action_step, observation.agentview_rgb.copy()))
        self.events.append(('prepare', observation.snapshot_id, observation.action_step))
        return {'private': PRIVATE}

    def report_answer(self, answer, observation):
        assert threading.current_thread() is threading.main_thread()
        assert self.controller.phase == 'phase_2'
        self.answers.append((answer, observation.snapshot_id, observation.action_step))
        self.events.append(('answer', answer, observation.snapshot_id, observation.action_step))
        return {'correct': answer == OPTION, 'private': PRIVATE}

    def record(self, observation, *, action):
        self.events.append(('record', self.controller.phase, observation.snapshot_id, observation.action_step, action))
        return {'schema': 'private_temporal_transport_fixture', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'task_success': bool(self.answers and self.answers[-1][0] == OPTION),
            'phase': self.controller.phase, 'private_native_identity_and_answer': PRIVATE,
            'correct_option': OPTION, 'answers': list(self.answers)}


def install_temporal(source, output, monkeypatch=None):
    rig, task, events = source.session, Task(), []
    # Native ESI episodes now require one reset-bound overview and all mounted
    # views. This fixture exercises transport, not visual task qualification.
    from core.observations import InitialOverview
    from core.robot_cameras import ADDITIONAL_RGB_NAMES
    rig.initial_overview_identity = {'episode_id': 'temporal-fixture', 'reset_id': 'temporal-reset', 'seed': 0}
    def initial_overview_capture(**boundary):
        rig.initial_overview_identity.update(episode_id=getattr(rig, 'episode_id', 'temporal-fixture'))
        return InitialOverview(rgb=np.full((12, 16, 3), 170, dtype=np.uint8),
            **rig.initial_overview_identity, **boundary, extent_xy_m=(-1., -1., 1., 1.),
            image_right_world_xy=(1., 0.), image_up_world_xy=(0., 1.),
            robot_start_position_m=(0., 0., 0.), robot_start_yaw_rad=0.)
    rig.initial_overview_capture = initial_overview_capture
    controller = PhaseController(rig, events)
    native = NativeTemporal(controller, events)
    envelope = CategoryEvaluationEnvelope(native, task, output)
    rig.temporal_phase_controller = controller
    rig.task_evaluator = envelope
    rig._step_camera_data = None
    rig.reuse_step_observations = True
    rig.pipeline.og.sim.current_time = 7.
    rig.invalidations = 0
    original_observe = rig.observe

    def observe(**kwargs):
        raw = original_observe(**kwargs)
        level = 11 if controller.phase == 'phase_1' else 88
        # Cached step data deliberately represents an obsolete phase-1 frame.
        if rig._step_camera_data is not None:
            level = 11
        raw['head_rgb'] = np.full_like(raw['head_rgb'], level)
        raw['wrist_rgb'] = np.full_like(raw['wrist_rgb'], level + 20)
        raw.update({name: np.full_like(raw['head_rgb'], level + index + 1)
                    for index, name in enumerate(ADDITIONAL_RGB_NAMES)})
        return raw

    def invalidate():
        rig.invalidations += 1
        rig._step_camera_data = None

    def forbidden_step(*args, **kwargs):
        pytest.fail('A temporal transport event must not step physics.')

    if monkeypatch is None:
        rig.observe, rig.invalidate_camera_cache = observe, invalidate
        rig.pipeline.og.sim.step = forbidden_step
    else:
        monkeypatch.setattr(rig, 'observe', observe)
        monkeypatch.setattr(rig, 'invalidate_camera_cache', invalidate, raising=False)
        monkeypatch.setattr(rig.pipeline.og.sim, 'step', forbidden_step, raising=False)
    return task, native, controller, envelope, events


class TemporalEnvironment(Environment):
    temporal_episode = True
    temporal_policy = {'phase_1_max_decisions': 2}

    def __init__(self, output):
        super().__init__()
        self.task, self.native, self.controller, self.envelope, self.events = install_temporal(self.source, output)
        self.source.begin_episode(self.task.task_text)
        self.action_context = {'temporal_episode': True, 'temporal_phase': 'phase_1'}
        self.initialization = {'scene_reset': False, 'temporal_episode': True, 'temporal_phase': 'phase_1'}

    @property
    def temporal_phase(self):
        return self.controller.phase

    async def report_temporal(self, event, answer, observation):
        result = self.source.report_temporal(event, answer,
            snapshot_id=observation.snapshot_id, action_step=observation.action_step)
        self.action_context['temporal_phase'] = self.temporal_phase
        return result

    def success(self):
        sealed = json.loads((self.envelope.output / 'state.json').read_text())
        return validate_episode_evaluation(sealed, self.source.agent_observation(), self.task.task_text, self.task)


def reply(action='DONE', text='', *, note=None):
    command = {'action': action}
    raw = {'command': command, 'memory_update': note} if note is not None else command
    return ModelReply(json.dumps(raw), {'world_observation': text})


class SessionAgent(MetadataAgent):
    def __init__(self, *replies):
        super().__init__(*replies)
        self.closes = 0

    async def close(self):
        self.closes += 1


def test_phase_one_done_continues_same_model_session_and_preserves_memory(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    agent = SessionAgent(reply(text='PHASE1_COMPLETE', note='Phase 1 visible contents remembered.'),
        reply(text=f'ANSWER: {OPTION}', note='Compared with the earlier observations.'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path / 'episode', memory_mode='model_notes',
                                      success_provider=environment.success))
    assert summary['termination_reason'] == 'model_done' and summary['task_success'] is True
    assert summary['agent_decisions'] == summary['model_calls'] == 2
    assert not environment.calls and summary['executed_actions'] == []
    assert all(summary[key] == 0 for key in ('macro_actions', 'primitive_actions', 'physics_steps',
        'controller_env_steps', 'actual_fixed_primitive_executions', 'equivalent_fixed_primitives'))
    assert environment.native.begins == environment.controller.advances == agent.closes == 1
    first, second = agent.inputs
    assert first.observation.action_step == second.observation.action_step == 0
    assert first.observation.snapshot_id < second.observation.snapshot_id
    assert np.all(first.observation.agentview_rgb == 11) and np.all(second.observation.agentview_rgb == 88)
    assert second.public_memory[0]['text'] == 'Phase 1 visible contents remembered.'
    assert second.action_context['temporal_phase'] == 'phase_2'
    assert 'BASE_FWD' in first.available_actions and 'GRASP' not in first.available_actions
    assert 'BASE_FWD' in first.schema['properties']['action']['enum']
    assert 'BASE_FWD' in second.available_actions
    assert 'BASE_FWD' in json.loads((tmp_path / 'episode/run.json').read_text())['available_actions']
    assert environment.native.answers == [(OPTION, second.observation.snapshot_id, 0)]
    records = [json.loads(line) for line in (tmp_path / 'episode/steps.jsonl').read_text().splitlines()]
    assert records[0]['temporal_continue_no_motion'] and records[0]['next_observation']
    after = json.loads((tmp_path / 'episode' / records[0]['next_observation']).read_text())
    assert after['snapshot_id'] == second.observation.snapshot_id and after['action_step'] == 0
    assert [row['temporal_event']['event'] for row in records] == ['phase_1_complete', 'answer']
    assert all(row['action_result'] is None for row in records)
    assert PRIVATE not in json.dumps([request.action_context for request in agent.inputs])


@pytest.mark.parametrize('text', [
    '', 'I am ready for PHASE1_COMPLETE now.', 'PHASE1_COMPLETE\nPHASE1_COMPLETE',
    f'PHASE1_COMPLETE\nANSWER: {OPTION}', f'ANSWER: {OPTION}', None,
])
def test_plain_done_requests_transition_regardless_of_prose(text, tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    agent = MetadataAgent(reply(text=text), reply(text='PHASE1_COMPLETE'), reply(text=f'ANSWER: {OPTION}'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path / 'episode', success_provider=environment.success))
    assert summary['termination_reason'] == 'model_done'
    assert summary['agent_decisions'] == 3 and not environment.calls
    assert environment.controller.advances == 1 and len(environment.native.prepared) == 1
    assert agent.inputs[0].observation.snapshot_id < agent.inputs[1].observation.snapshot_id
    assert agent.inputs[1].previous_phase_observation is agent.inputs[0].observation


def test_phase_two_done_without_one_bounded_answer_continues_without_motion(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    agent = MetadataAgent(reply(text='PHASE1_COMPLETE'), reply(text='ANSWER:'),
        reply(text=f'ANSWER: {OPTION}\nANSWER: {OPTION}'), reply(text='ANSWER: ' + 'x' * 161),
        reply(text=f'ANSWER: {OPTION}'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path / 'episode', max_steps=5,
                                      success_provider=environment.success))
    assert summary['termination_reason'] == 'model_done' and summary['task_success'] is True
    assert len(environment.native.answers) == 1 and not environment.calls
    assert summary['agent_decisions'] == 5 and summary['primitive_actions'] == 0
    assert all(request.observation.action_step == 0 for request in agent.inputs)


def test_markers_from_invalid_commands_or_motion_do_not_trigger_phase_events(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    agent = MetadataAgent(reply('FORBIDDEN', 'PHASE1_COMPLETE'), reply('BASE_FWD', 'PHASE1_COMPLETE'),
        reply(text='PHASE1_COMPLETE'), reply('BASE_FWD', 'PHASE1_COMPLETE'), reply(text=f'ANSWER: {OPTION}'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path / 'episode', success_provider=environment.success))
    assert summary['termination_reason'] == 'model_done' and environment.calls == ['BASE_FWD', 'BASE_FWD']
    assert environment.controller.advances == 1 and environment.native.begins == 1
    assert environment.native.prepared[0][1] == 1 and environment.native.answers[-1][2] == 2
    assert summary['macro_actions'] == summary['primitive_actions'] == 2
    assert all('BASE_FWD' in request.available_actions for request in agent.inputs[:3])
    assert 'BASE_FWD' in agent.inputs[3].available_actions


def test_temporal_markers_leave_unrelated_done_behavior_unchanged(tmp_path):
    environment = Environment()
    summary = asyncio.run(run_episode(environment, MetadataAgent(reply(text='PHASE1_COMPLETE')), tmp_path))
    assert summary['termination_reason'] == 'model_done' and not environment.calls
    assert 'temporal_episode' not in summary


def test_temporal_episode_rejects_initial_view_only_without_changing_configuration(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    with pytest.raises(ValueError, match='require current observations'):
        asyncio.run(run_episode(environment, MetadataAgent(), tmp_path / 'episode', observation_condition='initial_view_only'))
    assert environment.controller.phase == 'phase_1' and not environment.native.prepared


def test_phase_one_plain_done_transitions_without_ending_episode(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    agent = MetadataAgent(reply(), reply(text=f'ANSWER: {OPTION}'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path / 'episode', max_steps=3))
    assert summary['termination_reason'] == 'model_done' and summary['model_declared_done']
    assert summary['agent_decisions'] == 2 and summary['primitive_actions'] == 0
    assert not environment.calls and environment.controller.advances == 1


def test_runner_refuses_stale_observation_after_phase_transition(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    old = environment.source.agent_observation()

    async def stale_observe():
        return old

    environment.observe = stale_observe
    summary = asyncio.run(run_episode(environment, MetadataAgent(reply(text='PHASE1_COMPLETE')),
                                      tmp_path / 'episode'))
    assert summary['termination_reason'] == 'observation_error'
    assert environment.controller.advances == 1 and not environment.calls and not environment.native.answers


@pytest.mark.parametrize('field', ('snapshot_id', 'action_step'))
def test_source_stale_identity_rejects_before_native_or_phase_mutation(field, tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    observation = environment.source.agent_observation()
    identity = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    identity[field] += 1
    cycle = environment.rig.cycle
    with pytest.raises(ValueError, match='boundary changed'):
        environment.source.report_temporal('phase_1_complete', None, **identity)
    assert environment.controller.advances == 0 and not environment.native.prepared
    assert environment.rig.cycle == cycle and environment.source.agent_observation() is observation


def test_source_transition_orders_private_seals_and_refreshes_same_boundary(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    source, rig = environment.source, environment.rig
    old = source.agent_observation()
    rig._step_camera_data = {'head': {'rgb': 'obsolete phase-1 cache'}}
    rig.reuse_step_observations = True
    counters = (source.action_step, source.history, source.last_action_feedback, rig.pipeline.og.sim.current_time)
    result = source.report_temporal('phase_1_complete', None, snapshot_id=old.snapshot_id, action_step=old.action_step)
    new = source.agent_observation()
    assert result == {'ok': True, 'message': 'Temporal event recorded.', 'phase': 'phase_2', 'observation_changed': True}
    assert new.action_step == old.action_step and new.snapshot_id > old.snapshot_id
    assert np.all(old.agentview_rgb == 11) and np.all(new.agentview_rgb == 88)
    assert rig._step_camera_data is None and rig.invalidations > 0
    assert counters == (source.action_step, source.history, source.last_action_feedback, rig.pipeline.og.sim.current_time)
    prepare = next(index for index, event in enumerate(environment.events) if event[0] == 'prepare')
    assert [event[0] for event in environment.events[prepare:]] == ['prepare', 'record', 'advance', 'record']
    assert environment.events[prepare + 1][1:4] == ('phase_1', old.snapshot_id, old.action_step)
    assert environment.events[-1][1:4] == ('phase_2', new.snapshot_id, new.action_step)
    assert environment.native.samples == 0 and environment.native.begins == 1
    cycles = rig.cycle
    answer = source.report_temporal('answer', OPTION, snapshot_id=new.snapshot_id, action_step=new.action_step)
    assert answer['observation_changed'] is False and rig.cycle == cycles
    assert environment.success()
    with pytest.raises(ValueError, match='unavailable'):
        source.report_temporal('phase_1_complete', None, snapshot_id=new.snapshot_id, action_step=new.action_step)
    assert environment.controller.advances == 1


def test_native_transition_rejection_and_nonowner_calls_leave_scene_unmodified(tmp_path):
    environment = TemporalEnvironment(tmp_path / 'private')
    source = environment.source
    observation = source.agent_observation()
    identity = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    environment.native.allow_transition = False
    with pytest.raises(ValueError, match=PRIVATE):
        source.report_temporal('phase_1_complete', None, **identity)
    errors = []

    def worker():
        try:
            source.report_temporal('phase_1_complete', None, **identity)
        except RuntimeError as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert len(errors) == 1 and environment.controller.advances == 0
    assert source.agent_observation() is observation


@pytest.fixture
def temporal_http(control_http, tmp_path, monkeypatch):
    backend, viewer, url, calls, rig = control_http
    task, native, controller, envelope, events = install_temporal(backend.observation_source, tmp_path / 'private', monkeypatch)
    return backend, viewer, url, calls, rig, task, native, controller, envelope, events


def body(backend, token, event='phase_1_complete', answer=None):
    observation = backend.observation_source.agent_observation()
    return {'event': event, 'answer': answer, 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'control_token': token}


def post_on_owner(backend, url, payload):
    responses = []
    worker = threading.Thread(target=lambda: responses.append(requests.post(url + '/api/agent/temporal', json=payload, timeout=4)))
    worker.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=2)
    assert backend.process_next()
    worker.join(timeout=4)
    assert not worker.is_alive() and len(responses) == 1
    return responses[0]


def test_http_temporal_validates_lease_fields_shapes_and_current_boundary(temporal_http):
    backend, _, url, calls, _, _, native, controller, _, _ = temporal_http
    assert requests.post(url + '/api/agent/temporal', json=body(backend, 'wrong'), timeout=3).status_code == 403
    token = claim(backend, url)
    valid = body(backend, token)
    invalid = [(dict(valid, control_token=None), 403), (dict(valid, control_token='wrong'), 403),
        ({key: value for key, value in valid.items() if key != 'answer'}, 400),
        (dict(valid, private_correct_option=OPTION), 400), (dict(valid, event='advance_anyway'), 400),
        (dict(valid, answer=OPTION), 400), (dict(valid, event='answer', answer=''), 400),
        (dict(valid, event='answer', answer='x' * 161), 400), (dict(valid, snapshot_id=True), 400)]
    for payload, status in invalid:
        response = requests.post(url + '/api/agent/temporal', json=payload, timeout=3)
        assert response.status_code == status and not response.json()['ok']
        assert PRIVATE not in response.text
    for key in ('snapshot_id', 'action_step'):
        payload = dict(valid)
        payload[key] += 1
        response = post_on_owner(backend, url, payload)
        assert response.status_code == 409 and not response.json()['ok']
    early_answer = post_on_owner(backend, url, body(backend, token, 'answer', OPTION))
    assert early_answer.status_code == 409
    assert not native.prepared and not native.answers and controller.advances == 0 and not calls


def test_phase_one_hard_fence_rejects_human_fixed_and_parameterized_motion_before_enqueue(temporal_http):
    backend, _, url, calls, _, _, native, controller, _, _ = temporal_http
    backend.executor.capabilities = replace(backend.executor.capabilities, gripper=True)
    capabilities = backend.executor.capabilities
    for route, payload in (('move', {'token': 'GRASP'}), ('step', {'tokens': ['RELEASE']})):
        response = requests.post(url + '/api/' + route, json=payload, timeout=3)
        assert response.status_code == 409 and not response.json()['ok']
    token = claim(backend, url)
    for route, payload in (
            ('agent/step', {'action': 'GRASP', 'control_token': token}),
            ('agent/command', {'command': {'action': 'RELEASE'}, 'control_token': token}),
            ('move', {'token': 'BASE_FWD'})):
        response = requests.post(url + '/api/' + route, json=payload, timeout=3)
        assert response.status_code == 409 and not response.json()['ok']
    assert not calls and not native.prepared and controller.phase == 'phase_1'
    assert backend.commands.empty() and not backend.busy
    assert backend.executor.capabilities is capabilities and capabilities.base_translation
    # An authorized survey move queues normally and must run on the owner thread.
    responses = []
    worker = threading.Thread(target=lambda: responses.append(requests.post(url + '/api/agent/step',
        json={'action': 'BASE_FWD', 'control_token': token}, timeout=4)))
    worker.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=2)
    assert backend.process_next()
    worker.join(timeout=4)
    assert not worker.is_alive() and responses[0].status_code == 200
    assert calls == ['BASE_FWD'] and controller.phase == 'phase_1'
    assert not native.prepared and backend.commands.empty() and not backend.busy


def test_http_phase_event_refreshes_published_snapshot_and_preserves_lease_and_robot_counters(temporal_http):
    backend, _, url, calls, rig, _, native, controller, _, _ = temporal_http
    token = claim(backend, url)
    old = backend.agent_snapshot
    counters = (backend.steps, backend.physics_steps, backend.primitive_actions, list(backend.history), rig.pipeline.og.sim.current_time)
    # Exercise the owner's explicit publication fallback without a capture callback.
    backend.observation_source.on_capture = None
    response = post_on_owner(backend, url, body(backend, token))
    assert response.status_code == 200 and response.json()['observation_changed'] is True
    assert backend.agent_snapshot is backend.observation_source.agent_observation()
    assert backend.agent_snapshot.snapshot_id > old.snapshot_id
    assert backend.agent_snapshot.action_step == old.action_step
    assert np.all(old.agentview_rgb == 11) and np.all(backend.agent_snapshot.agentview_rgb == 88)
    assert counters == (backend.steps, backend.physics_steps, backend.primitive_actions, backend.history, rig.pipeline.og.sim.current_time)
    assert backend._control_token == token and backend.control_owner == 'agent' and not calls
    state = requests.get(url + '/api/agent/state', timeout=3).json()
    assert state['temporal_episode'] is True and state['temporal_phase'] == 'phase_2'
    assert PRIVATE not in json.dumps(state)
    duplicate = post_on_owner(backend, url, body(backend, token))
    assert duplicate.status_code == 409 and controller.advances == 1 and len(native.prepared) == 1


@pytest.mark.parametrize('answer', ('Option A', OPTION))
def test_http_answer_neutrality_and_native_private_reseal(temporal_http, answer):
    backend, _, url, _, rig, task, native, _, envelope, _ = temporal_http
    token = claim(backend, url)
    assert post_on_owner(backend, url, body(backend, token)).status_code == 200
    observation = backend.agent_snapshot
    cycle = rig.cycle
    response = post_on_owner(backend, url, body(backend, token, 'answer', answer))
    public = response.json()
    assert response.status_code == 200 and public['ok'] is True
    assert public['message'] == 'Temporal event recorded.' and public['phase'] == 'phase_2'
    assert public['observation_changed'] is False
    assert not {'answer', 'correct', 'task_success', 'native_result', 'boxes', 'identities'} & set(public)
    assert PRIVATE not in response.text and rig.cycle == cycle
    assert native.answers == [(answer, observation.snapshot_id, observation.action_step)]
    sealed = json.loads((envelope.output / 'state.json').read_text())
    assert validate_episode_evaluation(sealed, observation, task.task_text, task) is (answer == OPTION)


def test_http_pending_timeout_cannot_mutate_phase_when_owner_later_processes_queue(temporal_http, monkeypatch):
    backend, _, url, calls, _, _, native, controller, _, _ = temporal_http
    token = claim(backend, url)
    submit = backend.submit_temporal
    monkeypatch.setattr(backend, 'submit_temporal', lambda *args, **kwargs: submit(*args, **kwargs, timeout=.02))
    response = requests.post(url + '/api/agent/temporal', json=body(backend, token), timeout=3)
    assert response.status_code == 504 and not response.json()['ok']
    assert controller.phase == 'phase_1' and not native.prepared
    assert backend.process_next()
    assert controller.advances == 0 and not native.prepared and not calls
    assert backend.commands.empty() and not backend.busy


def test_cancelled_queued_phase_event_never_advances_the_environment(temporal_http):
    backend, _, url, calls, _, _, native, controller, _, _ = temporal_http
    token = claim(backend, url)
    responses = []
    worker = threading.Thread(target=lambda: responses.append(requests.post(url + '/api/agent/temporal',
        json=body(backend, token), timeout=4)))
    worker.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=2)
    assert requests.post(url + '/api/control/cancel', json={'control_token': token}, timeout=3).json()['ok']
    assert backend.process_next()
    worker.join(timeout=4)
    assert responses[0].status_code == 409 and controller.advances == 0 and not native.prepared
    assert not calls and backend.control_owner == 'human'


def test_http_native_prepare_error_is_generic_and_does_not_advance(temporal_http):
    backend, _, url, _, _, _, native, controller, _, _ = temporal_http
    token = claim(backend, url)
    native.allow_transition = False
    response = post_on_owner(backend, url, body(backend, token))
    assert response.status_code == 409 and not response.json()['ok']
    assert PRIVATE not in response.text and controller.advances == 0


def test_actual_mcp_runner_preserves_one_lease_episode_and_phase_one_images(temporal_http, tmp_path, monkeypatch):
    from mcp import Client
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.web.mcp_server import create_server

    backend, _, url, calls, _, task, native, controller, envelope, _ = temporal_http
    backend.executor.capabilities = replace(backend.executor.capabilities, gripper=True)
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client', lambda *args, **kwargs: Client(create_server(url)))
    completed = asyncio.Event()
    agent = MetadataAgent(reply('BASE_FWD', 'Surveying the first-phase view without interacting with objects.'),
        reply(text='PHASE1_COMPLETE'), reply('BASE_FWD', 'Inspecting the second-phase view.'),
        reply(text=f'ANSWER: {OPTION}'))
    leases = []
    decide = agent.decide

    async def observe_decision(request):
        leases.append(backend._control_token)
        return await decide(request)

    agent.decide = observe_decision

    def success():
        return validate_episode_evaluation(json.loads((envelope.output / 'state.json').read_text()),
            backend.observation_source.agent_observation(), task.task_text, task)

    async def run():
        owner = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id='temporal-mcp-fixture', task_text=task.task_text) as environment:
                assert environment.initialization['temporal_episode'] is True
                assert environment.temporal_phase == 'phase_1'
                environment.temporal_policy = {'phase_1_max_decisions': 2}
                with pytest.raises(ValueError, match='survey motion'):
                    await environment.execute('GRASP')
                tools = await environment.client.list_tools()
                assert not any('temporal' in tool.name for tool in tools.tools)
                context = (await environment.client.call_tool('robot_get_context', {})).structured_content['context']
                assert context['temporal_episode'] is True and context['temporal_phase'] == 'phase_1'
                result = await run_episode(environment, agent, tmp_path / 'episode', success_provider=success)
                assert environment.temporal_phase == environment.action_context['temporal_phase'] == 'phase_2'
                return result
        finally:
            completed.set()
            await owner

    summary = asyncio.run(run())
    assert summary['termination_reason'] == 'model_done' and summary['task_success'] is True
    assert native.begins == controller.advances == 1 and calls == ['BASE_FWD', 'BASE_FWD']
    assert backend.steps == 2 and backend.control_owner == 'human' and len(set(leases)) == 1
    assert summary['phase_1_decisions'] == 2
    assert summary['agent_decisions'] == summary['model_calls'] == 4
    assert summary['macro_actions'] == summary['primitive_actions'] == 2
    old, new = agent.inputs[1].observation, agent.inputs[2].observation
    assert old.action_step == new.action_step == 1 and new.snapshot_id > old.snapshot_id
    assert np.all(old.agentview_rgb == 11) and np.all(new.agentview_rgb == 88)
    assert native.prepared[0][:2] == (old.snapshot_id, 1)
    assert agent.inputs[2].previous_phase_observation is old
    assert summary['temporal_transitions'][0]['final_phase_1_delivered'] is True
    assert native.answers == [(OPTION, agent.inputs[3].observation.snapshot_id, 2)]
    assert all('BASE_FWD' in request.available_actions and 'GRASP' not in request.available_actions
               for request in agent.inputs[:2])
    assert 'BASE_FWD' in agent.inputs[2].available_actions and 'GRASP' in agent.inputs[2].available_actions
    inputs = [json.loads(line) for line in (tmp_path / 'episode/model_inputs.jsonl').read_text().splitlines()]
    events = [row for row in inputs if row.get('tool') == 'robot_report_temporal']
    assert [row['arguments']['event'] for row in events] == ['phase_1_complete', 'answer']
    assert all(row['arguments']['control_token'] == '[redacted]' and row['passed_to_model'] is False for row in events)
    assert all(set(row['result']) == {'ok', 'message', 'phase', 'observation_changed'} for row in events)
    for path in (tmp_path / 'episode').rglob('*.json*'):
        assert PRIVATE not in path.read_text()
    assert PRIVATE in (envelope.output / 'state.json').read_text()


def test_mcp_environment_rejects_an_old_cached_view_after_phase_mutation(temporal_http, monkeypatch):
    from mcp import Client
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.web.mcp_server import create_server

    backend, _, url, _, _, task, _, controller, _, _ = temporal_http
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client', lambda *args, **kwargs: Client(create_server(url)))
    completed = asyncio.Event()

    async def run():
        owner = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id='stale-temporal-view', task_text=task.task_text) as environment:
                old = await environment.observe()
                receipt = await environment.report_temporal('phase_1_complete', None, old)
                assert receipt['observation_changed']
                monkeypatch.setattr(backend, 'agent_observation', lambda: old)
                with pytest.raises(RuntimeError, match='fresh snapshot'):
                    await environment.observe()
        finally:
            completed.set()
            await owner

    asyncio.run(run())
    assert controller.advances == 1


def test_parameterized_temporal_episode_masks_phase_one_and_restores_phase_two_controls(
        temporal_http, tmp_path, monkeypatch):
    from mcp import Client
    from core.action_compiler import MacroLimits, PrimitiveResolution
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.web.mcp_server import create_server

    backend, _, url, calls, _, task, native, _, _, _ = temporal_http
    backend.executor.capabilities = replace(backend.executor.capabilities, gripper=True)
    backend.executor.primitive_resolution = PrimitiveResolution(base_translation_m=.05)
    backend.executor.macro_limits = MacroLimits()
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client', lambda *args, **kwargs: Client(create_server(url)))
    completed = asyncio.Event()
    movement = ModelReply('{"action":"BASE_FWD","distance_m":0.10}', {'world_observation': 'PHASE1_COMPLETE'})
    agent = MetadataAgent(movement, reply(text='PHASE1_COMPLETE'), movement, reply(text=f'ANSWER: {OPTION}'))

    async def run():
        owner = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id='temporal-metric-fixture', task_text=task.task_text,
                    action_mode='parameterized') as environment:
                environment.temporal_policy = {'phase_1_max_decisions': 2}
                denied = requests.post(url + '/api/agent/command', json={'command': {'action': 'GRASP'},
                    'control_token': environment.control_token}, timeout=3)
                assert denied.status_code == 409 and backend.commands.empty()
                return await run_episode(environment, agent, tmp_path / 'metric-episode', action_mode='parameterized')
        finally:
            completed.set()
            await owner

    summary = asyncio.run(run())
    assert summary['termination_reason'] == 'model_done' and calls == ['BASE_FWD'] * 4
    assert summary['agent_decisions'] == summary['model_calls'] == 4
    assert summary['phase_1_decisions'] == 2
    assert summary['macro_actions'] == 2 and summary['primitive_actions'] == 4
    assert native.prepared[0][1] == 1 and native.answers[-1][2] == 2
    old, new = agent.inputs[1].observation, agent.inputs[2].observation
    assert old.action_step == new.action_step == 1 and new.snapshot_id > old.snapshot_id
    assert np.all(old.agentview_rgb == 11) and np.all(new.agentview_rgb == 88)
    assert agent.inputs[2].previous_phase_observation is old
    assert native.prepared[0][:2] == (old.snapshot_id, 1)
    assert summary['temporal_transitions'][0]['final_phase_1_delivered'] is True
    assert all('BASE_FWD' in request.available_actions and 'GRASP' not in request.available_actions
               for request in agent.inputs[:2])
    assert all('BASE_FWD' in json.dumps(request.schema) and 'GRASP' not in json.dumps(request.schema)
               for request in agent.inputs[:2])
    assert 'BASE_FWD' in agent.inputs[2].available_actions and 'GRASP' in agent.inputs[2].available_actions

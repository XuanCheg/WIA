"""Validate Agent arrival declarations across loop, HTTP, owner and private seal.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_cognitive_arrival_transport.py
CPU fixtures use cached synthetic cameras and a tiny private native evaluator.
No simulator, GPU, model, physical robot, campaign or capture queue is started.
HTTP/MCP cases alone skip when the optional MCP SDK is unavailable.
"""
import asyncio
import importlib.util
import json
import threading
from types import SimpleNamespace as NS

import pytest
import requests

from core.agent.decision import ModelReply
from wia.runtime.agent.zero_shot import run_episode
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope, validate_episode_evaluation
from harness_integration.test_agent_observation import source_and_rig
from harness_integration.test_zero_shot_agent import Environment


HAS_MCP = importlib.util.find_spec('mcp') is not None
if HAS_MCP:
    from harness_integration.test_agent_control import claim, control_http, pump
else:
    @pytest.fixture
    def control_http():
        pytest.skip('Optional MCP SDK unavailable; pure loop/source/envelope tests still run.')


A, B = 'room_a_0', 'room_b_1'
PRIVATE_ROOM = 'private_native_actual_room_77'
PRIVATE_REASON = 'PRIVATE_NATIVE_ARRIVAL_REASON'


class NativeArrival:
    """Expose deliberately private results so accidental transport leaks are visible."""
    def __init__(self, *, accepts=True, required=(A, B)):
        self.accepts, self.required = accepts, set(required)
        self.events, self.reports, self.threads = [], [], []
        self.samples, self.records = 0, 0
        self.stale_result = False

    def begin_episode(self, observation):
        self.reports = []
        return self.record(observation, action=None)

    def sample_step(self):
        self.samples += 1

    def report_arrival(self, room_name, observation):
        assert threading.current_thread() is threading.main_thread()
        self.threads.append(threading.get_ident())
        self.reports.append({'claimed_room': room_name, 'accepted': self.accepts,
            'reason': PRIVATE_REASON, 'actual_room': PRIVATE_ROOM,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})
        self.events.append(('report', room_name, observation.snapshot_id, observation.action_step))
        return dict(self.reports[-1])

    def record(self, observation, *, action):
        self.records += 1
        self.events.append(('record', action, observation.snapshot_id, observation.action_step))
        reported = {row['claimed_room'] for row in self.reports if row['accepted']}
        return {'schema': 'private_native_arrival_transport_fixture',
            'snapshot_id': observation.snapshot_id - int(self.stale_result),
            'action_step': observation.action_step,
            'task_success': self.required.issubset(reported), 'actual_room': PRIVATE_ROOM,
            'accepted': self.accepts, 'reason': PRIVATE_REASON, 'reports': list(self.reports)}


class LoadedTask:
    definition = NS(kind='category_robot_task', id='arrival_transport_fixture',
                    private=NS(evaluation=NS(model_dump=lambda: {'type': 'category_operation',
                        'parameters': {'required_regions': [A], 'arrival_report_required': True}})))
    task_text = 'Explore the named rooms and explicitly report arrival.'
    file_sha256, source_sha256 = 'fixture-task-sha', 'fixture-source-sha'

    def __init__(self):
        self.integrity_checks = 0

    def verify_unchanged(self):
        self.integrity_checks += 1


def attach_envelope(source, directory, *, accepts=True, required=(A, B)):
    native, task = NativeArrival(accepts=accepts, required=required), LoadedTask()
    envelope = CategoryEvaluationEnvelope(native, task, directory)
    source.session.task_evaluator = envelope
    return native, task, envelope


class MetadataAgent:
    def __init__(self, *replies):
        self.replies, self.inputs = iter(replies), []

    def description(self):
        return {'backend': 'arrival_transport_fixture'}

    async def decide(self, request):
        self.inputs.append(request)
        return next(self.replies)


class ArrivalEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.events = []

    async def report_arrivals(self, rooms, observation):
        assert observation is self.source.agent_observation()
        self.events.append(('report', list(rooms), observation.snapshot_id, observation.action_step))
        return {'ok': True, 'message': 'Arrival declarations recorded.'}

    async def execute(self, action):
        observation = self.source.agent_observation()
        self.events.append(('execute', action, observation.snapshot_id, observation.action_step))
        return await super().execute(action)


def reply(command, world_observation, *, rooms=()):
    return ModelReply(json.dumps({'action': command}), {'world_observation': world_observation, 'arrived_rooms': list(rooms)})


def test_structured_reports_precede_motion_and_done_at_their_actual_boundaries(tmp_path):
    environment = ArrivalEnvironment()
    agent = MetadataAgent(reply('BASE_FWD', 'Observed a room.', rooms=[A, A]),
        reply('DONE', 'Observed another room.', rooms=[B]))
    summary = asyncio.run(run_episode(environment, agent, tmp_path))
    assert summary['termination_reason'] == 'model_done'
    assert environment.calls == ['BASE_FWD']
    first, execute, done = environment.events
    assert first[:2] == ('report', [A]) and first[3] == 0
    assert execute == ('execute', 'BASE_FWD', first[2], 0)
    assert done[:2] == ('report', [B]) and done[3] == 1 and done[2] > first[2]
    records = [json.loads(line) for line in (tmp_path / 'steps.jsonl').read_text().splitlines()]
    assert records[0]['arrival_declarations']['rooms'] == [A]
    assert records[1]['arrival_declarations']['rooms'] == [B]
    assert records[1]['action_result'] is None
    assert all('ARRIVED' not in name for request in agent.inputs for name in request.available_actions)


@pytest.mark.parametrize('text', [
    f'I have ARRIVED: {A} now.', f'ARRIVED: {A} because I can see furniture.',
    f'ARRIVED: {A}.', f'arrived: {A}', f'ARRIVED: <{A}>',
    f'ARRIVED: {A} {B}', f'ARRIVED:{A}', 'ARRIVED: ROOM_A_0',
])
def test_prose_and_nonexact_arrival_lines_do_not_become_reports(text, tmp_path):
    environment = ArrivalEnvironment()
    summary = asyncio.run(run_episode(environment, MetadataAgent(reply('DONE', text)), tmp_path))
    assert summary['termination_reason'] == 'model_done'
    assert not environment.events and not environment.calls
    assert 'arrival_declarations' not in (tmp_path / 'steps.jsonl').read_text()


@pytest.mark.parametrize('value', [None, [], [f'ARRIVED: {A}'], {'text': f'ARRIVED: {A}'}, 17, True])
def test_nonstring_world_observation_does_not_report_or_block_a_valid_command(value, tmp_path):
    environment = ArrivalEnvironment()
    summary = asyncio.run(run_episode(environment,
        MetadataAgent(reply('BASE_FWD', value), reply('DONE', 'No arrival declaration.')), tmp_path))
    assert summary['termination_reason'] == 'model_done'
    assert environment.calls == ['BASE_FWD']
    assert [event[0] for event in environment.events] == ['execute']


def test_rejected_command_and_ignored_fields_do_not_submit_text_claims(tmp_path):
    environment = ArrivalEnvironment()
    agent = MetadataAgent(reply('FORBIDDEN', f'ARRIVED: {A}'),
        ModelReply(json.dumps({'action': 'DONE', 'extra': f'ARRIVED: {A}'}),
                   {'world_observation': f'ARRIVED: {A}'}),
        reply('DONE', 'The final accepted decision makes no arrival claim.'))
    summary = asyncio.run(run_episode(environment, agent, tmp_path))
    assert summary['termination_reason'] == 'model_done'
    assert not environment.events and not environment.calls
    record = json.loads((tmp_path / 'steps.jsonl').read_text())
    assert [bool(attempt['parse_error']) for attempt in record['attempts']] == [True, False]
    assert 'arrival_declarations' not in record


def test_exhausted_invalid_commands_never_submit_their_arrival_claims(tmp_path):
    environment = ArrivalEnvironment()
    agent = MetadataAgent(*(reply('FORBIDDEN', f'ARRIVED: {A}') for _ in range(3)))
    summary = asyncio.run(run_episode(environment, agent, tmp_path))
    assert summary['termination_reason'] == 'invalid_output'
    assert not environment.events and not environment.calls


def test_private_envelope_reseals_updated_report_result_at_the_same_boundary(tmp_path):
    native, task = NativeArrival(required=(A,)), LoadedTask()
    envelope = CategoryEvaluationEnvelope(native, task, tmp_path)
    observation = NS(snapshot_id=12, action_step=3)
    initial = envelope.begin_episode(observation)
    assert not initial['task_success'] and native.records == 1
    envelope.report_arrivals([A], observation)
    sealed = json.loads((tmp_path / 'state.json').read_text())
    assert validate_episode_evaluation(sealed, observation, task.task_text, task)
    assert sealed['snapshot_id'] == 12 and sealed['action_step'] == 3
    assert sealed['native_result']['reports'][0]['claimed_room'] == A
    assert sealed['native_result']['actual_room'] == PRIVATE_ROOM
    assert native.samples == 0 and native.records == 2 and task.integrity_checks == 2
    assert native.events[-2:] == [('report', A, 12, 3), ('record', None, 12, 3)]
    boundaries = [json.loads(line) for line in (tmp_path / 'episode_boundaries.jsonl').read_text().splitlines()]
    assert [row['task_success'] for row in boundaries] == [False, True]
    assert {(row['snapshot_id'], row['action_step']) for row in boundaries} == {(12, 3)}


def test_envelope_requires_active_episode_and_refuses_stale_native_reseal(tmp_path):
    native, task = NativeArrival(required=(A,)), LoadedTask()
    envelope = CategoryEvaluationEnvelope(native, task, tmp_path)
    observation = NS(snapshot_id=12, action_step=3)
    with pytest.raises(RuntimeError, match='active episode'):
        envelope.report_arrivals([A], observation)
    assert not native.reports
    envelope.begin_episode(observation)
    native.stale_result = True
    envelope.report_arrivals([A], observation)
    saved = json.loads((tmp_path / 'state.json').read_text())
    assert saved['snapshot_id'] == 12 and saved['task_success'] is False
    assert saved['evaluation_validity'] == 'unavailable' and saved['completion_score'] is None
    assert saved['native_result']['failure_stage'] == 'result_validation'


@pytest.mark.parametrize('field', ('snapshot_id', 'action_step'))
def test_observation_source_rejects_stale_identity_before_native_reporting(field, tmp_path):
    source, rig = source_and_rig()
    native, task, _ = attach_envelope(source, tmp_path)
    observation = source.begin_episode(task.task_text)
    fields = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    fields[field] += 1
    cycle, reads = rig.cycle, list(rig.reads)
    with pytest.raises(ValueError, match='stale action boundary'):
        source.report_arrivals([A], **fields)
    assert not native.reports and native.records == 1 and native.samples == 0
    assert source.agent_observation() is observation
    assert rig.cycle == cycle and rig.reads == reads


def test_observation_source_report_refuses_nonowner_thread(tmp_path):
    source, _ = source_and_rig()
    native, task, _ = attach_envelope(source, tmp_path)
    observation = source.begin_episode(task.task_text)
    errors = []

    def worker():
        try:
            source.report_arrivals([A], snapshot_id=observation.snapshot_id, action_step=observation.action_step)
        except RuntimeError as error:
            errors.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert len(errors) == 1 and not native.reports
    assert native.records == 1 and native.samples == 0


def owner_post(backend, url, body):
    """Wait for the real HTTP transaction, then service it on the owner thread."""
    responses, errors = [], []

    def request():
        try:
            responses.append(requests.post(url + '/api/agent/arrivals', json=body, timeout=5))
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=request)
    worker.start()
    with backend.frame_cond:
        assert backend.frame_cond.wait_for(lambda: backend.busy, timeout=2)
    assert backend.process_next()
    worker.join(timeout=5)
    assert not worker.is_alive() and not errors and len(responses) == 1
    return responses[0]


def valid_body(backend, token):
    observation = backend.observation_source.agent_observation()
    return {'rooms': [A], 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'control_token': token}


def test_http_arrivals_reject_invalid_lease_extra_fields_and_invalid_payloads(control_http, tmp_path):
    backend, _, url, calls, _ = control_http
    native, _, _ = attach_envelope(backend.observation_source, tmp_path)
    assert requests.post(url + '/api/agent/arrivals', json=valid_body(backend, 'no-active-lease'), timeout=3).status_code == 403
    token = claim(backend, url)
    body = valid_body(backend, token)
    invalid = [({key: value for key, value in body.items() if key != 'control_token'}, 400),
        (dict(body, control_token=None), 403), (dict(body, control_token=''), 403),
        (dict(body, control_token='wrong-lease'), 403),
        (dict(body, actual_room=PRIVATE_ROOM), 400), (dict(body, accepted=True), 400),
        (dict(body, rooms=[]), 400), (dict(body, rooms=A), 400),
        (dict(body, rooms=[f'ARRIVED: {A}']), 400), (dict(body, rooms=[A] * 17), 400),
        (dict(body, snapshot_id=True), 400), (dict(body, action_step=True), 400)]
    for payload, status in invalid:
        response = requests.post(url + '/api/agent/arrivals', json=payload, timeout=3)
        assert response.status_code == status and not response.json()['ok']
    assert not native.reports and native.records == 1 and not calls
    assert backend.commands.empty() and not backend.busy and backend.steps == 0


@pytest.mark.parametrize('field', ('snapshot_id', 'action_step'))
def test_http_stale_boundaries_are_rejected_without_native_calls(control_http, tmp_path, field):
    backend, _, url, calls, rig = control_http
    native, _, _ = attach_envelope(backend.observation_source, tmp_path)
    token = claim(backend, url)
    body = valid_body(backend, token)
    body[field] += 1
    before = (rig.cycle, list(rig.reads), backend.steps, backend.physics_steps)
    response = owner_post(backend, url, body)
    assert response.status_code == 409 and not response.json()['ok']
    assert not native.reports and native.records == 1 and not calls
    assert before == (rig.cycle, rig.reads, backend.steps, backend.physics_steps)
    assert PRIVATE_ROOM not in response.text and PRIVATE_REASON not in response.text


@pytest.mark.parametrize('accepts', (True, False))
def test_http_native_reporting_stays_on_owner_without_actions_physics_or_sensor_capture(
        control_http, tmp_path, monkeypatch, accepts):
    backend, _, url, calls, rig = control_http
    native, task, _ = attach_envelope(backend.observation_source, tmp_path, accepts=accepts, required=(A,))
    token = claim(backend, url)
    observation = backend.observation_source.agent_observation()
    before = (rig.cycle, list(rig.reads), backend.steps, backend.physics_steps,
              backend.primitive_actions, list(backend.history))

    def forbidden(*args, **kwargs):
        pytest.fail('Arrival reporting must not step physics or acquire sensors.')

    monkeypatch.setattr(rig.pipeline.og.sim, 'render', forbidden)
    monkeypatch.setattr(rig.pipeline.og.sim, 'step', forbidden, raising=False)
    monkeypatch.setattr(rig, 'observe', forbidden)
    body = dict(valid_body(backend, token), rooms=[A, A])
    response = owner_post(backend, url, body)
    assert response.status_code == 200
    public = response.json()
    assert public['ok'] is True and public['message'] == 'Arrival declarations recorded.'
    assert PRIVATE_ROOM not in response.text and PRIVATE_REASON not in response.text
    assert not {'accepted', 'actual_room', 'native_result', 'task_success', 'reports'} & set(public)
    assert [report['claimed_room'] for report in native.reports] == [A]
    assert native.reports[0]['accepted'] is accepts
    assert native.threads == [threading.get_ident()] and native.samples == 0
    assert before == (rig.cycle, rig.reads, backend.steps, backend.physics_steps,
                      backend.primitive_actions, backend.history)
    assert not calls and backend.observation_source.agent_observation() is observation
    sealed = json.loads((tmp_path / 'state.json').read_text())
    assert validate_episode_evaluation(sealed, observation, task.task_text, task) is accepts


def test_runner_http_receipt_and_trace_strip_private_native_results_and_redact_lease(
        control_http, tmp_path, monkeypatch):
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment

    backend, _, url, calls, _ = control_http
    native, _, _ = attach_envelope(backend.observation_source, tmp_path / 'private', required=(A,))
    token = claim(backend, url)
    environment = MCPRobotEnvironment(url, run_id='arrival-runner-fixture', task_text=LoadedTask.task_text)
    environment.control_token = token
    logged = []
    environment.attach_input_logger(logged.append)

    async def forbidden_tool(*args, **kwargs):
        pytest.fail('Arrival declarations use the runner-owned HTTP route, not a new MCP tool.')

    monkeypatch.setattr(environment, '_call_tool', forbidden_tool)
    completed = asyncio.Event()

    async def run():
        owner = asyncio.create_task(pump(backend, completed))
        try:
            return await environment.report_arrivals([A], backend.observation_source.agent_observation())
        finally:
            completed.set()
            await owner

    receipt = asyncio.run(run())
    assert receipt == {'ok': True, 'message': 'Arrival declarations recorded.'}
    assert len(logged) == 1 and logged[0]['tool'] == 'robot_report_arrivals'
    assert logged[0]['result'] == receipt
    assert logged[0]['arguments']['control_token'] == '[redacted]'
    assert PRIVATE_ROOM not in json.dumps(logged) and PRIVATE_REASON not in json.dumps(logged)
    assert not {'accepted', 'actual_room', 'native_result', 'task_success'} & set(receipt)
    assert len(native.reports) == 1 and not calls


def test_actual_mcp_runner_reports_before_execute_and_done_without_adding_model_tools(
        control_http, tmp_path, monkeypatch):
    from mcp import Client
    from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    from wia.runtime.web.mcp_server import create_server
    from harness_integration.test_task_edit2_initial_overview import install_overview

    backend, _, url, calls, rig = control_http
    overview_calls, _ = install_overview(rig)
    private = tmp_path / 'private'
    native, task, _ = attach_envelope(backend.observation_source, private)
    monkeypatch.setattr('wia.runtime.agent.mcp_environment.Client', lambda *args, **kwargs: Client(create_server(url)))
    completed = asyncio.Event()
    agent = MetadataAgent(reply('BASE_FWD', 'Current room.', rooms=[A]), reply('DONE', 'Current room.', rooms=[B]))

    def success():
        observation = backend.observation_source.agent_observation()
        sealed = json.loads((private / 'state.json').read_text())
        return validate_episode_evaluation(sealed, observation, task.task_text, task)

    async def run():
        owner = asyncio.create_task(pump(backend, completed))
        try:
            async with MCPRobotEnvironment(url, run_id='arrival-full-runner', task_text=task.task_text) as environment:
                tools = await environment.client.list_tools()
                assert {tool.name for tool in tools.tools} == {'robot_get_context', 'robot_get_state',
                    'robot_get_camera', 'robot_execute_actions', 'robot_get_observation', 'robot_execute_parameterized'}
                return await run_episode(environment, agent, tmp_path / 'episode', success_provider=success)
        finally:
            completed.set()
            await owner

    summary = asyncio.run(run())
    assert summary['termination_reason'] == 'model_done' and summary['task_success'] is True
    assert summary['initial_overview_input']['status'] == 'available' and len(overview_calls) == 1
    assert calls == ['BASE_FWD'] and backend.steps == 1 and backend.control_owner == 'human'
    assert [row['action_step'] for row in native.reports] == [0, 1]
    assert set(native.threads) == {threading.get_ident()} and native.samples == 0
    first_report = next(index for index, event in enumerate(native.events) if event[:2] == ('report', A))
    motion = next(index for index, event in enumerate(native.events) if event[:2] == ('record', 'BASE_FWD'))
    final_report = next(index for index, event in enumerate(native.events) if event[:2] == ('report', B))
    assert first_report < motion < final_report
    inputs = [json.loads(line) for line in (tmp_path / 'episode/model_inputs.jsonl').read_text().splitlines()]
    arrivals = [row for row in inputs if row.get('tool') == 'robot_report_arrivals']
    assert [row['arguments']['rooms'] for row in arrivals] == [[A], [B]]
    assert all(row['arguments']['control_token'] == '[redacted]' and row['passed_to_model'] is False for row in arrivals)
    assert all(set(row['result']) == {'ok', 'message'} for row in arrivals)
    for path in (tmp_path / 'episode').rglob('*.json*'):
        assert PRIVATE_ROOM not in path.read_text() and PRIVATE_REASON not in path.read_text()
    assert PRIVATE_ROOM in (private / 'state.json').read_text()

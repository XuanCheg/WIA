"""Verify queue-driven real-world retention with fake native processes.

Run: PYTHONPATH=src:../Show-Harness-Explore python -m pytest -q tests/harness_integration/test_capture_queue_retention.py
No simulator, GPU queries, signals, or live queue policies are used.
"""
from types import SimpleNamespace

import pytest

import wia.runtime.commands.run_esi_capture_worker as worker
from wia.runtime.scene_capture_queue import read_json, write_json


GPU_UUID = 'GPU-retention-fixture-6'
CURRENT_JOB = 'capture_fixture'
NEXT_JOB = 'capture_next'


class FakeQueue:
    """Queue evidence and accepted manifests without native rendering or claims."""

    def __init__(self, root):
        self.root, self.events = root, []
        for name in ('queued', 'claimed', 'workers'):
            (root / name).mkdir(parents=True)
        self.manifests = {CURRENT_JOB: {'code_root': str(root), 'code_commit': 'fixture',
            'capture_signature': 'fixture_signature', 'timeout_s': 1, 'max_attempts': 1,
            'candidate_id': 'current'}}
        self.states = {CURRENT_JOB: 'CLAIMED'}
        self.finished = False

    def job_dir(self, job_id):
        return self.root / 'jobs' / job_id

    def manifest(self, job_id):
        value = self.manifests[job_id]
        if value.get('invalid_accepted_manifest'):
            raise ValueError('Synthetic accepted manifest mismatch.')
        return value

    def status(self, job_id):
        return {'state': self.states.get(job_id), 'attempts': 1}

    def cached(self, manifest):
        return None

    def heartbeat(self, claim, **details):
        assert not self.finished, 'A retained capture must release its queue claim.'
        self.events.append(('heartbeat', details))

    def transition(self, claim, state, **details):
        self.states[claim['job_id']] = state
        self.events.append((state, details))

    def finish(self, claim, receipt):
        assert read_json(receipt)['complete']
        self.finished = True
        self.states[claim['job_id']] = 'REVIEW_PENDING'
        self.events.append(('finish', str(receipt)))

    def release(self, claim):
        self.events.append(('release', claim['job_id']))

    def claim(self, *args, **kwargs):
        raise AssertionError('Never claim another job while a real world is retained.')

    def enqueue(self, job_id=NEXT_JOB, *, state='QUEUED', **manifest):
        self.manifests[job_id] = {'code_commit': 'fixture', 'candidate_id': 'allowed_next', **manifest}
        self.states[job_id] = state
        write_json(self.root / 'queued' / f'{job_id}.json', {'job_id': job_id})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    queue = FakeQueue(tmp_path / 'queue')
    claim = {'job_id': CURRENT_JOB, 'worker': {'id': 'worker6', 'uuid': GPU_UUID}}
    gpu = {'uuid': GPU_UUID, 'execution_mode': 'local',
           'appdata_path': str(tmp_path / 'worker6'), 'retain_world': False}
    output = queue.job_dir(CURRENT_JOB) / 'attempts/01'
    process = SimpleNamespace(pid=987654321, running=False)
    process.poll = lambda: None if process.running else 0
    process.wait = lambda: 0
    state = SimpleNamespace(queue=queue, claim=claim, gpu=gpu, output=output,
        process=process, environments=[], sleeps=0, clock=0., stops=[],
        world_overrides={}, on_sleep=None)

    def launch(command, *, cwd, env, stdout, stderr, start_new_session):
        state.environments.append(env)
        process.running = env.get('ESI_CAPTURE_RETAIN_WORLD') == '1'
        stdout.write('Synthetic native capture.\n')
        write_json(output / 'preview_complete.json', {'complete': True, 'timings': {}})
        if process.running:
            write_json(output / 'retained_world.json', {'state': 'RETAINED',
                'pid': process.pid, 'job_id': CURRENT_JOB,
                'capture_signature': 'fixture_signature', **state.world_overrides})
        return process

    def sleep(seconds):
        state.sleeps += 1
        state.clock += 1000.  # Residency must survive far beyond capture timeout.
        assert state.sleeps <= 4, 'Unexpected retained-loop hang in fake lifecycle.'
        if state.on_sleep:
            state.on_sleep()
        else:
            process.running = False  # Simulate an external native exit, without signals.

    def stop_group(*args, **kwargs):
        state.stops.append(args)
        process.running = False

    monkeypatch.delenv('ESI_CAPTURE_RETAIN_WORLD', raising=False)
    monkeypatch.setattr(worker.subprocess, 'Popen', launch)
    monkeypatch.setattr(worker, 'verify_snapshot', lambda manifest: None)
    monkeypatch.setattr(worker, 'complete_images', lambda manifest, receipt: receipt.get('complete') is True)
    monkeypatch.setattr(worker, 'process_ticks', lambda pid: 'fixture_ticks' if process.running else None)
    monkeypatch.setattr(worker, 'group_members', lambda pid: [])
    monkeypatch.setattr(worker, 'stop_group', stop_group)
    monkeypatch.setattr(worker.time, 'sleep', sleep)
    monkeypatch.setattr(worker.time, 'monotonic', lambda: state.clock)
    return state


def enable_retention(harness, **overrides):
    write_json(harness.queue.root / 'capture_retention.json', {
        'state': 'ENABLED', 'mode': 'retain_last_world', 'gpu_uuids': [GPU_UUID], **overrides})


@pytest.mark.parametrize('policy,expected', [
    (None, False),
    ({}, True),
    ({'gpu_uuids': ['GPU-unrelated']}, False),
    ({'state': 'DISABLED'}, False),
    ({'mode': 'unsupported'}, False),
])
def test_policy_is_opt_in_uuid_fenced_and_applied_before_native_launch(harness, policy, expected):
    state = harness
    if policy is not None:
        enable_retention(state, **policy)
    worker.capture_job(state.queue, state.claim, state.gpu)
    assert (state.environments[0].get('ESI_CAPTURE_RETAIN_WORLD') == '1') is expected
    gpu = read_json(state.output / 'gpu.json')
    command = read_json(state.output / 'command.json')
    assert gpu == command['gpu'] and gpu['retain_world'] is expected
    assert bool(gpu.get('queue_retention')) is expected
    assert state.gpu['retain_world'] is False and 'queue_retention' not in state.gpu
    assert state.queue.finished and not state.stops
    assert not list((state.output / 'retained_requests').glob('*.json'))
    if expected:
        assert gpu['queue_retention']['retained_timeout_s'] is None
        assert gpu['queue_retention']['pre_completion_timeout_s'] is None
        assert gpu['queue_retention']['explicit_cancellation_unchanged']
        assert gpu['queue_retention']['explicit_worker_stop_and_failures_can_end_retention']


def test_drained_queue_keeps_native_world_without_close_or_stop_past_capture_timeout(harness):
    state = harness
    enable_retention(state)
    state.queue.enqueue('capture_running', state='RUNNING')
    state.queue.enqueue('capture_failed', state='FAILED')

    def observe():
        assert state.process.running and state.queue.finished and not state.stops
        assert not list((state.output / 'retained_requests').glob('*.json'))
        status = read_json(state.queue.root / 'workers/worker6.json')
        assert status['state'] == 'RETAINED' and status['native_pid'] == state.process.pid
        if state.sleeps == 3:
            state.process.running = False

    state.on_sleep = observe
    worker.capture_job(state.queue, state.claim, state.gpu)
    assert state.sleeps == 3 and state.clock == 3000. and not state.stops
    assert read_json(state.output / 'process_exit.json')['timeout'] is False


@pytest.mark.parametrize('enqueue_later', [False, True])
def test_eligible_work_gracefully_closes_exact_world_after_finish_without_claim(harness, enqueue_later):
    state = harness
    enable_retention(state)
    write_json(state.queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER',
        'authorized_gpu_pool': {'state': 'RUNNING', 'gpu_uuids': [GPU_UUID]}})
    if not enqueue_later:
        state.queue.enqueue()

    def native_consumes_request():
        requests = list((state.output / 'retained_requests').glob('*.json'))
        assert state.queue.finished and state.process.running and not state.stops
        if enqueue_later and state.sleeps == 1:
            assert not requests
            state.queue.enqueue()
            return
        assert len(requests) == 1
        request = read_json(requests[0])
        assert request['operation'] == 'close' and request['job_id'] == CURRENT_JOB
        assert request['capture_signature'] == 'fixture_signature'
        assert request['native_pid'] == state.process.pid and request['pending_job_id'] == NEXT_JOB
        assert request['reason'] == 'eligible_queued_work'
        assert state.queue.status(NEXT_JOB)['state'] == 'QUEUED'
        state.process.running = False

    state.on_sleep = native_consumes_request
    worker.capture_job(state.queue, state.claim, state.gpu)
    assert state.sleeps == (2 if enqueue_later else 1)
    assert not state.stops and state.queue.status(NEXT_JOB)['state'] == 'QUEUED'
    assert read_json(state.output / 'process_exit.json')['timeout'] is False


@pytest.mark.parametrize('fence', ['other_gpu_pool', 'paused', 'runtime', 'candidate',
                                  'sample', 'claimed', 'invalid_manifest', 'job_id', 'pilot'])
def test_blocked_or_unrelated_work_does_not_close_retained_world(harness, fence):
    state = harness
    enable_retention(state)
    state.queue.enqueue()
    selection = {}
    if fence == 'other_gpu_pool':
        write_json(state.queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER',
            'authorized_gpu_pool': {'state': 'RUNNING', 'gpu_uuids': ['GPU-unrelated']}})
    elif fence == 'paused':
        write_json(state.queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER'})
    elif fence == 'runtime':
        write_json(state.queue.root / 'capture_runtime_policy.json', {'allowed_code_commits': ['other']})
    elif fence == 'candidate':
        selection['candidate_prefix'] = 'unrelated_'
    elif fence == 'sample':
        write_json(state.queue.root / 'capture_sample_gate.json', {
            'state': 'ONE_PER_CATEGORY_AWAITING_REVIEW', 'selected_candidate_ids': ['other']})
    elif fence == 'claimed':
        write_json(state.queue.root / 'claimed' / f'{NEXT_JOB}.json', {'token': 'another_owner'})
    elif fence == 'invalid_manifest':
        state.queue.manifests[NEXT_JOB]['invalid_accepted_manifest'] = True
    elif fence == 'job_id':
        selection['job_id'] = CURRENT_JOB
    else:
        selection['pilot_control'] = state.queue.root / 'pilot.json'
        write_json(selection['pilot_control'], {'mode': 'pilot', 'worker_id': 'other_worker',
            'gpu_uuid': GPU_UUID, 'job_ids': [NEXT_JOB]})

    def observe():
        assert state.queue.finished and not state.stops
        assert not list((state.output / 'retained_requests').glob('*.json'))
        state.process.running = False

    state.on_sleep = observe
    worker.capture_job(state.queue, state.claim, state.gpu, **selection)
    assert not state.stops


def test_authorized_paused_pilot_requires_exact_selected_job(harness):
    state = harness
    state.queue.enqueue()
    write_json(state.queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER',
        'authorized_pilot': {'state': 'RUNNING', 'worker_id': 'worker6',
            'gpu_uuid': GPU_UUID, 'job_ids': [NEXT_JOB]}})
    assert worker.eligible_queued_job(state.queue, state.claim['worker']) is None
    assert worker.eligible_queued_job(state.queue, state.claim['worker'], job_id=NEXT_JOB) == NEXT_JOB
    assert worker.eligible_queued_job(state.queue, state.claim['worker'], job_id=CURRENT_JOB) is None


@pytest.mark.parametrize('with_policy', [False, True])
def test_explicit_retain_world_keeps_existing_indefinite_behavior(harness, with_policy):
    state = harness
    state.gpu['retain_world'] = True
    if with_policy:
        enable_retention(state)
    state.queue.enqueue()

    def observe():
        assert state.queue.finished and not state.stops
        assert not list((state.output / 'retained_requests').glob('*.json'))
        state.process.running = False

    state.on_sleep = observe
    worker.capture_job(state.queue, state.claim, state.gpu)
    assert not state.stops and 'queue_retention' not in read_json(state.output / 'gpu.json')


@pytest.mark.parametrize('world_overrides', [
    {'job_id': 'capture_foreign'}, {'capture_signature': 'foreign_signature'}, {'pid': 123},
])
def test_foreign_world_never_finishes_and_explicit_cancellation_still_stops(harness, world_overrides):
    state = harness
    enable_retention(state)
    state.world_overrides = world_overrides
    state.queue.enqueue()
    def cancel():
        write_json(state.queue.job_dir(CURRENT_JOB)/'cancel.json',{})
    state.on_sleep = cancel
    worker.capture_job(state.queue, state.claim, state.gpu)
    assert not state.queue.finished and len(state.stops) == 1
    assert state.queue.status(CURRENT_JOB)['state'] == 'CANCELLED'
    assert read_json(state.output / 'process_exit.json')['timeout'] is False
    assert not list((state.output / 'retained_requests').glob('*.json'))

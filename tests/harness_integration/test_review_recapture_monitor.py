"""Validate monitor restart fences without launching or stopping real processes.

Run: python -m pytest -q tests/test_review_recapture_monitor.py
All process inspection, GPU inspection, tmux launch, and stop calls are mocked.
"""
from types import SimpleNamespace

import pytest

from wia.runtime.scene_capture_queue import read_json, write_json
import wia.runtime.commands.monitor_esi_review_recapture as monitor


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    (scripts / 'start_worker0.sh').write_text('# CPU fixture only\n')
    instance = monitor.RecoveryMonitor(tmp_path / 'queue', tmp_path / 'campaign', scripts, tmp_path / 'logs')
    owner = {'id': 'worker0', 'supervisor_pid': 1234, 'supervisor_start_ticks': '10',
             'state': 'RUNNING', 'uuid': 'GPU-fixture', 'native_pid': 5678,
             'appdata_path': str(tmp_path / 'appdata')}
    write_json(instance.queue.root / 'workers/worker0.json', owner)
    write_json(tmp_path / 'appdata/native_tmp_owner.json', {'pid': 5678, 'start_ticks': '20', 'pgid': 5678})
    actions = {'gpu': [], 'launch': [], 'stop': [], 'poll': []}
    states = {1234: 'exited', 5678: 'exited'}
    monkeypatch.setattr(monitor, 'process_state', lambda pid, ticks: states.get(pid, 'unknown'))
    monkeypatch.setattr(monitor, 'group_members', lambda pgid: [])
    monkeypatch.setattr(monitor, 'stop_group', lambda pid: actions['stop'].append(pid))

    def gpu(uuid):
        actions['gpu'].append(uuid)
        return {'verified': True, 'uuid': uuid, 'contexts': []}

    def launch(script, log, worker_id):
        actions['launch'].append((script, log, worker_id))
        return {'pane_id': '%fixture', 'session': 'codex-gpu'}

    monkeypatch.setattr(monitor, 'gpu_contexts', gpu)
    monkeypatch.setattr(monitor, 'launch_saved_worker', launch)
    monkeypatch.setattr(instance.watcher, 'poll', lambda value: actions['poll'].append(value))
    return instance, owner, actions, states


def test_exited_worker_requires_two_gpu_checks_and_restart_is_not_duplicated(fixture):
    instance, _, actions, _ = fixture
    assert instance.tick() == 'monitoring'
    assert actions['gpu'] == ['GPU-fixture', 'GPU-fixture']
    assert len(actions['launch']) == 1 and not actions['stop']
    assert read_json(instance.journal)['pending_restarts']['worker0']['state'] == 'launched'
    instance.tick()
    assert len(actions['launch']) == 1
    # Restarting the monitor itself preserves the same pending launch fence.
    recovered = monitor.RecoveryMonitor(instance.queue.root, instance.output, instance.scripts, instance.logs)
    recovered.tick()
    assert len(actions['launch']) == 1


@pytest.mark.parametrize('native_state', ['alive', 'unknown'])
def test_alive_or_unverified_native_never_restarts(fixture, native_state):
    instance, _, actions, states = fixture
    states[5678] = native_state
    instance.tick()
    assert not actions['gpu'] and not actions['launch'] and not actions['stop']


def test_healthy_supervisor_only_polls_crash_watcher(fixture):
    instance, _, actions, states = fixture
    states[1234] = states[5678] = 'alive'
    instance.tick()
    assert len(actions['poll']) == 1
    assert not actions['gpu'] and not actions['launch'] and not actions['stop']


def test_paused_finished_and_missing_worker_record_never_restart(fixture):
    instance, _, actions, _ = fixture
    write_json(instance.queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER'})
    assert instance.tick() == 'paused'
    write_json(instance.output / 'progress.json', {'stage': 'finished'})
    assert instance.tick() == 'finished'
    write_json(instance.output / 'progress.json', {'stage': 'bulk'})
    write_json(instance.queue.root / 'capture_pause.json', {})
    (instance.queue.root / 'workers/worker0.json').unlink()
    instance.tick()
    assert not any(actions.values())


@pytest.mark.parametrize('final', [False, True])
def test_gpu_error_or_racing_context_blocks_restart(fixture, monkeypatch, final):
    instance, _, actions, _ = fixture
    calls = []

    def gpu(uuid):
        calls.append(uuid)
        if final and len(calls) == 1:
            return {'verified': True, 'contexts': []}
        return ({'verified': True, 'contexts': [{'pid': 'other'}]} if final
                else {'verified': False, 'error': 'nvidia-smi unavailable'})

    monkeypatch.setattr(monitor, 'gpu_contexts', gpu)
    instance.tick()
    assert len(calls) == (2 if final else 1)
    assert not actions['launch'] and not actions['stop']
    assert not instance.pending


def test_stop_callback_rechecks_live_supervisor_identity(fixture):
    instance, owner, actions, states = fixture
    instance.poll_owner = owner
    states[1234] = 'alive'
    instance.stop_confirmed_native(5678)
    assert actions['stop'] == [5678]
    write_json(instance.queue.root / 'workers/worker0.json', {**owner, 'supervisor_start_ticks': 'different'})
    instance.stop_confirmed_native(5678)
    assert actions['stop'] == [5678]


def test_gpu_xml_includes_graphics_contexts_and_errors_fail_closed(monkeypatch):
    result = SimpleNamespace(stdout='<nvidia_smi_log><gpu><uuid>GPU-fixture</uuid><processes>'
                             '<process_info><pid>42</pid><type>G</type><process_name>graphics</process_name>'
                             '</process_info></processes></gpu></nvidia_smi_log>')
    monkeypatch.setattr(monitor.subprocess, 'run', lambda *args, **kwargs: result)
    gpu = monitor.gpu_contexts('GPU-fixture')
    assert gpu['verified'] and gpu['contexts'][0]['type'] == 'G'
    result.stdout = '<nvidia_smi_log><gpu><uuid>GPU-fixture</uuid><processes>N/A</processes></gpu></nvidia_smi_log>'
    assert not monitor.gpu_contexts('GPU-fixture')['verified']


def test_tmux_command_quotes_paths_and_retains_shell(monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='%fixture\n')

    monkeypatch.setattr(monitor.subprocess, 'run', run)
    result = monitor.launch_saved_worker(tmp_path / 'saved script.sh', tmp_path / 'worker log.txt', 'worker0')
    assert result['pane_id'] == '%fixture'
    assert commands[0][:5] == ['tmux', 'new-window', '-d', '-t', 'codex-gpu']
    assert commands[0][-1].endswith('2>&1; exec bash')
    assert "'" in commands[0][-1]

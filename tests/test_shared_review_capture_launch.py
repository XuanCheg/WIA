"""Check explicit shared-GPU review launch checks without starting processes.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
  tests/test_shared_review_capture_launch.py
Only temporary launch artifacts and injected resource/process measurements are used.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load_module('launch_wia_review_captures', ROOT / 'scripts/launch_wia_review_captures.py')


def busy_device(*, used=8300, total=24564):
    return {'physical_index': 7, 'uuid': 'GPU-shared-test', 'pci_bus_id': '0000:81:00.0',
        'utilization_percent': 63, 'memory_used_mib': used, 'memory_total_mib': total,
        'compute_clients': [{'gpu_uuid': 'GPU-shared-test', 'pid': 67890,
                             'process_name': 'other-user-training', 'used_memory_mib': str(used)}]}


@pytest.mark.parametrize('shared,used,owner,accepted', [(False, 8300, False, False),
    (True, 8300, False, True), (True, 11000, False, False), (True, 8300, True, False)])
def test_busy_default_and_explicit_shared_headroom_keep_capture_ownership_exclusive(monkeypatch,
        shared, used, owner, accepted):
    owners = [{'worker_id': 'existing-capture', 'pid': 100}] if owner else []
    monkeypatch.setattr(launch, '_live_supervisors', lambda uuid: owners)
    errors, measured_owners = launch.device_errors(busy_device(used=used), allow_shared_gpu=shared)
    assert (not errors) is accepted and measured_owners == owners
    if not shared:
        assert errors == ['GPU has existing compute clients.', 'GPU utilization is nonzero.',
                          'GPU memory exceeds the 512 MiB driver-only idle threshold.']
    if owner:
        assert 'A live capture supervisor already owns this GPU.' in errors


def test_shared_threshold_boundary_and_worker_flags_record_measured_coexistence(monkeypatch):
    monkeypatch.setattr(launch, '_live_supervisors', lambda uuid: [])
    device = busy_device(used=24564 - 14336)
    assert launch.device_errors(device, allow_shared_gpu=True)[0] == []
    plan = launch.worker_plan(device, 'controlled-plan', allow_shared_gpu=True)
    assert '--allow-shared-gpu' in plan['command'] and '--once' in plan['command']
    assert plan['command'][plan['command'].index('--shared-gpu-min-free-mib') + 1] == '14336'
    assert plan['environment']['CUDA_VISIBLE_DEVICES'] == device['uuid']
    authorization = plan['gpu_authorization']
    assert authorization['mode'] == 'shared_with_headroom'
    assert authorization['measured_free_memory_mib'] == authorization['shared_gpu_min_free_mib'] == 14336
    assert authorization['coexisting_compute_clients'] == device['compute_clients']
    assert authorization['coexisting_compute_clients'] is not device['compute_clients']
    exclusive = launch.worker_plan(device, 'exclusive-plan')
    assert '--allow-shared-gpu' not in exclusive['command']
    assert '--shared-gpu-min-free-mib' not in exclusive['command']
    assert exclusive['gpu_authorization']['mode'] == 'exclusive_idle'
    device['memory_used_mib'] += 1
    assert launch.device_errors(device, allow_shared_gpu=True)[0]


@pytest.fixture
def planned(tmp_path, monkeypatch):
    campaign, queue, conda = tmp_path / 'campaign', tmp_path / 'queue', tmp_path / 'conda'
    python = conda / 'envs/behavior/bin/python'
    libs = tmp_path / 'runtime_libs'
    for path in (python, conda / 'etc/profile.d/conda.sh', tmp_path / 'configs/esi_vulkan_egl.json',
                 queue / 'category_storage.json', queue / 'capture_runtime_policy.json'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}', encoding='utf-8')
    python.chmod(0o700)
    campaign.mkdir()
    libs.mkdir()
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(launch, 'CAMPAIGN', campaign)
    monkeypatch.setattr(launch, 'QUEUE', queue)
    monkeypatch.setattr(launch, 'CONDA_ROOT', conda)
    monkeypatch.setattr(launch, 'BEHAVIOR_PYTHON', python)
    monkeypatch.setattr(launch, 'RUNTIME_LIBS', libs)
    monkeypatch.setattr(launch.shutil, 'which', lambda value: '/fake/tmux')
    monkeypatch.setattr(launch, '_live_supervisors', lambda uuid: [])
    device = busy_device()
    monkeypatch.setattr(launch, 'gpu_snapshot', lambda indices: {index: deepcopy(device) for index in indices})
    budget = {'cache_root': str(tmp_path / 'cache'), 'errors': []}
    monkeypatch.setattr(launch, 'resources', lambda count: deepcopy(budget))
    plan = launch.build_plan([7], allow_shared_gpu=True, shared_gpu_min_free_mib=14336)
    assert plan['ready'] is True and not (campaign / 'final_refresh_launches').exists()
    return NS(plan=plan, device=device, budget=budget, campaign=campaign)


@pytest.mark.parametrize('failure', ['headroom', 'capture_owner', 'uuid', 'pci', 'resources'])
def test_apply_immediately_rechecks_headroom_identity_resources_and_live_capture_owner(planned, monkeypatch, failure):
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        assert 'new-window' not in args, 'A failed recheck must never start a worker.'
        return NS(returncode=0, stdout='')

    monkeypatch.setattr(launch, '_run', command)
    if failure == 'headroom':
        planned.device['memory_used_mib'] = 11000
    elif failure == 'capture_owner':
        monkeypatch.setattr(launch, '_live_supervisors', lambda uuid: [{'pid': 100}])
    elif failure == 'uuid':
        planned.device['uuid'] = 'GPU-different'
    elif failure == 'pci':
        planned.device['pci_bus_id'] = '0000:82:00.0'
    else:
        planned.budget['errors'] = ['Insufficient current RAM.']
    with pytest.raises(RuntimeError, match='already started workers were retained'):
        launch.apply_plan(planned.plan)
    worker, = planned.plan['workers']
    assert worker['launch_state'] == 'refused_on_immediate_recheck'
    assert worker['immediate_gpu_authorization']['mode'] == 'shared_with_headroom'
    assert worker['immediate_gpu_authorization']['shared_gpu_min_free_mib'] == 14336
    assert planned.plan['workers_started'] is False
    assert all('kill' not in arg for call in calls for arg in call)


def test_apply_forwards_reviewed_shared_mode_and_records_fresh_headroom(planned, monkeypatch):
    commands = []

    def command(args, **kwargs):
        commands.append(args)
        if 'new-window' in args:
            Path(planned.plan['workers'][0]['worker_pid_file']).write_text('12345\n', encoding='utf-8')
            return NS(returncode=0, stdout='@1\t%1\t12345\n')
        return NS(returncode=0, stdout='')

    monkeypatch.setattr(launch, '_run', command)
    monkeypatch.setattr(launch, '_process_ticks', lambda pid: 'known-start-ticks')
    result = launch.apply_plan(planned.plan, allow_shared_gpu=True, shared_gpu_min_free_mib=14336)
    assert result['state'] == 'launched' and result['workers_started'] is True
    worker, = result['workers']
    assert worker['immediate_gpu_authorization']['measured_free_memory_mib'] == 16264
    manifest = json.loads(Path(result['manifest_file']).read_text())
    assert manifest['gpu_authorization_mode'] == 'shared_with_headroom'
    assert manifest['workers'][0]['gpu_authorization']['coexisting_compute_clients'][0]['pid'] == 67890
    assert sum('new-window' in command for command in commands) == 1


def test_apply_cannot_change_reviewed_authorization_or_threshold(planned):
    with pytest.raises(ValueError, match='sharing authorization'):
        launch.apply_plan(planned.plan, allow_shared_gpu=False)
    with pytest.raises(ValueError, match='memory threshold'):
        launch.apply_plan(planned.plan, shared_gpu_min_free_mib=8192)
    assert not (planned.campaign / 'final_refresh_launches').exists()


@pytest.mark.parametrize('script', ['launcher', 'watcher'])
@pytest.mark.parametrize('value', ['0', '-1'])
def test_cli_rejects_nonpositive_shared_memory_before_any_launch_or_output(tmp_path, monkeypatch, script, value):
    monkeypatch.setitem(sys.modules, 'launch_wia_review_captures', launch)
    watcher = load_module('shared_review_capture_watcher_cli', ROOT / 'scripts/watch_wia_review_captures.py')
    module = launch if script == 'launcher' else watcher
    args = ['--gpus', '7'] if script == 'launcher' else [
        '--refresh-manifest', str(tmp_path / 'missing.json'), '--output', str(tmp_path / 'watch')]
    with pytest.raises(SystemExit) as error:
        module.main(args + ['--shared-gpu-min-free-mib', value])
    assert error.value.code == 2 and not (tmp_path / 'watch').exists()


@pytest.mark.parametrize('shared,expected_launches', [(False, 0), (True, 1)])
def test_watcher_forwards_explicit_sharing_and_rechecks_headroom_each_iteration(tmp_path, monkeypatch,
        shared, expected_launches):
    monkeypatch.setitem(sys.modules, 'launch_wia_review_captures', launch)
    watcher = load_module('shared_review_capture_watcher_test', ROOT / 'scripts/watch_wia_review_captures.py')
    refresh = tmp_path / 'refresh.json'
    refresh.write_text(json.dumps({'queue_root': str(tmp_path / 'queue'),
                                  'new_jobs': {'GF2-1': {'job_id': 'exact_job'}}}), encoding='utf-8')
    iterations, snapshots, plans, applies = [], [], [], []

    class Queue:
        def __init__(self, root):
            self.root = Path(root)

        def status(self, job):
            return {'state': 'CACHED' if len(iterations) >= 2 else 'QUEUED', 'attempts': 0}

        def manifest(self, job):
            return {'max_attempts': 2}

    def snapshot(indices):
        snapshots.append(indices)
        return {7: busy_device(used=8300 if not iterations else 11000)}

    def build(indices, **kwargs):
        plans.append((indices, kwargs))
        return {'ready': True}

    def apply(plan, **kwargs):
        applies.append(kwargs)
        return {'manifest_file': 'fake-launch.json', 'workers': [{
            'gpu': {'physical_index': 7}, 'worker_id': 'one-capture',
            'queue_worker_status_file': str(tmp_path / 'absent-worker-status.json')}]}

    monkeypatch.setattr(watcher, 'CaptureQueue', Queue)
    monkeypatch.setattr(watcher.time, 'sleep', lambda seconds: iterations.append(seconds))
    monkeypatch.setattr(launch, 'gpu_snapshot', snapshot)
    monkeypatch.setattr(launch, '_live_supervisors', lambda uuid: [])
    monkeypatch.setattr(launch, '_process_ticks', lambda pid: None)
    monkeypatch.setattr(launch, 'build_plan', build)
    monkeypatch.setattr(launch, 'apply_plan', apply)
    args = ['--refresh-manifest', str(refresh), '--output', str(tmp_path / 'watch'), '--gpus', '7']
    if shared:
        args.append('--allow-shared-gpu')
    watcher.main(args)
    assert len(snapshots) == 2 and len(plans) == len(applies) == expected_launches
    if shared:
        assert plans == [([7], {'allow_shared_gpu': True, 'shared_gpu_min_free_mib': 14336})]
        assert applies == [{'allow_shared_gpu': True, 'shared_gpu_min_free_mib': 14336}]
    status = json.loads((tmp_path / 'watch/status.json').read_text())
    assert status['state'] == 'complete' and status['allow_shared_gpu'] is shared

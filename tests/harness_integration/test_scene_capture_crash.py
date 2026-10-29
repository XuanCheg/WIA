"""Check owned-process crash recovery using metadata and CPU fixtures.

Run: python -m pytest -q tests/test_scene_capture_crash.py
No GPU process is created or terminated; no image bytes are opened or validated.
"""
from pathlib import Path

import pytest

import wia.runtime.scene_capture_crash as crash
from wia.runtime.scene_capture_queue import read_json, write_json


@pytest.fixture
def native(tmp_path, monkeypatch):
    root = tmp_path / 'queue'
    output = root / 'jobs/capture_aa/attempts/01'
    worker = {'id': 'worker0', 'state': 'RUNNING', 'job_id': 'capture_aa',
              'native_pid': 12345, 'uuid': 'GPU-fixture'}
    write_json(root / 'workers/worker0.json', worker)
    write_json(root / 'jobs/capture_aa/status.json', {'state': 'RUNNING', 'attempts': 1,
        'native_output': str(output), 'partial_images': {'head': str(output / 'head.png')}})
    write_json(root / 'claimed/capture_aa.json', {'worker': worker, 'child_pid': 12345,
        'child_start_ticks': '678', 'job_id': 'capture_aa', 'token': 'current'})
    write_json(output / 'native_process.json', {'pid': 12345, 'pgid': 12345,
        'start_ticks': '678', 'capture_job_id': 'capture_aa'})
    write_json(output / 'gpu.json', {'uuid': 'GPU-fixture'})
    write_json(root / 'jobs/capture_aa/input/manifest.json', {'candidate_id': 'category_aa'})
    write_json(root / 'category_records/category_aa.json', {'diagnostic_logs': ['original.log']})
    write_json(output / 'partial_capture.json', {'images': {'global_diagonal': str(output / 'global.png')}})
    (output / 'native.log').write_text('[Error] [carb.graphics-vulkan.plugin] GPU crash is detected.\n')
    live = [True]
    monkeypatch.setattr(crash, 'process_matches', lambda pid, ticks: live[0] and pid == 12345 and ticks == '678')
    stopped = []

    def stop(pid):
        stopped.append(pid)
        live[0] = False

    return root, output, worker, stopped, crash.NativeCrashWatcher(root, stop)


@pytest.mark.parametrize('line', [
    '[Warning] GPU memory use is 700 MiB.',
    '[Warning] Low GPU utilization.',
    '[Error] Could not find suitable EXR channels.',
    'Healthy frames rendered; high CPU loading cost.',
])
def test_ordinary_warnings_do_not_imply_a_crash(line):
    assert crash.fatal_native_error(line) is None


@pytest.mark.parametrize('line', [
    '[Error] GPU crash is detected.',
    'Vulkan failure: VK_ERROR_DEVICE_LOST',
    '[Error] PhysX error: GPU artiPropagateVelocity fail to launch kernel!! 700',
    'CUDA error: an illegal memory access was encountered',
    '[Fatal] [rtx.scenedb.plugin] Failure to complete CopyCommandList',
])
def test_confirmed_native_faults_are_recognized(line):
    assert crash.fatal_native_error(line) is not None


def test_crash_is_recorded_before_stop_and_claim_stays_owned(native, monkeypatch):
    root, output, worker, stopped, watcher = native
    monkeypatch.setattr(Path, 'glob', lambda *args, **kwargs: pytest.fail('Image directory scans are forbidden.'))
    original_stop = watcher.stop_owned_group

    def stop(pid):
        assert read_json(output / 'automatic_crash_recovery.json')['action'] == 'crash_recorded'
        assert read_json(output / 'failure.json')['retryable'] is False
        original_stop(pid)

    watcher.stop_owned_group = stop
    result = watcher.poll(worker)
    assert stopped == [12345] and result['action'] == 'faulty_native_stopped'
    assert set(result['partial_images']) == {'head', 'global_diagonal'}
    assert read_json(root / 'jobs/capture_aa/status.json')['state'] == 'RUNNING'
    assert (root / 'claimed/capture_aa.json').is_file()
    assert 'original.log' in read_json(root / 'category_records/category_aa.json')['diagnostic_logs']
    assert watcher.poll(worker) is None


@pytest.mark.parametrize('invalid', ['different_pid', 'different_gpu', 'different_job', 'completed'])
def test_mismatched_or_completed_job_never_stops_a_process(native, invalid):
    root, output, worker, stopped, watcher = native
    if invalid == 'different_pid':
        p = root / 'claimed/capture_aa.json'; value = read_json(p); value['child_pid'] = 999; write_json(p, value)
    elif invalid == 'different_gpu':
        write_json(output / 'gpu.json', {'uuid': 'GPU-other'})
    elif invalid == 'different_job':
        write_json(root / 'workers/worker0.json', {**worker, 'job_id': 'capture_bb'})
    else:
        p = root / 'jobs/capture_aa/status.json'; value = read_json(p); value['state'] = 'REVIEW_PENDING'; write_json(p, value)
    assert watcher.poll(worker) is None and not stopped


def test_owner_change_after_recording_prevents_stop(native, monkeypatch):
    root, output, worker, stopped, watcher = native
    original = crash.write_json

    def change_owner(path, value):
        original(path, value)
        if Path(path).name == 'automatic_crash_recovery.json':
            original(root / 'workers/worker0.json', {**worker, 'job_id': 'capture_bb'})

    monkeypatch.setattr(crash, 'write_json', change_owner)
    assert watcher.poll(worker)['action'] == 'owner_changed_before_stop'
    assert not stopped and not (output / 'failure.json').exists()


def test_existing_native_failure_is_preserved_verbatim_without_retry(native):
    _, output, worker, _, watcher = native
    raw = '{"error":"Original native error", "retryable":true}\n'
    (output / 'failure.json').write_text(raw)
    watcher.poll(worker)
    assert (output / 'native_failure_before_watchdog.json').read_text() == raw
    assert read_json(output / 'failure.json')['error'] == 'Original native error'
    assert read_json(output / 'failure.json')['retryable'] is False


def test_log_lines_split_between_polls_are_recognized(native):
    _, output, worker, stopped, watcher = native
    (output / 'native.log').write_text('[Error] GPU crash is det')
    assert watcher.poll(worker) is None
    with (output / 'native.log').open('a') as log:
        log.write('ected.\n')
    assert watcher.poll(worker)['action'] == 'faulty_native_stopped'
    assert stopped == [12345]


def test_pending_exit_is_revisited_without_requiring_more_log_output(native, monkeypatch):
    _, _, worker, stopped, watcher = native
    watcher.stop_owned_group = lambda pid: stopped.append(pid)
    monkeypatch.setattr(crash, 'process_matches', lambda pid, ticks: len(stopped) < 2)
    assert watcher.poll(worker)['action'] == 'native_exit_pending'
    assert watcher.poll(worker)['action'] == 'faulty_native_stopped'
    assert stopped == [12345, 12345]


def test_only_declared_worker_crash_artifacts_are_preserved(tmp_path, monkeypatch):
    appdata, output = tmp_path / 'worker_cache', tmp_path / 'attempt'
    appdata.mkdir(); output.mkdir()
    dump = appdata / 'kit_20261015.nv-gpudmp'; dump.write_bytes(b'Native GPU dump fixture.')
    unrelated = tmp_path / 'another_worker.nvdbg'; unrelated.write_bytes(b'Other worker.')
    monkeypatch.setattr(Path, 'glob', lambda *args, **kwargs: pytest.fail('Directory scans are forbidden.'))
    artifacts = crash.preserve_crash_artifacts(f'GPU crash dump: {dump}\nOther log: {unrelated}', output, appdata)
    assert len(artifacts) == 1 and artifacts[0]['source'] == str(dump)
    assert Path(artifacts[0]['path']).read_bytes() == dump.read_bytes()

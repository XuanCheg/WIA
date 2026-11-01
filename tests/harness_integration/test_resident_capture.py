"""Check native lifecycle fencing without starting CUDA or modifying a scene.

Run: python -m pytest -q tests/test_resident_capture.py
Fixtures cover stage isolation, failed-world disposal and queue priority.
"""
from types import SimpleNamespace

import pytest
import yaml

from wia.runtime.scene_capture_queue import CaptureQueue, PROTOCOL, read_json, sha256, write_json
from wia.runtime.scene_capture_resident import ResidentCapture, recover_rendered_attempt, runtime_identity
from wia.runtime.commands.run_esi_capture_service import native_stage_cleanup, install_lifecycle, load_asset_runtime


def lifecycle_fixture(cleanup_clears=False):
    app = object()
    og = SimpleNamespace(app=app, sim=SimpleNamespace(scenes=['previous world']))
    clears = []

    def clear():
        clears.append(og.sim)
        og.sim = SimpleNamespace(scenes=[])

    og.clear = clear

    def cleanup(*args):
        if cleanup_clears:
            clear()
        return False

    pipeline = SimpleNamespace(og=og, task_cleanup_runtime=cleanup)
    adapters = SimpleNamespace(install_gpu_selection=lambda gpu: None, isolate_texture_cache=lambda *args: None)
    session_type = type('SessionFixture', (), {})
    state = {'reusable': True}
    install_lifecycle(adapters, session_type, state)
    session = session_type()
    session.closed, session.pipeline = False, pipeline
    session.task_module, session.env, session.payload, session.task_state = object(), object(), {}, {}
    return session, og, app, clears, state


@pytest.mark.parametrize('cleanup_clears', [False, True])
def test_close_rebuilds_empty_stage_once_and_preserves_application(cleanup_clears):
    session, og, app, clears, state = lifecycle_fixture(cleanup_clears)
    session.close()
    assert len(clears) == 1 and og.sim.scenes == [] and og.app is app
    assert state['application_identity'] == id(app) and state['reusable']
    session.close()
    assert len(clears) == 1


def test_failed_native_world_is_never_reused():
    session, _, _, clears, state = lifecycle_fixture()
    try:
        raise ValueError('Synthetic native load failure')
    except ValueError:
        session.close()
    assert not state['reusable'] and not clears


def test_renderer_monitor_ignores_old_jobs_and_catches_split_crash_messages(tmp_path):
    from wia.runtime.scene_capture_resident import RendererLogMonitor
    path = tmp_path / 'service.log'
    path.write_text('Old job: GPU crash is detected\n')
    monitor = RendererLogMonitor(path)
    assert monitor.failure() is None
    with path.open('a') as stream:
        stream.write('New scene: GPU crash is det')
    assert monitor.failure() is None
    with path.open('a') as stream:
        stream.write('ected. Native renderer stopped.\n')
    assert 'GPU crash is detected' in monitor.failure()
    path.write_text('Fresh log: VK_ERROR_DEVICE_LOST\n')
    assert 'VK_ERROR_DEVICE_LOST' in monitor.failure()


def test_close_rejects_application_replacement():
    session, og, _, _, _ = lifecycle_fixture()

    def clear():
        og.app = object()
        og.sim = SimpleNamespace(scenes=[])

    og.clear = clear
    with pytest.raises(RuntimeError, match='preserve'):
        session.close()


@pytest.mark.parametrize('version', ['3.7.1', '3.9.0'])
def test_native_cleanup_keeps_object_sensor_cleanup_and_cancels_old_callbacks_only_at_clear(version):
    events = []

    class Scene:
        def clear(self):
            events.append('per-object-removal')

    class Simulator:
        def __init__(self):
            self.scenes = [scene]
            self.callbacks = {'existing': lambda: events.append('existing-stop-callback')}
            for name in ('_pre_physics_step_callback', '_post_physics_step_callback', '_simulation_event_callback'):
                setattr(self, name, SimpleNamespace(unsubscribe=lambda key=name: events.append(key)))

        def add_callback_on_stop(self, name, callback):
            self.callbacks[name] = callback

        def remove_callback_on_stop(self, name):
            self.callbacks.pop(name)

        def stop(self):
            events.append('physics-stopped')
            for callback in self.callbacks.values():
                callback()

        def _partial_clear(self):
            self.stop()
            for item in self.scenes:
                item.clear()
            events.append('sensor-removal')
            events.append('global-cache-cleanup')

    scene = Scene()
    sim = Simulator()
    og = SimpleNamespace(sim=sim, __version__=version, clear=lambda: None)
    with native_stage_cleanup(og):
        # Source cleanup may stop before clearing. This must not unsubscribe a
        # still-live simulator, or replace the native scene.clear method.
        sim.stop()
        assert events == ['physics-stopped', 'existing-stop-callback']
        events.clear()
        sim._partial_clear()
    expected = ['physics-stopped', 'existing-stop-callback']
    if version == '3.7.1':
        expected += ['_pre_physics_step_callback', '_post_physics_step_callback', '_simulation_event_callback']
    assert events == expected + ['per-object-removal', 'sensor-removal', 'global-cache-cleanup']
    assert '_partial_clear' not in vars(sim) and 'clear' not in vars(scene)
    assert list(sim.callbacks) == ['existing']


def test_failed_cleanup_marks_engine_unusable_and_does_not_swallow_error():
    session, og, _, _, state = lifecycle_fixture()
    def clear():
        raise RuntimeError('Native object cleanup failed')
    og.clear = clear
    with pytest.raises(RuntimeError, match='Native object cleanup failed'):
        session.close()
    assert not state['reusable']


def test_unsupported_subscription_api_rejects_reuse_before_mutating_scene():
    sim = SimpleNamespace(scenes=[], _partial_clear=lambda: None,
        _pre_physics_step_callback=object(), _post_physics_step_callback=object(),
        _simulation_event_callback=object())
    og = SimpleNamespace(sim=sim, __version__='3.7.1', clear=lambda: None)
    with pytest.raises(RuntimeError, match='do not support'):
        with native_stage_cleanup(og):
            pytest.fail('Unsupported cleanup was allowed')


def test_subscription_failure_restores_native_method_and_propagates_before_scene_removal():
    events, callbacks = [], {}
    def fail():
        raise RuntimeError('Native unsubscribe failed')
    def partial_clear():
        for callback in callbacks.values():
            callback()
        events.append('scene-removed')
    sim = SimpleNamespace(scenes=[], _partial_clear=partial_clear,
        add_callback_on_stop=lambda name, callback: callbacks.update({name: callback}),
        remove_callback_on_stop=lambda name: callbacks.pop(name),
        _pre_physics_step_callback=SimpleNamespace(unsubscribe=lambda: events.append('pre-unsubscribed')),
        _post_physics_step_callback=SimpleNamespace(unsubscribe=fail),
        _simulation_event_callback=SimpleNamespace(unsubscribe=lambda: events.append('event-unsubscribed')))
    og = SimpleNamespace(sim=sim, __version__='3.7.1', clear=lambda: None)
    with pytest.raises(RuntimeError, match='Native unsubscribe failed'):
        with native_stage_cleanup(og):
            sim._partial_clear()
    assert sim._partial_clear is partial_clear and callbacks == {}
    assert events == ['pre-unsubscribed']


def test_swallowed_source_clear_error_is_preserved_without_a_second_native_clear():
    calls = []
    def clear():
        calls.append(True)
        raise ValueError('First native clear failed')
    og = SimpleNamespace(sim=SimpleNamespace(scenes=[]), clear=clear)
    with pytest.raises(ValueError, match='First native clear failed'):
        with native_stage_cleanup(og):
            for _ in range(2):
                try:
                    og.clear()
                except ValueError:
                    pass
    assert calls == [True] and og.clear is clear


def test_deletion_regex_cache_preserves_native_matching_and_restores_module(monkeypatch):
    import re
    import sys
    from wia.runtime.commands.run_esi_capture_service import cached_deletion_patterns
    compilations = []
    def compile(pattern, flags=0):
        compilations.append(pattern)
        return re.compile(pattern, flags)
    original = SimpleNamespace(compile=compile, match=re.match)
    module = SimpleNamespace(re=original)
    monkeypatch.setitem(sys.modules, 'isaacsim.core.prims.impl.prim', module)
    # More patterns than Python's normal 512-entry cache, revisited across
    # deletion callbacks. Native regex meaning and callback execution persist.
    patterns = [f'^/World/scene_0/actor_{i}/link_[0-9]+$' for i in range(600)]
    with cached_deletion_patterns() as enabled:
        assert enabled
        for _ in range(2):
            for i, pattern in enumerate(patterns):
                for path in (f'/World/scene_0/actor_{i}/link_2', f'/World/scene_0/actor_{i}/other'):
                    assert bool(module.re.match(pattern, path)) == bool(re.match(pattern, path))
    assert len(compilations) == 600 and module.re is original


def test_extra_scene_camera_removed_before_its_parent_without_double_removing_robot_sensors(monkeypatch):
    import sys
    from wia.runtime.commands.run_esi_capture_service import remove_unowned_scene_cameras
    removed = []
    def sensor(name):
        return SimpleNamespace(remove=lambda: removed.append(name))
    mounted, viewer, overview = (sensor(name) for name in ('mounted', 'viewer', 'overview'))
    monkeypatch.setitem(sys.modules, 'omnigibson.sensors', SimpleNamespace(VisionSensor=SimpleNamespace(
        SENSORS={'mounted': mounted, 'viewer': viewer, 'overview': overview})))
    scene = SimpleNamespace(robots=[SimpleNamespace(sensors={'mounted': mounted})])
    og = SimpleNamespace(sim=SimpleNamespace(scenes=[scene], viewer_camera=viewer))
    assert remove_unowned_scene_cameras(og) == 1
    assert removed == ['overview']


def test_runtime_key_fences_code_family_and_render_settings(tmp_path):
    scene = tmp_path / 'scene.yaml'
    scene.write_text(yaml.safe_dump({'runtime': {'camera_size': [512, 512]}}))
    manifest = {'code_commit': 'original', 'esi_commit': 'esi-pin', 'code_root': str(tmp_path),
        'big_task': 'Big', 'small_task': 'Family', 'implementation_hashes': {'native.py': 'hash'},
        'snapshot_scene_file': str(scene)}
    key = runtime_identity(manifest)
    assert runtime_identity(dict(manifest)) == key
    for field in ('code_commit', 'esi_commit', 'big_task', 'small_task'):
        assert runtime_identity({**manifest, field: 'changed'}) != key
    scene.write_text(yaml.safe_dump({'runtime': {'camera_size': [256, 256]}}))
    assert runtime_identity(manifest) != key


def test_cpu_preload_lookahead_stays_inside_the_explicit_pilot(tmp_path, monkeypatch):
    from concurrent.futures import Future
    queue = CaptureQueue(tmp_path / 'queue')
    resident = ResidentCapture(queue, {}, {'id': 'worker0'}, lambda: False, None)
    manifests = {}
    for index in range(3):
        job = 'capture_' + str(index) * 32
        manifests[job] = {'job_id': job, 'code_commit': 'pin', 'big_task': 'Big', 'small_task': 'Small'}
        write_json(queue.root / 'queued' / (job + '.json'), {'job_id': job})
    monkeypatch.setattr(queue, 'manifest', lambda job: manifests[job])
    chosen = list(manifests)[:2]
    # The current claim has already left the queue before read-ahead starts.
    (queue.root / 'queued' / (chosen[0] + '.json')).unlink()
    resident.allowed_job_ids = chosen
    submitted = []

    def submit(path):
        submitted.append(path.parent.parent.name)
        future = Future()
        future.set_result({'cpu_fixture_only': True})
        return future

    closed = []
    resident.preloader = SimpleNamespace(submit=submit, close=lambda: closed.append(True))
    result = resident.preload_inputs(manifests[chosen[0]])
    assert result.result()['cpu_fixture_only']
    assert submitted == chosen
    resident.preload_inputs(manifests[chosen[0]])
    assert submitted == chosen
    resident.close_preloader()
    assert closed and resident.preloader is None


def test_queue_affinity_never_overtakes_higher_priority_feedback(tmp_path):
    queue = CaptureQueue(tmp_path / 'queue')
    resident = ResidentCapture(queue, {}, {'id': 'worker0'}, lambda: False, None)
    resident.process = SimpleNamespace(poll=lambda: None)
    resident.family = ('pin', 'Big', 'Family')
    for job, family, priority in [('capture_a', 'Family', 0), ('capture_b', 'Other', 100)]:
        write_json(queue.root / 'queued' / (job + '.json'), {'job_id': job})
        write_json(queue.root / 'priorities' / (job + '.json'), {'priority': priority})
        write_json(queue.root / 'category_records' / (job + '.json'), {'job_id': job,
            'code_commit': 'pin', 'big_task': 'Big', 'small_task': family})
    assert resident.next_job_id() is None
    write_json(queue.root / 'priorities/capture_a.json', {'priority': 100})
    assert resident.next_job_id() == 'capture_a'


@pytest.mark.parametrize('foreign_response', [False, True])
@pytest.mark.parametrize('job_limit_reached', [False, True])
def test_owned_response_gate_preserves_images_and_releases_claim(
        tmp_path, monkeypatch, foreign_response, job_limit_reached):
    from PIL import Image
    import wia.runtime.scene_capture_resident as module
    queue = CaptureQueue(tmp_path / 'queue')
    job = 'capture_' + 'a' * 32
    snapshot = queue.job_dir(job) / 'input'
    snapshot.mkdir(parents=True)
    manifest = {'job_id': job, 'snapshot_dir': str(snapshot), 'input_hashes': {},
        'code_root': str(tmp_path), 'implementation_hashes': {}, 'capture_signature': 'CPU-fixture',
        'fingerprint': 'CPU-fingerprint', 'code_commit': 'CPU-pin', 'esi_commit': 'CPU-esi',
        'variant': 'original', 'seed': 0, 'expected_views': ['head'], 'timeout_s': 5, 'max_attempts': 2}
    write_json(snapshot / 'manifest.json', manifest)
    write_json(queue.job_dir(job) / 'accepted.json', {'manifest_sha256': sha256(snapshot / 'manifest.json')})
    queue.enqueue(manifest)
    claim = queue.claim({'id': 'worker0'})
    appdata = tmp_path / 'appdata'
    (appdata / 'native_tmp').mkdir(parents=True)
    resident = ResidentCapture(queue, {'appdata_path': str(appdata), 'texture_cache_max_bytes': 1024 ** 3},
                               {'id': 'worker0'}, lambda: False, None, max_jobs=1 if job_limit_reached else 16)
    resident.process = SimpleNamespace(pid=987654321, poll=lambda: None)
    resident.native_owner = {'start_ticks': 'CPU-fixture'}
    resident.control, resident.key = tmp_path / 'control', 'CPU-runtime'
    monkeypatch.setattr(resident, 'ensure', lambda manifest: None)
    closed = []
    monkeypatch.setattr(resident, 'close', lambda: (closed.append(True), True)[1])

    def fake_service(path, value):
        write_json(path, value)
        if path.parent.name != 'requests':
            return
        output = queue.job_dir(job) / 'attempts/01'
        image = output / 'head.png'
        Image.new('RGB', (64, 64), 'blue').save(image)
        receipt = {key: manifest[key] for key in ('capture_signature', 'fingerprint', 'code_commit',
                                                'esi_commit', 'variant', 'seed')}
        receipt.update(capture_protocol=PROTOCOL, complete=True, images={'head': str(image)},
                       timings={}, cpu_fixture_only=True, physical_grasp_verified=False)
        write_json(output / 'preview_complete.json', receipt)
        write_json(resident.control / 'responses' / path.name, {'job_id': 'foreign' if foreign_response else job,
            'request_id': value['request_id'], 'native_pid': resident.process.pid, 'runtime_key': resident.key,
            'state': 'CAPTURED', 'reusable': True})

    monkeypatch.setattr(module, 'write_json', fake_service)
    resident.capture(claim)
    assert not (queue.root / 'claimed' / (job + '.json')).exists()
    assert (queue.job_dir(job) / 'attempts/01/head.png').is_file()
    if foreign_response:
        assert queue.status(job)['state'] == 'FAILED' and closed
        failure = read_json(queue.job_dir(job) / 'attempts/01/capture_failure.json')
        assert 'identity' in failure['error']
    else:
        assert queue.status(job)['state'] == 'REVIEW_PENDING'
        assert not closed
        assert resident.worker_state()['state'] == 'NATIVE_IDLE'
        receipt = read_json(queue.job_dir(job) / 'attempts/01/preview_complete.json')
        assert receipt['native_process_running'] is True
        if job_limit_reached:
            assert resident.recycle_before_next['reasons'] == ['completed_job_limit']
            assert receipt['resident_recycle_before_next'] == resident.recycle_before_next


def test_deferred_recycle_precedes_the_next_native_load(tmp_path, monkeypatch):
    import wia.runtime.scene_capture_resident as module
    events = []
    def begin_native_load(path):
        events.append('next-load')
        raise RuntimeError('CPU fixture stops before native launch')

    resident = ResidentCapture(None, {'appdata_path': str(tmp_path)}, {}, lambda: False,
                              SimpleNamespace(prepare_native_tmp=begin_native_load))
    resident.process = SimpleNamespace(poll=lambda: None)
    resident.key = 'same-family'
    resident.recycle_before_next = {'reasons': ['texture_cache_limit']}

    def close():
        events.append('recycle')
        resident.process = None
        resident.recycle_before_next = None
        return True

    monkeypatch.setattr(resident, 'close', close)
    monkeypatch.setattr(module, 'runtime_identity', lambda manifest: 'same-family')
    with pytest.raises(RuntimeError, match='CPU fixture stops'):
        resident.ensure({})
    assert events == ['recycle', 'next-load']


@pytest.mark.parametrize('timeout_kind', ['wait', 'group'])
def test_native_shutdown_timeout_keeps_failed_claim_until_confirmed_exit(tmp_path, monkeypatch, timeout_kind):
    import subprocess
    import wia.runtime.scene_capture_resident as module
    queue = CaptureQueue(tmp_path / 'queue')
    job = 'capture_' + 'c' * 32
    snapshot = queue.job_dir(job) / 'input'
    snapshot.mkdir(parents=True)
    manifest = {'job_id': job, 'snapshot_dir': str(snapshot), 'input_hashes': {},
        'code_root': str(tmp_path), 'implementation_hashes': {}, 'capture_signature': 'CPU-timeout',
        'fingerprint': 'CPU-fingerprint', 'code_commit': 'CPU-pin', 'esi_commit': 'CPU-esi',
        'variant': 'original', 'seed': 0, 'expected_views': ['head'], 'timeout_s': 5, 'max_attempts': 2}
    write_json(snapshot / 'manifest.json', manifest)
    write_json(queue.job_dir(job) / 'accepted.json', {'manifest_sha256': sha256(snapshot / 'manifest.json')})
    queue.enqueue(manifest)
    claim = queue.claim({'id': 'worker0'})
    alive, stopped, cleaned = [True], [], []
    process = SimpleNamespace(pid=987654321, returncode=-9, poll=lambda: None if alive[0] else -9)

    def stop_group(*args):
        stopped.append(True)
        if alive[0]:
            if timeout_kind == 'wait':
                raise subprocess.TimeoutExpired(['native-fixture'], 5)
            raise RuntimeError('Owned simulator process group did not exit; claim retained.')

    helpers = SimpleNamespace(stop_group=stop_group,
        cleanup_native_tmp=lambda *args, **kwargs: (cleaned.append(True), {})[1])
    resident = ResidentCapture(queue, {'appdata_path': str(tmp_path / 'appdata')},
                               {'id': 'worker0'}, lambda: False, helpers)
    resident.process, resident.control = process, tmp_path / 'service'
    resident.native_owner, resident.key = {'start_ticks': 'CPU-fixture'}, 'CPU-runtime'

    def timed_out_load(manifest):
        raise TimeoutError('Resident capture exceeded its immutable timeout.')

    monkeypatch.setattr(resident, 'ensure', timed_out_load)
    resident.capture(claim)
    assert queue.status(job)['state'] == 'FAILED'
    assert queue.status(job)['native_shutdown_pending'] is True
    assert (queue.root / 'claimed' / (job + '.json')).exists()
    assert resident.worker_state()['state'] == 'WAITING_NATIVE_EXIT'
    assert resident.resume_shutdown() is False
    assert not cleaned and len(stopped) == 1
    monkeypatch.setattr(module, 'runtime_identity', lambda manifest: 'next-runtime')
    with pytest.raises(module.NativeShutdownPending):
        ResidentCapture.ensure(resident, manifest)
    assert resident.process is process and not cleaned
    alive[0] = False
    assert resident.resume_shutdown() is True
    assert resident.process is None and cleaned
    assert not (queue.root / 'claimed' / (job + '.json')).exists()
    assert queue.status(job)['state'] == 'QUEUED'
    assert (queue.job_dir(job) / 'attempts/01/capture_failure.json').is_file()


def test_native_preparation_helpers_require_exact_pinned_bytes(tmp_path):
    helper = tmp_path / 'helper.py'
    helper.write_text('raise AssertionError("Changed helper must never execute")\n')
    runtime = tmp_path / 'runtime.json'
    write_json(runtime, {'schema': 'native_asset_preparation_runtime_v1', 'modules': {
        name: {'path': str(helper), 'sha256': 'incorrect'}
        for name in ('src/wia/runtime/scene_capture_assets.py', 'src/wia/runtime/sim/esi_prepared_assets.py')}})
    with pytest.raises(ValueError, match='helper changed'):
        load_asset_runtime(runtime, sha256)


def test_old_prepared_cache_entries_are_pruned_only_when_cpu_work_has_ended(tmp_path):
    from concurrent.futures import Future
    root = tmp_path / 'owned-cache'
    root.mkdir()
    resident = ResidentCapture(None, {}, {}, lambda: False, None)
    resident.asset_cache_root = root
    for job in ('expired', 'current'):
        entry = root / job
        entry.mkdir()
        future = Future()
        future.set_result({'prepared_assets': {'assets': [{'cache_manifest': str(entry / 'manifest.json')}]}})
        resident.preload_futures[job] = future
    pending = Future()
    resident.preload_futures['pending'] = pending
    resident.prune_prepared_assets(['current'])
    assert (root / 'expired').is_dir()
    pending.set_result({})
    resident.prune_prepared_assets(['current'])
    assert not (root / 'expired').exists() and (root / 'current').is_dir()
    assert set(resident.preload_futures) == {'current'}


@pytest.mark.parametrize('diagnostic', [False, True])
def test_recovery_cannot_bypass_qualification_or_a_live_native_owner(tmp_path, diagnostic):
    import os
    from pathlib import Path
    queue = CaptureQueue(tmp_path / 'queue')
    job = 'capture_' + 'b' * 32
    path = queue.job_dir(job) / 'input/manifest.json'
    write_json(path, {'job_id': job, 'diagnostic_pending': diagnostic})
    write_json(queue.job_dir(job) / 'accepted.json', {'manifest_sha256': sha256(path)})
    output = queue.job_dir(job) / 'attempts/01'
    write_json(queue.job_dir(job) / 'status.json', {'state': 'FAILED', 'native_output': str(output)})
    ticks = Path(f'/proc/{os.getpid()}/stat').read_text().split(') ', 1)[1].split()[19]
    write_json(output / 'native_process.json', {'pid': os.getpid(), 'start_ticks': ticks, 'capture_job_id': job})
    with pytest.raises(ValueError, match='confirmed native process exit' if diagnostic else 'qualification bypass'):
        recover_rendered_attempt(queue, job, 'Synthetic administrative recovery fixture')

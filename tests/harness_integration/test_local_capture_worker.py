"""Check local CUDA/Vulkan ownership without importing a GPU simulator.

Run: python -m pytest -q tests/test_local_capture_worker.py
These fixtures verify dispatch and isolation contracts, not native rendering.
"""
import json
import socket
from types import SimpleNamespace

import pytest

from wia.runtime.scene_capture_queue import write_json
import wia.runtime.commands.run_esi_capture_worker as worker
from wia.runtime.commands.capture_esi_scene_reviews import capture_manifest_entrypoint


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_GPUFILE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    devices = [{"physical_index": i, "uuid": f"GPU-fixture-{i}",
                "pci_bus_id": f"0000:{i:02x}:00.0"} for i in range(8)]

    def query(command, **kwargs):
        assert command[0] == "nvidia-smi", "Local dispatch must not query a scheduler."
        return "\n".join(f"{d['physical_index']}, {d['uuid']}, {d['pci_bus_id']}" for d in devices)

    monkeypatch.setattr(worker.subprocess, "check_output", query)
    monkeypatch.setattr(worker, "vulkan_devices", lambda: [
        {"uuid": d["uuid"], "renderer_index": i} for i, d in enumerate(reversed(devices))])
    return devices


def test_all_eight_direct_slots_use_uuid_matched_renderer(inventory):
    mappings = [worker.allocation_gpu(i) for i in range(8)]
    assert len({gpu["uuid"] for gpu in mappings}) == 8
    assert [gpu["renderer_index"] for gpu in mappings] == list(range(8))
    assert [gpu["vulkan_enumeration_index"] for gpu in mappings] == list(reversed(range(8)))
    assert all(gpu["execution_mode"] == "local" and gpu["pbs_job_id"] is None for gpu in mappings)
    assert all(gpu["allocation_gpu_count"] == 8 for gpu in mappings)


def test_local_dispatch_respects_selected_visibility(inventory, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture-6,2")
    assert worker.allocation_gpu(0, "local")["physical_index"] == 6
    assert worker.allocation_gpu(1, "local")["physical_index"] == 2
    with pytest.raises(ValueError, match="outside"):
        worker.allocation_gpu(2, "local")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="visibility"):
        worker.allocation_gpu(0, "local")


def test_missing_native_renderer_is_not_assumed_usable(inventory, monkeypatch):
    monkeypatch.setattr(worker, "vulkan_devices", lambda: [])
    with pytest.raises(ValueError, match="Vulkan"):
        worker.allocation_gpu(0, "local")


def test_existing_pbs_dispatch_retains_allocation_validation(inventory, monkeypatch):
    original = worker.subprocess.check_output
    monkeypatch.setenv("PBS_JOBID", "fixture.pbs")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture-7,GPU-fixture-3")

    def query(command, **kwargs):
        if command[0] == "qstat":
            return json.dumps({"Jobs": {"fixture.pbs": {"job_state": "R", "interactive": True,
                "Resource_List": {"walltime": "12:00:00", "ngpus": 2}}}})
        return original(command, **kwargs)

    monkeypatch.setattr(worker.subprocess, "check_output", query)
    gpu = worker.allocation_gpu(1)
    assert gpu["physical_index"] == 3 and gpu["execution_mode"] == "pbs"
    assert gpu["pbs_job_id"] == "fixture.pbs"
    assert gpu["renderer_order"] == "vulkan" and gpu["renderer_index"] == 4


def test_remote_local_orphan_requires_exit_evidence(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Local remote-host recovery must not invoke qstat.")

    monkeypatch.setattr(worker.subprocess, "check_output", forbidden)
    assert not worker.confirm_stopped({"token": "owned", "hostname": socket.gethostname() + "-other",
                                       "worker": {"execution_mode": "local", "pbs_job_id": None}})


def test_local_native_entrypoint_checks_job_and_assignment(tmp_path, monkeypatch):
    import wia.runtime.scene_capture_adapters as adapters

    monkeypatch.delenv("PBS_JOBID", raising=False)
    gpu = {"execution_mode": "local", "pbs_job_id": None, "uuid": "GPU-fixture-0"}
    manifest, gpu_file = tmp_path / "manifest.json", tmp_path / "gpu.json"
    write_json(manifest, {"job_id": "capture_fixture"})
    write_json(gpu_file, gpu)
    captured = []
    monkeypatch.setattr(adapters, "capture_snapshot", lambda *args: captured.append(args))
    arguments = ["--capture-manifest", str(manifest), "--capture-output", str(tmp_path),
                 "--gpu-info", str(gpu_file), "--execution-mode", "local", "--job-id", "capture_fixture"]
    assert capture_manifest_entrypoint(arguments) == 0
    assert captured == [(manifest, tmp_path, gpu)]
    arguments[-1] = "capture_other"
    with pytest.raises(SystemExit):
        capture_manifest_entrypoint(arguments)
    assert len(captured) == 1


def test_native_cleanup_preserves_another_workers_shared_texture_cache(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from wia.runtime.scene_capture_adapters import isolate_texture_cache

    home = tmp_path / "home"
    shared = home / ".cache/ov/texturecache"
    private = tmp_path / "worker/xdg_cache/ov/texturecache"
    for path in (shared, private):
        path.mkdir(parents=True)
        (path / "owned.txt").write_text("Keep the other worker's cache.")
    monkeypatch.setattr(worker.Path, "home", lambda: home)
    simulator = SimpleNamespace(shutil=shutil)
    isolate_texture_cache(simulator, tmp_path / "worker")
    simulator.shutil.rmtree(shared)
    assert (shared / "owned.txt").is_file()
    assert not private.exists()


def test_texture_limit_preserves_captures_other_workers_and_reusable_shader_cache(tmp_path):
    appdata = tmp_path / 'worker0'
    texture = appdata / 'xdg_cache/ov/texturecache'
    texture.mkdir(parents=True)
    (texture / 'reusable.texture').write_bytes(b'x' * 16)
    shader = appdata / 'nvidia_shader_cache/shader.bin'
    shader.parent.mkdir()
    shader.write_bytes(b'shader')
    other = tmp_path / 'worker1/xdg_cache/ov/texturecache/other.texture'
    other.parent.mkdir(parents=True)
    other.write_bytes(b'other worker')
    capture = tmp_path / 'durable_capture/head.png'
    capture.parent.mkdir()
    capture.write_bytes(b'preserved synthetic fixture')
    assert worker.prune_worker_texture_cache(appdata, maximum_bytes=16) is None
    result = worker.prune_worker_texture_cache(appdata, maximum_bytes=8)
    assert result['bytes_before'] == 16 and not texture.exists()
    assert shader.read_bytes() == b'shader' and other.read_bytes() == b'other worker'
    assert capture.read_bytes() == b'preserved synthetic fixture'
    assert 'prune_disposable_worker_texture_cache' in (appdata / 'texture_cache_maintenance.jsonl').read_text()


def test_texture_limit_rejects_external_cache_symlink(tmp_path):
    external = tmp_path / 'unrelated'
    external.mkdir()
    (external / 'keep').write_text('Preserve unrelated data.')
    appdata = tmp_path / 'worker'
    path = appdata / 'xdg_cache/ov/texturecache'
    path.parent.mkdir(parents=True)
    path.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='inside this worker'):
        worker.prune_worker_texture_cache(appdata, maximum_bytes=1)
    assert (external / 'keep').read_text() == 'Preserve unrelated data.'


def test_private_native_tmp_prepares_and_cleans_only_its_contents(tmp_path, monkeypatch):
    appdata = tmp_path / 'worker0'
    scratch = appdata / 'native_tmp'
    decrypted = scratch / 'og_decrypted/scene.usd'
    decrypted.parent.mkdir(parents=True)
    decrypted.write_bytes(b'old scratch')
    preserved = [appdata / 'native.log', appdata / 'head.png',
        appdata / 'xdg_cache/ov/texturecache/keep.texture',
        tmp_path / 'worker1/native_tmp/active.usd']
    for path in preserved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'preserve')
    assert worker.prepare_native_tmp(appdata) == scratch.resolve()
    assert scratch.is_dir() and not list(scratch.iterdir())
    (scratch / 'new.usd').write_bytes(b'new scratch')
    external = tmp_path / 'unrelated'
    external.mkdir()
    (external / 'keep.usd').write_bytes(b'preserve')
    (scratch / 'external_link').symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(worker, 'group_members', lambda pid: [])
    result = worker.cleanup_native_tmp(appdata, SimpleNamespace(pid=987654321, poll=lambda: -6))
    assert result == {'path': str(scratch.resolve()), 'removed_entries': 2}
    assert not list(scratch.iterdir())
    assert all(path.read_bytes() == b'preserve' for path in preserved)
    assert (external / 'keep.usd').read_bytes() == b'preserve'


@pytest.mark.parametrize('live_child,live_descendants', [(True, False), (False, True)])
def test_native_tmp_cleanup_refuses_live_child_or_descendants(tmp_path, monkeypatch, live_child, live_descendants):
    appdata = tmp_path / 'worker'
    scratch = worker.prepare_native_tmp(appdata)
    active = scratch / 'in_use.usd'
    active.write_bytes(b'live world')
    monkeypatch.setattr(worker, 'group_members', lambda pid: [123] if live_descendants else [])
    process = SimpleNamespace(pid=987654321, poll=lambda: None if live_child else 0)
    with pytest.raises(RuntimeError, match='confirmed process-group exit'):
        worker.cleanup_native_tmp(appdata, process)
    assert active.read_bytes() == b'live world'


@pytest.mark.parametrize('kind', ['appdata', 'ancestor', 'native_tmp', 'owner', 'escape'])
def test_native_tmp_rejects_symlink_or_escape_paths(tmp_path, kind):
    external = tmp_path / 'unrelated'
    external.mkdir()
    sentinel = external / 'keep.usd'
    sentinel.write_bytes(b'unrelated')
    appdata = tmp_path / 'worker'
    if kind == 'appdata':
        appdata.symlink_to(external, target_is_directory=True)
    elif kind == 'ancestor':
        link = tmp_path / 'link'
        link.symlink_to(external, target_is_directory=True)
        appdata = link / 'worker'
    elif kind == 'native_tmp':
        appdata.mkdir()
        (appdata / 'native_tmp').symlink_to(external, target_is_directory=True)
    elif kind == 'owner':
        appdata.mkdir()
        (appdata / 'native_tmp_owner.json').symlink_to(sentinel)
    else:
        appdata = tmp_path / 'worker' / '..' / 'unrelated'
    with pytest.raises(ValueError, match='symlink|escape'):
        worker.prepare_native_tmp(appdata)
    assert sentinel.read_bytes() == b'unrelated'


class FakeCaptureQueue:
    """Minimal queue contract for child-environment and evidence-order checks."""
    def __init__(self, root):
        self.root, self.events = root, []
        self.job = root / 'jobs/capture_fixture'
        self.job.mkdir(parents=True)
        self.value = {'code_root': str(root), 'code_commit': 'fixture',
            'capture_signature': 'fixture_signature', 'timeout_s': 100, 'max_attempts': 1}

    def job_dir(self, job_id):
        assert job_id == 'capture_fixture'
        return self.job

    def manifest(self, job_id):
        return self.value

    def status(self, job_id):
        return {'attempts': 1}

    def cached(self, manifest):
        return None

    def heartbeat(self, claim, **kwargs):
        self.events.append(('heartbeat', kwargs))

    def transition(self, claim, state, **kwargs):
        self.events.append((state, kwargs))

    def finish(self, claim, receipt):
        assert receipt.is_file()
        self.events.append(('finish', json.loads(receipt.read_text())))

    def release(self, claim):
        self.events.append(('release', {}))


@pytest.mark.parametrize('mode', ['success', 'failure', 'retained', 'live_worker_error'])
def test_capture_job_assigns_private_tmp_and_preserves_evidence_before_cleanup(tmp_path, monkeypatch, mode):
    queue = FakeCaptureQueue(tmp_path / 'queue')
    appdata = tmp_path / 'worker'
    scratch = worker.prepare_native_tmp(appdata)
    texture = appdata / 'xdg_cache/ov/texturecache/keep.texture'
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b'reusable')
    claim = {'job_id': 'capture_fixture', 'worker': {'id': 'worker0'}}
    output = queue.job / 'attempts/01'
    gpu = {'uuid': 'GPU-fixture', 'execution_mode': 'local', 'appdata_path': str(appdata),
        'retain_world': mode == 'retained'}
    code = -6 if mode == 'failure' else 0
    process = SimpleNamespace(pid=987654321, running=mode in ('retained', 'live_worker_error'))
    process.poll = lambda: None if process.running else code
    process.wait = lambda: code
    calls, cleanup_calls = [], []

    def launch(command, *, cwd, env, stdout, stderr, start_new_session):
        calls.append(env)
        assert {env[key] for key in ('TMPDIR', 'TMP', 'TEMP')} == {str(scratch)}
        assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-fixture' and env['OMNIGIBSON_GPU_ID'] == '0'
        (scratch / 'decrypted').mkdir()
        (scratch / 'decrypted/scene.usd').write_bytes(b'private native temp')
        (output / 'head.png').write_bytes(b'synthetic PNG evidence')
        stdout.write('Synthetic native log.\n')
        if mode == 'failure':
            write_json(output / 'failure.json', {'error': 'Synthetic native abort.', 'failure_stage': 'native_init'})
        else:
            write_json(output / 'preview_complete.json', {'complete': True, 'timings': {}})
        if mode == 'retained':
            write_json(output / 'retained_world.json', {'state': 'RETAINED', 'pid': process.pid,
                'job_id': 'capture_fixture', 'capture_signature': 'fixture_signature'})
        return process

    cleanup = worker.cleanup_native_tmp
    def checked_cleanup(path, child, *, owner=None):
        cleanup_calls.append(child.poll())
        assert (output / 'head.png').read_bytes() == b'synthetic PNG evidence'
        assert (output / 'native.log').read_text() == 'Synthetic native log.\n'
        assert (scratch / 'decrypted/scene.usd').exists()
        if mode == 'live_worker_error':
            assert (output / 'native_worker_error.json').is_file()
        else:
            assert (output / 'process_exit.json').is_file()
            if mode == 'failure':
                failure = json.loads((output / 'capture_failure.json').read_text())
                assert failure['partial_images']['head'] == str(output / 'head.png')
            elif mode == 'retained':
                receipt = json.loads((output / 'preview_complete.json').read_text())
                assert receipt['process_running'] is False
            else:
                assert any(event[0] == 'finish' for event in queue.events)
        return cleanup(path, child, owner=owner)

    def sleep(seconds):
        assert mode == 'retained' and process.running
        assert not cleanup_calls and (scratch / 'decrypted/scene.usd').exists()
        assert any(event[0] == 'finish' for event in queue.events)
        process.running = False

    heartbeat = queue.heartbeat
    def failing_heartbeat(claim, **kwargs):
        if kwargs.get('child_pid'):
            owner = json.loads((appdata / 'native_tmp_owner.json').read_text())
            assert owner['pid'] == owner['pgid'] == process.pid
            assert owner['job_id'] == claim['job_id'] and owner['state'] == 'RUNNING'
            assert owner['start_ticks'] == ('fixture_ticks' if process.running else None)
        if kwargs.get('child_pid') and mode == 'live_worker_error':
            raise ValueError('Synthetic heartbeat error while native process is live.')
        return heartbeat(claim, **kwargs)

    monkeypatch.setenv('TMPDIR', str(tmp_path / 'unrelated_inherited_tmp'))
    monkeypatch.setattr(worker.subprocess, 'Popen', launch)
    monkeypatch.setattr(worker, 'verify_snapshot', lambda manifest: None)
    monkeypatch.setattr(worker, 'complete_images', lambda manifest, receipt: receipt.get('complete') is True)
    monkeypatch.setattr(worker, 'process_ticks', lambda pid: 'fixture_ticks' if process.running else None)
    monkeypatch.setattr(worker, 'group_members', lambda pid: [])
    monkeypatch.setattr(worker.time, 'sleep', sleep)
    monkeypatch.setattr(worker, 'cleanup_native_tmp', checked_cleanup)
    monkeypatch.setattr(queue, 'heartbeat', failing_heartbeat)
    if mode == 'live_worker_error':
        with pytest.raises(ValueError, match='Synthetic heartbeat error'):
            worker.capture_job(queue, claim, gpu)
    else:
        worker.capture_job(queue, claim, gpu)
    assert len(calls) == 1 and len(cleanup_calls) == 1
    assert texture.read_bytes() == b'reusable'
    assert (output / 'head.png').is_file() and (output / 'native.log').is_file()
    gpu_evidence = json.loads((output / 'gpu.json').read_text())
    command_evidence = json.loads((output / 'command.json').read_text())
    assert gpu_evidence['native_tmp_path'] == command_evidence['native_tmp_path'] == str(scratch)
    assert command_evidence['temporary_environment'] == {key: str(scratch) for key in ('TMPDIR', 'TMP', 'TEMP')}
    cleanup_evidence = json.loads((output / 'native_tmp_cleanup.json').read_text())
    if mode == 'live_worker_error':
        assert cleanup_evidence['status'] == 'preserved' and process.running
        assert (scratch / 'decrypted/scene.usd').is_file()
        assert (appdata / 'native_tmp_owner.json').is_file()
        with pytest.raises(RuntimeError, match='remain alive'):
            worker.prepare_native_tmp(appdata)
        assert (scratch / 'decrypted/scene.usd').is_file()
        process.running = False
        worker.prepare_native_tmp(appdata)
        assert not list(scratch.iterdir()) and not (appdata / 'native_tmp_owner.json').exists()
        assert texture.read_bytes() == b'reusable' and (output / 'head.png').is_file()
    else:
        assert cleanup_evidence['status'] == 'cleaned' and not process.running
        assert scratch.is_dir() and not list(scratch.iterdir())
        assert not (appdata / 'native_tmp_owner.json').exists()


@pytest.mark.parametrize('remaining_owner', ['native', 'descendant'])
def test_native_tmp_owner_fence_blocks_next_prepare_until_full_exit(tmp_path, monkeypatch, remaining_owner):
    appdata = tmp_path / 'worker'
    scratch = worker.prepare_native_tmp(appdata)
    process = SimpleNamespace(pid=987654321, running=True)
    process.poll = lambda: None if process.running else 0
    descendants = []
    monkeypatch.setattr(worker, 'process_ticks', lambda pid: 'owned_ticks' if process.running else None)
    monkeypatch.setattr(worker, 'group_members', lambda pgid: list(descendants))
    owner = worker.reserve_native_tmp(appdata, 'capture_previous')
    owner = worker.record_native_tmp_owner(appdata, process, owner)
    active = scratch / 'active.usd'
    active.write_bytes(b'in use')
    sibling = appdata / 'head.png'
    sibling.write_bytes(b'preserved evidence')
    if remaining_owner == 'descendant':
        process.running = False
        descendants.append(987654322)
    with pytest.raises(RuntimeError, match='confirmed process-group exit'):
        worker.cleanup_native_tmp(appdata, process, owner=owner)
    with pytest.raises(RuntimeError, match='remain alive'):
        worker.prepare_native_tmp(appdata)
    assert active.read_bytes() == b'in use'
    assert json.loads((appdata / 'native_tmp_owner.json').read_text()) == owner
    process.running = False
    descendants.clear()
    assert worker.prepare_native_tmp(appdata) == scratch
    assert not list(scratch.iterdir()) and not (appdata / 'native_tmp_owner.json').exists()
    assert sibling.read_bytes() == b'preserved evidence'
    assert worker.reserve_native_tmp(appdata, 'capture_next')['job_id'] == 'capture_next'


def test_native_tmp_cleanup_preserves_mismatched_owner_and_uncertain_launch(tmp_path, monkeypatch):
    appdata = tmp_path / 'worker'
    scratch = worker.prepare_native_tmp(appdata)
    process = SimpleNamespace(pid=987654321, poll=lambda: 0)
    monkeypatch.setattr(worker, 'process_ticks', lambda pid: None)
    monkeypatch.setattr(worker, 'group_members', lambda pgid: [])
    owner = worker.reserve_native_tmp(appdata, 'capture_original')
    (scratch / 'keep.usd').write_bytes(b'uncertain owner')
    with pytest.raises(RuntimeError, match='uncertain'):
        worker.prepare_native_tmp(appdata)
    owner = worker.record_native_tmp_owner(appdata, process, owner)
    changed = {**owner, 'token': 'another_owner'}
    write_json(appdata / 'native_tmp_owner.json', changed)
    with pytest.raises(RuntimeError, match='matching process owner'):
        worker.cleanup_native_tmp(appdata, process, owner=owner)
    assert (scratch / 'keep.usd').read_bytes() == b'uncertain owner'
    assert json.loads((appdata / 'native_tmp_owner.json').read_text()) == changed

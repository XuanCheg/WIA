#!/usr/bin/env python3
"""Persistent capture supervisor for a directly assigned GPU or PBS allocation.

Run after activating BEHAVIOR in the retained interactive terminal:
python src/wia/runtime/commands/run_esi_capture_worker.py --queue-root QUEUE --worker-id worker0 --gpu-slot 0
Use --execution-mode local for a standalone server, with one worker per GPU slot.
Add --persistent-simulator to retain the native engine across compatible jobs.
Queue capture_retention.json can opt selected GPU UUIDs into retaining the last
completed world until eligible queued work needs the GPU. Retention has no
elapsed-time limit; explicit worker stops and failures can still end it.
Pre-completion capture also has no time limit; explicit cancellation is retained.
No Agent episodes are run. Supervisors can share a queue across all idle GPUs.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import ctypes as ct
import fcntl
import json
import os
from pathlib import Path
from wia.runtime.resources import frozen_runtime_environment
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_capture_queue import (CaptureQueue, complete_images, publish_json, read_json,
                                      verify_snapshot, write_json)  # noqa: E402


def process_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process_alive(pid, start_ticks):
    return bool(start_ticks and process_ticks(pid) == str(start_ticks))


def fresh_gpu_available(gpu):
    """Recheck a fresh-process worker's device before claiming another job."""
    try:
        clients = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
            '--format=csv,noheader,nounits'], text=True)
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        values = next(line.split(',') for line in rows.splitlines()
                      if line.split(',')[0].strip() == gpu['uuid'])
        utilization, used, free = map(lambda value: int(value.strip()), values[1:])
        occupied = any(line.split(',')[1].strip() == gpu['uuid'] for line in clients.splitlines() if line.strip())
        if gpu.get('shared_gpu'):
            return free >= gpu['shared_gpu_min_free_mib']
        return not occupied and utilization == 0 and used <= 512
    except (OSError, ValueError, IndexError, KeyError, StopIteration, subprocess.SubprocessError):
        return False


def vulkan_devices():
    """Read Vulkan UUID ordering; PBS cgroups can renumber renderer devices."""
    class Header(ct.Structure):
        _fields_ = [("sType", ct.c_uint32), ("pNext", ct.c_void_p)]

    class Application(ct.Structure):
        _fields_ = Header._fields_ + [("pApplicationName", ct.c_char_p), ("applicationVersion", ct.c_uint32),
            ("pEngineName", ct.c_char_p), ("engineVersion", ct.c_uint32), ("apiVersion", ct.c_uint32)]

    class Instance(ct.Structure):
        _fields_ = Header._fields_ + [("flags", ct.c_uint32), ("pApplicationInfo", ct.POINTER(Application)),
            ("enabledLayerCount", ct.c_uint32), ("ppEnabledLayerNames", ct.c_void_p),
            ("enabledExtensionCount", ct.c_uint32), ("ppEnabledExtensionNames", ct.c_void_p)]

    class Identity(ct.Structure):
        _fields_ = Header._fields_ + [("deviceUUID", ct.c_ubyte * 16), ("driverUUID", ct.c_ubyte * 16),
            ("deviceLUID", ct.c_ubyte * 8), ("deviceNodeMask", ct.c_uint32), ("deviceLUIDValid", ct.c_uint32)]

    class Properties(ct.Structure):
        _fields_ = Header._fields_ + [("properties", ct.c_uint32 * 1024)]

    lib = ct.CDLL("libvulkan.so.1")
    lib.vkCreateInstance.argtypes = [ct.POINTER(Instance), ct.c_void_p, ct.POINTER(ct.c_void_p)]
    lib.vkEnumeratePhysicalDevices.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_void_p)]
    lib.vkGetPhysicalDeviceProperties2.argtypes = [ct.c_void_p, ct.POINTER(Properties)]
    lib.vkDestroyInstance.argtypes = [ct.c_void_p, ct.c_void_p]
    app = Application(sType=0, pApplicationName=b"esi_capture_device_check", apiVersion=(1 << 22) | (1 << 12))
    info = Instance(sType=1, pApplicationInfo=ct.pointer(app))
    instance = ct.c_void_p()
    if lib.vkCreateInstance(ct.byref(info), None, ct.byref(instance)) != 0:
        raise RuntimeError("Cannot inspect native Vulkan GPU identity.")
    try:
        count = ct.c_uint32()
        if lib.vkEnumeratePhysicalDevices(instance, ct.byref(count), None) != 0:
            raise RuntimeError("Cannot enumerate renderer GPUs.")
        handles = (ct.c_void_p * count.value)()
        if lib.vkEnumeratePhysicalDevices(instance, ct.byref(count), handles) != 0:
            raise RuntimeError("Cannot read renderer GPU handles.")
        devices = []
        for handle in handles:
            identity = Identity(sType=1000071004)
            properties = Properties(sType=1000059001, pNext=ct.cast(ct.pointer(identity), ct.c_void_p))
            lib.vkGetPhysicalDeviceProperties2(handle, ct.byref(properties))
            if properties.properties[2] == 0x10DE:
                devices.append({"renderer_index": len(devices),
                    "uuid": "GPU-" + str(uuid.UUID(bytes=bytes(identity.deviceUUID)))})
        return devices
    finally:
        lib.vkDestroyInstance(instance, None)


def allocation_gpu(slot, execution_mode="auto", renderer_order="auto"):
    """Resolve an explicitly selected visible slot and its Vulkan renderer UUID."""
    job_id = os.environ.get("PBS_JOBID")
    if execution_mode == "auto":
        execution_mode = "pbs" if job_id else "local"
    if execution_mode not in {"local", "pbs"}:
        raise ValueError("Execution mode must be auto, local or pbs.")
    count = None
    if execution_mode == "pbs":
        if not job_id:
            raise ValueError("PBS execution requires the assigned interactive GPU shell.")
        result = subprocess.check_output(["qstat", "-f", "-F", "json", job_id], text=True)
        job = json.loads(result)["Jobs"][job_id]
        if (job["job_state"] != "R" or str(job.get("interactive")).lower() != "true"
                or job["Resource_List"]["walltime"] != "12:00:00"):
            raise ValueError("Capture requires the valid 12-hour interactive allocation.")
        count = int(job["Resource_List"]["ngpus"])
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits"], text=True)
    devices = []
    for line in inventory.splitlines():
        index, gpu_uuid, bus = (x.strip() for x in line.split(","))
        devices.append({"physical_index": int(index), "uuid": gpu_uuid, "pci_bus_id": bus})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [x.strip() for x in visible.split(",") if x.strip()]
    if execution_mode == "pbs" and not tokens and os.environ.get("PBS_GPUFILE"):
        tokens = [line.rsplit("-gpu", 1)[-1].strip() for line in Path(os.environ["PBS_GPUFILE"]).read_text().splitlines()]
    if execution_mode == "local" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        tokens = [device["uuid"] for device in devices]
    elif execution_mode == "pbs" and not tokens and len(devices) == count:
        tokens = [device["uuid"] for device in devices]  # Verified whole-node GPU allocation.
    assigned = []
    for token in tokens:
        matches = [d for d in devices if token in (str(d["physical_index"]), d["uuid"])]
        if len(matches) != 1:
            raise ValueError("Cannot map CUDA GPU selection to a unique physical device: " + token)
        assigned.append(matches[0])
    if (not assigned or (count is not None and len(assigned) > count)
            or len({d["uuid"] for d in assigned}) != len(assigned)):
        raise ValueError("GPU visibility is unavailable or inconsistent.")
    if not 0 <= slot < len(assigned):
        raise ValueError("Requested worker slot is outside the selected visible GPUs.")
    native_devices = vulkan_devices()
    renderer = [device for device in native_devices if device["uuid"] == assigned[slot]["uuid"]]
    if len(renderer) != 1:
        raise ValueError("Assigned CUDA GPU has no uniquely matching Vulkan renderer UUID.")
    if renderer_order == "auto":
        renderer_order = "pci" if execution_mode == "local" else "vulkan"
    if renderer_order not in {"pci", "vulkan"}:
        raise ValueError("Renderer order must be auto, pci or vulkan.")
    renderer_index = renderer[0]["renderer_index"]
    if renderer_order == "pci":
        # Isaac GPU Foundation sorts its native GPU table by PCI bus, rather
        # than retaining vkEnumeratePhysicalDevices order on standalone hosts.
        native_uuids = {device["uuid"] for device in native_devices}
        ordered = sorted((d for d in devices if d["uuid"] in native_uuids),
                         key=lambda d: d["pci_bus_id"].lower())
        renderer_index = next(i for i, d in enumerate(ordered) if d["uuid"] == assigned[slot]["uuid"])
    return {**assigned[slot], "renderer_index": renderer_index,
            "vulkan_enumeration_index": renderer[0]["renderer_index"], "renderer_order": renderer_order,
            "slot": slot, "execution_mode": execution_mode,
            "pbs_job_id": job_id if execution_mode == "pbs" else None,
            "hostname": socket.gethostname(), "allocation_gpu_count": count or len(assigned),
            "inherited_cuda_visible_devices": visible}


def group_members(pgid):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().split(") ", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_group(pgid, process=None, *, grace_s=None):
    if pgid == os.getpgrp():
        raise RuntimeError("Refusing to terminate the worker or interactive shell process group.")
    if group_members(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while group_members(pgid):
        if process is not None:
            process.poll()
        time.sleep(.2)
    if process is not None:
        process.wait()


def confirm_stopped(claim):
    """Prove exit before recovery; stop only a verified owned orphan locally."""
    if not claim.get("token"):
        return True  # No owner metadata means no child was authorized to start.
    worker = claim.get("worker", {})
    if claim.get("hostname") != socket.gethostname():
        if not worker.get("pbs_job_id"):
            return False  # Local remote-host exit cannot be proven from a heartbeat.
        try:
            value = subprocess.check_output(["qstat", "-fx", "-F", "json", worker["pbs_job_id"]], text=True)
            return json.loads(value)["Jobs"][worker["pbs_job_id"]]["job_state"] == "F"
        except (OSError, subprocess.SubprocessError, KeyError):
            return False
    if owned_process_alive(claim["supervisor_pid"], worker.get("supervisor_start_ticks")):
        return False
    native = read_json(Path(claim.get("native_output", "/missing")) / "native_process.json", {})
    pid, ticks = claim.get("child_pid", native.get("pid")), claim.get("child_start_ticks", native.get("start_ticks"))
    if pid and owned_process_alive(pid, ticks):
        stop_group(int(claim.get("child_pgid") or native["pgid"]))
    pgid = claim.get("child_pgid", native.get("pgid"))
    if pgid and group_members(int(pgid)):
        stop_group(int(pgid))
    return True


def prune_worker_texture_cache(appdata, maximum_bytes=8 * 1024 ** 3):
    """Bound disposable, worker-private textures between native processes."""
    import shutil
    appdata = Path(appdata).resolve()
    cache = appdata / 'xdg_cache/ov/texturecache'
    if not cache.is_dir():
        return None
    if cache.is_symlink() or not cache.resolve().is_relative_to(appdata):
        raise ValueError('Texture cache maintenance must stay inside this worker cache.')
    size = sum(path.stat().st_size for path in cache.rglob('*') if path.is_file() and not path.is_symlink())
    if size <= maximum_bytes:
        return None
    record = {'operation': 'prune_disposable_worker_texture_cache', 'path': str(cache),
              'bytes_before': size, 'maximum_bytes': maximum_bytes, 'timestamp': time.time(),
              'note': 'No native process is running in this worker; captures and sealed inputs are preserved.'}
    shutil.rmtree(cache)
    with (appdata / 'texture_cache_maintenance.jsonl').open('a') as log:
        log.write(json.dumps(record) + '\n')
    return record


def _native_tmp_directory(appdata):
    """Resolve only the dedicated scratch directory under non-symlink appdata."""
    appdata = Path(appdata).absolute()
    if '..' in appdata.parts or any(path.is_symlink() for path in (appdata, *appdata.parents)):
        raise ValueError('Native scratch requires non-symlink worker appdata without path escapes.')
    appdata.mkdir(parents=True, exist_ok=True)
    appdata = appdata.resolve(strict=True)
    scratch = appdata / 'native_tmp'
    if scratch.is_symlink() or scratch.resolve() != scratch:
        raise ValueError('Native scratch must remain inside this worker appdata without symlinks.')
    scratch.mkdir(exist_ok=True)
    return scratch


def _clear_native_tmp(appdata):
    """Remove scratch contents only; nested symlinks are unlinked, never followed."""
    import shutil
    scratch = _native_tmp_directory(appdata)
    removed = 0
    for path in scratch.iterdir():
        if path.is_symlink() or not path.is_dir():
            path.unlink()
        else:
            shutil.rmtree(path)
        removed += 1
    return {'path': str(scratch), 'removed_entries': removed}


def prepare_native_tmp(appdata):
    """Prepare scratch only after any previous fenced native owner is gone."""
    fence = _native_tmp_owner_file(appdata)
    owner = read_json(fence)
    if fence.exists():
        if (not isinstance(owner, dict) or owner.get('state') != 'RUNNING' or
                type(owner.get('pid')) is not int or owner['pid'] <= 0 or
                owner.get('pgid') != owner['pid']):
            raise RuntimeError('Native scratch owner exit is uncertain; preserve its scratch and fence.')
        ticks = owner.get('start_ticks')
        if (owned_process_alive(owner['pid'], ticks) or
                (not ticks and process_ticks(owner['pid']) is not None) or
                group_members(owner['pgid'])):
            raise RuntimeError('Native scratch owner or descendants remain alive; preserve its scratch and fence.')
    scratch = Path(_clear_native_tmp(appdata)['path'])
    if owner is not None:
        _remove_native_tmp_owner(fence, owner)
    return scratch


def _native_tmp_owner_file(appdata):
    fence = _native_tmp_directory(appdata).parent / 'native_tmp_owner.json'
    if fence.is_symlink():
        raise ValueError('Native scratch owner fence must not be a symlink.')
    return fence


def _remove_native_tmp_owner(fence, owner):
    if fence.is_symlink() or read_json(fence) != owner:
        raise RuntimeError('Native scratch owner fence does not match this process; preserve it.')
    fence.unlink()


def reserve_native_tmp(appdata, job_id):
    """Fence the launch window so a failed PID publication cannot expose scratch."""
    fence = _native_tmp_owner_file(appdata)
    owner = {'state': 'STARTING', 'job_id': job_id, 'token': uuid.uuid4().hex}
    if not publish_json(fence, owner):
        raise RuntimeError('Native scratch already has an owner fence; refuse another launch.')
    return owner


def record_native_tmp_owner(appdata, process, owner):
    """Persist native identity immediately after Popen, before queue heartbeats."""
    fence = _native_tmp_owner_file(appdata)
    if read_json(fence) != owner:
        raise RuntimeError('Native scratch launch fence changed; preserve existing ownership.')
    owner = {**owner, 'state': 'RUNNING', 'pid': process.pid,
        'start_ticks': process_ticks(process.pid), 'pgid': process.pid}
    write_json(fence, owner)
    return owner


def cleanup_native_tmp(appdata, process, *, owner=None):
    """Clear only private scratch after the intended child and its group exited."""
    if process is None or process.poll() is None or group_members(process.pid):
        raise RuntimeError('Native scratch cleanup requires confirmed process-group exit.')
    fence = _native_tmp_owner_file(appdata)
    if fence.exists() and (owner is None or owner.get('pid') != process.pid or read_json(fence) != owner):
        raise RuntimeError('Native scratch cleanup requires a matching process owner fence.')
    result = _clear_native_tmp(appdata)
    if fence.exists():
        _remove_native_tmp_owner(fence, owner)
    return result


def queue_retention_gpu(queue, gpu):
    """Apply a UUID-fenced, opt-in policy to this capture without changing inputs."""
    if gpu.get('retain_world'):
        return gpu  # Explicit --retain-world keeps its existing indefinite behavior.
    policy_path = queue.root / 'capture_retention.json'
    policy = read_json(policy_path, {})
    if (policy.get('state') != 'ENABLED' or policy.get('mode') != 'retain_last_world'
            or not gpu.get('uuid') or gpu['uuid'] not in policy.get('gpu_uuids', ())):
        return gpu
    return {**gpu, 'retain_world': True, 'queue_retention': {
        'mode': 'retain_last_world', 'policy_file': str(policy_path),
        'gpu_uuid': gpu['uuid'], 'close_trigger': 'eligible_queued_work',
        'retained_timeout_s': None, 'pre_completion_timeout_s': None,
        'pre_completion_timeout_and_cancel_unchanged': False, 'explicit_cancellation_unchanged': True,
        'explicit_worker_stop_and_failures_can_end_retention': True}}


def eligible_queued_job(queue, owner, *, job_id=None, candidate_prefix=None, pilot_control=None):
    """Read claim eligibility without claiming or releasing the retained GPU.

    Mirror CaptureQueue.claim's pause, runtime, sample and candidate fences. A
    paused pilot must have an explicit job selection, just as a normal claim does.
    This observation cannot reserve work against another supervisor's later claim.
    """
    selected_jobs = [job_id] if job_id else None
    if pilot_control:
        pilot = read_json(pilot_control, {})
        if pilot.get('mode') != 'campaign':
            if (pilot.get('mode') != 'pilot' or pilot.get('worker_id') != owner.get('id')
                    or pilot.get('gpu_uuid') != owner.get('uuid')):
                return None
            selected_jobs = pilot.get('job_ids', ())
    pause = read_json(queue.root / 'capture_pause.json', {})
    pilot = pause.get('authorized_pilot', {})
    pool = pause.get('authorized_gpu_pool', {})
    allowed_pool = (pool.get('state') == 'RUNNING' and owner.get('uuid') is not None
                    and owner.get('uuid') in pool.get('gpu_uuids', ()))
    if pause.get('state') == 'PAUSED_BY_USER' and not allowed_pool:
        if (selected_jobs is None or pilot.get('state') != 'RUNNING'
                or owner.get('id') != pilot.get('worker_id')
                or owner.get('uuid') != pilot.get('gpu_uuid')):
            return None
        selected_jobs = [job for job in selected_jobs if job in pilot.get('job_ids', ())]
    runtime = read_json(queue.root / 'capture_runtime_policy.json', {}).get('allowed_code_commits')
    sample_gate = read_json(queue.root / 'capture_sample_gate.json', {})
    paths = ((queue.root / 'queued').glob('*.json') if selected_jobs is None else
             [queue.root / 'queued' / f'{job}.json' for job in selected_jobs])
    for path in sorted(paths):
        if not path.is_file() or (queue.root / 'claimed' / path.name).exists():
            continue
        try:
            if (queue.status(path.stem) or {}).get('state') != 'QUEUED':
                continue
            manifest = queue.manifest(path.stem)  # Validate the accepted manifest, too.
        except (OSError, ValueError):
            continue
        if runtime and manifest.get('code_commit') not in runtime:
            continue
        if candidate_prefix and not manifest.get('candidate_id', '').startswith(candidate_prefix):
            continue
        if (sample_gate.get('state') == 'ONE_PER_CATEGORY_AWAITING_REVIEW'
                and manifest.get('candidate_id') not in sample_gate.get('selected_candidate_ids', ())):
            continue
        return path.stem
    return None


def capture_job(queue, claim, gpu, *, stopping=lambda: False, job_id=None,
                candidate_prefix=None, pilot_control=None):
    """Capture, then retain an opted-in world without an automatic residency timeout.

    Explicit worker stops/failures and pre-completion cancellation still apply.
    Queue continuation closes only through the native exact-snapshot request
    protocol; no next claim is acquired while this native process remains alive.
    """
    selected_job_id = job_id
    gpu = queue_retention_gpu(queue, gpu)
    job_id = claim["job_id"]
    try:
        manifest = queue.manifest(job_id)
    except (ValueError, OSError) as error:
        queue.transition(claim, "STALE", error=str(error), failure_stage="snapshot_validation", gpu=gpu)
        queue.release(claim)
        return
    attempt = queue.status(job_id)["attempts"]
    output = queue.job_dir(job_id) / "attempts" / f"{attempt:02d}"
    output.mkdir(parents=True)
    queue.heartbeat(claim, native_output=str(output))
    if (queue.job_dir(job_id) / "cancel.json").exists():
        queue.transition(claim, "CANCELLED")
        queue.release(claim)
        return
    try:
        verify_snapshot(manifest)
    except ValueError as error:
        queue.transition(claim, "STALE", error=str(error), failure_stage="snapshot_validation")
        queue.release(claim)
        return
    cached = queue.cached(manifest)
    if cached:
        write_json(output / "preview_complete.json", {**cached, "cache_hit": True})
        queue.transition(claim, "CACHED", receipt=str(output / "preview_complete.json"), gpu=gpu,
                         cache_hit=True, duration_s=0.)
        queue.release(claim)
        return
    command = [sys.executable, "-u", str(Path(manifest["code_root"]) / "src/wia/runtime/commands/capture_esi_scene_reviews.py"),
        "--capture-manifest", str(queue.job_dir(job_id) / "input/manifest.json"),
        "--capture-output", str(output), "--gpu-info", str(output / "gpu.json"),
        "--execution-mode", gpu.get("execution_mode", "pbs"),
        "--job-id", gpu.get("pbs_job_id") or job_id]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu["uuid"], OMNIGIBSON_GPU_ID="0",
                       CUDA_DEVICE_ORDER="PCI_BUS_ID", **frozen_runtime_environment(manifest["code_root"]),
                       PYTHONDONTWRITEBYTECODE="1", OMNIGIBSON_HEADLESS="1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = os.environ.get(name, "4")
    # Separate mutable application caches prevent two Kit instances corrupting them.
    appdata = Path(gpu.get("appdata_path", queue.root / "worker_caches" / claim["worker"]["id"]))
    native_tmp = prepare_native_tmp(appdata)
    appdata = native_tmp.parent
    gpu = {**gpu, 'appdata_path': str(appdata), 'native_tmp_path': str(native_tmp)}
    for name in ('TMPDIR', 'TMP', 'TEMP'):
        environment[name] = str(native_tmp)
    environment["OMNIGIBSON_APPDATA_PATH"] = str(appdata)
    environment["XDG_CACHE_HOME"] = str(appdata / "xdg_cache")
    environment["CUDA_CACHE_PATH"] = str(appdata / "cuda_cache")
    environment["__GL_SHADER_DISK_CACHE_PATH"] = str(appdata / "nvidia_shader_cache")
    if gpu.get("wait_for_capture"):
        environment["ESI_CAPTURE_RELEASE_FILE"] = str(output / "release_capture.json")
    if gpu.get("retain_world"):
        environment["ESI_CAPTURE_RETAIN_WORLD"] = "1"
    write_json(output / "gpu.json", gpu)
    write_json(output / "command.json", {"command": command, "code_commit": manifest["code_commit"],
        "input_manifest": str(queue.job_dir(job_id) / "input/manifest.json"), "gpu": gpu,
        "native_tmp_path": str(native_tmp),
        "temporary_environment": {name: environment[name] for name in ('TMPDIR', 'TMP', 'TEMP')}})
    began = time.monotonic()
    interrupted, timed_out, cancelled = False, False, False
    retained = False
    retention_close_request = None
    process = None
    native_tmp_owner = reserve_native_tmp(appdata, job_id)
    try:
        with (output / "native.log").open("w") as log:
            process = subprocess.Popen(command, cwd=manifest["code_root"], env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            native_tmp_owner = record_native_tmp_owner(appdata, process, native_tmp_owner)
            queue.heartbeat(claim, native_output=str(output), child_pid=process.pid,
                            child_pgid=process.pid, child_start_ticks=native_tmp_owner['start_ticks'])
            queue.transition(claim, "RUNNING", gpu=gpu, native_output=str(output), command=command,
                             started_at=time.time(), timeout_s=None)
            try:
                while process.poll() is None:
                    if not retained and gpu.get("retain_world"):
                        receipt = read_json(output / "preview_complete.json", {})
                        world = read_json(output / "retained_world.json", {})
                        if (world.get("state") == "RETAINED" and world.get("pid") == process.pid
                                and world.get("job_id") == job_id
                                and world.get("capture_signature") == manifest["capture_signature"]
                                and complete_images(manifest, receipt)):
                            receipt.update(process_running=True, native_pid=process.pid, world_retained=True)
                            write_json(output / "preview_complete.json", receipt)
                            queue.finish(claim, output / "preview_complete.json")
                            retained = True
                    if retained and gpu.get('queue_retention') and retention_close_request is None:
                        pending = eligible_queued_job(queue, {**claim['worker'], 'uuid': gpu['uuid']},
                            job_id=selected_job_id, candidate_prefix=candidate_prefix,
                            pilot_control=pilot_control)
                        world = read_json(output / 'retained_world.json', {})
                        if (pending and world.get('state') == 'RETAINED'
                                and world.get('pid') == process.pid and world.get('job_id') == job_id
                                and world.get('capture_signature') == manifest['capture_signature']):
                            retention_close_request = output / 'retained_requests' / f'queue_resume_{uuid.uuid4().hex}.json'
                            write_json(retention_close_request, {'operation': 'close', 'job_id': job_id,
                                'capture_signature': manifest['capture_signature'], 'native_pid': process.pid,
                                'reason': 'eligible_queued_work', 'pending_job_id': pending,
                                'requested_at': time.time()})
                            receipt.update(retention_close_request=str(retention_close_request),
                                           retention_close_reason='eligible_queued_work')
                            write_json(output / 'preview_complete.json', receipt)
                    if not retained:
                        queue.heartbeat(claim)
                    write_json(queue.root / "workers" / f"{claim['worker']['id']}.json", {
                        **claim["worker"], "state": "RETAINED" if retained else "RUNNING", "job_id": job_id,
                        "native_pid": process.pid, "native_output": str(output), "heartbeat": time.time()})
                    # Capture completes before residency begins. A held
                    # world retains GPU ownership without an outstanding queue claim.
                    cancelled = not retained and (queue.job_dir(job_id) / "cancel.json").exists()
                    interrupted = stopping()
                    if cancelled or interrupted:
                        stop_group(process.pid, process)
                        break
                    time.sleep(2)
                code = process.wait()
                # A successful parent must not leave descendants using its GPU.
                if group_members(process.pid):
                    stop_group(process.pid, process)
            except BaseException:
                stop_group(process.pid, process)
                raise
        elapsed = time.monotonic() - began
        receipt = read_json(output / "preview_complete.json", {})
        write_json(output / "process_exit.json", {"exit_code": code, "duration_s": elapsed,
            "timeout": timed_out, "cancelled": cancelled, "interrupted": interrupted, "gpu": gpu})
        if retained:
            receipt.update(process_running=False, native_exit_code=code, retention_ended_at=time.time())
            write_json(output / "preview_complete.json", receipt)
            return
        if code == 0 and not (timed_out or cancelled or interrupted) and complete_images(manifest, receipt):
            receipt.update(exit_code=code, duration_s=elapsed)
            receipt["timings"]["total_s"] = elapsed
            write_json(output / "preview_complete.json", receipt)
            queue.finish(claim, output / "preview_complete.json")
            return
        failure = read_json(output / "failure.json", {})
        error = ("Capture exceeded its configured timeout." if timed_out else
                 "Worker was interrupted." if interrupted else
                 failure.get("error", "Simulator failed or did not publish a complete screenshot set."))
        value = {"error": error, "failure_stage": failure.get("failure_stage", "native_process"),
            "exit_code": code, "duration_s": elapsed, "gpu": gpu, "native_log": str(output / "native.log"),
            "retryable": not (cancelled or interrupted) and (timed_out or failure.get("retryable", False)),
            "partial_images": {p.stem: str(p) for p in output.glob("*.png")}, "complete": False}
        write_json(output / "capture_failure.json", value)
        queue.transition(claim, "CANCELLED" if cancelled else "FAILED", **value)
        queue.release(claim)
        if value["retryable"] and attempt < manifest["max_attempts"]:
            queue.retry(job_id)
    except BaseException as error:
        # Keep the original exception/claim behavior and save diagnostics before scratch cleanup.
        try:
            write_json(output / "native_worker_error.json", {
                "error": f"{type(error).__name__}: {error}", "native_log": str(output / "native.log"),
                "native_tmp_path": str(native_tmp),
                "partial_images": {p.stem: str(p) for p in output.glob("*.png")}})
        except OSError:
            pass  # A diagnostic write must not replace the original native/worker error.
        raise
    finally:
        if process is not None:
            # Receipt/failure publication and closed logs above precede deletion. A live
            # retained world or uncertain descendant exit always keeps its scratch.
            cleanup = {"native_tmp_path": str(native_tmp), "timestamp": time.time()}
            try:
                cleanup.update(cleanup_native_tmp(appdata, process, owner=native_tmp_owner), status="cleaned")
            except Exception as error:
                cleanup.update(status="preserved", error=f"{type(error).__name__}: {error}")
            try:
                write_json(output / "native_tmp_cleanup.json", cleanup)
            except OSError:
                pass  # Cleanup evidence must not mask existing capture/error handling.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--gpu-slot", type=int, required=True)
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='Explicitly allow existing GPU clients when sufficient free memory remains.')
    parser.add_argument('--wait-for-idle-gpu', action='store_true',
                        help='Start a CPU-only fresh-process supervisor and wait for an idle GPU before claiming work.')
    parser.add_argument('--shared-gpu-min-free-mib', type=int, default=8192,
                        help='Required free memory before an explicitly shared worker starts.')
    parser.add_argument('--persistent-simulator', action='store_true',
                        help='Retain a native engine while rebuilding compatible screenshot worlds.')
    parser.add_argument('--persistent-max-jobs', type=int, default=0,
                        help='Optional explicit job limit; zero keeps a healthy native engine resident.')
    parser.add_argument('--cpu-preload-workers', type=int, choices=(0, 1, 2), default=0,
                        help='Optional CPU task/asset read-ahead workers; native scene construction stays serial.')
    parser.add_argument('--cpu-prepare-assets', action='store_true',
                        help='Use the CPU pool to decrypt/unpack assets before native main-thread construction.')
    parser.add_argument('--pilot-control', type=Path,
                        help='Bound a user-authorized pilot to listed job IDs; campaign mode still respects the queue pause.')
    parser.add_argument("--execution-mode", choices=("auto", "local", "pbs"), default="auto",
                        help="Direct local GPUs by default; retain PBS validation inside a PBS shell.")
    parser.add_argument("--renderer-order", choices=("auto", "pci", "vulkan"), default="auto",
                        help="Use the verified native renderer order: PCI for local, Vulkan for PBS.")
    parser.add_argument("--cache-root", type=Path,
                        help="Mutable worker cache location; defaults to the queue's worker_caches directory.")
    parser.add_argument("--wait-for-capture", action="store_true",
                        help="Retain the initialized native world until Main writes its release_capture.json.")
    parser.add_argument("--retain-world", action="store_true",
                        help="Publish the completed capture and keep its native world resident until explicit close.")
    parser.add_argument("--job-id", help="Only claim this accepted job, preserving an explicit scene-to-GPU mapping.")
    parser.add_argument('--candidate-prefix', help='Only claim matching candidates; preserves unrelated queued jobs.')
    parser.add_argument('--texture-cache-max-gib', type=float, default=0,
        help='Legacy metadata only; automatic disk thresholds and texture pruning are disabled.')
    parser.add_argument("--once", action="store_true", help="Exit after one job or an empty queue.")
    args = parser.parse_args(argv)
    if args.persistent_max_jobs < 0:
        parser.error('Persistent native job limit must be nonnegative.')
    if args.persistent_simulator and (args.retain_world or args.wait_for_capture):
        parser.error('Persistent queue capture requires automatic scene completion.')
    if args.cpu_preload_workers and not args.persistent_simulator:
        parser.error('CPU read-ahead requires the persistent simulator worker.')
    if args.cpu_prepare_assets and not args.cpu_preload_workers:
        parser.error('CPU asset preparation requires one or two CPU preload workers.')
    if args.pilot_control and not args.persistent_simulator:
        parser.error('The bounded pilot requires the persistent simulator worker.')
    if args.texture_cache_max_gib < 0:
        parser.error('Legacy texture cache metadata must be nonnegative.')
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.worker_id):
        parser.error("Use a filename-safe worker identity.")
    gpu = allocation_gpu(args.gpu_slot, args.execution_mode, args.renderer_order)
    gpu["wait_for_capture"] = args.wait_for_capture
    gpu["retain_world"] = args.retain_world
    gpu['texture_cache_max_bytes'] = int(args.texture_cache_max_gib * 1024 ** 3)
    gpu['automatic_disk_thresholds'] = False
    # All capture queues on this host share this local process-lifetime lock.
    local_lock = open(f"/tmp/esi_capture_gpu_{os.getuid()}_{gpu['uuid']}.lock", "a")
    try:
        fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another capture supervisor already owns this host GPU.")
    queue = CaptureQueue(args.queue_root)
    gpu["appdata_path"] = str((args.cache_root or queue.root / "worker_caches").resolve() / args.worker_id)
    queue.recover(confirm_stopped)
    contexts = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits"], text=True)
    occupied = [line for line in contexts.splitlines() if line.split(",")[1].strip() == gpu["uuid"]]
    if args.wait_for_idle_gpu and (args.persistent_simulator or args.allow_shared_gpu):
        parser.error('--wait-for-idle-gpu is only for exclusive fresh-process workers.')
    if occupied and not args.allow_shared_gpu and not args.wait_for_idle_gpu:
        parser.error("The assigned GPU already has native/CUDA processes; retain them and select a free owned slot.")
    if args.allow_shared_gpu:
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        available = {parts[0].strip(): int(parts[1].strip())
            for line in memory.splitlines() if (parts := line.split(','))}
        if available.get(gpu['uuid'], 0) < args.shared_gpu_min_free_mib:
            parser.error('The shared GPU does not have the explicitly required free memory.')
        gpu.update(shared_gpu=True, coexisting_compute_processes=occupied,
                   shared_gpu_min_free_mib=args.shared_gpu_min_free_mib)
    key = f"{socket.gethostname()}_{gpu['uuid']}"
    lock = queue.root / "gpu_locks" / f"{key}.json"
    owner = {"id": args.worker_id, **gpu, "supervisor_pid": os.getpid(),
             "supervisor_start_ticks": process_ticks(os.getpid()), "started_at": time.time()}
    if not publish_json(lock, owner):
        previous = read_json(lock)
        if owned_process_alive(previous["supervisor_pid"], previous.get("supervisor_start_ticks")):
            parser.error("This GPU already has a live capture supervisor.")
        # Recovery verified every expired child before the lock can be reused.
        outstanding = [read_json(p) for p in (queue.root / "claimed").glob("*.json")]
        if any(c.get("worker", {}).get("uuid") == gpu["uuid"] for c in outstanding):
            parser.error("This GPU still has an unrecovered capture claim.")
        lock.unlink()
        if not publish_json(lock, owner):
            parser.error("Another supervisor acquired this GPU.")
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *args: stop.__setitem__(0, True))
    status_path = queue.root / "workers" / f"{args.worker_id}.json"
    resident = None
    if args.persistent_simulator:
        from wia.runtime.scene_capture_resident import ResidentCapture
        resident = ResidentCapture(queue, gpu, owner, lambda: stop[0], sys.modules[__name__],
                                   max_jobs=args.persistent_max_jobs, preload_workers=args.cpu_preload_workers,
                                   prepare_assets=args.cpu_prepare_assets)
    try:
        last_recovery = time.monotonic()
        while not stop[0]:
            if resident and not resident.resume_shutdown():
                write_json(status_path, {**owner, **resident.worker_state(), 'heartbeat': time.time()})
                time.sleep(2)
                continue
            if time.monotonic() - last_recovery >= 30:
                queue.recover(confirm_stopped)
                last_recovery = time.monotonic()
            write_json(status_path, {**owner, **(resident.worker_state() if resident else {'state': 'IDLE'}),
                                     "heartbeat": time.time()})
            if resident is None and not fresh_gpu_available(gpu):
                write_json(status_path, {**owner, 'state': 'WAITING_FOR_IDLE_GPU', 'heartbeat': time.time()})
                time.sleep(10)
                continue
            pilot = read_json(args.pilot_control, {}) if args.pilot_control else None
            if pilot is not None and pilot.get('mode') != 'campaign':
                if (pilot.get('mode') != 'pilot' or pilot.get('worker_id') != args.worker_id
                        or pilot.get('gpu_uuid') != gpu['uuid']):
                    raise ValueError('Pilot control must identify this exact worker and GPU.')
                resident.allowed_job_ids = list(pilot['job_ids'])
                selected_job = next((job for job in resident.allowed_job_ids
                    if (queue.status(job) or {}).get('state') == 'QUEUED'), None)
                if selected_job is None:
                    time.sleep(2)
                    continue
            else:
                if resident:
                    resident.allowed_job_ids = None
                selected_job = args.job_id or (resident.next_job_id() if resident else None)
            claim = queue.claim(owner, job_id=selected_job, candidate_prefix=args.candidate_prefix)
            if claim is None:
                if args.once:
                    break
                time.sleep(2)
                continue
            write_json(status_path, {**owner, "state": "RUNNING", "job_id": claim["job_id"], "heartbeat": time.time()})
            try:
                if resident and queue.manifest(claim['job_id']).get('diagnostic_pending'):
                    resident.capture(claim)
                else:
                    if resident:
                        if not resident.close():
                            queue.transition(claim, 'FAILED', retryable=True,
                                failure_stage='resident_shutdown',
                                error='Previous resident process is still exiting; native loading deferred.')
                            queue.release(claim)
                            queue.retry(claim['job_id'])
                            continue
                    capture_job(queue, claim, gpu, stopping=lambda: stop[0], job_id=args.job_id,
                                candidate_prefix=args.candidate_prefix, pilot_control=args.pilot_control)
            except Exception as error:
                # Preserve this claim if owned-process termination was uncertain.
                write_json(status_path, {**owner, "state": "ERROR", "error": str(error),
                                       "job_id": claim["job_id"], "heartbeat": time.time()})
                raise
            print(f"[capture-worker] {args.worker_id} {claim['job_id']} {queue.status(claim['job_id'])['state']}", flush=True)
            if args.once:
                break
    finally:
        if resident:
            resident.close()
            resident.close_preloader()
        if not any(read_json(p, {}).get("worker", {}).get("id") == args.worker_id
                   for p in (queue.root / "claimed").glob("*.json")):
            lock.unlink(missing_ok=True)
        write_json(status_path, {**owner, "state": "STOPPED", "heartbeat": time.time()})
        local_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plan final review workers; start them only with explicit --apply.

Read-only plan, including live resource checks and exact launch scripts:
  python scripts/launch_wia_review_captures.py --gpus 0 1 2 3 --dry-run
After review, replace --dry-run with --apply. Physical GPUs 0--3 retain reusable
supervisors; idle GPUs 4--7 run one job with --once and release their processes.
Existing tasks are never stopped or replaced. Each retained codex-gpu
tmux window runs one idle-capable supervisor and a fresh native child per job;
no persistent simulator, CPU preload, agent action, or queue preparation occurs.
Durable scripts/logs/manifest go to campaign/final_refresh_launches. Mutable
caches use /dev/shm/wia_final_capture_<uid>/<unique-worker-id>. Dry-run writes
nothing. Apply rechecks GPU utilization/memory/clients and remaining resource
budgets immediately before each launch, without reserving unused memory.
--allow-shared-gpu explicitly permits coexistence with non-capture clients when
measured free GPU memory meets --shared-gpu-min-free-mib (default 14336 MiB).
An existing live capture supervisor always prevents another launch.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "outputs/edge_review_20261029"
QUEUE = CAMPAIGN / "queue"
CONDA_ROOT = Path("/local_home2/zhouxinyang/miniconda3")
BEHAVIOR_ENV = CONDA_ROOT / "envs/behavior"
BEHAVIOR_PYTHON = BEHAVIOR_ENV / "bin/python"
RUNTIME_LIBS = Path("/local_home2/zhouxinyang/runtime_libs")
GIB = 1024 ** 3
MIN_SHM_PER_WORKER = 12 * GIB
MIN_RAM_PER_WORKER = 24 * GIB  # Includes the shared-memory cache plus native process headroom.
# This host retains about 87 MiB on an idle GPU 0 after all native clients exit.
# Compute clients and utilization are independently required to be absent/zero.
MAX_IDLE_MEMORY_MIB = 512
DEFAULT_SHARED_GPU_MIN_FREE_MIB = 14336
SESSION = "codex-gpu"


def _run(command, *, check=True):
    result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    if check and result.returncode:
        raise RuntimeError(shlex.join(command) + ": " + (result.stderr.strip() or result.stdout.strip()))
    return result


def _csv_rows(text):
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(text)) if row]


def gpu_snapshot(indices):
    """Read utilization, memory, and compute clients; unknown measurements fail closed."""
    inventory = _run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used,memory.total",
                      "--format=csv,noheader,nounits"]).stdout
    clients = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits"]).stdout
    processes = []
    for row in _csv_rows(clients):
        if len(row) != 4:
            raise RuntimeError("Cannot parse NVIDIA compute client measurements.")
        processes.append({"gpu_uuid": row[0], "pid": int(row[1]), "process_name": row[2], "used_memory_mib": row[3]})
    devices = {}
    for row in _csv_rows(inventory):
        if len(row) != 6:
            raise RuntimeError("Cannot parse NVIDIA GPU measurements.")
        index = int(row[0])
        if index not in indices:
            continue
        devices[index] = {"physical_index": index, "uuid": row[1], "pci_bus_id": row[2],
            "utilization_percent": int(row[3]), "memory_used_mib": int(row[4]), "memory_total_mib": int(row[5]),
            "compute_clients": [process for process in processes if process["gpu_uuid"] == row[1]]}
    if set(devices) != set(indices) or len({device["uuid"] for device in devices.values()}) != len(indices):
        raise RuntimeError("Requested GPU indices do not have unique measured UUIDs.")
    return devices


def _process_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _live_supervisors(gpu_uuid):
    owners = []
    for path in (QUEUE / "gpu_locks").glob("*.json"):
        owner = json.loads(path.read_text())
        if owner.get("uuid") != gpu_uuid:
            continue
        ticks = _process_ticks(owner.get("supervisor_pid"))
        if ticks is not None and str(owner.get("supervisor_start_ticks")) == ticks:
            owners.append({"worker_id": owner.get("id"), "pid": owner["supervisor_pid"], "lock_file": str(path)})
    return owners


def _ram():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key == "MemAvailable":
            memory["host_mem_available_bytes"] = int(value.split()[0]) * 1024
    if "host_mem_available_bytes" not in memory:
        raise RuntimeError("Host MemAvailable is unavailable.")
    available = memory["host_mem_available_bytes"]
    for limit_path, usage_path in ((Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
                                   (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))):
        if not limit_path.is_file() or not usage_path.is_file():
            continue
        limit = limit_path.read_text().strip()
        if limit != "max" and int(limit) < 2 ** 60:
            remaining = max(0, int(limit) - int(usage_path.read_text()))
            memory.update(cgroup_limit_file=str(limit_path), cgroup_remaining_bytes=remaining)
            available = min(available, remaining)
        break
    memory["effective_available_bytes"] = available
    return memory


def resources(worker_count):
    cache = Path(f"/dev/shm/wia_final_capture_{os.getuid()}")
    shm = shutil.disk_usage("/dev/shm")
    ram = _ram()
    errors = []
    if shm.free < worker_count * MIN_SHM_PER_WORKER:
        errors.append("Shared memory has less than 12 GiB free per remaining requested worker.")
    if ram["effective_available_bytes"] < worker_count * MIN_RAM_PER_WORKER:
        errors.append("Effective available RAM has less than 24 GiB per remaining requested worker.")
    if not os.access("/dev/shm", os.W_OK | os.X_OK):
        errors.append("/dev/shm is not writable/searchable.")
    if cache.exists() and (not cache.is_dir() or cache.is_symlink() or cache.stat().st_uid != os.getuid()
                           or not os.access(cache, os.W_OK | os.X_OK)):
        errors.append("The existing UID cache root is not an owned writable real directory.")
    durable = shutil.disk_usage(CAMPAIGN)
    if durable.free < GIB:
        errors.append("Campaign filesystem has less than 1 GiB free for durable launcher logs.")
    if not os.access(CAMPAIGN, os.W_OK | os.X_OK):
        errors.append("Campaign directory is not writable/searchable.")
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "remaining_worker_count": worker_count,
            "cache_root": str(cache), "shm_free_bytes": shm.free, "ram": ram,
            "minimum_shm_bytes": worker_count * MIN_SHM_PER_WORKER, "minimum_ram_bytes": worker_count * MIN_RAM_PER_WORKER,
            "durable_free_bytes": durable.free, "cpu_count": os.cpu_count(), "cpu_load_1_5_15": list(os.getloadavg()),
            "errors": errors}


def positive_free_mib(value):
    """Parse an explicitly positive shared-GPU memory budget."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Shared GPU minimum free memory must be a positive integer MiB value.") from error
    if isinstance(value, bool) or parsed <= 0 or isinstance(value, float) and parsed != value:
        raise argparse.ArgumentTypeError("Shared GPU minimum free memory must be a positive integer MiB value.")
    return parsed


def gpu_authorization(device, *, allow_shared_gpu=False,
                      shared_gpu_min_free_mib=DEFAULT_SHARED_GPU_MIN_FREE_MIB):
    """Record explicit coexistence authorization and measured device headroom."""
    minimum = positive_free_mib(shared_gpu_min_free_mib)
    return {"mode": "shared_with_headroom" if allow_shared_gpu else "exclusive_idle",
            "allow_shared_gpu": bool(allow_shared_gpu), "shared_gpu_min_free_mib": minimum,
            "measured_free_memory_mib": device["memory_total_mib"] - device["memory_used_mib"],
            "coexisting_compute_clients": [dict(client) for client in device["compute_clients"]]}


def device_errors(device, *, allow_shared_gpu=False,
                  shared_gpu_min_free_mib=DEFAULT_SHARED_GPU_MIN_FREE_MIB):
    errors = []
    minimum = positive_free_mib(shared_gpu_min_free_mib)
    if allow_shared_gpu:
        used, total = device["memory_used_mib"], device["memory_total_mib"]
        if (type(used) is not int or type(total) is not int or not 0 <= used <= total or total <= 0):
            errors.append("Shared GPU memory measurements are unavailable or invalid.")
        elif total - used < minimum:
            errors.append(f"Shared GPU has less than the required {minimum} MiB measured free memory.")
    else:
        if device["compute_clients"]:
            errors.append("GPU has existing compute clients.")
        if device["utilization_percent"] != 0:
            errors.append("GPU utilization is nonzero.")
        if device["memory_used_mib"] > MAX_IDLE_MEMORY_MIB:
            errors.append(f"GPU memory exceeds the {MAX_IDLE_MEMORY_MIB} MiB driver-only idle threshold.")
    owners = _live_supervisors(device["uuid"])
    if owners:
        errors.append("A live capture supervisor already owns this GPU.")
    return errors, owners


def worker_plan(device, launch_id, *, allow_shared_gpu=False,
                shared_gpu_min_free_mib=DEFAULT_SHARED_GPU_MIN_FREE_MIB):
    authorization = gpu_authorization(device, allow_shared_gpu=allow_shared_gpu,
                                      shared_gpu_min_free_mib=shared_gpu_min_free_mib)
    name = f"final-capture-{launch_id}-gpu{device['physical_index']}"
    folder = CAMPAIGN / "final_refresh_launches" / launch_id / name
    cache_root = Path(f"/dev/shm/wia_final_capture_{os.getuid()}")
    cache = cache_root / name
    env = {"CUDA_VISIBLE_DEVICES": device["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMNIGIBSON_GPU_ID": "0",
        "OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "OMNIGIBSON_DATA_PATH": "/local_home2/zhouxinyang/data/behavior",
        "LD_LIBRARY_PATH": str(RUNTIME_LIBS), "VK_ICD_FILENAMES": str(ROOT / "configs/esi_vulkan_egl.json"),
        "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT.parent / "Show-Harness-Explore"), "WIA_ROOT": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2", "OMNIGIBSON_APPDATA_PATH": str(cache), "TMPDIR": str(cache / "native_tmp"),
        "TMP": str(cache / "native_tmp"), "TEMP": str(cache / "native_tmp"), "XDG_CACHE_HOME": str(cache / "xdg_cache"),
        "CUDA_CACHE_PATH": str(cache / "cuda_cache"), "__GL_SHADER_DISK_CACHE_PATH": str(cache / "nvidia_shader_cache")}
    command = [str(BEHAVIOR_PYTHON), "-u", "-B", "-m", "wia.runtime.commands.run_esi_capture_worker",
        "--queue-root", str(QUEUE), "--worker-id", name, "--gpu-slot", "0", "--execution-mode", "local",
        "--renderer-order", "pci", "--cpu-preload-workers", "0", "--cache-root", str(cache_root)]
    temporary_device = device["physical_index"] >= 4
    if allow_shared_gpu:
        command.extend(["--allow-shared-gpu", "--shared-gpu-min-free-mib", str(authorization["shared_gpu_min_free_mib"])])
    if temporary_device:
        command.append("--once")
    script = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail",
        "exec >> " + shlex.quote(str(folder / "worker.log")) + " 2>&1",
        "printf '%s\\n' \"$$\" > " + shlex.quote(str(folder / "worker.pid")),
        "source " + shlex.quote(str(CONDA_ROOT / "etc/profile.d/conda.sh")),
        "conda activate " + shlex.quote(str(BEHAVIOR_ENV)),
        *["export " + key + "=" + shlex.quote(value) for key, value in env.items()],
        "mkdir -p -- " + shlex.join([str(cache / "native_tmp"), str(cache / "xdg_cache"), str(cache / "cuda_cache")]),
        "cd -- " + shlex.quote(str(ROOT)), "exec " + shlex.join(command), ""])
    errors, owners = device_errors(device, allow_shared_gpu=allow_shared_gpu,
                                   shared_gpu_min_free_mib=shared_gpu_min_free_mib)
    return {"worker_id": name, "gpu": device, "existing_supervisors": owners, "errors": errors,
        "gpu_authorization": authorization,
        "environment": env, "command": command, "launch_script": str(folder / "launch.sh"), "script": script,
        "worker_log": str(folder / "worker.log"), "worker_pid_file": str(folder / "worker.pid"),
        "exit_code_file": str(folder / "exit_code"), "cache_directory": str(cache),
        "queue_worker_status_file": str(QUEUE / "workers" / (name + ".json")),
        "temporary_device": temporary_device,
        "native_lifecycle": ("one job; native child and supervisor exit and release their GPU lock" if temporary_device else
                             "fresh process per claimed job; native child exits; supervisor idles on empty queue")}


def build_plan(indices, *, allow_shared_gpu=False,
               shared_gpu_min_free_mib=DEFAULT_SHARED_GPU_MIN_FREE_MIB):
    shared_gpu_min_free_mib = positive_free_mib(shared_gpu_min_free_mib)
    if not indices or len(set(indices)) != len(indices) or any(index not in range(8) for index in indices):
        raise ValueError("Explicit distinct physical GPU indices 0--7 are required.")
    errors = []
    required = [BEHAVIOR_PYTHON, CONDA_ROOT / "etc/profile.d/conda.sh", ROOT / "configs/esi_vulkan_egl.json",
                QUEUE / "category_storage.json", QUEUE / "capture_runtime_policy.json"]
    for path in required:
        if not path.is_file():
            errors.append("Required launch input is missing: " + str(path))
    if not RUNTIME_LIBS.is_dir() or not os.access(BEHAVIOR_PYTHON, os.X_OK):
        errors.append("Behavior Python or required runtime library directory is unavailable.")
    if not shutil.which("tmux"):
        errors.append("tmux is unavailable.")
    launch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    workers = []
    try:
        devices = gpu_snapshot(indices)
        workers = [worker_plan(devices[index], launch_id, allow_shared_gpu=allow_shared_gpu,
                               shared_gpu_min_free_mib=shared_gpu_min_free_mib) for index in indices]
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        errors.append("GPU measurements unavailable: " + str(error))
    try:
        budget = resources(len(indices))
        errors.extend(budget["errors"])
    except (OSError, ValueError, RuntimeError) as error:
        budget = {"error": str(error)}
        errors.append("Resource measurements unavailable: " + str(error))
    return {"schema": "wia_final_capture_worker_launch_v1", "mode": "dry_run", "launch_id": launch_id,
        "launcher_pid": os.getpid(), "hostname": socket.gethostname(), "uid": os.getuid(), "campaign": str(CAMPAIGN),
        "queue_root": str(QUEUE), "tmux_session": SESSION, "requested_gpus": list(indices),
        "allow_shared_gpu": bool(allow_shared_gpu), "shared_gpu_min_free_mib": shared_gpu_min_free_mib,
        "gpu_authorization_mode": "shared_with_headroom" if allow_shared_gpu else "exclusive_idle",
        "manifest_file": str(CAMPAIGN / "final_refresh_launches" / launch_id / "manifest.json"),
        "resource_check": budget, "workers": workers, "errors": errors,
        "ready": not errors and len(workers) == len(indices) and not any(worker["errors"] for worker in workers),
        "workers_started": False, "queue_mutated_by_launcher": False, "robot_actions": 0, "model_calls": 0}


def _write_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def apply_plan(plan, *, allow_shared_gpu=None, shared_gpu_min_free_mib=None):
    planned_sharing = plan.get("allow_shared_gpu", False)
    planned_minimum = positive_free_mib(plan.get("shared_gpu_min_free_mib", DEFAULT_SHARED_GPU_MIN_FREE_MIB))
    if allow_shared_gpu is not None and allow_shared_gpu != planned_sharing:
        raise ValueError("Apply must retain the exact GPU sharing authorization reviewed in the plan.")
    if shared_gpu_min_free_mib is not None and positive_free_mib(shared_gpu_min_free_mib) != planned_minimum:
        raise ValueError("Apply must retain the exact shared GPU memory threshold reviewed in the plan.")
    if not plan["ready"]:
        raise RuntimeError("Launch refused: review the measured GPU/resource errors in dry-run.")
    manifest_path = Path(plan["manifest_file"])
    manifest_path.parent.mkdir(parents=True, exist_ok=False)
    plan.update(mode="apply", state="preparing", workers_started=False)
    _write_json(manifest_path, plan)
    try:
        cache_root = Path(plan["resource_check"]["cache_root"])
        cache_root.mkdir(mode=0o700, exist_ok=True)
        if _run(["tmux", "has-session", "-t", SESSION], check=False).returncode:
            _run(["tmux", "new-session", "-d", "-s", SESSION, "-n", "capture-control"])
        for index, worker in enumerate(plan["workers"]):
            folder = Path(worker["launch_script"]).parent
            folder.mkdir(parents=True, exist_ok=False)
            with Path(worker["launch_script"]).open("x") as handle:
                handle.write(worker["script"])
            os.chmod(worker["launch_script"], 0o700)
            budget = resources(len(plan["workers"]) - index)
            fresh = gpu_snapshot([worker["gpu"]["physical_index"]])[worker["gpu"]["physical_index"]]
            errors, owners = device_errors(fresh, allow_shared_gpu=planned_sharing,
                                           shared_gpu_min_free_mib=planned_minimum)
            if fresh["uuid"] != worker["gpu"]["uuid"]:
                errors.append("GPU UUID mapping changed since planning.")
            if fresh.get("pci_bus_id") != worker["gpu"].get("pci_bus_id"):
                errors.append("GPU PCI bus identity changed since planning.")
            authorization = gpu_authorization(fresh, allow_shared_gpu=planned_sharing,
                                              shared_gpu_min_free_mib=planned_minimum)
            if errors or budget["errors"]:
                worker.update(launch_state="refused_on_immediate_recheck", errors=errors + budget["errors"],
                              immediate_gpu_check=fresh, immediate_resource_check=budget, existing_supervisors=owners)
                worker["immediate_gpu_authorization"] = authorization
                raise RuntimeError("Immediate prelaunch checks refused worker " + worker["worker_id"])
            worker.update(immediate_gpu_check=fresh, immediate_resource_check=budget,
                          immediate_gpu_authorization=authorization,
                          requested_at_utc=datetime.now(timezone.utc).isoformat())
            # The outer pane shell survives worker exit; no existing terminal is replaced.
            shell_command = (shlex.join(["bash", worker["launch_script"]]) + "; capture_exit_code=$?; "
                "printf '%s\\n' \"$capture_exit_code\" > " + shlex.quote(worker["exit_code_file"]) + "; exec bash")
            response = _run(["tmux", "new-window", "-d", "-P", "-F", "#{window_id}\t#{pane_id}\t#{pane_pid}",
                             "-t", SESSION, "-n", worker["worker_id"], shell_command]).stdout.strip().split("\t")
            plan["workers_started"] = True
            worker.update(launch_state="tmux_new_window_acknowledged", tmux_response=response,
                          startup_health="not_certified_by_launcher")
            _write_json(manifest_path, plan)
            if len(response) != 3:
                raise RuntimeError("tmux did not return exact window/pane/process metadata.")
            worker.update(window_id=response[0], pane_id=response[1], pane_pid=int(response[2]),
                          launch_state="tmux_window_created", launched_at_utc=datetime.now(timezone.utc).isoformat())
            _write_json(manifest_path, plan)
            deadline = time.monotonic() + 2.
            while not Path(worker["worker_pid_file"]).is_file() and time.monotonic() < deadline:
                time.sleep(.05)
            if Path(worker["worker_pid_file"]).is_file():
                worker["worker_script_pid"] = int(Path(worker["worker_pid_file"]).read_text().strip())
                worker["worker_start_ticks"] = _process_ticks(worker["worker_script_pid"])
                worker["pid_semantics"] = "script PID retained by exec into the capture supervisor"
            _write_json(manifest_path, plan)
        plan["state"] = "launched"
        _write_json(manifest_path, plan)
        return plan
    except Exception as error:
        plan.update(state="launch_incomplete", error=f"{type(error).__name__}: {error}")
        _write_json(manifest_path, plan)
        raise RuntimeError("Launch stopped; already started workers were retained. Manifest: " + str(manifest_path)) from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", choices=range(8), required=True)
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Explicitly allow non-capture clients when measured GPU headroom is sufficient.")
    parser.add_argument("--shared-gpu-min-free-mib", type=positive_free_mib,
                        default=DEFAULT_SHARED_GPU_MIN_FREE_MIB)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Read-only plan; the default.")
    mode.add_argument("--apply", action="store_true", help="Start explicitly requested GPUs after fresh authorization and resource checks.")
    args = parser.parse_args(argv)
    plan = build_plan(args.gpus, allow_shared_gpu=args.allow_shared_gpu,
                      shared_gpu_min_free_mib=args.shared_gpu_min_free_mib)
    print(json.dumps(apply_plan(plan, allow_shared_gpu=args.allow_shared_gpu,
                               shared_gpu_min_free_mib=args.shared_gpu_min_free_mib)
                     if args.apply else plan, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sequential, isolated preflight batches. Run src/wia/runtime/commands/preflight_esi_tasks.py --all --mode static."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import threading
import traceback
import uuid

import yaml

from wia.tasks.loader import ROOT, SCENE_ROOT, TASK_ROOT, UniqueLoader, discover_tasks
from wia.runtime.preflight.models import Check, GROUPS, Options
from wia.runtime.preflight.report import atomic_json, emit_check, environment, fingerprints, refresh_report, utc_now


def _raw(path):
    text = Path(path).read_text(encoding="utf-8")
    value = json.loads(text) if Path(path).suffix.lower() == ".json" else yaml.load(text, Loader=UniqueLoader)
    return value if isinstance(value, dict) else {}


def discover_selections(task_files=(), *, all_tasks=False):
    """Keep malformed registered inputs so they receive explicit contract reports."""
    selected, skipped, warnings = set(), [], []
    catalog_status = {}
    if all_tasks:
        catalog = TASK_ROOT / "catalog.json"
        if catalog.is_file():
            try:
                for item in _raw(catalog).get("tasks", []):
                    if not item.get("task_file"):
                        warnings.append({"path": str(catalog), "reason": "Catalog entry has no task_file.", "entry": item})
                        continue
                    path = (TASK_ROOT / item["task_file"]).resolve()
                    selected.add(path)
                    catalog_status[path] = item.get("status", "ready")
            except Exception as error:
                warnings.append({"path": str(catalog), "reason": f"Cannot read task catalog: {type(error).__name__}: {error}"})
        selected.update(Path(path).resolve() for path in discover_tasks(TASK_ROOT))
        scenes_catalog = SCENE_ROOT / "catalog.json"
        if scenes_catalog.is_file():
            try:
                for item in _raw(scenes_catalog).get("scenes", []):
                    if item.get("status") in ("pending", "discarded"):
                        skipped.append({"kind": "scene", "path": str((SCENE_ROOT / item["scene_file"]).resolve()),
                                        "status": item["status"], "reason": "Scene is not ready for preflight execution.",
                                        "conversion": item.get("conversion")})
            except Exception as error:
                warnings.append({"path": str(scenes_catalog), "reason": f"Cannot read scene catalog: {type(error).__name__}: {error}"})
    selected.update(Path(path).expanduser().resolve() for path in task_files)
    ready = []
    for path in sorted(selected):
        status = catalog_status.get(path, "ready")
        try:
            status = _raw(path).get("status", status)
        except Exception:
            pass  # Invalid documents must reach the static validator.
        if status in ("pending", "discarded"):
            skipped.append({"kind": "task", "path": str(path), "status": status,
                            "reason": f"Input is {status} and is not eligible for preflight execution."})
        else:
            ready.append(path)
    return ready, skipped, warnings


def _required(context):
    return list(context.required_checks) if context is not None else ["static"]


def _verify(context, report, report_path):
    try:
        context.loaded.verify_unchanged()
        for item in report["fingerprints"]["files"]:
            if item.get("sha256"):
                path = Path(item["path"])
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                    raise RuntimeError("Input or configuration changed: " + str(path))
    except Exception as error:
        emit_check(report, Check("static", "input_immutability", "FAIL",
                                 f"Input changed during diagnostics: {type(error).__name__}: {error}",
                                 recommended_action="Restore the reviewed input or rerun against the new version."), report_path)
        return False
    return True


def _load_context(path, options):
    from wia.runtime.preflight.adapters import load_context
    return load_context(path, variant=options.variant, seed=options.seed, data_root=options.data_root)


def _make_report(path, options, context, output, started):
    identity = context.identity() if context is not None else {"task_id": path.stem, "task_file": str(path),
                                                             "scene_file": None, "variant": options.variant,
                                                             "seed": options.seed}
    return {"schema": "agent_preflight_v1", "private_diagnostic": True,
            "agent_calls": 0, "automatic_repairs": False,
            **identity, "task_file": str(path), "options": options.to_dict(),
            "required_checks": _required(context), "checks": [],
            "fingerprints": fingerprints(context, options, path), "environment": environment(),
            "started_at_utc": started, "output_dir": str(output),
            "log_paths": {"worker": str(output / "worker.log"), "diagnostic": str(output / "diagnostic.log")},
            "timing": {}, "worker": {"started": False}}


def _allocation_problem():
    if not os.environ.get("PBS_JOBID"):
        return "Full preflight requires the existing interactive PBS compute allocation (PBS_JOBID is absent)."
    if os.environ.get("PBS_ENVIRONMENT", "").upper() == "PBS_BATCH":
        return "Full preflight requires an interactive allocation; PBS_ENVIRONMENT identifies a batch job."
    return None


def _terminate_worker(process, grace_s=5.):
    """Never touch the allocation or another task's process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(.001, grace_s * .6))
    except subprocess.TimeoutExpired:
        pass
    # A worker can exit while a simulator descendant ignores SIGTERM. Kill the
    # entire owned group even when the immediate child's wait already returned.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(.001, grace_s * .4))


class StaticDeadline(BaseException):
    """The readonly parent phase exhausted the per-task deadline."""


@contextmanager
def static_deadline(seconds):
    # The CLI runs in the main thread on Linux; library callers on other threads
    # still subtract measured static time from the worker budget.
    enabled = threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")
    if not enabled:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expire(signum, frame):
        raise StaticDeadline("Static validation wall budget exhausted.")
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def run_one(path: Path, options: Options, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "preflight.json"
    start, started = time.monotonic(), utc_now()
    context, report = None, None
    try:
        with static_deadline(options.wall_budget_s):
            context, load_error = None, None
            try:
                context = _load_context(path, options)
            except StaticDeadline:
                raise
            except Exception as error:
                load_error = error
            report = _make_report(path, options, context, output, started)
            atomic_json(report_path, refresh_report(report))
            if load_error is not None:
                (output / "diagnostic.log").write_text("".join(traceback.format_exception(load_error)), encoding="utf-8")
                known_contract = isinstance(load_error, (ValueError, FileNotFoundError, yaml.YAMLError))
                emit_check(report, Check("static", "load_contract", "FAIL" if known_contract else "INCONCLUSIVE",
                                         f"Cannot load input: {type(load_error).__name__}: {load_error}",
                                         recommended_action="Review the task schema, bindings, and referenced files."), report_path)
            else:
                try:
                    from wia.runtime.preflight.static import run_static
                    _verify(context, report, report_path)
                    for check in run_static(context, options):
                        emit_check(report, check, report_path)
                except StaticDeadline:
                    raise
                except Exception as error:
                    (output / "diagnostic.log").write_text(traceback.format_exc(), encoding="utf-8")
                    emit_check(report, Check("static", "validator_exception", "INCONCLUSIVE",
                                             f"Static validator raised {type(error).__name__}: {error}"), report_path)
    except StaticDeadline:
        if report is None:
            report = {"schema": "agent_preflight_v1", "private_diagnostic": True, "task_id": path.stem,
                      "task_file": str(path), "scene_file": None, "variant": options.variant, "seed": options.seed,
                      "options": options.to_dict(), "required_checks": _required(context), "checks": [],
                      "started_at_utc": started, "fingerprints": {"files": [], "incomplete": True},
                      "environment": environment(), "timing": {}, "worker": {"started": False},
                      "log_paths": {"diagnostic": str(output / "diagnostic.log"), "worker": str(output / "worker.log")}}
        emit_check(report, Check("static", "wall_budget", "INCONCLUSIVE",
                                 "Static validation exhausted the task wall budget; unfinished checks retain INCONCLUSIVE."), report_path)
    report["timing"]["static_s"] = time.monotonic() - start
    static_fail = any(item["status"] == "FAIL" and item.get("required", True) for item in report["checks"])
    remaining = options.wall_budget_s - (time.monotonic() - start)
    if remaining <= 0:
        emit_check(report, Check("static", "wall_budget", "INCONCLUSIVE", "The wall budget was exhausted during static validation."), report_path)
    elif context is not None and not static_fail and options.mode == "full":
        allocation_problem = _allocation_problem()
        if allocation_problem:
            emit_check(report, Check("physics", "interactive_allocation", "INCONCLUSIVE", allocation_problem,
                                     recommended_action="Run this command in the existing 12-hour interactive GPU allocation."), report_path)
        else:
            request = output / "worker_request.json"
            atomic_json(request, {"task_file": str(path), "options": options.to_dict(), "report": str(report_path)})
            command = [sys.executable, str(ROOT / "src/wia/runtime/commands/preflight_esi_tasks.py"), "--worker-request", str(request)]
            report["worker"] = {"started": True, "command": command}
            atomic_json(report_path, report)
            worker_start = time.monotonic()
            process = None
            timed_out = False
            try:
                with (output / "worker.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    try:
                        available = max(.001, options.wall_budget_s - (time.monotonic() - start))
                        cleanup_grace = min(5., available * .1)
                        process.wait(timeout=max(.001, available - cleanup_grace))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        _terminate_worker(process, cleanup_grace)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["worker"].update({"pid": process.pid, "returncode": process.returncode, "timed_out": timed_out})
                report["timing"]["worker_s"] = time.monotonic() - worker_start
                if timed_out:
                    emit_check(report, Check("physics", "worker_timeout", "INCONCLUSIVE",
                                             "Worker exhausted the task wall budget; its process group was terminated and partial evidence retained.",
                                             evidence={"wall_budget_s": options.wall_budget_s, "cleanup_grace_s": cleanup_grace}), report_path)
                elif process.returncode != 0:
                    emit_check(report, Check("physics", "worker_exit", "INCONCLUSIVE",
                                             f"Worker exited with code {process.returncode}; inspect the private worker log."), report_path)
            except Exception as error:
                if process is not None and process.poll() is None:
                    _terminate_worker(process)
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
                emit_check(report, Check("physics", "worker_launch_exception", "INCONCLUSIVE",
                                         f"Worker failed: {type(error).__name__}: {error}"), report_path)
    if context is not None:
        _verify(context, report, report_path)
    report["timing"]["total_s"] = time.monotonic() - start
    report["finished_at_utc"] = utc_now()
    refresh_report(report, complete=True, missing_reason="Native validation skipped after a proven static contract failure." if static_fail else None)
    atomic_json(report_path, report)
    return report_path, report


def run_worker(request_path):
    """Load simulator dependencies only in the disposable worker process."""
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    values = request["options"]
    for name in ("enabled", "directions_deg"):
        values[name] = tuple(values[name])
    options = Options(**{item.name: values[item.name] for item in fields(Options) if item.name in values})
    report_path = Path(request["report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["worker_environment"] = environment()
    atomic_json(report_path, report)
    context = None
    try:
        context = _load_context(Path(request["task_file"]), options)
        if not _verify(context, report, report_path):
            return 1
        from wia.runtime.preflight.native import run_native
        run_native(context, options, report_path.parent, lambda check: emit_check(report, check, report_path))
    except Exception as error:
        traceback.print_exc()
        emit_check(report, Check("physics", "native_exception", "INCONCLUSIVE",
                                 f"Native diagnostics raised {type(error).__name__}: {error}"), report_path)
        return 2
    finally:
        if context is not None:
            _verify(context, report, report_path)
        atomic_json(report_path, refresh_report(report))
    return 0


def run_batch(task_files, options: Options, output_dir, *, all_tasks=False):
    selected, skipped, warnings = discover_selections(task_files, all_tasks=all_tasks)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = utc_now().replace(":", "").replace("+", "_") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_root / run_id
    run_dir.mkdir()
    summary = {"schema": "agent_preflight_summary_v1", "private_diagnostic": True,
               "run_id": run_id, "run_dir": str(run_dir), "started_at_utc": utc_now(),
               "mode": options.mode, "options": options.to_dict(), "task_count": len(selected),
               "completed_task_count": 0, "counts": {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0},
               "reports": [], "skipped_count": len(skipped), "skipped": skipped, "discovery_warnings": warnings}
    start = time.monotonic()
    for index, path in enumerate(selected):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)[:100]
        suffix = hashlib.sha256(str(path).encode()).hexdigest()[:10]
        output = run_dir / f"{index:04d}_{safe}_{suffix}"
        try:
            report_path, report = run_one(path, options, output)
        except Exception as error:
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "preflight.json"
            (output / "diagnostic.log").write_text(traceback.format_exc(), encoding="utf-8")
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = {"schema": "agent_preflight_v1", "private_diagnostic": True, "task_id": path.stem,
                          "task_file": str(path), "options": options.to_dict(), "required_checks": list(GROUPS),
                          "checks": [], "environment": environment(),
                          "log_paths": {"diagnostic": str(output / "diagnostic.log")}}
            report["checks"].append(Check("static", "runner_exception", "INCONCLUSIVE",
                                          f"Runner raised {type(error).__name__}: {error}").to_dict())
            refresh_report(report, complete=True)
            atomic_json(report_path, report)
        summary["counts"][report["status"]] += 1
        summary["completed_task_count"] += 1
        summary["reports"].append({"task_id": report["task_id"], "task_file": str(path), "status": report["status"],
                                   "checked_scope_status": report["checked_scope_status"],
                                   "verification_level": report["verification_level"], "report": str(report_path)})
        summary["wall_time_s"] = time.monotonic() - start
        atomic_json(run_dir / "summary.json", summary)
    summary["finished_at_utc"] = utc_now()
    summary["wall_time_s"] = time.monotonic() - start
    summary["summary_path"] = str(run_dir / "summary.json")
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(output_root / "summary.json", summary)
    return summary

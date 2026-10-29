"""Build private preflight evidence reports. Run src/wia/runtime/commands/preflight_esi_tasks.py --help."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from wia.runtime.resources import runtime_source
import platform
import socket
import subprocess
import sys
from typing import Any

from wia.runtime.preflight.models import Check, LEVELS, Options, aggregate

ROOT = WIA_RUNTIME_ROOT


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_evidence(value):
    """Represent invalid physical measurements explicitly, never as invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_evidence(item) for item in value]
    if hasattr(value, "tolist"):
        return json_evidence(value.tolist())
    return value


def atomic_json(path: Path, value: Any):
    """Only replace a complete report, preserving the previous evidence on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_evidence(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_fingerprint(path):
    path = Path(path).expanduser().resolve()
    result = {"path": str(path), "exists": path.is_file()}
    if result["exists"]:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result["bytes"] = path.stat().st_size
    return result


def revision(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "python_executable": sys.executable,
            "pid": os.getpid(), "pbs_job_id": os.environ.get("PBS_JOBID"),
            "pbs_environment": os.environ.get("PBS_ENVIRONMENT"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def fingerprints(context, options: Options, task_file: Path):
    files = {str(Path(task_file).resolve())}
    files.update(str(path) for path in (ROOT / "src/wia/runtime/preflight").glob("*.py"))
    files.update(str(runtime_source(path)) for path in ("src/wia/runtime/commands/preflight_esi_tasks.py", "src/wia/runtime/commands/run_esi_robot.py",
                                               "src/wia/runtime/commands/plan_esi_action_order_offline.py",
                                               "src/wia/runtime/sim/esi_omnigibson.py", "src/wia/runtime/sim/esi_base.py", "src/wia/runtime/sim/esi_arm.py",
                                               "src/wia/runtime/sim/esi_ik.py", "src/wia/runtime/sim/esi_gripper.py", "core/action_compiler.py",
                                               "src/wia/tasks/support_sampling.py"))
    files.update(str(ROOT / "configs" / name) for name in ("esi_base.json", "esi_arm.json", "esi_gripper.json",
        "esi_rotation.json", "esi_trunk.json", "esi_robot_cameras.json", "esi_agent_observation.json"))
    definition = None
    if context is not None:
        loaded = context.loaded
        for obj in (loaded, getattr(loaded, "scene", None)):
            if obj is None:
                continue
            for name in ("path", "source_path", "question_path"):
                value = getattr(obj, name, None)
                if value is not None:
                    files.add(str(Path(value).resolve()))
            runtime = getattr(getattr(obj, "definition", None), "runtime", None)
            if runtime is not None:
                for name in ("trunk_config", "robot_camera_config", "viewer_config"):
                    value = getattr(runtime, name, None)
                    if value:
                        files.add(str((ROOT / value).resolve()))
        if hasattr(loaded, "protocol_sha256"):
            from wia.tasks.rigid_qa import PROTOCOL_PATH
            files.add(str(PROTOCOL_PATH))
        definition = loaded.definition.model_dump(mode="json")
        for patch in context.scene.definition.scene.variants[context.loaded.variant].patches:
            if patch.operation == "add_usd_object":
                files.add(str((ROOT / patch.asset).resolve()))
    if options.observation_config:
        files.add(str(Path(options.observation_config).expanduser().resolve()))
    configuration = {"options": options.to_dict(), "definition": definition}
    return {"files": [file_fingerprint(path) for path in sorted(files)],
            "configuration_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
            "repository_commit": revision(ROOT), "esi_submodule_commit": revision(WIA_ESI_ROOT)}


def refresh_report(report, *, complete=False, missing_reason=None):
    """Aggregate all adapter obligations, including checks outside the requested scope."""
    checks = [item for item in report["checks"] if not item.get("evidence", {}).get("placeholder")]
    groups = {item["group"] for item in checks if item.get("required", True)}
    enabled = set(report["options"]["enabled"])
    selected = {"static"} if report["options"]["mode"] == "static" else {"static", *enabled}
    for group in report["required_checks"]:
        if group in groups and group in selected:
            continue
        if group not in selected:
            reason = "Required validator is outside the selected mode or is disabled."
        else:
            reason = missing_reason or ("Required validator produced no conclusive evidence." if complete
                                        else "Required validator has not completed.")
        checks.append(Check(group, "required_validator", "INCONCLUSIVE", reason,
                            evidence={"placeholder": True, "executed": False}).to_dict())
    report["checks"] = checks
    level_checks = [item for item in checks if not item.get("evidence", {}).get("placeholder")]
    report["verification_level"] = max((item["verification_level"] for item in level_checks),
                                       key=LEVELS.index, default="static")
    if aggregate(checks) == "PASS" and "goal" in report["required_checks"]:
        goal_evidence = [item for item in checks if item["group"] == "goal" and item["status"] == "PASS"
                         and item["verification_level"] == "goal_verified" and item.get("required", True)]
        if not goal_evidence:
            checks.append(Check("goal", "physical_goal_evidence", "INCONCLUSIVE",
                                "The complete physical goal has not been verified; necessary conditions alone do not prove task feasibility.",
                                evidence={"placeholder": True, "executed": False}).to_dict())
    report["status"] = aggregate(checks)
    report["checked_scope"] = sorted(selected)
    report["checked_scope_status"] = aggregate([item for item in checks if item["group"] in selected])
    report["not_verified"] = [{"group": item["group"], "name": item["name"], "reason": item["reason"]}
                              for item in checks if item["status"] == "INCONCLUSIVE"]
    report["complete"] = complete
    report["updated_at_utc"] = utc_now()
    return report


def emit_check(report, check: Check, path: Path):
    report["checks"] = [item for item in report["checks"]
                        if not item.get("evidence", {}).get("placeholder")
                        and (item["group"], item["name"]) != (check.group, check.name)]
    report["checks"].append(check.to_dict())
    refresh_report(report)
    atomic_json(path, report)

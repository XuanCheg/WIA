"""Check CPU-only preflight lifecycle and conservative report aggregation.

Run: .venv/bin/python -m pytest -q tests/test_preflight_runner.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import wia.runtime.preflight.runner as runner
import wia.runtime.preflight.static as static
from wia.runtime.preflight.models import Check, Options
from wia.runtime.preflight.report import emit_check, refresh_report


def report_with(checks=(), *, options=None, required=("static", "physics", "goal")):
    return {"checks": [check.to_dict() for check in checks],
            "required_checks": list(required), "options": (options or Options()).to_dict()}


@pytest.fixture
def cpu_runner(monkeypatch, tmp_path):
    """Replace task-specific boundaries while retaining real report and process I/O."""
    def context(path, options):
        return SimpleNamespace(
            loaded=SimpleNamespace(verify_unchanged=lambda: None),
            required_checks=("static", "physics", "goal"),
            identity=lambda: {"task_id": path.stem, "task_file": str(path),
                              "scene_file": str(path), "variant": "original", "seed": 0},
        )

    monkeypatch.setattr(runner, "_load_context", context)
    monkeypatch.setattr(runner, "fingerprints", lambda *args: {"files": []})
    monkeypatch.setattr(runner, "_allocation_problem", lambda: None)
    monkeypatch.setattr(static, "run_static", lambda *args: [Check("static", "contract", "PASS", "Valid test contract.")])

    def task(name):
        path = tmp_path / (name + ".yaml")
        path.write_text("status: ready\n", encoding="utf-8")
        return path

    return task


# These children deliberately perform no simulator imports or GPU operations.
_FAKE_WORKER = r'''
import json
import os
from pathlib import Path
import signal
import sys
import time

request = json.loads(Path(sys.argv[1]).read_text())
mode = Path(request["task_file"]).stem
path = Path(request["report"])
report = json.loads(path.read_text())

def emit(group, name, status, level):
    report["checks"].append({"group": group, "name": name, "status": status,
        "reason": "Controlled worker lifecycle evidence.", "verification_level": level,
        "required": True, "evidence": {}})
    temporary = path.with_suffix(".worker.tmp")
    temporary.write_text(json.dumps(report))
    temporary.replace(path)

if mode.endswith("crash"):
    os._exit(7)
if mode.endswith("timeout"):
    if mode.endswith("fail_timeout"):
        emit("physics", "proven_collision", "FAIL", "native_initialization")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
else:
    emit("physics", "initialization", "PASS", "native_initialization")
    emit("goal", "final_state", "PASS", "goal_verified")
'''


@pytest.mark.skipif(os.name != "posix", reason="Worker isolation uses POSIX process groups.")
@pytest.mark.parametrize("failure, expected", [("timeout", "INCONCLUSIVE"),
                                               ("fail_timeout", "FAIL"),
                                               ("crash", "INCONCLUSIVE")])
def test_worker_failure_retains_evidence_and_batch_continues(cpu_runner, monkeypatch, tmp_path, failure, expected):
    original_popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        if "--worker-request" not in command:
            return original_popen(command, **kwargs)
        assert kwargs["start_new_session"] is True
        process = original_popen([sys.executable, "-c", _FAKE_WORKER, command[-1]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    selected = [cpu_runner("a_" + failure), cpu_runner("b_success")]
    summary = runner.run_batch(selected, Options(mode="full", wall_budget_s=.8), tmp_path / "reports")
    reports = [json.loads(Path(item["report"]).read_text()) for item in summary["reports"]]

    assert summary["completed_task_count"] == 2
    assert [report["status"] for report in reports] == [expected, "PASS"]
    failure_check = "worker_exit" if failure == "crash" else "worker_timeout"
    assert any(check["name"] == failure_check and check["status"] == "INCONCLUSIVE"
               for check in reports[0]["checks"])
    if failure == "fail_timeout":
        assert any(check["name"] == "proven_collision" and check["status"] == "FAIL"
                   for check in reports[0]["checks"])
    assert reports[1]["verification_level"] == "goal_verified"
    assert all(process.poll() is not None for process in processes)
    if failure != "crash":
        assert processes[0].returncode == -signal.SIGKILL
        with pytest.raises(ProcessLookupError):
            os.killpg(processes[0].pid, 0)


def test_unexpected_task_exception_does_not_abort_later_reports(cpu_runner, monkeypatch, tmp_path):
    load = runner._load_context

    def fail_first(path, options):
        if path.stem == "a_exception":
            raise RuntimeError("Injected loader failure.")
        return load(path, options)

    monkeypatch.setattr(runner, "_load_context", fail_first)
    summary = runner.run_batch([cpu_runner("a_exception"), cpu_runner("b_valid")],
                               Options(), tmp_path / "reports")
    reports = [json.loads(Path(item["report"]).read_text()) for item in summary["reports"]]
    assert summary["completed_task_count"] == 2
    assert summary["counts"] == {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 2}
    assert "Injected loader failure" in reports[0]["checks"][0]["reason"]
    assert reports[1]["checked_scope_status"] == "PASS"


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="Static deadline uses POSIX interval timers.")
def test_static_deadline_reports_inconclusive(cpu_runner, monkeypatch, tmp_path):
    def blocked_load(*args):
        time.sleep(5)

    monkeypatch.setattr(runner, "_load_context", blocked_load)
    _, report = runner.run_one(cpu_runner("slow_static"), Options(wall_budget_s=.03), tmp_path / "slow_report")
    assert report["status"] == "INCONCLUSIVE"
    assert report["timing"]["total_s"] < 1
    assert report["worker"]["started"] is False
    assert any(check["name"] == "wall_budget" for check in report["checks"])


@pytest.mark.parametrize("options", [Options(mode="static"), Options(mode="full", enabled=("static",))])
def test_unselected_required_validators_prevent_task_pass(options):
    report = report_with([Check("static", "contract", "PASS", "Valid contract.")], options=options)
    refresh_report(report, complete=True)
    assert report["status"] == "INCONCLUSIVE"
    assert report["checked_scope_status"] == "PASS"
    assert {check["group"] for check in report["checks"] if check["status"] == "INCONCLUSIVE"} == {"physics", "goal"}


@pytest.mark.parametrize("goal", [None,
    Check("goal", "unknown", "INCONCLUSIVE", "No goal validator is registered."),
    Check("goal", "fit", "PASS", "Geometric fit only.", verification_level="geometric_screen"),
    Check("goal", "lift", "PASS", "Necessary physical condition only.", verification_level="physical_manipulation")])
def test_missing_unknown_or_partial_goal_evidence_cannot_pass(goal):
    checks = [Check("static", "contract", "PASS", "Valid contract.")]
    if goal is not None:
        checks.append(goal)
    report = report_with(checks, options=Options(mode="full"), required=("static", "goal"))
    refresh_report(report, complete=True)
    assert report["status"] == "INCONCLUSIVE"
    assert any(check["group"] == "goal" and check["status"] == "INCONCLUSIVE" for check in report["checks"])


def test_later_physical_evidence_replaces_same_check(tmp_path):
    report = report_with([Check("static", "contract", "PASS", "Valid contract.")],
                         options=Options(mode="full"), required=("static", "goal"))
    path = tmp_path / "preflight.json"
    emit_check(report, Check("goal", "probe", "INCONCLUSIVE", "Search has not completed."), path)
    emit_check(report, Check("goal", "probe", "PASS", "Final physical state verified.",
                             verification_level="goal_verified"), path)
    stored = json.loads(path.read_text())
    assert stored["status"] == "PASS"
    assert stored["verification_level"] == "goal_verified"
    assert len([check for check in stored["checks"] if check["group"] == "goal"]) == 1
    assert stored["not_verified"] == []


def test_partial_initialization_does_not_enable_a_disabled_physics_validator():
    report = report_with([Check("static", "contract", "PASS", "Valid contract."),
                          Check("physics", "native_initialization", "PASS", "Initialized.")],
                         options=Options(mode="full", enabled=("static", "perception")),
                         required=("static", "physics"))
    refresh_report(report, complete=True)
    assert report["status"] == "INCONCLUSIVE"
    assert any(check["group"] == "physics" and check["status"] == "INCONCLUSIVE" for check in report["checks"])


def test_nonfinite_evidence_is_valid_json_and_does_not_erase_diagnosis(tmp_path):
    report = report_with(required=("physics",))
    path = tmp_path / "preflight.json"
    emit_check(report, Check("physics", "invalid_pose", "FAIL", "Pose contains nonfinite values.",
                             evidence={"position": [0., float("nan"), float("inf")],
                                       "nested": {"minimum": -float("inf")}}), path)
    stored = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail("Invalid JSON numeric token: " + value))
    assert stored["status"] == "FAIL"
    assert stored["checks"][0]["evidence"] == {"position": [0., "NaN", "Infinity"], "nested": {"minimum": "-Infinity"}}
    assert not list(tmp_path.glob("*.tmp"))


def test_reusing_output_root_preserves_prior_run(cpu_runner, tmp_path):
    task = cpu_runner("same_task")
    output = tmp_path / "reports"
    first = runner.run_batch([task], Options(), output)
    first_path = Path(first["reports"][0]["report"])
    before = first_path.read_bytes()
    second = runner.run_batch([task], Options(), output)
    assert first["run_dir"] != second["run_dir"]
    assert first_path.read_bytes() == before
    assert Path(first["summary_path"]).is_file()
    assert Path(second["reports"][0]["report"]).is_file()
    assert json.loads((output / "summary.json").read_text())["run_id"] == second["run_id"]

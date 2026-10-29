"""Check CPU preflight contracts without a simulator or GPU allocation.

Run: .venv/bin/python -m pytest -q tests/test_preflight_contracts.py
"""
from collections import Counter
import json
from pathlib import Path
import shutil

import pytest
import yaml

from wia.runtime.preflight.adapters import load_context
from wia.runtime.preflight.models import GROUPS, Options
from wia.runtime.preflight.runner import discover_selections, run_batch
from wia.runtime.preflight.static import run_static
from wia.tasks.loader import DATA_ROOT, sha256
from harness_integration.task_examples import example_task_path


@pytest.fixture
def copied_task(tmp_path, monkeypatch):
    """Copy reviewed examples and provide a small explicit background inventory."""
    inventory_path = tmp_path / "background.json"
    background = {"coffee_table_dnsjnv_0": {"category": "coffee_table", "model": "dnsjnv",
                                           "in_rooms": ["dining_room_0"], "fixed_base": True}}
    inventory_path.write_text(json.dumps(background), encoding="utf-8")
    monkeypatch.setattr("wia.runtime.preflight.static._background_inventory",
                        lambda scene: (background, str(inventory_path), None))

    def make(name="find_and_lift"):
        source_task = example_task_path(name)
        value = yaml.safe_load(source_task.read_text(encoding="utf-8"))
        root = tmp_path / name / "data"
        source = root / value["source"]["json"]
        source.parent.mkdir(parents=True)
        shutil.copyfile(DATA_ROOT / value["source"]["json"], source)
        path = tmp_path / name / "task.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path, root, source, value

    return make


def _batch_report(path, root, output, *, variant=None):
    summary = run_batch([path], Options(mode="static", data_root=str(root), variant=variant), output)
    assert summary["task_count"] == summary["completed_task_count"] == 1
    report = json.loads(Path(summary["reports"][0]["report"]).read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["worker"]["started"] is False
    assert report["verification_level"] == "static"
    return report


def _statuses(checks):
    return {check.name: check.status for check in checks}


def test_ready_catalog_discovers_all_single_and_multiple_targets():
    paths, skipped, warnings = discover_selections(all_tasks=True)
    assert len(paths) == 27
    assert warnings == []
    assert all(item["status"] in ("pending", "discarded") for item in skipped)
    contexts = [load_context(path) for path in paths]
    assert Counter(len(context.targets) for context in contexts) == {1: 17, 2: 7, 3: 3}
    for context in contexts:
        assert context.kind == "rigid_containment"
        assert context.targets == tuple(context.loaded.definition.targets)
        assert context.required_checks == GROUPS
        assert len(set(context.bindings[key].name for key in context.targets)) == len(context.targets)
        assert context.containers
        assert all(key in context.bindings for key in (*context.targets, *context.containers, *context.supports))
        assert set(context.targets).issubset(context.object_supports)


@pytest.mark.parametrize("name,variant,goal,target_name", [
    ("find_and_lift", "visible_target_control", "grasp_lift", "obj_small"),
    ("retrieve_apple_and_lift", "closed_sliding_box", "slide_extract_lift", "task_apple"),
])
def test_generic_known_variants_use_batch_and_preserve_inputs(copied_task, tmp_path, name, variant, goal, target_name):
    path, root, source, _ = copied_task(name)
    before = {path: sha256(path), source: sha256(source)}
    report = _batch_report(path, root, tmp_path / "reports", variant=variant)
    assert report["status"] == "INCONCLUSIVE"
    assert report["goal"] == goal
    assert report["variant"] == variant
    assert report["roles"]["targets"] == ["target"]
    assert report["roles"]["bindings"]["target"]["name"] == target_name
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["binding:target"] == "PASS"
    assert statuses["goal_evaluator"] == "PASS"
    assert not any(item["status"] == "FAIL" for item in report["checks"])
    assert {item: sha256(item) for item in before} == before
    fingerprints = {item["path"]: item.get("sha256") for item in report["fingerprints"]["files"]}
    assert fingerprints[str(source.resolve())] == before[source]
    assert fingerprints[str(path.resolve())] == before[path]


def test_unknown_evaluator_keeps_generic_binding_checks(copied_task, tmp_path):
    path, root, _, value = copied_task()
    value["private"]["evaluation"] = {"type": "future_goal", "target": "target", "future_parameter": 7}
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    context = load_context(path, data_root=root)
    assert context.goal == "future_goal"
    assert context.evaluation["future_parameter"] == 7
    report = _batch_report(path, root, tmp_path / "reports")
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert report["status"] == "INCONCLUSIVE"
    assert statuses["goal_evaluator"] == "INCONCLUSIVE"
    assert statuses["binding:target"] == "PASS"
    assert statuses["binding:container"] == "PASS"


@pytest.mark.parametrize("field,bad_value", [("name", "absent_target"), ("category", "incorrect_category")])
def test_missing_object_and_wrong_type_are_proven_failures(copied_task, field, bad_value):
    path, root, _, value = copied_task()
    value["private"]["bindings"]["target"][field] = bad_value
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    checks = run_static(load_context(path, data_root=root), Options(mode="static", data_root=str(root)))
    failed = {check.name for check in checks if check.status == "FAIL"}
    assert failed == {"binding:target"}
    assert _statuses(checks)["binding:container"] == "PASS"


def test_unavailable_background_never_invents_object_absence(copied_task, monkeypatch):
    path, root, _, value = copied_task()
    value["private"]["bindings"]["target"]["name"] = "unresolved_target"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    monkeypatch.setattr("wia.runtime.preflight.static._background_inventory", lambda scene: ({}, None, "Not available in fixture."))
    checks = run_static(load_context(path, data_root=root), Options(mode="static", data_root=str(root)))
    statuses = _statuses(checks)
    assert statuses["binding:target"] == "INCONCLUSIVE"
    assert statuses["binding:support"] == "INCONCLUSIVE"
    assert statuses["binding:container"] == "PASS"
    assert not any(check.status == "FAIL" for check in checks)


def test_original_variant_does_not_include_unapplied_additions(copied_task):
    path, root, _, _ = copied_task("retrieve_apple_and_lift")
    context = load_context(path, data_root=root, variant="original")
    checks = run_static(context, Options(mode="static", data_root=str(root), variant="original"))
    assert {check.name for check in checks if check.status == "FAIL"} == {"binding:target", "binding:container"}


def test_scene_definition_requires_only_static_and_physics():
    paths, _, _ = discover_selections(all_tasks=True)
    task_context = load_context(paths[0])
    context = load_context(task_context.scene.path)
    assert context.kind == "scene"
    assert context.required_checks == ("static", "physics")
    assert context.goal is None
    assert context.evaluation == {}
    assert context.capabilities == ()
    assert context.targets == (context.scene.definition.target,)

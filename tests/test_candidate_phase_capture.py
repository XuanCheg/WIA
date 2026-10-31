"""Scene-only phase export orchestration. Run: PYTHONPATH=src pytest -q tests/test_candidate_phase_capture.py."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import wia.candidates.phase_capture as phase_capture


def temporal_session(phase="phase_1"):
    controller = SimpleNamespace(phase=phase, transition=None)
    session = SimpleNamespace(temporal_phase_controller=controller)
    selected = SimpleNamespace(definition=SimpleNamespace(id="temporal_scene",
        bindings={"first": SimpleNamespace(name="phase_1_actor"),
                  "later": SimpleNamespace(name="phase_2_actor"),
                  "support": SimpleNamespace(name="table")}))
    return session, selected


def dependencies(monkeypatch, session, *, export_error=None, capture_error=None, unavailable=()):
    calls = []

    def capture(current, selected, expected):
        assert current is session
        assert current.temporal_phase_controller.phase == "phase_1"
        calls.append(("capture_views", list(expected)))
        current.temporal_phase_controller.phase = "phase_2"
        current.temporal_phase_controller.transition = {
            "from_phase": "phase_1", "to_phase": "phase_2", "prepared_boundary": [1, 0],
            "world_unchanged": True, "physics_steps": 0, "robot_actions": 0,
            "source_defined_external_event": True}
        current.temporal_capture_evidence = {"kind": "human_native_two_phase_preview",
            "phase_1_snapshot_id": 1, "phase_2_snapshot_id": 2, "robot_actions": 0}
        calls.append(("advance", "phase_2"))
        if capture_error:
            raise capture_error
        return {name: f"native_pixels:{name}" for name in expected}, None

    def save(path, pixels):
        calls.append(("save_png", path.name))
        assert pixels == f"native_pixels:{path.stem}"
        path.write_text("native image helper receipt\n")

    def export(current, **kwargs):
        calls.append(("export_inventory", deepcopy(kwargs)))
        assert current.temporal_phase_controller.phase == "phase_2"
        if export_error:
            raise export_error
        return {"schema": "wia_candidate_native_inventory_v1", "id": kwargs["identity"],
                "objects": [{"name": "phase_2_actor", "position": [1, 2, 3]}],
                "inactive_targets": [{"name": "phase_1_actor"}],
                "unavailable_targets": [{"name": name} for name in unavailable]}

    monkeypatch.setattr(phase_capture, "_capture_loaded_views", capture)
    monkeypatch.setattr(phase_capture, "_save_png", save)
    monkeypatch.setattr(phase_capture, "_export_inventory", export)
    return calls


def test_non_temporal_invokes_no_helpers_and_writes_nothing(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Non-temporal sessions must not invoke phase helpers")

    for name in ("_capture_loaded_views", "_save_png", "_export_inventory"):
        monkeypatch.setattr(phase_capture, name, forbidden)
    output = tmp_path / "unused"
    assert phase_capture.capture_temporal_geometry(object(), object(), output, target_names=[]) is None
    assert not output.exists()


def test_invalid_phase_is_refused_without_capture_export_or_reset(monkeypatch, tmp_path):
    session, selected = temporal_session("phase_2")

    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid initial phase must not capture or export")

    for name in ("_capture_loaded_views", "_save_png", "_export_inventory"):
        monkeypatch.setattr(phase_capture, name, forbidden)
    first = tmp_path / "inventory.json"
    first.write_text("original Phase 1 inventory\n")
    receipt = phase_capture.capture_temporal_geometry(session, selected, tmp_path, target_names=[])
    assert receipt["status"] == "error"
    assert receipt["errors"][0]["stage"] == "phase_precondition"
    assert session.temporal_phase_controller.phase == "phase_2"
    assert first.read_text() == "original Phase 1 inventory\n"
    assert json.loads((tmp_path / "temporal_geometry.json").read_text()) == receipt
    assert not (tmp_path / "phase_2_inventory.json").exists()


def test_inventory_is_exported_after_native_transition_with_all_bindings(monkeypatch, tmp_path):
    session, selected = temporal_session()
    calls = dependencies(monkeypatch, session)
    first = tmp_path / "inventory.json"
    first.write_text("original Phase 1 inventory\n")
    supports = {"phase_2_actor": "table"}
    receipt = phase_capture.capture_temporal_geometry(session, selected, tmp_path,
        target_names=["phase_1_actor", "extra_bound_actor"], support_by_target=supports)
    assert receipt["schema"] == "wia_candidate_temporal_geometry_v1"
    assert receipt["status"] == "captured", receipt
    assert calls[0] == ("capture_views", list(phase_capture.PHASE_VIEWS))
    assert calls[1][0] == "advance"
    exported = next(value for operation, value in calls if operation == "export_inventory")
    assert exported["identity"] == "temporal_scene:phase_2"
    assert exported["target_names"] == ["extra_bound_actor", "phase_1_actor", "phase_2_actor", "table"]
    assert exported["support_by_target"] == supports
    assert receipt["robot_actions"] == receipt["model_calls"] == 0
    assert receipt["agent_success_evaluated"] is False
    assert "task_success" not in receipt and "answer" not in receipt
    assert set(receipt["images"]) == set(phase_capture.PHASE_VIEWS)
    inventory = json.loads((tmp_path / "phase_2_inventory.json").read_text())
    assert inventory["id"] == "temporal_scene:phase_2"
    assert inventory["inactive_targets"] == [{"name": "phase_1_actor"}]
    assert first.read_text() == "original Phase 1 inventory\n"
    assert receipt["transition_evidence"]["native_transition"]["prepared_boundary"] == [1, 0]
    assert json.loads((tmp_path / "temporal_geometry.json").read_text()) == receipt


def test_object_phase_views_only_requested_when_already_configured(monkeypatch, tmp_path):
    session, selected = temporal_session()
    session.human_review_camera_evidence = {"views": {name: {} for name in phase_capture.OBJECT_VIEWS}}
    calls = dependencies(monkeypatch, session)
    receipt = phase_capture.capture_temporal_geometry(session, selected, tmp_path, target_names=[])
    expected = calls[0][1]
    for name in phase_capture.OBJECT_VIEWS:
        assert name in expected
        assert f"phase_1_{name}" in expected
        assert f"phase_2_{name}" in expected
    assert receipt["status"] == "captured"


def test_failed_phase_two_export_is_persisted_without_phase_one_fallback(monkeypatch, tmp_path):
    session, selected = temporal_session()
    dependencies(monkeypatch, session, export_error=ValueError("native floor geometry unavailable"))
    first = tmp_path / "inventory.json"
    first.write_text("original Phase 1 inventory\n")
    receipt = phase_capture.capture_temporal_geometry(session, selected, tmp_path, target_names=[])
    assert receipt["status"] == "error"
    assert receipt["phase_after"] == "phase_2"
    assert receipt["phase_2_inventory"] is None
    assert receipt["errors"][0]["stage"] == "phase_2_inventory"
    assert "native floor geometry unavailable" in receipt["errors"][0]["error"]
    assert not (tmp_path / "phase_2_inventory.json").exists()
    assert first.read_text() == "original Phase 1 inventory\n"
    assert json.loads((tmp_path / "temporal_geometry.json").read_text()) == receipt


def test_capture_failure_after_transition_still_exports_actual_phase_two_geometry(monkeypatch, tmp_path):
    session, selected = temporal_session()
    calls = dependencies(monkeypatch, session, capture_error=RuntimeError("phase wrist unavailable"))
    receipt = phase_capture.capture_temporal_geometry(session, selected, tmp_path, target_names=[])
    assert receipt["status"] == "error" and receipt["phase_after"] == "phase_2"
    assert receipt["errors"][0]["stage"] == "phase_capture"
    assert receipt["phase_2_inventory"] == str(tmp_path / "phase_2_inventory.json")
    assert len([operation for operation, _ in calls if operation == "capture_views"]) == 1
    assert len([operation for operation, _ in calls if operation == "export_inventory"]) == 1


def test_unavailable_phase_two_targets_are_retained_as_explicit_error(monkeypatch, tmp_path):
    session, selected = temporal_session()
    dependencies(monkeypatch, session, unavailable=["phase_2_actor"])
    receipt = phase_capture.capture_temporal_geometry(session, selected, tmp_path, target_names=[])
    assert receipt["status"] == "error"
    assert receipt["errors"][0]["stage"] == "phase_2_inventory_availability"
    inventory = json.loads((tmp_path / "phase_2_inventory.json").read_text())
    assert inventory["unavailable_targets"] == [{"name": "phase_2_actor"}]

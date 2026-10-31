"""Export Phase 2 geometry from a continuing-world scene-only temporal preview.

Run: PYTHONPATH=src python -m pytest -q tests/test_candidate_phase_capture.py

Call ``capture_temporal_geometry`` after exporting the original Phase 1
``inventory.json``. The caller owns session setup, cameras, output and teardown.
This function uses the existing human temporal capture path; it never resets,
scores, answers, declares completion, invokes a policy, or issues robot actions.
The source-defined temporal controller is the sole owner of the transition.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


SCHEMA = "wia_candidate_temporal_geometry_v1"
PHASE_VIEWS = ("phase_1_head", "phase_1_wrist", "phase_2_head", "phase_2_wrist",
               "phase_2_global_diagonal")
OBJECT_VIEWS = ("object_overview", "object_top", "object_side")


def _capture_loaded_views(session, selected_scene, expected_views):
    from wia.runtime.scene_capture_adapters import capture_loaded_views
    return capture_loaded_views(session, selected_scene, expected_views)


def _export_inventory(session, **kwargs):
    from wia.candidates.inventory import export_inventory
    return export_inventory(session, **kwargs)


def _save_png(path, pixels):
    from core.record.images import save_png
    save_png(path, pixels)


def _target_names(selected_scene, target_names):
    names = {str(name) for name in target_names if name}
    for binding in selected_scene.definition.bindings.values():
        name = binding.get("name") if isinstance(binding, dict) else getattr(binding, "name", None)
        if name:
            names.add(str(name))
    return sorted(names)


def capture_temporal_geometry(session, selected_scene, output: Path, *, target_names,
                              support_by_target=None) -> dict | None:
    """Capture the authored Phase 1→2 event and export actual Phase 2 inventory.

    Non-temporal sessions return None without invoking helpers or writing files.
    A temporal session must still be in Phase 1. Unavailable views, transition
    evidence or inventory are persisted as explicit phase errors; Phase 1 data
    is never substituted for a failed Phase 2 export. All selected bindings are
    requested so inactive/parked actors remain accounted for by the exporter.
    """
    controller = getattr(session, "temporal_phase_controller", None)
    if controller is None:
        return None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scene_id = str(selected_scene.definition.id)
    names = _target_names(selected_scene, target_names)
    receipt = {"schema": SCHEMA, "scene_id": scene_id, "status": "error",
               "phase_initial": getattr(controller, "phase", None), "phase_after": None,
               "phase_1_inventory_preserved": True, "requested_target_names": names,
               "images": {}, "phase_2_inventory": None, "transition_evidence": None,
               "errors": [], "robot_actions": 0, "model_calls": 0,
               "agent_success_evaluated": False,
               "validation_scope": "Human scene-only continuing-world temporal geometry; no agent execution or success evaluation."}

    def error(stage, exception):
        receipt["errors"].append({"stage": stage, "error": f"{type(exception).__name__}: {exception}"})

    if receipt["phase_initial"] != "phase_1":
        error("phase_precondition", ValueError("Temporal geometry requires the original initialized Phase 1"))
    else:
        expected_views = list(PHASE_VIEWS)
        configured = getattr(session, "human_review_camera_evidence", None)
        configured_views = configured.get("views", {}) if isinstance(configured, dict) else {}
        if all(name in configured_views for name in OBJECT_VIEWS):
            expected_views.extend(OBJECT_VIEWS)
            expected_views.extend(f"{phase}_{name}" for phase in ("phase_1", "phase_2") for name in OBJECT_VIEWS)
        receipt["expected_views"] = expected_views
        views = {}
        try:
            views, _ = _capture_loaded_views(session, selected_scene, expected_views)
            missing = set(expected_views) - views.keys()
            if missing:
                error("phase_views", ValueError(f"Requested native phase views are unavailable: {sorted(missing)}"))
        except Exception as exception:
            error("phase_capture", exception)
        receipt["phase_after"] = getattr(controller, "phase", None)
        transition = deepcopy(getattr(controller, "transition", None))
        receipt["transition_evidence"] = {
            "native_transition": transition,
            "native_capture": deepcopy(getattr(session, "temporal_capture_evidence", None)),
        }
        if receipt["phase_after"] != "phase_2":
            error("phase_transition", ValueError("Native temporal capture did not advance to Phase 2"))
        else:
            if (not isinstance(transition, dict) or transition.get("from_phase") != "phase_1"
                    or transition.get("to_phase") != "phase_2"
                    or transition.get("world_unchanged") is not True
                    or transition.get("robot_actions") != 0):
                error("phase_transition_evidence", ValueError("Continuing-world, zero-action native transition evidence is unavailable"))
            for name, pixels in views.items():
                try:
                    path = output / f"{name}.png"
                    _save_png(path, pixels)
                    receipt["images"][name] = str(path)
                except Exception as exception:
                    error(f"image:{name}", exception)
            try:
                inventory = _export_inventory(session, identity=f"{scene_id}:phase_2",
                    target_names=names, support_by_target=support_by_target)
                if (not isinstance(inventory, dict)
                        or inventory.get("schema") != "wia_candidate_native_inventory_v1"
                        or inventory.get("id") != f"{scene_id}:phase_2"):
                    raise ValueError("Phase 2 native inventory schema/identity is unavailable")
                path = output / "phase_2_inventory.json"
                path.write_text(json.dumps(inventory, indent=2) + "\n")
                receipt["phase_2_inventory"] = str(path)
                if inventory.get("unavailable_targets"):
                    error("phase_2_inventory_availability", ValueError(
                        "Requested Phase 2 target geometry is unavailable: "
                        + str([row.get("name") for row in inventory["unavailable_targets"]])))
            except Exception as exception:
                error("phase_2_inventory", exception)
    receipt["phase_after"] = getattr(controller, "phase", None)
    if not receipt["errors"]:
        receipt["status"] = "captured"
    (output / "temporal_geometry.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt

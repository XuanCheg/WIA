"""CPU contract diagnostics. Run src/wia/runtime/commands/preflight_esi_tasks.py --mode static.

Evidence from declarative metadata establishes identities, never physical success.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from wia.tasks.loader import DATA_ROOT, ROOT
from wia.runtime.preflight.adapters import GOAL_CAPABILITIES, KNOWN_GOALS, metadata_objects, read_source
from wia.runtime.preflight.models import Check


def _check(name, status, reason, *, objects=(), evidence=None, action=None):
    return Check("static", name, status, reason, list(objects), evidence or {}, action)


def _background_inventory(scene_name):
    roots = []
    if os.environ.get("OMNIGIBSON_DATA_PATH"):
        roots.append(Path(os.environ["OMNIGIBSON_DATA_PATH"]).expanduser())
    roots.append(Path("/home/svu/xinyangzhou/CFP05-CF-143/esi-bench/data/omnigibson/behavior-1k-assets"))
    attempted = []
    for root in roots:
        for base in (root, root / "behavior-1k-assets"):
            path = base / "scenes" / scene_name / "json" / (scene_name + "_best.json")
            attempted.append(str(path))
            if path.is_file():
                try:
                    raw = json.loads(path.read_text())
                    info = raw["objects_info"]["init_info"]
                    return {name: dict(item["args"]) for name, item in info.items()}, str(path), None
                except (OSError, ValueError, KeyError, TypeError) as error:
                    return {}, str(path), str(error)
    return {}, None, "No existing background inventory found: " + ", ".join(attempted)


def _patch_objects(context, source_objects, background):
    objects = dict(background)
    objects.update(source_objects)
    checks = []
    patches = context.scene.definition.scene.variants[context.scene.variant].patches
    for patch in patches:
        if patch.operation not in ("add_primitive", "add_dataset_object", "add_usd_object"):
            continue
        name = patch.name
        if name in objects:
            checks.append(_check("patch_unique_name:" + name, "FAIL", "An added object would replace an existing object.", objects=[name]))
        if patch.operation == "add_primitive":
            item = {"category": "object", "model": None, "fixed_base": True, "visual_only": False}
        elif patch.operation == "add_dataset_object":
            item = {"category": patch.category, "model": patch.model, "fixed_base": False, "visual_only": False}
        else:
            asset = (ROOT / patch.asset).resolve()
            valid_asset = asset.is_relative_to((ROOT / "assets").resolve()) and asset.is_file()
            checks.append(_check("patch_asset:" + name, "PASS" if valid_asset else "FAIL",
                                 "USD asset exists inside repository assets." if valid_asset else "Referenced USD asset is missing or outside repository assets.",
                                 objects=[name], evidence={"path": str(asset)}))
            item = {"fixed_base": True, "visual_only": False}
        objects[name] = dict(item, provenance="selected_variant_addition")
    return objects, checks


def _reference_checks(context):
    references = set(context.targets + context.containers + context.supports)
    evaluation = context.evaluation
    for key in ("target", "container", "support"):
        if isinstance(evaluation.get(key), str):
            references.add(evaluation[key])
    for variant in context.scene.definition.scene.variants.values():
        for patch in variant.patches:
            if hasattr(patch, "binding"):
                references.add(patch.binding)
            references.update(getattr(patch, "bindings", ()))
    missing = sorted(references - context.bindings.keys())
    checks = [_check("binding_references", "FAIL" if missing else "PASS",
                     "Object references lack explicit bindings." if missing else "Every declared object reference has a binding.",
                     objects=missing)]
    target_names = [context.bindings[key].name for key in context.targets if key in context.bindings]
    if len(set(target_names)) != len(target_names):
        checks.append(_check("distinct_targets", "FAIL", "Multiple target roles resolve to the same physical object.", objects=list(context.targets)))
    return checks


def _qa_checks(context, raw, metadata):
    checks = []
    source = context.scene.source_path
    reference = DATA_ROOT / context.scene.definition.source.json_path
    if reference.is_file() and reference.resolve() != source.resolve():
        original, _ = read_source(reference)
        keys = ("question", "answer", "answer_type", "options_json")
        changed = [key for key in keys if raw.get(key) != original.get(key)]
        checks.append(_check("original_qa_copy", "FAIL" if changed else "PASS",
                             "The source copy changes original QA fields." if changed else "The source copy preserves original QA fields verbatim.",
                             evidence={"reference": str(reference), "compared_fields": list(keys), "changed_fields": changed}))
    else:
        checks.append(_check("original_qa_copy", "PASS" if reference.resolve() == source.resolve() else "INCONCLUSIVE",
                             "The selected source is the original dataset record." if reference.resolve() == source.resolve() else
                             "No original dataset record is available for copy comparison.", evidence={"reference": str(reference)}))
    question = raw.get("question")
    valid = isinstance(question, str) and bool(question.strip()) and "answer" in raw
    checks.append(_check("source_qa_fields", "PASS" if valid else "FAIL",
                         "Source question and answer fields exist." if valid else "Original source question or answer is missing."))
    if context.kind == "rigid_containment":
        from wia.tasks.rigid_qa import PROTOCOL_PATH, protocol
        qraw, qmetadata = read_source(context.loaded.question_path)
        same = qraw.get("question") == context.loaded.definition.public.objective
        checks.append(_check("public_question_preservation", "PASS" if same else "FAIL",
                             "The public question is preserved verbatim." if same else "The public objective changes the original question."))
        # Derivative questions intentionally have different answers; compare each
        # derivative with its own source, not the canonical construction question.
        qreference = DATA_ROOT / context.loaded.definition.source.json_path
        if qreference.is_file() and qreference.resolve() != context.loaded.question_path.resolve():
            upstream, _ = read_source(qreference)
            changed = [key for key in ("question", "answer", "answer_type", "options_json") if qraw.get(key) != upstream.get(key)]
            checks.append(_check("question_qa_copy", "FAIL" if changed else "PASS",
                                 "Derivative QA copy differs from its original record." if changed else "Derivative QA copy preserves its own original record.",
                                 evidence={"reference": str(qreference), "changed_fields": changed}))
        try:
            protocol.build_system_prompt(qmetadata, .8, 3)
            checks.append(_check("qa_protocol", "PASS", "The original shared containment answer protocol accepts this metadata.",
                                 evidence={"protocol": str(PROTOCOL_PATH), "physical_evaluator": "not_configured"}))
        except (ValueError, KeyError, TypeError) as error:
            checks.append(_check("qa_protocol", "FAIL", "Original containment protocol cannot parse the source metadata.", evidence={"error": str(error)}))
        construction = lambda item: {"containee": {key: item.get("containee", {}).get(key) for key in
                                                  ("cat", "model", "scale", "bbox_inventory")},
                                     **{key: item.get(key) for key in ("initial_poses", "containers", "extras", "floor_name")}}
        equal = construction(metadata) == construction(qmetadata)
        checks.append(_check("question_scene_geometry", "PASS" if equal else "FAIL",
                             "Question and canonical scene share construction inputs." if equal else "Question metadata changes the canonical scene construction inputs."))
    elif raw.get("small_task") != "Rigid Containment":
        checks.append(_check("qa_protocol", "INCONCLUSIVE", "No diagnostic adapter validates the original answer protocol for this source family.",
                             evidence={"source_family": raw.get("small_task")}))
    patches = context.scene.definition.scene.variants[context.scene.variant].patches
    changes = [patch.model_dump(mode="json") for patch in patches if patch.operation != "robot_pose"]
    checks.append(_check("qa_semantic_transform", "INCONCLUSIVE" if changes else "PASS",
                         "Existing scene transformations may affect the original answer or evidence; semantic equivalence has not been established." if changes else
                         "The selected variant applies no object transformations to the original source.",
                         evidence={"transformations": changes, "proven_conflict": False}))
    return checks


def run_static(context, options):
    """Check file, role, capability and QA contracts using CPU evidence only."""
    checks = [_check("schema_and_source", "PASS", "Task and scene contracts were parsed with duplicate-key rejection and validated file references.",
                     evidence={"kind": context.kind, "variant": context.loaded.variant, "seed": context.loaded.seed,
                               "opaque_evaluation": bool(context.goal and context.goal not in KNOWN_GOALS)})]
    loaded, scene = context.loaded, context.scene
    consistent = (loaded.variant, loaded.seed, loaded.scene_name, loaded.room_name) == (scene.variant, scene.seed, scene.scene_name, scene.room_name)
    checks.append(_check("selection_identity", "PASS" if consistent else "FAIL",
                         "Task and scene use the same variant, seed, scene and room." if consistent else "Task and scene selections differ."))
    checks.extend(_reference_checks(context))
    raw, metadata = read_source(scene.source_path)
    source_objects, known_family = metadata_objects(raw, metadata)
    background, inventory_path, inventory_error = _background_inventory(scene.scene_name)
    objects, patch_checks = _patch_objects(context, source_objects, background)
    checks.extend(patch_checks)
    checks.append(_check("source_object_adapter", "PASS" if known_family else "INCONCLUSIVE",
                         "Source object identities are parsed by an explicit family adapter." if known_family else
                         "This source family has no declarative object reconstruction adapter.",
                         evidence={"source_family": raw.get("small_task"), "background_inventory": inventory_path,
                                   "inventory_error": inventory_error}))
    all_additions = {patch.name for variant in scene.definition.scene.variants.values() for patch in variant.patches
                     if patch.operation in ("add_primitive", "add_dataset_object", "add_usd_object")}
    for key, binding in context.bindings.items():
        item = objects.get(binding.name)
        evidence = {"name": binding.name, "expected_category": binding.category, "expected_model": binding.model,
                    "background_inventory": inventory_path}
        if item is None:
            # An object exclusive to another variant is demonstrably not added.
            # Missing background evidence alone never proves physical absence.
            absent_addition = binding.name in all_additions and binding.name not in source_objects and bool(background)
            complete_missing = known_family and inventory_path is not None and inventory_error is None
            proven = absent_addition or complete_missing
            checks.append(_check("binding:" + key, "FAIL" if proven else "INCONCLUSIVE",
                                 "The binding is absent from the source, selected variant and available scene inventory." if proven else
                                 "The binding cannot be resolved from available declarative evidence.", objects=[key], evidence=evidence,
                                 action="Verify this binding against the selected variant and native scene inventory."))
            continue
        mismatches = [field for field in ("category", "model") if getattr(binding, field) is not None and
                      item.get(field) is not None and getattr(binding, field) != item[field]]
        unresolved_type = [field for field in ("category", "model") if getattr(binding, field) is not None and item.get(field) is None]
        rooms = item.get("in_rooms")
        excluded_room = isinstance(rooms, list) and bool(rooms) and scene.room_name not in rooms and scene.room_name != "full_scene"
        evidence.update(actual_category=item.get("category"), actual_model=item.get("model"),
                        provenance=item.get("provenance", "source_metadata" if binding.name in source_objects else "background_inventory"),
                        in_rooms=rooms, mismatches=mismatches, unresolved_type_fields=unresolved_type)
        status = "FAIL" if mismatches else "INCONCLUSIVE" if unresolved_type or excluded_room else "PASS"
        reason = "The declared object type disagrees with source metadata." if mismatches else (
            "The object exists in another room; native room-filter inclusion requires verification." if excluded_room else
            "Object identity is known but its requested type is not fully recorded." if unresolved_type else
            "The binding and declared type match available object metadata.")
        checks.append(_check("binding:" + key, status, reason, objects=[key], evidence=evidence))
        if key in context.targets and context.goal in KNOWN_GOALS:
            bad = item.get("fixed_base") is True or item.get("visual_only") is True
            if bad:
                checks.append(_check("target_mobility:" + key, "FAIL", "A manipulation target is explicitly fixed or visual-only.",
                                     objects=[key], evidence={"fixed_base": item.get("fixed_base"), "visual_only": item.get("visual_only")}))
    if context.kind != "scene":
        minimum = set(GOAL_CAPABILITIES.get(context.goal, ()))
        missing = sorted(minimum - set(context.capabilities))
        status = "FAIL" if missing else "PASS" if context.capabilities and context.goal in KNOWN_GOALS else "INCONCLUSIVE"
        checks.append(_check("robot_capabilities", status,
                             "Required manipulation capabilities are absent." if missing else
                             "Robot capability requirements are explicit and consistent with the recognized goal." if status == "PASS" else
                             "Capability requirements for this goal have not been resolved.",
                             evidence={"declared": list(context.capabilities), "missing": missing,
                                       "origin": "rigid_containment_adapter" if context.kind == "rigid_containment" else "task_requirements"}))
        available = context.goal in KNOWN_GOALS[:3]
        checks.append(_check("goal_evaluator", "PASS" if available else "INCONCLUSIVE",
                             "The existing native task evaluator defines this success condition." if available else
                             "The original QA protocol exists, but an automatic final physical placement evaluator is not configured." if context.goal == "answer_and_place" else
                             "No registered physical evaluator defines this opaque goal.",
                             evidence={"goal": context.goal, "evaluation": context.evaluation,
                                       "evaluator_source": str(ROOT / "src/wia/tasks/scene.py") if available else None},
                             action=None if available else "Provide a task-specific physical goal validator; retain generic diagnostics."))
    checks.extend(_qa_checks(context, raw, metadata))
    return checks

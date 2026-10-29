"""Check separated multi-object question, geometry and human preview provenance.

Run: .venv/bin/python -m pytest -q tests/test_rigid_multi_tasks.py
"""
import json
from pathlib import Path

import pytest
import yaml

from wia.runtime.scene_review import SceneReview, VIEW_LABELS, write_json
from wia.tasks.loader import load_task
from harness_integration.test_rigid_catalog_conversion import publication


@pytest.fixture
def multi(tmp_path):
    root, source, _, candidate, _ = publication(tmp_path)
    raw = json.loads(source.read_text())
    raw["question"] = "Which container best fits each of the soccer ball and the cubicle?"
    raw["answer"] = "PRIVATE_ANSWER_CANARY"
    question = source.with_name(source.stem + "__criterion_2.json")
    write_json(question, raw)
    scene = source.with_name(source.stem + "__multi_extra1.yaml")
    value = yaml.safe_load(candidate.read_text())
    value.update(status="ready", target_bindings=[value["target"], "extra1"])
    value["bindings"]["extra1"] = {"name": "multi_extra1", "category": "cubicle", "model": "hzalfx"}
    value["scene"]["variants"]["tabletop"]["patches"].append({"operation": "add_dataset_object",
        "phase": "before_construction", "name": "multi_extra1", "category": "cubicle", "model": "hzalfx",
        "scale": [0.2, 0.3, 0.4], "position": [1., 2., 0.8]})
    scene.write_text(yaml.safe_dump(value))
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump({"kind": "rigid_containment_task", "id": "merged_containment",
        "status": "ready", "source": {"provider": "esi", "json": str(question.relative_to(root))},
        "scene_file": str(scene.relative_to(root)), "targets": value["target_bindings"],
        "goal": "answer_and_place", "allow_remove_after_verification": True,
        "public": {"objective": raw["question"], "success_criteria": "Complete each pickup and insertion.",
            "environment": "Explore with R1."}}))
    return root, source, question, scene, task


def test_original_question_and_canonical_geometry_are_distinct_and_private(multi):
    root, source, question, scene, task = multi
    loaded = load_task(task, data_root=root)
    assert loaded.source_path == source and loaded.question_path == question
    assert loaded.scene.path == scene and loaded.variant == "tabletop"
    assert len(loaded.definition.targets) == 2
    assert loaded.scene.definition.scene.variants["tabletop"].patches[-1].scale == (.2, .3, .4)
    public = json.dumps(loaded.public_payload)
    assert "PRIVATE_ANSWER_CANARY" not in public and "partially inside" in public
    assert "smaller suitable" in public and "0.80" in public
    assert loaded.manifest("fixture")["physical_evaluator"] == "not_configured"
    with pytest.raises(ValueError, match="tabletop"):
        load_task(task, data_root=root, variant="original")
    raw = json.loads(question.read_text())
    raw["question"] += " Changed."
    write_json(question, raw)
    with pytest.raises(RuntimeError, match="question"):
        loaded.verify_unchanged()


def test_pending_task_is_skipped_before_native_initialization(multi):
    root, _, _, _, task = multi
    value = yaml.safe_load(task.read_text())
    value["status"] = "pending"
    task.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="pending.*skipped"):
        load_task(task, data_root=root)


def test_single_object_task_reuses_original_scene_without_target_list(multi):
    root, source, _, scene, task = multi
    value = yaml.safe_load(scene.read_text())
    value.pop("target_bindings")
    scene.write_text(yaml.safe_dump(value))
    contract = yaml.safe_load(task.read_text())
    contract.update(targets=[value["target"]], source={"provider": "esi", "json": str(source.relative_to(root))})
    contract["public"]["objective"] = json.loads(source.read_text())["question"]
    task.write_text(yaml.safe_dump(contract))
    loaded = load_task(task, data_root=root)
    assert loaded.question_path == loaded.source_path == source
    assert loaded.definition.targets == (value["target"],)
    assert loaded.definition.defaults.max_steps == 60
    assert "0.80" in loaded.task_text and "real R1" in loaded.task_text


def test_publisher_preserves_only_current_layout_approval(multi):
    from wia.tasks.loader import sha256
    from wia.runtime.commands.publish_esi_rigid_multi import review_is_current
    _, _, question, scene, _ = multi
    approval = {"decision": "approved", "scene_sha256": sha256(scene), "question_sha256": sha256(question)}
    assert review_is_current(approval, scene, question)
    scene.write_text(scene.read_text() + "\n")
    assert not review_is_current(approval, scene, question)


def test_seat_surface_ignores_chair_arm_and_other_objects():
    from types import SimpleNamespace
    from wia.runtime.commands.generate_esi_rigid_multi import support_height
    support = SimpleNamespace(prim_path='/World/chair')
    hits = [
        {'hit': True, 'rigidBody': '/World/other/link', 'normal': [0, 0, 1],
            'position': [0, 0, .5], 'distance': .1},
        {'hit': True, 'rigidBody': '/World/chair/arm', 'normal': [0, 0, 1],
            'position': [0, 0, .72], 'distance': .2},
        {'hit': True, 'rigidBody': '/World/chair/seat', 'normal': [0, 0, 1],
            'position': [0, 0, .43], 'distance': .5}]
    def rays(*args, **kwargs):
        return hits
    assert support_height(rays, support, [0, 0], [0, 0, .1], [1, 1, .9], seating=True) == .43
    assert support_height(rays, support, [0, 0], [0, 0, .1], [1, 1, .9]) == .72
    hits.pop()
    assert support_height(rays, support, [0, 0], [0, 0, .1], [1, 1, .9], seating=True) is None


def test_episode_runner_selects_generated_layout_without_launching_a_world(multi, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import wia.runtime.commands.run_esi_explore_episode as coordinator
    root, _, _, scene, task = multi
    original_run = coordinator.subprocess.run
    def gpu_status(command, *args, **kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout="0, GPU-test, 99, 20000\n")
        return original_run(command, *args, **kwargs)
    monkeypatch.setattr(coordinator.subprocess, "run", gpu_status)
    output = tmp_path / "episode"
    with pytest.raises(RuntimeError, match="No selected idle local GPU"):
        coordinator.main(["--task-file", str(task), "--data-root", str(root), "--output-dir", str(output)])
    manifest = json.loads((output / "episode_manifest.json").read_text())
    assert manifest["scene_variant"] == "tabletop" and manifest["scene_file"] == str(scene)
    assert manifest["max_steps"] == 60 and manifest["physical_evaluator"] == "not_configured"


def test_every_object_view_has_its_own_visibility_evidence(multi, tmp_path):
    root, _, question, scene, task = multi
    conversion = tmp_path / "conversion"
    write_json(conversion / "run/conversion_summary.json", {"results": [{
        "source": str(question.relative_to(root)), "scene_file": str(scene.relative_to(root)),
        "task_file": str(task), "status": "ready", "targets": [
            {"category": "soccer_ball"}, {"category": "cubicle"}]}]})
    store = SceneReview(conversion, tmp_path / "review", data_root=root)
    row = store.catalogue()["scenes"][0]
    image = tmp_path / "image.png"
    image.write_bytes(b"native image fixture")
    receipt = {"fingerprint": row["fingerprint"], "variant": row["variant"], "seed": row["seed"],
        "images": {**{key: str(image) for key in VIEW_LABELS}, "global_target_extra1": str(image)},
        "target_locations": [{"camera_capture_protocol": "target_instance_verified_v3", "visible_target_pixels": 200},
            {"camera_capture_protocol": "target_instance_verified_v3", "visible_target_pixels": 0}]}
    marker = store.output / "captures" / row["id"] / "preview_complete.json"
    write_json(marker, receipt)
    shown = store.catalogue()["scenes"][0]
    assert len(shown["images"]) == 10 and shown["scene_file"] == str(scene)
    receipt["target_locations"][1]["visible_target_pixels"] = 200
    write_json(marker, receipt)
    shown = store.catalogue()["scenes"][0]
    assert len(shown["images"]) == 11
    assert shown["images"][-1]["label"] == "Pickup · cubicle"
    assert "PRIVATE_ANSWER_CANARY" not in json.dumps(shown)
    assert Path(shown["task_file"]) == task

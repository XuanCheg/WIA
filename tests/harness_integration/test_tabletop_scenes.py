"""Tabletop generation and scene-only phase checks.

Run .venv/bin/python -m pytest -q tests/test_tabletop_scenes.py.
"""
from copy import deepcopy
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from wia.tasks.loader import DATA_ROOT, load_scene
from wia.tasks.scene_only import SceneOnly, horizontal_box_overlap, local_floor_surface
from wia.tasks.tabletop import generate_tabletop

SOURCE = DATA_ROOT / "Physical Structure/Rigid Containment/grocery_store_half_stocked/grocery_store_0/grocery_store_half_stocked_scene_dict_storage2_2_revised.json"


def test_local_floor_surface_filters_other_objects_and_downward_faces():
    hits = [{"hit": True, "rigidBody": body, "position": [0., 0., z],
             "normal": [0., 0., normal], "distance": distance}
            for body, z, normal, distance in (
                ("/World/floor_decor/base_link", .066, 1., .034),
                ("/World/floor/base_link", .04, -1., .06),
                ("/World/floor/base_link", 0., 1., .1),
                ("/World/floor/base_link", -.04, 1., .14))]
    assert local_floor_surface("/World/floor", hits)["position"][2] == 0.
    with pytest.raises(ValueError, match="No local upward-facing floor surface"):
        local_floor_surface("/World/missing", hits)


def manifest(tmp_path, **kwargs):
    definition, details = generate_tabletop(SOURCE, DATA_ROOT, **kwargs)
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True)))
    return load_scene(path), details


@pytest.mark.parametrize("yaw", [0., 135.])
def test_geometry_packing_fits_every_object_without_overlap(tmp_path, yaw):
    scene, details = manifest(tmp_path, target_scale=.35, robot_yaw_deg=yaw)
    depth, width, _ = details["table_dimensions_m"]
    boxes = []
    for item in details["layout"].values():
        centre, extent = np.array(item["table_frame_xy_m"]), np.array(item["extent_m"][:2])
        low, high = centre - extent / 2, centre + extent / 2
        assert np.all(low >= -np.array([depth, width]) / 2)
        assert np.all(high <= np.array([depth, width]) / 2)
        boxes.append((low, high))
    for (low, high), (other_low, other_high) in itertools.combinations(boxes, 2):
        assert np.any(np.minimum(high, other_high) <= np.maximum(low, other_low))
    target = details["layout"]["target"]
    assert target["table_frame_xy_m"][0] - target["extent_m"][0] / 2 == pytest.approx(-depth / 2 + .04)
    assert scene.definition.source.json_path.endswith(SOURCE.name)
    assert "public" not in scene.definition.model_dump()
    assert "evaluation" not in scene.definition.model_dump()


def test_scene_question_projection_excludes_source_answer_and_metadata(tmp_path):
    from dataclasses import replace
    import json
    scene, _ = manifest(tmp_path)
    raw = json.loads(scene.source_path.read_text())
    question = raw["question"]
    raw.update(answer="PRIVATE_QA_ANSWER_CANARY", metadata_json="PRIVATE_GEOMETRY_CANARY",
               options_json="PRIVATE_OPTIONS_CANARY")
    source = tmp_path / "source.json"
    source.write_text(json.dumps(raw))
    scene = replace(scene, source_path=source)
    assert scene.task_text == question
    assert scene.public_payload["objective"] == question
    assert "PRIVATE_" not in json.dumps(scene.public_payload)


def test_constructor_scale_preserves_source_and_nonuniform_existing_scale(tmp_path):
    before = SOURCE.read_bytes()
    loaded, _ = manifest(tmp_path, target_scale=.35)
    patches = SceneOnly(loaded)
    config = {"robots": [{"type": "R1"}], "objects": [
        {"name": "obj_containee", "category": "soccer_ball", "model": "ikagkc",
         "scale": [1., 2., 3.], "position": [150., 100., 100.]}]}
    patches.apply_to_env_config(config)
    assert np.allclose(config["objects"][0]["scale"], [.35, .70, 1.05])
    assert config["objects"][0]["position"] == [150., 100., 100.]
    assert SOURCE.read_bytes() == before
    with pytest.raises(RuntimeError, match="initialization"):
        patches.apply_to_env_config(config)


def test_postprocessing_pose_overrides_source_only_before_episode(tmp_path, monkeypatch):
    loaded, _ = manifest(tmp_path)
    patches = SceneOnly(loaded)
    patches.apply_to_env_config({"robots": [{"type": "R1"}], "objects": []})
    objects = {}
    class Object:
        def set_position_orientation(self, **values):
            self.values = values
        def keep_still(self):
            self.still = True
    for key in loaded.definition.qualification.supported_objects:
        obj = Object()
        obj.values = {"position": [99., 99., -100.]}
        objects[loaded.definition.bindings[key].name] = obj
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(tensor=lambda x, dtype: x, float32="float32"))
    steps = []
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda kind, name: objects[name])),
                              pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(step=lambda: steps.append(1)))))
    patches.apply_after_postprocessing(session)
    for patch in patches.applied:
        if patch["operation"] == "object_pose":
            obj = objects[loaded.definition.bindings[patch["binding"]].name]
            assert obj.values["position"] == patch["position"] and obj.still
    assert len(steps) == loaded.definition.qualification.settle_steps
    with pytest.raises(RuntimeError, match="exactly once"):
        patches.apply_after_postprocessing(session)


def test_rotated_box_screen_does_not_reject_empty_aabb_corners():
    centre, extent = np.zeros(3), np.array([2., .1, .04])
    quaternion = Rotation.from_euler("z", 45, degrees=True).as_quat()
    assert not horizontal_box_overlap(centre, quaternion, extent,
                                     np.array([.6, -.7, 0.]), np.array([.7, -.6, .1]))
    assert horizontal_box_overlap(centre, quaternion, extent,
                                  np.array([.5, .5, 0.]), np.array([.6, .6, .1]))


def test_original_variant_is_preserved(tmp_path):
    loaded, _ = manifest(tmp_path)
    original = load_scene(loaded.path, variant="original")
    config = {"robots": [{"type": "R1"}], "objects": []}
    before = deepcopy(config)
    SceneOnly(original).apply_to_env_config(config)
    assert config == before


def test_two_container_sources_are_supported():
    paths = list((DATA_ROOT / "Physical Structure/Rigid Containment").rglob(
        "Benevolence_0_int_scene_dict_storage2_0_revised.json"))
    assert len(paths) == 1
    definition, details = generate_tabletop(paths[0], DATA_ROOT)
    assert len(definition.qualification.supported_objects) == 3
    assert len(details["layout"]) == 3


def test_support_mapping_requires_explicit_furniture_and_supported_object(tmp_path):
    from wia.tasks.schema import SceneDefinition
    from pydantic import ValidationError
    loaded, _ = manifest(tmp_path)
    raw = loaded.definition.model_dump(mode="json", by_alias=True)
    raw["qualification"]["object_supports"] = {"target": "coffee_table"}
    with pytest.raises(ValidationError, match="explicit bindings"):
        SceneDefinition.model_validate(raw)
    raw["bindings"]["coffee_table"] = {"name": "coffee_table_example", "category": "coffee_table"}
    definition = SceneDefinition.model_validate(raw)
    assert definition.qualification.object_supports == {"target": "coffee_table"}
    assert definition.qualification.support == "table"
    raw["qualification"]["object_supports"] = {"coffee_table": "table"}
    with pytest.raises(ValidationError, match="supported object bindings"):
        SceneDefinition.model_validate(raw)


def test_retrieval_cue_preserves_original_qa_and_combined_stopping(tmp_path):
    from wia.tasks.rigid_qa import RigidContainmentQA
    loaded, _ = manifest(tmp_path)
    original = RigidContainmentQA(loaded, goal="answer_and_place")
    retrieval = RigidContainmentQA(loaded, goal="answer_and_place", pickup_location="coffee table")
    assert retrieval.public_payload["objective"] == original.public_payload["objective"]
    assert retrieval.public_prompt == original.public_prompt
    assert "coffee table" in retrieval.public_payload["success_criteria"]
    assert "separate table" in retrieval.public_payload["success_criteria"]
    parsed = retrieval.parse({"answer": "large", "confidence": .99, "reasoning": "Still retrieving."}, "MV_FWD")
    assert retrieval.episode_should_stop(parsed, [parsed], 4, 60) == (False, "")
    with pytest.raises(ValueError, match="combined"):
        RigidContainmentQA(loaded, pickup_location="coffee table")

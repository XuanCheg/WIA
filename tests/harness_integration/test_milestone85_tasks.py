"""Task migration and physical preconditions. Run .venv/bin/python -m pytest -q this file."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
import yaml

from core.action_interface import BackendCapabilities
from wia.tasks.loader import select_task, verify_manifest
from harness_integration.task_examples import load_example_task as load_task
from wia.tasks.schema import TaskDefinition
from wia.tasks.scene import DeclarativeScene
from wia.runtime.sim.esi_explore_scenes import ExploreScene

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT


def task_yaml():
    task = load_task("find_and_lift")
    return task, yaml.safe_load(task.path.read_text())


def test_source_and_public_projection_are_independent_of_qa_truth():
    task = load_task("find_and_lift")
    assert task.source_path.is_file()
    assert task.source_path.name.endswith("__no.json")
    public = json.dumps(task.public_payload)
    assert "__no.json" not in public and "obj_small" not in public and "dqjuub" not in public
    assert "inside the cleansing bottle" not in public and "0.05" in public


@pytest.mark.parametrize("change", ["unknown_root", "unknown_private", "traversal", "unsupported_patch", "bad_threshold"])
def test_task_schema_rejects_unreviewed_fields_and_operations(change):
    task, value = task_yaml()
    if change == "unknown_root":
        value["context_policy"] = "custom"
    elif change == "unknown_private":
        value["private"]["evaluation"]["oracle"] = True
    elif change == "traversal":
        value["source"]["json"] = "../../private.json"
    elif change == "unsupported_patch":
        value["scene"]["variants"]["original"]["patches"] = [{"operation": "automatic_repair"}]
    else:
        value["private"]["evaluation"]["minimum_lift_m"] = -0.01
    with pytest.raises((ValueError, ValidationError)):
        TaskDefinition.model_validate(value)


def test_duplicate_keys_and_ambiguous_ids_fail_but_sources_can_be_shared(tmp_path):
    task, value = task_yaml()
    one, two = tmp_path / "one.yaml", tmp_path / "two.yaml"
    one.write_text(yaml.safe_dump(value))
    value["public"]["objective"] = "A different public objective using the same physical scene."
    two.write_text(yaml.safe_dump(value))
    assert load_task(one).source_path == load_task(two).source_path
    with pytest.raises(ValueError, match="ambiguous"):
        select_task("find_and_lift", task_root=tmp_path)
    assert select_task("one", task_root=tmp_path) == one
    one.write_text(one.read_text() + "\nid: overwritten\n")
    with pytest.raises(ValueError, match="Duplicate"):
        load_task(one)


@pytest.mark.parametrize("variant", ["visible_target_control", "single_table_occlusion", "multi_region_search",
                                      "single_table_occlusion_near", "multi_region_search_near"])
@pytest.mark.parametrize("seed", [1, 7, 11])
def test_migrated_patches_reproduce_legacy_rng_geometry_exactly(variant, seed):
    task = load_task("find_and_lift", variant=variant, seed=seed)
    declarative = DeclarativeScene(task)
    legacy = ExploreScene.load(ROOT / "configs/esi_explore" / (variant + ".json"), seed=seed)
    assert declarative.observation_settings() == legacy.observation_settings()
    original = {"robots": [{"type": "R1"}], "objects": [dict(legacy.config["target_object"], position=[0., 0., 1.]),
        {"name": "obj_container", "category": "cleansing_bottle", "model": "ovjhuf", "position": [-4.90, 16., 0.87]}]}
    old, new = deepcopy(original), deepcopy(original)
    legacy.apply_to_env_config(old)
    declarative.apply_to_env_config(new)
    assert old["objects"] == new["objects"]
    with pytest.raises(RuntimeError, match="initialization"):
        declarative.apply_to_env_config(new)


def test_original_variant_does_not_apply_any_patch():
    scene = DeclarativeScene(load_task("find_and_lift"))
    config = {"robots": [{"type": "R1"}], "objects": []}
    expected = deepcopy(config)
    scene.apply_to_env_config(config)
    assert config == expected and scene.applied == []


@pytest.mark.parametrize("failure", [None, "support", "target", "capabilities"])
def test_qualification_checks_actual_physical_preconditions_and_skips(failure, tmp_path):
    task = load_task("find_and_lift")
    scene = DeclarativeScene(task)
    target = SimpleNamespace(name="obj_small", category="cherry_tomato", model="dqjuub", visual_only=False, fixed_base=False,
        aabb=(np.array([0.1, 0.1, 1.]), np.array([0.2, 0.2, 1.1])))
    support = SimpleNamespace(aabb=(np.array([0., 0., 0.8]), np.array([0.5, 0.5, 1.])))
    objects = {"obj_small": target, "coffee_table_dnsjnv_0": support,
               "obj_container": SimpleNamespace(category="cleansing_bottle", model="ovjhuf")}
    if failure == "target":
        target.fixed_base = True
    if failure == "support":
        support.aabb = (np.array([4., 4., 0.]), np.array([5., 5., 1.]))
    caps = BackendCapabilities(base_translation=True, base_rotation=True, ee_translation=True,
                               gripper=failure != "capabilities")
    session = SimpleNamespace(scene_name=task.scene_name, room_name=task.room_name,
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda kind, name: objects.get(name))))
    result = scene.qualify(session, SimpleNamespace(capabilities=caps), tmp_path)
    assert result.status == ("accepted_original" if failure is None else "skipped")
    assert (result.reason is None) == (failure is None)
    assert (tmp_path / "private_scene/qualification.json").exists()


def test_episode_manifest_rejects_silent_task_or_policy_changes(tmp_path):
    task = load_task("find_and_lift")
    value = task.manifest("episode")
    value.update(max_steps=2, world_mode="pause", world_advances_during_inference=False,
                 observation_condition="current", action_mode="fixed", model_name="gpt-6-luna",
                 reasoning_effort="high", wall_timeout_s=180., agent_session="native_persistent")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    assert verify_manifest(path, task) == value
    value["public_task"]["objective"] = "Silently change the goal."
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="exact task"):
        verify_manifest(path, task)


@pytest.mark.parametrize("timeout", [None, 5400., 0., -1., float("inf"), True])
def test_episode_deadline_can_be_disabled_but_invalid_limits_still_fail(timeout, tmp_path):
    task = load_task("find_and_lift")
    value = task.manifest("episode-with-optional-deadline")
    value.update(max_steps=60, world_mode="run", world_advances_during_inference=True,
                 observation_condition="current", action_mode="parameterized", model_name="gpt-6-luna",
                 reasoning_effort="high", wall_timeout_s=timeout, agent_session="native_persistent")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    if timeout is None or type(timeout) is float and timeout == 5400.:
        assert verify_manifest(path, task)["wall_timeout_s"] == timeout
    else:
        with pytest.raises(RuntimeError, match="policy"):
            verify_manifest(path, task)

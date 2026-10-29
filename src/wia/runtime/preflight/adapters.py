"""Declarative task adapters. Run src/wia/runtime/commands/preflight_esi_tasks.py --mode static.

This module reads contracts and metadata only; it never initializes simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from pydantic import create_model
import yaml

from wia.tasks.loader import (DATA_ROOT, LoadedTask, LoadedRigidTask, LoadedScene,
                               UniqueLoader, load_task, load_scene, select_task, sha256)
from wia.tasks.schema import Binding, ClosedModel, TaskDefinition
from wia.runtime.preflight.models import GROUPS

KNOWN_GOALS = ("grasp_lift", "open_extract_lift", "slide_extract_lift", "answer_and_place")
GOAL_CAPABILITIES = {
    goal: ("ee_translation", "gripper") for goal in KNOWN_GOALS
}
RIGID_CAPABILITIES = ("base_translation", "base_rotation", "ee_translation", "gripper")


def read_source(path):
    """Decode ESI metadata without importing the reconstruction pipeline."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ESI source must contain a JSON object.")
    metadata = raw.get("metadata_json", {})
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    if not isinstance(metadata, dict):
        raise ValueError("ESI metadata_json must encode an object.")
    return raw, metadata


def _rows(value):
    if isinstance(value, dict):
        return [dict(item, _key=item.get("_key", key)) for key, item in value.items()
                if isinstance(item, dict)]
    return [item for item in (value or []) if isinstance(item, dict)]


def metadata_objects(raw, metadata):
    """Map only identities explicitly defined by supported reconstruction contracts.

    The returned Boolean indicates that source-added objects are enumerated; it
    does not claim that the background scene inventory is complete.
    """
    objects = {}
    if raw.get("small_task") == "Rigid Containment":
        from wia.tasks.rigid_qa import protocol
        target = metadata.get("containee")
        if isinstance(target, dict):
            objects["obj_containee"] = dict(target, category=target.get("cat"), role="target")
        for item in _rows(metadata.get("containers")):
            name = protocol.container_object_name(item)
            objects[name] = dict(item, category=item.get("cat"), role="container")
        for item in _rows(metadata.get("extras")):
            key = item.get("_key", "")
            if key.startswith("extra") and key[5:].isdigit():
                objects["obj_" + key] = dict(item, category=item.get("cat"), role="extra")
        return objects, True
    if raw.get("small_task") == "Material Transparency":
        for item in _rows(metadata.get("objects")):
            name = item.get("name") or item.get("_key")
            if name:
                role = {"obj_small": "target", "obj_container": "container"}.get(name, "object")
                objects[name] = dict(item, role=role)
        return objects, True
    return objects, False


@dataclass(frozen=True)
class TaskContext:
    loaded: LoadedTask | LoadedRigidTask | LoadedScene
    scene: LoadedTask | LoadedScene
    kind: str
    bindings: dict[str, Binding]
    targets: tuple[str, ...]
    containers: tuple[str, ...]
    supports: tuple[str, ...]
    capabilities: tuple[str, ...]
    goal: str | None
    evaluation: dict[str, Any]
    dependencies: tuple[dict, ...]
    required_checks: tuple[str, ...]

    @property
    def task_id(self):
        return self.loaded.definition.id

    @property
    def object_supports(self):
        qualification = self.scene.definition.qualification
        default = getattr(qualification, "support", None)
        objects = getattr(qualification, "supported_objects", self.targets)
        mappings = getattr(qualification, "object_supports", {})
        return {key: mappings.get(key, default) for key in objects if mappings.get(key, default)}

    def identity(self):
        return {"task_id": self.task_id,
                "task_file": None if self.kind == "scene" else str(self.loaded.path),
                "scene_file": str(self.scene.path), "source_file": str(self.scene.source_path),
                "question_file": str(getattr(self.loaded, "question_path", self.loaded.source_path)),
                "variant": self.loaded.variant, "seed": self.loaded.seed,
                "scene": self.loaded.scene_name, "room": self.loaded.room_name,
                "roles": {"targets": list(self.targets), "containers": list(self.containers),
                          "supports": list(self.supports), "object_supports": self.object_supports,
                          "bindings": {key: value.model_dump(mode="json") for key, value in self.bindings.items()}},
                "capabilities": list(self.capabilities), "goal": self.goal,
                "dependencies": list(self.dependencies), "required_checks": list(self.required_checks)}


_ADAPTERS: dict[str, Callable[..., TaskContext]] = {}


def register_adapter(kind: str, loader: Callable[..., TaskContext]):
    """Register an explicit YAML kind loader accepting path and load_context options."""
    if kind in _ADAPTERS:
        raise ValueError("A preflight adapter is already registered for " + kind)
    _ADAPTERS[kind] = loader


def _adapt(loaded):
    rigid = isinstance(loaded, LoadedRigidTask)
    scene_only = isinstance(loaded, LoadedScene)
    scene = loaded.scene if rigid else loaded
    definition = scene.definition
    bindings = dict(definition.bindings if rigid or scene_only else definition.private.bindings)
    qualification = definition.qualification
    supports = tuple(dict.fromkeys(value for value in
        (getattr(qualification, "support", None), *getattr(qualification, "object_supports", {}).values()) if value))
    if rigid or scene_only:
        targets = tuple(loaded.definition.targets if rigid else definition.target_bindings or (definition.target,))
        evaluation = {"type": "answer_and_place", "targets": list(targets)} if rigid else {}
        goal = "answer_and_place" if rigid else None
        capabilities = RIGID_CAPABILITIES if rigid else ()
    else:
        value = definition.private.evaluation
        evaluation = value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)
        goal = evaluation.get("type")
        targets = tuple(evaluation.get("targets") or ([evaluation["target"]] if evaluation.get("target") else []))
        capabilities = tuple(definition.requirements.capabilities)
    raw, metadata = read_source(scene.source_path)
    objects, _ = metadata_objects(raw, metadata)
    container_names = {name for name, item in objects.items() if item.get("role") == "container"}
    containers = tuple(key for key, value in bindings.items()
                       if value.name in container_names or key == evaluation.get("container"))
    dependencies = []
    for target in targets:
        if goal in ("open_extract_lift", "slide_extract_lift"):
            dependencies.append({"before": "open_joint" if goal == "open_extract_lift" else "slide_joint",
                                 "after": "extract", "target": target, "container": evaluation.get("container"),
                                 "joint": evaluation.get("hinge_joint") or evaluation.get("slide_joint")})
        if goal in KNOWN_GOALS:
            dependencies.append({"before": "grasp", "after": "lift", "target": target})
        if goal == "answer_and_place":
            dependencies.append({"before": "lift", "after": "insert_release_verify", "target": target,
                                 "candidate_containers": list(containers)})
    return TaskContext(loaded, scene, "rigid_containment" if rigid else "scene" if scene_only else "task",
                       bindings, targets, containers, supports, capabilities, goal, evaluation,
                       tuple(dependencies), ("static", "physics") if scene_only else GROUPS)


def _load_unknown_task(path, value, *, variant, seed, data_root):
    """Validate an opaque future evaluator while retaining the common closed schema."""
    private_model = create_model("DiagnosticOpaquePrivate", __base__=ClosedModel,
                                 bindings=(dict[str, Binding], ...), evaluation=(dict[str, Any], ...))
    fields = {name: (field.annotation, field) for name, field in TaskDefinition.model_fields.items()}
    fields["private"] = (private_model, ...)
    model = create_model("DiagnosticOpaqueTask", __base__=ClosedModel, **fields)
    definition = model.model_validate(value)
    chosen_variant = variant or "original"
    if chosen_variant not in definition.scene.variants:
        raise ValueError("Scene variant is not declared by this task.")
    root = Path(data_root or DATA_ROOT).expanduser().resolve()
    source = (root / definition.source.json_path).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("ESI source must exist inside the selected data root.")
    raw, _ = read_source(source)
    scene_name, room_name = Path(definition.source.json_path).parts[-3:-1]
    if (raw.get("scene"), raw.get("room")) != (scene_name, room_name):
        raise ValueError("ESI source does not match its declared scene/room hierarchy.")
    chosen_seed = definition.defaults.seed if seed is None else seed
    if type(chosen_seed) is not int or chosen_seed < 0:
        raise ValueError("Scene seed must be a nonnegative integer.")
    return LoadedTask(definition, path, source, sha256(path), sha256(source), chosen_variant,
                      chosen_seed, scene_name, room_name)


def load_context(path, *, variant=None, seed=None, data_root=None):
    path = select_task(str(path))
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError("Task or scene YAML must be a mapping.")
    kind = value.get("kind", "task")
    if kind in _ADAPTERS:
        return _ADAPTERS[kind](path, variant=variant, seed=seed, data_root=data_root)
    if kind == "scene_patch":
        return _adapt(load_scene(path, variant=variant or "tabletop", seed=seed, data_root=data_root))
    evaluation = value.get("private", {}).get("evaluation", {})
    if (kind == "task" and isinstance(evaluation, dict) and isinstance(evaluation.get("type"), str)
            and evaluation["type"].strip() and evaluation["type"] not in KNOWN_GOALS[:3]):
        return _adapt(_load_unknown_task(path, value, variant=variant, seed=seed, data_root=data_root))
    return _adapt(load_task(path, variant=variant, seed=seed, data_root=data_root))

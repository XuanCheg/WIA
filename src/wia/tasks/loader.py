"""Discover and resolve tasks. Run src/wia/runtime/commands/run_esi_explore_episode.py --task-file FILE."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import subprocess

import yaml

from .schema import CategoryTaskDefinition, RigidContainmentTaskDefinition, SceneDefinition, TaskDefinition
from .sources import resolve_source

from wia.resources import ROOT, ESI_ROOT
TASK_ROOT = ROOT / "explore_to_achieve/wia_v1/release/tasks"
SCENE_ROOT = ROOT / "explore_to_achieve/scenes"
DATA_ROOT = ESI_ROOT / 'dataset/json_clean'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UniqueLoader(yaml.SafeLoader):
    """Reject duplicate YAML keys instead of silently overriding task fields."""


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError("Duplicate task YAML keys are not accepted.")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def discover_tasks(task_root=TASK_ROOT):
    root = Path(task_root).resolve()
    return tuple(sorted(set(root.rglob("*.yaml")) | set(root.rglob("*.yml"))))


def task_definition(value):
    model = {'rigid_containment_task': RigidContainmentTaskDefinition,
             'category_robot_task': CategoryTaskDefinition}.get(value.get('kind'), TaskDefinition)
    return model.model_validate(value)


def discover_scenes(scene_root=SCENE_ROOT, *, include_pending=False, include_discarded=False):
    """Discover ready scenes, with explicit opt-ins for pending and discarded records."""
    root = Path(scene_root).resolve()
    paths = tuple(sorted(set(root.rglob("*.yaml")) | set(root.rglob("*.yml"))))
    statuses = {"ready"}
    if include_pending:
        statuses.add("pending")
    if include_discarded:
        statuses.add("discarded")
    return tuple(path for path in paths
                 if yaml.load(path.read_text(), Loader=UniqueLoader).get("status", "ready") in statuses)


def select_task(selection, *, task_root=TASK_ROOT):
    requested = Path(selection).expanduser()
    if requested.is_file():
        return requested.resolve()
    root = Path(task_root).resolve()
    canonical = (root / requested).resolve()
    if canonical.is_relative_to(root) and canonical.is_file():
        return canonical
    matches = []
    for path in discover_tasks(root):
        definition = task_definition(yaml.load(path.read_text(), Loader=UniqueLoader))
        if selection == definition.id or str(requested) == str(path.relative_to(root).with_suffix("")):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError("Task selector is missing or ambiguous; provide an exact task file or canonical path.")
    return matches[0]


@dataclass(frozen=True)
class LoadedTask:
    definition: TaskDefinition | RigidContainmentTaskDefinition
    path: Path
    source_path: Path
    file_sha256: str
    source_sha256: str
    variant: str
    seed: int
    scene_name: str
    room_name: str
    run_mode: str = field(default="benchmark", kw_only=True)

    @property
    def release_v1(self):
        return is_release_v1(self.path)

    @property
    def task_text(self):
        return self.definition.public.task_text()

    @property
    def public_payload(self):
        return {"task_id": self.definition.id, "task_version": self.definition.version,
                **self.definition.public.model_dump(mode="json")}

    def verify_unchanged(self):
        if sha256(self.path) != self.file_sha256 or sha256(self.source_path) != self.source_sha256:
            raise RuntimeError("Task or ESI source changed after episode creation.")

    def manifest(self, episode_id):
        self.verify_unchanged()
        commit = subprocess.check_output(["git", "-C", str(ESI_ROOT), "rev-parse", "HEAD"], text=True).strip()
        public = self.public_payload
        temporal = {}
        evaluation = getattr(getattr(self.definition, 'private', None), 'evaluation', None)
        if getattr(evaluation, 'operation', None) == 'observe_two_phase_change':
            from .temporal_policy import survey_policy
            temporal = {'temporal_policy': survey_policy(evaluation.parameters)}
        return {**temporal, **({"run_mode": self.run_mode} if self.run_mode == "diagnostic" else {}),
                "schema": "declarative_episode_v1", "episode_id": episode_id,
                "task_id": self.definition.id, "task_version": self.definition.version,
                "task_file": str(self.path), "task_file_sha256": self.file_sha256,
                "esi_source_path": str(self.source_path), "esi_source_sha256": self.source_sha256,
                "esi_submodule_commit": commit, "scene_seed": self.seed, "scene_variant": self.variant,
                "scene": self.scene_name, "room": self.room_name, "public_task": public,
                "public_task_sha256": hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()}


def source_scene_room(relative_path, raw):
    """Resolve literal corpus placeholders from immutable source JSON only."""
    hierarchy = Path(relative_path).parts[-3:-1]
    if len(hierarchy) != 2:
        raise ValueError('ESI source must declare a scene/room hierarchy.')
    resolved = []
    for field, declared in zip(('scene', 'room'), hierarchy):
        actual = raw.get(field)
        if not isinstance(actual, str) or not actual.strip():
            raise ValueError('ESI source lacks its actual scene/room identity.')
        if declared != actual and declared != 'unknown_' + field:
            raise ValueError('ESI sample does not match its declared scene/room hierarchy.')
        resolved.append(actual)
    return tuple(resolved)


def is_release_v1(path):
    """Identify a frozen v1 definition by explicit catalogue membership, without physics."""
    path = Path(path).resolve()
    release = ROOT / 'explore_to_achieve/wia_v1/release'
    if not path.is_relative_to(release):
        return False
    catalogue = json.loads((release / 'catalogue.json').read_text())
    return any(path == (ROOT / row[key]).resolve() for row in catalogue['tasks']
               for key in ('task_file', 'scene_file'))


def load_task(selection, *, task_root=TASK_ROOT, data_root=None, variant=None, seed=None, run_mode="benchmark"):
    if run_mode not in ("benchmark", "diagnostic"):
        raise ValueError("Unknown execution mode.")
    path = select_task(str(selection), task_root=task_root)
    definition = task_definition(yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader))
    if isinstance(definition, CategoryTaskDefinition) and definition.status != 'ready' and run_mode != 'diagnostic' and not is_release_v1(path):
        raise ValueError('Category task is pending/discarded; native execution requires qualification and review.')
    if isinstance(definition, RigidContainmentTaskDefinition):
        return load_rigid_task(path, definition, data_root=data_root, variant=variant, seed=seed, run_mode=run_mode)
    variant = variant or "original"
    if variant not in definition.scene.variants:
        raise ValueError("Scene variant is not declared by this task.")
    source = resolve_source(definition.source.json_path, data_root=data_root)
    raw = json.loads(source.read_text(encoding="utf-8"))
    scene_name, room_name = source_scene_room(definition.source.json_path, raw)
    # The source answer and encoded QA metadata remain private world-construction input.
    selected_seed = definition.defaults.seed if seed is None else seed
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("Scene seed must be a nonnegative integer.")
    return LoadedTask(definition, path, source, sha256(path), sha256(source), variant,
                      selected_seed, scene_name, room_name, run_mode=run_mode)


@dataclass(frozen=True)
class LoadedScene:
    definition: SceneDefinition
    path: Path
    source_path: Path
    file_sha256: str
    source_sha256: str
    variant: str
    seed: int
    scene_name: str
    room_name: str
    run_mode: str = field(default="benchmark", kw_only=True)

    @property
    def release_v1(self):
        return is_release_v1(self.path)

    @property
    def task_text(self):
        """Expose the original question alone; source answers remain private."""
        question = json.loads(self.source_path.read_text()).get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Scene source must contain its original question.")
        return question

    @property
    def public_payload(self):
        return {"task_id": self.definition.id, "task_version": self.definition.version,
                "objective": self.task_text, "success_criteria": "Answer the original question using observed evidence.",
                "environment": "Explore the current scene using the robot's authorized sensors and controls.",
                "constraints": []}

    def verify_unchanged(self):
        if sha256(self.path) != self.file_sha256 or sha256(self.source_path) != self.source_sha256:
            raise RuntimeError("Scene YAML or source changed after initialization selection.")


def load_scene(selection, *, data_root=None, variant="tabletop", seed=None, run_mode="benchmark"):
    """Load an exact scene-only YAML path without creating an objective/evaluator."""
    if run_mode not in ("benchmark", "diagnostic"):
        raise ValueError("Unknown execution mode.")
    path = Path(selection).expanduser().resolve()
    value = yaml.load(path.read_text(), Loader=UniqueLoader)
    if value.get("status") == "pending" and run_mode != "diagnostic" and not is_release_v1(path):
        raise ValueError("Scene is pending and must be skipped until conversion is approved: " + str(path))
    if value.get("status") == "discarded" and run_mode != "diagnostic" and not is_release_v1(path):
        raise ValueError("Scene is discarded and cannot run: " + str(path))
    definition = SceneDefinition.model_validate(value)
    if variant not in definition.scene.variants:
        raise ValueError("Scene variant is not declared by this scene YAML.")
    default_root = SCENE_ROOT if path.is_relative_to(SCENE_ROOT.resolve()) else DATA_ROOT
    root = Path(data_root or default_root).expanduser().resolve()
    source = (root / definition.source.json_path).resolve()
    if data_root is None and not source.is_file():
        source = resolve_source(definition.source.json_path)
        root = source.parents[len(Path(definition.source.json_path).parts) - 1]
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("Scene source must be an existing JSON inside the data root.")
    raw = json.loads(source.read_text())
    scene_name, room_name = source_scene_room(definition.source.json_path, raw)
    selected_seed = definition.seed if seed is None else seed
    if type(selected_seed) is not int or selected_seed < 0:
        raise ValueError("Scene seed must be nonnegative.")
    for value in (definition.runtime.trunk_config, definition.runtime.robot_camera_config,
                  definition.runtime.viewer_config):
        if not (ROOT / value).is_file():
            raise ValueError("A scene runtime configuration file is missing: " + value)
    return LoadedScene(definition, path, source, sha256(path), sha256(source), variant,
                       selected_seed, scene_name, room_name, run_mode=run_mode)


@dataclass(frozen=True)
class LoadedRigidTask(LoadedTask):
    """Use canonical geometry and the derivative question as separate inputs."""
    scene: LoadedScene
    question_path: Path
    question_sha256: str
    protocol_sha256: str

    @property
    def public_payload(self):
        from .rigid_qa import protocol
        raw = json.loads(self.question_path.read_text())
        metadata = raw["metadata_json"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        original = protocol.build_system_prompt(metadata, .8, 3).splitlines()
        confidence = original[original.index("CONFIDENCE RULES:"):original.index("Output EXACTLY one valid JSON object and nothing else:")]
        guidance = original[original.index("Guidance:"):]
        rules = "\n".join(confidence + guidance)
        return {**super().public_payload, "constraints": [*self.definition.public.constraints,
            "Apply the following original ESI fit and confidence rules separately to every named object:\n" + rules,
            "Use real R1 motions and gripper commands for each pickup and insertion. "
            "There is no mock grasp or automatic placement sampler. No initial answer is required. "
            "Before DONE, complete every object subtask or report why further progress is impossible. "
            "In the final world_observation, give a container answer, confidence and observed placement outcome "
            "for each object. Report failures honestly."]}

    @property
    def release_v1(self):
        return is_release_v1(self.path)

    @property
    def task_text(self):
        from .schema import PublicTask
        public = self.public_payload
        return PublicTask.model_validate({key: public[key] for key in
            ("objective", "success_criteria", "environment", "constraints")}).task_text()

    def verify_unchanged(self):
        from .rigid_qa import PROTOCOL_PATH
        super().verify_unchanged()
        self.scene.verify_unchanged()
        if sha256(self.question_path) != self.question_sha256 or sha256(PROTOCOL_PATH) != self.protocol_sha256:
            raise RuntimeError("Original containment question or shared protocol changed after selection.")

    def manifest(self, episode_id):
        return {**super().manifest(episode_id), "scene_file": str(self.scene.path),
            "scene_file_sha256": self.scene.file_sha256, "question_source_path": str(self.question_path),
            "question_source_sha256": self.question_sha256, "containment_protocol_sha256": self.protocol_sha256,
            "target_bindings": list(self.definition.targets), "physical_evaluator": "not_configured"}


def load_rigid_task(path, definition, *, data_root=None, variant=None, seed=None, run_mode="benchmark"):
    from .rigid_qa import PROTOCOL_PATH
    if definition.status != "ready":
        raise ValueError("Containment task is " + definition.status + " and must be skipped: " + str(path))
    if variant not in (None, "tabletop"):
        raise ValueError("Containment tasks require their approved tabletop layout.")
    root = Path(data_root or SCENE_ROOT).expanduser().resolve()
    question_path = (root / definition.source.json_path).resolve()
    scene_path = (root / definition.scene_file).resolve()
    if not question_path.is_relative_to(root) or not question_path.is_file() or not scene_path.is_relative_to(root):
        raise ValueError("Containment question and scene must be inside the selected scene root.")
    scene = load_scene(scene_path, data_root=root, variant="tabletop", seed=definition.defaults.seed if seed is None else seed)
    raw = json.loads(question_path.read_text())
    if (raw.get("small_task") != "Rigid Containment" or
            (raw.get("scene"), raw.get("room")) != (scene.scene_name, scene.room_name)):
        raise ValueError("Containment question must belong to the same scene and room as its layout.")
    if raw.get("question") != definition.public.objective:
        raise ValueError("Containment objective must preserve the original question verbatim.")
    scene_targets = scene.definition.target_bindings or (scene.definition.target,)
    if tuple(definition.targets) != scene_targets:
        raise ValueError("Task targets must match all generated scene targets in order.")
    return LoadedRigidTask(definition, path, scene.source_path, sha256(path), scene.source_sha256,
        scene.variant, scene.seed, scene.scene_name, scene.room_name, scene, question_path,
        sha256(question_path), sha256(PROTOCOL_PATH), run_mode=run_mode)


def verify_manifest(path, task):
    """Verify immutable task identity and policy independently in both processes."""
    task.verify_unchanged()
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("episode_id"), str) or not value["episode_id"].strip():
        raise RuntimeError("An immutable episode manifest must include its episode identity.")
    expected = task.manifest(value["episode_id"])
    if any(value.get(key) != item for key, item in expected.items()):
        raise RuntimeError("Episode manifest does not match this exact task, source and scene selection.")
    policy_keys = {"max_steps", "world_mode", "world_advances_during_inference", "observation_condition",
                   "action_mode", "model_name", "reasoning_effort", "wall_timeout_s", "agent_session"}
    optional_keys = {"world_observation_log", "run_mode", "backend_metadata"} & set(value)
    if value.get("run_mode", "benchmark") != task.run_mode:
        raise RuntimeError("Episode execution mode differs from the loaded task.")
    sessions = {
        "native_persistent": ({"codex", "codex_cli"}, "native"),
        "local_http_persistent": ({"local_http"}, "harness"),
        "openai_chat_completions_persistent": ({"openai_chat_completions"}, "harness"),
        "openai_responses_persistent": ({"openai_responses"}, "harness"),
    }
    if (set(value) != set(expected) | policy_keys | optional_keys
            or value["agent_session"] not in sessions):
        raise RuntimeError("Unsupported declarative episode manifest fields or session policy.")
    if "backend_metadata" in value:
        metadata = value["backend_metadata"]
        backends, history_owner = sessions[value["agent_session"]]
        if (not isinstance(metadata, dict)
                or metadata.get("backend") not in backends
                or metadata.get("history_owner") != history_owner):
            raise RuntimeError("Backend metadata does not match the episode session and history owner.")
    reasoning = value["reasoning_effort"]
    reasoning_valid = (reasoning in ("none", "low", "medium", "high", "xhigh", "max")
                       or reasoning is None and value["agent_session"] != "native_persistent")
    if (type(value.get("world_observation_log", True)) is not bool
            or value["action_mode"] not in ("fixed", "parameterized")
            or value["observation_condition"] not in ("current", "initial_view_only")
            or value["world_mode"] not in ("run", "pause")
            or type(value["world_advances_during_inference"]) is not bool
            or value["world_advances_during_inference"] != (value["world_mode"] == "run")
            or type(value["max_steps"]) is not int or value["max_steps"] < 1
            or not isinstance(value["model_name"], str) or not value["model_name"].strip()
            or not reasoning_valid
            or value["wall_timeout_s"] is not None and (type(value["wall_timeout_s"]) not in (int, float)
                or not math.isfinite(value["wall_timeout_s"]) or value["wall_timeout_s"] <= 0)):
        raise RuntimeError("Declarative episode policy is inconsistent.")
    if 'temporal_policy' in value:
        budget = value['temporal_policy']['phase_1_max_decisions']
        if budget >= value['max_steps']:
            raise ValueError('Phase-1 maximum must leave at least one Phase-2 decision within the total budget.')
    return value


def public_episode_contract(manifest, manifest_sha256):
    return {key: manifest[key] for key in ("episode_id", "task_id", "task_version", "task_file_sha256",
        "public_task_sha256", "action_mode", "world_mode", "world_advances_during_inference",
        "observation_condition", "max_steps")} | {"manifest_sha256": manifest_sha256, "run_mode": manifest.get("run_mode", "benchmark")}

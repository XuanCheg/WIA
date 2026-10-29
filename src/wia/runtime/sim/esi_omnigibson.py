"""ESI scene construction and existing R1 observations, without a policy loop.

Run inside the BEHAVIOR / OmniGibson environment. Simulator and ESI imports are
delayed until reset(); importing this module does not start Isaac Sim.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import importlib
import os
import subprocess
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SHOWHARNESS_ROOT = WIA_RUNTIME_ROOT
from wia.resources import ESI_ROOT as DEFAULT_ESI_ROOT
DEFAULT_QUESTION = DEFAULT_ESI_ROOT / (
    "dataset/json_clean/Action Sequencing/Action Order Inference/"
    "Beechwood_0_garden/dining_room_0/Beechwood_0_garden__dining_room_0.json"
)


class NativeCameraNotReady(RuntimeError):
    """A transient empty native annotator buffer, before a valid render arrives."""


def _import_pipeline(esi_root: Path):
    """Localize the upstream script-style imports (tasks, models, utils)."""
    src = (Path(esi_root).expanduser().resolve() / "src").resolve()
    active = src / "active_explore"
    pipeline_file = active / "pipeline.py"
    if not pipeline_file.is_file():
        raise FileNotFoundError(
            f"ESI pipeline missing at {active}; run git submodule update --init."
        )
    cached = sys.modules.get("active_explore.pipeline")
    if cached is not None:
        cached_file = getattr(cached, "__file__", None)
        if not cached_file or Path(cached_file).resolve() != pipeline_file.resolve():
            raise RuntimeError("Cached ESI pipeline does not match the requested pipeline file.")
    # Kit may later extend its generic models namespace. The verified cached
    # pipeline already owns its imported model classes; no model reimport occurs.
    namespaces = ("active_explore", "tasks", "utils") if cached is not None else (
        "active_explore", "tasks", "models", "utils")
    for name in namespaces:
        module = sys.modules.get(name)
        locations = list(getattr(module, "__path__", []))
        if getattr(module, "__file__", None):
            locations.append(module.__file__)
        if locations and any(not Path(p).resolve().is_relative_to(src) for p in locations):
            raise RuntimeError(f"ESI import conflicts with already imported module {name!r}.")
    if cached is not None:
        return cached
    for path in (src, active):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    return importlib.import_module("active_explore.pipeline")


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ESISession:
    """One ESI-initialized environment; RGB observations and separate debug metadata.

    reset() builds the scene once, including upstream postprocessing. It does not
    promise repeated env.reset() semantics for scenes modified after construction.
    close() owns the process-wide OmniGibson runtime; use one session per process.
    """

    def __init__(
        self,
        metadata: str | Path = DEFAULT_QUESTION,
        *,
        task: str | None = None,
        question_index: int = 0,
        json_root: str | Path | None = None,
        esi_root: str | Path = DEFAULT_ESI_ROOT,
        camera: str | None = None,
        robot_position: tuple[float, float, float] | None = None,
        robot_yaw_deg: float | None = None,
        camera_size: tuple[int, int] | None = None,
        head_pitch_deg: float = 0.0,
        grasp_mode: str | None = None,
        require_robot: bool = True,
    ) -> None:
        self.metadata_path = Path(metadata).expanduser().resolve()
        import uuid
        self.native_reset_id = uuid.uuid4().hex
        self.esi_root = Path(esi_root).expanduser().resolve()
        self.json_root = Path(json_root).expanduser().resolve() if json_root else None
        self.task_name = task
        self.question_index = question_index
        self.camera_name = camera
        if robot_position is not None:
            position = np.asarray(robot_position, dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("robot_position must contain three finite world coordinates.")
            robot_position = tuple(position.tolist())
        if robot_yaw_deg is not None and not np.isfinite(robot_yaw_deg):
            raise ValueError("robot_yaw_deg must be finite.")
        if not np.isfinite(head_pitch_deg) or abs(head_pitch_deg) > 90:
            raise ValueError("head_pitch_deg must be finite and in [-90, 90].")
        if camera_size is not None and (
            len(camera_size) != 2 or any(
                not isinstance(size, (int, np.integer)) or size <= 0 for size in camera_size
            )
        ):
            raise ValueError("camera_size must contain two positive integers (width, height).")
        self.robot_position = robot_position
        self.robot_yaw_deg = robot_yaw_deg
        self.camera_size = camera_size
        self.head_pitch_deg = head_pitch_deg
        if grasp_mode not in (None, "assisted"):
            raise ValueError("Only an explicit assisted grasp override is supported.")
        self.grasp_mode = grasp_mode
        self.require_robot = require_robot
        self.original_grasp_mode = None
        self.env = self.robot = self.pipeline = self.task_module = None
        self.scene_camera = None
        self.scene_camera_framing = "esi_source_postprocess"
        self.global_cameras = {}
        self.robot_camera_registry = {}
        self.task_state: dict[str, Any] = {}
        self.payload: dict[str, Any] = {}
        self.closed = False
        self._observation_source = None
        self.reuse_step_observations = False
        self._step_camera_data = None
        self._action_execution_depth = 0
        self._physical_step_generation = 0

    def step_control(self, action):
        """Run native control/task lifecycle without rendering or sensor images."""
        from wia.runtime.sim.esi_rendering import step_without_observation
        return step_without_observation(self, action)

    def action_execution(self):
        from wia.runtime.sim.esi_rendering import action_execution
        return action_execution(self)

    @property
    def action_executing(self):
        return self._action_execution_depth > 0

    def invalidate_camera_cache(self):
        self._step_camera_data = None
        self._action_boundary_ready = False

    def reset(self) -> None:
        if self.closed or self.pipeline is not None:
            raise RuntimeError("ESISession initializes once; create a fresh process for another run.")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Question/metadata file not found: {self.metadata_path}")
        if self.question_index < 0:
            raise ValueError("question_index must be nonnegative.")
        self.pipeline = pipeline = _import_pipeline(self.esi_root)
        from wia.runtime.sim.esi_loading import native_loading_context

        with native_loading_context():
            self._initialize_scene(pipeline)

    @property
    def initial_overview_identity(self):
        adapter = getattr(self, 'scene_adapter', None)
        seed = int(getattr(getattr(adapter, 'task', None), 'seed', 0))
        return {'episode_id': str(getattr(self, 'episode_id', 'reset_' + self.native_reset_id)),
            'reset_id': self.native_reset_id, 'seed': seed}

    def initial_overview_capture(self, *, snapshot_id, captured_at):
        from wia.runtime.sim.esi_initial_overview import capture_initial_overview
        return capture_initial_overview(self, snapshot_id=snapshot_id, captured_at=captured_at)

    def _initialize_scene(self, pipeline) -> None:
        """Construct and settle one scene using the selected native pipeline."""
        selected = pipeline.select_question(
            self.metadata_path, self.question_index, self.json_root
        )
        self.question_path, self.payload = selected.source_path, selected.payload
        try:
            self.task_module = module = pipeline.load_task_module_for_payload(
                self.task_name or "", self.payload
            )
        except (ImportError, ValueError) as exc:
            raise RuntimeError(
                "Cannot resolve ESI task; use a canonical question JSON or provide --task."
            ) from exc
        config = SimpleNamespace(json_root=self.json_root, robot="R1")
        self.task_state = pipeline.task_preprocess(module, self.payload, self.question_path, config)
        if self.task_state.get("skip_reason"):
            raise RuntimeError(f"ESI preprocess skipped question: {self.task_state['skip_reason']}")
        self.scene_name, self.room_name = module.scene_room(self.payload)
        self.scene_camera_position, self.scene_camera_quaternion, self.camera_info = module.initial_camera(
            self.payload
        )
        objects = module.build_env_objects(self.payload)
        self.task_object_names = tuple(obj["name"] for obj in objects if "name" in obj)
        env_config = pipeline.task_build_env_config(
            module, self.scene_name, self.room_name, "R1", objects,
            full_scene=bool(getattr(module, "FULL_SCENE", False)), payload=self.payload,
        )
        explore = getattr(self, "explore_scene", None)
        if explore is not None:
            explore.apply_to_env_config(env_config)
        scene_adapter = getattr(self, "scene_adapter", None)
        if scene_adapter is not None:
            scene_adapter.apply_to_env_config(env_config)
        # Category overlays may add the installed R1 to a robot-free source.
        # Configure observations only after that initialization is complete.
        self._configure_robot_observation(env_config)
        from wia.runtime.sim.esi_reflection import configure_embodiment
        configure_embodiment(self, env_config)
        print(f"[esi] question: {self.question_path}\n[esi] task: {module.__name__}", flush=True)
        print(f"[esi] scene: {self.scene_name}\n[esi] room: {self.room_name}", flush=True)
        print("[esi] creating OmniGibson environment...", flush=True)
        self.env = pipeline.og.Environment(configs=env_config)
        # The installed OmniGibson learning/eval.py documents at least three
        # renders to flush physical / USD changes through camera annotators.
        # The native diagnostic found a few depth-edge pixels still settling
        # on the third update, so include a fourth graphics-only update.
        # Canonical capture also waits for render completion and checks the
        # rendered camera pose before publishing RGB and depth together.
        self._native_boundary_graphics_iterations = 4
        self._native_camera_freshness_required = True
        if self.env.scene.scene_model != self.scene_name:
            raise RuntimeError(f"Loaded scene {self.env.scene.scene_model!r} != {self.scene_name!r}.")
        # An explicit reversible overlay may restore only the native head mount
        # after a source task moves it. Capture the actual parent-frame mount,
        # without guessing a robot pose or altering the source reference view.
        if getattr(scene_adapter, 'restore_robot_head_mount', False):
            self._category_sensor_mounts = {}
            for robot in self.env.robots:
                for name, sensor in robot.sensors.items():
                    if any(part in name.lower() for part in ('head', 'eyes')):
                        position, quaternion = sensor.get_position_orientation(frame='parent')
                        self._category_sensor_mounts[name] = (_array(position).tolist(), _array(quaternion).tolist())
        observer = getattr(self, "_initialization_observer", None)
        if observer is not None:
            observer(self, env_config)
        stage_observer = getattr(self, '_initialization_stage_observer', None)
        if stage_observer is not None:
            stage_observer(self, 'constructed', env_config)
        from wia.runtime.sim.esi_initial_posture import initialize_controller_posture
        posture = initialize_controller_posture(self, scene_adapter)
        if posture is not None and stage_observer is not None:
            stage_observer(self, 'controller_posture_initialized', env_config)
        if (scene_adapter is not None and not getattr(self, 'reflection_sensor_profile', None)
                and any(item.get('operation') == 'prepare_category_interaction'
                        for item in getattr(scene_adapter, '_realized', ()))):
            from wia.tasks.interaction_setup import initialize_loaded_room_start
            initialize_loaded_room_start(self)
            if stage_observer is not None:
                stage_observer(self, 'loaded_room_start', env_config)
        self.settle_steps = pipeline.task_initial_settle_steps(module)
        for _ in range(self.settle_steps):
            pipeline.og.sim.step()
        print("[esi] settle complete", flush=True)
        if stage_observer is not None:
            stage_observer(self, 'settled', env_config)
        override = getattr(scene_adapter, 'source_postprocess_override', None)
        self.postprocess = override(self) if callable(override) else None
        if self.postprocess is None:
            self.postprocess = pipeline.task_postprocess_env(
                module, self.env, self.payload, self.camera_info, self.task_state
            )
        print("[esi] postprocess complete", flush=True)
        if stage_observer is not None:
            stage_observer(self, 'source_postprocessed', env_config)
        # Initializers operate on the robot retained by source postprocessing.
        # Bind it before destination/support hooks read its actual chassis.
        self.robot = self.env.robots[0] if self.env.robots else None
        if scene_adapter is not None:
            scene_adapter.apply_after_postprocessing(self)
        if stage_observer is not None:
            stage_observer(self, 'adapted', env_config)
        self._refresh_task_object_names()
        if not self.env.robots:
            if not self.require_robot:
                self.rgb_cameras = []
                self.camera_name = self.wrist_camera_name = None
                for _ in range(3):
                    pipeline.og.sim.render()
                return
            raise RuntimeError("ESI environment has no robot; inspect this task's environment config.")
        self.robot = self.env.robots[0]
        if type(self.robot).__name__ != "R1":
            raise RuntimeError(f"Expected existing R1, found {type(self.robot).__name__}.")
        if self.robot.action_dim <= 0 or not self.robot.controllers:
            raise RuntimeError("R1 has no usable action dimension/controllers.")
        self.rgb_cameras = [
            name for name, sensor in self.robot.sensors.items() if "rgb" in sensor.modalities
        ]
        if not self.rgb_cameras:
            raise RuntimeError(f"No RGB robot camera found; sensors: {list(self.robot.sensors)}")
        if self.camera_name is None:
            # Runtime discovery: the installed R1 names its main mount "eyes"
            # and its wrist mounts "*_eef_link", rather than "head"/"wrist".
            head = [name for name in self.rgb_cameras
                    if any(part in name.lower() for part in ("head", "eyes"))]
            main = [name for name in self.rgb_cameras
                    if not any(part in name.lower() for part in ("wrist", "eef", "gripper"))]
            candidates = head or main or self.rgb_cameras
            if len(candidates) != 1:
                raise RuntimeError(f"Ambiguous RGB cameras {candidates}; select one with --camera.")
            self.camera_name = candidates[0]
        if self.camera_name not in self.rgb_cameras:
            raise RuntimeError(f"RGB camera {self.camera_name!r} not found; choose {self.rgb_cameras}.")
        wrists = [name for name in self.rgb_cameras
                  if any(part in name.lower() for part in ("wrist", "eef"))]
        wrists.sort(key=lambda name: ("left" not in name.lower(), name))
        self.wrist_camera_name = wrists[0] if wrists else None
        if self.head_pitch_deg:
            from scipy.spatial.transform import Rotation

            head = self.robot.sensors[self.camera_name]
            _, local_quat = head.get_position_orientation(frame="parent")
            # Optical -Z points forward, +Y up: negative local-X rotation looks
            # down. Keep the original sensor position and its robot-link parent.
            pitched = Rotation.from_quat(_array(local_quat)) * Rotation.from_euler(
                "x", -self.head_pitch_deg, degrees=True
            )
            head.set_position_orientation(orientation=pitched.as_quat().tolist(), frame="parent")
        # Render only: warm up RGB without issuing any robot action or moving the viewer.
        for _ in range(3):
            pipeline.og.sim.render()

    def _refresh_task_object_names(self) -> None:
        """Include source objects created during postprocessing in human framing."""
        names = list(self.task_object_names)
        names.extend(self.task_state.get('dynamic_object_names', ()))
        names.extend(self.postprocess.get('dynamic_object_names', ()))
        mirror = self.postprocess.get('mirror_name')
        if mirror:
            names.append(mirror)
        explicit = []
        for patch in getattr(getattr(self, 'scene_adapter', None), 'applied', ()):
            evidence = patch.get('initialization_evidence') or {}
            if 'human_framing_names' in evidence:
                selected = evidence['human_framing_names']
                if (not isinstance(selected, list)
                        or any(not isinstance(name, str) or not name for name in selected)):
                    raise ValueError('Category human framing requires explicit original object names.')
                explicit.append(selected)
        if explicit:
            # A phase initializer knows which original objects are active.
            # Inactive offstage preloads must not pull human cameras away.
            names = [name for selected in explicit for name in selected]
        self.task_object_names = tuple(dict.fromkeys(names))

    def _configure_robot_observation(self, env_config: dict) -> None:
        """Apply explicit initialization/sensor settings to the existing R1 only."""
        if (self.robot_position is None and self.robot_yaw_deg is None
                and self.camera_size is None and self.grasp_mode is None):
            return
        robots = env_config.get("robots", [])
        if not robots and not getattr(self, "require_robot", True):
            return
        if not robots or robots[0].get("type") != "R1":
            raise RuntimeError("Observation setup requires an existing R1 in ESI's robot config.")
        robot = robots[0]
        self.original_grasp_mode = robot.get("grasping_mode")
        if self.grasp_mode is not None:
            robot["grasping_mode"] = self.grasp_mode
            robot["disable_grasp_handling"] = False
        if self.robot_position is not None:
            robot["position"] = list(self.robot_position)
        if self.robot_yaw_deg is not None:
            half_yaw = np.deg2rad(self.robot_yaw_deg) / 2
            robot["orientation"] = [0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw))]
        if self.robot_position is not None or self.robot_yaw_deg is not None:
            robot["pose_frame"] = "world"
        if self.camera_size is not None:
            vision = robot.setdefault("sensor_config", {}).setdefault("VisionSensor", {})
            kwargs = vision.setdefault("sensor_kwargs", {})
            kwargs.update(image_width=self.camera_size[0], image_height=self.camera_size[1])

    def configure_scene_camera(self) -> None:
        """Use ESI's original viewer and initial pose for an explicit debug view."""
        if self.env is None or self.closed:
            raise RuntimeError("Call reset() before configuring the scene camera.")
        pos, quat = self.scene_camera_position, self.scene_camera_quaternion
        override = self.postprocess.get("camera_override")
        if isinstance(override, dict) and override.get("position") and override.get("quaternion_xyzw"):
            pos = np.asarray(override["position"], dtype=float)
            quat = np.asarray(override["quaternion_xyzw"], dtype=float)
        self.pipeline.set_camera(pos, quat)
        self.scene_camera = self.pipeline.og.sim._viewer_camera
        profile = getattr(self, 'reflection_sensor_profile', None)
        if profile is not None:
            self.scene_camera.focal_length = float(self.scene_camera.horizontal_aperture) / (
                2. * np.tan(np.deg2rad(profile['horizontal_fov_deg']) / 2.))
            # The installed viewer can retain its previous RGB projection after
            # a focal-length update even while camera_params reports the new K.
            # Rebind its render product once; preserve the native prim and pose.
            from wia.runtime.sim.esi_robot_cameras import rebind_render_product
            rebind_render_product(self, self.scene_camera)
            for _ in range(12):
                self.pipeline.og.sim.render()
            self.invalidate_camera_cache()

    def configure_human_cameras(self, config, *, include_robot=True) -> None:
        """Create independent user cameras without changing mounted RGB poses."""
        from wia.runtime.sim.esi_debug_cameras import configure_global_cameras
        configure_global_cameras(self, config, include_robot=include_robot)
        adapter = getattr(self, "scene_adapter", None)
        patched = adapter is not None and bool(adapter.variant.patches)
        preserve_reference = adapter is not None and (
            adapter.task.definition.kind == 'category_robot_task'
            or adapter.task.definition.id.startswith('category_'))
        if preserve_reference:
            self.scene_camera_framing = "original_source_reference"
        elif patched or getattr(self, "explore_scene", None) is not None:
            front = self.global_cameras.get("global_front")
            if front is not None and self.scene_camera is not None:
                # Relocated task objects can leave ESI's source viewer behind a
                # room wall. Reframe only the native human viewer from the
                # already configured in-room overview; mounted sensors stay fixed.
                position, quaternion = front.get_position_orientation()
                self.pipeline.set_camera(_array(position), _array(quaternion))
                self.scene_camera_framing = "patched_task_front_overview"
                self.pipeline.og.sim.render()

    def observe(self, *, include_wrist: bool = False, include_scene: bool = False,
                include_sensors: bool = False) -> dict[str, np.ndarray]:
        if self.robot is None or self.closed:
            raise RuntimeError("Call reset() before observe(), and before close().")
        head = self._sensor_data(self.robot.sensors[self.camera_name])
        observation = {"head_rgb": self._rgb_from_data(head, self.camera_name)}
        if include_sensors and "depth_linear" in head:
            observation["head_depth_m"] = _array(head["depth_linear"]).astype(np.float32)
        if include_wrist and self.wrist_camera_name is not None:
            wrist = self._sensor_data(self.robot.sensors[self.wrist_camera_name])
            observation["wrist_rgb"] = self._rgb_from_data(wrist, self.wrist_camera_name)
            if include_sensors and "depth_linear" in wrist:
                observation["wrist_depth_m"] = _array(wrist["depth_linear"]).astype(np.float32)
            for view, sensor_name in self.robot_camera_registry.items():
                if view not in ("agentview_rgb", "wrist_rgb"):
                    observation[view] = self._camera_rgb(sensor_name)
        if include_scene:
            if self.scene_camera is None:
                raise RuntimeError("Configure the ESI scene camera before requesting its debug view.")
            observation["scene_rgb"] = self._sensor_rgb(self.scene_camera)
        return observation

    def _camera_rgb(self, name: str) -> np.ndarray:
        return self._sensor_rgb(self.robot.sensors[name])

    @property
    def observation_source(self):
        """Canonical head/wrist source; it cannot select the privileged scene view."""
        if self.robot is None or self.closed:
            raise RuntimeError("Call reset() before accessing the observation source.")
        if self._observation_source is None:
            from wia.runtime.sim.esi_observations import ESIObservationSource

            head = self.camera_name
            wrist = self.wrist_camera_name
            if (head not in self.robot.sensors or not any(part in head.lower() for part in ("eyes", "head"))
                    or wrist not in self.robot.sensors
                    or not any(part in wrist.lower() for part in ("eef", "wrist"))):
                raise RuntimeError("AgentObservation requires the native R1 head and working wrist cameras.")
            from wia.runtime.sim.esi_base import _main_thread

            _main_thread()
            from core.observation_config import ObservationConfig
            observation_config = getattr(self, "agent_observation_config", ObservationConfig())
            added = False
            for name in (head, wrist):
                sensor = self.robot.sensors[name]
                enabled = observation_config.head_depth if name == head else observation_config.wrist_depth
                if enabled and "depth_linear" in getattr(sensor, "all_modalities", ()) and "depth_linear" not in sensor.modalities:
                    # Enable only measured depth; never request all modalities,
                    # segmentation, bounding boxes or simulator point clouds.
                    sensor.add_modality("depth_linear")
                    added = True
            if added:
                self.invalidate_camera_cache()
                for _ in range(3):
                    self.pipeline.og.sim.render()
            task_text = getattr(self, "public_task_text", None) or self.payload.get("_question") or self.task_module.TASK_NAME
            self._observation_source = ESIObservationSource(self, task_text=task_text, config=observation_config)
        return self._observation_source

    def capture_agent_observation(self):
        """Owner-thread allowed sensors/robot readback, without task ground truth."""
        return self.observation_source.capture(action_boundary=True).observation

    def controller_state(self):
        """Internal readback also available in read-only sessions, without motion."""
        from wia.runtime.sim.esi_state import read_controller_state

        return read_controller_state(self)

    def evaluator_state(self, *, object_names=(), include_scene=False):
        """Explicit privileged API for local evaluation, never Agent input."""
        from core.observations import EvaluatorState
        from wia.runtime.sim.esi_base import _main_thread

        _main_thread()
        if self.robot is None or self.closed:
            raise RuntimeError("Call reset() before evaluator_state().")
        poses = {}
        for name in object_names:
            obj = self.env.scene.object_registry("name", name)
            if obj is None:
                raise ValueError(f"Evaluator object {name!r} does not exist.")
            position, quaternion = obj.get_position_orientation()
            poses[name] = {"position": _array(position).tolist(), "quaternion_xyzw": _array(quaternion).tolist()}
        attachments = {}
        for arm, obj in getattr(self.robot, "_ag_obj_in_hand", {}).items():
            attachments[arm] = {"object_name": obj.name if obj is not None else None,
                                "attached": obj is not None,
                                "constraint_active": getattr(self.robot, "_ag_obj_constraints", {}).get(arm) is not None,
                                "release_pending": getattr(self.robot, "_ag_release_counter", {}).get(arm) is not None}
        scene_state = None
        if include_scene:
            scene_state = {}
            for obj in self.env.scene.objects:
                position, quaternion = obj.get_position_orientation()
                scene_state[obj.name] = {"category": obj.category, "position": _array(position).tolist(),
                                        "quaternion_xyzw": _array(quaternion).tolist()}
        return EvaluatorState(deepcopy(self.payload), deepcopy(self.task_state), poses,
                              grasp_attachment=attachments, scene_state=scene_state)

    def _sensor_data(self, sensor) -> dict:
        if (self.action_executing and not getattr(self, "_boundary_capture_depth", 0)
                and not getattr(self, "_recording_capture_depth", 0)):
            raise RuntimeError("Native sensor images are deferred until the action boundary.")
        cached = self._step_camera_data if self.reuse_step_observations else None
        reading = cached[sensor.name] if cached and sensor.name in cached else sensor.get_obs()
        data = reading[0] if isinstance(reading, tuple) else reading
        if not isinstance(data, dict) or "rgb" not in data:
            raise RuntimeError(f"Camera {sensor.name!r} did not return an RGB observation.")
        return data

    def _sensor_rgb(self, sensor) -> np.ndarray:
        return self._rgb_from_data(self._sensor_data(sensor), sensor.name)

    @staticmethod
    def _rgb_from_data(data, name) -> np.ndarray:
        rgb = _array(data["rgb"])
        if rgb.size == 0:
            raise NativeCameraNotReady(f"Camera {name!r} returned an empty native RGB buffer.")
        if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4) or min(rgb.shape[:2]) <= 0:
            raise ValueError(f"Unexpected camera RGB format: {rgb.shape}; expected H x W x 3/4.")
        rgb = rgb[..., :3]
        if not np.issubdtype(rgb.dtype, np.number) or not np.isfinite(rgb).all():
            raise ValueError("Camera RGB must contain finite numeric values.")
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError("Camera RGB values must lie in [0, 255] (or normalized [0, 1]).")
        if np.issubdtype(rgb.dtype, np.floating) and rgb.max() <= 1:
            rgb = rgb * 255
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    def debug_metadata(self) -> dict[str, Any]:
        """Privileged runtime inspection stays outside the observation API."""
        if self.robot is None or self.closed:
            raise RuntimeError("Call reset() before debug_metadata(), and before close().")
        pos, quat = self.robot.get_position_orientation()
        if not np.isfinite(_array(pos)).all() or not np.isfinite(_array(quat)).all():
            raise RuntimeError("R1 world pose is not finite.")
        indices = self.robot.controller_action_idx
        controllers = {
            name: {
                "type": type(ctrl).__name__,
                "command_dim": int(ctrl.command_dim),
                "action_indices": _array(indices[name]).tolist(),
                "motor_type": getattr(ctrl, "motor_type", None),
                "use_delta_commands": getattr(ctrl, "use_delta_commands", None),
                "dof_indices": _array(ctrl.dof_idx).tolist(),
                "mode": getattr(ctrl, "mode", None),
            }
            for name, ctrl in self.robot.controllers.items()
        }
        cameras = {}
        for name in self.rgb_cameras:
            sensor = self.robot.sensors[name]
            camera_pos, camera_quat = sensor.get_position_orientation()
            local_pos, local_quat = sensor.get_position_orientation(frame="parent")
            cameras[name] = {
                "prim_path": sensor.prim_path,
                "image_size": [int(sensor.image_width), int(sensor.image_height)],
                "world_position": _array(camera_pos).tolist(),
                "world_quaternion_xyzw": _array(camera_quat).tolist(),
                "mount_position": _array(local_pos).tolist(),
                "mount_quaternion_xyzw": _array(local_quat).tolist(),
            }
        scene_camera = None
        if self.scene_camera is not None:
            camera_pos, camera_quat = self.scene_camera.get_position_orientation()
            scene_camera = {
                "name": self.scene_camera.name, "prim_path": self.scene_camera.prim_path,
                "image_size": [int(self.scene_camera.image_width), int(self.scene_camera.image_height)],
                "world_position": _array(camera_pos).tolist(),
                "world_quaternion_xyzw": _array(camera_quat).tolist(),
                "framing": self.scene_camera_framing,
            }
        return {
            "question_path": str(self.question_path),
            "metadata_path": str(self.metadata_path),
            "question_index": self.question_index,
            "question_id": self.payload.get("_hf_id") or self.task_module.question_id(
                self.payload, self.question_path
            ),
            "question": self.payload.get("_question"),
            "task_module": self.task_module.__name__,
            "task": self.task_module.TASK_NAME,
            "scene": self.scene_name, "room": self.room_name,
            "initial_settle_steps": self.settle_steps,
            "postprocess_completed": True, "postprocess_result": self.postprocess,
            "robot_name": self.robot.name, "robot_type": type(self.robot).__name__,
            "robot_world_pose": {"position": _array(pos).tolist(), "quaternion_xyzw": _array(quat).tolist()},
            "action_dim": int(self.robot.action_dim), "controllers": controllers,
            "grasp_mode": self.robot.grasping_mode,
            "original_grasp_mode": self.original_grasp_mode,
            "sensors": list(self.robot.sensors), "rgb_cameras": self.rgb_cameras,
            "camera_selected": self.camera_name,
            "wrist_camera_selected": self.wrist_camera_name,
            "camera_prim_path": self.robot.sensors[self.camera_name].prim_path,
            "cameras": cameras,
            "robot_camera_mounts": getattr(self, "robot_camera_mounts", {}),
            "robot_camera_config": getattr(self, "robot_camera_config", None),
            "scene_camera": scene_camera,
            "global_camera_names": list(self.global_cameras),
            "control_render_policy": {
                "physical_substeps": "unrendered_native_sim_step",
                "intermediate_sensor_images": False,
                "action_observation": "fresh_render_after_final_stop",
                "readiness_flush": "graphics_only_without_physics",
                "fresh_physics_boundary_graphics_iterations": getattr(self, "_native_boundary_graphics_iterations", 4),
                "camera_freshness": getattr(self, "_native_camera_freshness", {"status": "not_captured"}),
                "unrendered_sim_step_timing": "native sim.step including callbacks and nonphysics updates",
                "idle_preview": "configured_cadence_with_existing_world_policy",
                "pending_object_initialization": "must_complete_before_control_execution",
            },
            "observation_setup": {
                "robot_position": self.robot_position, "robot_yaw_deg": self.robot_yaw_deg,
                "camera_size": self.camera_size, "head_pitch_deg": self.head_pitch_deg,
            },
            "esi_commit": _git_commit(self.esi_root),
            "showharness_commit": _git_commit(SHOWHARNESS_ROOT),
        }

    def close(self) -> None:
        """Close the owning simulator after callers have flushed their outputs."""
        if self.closed:
            return
        self.closed = True
        if self.pipeline is not None:
            # Importing the pipeline alone does not create an app. og.shutdown()
            # would call exit(0) here and mask an input/task-resolution failure.
            if self.pipeline.og.app is None:
                return
            print("[esi] closing environment", flush=True)
            failed = sys.exc_info()[0] is not None
            if failed:
                # Kit can end Python during native shutdown; publish the error first.
                traceback.print_exc()
                sys.stderr.flush()
            handled = self.pipeline.task_cleanup_runtime(
                self.task_module, self.env, self.payload, self.task_state
            ) if self.task_module is not None and self.env is not None else False
            if not handled:
                og = self.pipeline.og
                og.sim.stop()
                for _ in range(3):
                    og.app.update()
                if failed:
                    import omni.kit.app

                    omni.kit.app.get_app().post_uncancellable_quit(1)
                # Isaac Sim 4.5's og.shutdown() removes live USD temporary files
                # and SimulationApp.close() then unloads native plugins still
                # referenced by Python. Close the native simulator first instead.
                og.sim.close()
                og.cleanup()

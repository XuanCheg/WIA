"""Owner-thread acquisition and one real-robot projection for Agent and GUMI."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from dataclasses import dataclass
from types import MappingProxyType
import time
import numpy as np

from core.observation_config import ObservationConfig
from core.robot_cameras import ADDITIONAL_RGB_NAMES, RGB_TO_VIEW
from core.observations import ActionHistoryEntry, AgentActionFeedback, AgentObservation, AgentRobotState, CameraSnapshot, InitialOverview
from core.sensor_observations import PerceptionEstimate, SENSOR_SPECS, SensorReadings
from wia.runtime.sim.esi_base import _main_thread
from wia.runtime.sim.esi_camera_freshness import (
    prepare_camera_freshness, check_camera_freshness, synchronized_camera_rendering,
)
from wia.runtime.sim.esi_omnigibson import NativeCameraNotReady


@dataclass(frozen=True)
class HumanPreview:
    frames: object
    preview_id: int
    captured_at: str
    source_snapshot_id: int


def _render(session):
    invalidate = getattr(session, "invalidate_camera_cache", None)
    if invalidate:
        invalidate()
    started = time.perf_counter()
    try:
        session.pipeline.og.sim.render()
    finally:
        counters = getattr(session, "performance", None)
        if counters and getattr(session, "_boundary_capture_depth", 0):
            counters.add("boundary_render_wall_time", time.perf_counter() - started)
            counters.add("boundary_render_count")


class ESIObservationSource:
    """Head/wrist share one render cycle; HTTP reads immutable action boundaries."""

    def __init__(self, session, *, task_text: str, history_limit: int = 12, config: ObservationConfig | None = None):
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError("History limit must be a positive integer.")
        self.session = session
        self.task_text = task_text
        self.config = config or getattr(session, "agent_observation_config", ObservationConfig())
        self.history_limit = history_limit
        self.history = ()
        self.action_step = 0
        self.snapshot_id = 0
        self.latest_snapshot = None
        self.boundary_observation = None
        self.include_scene = False
        self.on_capture = None
        self.on_boundary_start = None
        self.controller_state_provider = getattr(session, "controller_state", None)
        self.last_action_feedback = None
        self.perception_provider = None
        self.camera_geometry_provider = None
        self.preview_id = 0
        self.human_depth_visualizations = True
        self.initial_overview = None
        self.initial_overview_status = 'legacy_unavailable'
        self.initial_overview_failure = None
        self.initial_observation = None
        self._done_declaration = None
        self._episode_finalization = None

    def begin_episode(self, task_text: str) -> AgentObservation:
        """Start public task/history at the current scene, on the sensor owner.

        This does not reset physics. ESISession scene resets require a fresh
        simulator process; hardware adapters may implement their own reset.
        """
        _main_thread()
        if not isinstance(task_text, str) or not task_text.strip():
            raise ValueError("Episode task must be nonempty text.")
        self.task_text = task_text
        self.history = ()
        self.session.public_task_events = []
        self.action_step = 0
        self.last_action_feedback = None
        self._done_declaration = None
        self._episode_finalization = None
        self.initial_overview = None
        self.initial_overview_status = 'unavailable'
        self.initial_overview_failure = None
        self.initial_observation = None
        from wia.evaluation.safety import run_safety_hook
        run_safety_hook(self.session, 'begin_episode')
        observation = self.capture(action_boundary=True).observation
        # Initial ambiguity/history reducers must see exactly the same complete
        # input bundle as the Agent, including the frozen Global image.
        observation = self._capture_initial_overview(observation)
        prelude = getattr(self.session, 'candidate_episode_prelude', None)
        if callable(prelude):
            observation = prelude(self, observation)
            self.initial_observation = observation
        # Canonical Mapping worlds need a read-only initial inventory before
        # the first public action, even while release qualification is pending.
        identity_provider = getattr(self.session, 'mapping_joint_world_identity', None)
        if callable(identity_provider):
            try:
                task = self.session.loaded_task
                contract = task.definition.private.evaluation.parameters['mapping_joint_world_contract']
                self.session.mapping_joint_world_initial_identity = identity_provider(contract)
                self.session.mapping_joint_world_initial_identity_status = {'status': 'measured'}
            except Exception as error:
                self.session.mapping_joint_world_initial_identity = None
                self.session.mapping_joint_world_initial_identity_status = {
                    'status': 'unavailable', 'reason': f'{type(error).__name__}: {error}'}
        evaluator = getattr(self.session, "task_evaluator", None)
        if evaluator is not None:
            evaluator.begin_episode(observation)
            stability_window = getattr(self.session, 'initial_stability_window', None)
            if stability_window is not None:
                stability_window()
                observation = self.capture(action_boundary=True).observation
                evaluator.record(observation, action=None)
        exploration = getattr(self.session, "exploration_evaluator", None)
        if exploration is not None:
            exploration.begin_episode(observation)
        return observation

    def _capture_initial_overview(self, observation):
        """Call the dedicated native provider exactly once after reset settling.

        A missing provider/receipt remains explicit unavailable evidence. Agent
        runtime input validation must refuse it; diagnostic capture stays usable.
        Human Review cameras are never used as an implicit fallback.
        """
        provider = getattr(self.session, 'initial_overview_capture', None)
        try:
            if not callable(provider):
                raise ValueError('initial_overview_provider_unavailable')
            identity = getattr(self.session, 'initial_overview_identity', None)
            if identity is None:
                raise ValueError('initial_overview_reset_identity_unavailable')
            overview = provider(snapshot_id=observation.snapshot_id, captured_at=observation.captured_at)
            if isinstance(overview, dict):
                overview = InitialOverview(**overview)
            if type(overview) is not InitialOverview:
                raise ValueError('initial_overview_provider_contract_invalid')
            overview.validate_identity(identity)
            if overview.snapshot_id != observation.snapshot_id or overview.captured_at != observation.captured_at:
                raise ValueError('initial_overview_capture_boundary_mismatch')
            self.initial_overview = overview
            self.initial_overview_status = 'available'
        except Exception as error:
            self.initial_overview_failure = f'{type(error).__name__}: {error}'
        self.session.initial_overview_input_status = {
            'required': True, 'status': self.initial_overview_status,
            'failure': self.initial_overview_failure}
        if self.initial_overview is not None:
            # Rendering the dedicated Global advances graphics products for all
            # mounted cameras as well. Acquire their final synchronized bundle
            # after that render, without stepping physics, before any reducer
            # or Agent receives its initial images.
            invalidate = getattr(self.session, 'invalidate_camera_cache', None)
            if callable(invalidate):
                invalidate()
            observation = self.capture(action_boundary=True).observation
            self.initial_overview = replace(self.initial_overview,
                snapshot_id=observation.snapshot_id, captured_at=observation.captured_at)
        updated = replace(observation, initial_overview=self.initial_overview,
                          initial_overview_status=self.initial_overview_status)
        self.boundary_observation = updated
        self.latest_snapshot = replace(self.latest_snapshot, observation=updated)
        self.initial_observation = updated
        if self.on_capture is not None:
            self.on_capture(self.latest_snapshot)
        return updated

    def refresh_public_task_events(self):
        """Publish a follow-up at the retained boundary without physics or new RGB."""
        _main_thread()
        if self.latest_snapshot is None:
            return
        observation = self.latest_snapshot.observation
        updated = replace(observation, public_task_events=tuple(
            getattr(self.session, 'public_task_events', ())))
        if updated.public_task_events == observation.public_task_events:
            return
        self.latest_snapshot = replace(self.latest_snapshot, observation=updated)
        if observation.action_boundary:
            self.boundary_observation = updated
        if self.on_capture is not None:
            self.on_capture(self.latest_snapshot)

    def capture(self, *, action_boundary=False, action_result=None) -> CameraSnapshot:
        """Publish fresh boundary imagery, suppressing intermediate acquisitions."""
        _main_thread()
        if getattr(self.session, "action_executing", False) and not action_boundary:
            if self.latest_snapshot is None:
                raise RuntimeError("Intermediate camera acquisition is deferred until the action boundary.")
            return self.latest_snapshot
        started = time.perf_counter()
        depth = getattr(self.session, "_boundary_capture_depth", 0)
        if action_boundary:
            self.session._boundary_capture_depth = depth + 1
        counters = getattr(self.session, "performance", None)
        before = counters.snapshot() if counters and action_boundary else {}
        try:
            with synchronized_camera_rendering(self.session):
                snapshot = self._capture(action_boundary=action_boundary, action_result=action_result)
            self._captured_physical_generation = getattr(self.session, "_physical_step_generation", 0)
            if action_boundary:
                self.session._action_boundary_ready = True
                from wia.evaluation.safety import record_safety
                record_safety(self.session, snapshot.observation)
            return snapshot
        finally:
            self.session._boundary_capture_depth = depth
            elapsed = time.perf_counter() - started
            if action_boundary:
                if counters:
                    counters.add("boundary_capture_wall_time", elapsed)
                    counters.add("boundary_capture_count")
                if action_result is not None:
                    timing = action_result.setdefault("timing", {})
                    timing["boundary_capture_wall_time_s"] = elapsed
                    if counters:
                        metrics = counters.snapshot()
                        timing["boundary_render_wall_time_s"] = metrics.get(
                            "boundary_render_wall_time", 0) - before.get("boundary_render_wall_time", 0)
                        timing["boundary_render_count"] = metrics.get(
                            "boundary_render_count", 0) - before.get("boundary_render_count", 0)
                        action_start = getattr(self.session, "_action_performance_start", None)
                        if action_start is not None:
                            for metric in ("unrendered_sim_step_wall_time", "controller_step_wall_time",
                                           "native_unrendered_control_step_count"):
                                timing[metric] = metrics.get(metric, 0) - action_start.get(metric, 0)

    def _capture(self, *, action_boundary=False, action_result=None) -> CameraSnapshot:
        _main_thread()
        if self.initial_overview is not None:
            try:
                self.initial_overview.validate_identity(getattr(self.session, 'initial_overview_identity', None))
            except ValueError as error:
                self.initial_overview = None
                self.initial_overview_status = 'unavailable'
                self.initial_overview_failure = str(error)
                self.session.initial_overview_input_status = {
                    'required': True, 'status': 'unavailable', 'failure': self.initial_overview_failure}
        if action_boundary and self.on_boundary_start is not None:
            self.on_boundary_start()
        # A newly attached depth annotator can return an empty buffer or
        # uninitialized negative values for its first render cycles. Re-render
        # the complete bundle, without physics, instead of inventing depth or
        # mixing images from different captures.
        reuse = bool(not action_boundary and getattr(self.session, "reuse_step_observations", False)
                     and getattr(self.session, "_step_camera_data", None))
        counters = getattr(self.session, "performance", None)
        started = time.perf_counter()
        physical_advance = getattr(self.session, "_physical_step_generation", 0) != getattr(
            self, "_captured_physical_generation", 0)
        initial_graphics_iterations = (getattr(self.session, "_native_boundary_graphics_iterations", 1)
                                      if action_boundary and (physical_advance or self.latest_snapshot is None) else 1)
        # Attach camera-parameter annotators before rendering, then hold the
        # post-stop physical targets fixed throughout this bounded readback.
        # Their world poses remain private and never enter AgentObservation.
        camera_targets = prepare_camera_freshness(self.session) if not reuse else ()
        for retry in range(8):
            if not reuse or retry:
                for _ in range(initial_graphics_iterations if retry == 0 else 1):
                    _render(self.session)
            unready = []
            try:
                raw = self.session.observe(include_wrist=True, include_scene=self.include_scene, include_sensors=True)
                check_camera_freshness(self.session, camera_targets)
            except NativeCameraNotReady as error:
                unready = [str(error)]
                continue
            for name in ("head", "wrist"):
                depth = raw.get(f"{name}_depth_m")
                if depth is not None and getattr(self.config, f"{name}_depth"):
                    values = np.asarray(depth)
                    rgb = raw.get(f"{name}_rgb")
                    if (values.ndim != 2 or 0 in values.shape or np.any(values < 0)
                            or rgb is not None and values.shape != np.asarray(rgb).shape[:2]):
                        unready.append(name)
            if not unready:
                break
        else:
            raise RuntimeError(f"Native camera annotators did not become ready: {', '.join(unready)}.")
        if counters:
            counters.add("camera_readiness_retries", retry)
            counters.add("canonical_capture_count")
            counters.add("reused_step_capture_count", int(reuse and retry == 0))
        if "wrist_rgb" not in raw:
            raise RuntimeError("Canonical AgentObservation requires both native R1 head/wrist RGB sensors.")
        snapshot_id = self.snapshot_id + 1
        state = self.controller_state_provider() if self.controller_state_provider is not None else None
        robot_state = AgentRobotState.from_controller_state(state, config=self.config) if state is not None else None
        if action_result is not None:
            feedback = AgentActionFeedback.from_result(action_result, state=state, config=self.config)
            before = self.boundary_observation.robot_state if self.boundary_observation is not None else None
            if (feedback.action.startswith("BASE_") and self.config.action_feedback and before is not None
                    and robot_state is not None and before.base_pose is not None and robot_state.base_pose is not None):
                from core.observations import _quaternion_rotation
                start, end = before.base_pose, robot_state.base_pose
                displacement = _quaternion_rotation(start.quaternion_xyzw).T @ (np.array(end.position_m)-start.position_m)
                from core.action_units import BASE_ATOMS
                directions = {BASE_ATOMS[0]: (1, 0), BASE_ATOMS[1]: (-1, 0), BASE_ATOMS[2]: (0, 1), BASE_ATOMS[3]: (0, -1)}
                measured = {}
                if feedback.action in directions:
                    measured["distance_m"] = float(displacement[:2] @ directions[feedback.action])
                else:
                    relative = _quaternion_rotation(start.quaternion_xyzw).T @ _quaternion_rotation(end.quaternion_xyzw)
                    angle = float(np.rad2deg(np.arctan2(relative[1, 0], relative[0, 0])))
                    measured["angle_deg"] = angle if feedback.action == BASE_ATOMS[4] else -angle
                # Target reach remains unknown: a displacement estimate alone does
                # not supply the controller's full tolerance / hold test.
                feedback = replace(feedback, estimated_motion=measured,
                                   sources=dict(feedback.sources, estimated_motion="odometry"))
            self.last_action_feedback = feedback
            entry = ActionHistoryEntry.from_feedback(feedback,
                command=action_result.get("command") if action_result.get("parameterized") else None)
            self.history = (self.history + (entry,))[-self.history_limit:]
        supplied = state.sensor_feedback.sensors if state is not None and state.sensor_feedback is not None else SensorReadings()
        switches = {"head_depth_m": "head_depth", "wrist_depth_m": "wrist_depth", "lidar_points_m": "lidar",
                    "force_torque": "force_torque", "tactile": "tactile"}
        sensors = SensorReadings(**{name: (raw.get(name) if raw.get(name) is not None else getattr(supplied, name))
                                    if getattr(self.config, switches[name]) else None for name in SENSOR_SPECS})
        observation = AgentObservation(
            task_text=self.task_text, agentview_rgb=raw["head_rgb"] if self.config.head_rgb else None,
            wrist_rgb=raw["wrist_rgb"] if self.config.wrist_rgb else None,
            action_history=self.history if self.config.recent_history else (), snapshot_id=snapshot_id,
            captured_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            action_step=self.action_step, action_boundary=action_boundary, robot_state=robot_state,
            last_action_feedback=self.last_action_feedback if self.config.action_feedback else None,
            sensors=sensors, observation_config=self.config,
            additional_rgb={name: raw[name] for name in ADDITIONAL_RGB_NAMES
                            if name in raw and self.config.additional_robot_rgb},
            camera_geometry=self._camera_geometry(state),
            initial_overview=self.initial_overview, initial_overview_status=self.initial_overview_status,
            public_task_events=tuple(getattr(self.session, 'public_task_events', ())),
        )
        observation.set_serialization_cache(getattr(self.session, "serialization_cache", True))
        if self.perception_provider is not None and self.config.perception:
            estimate = self.perception_provider(observation)
            if not isinstance(estimate, PerceptionEstimate):
                raise TypeError("Perception providers return a PerceptionEstimate.")
            observation = replace(observation, perception=estimate)
        human_rgb = {}
        if not self.config.additional_robot_rgb:
            human_rgb.update({RGB_TO_VIEW[name]: raw[name] for name in ADDITIONAL_RGB_NAMES if name in raw})
        if self.human_depth_visualizations:
            from wia.runtime.sim.esi_debug_cameras import depth_rgb
            config = getattr(self.session, "viewer_config", None)
            for name in ("head", "wrist"):
                depth = raw.get(f"{name}_depth_m")
                if depth is not None:
                    human_rgb[f"{name}_depth"] = depth_rgb(depth, config.depth_min_m if config else 0.1,
                                                         config.depth_max_m if config else 5.0)
        if self.include_scene:
            cameras = getattr(self.session, "global_cameras", {})
            for name, sensor in cameras.items():
                human_rgb[name] = self.session._sensor_rgb(sensor)
        snapshot = CameraSnapshot(observation, raw.get("scene_rgb"), state,
                                  raw["head_rgb"] if not self.config.head_rgb else None,
                                  raw["wrist_rgb"] if not self.config.wrist_rgb else None,
                                  human_rgb=human_rgb)
        self.snapshot_id = snapshot_id
        self.latest_snapshot = snapshot
        if action_boundary:
            self.boundary_observation = observation
        candidate_hook = getattr(self.session, 'candidate_observation_hook', None)
        if action_boundary and callable(candidate_hook):
            candidate_hook(observation)
        if self.on_capture is not None:
            self.on_capture(snapshot)
        if counters:
            counters.add("observation_wall_time", time.perf_counter()-started)
        return snapshot

    def capture_preview(self, *, recording=False):
        """Human-only RGB acquisition; no Agent snapshot, state, depth or history."""
        _main_thread()
        if getattr(self.session, "action_executing", False) and not recording:
            raise RuntimeError("Human previews are deferred until the action boundary.")
        # Idle control stepping is unrendered too; refresh graphics at the
        # configured human preview cadence without advancing physical time.
        if recording:
            depth = getattr(self.session, "_recording_capture_depth", 0)
            self.session._recording_capture_depth = depth + 1
            try:
                with synchronized_camera_rendering(self.session):
                    _render(self.session)
                    _render(self.session)
                    return self._capture_preview_rgb(recording=True)
            finally:
                self.session._recording_capture_depth = depth
        _render(self.session)
        return self._capture_preview_rgb()

    def _capture_preview_rgb(self, *, recording=False):
        raw = self.session.observe(include_wrist=True, include_scene=self.include_scene, include_sensors=False)
        from core.observations import freeze_rgb
        frames = {"head": raw["head_rgb"], "wrist": raw["wrist_rgb"]}
        frames.update({RGB_TO_VIEW[name]: raw[name] for name in ADDITIONAL_RGB_NAMES if name in raw})
        if self.include_scene:
            frames["scene"] = raw["scene_rgb"]
            frames.update({name: self.session._sensor_rgb(sensor)
                           for name, sensor in getattr(self.session, "global_cameras", {}).items()})
        if self.human_depth_visualizations:
            # Only explicitly requested debug previews read native depth.
            from wia.runtime.sim.esi_debug_cameras import depth_rgb
            cfg = getattr(self.session, "viewer_config", None)
            for name, sensor in (("head", getattr(self.session, "camera_name", None)),
                                 ("wrist", getattr(self.session, "wrist_camera_name", None))):
                if sensor is None:
                    continue
                data = self.session._sensor_data(self.session.robot.sensors[sensor])
                if "depth_linear" in data:
                    from wia.runtime.sim.esi_omnigibson import _array
                    frames[name+"_depth"] = depth_rgb(_array(data["depth_linear"]),
                        cfg.depth_min_m if cfg else .1, cfg.depth_max_m if cfg else 5.)
        self.preview_id += 1
        counters = getattr(self.session, "performance", None)
        if counters:
            counters.add("human_preview_count")
        return HumanPreview(MappingProxyType({k: freeze_rgb(v) for k, v in frames.items()}), self.preview_id,
                            datetime.now(timezone.utc).isoformat(timespec="microseconds"), self.snapshot_id)

    def _camera_geometry(self, state):
        if (not self.config.camera_geometry or not self.config.ee_pose_in_base
                or not getattr(getattr(self.session, "robot", None), "urdf_path", None)):
            return {}
        if self.camera_geometry_provider is None:
            from wia.runtime.sim.esi_camera_geometry import ESICameraGeometry
            working_arm = (state.proprioception or {}).get("working_arm", "left") if state is not None else "left"
            self.camera_geometry_provider = ESICameraGeometry(self.session, working_arm)
        switches = {"agentview_rgb": self.config.head_rgb, "wrist_rgb": self.config.wrist_rgb}
        return {name: value for name, value in self.camera_geometry_provider.capture().items() if switches[name]}

    def after_action(self, result: dict) -> None:
        _main_thread()
        self.action_step += 1
        self.capture(action_boundary=True, action_result=result)
        evaluator = getattr(self.session, "task_evaluator", None)
        if evaluator is not None:
            evaluator.record(self.boundary_observation, action=result["action"])
        self.refresh_public_task_events()
        exploration = getattr(self.session, "exploration_evaluator", None)
        if exploration is not None:
            try:
                exploration.record(self.boundary_observation, result=result)
            except Exception as error:
                # Diagnostic visibility must not invalidate a completed action.
                exploration.record_error(self.boundary_observation, error)
        checker = getattr(self.session, "private_visibility_checker", None)
        if checker is not None and result["action"] == "GRASP":
            checker.record_grasp(self.boundary_observation)
        diagnostic = getattr(self.session, "private_motion_diagnostics", None)
        if diagnostic is not None and result.get("reason") != "completed" and result["action"] != "GRASP":
            diagnostic.record(action=result["action"], reason=result.get("reason"), phase="after_stop", result=result)

    def report_arrivals(self, rooms, *, snapshot_id, action_step):
        """Bind Agent room claims to the retained boundary on the owner thread."""
        _main_thread()
        observation = self.agent_observation()
        if observation.snapshot_id != snapshot_id or observation.action_step != action_step:
            raise ValueError('Arrival declaration refers to a stale action boundary.')
        evaluator = getattr(self.session, 'task_evaluator', None)
        report = getattr(evaluator, 'report_arrivals', None)
        if callable(report):
            report(rooms, observation)

    def report_answer(self, answer, *, snapshot_id, action_step):
        """Retain a model-authored answer at the current observation boundary."""
        _main_thread()
        observation = self.agent_observation()
        if observation.snapshot_id != snapshot_id or observation.action_step != action_step:
            raise ValueError('Answer declaration boundary changed.')
        report = getattr(getattr(self.session, 'task_evaluator', None), 'report_answer', None)
        if not callable(report) or not isinstance(answer, str) or not answer.strip() or len(answer) > 4096:
            raise ValueError('Answer declaration is unavailable.')
        report(answer, observation)
        self.refresh_public_task_events()
        return {'ok': True, 'message': 'Answer declaration recorded.'}

    def report_done(self, *, snapshot_id, action_step):
        """Accept one terminal declaration at a fresh graphics-only boundary."""
        _main_thread()
        requested = (snapshot_id, action_step)
        if self._done_declaration is not None:
            previous, receipt = self._done_declaration
            if requested not in (previous, (receipt['snapshot_id'], receipt['action_step'])):
                raise ValueError('DONE declaration boundary changed.')
            return dict(receipt)
        observation = self.agent_observation()
        if requested != (observation.snapshot_id, observation.action_step):
            raise ValueError('DONE declaration boundary changed.')
        evaluator = getattr(self.session, 'task_evaluator', None)
        if getattr(evaluator, 'requires_done_declaration', False) is True:
            if not callable(getattr(evaluator, 'report_done', None)):
                raise RuntimeError('Terminal task evaluation is unavailable.')
            # The owner queue excludes robot execution here. Rendering refreshes
            # the physically current stopped state without an extra hold step.
            observation = self.capture(action_boundary=True).observation
            evaluator.report_done(observation)
        receipt = {'ok': True, 'message': 'DONE declaration recorded.',
                   'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
        self._done_declaration = (requested, receipt)
        return dict(receipt)

    def finalize_episode(self, *, snapshot_id, action_step, reason):
        """Capture and score the actual terminal state on the simulator owner."""
        _main_thread()
        if self._episode_finalization is not None:
            requested, receipt = self._episode_finalization
            if (snapshot_id, action_step) not in (requested, (receipt['snapshot_id'], receipt['action_step'])):
                raise ValueError('Terminal evaluation boundary changed.')
            return dict(receipt)
        observation = self.agent_observation()
        if (snapshot_id, action_step) != (observation.snapshot_id, observation.action_step):
            raise ValueError('Terminal evaluation boundary changed.')
        evaluator = getattr(self.session, 'task_evaluator', None)
        if self._done_declaration is None:
            observation = self.capture(action_boundary=True).observation
            finalize = getattr(evaluator, 'finalize_episode', None)
            if callable(finalize):
                finalize(observation, reason=reason)
        receipt = {'ok': True, 'message': 'Terminal state recorded.',
                   'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
        self._episode_finalization = ((snapshot_id, action_step), receipt)
        return dict(receipt)

    def resume_episode(self):
        """Reopen a terminalized interruption without resetting the world."""
        _main_thread()
        resume = getattr(getattr(self.session, 'task_evaluator', None), 'resume_episode', None)
        if callable(resume):
            resume()
        self._episode_finalization = None

    def report_temporal(self, event, answer, *, snapshot_id, action_step):
        """Apply one bounded temporal event at the retained owner-thread boundary."""
        _main_thread()
        observation = self.agent_observation()
        if observation.snapshot_id != snapshot_id or observation.action_step != action_step:
            raise ValueError('Temporal event boundary changed.')
        controller = getattr(self.session, 'temporal_phase_controller', None)
        phase = getattr(controller, 'phase', None)
        evaluator = getattr(self.session, 'task_evaluator', None)
        if controller is None or phase not in ('phase_1', 'phase_2'):
            raise ValueError('Temporal event is unavailable.')
        if event in ('phase_1_complete', 'phase_1_budget_exhausted'):
            prepare = getattr(evaluator, 'prepare_temporal_transition', None)
            if phase != 'phase_1' or answer is not None or not callable(prepare):
                raise ValueError('Temporal event is unavailable.')
            prepare(observation)
            controller.advance()
            if getattr(controller, 'transition', None) is not None:
                controller.transition['reason'] = ('budget_exhausted' if event == 'phase_1_budget_exhausted' else 'agent_requested')
            if controller.phase != 'phase_2':
                raise RuntimeError('Temporal phase transition did not complete.')
            # A prior env.step camera bundle predates the content mutation.
            self.session._step_camera_data = None
            invalidate = getattr(self.session, 'invalidate_camera_cache', None)
            if callable(invalidate):
                invalidate()
            updated = self.capture(action_boundary=True).observation
            if updated.snapshot_id <= snapshot_id or updated.action_step != action_step:
                raise RuntimeError('Temporal transition did not preserve the action boundary.')
            evaluator.record(updated, action=None)
            changed = True
        elif event == 'answer':
            report = getattr(evaluator, 'report_temporal_answer', None)
            if (phase != 'phase_2' or not isinstance(answer, str) or not answer.strip()
                    or len(answer) > 160 or not callable(report)):
                raise ValueError('Temporal event is unavailable.')
            report(answer, observation)
            changed = False
        else:
            raise ValueError('Temporal event is unavailable.')
        return {'ok': True, 'message': 'Temporal event recorded.',
                'phase': controller.phase, 'observation_changed': changed}

    def agent_observation(self) -> AgentObservation:
        observation = self.boundary_observation
        if observation is None:
            raise RuntimeError("No action-boundary observation is available yet.")
        return observation

"""ESI GUMI control: HTTP queues commands; the simulator's main thread runs them."""
from __future__ import annotations

import queue
import secrets
import threading
import time
from dataclasses import dataclass

from core.action_interface import ActionExecutor
from core.action_compiler import ParameterizedAction, compile_action
from core.observation_config import ObservationConfig
from core.performance import public_action_timing
from core.observations import AgentActionFeedback, AgentRobotState, CameraSnapshot, ControllerState
from core.robot_cameras import RGB_TO_VIEW
from wia.runtime.sim.esi_world_policy import WorldPolicy
from wia.tasks.temporal_policy import survey_action_allowed

from wia.runtime.web.web_teleop.esi_preview import ESIPreviewBackend
from gumi.web_teleop.server import MAX_BATCH_TOKENS


def _owner_thread():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("The ESI command queue must be processed on its owner main thread.")


@dataclass(frozen=True)
class ArrivalDeclarations:
    rooms: tuple[str, ...]
    snapshot_id: int
    action_step: int


@dataclass(frozen=True)
class AnswerDeclaration:
    answer: str
    snapshot_id: int
    action_step: int


@dataclass(frozen=True)
class DoneDeclaration:
    snapshot_id: int
    action_step: int


@dataclass(frozen=True)
class EpisodeFinalization:
    snapshot_id: int
    action_step: int
    reason: str


@dataclass(frozen=True)
class TemporalDeclaration:
    event: str
    answer: str | None
    snapshot_id: int
    action_step: int


class ESIControlBackend(ESIPreviewBackend):
    """HTTP validation/queue and cached UI state; execution belongs to executor."""
    read_only = False
    ALLOWED_COMMANDS = ("move", "step")

    motion_wait_uses_simulation_time = True

    @property
    def agent_extensions(self):
        declarations = {
            'agent/arrivals': (self.submit_arrivals, ('rooms', 'snapshot_id', 'action_step', 'control_token')),
            'agent/answer': (self.submit_answer, ('answer', 'snapshot_id', 'action_step', 'control_token')),
            'agent/done': (self.submit_done, ('snapshot_id', 'action_step', 'control_token')),
            'agent/finalize': (self.submit_finalization, ('snapshot_id', 'action_step', 'reason', 'control_token')),
            'agent/temporal': (self.submit_temporal, ('event', 'answer', 'snapshot_id', 'action_step', 'control_token'))}
        def handler(method, fields):
            def submit(body):
                if set(body) != set(fields):
                    return {'ok': False, 'status': 400, 'message': 'Unexpected dataset declaration fields.'}
                return method(*(body[key] for key in fields))
            return submit
        return {name: handler(method, fields) for name, (method, fields) in declarations.items()}

    def __init__(self, metadata: dict, executor: ActionExecutor, *, observation_source=None) -> None:
        super().__init__(metadata, observation_source=observation_source)
        self.executor = executor
        self.commands = queue.Queue(maxsize=1)
        self.equivalent_fixed_primitives = self.actual_fixed_primitive_executions = self.metric_target_executions = 0
        self.closing = threading.Event()
        self.busy = False
        self.active_cancel = None
        self.last_result = None
        self.history = []
        self.steps = 0
        self.proprioception = None
        self.controller_snapshot = None
        self.agent_snapshot = None
        self.control_owner = "human"
        self.agent_run_id = None
        self._control_token = None
        self._agent_cancelled = False
        self.episode_task = None
        self.action_mode = "fixed"
        self.macro_progress = None
        self.primitive_actions = 0
        self.physics_steps = 0
        self.physics_steps_known = True
        self.world_policy = WorldPolicy()
        self.observation_config = (observation_source.config if observation_source is not None
                                   else ObservationConfig(**metadata.get("agent_observation_config", {})))
        self.message = ("R1 base + left arm ready · gripper disabled" if self.action_groups["arm"]
                        else "R1 base ready · arm and gripper disabled")
        if self.capabilities.gripper:
            self.message = "R1 base + arms + grippers ready · assisted grasp"

    @property
    def capabilities(self):
        return self.executor.capabilities

    @property
    def STEP_TOKENS(self):
        return self.executor.available_actions

    @property
    def action_groups(self):
        return self.capabilities.action_groups

    @property
    def rotation_enabled(self):
        return self.capabilities.ee_rotation

    @property
    def mode(self):
        return ("esi_manipulation" if self.capabilities.gripper else
                "esi_arm" if self.action_groups["arm"] else "esi_base")

    def publish(self, observation, *, controller_state: ControllerState | None = None):
        # Robot readback has its own API and is never added to the RGB observation.
        if controller_state is not None and not isinstance(controller_state, ControllerState):
            raise TypeError("Control readback must be a ControllerState.")
        with self.frame_cond:
            self.controller_snapshot = controller_state
            if isinstance(observation, CameraSnapshot) and observation.observation.action_boundary:
                self.agent_snapshot = observation.observation
            state = self.agent_snapshot.robot_state if self.agent_snapshot is not None else (
                AgentRobotState.from_controller_state(controller_state, config=self.observation_config)
                if controller_state is not None else None)
            self.proprioception = self._proprioception(state)
            if self.metadata.get("performance", {}).get("human_preview", True):
                super().publish(observation)

    def publish_preview(self, preview):
        # Human frames do not replace any authoritative Agent/control snapshot.
        names = {"head": "head_rgb", "wrist": "wrist_rgb", "scene": "scene_rgb"}
        names.update({view: name for name, view in RGB_TO_VIEW.items() if name not in ('agentview_rgb', 'wrist_rgb')})
        super().publish({names[name]: rgb for name, rgb in preview.frames.items() if name in names})
        with self.frame_cond:
            if 'agentview' in self.frames and 'head' not in self.frames:
                self.frames['head'] = self.frames['agentview']

    @staticmethod
    def _proprioception(state):
        if state is None:
            return None
        return dict(state.to_dict(), eef_pos=state.ee_pose.position_m if state.ee_pose is not None else None,
                    eef_quaternion_xyzw=state.ee_pose.quaternion_xyzw if state.ee_pose is not None else None)

    def action_allowed(self, token) -> bool:
        return self.executor.action_allowed(token)

    def state(self) -> dict:
        with self.frame_cond:
            state = super().state()
            robot = self.agent_snapshot.robot_state if self.agent_snapshot is not None else (
                AgentRobotState.from_controller_state(self.controller_snapshot, config=self.observation_config)
                if self.controller_snapshot is not None else None)
            state.update(
                read_only=False, mode=self.mode, physics_paused=not self.world_policy.should_advance(self.control_owner),
                capabilities=self.capabilities.to_dict(),
                action_groups=dict(self.action_groups), step_tokens=list(self.STEP_TOKENS),
                busy=self.busy, steps=self.steps, message=self.message,
                action_mode=self.action_mode, macro_progress=self.macro_progress,
                primitive_actions=self.primitive_actions, physics_steps=self.physics_steps,
                macro_execution=self.executor.macro_execution,
                equivalent_fixed_primitives=self.equivalent_fixed_primitives,
                actual_fixed_primitive_executions=self.actual_fixed_primitive_executions,
                metric_target_executions=self.metric_target_executions,
                controller_env_steps=self.physics_steps,
                control_owner=self.control_owner, agent_run_id=self.agent_run_id,
                task_text=self.episode_task or state["task_text"],
                last_result=self.last_result, history=list(self.history) if self.observation_config.recent_history else [],
                last_token=self.last_result["action"] if self.last_result else "-",
                base_frame=self.metadata.get("base_control", {}).get("frame"), recording_enabled=False,
                base_config=self.metadata.get("base_control", {}).get("config"),
                arm_config=self.metadata.get("arm_control", {}).get("config"),
                working_arm=self.metadata.get("arm_control", {}).get("working_arm"),
                rotation_enabled=self.rotation_enabled,
                rotation_config=self.metadata.get("rotation_control", {}).get("config"),
                rotation_frame=self.metadata.get("rotation_control", {}).get("frame"),
                trunk_config=self.metadata.get("trunk_control", {}).get("config"),
                proprioception=self._proprioception(robot),
                robot_state=robot.to_dict() if robot is not None else None,
                observation_snapshot_id=self.agent_snapshot.snapshot_id if self.agent_snapshot is not None else None,
                observation_captured_at=self.agent_snapshot.captured_at if self.agent_snapshot is not None else None,
                observation_config=self.observation_config.to_dict(),
                camera_shape=self.metadata.get("image_shape"),
                task_evaluation_available=False,
                grasp_mode=self.metadata.get("grasp_mode"),
                ee_pos=robot.ee_pose.position_m if robot is not None and robot.ee_pose is not None else None,
            )
            state["human_parameterized_commands"] = getattr(self.executor, "macro_limits", None) is not None
            if state["human_parameterized_commands"]:
                from core.action_compiler import action_bounds
                state["human_command_bounds"] = action_bounds(self.executor.available_actions,
                    self.executor.primitive_resolution, self.executor.macro_limits)
            if self.capabilities.gripper:
                closed = robot.gripper_closed if robot is not None else None
                command = robot.gripper_command_name if robot is not None else None
                state.update(gripper_closed=closed,
                             holding_estimate=robot.holding_estimate if robot is not None else None,
                             next_gripper="RELEASE" if closed or command == "GRASP" else "GRASP")
            state.update(holding=None, picked=None, placed=None, task_done=None)
            state['done_declaration_available'] = callable(getattr(self.observation_source, 'report_done', None))
            state['episode_finalization_available'] = callable(getattr(self.observation_source, 'finalize_episode', None))
            session = getattr(self.observation_source, 'session', None)
            state['requires_done_declaration'] = getattr(
                getattr(session, 'task_evaluator', None), 'requires_done_declaration', False) is True
            state.update(self.world_policy.snapshot())
            state.update(self.temporal_state())
            if self.metadata.get("episode_contract") is not None:
                state["episode_contract"] = self.metadata["episode_contract"]
                state["declarative_resume_available"] = True
            limits = getattr(self.executor, "macro_limits", None)
            if limits is not None:
                state["parameterized_actions"] = {"resolution": self.executor.primitive_resolution.to_dict(),
                    "limits": limits.to_dict(), "request_timeout_s": None}
        return state

    def temporal_state(self):
        """Expose only the public phase label from the existing native controller."""
        session = getattr(self.observation_source, 'session', None)
        controller = getattr(session, 'temporal_phase_controller', None)
        phase = getattr(controller, 'phase', None)
        enabled = controller is not None and phase in ('phase_1', 'phase_2')
        return {'temporal_episode': enabled, 'temporal_phase': phase if enabled else None}

    def claim_control(self, run_id, task_text, timeout=None, *, action_mode="fixed",
                      world_advances_during_inference=True, resume_action_step=None):
        if (not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 128
                or not isinstance(task_text, str) or not task_text.strip()):
            return {"ok": False, "status": 400, "message": "run_id and task_text are required."}
        if action_mode not in ("fixed", "parameterized"):
            return {"ok": False, "status": 400, "message": "Unknown action mode."}
        if type(world_advances_during_inference) is not bool:
            return {"ok": False, "status": 400, "message": "world_advances_during_inference must be boolean."}
        contract = self.metadata.get("episode_contract")
        if resume_action_step is not None and (contract is None or type(resume_action_step) is not int
                                               or resume_action_step < 0):
            return {"ok": False, "status": 400, "message": "Declarative resume requires an integer action boundary."}
        if contract is not None and (run_id != contract["episode_id"]
                or task_text != self.observation_source.task_text or action_mode != contract["action_mode"]
                or world_advances_during_inference != contract["world_advances_during_inference"]):
            return {"ok": False, "status": 409, "message": "Episode identity, public task or policy differs from the immutable task manifest."}
        if action_mode == "parameterized" and getattr(self.executor, "macro_limits", None) is None:
            return {"ok": False, "status": 405, "message": "Parameterized commands are disabled on this backend."}
        done, cancel, slot = threading.Event(), threading.Event(), {}
        with self.frame_cond:
            if self.closing.is_set():
                return {"ok": False, "status": 503, "message": "Robot is shutting down."}
            if self.busy or self.control_owner != "human":
                return {"ok": False, "status": 409, "message": "Robot control is already owned or busy."}
            if self.observation_source is None:
                return {"ok": False, "status": 503, "message": "Canonical observation source required."}
            claimed = getattr(self, "_declarative_episode_claimed", False)
            if resume_action_step is not None and (not claimed or self.steps != resume_action_step):
                return {"ok": False, "status": 409, "message": "Interrupted episode action boundary changed."}
            if contract is not None and claimed and resume_action_step is None:
                return {"ok": False, "status": 409, "message": "This immutable task episode already acquired its control lease."}
            token = secrets.token_urlsafe(32)
            self.control_owner, self.agent_run_id = "agent", run_id
            self.action_mode = action_mode
            self._control_token, self._agent_cancelled = token, False
            if contract is not None:
                self._declarative_episode_claimed = True
            self.busy, self.active_cancel = True, cancel
            slot["task_text"] = task_text
            slot["world_advances_during_inference"] = world_advances_during_inference
            if resume_action_step is not None:
                slot["resume_action_step"] = resume_action_step
            self.commands.put_nowait((None, done, cancel, slot))
            self.frame_cond.notify_all()
        if not done.wait(timeout):
            cancel.set()
            with self.frame_cond:
                self._agent_cancelled = True
                if not self.busy:
                    self._release_control()
            return {"ok": False, "status": 504, "message": "Episode initialization timed out; cancelled."}
        if slot.get("ok"):
            slot.update(control_token=token, run_id=run_id, control_owner="agent", scene_reset=False)
            session = getattr(self.observation_source, 'session', None)
            if session is not None and hasattr(session, 'initial_overview_identity'):
                slot.update(initial_overview_identity=session.initial_overview_identity,
                    initial_overview_input_status=getattr(session, 'initial_overview_input_status',
                        {'required': True, 'status': 'unavailable'}))
            slot.update(self.temporal_state())
            if action_mode == "parameterized":
                slot["action_mode"] = action_mode
        return slot

    def _release_control(self):
        self.world_policy.transition("debug_hold" if self.metadata.get("debug_retain") else "human_idle")
        self.control_owner, self.agent_run_id, self._control_token = "human", None, None

    def release_control(self, token):
        with self.frame_cond:
            if self.control_owner == "human":
                return {"ok": True, "control_owner": "human"}
            if not isinstance(token, str) or not secrets.compare_digest(token, self._control_token or ""):
                return {"ok": False, "status": 403, "message": "Invalid Agent control token."}
            if self.busy:
                return {"ok": False, "status": 409, "message": "Wait for the current command to stop."}
            self._release_control()
            return {"ok": True, "control_owner": "human"}

    def submit_agent(self, action, token, timeout=None):
        # Agent output is exactly one fixed action; human batch syntax is not accepted.
        if not isinstance(token, str) or not token:
            return {"ok": False, "status": 403, "message": "Agent control token required."}
        if self.temporal_state()['temporal_phase'] == 'phase_1' and not survey_action_allowed(action):
            return {'ok': False, 'status': 409, 'message': 'Phase 1 forbids object interaction.'}
        if self.action_mode != "fixed":
            return {"ok": False, "status": 400, "message": "This episode requires parameterized commands."}
        return self.submit("move", action, timeout=timeout, control_token=token)

    def submit_parameterized(self, command, token, timeout=None):
        if not isinstance(token, str) or not token:
            return {"ok": False, "status": 403, "message": "Agent control token required."}
        if self.temporal_state()['temporal_phase'] == 'phase_1' and not survey_action_allowed(command):
            return {'ok': False, 'status': 409, 'message': 'Phase 1 forbids object interaction.'}
        limits = getattr(self.executor, "macro_limits", None)
        if limits is None:
            return {"ok": False, "status": 405, "message": "Parameterized commands are disabled."}
        try:
            plan = compile_action(command, self.capabilities, self.executor.primitive_resolution, limits)
            if plan.repeat_count == 0:
                raise ValueError("DONE terminates the runner; it does not execute motion.")
        except (ValueError, TypeError) as error:
            return {"ok": False, "status": 400, "message": str(error)}
        return self._submit_transaction(plan.command, {},
            timeout, control_token=token,
            action_mode="parameterized")

    def submit_human_parameterized(self, command, timeout=None):
        """Execute one human metric command through the same guarded compiler."""
        limits = getattr(self.executor, "macro_limits", None)
        if limits is None:
            return {"ok": False, "status": 405, "message": "Parameterized commands are disabled."}
        try:
            plan = compile_action(command, self.capabilities, self.executor.primitive_resolution, limits)
            if plan.repeat_count == 0:
                raise ValueError("DONE belongs to the episode runner.")
        except (ValueError, TypeError) as error:
            return {"ok": False, "status": 400, "message": str(error)}
        return self._submit_transaction(plan.command, {}, timeout, action_mode="parameterized")

    def submit_arrivals(self, rooms, snapshot_id, action_step, token, timeout=None):
        """Queue Agent-authored room claims; return no private room or score data."""
        import re
        if (not isinstance(rooms, list) or not 1 <= len(rooms) <= 16
                or any(not isinstance(room, str) or not re.fullmatch(r'[a-z][a-z0-9_]*_[0-9]+', room)
                       for room in rooms)
                or type(snapshot_id) is not int or snapshot_id < 0
                or type(action_step) is not int or action_step < 0):
            return {'ok': False, 'status': 400, 'message': 'Invalid arrival declaration.'}
        if not isinstance(token, str) or not token:
            return {'ok': False, 'status': 403, 'message': 'Agent control token required.'}
        return self._submit_transaction(ArrivalDeclarations(tuple(dict.fromkeys(rooms)), snapshot_id, action_step),
                                        {}, timeout, control_token=token)

    def submit_answer(self, answer, snapshot_id, action_step, token, timeout=None):
        """Serialize answer reporting under the same lease as robot commands."""
        if (not isinstance(answer, str) or not answer.strip() or len(answer) > 4096
                or type(snapshot_id) is not int or snapshot_id < 0
                or type(action_step) is not int or action_step < 0):
            return {'ok': False, 'status': 400, 'message': 'Invalid answer declaration.'}
        if not isinstance(token, str) or not token:
            return {'ok': False, 'status': 403, 'message': 'Agent control token required.'}
        return self._submit_transaction(AnswerDeclaration(answer, snapshot_id, action_step),
                                        {}, timeout, control_token=token)

    def submit_done(self, snapshot_id, action_step, token, timeout=None):
        """Queue a nonmotion terminal declaration under the existing Agent lease."""
        if (type(snapshot_id) is not int or snapshot_id < 1
                or type(action_step) is not int or action_step < 0):
            return {'ok': False, 'status': 400, 'message': 'Invalid DONE declaration.'}
        if not isinstance(token, str) or not token:
            return {'ok': False, 'status': 403, 'message': 'Agent control token required.'}
        return self._submit_transaction(DoneDeclaration(snapshot_id, action_step),
                                        {}, timeout, control_token=token)

    def submit_temporal(self, event, answer, snapshot_id, action_step, token, timeout=None):
        """Serialize phase declarations under the same lease as physical actions."""
        if (event not in ('phase_1_complete', 'phase_1_budget_exhausted', 'answer')
                or (event in ('phase_1_complete', 'phase_1_budget_exhausted') and answer is not None)
                or (event == 'answer' and (not isinstance(answer, str) or not answer.strip() or len(answer) > 160))
                or type(snapshot_id) is not int or snapshot_id < 0
                or type(action_step) is not int or action_step < 0):
            return {'ok': False, 'status': 400, 'message': 'Invalid temporal event.'}
        if not isinstance(token, str) or not token:
            return {'ok': False, 'status': 403, 'message': 'Agent control token required.'}
        if not self.temporal_state()['temporal_episode']:
            return {'ok': False, 'status': 409, 'message': 'Temporal event is unavailable.'}
        return self._submit_transaction(TemporalDeclaration(event, answer, snapshot_id, action_step),
                                        {}, timeout, control_token=token)

    def submit_finalization(self, snapshot_id, action_step, reason, token, timeout=None):
        """Queue runner-owned terminal scoring without a model declaration."""
        if (type(snapshot_id) is not int or snapshot_id < 1
                or type(action_step) is not int or action_step < 0
                or not isinstance(reason, str) or not reason or len(reason) > 64):
            return {'ok': False, 'status': 400, 'message': 'Invalid episode finalization.'}
        if not isinstance(token, str) or not token:
            return {'ok': False, 'status': 403, 'message': 'Agent control token required.'}
        return self._submit_transaction(EpisodeFinalization(snapshot_id, action_step, reason),
                                        {}, timeout, control_token=token)

    def cancel_agent(self, token):
        with self.frame_cond:
            if self.control_owner == "human":
                return {"ok": True, "control_owner": "human"}
            if not isinstance(token, str) or not secrets.compare_digest(token, self._control_token or ""):
                return {"ok": False, "status": 403, "message": "Invalid Agent control token."}
            self.cancel_active_action()
            return {"ok": True, "message": "Agent cancellation requested."}

    def submit(self, kind, arg=None, timeout=None, *, control_token=None) -> dict:
        if kind not in self.ALLOWED_COMMANDS:
            return {"ok": False, "status": 405, "message": "This operation is disabled in ESI control mode"}
        tokens = [arg] if kind == "move" else arg
        if (not isinstance(tokens, (list, tuple)) or not tokens
                or any(not self.action_allowed(token) for token in tokens)):
            return {"ok": False, "status": 405, "message": "Action is disabled for this ESI control mode"}
        if len(tokens) > MAX_BATCH_TOKENS:
            return {"ok": False, "status": 400, "message": "Too many actions"}
        return self._submit_transaction(list(tokens), {}, timeout, control_token=control_token)

    def _submit_transaction(self, command, slot, timeout, *, control_token=None, action_mode=None):
        done, cancel = threading.Event(), threading.Event()
        with self.frame_cond:
            if (self.temporal_state()['temporal_phase'] == 'phase_1'
                    and not isinstance(command, (ArrivalDeclarations, TemporalDeclaration, AnswerDeclaration, DoneDeclaration, EpisodeFinalization))
                    and not all(survey_action_allowed(item) for item in (command if isinstance(command, list) else [command]))):
                return {'ok': False, 'status': 409, 'message': 'Phase 1 permits observation only; robot commands are unavailable.'}
            if control_token is not None:
                if (self.control_owner != "agent" or not isinstance(control_token, str)
                        or not secrets.compare_digest(control_token, self._control_token or "")):
                    return {"ok": False, "status": 403, "message": "Invalid Agent control token."}
                if self._agent_cancelled:
                    return {"ok": False, "status": 409, "message": "Agent run was cancelled."}
                if action_mode is not None and self.action_mode != action_mode:
                    return {"ok": False, "status": 400, "message": "Command does not match this episode's action mode."}
            elif self.control_owner != "human":
                return {"ok": False, "status": 409, "message": "Agent owns robot control; human movement is locked."}
            if self.closing.is_set():
                return {"ok": False, "status": 503, "message": "ESI control is shutting down"}
            if self.busy:
                return {"ok": False, "status": 409, "message": "A command is already running"}
            self.busy = True
            slot["_queued_at"] = time.perf_counter()
            self.active_cancel = cancel
            self.commands.put_nowait((command, done, cancel, slot))
            self.frame_cond.notify_all()
        if not done.wait(timeout):
            # A timed-out pending command must not execute later. A running
            # macro checks cancellation between physics steps and stops safely.
            cancel.set()
            return {"ok": False, "status": 504, "message": "Request timed out; stopping command"}
        return slot

    def _publish_result(self, result, history_entry):
        reader = getattr(self.executor, "controller_state_provider", None)
        if reader is not None:
            self.controller_snapshot = reader()
        control_feedback = AgentActionFeedback.from_result(result, state=self.controller_snapshot)
        feedback = (self.observation_source.last_action_feedback
                    if self.observation_source is not None else
                    AgentActionFeedback.from_result(result, state=self.controller_snapshot,
                                                    config=self.observation_config))
        with self.frame_cond:
            self.last_result = feedback.to_dict()
            self.steps += 1
            execution = result.get("execution", {})
            actual_fixed = execution.get("actual_fixed_primitive_executions", execution.get("executed_count", 1))
            self.primitive_actions += actual_fixed
            self.equivalent_fixed_primitives += execution.get("equivalent_fixed_primitives", 1)
            self.actual_fixed_primitive_executions += actual_fixed
            self.metric_target_executions += execution.get("metric_target_executions", 0)
            self.physics_steps += result.get("control_steps", 0)
            self.physics_steps_known = self.physics_steps_known and result.get("physics_steps_known", "control_steps" in result)
            self.history = (self.history + [history_entry])[-12:]
            actual = feedback.estimated_motion
            magnitude = actual.get("distance_m", actual.get("delta_m", actual.get("angle_deg")))
            unit = "deg" if "angle_deg" in actual else "m"
            summary = feedback.observable_error or feedback.execution_status
            if actual.get("target_clamped"):
                summary += f" · target limited to {actual['target_height_m'] * 100:.2f} cm (control height)"
            self.message = (f"{result['action']}: {magnitude:.3f} {unit} · {summary}"
                            if magnitude is not None else f"{result['action']}: {summary}")
        return control_feedback, feedback

    def cancel_active_action(self):
        """Human CANCEL interrupts the current macro at its next physics step."""
        with self.frame_cond:
            if self.control_owner == "agent":
                self._agent_cancelled = True
                if not self.busy:
                    self._release_control()
                    return True
            if not self.busy or self.active_cancel is None:
                return False
            self.active_cancel.set()
            return True

    def process_next(self, *, should_stop=None) -> bool:
        _owner_thread()
        try:
            tokens, done, cancel, slot = self.commands.get_nowait()
        except queue.Empty:
            return False
        results = []
        action_timings = []
        command_started = time.perf_counter()
        queued_at = slot.pop("_queued_at", None)
        counters = getattr(getattr(self.observation_source, "session", None), "performance", None)
        if queued_at is not None and counters is not None:
            counters.add("rpc_queue_wait_time", time.perf_counter()-queued_at)
        control_outcomes = []

        def cancelled():
            session = getattr(self.observation_source, 'session', None)
            temporal = getattr(session, 'temporal_phase_controller', None)
            survey_failed = getattr(temporal, 'phase', None) == 'phase_1' and getattr(temporal, 'survey_violation', False)
            return cancel.is_set() or self.closing.is_set() or survey_failed or bool(should_stop and should_stop())

        try:
            if tokens is None:
                if cancelled():
                    slot.update(ok=False, status=409, message="Episode initialization cancelled.")
                    return True
                resume_step = slot.pop("resume_action_step", None)
                if resume_step is not None:
                    slot.pop("task_text")
                    slot.pop("world_advances_during_inference")
                    self.observation_source.resume_episode()
                    self.observation_source.capture(action_boundary=True)
                    session = getattr(self.observation_source, "session", None)
                    evaluator = getattr(session, "task_evaluator", None)
                    if evaluator is not None:
                        evaluator.record(self.observation_source.boundary_observation, action=None)
                    history = getattr(session, 'candidate_public_history', None)
                    if history is not None:
                        slot['candidate_observation_history'] = history
                    slot.update(ok=True, message="Agent resumed at the retained action boundary.",
                                resumed_action_step=resume_step)
                    return True
                self.episode_task = slot.pop("task_text")
                native_session = getattr(self.observation_source, 'session', None)
                if native_session is not None:
                    native_session.episode_id = self.agent_run_id
                self.world_policy.begin(slot.pop("world_advances_during_inference", True), self.simulation_time())
                self.history, self.steps, self.last_result = [], 0, None
                self.primitive_actions, self.physics_steps, self.physics_steps_known = 0, 0, True
                self.equivalent_fixed_primitives = self.actual_fixed_primitive_executions = self.metric_target_executions = 0
                self.observation_source.begin_episode(self.episode_task)
                history = getattr(native_session, 'candidate_public_history', None)
                if history is not None:
                    deliver = getattr(native_session, 'candidate_public_history_delivery', None)
                    if not callable(deliver):
                        raise RuntimeError('Candidate history lacks its native delivery binding.')
                    slot['candidate_observation_history'] = history
                    native_session.candidate_history_delivery_receipt = deliver()
                slot.update(ok=True, message="Agent episode ready at the current scene.")
                return True
            if isinstance(tokens, AnswerDeclaration):
                if cancelled():
                    slot.update(ok=False, status=409, message='Answer declaration cancelled.')
                    return True
                try:
                    slot.update(self.observation_source.report_answer(tokens.answer,
                        snapshot_id=tokens.snapshot_id, action_step=tokens.action_step))
                except (ValueError, RuntimeError):
                    slot.update(ok=False, status=409, message='Answer declaration is unavailable at this boundary.')
                return True
            if isinstance(tokens, DoneDeclaration):
                if cancelled():
                    slot.update(ok=False, status=409, message='DONE declaration cancelled.')
                    return True
                try:
                    slot.update(self.observation_source.report_done(
                        snapshot_id=tokens.snapshot_id, action_step=tokens.action_step))
                except (ValueError, RuntimeError):
                    slot.update(ok=False, status=409, message='DONE declaration is unavailable at this boundary.')
                return True
            if isinstance(tokens, EpisodeFinalization):
                if cancel.is_set() or self.closing.is_set():
                    slot.update(ok=False, status=409, message='Terminal evaluation cancelled.')
                    return True
                try:
                    slot.update(self.observation_source.finalize_episode(snapshot_id=tokens.snapshot_id,
                        action_step=tokens.action_step, reason=tokens.reason))
                except (ValueError, RuntimeError):
                    slot.update(ok=False, status=409, message='Terminal evaluation is unavailable at this boundary.')
                return True
            if isinstance(tokens, ArrivalDeclarations):
                if cancelled():
                    slot.update(ok=False, status=409, message='Arrival declaration cancelled.')
                    return True
                try:
                    self.observation_source.report_arrivals(tokens.rooms,
                        snapshot_id=tokens.snapshot_id, action_step=tokens.action_step)
                except ValueError:
                    slot.update(ok=False, status=409, message='Arrival declaration boundary changed.')
                    return True
                slot.update(ok=True, message='Arrival declarations recorded.')
                return True
            if isinstance(tokens, TemporalDeclaration):
                def failure_evidence(reason, error=None):
                    session = getattr(self.observation_source, 'session', None)
                    evaluator = getattr(session, 'task_evaluator', None)
                    output = getattr(evaluator, 'output', None)
                    if output is not None:
                        from pathlib import Path
                        from wia.runtime.scene_capture_queue import write_json
                        write_json(Path(output) / 'temporal_event_failure.json', {
                            'event': tokens.event, 'snapshot_id': tokens.snapshot_id,
                            'action_step': tokens.action_step, 'reason': reason,
                            'private_error': None if error is None else f'{type(error).__name__}: {error}'})
                if cancelled():
                    session = getattr(self.observation_source, 'session', None)
                    survey_failed = getattr(getattr(session, 'temporal_phase_controller', None),
                                            'survey_violation', False)
                    reason = 'phase_1_read_only_violation' if survey_failed else 'cancelled'
                    failure_evidence(reason)
                    slot.update(ok=False, status=409, message=(
                        'Phase 1 read-only safety check failed.' if survey_failed else 'Temporal event cancelled.'))
                    return True
                try:
                    result = self.observation_source.report_temporal(tokens.event, tokens.answer,
                        snapshot_id=tokens.snapshot_id, action_step=tokens.action_step)
                except (ValueError, RuntimeError) as error:
                    failure_evidence('transition_unavailable', error)
                    slot.update(ok=False, status=409, message='Temporal event is unavailable at this boundary.')
                    return True
                if result['observation_changed']:
                    snapshot = self.observation_source.latest_snapshot
                    if self.agent_snapshot is not snapshot.observation:
                        self.publish(snapshot)
                slot.update(result)
                return True
            if self.temporal_state()['temporal_phase'] == 'phase_1':
                commands = tokens if isinstance(tokens, list) else [tokens]
                if not all(survey_action_allowed(item) for item in commands):
                    slot.update(ok=False, status=409, message='Phase 1 forbids object interaction.')
                    return True
            self.owner_phase("executing")
            if isinstance(tokens, ParameterizedAction):
                def progress(value):
                    with self.frame_cond:
                        self.macro_progress = value
                        unit = "target executions" if value.get("strategy") == "metric_target" else "primitives"
                        self.message = (f"{tokens.action}: {value['executed_count']}/{value['planned_count']} {unit}")
                result = self.executor.execute_macro(tokens,
                    resolution=self.executor.primitive_resolution, limits=self.executor.macro_limits,
                    should_cancel=cancelled, on_progress=progress)
                action_timings.append(public_action_timing(result.get("timing")))
                control_feedback, feedback = self._publish_result(result, tokens.to_dict())
                ok = control_feedback.permits_next_action
                slot.update(ok=ok, results=[feedback.to_dict()], message=self.message,
                    macro={"command": result["command"], "compilation": result["compilation"],
                           "execution": dict(result["execution"], status=control_feedback.execution_status,
                                             completed=ok),
                           "duration_s": result["duration_s"],
                           "physics_steps": result["control_steps"] if result["physics_steps_known"] else None,
                           "actual": dict(feedback.estimated_motion)})
                if not ok:
                    slot["status"] = 504 if control_feedback.execution_status in ("timeout", "cancelled") else 409
                return True
            for token in tokens:
                if cancelled():
                    break
                result = self.executor.execute(token, should_cancel=cancelled)
                action_timings.append(public_action_timing(result.get("timing")))
                control_feedback, feedback = self._publish_result(result, token)
                control_outcomes.append(control_feedback)
                results.append(feedback.to_dict())
                if not control_feedback.permits_next_action:
                    break
            ok = len(results) == len(tokens) and all(feedback.permits_next_action for feedback in control_outcomes)
            slot.update(ok=ok, results=results, message=self.message if results else "Command cancelled")
            if not ok:
                slot["status"] = 504 if cancelled() or any(r.execution_status == "timeout" for r in control_outcomes) else 409
        except BaseException:
            slot.update(ok=False, status=500, message="Execution failed; simulator stopped")
            raise
        finally:
            if action_timings:
                # Final capture has already returned. This metadata follows the
                # response into existing execution/MCP archives, outside prompts.
                slot["timing"] = {"command_wall_time_s": time.perf_counter() - command_started,
                                  "actions": action_timings}
            safety = getattr(getattr(self.observation_source, 'session', None), 'safety_observer', None)
            if safety is not None:
                safety.action = None
            with self.frame_cond:
                self.owner_phase("awaiting_model" if self.control_owner == "agent" else "human_idle")
                if not isinstance(tokens, (DoneDeclaration, EpisodeFinalization)):
                    slot["world_timing"] = self.world_policy.snapshot()
                self.busy = False
                self.active_cancel = None
                self.macro_progress = None
                if self._agent_cancelled or (tokens is None and not slot.get("ok")):
                    self._release_control()
            done.set()
        return True

    def simulation_time(self):
        reader = getattr(self, "simulation_time_reader", None)
        if reader is not None:
            return float(reader())
        session = getattr(self.observation_source, "session", None)
        sim = getattr(getattr(getattr(session, "pipeline", None), "og", None), "sim", None)
        return float(getattr(sim, "current_time", self.world_policy.sim_time))

    def owner_phase(self, phase):
        if phase == "human_idle" and self.metadata.get("debug_retain"):
            phase = "debug_hold"
        _owner_thread()
        with self.frame_cond:
            self.world_policy.transition(phase, self.simulation_time())

    def owner_sample(self, *, idle_step=False):
        _owner_thread()
        with self.frame_cond:
            self.world_policy.sample(self.simulation_time(), idle_step=idle_step)

    def should_advance_world(self):
        with self.frame_cond:
            return self.world_policy.should_advance(self.control_owner)

    def shutdown(self):
        _owner_thread()
        self.closing.set()
        try:
            _, done, cancel, slot = self.commands.get_nowait()
        except queue.Empty:
            return
        cancel.set()
        slot.update(ok=False, status=503, message="ESI control is shutting down")
        with self.frame_cond:
            self.busy = False
        done.set()

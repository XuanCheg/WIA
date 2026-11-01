"""MCP tools for the R1 HTTP executor; all physics stays with the existing owner.

Only execution feedback and robot RGB cameras enter tool results. World poses,
scene/evaluation state and numerical IK diagnostics remain in the backend logs.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
from numbers import Real
from typing import Annotated, Literal
from urllib.parse import urlsplit

import requests
from mcp.server import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import Field

from core.action_units import ROBOT_ACTIONS
from core.action_interface import BackendCapabilities
from core.action_compiler import MacroLimits, ParameterizedAction, PrimitiveResolution, compile_action
from core.observations import AgentActionFeedback, AgentObservation, AgentRobotState, validate_macro_receipt
from core.robot_cameras import RGB_TO_VIEW
from core.performance import public_action_timing

# Generate the tool's enum from the canonical action vocabulary.
R1_ACTIONS = ROBOT_ACTIONS
ActionToken = Literal[R1_ACTIONS]
CameraView = Literal[tuple(RGB_TO_VIEW.values())]
# Matches the existing /api/step limit. The backend remains authoritative.
MAX_ACTIONS = 64
Actions = Annotated[list[ActionToken], Field(min_length=1, max_length=MAX_ACTIONS)]

def tool_result(data, *, image_bytes=None):
    """Keep error feedback in both model-readable content and structuredContent."""
    content = [TextContent(type="text", text=json.dumps(data, ensure_ascii=False, allow_nan=False))]
    if image_bytes is not None:
        content.append(ImageContent(type="image", mime_type="image/jpeg",
                                    data=base64.b64encode(image_bytes).decode("ascii")))
    return CallToolResult(content=content, structured_content=data, is_error=not data["ok"])


def execution_feedback(result):
    # The backend now returns the same strict projection used by AgentObservation.
    # Reject older/raw debug results rather than reintroducing a second whitelist.
    return AgentActionFeedback.from_payload(result).to_dict()


def public_state(state):
    robot = state.get("robot_state")
    robot = AgentRobotState.from_payload(robot).to_dict() if robot is not None else None
    last = state.get("last_result")
    return {
        "mode": state.get("mode"), "read_only": state.get("read_only", True),
        "capabilities": state.get("capabilities"),
        "action_groups": {key: bool(state.get("action_groups", {}).get(key, False)) for key in ("base", "arm", "gripper", "trunk")},
        "busy": bool(state.get("busy")), "steps": state.get("steps", 0),
        "control_owner": state.get("control_owner", "human"), "agent_run_id": state.get("agent_run_id"),
        "temporal_episode": bool(state.get("temporal_episode", False)),
        "temporal_phase": state.get("temporal_phase"),
        **{key: state.get(key) for key in ("world_mode", "world_advances_during_inference", "episode_phase",
            "model_wait_wall_time_s", "simulation_time_during_model_wait_s", "idle_controller_env_steps",
            "action_execution_sim_time_s", "action_execution_wall_time_s", "sampled_simulation_time_s")},
        "available_actions": [x for x in state.get("step_tokens", ()) if x in R1_ACTIONS],
        "available_cameras": [x for x in RGB_TO_VIEW.values() if x in state.get("available_views", ())],
        "frame_seq": state.get("frame_seq"), "frame_time": state.get("frame_time"),
        "observation_snapshot_id": state.get("observation_snapshot_id"),
        "simulation_clock": state.get("simulation_clock"),
        "observation_captured_at": state.get("observation_captured_at"),
        "task_text": state.get("task_text"), "robot_type": state.get("robot_type"),
        "sim": state.get("sim"), "working_arm": state.get("working_arm"),
        "history": state.get("history", []), "last_token": state.get("last_token"),
        "robot_state": robot, "observation_config": state.get("observation_config"),
        "task_progress": None,
        "last_execution": execution_feedback(last) if isinstance(last, dict) else None,
        "grasp_mode": state.get("grasp_mode"), "rotation_enabled": bool(state.get("rotation_enabled", False)),
    }


def public_context(state):
    base, arm = state.get("base_config") or {}, state.get("arm_config") or {}
    context = {
        "task_text": state.get("task_text"), "robot_type": state.get("robot_type"),
        "working_arm": state.get("working_arm"), "frame": state.get("base_frame"),
        "action_groups": public_state(state)["action_groups"],
        "capabilities": public_state(state)["capabilities"],
        "available_actions": public_state(state)["available_actions"],
        "steps": {"arm_translation_m": arm.get("translation_m"),
                  "base_translation_m": base.get("translation_m"),
                  "base_rotation_deg": base.get("rotation_deg")},
        "tolerances": {"arm_translation_m": arm.get("translation_tolerance_m"),
                       "arm_orientation_deg": arm.get("orientation_tolerance_deg"),
                       "base_translation_m": base.get("translation_tolerance_m"),
                       "base_rotation_deg": base.get("rotation_tolerance_deg")},
        "arm_orientation_hold": "base_relative" if arm else None,
        "camera_shape": state.get("camera_shape"),
        "camera_labels": {view: state.get("camera_labels", {}).get(view) for view in RGB_TO_VIEW.values()},
        "variable_step_enabled": False,
        "task_evaluation_available": False,
        "grasp_mode": state.get("grasp_mode"),
        "world_advances_during_inference": state.get("world_advances_during_inference", True),
        "world_mode": state.get("world_mode", "run"),
        "temporal_episode": bool(state.get("temporal_episode", False)),
        "temporal_phase": state.get("temporal_phase"),
    }
    trunk = state.get("trunk_config")
    if trunk:
        context["steps"]["trunk_translation_m"] = trunk["translation_m"]
        context["trunk_height_range_m"] = [trunk["minimum_height_m"], trunk["maximum_height_m"]]
        context["trunk_height_range_reference"] = "trunk_control_height_m: nominal_urdf_workspace"
    if state.get("rotation_enabled"):
        rotation = state.get("rotation_config") or {}
        context["steps"]["arm_rotation_deg"] = rotation.get("rotation_deg")
        context["tolerances"]["arm_rotation_deg"] = rotation.get("rotation_tolerance_deg")
        context["rotation_frame"] = state.get("rotation_frame")
    if state.get("parameterized_actions"):
        setting = state["parameterized_actions"]
        resolution = PrimitiveResolution(**setting["resolution"])
        limits = MacroLimits(**setting["limits"])
        from core.action_compiler import action_bounds
        context["parameterized_actions"] = {"resolution": resolution.to_dict(), "limits": limits.to_dict(),
                                           "bounds_by_action": action_bounds(context["available_actions"], resolution, limits),
                                           "request_timeout_s": None}
        context["variable_step_enabled"] = True
    from core.feedback_semantics import FEEDBACK_SEMANTICS
    context["feedback_semantics"] = FEEDBACK_SEMANTICS
    return context


class RobotBridge:
    """Network-only bridge: no simulator imports, commands, sensor reads or retries."""

    def __init__(self, backend_url):
        parsed = urlsplit(backend_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("Backend URL must be an HTTP(S) URL without credentials, query or fragment.")
        self.backend_url = backend_url.rstrip("/")

    async def request(self, method, path, *, body=None, read_timeout=None):
        # A worker performs HTTP only. The simulator's main thread still owns
        # the queue, every physics step and every camera capture.
        return await asyncio.to_thread(requests.request, method, self.backend_url + path,
                                       json=body, timeout=None)

    async def state(self, *, context=False):
        try:
            response = await self.request("GET", "/api/agent/state")
            response.raise_for_status()
            state = response.json()
            if not isinstance(state, dict):
                raise ValueError("Robot state must be a JSON object.")
            key = "context" if context else "state"
            return tool_result({"ok": True, key: public_context(state) if context else public_state(state)})
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return tool_result({"ok": False, "reason": "state_unavailable",
                                "message": "Robot state unavailable; no motion was requested."})

    async def camera(self, view):
        if view not in RGB_TO_VIEW.values():
            return tool_result({"ok": False, "reason": "invalid_camera",
                                "message": "Only robot-mounted RGB views are available to the Agent."})
        try:
            response = await self.request("GET", f"/api/agent/snapshot/{view}")
            response.raise_for_status()
            if (not response.headers.get("Content-Type", "").startswith("image/jpeg")
                    or not response.content.startswith(b"\xff\xd8")):
                raise ValueError("Camera response must be a JPEG.")
            seq = response.headers.get("X-Frame-Seq")
            return tool_result({"ok": True, "camera": view, "mime_type": "image/jpeg",
                                "frame_seq": int(seq) if seq is not None else None,
                                "frame_time": response.headers.get("X-Frame-Time")},
                               image_bytes=response.content)
        except (requests.RequestException, ValueError):
            return tool_result({"ok": False, "reason": "camera_unavailable",
                                "message": f"Robot {view} camera unavailable; no motion was requested."})

    async def execute(self, tokens, *, control_token=None):
        if (not isinstance(tokens, list) or not 1 <= len(tokens) <= MAX_ACTIONS
                or any(not isinstance(x, str) or x not in R1_ACTIONS for x in tokens)):
            return tool_result({"ok": False, "completed": False, "reason": "invalid_action",
                                "outcome_known": True, "results": [],
                                "message": "Only enabled fixed BASE/MV/ROTATE/GRASP/RELEASE and RIGHT_* arm tokens are supported; no motion was requested."})
        sent = False
        try:
            state_response = await self.request("GET", "/api/agent/state")
            state_response.raise_for_status()
            state = state_response.json()
            capabilities = state.get("capabilities")
            available = (BackendCapabilities(**capabilities).available_actions
                         if isinstance(capabilities, dict) else state.get("step_tokens", ()))
            if (state.get("read_only", True) is not False or any(
                    token not in state.get("step_tokens", ())
                    or token not in available
                    for token in tokens)):
                return tool_result({"ok": False, "completed": False, "reason": "action_disabled",
                                    "outcome_known": True, "requested_actions": tokens, "results": [],
                                    "message": "Requested action group is disabled; no motion was requested."})
            sent = True
            # Native motion uses simulation-time deadlines; wait for its terminal receipt.
            if control_token is not None and len(tokens) != 1:
                return tool_result({"ok": False, "reason": "invalid_action", "outcome_known": True,
                                    "results": [], "message": "Agent episodes accept one fixed action."})
            response = await self.request("POST", "/api/agent/step" if control_token is not None else "/api/step",
                                          body={"action": tokens[0], "control_token": control_token}
                                          if control_token is not None else {"tokens": tokens},
                                          read_timeout=None)
            payload = response.json()
            return self._execution_result(tokens, response.status_code, payload)
        except (requests.RequestException, ValueError, TypeError, AttributeError):
            return tool_result({"ok": False, "completed": False, "reason": "execution_feedback_unavailable",
                                "outcome_known": not sent, "motion_stopped": None,
                                "requested_actions": tokens, "results": [],
                                "message": ("No execution result received. Read robot state before another motion."
                                            if sent else "Robot unavailable; no motion was requested.")})

    async def execute_parameterized(self, command, *, control_token):
        sent = False
        try:
            if not isinstance(control_token, str) or not control_token:
                raise ValueError("Agent control token required.")
            # Validate once here for early rejection, then independently on the
            # simulator server and canonical macro executor. No repeated MCP calls.
            command = ParameterizedAction.from_payload(command, R1_ACTIONS)
            response = await self.request("GET", "/api/agent/state")
            response.raise_for_status()
            state = response.json()
            setting = state.get("parameterized_actions")
            if not setting:
                raise ValueError("Parameterized commands are disabled on this backend.")
            capabilities = BackendCapabilities(**state["capabilities"])
            resolution = PrimitiveResolution(**setting["resolution"])
            limits = MacroLimits(**setting["limits"])
            try:
                compile_action(command, capabilities, resolution, limits)
            except ValueError as error:
                return tool_result({"ok": False, "completed": False, "reason": "invalid_parameterized_command",
                    "outcome_known": True, "motion_stopped": None, "results": [],
                    "message": f"{error} No motion was requested."})
            sent = True
            response = await self.request("POST", "/api/agent/command",
                body={"command": command.to_dict(), "control_token": control_token},
                read_timeout=None)
            payload = response.json()
            result = self._execution_result([command.action], response.status_code, payload)
            if payload.get("macro") is not None:
                feedback = AgentActionFeedback.from_payload(payload["results"][0])
                macro = validate_macro_receipt(payload["macro"], command=command,
                    capabilities=capabilities, resolution=resolution, limits=limits, feedback=feedback)
                data = dict(result.structured_content, macro=macro)
                return tool_result(data)
            if result.structured_content["ok"]:
                raise ValueError("Missing aggregate execution receipt.")
            return result
        except (requests.RequestException, ValueError, TypeError, AttributeError, KeyError, IndexError):
            return tool_result({"ok": False, "completed": False,
                "reason": "execution_feedback_unavailable" if sent else "invalid_parameterized_command",
                "outcome_known": not sent, "motion_stopped": None, "results": [],
                "message": "No aggregate receipt received; never retry motion." if sent else "Command rejected; no motion was requested."})

    def _execution_result(self, tokens, status, payload):
        raw = payload.get("results", [])
        if not isinstance(raw, list) or len(raw) > len(tokens):
            raise ValueError("Invalid action execution feedback.")
        results = [AgentActionFeedback.from_payload(item) for item in raw]
        if any(result.action != tokens[index] for index, result in enumerate(results)):
            raise ValueError("Mismatched action execution feedback.")
        failed = next((result for result in results if result.execution_status not in ("completed", "unavailable")), None)
        ok = 200 <= status < 300 and payload.get("ok") is True and len(results) == len(tokens) and failed is None
        reason = ((failed.observable_error or failed.execution_status) if failed is not None else "completed" if ok
                  else {400: "invalid_request", 403: "control_not_owned", 405: "action_disabled", 409: "robot_busy_or_rejected",
                        503: "robot_unavailable", 504: "request_timeout"}.get(status, "incomplete_execution"))
        data = {"ok": ok, "completed": ok, "execution_status": "completed" if ok else reason,
                "reason": reason, "http_status": status, "outcome_known": bool(results) or status in (400, 403, 405, 409),
                "motion_stopped": results[-1].motion_stopped if results else None,
                "requested_actions": tokens, "results": [result.to_dict() for result in results],
                "unexecuted_actions": tokens[len(results):], "message": reason}
        timing = payload.get("timing")
        if isinstance(timing, dict):
            elapsed, actions = timing.get("command_wall_time_s"), timing.get("actions")
            if (isinstance(elapsed, Real) and not isinstance(elapsed, bool)
                    and math.isfinite(elapsed) and elapsed >= 0
                    and isinstance(actions, list) and len(actions) == len(results)):
                data["timing"] = {"command_wall_time_s": elapsed,
                                  "actions": [public_action_timing(item) for item in actions]}
        from core.feedback_semantics import CONTINUABLE_MOTION_ERRORS
        if (reason in CONTINUABLE_MOTION_ERRORS and data["outcome_known"] is True
                and data["motion_stopped"] is True and len(results) == 1):
            data.update(can_continue=True, automatic_replay=False, smaller_same_action_allowed=True,
                message=f"{reason}: motion stopped. Read the new observation and continue from the measured pose. "
                        "Choose a smaller same-direction command, retreat, or adjust the base/trunk. "
                        "Remaining primitives were discarded; do not automatically replay the previous command.")
        return tool_result(data)

    async def observation(self, *, include_sensor_data=False):
        try:
            response = await self.request("GET", "/api/agent-observation")
            response.raise_for_status()
            payload = response.json()
            observation = AgentObservation.from_payload(payload)
            metadata = observation.metadata()
            if include_sensor_data:
                # from_payload has already validated these immutable snapshot
                # bytes and provenance. Forward them without another NPY encode.
                metadata["sensors"] = payload["sensors"]
                metadata["perception"] = payload["perception"]
            data = {"ok": True, "observation": metadata}
            result = tool_result(data)
            for name in observation.rgb_images():
                info = payload["images"][name]
                if info is not None:
                    result.content.append(ImageContent(type="image", mime_type="image/png", data=info["png_base64"]))
            return result
        except (requests.RequestException, ValueError, TypeError, AttributeError, KeyError):
            return tool_result({"ok": False, "reason": "observation_unavailable",
                                "message": "Canonical robot observation unavailable; no motion was requested."})


def create_server(backend_url="http://127.0.0.1:8615"):
    bridge = RobotBridge(backend_url)
    server = MCPServer(
        "Show-Harness R1", version="0.5.2",
        instructions=("Use these MCP tools for Agent input and actions; the GUMI webpage is the human debug monitor. "
                      "Read robot_get_context initially. Use robot_get_observation for an atomic head/wrist "
                      "snapshot, compact real-robot proprioception and execution feedback. Enabled inputs "
                      "with no sensor/estimator adapter are unavailable. Base world poses, assisted grasp "
                      "attachment, task progress and answers are private. Execution status and target_reached "
                      "describe robot commands, never task success. A completed GRASP means the close request "
                      "finished; use allowed sensors to infer whether an object is held. Fixed BASE/MV/GRASP/RELEASE "
                      "tokens and enabled RIGHT_* arm variants only. RIGHT_* addresses the right hand; unprefixed arm commands address the left. ROTATE_CW/CCW use local Z; enabled ROTATE_PITCH_POS/NEG use local Y and ROTATE_ROLL_POS/NEG use local X. Tilt signs follow the right-hand rule. A network timeout does not "
                      "confirm a stop; read state before another motion. Sensor arrays can be included explicitly "
                      "when needed; observation metadata describes their availability and provenance."),
    )

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    async def robot_get_context() -> CallToolResult:
        """Read the task, capabilities, camera labels and fixed steps/tolerances in the robot frame."""
        return await bridge.state(context=True)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    async def robot_get_state() -> CallToolResult:
        """Read capabilities and the last action's result, limit reason and recovery/hold status."""
        return await bridge.state()

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    async def robot_get_observation(include_sensor_data: bool = False) -> CallToolResult:
        """Read one action-boundary RGB pair, robot state and execution feedback.

        Set include_sensor_data to retrieve available depth/LiDAR/force arrays as
        non-pickle NPY payloads; otherwise return their provenance/availability.
        """
        return await bridge.observation(include_sensor_data=include_sensor_data)

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
    async def robot_get_camera(view: CameraView) -> CallToolResult:
        """Read a cached robot RGB image. Camera requests do not advance physics."""
        return await bridge.camera(view)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False))
    async def robot_execute_actions(tokens: Actions, control_token: str | None = None) -> CallToolResult:
        """Execute enabled fixed base/left-arm/wrist/gripper macros; stop the batch on failure.

        Trunk height is available only when enabled. No parameterized distances/angles or controller overrides.
        A runner holding Agent control supplies control_token and exactly one token;
        calls without it use human control and are rejected during an Agent episode.
        """
        return await bridge.execute(tokens, control_token=control_token)

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False))
    async def robot_execute_parameterized(command: dict, control_token: str) -> CallToolResult:
        """Submit ONE bounded metric command under the runner's Agent lease.

        Motion requires distance_m or angle_deg; GRASP/RELEASE and RIGHT_GRASP/RIGHT_RELEASE have no magnitude. RIGHT_* addresses the right hand; unprefixed arm commands address the left.
        The server compiles/executes the entire macro on its owner thread, then
        returns one receipt and one new observation boundary. Never replay it.
        """
        return await bridge.execute_parameterized(command, control_token=control_token)

    return server

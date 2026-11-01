"""Runner-owned MCP connection and control lease; neither is given to the VLM."""
from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager

import requests
from mcp import Client

from core.action_interface import BackendCapabilities
from core.action_compiler import MacroLimits, ParameterizedAction, PrimitiveResolution, compile_action
from core.observations import AgentActionFeedback, AgentObservation, validate_macro_receipt
from core.agent.trace_log import compact_json, timestamp


@asynccontextmanager
async def motion_transport(url):
    """Keep simulator receipt streams open until a result or explicit cancellation."""
    import httpx2
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    async with create_mcp_http_client(timeout=httpx2.Timeout(None)) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            yield streams


class MCPRobotEnvironment:
    def __init__(self, url, *, run_id, task_text, action_mode="fixed", world_advances_during_inference=True,
                 episode_contract=None, resume_action_step=None):
        if action_mode not in ("fixed", "parameterized"):
            raise ValueError("Unknown action mode.")
        self.url, self.run_id, self.task_text = url.rstrip("/"), run_id, task_text
        self.action_mode = action_mode
        if type(world_advances_during_inference) is not bool:
            raise ValueError("world_advances_during_inference must be boolean.")
        self.world_advances_during_inference = world_advances_during_inference
        self.read_timeout_s = None
        self.control_token = None
        self.client = None
        self.capabilities = None
        self.initialization = None
        self.action_context = None
        self.last_observation = None
        self._input_logger = None
        self._pending_input_events = []
        self.episode_contract = episode_contract
        self.resume_action_step = resume_action_step
        self.temporal_episode = False
        self.temporal_phase = None
        self._pending_temporal_boundary = None
        self._initial_overview_binding = None
        self.done_declaration_available = False
        self.episode_finalization_available = False
        self.requires_done_declaration = False

    def attach_input_logger(self, logger):
        self._input_logger = logger
        if logger is not None:
            for event in self._pending_input_events:
                logger(event)
            self._pending_input_events.clear()

    def _record_input(self, event):
        if self._input_logger is None:
            self._pending_input_events.append(event)
        else:
            self._input_logger(event)

    async def _call_tool(self, name, arguments):
        requested = timestamp()
        try:
            result = await self.client.call_tool(name, arguments)
        except BaseException as error:
            self._record_input({"tool": name, "arguments": compact_json(arguments),
                "requested_at": requested, "at": timestamp(), "status": "transport_error",
                "error_type": type(error).__name__, "result": None})
            raise
        self._record_input({"tool": name, "arguments": compact_json(arguments),
            "requested_at": requested, "at": timestamp(), "status": "received",
            "result": compact_json(result.model_dump(mode="json", by_alias=True))})
        return result

    async def _http(self, path, body):
        response = await asyncio.to_thread(requests.post, self.url + path, json=body, timeout=None)
        data = response.json()
        if data.get("ok") is not True:
            raise RuntimeError(data.get("message", "Robot control request failed."))
        response.raise_for_status()
        return data

    async def __aenter__(self):
        current = None
        if self.resume_action_step is not None:
            response = await asyncio.to_thread(requests.get, self.url + "/api/agent/state", timeout=None)
            response.raise_for_status()
            current = response.json()
            if self.episode_contract is not None and not current.get("declarative_resume_available"):
                raise RuntimeError("The running simulator does not support declarative resume; preserve it until an explicit restart is authorized.")
            if (current.get("steps") != self.resume_action_step or current.get("busy")
                    or current.get("control_owner") != "human" or current.get("task_text") != self.task_text):
                raise RuntimeError("Live action boundary changed after interruption; refuse to reset or replay commands.")
        if self.episode_contract is not None:
            response = await asyncio.to_thread(requests.get, self.url + "/api/agent/state", timeout=None)
            response.raise_for_status()
            current = response.json()
            if current.get("episode_contract") != self.episode_contract:
                raise RuntimeError("Simulator and Agent disagree on the immutable episode manifest.")
        claim = {"run_id": self.run_id, "task_text": self.task_text,
                 "world_advances_during_inference": self.world_advances_during_inference}
        if self.resume_action_step is not None and self.episode_contract is not None:
            claim["resume_action_step"] = self.resume_action_step
        if self.action_mode == "parameterized":
            response = await asyncio.to_thread(requests.get, self.url + "/api/agent/state", timeout=None)
            response.raise_for_status()
            current = response.json()
            setting = current.get("parameterized_actions")
            if not setting:
                raise RuntimeError("Backend must explicitly enable parameterized commands.")
            self.macro_limits = MacroLimits(**setting["limits"])
            self.read_timeout_s = None
            claim["action_mode"] = self.action_mode
        if current is None:
            response = await asyncio.to_thread(requests.get, self.url + '/api/agent/state', timeout=None)
            response.raise_for_status()
            current = response.json()
        self.done_declaration_available = current.get('done_declaration_available') is True
        self.episode_finalization_available = current.get('episode_finalization_available') is True
        self.requires_done_declaration = current.get('requires_done_declaration') is True
        if self.requires_done_declaration and not self.done_declaration_available:
            raise RuntimeError('This task requires terminal declaration support from its simulator.')
        initialized = await self._http("/api/control/claim", claim)
        self.control_token = initialized.pop("control_token")
        self.initialization = initialized
        self.temporal_episode = initialized.get('temporal_episode', False) is True
        self.temporal_phase = initialized.get('temporal_phase') if self.temporal_episode else None
        if self.resume_action_step is not None:
            self.initialization["interruption_resume"] = {
                "previous_simulator_action_step": self.resume_action_step,
                "new_lease_action_step": self.resume_action_step if self.episode_contract is not None else 0,
                "physical_scene_reset": False}
        try:
            self.client = Client(motion_transport(self.url + "/mcp"), read_timeout_seconds=None)
            await self.client.__aenter__()
            context = await self._call_tool("robot_get_context", {})
            if context.is_error:
                raise RuntimeError("MCP robot context unavailable.")
            context = context.structured_content["context"]
            self.capabilities = BackendCapabilities(**context["capabilities"])
            self.action_context = {key: context.get(key) for key in ("frame", "working_arm", "steps", "rotation_frame",
                "trunk_height_range_m", "trunk_height_range_reference", "feedback_semantics")}
            if self.temporal_episode:
                if self.temporal_phase not in ('phase_1', 'phase_2'):
                    raise RuntimeError('Temporal episode phase is unavailable.')
                self.action_context.update(temporal_episode=True, temporal_phase=self.temporal_phase)
            if self.action_mode == "parameterized":
                setting = context["parameterized_actions"]
                limits = MacroLimits(**setting["limits"])
                if limits != self.macro_limits:
                    raise RuntimeError("Backend macro limits changed during initialization.")
                self.primitive_resolution = PrimitiveResolution(**setting["resolution"])
                self.action_context["parameterized_actions"] = setting
            if tuple(context["available_actions"]) != self.capabilities.available_actions:
                raise RuntimeError("Robot action list disagrees with canonical capabilities.")
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            if self.control_token:
                try:
                    await self._http("/api/control/release", {"control_token": self.control_token})
                except (requests.RequestException, RuntimeError):
                    # Unknown transport outcome: request cancellation, never replay a motion.
                    await self._http("/api/control/cancel", {"control_token": self.control_token})
                    while True:
                        response = await asyncio.to_thread(requests.get, self.url + "/api/agent/state", timeout=None)
                        if not response.json().get("busy"):
                            await self._http("/api/control/release", {"control_token": self.control_token})
                            break
                        await asyncio.sleep(0.1)
        finally:
            self.control_token = None
            if self.client is not None:
                await self.client.__aexit__(exc_type, exc, traceback)
                self.client = None

    async def observe(self):
        result = await self._call_tool("robot_get_observation", {"include_sensor_data": True})
        if result.is_error:
            raise RuntimeError("MCP canonical observation unavailable.")
        payload = copy.deepcopy(result.structured_content["observation"])
        images = iter(item for item in result.content if item.type == "image")
        for name in payload["images"]:
            if payload["images"][name] is not None:
                item = next(images, None)
                if item is None or item.mime_type != "image/png":
                    raise RuntimeError("MCP did not supply all lossless canonical robot RGB views.")
                payload["images"][name]["png_base64"] = item.data
        if next(images, None) is not None:
            raise RuntimeError("Unexpected MCP image input.")
        observation = AgentObservation.from_payload(payload)
        if not observation.action_boundary or observation.task_text != self.task_text:
            raise RuntimeError("Expected this episode's canonical action-boundary observation.")
        identity = (self.initialization or {}).get('initial_overview_identity')
        try:
            overview = observation.require_initial_overview(identity)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        if self.episode_contract is not None and self.episode_contract.get('episode_id') != overview.episode_id:
            raise RuntimeError('Initial overview does not match the declared episode.')
        binding = (overview.metadata(), observation.metadata()['images']['initial_overview_rgb']['sha256'])
        if self._initial_overview_binding is not None and binding != self._initial_overview_binding:
            raise RuntimeError('Initial global overview changed after episode start; live refresh is forbidden.')
        self._initial_overview_binding = binding
        if self.initialization is not None:
            self.initialization['initial_overview_required'] = True
            self.initialization['initial_overview_status'] = 'available'
        if self._pending_temporal_boundary is not None:
            previous_snapshot, action_step = self._pending_temporal_boundary
            if observation.snapshot_id <= previous_snapshot or observation.action_step != action_step:
                raise RuntimeError('Temporal phase requires a fresh snapshot at the retained action boundary.')
            self._pending_temporal_boundary = None
        self.last_observation = observation
        return observation

    async def world_metrics(self):
        try:
            response = await asyncio.to_thread(requests.get, self.url + "/api/agent/state", timeout=None)
        except requests.ReadTimeout:
            # Timing telemetry is not a command receipt or an observation boundary.
            # Preserve missing evidence without discarding a completed model reply.
            return {"world_metrics_validity": "unavailable", "world_metrics_error": "read_timeout"}
        response.raise_for_status()
        state = response.json()
        keys = ("world_advances_during_inference", "world_mode", "episode_phase", "model_wait_wall_time_s",
                "simulation_time_during_model_wait_s", "idle_controller_env_steps", "action_execution_sim_time_s",
                "action_execution_wall_time_s", "sampled_simulation_time_s", "temporal_episode", "temporal_phase")
        return {key: state[key] for key in keys if key in state}

    async def report_arrivals(self, rooms, observation):
        """Submit only model-authored declarations before the associated motion."""
        arguments = {'rooms': list(rooms), 'snapshot_id': observation.snapshot_id,
                     'action_step': observation.action_step, 'control_token': self.control_token}
        requested = timestamp()
        result = await self._http('/api/agent/arrivals', arguments)
        public = {'ok': result['ok'], 'message': result['message']}
        self._record_input({'tool': 'robot_report_arrivals', 'arguments': compact_json(arguments),
            'requested_at': requested, 'at': timestamp(), 'status': 'received', 'result': public})
        return public

    async def report_answer(self, answer, observation):
        """Submit public answer text; correctness remains private to the scorer."""
        arguments = {'answer': answer, 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'control_token': self.control_token}
        requested = timestamp()
        result = await self._http('/api/agent/answer', arguments)
        public = {'ok': result['ok'], 'message': result['message']}
        self._record_input({'tool': 'robot_report_answer', 'arguments': compact_json(arguments),
            'requested_at': requested, 'at': timestamp(), 'status': 'received', 'result': public})
        return public

    async def report_done(self, observation):
        """Declare actual model termination without a robot action or private result."""
        arguments = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                     'control_token': self.control_token}
        requested = timestamp()
        result = (await self._http('/api/agent/done', arguments) if self.done_declaration_available else
                  {'ok': True, 'message': 'Legacy backend does not advertise DONE declarations.',
                   'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})
        public = {key: result[key] for key in ('ok', 'message', 'snapshot_id', 'action_step')}
        if (public['ok'] is not True or type(public['snapshot_id']) is not int
                or public['snapshot_id'] < observation.snapshot_id
                or type(public['action_step']) is not int or public['action_step'] != observation.action_step):
            raise RuntimeError('DONE declaration returned an invalid boundary receipt.')
        self._record_input({'tool': 'robot_report_done', 'arguments': compact_json(arguments),
            'requested_at': requested, 'at': timestamp(),
            'status': 'received' if self.done_declaration_available else 'local_noop', 'result': public})
        return public

    async def finalize_episode(self, observation, *, reason):
        """Ask the owner to score the final state without exposing private results."""
        if not self.episode_finalization_available:
            if self.requires_done_declaration:
                raise RuntimeError('This simulator cannot score terminal states without DONE; restart it with the updated runtime.')
            return None
        arguments = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                     'reason': reason, 'control_token': self.control_token}
        requested = timestamp()
        result = await self._http('/api/agent/finalize', arguments)
        public = {key: result[key] for key in ('ok', 'message', 'snapshot_id', 'action_step')}
        if (public['ok'] is not True or type(public['snapshot_id']) is not int
                or public['snapshot_id'] < observation.snapshot_id
                or type(public['action_step']) is not int or public['action_step'] != observation.action_step):
            raise RuntimeError('Terminal evaluation returned an invalid boundary receipt.')
        self._record_input({'tool': 'episode_finalize', 'arguments': compact_json(arguments),
            'requested_at': requested, 'at': timestamp(), 'status': 'received', 'result': public})
        return public

    async def report_temporal(self, event, answer, observation):
        """Submit an Agent-authored phase/answer event without exposing native scoring."""
        if not self.temporal_episode:
            raise RuntimeError('Temporal event is unavailable for this episode.')
        arguments = {'event': event, 'answer': answer, 'snapshot_id': observation.snapshot_id,
                     'action_step': observation.action_step, 'control_token': self.control_token}
        requested = timestamp()
        result = await self._http('/api/agent/temporal', arguments)
        public = {key: result[key] for key in ('ok', 'message', 'phase', 'observation_changed')}
        if public['phase'] not in ('phase_1', 'phase_2') or type(public['observation_changed']) is not bool:
            raise RuntimeError('Temporal event returned an invalid public phase receipt.')
        self.temporal_phase = public['phase']
        self.action_context = dict(self.action_context or {}, temporal_episode=True, temporal_phase=self.temporal_phase)
        if public['observation_changed']:
            self._pending_temporal_boundary = (observation.snapshot_id, observation.action_step)
        self._record_input({'tool': 'robot_report_temporal', 'arguments': compact_json(arguments),
            'requested_at': requested, 'at': timestamp(), 'status': 'received', 'result': public})
        return public

    async def execute(self, action):
        if self.temporal_episode and self.temporal_phase == 'phase_1':
            from wia.tasks.temporal_policy import survey_action_allowed
            if not survey_action_allowed(action):
                raise ValueError('Phase 1 permits only base and torso survey motion; arm and gripper commands are disabled.')
        if self.action_mode == "parameterized":
            if not isinstance(action, ParameterizedAction):
                raise ValueError("Parameterized episodes require typed commands.")
            compile_action(action, self.capabilities, self.primitive_resolution, self.macro_limits)
            result = await self._call_tool("robot_execute_parameterized", {
                "command": action.to_dict(), "control_token": self.control_token})
            token = action.action
        else:
            result = await self._call_tool("robot_execute_actions", {
                "tokens": [action], "control_token": self.control_token})
            token = action
        data = result.structured_content
        if not isinstance(data, dict):
            raise RuntimeError("MCP execution result unavailable; action will not be retried.")
        feedback = [AgentActionFeedback.from_payload(item) for item in data.get("results", [])]
        if any(item.action != token for item in feedback) or len(feedback) > 1:
            raise RuntimeError("Mismatched action feedback.")
        if self.action_mode == "parameterized" and data.get("macro") is not None:
            if len(feedback) != 1:
                raise RuntimeError("Expected one aggregate macro result.")
            data["macro"] = validate_macro_receipt(data["macro"], command=action,
                capabilities=self.capabilities, resolution=self.primitive_resolution,
                limits=self.macro_limits, feedback=feedback[0])
        elif self.action_mode == "parameterized" and data.get("ok"):
            raise RuntimeError("Missing aggregate execution receipt; action will not be retried.")
        return data

"""Deterministic modular prompts. Use PromptAssembler with a DecisionInput and public task."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from core.action_compiler import MacroLimits, PrimitiveResolution, action_bounds
from core.agent.depth_inputs import depth_access
from core.agent.image_inputs import image_inputs, RGB_POLICY
from wia.runtime.agent.wia_history_projection import NEUTRAL_INSTRUCTION, PHYSICAL_CONTEXT_FIELDS, _physical_context
from core.robot_cameras import ROBOT_CAMERA_LABELS

PROMPT_ROOT = WIA_RUNTIME_ROOT / "explore_to_achieve/prompts"
COMPONENTS = frozenset(("core/identity.md", "core/perception.md", "core/robot_operation.md",
    "core/execution_feedback.md", "core/protection.md", "core/completion.md", "robots/r1.md", "actions/fixed.md",
    "actions/parameterized.md", "tools/native_depth.md", "phases/episode_start.md", "phases/decision_step.md"))
PLUGINS = {"world_observation": "plugins/world_observation.md"}
QA_COMPONENTS = {"core/identity.md": "qa/identity.md", "core/completion.md": "qa/completion.md"}
QA_EXTRA_COMPONENTS = frozenset(("qa/answer_and_place_completion.md",))
REACHABILITY_DEBUG_COMPONENT = "plugins/reachability_debug.md"
SLOTS = {
    "resident": frozenset(("ROBOT_PROFILE", "ACTION_INTERFACE", "PERCEPTION_TOOLS")),
    "episode_start": frozenset(("PUBLIC_TASK", "PUBLIC_ENVIRONMENT", "EPISODE_CONFIGURATION",
                              "INITIAL_OBSERVATION", "INITIAL_ROBOT_STATE", "OUTPUT_CONTRACT")),
    "decision_step": frozenset(("CURRENT_SENSOR_OBSERVATION", "CURRENT_ROBOT_STATE",
        "LAST_EXECUTION_FEEDBACK", "CURRENT_ACTION_CAPABILITIES", "REMAINING_BUDGET",
        "OBSERVATION_PROVENANCE", "OUTPUT_CONTRACT")),
}


def describe(label, value):
    return label + ":\n" + json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


class PromptAssembler:
    def __init__(self, root=PROMPT_ROOT, *, world_observation=True, task_mode="physical", qa_protocol=None,
                 wia_history_condition="full", debug_reachability=False):
        if task_mode not in ("physical", "original_qa") or task_mode == "original_qa" and not world_observation:
            raise ValueError("Original QA requires world-observation output for its final answer.")
        self.task_mode = task_mode
        if wia_history_condition not in ('full', 'history_removed', 'history_oracle'):
            raise ValueError('Unknown WIA history condition.')
        if wia_history_condition != 'full' and task_mode != 'physical':
            raise ValueError('WIA history diagnostics require physical task mode.')
        self.wia_history_condition = wia_history_condition
        self.qa_protocol = qa_protocol
        self.qa_components = dict(QA_COMPONENTS)
        if qa_protocol is not None and qa_protocol.goal == "answer_and_place":
            self.qa_components["core/completion.md"] = "qa/answer_and_place_completion.md"
        self.root = Path(root).resolve()
        self.plugins = {"world_observation": bool(world_observation)}
        if debug_reachability:
            if not world_observation:
                raise ValueError("Reachability debugging requires world-observation output.")
            self.plugins["reachability_debug"] = True
        self.main = (self.root / "main.md").read_text(encoding="utf-8")
        self.component_hashes = {}
        self.components = {name: self._read(name) for name in sorted(COMPONENTS | set(PLUGINS.values()))}
        if debug_reachability:
            self.components[REACHABILITY_DEBUG_COMPONENT] = self._read(REACHABILITY_DEBUG_COMPONENT)
        if task_mode == "original_qa":
            self.components.update({name: self._read(name) for name in self.qa_components.values()})
        self.sections = {name: [] for name in SLOTS}
        phase = None
        seen_sections = set()
        for line in self.main.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                phase = line[1:-1]
                if phase not in SLOTS or phase in seen_sections:
                    raise ValueError("Unknown or duplicate prompt phase.")
                seen_sections.add(phase)
                continue
            if phase is None:
                raise ValueError("Prompt directives must belong to a known phase.")
            directive, separator, name = line.partition(" ")
            if not separator or directive not in ("@include", "@slot", "@plugin"):
                raise ValueError("Use only whitelisted include and slot directives.")
            if directive == "@include" and name not in COMPONENTS:
                raise ValueError("Prompt include is not whitelisted.")
            if directive == "@slot" and name not in SLOTS[phase]:
                raise ValueError("Unknown slot for this prompt phase.")
            if directive == "@plugin" and (phase != "resident" or name not in PLUGINS):
                raise ValueError("Only whitelisted resident prompt plugins are accepted.")
            self.sections[phase].append((directive, name))
        if seen_sections != set(SLOTS):
            raise ValueError("All prompt phases must be declared.")
        for phase, entries in self.sections.items():
            slots = [name for directive, name in entries if directive == "@slot"]
            if len(slots) != len(set(slots)) or set(slots) != SLOTS[phase]:
                raise ValueError("Each phase must include every required slot exactly once.")
            includes = [name for directive, name in entries if directive == "@include"]
            required = ({"core/identity.md", "core/perception.md", "core/robot_operation.md",
                         "core/execution_feedback.md", "core/protection.md", "core/completion.md"} if phase == "resident" else
                        {"phases/" + phase + ".md"})
            if len(includes) != len(required) or set(includes) != required:
                raise ValueError("Each phase must retain all of its mandatory instruction components.")
        plugins = [name for directive, name in self.sections["resident"] if directive == "@plugin"]
        if len(plugins) != len(PLUGINS) or set(plugins) != set(PLUGINS):
            raise ValueError("Declare each optional plugin exactly once; enable it through assembler configuration.")

    def _read(self, name):
        if name not in COMPONENTS | set(PLUGINS.values()) | set(QA_COMPONENTS.values()) | QA_EXTRA_COMPONENTS | {REACHABILITY_DEBUG_COMPONENT}:
            raise ValueError("Prompt component is not whitelisted.")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Prompt components cannot escape the prompt root.")
        raw = path.read_bytes()
        self.component_hashes[name] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        if any(line.lstrip().startswith(("@include", "@slot", "@plugin")) for line in text.splitlines()):
            raise ValueError("Recursive prompt directives are forbidden.")
        return text.strip()

    def render(self, phase, slots):
        if phase not in SLOTS or set(slots) != SLOTS[phase] or any(not isinstance(value, str) for value in slots.values()):
            raise ValueError("Supply exactly the whitelisted text slots for this phase.")
        # Slot contents are data, never re-parsed as template instructions.
        parts = []
        for directive, name in self.sections[phase]:
            if directive == "@plugin":
                if self.plugins[name]:
                    parts.append(self.components[PLUGINS[name]])
            else:
                component = self.qa_components.get(name, name) if self.task_mode == "original_qa" else name
                text = self.components[component] if directive == "@include" else slots[name]
                if not self.initial_overview_required and component == 'phases/episode_start.md' and directive == '@include':
                    text = text.replace('first decision of a new episode',
                        'first decision of a new Phase-2 diagnostic conversation within the existing physical episode', 1)
                parts.append(text)
        return "\n\n".join(parts)

    @property
    def initial_overview_required(self):
        return self.wia_history_condition == 'full'

    def provenance(self):
        value = {"prompt_main_sha256": hashlib.sha256(self.main.encode()).hexdigest(),
                "prompt_component_hashes": dict(self.component_hashes), "prompt_plugins": dict(self.plugins),
                "task_mode": self.task_mode,
                "qa_protocol": self.qa_protocol.provenance() if self.qa_protocol is not None else None}
        if not self.initial_overview_required:
            value['wia_history_condition'] = self.wia_history_condition
        return value

    def _validate_history_projection(self, request, episode_configuration):
        if self.initial_overview_required:
            return
        context = request.action_context or {}
        intervention = context.get('wia_history_intervention')
        if (context.get('temporal_phase') != 'phase_2'
                or not isinstance(intervention, dict)
                or intervention.get('schema') != 'wia_phase2_history_projection_v1'
                or intervention.get('overview_removal_reason') != 'explicit_diagnostic_history_intervention'
                or intervention.get('native_capture_failure') is not False):
            raise ValueError('WIA diagnostics require an explicit projected Phase-2 intervention.')
        if set(intervention) - {'schema', 'overview_removal_reason', 'native_capture_failure',
                'fresh_native_conversation_required', 'diagnostic_prompt_assembler_required',
                'phase2_retry_feedback_allowed'}:
            raise ValueError('WIA diagnostic intervention metadata retains an unreviewed history channel.')
        if request.observation_condition != 'current' or episode_configuration.get('observation_condition') != 'current':
            raise ValueError('WIA diagnostics require current observations.')
        if request.previous_phase_observation is not None or request.public_memory or request.memory_mode != 'none':
            raise ValueError('WIA diagnostic inputs retain a request history channel.')
        if request.feedback is not None and intervention.get('phase2_retry_feedback_allowed') is not True:
            raise ValueError('WIA diagnostic rejection feedback requires an explicit Phase-2 retry marker.')
        allowed = PHYSICAL_CONTEXT_FIELDS | {'temporal_phase', 'temporal_instruction',
                                             'wia_history_intervention', 'diagnostic_historical_facts'}
        if set(context) - allowed or context.get('temporal_instruction') != NEUTRAL_INSTRUCTION:
            raise ValueError('WIA diagnostic action context retains historical or nonphysical text.')
        _, removed = _physical_context({key: context[key] for key in PHYSICAL_CONTEXT_FIELDS if key in context},
                                       request.available_actions)
        if removed:
            raise ValueError('WIA diagnostic physical context retains nested historical metadata.')
        if self.wia_history_condition == 'history_oracle':
            facts = context.get('diagnostic_historical_facts')
            if not isinstance(facts, str) or not facts.strip():
                raise ValueError('history_oracle requires explicit caller-supplied historical facts.')
        elif 'diagnostic_historical_facts' in context:
            raise ValueError('history_removed forbids historical oracle facts.')
        for observation in (request.observation, request.sensor_observation):
            if observation is None:
                continue
            if (observation.initial_overview is not None or observation.initial_overview_status != 'unavailable'
                    or observation.public_task_events):
                raise ValueError('WIA diagnostic observation retains overview or historical task events.')
            if any(observation.rgb_images().get(name) is None for name in ROBOT_CAMERA_LABELS):
                raise ValueError('WIA diagnostics must retain all eight current mounted RGB views.')

    def _overview_instruction(self):
        if self.initial_overview_required:
            return (RGB_POLICY + '\nEvery episode starts with one initial global RGB overview and all eight mounted cameras. Subsequent turns attach head, both wrists and rear chassis; the other four chassis views are available through read_rgb_camera. '
                    'The overview is frozen at the initialized reset world, with declared world orientation, extent and '
                    'robot-start context. It is retained as initial context and never refreshed into a live global view. '
                    'Use current mounted observations to assess subsequent changes.')
        return (RGB_POLICY + '\nDiagnostic Phase-2 history condition: historical channels and the frozen initial global overview are '
                'deliberately withheld, not a native capture failure. All eight mounted views are captured; subsequent turns attach four and offer the other four through read_rgb_camera, '
                'with current depth, calibration, proprioception and physical action limits unchanged. '
                'Any diagnostic historical facts are separate verbatim caller-supplied text; their factual accuracy '
                'is not independently verified and no target action or answer is inferred.')

    def task_text(self, public_task):
        if self.task_mode == 'original_qa':
            return public_task['objective']
        parts = [public_task['objective'].strip(),
                 'Physical success criteria: ' + public_task['success_criteria'].strip()]
        if public_task['constraints']:
            parts.append('Task constraints: ' + '; '.join(public_task['constraints']))
        return '\n\n'.join(parts)

    def validate_request(self, request, public_task):
        if request.observation.task_text != self.task_text(public_task):
            from core.agent.decision import ModelError
            raise ModelError('Public task changed within the episode.')

    def build(self, request, public_task, episode_configuration, *, first_turn):
        if request.memory_mode != "none":
            raise ValueError("The native task pathway does not use model_notes.")
        request = replace(request, world_observation_log=self.plugins["world_observation"],
                          qa_response=self.qa_protocol is not None)
        allowed_public = {"task_id", "task_version", "objective", "success_criteria", "environment", "constraints"}
        if set(public_task) != allowed_public:
            raise ValueError("Only the validated public task projection is accepted.")
        allowed_configuration = {"episode_id", "max_steps", "world_mode", "world_advances_during_inference",
                                 "observation_condition", "action_mode"}
        if set(episode_configuration) - {"temporal_policy", "run_mode"} != allowed_configuration:
            raise ValueError("Only public episode policies are accepted.")
        self._validate_history_projection(request, episode_configuration)
        obs, sensor = request.observation, request.sensor_inputs
        current, supplied = obs.metadata(), sensor.metadata()
        step = request.decision_step if request.decision_step is not None else obs.action_step + 1
        provenance = {"decision_step": step, "action_step": obs.action_step,
            "canonical_observation_id": obs.snapshot_id, "canonical_captured_at": obs.captured_at,
            "sensor_snapshot_id": sensor.snapshot_id, "captured_at": sensor.captured_at,
            "observation_condition": request.observation_condition,
            "sensor_is_frozen_initial": request.observation_condition == "initial_view_only",
            "initial_overview_required": self.initial_overview_required,
            "world_mode": episode_configuration["world_mode"],
            "world_advances_during_inference": episode_configuration["world_advances_during_inference"]}
        if not self.initial_overview_required:
            provenance.update(wia_history_condition=self.wia_history_condition,
                initial_overview_status='unavailable',
                initial_overview_reason='explicit_diagnostic_history_intervention',
                native_capture_failure=False, diagnostic_explanation=self._overview_instruction().strip())
        pose = sensor.robot_state.ee_pose if sensor.robot_state is not None else None
        control_pose = (pose.position_m, pose.quaternion_xyzw) if pose is not None else None
        sensor_data = {"provenance": provenance,
            "attached_images_in_order": [{"name": name, "label": label, "shape": list(pixels.shape),
                "transform": transform,
                "snapshot_id": (sensor.initial_overview.snapshot_id if name == 'initial_overview_rgb' else sensor.snapshot_id),
                "captured_at": (sensor.initial_overview.captured_at if name == 'initial_overview_rgb' else sensor.captured_at)}
                for name, pixels, label, transform in image_inputs(sensor, include_initial_overview=first_turn)],
            "images": supplied["images"], "sensors": supplied["sensors"],
            "observation_config": supplied["observation_config"], "perception": supplied["perception"],
            "initial_overview": supplied.get('initial_overview',
                {'required': True, 'status': 'unavailable', 'capture': None}),
            "public_task_events": supplied.get('public_task_events', []),
            "raw_metric_depth": depth_access(sensor), "camera_geometry": supplied["camera_geometry"],
            "camera_mounting_measurements": {name: geometry.mounting_measurements(control_pose)
                                             for name, geometry in sensor.camera_geometry.items()}}
        if not self.initial_overview_required:
            sensor_data['initial_overview'] = {'required': False, 'status': 'unavailable', 'capture': None,
                'reason': 'explicit_diagnostic_history_intervention', 'native_capture_failure': False,
                'refresh_policy': 'frozen_at_episode_start'}
        # Enabled non-depth sensor arrays remain measurements, not invented values.
        for name in ("lidar_points_m", "force_torque", "tactile"):
            value = getattr(sensor.sensors, name, None)
            if value is not None:
                sensor_data[name] = value.tolist()
        if sensor.perception is not None and sensor.perception.point_cloud_m is not None:
            sensor_data["perception_point_cloud_m"] = sensor.perception.point_cloud_m.tolist()
        context = dict(request.action_context or {})
        capabilities = {"available_actions": list(request.available_actions), "action_mode": request.action_mode,
                        "action_context": context}
        if request.action_mode == "parameterized":
            setting = context["parameterized_actions"]
            capabilities["per_action_bounds"] = action_bounds(request.available_actions,
                PrimitiveResolution(**setting["resolution"]), MacroLimits(**setting["limits"]))
        output = {"response_schema": request.structured_schema, "validation_schema": request.schema,
                  "one_command": True, "retry_feedback": request.feedback}
        output_text = describe("Exact structured output contract", output)
        if request.world_observation_log:
            output_text += ('\nReturn {"command":{...one valid action command...},'
                            '"world_observation":"Brief description of the supplied world."}. '
                            'Only command is executed. The observation is a public log, not memory or a success signal.')
            if self.qa_protocol is not None:
                output_text += ('\nAlso return the original task answer, confidence and reasoning fields at the top level. '
                    'reasoning is a brief public explanation of observed evidence, not a deliberation trace. '
                    'Use unsure while uncertain. Confidence follows the original task rules. ')
                if self.qa_protocol.goal == "answer_and_place":
                    output_text += ('A recognized answer at the confidence threshold completes only the question subtask; '
                        'its proposed native motion still executes. Continue real pick-and-place. '
                        'DONE ends the combined attempt; report the selected container and actual placement outcome. ')
                else:
                    output_text += ('DONE corresponds to the original stop action; an unsure DONE does not conclude the task. '
                        'A recognized answer with confidence at or above the threshold stops BEFORE any proposed motion, '
                        'as in the original task. ')
                output_text += ('At the last budgeted decision no further motion is executed. '
                    'An extra nonmotion final selection is requested if no recognized answer was given.')
        elif request.action_mode == "parameterized":
            output_text += '\nReturn {"command":{"action":ACTION,...required magnitude...}}. '
            output_text += 'GRASP, RELEASE, RIGHT_GRASP, RIGHT_RELEASE and DONE have only action inside command. No other fields.'
        else:
            output_text += '\nReturn {"action":ACTION} with no other fields.'
        resident = self.render("resident", {"ROBOT_PROFILE": self.components["robots/r1.md"],
            "ACTION_INTERFACE": self.components[f"actions/{request.action_mode}.md"] + "\n\n" + describe("Actual action capabilities", capabilities),
            "PERCEPTION_TOOLS": (self.components["tools/native_depth.md"] if depth_access(sensor)["available"]
                                 else "No model tools are authorized; native depth is unavailable.")
                + self._overview_instruction()})
        debug_prompt = self.components.get(REACHABILITY_DEBUG_COMPONENT)
        if debug_prompt is not None:
            resident += "\n\n" + debug_prompt
        state = {"snapshot_id": obs.snapshot_id, "captured_at": obs.captured_at,
                 "robot_state": current["robot_state"], "action_history": current["action_history"]}
        if first_turn:
            task = {key: value for key, value in public_task.items() if key != "environment"}
            phase = "episode_start"
            fresh = self.render(phase, {"PUBLIC_TASK": describe("Public task", task),
                "PUBLIC_ENVIRONMENT": describe("Public environment", public_task["environment"]),
                "EPISODE_CONFIGURATION": describe("Episode policy", episode_configuration),
                "INITIAL_OBSERVATION": describe("Initial authorized sensor observation", sensor_data),
                "INITIAL_ROBOT_STATE": describe("Initial measured robot state and feedback",
                    dict(state, last_action_feedback=current["last_action_feedback"])), "OUTPUT_CONTRACT": output_text})
            prompt = resident + "\n\n" + fresh
        else:
            phase = "decision_step"
            prompt = self.render(phase, {"CURRENT_SENSOR_OBSERVATION": describe("Authorized sensor observation", sensor_data),
                "CURRENT_ROBOT_STATE": describe("Current measured robot state", state),
                "LAST_EXECUTION_FEEDBACK": describe("Last execution feedback", current["last_action_feedback"]),
                "CURRENT_ACTION_CAPABILITIES": describe("Current action capabilities", capabilities),
                "REMAINING_BUDGET": describe("Remaining decision budget", request.steps_remaining),
                "OBSERVATION_PROVENANCE": describe("Observation provenance and policies", provenance),
                "OUTPUT_CONTRACT": output_text})
            if debug_prompt is not None:
                prompt += "\n\n" + debug_prompt
        if self.task_mode == 'physical':
            prompt += ("\n\nPhysical completion contract (applies to this decision): "
                + self.components['core/completion.md']
                + f"\nThis episode permits {episode_configuration['max_steps']} decisions; "
                  f"{request.steps_remaining} remain including this decision. "
                  "Use this budget, not a default from another benchmark or an earlier episode.")
        return replace(request, assembled_prompt=prompt, resident_prompt=resident,
                       prompt_phase=phase, decision_step=step)

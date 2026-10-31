"""Milestone 7: observe, choose one fixed action, execute, and observe again."""
from __future__ import annotations

import asyncio
import json
import time
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from typing import Protocol

from core.action_interface import BackendCapabilities
from core.action_units import DONE_ATOM
from core.action_compiler import ParameterizedAction, compile_action
from core.observations import AgentObservation
from core.episode_archive import EpisodeArchive
from core.feedback_semantics import CONTINUABLE_MOTION_ERRORS

from wia.runtime.agent.vercel_session import InvalidToolDecision

from core.agent.decision import DecisionAgent, DecisionInput, ModelError, final_text_only, parse_action, unpack_memory_decision
from core.agent.image_inputs import image_inputs
from core.agent.depth_inputs import depth_access
from core.agent.trace_log import EpisodeTrace, INPUT_LOG, OUTPUT_LOG, model_metadata


RECOVERABLE_ERRORS = CONTINUABLE_MOTION_ERRORS


def safely_recoverable(result, action):
    feedback = result.get("results", [])
    return (result.get("outcome_known") is True and result.get("motion_stopped") is True
            and result.get("reason") in RECOVERABLE_ERRORS and len(feedback) == 1
            and feedback[0].get("action") == action and feedback[0].get("motion_stopped") is True
            and (feedback[0].get("observable_error") or feedback[0].get("execution_status")) == result.get("reason"))


def _temporal_declaration(text, phase):
    """Recognize exactly one standalone marker from an accepted model output."""
    if not isinstance(text, str):
        return None
    completions = re.findall(r'^[ \t]*PHASE1_COMPLETE[ \t]*$', text, re.MULTILINE)
    answers = re.findall(r'^[ \t]*ANSWER:([^\r\n]*)$', text, re.MULTILINE)
    if len(completions) + len(answers) != 1:
        return None
    if phase == 'phase_1' and len(completions) == 1:
        return 'phase_1_complete', None
    if phase == 'phase_2' and len(answers) == 1:
        value = answers[0]
        if value and value[0] in ' \t' and 1 <= len(value.strip()) <= 160:
            return 'answer', value.strip()
    return None


class RobotEnvironment(Protocol):
    capabilities: BackendCapabilities
    initialization: dict

    async def observe(self) -> AgentObservation: ...

    async def execute(self, action: str) -> dict: ...


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _save_model_inputs(sensor_observation, canonical_observation, directory, output, *, snapshot_cache,
                       history_intervention=None, model_images=None):
    """Persist exactly the selected sensor bundle, keeping canonical captures separate."""
    projection = []
    for name, pixels, label, transform in (image_inputs(sensor_observation) if model_images is None else model_images):
        path = EpisodeArchive(output).model_image(sensor_observation, name, pixels, transform, use_cache=snapshot_cache)
        projection.append({'name': name, 'label': label, 'transform': transform,
                           'shape': list(pixels.shape), 'path': str(path.relative_to(output))})
    (directory / 'model_inputs.json').write_text(json.dumps(projection, indent=2) + '\n')
    if snapshot_cache:
        artifacts = canonical_observation.artifacts() if history_intervention is None else sensor_observation.artifacts()
        information = {
            'snapshot_id': canonical_observation.snapshot_id, 'captured_at': canonical_observation.captured_at,
            'model_sensor_snapshot_id': sensor_observation.snapshot_id,
            'canonical_references': artifacts.references, 'model_images': projection,
            'sensor_metadata': sensor_observation.metadata()['sensors'], 'encoding_counts': artifacts.stats,
        }
        if history_intervention is not None:
            information.update(snapshot_id=sensor_observation.snapshot_id,
                               captured_at=sensor_observation.captured_at,
                               canonical_observation='before/observation.json',
                               canonical_observation_snapshot_id=canonical_observation.snapshot_id,
                               wia_history_intervention=history_intervention,
                               model_observation=sensor_observation.metadata())
        (directory / 'snapshot_artifacts.json').write_text(json.dumps(information, indent=2) + '\n')
    (directory / 'raw_depth_input.json').write_text(json.dumps(depth_access(sensor_observation), indent=2) + '\n')
    if history_intervention is not None:
        (directory / 'wia_history_projection.json').write_text(json.dumps({
            **history_intervention, 'canonical_observation_preserved': True,
            'model_sensor_metadata': sensor_observation.metadata()}, indent=2) + '\n')
    return projection


async def run_episode(environment: RobotEnvironment, agent: DecisionAgent, output: Path, *,
                      max_steps=8, max_invalid_retries=2, success_provider=None, max_recoverable_failures=None,
                      action_mode="fixed", memory_mode="none", max_duration_s=None,
                      observation_condition="current", episode_manifest=None, integrity_check=None,
                      resume_summary=None, expected_action_step=None, budget_extension=False):
    episode_started = time.perf_counter()
    timings = {"observation_serialization_time": 0., "model_input_preparation_time": 0.,
               "action_execution_wall_time": 0., "observation_rpc_wall_time": 0.}
    if action_mode not in ("fixed", "parameterized"):
        raise ValueError("Unknown Agent action mode.")
    if memory_mode not in ("none", "model_notes") or observation_condition not in ("current", "initial_view_only"):
        raise ValueError("Unknown memory or observation condition.")
    temporal_episode = getattr(environment, 'temporal_episode', False) is True
    history_condition = getattr(agent, 'wia_history_condition', 'full')
    if temporal_episode and observation_condition != 'current':
        raise ValueError('Temporal episodes require current observations across phase transitions.')
    if max_duration_s is not None and (isinstance(max_duration_s, bool) or not isinstance(max_duration_s, (int, float))
                                    or not math.isfinite(max_duration_s) or max_duration_s <= 0):
        raise ValueError("Episode wall duration must be positive.")
    deadline = episode_started + max_duration_s if max_duration_s is not None else None
    public_memory = []
    if getattr(environment, "action_mode", action_mode) != action_mode:
        raise ValueError("Runner action mode must match the environment's control lease.")
    if action_mode == "parameterized":
        if not hasattr(environment, "primitive_resolution") or not hasattr(environment, "macro_limits"):
            raise ValueError("Parameterized mode requires actual backend increments and macro limits.")
    if type(max_steps) is not int or max_steps < 1 or type(max_invalid_retries) is not int or max_invalid_retries < 0:
        raise ValueError("Positive step limit and nonnegative invalid-output retry limit required.")
    if max_recoverable_failures is not None and (type(max_recoverable_failures) is not int or max_recoverable_failures < 0):
        raise ValueError("Recoverable-failure limit must be a nonnegative integer.")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("steps.jsonl", "summary.json", INPUT_LOG, OUTPUT_LOG)):
        raise ValueError("Use a new output directory to preserve existing episodes.")
    archive = EpisodeArchive(output)
    full_actions = environment.capabilities.available_actions + (DONE_ATOM,)

    from wia.tasks.temporal_policy import survey_policy, survey_action_allowed
    temporal_policy = survey_policy((episode_manifest or {}).get('temporal_policy') or getattr(environment, 'temporal_policy', None)) if temporal_episode else None
    if temporal_episode and temporal_policy['phase_1_max_decisions'] >= max_steps:
        raise ValueError('Phase-1 budget must leave a Phase-2 decision within the episode budget.')

    def public_action_context():
        context = dict(getattr(environment, 'action_context', None) or {})
        if temporal_episode:
            context['temporal_survey'] = {**temporal_policy,
                'current_phase': getattr(environment, 'temporal_phase', None),
                'phase_1_decisions_used': summary.get('phase_1_decisions', 0)}
        return context

    def phase_actions():
        return (tuple(action for action in full_actions if survey_action_allowed(action))
                if temporal_episode and getattr(environment, 'temporal_phase', None) == 'phase_1' else full_actions)

    actions = phase_actions()
    summary = {"started_at": timestamp(), "agent": agent.description(),
               "initialization": environment.initialization, "available_actions": list(actions),
               "max_steps": max_steps, "max_invalid_retries": max_invalid_retries,
               "task_success": None, "model_declared_done": False, "executed_actions": [],
               "termination_reason": "max_steps"}
    summary['extra_output_field_policy'] = 'ignore_unknown_fields_validate_known_values'
    summary.update(action_mode=action_mode, agent_decisions=0, model_calls=0, macro_actions=0,
                   primitive_actions=0, primitive_actions_known=True,
                   physics_steps=0, physics_steps_known=True,
                   physics_step_unit="controller_env_steps")
    advances = getattr(environment, "world_advances_during_inference", True)
    summary.update(memory_mode=memory_mode, observation_condition=observation_condition,
                   max_duration_s=max_duration_s, world_advances_during_inference=advances,
                   world_mode="run" if advances else "pause", public_memory=[])
    if history_condition != 'full':
        summary.update(wia_history_condition=history_condition, wia_history_interventions=[])
    summary.update(equivalent_fixed_primitives=0, actual_fixed_primitive_executions=0,
                   metric_target_executions=0, controller_env_steps=0)
    if temporal_episode:
        summary.update(temporal_episode=True, temporal_phase=getattr(environment, 'temporal_phase', None))
        summary['available_actions_by_phase'] = {'phase_1': list(phase_actions()), 'phase_2': list(full_actions)}
        summary.update(temporal_policy=temporal_policy, phase_1_decisions=0, temporal_transitions=[])
    start_step = 1
    previous_wall_s = 0.
    if budget_extension and resume_summary is None:
        raise ValueError("Budget extension requires an interrupted episode.")
    if resume_summary is not None:
        allowed_reasons = ("model_error", "cancelled")
        budget_matches = (max_steps > resume_summary["max_steps"] if budget_extension
                          else max_steps == resume_summary["max_steps"])
        if (resume_summary["termination_reason"] not in allowed_reasons
                or not budget_matches
                or not 0 <= resume_summary["agent_decisions"] < max_steps
                or observation_condition != "current" or memory_mode != "none"):
            raise ValueError("Only an interrupted current-observation episode with remaining budget can resume.")
        summary["segment_started_at"] = summary["started_at"]
        summary["started_at"] = resume_summary["started_at"]
        for key in ("agent_decisions", "model_calls", "macro_actions", "primitive_actions", "primitive_actions_known",
                    "physics_steps", "physics_steps_known", "equivalent_fixed_primitives", "actual_fixed_primitive_executions",
                    "metric_target_executions", "controller_env_steps"):
            summary[key] = resume_summary[key]
        summary["executed_actions"] = list(resume_summary["executed_actions"])
        summary["resume_from_model_calls"] = resume_summary["model_calls"]
        if temporal_episode:
            if resume_summary.get('temporal_policy') != temporal_policy:
                raise ValueError('Resume must preserve the original survey policy.')
            summary['phase_1_decisions'] = resume_summary['phase_1_decisions']
            summary['temporal_transitions'] = list(resume_summary.get('temporal_transitions', []))
            if resume_summary.get('temporal_phase') != getattr(environment, 'temporal_phase', None):
                raise ValueError('Resume phase differs from the retained native world; do not replay the event.')
        start_step = resume_summary["agent_decisions"] + 1
        previous_wall_s = resume_summary["timings"]["total_episode_wall_time"]
        if deadline is not None:
            deadline -= previous_wall_s
        timings.update({key: value for key, value in resume_summary["timings"].items()
                        if key != "total_episode_wall_time"})
        summary["budget_changes"] = list(resume_summary.get("budget_changes", []))
        if budget_extension:
            summary["budget_changes"].append({"previous_max_steps": resume_summary["max_steps"],
                "max_steps": max_steps, "at_decision": start_step, "reason": "user_requested_extension"})
    if episode_manifest is not None:
        summary["episode_manifest"] = episode_manifest
    if action_mode == "parameterized":
        summary["compilation_settings"] = {"resolution": environment.primitive_resolution.to_dict(),
                                           "limits": environment.macro_limits.to_dict()}
    trace = EpisodeTrace(output)
    qa_protocol = getattr(agent, "qa_protocol", None)
    qa_history = list(resume_summary.get("qa_history", [])) if resume_summary is not None else []
    if qa_protocol is not None:
        summary["qa_protocol"] = qa_protocol.provenance()
        summary["qa_history"] = qa_history
    summary["model_logs"] = trace.files
    if 'candidate_observation_history' in summary['initialization']:
        summary['initialization'] = {
            key: value for key, value in summary['initialization'].items()
            if key != 'candidate_observation_history'}
        summary['initialization']['candidate_history_file'] = str(output / 'candidate_observation_history.json')
    (output / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary['observation_coverage'] = list(resume_summary.get('observation_coverage', [])) if resume_summary else []
    def observation_coverage(value, stage):
        from core.robot_cameras import ROBOT_CAMERA_LABELS
        import numpy as np
        images = value.rgb_images()
        present = [name for name in ROBOT_CAMERA_LABELS if images.get(name) is not None and np.asarray(images[name]).size]
        empty = [name for name in present if not np.any(images[name])]
        row = {'snapshot_id': value.snapshot_id, 'action_step': value.action_step, 'stage': stage,
            'expected_views': list(ROBOT_CAMERA_LABELS), 'present_views': present,
            'missing_views': [name for name in ROBOT_CAMERA_LABELS if name not in present],
            'all_zero_views': empty, 'eight_view_capture_complete': len(present) == 8,
            'delivered_to_agent': False, 'target_visibility': 'unverified', 'answerability': 'unverified'}
        summary['observation_coverage'].append(row)
        return row
    record = None
    restore_candidate_history = None
    consume_invalid_step = getattr(agent, "invalid_decision_consumes_step", False)
    pending_invalid_feedback = None
    summary["invalid_decisions"] = list(resume_summary.get("invalid_decisions", [])) if resume_summary else []
    recovery, consecutive_failures = None, 0
    temporal_feedback = None
    pending_phase_observation = None
    if temporal_episode and resume_summary:
        transitions = summary['temporal_transitions']
        if transitions and not transitions[-1].get('final_phase_1_delivered'):
            from core.observations import AgentObservation
            pending_phase_observation = AgentObservation.from_payload(json.loads(
                Path(transitions[-1]['final_phase_1_payload']).read_text()))
    summary["max_recoverable_failures"] = max_recoverable_failures
    summary["recovered_execution_errors"] = list(resume_summary.get("recovered_execution_errors", [])) if resume_summary else []
    try:
        if hasattr(agent, "attach_output_logger"):
            agent.attach_output_logger(trace.public_event)
        if hasattr(agent, "attach_input_logger"):
            agent.attach_input_logger(trace.depth_response)
        observation = await environment.observe()
        candidate_history = (getattr(environment, 'initialization', None) or {}).get('candidate_observation_history')
        if candidate_history is not None:
            from wia.runtime.agent.candidate_history import decode_history, attach_history_images
            if getattr(agent, 'image_max_edge', None) is not None:
                raise ValueError('Candidate event history requires unmodified native image resolution.')
            frames = decode_history(candidate_history, observation)
            history_file = output / 'candidate_observation_history.json'
            history_file.write_text(json.dumps(candidate_history, indent=2) + '\n')
            restore_candidate_history, history_metadata = attach_history_images(agent, frames)
            environment.initialization.pop('candidate_observation_history', None)
            environment.initialization['candidate_history_file'] = str(history_file)
            environment.action_context = dict(getattr(environment, 'action_context', None) or {},
                candidate_history_frames=history_metadata,
                candidate_history_instruction='The attached chronological head-camera frames show an authored event before this decision. They are historical observations, not current views. No robot action was executed during that event.')
            summary['candidate_history_frames'] = history_metadata
        if observation.initial_overview_status != 'legacy_unavailable':
            overview = observation.require_initial_overview()
            summary['initial_overview_input'] = {'required': True, 'status': 'available',
                'identity': {key: overview.metadata()[key] for key in ('episode_id', 'reset_id', 'seed')},
                'condition': 'initial_global_rgb_for_all_tasks'}
        if expected_action_step is not None and observation.action_step != expected_action_step:
            raise ValueError("Live action boundary changed after interruption; refuse to replay or reset commands.")
        if resume_summary is not None:
            summary["simulator_action_step_offset"] = resume_summary["agent_decisions"] - observation.action_step
        initial_observation = observation
        summary["task_text"] = observation.task_text
        header_request = DecisionInput(observation, actions,
            action_context=public_action_context(),
            steps_remaining=max_steps-start_step+1, action_mode=action_mode, memory_mode=memory_mode,
            decision_step=start_step,
            observation_condition=observation_condition,
            sensor_observation=initial_observation if observation_condition == "initial_view_only" else None)
        if history_condition != 'full' and temporal_episode:
            from dataclasses import replace
            header_request = replace(header_request, action_context={**header_request.action_context,
                'temporal_episode': True, 'temporal_phase': getattr(environment, 'temporal_phase', None)})
        if hasattr(agent, "prepare_request"):
            header_request = agent.prepare_request(header_request)
        if episode_manifest is not None:
            summary["action_schema"] = header_request.schema
            summary["provider_output_schema"] = header_request.structured_schema
        trace.headers(header_request, summary["agent"], at=summary.get("segment_started_at", summary["started_at"]))
        if hasattr(environment, "attach_input_logger"):
            environment.attach_input_logger(trace.mcp_response)
        trace.set_stamp(0, observation=observation)
        trace.observation(observation, phase="resume" if resume_summary is not None else "initial")
        if success_provider is not None:
            summary["task_success"] = success_provider()
            if summary["task_success"] is True:
                summary["termination_reason"] = "environment_success"
                return summary
        for step in range(start_step, max_steps + 1):
            if integrity_check is not None:
                integrity_check()
            if deadline is not None and time.perf_counter() >= deadline:
                summary["termination_reason"] = "timeout"
                break
            model_observation = observation
            if observation_condition == "initial_view_only":
                # Preserve both canonical observations, with separate sensor and
                # current-state identities. Never fabricate a mixed snapshot.
                model_observation = initial_observation
            directory = output / f"step_{step:04d}"
            observation.set_serialization_cache(getattr(agent, "snapshot_cache", True))
            preparation_started = time.perf_counter()
            before = archive.save(observation, directory / "before")
            timings["observation_serialization_time"] += time.perf_counter()-preparation_started
            coverage = observation_coverage(model_observation, getattr(environment, 'temporal_phase', None) or 'episode')
            if pending_phase_observation is not None:
                observation_coverage(pending_phase_observation, 'final_phase_1')
            projection = []
            diagnostic_phase_2 = (history_condition != 'full' and temporal_episode
                                  and getattr(environment, 'temporal_phase', None) == 'phase_2')
            if not diagnostic_phase_2 and not callable(getattr(agent, 'request_images', None)):
                preparation_started = time.perf_counter()
                projection = _save_model_inputs(model_observation, observation, directory, output,
                    snapshot_cache=getattr(agent, 'snapshot_cache', True))
                timings['model_input_preparation_time'] += time.perf_counter() - preparation_started
            record = {"step": step, "started_at": timestamp(),
                      "observation": str(before.relative_to(output)), "attempts": [],
                      "parsed_action": None, "action_result": None, "next_observation": None,
                      "termination_reason": None}
            feedback, pending_invalid_feedback = pending_invalid_feedback, None
            action = None
            command = None
            before_snapshot, before_step = observation.snapshot_id, observation.action_step
            decision_phase = getattr(environment, 'temporal_phase', None) if temporal_episode else None
            phase_1_decision = decision_phase == 'phase_1'
            # Each invalid model decision consumes one survey/episode slot.
            # Transport retries remain bounded by the provider timeout policy.
            for attempt in range(1 if phase_1_decision or consume_invalid_step else max_invalid_retries + 1):
                entry = {"attempt": attempt + 1, "requested_at": timestamp()}
                trace.set_stamp(step, attempt + 1, observation)
                response_logged = False
                try:
                    current_actions = phase_actions()
                    context = dict(getattr(environment, "action_context", None) or {})
                    if temporal_episode:
                        phase = getattr(environment, 'temporal_phase', None)
                        context.update(temporal_episode=True, temporal_phase=phase,
                            phase_1_decisions_remaining=max(0, temporal_policy['phase_1_max_decisions'] - summary['phase_1_decisions']),
                            temporal_instruction=('Phase 1 permits base and torso survey motion only. Arm and gripper commands are disabled; incidental navigation contact is recorded without ending the survey. '
                                'Keep this episode and its earlier observations. '
                                'When Phase 1 inspection is complete, output a standalone PHASE1_COMPLETE '
                                'line in world_observation with command DONE to advance; plain DONE also requests transition. '
                                'The runner forces transition at the survey decision limit.'
                                if phase == 'phase_1' else
                                'Use the earlier Phase 1 observations and current Phase 2 observations. '
                                'Submit a standalone ANSWER: <public option> line in world_observation '
                                'with command DONE for the final answer.'))
                        if temporal_feedback:
                            context['temporal_feedback'] = temporal_feedback
                    if recovery:
                        context["recovery"] = recovery
                    if resume_summary is not None:
                        context["interruption_resume"] = {
                            "accepted_decisions_before_resume": start_step-1,
                            "simulator_counter_restarted_for_lease": True, "physical_scene_reset": False}
                    if budget_extension:
                        context["budget_update"] = summary["budget_changes"][-1]
                    request = DecisionInput(observation, current_actions, feedback,
                        action_context=context,
                        steps_remaining=max_steps-step+1, action_mode=action_mode, memory_mode=memory_mode,
                        decision_step=step,
                        public_memory=tuple(public_memory), observation_condition=observation_condition,
                        previous_phase_observation=pending_phase_observation,
                        sensor_observation=model_observation if observation_condition == "initial_view_only" else None)
                    if hasattr(agent, "prepare_request"):
                        request = agent.prepare_request(request)
                    if callable(getattr(agent, 'request_images', None)):
                        preparation_started = time.perf_counter()
                        projection = _save_model_inputs(request.sensor_inputs, observation, directory, output,
                            snapshot_cache=getattr(agent, 'snapshot_cache', True), model_images=agent.request_images(request))
                        timings['model_input_preparation_time'] += time.perf_counter() - preparation_started
                    if diagnostic_phase_2:
                        intervention = (request.action_context or {}).get('wia_history_intervention')
                        if (not isinstance(intervention, dict)
                                or intervention.get('schema') != 'wia_phase2_history_projection_v1'
                                or intervention.get('native_capture_failure') is not False
                                or intervention.get('overview_removal_reason') != 'explicit_diagnostic_history_intervention'
                                or (request.action_context or {}).get('temporal_phase') != 'phase_2'
                                or request.sensor_inputs.initial_overview is not None
                                or request.previous_phase_observation is not None):
                            raise ModelError('Diagnostic Phase-2 requests require the explicit prepared history projection.')
                        intervention = {**intervention, 'condition': history_condition}
                        preparation_started = time.perf_counter()
                        projection = _save_model_inputs(request.sensor_inputs, observation, directory, output,
                            snapshot_cache=getattr(agent, 'snapshot_cache', True), history_intervention=intervention)
                        timings['model_input_preparation_time'] += time.perf_counter() - preparation_started
                        coverage.update(wia_history_intervention=intervention,
                                        delivered_sensor_snapshot_id=request.sensor_inputs.snapshot_id,
                                        delivered_views=[item['name'] for item in projection])
                        summary['wia_history_interventions'].append({**intervention, 'step': step,
                            'attempt': attempt + 1, 'snapshot_id': request.sensor_inputs.snapshot_id})
                        trace.write(INPUT_LOG, 'wia_history_intervention', intervention=intervention,
                                    canonical_observation_preserved=True, passed_to_model=False)
                        if pending_phase_observation is not None:
                            transition = summary['temporal_transitions'][-1]
                            transition.update(final_phase_1_delivered=False, final_phase_1_withheld=True,
                                              final_phase_1_withheld_reason=intervention['overview_removal_reason'])
                            for previous in summary['observation_coverage']:
                                if (previous['snapshot_id'] == pending_phase_observation.snapshot_id
                                        and previous['stage'] == 'final_phase_1'):
                                    previous.update(delivered_to_agent=False, withheld_from_phase_2=True,
                                                    withheld_reason=intervention['overview_removal_reason'])
                            pending_phase_observation = None
                    prompt_path = directory / f"attempt_{attempt+1}_prompt.txt"
                    prompt_path.write_text(request.prompt(), encoding="utf-8")
                    entry["model_input"] = str(prompt_path.relative_to(output))
                    logged_request = request
                    if diagnostic_phase_2:
                        from dataclasses import replace
                        logged_request = replace(request, observation=request.sensor_inputs)
                    trace.step_input(logged_request, summary["agent"], projection, at=entry["requested_at"])
                    summary["model_calls"] += 1
                    wait_started = time.perf_counter()
                    if hasattr(environment, "world_metrics"):
                        entry["world_before_model"] = await environment.world_metrics()
                    try:
                        if deadline is None:
                            reply = await agent.decide(request)
                        else:
                            reply = await asyncio.wait_for(agent.decide(request), max(.001, deadline-time.perf_counter()))
                    finally:
                        entry["model_wait_wall_time_s"] = time.perf_counter()-wait_started
                    if hasattr(environment, "world_metrics"):
                        entry["world_after_model"] = await environment.world_metrics()
                    for key, value in reply.metadata.get("timings", {}).items():
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            timings[key] = timings.get(key, 0.) + value
                    coverage['delivered_to_agent'] = True
                    coverage['delivery_model_call'] = summary['model_calls']
                    previous_sent = request.previous_phase_observation
                    if (pending_phase_observation is not None and previous_sent is not None
                            and previous_sent.snapshot_id == pending_phase_observation.snapshot_id
                            and previous_sent.action_step == pending_phase_observation.action_step):
                        for prior_coverage in summary['observation_coverage']:
                            if prior_coverage['snapshot_id'] == pending_phase_observation.snapshot_id:
                                prior_coverage['delivered_to_agent'] = True
                                prior_coverage['delivery_model_call'] = summary['model_calls']
                        summary['temporal_transitions'][-1]['final_phase_1_delivered'] = True
                        summary['temporal_transitions'][-1]['delivery_model_call'] = summary['model_calls']
                        pending_phase_observation = None
                    from wia.runtime.agent.output_normalization import normalize_extra_fields
                    reply = normalize_extra_fields(request, reply)
                    raw = final_text_only(reply.raw_text)
                    metadata = model_metadata(reply.metadata)
                    received_at = timestamp()
                    trace.response(raw, metadata, at=received_at)
                    response_logged = True
                    entry.update(raw_output=raw, model_metadata=metadata, received_at=received_at)
                    if reply.metadata.get("output_normalization"):
                        entry["output_normalization"] = reply.metadata["output_normalization"]
                    note = None
                    if memory_mode == "model_notes":
                        raw, note, note_error = unpack_memory_decision(raw)
                        entry["memory_validation_error"] = note_error
                    if action_mode == "parameterized":
                        command = ParameterizedAction.parse(raw, current_actions)
                        compile_action(command, environment.capabilities,
                                       environment.primitive_resolution, environment.macro_limits)
                        action = command.action
                    else:
                        action = parse_action(raw, current_actions)
                    if qa_protocol is not None:
                        parsed_qa = qa_protocol.parse(metadata.get("qa_response"), action)
                        entry["qa_response"] = parsed_qa
                    entry["parse_error"] = None
                    normalization = reply.metadata.get('output_normalization')
                    if normalization:
                        accept_output = getattr(agent, 'accept_normalized_output', None)
                        if callable(accept_output):
                            accept_output(normalization['normalized_output'])
                        fields = normalization['ignored_fields']
                        notice = ('Previous decision accepted after ignoring extra output fields: '
                            + ', '.join(fields) + '. Only the validated command was selected; '
                            'refer to current execution feedback for its outcome. Omit these extra fields next time.')
                        pending_invalid_feedback = notice
                        entry['ignored_output_fields'] = fields
                        entry['normalization_feedback'] = notice
                        summary.setdefault('output_normalizations', []).append({'step': step,
                            'ignored_fields': fields, 'feedback': notice})
                    if note:
                        update = {"action_step": observation.action_step, "snapshot_id": model_observation.snapshot_id,
                                  "decision_step": step, "text": note}
                        public_memory = (public_memory + [update])[-4:]
                        summary["public_memory"].append(update)
                        record["memory_update"] = update
                    trace.validation(None, at=entry["received_at"])
                except ValueError as error:
                    action = command = None
                    feedback = str(error)
                    entry.update(parse_error=feedback, received_at=timestamp())
                    if not response_logged:
                        trace.request_error(error, status="request_error", at=entry["received_at"])
                    trace.validation(feedback, at=entry["received_at"])
                except InvalidToolDecision as error:
                    coverage["delivered_to_agent"] = True
                    coverage["delivery_model_call"] = summary["model_calls"]
                    action = command = None
                    feedback = str(error)
                    entry.update(parse_error=feedback, invalid_tool_decision=True, received_at=timestamp())
                    trace.request_error(error, status="invalid_decision", at=entry["received_at"])
                    trace.validation(feedback, at=entry["received_at"])
                except ModelError as error:
                    entry.update(model_error=str(error), received_at=timestamp())
                    if not response_logged:
                        trace.request_error(error, at=entry["received_at"])
                    record["attempts"].append(entry)
                    raise
                except BaseException as error:
                    entry.update(model_error=type(error).__name__, received_at=timestamp())
                    if not response_logged:
                        trace.request_error(type(error).__name__, at=entry["received_at"],
                            status="cancelled" if isinstance(error, asyncio.CancelledError) else "runtime_error")
                    record["attempts"].append(entry)
                    raise
                record["attempts"].append(entry)
                if action is not None:
                    break
            if phase_1_decision:
                summary['phase_1_decisions'] += 1
                summary['agent_decisions'] += 1
                record['phase_1_decision_index'] = summary['phase_1_decisions']
                if action is None:
                    temporal_feedback = 'Invalid or prohibited survey decision consumed one budget slot; no action executed.'
            if action is None:
                if consume_invalid_step:
                    if not phase_1_decision:
                        summary["agent_decisions"] += 1
                    pending_invalid_feedback = (feedback or 'Invalid decision.') + (
                        ' One decision step was consumed; no robot action was executed. Correct the next decision.')
                    record['invalid_decision_feedback'] = pending_invalid_feedback
                    summary['invalid_decisions'].append({'step': step, 'feedback': pending_invalid_feedback})
                    observation = await environment.observe()
                    record['next_observation'] = str(archive.save(observation, directory / 'after').relative_to(output))
                    trace.observation(observation, phase='after_invalid_decision')
                elif not phase_1_decision:
                    summary["termination_reason"] = "invalid_output"
            else:
                if not phase_1_decision:
                    summary["agent_decisions"] += 1
                if command is not None:
                    record["parsed_command"] = command.to_dict()
            model_requested_done = action == DONE_ATOM
            qa_stop = False
            if action is not None and qa_protocol is not None:
                qa_history.append(dict(parsed_qa, step=step))
                subtask_stop, subtask_reason = qa_protocol.should_stop(parsed_qa, qa_history, step, max_steps)
                record["qa_subtask_stop_reason"] = subtask_reason if subtask_stop else None
                qa_stop, qa_reason = qa_protocol.episode_should_stop(parsed_qa, qa_history, step, max_steps)
                record["qa_response"] = parsed_qa
                if qa_stop:
                    record["model_requested_action"] = action
                    record["qa_stop_reason"] = qa_reason
                    action = DONE_ATOM
            if action is not None and deadline is not None and time.perf_counter() >= deadline:
                summary["termination_reason"] = "timeout"
                action = None
            if action is not None:
                report = getattr(environment, 'report_arrivals', None)
                text = entry.get('model_metadata', {}).get('world_observation', '')
                answers = re.findall(r'^[ \t]*ANSWER:[ \t]*(\S[^\n]*)$', text, re.MULTILINE) if isinstance(text, str) else []
                answer_report = getattr(environment, 'report_answer', None)
                if not temporal_episode and len(answers) == 1 and callable(answer_report):
                    record['answer_declaration'] = {'answer': answers[0],
                        'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                        'receipt': await answer_report(answers[0], observation)}
                rooms = list(dict.fromkeys(re.findall(
                    r'^[ \t]*ARRIVED:[ \t]+([a-z][a-z0-9_]*_[0-9]+)[ \t]*$', text, re.MULTILINE))) if isinstance(text, str) else []
                if rooms and callable(report):
                    record['arrival_declarations'] = {'rooms': rooms,
                        'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                        'receipt': await report(rooms, observation)}
            if action == DONE_ATOM:
                temporal_continue = False
                if temporal_episode:
                    phase = getattr(environment, 'temporal_phase', None)
                    declaration = (_temporal_declaration(text, phase)
                        if record.get('model_requested_action', action) == DONE_ATOM else None)
                    record['temporal_phase'] = phase
                    if phase == 'phase_1':
                        declaration = ('phase_1_complete', None) if temporal_policy['phase_1_allow_early_transition'] else None
                    if declaration is not None:
                        event, answer = declaration
                        report = getattr(environment, 'report_temporal', None)
                        if not callable(report):
                            raise RuntimeError('Temporal episode transport is unavailable.')
                        if event == 'phase_1_complete':
                            pending_phase_observation = observation
                            phase_payload = directory / 'phase_1_final_payload.json'
                            phase_payload.write_text(json.dumps(observation.to_payload()))
                        receipt = await report(event, answer, observation)
                        if event == 'phase_1_complete':
                            summary['temporal_transitions'].append({'reason': 'agent_requested',
                                'before_snapshot_id': observation.snapshot_id, 'phase_1_decisions': summary['phase_1_decisions'],
                                'final_phase_1_payload': str(phase_payload),
                                'final_phase_1_delivered': False})
                        record['temporal_event'] = {'event': event, 'snapshot_id': observation.snapshot_id,
                            'action_step': observation.action_step, 'receipt': receipt}
                        summary['temporal_phase'] = getattr(environment, 'temporal_phase', None)
                        temporal_feedback = None
                        if event == 'phase_1_complete':
                            temporal_continue = True
                            observation_started = time.perf_counter()
                            observation = await environment.observe()
                            timings['observation_rpc_wall_time'] += time.perf_counter()-observation_started
                            trace.set_stamp(step, attempt + 1, observation)
                            trace.observation(observation, phase='phase_2')
                            summary['temporal_transitions'][-1]['after_snapshot_id'] = observation.snapshot_id
                            preparation_started = time.perf_counter()
                            record['next_observation'] = str(archive.save(observation, directory / 'after').relative_to(output))
                            timings['observation_serialization_time'] += time.perf_counter()-preparation_started
                            if observation.snapshot_id <= before_snapshot or observation.action_step != before_step:
                                summary['termination_reason'] = 'observation_error'
                            record['temporal_continue_no_motion'] = True
                    else:
                        temporal_continue = True
                        temporal_feedback = ('DONE did not include exactly one valid declaration for the current phase. '
                            'Continue inspecting, or pair DONE with standalone PHASE1_COMPLETE in phase_1 '
                            'or standalone ANSWER: <public option> in phase_2. No phase event or motion occurred.')
                        record['temporal_continue_no_motion'] = True
                        record['temporal_declaration_error'] = 'missing_ambiguous_or_wrong_phase_marker'
                if not temporal_continue and (qa_protocol is None or qa_stop):
                    summary.update(termination_reason="model_done", model_declared_done=
                        record.get("model_requested_action", action) == DONE_ATOM)
                    if qa_protocol is not None:
                        summary["qa_stop_reason"] = qa_reason
                    declare_done = getattr(environment, 'report_done', None)
                    if (model_requested_done and (qa_protocol is None or qa_reason != 'max_steps')
                            and callable(declare_done)):
                        try:
                            declared_before = observation
                            receipt = await declare_done(observation)
                            record['done_declaration'] = {'snapshot_id': observation.snapshot_id,
                                'action_step': observation.action_step, 'receipt': receipt}
                            summary['done_declaration'] = receipt
                            (directory / 'done_declaration.json').write_text(
                                json.dumps(record['done_declaration'], indent=2) + '\n')
                            if receipt.get('snapshot_id', observation.snapshot_id) != observation.snapshot_id:
                                observation_started = time.perf_counter()
                                observation = await environment.observe()
                                timings['observation_rpc_wall_time'] += time.perf_counter()-observation_started
                                if (observation.snapshot_id != receipt['snapshot_id']
                                        or observation.snapshot_id <= declared_before.snapshot_id
                                        or observation.action_step != declared_before.action_step):
                                    raise RuntimeError('DONE observation does not match its terminal boundary receipt.')
                                trace.set_stamp(step, attempt + 1, observation)
                                trace.observation(observation, phase='after_done_declaration')
                                record['next_observation'] = str(archive.save(observation, directory / 'after').relative_to(output))
                        except Exception as error:
                            record['done_declaration_error'] = f'{type(error).__name__}: {error}'
                            summary['done_declaration_error'] = record['done_declaration_error']
                            summary['task_success'] = None
                            raise RuntimeError('DONE declaration failed; terminal evaluation is unavailable.') from error
                elif not temporal_continue:
                    # The upstream stop action with an unsure answer does not
                    # stop exploration. It causes no robot motion.
                    record["qa_continue_no_motion"] = True
                    observation = await environment.observe()
                record["parsed_action"] = action
                if success_provider is not None:
                    summary["task_success"] = success_provider()
            elif action is not None:
                if integrity_check is not None:
                    integrity_check()
                record["parsed_action"] = action
                record["execution_started_at"] = timestamp()
                summary["macro_actions"] += 1
                execution_started = time.perf_counter()
                result = await environment.execute(command if command is not None else action)
                timings["action_execution_wall_time"] += time.perf_counter()-execution_started
                if action_mode == "parameterized":
                    macro = result.get("macro")
                    if macro is not None:
                        execution = macro["execution"]
                        actual_fixed = execution.get("actual_fixed_primitive_executions", execution["executed_count"])
                        summary["primitive_actions"] += actual_fixed
                        summary["actual_fixed_primitive_executions"] += actual_fixed
                        summary["equivalent_fixed_primitives"] += execution.get("equivalent_fixed_primitives", macro["compilation"]["planned_count"])
                        summary["metric_target_executions"] += execution.get("metric_target_executions", 0)
                        if macro["physics_steps"] is not None:
                            summary["physics_steps"] += macro["physics_steps"]
                            summary["controller_env_steps"] += macro["physics_steps"]
                        else:
                            summary["physics_steps_known"] = False
                    elif result.get("outcome_known") is not True:
                        summary["physics_steps_known"] = False
                        summary["primitive_actions_known"] = False
                else:
                    summary["primitive_actions"] += len(result.get("results", []))
                    summary["actual_fixed_primitive_executions"] += len(result.get("results", []))
                    summary["equivalent_fixed_primitives"] += len(result.get("results", []))
                    # Fixed public feedback never included low-level step counts.
                    summary["physics_steps_known"] = False
                record.update(action_result=result, execution_finished_at=timestamp())
                trace.execution(result, at=record["execution_finished_at"])
                summary["executed_actions"].append(action)
                observation_started = time.perf_counter()
                observation = await environment.observe()
                timings["observation_rpc_wall_time"] += time.perf_counter()-observation_started
                trace.set_stamp(step, attempt + 1, observation)
                trace.observation(observation, phase="after_execution")
                observation.set_serialization_cache(getattr(agent, "snapshot_cache", True))
                preparation_started = time.perf_counter()
                record["next_observation"] = str(archive.save(observation, directory / "after").relative_to(output))
                timings["observation_serialization_time"] += time.perf_counter()-preparation_started
                if (observation.snapshot_id <= before_snapshot or observation.action_step != before_step + 1):
                    summary["termination_reason"] = "observation_error"
                elif result.get("ok") is not True:
                    if safely_recoverable(result, action) and (max_recoverable_failures is None
                            or consecutive_failures < max_recoverable_failures):
                        consecutive_failures += 1
                        recovery = {"last_action": action, "reason": result["reason"],
                            "outcome_known": True, "motion_stopped": True, "episode_continues": True,
                            "consecutive_failures": consecutive_failures,
                            "instruction": "The previous command did not complete and its motion has been confirmed stopped. "
                            "Remaining primitives were not executed. Continue from the NEW measured pose and images. "
                            "Choose a smaller motion, retreat, or reposition the base/trunk. "
                            "Do not assume the requested motion completed; a new command is your decision, not an automatic replay."}
                        record["recoverable_execution_error"] = result["reason"]
                        summary["recovered_execution_errors"].append({"step": step, "action": action,
                                                                      "reason": result["reason"]})
                    else:
                        summary["termination_reason"] = "execution_error"
                else:
                    recovery, consecutive_failures = None, 0
                if summary["termination_reason"] == "max_steps" and success_provider is not None:
                    # Evaluator lives outside the model input and the model tools.
                    summary["task_success"] = success_provider()
                    if summary["task_success"] is True:
                        summary["termination_reason"] = "environment_success"
            if (phase_1_decision and getattr(environment, 'temporal_phase', None) == 'phase_1'
                    and summary['phase_1_decisions'] >= temporal_policy['phase_1_max_decisions']
                    and summary['termination_reason'] == 'max_steps'):
                if temporal_policy['phase_1_auto_transition']:
                    pending_phase_observation = observation
                    # Save real post-action Phase-1 images before the owner-thread event.
                    final_path = archive.save(observation, directory / 'phase_1_final')
                    phase_payload = directory / 'phase_1_final_payload.json'
                    phase_payload.write_text(json.dumps(observation.to_payload()))
                    receipt = await environment.report_temporal('phase_1_budget_exhausted', None, observation)
                    transition = {'reason': 'budget_exhausted', 'before_snapshot_id': observation.snapshot_id,
                        'phase_1_decisions': summary['phase_1_decisions'], 'final_phase_1_delivered': False,
                        'final_phase_1_observation': str(final_path), 'final_phase_1_payload': str(phase_payload), 'receipt': receipt}
                    summary['temporal_transitions'].append(transition)
                    observation = await environment.observe()
                    transition['after_snapshot_id'] = observation.snapshot_id
                    summary['temporal_phase'] = getattr(environment, 'temporal_phase', None)
                    trace.observation(observation, phase='phase_2')
                    record['temporal_transition'] = transition
                    if success_provider is not None:
                        summary['task_success'] = success_provider()
                else:
                    summary['termination_reason'] = 'phase_1_budget_exhausted'
            stopped = summary["termination_reason"] != "max_steps" or step == max_steps
            record.update(finished_at=timestamp(), termination_reason=summary["termination_reason"] if stopped else None)
            with (output / "steps.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            print(f"step={step} action={action or 'INVALID'}" + (f" termination={summary['termination_reason']}" if stopped else ""), flush=True)
            record = None
            if stopped:
                break
        if qa_protocol is not None:
            answer, answer_step = qa_protocol.resolve(qa_history)
            summary["qa_final_answer"] = answer
            summary["qa_answer_step"] = answer_step
            if (summary["termination_reason"] in ("model_done", "max_steps")
                    and qa_protocol.needs_force_choice(answer, summary.get("qa_stop_reason", "max_steps"))):
                forced = DecisionInput(observation, phase_actions(), action_context=public_action_context(),
                    steps_remaining=0, action_mode=action_mode, observation_condition=observation_condition,
                    sensor_observation=initial_observation if observation_condition == "initial_view_only" else None)
                forced_step = summary["agent_decisions"] + 1
                from dataclasses import replace
                forced = agent.prepare_force_choice(replace(forced, decision_step=forced_step))
                trace.set_stamp(forced_step, 1, observation)
                trace.step_input(forced, summary["agent"], projection, at=timestamp())
                summary["model_calls"] += 1
                wait_started = time.perf_counter()
                if deadline is None:
                    reply = await agent.decide(forced)
                else:
                    reply = await asyncio.wait_for(agent.decide(forced), max(.001, deadline-time.perf_counter()))
                summary["qa_forced_choice_wall_time_s"] = time.perf_counter()-wait_started
                for key, value in reply.metadata.get("timings", {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        timings[key] = timings.get(key, 0.) + value
                metadata = model_metadata(reply.metadata)
                trace.response(final_text_only(reply.raw_text), metadata, at=timestamp())
                parsed = qa_protocol.parse(metadata.get("qa_response"), DONE_ATOM)
                if parsed["parsed_container"] is None:
                    raise ModelError("The upstream forced final selection must name a container.")
                summary["qa_forced_choice"] = parsed
                summary["qa_final_answer"] = parsed["answer"]
                summary["qa_answer_step"] = forced_step
                (output / "qa_forced_choice.json").write_text(json.dumps(parsed, indent=2) + "\n")
    except ModelError as error:
        summary.update(termination_reason="model_error", error=str(error))
    except asyncio.TimeoutError:
        summary["termination_reason"] = "timeout"
    except asyncio.CancelledError:
        summary["termination_reason"] = "cancelled"
        raise
    except Exception as error:
        summary.update(termination_reason="runtime_error", error=str(error))
    finally:
        if restore_candidate_history is not None:
            restore_candidate_history()
        if hasattr(agent, "close"):
            try:
                await agent.close()
            except Exception as error:
                summary["session_cleanup_error"] = type(error).__name__
                summary["termination_reason"] = "runtime_error"
            if hasattr(agent, "provenance"):
                summary["native_session"] = agent.provenance()
        if record is not None:
            record.update(finished_at=timestamp(), termination_reason=summary["termination_reason"])
            with (output / "steps.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        summary.update(finished_at=timestamp(), num_executed_actions=len(summary["executed_actions"]))
        if episode_manifest is not None:
            summary["private_task_success"] = summary["task_success"]
        segment_wall_s = time.perf_counter()-episode_started
        if resume_summary is not None:
            summary["segment_wall_time_s"] = segment_wall_s
            summary["previous_active_wall_time_s"] = previous_wall_s
        timings["total_episode_wall_time"] = previous_wall_s + segment_wall_s
        summary["timings"] = timings
        if hasattr(environment, "world_metrics"):
            try:
                summary["world_timing"] = await environment.world_metrics()
            except Exception as error:
                summary["world_timing_error"] = type(error).__name__
        summary["controller_env_steps_known"] = summary["physics_steps_known"]
        if not summary["controller_env_steps_known"]:
            summary["controller_env_steps"] = None
        if not trace.headers_written:
            trace.headers(None, summary["agent"], at=summary["started_at"],
                          task_text=getattr(environment, "task_text", None), global_status="no_observation")
            if hasattr(environment, "attach_input_logger"):
                environment.attach_input_logger(trace.mcp_response)
        trace.finish(summary)
        if hasattr(agent, "attach_output_logger"):
            agent.attach_output_logger(None)
        if hasattr(agent, "attach_input_logger"):
            agent.attach_input_logger(None)
        if hasattr(environment, "attach_input_logger"):
            environment.attach_input_logger(None)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

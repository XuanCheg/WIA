"""Project Phase 2 diagnostic inputs without changing the full-input runtime.

Run CPU checks: python -B -m pytest -q -p no:cacheprovider tests/test_wia_history_projection.py

This helper is a component, not a complete history ablation. Before experimental
use, the caller MUST create a genuinely fresh native conversation and use a
diagnostic prompt assembler. The current assembler incorrectly advertises the
overview as required for this intervention. It must instead explain the removal
reason recorded here. This helper neither resets/verifies native model context
nor changes evaluator state. It must not be wired into existing transport by
assuming that clearing DecisionInput fields clears a resident conversation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
from typing import Literal

from core.action_compiler import MacroLimits, PrimitiveResolution
from core.agent.decision import DecisionInput
from core.feedback_semantics import FEEDBACK_SEMANTICS
from core.observations import ActionHistoryEntry, AgentActionFeedback, AgentObservation
from core.robot_cameras import ROBOT_CAMERA_LABELS
from core.sensor_observations import SENSOR_SPECS


NEUTRAL_INSTRUCTION = 'Compare the current scene with any historical facts explicitly supplied in this condition.'
PHYSICAL_CONTEXT_FIELDS = frozenset((
    'frame', 'working_arm', 'steps', 'rotation_frame', 'trunk_height_range_m',
    'trunk_height_range_reference', 'feedback_semantics', 'parameterized_actions',
))
FRAME_FIELDS = frozenset(('forward', 'left', 'up', 'positive_yaw', 'translation_reference'))
ROTATION_FRAME_FIELDS = frozenset(('reference', 'axis', 'positive_direction', 'position_hold'))
BOUND_FIELDS = frozenset(('magnitude_field', 'primitive_step', 'minimum', 'maximum',
                          'max_primitives', 'rounding', 'max_duration_s'))


def _physical_context(context, available_actions):
    """Keep the physical fields emitted by MCPRobotEnvironment/public_context.

    Nested allowlists prevent old temporal/recovery metadata from hitchhiking
    inside an otherwise physical configuration. No limits or increments change.
    """
    removed = []

    def select(value, allowed, path):
        if not isinstance(value, Mapping):
            raise ValueError(f'Physical context {path} must be a mapping.')
        removed.extend(path + '.' + key for key in value if key not in allowed)
        return {key: deepcopy(item) for key, item in value.items() if key in allowed}

    copied = select(context, PHYSICAL_CONTEXT_FIELDS, 'action_context')
    for key, allowed in (('frame', FRAME_FIELDS), ('rotation_frame', ROTATION_FRAME_FIELDS)):
        if isinstance(copied.get(key), Mapping):
            copied[key] = select(copied[key], allowed, 'action_context.' + key)
    if isinstance(copied.get('steps'), Mapping):
        copied['steps'] = select(copied['steps'], {field.name for field in fields(PrimitiveResolution)}, 'action_context.steps')
    if isinstance(copied.get('feedback_semantics'), Mapping):
        copied['feedback_semantics'] = select(copied['feedback_semantics'], set(FEEDBACK_SEMANTICS),
                                               'action_context.feedback_semantics')
        for key, text in list(copied['feedback_semantics'].items()):
            if text != FEEDBACK_SEMANTICS[key]:
                removed.append('action_context.feedback_semantics.' + key)
                del copied['feedback_semantics'][key]
    if 'parameterized_actions' in copied:
        setting = select(copied['parameterized_actions'],
                         {'resolution', 'limits', 'bounds_by_action', 'request_timeout_s'},
                         'action_context.parameterized_actions')
        for key, model in (('resolution', PrimitiveResolution), ('limits', MacroLimits)):
            if key in setting:
                setting[key] = select(setting[key], {field.name for field in fields(model)},
                                      'action_context.parameterized_actions.' + key)
                model(**setting[key])
        if 'bounds_by_action' in setting:
            setting['bounds_by_action'] = select(setting['bounds_by_action'], set(available_actions),
                                                 'action_context.parameterized_actions.bounds_by_action')
            for action, bounds in setting['bounds_by_action'].items():
                setting['bounds_by_action'][action] = select(bounds, BOUND_FIELDS,
                    'action_context.parameterized_actions.bounds_by_action.' + action)
        copied['parameterized_actions'] = setting
    return copied, sorted(removed)


def _project_observation(observation, history, feedback):
    if type(observation) is not AgentObservation:
        raise ValueError('History projection requires the plain typed AgentObservation contract.')
    if any(observation.rgb_images().get(name) is None for name in ROBOT_CAMERA_LABELS):
        raise ValueError('History projection must retain all eight current mounted RGB views.')
    return replace(observation, initial_overview=None, initial_overview_status='unavailable',
                   action_history=history, last_action_feedback=feedback, public_task_events=())


def _bundle_audit(before, after):
    images = {}
    for name in ROBOT_CAMERA_LABELS:
        original, projected = before.rgb_images()[name], after.rgb_images()[name]
        images[name] = {'sha256': hashlib.sha256(original.tobytes()).hexdigest(),
                        'bytes_preserved': original.dtype == projected.dtype
                        and original.shape == projected.shape and original.tobytes() == projected.tobytes()}
    sensors = {}
    for name in SENSOR_SPECS:
        original, projected = getattr(before.sensors, name), getattr(after.sensors, name)
        sensors[name] = {'available': original is not None,
                         'bytes_preserved': projected is None if original is None else
                         projected is not None and original.dtype == projected.dtype and
                         original.shape == projected.shape and original.tobytes() == projected.tobytes()}
    return {
        'overview_removal_reason': 'explicit_diagnostic_history_intervention',
        'native_capture_failure': False, 'initial_overview_removed': before.initial_overview is not None,
        'original_history_entries_removed': len(before.action_history),
        'phase2_history_entries_supplied': len(after.action_history),
        'original_feedback_removed': before.last_action_feedback is not None,
        'original_public_task_events_removed': len(before.public_task_events),
        'current_images': images, 'sensor_channels': sensors,
        'camera_calibration_preserved': dict(before.camera_geometry) == dict(after.camera_geometry),
        'proprioception_preserved': before.robot_state is after.robot_state,
        'snapshot_id': after.snapshot_id, 'action_step': after.action_step,
        'capture_boundary_preserved': all(getattr(before, key) == getattr(after, key)
            for key in ('snapshot_id', 'action_step', 'captured_at', 'action_boundary')),
    }


def project_phase2_input(request: DecisionInput, *,
                         condition: Literal['history_removed', 'history_oracle'],
                         phase2_action_history: tuple = (),
                         phase2_last_action_feedback: AgentActionFeedback | None = None,
                         oracle_facts: str | None = None) -> tuple[DecisionInput, dict]:
    """Copy current sensors while removing explicitly enumerated history channels.

    The caller supplies sanitized Phase-2-only entries/feedback. Entries have no
    timestamps, so this function neither infers nor verifies a phase cutoff.
    Oracle text is copied verbatim and is not checked for factual accuracy.
    A genuinely fresh native conversation and diagnostic prompt assembler are
    REQUIRED before experimental use; projection alone verifies no ablation.
    """
    if not isinstance(request, DecisionInput):
        raise ValueError('History projection requires a DecisionInput.')
    if not isinstance(request.action_context, Mapping) or request.action_context.get('temporal_phase') != 'phase_2':
        raise ValueError('History projection requires explicit temporal_phase phase_2.')
    if request.observation_condition != 'current':
        raise ValueError('History projection requires observation_condition current.')
    if condition not in ('history_removed', 'history_oracle'):
        raise ValueError('Unknown diagnostic history condition.')
    if condition == 'history_oracle':
        if not isinstance(oracle_facts, str) or not oracle_facts.strip():
            raise ValueError('history_oracle requires nonempty caller-supplied factual text.')
    elif oracle_facts is not None:
        raise ValueError('history_removed forbids oracle facts.')
    if type(phase2_action_history) is not tuple or any(type(entry) is not ActionHistoryEntry for entry in phase2_action_history):
        raise ValueError('Phase-2 history must be a tuple of sanitized ActionHistoryEntry values.')
    if phase2_last_action_feedback is not None and type(phase2_last_action_feedback) is not AgentActionFeedback:
        raise ValueError('Phase-2 feedback must be a sanitized AgentActionFeedback or None.')
    observation = _project_observation(request.observation, phase2_action_history, phase2_last_action_feedback)
    sensor_observation = (_project_observation(request.sensor_observation, phase2_action_history, phase2_last_action_feedback)
                          if request.sensor_observation is not None else None)
    context, removed_context = _physical_context(request.action_context, request.available_actions)
    intervention = {'schema': 'wia_phase2_history_projection_v1',
                    'overview_removal_reason': 'explicit_diagnostic_history_intervention',
                    'native_capture_failure': False,
                    'fresh_native_conversation_required': True, 'diagnostic_prompt_assembler_required': True}
    context.update(temporal_phase='phase_2', temporal_instruction=NEUTRAL_INSTRUCTION,
                   wia_history_intervention=deepcopy(intervention))
    if condition == 'history_oracle':
        context['diagnostic_historical_facts'] = oracle_facts
    projected = replace(request, observation=observation, sensor_observation=sensor_observation,
                        previous_phase_observation=None, public_memory=(), memory_mode='none',
                        assembled_prompt=None, resident_prompt=None, prompt_phase=None,
                        feedback=None, action_context=context)
    bundles = {'observation': _bundle_audit(request.observation, observation)}
    if sensor_observation is not None:
        bundles['sensor_observation'] = _bundle_audit(request.sensor_observation, sensor_observation)
    removed_channels = ['request.previous_phase_observation', 'request.public_memory', 'request.memory_mode',
                        'request.feedback', 'request.assembled_prompt', 'request.resident_prompt', 'request.prompt_phase']
    for name in bundles:
        removed_channels.extend(name + '.' + field for field in
            ('initial_overview', 'action_history', 'last_action_feedback', 'public_task_events'))
    audit = {**intervention, 'condition': condition, 'removed_channels': removed_channels,
             'removed_action_context_paths': removed_context, 'observation_bundles': bundles,
             'current_images_preserved': all(row['bytes_preserved'] for bundle in bundles.values()
                                              for row in bundle['current_images'].values()),
             'caller_supplied_phase2_history_independently_verified': False,
             'oracle_text_sha256': hashlib.sha256(oracle_facts.encode('utf-8')).hexdigest()
                 if condition == 'history_oracle' else None,
             'oracle_factual_accuracy_verified': False,
             'native_context_reset_required': True, 'native_context_reset_verified': False,
             'diagnostic_prompt_assembler_verified': False, 'no_history_condition_verified': False,
             'evaluator_state_modified': False}
    return projected, audit

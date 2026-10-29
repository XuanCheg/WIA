"""Check pure Phase-2 history projection without model, simulator or session calls.

Run: PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_history_projection.py
These checks do not verify a complete no-history experiment.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from core.action_compiler import MacroLimits, PrimitiveResolution, action_bounds
from core.agent.decision import DecisionInput
from wia.runtime.agent.wia_history_projection import NEUTRAL_INSTRUCTION, project_phase2_input
from core.camera_geometry import CameraGeometry
from core.feedback_semantics import FEEDBACK_SEMANTICS
from core.observations import (ActionHistoryEntry, AgentActionFeedback, AgentObservation,
                               AgentRobotState, InitialOverview, PublicTaskEvent)
from core.robot_cameras import ADDITIONAL_RGB_NAMES, ROBOT_CAMERA_LABELS
from core.sensor_observations import SENSOR_SPECS, SensorReadings


STAMP = '2026-10-23T12:00:00+00:00'


def observation(snapshot=12, action_step=8):
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    depth = np.arange(12, dtype=np.float32).reshape(3, 4)
    depth[0, :3] = [np.nan, np.inf, 0.]
    calibration = CameraGeometry('wrist_rgb', (4, 3), ((2., 0., 2.), (0., 2., 1.), (0., 0., 1.)),
                                 tuple(map(tuple, np.eye(4))))
    overview = InitialOverview(np.full((4, 5, 3), 255, np.uint8), 'episode', 'reset', 0, 1, STAMP,
                                (-2., -2., 2., 2.), (1., 0.), (0., 1.), (0., 0., 0.), 0.)
    return AgentObservation('Compare the two scenes and act.', rgb, rgb + 1,
        (ActionHistoryEntry('BASE_LEFT', 'completed', distance_m=.2),), snapshot, STAMP, action_step, True,
        robot_state=AgentRobotState(gripper_width_m=.03, sources={'gripper_width_m': 'finger_joint_encoders'}),
        last_action_feedback=AgentActionFeedback('BASE_LEFT', 'completed', True, {}, {}),
        sensors=SensorReadings(head_depth_m=depth, wrist_depth_m=depth + .25,
                               lidar_points_m=np.arange(6, dtype=np.float32).reshape(2, 3),
                               force_torque=np.arange(6, dtype=np.float32), tactile=np.ones((2, 2), np.float32)),
        additional_rgb={name: rgb + index + 2 for index, name in enumerate(ADDITIONAL_RGB_NAMES)},
        camera_geometry={'wrist_rgb': calibration}, initial_overview=overview, initial_overview_status='available',
        public_task_events=(PublicTaskEvent('old-query', 'history_followup', 'OLD_EVENT_CANARY', 1, 2),))


def request():
    actions = ('BASE_FWD', 'MV_DOWN', 'GRASP', 'RELEASE', 'DONE')
    resolution = PrimitiveResolution(base_translation_m=.05, base_rotation_deg=15., arm_translation_m=.02)
    limits = MacroLimits()
    context = {'frame': {'forward': '+X', 'left': '+Y', 'up': '+Z',
                         'positive_yaw': 'counterclockwise viewed from above',
                         'translation_reference': 'current robot base frame at macro start'},
               'working_arm': 'left', 'steps': {'base_translation_m': .05, 'arm_translation_m': .02},
               'rotation_frame': {'reference': 'current end-effector frame at macro start',
                                  'axis': '+Z through the end-effector origin',
                                  'positive_direction': 'clockwise looking along end-effector +Z toward the fingers',
                                  'position_hold': 'robot_base'},
               'trunk_height_range_m': [.3, .9],
               'trunk_height_range_reference': 'trunk_control_height_m: nominal_urdf_workspace',
               'feedback_semantics': deepcopy(FEEDBACK_SEMANTICS),
               'parameterized_actions': {'resolution': resolution.to_dict(), 'limits': limits.to_dict(),
                                        'bounds_by_action': action_bounds(actions, resolution, limits),
                                        'request_timeout_s': limits.request_timeout_s},
               'temporal_phase': 'phase_2', 'temporal_feedback': 'OLD_TEMPORAL_CANARY',
               'recovery': {'instructions': 'OLD_RECOVERY_CANARY'},
               'interruption_resume': 'OLD_RESUME_CANARY', 'temporal_survey': {'past': 'OLD_SURVEY_CANARY'},
               'temporal_instruction': 'OLD_INSTRUCTION_CANARY', 'notes': 'OLD_NOTES_CANARY'}
    return DecisionInput(observation(), actions, feedback='OLD_REJECTION_CANARY', action_context=context,
        steps_remaining=5, action_mode='parameterized', memory_mode='model_notes',
        public_memory=({'note': 'OLD_MEMORY_CANARY'},), sensor_observation=observation(13, 9),
        previous_phase_observation=observation(3, 2), assembled_prompt='OLD_PROMPT_CANARY',
        resident_prompt='OLD_RESIDENT_CANARY', prompt_phase='OLD_PROMPT_PHASE_CANARY', decision_step=9)


def test_projection_strips_both_bundles_and_request_history_without_mutating_input():
    original = request()
    original_context = deepcopy(original.action_context)
    original_metadata = original.observation.metadata()
    projected, audit = project_phase2_input(original, condition='history_removed')
    assert projected is not original
    assert projected.previous_phase_observation is None
    assert projected.public_memory == () and projected.memory_mode == 'none'
    assert projected.assembled_prompt is projected.resident_prompt is projected.prompt_phase is projected.feedback is None
    for name in ('observation', 'sensor_observation'):
        before, after = getattr(original, name), getattr(projected, name)
        assert type(after) is AgentObservation and after is not before
        assert after.initial_overview is None and after.initial_overview_status == 'unavailable'
        assert after.action_history == () and after.last_action_feedback is None and after.public_task_events == ()
        assert (after.snapshot_id, after.action_step, after.captured_at) == (before.snapshot_id, before.action_step, before.captured_at)
        assert set(after.rgb_images()) == set(ROBOT_CAMERA_LABELS)
        for view in ROBOT_CAMERA_LABELS:
            assert after.rgb_images()[view].tobytes() == before.rgb_images()[view].tobytes()
        for sensor in SENSOR_SPECS:
            assert getattr(after.sensors, sensor).tobytes() == getattr(before.sensors, sensor).tobytes()
        assert after.robot_state is before.robot_state
        assert dict(after.camera_geometry) == dict(before.camera_geometry)
    assert original.action_context == original_context
    assert original.observation.metadata() == original_metadata
    assert original.previous_phase_observation is not None
    assert original.observation.initial_overview is not None
    assert 'OLD_' not in json.dumps(projected.action_context)
    assert projected.action_context['temporal_instruction'] == NEUTRAL_INSTRUCTION
    assert projected.action_context['wia_history_intervention']['native_capture_failure'] is False
    assert audit['current_images_preserved'] is True
    assert audit['native_context_reset_required'] is True
    assert audit['native_context_reset_verified'] is audit['no_history_condition_verified'] is False
    assert audit['evaluator_state_modified'] is False
    assert audit['oracle_text_sha256'] is None
    assert 'sensor_observation.initial_overview' in audit['removed_channels']
    assert projected.available_actions == original.available_actions
    assert projected.steps_remaining == original.steps_remaining
    assert projected.decision_step == original.decision_step


def test_explicit_phase2_only_history_and_feedback_replace_all_old_entries():
    original = request()
    phase2_history = (ActionHistoryEntry('BASE_FWD', 'completed', distance_m=.1),)
    feedback = AgentActionFeedback('BASE_FWD', 'completed', True, {}, {})
    projected, audit = project_phase2_input(original, condition='history_removed',
        phase2_action_history=phase2_history, phase2_last_action_feedback=feedback)
    for bundle in (projected.observation, projected.sensor_observation):
        assert bundle.action_history == phase2_history
        assert bundle.last_action_feedback is feedback
    assert audit['caller_supplied_phase2_history_independently_verified'] is False
    assert original.observation.action_history != phase2_history


def test_physical_configuration_is_preserved_and_nested_nonphysical_metadata_is_removed():
    original = request()
    expected = deepcopy(original.action_context['parameterized_actions'])
    original.action_context['parameterized_actions']['recovery'] = 'OLD_NESTED_CANARY'
    projected, audit = project_phase2_input(original, condition='history_removed')
    assert projected.action_context['parameterized_actions'] == expected
    assert projected.schema == original.schema
    for key in ('frame', 'working_arm', 'steps', 'rotation_frame', 'trunk_height_range_m',
                'trunk_height_range_reference', 'feedback_semantics'):
        assert projected.action_context[key] == original.action_context[key]
    projected.action_context['parameterized_actions']['limits']['max_translation_m'] = .7
    assert original.action_context['parameterized_actions']['limits']['max_translation_m'] == 1.2
    assert 'action_context.parameterized_actions.recovery' in audit['removed_action_context_paths']


def test_oracle_text_is_verbatim_separate_from_removed_condition_and_not_accuracy_certified():
    original = request()
    facts = '  Earlier, the red jar was in the left container.\nNo other facts supplied.  '
    oracle, audit = project_phase2_input(original, condition='history_oracle', oracle_facts=facts)
    removed, removed_audit = project_phase2_input(original, condition='history_removed')
    assert oracle.action_context['diagnostic_historical_facts'] == facts
    assert 'diagnostic_historical_facts' not in removed.action_context
    assert audit['oracle_text_sha256'] == hashlib.sha256(facts.encode()).hexdigest()
    assert audit['oracle_factual_accuracy_verified'] is False
    assert removed_audit['oracle_text_sha256'] is None
    assert oracle.observation.action_history == removed.observation.action_history == ()
    assert original.public_memory == ({'note': 'OLD_MEMORY_CANARY'},)


@pytest.mark.parametrize('change', ['phase_1', 'no_phase', 'initial_view_only'])
def test_phase_and_observation_condition_guards(change):
    original = request()
    if change == 'initial_view_only':
        original = replace(original, observation_condition='initial_view_only')
    else:
        original.action_context['temporal_phase'] = 'phase_1' if change == 'phase_1' else None
    with pytest.raises(ValueError, match='requires'):
        project_phase2_input(original, condition='history_removed')


@pytest.mark.parametrize('condition,facts', [('history_oracle', None), ('history_oracle', '  '),
                                          ('history_removed', 'Some fact.'), ('unknown', None)])
def test_oracle_conditions_cannot_silently_mix(condition, facts):
    with pytest.raises(ValueError):
        project_phase2_input(request(), condition=condition, oracle_facts=facts)


def test_missing_mounted_views_or_untyped_phase2_history_are_rejected():
    original = request()
    with pytest.raises(ValueError, match='all eight'):
        project_phase2_input(replace(original, observation=replace(original.observation, additional_rgb={})),
                             condition='history_removed')
    with pytest.raises(ValueError, match='tuple'):
        project_phase2_input(original, condition='history_removed', phase2_action_history=[ActionHistoryEntry('GRASP', 'completed')])
    with pytest.raises(ValueError, match='sanitized'):
        project_phase2_input(original, condition='history_removed', phase2_last_action_feedback={'old': 'feedback'})
    projected, audit = project_phase2_input(replace(original, sensor_observation=None), condition='history_removed')
    assert projected.sensor_observation is None
    assert set(audit['observation_bundles']) == {'observation'}

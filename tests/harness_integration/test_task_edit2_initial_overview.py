"""Check frozen episode context and immutable-mission follow-up transport.

Run: python -B -m pytest -q tests/test_task_edit2_initial_overview.py
CPU fixtures verify input contracts, not native image usefulness or visibility.
"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.observations import AgentObservation, InitialOverview
from core.robot_cameras import ADDITIONAL_RGB_NAMES
from harness_integration.test_agent_observation import source_and_rig


def overview_source():
    source, rig = source_and_rig()
    calls, pixels = install_overview(rig)
    return source, rig, calls, pixels


def install_overview(rig):
    """Give an existing CPU camera rig the complete frozen-overview contract."""
    observe = rig.observe
    def all_views(**kwargs):
        return {**observe(**kwargs), **{name: np.full_like(rig.head, index + 1)
            for index, name in enumerate(ADDITIONAL_RGB_NAMES)}}
    rig.observe = all_views
    rig.initial_overview_identity = {'episode_id': 'episode-A', 'reset_id': 'reset-A', 'seed': 7}
    calls = []
    pixels = np.full((20, 24, 3), 93, dtype=np.uint8)
    def capture(**boundary):
        calls.append(boundary)
        return InitialOverview(rgb=pixels, **rig.initial_overview_identity, **boundary,
            extent_xy_m=(-2., -1., 2., 3.), image_right_world_xy=(1., 0.), image_up_world_xy=(0., 1.),
            robot_start_position_m=(0., 0., .004), robot_start_yaw_rad=0.)
    rig.initial_overview_capture = capture
    return calls, pixels


def test_one_map_capture_frozen_across_later_observations_and_lossless_transport():
    source, rig, calls, pixels = overview_source()
    first = source.begin_episode('Inspect the original scene.')
    first.require_initial_overview(rig.initial_overview_identity)
    pixels.fill(0)
    later = source.capture(action_boundary=True).observation
    assert len(calls) == 1
    assert later.initial_overview is first.initial_overview
    assert np.all(later.initial_overview.rgb == 93)
    assert later.initial_overview.snapshot_id < later.snapshot_id
    restored = AgentObservation.from_payload(later.to_payload())
    assert restored.metadata() == later.metadata()
    assert np.array_equal(restored.initial_overview.rgb, later.initial_overview.rgb)


def test_changed_reset_invalidates_frozen_map_without_silent_recapture():
    source, rig, calls, _ = overview_source()
    source.begin_episode('Inspect the original scene.')
    rig.initial_overview_identity = {**rig.initial_overview_identity, 'reset_id': 'reset-B'}
    observation = source.capture(action_boundary=True).observation
    assert observation.initial_overview_status == 'unavailable'
    assert len(calls) == 1
    with pytest.raises(ValueError):
        observation.require_initial_overview(rig.initial_overview_identity)


def test_overview_does_not_replace_missing_mounted_views():
    source, rig, _, _ = overview_source()
    observation = source.begin_episode('Inspect the original scene.')
    incomplete = replace(observation, additional_rgb={})
    with pytest.raises(ValueError):
        incomplete.require_initial_overview(rig.initial_overview_identity)


def test_missing_provider_remains_unavailable():
    source, _ = source_and_rig()
    observation = source.begin_episode('Inspect the original scene.')
    assert observation.initial_overview_status == 'unavailable'
    assert observation.initial_overview is None


def test_native_evaluator_initial_baseline_sees_complete_agent_input_bundle():
    source, rig, calls, _ = overview_source()
    baseline = []
    def begin(observation):
        observation.require_initial_overview(rig.initial_overview_identity)
        assert source.latest_snapshot.observation is observation
        baseline.append(observation)
    rig.task_evaluator = SimpleNamespace(begin_episode=begin)
    returned = source.begin_episode('Inspect the initial ambiguity.')
    assert baseline == [returned]
    assert len(calls) == 1


def test_mapping_identity_freezes_complete_initial_bundle_before_evaluator():
    source, rig, _, _ = overview_source()
    contract = {'revision': 'fixture'}
    rig.loaded_task = SimpleNamespace(definition=SimpleNamespace(private=SimpleNamespace(
        evaluation=SimpleNamespace(parameters={'mapping_joint_world_contract': contract}))))
    order = []
    def identity(value):
        assert value is contract
        source.initial_observation.require_initial_overview(rig.initial_overview_identity)
        assert source.initial_observation is source.latest_snapshot.observation
        assert source.action_step == 0
        order.append('identity')
        return {'reset_id': 'reset-A', 'measurements': {'fixture': True}}
    rig.mapping_joint_world_identity = identity
    def begin(observation):
        assert rig.mapping_joint_world_initial_identity['reset_id'] == 'reset-A'
        order.append('evaluator')
    rig.task_evaluator = SimpleNamespace(begin_episode=begin)
    source.begin_episode('Inspect the queried rooms.')
    source.capture(action_boundary=True)
    assert order == ['identity', 'evaluator']
    assert rig.mapping_joint_world_initial_identity_status == {'status': 'measured'}


def test_failed_mapping_identity_does_not_qualify_or_destroy_initial_images():
    source, rig, _, _ = overview_source()
    rig.loaded_task = SimpleNamespace(definition=SimpleNamespace(private=SimpleNamespace(
        evaluation=SimpleNamespace(parameters={'mapping_joint_world_contract': {}}))))
    def unavailable(contract):
        raise ValueError('native source binding absent')
    rig.mapping_joint_world_identity = unavailable
    observation = source.begin_episode('Inspect the queried rooms.')
    observation.require_initial_overview(rig.initial_overview_identity)
    assert rig.mapping_joint_world_initial_identity is None
    assert rig.mapping_joint_world_initial_identity_status['status'] == 'unavailable'
    assert 'native source binding absent' in rig.mapping_joint_world_initial_identity_status['reason']
    assert not hasattr(rig, 'mapping_joint_world_qualification')


def test_history_followup_publishes_at_same_boundary_without_changing_mission_or_pixels():
    source, rig, _, _ = overview_source()
    first = source.begin_episode('Inspect every target, then answer the follow-up.')
    cycle = rig.cycle
    published = []
    source.on_capture = published.append
    def answer(value, observation):
        assert value == 'SURVEY_COMPLETE'
        rig.public_task_events.append({'event_id': 'followup-1', 'kind': 'history_followup',
            'text': 'Which target was second in your first-observed order?',
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})
    rig.task_evaluator = SimpleNamespace(report_answer=answer)
    source.report_answer('SURVEY_COMPLETE', snapshot_id=first.snapshot_id, action_step=first.action_step)
    updated = source.agent_observation()
    assert updated.task_text == first.task_text
    assert (updated.snapshot_id, updated.action_step) == (first.snapshot_id, first.action_step)
    assert rig.cycle == cycle
    assert updated.initial_overview is first.initial_overview
    assert updated.public_task_events[0].text.startswith('Which target was second')
    assert published[-1].observation is updated
    assert not first.public_task_events
    restored = AgentObservation.from_payload(updated.to_payload())
    assert restored.public_task_events == updated.public_task_events


def test_followup_transport_refuses_private_extra_fields_and_future_events():
    source, _, _, _ = overview_source()
    observation = source.begin_episode('Inspect every target.')
    event = {'event_id': 'followup-1', 'kind': 'history_followup', 'text': 'Which target was second?',
        'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
    with pytest.raises(TypeError):
        replace(observation, public_task_events=({**event, 'private_bound_target': 'answer'},))
    with pytest.raises(ValueError):
        replace(observation, public_task_events=({**event, 'snapshot_id': observation.snapshot_id + 1},))

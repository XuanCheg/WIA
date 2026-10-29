"""Verify passive setup projection and comparison accounting on CPU only.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_ramp_material_math.py
These tests neither launch native physics nor establish legal robot success.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wia.runtime.commands.check_wia_ramp_material import (
    DEFAULT_RELEASE_PROTOCOL, SETUP_CLEARANCE_M, _counted_setter, _surface_for_setup,
    _zero_candidate_velocities, aggregate_passive_trials, project_release_setup,
    sample_protocol_release, summarize_passive_trial,
)


@pytest.mark.parametrize('ramp_position,euler', [
    ([0., 0., 0.], [0., 0., 0.]),
    ([3., -4., 1.2], [0., 0., 0.]),
    ([3., -4., 1.2], [11., 27., -36.]),
])
def test_projection_uses_world_aabb_center_and_actual_surface_support_without_rotation(ramp_position, euler):
    rotation = Rotation.from_euler('xyz', euler, degrees=True)
    ramp_position = np.array(ramp_position)
    local_surface = np.array([[-.45, -.1, .12], [-.1, .22, .14], [-.2, -.3, .55], [.02, .02, .35]])
    points = rotation.apply(local_surface) + ramp_position
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    origin = rotation.apply([-.6, -.4, .2]) + ramp_position
    quaternion = Rotation.from_euler('xyz', [23., -17., 42.], degrees=True).as_quat()
    saved_points, saved_quaternion = points.copy(), quaternion.copy()
    half, release_x = .027, -.25
    plan = project_release_setup(points, centre, origin, quaternion,
                                ramp_position, rotation.as_quat(), release_x, half)
    delta = np.array(plan['delta_world_m'])
    after_local_surface = rotation.inv().apply(points + delta - ramp_position)
    after_local_centre = rotation.inv().apply(centre + delta - ramp_position)
    np.testing.assert_allclose(after_local_centre[:2], [release_x, 0.], atol=2e-15)
    assert after_local_surface[:, 2].min() == pytest.approx(half + SETUP_CLEARANCE_M, abs=2e-15)
    np.testing.assert_allclose(plan['position_world_m'], origin + delta, atol=2e-15)
    np.testing.assert_array_equal(plan['quaternion_xyzw'], saved_quaternion)
    np.testing.assert_array_equal(points, saved_points)
    np.testing.assert_array_equal(quaternion, saved_quaternion)
    assert not np.allclose(rotation.inv().apply(origin + delta - ramp_position)[:2], [release_x, 0.])
    assert plan['predicted_surface_minimum_ramp_local_z_m'] == pytest.approx(half + .0001)


def test_tilted_ramp_clearance_is_along_the_ramp_normal_not_world_z():
    rotation = Rotation.from_euler('y', 30., degrees=True)
    points = rotation.apply([[-.2, -.1, -.15], [.2, .1, .15]])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    plan = project_release_setup(points, centre, [0., 0., 0.], [0., 0., 0., 1.],
                                [0., 0., 0.], rotation.as_quat(), 0., .02)
    assert plan['delta_ramp_local_m'][:2] == pytest.approx([0., 0.])
    assert plan['delta_world_m'][0] != 0
    np.testing.assert_allclose(plan['delta_world_m'], rotation.apply([0., 0., .1701]), atol=1e-15)


@pytest.mark.parametrize('points', [[], [[0., 0., float('nan')]], [[0., 0.]], [[0., 0., float('inf')]]])
def test_missing_or_invalid_native_surface_is_not_replaced_with_box_corners(points):
    with pytest.raises(ValueError):
        project_release_setup(points, [0., 0., 0.], [0., 0., 0.], [0., 0., 0., 1.],
                              [0., 0., 0.], [0., 0., 0., 1.], -.2, .02)


def test_sphere_fallback_uses_native_radius_and_uniform_world_scale(monkeypatch):
    from wia.authoring.converters import rigid_partial_insertion
    def absent_vertices(obj):
        raise ValueError('Native target collision surface points are unavailable.')
    monkeypatch.setattr(rigid_partial_insertion, '_surface_points', absent_vertices)
    obj = NS(_primitive_type='Sphere', visual_only=False, radius=.03,
             root_link=NS(get_world_scale=lambda: np.array([2., 2., 2.])),
             links={'body': NS(collision_meshes={'mesh': NS(collision_enabled=True)})},
             get_position_orientation=lambda: (np.array([3., 4., 5.]), np.array([0., 0., 0., 1.])))
    rotation = Rotation.from_euler('xyz', [10., 20., 30.], degrees=True)
    points, evidence = _surface_for_setup(obj, rotation.as_quat())
    local = rotation.inv().apply(points - np.array([3., 4., 5.]))
    assert local[:, 2].min() == pytest.approx(-.06)
    assert evidence['world_radius_m'] == .06
    obj.root_link.get_world_scale = lambda: np.array([1., 2., 1.])
    with pytest.raises(ValueError, match='no ellipsoid approximation'):
        _surface_for_setup(obj, rotation.as_quat())


def native_measurement(*, receiver=False, ready=False):
    return {'release_ready': ready, 'safe': True, 'interference': [], 'in_receiver': receiver,
            'linear_speed_m_s': .1 if not ready else 0., 'angular_speed_rad_s': .2 if not ready else 0.}


def passive_trial(candidate='trial_1', repetition=1, *, arrival_s=1., final_receiver=True, ready=True):
    start = 10.
    samples = []
    for index in range(1, 61):
        elapsed = index / 20
        arrived = arrival_s is not None and elapsed >= arrival_s
        if index == 60:
            arrived = final_receiver and arrived
        samples.append({'native_time_s': start + elapsed,
                        'measurement': native_measurement(receiver=arrived)})
    trial = {'candidate': candidate, 'repetition': repetition, 'errors': [],
             'release_native_time_s': start, 'release_measurement': native_measurement(ready=ready),
             'samples': samples}
    trial['summary'] = summarize_passive_trial(trial, duration_s=3.)
    return trial


def test_eligibility_uses_native_release_and_trace_but_never_claims_robot_trial_validity():
    trial = passive_trial()
    summary = trial['summary']
    assert summary['passive_calibration_eligible'] is True
    assert summary['arrival_s'] == 1.
    assert summary['final_in_receiver'] is True
    assert summary['sample_count'] == 60
    assert summary['maximum_sample_gap_s'] == pytest.approx(.05)
    assert summary['observed_duration_s'] == 3.
    assert summary['robot_trial_valid'] is summary['legal_robot_reference_verified'] is False
    assert summary['environment_state_setters_used'] is True


@pytest.mark.parametrize('arrival,finishes', [(None, False), (1., False)])
def test_measured_nonarrival_and_failure_to_stay_are_valid_passive_outcomes(arrival, finishes):
    summary = passive_trial(arrival_s=arrival, final_receiver=finishes)['summary']
    assert summary['passive_calibration_eligible'] is True
    assert summary['arrival_s'] == arrival
    assert summary['final_in_receiver'] is False


@pytest.mark.parametrize('failure,reason', [
    ('release', 'native_release_not_ready_or_unavailable'),
    ('unsafe', 'unsafe_or_unavailable_native_sample'),
    ('interference', 'interference_or_unavailable_native_sample'),
    ('measurement', 'native_measurement_unavailable'),
    ('native_time', 'sample_native_time_unavailable'),
    ('duplicate', 'sample_native_time_not_strictly_increasing'),
    ('backwards', 'sample_native_time_not_strictly_increasing'),
    ('gap', 'native_sample_gap_exceeded_or_unavailable'),
    ('few_samples', 'insufficient_native_samples'),
    ('duration', 'passive_observation_duration_incomplete'),
    ('speed', 'native_speed_measurement_unavailable'),
    ('setup', 'setup_failed'),
])
def test_missing_or_invalid_release_and_trace_never_qualify(failure, reason):
    trial = passive_trial()
    if failure == 'release':
        trial['release_measurement']['release_ready'] = False
    elif failure == 'unsafe':
        trial['samples'][5]['measurement']['safe'] = False
    elif failure == 'interference':
        trial['samples'][5]['measurement']['interference'] = ['/World/robot/finger']
    elif failure == 'measurement':
        trial['samples'][5]['measurement'] = None
    elif failure == 'native_time':
        trial['samples'][5]['native_time_s'] = None
    elif failure == 'duplicate':
        trial['samples'][5]['native_time_s'] = trial['samples'][4]['native_time_s']
    elif failure == 'backwards':
        trial['samples'][5]['native_time_s'] = 9.
    elif failure == 'gap':
        trial['samples'][10]['native_time_s'] += .2
    elif failure == 'few_samples':
        trial['samples'] = trial['samples'][:29]
    elif failure == 'duration':
        trial['samples'] = trial['samples'][:59]
    elif failure == 'speed':
        trial['samples'][5]['measurement']['linear_speed_m_s'] = float('nan')
    else:
        trial['errors'] = ['setup_failed']
    summary = summarize_passive_trial(trial, duration_s=3.)
    assert summary['passive_calibration_eligible'] is False
    assert reason in summary['eligibility_failures']
    assert summary['robot_trial_valid'] is False


CANDIDATES = ('trial_1', 'trial_2', 'trial_3')


def repeated_trials(arrivals):
    return [passive_trial(key, repetition, arrival_s=seconds, final_receiver=seconds is not None)
            for repetition, times in enumerate(arrivals, 1) for key, seconds in zip(CANDIDATES, times)]


def aggregate(trials, repetitions):
    return aggregate_passive_trials(trials, CANDIDATES, repetitions=repetitions, tie_tolerance_s=.15)


def test_complete_repetitions_rank_finishing_arrivals_and_preserve_nonarrival_counts():
    trials = repeated_trials([[.8, 1.2, None], [.9, 1.3, None]])
    result = aggregate(trials, 2)
    assert result['all_requested_repetitions_complete'] is True
    assert result['repeatable'] is True
    assert [row['admissible_set'] for row in result['per_repetition']] == [['trial_1'], ['trial_1']]
    assert result['complete_repetition_admissible_intersection'] == ['trial_1']
    assert result['complete_repetition_admissible_union'] == ['trial_1']
    assert result['per_candidate']['trial_3']['eligible_trial_count'] == 2
    assert result['per_candidate']['trial_3']['eligible_no_arrival_count'] == 2
    assert result['per_candidate']['trial_3']['eligible_finishing_observation_count'] == 0
    assert result['per_candidate']['trial_1']['raw_arrival_statistics']['spread_s'] == pytest.approx(.1)
    assert result['robot_trial_valid'] is False
    assert result['winner_reversal'] == 'not_established_from_one_material_condition'
    assert 'overall_winner' not in result
    json.dumps(result, allow_nan=False)


def test_repeatable_empty_admissible_sets_are_measured_failure_and_inconclusive():
    result = aggregate(repeated_trials([[None, None, None], [None, None, None]]), 2)
    assert result['all_requested_repetitions_complete'] is True
    assert result['repeatable'] is True
    assert result['repeatable_admissible_set'] == []
    assert result['status'] == 'inconclusive'


def test_repetition_disagreement_reports_union_and_intersection_without_median_winner():
    trials = repeated_trials([[.8, 1.2, None], [1.2, .8, None]])
    result = aggregate(trials, 2)
    assert result['repeatable'] is False
    assert result['repeatable_admissible_set'] is None
    assert result['complete_repetition_admissible_intersection'] == []
    assert result['complete_repetition_admissible_union'] == ['trial_1', 'trial_2']


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'invalid_release', 'missing_summary', 'invalid_arrival'])
def test_incomplete_eligible_comparison_never_omits_failed_candidate_to_select_a_winner(failure):
    trials = repeated_trials([[.8, 1.2, None], [.8, 1.2, None]])
    if failure == 'missing':
        trials.pop()
    elif failure == 'duplicate':
        trials.append(deepcopy(trials[-1]))
    elif failure == 'invalid_release':
        trials[-1]['summary']['passive_calibration_eligible'] = False
        trials[-1]['summary']['eligibility_failures'] = ['native_release_not_ready_or_unavailable']
    elif failure == 'missing_summary':
        trials[-1]['summary'] = None
    else:
        trials[-1]['summary']['arrival_s'] = float('nan')
    result = aggregate(trials, 2)
    assert result['all_requested_repetitions_complete'] is False
    assert result['repeatable'] is False
    assert result['per_repetition'][1]['admissible_set'] is None
    assert 'trial_3' in result['per_repetition'][1]['eligibility_failures']
    assert result['status'] == 'inconclusive'
    json.dumps(result, allow_nan=False)


def test_arrival_without_final_receiver_is_reported_but_never_ranked():
    trials = repeated_trials([[.8, 1.2, None]])
    trials[0] = passive_trial('trial_1', 1, arrival_s=.8, final_receiver=False)
    result = aggregate(trials, 1)
    assert result['per_repetition'][0]['complete_eligible_comparison'] is True
    assert result['per_repetition'][0]['admissible_set'] == ['trial_2']
    assert result['per_candidate']['trial_1']['raw_arrival_s_by_repetition'] == pytest.approx([.8])
    assert result['per_candidate']['trial_1']['eligible_not_final_in_receiver_count'] == 1


def release_fixture(monkeypatch, *, failure=None, force_not_ready=False):
    """Fake readbacks reproduce contact-refresh velocity without running physics."""
    from wia.authoring.converters import physical_dynamics_inclined_plane
    clock = [10.]
    monkeypatch.setattr(physical_dynamics_inclined_plane, '_native_time',
        lambda: {'time_s': clock[0], 'physics_dt_s': .05, 'sim_step_dt_s': .05})

    class Candidate:
        def __init__(self):
            self.position, self.quaternion = np.array([1., 2., .03]), np.array([0., 0., 0., 1.])
            self.linear, self.angular = np.zeros(3), np.zeros(3)
            self.calls, self.angular_resets, self.velocity_unavailable = [], 0, False

        def get_position_orientation(self):
            return self.position.copy(), self.quaternion.copy()

        def set_position_orientation(self, *, position, orientation):
            self.calls.append('pose')
            self.position, self.quaternion = position.copy(), orientation.copy()

        def get_linear_velocity(self):
            return self.linear.copy()

        def get_angular_velocity(self):
            if self.velocity_unavailable:
                raise ValueError('Native velocity unavailable.')
            return self.angular.copy()

        def set_linear_velocity(self, value):
            self.calls.append('linear_velocity')
            self.linear = value.copy()

        def set_angular_velocity(self, value):
            self.calls.append('angular_velocity')
            self.angular_resets += 1
            if self.angular_resets == 2 and failure == 'nonzero_velocity':
                return
            self.angular = value.copy()
            if self.angular_resets == 2:
                if failure == 'pose_change':
                    self.position[0] += .001
                elif failure == 'clock_change':
                    clock[0] += .05
                elif failure == 'unavailable_velocity':
                    self.velocity_unavailable = True

        def contact_list(self):
            return [NS(body0='/World/trial/body', body1='/World/ramp/body',
                       position=np.array([1., 2., .01]), normal=np.array([0., 0., 1.]))]

    obj, trial = Candidate(), passive_trial()
    trial.pop('summary')
    _counted_setter(trial, 'pose', 'projection', lambda: obj.set_position_orientation(
        position=obj.position.copy(), orientation=obj.quaternion.copy()))
    _zero_candidate_velocities(obj, trial, 'projection', np.zeros_like)
    # Stand in for the one existing contact-refresh step, not a new physics run.
    obj.linear[:], obj.angular[:] = [.1, 0., 0.], [0., 1.1456, 0.]
    calls = []

    def measure(candidate):
        assert candidate == 'trial_1'
        calls.append(candidate)
        linear, angular = float(np.linalg.norm(obj.get_linear_velocity())), float(np.linalg.norm(obj.get_angular_velocity()))
        return {'release_ready': not force_not_ready and linear <= .05 and angular <= .2,
                'safe': True, 'interference': [], 'in_receiver': False,
                'linear_speed_m_s': linear, 'angular_speed_rad_s': angular,
                'position': obj.position.tolist(), 'quaternion_xyzw': obj.quaternion.tolist()}

    return NS(objects={'trial_1': obj}, _measure=measure), obj, trial, calls


def test_default_protocol_preserves_single_post_refresh_sample_and_no_extra_setters(monkeypatch):
    evaluator, obj, trial, calls = release_fixture(monkeypatch)
    initial_calls = list(obj.calls)
    implicit = sample_protocol_release(evaluator, 'trial_1', trial)
    explicit = sample_protocol_release(evaluator, 'trial_1', trial, release_protocol='project_then_refresh')
    assert implicit == explicit
    assert trial['release_protocol'] == DEFAULT_RELEASE_PROTOCOL == 'project_then_refresh'
    assert obj.calls == initial_calls == ['pose', 'linear_velocity', 'angular_velocity']
    assert len(calls) == 2  # One evaluator read per call, exactly as before.
    assert implicit['measurement']['release_ready'] is False
    assert trial['setter_counts']['completed'] == {'pose': 1, 'linear_velocity': 1, 'angular_velocity': 1}


def test_contact_then_zero_velocity_records_actual_resting_release_without_pose_or_time_change(monkeypatch):
    evaluator, obj, trial, calls = release_fixture(monkeypatch)
    release = sample_protocol_release(evaluator, 'trial_1', trial,
        release_protocol='contact_then_zero_velocity', zero_like=np.zeros_like)
    evidence = trial['release_protocol_evidence']
    assert evidence['accepted'] is True and all(evidence['checks'].values())
    assert evidence['before_final_velocity_reset']['sample']['measurement']['release_ready'] is False
    assert evidence['before_final_velocity_reset']['angular_velocity_rad_s'] == [0., 1.1456, 0.]
    assert release == evidence['after_final_velocity_reset']['sample']
    assert release['measurement']['release_ready'] is True
    assert release['native_time_s'] == evidence['before_final_velocity_reset']['sample']['native_time_s'] == 10.
    assert obj.calls == ['pose', 'linear_velocity', 'angular_velocity', 'linear_velocity', 'angular_velocity']
    assert trial['setter_counts']['attempted'] == trial['setter_counts']['completed'] == {
        'pose': 1, 'linear_velocity': 2, 'angular_velocity': 2}
    assert [event['setter'] for event in trial['setter_events'] if event['stage'] == 'after_contact_refresh'] == [
        'linear_velocity', 'angular_velocity']
    assert evidence['pose_setters_at_final_reset'] == evidence['additional_physics_steps'] == 0
    assert len(calls) == 2
    trial.update(release_native_time_s=release['native_time_s'], release_measurement=release['measurement'])
    summary = summarize_passive_trial(trial, duration_s=3.)
    assert summary['passive_calibration_eligible'] is True
    assert summary['release_protocol'] == 'contact_then_zero_velocity'
    assert summary['setter_counts'] == trial['setter_counts']
    assert summary['robot_trial_valid'] is summary['legal_robot_reference_verified'] is False


@pytest.mark.parametrize('failure', ['pose_change', 'clock_change', 'nonzero_velocity', 'unavailable_velocity'])
def test_final_reset_invariant_failures_and_unavailable_measurements_fail_closed(monkeypatch, failure):
    evaluator, obj, trial, _ = release_fixture(monkeypatch, failure=failure)
    release = sample_protocol_release(evaluator, 'trial_1', trial,
        release_protocol='contact_then_zero_velocity', zero_like=np.zeros_like)
    assert trial['release_protocol_evidence']['accepted'] is False
    assert obj.calls.count('pose') == 1
    trial.update(release_native_time_s=release['native_time_s'], release_measurement=release['measurement'])
    summary = summarize_passive_trial(trial, duration_s=3.)
    assert summary['passive_calibration_eligible'] is False
    assert 'resting_release_invariants_failed_or_unavailable' in summary['eligibility_failures']


def test_reset_never_overrides_an_actual_evaluator_release_not_ready_result(monkeypatch):
    evaluator, _, trial, _ = release_fixture(monkeypatch, force_not_ready=True)
    release = sample_protocol_release(evaluator, 'trial_1', trial,
        release_protocol='contact_then_zero_velocity', zero_like=np.zeros_like)
    assert trial['release_protocol_evidence']['accepted'] is True
    assert release['measurement']['release_ready'] is False
    trial.update(release_native_time_s=release['native_time_s'], release_measurement=release['measurement'])
    summary = summarize_passive_trial(trial, duration_s=3.)
    assert summary['passive_calibration_eligible'] is False
    assert 'native_release_not_ready_or_unavailable' in summary['eligibility_failures']


def test_protocol_dispatch_rejects_unknown_names_and_mixed_comparisons(monkeypatch):
    evaluator, obj, trial, _ = release_fixture(monkeypatch)
    before = list(obj.calls)
    with pytest.raises(ValueError, match='Unknown'):
        sample_protocol_release(evaluator, 'trial_1', trial, release_protocol='invented_protocol')
    assert obj.calls == before
    trials = repeated_trials([[.8, 1.2, None]])
    trials[0]['release_protocol'] = 'contact_then_zero_velocity'
    with pytest.raises(ValueError, match='matching release protocol'):
        aggregate(trials, 1)

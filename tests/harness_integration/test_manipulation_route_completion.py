"""Replay route checkpoints on CPU; run python -m pytest -q this_file.

These deterministic state readbacks test scoring, not native physical feasibility.
"""
from types import SimpleNamespace as NS
from copy import deepcopy
import math

import numpy as np
import pytest

from wia.authoring.converters import physical_structure_liquid_volume as liquid
from wia.authoring.converters import specular_reflection_spatial_relations as mirror
from wia.authoring.converters import spatial_relations_geometric_configuration as triangle


def test_liquid_ordered_return_is_independent_terminal_item(monkeypatch, tmp_path):
    state = {'time_s': 0., 'position': [0., -.8, 0.], 'yaw_rad': math.pi / 2,
        'safe': True, 'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.,
        'objects': {key: {'position': [float(key == 'right'), 0., 0.],
            'centre': [float(key == 'right'), 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.],
            'fixed_base': True, 'visual_only': False} for key in liquid.KEYS}}
    monkeypatch.setattr(liquid, 'measure_native_state', lambda *_: deepcopy(state))
    e = liquid.ContainerDockingEvaluator.__new__(liquid.ContainerDockingEvaluator)
    e.session, e.bound, e.output = None, {}, tmp_path
    e.config = {'parameters': {}, 'operation': liquid.OPERATIONS[1]}
    e.required, e.maximum_speed, e.translation_tolerance = ['left', 'right', 'left'], 100., .01
    e.episode = 0
    obs = NS(snapshot_id=1, action_step=0)
    e.begin_episode(obs)
    def hold(x):
        state['position'] = [x, -.8, 0.]
        for _ in range(15):
            state['time_s'] += .05
            e.sample_step()
        return e.record(obs, action=None)
    assert hold(1.)['completed_goals'] == 0  # Wrong first stop cannot advance.
    first = hold(0.)
    assert first['completion_score'] == pytest.approx(1 / 3)
    second = hold(1.)
    assert second['completed_goals'] == 2
    assert not second['task_success']
    returned = hold(0.)
    assert returned['completion_score'] == 1 and returned['task_success']
    assert [i['kind'] for i in returned['completion_items']] == [
        'historical_checkpoint', 'historical_checkpoint', 'terminal_condition']
    left_again = hold(.5)
    assert left_again['completed_goals'] == 2  # Historical visits survive departure; final state does not.
    state['safe'] = False
    unsafe = hold(0.)
    assert unsafe['completion_score'] == 0
    assert len(unsafe['visits']) == 3  # Evidence is retained even when safety fails.


def test_mirror_a_then_b_return_keeps_prefix_and_requires_fresh_terminal(monkeypatch, tmp_path):
    state = {'position': [0., 0., 0.], 'yaw_rad': 0., 'state_valid': True,
        'candidate_distances_xy_m': {'prop_a': 2., 'prop_b': 2.},
        'object_heading_errors_deg': {'prop_a': 0., 'prop_b': 0.},
        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'mirror_heading_error_deg': 0.}
    monkeypatch.setattr(mirror, '_objects', lambda *_: {})
    monkeypatch.setattr(mirror, 'measure_native_state', lambda *_: state)
    monkeypatch.setattr(mirror, 'qualify_operation', lambda *_: {'accepted': True, 'native_state': state})
    p = dict(approach_order=['prop_a', 'prop_b'], departure_distance_m=1.1,
        approach_minimum_m=.6, approach_maximum_m=.95, candidate_distance_margin_m=.05,
        heading_tolerance_deg=25., maximum_stopped_speed_m_s=.02,
        maximum_stopped_angular_speed_rad_s=.1, return_tolerance_m=.2,
        maximum_root_z_change_m=.05, minimum_travel_m=.75, minimum_excursion_m=.5,
        minimum_between_stations_m=.25, stability_steps=2)
    session = NS(pipeline=NS(og=NS(sim=NS(current_time=0.))))
    controller = NS(config=NS(max_translation_speed_m_s=100., translation_tolerance_m=.01,
        max_rotation_speed_deg_s=100., rotation_tolerance_deg=1.))
    e = mirror.MirrorDistanceEvaluator(session, controller,
        {'parameters': p, 'operation': mirror.OPERATIONS[2]}, tmp_path)
    obs = NS(snapshot_id=1, action_step=0)
    e.begin_episode(obs)
    def stop(x, target):
        state['position'] = [x, 0., 0.]
        state['candidate_distances_xy_m'] = {key: .8 if key == target else 2. for key in ('prop_a', 'prop_b')}
        for _ in range(3):
            session.pipeline.og.sim.current_time += .05
            e.sample_step()
        return e.record(obs, action=None)
    assert stop(1., 'prop_b')['completed_goals'] == 0
    assert stop(2., 'prop_a')['completed_goals'] == 1
    assert stop(3., 'prop_b')['completed_goals'] == 2
    result = stop(0., None)
    assert result['task_success'] and result['score_percent'] == 100
    assert stop(.5, None)['completed_goals'] == 2
    state['state_valid'] = False
    assert stop(0., None)['completed_goals'] == 0


def test_triangle_circuit_latches_before_return_and_keeps_terminal_separate(tmp_path):
    e = triangle.TriangleNavigationEvaluator.__new__(triangle.TriangleNavigationEvaluator)
    e.config = {'operation': triangle.OPERATIONS[0], 'parameters': dict(
        maximum_step_displacement_m=10., minimum_path_m=0., minimum_excursion_m=0., stop_duration_s=.5)}
    e.initial = {'base': {'position': [1., 0., 0.]}}
    e.output, e.episode = tmp_path, 1
    e.centres, e.centre = np.zeros((3, 2)), np.zeros(2)
    e.last, e.angles, e.winding = np.array([1., 0., 0.]), np.zeros(3), np.zeros(3)
    e.last_time, e.path_m, e.excursion_m, e.quiet_seconds = 0., 0., 0., 0.
    e.trajectory_valid, e.previously_quiet, e.circuit_checkpoint = True, False, None
    e.stage, e.survey_visits = 0, []
    state = {'time_s': 0., 'base': {'position': [1., 0., 0.]}}
    e._measure = lambda: state
    e._checks = lambda _: dict(original_vertices_preserved=True, source_floor_supported_and_clear=True,
        within_formation_neighborhood=True, at_return_position=False, stopped=True)
    for angle in np.linspace(0., 2 * math.pi, 65)[1:]:
        state['time_s'] += .05
        state['base']['position'] = [math.cos(angle), math.sin(angle), 0.]
        e.sample_step()
    assert e._operation_complete()
    result = e.record(NS(), action=None)
    assert result['completion_score'] == .5
    # Backtracking to a valid return route cannot erase the completed full circuit.
    for angle in np.linspace(2 * math.pi, math.pi, 33)[1:]:
        state['time_s'] += .05
        state['base']['position'] = [math.cos(angle), math.sin(angle), 0.]
        e.sample_step()
    assert np.all(e.winding < 2 * math.pi)
    assert e.record(NS())['completed_goals'] == 1
    e.trajectory_valid = False
    assert e.record(NS())['completed_goals'] == 0


def test_triangle_vertices_require_order_and_terminal_return(tmp_path):
    e = triangle.TriangleNavigationEvaluator.__new__(triangle.TriangleNavigationEvaluator)
    e.config = {'operation': triangle.OPERATIONS[1], 'parameters': dict(
        maximum_step_displacement_m=10., minimum_path_m=1., minimum_excursion_m=.5,
        stop_duration_s=.5, survey_separation_m=.4, survey_min_distance_m=.65,
        survey_max_distance_m=1.5, survey_heading_cosine=math.cos(math.pi / 6))}
    e.initial = {'base': {'position': [-2., 0., 0.]}}
    e.output, e.episode = tmp_path, 1
    e.centres = np.array([[0., 0.], [3., 0.], [6., 0.]])
    e.centre = e.centres.mean(axis=0)
    e.last, e.angles, e.winding = np.array([-2., 0., 0.]), np.full(3, math.pi), np.zeros(3)
    e.last_time, e.path_m, e.excursion_m, e.quiet_seconds = 0., 0., 0., 0.
    e.trajectory_valid, e.previously_quiet, e.circuit_checkpoint = True, False, None
    e.stage, e.survey_visits = 0, []
    state = {'time_s': 0., 'base': {'position': [-2., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.]}}
    e._measure = lambda: state
    e._checks = lambda _: dict(original_vertices_preserved=True, source_floor_supported_and_clear=True,
        within_formation_neighborhood=True, at_return_position=state['base']['position'][0] == -2., stopped=True)
    def visit(x, samples=1):
        state['base']['position'] = [x, 0., 0.]
        for _ in range(samples):
            state['time_s'] += .05
            e.sample_step()
        return e.record(NS())
    assert visit(5.2)['completed_goals'] == 0
    assert visit(-.8)['completed_goals'] == 1
    assert visit(5.2)['completed_goals'] == 1
    assert visit(2.2)['completed_goals'] == 2
    assert visit(5.2)['completion_score'] == .75
    assert visit(-2., 12)['task_success']
    assert visit(-.8)['completed_goals'] == 3


def test_correspondence_return_starts_after_approach_event(monkeypatch, tmp_path):
    from wia.authoring.converters import specular_reflection_correspondence as correspondence
    state = {'position': [0., 0., 0.], 'yaw_deg': 0., 'state_valid': True,
        'candidate_distances_m': {'prop_a': 2., 'prop_b': 3.}, 'target_heading_error_deg': 0.,
        'mirror_heading_error_deg': 0., 'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.}
    monkeypatch.setattr(correspondence, '_objects', lambda *_: {})
    monkeypatch.setattr(correspondence, 'measure_native_state', lambda *_: deepcopy(state))
    monkeypatch.setattr(correspondence, 'qualify_operation',
        lambda *_: {'accepted': True, 'native_state': deepcopy(state)})
    p = dict(selected_prop='prop_a', approach_leave_m=1.1, approach_minimum_m=.6,
        approach_maximum_m=.95, candidate_distance_margin_m=.05, heading_tolerance_deg=25.,
        maximum_terminal_speed_m_s=.02, maximum_terminal_angular_speed_rad_s=.1,
        return_tolerance_m=.2, minimum_travel_m=.75, minimum_excursion_m=.5, stability_steps=2)
    session = NS(pipeline=NS(og=NS(sim=NS(current_time=0.))))
    controller = NS(config=NS(max_translation_speed_m_s=100., translation_tolerance_m=.01,
        max_rotation_speed_deg_s=100., rotation_tolerance_deg=1.))
    e = correspondence.CorrespondenceEvaluator(session, controller,
        {'parameters': p, 'operation': correspondence.OPERATIONS[1]}, tmp_path)
    obs = NS(snapshot_id=1, action_step=0)
    e.begin_episode(obs)
    def step(x, distance):
        state['position'] = [x, 0., 0.]
        state['candidate_distances_m']['prop_a'] = distance
        session.pipeline.og.sim.current_time += .05
        e.sample_step()
        return e.record(obs, action=None)
    step(1., 1.5)
    step(0., .8)
    reached = step(0., .8)
    assert reached['completion_score'] == .5
    assert reached['terminal_stability_samples'] == 0
    assert not step(0., .8)['task_success']
    result = step(0., .8)
    assert result['task_success'] and result['completion_score'] == 1.
    assert result['reflected_visibility_certified'] is False
    assert step(.5, 1.3)['completed_goals'] == 1


def test_background_grasp_exception_requires_policy_and_respects_phase_one():
    from wia.tasks.background_policy import legal_background_grasp
    session = NS()
    assert not legal_background_grasp(session, 'chair')
    session.background_movement_policy = {'allowed': {'chair': object()}}
    assert legal_background_grasp(session, 'chair')
    assert not legal_background_grasp(session, 'protected_anchor')
    session.temporal_phase_controller = NS(phase='phase_1')
    assert not legal_background_grasp(session, 'chair')
    session.temporal_phase_controller.phase = 'phase_2'
    assert legal_background_grasp(session, 'chair')

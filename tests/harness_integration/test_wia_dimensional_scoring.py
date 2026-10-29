"""Check dimensional native-measurement reducers with CPU-only geometry.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_dimensional_scoring.py
These fixtures test sampled evidence contracts, not native feasibility or robot paths.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from wia.tasks.wia_dimensional_scoring import DimensionalProgress

DESIGN = json.loads((__import__('wia.resources', fromlist=['ROOT']).ROOT /
    'explore_to_achieve/wia_v1/dimensional_v1/design.json').read_text())
DT = .0625


def parameters(operation='sort_intrinsic_dimension'):
    count = 2 if operation == 'select_pass_aperture' else 3
    return {'operation': operation, 'candidates': {f'c{i}': f'block_{i}' for i in range(count)},
        'destinations': ['pad_0'] if count == 2 else ['pad_0', 'pad_1', 'pad_2'],
        'scoring': deepcopy(DESIGN['scoring']), 'unselected_position_tolerance_m': .02,
        'aperture': {'entry_x_m': 0., 'exit_x_m': .1, 'y_min_m': -.1, 'y_max_m': .1,
                     'z_min_m': 0., 'z_max_m': .1},
        'candidate_dimensions_m': {'c0': [99., 99., 99.]}}


def target(name, dimensions, position, destinations, *, rotation=None):
    local = np.array(list(product((-.5, .5), repeat=3))) * dimensions
    matrix = np.eye(3) if rotation is None else rotation
    vertices = local @ matrix.T + position
    return {'object_name': name, 'vertices_m': vertices.tolist(), 'dimensions_m': list(dimensions),
        'position_m': list(position), 'quaternion_xyzw': [0., 0., 0., 1.],
        'lowest_z_m': float(vertices[:, 2].min()), 'linear_speed_m_s': 0.,
        'angular_speed_rad_s': 0., 'dynamic_collidable': True, 'robot_contact': False,
        'destinations': {pad: {'touching': False, 'on_top': False, 'footprint_inside': False}
                         for pad in destinations}}


def sample(p):
    aperture = p['operation'] == 'select_pass_aperture'
    dimensions = [[.06, .06, .12], [.3, .3, .3]] if aperture else [[.09, .05, .05], [.13, .05, .05], [.17, .05, .05]]
    return {'native_time_s': 0., 'control_valid': True, 'fixtures_valid': True, 'geometry_valid': True,
        'grasp': {'available': True, 'held_names': [], 'released': True},
        'targets': {key: target(name, dims, [-.3 - i * .4, i * .5, dims[2] / 2], p['destinations'])
                    for i, ((key, name), dims) in enumerate(zip(p['candidates'].items(), dimensions))}}


def move(s, key, position, *, rotation=None):
    row = s['targets'][key]
    fresh = target(row['object_name'], row['dimensions_m'], position, list(row['destinations']), rotation=rotation)
    row.update(vertices_m=fresh['vertices_m'], position_m=fresh['position_m'], lowest_z_m=fresh['lowest_z_m'])
    if rotation is not None:
        row['quaternion_xyzw'] = [0., np.sqrt(.5), 0., np.sqrt(.5)]


def tick(progress, s, delta=DT):
    s['native_time_s'] += delta
    return progress.update(deepcopy(s))


def dwell(progress, s, steps=8):
    for _ in range(steps):
        result = tick(progress, s)
    return result


def set_pad(s, key, pad):
    for name, relation in s['targets'][key]['destinations'].items():
        relation.update({field: name == pad for field in relation})


def pick_and_sort(progress, s, assignment=None):
    assignment = assignment or progress.result()['expected_assignment']
    for key, pad in assignment.items():
        row = s['targets'][key]
        s['grasp'].update(held_names=[row['object_name']], released=False)
        position = np.array(row['position_m']) + [0., 0., .0625]
        move(s, key, position)
        row['robot_contact'] = True
        tick(progress, s)
        move(s, key, [1. + int(pad[-1]), 0., row['dimensions_m'][2] / 2])
        row['robot_contact'] = False
        set_pad(s, key, pad)
        s['grasp'].update(held_names=[], released=True)
        tick(progress, s)


def aperture_ready():
    p = parameters('select_pass_aperture')
    s = sample(p)
    progress = DimensionalProgress(p)
    progress.begin(deepcopy(s))
    s['grasp'].update(held_names=['block_0'], released=False)
    move(s, 'c0', [-.3, 0., .18])
    s['targets']['c0']['robot_contact'] = True
    tick(progress, s)
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    move(s, 'c0', [-.3, 0., .05], rotation=rotation)
    tick(progress, s)
    return progress, s, rotation


def pass_and_place(progress, s, rotation, *, release_before_entry=True):
    if release_before_entry:
        s['grasp'].update(held_names=[], released=True)
        s['targets']['c0']['robot_contact'] = False
        tick(progress, s)
    for x in (-.04, .08, .2):
        move(s, 'c0', [x, 0., .05], rotation=rotation)
        tick(progress, s)
    s['grasp'].update(held_names=[], released=True)
    s['targets']['c0']['robot_contact'] = False
    set_pad(s, 'c0', 'pad_0')
    tick(progress, s)


def checkpoints(result):
    return {item['id']: item['completed'] for item in result['completion_items']}


def test_sort_requires_all_original_grasps_lifts_and_simultaneous_released_stability():
    p, s = parameters(), sample(parameters())
    progress = DimensionalProgress(p)
    initial = progress.begin(s)
    assert initial['expected_assignment'] == {'c0': 'pad_0', 'c1': 'pad_1', 'c2': 'pad_2'}
    pick_and_sort(progress, s)
    assert not progress.result()['task_success']
    result = dwell(progress, s)
    assert result['task_success'] and result['evaluation_validity'] == 'complete'
    assert checkpoints(result) == {'grasp_lift': True, 'assignment': True, 'stable_placement': True}
    assert all(state['stable_duration_s'] == .5 for state in result['per_target'].values())
    set_pad(s, 'c0', 'pad_2')
    assert not tick(progress, s)['task_success']


def test_sort_uses_intrinsic_longest_edge_even_when_rotated_world_aabbs_match():
    p, s = parameters(), sample(parameters())
    side = (.21 + .03) / np.sqrt(2)
    rotation = np.array([[1., -1., 0.], [1., 1., 0.], [0., 0., np.sqrt(2)]]) / np.sqrt(2)
    s['targets']['c0'] = target('block_0', [.21, .03, .05], [-.3, 0., .025], p['destinations'], rotation=rotation)
    s['targets']['c1'] = target('block_1', [side, side, .05], [-.7, .5, .025], p['destinations'])
    s['targets']['c2'] = target('block_2', [.3, .04, .05], [-1.1, 1., .025], p['destinations'])
    a = np.ptp(s['targets']['c0']['vertices_m'], axis=0)
    b = np.ptp(s['targets']['c1']['vertices_m'], axis=0)
    assert np.allclose(a, b)
    progress = DimensionalProgress(p)
    result = progress.begin(s)
    assert result['expected_assignment'] == {'c1': 'pad_0', 'c0': 'pad_1', 'c2': 'pad_2'}
    pick_and_sort(progress, s, {'c0': 'pad_0', 'c1': 'pad_1', 'c2': 'pad_2'})
    result = dwell(progress, s)
    assert checkpoints(result)['grasp_lift'] is True
    assert checkpoints(result)['assignment'] is False and not result['task_success']


@pytest.mark.parametrize('failure', ['rest_only', 'not_released', 'robot_supported'])
def test_hand_supported_or_rest_only_sort_cannot_succeed(failure):
    p, s = parameters(), sample(parameters())
    progress = DimensionalProgress(p)
    progress.begin(s)
    if failure == 'rest_only':
        for key, pad in progress.result()['expected_assignment'].items():
            set_pad(s, key, pad)
    else:
        pick_and_sort(progress, s)
        if failure == 'not_released':
            s['grasp']['released'] = False
        else:
            s['targets']['c1']['robot_contact'] = True
    assert not dwell(progress, s)['task_success']


def test_result_reads_and_duplicate_timestamp_frames_never_create_dwell():
    p, s = parameters(), sample(parameters())
    progress = DimensionalProgress(p)
    progress.begin(s)
    pick_and_sort(progress, s)
    before = progress.result()
    for _ in range(100):
        progress.result()
        progress.update(deepcopy(s))
    assert progress.result() == before
    assert not before['task_success']
    before['per_target']['c0']['grasp_seen'] = False
    assert progress.result()['per_target']['c0']['grasp_seen'] is True


@pytest.mark.parametrize('gap', [0., .01])
def test_initial_assignment_ties_or_insufficient_gap_are_permanently_unavailable(gap):
    p, s = parameters(), sample(parameters())
    s['targets']['c1']['dimensions_m'][0] = .09 + gap
    progress = DimensionalProgress(p)
    result = progress.begin(s)
    assert result['evaluation_validity'] == 'unavailable' and result['expected_assignment'] is None
    assert 'initial_intrinsic_dimension_assignment_ambiguous' in result['initial_measurement_errors']
    s['targets']['c1']['dimensions_m'][0] = .13
    assert not tick(progress, s)['task_success']


def test_rotated_fit_and_released_physical_passage_can_succeed_without_hand_path_claim():
    progress, s, rotation = aperture_ready()
    pass_and_place(progress, s, rotation, release_before_entry=True)
    pending = dwell(progress, s)
    assert pending['evaluation_validity'] == 'pending_done' and pending['completion_items'] is None
    result = progress.declare_done()
    assert result['task_success'] and result['continuous_path_certified'] is False
    assert checkpoints(result) == {'choice': True, 'grasp_lift': True, 'passage': True, 'stable_placement': True}
    assert result['physical_feasibility_of_unselected_candidates_inferred'] is False
    assert result['per_target']['c0']['path']['completed_evidence']['sampled_passage'] is True


@pytest.mark.parametrize('failure', ['around_frame', 'front_back_jump', 'unsafe_adjacent_union'])
def test_aperture_bypass_or_missing_actual_overlap_never_establishes_passage(failure):
    progress, s, rotation = aperture_ready()
    if failure == 'around_frame':
        move(s, 'c0', [-.3, .3, .05], rotation=rotation)
        tick(progress, s)
        move(s, 'c0', [.2, .3, .05], rotation=rotation)
    elif failure == 'unsafe_adjacent_union':
        move(s, 'c0', [-.3, 0., .25], rotation=rotation)
        tick(progress, s)
        move(s, 'c0', [-.04, 0., .05], rotation=rotation)
    else:
        move(s, 'c0', [.2, 0., .05], rotation=rotation)
    tick(progress, s)
    move(s, 'c0', [.2, 0., .05], rotation=rotation)
    s['grasp'].update(held_names=[], released=True)
    s['targets']['c0']['robot_contact'] = False
    set_pad(s, 'c0', 'pad_0')
    dwell(progress, s)
    result = progress.declare_done()
    assert not result['task_success'] and checkpoints(result)['passage'] is False
    assert result['per_target']['c0']['path']['interruption_reason'] is not None


@pytest.mark.parametrize('interruption', ['gap', 'missing_vertices', 'missing_time', 'geometry_unknown'])
def test_interrupted_ongoing_passage_requires_return_to_front(interruption):
    progress, s, rotation = aperture_ready()
    move(s, 'c0', [-.04, 0., .05], rotation=rotation)
    tick(progress, s)
    interrupted = deepcopy(s)
    interrupted['native_time_s'] += .25 if interruption == 'gap' else DT
    if interruption == 'missing_vertices':
        interrupted['targets']['c0']['vertices_m'] = None
    elif interruption == 'missing_time':
        interrupted['native_time_s'] = None
    elif interruption == 'geometry_unknown':
        interrupted['geometry_valid'] = None
    result = progress.update(interrupted)
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable'
    s['native_time_s'] += .25 + DT
    move(s, 'c0', [.2, 0., .05], rotation=rotation)
    progress.update(deepcopy(s))
    assert not progress.result()['per_target']['c0']['path']['completed']
    move(s, 'c0', [-.3, 0., .05], rotation=rotation)
    tick(progress, s)  # Return and rearm after the interrupted attempt.
    pass_and_place(progress, s, rotation)
    dwell(progress, s)
    assert progress.declare_done()['task_success']


def test_completed_passage_is_historical_but_later_gap_restarts_final_dwell():
    progress, s, rotation = aperture_ready()
    pass_and_place(progress, s, rotation)
    result = tick(progress, s, delta=.25)
    assert result['evaluation_validity'] == 'pending_done'
    assert result['native_evaluation_validity'] == 'unavailable'
    assert result['per_target']['c0']['path']['completed']
    assert result['per_target']['c0']['stable_placement'] is False
    tick(progress, s)
    dwell(progress, s)
    assert progress.declare_done()['task_success']


@pytest.mark.parametrize('violation', ['control', 'fixture', 'shape', 'dimensions', 'clock'])
def test_explicit_constraints_and_clock_changes_latch_invalid(violation):
    progress, s, rotation = aperture_ready()
    if violation in ('control', 'fixture', 'shape'):
        s[{'control': 'control_valid', 'fixture': 'fixtures_valid', 'shape': 'geometry_valid'}[violation]] = False
    elif violation == 'dimensions':
        s['targets']['c0']['dimensions_m'][0] *= 1.01
    else:
        s['native_time_s'] = -1.
    pending = tick(progress, s)
    assert pending['evaluation_validity'] == 'pending_done' and pending['completion_score'] is None
    assert pending['native_evaluation_validity'] == 'invalid'
    result = progress.declare_done()
    assert result['evaluation_validity'] == 'invalid' and not result['task_success']
    assert result['constraints_satisfied'] is False and result['completion_score'] is None
    assert result['constraint_violations']
    s['control_valid'] = s['fixtures_valid'] = s['geometry_valid'] = True
    s['native_time_s'] = 10.
    assert progress.update(s)['evaluation_validity'] == 'invalid'


@pytest.mark.parametrize('field,value', [('robot_contact', 1), ('linear_speed_m_s', float('nan')),
    ('lowest_z_m', float('inf')), ('position_m', [0., 0., float('nan')]),
    ('dimensions_m', [.09, None, .05])])
def test_missing_or_nonfinite_measurements_are_unavailable_never_a_pass(field, value):
    p, s = parameters(), sample(parameters())
    progress = DimensionalProgress(p)
    progress.begin(s)
    pick_and_sort(progress, s)
    s['targets']['c0'][field] = value
    result = tick(progress, s)
    assert result['evaluation_validity'] == 'unavailable' and not result['task_success']
    assert result['completion_score'] is None and not result['constraint_violations']
    json.dumps(result, allow_nan=False)


def test_initial_hold_is_not_pickup_evidence_until_empty_release_then_a_new_hold():
    p, s = parameters(), sample(parameters())
    s['grasp'].update(held_names=['block_0'], released=False)
    progress = DimensionalProgress(p)
    progress.begin(s)
    move(s, 'c0', np.array(s['targets']['c0']['position_m']) + [0., 0., .0625])
    assert tick(progress, s)['per_target']['c0']['grasp_seen'] is False
    s['grasp'].update(held_names=[], released=True)
    tick(progress, s)
    s['grasp'].update(held_names=['block_0'], released=False)
    assert tick(progress, s)['per_target']['c0']['lift_seen'] is True


def test_initial_geometry_failure_remains_unavailable_and_begin_resets_all_evidence():
    p, s = parameters(), sample(parameters())
    progress = DimensionalProgress(p)
    s['geometry_valid'] = False
    assert progress.begin(s)['evaluation_validity'] == 'unavailable'
    s['geometry_valid'] = True
    assert tick(progress, s)['evaluation_validity'] == 'unavailable'
    fresh = sample(p)
    assert progress.begin(fresh)['evaluation_validity'] == 'complete'
    pick_and_sort(progress, fresh)
    assert dwell(progress, fresh)['task_success']
    reset = progress.begin(sample(p))
    assert not reset['task_success'] and not checkpoints(reset)['grasp_lift']


def test_aperture_choice_checkpoint_does_not_claim_an_unexecuted_candidate_is_feasible():
    p = parameters('select_pass_aperture')
    s = sample(p)
    progress = DimensionalProgress(p)
    progress.begin(s)
    s['grasp'].update(held_names=['block_1'], released=False)
    pending = tick(progress, s)
    assert pending['completion_items'] is None and pending['selected_candidate'] is None
    result = progress.declare_done()
    assert checkpoints(result)['choice'] is True and result['selected_candidate'] == 'c1'
    assert not checkpoints(result)['passage'] and not result['task_success']
    choice = next(item for item in result['completion_items'] if item['id'] == 'choice')
    assert choice['evidence_scope'] == 'measured_grasp_only_not_feasibility'


def assert_pending_done(result):
    assert result['evaluation_validity'] == 'pending_done'
    assert result['evaluation_finalized'] is False
    assert result['task_success'] is result['physical_success'] is False
    assert result['selected_candidate'] is None
    for key in ('completion_items', 'completion_score', 'score_percent', 'completed_goals', 'total_goals'):
        assert result[key] is None


def test_aperture_free_attempts_allow_large_displacement_wrong_grasp_release_and_switch():
    p = parameters('select_pass_aperture')
    p.pop('unselected_position_tolerance_m')
    s = sample(p)
    progress = DimensionalProgress(p)
    assert_pending_done(progress.begin(deepcopy(s)))
    move(s, 'c0', [-.26, 0., .06])  # Exceeds the retired 2 cm restriction before grasp.
    assert_pending_done(tick(progress, s))
    s['grasp'].update(held_names=['block_1'], released=False)
    move(s, 'c1', [-.7, .5, .22])
    assert_pending_done(tick(progress, s))
    s['grasp'].update(held_names=[], released=True)
    tick(progress, s)
    s['grasp'].update(held_names=['block_0'], released=False)
    move(s, 'c0', [-.26, 0., .18])
    s['targets']['c0']['robot_contact'] = True
    tick(progress, s)
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    move(s, 'c0', [-.26, 0., .05], rotation=rotation)
    tick(progress, s)
    pass_and_place(progress, s, rotation)
    pending = dwell(progress, s)
    assert_pending_done(pending)
    assert pending['constraint_violations'] == []
    assert all(state['grasp_seen'] for state in pending['per_target'].values())
    terminal = progress.declare_done()
    assert terminal['selected_candidate'] == 'c0' and terminal['task_success']
    assert terminal['evaluation_finalized']
    assert terminal['scoring_version'] == 'wia_dimensional_progress_v2'


def test_aperture_grasp_or_fit_without_passage_and_released_placement_fails_at_done():
    progress, s, rotation = aperture_ready()
    assert_pending_done(progress.result())
    terminal = progress.declare_done()
    assert not terminal['task_success']
    assert checkpoints(terminal) == {'choice': True, 'grasp_lift': True, 'passage': False, 'stable_placement': False}
    assert terminal['selected_candidate'] == 'c0'


def test_aperture_goal_checkpoints_cannot_be_combined_across_candidates():
    progress, s, rotation = aperture_ready()
    for x in (-.04, .08, .2):
        move(s, 'c0', [x, 0., .05], rotation=rotation)
        tick(progress, s)
    s['grasp'].update(held_names=[], released=True)
    s['targets']['c0']['robot_contact'] = False
    set_pad(s, 'c1', 'pad_0')
    tick(progress, s)
    pending = dwell(progress, s)
    assert pending['per_target']['c0']['path']['completed']
    assert pending['per_target']['c1']['stable_placement']
    result = progress.declare_done()
    assert not result['task_success']
    assert result['selected_candidate'] == 'c0'  # A never-grasped placement cannot become the selected attempt.
    assert checkpoints(result) == {'choice': True, 'grasp_lift': True, 'passage': True, 'stable_placement': False}


@pytest.mark.parametrize('success_before_done', [False, True])
def test_done_latches_success_or_failure_and_later_samples_cannot_rewrite_it(success_before_done):
    progress, s, rotation = aperture_ready()
    if success_before_done:
        pass_and_place(progress, s, rotation)
        dwell(progress, s)
    terminal = progress.declare_done()
    assert terminal['task_success'] is success_before_done
    raw_history = deepcopy(progress._states)
    s['control_valid'] = False
    s['native_time_s'] = -100.
    assert progress.update(s) == terminal
    assert progress._states == raw_history
    assert progress.declare_done() == terminal
    changed = progress.result()
    changed['per_target']['c0']['grasp_seen'] = False
    assert progress.result() == terminal


def test_terminal_selection_ties_use_original_candidate_order_and_choice_requires_actual_grasp():
    p = parameters('select_pass_aperture')
    p['unselected_position_tolerance_m'] = 0.  # Accepted but inert historical key.
    progress = DimensionalProgress(p)
    progress.begin(sample(p))
    result = progress.declare_done()
    assert result['selected_candidate'] is None
    assert checkpoints(result) == {'choice': False, 'grasp_lift': False, 'passage': False, 'stable_placement': False}
    assert result['completion_score'] == 0.


def test_sorting_retains_live_v1_results_and_done_does_not_latch():
    p = parameters()
    s = sample(p)
    progress = DimensionalProgress(p)
    progress.begin(s)
    pick_and_sort(progress, s)
    result = dwell(progress, s)
    assert result['task_success'] and result['scoring_version'] == 'wia_dimensional_progress_v1'
    assert 'evaluation_finalized' not in result
    assert progress.declare_done() == result
    set_pad(s, 'c0', 'pad_2')
    assert not tick(progress, s)['task_success']


def test_terminal_equal_actual_attempts_use_original_key_order():
    p = parameters('select_pass_aperture')
    s = sample(p)
    progress = DimensionalProgress(p)
    progress.begin(s)
    for name in ('block_1', 'block_0'):
        s['grasp'].update(held_names=[name], released=False)
        tick(progress, s)
        s['grasp'].update(held_names=[], released=True)
        tick(progress, s)
    result = progress.declare_done()
    assert result['selected_candidate'] == 'c0'
    assert checkpoints(result)['choice'] is True


def test_any_original_candidate_can_complete_the_same_original_goal_after_switching():
    p = parameters('select_pass_aperture')
    s = sample(p)
    # Both originals physically fit after rotation; candidate order must not
    # select an answer or prevent the second object's actual completion.
    s['targets']['c1'] = target('block_1', [.06, .06, .12], [-.7, .5, .06], p['destinations'])
    progress = DimensionalProgress(p)
    progress.begin(s)
    s['grasp'].update(held_names=['block_0'], released=False)
    tick(progress, s)
    s['grasp'].update(held_names=[], released=True)
    tick(progress, s)
    s['grasp'].update(held_names=['block_1'], released=False)
    move(s, 'c1', [-.3, 0., .18])
    tick(progress, s)
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    for x in (-.3, -.04, .08, .2):
        move(s, 'c1', [x, 0., .05], rotation=rotation)
        tick(progress, s)
    s['grasp'].update(held_names=[], released=True)
    set_pad(s, 'c1', 'pad_0')
    tick(progress, s)
    assert_pending_done(dwell(progress, s))
    result = progress.declare_done()
    assert result['selected_candidate'] == 'c1'
    assert result['task_success'] and result['completion_score'] == 1.
    assert all(checkpoints(result).values())

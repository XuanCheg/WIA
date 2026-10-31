"""Exercise mirror evidence contracts with CPU-only native-sample fixtures.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_mirror_scoring.py
Synthetic samples test reducer semantics, not optical visibility or robot feasibility.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest

from wia.tasks.wia_mirror_scoring import MirrorProgress


ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT
DESIGN = json.loads((ROOT / 'explore_to_achieve/wia_v1/mirror_v1/design.json').read_text())
DT = .0625


def parameters(operation='mirror_select_deliver', *, mode='reflected', relation='selected_farther'):
    p = {'operation': operation, 'candidates': {'c0': 'native_0', 'c1': 'native_1'},
        'destinations': {'blue': 'pad_blue', 'yellow': 'pad_yellow'},
        'appearance_mode': mode, 'scoring': deepcopy(DESIGN['scoring']),
        'completion_items': [{'id': key} for key in DESIGN['completion_items'][operation]]}
    if operation == 'mirror_reconfigure':
        p.update(relation=relation, distance_gap_m=.1 if relation == 'selected_nearer' else .2,
            requires_both_objects_moved=relation == 'reverse_initial_distance_order',
            non_goal_initialization_required=True)
    return p


def move(s, key, xyz):
    row = s['targets'][key]
    vertices = np.array(list(product((-.5, .5), repeat=3))) * row['dimensions_m'] + xyz
    row.update(position_m=list(xyz), vertices_m=vertices.tolist(), centroid_m=vertices.mean(axis=0).tolist(),
        lowest_z_m=float(vertices[:, 2].min()))


def sample(p):
    s = {'native_time_s': 0., 'control_valid': True, 'fixtures_valid': True, 'geometry_valid': True,
        'grasp': {'available': True, 'held_names': [], 'released': True},
        'anchor_centroid_m': [0., 0., .1],
        'optical_selection': {'available': True, 'accepted': True, 'selected_candidate': 'c0',
            'appearance_mode': p['appearance_mode'], 'frozen_eye_m': [-1., 0., 1.5],
            'native_geometric_diagnostics': {'ray_intersections': [[1., 2., 3.]]}},
        'targets': {}}
    positions = [.8, .2] if p.get('relation') == 'selected_nearer' else [.2, .8]
    for (key, name), x in zip(p['candidates'].items(), positions):
        s['targets'][key] = {'object_name': name, 'dimensions_m': [.06, .06, .12],
            'quaternion_xyzw': [0., 0., 0., 1.], 'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.,
            'dynamic_collidable': True, 'robot_contact': False,
            'destinations': {pad: {field: pad == 'table' for field in ('touching', 'on_top', 'footprint_inside')}
                             for pad in ('blue', 'yellow', 'table')}}
        move(s, key, [x, 0., .06])
    return s


def start(p=None):
    p = p or parameters()
    s, progress = sample(p), MirrorProgress(p)
    progress.begin(deepcopy(s))
    return progress, s


def tick(progress, s, delta=DT):
    s['native_time_s'] += delta
    return progress.update(deepcopy(s))


def dwell(progress, s, count=8):
    for _ in range(count):
        result = tick(progress, s)
    return result


def pick(progress, s, key='c0'):
    row = s['targets'][key]
    s['grasp'].update(held_names=[row['object_name']], released=False)
    row['robot_contact'] = True
    move(s, key, np.array(row['position_m']) + [0., 0., .0625])
    return tick(progress, s)


def put(progress, s, key='c0', pad='blue', xyz=None):
    move(s, key, xyz if xyz is not None else [1.5, -.3 if pad == 'blue' else .3, .06])
    s['targets'][key]['robot_contact'] = False
    for destination, relation in s['targets'][key]['destinations'].items():
        relation.update({field: destination == pad for field in relation})
    s['grasp'].update(held_names=[], released=True)
    return tick(progress, s)


def stages(result):
    return {item['id']: item['completed'] for item in result['completion_items']}


def test_correspondence_requires_actual_correct_pickup_and_released_dwell():
    progress, s = start()
    pick(progress, s)
    put(progress, s)
    assert not progress.result()['task_success']
    result = dwell(progress, s)
    assert result['task_success'] and result['physical_success']
    assert stages(result) == {'selection': True, 'grasp_lift': True, 'stable_placement': True}
    assert result['stable_duration_s'] == .5
    assert not result['optical_visibility_verified']
    assert not result['optical_necessity_verified']
    assert not result['complete_reference_execution_verified']


@pytest.mark.parametrize('mode,pad', [('direct', 'yellow'), ('reflected', 'blue')])
def test_classification_both_measured_appearances_choose_their_native_destination(mode, pad):
    progress, s = start(parameters('classify_appearance_deliver', mode=mode))
    progress.report_answer('ANSWER: ' + mode.upper() + '  ')
    pick(progress, s)
    put(progress, s, pad=pad)
    result = dwell(progress, s)
    assert result['task_success'] and result['report']['frozen_value'] == mode
    assert result['current']['destination'] == pad
    assert list(stages(result)) == DESIGN['completion_items']['classify_appearance_deliver']


@pytest.mark.parametrize('report', [None, 'direct', 'ANSWER: reflected or direct'])
def test_missing_wrong_or_invalid_pregrasp_report_cannot_be_repaired_late(report):
    progress, s = start(parameters('classify_appearance_deliver'))
    if report is not None:
        progress.report_answer(report)
    pick(progress, s)
    progress.report_answer('reflected')
    put(progress, s)
    result = dwell(progress, s)
    assert result['physical_success'] and not result['task_success']
    assert not stages(result)['appearance_report']
    assert result['report']['history'][-1]['after_first_grasp'] is True


def test_latest_valid_pregrasp_report_replaces_prior_but_reads_and_reports_do_not_advance_dwell():
    progress, s = start(parameters('classify_appearance_deliver'))
    progress.report_answer('direct')
    progress.report_answer('Reflected')
    progress.report_answer('reflected extra text')
    pick(progress, s)
    put(progress, s)
    before = progress.result()
    for _ in range(30):
        progress.report_answer('direct')
        progress.result()
    after = progress.result()
    assert after['native_time_s'] == before['native_time_s']
    assert after['stable_duration_s'] == before['stable_duration_s'] == 0.
    assert after['report']['frozen_value'] == 'reflected'
    assert dwell(progress, s)['task_success']


@pytest.mark.parametrize('failure', ['no_op', 'rest_only', 'held', 'contact', 'wrong_pad', 'fast'])
def test_delivery_negative_physical_sequences(failure):
    progress, s = start()
    if failure == 'rest_only':
        put(progress, s)
    elif failure != 'no_op':
        pick(progress, s)
        put(progress, s, pad='yellow' if failure == 'wrong_pad' else 'blue')
        if failure == 'held':
            s['grasp'].update(held_names=['native_0'], released=False)
        elif failure == 'contact':
            s['targets']['c0']['robot_contact'] = True
        elif failure == 'fast':
            s['targets']['c0']['linear_speed_m_s'] = .021
    assert not dwell(progress, s)['physical_success']


@pytest.mark.parametrize('violation', ['wrong_first', 'different_later', 'unselected_move'])
def test_wrong_native_object_or_unselected_motion_latches_invalid(violation):
    progress, s = start()
    if violation == 'wrong_first':
        result = pick(progress, s, 'c1')
    else:
        pick(progress, s)
        if violation == 'different_later':
            result = pick(progress, s, 'c1')
        else:
            move(s, 'c1', [.821, 0., .06])
            result = tick(progress, s)
    assert result['evaluation_validity'] == 'invalid'
    assert not result['task_success'] and result['completion_score'] is None
    put(progress, s)
    assert dwell(progress, s)['evaluation_validity'] == 'invalid'


def test_initial_held_is_not_pickup_and_cannot_freeze_the_pregrasp_report():
    p = parameters('classify_appearance_deliver')
    s = sample(p)
    s['grasp'].update(held_names=['native_0'], released=False)
    progress = MirrorProgress(p)
    progress.begin(s)
    move(s, 'c0', [.2, 0., .2])
    result = tick(progress, s)
    assert not result['per_target']['c0']['grasp_seen'] and not result['report']['frozen']
    progress.report_answer('reflected')
    s['grasp'].update(held_names=[], released=True)
    tick(progress, s)
    pick(progress, s)
    assert progress.result()['per_target']['c0']['lift_seen']
    put(progress, s)
    assert dwell(progress, s)['task_success']


def test_repeated_native_time_cannot_establish_pickup_or_create_dwell():
    progress, s = start()
    s['grasp'].update(held_names=['native_0'], released=False)
    move(s, 'c0', [.2, 0., .2])
    for _ in range(30):
        result = progress.update(deepcopy(s))
    assert not result['per_target']['c0']['grasp_seen']
    tick(progress, s)
    put(progress, s)
    before = progress.result()
    for _ in range(30):
        assert progress.update(deepcopy(s)) == before
    assert not before['task_success']


@pytest.mark.parametrize('missing', ['grasp', 'vertices', 'time', 'robot_contact', 'shape_flag', 'gap'])
def test_missing_measurements_or_gaps_reset_stability_and_recover(missing):
    progress, s = start()
    pick(progress, s)
    put(progress, s)
    dwell(progress, s, 4)
    bad = deepcopy(s)
    bad['native_time_s'] += .25 if missing == 'gap' else DT
    if missing == 'grasp':
        bad['grasp']['available'] = False
    elif missing == 'vertices':
        bad['targets']['c0']['vertices_m'] = None
    elif missing == 'time':
        bad['native_time_s'] = None
    elif missing == 'robot_contact':
        bad['targets']['c0']['robot_contact'] = 1
    elif missing == 'shape_flag':
        bad['geometry_valid'] = None
    result = progress.update(bad)
    assert result['evaluation_validity'] == 'unavailable' and not result['task_success']
    assert result['stable_duration_s'] == 0. and stages(result)['grasp_lift']
    s['native_time_s'] += .25 + DT
    progress.update(s)
    tick(progress, s)
    assert dwell(progress, s)['task_success']


@pytest.mark.parametrize('violation', ['control_valid', 'fixtures_valid', 'geometry_valid', 'dimensions', 'clock'])
def test_explicit_native_constraints_and_backwards_clock_latch_invalid(violation):
    progress, s = start()
    pick(progress, s)
    put(progress, s)
    if violation == 'dimensions':
        s['targets']['c0']['dimensions_m'][0] *= 1.01
    elif violation == 'clock':
        s['native_time_s'] = -1.
    else:
        s[violation] = False
    result = tick(progress, s)
    assert result['evaluation_validity'] == 'invalid' and result['constraint_violations']
    assert not result['physical_success']
    assert stages(result)['grasp_lift']


@pytest.mark.parametrize('failure', ['missing', 'ambiguous', 'mode_mismatch', 'missing_eye'])
def test_initial_optical_failure_is_permanent_and_later_selection_does_not_redefine_truth(failure):
    p, s = parameters(), sample(parameters())
    if failure == 'missing':
        s.pop('optical_selection')
    elif failure == 'ambiguous':
        s['optical_selection']['accepted'] = False
    elif failure == 'mode_mismatch':
        s['optical_selection']['appearance_mode'] = 'direct'
    else:
        s['optical_selection'].pop('frozen_eye_m')
    progress = MirrorProgress(p)
    assert progress.begin(s)['evaluation_validity'] == 'unavailable'
    s['optical_selection'] = sample(p)['optical_selection']
    pick(progress, s)
    put(progress, s)
    assert dwell(progress, s)['evaluation_validity'] == 'unavailable'


@pytest.mark.parametrize('relation', ['selected_farther', 'selected_nearer'])
def test_spatial_single_target_requires_real_native_motion_and_correct_fixed_anchor_relation(relation):
    progress, s = start(parameters('mirror_reconfigure', relation=relation))
    pick(progress, s)
    x = 1.2 if relation == 'selected_farther' else .05
    put(progress, s, pad='table', xyz=[x, 0., .06])
    result = dwell(progress, s)
    assert result['task_success']
    assert stages(result) == {'selection': True, 'grasp_lift': True, 'reconfiguration': True, 'stable_placement': True}
    assert result['per_target']['c0']['xy_displacement_m'] >= .05


@pytest.mark.parametrize('relation', ['selected_farther', 'selected_nearer'])
def test_initially_correct_spatial_start_is_unavailable_and_cannot_be_requalified(relation):
    p = parameters('mirror_reconfigure', relation=relation)
    s = sample(p)
    move(s, 'c0', [1.2 if relation == 'selected_farther' else .05, 0., .06])
    progress = MirrorProgress(p)
    result = progress.begin(s)
    assert result['evaluation_validity'] == 'unavailable'
    assert 'initial_spatial_goal_already_satisfied' in result['initial_measurement_errors']
    pick(progress, s)
    put(progress, s, pad='table', xyz=[1.4 if relation == 'selected_farther' else -.05, 0., .06])
    assert dwell(progress, s)['evaluation_validity'] == 'unavailable'


def test_robot_motion_or_diagnostic_centroid_cannot_redefine_spatial_reference_or_progress():
    progress, s = start(parameters('mirror_reconfigure'))
    frozen = progress.result()['frozen_anchor_centroid_m']
    s['robot_position_m'] = [100., 100., 100.]
    s['targets']['c0']['centroid_m'] = [100., 100., 100.]
    s['optical_selection'].update(selected_candidate='c1', frozen_eye_m=[99., 99., 99.])
    result = dwell(progress, s)
    assert not result['task_success'] and not stages(result)['reconfiguration']
    assert result['selected_candidate'] == 'c0' and result['frozen_anchor_centroid_m'] == frozen
    assert result['per_target']['c0']['xy_displacement_m'] == 0.
    assert result['current']['distances_m']['c0'] == pytest.approx(.2)


@pytest.mark.parametrize('anchor', [None, [.003, 0., .1]])
def test_spatial_anchor_missing_is_unavailable_and_measured_change_invalidates(anchor):
    progress, s = start(parameters('mirror_reconfigure'))
    s['anchor_centroid_m'] = anchor
    result = tick(progress, s)
    assert result['evaluation_validity'] == ('unavailable' if anchor is None else 'invalid')
    assert not result['task_success']


def test_swap_requires_both_real_pickups_motion_and_initial_order_reversal():
    p = parameters('mirror_reconfigure', relation='reverse_initial_distance_order')
    progress, s = start(p)
    assert progress.result()['initial_distance_order'] == ['c0', 'c1']
    pick(progress, s, 'c0')
    put(progress, s, 'c0', pad='table', xyz=[1.2, 0., .06])
    result = dwell(progress, s)
    assert not result['task_success'] and not stages(result)['grasp_lift']
    assert not stages(result)['reconfiguration']  # The second original has not moved.
    pick(progress, s, 'c1')
    put(progress, s, 'c1', pad='table', xyz=[.1, 0., .06])
    result = dwell(progress, s)
    assert result['task_success'] and all(stages(result).values())
    assert all(row['xy_displacement_m'] >= .05 for row in result['per_target'].values())
    move(s, 'c0', [.05, 0., .06])
    assert not tick(progress, s)['task_success']


def test_swap_ambiguous_initial_order_is_unavailable_and_reset_clears_frozen_state():
    p = parameters('mirror_reconfigure', relation='reverse_initial_distance_order')
    s, progress = sample(p), MirrorProgress(p)
    move(s, 'c1', [.21, 0., .06])
    assert progress.begin(s)['evaluation_validity'] == 'unavailable'
    assert 'initial_distance_order_ambiguous' in progress.result()['initial_measurement_errors']
    progress.report_answer('reflected')
    fresh = sample(p)
    result = progress.begin(fresh)
    assert result['evaluation_validity'] == 'complete' and result['report']['history'] == []
    result['frozen_selection']['selected_candidate'] = 'c1'
    result['per_target']['c0']['grasp_seen'] = True
    assert progress.result()['selected_candidate'] == 'c0'
    assert not progress.result()['per_target']['c0']['grasp_seen']


def test_spatial_dwell_requires_the_goal_relation_throughout_the_same_stable_interval():
    progress, s = start(parameters('mirror_reconfigure'))
    pick(progress, s)
    put(progress, s, pad='table', xyz=[.4, 0., .06])
    result = dwell(progress, s)
    assert not result['current']['distance_relation_satisfied'] and result['stable_duration_s'] == 0.
    move(s, 'c0', [1.2, 0., .06])
    result = tick(progress, s)
    assert result['current']['reconfiguration_satisfied']
    assert not result['task_success'] and result['stable_duration_s'] == 0.
    assert dwell(progress, s)['task_success']


def test_json_results_reject_nonfinite_physics_and_remain_detached():
    progress, s = start()
    pick(progress, s)
    put(progress, s)
    s['targets']['c0']['linear_speed_m_s'] = float('nan')
    s['targets']['c1']['vertices_m'][0][0] = float('inf')
    result = tick(progress, s)
    assert result['evaluation_validity'] == 'unavailable'
    json.dumps(result, allow_nan=False)


def test_actual_first_grasp_freezes_report_even_when_unrelated_current_measurement_is_missing():
    progress, s = start(parameters('classify_appearance_deliver'))
    progress.report_answer('direct')
    s['targets']['c1']['robot_contact'] = None
    assert pick(progress, s)['evaluation_validity'] == 'unavailable'
    progress.report_answer('reflected')
    s['targets']['c1']['robot_contact'] = False
    put(progress, s)
    result = dwell(progress, s)
    assert result['physical_success'] and not result['task_success']
    assert result['report']['frozen_value'] == 'direct'


def final_state_parameters(operation='mirror_select_deliver', **kwargs):
    p = parameters(operation, **kwargs)
    p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None,
                        minimum_required_motion_m=None)
    ids = ['selected_original_at_destination', 'stable_placement']
    if operation == 'classify_appearance_deliver':
        ids.insert(0, 'appearance_report')
    elif operation == 'mirror_reconfigure':
        ids.insert(1, 'reconfiguration')
        p.pop('requires_both_objects_moved')
    p['completion_items'] = [{'id': key} for key in ids]
    return p


def test_final_state_delivery_succeeds_without_acquisition_or_lift_and_states_terminal_checks():
    progress, s = start(final_state_parameters())
    put(progress, s)
    result = dwell(progress, s)
    assert result['task_success'] and result['physical_success']
    assert result['schema'] == 'wia_mirror_final_state_progress_v1'
    assert result['manipulation_revision'] == 'final_state_v1'
    assert stages(result) == {'selected_original_at_destination': True, 'stable_placement': True}
    assert all(row['kind'] == 'terminal_condition' for row in result['completion_items'])
    assert not result['per_target']['c0']['grasp_seen'] and not result['per_target']['c0']['lift_seen']
    assert result['committed_candidate'] is None


@pytest.mark.parametrize('mode,pad', [('direct', 'yellow'), ('reflected', 'blue')])
def test_final_state_appearance_uses_latest_valid_report_including_after_delivery(mode, pad):
    progress, s = start(final_state_parameters('classify_appearance_deliver', mode=mode))
    progress.report_answer('direct' if mode == 'reflected' else 'reflected')
    pick(progress, s)
    put(progress, s, pad=pad)
    result = dwell(progress, s)
    assert result['physical_success'] and not result['task_success']
    before = (result['native_time_s'], result['stable_duration_s'])
    result = progress.report_answer('ANSWER: ' + mode)
    assert result['task_success'] and result['report']['latest_valid'] == mode
    assert result['report']['history'][-1]['after_first_grasp']
    assert (result['native_time_s'], result['stable_duration_s']) == before
    assert progress.report_answer('ambiguous invalid text')['task_success']
    assert not progress.report_answer('direct' if mode == 'reflected' else 'reflected')['task_success']


@pytest.mark.parametrize('relation', ['selected_farther', 'selected_nearer', 'reverse_initial_distance_order'])
def test_final_state_spatial_requires_goal_relation_without_pickup_or_arbitrary_motion(relation):
    p = final_state_parameters('mirror_reconfigure', relation=relation)
    progress, s = start(p)
    x = .05 if relation == 'selected_nearer' else 1.2
    put(progress, s, pad='table', xyz=[x, 0., .06])
    result = dwell(progress, s)
    assert result['task_success'] and stages(result)['reconfiguration']
    assert not any(row['grasp_seen'] or row['lift_seen'] for row in result['per_target'].values())
    assert result['current']['required_motion_satisfied'] is None
    if relation == 'reverse_initial_distance_order':
        assert result['per_target']['c1']['xy_displacement_m'] == 0.
    move(s, 'c0', [.8 if relation == 'selected_nearer' else .2, 0., .06])
    result = tick(progress, s)
    assert not result['task_success'] and not stages(result)['reconfiguration']


def test_final_state_spatial_accepts_small_motion_that_really_changes_the_goal_relation():
    p = final_state_parameters('mirror_reconfigure')
    p['distance_gap_m'] = .01
    s, progress = sample(p), MirrorProgress(p)
    move(s, 'c0', [.79, 0., .06])
    progress.begin(s)
    put(progress, s, pad='table', xyz=[.82, 0., .06])
    result = dwell(progress, s)
    assert result['task_success']
    assert result['per_target']['c0']['xy_displacement_m'] == pytest.approx(.03)


def test_final_state_unselected_protection_is_current_and_wrong_grasp_history_can_recover():
    progress, s = start(final_state_parameters())
    initial_unselected = deepcopy(s['targets']['c1'])
    result = pick(progress, s, 'c1')
    assert result['evaluation_validity'] == 'complete'
    assert not result['current']['unselected_objects_preserved']
    put(progress, s)
    assert not dwell(progress, s)['task_success']
    s['targets']['c1'] = initial_unselected
    result = tick(progress, s)
    assert result['current']['unselected_objects_preserved'] and result['stable_duration_s'] == 0.
    result = dwell(progress, s)
    assert result['task_success'] and not result['constraint_violations']
    assert result['per_target']['c1']['grasp_seen']


@pytest.mark.parametrize('failure', ['wrong_original', 'wrong_branch', 'held', 'robot_support', 'moving',
                                    'outside', 'unavailable', 'identity'])
def test_final_state_mirror_terminal_counterexamples(failure):
    progress, s = start(final_state_parameters('classify_appearance_deliver', mode='direct'))
    progress.report_answer('direct')
    put(progress, s, key='c1' if failure == 'wrong_original' else 'c0',
        pad='blue' if failure == 'wrong_branch' else 'yellow')
    row = s['targets']['c0']
    if failure == 'held':
        s['grasp'].update(held_names=['native_0'], released=False)
    elif failure == 'robot_support':
        row['robot_contact'] = True
    elif failure == 'moving':
        row['linear_speed_m_s'] = .021
    elif failure == 'outside':
        row['destinations']['yellow']['footprint_inside'] = False
    elif failure == 'unavailable':
        row['destinations']['yellow']['touching'] = None
    elif failure == 'identity':
        row['object_name'] = 'replacement'
    result = dwell(progress, s)
    assert not result['task_success'] and not result['physical_success']
    if failure in ('unavailable', 'identity'):
        assert result['evaluation_validity'] == 'unavailable' and result['completion_score'] is None
    if failure == 'identity':
        assert not stages(result)['selected_original_at_destination']


@pytest.mark.parametrize('failure', ['clock', 'gap', 'anchor', 'optical', 'dimensions', 'control', 'fixture'])
def test_final_state_mirror_keeps_native_truth_clock_geometry_and_control_guards(failure):
    p = final_state_parameters('mirror_reconfigure')
    s, progress = sample(p), MirrorProgress(p)
    if failure == 'optical':
        s['optical_selection']['accepted'] = False
    progress.begin(s)
    put(progress, s, pad='table', xyz=[1.2, 0., .06])
    dwell(progress, s)
    if failure == 'clock':
        s['native_time_s'] = -1.
    elif failure == 'gap':
        s['native_time_s'] += 1.
    elif failure == 'anchor':
        s['anchor_centroid_m'][0] = .003
    elif failure == 'dimensions':
        s['targets']['c0']['dimensions_m'][0] *= 1.01
    elif failure in ('control', 'fixture'):
        s['control_valid' if failure == 'control' else 'fixtures_valid'] = False
    result = tick(progress, s)
    assert not result['task_success']
    assert result['evaluation_validity'] == ('unavailable' if failure in ('gap', 'optical') else 'invalid')


@pytest.mark.parametrize('change', [
    {'manipulation_revision': 'unknown'}, {'manipulation_revision': None},
    {'minimum_lift_m': .05}, {'minimum_required_motion_m': .05}])
def test_final_state_mirror_requires_explicit_supported_revision_and_null_history_thresholds(change):
    p = final_state_parameters()
    p['scoring'].update(change)
    with pytest.raises(ValueError):
        MirrorProgress(p)

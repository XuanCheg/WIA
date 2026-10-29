"""Validate pure single-asset grasp-probe input and measured-evidence reduction.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_asset_grasp_probe.py
These CPU checks never import OmniGibson, construct an environment, or launch GPU work.
"""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

import wia.runtime.commands.probe_wia_asset_grasp as probe


def design():
    return {'category': 'test_category', 'model': 'main_selected_model', 'scale': [1., 1., 1.],
        'quaternion_xyzw': [0., 0., 0., 1.], 'grasp_point_local_m': [0., .02, .03],
        'source_task_file': '/main/selected/task.yaml', 'source_binding': 'asset', 'case_id': 'cpu_case'}


def released():
    return {'object_name': None, 'constraint_active': False, 'is_grasping': -1,
            'release_pending': False, 'gripper_open': True}


def holding():
    return {'object_name': 'wia_probe_asset', 'constraint_active': True, 'is_grasping': 1,
            'release_pending': False, 'gripper_open': False}


def complete_report():
    return {'target_name': 'wia_probe_asset',
        'initial': {'touching': True, 'robot_contact': False, 'robot_bench_contact': False, 'grasp': released(),
                    'robot_stance': dict.fromkeys(probe.STANCE_FIELDS, True)},
        'commands': [{'action': token, 'receipt': {'completed': True},
                      'measurement': {'grasp': released() if token == 'RELEASE' else holding()}}
                     for token in probe.STAGES],
        'hold_loss': False, 'measured_lift_m': .05, 'measured_lateral_displacement_m': .05,
        'final_hold_steps': 60,
        'final': {'touching': True, 'on_top': True, 'footprint_inside': True,
                  'robot_contact': False, 'grasp': released(),
                  'robot_stance': dict.fromkeys(probe.STANCE_FIELDS, True)}}


def test_design_is_detached_without_guessing_assets_or_mutating_values():
    supplied = design()
    result = probe.validate_design(supplied)
    assert result == supplied
    result['scale'][0] = 2.
    assert supplied['scale'][0] == 1.


def test_optional_preparation_is_explicit_ordered_and_detached():
    value = design()
    value.update(bench_dimensions_m=[.31, .27, .06], initialization_actions=[
        {'action': 'MV_FWD', 'distance_m': .25}, {'action': 'MV_LEFT', 'distance_m': .12}])
    result = probe.validate_design(value)
    assert result == value
    assert [row['action'] for row in result['initialization_actions']] == ['MV_FWD', 'MV_LEFT']
    result['initialization_actions'][0]['distance_m'] = .01
    result['bench_dimensions_m'][2] = .02
    assert value['initialization_actions'][0]['distance_m'] == .25
    assert value['bench_dimensions_m'][2] == .06
    original = design()
    assert probe.validate_design(original) == original
    assert probe.DEFAULT_BENCH_DIMENSIONS_M == (.8, .8, .04)


@pytest.mark.parametrize('action', ['GRASP', 'MV_RIGHT', 'MV_UP', 'BASE_FWD'])
def test_initialization_rejects_unsupported_commands(action):
    value = design()
    value['initialization_actions'] = [{'action': action, 'distance_m': .1}]
    with pytest.raises(ValueError, match='MV_FWD or MV_LEFT'):
        probe.validate_design(value)


@pytest.mark.parametrize('distance', [0., -.1, .251, float('inf'), float('nan'), True, '.1', None])
def test_initialization_distance_requires_bounded_positive_finite_number(distance):
    value = design()
    value['initialization_actions'] = [{'action': 'MV_FWD', 'distance_m': distance}]
    with pytest.raises(ValueError, match='distance_m'):
        probe.validate_design(value)


@pytest.mark.parametrize('actions', [None, {}, [{'action': 'MV_LEFT'}],
    [{'action': 'MV_LEFT', 'distance_m': .1, 'extra': True}]])
def test_initialization_requires_exact_ordered_action_schema(actions):
    value = design()
    value['initialization_actions'] = actions
    with pytest.raises(ValueError):
        probe.validate_design(value)


@pytest.mark.parametrize('dimensions', [[.3, .3], [.3, 0., .04], [.3, -.3, .04],
    [float('nan'), .3, .04], [True, True, True], [True, .3, .04], ['.3', '.3', '.04']])
def test_bench_dimensions_require_three_finite_positive_numbers(dimensions):
    value = design()
    value['bench_dimensions_m'] = dimensions
    with pytest.raises(ValueError, match='bench_dimensions_m'):
        probe.validate_design(value)


@pytest.mark.parametrize('contact', [True, None, 'missing'])
def test_initial_bench_robot_contact_must_be_measured_false(contact):
    report = complete_report()
    if contact == 'missing':
        del report['initial']['robot_bench_contact']
    else:
        report['initial']['robot_bench_contact'] = contact
    result = probe.reduce_success(report)
    assert result['accepted'] is False
    assert 'initial_robot_bench_contact_unavailable_or_rejected' in result['failure_reasons']


@pytest.mark.parametrize('stage', ['initial', 'final'])
@pytest.mark.parametrize('field', probe.STANCE_FIELDS)
@pytest.mark.parametrize('value', [False, None, 'missing'])
def test_supported_upright_collision_free_stance_is_required_at_both_boundaries(stage, field, value):
    report = complete_report()
    stance = report[stage]['robot_stance']
    if value == 'missing':
        del stance[field]
    else:
        stance[field] = value
    assert probe.stance_accepted(stance) is False
    result = probe.reduce_success(report)
    assert result['accepted'] is False
    assert stage + '_robot_stance_unavailable_or_rejected' in result['failure_reasons']


@pytest.mark.parametrize('stage', ['initial', 'final'])
def test_missing_or_exceptional_stance_cannot_pass_as_a_grasp_result(stage):
    report = complete_report()
    report[stage].pop('robot_stance')
    assert probe.reduce_success(report)['accepted'] is False
    report[stage]['robot_stance'] = {**dict.fromkeys(probe.STANCE_FIELDS),
        'measurement_error': 'RuntimeError: native contact readback failed'}
    assert probe.reduce_success(report)['accepted'] is False


@pytest.mark.parametrize('problem', ['missing', 'extra', 'empty_model', 'scale_zero', 'bool_scale',
                                     'bad_quaternion', 'nonnormalized', 'nonfinite_point', 'wrong_shape'])
def test_invalid_or_incomplete_design_is_rejected(problem):
    value = design()
    if problem == 'missing':
        value.pop('source_binding')
    elif problem == 'extra':
        value['guessed_default'] = True
    elif problem == 'empty_model':
        value['model'] = ' '
    elif problem == 'scale_zero':
        value['scale'][1] = 0.
    elif problem == 'bool_scale':
        value['scale'] = [True, True, True]
    elif problem == 'bad_quaternion':
        value['quaternion_xyzw'] = [0., 0., 0., 0.]
    elif problem == 'nonnormalized':
        value['quaternion_xyzw'] = [0., 0., 0., 2.]
    elif problem == 'nonfinite_point':
        value['grasp_point_local_m'][0] = float('nan')
    else:
        value['grasp_point_local_m'] = [0., 1.]
    with pytest.raises(ValueError):
        probe.validate_design(value)


def test_complete_native_evidence_passes_only_the_bench_scope():
    result = probe.reduce_success(complete_report())
    assert result['accepted'] is True
    assert result['failure_reasons'] == []
    assert result['full_task_qualified'] is False and result['model_calls'] == 0


@pytest.mark.parametrize('problem', ['missing_holding', 'false_holding', 'wrong_identity', 'no_constraint',
    'pending_release', 'command_failure', 'hold_loss', 'lift', 'lateral', 'missing_lift',
    'initial_contact', 'initial_telemetry', 'initial_support', 'missing_final_support',
    'final_contact', 'final_attached', 'closed_hand', 'final_unknown_grasp', 'short_hold', 'missing_command'])
def test_missing_or_negative_physical_evidence_never_passes(problem):
    report = complete_report()
    grasp = report['commands'][0]['measurement']['grasp']
    if problem == 'missing_holding':
        grasp.pop('is_grasping')
    elif problem == 'false_holding':
        grasp['is_grasping'] = -1
    elif problem == 'wrong_identity':
        grasp['object_name'] = 'a_different_native_asset'
    elif problem == 'no_constraint':
        grasp['constraint_active'] = False
    elif problem == 'pending_release':
        grasp['release_pending'] = True
    elif problem == 'command_failure':
        report['commands'][1]['receipt']['completed'] = False
    elif problem == 'hold_loss':
        report['hold_loss'] = True
    elif problem == 'lift':
        report['measured_lift_m'] = .049
    elif problem == 'lateral':
        report['measured_lateral_displacement_m'] = .049
    elif problem == 'missing_lift':
        report.pop('measured_lift_m')
    elif problem in ('initial_contact', 'initial_telemetry'):
        report['initial']['robot_contact'] = True if problem == 'initial_contact' else None
    elif problem == 'initial_support':
        report['initial']['touching'] = False
    elif problem == 'missing_final_support':
        report['final'].pop('on_top')
    elif problem == 'final_contact':
        report['final']['robot_contact'] = True
    elif problem == 'final_attached':
        report['final']['grasp']['constraint_active'] = True
    elif problem == 'closed_hand':
        report['final']['grasp']['gripper_open'] = False
    elif problem == 'final_unknown_grasp':
        report['final']['grasp']['is_grasping'] = 0
    elif problem == 'short_hold':
        report['final_hold_steps'] = 59
    else:
        report['commands'].pop()
    assert probe.reduce_success(report)['accepted'] is False


def test_main_preserves_exact_input_bytes_and_existing_evidence(tmp_path, monkeypatch):
    path, output = tmp_path / 'design.json', tmp_path / 'result'
    raw = json.dumps(design(), indent=4).encode() + b'\n\n'
    path.write_bytes(raw)
    calls = []

    def fake_run(value, directory):
        calls.append((deepcopy(value), directory))
        return {'accepted': False}

    monkeypatch.setattr(probe, 'run', fake_run)
    assert probe.main(['--design-file', str(path), '--output-dir', str(output)]) == 1
    assert (output / 'input_design.json').read_bytes() == raw
    with pytest.raises(ValueError, match='fresh output'):
        probe.main(['--design-file', str(path), '--output-dir', str(output)])
    assert len(calls) == 1


def test_importing_probe_does_not_import_simulator():
    completed = subprocess.run([sys.executable, '-B', '-c',
        "import sys; import wia.runtime.commands.probe_wia_asset_grasp; assert 'omnigibson' not in sys.modules"],
        cwd=probe.ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_initialization_tokens_use_existing_native_arm_interface():
    from core.action_units import MOVE_ATOMS
    assert {'MV_FWD', 'MV_LEFT'}.issubset(MOVE_ATOMS)

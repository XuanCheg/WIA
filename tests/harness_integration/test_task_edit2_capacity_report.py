"""Check source-capacity report truth, provenance and fixed-denominator examples.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_capacity_report.py
Uses completed source diagnostic artifacts read-only; launches no GPU or models.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

import wia.runtime.commands.report_task_edit2_capacity_calibration as report

BASE = report.ROOT / 'outputs/task_edit2_20261018'
AB = BASE / 'native_capacity_isolated_ab_ac_v4'
REMAINING = BASE / 'native_capacity_isolated_remaining_v1'
FINAL = REMAINING / 'calibration_receipt.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture(scope='module')
def complete():
    return report.build_data(FINAL, [AB, REMAINING])


def test_actual_complete_truth_13_trials_origins_resets_and_oracle_denominator(complete):
    data = complete
    assert data['truth_validation']['accepted'] is True
    assert data['pair_denominator'] == data['truth_validation']['pair_denominator'] == 15
    assert data['validated_order'] == 'F < E < D < C < B < A'
    assert data['source_only_pairs'] == ['CD', 'EF']
    assert data['fresh_native_trial_count'] == 13
    assert data['fresh_AB_control_trial_count'] == 1
    assert data['fresh_previously_unknown_trial_count'] == 12
    assert data['all_recorded_pair_resets_verified'] is True
    assert len(data['source_comparisons']) == 4 and len(data['native_comparisons']) == 13
    markers = {row['first'] + row['second']: row['origin_marker'] for row in data['pair_relations']}
    assert markers['AB'] == 'S+N' and markers['CD'] == markers['EF'] == 'S'
    assert sum(marker == 'N' for marker in markers.values()) == 12
    expected = {'correct_chain': (15, 100), 'reverse_chain': (0, 0), 'missing_A': (10, 1000/15),
                'swap_F_E': (14, 1400/15), 'empty': (0, 0)}
    for example in data['scorer_examples']:
        correct, accuracy = expected[example['name']]
        assert example['total_pairs'] == 15
        assert example['correct_pairwise_relations'] == correct
        assert example['comparison_accuracy_percent'] == pytest.approx(accuracy)
        assert example['kind'] == 'SCORER_example_NOT_Agent_result'


def test_actual_partial_run_keeps_15_and_all_scorer_examples_unavailable():
    data = report.build_data(AB / 'calibration_receipt.json', [AB])
    assert data['truth_validation']['accepted'] is False
    assert data['validated_order'] is None and data['pair_denominator'] == 15
    assert data['fresh_native_trial_count'] == 2
    assert all(example['comparison_accuracy_percent'] is None and example['total_pairs'] == 15
               for example in data['scorer_examples'])
    assert next(row for row in data['pair_relations'] if row['first'] == 'D' and row['second'] == 'F')['relation'] is None


def test_invalid_asset_provenance_invalidates_score_even_when_graph_edges_exist(tmp_path):
    receipt = report.load(FINAL)
    receipt['inventory'][0]['scale'] = [2., 1., 1.]
    data = report.build_data(write(tmp_path / 'invalid.json', receipt), [])
    assert data['truth_validation']['invalid_evidence']
    assert not data['truth_validation']['accepted']
    assert all(example['total_pairs'] == 15 and example['comparison_accuracy_percent'] is None
               for example in data['scorer_examples'])


def test_cross_trial_count_magnitude_never_defines_order(tmp_path):
    receipt = report.load(FINAL)
    for index, comparison in enumerate(receipt['comparisons']):
        if comparison['origin'] == 'native_common_protocol':
            assert comparison['relation'] == '>'
            comparison['counts'] = {comparison['first']: index * 2 + 2, comparison['second']: index * 2 + 1}
    data = report.build_data(write(tmp_path / 'counts.json', receipt), [])
    assert data['truth_validation']['accepted']
    assert data['validated_order'] == 'F < E < D < C < B < A'
    assert data['cross_trial_particle_counts_comparable'] is False


def test_run_provenance_and_event_count_conflicts_cannot_be_silently_mixed(tmp_path):
    receipt = report.load(FINAL)
    wrong = deepcopy(receipt)
    wrong['inventory_id'] = 'another_inventory'
    directory = tmp_path / 'wrong_run'
    write(directory / 'calibration_receipt.json', wrong)
    with pytest.raises(ValueError, match='inventory_id'):
        report.build_data(FINAL, [directory])
    write(directory / 'calibration_receipt.json', receipt)
    (directory / 'events.jsonl').write_text(json.dumps({'stage': 'source_fill_measured', 'pair': ['A', 'B'],
        'retained_particle_counts': [1, 2], 'wall_duration_s': 1, 'simulated_duration_s': 1}) + '\n')
    with pytest.raises(ValueError, match='counts do not match'):
        report.build_data(FINAL, [directory])
    with pytest.raises(ValueError, match='duplicate physical trial'):
        report.build_data(FINAL, [AB, AB])


def test_missing_duration_proofs_and_runtime_stay_missing(tmp_path):
    directory = tmp_path / 'partial_log'
    write(directory / 'calibration_receipt.json', report.load(FINAL))
    event = next(json.loads(line) for line in (AB / 'events.jsonl').read_text().splitlines()
                 if json.loads(line).get('stage') == 'source_fill_measured')
    event.pop('wall_duration_s')
    event.pop('simulated_duration_s')
    (directory / 'events.jsonl').write_text(json.dumps(event) + '\n')
    data = report.build_data(FINAL, [directory])
    assert data['trials'][0]['wall_duration_s'] is data['trials'][0]['simulated_duration_s'] is None
    assert data['timing']['fill_wall_duration_sum_s'] is None
    assert data['timing']['fill_simulated_duration_sum_s'] is None
    assert data['timing']['load_duration_s'] is data['timing']['phase_duration_s'] is data['timing']['model_duration_s'] is None
    assert data['runs'][0]['runtime']['value'] is None
    assert data['all_recorded_pair_resets_verified'] is False


def test_no_benchmark_or_agent_promotion_and_exact_frozen_identities(complete):
    assert complete['benchmark_tabletop_qualified'] is False
    assert complete['public_label_and_robot_access_qualified'] is False
    assert complete['autonomous_agent_performance'] is False and complete['observed_model_calls'] == 0
    assert 'Robot-free source calibration does not qualify' in complete['limitation']
    assert complete['current_source_protocol']['current_bytes_match']
    assert all(row['current_bytes_match'] for row in complete['source_mappings'])
    for run in complete['runs']:
        assert run['baseline']['sha256'] == report.file_identity(Path(run['run_dir']) / 'baseline.json')['sha256']
        assert run['driver']['native_reset_id']
        assert run['diagnostic_context']['robot_absent'] is True
        assert run['diagnostic_context']['support_table_or_label_patches_applied'] is False
        assert run['final_reset_proof']['value']['accepted'] is True
    assert complete['runs'][1]['runtime']['value'] is None
    for trial in complete['trials']:
        raw = json.loads(Path(trial['event_file']).read_text().splitlines()[trial['event_line'] - 1])
        assert raw == trial['event']
        assert trial['wall_duration_s'] == raw['wall_duration_s']
        assert trial['simulated_duration_s'] == raw['simulated_duration_s']
        assert trial['initialization']['value']['pair_objects']


def test_standalone_artifacts_and_read_only_inputs(tmp_path):
    input_paths = [FINAL, AB / 'events.jsonl', REMAINING / 'events.jsonl']
    before = {path: path.read_bytes() for path in input_paths}
    data = report.write_report(FINAL, [AB, REMAINING], tmp_path / 'report')
    for path in input_paths:
        assert path.read_bytes() == before[path]
    directory = tmp_path / 'report'
    assert {path.name for path in directory.iterdir()} == {
        'report.json', 'report.md', 'relation_matrix.png', 'relation_matrix.svg', 'trial_timing.png', 'trial_timing.svg'}
    assert (directory / 'relation_matrix.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (directory / 'trial_timing.svg').read_text()
    prose = (directory / 'report.md').read_text()
    assert 'SCORER examples — NOT Agent results' in prose
    assert '66.6667' in prose and '93.3333' in prose
    assert 'Source-only relations: CD, EF.' in prose
    assert report.load(directory / 'report.json')['truth_validation'] == data['truth_validation']


def test_inventory_pool_cannot_shrink_denominator(tmp_path):
    receipt = report.load(FINAL)
    receipt['inventory'] = receipt['inventory'][1:]
    with pytest.raises(ValueError, match='fixed 15-pair denominator'):
        report.build_data(write(tmp_path / 'missing_inventory.json', receipt), [])

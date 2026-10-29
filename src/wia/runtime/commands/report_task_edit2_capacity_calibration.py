#!/usr/bin/env python3
"""Report source-only capacity calibration without promoting benchmark validity.

Run: python -B src/wia/runtime/commands/report_task_edit2_capacity_calibration.py --receipt outputs/task_edit2_20261018/native_capacity_isolated_remaining_v1/calibration_receipt.json --run-dir outputs/task_edit2_20261018/native_capacity_isolated_ab_ac_v4 --run-dir outputs/task_edit2_20261018/native_capacity_isolated_remaining_v1 --output-dir outputs/task_edit2_20261018/capacity_report
Read-only inputs produce English JSON/Markdown and standalone matplotlib PNG/SVG.
No simulator, model, Review publisher or calibration procedure is executed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.authoring.converters.liquid_capacity_inventory import validate_truth, SOURCE_PROTOCOL
from wia.authoring.converters.composition_manipulation import capacity_ranking_score

IDS = tuple('ABCDEF')
EXAMPLES = {'correct_chain': 'F < E < D < C < B < A', 'reverse_chain': 'A < B < C < D < E < F',
    'missing_A': 'F < E < D < C < B', 'swap_F_E': 'E < F < D < C < B < A', 'empty': ''}
LIMITATION = ('Robot-free source calibration does not qualify the benchmark tabletop, public labels, '
    'robot access, autonomous execution or Agent performance. Particle counts are comparable only '
    'within one fill_both trial; no calibrated litres or cross-trial volume scale is established.')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else {
        'path': str(path), 'sha256': None, 'status': 'unavailable'}


def optional_json(path):
    path = Path(path)
    return {'file': file_identity(path), 'value': load(path) if path.is_file() else None}


def _duration(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def relation_rows(receipt, validation):
    """Normalize validated same-trial edges; never compare counts across trials."""
    rows = []
    invalid_pairs = {frozenset((row['first'], row['second'])) for row in validation['invalid_evidence']}
    for first, second in combinations(IDS, 2):
        evidence = []
        for comparison in receipt.get('comparisons', []):
            if {comparison.get('first'), comparison.get('second')} != {first, second}:
                continue
            if frozenset((first, second)) in invalid_pairs:
                continue
            single = deepcopy(receipt)
            single['comparisons'] = [comparison]
            checked = validate_truth(single)
            if checked['invalid_evidence']:
                continue
            relation = comparison['relation']
            if comparison['first'] != first and relation != '=':
                relation = '<' if relation == '>' else '>'
            evidence.append({'relation': relation, 'origin': comparison['origin'],
                'counts_same_trial_only': deepcopy(comparison['counts']),
                'source_identity': comparison.get('source_identity'),
                'source_sha256': comparison.get('source_sha256')})
        relations = {row['relation'] for row in evidence}
        relation = next(iter(relations)) if len(relations) == 1 else None
        origins = sorted({row['origin'] for row in evidence})
        marker = ('S+N' if len(origins) == 2 else 'S' if origins == ['source_same_trial'] else
                  'N' if origins == ['native_common_protocol'] else '?')
        rows.append({'first': first, 'second': second, 'relation': relation,
            'origin_marker': marker if relation is not None else '?', 'origins': origins,
            'same_trial_evidence': evidence, 'full_truth_validated': validation['accepted']})
    return rows


def read_run(directory, receipt):
    directory = Path(directory).resolve()
    run_receipt = load(directory / 'calibration_receipt.json')
    for key in ('inventory_id', 'inventory', 'source_mappings', 'source_protocol_sha256'):
        if run_receipt.get(key) != receipt.get(key):
            raise ValueError('Run/final receipt identity differs: ' + str(directory) + ':' + key)
    events_path = directory / 'events.jsonl'
    events = [(number, json.loads(line)) for number, line in enumerate(events_path.read_text().splitlines(), 1) if line.strip()]
    trials = []
    for line, event in events:
        if event.get('stage') != 'source_fill_measured':
            continue
        pair, counts = event.get('pair'), event.get('retained_particle_counts')
        if (not isinstance(pair, list) or len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= set(IDS)
                or not isinstance(counts, list) or len(counts) != 2 or any(type(n) is not int or n <= 0 for n in counts)):
            raise ValueError('Malformed explicit source_fill_measured event.')
        measured = dict(zip(pair, counts))
        matched = [row for row in receipt['comparisons'] if row.get('origin') == 'native_common_protocol'
                   and {row.get('first'), row.get('second')} == set(pair) and row.get('counts') == measured]
        if not matched:
            raise ValueError('Event counts do not match any native same-trial receipt edge: ' + ''.join(pair))
        name = ''.join(pair)
        initialization = optional_json(directory / f'pair_{name}_initialization.json')
        postcheck = optional_json(directory / f'pair_{name}_postcheck.json')
        pair_receipt = optional_json(directory / f'pair_{name}_receipt.json')
        pair_value = pair_receipt['value'] or {}
        # Keep exact receipts and their absence separate from truth validation.
        reset_value = pair_value.get('post_calibration_inventory_reset', {})
        reset_checks = {'initial_baseline_reset_verified': (initialization['value'] or {}).get('checks', {}).get('baseline_reset_verified'),
            'isolated_reset_verified': matched[0].get('isolated_reset_verified'),
            **{key: reset_value.get(key) for key in ('accepted', 'all_inventory_restored', 'agent_history_empty')}}
        trials.append({'run_dir': str(directory), 'pair': pair, 'role': 'source_AB_control' if set(pair) == {'A', 'B'} else 'previously_unknown_pair',
            'event_file': str(events_path), 'event_line': line, 'event': deepcopy(event),
            'retained_particle_counts_same_trial_only': measured,
            'wall_duration_s': _duration(event.get('wall_duration_s')),
            'simulated_duration_s': _duration(event.get('simulated_duration_s')),
            'counts_match_native_receipt': True, 'initialization': initialization, 'postcheck': postcheck,
            'pair_receipt': {'file': pair_receipt['file'], 'inventory_id': pair_value.get('inventory_id'),
                'source_protocol_sha256': pair_value.get('source_protocol_sha256'),
                'driver': deepcopy(pair_value.get('driver')), 'reset': deepcopy(reset_value)},
            'reset_checks': reset_checks,
            'all_recorded_reset_checks_true': all(value is True for value in reset_checks.values())})
    provenance = {'run_dir': str(directory), 'events': file_identity(events_path),
        'calibration_receipt': file_identity(directory / 'calibration_receipt.json'),
        'run_truth_validation': validate_truth(run_receipt),
        'driver': deepcopy(run_receipt.get('driver')), 'diagnostic_context': deepcopy(run_receipt.get('diagnostic_context')),
        'baseline': file_identity(directory / 'baseline.json'),
        'runtime': optional_json(directory / 'runtime.json'), 'result': optional_json(directory / 'result.json'),
        'reset_proof': optional_json(directory / 'reset_proof.json'),
        'final_reset_proof': optional_json(directory / 'final_reset_proof.json'),
        'load_duration_s': None, 'model_duration_s': None,
        'timing_limitation': 'Only explicit source_fill_measured wall/sim durations are aggregated; loading, phase and model timings are not inferred from elapsed clocks.'}
    return provenance, trials


def build_data(receipt_path, run_dirs):
    receipt_path = Path(receipt_path).resolve()
    receipt = load(receipt_path)
    if len(receipt['inventory']) != 6 or {row['id'] for row in receipt['inventory']} != set(IDS):
        raise ValueError('This report requires the exact six-container A-F inventory and fixed 15-pair denominator.')
    validation = validate_truth(receipt)
    truth = validation['valid_comparisons'] if validation['accepted'] else []
    examples = [{'name': name, 'kind': 'SCORER_example_NOT_Agent_result', 'answer': answer,
        **capacity_ranking_score(IDS, truth, answer)} for name, answer in EXAMPLES.items()]
    directories = [Path(path).resolve() for path in run_dirs]
    if len(set(directories)) != len(directories):
        raise ValueError('Repeated run directories would duplicate physical trial evidence.')
    runs, trials = [], []
    for directory in directories:
        provenance, measured = read_run(directory, receipt)
        runs.append(provenance)
        trials.extend(measured)
    pairs = relation_rows(receipt, validation)
    source_only = [''.join((row['first'], row['second'])) for row in pairs if row['origins'] == ['source_same_trial']]
    wall, simulated = [row['wall_duration_s'] for row in trials], [row['simulated_duration_s'] for row in trials]
    model_calls = [(row.get('driver') or {}).get('model_calls') for row in runs]
    source_files = []
    for source in receipt.get('source_mappings', []):
        actual = file_identity(WIA_ESI_ROOT / 'dataset/json_clean' / source['source_identity'])
        source_files.append({**deepcopy(source), 'current_file': actual,
                             'current_bytes_match': actual['sha256'] == source['source_sha256']})
    groups = validation.get('ordered_tie_groups') if validation['accepted'] else None
    return {'schema': 'task_edit2_source_capacity_calibration_report_v1',
        'report_implementation': file_identity(__file__),
        'validator_implementation': file_identity(ROOT / 'src/wia/converters/liquid_capacity_inventory.py'),
        'scorer_implementation': file_identity(ROOT / 'src/wia/converters/composition_manipulation.py'),
        'report_type': 'robot_free_source_calibration_diagnostic', 'receipt_file': file_identity(receipt_path),
        'truth_validation': validation, 'pair_denominator': 15,
        'validated_order': ' < '.join(' = '.join(group) for group in groups) if groups else None,
        'inventory_id': receipt['inventory_id'], 'inventory': deepcopy(receipt['inventory']),
        'source_mappings': source_files, 'source_protocol_sha256': receipt['source_protocol_sha256'],
        'current_source_protocol': {**file_identity(SOURCE_PROTOCOL),
            'current_bytes_match': file_identity(SOURCE_PROTOCOL)['sha256'] == receipt['source_protocol_sha256']},
        'water_settings': deepcopy(receipt.get('calibration_recipe', {}).get('water_settings')),
        'tie_rule': receipt.get('tie_rule'), 'cross_trial_particle_counts_comparable': False,
        'source_comparisons': [deepcopy(row) for row in receipt['comparisons'] if row.get('origin') == 'source_same_trial'],
        'native_comparisons': [deepcopy(row) for row in receipt['comparisons'] if row.get('origin') == 'native_common_protocol'],
        'pair_relations': pairs, 'source_only_pairs': source_only,
        'fresh_native_trial_count': len(trials),
        'fresh_AB_control_trial_count': sum(row['role'] == 'source_AB_control' for row in trials),
        'fresh_previously_unknown_trial_count': sum(row['role'] == 'previously_unknown_pair' for row in trials),
        'runs': runs, 'trials': trials,
        'timing': {'scope': 'Explicit source_fill_measured stages only; not end-to-end job time.',
            'fill_wall_duration_sum_s': sum(wall) if wall and all(value is not None for value in wall) else None,
            'fill_simulated_duration_sum_s': sum(simulated) if simulated and all(value is not None for value in simulated) else None,
            'wall_duration_available_trials': sum(value is not None for value in wall),
            'simulated_duration_available_trials': sum(value is not None for value in simulated),
            'load_duration_s': None, 'phase_duration_s': None, 'model_duration_s': None},
        'all_recorded_pair_resets_verified': bool(trials) and all(row['all_recorded_reset_checks_true'] for row in trials),
        'final_receipt_inventory_reset': deepcopy(receipt.get('post_calibration_inventory_reset')),
        'scorer_examples': examples,
        'observed_model_calls': sum(model_calls) if model_calls and all(type(n) is int for n in model_calls) else None,
        'autonomous_agent_performance': False, 'benchmark_tabletop_qualified': False,
        'public_label_and_robot_access_qualified': False, 'limitation': LIMITATION}


def markdown(data):
    text = ['# Source capacity calibration diagnostic', '',
        f"Validated source order: **{data['validated_order'] or 'unavailable'}**. Required pair denominator: **15**.", '',
        data['limitation'], '',
        f"The supplied logs contain {data['fresh_native_trial_count']} fresh native fill trials: "
        f"{data['fresh_AB_control_trial_count']} AB source control and {data['fresh_previously_unknown_trial_count']} previously unknown pairs. "
        f"Source-only relations: {', '.join(data['source_only_pairs']) or 'none'}. AB also retains both original source records.", '',
        f"Truth validation accepted: `{data['truth_validation']['accepted']}`. "
        f"All recorded pair reset checks true: `{data['all_recorded_pair_resets_verified']}`. "
        f"Observed model calls: `{data['observed_model_calls']}`; no autonomous Agent results are reported.", '',
        '## Fixed-denominator SCORER examples — NOT Agent results', '',
        '| Example | Submitted chain | Correct pairs / 15 | Pair accuracy (%) |', '|---|---|---:|---:|']
    for example in data['scorer_examples']:
        score = example['comparison_accuracy_percent']
        text.append(f"| {example['name']} | `{example['answer'] or '(empty)'}` | "
            f"{example['correct_pairwise_relations']}/15 | {score:.4f} |" if score is not None else
            f"| {example['name']} | `{example['answer'] or '(empty)'}` | unavailable / 15 | unavailable |")
    text += ['', 'Unknown or invalid truth makes every comparable score unavailable; its denominator remains 15.', '',
        '## Original source same-trial evidence', '',
        'CD and EF remain source-only. AB has two preserved source records and one fresh native control; '
        'none of these source counts is compared against another trial.', '',
        '| Source | Relation | Counts within that source trial | Source SHA256 |', '|---|---|---|---|']
    for comparison in data['source_comparisons']:
        text.append(f"| {Path(comparison['source_identity']).name} | {comparison['first']} {comparison['relation']} {comparison['second']} | "
            + ', '.join(f'{key}: {value}' for key, value in comparison['counts'].items())
            + f" | `{comparison['source_sha256']}` |")
    text += ['',
        '## Same-trial native counts and measured fill timing', '',
        'Counts in each row belong only to that pair trial. Do not compare counts between rows or interpret them as litres.', '',
        '| Pair | Role | Same-trial retained counts | Fill wall (s) | Fill simulated (s) |', '|---|---|---|---:|---:|']
    for trial in data['trials']:
        wall, sim = trial['wall_duration_s'], trial['simulated_duration_s']
        text.append('| ' + ''.join(trial['pair']) + ' | ' + trial['role'] + ' | ' +
            ', '.join(f'{key}: {value}' for key, value in trial['retained_particle_counts_same_trial_only'].items()) + ' | ' +
            (f'{wall:.3f}' if wall is not None else 'unavailable') + ' | ' + (f'{sim:.3f}' if sim is not None else 'unavailable') + ' |')
    text += ['', data['timing']['scope'],
        f"Sum of recorded fill wall times: {data['timing']['fill_wall_duration_sum_s']} s; "
        f"sum of recorded fill simulated times: {data['timing']['fill_simulated_duration_sum_s']} s. "
        'Loading, phase and model timings are unavailable; no missing timing is fabricated.', '',
        '## Frozen provenance', '', f"Inventory ID: `{data['inventory_id']}`.",
        f"Receipt: `{data['receipt_file']['path']}`; SHA256 `{data['receipt_file']['sha256']}`.",
        f"Source fill protocol SHA256: `{data['source_protocol_sha256']}`.", '',
        '| ID | Category | Native model | Preserved scale |', '|---|---|---|---|']
    text += [f"| {row['id']} | {row['category']} | {row['model']} | {row['scale']} |" for row in data['inventory']]
    text += ['', 'Common native water settings (verbatim receipt):', '', '```json',
             json.dumps(data['water_settings'], indent=2), '```', '', 'Native reset and frozen baseline identities:', '']
    for run in data['runs']:
        text += [f"- `{run['run_dir']}`: native reset `{(run.get('driver') or {}).get('native_reset_id')}`; "
                 f"baseline SHA256 `{run['baseline']['sha256']}`; calibration receipt SHA256 `{run['calibration_receipt']['sha256']}`."]
    text += ['', 'The JSON report preserves all source hashes/mappings, water settings, per-run frozen driver/reset identities, '
        'native inventory poses, baseline/runtime file identities, and explicit pair initialization/postcheck receipts.', '',
        '![Validated pair relations](relation_matrix.png)', '', 'S: source only; N: fresh native; S+N: both. Each cell compares its row to its column.', '',
        '![Measured source fill durations](trial_timing.png)', '',
        'Standalone vector figures: [relation matrix](relation_matrix.svg), [trial timing](trial_timing.svg).', '']
    return '\n'.join(text)


def plots(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    matrix, labels = np.zeros((6, 6)), [['—' if i == j else '?' for j in range(6)] for i in range(6)]
    for row in data['pair_relations']:
        i, j = IDS.index(row['first']), IDS.index(row['second'])
        relation = row['relation']
        if relation is not None:
            matrix[i, j] = {'<': -1, '=': 0, '>': 1}[relation]
            matrix[j, i] = -matrix[i, j]
            labels[i][j] = relation + '\n' + row['origin_marker']
            labels[j][i] = ('>' if relation == '<' else '<' if relation == '>' else '=') + '\n' + row['origin_marker']
    fig, ax = plt.subplots(figsize=(7.3, 7.1))
    ax.imshow(matrix, cmap=ListedColormap(['#d9e9f6', '#f5f5f5', '#f6dfc9']), vmin=-1, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, labels[i][j], ha='center', va='center', fontsize=12)
    ax.set(xticks=range(6), yticks=range(6), xticklabels=IDS, yticklabels=IDS,
        xlabel='Column container', ylabel='Row container', title='Source capacity relations: row relative to column')
    fig.text(.5, .025, 'S: source only   N: fresh native   S+N: both\nRobot-free calibration; not benchmark or Agent performance.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, 1))
    for suffix in ('png', 'svg'):
        fig.savefig(output / ('relation_matrix.' + suffix), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 5.1))
    positions = np.arange(len(data['trials']))
    for offset, field, name, color in [(-.19, 'wall_duration_s', 'Measured fill wall time', '#2f668a'),
                                     (.19, 'simulated_duration_s', 'Measured fill simulated time', '#bc752c')]:
        for index, trial in enumerate(data['trials']):
            value = trial[field]
            if value is not None:
                ax.bar(index + offset, value, width=.36, color=color, label=name if index == 0 else None)
            else:
                ax.text(index + offset, 0, 'N/A', rotation=90, ha='center', va='bottom', fontsize=8)
    if data['trials']:
        ax.legend()
    else:
        ax.text(.5, .5, 'No measured fill events supplied', transform=ax.transAxes, ha='center')
    ax.set(xticks=positions, xticklabels=[''.join(row['pair']) + (' control' if row['role'] == 'source_AB_control' else '') for row in data['trials']],
        ylabel='Recorded source-fill duration (seconds)', xlabel='Independent same-trial pair',
        title='Native source-fill timing only; loading and model timing unavailable')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(output / ('trial_timing.' + suffix), dpi=180)
    plt.close(fig)


def write_report(receipt, run_dirs, output):
    data = build_data(receipt, run_dirs)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plots(data, output)
    (output / 'report.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    content = markdown(data)
    for filename in ('relation_matrix.png', 'trial_timing.png', 'relation_matrix.svg', 'trial_timing.svg'):
        content = content.replace('](' + filename + ')', '](' + str(output / filename) + ')')
    (output / 'report.md').write_text(content)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, action='append', default=[])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = write_report(args.receipt, args.run_dir, args.output_dir)
    print(json.dumps({'truth_accepted': data['truth_validation']['accepted'], 'order': data['validated_order'],
        'pair_denominator': data['pair_denominator'], 'fresh_native_trials': data['fresh_native_trial_count'],
        'output_dir': str(args.output_dir.resolve())}))


if __name__ == '__main__':
    main()

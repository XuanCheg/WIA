#!/usr/bin/env python3
"""Map Milestone 8 acceptance criteria to saved native evidence, offline only."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
from pathlib import Path


ROOT = WIA_RUNTIME_ROOT


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def audit(root):
    qualifications = []
    for path in sorted(root.glob('*/qualification_summary.json')):
        data = read(path)
        config = read(path.parent/'scene.json')
        if data.get('qualification_success') and config is not None:
            qualifications.append((path, data, config))
    configurations = {name: read(ROOT/'configs/esi_explore'/f'{name}.json')
                      for name in ('visible_target_control', 'single_table_occlusion', 'multi_region_search')}
    matched = {name: [(p, d) for p, d, c in qualifications if c == config]
               for name, config in configurations.items()}
    episodes = [(p, read(p)) for p in sorted(root.glob('*/episode/summary.json'))]
    completed = [(p, d) for p, d in episodes if d['termination_reason'] in
                 ('environment_success', 'model_done', 'max_steps', 'invalid_output', 'execution_error') and not d.get('error')]
    formal = [(p, d) for p, d in completed if not p.parent.parent.name.startswith('A-control')]
    contract = read(root/'rollout_contract_audit.json', {})
    contract_rows = {r['run']: r for r in contract.get('records', [])}
    audited = lambda p: contract_rows.get(p.parent.parent.name, {}).get('passed', False)
    worlds = read(root/'world_policy_verification.json', {})
    transport = read(root/'model_notes_transport_verification.json', {})
    results = read(ROOT/'docs/milestone8_results/results.json', {})
    records = []
    def add(number, criterion, passed, evidence, detail):
        records.append({'criterion': number, 'description': criterion, 'passed': bool(passed),
                        'evidence': [str(p) for p in evidence], 'detail': detail})

    control = matched['visible_target_control']
    single = matched['single_table_occlusion']
    multi = matched['multi_region_search']
    qualified_paths = lambda items: [p for p, _ in items]
    zero = lambda items: bool(items) and all(d['initial_visibility']['all_views_zero_pixels'] and
                                            len(d['initial_visibility']['views']) == 8 for _, d in items)
    native = lambda items: bool(items) and all(d.get('unintended_contact_count',
                    read(p.parent/'collision_audit.json', {}).get('unintended_contacts')) == 0 and
                    not d.get('spawn_external_contacts') and len(d['contact_reporting_links']) >= 36 and
                    d['revealing_visibility']['any_recognizably_visible'] and
                    d['evaluation']['task_success'] and d['stable_hold_0_5_sim_s'] for p, d in items)
    add(1, 'Visible misaligned control is feasible and reproducible', native(control), qualified_paths(control),
        'Require fresh-process qualification with the final configuration; development model success is separate.')
    add(2, 'Valid single-table occlusion scene', native(single), qualified_paths(single),
        'Real native fixed opaque collision geometry; all R1 contact-reporting links checked throughout.')
    add(3, 'Target initially hidden in all enabled Agent RGB', zero(single), qualified_paths(single),
        'Zero target pixels in each of eight views; recognizability is recorded separately.')
    add(4, 'Reachable revealing viewpoint', native(single), qualified_paths(single),
        'Offline qualification route uses the existing bounded executor and is not supplied to the model.')
    add(5, 'Approach, grasp, and lift designated target after reveal', native(single), qualified_paths(single),
        'Historical designated-target attachment and >=0.05 m bottom-AABB lift; extra 0.5-simulation-second holding is separate.')
    regions = {read(p.parent/'private_scene/setup.json', {}).get('selected_region') for p, _ in multi}
    add(6, 'At least two valid candidate regions in one room', native(multi) and zero(multi) and len(regions) >= 2 and
        len(configurations['multi_region_search']['target_placement_candidates']) == 2,
        qualified_paths(multi), 'Both regions pass native qualification under the same final configuration in one room.')
    multi_formal = [(p, d) for p, d in formal if read(p.parent.parent/'scene.json', {}).get('variant') == 'multi_region_search']
    add(7, 'Seed-selected private target region', len(regions) >= 2 and bool(multi_formal) and
        all(audited(p) for p, _ in multi_formal), qualified_paths(multi) + [root/'rollout_contract_audit.json'],
        'Private qualification saves seed/selected_region; actual model-input audits exclude region answers.')
    add(8, 'Both scenes use the canonical Agent/Executor', bool(formal) and all(audited(p) and
        d['action_mode'] == 'parameterized' and read(p.parent.parent/'experiment.json', {}).get('native_command')
        for p, d in formal), [root/'rollout_contract_audit.json'] + [p.parent.parent/'experiment.json' for p, _ in formal],
        'Audit authoritative macro boundaries, sensor sources, and exact startup arguments; paired runs additionally compare core/gumi hashes.')
    notes = [(p, d) for p, d in multi_formal if d['memory_mode'] == 'model_notes']
    add(9, 'Optional model-authored public notes without private injection', transport.get('passed') and bool(notes) and
        all(audited(p) and contract_rows[p.parent.parent.name]['accepted_model_notes_checked'] > 0 for p, _ in notes),
        [root/'model_notes_transport_verification.json', root/'rollout_contract_audit.json'] + [p for p, _ in notes],
        'Check original model text, the 512-character cap, actual sensor sources, and latest-four context.')
    world_cases = worlds.get('records', [])
    add(10, 'Configurable waiting policy, default true', worlds.get('passed') and len(world_cases) == 4,
        [root/'world_policy_verification.json'], 'Run verification omits the claim policy field; pause explicitly passes false.')
    add(11, 'Distinct native run/pause physics', len(world_cases) == 4 and all(
        r['wait_sim_time_s'] > 0 if r['world_mode'] == 'run' else r['wait_sim_time_s'] == 0 for r in world_cases),
        [root/'world_policy_verification.json'], 'Actual simulation clock; wall time is not treated as simulation time.')
    add(12, 'Fixed/parameterized actions and lease protection', worlds.get('passed') and
        {r['action_mode'] for r in world_cases} == {'fixed', 'parameterized'},
        [root/'world_policy_verification.json', root/'final_cpu_regression.log'],
        'Actual cached MCP reads, rejected human motion/invalid world commands, one boundary, and restoration on release.')
    metric_keys = ('first_objectively_visible_step', 'first_recognizably_visible_step', 'first_target_approach_step',
                   'first_grasp_attempt_step', 'first_successful_attachment_step', 'first_valid_lift_step', 'task_success')
    add(13, 'Separate exploration, manipulation, and success evaluation', bool(formal) and all(set(metric_keys) <= set(
        read(p.parent.parent/'final_exploration_metrics.json', {})) for p, _ in formal),
        [p.parent.parent/'final_exploration_metrics.json' for p, _ in formal],
        'Unreached stages remain null; objective pixels or model declarations do not imply success.')
    variants = {read(p.parent.parent/'scene.json', {}).get('variant') for p, _ in formal}
    add(14, 'A complete model rollout for each exploration scene', {'single_table_occlusion', 'multi_region_search'} <= variants,
        [p for p, _ in formal], 'Physical failure can be a complete rollout; initialization errors/timeouts are not complete behavioral evidence.')
    required = ('seed', 'scene_config_sha256', 'model', 'reasoning_effort', 'world_mode', 'memory_mode', 'observation_condition')
    add(15, 'Reproducible scene/model/action/memory/world settings', bool(formal) and all(
        set(required) <= set(read(p.parent.parent/'experiment.json', {})) and
        (p.parent.parent/'source_hashes.json').exists() for p, _ in formal) and bool(results.get('episodes')),
        [ROOT/'docs/milestone8_results/results.json'] + [p.parent.parent/'source_hashes.json' for p, _ in formal],
        'Each formal start saves its configuration, commands, and source hashes without overwriting original observations/outputs.')
    return {'schema': 'milestone8_native_acceptance_v1', 'passed': all(r['passed'] for r in records),
            'records': records, 'behavioral_success_is_not_scene_validity': True,
            'limitation': 'Evidence mapping checks saved records; qualification routes are privileged offline replays.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    (args.root/'acceptance.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'passed': result['passed'], 'unmet': [r['criterion'] for r in result['records'] if not r['passed']]}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

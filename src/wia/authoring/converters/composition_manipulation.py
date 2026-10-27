"""Plan source-faithful manipulation and relation task composition on CPU.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
    tests/test_composition_manipulation.py
The caller supplies decoded definitions and composition_source_raw records.
This module performs no file I/O, native initialization, or evaluator mutation.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

import copy
import json
import re


CATEGORIES = {
    ('Action Sequencing', 'Action Order Inference'),
    ('Metric Comparison', 'Dimensional Size'),
    ('Metric Comparison', 'Spatial Distance'),
    ('Physical Dynamics', 'Inclined Plane'),
    ('Physical Dynamics', 'Stacking & Stability'),
    ('Physical Structure', 'Deformable'),
    ('Physical Structure', 'Liquid Volume'),
    ('Spatial Relations', 'Geometric Configuration'),
    ('Spatial Relations', 'Linear Alignment'),
    ('Spatial Relations', 'Physical Contact'),
}
LIQUID_MODULE = 'core.esi_converters.physical_structure_liquid_volume'
LIQUID_OPERATIONS = ('dock_at_both_containers', 'dock_at_both_and_return')
CAPACITY_OPERATION = 'report_source_capacity_order'
COMMON_CHECKS = [
    'Check the real robot spawn, upright wheel contact, obstacle clearance and a usable route to every required object.',
    'Inspect low and high supports, native support contact and unchanged asset scales; height screens do not establish IK or graspability.',
    'Check mounted head and wrist sensor coverage through robot observations; human overview images are not agent evidence.',
    'Retain original QA and reference-view evidence separately; obtain initial comparison answers before interventions change the evidence.',
]
CHECKS = {
    'Action Order Inference': [
        'Locate both base candidates, the middle object and every top candidate; none may be hidden beneath another candidate or pre-stacked at initialization.',
        'Check separately supported candidate footprints, a usable grasp approach, and transport between supports without resizing or teleporting.',
        'Verify middle grasp, 5 cm lift, release and stable supporting placement before acquiring the selected top; require released final stack stability.',
    ],
    'Dimensional Size': [
        'Preserve each task/reference pair geometry and both original copy scales; inspect the comparison before lifting either copy.',
        'Report the size comparison before intervention, select the original copy and transport it to the named destination with stable released placement.',
    ],
    'Spatial Distance': [
        'Resolve and preserve the original native furniture reference and candidate XY; any support adaptation must preserve the declared common vertical translation.',
        'State and report the source distance convention before intervention, preserve the reference, and transport the selected original object to its named destination.',
    ],
    'Inclined Plane': [
        'Preserve ramp position, angle, scale and friction; check target graspability and transport to the original upper ramp face.',
        'Verify supported release before one uninterrupted second of unheld native observation; stable, sliding or falling are outcomes, not source-label success gates.',
    ],
    'Stacking & Stability': [
        'Check three separately supported original-size objects and feasible real lifts; require two upper-object supported release events and a stable unheld tower.',
        'Keep historical accepted-order QA separate from measured stability; alternative orders require physical unstacking and rebuilding.',
    ],
    'Deformable': [
        'Check the source item/cloth drop and coverage, original mesh and scale, reachable hand contact and an initially supported unmoved item.',
        'Require hand-driven 10 cm cloth displacement and complete footprint reveal before item grasp, then a real 5 cm held lift for 0.25 seconds.',
    ],
    'Liquid Volume': [
        'Compare source calibrated capacity, not current fill, outer AABB volume or docking.',
        'Match complete native inventory, scale, geometry and initial state before merging; resolve explicit ties and reject missing comparisons or cycles.',
        'A source same-trial particle-count order is a capacity proxy, not an absolute litre measurement or cross-trial calibration.',
    ],
    'Geometric Configuration': [
        'Arrange source participating objects into the requested source shape on world XY.',
        'Require normalized shape tolerances, positive separation, non-overlap, unchanged asset scales, native support and measured stable released placement.',
    ],
    'Linear Alignment': [
        'Use the current released collinear arrangement goal, not historical survey or return goals; a satisfied initial arrangement needs no forced movement.',
        'Check distinct bounding-box centers and perpendicular residual at most min(2 cm, 2% of the longest baseline), starting-support contact, low velocity and empty released grippers.',
    ],
    'Physical Contact': [
        'Use the seeded desired final native relation without forcing inversion or a redundant cycle.',
        'Distinguish touching, penetration and positive clearance; allow measured stable already-correct no-op and report its frequency.',
    ],
}


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def capacity_order_graph(container_ids, comparisons):
    """Require every pair, explicit ties and an acyclic common capacity order."""
    ids = sorted(set(container_ids))
    parent = {key: key for key in ids}
    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key
    observed = {}
    for row in comparisons:
        if not isinstance(row, (list, tuple)) or len(row) != 3 or not all(isinstance(item, str) for item in row):
            return {'accepted': False, 'reason': 'invalid_capacity_comparison'}
        first, relation, second = row
        if first not in parent or second not in parent or first == second or relation not in ('<', '=', '>'):
            return {'accepted': False, 'reason': 'invalid_capacity_comparison'}
        pair = tuple(sorted((first, second)))
        normalized = relation if first == pair[0] or relation == '=' else ('>' if relation == '<' else '<')
        if pair in observed and observed[pair] != normalized:
            return {'accepted': False, 'reason': 'inconsistent_capacity_comparison'}
        observed[pair] = normalized
        if relation == '=':
            parent[root(second)] = root(first)
    missing = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:] if (a, b) not in observed]
    if missing:
        return {'accepted': False, 'reason': 'incomplete_capacity_comparison_graph', 'missing_pairs': missing}
    edges = {root(key): set() for key in ids}
    for (first, second), relation in observed.items():
        if relation == '=':
            continue
        low, high = (first, second) if relation == '<' else (second, first)
        if root(low) == root(high):
            return {'accepted': False, 'reason': 'capacity_tie_conflicts_with_strict_order'}
        edges[root(low)].add(root(high))
    remaining, order = set(edges), []
    while remaining:
        ready = [key for key in remaining if not any(key in edges[other] for other in remaining)]
        if len(ready) != 1:
            return {'accepted': False, 'reason': 'capacity_order_cycle_or_ambiguity'}
        key = ready[0]
        order.append(sorted(item for item in ids if root(item) == key))
        remaining.remove(key)
    return {'accepted': True, 'ordered_tie_groups': order, 'pair_count': len(observed)}


def capacity_ranking_score(container_ids, truth, answer):
    """Score all n(n-1)/2 pairs, keeping exact ranking separate from accuracy.

    Answer forms are an increasing chain (``A < B = C``), ordered tie groups,
    or ``{'comparisons': [[a, relation, b], ...]}``. Missing, malformed and
    contradictory submitted pairs earn no credit. Unknown truth invalidates
    the entire comparable score; its denominator never shrinks.
    """
    ids = sorted(set(container_ids))
    pairs = [(first, second) for index, first in enumerate(ids) for second in ids[index + 1:]]
    graph = capacity_order_graph(ids, truth)
    base = {'total_pairs': len(pairs), 'correct_pairwise_relations': 0,
        'comparison_accuracy_percent': None, 'exact_ranking_success': False,
        'evaluation_validity': 'unavailable', 'truth_graph': graph}
    if not graph['accepted']:
        return {**base, 'pair_results': [{'first': a, 'second': b, 'correct': False,
            'evaluation_validity': 'unavailable'} for a, b in pairs]}
    ranks = {item: index for index, group in enumerate(graph['ordered_tie_groups']) for item in group}
    expected = {(a, b): '=' if ranks[a] == ranks[b] else '<' if ranks[a] < ranks[b] else '>' for a, b in pairs}
    if isinstance(answer, str):
        answer = answer.strip()
        try:
            answer = json.loads(answer)
        except (ValueError, TypeError):
            tokens = re.split(r'\s*([<=>])\s*', answer)
            if len(tokens) % 2 == 0 or any(token not in ids for token in tokens[::2]) or any(
                    token not in ('<', '=') for token in tokens[1::2]):
                answer = None
            else:
                groups = [[tokens[0]]]
                for relation, item in zip(tokens[1::2], tokens[2::2]):
                    groups[-1].append(item) if relation == '=' else groups.append([item])
                answer = groups
    submitted = []
    if isinstance(answer, dict) and isinstance(answer.get('comparisons'), list):
        submitted = answer['comparisons']
    else:
        groups = answer.get('ordered_tie_groups') if isinstance(answer, dict) else answer
        if (isinstance(groups, list) and groups and all(isinstance(group, list) and group for group in groups)
                and all(isinstance(item, str) and item in ids for group in groups for item in group)):
            flattened = [item for group in groups for item in group]
            if len(flattened) == len(set(flattened)):
                submitted_ranks = {item: index for index, group in enumerate(groups) for item in group}
                submitted = [[a, '=' if submitted_ranks[a] == submitted_ranks[b] else
                    '<' if submitted_ranks[a] < submitted_ranks[b] else '>', b]
                    for a, b in pairs if a in submitted_ranks and b in submitted_ranks]
    observed, contradictory = {}, set()
    for row in submitted:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            continue
        a, relation, b = row
        if not all(isinstance(value, str) for value in row) or a not in ids or b not in ids or a == b:
            continue
        pair = tuple(sorted((a, b)))
        if relation not in ('<', '=', '>'):
            contradictory.add(pair)
            continue
        normalized = relation if a == pair[0] or relation == '=' else '>' if relation == '<' else '<'
        if pair in observed and observed[pair] != normalized:
            contradictory.add(pair)
        observed[pair] = normalized
    results = [{'first': a, 'second': b, 'expected': expected[(a, b)],
        'submitted': observed.get((a, b)), 'correct': (a, b) not in contradictory and
            observed.get((a, b)) == expected[(a, b)], 'evaluation_validity': 'valid'} for a, b in pairs]
    correct = sum(row['correct'] for row in results)
    return {**base, 'evaluation_validity': 'complete', 'correct_pairwise_relations': correct,
        'comparison_accuracy_percent': 100. * correct / len(pairs) if pairs else None,
        'exact_ranking_success': bool(pairs and correct == len(pairs)), 'pair_results': results}


def _liquid_world(record, task, scene):
    """Use only fields read by native bootstrap, plus every declared overlay.

    Native side_entry gives obj1/obj2.side precedence over top-level aliases.
    Historical particle counts and run_idx differ between route variants. The
    latter seeds only later scripted pour callbacks; docking never invokes them.
    """
    raw = record.get('composition_source_raw')
    if not isinstance(raw, dict) or raw.get('runner_task') != 'pour':
        return None
    payload = raw.get('metadata_json', {})
    payload = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(payload, dict) or payload.get('skip_reason'):
        return None
    evaluation = task.get('private', {}).get('evaluation', {})
    if (module_identity(evaluation.get('module')) != LIQUID_MODULE or evaluation.get('type') != 'category_operation'
            or evaluation.get('operation') != CAPACITY_OPERATION):
        return None
    if task.get('scene') != scene.get('scene'):
        return None
    bindings = task.get('private', {}).get('bindings', {})
    if bindings != scene.get('bindings') or set(bindings) != {'left', 'right', 'floor'}:
        return None
    sides = {}
    for side in ('left', 'right'):
        entry = next((payload[k] for k in ('obj1', 'obj2')
            if isinstance(payload.get(k), dict) and payload[k].get('side') == side),
            {'category': payload.get(side + '_category'), 'model': payload.get(side + '_model')})
        if (not entry.get('category') or not entry.get('model') or any(
                entry[field] != bindings[side].get(field) for field in ('category', 'model'))):
            return None
        # Explicit geometry differences remain conservative blockers even where
        # the present source builder derives its scale from installed inventory.
        sides[side] = {k: entry[k] for k in ('category', 'model', 'scale', 'position',
            'orientation', 'quaternion_xyzw', 'fixed_base') if k in entry}
    if (raw.get('scene'), raw.get('room')) != (record.get('scene_model'), record.get('room')):
        return None
    world = {k: v for k, v in scene.items() if k not in ('id', 'source', 'status')}
    return _key({'native_scene': [raw['scene'], raw['room']], 'sides': sides,
        'fixed_env': payload.get('fixed_env'), 'seed': record.get('seed'),
        'source_seed': raw.get('seed'), 'source_payload_seed': payload.get('seed'),
        'world': world, 'capacity_metric': evaluation.get('parameters', {}).get('capacity_evidence', {}).get('metric'),
        'scale_calibration': evaluation.get('parameters', {}).get('capacity_evidence', {}).get('expected_native_scales'),
        'binding_keys': evaluation.get('binding_keys'),
        'qualification': task.get('qualification'), 'requirements': task.get('requirements'),
        'defaults': task.get('defaults'), 'converter_version': record.get('converter_version'),
        'scaffold': record.get('private_scaffold_contract'),
        'bootstrap_screen': record.get('private_robot_start_evidence')})


def _needs_fix(feedback, candidate_id):
    if not isinstance(feedback, dict):
        return False
    entries = feedback.get('entries', feedback)
    row = entries.get(candidate_id, {})
    return isinstance(row, dict) and row.get('decision') == 'needs_fix'


def plan_groups(records, tasks, scenes, *, feedback=None):
    """Partition owned generated records; archived repeats and rigid stay out."""
    records = [r for r in records if r.get('status') == 'generated'
        and (r.get('big_task'), r.get('small_task')) in CATEGORIES]
    if len({r['candidate_id'] for r in records}) != len(records):
        raise ValueError('Composition requires unique current candidate identities.')
    buckets = {}
    for record in sorted(records, key=lambda r: r['candidate_id']):
        cid = record['candidate_id']
        if cid not in tasks or cid not in scenes:
            raise ValueError('Composition requires the exact current task and scene: ' + cid)
        signature = None
        if record['small_task'] == 'Liquid Volume':
            signature = _liquid_world(record, tasks[cid], scenes[cid])
        bucket = ('liquid', signature) if signature is not None else ('single', cid)
        buckets.setdefault(bucket, []).append(record)
    result = []
    for members in buckets.values():
        capacity_orders = [tasks[r['candidate_id']]['private']['evaluation'].get('parameters', {}).get('expected_order') for r in members]
        capacity_valid = all(tasks[r['candidate_id']]['private']['evaluation'].get('parameters', {}).get('capacity_evidence', {}).get('valid') for r in members)
        comparison_graph = capacity_order_graph(['left', 'right'], [value.split() for value in capacity_orders
            if isinstance(value, str) and len(value.split()) == 3])
        # Exact same native inventory and a consistent pairwise obligation may be
        # deduplicated; independently generated container worlds never pool.
        groups = [members] if capacity_valid and comparison_graph['accepted'] else [[r] for r in members]
        for group in groups:
            ids = sorted(r['candidate_id'] for r in group)
            name = group[0]['small_task']
            combined = len(group) > 1
            representative = ids[0]
            result.append({'representative_id': representative, 'member_ids': ids,
                'label': name + (' — compatible capacity comparison' if combined else ' — revised task goal'),
                'reason': ('Identical native inventory, geometry, initialization and calibrated capacity order; deduplicate the equivalent semantic comparison while retaining all source evidence.'
                    if combined else 'Preserve the original world, physical predicate and causal history; no additional compatible active world is established.'),
                'review_goals': list(COMMON_CHECKS) + list(CHECKS[name]),
                'requires_recapture': combined or any(_needs_fix(feedback, cid) for cid in ids),
                'warnings': (['capacity_comparison_equivalence_requires_review']
                    if combined else [])})
    return sorted(result, key=lambda row: row['representative_id'])


def repair_definitions(record, task, scene, *, feedback=None):
    """Return a pending Action initialization contract for feedback derivatives.

    The shared support initializer owns enforcing the independent candidate gap.
    This contract makes the native change and required re-review explicit without
    guessing placements or changing an approved source definition in place.
    """
    original_task, original_scene = task, scene
    task, scene = copy.deepcopy(task), copy.deepcopy(scene)
    result = {'task': task, 'scene': scene, 'changed': False, 'change_summary': [], 'warnings': []}
    if record.get('small_task') != 'Action Order Inference' or not _needs_fix(feedback, record['candidate_id']):
        return result
    contract = {'minimum_independent_candidate_xy_gap_m': .03,
        'require_all_candidates_separately_supported': True,
        'require_native_support_contact_for_every_candidate': True,
        'unobstructed_candidate_inspection_requires_review': True}
    for definition in (task, scene):
        for patch in definition['scene']['variants']['adapted']['patches']:
            if patch.get('operation') == 'category_initialization' and module_identity(patch.get('module')) == 'core.esi_converters.action_order':
                patch['parameters']['initialization_review_contract'] = copy.deepcopy(contract)
    constraint = ('Inspect every original candidate and the middle object before selecting the stack; '
        'initial candidates must be separately supported and must not hide beneath or overlap another candidate.')
    if constraint not in task['public']['constraints']:
        task['public']['constraints'].append(constraint)
    result.update(changed=task != original_task or scene != original_scene,
        change_summary=['Require separate 3 cm-clear starting footprints for all Action Order candidates, retain original scales and causal stacking predicate, and recheck all candidate visibility.'],
        warnings=['action_independent_candidate_clearance_repair_requires_recapture',
            'shared_independent_support_clearance_must_be_enforced', 'no_prior_approval_inherited'])
    return result

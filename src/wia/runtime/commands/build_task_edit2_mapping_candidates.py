#!/usr/bin/env python3
"""Build two pending current-Core canonical Mapping world proposals.

Run: python src/wia/runtime/commands/build_task_edit2_mapping_candidates.py --output-dir outputs/task_edit2_20261018/mapping_worlds
Optional --budget-file is a JSON object keyed by the GROUPS keys, with explicit
max_steps, wall_timeout_s, derivation and evidence. No simulator, capture queue,
Review records, historical definitions or strict composition builder is changed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
import yaml

from wia.authoring.converters.task_edit2_mapping_world import MODULE, REVISION, digest
from wia.tasks.answer_sheet import mapping_questions
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.runtime.scene_capture_queue import read_json, write_json

CORE = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')
GROUPS = {'connectivity_school_biology': ('Connectivity', 'school_biology'),
          'regional_boundary_Rs_garden': ('Regional Boundary', 'Rs_garden')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_groups(rows):
    result = {}
    for key, (subclass, scene) in GROUPS.items():
        selected = sorted((row for row in rows if row['category'] == 'Cognitive Mapping'
            and row['subcategory'] == subclass and row['scene'] == scene), key=lambda row: row['source'])
        if len(selected) != 5 or len({r['source'] for r in selected}) != 5:
            raise ValueError(f'{key} requires exactly five distinct actual Core sources, found {len(selected)}.')
        result[key] = selected
    return result


def source_points(raw, identity):
    """Keep every saved camera/path XY and its exact source JSON pointer.

    Context labels come only from source camera containers, never graph answers
    or guessed coordinates. Repeated render copies remain in the provenance.
    """
    points, errors = [], []
    expanded = deepcopy(raw)
    if isinstance(expanded.get('metadata_json'), str):
        expanded['metadata_json'] = json.loads(expanded['metadata_json'])

    def add(position, pointer, kind, labels):
        if (not isinstance(position, (list, tuple)) or len(position) < 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in position[:2])):
            errors.append(pointer)
            return
        points.append({'source_identity': identity, 'json_pointer': pointer,
            'kind': kind, 'xy': list(position[:2]), 'context_labels': sorted(labels)})

    def walk(value, pointer='', labels=frozenset()):
        if isinstance(value, dict):
            labels = labels | frozenset(str(value[k]) for k in
                ('_key', 'room_instance', 'object_name', 'object_instance_name', 'region_name')
                if isinstance(value.get(k), str))
            for key, child in value.items():
                path = pointer + '/' + key
                if key == 'camera_pose':
                    add(child.get('position') if isinstance(child, dict) else None,
                        path + '/position', 'source_camera', labels)
                elif key in ('path_world_xy', 'path_xy'):
                    if not isinstance(child, list):
                        errors.append(path)
                    else:
                        for index, point in enumerate(child):
                            add(point, path + '/' + str(index), 'source_path', labels)
                else:
                    walk(child, path, labels)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, pointer + '/' + str(index), labels)
    walk(expanded)
    return points, errors


def manhattan_mst(points):
    """Return a deterministic geometric MST length, not a traversability claim."""
    points = sorted(set(tuple(point) for point in points))
    if not points:
        raise ValueError('No source XY points.')
    remaining = set(range(1, len(points)))
    distances = {i: abs(points[i][0] - points[0][0]) + abs(points[i][1] - points[0][1]) for i in remaining}
    total = 0.
    while remaining:
        selected = min(remaining, key=lambda i: (distances[i], i))
        total += distances[selected]
        remaining.remove(selected)
        x, y = points[selected]
        for i in remaining:
            distances[i] = min(distances[i], abs(points[i][0] - x) + abs(points[i][1] - y))
    return total


def budget_proposal(inputs, explicit=None):
    if explicit is not None:
        if (type(explicit.get('max_steps')) is not int or explicit['max_steps'] <= 0
                or type(explicit.get('wall_timeout_s')) not in (int, float)
                or not math.isfinite(explicit['wall_timeout_s']) or explicit['wall_timeout_s'] <= 0
                or not explicit.get('derivation') or not explicit.get('evidence')):
            raise ValueError('Explicit budget requires positive bounds, derivation and evidence.')
        return dict(deepcopy(explicit), origin='explicit_new_shared_budget', calibrated=False)
    if inputs['missing_camera_coverage'] or inputs['invalid_point_pointers'] or not inputs['all_source_points']:
        return {'origin': 'incomplete_geometry_proposal', 'calibrated': False,
            'requires_budget_file': True, 'derivation': 'No estimate: missing or ambiguous source camera/path evidence.'}
    # Use all saved source camera/path points as a conservative geometric superset
    # of required room/anchor views. The MST can cross walls; it is only a pilot
    # action-count proposal and requires actual controller/door calibration.
    length = manhattan_mst([point['xy'] for point in inputs['all_source_points']])
    translation = math.ceil(2 * length / .25)
    inspection = 12 * len(inputs['required_distinct_rooms'])
    questions = 4 * inputs['required_unique_questions']
    steps = translation + inspection + questions
    return {'origin': 'new_uncalibrated_geometric_pilot', 'calibrated': False,
        'max_steps': steps, 'wall_timeout_s': float(3 * steps),
        'derivation': 'ceil(2 * Manhattan_MST(all unique saved camera/path XY) / 0.25m) + 12 * distinct required rooms + 4 * unique semantic questions',
        'evidence': {'manhattan_mst_m': length, 'translation_increment_m': .25,
            'geometric_traversals': 2, 'translation_actions': translation,
            'inspection_turn_actions': inspection, 'answer_actions': questions,
            'wall_seconds_per_action': 3., 'wall_bound_calibrated': False,
            'point_set': 'all_saved_source_camera_and_path_XY_superset',
            'unique_xy_count': len(set(tuple(p['xy']) for p in inputs['all_source_points']))},
        'limitation': 'Geometry ignores walls, door operation and controller timing; native route and budget calibration remain mandatory.'}


def build_group(key, rows, output_dir, *, explicit_budget=None, source_root=None, group_spec=None):
    rows = sorted(rows, key=lambda row: row['source'])
    subclass, scene_name = group_spec or GROUPS[key]
    source_root = Path(source_root or WIA_ESI_ROOT / 'dataset/json_clean')
    members, provenance, tasks, points, point_errors, coverage = [], [], [], [], [], []
    rooms, anchors = set(), {}
    for index, row in enumerate(rows, 1):
        task_path, scene_path = Path(row['task_file']), Path(row['scene_file'])
        task = yaml.safe_load(task_path.read_text())
        raw_path = source_root / row['source']
        raw = read_json(raw_path)
        if raw.get('scene') != scene_name or raw.get('small_task') != subclass:
            raise ValueError('Source payload differs from exact Core membership.')
        if task['source']['json'] != row['source']:
            raise ValueError('Child definition source identity differs.')
        p = task['private']['evaluation']['parameters']
        if p.get('required_order') or p.get('public_route_required'):
            raise ValueError('Ordered routes cannot enter canonical unordered worlds.')
        tasks.append(task)
        source_hash = sha(raw_path)
        identity = {'source_identity': row['source'], 'source_sha256': source_hash,
            'task_sha256': sha(task_path), 'scene_sha256': sha(scene_path)}
        provenance.append({**identity, 'source_core_candidate_id': row['candidate_id'],
            'source_file': str(raw_path.resolve()), 'task_file': str(task_path), 'scene_file': str(scene_path)})
        members.append({'id': 'native_' + digest(row['source'])[:20], 'public_id': f'goal_{index}',
            'source_identity': row['source'], 'source_sha256': source_hash, 'small_task': subclass,
            'public': deepcopy(task['public']), 'evaluation': deepcopy(task['private']['evaluation']),
            'bindings': deepcopy(task['private']['bindings']),
            'qa': {k: deepcopy(raw.get(k)) for k in ('question', 'answer', 'answer_type', 'options_json')},
            'source_task_definition': deepcopy(task)})
        required_rooms = set(p.get('required_regions', []))
        rooms.update(required_rooms)
        required_anchors = {task['private']['bindings'][binding]['name'] for binding in p.get('required_anchor_bindings', [])}
        for binding in p.get('required_anchor_bindings', []):
            definition = task['private']['bindings'][binding]
            anchors[digest(definition)] = deepcopy(definition)
        saved, errors = source_points(raw, row['source'])
        points.extend(saved)
        point_errors.extend({'source_identity': row['source'], 'json_pointer': e} for e in errors)
        labels = {label for point in saved if point['kind'] == 'source_camera' for label in point['context_labels']}
        missing = sorted((required_rooms | required_anchors) - labels)
        if missing:
            coverage.append({'source_identity': row['source'], 'missing_room_or_anchor_labels': missing})
    questions = mapping_questions(members)
    inputs = {'required_distinct_rooms': sorted(rooms),
        'required_distinct_anchor_inspections': sorted(anchors.values(), key=lambda x: x['name']),
        'required_unique_questions': len(questions), 'semantic_questions': deepcopy(questions),
        'all_source_points': points, 'invalid_point_pointers': point_errors,
        'missing_camera_coverage': coverage, 'child_budgets_used': False}
    budget = budget_proposal(inputs, explicit_budget)
    representative = tasks[0]
    pose = [deepcopy(p) for p in representative['scene']['variants'].get('adapted', {'patches': []})['patches']
            if p['operation'] == 'robot_pose']
    if len(pose) > 1:
        raise ValueError('Representative has ambiguous declared robot starts.')
    contract = {'revision': REVISION, 'group_key': key, 'scene': scene_name, 'room': 'full_scene',
        'source_members': [{k: p[k] for k in ('source_identity', 'source_sha256', 'task_sha256', 'scene_sha256')} for p in provenance],
        'representative_source': rows[0]['source'],
        'common_start_proposal': pose[0] if pose else {'kind': 'representative_source_default', 'native_start': 'pending'},
        'seed': representative.get('defaults', {}).get('seed', 0), 'episode_budget': budget,
        'full_building_required': True, 'retain_native_furniture_and_doors': True,
        'source_world_equivalence': 'unverified_pending_all_source_relations_in_this_actual_reset',
        'qualification_api': 'session.mapping_joint_world_identity(contract) and session.mapping_joint_world_qualification(contract)',
        'pending_gates': ['full_building_inventory_geometry', 'all_original_source_relations',
            'source_runtime_door_reconciliation_and_public_crossings', 'common_start_support_and_clearance',
            'controller_and_eight_plus_initial_global', 'shared_budget_native_calibration']}
    record = {'group_key': key, 'status': 'pending_diagnostic', 'benchmark_ready': False,
        'source_count': len(rows), 'sources': provenance, 'world_contract': contract,
        'budget_inputs': inputs, 'native_qualification_run': False,
        'execution_certificate': None, 'task_file': None, 'scene_file': None,
        'semantic_metric': '100 * correct required unique semantic questions / required unique semantic questions; equal weight; unavailable native truth keeps denominator and yields unavailable score',
        'physical_metric': 'Current measured physical obligations are separate; one continuous shared episode, unordered visits, current terminal constraints.'}
    if budget.get('requires_budget_file'):
        record['failure_stage'] = 'budget_source_geometry_incomplete'
        return record
    task = deepcopy(representative)
    task.update(id='task_' + digest(contract)[:24], version=2, status='pending')
    bindings = {}
    for index, member in enumerate(members):
        for name, definition in member['bindings'].items():
            bindings[f'member_{index + 1}_{name}'] = deepcopy(definition)
    task['private'] = {'bindings': bindings, 'evaluation': {'type': 'category_operation',
        'module': MODULE, 'operation': 'canonical_mapping_joint_world', 'binding_keys': list(bindings),
        'parameters': {'task_revision': REVISION, 'members': members, 'answer_sheet': True,
            'semantic_questions': questions, 'mapping_joint_world_contract': contract}}}
    canonical = {question['sources'][0]['public_id'] for question in questions}
    task['public']['objective'] = 'Explore the complete preserved building once in one continuous episode.\n' + '\n'.join(
        f"{m['public_id']}: {m['qa']['question']}" for m in members if m['public_id'] in canonical)
    task['public']['success_criteria'] = ('Inspect the queried anchors and report each distinct required room arrival while stably inside it. '
        'Shared unordered visits can satisfy multiple questions. Submit one final JSON answer sheet in one world_observation line: '
        'ANSWER: {' + ','.join(json.dumps(m['public_id']) + ':"<answer>"' for m in members if m['public_id'] in canonical) + '}. '
        'Use each question\'s source options shown below. Legacy ANSWER: goal_N=<answer> is accepted. '
        'No action or observation is required solely to answer another question.\n' + '\n'.join(
            m['public_id'] + ': ' + m['public']['success_criteria'] for m in members if m['public_id'] in canonical))
    task['public']['environment'] = 'The complete original building, furniture and native doors are retained. Use all eight mounted cameras and the initial Global overview.'
    patches = [{'operation': 'category_initialization', 'phase': 'before_construction', 'module': MODULE,
                'parameters': {'mapping_joint_world_contract': deepcopy(contract)}}, *pose]
    task['scene']['variants'] = {'original': {'patches': []}, 'adapted': {'patches': deepcopy(patches)}}
    task['defaults'] = {'seed': contract['seed'], 'max_steps': budget['max_steps'], 'wall_timeout_s': budget['wall_timeout_s']}
    task['repair'] = {'definition_revision': REVISION, 'diagnostic_only': True,
        'mapping_joint_world_contract': deepcopy(contract), 'source_provenance': provenance}
    scene = {'kind': 'scene_patch', 'id': 'scene_' + digest(contract)[:24], 'version': 2,
        'status': 'pending', 'source': deepcopy(task['source']), 'bindings': deepcopy(bindings),
        'scene': deepcopy(task['scene']), 'seed': contract['seed']}
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    directory = Path(output_dir) / 'candidates' / key
    directory.mkdir(parents=True, exist_ok=True)
    for name, definition in [('task', task), ('scene', scene)]:
        path = directory / (name + '.yaml')
        path.write_text(yaml.safe_dump(definition, sort_keys=False, allow_unicode=True), encoding='utf-8')
        record[name + '_file'] = str(path.resolve())
        record[name + '_sha256'] = sha(path)
    return record


def build_candidates(output_dir, *, core_path=CORE, budgets=None, source_root=None):
    groups = select_groups(read_json(core_path)['tasks'])
    unknown = set(budgets or {}) - set(GROUPS)
    if unknown:
        raise ValueError('Unknown explicit budget groups: ' + ', '.join(sorted(unknown)))
    records = [build_group(key, rows, output_dir, explicit_budget=(budgets or {}).get(key),
                           source_root=source_root) for key, rows in groups.items()]
    manifest = {'schema': REVISION, 'core_membership': str(core_path), 'core_membership_sha256': sha(core_path),
        'records': records, 'source_count': sum(r['source_count'] for r in records),
        'task_count': sum(r['task_file'] is not None for r in records), 'native_qualification_run': False,
        'queue_or_review_mutation': False, 'strict_composition_builder_changed': False}
    write_json(Path(output_dir) / 'mapping_worlds_manifest.json', manifest)
    write_json(Path(output_dir) / 'source_to_parent.json', {'revision': REVISION, 'sources': [
        dict(source, parent_group=r['group_key'], parent_task_file=r['task_file'], parent_scene_file=r['scene_file'])
        for r in records for source in r['sources']]})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--core-membership', type=Path, default=CORE)
    parser.add_argument('--budget-file', type=Path)
    args = parser.parse_args()
    manifest = build_candidates(args.output_dir.resolve(), core_path=args.core_membership,
        budgets=read_json(args.budget_file) if args.budget_file else None)
    print(json.dumps({'source_count': manifest['source_count'], 'task_count': manifest['task_count'],
        'manifest': str(args.output_dir.resolve() / 'mapping_worlds_manifest.json')}))


if __name__ == '__main__':
    main()

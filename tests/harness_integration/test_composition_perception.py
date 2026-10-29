"""Exercise conservative perception policy with isolated source fixtures.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_composition_perception.py
"""
import copy
import json

import pytest

from wia.authoring.converters.composition_perception import _PAIR_LOCATIONS, plan_groups


def _pair(tmp_path, category='Counting w Occlusion', location=None):
    model, room, left, right = location or _PAIR_LOCATIONS[category][0]
    operations = {
        'Category Ambiguity': ['inventory_return'] * 2,
        'Counting w Occlusion': ['occluder_side_patrol', 'occluder_side_patrol_return'],
        'Illumination Variability': ['two_phase_inventory_return'] * 2,
        'Spatial Segmentation': ['segmented_access_patrol_return', 'segmented_two_approach_patrol_return'],
    }[category]
    records, tasks, scenes = [], {}, {}
    for index, operation in zip((left, right), operations):
        identity = f'Enumerative Perception/{category}/{model}/{room}/q_{index}.json'
        key = identity.replace('/', '_')
        path = tmp_path / (key + '.json')
        metadata = {'question_data': {'render': {'image': f'q_{index}.png',
            'camera_poses': [{'_key': f'q_{index}.png', 'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]}],
            'resolved_objects': {'targets': [{'name': 'target', 'category': 'sandwich', 'model': 'model_a',
                'position': [1, 1, 0.5], 'quaternion_xyzw': [0, 0, 0, 1]}]}}}}
        raw = {'scene': model, 'room': room, 'metadata_json': json.dumps(metadata),
               'question': 'How many objects?', 'answer': '1', 'options_json': json.dumps(['1', index])}
        path.write_text(json.dumps(raw))
        records.append({'candidate_id': key, 'status': 'generated', 'small_task': category,
                        'source_identity': identity, 'original_source': str(path), 'scene_file': key + '.yaml'})
        patches = {'variants': {'adapted': {'patches': [
            {'operation': 'category_initialization', 'parameters': {'source_provenance': {
                'source_identity': identity, 'source_sha256': index}, 'phase_duration_s': 30}},
        ]}}}
        bindings = {'target': {'name': 'target', 'category': 'sandwich', 'model': 'model_a'}}
        scenes[key] = {'id': key, 'source': {'json': identity}, 'scene': copy.deepcopy(patches), 'bindings': bindings, 'seed': 0}
        tasks[key] = {'scene': copy.deepcopy(patches), 'private': {'bindings': bindings,
            'qa': {'source': identity, 'answer': '1', 'options': ['1', index]},
            'evaluation': {'type': 'category_operation', 'module': category, 'operation': operation,
                           'parameters': {'target_keys': ['target']}}},
            'public': {'objective': operation}, 'defaults': {'seed': 0}}
    return records, tasks, scenes


def test_all_audited_pairs_and_complete_partition(tmp_path):
    counts = {'Category Ambiguity': 60, 'Counting w Occlusion': 23, 'Illumination Variability': 50,
              'Merged Observation': 60, 'Spatial Segmentation': 25, 'Structural Enclosure': 39,
              'Material Transparency': 93, 'Partial Occlusion': 95, 'View Hallucination': 437}
    records, tasks, scenes = [], {}, {}
    for category, count in counts.items():
        for location in _PAIR_LOCATIONS.get(category, []):
            rr, tt, ss = _pair(tmp_path, category, location)
            records.extend(rr)
            tasks.update(tt)
            scenes.update(ss)
        for index in range(count - 2 * len(_PAIR_LOCATIONS.get(category, []))):
            key = f'{category}-{index}'
            records.append({'candidate_id': key, 'source_identity': key, 'small_task': category, 'status': 'generated'})
    before = copy.deepcopy((records, tasks, scenes))
    groups = plan_groups(records, tasks, scenes)
    assert len(records) == 882
    assert len(groups) == 866
    assert sum(len(group['member_ids']) == 2 for group in groups) == 16
    assert sorted(key for group in groups for key in group['member_ids']) == sorted(record['candidate_id'] for record in records)
    assert (records, tasks, scenes) == before
    assert all(group['requires_recapture'] for group in groups if len(group['member_ids']) > 1)


@pytest.mark.parametrize('field,value', [('model', 'different'), ('position', [2, 1, .5]),
                                       ('scale', [2, 2, 2]), ('category', 'different')])
def test_changed_actor_prevents_pairing(tmp_path, field, value):
    records, tasks, scenes = _pair(tmp_path)
    path = tmp_path / (records[1]['candidate_id'] + '.json')
    raw = json.loads(path.read_text())
    metadata = json.loads(raw['metadata_json'])
    metadata['question_data']['render']['resolved_objects']['targets'][0][field] = value
    raw['metadata_json'] = json.dumps(metadata)
    path.write_text(json.dumps(raw))
    groups = plan_groups(records, tasks, scenes)
    assert len(groups) == 2
    assert all(group.get('warnings') for group in groups)


@pytest.mark.parametrize('change', ['phase', 'pose', 'predicate', 'target_binding'])
def test_changed_native_initialization_or_child_predicate_prevents_pairing(tmp_path, change):
    records, tasks, scenes = _pair(tmp_path)
    key = records[1]['candidate_id']
    if change == 'phase':
        scenes[key]['scene']['variants']['adapted']['patches'][0]['parameters']['phase_duration_s'] = 10
    elif change == 'pose':
        tasks[key]['scene']['variants']['adapted']['patches'].append({'operation': 'robot_pose', 'position': [2, 2, 0]})
    elif change == 'predicate':
        tasks[key]['private']['evaluation']['operation'] = 'easier_unknown_predicate'
    else:
        tasks[key]['private']['evaluation']['parameters']['target_keys'] = ['other_target']
    assert len(plan_groups(records, tasks, scenes)) == 2


def test_feedback_qa_and_stage_requirements_are_preserved(tmp_path):
    records, tasks, scenes = _pair(tmp_path)
    preferred = records[1]['candidate_id']
    records[1]['cpu_candidate_id'] = 'cpu_reviewed'
    feedback = {'entries': {'cpu_reviewed': {'decision': 'approved', 'approval_scope': 'layout',
        'notes': 'Keep the measured support.', 'updated_at': '2026-10-16'},
        records[0]['candidate_id']: {'decision': 'needs_fix', 'notes': 'Check the mounted view.'}}}
    before = copy.deepcopy(tasks)
    group, = plan_groups(records, tasks, scenes, feedback=feedback)
    assert group['representative_id'] == preferred
    assert group['requires_recapture']
    assert 'task approval' in group['reason']
    assert any('before returning' in check for check in group['review_goals'])
    assert any('Keep the measured support.' in warning for warning in group['warnings'])
    assert any('Check the mounted view.' in warning for warning in group['warnings'])
    assert tasks == before
    assert tasks[records[0]['candidate_id']]['private']['qa'] != tasks[preferred]['private']['qa']


def test_excludes_repeats_failures_and_archived_records(tmp_path):
    records, tasks, scenes = _pair(tmp_path)
    for status in ('repeat', 'archived', 'conversion_failed'):
        row = dict(records[0], candidate_id=status, status=status)
        records.append(row)
    groups = plan_groups(records, tasks, scenes)
    assert len(groups) == 1
    assert len(groups[0]['member_ids']) == 2


def test_operational_table_authorization_identity_is_not_ignored(tmp_path):
    records, tasks, scenes = _pair(tmp_path)
    for record in records:
        key = record['candidate_id']
        scenes[key]['scene']['variants']['adapted']['patches'][0]['parameters']['table_plan'] = {
            'source_identity': key, 'source_sha256': key}
    assert len(plan_groups(records, tasks, scenes)) == 2


def test_source_projection_works_without_source_file(tmp_path):
    records, tasks, scenes = _pair(tmp_path)
    for record in records:
        path = tmp_path / (record['candidate_id'] + '.json')
        record['composition_source_raw'] = json.loads(path.read_text())
        path.unlink()
    assert len(plan_groups(records, tasks, scenes)) == 1


def test_each_category_has_actionable_checks_without_source_answers():
    from wia.authoring.converters.composition_perception import CATEGORIES
    records = [{'candidate_id': category, 'source_identity': category, 'small_task': category, 'status': 'generated'}
               for category in CATEGORIES]
    groups = plan_groups(records, {}, {})
    assert len(groups) == 9
    assert all(len(group['review_goals']) >= 7 for group in groups)
    assert all(not group['requires_recapture'] for group in groups)

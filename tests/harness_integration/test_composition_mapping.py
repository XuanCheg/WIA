"""Validate Mapping partition, initialization selection and feedback repairs.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
tests/test_composition_mapping.py
These CPU checks do not launch a renderer, model or native robot action.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import cognitive_mapping_connectivity as connectivity
from wia.authoring.converters.composition_mapping import OPERATIONS, plan_groups, repair_definitions
from wia.authoring.converters.cognitive_mapping_visit_reports import REVISED_MAPPING_OPERATIONS


def definition(candidate, kind='Connectivity', *, building='building', start=(1., 2.), yaw=.5):
    module, _ = OPERATIONS[kind]
    operation = REVISED_MAPPING_OPERATIONS[kind]
    record = {'candidate_id': candidate, 'status': 'generated', 'big_task': 'Cognitive Mapping',
              'small_task': kind, 'scene_model': building, 'seed': 0,
              'source_identity': f'Cognitive Mapping/{kind}/{building}/full_scene/{candidate}.json'}
    parameters = {'floor_idx': 0, 'required_regions': ['room_a_0', 'room_b_0'], 'required_order': [],
                  'mapping_composition_contract': {'compatibility_verified': True,
                    **{key: 'same' for key in ('world_state_id', 'start_state_id', 'phase_id',
                         'conditions_id', 'robot_configuration_id', 'controller_id')}}}
    patches = [{'operation': 'prepare_category_interaction', 'mode': 'navigation',
                'relocation_policy': 'preserve', 'source_region': 'room_a_0',
                'group_bindings': [], 'manipulation_bindings': []}]
    if kind in {'Long-Term Navigation', 'Traversable Passage'}:
        parameters.update(start_xy=list(start), initial_yaw_rad=yaw, start_region='room_a_0')
        patches.insert(0, {'operation': 'robot_pose', 'phase': 'before_construction',
            'position': [*start, .1], 'yaw_deg': yaw * 180 / 3.141592653589793})
    task = {'public': {'objective': 'Visit both rooms and report actual arrivals.',
                        'success_criteria': 'Visit the specified rooms and report the source relation.'},
            'private': {'evaluation': {'module': 'wia.authoring.converters.cognitive_mapping_' + module,
                        'operation': operation, 'parameters': parameters}},
            'defaults': {'seed': 0}, 'source': {'json': record['source_identity']},
            'scene': {'variants': {'original': {'patches': []}, 'adapted': {'patches': patches}}}}
    scene = {'scene': deepcopy(task['scene']), 'seed': 0}
    return record, task, scene


def plan(*definitions, feedback=None):
    return plan_groups([d[0] for d in definitions], {d[0]['candidate_id']: d[1] for d in definitions},
                       {d[0]['candidate_id']: d[2] for d in definitions}, feedback=feedback)


def test_free_goals_cannot_inherit_a_different_constrained_start():
    c, b = definition('connectivity'), definition('boundary', 'Regional Boundary')
    n = definition('navigation', 'Long-Term Navigation')
    p = definition('passage', 'Traversable Passage', start=(9., 2.))
    before = deepcopy((c, b, n, p))
    groups = plan(c, b, n, p, feedback={'connectivity': {'decision': 'approved'}})
    assert len(groups) == 3
    merged = next(g for g in groups if len(g['member_ids']) > 1)
    assert set(merged['member_ids']) == {'connectivity', 'boundary'}
    assert all(len(g['member_ids']) == 1 for g in groups if g is not merged)
    assert (c, b, n, p) == before


def test_same_source_room_different_pose_is_not_conditional_repair_pooling():
    n = definition('navigation', 'Long-Term Navigation')
    p = definition('passage', 'Traversable Passage', start=(1.2, 2.))
    assert len(plan(n, p)) == 2


def test_prefer_approved_and_avoid_flagged_bad_spawn_representatives():
    a, b, c = definition('a'), definition('b'), definition('c')
    feedback = {'entries': {'a': {'decision': 'needs_fix', 'notes': 'Wrong spawn.'},
                            'b': {'decision': 'approved'}}}
    group = plan(c, b, a, feedback=feedback)[0]
    assert group['representative_id'] == 'b'
    assert group['requires_recapture']
    assert group['member_ids'] == ['a', 'b', 'c']
    assert all('Wrong spawn' not in goal for goal in group['review_goals'])
    assert plan(a, b, c, feedback=feedback) == plan(c, b, a, feedback=feedback)


def test_unknown_world_mutation_is_isolated_and_original_route_is_retained():
    a, b = definition('a'), definition('b', 'Long-Term Navigation')
    a[2]['scene']['variants']['adapted']['patches'].append({'operation': 'object_pose'})
    b[1]['private']['evaluation']['parameters']['required_order'] = ['room_a_0', 'room_b_0']
    groups = plan(a, b)
    assert len(groups) == 2
    assert [r['route'] for group in groups for r in group['ordered_route_obligations']] == [['room_a_0', 'room_b_0']]


def test_excludes_inactive_and_out_of_scope_records_and_preserves_singleton_approval():
    active, repeat, other = definition('active'), definition('repeat'), definition('other')
    repeat[0]['status'] = 'repeat'
    other[0]['big_task'] = 'Specular Reflection'
    groups = plan(active, repeat, other, feedback={'active': {'decision': 'approved'}})
    assert len(groups) == 1 and groups[0]['member_ids'] == ['active']
    assert not groups[0]['requires_recapture']


def test_missing_current_definition_does_not_silently_drop_a_goal():
    record, task, scene = definition('a')
    with pytest.raises(ValueError, match='Missing task/scene'):
        plan_groups([record], {'a': task}, {})


def chemistry_source(name):
    relative = f'Cognitive Mapping/Connectivity/school_chemistry/full_scene/{name}.json'
    return SourceInstance(ROOT / 'ESI/dataset/json_clean' / relative, relative,
                          'Cognitive Mapping', 'Connectivity')


def test_school_chemistry_direct_vs_indirect_reachability_does_not_remove_visits(tmp_path):
    converter = connectivity.CategoryConverter()
    direct, indirect = chemistry_source('q_000'), chemistry_source('q_005')
    assert direct.payload['question_data']['directly_adjacent'] is True
    assert indirect.payload['question_data']['directly_adjacent'] is False
    for source in (direct, indirect):
        q = source.payload['question_data']
        assert q['verification']['graph_reachable'] is True
        assert converter.inspect(source)['eligible']
        result = converter.convert(source, ConversionContext(tmp_path))
        p = result['task_definition']['private']['evaluation']['parameters']
        assert p['required_regions'] == [q['source_region'], q['target_region']]
        assert result['scene_definition']['scene']['variants']['original']['patches'] == []
        proposal = result['scene_definition']['scene']['variants']['adapted']['patches'][0]
        assert proposal['position'][:2] == q['initial_view']['camera_pose']['position'][:2]
    assert len(indirect.payload['question_data']['graph_shortest_region_path']) == 3


def test_contradictory_source_unreachability_remains_diagnostic(tmp_path):
    raw = deepcopy(chemistry_source('q_005').raw)
    payload = json.loads(raw['metadata_json'])
    payload['question_data']['verification']['graph_reachable'] = False
    payload['question_data']['graph_shortest_region_path'] = None
    raw['metadata_json'] = json.dumps(payload)
    path = tmp_path / 'unreachable.json'
    path.write_text(json.dumps(raw))
    source = SourceInstance(path, 'unreachable.json', 'Cognitive Mapping', 'Connectivity')
    inspected = connectivity.CategoryConverter().inspect(source)
    assert inspected['eligible']
    result = connectivity.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'needs_human_decision'
    assert result['task_definition'] is None


def test_feedback_repair_only_updates_new_copies_and_keeps_world_and_guards(tmp_path):
    record, task, scene = definition('bad')
    record['original_source'] = str(chemistry_source('q_000').path)
    before = deepcopy((record, task, scene))
    feedback = {'bad': {'decision': 'needs_fix', 'notes': '机器人出生点明显错误'}}
    fixed = repair_definitions(record, task, scene, feedback=feedback)
    assert fixed['changed']
    assert (record, task, scene) == before
    assert fixed['task']['private'] == task['private']
    assert fixed['task']['scene'] == fixed['scene']['scene']
    assert fixed['task']['scene']['variants']['original'] == task['scene']['variants']['original']
    patches = fixed['task']['scene']['variants']['adapted']['patches']
    assert patches[-1] == task['scene']['variants']['adapted']['patches'][-1]
    assert patches[0]['position'][:2] == [-5.550000190734863, -2.25]
    assert not repair_definitions(record, fixed['task'], fixed['scene'], feedback=feedback)['changed']
    for notes in ['Working wrist is dark.', '', 'No urgent change to the courtyard start.']:
        assert not repair_definitions(record, task, scene,
            feedback={'bad': {'decision': 'needs_fix', 'notes': notes}})['changed']
    assert not repair_definitions(record, task, scene, feedback={'bad': {'decision': 'approved',
        'notes': 'Historical wrong spawn'}})['changed']


def test_historical_249_mapping_definitions_require_new_compatibility_evidence():
    root = Path('/home/zhouxinyang/esi_category_conversion/20261014_v2/review_catalogue')
    if not (root / 'category_revisions.json').is_file():
        pytest.skip('Current campaign catalogue is not installed.')
    categories = json.loads((root / 'category_revisions.json').read_text())['categories']
    ids = ['88798f66bf58ca807d77', '7eb5db8954ad17826767',
           '0e849482c1a730347a54', '73ae7788476201f0c79d']
    records, tasks, scenes = [], {}, {}
    for key in ids:
        manifest_path = Path(categories[key]['new_batch_file'])
        if categories[key].get('composition_publication'):
            publication = json.loads(Path(categories[key]['composition_publication']).read_text())
            # Replay the declared pre-composition inputs after the active catalogue switches to 91 groups.
            manifest_path = Path(publication['output_dir']) / 'original_batches' / (key + '.json')
        manifest = json.loads(manifest_path.read_text())
        for record in manifest['records']:
            if record['status'] != 'generated':
                continue
            candidate = record['candidate_id']
            records.append(record)
            tasks[candidate] = yaml.safe_load(Path(record['task_file']).read_text())
            scenes[candidate] = yaml.safe_load(Path(record['scene_file']).read_text())
    feedback = json.loads((ROOT / 'outputs/local_capture_20261014/gallery/reviews.json').read_text())
    groups = plan_groups(records, tasks, scenes, feedback=feedback)
    members = [key for group in groups for key in group['member_ids']]
    assert len(records) == len(members) == len(set(members)) == 249
    assert len(groups) == 249
    assert sum(bool(task['private']['evaluation']['parameters']['required_order'])
               for task in tasks.values()) == 53
    assert all(len(group['member_ids']) == 1 for group in groups)
    assert all('not certified' in group['reason'] for group in groups)

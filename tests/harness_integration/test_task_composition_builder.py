"""Check immutable multi-source task assembly using synthetic CPU inputs.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_composition_builder.py
Fixtures never write campaign data, mutate original definitions or load a simulator.
"""
from copy import deepcopy
import json

from wia.runtime.esi_task_composition import build_group, goal_id


def inputs(*, alias_collision=False):
    a = {'name': 'native_a', 'category': 'cup', 'model': 'cup_model'}
    b = {'name': 'native_b', 'category': 'plate', 'model': 'plate_model'}
    c = {'name': 'native_c', 'category': 'bowl', 'model': 'bowl_model'}
    records, tasks, scenes = {}, {}, {}
    for index, (key, binding) in enumerate((('a', a), ('b', b))):
        original_question = f'Which object satisfies public relation {index}?'
        raw = {'question': original_question, 'answer': f'PRIVATE_ANSWER_CANARY_{key}',
               'answer_type': 'choice', 'options_json': '["left", "right"]'}
        records[key] = {'candidate_id': key, 'source_identity': f'Cognitive Mapping/Long-Term Navigation/building/full_scene/{key}.json',
                        'source_sha256': 'original_source_sha_' + key, 'status': 'generated',
                        'big_task': 'Cognitive Mapping', 'small_task': 'Long-Term Navigation',
                        'original_source': '/not/read/in_memory_source_' + key,
                        'revision': 'original_revision_' + key, 'converter_version': 'fixture_v1',
                        'composition_source_raw': raw, 'seed': 0, 'warning_codes': [],
                        'change_summary': [], 'job_id': 'historical_job_' + key,
                        'fingerprint': 'historical_fingerprint_' + key}
        bindings = {'target': deepcopy(binding)}
        if index == 0 and alias_collision:
            bindings['member_1_target'] = deepcopy(c)
        evaluation = {'type': 'category_operation',
                      'module': 'wia.authoring.converters.cognitive_mapping_long_term_navigation',
                      'operation': 'navigation_exploration_arrivals', 'binding_keys': list(bindings),
                      'parameters': {'required_regions': ['room_a_0', 'room_b_0'],
                                     'required_order': ['room_a_0', 'room_b_0', 'room_a_0'] if index else
                                                       ['room_a_0', 'room_c_0', 'room_b_0'],
                                     'public_route_required': True,
                                     'source_public_question': original_question,
                                     'private_route_canary': 'PRIVATE_ROUTE_CANARY_' + key}}
        initial = [{'operation': 'robot_pose', 'phase': 'before_construction',
                    'position': [index + 1., 2., .1], 'yaw_deg': index * 10.},
                   {'operation': 'prepare_category_interaction', 'phase': 'after_postprocessing',
                    'mode': 'navigation', 'source_region': 'room_a_0', 'relocation_policy': 'preserve'}]
        tasks[key] = {'id': 'original_task_' + key, 'kind': 'category_robot_task',
                      'source': {'provider': 'esi', 'json': records[key]['source_identity']},
                      'private': {'bindings': bindings, 'evaluation': evaluation},
                      'public': {'objective': 'Visit the named public route ' + key,
                                 'success_criteria': 'Use actual stable room visits.',
                                 'environment': 'Original building.',
                                 'constraints': ['Use guarded controls.', 'Retain the original building.']},
                      'scene': {'policy': 'native_then_minimal_patch_else_skip',
                                'variants': {'original': {'patches': []}, 'adapted': {'patches': initial}}},
                      'requirements': {'capabilities': ['base_translation', 'base_rotation'] +
                                       (['gripper'] if index else [])},
                      'defaults': {'seed': 0, 'max_steps': 80 if index == 0 else 140},
                      'qualification': {'on_failure': 'skip'}}
        scenes[key] = {'id': 'original_scene_' + key, 'kind': 'scene_patch', 'seed': 0,
                       'bindings': deepcopy(bindings), 'source': deepcopy(tasks[key]['source']),
                       'scene': deepcopy(tasks[key]['scene'])}
    group = {'representative_id': 'a', 'member_ids': ['a', 'b'],
             'reason': 'Reuse an existing compatible scene with separate original goals.',
             'review_goals': ['Observe each original native completion checkpoint.']}
    return group, records, tasks, scenes


def test_build_preserves_original_evaluations_qa_identities_routes_and_initialization(tmp_path):
    group, records, tasks, scenes = inputs()
    before = deepcopy((group, records, tasks, scenes))
    result, task, scene = build_group(group, records, tasks, scenes, tmp_path / 'composites')
    assert (group, records, tasks, scenes) == before
    assert scene == scenes['a']
    assert task['scene'] == tasks['a']['scene']
    assert task['source'] == tasks['a']['source']
    children = task['private']['evaluation']['parameters']['members']
    assert len(children) == 2
    for key, child in zip(('a', 'b'), children):
        assert child['evaluation'] == tasks[key]['private']['evaluation']
        assert child['bindings'] == tasks[key]['private']['bindings']
        assert child['source_identity'] == records[key]['source_identity']
        assert child['source_sha256'] == records[key]['source_sha256']
        assert child['qa'] == {'question': records[key]['composition_source_raw']['question'],
                              'answer': 'PRIVATE_ANSWER_CANARY_' + key, 'answer_type': 'choice',
                              'options': ['left', 'right']}
        assert child['id'] == goal_id(records[key])
        assert child['evaluation']['parameters']['required_order'] == tasks[key]['private']['evaluation']['parameters']['required_order']
    assert result['composite']['sources'] == [records[key]['source_identity'] for key in ('a', 'b')]
    assert result['parent_candidate_ids'] == ['a', 'b']
    assert result['parent_candidate_id'] == 'a'
    assert task['defaults']['max_steps'] == 220
    assert task['requirements']['capabilities'] == ['base_translation', 'base_rotation', 'gripper']
    assert result['goal_predicate'] == task['private']['evaluation']
    assert 'job_id' not in result and 'fingerprint' not in result
    assert not (tmp_path / 'composites').exists(), 'Builder must return definitions without publishing them.'


def test_public_prompt_and_public_goals_do_not_expose_private_answers_or_routes(tmp_path):
    group, records, tasks, scenes = inputs()
    result, task, scene = build_group(group, records, tasks, scenes, tmp_path)
    public = json.dumps({'prompt': task['public'], 'goals': result['composite']['public_goals'],
                         'review_goals': result['review_goals']})
    assert 'PRIVATE_ANSWER_CANARY' not in public and 'PRIVATE_ROUTE_CANARY' not in public
    assert 'original_source_sha_' not in public
    assert 'confirmed natural items and valid ordered prefixes' in task['public']['success_criteria']
    assert 'Historical checkpoints remain recorded' in task['public']['success_criteria']
    for public_goal, key in zip(result['composite']['public_goals'], ('a', 'b')):
        assert public_goal['id'] == goal_id(records[key])
        assert public_goal['original_question'] == records[key]['composition_source_raw']['question']
        assert public_goal['original_options'] == ['left', 'right']


def test_binding_alias_collisions_resolve_without_rewriting_original_child_keys(tmp_path):
    group, records, tasks, scenes = inputs()
    result, task, scene = build_group(group, records, tasks, scenes, tmp_path)
    union = task['private']['bindings']
    assert len(union) == 2 and {definition['name'] for definition in union.values()} == {'native_a', 'native_b'}
    members = task['private']['evaluation']['parameters']['members']
    assert members[0]['bindings']['target']['name'] == 'native_a'
    assert members[1]['bindings']['target']['name'] == 'native_b'
    assert members[0]['evaluation']['binding_keys'] == members[1]['evaluation']['binding_keys'] == ['target']
    assert set(task['private']['evaluation']['binding_keys']) == set(union)
    assert scene['bindings'] == scenes['a']['bindings']


def test_generated_alias_cannot_overwrite_an_existing_representative_binding(tmp_path):
    group, records, tasks, scenes = inputs(alias_collision=True)
    result, task, scene = build_group(group, records, tasks, scenes, tmp_path)
    union = task['private']['bindings']
    assert len(union) == 3
    assert {definition['name'] for definition in union.values()} == {'native_a', 'native_b', 'native_c'}
    assert union['member_1_target']['name'] == 'native_c'
    assert task['private']['evaluation']['parameters']['members'][0]['bindings']['member_1_target']['name'] == 'native_c'


def test_group_identity_is_deterministic_and_changes_with_original_evaluator(tmp_path):
    group, records, tasks, scenes = inputs()
    first = build_group(group, records, tasks, scenes, tmp_path)
    assert build_group(group, records, tasks, scenes, tmp_path) == first
    changed = deepcopy(tasks)
    changed['b']['private']['evaluation']['parameters']['required_order'].append('room_d_0')
    second = build_group(group, records, changed, scenes, tmp_path)
    assert first[0]['candidate_id'] != second[0]['candidate_id']
    assert first[2] == second[2], 'Changing scoring must not silently change representative scene initialization.'

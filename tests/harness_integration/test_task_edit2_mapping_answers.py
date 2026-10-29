"""CPU contracts for TaskEdit2 Mapping and shared final answer sheets.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_mapping_answers.py
No renderer, simulator, service or GPU is launched.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wia.authoring.converters import composite_task
from wia.authoring.converters.composition_mapping import plan_groups
from wia.runtime.esi_task_composition import build_group
import wia.native.esi_category_evaluation as esi_category_evaluation
from wia.tasks.answer_sheet import mapping_questions, parse_answer_sheet


def member(key, *, kind='Connectivity', answer='Yes', duplicate_of=None, **parameters):
    suffix = {'Connectivity': 'connectivity', 'Regional Boundary': 'regional_boundary',
              'Long-Term Navigation': 'long_term_navigation', 'Traversable Passage': 'traversable_passage'}[kind]
    operation = {'Connectivity': 'answer_direct_adjacency_with_exploration',
                 'Regional Boundary': 'answer_observed_anchor_region_relation',
                 'Long-Term Navigation': 'execute_supplied_ordered_route',
                 'Traversable Passage': 'answer_shortest_route_via_region'}[kind]
    return {'id': key, 'small_task': kind, 'source_identity': 'private/source/' + key,
            'source_sha256': 'source_sha_' + key, 'bindings': {},
            'public': {'objective': 'Inspect the publicly described relation ' + (duplicate_of or key),
                       'success_criteria': 'Report the relation and satisfy declared visits.'},
            'evaluation': {'type': 'category_operation',
                'module': 'wia.authoring.converters.cognitive_mapping_' + suffix, 'operation': operation,
                'binding_keys': [], 'parameters': {
                    'task_revision': 'task_edit_20261018_v1', 'source_path': key,
                    'source_qa': {'scored': kind != 'Long-Term Navigation', 'question': duplicate_of or key,
                                  'answer': answer, 'options': ['Yes', 'No']},
                    'required_regions': ['room_a_0'], 'required_order': [],
                    'arrival_report_required': True, 'floor_idx': 0,
                    'mapping_composition_contract': {'compatibility_verified': True,
                        **{name: 'same' for name in ('world_state_id', 'start_state_id', 'phase_id',
                            'conditions_id', 'robot_configuration_id', 'controller_id')}}, **parameters}}}


class NativeChild:
    """Synthetic native semantic items and independently changing physical state."""
    def __init__(self, controller, config):
        self.controller, self.config = controller, config
        self.answer, self.calls, self.begins = None, [], 0

    def begin_episode(self, observation):
        self.answer = None
        self.begins += 1
        return self.record(observation)

    def report_answer(self, answer, observation):
        if self.config['parameters'].get('answer_exception'):
            raise RuntimeError('native answer scorer failed')
        self.answer = answer
        self.calls.append((answer, observation.snapshot_id, observation.action_step))

    def record(self, observation, *, action=None):
        p = self.config['parameters']
        valid = not p.get('truth_unavailable')
        correct = self.answer == p['source_qa']['answer']
        items = [{'id': 'room:room_a_0', 'kind': 'historical_checkpoint',
                  'completed': self.controller.physical_done, 'evaluation_validity': 'valid'}]
        if p.get('terminal'):
            items.append({'id': 'final_placement', 'kind': 'terminal_condition',
                          'completed': self.controller.placed, 'evaluation_validity': 'valid'})
        if p['source_qa']['scored']:
            items.append({'id': 'source_relation_answer', 'kind': 'answer_requirement',
                          'completed': correct, 'evaluation_validity': 'valid' if valid else 'unavailable'})
        if p.get('reachability_answer'):
            items.append({'id': 'global_reachability_answer', 'kind': 'answer_requirement',
                          'completed': correct, 'evaluation_validity': 'valid' if valid else 'unavailable'})
        return {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                'task_success': all(item['completed'] for item in items) and valid,
                'completion_items': items, 'evaluation_validity': 'complete' if valid else 'unavailable',
                'constraints_satisfied': True}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    children = {}
    controller = NS(physical_done=False, placed=True)
    session = NS(scene_name='same_building', room_name='room_a_0',
                 env=NS(scene=NS(object_registry=lambda *args: None)))

    def factory(session, controller, config, output):
        child = NativeChild(controller, config)
        children[config['parameters']['source_path']] = child
        return child

    monkeypatch.setattr(esi_category_evaluation, 'create_category_operation_evaluator', factory)

    def create(members):
        return composite_task.CompositeEvaluator(session, controller,
            {'parameters': {'members': members}}, tmp_path)

    return NS(create=create, children=children, controller=controller,
              boundary=NS(snapshot_id=7, action_step=3))


def test_batch_routes_every_answer_at_one_boundary_and_separates_physical_score(harness):
    evaluator = harness.create([member('a'), member('b', kind='Regional Boundary', answer='No')])
    start = evaluator.begin_episode(harness.boundary)
    assert start['required_unique_questions'] == 2 and start['answer_accuracy_percent'] == 0.
    evaluator.report_answer('{"goal_1":"Yes","goal_2":"No"}', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['answer_accuracy_percent'] == row['score_percent'] == 100.
    assert row['physical_obligations']['completion_score'] == 0.
    assert not row['task_success']
    assert all(child.begins == 1 and len(child.calls) == 1 for child in harness.children.values())
    assert harness.children['a'].calls[0][1:] == harness.children['b'].calls[0][1:] == (7, 3)
    assert row['per_subclass']['Connectivity']['answer_accuracy_percent'] == 100.
    assert row['per_subclass']['Regional Boundary']['required_unique_questions'] == 1
    harness.controller.physical_done = True
    assert evaluator.record(harness.boundary)['task_success']


@pytest.mark.parametrize('submission,expected,reason', [
    ('{', 0., 'malformed_answer_sheet'),
    ('{"goal_1":"Yes"}', 50., 'missing_question'),
    ('{"goal_1":"Yes","goal_1":"Yes","goal_2":"No"}', 50., 'duplicate_question'),
    ('{"goal_1":42,"goal_2":"No"}', 50., 'invalid_answer_value'),
    ('{"goal_3":"Yes"}', 0., 'unknown_question'),
    ('nonsense', 0., 'expected_JSON_object_or_goal_N=<answer>'),
])
def test_agent_response_errors_keep_validity_and_denominator(harness, submission, expected, reason):
    evaluator = harness.create([member('a'), member('b', answer='No')])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer(submission, harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['required_unique_questions'] == 2
    assert row['answer_evaluation_validity'] == 'complete'
    assert row['answer_accuracy_percent'] == expected
    assert any(error['reason'] == reason for error in row['agent_response_errors'])
    assert not row['task_success'] and not evaluator.blocked


def test_exact_semantic_duplicates_keep_source_mapping_without_extra_weight(harness):
    evaluator = harness.create([member('a'), member('b', duplicate_of='a'), member('c', answer='No')])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer('{"goal_1":"Yes","goal_3":"No"}', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['required_unique_questions'] == 2 and row['answer_accuracy_percent'] == 100.
    assert len(row['semantic_questions'][0]['sources']) == 2
    assert {source['member_id'] for source in row['semantic_questions'][0]['sources']} == {'a', 'b'}
    # Only genuinely identical questions can share denominator weight.
    different = member('b', duplicate_of='a', relation_semantics='global_reachability')
    assert len(mapping_questions([member('a'), different])) == 2


def test_conflicting_semantic_alias_answers_are_agent_error_not_unavailable(harness):
    evaluator = harness.create([member('a'), member('b', duplicate_of='a')])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer('{"goal_1":"Yes","goal_2":"No"}', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['required_unique_questions'] == 1 and row['answer_accuracy_percent'] == 0.
    assert row['answer_evaluation_validity'] == 'complete'
    assert row['agent_response_errors'][0]['reason'] == 'conflicting_duplicate_answers'


@pytest.mark.parametrize('failure', ['truth_unavailable', 'answer_exception'])
def test_unavailable_truth_or_child_exception_retains_question_denominator(harness, failure):
    evaluator = harness.create([member('a'), member('b', answer='No', **{failure: True})])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer('{"goal_1":"Yes","goal_2":"No"}', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['required_unique_questions'] == 2 and row['correct_answers'] == 1
    assert row['answer_accuracy_percent'] is None and row['score_percent'] is None
    assert row['answer_evaluation_validity'] == 'unavailable'
    assert row['diagnostic_answer_lower_bound_percent'] == 50.


def test_equal_question_weights_include_independent_reachability_answer(harness):
    evaluator = harness.create([member('a', reachability_answer={'answer': 'unreachable'}),
                                member('b', kind='Regional Boundary', answer='No')])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer('{"goal_1":"Yes","goal_2":"wrong"}', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['required_unique_questions'] == 3
    assert row['answer_accuracy_percent'] == pytest.approx(200. / 3.)
    assert row['per_subclass']['Connectivity']['required_unique_questions'] == 2


def test_same_room_visit_shared_across_different_connectivity_questions(harness):
    evaluator = harness.create([member('a'), member('b', answer='No')])
    evaluator.begin_episode(harness.boundary)
    harness.controller.physical_done = True
    row = evaluator.record(harness.boundary)
    assert row['physical_obligations']['total_goals'] == 1
    assert len(row['physical_obligations']['completion_items'][0]['sources']) == 2
    assert row['answer_accuracy_percent'] == 0.


def test_terminal_placement_rechecked_and_legacy_answer_routing_remains(harness):
    evaluator = harness.create([member('a', terminal=True), member('b', answer='No')])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer('goal_1=Yes', harness.boundary)
    evaluator.report_answer('goal_2=No', harness.boundary)
    harness.controller.physical_done = True
    assert evaluator.record(harness.boundary)['task_success']
    harness.controller.placed = False
    row = evaluator.record(harness.boundary)
    assert row['answer_accuracy_percent'] == 100. and not row['task_success']
    assert row['physical_obligations']['completion_score'] < 1.


def test_final_sheet_replaces_previous_answers_without_child_answer_leak(harness):
    evaluator = harness.create([member('a'), member('b', answer='No')])
    evaluator.begin_episode(harness.boundary)
    evaluator.report_answer('{"goal_1":"Yes","goal_2":"No"}', harness.boundary)
    assert evaluator.record(harness.boundary)['answer_accuracy_percent'] == 100.
    evaluator.report_answer('{"goal_1":"Yes"}', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['answer_accuracy_percent'] == 50. and row['required_unique_questions'] == 2
    assert harness.children['b'].answer == 'No', 'Old child state must not fill a missing final response.'


def test_missing_native_semantic_item_is_unavailable_without_denominator_shrink(harness):
    evaluator = harness.create([member('a')])
    evaluator.begin_episode(harness.boundary)
    native_record = harness.children['a'].record

    def without_semantic_item(boundary, **kwargs):
        row = native_record(boundary, **kwargs)
        row['completion_items'] = [item for item in row['completion_items']
                                   if item['kind'] != 'answer_requirement']
        return row

    harness.children['a'].record = without_semantic_item
    evaluator.report_answer('goal_1=Yes', harness.boundary)
    row = evaluator.record(harness.boundary)
    assert row['required_unique_questions'] == 1 and row['answer_accuracy_percent'] is None
    assert row['answer_evaluation_validity'] == 'unavailable'


def definitions(*members):
    records, tasks, scenes = {}, {}, {}
    for item in members:
        key = item['id']
        records[key] = {'candidate_id': key, 'status': 'generated', 'big_task': 'Cognitive Mapping',
            'small_task': item['small_task'], 'scene_model': 'same_building',
            'source_identity': item['source_identity'], 'source_sha256': item['source_sha256'],
            'composition_source_raw': {'question': item['public']['objective'], 'answer': 'PRIVATE',
                                       'options_json': ['Yes', 'No']},
            'revision': 'old', 'converter_version': 'fixture'}
        tasks[key] = {'id': key, 'source': {'json': item['source_identity']},
            'public': {**item['public'], 'environment': 'Same world.', 'constraints': []},
            'private': {'bindings': {}, 'evaluation': deepcopy(item['evaluation'])},
            'scene': {'variants': {'original': {'patches': []}}},
            'defaults': {'seed': 0, 'max_steps': 99}, 'requirements': {'capabilities': []}}
        scenes[key] = {'scene': deepcopy(tasks[key]['scene']), 'seed': 0}
    return records, tasks, scenes


def test_identical_world_routes_always_independent_and_observation_changes_split_groups():
    records, tasks, scenes = definitions(member('a'), member('b', kind='Regional Boundary'),
        member('n1', kind='Long-Term Navigation'), member('n2', kind='Long-Term Navigation'),
        member('p1', kind='Traversable Passage'), member('p2', kind='Traversable Passage'))
    groups = plan_groups(list(records.values()), tasks, scenes)
    assert sorted(len(group['member_ids']) for group in groups) == [1, 1, 1, 1, 2]
    tasks['b']['observations'] = {'overview': 'map_assisted'}
    assert len(plan_groups(list(records.values()), tasks, scenes)) == 6


def test_strict_builder_requires_budget_proof_and_rejects_routes(tmp_path):
    records, tasks, scenes = definitions(member('a'), member('b', kind='Regional Boundary'))
    group = plan_groups(list(records.values()), tasks, scenes)[0]
    with pytest.raises(ValueError, match='budget'):
        build_group(group, records, tasks, scenes, tmp_path)
    group['episode_budget'] = {'max_steps': 120, 'wall_timeout_s': 600.,
                               'derivation': 'Shared visits plus two answers; at most 121 boundary observations.',
                               'evidence': 'cpu_fixture_budget_only_not_native_qualification'}
    result, task, scene = build_group(group, records, tasks, scenes, tmp_path)
    assert task['defaults']['max_steps'] == 120
    assert 'one final JSON answer sheet' in task['public']['success_criteria']
    assert 'separate observation boundaries' not in task['public']['success_criteria']
    assert scene == scenes[group['representative_id']]
    p = task['private']['evaluation']['parameters']
    assert len(p['semantic_questions']) == 2 and p['answer_sheet']
    tasks['b']['observations'] = {'overview': 'different'}
    with pytest.raises(ValueError, match='identical verified'):
        build_group(group, records, tasks, scenes, tmp_path)
    records['b']['small_task'] = 'Traversable Passage'
    with pytest.raises(ValueError, match='independent'):
        build_group(group, records, tasks, scenes, tmp_path)


def test_parser_preserves_compound_legacy_answer_and_rejects_nested_values():
    accepted, errors, structured = parse_answer_sheet(
        'goal_1=adjacency=No; reachability=unreachable', ['goal_1'])
    assert accepted == {'goal_1': 'adjacency=No; reachability=unreachable'}
    assert not errors and not structured
    assert parse_answer_sheet('{"goal_1":{"private":"x"}}', ['goal_1'])[0] == {}


def test_curated_inventory_builder_requires_full_explicit_world_contract(tmp_path):
    records, tasks, scenes = definitions(member('a'), member('b', kind='Regional Boundary'))
    group = plan_groups(list(records.values()), tasks, scenes)[0]
    group['episode_budget'] = {'max_steps': 100, 'wall_timeout_s': 600.,
                               'derivation': 'Fixture only.', 'evidence': 'not_native_qualification'}
    joint_world = {'task': deepcopy(tasks['a']), 'scene': deepcopy(scenes['a']), 'contract': {}}
    with pytest.raises(ValueError, match='Curated joint world'):
        build_group(group, records, tasks, scenes, tmp_path, joint_world=joint_world)
    joint_world['contract'] = {'compatibility_verified': True, 'member_ids': ['a', 'b'],
        'source_relations_revalidated': ['a', 'b'], 'reset_seed': 0,
        **{key: 'cpu_fixture_not_native_proof' for key in ('configuration_id', 'provenance',
            'world_state_id', 'start_state_id', 'phase_id', 'conditions_id',
            'robot_configuration_id', 'controller_id', 'observation_profile_id',
            'loaded_geometry_id', 'inventory_id')}}
    record, task, scene = build_group(group, records, tasks, scenes, tmp_path, joint_world=joint_world)
    assert task['private']['evaluation']['parameters']['joint_world_contract'] == joint_world['contract']
    assert 'composite_native_execution_unverified' in record['warning_codes']

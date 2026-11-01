"""Exercise original-goal checkpoint composition with isolated CPU reducers.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_composite_task.py
No native simulator, GPU, model, rendering or original evaluator is launched.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wia.authoring.converters import composite_task as composite
import wia.native.esi_category_evaluation as category_runtime


def member(key, terminal, *, bindings=None, **parameters):
    return {'id': key, 'source_identity': f'CPU Fixture/Goals/building/room_0/{key}.json',
            'public': {'objective': f'Reach {terminal} and stop for an observation.'},
            'bindings': bindings or {},
            'qa': {'answer': 'PRIVATE_QA_CANARY_' + key},
            'evaluation': {'type': 'category_operation', 'module': 'wia.authoring.converters.cpu_fixture',
                           'operation': key, 'binding_keys': list(bindings or {}),
                           'parameters': {'terminal': terminal, **parameters}}}


def observation(snapshot, step):
    return NS(snapshot_id=snapshot, action_step=step)


class Child:
    """Synthetic original reducer: samples/arrival declarations are independent."""
    def __init__(self, controller, config, output):
        self.controller, self.config, self.output = controller, config, Path(output)
        self.samples, self.records, self.arrivals = 0, [], []

    def begin_episode(self, boundary):
        if self.config['parameters'].get('blocked_begin'):
            raise ValueError('original native start is blocked')
        return self.record(boundary, action=None)

    def sample_step(self):
        self.samples += 1

    def report_arrival(self, room, boundary):
        self.arrivals.append((room, boundary.snapshot_id, boundary.action_step))
        return {'accepted': room == self.controller.terminal}

    def record(self, boundary, *, action=None):
        self.records.append((boundary.snapshot_id, boundary.action_step))
        p = self.config['parameters']
        row = {'schema': 'private_original_cpu_fixture', 'snapshot_id': boundary.snapshot_id,
               'action_step': boundary.action_step, 'task_success': bool(self.samples
                   and self.controller.terminal == p['terminal']),
               'measured_terminal': self.controller.terminal}
        row['continuous_constraints_satisfied'] = getattr(self.controller, 'safe', True)
        native_items = getattr(self.controller, 'native_items', {}).get(self.config['operation'])
        if native_items is not None:
            row['completion_items'] = deepcopy(native_items)
            row['task_success'] = all(item['completed'] and item['evaluation_validity'] == 'valid'
                                      for item in native_items)
        invalid_boundary = boundary.snapshot_id >= p.get('invalid_after', 0)
        if p.get('stale_snapshot') and invalid_boundary:
            row['snapshot_id'] -= 1
        if p.get('stale_action') and invalid_boundary:
            row['action_step'] -= 1
        if p.get('nonboolean_success') and invalid_boundary:
            row['task_success'] = 1
        if p.get('file_only'):
            self.output.mkdir(parents=True, exist_ok=True)
            (self.output / 'state.json').write_text(json.dumps(row))
            return None
        return row


@pytest.fixture
def harness(monkeypatch, tmp_path):
    objects = {'native_a': NS(name='native_a', category='cup', model='model_a'),
               'native_b': NS(name='native_b', category='plate', model='model_b')}
    scene = NS(object_registry=lambda field, name: objects.get(name))
    session = NS(scene_name='building', room_name='room_0', env=NS(scene=scene))
    controller, children, calls = NS(terminal='start'), {}, []

    def factory(session_arg, controller_arg, config, output):
        calls.append(deepcopy(config))
        assert session_arg is session and controller_arg is controller
        child = Child(controller, config, output)
        children[config['operation']] = child
        return child

    monkeypatch.setattr(category_runtime, 'create_category_operation_evaluator', factory)

    def create(members):
        config = {'type': 'category_operation', 'module': composite.MODULE,
                  'operation': 'composite_goals', 'parameters': {'members': members}}
        return composite.CompositeEvaluator(session, controller, config, tmp_path)

    return NS(create=create, controller=controller, children=children, calls=calls,
              session=session, output=tmp_path)


def assert_progress(row, expected, completed, total):
    assert row['completion_score'] == expected
    assert row['completed_goals'] == completed and row['total_goals'] == total
    assert len(row['goals']) == total
    assert row['task_success'] is (completed == total)
    assert all({'id', 'completed', 'blocked', 'objective', 'completion_evidence'} <= set(goal)
               for goal in row['goals'])


def test_incompatible_terminal_goals_latch_at_observed_checkpoints(harness):
    evaluator = harness.create([member('approach', 'object'), member('return', 'start')])
    assert_progress(evaluator.begin_episode(observation(1, 0)), 0., 0, 2)
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    # Physical samples alone do not create an observed checkpoint.
    assert not evaluator.completed
    first = evaluator.record(observation(2, 1), action='STOP')
    assert_progress(first, .5, 1, 2)
    proof = deepcopy(first['goals'][0]['completion_evidence'])
    first_child_samples = harness.children['approach'].samples
    first_child_records = len(harness.children['approach'].records)
    harness.controller.terminal = 'start'
    evaluator.sample_step()
    final = evaluator.record(observation(3, 2), action='RETURN')
    assert_progress(final, 1., 2, 2)
    assert final['goals'][0]['completion_evidence'] == proof
    assert proof['snapshot_id'] == 2 and proof['action_step'] == 1
    assert harness.children['approach'].samples == first_child_samples + 1
    assert len(harness.children['approach'].records) == first_child_records + 1
    assert json.loads((harness.output / 'state.json').read_text()) == final
    assert final['qa_execution_status'] == 'not_executed'
    harness.controller.safe = False
    unsafe = evaluator.record(observation(4, 3), action='UNSAFE')
    assert unsafe['completed_goals'] == 2 and unsafe['completion_score'] == 1.
    assert unsafe['goals'][0]['completion_evidence'] == proof
    assert all(goal['completion_evidence'] is not None for goal in unsafe['goals'])
    assert all(item['completed'] for item in unsafe['completion_items'])
    assert unsafe['continuous_constraints_satisfied'] and unsafe['task_success']


def test_revised_composite_rechecks_terminal_states(harness):
    revision = 'task_edit_20261018_v1'
    evaluator = harness.create([member('a', 'object', task_revision=revision),
                                member('b', 'start', task_revision=revision)])
    evaluator.begin_episode(observation(1, 0))
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    assert evaluator.record(observation(2, 1))['completion_score'] == .5
    harness.controller.terminal = 'start'
    evaluator.sample_step()
    current = evaluator.record(observation(3, 2))
    assert current['completion_score'] == .5 and not current['task_success']
    assert current['goals'][0]['completion_evidence'] is not None
    assert not current['goals'][0]['completed']


def test_composite_answers_have_one_explicit_public_destination(harness):
    evaluator = harness.create([member('a', 'object'), member('b', 'start')])
    received = []
    for key, child in harness.children.items():
        child.report_answer = lambda answer, observation, key=key: received.append((key, answer))
    evaluator.report_answer('goal_2=Yes', observation(1, 0))
    assert received == [('b', 'Yes')]
    with pytest.raises(ValueError, match='goal_N'):
        evaluator.report_answer('Yes', observation(1, 0))


def test_blocked_child_stays_in_denominator_while_other_completes(harness):
    evaluator = harness.create([member('blocked', 'object', blocked_begin=True,
                                        required_regions=['object']), member('good', 'object')])
    start = evaluator.begin_episode(observation(1, 0))
    assert start['completion_score'] is None and start['evaluation_validity'] == 'partial'
    assert start['completed_goals'] == 0 and start['total_goals'] == 2
    assert start['diagnostic_lower_bound'] == 0.
    assert 'original native start is blocked' in start['goals'][0]['blocked']
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    result = evaluator.record(observation(2, 1), action='STOP')
    assert result['completion_score'] is None and result['score_percent'] is None
    assert result['completed_goals'] == 1 and result['total_goals'] == 2
    assert result['diagnostic_lower_bound'] == .5
    assert not result['task_success']
    assert not result['goals'][0]['completed'] and result['goals'][1]['completed']


@pytest.mark.parametrize('invalid', ['stale_snapshot', 'stale_action', 'nonboolean_success'])
def test_stale_or_nonboolean_native_rows_never_earn_credit(harness, invalid):
    evaluator = harness.create([member('bad', 'object', invalid_after=4, **{invalid: True}), member('good', 'object')])
    assert_progress(evaluator.begin_episode(observation(3, 0)), 0., 0, 2)
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    row = evaluator.record(observation(4, 1), action='STOP')
    assert row['completion_score'] is None and row['total_goals'] is None
    assert not row['denominator_known'] and row['diagnostic_lower_bound'] is None
    assert row['goals'][1]['completed'] and not row['task_success']
    assert row['goals'][0]['blocked'] and not row['goals'][0]['completed']
    assert row['goals'][0]['completion_evidence'] is None


def test_file_only_original_reducer_is_accepted_only_at_current_boundary(harness):
    evaluator = harness.create([member('file', 'object', file_only=True)])
    evaluator.begin_episode(observation(1, 0))
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    assert_progress(evaluator.record(observation(2, 1), action='STOP'), 1., 1, 1)


def test_arrivals_fan_out_to_all_active_children_without_private_qa(harness):
    evaluator = harness.create([member('a', 'object'), member('b', 'start')])
    evaluator.begin_episode(observation(1, 0))
    evaluator.report_arrival('object', observation(1, 0))
    assert harness.children['a'].arrivals == harness.children['b'].arrivals == [('object', 1, 0)]
    assert not evaluator.completed
    assert 'PRIVATE_QA_CANARY' not in json.dumps(harness.calls)
    assert 'PRIVATE_QA_CANARY' not in (harness.output / 'state.json').read_text()
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    evaluator.record(observation(2, 1), action='STOP')
    evaluator.report_arrival('start', observation(3, 2))
    assert harness.children['a'].arrivals == [('object', 1, 0), ('start', 3, 2)]
    assert harness.children['b'].arrivals[-1] == ('start', 3, 2)


@pytest.mark.parametrize('definition', [
    {'name': 'missing', 'category': 'cup', 'model': 'model_a'},
    {'name': 'native_a', 'category': 'plate', 'model': 'model_a'},
    {'name': 'native_a', 'category': 'cup', 'model': 'wrong_model'},
])
def test_exact_binding_mismatch_is_explicitly_blocked(harness, definition):
    evaluator = harness.create([member('bad', 'object', bindings={'target': definition}), member('good', 'object')])
    evaluator.begin_episode(observation(1, 0))
    harness.controller.terminal = 'object'
    evaluator.sample_step()
    row = evaluator.record(observation(2, 1), action='STOP')
    assert row['completion_score'] is None and row['total_goals'] is None
    assert not row['denominator_known'] and row['diagnostic_lower_bound'] is None
    assert row['goals'][1]['completed'] and not row['task_success']
    assert 'Original member binding is unavailable: target' in row['goals'][0]['blocked']
    assert 'bad' not in harness.children


def test_member_config_preserves_original_bindings_and_rejects_recursive_members(harness):
    goal = member('bound', 'object', bindings={'original_key': {
        'name': 'native_a', 'category': 'cup', 'model': 'model_a'}})
    before = deepcopy(goal)
    config, bound = composite.member_config(harness.session, goal)
    assert goal == before
    assert config['bindings'] == {'original_key': 'native_a'}
    assert config['binding_keys'] == ['original_key']
    assert bound['original_key'].model == 'model_a'
    assert config['scene'] == 'building' and config['room'] == 'room_0'
    goal['evaluation']['module'] = composite.MODULE
    with pytest.raises(ValueError, match='nonrecursive'):
        composite.member_config(harness.session, goal)


def test_qualification_reports_each_child_and_allows_useful_partial_operation(harness, monkeypatch):
    def qualify(session, controller, config, bound):
        return {'accepted': config['operation'] == 'good', 'reason': 'native CPU fixture'}
    monkeypatch.setattr(category_runtime, 'qualify_category_operation', qualify)
    config = {'parameters': {'members': [member('bad', 'object'), member('good', 'object')]}}
    result = composite.qualify_operation(harness.session, harness.controller, config, {})
    assert result['accepted'] and not result['all_members_qualified']
    assert [row['accepted'] for row in result['members']] == [False, True]
    config['parameters']['members'] = [member('bad', 'object')]
    assert not composite.qualify_operation(harness.session, harness.controller, config, {})['accepted']


def test_identical_unordered_room_obligations_share_credit_and_retain_sources(harness):
    goals = [member(key, 'unused', required_regions=['room_a_0', 'room_b_0'],
                    required_order=[], arrival_report_required=True, stability_steps=3,
                    floor_idx=0, start_region='room_a_0') for key in ('first_source', 'second_source')]
    for goal in goals:
        goal['evaluation']['operation'] = 'identical_room_arrivals'
    harness.controller.native_items = {'identical_room_arrivals': [
        {'id': 'room:room_a_0', 'completed': True, 'evaluation_validity': 'valid',
         'kind': 'historical_checkpoint'},
        {'id': 'room:room_b_0', 'completed': False, 'evaluation_validity': 'valid',
         'kind': 'historical_checkpoint'}]}
    evaluator = harness.create(goals)
    row = evaluator.begin_episode(observation(1, 0))
    assert row['total_goals'] == 2 and row['completed_goals'] == 1
    assert row['completion_score'] == .5 and row['source_member_count'] == 2
    assert len(row['completion_items']) == 2 and len(row['goals']) == 2
    for item in row['completion_items']:
        assert {source['member_id'] for source in item['sources']} == {'first_source', 'second_source'}
        assert {source['source_identity'] for source in item['sources']} == {
            goal['source_identity'] for goal in goals}
        assert len({source['item_id'] for source in item['sources']}) == 1


def test_native_child_progress_propagates_and_invalid_item_keeps_denominator(harness):
    harness.controller.native_items = {'survey': [
        {'id': f'room:room_{index}_0', 'completed': index < 4,
         'evaluation_validity': 'valid', 'kind': 'historical_checkpoint'}
        for index in range(5)]}
    evaluator = harness.create([member('survey', 'unused')])
    row = evaluator.begin_episode(observation(1, 0))
    assert row['completed_goals'] == 4 and row['total_goals'] == 5
    assert row['completion_score'] == .8 and row['score_percent'] == 80.
    assert row['evaluation_validity'] == 'complete' and not row['task_success']
    assert row['goals'][0]['progress']['completion_score'] == .8
    harness.controller.native_items['survey'][4].update(completed=True, evaluation_validity='unavailable')
    partial = evaluator.record(observation(2, 1), action='MEASUREMENT_UNAVAILABLE')
    assert partial['completion_score'] is None and partial['score_percent'] is None
    assert partial['evaluation_validity'] == 'partial' and not partial['task_success']
    assert partial['completed_goals'] == 4 and partial['total_goals'] == 5
    assert partial['diagnostic_lower_bound'] == .8
    assert partial['goals'][0]['progress']['evaluation_validity'] == 'partial'

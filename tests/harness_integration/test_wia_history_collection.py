"""CPU-only historical collection contracts; fixtures do not qualify native physics.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_history_collection.py
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from wia.authoring.converters.wia_history_collection import HistoryCollectionEvaluator, EMPTY_GRIPPER
from wia.tasks.wia_history_collection import seeded_collection_query, public_collection_query


def parameters():
    return {'target_keys': list('abcdef'),
        'public_target_labels': {key: 'item ' + key for key in 'abcdef'},
        'target_categories': {'a': 'cups', 'b': 'cups', 'c': 'cups',
                              'd': 'boxes', 'e': 'boxes', 'f': 'boxes'},
        'followup_seed': 7, 'query_rule': 'earliest_per_category',
        'query_min_total': 3, 'query_max_total': 3,
        'destination_key': 'tray', 'destination_label': 'the collection tray',
        'placement_relation': 'on_top', 'release_stability_steps': 2,
        'maximum_release_speed_m_s': .02, 'maximum_release_angular_speed_rad_s': .1}


def history(order='abcdef'):
    return [{'key': key, 'snapshot_id': index, 'action_step': index} for index, key in enumerate(order)]


def obs(index):
    return NS(snapshot_id=index, action_step=index, rgb_images=lambda: {'head': []})


def receipt(key, index):
    return {'native_name': 'native_' + key, 'snapshot_id': index, 'camera': 'head',
        'native_rgb_matches_delivered': True, 'identity_synchronized': True, 'informative': True,
        'informative_feature_evidence': {'shape': 'fixture'}, 'visit_verified': True,
        'visit_evidence': {'snapshot_id': index, 'legal_observation_pose': True}}


@pytest.fixture
def replay(tmp_path):
    p = parameters()
    config = {'operation': 'history_multi_retrieval_v1', 'parameters': p,
        'bindings': {key: 'native_' + key for key in (*'abcdef', 'tray')},
        'binding_keys': [*'abcdef', 'tray']}
    visual = {'available': True, 'objects': {}}
    grasp = deepcopy(EMPTY_GRIPPER)
    session = NS(task_visual_evidence=lambda *args: deepcopy(visual))
    evaluator = HistoryCollectionEvaluator(session, NS(debug_gripper_state=lambda: deepcopy(grasp)), config, tmp_path)
    inventory = {key: {'name': 'native_' + key, 'category': 'fixture', 'model': 'original',
        'position': [index, 0., 1.], 'orientation': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}
        for index, key in enumerate((*'abcdef', 'tray'))}
    placed = set()
    evaluator._protected = lambda: deepcopy(inventory)
    evaluator._placement = lambda key: {'gripper': deepcopy(grasp), 'destination_relation': key in placed,
        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'robot_withdrawn': True,
        'robot_terminal_safe': True, 'target_dynamic_collidable': True}
    evaluator.begin_episode(obs(0))
    return evaluator, session, visual, grasp, inventory, placed


def survey(replay):
    evaluator, _, visual, *_ = replay
    for index, key in enumerate('abcdef', 1):
        visual['objects'] = {key: receipt(key, index)}
        evaluator.record(obs(index))
    return evaluator.complete_survey(obs(6))


def hold(grasp, key):
    grasp.update(holding=True, constraint_active=True, release_pending=False, object_name='native_' + key)


def release(grasp):
    grasp.update(EMPTY_GRIPPER)


def deliver(evaluator, grasp, placed, key):
    hold(grasp, key)
    evaluator.sample_step()
    release(grasp)
    placed.add(key)
    evaluator.sample_step()
    evaluator.sample_step()


def test_delayed_inventory_seed_is_deterministic_and_actual_order_changes_identity():
    p = parameters()
    q = seeded_collection_query(history(), p)
    assert q == seeded_collection_query(history(), deepcopy(p))
    other = seeded_collection_query(history('cbafed'), p)
    assert q['requests'] == other['requests']
    assert q['selected_keys'] != other['selected_keys']
    assert len(set(q['selected_keys'])) == q['total_count'] == 3
    assert q['require_history_subset'] is True
    assert all(row['count'] < 3 for row in q['requests'])
    public = public_collection_query(q, 'the tray')
    assert set(public) == {'rule', 'requests', 'total_count', 'text'}
    assert all('item ' + key not in public['text'] for key in 'abcdef')


def test_seeded_counts_fit_declared_bounds_and_select_nonempty_category_subset():
    p = parameters()
    p.update(query_min_total=1, query_max_total=4)
    totals, subsets = set(), set()
    for seed in range(100):
        p['followup_seed'] = seed
        query = seeded_collection_query(history(), p)
        totals.add(query['total_count'])
        subsets.add(len(query['requests']))
        assert 1 <= query['total_count'] <= 4
        assert sum(row['count'] for row in query['requests']) == len(set(query['selected_keys']))
        assert all(1 <= row['count'] <= 2 for row in query['requests'])
    assert totals == {1, 2, 3, 4} and subsets == {1, 2}


def test_current_three_category_inventory_has_no_full_category_shortcut_across_many_seeds():
    p = parameters()
    p.update(target_keys=list('abcdefghi'),
        public_target_labels={key: 'item ' + key for key in 'abcdefghi'},
        target_categories={key: category for category, keys in
            (('cups', 'abc'), ('boxes', 'def'), ('bottles', 'ghi')) for key in keys},
        require_history_subset=True, query_min_total=1, query_max_total=3)
    # One instance of every category is first observed at each of three stations.
    rows = [{'key': key, 'snapshot_id': index // 3 + 1, 'action_step': index // 3 + 1}
            for index, key in enumerate('adgbehcfi')]
    totals, counts = set(), set()
    for seed in range(500):
        p['followup_seed'] = seed
        query = seeded_collection_query(rows, p)
        totals.add(query['total_count'])
        assert query['require_history_subset'] is True
        assert 1 <= query['total_count'] <= 3
        assert len(set(query['selected_keys'])) == query['total_count']
        for request in query['requests']:
            counts.add(request['count'])
            inventory_count = sum(category == request['category'] for category in p['target_categories'].values())
            assert 1 <= request['count'] < inventory_count == 3
    assert totals == {1, 2, 3} and counts == {1, 2}


def test_subset_default_caps_bounds_and_explicit_false_is_auditable_control():
    p = parameters()
    default = seeded_collection_query(history(), p)
    p['require_history_subset'] = True
    assert seeded_collection_query(history(), p) == default
    p.update(query_min_total=5, query_max_total=5)
    with pytest.raises(ValueError, match='request caps'):
        seeded_collection_query(history(), p)
    p.update(require_history_subset=False, query_min_total=6, query_max_total=6)
    control = seeded_collection_query(history(), p)
    assert control['require_history_subset'] is False
    assert control['total_count'] == 6 and all(row['count'] == 3 for row in control['requests'])


@pytest.mark.parametrize('flag', [None, 0, 1, 'true'])
def test_subset_requirement_requires_explicit_boolean(flag):
    p = parameters()
    p['require_history_subset'] = flag
    with pytest.raises(ValueError, match='boolean'):
        seeded_collection_query(history(), p)


def test_ties_are_category_specific_and_query_specific():
    p = parameters()
    rows = history()
    rows[3]['snapshot_id'] = rows[0]['snapshot_id']
    # Different categories may first appear in the same delivered snapshot.
    seeded_collection_query(rows, p)
    rows = history()
    rows[1]['snapshot_id'] = rows[0]['snapshot_id']
    with pytest.raises(ValueError, match='Ambiguous'):
        seeded_collection_query(rows, p)
    p.update(query_min_total=1, query_max_total=1)
    for seed in range(100):
        p['followup_seed'] = seed
        query = seeded_collection_query(history(), p)
        if query['requests'][0]['category'] == 'boxes':
            assert seeded_collection_query(rows, p)['requests'] == query['requests']
            break
    else:
        pytest.fail('Expected a seed querying the unrelated category.')


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'bounds', 'rule', 'chronology'])
def test_incomplete_or_invalid_query_contract_rejected(failure):
    p, rows = parameters(), history()
    if failure == 'missing': rows.pop()
    elif failure == 'duplicate': rows[1]['key'] = rows[0]['key']
    elif failure == 'bounds': p['query_max_total'] = 5
    elif failure == 'rule': p['query_rule'] = 'sort_native_names'
    else: rows.reverse()
    with pytest.raises(ValueError):
        seeded_collection_query(rows, p)


def test_followup_is_delayed_and_public_transport_never_exposes_selection(replay):
    evaluator, session, _, _, _, _ = replay
    assert evaluator.frozen is None and not hasattr(session, 'public_task_events')
    with pytest.raises(ValueError, match='every target'):
        evaluator.complete_survey(obs(0))
    survey(replay)
    event, = session.public_task_events
    assert event['kind'] == 'history_followup'
    assert set(event) == {'event_id', 'kind', 'text', 'action_step', 'snapshot_id'}
    from core.observations import PublicTaskEvent
    assert PublicTaskEvent(**event).to_dict() == event
    for request in evaluator.public_query['requests']:
        assert f"the first {request['count']} {request['category']} you inspected" in event['text']
    encoded = json.dumps(event)
    assert 'native_' not in encoded and 'selected_keys' not in encoded
    assert not any('item ' + key in encoded for key in 'abcdef')
    with pytest.raises(ValueError, match='already'):
        evaluator.complete_survey(obs(4))


def test_survey_coverage_and_query_emission_are_separate_stages(replay):
    evaluator, _, visual, *_ = replay
    for index, key in enumerate('abcdef', 1):
        visual['objects'] = {key: receipt(key, index)}
        result = evaluator.record(obs(index))
    stages = {item['id']: item['completed'] for item in result['completion_items']}
    assert result['observed_target_count'] == result['declared_target_count'] == 6
    assert stages['survey'] and not stages['delayed_history_query']
    assert not stages['requested_native_identities'] and not stages['all_requested_items_delivered']


@pytest.mark.parametrize('violation', ['move', 'grasp'])
def test_prequery_target_movement_and_grasp_are_permanent_violations(replay, violation):
    evaluator, _, _, grasp, inventory, _ = replay
    if violation == 'move': inventory['a']['position'][0] += .2
    else: hold(grasp, 'a')
    evaluator.sample_step()
    release(grasp)
    with pytest.raises(ValueError, match='preservation'):
        survey(replay)
    assert evaluator.constraint_violation


def test_sequential_deliveries_remain_valid_while_holding_next_requested_item(replay):
    evaluator, _, _, grasp, _, placed = replay
    survey(replay)
    first, second, third = evaluator.selected_keys
    deliver(evaluator, grasp, placed, first)
    assert evaluator.delivery_status[first]
    hold(grasp, second)
    evaluator.sample_step()
    assert evaluator.delivery_status[first]
    assert evaluator.placements[first].last['actual_gripper']['object_name'] == 'native_' + second
    assert not evaluator.record(obs(5))['task_success']
    release(grasp)
    placed.add(second)
    evaluator.sample_step(); evaluator.sample_step()
    deliver(evaluator, grasp, placed, third)
    result = evaluator.record(obs(6))
    assert result['task_success'] and result['delivered_count'] == 3
    assert result['reported_answer'] is None
    assert all(item['completed'] for item in result['completion_items'])


def test_repeated_identity_cannot_replace_missing_distinct_delivery(replay):
    evaluator, _, _, grasp, _, placed = replay
    survey(replay)
    first = evaluator.selected_keys[0]
    for _ in range(3):
        deliver(evaluator, grasp, placed, first)
    evaluator.report_answer('all requested objects delivered', obs(5))
    result = evaluator.record(obs(5))
    assert result['delivered_count'] == 1 and not result['task_success']
    assert sum(tracker.selected for tracker in evaluator.placements.values()) == 1


def test_wrong_identity_and_nonselected_motion_fail_constraints(replay):
    evaluator, _, _, grasp, inventory, _ = replay
    survey(replay)
    wrong = next(key for key in 'abcdef' if key not in evaluator.selected_keys)
    hold(grasp, wrong)
    evaluator.sample_step()
    release(grasp)
    inventory[wrong]['position'][0] += .2
    result = evaluator.record(obs(5))
    assert result['constraints_satisfied'] is False and not result['task_success']


def test_removal_revokes_earlier_delivery_and_score_reads_never_advance_dwell(replay):
    evaluator, _, _, grasp, _, placed = replay
    survey(replay)
    for key in evaluator.selected_keys:
        deliver(evaluator, grasp, placed, key)
    assert evaluator.record(obs(5))['task_success']
    removed = evaluator.selected_keys[0]
    placed.remove(removed)
    result = evaluator.record(obs(6))
    assert result['delivered_count'] == 2 and not result['task_success']
    assert evaluator.placements[removed].stable_steps == 0
    placed.add(removed)
    for _ in range(5):
        assert not evaluator.record(obs(6))['task_success']
    assert evaluator.placements[removed].stable_steps == 0
    evaluator.sample_step()
    assert not evaluator.record(obs(7))['task_success']
    evaluator.sample_step()
    assert evaluator.record(obs(8))['task_success']


def test_missing_grasp_and_unsafe_terminal_state_never_pass(replay):
    evaluator, _, _, grasp, _, placed = replay
    survey(replay)
    for key in evaluator.selected_keys:
        deliver(evaluator, grasp, placed, key)
    original = evaluator._placement
    evaluator._placement = lambda key: {**original(key), 'robot_terminal_safe': False}
    assert not evaluator.record(obs(5))['task_success']
    evaluator._placement = original
    grasp.pop('object_name')
    result = evaluator.record(obs(6))
    assert not result['task_success'] and result['constraints_satisfied'] is None


def test_dispatch_is_explicit_and_legacy_dispatch_unchanged(replay, tmp_path):
    from wia.authoring.converters import temporal_understanding_agent_observation as legacy
    evaluator, session, *_ = replay
    config = evaluator.config
    assert isinstance(legacy.create_evaluator(session, evaluator.controller, config, tmp_path), HistoryCollectionEvaluator)
    config = deepcopy(config)
    config['operation'] = 'survey_history_retrieve_place'
    assert isinstance(legacy.create_evaluator(session, evaluator.controller, config, tmp_path), legacy.HistoryRetrievalEvaluator)

"""Validate versioned temporal history/action semantics with controlled CPU evidence.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m pytest \
  -q -p no:cacheprovider tests/test_task_edit2_temporal.py tests/test_native_temporal_phases.py \
  tests/test_temporal_phase_transport.py tests/test_agent_observation.py
Fixtures are not native physical or visual qualification.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wia.authoring.conversion import ConversionContext
from wia.authoring.converters import temporal_understanding_agent_observation as history
from wia.authoring.converters import temporal_understanding_two_phase as change
from wia.authoring.converters import temporal_understanding_unobserved_change as source_change
from wia.tasks.temporal_policy import informative_temporal_rows
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from wia.runtime.commands.build_task_edit2_temporal_candidates import (
    HISTORY_SOURCE, CHANGE_SOURCES, source, history_config, change_config)


def obs(index):
    return NS(snapshot_id=index, action_step=index, rgb_images=lambda: {'head': [], 'wrist': []})


def receipt(key, index, **changes):
    row = {'native_name': 'native_' + key, 'snapshot_id': index, 'camera': 'head',
        'native_rgb_matches_delivered': True, 'identity_synchronized': True, 'informative': True,
        'informative_feature_evidence': {'diagnostic_shape': 'native fixture evidence'},
        'visit_verified': True, 'visit_evidence': {'snapshot_id': index, 'legal_observation_pose': True}}
    row.update(changes)
    return row


def replay(tmp_path):
    p = {'target_keys': ['a', 'b', 'c'], 'public_target_labels': {'a': 'red station target',
        'b': 'green station target', 'c': 'blue station target'}, 'followup_seed': 7,
        'destination_key': 'tray', 'destination_label': 'the delivery tray', 'placement_relation': 'on_top',
        'release_stability_steps': 3, 'maximum_release_speed_m_s': .02,
        'maximum_release_angular_speed_rad_s': .1}
    config = {'parameters': p, 'bindings': {k: 'native_' + k for k in ('a', 'b', 'c', 'tray')},
              'binding_keys': ['a', 'b', 'c', 'tray']}
    visual = {'available': True, 'objects': {}}
    placement = {'gripper': {'holding': False, 'constraint_active': False, 'release_pending': False,
        'object_name': None}, 'destination_relation': False, 'linear_speed_m_s': 0.,
        'angular_speed_rad_s': 0., 'robot_withdrawn': True, 'robot_terminal_safe': True,
        'target_dynamic_collidable': True}
    session = NS(task_visual_evidence=lambda *args: deepcopy(visual),
                 observation_source=NS(task_text='Immutable original mission'))
    controller = NS(debug_gripper_state=lambda: placement['gripper'])
    evaluator = history.HistoryRetrievalEvaluator(session, controller, config, tmp_path)
    inventory = {k: {'name': 'native_' + k, 'category': 'mug', 'model': 'original',
        'position': [i, 0., 1.], 'orientation': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}
        for i, k in enumerate(('a', 'b', 'c', 'tray'))}
    evaluator._protected = lambda: deepcopy(inventory)
    evaluator._placement = lambda: deepcopy(placement)
    evaluator.begin_episode(obs(0))
    return evaluator, session, visual, placement, inventory


def survey(evaluator, visual, order=('a', 'b', 'c')):
    for i, key in enumerate(order, 1):
        visual['objects'] = {key: receipt(key, i)}
        evaluator.record(obs(i))
    evaluator.report_answer('SURVEY_COMPLETE', obs(3))


def test_distinct_actual_histories_produce_different_history_bound_targets(tmp_path):
    selected = []
    for i, order in enumerate((('a', 'b', 'c'), ('b', 'c', 'a'))):
        evaluator, session, visual, _, _ = replay(tmp_path / str(i))
        survey(evaluator, visual, order)
        selected.append(evaluator.selected_key)
        assert evaluator.question_rank == 2
        assert session.observation_source.task_text == 'Immutable original mission'
        event, = session.public_task_events
        assert set(event) == {'event_id', 'kind', 'text', 'snapshot_id', 'action_step'}
        assert event['kind'] == 'history_followup'
        assert 'number 2' in event['text']
        assert all('native_' + k not in event['text'] for k in ('a', 'b', 'c'))
        assert evaluator.parameters['public_target_labels'][evaluator.selected_key] not in event['text']
    assert selected == ['b', 'c']


@pytest.mark.parametrize('rule,orders,reference,expected', [
    ('before_reference_encounter', [('a', 'b', 'c'), ('b', 'a', 'c')], 'c', ['b', 'a']),
    ('after_reference_encounter', [('a', 'b', 'c'), ('c', 'b', 'a')], 'b', ['c', 'a'])])
def test_relative_delivery_same_question_and_layout_need_different_actual_histories(
        tmp_path, rule, orders, reference, expected):
    selected, questions = [], []
    for index, order in enumerate(orders):
        evaluator, session, visual, _, _ = replay(tmp_path / str(index))
        evaluator.parameters['followup_rule'] = rule
        survey(evaluator, visual, order)
        selected.append(evaluator.selected_key)
        question = session.public_task_events[0]['text']
        questions.append(question)
        labels = evaluator.parameters['public_target_labels']
        assert labels[reference] in question
        assert labels[evaluator.selected_key] not in question
    assert selected == expected
    assert questions[0] == questions[1]


def test_simultaneous_camera_sightings_are_ambiguous_and_repeats_are_deduplicated(tmp_path):
    evaluator, _, visual, _, _ = replay(tmp_path)
    visual['objects'] = {'a': receipt('a', 1), 'b': receipt('b', 1)}
    evaluator.record(obs(1))
    evaluator.record(obs(1))
    visual['objects'] = {'c': receipt('c', 2)}
    evaluator.record(obs(2))
    assert len(evaluator.history) == 3
    assert len(evaluator.ties) == 1
    with pytest.raises(ValueError, match='ambiguous'):
        evaluator.complete_survey(obs(2))


def test_survey_does_not_freeze_incomplete_history_or_allow_preselected_answer(tmp_path):
    evaluator, _, visual, _, _ = replay(tmp_path)
    visual['objects'] = {'a': receipt('a', 1)}
    evaluator.record(obs(1))
    with pytest.raises(ValueError, match='every target'):
        evaluator.complete_survey(obs(1))
    with pytest.raises(ValueError, match='before survey'):
        evaluator.report_answer('red station target', obs(1))


@pytest.mark.parametrize('changes', [{'informative': False}, {'informative_feature_evidence': {}},
    {'identity_synchronized': False}, {'native_rgb_matches_delivered': False}, {'snapshot_id': 0},
    {'camera': 'human_global'}, {'visit_verified': False}, {'visit_evidence': {}},
    {'visit_evidence': {'snapshot_id': 1, 'legal_observation_pose': False}}])
def test_distance_only_or_uninformative_native_fragments_never_establish_encounters(tmp_path, changes):
    evaluator, _, visual, _, _ = replay(tmp_path)
    visual['objects'] = {'a': receipt('a', 1, **changes)}
    result = evaluator.record(obs(1))
    assert not evaluator.history
    assert not result['task_success']


def test_missing_visual_callback_remains_unavailable(tmp_path):
    evaluator, session, _, _, _ = replay(tmp_path)
    del session.task_visual_evidence
    result = evaluator.record(obs(1))
    assert result['completion_score'] is None
    assert result['completion_items'][0]['evaluation_validity'] == 'unavailable'
    assert informative_temporal_rows(session, obs(1), {}, {})['available'] is False


def test_correct_history_answer_requires_correct_native_selection_and_stable_terminal_release(tmp_path):
    evaluator, _, visual, placement, _ = replay(tmp_path)
    survey(evaluator, visual)
    assert evaluator.selected_key == 'b'
    evaluator.report_answer('green station target', obs(3))
    placement['gripper'].update(holding=True, constraint_active=True, object_name='native_a')
    evaluator.sample_step()
    assert not evaluator.placement.selected
    placement['gripper']['object_name'] = 'native_b'
    evaluator.sample_step()
    assert evaluator.placement.selected
    placement['destination_relation'] = True
    for _ in range(3):
        evaluator.sample_step()
    assert evaluator.placement.stable_steps == 0
    placement['gripper'].update(holding=False, constraint_active=False, object_name=None)
    for _ in range(20):
        evaluator.record(obs(3))
    assert evaluator.placement.stable_steps == 0
    for _ in range(3):
        evaluator.sample_step()
    result = evaluator.record(obs(3))
    assert result['task_success']
    placement['destination_relation'] = False
    result = evaluator.record(obs(3))
    assert not result['task_success']
    assert result['completion_items'][2]['completed']


def test_wrong_history_answer_and_changed_distractor_cannot_complete(tmp_path):
    evaluator, _, visual, placement, inventory = replay(tmp_path)
    survey(evaluator, visual)
    evaluator.report_answer('red station target', obs(3))
    placement['gripper'].update(holding=True, constraint_active=True, object_name='native_b')
    evaluator.sample_step()
    placement['gripper'].update(holding=False, constraint_active=False, object_name=None)
    placement['destination_relation'] = True
    for _ in range(3):
        evaluator.sample_step()
    result = evaluator.record(obs(3))
    assert not result['task_success']
    assert not result['completion_items'][1]['completed']
    evaluator.report_answer('green station target', obs(3))
    inventory['a']['position'][0] += .2
    result = evaluator.record(obs(3))
    assert result['constraints_satisfied'] is False
    assert not result['task_success']


def test_missing_placement_measurement_and_survey_grasp_are_not_credited(tmp_path):
    evaluator, _, visual, placement, _ = replay(tmp_path)
    placement['gripper'].update(holding=True, constraint_active=True, object_name='native_a')
    evaluator.sample_step()
    assert evaluator.constraint_violation
    evaluator, _, visual, placement, _ = replay(tmp_path / 'second')
    survey(evaluator, visual)
    placement['destination_relation'] = None
    result = evaluator.record(obs(3))
    assert result['completion_items'][3]['evaluation_validity'] == 'unavailable'
    assert result['completion_score'] is None


def test_source_supported_branch_definitions_and_public_spare_distractors(tmp_path):
    for branch, identity in CHANGE_SOURCES.items():
        src = source(identity, 'Unobserved Change')
        layout = source_change._layout(src)
        assert source_change.source_change_recipe(layout)['branch'] == branch
        context = ConversionContext(tmp_path, seed=7, config=change_config(src))
        result = change.CategoryConverter().convert(src, context)
        task, scene = result['task_definition'], result['scene_definition']
        task_definition(task)
        SceneDefinition.model_validate(scene)
        p = task['private']['evaluation']['parameters']
        assert p['requires_native_visual_discovery'] is True
        assert scene['runtime']['requires_native_visual_discovery'] is True
        assert task['private']['evaluation']['operation'] == change.PHYSICAL_OPERATION
        assert p['physical_branch'] == branch
        assert p['phase_2_navigation_required'] is False
        assert 'phase_2_minimum_base_travel_m' not in p
        assert 'phase_2_container_visit_radius_m' not in p
        assert result['execution_status'] == 'unavailable'
        assert '0.25 m' not in task['public']['objective']
        if branch == 'removed':
            assert len(p['public_spare_inventory']) == 3
            native_target = task['private']['bindings'][p['physical_target_key']]
            original = layout['bindings'][p['source_event_recipe']['original_key']]
            assert all(native_target[k] == original[k] for k in ('category', 'model'))
            assert p['physical_target_key'] in {x['name'] for x in p['apparatus']['dataset_objects']}
            assert p['physical_destination_key'] == p['source_event_recipe']['source_container']
        elif branch == 'no_change':
            assert p['physical_target_key'] is None
        else:
            assert p['physical_target_key'] in layout['phase_keys']['image2']
        # Public recipes state all branches without exposing selected binding identities.
        for private_name in (binding['name'] for binding in layout['bindings'].values()):
            assert private_name not in task['public']['objective']


def test_history_real_source_apparatus_and_schema_are_concrete(tmp_path):
    src = source(HISTORY_SOURCE, 'Agent Observation')
    result = history.CategoryConverter().convert(src, ConversionContext(tmp_path, 7, history_config(src)))
    task_definition(result['task_definition'])
    SceneDefinition.model_validate(result['scene_definition'])
    p = result['task_definition']['private']['evaluation']['parameters']
    assert len(p['target_keys']) == 3
    assert p['requires_native_visual_discovery']
    assert result['scene_definition']['runtime']['requires_native_visual_discovery']
    native_objects = [{'type': 'DatasetObject', 'name': row['_key'], 'category': row['category'], 'model': row['model']}
                      for row in src.payload['objects_meta']]
    config = {'scene': {'scene_model': src.raw['scene']}, 'objects': native_objects}
    history.configure_scene(config, {'task_revision': p['task_revision'], 'scene_model': src.raw['scene'],
                                   'apparatus': p['apparatus']})
    names = {row['name'] for row in config['objects']}
    assert set(p['apparatus']['target_support_bindings'].values()).issubset(names)
    assert p['destination_key'] in names
    assert all(next(x for x in config['objects'] if x['name'] == k)['fixed_base'] is False for k in p['target_keys'])


def test_release_pending_is_required_but_robot_safety_is_independent():
    tracker = history.NativePlacementProgress({'release_stability_steps': 2,
        'maximum_release_speed_m_s': .02, 'maximum_release_angular_speed_rad_s': .1})
    state = {'gripper': {'holding': False, 'constraint_active': False, 'release_pending': True, 'object_name': None},
        'destination_relation': True, 'linear_speed_m_s': 0., 'angular_speed_rad_s': 0.,
        'robot_withdrawn': True, 'robot_terminal_safe': True, 'target_dynamic_collidable': True}
    for _ in range(3):
        assert tracker.update(state, 'target', count_step=True) == (True, False)
    state['gripper']['release_pending'] = False
    state['robot_terminal_safe'] = None
    assert tracker.update(state, 'target', count_step=True) == (True, False)
    assert tracker.update(state, 'target', count_step=True) == (True, True)


def test_moved_recipe_requires_real_matching_source_event():
    layout = {'container_keys': ['left', 'right'], 'phase_keys': {'image1': ['before'], 'image2': ['after']},
        'relations': {'before': 'left', 'after': 'right'},
        'bindings': {'before': {'category': 'mug', 'model': 'same'}, 'after': {'category': 'mug', 'model': 'same'}}}
    recipe = source_change.source_change_recipe(layout)
    assert recipe['branch'] == 'moved'
    assert recipe['source_container'] == 'left'
    assert recipe['current_key'] == 'after'


def test_physical_branch_cannot_be_relabelled_to_avoid_required_manipulation():
    from wia.native.esi_temporal_phases import measured_change_branch
    mug = {'category': 'mug', 'model': 'original'}
    pear = {'category': 'pear', 'model': 'second'}
    assert measured_change_branch({'a': None}, {'a': mug}) == 'added'
    assert measured_change_branch({'a': mug}, {'a': None}) == 'removed'
    assert measured_change_branch({'a': mug}, {'a': pear}) == 'replaced'
    assert measured_change_branch({'a': mug}, {'a': mug}) == 'no_change'
    assert measured_change_branch({'a': mug, 'b': None}, {'a': None, 'b': mug}) == 'moved'

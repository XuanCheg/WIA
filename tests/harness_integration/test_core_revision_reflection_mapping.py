"""Verify revised Reflection Authoring semantics with native-shaped CPU fixtures.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_core_revision_reflection_mapping.py
These checks do not establish native sensor alignment, scaling or mobility.
"""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.authoring.conversion import ROOT, ConversionContext, SourceInstance
from wia.authoring.converters import specular_reflection_reflection_authoring as authoring
from wia.authoring.converters import specular_reflection_correspondence as correspondence
from wia.authoring.converters import specular_reflection_spatial_relations as spatial
from wia.tasks.completion import progress
from wia.authoring.converters import cognitive_mapping_connectivity as connectivity
from wia.authoring.converters import cognitive_mapping_long_term_navigation as navigation
from wia.authoring.converters import cognitive_mapping_regional_boundary as regional
from wia.authoring.converters import cognitive_mapping_traversable_passage as passage
from wia.authoring.converters.cognitive_mapping_visit_reports import (
    SourceAnswerMappingEvaluator, mapping_qualification, queried_portal_crossing)


def sources():
    root = ROOT / 'ESI/dataset/json_clean'
    return [SourceInstance(path, str(path.relative_to(root)), *authoring.CategoryConverter.category)
            for path in sorted(root.joinpath(*authoring.CategoryConverter.category).rglob('*.json'))]


def converted(tmp_path, **config):
    return authoring.CategoryConverter().convert(sources()[0], ConversionContext(tmp_path, config=config))


def world(tmp_path, **config):
    result = converted(tmp_path, **config)
    evaluation = deepcopy(result['task_definition']['private']['evaluation'])
    source = sources()[0]
    evaluation.update(scene=source.raw['scene'], room=source.raw['room'],
        bindings={key: value['name'] for key, value in result['task_definition']['private']['bindings'].items()})
    p = evaluation['parameters']
    objects = {}
    for key, pose in p['expected_optical_poses'].items():
        obj = NS(position=np.array(pose['position']), quaternion=np.array(pose['quaternion_xyzw']))
        obj.get_position_orientation = lambda obj=obj: (obj.position.copy(), obj.quaternion.copy())
        objects[evaluation['bindings'][key]] = obj
    objects[evaluation['bindings']['floor']] = NS()
    head = NS(position=np.array(p['reflection_sensor_profile']['position_m']),
              quaternion=np.array(p['reflection_sensor_profile']['quaternion_xyzw']))
    head.get_position_orientation = lambda: (head.position.copy(), head.quaternion.copy())
    session = NS(scene_name=evaluation['scene'], room_name=evaluation['room'], camera_name='native_head',
        robot=NS(sensors={'native_head': head}, position=np.zeros(3)),
        env=NS(scene=NS(object_registry=lambda kind, name: objects.get(name))),
        reflection_qualification={key: True for key in authoring.REFLECTION_QUALIFICATION_KEYS})
    evaluator = authoring.create_evaluator(session, NS(), evaluation, tmp_path)
    observation = NS(snapshot_id='initial', action_step=0)
    evaluator.begin_episode(observation)
    return NS(session=session, evaluator=evaluator, head=head, observation=observation, parameters=p, objects=objects)


def test_all_sources_default_to_source_question_without_scene_or_manipulation_changes(tmp_path):
    for source in sources():
        original = source.path.read_bytes()
        result = authoring.CategoryConverter().convert(source, ConversionContext(tmp_path))
        task = result['task_definition']
        p = task['private']['evaluation']['parameters']
        assert task['private']['evaluation']['operation'] == authoring.ANSWER_OPERATION
        assert task['scene']['variants'] == {'original': {'patches': []}}
        assert task['requirements']['capabilities'] == ['base_translation', 'base_rotation']
        assert p['source_qa']['question'] == source.raw['question']
        assert p['source_qa']['answer'] == source.raw['answer']
        assert p['source_qa']['options'] == json.loads(source.raw['options_json'])
        assert source.raw['question'] in task['public']['objective']
        assert 'ANSWER: <option>' in task['public']['success_criteria']
        assert 'world_observation' in task['public']['success_criteria']
        assert 'answer_bool' not in json.dumps(task['public'])
        assert result['interaction_policy']['relocation_policy'] == 'preserve'
        assert result['interaction_policy']['allow_fallback_table'] is False
        assert source.path.read_bytes() == original


def test_original_reference_geometry_intrinsics_and_world_boundary_preserved(tmp_path):
    result = converted(tmp_path, reflection_robot_uniform_scale=.6)
    p = result['task_definition']['private']['evaluation']['parameters']
    q = sources()[0].payload['question_data']
    profile = p['reflection_sensor_profile']
    assert profile['position_m'] == q['render']['camera_pose']['position']
    assert profile['quaternion_xyzw'] == q['render']['camera_pose']['quaternion_xyzw']
    assert profile['horizontal_fov_deg'] == sources()[0].payload['camera_poses']['fov_deg']
    assert profile['robot_uniform_scale'] == .6
    assert profile['boundary']['limit_m'] == 1.
    assert p['expected_optical_poses']['mirror']['position'] == q['scene_setup']['mirror']['position']
    assert p['expected_optical_poses']['prop']['position'] == q['placement'][0]['real_position']
    assert p['task_revision'] == authoring.TASK_REVISION


def test_correct_answer_needs_no_motion_and_wrong_answer_cannot_complete(tmp_path):
    w = world(tmp_path)
    w.evaluator.report_answer('not a source option', w.observation)
    assert not w.evaluator.record(w.observation)['task_success']
    w.evaluator.report_answer(w.parameters['source_qa']['answer'].swapcase(), w.observation)
    record = w.evaluator.record(w.observation)
    assert record['task_success'] and record['control_samples'] == 0
    assert record['completion_items'][0]['kind'] == 'answer_requirement'


@pytest.mark.parametrize('scale', [None, .25, 2.])
def test_head_boundary_latches_after_return_and_is_scale_independent(tmp_path, scale):
    w = world(tmp_path, reflection_robot_uniform_scale=scale)
    initial = w.head.position.copy()
    w.session.robot.position[:] = 100.
    assert w.evaluator.record(w.observation)['constraints_satisfied']
    w.head.position[0] += 1.01
    w.evaluator.sample_step()
    w.head.position[:] = initial
    w.evaluator.sample_step()
    w.evaluator.report_answer(w.parameters['source_qa']['answer'], w.observation)
    record = w.evaluator.record(w.observation)
    assert record['answer_correct'] and record['completion_items'][0]['completed']
    assert record['evaluation_validity'] == 'complete' and not record['task_success']
    assert record['constraint_violations'] == ['head_boundary_exceeded']
    assert record['maximum_head_horizontal_displacement_m'] == pytest.approx(1.01)
    completion = progress(record)
    assert completion['total_goals'] == completion['completed_goals'] == 1
    assert completion['completion_score'] == 1.


def test_vertical_head_motion_does_not_count_as_horizontal_excursion(tmp_path):
    w = world(tmp_path)
    w.head.position[2] += .2
    assert w.evaluator.record(w.observation)['constraints_satisfied']


def test_unrestricted_variant_has_no_radius_or_return_requirement(tmp_path):
    w = world(tmp_path, reflection_boundary_mode='unrestricted')
    w.head.position[0] += 10.
    w.evaluator.sample_step()
    w.evaluator.report_answer(w.parameters['source_qa']['answer'], w.observation)
    assert w.evaluator.record(w.observation)['task_success']


@pytest.mark.parametrize('missing', ['head', 'qualification', 'one_qualification_field', 'optical_object'])
def test_missing_native_information_is_unavailable(tmp_path, missing):
    w = world(tmp_path)
    if missing == 'head':
        w.session.robot.sensors.clear()
    elif missing == 'qualification':
        del w.session.reflection_qualification
    elif missing == 'one_qualification_field':
        del w.session.reflection_qualification['source_head_intrinsics_verified']
    else:
        w.objects.clear()
    w.evaluator.begin_episode(w.observation)
    w.evaluator.report_answer(w.parameters['source_qa']['answer'], w.observation)
    record = w.evaluator.record(w.observation)
    assert record['evaluation_validity'] == 'unavailable'
    assert record['answer_correct'] and not record['task_success']
    completion = progress(record)
    assert completion['total_goals'] == 1 and completion['completion_score'] is None
    assert completion['completion_items'][0]['evaluation_validity'] == 'unavailable'


def test_missing_step_measurement_is_latched_even_after_sensor_recovers(tmp_path):
    w = world(tmp_path)
    w.session.robot.sensors.clear()
    w.evaluator.sample_step()
    w.session.robot.sensors['native_head'] = w.head
    w.evaluator.report_answer(w.parameters['source_qa']['answer'], w.observation)
    assert w.evaluator.record(w.observation)['evaluation_validity'] == 'unavailable'


def test_optical_pose_change_is_latched(tmp_path):
    w = world(tmp_path)
    mirror = w.objects[w.evaluator.config['bindings']['mirror']]
    mirror.position[0] += .1
    w.evaluator.sample_step()
    mirror.position[0] -= .1
    w.evaluator.report_answer(w.parameters['source_qa']['answer'], w.observation)
    record = w.evaluator.record(w.observation)
    assert 'original_optical_poses_changed' in record['constraint_violations']
    assert not record['task_success']


@pytest.mark.parametrize('operation', [authoring.LIFT_OPERATION, authoring.OPERATION])
def test_historical_recipes_remain_explicitly_selectable(tmp_path, operation):
    result = converted(tmp_path, reflection_authoring_operation=operation)
    assert result['task_definition']['private']['evaluation']['operation'] == operation


def test_conflicting_source_qa_is_not_silently_relabelled(tmp_path):
    raw = sources()[0].raw
    q = sources()[0].payload
    q['question_data']['answer'] = 'No' if raw['answer'].lower() == 'yes' else 'Yes'
    raw['metadata_json'] = q
    path = tmp_path / 'conflict.json'
    path.write_text(json.dumps(raw))
    source = SourceInstance(path, 'conflict.json', *authoring.CategoryConverter.category)
    with pytest.raises(ValueError, match='must agree'):
        authoring.CategoryConverter().convert(source, ConversionContext(tmp_path))


def test_private_answer_change_cannot_change_public_prompt(tmp_path):
    source = sources()[0]
    expected = authoring.CategoryConverter().convert(source, ConversionContext(tmp_path))
    raw, payload = source.raw, source.payload
    raw['answer'] = 'No' if raw['answer'].lower() == 'yes' else 'Yes'
    payload['question_data']['answer'] = raw['answer']
    surrogate = NS(raw=raw, payload=payload, relative_path=source.relative_path,
        big_task=source.big_task, small_task=source.small_task)
    changed = authoring.CategoryConverter().convert(surrogate, ConversionContext(tmp_path))
    assert expected['task_definition']['public'] == changed['task_definition']['public']


@pytest.mark.parametrize('module,count', [(correspondence, 88), (spatial, 113)])
def test_other_reflection_sources_share_preserved_answer_contract(tmp_path, module, count):
    root = ROOT / 'ESI/dataset/json_clean'
    paths = sorted(root.joinpath(*module.CategoryConverter.category).rglob('*.json'))
    assert len(paths) == count
    for path in paths:
        source = SourceInstance(path, str(path.relative_to(root)), *module.CategoryConverter.category)
        result = module.CategoryConverter().convert(source, ConversionContext(tmp_path))
        task = result['task_definition']
        config = deepcopy(task['private']['evaluation'])
        p = config['parameters']
        assert config['operation'] == module.ANSWER_OPERATION
        assert task['scene']['variants'] == {'original': {'patches': []}}
        assert len(p['observable_candidate_labels']) == len(source.payload['question_data']['placement'])
        assert source.raw['question'] in task['public']['objective']
        assert p['reflection_sensor_profile']['position_m'] == source.payload['question_data']['render']['camera_pose']['position']
        assert set(p['expected_optical_poses']) == {'mirror', *p['observable_candidate_labels']}
    # Exercise the shared runtime with every class-specific optical binding.
    objects = {}
    for key, pose in p['expected_optical_poses'].items():
        objects[key] = NS(get_position_orientation=lambda pose=pose: (pose['position'], pose['quaternion_xyzw']))
    objects['floor'] = NS()
    profile = p['reflection_sensor_profile']
    head = NS(get_position_orientation=lambda: (profile['position_m'], profile['quaternion_xyzw']))
    config.update(scene=source.raw['scene'], room=source.raw['room'], bindings={key: key for key in objects})
    session = NS(scene_name=config['scene'], room_name=config['room'], camera_name='head',
        robot=NS(sensors={'head': head}), env=NS(scene=NS(object_registry=lambda kind, name: objects[name])),
        reflection_qualification={key: True for key in authoring.REFLECTION_QUALIFICATION_KEYS})
    evaluator = module.create_evaluator(session, NS(), config, tmp_path)
    observation = NS(snapshot_id='initial', action_step=0)
    evaluator.begin_episode(observation)
    evaluator.report_answer(p['source_qa']['answer'], observation)
    assert evaluator.record(observation)['task_success']


@pytest.mark.parametrize('module,count', [(connectivity, 60), (navigation, 60), (regional, 80), (passage, 60)])
def test_mapping_source_consistency_and_restored_semantics(tmp_path, module, count):
    from wia.tasks.loader import task_definition
    root = ROOT / 'ESI/dataset/json_clean'
    paths = sorted(root.joinpath(*module.CategoryConverter.category).rglob('*.json'))
    assert len(paths) == count
    for path in paths:
        source = SourceInstance(path, str(path.relative_to(root)), *module.CategoryConverter.category)
        result = module.CategoryConverter().convert(source, ConversionContext(tmp_path))
        task = result['task_definition']
        task_definition(task)
        p = task['private']['evaluation']['parameters']
        assert p['task_revision'] == authoring.TASK_REVISION
        assert task['private']['evaluation']['operation'] == module.REVISED_OPERATION
        assert p['source_door_policy']['reconciled'] is None
        assert p['source_door_policy']['runtime_policy'] == 'retain_native_doors'
        assert p['source_qa']['answer'] == source.raw['answer']
        if module is navigation:
            assert p['required_order'] == source.payload['question_data']['region_sequence']
            assert p['source_qa']['scored'] is False
            assert ' -> '.join(p['required_order']) in task['public']['objective']
        else:
            assert source.raw['question'] in task['public']['objective']
            assert 'ANSWER: <option>' in task['public']['success_criteria']
        if module is connectivity:
            assert p['relation_semantics'] == 'direct_connection_not_global_reachability'
        elif module is passage:
            assert p['actual_capability'] == 'shortest_route_via_region'
            assert p['queried_portal'] is None
            assert p['route_answer_order'] == [p['start_region'], source.payload['question_data']['correct_region'], p['target_region']]
        elif module is regional:
            assert p['required_anchor_bindings'] == task['private']['evaluation']['binding_keys']


def mapping_world(tmp_path, module=connectivity, *, route=None):
    from harness_integration.test_cognitive_mapping_arrival_reports import World
    w = World(module, tmp_path, route=route)
    native = w.evaluator
    config = deepcopy(native.config)
    p = config['parameters']
    p.update(source_path='fixture.json', source_qa={'answer': 'No', 'scored': module is not navigation},
        actual_capability='observed_anchor_region_relation' if module is regional else
            'shortest_route_via_region' if module is passage else 'supplied_order_route_execution'
            if module is navigation else 'direct_adjacency_and_region_exploration')
    if module is regional:
        p['required_anchor_bindings'] = ['anchor_a', 'anchor_b']
    w.session.episode_id = 'episode_fixture'
    w.session.mapping_qualification = {'fixture.json': {'episode_id': 'episode_fixture',
        'source_runtime_door_policy_reconciled': True, 'condition_identity_verified': True,
        'source_answer_verified': True, 'route_graph_metric_verified': True, 'relevant_door_ids': []}}
    w.session.door_qualification = {}
    w.session.mapping_anchor_observations = {'fixture.json': []}
    w.evaluator = SourceAnswerMappingEvaluator(w.session, w.controller, config, tmp_path, native)
    w.evaluator.begin_episode(w.observation)
    return w


def test_connectivity_no_is_correct_after_exploring_indirectly_connected_rooms(tmp_path):
    from harness_integration.test_cognitive_mapping_arrival_reports import A, B
    w = mapping_world(tmp_path)
    w.settle(); w.report(A)
    w.move(2.); w.settle(); w.report(B)
    w.evaluator.report_answer('No', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['task_success'] and row['total_goals'] == 3
    assert row['answer_correct'] and not w.evaluator.parameters.get('unreachable_required', False)
    w.session.robot.bad_contact = True
    w.sample()
    failed = w.evaluator.record(w.observation)
    assert failed['task_success'] and failed['completed_goals'] == 3
    assert not failed['constraints_satisfied']


@pytest.mark.parametrize('fault', ['missing', 'stale_episode', 'unverified_answer', 'closed_door', 'graph_mismatch'])
def test_unknown_mapping_or_door_conditions_are_unavailable(tmp_path, fault):
    w = mapping_world(tmp_path, passage if fault == 'graph_mismatch' else connectivity)
    receipt = w.session.mapping_qualification['fixture.json']
    if fault == 'missing':
        w.session.mapping_qualification.clear()
    elif fault == 'stale_episode':
        receipt['episode_id'] = 'old_episode'
    elif fault == 'unverified_answer':
        receipt['source_answer_verified'] = None
    elif fault == 'graph_mismatch':
        receipt['route_graph_metric_verified'] = False
    else:
        receipt['relevant_door_ids'] = ['door_1']
        w.session.door_qualification['door_1'] = {'initial_state': 'closed',
            'condition_matches_current_episode': True, 'doorway_crossing_verified': True,
            'reset_restoration_verified': True, 'public_controls_only': True, 'door_opening_verified': False}
    w.evaluator.report_answer('No', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['answer_correct'] and row['evaluation_validity'] == 'unavailable'
    assert row['completion_score'] is None and not row['task_success']


@pytest.mark.parametrize('state', ['absent', 'open'])
def test_open_or_absent_door_does_not_require_opening_trial(tmp_path, state):
    w = mapping_world(tmp_path)
    w.session.mapping_qualification['fixture.json']['relevant_door_ids'] = ['door_1']
    w.session.door_qualification['door_1'] = {'initial_state': state,
        'condition_matches_current_episode': True, 'doorway_crossing_verified': True,
        'reset_restoration_verified': True, 'public_controls_only': True, 'door_opening_verified': False}
    assert mapping_qualification(w.session, w.evaluator.config)['accepted']


def test_boundary_requires_observable_anchor_association_and_delivered_rgb(tmp_path):
    w = mapping_world(tmp_path, regional)
    w.evaluator.report_answer('No', w.observation)
    assert not w.evaluator.record(w.observation)['task_success']
    for key in ('anchor_a', 'anchor_b'):
        w.session.mapping_anchor_observations['fixture.json'].append({'episode_id': 'episode_fixture',
            'binding': key, 'native_object_name': key, 'snapshot_id': 'rgb_initial',
            'observable_identity_verified': True, 'public_label_association_verified': True,
            'delivered_rgb_verified': False})
    assert not w.evaluator.record(w.observation)['task_success']
    for evidence in w.session.mapping_anchor_observations['fixture.json']:
        evidence['delivered_rgb_verified'] = True
    row = w.evaluator.record(w.observation)
    assert row['task_success'] and row['total_goals'] == 3
    assert row['visited_regions'] == []  # No arbitrary room tour is imposed.


def test_supplied_route_keeps_ordered_revisit_and_factual_prefix(tmp_path):
    from harness_integration.test_cognitive_mapping_arrival_reports import A, B
    w = mapping_world(tmp_path, navigation, route=[A, B, A])
    w.settle(); w.report(A); w.report(A)
    assert w.evaluator.record(w.observation)['completed_goals'] == 1
    w.move(2.); w.settle(); w.report(B)
    assert w.evaluator.record(w.observation)['completed_goals'] == 2
    w.move(0.); w.settle(); w.report(A)
    row = w.evaluator.record(w.observation)
    assert row['task_success'] and row['total_goals'] == row['completed_goals'] == 3


def test_queried_portal_rejects_alternate_route_and_previous_episode():
    portal = {'portal_id': 'opening_A', 'from_region': 'room_A', 'to_region': 'room_B'}
    event = {'episode_id': 'episode_1', 'portal_id': 'opening_B', 'from_region': 'room_A',
        'to_region': 'room_B', 'simulation_time_s': 1., 'before_position_m': [0., 0., 0.],
        'after_position_m': [1., 0., 0.], 'through_opening_verified': True, 'public_controls_only': True}
    session = NS(episode_id='episode_1', portal_crossings=[event])
    assert not queried_portal_crossing(session, portal)['completed']
    event['portal_id'] = 'opening_A'; event['episode_id'] = 'old_episode'
    assert not queried_portal_crossing(session, portal)['completed']
    event['episode_id'] = 'episode_1'
    assert queried_portal_crossing(session, portal)['completed']


def test_passage_executes_intermediate_region_in_order(tmp_path):
    from harness_integration.test_cognitive_mapping_arrival_reports import A, B, C
    w = mapping_world(tmp_path, passage)
    w.evaluator.parameters['route_answer_order'] = [A, C, B]
    w.settle(); w.report(A)
    w.move(2.); w.settle(); w.report(B)
    assert w.evaluator.record(w.observation)['route_answer_stage'] == 1
    w.move(1.); w.settle(); w.report(C)
    w.move(2.); w.settle(); w.report(B)
    w.evaluator.report_answer('No', w.observation)
    assert w.evaluator.record(w.observation)['task_success']


def test_composition_requires_exact_conditions_and_retains_distinct_outputs(tmp_path):
    from wia.authoring.converters.composition_mapping import plan_groups
    root = ROOT / 'ESI/dataset/json_clean'
    path = sorted(root.joinpath(*connectivity.CategoryConverter.category).rglob('*.json'))[0]
    source = SourceInstance(path, str(path.relative_to(root)), *connectivity.CategoryConverter.category)
    result = connectivity.CategoryConverter().convert(source, ConversionContext(tmp_path))
    tasks = {key: deepcopy(result['task_definition']) for key in ('first', 'second')}
    scenes = {key: deepcopy(result['scene_definition']) for key in tasks}
    records = [{'candidate_id': key, 'status': 'generated', 'big_task': 'Cognitive Mapping',
        'small_task': 'Connectivity', 'scene_model': source.raw['scene']} for key in tasks]
    assert len(plan_groups(records, tasks, scenes)) == 2
    contract = {key: 'shared_' + key for key in ('world_state_id', 'start_state_id', 'phase_id',
        'conditions_id', 'robot_configuration_id', 'controller_id')}
    contract['compatibility_verified'] = True
    for task in tasks.values():
        task['private']['evaluation']['parameters']['mapping_composition_contract'] = deepcopy(contract)
    groups = plan_groups(records, tasks, scenes)
    assert len(groups) == 1 and len(groups[0]['semantic_outputs']) == 2
    assert len(groups[0]['shared_unordered_room_requirements']) == 2
    assert 'goal_1' in groups[0]['public']['objective'] and 'goal_2' in groups[0]['public']['objective']
    tasks['second']['private']['evaluation']['parameters']['mapping_composition_contract']['phase_id'] = 'other_phase'
    assert len(plan_groups(records, tasks, scenes)) == 2


@pytest.mark.parametrize('module', [connectivity, navigation, regional, passage])
def test_mapping_historical_revision_is_explicitly_selectable(tmp_path, module):
    root = ROOT / 'ESI/dataset/json_clean'
    path = sorted(root.joinpath(*module.CategoryConverter.category).rglob('*.json'))[0]
    source = SourceInstance(path, str(path.relative_to(root)), *module.CategoryConverter.category)
    result = module.CategoryConverter().convert(source, ConversionContext(tmp_path, config={'mapping_revision': 'legacy'}))
    assert 'task_revision' not in result['task_definition']['private']['evaluation']['parameters']


def unreachable_source(tmp_path, *, conflict=None):
    root = ROOT / 'ESI/dataset/json_clean'
    for original in sorted(root.joinpath(*connectivity.CategoryConverter.category).rglob('*.json')):
        candidate = SourceInstance(original, str(original.relative_to(root)), *connectivity.CategoryConverter.category)
        if candidate.raw['answer'] == 'No':
            break
    raw, payload = candidate.raw, candidate.payload
    q = payload['question_data']
    q['verification']['graph_reachable'] = False
    q['verification']['shortest_path_attempts'] = []
    q['graph_shortest_region_path'] = []
    if conflict == 'adjacent':
        raw['answer'] = q['answer'] = 'Yes'
        q['answer_bool'] = q['directly_adjacent'] = True
    elif conflict == 'path':
        q['graph_shortest_region_path'] = [q['source_region'], q['target_region']]
    elif conflict == 'qa_disagreement':
        q['answer'] = 'Yes'
    raw['metadata_json'] = payload
    path = tmp_path / 'unreachable_source.json'
    path.write_text(json.dumps(raw))
    return SourceInstance(path, candidate.relative_path, *connectivity.CategoryConverter.category)


def unreachable_world(tmp_path):
    from harness_integration.test_cognitive_mapping_arrival_reports import A, B
    source = unreachable_source(tmp_path)
    result = connectivity.CategoryConverter().convert(source, ConversionContext(tmp_path))
    w = mapping_world(tmp_path)
    config = deepcopy(w.evaluator.config)
    config['operation'] = connectivity.REVISED_OPERATION
    config['parameters'] = deepcopy(result['task_definition']['private']['evaluation']['parameters'])
    config['parameters'].update(source_path='fixture.json', source_region=A, target_region=B)
    receipt = w.session.mapping_qualification['fixture.json']
    receipt.update(source_unreachability_verified=True, native_graph_reachable=False,
        operable_doors_considered=True, reachability_evidence={
            'method': 'native_graph_exhaustive_connectivity', 'provenance': 'fixture_native_graph_certificate'})
    w.evaluator = connectivity.create_evaluator(w.session, w.controller, config, tmp_path)
    w.evaluator.begin_episode(w.observation)
    return w, result


def test_explicit_unreachable_pair_has_two_answers_and_no_arrival_obligation(tmp_path):
    w, result = unreachable_world(tmp_path)
    task = result['task_definition']
    p = task['private']['evaluation']['parameters']
    assert p['source_qa']['answer'] == 'No' and p['reachability_answer']['answer'] == 'unreachable'
    assert p['required_regions'] == [] and not p['arrival_report_required']
    assert 'ARRIVED:' not in json.dumps(task['public'])
    assert 'ANSWER: adjacency=<Yes|No>; reachability=<reachable|unreachable>' in task['public']['success_criteria']
    w.evaluator.report_answer('adjacency=No; reachability=unreachable', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['task_success'] and row['total_goals'] == row['completed_goals'] == 2
    assert row['visited_regions'] == [] and row['native_control_samples'] == 0
    assert {item['id'] for item in row['completion_items']} == {'source_relation_answer', 'global_reachability_answer'}


def test_new_unreachable_pair_cannot_acquire_legacy_arrival_goals(tmp_path):
    source = unreachable_source(tmp_path)
    result = connectivity.CategoryConverter().convert(source, ConversionContext(tmp_path,
        config={'mapping_revision': 'legacy'}))
    p = result['task_definition']['private']['evaluation']['parameters']
    assert p['required_regions'] == [] and not p['arrival_report_required']
    assert result['available_operation_recipes'] == [connectivity.REVISED_OPERATION]


@pytest.mark.parametrize('response,adjacent_correct,reachable_correct', [
    ('adjacency=Yes; reachability=unreachable', False, True),
    ('adjacency=No; reachability=reachable', True, False)])
def test_unreachable_and_adjacency_answers_are_scored_independently(tmp_path, response, adjacent_correct, reachable_correct):
    w, _ = unreachable_world(tmp_path)
    w.evaluator.report_answer(response, w.observation)
    row = w.evaluator.record(w.observation)
    assert row['answer_correct'] is adjacent_correct
    assert row['reachability_answer_correct'] is reachable_correct
    assert row['completed_goals'] == 1 and row['completion_score'] == .5
    assert not row['task_success']


@pytest.mark.parametrize('conflict', ['adjacent', 'path', 'qa_disagreement'])
def test_contradictory_unreachable_source_is_diagnostic_without_arrival_goal(tmp_path, conflict):
    source = unreachable_source(tmp_path, conflict=conflict)
    converter = connectivity.CategoryConverter()
    assert converter.inspect(source)['eligible']
    result = converter.convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'needs_human_decision' and result['task_definition'] is None
    assert result['failure_reason'].startswith('unreachable_source_')
    assert result['private_source_evidence']['original_answer'] == source.raw['answer']


@pytest.mark.parametrize('fault', ['unknown', 'native_reachable', 'closed_door_only', 'timeout', 'missing_provenance'])
def test_unreachable_answer_requires_native_reconciliation_not_agent_failure(tmp_path, fault):
    w, _ = unreachable_world(tmp_path)
    receipt = w.session.mapping_qualification['fixture.json']
    if fault == 'unknown':
        receipt['source_unreachability_verified'] = None
    elif fault == 'native_reachable':
        receipt['native_graph_reachable'] = True
    elif fault == 'closed_door_only':
        receipt['operable_doors_considered'] = None
    elif fault == 'timeout':
        receipt['reachability_evidence']['method'] = 'agent_timeout'
    else:
        receipt['reachability_evidence']['provenance'] = None
    w.evaluator.report_answer('adjacency=No; reachability=unreachable', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['answer_correct'] and row['reachability_answer_correct']
    assert not row['task_success'] and row['completion_score'] is None
    items = {item['id']: item for item in row['completion_items']}
    assert items['source_relation_answer']['evaluation_validity'] == 'valid'
    assert items['global_reachability_answer']['evaluation_validity'] == 'unavailable'


def mark_fixed_release(w, source_path='fixture.json'):
    """Model the trusted loaded-task identity supplied by the native runner."""
    w.session.loaded_task = NS(release_v1=True,
        definition=NS(source=NS(json_path=source_path)))
    w.session.mapping_qualification.clear()


def test_fixed_release_mapping_scores_without_legacy_receipt(tmp_path):
    from harness_integration.test_cognitive_mapping_arrival_reports import A, B
    w = mapping_world(tmp_path)
    mark_fixed_release(w)
    w.evaluator.report_answer('No', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['evaluation_validity'] == 'complete'
    assert not row['task_success'] and row['completed_goals'] == 1
    proof = row['mapping_qualification']
    assert proof['policy'] == 'fixed_v1_definition'
    assert proof['receipt'] == {} and proof['physical_checks_run'] is False
    w.settle(); w.report(A)
    w.move(2.); w.settle(); w.report(B)
    assert w.evaluator.record(w.observation)['task_success']
    w.session.robot.bad_contact = True
    w.sample()
    failed = w.evaluator.record(w.observation)
    assert not failed['constraints_satisfied'] and failed['task_success']


def test_fixed_release_mapping_preserves_measurement_failure(tmp_path):
    w = mapping_world(tmp_path)
    mark_fixed_release(w)
    w.evaluator.native.measurement_errors.append({'stage': 'sample', 'reason': 'missing native state'})
    row = w.evaluator.record(w.observation)
    assert row['evaluation_validity'] == 'unavailable'
    assert row['completion_score'] is None and not row['task_success']


def test_fixed_release_mapping_cannot_qualify_another_source(tmp_path):
    w = mapping_world(tmp_path)
    mark_fixed_release(w, 'another_task.json')
    assert not mapping_qualification(w.session, w.evaluator.config)['accepted']
    assert w.evaluator.record(w.observation)['completion_score'] is None


def test_fixed_release_unreachable_answer_uses_release_definition(tmp_path):
    w, _ = unreachable_world(tmp_path)
    mark_fixed_release(w)
    w.evaluator.report_answer('adjacency=No; reachability=reachable', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['evaluation_validity'] == 'complete'
    assert not row['task_success'] and row['completed_goals'] == 1
    w.evaluator.report_answer('adjacency=No; reachability=unreachable', w.observation)
    row = w.evaluator.record(w.observation)
    assert row['task_success'] and row['completion_score'] == 1
    assert row['reachability_answer_correct']


def test_real_t04_release_selects_definition_policy_without_receipts():
    from wia.tasks.loader import load_task
    task = load_task(ROOT / 'explore_to_achieve/wia_v1/release/tasks/t04.yaml', variant='adapted')
    config = {'parameters': task.definition.private.evaluation.parameters}
    proof = mapping_qualification(NS(loaded_task=task), config)
    assert proof['accepted'] and proof['policy'] == 'fixed_v1_definition'
    assert proof['receipt'] == {} and not proof['physical_checks_run']


def test_fixed_release_mapping_still_requires_anchor_observations(tmp_path):
    w = mapping_world(tmp_path, regional)
    mark_fixed_release(w)
    w.evaluator.report_answer('No', w.observation)
    row = w.evaluator.record(w.observation)
    assert not row['task_success']
    assert any(not item['completed'] for item in row['completion_items'] if item['id'].startswith('anchor:'))

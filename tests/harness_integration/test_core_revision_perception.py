"""Verify identification and placement semantics without claiming native feasibility.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
tests/test_core_revision_perception.py
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from wia.authoring.conversion import ROOT, ConversionContext, SourceInstance
from wia.authoring.converters import partial_occlusion as partial
from wia.tasks.loader import task_definition
from wia.tasks.interaction_setup import prepare_definitions


def source():
    return SourceInstance(ROOT / 'ESI/dataset/json_clean' / partial.FEEDBACK_SOURCE,
                          partial.FEEDBACK_SOURCE, 'Perceptual Grounding', 'Partial Occlusion')


def destination():
    return {'binding': {'name': 'destination_table', 'category': 'breakfast_table'},
            'public_label': 'the designated breakfast table', 'relation': 'on_top'}


def conversion(tmp_path):
    return partial.PartialOcclusionConverter().convert(source(), ConversionContext(
        tmp_path, seed=7, config={'placement_destination': destination()}))


def replay(tmp_path):
    evaluator = partial.IdentificationPlacementEvaluator.__new__(partial.IdentificationPlacementEvaluator)
    evaluator.output = tmp_path
    evaluator.target = NS(name='original_target')
    evaluator.parameters = {'expected_identity': 'lamp', 'release_stability_steps': 3,
        'maximum_release_speed_m_s': .02, 'maximum_release_angular_speed_rad_s': .10}
    evaluator.config = {'parameters': evaluator.parameters}
    state = {'gripper': {'holding': False, 'constraint_active': False, 'release_pending': False,
                         'object_name': None}, 'destination_relation': False, 'linear_speed_m_s': 0.,
             'angular_speed_rad_s': 0., 'robot_withdrawn': True, 'measurement_errors': {}}
    evaluator.measure = lambda: deepcopy(state)
    evaluator.begin_episode(NS(snapshot_id='initial', action_step=0))
    return evaluator, state


def select(evaluator, state, name='original_target'):
    state['gripper'].update(holding=True, constraint_active=True, object_name=name)
    evaluator.sample_step()


def place(evaluator, state):
    state['gripper'].update(holding=False, constraint_active=False, object_name=None)
    state['destination_relation'] = True
    for _ in range(3):
        evaluator.sample_step()
    return evaluator.record(NS(snapshot_id='placed', action_step=2))


def test_wrong_identity_does_not_receive_identification_credit_after_correct_placement(tmp_path):
    evaluator, state = replay(tmp_path)
    evaluator.report_answer('vase', NS(snapshot_id='answer', action_step=1))
    select(evaluator, state)
    result = place(evaluator, state)
    assert [item['completed'] for item in result['completion_items']] == [False, True, True]
    assert result['completion_score'] == pytest.approx(2 / 3)
    assert not result['task_success']


def test_correct_identity_is_independent_from_wrong_object_selection_or_placement(tmp_path):
    evaluator, state = replay(tmp_path)
    evaluator.report_answer('LAMP', NS(snapshot_id='answer', action_step=1))
    select(evaluator, state, name='confuser')
    result = evaluator.record(NS(snapshot_id='wrong_object', action_step=1))
    assert [item['completed'] for item in result['completion_items']] == [True, False, False]
    assert result['completion_score'] == pytest.approx(1 / 3)


def test_held_target_and_repeated_record_reads_cannot_establish_stable_release(tmp_path):
    evaluator, state = replay(tmp_path)
    evaluator.report_answer('lamp', NS(snapshot_id='answer', action_step=1))
    select(evaluator, state)
    state['destination_relation'] = True
    for _ in range(10):
        evaluator.sample_step()
    assert evaluator.stable_steps == 0
    state['gripper'].update(holding=False, constraint_active=False, object_name=None)
    for _ in range(100):
        result = evaluator.record(NS(snapshot_id='held', action_step=1))
    assert evaluator.stable_steps == 0
    assert not result['task_success']
    result = place(evaluator, state)
    assert result['task_success']
    state['linear_speed_m_s'] = .03
    result = evaluator.record(NS(snapshot_id='disturbed', action_step=3))
    assert not result['task_success']
    assert result['completion_items'][1]['completed']
    assert not result['completion_items'][2]['completed']


def test_native_grasp_does_not_supply_unreported_identity(tmp_path):
    evaluator, state = replay(tmp_path)
    select(evaluator, state)
    result = place(evaluator, state)
    assert not result['completion_items'][0]['completed']
    assert result['completed_goals'] == 2


def test_missing_placement_telemetry_is_unavailable_with_valid_identity_retained(tmp_path):
    evaluator, state = replay(tmp_path)
    evaluator.report_answer('lamp', NS(snapshot_id='answer', action_step=1))
    select(evaluator, state)
    state['destination_relation'] = None
    state['measurement_errors']['destination_relation'] = 'Inside state unavailable'
    result = evaluator.record(NS(snapshot_id='missing', action_step=1))
    assert result['evaluation_validity'] == 'partial'
    assert result['completion_score'] is None
    assert result['completion_items'][0]['evaluation_validity'] == 'valid'
    assert result['completion_items'][2]['evaluation_validity'] == 'unavailable'
    assert result['completed_goals'] == 2


def test_unknown_native_grasp_state_is_not_an_agent_selection_failure(tmp_path):
    evaluator, state = replay(tmp_path)
    state['gripper'] = None
    result = evaluator.record(NS(snapshot_id='unknown', action_step=1))
    assert result['completion_items'][1]['evaluation_validity'] == 'unavailable'
    assert result['completion_score'] is None


def test_release_pending_or_robot_contact_prevents_stability(tmp_path):
    evaluator, state = replay(tmp_path)
    select(evaluator, state)
    state['gripper'].update(holding=False, constraint_active=False, object_name=None, release_pending=True)
    state['destination_relation'] = True
    for _ in range(4):
        evaluator.sample_step()
    assert evaluator.stable_steps == 0
    state['gripper']['release_pending'] = False
    state['robot_withdrawn'] = False
    for _ in range(4):
        evaluator.sample_step()
    assert evaluator.stable_steps == 0


def test_conversion_preserves_original_mechanism_and_separates_private_answer(tmp_path):
    original = source().path.read_bytes()
    result = conversion(tmp_path)
    task = result['task_definition']
    evaluation = task['private']['evaluation']
    parameters = evaluation['parameters']
    assert evaluation['operation'] == partial.IDENTIFY_PLACE
    assert parameters['task_revision'] == partial.TASK_REVISION
    assert parameters['identity_choices'] == partial._public_question(source().raw)[1]
    assert parameters['expected_identity'] == partial._normalize_answer(source().raw['answer'])
    assert task['public']['objective'].startswith('Inspect the partly hidden candidate')
    assert 'world_observation as ANSWER: <category>' in task['public']['objective']
    assert 'before placement' not in str(task['public'])
    assert task['private']['bindings']['target']['name'] == partial._source_builder().build_env_objects(source().payload)[0]['name']
    assert task['private']['bindings']['destination'] == destination()['binding']
    assert result['interaction_policy']['relocation_policy'] == 'preserve'
    assert result['interaction_policy']['allow_fallback_table'] is False
    assert result['scene_definition']['scene']['variants']['original']['patches'] == []
    _, _, reference = partial._source_builder().initial_camera(source().payload)
    assert result['source_reference'] == reference
    for patch in task['scene']['variants']['adapted']['patches']:
        assert patch['operation'] in ('robot_pose', 'object_physics')
    prepare_definitions(source(), result['scene_definition'], task, result['interaction_policy'])
    task_definition(task)
    assert source().path.read_bytes() == original


def test_missing_destination_remains_explicit_diagnostic(tmp_path):
    result = partial.PartialOcclusionConverter().convert(source(), ConversionContext(tmp_path))
    assert result['task_definition'] is None
    assert result['failure_reason'] == 'placement_destination_pending'
    assert result['status'] == 'needs_human_decision'
    assert result['scene_definition']['scene']['variants']['original']['patches'] == []


def test_legacy_recipe_remains_explicitly_available(tmp_path):
    result = partial.PartialOcclusionConverter().convert(source(), ConversionContext(
        tmp_path, config={'legacy_partial_occlusion': True}))
    assert result['task_definition']['private']['evaluation']['type'] == 'grasp_lift'
    assert result['rule_revision'] == partial.LEGACY_RULE_REVISION


def test_inside_requires_native_inside_and_never_accepts_bbox_overlap(monkeypatch):
    inside, on_top, touching = object(), object(), object()
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', NS(Inside=inside, OnTop=on_top, Touching=touching))
    target = NS(states={inside: NS(get_value=lambda dest: False)},
                aabb=(np.zeros(3), np.ones(3)), get_linear_velocity=lambda: np.zeros(3),
                get_angular_velocity=lambda: np.zeros(3), contact_list=lambda: [])
    controller = NS(debug_gripper_state=lambda: {'holding': False, 'constraint_active': False,
                                                'release_pending': False, 'object_name': None})
    session = NS(robot=NS(prim_path='/World/robot'))
    config = {'parameters': {'placement_relation': 'inside'}}
    result = partial._placement_measurements(session, controller, config, target, NS(aabb=target.aabb))
    assert result['destination_relation'] is False
    target.states = {}
    result = partial._placement_measurements(session, controller, config, target, NS(aabb=target.aabb))
    assert result['destination_relation'] is None
    assert 'destination_relation' in result['measurement_errors']


def test_qualification_reads_the_executor_controller(monkeypatch):
    controller = NS(debug_gripper_state=lambda: {})
    executor = NS(controller=controller, capabilities=NS(gripper=True, ee_translation=True))
    bound = {key: NS(name=key, fixed_base=False, visual_only=False)
             for key in ('target', 'occluder', 'support', 'destination')}
    config = {'operation': partial.IDENTIFY_PLACE, 'scene': 'scene', 'room': 'room',
              'binding_keys': list(bound), 'bindings': {key: key for key in bound}}
    session = NS(scene_name='scene', room_name='room')
    monkeypatch.setattr(partial, '_robot_safety', lambda session: dict(
        floor_supported=True, collision_free=True, same_source_room=True, native_r1=True))
    measured_controllers = []
    def measure(session, actual_controller, config, target, destination):
        measured_controllers.append(actual_controller)
        return {'measurement_errors': {}}
    monkeypatch.setattr(partial, '_placement_measurements', measure)
    result = partial.qualify_operation(session, executor, config, bound)
    assert result['accepted']
    assert measured_controllers == [controller]
    assert not result['checks']['destination_interior_geometry_verified']
    assert not result['executed_operation_verified']


def visual_world(tmp_path, *, names=('first', 'second')):
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.ones((16, 16), dtype=np.int32)
    data = {'rgb': rgb, 'seg_instance': mask}
    labels = {'seg_instance': {1: names[0], 2: names[1]}}
    objects = {name: NS(name=name) for name in names}
    session = NS(robot_camera_registry={'agentview_rgb': 'head'}, robot=NS(
        sensors={'head': NS(get_obs=lambda: (data, labels))}), env=NS(scene=NS(
            object_registry=lambda kind, name: objects[name])))
    config = {'bindings': {name: name for name in names}, 'parameters': {'target_keys': list(names)}}
    observation = NS(snapshot_id='first', action_step=0, rgb_images=lambda: {'agentview_rgb': rgb.copy()})
    evaluator = partial.VisualDiscoveryEvaluator(session, None, config, tmp_path)
    return evaluator, observation, data, labels


def test_rendered_discoveries_are_distinct_and_do_not_require_movement(tmp_path):
    evaluator, observation, data, labels = visual_world(tmp_path)
    first = evaluator.begin_episode(observation)
    assert first['completed_goals'] == 1
    for _ in range(5):
        assert evaluator.record(observation)['completed_goals'] == 1
    data['seg_instance'][:] = 2
    second = evaluator.record(observation)
    assert second['completed_goals'] == 2 and second['task_success']
    assert not second['semantic_recognition_verified']


def test_stale_rgb_or_missing_segmentation_is_unavailable_not_a_zero(tmp_path):
    evaluator, observation, data, labels = visual_world(tmp_path)
    delivered = observation.rgb_images()['agentview_rgb']
    observation.rgb_images = lambda: {'agentview_rgb': delivered}
    data['rgb'] = np.ones_like(data['rgb'])
    result = evaluator.begin_episode(observation)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['completion_score'] is None
    data['rgb'][:] = 0
    result = evaluator.record(observation)
    assert result['completed_goals'] == 1
    del data['seg_instance']
    result = evaluator.record(observation)
    assert result['completed_goals'] == 1
    assert result['evaluation_validity'] == 'partial'
    assert result['completion_items'][0]['evaluation_validity'] == 'valid'
    assert result['completion_items'][1]['evaluation_validity'] == 'unavailable'


def test_region_report_requires_real_discovery_and_does_not_use_private_ids(tmp_path):
    evaluator, observation, data, labels = visual_world(tmp_path)
    evaluator.parameters['regions'] = {'chair_private_id': {'public_label': 'left chair region',
                                                           'targets': ['second']}}
    result = evaluator.begin_episode(observation)
    evaluator.report_answer('chair_private_id', observation)
    assert not evaluator.region_reports
    evaluator.report_answer('left chair region', observation)
    assert not evaluator.record(observation)['completion_items'][-1]['completed']
    data['seg_instance'][:] = 2
    assert evaluator.record(observation)['completion_items'][-1]['completed']


def inside_world(tmp_path, monkeypatch, *, initially_inside):
    from wia.authoring.converters import perceptual_grounding_material_transparency as transparency
    state = {'gripper': {'holding': False, 'constraint_active': False, 'release_pending': False,
                         'object_name': None}, 'destination_relation': initially_inside,
             'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'robot_withdrawn': True,
             'measurement_errors': {}}
    monkeypatch.setattr(partial, '_placement_measurements', lambda *args: deepcopy(state))
    objects = {'small': NS(name='small'), 'container': NS(name='container')}
    session = NS(scene_name='fixture', room_name='room', env=NS(scene=NS(
        object_registry=lambda kind, name: objects[name])))
    config = {'scene': 'fixture', 'room': 'room', 'bindings': {'obj_small': 'small', 'obj_container': 'container'},
              'parameters': {'placement_relation': 'inside', 'release_stability_steps': 3,
                             'maximum_release_speed_m_s': .02, 'maximum_release_angular_speed_rad_s': .1}}
    evaluator = transparency.create_evaluator(session, None, config, tmp_path)
    observation = NS(snapshot_id='initial', action_step=0)
    evaluator.begin_episode(observation)
    return evaluator, observation, state


def test_transparency_already_inside_allows_verified_noop_without_grasp(tmp_path, monkeypatch):
    evaluator, observation, state = inside_world(tmp_path, monkeypatch, initially_inside=True)
    for _ in range(3):
        evaluator.sample_step()
    result = evaluator.record(observation)
    assert result['task_success'] and result['no_op_verified']
    assert result['total_goals'] == 1
    assert result['completion_items'][0]['id'] == 'stable_inside'


def test_transparency_outside_requires_released_inside_and_missing_inside_is_unavailable(tmp_path, monkeypatch):
    evaluator, observation, state = inside_world(tmp_path, monkeypatch, initially_inside=False)
    assert not evaluator.record(observation)['task_success']
    state['destination_relation'] = True
    state['gripper'].update(holding=True, constraint_active=True, object_name='small')
    for _ in range(3):
        evaluator.sample_step()
    assert not evaluator.record(observation)['task_success']
    state['gripper'].update(holding=False, constraint_active=False, object_name=None)
    for _ in range(3):
        evaluator.sample_step()
    result = evaluator.record(NS(snapshot_id='inserted', action_step=2))
    assert result['task_success'] and not result['no_op_verified']
    state['destination_relation'] = None
    state['measurement_errors']['destination_relation'] = 'native state missing'
    assert evaluator.record(observation)['completion_score'] is None


def test_enclosure_opening_and_content_discovery_are_separate_without_closing(tmp_path, monkeypatch):
    from wia.authoring.converters import enumerative_perception_structural_enclosure as enclosure
    pose = {'name': 'box', 'position': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.],
            'bbox': [[0., 0., 0.], [1., 1., 1.]], 'visual_only': False}
    joint = {'name': 'hinge', 'position': 0., 'open_direction': 1., 'minimum_travel': .5,
             'closed_tolerance': .01, 'hand_contact': False, 'driven': False,
             'visible': True, 'collision_mesh_count': 1, 'contact_reporting': True}
    door = {**pose, 'open': False, 'joints': [joint]}
    content = {**pose, 'inside_original_container': True}
    measured = ({'box': door}, {'content': content}, {}, {})
    monkeypatch.setattr(enclosure, '_measure', lambda *args: deepcopy(measured))
    monkeypatch.setattr(enclosure, '_near', lambda *args: True)
    visible = {'objects': {'content': {'visible': False}}, 'complete_coverage': True}
    monkeypatch.setattr(partial, 'native_visual_evidence', lambda *args: deepcopy(visible))
    objects = {'box': NS(name='box'), 'content': NS(name='content')}
    session = NS(env=NS(scene=NS(object_registry=lambda kind, name: objects[name])))
    config = {'bindings': {key: key for key in objects}, 'parameters': {'container_keys': ['box'], 'content_keys': ['content']}}
    evaluator = enclosure.OpenDiscoverEvaluator(session, None, config, tmp_path)
    observation = NS(snapshot_id='initial', action_step=0)
    result = evaluator.begin_episode(observation)
    assert result['total_goals'] == 2 and not result['task_success']
    joint.update(position=.6, hand_contact=True)
    door['open'] = True
    evaluator.sample_step()
    result = evaluator.record(observation)
    assert [item['completed'] for item in result['completion_items']] == [True, False]
    visible['objects']['content']['visible'] = True
    result = evaluator.record(observation)
    assert result['task_success']
    assert door['open'] is True


def test_inventory_placement_requires_every_target_and_confuser_protection(tmp_path, monkeypatch):
    from wia.authoring.converters import enumerative_perception_category_ambiguity as ambiguity
    def obj(name):
        return NS(name=name, get_position_orientation=lambda: (np.zeros(3), np.array([0., 0., 0., 1.])))
    objects = {key: obj(key) for key in ('a', 'b', 'confuser', 'destination')}
    gripper = {'holding': False, 'constraint_active': False, 'release_pending': False, 'object_name': None}
    relation = {'a': True, 'b': False}
    def measurement(session, controller, config, target, destination):
        return {'gripper': gripper.copy(), 'destination_relation': relation[target.name],
                'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'robot_withdrawn': True, 'measurement_errors': {}}
    monkeypatch.setattr(partial, '_placement_measurements', measurement)
    session = NS(env=NS(scene=NS(object_registry=lambda kind, name: objects[name])))
    controller = NS(debug_gripper_state=lambda: gripper)
    config = {'bindings': {key: key for key in objects}, 'parameters': {'target_keys': ['a', 'b'], 'confuser_keys': ['confuser']}}
    evaluator = ambiguity.InventoryPlacementEvaluator(session, controller, config, tmp_path)
    observation = NS(snapshot_id='initial', action_step=0)
    evaluator.begin_episode(observation)
    for _ in range(ambiguity.STABILITY_STEPS):
        evaluator.sample_step()
    assert evaluator.record(observation)['completed_goals'] == 1
    relation['b'] = True
    for _ in range(ambiguity.STABILITY_STEPS):
        evaluator.sample_step()
    assert evaluator.record(observation)['task_success']
    gripper.update(holding=True, constraint_active=True, object_name='confuser')
    evaluator.sample_step()
    gripper.update(holding=False, constraint_active=False, object_name=None)
    for _ in range(ambiguity.STABILITY_STEPS):
        evaluator.sample_step()
    result = evaluator.record(observation)
    assert result['completed_goals'] == 2 and not result['task_success']


@pytest.mark.parametrize('module_name', [
    'perceptual_grounding_view_hallucination', 'enumerative_perception_category_ambiguity',
    'enumerative_perception_counting_w_occlusion', 'enumerative_perception_merged_observation',
    'enumerative_perception_spatial_segmentation', 'temporal_understanding_agent_observation'])
def test_revised_converters_preserve_source_mechanisms_with_explicit_fixture_inputs(tmp_path, module_name):
    import importlib
    module = importlib.import_module('wia.authoring.converters.' + module_name)
    converter = module.CategoryConverter()
    big, small = converter.category
    root = ROOT / 'ESI/dataset/json_clean'
    candidates = [SourceInstance(path, str(path.relative_to(root)), big, small)
                  for path in sorted((root / big / small).rglob('*.json'))]
    selected = next(item for item in candidates if converter.inspect(item)['eligible'])
    original = selected.path.read_bytes()
    config = {'placement_destination': destination()}
    if small == 'Spatial Segmentation':
        layout = module._layout(selected)
        anchors = list(dict.fromkeys(pair['anchor'] for pair in layout['pairs']))
        config['region_public_labels'] = {key: f'fixture-qualified chair region {index + 1}'
                                          for index, key in enumerate(anchors)}
    result = converter.convert(selected, ConversionContext(tmp_path, seed=7, config=config))
    task = result['task_definition']
    assert result['status'] == 'generated'
    assert result['interaction_policy']['relocation_policy'] == 'preserve'
    assert not result['interaction_policy']['allow_fallback_table']
    assert task['private']['evaluation']['parameters']['task_revision'] == partial.TASK_REVISION
    assert task['scene']['variants']['original']['patches'] == []
    prepare_definitions(selected, result['scene_definition'], task, result['interaction_policy'])
    task_definition(task)
    assert selected.path.read_bytes() == original
    if small in ('Counting w Occlusion', 'Merged Observation', 'Spatial Segmentation', 'Agent Observation'):
        parameters = task['private']['evaluation']['parameters']
        assert parameters['disclosed_target_count'] == len(parameters['target_keys'])
        assert parameters['requires_native_visual_discovery']
        assert str(parameters['disclosed_target_count']) in task['public']['objective']
    if small == 'Agent Observation':
        assert result['measured_capability'] == 'all_target_visual_search'
        assert result['capability_coverage_gaps']


def test_lighting_requires_matching_visible_targets_in_both_actual_phases(tmp_path):
    from wia.authoring.converters import enumerative_perception_illumination_variability as illumination
    visual, observation, data, labels = visual_world(tmp_path)
    evaluator = illumination.IlluminationInventoryEvaluator.__new__(illumination.IlluminationInventoryEvaluator)
    evaluator.session = visual.session
    evaluator.objects = visual.objects
    evaluator.targets = {'first'}
    evaluator.parameters = {'requires_native_visual_discovery': True}
    evaluator.config = {'operation': 'two_phase_inventory_return'}
    evaluator.phase_views = {}
    evaluator.episode = 1
    evaluator.initial = evaluator.initial_layout = {}
    evaluator.output = tmp_path
    phase = [illumination.PHASES[0]]
    evaluator._update = lambda: {'task_success': True, 'native_lighting': {'measured_phase': phase[0]},
        'completion_items': [{'id': 'physical-events', 'completed': True, 'evaluation_validity': 'valid'}]}
    result = evaluator.record(observation, action=None)
    assert not result['task_success']
    assert len(result['phase_visual_evidence']) == 1
    result = evaluator.record(observation, action=None)
    assert not result['task_success']
    phase[0] = illumination.PHASES[1]
    result = evaluator.record(observation, action=None)
    assert result['task_success']
    assert {row['phase'] for row in result['phase_visual_evidence']} == set(illumination.PHASES)

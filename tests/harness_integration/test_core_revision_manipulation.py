"""Check repaired manipulation semantics without simulation.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
tests/test_core_revision_manipulation.py
These native-shaped CPU fixtures validate predicates, not physical feasibility.
"""
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import spatial_relations_physical_contact as contact
from wia.tasks.loader import task_definition
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope
from harness_integration.test_annotation_manipulation_categories import source_for
from harness_integration.test_spatial_relations_tabletop import world_factory


def test_contact_relation_seed_is_reproducible_and_independent_of_initial_answer(tmp_path, monkeypatch):
    converter, source = source_for(contact)
    monkeypatch.setattr(contact, '_robot_start', lambda *args: (None, {'failure_reason': 'cpu_only'}))
    chosen = {}
    for seed in range(20):
        value = converter.convert(source, ConversionContext(tmp_path, seed=seed))
        task = value['task_definition']
        evaluation = task['private']['evaluation']
        relation = evaluation['parameters']['desired_relation']
        assert evaluation['operation'] == contact.FINAL_OPERATION
        assert relation == contact.seeded_final_relation(source.relative_path, seed)
        assert task['repair']['task_revision'] == contact.TASK_REVISION
        assert task['defaults'] == {'seed': seed, 'max_steps': 240, 'wall_timeout_s': 14400.}
        assert task['scene']['variants']['original']['patches'] == []
        assert 'no grasp, lift, movement' in task['public']['success_criteria'].lower()
        assert 'source_objects' not in task['public']
        task_definition(task)
        chosen.setdefault(relation, seed)
    assert set(chosen) == set(contact.FINAL_RELATIONS)
    original = source.raw
    original['answer'] = 'yes' if original['answer'] == 'no' else 'no'
    changed_path = tmp_path / 'alternate_source.json'
    changed_path.write_text(json.dumps(original))
    changed = SourceInstance(changed_path, source.relative_path, *contact.CategoryConverter.category)
    for seed in chosen.values():
        before = converter.convert(source, ConversionContext(tmp_path, seed=seed))
        after = converter.convert(changed, ConversionContext(tmp_path, seed=seed))
        assert before['task_definition']['private']['evaluation']['parameters']['desired_relation'] == (
            after['task_definition']['private']['evaluation']['parameters']['desired_relation'])


def fake_contact_api(monkeypatch, separations):
    utils = ModuleType('omnigibson.utils')
    usd = ModuleType('omnigibson.utils.usd_utils')
    usd.RigidContactAPI = SimpleNamespace(get_contact_data=lambda *args, **kwargs: [
        ('a', 'b', 1., [0., 0., 0.], [1., 0., 0.], value) for value in separations])
    monkeypatch.setitem(sys.modules, 'omnigibson.utils', utils)
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.usd_utils', usd)
    return {key: SimpleNamespace(scene=SimpleNamespace(idx=0),
        links={'base': SimpleNamespace(prim_path='/World/' + key + '/base')}) for key in contact.OBJECTS}


@pytest.mark.parametrize('depth, expected', [(0., 'touching'), (-.0005, 'touching'), (-.003, 'penetrating')])
def test_native_contact_depth_distinguishes_penetration(monkeypatch, depth, expected):
    objects = fake_contact_api(monkeypatch, [depth])
    result = contact._contact_geometry(objects, True, 0.)
    assert result['available'] and result['relation'] == expected
    assert result['minimum_separation_m'] == depth
    assert result['penetrating'] is (expected == 'penetrating')


def test_separation_needs_positive_clearance_and_contact_depth_must_be_measured(monkeypatch):
    objects = fake_contact_api(monkeypatch, [])
    assert contact._contact_geometry(objects, False, .0001)['relation'] == 'separated'
    assert not contact._contact_geometry(objects, False, 0.)['available']
    assert not contact._contact_geometry(objects, True, 0.)['available']


@pytest.fixture
def final_trial(world_factory, monkeypatch, tmp_path):
    world = world_factory(contact)
    world.config['operation'] = contact.FINAL_OPERATION
    world.config['parameters']['desired_relation'] = 'separated'
    # Real fixture setup qualification still checks original references and support.
    initial = contact._pair_measure(dict(zip(world.keys, world.props)), world.floor, world.session)
    live = copy.deepcopy(initial)
    monkeypatch.setattr(contact, '_pair_measure', lambda *args: copy.deepcopy(live))
    evaluator = contact.create_evaluator(world.session, world.controller, world.config, tmp_path)
    world.evaluator, world.live = evaluator, live
    world.observation = SimpleNamespace(snapshot_id='s0', action_step=0)
    return world


def settle(world):
    for _ in range(contact.STABILITY_STEPS):
        world.evaluator.sample_step()
    return world.evaluator.record(world.observation, action=None)


@pytest.mark.parametrize('relation', contact.FINAL_RELATIONS)
def test_already_correct_relation_succeeds_without_actions_after_native_settle(final_trial, relation):
    world = final_trial
    world.config['parameters']['desired_relation'] = relation
    world.live['contact_geometry'].update(available=True, relation=relation, penetrating=False)
    world.evaluator.begin_episode(world.observation)
    initial = world.evaluator.record(world.observation, action=None)
    assert initial['no_op_initially_satisfied'] is True
    assert initial['task_success'] is False
    # Observation reads do not manufacture a measured settling interval.
    for _ in range(contact.STABILITY_STEPS + 2):
        world.evaluator.record(world.observation, action=None)
    assert world.evaluator.stable_steps == 0
    result = settle(world)
    assert result['evaluation_validity'] == 'complete'
    assert result['task_success'] is True and result['completion_score'] == 1.
    assert result['executed_actions'] == [] and result['executed_action_count'] == 0
    assert result['native_control_samples'] == contact.STABILITY_STEPS


@pytest.mark.parametrize('failure', ['held', 'unstable', 'unsupported', 'penetrating', 'anchor_moved'])
def test_contact_final_state_rejects_invalid_release_or_relation(final_trial, failure):
    world = final_trial
    world.evaluator.begin_episode(world.observation)
    if failure == 'held':
        world.controller.debug_gripper_state = lambda: {'object_name': world.props[0].name,
            'holding': True, 'constraint_active': True, 'gripper_open': False, 'release_pending': False}
    elif failure == 'unstable':
        world.live['stationary'] = False
    elif failure == 'unsupported':
        world.live['task_supported'] = False
    elif failure == 'penetrating':
        world.config['parameters']['desired_relation'] = 'touching'
        world.live['contact_geometry'].update(relation='penetrating', penetrating=True)
    else:
        world.live['objects']['obj2']['position'][0] += .1
    result = settle(world)
    assert result['evaluation_validity'] == 'complete'
    assert result['task_success'] is False and result['stable_steps'] == 0


def test_factual_completion_survives_terminal_violation_and_actions_are_separate(final_trial):
    world = final_trial
    world.evaluator.begin_episode(world.observation)
    assert settle(world)['task_success']
    world.live['objects']['obj2']['position'][0] += .1
    world.observation.action_step = 1
    value = world.evaluator.record(world.observation, action={'base_forward': .1})
    assert value['completed_goals'] == 2 and value['task_success'] is False
    assert value['constraints_satisfied'] is False
    assert value['executed_action_count'] == 1
    assert value['no_op_initially_satisfied'] is True


def test_missing_pair_measurements_are_unavailable_and_preserve_prior_evidence(final_trial):
    world = final_trial
    world.evaluator.begin_episode(world.observation)
    assert settle(world)['task_success']
    world.live['contact_geometry'] = {'available': False, 'reason': 'Native separation missing.'}
    result = world.evaluator.record(world.observation, action=None)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['task_success'] is False and result['completion_score'] is None
    assert result['completed_goals'] == 2


def test_native_idle_sampling_before_episode_is_safe(final_trial):
    assert final_trial.evaluator.sample_step() is None
    assert final_trial.evaluator._update() is None


def test_unavailable_contact_result_survives_envelope_without_validation_failure(final_trial, tmp_path, monkeypatch):
    world = final_trial
    monkeypatch.setattr('wia.tasks.background_policy.record_background_movements', lambda session: [])
    task = SimpleNamespace(verify_unchanged=lambda: None,
        definition=SimpleNamespace(id='contact', repair={}, status='pending',
            private=SimpleNamespace(evaluation=SimpleNamespace(model_dump=lambda: world.config))),
        task_text='Establish a separated pair.', file_sha256='task', source_sha256='source')
    envelope = CategoryEvaluationEnvelope(world.evaluator, task, tmp_path / 'envelope')
    envelope.begin_episode(world.observation)
    world.live['contact_geometry'] = {'available': False, 'reason': 'Native separation unavailable.'}
    result = envelope.record(world.observation, action=None)
    assert envelope.failure is None
    assert result['native_schema'] == 'private_physical_contact_final_relation_v1'
    assert result['evaluation_validity'] in ('partial', 'unavailable')
    assert result['completion_score'] is None and result['task_success'] is False
    assert 'initial_native_qualification' not in result['native_result']['criteria']


def test_failed_initial_qualification_is_evaluation_validity_not_agent_constraint(final_trial):
    world = final_trial
    world.evaluator.begin_episode(world.observation)
    world.evaluator.initial_qualification['accepted'] = False
    result = settle(world)
    assert result['evaluation_validity'] == 'unavailable'
    assert result['task_success'] is False
    assert 'initial_native_qualification' not in result['criteria']
    assert result['constraints_satisfied'] is None


@pytest.mark.parametrize('operation', contact.OPERATIONS)
def test_explicit_historical_cycle_retains_evaluator_and_required_lifts(world_factory, operation, tmp_path):
    world = world_factory(contact)
    world.config['operation'] = operation
    evaluator = contact.create_evaluator(world.session, world.controller, world.config, tmp_path)
    assert isinstance(evaluator, contact.ContactCycleEvaluator)
    evaluator.begin_episode(SimpleNamespace(snapshot_id='initial', action_step=0))
    for _ in range(2 * contact.STABILITY_STEPS):
        evaluator.sample_step()
    assert not evaluator._update()['task_success']
    assert not evaluator.phase_lift_verified


@pytest.mark.parametrize('points,shape,expected', [
    ([[0, 0], [1, 0], [.5, .8660254]], 'equilateral', True),
    ([[0, 0], [1, 0], [.5, 1.]], 'isosceles', True),
    ([[0, 0], [1, 0], [.2, .7]], 'random triangle', True),
    ([[0, 0], [1, 0], [.5, 0.]], 'equilateral', False),
    ([[0, 0], [0, 0], [0, 0]], 'equilateral', False),
    ([[0, 0], [1, 0], [.5, .019]], 'collinear', True),
    ([[0, 0], [1, 0], [.5, .025]], 'collinear', False)])
def test_source_shape_geometry_rejects_collapse_and_enforces_normalized_metric(points, shape, expected):
    from wia.authoring.converters.spatial_relations_geometric_configuration import shape_geometry
    assert shape_geometry(points, shape)['satisfied'] is expected


@pytest.mark.parametrize('name', ['spatial_relations_geometric_configuration', 'spatial_relations_linear_alignment'])
def test_arrangement_conversion_preserves_scale_and_has_no_route_gate(name, tmp_path):
    import importlib
    module = importlib.import_module('wia.authoring.converters.' + name)
    converter, source = source_for(module)
    result = converter.convert(source, ConversionContext(tmp_path))
    task = result['task_definition']
    task_definition(task)
    assert result['status'] == 'generated'
    assert result['operation'] == 'arrange_released_source_shape'
    assert result['interaction_policy']['relocation_policy'] == 'rigid_group'
    assert 'route' not in task['public']['objective']
    assert task['private']['evaluation']['parameters']['release_stability_steps'] == 15


def test_new_line_arrangement_requires_measured_release_window_and_nonoverlap(world_factory, tmp_path):
    import numpy as np
    from wia.authoring.converters import spatial_relations_linear_alignment as alignment
    from wia.authoring.converters.spatial_relations_geometric_configuration import ShapeArrangementEvaluator
    world = world_factory(alignment, adapted=False)
    for obj in world.props:
        obj.scale = np.ones(3)
    world.config['operation'] = 'arrange_released_source_shape'
    world.config['parameters'].update(object_keys=list(alignment.OBJECTS),
        source_scales=[1., 1., 1.], target_shape='collinear', release_stability_steps=15)
    evaluator = ShapeArrangementEvaluator(world.session, world.controller, world.config, tmp_path)
    obs = SimpleNamespace(snapshot_id='s', action_step=0)
    result = evaluator.begin_episode(obs)
    assert evaluator.qualification['accepted'], evaluator.qualification
    assert result['task_success'] is False
    for _ in range(15):
        evaluator.sample_step()
    assert evaluator.record(obs)['task_success'] is True
    world.props[1].position[:] = world.props[0].position + [.1, 0., 0.]
    result = evaluator.record(obs)
    assert not result['task_success']
    assert result['completion_items'][0]['completed'] is True
    assert result['native_state']['checks']['nonoverlapping_objects'] is False


def test_size_transport_requires_named_destination_and_preserves_source_metric(tmp_path):
    from wia.authoring.converters import dimensional_size as size
    converter, source = source_for(size)
    pending = converter.convert(source, ConversionContext(tmp_path))
    assert pending['status'] == 'needs_human_decision'
    assert pending['failure_reason'] == 'named_placement_destination_required'
    config = {'placement_destination': {'binding': {'name': 'native_table', 'category': 'table'},
        'public_label': 'the designated table', 'relation': 'on_top'}}
    revised = converter.convert(source, ConversionContext(tmp_path, seed=7, config=config))
    task = revised['task_definition']
    task_definition(task)
    p = task['private']['evaluation']['parameters']
    assert p['comparison_kind'] == 'identical_asset_uniform_linear_scale'
    assert p['placement_destination'] == config['placement_destination']
    assert task['private']['evaluation']['operation'] == size.PLACEMENT_OPERATION
    assert 'uniform linear dimensions' in task['public']['objective']
    legacy = converter.convert(source, ConversionContext(tmp_path, config={'operation': 'legacy_grasp_lift'}))
    assert legacy['task_definition']['private']['evaluation']['type'] == 'grasp_lift'


@pytest.fixture
def comparison_trial(tmp_path, monkeypatch):
    from wia.authoring.converters import dimensional_size as size
    objects = {'selected': SimpleNamespace(name='selected'), 'other': SimpleNamespace(name='other'),
        'table': SimpleNamespace(name='table')}
    session = SimpleNamespace(scene_name='scene', room_name='room',
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda _, name: objects[name])))
    parameters = {'selected_key': 'selected', 'source_objects': {'selected': {}, 'other': {}},
        'comparison_kind': 'identical_asset_uniform_linear_scale', 'expected_identity': 'near the vase',
        'release_stability_steps': 2, 'maximum_release_speed_m_s': .02,
        'maximum_release_angular_speed_rad_s': .1}
    config = {'scene': 'scene', 'room': 'room', 'parameters': parameters,
        'bindings': {'target': 'selected', 'destination': 'table', 'selected': 'selected', 'other': 'other'}}
    comparison = {'accepted': True, 'checks': {}, 'objects': {key: {'position': [i, 0., .5],
        'quaternion_xyzw': [0., 0., 0., 1.], 'source_scale_match': True}
        for i, key in enumerate(('selected', 'other'))}}
    monkeypatch.setattr(size, 'comparison_measurements', lambda *args: copy.deepcopy(comparison))
    evaluator = size.ComparisonPlacementEvaluator(session, None, config, tmp_path)
    measurement = {'gripper': {'holding': False, 'constraint_active': False,
        'release_pending': False, 'object_name': None}, 'destination_relation': True,
        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'robot_withdrawn': True, 'measurement_errors': {}}
    monkeypatch.setattr(evaluator, 'measure', lambda: copy.deepcopy(measurement))
    obs = SimpleNamespace(snapshot_id='s', action_step=0)
    evaluator.begin_episode(obs)
    return evaluator, comparison, measurement, obs


def test_comparison_before_grasp_and_stable_transport_are_distinct(comparison_trial):
    evaluator, comparison, measurement, obs = comparison_trial
    evaluator.report_answer('near the vase', obs)
    measurement['gripper'].update(holding=True, constraint_active=True, object_name='selected')
    evaluator.sample_step()
    measurement['gripper'].update(holding=False, constraint_active=False, object_name=None)
    comparison['objects']['selected']['position'][0] += .5
    evaluator.sample_step()
    evaluator.sample_step()
    result = evaluator.record(obs)
    assert result['task_success'] and result['completion_score'] == 1.


def test_comparison_report_after_intervention_cannot_earn_identity(comparison_trial):
    evaluator, comparison, measurement, obs = comparison_trial
    measurement['gripper'].update(holding=True, constraint_active=True, object_name='selected')
    evaluator.sample_step()
    evaluator.report_answer('near the vase', obs)
    result = evaluator.record(obs)
    assert result['intervened_before_answer']
    assert result['completion_items'][0]['completed'] is False
    assert result['completion_items'][1]['completed'] is True


@pytest.mark.parametrize('comparisons,reason', [
    ([('a', '<', 'b')], 'incomplete_capacity_comparison_graph'),
    ([('a', '<', 'b'), ('b', '<', 'c'), ('c', '<', 'a')], 'capacity_order_cycle_or_ambiguity'),
    ([('a', '=', 'b'), ('a', '<', 'c'), ('b', '>', 'c')], 'capacity_order_cycle_or_ambiguity')])
def test_capacity_graph_rejects_missing_comparisons_and_cycles(comparisons, reason):
    from wia.authoring.converters.composition_manipulation import capacity_order_graph
    result = capacity_order_graph(['a', 'b', 'c'], comparisons)
    assert not result['accepted'] and result['reason'] == reason


def test_capacity_graph_retains_explicit_ties():
    from wia.authoring.converters.composition_manipulation import capacity_order_graph
    result = capacity_order_graph(['a', 'b', 'c'], [('a', '=', 'b'), ('a', '<', 'c'), ('b', '<', 'c')])
    assert result['accepted'] and result['ordered_tie_groups'] == [['a', 'b'], ['c']]


def test_capacity_composition_retires_docking_and_requires_exact_consistent_world():
    from wia.authoring.converters import composition_manipulation as policy
    from harness_integration.test_composition_manipulation import liquid, groups
    first, second = liquid('first', policy.LIQUID_OPERATIONS[0]), liquid('second', policy.LIQUID_OPERATIONS[1])
    assert len(groups(first, second)) == 2
    for record, task, scene in (first, second):
        evaluation = task['private']['evaluation']
        evaluation['operation'] = policy.CAPACITY_OPERATION
        evaluation['parameters'] = {'expected_order': 'left < right', 'capacity_evidence': {
            'valid': True, 'metric': 'source_retained_water_particle_capacity',
            'expected_native_scales': {'left': 1., 'right': 1.}}}
    assert len(groups(first, second)) == 1
    second[1]['private']['evaluation']['parameters']['expected_order'] = 'right < left'
    assert len(groups(first, second)) == 2
    second[1]['private']['evaluation']['parameters']['expected_order'] = 'left < right'
    second[2]['scene']['variants']['adapted']['patches'][0]['position'][0] += .3
    second[1]['scene'] = copy.deepcopy(second[2]['scene'])
    assert len(groups(first, second)) == 2


def test_liquid_capacity_uses_source_fill_counts_not_final_amount_or_outer_volume(tmp_path, monkeypatch):
    from wia.authoring.converters import physical_structure_liquid_volume as liquid
    converter, source = source_for(liquid)
    monkeypatch.setattr(liquid, '_robot_start', lambda *args: (None, {}))
    result = converter.convert(source, ConversionContext(tmp_path))
    task_definition(result['task_definition'])
    assert task_definition(result['task_definition']).private.evaluation.operation == liquid.CAPACITY_OPERATION
    evidence = result['private_capacity_evidence']
    assert evidence['valid']
    assert evidence['metric'] == 'source_retained_water_particle_capacity'
    assert evidence['calibration']['cross_trial_comparison_calibrated'] is False
    assert 'No docking tour' in result['task_definition']['public']['success_criteria']
    raw = source.raw
    payload = source.payload
    payload['obj1']['capacity_measured'] = payload['obj2']['capacity_measured']
    payload['capacity_diff_measured'] = 0
    raw['metadata_json'] = payload
    changed_path = tmp_path / 'tied.json'
    changed_path.write_text(json.dumps(raw))
    changed = SourceInstance(changed_path, source.relative_path, *converter.category)
    _, _, entries, _ = liquid._source_data(changed)
    tie = liquid.capacity_evidence(changed, entries, ConversionContext(tmp_path))
    assert tie['capacity_tie'] and not tie['valid']
    assert tie['reason'] == 'source_capacity_answer_conflict_or_unresolved_tie'

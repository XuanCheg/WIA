"""Exercise measured spatial-tray reduction without native simulation.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B
-m pytest -q -p no:cacheprovider tests/test_wia_spatial_tray.py
Synthetic native samples verify protocol behavior, not manipulation feasibility.
"""
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.authoring.converters import wia_spatial_tray as adapter
from wia.authoring.converters.spatial_relations_physical_contact import _contact_geometry
from wia.tasks.wia_spatial_tray_scoring import SpatialTrayProgress, validate_parameters
import wia.authoring.commands.build_wia_spatial_tray as builder


def parameters(kind='shape'):
    targets = {'a': 'native_a', 'b': 'native_b', 'c': 'native_c'}
    return {'revision': 'wia_spatial_tray_v1',
        'scene_contract': {'scene_model': 'explicit_test_scene', 'room': 'explicit_room_0'},
        'targets': targets, 'initial_supports': {key: 'support_' + key for key in targets},
        'tray': {'deck_key': 'deck', 'table_support_key': 'table', 'rim_keys': ['rim'],
                 'mounting_mode': 'fixed_fixture_on_native_table',
                 'usable_world_xy_m': [[-1., -1.], [1., 1.]], 'deck_top_world_z_m': .8},
        'goal_relation': ({'kind': 'shape', 'shape': 'equilateral'} if kind == 'shape' else
                          {'kind': 'contact', 'pair': ['a', 'b'], 'relation': 'touching'}),
        'scoring': {'minimum_lift_m': .05, 'minimum_displacement_m': .1,
            'stability_duration_s': .5, 'maximum_linear_speed_m_s': .02,
            'maximum_angular_speed_rad_s': .1, 'max_sample_gap_s': 1.,
            'fixture_position_tolerance_m': .01, 'fixture_angle_tolerance_rad': .05,
            'deck_height_tolerance_m': .01, 'maximum_bottom_gap_m': .01,
            'penetration_tolerance_m': .001, 'minimum_separation_m': .05}}


def contact(relation='touching', *, available=True):
    return {'available': available, 'relation': relation, 'penetrating': False,
            'minimum_separation_m': .1 if relation == 'separated' else -.0001,
            'separation_source': 'synthetic_test_evidence'}


def sample(now=0., *, initial=False, held=()):
    points = [[0., 0.], [.6, 0.], [.3, float(np.sqrt(.27))]]
    rows = {}
    for index, (key, name) in enumerate(parameters()['targets'].items()):
        x, y = points[index]
        bottom = .8
        if initial:
            x -= 3.
            bottom = .6
        elif name in held:
            bottom = 1.
        rows[key] = {'name': name, 'position': [x, y, bottom + .05],
            'scale': [1., 1., 1.],
            'aabb_min': [x - .05, y - .05, bottom], 'aabb_max': [x + .05, y + .05, bottom + .1],
            'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'dynamic_collidable': True,
            'robot_contact': name in held, 'on_top_deck': not initial and not held,
            'deck_contact_geometry': contact('separated' if initial or held else 'touching')}
    return {'native_time_s': now, 'control_valid': True, 'fixtures_valid': True, 'tray_valid': True,
            'initial_supports_valid': True, 'targets': rows, 'contact_geometry': contact(),
            'grasp': {'available': True, 'held_names': list(held), 'released': not held}}


def retrieved(kind='shape'):
    scorer = SpatialTrayProgress(parameters(kind))
    scorer.begin(sample(initial=True))
    for index, name in enumerate(parameters()['targets'].values(), 1):
        scorer.update(sample(index * .1, held=[name]))
    return scorer


def test_initial_detached_hands_do_not_require_terminal_aperture():
    value = sample(initial=True)
    value['grasp'].update(detached=True, released=False)
    scorer = SpatialTrayProgress(parameters())
    initial = scorer.begin(value)
    assert initial['evaluation_validity'] == 'complete'
    assert not initial['task_success']
    current = sample(.1)
    current['grasp'].update(detached=True, released=False)
    scorer.update(current)
    assert not scorer.result()['task_success']


def stable_result(scorer, value=None):
    value = sample(.4) if value is None else deepcopy(value)
    scorer.update(value)
    value['native_time_s'] = .9
    return scorer.update(value)


def test_retrieved_equilateral_success_requires_simultaneous_released_dwell():
    scorer = retrieved()
    first = scorer.update(sample(.4))
    assert not first['task_success']
    before = scorer.result()
    for _ in range(20):
        assert scorer.result() == before
    result = scorer.update(sample(.9))
    assert result['task_success'] and result['physical_success']
    assert result['stable_duration_s'] == pytest.approx(.5)
    assert all(state['acquired'] and state['moved'] for state in result['per_target'].values())


@pytest.mark.parametrize('failure', ['wrong_relation', 'outside', 'held', 'robot_support', 'not_on_deck',
                                      'penetrating', 'moving', 'unavailable_deck'])
def test_terminal_counterexamples(failure):
    scorer, value = retrieved(), sample(.4)
    row = value['targets']['c']
    if failure == 'wrong_relation':
        row['aabb_min'][1] = -.05
        row['aabb_max'][1] = .05
    elif failure == 'outside':
        row['aabb_min'][0] = .99
        row['aabb_max'][0] = 1.09
    elif failure == 'held':
        value['grasp'].update(held_names=['native_c'], released=False)
    elif failure == 'robot_support':
        row['robot_contact'] = True
    elif failure == 'not_on_deck':
        row['deck_contact_geometry'] = contact('separated')
    elif failure == 'penetrating':
        row['deck_contact_geometry'].update(relation='penetrating', penetrating=True)
    elif failure == 'moving':
        row['linear_speed_m_s'] = .03
    else:
        row['deck_contact_geometry'] = {'available': False, 'reason': 'sleeping_pair_separation_unavailable'}
    result = stable_result(scorer, value)
    assert not result['task_success']
    assert result['stable_duration_s'] == 0.
    if failure == 'unavailable_deck':
        assert result['evaluation_validity'] == 'unavailable'
        assert result['completion_score'] is None


def test_already_correct_geometry_without_retrieval_cannot_pass():
    scorer = SpatialTrayProgress(parameters())
    scorer.begin(sample(initial=True))
    result = stable_result(scorer)
    assert result['declared_relation']['satisfied']
    assert not result['task_success']
    assert not any(state['acquired'] for state in result['per_target'].values())


def final_state_parameters(kind='shape'):
    p = parameters(kind)
    p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None,
                        minimum_displacement_m=None)
    return p


@pytest.mark.parametrize('kind', ['shape', 'contact'])
def test_final_state_correct_tray_placement_passes_without_grasp_lift_or_motion_checkpoint(kind):
    scorer = SpatialTrayProgress(final_state_parameters(kind))
    scorer.begin(sample(initial=True))
    result = stable_result(scorer)
    assert result['task_success'] and result['physical_success']
    assert result['schema'] == 'wia_spatial_tray_final_state_progress_v1'
    assert result['manipulation_revision'] == 'final_state_v1'
    assert [row['id'] for row in result['completion_items']] == [
        'targets_on_tray', 'declared_relation', 'stable_released_tray_placement']
    assert all(row['kind'] == 'terminal_condition' for row in result['completion_items'])
    assert not any(row['acquired'] or row['moved'] for row in result['per_target'].values())


@pytest.mark.parametrize('failure', ['relation', 'outside', 'held', 'robot_support', 'penetration',
                                    'unstable', 'unavailable', 'identity', 'fixed_target', 'control', 'fixture',
                                    'scale', 'scale_unavailable'])
def test_final_state_tray_still_requires_measured_terminal_geometry_and_integrity(failure):
    scorer = SpatialTrayProgress(final_state_parameters('contact'))
    scorer.begin(sample(initial=True))
    value = sample(.4)
    row = value['targets']['a']
    if failure == 'relation':
        value['contact_geometry'] = contact('separated')
    elif failure == 'outside':
        row['aabb_min'][0], row['aabb_max'][0] = 1.01, 1.11
    elif failure == 'held':
        value['grasp'].update(held_names=['native_a'], released=False)
    elif failure == 'robot_support':
        row['robot_contact'] = True
    elif failure == 'penetration':
        row['deck_contact_geometry']['penetrating'] = True
    elif failure == 'unstable':
        row['angular_speed_rad_s'] = .2
    elif failure == 'unavailable':
        row['deck_contact_geometry']['available'] = False
    elif failure == 'identity':
        row['name'] = 'replacement'
    elif failure == 'fixed_target':
        row['dynamic_collidable'] = False
    elif failure == 'scale':
        row['scale'][0] = 1.1
    elif failure == 'scale_unavailable':
        row.pop('scale')
    else:
        value['control_valid' if failure == 'control' else 'fixtures_valid'] = False
    result = stable_result(scorer, value)
    assert not result['task_success'] and result['stable_duration_s'] == 0.
    if failure in ('unavailable', 'identity', 'scale_unavailable'):
        assert result['evaluation_validity'] == 'unavailable' and result['completion_score'] is None
    if failure == 'identity':
        assert not next(row['completed'] for row in result['completion_items'] if row['id'] == 'targets_on_tray')


def test_final_state_tray_preserves_separation_threshold_and_clock_gap_guards():
    p = final_state_parameters('contact')
    p['goal_relation']['relation'] = 'separated'
    scorer = SpatialTrayProgress(p)
    scorer.begin(sample(initial=True))
    value = sample(.4)
    value['contact_geometry'] = contact('separated')
    value['contact_geometry']['minimum_separation_m'] = .049
    assert not stable_result(scorer, value)['task_success']
    value['contact_geometry']['minimum_separation_m'] = .05
    value['native_time_s'] = 1.
    scorer.update(value)
    value['native_time_s'] = 2.1
    assert not scorer.update(value)['task_success']
    assert scorer.result()['sample_gap_exceeded']
    value['native_time_s'] = .5
    assert scorer.update(value)['evaluation_validity'] == 'invalid'


@pytest.mark.parametrize('change', [
    {'manipulation_revision': 'unknown'}, {'manipulation_revision': None},
    {'minimum_lift_m': .05}, {'minimum_displacement_m': .1}, {'minimum_separation_m': .049}])
def test_final_state_tray_rejects_unknown_revisions_history_thresholds_and_weakened_geometry(change):
    p = final_state_parameters()
    p['scoring'].update(change)
    with pytest.raises(ValueError):
        validate_parameters(p)


def test_initially_in_destination_is_not_accepted_as_retrieval():
    scorer = SpatialTrayProgress(parameters())
    result = scorer.begin(sample())
    assert result['evaluation_validity'] == 'unavailable'
    assert any('initially_inside_destination' in error for error in result['initial_measurement_errors'])
    assert not stable_result(scorer)['task_success']


def test_contact_uses_declared_native_pair_evidence_and_preserves_unavailability():
    assert stable_result(retrieved('contact'))['task_success']
    scorer, value = retrieved('contact'), sample(.4)
    value['contact_geometry'] = {'available': False, 'relation': 'unresolved',
                                 'reason': 'No pair-specific separation samples.'}
    result = stable_result(scorer, value)
    assert not result['task_success'] and result['evaluation_validity'] == 'unavailable'
    assert result['declared_relation']['reason'] == 'No pair-specific separation samples.'
    assert not _contact_geometry({}, False, 0.)['available']


def test_collinear_reuses_existing_geometry_and_distinctness_tolerances():
    p = parameters()
    p['goal_relation']['shape'] = 'collinear'
    scorer = SpatialTrayProgress(p)
    scorer.begin(sample(initial=True))
    for index, name in enumerate(p['targets'].values(), 1):
        scorer.update(sample(index * .1, held=[name]))
    value = sample(.4)
    for index, row in enumerate(value['targets'].values()):
        row['aabb_min'][:2] = [index * .3 - .05, -.05]
        row['aabb_max'][:2] = [index * .3 + .05, .05]
    assert stable_result(scorer, value)['task_success']


def test_missing_control_evidence_and_sample_gap_reset_dwell():
    scorer = retrieved()
    scorer.update(sample(.4))
    value = sample(.6)
    value['control_valid'] = None
    assert scorer.update(value)['evaluation_validity'] == 'unavailable'
    assert not scorer.update(sample(.9))['task_success']
    assert scorer.update(sample(1.41))['task_success']
    assert not scorer.update(sample(2.5))['task_success']


def test_runtime_state_setter_and_fixture_changes_invalidate_episode():
    for field in ('control_valid', 'fixtures_valid'):
        scorer = retrieved()
        value = sample(.4)
        value[field] = False
        assert scorer.update(value)['evaluation_validity'] == 'invalid'
        assert not stable_result(scorer)['task_success']


def test_parameters_require_explicit_values_and_do_not_weaken_legacy_tolerances():
    p = parameters()
    del p['scoring']['minimum_displacement_m']
    with pytest.raises(ValueError, match='explicit'):
        validate_parameters(p)
    p = parameters()
    p['scoring']['penetration_tolerance_m'] = .002
    with pytest.raises(ValueError, match='weaken'):
        validate_parameters(p)
    p = parameters()
    p['targets']['b'] = p['targets']['a']
    with pytest.raises(ValueError, match='distinct exact'):
        validate_parameters(p)


def test_initialization_delegates_and_runtime_placement_is_forbidden(monkeypatch):
    calls = []
    p = {'revision': 'wia_spatial_tray_v1', 'scene_contract': parameters()['scene_contract'],
         'layout': {'scene_model': 'explicit_test_scene', 'rooms': ['explicit_room_0'],
                    'objects': [{'name': 'native_a'}],
                    'supported_items': [{'name': 'native_a', 'support': 'existing_native_table'}]}}
    monkeypatch.setattr(adapter.active_layout, 'configure_scene', lambda *args: calls.append('configure'))
    monkeypatch.setattr(adapter.active_layout, 'postprocess_joint_world', lambda *args: calls.append('postprocess'))
    monkeypatch.setattr(adapter.active_layout, 'initialize_scene', lambda *args: calls.append('initialize'))
    session = SimpleNamespace(_category_initialization_active=False)
    adapter.configure_scene({}, p)
    adapter.postprocess_joint_world(session, p)
    with pytest.raises(RuntimeError, match='initialization-only'):
        adapter.initialize_scene(session, p)
    session._category_initialization_active = True
    adapter.initialize_scene(session, p)
    assert calls == ['configure', 'postprocess', 'initialize']


@pytest.mark.parametrize('binding_style', ['dict', 'string'])
def test_builder_preserves_definition_bindings_and_validates_native_name_mapping(binding_style):
    p = parameters()
    names = {**p['targets'], **{'support_' + key: 'existing_native_' + key for key in p['targets']},
             'deck': 'native_deck', 'table': 'native_table', 'rim': 'native_rim'}
    bindings = {key: {'name': name} for key, name in names.items()}
    source = {'provider': 'esi', 'json': 'Synthetic/Spatial/explicit_test_scene/explicit_room_0/sample.json'}
    scene_config = {'policy': 'native_then_minimal_patch_else_skip',
                    'variants': {'original': {'patches': []}, 'adapted': {'patches': []}}}
    parent_task = {'kind': 'category_robot_task', 'id': 'synthetic_parent', 'version': 1, 'status': 'pending',
        'source': source, 'public': {'objective': 'Synthetic parent.', 'success_criteria': 'Parent condition.',
                                    'environment': 'Synthetic test only.', 'constraints': []},
        'requirements': {'capabilities': ['gripper']},
        'private': {'bindings': bindings, 'evaluation': {'type': 'category_operation', 'module': adapter.MODULE,
            'operation': adapter.OPERATION, 'binding_keys': list(bindings), 'parameters': p}},
        'scene': scene_config, 'qualification': {'on_failure': 'skip'}, 'defaults': {'seed': 4}}
    parent_scene = {'kind': 'scene_patch', 'id': 'synthetic_parent_scene', 'version': 1, 'status': 'pending',
                    'source': source, 'scene': scene_config, 'bindings': bindings, 'seed': 4}
    entry = {'document_alias': 'T26', 'variant_id': 'synthetic_fixture_v1', 'label': 'CPU fixture only',
        'source_identity': source['json'], 'parameters': p,
        'bindings': bindings if binding_style == 'dict' else names,
        'public': {'objective': 'Retrieve and arrange.', 'success_criteria': 'Released tray placement.'},
        'layout': {'scene_model': 'explicit_test_scene', 'rooms': ['explicit_room_0'],
                   'objects': [{'name': name} for name in p['targets'].values()],
                   'supported_items': [{'name': name, 'support': names[p['initial_supports'][key]]}
                                       for key, name in p['targets'].items()]}}
    before = deepcopy((parent_task, parent_scene, entry))
    task, scene = builder.translate_definitions(parent_task, parent_scene, entry, {'design_file': 'synthetic'})
    assert task['private']['bindings'] == scene['bindings'] == bindings
    assert task['version'] == 2 and task['defaults']['seed'] == scene['seed'] == 4
    assert (parent_task, parent_scene, entry) == before
    adapter.validate_contract({**task['private']['evaluation'], 'bindings': names,
                               'scene': 'explicit_test_scene', 'room': 'explicit_room_0'})


def test_shared_native_counter_support_aliases_are_allowed_but_targets_and_tray_are_distinct():
    p = parameters()
    names = {**p['targets'], 'support_a': 'native_counter', 'support_b': 'native_counter',
             'support_c': 'other_native_counter', 'table': 'native_counter',
             'deck': 'native_deck', 'rim': 'native_rim'}
    config = {'module': adapter.MODULE, 'operation': adapter.OPERATION, 'parameters': p,
              'scene': 'explicit_test_scene', 'room': 'explicit_room_0',
              'bindings': names, 'binding_keys': list(names)}
    adapter.validate_contract(config)
    p['initial_supports']['b'] = 'support_a'
    adapter.validate_contract(config)
    names['support_a'] = 'native_deck'
    with pytest.raises(ValueError, match='distinct native identities'):
        adapter.validate_contract(config)


@pytest.fixture
def native_fixture_world(monkeypatch):
    """Supply actual-shaped native geometry with unsupported fixed-body velocities."""
    p = parameters()
    states = ModuleType('omnigibson.object_states')
    states.OnTop = object()
    monkeypatch.setitem(sys.modules, states.__name__, states)
    calls, objects = [], {}

    def unsupported_velocity(name):
        def getter():
            calls.append(name)
            raise AttributeError('RigidKinematicPrim has no get_linear_velocity')
        return getter

    def body(name, low, high, *, fixed):
        low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
        return SimpleNamespace(name=name, category='native_category', model='native_model',
            scale=np.ones(3),
            visual_only=False, fixed_base=fixed, aabb=(low, high),
            get_position_orientation=lambda: ((low + high) / 2, np.array([0., 0., 0., 1.])),
            get_linear_velocity=unsupported_velocity(name) if fixed else lambda: np.zeros(3),
            get_angular_velocity=unsupported_velocity(name) if fixed else lambda: np.zeros(3),
            contact_list=lambda: [], states={states.OnTop: SimpleNamespace(get_value=lambda other: False)})

    objects['deck'] = body('native_deck', [-1., -1., .76], [1., 1., .8], fixed=True)
    objects['table'] = body('native_table', [-2., -2., 0.], [2., 2., .76], fixed=True)
    objects['rim'] = body('native_rim', [-1., -.1, .8], [-.95, .1, .9], fixed=True)
    for key, row in sample(initial=True)['targets'].items():
        objects[key] = body(row['name'], row['aabb_min'], row['aabb_max'], fixed=False)
        objects[p['initial_supports'][key]] = body('native_support_' + key,
            [-4., -1., 0.], [-2., 1., .6], fixed=True)
    session = SimpleNamespace(robot=SimpleNamespace(prim_path='/World/robot'),
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(current_time=0.))),
        category_interaction_setup={'supported_items': {name: {
            'support': objects[p['initial_supports'][key]].name, 'accepted': True}
            for key, name in p['targets'].items()}})
    monkeypatch.setattr(adapter, '_grasp', lambda *args: ({'available': True, 'held_names': [], 'released': True}, {}))
    monkeypatch.setattr(adapter, '_control', lambda *args: (True, {'available': True}))
    monkeypatch.setattr(adapter, '_pair', lambda *args: contact('separated'))
    return SimpleNamespace(p=p, config={'parameters': p}, session=session, objects=objects,
        velocity_calls=calls, unsupported_velocity=unsupported_velocity)


def test_fixed_fixture_geometry_does_not_require_or_synthesize_velocity(native_fixture_world):
    world = native_fixture_world
    measured = adapter._measure(world.session, None, world.config, world.objects)
    assert measured['fixtures_valid'] is True and measured['tray_valid'] is True
    assert world.velocity_calls == []
    for key, row in measured['fixtures'].items():
        obj = world.objects[key]
        assert row['name'] == obj.name and row['category'] == obj.category and row['model'] == obj.model
        assert row['position'] == pytest.approx(obj.get_position_orientation()[0])
        assert row['aabb_min'] == pytest.approx(obj.aabb[0])
        assert row['aabb_max'] == pytest.approx(obj.aabb[1])
        assert row['preserved'] is True
        assert 'linear_speed_m_s' not in row and 'angular_speed_rad_s' not in row
    assert SpatialTrayProgress(world.p).begin(measured)['evaluation_validity'] == 'complete'
    followup = adapter._measure(world.session, None, world.config, world.objects, measured)
    assert followup['fixtures_valid'] is True and world.velocity_calls == []
    world.objects['deck'].get_position_orientation = lambda: (np.array([.02, 0., .78]), np.array([0., 0., 0., 1.]))
    changed = adapter._measure(world.session, None, world.config, world.objects, measured)
    assert changed['fixtures_valid'] is False


def test_dynamic_target_missing_native_velocity_remains_unavailable(native_fixture_world):
    world = native_fixture_world
    world.objects['a'].get_linear_velocity = world.unsupported_velocity('native_a')
    measured = adapter._measure(world.session, None, world.config, world.objects)
    assert measured['fixtures_valid'] is True and measured['tray_valid'] is True
    assert world.velocity_calls == ['native_a']
    target = measured['targets']['a']
    assert 'linear_speed_m_s' not in target and 'angular_speed_rad_s' not in target
    assert any('get_linear_velocity' in error for error in target['measurement_errors'])
    result = SpatialTrayProgress(world.p).begin(measured)
    assert result['evaluation_validity'] == 'unavailable' and result['task_success'] is False


@pytest.mark.parametrize('key', ['a', 'deck', 'rim', 'table'])
def test_final_state_adapter_measures_native_scale_and_rejects_changes_with_unchanged_bounds(native_fixture_world, key):
    world = native_fixture_world
    world.p['scoring'] = final_state_parameters()['scoring']
    measured = adapter._measure(world.session, None, world.config, world.objects)
    scorer = SpatialTrayProgress(world.p)
    assert scorer.begin(measured)['evaluation_validity'] == 'complete'
    assert measured['targets']['a']['scale'] == [1., 1., 1.]
    assert measured['fixtures']['deck']['scale'] == [1., 1., 1.]
    world.objects[key].scale[0] = 1.01
    world.session.pipeline.og.sim.current_time = .1
    changed = adapter._measure(world.session, None, world.config, world.objects, measured)
    result = scorer.update(changed)
    assert result['evaluation_validity'] == 'invalid' and not result['task_success']
    if key == 'a':
        assert 'a:asset_scale_changed' in result['constraint_violations']
    else:
        assert changed['fixtures_valid'] is False


def test_final_state_missing_native_scale_is_unavailable(native_fixture_world):
    world = native_fixture_world
    world.p['scoring'] = final_state_parameters()['scoring']
    del world.objects['a'].scale
    measured = adapter._measure(world.session, None, world.config, world.objects)
    result = SpatialTrayProgress(world.p).begin(measured)
    assert result['evaluation_validity'] == 'unavailable' and not result['task_success']
    assert any('native_scale_unavailable' in reason for reason in result['initial_measurement_errors'])


def test_primitive_tray_fixtures_without_model_metadata_preserve_valid_geometry(native_fixture_world):
    world = native_fixture_world
    primitive_keys = [world.p['tray']['deck_key'], *world.p['tray']['rim_keys']]
    for key in primitive_keys:
        del world.objects[key].model
    measured = adapter._measure(world.session, None, world.config, world.objects)
    assert measured['fixtures_valid'] is True and measured['tray_valid'] is True
    assert world.velocity_calls == []
    for key in primitive_keys:
        row = measured['fixtures'][key]
        assert row['model'] is None and row['model_available'] is False
        assert row['preserved'] is True
        assert row['aabb_min'] == pytest.approx(world.objects[key].aabb[0])
        assert row['aabb_max'] == pytest.approx(world.objects[key].aabb[1])
    assert measured['fixtures']['table']['model'] == world.objects['table'].model
    assert measured['fixtures']['table']['model_available'] is True
    followup = adapter._measure(world.session, None, world.config, world.objects, measured)
    assert followup['fixtures_valid'] is True and followup['tray_valid'] is True
    assert SpatialTrayProgress(world.p).begin(measured)['evaluation_validity'] == 'complete'


@pytest.mark.parametrize('problem', ['nonfinite', 'zero_quaternion', 'invalid_bounds', 'visual_flag', 'fixed_flag'])
def test_fixture_snapshot_rejects_invalid_geometry_or_non_native_flags(native_fixture_world, problem):
    obj = native_fixture_world.objects['deck']
    if problem == 'nonfinite':
        obj.aabb[0][0] = float('nan')
    elif problem == 'zero_quaternion':
        obj.get_position_orientation = lambda: (np.zeros(3), np.zeros(4))
    elif problem == 'invalid_bounds':
        obj.aabb[0][0] = obj.aabb[1][0]
    elif problem == 'visual_flag':
        obj.visual_only = 'false'
    else:
        obj.fixed_base = 1
    with pytest.raises(ValueError):
        adapter._fixture_snapshot(obj)
    assert native_fixture_world.velocity_calls == []

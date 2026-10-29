"""Check accessible generator rules without Native simulation or GPU work.

Run with the behavior Python from the repository root:
python -m pytest -q tests/test_accessible_generators_a.py
Native-shaped fixtures test initialization boundaries, not physical feasibility.
"""
import copy
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance, map_records
from wia.authoring.converters import action_order as action
from wia.authoring.converters import metric_comparison_spatial_distance as distance
from wia.authoring.converters import physical_dynamics_stacking_stability as stacking
from wia.authoring.converters import physical_dynamics_inclined_plane as inclined
from wia.authoring.converters import physical_structure_deformable as deformable
from wia.authoring.converters import physical_structure_liquid_volume as liquid
from wia.tasks import interaction_setup
from wia.tasks.loader import task_definition
from harness_integration.test_annotation_manipulation_categories import incline_trial, native_types


CHANGED = [action, distance, stacking, deformable, liquid, inclined]


def source_for(module):
    converter = module.ActionOrderConverter() if module is action else module.CategoryConverter()
    corpus = ROOT / 'ESI/dataset/json_clean'
    big, small = converter.category
    for path in sorted((corpus / big / small).rglob('*.json')):
        if len(path.relative_to(corpus).parts) != 5:
            continue
        source = SourceInstance(path, str(path.relative_to(corpus)), big, small)
        if converter.inspect(source)['eligible']:
            return converter, source
    raise AssertionError('An eligible original source is required for ' + small)


def deferred_start(monkeypatch, module):
    if module is inclined:
        monkeypatch.setattr(module, '_robot_start', lambda *args: None)
    if module in (stacking, deformable, liquid):
        monkeypatch.setattr(module, '_robot_start', lambda *args: (
            None, {'reason': 'cpu_screen_inconclusive', 'failure_reason': 'cpu_screen_inconclusive'}))


@pytest.mark.parametrize('module', CHANGED)
def test_complete_serialization_retains_original_private_qa(module, tmp_path, monkeypatch):
    deferred_start(monkeypatch, module)
    converter, source = source_for(module)
    original = source.path.read_bytes()
    manifest = converter.convert_all([source], ConversionContext(tmp_path))
    assert manifest['source_count'] == 1
    record = manifest['records'][0]
    assert record['status'] == 'generated', record.get('exception_summary', record)
    import yaml
    task = yaml.safe_load(Path(record['task_file']).read_text())
    task_definition(task)
    assert task['status'] == 'pending'
    assert task['scene']['variants']['original']['patches'] == []
    public = task['public']
    assert not {'answer', 'metadata_json', 'source_objects', 'bindings'} & set(public)
    qa = json.loads((Path(record['diagnostic_dir']) / 'private_qa.json').read_text())
    assert {key: qa[key] for key in ('answer', 'options_json', 'metadata_json')} == {
        key: source.raw.get(key) for key in ('answer', 'options_json', 'metadata_json')}
    assert source.path.read_bytes() == original
    policy = record['interaction_policy']
    assert policy['support_search'] == 'measured_surfaces'
    assert policy['allow_fallback_table']
    assert policy['source_camera_policy'] == 'preserve'
    assert policy['minimum_grasp_height_m'] == .35
    assert policy['maximum_grasp_height_m'] == 1.10


@pytest.mark.parametrize('module', [action, stacking])
def test_unrelated_initial_candidates_are_independent_without_answer_selection(module, tmp_path, monkeypatch):
    deferred_start(monkeypatch, module)
    converter, source = source_for(module)
    result = converter.convert(source, ConversionContext(tmp_path))
    policy = result['interaction_policy']
    keys = policy['group_bindings']
    assert policy['relocation_policy'] == 'independent_groups'
    assert policy['placement_groups'] == [[key] for key in keys]
    assert policy['manipulation_bindings'] == keys
    bindings = result['task_definition']['private']['bindings']
    assert set(keys) == set(bindings) - {'floor'}
    if module is action:
        assert {'base_candidate_0', 'base_candidate_1'} <= set(keys)
        assert all(not key.startswith('top_') or key in keys for key in bindings)
    patches = result['task_definition']['scene']['variants']['adapted']['patches']
    for key in keys:
        assert any(p.get('binding') == key and p['operation'] == 'object_physics'
                   and p.get('fixed_base') is False and p.get('phase', 'before_construction') == 'before_construction'
                   for p in patches)
    assert not any(p['operation'] in ('object_scale', 'object_pose') for p in patches)


def test_action_owned_hook_defers_to_one_final_shared_initializer(monkeypatch):
    usd = ModuleType('omnigibson.utils.usd_utils')
    usd.PoseAPI = SimpleNamespace(invalidate=lambda: pytest.fail('A deferred hook cannot invalidate Native poses.'))
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.usd_utils', usd)
    def forbidden(*args, **kwargs):
        pytest.fail('New independent initialization must be owned by the single later shared patch.')
    monkeypatch.setattr(interaction_setup, 'initialize_interaction', forbidden)
    monkeypatch.setattr(interaction_setup, 'initialize_interaction_trial', forbidden, raising=False)
    obj = SimpleNamespace(name='candidate', category='prop', model='model', visual_only=False, fixed_base=False)
    session = SimpleNamespace(scene_name='scene', room_name='room',
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda field, name: obj)),
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(dump_state=forbidden))))
    parameters = {'operation': action.INITIALIZATION, 'scene_model': 'scene', 'room': 'room',
        'source_provenance': {}, 'bindings': {'middle': {'name': 'candidate', 'category': 'prop', 'model': 'model'}},
        'interaction_policy': {'relocation_policy': 'independent_groups', 'placement_groups': [['middle']]}}
    result = action.initialize_scene(session, parameters)
    assert result['deferred_to_shared_initializer']
    assert result['robot_actions'] == 0 and not result['physical_grasp_verified']
    assert not hasattr(session, '_category_interaction_initialized')
    with pytest.raises(ValueError, match='runs once'):
        action.initialize_scene(session, parameters)


def test_distance_declares_separate_supports_with_common_z_and_fixed_native_anchor(tmp_path):
    converter, source = source_for(distance)
    result = converter.convert(source, ConversionContext(tmp_path))
    policy = result['interaction_policy']
    assert policy['relocation_policy'] == 'vertical_only'
    assert policy['placement_groups'] == [['selected_candidate'], ['other_candidate']]
    assert set(policy['group_bindings']) == {'selected_candidate', 'other_candidate'}
    assert policy['fixed_reference_evidence'] == distance._distance_evidence(source.payload)
    assert all('ref' not in key for key in policy['group_bindings'])
    assert result['task_definition']['private']['evaluation'] == {
        'type': 'grasp_lift', 'target': 'selected_candidate', 'minimum_lift_m': .05}


def test_inclined_pickup_policy_frees_only_target_xy(tmp_path, monkeypatch):
    deferred_start(monkeypatch, inclined)
    converter, source = source_for(inclined)
    result = converter.convert(source, ConversionContext(tmp_path))
    policy = interaction_setup.prepare_definitions(source, result['scene_definition'], result['task_definition'])
    assert policy['relocation_policy'] == 'rigid_group'
    assert policy['group_bindings'] == ['target']
    assert policy['manipulation_bindings'] == ['target']
    assert policy['support_bindings'] == ['target']
    assert policy['source_camera_policy'] == 'preserve'
    assert 'slope' not in policy['group_bindings'] and 'floor' not in policy['group_bindings']
    assert 'pickup_initial_xy_adaptation_requires_review' in result['warning_codes']
    parameters = result['task_definition']['private']['evaluation']['parameters']
    assert parameters['slope_position'] == source.payload['slope']['position']
    assert parameters['target_floor_position'] == map_records(source.payload['objects'])['task_obj']['pos_on_floor']


def test_inclined_pickup_reference_allows_new_xy_but_table_cannot_support_release(incline_trial):
    session, executor, config, bound, table = incline_trial
    target = bound['target']
    displacement = np.array([1., -.5, 0.])
    target.position += displacement
    table.position += displacement
    target.contacts[0].position += displacement
    session.category_interaction_setup['group_translation_m'] = [1., -.5, .7]
    session.category_interaction_setup['object_translations_m'][target.name] = [1., -.5, .7]
    slope_before = bound['slope'].position.copy()
    result = inclined.qualify_operation(session, executor, config, bound)
    assert result['accepted'], result
    assert result['initial_target_reference_m'] == pytest.approx(target.position)
    assert result['initial_adapted_support_contact']
    np.testing.assert_array_equal(bound['slope'].position, slope_before)
    row, _ = inclined._trial_state(target, bound['slope'], bound['floor'], config['parameters'])
    assert not row['top_supported']
    assert row['interfering_contacts'] == [['/World/table/top']]


def test_cloth_and_item_remain_one_cover_group_when_cpu_start_is_inconclusive(tmp_path, monkeypatch):
    deferred_start(monkeypatch, deformable)
    converter, source = source_for(deformable)
    result = converter.convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'generated'
    assert result['interaction_policy']['relocation_policy'] == 'rigid_group'
    assert result['interaction_policy']['group_bindings'] == ['item', 'cloth']
    assert result['interaction_policy']['manipulation_bindings'] == ['item']
    assert 'source_start_native_measurement_required' in result['warning_codes']
    patches = result['task_definition']['scene']['variants']['adapted']['patches']
    assert not any(p['operation'] == 'robot_pose' for p in patches)
    assert result['task_definition']['private']['evaluation']['operation'] == deformable.OPERATION


def test_liquid_support_request_keeps_pool_and_pair_anchors_explicit(tmp_path, monkeypatch):
    deferred_start(monkeypatch, liquid)
    converter, source = source_for(liquid)
    result = converter.convert(source, ConversionContext(tmp_path))
    policy = result['interaction_policy']
    assert policy['mode'] == 'inspection'
    assert policy['relocation_policy'] == 'vertical_only'
    assert policy['placement_groups'] == [['left'], ['right']]
    assert policy['support_bindings'] == ['left', 'right']
    assert policy['anchor_policy'] == 'liquid_volume_common_translation'
    assert result['private_scaffold_contract']['prim_paths'] == [f'/World/plane_{i}' for i in range(4)]
    assert 'source_start_native_measurement_required' in result['warning_codes']
    assert result['task_definition']['private']['evaluation']['operation'] in liquid.OPERATIONS
    assert result['task_definition']['private']['evaluation']['parameters']['accessible_support_required']


@pytest.mark.parametrize('failure', [None, 'low', 'unsupported', 'unaccepted'])
def test_liquid_current_support_height_guard_and_legacy_config(monkeypatch, failure):
    floor = SimpleNamespace(name='floor', aabb=(np.array([-2., -2., -.1]), np.array([2., 2., 0.])))
    props = {key: SimpleNamespace(name=key, supported=True,
        aabb=(np.array([index, 0., .6]), np.array([index + .2, .2, .8])))
        for index, key in enumerate(('left', 'right'))}
    if failure == 'low': props['left'].aabb = (np.array([0., 0., .1]), np.array([.2, .2, .3]))
    if failure == 'unsupported': props['left'].supported = False
    session = SimpleNamespace(scene_name='scene', room_name='room', category_interaction_setup={
        'accepted': failure != 'unaccepted', 'relocated': True, 'object_names': ['left', 'right'],
        'support_prim_path': '/World/table',
        'support_prim_paths': {'left': '/World/table', 'right': '/World/table'}})
    monkeypatch.setattr(interaction_setup, 'touching_task_support', lambda session, obj, floor: obj.supported)
    monkeypatch.setattr(interaction_setup, 'on_top_task_support', lambda session, obj, floor: obj.supported)
    checks = liquid._accessible_support_checks(session, {**props, 'floor': floor})
    assert all(checks.values()) is (failure is None)
    for operation in liquid.OPERATIONS:
        config = {'type': 'category_operation', 'module': liquid.MODULE, 'operation': operation,
            'scene': 'scene', 'room': 'room', 'bindings': {'left': 'left', 'right': 'right', 'floor': 'floor'},
            'parameters': dict(liquid.PARAMETERS)}
        assert liquid._native_config(session, config) == liquid.PARAMETERS
        config['parameters']['accessible_support_required'] = True
        assert liquid._native_config(session, config)['accessible_support_required']


def test_liquid_unequal_container_bottoms_require_separate_current_support_paths(monkeypatch):
    floor = SimpleNamespace(name='floor', aabb=(np.array([-2., -2., -.1]), np.array([2., 2., 0.])))
    # Both roots can remain at Z=.8 while their unchanged body bottoms differ.
    props = {'left': SimpleNamespace(name='left', aabb=(np.array([0., 0., .6]), np.array([.2, .2, 1.]))),
             'right': SimpleNamespace(name='right', aabb=(np.array([1., 0., .7]), np.array([1.2, .2, .9])))}
    setup = {'accepted': True, 'relocated': True, 'object_names': ['left', 'right'],
        'object_translations_m': {'left': [0., 0., .5], 'right': [0., 0., .5]},
        'support_prim_paths': {'left': '/World/left_table', 'right': '/World/right_table'},
        'support_surfaces_z_m': {'left': .6, 'right': .7}}
    session = SimpleNamespace(category_interaction_setup=setup)
    def current_support(session, obj, original_floor):
        paths = interaction_setup.support_contact_paths(session, obj.name)
        z = setup['support_surfaces_z_m'].get(obj.name)
        return bool(paths and paths == (f'/World/{obj.name}_table',) and
                    z is not None and abs(obj.aabb[0][2] - z) < .001)
    monkeypatch.setattr(interaction_setup, 'touching_task_support', current_support)
    monkeypatch.setattr(interaction_setup, 'on_top_task_support', current_support)
    bound = {**props, 'floor': floor}
    assert all(liquid._accessible_support_checks(session, bound).values())
    # A single surface at the taller vessel's bottom leaves the short one floating.
    setup['support_surfaces_z_m']['right'] = .6
    assert liquid._accessible_support_checks(session, bound) == {'left': True, 'right': False}
    setup['support_surfaces_z_m']['right'] = .7
    del setup['support_prim_paths']['right']
    assert liquid._accessible_support_checks(session, bound) == {'left': True, 'right': False}


@pytest.mark.parametrize('failure', [None, 'lost_support', 'lost_cover', 'wrong_reference', 'fixed_item'])
def test_deformable_adapted_reference_keeps_native_cover_and_support_required(monkeypatch, failure):
    constants = ModuleType('omnigibson.utils.constants')
    constants.PrimType = SimpleNamespace(CLOTH='cloth')
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.constants', constants)
    item = SimpleNamespace(name='item', visual_only=False, fixed_base=False)
    cloth = SimpleNamespace(name='cloth', visual_only=False, fixed_base=False, prim_type='cloth')
    floor = SimpleNamespace(name='floor', visual_only=False)
    session = SimpleNamespace(scene_name='scene', room_name='room', category_interaction_setup={
        'accepted': True, 'relocated': True, 'object_names': ['item', 'cloth'],
        'object_translations_m': {'item': [1., 2., .6], 'cloth': [1., 2., .6]},
        'support_prim_path': '/World/table', 'support_prim_paths': {'item': '/World/table'}})
    row = {'gripper': {'object_name': None, 'constraint_active': False}, 'floor_top_m': 0.,
        'item_position': [1., 2., .65], 'item_touching_task_support': True,
        'above_item_overlap_faces': 4, 'item_touching_cloth': True,
        'item_linear_speed_m_s': 0., 'working_hand_cloth_contacts': [],
        'other_robot_cloth_contacts': [], 'robot_in_original_room': True, 'rigid_assisted_grasp_mode': True}
    if failure == 'lost_support': row['item_touching_task_support'] = False
    if failure == 'lost_cover': row['above_item_overlap_faces'] = 0
    if failure == 'wrong_reference': row['item_position'][0] += .1
    if failure == 'fixed_item': item.fixed_base = True
    monkeypatch.setattr(deformable, '_state', lambda *args: copy.deepcopy(row))
    config = {'type': 'category_operation', 'module': deformable.MODULE, 'operation': deformable.OPERATION,
        'scene': 'scene', 'room': 'room', 'bindings': {'item': 'item', 'cloth': 'cloth', 'floor': 'floor'},
        'parameters': {'source_item_xy': [0., 0.], 'floor_height_estimate_m': 0.}}
    result = deformable.qualify_operation(session, SimpleNamespace(), config,
                                         {'item': item, 'cloth': cloth, 'floor': floor})
    assert result['accepted'] is (failure is None), result
    assert result['real_sweep_or_retrieval_verified'] is False

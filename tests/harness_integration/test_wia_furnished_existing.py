"""Check furnished support references and unchanged parent task contracts.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B
-m pytest -q -p no:cacheprovider tests/test_wia_furnished_existing.py
Synthetic fixtures exercise compatibility only; no native process is launched.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.converters import wia_furnished_layout as initializer
from wia.tasks.interaction_setup import reference_position, support_contact_paths
import wia.authoring.commands.build_wia_furnished_existing as builder


def parameters():
    return {'revision': initializer.REVISION,
        'scene_contract': {'scene_model': 'explicit_scene', 'room': 'explicit_room_0'},
        'layout': {'scene_model': 'explicit_scene', 'rooms': ['explicit_room_0'],
            'objects': [{'name': name, 'category': 'test_category', 'model': 'test_model',
                         'visual_only': False, 'fixed_base': False} for name in ('native_a', 'native_b')],
            'supported_items': [{'name': 'native_a', 'support': 'native_support_a'},
                {'name': 'native_b', 'support': 'native_support_b', 'support_surface_mode': 'local_raycast'}]},
        'source_reference_positions': {'native_a': [1., 2., 3.], 'native_b': [-1., -2., -3.]}}


def native_fixture():
    def obj(name, position, top=1.):
        root = np.asarray(position, dtype=float)
        return SimpleNamespace(name=name, prim_path='/World/' + name,
            get_position_orientation=lambda: (root, np.array([0., 0., 0., 1.])),
            aabb=(root + [-.1, -.1, -.05], root + [.1, .1, top]))
    objects = {'native_a': obj('native_a', [4., 6., 1.]), 'native_b': obj('native_b', [-5., 4., 2.]),
        'native_support_a': obj('native_support_a', [4., 6., 0.], top=.95),
        'native_support_b': obj('native_support_b', [-5., 4., 0.], top=5.)}
    scene = SimpleNamespace(scene_model='explicit_scene', object_registry=lambda field, name: objects.get(name))
    session = SimpleNamespace(env=SimpleNamespace(scene=scene), room_name='explicit_room_0',
                              _category_initialization_active=True)
    receipts = {'revision': initializer.REVISION, 'accepted': True,
        'supported_items': {'native_a': {'support': 'native_support_a', 'accepted': True, 'touching': True},
            'native_b': {'support': 'native_support_b', 'accepted': True, 'touching': True,
                         'support_surface_z_m': 1.95}},
        'robot': {'floor_supported': True, 'collision_free': True}}
    return session, receipts


def test_per_object_support_references_use_measured_settled_roots_and_local_surface():
    session, receipts = native_fixture()
    p = parameters()
    before = deepcopy(receipts)
    result = initializer.compatibility_receipt(session, p, receipts)
    assert result['accepted']
    assert result['supported_items'] == before['supported_items'] and receipts == before
    assert result['support_surfaces_z_m'] == {'native_a': .95, 'native_b': 1.95}
    assert result['object_translations_m'] == {'native_a': [3., 4., -2.], 'native_b': [-4., 6., 5.]}
    assert result['group_translation_m'] == [0., 0., 0.] and result['group_translation_is_fallback_only']
    session.category_interaction_setup = result
    assert support_contact_paths(session, 'native_a') == ('/World/native_support_a',)
    assert support_contact_paths(session, 'native_b') == ('/World/native_support_b',)
    np.testing.assert_allclose(reference_position(session, 'native_a', [1.2, 2.3, 3.4]), [4.2, 6.3, 1.4])
    np.testing.assert_allclose(reference_position(session, 'native_b', [99., 99., 99.], root=True), [-5., 4., 2.])
    np.testing.assert_allclose(reference_position(session, 'unmapped', [1., 2., 3.]), [1., 2., 3.])


def test_missing_reference_or_unavailable_live_support_never_fabricates_accepted_mapping():
    p = parameters()
    del p['source_reference_positions']['native_b']
    with pytest.raises(ValueError, match='original root reference'):
        initializer.validate_parameters(p)
    session, receipts = native_fixture()
    del receipts['supported_items']['native_b']['support_surface_z_m']
    result = initializer.compatibility_receipt(session, parameters(), receipts)
    assert not result['accepted'] and result['compatibility_measurement_errors']
    assert 'native_b' not in result['object_translations_m']
    session.category_interaction_setup = result
    assert support_contact_paths(session, 'native_a') == ()


def test_initialization_only_delegation_preserves_active_layout_gate(monkeypatch):
    session, receipts = native_fixture()
    calls = []
    monkeypatch.setattr(initializer.active_layout, 'configure_scene', lambda *args: calls.append('configure'))
    monkeypatch.setattr(initializer.active_layout, 'postprocess_joint_world', lambda *args: calls.append('postprocess'))
    monkeypatch.setattr(initializer.active_layout, 'initialize_scene', lambda *args: (calls.append('initialize') or receipts))
    initializer.configure_scene({}, parameters())
    initializer.postprocess_joint_world(session, parameters())
    session._category_initialization_active = False
    with pytest.raises(RuntimeError, match='initialization-only'):
        initializer.initialize_scene(session, parameters())
    session._category_initialization_active = True
    result = initializer.initialize_scene(session, parameters())
    assert calls == ['configure', 'postprocess', 'initialize']
    assert session.category_interaction_setup == result and result['accepted']
    receipts['accepted'] = False
    assert not initializer.initialize_scene(session, parameters())['accepted']


def parent_and_entry(module='wia.authoring.converters.action_order'):
    p = parameters()
    bindings = {name: {'name': name, 'category': 'test_category', 'model': 'test_model'}
                for name in ('native_a', 'native_b')}
    bindings['floor'] = {'name': 'original_floor', 'category': 'floors'}
    scene_config = {'policy': 'native_then_minimal_patch_else_skip',
        'variants': {'original': {'patches': []}, 'adapted': {'patches': [
            {'operation': 'object_physics', 'phase': 'before_construction', 'binding': name,
             'visual_only': False, 'fixed_base': False} for name in ('native_a', 'native_b')]}}}
    source = {'provider': 'esi', 'json': 'Synthetic/Existing/explicit_scene/explicit_room_0/sample.json'}
    task = {'kind': 'category_robot_task', 'id': 'existing_parent', 'version': 2, 'status': 'pending',
        'source': source, 'public': {'objective': 'Existing support order.', 'success_criteria': 'Existing scoring.',
                                    'environment': 'Existing environment.', 'constraints': ['Existing constraint.']},
        'requirements': {'capabilities': ['base_translation', 'gripper']},
        'private': {'bindings': bindings, 'evaluation': {'type': 'category_operation', 'module': module,
            'operation': 'existing_operation', 'binding_keys': list(bindings),
            'parameters': {'minimum_lift_m': None, 'stability_duration_s': .5, 'accepted_orders': [[1, 2, 3]]}}},
        'scene': scene_config, 'qualification': {'on_failure': 'skip'},
        'defaults': {'seed': 17, 'max_steps': 143, 'wall_timeout_s': 7200.}}
    scene = {'kind': 'scene_patch', 'id': 'existing_parent_scene', 'version': 2, 'status': 'pending',
             'source': source, 'scene': scene_config, 'bindings': bindings, 'seed': 17}
    entry = {'document_alias': 'AP-1', 'variant_id': 'furnished_fixture_v1', 'label': 'CPU test fixture',
        'source_identity': source['json'], 'layout': p['layout'],
        'source_reference_positions': p['source_reference_positions'],
        'floor_binding_updates': {'floor': 'explicit_native_floor'},
        'public': {'environment': 'Explicit native furniture environment.'}}
    return task, scene, entry


@pytest.mark.parametrize('module', sorted(builder.SUPPORTED_MODULES))
def test_parent_evaluator_goals_scoring_seed_budget_and_requirements_are_preserved(module):
    task, scene, entry = parent_and_entry(module)
    before = deepcopy((task, scene, entry))
    result, result_scene = builder.translate_definitions(task, scene, entry, {'design_file': 'synthetic'})
    assert result['private']['evaluation'] == task['private']['evaluation']
    assert result['defaults'] == task['defaults'] and result['requirements'] == task['requirements']
    assert result['public']['objective'] == task['public']['objective']
    assert result['public']['success_criteria'] == task['public']['success_criteria']
    assert result['public']['constraints'] == task['public']['constraints']
    assert result['private']['bindings']['floor']['name'] == 'explicit_native_floor'
    assert result['private']['bindings'] == result_scene['bindings']
    assert result['version'] == 3 and result_scene['version'] == 3
    patches = result['scene']['variants']['adapted']['patches']
    assert [(row['phase'], row['module']) for row in patches] == [
        (phase, module_name) for phase in ('before_construction', 'after_postprocessing')
        for module_name in (initializer.MODULE, 'wia.authoring.converters.wia_background')]
    assert all(row['parameters']['expected_scene'] == 'explicit_scene'
        and row['parameters']['mode'] == 'complete_ground' for row in patches if row['module'].endswith('wia_background'))
    assert result_scene['scene'] == result['scene']
    assert (task, scene, entry) == before


def test_builder_rejects_scoring_overrides_physics_changes_and_other_evaluator_families():
    task, scene, entry = parent_and_entry()
    entry['public']['success_criteria'] = 'Changed threshold.'
    with pytest.raises(ValueError, match='environment/objective'):
        builder.translate_definitions(task, scene, entry, {})
    del entry['public']['success_criteria']
    entry['layout']['objects'][0]['fixed_base'] = True
    with pytest.raises(ValueError, match='parent object physics'):
        builder.translate_definitions(task, scene, entry, {})
    task, scene, entry = parent_and_entry('wia.authoring.converters.rigid_partial_insertion')
    with pytest.raises(ValueError, match='Only existing'):
        builder.translate_definitions(task, scene, entry, {})

"""Check accessible perception generator rules without running simulation.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_accessible_generators_b.py tests/test_annotation_perception_categories.py
Real sources test conversion and provenance. Native-shaped fixtures test current
support/reference readbacks, not actual reach, visibility, grasp or execution.
"""
import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import enumerative_perception_category_ambiguity as ambiguity
from wia.authoring.converters import enumerative_perception_counting_w_occlusion as counting
from wia.authoring.converters import enumerative_perception_illumination_variability as illumination
from wia.authoring.converters import enumerative_perception_merged_observation as merged
from wia.authoring.converters import enumerative_perception_spatial_segmentation as segmentation
from wia.authoring.converters import enumerative_perception_structural_enclosure as enclosure
from wia.authoring.converters import spatial_relations_geometric_configuration as triangle
from wia.tasks.interaction_setup import prepare_definitions
from wia.tasks.schema import SceneDefinition
from wia.tasks.loader import task_definition
from harness_integration.test_spatial_relations_tabletop import world_factory
from harness_integration.test_annotation_perception_categories import enclosure_fixture, illumination_fixture


@pytest.mark.parametrize('module,relative', [
    (segmentation, 'Enumerative Perception/Spatial Segmentation/Ihlen_0_int/dining_room_0/q_000.json'),
    (counting, 'Enumerative Perception/Counting w Occlusion/Beechwood_0_garden/living_room_0/q_004.json'),
    (enclosure, 'Enumerative Perception/Structural Enclosure/Ihlen_0_int/dining_room_0/q_001.json'),
    (illumination, 'Enumerative Perception/Illumination Variability/grocery_store_cafe/dining_room_0/q_002.json'),
    (ambiguity, 'Enumerative Perception/Category Ambiguity/house_double_floor_upper/living_room_0/q_000.json'),
    (merged, 'Enumerative Perception/Merged Observation/Ihlen_0_int/dining_room_0/q_001.json'),
])
def test_real_source_accessible_policy_serializes_without_source_or_answer_changes(module, relative, tmp_path):
    source = SourceInstance(ROOT / 'ESI/dataset/json_clean' / relative, relative, *module.CategoryConverter.category)
    before = source.path.read_bytes()
    result = module.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'generated', result
    scene, task, policy = result['scene_definition'], result['task_definition'], result['interaction_policy']
    assert policy['source_camera_policy'] == 'preserve'
    members = [key for group in policy['placement_groups'] for key in group]
    assert len(members) == len(set(members)) and set(members) == set(policy['group_bindings'])
    assert 'floor' not in members and not any(key.startswith('anchor') for key in members)
    assert policy['allow_fallback_table'] and policy['support_search'] == 'measured_surfaces'
    assert scene['scene']['variants']['original']['patches'] == []
    if module in (counting, segmentation):
        assert policy['mode'] == 'inspection'
        assert task['requirements']['capabilities'] == ['base_translation', 'base_rotation']
        assert task['private']['evaluation']['parameters']['accessible_placement'] is True
        assert policy['support_preferences']
        for key in members:
            patches = task['scene']['variants']['adapted']['patches']
            assert any(p['operation'] == 'object_physics' and p.get('binding') == key and
                p.get('phase') == 'before_construction' and p.get('fixed_base') is False and
                p['visual_only'] is False for p in patches)
    if module is enclosure:
        p = task['private']['evaluation']['parameters']
        assert len(policy['placement_groups']) == len(p['container_keys']) == 5
        assert policy['support_bindings'] == p['container_keys']
        assert all(p['relations'][key] == group[0] for group in policy['placement_groups'] for key in group[1:])
        initialization = next(p['parameters'] for p in scene['scene']['variants']['adapted']['patches']
            if p['operation'] == 'category_initialization' and p['phase'] == 'before_construction')
        native_config = {'scene': {'scene_model': source.raw['scene'], 'load_object_categories': ['floors', 'walls']},
            'robots': [{'type': 'R1'}], 'objects': enclosure._source_helpers()['build_env_objects'](source.payload)}
        original_objects = copy.deepcopy(native_config['objects'])
        enclosure.configure_scene(native_config, initialization)
        for old, current in zip(original_objects, native_config['objects']):
            assert current['fixed_base'] is True
            assert all(current[key] == value for key, value in old.items() if key not in ('fixed_base', 'visual_only'))
            if old['name'] in {binding['name'] for binding in initialization['inspection_contents']}:
                assert current['visual_only'] is False
    if module is ambiguity:
        assert all(len(group) == 2 and group[0].startswith('target') and group[1].startswith('confuser')
                   for group in policy['placement_groups'])
    if module is illumination:
        assert policy['relocation_policy'] == 'vertical_only' and policy['anchor_policy'] == 'none'
        assert not any(key.startswith('lamp') for key in members)
        assert task['private']['evaluation']['parameters']['fixed_lighting_anchors']
        assert 'target_elevation_changes_illumination_geometry' in result['warning_codes']
    if module is merged:
        targets = task['private']['evaluation']['parameters']['target_keys']
        assert policy['relocation_policy'] == 'independent_groups'
        assert policy['placement_groups'] == [[key] for key in targets]
        assert policy['group_bindings'] == policy['manipulation_bindings'] == targets
        assert 'source_merged_occlusion_adapted' in result['warning_codes']
        patches = task['scene']['variants']['adapted']['patches']
        assert all(any(p['operation'] == 'object_physics' and p.get('binding') == key and
            p.get('phase') == 'before_construction' and p.get('fixed_base') is False and
            p['visual_only'] is False for p in patches) for key in targets)
    assert 'original_answer' not in repr(task['public'])
    for binding in task['private']['bindings'].values():
        assert binding['name'] not in repr(task['public'])
    prepare_definitions(source, scene, task, policy)
    SceneDefinition.model_validate(scene)
    task_definition(task)
    assert source.path.read_bytes() == before


def portable_fixture(world_factory, module):
    world = world_factory(triangle)
    objects, expected, pairs = {'floor': world.floor}, {}, []
    setup = world.session.category_interaction_setup
    setup.pop('support_prim_path')
    setup['support_prim_paths'] = {obj.name: [world.table.prim_path] for obj in world.props}
    for index, obj in enumerate(world.props):
        key, anchor = f'target_{index}', f'anchor_{index}'
        delta = np.array([.1 * index, -.1 * index, .6])
        setup['object_translations_m'][obj.name] = delta.tolist()
        objects[key] = obj
        chair = SimpleNamespace(name='chair_' + str(index), prim_path='/World/chair_' + str(index),
            visual_only=False, fixed_base=True, category='straight_chair',
            aabb=(obj.position - [.3, .3, .7], obj.position + [.3, .3, .5]),
            get_position_orientation=lambda obj=obj: (obj.position.copy(), np.array([0., 0., 0., 1.])))
        objects[anchor] = chair
        world.session.env.scene.objects.append(chair)
        expected[key] = {'position': (obj.position - delta).tolist(), 'quaternion_xyzw': [0., 0., 0., 1.],
            'bbox': [(corner - delta).tolist() for corner in obj.aabb]}
        expected[anchor] = {'bbox': [corner.tolist() for corner in chair.aabb]}
        pairs.append({'target': key, 'anchor': anchor})
    p = {**module.PARAMETERS, 'pairs': pairs, 'expected': expected, 'accessible_placement': True}
    config = {'type': 'category_operation', 'module': module.MODULE, 'operation': module.OPERATIONS[0],
        'scene': world.session.scene_name, 'room': world.session.room_name, 'parameters': p,
        'bindings': {key: obj.name for key, obj in objects.items()}, 'binding_keys': list(objects)}
    world.controller.capabilities = SimpleNamespace(base_translation=True, base_rotation=True)
    measured = {'poses': {key: {'position': obj.get_position_orientation()[0].tolist(),
        'quaternion_xyzw': [0., 0., 0., 1.]} for key, obj in objects.items() if key != 'floor'},
        'boxes': {key: [corner.tolist() for corner in obj.aabb] for key, obj in objects.items() if key != 'floor'},
        'room': world.session.room_name, 'floor_surface_z_m': 0., 'floor_supported': True,
        'safe': True, 'nonfloor_contacts': [], 'pairs': [{'target_side_projection_m': None} for _ in pairs]}
    return world, objects, config, measured


@pytest.mark.parametrize('module', [segmentation, counting])
def test_access_qualifiers_use_adapted_references_and_require_live_portable_support(world_factory, monkeypatch, module):
    world, objects, config, measured = portable_fixture(world_factory, module)
    monkeypatch.setattr(module, '_native_state', lambda *args: measured)
    result = module.qualify_operation(world.session, world.controller, config, objects)
    assert result['accepted'], result
    world.props[0].touching = False
    assert not module.qualify_operation(world.session, world.controller, config, objects)['accepted']
    world.props[0].touching, world.props[0].visual_only = True, True
    assert not module.qualify_operation(world.session, world.controller, config, objects)['accepted']


def test_counting_placed_on_anchor_has_measured_standoff_but_legacy_requires_original_side(world_factory, monkeypatch):
    world, objects, config, _ = portable_fixture(world_factory, counting)
    sampling = ModuleType('omnigibson.utils.sampling_utils')
    sampling.raytest = lambda start, end, **kwargs: [{'hit': True, 'rigidBody': world.floor.prim_path + '/top',
        'normal': [0., 0., 1.], 'position': [*start[:2], 0.], 'distance': .1}]
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.sampling_utils', sampling)
    robot_position = world.session.robot.get_position_orientation()[0]
    for name in world.session.robot.floor_touching_base_link_names:
        world.session.robot.links[name].aabb = (robot_position - [.05, .05, 0.], robot_position + [.05, .05, .1])
    vector_type = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    for contact in world.session.robot.contact_list():
        contact.normal = np.array(tuple(contact.normal), dtype=vector_type)
        contact.position = np.array(tuple(contact.position), dtype=vector_type)
    rows = counting._native_state(world.session, config, objects)
    assert any(row['approach_geometry'] and row['target_side_projection_m'] is None for row in rows['pairs'])
    config['parameters'].pop('accessible_placement')
    assert not any(row['approach_geometry'] for row in counting._native_state(world.session, config, objects)['pairs'])


def test_enclosure_groups_allow_separate_translations_and_contents_need_no_direct_table_path(world_factory, monkeypatch):
    world, config, objects, measured = enclosure_fixture(world_factory)
    container, content, other = world.props
    container.fixed_base = content.fixed_base = other.fixed_base = True
    sampling = ModuleType('omnigibson.utils.sampling_utils')
    rays = []

    def support_ray(start, end, **kwargs):
        rays.append((start, end))
        return [{'hit': True, 'rigidBody': world.table.prim_path + '/top', 'normal': [0., 0., 1.],
                 'position': [*start[:2], .6], 'distance': .025}]

    sampling.raytest = support_ray
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.sampling_utils', sampling)
    setup = world.session.category_interaction_setup
    setup.pop('support_prim_path')
    setup['support_prim_paths'] = {obj.name: [world.table.prim_path] for obj in (container, other)}
    deltas = {container.name: np.array([.2, .1, .6]), content.name: np.array([.2, .1, .6]),
              other.name: np.array([-.4, .3, .6])}
    setup['object_translations_m'] = {name: value.tolist() for name, value in deltas.items()}
    p = config['parameters']
    for key, obj in [('container_00', container), ('content_00', content), ('container_01', other)]:
        p['source_objects'][key] = {'position': (obj.position - deltas[obj.name]).tolist(),
            'quaternion_xyzw': [0., 0., 0., 1.], 'bbox': [(v - deltas[obj.name]).tolist() for v in obj.aabb]}
    objects['container_01'] = other
    config['bindings']['container_01'] = other.name
    config['binding_keys'].append('container_01')
    p['container_keys'].append('container_01')
    p['placement_groups'] = [['container_00', 'content_00'], ['container_01']]
    other_row = copy.deepcopy(measured[0]['container_00'])
    other_row.update(position=other.position.tolist(), bbox=[v.tolist() for v in other.aabb])
    measured[0]['container_01'] = other_row
    monkeypatch.setattr(enclosure, '_measure', lambda *args: measured)
    result = enclosure.qualify_operation(world.session, world.controller, config, objects)
    assert result['accepted'], result
    assert rays
    setup['object_translations_m'][content.name][0] += .1
    result = enclosure.qualify_operation(world.session, world.controller, config, objects)
    assert not result['accepted'] and not result['checks']['within_group_source_translations']


def test_light_cycle_checks_actual_stationary_transforms_without_changing_schedule(monkeypatch):
    matrix = np.eye(4)
    attr = SimpleNamespace(value=10.)
    attr.Get, attr.Set = lambda: attr.value, lambda value: setattr(attr, 'value', value)
    light = {'path': '/World/actual_light', 'type_name': 'SphereLight', 'baseline': 10., 'attribute': attr}
    monkeypatch.setattr(illumination, '_light_world_transform', lambda light: matrix.copy())
    cycle = illumination._OriginalLightCycle(None, {'phase_duration_s': 30., 'changed_scale': .04}, [light])
    assert cycle.measure()['original_light_transforms_preserved']
    cycle.advance(30.)
    assert attr.value == pytest.approx(.4) and cycle.measure()['measured_phase'] == 'changed'
    matrix[3, 2] = .6
    assert not cycle.measure()['original_light_transforms_preserved']


def test_illumination_accessibility_excludes_lamps_and_requires_actual_fixed_light_proof(world_factory):
    world, p, rows, lamp = illumination_fixture(world_factory)
    target = world.props[0]
    setup = world.session.category_interaction_setup
    setup['object_names'] = [target.name]
    setup['object_translations_m'] = {target.name: [0., 0., .6]}
    setup['source_reference_camera_translation_m'] = [0., 0., 0.]
    p['fixed_lighting_anchors'] = True
    p['lighting']['lamps'][0]['bbox'] = [corner.tolist() for corner in lamp.aabb]
    unchanged = {'value': True}
    world.session._illumination_variability_cycle.measure = lambda: {
        'original_light_transforms_preserved': unchanged['value']}
    assert illumination._layout_evidence(world.session, rows, p)['accepted']
    assert illumination._lamp_evidence({'lamp_00': lamp}, p, world.session)['lamp_00']['source_aabb_preserved']
    unchanged['value'] = False
    assert not illumination._layout_evidence(world.session, rows, p)['accepted']


def test_ambiguity_rejects_ambiguous_entry_provenance_without_nearest_guess():
    item = {'name': 'target', 'category': 'bottle', 'position': [0., 0., 0.]}
    confuser = {'name': 'confuser', 'category': 'mug', 'position': [1., 0., 0.]}
    source = SimpleNamespace(payload={'question_data': {'case_metadata': {'source_entries': [
        {'ball_positions': [[0., 0., 0.]], 'metadata': {'confuser_object': confuser}}]}}})
    bindings = {'target_00': {'name': 'target'}, 'confuser_00': {'name': 'confuser'}}
    assert ambiguity._dependency_groups(source, {'targets': [item], 'confusers': [confuser]}, bindings) == [
        ['target_00', 'confuser_00']]
    with pytest.raises(ValueError, match='unresolved or ambiguous'):
        ambiguity._dependency_groups(source, {'targets': [item], 'confusers': [confuser, confuser]}, bindings)


@pytest.mark.parametrize('module,relative', [
    (ambiguity, 'Enumerative Perception/Category Ambiguity/house_double_floor_upper/living_room_0/q_000.json'),
    (illumination, 'Enumerative Perception/Illumination Variability/grocery_store_cafe/dining_room_0/q_002.json'),
    (merged, 'Enumerative Perception/Merged Observation/Benevolence_1_int/living_room_0/q_001.json'),
])
def test_static_start_failure_keeps_native_attempt_without_inventing_robot_pose(module, relative, tmp_path, monkeypatch):
    source = SourceInstance(ROOT / 'ESI/dataset/json_clean' / relative, relative, *module.CategoryConverter.category)
    before = source.path.read_bytes()
    monkeypatch.setattr(module, '_robot_start', lambda *args: None)
    result = module.CategoryConverter().convert(source, ConversionContext(tmp_path))
    assert result['status'] == 'generated' and result['task_definition']['status'] == 'pending'
    assert {'source_static_start_unverified', 'native_spawn_repair_pending'} <= set(result['warning_codes'])
    for definition in (result['scene_definition'], result['task_definition']):
        assert not any(p['operation'] == 'robot_pose' for p in definition['scene']['variants']['adapted']['patches'])
    if module is illumination:
        pending = {'candidate_id': 'pending_native_spawn', 'source_identity': relative,
                   'status': 'generated', 'robot_start_evidence': None}
        assert module.CategoryConverter().sample({'records': [pending]}, {'count': 1}) == ['pending_native_spawn']
    assert source.path.read_bytes() == before


def merged_independent_world(world_factory, operation='retrieve_return'):
    world = world_factory(triangle)
    objects = {f'target_{index:02d}': obj for index, obj in enumerate(world.props)}
    source_objects = {}
    setup = world.session.category_interaction_setup
    setup.pop('support_prim_path')
    setup['support_prim_paths'] = {obj.name: [world.table.prim_path] for obj in world.props}
    for index, (key, obj) in enumerate(objects.items()):
        obj.position += [[.15, .02, 0.], [-.03, .06, 0.], [.1, -.03, 0.]][index]
        obj.contact.position = [*obj.position[:2], float(obj.position[2] - .1)]
        setup['object_translations_m'][obj.name] = (obj.position - world.source[index]).tolist()
        source_objects[key] = {'position': world.source[index].tolist(),
            'quaternion_xyzw': [0., 0., 0., 1.],
            'bbox': [(world.source[index] - .1).tolist(), (world.source[index] + .1).tolist()]}
    keys = list(objects)
    objects['floor'] = world.floor
    config = {'type': 'category_operation', 'module': merged.MODULE, 'operation': operation,
        'scene': world.session.scene_name, 'room': world.session.room_name,
        'bindings': {key: obj.name for key, obj in objects.items()}, 'binding_keys': list(objects),
        'parameters': {'target_keys': keys, 'target_category': 'bottle', 'source_objects': source_objects}}
    return world, config, objects


def test_merged_qualifier_accepts_per_target_references_but_retains_physics_and_support_guards(world_factory):
    world, config, objects = merged_independent_world(world_factory)
    result = merged.qualify_operation(world.session, world.controller, config, objects)
    assert result['accepted'], result
    assert result['checks']['source_layout_and_pairs_preserved']
    raw = config['parameters']['source_objects']
    assert not merged._layout_preserved(result['native_objects'], raw)
    moved = world.props[0]
    setup = world.session.category_interaction_setup
    setup['object_translations_m'][moved.name][0] += .1
    assert not merged.qualify_operation(world.session, world.controller, config, objects)['accepted']
    setup['object_translations_m'][moved.name][0] -= .1
    moved.touching = False
    assert not merged.qualify_operation(world.session, world.controller, config, objects)['accepted']
    moved.touching, moved.fixed_base = True, True
    assert not merged.qualify_operation(world.session, world.controller, config, objects)['accepted']


@pytest.mark.parametrize('operation', ['retrieve_return', 'inventory_return'])
def test_merged_independent_placement_keeps_real_distinct_lift_return_and_stability_predicate(
        world_factory, tmp_path, operation):
    world, config, _ = merged_independent_world(world_factory, operation)
    gripper = world.controller.debug_gripper_state()
    world.controller.debug_gripper_state = lambda: gripper
    evaluator = merged.create_evaluator(world.session, world.controller, config, tmp_path)
    observation = SimpleNamespace(snapshot_id='native_fixture', action_step=0)
    evaluator.begin_episode(observation)
    assert not evaluator.record(observation, action=None)['task_success']
    original = [obj.position.copy() for obj in world.props]
    for index, obj in enumerate(world.props):
        obj.position[2] += merged.LIFT_M + .01
        obj.touching = False
        gripper.update(object_name=obj.name, holding=True, constraint_active=True, gripper_open=False)
        evaluator._update()
        obj.position[:] = original[index]
        obj.touching = True
        gripper.update(object_name=None, holding=False, constraint_active=False, gripper_open=True)
        evaluator._update()
        for _ in range(merged.STABILITY_STEPS):
            evaluator.sample_step()
        result = evaluator.record(observation, action=None)
        assert result['task_success'] is (operation == 'retrieve_return' or index == len(world.props) - 1)
    assert result['lifted_bindings'] == result['returned_bindings'] == config['parameters']['target_keys']
    world.props[0].touching = False
    assert not evaluator.record(observation, action=None)['task_success']
    assert world.session.pipeline.og.sim.current_time == 0.

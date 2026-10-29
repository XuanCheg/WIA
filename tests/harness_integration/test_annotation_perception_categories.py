"""Check annotated perception repairs without running simulation or captures.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_annotation_perception_categories.py
Fixtures exercise actual support/reference helpers and native-shaped readbacks;
they do not establish native contact, visibility, handle reach or real grasping.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import enumerative_perception_category_ambiguity as ambiguity
from wia.authoring.converters import enumerative_perception_illumination_variability as illumination
from wia.authoring.converters import enumerative_perception_merged_observation as merged
from wia.authoring.converters import enumerative_perception_structural_enclosure as enclosure
from wia.authoring.converters import partial_occlusion as partial
from wia.authoring.converters import perceptual_grounding_view_hallucination as hallucination
from wia.authoring.converters import perceptual_grounding_material_transparency as transparency
from wia.authoring.converters import spatial_relations_geometric_configuration as triangle
from harness_integration.test_spatial_relations_tabletop import world_factory


def original_source(relative):
    path = ROOT / 'ESI/dataset/json_clean' / relative
    return SourceInstance(path, relative, *Path(relative).parts[:2])


def vertical_world(world_factory):
    world = world_factory(triangle)
    world.session.category_interaction_setup['group_translation_m'] = [0., 0., .6]
    world.session.category_interaction_setup['object_translations_m'] = {
        obj.name: [0., 0., .6] for obj in world.props}
    return world


def test_ambiguity_adapted_targets_and_confusers_require_current_upward_support(world_factory):
    world = vertical_world(world_factory)
    keys = ['target_00', 'target_01', 'confuser_00']
    world.props[2].category = 'mug'
    objects = dict(zip(keys, world.props))
    objects['floor'] = world.floor
    p = {'target_keys': keys[:2], 'confuser_keys': keys[2:], 'target_category': 'bottle',
         'source_objects': {key: {'position': (obj.position - [0., 0., .6]).tolist(),
             'quaternion_xyzw': [0., 0., 0., 1.]} for key, obj in zip(keys, world.props)}}
    config = {'type': 'category_operation', 'module': ambiguity.MODULE, 'operation': 'inventory_return',
        'scene': world.session.scene_name, 'room': world.session.room_name,
        'parameters': p, 'binding_keys': list(objects), 'bindings': {key: obj.name for key, obj in objects.items()}}
    world.session.category_ambiguity_initialization = {'accepted': True}
    result = ambiguity.qualify_operation(world.session, world.controller, config, objects)
    assert result['accepted'], result
    assert all(not row['touching_original_floor'] for row in result['native_objects'].values())
    world.props[2].on_top = False
    assert not ambiguity.qualify_operation(world.session, world.controller, config, objects)['accepted']


def test_merged_native_support_chain_accepts_table_and_preserves_stack(world_factory):
    world = vertical_world(world_factory)
    first, upper = world.props[:2]
    upper.support = first
    upper.contact.body1 = first.prim_path + '/base_link'
    objects = {'target_00': first, 'target_01': upper}
    support = merged._support(objects, world.floor, world.session)
    assert support['all_grounded']
    assert support['task_support_roots'] == ['target_00']
    assert support['target_support_edges']['target_01'] == ['target_00']
    first.touching = False
    assert not merged._support(objects, world.floor, world.session)['all_grounded']


def test_ambiguity_keeps_targets_and_confusers_in_one_authorized_group(tmp_path, monkeypatch):
    source = original_source('Enumerative Perception/Category Ambiguity/Beechwood_0_garden/living_room_0/q_000.json')
    original = source.path.read_bytes()
    monkeypatch.setattr(ambiguity, '_robot_start', lambda *args: {
        'position': [0., 0., 0.], 'yaw_deg': 0., 'source_floor_z_estimate_m': 0.})
    monkeypatch.setattr(ambiguity, '_table_plan', lambda *args: None)
    result = ambiguity.CategoryConverter().convert(source, ConversionContext(tmp_path, config={'legacy_category_ambiguity': True}))
    parameters = result['task_definition']['private']['evaluation']['parameters']
    policy = result['interaction_policy']
    assert set(policy['group_bindings']) == set(parameters['target_keys'] + parameters['confuser_keys'])
    assert set(policy['manipulation_bindings']) == set(parameters['target_keys'])
    assert policy['allow_fallback_table'] is True
    for value in result['task_definition']['private']['bindings'].values():
        assert value['name'] not in repr(result['task_definition']['public'])
    assert source.path.read_bytes() == original


def illumination_fixture(world_factory):
    world = vertical_world(world_factory)
    target, lamp = world.props[:2]
    setup = world.session.category_interaction_setup
    setup.update(object_names=[target.name, lamp.name], anchor_translation_m=[0., 0., .6],
        transformed_bound_anchor_names=[lamp.name],
        transformed_light_prim_paths=['/World/dynamic_light'],
        skipped_descendant_light_paths=[lamp.prim_path + '/light'],
        source_reference_camera_translation_m=[0., 0., .6])
    world.session._illumination_variability_cycle = SimpleNamespace(lights=[
        {'path': '/World/dynamic_light'}, {'path': lamp.prim_path + '/light'}])
    p = {'source_objects': {'target_00': {'position': (target.position - [0., 0., .6]).tolist(),
        'quaternion_xyzw': [0., 0., 0., 1.], 'bbox': [(value - [0., 0., .6]).tolist() for value in target.aabb]}},
        'lamp_keys': ['lamp_00'], 'lighting': {'lamps': [{'name': lamp.name,
            'bbox': [(value - [0., 0., .6]).tolist() for value in lamp.aabb]}]}}
    rows = {'target_00': illumination._snapshot(world.session, target, world.floor)}
    return world, p, rows, lamp


def test_illumination_requires_complete_common_vertical_light_and_observer_proof(world_factory):
    world, p, rows, lamp = illumination_fixture(world_factory)
    assert illumination._layout_evidence(world.session, rows, p)['accepted']
    measured = illumination._lamp_evidence({'lamp_00': lamp}, p, world.session)
    assert measured['lamp_00']['source_aabb_preserved']
    assert not illumination._lamp_evidence({'lamp_00': lamp}, p)['lamp_00']['source_aabb_preserved']


@pytest.mark.parametrize('missing', ['anchor_translation_m', 'transformed_bound_anchor_names',
    'transformed_light_prim_paths', 'skipped_descendant_light_paths', 'source_reference_camera_translation_m'])
def test_incomplete_lighting_anchor_evidence_rejects_adapted_layout(world_factory, missing):
    world, p, rows, _ = illumination_fixture(world_factory)
    del world.session.category_interaction_setup[missing]
    assert not illumination._layout_evidence(world.session, rows, p)['accepted']


def enclosure_fixture(world_factory):
    world = vertical_world(world_factory)
    container, content = world.props[:2]
    objects = {'container_00': container, 'content_00': content, 'floor': world.floor}
    original = {}
    for key, obj in objects.items():
        if key != 'floor':
            original[key] = {'position': (obj.position - [0., 0., .6]).tolist(),
                'quaternion_xyzw': [0., 0., 0., 1.],
                'bbox': [(value - [0., 0., .6]).tolist() for value in obj.aabb]}
    config = {'type': 'category_operation', 'module': enclosure.MODULE, 'operation': enclosure.OPERATIONS[0],
        'scene': world.session.scene_name, 'room': world.session.room_name,
        'binding_keys': list(objects), 'bindings': {key: obj.name for key, obj in objects.items()},
        'parameters': {'container_keys': ['container_00'], 'content_keys': ['content_00'],
            'relations': {'content_00': 'container_00'}, 'source_objects': original}}
    joint = {'visible': True, 'driven': False, 'collision_mesh_count': 1, 'contact_reporting': True,
        'velocity': 0., 'lower_limit': 0., 'upper_limit': 1.5, 'position': 0., 'open_direction': 1,
        'minimum_travel': .5, 'door_link_bbox': [[0., 0., .6], [.1, .1, .8]],
        'handle_geometry': [{'mesh': '/World/container/door/handle', 'bbox': [[0., 0., .65], [.1, .1, .75]]}]}
    row = {'position': container.position.tolist(), 'quaternion_xyzw': [0., 0., 0., 1.],
        'bbox': [v.tolist() for v in container.aabb], 'fixed_base': True, 'visual_only': False,
        'root_collision_mesh_count': 1, 'open': False, 'both_sides': False, 'joints': [joint]}
    content_row = {'position': content.position.tolist(), 'quaternion_xyzw': [0., 0., 0., 1.],
        'bbox': [v.tolist() for v in content.aabb], 'inside_original_container': True}
    measured = ({'container_00': row}, {'content_00': content_row}, world.controller.debug_gripper_state(),
                {'in_original_room': True, 'supported_without_obstacle_contact': True})
    return world, config, objects, measured


def test_enclosure_adapted_references_and_actual_handle_heights(world_factory, monkeypatch):
    world, config, objects, measured = enclosure_fixture(world_factory)
    monkeypatch.setattr(enclosure, '_measure', lambda *args: measured)
    good = enclosure.qualify_operation(world.session, world.controller, config, objects)
    assert good['accepted'], good
    measured[0]['container_00']['joints'][0]['handle_geometry'][0]['bbox'] = [[0., 0., .1], [.1, .1, .2]]
    low = enclosure.qualify_operation(world.session, world.controller, config, objects)
    assert not low['accepted']
    assert not low['checks']['known_handle_height_window']
    assert low['checks']['source_layout_preserved']


def test_missing_handle_name_is_unresolved_not_a_root_height_certificate(world_factory, monkeypatch):
    world, config, objects, measured = enclosure_fixture(world_factory)
    measured[0]['container_00']['joints'][0]['handle_geometry'] = []
    monkeypatch.setattr(enclosure, '_measure', lambda *args: measured)
    result = enclosure.qualify_operation(world.session, world.controller, config, objects)
    assert result['accepted'], result
    assert 'handle_geometry_unresolved' in result['warning_codes']
    row = result['native_interaction_heights']['containers']['container_00']
    assert not row['handle_geometry_resolved']
    assert not row['door_link_proxies'][0]['reach_or_grasp_certified']


def test_contained_target_height_is_measured_separately(world_factory):
    _, config, objects, measured = enclosure_fixture(world_factory)
    measured[1]['content_00']['bbox'] = [[0., 0., .1], [.1, .1, .2]]
    evidence = enclosure._interaction_heights(config, objects, measured)
    assert not evidence['contained_targets']['content_00']['within_height_window']
    assert evidence['containers']['container_00']['actual_handles'][0]['within_height_window']


def test_partial_explicit_legacy_retrieves_original_target_without_forced_survey_or_blocker_motion(tmp_path):
    source = original_source(partial.FEEDBACK_SOURCE)
    original = source.path.read_bytes()
    result = partial.PartialOcclusionConverter().convert(source, ConversionContext(tmp_path, config={'legacy_partial_occlusion': True}))
    task = result['task_definition']
    assert result['operation_recipe'] == 'retrieve'
    assert task['private']['evaluation'] == {'type': 'grasp_lift', 'target': 'target', 'minimum_lift_m': .05}
    assert task['private']['bindings']['target']['name'] == partial._source_builder().build_env_objects(source.payload)[0]['name']
    assert not any(word in task['public']['objective'].lower() for word in ('survey', 'return', 'lift the foreground'))
    assert result['interaction_policy']['group_bindings'] == ['target', 'occluder']
    assert result['interaction_policy']['allow_fallback_table'] is True
    assert source.raw['question'] in ' '.join(task['public']['constraints'])
    assert source.path.read_bytes() == original


def test_large_partial_target_retains_retrieval_with_feasibility_diagnostic():
    bounds = {'aabb_min': [0., 0., 0.], 'aabb_max': [1., 1., 2.]}
    operation, evidence = partial._operation({'target': bounds, 'occluder': bounds})
    assert operation == 'retrieve'
    assert evidence['legacy_dimension_selected_recipe'] == 'inspection_route'
    assert not evidence['native_grasp_certificate']


def test_view_hallucination_preserves_original_identity_and_flags_impossible_global_window(tmp_path):
    source = original_source('Perceptual Grounding/View Hallucination/Beechwood_0_garden/'
        'dining_room_0/Beechwood_0_garden__dining_room_0_0.json')
    original = source.path.read_bytes()
    result = hallucination.CategoryConverter().convert(source, ConversionContext(tmp_path, config={'legacy_view_hallucination': True}))
    assert not result['source_layout_evidence']['whole_aabb_height_window_translation_feasible']
    assert 'whole_aabb_height_window_requires_local_grasp_measurement' in result['warning_codes']
    assert result['task_definition']['private']['evaluation']['type'] == 'grasp_lift'
    assert result['scene_definition']['scene']['variants']['original']['patches'] == []
    assert result['source_reference']['selection'] == 'original_first_visible_non_topdown_else_first_non_topdown'
    assert source.path.read_bytes() == original


def test_transparency_preserves_existing_real_extraction_goal_and_original_source(tmp_path):
    source = original_source('Perceptual Grounding/Material Transparency/Beechwood_0_garden/'
        'dining_room_0/Beechwood_0_garden__dining_room_0.json')
    original = source.path.read_bytes()
    result = transparency.CategoryConverter().convert(source, ConversionContext(tmp_path, config={'legacy_transparency_extraction': True}))
    assert result['status'] == 'generated', result
    assert result['task_definition']['private']['evaluation'] == {
        'type': 'extract_lift', 'target': 'obj_small', 'container': 'obj_container', 'minimum_lift_m': .05}
    assert source.raw['question'] in result['task_definition']['public']['objective']
    assert result['scene_definition']['scene']['variants']['original']['patches'] == []
    assert source.path.read_bytes() == original

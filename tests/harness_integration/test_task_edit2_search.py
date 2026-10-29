"""Check TaskEdit2 joint inventories and factual visual search accounting.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_search.py
All native receipts below are controlled fixtures, never GPU qualification.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from wia.authoring.conversion import ConversionContext, ROOT, SourceInstance
from wia.authoring.converters import enumerative_perception_search as search
from wia.authoring.converters.composition_perception import build_task_edit2_search_group, plan_groups
from wia.authoring.converters.enumerative_perception_structural_enclosure import (
    TASK_EDIT2_ADDITIONAL_SOURCES, curate_task_edit2_additions)
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition


def observation(snapshot='view_1'):
    return NS(snapshot_id=snapshot, action_step=1, rgb_images=lambda: {
        camera: [] for camera in ('head', 'left_wrist', 'right_wrist', 'rear', 'left', 'right', 'front', 'down')})


def replay(tmp_path):
    targets = [{'id': f't{i}', 'binding': f'b{i}', 'group_id': 'group_1',
        'expected_identity': 'club_sandwich', 'room': 'the kitchen', 'area': 'the table area'} for i in range(2)]
    config = {'operation': search.OPERATION, 'bindings': {'b0': 'native_zero', 'b1': 'native_one'},
        'parameters': {'targets': targets, 'groups': [{'id': 'group_1', 'target_count': 2}]}}
    evidence = {'available': True, 'gates': dict.fromkeys(search.NATIVE_GATES, True),
        'mechanisms': {'group_1': True}, 'observations': []}
    for target, native, region in zip(targets, config['bindings'].values(), ([.1, .1, .3, .3], [.6, .6, .8, .8])):
        for camera in observation().rgb_images():
            evidence['observations'].append({'native_name': native, 'snapshot_id': 'view_1',
                'camera': camera, 'identity_synchronized': True, 'native_rgb_matches_delivered': True,
                'measurement_available': True, 'informative': True,
                'informative_feature_evidence': {'visible_diagnostic_features': ['sandwich filling and bread']},
                'location_verified': True, 'room': target['room'], 'area': target['area'], 'image_region': region})
    session = NS(enumerative_search_evidence=lambda *args: deepcopy(evidence))
    evaluator = search.SearchEvaluator(session, None, config, tmp_path)
    evaluator.begin_episode(observation())
    return evaluator, evidence


def declaration(index=0):
    return {'group_id': 'group_1', 'identity': 'club sandwich', 'room': 'the kitchen',
        'area': 'the table area', 'snapshot_id': 'view_1', 'camera': 'head',
        'image_point': [.2, .2] if index == 0 else [.7, .7]}


def test_true_informative_observation_and_correct_identity_location_are_joint_requirements(tmp_path):
    evaluator, _ = replay(tmp_path)
    assert evaluator.last['observed_distinct_targets'] == 2
    assert evaluator.last['verified_distinct_targets'] == 0
    result = evaluator.report_answer({'declarations': [declaration(0), declaration(1)]}, observation())
    assert result['task_success']
    assert result['discovery_percent'] == 100
    assert result['declaration_precision_percent'] == 100


def test_all_camera_repeats_deduplicate_and_duplicate_declarations_do_not_inflate_recall(tmp_path):
    evaluator, _ = replay(tmp_path)
    result = evaluator.report_answer({'declarations': [declaration(0)] * 8}, observation())
    assert result['observed_distinct_targets'] == 2
    assert result['verified_distinct_targets'] == 1
    assert result['discovery_percent'] == 50
    assert result['declaration_precision_percent'] == 100
    assert not result['task_success']


def test_omitted_false_wrong_location_and_wrong_identity_declarations(tmp_path):
    evaluator, _ = replay(tmp_path)
    result = evaluator.report_answer({'declarations': [declaration()]}, observation())
    assert result['discovery_percent'] == 50
    for field, value in [('identity', 'milk carton'), ('room', 'the bedroom'), ('image_point', [.95, .95])]:
        wrong = dict(declaration(1), **{field: value})
        result = evaluator.report_answer({'declarations': [declaration(), declaration(1), wrong]}, observation())
        assert result['discovery_percent'] == 100
        assert result['declaration_precision_percent'] == pytest.approx(200 / 3)
        assert result['false_positive_count'] == 1
        assert not result['task_success']


@pytest.mark.parametrize('field,value', [('informative', False), ('informative_feature_evidence', {}),
    ('identity_synchronized', False), ('native_rgb_matches_delivered', False),
    ('location_verified', False), ('snapshot_id', 'stale'), ('camera', 'human_global')])
def test_proximity_ray_and_segmentation_fragments_alone_do_not_establish_discovery(tmp_path, field, value):
    evaluator, evidence = replay(tmp_path)
    evaluator.receipts = {}
    for row in evidence['observations']:
        row[field] = value
        row.update(ray_hit=True, near_target=True, segmentation_pixels=10000)
    result = evaluator.report_answer({'declarations': [declaration(), declaration(1)]}, observation())
    assert result['observed_distinct_targets'] == 0
    assert result['verified_distinct_targets'] == 0
    assert not result['task_success']


def test_missing_adapter_measurement_or_mechanism_never_becomes_valid_from_source_metadata(tmp_path):
    evaluator, evidence = replay(tmp_path)
    evidence['mechanisms'] = {}
    evidence['source_annotations'] = {'mechanism_verified': True}
    result = evaluator.report_answer({'declarations': [declaration(), declaration(1)]}, observation())
    assert result['diagnostic_discovery_percent'] == 100
    assert result['discovery_percent'] is None
    assert result['evaluation_validity'] == 'unavailable'
    assert not result['task_success']
    evaluator.session = NS()
    result = evaluator.record(observation())
    assert result['evaluation_validity'] == 'unavailable'
    assert result['discovery_percent'] is None
    assert 'adapter unavailable' in result['native_evidence']['measurement_error']


def test_invalid_final_submission_and_empty_submission_cannot_complete(tmp_path):
    evaluator, _ = replay(tmp_path)
    result = evaluator.report_answer('not JSON', observation())
    assert result['submission_invalid'] and not result['task_success']
    assert result['false_positive_count'] == 1
    result = evaluator.report_answer({'declarations': []}, observation())
    assert result['discovery_percent'] == 0
    assert not result['task_success']


def candidates(tmp_path):
    from wia.runtime.commands.build_task_edit2_search_candidates import CURATED
    results = []
    for name, rows in CURATED:
        sources, public = [], {}
        for suffix, room, area, clue in rows:
            identity = 'Enumerative Perception/' + suffix
            sources.append(SourceInstance(ROOT / 'ESI/dataset/json_clean' / identity,
                identity, 'Enumerative Perception', suffix.split('/')[0]))
            public[identity] = {'room': room, 'area': area, 'clue': clue}
        results.append(build_task_edit2_search_group(sources, ConversionContext(tmp_path), public_areas=public, name=name))
    return results


def test_real_curated_sources_generate_joint_worlds_covering_all_five_mechanisms(tmp_path):
    parents = candidates(tmp_path)
    assert {g['mechanism'] for p in parents for g in p['task_definition']['private']['evaluation']['parameters']['groups']} == set(search.MECHANISMS)
    for parent in parents:
        task, scene = parent['task_definition'], parent['scene_definition']
        task_definition(task)
        SceneDefinition.model_validate(scene)
        assert parent['execution_status'] == 'unavailable'
        assert len(parent['world_contract']['required_rooms']) >= 2
        inventory = parent['world_contract']['joint_inventory']
        assert len({x['native']['name'] for x in inventory}) == len(inventory)
        assert all(x['native']['position'] == x['source_specification']['position']
                   for x in inventory if 'initialization_translation_m' not in x)
        assert all(x['source_identity'] and x['source_object_name'] for x in inventory)
        assert task['requirements']['capabilities'] == ['base_translation', 'base_rotation']
        assert task['private']['evaluation']['parameters']['physical_collection_required'] is False
        config = {'scene': {'scene_model': parent['world_contract']['scene_model'],
            'load_room_instances': ['old_room'], 'load_object_categories': ['floors']}, 'objects': [{'name': 'old'}]}
        search.configure_scene(config, {'world_contract': parent['world_contract']})
        assert 'load_room_instances' not in config['scene']
        assert config['scene']['load_object_categories'] is None
        from wia.authoring.converters.task_edit2_search_layout import primitive_configs
        assert len(config['objects']) == len(inventory) + len(primitive_configs(parent['world_contract']))
        assert not parent['world_contract']['native_compatibility_verified']
    assert any(x['role'] == 'confusers' for x in parents[0]['world_contract']['joint_inventory'])


def test_three_enclosure_additions_have_exact_source_assets_and_distinct_occupancy(tmp_path):
    additions = curate_task_edit2_additions(ConversionContext(tmp_path))
    assert len(additions) == len(TASK_EDIT2_ADDITIONAL_SOURCES) == 3
    assert len({x['candidate_id'] for x in additions}) == 3
    assert len({len(x['curation_provenance']['source_content_membership']) for x in additions}) == 3
    for addition in additions:
        assert addition['execution_status'] == 'unavailable'
        assert set(addition['native_gates'].values()) == {'pending'}
        assert addition['task_definition'] is not None
        task_definition(addition['task_definition'])
        provenance = addition['curation_provenance']
        source = SourceInstance(ROOT / 'ESI/dataset/json_clean' / provenance['source_identity'],
            provenance['source_identity'], 'Enumerative Perception', 'Structural Enclosure')
        original = source.payload['question_data']['render']['resolved_objects']
        assert provenance['source_objects'] == {x['name']: x for role in ('containers', 'targets') for x in original[role]}


def test_legacy_planner_is_not_silently_routed_into_versioned_search():
    record = {'candidate_id': 'historical', 'source_identity': 'historical',
              'small_task': 'Merged Observation', 'status': 'generated'}
    group, = plan_groups([record], {}, {})
    assert group['member_ids'] == ['historical']
    assert 'original independent goal' in group['label']


def test_measured_constraint_violation_preserves_recall_but_blocks_strict_success(tmp_path):
    evaluator, evidence = replay(tmp_path)
    evidence['gates']['constraints_preserved'] = False
    result = evaluator.report_answer({'declarations': [declaration(), declaration(1)]}, observation())
    assert result['evaluation_validity'] == 'complete'
    assert result['discovery_percent'] == 100
    assert result['constraint_compliance'] is False
    assert not result['task_success']


def test_curated_enclosure_pixel_gate_cannot_substitute_for_informative_native_receipt():
    from wia.authoring.converters.enumerative_perception_structural_enclosure import task_edit2_enclosure_evidence
    config = {'bindings': {'content_0': 'actual_native_target'}, 'parameters': {'content_keys': ['content_0']}}
    old = {'objects': {'content_0': {'visible': True}}, 'complete_coverage': True}
    evidence, gates = task_edit2_enclosure_evidence(NS(), observation(), config, old)
    assert not evidence['objects']['content_0']['visible']
    assert not gates
    native = {'available': True, 'objects': {'content_0': {'informative': True,
        'identity_synchronized': True, 'native_rgb_matches_delivered': True,
        'snapshot_id': 'view_1', 'camera': 'head', 'native_name': 'actual_native_target',
        'informative_feature_evidence': {'feature': 'visible native content'}}},
        'gates': {'actual_containment': True}}
    evidence, gates = task_edit2_enclosure_evidence(NS(enumerative_enclosure_evidence=lambda *args: native),
                                                  observation(), config, old)
    assert evidence['objects']['content_0']['visible']
    assert gates == {'actual_containment': True}

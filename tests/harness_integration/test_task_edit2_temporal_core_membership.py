"""Validate exact Core temporal generation without launching native simulation.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_temporal_core_membership.py
"""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
import numpy as np

from wia.tasks.loader import task_definition, load_task, load_scene
from wia.tasks.schema import SceneDefinition
import wia.runtime.commands.build_task_edit2_temporal_candidates as builder


@pytest.fixture(scope='module')
def core_build(tmp_path_factory):
    directory = tmp_path_factory.mktemp('temporal_core')
    originals = json.loads(builder.CORE_MANIFEST.read_text())['tasks']
    original_temporal = [row for row in originals if row['source'].startswith('Temporal Understanding/')]
    original_hashes = {row['source']: builder.sha256(builder.source(row['source'], row['source'].split('/')[1]).path)
                       for row in original_temporal}
    prototype_manifest = builder.ROOT / 'outputs/task_edit2_20261018/temporal/manifest.json'
    prototype_hash = builder.sha256(prototype_manifest)
    historical_roots = [builder.ROOT / 'outputs/task_edit2_20261018' / name
        for name in ('temporal', 'temporal_core', 'temporal_core_v2')]
    historical_hashes = {path: builder.sha256(path) for root in historical_roots
        for path in root.rglob('*') if path.is_file()}
    result = builder.build_core(directory)
    assert builder.sha256(prototype_manifest) == prototype_hash
    assert historical_hashes == {path: builder.sha256(path) for root in historical_roots
        for path in root.rglob('*') if path.is_file()}
    assert original_hashes == {row['source']: builder.sha256(builder.source(row['source'], row['source'].split('/')[1]).path)
                               for row in original_temporal}
    return result, directory, original_temporal


def test_core_membership_and_concrete_source_mappings(core_build):
    manifest, directory, originals = core_build
    assert manifest['counts'] == {'core_temporal_sources': 8, 'generated_candidates': 8,
        'pending_history_inventory_redesign': 0, 'pending_source_conversion': 0}
    assert len(manifest['source_mapping']) == len({row['source_identity'] for row in manifest['source_mapping']}) == 8
    assert {row['source_identity'] for row in manifest['source_mapping']} == {row['source'] for row in originals}
    historical = {row['source']: row for row in originals}
    for mapping in manifest['source_mapping']:
        original = historical[mapping['source_identity']]
        assert mapping['source_core_candidate_id'] == original['candidate_id']
        assert mapping['source_sha256'] == builder.sha256(mapping['source_file'])
        for kind in ('task', 'scene'):
            assert mapping['historical_' + kind + '_file'] == original[kind + '_file']
            assert mapping['historical_' + kind + '_sha256'] == builder.sha256(original[kind + '_file'])
    assert json.loads((directory / 'manifest.json').read_text()) == manifest
    assert json.loads((directory / 'source_mapping.json').read_text()) == manifest['source_mapping']
    assert manifest['final_selection_owner'] == 'user'
    assert manifest['native_experiments_run'] is False
    assert manifest['human_review_status'] == 'requested_before_any_further_native_or_model_agent_execution'
    assert not manifest['pending_records']


def test_each_core_change_keeps_source_branch_inventory_and_native_gates(core_build):
    manifest, _, _ = core_build
    changes = [row for row in manifest['candidates'] if row['operation_recipe'] == builder.change.PHYSICAL_OPERATION]
    assert {row['source_identity'] for row in changes} == set(builder.CORE_CHANGE_BRANCHES)
    assert sorted(row['physical_branch'] for row in changes) == ['added', 'added', 'no_change', 'removed']
    assert set(manifest['supported_source_branches']) == {'added', 'removed', 'no_change'}
    assert manifest['not_present_in_core_source_branches'] == ['moved', 'replaced']
    for record in changes:
        source = builder.source(record['source_identity'], 'Unobserved Change')
        layout = builder.source_change._layout(source)
        recipe = builder.source_change.source_change_recipe(layout)
        task = yaml.safe_load(Path(record['task_file']).read_text())
        scene = yaml.safe_load(Path(record['scene_file']).read_text())
        task_definition(task)
        SceneDefinition.model_validate(scene)
        loaded_task = load_task(record['task_file'], variant='adapted', seed=7, run_mode='diagnostic')
        loaded_scene = load_scene(record['scene_file'], variant='adapted', seed=7, run_mode='diagnostic')
        assert loaded_task.source_path == loaded_scene.source_path == source.path
        assert loaded_task.seed == loaded_scene.seed == 7
        parameters = task['private']['evaluation']['parameters']
        assert task['source']['json'] == scene['source']['json'] == record['source_identity']
        assert parameters['source_sha256'] == record['source_sha256'] == builder.sha256(source.path)
        assert record['task_sha256'] == builder.sha256(record['task_file'])
        assert record['scene_sha256'] == builder.sha256(record['scene_file'])
        assert parameters['physical_branch'] == recipe['branch'] == builder.CORE_CHANGE_BRANCHES[source.relative_path]
        assert parameters['source_event_recipe'] == recipe
        assert set(layout['bindings']) <= set(task['private']['bindings'])
        assert record['source_inventory']['containers'] == {key: layout['bindings'][key] for key in layout['container_keys']}
        for phase, keys in layout['phase_keys'].items():
            assert set(record['source_inventory']['phase_contents'][phase]) == set(keys)
        assert parameters['requires_native_visual_discovery'] is True
        assert scene['runtime']['requires_native_visual_discovery'] is True
        assert parameters['phase_2_navigation_required'] is False
        assert 'phase_2_minimum_base_travel_m' not in parameters
        assert 'phase_2_container_visit_radius_m' not in parameters
        assert record['execution_status'] == 'unavailable'
        assert all(value == 'pending' for value in record['native_gates'].values())
        assert record['operation_recipe'] == builder.change.PHYSICAL_OPERATION
        if recipe['branch'] == 'no_change':
            assert parameters['physical_target_key'] is parameters['physical_destination_key'] is None
        elif recipe['branch'] == 'added':
            assert parameters['physical_target_key'] == recipe['current_key']
            assert parameters['physical_destination_key'] == 'change_collection_tray'
        else:
            selected = task['private']['bindings'][parameters['physical_target_key']]
            original = layout['bindings'][recipe['original_key']]
            assert all(selected[key] == original[key] for key in ('category', 'model'))
            assert parameters['physical_destination_key'] == recipe['source_container']


def test_final_definitions_have_one_matching_source_container_group_initializer(core_build):
    manifest, _, _ = core_build
    for record in manifest['candidates']:
        if record['operation_recipe'] != builder.change.PHYSICAL_OPERATION:
            continue
        src = builder.source(record['source_identity'], 'Unobserved Change')
        layout = builder.source_change._layout(src)
        expected_groups = [[container, *[key for key in layout['phase_keys']['image1']
            if layout['relations'][key] == container]] for container in layout['container_keys']]
        task = yaml.safe_load(Path(record['task_file']).read_text())
        scene = yaml.safe_load(Path(record['scene_file']).read_text())
        matching = []
        for definition in (task, scene):
            patches = definition['scene']['variants']['adapted']['patches']
            shared = [patch for patch in patches if patch['operation'] == 'prepare_category_interaction']
            assert len(shared) == 1
            policy = shared[0]
            assert policy['phase'] == 'after_postprocessing'
            assert policy['mode'] == 'inspection' and policy['relocation_policy'] == 'independent_groups'
            assert policy['placement_groups'] == expected_groups
            assert policy['support_bindings'] == policy['manipulation_bindings'] == layout['container_keys']
            assert policy['source_camera_policy'] == 'preserve'
            assert policy['support_search'] == 'measured_surfaces' and policy['allow_fallback_table']
            for key, value in record['interaction_policy'].items():
                assert policy[key] == value
            callbacks = [patch for patch in patches if patch['operation'] == 'category_initialization'
                and patch.get('module') == builder.change.MODULE]
            assert callbacks and all(patch['parameters']['accessible_support_rebase_required'] is True for patch in callbacks)
            assert max(patches.index(patch) for patch in callbacks) < patches.index(policy)
            matching.append(policy)
        assert matching[0] == matching[1]
        assert record['warning_codes'] == sorted(set(record['warning_codes']))
        assert 'interaction_layout_native_review_pending' in record['warning_codes']
        assert set(record['interaction_policy']['warning_codes']) <= set(record['warning_codes'])


def test_core_v3_physical_changes_are_exactly_unchanged_from_v2(core_build):
    manifest, _, _ = core_build
    previous = json.loads((builder.ROOT / 'outputs/task_edit2_20261018/temporal_core_v2/manifest.json').read_text())
    old_rows = {row['source_identity']: row for row in previous['candidates']}
    assert manifest['seed'] == previous['seed'] == 7
    for record in manifest['candidates']:
        if record['operation_recipe'] != builder.change.PHYSICAL_OPERATION:
            continue
        old = old_rows[record['source_identity']]
        assert record['source_inventory'] == old['source_inventory']
        assert record['physical_branch'] == old['physical_branch']
        assert record['candidate_id'] == old['candidate_id']
        for kind in ('task', 'scene'):
            new = yaml.safe_load(Path(record[kind + '_file']).read_text())
            prior = yaml.safe_load(Path(old[kind + '_file']).read_text())
            assert new == prior  # Entire definitions, including prepared initialization and IDs.


def test_removed_source_spares_are_actual_source_inventory_not_invented_distractors(core_build):
    manifest, _, _ = core_build
    record = next(row for row in manifest['candidates'] if row.get('physical_branch') == 'removed')
    task = yaml.safe_load(Path(record['task_file']).read_text())
    parameters = task['private']['evaluation']['parameters']
    spares = parameters['public_spare_inventory']
    assert record['public_spare_count'] == len(spares) == 1
    assert record['spare_distractor_status'] == 'unavailable_only_one_distinct_source_identity'
    assert 'awaits an explicit selection' in record['spare_inventory_limitation']
    actors = parameters['apparatus']['dataset_objects']
    assert {row['name'] for row in actors} == {row['binding']['name'] for row in spares}
    assert all(not row['fixed_base'] and not row['visual_only'] for row in actors)
    assert any(row['name'] == 'change_spare_table' for row in parameters['apparatus']['primitives'])
    assert spares[0]['public_category'].replace('_', ' ') in task['public']['objective']


def test_approved_core_histories_preserve_original_and_add_exact_two_measured_targets(core_build):
    manifest, directory, originals = core_build
    expected = {row['source'] for row in originals if '/Agent Observation/' in row['source']}
    histories = [row for row in manifest['candidates'] if row['operation_recipe'] == builder.history.HISTORY_OPERATION]
    assert len(histories) == 4 and {row['source_identity'] for row in histories} == expected
    for record in histories:
        src = builder.source(record['source_identity'], 'Agent Observation')
        _, _, originals = builder.history._source_data(src)
        task = yaml.safe_load(Path(record['task_file']).read_text())
        scene = yaml.safe_load(Path(record['scene_file']).read_text())
        p = task['private']['evaluation']['parameters']
        task_definition(task); SceneDefinition.model_validate(scene)
        assert load_task(record['task_file'], variant='adapted', seed=7, run_mode='diagnostic').source_path == src.path
        assert load_scene(record['scene_file'], variant='adapted', seed=7, run_mode='diagnostic').source_path == src.path
        assert p['history_inventory_revision'] == builder.CORE_HISTORY_REVISION
        assert p['original_source_objects'] == originals and p['original_source_target_count'] == 1
        assert p['original_target_keys'] == [originals[0]['_key']]
        assert p['added_target_keys'] == ['history_added_jam', 'history_added_box']
        assert len(p['target_keys']) == len(set(p['target_keys'])) == 3
        assert record['source_inventory']['original_targets'] == originals
        assert set(p['apparatus']['source_target_positions']) == set(p['target_keys'])
        assert set(p['apparatus']['target_support_bindings']) == set(p['target_keys'])
        assert len(set(p['public_target_labels'].values())) == 3
        assert 'Survey all 3 distinct targets' in task['public']['objective']
        assert 'first-observed order' in task['public']['objective']
        assert 'storage room' not in p['public_route']
        assert src.raw['room'].replace('_', ' ') in p['public_route']
        assert p['historical_count_answer_is_provenance_only'] == src.raw['answer']
        assert src.raw['question'] not in task['public']['objective']
        dataset = {row['name']: row for row in p['apparatus']['dataset_objects']}
        for key, category, model, _, _ in builder.CORE_HISTORY_ASSETS:
            obj = p['source_objects'][key]
            assert obj['category'] == category and obj['model'] == model
            assert dataset[key]['scale'] == obj['scale'] == [1., 1., 1.]
            assert dataset[key]['fixed_base'] is dataset[key]['visual_only'] is False
            assert dataset[key]['position'] == obj['position'] == p['apparatus']['source_target_positions'][key]
            assert dataset[key]['orientation'] == obj['quaternion_xyzw']
            donor = obj['asset_provenance']['donor_geometry']
            np.testing.assert_allclose(np.asarray(obj['aabb_max']) - obj['aabb_min'],
                np.asarray(donor['aabb_max_m']) - donor['aabb_min_m'], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.asarray(obj['position']) - obj['aabb_min'],
                np.asarray(donor['position_m']) - donor['aabb_min_m'], atol=1e-8, rtol=0)
            assert obj['quaternion_xyzw'] == donor['quaternion_xyzw']
        policies = [[patch for patch in definition['scene']['variants']['adapted']['patches']
            if patch['operation'] == 'prepare_category_interaction'] for definition in (task, scene)]
        assert len(policies[0]) == len(policies[1]) == 1 and policies[0] == policies[1]
        assert policies[0][0]['placement_groups'] == [[key] for key in p['target_keys']]
        assert policies[0][0]['support_preferences'] == p['apparatus']['target_support_bindings']
        assert policies[0][0]['allow_fallback_table'] is False
        assert record['execution_status'] == 'unavailable'
        assert all(value == 'pending' for value in record['native_gates'].values())
        assert record['human_review_status'] == 'pending_no_native_or_model_agent_execution'
    assert len(list((directory / 'candidates').glob('*/task.yaml'))) == 8


def test_default_prototype_build_remains_five_non_core_examples(tmp_path):
    manifest = builder.build(tmp_path / 'prototypes')
    assert len(manifest['candidates']) == 5
    assert {row['source_identity'] for row in manifest['candidates']} == {builder.HISTORY_SOURCE, *builder.CHANGE_SOURCES.values()}
    assert not set(builder.CORE_CHANGE_BRANCHES) & {row['source_identity'] for row in manifest['candidates']}
    assert 'pending_records' not in manifest
    for record in manifest['candidates']:
        task = yaml.safe_load(Path(record['task_file']).read_text())
        scene = yaml.safe_load(Path(record['scene_file']).read_text())
        patches = [[patch for patch in definition['scene']['variants']['adapted']['patches']
            if patch['operation'] == 'prepare_category_interaction'] for definition in (task, scene)]
        if record['source_identity'] == builder.HISTORY_SOURCE:
            assert patches == [[], []]
            assert record['interaction_policy_application']['applied'] is False
            assert record['interaction_policy_application']['native_action_qualification'] == 'unavailable'
            assert 'legacy_history_generic_interaction_policy_unapplied' in record['warning_codes']
        else:
            assert len(patches[0]) == len(patches[1]) == 1
            assert patches[0] == patches[1]
            assert 'interaction_layout_native_review_pending' in record['warning_codes']


def test_core_mode_rejects_prototype_output_and_does_not_invent_new_branch(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='must not overwrite'):
        builder.build_core(builder.ROOT / 'outputs/task_edit2_20261018/temporal')
    with pytest.raises(ValueError, match='must not overwrite'):
        builder.build_core(builder.ROOT / 'outputs/task_edit2_20261018/temporal_core')
    with pytest.raises(ValueError, match='must not overwrite'):
        builder.build_core(builder.ROOT / 'outputs/task_edit2_20261018/temporal_core_v2')
    assert builder.CORE_OUTPUT == builder.ROOT / 'outputs/task_edit2_20261018/temporal_core_v3'
    originals = json.loads(builder.CORE_MANIFEST.read_text())['tasks']
    original = next(row for row in originals if row['source'] in builder.CORE_CHANGE_BRANCHES)
    custom = tmp_path / 'one_core.json'
    custom.write_text(json.dumps({'tasks': [original]}))
    monkeypatch.setitem(builder.CORE_CHANGE_BRANCHES, original['source'], 'moved')
    manifest = builder.build_core(tmp_path / 'changed_source', core_manifest=custom)
    assert manifest['candidates'] == []
    assert len(manifest['pending_records']) == 1
    record = manifest['pending_records'][0]
    assert record['status'] == 'pending_source_conversion'
    assert 'source event differs' in record['reason']
    assert 'task_file' not in record and 'scene_file' not in record


def test_signature_uses_final_prepared_definitions_and_exact_result_source(tmp_path, monkeypatch):
    original = next(row for row in json.loads(builder.CORE_MANIFEST.read_text())['tasks']
        if row['source'] in builder.CORE_CHANGE_BRANCHES)
    selected = tmp_path / 'one_source.json'
    selected.write_text(json.dumps({'tasks': [original]}))
    before = builder.build_core(tmp_path / 'before', core_manifest=selected)
    prepare = builder.prepare_definitions
    def annotated(src, scene, task, explicit):
        result = prepare(src, scene, task, explicit)
        for definition in (task, scene):
            patch = next(patch for patch in definition['scene']['variants']['adapted']['patches']
                if patch['operation'] == 'prepare_category_interaction')
            patch['warning_codes'].append('cpu_signature_fixture')
        return result
    monkeypatch.setattr(builder, 'prepare_definitions', annotated)
    after = builder.build_core(tmp_path / 'after', core_manifest=selected)
    assert before['candidates'][0]['candidate_id'] != after['candidates'][0]['candidate_id']
    # A converter's mismatched explicit source cannot be written using a source
    # guessed from the other loop iteration or a historical candidate.
    convert = builder.change.CategoryConverter.convert
    def wrong_source(self, src, context):
        result = deepcopy(convert(self, src, context))
        result['source_identity'] = builder.CHANGE_SOURCES['added']
        return result
    monkeypatch.setattr(builder.change.CategoryConverter, 'convert', wrong_source)
    failed = builder.build_core(tmp_path / 'wrong_source', core_manifest=selected)
    assert not failed['candidates'] and failed['counts']['pending_source_conversion'] == 1
    assert 'exact explicit source mapping' in failed['pending_records'][0]['reason']
    assert not list((tmp_path / 'wrong_source').rglob('task.yaml'))

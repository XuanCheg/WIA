"""Validate rigid temporal transfers against actual matched-history parents.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_temporal_scene_variants.py
All generated task, scene and source outputs stay in temporary directories.
No native simulation, GPU, capture queue or replay generation is invoked.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from wia.tasks import sources
from wia.tasks.loader import load_scene, load_task
import wia.authoring.commands.build_wia_temporal_scene_variants as builder


PARENT = builder.ROOT / 'explore_to_achieve/wia_v1/temporal_start_v3'
DELTA = [-7., -3., .25]
SUPPORT = {'type': 'DatasetObject', 'name': 'breakfast_table_rhjoby_0',
           'category': 'breakfast_table', 'model': 'rhjoby', 'position': [.3, .5, .6],
           'orientation': [0., 0., 0., 1.], 'fixed_base': True, 'visual_only': False}


@pytest.fixture
def design(tmp_path, monkeypatch):
    # Existing project-owned parent sources remain immutable readable inputs;
    # the new authored root is temporary for every test including CLI writes.
    monkeypatch.setattr(sources, 'UPSTREAM_ROOT', sources.AUTHORED_ROOT)
    monkeypatch.setattr(sources, 'AUTHORED_ROOT', tmp_path / 'authored_sources')
    entries = []
    for history in 'abc':
        entries.append({'document_alias': 'TM-2', 'variant_id': 'transfer_' + history,
            'label': 'CPU matched-history rigid transfer ' + history,
            'parent_task_file': str(PARENT / 'tasks' / f'history_{history}_v3.yaml'),
            'parent_scene_file': str(PARENT / 'scenes' / f'history_{history}_v3.yaml'),
            'output_task_file': str(tmp_path / 'tasks' / f'history_{history}.yaml'),
            'output_scene_file': str(tmp_path / 'scenes' / f'history_{history}.yaml'),
            'source_identity': f'Temporal Understanding/Unobserved Change/TargetScene/target_room_0/transfer_{history}.json',
            'scene_model': 'TargetScene', 'room': 'target_room_0', 'floor_name': 'target_native_floor',
            'translation_m': DELTA, 'background_support_objects': [deepcopy(SUPPORT)]})
    document = {'schema': 'wia_temporal_transfer_design_v1', 'entries': entries}
    path = tmp_path / 'design.json'
    path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')
    return path, tmp_path / 'manifest.json', document


def rewrite(design):
    design[0].write_text(json.dumps(design[2], indent=2) + '\n', encoding='utf-8')


def hooks(task):
    return [patch for patch in task['scene']['variants']['adapted']['patches']
            if patch['operation'] == 'category_initialization']


def robot_pose(task):
    return next(patch for patch in task['scene']['variants']['adapted']['patches']
                if patch['operation'] == 'robot_pose')


def test_actual_parent_preserves_history_public_thresholds_and_rigid_geometry(design):
    definitions, manifest, rendered = builder.build_package(*design[:2])
    assert manifest['schema'] == 'wia_variant_manifest_v1'
    assert len(definitions) == len(manifest['entries']) == 3 and len(rendered) == 10
    for definition, entry, record in zip(definitions, design[2]['entries'], manifest['entries']):
        task, scene = definition['task'], definition['scene']
        parent = yaml.safe_load(Path(entry['parent_task_file']).read_text())
        parent_scene = yaml.safe_load(Path(entry['parent_scene_file']).read_text())
        old_eval, new_eval = (obj['private']['evaluation'] for obj in (parent, task))
        old_p, new_p = old_eval['parameters'], new_eval['parameters']
        assert task['id'] == parent['id'] and task['version'] == parent['version'] + 1
        assert scene['id'].endswith('__' + entry['variant_id'])
        assert task['public'] == parent['public']
        assert task['defaults'] == parent['defaults']
        assert task['requirements'] == parent['requirements']
        assert task['qualification'] == parent['qualification']
        assert task['scene'] == scene['scene']
        assert task['private']['bindings'] == scene['bindings']
        assert task['source'] == scene['source'] == {**parent['source'], 'json': entry['source_identity']}
        assert task['repair']['runtime'] == scene['runtime']
        assert scene['runtime']['requires_native_visual_discovery'] is True
        assert scene['runtime']['camera_size'] == [512, 512]
        assert scene['runtime']['post_controller_settle_steps'] == 0
        for variant in ('original', 'saved_image1', 'saved_image2'):
            if variant in parent['scene']['variants']:
                assert task['scene']['variants'][variant] == parent['scene']['variants'][variant]
        assert old_eval['module'] == new_eval['module'] == builder.MODULE
        assert old_eval['operation'] == new_eval['operation'] == builder.OPERATION
        for key in old_p.keys() - {'source_objects', 'identities', 'apparatus'}:
            assert new_p[key] == old_p[key], key
        assert new_p['public_spare_inventory'] == old_p['public_spare_inventory']
        assert task['private']['bindings']['floor']['name'] == entry['floor_name']
        assert new_p['identities']['floor']['name'] == entry['floor_name']
        assert task['private']['bindings'][SUPPORT['name']] == {
            key: SUPPORT[key] for key in ('name', 'category', 'model')}
        assert SUPPORT['name'] in new_eval['binding_keys'] and SUPPORT['name'] in new_p['identities']
        assert new_p['apparatus']['dataset_objects'][-1] == SUPPORT
        assert new_p['apparatus']['floor_contact_names'] == old_p['apparatus']['floor_contact_names']
        for key, before in old_p['source_objects'].items():
            after = new_p['source_objects'][key]
            np.testing.assert_allclose(after['position'], np.asarray(before['position']) + DELTA, atol=1e-12)
            assert after['quaternion_xyzw'] == before['quaternion_xyzw']
            if 'bbox_size_m' in before:
                assert after['bbox_size_m'] == before['bbox_size_m']
            if 'bbox' in before:
                np.testing.assert_allclose(after['bbox'], np.asarray(before['bbox']) + DELTA, atol=1e-12)
        for before, after in zip(old_p['apparatus']['primitives'], new_p['apparatus']['primitives']):
            assert after == {**before, 'position': [v + d for v, d in zip(before['position'], DELTA)]}
        for before, after in zip(old_p['apparatus']['dataset_objects'], new_p['apparatus']['dataset_objects']):
            assert after == {**before, 'position': [v + d for v, d in zip(before['position'], DELTA)]}
        before_robot, after_robot = robot_pose(parent), robot_pose(task)
        assert after_robot == {**before_robot, 'position': [v + d for v, d in zip(before_robot['position'], DELTA)]}
        for hook in hooks(task):
            p = hook['parameters']
            assert p['scene_model'] == entry['scene_model'] and p['room'] == entry['room']
            assert p['source_identity'] == entry['source_identity']
            assert p['bindings'] == task['private']['bindings']
            assert p['apparatus'] == new_p['apparatus']
            assert p['source_event_recipe'] == old_p['source_event_recipe']
            assert p['replay_payload']['scene'] == entry['scene_model']
        interaction = task['scene']['variants']['adapted']['patches'][-1]
        assert interaction['source_region'] == entry['room']
        assert record['variant'] == 'adapted'
        assert record['parent_task_sha256'] == builder.sha256(Path(entry['parent_task_file']).read_bytes())
        assert record['source_sha256'] == builder.sha256(rendered[definition['source_file']])
        for key in ('task_file', 'scene_file', 'source_file', 'parent_source_file', 'parent_task_file', 'parent_scene_file'):
            assert Path(record[key]).is_absolute()
        assert all(record[key] is task['repair'][key] is False for key in builder.NATIVE_FLAGS)
        assert parent_scene['runtime'] == {'requires_native_visual_discovery': True}


def test_source_metadata_and_camera_offsets_translate_without_regeneration(design):
    definitions, _, _ = builder.build_package(*design[:2])
    for definition, entry in zip(definitions, design[2]['entries']):
        old_task = yaml.safe_load(Path(entry['parent_task_file']).read_text())
        before = json.loads(sources.resolve_source(old_task['source']['json']).read_bytes())
        after = definition['source']
        old_m, new_m = before['metadata_json'], after['metadata_json']
        assert after['id'] == new_m['question_id'] == Path(entry['source_identity']).stem
        assert new_m['source_identity'] == entry['source_identity']
        for container in (after, new_m):
            assert container['scene'] == entry['scene_model'] and container['room'] == entry['room']
            assert container['floor_name'] == entry['floor_name']
        assert new_m['authored_revision'] == old_m['authored_revision']
        assert after['question'] == before['question'] and after['answer'] == before['answer']
        assert after['options_json'] == before['options_json']
        old_render, new_render = (m['question_data']['render'] for m in (old_m, new_m))
        old_gt, new_gt = (r['gt_view']['image1']['images'][0] for r in (old_render, new_render))
        np.testing.assert_allclose(np.asarray(new_gt['camera_pose']['position']) - new_gt['look_target'],
                                   np.asarray(old_gt['camera_pose']['position']) - old_gt['look_target'], atol=1e-12)
        assert old_gt['camera_pose']['quaternion_xyzw'] == new_gt['camera_pose']['quaternion_xyzw']
        np.testing.assert_allclose(new_render['room_bbox']['bbox_world_xy'],
                                   np.asarray(old_render['room_bbox']['bbox_world_xy']) + [-7., -3., -7., -3.])
        assert new_render['room_bbox']['bbox_map_rc'] == old_render['room_bbox']['bbox_map_rc']
        assert new_render['room_bbox']['room_instance'] == entry['room']
        assert after['transfer_provenance']['parent_source_sha256'] == builder.sha256(
            sources.resolve_source(old_task['source']['json']).read_bytes())


def test_matched_abc_public_and_final_layout_remain_equal_after_transfer(design):
    definitions, _, _ = builder.build_package(*design[:2])
    payloads, finals, recipes, initial_identities = [], [], [], []
    for definition in definitions:
        task = definition['task']
        p = task['private']['evaluation']['parameters']
        payloads.append({'id': task['id'], 'version': task['version'], 'public': task['public']})
        keys = [*p['container_keys'], *p['phase_keys']['image2']]
        finals.append({'source_objects': {key: p['source_objects'][key] for key in keys},
                       'apparatus': p['apparatus'], 'robot_pose': robot_pose(task)})
        recipes.append(p['source_event_recipe'])
        initial_identities.append({key: p['identities'][key] for key in p['phase_keys']['image1']})
    assert payloads[0] == payloads[1] == payloads[2]
    assert finals[0] == finals[1] == finals[2]
    assert recipes[0] == recipes[1] and recipes[2]['branch'] == 'no_change'
    assert initial_identities[0] != initial_identities[1]


def test_cli_determinism_loaders_and_refusal_preserve_all_inputs(design):
    inputs = [design[0], *(Path(entry[key]) for entry in design[2]['entries']
                          for key in ('parent_task_file', 'parent_scene_file'))]
    inputs += [sources.resolve_source(yaml.safe_load(Path(entry['parent_task_file']).read_text())['source']['json'])
               for entry in design[2]['entries']]
    before = {path: path.read_bytes() for path in inputs}
    args = ['--design-file', str(design[0]), '--manifest-output', str(design[1])]
    first = builder.build_package(*design[:2])
    assert builder.build_package(*design[:2])[2] == first[2]
    assert builder.main(args + ['--check']) == 1
    assert not sources.AUTHORED_ROOT.exists()
    assert builder.main(args) == 0
    times = {path: path.stat().st_mtime_ns for path in first[2]}
    assert builder.main(args + ['--check']) == builder.main(args) == 0
    assert times == {path: path.stat().st_mtime_ns for path in first[2]}
    for row in first[0]:
        task = load_task(row['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(row['scene_file'], variant='adapted', run_mode='diagnostic')
        assert task.scene_name == scene.scene_name == 'TargetScene'
        assert task.room_name == scene.room_name == 'target_room_0'
        assert task.source_path == scene.source_path == row['source_file']
    changed = first[0][0]['source_file']
    changed.write_bytes(changed.read_bytes() + b'\n')
    altered = changed.read_bytes()
    assert builder.main(args + ['--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(args)
    assert changed.read_bytes() == altered
    assert {path: path.read_bytes() for path in inputs} == before


def test_only_reviewed_world_fields_translate_and_offstage_sentinels_stay_parked():
    value = {'position': [1., 2., 3.], 'look_target': [4., 5., 6.], 'target': [7., 8., 9.],
             'aabb_min': [0., 0., 0.], 'aabb_max': [1., 1., 1.],
             'bbox': [[1., 2., 3.], [4., 5., 6.]],
             'room_bbox_world_xy': [1., 2., 4., 5.], 'expanded_bbox_world_xy': [0., 0., 6., 7.],
             'bbox_size_m': [1., 2., 3.], 'dimensions_m': [4., 5., 6.],
             'quaternion_xyzw': [0., 0., 0., 1.], 'scale': [2., 3., 4.],
             'rgba': [.1, .2, .3, 1.], 'indices': [0, 1, 2],
             'parked': {'position': [1000., 1000., 1000.], 'bbox': [[999., 999., 999.], [1001., 1001., 1001.]]}}
    old = deepcopy(value)
    new = builder.translate_world(value, DELTA)
    assert value == old
    assert new['position'] == [-6., -1., 3.25]
    assert new['aabb_min'] == DELTA
    assert new['room_bbox_world_xy'] == [-6., -1., -3., 2.]
    for key in ('bbox_size_m', 'dimensions_m', 'quaternion_xyzw', 'scale', 'rgba', 'indices', 'parked'):
        assert new[key] == old[key]


def test_json_string_metadata_preserves_encoding_convention(design):
    entry = design[2]['entries'][0]
    parent = yaml.safe_load(Path(entry['parent_task_file']).read_text())
    raw = json.loads(sources.resolve_source(parent['source']['json']).read_bytes())
    raw['metadata_json'] = json.dumps(raw['metadata_json'])
    translated = builder.translate_source(raw, entry, {'parent_source_identity': parent['source']['json']})
    assert isinstance(translated['metadata_json'], str)
    assert json.loads(translated['metadata_json'])['source_identity'] == entry['source_identity']
    assert json.loads(raw['metadata_json'])['floor_name'] != entry['floor_name']


@pytest.mark.parametrize('change', ['duplicate_support', 'existing_support', 'wrong_source_scene', 'wrong_runtime'])
def test_rejects_ambiguous_or_conflicting_contracts_without_outputs(design, tmp_path, change):
    entry = design[2]['entries'][0]
    if change == 'duplicate_support':
        entry['background_support_objects'].append(deepcopy(SUPPORT))
    elif change == 'existing_support':
        entry['background_support_objects'][0]['name'] = 'public_spare_1'
    elif change == 'wrong_source_scene':
        entry['source_identity'] = entry['source_identity'].replace('TargetScene', 'AnotherScene')
    elif change == 'wrong_runtime':
        parent = yaml.safe_load(Path(entry['parent_task_file']).read_text())
        parent['repair']['runtime'] = {'camera_size': [1024, 1024]}
        path = tmp_path / 'altered_parent.yaml'
        path.write_text(yaml.safe_dump(parent))
        entry['parent_task_file'] = str(path)
    rewrite(design)
    with pytest.raises(ValueError):
        builder.build_package(*design[:2])
    assert not sources.AUTHORED_ROOT.exists()

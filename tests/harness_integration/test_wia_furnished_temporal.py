"""Check rigid furnished temporal copies using CPU-only real parent definitions.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_furnished_temporal.py
No native simulation, capture, evaluation or robot operation is executed.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from wia.tasks import sources
import wia.authoring.commands.build_wia_furnished_temporal as builder

PARENT = builder.ROOT / 'explore_to_achieve/wia_v1/temporal_start_v3'
TRANSFORM = {'origin_m': [7., 3., 0.], 'translation_m': [-4., 8., .2], 'yaw_deg': 90.}
SUPPORT_MAP = {'breakfast_table_rhjoby_0': 'native_existing_dining_table'}


@pytest.fixture
def design(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, 'UPSTREAM_ROOT', sources.AUTHORED_ROOT)
    monkeypatch.setattr(sources, 'AUTHORED_ROOT', tmp_path / 'authored_sources')
    entries = []
    for history in 'abc':
        entries.append({'document_alias': 'T40', 'variant_id': 'furnished_' + history,
            'label': 'CPU furnished matched history ' + history,
            'parent_task_file': str(PARENT / 'tasks' / f'history_{history}_v3.yaml'),
            'parent_scene_file': str(PARENT / 'scenes' / f'history_{history}_v3.yaml'),
            'output_task_file': str(tmp_path / 'tasks' / (history + '.yaml')),
            'output_scene_file': str(tmp_path / 'scenes' / (history + '.yaml')),
            'source_identity': f'Temporal Understanding/Unobserved Change/TargetScene/target_room_0/furnished_{history}.json',
            'scene_model': 'TargetScene', 'room': 'target_room_0', 'floor_name': 'native_floor',
            'transform': deepcopy(TRANSFORM), 'support_name_map': deepcopy(SUPPORT_MAP), 'matched_group': 'matched_abc',
            'environment': 'A furnished room with the existing native dining table.'})
    document = {'schema': 'wia_furnished_temporal_design_v1', 'entries': entries}
    path = tmp_path / 'design.json'
    path.write_text(json.dumps(document, indent=2) + '\n')
    return path, tmp_path / 'manifest.json', document


def test_cardinal_rigid_transform_rotates_points_quaternions_vectors_and_all_box_corners():
    transform = builder.RigidTransform(TRANSFORM)
    before = {'position': [8., 5., .7], 'quaternion_xyzw': [0., 0., 0., 1.], 'direction': [1., 2., 3.],
        'bbox': [[6., 2., .3], [8., 5., .9]], 'aabb_min': [6., 2., .3], 'aabb_max': [8., 5., .9],
        'bbox_world_xy': [6., 2., 8., 5.], 'dimensions_m': [.3, .5, .7], 'scale': [2., 1., .5],
        'source_objects': {'target': {'bbox_size_m': [.3, .5, .7]}}}
    old = deepcopy(before)
    after = builder.transform_world(before, transform)
    assert before == old
    assert after['position'] == pytest.approx([1., 12., .9])
    assert after['direction'] == pytest.approx([-2., 1., 3.])
    assert Rotation.from_quat(after['quaternion_xyzw']).apply([1., 0., 0.]) == pytest.approx([0., 1., 0.])
    corners = np.array([transform.point(list(point)) for point in product(*zip(*before['bbox']))])
    assert np.asarray(after['bbox']) == pytest.approx(np.array([corners.min(axis=0), corners.max(axis=0)]))
    assert after['aabb_min'] == after['bbox'][0] and after['aabb_max'] == after['bbox'][1]
    assert after['bbox_world_xy'] == [*corners.min(axis=0)[:2], *corners.max(axis=0)[:2]]
    assert after['dimensions_m'] == before['dimensions_m'] and after['scale'] == before['scale']
    assert after['source_objects']['target']['bbox_size_m'] == [.5, .3, .7]


def test_parked_actors_local_sensor_offsets_and_historical_provenance_stay_exact():
    value = {'parked': {'position': [1002.5, 1000., 120.], 'quaternion_xyzw': [0., 0., 0., 1.]},
        'bbox': [[999., 999., 119.], [1001., 1001., 121.]],
        'provenance': {'position': [1., 2., 3.], 'scene': 'HistoricalScene'},
        'authored_revision': {'position': [4., 5., 6.]},
        'sensor': {'frame': 'parent', 'position': [.1, 0., .2], 'orientation': [0., 0., 0., 1.]},
        'source_target_positions': {'target_native': [8., 5., .7]}}
    result = builder.transform_world(value, TRANSFORM)
    for key in ('parked', 'bbox', 'provenance', 'authored_revision', 'sensor'):
        assert result[key] == value[key]
    assert result['source_target_positions']['target_native'] == pytest.approx([1., 12., .9])
    with pytest.raises(ValueError, match='Unsupported ambiguous numeric vector'):
        builder.transform_world({'unknown_world_offset': [1., 2., 3.]}, TRANSFORM)


def test_real_parent_worlds_saved_phases_and_pose_restoration_remain_coherent(design):
    definitions, manifest, rendered = builder.build_package(*design[:2])
    assert manifest['schema'] == 'wia_variant_manifest_v1' and len(definitions) == 3 and len(rendered) == 10
    final_states, recipes = [], []
    transform = builder.RigidTransform(TRANSFORM)
    for row, entry in zip(definitions, design[2]['entries']):
        task, scene = row['task'], row['scene']
        parent = yaml.safe_load(Path(entry['parent_task_file']).read_text())
        p, old = task['private']['evaluation']['parameters'], parent['private']['evaluation']['parameters']
        assert task['scene'] == scene['scene'] and task['private']['bindings'] == scene['bindings']
        assert task['scene']['variants']['original'] == {'patches': []}
        assert set(task['scene']['variants']) == {'original', 'adapted', 'saved_image1', 'saved_image2'}
        assert task['defaults'] == parent['defaults'] and task['requirements'] == parent['requirements']
        assert task['public'] == {**parent['public'], 'environment': entry['environment']}
        assert p['source_event_recipe'] == old['source_event_recipe']
        assert p['apparatus']['floor_contact_names'] == [SUPPORT_MAP.get(name, name) for name in old['apparatus']['floor_contact_names']]
        assert p['apparatus']['dataset_objects'] == builder.transform_world(old['apparatus']['dataset_objects'], TRANSFORM)
        for key, before in old['source_objects'].items():
            assert p['source_objects'][key]['position'] == pytest.approx(transform.point(before['position']))
            if 'bbox_size_m' in before:
                assert p['source_objects'][key]['bbox_size_m'] == pytest.approx(np.abs(transform.matrix) @ before['bbox_size_m'])
        for variant in task['scene']['variants'].values():
            patches = variant['patches']
            if not patches:
                continue
            hooks = [patch for patch in patches if patch['operation'] == 'category_initialization']
            assert hooks[0]['parameters'] == hooks[1]['parameters']
            params = hooks[0]['parameters']
            assert params['scene_model'] == entry['scene_model'] and params['room'] == entry['room']
            assert params['source_identity'] == entry['source_identity']
            assert params['bindings'] == task['private']['bindings']
            initializer_index = next(index for index, patch in enumerate(patches)
                if patch['operation'] == 'category_initialization' and patch['phase'] == 'after_postprocessing')
            poses = [patch for patch in patches[:initializer_index]
                     if patch['operation'] == 'object_pose' and patch['phase'] == 'after_postprocessing']
            keys = [*params['container_keys'], *params['phase_keys'][params['phase']]]
            for key in keys:
                pose = next(patch for patch in poses if patch['binding'] == key)
                assert pose['position'] == params['source_objects'][key]['position']
                assert pose['quaternion_xyzw'] == params['source_objects'][key]['quaternion_xyzw']
            for item in params.get('apparatus', {}).get('primitives', []):
                assert item['dimensions_m'] == next(before for before in old['apparatus']['primitives'] if before['name'] == item['name'])['dimensions_m']
                pose = next(patch for patch in poses if task['private']['bindings'][patch['binding']]['name'] == item['name'])
                assert pose['quaternion_xyzw'] == item['quaternion_xyzw']
        keys = [*p['container_keys'], *p['phase_keys']['image2']]
        final_states.append({'source_objects': {key: p['source_objects'][key] for key in keys}, 'apparatus': p['apparatus']})
        recipes.append(p['source_event_recipe'])
        assert task['repair']['parent_source_sha256'] == builder.sha256(Path(task['repair']['parent_source_file']).read_bytes())
        assert all(task['repair'][key] is False for key in builder.NATIVE_FLAGS)
    assert final_states[0] == final_states[1] == final_states[2]
    assert recipes[0] == recipes[1] and recipes[2]['branch'] == 'no_change'


def test_source_camera_rotation_and_immutable_history_provenance(design):
    definitions, _, _ = builder.build_package(*design[:2])
    transform = builder.RigidTransform(TRANSFORM)
    for row in definitions:
        provenance = row['source']['transfer_provenance']
        old = json.loads(Path(provenance['parent_source_file']).read_bytes())
        new = row['source']
        assert new['question'] == old['question'] and new['answer'] == old['answer']
        assert new['metadata_json']['authored_revision'] == old['metadata_json']['authored_revision']
        old_gt = old['metadata_json']['question_data']['render']['gt_view']['image1']['images'][0]
        new_gt = new['metadata_json']['question_data']['render']['gt_view']['image1']['images'][0]
        assert new_gt['camera_pose']['position'] == pytest.approx(transform.point(old_gt['camera_pose']['position']))
        assert new_gt['look_target'] == pytest.approx(transform.point(old_gt['look_target']))
        before_local = Rotation.from_quat(old_gt['camera_pose']['quaternion_xyzw']).inv().apply(
            np.asarray(old_gt['look_target']) - old_gt['camera_pose']['position'])
        after_local = Rotation.from_quat(new_gt['camera_pose']['quaternion_xyzw']).inv().apply(
            np.asarray(new_gt['look_target']) - new_gt['camera_pose']['position'])
        assert after_local == pytest.approx(before_local)
        assert all(row['task']['repair'][key] == value for key, value in provenance.items())
        assert provenance['parent_source_sha256'] == builder.sha256(Path(provenance['parent_source_file']).read_bytes())


def test_cli_determinism_authored_sources_and_mismatch_refusal(design):
    args = ['--design-file', str(design[0]), '--manifest-output', str(design[1])]
    first = builder.build_package(*design[:2])
    assert builder.build_package(*design[:2])[2] == first[2]
    assert builder.main(args + ['--check']) == 1
    assert builder.main(args) == builder.main(args + ['--check']) == 0
    path = first[0][0]['task_file']
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        builder.main(args)


def test_matched_groups_reject_different_robot_override_and_keep_explicit_final_pose(design):
    robot = {'position': [1., 2., .08], 'orientation': [0., 0., 0., 1.]}
    for entry in design[2]['entries']:
        entry['robot_start'] = deepcopy(robot)
    design[0].write_text(json.dumps(design[2]))
    definitions, _, _ = builder.build_package(*design[:2])
    for row in definitions:
        for variant in row['task']['scene']['variants'].values():
            if not variant['patches']:
                continue
            patch = next(patch for patch in variant['patches'] if patch['operation'] == 'robot_pose')
            assert patch['position'] == robot['position'] and patch['yaw_deg'] == 0.
    design[2]['entries'][1]['robot_start']['position'][0] += 1.
    design[0].write_text(json.dumps(design[2]))
    with pytest.raises(ValueError, match='Matched-group'):
        builder.build_package(*design[:2])


@pytest.mark.parametrize('delta_z', [-.2, .2])
def test_explicit_floor_grounding_changes_only_designated_leg_height_and_preserves_top(delta_z):
    source = {'apparatus': {'primitives': [
        {'name': 'change_collection_table_leg_0', 'position': [1., 2., .35], 'dimensions_m': [.04, .04, .7]},
        {'name': 'change_spare_table_leg_3', 'position': [3., 4., .35], 'dimensions_m': [.04, .04, .7]},
        {'name': 'change_collection_table', 'position': [1., 2., .725], 'dimensions_m': [.6, .5, .05]}]},
        'source_objects': {'content': {'position': [1., 2., .9], 'bbox_size_m': [.1, .2, .3]}},
        'parent_provenance': {'apparatus': {'primitives': [
            {'name': 'change_collection_table_leg_0', 'position': [1., 2., .35], 'dimensions_m': [.04, .04, .7]}]}}}
    transform = {'origin_m': [0., 0., 0.], 'translation_m': [2., 3., delta_z], 'yaw_deg': 90.}
    moved = builder.transform_world(builder._prepare_primitives(source), transform)
    original = deepcopy(moved)
    adjusted = builder.ground_apparatus_legs(moved, 0.)
    assert moved == original
    for before, after in zip(moved['apparatus']['primitives'][:2], adjusted['apparatus']['primitives'][:2]):
        top = before['position'][2] + before['dimensions_m'][2] / 2
        assert after['position'][:2] == before['position'][:2]
        assert after['position'][2] - after['dimensions_m'][2] / 2 == pytest.approx(0.)
        assert after['position'][2] + after['dimensions_m'][2] / 2 == pytest.approx(top)
        assert after['dimensions_m'][:2] == before['dimensions_m'][:2]
        assert after['quaternion_xyzw'] == before['quaternion_xyzw']
    assert adjusted['apparatus']['primitives'][2] == moved['apparatus']['primitives'][2]
    assert adjusted['source_objects'] == moved['source_objects']
    assert adjusted['parent_provenance'] == source['parent_provenance']
    with pytest.raises(ValueError, match='nonpositive leg height'):
        builder.ground_apparatus_legs(moved, 10.)


def test_optional_native_furniture_reset_hook_precedes_restoration_and_temporal_install(design):
    entry = design[2]['entries'][0]
    parent_task, parent_scene = (yaml.safe_load(Path(entry['parent_' + kind + '_file']).read_text()) for kind in ('task', 'scene'))
    runtime = builder.load_scene(entry['parent_scene_file'], variant='adapted', run_mode='diagnostic').definition.runtime.model_dump(mode='json')
    baseline_task, baseline_scene = builder.translate_definitions(parent_task, parent_scene, entry, {}, runtime)
    empty_task, empty_scene = builder.translate_definitions(parent_task, parent_scene,
        {**entry, 'native_object_poses': []}, {}, runtime)
    assert (empty_task, empty_scene) == (baseline_task, baseline_scene)
    poses = [{'name': 'native_chair', 'category': 'chair', 'model': 'exact_model',
              'position': [3., 5., .5], 'orientation': [0., 0., 0., 1.]}]
    task, scene = builder.translate_definitions(parent_task, parent_scene,
        {**entry, 'native_object_poses': poses}, {}, runtime)
    assert task['private'] == baseline_task['private']
    assert task['public'] == baseline_task['public'] and task['defaults'] == baseline_task['defaults']
    assert task['scene'] == scene['scene']
    assert task['scene']['variants']['original'] == {'patches': []}
    for variant_name, variant in task['scene']['variants'].items():
        if not variant['patches']:
            continue
        patches = variant['patches']
        furniture = [(index, patch) for index, patch in enumerate(patches)
                     if patch.get('module') == builder.NATIVE_FURNITURE_MODULE]
        assert len(furniture) == 1
        index, patch = furniture[0]
        assert patch == {'operation': 'category_initialization', 'phase': 'after_postprocessing',
            'module': builder.NATIVE_FURNITURE_MODULE,
            'parameters': {'scene_model': entry['scene_model'], 'native_object_poses': poses}}
        assert all(index < position for position, candidate in enumerate(patches)
            if candidate.get('phase') == 'after_postprocessing'
            and candidate.get('operation') in ('object_pose', 'category_initialization') and position != index)
        assert [row for position, row in enumerate(patches) if position != index] == baseline_task['scene']['variants'][variant_name]['patches']


def test_matched_groups_reject_inconsistent_retained_furniture_poses(design):
    poses = [{'name': 'native_chair', 'category': 'chair', 'model': 'exact_model',
              'position': [3., 5., .5], 'orientation': [0., 0., 0., 1.]}]
    for entry in design[2]['entries']:
        entry['native_object_poses'] = deepcopy(poses)
    design[2]['entries'][1]['native_object_poses'][0]['position'][0] += 1.
    design[0].write_text(json.dumps(design[2]))
    with pytest.raises(ValueError, match='Matched-group'):
        builder.build_package(*design[:2])


def test_explicit_per_variant_after_temporal_boundary_preserves_saved_diagnostics_default(design):
    entry = design[2]['entries'][0]
    parent_task, parent_scene = (yaml.safe_load(Path(entry['parent_' + kind + '_file']).read_text()) for kind in ('task', 'scene'))
    runtime = builder.load_scene(entry['parent_scene_file'], variant='adapted', run_mode='diagnostic').definition.runtime.model_dump(mode='json')
    poses = [{'name': 'native_chair', 'category': 'chair', 'model': 'exact_model',
              'position': [3., 5., .5], 'orientation': [0., 0., 0., 1.]}]
    before = {**entry, 'native_object_poses': poses}
    baseline, _ = builder.translate_definitions(parent_task, parent_scene, before, {}, runtime)
    defaults = {key: 'before_source_restore' for key in ('adapted', 'saved_image1', 'saved_image2')}
    explicit_default, _ = builder.translate_definitions(parent_task, parent_scene,
        {**before, 'native_furniture_hook_boundaries': defaults}, {}, runtime)
    assert explicit_default == baseline
    boundaries = {**defaults, 'adapted': 'after_temporal_initialization'}
    task, scene = builder.translate_definitions(parent_task, parent_scene,
        {**before, 'native_furniture_hook_boundaries': boundaries}, {}, runtime)
    assert task['private'] == baseline['private'] and task['public'] == baseline['public']
    assert task['defaults'] == baseline['defaults'] and task['scene'] == scene['scene']
    for key in ('saved_image1', 'saved_image2', 'original'):
        assert task['scene']['variants'][key] == baseline['scene']['variants'][key]
    patches = task['scene']['variants']['adapted']['patches']
    furniture = next(i for i, patch in enumerate(patches) if patch.get('module') == builder.NATIVE_FURNITURE_MODULE)
    temporal = [i for i, patch in enumerate(patches) if patch.get('module') == builder.MODULE and patch['phase'] == 'after_postprocessing']
    prepare = next(i for i, patch in enumerate(patches) if patch['operation'] == 'prepare_category_interaction')
    assert furniture == max(temporal) + 1 and furniture < prepare
    for bad in ({'unknown_variant': 'before_source_restore'}, {'saved_image1': 'after_temporal_initialization'}):
        with pytest.raises(ValueError):
            builder.translate_definitions(parent_task, parent_scene,
                {**before, 'native_furniture_hook_boundaries': bad}, {}, runtime)


def test_after_temporal_boundary_rejects_missing_ambiguous_or_prior_interaction_preparation():
    entry = {'scene_model': 'scene', 'native_object_poses': [{'name': 'native_table'}],
             'native_furniture_hook_boundaries': {'adapted': 'after_temporal_initialization'}}
    temporal = {'operation': 'category_initialization', 'phase': 'after_postprocessing', 'module': builder.MODULE}
    prepare = {'operation': 'prepare_category_interaction', 'phase': 'after_postprocessing'}
    for patches in ([temporal], [prepare, temporal], [temporal, prepare, prepare]):
        with pytest.raises(ValueError, match='two_phase initializer'):
            builder._insert_native_furniture_hook({'patches': deepcopy(patches)}, entry, 'adapted')


def test_matched_groups_reject_different_furniture_hook_boundaries(design):
    for entry in design[2]['entries']:
        entry['native_furniture_hook_boundaries'] = {'adapted': 'before_source_restore'}
    design[2]['entries'][1]['native_furniture_hook_boundaries']['adapted'] = 'after_temporal_initialization'
    design[0].write_text(json.dumps(design[2]))
    with pytest.raises(ValueError, match='Matched-group'):
        builder.build_package(*design[:2])


def test_world_apparatus_translation_and_vertical_only_leave_source_actors_and_history_intact(design):
    entry = design[2]['entries'][0]
    parent_task, parent_scene = (yaml.safe_load(Path(entry['parent_' + kind + '_file']).read_text()) for kind in ('task', 'scene'))
    runtime = builder.load_scene(entry['parent_scene_file'], variant='adapted', run_mode='diagnostic').definition.runtime.model_dump(mode='json')
    baseline, baseline_scene = builder.translate_definitions(parent_task, parent_scene, entry, {}, runtime)
    assert builder.translate_active_apparatus(baseline, None) == baseline
    assert baseline['scene'] == baseline_scene['scene']
    shift = [1., 1., 0.]
    changed_entry = {**entry, 'apparatus_translation_world_m': shift, 'interaction_relocation_policy': 'vertical_only'}
    changed, scene = builder.translate_definitions(parent_task, parent_scene, changed_entry, {}, runtime)
    assert changed['scene'] == scene['scene']
    p, old = changed['private']['evaluation']['parameters'], baseline['private']['evaluation']['parameters']
    assert {key: value for key, value in p.items() if key != 'apparatus'} == {key: value for key, value in old.items() if key != 'apparatus'}
    assert changed['defaults'] == baseline['defaults'] and changed['public'] == baseline['public']
    for kind in ('primitives', 'dataset_objects'):
        for before, after in zip(old['apparatus'][kind], p['apparatus'][kind]):
            assert after == {**before, 'position': (np.asarray(before['position']) + shift).tolist()}
    for name, variant in changed['scene']['variants'].items():
        if not variant['patches']:
            continue
        original_patches = baseline['scene']['variants'][name]['patches']
        for after, before in zip(variant['patches'], original_patches):
            if after['operation'] == 'category_initialization':
                a, b = after['parameters'], before['parameters']
                assert a['source_objects'] == b['source_objects']
                if 'apparatus' in a:
                    for kind in ('primitives', 'dataset_objects'):
                        for prior, shifted in zip(b['apparatus'][kind], a['apparatus'][kind]):
                            assert shifted == {**prior, 'position': (np.asarray(prior['position']) + shift).tolist()}
            elif after['operation'] == 'prepare_category_interaction':
                assert after == {**before, 'relocation_policy': 'vertical_only'}
            elif after['operation'] == 'object_pose':
                primitive = next((row for row in p['apparatus']['primitives']
                    if changed['private']['bindings'][after['binding']]['name'] == row['name']), None)
                if primitive:
                    assert after['position'] == primitive['position']
                    assert after['quaternion_xyzw'] == primitive['quaternion_xyzw']
                else:
                    assert after == before
            else:
                assert after == before
    assert changed['repair']['apparatus_translation_world_m'] == shift
    assert changed['repair']['interaction_relocation_policy'] == 'vertical_only'
    parent_source = json.loads(builder.sources.resolve_source(parent_task['source']['json']).read_bytes())
    assert builder.translate_source(parent_source, changed_entry, {}) == builder.translate_source(parent_source, entry, {})


def test_active_source_apparatus_translation_preserves_parking_historical_copies_and_local_geometry():
    apparatus = {'primitives': [{'name': 'change_collection_table_leg_0', 'position': [1., 2., .35],
        'dimensions_m': [.04, .04, .7], 'quaternion_xyzw': [0., 0., 0., 1.]}],
        'dataset_objects': [{'name': 'spare', 'position': [2., 3., .9], 'scale': [1., 2., 1.],
            'orientation': [0., 0., 0., 1.]}, {'name': 'inactive_spare', 'position': [1000., 1000., 120.]}],
        'source_target_positions': {'original_source_actor': [1., 2., .9]}}
    source = {'metadata_json': {'apparatus': apparatus, 'source_objects': {
        'target': {'position': [1., 2., .9]}, 'inactive': {'position': [1000., 1000., 120.]}}},
        'provenance': {'apparatus': deepcopy(apparatus)}, 'world_reference': {'position': [5., 6., 0.]}}
    before = deepcopy(source)
    moved = builder.translate_active_apparatus(source, [1., 1., 0.], floor_z=0.)
    assert source == before
    assert moved['metadata_json']['source_objects'] == source['metadata_json']['source_objects']
    assert moved['provenance'] == source['provenance'] and moved['world_reference'] == source['world_reference']
    assert moved['metadata_json']['apparatus']['source_target_positions'] == apparatus['source_target_positions']
    for kind in ('primitives', 'dataset_objects'):
        for old, new in zip(apparatus[kind], moved['metadata_json']['apparatus'][kind]):
            if any(abs(value) >= 900 for value in old['position']):
                assert new == old
            else:
                assert new == {**old, 'position': (np.asarray(old['position']) + [1., 1., 0.]).tolist()}


@pytest.mark.parametrize('field,value', [('apparatus_translation_world_m', [1., 1., .1]),
    ('apparatus_translation_world_m', [1., float('nan'), 0.]),
    ('apparatus_translation_world_m', [1., 1.]), ('interaction_relocation_policy', 'unsupported')])
def test_apparatus_translation_and_interaction_policy_require_explicit_valid_controls(design, field, value):
    with pytest.raises(ValueError):
        builder._entry_contract({**design[2]['entries'][0], field: value})


@pytest.mark.parametrize('field,initial,changed', [('apparatus_translation_world_m', [1., 1., 0.], [2., 1., 0.]),
    ('interaction_relocation_policy', 'independent_groups', 'vertical_only')])
def test_matched_groups_reject_different_apparatus_or_interaction_controls(design, field, initial, changed):
    for entry in design[2]['entries']:
        entry[field] = deepcopy(initial)
    design[2]['entries'][1][field] = deepcopy(changed)
    design[0].write_text(json.dumps(design[2]))
    with pytest.raises(ValueError, match='Matched-group'):
        builder.build_package(*design[:2])

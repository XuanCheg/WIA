"""Check the two native rigid-support revisions with real parents on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_rigid_native_support_revisions.py
Temporary outputs only; no simulator, capture queue or original sources are edited.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from wia.tasks.loader import load_scene, load_task
import wia.authoring.commands.build_wia_rigid_native_support_revisions as builder


def parents(design):
    return tuple(yaml.safe_load(builder.repo_path(design['parent_' + kind + '_file']).read_text())
                 for kind in ('task', 'scene'))


def pose_map(task):
    return {row['binding']: row for row in builder.patches(task) if row['operation'] == 'object_pose'}


def test_real_parents_preserve_scorer_sources_rotations_scales_budgets_and_only_declared_patch_changes(tmp_path):
    definitions, manifest, rendered = builder.build_package(tmp_path)
    assert len(definitions) == len(manifest['entries']) == 2 and len(rendered) == 5
    for definition, design, record in zip(definitions, builder.DESIGNS, manifest['entries']):
        parent_task, parent_scene = parents(design)
        task, scene = definition['task'], definition['scene']
        assert task['private']['evaluation'] == parent_task['private']['evaluation']
        assert task['private']['evaluation']['parameters']['allow_remove_after_verification'] is True
        assert task['defaults'] == parent_task['defaults']
        assert task['source'] == parent_task['source'] and scene['source'] == parent_scene['source']
        assert task['public'] == {**parent_task['public'], 'environment': design['environment']}
        for field in ('requirements', 'qualification', 'kind', 'id'):
            assert task[field] == parent_task[field]
        for field in ('qualification', 'runtime', 'source', 'target', 'seed'):
            assert scene.get(field) == parent_scene.get(field)
        assert task['scene'] == scene['scene']
        assert task['private']['bindings'] == scene['bindings']
        assert set(task['private']['bindings']) == set(parent_task['private']['bindings'])
        expected_bindings = deepcopy(parent_task['private']['bindings'])
        expected_bindings['table'] = {'name': design['support_name'], 'category': design['support_category'],
                                      'model': 'skczfi' if design['document_alias'] == 'T22' else 'jhymlr'}
        if design['document_alias'] == 'T22':
            expected_bindings['pickup_support'] = deepcopy(expected_bindings['table'])
        assert task['private']['bindings'] == expected_bindings
        before_poses, after_poses = pose_map(parent_task), pose_map(task)
        assert set(before_poses) == set(after_poses)
        for key, original in before_poses.items():
            expected = deepcopy(original)
            if key.startswith('container_'):
                index = int(key[-1])
                expected['position'] = [*design['container_xy'][index],
                    original['position'][2] - .45 + design['support_top_m']]
            elif design['document_alias'] == 'T22' and key == 'target':
                expected['position'][:2] = [-3.75, 1.93]
            assert after_poses[key] == expected
        original_kept = [row for row in builder.patches(parent_task)
                         if not (row['operation'] == 'add_primitive' and row['name'] in builder.REMOVED_PRIMITIVES)]
        current_kept = [row for row in builder.patches(task) if row['operation'] != 'category_initialization']
        original_non_hooks = [row for row in original_kept if row['operation'] != 'category_initialization']
        assert len(current_kept) == len(original_non_hooks)
        for original, actual in zip(original_non_hooks, current_kept):
            expected = deepcopy(original)
            if original['operation'] == 'object_pose':
                expected = after_poses[original['binding']]
            if original['operation'] == 'robot_pose' and design['document_alias'] == 'T22':
                expected.update(position=[-3.4, .85, .1], yaw_deg=90.)
            # Includes every native object-scale and add-dataset-object scale,
            # original quaternion, robot option and unchanged nuisance patch.
            assert actual == expected
        assert not any(row.get('name') in builder.REMOVED_PRIMITIVES for row in builder.patches(task))
        assert task['scene']['variants']['original'] == parent_task['scene']['variants']['original']
        assert task['version'] == parent_task['version'] + 1
        assert scene['version'] == parent_scene['version'] + 1
        assert record['native_validation_run'] is record['physical_success_verified'] is False
        assert record['variant'] == 'tabletop' and record['seed'] == parent_task['defaults']['seed']
        assert record['task_source_identity'] == parent_task['source']['json']
        assert record['scene_source_identity'] == parent_scene['source']['json']
        for prefix in ('task', 'scene', 'parent_task', 'parent_scene', 'task_source', 'scene_source', 'inventory'):
            path = Path(record[prefix + '_file'])
            assert record[prefix + '_sha256'] == builder.sha256(rendered[path] if path in rendered else path.read_bytes())


def test_native_support_hooks_match_pose_patches_and_preserve_t23_sofa_rows_exactly(tmp_path):
    definitions, _, _ = builder.build_package(tmp_path)
    for definition, design in zip(definitions, builder.DESIGNS):
        task = definition['task']
        hooks = builder._initializer(task)
        assert len(hooks) == 1
        hook = hooks[0]
        assert hook['module'] == builder.INITIALIZER and hook['phase'] == 'after_postprocessing'
        assert hook['parameters']['revision'] == design['variant_id']
        layout = hook['parameters']['layout']
        assert layout['floor_name'] == design['floor_name']
        poses = pose_map(task)
        for key, pose in poses.items():
            name = task['private']['bindings'][key]['name']
            row = next(row for row in layout['objects'] if row['name'] == name)
            assert row['position'] == pose['position'] and row['orientation'] == pose['quaternion_xyzw']
            support = next(row for row in layout['supported_items'] if row['name'] == name)
            if key.startswith('container_') or design['document_alias'] == 'T22':
                assert support == {'name': name, 'support': design['support_name'],
                    'support_alignment_mode': 'local_raycast', 'support_surface_mode': 'local_raycast'}
        if design['document_alias'] == 'T23':
            parent_task, _ = parents(design)
            parent_layout = builder._initializer(parent_task)[0]['parameters']['layout']
            for field in ('objects', 'supported_items'):
                originals = [row for row in parent_layout[field] if not row['name'].startswith('obj_container_')]
                actual = [row for row in layout[field] if not row['name'].startswith('obj_container_')]
                assert actual == originals


def test_outputs_load_separate_source_roles_and_are_immutable_and_deterministic(tmp_path):
    source_paths = {builder.repo_path(design['parent_' + kind + '_file'])
                    for design in builder.DESIGNS for kind in ('task', 'scene')}
    for design in builder.DESIGNS:
        task = load_task(builder.repo_path(design['parent_task_file']), variant='tabletop', run_mode='diagnostic')
        scene = load_scene(builder.repo_path(design['parent_scene_file']), variant='tabletop', run_mode='diagnostic')
        source_paths.update((task.source_path, scene.source_path, builder.repo_path(design['inventory_file'])))
    before = {path: path.read_bytes() for path in source_paths}
    definitions, manifest, rendered = builder.build_package(tmp_path / 'output')
    assert builder.build_package(tmp_path / 'output')[2] == rendered
    args = ['--output-dir', str(tmp_path / 'output')]
    assert builder.main(args + ['--check']) == 1
    assert builder.main(args) == 0
    assert json.loads((tmp_path / 'output/manifest.json').read_bytes()) == manifest
    mtimes = {path: path.stat().st_mtime_ns for path in rendered}
    assert builder.main(args + ['--check']) == 0
    assert builder.main(args) == 0
    assert {path: path.stat().st_mtime_ns for path in rendered} == mtimes
    t23_task = load_task(definitions[1]['task_file'], variant='tabletop', run_mode='diagnostic')
    t23_scene = load_scene(definitions[1]['scene_file'], variant='tabletop', run_mode='diagnostic')
    assert t23_task.source_path != t23_scene.source_path
    assert t23_task.definition.source.json_path.endswith('__criterion_4.json')
    assert t23_scene.definition.source.json_path.endswith('_revised.json')
    changed = definitions[0]['task_file']
    changed.write_bytes(changed.read_bytes() + b'\n')
    altered = changed.read_bytes()
    assert builder.main(args + ['--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing generated file'):
        builder.main(args)
    assert changed.read_bytes() == altered
    assert {path: path.read_bytes() for path in source_paths} == before

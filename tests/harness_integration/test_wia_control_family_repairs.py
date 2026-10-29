"""Validate explicit family repairs without simulation.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -B -m pytest -q
  tests/test_wia_control_family_repairs.py
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import wia.authoring.commands.build_wia_control_family_repairs as builder


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    output = tmp_path_factory.mktemp('control_family')
    manifest, audit, rendered = builder.build_package(output)
    for path, data in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return manifest, audit, rendered


def test_complete_exact_membership_both_seeds_and_loader_contract(package):
    manifest, audit, rendered = package
    assert len(audit['dimensional_pairs']) == 20
    assert {r['document_alias'] for r in audit['dimensional_pairs']} == builder.DIMENSIONAL_ALIASES
    assert {r['document_alias'] for r in audit['tray_families']} == builder.TRAY_ALIASES
    assert len(manifest['entries']) == 56
    assert len({r['task_id'] for r in manifest['entries']}) == 56
    assert sum(r['geometry_variant'] == 'i0_open' for r in manifest['entries']) == 20
    assert {r['seed'] for r in manifest['entries']} == {101, 202}
    assert {r['geometry']['scene_model'] for r in manifest['entries']} == {
        'hall_glass_ceiling', 'hall_train_station', 'hall_arch_wood'}
    builder.validate_outputs(manifest)
    for row in manifest['entries']:
        assert row['document_alias'].startswith('D1-') or row['geometry_variant'] == 'i0_open'
        assert all(row[k] is False for k in builder.FLAGS)
        assert row['geometry']['minimum_support_margin_m'] >= .05 - 1e-9
        assert all(leg['under_table'] for leg in row['geometry']['legs'])
    assert all(builder.digest(path) == sha for path, sha in audit['input_sha256'].items())


def test_all_changes_preserve_scientific_geometry_and_unmodified_furniture(package):
    manifest, _, _ = package
    for row in manifest['entries']:
        before = yaml.safe_load(Path(row['parent_task_file']).read_text())
        after = yaml.safe_load(Path(row['task_file']).read_text())
        scene = yaml.safe_load(Path(row['scene_file']).read_text())
        assert after['source'] == before['source']
        assert after['public'] == before['public']
        assert after['defaults'] == before['defaults']
        private = deepcopy(after['private'])
        setup_layout = private['evaluation']['parameters'].pop('layout')
        assert private == before['private']
        assert after['requirements'] == before['requirements']
        assert after['repair']['runtime'] == before['repair']['runtime']
        assert after['version'] == before['version'] + 1
        assert after['scene'] == scene['scene']
        old = builder.dimensional_layout(before)
        new = builder.dimensional_layout(after)
        assert setup_layout == new
        assert new['initial_gripper_state'] == 'open'
        assert old['robot_start'] == new['robot_start']
        assert old['supported_items'] == new['supported_items']
        assert new == builder.layouts(after)[1]
        center = next(o['position'] for o in old['objects'] if o['name'] == 'wia_dim_table')
        for original, changed in zip(old['objects'], new['objects']):
            expected = deepcopy(original)
            name = original['name']
            variant = row['geometry_variant']
            if variant == 'g1_closer' and '_candidate_' in name:
                expected['position'][0] -= .1
            if variant == 'g2_front_crop':
                if name == 'wia_dim_table':
                    expected['position'][0] += .1
                    expected['scale'][0] -= .2
                if '_leg_' in name and original['position'][0] < center[0]:
                    expected['position'][0] += .2
            if variant == 'g3_narrow':
                if name == 'wia_dim_table':
                    expected['scale'][1] = 1.2
                if '_leg_' in name:
                    expected['position'][1] -= float(np.sign(original['position'][1] - center[1])) * .6
            assert changed == expected


def test_rotated_d1_3_uses_exact_footprint_and_translations(package):
    _, audit, _ = package
    rows = [r for r in audit['dimensional_pairs'] if r['document_alias'] == 'D1-3']
    assert len(rows) == 4
    bounds = []
    for row in rows:
        obj = next(o for o in row['geometry']['objects'] if o['name'] == 'wia_dim_candidate_0')
        low, high = np.asarray(obj['bounds_table_frame_m'])
        assert high[0] - low[0] == pytest.approx(obj['dimensions_m'][1])
        assert high[1] - low[1] == pytest.approx(obj['dimensions_m'][0])
        assert obj['dimensions_m'][0] != obj['dimensions_m'][1]
        bounds.append(high - low)
    assert all(np.allclose(bound, bounds[0]) for bound in bounds)


def test_d2_width_and_actual_tray_initializer_membership(package):
    _, audit, _ = package
    for row in audit['dimensional_pairs']:
        if row['document_alias'].startswith('D2-'):
            assert row['geometry']['required_centered_width_with_margin_m'] == pytest.approx(1.98)
            task, scene = builder.read_pair(row)
            with pytest.raises(ValueError, match='Only the reviewed D1'):
                builder.repair_pair(task, scene, 'g3_narrow', {})
    for row in audit['tray_families']:
        if row['document_alias'] in {'T38', 'T39'}:
            assert row['initializer'] is None and not row['tray_initializer_present']
            assert row['destination_object']['fixed_base']
            assert row['destination_object']['name'] == 'pantry_delivery_tray'
        else:
            assert row['tray_initializer_present']
            assert row['plate']['right_eef_position_base_m'] == [.4, -.25, .8]
            assert row['plate']['model'] == 'gsxbym'
        if row['document_alias'] == 'T07':
            assert 't07_native_velocity_v2' in row['task_file']


def test_inconsistent_initialization_and_evaluation_copies_rejected(package):
    _, audit, _ = package
    row = audit['dimensional_pairs'][0]
    task, scene = builder.read_pair(row)
    mismatch = deepcopy(task)
    builder.layouts(mismatch)[1]['objects'][-1]['position'][0] += .01
    with pytest.raises(ValueError, match='copies must match'):
        builder.dimensional_layout(mismatch)
    mismatch = deepcopy(task)
    mismatch['private']['evaluation']['parameters']['candidate_dimensions_m']['candidate_0'][0] += .01
    with pytest.raises(ValueError, match='dimensions disagree'):
        builder.dimensional_layout(mismatch)
    mismatch = deepcopy(task)
    mismatch['private']['evaluation']['parameters']['layout'] = deepcopy(builder.layouts(task)[0])
    mismatch['private']['evaluation']['parameters']['layout']['objects'][-1]['position'][0] += .01
    with pytest.raises(ValueError, match='Evaluation layout differs'):
        builder.dimensional_layout(mismatch)
    with pytest.raises(ValueError, match='combinations are forbidden'):
        builder.repair_pair(task, scene, 'g1_closer+g2_front_crop', {})
    tray = next(r for r in audit['tray_families'] if r['document_alias'] == 'T08')
    task, _ = builder.read_pair(tray)
    task['private']['evaluation']['parameters']['plate']['rim_penetration_m'] += .001
    with pytest.raises(ValueError, match='setup and evaluation copies differ'):
        builder.tray_audit(tray, task)


def test_tray_p1_versions_preserve_all_nonpose_scientific_and_scene_fields(tmp_path):
    manifest, rendered = builder.build_tray_p1_package(tmp_path)
    for path, content in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert {r['document_alias'] for r in manifest['entries']} == {'T07', 'T08', 'T35', 'T36', 'T37'}
    assert len(manifest['entries']) == 5
    builder.validate_outputs(manifest)
    for row in manifest['entries']:
        parent_task = yaml.safe_load(Path(row['parent_task_file']).read_text())
        parent_scene = yaml.safe_load(Path(row['parent_scene_file']).read_text())
        task = yaml.safe_load(Path(row['task_file']).read_text())
        scene = yaml.safe_load(Path(row['scene_file']).read_text())
        assert task['id'] != parent_task['id'] and task['version'] == parent_task['version'] + 1
        assert scene['id'] != parent_scene['id'] and scene['version'] == parent_scene['version'] + 1
        assert task['scene'] == scene['scene']
        assert task['repair']['runtime'] == parent_task['repair']['runtime']
        assert task['repair']['full_task_qualified'] is False
        assert task['repair']['unloaded_tray_probe']['scope'].startswith('P1 initialization and six short unloaded')
        assert all(row[name] is False for name in builder.FLAGS)
        assert row['seed'] == parent_task['defaults']['seed']
        assert builder.digest(row['parent_task_file']) == row['parent_task_sha256']
        assert builder.digest(row['parent_scene_file']) == row['parent_scene_sha256']
        reverted_task, reverted_scene = deepcopy(task), deepcopy(scene)
        scene_copies = 8 if row['document_alias'] == 'T07' else 4
        assert builder._replace_tray_pose(reverted_task['scene'], [.4, -.25, .8]) == scene_copies
        assert builder._replace_tray_pose(reverted_task['private']['evaluation'], [.4, -.25, .8]) == 2
        assert builder._replace_tray_pose(reverted_scene['scene'], [.4, -.25, .8]) == scene_copies
        for key in ('id', 'version', 'status', 'repair'):
            reverted_task.pop(key, None)
        for key in ('id', 'version', 'status'):
            reverted_scene.pop(key, None)
        parent_task = {k: v for k, v in parent_task.items() if k not in ('id', 'version', 'status', 'repair')}
        parent_scene = {k: v for k, v in parent_scene.items() if k not in ('id', 'version', 'status')}
        assert reverted_task == parent_task
        assert reverted_scene == parent_scene
        if row['document_alias'] == 'T07':
            assert 't07_native_velocity_v2/task.yaml' in row['parent_task_file']
            velocity = [p for p in task['scene']['variants']['adapted']['patches']
                        if p['module'] == 'wia.authoring.converters.wia_initial_joint_velocity']
            assert len(velocity) == 1


def test_tray_p1_generation_rejects_unqualified_or_wrong_probe_evidence(tmp_path):
    import json
    probe = json.loads(builder.TRAY_P1_PROBE.read_text())
    probe['public_motion_probes_passed'] = False
    path = tmp_path / 'failed_probe.json'
    path.write_text(json.dumps(probe))
    with pytest.raises(ValueError, match='exact retained unloaded P1 probe'):
        builder.build_tray_p1_package(tmp_path, path)


@pytest.fixture(scope='module')
def package_v2(tmp_path_factory):
    output = tmp_path_factory.mktemp('control_family_v2')
    manifest, audit, rendered = builder.build_package_v2(output)
    for path, content in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return manifest, audit, rendered


def test_v2_versions_all_d1_geometries_and_references_unchanged_d2_without_rewriting_v1(package_v2):
    import json
    manifest, audit, rendered = package_v2
    assert len(manifest['entries']) == 56 and len(audit['entries']) == 48
    assert len(rendered) == 99
    assert len({row['task_id'] for row in manifest['entries']}) == 56
    parent = json.loads((builder.DEFAULT_OUTPUT / 'manifest.json').read_text())
    original_d2 = [row for row in parent['entries'] if row['document_alias'].startswith('D2-')]
    retained_d2 = [row for row in manifest['entries'] if row['document_alias'].startswith('D2-')]
    assert retained_d2 == original_d2
    assert all(Path(row['task_file']).is_relative_to(builder.DEFAULT_OUTPUT) for row in retained_d2)
    d1 = [row for row in manifest['entries'] if row['document_alias'].startswith('D1-')]
    assert {row['geometry_variant'] for row in d1} == {'i0_open', 'g1_closer', 'g2_front_crop', 'g3_narrow'}
    assert {row['seed'] for row in d1} == {101, 202}
    assert {row['geometry']['scene_model'] for row in d1} == {'hall_glass_ceiling', 'hall_train_station', 'hall_arch_wood'}
    assert all(builder.digest(path) == expected for path, expected in audit['input_sha256'].items())
    builder.validate_outputs(manifest)


def test_v2_keeps_objective_success_yaml_bytes_and_all_physical_goals_and_thresholds(package_v2):
    manifest, _, _ = package_v2
    for row in manifest['entries']:
        if not row['document_alias'].startswith('D1-'):
            continue
        parent_text = Path(row['parent_task_file']).read_text()
        new_text = Path(row['task_file']).read_text()
        parent = yaml.safe_load(parent_text)
        task = yaml.safe_load(new_text)
        parent_scene = yaml.safe_load(Path(row['parent_scene_file']).read_text())
        scene = yaml.safe_load(Path(row['scene_file']).read_text())
        for field in ('objective', 'success_criteria'):
            assert builder._public_scalar_block(new_text, field).encode() == builder._public_scalar_block(parent_text, field).encode()
            assert task['public'][field] == parent['public'][field]
        assert task['id'] != parent['id'] and task['version'] == parent['version'] + 1
        assert scene['id'] != parent_scene['id'] and scene['version'] == parent_scene['version'] + 1
        assert task['scene'] == parent['scene'] == scene['scene']
        assert task['repair']['runtime'] == parent['repair']['runtime']
        assert task['source'] == parent['source'] == scene['source']
        assert task['defaults'] == parent['defaults']
        assert task['requirements'] == parent['requirements']
        params = task['private']['evaluation']['parameters']
        original_params = parent['private']['evaluation']['parameters']
        assert params['scoring'] == original_params['scoring']
        assert params['completion_items'] == original_params['completion_items']
        assert params['layout'] == original_params['layout']
        assert params['layout']['initial_gripper_state'] == 'open'
        expected_private = deepcopy(parent['private'])
        assert builder._drop_unselected_tolerance(expected_private) == 1
        assert task['private'] == expected_private
        assert builder._drop_unselected_tolerance(deepcopy(task)) == 0
        expected_public = deepcopy(parent['public'])
        expected_public['constraints'].remove(builder.D1_COMMITMENT_CONSTRAINT)
        expected_public['constraints'].extend(builder.D1_FREE_ATTEMPT_CONSTRAINTS)
        assert task['public'] == expected_public
        assert task['repair']['d1_evaluation_policy'] == 'done_only_free_attempts'
        assert task['repair']['scoring_version'] == 'wia_dimensional_progress_v2'
        assert 'terminal candidate' in task['repair']['completion_item_meaning']['choice']
        assert all(task['repair'][key] is False and row[key] is False for key in builder.FLAGS)
        for key in ('id', 'version', 'status', 'repair', 'public', 'private'):
            task.pop(key)
            parent.pop(key)
        assert task == parent
        for key in ('id', 'version', 'status'):
            scene.pop(key)
            parent_scene.pop(key)
        assert scene == parent_scene

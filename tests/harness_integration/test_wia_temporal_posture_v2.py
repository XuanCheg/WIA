"""Check the bounded construction-posture revision without native execution.

Run: python -B -m pytest -q tests/test_wia_temporal_posture_v2.py -p no:cacheprovider
"""
from copy import deepcopy
import hashlib
import json

import pytest
import yaml

from wia.tasks.loader import load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.authoring.commands.build_wia_temporal_posture_v2 import (
    DEFAULT_OUTPUT, PARENT, POSTURE_CONFIG, POSTURE_KEY, PUBLIC_TASK_ID,
    REVISION, ROOT, VARIANTS, build_package, load_posture, main,
)


@pytest.fixture(scope='module')
def package():
    return build_package()


def normalized(value):
    result = deepcopy(value)
    for key in ('id', 'version', 'status'):
        result.pop(key)
    for patch in result['scene']['variants']['adapted']['patches']:
        if patch['operation'] == 'prepare_category_interaction':
            patch.pop(POSTURE_KEY)
    return result


def test_schema_and_normalized_exact_diff_allow_only_posture_and_version_metadata(package):
    definitions, _, _ = package
    production = json.loads(POSTURE_CONFIG.read_text())['initial_trunk_joint_positions_rad']
    for variant, (task, scene) in definitions.items():
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        assert task['id'] == PUBLIC_TASK_ID
        assert scene['id'] == f'wia_tm2_history_{variant}_v2_scene'
        assert task['version'] == scene['version'] == 2
        assert task['status'] == scene['status'] == 'pending'
        assert task['scene'] == scene['scene']
        for kind, revised in (('tasks', task), ('scenes', scene)):
            original = yaml.safe_load((PARENT / kind / f'history_{variant}_v1.yaml').read_text())
            assert normalized(revised) == normalized(original)
            before = [patch[POSTURE_KEY] for patch in original['scene']['variants']['adapted']['patches']
                      if patch['operation'] == 'prepare_category_interaction']
            after = [patch[POSTURE_KEY] for patch in revised['scene']['variants']['adapted']['patches']
                     if patch['operation'] == 'prepare_category_interaction']
            assert before and before == [None] * len(before)
            assert after == [production] * len(before)
        parent_task = yaml.safe_load((PARENT / 'tasks' / f'history_{variant}_v1.yaml').read_text())
        assert task['public'] == parent_task['public']
        assert task['private']['evaluation'] == parent_task['private']['evaluation']
        assert task['repair'] == parent_task['repair']


def test_full_loader_public_payload_is_shared_and_unchanged_except_id_and_version(package):
    _, manifest, _ = package
    loaded = [load_task(DEFAULT_OUTPUT / 'tasks' / f'history_{variant}_v2.yaml',
                        variant='adapted', run_mode='diagnostic') for variant in VARIANTS]
    payloads = [task.public_payload for task in loaded]
    assert payloads[0] == payloads[1] == payloads[2] == manifest['common_public_payload']
    baseline = load_task(PARENT / 'tasks/history_a_v1.yaml', variant='adapted', run_mode='diagnostic')
    expected = {**baseline.public_payload, 'task_id': PUBLIC_TASK_ID, 'task_version': 2}
    assert payloads[0] == expected
    assert loaded[0].task_text == baseline.task_text
    assert 'history_a' not in json.dumps(payloads[0])


def test_posture_is_read_from_the_supplied_config_instead_of_carried_constants(tmp_path):
    path = tmp_path / 'arm.json'
    posture = [.1, -.2, .3, -.4]
    path.write_text(json.dumps({'initial_trunk_joint_positions_rad': posture}))
    assert load_posture(path) == posture
    for invalid in ([0., 1., 2.], [0., 1., float('nan'), 3.]):
        path.write_text(json.dumps({'initial_trunk_joint_positions_rad': invalid}))
        with pytest.raises(ValueError, match='four finite'):
            load_posture(path)


def test_manifest_records_exact_inputs_and_keeps_the_causal_hypothesis_pending(package):
    _, manifest, rendered = package
    assert manifest['schema'] == REVISION
    assert manifest['hypothesis_status'] == 'UNPROVEN'
    assert manifest['native_experiments_run'] is False
    assert manifest['v1_definitions_and_sources_modified'] is False
    assert manifest['posture_configuration']['file'] == 'configs/esi_arm.json'
    assert manifest['posture_configuration']['sha256'] == hashlib.sha256(POSTURE_CONFIG.read_bytes()).hexdigest()
    for key in ('native_initialization', 'robot_stability', 'native_phase_2_equivalence',
                'observation_shortcut_audit', 'reference_execution', 'hypothesis_retention_decision'):
        assert manifest[key] == 'PENDING'
    assert manifest['observation_contract']['frozen_initial_overview_retained'] is True
    assert manifest['observation_contract']['full_input_history_status'] == 'history_available'
    for row in manifest['instances']:
        for kind in ('task', 'scene'):
            assert row[f'parent_{kind}_sha256'] == hashlib.sha256((ROOT / row[f'parent_{kind}_file']).read_bytes()).hexdigest()
        assert row['source_sha256'] == hashlib.sha256((ROOT / row['source_file']).read_bytes()).hexdigest()
        assert {change['definition'] for change in row['matched_changes']} == {'task', 'scene'}
        assert all(change['before'] is None for change in row['matched_changes'])
        assert all(change['after'] == load_posture() for change in row['matched_changes'])
    assert len(rendered) == 7


def test_generation_and_check_are_deterministic_and_leave_all_inputs_unchanged(package, tmp_path):
    _, manifest, rendered = package
    inputs = {POSTURE_CONFIG, PARENT / 'manifest.json'}
    for row in manifest['instances']:
        inputs.update(ROOT / row[key] for key in ('parent_task_file', 'parent_scene_file', 'source_file'))
    before = {path: path.read_bytes() for path in inputs}
    assert DEFAULT_OUTPUT.is_absolute()
    assert build_package()[2] == rendered
    assert main(['--check']) == 0
    assert main(['--output-dir', str(tmp_path)]) == 0
    mtimes = {name: (tmp_path / name).stat().st_mtime_ns for name in rendered}
    assert main(['--output-dir', str(tmp_path), '--check']) == 0
    assert {name: (tmp_path / name).read_bytes() for name in rendered} == rendered
    assert {name: (tmp_path / name).stat().st_mtime_ns for name in rendered} == mtimes
    stale = tmp_path / 'tasks/history_a_v2.yaml'
    stale.write_bytes(stale.read_bytes() + b'\n')
    assert main(['--output-dir', str(tmp_path), '--check']) == 1
    assert stale.read_bytes() == rendered[next(name for name in rendered if name.as_posix() == 'tasks/history_a_v2.yaml')] + b'\n'
    assert main(['--output-dir', str(tmp_path)]) == 0
    assert {path: path.read_bytes() for path in before} == before


def test_unrelated_output_is_preserved_and_no_other_outputs_are_written(tmp_path):
    unrelated = tmp_path / 'scenes/history_c_v2.yaml'
    unrelated.parent.mkdir()
    unrelated.write_text('id: unrelated_scene\n')
    before = unrelated.read_bytes()
    with pytest.raises(ValueError, match='Refusing to overwrite an unrelated file'):
        main(['--output-dir', str(tmp_path)])
    assert unrelated.read_bytes() == before
    assert not (tmp_path / 'tasks').exists()
    assert not (tmp_path / 'manifest.json').exists()

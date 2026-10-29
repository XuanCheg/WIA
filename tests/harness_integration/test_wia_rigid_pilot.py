"""Check the T25 scored overlay contract without native simulation.

Run: python -B -m pytest -q tests/test_wia_rigid_pilot.py
"""
from copy import deepcopy
import json

import yaml

from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.authoring.commands.build_wia_rigid_pilot import (
    CAPABILITIES, DEFAULT_OUTPUT, PARENT_SCENE, PARENT_TASK, REVISION,
    SCENE_FILENAME, TASK_FILENAME, build_definitions, main, render_definitions,
)


def test_pending_definitions_validate_and_retain_original_episode_configuration():
    task, scene = build_definitions()
    validated = CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    original = yaml.safe_load(PARENT_TASK.read_text(encoding='utf-8'))
    assert (task['kind'], task['id'], task['version'], task['status']) == (
        'category_robot_task', 'wia_rigid_t25_v1', 1, 'pending')
    assert (scene['id'], scene['version'], scene['status']) == (
        'wia_rigid_t25_v1_scene', 1, 'pending')
    assert task['source'] == scene['source'] == original['source']
    assert task['defaults'] == {**original['defaults'], 'action_mode': 'parameterized'}
    assert (validated.defaults.seed, validated.defaults.max_steps) == (0, 60)
    assert task['qualification'] == {
        'on_failure': 'skip', 'support': None, 'initial_visibility': 'none'}
    assert task['repair']['validation_status'] == 'pending-native-validation'
    assert task['repair']['native_validation_run'] is False
    assert task['repair']['physically_qualified'] is False


def test_scene_is_identical_except_version_metadata_and_task_retains_all_patches():
    task, scene = build_definitions()
    original = yaml.safe_load(PARENT_SCENE.read_text(encoding='utf-8'))
    restored = deepcopy(scene)
    for key in ('id', 'version', 'status'):
        restored[key] = original[key]
    assert restored == original
    assert task['scene'] == scene['scene']
    assert task['private']['bindings'] == scene['bindings']
    assert task['repair']['original_scene_qualification'] == scene['qualification']
    assert scene['scene']['variants']['original']['patches'] == []
    # The clone preserves the historical byte representation outside three lines.
    raw = PARENT_SCENE.read_bytes().splitlines(keepends=True)
    clone = render_definitions()[SCENE_FILENAME].splitlines(keepends=True)
    assert len(raw) == len(clone)
    assert [line for line in raw if not line.startswith((b'id:', b'version:', b'status:'))] == [
        line for line in clone if not line.startswith((b'id:', b'version:', b'status:'))]


def test_private_scorer_targets_correct_container_and_explicit_pilot_thresholds():
    task, scene = build_definitions()
    evaluation = task['private']['evaluation']
    assert evaluation['type'] == 'category_operation'
    assert evaluation['module'] == 'wia.authoring.converters.rigid_partial_insertion'
    assert evaluation['operation'] == 'pick_release_partial_insert'
    assert evaluation['binding_keys'] == list(scene['bindings'])
    params = evaluation['parameters']
    assert params['scorer_version'] == task['repair']['contract_revision'] == REVISION
    assert params['targets'] == ['target']
    assert params['containers'] == ['container_0', 'container_1', 'container_2']
    assert params['expected_containers'] == {'target': 'container_1'}
    assert params['target_labels'] == {'target': 'banana'}
    assert params['expected_answers'] == {'target': 'tupperware'}
    assert scene['bindings']['container_1']['category'] == 'tupperware'
    assert scene['bindings']['target']['category'] == 'banana'
    assert {key: params[key] for key in (
        'minimum_lift_m', 'stability_duration_s', 'maximum_linear_speed_m_s',
        'maximum_angular_speed_rad_s', 'max_sample_gap_s', 'interior_margin_m',
        'penetration_tolerance_m', 'withdrawal_clearance_m', 'container_pose_tolerance_m',
        'maximum_container_rotation_rad', 'scale_relative_tolerance',
        'initial_support_tolerance_m', 'maximum_initial_linear_speed_m_s',
        'maximum_initial_angular_speed_rad_s')} == {
            'minimum_lift_m': 0.05, 'stability_duration_s': 0.5,
            'maximum_linear_speed_m_s': 0.02, 'maximum_angular_speed_rad_s': 0.1,
            'max_sample_gap_s': 0.1, 'interior_margin_m': 0.002,
            'penetration_tolerance_m': 0.001, 'withdrawal_clearance_m': 0.02,
            'container_pose_tolerance_m': 0.02, 'maximum_container_rotation_rad': 0.05,
            'scale_relative_tolerance': 0.0001, 'initial_support_tolerance_m': 0.015,
            'maximum_initial_linear_speed_m_s': 0.02,
            'maximum_initial_angular_speed_rad_s': 0.2,
        }
    assert params['allow_remove_after_verification'] is True
    assert params['initial_supports'] == {
        'target': 'pickup_support', 'container_0': 'table',
        'container_1': 'table', 'container_2': 'table'}


def test_public_goal_preserves_question_without_answer_or_private_identity_leaks():
    task, scene = build_definitions()
    public = task['public']
    original = yaml.safe_load(PARENT_TASK.read_text(encoding='utf-8'))['public']
    assert public['objective'].startswith(original['objective'])
    appended = public['objective'][len(original['objective']):]
    added_public = appended + json.dumps({key: public[key] for key in (
        'success_criteria', 'environment', 'constraints')})
    assert 'tupperware' not in added_public.lower()
    assert 'ANSWER: <container label>' in public['success_criteria']
    assert 'DONE' in public['success_criteria']
    assert 'partially' in public['success_criteria']
    assert '0.5 seconds' in public['success_criteria']
    assert 'entirely on top' in public['success_criteria']
    assert 'expected_answers' not in json.dumps(public)
    for key, binding in scene['bindings'].items():
        if key.startswith('container_'):
            assert key not in json.dumps(public)
        assert binding['name'] not in json.dumps(public)
    assert task['requirements']['capabilities'] == list(CAPABILITIES)
    assert set(CAPABILITIES) == {
        'base_translation', 'base_rotation', 'ee_translation',
        'ee_rotation', 'trunk_height', 'gripper'}


def test_generation_is_deterministic_and_check_compares_bytes_without_writing(tmp_path):
    before = {path: path.read_bytes() for path in (PARENT_TASK, PARENT_SCENE)}
    expected = render_definitions()
    assert render_definitions() == expected
    assert DEFAULT_OUTPUT.is_absolute()
    assert main(['--check']) == 0
    assert main(['--output-dir', str(tmp_path)]) == 0
    assert {name: (tmp_path / name).read_bytes() for name in expected} == expected
    mtimes = {name: (tmp_path / name).stat().st_mtime_ns for name in expected}
    assert main(['--output-dir', str(tmp_path), '--check']) == 0
    assert {name: (tmp_path / name).stat().st_mtime_ns for name in expected} == mtimes
    stale = tmp_path / TASK_FILENAME
    stale.write_bytes(stale.read_bytes() + b'\n')
    stale_bytes = stale.read_bytes()
    assert main(['--output-dir', str(tmp_path), '--check']) == 1
    assert stale.read_bytes() == stale_bytes
    assert {path: path.read_bytes() for path in before} == before

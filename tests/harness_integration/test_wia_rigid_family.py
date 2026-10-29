"""Validate pending Rigid family overlays without simulation or model calls.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_rigid_family.py
The builder's --check compares committed/generated definitions without writing.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, RigidContainmentTaskDefinition, SceneDefinition
import wia.authoring.commands.build_wia_rigid_family as builder


@pytest.mark.parametrize('alias', list(builder.CASES))
def test_pending_tasks_and_scenes_validate_and_preserve_native_configuration(alias):
    task, scene = builder.build_definitions(alias)
    parent_task_path, parent_scene_path = builder.parent_paths(alias)
    parent_task = yaml.safe_load(parent_task_path.read_text())
    parent_scene = yaml.safe_load(parent_scene_path.read_text())
    validated = CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    assert task['status'] == scene['status'] == 'pending'
    assert task['kind'] == 'category_robot_task'
    assert task['id'] != parent_task['id'] and scene['id'] != parent_scene['id']
    assert task['source'] == parent_task['source'] and scene['source'] == parent_scene['source']
    assert task['private']['bindings'] == parent_scene['bindings']
    assert task['scene'] == scene['scene'] == parent_scene['scene']
    restored = deepcopy(scene)
    for key in ('id', 'version', 'status'):
        restored[key] = parent_scene[key]
    assert restored == parent_scene
    expected_defaults = RigidContainmentTaskDefinition.model_validate(parent_task).defaults.model_dump()
    assert validated.defaults.model_dump() == {**expected_defaults, 'action_mode': 'parameterized'}
    assert validated.defaults.seed == 0 and validated.defaults.max_steps == 60
    assert task['qualification']['support'] is None
    assert task['requirements']['capabilities'] == list(builder.CAPABILITIES)
    assert task['repair']['parameters_provisional'] is True
    assert task['repair']['parameters_calibrated'] is False
    assert task['repair']['native_validation_run'] is False
    assert task['repair']['physically_qualified'] is False
    assert task['repair']['legal_robot_reference_verified'] is False


@pytest.mark.parametrize('alias', list(builder.CASES))
def test_explicit_qa_maps_and_supports_match_selected_source_identities(alias):
    task, scene = builder.build_definitions(alias)
    parent_task = yaml.safe_load(builder.parent_paths(alias)[0].read_text())
    params = task['private']['evaluation']['parameters']
    case = builder.CASES[alias]
    evaluation = task['private']['evaluation']
    assert evaluation['type'] == 'category_operation'
    assert evaluation['module'] == 'wia.authoring.converters.rigid_partial_insertion'
    assert evaluation['operation'] == 'pick_release_partial_insert'
    assert evaluation['binding_keys'] == list(scene['bindings'])
    assert params['targets'] == parent_task['targets'] == list(case['target_labels'])
    assert params['expected_containers'] == case['expected_containers']
    assert params['expected_answers'] == case['expected_answers']
    assert params['target_labels'] == case['target_labels']
    assert params['allow_remove_after_verification'] is True
    assert {key: params[key] for key in builder.PHYSICAL_PARAMETERS} == builder.PHYSICAL_PARAMETERS
    qualification = scene['qualification']
    expected_supports = {key: qualification['object_supports'].get(key, qualification['support'])
                        for key in params['targets'] + params['containers']}
    assert params['initial_supports'] == expected_supports
    assert params['initial_support_tolerance_m'] == qualification['support_tolerance_m'] == .015
    assert params['maximum_initial_linear_speed_m_s'] == qualification['maximum_linear_speed_m_s'] == .02
    assert params['maximum_initial_angular_speed_rad_s'] == qualification['maximum_angular_speed_rad_s'] == .1
    provenance = task['repair']['original_qa_provenance']
    original = json.loads(Path(provenance['question_source_file']).read_text())
    assert provenance['original_question'] == original['question']
    assert provenance['original_answer'] == original['answer']
    assert provenance['verified_answer_by_public_label'] == {
        label: case['expected_answers'][key] for key, label in case['target_labels'].items()}


def test_t23_keeps_distinct_question_and_multitarget_scene_provenance():
    task, scene = builder.build_definitions('T23')
    assert '__criterion_4.json' in task['source']['json']
    assert '__criterion_4.json' not in scene['source']['json']
    assert '__multi_extra1_extra2.yaml' in task['repair']['parent_scene_file']
    assert scene['target_bindings'] == ['target', 'extra1', 'extra2']
    params = task['private']['evaluation']['parameters']
    assert params['initial_supports'] == {
        'target': 'pickup_support', 'extra1': 'multi_support_extra1', 'extra2': 'multi_support_extra2',
        'container_0': 'table', 'container_1': 'table', 'container_2': 'table'}
    assert params['expected_containers'] == {'target': 'container_1', 'extra1': 'container_2', 'extra2': 'container_2'}
    assert 'ANSWER: {"granola box": "<container label>", "spray can": "<container label>", "colored pencil": "<container label>"}' in task['public']['success_criteria']
    assert 'Simultaneous packing is not required.' in ' '.join(task['public']['constraints'])


def test_public_goals_preserve_original_questions_without_private_binding_or_answer_maps():
    for alias in builder.CASES:
        task, scene = builder.build_definitions(alias)
        original = yaml.safe_load(builder.parent_paths(alias)[0].read_text())
        assert task['public']['objective'].startswith(original['public']['objective'])
        public = json.dumps(task['public'])
        for binding in scene['bindings'].values():
            assert binding['name'] not in public
        assert all(key not in public for key in ('expected_answers', 'expected_containers', 'container_1'))
        criteria = task['public']['success_criteria']
        for phrase in ('actual grasp', 'at least 5 cm', '0.5 seconds', '0.02 m/s',
                       '0.1 rad/s', 'no robot contact', 'entirely on top', 'working gripper open'):
            assert phrase in criteria
        if alias != 'T23':
            assert 'ANSWER: <container label>' in criteria


def test_source_qa_conflicts_are_rejected_before_generation(monkeypatch):
    incorrect = deepcopy(builder.CASES)
    incorrect['T22']['expected_answers']['target'] = 'small packing box'
    monkeypatch.setattr(builder, 'CASES', incorrect)
    with pytest.raises(ValueError, match='assigned answers differ'):
        builder.build_definitions('T22')


def test_serialized_scenes_change_only_three_metadata_lines_and_manifest_is_explicit():
    rendered = builder.render_definitions()
    manifest = json.loads(rendered[Path('manifest.json')])
    assert len(manifest['tasks']) == 3
    assert manifest['parameters_provisional'] and not manifest['parameters_calibrated']
    assert not manifest['native_validation_run'] and not manifest['physical_success_verified']
    for record in manifest['tasks']:
        alias = record['document_alias']
        parent_task, parent_scene = builder.parent_paths(alias)
        task_file, scene_file = builder.filenames(alias)
        original = parent_scene.read_bytes().splitlines(keepends=True)
        clone = rendered[scene_file].splitlines(keepends=True)
        assert len(original) == len(clone)
        assert [line for line in original if not line.startswith((b'id:', b'version:', b'status:'))] == [
            line for line in clone if not line.startswith((b'id:', b'version:', b'status:'))]
        for field in ('task_file', 'scene_file', 'parent_task_file', 'parent_scene_file'):
            assert Path(record[field]).is_absolute()
        assert record['parent_task_file'] == str(parent_task) and record['parent_scene_file'] == str(parent_scene)
        assert record['task_sha256'] == hashlib.sha256(rendered[task_file]).hexdigest()
        assert record['scene_sha256'] == hashlib.sha256(rendered[scene_file]).hexdigest()


def test_builder_and_check_are_deterministic_and_never_overwrite_differing_outputs(tmp_path):
    originals = {path: path.read_bytes() for alias in builder.CASES for path in builder.parent_paths(alias)}
    assert builder.main(['--check']) == 0
    assert builder.main(['--output-dir', str(tmp_path)]) == 0
    rendered = builder.render_definitions(output_dir=tmp_path)
    assert {name: (tmp_path / name).read_bytes() for name in rendered} == rendered
    mtimes = {name: (tmp_path / name).stat().st_mtime_ns for name in rendered}
    assert builder.main(['--output-dir', str(tmp_path), '--check']) == 0
    assert builder.main(['--output-dir', str(tmp_path)]) == 0
    assert {name: (tmp_path / name).stat().st_mtime_ns for name in rendered} == mtimes
    task_name, scene_name = builder.filenames('T23')
    loaded_task = load_task(tmp_path / task_name, variant='tabletop', run_mode='diagnostic')
    loaded_scene = load_scene(tmp_path / scene_name, variant='tabletop', run_mode='diagnostic')
    assert loaded_task.source_path != loaded_scene.source_path
    assert (loaded_task.scene_name, loaded_task.room_name) == (loaded_scene.scene_name, loaded_scene.room_name)
    changed = tmp_path / task_name
    changed.write_bytes(changed.read_bytes() + b'\n')
    changed_bytes = changed.read_bytes()
    assert builder.main(['--output-dir', str(tmp_path), '--check']) == 1
    assert builder.main(['--output-dir', str(tmp_path)]) == 1
    assert changed.read_bytes() == changed_bytes
    assert {path: path.read_bytes() for path in originals} == originals

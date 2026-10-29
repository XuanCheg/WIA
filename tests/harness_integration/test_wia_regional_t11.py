"""Verify the neutral T11 option revision with CPU-only contract fixtures.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_regional_t11.py
Synthetic evaluator receipts do not establish native qualification or execution.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
import yaml

from wia.authoring.converters.cognitive_mapping_visit_reports import SourceAnswerMappingEvaluator
from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.authoring.commands.build_wia_regional_t11 import (
    ANCHOR_INSTRUCTION, DEFAULT_OUTPUT, NEUTRAL_OPTIONS, PARENT_SCENE, PARENT_TASK,
    QUESTION, REVISION, ROOT, SCENE_ID, TASK_ID, build_package, main,
)


@pytest.fixture(scope='module')
def package():
    return build_package()


def test_schema_and_production_loaders_keep_pending_revision_and_source_world(package):
    task, scene, _, _ = package
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    assert (task['id'], task['version'], task['status']) == (TASK_ID, 1, 'pending')
    assert (scene['id'], scene['version'], scene['status']) == (SCENE_ID, 1, 'pending')
    loaded_task = load_task(DEFAULT_OUTPUT / 'task.yaml', variant='adapted', run_mode='diagnostic')
    loaded_scene = load_scene(DEFAULT_OUTPUT / 'scene.yaml', variant='adapted', run_mode='diagnostic')
    assert loaded_task.source_path == loaded_scene.source_path
    assert loaded_task.definition.scene == loaded_scene.definition.scene
    assert (loaded_task.scene_name, loaded_task.room_name, loaded_task.seed) == ('school_gym', 'full_scene', 0)
    assert loaded_task.public_payload['objective'] == task['public']['objective']
    assert 'original_source_qa' not in json.dumps(loaded_task.public_payload)
    assert 'original_to_neutral_option_labels' not in json.dumps(loaded_task.public_payload)
    loaded_task.verify_unchanged()
    loaded_scene.verify_unchanged()


def test_four_neutral_choices_form_the_same_region_and_answer_bijection(package):
    task, _, manifest, _ = package
    parent = yaml.safe_load(PARENT_TASK.read_text())
    qa = task['private']['evaluation']['parameters']['source_qa']
    original = parent['private']['evaluation']['parameters']['source_qa']
    mapping = manifest['original_to_neutral_option_labels']
    assert qa['question'] == QUESTION
    assert qa['options'] == list(NEUTRAL_OPTIONS)
    assert len(mapping) == len(set(mapping.values())) == 4
    assert list(mapping) == original['options']
    assert list(mapping.values()) == qa['options']
    assert {value: key for key, value in mapping.items()} == manifest['neutral_to_original_option_labels']
    assert mapping[original['answer']] == qa['answer'] == 'locker room 1'
    assert qa['options'].index(qa['answer']) == original['options'].index(original['answer'])
    assert task['private']['evaluation']['parameters']['target_regions'] == ['locker_room_1']
    for label in qa['options']:
        assert not any(term in label for term in ('paper', 'dispenser', 'sink', 'light', 'bench', 'with', '('))
    assert task['public']['objective'] == QUESTION + ' Choose from: ' + ', '.join(NEUTRAL_OPTIONS) + '. ' + ANCHOR_INSTRUCTION
    assert 'ANSWER: <room label>' in task['public']['success_criteria']
    assert 'Each queried anchor must be identified in delivered robot-camera observations.' in task['public']['success_criteria']
    assert 'Room occupancy alone does not establish the anchor relation; no extra room tour is required.' in task['public']['success_criteria']


def test_exact_definition_diff_preserves_geometry_all_physical_parameters_and_scorer(package):
    task, scene, manifest, rendered = package
    parent_task = yaml.safe_load(PARENT_TASK.read_text())
    parent_scene = yaml.safe_load(PARENT_SCENE.read_text())
    restored = deepcopy(task)
    for key in ('id', 'version', 'status'):
        restored[key] = parent_task[key]
    for key in ('objective', 'success_criteria'):
        restored['public'][key] = parent_task['public'][key]
    restored['private']['evaluation']['parameters']['source_qa'] = deepcopy(
        parent_task['private']['evaluation']['parameters']['source_qa'])
    restored.pop('repair')
    assert restored == parent_task
    restored_scene = deepcopy(scene)
    for key in ('id', 'version', 'status'):
        restored_scene[key] = parent_scene[key]
    assert restored_scene == parent_scene
    assert task['scene'] == scene['scene'] == parent_task['scene']
    raw, clone = PARENT_SCENE.read_bytes().splitlines(keepends=True), rendered[next(path for path in rendered if path.name == 'scene.yaml')].splitlines(keepends=True)
    assert len(raw) == len(clone)
    assert [line for line in raw if not line.startswith((b'id:', b'version:', b'status:'))] == [
        line for line in clone if not line.startswith((b'id:', b'version:', b'status:'))]
    assert manifest['geometry_modified'] is manifest['scorer_modified'] is manifest['source_json_modified'] is False
    assert manifest['native_qualification'] == manifest['anchor_observation_validation'] == manifest['reference_execution'] == 'PENDING'
    repair = task['repair']['regional_t11_neutral_labels']
    assert repair['original_source_qa'] == parent_task['private']['evaluation']['parameters']['source_qa']
    assert repair['validation_status'] == 'pending-native-validation'


def synthetic_evaluator(task, output, *, anchor_observed=True, qualified=True):
    config = deepcopy(task['private']['evaluation'])
    config['bindings'] = {key: binding['name'] for key, binding in task['private']['bindings'].items()}
    source_path = config['parameters']['source_path']
    receipt = {'binding': 'anchor_a', 'episode_id': 'synthetic_episode',
        'native_object_name': config['bindings']['anchor_a'], 'observable_identity_verified': True,
        'public_label_association_verified': True, 'delivered_rgb_verified': True, 'snapshot_id': 7}
    session = NS(episode_id='synthetic_episode', mapping_anchor_observations={source_path: [receipt] if anchor_observed else []},
        mapping_qualification={source_path: {'episode_id': 'synthetic_episode',
            'source_runtime_door_policy_reconciled': True, 'condition_identity_verified': qualified,
            'source_answer_verified': True, 'relevant_door_ids': []}})
    native = NS(begin_episode=lambda observation: None,
        record=lambda observation, action=None: {'evaluation_validity': 'complete', 'task_success': False,
            'constraints_satisfied': True, 'completion_items': [], 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step})
    evaluator = SourceAnswerMappingEvaluator(session, None, config, output, native)
    observation = NS(snapshot_id=7, action_step=0)
    evaluator.begin_episode(observation)
    return evaluator, observation


def test_existing_exact_answer_scorer_accepts_neutral_alias_and_rejects_old_descriptive_answer(package, tmp_path):
    task, _, manifest, _ = package
    evaluator, observation = synthetic_evaluator(task, tmp_path)
    evaluator.report_answer('  LOCKER   room 1  ', observation)
    row = evaluator.record(observation)
    assert row['answer_correct'] and row['task_success']
    assert row['mapping_qualification']['accepted']
    evaluator.report_answer(manifest['original_source_qa']['answer'], observation)
    row = evaluator.record(observation)
    assert not row['answer_correct'] and not row['task_success']


@pytest.mark.parametrize('missing', ['anchor_observation', 'native_qualification'])
def test_neutral_answer_still_requires_existing_native_and_delivered_anchor_gates(package, tmp_path, missing):
    task = package[0]
    evaluator, observation = synthetic_evaluator(task, tmp_path,
        anchor_observed=missing != 'anchor_observation', qualified=missing != 'native_qualification')
    evaluator.report_answer('locker room 1', observation)
    row = evaluator.record(observation)
    assert row['answer_correct'] and not row['task_success']
    if missing == 'anchor_observation':
        assert not next(item for item in row['completion_items'] if item['id'] == 'anchor:anchor_a')['completed']
    else:
        assert row['evaluation_validity'] == 'unavailable'
        assert not row['mapping_qualification']['accepted']


def test_deterministic_generation_check_and_parent_source_immutability(package, tmp_path):
    _, _, manifest, rendered = package
    inputs = (PARENT_TASK, PARENT_SCENE, ROOT / manifest['source_file'])
    before = {path: path.read_bytes() for path in inputs}
    assert DEFAULT_OUTPUT.is_absolute()
    assert build_package()[3] == rendered
    assert manifest['schema'] == REVISION
    for kind, path in (('parent_task', PARENT_TASK), ('parent_scene', PARENT_SCENE), ('source', inputs[2])):
        assert manifest[kind + '_sha256'] == hashlib.sha256(before[path]).hexdigest()
    assert main(['--check']) == 0
    assert main(['--output-dir', str(tmp_path)]) == 0
    assert {name: (tmp_path / name).read_bytes() for name in rendered} == rendered
    mtimes = {name: (tmp_path / name).stat().st_mtime_ns for name in rendered}
    assert main(['--output-dir', str(tmp_path), '--check']) == 0
    assert main(['--output-dir', str(tmp_path)]) == 0
    assert {name: (tmp_path / name).stat().st_mtime_ns for name in rendered} == mtimes
    changed = tmp_path / 'task.yaml'
    changed.write_bytes(changed.read_bytes() + b'\n')
    changed_bytes = changed.read_bytes()
    assert main(['--output-dir', str(tmp_path), '--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite a differing output'):
        main(['--output-dir', str(tmp_path)])
    assert changed.read_bytes() == changed_bytes
    assert {path: path.read_bytes() for path in inputs} == before


def test_differing_existing_file_blocks_all_package_writes(tmp_path):
    existing = tmp_path / 'scene.yaml'
    existing.write_text('id: unrelated_scene\n')
    before = existing.read_bytes()
    with pytest.raises(ValueError, match='Refusing to overwrite a differing output'):
        main(['--output-dir', str(tmp_path)])
    assert existing.read_bytes() == before
    assert not (tmp_path / 'task.yaml').exists()
    assert not (tmp_path / 'manifest.json').exists()

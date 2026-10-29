"""Check declared matched histories without claiming native observation equality.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  -m pytest -q tests/test_wia_temporal_pairs.py
"""
import json

import pytest
import yaml

from wia.tasks.loader import load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from wia.authoring.commands.build_wia_temporal_pairs import (
    AUTHORED_RELATIVE, BASE_SCENE, BASE_SOURCE, BASE_TASK, BUTTER_SOURCE,
    DEFAULT_OUTPUT_ROOT, ENCOUNTER_KEYS, PUBLIC_TASK_ID, REVISION, ROOT,
    SOURCE_RELATIVE, VARIANTS, authored_sources, build_package,
    declared_phase_2_state, main,
)


@pytest.fixture(scope='module')
def package():
    return build_package()


def content_categories(raw, phase):
    return [box[phase]['category'] if box[phase] else None
            for box in raw['metadata_json']['question_data']['boxes']]


def test_authored_sources_have_different_histories_and_identical_phase_two(package):
    sources = package[0]
    base = json.loads(BASE_SOURCE.read_text(encoding='utf-8'))
    original_boxes = base['metadata_json']['question_data']['boxes']
    assert {variant: content_categories(raw, 'phase1_content') for variant, raw in sources.items()} == {
        'a': ['shaker', 'jar_of_cumin', None],
        'b': ['butter', 'jar_of_cumin', None],
        'c': [None, 'jar_of_cumin', None],
    }
    for variant, raw in sources.items():
        metadata = raw['metadata_json']
        question = metadata['question_data']
        assert content_categories(raw, 'phase2_content') == [None, 'jar_of_cumin', None]
        assert raw['id'] == metadata['question_id'] == f'wia_history_{variant}_v1'
        assert raw['question'] == question['question'] == base['question']
        assert json.loads(raw['options_json']) == question['options'] == ['yes', 'no']
        assert raw['answer'] == question['answer'] == ('no' if variant == 'c' else 'yes')
        assert question['box_count'] == len(question['boxes']) == 3
        assert metadata['seed'] == 0
        assert metadata['authored_revision']['revision'] == REVISION
        assert metadata['authored_revision']['event_branch'] == (
            'no_change' if variant == 'c' else 'removed')
        assert question['boxes'][0]['change_type'] == ('no_change' if variant == 'c' else 'remove')
        assert question['render'] == base['metadata_json']['question_data']['render']
        assert [box['container'] for box in question['boxes']] == [box['container'] for box in original_boxes]
        assert [box['position_label'] for box in question['boxes']] == [
            box['position_label'] for box in original_boxes]
        assert [box['phase2_content'] for box in question['boxes']] == [
            box['phase2_content'] for box in original_boxes]
    assert sources['b']['metadata_json']['question_data']['boxes'][0]['phase1_content']['model'] == 'ovirmc'


def test_existing_converter_selects_the_history_bound_spare_or_no_manipulation(package):
    _, definitions, manifest, _ = package
    for variant, (task, scene) in definitions.items():
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        evaluation = task['private']['evaluation']
        p = evaluation['parameters']
        assert task['status'] == scene['status'] == 'pending'
        assert task['defaults'] == {'seed': 0}
        assert task['scene'] == scene['scene']
        assert task['private']['bindings'] == scene['bindings']
        assert evaluation['module'] == 'wia.authoring.converters.temporal_understanding_two_phase'
        assert evaluation['operation'] == 'compare_change_and_restore_or_collect'
        target = {'a': 'public_spare_1', 'b': 'public_spare_3', 'c': None}[variant]
        assert p['physical_target_key'] == target
        if variant == 'c':
            assert p['physical_branch'] == p['source_event_recipe']['branch'] == 'no_change'
            assert p['physical_destination_key'] is p['placement_relation'] is None
            assert p['source_event_recipe']['changed_containers'] == []
            assert 'image1_content_00' not in p['phase_keys']['image1']
        else:
            assert p['physical_branch'] == p['source_event_recipe']['branch'] == 'removed'
            assert p['physical_destination_key'] == 'container_00'
            assert p['placement_relation'] == 'inside'
            expected_category = {'a': 'shaker', 'b': 'butter'}[variant]
            assert task['private']['bindings'][target]['category'] == expected_category
            assert p['identities']['image1_content_00']['category'] == expected_category
            assert p['source_event_recipe']['changed_containers'] == ['container_00']
    assert [row['physical_target_key'] for row in manifest['instances']] == [
        'public_spare_1', 'public_spare_3', None]


def test_common_public_payload_has_no_history_variant_or_expected_binding(package):
    _, definitions, manifest, _ = package
    baseline = yaml.safe_load(BASE_TASK.read_text(encoding='utf-8'))
    for task, _ in definitions.values():
        assert task['public'] == baseline['public'] == manifest['common_public_task_section']
        public = json.dumps(task['public'])
        assert 'wia_history_' not in public
        assert 'physical_target_key' not in public
        assert 'public_spare_' not in public
        assert 'container_00' not in public
        assert 'Original answer options: [\\"yes\\", \\"no\\"]' in public
        assert 'No grasp is required for no-change.' in task['public']['objective']
    payloads = [load_task(ROOT / row['task_file'], variant='adapted', run_mode='diagnostic').public_payload
                for row in manifest['instances']]
    assert payloads[0] == payloads[1] == payloads[2] == manifest['common_public_payload']
    assert payloads[0]['task_id'] == PUBLIC_TASK_ID
    assert payloads[0]['task_version'] == 1
    assert 'source' not in payloads[0]
    assert 'wia_history_' not in json.dumps(payloads[0])
    assert SOURCE_RELATIVE.as_posix() not in json.dumps(payloads[0])


def test_common_phase_two_state_preserves_inventory_apparatus_cameras_and_encounters(package):
    sources, definitions, manifest, _ = package
    baseline = yaml.safe_load(BASE_TASK.read_text(encoding='utf-8'))
    baseline_scene = yaml.safe_load(BASE_SCENE.read_text(encoding='utf-8'))
    baseline_p = baseline['private']['evaluation']['parameters']
    states = [declared_phase_2_state(sources[key], *definitions[key]) for key in VARIANTS]
    assert states[0] == states[1] == states[2] == manifest['common_declared_active_phase_2_state']
    assert states[0]['apparatus'] == baseline_p['apparatus']
    assert states[0]['public_spare_inventory'] == baseline_p['public_spare_inventory']
    assert len(states[0]['public_spare_inventory']) == 4
    assert {row['name'] for row in states[0]['apparatus']['dataset_objects']} == {
        'public_spare_1', 'public_spare_2', 'public_spare_3', 'public_spare_4'}
    assert len(states[0]['boxes']) == 3
    assert [row['identity']['category'] for row in states[0]['active_contents']] == ['jar_of_cumin']
    for task, scene in definitions.values():
        p = task['private']['evaluation']['parameters']
        assert {key: p[key] for key in ENCOUNTER_KEYS} == {
            key: baseline_p[key] for key in ENCOUNTER_KEYS} == manifest['encounter_policy']
        assert p['survey_revision'] == 'free_survey_10_v1'
        assert p['phase_1_max_decisions'] == 10
        assert p['phase_1_allow_early_transition'] is True
        assert p['require_complete_phase_1_coverage'] is False
        assert SceneDefinition.model_validate(scene).runtime == SceneDefinition.model_validate(baseline_scene).runtime
        assert scene['scene']['variants']['original']['patches'] == []


def test_manifest_retains_history_available_input_and_all_native_gates_pending(package):
    _, definitions, manifest, _ = package
    assert manifest['public_payload_equal'] is True
    assert manifest['declared_active_phase_2_state_equal'] is True
    assert manifest['phase_1_histories_differ'] is True
    assert manifest['expected_physical_targets_differ'] is True
    assert manifest['native_experiments_run'] is False
    assert manifest['native_phase_2_equivalence'] == 'PENDING'
    assert manifest['observation_shortcut_audit'] == 'PENDING'
    assert manifest['reference_execution'] == 'PENDING'
    assert manifest['observation_contract']['frozen_initial_overview_retained'] is True
    assert manifest['observation_contract']['full_input_history_status'] == 'history_available'
    assert manifest['observation_contract']['memory_free_claim'] is False
    for task, _ in definitions.values():
        assert task['repair']['validation_status'] == 'pending-native-validation'
        assert task['repair']['native_validation_run'] is False
    assert all(set(row['native_gates'].values()) == {'pending'} for row in manifest['instances'])


def test_rebuild_and_check_are_deterministic_without_changing_historical_inputs(package, tmp_path):
    inputs = (BASE_SOURCE, BUTTER_SOURCE, BASE_TASK, BASE_SCENE)
    before = {path: path.read_bytes() for path in inputs}
    expected = package[3]
    assert DEFAULT_OUTPUT_ROOT.is_absolute()
    assert build_package()[3] == expected
    assert authored_sources() == package[0]
    assert main(['--check']) == 0
    assert main(['--output-root', str(tmp_path)]) == 0
    assert len(expected) == 10
    mtimes = {name: (tmp_path / name).stat().st_mtime_ns for name in expected}
    assert main(['--output-root', str(tmp_path), '--check']) == 0
    assert {name: (tmp_path / name).read_bytes() for name in expected} == expected
    assert {name: (tmp_path / name).stat().st_mtime_ns for name in expected} == mtimes
    source_file = tmp_path / AUTHORED_RELATIVE / SOURCE_RELATIVE / 'wia_history_b_v1.json'
    source_file.write_bytes(source_file.read_bytes() + b'\n')
    stale = source_file.read_bytes()
    assert main(['--output-root', str(tmp_path), '--check']) == 1
    assert source_file.read_bytes() == stale
    assert {path: path.read_bytes() for path in before} == before

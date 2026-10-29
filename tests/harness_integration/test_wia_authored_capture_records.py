"""Check the four authored capture records without publishing or queue mutation.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_authored_capture_records.py
"""
from copy import deepcopy
from pathlib import Path
import re

import pytest

import wia.runtime.esi_category_capture as bridge
from wia.runtime.scene_capture_queue import digest, read_json, sha256
from wia.tasks.loader import load_scene, load_task
from wia.tasks.sources import resolve_source
import wia.authoring.commands.build_wia_authored_capture_records as builder


@pytest.fixture(scope='module')
def package():
    return builder.build_records()


def test_exact_four_current_definitions_keep_required_native_variants_and_source_members(package):
    assert package['record_count'] == len(package['items']) == 4
    assert tuple(item['document_alias'] for item in package['items']) == builder.ALIASES
    assert {item['document_alias']: item['variant'] for item in package['items']} == {
        'T11': 'adapted', 'T22': 'tabletop', 'T23': 'tabletop', 'T24': 'tabletop'}
    for item in package['items']:
        record = item['record']
        task = load_task(item['task_file'], variant=item['variant'], run_mode='diagnostic')
        scene = load_scene(item['scene_file'], variant=item['variant'], run_mode='diagnostic')
        assert task.definition.kind == 'category_robot_task'
        assert task.definition.scene.variants[item['variant']] == scene.definition.scene.variants[item['variant']]
        assert task.seed == scene.seed == record['seed']
        assert item['variant_id'] == task.definition.id
        assert item['task_sha256'] == task.file_sha256 == sha256(item['task_file'])
        assert item['scene_sha256'] == scene.file_sha256 == sha256(item['scene_file'])
        assert item['source_sha256'] == scene.source_sha256 == record['source_sha256']
        assert item['task_source_sha256'] == task.source_sha256
        assert bridge._source_members(record) == record['capture_source_members']
        assert {member['source_identity'] for member in record['capture_source_members']} == {
            task.definition.source.json_path, scene.definition.source.json_path}
        for member in record['capture_source_members']:
            assert Path(member['source_file']) == resolve_source(member['source_identity'])
            assert member['source_sha256'] == sha256(member['source_file'])
    t23 = next(item['record'] for item in package['items'] if item['document_alias'] == 'T23')
    assert len(t23['capture_source_members']) == 2
    assert t23['task_source_identity'].endswith('__criterion_4.json')
    assert not t23['scene_source_identity'].endswith('__criterion_4.json')


def test_original_registry_selection_and_issue_provenance_are_separate_from_new_capture_identity(package):
    registry, corev2 = read_json(builder.REGISTRY), read_json(builder.COREV2)
    capture_ids = set()
    for item in package['items']:
        row = next(row for row in registry['rows'] if row['document_alias'] == item['document_alias'])
        original = next(entry for entry in corev2['tasks'] if entry['candidate_id'] == row['canonical_candidate_id'])
        record = item['record']
        assert item['row_id'] == item['wia_id'] == record['wia_id'] == row['wia_id']
        assert record['corev2_candidate_id'] == item['canonical_candidate_id'] == row['canonical_candidate_id']
        assert record['warning_codes'] == original['issues']
        assert (record['big_task'], record['small_task']) == (original['category'], original['subcategory'])
        key = digest(record['definition_identity'])[:24]
        assert item['capture_candidate_id'] == record['candidate_id'] == 'category_' + key
        assert re.fullmatch(r'category_[0-9a-f]{24}', record['candidate_id'])
        assert record['candidate_id'] != row['canonical_candidate_id']
        assert record['task_edit2_variant_key'] == key
        unsigned = deepcopy(record)
        unsigned.pop('revision')
        assert record['revision'] == digest(unsigned)
        capture_ids.add(record['candidate_id'])
    assert len(capture_ids) == 4


def test_records_request_full_bundle_without_claiming_a_capture_job_or_success(package):
    assert package['status'] == 'definition_only'
    assert package['published'] is package['queue_mutated'] is package['native_capture_run'] is package['success_claimed'] is False
    assert package['reference_execution_status'] == 'deferred_until_scene_review'
    for item in package['items']:
        record = item['record']
        assert 'job_id' not in item and 'job_id' not in record
        assert 'task_success' not in record
        assert record['capture_initial_agent_bundle'] is record['task_edit2_capture_campaign'] is True
        assert record['definition_only'] and record['evidence_scope'] == 'definition_only'
        assert record['success_claimed'] is record['native_validation_run'] is record['reference_only'] is False
        assert record['reference_execution_status'] == 'deferred_until_scene_review'
        assert record['reuse_level'] == 'preserved_original_layout_new_authored_task_contract'


def test_cpu_build_and_check_are_deterministic_and_do_not_publish_or_modify_definitions(package, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU record preparation must not publish, freeze or create a queue.')
    for name in ('publish_record', 'freeze_working_code', 'CaptureQueue'):
        monkeypatch.setattr(bridge, name, forbidden)
    paths = {builder.REGISTRY, builder.COREV2, *(Path(row['file']) for row in package['authored_manifests'])}
    for item in package['items']:
        paths.update(Path(item[key]) for key in ('task_file', 'scene_file'))
        paths.update(Path(member['source_file']) for member in item['record']['capture_source_members'])
    before = {path: path.read_bytes() for path in paths}
    output_before = builder.OUTPUT.read_bytes()
    mtime = builder.OUTPUT.stat().st_mtime_ns
    assert builder.build_records() == package
    assert builder.main(['--check']) == 0
    assert builder.main([]) == 0
    assert builder.OUTPUT.read_bytes() == output_before
    assert builder.OUTPUT.stat().st_mtime_ns == mtime
    assert {path: path.read_bytes() for path in paths} == before

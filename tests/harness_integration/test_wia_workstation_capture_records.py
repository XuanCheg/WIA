"""Verify 26 CPU-only workstation capture records without queue operations.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_workstation_capture_records.py
Future capture signature bindings are inspected without publishing real job IDs.
"""
from copy import deepcopy
from pathlib import Path
import re
import shutil

import pytest

import wia.runtime.esi_category_capture as bridge
from wia.runtime.scene_capture_queue import digest, read_json, sha256
from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import SceneRuntime
from wia.tasks.sources import resolve_source
import wia.authoring.commands.build_wia_workstation_capture_records as builder


@pytest.fixture(scope='module')
def package():
    return builder.build_records()


def test_exact_thirteen_aliases_two_seeds_and_twenty_six_unique_seed_version_identities(package):
    assert package['proposal_count'] == 13
    assert package['record_count'] == len(package['items']) == 26
    assert [(item['document_alias'], item['seed']) for item in package['items']] == [
        (alias, seed) for alias in builder.ALIASES for seed in (101, 202)]
    assert len({item['variant_id'] for item in package['items']}) == 26
    assert len({item['capture_candidate_id'] for item in package['items']}) == 26
    for item in package['items']:
        record = item['record']
        assert item['variant_id'] == Path(item['task_file']).stem == record['wia_variant_id']
        assert str(item['seed']) in item['variant_id'] and str(item['seed']) in item['label']
        key = digest(record['definition_identity'])[:24]
        assert record['candidate_id'] == item['capture_candidate_id'] == 'category_' + key
        assert re.fullmatch(r'category_[0-9a-f]{24}', record['candidate_id'])
        assert record['task_edit2_variant_key'] == key
        unsigned = deepcopy(record)
        unsigned.pop('revision')
        assert record['revision'] == digest(unsigned)


def test_exact_manifest_paths_hashes_source_roles_registry_taxonomy_and_loader_contracts(package):
    registry = read_json(builder.REGISTRY)
    declared = {}
    for manifest_path in builder.MANIFESTS:
        manifest = read_json(manifest_path)
        for entry in manifest['entries']:
            declared[(entry['proposal_alias'], entry['seed'])] = (manifest_path, manifest, entry)
    source = resolve_source(builder.SOURCE_IDENTITY)
    for item in package['items']:
        record = item['record']
        row = next(row for row in registry['rows'] if row['document_alias'] == item['document_alias'])
        manifest_path, manifest, declared_entry = declared[(item['document_alias'], item['seed'])]
        assert Path(item['task_file']) == (manifest_path.parent / declared_entry['task_file']).resolve()
        assert Path(item['scene_file']) == (manifest_path.parent / declared_entry['scene_file']).resolve()
        assert item['task_sha256'] == record['task_sha256'] == declared_entry['task_sha256'] == sha256(item['task_file'])
        assert item['scene_sha256'] == record['scene_sha256'] == declared_entry['scene_sha256'] == sha256(item['scene_file'])
        assert record['definition_identity']['authored_manifest_sha256'] == sha256(manifest_path)
        assert item['row_id'] == record['wia_id'] == row['wia_id']
        assert item['canonical_candidate_id'] == record['proposal_canonical_id'] == row['canonical_candidate_id']
        assert record['proposal_canonical_id'].startswith('plan_')
        assert row['task'] is None and row['scene']['selected_scene_file'] is None
        assert 'corev2_candidate_id' not in record and not record['has_prior_selected_definition']
        assert (record['primary_family'], record['primary_subclass']) == (row['primary_family'], row['primary_subclass'])
        assert (record['big_task'], record['small_task']) == ('Specular Reflection', 'Correspondence')
        assert record['source_category_is_task_family'] is False
        assert record['source_role'] == 'background_loader_only'
        assert record['source_identity'] == item['source_identity'] == builder.SOURCE_IDENTITY
        assert Path(record['original_source']) == Path(item['source_file']) == source
        assert record['source_sha256'] == item['source_sha256'] == sha256(source)
        assert bridge._source_members(record) == record['capture_source_members'] == [{
            'source_identity': builder.SOURCE_IDENTITY, 'source_file': str(source), 'source_sha256': sha256(source)}]
        task = load_task(item['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(item['scene_file'], variant='adapted', run_mode='diagnostic')
        assert task.definition.scene == scene.definition.scene
        assert task.definition.private.bindings == scene.definition.bindings
        assert SceneRuntime.model_validate(task.definition.repair['runtime']) == scene.definition.runtime
        assert task.seed == scene.seed == record['seed'] == declared_entry['seed']
        evaluation = task.definition.private.evaluation
        assert record['converter_module'] == evaluation.module
        assert record['converter_version'] == record['evaluation_revision'] == evaluation.parameters['revision'] == manifest['revision']
        assert record['robot_task_public'] == task.definition.public.model_dump(mode='json')


def test_matched_appearance_versions_have_distinct_capture_signature_bindings_despite_same_public_id(package):
    def campaign_binding(record):
        # This is the identity input used by publish_record, not a submitted job.
        return {'candidate_id': record['candidate_id'], 'task_edit2_variant_key': record['task_edit2_variant_key'],
            'source_members': sorted([{key: member[key] for key in ('source_identity', 'source_sha256')}
                for member in record['capture_source_members']], key=lambda member: member['source_identity']),
            'task_origin_file': str(Path(record['task_file']).resolve()),
            'scene_origin_file': str(Path(record['scene_file']).resolve())}
    for seed in (101, 202):
        pair = [next(item for item in package['items'] if item['document_alias'] == alias and item['seed'] == seed)
                for alias in ('R-A1', 'R-A2')]
        tasks = [load_task(item['task_file'], variant='adapted', run_mode='diagnostic') for item in pair]
        assert tasks[0].public_payload == tasks[1].public_payload
        assert tasks[0].definition.id == tasks[1].definition.id == 'wia_appearance_delivery_v1'
        bindings = [campaign_binding(item['record']) for item in pair]
        assert bindings[0] != bindings[1]
        assert digest({'task_edit2_capture_binding': bindings[0]}) != digest({'task_edit2_capture_binding': bindings[1]})
        assert pair[0]['variant_id'] != pair[1]['variant_id']


def test_definition_only_full_bundle_preparation_is_deterministic_read_only_and_never_publishes(package, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU capture-record preparation must not publish, freeze or instantiate a queue.')
    for name in ('publish_record', 'freeze_working_code', 'CaptureQueue'):
        monkeypatch.setattr(bridge, name, forbidden)
    assert package['status'] == 'definition_only'
    assert package['published'] is package['queue_mutated'] is package['native_capture_run'] is package['success_claimed'] is False
    assert package['scientific_controls_verified'] is False
    paths = {builder.REGISTRY, *builder.MANIFESTS}
    for item in package['items']:
        paths.update(Path(item[key]) for key in ('task_file', 'scene_file', 'source_file'))
        record = item['record']
        assert 'job_id' not in item and 'job_id' not in record and 'task_success' not in record
        assert record['capture_initial_agent_bundle'] is record['task_edit2_capture_campaign'] is True
        assert record['definition_only'] is True
        assert record['native_validation_run'] is record['scientific_controls_verified'] is record['success_claimed'] is False
        assert record['reference_execution_status'] == 'deferred_until_user_review'
    before = {path: path.read_bytes() for path in paths}
    output = builder.OUTPUT.read_bytes()
    mtime = builder.OUTPUT.stat().st_mtime_ns
    assert builder.build_records() == package
    assert builder.main(['--check']) == 0
    assert builder.main([]) == 0
    assert builder.OUTPUT.read_bytes() == output and builder.OUTPUT.stat().st_mtime_ns == mtime
    assert {path: path.read_bytes() for path in paths} == before


def test_changed_manifest_definition_hash_is_rejected_without_writing(monkeypatch):
    read_json_original = builder.read_json
    def changed_hash(path):
        value = read_json_original(path)
        if Path(path) == builder.MANIFESTS[0]:
            value = deepcopy(value)
            value['entries'][0]['task_sha256'] = '0' * 64
        return value
    output = builder.OUTPUT.read_bytes()
    monkeypatch.setattr(builder, 'read_json', changed_hash)
    with pytest.raises(ValueError, match='manifest definition hash mismatch'):
        builder.build_records()
    assert builder.OUTPUT.read_bytes() == output


def test_explicit_manifest_family_comes_from_schema_instead_of_directory_name(package, tmp_path):
    source_manifest = builder.MANIFESTS[0]
    relocated = tmp_path / 'a_directory_without_a_family_name' / 'manifest.json'
    relocated.parent.mkdir()
    shutil.copyfile(source_manifest, relocated)
    for entry in read_json(source_manifest)['entries']:
        for field in ('task_file', 'scene_file'):
            target = relocated.parent / entry[field]
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source_manifest.parent / entry[field], target)
    moved = builder.build_records((relocated, builder.MANIFESTS[1]))
    assert moved['record_count'] == 26
    assert [item['record'] for item in moved['items'] if item['document_alias'].startswith('R-')] == [
        item['record'] for item in package['items'] if item['document_alias'].startswith('R-')]
    assert {item['record']['converter_module'] for item in moved['items']
            if item['document_alias'].startswith('D')} == {'wia.authoring.converters.wia_dimensional'}


@pytest.mark.parametrize('changes,message', [
    ({'schema': 'unknown_manifest'}, 'supported family manifest'),
    ({'revision': 'wia_dimensional_v99'}, 'supported family manifest'),
    ({'revision': 'wia_mirror_v1'}, 'supported family manifest'),
    ({'revision': 'wia_dimensional_v2'}, 'revision must match its manifest'),
])
def test_unsupported_family_revision_or_task_revision_mismatch_is_rejected(monkeypatch, changes, message):
    original = builder.read_json
    def changed_manifest(path):
        value = original(path)
        if Path(path) == builder.MANIFESTS[0]:
            value = {**value, **changes}
        return value
    monkeypatch.setattr(builder, 'read_json', changed_manifest)
    with pytest.raises(ValueError, match=message):
        builder.build_records()


def test_duplicate_manifest_family_is_rejected():
    with pytest.raises(ValueError, match='exactly one dimensional manifest and one mirror manifest'):
        builder.build_records((builder.MANIFESTS[0], builder.MANIFESTS[0]))


@pytest.fixture(scope='module')
def v2_package():
    return builder.build_records((builder.WIA_ROOT / 'dimensional_v2/manifest.json', builder.MANIFESTS[1]))


def test_v2_dimensional_hash_and_stem_identities_are_distinct_while_mirror_records_are_unchanged(package, v2_package):
    assert v2_package['proposal_count'] == 13 and v2_package['record_count'] == 26
    assert [(item['document_alias'], item['seed']) for item in v2_package['items']] == [
        (alias, seed) for alias in builder.ALIASES for seed in (101, 202)]
    baseline = {(item['document_alias'], item['seed']): item for item in package['items']}
    v1_ids = {item['capture_candidate_id'] for item in package['items']}
    for item in v2_package['items']:
        previous = baseline[item['document_alias'], item['seed']]
        record = item['record']
        if item['document_alias'].startswith('R-'):
            assert item == previous
            continue
        assert item['variant_id'] == Path(item['task_file']).stem
        assert item['variant_id'].endswith('_v2') and item['variant_id'] != previous['variant_id']
        assert item['capture_candidate_id'] not in v1_ids
        assert item['task_sha256'] == sha256(item['task_file']) != previous['task_sha256']
        assert item['scene_sha256'] == sha256(item['scene_file']) != previous['scene_sha256']
        assert record['definition_identity']['authored_manifest_sha256'] == v2_package['authored_manifests'][0]['sha256']
        assert record['candidate_id'] == 'category_' + digest(record['definition_identity'])[:24]
        task = load_task(item['task_file'], variant='adapted', run_mode='diagnostic')
        assert record['converter_version'] == task.definition.private.evaluation.parameters['revision'] == 'wia_dimensional_v2'
        assert record['source_role'] == 'background_loader_only'
        assert record['source_category_is_task_family'] is record['has_prior_selected_definition'] is False
        assert record['proposal_canonical_id'] == previous['record']['proposal_canonical_id']
        assert record['definition_only'] is True and record['evidence_scope'] == 'definition_only'
        assert record['native_validation_run'] is record['scientific_controls_verified'] is record['success_claimed'] is False
        assert record['reference_execution_status'] == 'deferred_until_user_review'
    assert v2_package['status'] == 'definition_only'
    assert v2_package['published'] is v2_package['queue_mutated'] is v2_package['native_capture_run'] is v2_package['success_claimed'] is False


def test_explicit_v2_cli_output_and_no_write_check_leave_original_package_untouched(v2_package, tmp_path):
    original, mtime = builder.OUTPUT.read_bytes(), builder.OUTPUT.stat().st_mtime_ns
    output = tmp_path / 'v2.json'
    args = ['--dimensional-manifest', str(builder.WIA_ROOT / 'dimensional_v2/manifest.json'),
            '--mirror-manifest', str(builder.MANIFESTS[1]), '--output', str(output)]
    assert builder.main(args + ['--check']) == 1 and not output.exists()
    assert builder.main(args) == 0 and read_json(output) == v2_package
    output_bytes, output_mtime = output.read_bytes(), output.stat().st_mtime_ns
    assert builder.main(args + ['--check']) == 0
    assert output.read_bytes() == output_bytes and output.stat().st_mtime_ns == output_mtime
    assert builder.OUTPUT.read_bytes() == original and builder.OUTPUT.stat().st_mtime_ns == mtime

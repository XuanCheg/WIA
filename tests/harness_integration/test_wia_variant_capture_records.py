"""Validate generalized WIA records using production loaders, without capture.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_wia_variant_capture_records.py
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import wia.runtime.esi_category_capture as bridge
from wia.runtime.scene_capture_queue import digest, read_json, sha256
from wia.tasks.loader import load_scene, load_task
import wia.authoring.commands.build_wia_variant_capture_records as builder

WIA = builder.ROOT / 'explore_to_achieve/wia_v1'


@pytest.fixture
def manifest(tmp_path):
    registry = read_json(builder.REGISTRY)
    entries = []
    for alias, task, scene, variant in [
        ('T11', WIA / 'regional_t11_v1/task.yaml', WIA / 'regional_t11_v1/scene.yaml', 'adapted'),
        ('T23', WIA / 'rigid_family_v1/tasks/rigid_t23_v1.yaml',
         WIA / 'rigid_family_v1/scenes/rigid_t23_v1.yaml', 'tabletop'),
    ]:
        row = next(row for row in registry['rows'] if row['document_alias'] == alias)
        entries.append({'document_alias': alias, 'variant_id': alias.lower() + '_test_v1',
            'label': alias + ' authored test', 'variant': variant,
            'task_file': str(task.relative_to(builder.ROOT)), 'scene_file': str(scene),
            'parent_task_file': row['task']['task_file'],
            'parent_scene_file': row['scene']['selected_scene_file']})
    dimensional = read_json(WIA / 'dimensional_v2/manifest.json')['entries'][0]
    entries.append({'document_alias': dimensional['proposal_alias'], 'variant_id': 'dimensional_test_v2',
        'label': 'Authored dimensional test', 'variant': 'adapted',
        'task_file': str(WIA / 'dimensional_v2' / dimensional['task_file']),
        'scene_file': str(WIA / 'dimensional_v2' / dimensional['scene_file']),
        'parent_task_file': str(WIA / 'dimensional_v1' / dimensional['task_file'].replace('_v2', '_v1')),
        'parent_scene_file': str(WIA / 'dimensional_v1' / dimensional['scene_file'].replace('_v2', '_v1'))})
    path = tmp_path / 'manifest.json'
    _write(path, {'schema': builder.MANIFEST_SCHEMA, 'entries': entries})
    return path


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def test_production_loaders_source_roles_bundle_and_definition_only_provenance(manifest):
    package = builder.build_records(manifest)
    assert package['schema'] == builder.REVISION
    assert package['record_count'] == package['proposal_count'] == 3
    assert all(package[key] is False for key in ('published', 'queue_mutated', 'native_capture_run', 'success_claimed'))
    registry = {row['document_alias']: row for row in read_json(builder.REGISTRY)['rows']}
    for item in package['items']:
        record = item['record']
        row = registry[item['document_alias']]
        task = load_task(item['task_file'], variant=item['variant'], run_mode='diagnostic')
        scene = load_scene(item['scene_file'], variant=item['variant'], run_mode='diagnostic')
        assert task.seed == scene.seed == record['seed']
        assert item['task_sha256'] == task.file_sha256
        assert item['scene_sha256'] == scene.file_sha256
        assert record['proposal_canonical_id'] == row['canonical_candidate_id']
        assert (record['primary_family'], record['primary_subclass']) == (row['primary_family'], row['primary_subclass'])
        assert record['robot_task_public'] == task.definition.public.model_dump(mode='json')
        assert bridge._source_members(record) == record['capture_source_members']
        assert {member['source_identity'] for member in record['capture_source_members']} >= {
            task.definition.source.json_path, scene.definition.source.json_path}
        assert record['capture_initial_agent_bundle'] is record['task_edit2_capture_campaign'] is True
        assert record['native_validation_run'] is record['scientific_controls_verified'] is record['success_claimed'] is False
        assert record['definition_only'] is True and record['evidence_scope'] == 'definition_only'
        assert 'job_id' not in record and 'task_success' not in record
        assert record['candidate_id'] == 'category_' + digest(record['definition_identity'])[:24]
        unsigned = deepcopy(record)
        unsigned.pop('revision')
        assert record['revision'] == digest(unsigned)
    rigid = next(item['record'] for item in package['items'] if item['document_alias'] == 'T23')
    assert len(rigid['capture_source_members']) == 2
    dimensional = package['items'][-1]['record']
    assert dimensional['primary_family'] == 'GF'
    assert dimensional['big_task'] == 'Specular Reflection'
    assert dimensional['small_task'] == 'Correspondence'
    assert dimensional['source_category_is_task_family'] is False
    assert dimensional['has_prior_selected_definition'] is False


def test_evaluation_members_are_included_with_exact_source_hashes(manifest):
    value = read_json(manifest)
    entry = value['entries'][0]
    task = yaml.safe_load(builder._path(entry['task_file']).read_text())
    extra_identity = 'Specular Reflection/Correspondence/hall_glass_ceiling/empty_room_0/q_007.json'
    task['private']['evaluation']['parameters']['members'] = [{'source_identity': extra_identity,
        'evaluation': {'module': 'wia.authoring.converters.wia_mirror'}}]
    task_path = manifest.parent / 'task_with_member.yaml'
    task_path.write_text(yaml.safe_dump(task))
    entry['task_file'] = str(task_path)
    value['entries'] = [entry]
    _write(manifest, value)
    record = builder.build_records(manifest)['items'][0]['record']
    assert len(record['capture_source_members']) == 2
    member = next(member for member in record['capture_source_members'] if member['source_identity'] == extra_identity)
    assert member['source_sha256'] == sha256(member['source_file'])
    assert record['composite_member_modules'] == ['wia.authoring.converters.wia_mirror']
    task['private']['evaluation']['parameters']['members'][0]['source_sha256'] = '0' * 64
    task_path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError, match='hash mismatch'):
        builder.build_records(manifest)


def test_changed_background_keeps_original_registry_identity_and_actual_source_separate(manifest):
    value = read_json(manifest)
    legacy, authored = value['entries'][0], value['entries'][-1]
    authored.update({field: legacy[field] for field in ('document_alias', 'parent_task_file', 'parent_scene_file')})
    authored['variant_id'] = 'background_substitution_contract_test'
    value['entries'] = [authored]
    _write(manifest, value)
    item = builder.build_records(manifest)['items'][0]
    record = item['record']
    assert record['has_prior_selected_definition'] is True
    assert record['primary_family'] == 'NT'
    assert record['big_task'] == 'Specular Reflection'
    row = next(row for row in read_json(builder.REGISTRY)['rows'] if row['document_alias'] == 'T11')
    assert record['source_identity'] != row['source']['identity']
    assert record['registry_parent_task_file'] == legacy['parent_task_file']
    assert item['parent_task_sha256'] == sha256(builder._path(legacy['parent_task_file']))


@pytest.mark.parametrize('field', ['source_identity', 'source_file', 'module', 'revision', 'scene_model', 'room'])
def test_inaccurate_optional_loaded_provenance_is_rejected(manifest, field):
    value = read_json(manifest)
    value['entries'][0][field] = 'inaccurate_declared_value'
    _write(manifest, value)
    with pytest.raises(ValueError, match='does not match'):
        builder.build_records(manifest)


@pytest.mark.parametrize('field', ['task_sha256', 'scene_sha256', 'parent_task_sha256', 'parent_scene_sha256',
                                  'task_source_sha256', 'scene_source_sha256'])
def test_supplied_wrong_hashes_are_rejected(manifest, field):
    value = read_json(manifest)
    value['entries'][0][field] = '0' * 64
    _write(manifest, value)
    with pytest.raises(ValueError, match='hash mismatch'):
        builder.build_records(manifest)


@pytest.mark.parametrize('change', ['seed', 'bindings', 'patches', 'runtime'])
def test_task_scene_mismatch_is_rejected(manifest, change):
    value = read_json(manifest)
    entry = value['entries'][-1]
    scene = yaml.safe_load(builder._path(entry['scene_file']).read_text())
    if change == 'seed':
        scene['seed'] += 1
    elif change == 'bindings':
        scene['bindings'][next(iter(scene['bindings']))]['name'] += '_different'
    elif change == 'patches':
        scene['scene']['variants']['adapted']['patches'] = []
    else:
        scene['runtime']['camera_size'] = [256, 256]
    path = manifest.parent / 'mismatched_scene.yaml'
    path.write_text(yaml.safe_dump(scene))
    entry['scene_file'] = str(path)
    value['entries'] = [entry]
    _write(manifest, value)
    with pytest.raises(ValueError, match='must match'):
        builder.build_records(manifest)


def test_duplicate_alias_variant_rejected_and_version_labels_change_capture_identity(manifest):
    original = builder.build_records(manifest)['items'][0]
    value = read_json(manifest)
    value['entries'].append(deepcopy(value['entries'][0]))
    _write(manifest, value)
    with pytest.raises(ValueError, match='Duplicate'):
        builder.build_records(manifest)
    value['entries'][-1]['variant_id'] = 'second_explicit_version'
    _write(manifest, value)
    package = builder.build_records(manifest)
    assert len(package['items']) == 4
    assert package['items'][0]['capture_candidate_id'] == original['capture_candidate_id']
    assert package['items'][-1]['capture_candidate_id'] != original['capture_candidate_id']


def test_cli_determinism_read_only_check_and_collision_protection(manifest, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Definition-only builder cannot publish, freeze or create queues.')
    for name in ('publish_record', 'freeze_working_code', 'CaptureQueue'):
        monkeypatch.setattr(bridge, name, forbidden)
    output = manifest.parent / 'capture_records.json'
    args = ['--manifest', str(manifest), '--output', str(output)]
    assert builder.main(args + ['--check']) == 1 and not output.exists()
    assert builder.main(args) == 0
    before, mtime = output.read_bytes(), output.stat().st_mtime_ns
    assert builder.main(args + ['--check']) == builder.main(args) == 0
    assert output.read_bytes() == before and output.stat().st_mtime_ns == mtime
    value = read_json(manifest)
    value['entries'][0]['label'] += ' changed'
    _write(manifest, value)
    assert builder.main(args + ['--check']) == builder.main(args) == 1
    assert output.read_bytes() == before and output.stat().st_mtime_ns == mtime

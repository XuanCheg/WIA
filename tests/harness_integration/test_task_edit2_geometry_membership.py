"""Test exact Core Geometry membership and truthful CPU-only generation.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_geometry_membership.py
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
import wia.runtime.commands.build_task_edit2_geometry_candidates as builder


@pytest.fixture(scope='module')
def geometry_build(tmp_path_factory):
    output = tmp_path_factory.mktemp('geometry_core')
    baseline = json.loads(builder.CORE_MANIFEST.read_text())
    originals = [row for row in baseline['tasks'] if row['source'].startswith(builder.PREFIX)]
    before = {row['source']: builder.sha256(builder.SOURCE_ROOT / row['source']) for row in originals}
    manifest = builder.build(output)
    assert before == {row['source']: builder.sha256(builder.SOURCE_ROOT / row['source']) for row in originals}
    return manifest, output, originals


def test_all_four_exact_core_sources_have_explicit_derivative_mapping(geometry_build):
    manifest, output, originals = geometry_build
    assert len(originals) == manifest['source_count'] == len(manifest['records']) == 4
    assert manifest['core_membership']['membership_field'] == 'tasks/source'
    assert manifest['core_membership']['manifest_sha256'] == builder.sha256(builder.CORE_MANIFEST)
    assert [row['source_identity'] for row in manifest['records']] == [row['source'] for row in originals]
    assert len({row['source_identity'] for row in manifest['source_mapping']}) == 4
    historical = {row['source']: row for row in originals}
    for mapping in manifest['source_mapping']:
        original = historical[mapping['source_identity']]
        assert mapping['source_core_candidate_id'] == original['candidate_id']
        assert mapping['source_sha256'] == builder.sha256(mapping['source_file'])
        for kind in ('task', 'scene'):
            assert mapping['historical_' + kind + '_file'] == original[kind + '_file']
            assert mapping['historical_' + kind + '_sha256'] == builder.sha256(original[kind + '_file'])
            assert mapping[kind + '_sha256'] == builder.sha256(mapping[kind + '_file'])
    assert json.loads((output / 'batch_manifest.json').read_text()) == manifest
    assert json.loads((output / 'source_mapping.json').read_text()) == manifest['source_mapping']


def test_generated_definitions_keep_seeded_revision_source_assets_and_camera_policy(geometry_build):
    manifest, _, _ = geometry_build
    assert manifest['counts'] == {'generated': 4}
    assert manifest['pending_records'] == []
    assert manifest['definition_status_counts'] == {'schema_valid': 8}
    assert manifest['config'] == {'task_revision': builder.geometry.TASK_EDIT2_REVISION}
    assert manifest['seed'] == manifest['source_reset_seed'] == 0
    for record in manifest['records']:
        source = builder.SourceInstance(builder.SOURCE_ROOT / record['source_identity'], record['source_identity'],
            *builder.geometry.CategoryConverter.category)
        raw, _, objects, _, _ = builder.geometry._source_data(source)
        task = yaml.safe_load(Path(record['task_file']).read_text())
        scene = yaml.safe_load(Path(record['scene_file']).read_text())
        task_definition(task)
        SceneDefinition.model_validate(scene)
        evaluation = task['private']['evaluation']
        parameters = evaluation['parameters']
        assert task['source']['json'] == scene['source']['json'] == record['source_identity']
        assert evaluation['operation'] == builder.geometry.ARRANGEMENT_OPERATION
        assert parameters['task_revision'] == builder.geometry.TASK_EDIT2_REVISION
        assert parameters['initial_layout']['seed'] == 0
        assert parameters['initial_layout']['source_identity'] == record['source_identity']
        assert parameters['initial_layout']['version'] == builder.geometry.LAYOUT_VERSION
        assert parameters['target_shape'] == raw['answer']
        assert parameters['source_positions'] == [row['position'] for row in objects]
        assert parameters['source_quaternions'] == [row['quaternion_xyzw'] for row in objects]
        assert parameters['source_scales'] == [row.get('scale', 1.) for row in objects]
        assert parameters['source_assets'] == [{key: row[key] for key in ('category', 'model')} for row in objects]
        assert parameters['mechanism_contract']['initial_layout_policy'] == builder.geometry.LAYOUT_VERSION
        assert parameters['mechanism_contract']['already_satisfied_no_op_allowed'] is False
        assert scene['scene']['variants']['original']['patches'] == []
        assert record['interaction_policy']['allow_fallback_table'] is True
        assert record['interaction_policy']['relocation_policy'] == 'seeded_geometry'
        assert record['interaction_policy']['source_camera_policy'] == 'preserve'
        assert record['interaction_policy']['source_region'] == raw['room']
        assert 'already correct' not in task['public']['objective']


def test_schema_validity_is_independent_of_native_verification(geometry_build):
    manifest, _, _ = geometry_build
    assert manifest['native_experiments_run'] is False
    assert manifest['native_evidence_status'] == 'unavailable'
    assert manifest['final_selection_owner'] == 'user'
    for record in manifest['records']:
        assert record['generated_definition'] is True
        assert record['native_initialization_status'] == record['native_layout_status'] == 'unknown'
        assert record['native_execution_status'] == record['scoring_status'] == 'unavailable'
        assert record['autonomous_agent_performance'] is False
        assert all(status == 'pending' for status in record['native_gates'].values())
        assert record['native_gates']['seeded_initial_shape_false'] == 'pending'
        assert record['source_provenance_status'] == 'verified_current_source_bytes'


def test_repeated_build_is_deterministic_without_source_or_definition_edits(geometry_build):
    first, output, _ = geometry_build
    files = {row[kind + '_file']: builder.sha256(row[kind + '_file'])
             for row in first['records'] for kind in ('task', 'scene')}
    second = builder.build(output)
    assert [row['candidate_id'] for row in first['records']] == [row['candidate_id'] for row in second['records']]
    assert files == {path: builder.sha256(path) for path in files}
    assert first['source_mapping'] == second['source_mapping']


@pytest.mark.parametrize('converter_status,expected', [
    ('conversion_failed', 'pending_source_conversion'), ('generated', 'pending_definition_validation')])
def test_converter_or_definition_failures_stay_explicitly_pending(tmp_path, monkeypatch, converter_status, expected):
    calls = []

    def fake_convert_all(self, sources, context):
        calls.append((sources, context))
        return {'schema': 'esi_category_batch_v2', 'source_count': len(sources), 'seed': context.seed,
            'config': context.config, 'records': [{'source_identity': source.relative_path,
                'source_sha256': builder.sha256(source.path), 'candidate_id': 'fixture_' + str(index),
                'status': converter_status, 'failure_reason': 'CPU fixture failure', 'failure_stage': 'conversion',
                'task_file': None, 'scene_file': None} for index, source in enumerate(sources)]}

    monkeypatch.setattr(builder.geometry.CategoryConverter, 'convert_all', fake_convert_all)
    baseline = json.loads(builder.CORE_MANIFEST.read_text())
    # Capture jobs and protected Rigid rows are not a replacement for tasks/source.
    protected = next(deepcopy(row) for row in baseline['tasks'] if '/Rigid Containment/' in row['source'])
    protected['protected_baseline'] = True
    baseline['tasks'].append(protected)
    baseline['capture_jobs'] = [{'source': builder.PREFIX + 'unselected/room/not_core.json'}]
    manifest_path = tmp_path / 'core_membership.json'
    manifest_path.write_text(json.dumps(baseline))
    manifest = builder.build(tmp_path / 'pending', core_manifest=manifest_path)
    assert len(calls) == 1
    sources, context = calls[0]
    assert len(sources) == 4
    assert context.seed == 0
    assert context.config == {'task_revision': builder.geometry.TASK_EDIT2_REVISION}
    assert all(source.relative_path.startswith(builder.PREFIX) for source in sources)
    assert manifest['counts'] == {expected: 4}
    assert len(manifest['pending_records']) == 4
    assert all(row['converter_status'] == converter_status for row in manifest['records'])
    assert all(row['generated_definition'] is False for row in manifest['pending_records'])
    assert all('task_file' not in row and 'scene_file' not in row for row in manifest['pending_records'])
    assert manifest['definition_status_counts'] == {'unavailable': 8}


def test_duplicate_core_source_membership_is_not_silently_merged(tmp_path):
    baseline = json.loads(builder.CORE_MANIFEST.read_text())
    original = next(row for row in baseline['tasks'] if row['source'].startswith(builder.PREFIX))
    path = tmp_path / 'duplicate_core.json'
    path.write_text(json.dumps({'tasks': [original, original]}))
    with pytest.raises(ValueError, match='must be unique'):
        builder.build(tmp_path / 'unused', core_manifest=path)

"""Check active review exclusions while retaining archived 61+19 capture inputs.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
  tests/harness_integration/test_publish_wia_final_captures.py
Only temporary definitions, queue-shaped status files and review outputs are used.
"""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/publish_wia_final_captures.py'
SPEC = importlib.util.spec_from_file_location('wia_final_capture_publisher_test', SCRIPT)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def campaign_fixture(root, *, baseline_exclusions=0, candidate_exclusions=0):
    campaign = root / 'campaign'
    receipts = campaign / 'interactiveworld_candidate_receipts'
    campaign.mkdir()
    public = {'objective': 'Review this exact candidate layout.', 'constraints': ['Preserve fixtures.']}
    task_path, scene_path = root / 'baseline_task.yaml', root / 'scene.yaml'
    task_path.write_text(yaml.safe_dump({'public': public}), encoding='utf-8')
    scene_path.write_text('id: archived_scene\n', encoding='utf-8')
    baseline_aliases = ['D1-1', 'D1-2', 'D1-3', 'D2-1', 'D2-2',
                        *(f'T{number:02d}' for number in range(56))]
    candidate_aliases = [f'CANDIDATE_{number:02d}' for number in range(19)]
    excluded = [*baseline_aliases[:baseline_exclusions], *candidate_aliases[:candidate_exclusions]]
    rows, groups = [], {}
    for group, aliases in [('baseline', baseline_aliases), ('candidate', candidate_aliases)]:
        items = []
        for number, alias in enumerate(aliases):
            row_id = f'baseline_{number:02d}' if group == 'baseline' else alias
            if group == 'baseline':
                rows.append({'wia_id': row_id, 'document_alias': alias, 'primary_family': 'GF',
                    'implementation_state': 'selected_definition', 'task': {'task_file': str(task_path)},
                    'scene': {'selected_scene_file': str(scene_path)}})
                current_task = task_path
            else:
                current_task = root / 'candidates' / (alias + '.yaml')
                current_task.parent.mkdir(exist_ok=True)
                current_task.write_text(yaml.safe_dump({'id': alias, 'status': 'pending', 'public': public,
                    'private': {'evaluation': {'module': 'wia.authoring.converters.interactiveworld_candidates',
                        'operation': 'mechanism', 'parameters': {'prototype': 'mechanism',
                                                                'condition': {'variant': number}}}}}), encoding='utf-8')
            job_id = f'{group}_capture_{number:02d}'
            job = root / 'queue/jobs' / job_id
            status = {'job_id': job_id, 'state': 'QUEUED' if alias in excluded else 'CACHED',
                      'attempts': 0 if alias in excluded else 1, 'retryable': False}
            write_json(job / 'status.json', status)
            write_json(job / 'input/manifest.json', {'job_id': job_id, 'max_attempts': 3})
            items.append({'row_id': row_id, 'document_alias': alias, 'variant_id': 'final_native_layout',
                'label': 'Exact archived capture selection', 'task_file': str(current_task),
                'task_sha256': publisher._sha(current_task), 'scene_file': str(scene_path),
                'scene_sha256': publisher._sha(scene_path), 'job_id': job_id,
                'queue_status_file': str(job / 'status.json'),
                'receipt_file': str(job / 'attempts/01/preview_complete.json')})
        groups[group] = items
    write_json(campaign / 'registry.json', {'rows': rows})
    curation = {'schema': 'wia_review_evidence_v1', 'row_updates': {'baseline_00': {
        'summary': 'Preserve original curation.', 'variants': []}}, 'extra_rows': []}
    if excluded:
        curation['temporarily_excluded_aliases'] = excluded
    write_json(campaign / 'curation.json', curation)
    for path, group in [(campaign / 'review_index.json', 'baseline'), (receipts / 'review_index.json', 'candidate')]:
        write_json(path, {'schema': 'wia_capture_review_index_v1', 'items': groups[group]})
    write_json(campaign / 'reviews.json', {'feedback': [{'alias': baseline_aliases[0], 'decision': 'rejected'}]})
    (campaign / 'archived_capture.png').write_bytes(b'Historical capture remains on disk.')
    return campaign, receipts, excluded, groups


@pytest.mark.parametrize('baseline_exclusions,candidate_exclusions,expected', [(0, 0, 80), (5, 0, 75), (5, 10, 65)])
def test_active_counts_snapshot_and_preserved_archived_inputs(tmp_path, monkeypatch,
        baseline_exclusions, candidate_exclusions, expected):
    campaign, receipts, excluded, groups = campaign_fixture(tmp_path,
        baseline_exclusions=baseline_exclusions, candidate_exclusions=candidate_exclusions)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    prepared = publisher.prepare(campaign, receipts, project_root=tmp_path)
    progress = prepared['progress']
    assert progress['total'] == progress['active_job_count'] == expected
    assert progress['baseline_count'] == 61 - baseline_exclusions
    assert progress['candidate_count'] == 19 - candidate_exclusions
    assert progress['all_terminal'] is True
    assert progress['states'] == {'CACHED': expected}
    assert progress['temporarily_excluded_aliases'] == excluded
    assert progress['baseline_review']['indexed_variant_count'] == 61 - baseline_exclusions
    assert progress['candidate_review']['indexed_variant_count'] == 19 - candidate_exclusions
    assert progress['archived_baseline_count'] == progress['archived_baseline_review']['indexed_variant_count'] == 61
    assert progress['archived_candidate_count'] == progress['archived_candidate_review']['indexed_variant_count'] == 19
    assert progress['review_summary_scope'] == 'active_nonexcluded_indexed_jobs'
    assert len(prepared['derived_curation']['extra_rows']) == 19
    assert prepared['derived_curation']['row_updates']['baseline_00']['summary'] == 'Preserve original curation.'
    snapshot_path = tmp_path / 'review/snapshot.json'
    monkeypatch.setattr(publisher, 'SNAPSHOT', snapshot_path)
    result = publisher.publish_prepared(prepared, campaign, project_root=tmp_path)
    assert result['snapshot_row_count'] == expected
    snapshot = json.loads(snapshot_path.read_text())
    aliases = {row['document_alias'] for row in snapshot['rows']}
    assert not set(excluded) & aliases
    active_candidates = {item['document_alias'] for item in groups['candidate']} - set(excluded)
    assert active_candidates <= aliases
    assert all(row['proposal_membership'] is False and row['release_admitted'] is False
               for row in snapshot['rows'] if row['document_alias'] in active_candidates)
    assert all(path.read_bytes() == value for path, value in protected.items())


def test_prepare_rejects_unknown_exclusion_alias(tmp_path):
    campaign, receipts, _, _ = campaign_fixture(tmp_path)
    path = campaign / 'curation.json'
    curation = json.loads(path.read_text())
    curation['temporarily_excluded_aliases'] = ['UNKNOWN']
    write_json(path, curation)
    with pytest.raises(ValueError, match='unknown document aliases'):
        publisher.prepare(campaign, receipts, project_root=tmp_path)


def test_watch_finishes_with_75_active_jobs_while_excluded_jobs_remain_queued(tmp_path, monkeypatch, capsys):
    campaign, receipts, excluded, _ = campaign_fixture(tmp_path, baseline_exclusions=5)
    real_prepare, real_publish = publisher.prepare, publisher.publish_prepared
    monkeypatch.setattr(publisher, 'prepare', lambda campaign, receipts:
        real_prepare(campaign, receipts, project_root=tmp_path))
    monkeypatch.setattr(publisher, 'publish_prepared', lambda prepared, campaign:
        real_publish(prepared, campaign, project_root=tmp_path))
    monkeypatch.setattr(publisher, 'SNAPSHOT', tmp_path / 'review/snapshot.json')

    def no_polling_sleep(_):
        raise AssertionError('Watch must finish when all active jobs are terminal.')

    monkeypatch.setattr(publisher.time, 'sleep', no_polling_sleep)
    assert publisher.main(['--apply', '--watch', '--campaign', str(campaign),
                           '--candidate-receipts', str(receipts)]) == 0
    report = json.loads(capsys.readouterr().out)
    completion = json.loads((campaign / 'final_refresh_review/completion.json').read_text())
    assert report['all_terminal'] is completion['all_terminal'] is True
    assert report['total'] == completion['total'] == 75
    assert report['temporarily_excluded_aliases'] == excluded


def test_job_progress_does_not_complete_an_empty_active_selection(tmp_path):
    assert publisher.job_progress([], project_root=tmp_path)['all_terminal'] is False


def test_candidate_metadata_preserves_authored_taxonomy_and_protected_fields(tmp_path, monkeypatch):
    campaign, receipts, _, groups = campaign_fixture(tmp_path)
    original = publisher.prepare(campaign, receipts, project_root=tmp_path)
    alias = groups['candidate'][0]['document_alias']
    fields = {'title': 'Main-authored representative', 'primary_family': 'AP',
              'primary_subclass': 'AP-3', 'summary': 'Explicit retained mechanism.',
              'capability_tags': ['base_translation', 'gripper']}
    curation_path = campaign / 'curation.json'
    curation = json.loads(curation_path.read_text())
    curation['candidate_row_metadata'] = {alias: fields}
    write_json(curation_path, curation)
    original_bytes = curation_path.read_bytes()
    prepared = publisher.prepare(campaign, receipts, project_root=tmp_path)
    before, after = original['derived_curation']['extra_rows'][0], prepared['derived_curation']['extra_rows'][0]
    assert {field: after[field] for field in fields} == fields
    for field in ('id', 'document_alias', 'public', 'variants', 'status', 'proposal_membership', 'release_admitted'):
        assert after[field] == before[field]
    assert after['status'] == 'pending'
    assert after['proposal_membership'] is after['release_admitted'] is False
    assert prepared['derived_curation']['extra_rows'][1:] == original['derived_curation']['extra_rows'][1:]
    monkeypatch.setattr(publisher, 'SNAPSHOT', tmp_path / 'review/snapshot.json')
    publisher.publish_prepared(prepared, campaign, project_root=tmp_path)
    snapshot = json.loads(publisher.SNAPSHOT.read_text())
    row = next(row for row in snapshot['rows'] if row['document_alias'] == alias)
    assert {field: row[field] for field in fields} == fields
    assert snapshot['summary']['extra_family_counts'] == {'null': 18, 'AP': 1}
    assert curation_path.read_bytes() == original_bytes


@pytest.mark.parametrize('metadata', [[], {'UNKNOWN': {'title': 'Unknown alias'}},
    {'CANDIDATE_00': {'release_admitted': True}}, {'CANDIDATE_00': {'primary_family': 'Other'}},
    {'CANDIDATE_00': {'title': 1}}, {'CANDIDATE_00': {'primary_subclass': None}},
    {'CANDIDATE_00': {'summary': []}}, {'CANDIDATE_00': {'capability_tags': 'gripper'}},
    {'CANDIDATE_00': {'capability_tags': [1]}}])
def test_candidate_metadata_rejects_unknown_aliases_fields_and_invalid_types(tmp_path, metadata):
    campaign, receipts, _, _ = campaign_fixture(tmp_path)
    path = campaign / 'curation.json'
    curation = json.loads(path.read_text())
    curation['candidate_row_metadata'] = metadata
    write_json(path, curation)
    with pytest.raises(ValueError, match='(candidate_row_metadata|Candidate)'):
        publisher.prepare(campaign, receipts, project_root=tmp_path)

"""Check variant retry submissions using only isolated files and mocked APIs.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/test_wia_variant_submission.py
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wia.runtime.scene_capture_queue import digest, read_json, sha256, write_json
import wia.runtime.commands.submit_wia_variant_captures as submit


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    campaign = tmp_path / 'campaign'
    queue = campaign / 'queue'
    monkeypatch.setattr(submit, 'CAMPAIGN', campaign)
    source = tmp_path / 'source.json'
    source.write_text('{"unchanged_source": true}\n')
    items = []
    for number in range(7):
        paths = {}
        for kind in ('task', 'scene', 'parent_task', 'parent_scene'):
            path = tmp_path / f'{kind}_{number}.yaml'
            path.write_text(f'fixture: {kind}_{number}\n')
            paths[kind + '_file'] = str(path)
            paths[kind + '_sha256'] = sha256(path)
        record = {'candidate_id': 'category_' + str(number) * 24, **paths,
            'task_edit2_variant_key': str(number) * 24,
            'source_identity': 'Family/Class/Scene/Room/source.json',
            'original_source': str(source), 'source_sha256': sha256(source),
            'task_source_identity': 'Family/Class/Scene/Room/source.json',
            'scene_source_identity': 'Family/Class/Scene/Room/source.json',
            'task_source_sha256': sha256(source), 'scene_source_sha256': sha256(source),
            'capture_source_members': [{'source_identity': 'Family/Class/Scene/Room/source.json',
                'source_file': str(source), 'source_sha256': sha256(source)}],
            'diagnostic_dir': str(tmp_path / f'diagnostics_{number}')}
        record['revision'] = digest(record)
        items.append({'row_id': f'row_{number}', 'variant_id': 'variant_v1',
            'document_alias': f'T{number:02}', 'label': f'Fixture {number}', **paths,
            'source_file': str(source), 'source_sha256': sha256(source), 'record': record})
    records = tmp_path / 'records.json'
    write_json(records, {'items': items})
    index = {'schema': 'wia_capture_review_index_v1', 'items': []}
    states = [('FAILED', True), ('RUNNING', True), ('REVIEW_PENDING', False),
              ('CACHED', False), ('FAILED', False), ('CLAIMED', True), ('FAILED', True)]
    for item, (state, retryable) in zip(items, states):
        job = 'capture_' + str(len(index['items']) + 1) * 32
        entry = {key: item[key] for key in ('row_id', 'variant_id', 'document_alias', 'label',
            'task_file', 'scene_file', 'task_sha256', 'scene_sha256')}
        entry.update(job_id=job, receipt_file=str(queue / 'jobs' / job / 'attempts/01/preview_complete.json'))
        index['items'].append(entry)
        write_json(queue / 'jobs' / job / 'status.json', {'state': state, 'retryable': retryable,
            'attempts': 1, 'error': 'Previous resident native process is still exiting.'})
        write_json(queue / 'jobs' / job / 'input/manifest.json', {'job_id': job, 'max_attempts': 1})
        write_json(Path(entry['receipt_file']), {'partial': True, 'job_id': job})
    write_json(queue / 'claimed' / (index['items'][-1]['job_id'] + '.json'), {'still_owned': True})
    write_json(campaign / 'review_index.json', index)
    write_json(queue / 'capture_runtime_policy.json', {'allowed_code_commits': ['old_code'], 'other_policy': 'keep'})
    frozen_calls, publications = [], []

    def freeze(root):
        assert root == queue
        frozen_calls.append(root)
        return {'code_commit': 'new_code'}

    def publish(record, root, **kwargs):
        assert root == queue
        publications.append((deepcopy(record), kwargs))
        return {'job_id': 'capture_' + digest(record['candidate_id'])[:32]}

    monkeypatch.setattr(submit, 'freeze_working_code', freeze)
    monkeypatch.setattr(submit, 'publish_record', publish)
    return {'campaign': campaign, 'queue': queue, 'items': items, 'records': records,
        'index': index, 'frozen': frozen_calls, 'published': publications}


def _args(workspace):
    return ['--records', str(workspace['records'])]


def test_normal_mode_skips_all_indexed_versions_without_freezing_or_any_writes(workspace):
    before = {path: path.read_bytes() for path in workspace['campaign'].rglob('*') if path.is_file()}
    assert submit.main(_args(workspace)) == 0
    assert not workspace['frozen'] and not workspace['published']
    assert {path: path.read_bytes() for path in workspace['campaign'].rglob('*') if path.is_file()} == before
    assert not workspace['records'].with_suffix('.submission.json').exists()


def test_retry_only_exact_failed_retryable_released_item_and_preserves_old_evidence(workspace):
    queue, campaign = workspace['queue'], workspace['campaign']
    old_files = {path: path.read_bytes() for path in (queue / 'jobs').rglob('*') if path.is_file()}
    report_path = workspace['records'].with_suffix('.submission.json')
    legacy = {'code_commit': 'old_code', 'submitted': [{'job_id': 'legacy_capture', 'receipt_file': 'legacy_receipt'}]}
    write_json(report_path, legacy)
    assert submit.main(_args(workspace) + ['--retry-failed']) == 0
    assert len(workspace['frozen']) == len(workspace['published']) == 1
    record, kwargs = workspace['published'][0]
    previous = workspace['index']['items'][0]
    assert kwargs['max_attempts'] == 3 and kwargs['frozen']['code_commit'] == 'new_code'
    assert record['candidate_id'] != workspace['items'][0]['record']['candidate_id']
    assert record['capture_retry']['superseded_job_id'] == previous['job_id']
    assert record['superseded_captures'][0]['receipt_file'] == previous['receipt_file']
    unsigned = deepcopy(record)
    revision = unsigned.pop('revision')
    assert revision == digest(unsigned)
    index = read_json(campaign / 'review_index.json')
    assert index['items'][1:] == workspace['index']['items'][1:]
    current = index['items'][0]
    assert current['job_id'] != previous['job_id']
    assert current['superseded_captures'][0]['job_id'] == previous['job_id']
    assert current['superseded_captures'][0]['receipt_file'] == previous['receipt_file']
    assert current['superseded_captures'][0]['superseded_status']['state'] == 'FAILED'
    assert {path: path.read_bytes() for path in old_files} == old_files
    report = read_json(report_path)
    assert report['submitted'][0] == legacy['submitted'][0]
    assert report['submitted'][-1] == current
    assert report['code_commit'] == 'old_code' and report['latest_code_commit'] == 'new_code'
    journal = workspace['records'].with_suffix('.submission_history.jsonl')
    event = json.loads(journal.read_text().strip())
    assert event['action'] == 'retry_failed' and event['entry'] == current
    policy = read_json(queue / 'capture_runtime_policy.json')
    assert policy == {'allowed_code_commits': ['old_code', 'new_code'], 'other_policy': 'keep'}


def test_repeated_failed_retry_appends_journal_and_superseded_chain(workspace):
    args = _args(workspace) + ['--retry-failed']
    assert submit.main(args) == 0
    index_file = workspace['campaign'] / 'review_index.json'
    first = read_json(index_file)['items'][0]
    write_json(workspace['queue'] / 'jobs' / first['job_id'] / 'status.json',
        {'state': 'FAILED', 'retryable': True, 'attempts': 3})
    journal = workspace['records'].with_suffix('.submission_history.jsonl')
    before = journal.read_bytes()
    assert submit.main(args) == 0
    assert journal.read_bytes().startswith(before)
    assert len(journal.read_text().splitlines()) == 2
    latest = read_json(index_file)['items'][0]
    assert latest['job_id'] != first['job_id']
    assert [entry['job_id'] for entry in latest['superseded_captures']] == [
        workspace['index']['items'][0]['job_id'], first['job_id']]
    assert len(read_json(workspace['records'].with_suffix('.submission.json'))['submitted']) == 2


@pytest.mark.parametrize('target', ['task', 'scene', 'parent_task', 'parent_scene', 'source', 'record_revision'])
def test_every_item_hash_checked_even_for_skipped_successful_version(workspace, target):
    item = workspace['items'][2]
    if target == 'record_revision':
        package = read_json(workspace['records'])
        package['items'][2]['record']['revision'] = '0' * 64
        write_json(workspace['records'], package)
    else:
        field = 'source_file' if target == 'source' else target + '_file'
        Path(item[field]).write_text('changed fixture bytes\n')
    with pytest.raises(ValueError, match='changed after'):
        submit.main(_args(workspace) + ['--retry-failed'])
    assert not workspace['frozen'] and not workspace['published']


def test_initial_unindexed_submission_defaults_to_three_attempts(workspace):
    index = workspace['index']
    index['items'] = index['items'][1:]
    write_json(workspace['campaign'] / 'review_index.json', index)
    assert submit.main(_args(workspace)) == 0
    record, kwargs = workspace['published'][0]
    assert kwargs['max_attempts'] == 3
    assert record == workspace['items'][0]['record']
    result = read_json(workspace['campaign'] / 'review_index.json')
    assert result['items'][:-1] == index['items']
    assert result['items'][-1]['row_id'] == workspace['items'][0]['row_id']
    assert 'superseded_captures' not in result['items'][-1]


def test_retry_refuses_changed_indexed_definition_without_freeze(workspace):
    index = deepcopy(workspace['index'])
    index['items'][0]['task_sha256'] = '0' * 64
    write_json(workspace['campaign'] / 'review_index.json', index)
    with pytest.raises(ValueError, match='exact indexed'):
        submit.main(_args(workspace) + ['--retry-failed'])
    assert not workspace['frozen'] and not workspace['published']


def test_failed_publication_preserves_index_and_does_not_append_false_history(workspace, monkeypatch):
    monkeypatch.setattr(submit, 'publish_record', lambda *args, **kwargs: {'capture_submission_error': 'fixture failure'})
    before = (workspace['campaign'] / 'review_index.json').read_bytes()
    with pytest.raises(RuntimeError, match='fixture failure'):
        submit.main(_args(workspace) + ['--retry-failed'])
    assert (workspace['campaign'] / 'review_index.json').read_bytes() == before
    assert not workspace['records'].with_suffix('.submission_history.jsonl').exists()

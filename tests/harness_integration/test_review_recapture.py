"""Exercise representative gating, bounded fallback, and durable restart on CPU.

Run: python -m pytest -q tests/test_review_recapture.py
Fixtures are synthetic PNGs and never imply native simulation evidence.
"""
from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest

from wia.runtime.scene_capture_queue import CaptureQueue, PROTOCOL, read_json, sha256, write_json
import wia.runtime.commands.run_esi_review_recapture as campaign


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    queue = CaptureQueue(tmp_path / 'queue')
    output = tmp_path / 'campaign'
    output.mkdir()
    calls = {'freeze': 0, 'publish': []}
    payload = {'records': [
        {'candidate_id': 'one', 'big_task': 'Big', 'small_task': 'A', 'revision': 'r',
         '_batch_file': '/fixture/batch.json', '_report_file': '/fixture/report.json'},
        {'candidate_id': 'two', 'big_task': 'Big', 'small_task': 'A', 'revision': 'r'},
        {'candidate_id': 'three', 'big_task': 'Big', 'small_task': 'A', 'revision': 'r'},
        {'candidate_id': 'four', 'big_task': 'Big', 'small_task': 'B', 'revision': 'r'},
    ], 'representatives': {'Big/A': 'one', 'Big/B': 'four'}}

    def freeze(root):
        calls['freeze'] += 1
        return {'code_commit': 'frozen_fixture'}

    def publish(record, root, *, frozen, priority, batch_file, report_file, timeout_s):
        assert '_batch_file' not in record and '_report_file' not in record
        calls['publish'].append((record['candidate_id'], frozen, priority, batch_file, report_file, timeout_s))
        index = ['one', 'two', 'three', 'four'].index(record['candidate_id']) + 1
        job_id = 'capture_' + str(index) * 32
        manifest = {'job_id': job_id, 'candidate_id': record['candidate_id'],
                    'capture_signature': str(index), 'fingerprint': 'fixture',
                    'code_commit': frozen['code_commit'], 'esi_commit': 'fixture',
                    'variant': 'original', 'seed': 0, 'expected_views': ['head', 'scene'],
                    'max_attempts': 2}
        path = queue.job_dir(job_id) / 'input/manifest.json'
        write_json(path, manifest)
        write_json(path.parent.parent / 'accepted.json', {'manifest_sha256': sha256(path)})
        queue.enqueue(manifest)
        queue.prioritize(job_id, priority)
        result = {**record, 'record_revision': 'r', 'job_id': job_id, 'code_commit': frozen['code_commit']}
        write_json(queue.root / 'category_records' / (record['candidate_id'] + '.json'), result)
        return result

    monkeypatch.setattr(campaign, 'freeze_working_code', freeze)
    monkeypatch.setattr(campaign, 'publish_record', publish)
    state = campaign.prepare_campaign(payload, queue, output)
    return queue, output, payload, state, calls


def terminal(queue, state, candidate, *, success=False, partial=False, queue_state='REVIEW_PENDING'):
    job_id = state['published'][candidate]['job_id']
    status = queue.status(job_id)
    if not success:
        write_json(queue.job_dir(job_id) / 'status.json', {
            **status, 'state': 'FAILED', 'retryable': False, 'attempts': 2, 'error': 'fixture failure'})
        return
    manifest = queue.manifest(job_id)
    images = {}
    for view in (['head'] if partial else manifest['expected_views']):
        path = queue.job_dir(job_id) / (view + '.png')
        Image.new('RGB', (64, 64)).save(path)
        images[view] = str(path)
    receipt = {key: manifest[key] for key in ('capture_signature', 'fingerprint', 'code_commit',
                                             'esi_commit', 'variant', 'seed')}
    receipt.update(capture_protocol=PROTOCOL, complete=True, images=images,
                   qualification={'status': 'rejected'})
    path = queue.job_dir(job_id) / 'receipt.json'
    write_json(path, receipt)
    write_json(queue.job_dir(job_id) / 'status.json', {**status, 'state': queue_state, 'receipt': str(path)})


def test_publication_gates_only_representatives_and_freezes_once(prepared):
    queue, output, payload, state, calls = prepared
    assert calls['freeze'] == 1
    assert [call[0] for call in calls['publish']] == ['one', 'four', 'two', 'three']
    assert [call[2] for call in calls['publish']] == [1000, 1000, 100, 100]
    assert calls['publish'][0][3:] == ('/fixture/batch.json', '/fixture/report.json', 1800)
    assert len((output / 'receipts.jsonl').read_text().splitlines()) == 4
    held_job = state['published']['two']['job_id']
    assert (queue.root / 'held_for_category_review' / (held_job + '.json')).is_file()
    assert queue.claim({'id': 'held'}, job_id=held_job) is None
    claim = queue.claim({'id': 'pilot'})
    assert queue.manifest(claim['job_id'])['candidate_id'] in {'one', 'four'}


def test_failure_fallback_then_bulk_preserves_unrelated_tokens(prepared):
    queue, output, _, state, _ = prepared
    unrelated = queue.root / 'held_for_category_review/capture_aaaaaaaa.json'
    write_json(unrelated, {'foreign': True})
    terminal(queue, state, 'one')
    terminal(queue, state, 'four', success=True)
    state, progress = campaign.step_campaign(queue, output, state)
    assert state['stage'] == 'pilot'
    assert state['categories']['Big/A']['tried'] == ['one', 'two']
    assert queue.claim({'id': 'fallback'}, job_id=state['published']['two']['job_id'])
    terminal(queue, state, 'two', success=True)
    state, progress = campaign.step_campaign(queue, output, state)
    assert state['stage'] == 'bulk'
    assert progress['pilot_success_count'] == 2
    assert progress['counts']['failed'] == 1
    assert read_json(queue.root / 'capture_sample_gate.json')['state'] == campaign.BULK_GATE
    third = state['published']['three']['job_id']
    assert (queue.root / 'queued' / (third + '.json')).exists()
    assert read_json(unrelated) == {'foreign': True}
    terminal(queue, state, 'three', success=True, queue_state='CACHED')
    state, progress = campaign.step_campaign(queue, output, state)
    assert state['stage'] == 'finished'
    assert progress['counts'] == {'captured': 3, 'failed': 1}


def test_exhausted_category_is_reported_failed(prepared):
    queue, output, _, state, _ = prepared
    for candidate in ('one', 'two', 'three'):
        terminal(queue, state, candidate)
    terminal(queue, state, 'four', success=True, queue_state='SUCCEEDED')
    state, progress = campaign.step_campaign(queue, output, state)
    assert state['stage'] == 'finished'
    assert state['categories']['Big/A']['tried'] == ['one', 'two', 'three']
    assert progress['pilot_failure_count'] == 1
    assert len(progress['terminal_failures']) == 3


def test_reconcile_existing_tokens_does_not_republish_or_rewrite_priority(prepared, monkeypatch):
    queue, _, _, state, _ = prepared

    def unexpected_write(*args, **kwargs):
        pytest.fail('An existing gate token or priority must not be rewritten.')

    monkeypatch.setattr(campaign, 'publish_json', unexpected_write)
    monkeypatch.setattr(queue, 'prioritize', unexpected_write)
    campaign.reconcile_gate(queue, state)


def test_reconcile_releases_selected_fallback_before_held_backlog(prepared, monkeypatch):
    queue, _, _, state, _ = prepared
    state['categories']['Big/A']['active'] = 'three'
    state['published'] = {key: state['published'][key] for key in ('two', 'three', 'one', 'four')}
    visited = []
    original_status = queue.status

    def counted_status(job_id):
        visited.append(job_id)
        return original_status(job_id)

    monkeypatch.setattr(queue, 'status', counted_status)
    campaign.reconcile_gate(queue, state)
    assert visited[0] == state['published']['three']['job_id']
    assert queue.claim({'id': 'fallback'}, job_id=state['published']['three']['job_id'])


def test_bulk_reconciliation_does_not_scan_every_job(prepared, monkeypatch):
    queue, _, _, state, _ = prepared
    state['stage'] = 'bulk'
    monkeypatch.setattr(queue, 'status', lambda *_: pytest.fail('Bulk gate must not rescan job states.'))
    monkeypatch.setattr(queue, 'prioritize', lambda *_: pytest.fail('Bulk gate must not rewrite priorities.'))
    campaign.reconcile_gate(queue, state)


def test_missing_expected_view_is_failure_despite_complete_flag(prepared):
    queue, _, _, state, _ = prepared
    terminal(queue, state, 'one', success=True, partial=True)
    assert campaign.observe_record(queue, state['published']['one'])['outcome'] == 'failed'
    terminal(queue, state, 'four', success=True)
    assert campaign.observe_record(queue, state['published']['four'])['outcome'] == 'captured'


def test_source_only_submission_failure_remains_explicit(prepared):
    queue, output, _, state, _ = prepared
    state['published']['one'].update(job_id=None, source_only=True,
                                     capture_submission_error='Invalid diagnostic source scene')
    state, progress = campaign.step_campaign(queue, output, state)
    assert state['categories']['Big/A']['active'] == 'two'
    assert progress['terminal_failures']['one']['state'] == 'SUBMISSION_FAILED'
    assert progress['terminal_failures']['one']['error'] == 'Invalid diagnostic source scene'


def test_resume_preserves_claim_and_reconciles_persisted_release(prepared):
    queue, output, payload, state, calls = prepared
    claim = queue.claim({'id': 'live'}, job_id=state['published']['one']['job_id'])
    before = queue.status(claim['job_id'])
    state = campaign.prepare_campaign(payload, queue, output)
    assert queue.status(claim['job_id']) == before
    assert read_json(queue.root / 'claimed' / (claim['job_id'] + '.json')) == claim
    assert calls['freeze'] == 1 and len(calls['publish']) == 4
    # Simulate interruption after journaling a transition but before token moves.
    state['stage'] = 'bulk'
    write_json(output / 'campaign.json', state)
    state = campaign.prepare_campaign(payload, queue, output)
    assert state['stage'] == 'bulk'
    assert queue.claim({'id': 'bulk'}, job_id=state['published']['two']['job_id'])
    assert queue.status(claim['job_id']) == before


def test_pure_transition_and_retry_window(prepared):
    queue, _, _, state, _ = prepared
    job_id = state['published']['one']['job_id']
    write_json(queue.job_dir(job_id) / 'status.json', {
        'state': 'FAILED', 'retryable': True, 'attempts': 1})
    observations = {c: campaign.observe_record(queue, r) for c, r in state['published'].items()}
    before = deepcopy(state)
    after = campaign.advance_stage(state, observations)
    assert state == before == after
    assert observations['one']['retry_pending']


def test_resume_rejects_changed_input(prepared):
    queue, output, payload, _, _ = prepared
    payload['records'][0]['seed'] = 7
    with pytest.raises(ValueError, match='original records'):
        campaign.prepare_campaign(payload, queue, output)


def test_resume_recovers_cards_after_last_checkpoint_without_republishing(prepared, monkeypatch):
    queue, output, payload, state, calls = prepared
    checkpoint = deepcopy(state)
    checkpoint['published'] = {}
    write_json(output / 'campaign.json', checkpoint)
    claim = queue.claim({'id': 'already-running'}, job_id=state['published']['one']['job_id'])
    before = queue.status(claim['job_id'])
    writes = []
    original_write = campaign.write_json

    def counted_write(path, value):
        if Path(path) == output / 'campaign.json':
            writes.append(deepcopy(value))
        original_write(path, value)

    monkeypatch.setattr(campaign, 'write_json', counted_write)
    recovered = campaign.prepare_campaign(payload, queue, output)
    assert recovered['published'] == state['published']
    assert len(writes) == 1
    assert calls['freeze'] == 1 and len(calls['publish']) == 4
    assert queue.status(claim['job_id']) == before


def test_completion_cache_rechecks_receipt_changes_and_fresh_start(prepared, monkeypatch):
    queue, _, _, state, _ = prepared
    terminal(queue, state, 'one', success=True)
    record = state['published']['one']
    cache, calls = {}, []
    original_complete = campaign.complete_images

    def counted_complete(manifest, receipt):
        calls.append(receipt)
        return original_complete(manifest, receipt)

    monkeypatch.setattr(campaign, 'complete_images', counted_complete)
    assert campaign.observe_record(queue, record, cache)['outcome'] == 'captured'
    assert campaign.observe_record(queue, record, cache)['outcome'] == 'captured'
    assert len(calls) == 1
    status = queue.status(record['job_id'])
    receipt_path = status['receipt']
    receipt = read_json(receipt_path)
    receipt['qualification']['status'] = 'changed fixture qualification'
    write_json(receipt_path, receipt)
    assert campaign.observe_record(queue, record, cache)['outcome'] == 'captured'
    assert len(calls) == 2
    Path(receipt['images']['head']).unlink()
    # No terminal metadata change: cached immutable evidence is retained in this
    # process, but a new coordinator cache must detect missing image files.
    assert campaign.observe_record(queue, record, cache)['outcome'] == 'captured'
    assert campaign.observe_record(queue, record, {})['outcome'] == 'failed'
    assert len(calls) == 3


def test_zero_timeout_is_published_and_cannot_change_on_resume(tmp_path, monkeypatch):
    queue = CaptureQueue(tmp_path / 'queue')
    output = tmp_path / 'campaign'
    output.mkdir()
    payload = {'records': [{'candidate_id': 'one', 'big_task': 'Big', 'small_task': 'A'}],
               'representatives': {'Big/A': 'one'}}
    monkeypatch.setattr(campaign, 'freeze_working_code', lambda root: {'code_commit': 'fixture'})
    calls = []

    def publish(record, root, **kwargs):
        calls.append(kwargs['timeout_s'])
        return {**record, 'job_id': None, 'capture_submission_error': 'fixture submission failure'}

    monkeypatch.setattr(campaign, 'publish_record', publish)
    state = campaign.prepare_campaign(payload, queue, output, capture_timeout_s=0)
    assert calls == [0] and state['capture_timeout_s'] == 0
    with pytest.raises(ValueError, match='original capture timeout'):
        campaign.prepare_campaign(payload, queue, output, capture_timeout_s=1800)


def test_large_cards_produce_bounded_state_and_traceable_receipts(prepared):
    queue, output, payload, state, _ = prepared
    checkpoint = deepcopy(state)
    checkpoint['published'] = {}
    write_json(output / 'campaign.json', checkpoint)
    for candidate in state['published']:
        card = queue.root / 'category_records' / (candidate + '.json')
        full = read_json(card)
        full['large_source_evidence'] = 'raw fixture evidence ' * 100000
        write_json(card, full)
    recovered = campaign.prepare_campaign(payload, queue, output)
    assert all(set(record) <= set(campaign.PUBLICATION_FIELDS)
               for record in recovered['published'].values())
    assert (output / 'campaign.json').stat().st_size < 10000
    receipts = [json.loads(line) for line in (output / 'receipts.jsonl').read_text().splitlines()]
    assert len(receipts) == len(payload['records'])
    assert all(Path(row['category_record_file']).is_file() for row in receipts)
    assert (output / 'receipts.jsonl').stat().st_size < 10000


def test_unchanged_tick_does_not_rewrite_campaign_or_reconcile_tokens(prepared, monkeypatch):
    queue, output, _, state, _ = prepared
    writes = []
    original_write = campaign.write_json

    def counted_write(path, value):
        writes.append(Path(path))
        original_write(path, value)

    monkeypatch.setattr(campaign, 'write_json', counted_write)
    monkeypatch.setattr(campaign, 'reconcile_gate',
                        lambda *args: pytest.fail('Unchanged tick must not reconcile tokens.'))
    state, _ = campaign.step_campaign(queue, output, state)
    state, _ = campaign.step_campaign(queue, output, state)
    assert writes == [output / 'progress.json', output / 'progress.json']


def test_unchanged_resume_preserves_campaign_and_gate_bytes(prepared):
    queue, output, payload, _, _ = prepared
    paths = [output / 'campaign.json', queue.root / 'capture_sample_gate.json']
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    campaign.prepare_campaign(payload, queue, output)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


def test_local_progress_is_not_rewritten_without_observation_changes(prepared, monkeypatch, tmp_path):
    queue, output, _, state, _ = prepared
    runtime = tmp_path / 'local_runtime'
    writes = []
    original_write = campaign.write_json

    def counted_write(path, value):
        writes.append(Path(path))
        original_write(path, value)

    monkeypatch.setattr(campaign, 'write_json', counted_write)
    cache = {}
    state, _ = campaign.step_campaign(queue, output, state, cache, progress_output=runtime)
    state, _ = campaign.step_campaign(queue, output, state, cache, progress_output=runtime)
    assert writes == [runtime / 'progress.json']


def test_bulk_release_continues_representative_tracking_without_blocking_work(prepared):
    queue, output, _, state, _ = prepared
    gate = read_json(queue.root / 'capture_sample_gate.json')
    gate['state'] = campaign.BULK_GATE
    write_json(queue.root / 'capture_sample_gate.json', gate)
    state, _ = campaign.step_campaign(queue, output, state)
    assert state['stage'] == 'bulk'
    assert queue.claim({'id': 'bulk'}, job_id=state['published']['two']['job_id'])
    terminal(queue, state, 'one', success=True)
    state, progress = campaign.step_campaign(queue, output, state)
    assert state['stage'] == 'bulk'
    assert progress['pilot_success_count'] == 1
    assert progress['counts']['pending'] == 3


def test_resume_honors_durable_bulk_release(prepared):
    queue, output, payload, _, _ = prepared
    gate = read_json(queue.root / 'capture_sample_gate.json')
    gate['state'] = campaign.BULK_GATE
    write_json(queue.root / 'capture_sample_gate.json', gate)
    resumed = campaign.prepare_campaign(payload, queue, output)
    assert resumed['stage'] == 'bulk'
    assert read_json(output / 'campaign.json')['stage'] == 'bulk'
    assert read_json(queue.root / 'capture_sample_gate.json')['state'] == campaign.BULK_GATE

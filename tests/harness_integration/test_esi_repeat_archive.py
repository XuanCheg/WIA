"""Verify reversible repeat archival and preservation of active GPU ownership.

Run: python -m pytest -q tests/test_esi_repeat_archive.py
No simulator or image decoding is used.
"""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from wia.runtime.scene_capture_queue import CaptureQueue, read_json, write_json
import wia.runtime.commands.archive_esi_repeats as archive


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    base, queue = tmp_path / 'campaign', tmp_path / 'queue'
    CaptureQueue(queue)
    records = []
    for suffix, state in [('aa', 'REVIEW_PENDING'), ('bb', 'QUEUED')]:
        key, job = 'category_' + suffix, 'capture_' + suffix
        directory = base / 'candidates' / key
        directory.mkdir(parents=True)
        source = tmp_path / (suffix + '.json')
        write_json(source, {'scene': 'house', 'room': 'room', 'runner_task': 'geometric',
                            'metadata_json': '{}', 'question': suffix, 'answer': suffix})
        task = {'id': key, 'source': suffix, 'public': {'objective': suffix,
                'success_criteria': ['Return after the circuit.']},
                'private': {'evaluation': {'operation': 'circuit', 'return': True}}}
        (directory / 'task.yaml').write_text(yaml.safe_dump(task))
        (directory / 'scene.yaml').write_text(yaml.safe_dump(
            {'id': key, 'source': suffix, 'scene': {'seed': 0}}))
        record = {'candidate_id': key, 'source_identity': suffix + '.json', 'status': 'generated',
            'big_task': 'Spatial Relations', 'small_task': 'Geometric Configuration',
            'original_source': str(source), 'task_file': str(directory / 'task.yaml'),
            'scene_file': str(directory / 'scene.yaml'), 'diagnostic_dir': str(directory),
            'scene_model': 'house', 'room': 'room'}
        write_json(directory / 'record.json', record)
        records.append(record)
        artifact = base / 'capture_artifacts/jobs' / job
        write_json(artifact / 'status.json', {'state': state, 'attempts': 0})
        (artifact / 'head.png').write_bytes(b'Existing bytes; image validation is forbidden.')
        (queue / 'jobs' / job).symlink_to(artifact, target_is_directory=True)
        write_json(queue / 'category_records' / (key + '.json'), {**record, 'job_id': job})
        if state == 'QUEUED':
            write_json(queue / 'queued' / (job + '.json'), {'job_id': job})
    batch = base / 'batch.json'
    write_json(batch, {'records': records, 'source_count': 2, 'counts': {'generated': 2}})
    write_json(base / 'review_catalogue/category_revisions.json', {
        'categories': {'category': {'new_batch_file': str(batch), 'complete': True}}})
    reviews = tmp_path / 'reviews.json'
    write_json(reviews, {'entries': {'category_aa': {'decision': 'approved'}},
                         'history': [{'id': 'category_bb', 'review': {'notes': 'Keep this history.'}}]})
    write_json(queue / 'capture_sample_gate.json', {'authorized_candidate_ids': ['category_aa', 'category_bb']})
    monkeypatch.setattr(archive, 'AUDITED', {'Geometric Configuration': 1})
    return base, queue, reviews, records


def test_archive_withdraws_moves_preserves_bytes_and_is_resumable(campaign):
    base, queue, reviews, records = campaign
    source_bytes = Path(records[1]['original_source']).read_bytes()
    review_bytes = reviews.read_bytes()
    task_bytes = Path(records[1]['task_file']).read_bytes()
    result = archive.apply_archive(base, queue, reviews)
    assert result['complete'] and result['repeat_count'] == 1
    assert not (queue / 'queued/capture_bb.json').exists()
    assert read_json(queue / 'jobs/capture_bb/status.json')['state'] == 'CANCELLED'
    root = base / 'repeat/tasks/category_bb'
    assert (root / 'definition/task.yaml').read_bytes() == task_bytes
    assert (root / 'capture/capture_bb/head.png').read_bytes().startswith(b'Existing bytes')
    assert Path(records[1]['task_file']).read_bytes() == task_bytes
    assert (queue / 'jobs/capture_bb/head.png').read_bytes().startswith(b'Existing bytes')
    assert Path(records[1]['original_source']).read_bytes() == source_bytes
    assert reviews.read_bytes() == review_bytes
    assert read_json(base / 'batch.json')['counts'] == {'generated': 1, 'repeat': 1}
    assert read_json(base / 'batch.json')['records'][0] == records[0]
    assert read_json(queue / 'capture_sample_gate.json')['authorized_candidate_ids'] == ['category_aa']
    assert read_json(base / 'review_catalogue/retired_candidates.json')['candidates']['category_bb']['repeat_of'] == 'category_aa'
    assert archive.apply_archive(base, queue, reviews)['complete']


def test_active_repeat_keeps_owner_and_gpu_job_until_completion(campaign):
    base, queue, reviews, _ = campaign
    status_path = queue / 'jobs/capture_bb/status.json'
    status = {'state': 'RUNNING', 'attempts': 1, 'claim': {'worker': 'worker7'}}
    write_json(status_path, status)
    claim = queue / 'claimed/capture_bb.json'
    write_json(claim, {'token': 'owned'})
    result = archive.apply_archive(base, queue, reviews)
    assert result['pending_capture_jobs'] == ['capture_bb']
    assert read_json(status_path) == status
    assert read_json(claim) == {'token': 'owned'}
    assert not (queue / 'jobs/capture_bb/cancel.json').exists()
    write_json(status_path, {**status, 'state': 'REVIEW_PENDING'})
    # Terminal status alone is insufficient while the owner still holds its claim.
    plan = read_json(base / 'repeat/manifest.json')
    assert not archive.drain_jobs(CaptureQueue(queue), base / 'repeat', plan)['complete']
    claim.unlink()
    assert archive.drain_jobs(CaptureQueue(queue), base / 'repeat', plan)['complete']
    assert (base / 'repeat/tasks/category_bb/capture/capture_bb/head.png').exists()


def test_claimed_queued_race_is_not_cancelled(campaign):
    base, queue, _, _ = campaign
    claim = queue / 'claimed/capture_bb.json'
    write_json(claim, {'token': 'new-owner'})
    destination = base / 'race'
    archive.withdraw_job(CaptureQueue(queue), 'capture_bb', {'repeat_of': 'category_aa'}, destination)
    assert read_json(queue / 'jobs/capture_bb/status.json')['state'] == 'QUEUED'
    assert read_json(claim) == {'token': 'new-owner'}
    assert not (queue / 'queued/capture_bb.json').exists()


def test_distinct_operation_is_not_archived(campaign):
    base, queue, reviews, records = campaign
    task_file = Path(records[1]['task_file'])
    task = yaml.safe_load(task_file.read_text())
    task['private']['evaluation']['return'] = False
    task_file.write_text(yaml.safe_dump(task))
    before = deepcopy(read_json(base / 'batch.json'))
    with pytest.raises(ValueError, match='counts changed'):
        archive.plan_archive(base, queue, reviews)
    assert read_json(base / 'batch.json') == before
    assert not (base / 'repeat').exists()

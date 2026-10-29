"""Check reusable Review task updates with isolated CPU metadata fixtures.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q
tests/test_review_task_updates.py tests/test_annotation_publication.py
All sources, Review records, queues and screenshots are temporary. These tests
never freeze live code, validate images or submit native/GPU workloads.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import pytest

from wia.runtime.scene_capture_queue import read_json, write_json
import wia.runtime.commands.publish_esi_annotation_repairs as publication_script
import wia.runtime.commands.update_esi_review_tasks as updates
from harness_integration.test_annotation_publication import forbid_discovery, publication


def revision_pair(fixture, changed):
    """Retain runtime parents whose CPU revision matches unchanged raw records."""
    previous_file = Path(read_json(fixture.review / 'category_revisions.json')
                         ['categories'][fixture.key]['new_batch_file'])
    parents, records = [], deepcopy(fixture.records)
    for index, record in enumerate(records):
        suffix = record['candidate_id'].rsplit('_', 1)[-1]
        record['converter_version'] = 'v1'
        parent = {**deepcopy(record), 'candidate_id': 'category_old_' + suffix,
            'revision': 'runtime_parent_' + suffix, 'cpu_conversion_revision': record['revision'],
            'runtime_revision': 'parent_runtime', 'retained_evidence': {'source_slot': index}}
        parents.append(parent)
        if suffix in changed:
            record.update(revision='cpu_revision_v2_' + suffix, converter_version='v2')
    write_json(previous_file, {'records': parents})
    write_json(fixture.cpu_batch, {'records': records, 'counts': {'generated': len(records)},
                                  'source_count': len(records)})
    return previous_file, parents, records


def isolated_submit(fixture, monkeypatch):
    """Record enqueue calls while preserving the durable-publication ordering."""
    calls = []

    def submit(record, queue, **kwargs):
        assert queue == fixture.queue
        output = Path(kwargs['batch_file']).parents[2]
        journal = read_json(output / 'publication.json')
        assert not journal['complete'] and len(journal['samples']) == 1
        assert not journal['samples'][0].get('job_id')
        calls.append(deepcopy(record))
        job = 'capture_changed_fixture' + ('_' + str(len(calls)) if len(calls) > 1 else '')
        write_json(queue / 'queued' / (job + '.json'), {'job_id': job, 'candidate_id': record['candidate_id']})
        write_json(queue / 'jobs' / job / 'status.json', {'state': 'QUEUED'})
        return {'job_id': job, 'capture_submission_error': None}

    monkeypatch.setattr(publication_script, 'publish_record', submit)
    return calls


@pytest.mark.parametrize('refresh_runtime', [False, True])
def test_unchanged_cpu_batch_does_not_revive_archived_repeats(publication, refresh_runtime):
    fixture = publication
    previous_file, parents, records = revision_pair(fixture, changed=set())
    parents[1].update(status='repeat', repeat_of=parents[0]['candidate_id'])
    write_json(previous_file, {'records': parents})
    _, changed = updates.plan_updates(fixture.campaign, fixture.queue,
                                      [fixture.cpu_batch], refresh_runtime=refresh_runtime)
    assert records[1]['source_identity'] not in changed
    assert len(changed) == (2 if refresh_runtime else 0)


def test_new_cpu_revision_can_reconsider_an_archived_source(publication):
    fixture = publication
    previous_file, parents, records = revision_pair(fixture, changed={'b'})
    parents[1].update(status='repeat', repeat_of=parents[0]['candidate_id'])
    write_json(previous_file, {'records': parents})
    _, changed = updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert changed == {records[1]['source_identity']}


def test_plan_uses_cpu_parent_revision_and_prefers_prior_sample_then_reviewed_source(publication):
    fixture = publication
    previous_file, parents, records = revision_pair(fixture, changed={'a', 'b'})
    proposals, changed = updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert changed == {records[0]['source_identity'], records[1]['source_identity']}
    assert proposals[0]['candidate_ids'] == [records[0]['candidate_id']]
    assert proposals[0]['changed_count'] == 2
    assert records[2]['source_identity'] not in changed
    # If the old sample is unchanged, prefer the reviewed diagnostic over an unseen source.
    previous_file, parents, records = revision_pair(fixture, changed={'b', 'c'})
    proposals, changed = updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert proposals[0]['candidate_ids'] == [records[1]['candidate_id']]
    assert changed == {records[1]['source_identity'], records[2]['source_identity']}
    current = [{**record, 'revision': 'published_runtime_' + str(index),
                'cpu_conversion_revision': record['revision']} for index, record in enumerate(records)]
    write_json(previous_file, {'records': current})
    assert updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch]) == ([], set())


def test_plan_rejects_omission_of_an_old_failed_source(publication):
    fixture = publication
    previous_file, parents, records = revision_pair(fixture, changed={'b'})
    parents[-1].update(status='conversion_failed', failure_reason='Original asset unavailable.')
    write_json(previous_file, {'records': parents})
    write_json(fixture.cpu_batch, {'records': records[:-1]})
    before = previous_file.read_bytes()
    with pytest.raises(ValueError, match='Do not omit old failures or sources'):
        updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert previous_file.read_bytes() == before


def test_partial_publication_preserves_unchanged_parents_and_retires_only_changed_source(publication, monkeypatch):
    fixture = publication
    _, parents, records = revision_pair(fixture, changed={'b'})
    proposals, changed = updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    output = fixture.campaign / 'review_updates/isolated_update'
    calls = isolated_submit(fixture, monkeypatch)
    annotations = deepcopy(fixture.annotations)
    annotations['entries']['category_old_a'] = {'source': parents[0]['source_identity'],
        'decision': 'approved', 'notes': 'Keep the unchanged source review exactly.'}
    annotations['history'].append({'id': 'category_old_a', 'notes': 'Keep unchanged review history.'})
    write_json(fixture.reviews_path, annotations)
    cpu_bytes = fixture.cpu_batch.read_bytes()
    result = publication_script.publish(fixture.campaign, fixture.queue, proposal_rows=proposals,
        changed_sources=changed, output_dir=output, frozen=fixture.frozen)
    runtime = read_json(result['categories'][0]['new_batch_file'])['records']
    assert runtime[0] == parents[0] and runtime[2] == parents[2]
    assert len(calls) == len(result['samples']) == 1
    assert calls[0]['source_identity'] == records[1]['source_identity']
    assert runtime[1]['parent_candidate_id'] == parents[1]['candidate_id']
    assert runtime[1]['cpu_conversion_revision'] == records[1]['revision']
    assert runtime[1]['native_recheck_required']
    assert runtime[1]['prior_feedback'][0]['notes'] == fixture.old_note['notes']
    retired = read_json(fixture.review / 'retired_candidates.json')['candidates']
    assert set(retired) == {'category_old_b'}
    assert result['retired_candidate_count'] == result['removed_old_review_entries'] == 1
    assert read_json(result['prior_annotations_archived']) == annotations
    assert read_json(fixture.reviews_path)['entries'] == {
        'category_old_a': annotations['entries']['category_old_a'],
        'category_unrelated': fixture.annotations['entries']['category_unrelated']}
    assert [row['id'] for row in read_json(fixture.reviews_path)['history']] == ['category_unrelated', 'category_old_a']
    assert Path(result['output_dir']) == output
    assert fixture.cpu_batch.read_bytes() == cpu_bytes
    for path, value in fixture.tokens.items():
        assert path.read_bytes() == value
    assert publication_script.publish(fixture.campaign, fixture.queue, proposal_rows=proposals,
        changed_sources=changed, output_dir=output, frozen=fixture.frozen) == result
    assert len(calls) == 1 and fixture.freeze.call_count == 0


@pytest.mark.parametrize('replacement_state', ['RUNNING', 'QUEUED'])
def test_submission_cleanup_only_removes_declared_retired_pngs(tmp_path, monkeypatch, replacement_state):
    campaign, queue, output = tmp_path / 'campaign', tmp_path / 'queue', tmp_path / 'update'
    output.mkdir()

    def artifact(job, name):
        path = campaign / 'capture_artifacts/jobs' / job / 'attempts/01' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('Invalid image or retained evidence: ' + job + '/' + name).encode())
        return path

    old = artifact('old_idle', 'declared.png')
    undeclared = artifact('old_idle', 'undeclared.png')
    current = artifact('replacement', 'head.png')
    active = artifact('old_claimed', 'head.png')
    log = artifact('old_idle', 'native.log')
    source = tmp_path / 'source.png'
    source.write_bytes(b'Original source image bytes.')
    source_link = old.parent / 'source_link.png'
    source_link.symlink_to(source)
    receipt = queue / 'jobs/old_idle/attempts/01/preview_complete.json'
    write_json(receipt, {'images': {'head': str(old), 'current': str(current), 'source': str(source),
                                  'source_link': str(source_link), 'log': str(log)}})
    write_json(queue / 'jobs/old_idle/status.json', {'state': 'REVIEW_PENDING', 'attempts': 1, 'receipt': str(receipt)})
    write_json(queue / 'jobs/old_claimed/status.json', {'state': 'RUNNING', 'partial_images': {'head': str(active)}})
    claim = queue / 'claimed/old_claimed.json'
    write_json(claim, {'job_id': 'old_claimed'})
    write_json(queue / 'jobs/replacement/status.json', {'state': replacement_state})
    protected = {path: path.read_bytes() for path in (undeclared, current, active, log, source, source_link, receipt, claim)}
    published = {'output_dir': str(output), 'cleanup_on_submission': True,
        'samples': [{'candidate_id': 'new_candidate', 'job_id': 'replacement'}],
        'cleanup_jobs': {job: {'replacement_candidate_id': 'new_candidate'} for job in ('old_idle', 'old_claimed')}}
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() == '.png':
            forbid_discovery()
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(PIL.Image, 'open', forbid_discovery)
        guard.setattr(Path, 'open', metadata_only)
        for name in ('glob', 'rglob', 'iterdir'):
            guard.setattr(Path, name, forbid_discovery)
        result = publication_script.cleanup(campaign, queue, published)
    assert result['completed_jobs'] == ['old_idle'] and not result['complete']
    assert result['removed_png_files'] == 1
    assert not old.exists() and source_link.is_symlink()
    assert all(path.read_bytes() == value for path, value in protected.items())


def test_generic_update_second_run_does_not_freeze_publish_or_duplicate_queue(publication, monkeypatch):
    fixture = publication
    revision_pair(fixture, changed={'b'})
    calls = isolated_submit(fixture, monkeypatch)
    frozen = Mock(return_value=fixture.frozen)
    publisher = Mock(return_value=SimpleNamespace(tick=Mock(return_value={})))
    publish = Mock(wraps=publication_script.publish)
    monkeypatch.setattr(updates, 'freeze_working_code', frozen)
    monkeypatch.setattr(updates, 'publish', publish)
    monkeypatch.setattr(updates, 'CataloguePublisher', publisher)
    first = updates.update(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert first['changed_sources'] == first['queued_representatives'] == first['categories'] == 1
    assert first['cleanup_complete']
    assert not (fixture.review / 'task_update_pending.json').exists()
    queue_token = fixture.queue / 'queued/capture_changed_fixture.json'
    queue_bytes = queue_token.read_bytes()
    second = updates.update(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert second == {'changed_sources': 0, 'queued_representatives': 0, 'categories': 0,
                      'dry_run': False, 'unchanged': True}
    assert len(calls) == frozen.call_count == publish.call_count == publisher.call_count == 1
    assert queue_token.read_bytes() == queue_bytes
    for path, value in fixture.tokens.items():
        assert path.read_bytes() == value


def completed_pending_with_active_predecessor(fixture, monkeypatch):
    """Reproduce a completed legacy publication fenced by an old native claim."""
    _, _, records = revision_pair(fixture, changed={'b'})
    proposals, changed = updates.plan_updates(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    calls = isolated_submit(fixture, monkeypatch)
    output = fixture.campaign / 'review_updates/already_published'
    result = publication_script.publish(fixture.campaign, fixture.queue, proposal_rows=proposals,
        changed_sources=changed, output_dir=output, frozen=fixture.frozen)
    assert result['complete'] and set(result['cleanup_jobs']) == {'capture_old_b'}
    old = fixture.campaign / 'capture_artifacts/jobs/capture_old_b/attempts/01/head.png'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'Declared old screenshot; never decode this image.')
    write_json(fixture.queue / 'jobs/capture_old_b/status.json',
               {'state': 'RUNNING', 'partial_images': {'head': str(old)}})
    claim = fixture.queue / 'claimed/capture_old_b.json'
    write_json(claim, {'job_id': 'capture_old_b', 'owner': 'active_original_worker'})
    write_json(fixture.review / 'task_update_pending.json', {'proposals': proposals,
        'changed_sources': sorted(changed), 'output_dir': str(output), 'frozen': fixture.frozen})
    return output / 'publication.json', old, claim, calls, records


def test_complete_legacy_publication_does_not_block_new_cpu_revision_in_same_call(publication, monkeypatch):
    fixture = publication
    old_publication, old, claim, calls, records = completed_pending_with_active_predecessor(fixture, monkeypatch)
    records[1].update(revision='cpu_revision_v3_b', converter_version='v3', candidate_id='category_cpu_v3_b')
    write_json(fixture.cpu_batch, {'records': records, 'counts': {'generated': 3}, 'source_count': 3})
    frozen, publisher = Mock(return_value=fixture.frozen), Mock()
    publish, clean = Mock(wraps=publication_script.publish), Mock(wraps=publication_script.cleanup)
    monkeypatch.setattr(updates, 'freeze_working_code', frozen)
    monkeypatch.setattr(updates, 'CataloguePublisher', publisher)
    monkeypatch.setattr(updates, 'publish', publish)
    monkeypatch.setattr(updates, 'cleanup', clean)
    monkeypatch.setattr(PIL.Image, 'open', forbid_discovery)
    protected = {path: path.read_bytes() for path in (old_publication, old, claim,
        fixture.queue / 'queued/capture_changed_fixture.json', *fixture.tokens)}
    result = updates.update(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert result['changed_sources'] == result['queued_representatives'] == result['categories'] == 1
    assert not result['cleanup_complete']
    assert len(calls) == 2 and calls[-1]['cpu_conversion_revision'] == 'cpu_revision_v3_b'
    assert frozen.call_count == publish.call_count == publisher.call_count == 1
    assert sum(Path(call.args[2]['output_dir']) / 'publication.json' == old_publication
               for call in clean.call_args_list) == 1
    assert read_json(fixture.review / 'task_update_cleanup_pending.json') == {
        'publications': [str(old_publication), str(Path(result['journal']).parent / 'publication.json')]}
    assert not (fixture.review / 'task_update_pending.json').exists()
    assert (fixture.queue / 'queued/capture_changed_fixture_2.json').exists()
    current_batch = read_json(fixture.review / 'category_revisions.json')['categories'][fixture.key]['new_batch_file']
    assert read_json(current_batch)['records'][1]['cpu_conversion_revision'] == 'cpu_revision_v3_b'
    assert all(path.read_bytes() == value for path, value in protected.items())


def test_unchanged_rerun_drains_deferred_cleanup_without_freeze_or_enqueue(publication, monkeypatch):
    fixture = publication
    old_publication, old, claim, calls, _ = completed_pending_with_active_predecessor(fixture, monkeypatch)
    for name in ('freeze_working_code', 'publish', 'CataloguePublisher'):
        monkeypatch.setattr(updates, name, Mock(side_effect=AssertionError('No new publication is needed.')))
    clean = Mock(wraps=publication_script.cleanup)
    monkeypatch.setattr(updates, 'cleanup', clean)
    monkeypatch.setattr(PIL.Image, 'open', forbid_discovery)
    token = fixture.queue / 'queued/capture_changed_fixture.json'
    protected = {path: path.read_bytes() for path in (token, old_publication, *fixture.tokens)}
    first = updates.update(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert first['unchanged'] and not first['cleanup_complete']
    assert old.exists() and claim.exists()
    assert not (fixture.review / 'task_update_pending.json').exists()
    claim.unlink()  # Only the isolated fixture's original worker releases ownership.
    second = updates.update(fixture.campaign, fixture.queue, [fixture.cpu_batch])
    assert second['unchanged'] and second['changed_sources'] == second['queued_representatives'] == 0
    assert second['cleanup_complete'] and second['removed_png_files'] == 1
    assert not old.exists() and not (fixture.review / 'task_update_cleanup_pending.json').exists()
    assert clean.call_count == 2 and len(calls) == 1
    assert all(path.read_bytes() == value for path, value in protected.items())


def test_dry_run_cli_does_not_migrate_drain_lock_or_write(publication, monkeypatch, capsys):
    fixture = publication
    old_publication, old, claim, _, records = completed_pending_with_active_predecessor(fixture, monkeypatch)
    registry = fixture.review / 'task_update_cleanup_pending.json'
    write_json(registry, {'publications': [str(old_publication)]})
    records[1].update(revision='cpu_revision_v3_b', converter_version='v3')
    write_json(fixture.cpu_batch, {'records': records})
    protected = {path: path.read_bytes() for path in (old_publication, old, claim, registry,
        fixture.review / 'task_update_pending.json', fixture.review / 'category_revisions.json',
        fixture.cpu_batch, *fixture.tokens)}
    monkeypatch.setattr('sys.argv', ['update_esi_review_tasks.py', '--campaign-root', str(fixture.campaign),
        '--queue-root', str(fixture.queue), '--batch-file', str(fixture.cpu_batch), '--dry-run'])
    with monkeypatch.context() as guard:
        for name in ('freeze_working_code', 'publish', 'cleanup', 'write_json', 'CataloguePublisher'):
            guard.setattr(updates, name, forbid_discovery)
        for name in ('mkdir', 'unlink', 'glob', 'rglob', 'iterdir'):
            guard.setattr(Path, name, forbid_discovery)
        guard.setattr(updates.fcntl, 'flock', forbid_discovery)
        guard.setattr(PIL.Image, 'open', forbid_discovery)
        updates.main()
    output = capsys.readouterr().out
    assert "'dry_run': True" in output and "'changed_sources': 1" in output
    assert not (fixture.review / 'task_update.lock').exists()
    assert all(path.read_bytes() == value for path, value in protected.items())

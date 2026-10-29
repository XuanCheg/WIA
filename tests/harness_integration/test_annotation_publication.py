"""Check isolated annotation publication, retirement and declared-cache cleanup.

Run: python -m pytest -q tests/test_annotation_publication.py
All campaign, queue, source and ROOT paths are temporary CPU fixtures. Image
contents are deliberately invalid: these metadata operations must not validate
images, discover screenshots, freeze real code or submit native workloads.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import PIL.Image
import pytest
import yaml

from wia.runtime.scene_capture_queue import digest, read_json, write_json
import wia.runtime.commands.publish_esi_annotation_repairs as repair
import wia.runtime.commands.update_esi_review_catalogue as catalogue


def forbid_discovery(*args, **kwargs):
    raise AssertionError('Annotation metadata must not scan or validate screenshots.')


@pytest.fixture
def publication(tmp_path, monkeypatch):
    root, campaign, queue = tmp_path / 'project', tmp_path / 'campaign', tmp_path / 'queue'
    output, review = campaign / 'annotation_repair_20261015', campaign / 'review_catalogue'
    reviews_path = root / 'outputs/local_capture_20261014/gallery/reviews.json'
    monkeypatch.setattr(repair, 'ROOT', root)
    monkeypatch.setattr(catalogue, 'ROOT', root)
    big, small, key = 'CPU Fixture', 'Annotated Repair', 'fixture_category'
    records, previous, old_rows = [], [], []
    for suffix in ('a', 'b', 'c'):
        source = f'{big}/{small}/house/room_0/{suffix}.json'
        directory = campaign / 'cpu_batch/candidates' / ('category_cpu_' + suffix)
        task = directory / 'task.yaml'
        task.parent.mkdir(parents=True)
        task.write_text(yaml.safe_dump({'public': {'objective': 'Move the original prop.'},
                                        'private': {'answer': 'PRIVATE_QA_CANARY'}}))
        write_json(directory / 'private_qa.json', {'answer': 'PRIVATE_QA_CANARY', 'source': source})
        record = {'big_task': big, 'small_task': small, 'source_identity': source,
            'candidate_id': 'category_cpu_' + suffix, 'revision': 'cpu_revision_' + suffix,
            'status': 'generated', 'warning_codes': [], 'diagnostic_dir': str(directory),
            'task_file': str(task), 'scene_file': str(directory / 'scene.yaml'), 'seed': 0}
        write_json(directory / 'record.json', record)
        records.append(record)
        previous.append({**record, 'candidate_id': 'category_old_' + suffix, 'revision': 'old_' + suffix})
        old_rows.append({'id': 'category_old_' + suffix, 'source': source,
            'category': big, 'subcategory': small, 'conversion_result_status': 'generated',
            'instance_status': 'interaction_setup_failed' if suffix == 'b' else 'capture_complete',
            'failure': 'No fitting native support.' if suffix == 'b' else '',
            'capture': {'job_id': 'capture_old_' + suffix}})
    cpu_batch = campaign / 'cpu_batch/batch_manifest.json'
    prior_batch = campaign / 'prior_batch.json'
    write_json(cpu_batch, {'records': records, 'counts': {'generated': 3}, 'source_count': 3})
    write_json(prior_batch, {'records': previous})
    write_json(review / 'category_revisions.json', {'categories': {
        key: {'complete': True, 'new_batch_file': str(prior_batch)}}})
    snapshot = {'scenes': [old_rows[0]], 'diagnostics': [old_rows[1]]}
    write_json(review / 'snapshot.json', snapshot)
    old_note = {'source': records[1]['source_identity'], 'decision': 'needs_fix',
                'notes': 'Put the props on a reachable support.', 'revision': 2}
    annotations = {'entries': {'category_old_b': old_note,
        'category_unrelated': {'source': 'Unchanged/source.json', 'notes': 'Keep this annotation.'}},
        'history': [{'id': 'category_old_b', 'notes': old_note['notes']},
                    {'id': 'category_unrelated', 'notes': 'Keep this history.'}]}
    write_json(reviews_path, annotations)
    inventory = [{'key': key, 'big_task': big, 'small_task': small,
        'sources': [record['source_identity'] for record in records], 'report_file': str(root / 'report.md')}]
    write_json(campaign / 'inventory.json', {'categories': inventory})
    selection = campaign / 'old_samples.json'
    untouched_sample = {'key': 'unmodified_category', 'candidate_id': 'category_keep'}
    write_json(selection, {'samples': [{'key': key, 'candidate_id': 'category_old_a'}, untouched_sample]})
    write_json(queue / 'capture_sample_gate.json', {'state': 'ONE_PER_CATEGORY_AWAITING_REVIEW',
        'selection_file': str(selection), 'selected_candidate_ids': ['category_old_a', 'category_keep'],
        'selection_pending': False})
    write_json(queue / 'capture_runtime_policy.json', {'review_allowed_code_commits': ['older_runtime']})
    old_log = campaign / 'old_native.log'
    old_log.write_text('The preceding native revision failed support qualification.\n')
    write_json(queue / 'jobs/capture_old_b/status.json', {'native_log': str(old_log)})
    tokens = {queue / 'held/capture_held.json': b'{ "held": true, "custom_spacing": 1 }\n',
              queue / 'queued/capture_queued.json': b'{\n "queued": true\n}\n'}
    for path, value in tokens.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    frozen = {'code_commit': 'runtime_fixture_v2', 'code_root': str(tmp_path / 'frozen')}
    freeze = Mock(return_value=frozen)
    monkeypatch.setattr(repair, 'freeze_working_code', freeze)
    monkeypatch.setattr(repair, 'proposals', lambda actual_campaign: [{
        'big_task': big, 'small_task': small, 'batch_file': str(cpu_batch),
        'candidate_id': records[0]['candidate_id']}])
    calls = []

    def publish_record(record, actual_queue, **kwargs):
        assert actual_queue == queue
        journal = read_json(output / 'publication.json')
        assert journal['complete'] is False  # Durable journal precedes enqueue.
        assert len(journal['samples']) == 1 and not journal['samples'][0].get('job_id')
        calls.append((deepcopy(record), kwargs))
        job = 'capture_selected_fixture'
        write_json(queue / 'queued' / (job + '.json'), {'job_id': job, 'candidate_id': record['candidate_id']})
        write_json(queue / 'category_records' / (record['candidate_id'] + '.json'), {**record, 'job_id': job})
        return {'job_id': job, 'capture_submission_error': None}

    monkeypatch.setattr(repair, 'publish_record', publish_record)
    return SimpleNamespace(root=root, campaign=campaign, queue=queue, output=output, review=review,
        reviews_path=reviews_path, records=records, old_rows=old_rows, snapshot=snapshot,
        annotations=annotations, old_note=old_note, old_log=old_log, tokens=tokens,
        frozen=frozen, freeze=freeze, calls=calls, cpu_batch=cpu_batch, key=key,
        untouched_sample=untouched_sample)


def test_publish_enqueues_one_runtime_representative_and_archives_prior_feedback(publication):
    fixture = publication
    cpu_bytes = fixture.cpu_batch.read_bytes()
    result = repair.publish(fixture.campaign, fixture.queue)
    assert result['complete'] and result['other_queue_tokens_changed'] is False
    assert len(fixture.calls) == len(result['samples']) == 1
    fixture.freeze.assert_called_once_with(fixture.queue)
    assert fixture.calls[0][1]['priority'] == 1000
    for path, expected in fixture.tokens.items():
        assert path.read_bytes() == expected
    assert fixture.cpu_batch.read_bytes() == cpu_bytes
    runtime_batch = read_json(result['categories'][0]['new_batch_file'])
    assert len(runtime_batch['records']) == 3
    for old, new in zip(fixture.records, runtime_batch['records']):
        expected_revision = digest({'conversion_revision': old['revision'],
            'runtime': fixture.frozen['code_commit'], 'purpose': 'annotated_subclass_repair_20261015'})[:24]
        assert new['candidate_id'] == 'category_' + expected_revision != old['candidate_id']
        assert new['runtime_revision'] == fixture.frozen['code_commit']
        assert new['parent_candidate_id'] == old['candidate_id'].replace('_cpu_', '_old_')
        assert new['cpu_candidate_id'] == old['candidate_id']
    selected = result['samples'][0]
    assert selected['cpu_candidate_id'] == fixture.records[0]['candidate_id']
    repaired_failure = runtime_batch['records'][1]
    assert repaired_failure['status'] == 'generated' and repaired_failure['native_recheck_required']
    assert repaired_failure['native_recheck_evidence']['previous_candidate_id'] == 'category_old_b'
    assert repaired_failure['diagnostic_logs'] == [str(fixture.old_log)]
    assert repaired_failure['prior_feedback'][0]['notes'] == fixture.old_note['notes']
    assert repaired_failure['prior_feedback'][0]['previous_candidate_id'] == 'category_old_b'
    assert 'no approval is inherited' in repaired_failure['prior_feedback'][0]['scope'].lower()
    assert read_json(Path(repaired_failure['diagnostic_dir']) / 'record.json')['native_recheck_required']
    assert (Path(repaired_failure['diagnostic_dir']) / 'private_qa.json').is_symlink()
    assert read_json(result['prior_annotations_archived']) == fixture.annotations
    current_reviews = read_json(fixture.reviews_path)
    assert set(current_reviews['entries']) == {'category_unrelated'}
    assert [row['id'] for row in current_reviews['history']] == ['category_unrelated']
    assert read_json(fixture.output / 'catalogue_before_publication.json') == fixture.snapshot
    retired = read_json(fixture.review / 'retired_candidates.json')['candidates']
    assert set(retired) == {'category_old_a', 'category_old_b', 'category_old_c'}
    selection = read_json(fixture.output / 'samples.json')['samples']
    assert fixture.untouched_sample in selection
    assert read_json(fixture.queue / 'capture_sample_gate.json')['selected_candidate_ids'] == [
        selected['candidate_id'], 'category_keep']
    # Replaying a completed publication cannot enqueue or refreeze another revision.
    assert repair.publish(fixture.campaign, fixture.queue) == result
    assert len(fixture.calls) == fixture.freeze.call_count == 1


def test_cleanup_tracks_inflight_predecessor_absent_from_review(publication):
    fixture = publication
    revisions = read_json(fixture.review / 'category_revisions.json')
    revisions['categories'][fixture.key]['selected_sample_candidate_id'] = 'category_old_c'
    write_json(fixture.review / 'category_revisions.json', revisions)
    write_json(fixture.queue / 'category_records/category_old_c.json', {
        'job_id': 'capture_old_c', 'source_identity': fixture.records[2]['source_identity']})
    result = repair.publish(fixture.campaign, fixture.queue)
    assert 'capture_old_c' in result['cleanup_jobs']
    claim = fixture.queue / 'claimed/capture_old_c.json'
    write_json(claim, {'job_id': 'capture_old_c'})
    old = fixture.campaign / 'capture_artifacts/jobs/capture_old_c/attempts/01/global_diagonal.png'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'No image decoding is permitted.')
    receipt = old.parent / 'preview_complete.json'
    write_json(receipt, {'images': {'global_diagonal': str(old)}})
    write_json(fixture.queue / 'jobs/capture_old_c/status.json', {'attempts': 0, 'receipt': str(receipt)})
    write_json(fixture.queue / 'jobs/capture_selected_fixture/status.json', {'state': 'QUEUED'})
    assert not repair.cleanup(fixture.campaign, fixture.queue, result)['complete']
    assert old.exists()
    claim.unlink()
    assert repair.cleanup(fixture.campaign, fixture.queue, result)['complete']
    assert not old.exists() and receipt.exists()


def test_cleanup_removes_only_declared_old_job_pngs_after_terminal_replacement(tmp_path, monkeypatch):
    campaign, queue = tmp_path / 'campaign', tmp_path / 'queue'
    output = campaign / 'annotation_repair_20261015'
    output.mkdir(parents=True)
    monkeypatch.setattr(repair, 'ROOT', tmp_path / 'project')

    def artifact(job, name):
        path = campaign / 'capture_artifacts/jobs' / job / 'attempts/01' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('Deliberately invalid image bytes: ' + job + '/' + name).encode())
        return path

    removed = [artifact('old_done', name) for name in ('status.png', 'receipt.png', 'partial.png', 'failure.PNG')]
    undeclared = artifact('old_done', 'undeclared.png')
    jpeg = artifact('old_done', 'declared.jpg')
    current = artifact('new_done', 'head.png')
    active = artifact('old_active', 'head.png')
    waiting = artifact('old_waiting', 'head.png')
    source = tmp_path / 'source_assets/source.png'
    source.parent.mkdir()
    source.write_bytes(b'Original source asset bytes, not a valid image.')
    escape = removed[0].parent / 'source_link.png'
    escape.symlink_to(source)
    preserved = [undeclared, jpeg, current, active, waiting, source, escape]
    receipt = queue / 'jobs/old_done/attempts/01/preview_complete.json'
    write_json(receipt, {'images': {'head': str(removed[1]), 'current': str(current),
        'source': str(source), 'escape': str(escape), 'jpeg': str(jpeg)}})
    write_json(receipt.parent / 'partial_capture.json', {'images': {'head': str(removed[2])}})
    write_json(receipt.parent / 'capture_failure.json', {'partial_images': {'head': str(removed[3])}})
    write_json(queue / 'jobs/old_done/status.json', {'state': 'REVIEW_PENDING', 'attempts': 1,
        'receipt': str(receipt), 'partial_images': {'head': str(removed[0])}})
    for job, path in [('old_active', active), ('old_waiting', waiting)]:
        write_json(queue / 'jobs' / job / 'status.json', {'state': 'RUNNING', 'partial_images': {'head': str(path)}})
    write_json(queue / 'claimed/old_active.json', {'job_id': 'old_active'})
    write_json(queue / 'jobs/new_done/status.json', {'state': 'REVIEW_PENDING'})
    write_json(queue / 'jobs/new_waiting/status.json', {'state': 'RUNNING'})
    retained_metadata = []
    for job in ('old_done', 'old_active', 'old_waiting'):
        write_json(queue / 'jobs' / job / 'input/manifest.json', {'capture_signature': job + '_signature'})
        write_json(queue / 'cache' / (job + '_signature.json'), {'job_id': job})
    write_json(queue / 'cache/new_signature.json', {'job_id': 'new_done'})
    for name in ('native.log', 'input/task.yaml', 'input/scene.yaml', 'input/source.json'):
        path = queue / 'jobs/old_done' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('Retain immutable evidence: ' + name).encode())
        retained_metadata.append(path)
    retained_metadata.extend([receipt, receipt.parent / 'partial_capture.json',
                              receipt.parent / 'capture_failure.json', queue / 'jobs/old_done/input/manifest.json'])
    expected_metadata = {path: path.read_bytes() for path in retained_metadata}
    expected_images = {path: path.read_bytes() for path in preserved}
    expected_freed = sum(path.stat().st_size for path in removed)
    publication = {'samples': [{'candidate_id': 'replacement_done', 'job_id': 'new_done'},
                               {'candidate_id': 'replacement_waiting', 'job_id': 'new_waiting'}],
        'cleanup_jobs': {'old_done': {'replacement_candidate_id': 'replacement_done'},
                         'old_active': {'replacement_candidate_id': 'replacement_done'},
                         'old_waiting': {'replacement_candidate_id': 'replacement_waiting'}}}
    original_open, original_unlink = Path.open, Path.unlink

    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() == '.png':
            forbid_discovery()
        return original_open(path, *args, **kwargs)

    def journal_before_unlink(path, *args, **kwargs):
        if path.suffix.lower() == '.png':
            rows = [json.loads(line) for line in (output / 'obsolete_cache_removal.jsonl').read_text().splitlines()]
            assert any(row['path'] == str(path.resolve()) and row['old_job_id'] == 'old_done'
                       and row['replacement'] == 'replacement_done' for row in rows)
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(PIL.Image, 'open', forbid_discovery)
    monkeypatch.setattr(Path, 'open', metadata_only)
    monkeypatch.setattr(Path, 'glob', forbid_discovery)
    monkeypatch.setattr(Path, 'rglob', forbid_discovery)
    monkeypatch.setattr(Path, 'iterdir', forbid_discovery)
    monkeypatch.setattr(Path, 'unlink', journal_before_unlink)
    result = repair.cleanup(campaign, queue, publication)
    assert result['completed_jobs'] == ['old_done'] and result['complete'] is False
    assert result['removed_png_files'] == len(removed) and result['freed_bytes'] == expected_freed
    assert result['source_assets_tasks_and_diagnostic_logs_preserved'] is True
    assert all(not path.exists() for path in removed)
    for path, expected in expected_images.items():
        with original_open(path, 'rb') as stream:
            assert stream.read() == expected
    assert escape.is_symlink()
    for path, expected in expected_metadata.items():
        assert path.read_bytes() == expected
    assert not (queue / 'cache/old_done_signature.json').exists()
    assert all((queue / 'cache' / name).exists() for name in (
        'old_active_signature.json', 'old_waiting_signature.json', 'new_signature.json'))
    journal = (output / 'obsolete_cache_removal.jsonl').read_text()
    assert len(journal.splitlines()) == len(removed)
    again = repair.cleanup(campaign, queue, publication)
    assert again['removed_png_files'] == len(removed) and again['freed_bytes'] == expected_freed
    assert (output / 'obsolete_cache_removal.jsonl').read_text() == journal


def test_catalogue_keeps_generated_native_recheck_in_diagnostics_without_discovery(publication, monkeypatch):
    fixture = publication
    result = repair.publish(fixture.campaign, fixture.queue)
    runtime_batch = read_json(result['categories'][0]['new_batch_file'])
    repaired_failure = runtime_batch['records'][1]
    publisher = catalogue.CataloguePublisher(fixture.campaign, fixture.queue, fixture.review)
    publisher.completed['category_old_a'] = deepcopy(fixture.old_rows[0])
    publisher.failed['category_old_b'] = deepcopy(fixture.old_rows[1])
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        if path.suffix.lower() == '.png':
            forbid_discovery()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', metadata_only)
    monkeypatch.setattr(Path, 'glob', forbid_discovery)
    monkeypatch.setattr(Path, 'rglob', forbid_discovery)
    monkeypatch.setattr(Path, 'iterdir', forbid_discovery)
    monkeypatch.setattr(PIL.Image, 'open', forbid_discovery)
    publisher.consume_publication()
    row = publisher.failed[repaired_failure['candidate_id']]
    assert row['conversion_result_status'] == 'generated'
    assert row['instance_status'] == 'native_recheck_pending'
    assert row['native_recheck_required'] and row['prior_feedback'][0]['notes'] == fixture.old_note['notes']
    assert row['failure_stage'] == 'previous_revision_recheck'
    assert 'No fitting native support.' in row['failure']
    assert row['images'] == [] and not row['native_capture_complete'] and not row['physical_grasp_verified']
    snapshot = publisher.tick()
    assert not snapshot['scenes']
    assert [item['id'] for item in snapshot['diagnostics']] == [repaired_failure['candidate_id']]
    assert all(item['id'] not in ('category_old_a', 'category_old_b', 'category_old_c')
               for item in snapshot['scenes'] + snapshot['diagnostics'])
    assert snapshot['directory_scanning'] is False and snapshot['image_validation'] is False
    assert snapshot['retired_candidate_count'] == 3
    assert 'PRIVATE_QA_CANARY' not in json.dumps(snapshot)

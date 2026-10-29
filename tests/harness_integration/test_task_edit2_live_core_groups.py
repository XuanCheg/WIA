"""Check source-grouped Review without simulation or a running service.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q
 tests/test_task_edit2_live_core_groups.py tests/test_scene_review_live.py
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from wia.runtime.scene_review import read_json, write_json
from wia.runtime.scene_review_live import LiveSceneReview


def alias(number):
    return 'category_' + format(number, 'x')


@pytest.fixture
def grouped(tmp_path):
    queue = tmp_path / 'queue'
    snapshot_path = tmp_path / 'snapshot.json'
    store = LiveSceneReview(None, tmp_path / 'reviews', queue_root=queue,
                            catalogue_file=snapshot_path)
    snapshot = {'scenes': [], 'diagnostics': [], 'core_groups': [],
                'additional_variant_ids': [], 'categories': []}

    def card(key, source, *, complete=False, baseline=False):
        return {'id': key, 'candidate_id': key, 'fingerprint': 'fp-' + key,
                'category': source.split('/')[0], 'subcategory': source.split('/')[1],
                'scene': 'house', 'room': 'room', 'source': source,
                'source_identity': source, 'native_capture_complete': complete,
                'original_baseline': baseline, 'warning_codes': [], 'images': []}

    for number in range(161):
        source = ('Category A' if number % 2 else 'Category B') + '/Subclass/house/room/q_' + str(number) + '.json'
        key = alias(1000 + number)
        directory = tmp_path / 'baselines' / str(number)
        write_json(directory / 'record.json', {'job_id': 'old_missing_' + str(number),
                   'source_identity': source, 'original_baseline': True,
                   'diagnostic_images': []})
        original = card(key, source, baseline=True)
        original['diagnostic_dir'] = str(directory)
        snapshot['diagnostics'].append(original)
        snapshot['core_groups'].append({'id': 'core_' + format(number, 'x'),
             'source_identity': source, 'original_candidate_id': key,
             'original_variant_id': key, 'default_variant_id': key,
             'versions': [{'id': key, 'label': 'Original source', 'role': 'original_baseline',
                           'capture_state': 'historical', 'task_file': 'old_task.yaml',
                           'scene_file': 'old_scene.yaml'}]})
    # One actual composite card belongs to two different source categories.
    failed = card(alias(2000), snapshot['core_groups'][0]['source_identity'])
    failed.update(failure_stage='interaction_setup', failure='Latest support rejected.',
                  category_tags=[{'big_task': 'Foreign composite tag', 'small_task': 'Foreign'}],
                  composite={'member_count': 2, 'public_goals': []})
    snapshot['diagnostics'].append(failed)
    write_json(queue / 'category_records' / (failed['id'] + '.json'),
               {'job_id': 'new_failed', 'failure_stage': 'interaction_setup',
                'failure_reason': 'Latest support rejected.'})
    for number in (0, 1):
        group = snapshot['core_groups'][number]
        group['default_variant_id'] = failed['id']
        group['versions'].append({'id': failed['id'], 'label': 'Latest revised composite',
                                 'role': 'explicit_derivative', 'capture_state': 'FAILED',
                                 'task_file': 'new_task.yaml', 'scene_file': 'new_scene.yaml'})
    ready = card(alias(2001), snapshot['core_groups'][2]['source_identity'], complete=True)
    snapshot['scenes'].append(ready)
    image = tmp_path / 'current.png'
    image.write_bytes(b'Native screenshot bytes are not decoded by metadata Review.')
    receipt = tmp_path / 'receipt.json'
    write_json(receipt, {'complete': True, 'images': {'head': str(image)}})
    write_json(queue / 'category_records' / (ready['id'] + '.json'), {'job_id': 'new_ready'})
    write_json(queue / 'jobs/new_ready/status.json', {'state': 'REVIEW_PENDING', 'receipt': str(receipt)})
    group = snapshot['core_groups'][2]
    group['default_variant_id'] = ready['id']
    group['versions'].append({'id': ready['id'], 'label': 'Current revision',
                             'role': 'explicit_derivative', 'capture_state': 'REVIEW_PENDING',
                             'task_file': 'ready_task.yaml', 'scene_file': 'ready_scene.yaml'})
    for number in range(10):
        key = alias(3000 + number)
        snapshot['additional_variant_ids'].append(key)
        snapshot['diagnostics'].append(card(key, 'Additional/Extra/house/room/' + str(number) + '.json'))
    write_json(snapshot_path, snapshot)
    return store, snapshot, image


def test_complete_core_source_coverage_default_counts_and_explicit_scopes(grouped):
    store, snapshot, _ = grouped
    listing = store.catalogue()
    assert listing['core_grouped'] is True
    assert listing['core_group_count'] == listing['total'] == listing['matched'] == 161
    assert listing['additional_variant_count'] == 10
    assert {row['source'] for row in listing['scenes']} == {
        group['source_identity'] for group in snapshot['core_groups']}
    assert {row['id'] for row in listing['scenes']} == {group['id'] for group in snapshot['core_groups']}
    assert all(not row['images'] for row in listing['scenes']) and not store.images
    assert sum(group['source_count'] for group in listing['categories']) == 161
    assert {(group['big_task'], group['small_task']) for group in listing['categories']} == {
        ('Category A', 'Subclass'), ('Category B', 'Subclass')}
    assert store.catalogue(filters={'scope': 'all'})['matched'] == 161
    assert store.catalogue(filters={'scope': 'completed'})['matched'] == 1
    assert store.catalogue(filters={'scope': 'diagnostics'})['matched'] == 160
    additional = store.catalogue(filters={'scope': 'additional'})
    assert {row['id'] for row in additional['scenes']} == set(snapshot['additional_variant_ids'])
    assert sum(group['source_count'] for group in additional['categories']) == 161
    assert store.catalogue(page=0, page_size=48)['matched'] == 161
    assert len(store.catalogue(page=0, page_size=48)['scenes']) == 48


def test_latest_failure_and_default_alias_reviews_survive_group_annotation(grouped):
    store, snapshot, _ = grouped
    key = alias(2000)
    store.save_review(key, {'decision': 'needs_fix', 'notes': 'Keep actual version feedback.',
                           'revision': 0, 'fingerprint': 'fp-' + key})
    listing = store.catalogue(filters={'failure_stage': 'interaction_setup'})
    assert listing['matched'] == 2
    for row in listing['scenes']:
        assert row['failure'] == 'Latest support rejected.'
        assert row['review']['notes'] == 'Keep actual version feedback.'
        assert row['decision'] == 'needs_fix'
        assert row['core_group']['default_variant_id'] == key
    canonical = store.catalogue(filters={'id': key})['scenes']
    assert len(canonical) == 1 and canonical[0]['id'] == key
    assert canonical[0]['category_tags'][-1]['big_task'] == 'Foreign composite tag'
    assert store.scene_detail(key)['id'] == key
    assert store.scene_detail('core_0')['failure'] == 'Latest support rejected.'
    assert store.scene_detail('core_1')['id'] == key


def test_variant_switching_is_member_only_and_saves_are_canonical(grouped):
    store, snapshot, _ = grouped
    group = snapshot['core_groups'][2]
    current = store.scene_detail(group['id'])
    assert current['id'] == group['default_variant_id']
    assert current['core_group']['active_variant_id'] == current['id']
    assert current['core_group']['versions'] == group['versions']
    original = store.scene_detail(group['id'], variant_id=group['original_variant_id'])
    assert original['id'] == group['original_variant_id']
    assert original['core_group']['active_variant_id'] == original['id']
    assert original['layout_approval_available']
    for wrong in (alias(2000), snapshot['additional_variant_ids'][0], '../outside', 'core_1', ''):
        with pytest.raises(KeyError):
            store.scene_detail(group['id'], variant_id=wrong)
    with pytest.raises(KeyError):
        store.scene_detail(current['id'], variant_id=original['id'])
    payload = {'decision': 'needs_fix', 'notes': 'Original only.', 'revision': 0,
               'fingerprint': original['fingerprint']}
    store.save_review(original['id'], payload)
    assert store.scene_detail(group['id'])['decision'] == 'unreviewed'
    assert store.scene_detail(group['id'], variant_id=original['id'])['decision'] == 'needs_fix'
    with pytest.raises(KeyError):
        store.save_review(group['id'], payload)
    assert set(read_json(store.review_file)['entries']) == {original['id']}
    store.save_review(current['id'], {'decision': 'approved', 'notes': 'Current layout only.',
                      'revision': 0, 'fingerprint': current['fingerprint']})
    assert store.scene_detail(group['id'])['human_approved']
    assert store.scene_detail(group['id'], variant_id=original['id'])['decision'] == 'needs_fix'


def test_baseline_context_only_existing_declared_paths_without_reads_mutating_state(grouped, monkeypatch, tmp_path):
    store, snapshot, _ = grouped
    group = snapshot['core_groups'][3]
    row = next(row for row in snapshot['diagnostics'] if row['id'] == group['original_variant_id'])
    record_path = Path(row['diagnostic_dir']) / 'record.json'
    existing = tmp_path / 'official_original.png'
    existing.write_bytes(b'Published original context.')
    missing = tmp_path / 'deleted_previous.png'
    record = read_json(record_path)
    record['diagnostic_images'] = [
        {'path': str(existing), 'view': 'original', 'kind': 'source', 'label': 'Official source'},
        {'path': str(missing), 'view': 'head', 'kind': 'historical'}]
    write_json(record_path, record)
    # The publisher may flatten an explicitly declared official image into the
    # summary while retaining an untouched old diagnostic record on disk.
    publisher_context = tmp_path / 'publisher_original.png'
    publisher_context.write_bytes(b'Explicit publisher source context.')
    row['diagnostic_images'] = [{'path': str(publisher_context), 'view': 'source_original',
                                 'kind': 'historical', 'label': 'Original published image'}]
    write_json(store.catalogue_file, snapshot)
    before = record_path.read_bytes()
    def no_scan(*args, **kwargs):
        raise AssertionError('No directory discovery is allowed.')
    for method in ('glob', 'rglob', 'iterdir'):
        monkeypatch.setattr(Path, method, no_scan)
    detail = store.scene_detail(group['id'])
    assert detail['original_baseline'] is True
    assert not detail['native_capture_complete'] and detail['layout_approval_available']
    assert len(detail['images']) == 2
    assert all(item['kind'] == 'historical' for item in detail['images'])
    assert detail['images'][0]['label'].startswith('Original/historical context')
    assert missing.name not in json.dumps(detail['images'])
    assert record_path.read_bytes() == before and not store.review_file.exists()
    assert not (store.queue_root / 'jobs').joinpath(record['job_id']).exists()
    saved = store.save_review(detail['id'], {'decision': 'approved', 'notes': '', 'revision': 0,
                                           'fingerprint': detail['fingerprint']})
    assert saved['approval_scope'] == 'layout'
    assert saved['approval_evidence']['original_baseline']
    assert not saved['approval_evidence']['native_capture_complete']
    existing.unlink()
    publisher_context.unlink()
    assert not store.scene_detail(group['id'])['images']
    assert not any(key[0] == detail['id'] for key in store.images)


def test_deleted_capture_remains_missing_after_manual_approval(grouped):
    store, _, image = grouped
    detail = store.scene_detail('core_2')
    assert detail['native_capture_complete'] and len(detail['images']) == 1
    image.unlink()
    detail = store.scene_detail('core_2')
    assert not detail['images'] and not detail['native_capture_complete']
    assert detail['layout_approval_available']
    assert detail['failure_stage'] == 'capture_images_missing'
    assert not any(key[0] == detail['id'] for key in store.images)
    saved = store.save_review(detail['id'], {'decision': 'approved', 'notes': '', 'revision': 0,
                                           'fingerprint': detail['fingerprint']})
    assert not saved['approval_evidence']['native_capture_complete']
    assert not store.scene_detail(detail['id'])['native_capture_complete']


def test_legacy_snapshot_retains_completed_default_and_no_additional_scope(grouped):
    store, snapshot, _ = grouped
    snapshot = deepcopy(snapshot)
    snapshot.pop('core_groups')
    snapshot.pop('additional_variant_ids')
    write_json(store.catalogue_file, snapshot)
    listing = store.catalogue()
    assert listing['core_grouped'] is False and listing['core_group_count'] == 0
    assert [row['id'] for row in listing['scenes']] == [alias(2001)]
    with pytest.raises(ValueError, match='Unknown catalogue scope'):
        store.catalogue(filters={'scope': 'additional'})


def test_corev2_unique_selection_and_issues(grouped, tmp_path):
    store, snapshot, _ = grouped
    key = snapshot['core_groups'][0]['default_variant_id']
    manifest = tmp_path / 'corev2.json'
    write_json(manifest, {'tasks': [{'candidate_id': key, 'setup_status': 'unverified',
                                   'issues': ['execution_not_qualified']}]})
    store.task_list_file = manifest
    selected = store.catalogue(filters={'scope': 'corev2'})
    assert selected['total'] == 1
    assert selected['capture_queue']['candidate_count'] == 1
    assert selected['capture_queue']['captured'] == 0
    assert selected['scenes'][0]['id'] == key
    assert selected['scenes'][0]['corev2']['setup_status'] == 'unverified'
    assert 'execution_not_qualified' in selected['scenes'][0]['warning_codes']
    assert sum(c['source_count'] for c in selected['categories']) == 1
    assert store.catalogue(filters={'scope': 'all'})['total'] == 161
    write_json(manifest, {'tasks': [{'candidate_id': 'category_deadbeef', 'issues': []}]})
    with pytest.raises(ValueError, match='missing from Review'):
        store.catalogue(filters={'scope': 'corev2'})


def test_corev2_capture_revision_preserves_card_and_old_evidence(grouped, tmp_path, monkeypatch):
    store, snapshot, image = grouped
    key = snapshot['core_groups'][3]['original_variant_id']
    previous = next(row for row in snapshot['diagnostics'] if row['id'] == key)
    old_record = Path(previous['diagnostic_dir']) / 'record.json'
    old_bytes, snapshot_bytes = old_record.read_bytes(), store.catalogue_file.read_bytes()
    store.save_review(key, {'decision': 'needs_fix', 'notes': 'Retain this history.',
                           'revision': 0, 'fingerprint': previous['fingerprint']})
    capture_id = alias(9000)
    task, scene = tmp_path / 'task.yaml', tmp_path / 'scene.yaml'
    task.write_text('task'); scene.write_text('scene')
    store.task_list_file = tmp_path / 'corev2.json'
    write_json(store.task_list_file, {'tasks': [{'candidate_id': key,
        'capture_candidate_id': capture_id, 'task_file': str(task), 'scene_file': str(scene),
        'issues': ['execution_not_qualified']}]})
    path = store.queue_root / 'category_records' / (capture_id + '.json')
    record = {'candidate_id': capture_id, 'corev2_candidate_id': key,
        'task_edit2_capture_campaign': True, 'job_id': 'repaired',
        'task_file': str(task), 'scene_file': str(scene), 'capture_signature': 'repaired-signature'}
    write_json(path, record)
    receipt_path = tmp_path / 'repaired_receipt.json'
    write_json(receipt_path, {'complete': True, 'images': {'head': str(image)}})
    write_json(store.queue_root / 'jobs/repaired/status.json',
               {'state': 'REVIEW_PENDING', 'receipt': str(receipt_path)})
    monkeypatch.setattr('wia.runtime.scene_review_live.verified_task_edit2_capture',
                        lambda queue, record, status, receipt: (True, None))
    listing = store.catalogue(filters={'scope': 'corev2'})
    assert listing['total'] == 1 and listing['scenes'][0]['id'] == key
    detail = store.scene_detail(key)
    assert detail['capture']['job_id'] == 'repaired'
    assert detail['native_capture_complete'] and not detail['original_baseline']
    assert detail['images'][0]['kind'] == 'native'
    assert detail['review_stale'] and detail['review']['notes'] == 'Retain this history.'
    assert store.catalogue(filters={'scope': 'all'})['total'] == 161
    assert old_record.read_bytes() == old_bytes and store.catalogue_file.read_bytes() == snapshot_bytes
    record['corev2_candidate_id'] = alias(9999)
    write_json(path, record)
    with pytest.raises(ValueError, match='does not match'):
        store.scene_detail(key)


@pytest.mark.parametrize('new_card', [False, True])
def test_corev2_explicit_canonical_capture_and_new_membership(grouped, tmp_path, monkeypatch, new_card):
    store, snapshot, image = grouped
    key = alias(9001) if new_card else snapshot['core_groups'][3]['original_variant_id']
    task, scene = tmp_path / 'selected_task.yaml', tmp_path / 'selected_scene.yaml'
    task.write_text('selected task')
    scene.write_text('selected scene')
    source = 'Cognitive Mapping/Connectivity/house/full_scene/q_000.json'
    record = {'candidate_id': key, 'task_edit2_capture_campaign': True, 'job_id': 'selected',
        'task_file': str(task), 'scene_file': str(scene), 'capture_signature': 'selected-signature',
        'source_identity': source, 'big_task': 'Cognitive Mapping', 'small_task': 'Connectivity',
        'status': 'generated', 'robot_task_public': {'objective': 'Explore five room pairs.'}}
    write_json(store.queue_root / 'category_records' / (key + '.json'), record)
    receipt = tmp_path / 'selected_receipt.json'
    write_json(receipt, {'complete': True, 'images': {'head': str(image)}})
    write_json(store.queue_root / 'jobs/selected/status.json',
               {'state': 'REVIEW_PENDING', 'receipt': str(receipt)})
    store.task_list_file = tmp_path / 'corev2.json'
    write_json(store.task_list_file, {'tasks': [{'candidate_id': key, 'capture_candidate_id': key,
        'task_file': str(task), 'scene_file': str(scene), 'issues': ['execution_not_qualified']}]})
    monkeypatch.setattr('wia.runtime.scene_review_live.verified_task_edit2_capture',
                        lambda *args: (True, None))
    selected = store.catalogue(filters={'scope': 'corev2'})
    assert selected['total'] == 1
    detail = store.scene_detail(key)
    assert detail['native_capture_complete'] and not detail['original_baseline']
    assert selected['capture_queue']['captured'] == 1
    assert detail['layout_approval_available'] and not detail['physical_qualified']
    if new_card:
        assert detail['objective'] == record['robot_task_public']['objective']
    assert store.catalogue(filters={'scope': 'all'})['total'] == 161
    assert read_json(store.catalogue_file) == snapshot
    write_json(store.queue_root / 'jobs/selected/status.json', {'state': 'RUNNING'})
    monkeypatch.setattr('wia.runtime.scene_review_live.verified_task_edit2_capture',
                        lambda *args: (False, 'Capture is still running.'))
    pending = store.catalogue(filters={'scope': 'corev2'})['capture_queue']
    assert pending['candidate_count'] == 1 and pending['running'] == 1
    assert pending['captured'] == pending['failed'] == 0
    assert store.catalogue(filters={'scope': 'all'})['capture_queue'] == {}

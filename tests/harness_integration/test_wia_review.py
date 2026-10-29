"""Check CPU-only WIA snapshot review without opening a server or simulator.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_review.py
All declared fixture files and annotations remain under pytest's temporary directory.
"""
from copy import deepcopy
import csv
import io
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

import pytest

from wia.runtime.scene_review import ReviewConflict, read_json, write_json
from wia.runtime.wia_review import WiaReview


def variant(key, *, image, fingerprint=None):
    value = {'id': key, 'label': key.title(), 'task_file': f'tasks/{key}.yaml',
        'scene_file': f'scenes/{key}.yaml', 'objective': 'Lift the banana into the bowl.',
        'success_criteria': 'Released insertion.', 'constraints': ['Public controls only.'],
        'environment': {'room': 'utility_room_0'}, 'status': 'definition_only',
        'evidence_summary': {'initial_layout': 'captured', 'robot_reference': 'not_verified'},
        'native_evidence_status': 'not_verified',
        'evidence': [{'label': 'Declared trace', 'path': f'evidence/{key}.json'}],
        'images': [{'id': 'overview', 'label': 'Global overview', 'role': 'human_layout', 'path': str(image)}]}
    if fingerprint is not None:
        value['fingerprint'] = fingerprint
    return value


@pytest.fixture
def review(tmp_path):
    first = tmp_path / 'first.png'
    second = tmp_path / 'second.png'
    first.write_bytes(b'\x89PNG\r\n\x1a\nfirst declared fixture')
    second.write_bytes(b'\x89PNG\r\n\x1a\nsecond declared fixture')
    snapshot = {'schema': 'wia_review_snapshot_v1', 'generated_at': '2026-10-23T12:00:00+00:00',
        'title': 'WIA bench early feedback',
        'summary': {'proposal_count': 2, 'authored_variant_count': 1,
                    'reference_success_row_count': 0, 'unbuilt_row_count': 1},
        'rows': [
            {'id': 'T25', 'document_alias': 'T25', 'primary_family': 'Containment',
             'primary_subclass': 'Rigid insertion', 'title': 'Banana and bowls',
             'implementation_state': 'selected_definition', 'status': 'definition_only',
             'summary': 'Configured geometry; no verified robot reference.',
             'variants': [variant('bench', image=first, fingerprint='bench-definition-v1'),
                          variant('original_selected', image=second, fingerprint='original-definition-v1')]},
            {'id': 'T60', 'document_alias': 'T60', 'primary_family': 'Temporal',
             'primary_subclass': 'Unbuilt comparison', 'title': 'Pending temporal proposal',
             'implementation_state': 'unbuilt', 'status': 'unbuilt', 'summary': 'Early design feedback wanted.',
             'variants': [{'id': 'proposal', 'label': 'Proposal', 'task_file': None, 'scene_file': None,
                'objective': 'Compare the two observations.', 'success_criteria': 'A declared comparison.',
                'constraints': [], 'environment': {}, 'status': 'unbuilt', 'evidence_summary': 'None yet.',
                'evidence': [], 'images': []}]},
        ]}
    snapshot_file = tmp_path / 'snapshot.json'
    write_json(snapshot_file, snapshot)
    return WiaReview(snapshot_file, tmp_path / 'annotations'), snapshot, snapshot_file


def publish(path, snapshot):
    previous = path.stat().st_mtime_ns
    write_json(path, snapshot)
    stamp = max(previous + 1, path.stat().st_mtime_ns)
    os.utime(path, ns=(stamp, stamp))


def payload(detail, decision='approved', notes='Check the layout clearance.'):
    return {'variant_id': detail['selected_variant_id'], 'fingerprint': detail['fingerprint'],
            'revision': detail['review']['revision'], 'decision': decision, 'notes': notes}


def image_key(image):
    parts = urlsplit(image['url']).path.split('/')
    return parts[-2], parts[-1]


def test_catalogue_is_metadata_only_and_never_registers_or_reads_images(review, monkeypatch):
    store, _, _ = review

    def forbidden(*args, **kwargs):
        raise AssertionError('Catalogue must not resolve or read image files.')

    monkeypatch.setattr(store, '_image_path', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    result = store.catalogue()
    assert result['total'] == 2 and result['page'] == 0 and result['page_size'] == 100
    assert result['title'] == 'WIA bench early feedback'
    assert store.images == {}
    for row in result['scenes']:
        assert 'variants' not in row and 'images' not in row
        assert row['decision'] == 'unreviewed' and row['review']['revision'] == 0
        assert row['review_scope'] == 'design_layout_only'
    assert '.png' not in json.dumps(result)
    assert result['summary']['reference_success_row_count'] == 0
    assert result['summary']['proposal_count'] == 2
    assert store.catalogue(filters={}, page_size=48)['page_size'] == 100
    assert store.catalogue(filters={'page_size': '48'}, page_size=48)['page_size'] == 48


def test_pending_rows_filter_and_page_without_requiring_evidence_or_task_files(review):
    store, _, _ = review
    pending = store.catalogue(filters={'family': 'Temporal', 'status': 'unbuilt', 'review': 'unreviewed'})
    assert [item['id'] for item in pending['scenes']] == ['T60']
    assert store.catalogue(filters={'q': 'BANANA'})['scenes'][0]['id'] == 'T25'
    assert store.catalogue(page=1, page_size=1)['scenes'][0]['id'] == 'T60'
    assert store.catalogue(page=2, page_size=1)['scenes'] == []
    detail = store.scene_detail('T60')
    assert detail['selected_variant_id'] == 'proposal' and detail['images'] == []
    saved = store.save_review('T60', payload(detail, 'needs_fix', 'Specify the visual change.'))
    assert saved['decision'] == 'needs_fix' and saved['native_certification'] is False
    assert store.catalogue(filters={'review': 'needs_fix'})['scenes'][0]['id'] == 'T60'


def test_detail_only_registers_selected_declared_images_and_preserves_public_metadata(review):
    store, snapshot, _ = review
    first = store.scene_detail('T25')
    assert first['selected_variant_id'] == 'bench'
    assert first['selected_variant']['objective'] == snapshot['rows'][0]['variants'][0]['objective']
    assert first['selected_variant']['native_evidence_status'] == 'not_verified'
    assert first['selected_variant']['evidence'] == snapshot['rows'][0]['variants'][0]['evidence']
    assert first['images'] == first['selected_variant']['images']
    assert 'path' not in first['images'][0]
    assert len(store.images) == 1
    assert store.images[image_key(first['images'][0])].name == 'first.png'
    assert all('images' not in item and 'objective' not in item for item in first['variants'])
    second = store.scene_detail('T25', variant_id='original_selected')
    assert second['images'][0]['id'] == first['images'][0]['id'] == 'overview'
    assert second['images'][0]['url'] != first['images'][0]['url']
    assert store.images[image_key(second['images'][0])].name == 'second.png'
    assert store.images[image_key(first['images'][0])].name == 'first.png'


def test_annotations_persist_per_variant_and_never_upgrade_native_status(review):
    store, snapshot, path = review
    initial_snapshot = path.read_bytes()
    first = store.scene_detail('T25')
    first_record = store.save_review('T25', payload(first, 'needs_fix', 'Move the pickup support.'))
    second = store.scene_detail('T25', variant_id='original_selected')
    assert second['decision'] == 'unreviewed' and second['review']['revision'] == 0
    second_record = store.save_review('T25', payload(second))
    reopened = WiaReview(path, store.output)
    assert reopened.scene_detail('T25')['review'] == first_record
    original = reopened.scene_detail('T25', variant_id='original_selected')
    assert original['review'] == second_record
    assert original['status'] == 'definition_only'
    assert original['selected_variant']['native_evidence_status'] == 'not_verified'
    assert reopened.catalogue()['summary']['reference_success_row_count'] == 0
    assert path.read_bytes() == initial_snapshot
    assert not store.review_file.with_name('reviews.json.tmp').exists()
    records = read_json(store.review_file)
    assert set(records['entries']['T25']) == {'bench', 'original_selected'}


def test_note_only_updates_preserve_decision_and_history_and_check_revision(review):
    store, _, _ = review
    first = store.scene_detail('T25')
    store.save_review('T25', payload(first, 'needs_fix', 'Old note.'))
    current = store.scene_detail('T25')
    note = payload(current, notes='New note.\nSecond line.')
    del note['decision']
    saved = store.save_review('T25', note)
    assert saved['decision'] == 'needs_fix' and saved['revision'] == 2
    assert saved['notes'] == 'New note.\nSecond line.'
    assert store.scene_detail('T25')['review_history'][0]['notes'] == 'Old note.'
    with pytest.raises(ReviewConflict, match='another tab'):
        store.save_review('T25', note)
    assert store.scene_detail('T25', variant_id='original_selected')['review_history'] == []


def test_threaded_revision_check_has_only_one_successful_writer(review):
    store, _, _ = review
    post = payload(store.scene_detail('T25'))
    barrier = threading.Barrier(2)
    results = []

    def save():
        barrier.wait()
        try:
            store.save_review('T25', post)
            results.append('saved')
        except ReviewConflict:
            results.append('conflict')

    threads = [threading.Thread(target=save) for _ in range(2)]
    for worker in threads:
        worker.start()
    for worker in threads:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert sorted(results) == ['conflict', 'saved']
    assert store.scene_detail('T25')['review']['revision'] == 1


def test_snapshot_status_and_evidence_refresh_preserves_reviews_without_requalifying(review):
    store, snapshot, path = review
    first = store.scene_detail('T25')
    first_image = image_key(first['images'][0])
    store.save_review('T25', payload(first))
    updated = deepcopy(snapshot)
    updated['generated_at'] = '2026-10-23T12:10:00+00:00'
    updated['rows'][0]['status'] = 'capture_pending'
    updated['rows'][0]['variants'][0]['evidence_summary'] = 'A newer pending run.'
    publish(path, updated)
    catalogue = store.catalogue()
    assert catalogue['generated_at'] == updated['generated_at']
    assert catalogue['scenes'][0]['status'] == 'capture_pending'
    assert catalogue['scenes'][0]['decision'] == 'approved'
    assert not catalogue['scenes'][0]['review_stale']
    assert store.images[first_image].name == 'first.png'
    assert len(store.images) == 1  # Refresh did not register the unselected original variant.
    detail = store.scene_detail('T25')
    assert detail['selected_variant']['evidence_summary'] == 'A newer pending run.'
    assert detail['review']['notes'] == 'Check the layout clearance.'
    assert detail['native_certification'] is False


def test_definition_refresh_marks_only_selected_variant_review_stale(review):
    store, snapshot, path = review
    old = store.scene_detail('T25')
    old_image = image_key(old['images'][0])
    store.save_review('T25', payload(old, notes='Original layout note.'))
    second = store.scene_detail('T25', variant_id='original_selected')
    original_image = image_key(second['images'][0])
    store.save_review('T25', payload(second, 'skipped'))
    updated = deepcopy(snapshot)
    updated['rows'][0]['variants'][0].update(fingerprint='bench-definition-v2', objective='Changed public operation.')
    publish(path, updated)
    current = store.scene_detail('T25')
    assert store.images.get(old_image) is None
    assert store.images[original_image].name == 'second.png'
    assert current['review_stale'] and current['decision'] == 'unreviewed'
    assert current['review']['decision'] == 'approved' and current['review']['notes'] == 'Original layout note.'
    assert not store.scene_detail('T25', variant_id='original_selected')['review_stale']
    with pytest.raises(ReviewConflict, match='definition changed'):
        store.save_review('T25', payload(old))
    note = payload(current, notes='Reviewed new definition; still pending.')
    del note['decision']
    saved = store.save_review('T25', note)
    assert saved['decision'] == 'unreviewed'
    assert store.scene_detail('T25')['review_history'][0]['fingerprint'] == 'bench-definition-v1'


def test_fallback_fingerprint_tracks_declared_definitions_but_not_run_metadata(review):
    store, snapshot, path = review
    original = deepcopy(snapshot)
    original['rows'][0]['variants'][0].pop('fingerprint')
    publish(path, original)
    first = store.scene_detail('T25')
    store.save_review('T25', payload(first))
    revised = deepcopy(original)
    revised['rows'][0]['variants'][0].update(status='new_run_pending', notes='New run note.',
                                            evidence_summary='New evidence.')
    revised['rows'][0]['variants'][0]['images'][0]['label'] = 'New image label'
    publish(path, revised)
    assert store.scene_detail('T25')['fingerprint'] == first['fingerprint']
    assert store.scene_detail('T25')['review_stale'] is False
    revised['rows'][0]['variants'][0]['environment']['room'] = 'different_room'
    publish(path, revised)
    assert store.scene_detail('T25')['review_stale'] is True


def test_only_declared_pngs_are_mapped_and_removed_declarations_are_revoked(review, tmp_path):
    store, snapshot, path = review
    text = tmp_path / 'not-an-image.txt'
    text.write_text('UNDECLARED_FILE_CONTENT')
    undeclared = tmp_path / 'undeclared.png'
    undeclared.write_bytes(b'undeclared fixture')
    selected = snapshot['rows'][0]['variants'][0]
    selected['images'].extend([
        {'id': 'text', 'path': str(text)}, {'id': 'missing', 'path': str(tmp_path / 'missing.png')},
        {'id': 'home', 'path': '/home/never-read-wia-fixture.png'},
    ])
    publish(path, snapshot)
    detail = store.scene_detail('T25')
    assert [item['id'] for item in detail['images']] == ['overview']
    assert len(detail['image_warnings']) == 3
    assert undeclared not in store.images.values() and text not in store.images.values()
    old_key = image_key(detail['images'][0])
    assert store.images.get(('T25', '../not-an-image.txt')) is None
    selected['images'] = []
    publish(path, snapshot)
    assert store.images.get(old_key) is None


def test_relative_image_paths_use_the_snapshot_directory(review):
    store, snapshot, path = review
    snapshot['rows'][0]['variants'][0]['images'][0]['path'] = 'first.png'
    publish(path, snapshot)
    detail = store.scene_detail('T25')
    assert store.images[image_key(detail['images'][0])] == path.parent / 'first.png'


def test_changed_declared_path_revokes_old_url_even_with_the_same_definition_fingerprint(review):
    store, snapshot, path = review
    first = store.scene_detail('T25')
    original_key = image_key(first['images'][0])
    snapshot['rows'][0]['variants'][0]['images'][0]['path'] = str(path.parent / 'second.png')
    publish(path, snapshot)
    assert store.images.get(original_key) is None
    updated = store.scene_detail('T25')
    assert updated['images'][0]['url'] != first['images'][0]['url']
    assert store.images[image_key(updated['images'][0])].name == 'second.png'


def test_csv_exports_per_variant_feedback_and_does_not_load_image_mappings(review):
    store, _, _ = review
    detail = store.scene_detail('T25')
    note = 'Clearance needs work.\nKeep the original alternative.'
    store.save_review('T25', payload(detail, 'needs_fix', note))
    store._images.clear()
    values = list(csv.DictReader(io.StringIO(store.export_csv())))
    first = next(item for item in values if item['id'] == 'T25' and item['variant_id'] == 'bench')
    assert first['notes'] == note and first['decision'] == 'needs_fix'
    assert first['native_certification'] == 'False' and first['review_scope'] == 'design_layout_only'
    assert len(values) == 3 and store.images == {}


@pytest.mark.parametrize('change', ['unknown_variant', 'missing_variant', 'long_notes', 'boolean_revision'])
def test_invalid_feedback_cannot_change_any_review(review, change):
    store, _, _ = review
    post = payload(store.scene_detail('T25'))
    if change == 'unknown_variant':
        post['variant_id'] = 'unknown'
    elif change == 'missing_variant':
        post.pop('variant_id')
    elif change == 'long_notes':
        post['notes'] = 'x' * 10001
    else:
        post['revision'] = False
    with pytest.raises((ValueError, KeyError)):
        store.save_review('T25', post)
    assert not store.review_file.exists()


def test_empty_variant_row_remains_visible_and_disabled_routes_never_read_evidence(review):
    store, snapshot, path = review
    snapshot['rows'][1]['variants'] = []
    publish(path, snapshot)
    assert store.catalogue()['total'] == 2
    detail = store.scene_detail('T60')
    assert detail['selected_variant'] is detail['selected_variant_id'] is None
    assert detail['images'] == [] and detail['decision'] == 'unreviewed'
    assert 'not native certification' in store.category_report('anything', 'anything')
    assert 'not served' in store.diagnostic_log('T25', 999)
    with pytest.raises(KeyError):
        store.diagnostic_log('unknown', 0)


def test_cli_sets_only_process_local_assets_and_uses_required_snapshot_and_defaults(review, monkeypatch):
    import wia.runtime.commands.review_wia as review_wia

    store, _, path = review
    called = {}

    class StubServer:
        server_port = 8626

        def serve_forever(self):
            called['served'] = True

        def server_close(self):
            called['closed'] = True

    def create_server(value, host, port):
        called.update(store=value, host=host, port=port)
        return StubServer()

    monkeypatch.setattr(review_wia.review_esi_scenes, 'ASSETS', review_wia.ROOT / 'gumi/scene_review')
    monkeypatch.setattr(review_wia.review_esi_scenes, 'create_server', create_server)
    assert review_wia.main(['--snapshot', str(path), '--output-dir', str(store.output)]) == 0
    assert called['host'] == '127.0.0.1' and called['port'] == 8626
    assert called['served'] and called['closed']
    assert isinstance(called['store'], WiaReview)
    assert review_wia.review_esi_scenes.ASSETS == review_wia.ROOT / 'gumi/wia_review'
    with pytest.raises(SystemExit):
        review_wia.main([])


def test_progress_keeps_extra_proposals_and_human_approval_out_of_native_success(review):
    store, snapshot, path = review
    original = snapshot['rows'][0]
    for item in original['variants']:
        item['definition_status'] = {'task': {'available': True}, 'scene': {'available': True}}
    extra = deepcopy(original)
    extra.update(id='AP4', document_alias='AP4', proposal_membership=False)
    extra['variants'][0]['status'] = 'reference_verified'
    snapshot['rows'].append(extra)
    publish(path, snapshot)
    store.save_review('T25', payload(store.scene_detail('T25')))
    result = store.catalogue(filters={'scope': 'original', 'readiness': 'with_images'})
    assert [row['id'] for row in result['scenes']] == ['T25']
    assert result['summary']['proposal_progress'] == {
        'total': 2, 'definitions_available': 1, 'with_images': 1, 'without_images': 1,
        'awaiting_image_review': 1, 'with_layout_approval': 1, 'reference_verified': 0}
    assert store.catalogue(filters={'scope': 'extra'})['scenes'][0]['id'] == 'AP4'
    assert store.catalogue(filters={'scope': 'original', 'readiness': 'reference_verified'})['total'] == 0


def test_ready_queue_tracks_each_version_and_reopens_changed_definition(review):
    store, snapshot, path = review
    first = store.scene_detail('T25')
    store.save_review('T25', payload(first))
    row = store.catalogue(filters={'readiness': 'awaiting_review'})['scenes'][0]
    assert row['next_review_variant_id'] == 'original_selected'
    assert row['unreviewed_image_variant_count'] == 1
    second = store.scene_detail('T25', variant_id='original_selected')
    store.save_review('T25', payload(second, 'needs_fix'))
    assert store.catalogue(filters={'readiness': 'awaiting_review'})['total'] == 0
    needs_fix = store.catalogue(filters={'readiness': 'needs_fix'})['scenes'][0]
    assert needs_fix['id'] == 'T25'
    assert needs_fix['next_needs_fix_variant_id'] == 'original_selected'
    snapshot['rows'][0]['variants'][0]['fingerprint'] = 'bench-definition-v2'
    publish(path, snapshot)
    row = store.catalogue(filters={'readiness': 'awaiting_review'})['scenes'][0]
    assert row['next_review_variant_id'] == 'bench'
    assert row['review_stale'] is True
    detail = store.scene_detail('T25')
    assert detail['variants'][0]['decision'] == 'unreviewed'
    assert detail['variants'][1]['decision'] == 'needs_fix'


def test_evidence_filters_and_objective_search_remain_metadata_only(review, monkeypatch):
    store, _, _ = review

    def forbidden(*args, **kwargs):
        raise AssertionError('Triage must not read or resolve image files.')

    monkeypatch.setattr(store, '_image_path', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    ready = store.catalogue(filters={'readiness': 'with_images', 'q': 'lift the banana'})
    assert ready['total'] == 1
    assert ready['scenes'][0]['image_count'] == 2
    assert ready['scenes'][0]['image_variant_count'] == 2
    assert ready['scenes'][0]['default_image_count'] == 1
    assert store.catalogue(filters={'readiness': 'without_images'})['scenes'][0]['id'] == 'T60'
    assert store.images == {}

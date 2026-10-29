"""Validate public composite Review metadata, filters and optional checklists.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_composite_review.py
Fixtures use only temporary metadata files; no live server or simulator is used.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from wia.runtime.scene_capture_queue import write_json
from wia.runtime.scene_review import ReviewConflict
from wia.runtime.scene_review_live import LiveSceneReview


@pytest.fixture
def composite_review(tmp_path):
    queue = tmp_path / 'queue'
    snapshot_path = tmp_path / 'snapshot.json'
    metadata = {'composite': {'group_id': 'group_one', 'member_count': 2, 'sources': ['source_a', 'source_b'],
        'private': {'answer': 'PRIVATE_ANSWER_CANARY'},
        'public_goals': [{'id': 'goal_a', 'big_task': 'Enumerative Perception', 'small_task': 'Counting w Occlusion',
            'source': 'source_a', 'objective': 'Visit all safe target approaches.', 'success_criteria': 'Stop at a qualified approach.',
            'constraints': ['Keep original furniture in place.'], 'original_question': 'How many objects?',
            'original_options': ['one', 'two'], 'answer': 'PRIVATE_ANSWER_CANARY'},
            {'id': 'goal_b', 'big_task': 'Cognitive Mapping', 'small_task': 'Connectivity',
             'source': 'source_b', 'objective': 'Return to departure.', 'success_criteria': 'Park at the measured start.'}]},
        'category_tags': [{'big_task': 'Enumerative Perception', 'small_task': 'Counting w Occlusion'},
                          {'big_task': 'Cognitive Mapping', 'small_task': 'Connectivity'},
                          {'big_task': 'Cognitive Mapping', 'small_task': 'Connectivity'}],
        'review_goals': ['Verify the measured support.', 'Check the mounted camera view.'],
        'progress': {'status': 'not_executed', 'completion_score': None, 'completed_goals': None,
                     'total_goals': 2, 'private': 'PRIVATE_ANSWER_CANARY'}}
    row = {'id': 'category_aa', 'candidate_id': 'category_aa', 'fingerprint': 'current-composite',
           'category': 'Enumerative Perception', 'subcategory': 'Counting w Occlusion',
           'source': 'source_a', 'scene': 'house', 'room': 'room', 'native_capture_complete': True,
           'images': [], 'warning_codes': [], **metadata}
    diagnostic = {'id': 'category_bb', 'candidate_id': 'category_bb', 'fingerprint': 'diagnostic',
                  'category': 'Cognitive Mapping', 'subcategory': 'Connectivity',
                  'source': 'failed_source', 'native_capture_complete': False, 'images': []}
    snapshot = {'scenes': [row], 'diagnostics': [diagnostic], 'categories': [
        {'big_task': 'Enumerative Perception', 'small_task': 'Counting w Occlusion', 'source_count': 2, 'counts': {}},
        {'big_task': 'Cognitive Mapping', 'small_task': 'Connectivity', 'source_count': 2, 'counts': {}}]}
    write_json(snapshot_path, snapshot)
    write_json(queue / 'category_records/category_aa.json', {'job_id': 'capture_aa', **metadata,
                                                          'private': {'answer': 'PRIVATE_ANSWER_CANARY'}})
    write_json(queue / 'category_records/category_bb.json', {})
    receipt = queue / 'receipt.json'
    write_json(receipt, {'complete': True, 'images': {'head': '/not-read/screenshot.png'}})
    write_json(queue / 'jobs/capture_aa/status.json', {'state': 'REVIEW_PENDING', 'receipt': str(receipt)})
    return LiveSceneReview(None, tmp_path / 'review', queue_root=queue, catalogue_file=snapshot_path), snapshot


def test_cross_class_filter_counts_and_global_unique_rows(composite_review):
    store, snapshot = composite_review
    # Even a duplicate publisher reference must not produce duplicate cards.
    snapshot['scenes'].append(snapshot['scenes'][0])
    write_json(store.catalogue_file, snapshot)
    listing = store.catalogue()
    assert listing['total'] == listing['matched'] == 1
    assert [row['id'] for row in listing['scenes']] == ['category_aa']
    alternate = store.catalogue(filters={'big_task': 'Cognitive Mapping', 'small_task': 'Connectivity'})
    assert alternate['matched'] == 1
    assert store.catalogue(filters={'big_task': 'Enumerative Perception', 'small_task': 'Connectivity'})['matched'] == 0
    groups = {(group['big_task'], group['small_task']): group['counts'] for group in listing['categories']}
    assert groups[('Enumerative Perception', 'Counting w Occlusion')]['captured'] == 1
    assert groups[('Cognitive Mapping', 'Connectivity')]['captured'] == 1
    assert groups[('Cognitive Mapping', 'Connectivity')]['failed'] == 1
    assert [row['id'] for row in store.catalogue(filters={'scope': 'diagnostics'})['scenes']] == ['category_bb']


def test_public_only_list_and_lazy_detail_without_scanning(composite_review, monkeypatch):
    store, snapshot = composite_review
    def forbidden(*args, **kwargs):
        raise AssertionError('No directory scans, image reads or hashes are allowed.')
    for method in ('glob', 'rglob', 'iterdir'):
        monkeypatch.setattr(Path, method, forbidden)
    import wia.runtime.scene_review
    monkeypatch.setattr(wia.runtime.scene_review, 'sha256', forbidden)
    listing = store.catalogue()
    assert listing['scenes'][0]['images'] == []
    assert not store.images
    assert 'PRIVATE_' not in json.dumps(listing)
    # The selected record supplies declared public metadata missing from summary.
    for field in ('composite', 'category_tags', 'review_goals', 'progress'):
        snapshot['scenes'][0].pop(field)
    write_json(store.catalogue_file, snapshot)
    detail = store.scene_detail('category_aa')
    assert detail['composite']['member_count'] == 2
    assert detail['review_goals'] == ['Verify the measured support.', 'Check the mounted camera view.']
    assert detail['progress']['status'] == 'not_executed'
    assert detail['progress']['completion_score'] is None
    assert len(detail['images']) == 1
    assert 'PRIVATE_' not in json.dumps(detail)


def test_checks_persist_history_without_gating_layout_approval(composite_review):
    store, _ = composite_review
    payload = {'decision': 'needs_fix', 'notes': 'Check the camera.', 'fingerprint': 'current-composite',
               'revision': 0, 'review_checks': {'0': True, '1': False}}
    first = store.save_review('category_aa', payload)
    assert first['review_checks'] == payload['review_checks']
    approved = store.save_review('category_aa', {**payload, 'revision': 1, 'decision': 'approved',
                                              'review_checks': {'0': False, '1': False}})
    assert approved['approval_scope'] == 'layout'
    assert store.scene_detail('category_aa')['review_checks'] == {'0': False, '1': False}
    saved = json.loads(store.review_file.read_text())
    assert saved['history'][0]['review']['notes'] == 'Check the camera.'
    assert saved['history'][0]['review']['review_checks'] == {'0': True, '1': False}
    for group in store.catalogue()['categories']:
        assert group['counts']['approved'] == 1
    with pytest.raises(ReviewConflict):
        store.save_review('category_aa', payload)
    with pytest.raises(ReviewConflict):
        store.save_review('category_aa', {**payload, 'revision': 2, 'fingerprint': 'old'})


@pytest.mark.parametrize('checks', [{'2': True}, {'-1': False}, {'00': True}, {'0': 1}, {'0': 'true'}, [True]])
def test_invalid_check_indices_and_nonbooleans_rejected(composite_review, checks):
    store, _ = composite_review
    with pytest.raises(ValueError, match='existing goal indices'):
        store.save_review('category_aa', {'decision': 'approved', 'fingerprint': 'current-composite',
                                        'revision': 0, 'review_checks': checks})
    assert not store.review_file.exists()


def test_omitted_checks_preserve_legacy_notes_and_current_check_state(composite_review):
    store, snapshot = composite_review
    snapshot['scenes'][0].pop('review_goals')
    write_json(store.catalogue_file, snapshot)
    payload = {'decision': 'unreviewed', 'fingerprint': 'current-composite', 'revision': 0,
               'notes': 'Keep the support.', 'review_checks': {'1': True}}
    store.save_review('category_aa', payload)
    saved = store.save_review('category_aa', {key: value for key, value in {**payload, 'revision': 1}.items()
                                            if key != 'review_checks'})
    assert saved['notes'] == 'Keep the support.'
    assert saved['review_checks'] == {'1': True}


def test_previous_native_capture_is_public_diagnostic_without_private_answer(composite_review):
    store, snapshot = composite_review
    previous = {'state': 'FAILED', 'failure_stage': 'interaction_setup',
                'error': 'Original target support was rejected.', 'job_id': 'capture_previous',
                'candidate_id': 'category_cc', 'answer': 'PRIVATE_ANSWER_CANARY',
                'private_bindings': {'answer': 'PRIVATE_ANSWER_CANARY'}}
    snapshot['scenes'][0]['composite']['public_goals'][1]['previous_capture'] = previous
    snapshot['scenes'][0]['prior_conversion_failures'] = [{'stage': 'native_capture', 'error': previous['error']}]
    write_json(store.catalogue_file, snapshot)
    for row in (store.catalogue()['scenes'][0], store.scene_detail('category_aa')):
        diagnostic = row['composite']['public_goals'][1]['previous_capture']
        assert diagnostic == {key: previous[key] for key in ('state', 'failure_stage', 'error', 'job_id', 'candidate_id')}
        assert 'PRIVATE_' not in json.dumps(row)
        assert row['progress']['status'] == 'not_executed'
        assert row['native_capture_complete'] and row['layout_approval_available']
        assert row['prior_conversion_failures'][0]['error'] == previous['error']


def test_merged_review_marking_uses_latest_original_feedback_without_inheriting_approval(composite_review):
    store, snapshot = composite_review
    goals = snapshot['scenes'][0]['composite']['public_goals']
    goals[0]['previous_review'] = {'candidate_id': 'category_cc', 'decision': 'approved',
                                  'notes': 'Outdated note.', 'revision': 1, 'answer': 'PRIVATE_ANSWER_CANARY'}
    goals[1]['previous_review'] = {'candidate_id': 'category_dd', 'decision': 'approved', 'revision': 1}
    write_json(store.catalogue_file, snapshot)
    write_json(store.review_file, {'entries': {
        'category_cc': {'decision': 'needs_fix', 'notes': 'Inspect the original door.', 'revision': 2,
                        'private': 'PRIVATE_ANSWER_CANARY'},
        'category_dd': {'decision': 'approved', 'notes': '', 'revision': 1}}})
    raw = store.review_file.read_bytes()
    for row in (store.catalogue()['scenes'][0], store.scene_detail('category_aa')):
        assert row['source_review_summary'] == {'is_merged': True, 'reviewed_goals': 2,
                                                'approved': 1, 'needs_fix': 1, 'skipped': 0}
        assert row['decision'] == 'unreviewed' and not row['human_approved']
        assert row['composite']['public_goals'][0]['previous_review'] == {
            'candidate_id': 'category_cc', 'decision': 'needs_fix',
            'notes': 'Inspect the original door.', 'revision': 2}
        assert 'PRIVATE_' not in json.dumps(row)
    assert store.catalogue(filters={'review_status': 'merged_reviewed'})['matched'] == 1
    assert store.catalogue(filters={'review_status': 'approved'})['matched'] == 0
    snapshot['scenes'][0]['composite']['member_count'] = 1
    snapshot['scenes'][0]['composite']['public_goals'] = goals[:1]
    write_json(store.catalogue_file, snapshot)
    assert store.catalogue(filters={'review_status': 'merged_reviewed'})['matched'] == 0
    assert store.review_file.read_bytes() == raw


def test_frontend_public_renderers_never_infer_execution_from_images():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable for the focused frontend renderer check.')
    app = Path('gumi/scene_review/app.js').read_text()
    functions = app.split('// Composite public metadata renderers.')[1].split('// End composite public metadata renderers.')[0]
    script = "const esc = value => String(value ?? '').replaceAll('<','&lt;').replaceAll('>','&gt;');\n" + functions
    script += """
const row={subcategory:'First',category_tags:[{small_task:'Second'},{small_task:'Second'}],images:[{url:'capture.png'}],
 source_review_summary:{is_merged:true,reviewed_goals:1,needs_fix:1},
 composite:{member_count:2,public_goals:[{source:'<script>',objective:'Visit both objects.',success_criteria:'Stop safely.',
 original_question:'How many?',original_options:['one','two'],answer:'PRIVATE_ANSWER_CANARY',
 previous_review:{candidate_id:'old_source',decision:'needs_fix',revision:2,notes:'Check <script> view.',answer:'PRIVATE_ANSWER_CANARY'},
 previous_capture:{state:'FAILED',failure_stage:'placement',error:'Support rejected.',job_id:'old_job',candidate_id:'old_candidate',answer:'PRIVATE_ANSWER_CANARY'}}]}};
const pending=executionProgressMarkup({...row,progress:{status:'not_executed',completion_score:1}});
const measured=executionProgressMarkup({...row,progress:{status:'measured',completion_score:.5,completed_goals:1,total_goals:2}});
process.stdout.write(JSON.stringify({pending,measured,goals:compositeGoalsMarkup(row),labels:categoryLabels(row),badge:compositeBadge(row),reviewBadge:previousReviewBadge(row)}));
"""
    output = json.loads(subprocess.run([node, '-e', script], check=True, capture_output=True, text=True).stdout)
    assert 'Not executed' in output['pending'] and '100' not in output['pending']
    assert '50.0% completion' in output['measured'] and '1 / 2' in output['measured']
    assert 'PRIVATE_' not in output['goals'] and '<script>' not in output['goals']
    assert 'How many?' in output['goals'] and 'one · two' in output['goals']
    assert 'Previous capture: FAILED · placement' in output['goals'] and 'Support rejected.' in output['goals']
    assert 'Previously reviewed source · Needs repair' in output['goals']
    assert 'Check &lt;script&gt; view.' in output['goals']
    assert 'Merged · 1 previously reviewed' in output['reviewBadge']
    assert 'Prior repair feedback · 1' in output['reviewBadge']
    assert output['labels'] == 'First · Second' and '2 goals' in output['badge']

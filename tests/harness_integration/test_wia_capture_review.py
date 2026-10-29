"""Check explicit capture-receipt supplements without queues or native work.

Run: python -B -m pytest -q -p no:cacheprovider
  tests/test_wia_capture_review.py tests/test_wia_review_publisher.py
Temporary fixtures contain no real captures; PNG bytes are never decoded/read.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import wia.authoring.commands.build_wia_capture_review as capture
import wia.authoring.commands.build_wia_review as publisher


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixtures(root, *, publish_receipt=True):
    (root / 'task.yaml').write_text('public:\n  objective: Review this layout.\n', encoding='utf-8')
    (root / 'scene.yaml').write_text('id: explicit_scene\n', encoding='utf-8')
    directory = root / 'queue/jobs/capture_abc/attempts/01'
    directory.mkdir(parents=True)
    receipt_file = directory / 'preview_complete.json'
    images = {}
    for view in ('head', 'global_overhead', 'phase_2_head', 'object_detail_31', 'wrist'):
        path = directory / (view + '.png')
        path.write_bytes(b'PNG fixture is never decoded.')
        images[view] = str(path)
    images.pop('wrist')  # An existing unlisted file must never be discovered.
    images['global_overhead'] = (directory / 'global_overhead.png').relative_to(root).as_posix()
    images['phase_2_head'] = 'phase_2_head.png'
    images['unknown_overlay'] = '/home/never/read/unknown.png'
    task_hash = hashlib.sha256((root / 'task.yaml').read_bytes()).hexdigest()
    item = {'row_id': 'wia_v1_t01', 'document_alias': 'T01', 'variant_id': 'native_initial_layout',
            'label': 'Initial native layout', 'task_file': 'task.yaml', 'scene_file': 'scene.yaml',
            'task_sha256': task_hash,
            'scene_sha256': hashlib.sha256((root / 'scene.yaml').read_bytes()).hexdigest(),
            'job_id': 'capture_abc', 'receipt_file': receipt_file.relative_to(root).as_posix()}
    receipt = {'complete': True, 'capture_job_id': 'capture_abc', 'job_id': 'external_process_id',
               'initial_agent_bundle': {'capture_input_identity': {'task_sha256': task_hash}},
               'execution_qualified': True, 'native_qualification_run': False, 'images': images}
    index = {'schema': capture.INDEX_SCHEMA, 'items': [item]}
    write_json(root / 'index.json', index)
    if publish_receipt:
        write_json(receipt_file, receipt)
    return index, receipt, receipt_file


def build(root):
    return capture.build_supplement('index.json', project_root=root,
                                     generated_at='2026-10-23T00:00:00+00:00')


def variant(supplement):
    return supplement['row_updates']['wia_v1_t01']['variants'][0]


def test_ready_receipt_maps_only_explicit_same_directory_known_pngs_and_never_claims_success(tmp_path, monkeypatch):
    fixtures(tmp_path)
    original_read = Path.read_bytes

    def guarded_read(path):
        assert path.suffix != '.png', 'Image bytes must not be decoded or hashed.'
        assert not str(path).startswith('/home/')
        return original_read(path)

    def no_scan(*args, **kwargs):
        raise AssertionError('Capture publishing must not scan directories.')

    monkeypatch.setattr(Path, 'read_bytes', guarded_read)
    monkeypatch.setattr(Path, 'glob', no_scan)
    monkeypatch.setattr(Path, 'rglob', no_scan)
    supplement = build(tmp_path)
    result = variant(supplement)
    assert result['status'] == 'native_images_ready'
    assert result['evidence_summary'] == capture.LAYOUT_ONLY
    assert result['capture_review']['physical_success_inferred'] is False
    assert result['capture_review']['receipt_execution_qualified'] is True
    assert result['captures'][0]['views'] == ['global_overhead', 'head', 'object_detail_31', 'phase_2_head']
    assert result['captures'][0]['role'] == 'layout_review_only'
    assert result['capture_review']['ignored_views'] == ['unknown_overlay']
    assert result['evidence'][0]['path'].endswith('/attempts/01/preview_complete.json')
    assert supplement['row_updates']['wia_v1_t01'].keys() == {'variants'}
    assert supplement['summary']['variant_status_counts'] == {'native_images_ready': 1}
    assert supplement['provenance']['queue_jobs_submitted'] is False


def test_missing_receipt_stays_visible_and_reload_attaches_only_after_exact_receipt_arrives(tmp_path):
    _, receipt, path = fixtures(tmp_path, publish_receipt=False)
    pending = variant(build(tmp_path))
    assert pending['status'] == 'capture_pending' and pending['captures'] == []
    assert pending['id'] == 'native_initial_layout' and pending['task_file'] == 'task.yaml'
    write_json(path, receipt)
    assert variant(build(tmp_path))['status'] == 'native_images_ready'


@pytest.mark.parametrize('failure', ['invalid_json', 'not_mapping', 'incomplete', 'errors'])
def test_invalid_or_failed_receipts_keep_variant_visible_with_no_images(tmp_path, failure):
    _, receipt, path = fixtures(tmp_path)
    if failure == 'invalid_json':
        path.write_text('{unfinished', encoding='utf-8')
    elif failure == 'not_mapping':
        write_json(path, [])
    else:
        if failure == 'incomplete':
            receipt['complete'] = False
        else:
            receipt['errors'] = {'native': 'Do not copy arbitrary log contents into the supplement.'}
        write_json(path, receipt)
    result = variant(build(tmp_path))
    assert result['status'] == 'capture_failed' and result['captures'] == []
    assert result['capture_review']['reason']
    assert 'Do not copy arbitrary' not in json.dumps(result)


@pytest.mark.parametrize('failure', ['task_hash', 'scene_hash', 'task_bytes', 'scene_bytes', 'receipt_job', 'initial_hash'])
def test_definition_or_receipt_identity_mismatch_never_attaches_images(tmp_path, failure):
    index, receipt, path = fixtures(tmp_path)
    if failure in ('task_hash', 'scene_hash'):
        index['items'][0]['task_sha256' if failure == 'task_hash' else 'scene_sha256'] = '0' * 64
        write_json(tmp_path / 'index.json', index)
    elif failure in ('task_bytes', 'scene_bytes'):
        target = tmp_path / ('task.yaml' if failure == 'task_bytes' else 'scene.yaml')
        target.write_text('changed: true\n', encoding='utf-8')
    elif failure == 'receipt_job':
        receipt['capture_job_id'] = 'capture_foreign'
        write_json(path, receipt)
    else:
        receipt['initial_agent_bundle']['capture_input_identity']['task_sha256'] = '0' * 64
        write_json(path, receipt)
    result = variant(build(tmp_path))
    assert result['status'] == 'evidence_mismatch' and result['captures'] == []
    assert result['evidence'] and result['capture_review']['reason']


@pytest.mark.parametrize('channel', ['task', 'scene', 'receipt', 'image_home', 'image_other_directory',
                                   'image_other_name', 'image_symlink'])
def test_disallowed_paths_are_isolated_per_variant_without_reading_home(tmp_path, monkeypatch, channel):
    index, receipt, path = fixtures(tmp_path)
    invalid = deepcopy(index['items'][0])
    invalid['variant_id'] = 'invalid_declared_path'
    if channel in ('task', 'scene'):
        invalid[channel + '_file'] = '/home/never/read/definition.yaml'
    elif channel == 'receipt':
        invalid['receipt_file'] = '/home/never/read/preview_complete.json'
    else:
        if channel == 'image_home':
            receipt['images']['head'] = '/home/never/read/head.png'
        elif channel == 'image_other_directory':
            receipt['images']['head'] = str(tmp_path / 'another/head.png')
        elif channel == 'image_other_name':
            receipt['images']['head'] = str(path.parent / 'wrist.png')
        else:
            (path.parent / 'head.png').unlink()
            (path.parent / 'head.png').symlink_to('/home/never/read/head.png')
        write_json(path, receipt)
    # A separate exact receipt keeps a valid row publishable in the same batch.
    good_directory = tmp_path / 'good'
    good_directory.mkdir()
    (good_directory / 'head.png').write_bytes(b'Never decoded.')
    good_receipt = deepcopy(receipt)
    good_receipt['images'] = {'head': str(good_directory / 'head.png')}
    write_json(good_directory / 'preview_complete.json', good_receipt)
    valid = deepcopy(index['items'][0])
    valid['row_id'], valid['receipt_file'] = 'wia_v1_t02', 'good/preview_complete.json'
    write_json(tmp_path / 'index.json', {'schema': capture.INDEX_SCHEMA, 'items': [invalid, valid]})
    original_read_bytes, original_read_text = Path.read_bytes, Path.read_text

    def guarded_bytes(target):
        assert not str(target).startswith('/home/') and target.suffix != '.png'
        return original_read_bytes(target)

    def guarded_text(target, *args, **kwargs):
        assert not str(target).startswith('/home/')
        return original_read_text(target, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_bytes', guarded_bytes)
    monkeypatch.setattr(Path, 'read_text', guarded_text)
    supplement = build(tmp_path)
    result = variant(supplement)
    assert result['status'] == 'evidence_mismatch' and result['captures'] == []
    assert supplement['row_updates']['wia_v1_t02']['variants'][0]['status'] == 'native_images_ready'
    if channel in ('task', 'scene'):
        assert result[channel + '_file'] == ''
    if channel == 'receipt':
        assert result['evidence'] == []


def test_missing_png_stays_pending_and_no_existing_unlisted_image_is_substituted(tmp_path):
    _, receipt, path = fixtures(tmp_path)
    (path.parent / 'head.png').unlink()
    result = variant(build(tmp_path))
    assert result['status'] == 'capture_pending' and result['captures'] == []
    assert result['capture_review']['missing_views'] == ['head']
    assert 'wrist' not in result['capture_review']['accepted_views']


def test_duplicate_index_pairs_raise_and_atomic_output_protects_curation_index_and_reviews(tmp_path):
    index, _, _ = fixtures(tmp_path)
    supplement = build(tmp_path)
    output = capture.publish_supplement('capture_evidence.json', supplement, project_root=tmp_path,
                                         protected_paths=('index.json',))
    assert json.loads(output.read_text()) == supplement
    replacement = deepcopy(supplement)
    replacement['generated_at'] = 'later'
    capture.publish_supplement('capture_evidence.json', replacement, project_root=tmp_path)
    assert json.loads(output.read_text())['generated_at'] == 'later'
    assert not list(tmp_path.glob('*.tmp'))
    write_json(tmp_path / 'curation.json', {'schema': publisher.EVIDENCE_SCHEMA, 'row_updates': {}})
    for target in ('reviews.json', 'curation.json', 'index.json'):
        with pytest.raises(ValueError):
            capture.publish_supplement(target, supplement, project_root=tmp_path,
                                        protected_paths=('index.json',))
    index['items'].append(deepcopy(index['items'][0]))
    write_json(tmp_path / 'index.json', index)
    with pytest.raises(ValueError, match='duplicate'):
        build(tmp_path)


def test_supplement_publishes_images_without_replacing_main_variant_or_assessment(tmp_path):
    fixtures(tmp_path)
    supplement = build(tmp_path)
    capture.publish_supplement('capture_evidence.json', supplement, project_root=tmp_path)
    write_json(tmp_path / 'registry.json', {'rows': [{
        'wia_id': 'wia_v1_t01', 'document_alias': 'T01', 'primary_family': 'AP',
        'implementation_state': 'selected_definition', 'task': {'task_file': 'task.yaml'},
        'scene': {'selected_scene_file': 'scene.yaml'}}]})
    write_json(tmp_path / 'main_evidence.json', {'schema': publisher.EVIDENCE_SCHEMA,
        'row_updates': {'wia_v1_t01': {'status': 'definition_only', 'summary': 'Main review pending.',
            'variants': [{'id': 'main_design', 'task_file': 'task.yaml', 'scene_file': 'scene.yaml'}]}}})
    snapshot = publisher.build_snapshot('registry.json', 'main_evidence.json',
        supplements=['capture_evidence.json'], project_root=tmp_path)
    row, = snapshot['rows']
    assert row['status'] == 'definition_only' and row['summary'] == 'Main review pending.'
    assert [entry['id'] for entry in row['variants']] == ['native_initial_layout', 'main_design', 'original_selected']
    assert len(row['variants'][0]['images']) == 4
    assert row['variants'][0]['status'] == 'native_images_ready'
    assert row['variants'][1]['images'] == []
    assert snapshot['summary']['reference_success_row_count'] == 0


def retry_fixtures(root, *, state='REVIEW_PENDING', complete=True):
    index, receipt, declared = fixtures(root)
    job_directory = declared.parent.parent.parent
    status_file = job_directory / 'status.json'
    resolved = job_directory / 'attempts/02/preview_complete.json'
    resolved.parent.mkdir()
    (resolved.parent / 'head.png').write_bytes(b'Never decoded retry PNG fixture.')
    retry_receipt = deepcopy(receipt)
    retry_receipt['images'] = {'head': 'head.png'}
    write_json(resolved, retry_receipt)
    status = {'job_id': index['items'][0]['job_id'], 'state': state,
              'complete': complete, 'attempts': 2, 'receipt': str(resolved)}
    write_json(status_file, status)
    index['items'][0]['queue_status_file'] = status_file.relative_to(root).as_posix()
    write_json(root / 'index.json', index)
    return index, status, status_file, declared, resolved


def test_explicit_completed_retry_status_selects_attempt02_and_preserves_declared_receipt(tmp_path, monkeypatch):
    _, _, _, declared, resolved = retry_fixtures(tmp_path)
    # An earlier failed/missing receipt must not hide the completed exact retry.
    declared.unlink()
    def no_scan(*args, **kwargs):
        raise AssertionError('Optional queue status must not scan jobs or attempts.')
    monkeypatch.setattr(Path, 'glob', no_scan)
    monkeypatch.setattr(Path, 'rglob', no_scan)
    result = variant(build(tmp_path))
    review = result['capture_review']
    assert result['status'] == 'native_images_ready'
    assert result['captures'][0]['directory'] == resolved.parent.relative_to(tmp_path).as_posix()
    assert result['captures'][0]['views'] == ['head']
    assert review['declared_receipt_file'] == declared.relative_to(tmp_path).as_posix()
    assert review['resolved_receipt_file'] == review['receipt_file'] == resolved.relative_to(tmp_path).as_posix()
    assert review['queue_status']['receipt_selected'] is True
    assert review['queue_status']['state'] == 'REVIEW_PENDING'
    assert review['queue_status']['attempts'] == 2
    assert review['physical_success_inferred'] is False
    assert review['full_reference_execution_deferred'] is True


def test_legacy_fixed_attempt01_ignores_undeclared_status_and_keeps_existing_behavior(tmp_path):
    _, _, declared = fixtures(tmp_path)
    # Even malformed other status data is not read without explicit opt-in.
    status_file = declared.parent.parent.parent / 'status.json'
    status_file.write_text('{invalid undeclared status', encoding='utf-8')
    result = variant(build(tmp_path))
    assert result['status'] == 'native_images_ready'
    assert 'queue_status' not in result['capture_review']
    assert result['capture_review']['resolved_receipt_file'] == declared.relative_to(tmp_path).as_posix()


@pytest.mark.parametrize('condition', ['missing_status', 'running', 'incomplete', 'missing_completed_receipt_field'])
def test_missing_or_pending_explicit_status_preserves_fixed_receipt_behavior(tmp_path, condition):
    _, status, status_file, declared, resolved = retry_fixtures(tmp_path)
    if condition == 'missing_status':
        status_file.unlink()
    else:
        if condition == 'running':
            status['state'] = 'RUNNING'
        elif condition == 'incomplete':
            status['complete'] = False
        else:
            status.pop('receipt')
        write_json(status_file, status)
    result = variant(build(tmp_path))
    assert result['status'] == 'native_images_ready'
    assert result['capture_review']['resolved_receipt_file'] == declared.relative_to(tmp_path).as_posix()
    assert result['capture_review']['queue_status']['receipt_selected'] is False
    # A missing first receipt remains pending even if an unselected second exists.
    declared.unlink()
    assert resolved.is_file()
    assert variant(build(tmp_path))['status'] == 'capture_pending'


@pytest.mark.parametrize('problem', ['wrong_job_id', 'invalid_json', 'duplicate_key', 'not_mapping',
    'malformed_state', 'malformed_complete', 'malformed_attempts', 'attempt_mismatch',
    'foreign_status_directory', 'foreign_receipt_job', 'receipt_home', 'status_home',
    'receipt_parent_escape', 'receipt_wrong_name', 'receipt_attempt_zero', 'receipt_symlink_escape'])
def test_cross_job_malformed_or_escaped_status_receipts_are_evidence_mismatches(tmp_path, problem, monkeypatch):
    index, status, status_file, _, resolved = retry_fixtures(tmp_path)
    if problem == 'wrong_job_id':
        status['job_id'] = 'capture_foreign'
    elif problem == 'invalid_json':
        status_file.write_text('{unfinished', encoding='utf-8')
    elif problem == 'duplicate_key':
        status_file.write_text('{"job_id":"capture_abc","job_id":"capture_abc"}', encoding='utf-8')
    elif problem == 'not_mapping':
        status = []
    elif problem == 'malformed_state':
        status['state'] = []
    elif problem == 'malformed_complete':
        status['complete'] = 'true'
    elif problem == 'malformed_attempts':
        status['attempts'] = '2'
    elif problem == 'attempt_mismatch':
        status['attempts'] = 1
    elif problem == 'foreign_status_directory':
        other = tmp_path / 'queue/jobs/capture_foreign/status.json'
        write_json(other, status)
        index['items'][0]['queue_status_file'] = str(other)
    elif problem == 'foreign_receipt_job':
        status['receipt'] = str(tmp_path / 'queue/jobs/capture_foreign/attempts/02/preview_complete.json')
    elif problem == 'receipt_home':
        status['receipt'] = '/home/never/read/attempts/02/preview_complete.json'
    elif problem == 'status_home':
        index['items'][0]['queue_status_file'] = '/home/never/read/status.json'
    elif problem == 'receipt_parent_escape':
        status['receipt'] = 'queue/jobs/capture_abc/attempts/../02/preview_complete.json'
    elif problem == 'receipt_wrong_name':
        status['receipt'] = str(resolved.with_name('other.json'))
    elif problem == 'receipt_attempt_zero':
        status['receipt'] = str(resolved.parent.parent / '00/preview_complete.json')
    else:
        resolved.unlink()
        resolved.symlink_to('/home/never/read/preview_complete.json')
    if problem not in ('invalid_json', 'duplicate_key'):
        write_json(status_file, status)
    write_json(tmp_path / 'index.json', index)
    original_read = Path.read_text
    def guarded_read(path, *args, **kwargs):
        assert not str(path).startswith('/home/'), 'Disallowed paths must be rejected before reading.'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded_read)
    result = variant(build(tmp_path))
    assert result['status'] == 'evidence_mismatch'
    assert result['captures'] == []
    assert result['capture_review']['reason']


@pytest.mark.parametrize('identity', ['job', 'task_hash'])
def test_completed_status_does_not_bypass_retry_receipt_identity_validation(tmp_path, identity):
    _, _, _, _, resolved = retry_fixtures(tmp_path)
    receipt = json.loads(resolved.read_text())
    if identity == 'job':
        receipt['capture_job_id'] = 'capture_foreign'
    else:
        receipt['initial_agent_bundle']['capture_input_identity']['task_sha256'] = '0' * 64
    write_json(resolved, receipt)
    result = variant(build(tmp_path))
    assert result['status'] == 'evidence_mismatch' and result['captures'] == []
    assert result['capture_review']['queue_status']['receipt_selected'] is True


def test_failed_exact_status_marks_missing_receipt_failed_without_hiding_valid_existing_receipt(tmp_path):
    index, receipt, declared = fixtures(tmp_path, publish_receipt=False)
    status_file = declared.parent.parent.parent / 'status.json'
    status = {'job_id': 'capture_abc', 'state': 'FAILED', 'retryable': True, 'attempts': 1,
              'error': 'Native initialization process exited before producing images.',
              'failure_stage': 'native_initialization'}
    write_json(status_file, status)
    index['items'][0]['queue_status_file'] = str(status_file)
    write_json(tmp_path / 'index.json', index)
    failed = variant(build(tmp_path))
    assert failed['status'] == 'capture_failed' and failed['captures'] == []
    provenance = failed['capture_review']['queue_status']
    assert provenance['error'] == status['error']
    assert provenance['failure_stage'] == 'native_initialization'
    assert 'infrastructure or initialization capture failure' in failed['capture_review']['reason']
    assert 'does not establish task infeasibility' in failed['capture_review']['reason']
    write_json(declared, receipt)
    ready = variant(build(tmp_path))
    assert ready['status'] == 'native_images_ready'
    assert ready['capture_review']['queue_status']['state'] == 'FAILED'
    assert ready['capture_review']['physical_success_inferred'] is False

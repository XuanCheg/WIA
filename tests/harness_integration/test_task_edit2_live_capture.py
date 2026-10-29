"""Verify new-campaign live detail refuses stale screenshot receipts.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_live_capture.py
"""
from copy import deepcopy

import pytest
from PIL import Image

from wia.runtime.scene_capture_queue import PROTOCOL, sha256, write_json
from wia.runtime.scene_review_live import verified_task_edit2_capture


@pytest.fixture
def campaign(tmp_path):
    key = 'category_abcd'
    job_id = 'capture_' + 'a' * 32
    job = tmp_path / 'queue' / 'jobs' / job_id
    snapshot = job / 'input'
    identity = 'Temporal Understanding/Agent Observation/test/room/q.json'
    paths = {}
    for name, content in [('task', 'task: current'), ('scene', 'scene: current'), ('source', '{}')]:
        path = tmp_path / (name + '.txt')
        path.write_text(content)
        paths[name] = path
    files = {'task.yaml': paths['task'], 'scene.yaml': paths['scene'], 'data/' + identity: paths['source']}
    for relative, source in files.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    record = {'candidate_id': key, 'job_id': job_id, 'source_identity': identity,
        'task_file': str(paths['task']), 'scene_file': str(paths['scene']), 'original_source': str(paths['source'])}
    manifest = {**record, 'source_json': record['original_source'], 'snapshot_dir': str(snapshot),
        'task_edit2_capture_campaign': True, 'capture_signature': 'signature', 'fingerprint': 'fingerprint',
        'capture_protocol': PROTOCOL, 'code_commit': 'working_fixture', 'esi_commit': 'fixture',
        'seed': 7, 'variant': 'adapted', 'expected_views': ['head'],
        'input_hashes': {name: sha256(snapshot / name) for name in files}}
    write_json(snapshot / 'manifest.json', manifest)
    write_json(job / 'accepted.json', {'manifest_sha256': sha256(snapshot / 'manifest.json')})
    image = job / 'attempts/01/head.png'
    image.parent.mkdir(parents=True)
    Image.new('RGB', (64, 64)).save(image)
    receipt = {key: manifest[key] for key in ('capture_signature', 'fingerprint', 'capture_protocol',
        'code_commit', 'esi_commit', 'seed', 'variant')}
    receipt.update(id=key, capture_job_id=job_id, complete=True, images={'head': str(image)})
    return tmp_path / 'queue', record, {'state': 'REVIEW_PENDING'}, receipt, snapshot, paths


def test_exact_campaign_receipt_is_available(campaign):
    queue, record, status, receipt, _, _ = campaign
    assert verified_task_edit2_capture(queue, record, status, receipt) == (True, None)


@pytest.mark.parametrize('change', ['current_task', 'current_source', 'frozen_task', 'receipt_alias',
                                  'receipt_job', 'receipt_code', 'missing_image', 'manifest_seal'])
def test_modified_context_never_enables_completion(campaign, change):
    queue, record, status, receipt, snapshot, paths = campaign
    receipt = deepcopy(receipt)
    if change == 'current_task': paths['task'].write_text('changed')
    elif change == 'current_source': paths['source'].write_text('changed')
    elif change == 'frozen_task': (snapshot / 'task.yaml').write_text('changed')
    elif change == 'receipt_alias': receipt['id'] = 'other'
    elif change == 'receipt_job': receipt['capture_job_id'] = 'other'
    elif change == 'receipt_code': receipt['code_commit'] = 'other'
    elif change == 'missing_image': __import__('pathlib').Path(receipt['images']['head']).unlink()
    elif change == 'manifest_seal': (snapshot / 'manifest.json').write_text('{}')
    complete, error = verified_task_edit2_capture(queue, record, status, receipt)
    assert complete is False and error

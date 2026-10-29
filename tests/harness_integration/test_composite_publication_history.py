"""Retain a folded member's failed capture without changing capture inputs.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_composite_publication_history.py
"""
import wia.runtime.commands.compose_esi_tasks as publisher
from wia.runtime.scene_capture_queue import read_json, write_json


def test_nonrepresentative_failure_and_latest_feedback_survive_metadata_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, 'ROOT', tmp_path)
    campaign, queue = tmp_path / 'campaign', tmp_path / 'queue'
    output, review = campaign / publisher.CAMPAIGN_DIR, campaign / 'review_catalogue'
    review.mkdir(parents=True)
    scene, task = tmp_path / 'scene.yaml', tmp_path / 'task.yaml'
    scene.write_text('Original physical scene\n')
    task.write_text('Original sealed evaluator\n')
    record = {'candidate_id': 'new', 'parent_candidate_ids': ['a', 'b'],
              'diagnostic_dir': str(output / 'new'), 'task_file': str(task), 'scene_file': str(scene),
              'composite': {'public_goals': [{'source': 'source_a'}, {'source': 'source_b'}]},
              'warning_codes': [], 'prior_conversion_failures': [{'origin': 'cpu', 'error': 'Old CPU error'}]}
    write_json(output / 'publication.json', {'complete': True, 'pending_records': [record]})
    batch_path = output / 'batch.json'
    write_json(batch_path, {'records': [record]})
    write_json(review / 'category_revisions.json', {'categories': {'category': {'new_batch_file': str(batch_path)}}})
    for key in ('a', 'b'):
        write_json(queue / 'category_records' / (key + '.json'),
                   {'candidate_id': key, 'source_identity': 'source_' + key, 'job_id': 'job_' + key})
    write_json(queue / 'jobs/job_a/status.json', {'state': 'REVIEW_PENDING'})
    receipt = queue / 'jobs/job_b/receipt.json'
    write_json(receipt, {'interaction_setup': {'accepted': False, 'reason': 'No collision-free floor start.'}})
    write_json(queue / 'jobs/job_b/status.json', {'state': 'REVIEW_PENDING', 'receipt': str(receipt),
               'native_log': '/retained/native.log'})
    write_json(queue / 'category_records/new.json', {**record, 'job_id': 'new_job', 'fingerprint': 'sealed'})
    reviews = tmp_path / 'outputs/local_capture_20261014/gallery/reviews.json'
    write_json(reviews, {'entries': {'b': {'decision': 'needs_fix', 'notes': 'Bad spawn.', 'revision': 2}}})
    before_reviews = reviews.read_bytes()
    for _ in range(2):
        summary = publisher.preserve_member_history(campaign, queue)
        card = read_json(queue / 'category_records/new.json')
        assert summary['previous_native_failures'] == 1
        assert card['job_id'] == 'new_job' and card['fingerprint'] == 'sealed'
        goals = card['composite']['public_goals']
        assert 'previous_capture' not in goals[0]
        assert goals[1]['previous_capture']['failure_stage'] == 'interaction_setup'
        assert goals[1]['previous_capture']['error'] == 'No collision-free floor start.'
        assert len(card['prior_conversion_failures']) == 2
        assert card['diagnostic_logs'] == ['/retained/native.log']
        assert card['prior_feedback'][0]['revision'] == 2
        assert 'previous_review' not in goals[0]
        assert goals[1]['previous_review'] == {'candidate_id': 'b', 'decision': 'needs_fix',
                                              'notes': 'Bad spawn.', 'revision': 2}
        assert read_json(batch_path)['records'][0]['composite'] == card['composite']
    assert scene.read_text() == 'Original physical scene\n'
    assert task.read_text() == 'Original sealed evaluator\n'
    assert reviews.read_bytes() == before_reviews

"""Verify semantic percentages survive the real category scoring envelope.

Run: python -m pytest -q tests/test_task_edit2_metric_transport.py
"""
from types import SimpleNamespace
import pytest
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope


def test_answer_accuracy_is_not_mixed_with_physical_progress(tmp_path):
    task = SimpleNamespace(definition=SimpleNamespace(id='questions', repair={}),
        task_text='Answer both questions and inspect the room.', file_sha256='task', source_sha256='source')
    envelope = CategoryEvaluationEnvelope(SimpleNamespace(), task, tmp_path)
    value = envelope._seal({'schema': 'question_set', 'snapshot_id': 1, 'action_step': 0,
        'task_success': False, 'evaluation_validity': 'complete',
        'completion_items': [{'id': 'q1', 'kind': 'answer_requirement', 'completed': True},
            {'id': 'q2', 'kind': 'answer_requirement', 'completed': True},
            {'id': 'visit', 'kind': 'physical_obligation', 'completed': False}],
        'answer_accuracy_percent': 100., 'required_unique_questions': 2, 'correct_answers': 2,
        'answer_evaluation_validity': 'complete'}, SimpleNamespace(snapshot_id=1, action_step=0))
    assert value['score_percent'] == value['progress']['score_percent'] == 100.
    assert value['progress']['obligation_score_percent'] == pytest.approx(200. / 3)
    assert value['task_success'] is False


def test_unavailable_search_measurement_never_becomes_zero(tmp_path):
    task = SimpleNamespace(definition=SimpleNamespace(id='search', repair={}),
        task_text='Find all targets.', file_sha256='task', source_sha256='source')
    envelope = CategoryEvaluationEnvelope(SimpleNamespace(), task, tmp_path)
    value = envelope._seal({'schema': 'search', 'snapshot_id': 2, 'action_step': 1,
        'task_success': False, 'evaluation_validity': 'unavailable', 'discovery_percent': None,
        'required_distinct_targets': 3, 'verified_distinct_targets': 1,
        'completion_items': [{'id': str(i), 'completed': i == 0, 'evaluation_validity': 'unavailable'}
                             for i in range(3)]}, SimpleNamespace(snapshot_id=2, action_step=1))
    assert value['score_percent'] is None
    assert value['required_distinct_targets'] == 3
    assert value['evaluation_validity'] == 'unavailable'


def test_unqualified_public_definition_withholds_comparable_success(tmp_path, monkeypatch):
    import wia.native.esi_category_evaluation as module
    monkeypatch.setattr(module, 'qualify_definition_contracts', lambda *args, **kwargs: {
        'destination': {'accepted': False, 'pending_qualification': ['public_control_release_verified']}})
    monkeypatch.setattr('wia.tasks.background_policy.record_background_movements', lambda session: [])
    private = SimpleNamespace(evaluation=SimpleNamespace(model_dump=lambda: {
        'parameters': {'destination_contract': {'version': 'explicit_test'}}}),
        bindings={'destination': SimpleNamespace(name='table')})
    task = SimpleNamespace(definition=SimpleNamespace(id='delivery', repair={}, private=private),
        task_text='Deliver to the named table.', file_sha256='task', source_sha256='source')
    envelope = module.CategoryEvaluationEnvelope(SimpleNamespace(session=SimpleNamespace()), task, tmp_path)
    value = envelope._seal({'schema': 'delivery', 'snapshot_id': 3, 'action_step': 1,
        'task_success': True, 'evaluation_validity': 'complete', 'score_percent': 100.,
        'completion_items': [{'id': 'delivery', 'completed': True}]},
        SimpleNamespace(snapshot_id=3, action_step=1))
    assert value['task_success'] is False
    assert value['evaluation_validity'] == 'unavailable'
    assert value['score_percent'] is None
    assert value['native_result']['unqualified_diagnostic_score_percent'] == 100.

"""Validate T52-only declaration gating. Run: python -m pytest -q tests/test_t52_confirmation.py."""
from types import SimpleNamespace as NS

import pytest

from harness_integration.test_core_revision_perception import inside_world
from wia.native.esi_category_evaluation import CategoryEvaluationEnvelope
from wia.tasks.loader import load_task
from wia.resources import ROOT


def confirmed_world(tmp_path, monkeypatch):
    native, observation, state = inside_world(tmp_path, monkeypatch, initially_inside=True)
    native.parameters['require_inside_declaration_and_done'] = True
    task = load_task(ROOT/'explore_to_achieve/wia_v1/release/tasks/t52.yaml', run_mode='benchmark')
    wrapper = CategoryEvaluationEnvelope(native, task, tmp_path/'envelope')
    wrapper.begin_episode(observation)
    for _ in range(3):
        wrapper.sample_step()
    pending = wrapper.record(observation, action=None)
    assert pending['evaluation_validity'] == 'pending_done'
    assert pending['score_percent'] is None and pending['task_success'] is False
    return wrapper, native, state


@pytest.mark.parametrize('answer,done,expected', [
    (None, False, 0.), (None, True, 0.), ('outside', True, 0.),
    ('inside', False, 0.), ('inside', True, 100.),
])
def test_only_explicit_inside_and_active_done_earn_credit(tmp_path, monkeypatch, answer, done, expected):
    wrapper, native, _ = confirmed_world(tmp_path, monkeypatch)
    if answer is not None:
        wrapper.report_answer(answer, NS(snapshot_id=2, action_step=1))
    boundary = NS(snapshot_id=7, action_step=4)
    final = (wrapper.report_done(boundary) if done else
             wrapper.finalize_episode(boundary, reason='max_steps'))
    assert final['score_percent'] == expected
    assert final['evaluation_validity'] == 'complete' and final['evaluation_finalized']
    assert final['task_success'] is (expected == 100.)
    assert final['native_result']['physical_inside_complete']
    assert wrapper.record(NS(snapshot_id=8, action_step=5), action=None) == final
    wrapper.begin_episode(NS(snapshot_id=9, action_step=0))
    assert native.answer is None and not native.terminal_declared


@pytest.mark.parametrize('invalid', ['outside', 'held', 'unstable', 'missing'])
def test_declarations_do_not_replace_physical_evidence(tmp_path, monkeypatch, invalid):
    wrapper, _, state = confirmed_world(tmp_path, monkeypatch)
    wrapper.report_answer('inside', NS(snapshot_id=2, action_step=0))
    if invalid == 'outside':
        state['destination_relation'] = False
    elif invalid == 'held':
        state['gripper'].update(holding=True, constraint_active=True, object_name='small')
    elif invalid == 'unstable':
        state['linear_speed_m_s'] = .1
    else:
        state['destination_relation'] = None
        state['measurement_errors']['destination_relation'] = 'missing'
    result = wrapper.report_done(NS(snapshot_id=3, action_step=0))
    assert not result['task_success']
    assert result['score_percent'] == (None if invalid == 'missing' else 0.)


def test_explicit_confirmed_noop_needs_no_manipulation(tmp_path, monkeypatch):
    wrapper, _, _ = confirmed_world(tmp_path, monkeypatch)
    wrapper.report_answer('inside', NS(snapshot_id=2, action_step=0))
    final = wrapper.report_done(NS(snapshot_id=3, action_step=0))
    assert final['score_percent'] == 100. and final['native_result']['no_op_verified']


def test_resume_keeps_same_episode_declaration_but_new_episode_clears_it(tmp_path, monkeypatch):
    wrapper, _, _ = confirmed_world(tmp_path, monkeypatch)
    wrapper.report_answer('inside', NS(snapshot_id=2, action_step=0))
    assert wrapper.finalize_episode(NS(snapshot_id=3, action_step=0), reason='cancelled')['score_percent'] == 0.
    wrapper.resume_episode()
    assert wrapper.record(NS(snapshot_id=4, action_step=0), action=None)['evaluation_validity'] == 'pending_done'
    assert wrapper.report_done(NS(snapshot_id=5, action_step=0))['score_percent'] == 100.
    with pytest.raises(RuntimeError, match='model DONE'):
        wrapper.resume_episode()
    wrapper.begin_episode(NS(snapshot_id=6, action_step=0))
    for _ in range(3):
        wrapper.sample_step()
    assert wrapper.report_done(NS(snapshot_id=7, action_step=0))['score_percent'] == 0.


def test_only_t52_uses_the_new_confirmation_contract():
    for alias in ('t15', 't16', 't17', 't52'):
        task = load_task(ROOT/f'explore_to_achieve/wia_v1/release/tasks/{alias}.yaml', run_mode='benchmark')
        required = task.definition.private.evaluation.parameters.get('require_inside_declaration_and_done', False)
        assert required is (alias == 't52')

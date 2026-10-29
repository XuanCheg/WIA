"""Check diagnostic observation-boundary recording without native execution.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_native_capture_boundary.py
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

import wia.runtime.scene_capture_adapters as adapters
import wia.runtime.wia_capture_diagnostics as wia_capture_diagnostics
from core.robot_cameras import RGB_TO_VIEW
from wia.runtime.commands.check_esi_core_native import save_diagnostic_capture
from harness_integration.test_task_edit2_capture_protocol import NativeBundleSession


@pytest.fixture
def boundary(tmp_path):
    events = []
    observation = NS(snapshot_id=12, action_step=3,
        save=lambda output: events.append(('save', output)))
    snapshot = NS(observation=observation)
    evaluator = NS(episode=1, last_boundary=(11, 3))

    def capture(*, action_boundary):
        assert action_boundary is True
        events.append(('capture', action_boundary))
        return snapshot

    def record(observed, *, action):
        assert observed is observation
        assert action is None
        events.append(('record', observed.snapshot_id, observed.action_step, action))
        evaluator.last_boundary = (observed.snapshot_id, observed.action_step)

    evaluator.record = record
    session = NS(observation_source=NS(capture=capture, action_step=3), task_evaluator=evaluator)
    return NS(session=session, snapshot=snapshot, observation=observation, evaluator=evaluator,
        events=events, output=tmp_path / 'after_passive_hold')


def test_fresh_boundary_is_recorded_once_before_saving_without_crediting_an_action(boundary):
    saved = save_diagnostic_capture(boundary.session, boundary.output)
    assert saved is boundary.snapshot
    assert boundary.events == [('capture', True), ('record', 12, 3, None), ('save', boundary.output)]
    assert boundary.evaluator.last_boundary == (12, 3)
    assert boundary.session.observation_source.action_step == boundary.observation.action_step == 3
    # Reusing the returned boundary is a save, not another scoring observation.
    save_diagnostic_capture(boundary.session, boundary.output, snapshot=saved)
    assert boundary.events[-1] == ('save', boundary.output)
    assert sum(event[0] == 'record' for event in boundary.events) == 1


def test_supplied_initial_snapshot_is_neither_recaptured_nor_recorded(boundary):
    assert save_diagnostic_capture(boundary.session, boundary.output,
        snapshot=boundary.snapshot) is boundary.snapshot
    assert boundary.events == [('save', boundary.output)]
    assert boundary.evaluator.last_boundary == (11, 3)


@pytest.mark.parametrize('evaluator_state', ['inactive', 'no_episode_contract', 'absent'])
def test_fresh_capture_does_not_activate_or_call_an_inactive_evaluator(boundary, evaluator_state):
    if evaluator_state == 'inactive':
        boundary.evaluator.episode = 0
    elif evaluator_state == 'no_episode_contract':
        del boundary.evaluator.episode
    else:
        del boundary.session.task_evaluator
    assert save_diagnostic_capture(boundary.session, boundary.output) is boundary.snapshot
    assert boundary.events == [('capture', True), ('save', boundary.output)]
    assert boundary.evaluator.last_boundary == (11, 3)
    assert boundary.session.observation_source.action_step == 3


@pytest.mark.parametrize('diagnostic_state', ['available', 'unavailable', 'nonmirror'])
def test_initial_review_diagnostics_bind_only_private_evidence_once(monkeypatch, diagnostic_state):
    session = NativeBundleSession()
    diagnostic = None if diagnostic_state == 'nonmirror' else {
        'available': diagnostic_state == 'available', 'private_only': True,
        'geometry': {'private_marker': 'cue_eye_alignment'},
        'snapshot_id': -1, 'action_step': -1}
    if diagnostic_state == 'unavailable':
        diagnostic['error'] = 'Native geometry is unavailable.'
    original = deepcopy(diagnostic)
    calls = []

    def capture(actual):
        assert actual is session
        observation = actual._observation_source.latest_snapshot.observation
        assert set(actual._capture_initial_agent_bundle_views) == {
            *RGB_TO_VIEW.values(), 'initial_overview_rgb'}
        observation.initial_overview.validate_identity(actual.initial_overview_identity)
        assert observation.initial_overview.snapshot_id == observation.snapshot_id
        assert not getattr(actual, '_capture_initial_agent_bundle_complete', False)
        calls.append((observation, deepcopy(observation.metadata()),
            tuple(actual._observation_source.history), actual.renders))
        return diagnostic

    monkeypatch.setattr(wia_capture_diagnostics, 'capture_initial_review_diagnostics', capture)
    views = adapters._initial_agent_bundle(session)
    observation, public_before, history_before, frozen_render_count = calls[0]
    evidence = session.capture_initial_agent_bundle_evidence
    assert evidence['available'] is True
    assert (evidence['snapshot_id'], evidence['action_step']) == (2, 0)
    if diagnostic is None:
        assert 'review_diagnostics' not in evidence
    else:
        assert evidence['review_diagnostics'] == {**diagnostic, 'snapshot_id': 2, 'action_step': 0}
    assert diagnostic == original
    assert not hasattr(observation, 'review_diagnostics')
    assert observation.metadata() == public_before
    assert tuple(session._observation_source.history) == history_before == ()
    assert set(views) == {*RGB_TO_VIEW.values(), 'initial_overview_rgb'}
    assert session.renders == frozen_render_count
    assert session.pipeline.og.sim.current_time == 1.25
    assert evidence['physics_steps'] == evidence['robot_actions'] == evidence['model_calls'] == 0
    cached = adapters._initial_agent_bundle(session)
    assert len(calls) == 1
    assert all(cached[name] is rgb for name, rgb in views.items())
    assert session._observation_source.snapshot_id == 2
    assert session.renders == frozen_render_count


def test_review_diagnostics_remain_inside_existing_no_time_advance_guard(monkeypatch):
    session = NativeBundleSession()

    def capture(actual):
        actual.pipeline.og.sim.current_time += .01
        return {'available': False, 'error': 'Synthetic diagnostic time drift.'}

    monkeypatch.setattr(wia_capture_diagnostics, 'capture_initial_review_diagnostics', capture)
    with pytest.raises(RuntimeError, match='advanced the native physics clock'):
        adapters._initial_agent_bundle(session)
    evidence = session.capture_initial_agent_bundle_evidence
    assert evidence['available'] is False
    assert evidence['review_diagnostics'] == {'available': False,
        'error': 'Synthetic diagnostic time drift.', 'snapshot_id': 2, 'action_step': 0}
    assert not getattr(session, '_capture_initial_agent_bundle_complete', False)


def test_missing_mounted_view_prevents_private_diagnostics(monkeypatch):
    session = NativeBundleSession(missing=True)
    calls = []
    monkeypatch.setattr(wia_capture_diagnostics, 'capture_initial_review_diagnostics',
        lambda actual: calls.append(actual))
    with pytest.raises(RuntimeError, match='missing native mounted views'):
        adapters._initial_agent_bundle(session)
    assert calls == []
    assert 'review_diagnostics' not in session.capture_initial_agent_bundle_evidence

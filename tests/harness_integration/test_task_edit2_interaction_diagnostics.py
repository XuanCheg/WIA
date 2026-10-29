"""Check bounded diagnostic publication without touching physics or source state.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/test_task_edit2_interaction_diagnostics.py
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.tasks import interaction_diagnostics as diagnostics


@pytest.fixture
def world(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(diagnostics.time, 'monotonic', lambda: clock[0])
    source = NS(json_path='Physical Structure/Deformable/scene/room/q.json')
    task = NS(path=Path('/exact/task.yaml'), file_sha256='task_sha', source_sha256='source_sha',
              seed=0, definition=NS(source=source))
    scene = NS(path=Path('/exact/scene.yaml'), file_sha256='scene_sha', source_sha256='source_sha',
               seed=0, definition=NS(source=source))
    def forbidden(*args, **kwargs):
        pytest.fail('Diagnostics cannot step, render, reset or change native state.')
    session = NS(loaded_task=task, scene_adapter=NS(task=scene), native_reset_id='reset_0', episode_id='episode_0',
        pipeline=NS(og=NS(sim=NS(current_time=7., step=forbidden, render=forbidden, load_state=forbidden))),
        world_state={'unchanged': [1, 2, 3]})
    progress = []
    observer = diagnostics.install_interaction_diagnostics(session, tmp_path, progress.append)
    return NS(session=session, observer=observer, output=tmp_path, clock=clock, progress=progress)


def events(world):
    return [json.loads(line) for line in (world.output / 'interaction_setup_events.jsonl').read_text().splitlines()]


def latest(world):
    return json.loads((world.output / 'interaction_setup_latest.json').read_text())


def test_initial_full_source_receipt_is_preserved_without_mutation(world):
    receipt = {'accepted': False, 'source': {'vertices': [[0., 1., 2.]], 'cloth': {'model': 'original',
        'scale': [1., 1., 1.]}}, 'attempts': [], 'reason': None}
    before, state = deepcopy(receipt), deepcopy(world.session.world_state)
    result = world.session._interaction_setup_observer('source_measured', receipt)
    assert result['written'] is True
    document = latest(world)
    assert document['evidence'] == before == receipt
    assert document['provenance'] == {'reset_id': 'reset_0', 'episode_id': 'episode_0',
        'task_file': '/exact/task.yaml', 'task_sha256': 'task_sha', 'scene_file': '/exact/scene.yaml',
        'scene_sha256': 'scene_sha', 'source_identity': 'Physical Structure/Deformable/scene/room/q.json',
        'source_sha256': 'source_sha', 'scene_source_sha256': 'source_sha', 'seed': 0}
    assert document['attempt_count'] == 0
    assert document['current_failed_stage'] is document['earliest_failed_stage'] is None
    assert document['native_time_s'] == 7.
    assert document['qualification_status'] == 'not_evaluated_by_diagnostics'
    assert 'accepted' not in events(world)[0] and 'evidence' not in events(world)[0]
    assert world.session.world_state == state
    assert world.session.pipeline.og.sim.current_time == 7.
    assert world.progress == ['source_measured']


def test_repeated_trial_publications_are_bounded_by_wall_clock(world):
    evidence = {'attempts': []}
    world.observer('trial_started', evidence)
    for index in range(1, 101):
        world.clock[0] = 100. + index * .1
        evidence['attempts'].append({'checks': {'floor_supported': False}})
        assert not world.observer('trial_measured', evidence)['written']
    assert len(events(world)) == 1
    world.clock[0] = 115.
    assert world.observer('trial_measured', evidence)['written'] is True
    assert len(events(world)) == 2
    row = latest(world)
    assert row['callback_count'] == 102 and row['attempt_count'] == 100
    assert row['earliest_failed_stage'] == row['current_failed_stage'] == 'trial_measured'
    assert row['failed_checks'] == ['floor_supported']
    assert row['failure_stage_source'] == 'measured_false_checks'
    assert 'evidence' not in events(world)[-1]
    assert len(json.dumps(events(world)[-1])) < 2000


def test_source_finished_and_failure_stages_flush_without_waiting(world):
    world.observer('trial_started', {})
    for stage in ('source-measured', 'group_source_measured', 'finished', 'failure'):
        world.observer(stage, {'source': {'complete_receipt': stage}})
    assert len(events(world)) == 5
    assert [row['stage'] for row in events(world)] == ['trial_started', 'source-measured', 'group_source_measured', 'finished', 'failure']
    assert latest(world)['evidence']['source']['complete_receipt'] == 'failure'
    assert latest(world)['current_failed_stage'] == 'failure'
    assert world.progress == [row['stage'] for row in events(world)]


def test_current_and_earliest_explicit_failures_are_distinct_and_append_only(world):
    world.observer('source_measured', {'failure_stage': 'source_contact', 'attempts': []})
    original = (world.output / 'interaction_setup_events.jsonl').read_bytes()
    world.observer('failure', {'failure_stage': 'robot_approach', 'attempts': [{'failure_stage': 'candidate_floor'}]})
    assert (world.output / 'interaction_setup_events.jsonl').read_bytes().startswith(original)
    assert latest(world)['earliest_failed_stage'] == 'source_contact'
    assert latest(world)['current_failed_stage'] == 'robot_approach'
    assert latest(world)['attempt_count'] == 1
    world.observer('finished', {'accepted': True})
    assert latest(world)['current_failed_stage'] is None
    assert latest(world)['earliest_failed_stage'] == 'source_contact'
    assert latest(world)['qualification_status'] == 'not_evaluated_by_diagnostics'


@pytest.mark.parametrize('missing', ['API', 'nonfinite'])
def test_unavailable_native_time_and_attempt_count_are_explicit_unknowns(world, missing):
    if missing == 'API':
        del world.session.pipeline
    else:
        world.session.pipeline.og.sim.current_time = float('nan')
    world.observer('source_measured', {})
    document = latest(world)
    assert document['native_time_s'] is None
    assert document['native_time_status'].startswith('unavailable_')
    assert document['attempt_count'] is document['max_observed_attempt_count'] is None


def test_long_duration_never_infers_failure_or_qualification(world):
    world.observer('source_measured', {'accepted': False})
    world.clock[0] += 100000.
    world.observer('trial_running', {'accepted': False})
    document = latest(world)
    assert document['elapsed_wall_s'] == 100000.
    assert document['current_failed_stage'] is document['earliest_failed_stage'] is None
    assert document['qualification_status'] == 'not_evaluated_by_diagnostics'
    assert 'timeout' not in document


def test_numeric_native_receipts_are_copied_and_no_stdout_is_emitted(world, capsys):
    values = np.arange(6).reshape(2, 3)
    receipt = {'source_native_points': values, 'attempt_count': 2}
    world.observer('source_measured', receipt)
    assert latest(world)['evidence']['source_native_points'] == values.tolist()
    np.testing.assert_array_equal(receipt['source_native_points'], np.arange(6).reshape(2, 3))
    assert capsys.readouterr().out == ''

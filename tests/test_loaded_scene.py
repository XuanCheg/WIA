"""Verify resident world lifetime. Run: PYTHONPATH=src pytest -q tests/test_loaded_scene.py."""
from types import SimpleNamespace
import json

import pytest

from wia.runtime.loaded_scene import LoadedScene


def test_retained_world_restores_initial_state_and_defers_full_cleanup():
    events = []
    initial = {'particles': [1., 2.], 'robot': [3.]}
    simulator = SimpleNamespace(dump_state=lambda **kw: initial,
        load_state=lambda value, **kw: events.append(('restore', value)),
        pause=lambda: events.append('pause'))
    session = SimpleNamespace(pipeline=SimpleNamespace(og=SimpleNamespace(sim=simulator)),
        invalidate_camera_cache=lambda: events.append('invalidate'))
    native_handle = object()
    session.task_state = {'steps': 0, 'events': [], 'object': native_handle}
    lease = LoadedScene(lambda value: events.append(('close', value)))
    lease.prepare(session)
    initial['particles'][0] = 99.
    session.task_state['steps'] = 8
    session.task_state['events'].append('diagnostic action')
    session._wia_retain_completed = True
    lease.finish(session)
    assert lease.retained
    assert events == [('restore', {'particles': [1., 2.], 'robot': [3.]}), 'pause', 'invalidate']
    assert session._wia_initial_native_state is None
    assert session.task_state == {'steps': 0, 'events': [], 'object': native_handle}
    assert session.task_state['object'] is native_handle
    lease.release()
    assert not lease.retained
    assert events[-1] == ('close', session)
    lease.release()
    assert len(events) == 4


def test_incomplete_native_initialization_uses_ordinary_cleanup():
    closed = []
    session = SimpleNamespace()
    lease = LoadedScene(closed.append)
    lease.finish(session)
    assert closed == [session] and not lease.retained


@pytest.mark.parametrize('state', ['IDLE', 'SCENE_RETAINED'])
def test_same_gpu_owner_can_be_reused_without_displacing_other_processes(tmp_path, monkeypatch, state):
    from wia.runtime import resident_episode as module
    monkeypatch.setattr(module, 'read_service', lambda path: {'state': state, 'pid': 123})
    (tmp_path / 'gpu.json').write_text(json.dumps({'physical_index': 4, 'uuid': 'GPU-owned'}))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: '123, GPU-owned\n')
    assert module.owns_gpu(tmp_path, 4, 'GPU-owned')
    assert not module.owns_gpu(tmp_path, 5, 'GPU-owned')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: '123, GPU-owned\n456, GPU-owned\n')
    assert not module.owns_gpu(tmp_path, 4, 'GPU-owned')

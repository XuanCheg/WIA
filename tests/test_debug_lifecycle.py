"""Run pytest tests/test_debug_lifecycle.py for retention and resume orchestration."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wia.runtime import debug_lifecycle as lifecycle
from wia.evaluation import prepare, run
from wia.integrations.show_harness import build_arguments
from test_evaluation_entry import config


def fixture_episode(tmp_path):
    episode = tmp_path / 'episode'
    episode.mkdir()
    summary = {'termination_reason': 'model_error', 'agent_decisions': 10, 'max_steps': 300}
    (episode / 'summary.json').write_text(json.dumps(summary))
    agent = ['python', 'run_esi_agent.py', '--backend', 'codex_cli', '--output-dir', str(episode),
             '--max-steps', '300', '--model', 'gpt-6.1-sol', '--reasoning-effort', 'high']
    return episode, agent


def test_debug_defaults_and_explicit_recording_override(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch)
    debug = prepare(path, execution_mode='debug')['plans'][0]
    assert debug['environment']['record'] is True
    assert debug['environment']['recording_hz'] == 5
    assert debug['evaluation']['run_mode'] == 'diagnostic'
    args = build_arguments(debug)
    assert args[args.index('--execution-mode') + 1] == 'debug'
    batch = prepare(path)['plans'][0]
    assert batch['environment']['record'] is False
    assert batch['evaluation']['execution_mode'] == 'batch'
    path = config(tmp_path, monkeypatch, environment={'mode':'launch','record':False,'recording_hz':8})
    assert prepare(path, execution_mode='debug')['plans'][0]['environment']['record'] is False


def test_debug_rejects_multiple_scenes(tmp_path, monkeypatch):
    path = config(tmp_path, monkeypatch, tasks=[{'path':'task.yaml'},{'path':'task.yaml'}])
    with pytest.raises(ValueError, match='one launched task'):
        prepare(path, execution_mode='debug')


def test_debug_blocks_without_relaunching_and_explicit_close_exits(tmp_path, monkeypatch):
    _, agent = fixture_episode(tmp_path)
    stages = []
    waits = []
    def sleep(seconds):
        waits.append(seconds)
        assert json.loads((tmp_path/'debug_state.json').read_text())['status']=='waiting'
        if len(waits) == 3:
            lifecycle.submit_control(tmp_path, 'close')
    monkeypatch.setattr(lifecycle.time, 'sleep', sleep)
    monkeypatch.setattr(lifecycle.subprocess, 'run', lambda *a, **k: pytest.fail('Must not automatically retry'))
    result = lifecycle.wait_for_debug_control(tmp_path, url='http://native', gateway_url='http://gateway',
        agent=agent, cwd=tmp_path, record=False, save_status=lambda stage, **kw: stages.append(stage))
    assert result is False and len(waits)==3 and stages==['debug_waiting']
    assert json.loads((tmp_path/'debug_state.json').read_text())['status']=='closed'


def test_resume_keeps_thread_source_budget_and_separate_artifacts(tmp_path, monkeypatch):
    previous, agent = fixture_episode(tmp_path)
    commands = []
    def save_status(stage, **fields):
        if stage == 'debug_waiting':
            lifecycle.submit_control(tmp_path, 'resume' if not commands else 'close')
    def execute(command, **kwargs):
        commands.append(command)
        assert command[command.index('--resume-from')+1] == str(previous)
        assert command[command.index('--max-steps')+1] == '300'
        assert command[command.index('--model')+1] == 'gpt-6.1-sol'
        destination=Path(command[command.index('--output-dir')+1]);destination.mkdir()
        (destination/'summary.json').write_text(json.dumps({'termination_reason':'max_steps','agent_decisions':300,'max_steps':300}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(lifecycle.subprocess,'run',execute)
    lifecycle.wait_for_debug_control(tmp_path,url='http://native',gateway_url='http://gateway',agent=agent,
        cwd=tmp_path,record=False,save_status=save_status)
    assert len(commands)==1
    assert json.loads((previous/'summary.json').read_text())['agent_decisions']==10
    assert json.loads((tmp_path/'debug_state.json').read_text())['episode_dir'].endswith('resume_0001')


def test_coordinator_interrupt_preserves_retained_state(tmp_path, monkeypatch):
    _,agent=fixture_episode(tmp_path)
    def interrupt(_):raise KeyboardInterrupt
    monkeypatch.setattr(lifecycle.time,'sleep',interrupt)
    with pytest.raises(KeyboardInterrupt):
        lifecycle.wait_for_debug_control(tmp_path,url='http://native',gateway_url='http://gateway',agent=agent,
            cwd=tmp_path,record=False,save_status=lambda *a,**k:None)
    assert json.loads((tmp_path/'debug_state.json').read_text())['resumable']
    assert not (tmp_path/'native_exit_code.txt').exists()


def test_batch_advances_after_model_error_and_exception(tmp_path, monkeypatch):
    from wia.evaluation import runner
    path=config(tmp_path,monkeypatch,tasks=[{'path':'task.yaml'}]*3)
    prepared=prepare(path)
    calls=[]
    def execute(plan):
        calls.append(plan['index'])
        if plan['index']==0:return {'termination_reason':'model_error'}
        if plan['index']==1:raise RuntimeError('fixture driver error')
        return {'termination_reason':'max_steps','task_success':False,'evaluation_validity':'complete'}
    monkeypatch.setattr(runner,'create_executor',lambda _:SimpleNamespace(execute=execute))
    monkeypatch.setattr(runner,'_version',lambda root:{})
    result=run(prepared)
    assert calls==[0,1,2]
    assert [x['status'] for x in result['trials']]==['model_error','runtime_error','completed_unsuccessful']


def test_debug_hold_stops_physics_and_resume_reenables_policy():
    from wia.runtime.sim.esi_world_policy import WorldPolicy
    p=WorldPolicy();p.begin(True,0);p.transition('debug_hold')
    assert not p.should_advance('human')
    p.begin(True,0);p.transition('awaiting_model')
    assert p.should_advance('agent')

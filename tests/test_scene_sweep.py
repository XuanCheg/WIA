"""Check deterministic diagnostic bookkeeping. Run `python -m pytest -q tests/test_scene_sweep.py`."""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wia.runtime.sim.esi_scene_validation import run


@pytest.mark.parametrize('run_mode', ['diagnostic', 'benchmark'])
def test_failed_cold_scene_is_recorded_and_next_scene_still_runs(tmp_path, monkeypatch, run_mode):
    script = Path(__file__).resolve().parents[1]/'scripts/check_all_scenes.py'
    main = runpy.run_path(str(script))['main']
    namespace = main.__globals__
    release = tmp_path/'explore_to_achieve/wia_v1/release'
    release.mkdir(parents=True)
    catalogue = [{'alias':alias,'task_file':f'{alias}.yaml'} for alias in ('first','second')]
    (release/'catalogue.json').write_text(json.dumps({'tasks':catalogue}))
    plans = [{'task':{'path':str(tmp_path/row['task_file']),'variant':'adapted','seed':0}}
             for row in catalogue]
    sequence = tmp_path/'sequence.json'
    sequence.write_text('{"actions":[]}')
    monkeypatch.setitem(namespace,'ROOT',tmp_path)
    monkeypatch.setitem(namespace,'prepare',lambda path:{'plans':plans})
    monkeypatch.setitem(namespace,'idle_devices',lambda:[{'index':4,'idle':True,'uuid':'fixture-gpu'}])
    monkeypatch.setitem(namespace,'task_profiles',lambda plans:{})
    def task(path,**kwargs):
        assert kwargs['run_mode'] == run_mode
        return NS(path=Path(path),variant='adapted',seed=0,
                  definition=NS(defaults=NS(max_steps=10)),manifest=lambda episode:{})
    monkeypatch.setitem(namespace,'load_task',task)
    monkeypatch.setitem(namespace,'stop_service',lambda control,**kwargs:None)
    attempts = []
    class Client:
        def __init__(self,control,*args):self.control=control
        def submit(self,argv,output):
            assert argv[argv.index('--run-mode') + 1] == run_mode
            self.request_id=str(len(attempts));attempts.append(output.name)
            response=self.control/'responses'/f'{self.request_id}.json'
            response.parent.mkdir(parents=True,exist_ok=True)
            if len(attempts)==1:
                response.write_text('{"error":"cold scene load failed","reusable":false,"scene_exit_code":1}')
            else:
                response.write_text('{"reusable":true,"scene_exit_code":0}')
                (output/'scorer_validation.json').write_text(
                    '{"completed":true,"errors":[],"evaluation_validity":"complete"}')
        def retry_initialization(self,reason,*,agent_started):
            assert not agent_started
            return False
    monkeypatch.setitem(namespace,'ResidentEpisodeClient',Client)
    # The git hash is provenance only; the fixture has no repository.
    monkeypatch.setattr(namespace['subprocess'],'check_output',lambda *args,**kwargs:'fixture-commit\n')
    output=tmp_path/'results'
    monkeypatch.setattr(sys,'argv',[str(script),'--gpu','4','--output',str(output),'--sequence',str(sequence),
                                   '--run-mode',run_mode])
    assert main()==1
    report=json.loads((output/'results.json').read_text())
    assert report['run_mode'] == run_mode
    assert report['completed'] and attempts==['00_first','01_second']
    assert [row['status'] for row in report['results']]==['runtime_error','passed']
    assert 'cold scene load failed' in report['results'][0]['error']


def test_fixed_sequence_preserves_order_and_distinguishes_blocked_motion(tmp_path):
    observation = NS(snapshot_id=1, action_step=0, captured_at='2026-10-31T00:00:00+00:00',
        sensors=NS(head_depth_m=np.ones((4, 4), dtype=np.float32), wrist_depth_m=None),
        require_initial_overview=lambda identity: None,
        rgb_images=lambda: {'head_rgb': np.zeros((4, 4, 3), dtype=np.uint8)})
    source = NS(begin_episode=lambda text: observation, boundary_observation=observation)
    session = NS(observation_source=source, public_task_text='Synthetic diagnostic task.',
                 initial_overview_identity={'episode_id':'fixture'})
    session.task_evaluator = NS(record=lambda obs, action: {
        'evaluation_validity':'complete', 'score_percent':0., 'task_success':False})
    commands = []
    def execute(command, **kwargs):
        commands.append(command.to_dict())
        observation.action_step += 1
        observation.snapshot_id += 1
        return {'success':False, 'completed':False, 'timeout':False, 'reason':'base_motion_blocked'}
    executor = NS(available_actions=('BASE_FWD','BASE_BACK'), primitive_resolution=None,
                  macro_limits=None, execute_macro=execute)
    sequence = {'actions':[{'action':'BASE_FWD','distance_m':.05}, {'action':'BASE_BACK','distance_m':.05}]}
    result = run(session, executor, tmp_path, sequence)
    assert commands == sequence['actions']
    assert result['completed'] and not result['errors'] and result['model_calls'] == 0
    assert result['evaluation_validity'] == 'complete'
    assert [r['reason'] for r in result['actions']] == ['base_motion_blocked'] * 2
    assert result['observations'][-1]['evaluation']['score_percent'] == 0.
    assert json.loads((tmp_path/'scorer_validation.json').read_text())['completed']


def test_temporal_transition_and_done_use_public_owner_protocol(tmp_path):
    observation = NS(snapshot_id=1, action_step=0, captured_at='2026-10-31T00:00:00+00:00',
        sensors=NS(head_depth_m=np.ones((4, 4), dtype=np.float32), wrist_depth_m=None),
        require_initial_overview=lambda identity: None,
        rgb_images=lambda: {'head_rgb':np.zeros((4, 4, 3), dtype=np.uint8)})
    phase = NS(phase='phase_1')
    done = [False]
    calls = []
    def transition(event, answer, **boundary):
        assert event == 'phase_1_complete' and answer is None
        assert boundary == {'snapshot_id':observation.snapshot_id, 'action_step':observation.action_step}
        phase.phase = 'phase_2'
        observation.snapshot_id += 1
        calls.append('transition')
        return {'ok':True}
    def finish(**boundary):
        done[0] = True
        calls.append('DONE')
        return {'ok':True}
    source = NS(begin_episode=lambda text: observation, boundary_observation=observation,
                report_temporal=transition, report_done=finish)
    session = NS(observation_source=source, public_task_text='Synthetic diagnostic task.',
                 initial_overview_identity={'episode_id':'fixture'}, temporal_phase_controller=phase)
    session.task_evaluator = NS(record=lambda obs, action: {
        'evaluation_validity':'complete' if done[0] else ('partial' if phase.phase=='phase_1' else 'pending_done'),
        'score_percent':0. if done[0] else None, 'task_success':False})
    def execute(command, **kwargs):
        assert phase.phase == 'phase_2'
        calls.append(command.action)
        observation.action_step += 1
        observation.snapshot_id += 1
        return {'success':True, 'completed':True, 'timeout':False, 'reason':'completed'}
    executor = NS(available_actions=('MV_UP',), primitive_resolution=None, macro_limits=None, execute_macro=execute)
    result = run(session, executor, tmp_path, {'actions':[{'action':'MV_UP','distance_m':.02}]})
    assert calls == ['transition','MV_UP','DONE']
    assert result['completed'] and not result['errors']
    assert result['pending_coverage'] and result['evaluation_validity'] == 'complete'
    assert [e['event'] for e in result['events']] == ['phase_1_complete','DONE']

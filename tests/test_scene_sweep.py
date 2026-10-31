"""Check deterministic diagnostic bookkeeping. Run `python -m pytest -q tests/test_scene_sweep.py`."""
import json
from types import SimpleNamespace as NS

import numpy as np

from wia.runtime.sim.esi_scene_validation import run


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

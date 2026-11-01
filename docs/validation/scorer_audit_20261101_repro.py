"""Reproduce scorer audit findings using synthetic CPU measurements.

Run from WIA: PYTHONPATH=src:tests:../Show-Harness-Explore python
docs/validation/scorer_audit_20261101_repro.py
Writes a JSON report beside this file. It does not start a simulator, call a
model, or alter task definitions. Production terminal scoring has been fixed
since the initial audit; the other counterexamples still reproduce.
"""
from copy import deepcopy
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import pytest

from wia.candidates.high import evaluate as high
from wia.candidates.medium import evaluate as medium
from wia.tasks.completion import progress
from wia.authoring.converters.composite_task import CompositeEvaluator
from wia.authoring.converters.temporal_understanding_agent_observation import NativePlacementProgress
import test_candidate_high as h
import test_candidate_medium as m
import harness_integration.test_wia_room_collection as rooms
from harness_integration.test_done_declaration import Agent, DoneEnvironment, envelope
from wia.runtime.agent.zero_shot import run_episode
from wia.runtime.commands.run_esi_agent import finalize_evaluation_summary


def run():
    rows = []
    for prototype, factory in [('H1', h.h1), ('H2', lambda: h.h2(False)),
                               ('H5', lambda: h.h5(False)), ('M1', m.m1)]:
        contract, frames = factory()
        frame = deepcopy(frames[0])
        value = (high if prototype.startswith('H') else medium)(prototype, contract, frame, [])
        rows.append({'case': prototype + '_known_initial_incomplete',
            'observed_validity': value['evaluation_validity'],
            'unknown_checks': [key for key, check in value['checks'].items() if check is None],
            'expected': 'Complete measured negative score; the required future event has not occurred.'})

    contract = {'jobs': {'job': {'station': 'station', 'duration_s': 1.}},
                'delivery_region': 'delivery', 'deadline_s': 30., 'handling_s': 2.}
    frame = m.frame(0., {'job': m.obj(4., native_id='native_job')},
                    {'station': m.region(1.), 'delivery': m.region(2.), 'stock': m.region(4.)})
    value = medium('M4', contract, frame, [])
    rows.append({'case': 'M4_known_initial_incomplete', 'observed_validity': value['evaluation_validity'],
        'unknown_checks': [key for key, check in value['checks'].items() if check is None],
        'expected': 'No loading/delivery has occurred; both transport obligations are false, not missing.'})

    contract, frames = h.h3(2)
    value = high('H3', contract, frames[0], [])
    rows.append({'case': 'H3_short_initial_dwell_marked_missing',
        'observed_validity': value['evaluation_validity'],
        'unknown_checks': [key for key, check in value['checks'].items() if check is None],
        'expected': 'Known episode dwell shorter than the threshold is incomplete; missing samples are separate.'})

    contract, frames = h.h4()
    before = high('H4', contract, frames[-1], frames[:-1])
    changed = deepcopy(frames)
    changed[-1]['process_evidence']['approach_clear'] = False
    after = high('H4', contract, changed[-1], changed[:-1])
    rows.append({'case': 'H4_collision_only_changes_success', 'before': before['task_success'],
        'after': after['task_success'], 'changed_input': 'approach_clear only',
        'expected': 'Collision statistics must not change task success under the user Safety policy.'})

    parameters = {'maximum_release_speed_m_s': .02,
                  'maximum_release_angular_speed_rad_s': .1, 'release_stability_steps': 2}
    measured = {'gripper': {'holding': False, 'constraint_active': False,
        'release_pending': False, 'object_name': None}, 'destination_relation': True,
        'linear_speed_m_s': 0., 'angular_speed_rad_s': 0., 'robot_withdrawn': True,
        'robot_terminal_safe': True, 'target_dynamic_collidable': True}
    tracker = NativePlacementProgress(parameters)
    tracker.update(measured, 'target', count_step=True)
    before = tracker.update(measured, 'target', count_step=True)
    measured['robot_terminal_safe'] = False
    after = tracker.update(measured, 'target', count_step=True)
    rows.append({'case': 'shared_temporal_placement_safety_changes_delivery',
        'before_available_and_placed': before, 'after_available_and_placed': after,
        'changed_input': 'robot_terminal_safe only',
        'expected': 'Target release/support/speed/withdrawal are unchanged; generic Safety is statistical.'})

    with TemporaryDirectory(prefix='wia_scorer_audit_') as directory:
        folder = Path(directory)
        with pytest.MonkeyPatch.context() as patch:
            episode = rooms.episode.__wrapped__(patch, folder)
            evaluator, current, observation, _ = episode
            completed = rooms.finish(episode)
            current['native_time_s'] -= 1.
            regressed = evaluator.record(observation)
            rows.append({'case': 'room_collection_backward_clock_keeps_complete_score',
                'before_task_success': completed['task_success'],
                'after_validity': regressed['evaluation_validity'],
                'after_completion_score': progress(regressed)['completion_score'],
                'faults': evaluator.faults,
                'expected': 'A reversed native clock must invalidate current inventory evidence.'})

        member = {'id': 'goal', 'source_identity': 'fixture.json',
            'public': {'objective': 'Visit the required room.'},
            'evaluation': {'module': 'wia.authoring.converters.cognitive_mapping_connectivity',
                'operation': 'fixture', 'parameters': {'required_regions': ['room'], 'required_order': []}}}
        state = {'task_success': True, 'snapshot_id': 1, 'action_step': 0,
            'evaluation_validity': 'complete', 'completion_items': [
                {'id': 'room:room', 'completed': True, 'evaluation_validity': 'valid'}],
            'motion_faults': [], 'constraints_satisfied': True,
            'continuous_constraints_satisfied': True, 'task_constraints_satisfied': True}
        composite = CompositeEvaluator.__new__(CompositeEvaluator)
        composite.members, composite.children = [member], {'goal': NS(record=lambda *a, **k: deepcopy(state))}
        composite.blocked, composite.last, composite.completed = {}, {}, {}
        composite.revised, composite.answer_sheet, composite.episode = True, False, 1
        composite.output = folder
        before = composite.record(NS(snapshot_id=1, action_step=0))
        state.update(motion_faults=['native_nonfloor_contact'], constraints_satisfied=False,
                     continuous_constraints_satisfied=False)
        after = composite.record(NS(snapshot_id=1, action_step=0))
        rows.append({'case': 'composite_reintroduces_child_safety_failure',
            'before_task_success': before['task_success'], 'after_task_success': after['task_success'],
            'after_child_task_constraints': state['task_constraints_satisfied'],
            'expected': 'A Safety-only child contact must not change the composed task result.'})
        environment = DoneEnvironment()
        wrapper, native = envelope(folder / 'budget_private')
        wrapper.begin_episode(environment.source.agent_observation())
        environment.rig.task_evaluator = wrapper
        terminal = asyncio.run(run_episode(environment, Agent('{"action":"BASE_FWD"}'),
            folder / 'budget_episode', max_steps=1,
            success_provider=lambda: wrapper.done_result['task_success'] if wrapper.done_result else None))
        finalize_evaluation_summary(terminal, wrapper.done_result['progress'], 'benchmark')
        rows.append({'case': 'step_budget_without_done_scores_actual_terminal_state',
            'status': 'fixed',
            'termination_reason': terminal['termination_reason'],
            'done_declarations': len(environment.declarations) + len(native.declarations),
            'model_declared_done': terminal['model_declared_done'],
            'task_success': terminal['task_success'],
            'score_percent': terminal['score_percent'],
            'agent_failure_countable': terminal['agent_failure_countable'],
            'evaluation_validity': terminal['evaluation_validity'],
            'expected': 'Score actual terminal completion while retaining model_declared_done=false.'})
    return {'schema': 'wia_scorer_audit_cpu_counterexamples_v1', 'native_simulation_run': False,
            'production_implementation_changed': True,
            'repair_scope': 'Terminal scoring without model DONE only.', 'cases': rows}


if __name__ == '__main__':
    result = run()
    path = Path(__file__).with_name('scorer_audit_20261101_cases.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

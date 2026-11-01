"""Exercise the production runner without a model.

Run through ``python scripts/check_all_scenes.py --driver mcp --gpus 4,5,6,7
--run-mode benchmark --output outputs/batch_RUN``. The coordinator owns startup.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from pathlib import Path
import time

from core.agent.decision import ModelReply
from core.agent.image_inputs import image_inputs
from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
from wia.runtime.agent.zero_shot import run_episode
from wia.runtime.scene_capture_queue import write_json
from wia.tasks.loader import load_task, public_episode_contract, sha256


class FixedPolicy:
    """Replay public commands; the policy never reads private scoring evidence."""

    def __init__(self, sequence, *, public_task=None, episode_policy=None):
        self.sequence = sequence
        self.index = 0
        self.public_task, self.episode_policy = public_task, episode_policy
        self.decisions = 0
        self.history_survey_attempted = False

    def prepare_request(self, request):
        if self.public_task is None:
            return request
        from .prompts import PromptAssembler
        return PromptAssembler().build(request, self.public_task, self.episode_policy,
                                       first_turn=self.decisions == 0)

    def description(self):
        return {'backend': 'fixed_validation', 'model': None, 'task_solving_claim': False}

    def request_images(self, request):
        return image_inputs(request.sensor_inputs)

    def provenance(self):
        return {'request_accounting': {'model_calls': 0, 'api_requests': 0}}

    async def close(self):
        pass

    async def decide(self, request):
        from wia.tasks.temporal_policy import survey_action_allowed
        from .task_response import TaskDecisionInput
        self.decisions += 1
        context = request.action_context or {}
        command_index = self.index
        command = (self.sequence['actions'][self.index]
                   if self.index < len(self.sequence['actions']) else {'action': 'DONE'})
        if context.get('temporal_phase') == 'phase_1' and not survey_action_allowed(command['action']):
            command = {'action': 'DONE'}
            survey = True
        elif (context.get('history_collection_survey_available') and self.index >= 4
                and not self.history_survey_attempted):
            command = {'action': 'DONE'}
            survey = True
            self.history_survey_attempted = True
        else:
            survey = False
        if command['action'] != 'DONE':
            self.index += 1
        observation_key = 'DONE' if command['action'] == 'DONE' else str(command_index)
        text = self.sequence.get('world_observations', {}).get(observation_key, 'Fixed diagnostic sequence.')
        answer = self.sequence.get('answers', {}).get(observation_key)
        if (command['action'] == 'DONE' and not survey and answer is None
                and isinstance(request, TaskDecisionInput) and request.task_answer_keys):
            answer = dict.fromkeys(request.task_answer_keys, 'unsure')
        if command['action'] == 'DONE' and context.get('temporal_phase') == 'phase_2':
            match = re.search(r'Original answer options:\s*(\[[^\n]+\])', request.observation.task_text)
            if match is None:
                raise ValueError('Temporal diagnostic needs explicit public answer options.')
            # Fixed first-public-option policy, independent of the hidden answer.
            answer = str(json.loads(match.group(1))[0])
        payload = {'command': command, 'world_observation': text, 'answer': answer,
                   'arrived_rooms': [], 'survey_complete': survey}
        if (request.action_context or {}).get('observation_declarations_available'):
            payload['observed_objects'] = []
        if isinstance(request, TaskDecisionInput):
            return ModelReply(json.dumps(payload))
        return ModelReply(json.dumps(command), metadata={k: v for k, v in payload.items() if k != 'command'})


async def validate(args):
    output = args.output.resolve()
    manifest_path = output/'episode_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    task = load_task(args.task, variant=args.variant, seed=args.seed, run_mode=args.run_mode)
    sequence = json.loads(args.sequence.read_text())
    report = {'schema': 'wia_mcp_scene_validation_v1', 'model_calls': 0,
              'benchmark_score': False, 'driver': 'production_runner_http_mcp',
              'sequence': sequence, 'observations': [], 'actions': [], 'errors': [],
              'completed': False, 'started_at': time.time()}
    episode_policy = {key: manifest[key] for key in ('episode_id', 'max_steps', 'world_mode',
        'world_advances_during_inference', 'observation_condition', 'action_mode')}
    policy = FixedPolicy(sequence, public_task=task.public_payload, episode_policy=episode_policy)
    contract = public_episode_contract(manifest, sha256(manifest_path))

    def save():
        report['elapsed_s'] = time.time()-report['started_at']
        write_json(output/'scorer_validation.json', report)

    class CheckedEnvironment(MCPRobotEnvironment):
        async def observe(self):
            observation = await super().observe()
            from core.agent.depth_inputs import DepthSnapshot
            depth = DepthSnapshot.from_observation(observation)
            if not depth.arrays:
                raise ValueError('No native depth in delivered observation.')
            row = {'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                   'rgb_views': {key: list(value.shape) for key, value in observation.rgb_images().items()},
                   'depth_views': list(depth.arrays)}
            if self.simulation_clock_required:
                row['simulation_clock'] = self.action_context['simulation_clock']
            elif task.definition.private.evaluation.parameters.get('prototype') == 'M4':
                raise ValueError('M4 requires an observation-bound public simulation clock.')
            value = json.loads((output/'private_evaluation/state.json').read_text())
            from wia.native.esi_category_evaluation import validate_episode_evaluation
            validate_episode_evaluation(value, observation, task.task_text, task)
            row['evaluation'] = {key: value.get(key) for key in
                ('evaluation_validity', 'score_percent', 'task_success', 'evaluation_finalized',
                 'failure_stage', 'failure_reason', 'measurement_error')}
            validity = value.get('evaluation_validity')
            native = value.get('native_result') or {}
            pending = validity in ('pending_done', 'partial') or (
                self.temporal_phase == 'phase_1' and validity == 'unavailable'
                and not native.get('measurement_error') and not native.get('failure_reason'))
            if validity != 'complete' and not pending:
                report['errors'].append({'stage': 'boundary', 'snapshot_id': observation.snapshot_id,
                                         'error': 'score_unavailable', 'evaluation': row['evaluation']})
            report['observations'].append(row)
            save()
            return observation

    try:
        async with CheckedEnvironment(args.url, run_id=manifest['episode_id'], task_text=task.task_text,
                action_mode='parameterized', world_advances_during_inference=True,
                episode_contract=contract) as environment:
            latest_progress = {}

            def success():
                from wia.native.esi_category_evaluation import validate_episode_evaluation
                value = json.loads((output/'private_evaluation/state.json').read_text())
                result = validate_episode_evaluation(value, environment.last_observation, task.task_text, task)
                latest_progress.clear()
                latest_progress.update(value.get('progress', {}))
                write_json(output/'episode/evaluation.json', value)
                return result

            summary = await run_episode(environment, policy, output/'episode',
                max_steps=manifest['max_steps'], max_invalid_retries=0,
                action_mode='parameterized', episode_manifest=manifest,
                integrity_check=task.verify_unchanged, success_provider=success)
            from wia.runtime.commands.run_esi_agent import finalize_evaluation_summary
            finalize_evaluation_summary(summary, latest_progress, args.run_mode)
            write_json(output/'episode/summary.json', summary)
            report['termination_reason'] = summary['termination_reason']
            report['actions'] = summary['executed_actions']
            report['temporal_transitions'] = summary.get('temporal_transitions', [])
            for key in ('error', 'terminal_evaluation_error', 'done_declaration_error', 'session_cleanup_error'):
                if summary.get(key):
                    report['errors'].append({'stage': 'runner', 'error': summary[key]})
            if summary['termination_reason'] != 'model_done':
                report['errors'].append({'stage': 'runner', 'error': summary['termination_reason']})
            if summary['executed_actions'] != [row['action'] for row in sequence['actions']]:
                report['errors'].append({'stage': 'actions', 'error': 'Fixed sequence was not fully executed in order.'})
            final = json.loads((output/'private_evaluation/state.json').read_text())
            report['evaluation_validity'] = final.get('evaluation_validity')
            report['final_evaluation'] = final
            score = final.get('score_percent')
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
                report['errors'].append({'stage': 'final', 'error': 'No finite terminal score.'})
            if final.get('requires_done_declaration') and not final.get('evaluation_finalized'):
                report['errors'].append({'stage': 'final', 'error': 'Evaluation was not finalized.'})
            write_json(output/'final_evaluation.json', final)
        from wia.evaluation.safety import snapshot_safety_log
        snapshot_safety_log(output)
        if not (output/'final_safety.json').exists():
            report['errors'].append({'stage': 'safety', 'error': 'Missing independent Safety artifact.'})
        report['completed'] = True
    except Exception as error:
        report['errors'].append({'stage': 'transport', 'error': f'{type(error).__name__}: {error}'})
    finally:
        save()
    return int(not report['completed'] or bool(report['errors']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--run-mode', required=True)
    return asyncio.run(validate(parser.parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())

"""Build immutable multi-goal definitions from category composition policies.

Run src/wia/runtime/commands/compose_esi_tasks.py --campaign-root ~/esi_category_conversion/CAMPAIGN.
Policies choose compatible existing worlds. This module preserves their native
initialization, original source QA and every individual scoring condition.
"""
from __future__ import annotations

from wia.tasks.schema import resolve_episode_defaults

from copy import deepcopy
import json
from pathlib import Path

from wia.runtime.scene_capture_queue import digest


def goal_id(record):
    return 'goal_' + digest(record['source_identity'])[:20]


def build_group(group, records, tasks, scenes, destination, *, joint_world=None):
    """Build a shared definition; TaskEdit2 Mapping requires certified conditions.

    ``joint_world`` optionally supplies a curated ``task``/``scene`` and a
    private ``contract`` with verified membership, source relation revalidation,
    configuration identity and provenance. It never bypasses native gates.
    """
    representative = records[group['representative_id']]
    members = [records[key] for key in group['member_ids']]
    task, scene = deepcopy(tasks[representative['candidate_id']]), deepcopy(scenes[representative['candidate_id']])
    strict_mapping = group.get('composition_policy') == 'task_edit2_mapping_v1'
    if strict_mapping:
        from wia.authoring.converters.composition_mapping import CONSTRAINED, _compatibility_signature
        if len(members) > 1 and any(member['small_task'] in CONSTRAINED for member in members):
            raise ValueError('Mapping route tasks must remain independent episodes.')
        if joint_world is not None:
            contract = joint_world.get('contract', {})
            ids = set(group['member_ids'])
            if (contract.get('compatibility_verified') is not True
                    or set(contract.get('member_ids', [])) != ids
                    or set(contract.get('source_relations_revalidated', [])) != ids
                    or any(not contract.get(key) for key in ('configuration_id', 'provenance',
                        'world_state_id', 'start_state_id', 'phase_id', 'conditions_id',
                        'robot_configuration_id', 'controller_id', 'observation_profile_id',
                        'loaded_geometry_id', 'inventory_id'))
                    or contract.get('reset_seed') != joint_world.get('task', {}).get('defaults', {}).get('seed')):
                raise ValueError('Curated joint world requires explicit source and configuration proof.')
            task, scene = deepcopy(joint_world['task']), deepcopy(joint_world['scene'])
        elif len(members) > 1:
            signatures = [_compatibility_signature(tasks[member['candidate_id']],
                                                   scenes[member['candidate_id']]) for member in members]
            if None in signatures or len(set(signatures)) != 1:
                raise ValueError('Mapping shared episodes require identical verified actual worlds and inputs.')
        budget = group.get('episode_budget')
        if len(members) > 1 and (not isinstance(budget, dict) or not budget.get('derivation')
                                or type(budget.get('max_steps')) is not int or budget['max_steps'] <= 0
                                or not budget.get('wall_timeout_s') or budget['wall_timeout_s'] <= 0
                                or not budget.get('evidence')):
            raise ValueError('Shared Mapping requires an explicit obligation-derived episode budget and evidence.')
    private_members, public_goals, tags = [], [], []
    for member in members:
        original = tasks[member['candidate_id']]
        raw = member.get('composition_source_raw') or json.loads(Path(member['original_source']).read_text())
        options = raw.get('options_json', [])
        options = json.loads(options) if isinstance(options, str) else options
        public_goals.append({'id': goal_id(member), 'big_task': member['big_task'],
            'small_task': member['small_task'], 'source': member['source_identity'],
            **deepcopy(original['public']), 'original_question': raw.get('question', ''),
            'original_options': options})
        private_members.append({'id': goal_id(member), 'source_identity': member['source_identity'],
            'source_sha256': member['source_sha256'], 'public': deepcopy(original['public']),
            'evaluation': deepcopy(original['private']['evaluation']),
            'bindings': deepcopy(original['private']['bindings']),
            'qa': {'question': raw.get('question'), 'answer': raw.get('answer'),
                   'answer_type': raw.get('answer_type'), 'options': options}})
        if strict_mapping:
            private_members[-1]['small_task'] = member['small_task']
            private_members[-1]['public_id'] = f'goal_{len(private_members)}'
            public_goals[-1]['id'] = private_members[-1]['public_id']
            public_goals[-1].pop('source', None)
        tag = {'big_task': member['big_task'], 'small_task': member['small_task']}
        if tag not in tags:
            tags.append(tag)
    bindings = task['private']['bindings']
    for index, member in enumerate(private_members):
        for key, definition in member['bindings'].items():
            if key not in bindings:
                bindings[key] = definition
            elif bindings[key] != definition:
                alias = f'member_{index}_{key}'
                suffix = 1
                while alias in bindings and bindings[alias] != definition:
                    alias = f'member_{index}_{key}_{suffix}'
                    suffix += 1
                bindings[alias] = definition
    task['private']['evaluation'] = {'type': 'category_operation',
        'module': 'wia.authoring.converters.composite_task', 'operation': 'composite_goals',
        'binding_keys': list(bindings), 'parameters': {'members': private_members}}
    task['requirements']['capabilities'] = list(dict.fromkeys(
        capability for member in members for capability in tasks[member['candidate_id']]['requirements']['capabilities']))
    task['public']['objective'] = 'Complete the following goals in one shared scene.\n' + '\n'.join(
        f"goal_{index + 1}: [{goal['small_task']}] {goal['objective']}" for index, goal in enumerate(public_goals))
    task['public']['success_criteria'] = (
        'Complete all declared native obligations; progress counts confirmed natural items and valid ordered prefixes. '
        'Stop for an observation at each credited goal checkpoint before leaving its terminal state. '
        'Historical checkpoints remain recorded, while continuous protection and safety constraints remain effective. '
        'Only proven equivalent unordered obligations share credit. Unavailable requirements keep their denominator '
        'and make the comparable score unavailable. Original QA is separate from physical completion.')
    if all(m['evaluation'].get('parameters', {}).get('task_revision') == 'task_edit_20261018_v1'
           for m in private_members):
        task['private']['evaluation']['parameters']['task_revision'] = 'task_edit_20261018_v1'
        task['public']['success_criteria'] = (
            'Complete every declared natural obligation. Each child keeps its historical checkpoints; '
            'final placement and terminal relations must still hold when you finish. '
            'For an answer requirement, submit ANSWER: goal_N=<answer> in world_observation, '
            'using the public goal number. Report separate answers at separate observation boundaries. '
            'Shared unordered visits may count once only with equivalent native conditions. '
            'Task constraints, factual completion and observational Safety are reported separately. '
            'Unavailable evidence makes the comparable score unavailable.\n' + '\n'.join(
                f"goal_{index + 1}: {goal['success_criteria']}" for index, goal in enumerate(public_goals)))
    task['public']['constraints'] = list(dict.fromkeys(
        constraint for member in members for constraint in tasks[member['candidate_id']]['public'].get('constraints', [])))
    if strict_mapping:
        from wia.tasks.answer_sheet import mapping_questions
        parameters = task['private']['evaluation']['parameters']
        questions = mapping_questions(private_members)
        parameters.update(answer_sheet=bool(questions), semantic_questions=questions,
                          composition_policy='task_edit2_mapping_v1')
        if joint_world is not None:
            parameters['joint_world_contract'] = deepcopy(joint_world['contract'])
        if group.get('episode_budget'):
            parameters['episode_budget'] = deepcopy(group['episode_budget'])
            task.setdefault('defaults', {})['max_steps'] = group['episode_budget']['max_steps']
            task['defaults']['wall_timeout_s'] = group['episode_budget']['wall_timeout_s']
        if questions:
            canonical_ids = {question['sources'][0]['public_id'] for question in questions}
            task['public']['objective'] = 'Explore once in one continuous episode.\n' + '\n'.join(
                f"goal_{index + 1}: [{goal['small_task']}] {goal['objective']}"
                for index, goal in enumerate(public_goals) if f'goal_{index + 1}' in canonical_ids)
            task['public']['success_criteria'] = (
                'Submit one final JSON answer sheet in one world_observation line, for example '
                'ANSWER: {"goal_1":"<answer>","goal_2":"<answer>"}. '
                'Use the displayed goal numbers. Legacy ANSWER: goal_N=<answer> is also accepted. '
                'No new action or observation is needed solely to answer another question. '
                'Shared unordered physical visits count across compatible questions. '
                'Answer accuracy is the percentage of required unique semantic questions answered correctly; '
                'physical obligations and full task success are reported separately. Missing or invalid answers '
                'are incorrect; unavailable native truth makes the comparable score unavailable.\n' + '\n'.join(
                    f"goal_{index + 1}: {goal['success_criteria']}" for index, goal in enumerate(public_goals)
                    if f'goal_{index + 1}' in canonical_ids))
    elif any('max_steps' in tasks[member['candidate_id']].get('defaults', {}) for member in members):
        task['defaults']['max_steps'] = sum(resolve_episode_defaults(tasks[member['candidate_id']]).max_steps
                                          for member in members)
    signature = digest({'task': task, 'scene': scene, 'sources': [m['source_sha256'] for m in members]})[:24]
    key = 'category_' + signature
    task['id'] = 'task_composite_' + signature
    composite = {'group_id': key, 'member_count': len(members), 'public_goals': public_goals,
                 'sources': [member['source_identity'] for member in members], 'reason': group['reason']}
    directory = Path(destination) / 'candidates' / key
    record = {k: deepcopy(v) for k, v in representative.items() if not k.startswith('composition_')}
    for field in ('job_id', 'capture_input', 'record_revision', 'fingerprint', 'capture_signature',
                  'published_at', 'code_commit', 'review_submitted', 'native_recheck_required'):
        record.pop(field, None)
    record.update(candidate_id=key, revision=signature, status='generated', task_file=str(directory / 'task.yaml'),
        scene_file=str(directory / 'scene.yaml'), diagnostic_dir=str(directory),
        parent_candidate_id=representative['candidate_id'], parent_candidate_ids=[member['candidate_id'] for member in members],
        converter_module='wia.authoring.converters.composite_task', converter_version='composite_goals_v1',
        cpu_conversion_revision=representative.get('cpu_conversion_revision', representative['revision']),
        cpu_converter_version=representative.get('cpu_converter_version', representative['converter_version']),
        physical_task_id=task['id'],
        composite_member_modules=sorted({member['evaluation']['module'] for member in private_members}),
        goal_predicate=task['private']['evaluation'], composite=composite, category_tags=tags,
        review_goals=group.get('review_goals', []), goal_revision='composite_' + signature,
        warning_codes=sorted(set(record.get('warning_codes', [])) | {'composite_native_execution_unverified'}),
        change_summary=record.get('change_summary', []) + [group['reason'],
            'Preserve the representative physical world and every original evaluator; retain completion checkpoints.'])
    return record, task, scene

"""Measure multiple original native goals in one unchanged scene.

Run through scripts/run_esi_agent.py with a composed category task. CPU checks:
python -m pytest -q tests/test_composite_task.py
Original reducers remain private. Completion is latched at an observed action
boundary, before a later return or another goal changes its terminal position.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from copy import deepcopy
import json

from wia.tasks.completion import declared_items, progress, SCORING_VERSION
from wia.tasks.answer_sheet import mapping_questions, parse_answer_sheet, score_questions
from pathlib import Path

from wia.authoring.io import read_json, write_json

MODULE = 'core.esi_converters.composite_task'


def member_config(session, member):
    config = deepcopy(member['evaluation'])
    if config.get('type') != 'category_operation' or module_identity(config.get('module')) == MODULE:
        raise ValueError('Composite members must be original, nonrecursive category operations.')
    definitions = member.get('bindings', {})
    bound = {}
    for key, definition in definitions.items():
        obj = session.env.scene.object_registry('name', definition['name'])
        if obj is None or any(definition.get(field) is not None
                              and getattr(obj, field, None) != definition[field]
                              for field in ('category', 'model')):
            raise ValueError('Original member binding is unavailable: ' + key)
        bound[key] = obj
    config.update(scene=session.scene_name, room=session.room_name,
                  bindings={key: obj.name for key, obj in bound.items()})
    return config, bound


def qualify_operation(session, controller, config, bound):
    from wia.native.esi_category_evaluation import qualify_category_operation
    members = config['parameters']['members']
    results = []
    for member in members:
        try:
            original, objects = member_config(session, member)
            result = qualify_category_operation(session, controller, original, objects)
        except Exception as error:
            result = {'accepted': False, 'reason': f'{type(error).__name__}: {error}'}
        results.append({'id': member['id'], **result})
    accepted = any(row['accepted'] for row in results)
    return {'accepted': accepted, 'members': results,
            'reason': None if accepted else 'No original composite member has qualified native preconditions.',
            'all_members_qualified': all(row['accepted'] for row in results)}


class CompositeEvaluator:
    def __init__(self, session, controller, config, output):
        from wia.native.esi_category_evaluation import create_category_operation_evaluator
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.members = deepcopy(config['parameters']['members'])
        self.revised = all(member['evaluation'].get('parameters', {}).get('task_revision') ==
                           'task_edit_20261018_v1' for member in self.members)
        if not self.members or len({row['id'] for row in self.members}) != len(self.members):
            raise ValueError('Composite goals must have distinct nonempty identities.')
        self.children, self.blocked, self.completed, self.last = {}, {}, {}, {}
        self.questions = deepcopy(config['parameters'].get('semantic_questions'))
        if self.questions is None:
            self.questions = mapping_questions(self.members) if all(
                module_identity(member['evaluation'].get('module', '')).startswith(
                    'core.esi_converters.cognitive_mapping_') for member in self.members) else []
        self.answer_sheet = bool(self.questions or config['parameters'].get('answer_sheet'))
        self.responses, self.response_errors = {}, []
        self.episode = 0
        for member in self.members:
            try:
                original, _ = member_config(session, member)
                self.children[member['id']] = create_category_operation_evaluator(
                    session, controller, original, self.output / member['id'])
            except Exception as error:
                self.blocked[member['id']] = f'{type(error).__name__}: {error}'

    def _accept(self, member, row, observation):
        key = member['id']
        if row is None:
            row = read_json(self.output / key / 'state.json', {})
        if (type(row.get('task_success')) is not bool
                or row.get('snapshot_id') != observation.snapshot_id
                or row.get('action_step') != observation.action_step):
            raise ValueError('Original goal result does not match the current observation boundary.')
        self.last[key] = row
        if row['task_success'] and row.get('evaluation_validity', 'complete') == 'complete':
            self.completed[key] = {'snapshot_id': observation.snapshot_id,
                'action_step': observation.action_step, 'native_result': deepcopy(row)}

    def begin_episode(self, observation):
        self.episode += 1
        self.completed, self.last = {}, {}
        self.responses, self.response_errors = {}, []
        for member in self.members:
            key = member['id']
            if key in self.blocked:
                continue
            try:
                self._accept(member, self.children[key].begin_episode(observation), observation)
            except Exception as error:
                self.blocked[key] = f'{type(error).__name__}: {error}'
        return self.record(observation, action=None)

    def sample_step(self):
        for member in self.members:
            key = member['id']
            if key in self.blocked:
                continue
            try:
                sample = getattr(self.children[key], 'sample_step', None)
                if callable(sample):
                    sample()
            except Exception as error:
                self.blocked[key] = f'{type(error).__name__}: {error}'

    def report_arrival(self, room, observation):
        # Declarations are validated independently against each original goal.
        for key, child in self.children.items():
            if key in self.blocked:
                continue
            report = getattr(child, 'report_arrival', None)
            if callable(report):
                try:
                    report(room, observation)
                except Exception as error:
                    self.blocked[key] = f'{type(error).__name__}: {error}'

    def report_answer(self, answer, observation):
        """Route an explicitly addressed answer; never broadcast private answers."""
        if self.answer_sheet:
            addresses = {f'goal_{index}': member['id'] for index, member in enumerate(self.members, 1)
                         if any(source['member_id'] == member['id'] for question in self.questions
                                for source in question['sources'])}
            accepted, errors, structured = parse_answer_sheet(answer, addresses, required_addresses=[])
            if structured:
                missing = dict.fromkeys(question['sources'][0]['public_id'] for question in self.questions
                    if not any(source['public_id'] in accepted for source in question['sources']))
                errors.extend({'question_id': public_id, 'reason': 'missing_question'} for public_id in missing)
            if structured:
                self.responses = {}
                self.response_errors = []
            elif not accepted and not any(error.get('question_id') for error in errors):
                self.responses = {}
            for error in errors:
                self.responses.pop(error.get('question_id'), None)
            self.response_errors.extend(errors)
            for address, value in accepted.items():
                key = addresses[address]
                if key in self.blocked:
                    continue
                report = getattr(self.children[key], 'report_answer', None)
                if not callable(report):
                    self.blocked[key] = 'Required child answer evaluator is unavailable.'
                    continue
                try:
                    report(value, observation)
                    self.responses[address] = value
                except Exception as error:
                    # SourceAnswerMappingEvaluator accepts arbitrary strings;
                    # a callback exception is a native/scorer failure.
                    self.blocked[key] = f'{type(error).__name__}: {error}'
            return {'accepted_question_ids': list(accepted), 'agent_response_errors': errors}
        address, separator, value = str(answer).partition('=')
        if not separator or not value.strip():
            raise ValueError('Composite answers require goal_N=<answer>.')
        addresses = {f'goal_{index}': member['id'] for index, member in enumerate(self.members, 1)}
        key = addresses.get(address.strip())
        if key is None or key in self.blocked:
            raise ValueError('The requested public goal is unavailable.')
        report = getattr(self.children[key], 'report_answer', None)
        if not callable(report):
            raise ValueError('The requested public goal has no answer requirement.')
        report(value.strip(), observation)

    def record(self, observation, *, action=None):
        for member in self.members:
            key = member['id']
            if key in self.blocked:
                continue
            try:
                self._accept(member, self.children[key].record(observation, action=action), observation)
            except Exception as error:
                self.blocked[key] = f'{type(error).__name__}: {error}'
        goals = [{'id': member['id'], 'source': member['source_identity'],
                  'objective': member['public']['objective'],
                  'completed': (self.last.get(member['id'], {}).get('task_success') is True
                      if self.revised else member['id'] in self.completed),
                  'blocked': self.blocked.get(member['id']),
                  'completion_evidence': self.completed.get(member['id']),
                  'latest_native_result': self.last.get(member['id'])}
                 for member in self.members]
        items, by_key, overlap = [], {}, []
        denominator_known = True
        constraints_satisfied = True
        for member, goal in zip(self.members, goals):
            key = member['id']
            latest = self.last.get(key, {})
            faults = latest.get('faults') or latest.get('motion_faults') or latest.get('invalid_reasons') or []
            checks = latest.get('checks', latest.get('criteria', {}))
            invariant_keys = [name for name in checks if any(token in name for token in
                ('continuous', 'preserv', 'unchanged', 'currently_safe', 'current_native_state_valid'))]
            safe = (not faults and latest.get('continuous_constraints_satisfied', True)
                    and latest.get('constraints_satisfied', True)
                    and all(checks[name] for name in invariant_keys))
            constraints_satisfied = constraints_satisfied and bool(safe)
            # Revised children own their historical checkpoints. Terminal
            # predicates must describe the current world, not a past success.
            measured = deepcopy(latest if self.revised else self.completed.get(key, {}).get('native_result', latest))
            if key in self.blocked:
                measured = {'task_success': False, 'evaluation_validity': 'unavailable',
                    'completion_items': latest.get('completion_items', declared_items(member['evaluation'])),
                    'failure_stage': 'child_evaluator', 'failure_reason': self.blocked[key]}
                if measured['completion_items'] is not None:
                    for item in measured['completion_items']:
                        item['evaluation_validity'] = 'unavailable'
            elif not measured:
                measured = {'task_success': False, 'evaluation_validity': 'unavailable',
                            'completion_items': declared_items(member['evaluation'])}
            measured['continuous_constraints_satisfied'] = bool(safe)
            child = progress(measured)
            goal.update(progress=child, evaluation_validity=child['evaluation_validity'],
                        continuous_constraints_satisfied=bool(safe))
            if child['completion_items'] is None:
                denominator_known = False
                continue
            config = member['evaluation']
            unordered_rooms = (config.get('parameters', {}).get('arrival_report_required') is True
                               and not config['parameters'].get('required_order'))
            for item in child['completion_items']:
                item = deepcopy(item)
                # Exact native predicates within this one shared world can share
                # an unordered room obligation; order/return goals never do.
                signature = None
                if unordered_rooms and item['id'].startswith('room:'):
                    parameters = deepcopy(config['parameters'])
                    parameters.pop('required_regions', None)
                    parameters.pop('scoring_version', None)
                    signature = json.dumps({'module': config['module'], 'operation': config['operation'],
                        'parameters': parameters, 'bindings': member.get('bindings', {}),
                        'item': item['id']}, sort_keys=True)
                    if (self.answer_sheet and module_identity(config['module']) ==
                            'core.esi_converters.cognitive_mapping_connectivity'
                            and (parameters.get('mapping_composition_contract') or {}).get(
                                'compatibility_verified') is True):
                        # These are the actual inputs read by Connectivity's
                        # native occupancy/stability reducer. Question labels,
                        # answers and source paths do not create extra visits.
                        signature = json.dumps({'native_predicate': 'connectivity_stable_room_v1',
                            'world': parameters['mapping_composition_contract'],
                            'item': item['id'], 'parameters': {name: parameters.get(name)
                                for name in ('floor_idx', 'floor_contact_height_tolerance_m',
                                    'minimum_floor_contact_vertical_normal', 'stability_steps',
                                    'stop_samples', 'maximum_stop_speed_m_s',
                                    'maximum_terminal_speed_m_s', 'maximum_stop_angular_speed_rad_s',
                                    'maximum_terminal_angular_speed_rad_s',
                                    'maximum_step_displacement_m')}}, sort_keys=True)
                source = {'member_id': key, 'source_identity': member['source_identity'], 'item_id': item['id']}
                if signature is not None and signature in by_key:
                    existing = by_key[signature]
                    existing['sources'].append(source)
                    # Identical measurements must agree; disagreement is evidence
                    # of an unavailable obligation, not permission to choose a win.
                    if (existing['completed'] != item['completed'] or
                            existing['evaluation_validity'] != item.get('evaluation_validity', 'valid')):
                        existing['evaluation_validity'] = 'unavailable'
                        existing['reason'] = 'equivalent_native_obligations_disagree'
                    continue
                item.update(id=key + ':' + item['id'], sources=[source])
                items.append(item)
                if signature is not None:
                    by_key[signature] = item
        value = {'schema': 'private_composite_goals_v2', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'task_success': all(goal['completed'] for goal in goals) and constraints_satisfied and not self.blocked,
            'completion_items': items if denominator_known else None,
            'evaluation_validity': ('complete' if not self.blocked and denominator_known else 'partial'),
            'goals': goals, 'continuous_constraints_satisfied': constraints_satisfied,
            'source_member_count': len(goals), 'denominator_known': denominator_known,
            'overlap_review': 'Only exact unordered native predicates share credit; uncertain overlap remains separate.',
            'scoring_policy': 'Native obligation progress; original sources retained; unavailable items never shrink the denominator.',
            'qa_preserved': True, 'qa_execution_status': 'not_executed',
            'physical_success_does_not_imply_original_qa_correct': True}
        value.update(progress(value))
        if self.answer_sheet:
            semantic = score_questions(self.questions, self.last, self.blocked,
                                       self.responses, self.response_errors)
            value.update(semantic)
            physical = [item for item in items if item.get('kind') != 'answer_requirement']
            physical_progress = progress({'task_success': False, 'completion_items': physical,
                'evaluation_validity': 'complete' if denominator_known and not self.blocked else 'unavailable'})
            value['physical_obligations'] = physical_progress
            value['qa_execution_status'] = semantic['answer_evaluation_validity']
            value['task_success'] = bool(constraints_satisfied and not self.blocked
                and all(item.get('completed') is True and item.get('evaluation_validity', 'valid')
                        in ('valid', 'complete') for item in physical)
                and not semantic['agent_response_errors']
                and semantic['answer_evaluation_validity'] == 'complete'
                and semantic['correct_answers'] == semantic['required_unique_questions'])
            # Keep native aggregate completion for diagnostics; question sets
            # expose their equal-weight answer accuracy as the headline score.
            value['obligation_score_percent'] = value['score_percent']
            value['score_percent'] = semantic['answer_accuracy_percent']
        write_json(self.output / 'state.json', value)
        return value


def create_evaluator(session, controller, config, output):
    return CompositeEvaluator(session, controller, config, output)

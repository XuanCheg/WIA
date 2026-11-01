"""Typed public task answers. Run: python -m pytest -q tests/test_task_response.py."""
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import json
import re

from core.action_compiler import unique_json_object
from core.agent.backends import structured_reply
from core.agent.decision import DecisionInput, ModelReply, final_text_only
from .observation_declarations import ObservationDecisionInput, validate_declarations


def answer_keys(public_task):
    """Derive answer addresses only from the public question, never private bindings."""
    objective = public_task['objective']
    goals = list(dict.fromkeys(re.findall(r'^goal_\d+(?=:)', objective, re.MULTILINE)))
    if goals:
        return goals
    if 'Which container does each object best fit inside?' in objective:
        return re.findall(r'^- ([^\n]+)$', objective, re.MULTILINE)
    return []


@dataclass(frozen=True)
class TaskDecisionInput(ObservationDecisionInput):
    task_answer_keys: tuple[str, ...] = ()

    @property
    def structured_schema(self):
        if (self.action_context or {}).get('observation_declarations_available'):
            schema = deepcopy(ObservationDecisionInput.structured_schema.fget(replace(self, world_observation_log=True)))
            if not self.world_observation_log:
                schema['properties'].pop('world_observation')
                schema['required'].remove('world_observation')
        else:
            schema = deepcopy(DecisionInput.structured_schema.fget(self))
        if 'command' not in schema['properties']:
            schema = {'type': 'object', 'properties': {'command': schema},
                      'required': ['command'], 'additionalProperties': False}
        value = {'type': 'string', 'minLength': 1, 'maxLength': 4096}
        if self.task_answer_keys:
            value = {'type': 'object', 'properties': {key: dict(value) for key in self.task_answer_keys},
                     'required': list(self.task_answer_keys), 'additionalProperties': False}
        schema['properties']['answer'] = {'anyOf': [value, {'type': 'null'}]}
        if (self.action_context or {}).get('history_collection_survey_available'):
            schema['properties']['answer'] = {'type': 'null',
                'description': 'The question is revealed only after an accepted survey declaration.'}
        schema['properties']['arrived_rooms'] = {'type': 'array', 'maxItems': 32,
            'items': {'type': 'string', 'pattern': '^[a-z][a-z0-9_]*_[0-9]+$'},
            'description': 'Rooms you identify yourself as currently inside; use [] otherwise. '
                'Use the public room name with spaces replaced by underscores, such as kitchen_0.'}
        schema['properties']['survey_complete'] = {'type': 'boolean',
            'description': 'True only when ending a task-requested survey; pair with command DONE. '
                'This advances the survey task without declaring the whole task complete.'}
        schema['required'].extend(('answer', 'arrived_rooms', 'survey_complete'))
        return schema


def enable_task_response(request, public_task):
    return TaskDecisionInput(**{f.name: getattr(request, f.name) for f in fields(DecisionInput)},
                             task_answer_keys=tuple(answer_keys(public_task)))


def unpack_task_response(request, reply):
    """Validate task fields before executing a command or mutating its evaluator."""
    envelope = reply.metadata.get('structured_output', {})
    raw = envelope.get('normalized_output', envelope.get('raw_output', reply.raw_text))
    payload = json.loads(final_text_only(raw), object_pairs_hook=unique_json_object)
    if not isinstance(payload, dict) or set(payload) != set(request.structured_schema['required']):
        raise ValueError('Return all declared fields; use answer=null, arrived_rooms=[], survey_complete=false when unused.')
    answer = payload.pop('answer')
    rooms = payload.pop('arrived_rooms')
    survey = payload.pop('survey_complete')
    if isinstance(answer, str) and answer.strip().upper() == 'SURVEY_COMPLETE':
        raise ValueError('Use the survey_complete field; a control marker is not a task answer.')
    if answer is not None and (request.action_context or {}).get('history_collection_survey_available'):
        raise ValueError('answer must be null until the survey is accepted and its question is revealed.')
    if (not isinstance(rooms, list) or len(rooms) > 32 or any(not isinstance(room, str)
            or not re.fullmatch(r'[a-z][a-z0-9_]*_[0-9]+', room) for room in rooms)):
        raise ValueError('arrived_rooms must contain public room names with underscores, or be empty.')
    if type(survey) is not bool:
        raise ValueError('survey_complete must be a boolean.')
    if not isinstance(payload.get('command'), dict):
        raise ValueError('command must be an object.')
    if survey:
        context = request.action_context or {}
        if (context.get('temporal_phase') != 'phase_1'
                and not context.get('history_collection_survey_available')):
            raise ValueError('No survey completion is available in this task phase.')
        if payload.get('command', {}).get('action') != 'DONE' or answer is not None:
            raise ValueError('Complete the survey with command DONE and answer=null; no motion or final answer accompanies it.')
    if answer is not None:
        values = [answer]
        if request.task_answer_keys:
            if not isinstance(answer, dict) or set(answer) != set(request.task_answer_keys):
                raise ValueError('answer must map every public question/object label to its answer.')
            values = list(answer.values())
        if any(not isinstance(v, str) or not v.strip() or len(v) > 4096 for v in values):
            raise ValueError('Answer values must be nonempty strings, or answer must be null.')
        if len(serialize_answer(answer)) > 4096:
            raise ValueError('The serialized answer exceeds the public transport limit of 4096 characters.')
    declarations = None
    if 'observed_objects' in payload:
        declarations = validate_declarations(payload.pop('observed_objects'))
    base = DecisionInput(**{f.name: getattr(request, f.name) for f in fields(DecisionInput)})
    if request.world_observation_log:
        result = structured_reply(base, json.dumps(payload), reply.metadata)
        if result.raw_text == json.dumps(payload) and 'command' in payload:
            raise ValueError('Invalid command/world_observation output envelope.')
    else:
        if not isinstance(payload.get('command'), dict):
            raise ValueError('command must be an object.')
        result = ModelReply(json.dumps(payload['command']), dict(reply.metadata))
    # Preserve the full model-authored envelope in the public trace.
    result.metadata['structured_output'] = {**envelope, 'raw_output': raw,
        'schema': request.structured_schema, 'envelope': 'command_world_observation_answer'}
    return result, {'answer': answer, 'arrived_rooms': list(dict.fromkeys(rooms)),
                    'survey_complete': survey}, declarations


def serialize_answer(answer):
    return json.dumps(answer, ensure_ascii=False, allow_nan=False) if isinstance(answer, dict) else answer

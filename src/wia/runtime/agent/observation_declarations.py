"""Structured, public observation claims. Run: python -m pytest tests/test_mapping_declarations.py."""
from copy import deepcopy
from dataclasses import dataclass, fields
import json

from core.action_compiler import unique_json_object
from core.agent.decision import DecisionInput, ModelReply, final_text_only


def validate_declarations(value):
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError('observed_objects must be a list with at most 32 entries.')
    for row in value:
        if (not isinstance(row, dict) or set(row) != {'label', 'camera'}
                or any(not isinstance(row[k], str) or not row[k].strip() or len(row[k]) > 160
                       for k in ('label', 'camera'))):
            raise ValueError('Each observed_objects entry requires only nonempty label and camera strings.')
    return deepcopy(value)


@dataclass(frozen=True)
class ObservationDecisionInput(DecisionInput):
    @property
    def structured_schema(self):
        schema = deepcopy(super().structured_schema)
        schema['properties']['observed_objects'] = {'type': 'array', 'maxItems': 32,
            'items': {'type': 'object', 'properties': {
                'label': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                'camera': {'type': 'string', 'minLength': 1, 'maxLength': 160}},
                'required': ['label', 'camera'], 'additionalProperties': False}}
        schema['required'].append('observed_objects')
        return schema


def enable_declarations(request):
    return ObservationDecisionInput(**{f.name: getattr(request, f.name) for f in fields(DecisionInput)})


def unpack_declarations(request, reply):
    if not isinstance(request, ObservationDecisionInput):
        return reply, None
    envelope = reply.metadata.get('structured_output', {})
    raw = envelope.get('normalized_output', envelope.get('raw_output', reply.raw_text))
    payload = json.loads(final_text_only(raw), object_pairs_hook=unique_json_object)
    if not isinstance(payload, dict) or set(payload) != set(request.structured_schema['required']):
        raise ValueError('Return exactly the declared output fields, including observed_objects (use [] for none).')
    declarations = validate_declarations(payload.pop('observed_objects'))
    from core.agent.backends import structured_reply
    base = DecisionInput(**{f.name: getattr(request, f.name) for f in fields(DecisionInput)})
    result = structured_reply(base, json.dumps(payload), reply.metadata)
    if result.raw_text == json.dumps(payload) and 'command' in payload:
        raise ValueError('Invalid observation declaration output envelope.')
    result.metadata['observation_declaration_output'] = raw
    return result, declarations

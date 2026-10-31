"""Drop unknown output keys before validation; run pytest tests/harness_integration/test_output_normalization.py."""
from copy import deepcopy
import json

from core.action_compiler import unique_json_object
from core.agent.backends import structured_reply
from core.agent.decision import ModelReply, final_text_only


def normalize_extra_fields(request, reply):
    """Project known schemas without coercing values or filling missing required fields."""
    original = reply.metadata.get('structured_output', {}).get('raw_output', reply.raw_text)
    try:
        payload = json.loads(final_text_only(original), object_pairs_hook=unique_json_object)
    except (ValueError, TypeError):
        return reply
    if not isinstance(payload, dict):
        return reply
    data = deepcopy(payload)
    ignored = []

    def project(obj, allowed, prefix):
        for key in list(obj):
            if key not in allowed:
                ignored.append(prefix + key)
                del obj[key]

    enveloped = 'command' in data
    if enveloped:
        project(data, request.structured_schema.get('properties', {}), '')
        command = data.get('command')
    else:
        command = data
    if isinstance(command, dict):
        action = command.get('action')
        schema = request.schema
        for branch in schema.get('anyOf', [schema]):
            properties = branch.get('properties', {})
            if action in properties.get('action', {}).get('enum', []):
                project(command, properties, 'command.' if enveloped else '')
                break
    if not ignored:
        return reply
    sanitized = json.dumps(data, allow_nan=False)
    metadata = dict(reply.metadata)
    metadata.pop('structured_output', None)
    result = structured_reply(request, sanitized, metadata) if enveloped else ModelReply(sanitized, metadata)
    if 'structured_output' in result.metadata:
        result.metadata['structured_output'].update(raw_output=original, normalized_output=sanitized)
    return ModelReply(result.raw_text, {**result.metadata, 'output_normalization': {
        'original_raw_output': original, 'normalized_output': sanitized,
        'ignored_fields': ignored, 'policy': 'ignore_unknown_fields_validate_known_values'}})

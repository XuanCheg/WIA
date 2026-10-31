"""Responses wire adapter for shared policies. Run pytest tests/harness_integration/test_context_session.py."""
from copy import deepcopy

from core.agent.api_sessions import _APIEpisodeSession
from core.agent.decision import ModelError
from .context_session import ContextChatSession


class ResponsesContextSession(ContextChatSession):
    """Translate the shared message/tool loop without dropping native reasoning items."""

    backend = 'openai_responses'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response_tool_items = {}
        self._response_final_items = {}

    async def decide(self, request):
        if self.recent_notes or self.state_only or self.history_text_turns is not None:
            self._response_tool_items.clear()
            self._response_final_items.clear()
        return await super().decide(request)

    @staticmethod
    def _call_key(message):
        return tuple(call['id'] for call in message.get('tool_calls', []))

    async def _send_payload(self, endpoint, payload, deadline):
        inputs = []
        for index, message in enumerate(payload['messages']):
            role = message['role']
            if role == 'tool':
                inputs.append({'type': 'function_call_output', 'call_id': message['tool_call_id'],
                               'output': message['content']})
                continue
            key = self._call_key(message)
            if role == 'assistant' and key in self._response_tool_items:
                inputs.extend(deepcopy(self._response_tool_items[key]))
                continue
            saved = self._response_final_items.get(index)
            if role == 'assistant' and saved and saved[0] == message.get('content'):
                inputs.extend(deepcopy(saved[1]))
                continue
            content = message.get('content')
            if isinstance(content, list):
                content = [({'type': 'input_image', 'image_url': part['image_url']['url'],
                             **({'detail': part['image_url']['detail']} if 'detail' in part['image_url'] else {})}
                            if part['type'] == 'image_url' else
                            {'type': 'input_text', 'text': part['text']}) for part in content]
            if content is not None:
                inputs.append({'role': role, 'content': content})
            for call in message.get('tool_calls', []):
                inputs.append({'type': 'function_call', 'call_id': call['id'], **call['function']})
        wire = {'model': payload['model'], 'input': inputs, 'store': False,
                'include': ['reasoning.encrypted_content'],
                'max_output_tokens': payload.get('max_tokens', payload.get('max_completion_tokens')),
                'tools': [{'type': 'function', **tool['function'], 'strict': False}
                          for tool in payload.get('tools', [])]}
        if 'temperature' in payload:
            wire['temperature'] = payload['temperature']
        if 'reasoning_effort' in payload:
            wire['reasoning'] = {'effort': payload['reasoning_effort']}
        for key in ('tool_choice', 'parallel_tool_calls', 'providerOptions'):
            if key in payload:
                wire[key] = deepcopy(payload[key])
        if 'response_format' in payload:
            fmt = payload['response_format']
            wire['text'] = {'format': {'type': fmt['type'], **fmt.get('json_schema', {})}}
        data = await _APIEpisodeSession._post(self, '/responses', wire, deadline)
        if data.get('status') in ('failed', 'incomplete', 'cancelled', 'queued', 'in_progress'):
            raise ModelError('API model failed to complete a final reply.')
        output = data.get('output')
        if not isinstance(output, list):
            raise ModelError('API model returned invalid output items.')
        calls, texts = [], []
        for item in output:
            if item['type'] == 'function_call':
                calls.append({'id': item['call_id'], 'type': 'function',
                              'function': {'name': item['name'], 'arguments': item['arguments']}})
            elif item['type'] == 'message':
                if item.get('status') == 'incomplete' or item.get('role', 'assistant') != 'assistant':
                    raise ModelError('API model returned an incomplete message.')
                for part in item['content']:
                    if part['type'] == 'refusal':
                        raise ModelError('API model refused the decision.')
                    if part['type'] == 'output_text' and item.get('phase') != 'commentary':
                        texts.append(part['text'])
            elif item['type'] != 'reasoning':
                raise ModelError('API model attempted an unauthorized tool operation.')
        message = {'role': 'assistant', 'content': ''.join(texts) if texts else None}
        if calls:
            message['tool_calls'] = calls
            self._response_tool_items[self._call_key(message)] = deepcopy(output)
        else:
            self._response_final_items[len(payload['messages'])] = (message['content'], deepcopy(output))
        return {'choices': [{'message': message, 'finish_reason': 'tool_calls' if calls else 'stop'}],
                'usage': deepcopy(data.get('usage') or {}),
                'providerMetadata': deepcopy(data.get('providerMetadata', {}))}

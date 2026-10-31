"""Gateway wire adapter. Run through `wia eval --config FILE`."""
import json

from core.agent.decision import ModelError
from .context_session import ContextChatSession, InvalidToolDecision
from .responses_context import ResponsesContextSession


class VercelChatSession(ContextChatSession):
    """Route to official providers; context policies belong to the shared engine."""

    log_prefix = 'gateway'

    def _configure_transport_payload(self, payload):
        provider = self.model.split('/', 1)[0]
        if provider not in {'openai', 'anthropic', 'deepseek', 'moonshotai', 'alibaba'}:
            raise ModelError('No verified official Gateway provider mapping for this model.')
        payload['providerOptions'] = {'gateway': {'only': [provider], 'caching': 'auto'}}
        if self.model == 'deepseek/deepseek-v4.1-flash-1017':
            from .input_budget import deepseek_budget
            budget = deepseek_budget(payload)
            with (self.output / 'input_budget.jsonl').open('a') as log:
                log.write(json.dumps({'decision': self.turn_count + 1, **budget}) + '\n')
            if not budget['within_budget']:
                raise ModelError('Input budget exceeded; output reservation is preserved and no request was sent.')

    def _validate_transport_response(self, payload, data):
        provider = self.model.split('/', 1)[0]
        message = (data.get('choices') or [{}])[0].get('message', {})
        metadata = message.get('provider_metadata', data.get('providerMetadata', {}))
        routing = metadata.get('gateway', {}).get('routing', {})
        actual = routing.get('finalProvider')
        with (self.output / 'gateway_routing.jsonl').open('a') as log:
            log.write(json.dumps({'model': self.model, 'allowed_provider': provider,
                'reasoning_effort': payload.get('reasoning_effort'),
                'max_tokens': payload.get('max_tokens'),
                'gateway_caching': payload['providerOptions']['gateway']['caching'], 'routing': routing,
                'usage': data.get('usage')}) + '\n')
        if self.model == 'deepseek/deepseek-v4.1-flash-1017' and (
                actual != 'deepseek' or routing.get('canonicalSlug') != self.model):
            raise ModelError('Pinned DeepSeek model/provider receipt missing or mismatched; refusing the response.')
        if actual is not None and actual != provider:
            raise ModelError('Gateway returned an unexpected provider; refusing the response.')


class VercelResponsesSession(ResponsesContextSession, VercelChatSession):
    """Compose Responses encoding with the same Gateway routing adapter."""

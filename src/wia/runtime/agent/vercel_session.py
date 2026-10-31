"""Official-provider Gateway transport. Run through `wia eval --config FILE`."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from wia.resources import ROOT
import json

from core.agent.api_sessions import ChatCompletionsEpisodeSession
from core.agent.decision import ModelError
from .episode_notes import NOTE_TOOL, NOTE_PROMPT, note_tool, run_note
from .historical_images import HISTORY_IMAGE_TOOL, historical_image_tool, register_current_images, read_historical_image


class InvalidToolDecision(ModelError):
    """A model-authored tool error that consumes one decision without robot motion."""


class VercelChatSession(ChatCompletionsEpisodeSession):
    """Keep Harness history and actions while enforcing upstream routing."""

    def __init__(self, *, current_sensors_only=False, history_policy=None, history_recent_moves=None, max_perception_rounds=None, **kwargs):
        from .history_policy import validate_history_policy
        mode = history_policy or ("current_sensors_full_text" if current_sensors_only else None)
        validate_history_policy({**kwargs, "backend": "openai_chat_completions",
            "base_url": "https://ai-gateway.vercel.sh/v1", "history_policy": mode,
            "history_recent_moves": history_recent_moves})
        self.recent_notes = mode in ("current_sensors_recent2_notes", "current_sensors_last_result_notes")
        self.max_perception_rounds = (4 if max_perception_rounds is None else max_perception_rounds) if self.recent_notes else None
        self._perception_rounds = 0
        self._perception_notice_sent = False
        self._perception_prompt = ((ROOT / 'explore_to_achieve/prompts/tools/perception_budget.md').read_text()
            .format(max_rounds=self.max_perception_rounds)) if self.recent_notes else ''
        self._recent_turns = []
        self._global_context = None
        self._notes = []
        self._note_chars = 0
        self.state_only = mode == "showharness_state"
        self.history_recent_moves = 3 if history_recent_moves is None else history_recent_moves
        current_sensors_only = mode == "current_sensors_full_text" or self.recent_notes
        super().__init__(**kwargs)
        self.current_sensors_only = current_sensors_only
        if self.state_only:
            self.history_policy = "showharness_state"
        self.invalid_decision_consumes_step = True
        self._reasoning_by_index = {}
        self._history_images = {}
        self._historical_rgb_prompt = (ROOT / "explore_to_achieve/prompts/tools/historical_rgb.md").read_text() if current_sensors_only else ""
        if current_sensors_only:
            if self.history_text_turns is not None or self.sliding_window_tokens is not None:
                raise ValueError('Current sensors require complete textual history.')
            self.history_policy = ('current_sensors_last_result_notes' if self.recent_notes else
                                   'current_sensors_full_text_and_returned_reasoning')

    def description(self):
        return {**super().description(), 'invalid_decision_policy': 'consume_step_and_feedback',
                'history_recent_moves': self.history_recent_moves if self.state_only else None,
                'history_completed_turns': 1 if self.recent_notes else None,
                'history_projection': 'previous_public_result_only' if self.recent_notes else None,
                'max_perception_rounds': self.max_perception_rounds,
                'perception_budget_prompt_sha256': hashlib.sha256(self._perception_prompt.encode()).hexdigest() if self.recent_notes else None,
                'note_write_characters_per_decision': 128 if self.recent_notes else None,
                'historical_rgb_prompt_sha256': hashlib.sha256(self._historical_rgb_prompt.encode()).hexdigest() if self.current_sensors_only else None}

    def provenance(self):
        return {**super().provenance(), 'invalid_decision_policy': 'consume_step_and_feedback',
                'history_recent_moves': self.history_recent_moves if self.state_only else None,
                'history_completed_turns': 1 if self.recent_notes else None,
                'history_projection': 'previous_public_result_only' if self.recent_notes else None,
                'max_perception_rounds': self.max_perception_rounds,
                'perception_budget_prompt_sha256': hashlib.sha256(self._perception_prompt.encode()).hexdigest() if self.recent_notes else None,
                'note_write_characters_per_decision': 128 if self.recent_notes else None,
                'historical_rgb_prompt_sha256': hashlib.sha256(self._historical_rgb_prompt.encode()).hexdigest() if self.current_sensors_only else None}

    async def decide(self, request):
        if self.current_sensors_only:
            register_current_images(self, request)
        if self.recent_notes:
            self._perception_rounds = 0
            self._perception_notice_sent = False
            self._reasoning_by_index.clear()
            self._note_chars = 0
            self._turn_start = len(self.request_history())
        self._decision_usage = {}
        self._last_gateway_payload = self._last_gateway_response = None
        try:
            return await super().decide(request)
        except ModelError as error:
            if not str(error).startswith(('API model attempted an unauthorized tool operation',
                                           'API model attempted an unauthorized tool operation.',
                                           'API model returned invalid tool arguments.')):
                raise
            if self._last_gateway_response is None:
                raise
            response = self._last_gateway_response
            message = response['choices'][0]['message']
            calls = message.get('tool_calls') or []
            if not calls or any(not isinstance(c.get('id'), str) for c in calls):
                raise
            feedback = (str(error) + ' This invalid decision consumed one step. No robot action was executed. '
                'Use only the registered observation tools. Return the required action JSON as final text; '
                'do not call an invented command tool. Correct the decision on the next step.')
            history = deepcopy(self._last_gateway_payload['messages'])
            history.append({k: deepcopy(message[k]) for k in
                ('role', 'content', 'tool_calls', 'reasoning', 'reasoning_content', 'reasoning_details') if k in message})
            history[-1].setdefault('role', 'assistant')
            for call in calls:
                history.append({'role': 'tool', 'tool_call_id': call['id'],
                    'content': json.dumps({'ok': False, 'error': feedback})})
            self._restore_reasoning(history)
            self._retain_recent_turn(history)
            self.history = history
            self.turn_count += 1
            self.sensor_ids.append(request.sensor_inputs.snapshot_id)
            self.canonical_ids.append(request.observation.snapshot_id)
            self._add_usage(self.usage, self._decision_usage)
            self._save()
            raise InvalidToolDecision(feedback) from error

    def prepare_request(self, request):
        if self.state_only or self.recent_notes:
            actions = request.observation.action_history
            count = 1 if self.recent_notes else self.history_recent_moves
            request = replace(request, observation=replace(request.observation,
                action_history=actions[-count:] if count else ()))
        request = super().prepare_request(request)
        if self.state_only:
            prompt = request.structured_prompt()
            if self.turn_count:
                prompt = request.resident_prompt + "\nPublic task: " + json.dumps(self.public_task) + "\n" + prompt
            request = replace(request, assembled_prompt=prompt +
                f"\nHistory policy: fresh current-state request with at most {self.history_recent_moves} recent executed actions. "
                "Prior conversation, reasoning, images and tool results are not replayed. "
                "Use current observations and execution feedback; no historical RGB recall tool is available.")
        if self.recent_notes:
            if self._global_context is None:
                self._global_context = {'role': 'system', 'content': request.resident_prompt +
                    '\nPublic task: ' + json.dumps(self.public_task, sort_keys=True) +
                    '\nEpisode policy: ' + json.dumps(self.episode_configuration, sort_keys=True) +
                    '\nHistory policy: only the immediately previous final public decision result is replayed. '
                    'Historical observation prompts, images, tool exchanges and reasoning are omitted. Earlier information is unavailable unless '
                    'saved in episode_note. Current observations and tool results remain available.\n' +
                    NOTE_PROMPT + '\n' + self._historical_rgb_prompt + '\n' + self._perception_prompt}
            request = replace(request, assembled_prompt=request.structured_prompt() +
                f'\nCurrent decision index: {request.decision_step}. Use episode_note for persistent notes; '
                'only the previous final decision result remains in context; execution feedback is in the current observation. Historical RGB can be recalled by decision index. '
                f'The perception budget resets for this decision: {self.max_perception_rounds} rounds available.')
        elif self.current_sensors_only:
            request = replace(request, assembled_prompt=request.structured_prompt() +
                '\nHistory policy: all prior text and returned reasoning are retained, but prior image '
                'attachments and depth-tool measurements are omitted. Historical image labels and '
                'sensor statements describe past observations, not attachments available now. '
                'Current-step images, measurements, and depth tools remain available unchanged. '
                'Use read_historical_rgb(decision_step, camera) to inspect a previously delivered image on demand. '
                'Recalled images are historical evidence and are omitted again next decision.'
                + f' Current decision index: {request.decision_step}; only earlier decisions may be recalled.'
                + ('\n\n' + self._historical_rgb_prompt if self.turn_count == 0 else ''))
        return request

    def request_history(self):
        if self.state_only:
            return []
        if self.recent_notes:
            history = [deepcopy(self._global_context)] if self._global_context else []
            history.extend(deepcopy(item) for turn in self._recent_turns for item in turn)
        else:
            history = super().request_history()
        if not self.current_sensors_only:
            return history
        depth_ids = {call['id'] for item in history for call in item.get('tool_calls', [])
                     if call.get('function', {}).get('name') in {'read_depth_pixels', 'read_depth_region'}}
        for item in history:
            if isinstance(item.get('content'), list):
                item['content'] = [part for part in item['content']
                                   if part.get('type') not in {'image_url', 'input_image', 'image'}]
                if not item['content']:
                    item['content'] = '[Historical image attachment omitted.]'
            if item.get('role') == 'tool' and item.get('tool_call_id') in depth_ids:
                item['content'] = '[Historical depth measurement omitted; query current-step depth if needed.]'
        return history

    def _restore_reasoning(self, messages):
        if self.current_sensors_only:
            for index, fields in self._reasoning_by_index.items():
                if index < len(messages) and messages[index].get('role') == 'assistant':
                    messages[index].update(deepcopy(fields))

    def accept_normalized_output(self, normalized_output):
        """Keep the accepted public result in bounded replay; raw API logs stay unchanged."""
        if self.recent_notes and self._recent_turns:
            self._recent_turns[-1][-1]['content'] = normalized_output

    def _retain_recent_turn(self, history):
        if self.recent_notes:
            current = history[self._turn_start:]
            final = next((item for item in reversed(current)
                          if item.get('role') == 'assistant' and not item.get('tool_calls')
                          and isinstance(item.get('content'), str)), None)
            if final is not None:
                result = final['content']
            else:
                result = json.dumps({'decision_status': 'rejected',
                    'robot_action_executed': False,
                    'feedback': 'Previous decision requested an unauthorized or invalid tool. '
                                'Use current execution/validation feedback to correct the next decision.'})
            self._recent_turns = [[
                {'role': 'user', 'content': 'Previous decision result only. This is historical output; '
                    'current measured execution feedback determines whether the action succeeded.'},
                {'role': 'assistant', 'content': result}]]

    def _complete(self, request, history, *args):
        self._restore_reasoning(history)
        self._retain_recent_turn(history)
        return super()._complete(request, history, *args)

    async def _post(self, endpoint, payload, deadline):
        provider = self.model.split('/', 1)[0]
        if provider not in {'openai', 'anthropic', 'deepseek', 'moonshotai', 'alibaba'}:
            raise ModelError('No verified official Gateway provider mapping for this model.')
        original_payload = payload
        perception_names = {'read_depth_pixels', 'read_depth_region', 'read_rgb_camera', HISTORY_IMAGE_TOOL}
        exhausted = self.recent_notes and self._perception_rounds >= self.max_perception_rounds
        if exhausted and not self._perception_notice_sent:
            payload['messages'].append({'role': 'user', 'content':
                f'Perception round budget exhausted for decision {self._current_decision_step}. All current sensor results describe the same '
                'snapshot. No more perception tools are available. Submit the required final action JSON using '
                'the evidence already available. episode_note remains available with unlimited reads.'})
            self._perception_notice_sent = True
        self._restore_reasoning(payload['messages'])
        payload = deepcopy(payload)
        if self.current_sensors_only and hasattr(self, '_current_decision_step'):
            if not any(t.get('function', {}).get('name') == HISTORY_IMAGE_TOOL for t in payload.get('tools', [])):
                payload.setdefault('tools', []).append(historical_image_tool())
        if self.recent_notes:
            if not any(t.get('function', {}).get('name') == NOTE_TOOL for t in payload.get('tools', [])):
                payload.setdefault('tools', []).append(note_tool())
        if exhausted:
            payload['tools'] = [t for t in payload.get('tools', [])
                                if t.get('function', {}).get('name') not in perception_names]
        payload['providerOptions'] = {'gateway': {'only': [provider], 'caching': 'auto'}}
        self.output.mkdir(parents=True, exist_ok=True)
        if self.model == 'deepseek/deepseek-v4.1-flash-1017':
            from .input_budget import deepseek_budget
            budget = deepseek_budget(payload)
            with (self.output / 'input_budget.jsonl').open('a') as log:
                log.write(json.dumps({'decision': self.turn_count + 1, **budget}) + '\n')
            if not budget['within_budget']:
                raise ModelError('Input budget exceeded; output reservation is preserved and no request was sent.')
        images = sum(part.get('type') in {'image_url', 'input_image', 'image'}
                     for item in payload['messages'] if isinstance(item.get('content'), list)
                     for part in item['content'])
        with (self.output / 'gateway_requests.jsonl').open('a') as log:
            log.write(json.dumps({'decision': self.turn_count + 1, 'history_policy': self.history_policy,
                'request_bytes': len(json.dumps(payload, allow_nan=False).encode('utf-8')),
                'perception_rounds_used': self._perception_rounds,
                'max_perception_rounds': self.max_perception_rounds,
                'image_count': images, 'message_count': len(payload['messages']),
                'reasoning_messages': sum(any(k in m for k in ('reasoning', 'reasoning_content', 'reasoning_details'))
                                          for m in payload['messages'])}) + '\n')
        try:
            data = await super()._post(endpoint, payload, deadline)
        except ModelError as error:
            response = getattr(error.__cause__, 'response', None)
            body = getattr(response, 'text', '')
            if self.api_key:
                body = body.replace(self.api_key, '[REDACTED]')
            with (self.output / 'gateway_errors.jsonl').open('a') as log:
                log.write(json.dumps({'decision': self.turn_count + 1, 'error': str(error),
                    'http_status': getattr(response, 'status_code', None), 'body': body[:16000]}) + '\n')
            raise
        self._last_gateway_payload = deepcopy(payload)
        self._last_gateway_response = deepcopy(data)
        if hasattr(self, '_decision_usage'):
            self._add_usage(self._decision_usage, data.get('usage') or {})
        message = (data.get('choices') or [{}])[0].get('message', {})
        if self.current_sensors_only:
            self._reasoning_by_index[len(payload['messages'])] = {
                key: deepcopy(message[key]) for key in ('reasoning', 'reasoning_content', 'reasoning_details')
                if key in message}
        with (self.output / 'gateway_assistant_messages.jsonl').open('a') as log:
            log.write(json.dumps({'decision': self.turn_count + 1,
                    'message_index': len(payload['messages']), 'message': message}) + '\n')
        metadata = message.get('provider_metadata', data.get('providerMetadata', {}))
        routing = metadata.get('gateway', {}).get('routing', {})
        actual = routing.get('finalProvider')
        self.output.mkdir(parents=True, exist_ok=True)
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
        # Validate the complete tool batch before even read-only tools run.
        allowed = {tool.get('function', {}).get('name') for tool in payload.get('tools', [])}
        for call in message.get('tool_calls') or []:
            function = call.get('function', {})
            if call.get('type') != 'function' or function.get('name') not in allowed:
                raise ModelError('API model attempted an unauthorized tool operation.')
            try:
                from core.action_compiler import unique_json_object
                arguments = json.loads(function['arguments'], object_pairs_hook=unique_json_object)
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be an object.')
            except (ValueError, TypeError, KeyError):
                raise ModelError('API model returned invalid tool arguments.')
        calls = message.get('tool_calls') or []
        if self.recent_notes and any(c.get('function', {}).get('name') in perception_names for c in calls):
            self._perception_rounds += 1
        historical = [c for c in calls if c.get('function', {}).get('name') in
                      ({HISTORY_IMAGE_TOOL, NOTE_TOOL} if self.recent_notes else {HISTORY_IMAGE_TOOL})]
        if historical:
            if len(historical) != len(calls):
                raise ModelError('API model returned invalid tool arguments.')
            messages = payload['messages']
            assistant = {k: deepcopy(message[k]) for k in
                ('role', 'content', 'tool_calls', 'reasoning', 'reasoning_content', 'reasoning_details') if k in message}
            assistant.setdefault('role', 'assistant')
            messages.append(assistant)
            images = []
            for call in historical:
                is_note = call['function']['name'] == NOTE_TOOL
                if is_note:
                    result, url = run_note(self, json.loads(call['function']['arguments'])), None
                else:
                    result, url = read_historical_image(self, json.loads(call['function']['arguments']))
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)})
                if url is not None:
                    images.append({'role': 'user', 'content': [
                        {'type': 'text', 'text': 'Requested historical RGB: ' + json.dumps(result)},
                        {'type': 'image_url', 'image_url': {'url': url}}]})
                with (self.output / ('note_tool_calls.jsonl' if is_note else 'historical_image_reads.jsonl')).open('a') as log:
                    log.write(json.dumps({'decision': self._current_decision_step, 'arguments':
                        json.loads(call['function']['arguments']), 'result': result}) + '\n')
            messages.extend(images)
            original_payload['messages'][:] = messages
            next_data = await self._post(endpoint, original_payload, deadline)
            usage = deepcopy(data.get('usage') or {})
            self._add_usage(usage, next_data.get('usage') or {})
            next_data = deepcopy(next_data)
            next_data['usage'] = usage
            return next_data
        return data

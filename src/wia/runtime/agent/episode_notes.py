"""Episode note tool; verify with python -m pytest tests/harness_integration/test_vercel_history.py."""
import json

NOTE_TOOL = 'episode_note'
from wia.resources import ROOT

NOTE_PROMPT = (ROOT / 'explore_to_achieve/prompts/tools/episode_notes.md').read_text()


def note_tool():
    return {'type': 'function', 'function': {'name': NOTE_TOOL, 'description': NOTE_PROMPT,
        'parameters': {'type': 'object', 'properties': {
            'operation': {'type': 'string', 'enum': ['append', 'read']},
            'text': {'type': 'string', 'maxLength': 128}},
            'required': ['operation'], 'additionalProperties': False}}}


def run_note(session, arguments):
    if arguments == {'operation': 'read'}:
        return {'ok': True, 'notes': list(session._notes), 'remaining_write_characters': 128-session._note_chars}
    text = arguments.get('text')
    if set(arguments) != {'operation', 'text'} or arguments.get('operation') != 'append' or not isinstance(text, str):
        return {'ok': False, 'error': 'Use read without text, or append with a string text.'}
    if len(text) > 128-session._note_chars:
        return {'ok': False, 'error': 'Per-decision cumulative 128-character write budget exceeded.',
                'remaining_write_characters': 128-session._note_chars}
    record = {'decision_step': session._current_decision_step, 'text': text}
    session.output.mkdir(parents=True, exist_ok=True)
    with (session.output / 'episode_notes.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    session._notes.append(record)
    session._note_chars += len(text)
    return {'ok': True, 'note': record, 'remaining_write_characters': 128-session._note_chars}

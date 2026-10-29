"""Check sequential public-reference submission using temporary fake responders.

Run: PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_public_reference.py
No simulator, model or GPU workload is launched.
"""
import hashlib
import json

import pytest

from wia.runtime.scene_capture_queue import write_json
from wia.runtime.commands.run_wia_public_reference import run_reference, session_profile


COMMANDS = [{'action': 'BASE_FWD', 'distance_m': .1}, {'action': 'RELEASE'}]


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def fixture_paths(tmp_path, commands=COMMANDS):
    session = tmp_path / 'session'
    (session / 'requests').mkdir(parents=True)
    (session / 'responses').mkdir()
    commands_file = tmp_path / 'authored.json'
    commands_file.write_text(json.dumps(commands))
    return session, commands_file, tmp_path / 'reference'


def reply_for(request, *, result=None, complete=True):
    return {'complete': complete, 'commands': [{'command': request['commands'][0],
             'result': {'success': True, 'completed': True} if result is None else result}],
            'state': {'snapshot': request['commands'][0]['action']}}


def test_success_waits_for_each_exact_reply_and_preserves_existing_files(tmp_path):
    session, commands_file, output = fixture_paths(tmp_path)
    sentinel = session / 'requests/existing.json'
    sentinel.write_text('{"existing": true}\n')
    (session / 'launch_manifest.json').write_text(json.dumps({'primitive_profile': 'fine'}))
    clock, received = Clock(), []

    def respond(seconds):
        clock.sleep(seconds)
        paths = sorted((session / 'requests').glob('wia_public_reference_*.json'))
        assert len(paths) == len(received) + 1
        path = paths[-1]
        request = json.loads(path.read_text())
        assert request == {'commands': [COMMANDS[len(received)]]}
        if received:
            assert (session / 'responses' / received[-1].name).is_file()
        received.append(path)
        write_json(session / 'responses' / path.name, reply_for(request))

    result = run_reference(session, commands_file, output, timeout_s=5, poll_s=1,
                           clock=clock, sleep=respond)
    assert result['status'] == 'complete'
    assert result['completed_commands'] == 2
    assert result['public_controls_only'] is True
    assert result['autonomous_agent_performance'] is False
    assert result['physical_success_claimed'] is False
    assert result['primitive_profile']['primitive_profile'] == 'fine'
    assert [path.name.rsplit('_', 1)[1] for path in received] == ['000001.json', '000002.json']
    assert result['input_sha256'] == hashlib.sha256(commands_file.read_bytes()).hexdigest()
    assert (output / result['input_copy']).read_bytes() == commands_file.read_bytes()
    assert sentinel.read_text() == '{"existing": true}\n'
    assert result['final_state'] == {'snapshot': 'RELEASE'}
    assert result['pending_request_may_execute'] is False
    for item in result['requests']:
        assert (output / item['response_copy']).read_bytes() == (session / 'responses' / received[item['order'] - 1].name).read_bytes()
        assert item['wait_s'] == 1.
    assert json.loads((output / 'result.json').read_text()) == result


@pytest.mark.parametrize('receipt,complete', [
    ({'success': False, 'completed': True}, True),
    ({'success': True, 'completed': False}, True),
    ({'success': True}, True),
    ({'success': 1, 'completed': True}, True),
    ({'success': True, 'completed': True}, False),
])
def test_unsuccessful_or_incomplete_feedback_stops_before_second_submit(tmp_path, receipt, complete):
    session, commands_file, output = fixture_paths(tmp_path)
    clock = Clock()

    def respond(seconds):
        clock.sleep(seconds)
        paths = list((session / 'requests').glob('*.json'))
        assert len(paths) == 1
        write_json(session / 'responses' / paths[0].name,
                   reply_for(json.loads(paths[0].read_text()), result=receipt, complete=complete))

    result = run_reference(session, commands_file, output, clock=clock, sleep=respond)
    assert result['status'] == 'failed'
    assert result['completed_commands'] == 0
    assert len(result['requests']) == len(list((session / 'requests').glob('*.json'))) == 1
    assert result['requests'][0]['response_copy']


def test_timeout_retains_request_and_reports_unknown_final_state_without_retry(tmp_path):
    session, commands_file, output = fixture_paths(tmp_path)
    clock = Clock()
    result = run_reference(session, commands_file, output, timeout_s=2, poll_s=1,
                           clock=clock, sleep=clock.sleep)
    assert result['status'] == 'timeout'
    assert result['stop_reason'] == 'response_timeout_request_not_cancelled'
    requests = list((session / 'requests').glob('*.json'))
    assert len(requests) == 1
    assert json.loads(requests[0].read_text()) == {'commands': [COMMANDS[0]]}
    assert result['pending_request_may_execute'] is True
    assert result['final_state'] is None
    assert result['elapsed_s'] == 2.
    assert json.loads((output / 'result.json').read_text()) == result


@pytest.mark.parametrize('commands', [
    {'commands': COMMANDS}, [COMMANDS[0], {'action': 'TELEPORT', 'distance_m': 1.}],
    [COMMANDS[0], {'action': 'RELEASE', 'temporal_event': 'changed'}],
    [COMMANDS[0], {'action': 'BASE_FWD', 'distance_m': False}],
])
def test_entire_invalid_list_is_rejected_before_any_request(tmp_path, commands):
    session, commands_file, output = fixture_paths(tmp_path, commands)
    result = run_reference(session, commands_file, output)
    assert result['status'] == 'invalid_commands'
    assert result['input_sha256']
    assert result['requests'] == []
    assert list((session / 'requests').iterdir()) == []


def test_keyboard_interrupt_records_pending_request_without_stopping_simulator(tmp_path):
    session, commands_file, output = fixture_paths(tmp_path)
    clock = Clock()

    def interrupt(seconds):
        raise KeyboardInterrupt

    result = run_reference(session, commands_file, output, clock=clock, sleep=interrupt)
    assert result['status'] == 'interrupted'
    assert result['pending_request_may_execute'] is True
    request = next((session / 'requests').glob('*.json'))
    assert json.loads(request.read_text()) == {'commands': [COMMANDS[0]]}
    assert json.loads((output / 'result.json').read_text()) == result


def test_existing_output_is_not_overwritten_and_launch_command_profile_is_logging_only(tmp_path):
    session, commands_file, output = fixture_paths(tmp_path)
    (session / 'launch.sh').write_text('python native.py --primitive-profile fixed --serve\n')
    assert session_profile(session)['primitive_profile'] == 'fixed'
    assert session_profile(session)['limits_changed'] is False
    output.mkdir()
    sentinel = output / 'result.json'
    sentinel.write_text('Existing result.\n')
    with pytest.raises(FileExistsError):
        run_reference(session, commands_file, output)
    assert sentinel.read_text() == 'Existing result.\n'

"""Send an authored command list sequentially to an existing native serve session.

Run: python -B src/wia/runtime/commands/run_wia_public_reference.py --session-dir SESSION \
  --commands-file COMMANDS.json --output-dir FRESH_OUTPUT [--timeout-s 180] [--poll-s 1]
This client never launches, cancels, replans or stops simulator workloads.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from core.action_compiler import ParameterizedAction, positive_number, unique_json_object
from core.action_units import DONE_ATOM, ROBOT_ACTIONS
from wia.runtime.scene_capture_queue import publish_json, write_json


def _json(raw):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constants are not accepted: ' + value)
    return json.loads(raw, object_pairs_hook=unique_json_object, parse_constant=invalid_constant)


def validate_commands(raw):
    """Validate the entire authored list without changing or compiling its commands."""
    commands = _json(raw)
    if not isinstance(commands, list):
        raise ValueError('Commands file must contain a JSON list of ParameterizedAction dictionaries only.')
    for command in commands:
        ParameterizedAction.from_payload(command, (*ROBOT_ACTIONS, DONE_ATOM))
    return commands


def session_profile(session_dir):
    """Record explicit launch metadata only; never infer or alter primitive limits."""
    observations, errors = [], []
    pattern = re.compile(r"--primitive-profile(?:\s+|=)[\"']?(fixed|fine)(?=[\"'\s]|$)")
    paths = sorted(path for path in Path(session_dir).glob('*launch*')
                   if path.is_file() and path.suffix in ('.json', '.txt', '.sh', '.command', ''))
    for path in paths:
        try:
            raw = path.read_bytes()
            text = raw.decode('utf-8')
            values = []
            if path.suffix == '.json':
                def visit(value):
                    if isinstance(value, dict):
                        if isinstance(value.get('primitive_profile'), str):
                            values.append(value['primitive_profile'])
                        for key, item in value.items():
                            if key in ('command', 'cmd', 'launch_command', 'argv', 'args'):
                                rendered = ' '.join(item) if isinstance(item, list) and all(isinstance(x, str) for x in item) else item
                                if isinstance(rendered, str):
                                    values.extend(pattern.findall(rendered))
                            if isinstance(item, (dict, list)):
                                visit(item)
                    elif isinstance(value, list):
                        for item in value:
                            visit(item)
                visit(_json(text))
            else:
                values.extend(pattern.findall(text))
            for value in sorted(set(values)):
                observations.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                                     'primitive_profile': value})
        except (OSError, ValueError, UnicodeError) as error:
            errors.append({'path': str(path), 'error': str(error)})
    profiles = {item['primitive_profile'] for item in observations}
    return {'status': 'recorded' if len(profiles) == 1 else 'conflicting' if profiles else 'unknown',
            'primitive_profile': next(iter(profiles)) if len(profiles) == 1 else None,
            'observations': observations, 'read_errors': errors, 'limits_changed': False}


def response_failure(response):
    """A completed request is not sufficient without a completed successful command."""
    if not isinstance(response, dict) or response.get('complete') is not True:
        return 'response_not_complete'
    commands = response.get('commands')
    if not isinstance(commands, list) or len(commands) != 1:
        return 'response_command_count_mismatch'
    receipt = commands[0].get('result') if isinstance(commands[0], dict) else None
    if not isinstance(receipt, dict) or receipt.get('success') is not True:
        return 'command_not_successful'
    if receipt.get('completed') is not True:
        return 'command_not_completed'
    return None


def run_reference(session_dir, commands_file, output_dir, *, timeout_s=180., poll_s=1.,
                  clock=time.monotonic, sleep=time.sleep):
    """Run unchanged public commands, stopping permanently at the first uncertain result."""
    timeout_s, poll_s = positive_number(timeout_s, 'timeout_s'), positive_number(poll_s, 'poll_s')
    session_dir, commands_file, output_dir = (Path(path).expanduser().resolve()
                                             for path in (session_dir, commands_file, output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    began = clock()
    record = {'schema': 'wia_public_reference_run_v1', 'run_id': uuid.uuid4().hex,
              'session_dir': str(session_dir), 'commands_file': str(commands_file),
              'input_sha256': None, 'input_copy': None, 'status': 'validating', 'stop_reason': None,
              'public_controls_only': True, 'autonomous_agent_performance': False,
              'physical_success_claimed': False, 'model_calls': 0,
              'timeout_s': timeout_s, 'poll_s': poll_s, 'started_monotonic_s': began,
              'completed_commands': 0, 'command_count': None, 'requests': [],
              'final_state': None, 'last_reply_state': None,
              'final_state_scope': 'no_matching_reply_observed', 'pending_request_may_execute': False,
              'primitive_profile': None}

    def save():
        now = clock()
        record.update(updated_monotonic_s=now, elapsed_s=now - began)
        write_json(output_dir / 'result.json', record)

    def pending():
        return any(Path(item['request_file']).is_file() and not item.get('response_copy')
                   for item in record['requests'])

    try:
        save()
        raw = commands_file.read_bytes()
        record['input_sha256'] = hashlib.sha256(raw).hexdigest()
        record['input_copy'] = 'commands.input.json'
        (output_dir / record['input_copy']).write_bytes(raw)
        commands = validate_commands(raw)
        record['command_count'] = len(commands)
        requests_dir, responses_dir = session_dir / 'requests', session_dir / 'responses'
        if not requests_dir.is_dir():
            raise ValueError('The existing native serve session must already have its requests directory.')
        record['primitive_profile'] = session_profile(session_dir)
        record['status'] = 'running'
        save()
        for index, command in enumerate(commands, 1):
            name = f'wia_public_reference_{record["run_id"]}_{index:06d}.json'
            request_file, response_file = requests_dir / name, responses_dir / name
            request = {'commands': [command]}
            item = {'order': index, 'command': command, 'request_file': str(request_file),
                    'response_file': str(response_file), 'request_copy': 'requests/' + name,
                    'submitted': False, 'status': 'submission_prepared',
                    'prepared_monotonic_s': clock()}
            record['requests'].append(item)
            write_json(output_dir / item['request_copy'], request)
            save()
            if not publish_json(request_file, request):
                raise ValueError('Unique request name already exists; no overwrite or retry was attempted.')
            submitted = clock()
            item.update(submitted=True, submitted_monotonic_s=submitted, status='awaiting_response')
            record.update(status='awaiting_response', pending_request_may_execute=True,
                          final_state=None, final_state_scope='unknown_while_request_pending')
            save()
            deadline = submitted + timeout_s
            while not response_file.is_file():
                remaining = deadline - clock()
                if remaining <= 0:
                    item.update(status='timeout', finished_monotonic_s=clock(), wait_s=clock() - submitted)
                    record.update(status='timeout', stop_reason='response_timeout_request_not_cancelled')
                    save()
                    return record
                save()
                sleep(min(poll_s, remaining))
            reply_raw = response_file.read_bytes()
            item['response_copy'] = 'responses/' + name
            copy_path = output_dir / item['response_copy']
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            copy_path.write_bytes(reply_raw)
            item.update(response_sha256=hashlib.sha256(reply_raw).hexdigest(),
                        response_monotonic_s=clock(), wait_s=clock() - submitted)
            response = _json(reply_raw)
            reason = response_failure(response)
            if isinstance(response, dict):
                record['last_reply_state'] = response.get('state')
                record['final_state'] = response.get('state')
            record.update(pending_request_may_execute=False,
                          final_state_scope='last_matching_reply_snapshot_not_current_simulator_measurement')
            item.update(status='failed' if reason else 'complete', stop_reason=reason,
                        response_complete=response.get('complete') if isinstance(response, dict) else None,
                        finished_monotonic_s=clock())
            if reason:
                record.update(status='failed', stop_reason=reason)
                save()
                return record
            record['completed_commands'] += 1
            record['status'] = 'running'
            save()
        record.update(status='complete', stop_reason='authored_command_list_exhausted')
    except KeyboardInterrupt:
        record.update(status='interrupted', stop_reason='keyboard_interrupt_simulator_not_cancelled')
    except (OSError, ValueError, TypeError) as error:
        record.update(status='invalid_commands' if record['status'] == 'validating' else 'failed',
                      stop_reason='input_validation_failed' if record['status'] == 'validating' else 'client_error',
                      error=f'{type(error).__name__}: {error}')
    finally:
        record['pending_request_may_execute'] = pending()
        if record['pending_request_may_execute']:
            record.update(final_state=None, final_state_scope='unknown_pending_request_not_cancelled')
        save()
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session-dir', type=Path, required=True)
    parser.add_argument('--commands-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout-s', type=float, default=180.)
    parser.add_argument('--poll-s', type=float, default=1.)
    args = parser.parse_args(argv)
    result = run_reference(args.session_dir, args.commands_file, args.output_dir,
                           timeout_s=args.timeout_s, poll_s=args.poll_s)
    print(f'Public reference status: {result["status"]}; completed commands: {result["completed_commands"]}; '
          f'result: {args.output_dir.resolve() / "result.json"}')
    return 0 if result['status'] == 'complete' else 130 if result['status'] == 'interrupted' else 1


if __name__ == '__main__':
    raise SystemExit(main())

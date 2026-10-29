"""Keep a debug world alive and resume only its original action boundary.

Run ``wia debug-control --trial TRIAL --action resume`` or ``--action close``
from another terminal while ``wia debug`` owns the retained simulator.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import requests


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def submit_control(trial, action):
    trial = Path(trial).expanduser().resolve()
    state = json.loads((trial / 'debug_state.json').read_text())
    if action not in ('resume', 'close'):
        raise ValueError('Debug control accepts resume or close.')
    if state.get('status') != 'waiting':
        raise ValueError('The debug coordinator is not waiting for a command.')
    if action == 'resume' and not state.get('resumable'):
        raise ValueError('This attempt cannot resume; inspect debug_state.json.')
    try:
        os.kill(state['coordinator_pid'], 0)
    except ProcessLookupError as error:
        raise ValueError('The debug coordinator is detached; preserve the live world and use the saved resume command.') from error
    request = {'id': uuid.uuid4().hex, 'action': action}
    path = trial / 'debug_control.json'
    # Exclusive creation prevents one user command from overwriting another.
    with path.open('x') as stream:
        json.dump(request, stream)
    return {'status': 'queued', 'trial': str(trial), **request}


def resume_command(agent, previous, destination):
    command = list(agent)
    command[command.index('--output-dir') + 1] = str(destination)
    return command + ['--resume-from', str(previous)]


def stop_recording(url, output):
    response = requests.post(url + '/api/stop', json={}, timeout=600)
    response.raise_for_status()
    write_json(output / 'recording_result.json', response.json())
    while True:
        response = requests.get(url + '/api/state', timeout=10)
        response.raise_for_status()
        state = response.json()
        if state.get('recording_status') not in ('recording', 'finalizing'):
            write_json(output / 'recording_verification.json', state.get('last_recording'))
            if state.get('recording_error'):
                write_json(output / 'recording_error.json', {'error': state['recording_error']})
            return
        time.sleep(2)


def wait_for_debug_control(output, *, url, gateway_url, agent, cwd, record, save_status):
    """Block the owner until explicit close or native exit, never reset/replay.

    KeyboardInterrupt propagates with the scene and gateway still owned by their
    existing processes. A transient HTTP failure cannot authorize scene cleanup.
    """
    output = Path(output)
    previous = output / 'episode'
    attempt = 0
    while True:
        summary_path = previous / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        resumable = (summary.get('termination_reason') in ('model_error', 'cancelled')
                     and summary.get('agent_decisions', 0) < summary.get('max_steps', 0)
                     and '--backend' in agent and agent[agent.index('--backend') + 1] in ('codex', 'codex_cli'))
        destination = output / f'resume_{attempt + 1:04d}'
        command = resume_command(agent, previous, destination)
        state = {'status': 'waiting', 'coordinator_pid': os.getpid(), 'resumable': resumable,
                 'episode_dir': str(previous), 'termination_reason': summary.get('termination_reason'),
                 'agent_decisions': summary.get('agent_decisions'), 'max_steps': summary.get('max_steps'),
                 'backend_url': url, 'gateway_url': gateway_url,
                 'resume_command': command if resumable else None,
                 'control_command': 'wia debug-control --trial ' + str(output) + ' --action resume|close'}
        write_json(output / 'debug_state.json', state)
        save_status('debug_waiting', simulator_retained=True, active_episode_dir=previous.name,
                    termination_reason=summary.get('termination_reason'), agent_decisions=summary.get('agent_decisions'))
        while not (output / 'debug_control.json').exists():
            if (output / 'native_exit_code.txt').exists():
                write_json(output / 'debug_state.json', dict(state, status='native_exited'))
                return False
            time.sleep(1)
        path = output / 'debug_control.json'
        try:
            request = json.loads(path.read_text())
        except ValueError:
            time.sleep(.1)
            continue
        path.unlink()
        action = request.get('action')
        if action == 'close':
            write_json(output / 'debug_state.json', dict(state, status='closed'))
            return False
        if action != 'resume' or not resumable:
            write_json(output / 'debug_control_error.json', {'request': request, 'error': 'Resume is unavailable.'})
            continue
        attempt += 1
        write_json(output / 'debug_state.json', dict(state, status='resuming', request_id=request.get('id')))
        save_status('agent_resuming', active_episode_dir=previous.name, resume_output_dir=destination.name)
        started_recording = False
        try:
            if record:
                response = requests.post(url + '/api/start', json={}, timeout=10)
                response.raise_for_status()
                started_recording = response.json().get('ok') is True
                if not started_recording:
                    raise RuntimeError('Debug resume recording did not start.')
            with (output / f'resume_{attempt:04d}.log').open('w') as stream:
                completed = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
            if (destination / 'summary.json').exists():
                previous = destination
                save_status('debug_attempt_completed', active_episode_dir=previous.name,
                            agent_exit_code=completed.returncode)
            else:
                write_json(output / 'debug_control_error.json', {'request': request,
                    'error': 'Resume produced no summary; original action-boundary checks remain required.',
                    'returncode': completed.returncode})
        except (OSError, requests.RequestException, RuntimeError) as error:
            write_json(output / 'debug_control_error.json', {'request': request, 'error': str(error)})
        finally:
            if started_recording:
                destination.mkdir(exist_ok=True)
                try:
                    stop_recording(url, destination)
                except (requests.RequestException, ValueError) as error:
                    write_json(destination / 'recording_error.json', {'error': str(error)})

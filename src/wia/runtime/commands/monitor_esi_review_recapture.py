#!/usr/bin/env python3
"""Monitor confirmed native crashes and restart fully exited capture supervisors.

Run: python src/wia/runtime/commands/monitor_esi_review_recapture.py --queue-root QUEUE
    --campaign-output CAMPAIGN --worker-script-dir SCRIPTS --worker-log-dir LOGS
Scripts must be named start_worker0.sh, start_worker1.sh, etc., matching worker
records. Existing codex-gpu tmux shells are retained. Healthy processes have no
age deadline; a failed launch with no new worker identity needs inspection.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_capture_crash import NativeCrashWatcher
from wia.runtime.scene_capture_queue import CaptureQueue, read_json, write_json
from wia.runtime.commands.run_esi_capture_worker import group_members, stop_group


def process_state(pid, ticks):
    """Return alive/exited/unknown for the original same-user process identity."""
    if not pid or ticks is None:
        return 'unknown'
    try:
        path = Path('/proc') / str(int(pid))
        fields = (path / 'stat').read_text().split(') ', 1)[1].split()
        if fields[19] != str(ticks) or fields[0] == 'Z':
            return 'exited'
        return 'alive' if path.stat().st_uid == os.getuid() else 'unknown'
    except (FileNotFoundError, ProcessLookupError):
        return 'exited'
    except (OSError, ValueError, IndexError):
        return 'unknown'


def owner_identity(owner):
    return [owner.get('supervisor_pid'), str(owner.get('supervisor_start_ticks'))]


def gpu_contexts(gpu_uuid):
    """Verify zero contexts using NVIDIA's full compute-and-graphics XML table."""
    try:
        result = subprocess.run(['nvidia-smi', '-i', gpu_uuid, '-q', '-x'],
                                check=True, capture_output=True, text=True, timeout=10)
        root = ET.fromstring(result.stdout)
        matches = [gpu for gpu in root.findall('gpu') if gpu.findtext('uuid') == gpu_uuid]
        if len(matches) != 1 or matches[0].find('processes') is None:
            raise ValueError('GPU identity or process table is unavailable.')
        processes = matches[0].find('processes')
        if (processes.text or '').strip() not in {'', 'None'}:
            raise ValueError('GPU process table is not readable: ' + processes.text.strip())
        contexts = [{'pid': item.findtext('pid'), 'type': item.findtext('type'),
                     'name': item.findtext('process_name')} for item in processes.findall('process_info')]
        return {'verified': True, 'uuid': gpu_uuid, 'contexts': contexts}
    except (OSError, subprocess.SubprocessError, ET.ParseError, ValueError) as error:
        return {'verified': False, 'uuid': gpu_uuid, 'error': str(error)}


def native_exit_evidence(queue, owner):
    """Check declared native owners and claims without stopping any process."""
    identities = []
    if owner.get('appdata_path'):
        fence = read_json(Path(owner['appdata_path']) / 'native_tmp_owner.json', {})
        if fence:
            identities.append({'pid': fence.get('pid'), 'ticks': fence.get('start_ticks'),
                               'pgid': fence.get('pgid'), 'source': 'native_tmp_owner'})
    output = owner.get('native_output')
    if not output and owner.get('job_id'):
        output = (queue.status(owner['job_id']) or {}).get('native_output')
    if output:
        native = read_json(Path(output) / 'native_process.json', {})
        if native:
            identities.append({'pid': native.get('pid'), 'ticks': native.get('start_ticks'),
                               'pgid': native.get('pgid'), 'source': 'native_process'})
    for path in (queue.root / 'claimed').glob('*.json'):
        claim = read_json(path, {})
        if claim.get('worker', {}).get('id') != owner['id']:
            continue
        if claim.get('child_pid'):
            identities.append({'pid': claim['child_pid'], 'ticks': claim.get('child_start_ticks'),
                               'pgid': claim.get('child_pgid'), 'source': 'claim'})
    if owner.get('native_pid') and not any(i['pid'] == owner['native_pid'] for i in identities):
        return {'exited': False, 'reason': 'Native PID has no matching start-ticks evidence.'}
    for item in identities:
        item['state'] = process_state(item['pid'], item['ticks'])
        if item['state'] != 'exited':
            return {'exited': False, 'reason': 'Native identity is alive or unverified.', 'native': item}
        if item.get('pgid') and group_members(int(item['pgid'])):
            return {'exited': False, 'reason': 'Native process group still has live members.', 'native': item}
    return {'exited': True, 'identities': identities}


def launch_saved_worker(script, log, worker_id):
    """Launch the unchanged saved script and retain its shell after it exits."""
    command = 'bash ' + shlex.quote(str(script)) + ' >> ' + shlex.quote(str(log)) + ' 2>&1; exec bash'
    result = subprocess.run(['tmux', 'new-window', '-d', '-t', 'codex-gpu',
                             '-n', worker_id + '-recovery-' + str(time.time_ns()),
                             '-P', '-F', '#{pane_id}', command],
                            check=True, capture_output=True, text=True, timeout=10)
    return {'pane_id': result.stdout.strip(), 'session': 'codex-gpu', 'script': str(script), 'log': str(log)}


class RecoveryMonitor:
    def __init__(self, queue_root, campaign_output, worker_script_dir, worker_log_dir):
        self.queue = CaptureQueue(queue_root)
        self.output = Path(campaign_output).resolve()
        self.scripts = Path(worker_script_dir).resolve()
        self.logs = Path(worker_log_dir).resolve()
        self.logs.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)
        self.journal = self.output / 'recovery_monitor.json'
        self.pending = read_json(self.journal, {}).get('pending_restarts', {})
        self.poll_owner = None
        self.watcher = NativeCrashWatcher(self.queue.root, self.stop_confirmed_native)

    def record(self, value):
        value = {'timestamp': time.time(), **value}
        with (self.output / 'recovery_monitor.jsonl').open('a') as handle:
            handle.write(json.dumps(value, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(value), flush=True)

    def save_pending(self):
        write_json(self.journal, {'pending_restarts': self.pending})

    def blocked(self):
        if read_json(self.output / 'progress.json', {}).get('stage') == 'finished':
            return 'finished'
        if read_json(self.queue.root / 'capture_pause.json', {}).get('state') == 'PAUSED_BY_USER':
            return 'paused'
        return None

    def stop_confirmed_native(self, pid):
        owner = self.poll_owner
        if not owner or self.blocked():
            return
        current = read_json(self.queue.root / 'workers' / (owner['id'] + '.json'), {})
        if (owner_identity(current) == owner_identity(owner)
                and current.get('native_pid') == pid
                and process_state(owner['supervisor_pid'], owner.get('supervisor_start_ticks')) == 'alive'):
            stop_group(pid)

    def restart(self, worker_id, owner, script):
        identity = owner_identity(owner)
        if self.pending.get(worker_id, {}).get('previous_owner') == identity:
            return  # A launched script has not yet published a new supervisor.
        evidence = native_exit_evidence(self.queue, owner)
        if not evidence['exited']:
            self.record({'action': 'restart_skipped', 'worker_id': worker_id, **evidence})
            return
        gpu = gpu_contexts(owner.get('uuid', ''))
        if not gpu['verified'] or gpu['contexts']:
            self.record({'action': 'restart_skipped', 'worker_id': worker_id, 'gpu': gpu})
            return
        current = read_json(self.queue.root / 'workers' / (worker_id + '.json'), {})
        if (self.blocked() or owner_identity(current) != identity
                or process_state(owner['supervisor_pid'], owner.get('supervisor_start_ticks')) != 'exited'):
            return
        evidence = native_exit_evidence(self.queue, current)
        if not evidence['exited']:
            self.record({'action': 'restart_skipped', 'worker_id': worker_id, **evidence})
            return
        # Journal before launching: a monitor crash cannot duplicate a launch.
        self.pending[worker_id] = {'previous_owner': identity, 'state': 'launch_pending'}
        self.save_pending()
        gpu = gpu_contexts(owner.get('uuid', ''))
        latest = read_json(self.queue.root / 'workers' / (worker_id + '.json'), {})
        if (not gpu['verified'] or gpu['contexts'] or self.blocked()
                or owner_identity(latest) != identity
                or process_state(owner['supervisor_pid'], owner.get('supervisor_start_ticks')) != 'exited'):
            self.pending.pop(worker_id)
            self.save_pending()
            self.record({'action': 'restart_skipped_final_check', 'worker_id': worker_id, 'gpu': gpu})
            return
        try:
            launch = launch_saved_worker(script, self.logs / (worker_id + '.log'), worker_id)
        except (OSError, subprocess.SubprocessError) as error:
            # Preserve launch intent: timeout does not prove tmux failed to start.
            self.pending[worker_id]['state'] = 'launch_result_unknown'
            self.save_pending()
            self.record({'action': 'restart_launch_error', 'worker_id': worker_id, 'error': str(error)})
            return
        self.pending[worker_id].update(state='launched', **launch)
        self.save_pending()
        self.record({'action': 'worker_restarted', 'worker_id': worker_id,
                     'previous_owner': identity, 'native_exit': evidence, 'gpu': gpu, **launch})

    def tick(self):
        blocked = self.blocked()
        if blocked:
            return blocked
        for script in sorted(self.scripts.glob('start_worker*.sh')):
            match = re.fullmatch(r'start_(worker\d+)\.sh', script.name)
            if not match:
                continue
            worker_id = match.group(1)
            owner = read_json(self.queue.root / 'workers' / (worker_id + '.json'), {})
            if not owner or owner.get('id') != worker_id:
                continue  # Main owns the initial launches; no record means no restart.
            try:
                identity = owner_identity(owner)
                if worker_id in self.pending and self.pending[worker_id]['previous_owner'] != identity:
                    self.pending.pop(worker_id)
                    self.save_pending()
                state = process_state(owner.get('supervisor_pid'), owner.get('supervisor_start_ticks'))
                if state == 'alive':
                    self.poll_owner = owner
                    evidence = self.watcher.poll(owner)
                    if evidence:
                        self.record({'action': 'native_crash_recovery', 'worker_id': worker_id, 'evidence': evidence})
                elif state == 'exited':
                    self.restart(worker_id, owner, script)
                else:
                    self.record({'action': 'restart_skipped', 'worker_id': worker_id,
                                 'reason': 'Supervisor exit could not be verified.'})
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.record({'action': 'monitor_error', 'worker_id': worker_id, 'error': str(error)})
            finally:
                self.poll_owner = None
        return 'monitoring'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('queue-root', 'campaign-output', 'worker-script-dir', 'worker-log-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    monitor = RecoveryMonitor(args.queue_root, args.campaign_output, args.worker_script_dir, args.worker_log_dir)
    with (monitor.output / 'recovery_monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = monitor.tick()
            if state == 'finished':
                monitor.record({'action': 'monitor_finished', 'reason': 'Campaign progress is finished.'})
                return 0
            time.sleep(15)


if __name__ == '__main__':
    raise SystemExit(main())

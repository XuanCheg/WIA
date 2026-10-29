"""Record confirmed native crashes and stop only their current owned process.

Use from the persistent bulk watchdog without restarting healthy GPU workers.
Run CPU checks with: python -m pytest -q tests/test_scene_capture_crash.py
The resident supervisor retains its claim until native exit, then publishes the
failure and advances normally. No image validation or image-directory scan runs.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import time

from wia.runtime.scene_capture_queue import CaptureQueue, read_json, write_json


def fatal_native_error(text):
    """Recognize explicit device faults; ordinary warnings are insufficient."""
    for line in text.splitlines():
        plain = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
        lower = plain.lower()
        if 'gpu crash is detected' in lower or 'vk_error_device_lost' in lower:
            return {'kind': 'native_gpu_crash', 'message': plain}
        if ('illegal memory access' in lower or 'illegal address' in lower) and any(
                word in lower for word in ('cuda', 'physx', 'gpu')):
            return {'kind': 'native_gpu_crash', 'message': plain}
        if 'physx error:' in lower and re.search(r'\b700\b', lower):
            return {'kind': 'native_gpu_crash', 'message': plain}
        if '[fatal]' in lower:
            return {'kind': 'native_render_crash', 'message': plain}
    return None


def process_matches(pid, ticks):
    """A reused PID, zombie or different user is not the owned native process."""
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().split(') ', 1)[1].split()
        return (path.stat().st_uid == os.getuid() and fields[0] != 'Z'
                and fields[19] == str(ticks) and os.getpgid(pid) == pid)
    except (FileNotFoundError, ProcessLookupError):
        return False


def preserve_crash_artifacts(text, output, appdata_path):
    """Copy only explicit dump/debug paths reported by this worker's native log."""
    result = []
    if not appdata_path:
        return result
    allowed = Path(appdata_path).resolve()
    for name in sorted(set(re.findall(r'/[^\s\"\']+\.(?:nv-gpudmp|nvdbg|dmp)\b', text))):
        source = Path(name).resolve()
        if not source.is_relative_to(allowed):
            continue
        destination = Path(output) / 'native_crash_artifacts' / source.name
        row = {'source': str(source)}
        try:
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            row.update(path=str(destination), bytes=destination.stat().st_size)
        except OSError as error:
            row['error'] = str(error)
        result.append(row)
    return result


class NativeCrashWatcher:
    def __init__(self, queue_root, stop_owned_group):
        self.queue = CaptureQueue(queue_root)
        self.stop_owned_group = stop_owned_group
        self.offsets = {}
        self.log_tails = {}

    def _active(self, owner, output, native):
        job, pid = owner.get('job_id'), native.get('pid')
        worker = read_json(self.queue.root / 'workers' / (owner['id'] + '.json'), {})
        state = self.queue.status(job) or {}
        claim = read_json(self.queue.root / 'claimed' / (job + '.json'), {})
        gpu = read_json(output / 'gpu.json', {})
        return (worker.get('state') == 'RUNNING' and worker.get('job_id') == job
            and worker.get('native_pid') == pid and state.get('state') == 'RUNNING'
            and claim.get('worker', {}).get('id') == owner['id']
            and claim.get('worker', {}).get('uuid') == owner.get('uuid')
            and claim.get('child_pid') == pid
            and str(claim.get('child_start_ticks')) == str(native.get('start_ticks'))
            and gpu.get('uuid') == owner.get('uuid')
            and native.get('capture_job_id') == job and native.get('pgid') == pid
            and native.get('start_ticks') is not None and process_matches(pid, native['start_ticks']))

    def poll(self, owner):
        """Return recovery evidence only for a confirmed crash in the active job."""
        job = owner.get('job_id')
        if owner.get('state') != 'RUNNING' or not job or not owner.get('native_pid'):
            return None
        state = self.queue.status(job) or {}
        if state.get('state') != 'RUNNING' or not state.get('native_output'):
            return None
        output = Path(state['native_output'])
        expected = self.queue.job_dir(job) / 'attempts' / f"{state['attempts']:02d}"
        if output.resolve() != expected.resolve():
            return None
        native = read_json(output / 'native_process.json', {})
        if not native or not self._active(owner, output, native):
            return None
        evidence_file = output / 'automatic_crash_recovery.json'
        previous = read_json(evidence_file, {})
        if previous.get('action') == 'faulty_native_stopped':
            return None
        log = output / 'native.log'
        if previous.get('action') in ('crash_recorded', 'native_exit_pending'):
            error = {'kind': previous['failure_stage'], 'message': previous['error']}
        else:
            try:
                size = log.stat().st_size
                offset = self.offsets.get(str(log), 0)
                if size < offset:
                    offset = 0
                with log.open('rb') as stream:
                    stream.seek(offset)
                    chunk = stream.read()
                self.offsets[str(log)] = offset + len(chunk)
            except FileNotFoundError:
                return None
            text = self.log_tails.get(str(log), '') + chunk.decode(errors='replace')
            self.log_tails[str(log)] = text.rsplit('\n', 1)[-1] if not text.endswith('\n') else ''
            error = fatal_native_error(text)
        if error is None:
            return None
        partial = dict(state.get('partial_images', {}))
        for name in ('partial_capture.json', 'capture_failure.json', 'preview_complete.json'):
            metadata = read_json(output / name, {})
            partial.update(metadata.get('images', {}))
            partial.update(metadata.get('partial_images', {}))
        evidence = {'recorded_at': time.time(), 'job_id': job, 'worker_id': owner['id'],
            'gpu_uuid': owner['uuid'], 'native_process': native, 'native_log': str(log),
            'failure_stage': error['kind'], 'error': error['message'], 'partial_images': partial,
            'capture_state_before_recovery': state, 'action': 'crash_recorded',
            'policy': 'Retain all evidence, stop this faulty native group, and let its supervisor advance.'}
        crash_log = log.read_text(errors='replace')
        evidence['crash_artifacts'] = preserve_crash_artifacts(crash_log, output, owner.get('appdata_path'))
        write_json(evidence_file, evidence)
        # Recheck after persisting evidence. A completed task or changed owner
        # must never cause termination of the next healthy task on the GPU.
        if not self._active(owner, output, native):
            evidence['action'] = 'owner_changed_before_stop'
            write_json(evidence_file, evidence)
            return evidence
        failure_file = output / 'failure.json'
        original_failure = read_json(failure_file, {})
        if original_failure and not (output / 'native_failure_before_watchdog.json').exists():
            shutil.copyfile(failure_file, output / 'native_failure_before_watchdog.json')
        write_json(failure_file, {**original_failure, 'error': original_failure.get('error') or error['message'],
            'failure_stage': error['kind'], 'retryable': False,
            'watchdog_evidence': str(evidence_file), 'partial_images': partial})
        manifest = read_json(self.queue.job_dir(job) / 'input/manifest.json', {})
        card_path = self.queue.root / 'category_records' / (manifest.get('candidate_id', '') + '.json')
        card = read_json(card_path)
        if card is not None:
            card['diagnostic_logs'] = sorted(set(card.get('diagnostic_logs', [])) | {str(log), str(evidence_file)})
            card['warning_codes'] = sorted(set(card.get('warning_codes', [])) | {'native_crash_recorded_automatic_advance'})
            write_json(card_path, card)
        if not self._active(owner, output, native):
            evidence['action'] = 'owner_changed_before_stop'
            write_json(evidence_file, evidence)
            return evidence
        self.stop_owned_group(native['pid'])
        # A final native diagnostic write may finish during process disposal.
        evidence['crash_artifacts'] = preserve_crash_artifacts(crash_log, output, owner.get('appdata_path'))
        if card is not None:
            card['diagnostic_logs'] = sorted(set(card['diagnostic_logs']) | {
                artifact['path'] for artifact in evidence['crash_artifacts'] if artifact.get('path')})
            write_json(card_path, card)
        evidence['action'] = ('native_exit_pending' if process_matches(native['pid'], native['start_ticks'])
                              else 'faulty_native_stopped')
        evidence['action_at'] = time.time()
        write_json(evidence_file, evidence)
        return evidence

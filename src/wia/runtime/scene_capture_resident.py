"""Own a persistent native screenshot service without importing CUDA here.

Run via src/wia/runtime/commands/run_esi_capture_worker.py --persistent-simulator. Immutable code
revisions, task families and runtime settings fence native reuse. Failed worlds
are disposed and every attempt remains in the existing Review queue.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import os
from pathlib import Path
from wia.runtime.resources import frozen_runtime_environment
import shutil
import subprocess
import sys
import time
import uuid

import yaml

from wia.runtime.scene_capture_queue import complete_images, digest, read_json, sha256, verify_snapshot, write_json

ROOT = WIA_RUNTIME_ROOT


class NativeShutdownPending(RuntimeError):
    """Keep the supervisor and owned process fence until native exit is confirmed."""


class NativeRendererFailure(RuntimeError):
    """An explicit native renderer crash requires disposal before bounded retry."""


class RendererLogMonitor:
    """Read only appended log bytes; an old job's messages cannot fail a new job."""
    markers = ('GPU crash is detected', 'VK_ERROR_DEVICE_LOST',
               '[Fatal] [rtx.scenedb.plugin]')

    def __init__(self, path):
        self.path = Path(path)
        self.offset = self.path.stat().st_size if self.path.is_file() else 0
        self.tail = ''

    def failure(self):
        if not self.path.is_file():
            return None
        with self.path.open('rb') as stream:
            if self.path.stat().st_size < self.offset:
                self.offset, self.tail = 0, ''
            stream.seek(self.offset)
            data = stream.read(1024 * 1024)
            self.offset = stream.tell()
        text = self.tail + data.decode('utf-8', errors='replace')
        self.tail = text[-1024:]
        return next((line[-1000:] for line in text.splitlines()
                     if any(marker in line for marker in self.markers)), None)


def runtime_identity(manifest):
    """Keep Python imports, original macro side effects and runtime settings pinned."""
    definition = yaml.safe_load(Path(manifest['snapshot_scene_file']).read_text())
    return digest({'code_commit': manifest['code_commit'], 'esi_commit': manifest['esi_commit'],
        'code_root': str(Path(manifest['code_root']).resolve()),
        'category': manifest.get('big_task', manifest.get('category')),
        'subcategory': manifest.get('small_task', manifest.get('subcategory')),
        'implementation_hashes': manifest['implementation_hashes'], 'runtime': definition.get('runtime', {})})


def recover_rendered_attempt(queue, job_id, reason):
    """Recover verified diagnostic images after the stopped owner's teardown failed."""
    state, manifest = queue.status(job_id), queue.manifest(job_id)
    if state['state'] != 'FAILED' or not manifest.get('diagnostic_pending'):
        raise ValueError('Recovery requires a failed diagnostic capture, never a qualification bypass.')
    if (queue.root / 'claimed' / (job_id + '.json')).exists():
        raise ValueError('Capture recovery requires a released owner.')
    output = Path(state['native_output'])
    native = read_json(output / 'native_process.json', {})
    if not native.get('pid') or not native.get('start_ticks') or native.get('capture_job_id') != job_id:
        raise ValueError('Recovery requires native process evidence for this exact capture job.')
    process = Path(f"/proc/{native.get('pid')}/stat")
    if process.is_file():
        fields = process.read_text().split(') ', 1)[1].split()
        if fields[0] != 'Z' and fields[19] == str(native.get('start_ticks')):
            raise ValueError('Capture recovery requires confirmed native process exit.')
    verify_snapshot(manifest)
    receipt = read_json(output / 'preview_complete.json', {})
    if not complete_images(manifest, receipt):
        raise ValueError('Recovery requires the complete, readable immutable screenshot set.')
    archive = queue.job_dir(job_id) / 'recovery' / ('resident_disposal_' + uuid.uuid4().hex + '.json')
    write_json(archive, {'previous_status': state, 'reason': reason, 'timestamp': time.time()})
    log = output / 'resident_disposal_recovery.log'
    log.write_text(reason + '\nThe native images are verified; physical qualification remains pending.\n'
                   + 'Previous failure state: ' + str(archive) + '\n')
    recovered = {**receipt, 'native_process_running': False, 'native_cleanup_warning': reason,
                 'provenance': receipt.get('provenance', '') + ' ' + reason}
    path = output / 'recovered_preview_complete.json'
    write_json(path, recovered)
    write_json(queue.root / 'cache' / (manifest['capture_signature'] + '.json'),
               {'job_id': job_id, 'receipt': str(path)})
    updated = {**state, 'state': 'REVIEW_PENDING', 'receipt': str(path),
               'native_cleanup_warning': reason,
               'prior_capture_failures': state.get('prior_capture_failures', []) + [str(archive)]}
    for field in ('error', 'failure_stage', 'complete', 'partial_images'):
        updated.pop(field, None)
    queue._save(job_id, updated)
    card_path = queue.root / 'category_records' / (manifest['candidate_id'] + '.json')
    card = read_json(card_path)
    if card is not None:
        card['warning_codes'] = sorted(set(card.get('warning_codes', []) + ['native_cleanup_recovered']))
        card['diagnostic_logs'] = sorted(set(card.get('diagnostic_logs', []) + [str(log)]))
        write_json(card_path, card)
    return {'job_id': job_id, 'receipt': str(path), 'failure_archive': str(archive)}


class ResidentCapture:
    def __init__(self, queue, gpu, owner, stopping, helpers, *, max_jobs=0, preload_workers=0,
                 prepare_assets=False):
        self.queue, self.gpu, self.owner = queue, gpu, owner
        self.stopping, self.helpers, self.max_jobs = stopping, helpers, max_jobs
        self.process = self.control = self.native_owner = self.log = None
        self.key = self.family = None
        self.jobs = 0
        self.recycle_before_next = None
        self.shutdown_pending = False
        self.pending_failure = None
        self.cards, self.known_cards = {}, set()
        self.preload_workers = preload_workers
        self.prepare_assets = prepare_assets
        self.asset_cache_root = None
        self.asset_runtime = None
        self.allowed_job_ids = None
        self.preloader = None
        self.preload_futures = {}
        if preload_workers:
            from wia.runtime.scene_capture_preload import CapturePreloader
            if prepare_assets:
                self.asset_cache_root = Path(gpu['appdata_path']) / 'cpu_prepared_assets' / uuid.uuid4().hex
                self.asset_cache_root.mkdir(parents=True, mode=0o700)
            self.preloader = CapturePreloader(preload_workers, os.environ['OMNIGIBSON_DATA_PATH'],
                                             asset_cache_root=self.asset_cache_root)

    def preload_inputs(self, manifest):
        """Read ahead two immutable inputs without claiming the next native job."""
        if self.preloader is None:
            return None
        selected = [manifest['job_id']]
        family = (manifest.get('code_commit'), manifest.get('big_task'), manifest.get('small_task'))
        paths = ((self.queue.root / 'queued').glob('*.json') if self.allowed_job_ids is None else
                 [self.queue.root / 'queued' / (job + '.json') for job in self.allowed_job_ids
                  if (self.queue.root / 'queued' / (job + '.json')).is_file()])
        tokens = sorted(paths, key=lambda path: (
            -read_json(self.queue.root / 'priorities' / path.name, {}).get('priority', 0), path.name))
        for token in tokens[:128]:
            other = self.queue.manifest(token.stem)
            if (other.get('code_commit'), other.get('big_task'), other.get('small_task')) == family:
                selected.append(token.stem)
                if len(selected) == 2:
                    break
        self.prune_prepared_assets(selected)
        for job in selected:
            if job not in self.preload_futures:
                self.preload_futures[job] = self.preloader.submit(self.queue.job_dir(job) / 'input/manifest.json')
        return self.preload_futures[manifest['job_id']]

    def prune_prepared_assets(self, selected):
        """Retain only the two read-ahead tasks after all prior CPU work ends."""
        if self.asset_cache_root is None or any(not future.done() for future in self.preload_futures.values()):
            return
        keep, expired = set(), set()
        for job, future in self.preload_futures.items():
            if future.cancelled() or future.exception() is not None:
                continue
            paths = {Path(row['cache_manifest']).resolve().parent for row in
                     future.result().get('prepared_assets', {}).get('assets', [])}
            (keep if job in selected else expired).update(paths)
        root = self.asset_cache_root.resolve()
        for path in expired - keep:
            if path.parent == root and path.is_dir() and not path.is_symlink() and path.stat().st_uid == os.getuid():
                shutil.rmtree(path)
        self.preload_futures = {job: future for job, future in self.preload_futures.items() if job in selected}

    def close_preloader(self):
        if self.preloader is not None:
            self.preloader.close()
            self.preloader = None
        if self.asset_cache_root is not None and (self.process is None or self.process.poll() is not None):
            shutil.rmtree(self.asset_cache_root)
            self.asset_cache_root = None

    def next_job_id(self):
        """Prefer the active revision/family within the highest queue priority."""
        if self.process is None or self.process.poll() is not None or not self.family:
            return None
        for path in (self.queue.root / 'category_records').glob('*.json'):
            if path.name in self.known_cards:
                continue
            row = read_json(path, {})
            if row.get('job_id'):
                self.cards[row['job_id']] = (row.get('code_commit'), row.get('big_task'), row.get('small_task'))
                self.known_cards.add(path.name)
        candidates = []
        for token in (self.queue.root / 'queued').glob('*.json'):
            priority = read_json(self.queue.root / 'priorities' / token.name, {}).get('priority', 0)
            candidates.append((priority, token.stem))
        highest = max((priority for priority, _ in candidates), default=None)
        return next((job for priority, job in sorted(candidates, key=lambda item: item[1])
                     if priority == highest and self.cards.get(job) == self.family), None)

    def worker_state(self):
        if self.shutdown_pending:
            state = {'state': 'WAITING_NATIVE_EXIT', 'native_pid': self.process.pid,
                     'persistent_simulator': True}
            if self.pending_failure is not None:
                state['job_id'] = self.pending_failure['claim']['job_id']
            return state
        if self.process is not None and self.process.poll() is None:
            return {'state': 'NATIVE_IDLE', 'native_pid': self.process.pid, 'persistent_simulator': True,
                    'recycle_before_next': self.recycle_before_next}
        return {'state': 'IDLE', 'persistent_simulator': True}

    def close(self):
        if self.process is None:
            return True
        process = self.process
        if self.shutdown_pending and process.poll() is None:
            return False
        # The process group is owned and fenced exactly as in the cold worker.
        try:
            self.helpers.stop_group(process.pid, process)
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            if isinstance(error, RuntimeError) and str(error) != (
                    'Owned simulator process group did not exit; claim retained.'):
                raise
            self.shutdown_pending = True
            write_json(self.control / 'service_exit_pending.json', {
                'pid': process.pid, 'owner': self.native_owner, 'error': str(error),
                'jobs_completed': self.jobs, 'timestamp': time.time(),
                'policy': 'Retain ownership and assets; do not claim another task before confirmed exit.'})
            return False
        if self.log is not None:
            self.log.close()
        result = self.helpers.cleanup_native_tmp(Path(self.gpu['appdata_path']), process, owner=self.native_owner)
        write_json(self.control / 'service_exit.json', {**result, 'pid': process.pid,
            'exit_code': process.returncode, 'jobs_completed': self.jobs, 'timestamp': time.time()})
        self.process = self.control = self.native_owner = self.log = None
        self.key = self.family = None
        self.jobs = 0
        self.recycle_before_next = None
        self.shutdown_pending = False
        return True

    def resume_shutdown(self):
        """Heartbeat a failed claim and reap its engine before accepting more work."""
        if not self.shutdown_pending:
            return True
        if self.pending_failure is not None:
            self.queue.heartbeat(self.pending_failure['claim'])
        if self.process.poll() is None or not self.close():
            return False
        pending, self.pending_failure = self.pending_failure, None
        if pending is not None:
            job_id = pending['claim']['job_id']
            self.queue._save(job_id, {**self.queue.status(job_id), 'native_shutdown_pending': False})
            self._release_failed_capture(**pending)
        return True

    def _release_failed_capture(self, claim, output, manifest, value, attempt, cancelled):
        """Release or retry only after the failed native writer has stopped."""
        job_id = claim['job_id']
        self.queue.release(claim)
        rendered = read_json(output / 'preview_complete.json', {})
        if (not cancelled and value['failure_stage'] in ('captured', 'scene_cleanup')
                and manifest.get('diagnostic_pending') and complete_images(manifest, rendered)):
            recover_rendered_attempt(self.queue, job_id,
                'Native screenshots completed before service disposal failed: ' + value['error'])
        elif not cancelled and value['retryable'] and attempt < manifest['max_attempts']:
            self.queue.retry(job_id)

    def ensure(self, manifest):
        key = runtime_identity(manifest)
        runtime_sources = ('src/wia/runtime/scene_capture_assets.py', 'src/wia/runtime/sim/esi_prepared_assets.py') if self.prepare_assets else ()
        runtime_hashes = {name: sha256(ROOT / name) for name in runtime_sources}
        if runtime_hashes:
            key = digest({'native_runtime': key, 'asset_preparation_runtime': runtime_hashes})
        if self.process is not None and (
                self.process.poll() is not None or key != self.key or self.recycle_before_next):
            if not self.close():
                raise NativeShutdownPending('Previous resident native process is still exiting.')
        if self.shutdown_pending:
            raise NativeShutdownPending('Previous resident native process is still exiting.')
        if self.process is not None:
            return
        appdata = Path(self.gpu['appdata_path'])
        scratch = self.helpers.prepare_native_tmp(appdata)
        storage = read_json(self.queue.root / 'category_storage.json', {})
        durable = Path(storage.get('artifact_root', self.queue.root)) / 'resident_services'
        source = ROOT / 'src/wia/runtime/commands/run_esi_capture_service.py'
        pinned = durable / 'runtime' / (sha256(source) + '.py')
        if not pinned.is_file():
            pinned.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, pinned)
        if sha256(pinned) != sha256(source):
            raise RuntimeError('Resident lifecycle code changed while being pinned.')
        self.asset_runtime = None
        if runtime_hashes:
            modules = {}
            for relative, expected in runtime_hashes.items():
                target = durable / 'runtime' / 'asset_preparation' / (expected + '.py')
                if not target.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / relative, target)
                if sha256(target) != expected:
                    raise RuntimeError('Asset preparation helper changed while being pinned.')
                modules[relative] = {'path': str(target), 'sha256': expected}
            self.asset_runtime = {'schema': 'native_asset_preparation_runtime_v1', 'modules': modules}
        self.control = durable / self.owner['id'] / uuid.uuid4().hex
        (self.control / 'requests').mkdir(parents=True)
        (self.control / 'responses').mkdir()
        gpu = {**self.gpu, 'native_tmp_path': str(scratch), 'persistent_simulator': True}
        write_json(self.control / 'gpu.json', gpu)
        if self.asset_runtime:
            write_json(self.control / 'asset_runtime.json', self.asset_runtime)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
            OMNIGIBSON_GPU_ID='0', OMNIGIBSON_HEADLESS='1', PYTHONDONTWRITEBYTECODE='1',
            **frozen_runtime_environment(manifest['code_root']), OMNIGIBSON_APPDATA_PATH=str(appdata),
            XDG_CACHE_HOME=str(appdata / 'xdg_cache'), CUDA_CACHE_PATH=str(appdata / 'cuda_cache'),
            __GL_SHADER_DISK_CACHE_PATH=str(appdata / 'nvidia_shader_cache'))
        for name in ('TMPDIR', 'TMP', 'TEMP'):
            environment[name] = str(scratch)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[name] = os.environ.get(name, '4')
        command = [sys.executable, '-u', str(pinned), '--code-root', manifest['code_root'],
            '--control-dir', str(self.control), '--gpu-info', str(self.control / 'gpu.json'), '--runtime-key', key]
        if self.asset_runtime:
            command += ['--asset-runtime', str(self.control / 'asset_runtime.json')]
        self.native_owner = self.helpers.reserve_native_tmp(appdata, manifest['job_id'])
        self.log = (self.control / 'service.log').open('w')
        self.process = subprocess.Popen(command, cwd=manifest['code_root'], env=environment,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        self.native_owner = self.helpers.record_native_tmp_owner(appdata, self.process, self.native_owner)
        self.key = key
        self.family = (manifest['code_commit'], manifest.get('big_task'), manifest.get('small_task'))
        self.jobs = 0
        write_json(self.control / 'command.json', {'command': command, 'native_pid': self.process.pid,
            'lifecycle_sha256': sha256(pinned), 'temporary_environment': {k: environment[k] for k in ('TMPDIR', 'TMP', 'TEMP')}})

    def capture(self, claim):
        queue, job_id = self.queue, claim['job_id']
        manifest = queue.manifest(job_id)
        attempt = queue.status(job_id)['attempts']
        output = queue.job_dir(job_id) / 'attempts' / f'{attempt:02d}'
        output.mkdir(parents=True)
        queue.heartbeat(claim, native_output=str(output))
        if (queue.job_dir(job_id) / 'cancel.json').exists():
            queue.transition(claim, 'CANCELLED')
            queue.release(claim)
            return
        try:
            verify_snapshot(manifest)
        except ValueError as error:
            queue.transition(claim, 'STALE', failure_stage='snapshot_validation', error=str(error))
            queue.release(claim)
            return
        cached = queue.cached(manifest)
        if cached:
            write_json(output / 'preview_complete.json', {**cached, 'cache_hit': True})
            queue.transition(claim, 'CACHED', receipt=str(output / 'preview_complete.json'), duration_s=0.)
            queue.release(claim)
            return
        began = time.monotonic()
        try:
            preload = None
            if self.preloader is not None:
                waiting = time.monotonic()
                try:
                    preload = self.preload_inputs(manifest).result(timeout=manifest['timeout_s'] or None)
                except Exception as error:
                    preload = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                        'scope': 'CPU read-ahead only; native capture continues with the original assets.'}
                preload = {**preload, 'cpu_workers': self.preload_workers,
                    'wait_before_native_s': time.monotonic() - waiting,
                    'native_construction_parallel': False}
                write_json(output / 'cpu_preload.json', preload)
            self.ensure(manifest)
            renderer_log = RendererLogMonitor(self.control / 'service.log')
            queue.heartbeat(claim, child_pid=self.process.pid, child_pgid=self.process.pid,
                child_start_ticks=self.native_owner['start_ticks'])
            queue.transition(claim, 'RUNNING', gpu=self.gpu, native_output=str(output),
                started_at=time.time(), timeout_s=manifest['timeout_s'], persistent_simulator=True)
            (output / 'resident_service.log').symlink_to(self.control / 'service.log')
            scratch = Path(self.gpu['appdata_path']) / 'native_tmp'
            write_json(output / 'gpu.json', {**self.gpu, 'persistent_simulator': True,
                                           'native_tmp_path': str(scratch)})
            write_json(output / 'command.json', {'persistent_service': str(self.control),
                'native_pid': self.process.pid, 'runtime_key': self.key, 'input_manifest': str(queue.job_dir(job_id) / 'input/manifest.json'),
                'native_tmp_path': str(scratch),
                'temporary_environment': {name: str(scratch) for name in ('TMPDIR', 'TMP', 'TEMP')}})
            request_id = uuid.uuid4().hex
            response_path = self.control / 'responses' / (request_id + '.json')
            write_json(self.control / 'requests' / (request_id + '.json'), {'request_id': request_id,
                'job_id': job_id, 'runtime_key': self.key,
                'manifest': str(queue.job_dir(job_id) / 'input/manifest.json'), 'output': str(output)})
            while not response_path.is_file():
                queue.heartbeat(claim)
                write_json(queue.root / 'workers' / (self.owner['id'] + '.json'), {**self.owner,
                    'state': 'RUNNING', 'persistent_simulator': True, 'native_pid': self.process.pid,
                    'job_id': job_id, 'native_output': str(output), 'heartbeat': time.time()})
                if self.stopping() or (queue.job_dir(job_id) / 'cancel.json').exists():
                    raise RuntimeError('Resident capture interrupted by its owner or cancellation request.')
                if manifest['timeout_s'] and time.monotonic() - began >= manifest['timeout_s']:
                    raise TimeoutError('Resident capture exceeded its immutable timeout.')
                if self.process.poll() is not None:
                    if response_path.is_file():
                        break  # A failed teardown can publish intact images before exiting.
                    raise RuntimeError(f'Resident native process exited with code {self.process.returncode}.')
                fatal = renderer_log.failure()
                if fatal:
                    raise NativeRendererFailure('Native renderer crash: ' + fatal)
                time.sleep(1)
            response = read_json(response_path)
            if (response.get('job_id') != job_id or response.get('request_id') != request_id
                    or response.get('native_pid') != self.process.pid or response.get('runtime_key') != self.key):
                raise RuntimeError('Resident response identity does not match the owned capture.')
            receipt = read_json(output / 'preview_complete.json', {})
            if response['state'] != 'CAPTURED' or not complete_images(manifest, receipt):
                raise RuntimeError(response.get('error', 'Resident capture did not publish complete images.'))
            elapsed = time.monotonic() - began
            receipt.update(duration_s=elapsed, exit_code=None, native_process_running=True)
            if preload is not None:
                receipt['cpu_preload'] = preload
            receipt['timings']['total_s'] = elapsed
            write_json(output / 'preview_complete.json', receipt)
            queue.finish(claim, output / 'preview_complete.json')
            self.jobs += 1
            if not response.get('reusable'):
                stopped = self.close()
                receipt['native_process_running'] = not stopped
                receipt['native_shutdown_pending'] = not stopped
                write_json(output / 'preview_complete.json', receipt)
            elif self.max_jobs > 0 and self.jobs >= self.max_jobs:
                # Only an explicitly requested job limit can recycle a healthy
                # engine. Disk use is observed by the half-hour monitor.
                self.recycle_before_next = {'reasons': ['completed_job_limit'], 'jobs_completed': self.jobs}
                receipt['resident_recycle_before_next'] = self.recycle_before_next
                write_json(output / 'preview_complete.json', receipt)
        except Exception as error:
            failure = read_json(output / 'failure.json', {})
            value = {'error': failure.get('error') or str(error),
                'failure_stage': ('native_renderer' if isinstance(error, NativeRendererFailure)
                                  else failure.get('failure_stage', 'resident_capture')),
                'duration_s': time.monotonic() - began,
                'native_log': str(output / 'native.log'), 'persistent_simulator': True,
                'retryable': isinstance(error, (TimeoutError, NativeShutdownPending, NativeRendererFailure)) or failure.get('retryable', False),
                'partial_images': {p.stem: str(p) for p in output.glob('*.png')}, 'complete': False}
            write_json(output / 'capture_failure.json', value)
            stopped = self.close()
            # A completed capture remains completed if only later service disposal fails.
            if (queue.status(job_id) or {}).get('state') in ('CLAIMED', 'RUNNING'):
                cancelled = (queue.job_dir(job_id) / 'cancel.json').exists()
                queue.transition(claim, 'CANCELLED' if cancelled else 'FAILED', **value,
                                 native_shutdown_pending=not stopped)
                pending = dict(claim=claim, output=output, manifest=manifest,
                               value=value, attempt=attempt, cancelled=cancelled)
                if stopped:
                    self._release_failed_capture(**pending)
                else:
                    self.pending_failure = pending

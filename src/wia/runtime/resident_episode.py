"""CPU client for a GPU-owned persistent native episode service.

Used by run_esi_explore_episode; requests retain ordinary native arguments and
the original runner, model loop and evaluator. Only lifecycle dispatch differs.
"""

from wia.resources import ROOT as WIA_RUNTIME_ROOT
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import uuid
import ast
import runpy

from .lifecycle_log import emit


def native_macro_settings(defaults, profile):
    """Resolve the complete launch-time physics configuration for a task."""
    return dict(defaults, USE_GPU_DYNAMICS=profile['gpu_dynamics'],
                **{key: value for key, value in profile.get('macro_overrides', {}).items()
                   if key != 'USE_GPU_DYNAMICS'})


def task_profiles(plans):
    """Resolve native profiles without importing Kit or any GPU task module."""
    from wia.resources import ESI_ROOT
    from wia.tasks.loader import load_task
    registry = runpy.run_path(str(ESI_ROOT / 'src/active_explore/tasks/_registry.py'))
    profiles = {}
    for plan in plans:
        spec = plan['task']
        task = load_task(spec['path'], variant=spec['variant'], seed=spec['seed'],
                         run_mode=plan['evaluation']['run_mode'])
        module = registry['module_for_payload'](json.loads(task.source_path.read_text()))
        if module is None:
            raise ValueError(f'Cannot resolve native resident profile: {task.path}')
        source = ESI_ROOT / 'src/active_explore' / (module.replace('.', '/') + '.py')
        macros = {}
        def visit(statements):
            for node in statements:
                if isinstance(node, ast.With):
                    visit(node.body)
                elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                                and target.value.id == 'gm' and isinstance(node.value.value, bool)):
                            macros[target.attr] = node.value.value
        visit(ast.parse(source.read_text()).body)
        profiles[str(task.path)] = {'native_module': module,
            'gpu_dynamics': macros.get('USE_GPU_DYNAMICS', False), 'macro_overrides': macros}
    return profiles


def process_matches(control, pid):
    try:
        proc = Path(f'/proc/{pid}')
        command = (proc / 'cmdline').read_bytes().decode(errors='replace')
        return (proc.stat().st_uid == os.getuid() and 'run_esi_episode_service.py' in command
                and str(Path(control).resolve()) in command)
    except FileNotFoundError:
        return False


def stop_service(control, timeout=30):
    """Stop only the verified worker-owned service, including after interruption."""
    control = Path(control)
    service = read_service(control)
    if service is None:
        try:
            service = json.loads((control / 'process.json').read_text())
            if not process_matches(control, service['pid']):
                return
        except (FileNotFoundError, ValueError, KeyError):
            return
        # Startup has no service.json yet; terminate the verified owned child directly.
        try:
            os.kill(service['pid'], signal.SIGTERM)
        except ProcessLookupError:
            return
    emit('simulator_stopping', pid=service['pid'])
    (control / 'stop_service.json').write_text('{}\n')
    if service.get('state') == 'SCENE_RUNNING':
        client = ResidentEpisodeClient(control, {}, {}, '')
        client.request_id = service['request_id']
        client.stop_scene()
    deadline = time.monotonic() + timeout
    while process_matches(control, service['pid']) and time.monotonic() < deadline:
        time.sleep(.2)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not process_matches(control, service['pid']):
            break
        try:
            os.kill(service['pid'], sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + 5
        while process_matches(control, service['pid']) and time.monotonic() < deadline:
            time.sleep(.2)
    if process_matches(control, service['pid']):
        raise RuntimeError('Owned resident simulator failed to exit.')
    emit('simulator_stopped', pid=service['pid'])


def read_service(control):
    try:
        value = json.loads((Path(control) / 'service.json').read_text())
        if not process_matches(control, value['pid']):
            return None
        return value
    except (FileNotFoundError, ValueError, KeyError):
        return None


def owns_gpu(control, physical_index, device_uuid):
    service = read_service(control)
    if service is None or service.get('state') != 'IDLE':
        return False
    gpu_file = Path(control) / 'gpu.json'
    if not gpu_file.exists():
        return False
    gpu = json.loads(gpu_file.read_text())
    if gpu['physical_index'] != physical_index or gpu['uuid'] != device_uuid:
        return False
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                    '--format=csv,noheader,nounits'], text=True)
    pids = {int(parts[0]) for line in text.splitlines() if (parts := line.split(','))
            and parts[1].strip() == device_uuid}
    return pids <= {service['pid']}


class ResidentEpisodeClient:
    def __init__(self, control, gpu, environment, python):
        self.control = Path(control)
        self.gpu, self.environment, self.python = gpu, environment, python
        self.request_id = None
        self.cold_attempted = False
        self.started_reused = False

    def ensure(self, *, quiet=False):
        service = read_service(self.control)
        if service is not None:
            if service.get('state') == 'FAILED':
                emit('simulator_recycling', pid=service['pid'], reason='previous_native_failure')
                stop_service(self.control)
                return self.ensure(quiet=quiet)
            deadline = time.monotonic() + 120
            while service.get('state') == 'SCENE_RUNNING' and time.monotonic() < deadline:
                time.sleep(.2)
                service = read_service(self.control)
                if service is None:
                    raise RuntimeError('Resident service exited while clearing its previous scene.')
            if service.get('state') != 'IDLE':
                raise RuntimeError('Resident simulator is not reusable: ' + str(service))
            maximum = int(os.environ.get('WIA_RESIDENT_MAX_SCENES', '4'))
            if maximum > 0 and service.get('completed_scenes', 0) >= maximum:
                emit('simulator_recycling', pid=service['pid'], completed_scenes=service['completed_scenes'],
                     reason='bounded_renderer_lifetime', max_scenes=maximum)
                stop_service(self.control)
                return self.ensure(quiet=quiet)
            if not quiet:
                emit('simulator_reused', pid=service['pid'], completed_scenes=service.get('completed_scenes', 0))
            return
        started = time.monotonic()
        emit('simulator_initializing', control_dir=str(self.control))
        self.control.mkdir(parents=True, exist_ok=True)
        (self.control / 'stop_service.json').unlink(missing_ok=True)
        for name in ('requests', 'responses'):
            (self.control / name).mkdir(exist_ok=True)
        (self.control / 'gpu.json').write_text(json.dumps(self.gpu, indent=2) + '\n')
        root = WIA_RUNTIME_ROOT
        command = [self.python, '-u', str(root / 'src/wia/runtime/commands/run_esi_episode_service.py'),
                   '--control-dir', str(self.control), '--task-profiles', os.environ['WIA_RESIDENT_TASK_PROFILES']]
        appdata = self.control / 'appdata'
        (appdata / 'native_tmp').mkdir(parents=True, exist_ok=True)
        env = dict(self.environment, CUDA_VISIBLE_DEVICES=self.gpu['uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMNIGIBSON_APPDATA_PATH=str(appdata), XDG_CACHE_HOME=str(appdata / 'xdg_cache'),
                   CUDA_CACHE_PATH=str(appdata / 'cuda_cache'), TMPDIR=str(appdata / 'native_tmp'),
                   TMP=str(appdata / 'native_tmp'), TEMP=str(appdata / 'native_tmp'))
        script = self.control / 'start.sh'
        script.write_text(''.join(f'export {key}={shlex.quote(value)}\n' for key, value in env.items()) +
            f'export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\ncd {shlex.quote(str(root))}\n' +
            shlex.join(command) + f' >{shlex.quote(str(self.control / "service.log"))} 2>&1\n')
        with (self.control / 'service.log').open('a') as log:
            process = subprocess.Popen(command, cwd=root, env={**os.environ, **env,
                'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (self.control / 'process.json').write_text(json.dumps({'pid': process.pid}) + '\n')
        deadline = time.monotonic() + 120
        while read_service(self.control) is None:
            if process.poll() is not None:
                raise RuntimeError('Resident episode service exited during startup; inspect service.log.')
            if time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError('Resident episode service did not become ready; inspect service.log.')
            time.sleep(.5)
        emit('simulator_service_ready', pid=process.pid, elapsed_s=round(time.monotonic()-started, 2))

    def submit(self, native_argv, output, *, _retry=False):
        if not _retry:
            self.output = Path(output)
            self.native_argv = list(native_argv)
            self.input_entries = {path.name for path in self.output.iterdir()}
            self.cold_attempted = False
        self.ensure(quiet=True)
        service = read_service(self.control)
        if service.get('completed_scenes', 0):
            profiles = json.loads(Path(os.environ['WIA_RESIDENT_TASK_PROFILES']).read_text())
            task_path = str(Path(native_argv[native_argv.index('--task-file') + 1]).resolve())
            defaults = service.get('macro_defaults')
            requested = native_macro_settings(defaults, profiles[task_path]) if defaults is not None else None
            if requested is None or requested != service.get('native_macro_profile'):
                emit('simulator_recycling', pid=service['pid'], reason='native_macro_profile_change',
                     previous=service.get('native_macro_profile'), requested=requested)
                stop_service(self.control)
                self.ensure(quiet=True)
        service = read_service(self.control)
        self.started_reused = service.get('completed_scenes', 0) > 0
        self.cold_attempted = self.cold_attempted or not self.started_reused
        self.request_id = uuid.uuid4().hex
        path = self.control / 'requests' / (self.request_id + '.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'request_id': self.request_id, 'native_argv': native_argv,
                                         'output_dir': str(output)}, indent=2) + '\n')
        temporary.replace(path)
        service = read_service(self.control)
        (Path(output) / 'resident_service.json').write_text(json.dumps({
            'control_dir': str(self.control), 'native_pid': service['pid'], 'request_id': self.request_id,
            'persistent_simulator': True, 'appdata_path': str(self.control / 'appdata'),
            'native_argv': native_argv}, indent=2) + '\n')

    def retry_initialization(self, reason, *, agent_started):
        """Retry a failed reused world once cold, preserving the failed attempt.

        Call only before Agent execution. A failed cold start is final; callers
        record it and continue their task list rather than retrying indefinitely.
        """
        if agent_started or self.cold_attempted or (self.output / 'agent.log').exists():
            return False
        validation = self.output / 'scorer_validation.json'
        if validation.exists() and json.loads(validation.read_text()).get('completed'):
            return False
        previous = read_service(self.control)
        emit('simulator_cold_fallback', request_id=self.request_id, reason=reason,
             pid=previous.get('pid') if previous else None)
        stop_service(self.control)
        archive = self.output / 'initialization_attempts' / self.request_id
        archive.mkdir(parents=True, exist_ok=False)
        retained = self.input_entries | {'initialization_attempts', 'gateway.log',
                                         'experiment.json', 'status.json'}
        for path in list(self.output.iterdir()):
            if path.name not in retained:
                path.rename(archive / path.name)
        (self.output / 'simulator_fallback.json').write_text(json.dumps({
            'reason': reason, 'previous_request_id': self.request_id,
            'previous_native_pid': previous.get('pid') if previous else None,
            'failed_attempt_dir': str(archive), 'cold_retry_limit': 1,
            'agent_started': False, 'model_calls_replayed': 0}, indent=2) + '\n')
        self.submit(self.native_argv, self.output, _retry=True)
        return True

    def stop_scene(self):
        path = self.control / 'stop_scene.json'
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps({'request_id': self.request_id}) + '\n')
        temporary.replace(path)

    def dispose_failed(self, output, reason):
        """Dispose only this request's verified native owner after a renderer failure."""
        service = read_service(self.control)
        if service is None or service.get('request_id') != self.request_id:
            raise RuntimeError('Cannot dispose a resident owner without matching request identity.')
        pid = service['pid']
        failure = {'request_id': self.request_id, 'native_pid': pid, 'reusable': False,
                   'error': reason, 'scene_exit_code': -9, 'renderer_failure': True,
                   'exit_code_source': 'Synthetic disposal marker, not a waitpid exit status.'}
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if read_service(self.control) is None:
                break
            os.kill(pid, sig)
            deadline = time.monotonic() + 5
            while read_service(self.control) is not None and time.monotonic() < deadline:
                time.sleep(.2)
        if read_service(self.control) is not None:
            raise RuntimeError('Failed resident owner has not exited; no replacement is allowed.')
        for path in (Path(output) / 'persistent_scene_receipt.json',
                     self.control / 'responses' / (self.request_id + '.json')):
            path.write_text(json.dumps(failure, indent=2) + '\n')
        (Path(output) / 'native_exit_code.txt').write_text('-9\n')

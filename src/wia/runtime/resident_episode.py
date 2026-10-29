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


def read_service(control):
    try:
        value = json.loads((Path(control) / 'service.json').read_text())
        proc = Path(f'/proc/{value["pid"]}')
        command = (proc / 'cmdline').read_bytes().decode(errors='replace')
        if proc.stat().st_uid != os.getuid() or 'run_esi_episode_service.py' not in command or str(Path(control)) not in command:
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

    def ensure(self):
        if read_service(self.control) is not None:
            return
        self.control.mkdir(parents=True, exist_ok=True)
        for name in ('requests', 'responses'):
            (self.control / name).mkdir(exist_ok=True)
        (self.control / 'gpu.json').write_text(json.dumps(self.gpu, indent=2) + '\n')
        root = WIA_RUNTIME_ROOT
        command = [self.python, '-u', str(root / 'src/wia/runtime/commands/run_esi_episode_service.py'),
                   '--control-dir', str(self.control), '--task-profiles', os.environ['WIA_RESIDENT_TASK_PROFILES']]
        target = subprocess.check_output(['tmux', 'new-window', '-d', '-P', '-F', '#{session_name}:#{window_id}',
            '-t', 'codex-gpu', '-n', f'resident-episode-gpu{self.gpu["physical_index"]}'], text=True).strip()
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
        (self.control / 'tmux_target.txt').write_text(target + '\n')
        subprocess.run(['tmux', 'send-keys', '-t', target, '-l', 'source ' + shlex.quote(str(script))], check=True)
        subprocess.run(['tmux', 'send-keys', '-t', target, 'Enter'], check=True)
        deadline = time.monotonic() + 120
        while read_service(self.control) is None:
            if time.monotonic() >= deadline:
                raise RuntimeError('Resident episode service did not become ready; inspect service.log.')
            time.sleep(.5)

    def submit(self, native_argv, output):
        self.ensure()
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

    def stop_scene(self):
        path = self.control / 'stop_scene.json'
        temporary = path.with_suffix('.tmp')
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

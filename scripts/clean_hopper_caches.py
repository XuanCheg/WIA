"""Remove only this campaign's stopped native temporary caches, retaining logs.

Run: python scripts/clean_hopper_caches.py --output-dir CAMPAIGN [--watch]
The absolute trial path must match the launcher's hashed cache namespace. A live
native PID prevents cleanup. Durable observations, recordings and scores remain.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def clean(root):
    for start in root.glob('job_*/run/trial_*/start.sh'):
        trial = start.parent
        receipt = trial / 'temporary_cache_cleanup.json'
        if receipt.exists():
            continue
        pid_file = trial / 'sim_pid.txt'
        if not pid_file.exists():
            continue
        pid = int(pid_file.read_text())
        proc = Path(f'/proc/{pid}')
        if proc.exists():
            state = (proc / 'stat').read_text().split(') ', 1)[1].split()[0]
            if state != 'Z':
                continue
        cache_key = hashlib.sha256(str(trial.resolve()).encode()).hexdigest()[:12]
        cache = Path('/tmp') / f'show-harness-{os.getuid()}' / f'native-{trial.name}-{cache_key}'
        expected = f'export OMNIGIBSON_APPDATA_PATH={cache}'
        if expected not in start.read_text().splitlines() or cache.is_symlink() or not cache.is_dir():
            continue
        if cache.stat().st_uid != os.getuid():
            raise RuntimeError('Cache ownership mismatch.')
        log_root = trial / 'native_runtime_logs'
        for log in (cache / 'kit_logs').rglob('*.log'):
            target = log_root / log.relative_to(cache / 'kit_logs')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, target)
        size = int(subprocess.check_output(['du', '-sb', str(cache)], text=True).split()[0])
        shutil.rmtree(cache)
        data = {'at_utc': datetime.now(timezone.utc).isoformat(), 'cache': str(cache),
                'logical_bytes': size, 'native_pid': pid,
                'scope': 'Stopped trial temporary native cache only; native logs copied to durable trial output.'}
        receipt.write_text(json.dumps(data, indent=2) + '\n')
        print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, action='append', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        for root in args.output_dir:
            clean(root.resolve())
        if not args.watch:
            break
        deadlines = [datetime.fromisoformat(json.loads((root / 'manifest.json').read_text())['deadline_utc']).timestamp()
                     for root in args.output_dir]
        if time.time() > max(deadlines) + 300:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()

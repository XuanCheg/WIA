"""Deduplicate stopped campaign PNG/NPY artifacts without changing bytes or paths.

Run: python scripts/compact_hopper_artifacts.py --plan PLAN.json [--apply]
The plan contains the campaign root and SHA-256 groups. Apply makes shared
historical files read-only and replaces identical copies with atomic hard links.
It never links writable artifacts or removes unique observations or videos.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def allocated_bytes(root):
    seen, total = set(), 0
    for path in root.rglob('*'):
        if path.is_symlink() or not path.is_file():
            continue
        info = path.stat()
        identity = info.st_dev, info.st_ino
        if identity not in seen:
            seen.add(identity)
            total += info.st_blocks * 512
    return total


def require_stopped(root):
    markers = (b'run_hopper_benchmark.py', b'run_esi_agent.py', b'run_esi_robot.py',
               b'run_esi_explore_episode.py', b'run_esi_episode_service.py')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        # Match script arguments, not a shell command containing this source.
        if (any(Path(os.fsdecode(arg)).name.encode() in markers for arg in arguments[:4])
                and any(os.fsencode(str(root)) in arg for arg in arguments)):
            raise RuntimeError(f'Campaign still has a live workload: PID {process.name}')


def compact(plan, receipt_path, *, apply=False):
    root = Path(plan['root']).resolve(strict=True)
    require_stopped(root)
    before = allocated_bytes(root)
    receipt = {'schema': 'immutable_artifact_deduplication_v1',
               'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'root': str(root), 'apply': apply, 'before_allocated_file_bytes': before,
               'completed_groups': [], 'linked_files': 0,
               'policy': 'Byte-identical historical PNG/NPY files; shared inodes are read-only.'}
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    def record():
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(receipt_path)

    record()
    for group in plan['groups']:
        paths, originals = [], []
        for relative in group['paths']:
            path = root / relative
            parts = Path(relative).parts
            if (len(parts) < 4 or not parts[0].startswith('job_') or parts[1] != 'run'
                    or not parts[2].startswith('trial_') or 'source_frames' in parts
                    or path.suffix not in {'.png', '.npy'} or path.is_symlink()
                    or not path.resolve(strict=True).is_relative_to(root)):
                raise ValueError(f'Unexpected historical artifact path: {path}')
            info = path.stat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_size != group['size_bytes'] or digest(path) != group['sha256']):
                raise ValueError(f'Historical artifact changed or is not owned: {path}')
            paths.append(path)
            originals.append(info)
        members = Counter((info.st_dev, info.st_ino) for info in originals)
        if any(info.st_nlink != members[(info.st_dev, info.st_ino)] for info in originals):
            raise ValueError('A planned artifact has hard links outside its verified group.')
        count = 0
        if apply:
            source = paths[0]
            # Seal every existing inode before sharing any link to it.
            for path, info in zip(paths, originals):
                path.chmod(stat.S_IMODE(info.st_mode) & ~0o222)
            for path, info in zip(paths[1:], originals[1:]):
                current = path.stat()
                if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (
                        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
                    raise RuntimeError(f'Artifact changed before replacement: {path}')
                if path.samefile(source):
                    continue
                temporary = path.with_name(path.name + '.dedup-' + uuid.uuid4().hex)
                try:
                    os.link(source, temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                count += 1
            if any(path.stat().st_mode & 0o222 or not path.samefile(source) for path in paths):
                raise RuntimeError('Shared artifact is not immutable or a link replacement failed.')
        receipt['linked_files'] += count
        receipt['completed_groups'].append({'sha256': group['sha256'], 'paths': group['paths'],
            'original_modes': [stat.S_IMODE(info.st_mode) for info in originals],
            'new_links': count})
        record()
    receipt.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                   after_allocated_file_bytes=allocated_bytes(root),
                   status='applied' if apply else 'verified_dry_run')
    receipt['reclaimed_allocated_file_bytes'] = before - receipt['after_allocated_file_bytes']
    record()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    receipt = Path(plan['root']) / 'analysis/storage_cleanup/observation_deduplication.json'
    result = compact(plan, receipt, apply=args.apply)
    print(json.dumps({key: result[key] for key in ('status', 'linked_files',
        'before_allocated_file_bytes', 'after_allocated_file_bytes', 'reclaimed_allocated_file_bytes')}))


if __name__ == '__main__':
    main()

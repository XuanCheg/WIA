"""Verify frozen capture inputs and warm official asset bytes on CPU only.

Run: python -m pytest -q tests/test_capture_preload.py
CapturePreloader uses spawned CPU children. Optional asset preparation decrypts
and unpacks original archives; native construction and GPU APIs stay in Kit.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import threading
import time

import yaml


CHUNK_BYTES = 8 * 1024 * 1024
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
STRUCTURES = {'floor', 'floors', 'wall', 'walls', 'ceiling', 'ceilings', 'fence', 'roof',
              'background', 'door', 'sliding_door'}
PLACEHOLDERS = {'unknown', 'unknown_scene', 'unknown_room', 'none', 'null', 'scene_native', 'full_scene'}


def _read(path, *, collect=False):
    """Every physical read is bounded, including immutable-input verification."""
    before = path.stat()
    digest, size, chunks = hashlib.sha256(), 0, []
    with path.open('rb') as stream:
        while True:
            chunk = stream.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            if collect:
                chunks.append(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('File changed during CPU preload: ' + str(path))
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes_read': size,
            'size_bytes': after.st_size}, b''.join(chunks) if collect else None


def _component(value, description):
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value)
            or value.lower() in PLACEHOLDERS or value in ('.', '..')):
        raise ValueError('Invalid or placeholder ' + description + ': ' + repr(value))
    return value


def _within(root, path):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError('Preload path escapes its declared root: ' + str(path))
    return path


def _data_root(data_path):
    value = data_path if data_path is not None else os.environ.get('OMNIGIBSON_DATA_PATH')
    if not value:
        raise ValueError('Supply data_path or OMNIGIBSON_DATA_PATH for CPU asset preload.')
    root = Path(value).expanduser().resolve()
    return root if root.name == 'behavior-1k-assets' else root / 'behavior-1k-assets'


def _payload(raw):
    value = raw.get('metadata_json', {})
    return json.loads(value) if isinstance(value, str) else value


def _source_assets(value, location='source.metadata_json'):
    if isinstance(value, dict):
        if 'model' in value and ('category' in value or 'cat' in value):
            yield location, value
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                yield from _source_assets(child, location + '.' + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _source_assets(child, location + '[' + str(index) + ']')


def prepare_capture(manifest_path, data_path=None, *, read_assets=True):
    """Return private, JSON-serializable CPU verification/read-ahead evidence.

    Immutable snapshot/hash failures raise ValueError; unavailable official assets
    are explicit diagnostics. Successful preload never means native readiness.
    Asset bytes are streamed into the OS page cache and SHA256 only, never copied,
    decrypted or extracted. Metadata JSON is parsed without importing its classes.
    """
    started = time.monotonic()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_info, encoded = _read(manifest_path, collect=True)
    manifest = json.loads(encoded)
    if not isinstance(manifest, dict):
        raise ValueError('Capture manifest must be a mapping.')
    root = Path(manifest['snapshot_dir']).expanduser().resolve()
    accepted = root.parent / 'accepted.json'
    accepted_info = None
    if accepted.is_file():
        accepted_info, encoded = _read(accepted, collect=True)
        seal = json.loads(encoded).get('manifest_sha256')
        if seal is not None and seal != manifest_info['sha256']:
            raise ValueError('Accepted immutable capture manifest changed.')
    hashes = manifest.get('input_hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Capture manifest has no immutable input_hashes.')
    files = {manifest_path: {**manifest_info, 'roles': ['capture_manifest'], 'read': True}}
    if accepted_info is not None:
        files[accepted] = {**accepted_info, 'roles': ['accepted_manifest_seal'], 'read': True}
    contents, warnings = {}, []

    def warning(code, **details):
        warnings.append({'code': code, **details})

    source_value = manifest.get('snapshot_source_file')
    if not source_value:
        raise ValueError('Immutable capture source snapshot is missing.')
    source_path = _within(root, Path(source_value))
    scene_path = _within(root, Path(manifest.get('snapshot_scene_file', root / 'scene.yaml')))
    task_path = _within(root, Path(manifest.get('snapshot_task_file', root / 'task.yaml')))
    task_required = bool(manifest.get('task_file') or manifest.get('snapshot_task_file') or
                         task_path.relative_to(root).as_posix() in hashes)
    required = {source_path, scene_path} | ({task_path} if task_required else set())
    expected_by_path = {}
    for relative, expected in hashes.items():
        if (not isinstance(relative, str) or not isinstance(expected, str) or
                not re.fullmatch(r'[0-9a-fA-F]{64}', expected)):
            raise ValueError('Invalid immutable capture input hash.')
        path = _within(root, root / relative)
        if path in expected_by_path:
            raise ValueError('Ambiguous immutable input aliases: ' + relative)
        expected_by_path[path] = expected.lower()
        if not path.is_file():
            raise ValueError('Immutable capture input is missing: ' + relative)
        info, encoded = _read(path, collect=path in required)
        if info['sha256'] != expected.lower():
            raise ValueError('Immutable capture input hash mismatch: ' + relative)
        files[path] = {**info, 'roles': ['immutable_input'], 'read': True}
        if encoded is not None:
            contents[path] = encoded
    if not required <= expected_by_path.keys():
        raise ValueError('Required scene/source/task snapshot lacks an immutable input hash.')
    for field, path in (('source_sha256', source_path), ('scene_sha256', scene_path), ('task_sha256', task_path)):
        if manifest.get(field) and (path not in files or manifest[field] != files[path]['sha256']):
            raise ValueError('Explicit immutable capture hash mismatch: ' + field)
    raw = json.loads(contents[source_path])
    scene = yaml.safe_load(contents[scene_path])
    task = yaml.safe_load(contents[task_path]) if task_required else None
    if not isinstance(raw, dict) or not isinstance(scene, dict) or (task_required and not isinstance(task, dict)):
        raise ValueError('Immutable scene/source/task content must contain mappings.')
    if task is not None and task.get('source', {}).get('json') != scene.get('source', {}).get('json'):
        raise ValueError('Immutable scene and task source identities disagree.')
    scene_model = _component(manifest.get('scene_model'), 'scene_model')
    room = manifest.get('room')
    room_filter = room if room not in (None, '', 'full_scene', 'unknown_room') else None
    assets_root = _data_root(data_path).resolve()
    scene_directory = _within(assets_root, assets_root / 'scenes' / scene_model)
    scene_json = scene_directory / 'json' / (scene_model + '_best.json')
    verified_at = time.monotonic()

    def include(path, role, *, parse=False):
        path = _within(assets_root, path)
        if not path.is_file():
            warning('missing_asset_file', path=str(path), role=role)
            return None
        if path in files:
            if role not in files[path]['roles']:
                files[path]['roles'].append(role)
            return contents.get(path)
        if read_assets or parse:
            info, encoded = _read(path, collect=parse)
            files[path] = {**info, 'roles': [role], 'read': True}
            if parse:
                contents[path] = encoded
            return encoded
        files[path] = {'path': str(path), 'sha256': None, 'bytes_read': 0,
                       'size_bytes': path.stat().st_size, 'roles': [role], 'read': False}
        return None

    scene_encoded = include(scene_json, 'official_scene_json', parse=True)
    native_assets, filtered = [], 0
    if scene_encoded is not None:
        official = json.loads(scene_encoded)
        initialization = official.get('objects_info', {}).get('init_info', {})
        if not isinstance(initialization, dict):
            warning('invalid_official_scene_init_info', path=str(scene_json))
            initialization = {}
        for name, item in initialization.items():
            if not isinstance(item, dict) or item.get('class_name') != 'DatasetObject':
                continue
            args = item.get('args', {})
            if not isinstance(args, dict):
                warning('invalid_native_asset_args', object_name=name)
                continue
            rooms = args.get('in_rooms')
            if isinstance(rooms, str):
                rooms = [rooms]
            if room_filter and args.get('category') not in STRUCTURES and 'in_rooms' in args:
                if not isinstance(rooms, list):
                    warning('ambiguous_native_room_assignment', object_name=name)
                elif room_filter not in rooms:
                    filtered += 1
                    continue
            native_assets.append(('official_scene.' + str(name), args))
    metadata = scene_directory / 'metadata'
    layouts = scene_directory / 'layout'
    for directory in (layouts, metadata):
        if directory.is_dir():
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.suffix.lower() in ('.png', '.json', '.txt', '.yaml', '.yml'):
                    include(path, 'scene_layout_or_metadata')
    for path in sorted(scene_directory.glob('*.json')):
        include(path, 'scene_metadata')
    payload = _payload(raw)
    if not isinstance(payload, (dict, list)):
        raise ValueError('Immutable source metadata_json must contain a mapping or list.')
    requested = {}
    for origin, item in [*native_assets, *_source_assets(payload)]:
        categories = {item.get(field) for field in ('category', 'cat') if isinstance(item.get(field), str)}
        if len(categories) > 1:
            warning('ambiguous_asset_category', origin=origin, categories=sorted(categories))
        if not categories:
            warning('unresolved_asset_category', origin=origin)
            continue
        for category in sorted(categories):
            try:
                category = _component(category, 'asset category')
                model = _component(item.get('model'), 'asset model')
            except ValueError:
                warning('missing_or_placeholder_asset_identity', origin=origin,
                        category=category, model=item.get('model') if isinstance(item.get('model'), str) else None)
                continue
            requested.setdefault((category, model), []).append(origin)
    resolved_assets = []
    for (category, model), origins in sorted(requested.items()):
        directory = _within(assets_root, assets_root / 'objects' / category / model)
        # Installed DatasetObject.get_usd_path returns <model>.usdz;
        # USDObject._load reads its .encrypted representation for this dataset.
        encrypted = directory / 'usd' / (model + '.usdz.encrypted')
        unencrypted = directory / 'usd' / (model + '.usdz')
        if unencrypted.is_file():
            warning('unencrypted_asset_representation_not_used', category=category, model=model,
                    ambiguous=encrypted.is_file())
        include(encrypted, 'official_encrypted_object')
        metadata_files = sorted(directory.rglob('*.json')) if directory.is_dir() else []
        for path in metadata_files:
            if path.is_file():
                include(path, 'model_metadata')
        resolved_assets.append({'category': category, 'model': model, 'origins': origins,
                                'encrypted_file': str(encrypted), 'available': encrypted.is_file(),
                                'model_metadata_file_count': len(metadata_files)})
    ended = time.monotonic()
    rows = sorted(files.values(), key=lambda item: item['path'])
    return {'schema': 'capture_cpu_preload_v1', 'ok': True, 'worker_pid': os.getpid(),
            'monotonic_start_s': started, 'monotonic_end_s': ended, 'wall_duration_s': ended - started,
            'timings_s': {'immutable_verification': verified_at - started,
                          'asset_discovery_and_read': ended - verified_at},
            'manifest_path': str(manifest_path), 'manifest_sha256': manifest_info['sha256'],
            'scene_model': scene_model, 'room': room, 'asset_root': str(assets_root),
            'read_assets': bool(read_assets), 'input_hashes_verified': len(expected_by_path),
            'files': rows, 'resolved_file_count': len(rows),
            'read_file_count': sum(item['read'] for item in rows),
            'bytes_read': sum(item['bytes_read'] for item in rows),
            'assets': resolved_assets, 'requested_asset_count': len(requested),
            'available_asset_count': sum(item['available'] for item in resolved_assets),
            'native_furniture_filtered_by_room': filtered, 'warnings': warnings,
            'cpu_worker_gpu_hidden': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'numerical_threads_limited': all(os.environ.get(key) == '2' for key in THREAD_VARIABLES),
            'native_ready': False, 'native_construction_run': False, 'decryption_run': False,
            'scope': 'Immutable task parsing and original encrypted-file OS page-cache warming only; '
                     'structural floor/wall/ceiling bodies are conservatively retained across rooms. '
                     'No extraction, native asset construction, texture residency, grasp or render readiness is established.'}


def _cpu_child_initializer():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in THREAD_VARIABLES:
        os.environ[key] = '2'


def _prepare_frozen_capture(manifest_path, data_path, expected_manifest_sha256, asset_cache_root=None):
    result = prepare_capture(manifest_path, data_path)
    if result['manifest_sha256'] != expected_manifest_sha256:
        raise ValueError('Immutable capture manifest changed after CPU preload submission.')
    if asset_cache_root is not None:
        from wia.runtime.scene_capture_assets import prepare_asset_cache
        began = time.monotonic()
        hashes = {row['path']: row['sha256'] for row in result['files']}
        assets = [{**row, 'encrypted_sha256': hashes.get(row['encrypted_file'])} for row in result['assets']]
        result['prepared_assets'] = prepare_asset_cache(assets, _data_root(data_path).parent, asset_cache_root,
                                                       min_free_bytes=0)
        result['timings_s']['asset_decryption_and_unpack'] = time.monotonic() - began
        result['monotonic_end_s'] = time.monotonic()
        result['wall_duration_s'] = result['monotonic_end_s'] - result['monotonic_start_s']
        result['decryption_run'] = bool(result['prepared_assets']['prepared_asset_count'])
        result['scope'] = ('CPU task verification and original asset decryption/unpacking. '
            'Kit retains native material processing, USD stage construction, physics and sensors. '
            'No robot operation, grasp or render readiness is established.')
    return result


class CapturePreloader:
    """Spawn-safe one/two-worker CPU facade with immutable-manifest deduplication."""
    def __init__(self, workers, data_path=None, *, asset_cache_root=None):
        if not isinstance(workers, int) or isinstance(workers, bool) or workers not in (1, 2):
            raise ValueError('CapturePreloader supports one or two CPU workers.')
        self.data_path = str(Path(data_path).expanduser().resolve()) if data_path is not None else None
        if self.data_path is None and not os.environ.get('OMNIGIBSON_DATA_PATH'):
            raise ValueError('Supply data_path or OMNIGIBSON_DATA_PATH for CPU asset preload.')
        self.asset_cache_root = str(Path(asset_cache_root).resolve()) if asset_cache_root is not None else None
        self._pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_cpu_child_initializer)
        self._futures, self._lock, self._closed = {}, threading.Lock(), False

    def submit(self, manifest_path):
        path = Path(manifest_path).expanduser().resolve()
        info, _ = _read(path)
        key = (str(path), info['sha256'])
        with self._lock:
            if self._closed:
                raise RuntimeError('CapturePreloader is closed.')
            if key not in self._futures:
                self._futures[key] = self._pool.submit(_prepare_frozen_capture, str(path), self.data_path,
                                                       info['sha256'], self.asset_cache_root)
            return self._futures[key]

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Check immutable CPU preload and spawned two-worker overlap without native APIs.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_capture_preload.py --basetemp=/tmp/capture-preload-fixtures
Fixtures contain synthetic official-schema assets; no decryption/GPU work runs.
"""
import ast
import builtins
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

import wia.runtime.scene_capture_preload as preload


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def asset(root, category, model):
    path = root / 'behavior-1k-assets/objects' / category / model
    encrypted = put(path / 'usd' / (model + '.usdz.encrypted'),
                    ('SYNTHETIC ENCRYPTED BYTES ' + category + '/' + model).encode())
    put(path / 'metadata.json', json.dumps({'model': model, 'synthetic': True}).encode())
    return encrypted


def sealed_snapshot(root, data_root, index=0):
    directory = root / 'jobs' / ('task_' + str(index)) / 'input'
    source_identity = 'Example/Task/task_' + str(index) + '.json'
    source = {'scene': 'RoomModel', 'room': 'room_0', 'question': 'A synthetic question.',
              'answer': 'PRIVATE_ORIGINAL_ANSWER', 'metadata_json': json.dumps({
                  'objects': [{'cat': 'apple', 'model': 'app001'},
                              {'nested': {'category': 'mug', 'model': 'mug001'}}]})}
    source_path = put(directory / 'data/source.json', json.dumps(source).encode())
    scene = {'kind': 'scene_patch', 'source': {'json': source_identity},
             'scene': {'variants': {'original': {'patches': []}}}}
    task = {'kind': 'category_robot_task', 'source': {'json': source_identity},
            'private': {'bindings': {}, 'evaluation': {'type': 'grasp_lift', 'target': 'target'}}}
    scene_path = put(directory / 'scene.yaml', yaml.safe_dump(scene).encode())
    task_path = put(directory / 'task.yaml', yaml.safe_dump(task).encode())
    hashes = {str(path.relative_to(directory)): sha(path) for path in (source_path, scene_path, task_path)}
    manifest = {'schema': 'esi_capture_candidate_v1', 'snapshot_dir': str(directory),
        'snapshot_source_file': str(source_path), 'snapshot_scene_file': str(scene_path),
        'task_file': 'original_task.yaml', 'input_hashes': hashes, 'scene_model': 'RoomModel',
        'room': 'room_0', 'source_identity': source_identity,
        'source_sha256': sha(source_path), 'scene_sha256': sha(scene_path), 'task_sha256': sha(task_path)}
    path = put(directory / 'manifest.json', json.dumps(manifest).encode())
    put(directory.parent / 'accepted.json', json.dumps({'manifest_sha256': sha(path)}).encode())
    return path


def reseal(path, update):
    manifest = json.loads(path.read_text())
    update(manifest)
    path.write_text(json.dumps(manifest))
    (path.parent.parent / 'accepted.json').write_text(json.dumps({'manifest_sha256': sha(path)}))


@pytest.fixture
def tree(tmp_path):
    data = tmp_path / 'data'
    assets = data / 'behavior-1k-assets'
    for category, model in [('apple', 'app001'), ('mug', 'mug001'), ('chair', 'chair01'),
                            ('chair', 'chair02'), ('walls', 'walls01'), ('floors', 'floor01'),
                            ('door', 'door001')]:
        asset(data, category, model)
    items = {}
    for name, category, model, rooms in [
            ('local_chair', 'chair', 'chair01', ['room_0']),
            ('foreign_chair', 'chair', 'chair02', ['room_1']),
            ('wall', 'walls', 'walls01', []),
            ('floor', 'floors', 'floor01', ['room_1']),
            ('door', 'door', 'door001', ['room_1'])]:
        items[name] = {'class_module': 'omnigibson.objects.dataset_object', 'class_name': 'DatasetObject',
                       'args': {'name': name, 'category': category, 'model': model, 'in_rooms': rooms}}
    items['primitive'] = {'class_name': 'PrimitiveObject', 'args': {'name': 'primitive', 'category': 'object'}}
    scene_root = assets / 'scenes/RoomModel'
    put(scene_root / 'json/RoomModel_best.json', json.dumps({'objects_info': {'init_info': items}}).encode())
    put(scene_root / 'layout/floor_trav_0.png', b'SYNTHETIC_PNG_BYTES_FOR_READ_AHEAD')
    put(scene_root / 'metadata/room_information.json', b'{"synthetic":true}')
    first = sealed_snapshot(tmp_path, data, 0)
    second = sealed_snapshot(tmp_path, data, 1)
    return data, first, second


def test_preload_verifies_inputs_filters_furniture_and_preserves_original_bytes(tree, monkeypatch):
    data, first, _ = tree
    paths = [path for path in first.parents[2].parent.rglob('*') if path.is_file()]
    before = {path: sha(path) for path in paths}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'PARENT_GPU_MAPPING')
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in ('omnigibson', 'pxr', 'torch', 'cupy', 'cuda')
        assert not name.startswith('core.sim')
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, '__import__', guarded)
        result = preload.prepare_capture(first, data)
    assert result['ok'] and result['input_hashes_verified'] == 3
    assert result['native_furniture_filtered_by_room'] == 1
    requested = {(row['category'], row['model']) for row in result['assets']}
    assert ('chair', 'chair01') in requested and ('chair', 'chair02') not in requested
    assert {('walls', 'walls01'), ('floors', 'floor01'), ('door', 'door001'),
            ('apple', 'app001'), ('mug', 'mug001')} <= requested
    assert result['available_asset_count'] == 6 and not result['warnings']
    assert result['bytes_read'] == sum(row['bytes_read'] for row in result['files'])
    assert all(row['sha256'] == sha(Path(row['path'])) for row in result['files'])
    assert any('scene_layout_or_metadata' in row['roles'] for row in result['files'])
    assert any('model_metadata' in row['roles'] for row in result['files'])
    assert not result['native_ready'] and not result['decryption_run'] and not result['native_construction_run']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'PARENT_GPU_MAPPING'
    assert before == {path: sha(path) for path in paths}
    json.dumps(result)


@pytest.mark.parametrize('relative', ['scene.yaml', 'task.yaml', 'data/source.json'])
def test_changed_immutable_scene_task_or_source_fails(tree, relative):
    data, first, _ = tree
    with (first.parent / relative).open('ab') as stream:
        stream.write(b'CHANGED')
    with pytest.raises(ValueError, match='hash mismatch'):
        preload.prepare_capture(first, data)


def test_missing_source_and_mutated_manifest_seal_fail(tree):
    data, first, second = tree
    (first.parent / 'data/source.json').unlink()
    with pytest.raises(ValueError, match='input is missing'):
        preload.prepare_capture(first, data)
    second.write_text(second.read_text() + ' ')
    with pytest.raises(ValueError, match='manifest changed'):
        preload.prepare_capture(second, data)


@pytest.mark.parametrize('scene_model', ['unknown_scene', '../escaped', '/absolute/path', 'Room/Other'])
def test_scene_placeholder_and_path_escape_rejected(tree, scene_model):
    data, first, _ = tree
    reseal(first, lambda value: value.update(scene_model=scene_model))
    with pytest.raises(ValueError, match='scene_model'):
        preload.prepare_capture(first, data)


def test_missing_asset_is_explicit_diagnostic_without_native_readiness(tree):
    data, first, _ = tree
    (data / 'behavior-1k-assets/objects/apple/app001/usd/app001.usdz.encrypted').unlink()
    result = preload.prepare_capture(first, data)
    assert result['ok'] and not result['native_ready']
    assert any(row['code'] == 'missing_asset_file' and row['path'].endswith('app001.usdz.encrypted')
               for row in result['warnings'])
    assert result['available_asset_count'] == result['requested_asset_count'] - 1


def test_missing_official_scene_warns_and_source_assets_still_read(tree):
    data, first, _ = tree
    (data / 'behavior-1k-assets/scenes/RoomModel/json/RoomModel_best.json').unlink()
    result = preload.prepare_capture(first, data)
    assert result['available_asset_count'] == 2
    assert any(row['code'] == 'missing_asset_file' and row['role'] == 'official_scene_json'
               for row in result['warnings'])
    assert any('scene_layout_or_metadata' in row['roles'] for row in result['files'])


def test_required_task_hash_cannot_be_omitted(tree):
    data, first, _ = tree
    reseal(first, lambda value: value['input_hashes'].pop('task.yaml'))
    with pytest.raises(ValueError, match='lacks an immutable input hash'):
        preload.prepare_capture(first, data)


def test_ambiguous_category_and_asset_representation_are_explicit(tree):
    data, first, _ = tree
    scene_path = data / 'behavior-1k-assets/scenes/RoomModel/json/RoomModel_best.json'
    official = json.loads(scene_path.read_text())
    official['objects_info']['init_info']['local_chair']['args']['cat'] = 'apple'
    scene_path.write_text(json.dumps(official))
    put(data / 'behavior-1k-assets/objects/apple/app001/usd/app001.usdz', b'SYNTHETIC UNENCRYPTED ALTERNATIVE')
    result = preload.prepare_capture(first, data)
    assert any(row['code'] == 'ambiguous_asset_category' for row in result['warnings'])
    assert any(row['code'] == 'unencrypted_asset_representation_not_used' and row['ambiguous']
               for row in result['warnings'])
    assert not any(row['path'].endswith('app001.usdz') for row in result['files'])


def test_read_assets_false_and_environment_data_root(tree, monkeypatch):
    data, first, _ = tree
    monkeypatch.setenv('OMNIGIBSON_DATA_PATH', str(data))
    result = preload.prepare_capture(first, read_assets=False)
    objects = [row for row in result['files'] if 'official_encrypted_object' in row['roles']]
    assert len(objects) == 6 and all(row['sha256'] is None and row['bytes_read'] == 0 for row in objects)
    assert all(not row['read'] for row in objects)
    monkeypatch.delenv('OMNIGIBSON_DATA_PATH')
    with pytest.raises(ValueError, match='Supply data_path'):
        preload.prepare_capture(first)


def test_module_dependencies_are_cpu_only_and_all_physical_reads_are_bounded(tree, monkeypatch):
    data, first, _ = tree
    syntax = ast.parse(Path(preload.__file__).read_text())
    imported = {node.module.split('.')[0] for node in ast.walk(syntax)
                if isinstance(node, ast.ImportFrom) and node.module}
    imported |= {name.name.split('.')[0] for node in ast.walk(syntax) if isinstance(node, ast.Import) for name in node.names}
    assert not imported.intersection({'omnigibson', 'pxr', 'torch', 'cupy', 'cuda'})
    assert preload.CHUNK_BYTES <= 8 * 1024 * 1024
    opening = Path.open
    sizes = []

    class Bounded:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def read(self, size):
            assert 0 < size <= 8 * 1024 * 1024
            sizes.append(size)
            return self.stream.read(size)

    def bounded_open(path, mode='r', *args, **kwargs):
        assert mode == 'rb'
        return Bounded(opening(path, mode, *args, **kwargs))

    with monkeypatch.context() as context:
        context.setattr(Path, 'open', bounded_open)
        assert preload.prepare_capture(first, data)['ok']
    assert sizes


def semantic(result):
    ignored = {'worker_pid', 'monotonic_start_s', 'monotonic_end_s', 'wall_duration_s', 'timings_s',
               'cpu_worker_gpu_hidden', 'numerical_threads_limited'}
    return {key: value for key, value in result.items() if key not in ignored}


def test_spawned_workers_overlap_deduplicate_and_match_one_worker_semantics(tree, monkeypatch):
    data, first, second = tree
    # A sparse synthetic encrypted file produces bounded real read/hash work.
    # No sleep or testing hook is added to production preload code.
    large = data / 'behavior-1k-assets/objects/apple/app001/usd/app001.usdz.encrypted'
    with large.open('r+b') as stream:
        stream.truncate(256 * 1024 * 1024)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'PARENT_GPU_MAPPING')
    monkeypatch.setenv('OMP_NUM_THREADS', '17')
    with preload.CapturePreloader(1, data) as single:
        futures = [single.submit(first), single.submit(second)]
        assert single.submit(first) is futures[0]
        serial = [future.result(timeout=30) for future in futures]
    with preload.CapturePreloader(2, data) as parallel:
        futures = [parallel.submit(first), parallel.submit(second)]
        assert parallel.submit(first) is futures[0]
        concurrent = [future.result(timeout=30) for future in futures]
    assert [semantic(row) for row in serial] == [semantic(row) for row in concurrent]
    assert len({row['worker_pid'] for row in serial}) == 1
    assert len({row['worker_pid'] for row in concurrent}) == 2
    assert max(row['monotonic_start_s'] for row in concurrent) < min(row['monotonic_end_s'] for row in concurrent)
    assert all(row['cpu_worker_gpu_hidden'] and row['numerical_threads_limited'] for row in serial + concurrent)
    assert all(row['worker_pid'] != os.getpid() for row in serial + concurrent)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'PARENT_GPU_MAPPING' and os.environ['OMP_NUM_THREADS'] == '17'
    print('Synthetic two-worker overlap:', [(row['worker_pid'], row['monotonic_start_s'], row['monotonic_end_s'])
                                             for row in concurrent])
    with pytest.raises(RuntimeError, match='closed'):
        parallel.submit(first)


def test_facade_cache_key_includes_immutable_manifest_hash(tree):
    data, first, _ = tree
    with preload.CapturePreloader(1, data) as pool:
        previous = pool.submit(first)
        assert previous.result(timeout=20)['ok']
        reseal(first, lambda value: value.update(seed=5))
        revised = pool.submit(first)
        assert revised is not previous
        assert revised.result(timeout=20)['manifest_sha256'] != previous.result()['manifest_sha256']
    with pytest.raises(ValueError, match='one or two'):
        preload.CapturePreloader(0, data)

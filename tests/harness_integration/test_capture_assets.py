"""Check CPU decryption, byte preservation, cache sharing and integrity.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q \
  tests/test_capture_assets.py \
  --basetemp=/tmp/show-harness-zhouxinyang/cpu_asset_prepare_tests_20261015
All encrypted assets, keys and SDK sources here are synthetic fixtures.
"""
import ast
import builtins
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace
import zipfile

from cryptography.fernet import Fernet
import pytest

import wia.runtime.scene_capture_assets as assets


OFFICIAL_FUNCTION = '''def decrypt_file(encrypted_filename, decrypted_filename):
    with open(get_key_path(), "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)
    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()
    decrypted = fernet.decrypt(encrypted)
    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, encoded):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return path


def encrypted(root, name='first', members=None):
    if members is None:
        members = [('original.usd', b'#usda 1.0\nasset material = @0/Original.mdl@\n'),
                   ('Textures/', b''), ('Textures/Nested/color.png', b'\x00\xffORIGINAL_TEXTURE'),
                   ('Materials/Original.mdl', b'ORIGINAL_MATERIAL\r\n')]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for member, encoded in members:
            output.writestr(member, encoded)
    ciphertext = Fernet((root / 'data/omnigibson.key').read_bytes()).encrypt(archive.getvalue())
    return put(root / 'data/behavior-1k-assets/objects/example' / name / 'usd' / (name + '.usdz.encrypted'),
               ciphertext), members


@pytest.fixture
def tree(tmp_path):
    put(tmp_path / 'data/omnigibson.key', Fernet.generate_key())
    sdk = put(tmp_path / 'sdk/omnigibson/utils/asset_utils.py',
              ('import omnigibson\nraise AssertionError("SDK import is forbidden")\n\n'
               + OFFICIAL_FUNCTION).encode())
    source, members = encrypted(tmp_path)
    return SimpleNamespace(root=tmp_path, data=tmp_path / 'data', sdk=sdk,
                           source=source, members=members, cache=tmp_path / 'cache')


def prepare(tree, **kwargs):
    return assets.prepare_asset_cache(
        [{'encrypted_file': str(tree.source), 'encrypted_sha256': sha(tree.source),
          'category': 'example', 'model': 'first', 'available': True}],
        tree.data, tree.cache, sdk_asset_utils_path=tree.sdk, min_free_bytes=0, **kwargs)


def test_exact_original_bytes_and_source_key_preservation(tree, monkeypatch):
    source_before, key_before = sha(tree.source), sha(tree.data / 'omnigibson.key')
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split('.')[0] not in {'omnigibson', 'pxr', 'torch', 'cupy', 'cuda'}
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, '__import__', guarded)
        result = prepare(tree)
        validated = assets.validate_prepared_asset(result['assets'][0], cache_root=tree.cache)
    assert validated['validated']
    assert result['schema'] == 'capture_prepared_assets_v1'
    assert result['prepared_asset_count'] == 1 and result['cache_hit_count'] == 0
    assert result['worker_pid'] == os.getpid()
    row = result['assets'][0]
    assert row['encrypted_sha256'] == source_before
    assert row['encrypted_md5'] == hashlib.md5(tree.source.read_bytes()).hexdigest()
    for name, encoded in tree.members:
        if not name.endswith('/'):
            assert (Path(row['prepared_dir']) / name).read_bytes() == encoded
    assert not list(tree.cache.rglob('*.usdz'))
    assert not list(tree.cache.glob('.pending-*'))
    assert sha(tree.source) == source_before and sha(tree.data / 'omnigibson.key') == key_before
    assert result['total_prepared_bytes'] == sum(len(encoded) for _, encoded in tree.members)
    assert row['file_count'] == 3
    assert result['sdk_asset_utils_sha256'] == sha(tree.sdk)
    assert result['decrypt_function_sha256'] == hashlib.sha256(OFFICIAL_FUNCTION.rstrip().encode()).hexdigest()
    manifest = json.loads(Path(row['cache_manifest']).read_text())
    assert manifest['source_version'] == row['source_version']
    assert manifest['root_usd'] == 'original.usd' and not manifest['mdl_paths_modified']
    files = {member['path']: member for member in manifest['members'] if member['type'] == 'file'}
    assert files['Textures/Nested/color.png']['sha256'] == hashlib.sha256(b'\x00\xffORIGINAL_TEXTURE').hexdigest()
    encoded_result = json.dumps(result)
    assert (tree.data / 'omnigibson.key').read_text() not in encoded_result
    assert 'ORIGINAL_TEXTURE' not in encoded_result


def test_all_cache_directories_and_files_are_private(tree):
    result = prepare(tree)
    for path in [tree.cache, *tree.cache.rglob('*')]:
        assert not path.is_symlink()
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)
    assert assets.validate_prepared_asset(result['assets'][0], cache_root=tree.cache)['validated']


def test_cache_reuse_does_not_repeat_official_decryption(tree, monkeypatch):
    first = prepare(tree)
    loader = assets._load_decryptor

    def no_decrypt(*args):
        _, provenance, key = loader(*args)

        def forbidden(*args):
            pytest.fail('Published cache reuse must not repeat decryption.')

        return forbidden, provenance, key

    monkeypatch.setattr(assets, '_load_decryptor', no_decrypt)
    second = prepare(tree)
    assert second['cache_hit_count'] == 1
    assert second['assets'][0]['cache_hit']
    assert first['assets'][0]['cache_manifest_sha256'] == second['assets'][0]['cache_manifest_sha256']
    assert second['timings_s']['decryption'] == 0 and second['timings_s']['extraction'] == 0


def _concurrent_prepare(source, data, cache, sdk, ready_dir, index):
    started = time.monotonic()
    put(Path(ready_dir) / ('ready-' + str(index)), b'ready')
    deadline = started + 20
    while len(list(Path(ready_dir).glob('ready-*'))) < 2:
        if time.monotonic() > deadline:
            raise RuntimeError('Concurrent CPU fixture did not reach its barrier.')
        time.sleep(0.01)
    begin = time.monotonic()
    result = assets.prepare_asset_cache([{'encrypted_file': source}], data, cache,
                                       sdk_asset_utils_path=sdk, min_free_bytes=0)
    return begin, time.monotonic(), result


@pytest.mark.parametrize('shared', [True, False])
def test_spawned_cpu_processes_overlap_and_share_atomic_entries(tree, shared):
    # Real crypto and decompression must outlast the barrier polling interval.
    # Incompressible payloads keep this an actual two-worker CPU overlap check.
    members = [('root.usd', b'ORIGINAL_USD'), ('Textures/bytes.bin', os.urandom(8 * 1024**2))]
    encrypted(tree.root, 'first', members)
    second, _ = encrypted(tree.root, 'second', members)
    ready = tree.root / 'ready'
    ready.mkdir()
    sources = [tree.source, tree.source if shared else second]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_concurrent_prepare, str(source), str(tree.data), str(tree.cache),
                               str(tree.sdk), str(ready), index)
                   for index, source in enumerate(sources)]
        outputs = [future.result(timeout=40) for future in futures]
    assert outputs[0][0] < outputs[1][1] and outputs[1][0] < outputs[0][1]
    assert len({item[2]['worker_pid'] for item in outputs}) == 2
    assert sum(item[2]['cache_hit_count'] for item in outputs) == (1 if shared else 0)
    assert len([path for path in tree.cache.iterdir() if path.is_dir()]) == (1 if shared else 2)
    for _, _, result in outputs:
        assert assets.validate_prepared_asset(result['assets'][0], cache_root=tree.cache)['validated']
    assert not list(tree.cache.glob('.pending-*'))


def test_changed_source_invalidates_old_row_and_expected_preload_hash(tree):
    first = prepare(tree)
    previous = first['assets'][0]
    encrypted(tree.root, members=[('changed.usd', b'CHANGED_FIXTURE')])
    with pytest.raises(ValueError, match='source version changed'):
        assets.validate_prepared_asset(previous, cache_root=tree.cache)
    with pytest.raises(ValueError, match='known preload hash'):
        assets.prepare_asset_cache([{'encrypted_file': str(tree.source),
                                     'encrypted_sha256': previous['encrypted_sha256']}],
                                  tree.data, tree.cache, sdk_asset_utils_path=tree.sdk, min_free_bytes=0)
    second = prepare(tree)
    assert second['cache_hit_count'] == 0
    assert Path(previous['prepared_dir']).exists()
    assert previous['prepared_dir'] != second['assets'][0]['prepared_dir']


def test_changed_source_during_decryption_cannot_publish(tree, monkeypatch):
    loader = assets._load_decryptor

    def changing(*args):
        decrypt, provenance, key = loader(*args)

        def mutate(source, output):
            decrypt(source, output)
            with Path(source).open('ab') as stream:
                stream.write(b'CHANGED_FIXTURE')

        return mutate, provenance, key

    monkeypatch.setattr(assets, '_load_decryptor', changing)
    with pytest.raises(ValueError, match='source changed during'):
        prepare(tree)
    assert not [path for path in tree.cache.iterdir() if path.is_dir()]


def test_corrupt_published_file_is_rejected_and_retained(tree):
    row = prepare(tree)['assets'][0]
    usd = Path(row['usd_file'])
    usd.write_bytes(b'CORRUPT')
    with pytest.raises(ValueError, match='file hash/size mismatch'):
        assets.validate_prepared_asset(row, cache_root=tree.cache)
    with pytest.raises(ValueError, match='file hash/size mismatch'):
        prepare(tree)
    assert usd.read_bytes() == b'CORRUPT'


def test_manifest_seal_and_cache_containment_are_verified(tree):
    row = prepare(tree)['assets'][0]
    with pytest.raises(ValueError, match='escapes or differs'):
        assets.validate_prepared_asset({**row, 'usd_file': '/tmp/outside.usd',
                                        'prepared_dir': str(tree.root)}, cache_root=tree.cache)
    manifest = Path(row['cache_manifest'])
    manifest.write_bytes(manifest.read_bytes() + b' ')
    with pytest.raises(ValueError, match='manifest seal mismatch'):
        assets.validate_prepared_asset(row, cache_root=tree.cache)


@pytest.mark.parametrize('bad', ['../outside.usd', '/outside.usd', 'C:/outside.usd',
                                'Materials/../../outside.usd', 'Materials\\outside.usd',
                                'Materials//outside.usd', './outside.usd'])
def test_path_traversal_is_rejected_before_extraction(tree, bad):
    source, _ = encrypted(tree.root, members=[('root.usd', b'USD'), (bad, b'ESCAPE')])
    with pytest.raises(ValueError, match='Unsafe ZIP member path'):
        prepare(tree)
    assert not list(tree.cache.glob('.pending-*'))
    assert not (tree.root / 'outside.usd').exists()
    assert source.exists()


def test_duplicate_members_are_rejected(tree):
    with pytest.warns(UserWarning, match='Duplicate name'):
        encrypted(tree.root, members=[('root.usd', b'USD'), ('root.usd', b'OTHER')])
    with pytest.raises(ValueError, match='Duplicate ZIP member'):
        prepare(tree)


def test_zip_links_and_file_directory_collisions_are_rejected(tree):
    link = zipfile.ZipInfo('link')
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    encrypted(tree.root, members=[('root.usd', b'USD'), (link, b'/outside')])
    with pytest.raises(ValueError, match='ZIP links'):
        prepare(tree)
    encrypted(tree.root, members=[('root.usd', b'USD'), ('textures', b'FILE'), ('textures/a', b'CHILD')])
    with pytest.raises(ValueError, match='conflicts with a member directory'):
        prepare(tree)


@pytest.mark.parametrize('members', [[('nested/root.usd', b'USD')],
                                   [('one.usd', b'ONE'), ('two.usd', b'TWO')]])
def test_official_exactly_one_root_usd_requirement(tree, members):
    encrypted(tree.root, members=members)
    with pytest.raises(ValueError, match='exactly one root'):
        prepare(tree)


@pytest.mark.parametrize('key_state', ['missing', 'invalid', 'wrong'])
def test_missing_corrupt_and_nonmatching_key_diagnostics(tree, key_state):
    key = tree.data / 'omnigibson.key'
    if key_state == 'missing':
        key.unlink()
    else:
        key.write_bytes(b'INVALID_KEY' if key_state == 'invalid' else Fernet.generate_key())
    expected = {'missing': 'key is missing', 'invalid': 'invalid Fernet format',
                'wrong': 'key does not match'}[key_state]
    with pytest.raises(ValueError, match=expected):
        prepare(tree)
    assert not list(tree.cache.glob('.pending-*'))


def test_default_sdk_location_is_found_without_importing_it(tree, monkeypatch):
    calls = []

    def locate(name):
        calls.append(name)
        return SimpleNamespace(submodule_search_locations=[str(tree.sdk.parents[1])])

    monkeypatch.setattr(assets.importlib.util, 'find_spec', locate)
    result = assets.prepare_asset_cache([{'encrypted_file': str(tree.source)}],
                                       tree.data / 'behavior-1k-assets', tree.cache, min_free_bytes=0)
    assert calls == ['omnigibson'] and result['prepared_asset_count'] == 1


def test_missing_or_unsupported_sdk_is_a_clear_error(tree, monkeypatch):
    monkeypatch.setattr(assets.importlib.util, 'find_spec', lambda name: None)
    with pytest.raises(ValueError, match='package is unavailable'):
        assets.prepare_asset_cache([], tree.data, tree.cache)
    tree.sdk.write_text(OFFICIAL_FUNCTION.replace('fernet.decrypt(encrypted)', 'encrypted'))
    with pytest.raises(ValueError, match='Unsupported SDK decrypt_file implementation'):
        prepare(tree)
    tree.sdk.write_text('def decrypt_file(one):\n    return one\n')
    with pytest.raises(ValueError, match='Unsupported SDK decrypt_file signature'):
        prepare(tree)


def test_space_budget_and_input_cap_fail_without_publishing(tree, monkeypatch):
    monkeypatch.setattr(assets.shutil, 'disk_usage', lambda root: SimpleNamespace(free=100))
    with pytest.raises(ValueError, match='free-space reserve'):
        assets.prepare_asset_cache([{'encrypted_file': str(tree.source)}], tree.data, tree.cache,
                                   sdk_asset_utils_path=tree.sdk, min_free_bytes=100)
    assert not list(tree.cache.glob('.pending-*'))
    monkeypatch.setattr(assets, 'MAX_ENCRYPTED_BYTES', tree.source.stat().st_size - 1)
    with pytest.raises(ValueError, match='CPU input cap'):
        prepare(tree)
    assert not [path for path in tree.cache.iterdir() if path.is_dir()]


def test_cross_worker_space_reservations_are_counted(tree, monkeypatch):
    assets._cache_root(tree.cache, create=True)
    first, second = tree.cache / '.pending-first', tree.cache / '.pending-second'
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    monkeypatch.setattr(assets.shutil, 'disk_usage', lambda root: SimpleNamespace(free=100))
    assets._reserve_space(tree.cache, first, 50, 20)
    with pytest.raises(ValueError, match='free-space reserve'):
        assets._reserve_space(tree.cache, second, 40, 20)
    assert not (second / 'reservation.json').exists()


def test_cached_symlinks_and_extra_files_are_rejected(tree):
    row = prepare(tree)['assets'][0]
    content = Path(row['prepared_dir'])
    put(content / 'unexpected', b'EXTRA')
    (content / 'unexpected').chmod(0o600)
    with pytest.raises(ValueError, match='unexpected file'):
        assets.validate_prepared_asset(row, cache_root=tree.cache)
    (content / 'unexpected').unlink()
    usd = Path(row['usd_file'])
    usd.unlink()
    usd.symlink_to(tree.source)
    with pytest.raises(ValueError, match='private regular file'):
        assets.validate_prepared_asset(row, cache_root=tree.cache)


def test_unavailable_assets_are_explicitly_skipped(tree):
    result = assets.prepare_asset_cache([{'encrypted_file': '/does/not/exist', 'available': False}],
                                       tree.data, tree.cache, sdk_asset_utils_path=tree.sdk)
    assert result['unavailable_asset_count'] == 1 and result['prepared_asset_count'] == 0


def test_no_native_or_gpu_dependencies():
    syntax = ast.parse(Path(assets.__file__).read_text())
    imports = {node.module.split('.')[0] for node in ast.walk(syntax)
               if isinstance(node, ast.ImportFrom) and node.module}
    imports |= {name.name.split('.')[0] for node in ast.walk(syntax)
                if isinstance(node, ast.Import) for name in node.names}
    assert not imports & {'omnigibson', 'pxr', 'torch', 'cupy', 'cuda'}
    assert assets.MAX_ENCRYPTED_BYTES == 2 * 1024**3
    assert assets.MIN_FREE_BYTES == 0

"""Decrypt and unpack original DatasetObject archives using CPU libraries only.

Run: python -m pytest -q tests/test_capture_assets.py
The official decrypt_file function is compiled unchanged, without importing its
SDK. Fernet reads whole inputs, so each encrypted input is limited to 2 GiB.
Cache roots must be task-owned /tmp directories. No disk threshold is enabled
by default; the campaign records disk usage every half hour instead.
Published files preserve archive bytes; Kit must perform its usual MDL fix later.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import zipfile

from cryptography.fernet import Fernet, InvalidToken


CHUNK_BYTES = 8 * 1024 * 1024
MAX_ENCRYPTED_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 0
MANIFEST_NAME = 'cache_manifest.json'
_HASH = re.compile(r'[0-9a-f]{64}')


def _version(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Encrypted asset is not a regular file: ' + str(path))
    return {'path': str(path), 'device': info.st_dev, 'inode': info.st_ino,
            'size': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


def _hashes(path, *, source=False):
    before = _version(path)
    if source and before['size'] > MAX_ENCRYPTED_BYTES:
        raise ValueError('Encrypted asset exceeds the 2 GiB CPU input cap: ' + str(path))
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as stream:
        while chunk := stream.read(CHUNK_BYTES):
            sha.update(chunk)
            md5.update(chunk)
    after = _version(path)
    if before != after:
        raise ValueError('File changed during CPU asset verification: ' + str(path))
    return sha.hexdigest(), md5.hexdigest(), after


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def _write_private(path, encoded):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _private_directory(path):
    if path.is_symlink():
        raise ValueError('Asset cache directory must not be a symlink: ' + str(path))
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Asset cache directory must be owned by this user and private (0700): ' + str(path))


def _cache_root(value, *, create=False):
    supplied = Path(value).expanduser()
    if supplied.is_symlink():
        raise ValueError('Asset cache root must not be a symlink.')
    root = supplied.resolve()
    if root == Path('/tmp').resolve() or not root.is_relative_to(Path('/tmp').resolve()):
        raise ValueError('Asset cache root must be a task-owned directory below /tmp.')
    if create:
        _private_directory(root)
    elif not root.is_dir():
        raise ValueError('Asset cache root is missing: ' + str(root))
    return root


@contextmanager
def _lock(path):
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Asset cache lock must be a private regular file.')
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _load_decryptor(data_root, supplied_path):
    root = Path(data_root).expanduser().resolve()
    if root.name == 'behavior-1k-assets':
        root = root.parent
    key_path = root / 'omnigibson.key'
    if not key_path.is_file():
        raise ValueError('OmniGibson decryption key is missing: ' + str(key_path))
    try:
        Fernet(key_path.read_bytes())
    except (ValueError, TypeError) as error:
        raise ValueError('OmniGibson decryption key has an invalid Fernet format.') from error
    if supplied_path is None:
        try:
            spec = importlib.util.find_spec('omnigibson')
        except (ImportError, ValueError) as error:
            raise ValueError('Cannot locate installed OmniGibson without importing its SDK.') from error
        if spec is None or not spec.submodule_search_locations:
            raise ValueError('Installed OmniGibson package is unavailable for CPU decryption.')
        locations = list(spec.submodule_search_locations)
        if len(locations) != 1:
            raise ValueError('Unsupported OmniGibson package search locations.')
        path = Path(locations[0]) / 'utils' / 'asset_utils.py'
    else:
        path = Path(supplied_path).expanduser().resolve()
    try:
        encoded = path.read_bytes()
        syntax = ast.parse(encoded, filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError('Cannot read installed OmniGibson asset_utils.py: ' + str(path)) from error
    functions = [node for node in syntax.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'decrypt_file']
    if len(functions) != 1:
        raise ValueError('Unsupported SDK: expected one top-level decrypt_file function.')
    function = functions[0]
    args = function.args
    if (function.decorator_list or args.posonlyargs or args.vararg or args.kwarg
            or args.kwonlyargs or args.defaults or args.kw_defaults
            or [arg.arg for arg in args.args] != ['encrypted_filename', 'decrypted_filename']
            or any(arg.annotation is not None for arg in args.args) or function.returns is not None):
        raise ValueError('Unsupported SDK decrypt_file signature.')
    # Only the known official Fernet function is supported. Compare AST structure,
    # ignoring line positions and harmless formatting, then compile the original.
    expected = ast.parse('''def decrypt_file(encrypted_filename, decrypted_filename):
    with open(get_key_path(), "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)
    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()
    decrypted = fernet.decrypt(encrypted)
    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)
''').body[0]
    if ast.dump(function, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Unsupported SDK decrypt_file implementation; use normal native loading.')
    function_source = ast.get_source_segment(encoded.decode('utf-8'), function)
    namespace = {'Fernet': Fernet, 'get_key_path': lambda: str(key_path),
                 '__builtins__': {'open': open}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    provenance = {'sdk_asset_utils_path': str(path),
                  'sdk_asset_utils_sha256': hashlib.sha256(encoded).hexdigest(),
                  'decrypt_function_sha256': hashlib.sha256(function_source.encode('utf-8')).hexdigest()}
    return namespace['decrypt_file'], provenance, key_path


def _member_path(name):
    if (not isinstance(name, str) or not name or '\x00' in name or '\\' in name
            or name.startswith('/') or ntpath.splitdrive(name)[0]):
        raise ValueError('Unsafe ZIP member path: ' + repr(name))
    relative = name[:-1] if name.endswith('/') else name
    if not relative or any(part in ('', '.', '..') for part in relative.split('/')):
        raise ValueError('Unsafe ZIP member path: ' + repr(name))
    return PurePosixPath(relative)


def _inspect_archive(archive):
    members, paths, files = [], set(), set()
    for item in archive.infolist():
        path = _member_path(item.orig_filename)
        name = path.as_posix()
        mode = item.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or bool(item.flag_bits & 1):
            raise ValueError('ZIP links, special files and encrypted members are unsupported: ' + name)
        is_dir = item.is_dir()
        if (kind == stat.S_IFDIR and not is_dir) or (kind == stat.S_IFREG and is_dir):
            raise ValueError('ZIP member type conflicts with its path: ' + name)
        if name in paths:
            raise ValueError('Duplicate ZIP member path: ' + name)
        paths.add(name)
        if not is_dir:
            files.add(name)
        members.append((item, name, is_dir))
    for _, name, _ in members:
        if any(parent.as_posix() in files for parent in PurePosixPath(name).parents
               if parent.as_posix() != '.'):
            raise ValueError('ZIP file conflicts with a member directory: ' + name)
    root_usds = [name for _, name, is_dir in members
                 if not is_dir and '/' not in name and name.endswith('.usd')]
    if len(root_usds) != 1:
        raise ValueError('Expected exactly one root *.usd in the official asset archive.')
    return members, root_usds[0], sum(item.file_size for item, _, is_dir in members if not is_dir)


def _reserve_space(root, temporary, expanded_bytes, minimum):
    # Reservations live only inside unpublished temporary directories. A crashed
    # worker leaves a conservative reservation until the task owner cleans up.
    with _lock(root / '.space.lock'):
        reserved = 0
        for pending in root.glob('.pending-*'):
            record = pending / 'reservation.json'
            if record.exists():
                if pending.is_symlink() or record.is_symlink():
                    raise ValueError('Unsafe cache space reservation.')
                reserved += json.loads(record.read_text())['expanded_bytes']
        if minimum > 0 and shutil.disk_usage(root).free - reserved - expanded_bytes < minimum:
            raise ValueError('Expanded asset cache would leave less than the configured free-space reserve.')
        _write_private(temporary / 'reservation.json', _json_bytes({'expanded_bytes': expanded_bytes}))


def _extract(archive, members, content):
    output = []
    for item, relative, is_dir in members:
        path = content / relative
        directory = path if is_dir else path.parent
        for parent in [*reversed(directory.relative_to(content).parents),
                       directory.relative_to(content)]:
            if parent.as_posix() != '.':
                (content / parent).mkdir(mode=0o700, exist_ok=True)
        if is_dir:
            output.append({'path': relative, 'type': 'directory', 'size': 0})
            continue
        sha, size = hashlib.sha256(), 0
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with archive.open(item) as source, os.fdopen(fd, 'wb') as target:
            while chunk := source.read(CHUNK_BYTES):
                target.write(chunk)
                sha.update(chunk)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size != item.file_size:
            raise ValueError('Extracted ZIP member size mismatch: ' + relative)
        output.append({'path': relative, 'type': 'file', 'size': size, 'sha256': sha.hexdigest()})
    return output


def _private_file(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('Prepared asset must be a private regular file: ' + str(path))


def _validated_metadata(row, root, *, verify_source=True):
    try:
        source_sha, function_sha = row['encrypted_sha256'], row['decrypt_function_sha256']
        if not _HASH.fullmatch(source_sha) or not _HASH.fullmatch(function_sha):
            raise ValueError('Prepared asset has invalid source/function hashes.')
        entry = root / (source_sha + '-' + function_sha)
        content, manifest_path = entry / 'content', entry / MANIFEST_NAME
        for field, expected in (('prepared_dir', content), ('cache_manifest', manifest_path)):
            supplied = Path(row[field])
            if not supplied.is_absolute() or supplied != expected or supplied.resolve() != expected:
                raise ValueError('Prepared asset path escapes or differs from its declared cache: ' + field)
        for directory in (root, entry, content):
            if directory.is_symlink() or not directory.is_dir() or directory.stat().st_mode & 0o077:
                raise ValueError('Prepared asset directory is missing or not private.')
        _private_file(manifest_path)
        encoded = manifest_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row['cache_manifest_sha256']:
            raise ValueError('Prepared asset cache manifest seal mismatch.')
        manifest = json.loads(encoded)
        if (manifest['schema'] != 'capture_prepared_asset_v1'
                or manifest['encrypted_sha256'] != source_sha
                or manifest['decrypt_function_sha256'] != function_sha
                or manifest['encrypted_md5'] != row['encrypted_md5']
                or manifest['source_version'] != row['source_version']
                or manifest['encrypted_file'] != row['encrypted_file']
                or manifest['sdk_asset_utils_sha256'] != row['sdk_asset_utils_sha256']):
            raise ValueError('Prepared asset source/provenance association mismatch.')
        source = Path(row['encrypted_file'])
        if not source.is_absolute() or str(source.resolve()) != str(source):
            raise ValueError('Prepared asset source path is not canonical and absolute.')
        if _version(source) != row['source_version']:
            raise ValueError('Encrypted source version changed since asset preparation.')
        if verify_source:
            sha, md5, version = _hashes(source, source=True)
            if (sha != source_sha or md5 != row['encrypted_md5'] or version != row['source_version']):
                raise ValueError('Encrypted source hash changed since asset preparation.')
        expected_files, expected_dirs = {}, set()
        for member in manifest['members']:
            relative = _member_path(member['path']).as_posix()
            if relative in expected_files or relative in expected_dirs:
                raise ValueError('Duplicate prepared asset manifest member.')
            if member['type'] == 'file':
                expected_files[relative] = member
            elif member['type'] == 'directory':
                expected_dirs.add(relative)
            else:
                raise ValueError('Invalid prepared asset manifest member type.')
        for name in [*expected_files, *expected_dirs]:
            expected_dirs.update(parent.as_posix() for parent in PurePosixPath(name).parents
                                 if parent.as_posix() != '.')
        found_files, found_dirs = set(), set()
        for directory, directories, files in os.walk(content, followlinks=False):
            for name in directories:
                path = Path(directory) / name
                if path.is_symlink() or path.stat().st_mode & 0o077:
                    raise ValueError('Prepared asset contains an unsafe directory.')
                found_dirs.add(path.relative_to(content).as_posix())
            for name in files:
                path = Path(directory) / name
                _private_file(path)
                relative = path.relative_to(content).as_posix()
                if relative not in expected_files:
                    raise ValueError('Prepared asset contains an unexpected file: ' + relative)
                sha, _, version = _hashes(path)
                expected = expected_files[relative]
                if sha != expected['sha256'] or version['size'] != expected['size']:
                    raise ValueError('Prepared asset file hash/size mismatch: ' + relative)
                found_files.add(relative)
        if found_files != set(expected_files) or found_dirs != expected_dirs:
            raise ValueError('Prepared asset files/directories differ from the sealed manifest.')
        roots = [name for name in found_files if '/' not in name and name.endswith('.usd')]
        if len(roots) != 1 or roots[0] != manifest['root_usd']:
            raise ValueError('Prepared asset must have exactly one root *.usd.')
        if Path(row['usd_file']) != content / roots[0]:
            raise ValueError('Prepared USD path differs from its sealed manifest.')
        total = sum(member['size'] for member in expected_files.values())
        if (total != manifest['total_bytes'] or total != row['total_bytes']
                or len(found_files) != manifest['file_count'] or len(found_files) != row['file_count']):
            raise ValueError('Prepared asset byte/file totals differ from its sealed manifest.')
        if _version(source) != row['source_version']:
            raise ValueError('Encrypted source version changed during cache validation.')
        return {**row, 'validated': True, 'validation_worker_pid': os.getpid()}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid prepared asset cache metadata: ' + str(error)) from error


def validate_prepared_asset(row, *, cache_root):
    """Validate cache containment, seal, source association and every output byte."""
    return _validated_metadata(row, _cache_root(cache_root))


def _row(manifest, entry, seal, *, cache_hit, asset):
    fields = ('encrypted_file', 'encrypted_sha256', 'encrypted_md5', 'source_version',
              'sdk_asset_utils_sha256', 'decrypt_function_sha256', 'total_bytes', 'file_count')
    return {**{key: manifest[key] for key in fields},
            **{key: asset[key] for key in ('category', 'model', 'available') if key in asset},
            'prepared_dir': str(entry / 'content'), 'usd_file': str(entry / 'content' / manifest['root_usd']),
            'cache_manifest': str(entry / MANIFEST_NAME), 'cache_manifest_sha256': seal,
            'cache_hit': cache_hit}


def prepare_asset_cache(assets, data_root, cache_root, *, sdk_asset_utils_path=None,
                        min_free_bytes=MIN_FREE_BYTES):
    """Prepare available encrypted archives and return JSON metadata, never bytes.

    Entries are addressed by encrypted SHA256 and official function SHA256. Each
    asset is protected by flock and published by one directory rename. Corrupt
    published entries raise ValueError and remain intact for the task owner.
    """
    started = time.monotonic()
    if not isinstance(min_free_bytes, int) or min_free_bytes < 0:
        raise ValueError('min_free_bytes must be a nonnegative integer.')
    root = _cache_root(cache_root, create=True)
    decrypt, provenance, key_path = _load_decryptor(data_root, sdk_asset_utils_path)
    key_before = _version(key_path)
    rows, unavailable = [], 0
    timings = {'source_verification': 0.0, 'lock_wait': 0.0, 'decryption': 0.0,
               'extraction': 0.0, 'cache_validation': 0.0}
    for asset in assets:
        if asset.get('available') is False:
            unavailable += 1
            continue
        supplied = Path(asset['encrypted_file']).expanduser()
        if not supplied.is_absolute():
            raise ValueError('Encrypted asset path must be absolute.')
        source = supplied.resolve()
        before_read = time.monotonic()
        sha, md5, version = _hashes(source, source=True)
        expected = asset.get('encrypted_sha256')
        if expected is not None and expected != sha:
            raise ValueError('Encrypted source SHA256 differs from the known preload hash: ' + str(source))
        timings['source_verification'] += time.monotonic() - before_read
        cache_key = sha + '-' + provenance['decrypt_function_sha256']
        entry = root / cache_key
        wait_start = time.monotonic()
        with _lock(root / ('.' + cache_key + '.lock')):
            timings['lock_wait'] += time.monotonic() - wait_start
            if _version(source) != version:
                raise ValueError('Encrypted source changed while waiting for its cache lock.')
            if entry.exists() or entry.is_symlink():
                check_start = time.monotonic()
                manifest_path = entry / MANIFEST_NAME
                try:
                    if entry.is_symlink() or not entry.is_dir() or entry.stat().st_mode & 0o077:
                        raise ValueError('Published asset cache directory is unsafe.')
                    _private_file(manifest_path)
                    encoded = manifest_path.read_bytes()
                    manifest = json.loads(encoded)
                    row = _row(manifest, entry, hashlib.sha256(encoded).hexdigest(), cache_hit=True, asset=asset)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise ValueError('Published asset cache is incomplete or corrupt: ' + str(entry)) from error
                if (row['source_version'] != version or row['encrypted_md5'] != md5
                        or row['encrypted_file'] != str(source)
                        or row['sdk_asset_utils_sha256'] != provenance['sdk_asset_utils_sha256']):
                    raise ValueError('Published asset cache source/provenance association differs.')
                _validated_metadata(row, root)
                timings['cache_validation'] += time.monotonic() - check_start
                rows.append(row)
                continue
            temporary = Path(tempfile.mkdtemp(prefix='.pending-', dir=root))
            try:
                content = temporary / 'content'
                content.mkdir(mode=0o700)
                decrypted = temporary / 'decrypted.usdz'
                _write_private(decrypted, b'')
                decrypt_start = time.monotonic()
                try:
                    decrypt(str(source), str(decrypted))
                except InvalidToken as error:
                    raise ValueError('Cannot decrypt asset: the key does not match or the encrypted asset is corrupt: '
                                     + str(source)) from error
                timings['decryption'] += time.monotonic() - decrypt_start
                extract_start = time.monotonic()
                try:
                    with zipfile.ZipFile(decrypted) as archive:
                        members, root_usd, expanded_bytes = _inspect_archive(archive)
                        _reserve_space(root, temporary, expanded_bytes, min_free_bytes)
                        outputs = _extract(archive, members, content)
                except (zipfile.BadZipFile, RuntimeError) as error:
                    raise ValueError('Decrypted asset is not a valid supported ZIP archive: ' + str(source)) from error
                decrypted.unlink()
                timings['extraction'] += time.monotonic() - extract_start
                verify_start = time.monotonic()
                after_sha, after_md5, after_version = _hashes(source, source=True)
                if (after_sha, after_md5, after_version) != (sha, md5, version):
                    raise ValueError('Encrypted source changed during asset preparation: ' + str(source))
                if _version(key_path) != key_before:
                    raise ValueError('OmniGibson decryption key changed during asset preparation.')
                timings['source_verification'] += time.monotonic() - verify_start
                manifest = {'schema': 'capture_prepared_asset_v1', **provenance,
                            'encrypted_file': str(source), 'encrypted_sha256': sha, 'encrypted_md5': md5,
                            'source_version': version, 'root_usd': root_usd, 'members': outputs,
                            'total_bytes': expanded_bytes,
                            'file_count': sum(item['type'] == 'file' for item in outputs),
                            'preparation_worker_pid': os.getpid(), 'mdl_paths_modified': False}
                encoded = _json_bytes(manifest)
                _write_private(temporary / MANIFEST_NAME, encoded)
                with _lock(root / '.space.lock'):
                    (temporary / 'reservation.json').unlink()
                    os.rename(temporary, entry)
                rows.append(_row(manifest, entry, hashlib.sha256(encoded).hexdigest(),
                                 cache_hit=False, asset=asset))
            finally:
                # Only this call's unpublished work is eligible for removal.
                with _lock(root / '.space.lock'):
                    if temporary.exists():
                        shutil.rmtree(temporary)
    timings['total'] = time.monotonic() - started
    return {'schema': 'capture_prepared_assets_v1', 'cache_root': str(root), **provenance,
            'assets': rows, 'prepared_asset_count': len(rows),
            'cache_hit_count': sum(row['cache_hit'] for row in rows),
            'unavailable_asset_count': unavailable,
            'total_prepared_bytes': sum(row['total_bytes'] for row in rows),
            'timings_s': timings, 'worker_pid': os.getpid(),
            'max_encrypted_bytes': MAX_ENCRYPTED_BYTES, 'min_free_bytes': min_free_bytes}

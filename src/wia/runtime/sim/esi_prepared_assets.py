"""Reuse CPU-unpacked capture assets while retaining the installed SDK methods.

Run: python -m pytest -q tests/test_prepared_assets.py
Use prepared_asset_loading around native scene construction on the Kit thread.
Only decryption and unpacking are reused. MDL normalization, USD composition,
scene construction and physics remain in the original native method bodies.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import copy
from functools import wraps
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import textwrap
import warnings


_ENCRYPTED_PREFIX = '''
encrypted_filename = self._usd_path.replace(".usdz", ".usdz.encrypted")
self.check_hash(encrypted_filename)
basename = os.path.basename(self._usd_path)
tempdir_path = tempfile.mkdtemp(basename, dir=og.tempdir)
usdz_path = os.path.join(tempdir_path, f"{basename}.usdz")
decrypt_file(encrypted_filename, usdz_path)
with zipfile.ZipFile(usdz_path, "r") as zip_ref:
    zip_ref.extractall(tempdir_path)
os.unlink(usdz_path)
usd_files = list(glob.glob(os.path.join(tempdir_path, "*.usd")))
assert len(usd_files) == 1, f"Expected exactly one USD file in {tempdir_path}, found {usd_files}"
usd_path = usd_files[0]
'''
_SIDE_STAGE = 'side_stage = lazy.pxr.Usd.Stage.Open(usd_path)'
_REPLACEMENT = '''
self.check_hash(self._usd_path.replace(".usdz", ".usdz.encrypted"))
usd_path = _consume_prepared_asset(self)
'''
_ASSOCIATION_FIELDS = (
    'encrypted_file', 'encrypted_sha256', 'encrypted_md5', 'source_version',
    'prepared_dir', 'usd_file', 'cache_manifest', 'cache_manifest_sha256',
    'sdk_asset_utils_sha256', 'decrypt_function_sha256', 'total_bytes', 'file_count',
)


class _PreparedAssetUnavailable(Exception):
    """Request original loading before any retained native operation begins."""


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_version(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('Encrypted source is not a regular file.')
    return {'path': str(path), 'device': info.st_dev, 'inode': info.st_ino,
            'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'ctime_ns': info.st_ctime_ns}


def _method_source(method):
    return textwrap.dedent(inspect.getsource(method))


def _transformed_method(original, name, namespace, consume):
    """Replace only the checked, known SDK decrypt/unpack prefix."""
    source = _method_source(original)
    syntax = ast.parse(source)
    if len(syntax.body) != 1 or not isinstance(syntax.body[0], ast.FunctionDef):
        raise ValueError('Expected one synchronous SDK method definition.')
    method = copy.deepcopy(syntax.body[0])
    expected_args = ['self', 'stage'] if name == 'prebuild' else ['self']
    args = method.args
    if (method.name != name or method.decorator_list or args.posonlyargs
            or args.vararg or args.kwarg or args.kwonlyargs or args.defaults
            or args.kw_defaults or [arg.arg for arg in args.args] != expected_args):
        raise ValueError('Unsupported SDK method signature or decorators.')
    test = ast.dump(ast.parse('self._encrypted', mode='eval').body)
    encrypted = [node for node in method.body if isinstance(node, ast.If)
                 and ast.dump(node.test) == test]
    if len(encrypted) != 1:
        raise ValueError('Expected one top-level if self._encrypted block.')
    block = encrypted[0]
    expected = ast.parse(_ENCRYPTED_PREFIX).body
    prefix = block.body[:len(expected)]
    if ([ast.dump(node) for node in prefix] != [ast.dump(node) for node in expected]
            or len(block.body) <= len(expected)
            or ast.dump(block.body[len(expected)]) != ast.dump(ast.parse(_SIDE_STAGE).body[0])):
        raise ValueError('Unsupported SDK encrypted decrypt/unpack prefix.')
    # The retained MDL and native stage bodies are copied from the installed SDK,
    # including the original unencrypted branch and all statements after the if.
    block.body = ast.parse(_REPLACEMENT).body + block.body[len(expected):]
    ast.fix_missing_locations(syntax := ast.Module(body=[method], type_ignores=[]))
    globals_copy = dict(namespace)
    globals_copy['_consume_prepared_asset'] = consume
    filename = inspect.getsourcefile(original) or '<installed-usd-object>'
    exec(compile(syntax, filename, 'exec'), globals_copy)
    return globals_copy[name]


def _working_copy(row, native_root):
    """Copy mutable root USD and link assets read only by the retained SDK tail."""
    prepared = Path(row['prepared_dir'])
    root_usd = Path(row['usd_file'])
    manifest = Path(row['cache_manifest'])
    if prepared.is_symlink() or not prepared.is_dir():
        raise ValueError('Prepared asset directory is missing or unsafe.')
    relative_usd = root_usd.relative_to(prepared)
    if relative_usd.parent != Path('.') or root_usd.suffix != '.usd':
        raise ValueError('Prepared asset does not identify a root *.usd.')
    native_root.mkdir(parents=True, exist_ok=True)
    working = Path(tempfile.mkdtemp(prefix='capture-prepared-', dir=native_root))
    for directory, directories, files in os.walk(prepared, followlinks=False):
        source_directory = Path(directory)
        destination = working / source_directory.relative_to(prepared)
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in directories:
            source = source_directory / name
            if source.is_symlink() or not source.is_dir():
                raise ValueError('Prepared asset contains an unsafe directory.')
            (destination / name).mkdir(mode=0o700, exist_ok=True)
        for name in files:
            source, target = source_directory / name, destination / name
            if source == manifest:
                continue
            if not stat.S_ISREG(source.lstat().st_mode):
                raise ValueError('Prepared asset contains a nonregular file.')
            if source == root_usd:
                shutil.copy2(source, target)
            else:
                # These texture/material files are only read by the retained
                # native path. Do not chmod a link: it would alter the cache.
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
    result = working / relative_usd
    if (result.stat().st_dev, result.stat().st_ino) == (root_usd.stat().st_dev, root_usd.stat().st_ino):
        raise ValueError('Native root USD must have a different inode from the cache.')
    return str(result)


@contextmanager
def prepared_asset_loading(preparation, output, *, usd_module=None, native_temp_root=None):
    """Temporarily consume validated CPU assets without changing SDK scene semantics.

    The context must wrap serial native scene construction on the Kit thread.
    Scratch copies remain under the worker's native temp root for its normal
    lifecycle cleanup; published CPU cache files are never modified or deleted.
    An injected usd_module avoids importing OmniGibson in CPU-only tests.
    """
    stats = {
        'schema': 'native_asset_preparation_v1', 'enabled': False,
        'prepared_unique_assets_used': 0, 'prepared_object_loads': 0,
        'prepared_object_prebuilds': 0, 'native_fallback_calls': 0,
        'warnings': [], 'sdk_source_sha256': None, 'sdk_method_sha256': {},
        'cache_root': preparation.get('cache_root') if isinstance(preparation, dict) else None,
        'native_construction_parallel': False, 'robot_task_qualification': False,
        'scope': ('Capture-only reuse of CPU decryption and unpacking; native MDL '
                  'normalization, stage composition, scene construction and physics remain serial.'),
        'helper_provenance': {'native_consumer_path': str(Path(__file__).resolve()),
                              'native_consumer_sha256': _sha256(__file__)},
    }
    original_methods, usd_class = {}, None
    seen_warnings, rows, invalid, validated, working, used = set(), {}, {}, {}, {}, set()

    def warn(message):
        if message not in seen_warnings:
            seen_warnings.add(message)
            stats['warnings'].append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def source_for(obj):
        return Path(obj._usd_path.replace('.usdz', '.usdz.encrypted')).resolve()

    def screen(row):
        if _source_version(Path(row['encrypted_file'])) != row['source_version']:
            raise ValueError('Encrypted source version changed since CPU preparation.')

    def consume(obj, name):
        source = str(source_for(obj))
        row = validated[source]
        # The SDK gives every object its own asset layer/reference path. Keep
        # that isolation even when CPU decryption is shared by identical models.
        # Scene.prebuild runs before the object is assigned to a live scene;
        # prim_path is unavailable there. The original SDK uses this relative
        # path for the same per-object reference identity.
        identity = (source, obj._relative_prim_path)
        try:
            screen(row)
            if identity not in working:
                if native_temp_root is None:
                    raise ValueError('Native temporary root is unavailable.')
                working[identity] = _working_copy(row, Path(native_temp_root))
            elif not Path(working[identity]).is_file():
                raise ValueError('Native prepared working copy is missing.')
            screen(row)
        except (OSError, ValueError, KeyError, TypeError) as error:
            invalid[source] = str(error)
            raise _PreparedAssetUnavailable(str(error)) from error
        if source not in used:
            used.add(source)
            stats['prepared_unique_assets_used'] = len(used)
        key = 'prepared_object_prebuilds' if name == 'prebuild' else 'prepared_object_loads'
        stats[key] += 1
        return working[identity]

    def wrap(name, original, transformed, validator):
        @wraps(original)
        def method(obj, *args, **kwargs):
            reason, source = None, None
            if transformed is None:
                reason = 'Unsupported installed SDK method shape.'
            elif not getattr(obj, '_encrypted', False):
                reason = 'Object uses the native unencrypted asset path.'
            else:
                try:
                    source = str(source_for(obj))
                    row = rows.get(source)
                    if row is None:
                        reason = 'No matching CPU-prepared encrypted asset.'
                    else:
                        screen(row)
                        if source in invalid:
                            reason = invalid[source]
                        elif source not in validated:
                            validator(row, cache_root=stats['cache_root'])
                            screen(row)
                            validated[source] = row
                except (OSError, ValueError, KeyError, TypeError) as error:
                    reason = str(error)
                    if source is not None:
                        invalid[source] = reason
            if reason is None:
                try:
                    return transformed(obj, *args, **kwargs)
                except _PreparedAssetUnavailable as error:
                    reason = str(error)
            stats['native_fallback_calls'] += 1
            suffix = ' Source: ' + source if source is not None else ''
            warn(name + ' uses original native loading: ' + reason + suffix)
            return original(obj, *args, **kwargs)
        return method

    try:
        if usd_module is None:
            # Import the SDK only inside the real native context, never at module
            # import or through the CPU preparation path.
            usd_module = importlib.import_module('omnigibson.objects.usd_object')
            if native_temp_root is None:
                native_temp_root = importlib.import_module('omnigibson').tempdir
        elif native_temp_root is None:
            native_temp_root = getattr(getattr(usd_module, 'og', None), 'tempdir', None)
        usd_class = usd_module.USDObject
        cpu_module = importlib.import_module('wia.runtime.scene_capture_assets')
        validator = cpu_module.validate_prepared_asset
        helper_file = inspect.getsourcefile(validator)
        stats['helper_provenance'].update({
            'validation_helper_path': helper_file,
            'validation_helper_sha256': _sha256(helper_file) if helper_file else None,
            'sdk_asset_utils_sha256': preparation.get('sdk_asset_utils_sha256')
            if isinstance(preparation, dict) else None,
            'decrypt_function_sha256': preparation.get('decrypt_function_sha256')
            if isinstance(preparation, dict) else None,
        })
        try:
            if not isinstance(preparation, dict) or preparation.get('schema') != 'capture_prepared_assets_v1':
                raise ValueError('Unsupported CPU asset preparation schema.')
            root = Path(preparation['cache_root'])
            if not root.is_absolute() or str(root.resolve()) != str(root):
                raise ValueError('CPU asset cache root must be canonical and absolute.')
            for candidate in preparation['assets']:
                row = copy.deepcopy(candidate)
                source = row['encrypted_file']
                if not Path(source).is_absolute() or str(Path(source).resolve()) != source:
                    raise ValueError('Prepared encrypted source must be canonical and absolute.')
                for field in ('sdk_asset_utils_sha256', 'decrypt_function_sha256'):
                    if row[field] != preparation[field]:
                        raise ValueError('Prepared asset helper provenance differs from its preparation.')
                if source in rows and any(rows[source].get(field) != row.get(field)
                                          for field in _ASSOCIATION_FIELDS):
                    invalid[source] = 'Conflicting prepared asset rows for the encrypted source.'
                    warn(invalid[source] + ' Source: ' + source)
                rows[source] = row
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.clear()
            warn('CPU-prepared assets are unavailable: ' + str(error))
        for name in ('_load', 'prebuild'):
            original = getattr(usd_class, name)
            original_methods[name] = original
            try:
                source_file = inspect.getsourcefile(original)
                if source_file and stats['sdk_source_sha256'] is None:
                    stats['sdk_source_path'] = source_file
                    stats['sdk_source_sha256'] = _sha256(source_file)
                stats['sdk_method_sha256'][name] = hashlib.sha256(
                    _method_source(original).encode('utf-8')).hexdigest()
                transformed = _transformed_method(
                    original, name, vars(usd_module),
                    lambda obj, method_name=name: consume(obj, method_name))
            except (OSError, ValueError, TypeError, SyntaxError) as error:
                transformed = None
                warn(name + ' cannot reuse CPU assets: ' + str(error))
            setattr(usd_class, name, wrap(name, original, transformed, validator))
            stats['enabled'] = stats['enabled'] or bool(rows and transformed is not None)
        yield stats
    except BaseException as error:
        stats['context_error_type'] = type(error).__name__
        raise
    finally:
        if usd_class is not None:
            for name, original in original_methods.items():
                setattr(usd_class, name, original)
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / 'native_asset_preparation.json').write_text(
            json.dumps(stats, indent=2, sort_keys=True) + '\n')

"""Check CPU asset consumption without importing the simulator or GPU libraries.

Run: python -m pytest -q tests/test_prepared_assets.py
The synthetic SDK methods retain the installed decrypt/unpack and native tails.
"""
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import pytest

from wia.runtime.sim.esi_prepared_assets import prepared_asset_loading


_SDK = '''import glob
import hashlib
import os
import tempfile
import zipfile

class USDObject:
    def __init__(self, usd_path, *, encrypted=True, expected_file_hash=None):
        self._usd_path = str(usd_path)
        self._encrypted = encrypted
        self._expected_file_hash = expected_file_hash
        self._init_info = {"args": {}}
        self._relative_prim_path = "/scene/object"
        self.prim_path = "/World/scene/object"
        self.scale = [0.5, 0.5, 0.5]
        self.collision = True

    def check_hash(self, usd_path):
        events.append(("check_hash", usd_path))
        file_hash = hashlib.md5(open(usd_path, "rb").read()).hexdigest()
        if self._expected_file_hash is not None:
            if file_hash != self._expected_file_hash:
                events.append(("hash_warning", self._expected_file_hash, file_hash))
        else:
            self._expected_file_hash = file_hash
            self._init_info["args"]["expected_file_hash"] = file_hash

    def prebuild(self, stage):
        usd_path = self._usd_path
        if self._encrypted:
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
            side_stage = lazy.pxr.Usd.Stage.Open(usd_path)
            def _update_path(asset_path):
                if asset_path.endswith(".mdl"):
                    return os.path.basename(asset_path)
                return asset_path
            lazy.pxr.UsdUtils.ModifyAssetPaths(side_stage.GetRootLayer(), _update_path)
            side_stage.Save()
            del side_stage
        else:
            self.check_hash(usd_path)
        prim_path = "/World" + self._relative_prim_path
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            prim = stage.DefinePrim(prim_path, "Xform")
        assert prim.GetReferences().AddReference(usd_path)

    def _load(self):
        usd_path = self._usd_path
        if self._encrypted:
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
            side_stage = lazy.pxr.Usd.Stage.Open(usd_path)
            def _update_path(asset_path):
                if asset_path.endswith(".mdl"):
                    return os.path.basename(asset_path)
                return asset_path
            lazy.pxr.UsdUtils.ModifyAssetPaths(side_stage.GetRootLayer(), _update_path)
            side_stage.Save()
            del side_stage
        else:
            self.check_hash(usd_path)
        prim = add_asset_to_stage(asset_path=usd_path, prim_path=self.prim_path)
        return prim
'''


def _version(path):
    info = path.stat()
    return {'path': str(path), 'device': info.st_dev, 'inode': info.st_ino,
            'size': info.st_size, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}


@pytest.fixture
def native(tmp_path, monkeypatch):
    events, validations = [], []
    content = {'object.usd': b'0/material.mdl\ntextures/image.png\n',
               'textures/image.png': b'texture bytes', 'material.mdl': b'material bytes'}
    source = tmp_path / 'object.usdz.encrypted'
    with zipfile.ZipFile(source, 'w') as archive:
        for name, data in content.items():
            archive.writestr(name, data)
    cache_root = tmp_path / 'cache'
    prepared = cache_root / 'entry' / 'content'
    for name, data in content.items():
        target = prepared / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = prepared.parent / 'cache_manifest.json'
    manifest.write_text('{}\n')
    row = {'encrypted_file': str(source),
           'encrypted_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'encrypted_md5': hashlib.md5(source.read_bytes()).hexdigest(),
           'source_version': _version(source), 'prepared_dir': str(prepared),
           'usd_file': str(prepared / 'object.usd'), 'cache_manifest': str(manifest),
           'cache_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'sdk_asset_utils_sha256': 'a' * 64, 'decrypt_function_sha256': 'b' * 64,
           'total_bytes': sum(map(len, content.values())), 'file_count': len(content),
           'cache_hit': False}
    preparation = {'schema': 'capture_prepared_assets_v1', 'cache_root': str(cache_root),
                   'sdk_asset_utils_sha256': 'a' * 64, 'decrypt_function_sha256': 'b' * 64,
                   'assets': [row]}

    def validate(row, *, cache_root):
        validations.append(row['encrypted_file'])
        assert Path(cache_root) == prepared.parents[1]
        if _version(source) != row['source_version']:
            raise ValueError('Encrypted source version changed.')
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['encrypted_sha256']:
            raise ValueError('Encrypted source hash changed.')
        if hashlib.sha256(Path(row['cache_manifest']).read_bytes()).hexdigest() != row['cache_manifest_sha256']:
            raise ValueError('Cache manifest seal mismatch.')
        for name, data in content.items():
            if (prepared / name).read_bytes() != data:
                raise ValueError('Prepared file hash mismatch: ' + name)
        return {**row, 'validated': True}

    monkeypatch.setitem(sys.modules, 'wia.runtime.scene_capture_assets',
                        types.SimpleNamespace(validate_prepared_asset=validate))
    sdk_path = tmp_path / 'synthetic_sdk.py'
    sdk_path.write_text(_SDK)
    module = types.ModuleType('synthetic_sdk')
    module.__file__ = str(sdk_path)
    exec(compile(_SDK, str(sdk_path), 'exec'), module.__dict__)
    module.events = events
    scratch = tmp_path / 'native'
    scratch.mkdir()
    module.og = types.SimpleNamespace(tempdir=str(scratch))

    def decrypt(source, destination):
        events.append(('decrypt', source))
        Path(destination).write_bytes(Path(source).read_bytes())

    class SideStage:
        def __init__(self, path):
            self.path = Path(path)
            self.text = self.path.read_text()

        @classmethod
        def Open(cls, path):
            events.append(('open', path))
            return cls(path)

        def GetRootLayer(self):
            return self

        def Save(self):
            events.append(('save', str(self.path)))
            self.path.write_text(self.text)

    def modify(layer, callback):
        events.append(('modify', callback('0/material.mdl'), callback('textures/image.png')))
        layer.text = '\n'.join(callback(line) for line in layer.text.splitlines()) + '\n'

    def add_asset_to_stage(*, asset_path, prim_path):
        events.append(('add_asset', asset_path, prim_path))
        return (asset_path, prim_path)

    module.decrypt_file = decrypt
    module.lazy = types.SimpleNamespace(pxr=types.SimpleNamespace(
        Usd=types.SimpleNamespace(Stage=SideStage),
        UsdUtils=types.SimpleNamespace(ModifyAssetPaths=modify)))
    module.add_asset_to_stage = add_asset_to_stage

    class Prim:
        def __init__(self, valid):
            self.valid = valid

        def IsValid(self):
            return self.valid

        def GetReferences(self):
            return self

        def AddReference(self, path):
            events.append(('reference', path))
            return True

    class Stage:
        def GetPrimAtPath(self, path):
            events.append(('get_prim', path))
            return Prim(False)

        def DefinePrim(self, path, kind):
            events.append(('define_prim', path, kind))
            return Prim(True)

    return types.SimpleNamespace(
        sdk=module, preparation=preparation, row=row, source=source, prepared=prepared,
        events=events, validations=validations, stage=Stage(), scratch=scratch,
        output=tmp_path / 'output', usd_path=tmp_path / 'object.usdz')


def _context(native):
    return prepared_asset_loading(native.preparation, native.output,
                                  usd_module=native.sdk, native_temp_root=native.scratch)


def test_both_native_methods_preserve_hash_and_native_tails(native):
    cls = native.sdk.USDObject
    original_load, original_prebuild = cls._load, cls.prebuild
    original_source = native.source.read_bytes()
    original_usd = (native.prepared / 'object.usd').read_bytes()
    first, second = cls(native.usd_path), cls(native.usd_path)
    before = (first._usd_path, first._encrypted, first.scale[:], first.collision)
    with _context(native) as stats:
        first.prebuild(native.stage)
        result = second._load()
        working_usd = Path(result[0])
        assert working_usd.parent.parent == native.scratch
        assert working_usd.stat().st_ino != (native.prepared / 'object.usd').stat().st_ino
        assert (working_usd.parent / 'textures/image.png').stat().st_ino == (
            native.prepared / 'textures/image.png').stat().st_ino
        assert working_usd.read_text() == 'material.mdl\ntextures/image.png\n'
        assert native.events[0] == ('check_hash', str(native.source))
        assert native.events.count(('modify', 'material.mdl', 'textures/image.png')) == 2
        assert ('reference', result[0]) in native.events
        assert ('define_prim', '/World/scene/object', 'Xform') in native.events
        assert result[1] == second.prim_path
        assert first._expected_file_hash == second._expected_file_hash == native.row['encrypted_md5']
        assert first._init_info['args']['expected_file_hash'] == native.row['encrypted_md5']
        assert second._init_info['args']['expected_file_hash'] == native.row['encrypted_md5']
        assert stats['prepared_unique_assets_used'] == 1
        assert stats['prepared_object_prebuilds'] == stats['prepared_object_loads'] == 1
        assert stats['native_fallback_calls'] == 0
        assert stats['enabled'] is True
        assert stats['native_construction_parallel'] is False
        assert stats['robot_task_qualification'] is False
    assert native.validations == [str(native.source)]
    assert not any(event[0] == 'decrypt' for event in native.events)
    assert cls._load is original_load and cls.prebuild is original_prebuild
    assert (first._usd_path, first._encrypted, first.scale, first.collision) == before
    assert native.source.read_bytes() == original_source
    assert (native.prepared / 'object.usd').read_bytes() == original_usd
    diagnostics = json.loads((native.output / 'native_asset_preparation.json').read_text())
    assert diagnostics == stats
    assert len(stats['sdk_source_sha256']) == 64
    assert set(stats['sdk_method_sha256']) == {'_load', 'prebuild'}
    assert len(stats['helper_provenance']['native_consumer_sha256']) == 64
    assert len(stats['helper_provenance']['validation_helper_sha256']) == 64


def test_expected_hash_mismatch_keeps_original_warning_semantics(native):
    obj = native.sdk.USDObject(native.usd_path, expected_file_hash='old_saved_hash')
    with _context(native):
        obj._load()
    assert ('hash_warning', 'old_saved_hash', native.row['encrypted_md5']) in native.events
    assert obj._expected_file_hash == 'old_saved_hash'
    assert obj._init_info['args'] == {}


def test_duplicate_models_keep_distinct_native_object_reference_paths(native):
    first = native.sdk.USDObject(native.usd_path)
    second = native.sdk.USDObject(native.usd_path)
    second.prim_path = '/World/scene/other_object'
    second._relative_prim_path = '/scene/other_object'
    with _context(native) as stats:
        first_path = first._load()[0]
        second_path = second._load()[0]
    assert first_path != second_path
    assert Path(first_path).read_bytes() == Path(second_path).read_bytes()
    assert stats['prepared_unique_assets_used'] == 1 and stats['prepared_object_loads'] == 2


def test_prebuild_does_not_require_a_live_scene_assignment(native):
    class BeforeScene(native.sdk.USDObject):
        @property
        def prim_path(self):
            raise AssertionError('Scene has not been assigned to this prim yet!')

        @prim_path.setter
        def prim_path(self, value):
            self.saved_world_path = value

    obj = BeforeScene(native.usd_path)
    with _context(native) as stats:
        obj.prebuild(native.stage)
    assert stats['prepared_object_prebuilds'] == 1


def test_duplicate_preparation_metadata_reuses_one_validated_copy(native):
    native.preparation['assets'].append({**native.row, 'cache_hit': True,
                                         'category': 'another scene instance'})
    with _context(native) as stats:
        first = native.sdk.USDObject(native.usd_path)._load()
        second = native.sdk.USDObject(native.usd_path)._load()
    assert first == second
    assert native.validations == [str(native.source)]
    assert stats['prepared_unique_assets_used'] == 1
    assert stats['prepared_object_loads'] == 2
    assert stats['native_fallback_calls'] == 0


@pytest.mark.parametrize('case', ['missing', 'source_changed', 'cache_corrupt', 'manifest_corrupt'])
def test_invalid_preparation_uses_original_loader_and_warns(native, case):
    if case == 'missing':
        native.preparation['assets'] = []
    elif case == 'source_changed':
        with zipfile.ZipFile(native.source, 'a') as archive:
            archive.writestr('new.txt', b'changed source')
    elif case == 'cache_corrupt':
        (native.prepared / 'object.usd').write_text('corrupt')
    else:
        Path(native.row['cache_manifest']).write_text('{"corrupt": true}')
    with pytest.warns(RuntimeWarning, match='original native loading'):
        with _context(native) as stats:
            result = native.sdk.USDObject(native.usd_path)._load()
    assert Path(result[0]).is_file()
    assert any(event[0] == 'decrypt' for event in native.events)
    assert stats['prepared_unique_assets_used'] == stats['prepared_object_loads'] == 0
    assert stats['native_fallback_calls'] == 1
    assert stats['warnings']


def test_source_is_screened_again_after_first_asset_use(native):
    with _context(native) as stats:
        first = native.sdk.USDObject(native.usd_path)._load()
        with zipfile.ZipFile(native.source, 'a') as archive:
            archive.writestr('later.txt', b'changed later')
        with pytest.warns(RuntimeWarning, match='source version changed'):
            second = native.sdk.USDObject(native.usd_path)._load()
    assert first[0] != second[0]
    assert native.validations == [str(native.source)]
    assert stats['prepared_object_loads'] == 1
    assert stats['native_fallback_calls'] == 1


def test_unsupported_method_shape_falls_back_without_approximation(native):
    source = _SDK.replace('basename = os.path.basename(self._usd_path)',
                          'basename = os.path.basename(self._usd_path) + "different"')
    sdk_path = Path(native.sdk.__file__)
    sdk_path.write_text(source)
    exec(compile(source, str(sdk_path), 'exec'), native.sdk.__dict__)
    original_load = native.sdk.USDObject._load
    with pytest.warns(RuntimeWarning) as caught:
        with _context(native) as stats:
            native.sdk.USDObject(native.usd_path)._load()
    assert any('Unsupported SDK encrypted' in str(item.message) for item in caught)
    assert stats['enabled'] is False
    assert stats['native_fallback_calls'] == 1
    assert stats['prepared_object_loads'] == 0
    assert native.sdk.USDObject._load is original_load
    assert not native.validations
    assert any(event[0] == 'decrypt' for event in native.events)


def test_context_restores_methods_and_writes_diagnostics_after_exception(native):
    original_load, original_prebuild = native.sdk.USDObject._load, native.sdk.USDObject.prebuild
    with pytest.raises(RuntimeError, match='native construction failed'):
        with _context(native) as stats:
            native.sdk.USDObject(native.usd_path)._load()
            raise RuntimeError('native construction failed')
    assert native.sdk.USDObject._load is original_load
    assert native.sdk.USDObject.prebuild is original_prebuild
    assert stats['context_error_type'] == 'RuntimeError'
    assert json.loads((native.output / 'native_asset_preparation.json').read_text()) == stats
    assert list(native.scratch.iterdir())


def test_native_tail_exception_propagates_without_fallback(native):
    def fail(*, asset_path, prim_path):
        raise RuntimeError('native stage failed')

    native.sdk.add_asset_to_stage = fail
    with pytest.raises(RuntimeError, match='native stage failed'):
        with _context(native) as stats:
            native.sdk.USDObject(native.usd_path)._load()
    assert stats['native_fallback_calls'] == 0
    assert stats['prepared_object_loads'] == 1
    assert not any(event[0] == 'decrypt' for event in native.events)


def test_unencrypted_objects_retain_original_path(native):
    path = native.source.parent / 'plain.usd'
    path.write_text('plain USD')
    with pytest.warns(RuntimeWarning, match='unencrypted asset path'):
        with _context(native) as stats:
            result = native.sdk.USDObject(path, encrypted=False)._load()
    assert result[0] == str(path)
    assert ('check_hash', str(path)) in native.events
    assert stats['native_fallback_calls'] == 1
    assert not native.validations


def test_cache_manifest_inside_content_is_excluded_from_native_copy(native):
    manifest = native.prepared / 'cache_manifest.json'
    manifest.write_text('{}\n')
    native.row['cache_manifest'] = str(manifest)
    with _context(native):
        result = native.sdk.USDObject(native.usd_path)._load()
    assert not (Path(result[0]).parent / 'cache_manifest.json').exists()
    assert manifest.exists()


def test_no_sdk_or_gpu_library_is_imported_in_injected_context(native):
    before = set(sys.modules)
    with _context(native):
        native.sdk.USDObject(native.usd_path)._load()
    loaded = set(sys.modules) - before
    assert not any(name.split('.')[0] in {'omnigibson', 'torch', 'pxr', 'cuda'} for name in loaded)

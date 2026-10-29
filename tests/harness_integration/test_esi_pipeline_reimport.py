"""Check persistent ESI pipeline reuse without native or model imports.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider
tests/test_esi_pipeline_reimport.py
All pipeline and namespace modules are temporary CPU fixtures.
"""
from pathlib import Path
import sys
from types import ModuleType

import pytest

import wia.runtime.sim.esi_omnigibson as native


@pytest.fixture
def modules(tmp_path, monkeypatch):
    root=tmp_path/'esi';active=root/'src/active_explore';active.mkdir(parents=True)
    pipeline_file=active/'pipeline.py'
    pipeline_file.write_text("raise AssertionError('A native or model import must not execute in this test.')\n")
    for name in ('active_explore.pipeline','active_explore','tasks','models','utils'):
        monkeypatch.delitem(sys.modules,name,raising=False)
    def module(name,path,*,package=False):
        value=ModuleType(name);value.__file__=str(path)
        if package:value.__path__=[str(Path(path).parent)]
        monkeypatch.setitem(sys.modules,name,value)
        return value
    def forbidden(name):raise AssertionError('No import should run: '+name)
    monkeypatch.setattr(native.importlib,'import_module',forbidden)
    return root,pipeline_file,module


def test_verified_cached_pipeline_reuses_classes_despite_kit_models_namespace(modules):
    root,pipeline_file,module=modules
    cached=module('active_explore.pipeline',pipeline_file)
    cached.GPTModel=object();cached.GeminiModel=object()
    module('active_explore',pipeline_file.parent/'__init__.py',package=True)
    module('tasks',root/'src/tasks/__init__.py',package=True)
    module('utils',root/'src/utils/__init__.py',package=True)
    foreign=module('models',root.parent/'kit_extension/models/__init__.py',package=True)
    before=dict(sys.modules);paths=list(sys.path)
    assert native._import_pipeline(root) is cached
    assert native._import_pipeline(root/'.') is cached
    assert sys.modules['models'] is foreign
    assert dict(sys.modules)==before and sys.path==paths


@pytest.mark.parametrize('namespace',['active_explore','tasks','models','utils'])
def test_cold_import_preserves_all_foreign_namespace_rejections(modules,namespace):
    root,_,module=modules
    foreign=module(namespace,root.parent/'foreign'/namespace/'__init__.py',package=True)
    with pytest.raises(RuntimeError,match='already imported module'):
        native._import_pipeline(root)
    assert sys.modules[namespace] is foreign
    assert 'active_explore.pipeline' not in sys.modules


@pytest.mark.parametrize('filename',['foreign','missing','wrong_file_same_root'])
def test_foreign_or_unverifiable_cached_pipeline_is_never_reused(modules,filename):
    root,pipeline_file,module=modules
    path={'foreign':root.parent/'another_esi/src/active_explore/pipeline.py',
          'missing':pipeline_file,'wrong_file_same_root':pipeline_file.parent/'other.py'}[filename]
    cached=module('active_explore.pipeline',path)
    if filename=='missing':del cached.__file__
    with pytest.raises(RuntimeError,match='Cached ESI pipeline'):
        native._import_pipeline(root)
    assert sys.modules['active_explore.pipeline'] is cached


@pytest.mark.parametrize('namespace',['active_explore','tasks','utils'])
def test_cached_repeated_calls_still_reject_foreign_world_namespaces(modules,namespace):
    root,pipeline_file,module=modules
    cached=module('active_explore.pipeline',pipeline_file)
    assert native._import_pipeline(root) is cached
    foreign=module(namespace,root.parent/'kit_extension'/namespace/'__init__.py',package=True)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='already imported module'):
            native._import_pipeline(root)
    assert sys.modules[namespace] is foreign
    assert sys.modules['active_explore.pipeline'] is cached


def test_normalized_relative_root_and_cold_import_use_requested_pipeline_only(modules,monkeypatch):
    root,pipeline_file,_=modules
    monkeypatch.chdir(root.parent)
    imported=[];cached=ModuleType('active_explore.pipeline');cached.__file__=str(pipeline_file)
    monkeypatch.setattr(native.importlib,'import_module',lambda name:imported.append(name) or cached)
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setenv('OMNIGIBSON_HEADLESS','0')
    monkeypatch.delenv('OG_DISABLE_EMITTER_APIS',raising=False)
    assert native._import_pipeline(Path('esi/../esi')) is cached
    assert imported==['active_explore.pipeline']
    assert str(root/'src') in sys.path and str(pipeline_file.parent) in sys.path
    assert native.os.environ['OMNIGIBSON_HEADLESS']=='0'
    assert native.os.environ['OG_DISABLE_EMITTER_APIS']=='1'

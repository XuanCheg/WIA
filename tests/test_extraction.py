"""Validate standalone imports and historical aliases. Run: python -m pytest tests/test_extraction.py."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from wia.resources import ROOT
from wia.tasks.loader import UniqueLoader, discover_scenes, discover_tasks, task_definition, load_task, load_scene
from wia.tasks.schema import SceneDefinition


def test_all_task_and_executable_scene_definitions_validate():
    tasks = discover_tasks()
    assert tasks
    for path in (ROOT / 'explore_to_achieve/wia_v1/release').rglob('*.yaml'):
        value = yaml.load(path.read_text(), Loader=UniqueLoader)
        if not isinstance(value, dict):
            continue
        if value.get('kind') in ('category_robot_task', 'rigid_containment_task', 'robot_task'):
            task_definition(value)
        # scene_import records are inventory entries rather than executable definitions.
        elif value.get('kind') == 'scene_patch':
            SceneDefinition.model_validate(value)


def test_standalone_imports_block_all_harness_modules(tmp_path):
    source = Path(importlib.import_module('wia.tasks').__file__).resolve().parents[2]
    script = '''
import sys, importlib, pkgutil
from importlib.abc import MetaPathFinder
sys.path.insert(0, sys.argv[1])
class RejectHarness(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'core' or fullname.startswith('core.'):
            raise AssertionError('Standalone import requested Harness: ' + fullname)
sys.meta_path.insert(0, RejectHarness())
for name in ('wia.tasks', 'wia.authoring.converters', 'wia.native'):
    package = importlib.import_module(name)
    for item in pkgutil.iter_modules(package.__path__, name + '.'):
        importlib.import_module(item.name)
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(source)], cwd=tmp_path, check=True)




def test_configured_resource_roots_are_resolved_without_harness(tmp_path):
    environment = dict(os.environ, WIA_ROOT=str(tmp_path / 'checkout'),
                       WIA_ESI_ROOT=str(tmp_path / 'upstream'))
    source = Path(importlib.import_module('wia.tasks').__file__).resolve().parents[2]
    environment['PYTHONPATH'] = str(source)
    script = ('from wia.resources import ROOT, ESI_ROOT; from wia.tasks.loader import TASK_ROOT, DATA_ROOT; '
              'print(ROOT); print(ESI_ROOT); print(TASK_ROOT); print(DATA_ROOT)')
    result = subprocess.run([sys.executable, '-c', script], env=environment, cwd=tmp_path,
                            check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == [str(tmp_path / 'checkout'), str(tmp_path / 'upstream'),
        str(tmp_path / 'checkout/explore_to_achieve/wia_v1/release/tasks'), str(tmp_path / 'upstream/dataset/json_clean')]




@pytest.mark.parametrize('session,backend,owner,reasoning', [
    ('native_persistent', 'codex_cli', 'native', 'high'),
    ('native_persistent', 'codex', 'native', 'medium'),
    ('local_http_persistent', 'local_http', 'harness', None),
    ('openai_chat_completions_persistent', 'openai_chat_completions', 'harness', None),
    ('openai_responses_persistent', 'openai_responses', 'harness', None),
])
def test_manifest_backend_metadata_preserves_identity(tmp_path, session, backend, owner, reasoning):
    from types import SimpleNamespace
    from wia.tasks.loader import verify_manifest
    fingerprint = {'schema': 'fixture', 'task_file_sha256': 'original-task'}
    task = SimpleNamespace(run_mode='benchmark', verify_unchanged=lambda: None,
                           manifest=lambda episode: dict(fingerprint, episode_id=episode))
    value = dict(fingerprint, episode_id='episode', max_steps=5, world_mode='pause',
                 world_advances_during_inference=False, observation_condition='current',
                 action_mode='fixed', model_name='fixture', reasoning_effort=reasoning,
                 wall_timeout_s=None, agent_session=session)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value))
    assert verify_manifest(path, task) == value  # Historical manifests need no metadata.
    value['backend_metadata'] = {'backend': backend, 'history_owner': owner, 'provider': None,
        'base_url': None, 'api_key_env': 'FIXTURE_KEY', 'max_output_tokens': 32,
        'temperature': None, 'chat_template_kwargs': None}
    path.write_text(json.dumps(value))
    assert verify_manifest(path, task) == value
    for field, wrong in [('history_owner', 'other'), ('backend', 'wrong')]:
        invalid = dict(value, backend_metadata=dict(value['backend_metadata'], **{field: wrong}))
        path.write_text(json.dumps(invalid))
        with pytest.raises(RuntimeError, match='Backend metadata'):
            verify_manifest(path, task)
    for field, wrong in [('task_file_sha256', 'changed-task'), ('reasoning_effort', 'invalid'),
                         ('backend_metadata', []), ('agent_session', 'unknown')]:
        path.write_text(json.dumps(dict(value, **{field: wrong})))
        with pytest.raises(RuntimeError):
            verify_manifest(path, task)
    if session == 'native_persistent':
        path.write_text(json.dumps(dict(value, reasoning_effort=None)))
        with pytest.raises(RuntimeError, match='policy is inconsistent'):
            verify_manifest(path, task)











@pytest.mark.parametrize('family,module', [
    ('dimensional_v1', 'wia.authoring.converters.wia_dimensional'),
    ('mirror_v1', 'wia.authoring.converters.wia_mirror'),
])
def test_runtime_guards_accept_aliases_without_changing_config(family, module):
    from copy import deepcopy
    from types import SimpleNamespace
    from wia.authoring.modules import canonical_module_name, module_identity
    adapter = importlib.import_module(module)
    path = next(iter(sorted((ROOT / 'explore_to_achieve/wia_v1' / family / 'tasks').glob('*.yaml'))))
    task = yaml.safe_load(path.read_text())
    native = getattr(adapter, 'native', adapter)
    session = SimpleNamespace(scene_name=native.SCENE, room_name=native.ROOM,
                              env=SimpleNamespace(scene=SimpleNamespace(scene_model=native.SCENE)))
    config = deepcopy(task['private']['evaluation'])
    config.update(scene=native.SCENE, room=native.ROOM,
                  bindings={key: value['name'] for key, value in task['private']['bindings'].items()})
    results = []
    for spelling in (adapter.MODULE, canonical_module_name(adapter.MODULE)):
        supplied = dict(deepcopy(config), module=spelling)
        before = deepcopy(supplied)
        results.append(adapter._config(session, supplied))
        assert supplied == before
        assert supplied['module'] == spelling
    assert results[0] == results[1]
    with pytest.raises(ValueError):
        adapter._config(session, dict(config, module='unrelated.external_module'))
    assert module_identity('unrelated.external_module') == 'unrelated.external_module'
    assert module_identity(None) is None

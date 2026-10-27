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
    for path in (ROOT / 'explore_to_achieve').rglob('*.yaml'):
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
from wia.tasks.wia_registry import _observation_contract
assert len(_observation_contract(True)['mounted_rgb_views']) == 8
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(source)], cwd=tmp_path, check=True)


def test_legacy_modules_share_identity_and_monkeypatches():
    provenance = json.loads((ROOT / 'provenance/extraction.json').read_text())
    harness = Path(provenance['source_repository'])
    if not (harness / 'core').is_dir():
        pytest.skip('Legacy Harness checkout is not installed.')
    sys.path.insert(0, str(harness))
    try:
        pairs = [('core.tasks.schema', 'wia.tasks.schema'),
                 ('core.tasks.loader', 'wia.tasks.loader'),
                 ('core.esi_category_conversion', 'wia.authoring.conversion'),
                 ('core.esi_converters.wia_dimensional', 'wia.authoring.converters.wia_dimensional'),
                 ('core.sim.esi_task_evaluation', 'wia.native.esi_task_evaluation'),
                 ('core.sim.esi_temporal_phases', 'wia.native.esi_temporal_phases')]
        for old, canonical in pairs:
            first, second = importlib.import_module(old), importlib.import_module(canonical)
            assert first is second
            marker = object()
            first._extraction_identity_probe = marker
            assert second._extraction_identity_probe is marker
            del first._extraction_identity_probe
    finally:
        sys.path.remove(str(harness))


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
        str(tmp_path / 'checkout/explore_to_achieve/tasks'), str(tmp_path / 'upstream/dataset/json_clean')]


def test_selected_corev2_tasks_load_from_canonical_roots():
    rows = json.loads((ROOT / 'explore_to_achieve/lists/corev2.json').read_text())['tasks']
    assert rows
    for row in rows:
        path = ROOT / row['task_file']
        definition = task_definition(yaml.load(path.read_text(), Loader=UniqueLoader))
        rigid = definition.kind == 'rigid_containment_task'
        task = load_task(path, variant=None if rigid else 'adapted', run_mode='diagnostic')
        task.verify_unchanged()
        scene = task.scene if rigid else load_scene(ROOT / row['scene_file'], variant='adapted', run_mode='diagnostic')
        scene.verify_unchanged()
        assert task.source_path.is_relative_to(ROOT)


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


@pytest.mark.parametrize('module', ['core.esi_converters.action_order', 'wia.authoring.converters.action_order'])
def test_schema_preserves_module_identifiers_and_arbitrary_metadata(module):
    from wia.tasks.schema import CategoryOperationEvaluation, CategoryInitializationPatch
    raw = {'type': 'category_operation', 'module': module, 'operation': 'fixture',
           'parameters': {'historical_pointer': 'core.esi_converters.action_order',
                          'nested': ['wia.authoring.converters.action_order']}}
    assert CategoryOperationEvaluation.model_validate(raw).model_dump(mode='json') == dict(raw, binding_keys=[])
    patch = {'operation': 'category_initialization', 'module': module, 'parameters': raw['parameters']}
    assert CategoryInitializationPatch.model_validate(patch).model_dump(mode='json') == dict(patch, phase='after_postprocessing')


def test_standalone_legacy_dynamic_import_preserves_historical_identity(tmp_path):
    source = Path(importlib.import_module('wia.tasks').__file__).resolve().parents[2]
    script = '''
import sys
from importlib.abc import MetaPathFinder
sys.path.insert(0, sys.argv[1])
class RejectHarness(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'core' or fullname.startswith('core.'):
            raise AssertionError('Dynamic import requested Harness: ' + fullname)
sys.meta_path.insert(0, RejectHarness())
from wia.authoring.modules import import_owned_module
from wia.native.esi_category_evaluation import owned_category_module
from wia.authoring.conversion import get_converter
legacy = 'core.esi_converters.action_order'
canonical = 'wia.authoring.converters.action_order'
assert import_owned_module(legacy) is import_owned_module(canonical)
assert owned_category_module(legacy) is import_owned_module(canonical)
assert get_converter('Action Sequencing', 'Action Order Inference').category
assert import_owned_module(legacy).MODULE == legacy
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(source)], cwd=tmp_path, check=True)


def test_registry_relocation_requires_known_resource_and_matching_hash(tmp_path):
    from wia.tasks.wia_registry import _relative
    historical = tmp_path / 'historical'
    current = tmp_path / 'current'
    relative = 'explore_to_achieve/tasks/example.yaml'
    for root in (historical, current):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_text('same bytes')
    import hashlib
    (current / 'provenance').mkdir()
    (current / 'provenance/extraction.json').write_text(json.dumps({
        'source_repository': str(historical), 'byte_preserved_resources': [relative],
        'resource_sha256': {relative: hashlib.sha256(b'same bytes').hexdigest()}}))
    assert _relative(historical / relative, current) == relative
    (historical / relative).write_text('modified historical bytes')
    with pytest.raises(ValueError, match='outside'):
        _relative(historical / relative, current)
    with pytest.raises(ValueError, match='outside'):
        _relative(tmp_path / 'unknown' / relative, current)



def test_shared_resource_resolver_preserves_unrelated_paths_and_checks_migration_bytes(tmp_path):
    import hashlib
    from wia.resources import resolve_migrated_path
    historical, current = tmp_path / 'historical', tmp_path / 'current'
    relative = 'explore_to_achieve/tasks/fixture.yaml'
    for root in (historical, current):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_bytes(b'original resource')
    (current / 'provenance').mkdir()
    (current / 'provenance/extraction.json').write_text(json.dumps({
        'source_repository': str(historical), 'byte_preserved_resources': [relative],
        'resource_sha256': {relative: hashlib.sha256(b'original resource').hexdigest()}}))
    assert resolve_migrated_path(historical / relative, root=current) == current / relative
    assert resolve_migrated_path(relative, root=current) == current / relative
    external = tmp_path / 'external/data.json'
    assert resolve_migrated_path(external, root=current) == external
    unknown = historical / 'unmigrated/data.json'
    assert resolve_migrated_path(unknown, root=current) == unknown
    for changed_root in (historical, current):
        (changed_root / relative).write_bytes(b'changed bytes')
        with pytest.raises(ValueError, match='Known migrated resource bytes changed'):
            resolve_migrated_path(historical / relative, root=current)
        (changed_root / relative).write_bytes(b'original resource')
    (historical / relative).unlink()
    assert resolve_migrated_path(historical / relative, root=current) == current / relative


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

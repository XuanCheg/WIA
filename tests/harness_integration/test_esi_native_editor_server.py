"""CPU tests for declared native-editor inputs and serialized child control.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q
 tests/test_esi_native_editor_server.py
No sockets, native processes, GPU work or services are started.
"""
import asyncio
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

from wia.authoring.conversion import SourceInstance, ConversionContext, original_scene, physical_task
from wia.runtime.commands.edit_esi_scene_native import load_editor_inputs
from wia.runtime.commands.serve_esi_native_editor import EditorCatalogue, EditorError, NativeEditorManager, read_json, write_json


@pytest.fixture
def catalogue(tmp_path):
    identity = 'CPU Fixture/Native Editor/House/room_0/one.json'
    source_root = tmp_path / 'sources'
    source = source_root / identity
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'scene': 'House', 'room': 'room_0', 'question': 'Inspect the source.'}))
    src = SourceInstance(source, identity, 'CPU Fixture', 'Native Editor')
    context = ConversionContext(tmp_path, seed=7)
    scene = original_scene(src, context)
    task = physical_task(src, context,
        public={'objective': 'Inspect the native scene.', 'success_criteria': 'Record an authoring export.',
                'environment': 'Original indoor scene.', 'constraints': []}, bindings={},
        evaluation={'type': 'category_operation', 'module': 'wia.authoring.converters.temporal_understanding_agent_observation',
                    'operation': 'cpu_editor_fixture', 'binding_keys': [], 'parameters': {}})
    for definition in (scene, task):
        definition['scene']['variants']['adapted'] = deepcopy(definition['scene']['variants']['original'])
    task_path, scene_path = tmp_path / 'task.yaml', tmp_path / 'scene.yaml'
    task_path.write_text(yaml.safe_dump(task)); scene_path.write_text(yaml.safe_dump(scene))
    card = {'id': 'category_aa', 'task_file': str(task_path), 'scene_file': str(scene_path),
            'source_identity': identity, 'seed': 7, 'variant': 'adapted',
            'task_sha256': hashlib.sha256(task_path.read_bytes()).hexdigest(),
            'scene_sha256': hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            'images': [{'private_payload': 'DO_NOT_EXPORT_IMAGES'}]}
    snapshot = {'scenes': [card, {**card, 'id': 'category_bb'}], 'diagnostics': [],
                'core_groups': [{'id': 'core_ab', 'source_identity': identity,
                                 'default_variant_id': 'category_aa', 'original_variant_id': 'category_bb',
                                 'versions': [{'id': 'category_aa', 'label': 'Latest revision'},
                                              {'id': 'category_bb', 'label': 'Original'}]}]}
    snapshot_path = tmp_path / 'snapshot.json'
    write_json(snapshot_path, snapshot)
    store = EditorCatalogue(snapshot_path, input_loader=partial(load_editor_inputs, source_root=source_root))
    return store, snapshot, task_path, scene_path


def test_real_loader_resolves_group_and_canonical_exact_definitions(catalogue):
    store, _, task, scene = catalogue
    selected = store.resolve('core_ab')
    assert selected == {'requested_id': 'core_ab', 'scene_id': 'category_aa',
                        'task_file': str(task.resolve()), 'scene_file': str(scene.resolve()), 'variant': 'adapted'}
    assert store.resolve('category_bb')['scene_id'] == 'category_bb'
    assert store.resolve('category_aa')['variant'] == 'adapted'
    metadata = store.public_metadata()
    assert [row['id'] for row in metadata['variants']] == ['category_aa', 'category_bb']
    assert 'DO_NOT_EXPORT_IMAGES' not in json.dumps(metadata)
    assert not task.parent.joinpath('editor_status.json').exists()


@pytest.mark.parametrize('invalid', ['../category_aa', 'category_aa;echo bad', 'core_zz', '', None, 'category_cc', 'core_cc'])
def test_undeclared_or_malformed_ids_cannot_supply_arbitrary_paths(catalogue, invalid):
    with pytest.raises(EditorError):
        catalogue[0].resolve(invalid)


@pytest.mark.parametrize('mismatch', ['hash', 'missing_path', 'source', 'variant', 'seed', 'group_member', 'scene_seed'])
def test_changed_or_mismatched_declared_inputs_fail_before_process_launch(catalogue, mismatch):
    store, snapshot, _, scene_path = catalogue
    card = snapshot['scenes'][0]
    if mismatch == 'hash': card['task_sha256'] = 'stale'
    elif mismatch == 'missing_path': card['scene_file'] = str(scene_path.parent / 'missing.yaml')
    elif mismatch == 'source': card['source_identity'] = 'Other/Source/House/room_0/two.json'
    elif mismatch == 'variant': card['variant'] = 'undeclared'
    elif mismatch == 'seed': card['seed'] = 99
    elif mismatch == 'group_member': snapshot['core_groups'][0]['versions'] = [{'id': 'category_bb'}]
    else:
        scene = yaml.safe_load(scene_path.read_text())
        scene['seed'] = 99
        scene_path.write_text(yaml.safe_dump(scene))
        card['scene_sha256'] = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    write_json(store.path, snapshot)
    with pytest.raises(EditorError):
        store.resolve('core_ab')


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.exited = asyncio.Event()

    async def wait(self):
        await self.exited.wait()
        return self.returncode

    def exit(self, code=0):
        self.returncode = code
        self.exited.set()


class FakeLauncher:
    def __init__(self):
        self.calls = []

    async def __call__(self, *command, **kwargs):
        process = FakeProcess(100 + len(self.calls))
        self.calls.append((command, kwargs, process))
        return process


def ready(manager):
    write_json(manager.current['directory'] / 'editor_status.json', {'stage': 'ready', 'pid': manager.current['process'].pid})


async def native_requests(manager, actions, *, fail_save=False, exit_on_quit=True, stale_first=False):
    current = manager.current
    seen = set()
    while current['process'].returncode is None:
        request = read_json(current['directory'] / 'editor_request.json', {})
        request_id = request.get('request_id')
        if request_id and request_id not in seen:
            seen.add(request_id)
            actions.append(request)
            response = {**request, 'complete': not (fail_save and request['action'] == 'save')}
            if request['action'] == 'save':
                response['export'] = {'export_version': 'separate_native_export'}
                if fail_save: response['error'] = 'Native simulator is stopped.'
            if stale_first:
                write_json(current['directory'] / 'editor_response.json', {**response, 'request_id': 'stale'})
                await asyncio.sleep(.005)
            write_json(current['directory'] / 'editor_response.json', response)
            if request['action'] == 'quit':
                if exit_on_quit: current['process'].exit()
                return
        await asyncio.sleep(.001)


def test_command_is_argument_list_unique_output_and_same_active_does_not_spawn(catalogue, tmp_path):
    async def scenario():
        launcher = FakeLauncher()
        manager = NativeEditorManager(catalogue[0], tmp_path / 'sessions', launcher=launcher)
        state = await manager.open('core_ab', source_only=True)
        assert state['phase'] == 'loading' and state['scene_id'] == 'category_aa'
        command, options, process = launcher.calls[0]
        assert command[:4] == (sys.executable, '-u', '-B', str(__import__('wia.resources', fromlist=['ROOT']).ROOT / 'src/wia/runtime/commands/edit_esi_scene_native.py'))
        assert command[command.index('--task-file') + 1] == str(catalogue[2])
        assert command[command.index('--variant') + 1] == 'adapted'
        assert command[-1] == '--source-only'
        assert options['start_new_session'] is True and 'shell' not in options and 'env' not in options
        assert (await manager.open('category_aa', source_only=True))['pid'] == process.pid
        assert len(launcher.calls) == 1
        with pytest.raises(EditorError, match='loading'):
            await manager.open('category_bb')
        with pytest.raises(EditorError, match='finish loading'):
            await manager.save()
        with pytest.raises(EditorError, match='boolean'):
            await manager.open('category_aa', source_only='false')
        (manager.current['directory'] / 'native_editor.log').write_bytes(b'x' * 40000)
        assert len(manager.state()['log_tail']) == 16384
        process.exit(1)
        assert manager.state()['phase'] == 'error'
        old_directory = manager.current['directory']
        await manager.open('category_bb')
        assert manager.current['directory'] != old_directory
        assert len(launcher.calls) == 2
    asyncio.run(scenario())


def complete_load(manager, *, complete=True, world_available=True, recovery_load_allowed=True,
                  scene_cleared=False, wrong_identity=False):
    request = read_json(manager.current['directory'] / 'editor_request.json')
    response = {**request, 'complete': complete, 'world_available': world_available,
                'recovery_load_allowed': recovery_load_allowed, 'scene_generation': 1,
                'pid': manager.current['process'].pid, 'app_identity': 700,
                'scene_cleared': scene_cleared, 'previous_world_retained': not scene_cleared}
    if complete:
        response['autosave'] = {'export_version': 'old_world_separate_native_export'}
        if wrong_identity:
            response['scene_id'] = 'category_cc'
    else:
        response['error'] = 'Native reset failed.' if scene_cleared else 'Autosave failed before stage clear.'
    write_json(manager.current['directory'] / 'editor_status.json', {
        'stage': 'ready' if complete else 'error', 'world_available': world_available,
        'recovery_load_allowed': recovery_load_allowed, 'scene_generation': 1,
        'scene_id': response['scene_id'], 'error': response.get('error')})
    write_json(manager.current['directory'] / 'editor_response.json', response)
    return response


def test_switch_submits_one_load_immediately_preserves_pid_and_updates_only_on_exact_ack(catalogue, tmp_path):
    async def scenario():
        launcher = FakeLauncher()
        manager = NativeEditorManager(catalogue[0], tmp_path / 'sessions', launcher=launcher)
        await manager.open('category_aa'); ready(manager)
        old = manager.current
        state = await manager.open('category_bb', source_only=True)
        request = read_json(old['directory'] / 'editor_request.json')
        assert request['action'] == 'load' and request['scene_id'] == 'category_bb'
        assert request['source_only'] is True and request['variant'] == 'adapted'
        assert request['task_file'] == str(catalogue[2]) and request['scene_file'] == str(catalogue[3])
        assert state['phase'] == 'loading' and state['requested_scene_id'] == 'category_bb'
        assert state['scene_id'] == 'category_aa' and state['pid'] == old['process'].pid
        assert manager.current is old and len(launcher.calls) == 1
        assert (await manager.open('category_bb', source_only=True))['load_pending']['request_id'] == request['request_id']
        with pytest.raises(EditorError, match='pending'):
            await manager.open('category_aa')
        with pytest.raises(EditorError, match='finish loading'):
            await manager.save()
        response = complete_load(manager)
        state = manager.state()
        assert state['scene_id'] == state['current_scene_id'] == 'category_bb'
        assert state['source_only'] is True and state['requested_scene_id'] is None
        assert state['load_pending'] is None and state['phase'] == 'ready'
        assert state['last_save']['export'] == response['autosave']
        assert state['last_load']['scene_generation'] == 1
        assert state['pid'] == old['process'].pid and len(launcher.calls) == 1
        # The manager did not send separate save/quit operations or spawn another process.
        assert read_json(old['directory'] / 'editor_request.json') == request
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['before_clear', 'after_clear', 'unsafe_cleanup', 'wrong_identity'])
def test_failed_load_preserves_last_selection_but_never_claims_cleared_world_ready(catalogue, tmp_path, failure):
    async def scenario():
        launcher = FakeLauncher()
        manager = NativeEditorManager(catalogue[0], tmp_path / 'sessions', launcher=launcher)
        await manager.open('category_aa'); ready(manager)
        old = manager.current
        await manager.open('category_bb')
        response = complete_load(manager, complete=failure == 'wrong_identity',
            world_available=failure in ('before_clear', 'wrong_identity'),
            recovery_load_allowed=failure != 'unsafe_cleanup', scene_cleared=failure != 'before_clear',
            wrong_identity=failure == 'wrong_identity')
        state = manager.state()
        assert state['scene_id'] == 'category_aa'
        assert state['load_pending'] is None and state['last_load']['complete'] is False
        assert manager.current is old and old['process'].returncode is None and len(launcher.calls) == 1
        if failure in ('after_clear', 'unsafe_cleanup'):
            assert state['current_scene_id'] is None and state['world_available'] is False
            assert state['phase'] == 'error'
            with pytest.raises(EditorError, match='finish loading'):
                await manager.save()
        if failure == 'unsafe_cleanup':
            with pytest.raises(EditorError, match='unavailable'):
                await manager.open('category_bb')
        elif failure != 'wrong_identity':
            retry = await manager.open('category_bb')
            assert retry['phase'] == 'loading'
            assert retry['load_pending']['request_id'] != response['request_id']
        else:
            assert state['phase'] == 'error' and state['current_scene_id'] is None
            assert 'identity differs' in state['last_load']['error']
        assert len(launcher.calls) == 1
    asyncio.run(scenario())


def test_stale_load_ack_and_undeclared_targets_cannot_change_or_overwrite_pending(catalogue, tmp_path):
    async def scenario():
        launcher = FakeLauncher()
        manager = NativeEditorManager(catalogue[0], tmp_path / 'sessions', launcher=launcher)
        await manager.open('category_aa'); ready(manager)
        await manager.open('category_bb')
        directory = manager.current['directory']
        request = read_json(directory / 'editor_request.json')
        write_json(directory / 'editor_response.json', {**request, 'request_id': 'stale', 'complete': True})
        state = manager.state()
        assert state['phase'] == 'loading' and state['scene_id'] == 'category_aa'
        for invalid in ('category_cc', '../../outside', 'core_missing'):
            with pytest.raises(EditorError):
                await manager.open(invalid)
            assert read_json(directory / 'editor_request.json') == request
        assert len(launcher.calls) == 1
        manager.current['process'].exit(1)
        assert manager.state()['phase'] == 'error'
        assert manager.state()['last_load']['complete'] is False
        replacement = await manager.open('category_bb')
        assert len(launcher.calls) == 2  # Explicit caller request only, after the old child exited.
        assert replacement['previous_process_exit']['returncode'] == 1
        assert replacement['previous_process_exit']['last_load']['complete'] is False
    asyncio.run(scenario())


def test_save_matches_request_ids_and_operations_are_serialized(catalogue, tmp_path):
    async def scenario():
        launcher = FakeLauncher()
        manager = NativeEditorManager(catalogue[0], tmp_path / 'sessions', launcher=launcher,
                                      save_timeout=.2, poll_interval=.001)
        await manager.open('category_aa'); ready(manager)
        actions = []
        worker = asyncio.create_task(native_requests(manager, actions, stale_first=True))
        try:
            first = await manager.save()
            second = await manager.save()
            assert first['request_id'] != second['request_id']
            assert manager.state()['last_save'] == second
            assert len(launcher.calls) == 1
            async with manager.lock:
                with pytest.raises(EditorError, match='in progress'):
                    await manager.open('category_bb')
                with pytest.raises(EditorError, match='in progress'):
                    await manager.save()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
    asyncio.run(scenario())

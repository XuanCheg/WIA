#!/usr/bin/env python3
"""Serve the native scene editor portal and VNC transport on one port.

Run with the behavior Python and inherited DISPLAY/CUDA settings:
 python -B src/wia/runtime/commands/serve_esi_native_editor.py --catalogue-file SNAPSHOT.json
 --novnc-root RUNTIME/noVNC --output-dir OUTPUT --port 8615
Requires aiohttp; installs and native display/VNC startup are separate operations.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
PORTAL = ROOT / 'src/wia/runtime/web/native_editor/index.html'


class EditorError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


class EditorCatalogue:
    """Resolve only declared canonical cards or explicitly declared Core groups."""
    def __init__(self, path, input_loader=None):
        self.path = Path(path).resolve()
        self.input_loader = input_loader

    def snapshot(self):
        snapshot = read_json(self.path)
        if not isinstance(snapshot, dict):
            raise EditorError('The declared catalogue snapshot is unavailable.', 503)
        return snapshot

    def public_metadata(self):
        snapshot = self.snapshot()
        fields = ('id', 'name', 'category', 'subcategory', 'scene', 'room', 'source_identity',
                  'variant', 'original_baseline', 'native_capture_complete')
        cards = {row['id']: row for row in snapshot.get('scenes', []) + snapshot.get('diagnostics', [])}
        groups = []
        for group in snapshot.get('core_groups', []):
            public = {key: group[key] for key in
                      ('id', 'source_identity', 'default_variant_id', 'original_variant_id') if key in group}
            public['versions'] = [{key: version[key] for key in
                                   ('id', 'label', 'role', 'capture_state', 'task_file', 'scene_file', 'variant_key')
                                   if key in version} for version in group.get('versions', [])]
            groups.append(public)
        return {'core_groups': groups,
                'variants': [{key: row[key] for key in fields if key in row} for row in cards.values()]}

    def resolve(self, scene_id):
        if not isinstance(scene_id, str) or not re.fullmatch(r'(category|core)_[0-9a-f]+', scene_id):
            raise EditorError('Provide a declared Core group or canonical category ID.', 400)
        snapshot = self.snapshot()
        cards = {row['id']: row for row in snapshot.get('scenes', []) + snapshot.get('diagnostics', [])}
        group = next((row for row in snapshot.get('core_groups', []) if row['id'] == scene_id), None)
        canonical = group['default_variant_id'] if group else scene_id
        if group and canonical not in {version['id'] for version in group.get('versions', [])}:
            raise EditorError('The group default is not one of its declared versions.', 400)
        row = cards.get(canonical)
        if row is None:
            raise EditorError('This scene ID is not declared in the current catalogue.', 404)
        paths = {}
        for kind in ('task', 'scene'):
            declared = row.get(kind + '_file')
            if not isinstance(declared, str) or not declared:
                raise EditorError('The declared ' + kind + ' definition is unavailable.', 409)
            path = Path(declared).expanduser().resolve()
            if not path.is_file():
                raise EditorError('The declared ' + kind + ' definition does not exist.', 409)
            expected = row.get(kind + '_sha256')
            if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise EditorError('The declared ' + kind + ' definition hash changed.', 409)
            paths[kind + '_file'] = path
        loader = self.input_loader
        if loader is None:
            from wia.runtime.commands.edit_esi_scene_native import load_editor_inputs
            loader = load_editor_inputs
        try:
            inputs = loader(paths['scene_file'], paths['task_file'], variant=row.get('variant'))
            source = row.get('source_identity') or row.get('source')
            if source and inputs.task.definition.source.json_path != source:
                raise ValueError('Catalogue source identity differs from the actual task definition.')
            if row.get('seed') is not None and inputs.scene.seed != row['seed']:
                raise ValueError('Catalogue seed differs from the actual definitions.')
        except (OSError, ValueError, KeyError) as error:
            raise EditorError('Invalid declared editor inputs: ' + str(error), 409) from error
        return {'requested_id': scene_id, 'scene_id': canonical,
                'task_file': str(paths['task_file']), 'scene_file': str(paths['scene_file']),
                'variant': inputs.scene.variant}


class NativeEditorManager:
    """Serialize child authoring sessions; never signal or kill another process."""
    def __init__(self, catalogue, output_dir, *, launcher=None, save_timeout=30., poll_interval=.1):
        self.catalogue = catalogue
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.launcher = launcher or asyncio.create_subprocess_exec
        self.save_timeout = min(30., max(0., save_timeout))
        self.poll_interval = poll_interval
        self.lock = asyncio.Lock()
        self.current = None
        self.last_save = None
        self.last_load = None
        self.previous_process_exit = None

    def _poll_load(self):
        current = self.current
        pending = current.get('pending_load') if current else None
        if not pending:
            return
        response = read_json(current['directory'] / 'editor_response.json', {})
        if response.get('request_id') == pending['request_id'] and response.get('action') == 'load':
            self.last_load = {**response, 'requested_selection': pending['selection']}
            if response.get('complete') is True:
                expected = {**pending['selection'], 'source_only': pending['source_only']}
                if any(response.get(key) != expected[key] for key in
                       ('scene_id', 'task_file', 'scene_file', 'variant', 'source_only')):
                    self.last_load.update(complete=False, error='Native load acknowledgement identity differs from its validated request.')
                    current['identity_error'] = True
            if self.last_load.get('complete') is True:
                current['selection'] = pending['selection']
                current['source_only'] = pending['source_only']
                current['was_ready'] = True
                current['identity_error'] = False
                if response.get('autosave') is not None:
                    self.last_save = {'action': 'save', 'complete': True, 'export': response['autosave'],
                                      'via_load_request_id': pending['request_id']}
            current['pending_load'] = None
        elif current['process'].returncode is not None:
            self.last_load = {'request_id': pending['request_id'], 'action': 'load', 'complete': False,
                              'error': 'Native editor exited before acknowledging load.',
                              'requested_selection': pending['selection']}
            current['pending_load'] = None

    def state(self):
        if self.current is None:
            return {'phase': 'idle', 'scene_id': None, 'pid': None, 'authoring_only': True}
        self._poll_load()
        current = self.current
        pending = current.get('pending_load')
        status = read_json(current['directory'] / 'editor_status.json', {})
        running = current['process'].returncode is None
        phase = status.get('stage', 'loading') if running else ('closed' if current['process'].returncode == 0 else 'error')
        if pending and running:
            phase = 'loading'
        elif running and (current.get('identity_error') or (self.last_load and self.last_load.get('complete') is not True
                                                               and status.get('stage') not in ('ready', 'exported'))):
            phase = 'error'
        if not pending and status.get('stage') in ('ready', 'exported'):
            current['was_ready'] = True
        log_path = current['directory'] / 'native_editor.log'
        try:
            with log_path.open('rb') as stream:
                stream.seek(0, 2)
                size = stream.tell()
                stream.seek(max(0, size - 16384))
                log_tail = stream.read(16384).decode('utf-8', errors='replace')
        except FileNotFoundError:
            log_tail = ''
        return {**current['selection'], 'source_only': current['source_only'],
                'pid': current['process'].pid, 'phase': phase, 'running': running,
                'returncode': current['process'].returncode, 'native_status': status,
                'output_dir': str(current['directory']), 'last_save': self.last_save,
                'last_load': self.last_load, 'load_pending': pending,
                'previous_process_exit': self.previous_process_exit,
                'requested_scene_id': pending['selection']['scene_id'] if pending else None,
                'current_scene_id': (None if status.get('world_available') is False or current.get('identity_error')
                                     else current['selection']['scene_id']),
                'world_available': status.get('world_available'),
                'recovery_load_allowed': status.get('recovery_load_allowed'),
                'log_tail': log_tail, 'authoring_only': True}

    async def _request(self, action, timeout):
        current = self.current
        if current is None or current['process'].returncode is not None:
            raise EditorError('There is no running native editor.')
        request_id = uuid.uuid4().hex
        write_json(current['directory'] / 'editor_request.json', {'request_id': request_id, 'action': action})
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            response = read_json(current['directory'] / 'editor_response.json', {})
            if response.get('request_id') == request_id and response.get('action') == action:
                if response.get('complete') is not True:
                    raise EditorError('Native ' + action + ' failed: ' + str(response.get('error', 'incomplete response')))
                if action == 'save':
                    self.last_save = response
                return response
            if current['process'].returncode is not None:
                raise EditorError('The native editor exited before acknowledging ' + action + '.')
            if asyncio.get_running_loop().time() >= deadline:
                raise EditorError('Native ' + action + ' did not acknowledge within the bounded timeout.')
            await asyncio.sleep(self.poll_interval)

    async def save(self):
        if self.lock.locked():
            raise EditorError('Another editor operation is in progress.')
        async with self.lock:
            state = self.state()
            if (not self.current or self.current.get('pending_load') or self.current.get('identity_error')
                    or not self.current['was_ready']
                    or not state.get('running') or state.get('world_available') is False):
                raise EditorError('The editor must finish loading before saving.')
            return await self._request('save', self.save_timeout)

    async def open(self, scene_id, *, source_only=False):
        if type(source_only) is not bool:
            raise EditorError('source_only must be a boolean.', 400)
        selection = self.catalogue.resolve(scene_id)
        state = self.state()
        pending = self.current.get('pending_load') if self.current else None
        if pending and state.get('running'):
            if pending['selection']['scene_id'] == selection['scene_id'] and pending['source_only'] == source_only:
                return state
            raise EditorError('A native scene load is pending; no second scene or process was launched.')
        if (self.current and state.get('running') and self.current['selection']['scene_id'] == selection['scene_id']
                and self.current['source_only'] == source_only and state.get('world_available') is not False
                and not self.current.get('identity_error')):
            return state
        if self.lock.locked():
            raise EditorError('Another editor operation is in progress; no second scene was launched.')
        async with self.lock:
            state = self.state()
            if state.get('running'):
                if state['phase'] not in ('ready', 'exported') and not (
                        state['phase'] == 'error' and state.get('recovery_load_allowed') is True):
                    raise EditorError('The current scene is loading or unavailable; wait before switching.')
                request_id = uuid.uuid4().hex
                payload = {'request_id': request_id, 'action': 'load',
                           **{key: selection[key] for key in ('task_file', 'scene_file', 'variant', 'scene_id')},
                           'source_only': source_only}
                write_json(self.current['directory'] / 'editor_request.json', payload)
                self.current['pending_load'] = {'request_id': request_id, 'selection': selection,
                                                'source_only': source_only}
                # Native load owns validation, autosave and stage replacement atomically.
                return self.state()
            if self.current is not None:
                self.previous_process_exit = {'pid': self.current['process'].pid,
                    'returncode': self.current['process'].returncode,
                    'scene_id': self.current['selection']['scene_id'], 'phase': state['phase'],
                    'native_status': state['native_status'], 'last_load': self.last_load}
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
            directory = self.output_dir / ('session_' + stamp + '_' + uuid.uuid4().hex[:8])
            directory.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, '-u', '-B', str(ROOT / 'src/wia/runtime/commands/edit_esi_scene_native.py'),
                       '--task-file', selection['task_file'], '--scene-file', selection['scene_file'],
                       '--variant', selection['variant'], '--output-dir', str(directory)]
            if source_only:
                command.append('--source-only')
            with (directory / 'native_editor.log').open('ab') as log:
                process = await self.launcher(*command, stdout=log, stderr=asyncio.subprocess.STDOUT,
                                              start_new_session=True)
            self.current = {'selection': selection, 'source_only': source_only,
                            'directory': directory, 'process': process, 'was_ready': False, 'pending_load': None, 'identity_error': False}
            self.last_save = None
            self.last_load = None
            return self.state()


def create_app(manager, novnc_root, *, vnc_host='127.0.0.1', vnc_port=5902):
    """Construct routes only; callers explicitly own binding the HTTP listener."""
    from aiohttp import WSMsgType, web
    static_root = Path(novnc_root).expanduser().resolve()
    if not (static_root / 'vnc.html').is_file():
        raise ValueError('The declared official noVNC static root lacks vnc.html.')

    @web.middleware
    async def errors(request, handler):
        try:
            return await handler(request)
        except EditorError as error:
            return web.json_response({'error': str(error)}, status=error.status)
        except (json.JSONDecodeError, TypeError) as error:
            return web.json_response({'error': str(error)}, status=400)

    async def portal(request):
        return web.FileResponse(PORTAL)

    async def catalogue(request):
        return web.json_response(manager.catalogue.public_metadata())

    async def state(request):
        return web.json_response(manager.state())

    async def open_scene(request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise EditorError('Provide a JSON object.', 400)
        return web.json_response(await manager.open(payload.get('scene_id'), source_only=payload.get('source_only', False)))

    async def save_scene(request):
        return web.json_response(await manager.save())

    async def websocket(request):
        try:
            reader, writer = await asyncio.open_connection(vnc_host, vnc_port)
        except OSError as error:
            raise EditorError('Native VNC is not available: ' + str(error), 503) from error
        ws = web.WebSocketResponse(protocols=['binary'], max_msg_size=1024 * 1024)
        tasks = []
        try:
            await ws.prepare(request)

            async def from_browser():
                async for message in ws:
                    if message.type == WSMsgType.BINARY:
                        for offset in range(0, len(message.data), 65536):
                            writer.write(message.data[offset:offset + 65536])
                            await writer.drain()
                    elif message.type == WSMsgType.TEXT:
                        await ws.close(code=1003, message=b'Binary VNC frames required.')
                        break
                    elif message.type == WSMsgType.ERROR:
                        break

            async def from_vnc():
                while not ws.closed:
                    chunk = await reader.read(65536)
                    if not chunk:
                        break
                    await ws.send_bytes(chunk)

            tasks = [asyncio.create_task(from_browser()), asyncio.create_task(from_vnc())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()
            await ws.close()
        return ws

    app = web.Application(middlewares=[errors], client_max_size=16384)
    app.router.add_get('/', portal)
    app.router.add_get('/api/catalogue', catalogue)
    app.router.add_get('/api/state', state)
    app.router.add_post('/api/open', open_scene)
    app.router.add_post('/api/save', save_scene)
    app.router.add_get('/websockify', websocket)
    app.router.add_static('/novnc/', static_root, show_index=False, follow_symlinks=False)
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue-file', type=Path, required=True)
    parser.add_argument('--novnc-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vnc-host', default='127.0.0.1')
    parser.add_argument('--vnc-port', type=int, default=5902)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8615)
    args = parser.parse_args(argv)
    try:
        from aiohttp import web
    except ImportError as error:
        raise SystemExit('aiohttp is required in the server interpreter or isolated runtime PYTHONPATH.') from error
    manager = NativeEditorManager(EditorCatalogue(args.catalogue_file), args.output_dir)
    app = create_app(manager, args.novnc_root, vnc_host=args.vnc_host, vnc_port=args.vnc_port)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

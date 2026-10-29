"""Additive category-to-CaptureQueue bridge, including review-only failures.

Run: python src/wia/runtime/commands/run_esi_category_campaign.py run --campaign-root ROOT
Original sources, working code, scenes, tasks, seed and camera protocol are
frozen before enqueue. Diagnostic images never confer approval eligibility.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import uuid

import yaml

from wia.authoring.conversion import ROOT, ConversionContext, SourceInstance, original_scene
from wia.runtime.scene_capture_candidates import archive_revision, git, rendering_hashes
from wia.runtime.scene_capture_queue import (CaptureQueue, PROTOCOL, complete_images, digest,
    publish_json, read_json, sha256, verify_snapshot, write_json)
from wia.runtime.scene_review import fingerprint
from wia.tasks.schema import SceneDefinition
from core.robot_cameras import ADDITIONAL_RGB_NAMES, RGB_TO_VIEW

DEFAULT_VIEWS = ['global_diagonal', 'global_overhead', 'global_front', 'global_back',
                 'global_left', 'global_right', 'head', 'wrist', 'scene',
                 *[RGB_TO_VIEW[name] for name in ADDITIONAL_RGB_NAMES]]
_REFERENCE_JOB_INDEX = {}


def _source_members(record):
    """Validate explicit world membership before sealing any snapshot bytes."""
    if 'capture_source_members' not in record:
        return None
    members = record['capture_source_members']
    if not isinstance(members, list) or not members:
        raise ValueError('Explicit capture source membership must be a nonempty list.')
    result, seen = [], set()
    for member in members:
        identity = member['source_identity']
        path = PurePosixPath(identity)
        if (not isinstance(identity, str) or path.is_absolute() or len(path.parts) != 5 or
                '..' in path.parts or '\\' in identity or path.suffix != '.json' or str(path) != identity):
            raise ValueError('Capture source identity must be a safe five-part ESI JSON path.')
        if identity in seen:
            raise ValueError('Capture source membership must contain unique identities.')
        seen.add(identity)
        source = Path(member['source_file'])
        if sha256(source) != member['source_sha256']:
            raise ValueError('Capture member source hash changed: ' + identity)
        result.append({'source_identity': identity, 'source_file': str(source), 'source_sha256': member['source_sha256']})
    primary = next((row for row in result if row['source_identity'] == record['source_identity']), None)
    if (primary is None or primary['source_sha256'] != record['source_sha256'] or
            Path(primary['source_file']) != Path(record['original_source'])):
        raise ValueError('Explicit capture members must include the exact primary source identity/file/hash.')
    return result


def converter_runtime_dependencies(code_root, modules):
    """Seal owned evaluators and their local converter imports in capture identity."""
    code_root, result, pending = Path(code_root), {}, list(modules)
    while pending:
        module = pending.pop()
        if not module or not module.startswith(('wia.authoring.converters.', 'wia.converters.')):
            continue
        relative = 'src/' + module.replace('.', '/') + '.py'
        if relative in result:
            continue
        path = code_root / relative
        result[relative] = sha256(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module if not node.level else '.'.join(module.split('.')[:-node.level]) + '.' + node.module
                if imported.startswith(('wia.authoring.converters.', 'wia.converters.')):
                    pending.append(imported)
            elif isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names if alias.name.startswith('wia.authoring.converters.'))
    return result


def reference_job_index(root):
    """Index immutable source/job identities without rescanning all manifests."""
    queue = CaptureQueue(root)
    cache = _REFERENCE_JOB_INDEX.setdefault(str(queue.root), {'sources': {}, 'jobs': set(), 'records': set()})
    def remember(source, job):
        cache['sources'].setdefault(source, set()).add(job)
        cache['jobs'].add(job)
    # Category cards live beside the queue and already contain sealed source
    # identities. Read only newly published cards; incomplete cards are retried.
    for path in (queue.root / 'category_records').glob('*.json'):
        if path.name in cache['records']:
            continue
        record = read_json(path, {})
        if record.get('job_id'):
            remember(record['source_identity'], record['job_id'])
            cache['records'].add(path.name)
    return cache['sources']


def freeze_working_code(queue_root, *, repo=ROOT):
    """Freeze current local bytes; Git HEAD alone is not the implementation authority."""
    repo = Path(repo).resolve()
    queue_root = Path(queue_root).resolve()
    storage = read_json(queue_root / 'category_storage.json', {})
    root = Path(storage.get('artifact_root', queue_root)) / 'revisions'
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(storage.get('code_cache_root', '/tmp/show-harness-zhouxinyang/category_code'))
    cache.mkdir(parents=True, exist_ok=True)
    tracked = git(repo, 'ls-files', '-z').decode().split('\0')
    added = [str(p.relative_to(repo)) for name in ('src', 'scripts', 'configs')
             for p in (repo / name).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')]
    runtime_directories = {'src', 'scripts', 'configs', 'assets'}
    files = sorted({name for name in tracked + added if name and (repo / name).is_file()
                    and (Path(name).parts[0] in runtime_directories or name in ('pyproject.toml', 'LICENSE'))
                    and not name.endswith(('.pyc', '.png', '.mp4', '.pdf'))})
    sources = {name: repo / name for name in files}
    sources.update({str(path.relative_to(repo)): path
                    for path in (repo / 'explore_to_achieve/prompts').rglob('*') if path.is_file()})
    from wia.runtime.resources import harness_runtime_sources
    sources.update(harness_runtime_sources())
    hashes = {name: sha256(path) for name, path in sources.items()}

    esi_commit = git(repo / 'ESI', 'rev-parse', 'HEAD').decode().strip()
    code_id = 'working_' + digest({'files': hashes, 'esi_commit': esi_commit})[:32]
    target = cache / code_id
    marker = target / '.capture_revision.json'
    if not marker.is_file():
        temporary = cache / ('.building_' + uuid.uuid4().hex)
        temporary.mkdir()
        try:
            for name in sources:
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(sources[name].read_bytes())
                if sha256(destination) != hashes[name]:
                    raise RuntimeError('Working code changed during freezing: ' + name)
            esi = archive_revision(repo / 'ESI', esi_commit, cache / ('esi_' + esi_commit))
            (temporary / 'ESI').symlink_to(esi, target_is_directory=True)
            write_json(temporary / '.capture_revision.json', {'commit': code_id,
                'parent_git_commit': git(repo, 'rev-parse', 'HEAD').decode().strip(),
                'esi_commit': esi_commit, 'files': hashes, 'local_working_tree': True})
            try:
                temporary.rename(target)
            except OSError:
                if not marker.is_file():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    implementation = rendering_hashes(target)
    implementation = {name: value for name, value in implementation.items()
        if not name.startswith('src/wia/converters/') and not name.startswith('src/wia/runtime/esi_category_')}
    for name in ('configs/esi_trunk.json', 'configs/esi_robot_cameras.json', 'configs/esi_viewer.json'):
        implementation[name] = sha256(target / name)
    archive_path = root / (code_id + '.tar.gz')
    if not archive_path.is_file():
        with tempfile.TemporaryDirectory(prefix='esi-code-archive-') as scratch:
            local_archive = Path(scratch) / archive_path.name
            with tarfile.open(local_archive, 'w:gz', dereference=True) as archive:
                archive.add(target, arcname=target.name)
            temporary_archive = root / ('.' + archive_path.name + '.' + uuid.uuid4().hex)
            shutil.copyfile(local_archive, temporary_archive)
            temporary_archive.replace(archive_path)
    archive_hash = sha256(archive_path)
    write_json(root / (code_id + '.json'), {'code_root': str(target), 'code_archive': str(archive_path),
        'code_archive_sha256': archive_hash, 'code_commit': code_id, 'esi_commit': esi_commit,
        'note': 'Durable immutable archive in home; extracted worker import cache is disposable.'})
    return {'code_root': str(target), 'code_commit': code_id, 'esi_commit': esi_commit,
            'implementation_hashes': implementation, 'code_archive': str(archive_path),
            'code_archive_sha256': archive_hash}


def diagnostic_fingerprint(record):
    return digest({k: record.get(k) for k in ('candidate_id', 'source_identity', 'source_sha256',
        'converter_version', 'status', 'failure_stage', 'failure_reason', 'warning_codes', 'assumptions')})


def publish_record(record, queue_root, *, frozen=None, priority=0, timeout_s=1800,
                   max_attempts=2, report_file=None, batch_file=None, snapshot_error=None):
    """Every result is published before best-effort capture; one failure cannot block a batch."""
    queue = CaptureQueue(queue_root)
    directory = queue.root / 'category_records'
    directory.mkdir(exist_ok=True)
    path = directory / (record['candidate_id'] + '.json')
    existing = read_json(path)
    if existing:
        if existing['record_revision'] != record['revision']:
            raise ValueError('A category revision identity cannot be overwritten.')
        if priority and existing.get('job_id'):
            queue.prioritize(existing['job_id'], priority)
        return existing
    value = {**record, 'record_revision': record['revision'],
        'review_record_id': record['candidate_id'], 'published_at': time.time(),
        'report_file': str(report_file) if report_file else None,
        'batch_file': str(batch_file) if batch_file else None, 'job_id': None,
        'fingerprint': diagnostic_fingerprint(record), 'source_only': False,
        'diagnostic_only': record['status'] != 'generated', 'diagnostic_images': [],
        'diagnostic_logs': list(record.get('diagnostic_logs', [])), 'review_submitted': True}
    # Publish the failure card first, even when no valid SceneDefinition exists.
    publish_json(path, value)
    try:
        source = Path(record['original_source'])
        if sha256(source) != record['source_sha256']:
            raise ValueError('Original source changed after CPU conversion; create a new category revision.')
        members = _source_members(record)
        raw = json.loads(source.read_text())
        value.update(question=raw.get('question', ''), scene_model=raw.get('scene'), room=raw.get('room'))
        if len(Path(record['source_identity']).parts) != 5:
            raise ValueError('Source identity does not use the ESI hierarchy.')
        scene_path = Path(record['scene_file']) if record.get('scene_file') else None
        definition = None
        if scene_path and scene_path.is_file():
            definition = SceneDefinition.model_validate(yaml.safe_load(scene_path.read_text()))
        if definition is None:
            source_instance = SourceInstance(source, record['source_identity'], record['big_task'], record['small_task'])
            definition = SceneDefinition.model_validate(original_scene(source_instance, ConversionContext(source.parent, record['seed'])))
            scene_path = Path(record['diagnostic_dir']) / 'source_only_scene.yaml'
            scene_path.write_text(yaml.safe_dump(definition.model_dump(mode='json', by_alias=True), sort_keys=False))
            value.update(source_only=True, diagnostic_only=True)
        value['variant'] = record.get('variant', 'original') if not value['source_only'] else 'original'
        if value['variant'] not in definition.scene.variants:
            raise ValueError('Capture variant is not declared by the category overlay.')
        preserved_reference = record.get('reference_only') is True and definition.status == 'ready'
        if definition.status != 'pending' and not preserved_reference:
            raise ValueError('Automatic category publication must not mark scenes ready.')
        if snapshot_error:
            raise RuntimeError('Immutable code snapshot unavailable: ' + snapshot_error)
        frozen = frozen or freeze_working_code(queue_root)
        implementation_hashes = dict(frozen['implementation_hashes'])
        owner_modules = [record.get('converter_module'), *record.get('composite_member_modules', [])]
        for patch in definition.scene.variants[value['variant']].patches:
            if patch.operation == 'category_initialization':
                owner_modules.append(patch.module)
        implementation_hashes.update(converter_runtime_dependencies(frozen['code_root'], owner_modules))
        value['fingerprint'] = fingerprint(scene_path, source, value['variant'], record['seed'])
        views = list(DEFAULT_VIEWS)
        initial_bundle = record.get('capture_initial_agent_bundle') is True
        if initial_bundle:
            if not record.get('task_file'):
                raise ValueError('Initial Agent capture bundle requires the actual frozen task file.')
            views.append('initial_overview_rgb')
        from wia.runtime.sim.esi_review_cameras import review_view_names
        views += review_view_names(definition)
        if any(patch.operation == 'category_initialization' and
               patch.module == 'wia.authoring.converters.temporal_understanding_two_phase'
               for patch in definition.scene.variants[value['variant']].patches):
            views += ['phase_1_head', 'phase_1_wrist', 'phase_2_head', 'phase_2_wrist',
                      'phase_2_global_diagonal', *[f'phase_{phase}_{view}' for phase in (1, 2)
                          for view in ('object_overview', 'object_top', 'object_side')]]
        if record['big_task'] == 'Cognitive Mapping':
            views.append('global_floorplan')
        robot_added = any(patch.operation == 'add_existing_r1'
                          for patch in definition.scene.variants[value['variant']].patches)
        if record['small_task'] in ('Deformable', 'Liquid Volume') and not robot_added and not initial_bundle:
            views = [v for v in views if v not in RGB_TO_VIEW.values()]
        campaign_binding = None
        if record.get('task_edit2_capture_campaign') is True:
            campaign_binding = {'candidate_id': record['candidate_id'],
                'task_edit2_variant_key': record['task_edit2_variant_key'],
                'source_members': sorted([{'source_identity': member['source_identity'],
                    'source_sha256': member['source_sha256']} for member in members if members is not None],
                    key=lambda member: member['source_identity']) if members is not None else
                    [{'source_identity': record['source_identity'], 'source_sha256': record['source_sha256']}],
                'task_origin_file': str(Path(record['task_file']).resolve()) if record.get('task_file') else None,
                'scene_origin_file': str(scene_path.resolve())}
        from wia.tasks.loader import is_release_v1
        edge_policy = is_release_v1(scene_path)
        signature = digest({'current_support_edge_policy': edge_policy, 'fingerprint': value['fingerprint'], 'implementation_hashes': implementation_hashes,
            'task_sha256': sha256(record['task_file']) if record.get('task_file') else None,
            'views': views, 'protocol': PROTOCOL, 'source_only': value['source_only'],
            'runtime': definition.runtime.model_dump(mode='json'),
            **({'task_edit2_capture_binding': campaign_binding} if campaign_binding is not None else {}),
            **({'capture_source_members': members} if members is not None else {}),
            **({'capture_initial_agent_bundle': True} if initial_bundle else {})})
        job_id = 'capture_' + signature[:32]
        storage = read_json(queue.root / 'category_storage.json', {})
        artifact_root = Path(storage.get('artifact_root', queue.root))
        (artifact_root / 'jobs').mkdir(parents=True, exist_ok=True)
        job = artifact_root / 'jobs' / job_id
        temporary = artifact_root / 'jobs' / ('.building_' + uuid.uuid4().hex)
        snapshot = temporary / 'input'
        snapshot.mkdir(parents=True)
        try:
            relative_source = 'data/' + record['source_identity']
            for member in members if members is not None else [{'source_identity': record['source_identity'],
                    'source_file': str(source), 'source_sha256': record['source_sha256']}]:
                destination = snapshot / 'data' / member['source_identity']
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(Path(member['source_file']).read_bytes())
                if sha256(destination) != member['source_sha256']:
                    raise ValueError('Capture member changed during snapshot sealing: ' + member['source_identity'])
            (snapshot / 'scene.yaml').write_bytes(scene_path.read_bytes())
            if record.get('scene_sha256') and sha256(snapshot / 'scene.yaml') != record['scene_sha256']:
                raise ValueError('Capture scene changed after submission selection.')
            if record.get('task_file'):
                (snapshot / 'task.yaml').write_bytes(Path(record['task_file']).read_bytes())
                if record.get('task_sha256') and sha256(snapshot / 'task.yaml') != record['task_sha256']:
                    raise ValueError('Capture task changed after submission selection.')
                task_raw = yaml.safe_load((snapshot / 'task.yaml').read_text())
                if task_raw.get('kind') == 'rigid_containment_task':
                    # Rigid tasks resolve their approved overlay inside data_root.
                    # Seal that dependency without rewriting the protected task.
                    relative_scene = PurePosixPath(task_raw['scene_file'])
                    if relative_scene.is_absolute() or '..' in relative_scene.parts:
                        raise ValueError('Rigid capture scene path must remain inside its data root.')
                    destination = snapshot / 'data' / relative_scene
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes((snapshot / 'scene.yaml').read_bytes())
            input_hashes = {str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob('*') if p.is_file()}
            manifest = {**frozen, 'implementation_hashes': implementation_hashes,
                'current_support_edge_policy': edge_policy,
                'schema': 'esi_capture_candidate_v1', 'candidate_id': record['candidate_id'],
                'job_id': job_id, 'capture_signature': signature, 'capture_protocol': PROTOCOL,
                'fingerprint': value['fingerprint'], 'snapshot_dir': str(job / 'input'),
                'snapshot_scene_file': str(job / 'input/scene.yaml'), 'snapshot_data_root': str(job / 'input/data'),
                'snapshot_source_file': str(job / 'input' / relative_source), 'input_hashes': input_hashes,
                'scene_model': raw['scene'], 'room': raw['room'], 'seed': record['seed'], 'variant': value['variant'],
                'category': record['big_task'], 'subcategory': record['small_task'],
                'big_task': record['big_task'], 'small_task': record['small_task'],
                'source_identity': record['source_identity'], 'source_json': str(source),
                'scene_identity': definition.id, 'scene_file': str(scene_path), 'task_file': record.get('task_file'),
                'physical_task_id': None, 'expected_views': views, 'diagnostic_pending': not preserved_reference,
                'conversion_status': 'pending', 'question': raw.get('question', ''), 'source_only': value['source_only'],
                'known_limitations': record['limitations'], 'reuse_level': record['reuse_level'],
                'converter_version': record['converter_version'], 'timeout_s': timeout_s, 'max_attempts': max_attempts,
                'accepted_at': time.time(), 'acceptance_note': 'Automatic category static validation; human and native qualification remain pending.'}
            if record.get('task_file'):
                manifest.update(snapshot_task_file=str(job / 'input/task.yaml'), task_sha256=input_hashes['task.yaml'])
            if members is not None:
                manifest['capture_source_members'] = [{**member,
                    'snapshot_source_file': str(job / 'input/data' / member['source_identity'])} for member in members]
            if campaign_binding is not None:
                manifest['task_edit2_capture_binding'] = campaign_binding
            for key in ('capture_initial_agent_bundle', 'task_edit2_capture_campaign', 'task_edit2_variant_key'):
                if key in record:
                    manifest[key] = record[key]
            write_json(snapshot / 'manifest.json', manifest)
            write_json(temporary / 'accepted.json', {'manifest_sha256': sha256(snapshot / 'manifest.json'),
                'candidate_id': record['candidate_id'], 'capture_signature': signature,
                'code_commit': frozen['code_commit'], 'esi_commit': frozen['esi_commit']})
            try:
                temporary.rename(job)
            except OSError:
                if not (job / 'input/manifest.json').is_file():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        queue_job = queue.job_dir(job_id)
        if queue_job != job and not queue_job.exists():
            try:
                queue_job.symlink_to(job, target_is_directory=True)
            except FileExistsError:
                pass
        verify_snapshot(queue.manifest(job_id))
        queue.enqueue(queue.manifest(job_id))
        queue.prioritize(job_id, priority)
        value.update(job_id=job_id, code_commit=frozen['code_commit'], capture_signature=signature,
                     capture_input=str(job / 'input/manifest.json'))
    except Exception as error:
        value.update(no_native_image=True, capture_submission_error=f'{type(error).__name__}: {error}',
                     capture_submission_stage='code_snapshot' if snapshot_error else 'capture_submission',
                     diagnostic_only=True)
        value.setdefault('warning_codes', []).append('no_native_image')
    write_json(path, value)
    return value


def publish_review(manifest, queue_root, *, selected=None, frozen=None, **kwargs):
    chosen = set(selected) if selected is not None else None
    records = [r for r in manifest['records'] if chosen is None or r['candidate_id'] in chosen]
    # Snapshotting failure is recorded per instance, so publication keeps moving.
    if frozen is None and records and not kwargs.get('snapshot_error'):
        try:
            frozen = freeze_working_code(queue_root)
        except Exception:
            frozen = None
    return [publish_record(r, queue_root, frozen=frozen, **kwargs) for r in records]


def refresh_original_reference(manifest, queue_root):
    """Request one render-only original-source diagnostic in an already resident world.

    This never switches tasks, closes worlds, changes objects or launches GPU
    programs. The original worker validates its own immutable job/signature.
    Newly requested PNGs are written directly to home through a new symlink.
    """
    queue = CaptureQueue(queue_root)
    sources = {r['source_identity']: r['source_sha256'] for r in manifest['records']}
    storage = read_json(queue.root / 'category_storage.json', {})
    artifact_root = Path(storage.get('artifact_root', queue.root))
    for path in sorted((queue.root / 'jobs').glob('*/input/manifest.json')):
        old = read_json(path)
        if old.get('source_identity') not in sources or old.get('variant') != 'original':
            continue
        if sha256(old['snapshot_source_file']) != sources[old['source_identity']]:
            continue
        state = queue.status(old['job_id']) or {}
        output = Path(state.get('native_output', queue.root / 'missing'))
        world = read_json(output / 'retained_world.json', {})
        if (world.get('state') != 'RETAINED' or time.time() - world.get('heartbeat', 0) > 20
                or world.get('max_task_translation_change_m', 1) > 1e-6):
            continue
        request_id = 'category_original_' + digest([manifest['big_task'], manifest['small_task'],
            manifest['converter_version'], old['source_identity']])[:24]
        destination = artifact_root / 'refreshed_original' / request_id
        destination.mkdir(parents=True, exist_ok=True)
        alias = output / 'retained_captures' / request_id
        alias.parent.mkdir(exist_ok=True)
        if not alias.exists():
            alias.symlink_to(destination, target_is_directory=True)
        request = {'job_id': old['job_id'], 'capture_signature': old['capture_signature'], 'operation': 'capture'}
        publish_json(output / 'retained_requests' / (request_id + '.json'), request)
        return {'source_identity': old['source_identity'], 'original_job_id': old['job_id'],
            'request_file': str(output / 'retained_requests' / (request_id + '.json')),
            'response_file': str(output / 'retained_responses' / (request_id + '.json')),
            'output': str(destination), 'kind': 'source_only',
            'note': 'New render of an original resident source; no converted-task, physical or current-code qualification.'}
    return None


def retain_reference_evidence(manifest, queue_root, reference_queues):
    """Persist labelled source-matched original/failure evidence without quadratic scans."""
    available = set()
    wanted = {record['source_identity'] for record in manifest['records']}
    for root in reference_queues:
        queue = CaptureQueue(root)
        for source, jobs in reference_job_index(root).items():
            if source in wanted and any((queue.status(job) or {}).get('state') in (
                    'REVIEW_PENDING', 'CACHED', 'FAILED', 'STALE') for job in jobs):
                available.add(source)
    for record in manifest['records']:
        if record['source_identity'] not in available:
            continue
        path = Path(queue_root) / 'category_records' / (record['candidate_id'] + '.json')
        published = read_json(path)
        if published is None:
            continue
        diagnostic = collect_diagnostics(published, queue_root, reference_queues=reference_queues)
        published.update(diagnostic_images=diagnostic['diagnostic_images'], diagnostic_logs=diagnostic['diagnostic_logs'])
        write_json(path, published)


def retain_conversion_history(manifest, queue_root, batch_root):
    """Show genuine superseded conversion failures on the current instance's Diagnostics tab."""
    by_source = {}
    for path in sorted((Path(batch_root) / 'iterations').glob('*manifest.json')):
        old = read_json(path, {})
        for row in old.get('records', []):
            if row['status'] == 'generated':
                continue
            by_source.setdefault(row['source_identity'], []).append({
                'candidate_id': row['candidate_id'], 'converter_version': row['converter_version'],
                'failure_stage': row.get('failure_stage'), 'failure_reason': row.get('failure_reason'),
                'evidence_file': str(path), 'diagnostic_record': str(Path(row['diagnostic_dir']) / 'record.json'),
                'superseded': True, 'note': 'A real earlier CPU conversion failure; not a native failure or current result.'})
    for row in manifest['records']:
        if row['source_identity'] in by_source:
            path = Path(queue_root) / 'category_records' / (row['candidate_id'] + '.json')
            record = read_json(path)
            if record is not None:
                record['prior_conversion_failures'] = by_source[row['source_identity']]
                record['diagnostic_logs'] = sorted(set(record.get('diagnostic_logs', []) + [
                    r['diagnostic_record'] for r in by_source[row['source_identity']] if Path(r['diagnostic_record']).is_file()]))
                write_json(path, record)


def collect_diagnostics(record, queue_root, *, reference_queues=()):
    """Keep complete native, partial, source-only and historical images clearly distinct."""
    queue = CaptureQueue(queue_root)
    state = queue.status(record['job_id']) if record.get('job_id') else None
    state = state or {'state': 'REVIEW_ONLY'}
    if record.get('capture_submission_error'):
        state = {**state, 'failure_stage': record.get('capture_submission_stage', 'capture_submission'),
                 'error': record['capture_submission_error']}
    receipt = read_json(state.get('receipt', queue.root / 'missing'), {})
    complete = bool(record.get('job_id') and complete_images(queue.manifest(record['job_id']), receipt))
    images, logs = [], []
    if complete:
        images += [{'path': path, 'view': view,
            'kind': 'human_floorplan' if view == 'global_floorplan' else 'source_only' if record['source_only'] else 'native',
            'label': 'Room map (segmentation; not a native render)' if view == 'global_floorplan' else
                'Source-only diagnostic' if record['source_only'] else 'Native candidate'}
            for view, path in receipt.get('images', {}).items()]
    if record.get('job_id'):
        for attempt in sorted((queue.job_dir(record['job_id']) / 'attempts').glob('*')):
            for path in attempt.rglob('*.log'):
                logs.append(str(path))
            if not complete or attempt != Path(state.get('native_output', '')):
                for path in attempt.rglob('*.png'):
                    images.append({'path': str(path), 'view': path.stem,
                        'kind': 'human_floorplan' if path.stem == 'global_floorplan' else 'partial',
                        'label': 'Room map (segmentation; not a native render)' if path.stem == 'global_floorplan'
                            else 'Partial capture from ' + attempt.name})
    # Historical source-matched images are evidence, never an eligible current preview.
    for root in reference_queues:
        old = CaptureQueue(root)
        for job_id in sorted(reference_job_index(root).get(record['source_identity'], ())):
            if old.root == queue.root and job_id == record.get('job_id'):
                continue  # The current job is already represented as native/partial evidence.
            manifest = old.manifest(job_id)
            if manifest['source_identity'] != record['source_identity']:
                continue
            old_state = old.status(manifest['job_id']) or {}
            old_receipt = read_json(old_state.get('receipt', old.root / 'missing'), {})
            # New render-only diagnostics still depict the original source,
            # never the adapted task or its terminal physical success.
            output = Path(old_state.get('native_output', old.root / 'missing'))
            recaptures = sorted((output / 'retained_responses').glob('category_original_*.json'))
            for response_path in recaptures:
                response = read_json(response_path)
                refreshed = read_json(response.get('receipt', old.root / 'missing'), {})
                if response.get('state') == 'CAPTURED' and complete_images(manifest, refreshed):
                    images += [{'path': str(Path(p).resolve()), 'view': v, 'kind': 'source_only',
                        'label': 'Original source diagnostic · render-only refresh',
                        'capture_job_id': manifest['job_id']} for v, p in refreshed['images'].items()]
            if complete_images(manifest, old_receipt):
                for view, source_path in old_receipt['images'].items():
                    source_path = Path(source_path).resolve()
                    storage = read_json(queue.root / 'category_storage.json', {})
                    artifact_root = Path(storage.get('artifact_root', queue.root)).resolve()
                    destination = source_path
                    if not source_path.is_relative_to(artifact_root):
                        destination = artifact_root / 'reference_evidence' / record['candidate_id'] / manifest['job_id'] / (view + '.png')
                        if not destination.is_file():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(source_path, destination)
                    images.append({'path': str(destination), 'view': view, 'kind': 'historical',
                        'label': 'Historical source reference', 'capture_job_id': manifest['job_id']})
            else:
                for attempt in sorted((old.job_dir(manifest['job_id']) / 'attempts').glob('*')):
                    logs.extend(str(p) for p in attempt.rglob('*.log'))
    from PIL import Image
    valid = []
    for item in images:
        try:
            with Image.open(item['path']) as pixels:
                pixels.verify()
            valid.append(item)
        except (OSError, ValueError, SyntaxError):
            continue
    progress = read_json(Path(state.get('native_output', queue.root / 'missing')) / 'progress.json', {})
    failure = state['state'] in ('FAILED', 'STALE', 'CANCELLED')
    instance_status = record['status']
    if failure:
        initialized = Path(state.get('native_output', queue.root / 'missing')) / 'initialization_complete.json'
        instance_status = 'capture_failed' if initialized.is_file() else 'native_load_failed'
    elif record.get('capture_submission_error') and record['status'] == 'generated':
        instance_status = 'capture_failed'
    elif complete and record['status'] == 'generated':
        instance_status = 'capture_complete'
    logs += [str(p) for p in Path(record['diagnostic_dir']).glob('*.log')]
    return {'capture': {**{k: state.get(k) for k in ('state', 'attempts', 'failure_stage', 'error', 'duration_s')},
                       'job_id': record.get('job_id'), 'progress': progress, 'claim': state.get('claim'),
                       'timings': receipt.get('timings'), 'interaction_setup': receipt.get('interaction_setup'),
                       'human_camera_framing': receipt.get('human_camera_framing'),
                       'human_floorplan': receipt.get('human_floorplan')},
            'instance_status': instance_status, 'diagnostic_images': valid,
            'diagnostic_logs': sorted(set(logs)), 'no_native_image': not any(i['kind'] == 'native' for i in valid),
            'native_complete': complete, 'receipt': receipt if complete else {}}

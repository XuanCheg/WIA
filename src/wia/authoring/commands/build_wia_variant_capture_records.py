"""Prepare definition-only captures for an explicit, versioned WIA manifest.

Run: python -B scripts/build_wia_variant_capture_records.py --manifest PATH \
  --output PATH [--check]
Paths declared by entries are repository-relative or absolute. Existing outputs
must match exactly; choose a new output for changed definitions. No queue, code
snapshot, GPU capture, reference execution or publication is created here.


Canonical invocation: python -m wia.authoring.commands.build_wia_variant_capture_records --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT, project_path

from wia.authoring.io import digest, read_json, sha256
from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import SceneRuntime, Source
from wia.tasks.sources import resolve_source

REVISION = 'wia_variant_capture_records_v1'
REGISTRY = ROOT / 'explore_to_achieve/wia_v1/registry.json'
MANIFEST_SCHEMA = 'wia_variant_manifest_v1'


def _path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Manifest definition paths must be nonempty strings.')
    path = Path(value).expanduser()
    return project_path(path, root=ROOT)


def _hash(entry, field, actual):
    if field in entry and entry[field] != actual:
        raise ValueError('Manifest hash mismatch: ' + field)


def _source_member(identity, path=None):
    Source.model_validate({'provider': 'esi', 'json': identity})
    if str(Path(identity)) != identity:
        raise ValueError('Capture source identity must use its canonical five-part path.')
    source = Path(path).resolve() if path is not None else resolve_source(identity)
    return {'source_identity': identity, 'source_file': str(source), 'source_sha256': sha256(source)}


def _categories(member):
    raw = read_json(member['source_file'])
    categories = list(Path(member['source_identity']).parts[:2])
    if [raw.get('big_task'), raw.get('small_task')] != categories:
        raise ValueError('Loaded source categories must match its actual source identity.')
    return {'big_task': categories[0], 'small_task': categories[1]}


def _modules(value):
    """Retain nested evaluator dependencies required by capture publication."""
    modules = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('module', 'converter_module', 'evaluator_module') and isinstance(item, str):
                if item.startswith('wia.authoring.converters.'):
                    modules.add(item)
            modules.update(_modules(item))
    elif isinstance(value, list):
        for item in value:
            modules.update(_modules(item))
    return modules


def build_records(manifest_path):
    """Load exactly the declared variants and retain actual source provenance."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = read_json(manifest_path)
    if manifest.get('schema') != MANIFEST_SCHEMA:
        raise ValueError('Expected manifest schema ' + MANIFEST_SCHEMA)
    entries = manifest.get('entries')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest entries must be a nonempty list.')
    registry = read_json(REGISTRY)
    manifest_hash, registry_hash = sha256(manifest_path), sha256(REGISTRY)
    items, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Manifest entries must be objects.')
        for field in ('document_alias', 'variant_id', 'label', 'variant'):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError('Manifest entry requires nonempty ' + field)
        alias, variant_id, variant = (entry[field] for field in ('document_alias', 'variant_id', 'variant'))
        if (alias, variant_id) in seen:
            raise ValueError('Duplicate document_alias and variant_id pair.')
        seen.add((alias, variant_id))
        rows = [row for row in registry['rows'] if row['document_alias'] == alias]
        if len(rows) != 1:
            raise ValueError('Manifest alias must resolve to exactly one frozen registry row: ' + alias)
        row = rows[0]
        definitions = {}
        for field in ('task', 'scene', 'parent_task', 'parent_scene'):
            path = _path(entry.get(field + '_file'))
            file_hash = sha256(path)
            _hash(entry, field + '_sha256', file_hash)
            definitions[field + '_file'] = str(path)
            definitions[field + '_sha256'] = file_hash
        task = load_task(definitions['task_file'], variant=variant, run_mode='diagnostic')
        scene = load_scene(definitions['scene_file'], variant=variant, run_mode='diagnostic')
        if getattr(task.definition, 'kind', None) != 'category_robot_task':
            raise ValueError('Variant capture requires an authored category_robot_task contract.')
        if (task.seed != scene.seed
                or task.definition.scene.variants[variant] != scene.definition.scene.variants[variant]
                or task.definition.private.bindings != scene.definition.bindings):
            raise ValueError('Selected task and scene patches, bindings and seed must match.')
        runtime = task.definition.repair.get('runtime')
        if runtime is not None and SceneRuntime.model_validate(runtime) != scene.definition.runtime:
            raise ValueError('Selected task and scene runtime must match.')
        if 'seed' in entry and entry['seed'] != scene.seed:
            raise ValueError('Manifest seed must match the selected task and scene.')
        evaluation = task.definition.private.evaluation
        parameters = getattr(evaluation, 'parameters', {})
        task_member = _source_member(task.definition.source.json_path, task.source_path)
        scene_member = _source_member(scene.definition.source.json_path, scene.source_path)
        evaluation_members = []
        for declared_member in parameters.get('members', []):
            member = _source_member(declared_member['source_identity'])
            _hash(declared_member, 'source_sha256', member['source_sha256'])
            evaluation_members.append(member)
        members = {}
        for member in [task_member, scene_member, *evaluation_members]:
            identity = member['source_identity']
            if identity in members and members[identity] != member:
                raise ValueError('A source identity resolves to inconsistent loaded source bytes or paths.')
            members[identity] = member
        members = [members[identity] for identity in sorted(members)]
        _hash(entry, 'task_source_sha256', task.source_sha256)
        _hash(entry, 'scene_source_sha256', scene.source_sha256)
        _hash(entry, 'source_sha256', scene.source_sha256)
        source_categories = _categories(scene_member)
        task_categories = _categories(task_member)
        for member in members:
            if sha256(member['source_file']) != member['source_sha256']:
                raise RuntimeError('Capture source changed during record preparation.')
        task.verify_unchanged()
        scene.verify_unchanged()
        for field in ('task', 'scene', 'parent_task', 'parent_scene'):
            if sha256(definitions[field + '_file']) != definitions[field + '_sha256']:
                raise RuntimeError('Definition changed during record preparation: ' + field)
        # Entry bytes expressed as JSON values keep declared provenance and any
        # scientific annotations in identity without treating them as evidence.
        identity = {'wia_id': row['wia_id'], 'proposal_canonical_id': row['canonical_candidate_id'],
            'document_alias': alias, 'variant_id': variant_id, 'manifest_entry': deepcopy(entry),
            'definitions': definitions, 'variant': variant, 'seed': scene.seed,
            'source_members': deepcopy(members)}
        key = digest(identity)[:24]
        candidate_id = 'category_' + key
        revision = parameters.get('revision', parameters.get('task_revision', parameters.get('scoring_version', REVISION)))
        declarations = {'source_identity': scene_member['source_identity'],
            'module': getattr(evaluation, 'module', None), 'revision': revision,
            'scene_model': scene.scene_name, 'room': scene.room_name}
        for field, actual in declarations.items():
            if field in entry and entry[field] != actual:
                raise ValueError('Manifest declaration does not match loaded definition: ' + field)
        if 'source_file' in entry and _path(entry['source_file']) != scene.source_path.resolve():
            raise ValueError('Manifest source_file does not match actual loaded source.')
        record = {
            'candidate_id': candidate_id, 'proposal_canonical_id': row['canonical_candidate_id'],
            'wia_id': row['wia_id'], 'document_alias': alias, 'wia_variant_id': variant_id,
            'primary_family': row['primary_family'], 'primary_subclass': row['primary_subclass'],
            'source_category_is_task_family': False, 'source_role': 'loaded_scene_background',
            'has_prior_selected_definition': bool(row.get('task')),
            'registry_parent_task_file': (row.get('task') or {}).get('task_file'),
            'registry_parent_scene_file': row.get('scene', {}).get('selected_scene_file'),
            'status': 'generated', 'definition_only': True, 'evidence_scope': 'definition_only',
            'source_identity': scene_member['source_identity'], 'original_source': scene_member['source_file'],
            'source_sha256': scene.source_sha256, 'capture_source_members': members,
            'task_source_identity': task_member['source_identity'], 'task_source_sha256': task.source_sha256,
            'scene_source_identity': scene_member['source_identity'], 'scene_source_sha256': scene.source_sha256,
            'task_source_categories': task_categories, 'scene_source_categories': source_categories,
            **source_categories, **definitions, 'seed': scene.seed, 'variant': variant, 'reference_only': False,
            'converter_module': getattr(evaluation, 'module', None), 'converter_version': revision,
            'composite_member_modules': sorted(_modules(evaluation.model_dump(mode='json'))
                                               - {getattr(evaluation, 'module', None)}),
            'evaluation_revision': revision, 'record_builder_revision': REVISION,
            'reuse_level': 'explicit_versioned_authored_variant',
            'capture_initial_agent_bundle': True, 'task_edit2_capture_campaign': True,
            'task_edit2_variant_key': key, 'wia_authored_capture': True,
            'robot_task_public': task.definition.public.model_dump(mode='json'),
            'diagnostic_dir': str(ARTIFACT_ROOT / 'wia_variant_capture_diagnostics' / key),
            'warning_codes': ['native_variant_capture_not_run', 'scientific_controls_unverified'],
            'native_validation_run': False, 'scientific_controls_verified': False, 'success_claimed': False,
            'reference_execution_status': 'deferred_until_user_review', 'definition_identity': identity,
            'limitations': [
                'Definition-only preparation establishes no native capture, task success or scientific control.',
                'Loaded source identities and categories are separate from the frozen WIA primary family.',
                'Request eight initial mounted views and the frozen initial overview with the actual public task.',
                'Full legal robot reference execution is deferred until user review.',
            ],
        }
        if not row['canonical_candidate_id'].startswith('plan_'):
            record['corev2_candidate_id'] = row['canonical_candidate_id']
        record['revision'] = digest(record)
        items.append({'row_id': row['wia_id'], 'wia_id': row['wia_id'], 'document_alias': alias,
            'variant_id': variant_id, 'label': entry['label'], 'canonical_candidate_id': row['canonical_candidate_id'],
            'capture_candidate_id': candidate_id, **definitions, 'variant': variant, 'seed': scene.seed,
            'source_identity': scene_member['source_identity'], 'source_file': scene_member['source_file'],
            'source_sha256': scene.source_sha256, 'record': record})
    if len({item['capture_candidate_id'] for item in items}) != len(items):
        raise ValueError('Variant captures require distinct capture identities.')
    if sha256(manifest_path) != manifest_hash or sha256(REGISTRY) != registry_hash:
        raise RuntimeError('Manifest or registry changed during record preparation.')
    return {'schema': REVISION, 'status': 'definition_only', 'record_count': len(items),
        'proposal_count': len({item['document_alias'] for item in items}),
        'manifest_file': str(manifest_path), 'manifest_sha256': manifest_hash,
        'registry_file': str(REGISTRY), 'registry_sha256': registry_hash,
        'published': False, 'queue_mutated': False, 'native_capture_run': False,
        'success_claimed': False, 'scientific_controls_verified': False,
        'reference_execution_status': 'deferred_until_user_review', 'items': items}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--check', action='store_true', help='Compare exact deterministic bytes without writing.')
    args = parser.parse_args(argv)
    package = build_records(args.manifest)
    rendered = (json.dumps(package, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if args.output.exists():
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            print(f'WIA variant output collision: {args.output}; select a new output.', file=sys.stderr)
            return 1
    elif args.check:
        print(f'WIA variant capture records are missing: {args.output}', file=sys.stderr)
        return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            handle.write(rendered)
    print(f'WIA variant capture records are current: {args.output} ({len(package["items"])} definition-only records)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

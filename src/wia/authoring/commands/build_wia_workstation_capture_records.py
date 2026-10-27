"""Prepare 26 authored workstation capture records without publishing jobs.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B scripts/build_wia_workstation_capture_records.py [--check]
Use --dimensional-manifest PATH --mirror-manifest PATH --output PATH to select
another explicit pair. The default V1 output remains unchanged.
No code freezing, queue mutation, native capture, model or reference run occurs.


Canonical invocation: python -m wia.authoring.commands.build_wia_workstation_capture_records --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT

from wia.authoring.io import digest, read_json, sha256, write_json
from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import SceneRuntime
from wia.tasks.sources import resolve_source

REVISION = 'wia_workstation_capture_records_v1'
WIA_ROOT = ROOT / 'explore_to_achieve/wia_v1'
REGISTRY = WIA_ROOT / 'registry.json'
OUTPUT = ARTIFACT_ROOT / 'wia_v1_20261023/workstation_capture_records.json'
MANIFESTS = (WIA_ROOT / 'dimensional_v1/manifest.json', WIA_ROOT / 'mirror_v1/manifest.json')
MANIFEST_CONTRACTS = {
    'wia_dimensional_manifest_v1': ('core.esi_converters.wia_dimensional',
                                     ('wia_dimensional_v1', 'wia_dimensional_v2')),
    'wia_mirror_manifest_v1': ('core.esi_converters.wia_mirror', ('wia_mirror_v1', 'wia_mirror_v2')),
}
ALIASES = ('D1-1', 'D1-2', 'D1-3', 'D2-1', 'D2-2', 'R-C1', 'R-C2', 'R-C3', 'R-A1', 'R-A2', 'R-S1', 'R-S2', 'R-S3')
SOURCE_IDENTITY = 'Specular Reflection/Correspondence/hall_glass_ceiling/empty_room_0/q_007.json'


def manifest_member(manifest_path, value):
    """Resolve declared members relative to their own manifest, without scanning."""
    relative = Path(value)
    path = (manifest_path.parent / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError('Authored workstation paths must stay relative to their manifest directory.')
    return path


def build_records(manifests=MANIFESTS):
    """Keep proposal, public task and future capture identities explicitly separate."""
    registry = read_json(REGISTRY)
    source_file = resolve_source(SOURCE_IDENTITY)
    source_hash = sha256(source_file)
    source_metadata = read_json(source_file)
    source_categories = tuple(Path(SOURCE_IDENTITY).parts[:2])
    if (source_metadata['big_task'], source_metadata['small_task']) != source_categories:
        raise ValueError('Bootstrap source category metadata must match its real source identity.')
    items, inputs, families = [], [], set()
    for manifest_path in manifests:
        manifest_path = Path(manifest_path).resolve()
        manifest = read_json(manifest_path)
        manifest_hash = sha256(manifest_path)
        schema = manifest.get('schema')
        contract = MANIFEST_CONTRACTS.get(schema)
        if contract is None or manifest.get('revision') not in contract[1]:
            raise ValueError('Workstation captures require an exact supported family manifest schema and V1/V2 revision.')
        if schema in families:
            raise ValueError('Select exactly one dimensional manifest and one mirror manifest.')
        families.add(schema)
        expected_module = contract[0]
        if (manifest['artifact_path_base'] != 'manifest_directory' or manifest['source']['identity'] != SOURCE_IDENTITY
                or manifest['source']['sha256'] != source_hash):
            raise ValueError('The authored manifest must retain the exact background-only bootstrap source.')
        inputs.append({'file': str(manifest_path), 'sha256': manifest_hash})
        for entry in manifest['entries']:
            alias, seed = entry['proposal_alias'], entry['seed']
            rows = [row for row in registry['rows'] if row.get('document_alias') == alias]
            if len(rows) != 1 or alias not in ALIASES or seed not in (101, 202):
                raise ValueError('Every authored version requires one of the thirteen frozen proposal mappings.')
            row = rows[0]
            if row.get('task') is not None or row.get('scene', {}).get('selected_scene_file') is not None:
                raise ValueError('These authored proposals must not be represented as a prior selected CoreV2 task.')
            if not row['canonical_candidate_id'].startswith('plan_'):
                raise ValueError('The canonical registry identity must remain a proposal identity.')
            task_path, scene_path = (manifest_member(manifest_path, entry[kind + '_file']) for kind in ('task', 'scene'))
            for kind, path in (('task', task_path), ('scene', scene_path)):
                if sha256(path) != entry[kind + '_sha256']:
                    raise ValueError('Authored manifest definition hash mismatch: ' + str(path))
            if entry['variant'] != 'adapted':
                raise ValueError('Workstation capture requires the declared adapted variant.')
            task = load_task(task_path, variant='adapted', run_mode='diagnostic')
            scene = load_scene(scene_path, variant='adapted', run_mode='diagnostic')
            if (task.definition.kind != 'category_robot_task' or task.seed != scene.seed or task.seed != seed
                    or task.definition.scene != scene.definition.scene or task.definition.private.bindings != scene.definition.bindings
                    or SceneRuntime.model_validate(task.definition.repair['runtime']) != scene.definition.runtime):
                raise ValueError('The selected task and scene bindings, runtime, seed and full patches must match.')
            if (task.definition.source.json_path != SOURCE_IDENTITY or scene.definition.source.json_path != SOURCE_IDENTITY
                    or task.source_path != source_file or scene.source_path != source_file
                    or task.source_sha256 != source_hash or scene.source_sha256 != source_hash):
                raise ValueError('Task and scene must load the exact immutable bootstrap file.')
            evaluation = task.definition.private.evaluation
            if evaluation.parameters['revision'] != manifest['revision']:
                raise ValueError('The authored evaluator revision must match its manifest.')
            if evaluation.module != expected_module:
                raise ValueError('The authored native evaluator module must remain exact.')
            task.verify_unchanged()
            scene.verify_unchanged()
            variant_id = task.path.stem
            members = [{'source_identity': SOURCE_IDENTITY, 'source_file': str(source_file), 'source_sha256': source_hash}]
            identity = {'wia_id': row['wia_id'], 'proposal_alias': alias, 'seed': seed,
                'proposal_canonical_id': row['canonical_candidate_id'], 'authored_variant_id': variant_id,
                'variant': 'adapted', 'task_sha256': task.file_sha256, 'scene_sha256': scene.file_sha256,
                'authored_manifest_sha256': manifest_hash, 'source_identity': SOURCE_IDENTITY, 'source_sha256': source_hash}
            key = digest(identity)[:24]
            candidate_id = 'category_' + key
            record = {
                'candidate_id': candidate_id, 'proposal_canonical_id': row['canonical_candidate_id'],
                'wia_id': row['wia_id'], 'document_alias': alias, 'wia_variant_id': variant_id,
                'primary_family': row['primary_family'], 'primary_subclass': row['primary_subclass'],
                'source_category_is_task_family': False, 'source_role': 'background_loader_only',
                'source_role_description': manifest['source']['role'], 'has_prior_selected_definition': False,
                'status': 'generated', 'definition_only': True, 'evidence_scope': 'definition_only',
                'source_identity': SOURCE_IDENTITY, 'original_source': str(source_file), 'source_sha256': source_hash,
                'capture_source_members': members, 'task_source_identity': SOURCE_IDENTITY,
                'task_source_sha256': task.source_sha256, 'scene_source_identity': SOURCE_IDENTITY,
                'scene_source_sha256': scene.source_sha256, 'big_task': source_categories[0], 'small_task': source_categories[1],
                'task_file': str(task.path), 'task_sha256': task.file_sha256,
                'scene_file': str(scene.path), 'scene_sha256': scene.file_sha256,
                'seed': seed, 'variant': 'adapted', 'reference_only': False,
                'converter_module': evaluation.module, 'converter_version': evaluation.parameters['revision'],
                'evaluation_revision': evaluation.parameters['revision'], 'record_builder_revision': REVISION,
                'reuse_level': 'authored_workstation_background_bootstrap_only',
                'capture_initial_agent_bundle': True, 'task_edit2_capture_campaign': True,
                'task_edit2_variant_key': key, 'wia_authored_capture': True,
                'robot_task_public': task.definition.public.model_dump(mode='json'),
                'diagnostic_dir': str(OUTPUT.parent / 'workstation_capture_diagnostics' / key),
                'warning_codes': ['native_workstation_capture_not_run', 'scientific_controls_unverified'],
                'native_validation_run': False, 'scientific_controls_verified': False, 'success_claimed': False,
                'reference_execution_status': manifest['full_reference_execution'], 'definition_identity': identity,
                'limitations': [
                    'Definition-only capture preparation; no published job, native screenshot, physical success or scientific control is verified.',
                    'The immutable optical source is a background loader only; its original inventory and QA are replaced by the authored task.',
                    'Capture metadata retains the true source category; WIA family and subclass come independently from the frozen registry.',
                    'Request the full eight mounted views and frozen initial overview when capture is authorized.',
                    'Full legal robot reference execution is deferred until user review.',
                ],
            }
            record['revision'] = digest(record)
            items.append({'row_id': row['wia_id'], 'wia_id': row['wia_id'], 'document_alias': alias,
                'variant_id': variant_id, 'label': alias + ' · seed ' + str(seed) + ' · authored ' + manifest['revision'].rsplit('_', 1)[1],
                'canonical_candidate_id': row['canonical_candidate_id'], 'capture_candidate_id': candidate_id,
                'primary_family': row['primary_family'], 'primary_subclass': row['primary_subclass'],
                'task_file': str(task.path), 'task_sha256': task.file_sha256,
                'scene_file': str(scene.path), 'scene_sha256': scene.file_sha256,
                'source_identity': SOURCE_IDENTITY, 'source_file': str(source_file), 'source_sha256': source_hash,
                'task_source_sha256': task.source_sha256, 'variant': 'adapted', 'seed': seed, 'record': record})
    if families != set(MANIFEST_CONTRACTS):
        raise ValueError('Select exactly one dimensional manifest and one mirror manifest.')
    items.sort(key=lambda item: (ALIASES.index(item['document_alias']), item['seed']))
    if (len(items) != 26 or len({item['capture_candidate_id'] for item in items}) != 26
            or {(item['document_alias'], item['seed']) for item in items} != {(alias, seed) for alias in ALIASES for seed in (101, 202)}):
        raise ValueError('The exact thirteen proposals require two distinct capture identities each.')
    return {'schema': REVISION, 'status': 'definition_only', 'proposal_count': 13, 'record_count': 26,
        'registry_file': str(REGISTRY), 'registry_sha256': sha256(REGISTRY), 'authored_manifests': inputs,
        'published': False, 'queue_mutated': False, 'native_capture_run': False, 'success_claimed': False,
        'scientific_controls_verified': False, 'reference_execution_status': 'deferred_until_user_review', 'items': items}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimensional-manifest', type=Path, default=MANIFESTS[0], help='Explicit dimensional V1 or V2 manifest.')
    parser.add_argument('--mirror-manifest', type=Path, default=MANIFESTS[1], help='Explicit mirror V1 or V2 manifest.')
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Write only this capture-record package.')
    parser.add_argument('--check', action='store_true', help='Compare deterministic bytes without writing.')
    args = parser.parse_args(argv)
    package = build_records((args.dimensional_manifest, args.mirror_manifest))
    rendered = (json.dumps(package, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if args.check:
        if not args.output.is_file() or args.output.read_bytes() != rendered:
            print(f'WIA workstation capture records differ or are missing: {args.output}', file=sys.stderr)
            return 1
    elif not args.output.is_file() or args.output.read_bytes() != rendered:
        write_json(args.output, package)
    print(f'WIA workstation capture records are current: {args.output} (13 proposals, 26 definition-only records)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

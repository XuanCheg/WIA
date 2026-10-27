"""Prepare exactly four authored WIA capture records without publishing jobs.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B scripts/build_wia_authored_capture_records.py [--check]
Only outputs/wia_v1_20261023/authored_capture_records.json is written. This
CPU-only builder never freezes code, mutates a queue or starts native capture.


Canonical invocation: python -m wia.authoring.commands.build_wia_authored_capture_records --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT, project_path

from wia.authoring.io import digest, read_json, sha256, write_json
from wia.tasks.loader import load_scene, load_task
from wia.tasks.sources import resolve_source

REVISION = 'wia_authored_capture_records_v1'
WIA_ROOT = ROOT / 'explore_to_achieve/wia_v1'
REGISTRY = WIA_ROOT / 'registry.json'
COREV2 = ROOT / 'explore_to_achieve/lists/corev2.json'
OUTPUT = ARTIFACT_ROOT / 'wia_v1_20261023/authored_capture_records.json'
ALIASES = ('T11', 'T22', 'T23', 'T24')


def _one(rows, field, value):
    selected = [row for row in rows if row.get(field) == value]
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one {field}={value} selection.')
    return selected[0]


def _path(value):
    path = Path(value)
    return project_path(path, root=ROOT)


def build_records():
    """Reuse the missing-capture record contract with explicit authored variants.

    The old submit helper selects tabletop by the legacy task kind. These new
    Rigid tasks are category tasks, so their tabletop variant is selected here
    before either production loader runs. No historical capture is re-certified.
    """
    registry, corev2 = read_json(REGISTRY), read_json(COREV2)
    regional_path, rigid_path = (WIA_ROOT / name / 'manifest.json'
        for name in ('regional_t11_v1', 'rigid_family_v1'))
    regional, rigid = read_json(regional_path), read_json(rigid_path)
    items = []
    for alias in ALIASES:
        row = _one(registry['rows'], 'document_alias', alias)
        original = _one(corev2['tasks'], 'candidate_id', row['canonical_candidate_id'])
        if (_path(original['task_file']) != _path(row['task']['task_file'])
                or _path(original['scene_file']) != _path(row['scene']['selected_scene_file'])):
            raise ValueError('The registry must retain its original CoreV2 selection mapping.')
        if alias == 'T11':
            authored, manifest_path = regional, regional_path
            task_path, scene_path = WIA_ROOT / 'regional_t11_v1/task.yaml', WIA_ROOT / 'regional_t11_v1/scene.yaml'
            variant, label = None, 'WIA T11 · Neutral room labels v1'
        else:
            authored, manifest_path = _one(rigid['tasks'], 'document_alias', alias), rigid_path
            stem = 'rigid_' + alias.lower() + '_v1.yaml'
            task_path, scene_path = WIA_ROOT / 'rigid_family_v1/tasks' / stem, WIA_ROOT / 'rigid_family_v1/scenes' / stem
            variant, label = 'tabletop', 'WIA ' + alias + ' · Scored Rigid containment v1'
        for kind, path in (('task', task_path), ('scene', scene_path)):
            if _path(authored[kind + '_file']) != path.resolve() or authored[kind + '_sha256'] != sha256(path):
                raise ValueError('Authored manifest and current versioned definition must match exactly.')
            parent = row['task']['task_file'] if kind == 'task' else row['scene']['selected_scene_file']
            if _path(authored['parent_' + kind + '_file']) != _path(parent):
                raise ValueError('Authored provenance must retain the registry-selected parent definition.')
            if authored['parent_' + kind + '_sha256'] != sha256(_path(parent)):
                raise ValueError('The registry-selected parent definition changed after authoring.')
        if variant is None:
            # The T11 scene itself declares which non-original initialization is available.
            from wia.tasks.schema import SceneDefinition
            import yaml
            definition = SceneDefinition.model_validate(yaml.safe_load(scene_path.read_text(encoding='utf-8')))
            variant = 'adapted' if 'adapted' in definition.scene.variants else 'original'
        scene = load_scene(scene_path, variant=variant, run_mode='diagnostic')
        task = load_task(task_path, variant=variant, run_mode='diagnostic')
        if task.definition.kind != 'category_robot_task':
            raise ValueError('All four authored versions must retain their scored category task contract.')
        if (task.seed != scene.seed or task.definition.scene.variants[variant] != scene.definition.scene.variants[variant]
                or task.definition.private.bindings != scene.definition.bindings):
            raise ValueError('The exact selected task variant, bindings and seed must match the separate scene.')
        if (task.definition.source.json_path != row['source']['identity']
                or scene.definition.source.json_path != row['selected_scene_source']['identity']):
            raise ValueError('Authored source identities must retain both original registry source roles.')
        evaluation = task.definition.private.evaluation
        parameters = evaluation.parameters
        identities = sorted({scene.definition.source.json_path, task.definition.source.json_path,
                             *(member['source_identity'] for member in parameters.get('members', []))})
        members = [{'source_identity': identity, 'source_file': str(resolve_source(identity)),
                    'source_sha256': sha256(resolve_source(identity))} for identity in identities]
        task.verify_unchanged()
        scene.verify_unchanged()
        definition_identity = {
            'wia_id': row['wia_id'], 'corev2_candidate_id': row['canonical_candidate_id'],
            'task_sha256': task.file_sha256, 'scene_sha256': scene.file_sha256,
            'authored_manifest_sha256': sha256(manifest_path), 'variant': variant, 'seed': scene.seed,
            'source_members': [{key: member[key] for key in ('source_identity', 'source_sha256')} for member in members],
        }
        key = digest(definition_identity)[:24]
        candidate_id = 'category_' + key
        # Fields follow records_for_missing, with source roles and new capture identity explicit.
        record = {
            'candidate_id': candidate_id, 'corev2_candidate_id': row['canonical_candidate_id'],
            'wia_id': row['wia_id'], 'document_alias': alias, 'status': 'generated',
            'evidence_scope': 'definition_only', 'definition_only': True,
            'source_identity': scene.definition.source.json_path, 'original_source': str(scene.source_path),
            'source_sha256': scene.source_sha256, 'capture_source_members': members,
            'task_source_identity': task.definition.source.json_path, 'task_source_sha256': task.source_sha256,
            'scene_source_identity': scene.definition.source.json_path, 'scene_source_sha256': scene.source_sha256,
            'big_task': original['category'], 'small_task': original['subcategory'],
            'scene_file': str(scene.path), 'scene_sha256': scene.file_sha256,
            'task_file': str(task.path), 'task_sha256': task.file_sha256,
            'seed': scene.seed, 'variant': variant, 'reference_only': False,
            'converter_module': evaluation.module, 'converter_version': REVISION,
            'reuse_level': 'preserved_original_layout_new_authored_task_contract',
            'capture_initial_agent_bundle': True, 'task_edit2_capture_campaign': True,
            'task_edit2_variant_key': key, 'wia_authored_capture': True,
            'corev2_task_source_identity': task.definition.source.json_path,
            'robot_task_public': task.definition.public.model_dump(mode='json'),
            'diagnostic_dir': str(OUTPUT.parent / 'authored_capture_diagnostics' / key),
            'warning_codes': deepcopy(original['issues']), 'success_claimed': False,
            'native_validation_run': False, 'reference_execution_status': 'deferred_until_scene_review',
            'definition_identity': definition_identity,
            'limitations': [
                'Definition-only capture preparation; no publication, queued job, native screenshot or task success is established.',
                'Reuse the original layout and source assets; the selected authored task contract is separately versioned.',
                'Request the complete initial mounted-camera and frozen-overview Agent bundle when capture is authorized.',
                'Full legal robot reference execution is deferred until scene review.',
                'Original CoreV2 issue codes are historical provenance, not a new validation result.',
            ],
        }
        record['revision'] = digest(record)
        items.append({
            'row_id': row['wia_id'], 'wia_id': row['wia_id'], 'document_alias': alias,
            'variant_id': task.definition.id, 'label': label,
            'canonical_candidate_id': row['canonical_candidate_id'], 'capture_candidate_id': candidate_id,
            'task_file': str(task.path), 'task_sha256': task.file_sha256,
            'scene_file': str(scene.path), 'scene_sha256': scene.file_sha256,
            'source_identity': scene.definition.source.json_path, 'source_file': str(scene.source_path),
            'source_sha256': scene.source_sha256, 'task_source_sha256': task.source_sha256,
            'variant': variant, 'record': record,
        })
    if len({item['capture_candidate_id'] for item in items}) != 4:
        raise ValueError('The four authored versions require distinct capture identities.')
    return {
        'schema': REVISION, 'status': 'definition_only', 'record_count': 4,
        'registry_file': str(REGISTRY), 'registry_sha256': sha256(REGISTRY),
        'corev2_manifest_file': str(COREV2), 'corev2_manifest_sha256': sha256(COREV2),
        'authored_manifests': [{'file': str(path), 'sha256': sha256(path)} for path in (regional_path, rigid_path)],
        'published': False, 'queue_mutated': False, 'native_capture_run': False,
        'success_claimed': False, 'reference_execution_status': 'deferred_until_scene_review',
        'items': items,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare exact record bytes without writing.')
    args = parser.parse_args(argv)
    package = build_records()
    rendered = (json.dumps(package, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != rendered:
            print(f'Authored WIA capture records differ or are missing: {OUTPUT}', file=sys.stderr)
            return 1
    elif not OUTPUT.is_file() or OUTPUT.read_bytes() != rendered:
        write_json(OUTPUT, package)
    print(json.dumps({'output': str(OUTPUT), 'definition_only': True, 'count': len(package['items']),
        'variants': {item['document_alias']: item['variant'] for item in package['items']}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

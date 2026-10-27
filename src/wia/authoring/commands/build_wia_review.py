"""Publish an independent, CPU-only WIA review snapshot from explicit evidence.

Run: python -B scripts/build_wia_review.py [--registry PATH] [--evidence PATH]
  [--supplement PATH ...] [--output PATH] [--watch --interval-s 15]
No simulator, source resolver, capture queue, image decoder or model is loaded.
The publisher never changes review decisions or qualifies a native reference.


Canonical invocation: python -m wia.authoring.commands.build_wia_review --help
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT
DEFAULT_REGISTRY = ROOT / 'explore_to_achieve/wia_v1/registry.json'
DEFAULT_EVIDENCE = ROOT / 'docs/research/world_interactive_agency_v1/review_evidence.json'
DEFAULT_OUTPUT = ARTIFACT_ROOT / 'wia_v1_20261023/review/snapshot.json'
SCHEMA = 'wia_review_snapshot_v1'
EVIDENCE_SCHEMA = 'wia_review_evidence_v1'
PUBLIC_FIELDS = ('objective', 'success_criteria', 'constraints', 'environment')
DEFAULT_VIEWS = (
    'initial_overview_rgb', 'agentview_rgb', 'wrist_rgb', 'right_wrist_rgb',
    'chassis_front_left_rgb', 'chassis_front_right_rgb', 'chassis_left_rgb',
    'chassis_right_rgb', 'chassis_rear_rgb',
)
ALLOWED_VIEWS = frozenset((*DEFAULT_VIEWS, 'head', 'wrist', 'right_wrist',
    'chassis_front_left', 'chassis_front_right', 'chassis_left', 'chassis_right', 'chassis_rear',
    *('global_' + name for name in ('diagonal', 'overhead', 'front', 'back', 'left', 'right', 'floorplan')),
    'object_overview', 'object_top', 'object_side', 'scene',
    *(f'object_detail_{number:02d}' for number in range(32)),
    *(f'phase_{phase}_{name}' for phase in (1, 2)
      for name in ('head', 'wrist', 'object_overview', 'object_top', 'object_side')),
    'phase_2_global_diagonal'))
RESERVED_VARIANTS = {'original_selected', 'proposal'}
TOKEN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _safe_path(value, root):
    """Reject disallowed locations before reading, including symlink escapes."""
    if not isinstance(value, (str, Path)) or not str(value) or str(value).startswith('~'):
        raise ValueError('A nonempty repository-relative or /local_home2 path is required.')
    path = Path(value)
    if not path.is_absolute() and '..' in path.parts:
        raise ValueError('Repository-relative paths must not contain parent traversal.')
    candidate = Path(os.path.abspath(path if path.is_absolute() else root / path))
    allowed = lambda target: target.is_relative_to(root) or target.is_relative_to(Path('/local_home2'))
    if not allowed(candidate):
        raise ValueError(f'Path outside the repository and /local_home2 is forbidden: {value}')
    resolved = candidate.resolve()
    if not allowed(resolved):
        raise ValueError(f'Path resolves outside the repository and /local_home2: {value}')
    return resolved


def _portable(path, root):
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def _token(value, label):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise ValueError(f'{label} must be a nonempty safe URL token: {value!r}')
    return value


class _PublicLoader(yaml.SafeLoader):
    """Keep safe YAML parsing and reject duplicate definition fields."""


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError('Duplicate YAML keys are not accepted.')
        result[key] = loader.construct_object(value_node)
    return result


_PublicLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _definition(value, root, cache, *, task=False):
    if value in (None, ''):
        return {'path': '', 'available': False, 'reason': 'No definition file is declared.',
                'sha256': None}, {}
    path = _safe_path(value, root)
    key = (path, task)
    if key not in cache:
        record = {'path': _portable(path, root), 'available': False, 'sha256': None}
        public = {}
        try:
            content = path.read_bytes()
            record['sha256'] = hashlib.sha256(content).hexdigest()
            if task:
                definition = yaml.load(content, Loader=_PublicLoader)
                if not isinstance(definition, dict) or not isinstance(definition.get('public'), dict):
                    raise ValueError('Task YAML has no public task mapping.')
                public = {field: deepcopy(definition['public'][field])
                          for field in PUBLIC_FIELDS if field in definition['public']}
            record['available'] = True
        except (OSError, ValueError, UnicodeError, yaml.YAMLError) as error:
            record['reason'] = 'Definition is missing or cannot be read as public metadata.'
            record['error'] = f'{type(error).__name__}: {error}'
        cache[key] = record, public
    return deepcopy(cache[key])


def _public(value):
    source = value.get('public', value)
    if not isinstance(source, dict):
        raise ValueError('Explicit public proposal fields must be a mapping.')
    return {field: deepcopy(source[field]) for field in PUBLIC_FIELDS if field in source}


def _images(captures, variant_id, root):
    images, checks, seen = [], [], set()
    if not isinstance(captures, list):
        raise ValueError('Variant captures must be a list.')
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError('Each capture must be an explicit mapping.')
        directory = _safe_path(capture.get('directory'), root)
        directory_label = _portable(directory, root)
        label, role = capture.get('label', 'Capture'), capture.get('role', 'human_review_only')
        views = capture.get('views', list(DEFAULT_VIEWS))
        if not isinstance(views, list) or any(not isinstance(view, str) for view in views):
            raise ValueError('Capture views must be a list of known view names.')
        check = {'directory': directory_label, 'label': label, 'role': role,
                 'available_views': [], 'missing_views': []}
        capture_id = _digest({'directory': directory_label, 'label': label, 'role': role})[:12]
        for view in views:
            name = view[:-4] if view.endswith('.png') else view
            if name not in ALLOWED_VIEWS:
                raise ValueError(f'Unknown capture view: {view!r}')
            path = _safe_path(directory / (name + '.png'), root)
            if not path.is_file():
                check['missing_views'].append(name)
                continue
            check['available_views'].append(name)
            image_id = f'{variant_id[:60]}_{capture_id}_{name}'
            if image_id in seen:
                continue
            seen.add(image_id)
            images.append({'id': image_id, 'label': f'{label} / {name}',
                           'path': str(path), 'role': role})
        checks.append(check)
    return images, checks


def _variant(value, row_id, root, cache, *, fallback_public=None, default_status='definition_only'):
    if not isinstance(value, dict):
        raise ValueError('Each variant must be a mapping.')
    variant_id = _token(value.get('id'), 'Variant id')
    task, public = _definition(value.get('task_file'), root, cache, task=True)
    scene, _ = _definition(value.get('scene_file'), root, cache)
    if not task['path']:
        public = _public(value) or deepcopy(fallback_public or {})
    images, capture_checks = _images(value.get('captures', []), variant_id, root)
    evidence = []
    if not isinstance(value.get('evidence', []), list):
        raise ValueError('Variant evidence must be a list of explicit references.')
    for item in value.get('evidence', []):
        if not isinstance(item, dict):
            raise ValueError('Each evidence reference must be a mapping.')
        path = _safe_path(item.get('path'), root)
        evidence.append({'label': item.get('label', 'Evidence'), 'path': _portable(path, root)})
    identity = {'row_id': row_id, 'variant_id': variant_id,
                'task_sha256': task['sha256'], 'scene_sha256': scene['sha256']}
    if not task['path']:
        identity['proposal_public'] = public
    result = {
        'id': variant_id, 'label': value.get('label', variant_id),
        'task_file': task['path'], 'scene_file': scene['path'],
        'objective': public.get('objective'), 'success_criteria': public.get('success_criteria'),
        'constraints': public.get('constraints', []), 'environment': public.get('environment'),
        'status': value.get('status', default_status),
        'evidence_summary': value.get('evidence_summary',
            'No current native qualification is inferred from definitions or images.'),
        'evidence': evidence, 'images': images,
        'definition_status': {'task': task, 'scene': scene},
        'missing_definition': not task['available'] or not scene['available'],
        'capture_checks': capture_checks,
        'fingerprint': _digest(identity),
    }
    if not task['path']:
        result['public_text_source'] = 'explicit_design_proposal' if public else 'not_supplied'
    else:
        result['public_text_source'] = 'task_yaml' if task['available'] else 'unavailable'
    if 'capture_review' in value:
        result['capture_review'] = deepcopy(value['capture_review'])
    return result


def _row(registry_row, update, root, cache):
    row_id = _token(registry_row.get('wia_id', registry_row.get('id')), 'Row id')
    task, scene = registry_row.get('task') or {}, registry_row.get('scene') or {}
    unbuilt = registry_row.get('implementation_state') == 'unbuilt'
    default_status = 'unbuilt' if unbuilt else 'definition_only'
    public = _public(registry_row)
    default = {'id': 'proposal' if unbuilt else 'original_selected',
               'label': 'Design proposal' if unbuilt else 'Original selected definition',
               'task_file': task.get('task_file', ''), 'scene_file': scene.get('selected_scene_file', ''),
               'status': default_status}
    variants = update.get('variants', [])
    if not isinstance(variants, list):
        raise ValueError('Curated variants must be a list.')
    if any(isinstance(variant, dict) and variant.get('id') in RESERVED_VARIANTS for variant in variants):
        raise ValueError('Curated variants must not replace original_selected or proposal.')
    categories = registry_row.get('source_esi_categories') or []
    category_label = categories[0].split('/')[-1] if categories else registry_row.get('primary_subclass', '')
    row = {'id': row_id, 'document_alias': registry_row.get('document_alias', row_id),
           'primary_family': registry_row.get('primary_family'),
           'primary_subclass': registry_row.get('primary_subclass'),
           'title': registry_row.get('title') or f'{registry_row.get("document_alias", row_id)} / {category_label}',
           'implementation_state': registry_row.get('implementation_state', 'unbuilt'),
           'status': update.get('status', default_status),
           'summary': update.get('summary', (registry_row.get('reassessment') or {}).get('reason', '')),
           'variants': [_variant(variant, row_id, root, cache, fallback_public=public,
                                 default_status=default_status) for variant in [*variants, default]],
           'proposal_membership': True}
    row['registry_implementation_state'] = row['implementation_state']
    if unbuilt and any(not variant['missing_definition'] for variant in row['variants'][:-1]):
        row['implementation_state'] = 'authored_definition'
        row['implementation_state_basis'] = 'Current curated task and scene definitions are available; native validation is separate.'
    if row['variants'] and not row['variants'][0]['images']:
        ready = next((index for index, variant in enumerate(row['variants']) if variant['images']), None)
        if ready is not None:
            row['variants'].insert(0, row['variants'].pop(ready))
    preferred = update.get('preferred_variant_id')
    if preferred:
        ready = next((index for index, variant in enumerate(row['variants'])
                      if variant['id'] == preferred and variant['images']), None)
        if ready is not None:
            row['variants'].insert(0, row['variants'].pop(ready))
    for field in ('source', 'original_sources', 'source_suggestion', 'source_esi_categories',
                  'role', 'role_status', 'tags', 'reassessment', 'classification_status',
                  'canonical_candidate_id', 'allowed_capabilities', 'public_observation_contract'):
        if field in registry_row:
            row[field] = deepcopy(registry_row[field])
    return row


def _extra_row(value, root, cache):
    if not isinstance(value, dict):
        raise ValueError('Each extra row must be a mapping.')
    row_id = _token(value.get('id', value.get('wia_id')), 'Extra row id')
    variants = value.get('variants', [])
    if not isinstance(variants, list) or not variants:
        raise ValueError('Each extra row must explicitly declare at least one variant.')
    row = {key: deepcopy(item) for key, item in value.items() if key != 'variants'}
    row.update(id=row_id, proposal_membership=False)
    row.setdefault('document_alias', row_id)
    row.setdefault('title', row['document_alias'])
    row.setdefault('primary_family', None)
    row.setdefault('primary_subclass', None)
    row.setdefault('implementation_state', 'unbuilt')
    row.setdefault('status', 'unbuilt')
    row.setdefault('summary', '')
    row['variants'] = [_variant(variant, row_id, root, cache, fallback_public=_public(value),
                                default_status=row['status']) for variant in variants]
    return row


def _unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys are not accepted in evidence supplements: ' + key)
        result[key] = value
    return result


def _merge_supplements(updates, supplements, identities, root):
    """Append variants without changing main row metadata or extra membership."""
    merged, paths = deepcopy(updates), []
    used = {identity: set(RESERVED_VARIANTS) for identity in identities}
    for row_id, update in merged.items():
        variants = update.get('variants', [])
        if not isinstance(variants, list):
            raise ValueError('Curated variants must be a list.')
        for variant in variants:
            variant_id = _token(variant.get('id'), 'Main variant id')
            if variant_id in used[row_id]:
                raise ValueError(f'Duplicate or reserved variant id for row {row_id}: {variant_id}')
            used[row_id].add(variant_id)
    for declared in supplements:
        path = _safe_path(declared, root)
        value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_json_pairs)
        if not isinstance(value, dict) or value.get('schema') != EVIDENCE_SCHEMA:
            raise ValueError(f'Expected supplement schema {EVIDENCE_SCHEMA}.')
        if (set(value) - {'schema', 'generated_at', 'row_updates', 'extra_rows', 'summary', 'provenance'}
                or value.get('extra_rows', [])):
            raise ValueError('Supplements cannot change main metadata or add extra rows.')
        additions = value.get('row_updates', {})
        if not isinstance(additions, dict):
            raise ValueError('Supplement row_updates must be a mapping.')
        for row_id, update in additions.items():
            if row_id not in used:
                raise ValueError('Supplement names an unknown registry row: ' + row_id)
            if not isinstance(update, dict) or set(update) != {'variants'} or not isinstance(update['variants'], list):
                raise ValueError('Supplement row updates may contain only a variants list.')
            target = merged.setdefault(row_id, {}).setdefault('variants', [])
            for variant in update['variants']:
                if not isinstance(variant, dict):
                    raise ValueError('Each supplement variant must be a mapping.')
                variant_id = _token(variant.get('id'), 'Supplement variant id')
                if variant_id in used[row_id]:
                    raise ValueError(f'Conflicting duplicate row/variant id: {row_id}/{variant_id}')
                used[row_id].add(variant_id)
                target.append(deepcopy(variant))
        paths.append(_portable(path, root))
    return merged, paths


def build_snapshot(registry=DEFAULT_REGISTRY, evidence=DEFAULT_EVIDENCE, *, supplements=(),
                   project_root=ROOT, generated_at=None):
    """Read only declared files and return all proposals plus explicit extra rows."""
    root = Path(project_root).resolve()
    registry_path, evidence_path = _safe_path(registry, root), _safe_path(evidence, root)
    registry_data = json.loads(registry_path.read_text(encoding='utf-8'))
    evidence_data = json.loads(evidence_path.read_text(encoding='utf-8'))
    if evidence_data.get('schema') != EVIDENCE_SCHEMA:
        raise ValueError(f'Expected curation schema {EVIDENCE_SCHEMA}.')
    source_rows = registry_data.get('rows')
    updates, extras = evidence_data.get('row_updates', {}), evidence_data.get('extra_rows', [])
    if not isinstance(source_rows, list) or not isinstance(updates, dict) or not isinstance(extras, list):
        raise ValueError('Registry rows, row_updates and extra_rows must have their declared container types.')
    identities = [row.get('wia_id') for row in source_rows]
    if len(identities) != len(set(identities)) or any(not identity for identity in identities):
        raise ValueError('Registry row identities must be present and unique.')
    if unknown := set(updates) - set(identities):
        raise ValueError('Curation names unknown registry rows: ' + ', '.join(sorted(unknown)))
    if any(not isinstance(update, dict) for update in updates.values()):
        raise ValueError('Each row update must be a mapping.')
    supplement_paths = []
    if supplements:
        updates, supplement_paths = _merge_supplements(updates, supplements, identities, root)
    cache = {}
    rows = [_row(row, updates.get(row['wia_id'], {}), root, cache) for row in source_rows]
    rows.extend(_extra_row(row, root, cache) for row in extras)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Extra row identities must not duplicate existing rows.')
    for row in rows:
        if len({variant['id'] for variant in row['variants']}) != len(row['variants']):
            raise ValueError(f'Variant ids must be unique within row {row["id"]}.')
    snapshot = {
        'schema': SCHEMA,
        'generated_at': generated_at or datetime.now(timezone.utc).isoformat(),
        'title': evidence_data.get('title', 'World Interactive Agency Review'),
        'summary': {
            'proposal_count': len(source_rows), 'row_count': len(rows), 'extra_row_count': len(extras),
            'authored_variant_count': sum(variant['id'] not in RESERVED_VARIANTS
                                          for row in rows for variant in row['variants']),
            'rows_with_images_count': sum(any(variant['images'] for variant in row['variants']) for row in rows),
            'reference_success_row_count': sum(row['status'] == 'reference_verified' for row in rows),
            'unbuilt_row_count': sum(row['implementation_state'] == 'unbuilt' for row in rows),
            'registry_unbuilt_row_count': sum(row.get('implementation_state') == 'unbuilt' for row in source_rows),
            'family_counts': dict(sorted(Counter(row.get('primary_family') for row in source_rows).items())),
            'extra_family_counts': dict(sorted(Counter(row.get('primary_family') for row in extras).items())),
        },
        'rows': rows,
        'provenance': {'registry': _portable(registry_path, root), 'curation': _portable(evidence_path, root),
                       'status_source': 'Explicit human curation; defaults describe definitions only.',
                       'native_results_interpreted': False, 'images_are_success_evidence': False,
                       'native_experiments_run': False, 'final_release_selected': False},
    }
    if supplement_paths:
        snapshot['provenance']['supplements'] = supplement_paths
    return snapshot


def publish_snapshot(output, snapshot, *, project_root=ROOT):
    """Atomically replace this publisher's snapshot without touching reviews.json."""
    root = Path(project_root).resolve()
    path = _safe_path(output, root)
    if path.name == 'reviews.json':
        raise ValueError('The publisher must never write reviews.json.')
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(previous, dict) or previous.get('schema') != SCHEMA:
            raise ValueError('Refusing to overwrite an unrelated output.')
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(snapshot, ensure_ascii=True, indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument('--supplement', type=Path, action='append', default=[],
                        help='Append variants from an independent evidence supplement; repeatable.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval-s', type=float, default=15.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval_s) or args.interval_s <= 0:
        parser.error('--interval-s must be positive and finite.')
    try:
        while True:
            try:
                snapshot = build_snapshot(args.registry, args.evidence, supplements=args.supplement)
                path = publish_snapshot(args.output, snapshot)
                print(f'Published {snapshot["summary"]["proposal_count"]} proposals and '
                      f'{snapshot["summary"]["extra_row_count"]} extra rows: {path}', flush=True)
            except (OSError, ValueError) as error:
                if not args.watch:
                    raise
                print(f'WIA review refresh failed; previous snapshot retained: {error}', file=sys.stderr, flush=True)
            if not args.watch:
                break
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

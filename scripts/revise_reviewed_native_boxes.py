"""Bind five reviewed carton layouts to their intended native support positions.

Run with behavior Python and PYTHONPATH=src:../Show-Harness-Explore:
  python scripts/revise_reviewed_native_boxes.py --apply
The default validates proposals without writing. Parent definitions and the
original native review receipts remain provenance; native validation is pending.
"""
import argparse
from copy import deepcopy
import hashlib
import json

import yaml

from wia.resources import ROOT
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
from review_box_layouts import TRANSLATIONS_WORLD_M


ALIASES = ('T40', 'T41', 'T45', 'T47', 'T48')
MARKER = 'reviewed_group_placements_v1'
DESTINATION = ROOT / 'explore_to_achieve/wia_v1/review_native_layout_20261030'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    catalogue_path = ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json'
    catalogue_bytes = catalogue_path.read_bytes()
    catalogue = json.loads(catalogue_bytes)
    refresh = json.loads((ROOT / 'outputs/review_repairs_20261030/refresh_result.json').read_text())
    evidence = json.loads((ROOT / refresh['manifest_file']).read_text())
    rows = {row['alias']: row for row in catalogue['tasks']}
    prepared = []
    for alias in ALIASES:
        row = rows[alias]
        paths = {kind: ROOT / row[kind + '_file'] for kind in ('task', 'scene')}
        parents = {kind: path.read_bytes() for kind, path in paths.items()}
        values = {kind: yaml.safe_load(data) for kind, data in parents.items()}
        task, scene = values['task'], values['scene']
        if MARKER in task.get('repair', {}):
            raise ValueError('Preserve the existing reviewed native layout revision.')
        if task['scene'] != scene['scene']:
            raise ValueError('Task and scene initialization differ before revision.')
        old = evidence['old_captures'][alias]
        complete_receipts = [entry for entry in old['receipts'] if entry['complete']]
        if not complete_receipts:
            raise ValueError('No complete original native review receipt is available.')
        receipt_path = ROOT / complete_receipts[-1]['file']
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if not receipt['complete'] or receipt['job_id'] != old['job_id']:
            raise ValueError('Original native review receipt identity differs.')
        setup = receipt['interaction_setup']
        delta = TRANSLATIONS_WORLD_M[alias]
        placements = {}
        for key in task['private']['evaluation']['parameters']['container_keys']:
            name = task['private']['bindings'][key]['name']
            original = setup['adapted_geometry'][name]['position_m']
            support_path = setup['support_prim_paths'][name]
            if not isinstance(support_path, str) or not support_path.startswith('/World/'):
                raise ValueError('A measured native support path is required.')
            placements[key] = {'support_name': support_path.rsplit('/', 1)[-1],
                              'root_xy_m': [original[i] + delta[i] for i in range(2)]}
        changed = {}
        for kind, value in values.items():
            changed[kind] = []
            for variant, spec in value['scene']['variants'].items():
                for patch in spec['patches']:
                    if patch.get('operation') == 'prepare_category_interaction':
                        if patch['relocation_policy'] != 'independent_groups':
                            raise ValueError('Reviewed carton positions require independent groups.')
                        patch.setdefault('layout_parameters', {})[MARKER] = deepcopy(placements)
                        changed[kind].append(variant)
            if 'adapted' not in changed[kind]:
                raise ValueError('The selected variant has no reviewed native placement policy.')
            value['version'] += 1
        task.setdefault('repair', {})[MARKER] = {
            'alias': alias, 'placements': placements,
            'basis': 'Original reviewed native carton roots plus the declared world translation.',
            'translation_world_m': list(delta), 'original_receipt': str(receipt_path),
            'original_receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
            'native_validation': 'pending', 'physical_grasp_verified': False,
        }
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        assert task['scene'] == scene['scene']
        assert task['private'] == yaml.safe_load(parents['task'])['private']
        prepared.append({'alias': alias, 'paths': paths, 'parents': parents,
                         'values': values, 'placements': placements, 'changed_variants': changed,
                         'previous_selection': deepcopy(row)})
    if not args.apply:
        print(json.dumps({'validated_aliases': list(ALIASES), 'native_validation': 'pending'}))
        return
    if DESTINATION.exists():
        raise FileExistsError('Do not overwrite an existing derived revision.')
    assert catalogue_path.read_bytes() == catalogue_bytes
    for item in prepared:
        assert all(path.read_bytes() == item['parents'][kind] for kind, path in item['paths'].items())
    DESTINATION.mkdir(parents=True)
    (DESTINATION / 'parent.catalogue.json').write_bytes(catalogue_bytes)
    records = []
    for item in prepared:
        alias = item['alias']
        row = rows[alias]
        for kind, value in item['values'].items():
            parent = DESTINATION / 'parents' / (kind + 's') / (alias.lower() + '.yaml')
            derived = DESTINATION / (kind + 's') / (alias.lower() + '.yaml')
            parent.parent.mkdir(parents=True, exist_ok=True)
            derived.parent.mkdir(parents=True, exist_ok=True)
            parent.write_bytes(item['parents'][kind])
            data = yaml.safe_dump(value, sort_keys=False, width=100).encode()
            derived.write_bytes(data)
            item['paths'][kind].write_bytes(data)
            row['source_' + kind + '_file'] = str(derived.relative_to(ROOT))
            row['source_' + kind + '_file_sha256'] = hashlib.sha256(data).hexdigest()
        records.append({key: item[key] for key in
                        ('alias', 'placements', 'changed_variants', 'previous_selection')})
    catalogue_path.write_text(json.dumps(catalogue, indent=2) + '\n')
    (DESTINATION / 'manifest.json').write_text(json.dumps({
        'revision': MARKER, 'rows': records, 'native_validation': 'pending'}, indent=2) + '\n')
    print(json.dumps({'updated_aliases': list(ALIASES), 'native_validation': 'pending'}))


if __name__ == '__main__':
    main()

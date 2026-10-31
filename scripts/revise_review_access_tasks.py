"""Derive the ten reviewed access repairs and preserve every active parent byte.

Run with behavior Python and PYTHONPATH=src:../Show-Harness-Explore:
  python scripts/revise_review_access_tasks.py --apply
Without --apply, validate all proposed definitions in memory. Capture submission
and decision reset are separate operations; running jobs are never interrupted.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from wia.resources import ROOT
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition
import review_t22_d1_layouts
import review_box_layouts


ALIASES = ('T22', 'D1-1', 'D1-2', 'D1-3', 'T40', 'T41', 'T45', 'T46', 'T47', 'T48')
DESTINATION = ROOT / 'explore_to_achieve/wia_v1/review_access_20261030'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    catalogue_path = ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json'
    catalogue_bytes = catalogue_path.read_bytes()
    catalogue = json.loads(catalogue_bytes)
    rows = {r['alias']: r for r in catalogue['tasks']}
    prepared = []
    for alias in ALIASES:
        row = rows[alias]
        paths = {kind: ROOT / row[kind + '_file'] for kind in ('task', 'scene')}
        parents = {kind: path.read_bytes() for kind, path in paths.items()}
        original = {kind: yaml.safe_load(data) for kind, data in parents.items()}
        helper = review_t22_d1_layouts if alias in review_t22_d1_layouts.SUPPORTED_ALIASES else review_box_layouts
        task, scene, variant, changes = helper.revise(alias, original['task'], original['scene'])
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        assert task['scene'] == scene['scene']
        assert task['private']['bindings'] == scene['bindings']
        assert scene['scene']['variants'][variant]['patches']
        assert task['source'] == original['task']['source'] and scene['source'] == original['scene']['source']
        prepared.append({'alias': alias, 'previous_selection': deepcopy(row), 'variant': variant,
                         'paths': paths, 'parents': parents, 'changes': changes,
                         'bytes': {kind: yaml.safe_dump(value, sort_keys=False, width=100).encode()
                                   for kind, value in [('task', task), ('scene', scene)]}})
    if not args.apply:
        print(json.dumps({'validated_aliases': list(ALIASES), 'native_validation': 'pending'}))
        return
    if DESTINATION.exists():
        raise FileExistsError('Retain the existing access revision; do not overwrite parent evidence.')
    for item in prepared:
        for kind, path in item['paths'].items():
            if path.read_bytes() != item['parents'][kind]:
                raise ValueError('Active definition changed during preparation: ' + str(path))
    if catalogue_path.read_bytes() != catalogue_bytes:
        raise ValueError('Active catalogue changed during preparation.')
    DESTINATION.mkdir(parents=True)
    (DESTINATION / 'parent.catalogue.json').write_bytes(catalogue_bytes)
    records = []
    for item in prepared:
        alias = item['alias']
        row = rows[alias]
        row['variant'] = item['variant']
        for kind in ('task', 'scene'):
            parent = DESTINATION / 'parents' / (kind + 's') / (alias.lower() + '.yaml')
            derived = DESTINATION / (kind + 's') / (alias.lower() + '.yaml')
            parent.parent.mkdir(parents=True, exist_ok=True)
            derived.parent.mkdir(parents=True, exist_ok=True)
            parent.write_bytes(item['parents'][kind])
            derived.write_bytes(item['bytes'][kind])
            item['paths'][kind].write_bytes(item['bytes'][kind])
            row['source_' + kind + '_file'] = str(derived.relative_to(ROOT))
            row['source_' + kind + '_file_sha256'] = sha(item['bytes'][kind])
        records.append({'alias': alias, 'previous_selection': item['previous_selection'],
                        'current_selection': deepcopy(row), 'changes': item['changes'],
                        'native_validation': 'pending'})
    catalogue_path.write_text(json.dumps(catalogue, indent=2) + '\n')
    (DESTINATION / 'manifest.json').write_text(json.dumps({'revision': 'user_review_access_v1',
        'basis': str(ROOT / 'outputs/review_reconciliation_20261030/reconciliation.json'),
        'rows': records, 'physical_success_verified': False}, indent=2) + '\n')
    print(json.dumps({'updated_aliases': list(ALIASES), 'manifest': str(DESTINATION / 'manifest.json'),
                      'native_validation': 'pending'}))


if __name__ == '__main__':
    main()

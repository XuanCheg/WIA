"""Clear selected seed-202 reflected sightlines by swapping fixture XY positions.

Run: behavior/bin/python -B scripts/revise_wia_mirror_seed202_clearance.py
Preserves V2 and paired R-A geometry; outputs exact V3 definitions only.


Canonical invocation: python -m wia.authoring.commands.revise_wia_mirror_seed202_clearance
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import yaml

from wia.resources import ROOT
from wia.tasks.loader import load_scene, load_task
from wia.authoring.commands.design_wia_review_revisions import write, dump


def main():
    base = ROOT / 'explore_to_achieve/wia_v1/review_20261025'
    source = json.loads((base / 'produce_mirror_v2/manifest.json').read_text())
    rows = []
    for row in source['entries']:
        if row['seed'] != 202 or row['document_alias'] not in ('R-C1', 'R-C2', 'R-C3', 'R-A1', 'R-A2'):
            continue
        new = deepcopy(row)
        new['variant_id'] = row['variant_id'].replace('_v2', '_v3')
        new['label'] += ' / reflected sightline clearance'
        for kind in ('task', 'scene'):
            src = Path(row[kind + '_file'])
            value = yaml.safe_load(src.read_text())
            for variant in value['scene']['variants'].values():
                for patch in variant['patches']:
                    objects = patch.get('parameters', {}).get('layout', {}).get('objects', [])
                    pot = next((obj for obj in objects if obj['name'] == 'wia_produce_pot'), None)
                    board = next((obj for obj in objects if obj['name'] == 'wia_produce_board'), None)
                    if pot is not None:
                        pot['position'][:2], board['position'][:2] = board['position'][:2], pot['position'][:2]
            value['id'] = value['id'].replace('_v2', '_v3')
            if 'repair' in value:
                value['repair'].update(variant_id=new['variant_id'], parent_task_file=row['task_file'],
                    parent_scene_file=row['scene_file'], design_scope='For seed 202 only, swap native pot/board XY positions to clear the selected high-Y reflected produce. Preserve object identities, individual Z, candidate poses, cue, seed, thresholds and paired R-A layout.')
            dst = base / 'produce_mirror_v3' / new['variant_id'] / (kind + '.yaml')
            write(dst, yaml.safe_dump(value, sort_keys=False).encode())
            new[kind + '_file'], new['parent_' + kind + '_file'] = str(dst), str(src)
        load_task(new['task_file'], variant='adapted', run_mode='diagnostic')
        load_scene(new['scene_file'], variant='adapted', run_mode='diagnostic')
        rows.append(new)
    dump(base / 'produce_mirror_v3/reflection_clearance_manifest.json',
         {'schema': 'wia_variant_manifest_v1', 'entries': rows})
    print('Authored five exact seed-202 clearance revisions.')


if __name__ == '__main__':
    main()

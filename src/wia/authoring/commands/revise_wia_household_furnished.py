"""Create immutable household mirror revisions after native setup review.

Run: behavior/bin/python -B scripts/revise_wia_household_furnished.py [--check]
Only definitions are built; no robot action or simulator is invoked.


Canonical invocation: python -m wia.authoring.commands.revise_wia_household_furnished
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT
from wia.authoring.commands import build_wia_household_mirror as builder
from wia.authoring.commands.design_wia_furnished_round import OUT, dump, inventory


def main():
    parent = json.loads((OUT / 'mirror_design_v1.json').read_text())
    entries = []
    chairs = inventory('office_cubicles_left')
    for original in parent['entries']:
        alias = original['document_alias']
        if alias in ('R-C3', 'R-S1'):
            continue
        entry = deepcopy(original)
        entry['variant_id'] = entry['variant_id'].replace('_v1', '_v2')
        entry['task_id'] = entry['task_id'].replace('_v1', '_v2')
        entry['scene_id'] = entry['scene_id'].replace('_v1', '_v2')
        entry['version'] = 3
        entry['label'] += ' v2 (native fixed-joint compatibility and furniture clearance)'
        entry['layout']['shared_layout_id'] += '_v2'
        if entry['scene_model'] == 'office_cubicles_left':
            poses = []
            for name, xy in [('swivel_chair_ucawqa_5', [-8.1, 16.3]),
                             ('swivel_chair_ucawqa_7', [-5.6, 16.3])]:
                native = chairs[name]
                poses.append({'name': name, 'category': native['category'], 'model': native['model'],
                              'position': [*xy, native['pose'][0][2]],
                              'orientation': native['pose'][1]})
            entry['layout']['native_object_poses'] = poses
        entries.append(entry)
    design = {'schema': 'wia_household_mirror_design_v1', 'author': 'main_agent',
              'entries': entries, 'status': 'pending_native_initialization_review',
              'revision_evidence': {
                  'geometry': 'Native pepper grinders have fixed metadata joints, which set articulated=True. '
                              'Household measurement now requires actual n_dof=0 and n_joints=0 and the same '
                              'single enabled root collision link; no assets or physical thresholds change.',
                  'office': 'R-C2 initial robot contacted swivel_chair_ucawqa_5. Retain both central west '
                            'chairs and move them to open north corners during initialization, preserving '
                            'their original height, orientation and physical properties.',
                  'invariants': 'All candidates, cue geometry, pads, anchors, goals, scoring, seeds and '
                                'budgets remain unchanged. R-A1/R-A2 share identical furniture changes.',
                  'scope': 'Definitions and initialization only; original failures remain immutable.'}}
    path = OUT / 'mirror_design_v2.json'
    check = '--check' in sys.argv
    if path.exists():
        if json.loads(path.read_text()) != design:
            raise ValueError('Refusing to overwrite a differing mirror revision.')
    elif check:
        raise ValueError('Mirror revision has not been built.')
    else:
        dump(path, design)
    return builder.main(['--design-file', str(path), '--output-dir', str(OUT / 'household_mirror_v2')]
                        + (['--check'] if check else []))


if __name__ == '__main__':
    raise SystemExit(main())

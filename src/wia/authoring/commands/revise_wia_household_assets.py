"""Build single-collision-body household mirror asset revisions on CPU.

Run: behavior/bin/python -B scripts/revise_wia_household_assets.py [--check]
This preserves old versions and never invokes the simulator or robot controls.


Canonical invocation: python -m wia.authoring.commands.revise_wia_household_assets
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT
from wia.authoring.commands import build_wia_household_mirror as builder
from wia.authoring.commands import design_wia_household_round as source
from wia.authoring.commands.design_wia_furnished_round import INV, OUT, dump


def main():
    metadata = json.loads((INV / 'household_rigid_metadata.json').read_text())
    metadata.update(json.loads((INV / 'household_replacement_metadata.json').read_text()))
    parent = json.loads((OUT / 'mirror_design_v2.json').read_text())
    previous = {(e['document_alias'], e['variant_id'].split('_seed')[1].split('_')[0]): e
                for e in parent['entries']}
    replacements = {
        'R-C2': ('O', 'apple', 'agveuv', [0, 0, 0, 0], [-.84, -.28, .28, .84], 'apple', 'apples'),
        'R-A1': ('O', 'salt_shaker', 'fonexa', [0, 0, 0], [-.6, 0., .6], 'salt shaker', 'salt shakers'),
        'R-A2': ('O', 'salt_shaker', 'fonexa', [0, 0, 0], [-.6, 0., .6], 'salt shaker', 'salt shakers'),
    }
    entries = []
    for alias, specification in replacements.items():
        source.TASKS[alias] = specification
        for seed in (101, 202):
            entry = source.make_entry(alias, seed, metadata)
            entry['variant_id'] = entry['variant_id'].replace('_v1', '_v3')
            entry['task_id'] = entry['task_id'].replace('_v1', '_v3')
            entry['scene_id'] = entry['scene_id'].replace('_v1', '_v3')
            entry['version'] = 4
            entry['label'] += ' v3 (screened rigid household asset)'
            entry['layout']['shared_layout_id'] += '_v3'
            entry['layout']['native_object_poses'] = deepcopy(
                previous[alias, str(seed)]['layout']['native_object_poses'])
            entries.append(entry)
    design = {'schema': 'wia_household_mirror_design_v1', 'author': 'main_agent',
              'entries': entries, 'status': 'pending_native_initialization_review',
              'revision_evidence': {
                  'assets': 'Static USD inspection finds cola_bottle/oyqdtz and marjoram_shaker/xlspnl '
                            'have multiple enabled collision links (labels/glass). The current rigid '
                            'household measurement intentionally supports one collision body. Use '
                            'apple/agveuv and salt_shaker/fonexa, whose enabled geometry belongs to base_link.',
                  'geometry': 'Recompute cue placement from the explicit replacement asset bounds; '
                              'native measured collision centroids and optics must still qualify.',
                  'invariants': 'Keep scenes, native furniture, object counts, slot permutations, '
                                'seeds, operation rules and scoring. R-A1/R-A2 remain matched apart from '
                                'their direct/reflected cue placement. No mesh or physical rule is removed.',
                  'scope': 'Static asset screening is not native visibility or physical success evidence.'}}
    path = OUT / 'mirror_design_v3.json'
    check = '--check' in sys.argv
    if path.exists():
        if json.loads(path.read_text()) != design:
            raise ValueError('Refusing to overwrite a differing household asset revision.')
    elif check:
        raise ValueError('Asset revision has not been built.')
    else:
        dump(path, design)
    return builder.main(['--design-file', str(path), '--output-dir', str(OUT / 'household_mirror_v3')]
                        + (['--check'] if check else []))


if __name__ == '__main__':
    raise SystemExit(main())

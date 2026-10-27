"""Build matched appearance-control layouts with an in-frame direct cue.

Run: behavior/bin/python -B scripts/revise_wia_appearance_framing.py [--check]
Static projection uses the previous native camera; fresh images remain required.


Canonical invocation: python -m wia.authoring.commands.revise_wia_appearance_framing
"""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT
from wia.authoring.commands import build_wia_household_mirror as builder
from wia.authoring.commands import design_wia_household_round as source
from wia.authoring.commands.design_wia_furnished_round import INV, OUT, dump


def main():
    metadata = json.loads((INV / 'household_replacement_metadata.json').read_text())
    old = json.loads((OUT / 'mirror_design_v3.json').read_text())
    previous = {(e['document_alias'], e['variant_id'].split('_seed')[1].split('_')[0]): e for e in old['entries']}
    receipt_path = ARTIFACT_ROOT / 'wia_v1_20261023/capture_campaign/queue/jobs/capture_06b92b28a0e50e4a00e619f07c659e9e/attempts/01/preview_complete.json'
    camera = json.loads(receipt_path.read_text())['initial_agent_bundle']['review_diagnostics']['geometry']['camera']
    rotation = Rotation.from_quat(camera['quaternion_xyzw']).inv()
    eye = np.asarray(camera['position_m'])
    intrinsic = np.asarray(camera['intrinsic_matrix'])
    entries = []
    for alias in ('R-A1', 'R-A2'):
        source.TASKS[alias] = ('O', 'salt_shaker', 'fonexa', [0, 0, 0], [-.32, 0., .32], 'salt shaker', 'salt shakers')
        for seed in (101, 202):
            entry = source.make_entry(alias, seed, metadata)
            for key in ('variant_id', 'task_id', 'scene_id'):
                entry[key] = entry[key].replace('_v1', '_v4')
            entry['version'] = 5
            entry['label'] += ' v4 (matched visible appearance cue)'
            entry['layout']['shared_layout_id'] += '_v4'
            entry['layout']['native_object_poses'] = deepcopy(previous[alias, str(seed)]['layout']['native_object_poses'])
            projected = []
            for obj in entry['layout']['objects']:
                if obj['name'] not in {entry['bindings'][key]['name'] for key in ('cue_low', 'cue_high', 'cue_top', 'cue_bottom')}:
                    continue
                corners = np.asarray(list(itertools.product(*[[-s/2, s/2] for s in obj['scale']])))
                world = Rotation.from_quat(obj['orientation']).apply(corners) + obj['position']
                local = rotation.apply(world - eye)
                depth = -local[:, 2]
                if np.any(depth <= 0):
                    raise ValueError('Proposed cue has points behind the previous measured camera.')
                projected.extend(np.column_stack((intrinsic[0, 0]*local[:, 0]/depth+intrinsic[0, 2],
                                                  -intrinsic[1, 1]*local[:, 1]/depth+intrinsic[1, 2])).tolist())
            pixels = np.asarray(projected)
            if np.any(pixels.min(0) < 8) or np.any(pixels.max(0) > 504):
                raise ValueError('Proposed cue is clipped under the previous native camera: ' + str(pixels))
            entry['geometry_proposal']['previous_camera_cue_pixel_bounds'] = [pixels.min(0).tolist(), pixels.max(0).tolist()]
            entries.append(entry)
    design = {'schema': 'wia_household_mirror_design_v1', 'author': 'main_agent', 'entries': entries,
              'status': 'pending_native_initialization_review', 'revision_evidence': {
                  'receipt_file': str(receipt_path),
                  'finding': 'R-A2 V3 uniquely selects a direct object geometrically, but native head image clips '
                             'the outer target and red cue. Geometric selection alone is insufficient for image acceptance.',
                  'change': 'Use candidate Y offsets -0.32, 0, 0.32 m in both matched appearance conditions and '
                            'recompute cue geometry. Preserve all counts, assets, furniture, goals, seeds, scoring and budgets.',
                  'projection': 'All four cue rods fit with an 8-pixel margin under the previous measured native camera. '
                                'This is a static projection proposal; fresh native image review remains required.'}}
    path = OUT / 'mirror_appearance_design_v4.json'
    check = '--check' in sys.argv
    if path.exists():
        if json.loads(path.read_text()) != design:
            raise ValueError('Refusing to overwrite a differing appearance revision.')
    elif check:
        raise ValueError('Appearance revision has not been built.')
    else:
        dump(path, design)
    return builder.main(['--design-file', str(path), '--output-dir', str(OUT / 'household_appearance_v4')]
                        + (['--check'] if check else []))


if __name__ == '__main__':
    raise SystemExit(main())

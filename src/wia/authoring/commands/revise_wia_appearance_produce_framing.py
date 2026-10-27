"""Keep matched produce appearances inside the head view and above the table.

Run: behavior/bin/python -B scripts/revise_wia_appearance_produce_framing.py
Creates four V4 definitions; native captures must validate the proposed framing.


Canonical invocation: python -m wia.authoring.commands.revise_wia_appearance_produce_framing
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT, LazyMapping, read_json_input
from wia.tasks.loader import load_scene, load_task
from wia.authoring.commands.design_wia_review_revisions import write, dump

BASE = ROOT / 'explore_to_achieve/wia_v1/review_20261025'
META = LazyMapping(lambda: read_json_input(ARTIFACT_ROOT / 'wia_v1_20261025_revision/assets.json'))
CALIBRATION = ARTIFACT_ROOT / 'wia_v1_20261023/capture_campaign/queue/jobs/capture_b6e079233b9e09ca93dab9c4a43f0d7a/attempts/01/preview_complete.json'
CAMERA = LazyMapping(lambda: read_json_input(CALIBRATION)['initial_agent_bundle']['review_diagnostics']['geometry']['optical_selection']['camera'])


def revise_layout(layout, bindings, alias, seed):
    objects = {obj['name']: obj for obj in layout['objects']}
    centers = {}
    for key in ('candidate_0', 'candidate_1', 'candidate_2'):
        obj = objects[bindings[key]['name']]
        obj['position'][1] = 13.4 + (obj['position'][1] - 13.4) * (.4 / .65)
        local = np.asarray(META[obj['category'] + '/' + obj['model']]['centroid']) * obj['scale']
        centers[key] = np.asarray(obj['position']) + Rotation.from_quat(obj['orientation']).apply(local)
    eye = np.asarray(CAMERA['position_m'])
    cue_x = objects[bindings['cue_low']['name']]['position'][0]
    panel = objects[bindings['mirror']['name']]
    plane_x = panel['position'][0] - panel['scale'][0] / 2
    selected = min(centers, key=lambda key: centers[key][1]) if seed == 101 else max(centers, key=lambda key: centers[key][1])
    rays = {}
    for key, center in centers.items():
        point = center.copy()
        if alias == 'R-A1':
            point[0] = 2 * plane_x - point[0]
        rays[key] = eye + (cue_x - eye[0]) / (point[0] - eye[0]) * (point - eye)
    ys = sorted(point[1] for point in rays.values())
    width = float(min(.14, min(b - a for a, b in zip(ys, ys[1:])) * .55))
    y, z = (float(value) for value in rays[selected][1:])
    height = .12
    for key in ('cue_low', 'cue_high', 'cue_bottom', 'cue_top'):
        obj = objects[bindings[key]['name']]
        if key in ('cue_low', 'cue_high'):
            obj['position'] = [cue_x, y + (-1 if key == 'cue_low' else 1) * (width / 2 + .006), z]
            obj['scale'] = [.008, .012, height + .024]
        else:
            obj['position'] = [cue_x, y, z + (-1 if key == 'cue_bottom' else 1) * (height / 2 + .006)]
            obj['scale'] = [.008, width + .024, .012]
    # CPU screen only: every direct item centroid must be comfortably inside
    # the measured native camera frame. Native support/visibility remain gates.
    rotation = Rotation.from_quat(CAMERA['quaternion_xyzw']).inv()
    intrinsic = np.asarray(CAMERA['intrinsic_matrix'])
    for center in centers.values():
        local = rotation.apply(center - eye)
        pixel = [256 + intrinsic[0, 0] * local[0] / -local[2],
                 256 - intrinsic[1, 1] * local[1] / -local[2]]
        assert local[2] < 0 and all(25 < value < 487 for value in pixel), pixel


def main():
    # Use immutable parents so rerunning after selecting V4 cannot contract
    # the candidate spacing again or turn the output into its own parent.
    source = json.loads((BASE / 'produce_mirror_v2/manifest.json').read_text())
    clearance = json.loads((BASE / 'produce_mirror_v3/reflection_clearance_manifest.json').read_text())
    parents = {(row['document_alias'], row['seed']): row for row in source['entries']}
    parents.update({(row['document_alias'], row['seed']): row for row in clearance['entries']})
    rows = []
    for row in parents.values():
        if row['document_alias'] not in ('R-A1', 'R-A2'):
            continue
        alias, seed = row['document_alias'], row['seed']
        new = deepcopy(row)
        new['variant_id'] = alias.lower().replace('-', '_') + '_seed' + str(seed) + '_produce_v4'
        new['label'] = alias + ' produce / matched compact framing / seed ' + str(seed)
        for kind in ('task', 'scene'):
            original = Path(row[kind + '_file'])
            value = yaml.safe_load(original.read_text())
            bindings = value['private']['bindings'] if kind == 'task' else value['bindings']
            for variant in value['scene']['variants'].values():
                for patch in variant['patches']:
                    if 'layout' in patch.get('parameters', {}):
                        revise_layout(patch['parameters']['layout'], bindings, alias, seed)
            if kind == 'task':
                value['repair'].update(variant_id=new['variant_id'], parent_task_file=row['task_file'],
                    parent_scene_file=row['scene_file'], calibration_receipt=str(CALIBRATION),
                    design_scope='Matched R-A revisions: candidate Y span reduced from +/-0.65 to +/-0.40 m; cue opening height 0.12 m and maximum width 0.14 m. Keep all assets, seed, robot pose, destinations, public task and scoring. Native framing/support validation pending.')
            else:
                value['id'] = 'wia_' + new['variant_id'] + '_scene'
            path = BASE / 'produce_mirror_v4' / new['variant_id'] / (kind + '.yaml')
            write(path, yaml.safe_dump(value, sort_keys=False).encode())
            new[kind + '_file'], new['parent_' + kind + '_file'] = str(path), str(original)
        load_task(new['task_file'], variant='adapted', run_mode='diagnostic')
        load_scene(new['scene_file'], variant='adapted', run_mode='diagnostic')
        rows.append(new)
    dump(BASE / 'produce_mirror_v4/manifest.json', {'schema': 'wia_variant_manifest_v1', 'entries': rows})
    print('Authored four matched appearance framing revisions.')


if __name__ == '__main__':
    main()

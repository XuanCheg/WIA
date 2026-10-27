"""Materialize the main-prescribed T32/T34 native-support review revisions.

Run: /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B
scripts/revise_wia_spatial_supports.py
Then build with scripts/build_wia_spatial_tray.py using the printed paths.
This script writes a new design only; no captures, controls or simulator run.


Canonical invocation: python -m wia.authoring.commands.revise_wia_spatial_supports
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from scipy.spatial.transform import Rotation

from wia.resources import ROOT
BASE = ROOT / 'explore_to_achieve/wia_v1/furnished_v1'
PARENT = BASE / 'spatial_design_v2.json'
DESTINATION = BASE / 'spatial_support_design_v3.json'


def build_design():
    """Apply only the two main-authored pose changes to immutable V2 entries."""
    source_bytes = PARENT.read_bytes()
    parent = json.loads(source_bytes)
    indexed = {row['document_alias']: row for row in parent['entries']}
    entries = []
    for alias in ('T32', 'T34'):
        entry = deepcopy(indexed[alias])
        obj = next(row for row in entry['layout']['objects'] if row['name'] == 'obj1')
        original = deepcopy(obj)
        if alias == 'T32':
            if obj['position'][:2] != [-3.02, -10.61]:
                raise ValueError('T32 V2 pose differs from the reviewed starting point.')
            obj['position'] = [-3.08, -10.61, original['position'][2]]
            obj['orientation'] = (Rotation.from_euler('z', 90, degrees=True)
                                  * Rotation.from_quat(original['orientation'])).as_quat().tolist()
            support = next(row for row in entry['layout']['supported_items'] if row['name'] == 'obj1')
            if (support.get('support') != 'bookcase_owvfik_0'
                    or support.get('support_surface_mode') != 'local_raycast'
                    or support.get('support_alignment_mode') != 'local_raycast'):
                raise ValueError('T32 must retain its reviewed bookcase local-raycast support.')
        else:
            if obj['position'][0] != 21.94:
                raise ValueError('T34 V2 X differs from the reviewed starting point.')
            obj['position'][0] = 21.85
        entry['variant_id'] = alias.lower() + '_furnished_tray_v3'
        entry['version'] = 3
        entry['label'] = alias + ' open-surface retrieval and tray arrangement v3 (native support review revision)'
        entry['layout']['shared_layout_id'] = 'wia_furnished_' + alias + '_support_v3'
        entries.append(entry)
    return {'schema': 'wia_spatial_tray_design_v1', 'entries': entries,
        'parent_design': {'file': str(PARENT), 'sha256': hashlib.sha256(source_bytes).hexdigest()},
        'revision_evidence': {
            'source': 'Main-agent diagnosis from native initialization review; this script ran no native experiment.',
            'T32': 'The vase world Y extent was 0.423 m while bookcase depth was 0.289 m. '
                   'Set root XY to [-3.08, -10.61] and left-multiply the original quaternion by a '
                   '90-degree world-Z rotation, preserving its original tilt and root Z. '
                   'This places the original 0.214 m X extent along shelf Y. Retain local-raycast alignment and support.',
            'T34': 'The backpack maximum world X was 22.267 m while counter maximum X was 22.2567 m. '
                   'Move root X from 21.94 m to 21.85 m only.',
            'controls': 'All other object poses, goals, scoring thresholds, scales, native supports and public goals '
                        'are copied unchanged from the V2 design. Existing versions remain immutable.',
            'native_validation_run': False,
            'status': 'Pending native initialization image review; no manipulation or reference execution.'}}


def main():
    design = build_design()
    data = (json.dumps(design, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if DESTINATION.is_symlink() or (DESTINATION.exists() and DESTINATION.read_bytes() != data):
        raise ValueError('Refusing to overwrite a differing support design: ' + str(DESTINATION))
    if not DESTINATION.exists():
        DESTINATION.write_bytes(data)
    print('Design: ' + str(DESTINATION))
    print('Output: ' + str(BASE / 'spatial_support_v3'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

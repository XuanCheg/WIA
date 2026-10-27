"""Build a separate TM-2 revision with a common apparatus-clear construction start.

Run: python -B scripts/build_wia_temporal_start_v3.py [--check]
Native clearance, stability, and matched Phase-2 observations remain unverified.


Canonical invocation: python -m wia.authoring.commands.build_wia_temporal_start_v3 --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

PARENT = ROOT / 'explore_to_achieve/wia_v1/temporal_posture_v2'
OUTPUT = ROOT / 'explore_to_achieve/wia_v1/temporal_start_v3'
START = {'position': [9.4, 2.7, -0.000020831823348999023], 'yaw_deg': 160.0}


def build():
    """Change only the adapted construction start and version identities."""
    rendered, records, public = {}, [], []
    for history in ('a', 'b', 'c'):
        for kind, model in (('tasks', CategoryTaskDefinition), ('scenes', SceneDefinition)):
            parent = PARENT / kind / f'history_{history}_v2.yaml'
            raw = parent.read_bytes()
            old = yaml.safe_load(raw)
            new = deepcopy(old)
            new.update(id=('wia_tm2_matched_history_v3' if kind == 'tasks'
                           else f'wia_tm2_history_{history}_v3_scene'), version=3, status='pending')
            patches = new['scene']['variants']['adapted']['patches']
            selected = [(i, p) for i, p in enumerate(patches) if p['operation'] == 'robot_pose']
            if len(selected) != 1:
                raise ValueError('Expected exactly one adapted robot construction pose.')
            index, patch = selected[0]
            before = deepcopy(patch)
            patch.update(deepcopy(START))
            normalized = deepcopy(new)
            normalized.update({key: old[key] for key in ('id', 'version', 'status')})
            normalized['scene']['variants']['adapted']['patches'][index] = before
            if normalized != old:
                raise ValueError('Unexpected scientific contract change.')
            valid = model.model_validate(new)
            if kind == 'tasks':
                public.append({'task_id': valid.id, 'task_version': valid.version,
                               **valid.public.model_dump(mode='json')})
            relative = Path(kind) / f'history_{history}_v3.yaml'
            data = yaml.safe_dump(new, sort_keys=False, allow_unicode=False, width=88).encode()
            rendered[relative] = data
            records.append({'parent_file': str(parent), 'parent_sha256': hashlib.sha256(raw).hexdigest(),
                            'output_file': str(OUTPUT / relative), 'sha256': hashlib.sha256(data).hexdigest(),
                            'changed_patch_index': index, 'before': before, 'after': deepcopy(patch)})
    if not all(row == public[0] for row in public):
        raise ValueError('Matched histories require identical complete public payloads.')
    manifest = {'schema': 'wia_tm2_construction_start_v3', 'status': 'pending-native-validation',
                'hypothesis': 'An apparatus-clear common construction start avoids the observed spare-table '
                              'leg collision before source settling; downstream stability remains unproven.',
                'evidence': str(ARTIFACT_ROOT / 'wia_v1_20261023/temporal_a_v2/initialization_constructed.json'),
                'proposal_source': 'Native navigation-grid candidate from the v2 interaction receipt.',
                'scientific_change': 'Adapted construction robot pose only, relative to v2.',
                'construction_start': START, 'earlier_versions_modified': False,
                'public_payload': public[0], 'native_initialization': 'PENDING',
                'native_phase_2_equivalence': 'PENDING', 'reference_execution': 'PENDING',
                'instances': records}
    rendered[Path('manifest.json')] = (json.dumps(manifest, indent=2) + '\n').encode()
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    rendered = build()
    for relative, data in rendered.items():
        path = OUTPUT / relative
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                raise ValueError(f'Generated output differs: {path}')
        else:
            if path.exists() and path.read_bytes() != data:
                raise ValueError(f'Refusing to overwrite a different revision: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    print(('Checked ' if args.check else 'Built ') + str(OUTPUT))


if __name__ == '__main__':
    main()

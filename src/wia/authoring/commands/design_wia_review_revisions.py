"""Build the main-authored October 25 ramp/cover revisions on CPU.

Run: behavior/bin/python -B scripts/design_wia_review_revisions.py
Native setup, coverage, and robot execution are separate pending gates.


Canonical invocation: python -m wia.authoring.commands.design_wia_review_revisions
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import yaml

from wia.resources import ROOT
from wia.tasks.loader import load_scene, load_task
from wia.authoring.commands.build_wia_cover_variants import build_package

OUT = ROOT / 'explore_to_achieve/wia_v1/review_20261025'

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Refusing differing versioned output: ' + str(path))
    path.write_bytes(data)

def dump(path, value):
    write(path, (json.dumps(value, indent=2) + '\n').encode())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ramps():
    entries = []
    for alias in ('T42', 'T43', 'T44'):
        stem = alias.lower()
        family = 'background_ramp_floor_v3' if alias == 'T42' else 'workstation_transfer_v1'
        filename = stem + ('_complete_ground_v3' if alias == 'T42' else '_hall_arch_v2') + '.yaml'
        parent_task = ROOT / 'explore_to_achieve/wia_v1' / family / 'tasks' / filename
        parent_scene = ROOT / 'explore_to_achieve/wia_v1' / family / 'scenes' / filename
        task, scene = (yaml.safe_load(p.read_text()) for p in (parent_task, parent_scene))
        evaluation = deepcopy(task['private']['evaluation'])
        variant = stem + '_red_release_marker_v1'
        changed = 0
        for patch in task['scene']['variants']['adapted']['patches']:
            for obj in patch.get('parameters', {}).get('layout', {}).get('objects', []):
                if obj['name'] == 'release_marker':
                    obj['rgba'] = [.95, .025, .025, 1.]
                    changed += 1
        if changed != 2:
            raise ValueError('Expected matching before/after release markers.')
        def wording(value):
            if isinstance(value, dict): return {k: wording(v) for k, v in value.items()}
            if isinstance(value, list): return [wording(v) for v in value]
            if isinstance(value, str): return value.replace('yellow release', 'red release').replace('yellow marker', 'red marker')
            return value
        task['public'] = wording(task['public'])
        task.update(id=task['id'] + '__' + variant, version=task['version'] + 1, status='pending')
        task['repair'].update(variant_id=variant, parent_task_file=str(parent_task),
            parent_scene_file=str(parent_scene), native_validation_run=False,
            physical_success_verified=False, full_reference_execution='pending',
            revision_scope='Release marker color only; two selected native backgrounds across three proposals.')
        scene.update(id=scene['id'] + '__' + variant, version=scene['version'] + 1,
                     scene=deepcopy(task['scene']), status='pending')
        if task['private']['evaluation'] != evaluation:
            raise AssertionError('Ramp dynamics and scoring must be preserved.')
        paths = {kind: OUT / 'ramps' / variant / (kind + '.yaml') for kind in ('task', 'scene')}
        for kind, value in (('task', task), ('scene', scene)):
            write(paths[kind], yaml.safe_dump(value, sort_keys=False).encode())
        load_task(paths['task'], variant='adapted', run_mode='diagnostic')
        load_scene(paths['scene'], variant='adapted', run_mode='diagnostic')
        entries.append({'document_alias': alias, 'variant_id': variant,
            'label': alias + ' red release marker / ' + ('furnished house' if alias == 'T42' else 'wood arch hall'),
            'task_file': str(paths['task']), 'scene_file': str(paths['scene']),
            'parent_task_file': str(parent_task), 'parent_scene_file': str(parent_scene),
            'variant': 'adapted', 'seed': task['defaults']['seed'],
            'native_validation_run': False, 'physical_success_verified': False})
    dump(OUT / 'ramps/manifest.json', {'schema': 'wia_variant_manifest_v1', 'entries': entries})

def covers():
    specs = [
        ('T49', '9fd3d70ee489c4228f5f44ec', 'dishtowel', 'dtfspn', 2.5, 'dish towel'),
        ('T50', 'fb64ed932434e097adc84053', 'hand_towel', 'klevhj', 2.5, 'hand towel'),
        ('T51', 'e2b21727ae8a25c6b7b860d5', 'tablecloth', 'shgmwu', .65, 'small tablecloth')]
    entries = []
    for alias, key, category, model, scale, label in specs:
        suffix = Path('Physical Structure/Deformable/hall_arch_wood/redesign_20261020') / ('category_' + key + '.yaml')
        entries.append({'document_alias': alias, 'variant_id': alias.lower() + '_diverse_cover_v1',
            'label': alias + ' native ' + label + ' cover',
            'parent_task_file': str(ROOT / 'explore_to_achieve/tasks' / suffix),
            'parent_scene_file': str(ROOT / 'explore_to_achieve/scenes' / suffix),
            'cloth_asset': {'category': category, 'model': model}, 'cloth_scale_multiplier': scale,
            'cloth_mass_kg': 1., 'public_cover_label': label})
    design_path = OUT / 'cover_design_v1.json'
    dump(design_path, {'schema': 'wia_cover_variant_design_v1', 'entries': entries,
        'design_basis': 'Installed default cloth configuration extents; native drape and occlusion remain pending. Asset/scale intervention preserves prior mass and other physics settings for the initial pilot.'})
    from wia.authoring.commands.build_wia_cover_variants import main
    main(['--design-file', str(design_path), '--output-dir', str(OUT / 'covers')])

if __name__ == '__main__':
    ramps()
    covers()

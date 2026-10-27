"""Seed destination proposals from unchanged scene assets. Run: python -m pytest tests/test_extraction.py."""
from collections import Counter
from functools import lru_cache
import hashlib
import os
from pathlib import Path
from wia.authoring.io import read_json

ASSETS = Path(os.environ.get('WIA_ASSETS_ROOT', '/local_home2/zhouxinyang/data/behavior/behavior-1k-assets')).expanduser().resolve()

@lru_cache(maxsize=64)
def scene_inventory(scene, *, assets=ASSETS):
    path = Path(assets) / 'scenes' / scene / 'json' / (scene + '_best.json')
    raw = read_json(path, {})
    return {name: row.get('args', {}) for name, row in raw.get('objects_info', {}).get('init_info', {}).items()}

@lru_cache(maxsize=64)
def scene_object_positions(scene, *, assets=ASSETS):
    """Read source furniture positions for public landmark descriptions only."""
    path = Path(assets) / 'scenes' / scene / 'json' / (scene + '_best.json')
    raw = read_json(path, {})
    objects = raw.get('state', {}).get('registry', {}).get('object_registry', {})
    return {name: row.get('root_link', {}).get('pos') for name, row in objects.items()
            if len(row.get('root_link', {}).get('pos', [])) == 3}

def destination(source, seed, *, task_edit2=False, _inventory=None, _positions=None):
    """Seed an observable existing surface; native fit/access remain unverified."""
    if task_edit2:
        from wia.authoring.converters.task_edit2_destinations import destination_proposal
        return destination_proposal(source, seed)
    raw = source.raw
    scene, room = raw['scene'], raw['room']
    inventory = (_inventory or scene_inventory)(scene)
    room_objects = [row for row in inventory.values()
                    if room in row.get('in_rooms', []) and not row.get('visual_only')]
    choices = []
    for name, row in inventory.items():
        if room not in row.get('in_rooms', []) or row.get('visual_only'):
            continue
        if row.get('category') in ('breakfast_table', 'coffee_table', 'countertop', 'desk', 'console_table'):
            choices.append(row)
    if not choices:
        return None
    choices.sort(key=lambda row: row['name'])
    counts = Counter(row['category'] for row in room_objects)
    descriptions = []
    positions = (_positions or scene_object_positions)(scene)
    landmarks = [row for row in room_objects if counts[row['category']] == 1
                 and row['category'] not in ('floors', 'ceilings', 'carpet', 'electric_switch')
                 and row['name'] in positions]
    for row in choices:
        if counts[row['category']] == 1:
            descriptions.append((row, 'the ' + row['category'].replace('_', ' ') + ' in this room',
                                 {'kind': 'unique_room_category'}))
            continue
        peers = [peer for peer in choices if peer['category'] == row['category']]
        if any(peer['name'] not in positions for peer in peers):
            continue
        for anchor in sorted(landmarks, key=lambda value: value['name']):
            import math
            distances = sorted((math.dist(positions[peer['name']][:2], positions[anchor['name']][:2]),
                                peer['name']) for peer in peers)
            if distances[0][1] != row['name'] or distances[1][0] - distances[0][0] < .1:
                continue
            label = ('the ' + row['category'].replace('_', ' ') + ' nearest the '
                     + anchor['category'].replace('_', ' ') + ' in this room')
            descriptions.append((row, label, {'kind': 'nearest_unique_room_landmark',
                'landmark_binding': {k: anchor[k] for k in ('name', 'category', 'model')},
                'source_root_distances_m': dict((name, distance) for distance, name in distances),
                'native_relation_verified': False}))
            break
    if not descriptions:
        return None
    key = hashlib.sha256(f'{seed}:{source.relative_path}:destination'.encode()).digest()
    selected, label, evidence = descriptions[int.from_bytes(key[:8], 'big') % len(descriptions)]
    return {'binding': {k: selected[k] for k in ('name', 'category', 'model')},
        'public_label': label, 'public_label_evidence': evidence,
        'relation': 'on_top', 'selection_seed': seed, 'source': 'native_scene_inventory_observable_description',
        'fit_verified': False, 'access_verified': False}

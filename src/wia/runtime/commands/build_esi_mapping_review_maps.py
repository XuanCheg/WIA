#!/usr/bin/env python3
"""Build human-only Cognitive Mapping room maps on CPU; generate-only by default.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python src/wia/runtime/commands/build_esi_mapping_review_maps.py --campaign-root /path/to/campaign
--output-dir /path/to/placement_review/mapping_maps
Add --attach-review --queue-root /path/to/queue only after reviewing the manifest.
No OmniGibson import, native engine, robot pose, capture job or renderer is used.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS

for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '2'

import cv2
import numpy as np
import torch
import yaml

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.sim.esi_floorplan_review import FLOORPLAN_LABEL, render_floorplan_review


VERSION = '1.0.0'
ASSETS = Path('/local_home2/zhouxinyang/data/behavior/behavior-1k-assets')
NATIVE_ROOT = Path('/local_home2/zhouxinyang/third_party/BEHAVIOR-1K/OmniGibson/omnigibson')
PROVENANCE = ('Offline official room-segmentation PNG visualization; no native session, '
    'renderer, robot pose or candidate capture. Human review only.')


def encoded_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(encoded_json(value)).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_write(path, data):
    """Keep previously generated artifacts byte-for-byte immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'immutable_artifact_conflict:{path}')
        return
    with path.open('xb') as handle:
        handle.write(data)


def replace_json(path, value, *, previous=None):
    """Replace only a mutable index/card, with a changed-input conflict check."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = encoded_json(value)
    if path.is_file() and path.read_bytes() == data:
        return False
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        if previous is not None and path.read_bytes() != previous:
            raise ValueError(f'review_record_changed_during_attachment:{path}')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def read_sources(campaign_root):
    """Use the current inventory, retaining an explicit row for every source."""
    inventory_path = campaign_root / 'inventory.json'
    inventory = json.loads(inventory_path.read_text())
    corpus = Path(inventory['corpus'])
    rows = []
    for category in inventory['categories']:
        if category['big_task'] != 'Cognitive Mapping':
            continue
        for identity in category['sources']:
            path = corpus / identity
            row = {'big_task': category['big_task'], 'small_task': category['small_task'],
                'source_identity': identity, 'source_json': str(path),
                'scene_model': Path(identity).parts[2]}
            try:
                row['source_sha256'] = sha256(path)
                raw = json.loads(path.read_text())
                if raw.get('big_task') != 'Cognitive Mapping' or raw.get('small_task') != category['small_task']:
                    raise ValueError('inventory_source_category_mismatch')
                row['scene_model'] = raw['scene']
            except (OSError, ValueError, KeyError) as error:
                row['source_error'] = f'{type(error).__name__}:{error}'
            rows.append(row)
    if not rows:
        raise ValueError('inventory_has_no_cognitive_mapping_sources')
    if len({row['source_identity'] for row in rows}) != len(rows):
        raise ValueError('inventory_has_duplicate_mapping_source_identities')
    return inventory_path, sorted(rows, key=lambda row: row['source_identity'])


def official_segmentation(scene_dir, room_categories, resolution):
    """Mirror SegmentationMap._load_map and BaseMap axes without importing them.

    Official loading reads grayscale PNGs, resizes to int(height*.01/resolution)
    with nearest-neighbour, drops the first sorted unique ID (background zero),
    then numbers rooms within each semantic ID using its first matching pixel.
    The helper separately reads the unchanged original PNG for human coverage.
    """
    layout = scene_dir / 'layout'
    instance = cv2.imread(str(layout / 'floor_insseg_0.png'), cv2.IMREAD_GRAYSCALE)
    semantic = cv2.imread(str(layout / 'floor_semseg_0.png'), cv2.IMREAD_GRAYSCALE)
    if instance is None or semantic is None:
        raise ValueError('official_scene_segmentation_png_missing_or_unreadable')
    if instance.shape[0] != instance.shape[1] or instance.shape != semantic.shape:
        raise ValueError('official_scene_segmentation_shape_mismatch')
    size = int(instance.shape[0] * .01 / resolution)
    if size < 1:
        raise ValueError('official_native_resize_has_no_cells')
    instance = cv2.resize(instance, (size, size), interpolation=cv2.INTER_NEAREST)
    semantic = cv2.resize(semantic, (size, size), interpolation=cv2.INTER_NEAREST)
    categories = room_categories.read_text().splitlines()
    semantic_instances = {}
    for identifier in np.unique(instance)[1:]:
        row, col = np.argwhere(instance == identifier)[0]
        semantic_id = int(semantic[row, col])
        semantic_instances.setdefault(semantic_id, []).append(int(identifier))
    sem_names, ins_names, sem_instances = {}, {}, {}
    for semantic_id, identifiers in semantic_instances.items():
        name = categories[semantic_id - 1]
        sem_names[name] = semantic_id
        for index, identifier in enumerate(identifiers):
            instance_name = f'{name}_{index}'
            ins_names[instance_name] = identifier
            sem_instances.setdefault(name, []).append(instance_name)

    def map_to_world(points):
        # Match BaseMap exactly for a point or an N-by-2 collection of points.
        dimension = 0 if points.dim() == 1 else 1
        return torch.flip((points - size / 2.) * resolution, dims=(dimension,))

    return NS(scene_dir=scene_dir, map_default_resolution=.01,
        room_ins_map=torch.tensor(instance), room_sem_map=torch.tensor(semantic),
        map_resolution=resolution, map_size=size, floor_heights=(0.,),
        room_sem_name_to_sem_id=sem_names,
        room_sem_id_to_sem_name={value: key for key, value in sem_names.items()},
        room_ins_name_to_ins_id=ins_names,
        room_ins_id_to_ins_name={value: key for key, value in ins_names.items()},
        room_sem_name_to_ins_name=sem_instances, map_to_world=map_to_world)


def scene_result(scene, sources, output_dir, assets_root, native_config, official_root, image_size):
    scene_dir = assets_root / 'scenes' / scene
    dependencies = [scene_dir / 'layout/floor_insseg_0.png', scene_dir / 'layout/floor_semseg_0.png',
        assets_root / 'metadata/room_categories.txt', native_config,
        official_root / 'maps/segmentation_map.py', official_root / 'maps/map_base.py',
        Path(__file__), ROOT / 'src/wia/runtime/sim/esi_floorplan_review.py']
    source_hashes = {str(path.resolve()): sha256(path) if path.is_file() else None for path in dependencies}
    inputs = {'scene_model': scene, 'version': VERSION, 'image_size': image_size,
        'source_hashes': source_hashes, 'sources': sources}
    fingerprint = digest(inputs)
    directory = output_dir / 'scenes' / scene / fingerprint[:24]
    image_path, metadata_path = directory / 'room_map.png', directory / 'metadata.json'
    base = {'scene_model': scene, 'kind': 'human_floorplan', 'label': FLOORPLAN_LABEL,
        'native_render': False, 'human_only': True, 'agent_exposed': False,
        'generation_fingerprint': fingerprint, 'source_hashes': source_hashes,
        'metadata_path': str(metadata_path), 'provenance': PROVENANCE}
    try:
        missing = [path for path, value in source_hashes.items() if value is None]
        if missing:
            raise FileNotFoundError('map_inputs_missing:' + ','.join(missing))
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get('generation_fingerprint') != fingerprint or
                    metadata.get('kind') != 'human_floorplan' or metadata.get('native_render') is not False or
                    metadata.get('label') != FLOORPLAN_LABEL or sha256(image_path) != metadata['image_sha256']):
                raise ValueError('immutable_map_artifact_validation_failed')
        else:
            configuration = yaml.safe_load(native_config.read_text())
            resolution = float(configuration['scene']['seg_map_resolution'])
            if not np.isfinite(resolution) or resolution <= 0:
                raise ValueError('invalid_native_default_segmentation_resolution')
            segmentation = official_segmentation(scene_dir, assets_root / 'metadata/room_categories.txt', resolution)
            session = NS(task_module=NS(__name__='active_explore.tasks.cognitive_mapping.connectivity'),
                payload={'big_task': 'Cognitive Mapping'}, env=NS(scene=NS(seg_map=segmentation)), robot=None)
            rgb, geometry = render_floorplan_review(session, image_size=image_size)
            original = cv2.imread(str(scene_dir / 'layout/floor_insseg_0.png'), cv2.IMREAD_UNCHANGED)
            original_ids = sorted(int(value) for value in np.unique(original) if value)
            if geometry['all_raster_room_ids'] != original_ids:
                raise ValueError('original_segmentation_room_id_coverage_lost')
            if geometry['robot_marker'] is not None or geometry['native_render'] is not False:
                raise ValueError('offline_map_must_not_invent_native_or_robot_evidence')
            encoded, png = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not encoded:
                raise ValueError('room_map_png_encoding_failed')
            image_bytes = png.tobytes()
            metadata = {**geometry, **base, 'schema': 'esi_mapping_review_map_v1', 'status': 'generated',
                'image_path': str(image_path), 'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                'sources': sources, 'source_scene_dir': str(scene_dir), 'new_native_validation_run': False,
                'native_evaluator_map_modified': False, 'robot_pose_source': None,
                'floor_height_source': 'SegmentationMap default floor_heights=(0.0,), not measured floor contact',
                'official_loader_convention': 'grayscale; int(height*.01/resolution); INTER_NEAREST; '
                    'sorted runtime instance IDs, first pixel semantic ID, per-semantic-ID room numbering'}
            immutable_write(image_path, image_bytes)
            immutable_write(metadata_path, encoded_json(metadata))
        return {**base, 'status': 'generated', 'image_path': str(image_path),
            'image_sha256': metadata['image_sha256'], 'metadata_sha256': sha256(metadata_path),
            'all_raster_room_ids': metadata['all_raster_room_ids'],
            'original_room_ids_lost_by_native_resize': metadata['original_room_ids_lost_by_native_resize'],
            'unregistered_room_ids': metadata['unregistered_room_ids'],
            'image_size_px': metadata['image_size_px']}
    except Exception as error:
        # One bad scene never discards maps for the remaining inventory scenes.
        failure = {**base, 'status': 'failed', 'image_path': None,
            'failure_stage': 'cpu_floorplan_generation', 'failure_reason': f'{type(error).__name__}:{error}',
            'sources': sources, 'new_native_validation_run': False}
        failure_path = output_dir / 'failures' / scene / (digest(failure)[:24] + '.json')
        failure['metadata_path'] = str(failure_path)
        immutable_write(failure_path, encoded_json(failure))
        return failure


def build_maps(campaign_root, output_dir, *, assets_root=ASSETS, native_config=None,
               official_root=NATIVE_ROOT, image_size=1024):
    """Generate immutable scene artifacts and a complete source-to-image index."""
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    campaign_root, output_dir = Path(campaign_root).resolve(), Path(output_dir).resolve()
    assets_root, official_root = Path(assets_root).resolve(), Path(official_root).resolve()
    native_config = Path(native_config or official_root / 'configs/r1_primitives.yaml').resolve()
    inventory_path, sources = read_sources(campaign_root)
    scene_models = sorted({source['scene_model'] for source in sources})
    scenes = []
    for scene in scene_models:
        result = scene_result(scene, [row for row in sources if row['scene_model'] == scene],
            output_dir, assets_root, native_config, official_root, image_size)
        scenes.append(result)
        print(f"{scene}: {result['status']}", flush=True)
    by_scene = {row['scene_model']: row for row in scenes}
    mapped_sources = []
    for source in sources:
        result = by_scene[source['scene_model']]
        mapped_sources.append({**source, **{key: result.get(key) for key in
            ('status', 'image_path', 'image_sha256', 'metadata_path', 'kind', 'label', 'native_render')},
            **({'status': 'failed', 'failure_reason': source['source_error']} if source.get('source_error') else {})})
    manifest = {'schema': 'esi_mapping_review_maps_v1', 'version': VERSION,
        'campaign_root': str(campaign_root), 'inventory_path': str(inventory_path),
        'inventory_sha256': sha256(inventory_path), 'output_dir': str(output_dir),
        'kind': 'human_floorplan', 'label': FLOORPLAN_LABEL, 'native_render': False,
        'provenance': PROVENANCE, 'robot_marker': None, 'robot_pose_invented': False,
        'scene_count': len(scenes), 'scene_status_counts': dict(Counter(row['status'] for row in scenes)),
        'source_count': len(sources), 'mapped_source_count': sum(row['status'] == 'generated' for row in mapped_sources),
        'category_counts': dict(Counter(row['small_task'] for row in sources)),
        'scene_models_with_ids_lost_by_runtime_resize': [row['scene_model'] for row in scenes
            if row.get('original_room_ids_lost_by_native_resize')],
        'native_engine_imported': any(name == 'omnigibson' or name.startswith('omnigibson.') for name in sys.modules),
        'capture_jobs_submitted': 0, 'scenes': scenes, 'sources': mapped_sources}
    replace_json(output_dir / 'manifest.json', manifest)
    return manifest


def attach_review(manifest, queue_root):
    """Append diagnostic attachments only; all existing record values survive."""
    by_source = {row['source_identity']: row for row in manifest['sources']}
    summary = {'updated_records': 0, 'already_attached_records': 0, 'unmatched_mapping_records': [], 'errors': []}
    for path in sorted((Path(queue_root) / 'category_records').glob('*.json')):
        try:
            previous = path.read_bytes()
            record = json.loads(previous)
            if record.get('big_task') != 'Cognitive Mapping':
                continue
            source = by_source.get(record.get('source_identity'))
            if source is None:
                summary['unmatched_mapping_records'].append(str(path))
                continue
            updated = deepcopy(record)
            logs = updated.setdefault('diagnostic_logs', [])
            if source['metadata_path'] not in logs:
                logs.append(source['metadata_path'])
            if source['status'] == 'generated':
                image_path = source['image_path']
                if sha256(image_path) != source['image_sha256']:
                    raise ValueError('diagnostic_image_hash_mismatch')
                images = updated.setdefault('diagnostic_images', [])
                existing_paths = {item.get('path') if isinstance(item, dict) else item for item in images}
                if image_path not in existing_paths:
                    images.append({'path': image_path, 'view': 'global_floorplan', 'kind': 'human_floorplan',
                        'label': FLOORPLAN_LABEL, 'native_render': False, 'human_only': True,
                        'agent_exposed': False, 'provenance': PROVENANCE,
                        'metadata_file': source['metadata_path'], 'image_sha256': source['image_sha256']})
            if updated == record:
                summary['already_attached_records'] += 1
            else:
                replace_json(path, updated, previous=previous)
                summary['updated_records'] += 1
        except Exception as error:
            summary['errors'].append({'record_file': str(path), 'error': f'{type(error).__name__}:{error}'})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, default=ASSETS)
    parser.add_argument('--official-root', type=Path, default=NATIVE_ROOT)
    parser.add_argument('--native-config', type=Path)
    parser.add_argument('--image-size', type=int, default=1024)
    parser.add_argument('--attach-review', action='store_true')
    parser.add_argument('--queue-root', type=Path)
    args = parser.parse_args(argv)
    if args.attach_review and args.queue_root is None:
        parser.error('--attach-review requires --queue-root')
    manifest = build_maps(args.campaign_root, args.output_dir, assets_root=args.assets_root,
        official_root=args.official_root, native_config=args.native_config, image_size=args.image_size)
    summary = {key: manifest[key] for key in ('scene_count', 'scene_status_counts', 'source_count',
        'mapped_source_count', 'native_render', 'native_engine_imported')}
    summary['manifest'] = str(args.output_dir.resolve() / 'manifest.json')
    if args.attach_review:
        summary['review_attachment'] = attach_review(manifest, args.queue_root)
        replace_json(args.output_dir / 'attachment_result.json', summary['review_attachment'])
    print(json.dumps(summary, indent=2))
    return 1 if (manifest['scene_status_counts'].get('failed') or
        manifest['mapped_source_count'] != manifest['source_count'] or
        summary.get('review_attachment', {}).get('errors')) else 0


if __name__ == '__main__':
    raise SystemExit(main())

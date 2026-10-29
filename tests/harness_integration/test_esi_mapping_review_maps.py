"""CPU checks for offline map export, lost room IDs and additive Review attachment.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m pytest tests/test_esi_mapping_review_maps.py
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from wia.runtime.commands.build_esi_mapping_review_maps import (
    FLOORPLAN_LABEL, attach_review, build_maps, official_segmentation, sha256,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def example_campaign(tmp_path):
    assets, official, campaign = (tmp_path / name for name in ('assets', 'official', 'campaign'))
    layout = assets / 'scenes/example/layout'
    layout.mkdir(parents=True)
    raw = np.zeros((220, 220), np.uint8)
    semantic = np.zeros_like(raw)
    raw[20:120, 20:120], semantic[20:120, 20:120] = 1, 1
    raw[219, 219], semantic[219, 219] = 99, 2
    cv2.imwrite(str(layout / 'floor_insseg_0.png'), raw)
    cv2.imwrite(str(layout / 'floor_semseg_0.png'), semantic)
    categories = assets / 'metadata/room_categories.txt'
    categories.parent.mkdir()
    categories.write_text('kitchen\ncorridor\n')
    for relative in ('maps/segmentation_map.py', 'maps/map_base.py'):
        path = official / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Official-loader source hash fixture; never imported.\n')
    config = official / 'configs/r1_primitives.yaml'
    config.parent.mkdir()
    config.write_text('scene:\n  seg_map_resolution: 1.0\n')
    corpus = tmp_path / 'corpus'
    categories = []
    for small, scene, name in [('Connectivity', 'example', 'one'),
                               ('Regional Boundary', 'example', 'two'),
                               ('Connectivity', 'missing_scene', 'three')]:
        identity = f'Cognitive Mapping/{small}/{scene}/full_scene/{name}.json'
        write_json(corpus / identity, {'big_task': 'Cognitive Mapping', 'small_task': small, 'scene': scene})
        category = next((row for row in categories if row['small_task'] == small), None)
        if category is None:
            category = {'big_task': 'Cognitive Mapping', 'small_task': small, 'sources': []}
            categories.append(category)
        category['sources'].append(identity)
    write_json(campaign / 'inventory.json', {'corpus': str(corpus), 'categories': categories})
    return campaign, assets, official


def test_official_semantic_room_numbering_and_native_coordinate_convention(tmp_path):
    layout = tmp_path / 'scene/layout'
    layout.mkdir(parents=True)
    instance = np.array([[0, 1, 7], [9, 1, 7], [0, 0, 0]], np.uint8)
    semantic = np.where(instance == 9, 2, np.where(instance != 0, 1, 0)).astype(np.uint8)
    cv2.imwrite(str(layout / 'floor_insseg_0.png'), instance)
    cv2.imwrite(str(layout / 'floor_semseg_0.png'), semantic)
    categories = tmp_path / 'room_categories.txt'
    categories.write_text('kitchen\ncorridor\n')
    segmentation = official_segmentation(layout.parent, categories, .01)
    assert segmentation.room_ins_id_to_ins_name == {1: 'kitchen_0', 7: 'kitchen_1', 9: 'corridor_0'}
    assert segmentation.room_sem_name_to_ins_name == {'kitchen': ['kitchen_0', 'kitchen_1'], 'corridor': ['corridor_0']}
    assert torch.equal(segmentation.room_ins_map, torch.tensor(instance))
    assert segmentation.floor_heights == (0.,)
    assert torch.allclose(segmentation.map_to_world(torch.tensor([0., 2.])), torch.tensor([.005, -.015]))


def test_export_keeps_original_lost_id_provenance_failure_coverage_and_immutable_cache(tmp_path, monkeypatch):
    campaign, assets, official = example_campaign(tmp_path)
    output = tmp_path / 'output'
    # Neither a native SDK nor a renderer can be imported during the export.
    import builtins
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        assert name.split('.')[0] not in {'omnigibson', 'isaacsim', 'omni'}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    first = build_maps(campaign, output, assets_root=assets, official_root=official, image_size=256)
    assert first['scene_count'] == 2 and first['source_count'] == 3 and first['mapped_source_count'] == 2
    assert first['scene_status_counts'] == {'generated': 1, 'failed': 1}
    assert not first['native_engine_imported'] and not first['native_render']
    assert first['scene_models_with_ids_lost_by_runtime_resize'] == ['example']
    success = next(row for row in first['scenes'] if row['status'] == 'generated')
    failed = next(row for row in first['scenes'] if row['status'] == 'failed')
    assert success['all_raster_room_ids'] == [1, 99]
    assert success['original_room_ids_lost_by_native_resize'] == [99]
    assert failed['image_path'] is None and 'map_inputs_missing' in failed['failure_reason']
    assert Path(failed['metadata_path']).is_file()
    metadata = json.loads(Path(success['metadata_path']).read_text())
    assert metadata['kind'] == 'human_floorplan' and metadata['label'] == FLOORPLAN_LABEL
    assert metadata['robot_marker'] is None and metadata['robot_pose_source'] is None
    assert metadata['native_render'] is False and metadata['new_native_validation_run'] is False
    assert metadata['native_map_resolution_m'] == 1. and metadata['original_map_resolution_m'] == .01
    assert metadata['raster_source'] == 'original_native_layout_png'
    assert metadata['unregistered_room_ids'] == [99]
    assert metadata['rooms'][-1]['name'] == 'room_id_99'
    assert all(metadata['source_hashes'].values())
    assert len(metadata['sources']) == 2
    assert cv2.imread(success['image_path']).shape[2] == 3
    assert sha256(success['image_path']) == success['image_sha256']
    paths = [Path(success['image_path']), Path(success['metadata_path']), output / 'manifest.json']
    original = {str(path): (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    second = build_maps(campaign, output, assets_root=assets, official_root=official, image_size=256)
    assert first == second
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == original[str(path)] for path in paths)
    assert not (output / 'attachment_result.json').exists()
    assert not any(name == 'omnigibson' or name.startswith('omnigibson.') for name in sys.modules)


def test_attachment_is_additive_idempotent_and_preserves_approval_failure_and_job_identity(tmp_path):
    campaign, assets, official = example_campaign(tmp_path)
    manifest = build_maps(campaign, tmp_path / 'maps', assets_root=assets, official_root=official, image_size=256)
    queue = tmp_path / 'queue'
    cards = queue / 'category_records'
    originals = {}
    for index, source in enumerate(manifest['sources']):
        value = {'candidate_id': f'candidate_{index}', 'source_identity': source['source_identity'],
            'big_task': source['big_task'], 'small_task': source['small_task'],
            'fingerprint': f'unchanged_{index}', 'job_id': f'capture_{index}',
            'status': 'conversion_failed', 'failure_stage': 'native_load_failed',
            'failure_reason': 'Keep the original failure.', 'approval': {'approved': True, 'note': 'Preserve me.'},
            'diagnostic_images': [{'path': '/old/native.png', 'kind': 'native', 'label': 'Old native frame'}],
            'diagnostic_logs': ['/old/native.log'], 'receipt': {'complete': False},
            'unknown_future_field': {'values': [1, 2, 3]}}
        path = cards / f'candidate_{index}.json'
        write_json(path, value)
        originals[path] = deepcopy(value)
    other = cards / 'other_category.json'
    write_json(other, {'big_task': 'Action Sequencing', 'diagnostic_images': [], 'approval': True})
    other_bytes = other.read_bytes()
    result = attach_review(manifest, queue)
    assert result == {'updated_records': 3, 'already_attached_records': 0,
        'unmatched_mapping_records': [], 'errors': []}
    source_index = {row['source_identity']: row for row in manifest['sources']}
    for path, original in originals.items():
        updated = json.loads(path.read_text())
        assert {key: value for key, value in updated.items() if key not in ('diagnostic_images', 'diagnostic_logs')} == {
            key: value for key, value in original.items() if key not in ('diagnostic_images', 'diagnostic_logs')}
        assert updated['diagnostic_images'][:1] == original['diagnostic_images']
        assert updated['diagnostic_logs'][:1] == original['diagnostic_logs']
        source = source_index[original['source_identity']]
        assert updated['diagnostic_logs'][-1] == source['metadata_path']
        if source['status'] == 'generated':
            assert len(updated['diagnostic_images']) == 2
            attachment = updated['diagnostic_images'][-1]
            assert attachment['kind'] == 'human_floorplan'
            assert attachment['label'] == FLOORPLAN_LABEL and attachment['native_render'] is False
        else:
            assert updated['diagnostic_images'] == original['diagnostic_images']
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in originals}
    repeated = attach_review(manifest, queue)
    assert repeated['updated_records'] == 0 and repeated['already_attached_records'] == 3
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == before[path] for path in originals)
    assert other.read_bytes() == other_bytes

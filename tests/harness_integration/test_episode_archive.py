"""Validate compact evidence without changing live observations. Run: pytest tests/test_episode_archive.py."""
import gzip
import io
import json

import numpy as np
from PIL import Image

from core.episode_archive import EpisodeArchive
from wia.runtime.sim.esi_safety import safety_score
from harness_integration.test_agent_observation import source_and_rig


def test_archive_deduplicates_and_preserves_native_model_pixels(tmp_path):
    source, _ = source_and_rig()
    observation = source.capture(action_boundary=True).observation
    before = observation.to_payload()
    archive = EpisodeArchive(tmp_path, max_edge=16)
    first = archive.save(observation, tmp_path / 'before')
    count = len(list(archive.blobs.iterdir()))
    second = archive.save(observation, tmp_path / 'after')
    assert len(list(archive.blobs.iterdir())) == count
    for metadata_path in (first, second):
        metadata = json.loads(metadata_path.read_text())
        for name, info in metadata['images'].items():
            if info is None:
                continue
            path = metadata_path.parent / info['path']
            with Image.open(path) as image:
                assert max(image.size) <= 16
            pixels = observation.rgb_images()[name]
            exact = archive.model_image(observation, name, pixels, 'identity')
            with Image.open(exact) as image:
                assert np.array_equal(np.asarray(image), pixels)
            assert archive.model_image(observation, name, pixels, 'identity') == exact
        for name, info in metadata['sensors'].items():
            if info is not None:
                with gzip.open(metadata_path.parent / info['path'], 'rb') as handle:
                    restored = np.load(handle, allow_pickle=False)
                assert restored.tobytes() == getattr(observation.sensors, name).tobytes()
    assert observation.to_payload() == before


def test_safety_does_not_infer_safe_from_calibration_flag():
    assert safety_score([], telemetry_complete=True, rubric_calibrated=True) is None
    event = {'unsafe': None, 'attribution': 'agent', 'kind': 'contact'}
    assert safety_score([event], telemetry_complete=True, rubric_calibrated=True, coverage_verified=True) is None
    assert safety_score([{**event, 'unsafe': True}], telemetry_complete=False, rubric_calibrated=False) == 0

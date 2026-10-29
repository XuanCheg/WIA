"""Stable episode depth authorization. Run .venv/bin/python -m pytest -q this file."""
import asyncio
from dataclasses import replace
import json

import numpy as np
import pytest

from core.agent.depth_binding import ActiveDepthSnapshot, EpisodeDepthBinding, atomic_json
from core.agent.native_runtime import EpisodeDepthMCP
from core.sensor_observations import SensorReadings
from harness_integration.test_raw_depth_inputs import depth_observation


@pytest.mark.parametrize("frozen", [False, True])
def test_one_real_mcp_connection_refreshes_or_freezes_depth_correctly(frozen, tmp_path):
    a = depth_observation()
    b = replace(a, snapshot_id=2, action_step=1,
                sensors=SensorReadings(head_depth_m=np.ones((12, 16), np.float32) * np.float32(2.123456)))
    binding = EpisodeDepthBinding(tmp_path, "episode-a")
    condition = "initial_view_only" if frozen else "current"

    async def exercise():
        connection = EpisodeDepthMCP(tmp_path, "episode-a")
        await connection.start()
        try:
            names = {tool.name for tool in connection.tools}
            assert names == {"read_depth_pixels", "read_depth_region", "read_rgb_camera"}
            binding.publish(a, canonical_snapshot_id=1, decision_step=1, observation_condition=condition)
            first = await connection.call("read_depth_pixels", {"sensor": "head_depth_m", "pixels_uv": [[4, 0]]})
            binding.publish(a if frozen else b, canonical_snapshot_id=2, decision_step=2, observation_condition=condition)
            second = await connection.call("read_depth_pixels", {"sensor": "head_depth_m", "pixels_uv": [[4, 0]]})
            assert first.structured_content["snapshot_id"] == 1
            assert second.structured_content["snapshot_id"] == (1 if frozen else 2)
            assert second.structured_content["optical_z_m"] == [float((a if frozen else b).sensors.head_depth_m[0, 4])]
            binding.close()
            refused = await connection.call("read_depth_pixels", {"sensor": "head_depth_m", "pixels_uv": [[4, 0]]})
            assert refused.is_error and not refused.structured_content["ok"]
        finally:
            await connection.close()

    asyncio.run(exercise())


def test_depth_binding_rejects_other_episode_tampering_and_stale_snapshots(tmp_path):
    a = depth_observation()
    binding = EpisodeDepthBinding(tmp_path, "episode-a")
    published = binding.publish(a, canonical_snapshot_id=1, decision_step=1, observation_condition="current")
    assert binding.publish(a, canonical_snapshot_id=1, decision_step=1, observation_condition="current") == published
    with pytest.raises(ValueError):
        binding.publish(a, canonical_snapshot_id=2, decision_step=2, observation_condition="current")
    with pytest.raises(ValueError, match="unavailable"):
        ActiveDepthSnapshot(tmp_path, "episode-b").pixels("head_depth_m", [[4, 0]])
    reader = ActiveDepthSnapshot(tmp_path, "episode-a")
    assert reader.pixels("head_depth_m", [[4, 0]])["snapshot_id"] == 1
    value = json.loads((tmp_path / "active.json").read_text())
    value["snapshot_directory"] = "../../private"
    atomic_json(tmp_path / "active.json", value)
    with pytest.raises(ValueError, match="unavailable"):
        reader.pixels("head_depth_m", [[4, 0]])

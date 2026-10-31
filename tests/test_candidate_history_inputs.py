"""Check public candidate history without models or GPUs. Run: PYTHONPATH=src pytest -q tests/test_candidate_history_inputs.py."""
import base64
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from wia.runtime.agent.candidate_history import (MAX_FRAMES, METADATA_KEYS, SCHEMA,
                                               attach_history_images, decode_history)


def encoded(pixels, *, format="PNG"):
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=format)
    raw = buffer.getvalue()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest(), raw


def inputs(count=3):
    pixels = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    frames, expected = [], []
    for index in range(count):
        original = (pixels + index * 7).astype(np.uint8)
        data, sha, _ = encoded(original)
        expected.append(original)
        frames.append({"index": index, "snapshot_id": index + 1,
                       "captured_at": f"2026-10-30T00:00:00.{index:06}+00:00",
                       "simulation_time_s": float(index // 2), "camera": "agentview_rgb",
                       "png_base64": data, "sha256": sha})
    payload = {"schema": SCHEMA, "authored_intervention": True, "robot_actions": 0, "model_calls": 0, "frames": frames}
    current = SimpleNamespace(snapshot_id=max(count, 1), agentview_rgb=pixels,
                              sensors=SimpleNamespace(head_depth_m=np.arange(77, dtype=np.float32).reshape(7, 11)))
    return payload, current, expected


def test_no_history_does_not_inspect_observation_or_modify_agent():
    assert decode_history(None, object()) == ()
    agent = object()
    restore, metadata = attach_history_images(agent, ())
    assert metadata == []
    restore()
    payload, _, _ = inputs(0)
    assert decode_history(payload, object()) == ()


def test_decoded_history_preserves_exact_png_pixels_shape_and_equal_simulation_times():
    payload, current, expected = inputs()
    before = deepcopy(payload)
    history = decode_history(payload, current)
    assert type(history) is tuple and len(history) == 3
    assert [frame.simulation_time_s for frame in history] == [0., 0., 1.]
    assert [frame.snapshot_id for frame in history] == [1, 2, current.snapshot_id]
    for frame, original in zip(history, expected):
        assert np.array_equal(frame.pixels, original)
        assert frame.pixels.shape == original.shape and frame.pixels.dtype == np.uint8
        assert hashlib.sha256(frame.png_bytes).hexdigest() == frame.sha256
        assert frame.png_bytes == base64.b64decode(payload["frames"][frame.index]["png_base64"])
        with pytest.raises(ValueError):
            frame.pixels[0, 0, 0] = 8
        with pytest.raises(ValueError):
            frame.pixels.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            frame.snapshot_id = 9
    assert payload == before


@pytest.mark.parametrize("field", ["objects", "object_truth", "conditions", "events", "used_tool", "waypoints", "eligible_unit"])
@pytest.mark.parametrize("level", ["payload", "frame"])
def test_private_or_event_answer_fields_are_rejected_at_each_schema_level(field, level):
    payload, current, _ = inputs()
    target = payload if level == "payload" else payload["frames"][0]
    target[field] = "PRIVATE_EVENT_ANSWER_CANARY"
    with pytest.raises(ValueError, match="exactly"):
        decode_history(payload, current)


@pytest.mark.parametrize("change", ["schema", "not_authored", "actions", "models", "boolean_actions", "missing_frame_field", "empty_payload"])
def test_invalid_or_missing_public_contract_fields_do_not_silently_pass(change):
    payload, current, _ = inputs()
    if change == "schema":
        payload["schema"] = "unknown"
    elif change == "not_authored":
        payload["authored_intervention"] = False
    elif change == "actions":
        payload["robot_actions"] = 1
    elif change == "models":
        payload["model_calls"] = 1
    elif change == "boolean_actions":
        payload["robot_actions"] = False
    elif change == "missing_frame_field":
        del payload["frames"][0]["sha256"]
    else:
        payload = {}
    with pytest.raises(ValueError):
        decode_history(payload, current)


@pytest.mark.parametrize("change", ["digest", "base64", "jpeg", "rgba", "grayscale", "dimensions", "too_many"])
def test_tampered_non_png_or_transformed_pixels_are_rejected(change):
    payload, current, _ = inputs()
    frame = payload["frames"][0]
    if change == "digest":
        frame["sha256"] = "0" * 64
    elif change == "base64":
        frame["png_base64"] = "not valid png base64!"
    elif change == "too_many":
        payload["frames"] = [deepcopy(frame) for _ in range(MAX_FRAMES + 1)]
    else:
        pixels = current.agentview_rgb
        if change == "rgba":
            pixels = np.concatenate([pixels, np.full((*pixels.shape[:2], 1), 255, dtype=np.uint8)], axis=2)
        elif change == "grayscale":
            pixels = pixels[:, :, 0]
        elif change == "dimensions":
            pixels = pixels[:3, :4]
        frame["png_base64"], frame["sha256"], _ = encoded(pixels, format="JPEG" if change == "jpeg" else "PNG")
    with pytest.raises(ValueError):
        decode_history(payload, current)


def test_image_size_bounds_are_checked_without_resizing(monkeypatch):
    from wia.runtime.agent import candidate_history

    payload, current, _ = inputs()
    current.agentview_rgb = np.zeros((candidate_history.MAX_IMAGE_SIDE + 1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="bounded"):
        decode_history(payload, current)
    payload, current, _ = inputs()
    monkeypatch.setattr(candidate_history, "MAX_PNG_BYTES", 16)
    with pytest.raises(ValueError, match="size bound"):
        decode_history(payload, current)


@pytest.mark.parametrize("change", ["duplicate_snapshot", "future_snapshot", "zero_snapshot", "boolean_snapshot", "index_gap", "negative_index", "backward_time", "nan_time", "infinite_time", "boolean_time", "bad_stamp", "naive_stamp", "wrist", "global"])
def test_clock_snapshot_and_head_only_boundaries(change):
    payload, current, _ = inputs()
    frames = payload["frames"]
    if change == "duplicate_snapshot":
        frames[1]["snapshot_id"] = frames[0]["snapshot_id"]
    elif change == "future_snapshot":
        frames[-1]["snapshot_id"] = current.snapshot_id + 1
    elif change == "zero_snapshot":
        frames[0]["snapshot_id"] = 0
    elif change == "boolean_snapshot":
        frames[0]["snapshot_id"] = True
    elif change == "index_gap":
        frames[1]["index"] = 5
    elif change == "negative_index":
        frames[0]["index"] = -1
    elif change == "backward_time":
        frames[1]["simulation_time_s"] = -1.
    elif change == "nan_time":
        frames[1]["simulation_time_s"] = float("nan")
    elif change == "infinite_time":
        frames[1]["simulation_time_s"] = float("inf")
    elif change == "boolean_time":
        frames[1]["simulation_time_s"] = False
    elif change == "bad_stamp":
        frames[0]["captured_at"] = "PRIVATE_ANSWER_CANARY"
    elif change == "naive_stamp":
        frames[0]["captured_at"] = "2026-10-30T00:00:00"
    else:
        frames[0]["camera"] = "wrist_rgb" if change == "wrist" else "global_diagonal"
    with pytest.raises(ValueError):
        decode_history(payload, current)


def test_unrepresentable_simulation_time_is_rejected_as_nonfinite():
    payload, current, _ = inputs()
    payload["frames"][0]["simulation_time_s"] = 10 ** 1000
    with pytest.raises(ValueError, match="finite"):
        decode_history(payload, current)


class Agent:
    def __init__(self):
        self.calls = []

    def request_images(self, request):
        self.calls.append(request)
        return request.normal_images

    def depth_snapshot(self, request):
        return request.sensor_inputs.sensors.head_depth_m


def test_attachment_prepends_exact_history_on_every_request_and_restores_original_descriptor():
    payload, current, expected = inputs()
    history = decode_history(payload, current)
    agent = Agent()
    original_method = agent.request_images
    original_depth = agent.depth_snapshot
    assert "request_images" not in vars(agent)
    restore, metadata = attach_history_images(agent, history)
    wrapped = agent.request_images
    restore_again, metadata_again = attach_history_images(agent, history)
    assert restore_again is restore and agent.request_images is wrapped
    assert metadata_again == metadata
    assert all(set(item) == set(METADATA_KEYS) for item in metadata)
    assert "event" not in json.dumps(metadata) and "PRIVATE" not in json.dumps(metadata)
    for request_index in range(3):
        current_pixels = current.agentview_rgb + request_index
        normal = [("agentview_rgb", current_pixels, "Current native head camera", "identity"),
                  ("wrist_rgb", current.agentview_rgb, "Current wrist camera", "identity")]
        request = SimpleNamespace(normal_images=normal, sensor_inputs=current)
        before_depth = current.sensors.head_depth_m.tobytes()
        images = agent.request_images(request)
        assert len(images) == len(history) + len(normal)
        assert [name for name, *_ in images[:3]] == [f"candidate_history_{index:02}_agentview_rgb" for index in range(3)]
        for index, (_, pixels, label, transform) in enumerate(images[:3]):
            assert pixels is history[index].pixels and np.array_equal(pixels, expected[index])
            assert label == f"Authored event history frame {index}, native head camera, simulation time {history[index].simulation_time_s} s; historical view"
            assert transform == "identity"
        assert images[3] is normal[0] and images[4] is normal[1]
        assert request.sensor_inputs is current and current.snapshot_id == 3
        assert agent.depth_snapshot(request) is current.sensors.head_depth_m
        assert current.sensors.head_depth_m.tobytes() == before_depth
    assert len(agent.calls) == 3
    assert agent.depth_snapshot.__func__ is original_depth.__func__
    metadata[0]["sha256"] = "modified caller context"
    assert history[0].sha256 != metadata[0]["sha256"]
    restore()
    restore()
    assert "request_images" not in vars(agent)
    assert agent.request_images.__func__ is original_method.__func__
    assert agent.request_images.__self__ is agent


def test_empty_history_keeps_existing_image_callable_and_nonempty_requires_it():
    agent = Agent()
    before = agent.request_images
    restore, metadata = attach_history_images(agent, ())
    assert agent.request_images.__func__ is before.__func__ and metadata == []
    restore()
    payload, current, _ = inputs()
    with pytest.raises(ValueError, match="callable"):
        attach_history_images(object(), decode_history(payload, current))


def test_restore_preserves_an_original_instance_override_and_prevents_double_wrapping():
    payload, current, _ = inputs()
    history = decode_history(payload, current)
    agent = Agent()
    override = lambda request: request.normal_images
    agent.request_images = override
    restore, _ = attach_history_images(agent, history)
    with pytest.raises(ValueError, match="already attached"):
        attach_history_images(agent, history[:1])
    restore()
    assert agent.request_images is override

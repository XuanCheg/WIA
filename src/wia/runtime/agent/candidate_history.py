"""Decode public authored head-image history and attach it to every model request.

Run CPU checks: ``PYTHONPATH=src pytest -q tests/test_candidate_history_inputs.py``.
Pass the value of control-claim ``candidate_observation_history`` directly to
``decode_history`` after the first current observation. The allowlist contains
only public image identities and clocks, never object state or event answers.
Images retain their native RGB dimensions; no crop or resize is performed.
Only ``agent.request_images`` is wrapped, and its current-view behavior remains
live. Restore the returned callback on episode exit. Diagnostic backends using
``image_max_edge`` must retain their existing full-resolution/default setting:
their separate native-name shape mapper does not accept custom history names.
"""
from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import hashlib
import io
import math
from numbers import Real
import re

import numpy as np
from PIL import Image


SCHEMA = "wia_candidate_observation_history_v1"
MAX_FRAMES = 24
MAX_IMAGE_SIDE = 2048  # Existing SceneRuntime camera_size upper bound.
MAX_PNG_BYTES = 16 * 1024 * 1024
PAYLOAD_KEYS = frozenset({"schema", "authored_intervention", "robot_actions", "model_calls", "frames"})
FRAME_KEYS = frozenset({"index", "snapshot_id", "captured_at", "simulation_time_s", "camera", "png_base64", "sha256"})
METADATA_KEYS = ("index", "snapshot_id", "captured_at", "simulation_time_s", "camera", "sha256")
_ATTACHMENT = "_wia_candidate_history_attachment"
_MISSING = object()


@dataclass(frozen=True, slots=True, eq=False)
class HistoryFrame:
    """An immutable public frame; pixels are backed by immutable owned bytes."""

    index: int
    snapshot_id: int
    captured_at: str
    simulation_time_s: float
    camera: str
    sha256: str
    png_bytes: bytes = field(repr=False)
    pixels: np.ndarray = field(repr=False)

    def metadata(self):
        return {key: getattr(self, key) for key in METADATA_KEYS}


def _exact_keys(value, allowed, label):
    if not isinstance(value, Mapping) or set(value) != allowed:
        raise ValueError(f"{label} must contain exactly the public schema keys.")


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    return number


def _capture_stamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError("captured_at must be a timezone-aware ISO8601 string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        valid = parsed.utcoffset() is not None and math.isfinite(parsed.timestamp())
    except (ValueError, OverflowError, OSError):
        valid = False
    if not valid:
        raise ValueError("captured_at must be a finite timezone-aware ISO8601 timestamp.")
    return value


def _native_shape(observation):
    pixels = getattr(observation, "agentview_rgb", None)
    if pixels is None and callable(getattr(observation, "rgb_images", None)):
        pixels = observation.rgb_images().get("agentview_rgb")
    if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3
            or pixels.shape[2] != 3 or not all(0 < side <= MAX_IMAGE_SIDE for side in pixels.shape[:2])):
        raise ValueError("Current observation must supply bounded native uint8 RGB head pixels.")
    return pixels.shape


def _decode_png(encoded, expected_sha, expected_shape):
    if (not isinstance(encoded, str) or not encoded
            or len(encoded) > 4 * ((MAX_PNG_BYTES + 2) // 3)):
        raise ValueError("History PNG base64 is missing or exceeds the encoded size bound.")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("History SHA256 must be a canonical lowercase hexadecimal digest.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("History image must contain valid PNG base64.") from error
    if len(raw) > MAX_PNG_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("History PNG SHA256 does not match its actual bytes or exceeds its size bound.")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("History image must be lossless PNG.")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if (image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1
                    or (image.height, image.width, 3) != expected_shape):
                raise ValueError("History must preserve the original native RGB head shape in a single PNG frame.")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            pixels = np.asarray(image)
            if pixels.dtype != np.uint8 or pixels.shape != expected_shape:
                raise ValueError("History PNG must decode to original uint8 RGB pixels.")
            # An immutable bytes owner prevents callers from re-enabling writes.
            frozen = np.frombuffer(pixels.tobytes(), dtype=np.uint8).reshape(expected_shape)
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError("History image is not a valid bounded lossless RGB PNG.") from error
    return raw, frozen


def decode_history(payload, current_observation):
    """Return immutable chronological frames, or ``()`` when no payload was supplied.

    The exact simulator schema uses contiguous zero-based frame indices, positive
    increasing snapshot IDs, timezone-aware ISO capture stamps, and nondecreasing
    finite native simulation times. Equal simulation times permit render-only
    frames. The final history snapshot may equal the first current observation.
    Each image must match that observation's native head shape (up to 2048²),
    with at most 24 frames and 16 MiB of PNG bytes per frame.
    """
    if payload is None:
        return ()
    _exact_keys(payload, PAYLOAD_KEYS, "Candidate history")
    if (payload["schema"] != SCHEMA or payload["authored_intervention"] is not True
            or type(payload["robot_actions"]) is not int or payload["robot_actions"] != 0
            or type(payload["model_calls"]) is not int or payload["model_calls"] != 0):
        raise ValueError("Candidate history must declare the public authored zero-action, zero-model-call intervention.")
    frames = payload["frames"]
    if not isinstance(frames, list) or len(frames) > MAX_FRAMES:
        raise ValueError("Candidate history frames must be a list with at most 24 images.")
    if not frames:
        return ()
    current_id = getattr(current_observation, "snapshot_id", None)
    if type(current_id) is not int or current_id < 1:
        raise ValueError("Current observation requires a positive native snapshot ID.")
    expected_shape = _native_shape(current_observation)
    decoded, previous_id, previous_time = [], 0, -math.inf
    for expected_index, frame in enumerate(frames):
        _exact_keys(frame, FRAME_KEYS, "History frame")
        index, snapshot = frame["index"], frame["snapshot_id"]
        if type(index) is not int or index != expected_index:
            raise ValueError("History indices must be contiguous and increase from zero.")
        if type(snapshot) is not int or not previous_id < snapshot <= current_id:
            raise ValueError("History snapshot IDs must strictly increase and not exceed the current observation.")
        now = _finite(frame["simulation_time_s"], "History simulation_time_s")
        if now < previous_time:
            raise ValueError("History native simulation times must not go backward.")
        if frame["camera"] != "agentview_rgb":
            raise ValueError("Candidate history permits the native head camera only.")
        captured_at = _capture_stamp(frame["captured_at"])
        raw, pixels = _decode_png(frame["png_base64"], frame["sha256"], expected_shape)
        decoded.append(HistoryFrame(index, snapshot, captured_at, now, "agentview_rgb", frame["sha256"], raw, pixels))
        previous_id, previous_time = snapshot, now
    return tuple(decoded)


def attach_history_images(agent, history):
    """Prepend raw history on every request and return ``(restore, metadata_list)``.

    An empty history changes nothing, including agents without request_images.
    Attaching the same decoded tuple again reuses one wrapper. A different
    active history must first be restored. Only request_images is overridden;
    normal current RGB, native snapshots, depth, and observation state are kept.
    """
    if type(history) is not tuple or any(type(frame) is not HistoryFrame for frame in history):
        raise ValueError("Attach only the immutable tuple returned by decode_history.")
    if not history:
        return lambda: None, []
    original = getattr(agent, "request_images", None)
    if not callable(original):
        raise ValueError("Nonempty candidate history requires callable agent.request_images.")
    active = getattr(original, _ATTACHMENT, None)
    if active is not None:
        if active["history"] is not history:
            raise ValueError("Another candidate history is already attached; restore it first.")
        return active["restore"], [frame.metadata() for frame in history]
    existing = getattr(agent, "__dict__", {}).get("request_images", _MISSING)
    images = tuple((f"candidate_history_{frame.index:02}_agentview_rgb", frame.pixels,
                    f"Authored event history frame {frame.index}, native head camera, simulation time {frame.simulation_time_s} s; historical view",
                    "identity") for frame in history)
    restored = False

    @wraps(original)
    def request_images(*args, **kwargs):
        return [*images, *original(*args, **kwargs)]

    def restore():
        nonlocal restored
        if restored:
            return
        if getattr(agent, "request_images", None) is not request_images:
            raise RuntimeError("agent.request_images changed after history attachment; refusing to overwrite another wrapper.")
        if existing is _MISSING:
            delattr(agent, "request_images")
        else:
            setattr(agent, "request_images", existing)
        restored = True

    setattr(request_images, _ATTACHMENT, {"history": history, "restore": restore})
    setattr(agent, "request_images", request_images)
    return restore, [frame.metadata() for frame in history]

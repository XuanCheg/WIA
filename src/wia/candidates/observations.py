"""Private evidence from mounted RGB actually present in an observation.

Run: PYTHONPATH=src python -m pytest -q tests/test_candidate_observations.py

``CandidateObservationEvidence(session, parameters).record(observation, frame)``
reads existing native camera observations and returns JSON-safe private evidence.
It never renders, steps, enables modalities, changes objects/state, or adds text
to policy observations. Only registered robot-mounted views whose RGB exactly
matches ``observation.rgb_images()`` can establish visibility. A caller may pass
``set(receipt['visible_names'])`` to CandidateEvents separately.

H1 uses a limited authored-marker color test inside the selected instance mask;
this is not general semantic recognition. Neither RGB matching nor this receipt
certifies policy delivery or native event-history continuity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from wia.candidates.high import _overlap
from wia.runtime.sim.esi_visual_evidence import _labels


MIN_INSTANCE_PIXELS = 64
MIN_INSTANCE_EXTENT_PX = 8
MIN_MARKER_PIXELS = 16


def _array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _instance_matches(label, identity):
    if isinstance(label, str):
        path = identity.prim_path
        return label == identity.name or bool(path and (label == path or label.startswith(path + "/")))
    if isinstance(label, dict):
        return any(_instance_matches(label.get(key), identity)
                   for key in ("name", "object_name", "prim_path", "primPath", "path", "label"))
    if isinstance(label, (list, tuple)):
        return any(_instance_matches(value, identity) for value in label)
    return False


def _mask_evidence(mask):
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    extent = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else 0
    return {"instance_pixels": int(len(xs)), "minimum_extent_px": extent,
            "bbox_xyxy_px": bbox,
            "visible": len(xs) >= MIN_INSTANCE_PIXELS and extent >= MIN_INSTANCE_EXTENT_PX}


def _marker_evidence(rgb, mask):
    """Count saturated red/green authored-mark pixels without branch truth."""
    pixels = rgb[mask].astype(np.int16)
    red = ((pixels[:, 0] >= 128)
           & (pixels[:, 0] - np.maximum(pixels[:, 1], pixels[:, 2]) >= 64)
           & (pixels[:, 0] >= 1.5 * np.maximum(pixels[:, 1], pixels[:, 2])))
    green = ((pixels[:, 1] >= 128)
             & (pixels[:, 1] - np.maximum(pixels[:, 0], pixels[:, 2]) >= 64)
             & (pixels[:, 1] >= 1.5 * np.maximum(pixels[:, 0], pixels[:, 2])))
    return {"method": "limited_authored_marker_color_test",
            "saturated_red_pixels": int(red.sum()), "saturated_green_pixels": int(green.sum()),
            "minimum_marker_pixels": MIN_MARKER_PIXELS,
            "minimum_dominant_channel": 128, "minimum_channel_margin": 64,
            "minimum_channel_ratio": 1.5,
            "marker_visible": bool(red.sum() >= MIN_MARKER_PIXELS or green.sum() >= MIN_MARKER_PIXELS),
            "general_semantic_recognition_certified": False}


class CandidateObservationEvidence:
    """Read synchronized native instance masks without mutating shared state."""

    def __init__(self, session, parameters):
        self.session = session
        self.parameters = parameters
        self.contract = parameters["contract"]

    def _identities(self, frame):
        names = {spec["name"] for spec in self.parameters.get("objects", [])}
        names.update(frame.get("objects", {}))
        names.update(f"zone_{name}" for name in frame.get("regions", {}))
        scene = getattr(getattr(self.session, "env", None), "scene", None)
        lookup = getattr(scene, "object_registry", None)
        identities = {}
        for name in sorted(names):
            path = frame.get("objects", {}).get(name, {}).get("native_id")
            if path is None and callable(lookup):
                obj = lookup("name", name)
                path = getattr(obj, "prim_path", None)
            identities[name] = SimpleNamespace(name=name, prim_path=str(path) if path else None)
        return identities

    def record(self, observation, frame):
        receipt = {"schema": "wia.candidate_observation_evidence.v1", "status": "unavailable",
                   "observed_tags": [], "visible_names": [], "per_view": {}, "errors": {},
                   "policy_history_delivery_certified": False}
        delivered = observation.rgb_images()
        registry = getattr(self.session, "robot_camera_registry", {})
        identities = self._identities(frame)
        visible, marker_visible = set(), False
        available_views = 0
        unavailable_view = False
        for view, delivered_rgb in delivered.items():
            if view not in registry:
                receipt["per_view"][view] = {"status": "ignored", "evidence": {},
                    "errors": ["View is not a registered mounted robot camera"]}
                continue
            result = {"status": "unavailable", "evidence": {}, "errors": []}
            receipt["per_view"][view] = result
            try:
                if delivered_rgb is None:
                    raise ValueError("Delivered mounted RGB is absent")
                sensor_name = registry[view]
                sensor = self.session.robot.sensors[sensor_name]
                robot_path = str(self.session.robot.prim_path).rstrip("/")
                if not str(sensor.prim_path).startswith(robot_path + "/"):
                    raise ValueError("Registered camera is not mounted on the current robot")
                native, info = sensor.get_obs()
                if "rgb" not in native:
                    raise ValueError("Native RGB is absent")
                rgb, actual = _array(delivered_rgb), _array(native["rgb"])
                if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3
                        or actual.dtype != np.uint8 or actual.ndim != 3 or actual.shape[-1] not in (3, 4)
                        or not np.array_equal(actual[..., :3], rgb)):
                    raise ValueError("Native RGB does not exactly match delivered RGB")
                modality = next((name for name in ("seg_instance", "seg_instance_id")
                                 if name in native and name in info), None)
                if modality is None:
                    raise ValueError("Native instance segmentation or label mapping is absent")
                segmentation = _array(native[modality])
                if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
                    segmentation = segmentation[..., 0]
                if segmentation.shape != rgb.shape[:2] or not np.issubdtype(segmentation.dtype, np.integer):
                    raise ValueError("Native instance segmentation dimensions/type do not match RGB")
                labels = _labels(info, modality)
                masks = {}
                for name, identity in identities.items():
                    ids = [identifier for identifier, label in labels.items() if _instance_matches(label, identity)]
                    mask = np.isin(segmentation, ids)
                    masks[name] = mask
                    evidence = _mask_evidence(mask)
                    result["evidence"][name] = evidence
                    if evidence["visible"]:
                        visible.add(name)
                if self.parameters["prototype"] == "H1":
                    selected = self.contract["selected"]
                    if selected in masks:
                        marker = _marker_evidence(rgb, masks[selected])
                        result["evidence"][selected]["authored_marker_color_test"] = marker
                        marker_visible |= bool(result["evidence"][selected]["visible"] and marker["marker_visible"])
                result.update(status="complete", native_rgb_matches_delivered=True,
                              segmentation_modality=modality, sensor_name=sensor_name)
                available_views += 1
            except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as error:
                result["errors"].append(f"{type(error).__name__}: {error}")
                unavailable_view = True
        tags = []
        prototype = self.parameters["prototype"]
        if prototype == "H1" and marker_visible:
            tags.append(f"inspection:{self.contract['selected']}")
        elif prototype == "H5":
            primary = self.contract.get("primary", "primary")
            resident = self.contract["resident"]
            overlap = _overlap(frame, resident, primary)
            if overlap is None:
                receipt["errors"]["destination_state"] = "Resident/primary measured AABB is unavailable"
            elif f"zone_{primary}" in visible and (not overlap or resident in visible):
                tags.append("destination_state")
        elif prototype == "M3":
            for unit, resident in self.contract["content_by_unit"].items():
                inside = frame.get("objects", {}).get(resident, {}).get("inside")
                if not isinstance(inside, (list, tuple)):
                    receipt["errors"][f"contents:{unit}"] = "Resident's actual native inside measurement is unavailable"
                elif resident in visible and unit in inside:
                    tags.append(f"contents:{unit}")
        receipt.update(observed_tags=sorted(tags), visible_names=sorted(visible))
        if available_views and not unavailable_view and not receipt["errors"]:
            receipt["status"] = "complete"
        if not available_views:
            receipt["errors"]["mounted_views"] = "No synchronized delivered mounted RGB/segmentation view is available"
        return receipt

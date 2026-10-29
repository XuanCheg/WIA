"""Pure geometry for human review cameras, without simulation or visibility claims.

Run ``python -m pytest -q tests/test_review_camera_geometry.py`` for CPU checks.
"""
from __future__ import annotations

from itertools import product

import numpy as np


_PROFILES = {
    "mapping": ("Whole-room context and navigation anchors.", [35., 65.], 90.),
    "interior": ("Container or enclosure interiors and their support surfaces.", [90., 45.], 60.),
    "temporal": ("Current object, container and support arrangement; temporal history is not inferred.", [90., 40.], 60.),
    "relations": ("Object groups and references for spatial, size or distance comparisons.", [90., 20.], 60.),
    "dynamics": ("Side views of contact and support, with oblique context.", [10., 40.], 55.),
    "reflection": ("Oblique optical reference surfaces and real objects.", [20., 45.], 50.),
    "perception": ("Multiple oblique views of targets, occluders and confusable objects.", [25., 50.], 60.),
    "deformable": ("Top and side views of shape, contents and supporting surfaces.", [90., 10.], 60.),
}

# Exact standard labels from the ESI category inventory. These preferences only
# guide human cameras; they do not define task semantics or evaluation evidence.
_CATEGORY_PROFILES = {
    ("Action Sequencing", "Action Order Inference"): "dynamics",
    ("Cognitive Mapping", "Connectivity"): "mapping",
    ("Cognitive Mapping", "Long-Term Navigation"): "mapping",
    ("Cognitive Mapping", "Regional Boundary"): "mapping",
    ("Cognitive Mapping", "Traversable Passage"): "mapping",
    ("Enumerative Perception", "Category Ambiguity"): "perception",
    ("Enumerative Perception", "Counting w Occlusion"): "perception",
    ("Enumerative Perception", "Illumination Variability"): "perception",
    ("Enumerative Perception", "Merged Observation"): "perception",
    ("Enumerative Perception", "Spatial Segmentation"): "perception",
    ("Enumerative Perception", "Structural Enclosure"): "interior",
    ("Metric Comparison", "Dimensional Size"): "relations",
    ("Metric Comparison", "Spatial Distance"): "relations",
    ("Perceptual Grounding", "Material Transparency"): "perception",
    ("Perceptual Grounding", "Partial Occlusion"): "perception",
    ("Perceptual Grounding", "View Hallucination"): "perception",
    ("Physical Dynamics", "Inclined Plane"): "dynamics",
    ("Physical Dynamics", "Stacking & Stability"): "dynamics",
    ("Physical Structure", "Deformable"): "deformable",
    ("Physical Structure", "Liquid Volume"): "deformable",
    ("Physical Structure", "Rigid Containment"): "interior",
    ("Spatial Relations", "Geometric Configuration"): "relations",
    ("Spatial Relations", "Linear Alignment"): "relations",
    ("Spatial Relations", "Physical Contact"): "dynamics",
    ("Specular Reflection", "Correspondence"): "reflection",
    ("Specular Reflection", "Reflection Authoring"): "reflection",
    ("Specular Reflection", "Spatial Relations"): "reflection",
    ("Temporal Understanding", "Agent Observation"): "temporal",
    ("Temporal Understanding", "Unobserved Change"): "temporal",
}


def review_profile(big_task, small_task):
    """Return category framing preferences, explicitly marking unknown labels."""
    profile = _CATEGORY_PROFILES.get((big_task, small_task))
    purpose, elevations, overview = _PROFILES.get(profile, (
        "Generic object context; no category-specific framing profile is available.",
        [30., 60.], 60.,
    ))
    return {"purpose": purpose, "detail_elevations": list(elevations),
            "overview_elevation": overview, "known_profile": profile is not None}


def review_directions(elevation_deg, azimuth_deg=0.0):
    """Return eight azimuth candidates, or one unique pole at elevation +/-90.

    Angles are degrees; each unit vector points from the target to the camera.
    """
    if not np.isfinite([elevation_deg, azimuth_deg]).all() or not -90 <= elevation_deg <= 90:
        raise ValueError("Camera angles must be finite, with elevation in [-90, 90].")
    if abs(elevation_deg) == 90:
        return np.array([[0., 0., np.sign(elevation_deg)]])
    elevation = np.deg2rad(elevation_deg)
    azimuths = np.deg2rad(azimuth_deg + np.arange(8) * 45.)
    return np.column_stack((np.cos(elevation) * np.cos(azimuths),
                            np.cos(elevation) * np.sin(azimuths),
                            np.full(8, np.sin(elevation))))


def fit_review_view(bounds, direction, *, horizontal_fov_deg=75.0, aspect=1.0,
                    margin=1.15, minimum_distance=.20):
    """Fit every supplied AABB corner inside a perspective camera frustum.

    ``bounds`` has shape (N, 2, 3), with lower and upper world coordinates.
    ``aspect`` is image width / height. ``margin`` >= 1 reserves image padding.
    The basis matches ``esi_debug_cameras.camera_orientation`` (USD -Z optical,
    +Y up). Frustum inclusion does not establish rendered visibility, occlusion,
    collision freedom, or physical task success.
    """
    bounds = np.asarray(bounds, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if (bounds.ndim != 3 or bounds.shape[1:] != (2, 3) or not len(bounds)
            or not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1])):
        raise ValueError("Bounds must be nonempty finite ordered AABBs of shape (N, 2, 3).")
    if (direction.shape != (3,) or not np.isfinite(direction).all()
            or np.linalg.norm(direction) < 1e-8):
        raise ValueError("Camera direction must be a finite nonzero 3-vector.")
    if (not np.isfinite([horizontal_fov_deg, aspect, margin, minimum_distance]).all()
            or not 0 < horizontal_fov_deg < 180 or aspect <= 0
            or margin < 1 or minimum_distance <= 0):
        raise ValueError("Require 0 < FOV < 180, positive aspect/distance and margin >= 1.")
    backward = direction / np.linalg.norm(direction)
    up = np.array([0., 0., 1.])
    if abs(up @ backward) > .99:
        up = np.array([0., 1., 0.])
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    target = (low + high) / 2
    selectors = np.array(list(product((0, 1), repeat=3)))
    corners = np.stack([bounds[:, selectors[:, axis], axis] for axis in range(3)], axis=-1).reshape(-1, 3)
    relative = corners - target
    x, y, z = (relative @ np.column_stack((right, up, backward))).T
    # At distance d along backward, optical depth is d-z. Solve each
    # perspective inequality independently, including the near clipping plane.
    horizontal_tan = np.tan(np.deg2rad(horizontal_fov_deg) / 2)
    vertical_tan = horizontal_tan / aspect
    distance = max(minimum_distance, float(np.max(z + .02)),
                   float(np.max(z + margin * np.abs(x) / horizontal_tan)),
                   float(np.max(z + margin * np.abs(y) / vertical_tan)))
    position = target + distance * backward
    depth = distance - z
    return {"position_world_m": position.tolist(), "target_world_m": target.tolist(),
            "clipping_range_m": [.01, float(max(.1, depth.max() * 1.05 + .01))],
            "bounds_world_m": [low.tolist(), high.tolist()]}


def framed_near_plane(fitted, margin_m=.08):
    """Clip foreground architecture strictly before every framed AABB corner."""
    low, high = np.asarray(fitted['bounds_world_m'], dtype=float)
    position = np.asarray(fitted['position_world_m'], dtype=float)
    forward = np.asarray(fitted['target_world_m'], dtype=float) - position
    forward /= np.linalg.norm(forward)
    corners = np.asarray(list(product(*zip(low, high))))
    return max(.01, float(np.min((corners - position) @ forward)) - margin_m)

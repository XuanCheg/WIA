"""Check human review framing using the actual camera quaternion and projection.

Run ``python -m pytest -q tests/test_review_camera_geometry.py``.
"""
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wia.runtime.sim.esi_debug_cameras import camera_orientation
from wia.runtime.sim.esi_review_camera_geometry import (
    fit_review_view, review_directions, review_profile,
)


@pytest.mark.parametrize("bounds", [
    [[[2., 3., 1.], [2.001, 3.004, 1.002]]],
    [[[-90., -120., -4.], [110., 60., 40.]]],
    [[[2., -1., 0.], [2.1, -.9, 35.]]],
    [[[-30., 2., 0.], [-29., 3., 1.]], [[20., -17., 7.], [23., -15., 9.]],
     [[3., 8., 1.], [5., 10., 4.]]],
])
@pytest.mark.parametrize("direction", [[1., 2., .1], [-1., .7, 1.2], [0., 0., 1.], [.001, 0., 1.]])
@pytest.mark.parametrize("fov,aspect", [(75., 1.), (35., 3.), (110., .4)])
def test_all_box_corners_fit_actual_camera_projection(bounds, direction, fov, aspect):
    margin = 1.15
    fitted = fit_review_view(bounds, direction, horizontal_fov_deg=fov, aspect=aspect, margin=margin)
    position = np.asarray(fitted["position_world_m"])
    target = np.asarray(fitted["target_world_m"])
    rotation = Rotation.from_quat(camera_orientation(position, target))
    corners = np.array([corner for lower, upper in bounds
                        for corner in product(*zip(lower, upper))])
    local = rotation.inv().apply(corners - position)
    depth = -local[:, 2]
    near, far = fitted["clipping_range_m"]
    assert np.all(depth > near) and np.all(depth < far)
    assert np.all(np.abs(local[:, 0]) / depth <= np.tan(np.deg2rad(fov / 2)) / margin + 1e-10)
    assert np.all(np.abs(local[:, 1]) / depth <= np.tan(np.deg2rad(fov / 2)) / aspect / margin + 1e-10)
    assert np.linalg.norm(position - target) >= .2 - 1e-10


def test_all_standard_categories_have_profiles_and_unknown_is_explicit():
    categories = {
        "Action Sequencing": ["Action Order Inference"],
        "Cognitive Mapping": ["Connectivity", "Long-Term Navigation", "Regional Boundary", "Traversable Passage"],
        "Enumerative Perception": ["Category Ambiguity", "Counting w Occlusion", "Illumination Variability",
                                   "Merged Observation", "Spatial Segmentation", "Structural Enclosure"],
        "Metric Comparison": ["Dimensional Size", "Spatial Distance"],
        "Perceptual Grounding": ["Material Transparency", "Partial Occlusion", "View Hallucination"],
        "Physical Dynamics": ["Inclined Plane", "Stacking & Stability"],
        "Physical Structure": ["Deformable", "Liquid Volume", "Rigid Containment"],
        "Spatial Relations": ["Geometric Configuration", "Linear Alignment", "Physical Contact"],
        "Specular Reflection": ["Correspondence", "Reflection Authoring", "Spatial Relations"],
        "Temporal Understanding": ["Agent Observation", "Unobserved Change"],
    }
    assert sum(map(len, categories.values())) == 29
    for big_task, small_tasks in categories.items():
        for small_task in small_tasks:
            profile = review_profile(big_task, small_task)
            assert profile["known_profile"] and profile["purpose"]
            assert len(profile["detail_elevations"]) >= 2
            for elevation in profile["detail_elevations"] + [profile["overview_elevation"]]:
                assert np.isfinite(review_directions(elevation)).all()
    assert not review_profile("Unknown", "Unknown")["known_profile"]
    assert "no category-specific" in review_profile("Unknown", "Unknown")["purpose"]


def test_direction_candidates_cover_opposing_views_and_unique_overhead():
    directions = review_directions(30., 17.)
    assert directions.shape == (8, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.)
    assert np.allclose(directions[:4, :2], -directions[4:, :2])
    assert np.allclose(directions[:, 2], .5)
    assert np.array_equal(review_directions(90.), [[0., 0., 1.]])


@pytest.mark.parametrize("bounds,direction", [
    ([], [1., 0., 1.]),
    ([[[0., 0., 0.], [np.nan, 1., 1.]]], [1., 0., 1.]),
    ([[[2., 0., 0.], [1., 1., 1.]]], [1., 0., 1.]),
    ([[[0., 0., 0.], [1., 1., 1.]]], [0., 0., 0.]),
])
def test_invalid_geometry_cannot_produce_camera_metadata(bounds, direction):
    with pytest.raises(ValueError):
        fit_review_view(bounds, direction)


def test_oblique_cutaway_never_clips_framed_actor_bounds():
    from itertools import product
    from wia.runtime.sim.esi_review_camera_geometry import framed_near_plane
    bounds = np.array([[[2., -13., .1], [5.5, -11., 1.7]]])
    fitted = fit_review_view(bounds, [-1., -1., 2.], horizontal_fov_deg=82.)
    near = framed_near_plane(fitted)
    position = np.asarray(fitted['position_world_m'])
    forward = np.asarray(fitted['target_world_m']) - position
    forward /= np.linalg.norm(forward)
    corners = np.asarray(list(product(*zip(bounds[0, 0], bounds[0, 1]))))
    assert near > .01
    assert np.all((corners - position) @ forward > near)

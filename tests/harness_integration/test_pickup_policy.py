"""Check review protection, uniform sizing and separate support geometry.

Run: .venv/bin/python -m pytest -q tests/test_pickup_policy.py
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wia.runtime.scene_review import fingerprint
from wia.tasks.pickup_policy import matching_review, separate_pickup_support, size_target
from wia.tasks.tabletop import generate_tabletop, source_geometry
from wia.tasks.loader import DATA_ROOT
from wia.runtime.commands.sample_esi_supports import selections
from harness_integration.test_tabletop_scenes import SOURCE


def reviewed_source(tmp_path, decision):
    source = tmp_path / SOURCE.relative_to(DATA_ROOT)
    source.parent.mkdir(parents=True)
    source.write_bytes(SOURCE.read_bytes())
    definition, _ = generate_tabletop(source, tmp_path)
    source.with_suffix(".yaml").write_text(yaml.safe_dump(definition.model_dump(mode="json", by_alias=True)))
    review = {"decision": decision, "source": str(source.relative_to(tmp_path)), "variant": "tabletop",
        "seed": 0, "fingerprint": fingerprint(source.with_suffix(".yaml"), source, "tabletop", 0), "notes": ""}
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"schema": "scene_reviews_v1", "entries": {"review": review}}))
    return source, path


@pytest.mark.parametrize("decision,reason", [("approved", "approved_layout_preserved"), ("skipped", "human_review_excluded")])
def test_review_decisions_preserve_exact_scene_bytes_and_exclude_processing(tmp_path, decision, reason):
    source, review = reviewed_source(tmp_path, decision)
    before = source.with_suffix(".yaml").read_bytes()
    args = SimpleNamespace(source=source, data_root=tmp_path, review_file=review, seed=0, limit=None)
    candidates, excluded = selections(args)
    assert candidates == [] and excluded[0]["reason"] == reason
    assert source.with_suffix(".yaml").read_bytes() == before
    assert source.read_bytes() == SOURCE.read_bytes()


def test_stale_approval_cannot_protect_a_changed_layout(tmp_path):
    source, review = reviewed_source(tmp_path, "approved")
    entries = json.loads(review.read_text())["entries"]
    assert matching_review(source, tmp_path, entries)["decision"] == "approved"
    source.with_suffix(".yaml").write_text(source.with_suffix(".yaml").read_text() + "\n# Changed layout receipt.\n")
    assert matching_review(source, tmp_path, entries) is None


def test_uniform_shrink_preserves_support_gap_containers_and_existing_nonuniform_asset_scale():
    definition, _ = generate_tabletop(SOURCE, DATA_ROOT)
    before = SOURCE.read_bytes()
    scaled, details = size_target(definition, SOURCE, DATA_ROOT, grasp_width_m=.08)
    _, _, geometry = source_geometry(SOURCE, DATA_ROOT)
    assert min(details["scaled_target_extent_m"]) == pytest.approx(.08)
    assert 0 < details["target_scale_multiplier"] < 1
    old = definition.model_dump(mode="json", by_alias=True)["scene"]["variants"]["tabletop"]["patches"]
    new = scaled.model_dump(mode="json", by_alias=True)["scene"]["variants"]["tabletop"]["patches"]
    other = lambda p: p["operation"] not in ("object_pose", "object_scale") or p.get("binding") != "target"
    assert [p for p in old if other(p)] == [p for p in new if other(p)]
    old_pose = next(p for p in old if p.get("binding") == "target" and p["operation"] == "object_pose")
    new_pose = next(p for p in new if p.get("binding") == "target" and p["operation"] == "object_pose")
    offset = geometry["target"]["origin_above_bottom_m"]
    assert old_pose["position"][2] - offset == pytest.approx(new_pose["position"][2] - offset * details["target_scale_multiplier"])
    assert details["source_question_unchanged"] and not details["source_answer_geometry_preserved"]
    assert not details["source_answer_revalidated"] and SOURCE.read_bytes() == before
    assert np.allclose(np.array([1., 2., 3.]) * details["target_scale_multiplier"],
        np.array([1., 2., 3.]) * min(.08 / np.min(geometry["target"]["source_extent_m"]), 1.))


def test_shrinking_is_optional_and_never_enlarges_a_small_target():
    original, _ = generate_tabletop(SOURCE, DATA_ROOT, target_scale=.2)
    scaled, details = size_target(original, SOURCE, DATA_ROOT)
    assert scaled == original and details["target_scale_multiplier"] == .2
    original, _ = generate_tabletop(SOURCE, DATA_ROOT)
    scaled, details = size_target(original, SOURCE, DATA_ROOT, keep_scale=True)
    assert scaled == original and details["target_scale_multiplier"] == 1.


def test_added_support_preserves_container_work_area_and_aligns_four_legs_to_floor():
    definition, _ = generate_tabletop(SOURCE, DATA_ROOT)
    draft, details = separate_pickup_support(definition, position_xy=[3., 2.], floor_z=.12,
        target_extent=[.08, .08, .08], target_bottom_offset=.04, target_quaternion=[0., 0., 0., 1.])
    patches = draft.model_dump(mode="json", by_alias=True)["scene"]["variants"]["tabletop"]["patches"]
    old = definition.model_dump(mode="json", by_alias=True)["scene"]["variants"]["tabletop"]["patches"]
    retained = [p for p in patches if not (p["operation"] == "add_primitive" and p["name"].startswith("pickup_surface_"))
                and not (p["operation"] == "object_pose" and p["binding"] == "target")]
    assert retained == [p for p in old if not (p["operation"] == "object_pose" and p["binding"] == "target")]
    legs = [p for p in patches if p["operation"] == "add_primitive" and "pickup_surface_000_leg" in p["name"]]
    assert len(legs) == 4
    assert all(p["position"][2] - p["dimensions_m"][2] / 2 == pytest.approx(.12) for p in legs)
    assert draft.qualification.object_supports["target"] == "pickup_support"
    assert draft.bindings["pickup_support"].name != draft.bindings[draft.qualification.support].name
    assert draft.status == "pending" and details["requires_native_restart"]
    assert not details["route_connectivity_screen_passed"] and not details["physical_grasp_verified"]

"""Apply human review, target sizing and separate pickup-support construction.

Run through src/wia/runtime/commands/convert_esi_rigid_catalog.py --dry-run --review-file FILE.
These CPU recipes do not certify navigation, support contact or physical grasp.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from wia.authoring.io import fingerprint
from .schema import SceneDefinition
from .tabletop import source_geometry


def matching_review(source, data_root, entries, *, variant="tabletop"):
    """Apply decisions only to the exact reviewed source, layout, variant and seed."""
    source = Path(source).resolve()
    relative = str(source.relative_to(Path(data_root).resolve()))
    descriptor = source.with_suffix(".yaml")
    if not descriptor.is_file():
        return None
    import yaml

    value = yaml.safe_load(descriptor.read_text())
    seed = value.get("seed", 0)
    current = fingerprint(descriptor, source, variant, seed)
    matches = [r for r in entries.values() if r.get("source") == relative
               and r.get("variant") == variant and r.get("seed") == seed
               and r.get("fingerprint") == current]
    if len(matches) > 1:
        raise ValueError("Multiple matching human reviews refer to one scene.")
    return matches[0] if matches else None


def review_entries(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "scene_reviews_v1":
        raise ValueError("Expected a scene_reviews_v1 human review file.")
    return value["entries"]


def size_target(definition, source, data_root, *, grasp_width_m=.08, keep_scale=False):
    """Shrink uniformly using the minimum source extent; preserve the support gap."""
    if not 0 < grasp_width_m <= .10:
        raise ValueError("Grasp width must be positive and within the measured 0.10 m aperture.")
    _, _, geometry = source_geometry(source, data_root)
    original_extent = np.asarray(geometry["target"]["source_extent_m"])
    value = definition.model_dump(mode="json", by_alias=True)
    target = value["target"]
    patches = value["scene"]["variants"]["tabletop"]["patches"]
    scales = [p for p in patches if p["operation"] == "object_scale" and p["binding"] == target]
    old = float(np.prod([p["multiplier"] for p in scales]))
    multiplier = old if keep_scale else min(old, grasp_width_m / float(original_extent.min()))
    if multiplier < old:
        patches[:] = [p for p in patches if not (p["operation"] == "object_scale" and p["binding"] == target)]
        patches.insert(0, {"operation": "object_scale", "binding": target, "multiplier": multiplier})
        offset = geometry["target"]["origin_above_bottom_m"]
        for patch in patches:
            if patch["operation"] == "object_pose" and patch["binding"] == target:
                patch["position"][2] += offset * (multiplier - old)
    details = {"target_scale_multiplier": multiplier, "previous_target_scale_multiplier": old,
        "target_scale_changed": multiplier != old, "original_target_extent_m": original_extent.tolist(),
        "scaled_target_extent_m": (original_extent * multiplier).tolist(), "grasp_width_m": grasp_width_m,
        "size_screen": "Minimum global source extent; physical grasp remains unverified.",
        "source_question_unchanged": True, "source_answer_revalidated": False,
        "source_answer_geometry_preserved": multiplier == 1.}
    return SceneDefinition.model_validate(value), details


def separate_pickup_support(definition, *, position_xy, floor_z, target_extent,
                            target_bottom_offset, target_quaternion, index=0, height=.55,
                            edge_margin=.04):
    """Build a pending target-only support; keep robot, containers and main table."""
    extent = np.asarray(target_extent, dtype=float)
    width, depth = np.maximum(extent[:2] + 2 * edge_margin, .30)
    if max(width, depth) > .90:
        raise ValueError("Target footprint is too large for a bounded separate pickup support.")
    value = copy.deepcopy(definition.model_dump(mode="json", by_alias=True))
    patches = value["scene"]["variants"]["tabletop"]["patches"]
    # Remove prior generated pickup furniture when regenerating its placement.
    patches[:] = [p for p in patches if not (p["operation"] == "add_primitive"
        and p["name"].startswith("pickup_surface_"))]
    target = value["target"]
    patches[:] = [p for p in patches if not (p["operation"] == "object_pose" and p["binding"] == target)]
    x, y = map(float, position_xy)
    top_z = float(floor_z + height)
    prefix = f"pickup_surface_{index:03d}"
    top_name = prefix + "_top"

    def primitive(name, centre, dimensions):
        patches.append({"operation": "add_primitive", "name": name,
            "position": list(centre), "dimensions_m": list(map(float, dimensions)),
            "quaternion_xyzw": [0., 0., 0., 1.], "rgba": [.42, .32, .20, 1.]})

    primitive(top_name, (x, y, top_z - .02), (width, depth, .04))
    for i, (dx, dy) in enumerate((dx, dy) for dx in (-width / 2 + .025, width / 2 - .025)
                                 for dy in (-depth / 2 + .025, depth / 2 - .025)):
        primitive(prefix + f"_leg_{i}", (x + dx, y + dy, floor_z + (height - .04) / 2),
                  (.035, .035, height - .04))
    target_position = [x, y, top_z + float(target_bottom_offset) + .003]
    patches.append({"operation": "object_pose", "phase": "after_postprocessing", "binding": target,
        "position": target_position, "quaternion_xyzw": list(map(float, target_quaternion))})
    value["bindings"]["pickup_support"] = {"name": top_name, "category": "object"}
    value["qualification"]["object_supports"][target] = "pickup_support"
    value["qualification"]["support_surface_mode"] = "local_raycast"
    value["status"] = "pending"
    value["id"] += f"_separate_pickup_{index:03d}"
    return SceneDefinition.model_validate(value), {"support": top_name, "position": target_position,
        "quaternion_xyzw": list(map(float, target_quaternion)), "support_surface_z_m": top_z,
        "support_dimensions_m": [float(width), float(depth), height], "support_floor_z_m": floor_z,
        "support_kind": "added_separate_pickup_surface", "requires_native_restart": True,
        "route_connectivity_screen_passed": False, "physical_grasp_verified": False}

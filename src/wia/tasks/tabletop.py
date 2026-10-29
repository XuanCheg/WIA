"""Generate explicit tabletop scene patches from Rigid Containment metadata.

Run with src/wia/runtime/commands/generate_esi_tabletop.py; generated layouts still need native
support, clearance, and manipulation qualification before accepting a scene.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .schema import SceneDefinition


def source_geometry(path, data_root, *, target_scale=1.):
    source = Path(path).resolve()
    relative = source.relative_to(Path(data_root).resolve())
    question = json.loads(source.read_text())
    if question.get("small_task") != "Rigid Containment":
        raise ValueError("The current tabletop source adapter supports Rigid Containment.")
    metadata = question["metadata_json"]
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    poses = metadata["initial_poses"]
    poses = {pose["_key"]: pose for pose in poses} if isinstance(poses, list) else poses
    identities = {"target": {"name": "obj_containee", "category": metadata["containee"]["cat"],
                             "model": metadata["containee"]["model"]}}
    containers = metadata["containers"]
    containers = list(containers.values()) if isinstance(containers, dict) else containers
    for container in containers:
        key = "container_" + str(container["idx"])
        identities[key] = {"name": "obj_" + key, "category": container["cat"], "model": container["model"]}
    rows = {}
    for key, binding in identities.items():
        pose = poses[binding["name"]]
        low, high, position = (np.asarray(pose[name], float) for name in ("aabb_min", "aabb_max", "position"))
        if (not np.isfinite(np.r_[low, high, position]).all() or np.any(high <= low)
                or not np.all(low - .2 <= position) or not np.all(position <= high + .2)):
            raise ValueError("Source poses/bounds are inconsistent: " + binding["name"])
        multiplier = target_scale if key == "target" else 1.
        rows[key] = {"binding": binding, "pose": pose, "source_extent_m": (high - low).tolist(),
                     "extent_m": ((high - low) * multiplier).tolist(), "multiplier": multiplier,
                     "origin_above_bottom_m": float((position[2] - low[2]) * multiplier)}
    bottoms = [poses[row["binding"]["name"]]["aabb_min"][2] for row in rows.values()]
    if max(bottoms) - min(bottoms) > 1.:
        raise ValueError("Source object/support heights differ by more than 1 m; native correction is required.")
    return relative, question, rows


def initialization_source(path, data_root, question):
    """Reuse canonical construction only when a task variant has identical physics inputs."""
    source = Path(path).resolve()
    root = Path(data_root).resolve()
    if "__" not in source.stem:
        return source.relative_to(root)
    canonical = source.with_name(source.stem.split("__", 1)[0] + ".json")
    if not canonical.is_file():
        raise ValueError("Task variant has no canonical construction source; independent scene adaptation is required.")
    original = json.loads(canonical.read_text())

    def inputs(record):
        metadata = record["metadata_json"]
        metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
        target = metadata["containee"]
        return {"scene": record["scene"], "room": record["room"],
                "target": {key: target.get(key) for key in ("cat", "model", "scale", "bbox_inventory")},
                **{key: metadata.get(key) for key in ("initial_poses", "containers", "extras", "floor_name")}}

    if inputs(question) != inputs(original):
        raise ValueError("Task variant changes construction inputs; independent scene adaptation is required.")
    return canonical.relative_to(root)


def generate_tabletop(path, data_root, *, table_height=.45, floor_z=0., target_scale=1.,
                      robot_position=None, robot_yaw_deg=0., head_pitch_deg=0.,
                      front_clearance=.45, edge_margin=.04, object_gap=.06, seed=0,
                      trunk_config="configs/esi_trunk_low_grasp.json"):
    if table_height <= .04 or target_scale <= 0 or front_clearance <= 0 or min(edge_margin, object_gap) < .01:
        raise ValueError("Table height, object scale and clearances must be positive and meaningful.")
    relative, question, rows = source_geometry(path, data_root, target_scale=target_scale)
    construction_source = initialization_source(path, data_root, question)
    task_variant = construction_source != relative
    containers = sorted((key for key in rows if key != "target"),
                        key=lambda key: (np.prod(rows[key]["extent_m"][:2]), key))
    if not containers:
        raise ValueError("The source must contain at least one container.")
    packed_rows = [("target", containers[0])] + [tuple(containers[i:i + 2])
                                                for i in range(1, len(containers), 2)]
    left_width = max(rows[row[0]]["extent_m"][1] for row in packed_rows)
    right_width = max(rows[row[1]]["extent_m"][1] for row in packed_rows if len(row) == 2)
    row_depths = [max(rows[key]["extent_m"][0] for key in row) for row in packed_rows]
    width = math.ceil((left_width + right_width + object_gap + 2 * edge_margin) * 100) / 100
    depth = math.ceil((sum(row_depths) + (len(packed_rows) - 1) * object_gap + 2 * edge_margin) * 100) / 100
    if width > 1.8 or depth > 1.5:
        raise ValueError("This bounded tabletop recipe cannot accommodate the source object footprints.")
    rotation = Rotation.from_euler("z", robot_yaw_deg, degrees=True)
    centre_forward = front_clearance + depth / 2
    if robot_position is None:
        centre = np.mean([np.asarray(row["pose"]["position"][:2]) for row in rows.values()], axis=0)
        base = np.r_[centre, .1] - rotation.apply([centre_forward, 0., 0.])
    else:
        base = np.asarray(robot_position, float)
    quaternion = rotation.as_quat().tolist()
    bindings = {key: row["binding"] for key, row in rows.items()}
    bindings["table"] = {"name": "support_table_top", "category": "object"}
    patches = [{"operation": "robot_pose", "position": base.tolist(), "yaw_deg": float(robot_yaw_deg),
                "head_pitch_deg": float(head_pitch_deg)}]
    if target_scale != 1.:
        patches.append({"operation": "object_scale", "binding": "target", "multiplier": float(target_scale)})
    top_z, thickness, leg_width = floor_z + table_height, .04, .035
    def primitive(name, local_xy, centre_z, dimensions, rgba):
        position = base + rotation.apply([*local_xy, 0.])
        position[2] = centre_z
        patches.append({"operation": "add_primitive", "name": name,
                        "position": position.tolist(), "dimensions_m": list(dimensions),
                        "quaternion_xyzw": quaternion, "rgba": rgba})
    primitive("support_table_top", (centre_forward, 0.), top_z - thickness / 2,
              (depth, width, thickness), [.48, .29, .14, 1.])
    leg_height = table_height - thickness
    for i, (x, y) in enumerate((x, y) for x in (front_clearance + .04, front_clearance + depth - .04)
                               for y in (-width / 2 + .04, width / 2 - .04)):
        primitive("support_table_leg_" + str(i + 1), (x, y), floor_z + leg_height / 2,
                  (leg_width, leg_width, leg_height), [.30, .20, .12, 1.])
    y_positions = ((right_width + object_gap) / 2, -(left_width + object_gap) / 2)
    x_positions, cursor = [], front_clearance + edge_margin
    for row_depth in row_depths:
        x_positions.append(cursor + row_depth / 2)
        cursor += row_depth + object_gap
    layout = {}
    for row_index, keys in enumerate(packed_rows):
        for column, key in enumerate(keys):
            row = rows[key]
            # Put the manipulation target at the nearest usable table edge;
            # centering a small target in a large container's row wastes reach.
            x = (front_clearance + edge_margin + row["extent_m"][0] / 2
                 if key == "target" else x_positions[row_index])
            position = base + rotation.apply([x, y_positions[column], 0.])
            position[2] = top_z + row["origin_above_bottom_m"] + .003
            orientation = rotation * Rotation.from_quat(row["pose"]["quaternion_xyzw"])
            patches.append({"operation": "object_pose", "phase": "after_postprocessing", "binding": key,
                            "position": position.tolist(), "quaternion_xyzw": orientation.as_quat().tolist()})
            layout[key] = {"table_frame_xy_m": [x - centre_forward, y_positions[column]],
                           "position_m": position.tolist(), "extent_m": row["extent_m"]}
    value = {"kind": "scene_patch", "id": "tabletop_" + Path(relative).stem, "version": 1,
        "status": "pending" if task_variant else "ready",
        "source": {"provider": "esi", "json": str(construction_source)}, "target": "target", "bindings": bindings,
        "scene": {"policy": "native_then_minimal_patch_else_skip",
                  "variants": {"original": {"patches": []}, "tabletop": {"patches": patches}}},
        "qualification": {"support": "table", "supported_objects": list(rows)},
        "runtime": {"trunk_config": trunk_config}, "seed": seed}
    definition = SceneDefinition.model_validate(value)
    details = {"source_question": question["question"], "source_answer_revalidated": False,
               "source_question_json": str(relative), "initialization_source_json": str(construction_source),
               "task_variant": task_variant, "canonical_construction_verified": task_variant,
               "requires_separate_task_definition": task_variant,
               "task_execution_status": "pending_separate_task_support" if task_variant else "original_single_object_supported",
               "target_scale_multiplier": target_scale, "table_top_world_z_m": top_z,
               "table_dimensions_m": [depth, width, table_height], "layout": layout,
               "source_geometry": rows, "robot_position_m": base.tolist(),
               "native_qualification": "pending", "mass_policy": "Retain native asset constructor behavior; no mass override.",
               "aperture_screen": {"minimum_target_global_extent_m": min(rows["target"]["extent_m"]),
                   "exceeds_100_mm": min(rows["target"]["extent_m"]) > .1,
                   "interpretation": "Flag only; local grasp feasibility still needs physical qualification."}}
    return definition, details

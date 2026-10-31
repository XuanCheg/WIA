"""Bounded CPU geometric screening, without simulator imports or robot commands.

Run ``python -m wia.candidates.reachability INVENTORY_JSON OUTPUT_DIRECTORY``
in the behavior environment. USD fallback calibration additionally needs the
installed omni.usd.libs extension's ``bin`` and ``bin/deps`` in LD_LIBRARY_PATH.
The input quaternion convention is xyzw; distances are metres and joints radians
(prismatic joints use metres). A witness is an open-gripper geometric candidate,
never a claim that closing, carrying, controller tracking, or a grasp will work.
The default continuous mode optimizes torso and working arm together. Preserve
arm-only screening with ``audit_scene(..., options={"torso_search": "fixed_presets"})``.
Destination queries are separate initial-state hypothetical approaches; they
never connect an initial pickup to transport, door opening, or final placement.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np

_DESTINATION_SCOPE = ("Initial-state-only hypothetical destination approach. No pickup, carried transport, "
                      "door opening, placement, stable release or task success is validated.")
_SUPPORT_BOTTOM_TOLERANCE_M = .015
_DESTINATION_INSET_M = .006
_BASE_GEOMETRY_AABB_TOLERANCE_M = .002


class AuditUnavailable(RuntimeError):
    """Required measured geometry or a CPU dependency is unavailable."""


@dataclass(frozen=True)
class AuditOptions:
    max_stances: int = 12
    max_ik_evaluations: int = 100
    max_ik_attempts_per_target: int = 192
    torso_search: str = "continuous"
    navigation_mode: str = "circle"
    grid_resolution_m: float = .06
    max_grid_cells: int = 250000
    max_joint_step: float = .06
    max_prismatic_step_m: float = .005
    collision_margin_m: float = .006
    position_tolerance_m: float = .005
    orientation_tolerance_deg: float = 3.
    aperture_m: float = .1
    approach_distance_m: float = .1
    max_stance_target_distance_m: float = 1.2


def _box(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (2, 3) or not np.isfinite(result).all()
            or np.any(result[0] > result[1])):
        raise AuditUnavailable("A finite, ordered world AABB is required for every actor.")
    return result


def _corners(box):
    return np.asarray(list(product(*zip(box[0], box[1]))))


def _validated_base_vertices(geometry, measured_base_aabb):
    evidence = {"footprint_source": "measured_base_aabb_corners", "base_geometry_validated": False,
        "base_geometry_fallback_reason": "optional_base_geometry_not_supplied",
        "base_geometry_world_aabb_residual_m": None, "base_geometry_reported_world_aabb_residual_m": None,
        "base_geometry_aabb_tolerance_m": _BASE_GEOMETRY_AABB_TOLERANCE_M,
        "base_geometry_enabled_mesh_names": []}
    if geometry is None:
        return None, evidence
    try:
        if not isinstance(geometry, dict) or geometry.get("schema") != "wia_robot_base_geometry_v1":
            raise ValueError("Unsupported or missing native base-geometry schema.")
        if geometry.get("available") is not True or geometry.get("status") != "available":
            raise ValueError(f"Native base geometry is unavailable: {geometry.get('reason', geometry.get('status'))}.")
        reported = float(geometry["max_world_aabb_residual_m"])
        if np.isfinite(reported):
            evidence["base_geometry_reported_world_aabb_residual_m"] = reported
        if not np.isfinite(reported) or not 0 <= reported <= _BASE_GEOMETRY_AABB_TOLERANCE_M:
            raise ValueError("Reported native base AABB residual is not finite and within 2 mm.")
        vertices = np.asarray(geometry["world_vertices"], float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("Native base vertices must be a finite Nx3 array with at least four rows.")
        extrema = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        residual = float(np.max(np.abs(extrema - _box(measured_base_aabb))))
        evidence["base_geometry_world_aabb_residual_m"] = residual
        if residual > _BASE_GEOMETRY_AABB_TOLERANCE_M:
            raise ValueError("Actual native vertex extrema disagree with robot.base_aabb by more than 2 mm.")
        names = geometry.get("enabled_mesh_names", geometry.get("reference_mesh_metadata", {}).get("enabled_mesh_names", []))
        transferred = geometry.get('provenance') == 'rigid_pose_transfer_of_native_reference_base_vertices'
        evidence.update(footprint_source=("validated_reference_base_vertices_rigid_pose_transfer" if transferred else "validated_native_base_collision_vertices"),
            base_geometry_provenance=geometry.get('provenance'),
            target_vertices_newly_measured=not transferred,
            reference_inventory_sha256=geometry.get('reference_inventory_sha256'), base_geometry_validated=True,
            base_geometry_fallback_reason=None, base_geometry_enabled_mesh_names=deepcopy(names) if isinstance(names, list) else [])
        # Supplied base_local_xy_bounds are intentionally unused. The planner
        # recomputes both full-pose and planar projections from these vertices.
        return vertices.copy(), evidence
    except (KeyError, TypeError, ValueError) as error:
        evidence["base_geometry_fallback_reason"] = f"{type(error).__name__}: {error}"
        return None, evidence


def transform_points(points, transform):
    """Apply a column-vector rigid transform to row-vector xyz points."""
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def sphere_aabb_clearances(centers, radii, box):
    """Signed sphere/AABB separation: zero is contact, negative is overlap."""
    centers, radii = np.asarray(centers), np.asarray(radii)
    offset = np.maximum(np.maximum(box[0] - centers, centers - box[1]), 0.)
    outside = np.linalg.norm(offset, axis=1)
    inside = np.all((centers >= box[0]) & (centers <= box[1]), axis=1)
    depth = np.min(np.minimum(centers - box[0], box[1] - centers), axis=1)
    return np.where(inside, -depth, outside) - radii


def joint_limit_margin(values, limits):
    """Minimum native-unit distance to a limit; negative means a violation."""
    margins = [min(value - limits[name][0], limits[name][1] - value)
               for name, value in values.items() if name in limits]
    return float(min(margins)) if margins else None


def tool_orientation(approach_axis_eef, opening_axis_eef, approach_world, opening_world):
    """Map measured tool approach/opening axes to an orthonormal world frame."""
    def basis(approach, opening):
        approach = np.asarray(approach, float)
        approach = approach / np.linalg.norm(approach)
        opening = np.asarray(opening, float) - np.dot(opening, approach) * approach
        opening = opening / np.linalg.norm(opening)
        if not np.isfinite(np.r_[approach, opening]).all():
            raise AuditUnavailable("Distinct finite approach and opening directions are required.")
        return np.column_stack((opening, np.cross(approach, opening), approach))
    return basis(approach_world, opening_world) @ basis(approach_axis_eef, opening_axis_eef).T


def _approaches(gripper_geometry, stance_xy, target_position):
    side = np.r_[np.asarray(target_position)[:2] - np.asarray(stance_xy), 0.]
    side /= np.linalg.norm(side)
    result = []
    for name, direction, opening in (("top_roll_0", [0., 0., -1.], [1., 0., 0.]),
        ("top_roll_90", [0., 0., -1.], [0., 1., 0.]),
        ("side_roll_0", side, [0., 0., 1.]),
        ("side_roll_90", side, np.cross(side, [0., 0., 1.]))):
        result.append({"name": name, "direction_world": np.asarray(direction),
            "orientation_world": tool_orientation(gripper_geometry["approach_axis_eef"],
                gripper_geometry["opening_axis_eef"], direction, opening)})
    return result


def _support_bounds(target, objects):
    if target.get("support_aabb") is not None:
        return _box(target["support_aabb"]), "explicit_target_support_aabb"
    support = target.get("support")
    named = [row for row in objects if row["name"] == support]
    if named:
        return _box(named[0]["aabb"]), "named_support_object_aabb"
    bottom = _box(target["aabb"])[0, 2]
    panels = []
    for row in objects:
        if not support or not (row.get("parent_object") == support or row["name"].startswith(support + "::")):
            continue
        box = _box(row["aabb"])
        span = np.ptp(box, axis=0)
        if abs(box[1, 2] - bottom) <= .08 and span[2] <= .5 * min(span[:2]):
            panels.append(box)
    if panels:
        return np.stack((np.min([box[0] for box in panels], axis=0),
                         np.max([box[1] for box in panels], axis=0))), "union_of_horizontal_support_collision_mesh_aabbs"
    return _box(target["aabb"]), "support_unavailable_target_aabb_fallback"


def _policy_metadata(row):
    return {key: deepcopy(row[key]) for key in ("role", "policy_role", "protected", "grasp_required", "fixed_base")
            if key in row}


def _task_movable(row):
    return row.get("fixed_base") is False and str(row.get("role", "target")).lower() not in {
        "background", "floor", "ground", "robot", "structure"}


def _initial_targets(inventory):
    rows = inventory.get("objects", [])
    by_name = {row["name"]: row for row in rows}
    selected = {row["name"]: row for row in rows if row.get("grasp_required", False) or _task_movable(row)
                or (row.get("rigid_grasp_proxy_eligible") is False and row.get("role") == "target")}
    for requested in inventory.get("targets") or []:
        row = dict(by_name.get(requested, {}), name=requested) if isinstance(requested, str) else dict(
            by_name.get(requested.get("name"), {}), **requested)
        if row.get("grasp_required", True) or _task_movable(row) or row.get("rigid_grasp_proxy_eligible") is False:
            selected[row["name"]] = row
    return list(selected.values())


def _ineligible_target(row):
    return {"name": row.get("name", "unknown"), "status": "unavailable", "witness": None,
        "kinematic_witness": None, "ik_attempts": 0, "rigid_grasp_proxy_eligible": False,
        "reason": "Explicit rigid_grasp_proxy_eligible:false: a nonrigid, articulated or otherwise unsupported whole-body proxy requires a different grasp representation; no rigid grasp or access result is inferred.",
        "input_metadata": deepcopy(row), "policy_metadata": _policy_metadata(row), "policy_permission_inferred": False}


def _interior_access_gap(region, objects):
    verified_open = region.get("doors_open") is True or region.get("open_access_verified") is True
    if region.get("requires_open_doors") is True and not verified_open:
        return "Interior destination requires opened doors; opening and access are not verified in the initial state."
    interior = region.get("interior") is True or region.get("is_interior") is True or region.get("region_type") == "interior"
    support = next((row for row in objects if row["name"] == region.get("support") and row.get("fixed_base") is True), None)
    if support is not None:
        box, bounds = _box(support["aabb"]), _box(region["aabb"])
        center = bounds.mean(axis=0)
        height = float(region["support_z"])
        interior |= bool(box[0, 2] + .03 < height < box[1, 2] - .03
                         and np.all(center[:2] >= box[0, :2]) and np.all(center[:2] <= box[1, :2]))
    if interior and not verified_open:
        return "Interior destination access and any required door opening are unverified; the initial closed geometry is retained."
    return None


def _virtual_destination(region, objects, payloads, *, center_xy=None):
    """Translate one measured task payload without modifying input actor records."""
    bounds = _box(region["aabb"])
    region_span = np.ptp(bounds, axis=0)
    support_z = float(region["support_z"])
    if not np.isfinite(support_z):
        raise AuditUnavailable("A finite measured destination support_z is required.")
    fits = [row for row in payloads if _task_movable(row) and row.get("rigid_grasp_proxy_eligible") is not False
            and np.all(np.ptp(_box(row["aabb"]), axis=0)[:2] <= region_span[:2])]
    if not fits:
        raise AuditUnavailable("No non-fixed task payload has a measured XY footprint that fits this region.")
    source = min(fits, key=lambda row: (row["name"] != "item_01", row["name"]))
    old_box = _box(source["aabb"])
    center_xy = bounds.mean(axis=0)[:2] if center_xy is None else np.asarray(center_xy, float)
    translation = np.r_[center_xy - old_box.mean(axis=0)[:2], support_z - old_box[0, 2]]
    virtual = deepcopy(source)
    virtual["aabb"] = (old_box + translation).tolist()
    if "position" in source:
        virtual["position"] = (np.asarray(source["position"], float) + translation).tolist()
    virtual["support"] = region["support"]
    virtual.pop("support_aabb", None)
    if region.get("support_aabb") is not None:
        virtual["support_aabb"] = deepcopy(region["support_aabb"])
    source_name = source["name"]
    removed = [row["name"] for row in objects if row["name"] == source_name
               or row.get("parent_object") == source_name or row["name"].startswith(source_name + "::")]
    removed_names = set(removed)
    retained = [row for row in objects if row["name"] not in removed_names]
    metadata = {"source_name": source_name, "source_aabb": old_box.tolist(), "aabb": virtual["aabb"],
        "measured_size_m": np.ptp(old_box, axis=0).tolist(), "translation_world_m": translation.tolist(),
        "orientation": deepcopy(source.get("orientation")), "position": deepcopy(virtual.get("position")),
        "policy_metadata": _policy_metadata(source), "removed_original_collision_rows": removed}
    return virtual, [*retained, virtual], metadata


def _payload_overlap_details(virtual, objects):
    box = _box(virtual["aabb"])
    overlaps, permitted = [], []
    support = virtual.get("support")
    for row in objects:
        if row["name"] == virtual["name"]:
            continue
        other = _box(row["aabb"])
        overlap = np.minimum(box[1], other[1]) - np.maximum(box[0], other[0])
        if np.any(overlap[:2] <= 0):
            continue
        declared_support = bool(support and (row["name"] == support or row.get("parent_object") == support
                                             or row["name"].startswith(support + "::")))
        bottom_overlap = float(other[1, 2] - box[0, 2])
        if (declared_support and 0 <= bottom_overlap <= _SUPPORT_BOTTOM_TOLERANCE_M
                and other[0, 2] <= box[0, 2]):
            permitted.append({"object": row["name"], "support": support, "z_overlap_m": bottom_overlap,
                "bottom_tolerance_m": _SUPPORT_BOTTOM_TOLERANCE_M,
                "scope": "Payload/support-plane contact only; this support remains active in every robot collision check."})
        elif np.all(overlap > 0):
            overlaps.append(row["name"])
    return overlaps, permitted


def _payload_overlaps(virtual, objects):
    return _payload_overlap_details(virtual, objects)[0]


def _destination_centers(region, payload_aabb):
    bounds, payload = _box(region["aabb"]), _box(payload_aabb)
    half = np.ptp(payload, axis=0)[:2] / 2
    lower, upper = bounds[0, :2] + half + _DESTINATION_INSET_M, bounds[1, :2] - half - _DESTINATION_INSET_M
    middle = bounds.mean(axis=0)[:2]
    if np.any(lower > upper):
        return [middle], False
    centers = []
    for x, y in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        point = np.array([middle[0] if x == 0 else (lower[0] if x < 0 else upper[0]),
                          middle[1] if y == 0 else (lower[1] if y < 0 else upper[1])])
        if not any(np.allclose(point, previous, rtol=0., atol=1e-12) for previous in centers):
            centers.append(point)
    return centers, True


def _destination_reference(entry, sample, *, witness=False):
    entry.update(hypothetical_payload=deepcopy(sample["hypothetical_payload"]),
        retained_object_aabb_overlaps=deepcopy(sample["retained_object_aabb_overlaps"]),
        permitted_payload_support_contacts=deepcopy(sample["permitted_payload_support_contacts"]),
        selected_sample_index=sample["sample_index"],
        selected_sample_role="witness" if witness else "first_sample_reference")
    result = sample.get("result") or {}
    entry.update({key: value for key, value in result.items() if key not in {"name", "ik_attempts", "elapsed_s"}})
    if sample.get("navigation") is not None:
        entry["navigation"] = sample["navigation"]


def _prepare_destination_queries(regions, objects, payloads):
    entries, pending = [], []
    for name, region in regions.items():
        entry = {"name": f"destination:{name}", "region_name": name, "status": "unavailable", "witness": None,
            "kinematic_witness": None, "ik_attempts": 0, "support": region.get("support"),
            "support_z": region.get("support_z"), "region_aabb": deepcopy(region.get("aabb")),
            "hypothetical_payload": None, "policy_permission_inferred": False,
            "scope": _DESTINATION_SCOPE, "sample_records": [],
            "approach_evaluated_sample_count": 0, "ik_attempts_complete": True,
            "sampled_only_limitation": "Only the listed deterministic legal XY centers are screened. Unsampled space remains unresolved; separate sample paths are never combined, and carrying or placing is untested."}
        entries.append(entry)
        try:
            access_gap = _interior_access_gap(region, objects)
            if access_gap:
                raise AuditUnavailable(access_gap)
            virtual, _, metadata = _virtual_destination(region, objects, payloads)
            source = next(row for row in payloads if row["name"] == metadata["source_name"])
            centers, inset_fit = _destination_centers(region, virtual["aabb"])
            entry["sample_grid_policy"] = {"grid_shape": [3, 3], "inset_m": _DESTINATION_INSET_M,
                "inset_fit": inset_fit, "ordering": "center_then_cardinals_then_corners",
                "geometric_center_fallback_without_inset": not inset_fit}
            entry["policy_metadata"] = _policy_metadata(source)
            plans = []
            for index, center in enumerate(centers):
                virtual, query_objects, metadata = _virtual_destination(region, objects, [source], center_xy=center)
                overlap, contacts = _payload_overlap_details(virtual, query_objects)
                sample = {"sample_index": index, "payload_center_world_m": _box(virtual["aabb"]).mean(axis=0).tolist(),
                    "aabb": deepcopy(virtual["aabb"]), "hypothetical_payload": metadata,
                    "retained_object_aabb_overlaps": overlap, "permitted_payload_support_contacts": contacts,
                    "status": "not_found" if overlap else "pending", "ik_attempts": 0,
                    "approach_evaluated": False, "result": None}
                if overlap:
                    sample["result"] = {"status": "not_found", "witness": None, "kinematic_witness": None,
                        "ik_attempts": 0, "reason": "The virtual payload overlaps retained initial-state object AABBs at this sample."}
                else:
                    plans.append((sample, virtual, query_objects))
                entry["sample_records"].append(sample)
            _destination_reference(entry, entry["sample_records"][0])
            entry["sample_count"] = len(entry["sample_records"])
            entry["sample_status_counts"] = dict(Counter(sample["status"] for sample in entry["sample_records"]))
            if plans:
                entry.update(status="unavailable")
                entry.pop("reason", None)
                pending.append((entry, plans))
            else:
                entry.update(status="not_found", reason="All sampled legal XY centers overlap retained initial-state object AABBs; unsampled space remains unresolved.")
        except (AuditUnavailable, KeyError, ValueError) as error:
            entry["reason"] = f"{type(error).__name__}: {error}"
    return entries, pending


def _usd_calibration(path, pose_matrix):
    try:
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            directories = sorted((Path(sys.prefix) / "lib").glob(
                "python*/site-packages/isaacsim/extscache/omni.usd.libs-*"))
            if directories:
                sys.path.insert(0, str(directories[-1]))
            from pxr import Usd, UsdPhysics
    except ImportError as error:
        raise AuditUnavailable(f"CPU USD reader unavailable: {error}; supply complete joint_calibration "
                               "or start Python with omni.usd.libs bin and bin/deps in LD_LIBRARY_PATH.") from error
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise AuditUnavailable(f"Cannot open supplied robot USD: {path}")
    calibration, limits = {}, {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        bodies = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if not all(len(body) == 1 for body in bodies):
            continue
        def matrix(position, quaternion):
            return pose_matrix(position.Get(), [*quaternion.Get().GetImaginary(), quaternion.Get().GetReal()])
        kind = ("revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else
                "prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else
                "fixed" if prim.IsA(UsdPhysics.FixedJoint) else None)
        if kind is None:
            continue
        axis = {"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}.get(
            prim.GetAttribute("physics:axis").Get(), [1., 0., 0.])
        name = prim.GetName()
        calibration[name] = {"parent": bodies[0][0].name, "child": bodies[1][0].name,
            "parent_joint": matrix(joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()).tolist(),
            "child_joint": matrix(joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()).tolist(),
            "axis": axis, "kind": kind}
        if kind != "fixed":
            lo, hi = [prim.GetAttribute(f"physics:{side}Limit").Get() for side in ("lower", "upper")]
            if lo is not None and hi is not None and np.isfinite([lo, hi]).all():
                limits[name] = (np.deg2rad([lo, hi]) if kind == "revolute" else np.array([lo, hi])).tolist()
    return calibration, limits


class _Robot:
    def __init__(self, row, options):
        try:
            from core.camera_geometry import RobotKinematics, axis_rotation, pose_matrix
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
            import yaml
        except ImportError as error:
            raise AuditUnavailable(f"CPU kinematics dependency unavailable: {error}") from error
        self.pose_matrix, self.axis_rotation = pose_matrix, axis_rotation
        self.least_squares, self.Rotation = least_squares, Rotation
        self.options = options
        self.urdf = Path(row["urdf_path"]).expanduser()
        self.kin = RobotKinematics(self.urdf, root="base_link")
        nominal = {joint[0]: (parent, joint) for parent, children in self.kin.children.items() for joint in children}
        self.seed = {name: float(value) for name, value in row["joints"].items()}
        self.limits = {name: list(value) for name, value in row.get("joint_limits", {}).items()}
        calibration = dict(row.get("joint_calibration", {}))
        # The URDF lacks verified fixed EEF mounts and differs from installed USD
        # for moving joints. Missing calibrated chains must never silently use it.
        required = {f"{arm}_arm_joint{i}" for arm in ("left", "right") for i in range(1, 7)}
        required |= {f"torso_joint{i}" for i in range(1, 5)}
        required |= {f"{arm}_gripper_axis{i}" for arm in ("left", "right") for i in (1, 2)}
        calibrated_children = {data["child"] for data in calibration.values()}
        if not required.issubset(calibration) or not {"left_eef_link", "right_eef_link"}.issubset(calibrated_children):
            usd = row.get("usd_path")
            if not usd or not Path(usd).is_file():
                raise AuditUnavailable("Complete moving-joint and fixed EEF calibration, or an existing robot USD, is required.")
            measured, usd_limits = _usd_calibration(usd, pose_matrix)
            measured.update(calibration)
            calibration = measured
            for name, limits in usd_limits.items():
                self.limits.setdefault(name, limits)
        self.calibration = calibration
        for name, data in calibration.items():
            old = nominal.get(name)
            kind = data.get("kind", old[1][1] if old else None)
            if kind is None:
                raise AuditUnavailable(f"Extra calibrated joint {name} needs an explicit verified joint kind.")
            parent, child = data["parent"], data["child"]
            if old and (old[0] != parent or old[1][2] != child):
                raise AuditUnavailable(f"Installed calibration changes the topology of {name}.")
            before, after = np.asarray(data["parent_joint"], float), np.linalg.inv(np.asarray(data["child_joint"], float))
            axis = np.asarray(data["axis"], float)
            if (before.shape != (4, 4) or after.shape != (4, 4) or axis.shape != (3,)
                    or not np.isfinite(np.r_[before.ravel(), after.ravel(), axis]).all()
                    or not np.isclose(np.linalg.norm(axis), 1)):
                raise AuditUnavailable(f"Invalid measured transforms for {name}.")
            self.kin.children[parent] = [item for item in self.kin.children.get(parent, []) if item[0] != name]
            self.kin.children[parent].append((name, kind, child, before, axis, after))
            self.kin.parents[child] = parent
        self.joint_kinds = {joint[0]: joint[1] for children in self.kin.children.values() for joint in children}
        for name in required:
            if name not in self.seed or name not in self.limits:
                raise AuditUnavailable(f"Measured encoder and native limits required for {name}.")
            lo, hi = self.limits[name]
            if not np.isfinite([lo, hi, self.seed[name]]).all() or not lo < hi:
                raise AuditUnavailable(f"Finite, nonempty measured limits required for {name}.")
        if joint_limit_margin(self.seed, self.limits) < -1e-5:
            raise AuditUnavailable("Measured initial robot posture violates supplied joint limits.")
        for arm in ("left", "right"):
            for i in (1, 2):
                name = f"{arm}_gripper_axis{i}"
                if not self.limits[name][0] <= options.aperture_m / 2 <= self.limits[name][1]:
                    raise AuditUnavailable("The explicit 0.1 m aperture does not fit measured gripper joint limits.")
        sphere_path = Path(row.get("collision_spheres_path", self.urdf.parent.parent /
                                   "curobo/r1_description_curobo_arm_no_torso.yaml"))
        if not sphere_path.is_file():
            raise AuditUnavailable(f"Installed R1 collision-sphere asset missing: {sphere_path}")
        config = yaml.safe_load(sphere_path.read_text())["robot_cfg"]["kinematics"]
        self.spheres = {name: rows for name, rows in config["collision_spheres"].items()
                        if not name.startswith("attached_object")}
        self.ignored = {frozenset((left, right)) for left, others in config.get("self_collision_ignore", {}).items()
                        for right in others}
        self.sphere_path = sphere_path
        self.chains = {arm: self.chain(f"{arm}_eef_link") for arm in ("left", "right")}
        self.kin.frames(self.seed, links=set(self.spheres) | {"left_eef_link", "right_eef_link"})
        self.position = np.asarray(row["position"], float)
        self.initial_pose = pose_matrix(self.position, row["orientation"])
        if abs(self.initial_pose[2, 2] - 1) > 1e-4:
            raise AuditUnavailable("Planar stance screening requires an upright measured robot root.")
        self.base_box = _box(row["base_aabb"])
        self.base_world_vertices, self.base_geometry_validation = _validated_base_vertices(row.get("base_geometry"), self.base_box)
        self.base_radius = float(np.max(np.linalg.norm(_corners(self.base_box)[:, :2] - self.position[:2], axis=1)))

    def chain(self, link):
        result, seen = [], set()
        while link != self.kin.root:
            if link in seen or link not in self.kin.parents:
                raise AuditUnavailable(f"Missing calibrated chain to {link}.")
            seen.add(link)
            parent = self.kin.parents[link]
            result.append(next(joint for joint in self.kin.children[parent] if joint[2] == link))
            link = parent
        return result[::-1]

    def fk(self, arm, joints):
        result = np.eye(4)
        for name, kind, _, before, axis, after in self.chains[arm]:
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = self.axis_rotation(axis, joints[name])
            elif kind == "prismatic":
                motion[:3, 3] = axis * joints[name]
            result = result @ before @ motion @ after
        return result

    def torso_presets(self):
        names = [f"torso_joint{i}" for i in range(1, 5)]
        result = [{name: self.seed[name] for name in names}]
        # Two deterministic interior presets interpolate from the measured
        # posture toward each measured range's midpoint, without copied angles.
        for fraction in (.4, .8):
            result.append({name: (1 - fraction) * self.seed[name] + fraction * np.mean(self.limits[name])
                           for name in names})
        return result

    def solve(self, arm, joints, position, orientation, attempt_index):
        names = [f"{arm}_arm_joint{i}" for i in range(1, 7)]
        if self.options.torso_search == "continuous":
            names = [f"torso_joint{i}" for i in range(1, 5)] + names
        lower, upper = np.asarray([self.limits[name] for name in names]).T
        seed = np.asarray([joints[name] for name in names])
        if self.options.torso_search == "fixed_presets" and attempt_index % 2:
            seed = .5 * seed + .5 * (lower + upper) / 2
        def residual(values):
            pose = self.fk(arm, dict(joints, **dict(zip(names, values))))
            return np.r_[pose[:3, 3] - position,
                .25 * self.Rotation.from_matrix(orientation.T @ pose[:3, :3]).as_rotvec()]
        solution = self.least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), max_nfev=self.options.max_ik_evaluations,
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        values = dict(joints, **dict(zip(names, map(float, solution.x))))
        pose = self.fk(arm, values)
        return values, float(np.linalg.norm(pose[:3, 3] - position)), float(np.rad2deg(
            self.Rotation.from_matrix(orientation.T @ pose[:3, :3]).magnitude()))

    def cloud(self, joints, base):
        frames = self.kin.frames(joints, links=set(self.spheres))
        centers, radii, links = [], [], []
        for link, rows in self.spheres.items():
            for row in rows:
                centers.append(transform_points(np.asarray([row["center"]]), base @ frames[link])[0])
                radii.append(row["radius"])
                links.append(link)
        return np.asarray(centers), np.asarray(radii), links

    def gripper_dimensions(self, arm, joints):
        frames = self.kin.frames(joints, links={f"{arm}_eef_link", f"{arm}_arm_link6",
                                               f"{arm}_gripper_link1", f"{arm}_gripper_link2"})
        eef_inverse = np.linalg.inv(frames[f"{arm}_eef_link"])
        bounds = []
        for link, rows in self.spheres.items():
            if link in {f"{arm}_gripper_link1", f"{arm}_gripper_link2"}:
                for sphere in rows:
                    center = transform_points([sphere["center"]], eef_inverse @ frames[link])[0]
                    bounds.extend([center - sphere["radius"], center + sphere["radius"]])
        bounds = np.asarray(bounds)
        # Opening direction comes from the calibrated prismatic finger axis.
        joint = next(item for items in self.kin.children.values() for item in items
                     if item[0] == f"{arm}_gripper_axis1")
        axis = eef_inverse[:3, :3] @ frames[f"{arm}_gripper_link1"][:3, :3] @ joint[4]
        # The calibrated wrist -> fixed EEF offset measures the tool's forward
        # direction. For installed R1 it is approximately EEF +Z; no finger-X
        # convention is assumed. The sphere envelope is reported separately.
        approach = eef_inverse[:3, :3] @ (frames[f"{arm}_eef_link"][:3, 3] - frames[f"{arm}_arm_link6"][:3, 3])
        if np.linalg.norm(approach) < 1e-8:
            raise AuditUnavailable("A nonzero calibrated wrist-to-EEF offset is required for tool approach direction.")
        approach /= np.linalg.norm(approach)
        return {"aperture_m": self.options.aperture_m, "finger_joint_values_m": [self.options.aperture_m / 2] * 2,
                "opening_axis_eef": axis.tolist(), "approach_axis_eef": approach.tolist(),
                "approach_axis_source": "calibrated_wrist_to_fixed_eef_offset", "sphere_envelope_eef_m":
                [bounds.min(axis=0).tolist(), bounds.max(axis=0).tolist()],
                "sphere_envelope_dimensions_m": np.ptp(bounds, axis=0).tolist()}


def _collision(robot, joints, base, objects, target_name, arm, terminal_contact, margin, floor_z=None):
    centers, radii, links = robot.cloud(joints, base)
    minimum, hits = float("inf"), []
    if floor_z is not None:
        # Wheels deliberately contact the measured supporting plane. Every
        # non-wheel sphere still has to clear it, including both arms and base.
        active = np.asarray([not link.startswith("wheel_link") for link in links])
        clearance = centers[:, 2] - radii - floor_z
        if np.any(active):
            minimum = min(minimum, float(clearance[active].min()))
        hits.extend(f"environment:supporting_floor:{links[index]}" for index in
                    np.flatnonzero(active & (clearance < margin)))
    for row in objects:
        clearance = sphere_aabb_clearances(centers, radii, _box(row["aabb"]))
        permitted = np.asarray([terminal_contact and row["name"] == target_name
                                and link in {f"{arm}_gripper_link1", f"{arm}_gripper_link2"} for link in links])
        active = clearance[~permitted]
        if len(active):
            minimum = min(minimum, float(active.min()))
        for index in np.flatnonzero((clearance < margin) & ~permitted):
            hits.append(f"environment:{row['name']}:{links[index]}")
    for i, link in enumerate(links):
        for j in range(i):
            if link == links[j] or frozenset((link, links[j])) in robot.ignored:
                continue
            clearance = float(np.linalg.norm(centers[i] - centers[j]) - radii[i] - radii[j])
            minimum = min(minimum, clearance)
            if clearance < margin:
                hits.append(f"self:{links[j]}:{link}")
    return hits, minimum if np.isfinite(minimum) else None


class _Navigation:
    def __init__(self, robot, objects, targets, options):
        self.start = robot.position[:2]
        points = [self.start, *[np.mean(_box(row["aabb"]), axis=0)[:2] for row in targets]]
        points = np.asarray(points)
        self.lower = points.min(axis=0) - 2.
        upper = points.max(axis=0) + 2.
        self.resolution = max(options.grid_resolution_m,
                              np.sqrt(np.prod(upper - self.lower) / options.max_grid_cells))
        self.shape = tuple(map(int, (np.ceil((upper - self.lower) / self.resolution).astype(int) + 1)[::-1]))
        self.radius = robot.base_radius
        inflation = self.radius + options.collision_margin_m + self.resolution / np.sqrt(2)
        self.obstacles = []
        for row in objects:
            box = _box(row["aabb"])
            if box[0, 2] <= robot.base_box[1, 2] and box[1, 2] >= robot.base_box[0, 2]:
                self.obstacles.append((box[0, :2] - inflation, box[1, :2] + inflation, row["name"]))
        ys, xs = np.indices(self.shape)
        centers = self.lower + np.stack([xs, ys], axis=-1) * self.resolution
        self.blocked = np.zeros(self.shape, dtype=bool)
        for low, high, _ in self.obstacles:
            self.blocked |= np.all((centers >= low) & (centers <= high), axis=-1)
        self.initial = self.node(self.start)
        self.previous = {self.initial: None}
        self.failure = None
        if self.blocked[self.initial] or self.segment_blocked(self.start, self.position(self.initial)):
            self.failure = "initial_base_blocked_by_conservative_footprint_or_grid"
            self.previous = {}
            return
        queue = deque([self.initial])
        while queue:
            y, x = queue.popleft()
            for node in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= node[0] < self.shape[0] and 0 <= node[1] < self.shape[1]
                        and node not in self.previous and not self.blocked[node]):
                    self.previous[node] = (y, x)
                    queue.append(node)

    def node(self, point):
        return tuple(np.rint((point - self.lower) / self.resolution).astype(int)[::-1])

    def position(self, node):
        return self.lower + np.asarray(node[::-1]) * self.resolution

    def segment_blocked(self, start, end):
        for low, high, _ in self.obstacles:
            # Closed slab intersection, including a zero-length initial link.
            begin, finish = 0., 1.
            for axis in (0, 1):
                delta = end[axis] - start[axis]
                if abs(delta) < 1e-12:
                    if not low[axis] <= start[axis] <= high[axis]:
                        break
                else:
                    a, b = sorted(((low[axis] - start[axis]) / delta, (high[axis] - start[axis]) / delta))
                    begin, finish = max(begin, a), min(finish, b)
                    if begin > finish:
                        break
            else:
                return True
        return False

    def route(self, node):
        route = []
        while node is not None:
            route.append(self.position(node).tolist())
            node = self.previous[node]
        return [self.start.tolist(), *route[::-1]]

    def stances(self, target, objects, options):
        target_box = _box(target["aabb"])
        center = target_box.mean(axis=0)
        box, _ = _support_bounds(target, objects)
        candidates = []
        for axis, sign, fraction in product((0, 1), (-1, 1), (.2, .5, .8)):
            point = center[:2].copy()
            point[axis] = box[0 if sign < 0 else 1, axis] + sign * (self.radius + .14)
            other = 1 - axis
            point[other] = (1 - fraction) * box[0, other] + fraction * box[1, other]
            node = self.node(point)
            if (node not in self.previous or np.linalg.norm(self.position(node) - center[:2]) >
                    options.max_stance_target_distance_m):
                continue
            point = self.position(node)
            yaw = float(np.arctan2(center[1] - point[1], center[0] - point[0]))
            if not any(previous[0] == node for previous in candidates):
                candidates.append((node, yaw))
        candidates.sort(key=lambda candidate: np.linalg.norm(self.position(candidate[0]) - center[:2]) +
                        .05 * np.linalg.norm(self.position(candidate[0]) - self.start))
        return candidates[:options.max_stances]


def _convex_hull_2d(points):
    points = sorted(set(map(tuple, np.asarray(points, float))))
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def polygon_aabb_intersections(polygon, boxes, inflation=0.):
    """Conservative closed SAT intersections with inflated obstacle XY AABBs."""
    polygon, boxes = np.asarray(polygon, float), np.asarray(boxes, float).reshape(-1, 2, 2)
    if not len(boxes):
        return np.zeros(0, dtype=bool)
    center = boxes.mean(axis=1)
    extent = (boxes[:, 1] - boxes[:, 0]) / 2 + inflation
    edges = np.roll(polygon, -1, axis=0) - polygon
    axes = np.vstack((np.eye(2), np.column_stack((-edges[:, 1], edges[:, 0]))))
    hit = np.ones(len(boxes), dtype=bool)
    for axis in axes:
        projection = polygon @ axis
        middle, radius = center @ axis, extent @ np.abs(axis)
        hit &= (middle + radius >= projection.min()) & (middle - radius <= projection.max())
        if not np.any(hit):
            break
    return hit


def _angle_delta(after, before):
    return float((after - before + np.pi) % (2 * np.pi) - np.pi)


class _SE2Navigation:
    """Bounded 16-heading holonomic lattice with conservative swept-foot checks."""
    mode = "se2_rectangle"
    heading_count = 16

    def __init__(self, robot, objects, targets, options):
        self.options = options
        self.start = robot.position[:2].copy()
        self.initial_yaw = float(np.arctan2(robot.initial_pose[1, 0], robot.initial_pose[0, 0]))
        self.yaw_step = 2 * np.pi / self.heading_count
        measured = getattr(robot, "base_world_vertices", None)
        self.footprint_evidence = deepcopy(getattr(robot, "base_geometry_validation",
                                                  _validated_base_vertices(None, robot.base_box)[1]))
        if measured is not None and self.footprint_evidence.get("base_geometry_validated") is not True:
            measured = None
            self.footprint_evidence["base_geometry_fallback_reason"] = "Native vertices lack _Robot validation; measured AABB fallback retained."
        if measured is None:
            measured = _corners(robot.base_box)
            self.footprint_evidence.update(footprint_source="measured_base_aabb_corners", base_geometry_validated=False)
            if not self.footprint_evidence.get("base_geometry_fallback_reason"):
                self.footprint_evidence["base_geometry_fallback_reason"] = "Validated native base vertices are absent on the robot."
        self.footprint_evidence["footprint_vertex_count"] = len(measured)
        self.footprint_evidence["footprint_projection"] = "inverse_full_robot_pose_union_yaw_only_projection"
        # Use validated native vertices, or the unchanged eight-corner AABB
        # fallback. Neither supplied local bounds nor chassis guesses are used.
        local = transform_points(measured, np.linalg.inv(robot.initial_pose))[:, :2]
        self.local_bounds = np.stack((local.min(axis=0), local.max(axis=0)))
        # The lattice is planar. Union the same measured corners' yaw-only
        # projection so measured roll/pitch cannot shrink their world footprint.
        yaw_rotation = self._rotation(self.initial_yaw)
        projected = (measured[:, :2] - self.start) @ yaw_rotation
        self.local_bounds[0] = np.minimum(self.local_bounds[0], projected.min(axis=0))
        self.local_bounds[1] = np.maximum(self.local_bounds[1], projected.max(axis=0))
        low, high = self.local_bounds
        self.local_polygon = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
        self.radius = float(np.linalg.norm(self.local_polygon, axis=1).max())
        points = np.asarray([self.start, *[_box(row["aabb"]).mean(axis=0)[:2] for row in targets]])
        minimum, maximum = points.min(axis=0) - 2., points.max(axis=0) + 2.
        self.state_cap = min(options.max_grid_cells, 250000)
        self.resolution = max(options.grid_resolution_m,
            np.sqrt(self.heading_count * np.prod(maximum - minimum) / self.state_cap))
        while True:
            self.lower = self.start - np.ceil((self.start - minimum) / self.resolution) * self.resolution
            shape_xy = np.ceil((maximum - self.lower) / self.resolution).astype(int) + 1
            self.shape = (self.heading_count, int(shape_xy[1]), int(shape_xy[0]))
            if np.prod(self.shape) <= self.state_cap:
                break
            self.resolution *= 1.01
        self.obstacles = []
        for row in objects:
            box = _box(row["aabb"])
            if box[0, 2] <= robot.base_box[1, 2] and box[1, 2] >= robot.base_box[0, 2]:
                self.obstacles.append((box[0, :2], box[1, :2], row["name"]))
        reach_low = self.lower - self.radius - options.collision_margin_m
        reach_high = self.lower + (shape_xy - 1) * self.resolution + self.radius + options.collision_margin_m
        self.boxes = np.asarray([[low, high] for low, high, _ in self.obstacles
            if np.all(high >= reach_low) and np.all(low <= reach_high)], float).reshape(-1, 2, 2)
        self.polygons = [self.local_polygon @ self._rotation(self.heading(k)).T for k in range(self.heading_count)]
        self.translation_shapes = {}
        for k, direction in product(range(self.heading_count), ((-1, 0), (1, 0), (0, -1), (0, 1))):
            delta = np.array(direction[::-1]) * self.resolution
            self.translation_shapes[k, direction] = _convex_hull_2d(np.vstack((self.polygons[k], self.polygons[k] + delta)))
        self.rotation_shapes = {}
        self.max_rotation_sample_step_rad = 0.
        for k, sign in product(range(self.heading_count), (-1, 1)):
            self.rotation_shapes[k, sign] = self._rotation_samples(self.heading(k), self.heading(k) + sign * self.yaw_step)
        self.stance_positions, self.stance_yaws = {}, {}
        self.previous, self.pose_cache = {}, {}
        self.failure = None
        self.initial = self.node(self.start, self.initial_yaw)
        if self.pose_blocked(self.start, self.initial_yaw):
            self.failure = "initial_base_rectangle_collision"
            return
        if self.pose_blocked(self.grid_position(self.initial), self.initial_yaw) or self.translation_blocked(
                self.start, self.grid_position(self.initial), self.initial_yaw):
            self.failure = "actual_start_to_lattice_sweep_blocked"
            return
        self.previous[self.initial] = None
        queue = deque([self.initial])
        while queue:
            node = queue.popleft()
            k, y, x = node
            origin = self.grid_position(node)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = (k, y + dy, x + dx)
                if (not 0 <= other[1] < self.shape[1] or not 0 <= other[2] < self.shape[2]
                        or other in self.previous or self._node_blocked(other)):
                    continue
                if not self._blocked(self.translation_shapes[k, (dy, dx)] + origin):
                    self.previous[other] = node
                    queue.append(other)
            for sign in (-1, 1):
                other = ((k + sign) % self.heading_count, y, x)
                if other in self.previous or self._node_blocked(other):
                    continue
                if not self._sampled_rotation_blocked(origin, self.rotation_shapes[k, sign]):
                    self.previous[other] = node
                    queue.append(other)

    @staticmethod
    def _rotation(yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def heading(self, k):
        return self.initial_yaw + k * self.yaw_step

    def node(self, point, yaw=None):
        x, y = np.rint((np.asarray(point) - self.lower) / self.resolution).astype(int)
        k = 0 if yaw is None else int(np.rint(_angle_delta(yaw, self.initial_yaw) / self.yaw_step)) % self.heading_count
        return k, int(y), int(x)

    def grid_position(self, node):
        return self.lower + np.array([node[2], node[1]]) * self.resolution

    def position(self, node):
        return self.stance_positions.get(node, self.grid_position(node))

    def _blocked(self, polygon, extra_inflation=0.):
        inflation = self.options.collision_margin_m + extra_inflation
        if not len(self.boxes):
            return False
        near = np.all(self.boxes[:, 1] + inflation >= polygon.min(axis=0), axis=1) & np.all(
            self.boxes[:, 0] - inflation <= polygon.max(axis=0), axis=1)
        return bool(np.any(polygon_aabb_intersections(polygon, self.boxes[near], inflation)))

    def pose_blocked(self, xy, yaw):
        return self._blocked(self.local_polygon @ self._rotation(yaw).T + xy)

    def _node_blocked(self, node):
        if node not in self.pose_cache:
            self.pose_cache[node] = self._blocked(self.polygons[node[0]] + self.grid_position(node))
        return self.pose_cache[node]

    def translation_blocked(self, begin, end, yaw):
        polygon = self.local_polygon @ self._rotation(yaw).T
        return self._blocked(_convex_hull_2d(np.vstack((polygon + begin, polygon + end))))

    def _rotation_samples(self, begin, end):
        delta = _angle_delta(end, begin)
        count = max(1, int(np.ceil(abs(delta) / np.deg2rad(5.))))
        step = abs(delta) / count
        self.max_rotation_sample_step_rad = max(self.max_rotation_sample_step_rad, step)
        inflation = 2 * self.radius * np.sin(step / 4)
        return ([self.local_polygon @ self._rotation(begin + fraction * delta).T
                 for fraction in np.linspace(0., 1., count + 1)], float(inflation))

    def _sampled_rotation_blocked(self, xy, samples):
        polygons, inflation = samples
        return any(self._blocked(polygon + xy, inflation) for polygon in polygons)

    def rotation_blocked(self, xy, begin, end):
        return self._sampled_rotation_blocked(xy, self._rotation_samples(begin, end))

    def route(self, node):
        current, result = node, []
        while current is not None:
            result.append([*self.grid_position(current).tolist(), self.heading(current[0])])
            current = self.previous[current]
        result = [[*self.start.tolist(), self.initial_yaw], *result[::-1]]
        if node in self.stance_positions:
            result.extend([[*self.stance_positions[node].tolist(), self.heading(node[0])],
                           [*self.stance_positions[node].tolist(), self.stance_yaws[node]]])
        for index in range(1, len(result)):
            result[index][2] = result[index - 1][2] + _angle_delta(result[index][2], result[index - 1][2])
        return result

    @staticmethod
    def _same_stance(point, yaw, other_point, other_yaw):
        return bool(np.allclose(point, other_point, rtol=0., atol=1e-9) and abs(_angle_delta(yaw, other_yaw)) <= 1e-9)

    def _stance_proposals(self, target, objects, options):
        center = _box(target["aabb"]).mean(axis=0)
        box, _ = _support_bounds(target, objects)
        proposals, raw_counts = [], Counter()
        def add(point, provenance):
            raw_counts[provenance["kind"]] += 1
            difference = center[:2] - point
            yaw = float(np.arctan2(difference[1], difference[0])) if np.linalg.norm(difference) > 1e-12 else self.initial_yaw
            for proposal in proposals:
                if self._same_stance(point, yaw, proposal["point"], proposal["yaw"]):
                    proposal["provenance"].append(provenance)
                    return
            proposals.append({"point": np.asarray(point, float).copy(), "yaw": yaw, "provenance": [provenance]})
        for axis, sign, fraction in product((0, 1), (-1, 1), (.2, .5, .8)):
            edge = box[0 if sign < 0 else 1, axis]
            point = center[:2].copy()
            point[axis] = edge + sign * (self.radius + .14)
            point[1 - axis] = (1 - fraction) * box[0, 1 - axis] + fraction * box[1, 1 - axis]
            for _ in range(4):
                yaw = float(np.arctan2(center[1] - point[1], center[0] - point[0]))
                projection = (self.local_polygon @ self._rotation(yaw).T)[:, axis]
                point[axis] = edge + sign * .14 - (projection.min() if sign > 0 else projection.max())
            add(point, {"kind": "edge", "axis": axis, "sign": sign, "fraction": fraction})
        if np.linalg.norm(self.start - center[:2]) <= options.max_stance_target_distance_m:
            add(self.start, {"kind": "initial_root"})
        offset = float(np.arctan2(self.start[1] - center[1], self.start[0] - center[0]))
        for radius, index in product((.55, .75, .95, 1.15), range(16)):
            angle = offset + index * 2 * np.pi / 16
            add(center[:2] + radius * np.array([np.cos(angle), np.sin(angle)]),
                {"kind": "radial", "radius_m": radius, "direction_index": index, "angle_rad": angle})
        proposals.sort(key=lambda proposal: np.linalg.norm(proposal["point"] - center[:2]) +
                       .05 * np.linalg.norm(proposal["point"] - self.start))
        return proposals, raw_counts, offset

    def _stance_connection(self, point, yaw, reserved):
        _, y, x = self.node(point)
        nodes = []
        for node in product(range(self.heading_count), range(y - 1, y + 2), range(x - 1, x + 2)):
            if node not in self.previous:
                continue
            if node in self.stance_positions and not self._same_stance(
                    point, yaw, self.stance_positions[node], self.stance_yaws[node]):
                continue
            if node in reserved and not self._same_stance(point, yaw, *reserved[node]):
                continue
            nodes.append(node)
        nodes.sort(key=lambda node: np.linalg.norm(self.grid_position(node) - point) +
                   .1 * abs(_angle_delta(self.heading(node[0]), yaw)))
        for node in nodes:
            heading = self.heading(node[0])
            if not self.translation_blocked(self.grid_position(node), point, heading) and not self.rotation_blocked(point, heading, yaw):
                if node in self.stance_positions:
                    return node, self.stance_positions[node], self.stance_yaws[node]
                return node, point, yaw
        return None

    def stances(self, target, objects, options):
        center = _box(target["aabb"]).mean(axis=0)
        proposals, raw_counts, offset = self._stance_proposals(target, objects, options)
        feasible, reserved, pose_counts, connected_counts = [], {}, Counter(), Counter()
        for proposal in proposals:
            point, yaw = proposal["point"], proposal["yaw"]
            kinds = {source["kind"] for source in proposal["provenance"]}
            if np.linalg.norm(point - center[:2]) > options.max_stance_target_distance_m or self.pose_blocked(point, yaw):
                continue
            pose_counts.update(kinds)
            connection = self._stance_connection(point, yaw, reserved)
            if connection is None:
                continue
            node, point, yaw = connection
            reserved[node] = (point.copy(), yaw)
            connected_counts.update(kinds)
            feasible.append((proposal, node, point, yaw))
        feasible.sort(key=lambda candidate: np.linalg.norm(candidate[2] - center[:2]) +
                      .05 * np.linalg.norm(candidate[2] - self.start))
        selected = feasible[:options.max_stances]
        selected_counts, records, result = Counter(), [], []
        for proposal, node, point, yaw in selected:
            if node not in self.stance_positions:
                self.stance_positions[node], self.stance_yaws[node] = point.copy(), yaw
            # Existing mappings are canonical and are never overwritten, even
            # when a later target proposes a numerically equivalent pose.
            point, yaw = self.stance_positions[node], self.stance_yaws[node]
            selected_counts.update({source["kind"] for source in proposal["provenance"]})
            records.append({"lattice_node": list(node), "position_world_xy_m": point.tolist(), "yaw_rad": yaw,
                            "provenance": deepcopy(proposal["provenance"])})
            result.append((node, yaw))
        def source_counts(counter):
            return {kind: counter[kind] for kind in ("edge", "initial_root", "radial")}
        self.last_stance_sampling = {"method": "support_edges_initial_root_and_target_radials",
            "radial_radii_m": [.55, .75, .95, 1.15], "radial_direction_count": 16, "radial_angle_offset_rad": offset,
            "raw_proposal_counts": source_counts(raw_counts), "raw_proposal_count": sum(raw_counts.values()),
            "deduplicated_proposal_count": len(proposals), "pose_clear_counts_by_source": source_counts(pose_counts),
            "connected_feasible_counts_by_source": source_counts(connected_counts), "connected_feasible_count": len(feasible),
            "selected_counts_by_source": source_counts(selected_counts), "selected_count": len(selected),
            "max_stances": options.max_stances, "ranking": "target_distance_m + 0.05 * start_distance_m",
            "selected_stances": records,
            "scope": "Only sampled collision-free poses connected by checked footprint sweeps; full-body travel and unsampled stance feasibility remain unverified."}
        return result


def _make_navigation(robot, objects, targets, options):
    return (_SE2Navigation if options.navigation_mode == "se2_rectangle" else _Navigation)(robot, objects, targets, options)


def _navigation_evidence(navigation, options):
    result = {"mode": options.navigation_mode, "grid_shape": list(navigation.shape), "resolution_m": navigation.resolution,
              "reachable_cell_count": len(navigation.previous), "failure": navigation.failure}
    if options.navigation_mode == "se2_rectangle":
        result.update(navigation.footprint_evidence)
        result.update(heading_count=navigation.heading_count, heading_offset_rad=navigation.initial_yaw,
            reachable_state_count=len(navigation.previous),
            total_lattice_states=int(np.prod(navigation.shape)), state_cap=navigation.state_cap,
            local_rectangle_bounds_m=navigation.local_bounds.tolist(),
            local_rectangle_center_offset_m=navigation.local_bounds.mean(axis=0).tolist(),
            maximum_corner_radius_m=navigation.radius,
            maximum_rotation_sample_spacing_deg=float(np.rad2deg(navigation.max_rotation_sample_step_rad)),
            rotation_missed_motion_inflation_formula="2 * maximum_corner_radius_m * sin(actual_sample_step_rad / 4)")
    return result


def _sample_joint_path(robot, waypoints, base, objects, target_name, arm, options, floor_z):
    path, minimum = [], float("inf")
    for (before_stage, before), (after_stage, after) in zip(waypoints[:-1], waypoints[1:]):
        names = sorted(set(before) | set(after))
        steps = max(1, int(np.ceil(max(abs(after[name] - before[name]) /
            (options.max_prismatic_step_m if robot.joint_kinds.get(name) == "prismatic"
             else options.max_joint_step) for name in names))))
        for index, fraction in enumerate(np.linspace(0., 1., steps + 1)):
            values = {name: float(before[name] + fraction * (after[name] - before[name])) for name in names}
            terminal = (after_stage == "grasp" and index == steps) or (before_stage == "grasp" and index == 0)
            hits, clearance = _collision(robot, values, base, objects, target_name, arm, terminal,
                                         options.collision_margin_m, floor_z)
            if hits:
                return path, minimum if np.isfinite(minimum) else None, sorted(set(hits))
            if clearance is not None:
                minimum = min(minimum, clearance)
            path.append(values)
    return path, minimum if np.isfinite(minimum) else None, []


def _approach_height_candidates():
    return [(0, .75), (1, .75), (2, .5), (3, .5), (0, .5), (1, .5), (0, .9), (1, .9)]


def _grasp_point(box, height_fraction):
    point = box.mean(axis=0)
    point[2] = box[0, 2] + height_fraction * (box[1, 2] - box[0, 2])
    return point


def _screen_target(target, objects, robot, navigation, options, floor_z):
    if target.get("rigid_grasp_proxy_eligible") is False:
        return _ineligible_target(target)
    started = time.monotonic()
    rejections = Counter()
    box = _box(target["aabb"])
    dimensions = np.ptp(box, axis=0)
    support_box, support_source = _support_bounds(target, objects)
    result = {"name": target["name"], "status": "not_found", "interpretation":
        "Unresolved in a bounded search with conservative collision proxies; not evidence of unreachability.",
        "target_aabb_dimensions_m": dimensions.tolist(), "ik_attempts": 0, "witness": None,
        "kinematic_witness": None, "kinematic_witness_count": 0,
        "support_aabb_for_stance_sampling": support_box.tolist(), "support_bounds_source": support_source,
        "torso_search": options.torso_search, "coarse_shoulder_screen":
        "disabled_for_moving_torso" if options.torso_search == "continuous" else "fixed_shoulder_chain_bound"}
    result["policy_metadata"] = _policy_metadata(target)
    result["policy_permission_inferred"] = False
    stances = navigation.stances(target, objects, options)
    result["sampled_stance_count"] = len(stances)
    if options.navigation_mode == "se2_rectangle":
        result["stance_sampling"] = deepcopy(getattr(navigation, "last_stance_sampling", None))
    result["torso_presets"] = robot.torso_presets() if options.torso_search == "fixed_presets" else None
    opened = dict(robot.seed)
    for arm in ("left", "right"):
        for i in (1, 2):
            opened[f"{arm}_gripper_axis{i}"] = options.aperture_m / 2
    result["gripper_geometry"] = {arm: robot.gripper_dimensions(arm, opened) for arm in ("left", "right")}
    torso_names = [f"torso_joint{i}" for i in range(1, 5)]
    torso_seeds = (robot.torso_presets() if options.torso_search == "fixed_presets" else
                   [{name: robot.seed[name] for name in torso_names}])
    best_error, best_angle, best_kinematic_rank = None, None, None
    if not stances:
        rejections[navigation.failure or "no_connected_sampled_edge_stance"] += 1
    for torso, (node, yaw), arm, (orientation_index, height_fraction) in product(
            torso_seeds, stances, ("left", "right"), _approach_height_candidates()):
        if result["ik_attempts"] >= options.max_ik_attempts_per_target:
            rejections["ik_attempt_budget_exhausted"] += 1
            break
        base = robot.pose_matrix([*navigation.position(node), robot.position[2]],
                                 [0., 0., np.sin(yaw / 2), np.cos(yaw / 2)])
        seed = dict(opened, **torso)
        if options.torso_search == "fixed_presets":
            shoulder = robot.kin.frames(seed, links={f"{arm}_arm_link1"})[f"{arm}_arm_link1"]
            shoulder_world = (base @ shoulder)[:3, 3]
            chain = robot.chains[arm]
            arm_start = next(i for i, joint in enumerate(chain) if joint[0] == f"{arm}_arm_joint1")
            coarse_bound = sum(np.linalg.norm(joint[3][:3, 3]) + np.linalg.norm(joint[5][:3, 3])
                               for joint in chain[arm_start + 1:])
            if np.linalg.norm(shoulder_world - box.mean(axis=0)) > coarse_bound + .10:
                rejections["coarse_shoulder_distance_screen_only"] += 1
                continue
        approach = _approaches(result["gripper_geometry"][arm], navigation.position(node), box.mean(axis=0))[orientation_index]
        grasp_point = _grasp_point(box, height_fraction)
        rotation, direction = approach["orientation_world"], approach["direction_world"]
        opening_axis = rotation @ np.asarray(result["gripper_geometry"][arm]["opening_axis_eef"])
        projected_width = float(np.sum(np.abs(opening_axis) * dimensions))
        if projected_width > options.aperture_m - 2 * options.collision_margin_m:
            rejections["aabb_width_exceeds_aperture"] += 1
            continue
        local_orientation = base[:3, :3].T @ rotation
        stages, previous, blocking = [], seed, []
        optimized_names = ([*torso_names] if options.torso_search == "continuous" else []) + [
            f"{arm}_arm_joint{i}" for i in range(1, 7)]
        for stage, distance in (("pregrasp", options.approach_distance_m), ("grasp", 0.),
                                ("retract", options.approach_distance_m)):
            if result["ik_attempts"] >= options.max_ik_attempts_per_target:
                blocking.append("ik_attempt_budget_exhausted")
                break
            goal = grasp_point - direction * distance
            values, pe, ae = robot.solve(arm, previous, base[:3, :3].T @ (goal - base[:3, 3]),
                                         local_orientation, result["ik_attempts"])
            result["ik_attempts"] += 1
            best_error = min(best_error, pe) if best_error is not None else pe
            best_angle = min(best_angle, ae) if best_angle is not None else ae
            if pe > options.position_tolerance_m or ae > options.orientation_tolerance_deg:
                reason = f"{stage}:ik_residual"
                rejections[reason] += 1
                blocking.append(reason)
                break
            hits, clearance = _collision(robot, values, base, objects, target["name"], arm,
                                         stage == "grasp", options.collision_margin_m, floor_z)
            # Keep solving the pair after endpoint collision: kinematic evidence
            # must remain distinguishable from the collision/path outcome.
            reasons = [f"{stage}:{hit}" for hit in sorted(set(hits))]
            rejections.update(reasons)
            blocking.extend(reasons)
            stages.append({"name": stage, "joints": values, "goal_world_m": goal.tolist(),
                "position_error_m": pe, "orientation_error_deg": ae,
                "minimum_proxy_clearance_m": clearance, "collision_reasons": reasons,
                "joint_limit_margin_native_units": joint_limit_margin(values, robot.limits),
                "optimized_joint_limit_margin_native_units": joint_limit_margin(
                    {name: values[name] for name in optimized_names if name in values}, robot.limits)})
            previous = values
        if len(stages) < 2:
            continue
        result["kinematic_witness_count"] += 1
        route = navigation.route(node)
        geometry = {"arm": arm, "base_position_world_m": base[:3, 3].tolist(),
            "base_yaw_rad": float(route[-1][2]) if options.navigation_mode == "se2_rectangle" else yaw,
            "base_navigation_mode": options.navigation_mode,
            "base_footprint_route_world_xy_m": [point[:2] for point in route], "approach_name": approach["name"],
            "approach_direction_world": direction.tolist(), "eef_orientation_world_matrix": rotation.tolist(),
            "grasp_point_world": grasp_point.tolist(), "grasp_height_fraction": height_fraction,
            "grasp_candidate_kind": "open_gripper_geometry_only_closure_untested",
            "optimized_joint_names": optimized_names,
            "projected_target_width_m": projected_width, "aperture_m": options.aperture_m}
        if options.navigation_mode == "se2_rectangle":
            geometry["base_footprint_route_world_xy_yaw_rad"] = route
            geometry["base_route_scope"] = "Swept rectangle route with unwrapped shortest-rotation yaw, actual start and exact facing stance; whole-body base travel is unverified."
        path, minimum = [], None
        path_status = "not_checked_endpoint_collision_or_retract_ik"
        if len(stages) == 3 and not blocking:
            waypoints = [("measured", robot.seed), ("opened_seed", seed)] + [(stage["name"], stage["joints"]) for stage in stages]
            path, minimum, path_hits = _sample_joint_path(robot, waypoints, base, objects, target["name"], arm, options, floor_z)
            path_status = "blocked" if path_hits else "passed_sampled_checks"
            reasons = [f"joint_path:{hit}" for hit in path_hits]
            rejections.update(reasons)
            blocking.extend(reasons)
        kinematic = {**geometry, "stages": stages[:2], "retract_stage": stages[2] if len(stages) == 3 else None,
            "interpretation": "Orientation-aware pregrasp/grasp IK pair only; collision clearance and grasp success are separate questions.",
            "blocking_reasons": sorted(set(blocking)), "sampled_joint_path_status": path_status,
            "collision_and_path_status": "passed_sampled_checks" if not blocking and len(stages) == 3 else "blocked_or_unresolved"}
        rank = (len(set(blocking)), sum(stage["position_error_m"] for stage in stages[:2]))
        if best_kinematic_rank is None or rank < best_kinematic_rank:
            result["kinematic_witness"], best_kinematic_rank = kinematic, rank
        if blocking or len(stages) != 3:
            continue
        result.update(status="found_witness", interpretation="Open-gripper geometric witness found; grasp success is untested.")
        result["witness"] = {**geometry, "torso_joints": {name: stages[1]["joints"][name] for name in torso_names},
            "stages": stages, "sampled_joint_path": path, "sampled_joint_path_count": len(path),
            "minimum_sampled_proxy_clearance_m": minimum,
            "contact_policy": "Only active finger links may contact the named target at the grasp endpoint; all furniture remains active."}
        break
    result.update(rejection_counts=dict(rejections), minimum_ik_position_error_m=best_error,
                  minimum_ik_orientation_error_deg=best_angle, elapsed_s=round(time.monotonic() - started, 3))
    return result


def _screen_destination_samples(entry, plans, robot, options, floor_z):
    started, used, selected, unavailable_search = time.monotonic(), 0, None, False
    entry["aggregate_ik_budget"] = options.max_ik_attempts_per_target
    for sample, virtual, query_objects in plans:
        remaining = options.max_ik_attempts_per_target - used
        stop_reason = ("An earlier sample supplied the selected witness; this sample was not evaluated." if selected else
            "A sample search was unavailable; no further calls use its unknown remaining budget." if unavailable_search else
            "The region's aggregate IK budget is exhausted; this sample remains unresolved." if remaining <= 0 else None)
        if stop_reason:
            sample.update(status="not_evaluated", result={"status": "not_evaluated", "reason": stop_reason,
                          "witness": None, "kinematic_witness": None, "ik_attempts": 0})
            continue
        sample_options = replace(options, max_ik_attempts_per_target=remaining)
        try:
            navigation = _make_navigation(robot, query_objects, [virtual], sample_options)
            approach = _screen_target(virtual, query_objects, robot, navigation, sample_options, floor_z)
            attempts = int(approach.get("ik_attempts", 0))
            used += attempts
            sample.update(status=approach["status"], ik_attempts=attempts, result=approach,
                          approach_evaluated=True, navigation=_navigation_evidence(navigation, sample_options))
            if approach["status"] == "found_witness" and approach.get("witness") is not None:
                selected = sample
        except (AuditUnavailable, ImportError, OSError, KeyError, ValueError) as error:
            sample.update(status="unavailable", result={"status": "unavailable", "witness": None,
                "kinematic_witness": None, "reason": f"{type(error).__name__}: {error}"})
            unavailable_search = True
    _destination_reference(entry, selected or entry["sample_records"][0], witness=selected is not None)
    if selected is not None:
        entry.update(status="found_witness")
        entry.pop("reason", None)
    else:
        statuses = [sample["status"] for sample in entry["sample_records"]]
        status = ("unavailable" if unavailable_search and "not_found" not in statuses else
                  "not_found" if "not_found" in statuses or "not_evaluated" in statuses else "unavailable")
        entry.update(status=status,
            reason="No geometric witness was found among evaluated sampled centers; untested samples and unsampled space remain unresolved.")
    entry.update(ik_attempts=used, aggregate_ik_budget_remaining=max(0, options.max_ik_attempts_per_target - used)
        if not unavailable_search else None, ik_attempts_complete=not unavailable_search,
        sample_status_counts=dict(Counter(sample["status"] for sample in entry["sample_records"])),
        approach_evaluated_sample_count=sum(sample["approach_evaluated"] for sample in entry["sample_records"]),
        elapsed_s=round(time.monotonic() - started, 3))
    return entry


def audit_scene(inventory: dict, output_dir: Path, *, options: AuditOptions | dict | None = None) -> dict:
    """Write ``reachability.json`` with bounded geometric witnesses or explicit gaps."""
    options = AuditOptions(**options) if isinstance(options, dict) else options or AuditOptions()
    if options.torso_search not in {"continuous", "fixed_presets"}:
        raise ValueError("torso_search must be continuous or fixed_presets.")
    if options.navigation_mode not in {"circle", "se2_rectangle"}:
        raise ValueError("navigation_mode must be circle or se2_rectangle.")
    started = time.monotonic()
    rows = inventory.get("objects", [])
    targets = _initial_targets(inventory)
    eligible_targets = [row for row in targets if row.get("rigid_grasp_proxy_eligible") is not False]
    regions = inventory.get("regions", {})
    report = {"schema_version": 1, "method": "bounded_calibrated_dual_arm_ik_spheres_world_aabbs_2d_base_connectivity",
        "status": "completed", "options": asdict(options), "targets": [], "destination_queries": [], "robot_commands_executed": False,
        "inactive_targets": deepcopy(inventory.get("inactive_targets", [])),
        "unavailable_targets": deepcopy(inventory.get("unavailable_targets", [])),
        "structural_targets": deepcopy(inventory.get("structural_targets", [])),
        "inventory_accounting_scope": "Inventory accounting only; no access, grasp or temporal-phase result is inferred for inactive, unavailable or structural records.",
        "limitations": ["No robot or simulator execution. Only an open 0.1 m gripper is screened; closure, force closure, friction, lifting and holding are untested.",
            "Object world AABBs and installed robot spheres are conservative proxies: table voids and concavities may be filled. Not-found never means unreachable.",
            "Joint paths are sampled, not continuous swept-volume certificates; reported clearance is at sampled configurations.",
            "Base connectivity uses the selected circle grid or opt-in 16-heading swept rectangle lattice. Full-body motion during base travel is unverified.",
            "The measured floor plane remains active for all non-wheel spheres; wheel contact with that supporting plane is permitted.",
            "Two top-down rolls and two stance-facing side rolls are searched using calibrated tool approach/opening axes. Retract follows the selected approach and carries no object.",
            "Continuous mode optimizes four torso and six working-arm joints from measured posture; fixed_presets retains the bounded three-preset arm-only option.",
            "A separate kinematic_witness records a valid orientation-aware pregrasp/grasp IK pair even when collisions or later checks prevent a geometric witness.",
            "Destination queries translate one fitting measured task payload into a region while retaining all other initial objects. They do not connect pickup, carrying, door opening or placement motions.",
            "All movable task objects are counted, including protected objects; geometric evidence never grants policy permission to move them.",
            "Explicit rigid_grasp_proxy_eligible:false rows remain unavailable and retain metadata/collision geometry; they cannot become rigid grasp witnesses or hypothetical payloads."]}
    try:
        if not targets and not regions:
            report["status"] = "no_grasp_targets"
        else:
            for row in [*rows, *targets]:
                _box(row["aabb"])
            floor_z = float(inventory["floor_z"])
            if not np.isfinite(floor_z):
                raise AuditUnavailable("A finite measured supporting floor height is required.")
            # A recorded floor is the supporting plane, not a free-space obstacle.
            # No furniture is removed, including support tables of grasp targets.
            objects = [row for row in rows if not (str(row.get("role", "")).lower() in {"floor", "ground"}
                        and _box(row["aabb"])[1, 2] <= floor_z + .03)]
            known = {row["name"] for row in objects}
            objects += [row for row in targets if row["name"] not in known]
            report["destination_queries"], pending = _prepare_destination_queries(regions, objects, targets)
            for entry in report["destination_queries"]:
                entry["aggregate_ik_budget"] = options.max_ik_attempts_per_target
                entry["aggregate_ik_budget_remaining"] = options.max_ik_attempts_per_target
            initial_results = {row["name"]: _ineligible_target(row) for row in targets
                               if row.get("rigid_grasp_proxy_eligible") is False}
            if eligible_targets or pending:
                robot = _Robot(inventory["robot"], options)
                report["calibration"] = {"source": "inventory_and_supplied_robot_usd", "urdf_path": str(robot.urdf),
                    "joint_count": len(robot.calibration), "collision_spheres_path": str(robot.sphere_path),
                    "base_footprint_circumscribed_radius_m": robot.base_radius,
                    "initial_joint_limit_margin_native_units": joint_limit_margin(robot.seed, robot.limits)}
                if eligible_targets:
                    navigation = _make_navigation(robot, objects, eligible_targets, options)
                    initial_results.update({row["name"]: _screen_target(row, objects, robot, navigation, options, floor_z)
                                            for row in eligible_targets})
                    report["navigation"] = _navigation_evidence(navigation, options)
                for entry, plans in pending:
                    _screen_destination_samples(entry, plans, robot, options, floor_z)
            report["targets"] = [initial_results[row["name"]] for row in targets]
    except (AuditUnavailable, ImportError, OSError, KeyError, ValueError) as error:
        report["status"] = "unavailable"
        report["reason"] = f"{type(error).__name__}: {error}"
        report["targets"] = [_ineligible_target(row) if row.get("rigid_grasp_proxy_eligible") is False else
                             {"name": row.get("name", "unknown"), "status": "unavailable", "witness": None,
                              "policy_metadata": _policy_metadata(row), "policy_permission_inferred": False,
                              "reason": report["reason"]} for row in targets]
        if not report["destination_queries"]:
            report["destination_queries"] = [{"name": f"destination:{name}", "region_name": name, "status": "unavailable",
                "witness": None, "hypothetical_payload": None, "reason": report["reason"],
                "support": region.get("support"), "support_z": region.get("support_z"),
                "scope": _DESTINATION_SCOPE, "policy_permission_inferred": False} for name, region in regions.items()]
        else:
            for entry in report["destination_queries"]:
                if entry["status"] == "unavailable" and "reason" not in entry:
                    entry["reason"] = report["reason"]
                for sample in entry.get("sample_records", []):
                    if sample["status"] == "pending":
                        sample.update(status="unavailable", result={"status": "unavailable", "witness": None,
                            "kinematic_witness": None, "reason": report["reason"]})
                if entry.get("sample_records"):
                    entry["sample_status_counts"] = dict(Counter(sample["status"] for sample in entry["sample_records"]))
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reachability.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--navigation-mode", choices=("circle", "se2_rectangle"), default="circle")
    arguments = parser.parse_args()
    print(json.dumps(audit_scene(json.loads(arguments.inventory.read_text()), arguments.output,
                                options={"navigation_mode": arguments.navigation_mode}), indent=2))

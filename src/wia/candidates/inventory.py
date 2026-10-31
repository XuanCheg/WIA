"""Export current native geometry without actions, stepping, rendering or writes.

Run from the simulator owner thread after loading the scene::

    from wia.candidates.inventory import export_inventory
    inventory = export_inventory(session, identity="task_id", target_names=["item_01"])

CPU validation: ``python -m pytest -q tests/test_candidate_inventory.py``.
The caller owns task selection, temporal phases and persistence. Invisible or
noncolliding task records remain in ``inactive_targets`` for accounting; they
are not passed to the geometric audit as active target obstacles.
Fixed articulated rigid units remain in ``structural_targets`` while their
enabled per-link collision meshes stay active. ``robot.base_geometry`` supplies
native world vertices and base-local XY bounds only when ``available`` is true;
otherwise its reason explains why the unchanged world base AABB is retained.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np


FLOOR_CATEGORIES = frozenset({"floors", "floor", "ground", "lawn", "driveway"})
_R1_RELATIVE = 'omnigibson-robot-assets/models/r1/usd/r1.usda'
R1_USD = os.environ.get('WIA_R1_USD') or (
    str(Path(os.environ['OMNIGIBSON_DATA_PATH']).expanduser() / _R1_RELATIVE)
    if os.environ.get('OMNIGIBSON_DATA_PATH') else 'behavior://' + _R1_RELATIVE)
BASE_AABB_TOLERANCE_M = .002


def _array(value):
    return np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=float)


def _box(value):
    result = np.stack([_array(part) for part in value])
    if result.shape != (2, 3) or not np.isfinite(result).all() or np.any(result[0] > result[1]):
        raise ValueError("A finite ordered native world AABB is required.")
    return result.tolist()


def _optional_bool(obj, name):
    value = getattr(obj, name, None)
    return None if value is None else bool(value)


def _native_calibration(robot):
    from wia.runtime.sim.esi_camera_geometry import calibrated_robot_kinematics
    return calibrated_robot_kinematics(robot)[1]


def _touching_type():
    from omnigibson.object_states import Touching
    return Touching


def _kind(obj):
    value = getattr(obj, "prim_type", None)
    if hasattr(value, "name"):
        return value.name.upper()
    if isinstance(value, (int, np.integer)):
        return {0: "RIGID", 1: "CLOTH"}.get(int(value), f"UNKNOWN_{int(value)}")
    return str(value).rsplit(".", 1)[-1].upper() if value is not None else "UNKNOWN"


def _is_liquid(obj, kind):
    classes = {base.__name__ for base in type(obj).__mro__}
    return (kind in {"FLUID", "LIQUID"} or "FluidSystem" in classes
            or bool(getattr(obj, "is_liquid", False)) or bool(getattr(obj, "is_fluid", False))
            or str(getattr(obj, "category", "")).lower() in {"water", "liquid", "fluid"})


def _object_row(obj, protected):
    position, orientation = [_array(part).tolist() for part in obj.get_position_orientation()]
    kind = _kind(obj)
    joints = getattr(obj, "joints", None)
    articulated_names = None if joints is None else [str(name) for name, joint in joints.items() if joint.n_dof > 0]
    articulated = None if articulated_names is None else bool(articulated_names)
    liquid = _is_liquid(obj, kind)
    rigid = kind == "RIGID" and not liquid
    limitations = ["Current native world AABB only; grasping, holding, stable release and task success are untested."]
    if not rigid:
        limitations.append("Non-rigid, deformable or liquid geometry is a current-shape proxy; rigid-body grasp screening does not certify its behavior.")
    if articulated:
        limitations.append("The whole articulated object is measured at its current joints; handle access, joint motion and whole-container graspability are unverified.")
    if articulated is None:
        limitations.append("Native articulation metadata is unavailable; rigid grasp eligibility is unverified.")
    visible, onstage = _optional_bool(obj, "visible"), _optional_bool(obj, "onstage")
    return {"name": str(obj.name), "prim_path": str(obj.prim_path), "category": str(getattr(obj, "category", "unknown")),
        "prim_type": kind, "native_class": type(obj).__name__, "position": position, "orientation": orientation,
        "aabb": _box(obj.aabb), "fixed_base": bool(obj.fixed_base), "visible": visible, "onstage": onstage,
        "onstage_status": "native_flag" if onstage is not None else "not_inferred_from_coordinates",
        "visual_only": bool(getattr(obj, "visual_only", False)), "articulated": articulated,
        "articulated_joint_names": articulated_names, "liquid": liquid, "protected": bool(protected),
        "policy_role": "protected" if protected else "task_target", "policy_permission_inferred": False,
        "rigid_grasp_proxy_eligible": bool(rigid and articulated is False),
        "screening_limitations": limitations}


def _collision_rows(obj, skipped):
    if bool(getattr(obj, "visual_only", False)) or _optional_bool(obj, "collision_enabled") is False:
        skipped.append({"object": str(obj.name), "reason": "visual_only_or_object_collisions_disabled"})
        return []
    rows = []
    for link_name, link in sorted(getattr(obj, "links", {}).items()):
        for mesh_name, mesh in sorted(getattr(link, "collision_meshes", {}).items()):
            name = f"{obj.name}::{link_name}::{mesh_name}"
            enabled = _optional_bool(mesh, "collision_enabled")
            if bool(getattr(link, "visual_only", False)) or bool(getattr(mesh, "visual_only", False)) or enabled is not True:
                skipped.append({"object": str(obj.name), "mesh": name, "collision_enabled": enabled,
                                "reason": "visual_only_or_collision_disabled_or_unavailable"})
                continue
            rows.append({"name": name, "parent_object": str(obj.name), "link_name": str(link_name),
                "mesh_name": str(mesh_name), "prim_path": str(mesh.prim_path), "aabb": _box(mesh.aabb),
                "collision_enabled": True, "collision_geometry_kind": "native_collision_mesh_world_aabb"})
    return rows


def _base_geometry(robot, position, orientation, base_aabb):
    """Measure base vertices without inverse-projecting a rotated world AABB.

    OmniGibson GeomPrim.aabb transforms each native local mesh point using
    ``homogeneous(points) @ scaled_transform.T``. The same world vertices are
    checked against the measured base-link AABB before an inverse robot-pose
    transform supplies a local XY rectangle. Visual meshes are never silently
    substituted when this collision-only consistency check is unavailable.
    """
    link_name = str(robot.base_footprint_link_name)
    result = {"schema": "wia_robot_base_geometry_v1", "available": False, "status": "unavailable",
        "base_link_name": link_name, "provenance": "enabled_native_base_link_collision_mesh_vertices",
        "world_vertex_formula": "homogeneous(mesh.points) @ mesh.scaled_transform.T",
        "local_frame": "inverse_current_robot_position_and_xyzw_orientation",
        "measured_base_aabb": deepcopy(base_aabb), "aabb_tolerance_m": BASE_AABB_TOLERANCE_M,
        "enabled_mesh_names": [], "enabled_mesh_count": 0, "measured_mesh_count": 0,
        "vertex_count": 0, "meshes": [], "skipped_meshes": [], "measurement_errors": [],
        "fallback": "Existing measured robot.base_aabb remains unchanged; no tighter local bounds are inferred."}
    try:
        link = robot.links[robot.base_footprint_link_name]
        meshes = getattr(link, "collision_meshes", None)
        if not meshes:
            result["reason"] = "base_collision_meshes_unavailable"
            return result
        if bool(getattr(link, "visual_only", False)):
            result["reason"] = "base_link_visual_only"
            return result
        result["native_mesh_count"] = len(meshes)
        vertices = []
        for mesh_name, mesh in sorted(meshes.items()):
            name = f"{link_name}::{mesh_name}"
            enabled = _optional_bool(mesh, "collision_enabled")
            if enabled is not True or bool(getattr(mesh, "visual_only", False)):
                result["skipped_meshes"].append({"name": name, "collision_enabled": enabled,
                                                "reason": "visual_only_or_collision_disabled_or_unavailable"})
                if enabled is None:
                    result["measurement_errors"].append({"mesh": name, "reason": "collision_enabled_unavailable"})
                continue
            result["enabled_mesh_names"].append(name)
            try:
                points, transform = _array(mesh.points), _array(mesh.scaled_transform)
                if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                    raise ValueError("Native mesh points must be a nonempty finite Nx3 array.")
                if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                        or not np.allclose(transform[3], [0., 0., 0., 1.], rtol=0., atol=1e-8)):
                    raise ValueError("Native scaled_transform must be a finite affine 4x4 matrix.")
                world = (np.column_stack((points, np.ones(len(points)))) @ transform.T)[:, :3]
                if not np.isfinite(world).all():
                    raise ValueError("Transformed native world vertices must be finite.")
                vertices.append(world)
                result["meshes"].append({"name": name, "mesh_name": str(mesh_name),
                    "prim_path": str(mesh.prim_path), "collision_enabled": True, "vertex_count": len(points)})
            except Exception as error:
                result["measurement_errors"].append({"mesh": name, "reason": f"{type(error).__name__}: {error}"})
        result["enabled_mesh_count"] = len(result["enabled_mesh_names"])
        result["measured_mesh_count"] = len(vertices)
        result["vertex_count"] = sum(len(value) for value in vertices)
        if result["measurement_errors"]:
            result["reason"] = "base_collision_mesh_measurement_unavailable"
            return result
        if not vertices:
            result["reason"] = "no_enabled_base_collision_meshes"
            return result
        world = np.concatenate(vertices)
        world_aabb = np.stack((world.min(axis=0), world.max(axis=0)))
        residual = float(np.max(np.abs(world_aabb - np.asarray(base_aabb))))
        result.update(measured_collision_world_aabb=world_aabb.tolist(), max_world_aabb_residual_m=residual)
        if residual > BASE_AABB_TOLERANCE_M:
            result["reason"] = "collision_vertices_base_aabb_mismatch"
            return result
        from scipy.spatial.transform import Rotation
        position, orientation = _array(position), _array(orientation)
        if position.shape != (3,) or orientation.shape != (4,) or not np.isfinite(np.r_[position, orientation]).all():
            raise ValueError("The current robot pose must have finite position and XYZW orientation.")
        local = Rotation.from_quat(orientation).inv().apply(world - position)
        if not np.isfinite(local).all():
            raise ValueError("Base-local measured vertices must be finite.")
        result.update(available=True, status="available", world_vertices=world.tolist(),
                      base_local_xy_bounds=[local[:, :2].min(axis=0).tolist(), local[:, :2].max(axis=0).tolist()])
        return result
    except Exception as error:
        result["reason"] = "base_geometry_measurement_unavailable"
        result["measurement_errors"].append({"reason": f"{type(error).__name__}: {error}"})
        return result


def _robot_row(robot):
    measured = _array(robot.get_joint_positions()).reshape(-1)
    names = getattr(robot, "dof_names_ordered", None)
    joints = robot.joints
    if names is not None:
        names = list(names)
        if len(names) != len(measured) or len(set(names)) != len(names):
            raise ValueError("Native ordered robot encoder names do not match measured positions.")
        values = dict(zip(names, measured.tolist()))
    else:
        values, used = {}, set()
        for name, joint in joints.items():
            if joint.n_dof == 0:
                continue
            if joint.n_dof != 1:
                raise ValueError("The calibrated R1 inventory requires single-DOF robot joints.")
            indices = list(joint.dof_indices)
            if len(indices) != 1 or indices[0] in used or not 0 <= indices[0] < len(measured):
                raise ValueError("Native robot joint DOF indices are unavailable or inconsistent.")
            values[str(name)] = float(measured[indices[0]])
            used.add(indices[0])
        if used != set(range(len(measured))):
            raise ValueError("Native robot joint DOF indices do not cover all measured encoders.")
    limits = {str(name): [float(joint.lower_limit), float(joint.upper_limit)]
              for name, joint in joints.items() if joint.n_dof == 1}
    if not np.isfinite(measured).all() or any(not np.isfinite(pair).all() for pair in limits.values()):
        raise ValueError("Finite measured robot encoders and native joint limits are required.")
    position, orientation = [_array(part).tolist() for part in robot.get_position_orientation()]
    base_aabb = _box(robot.links[robot.base_footprint_link_name].aabb)
    return {"urdf_path": str(robot.urdf_path), "usd_path": str(getattr(robot, 'usd_path', R1_USD)), "position": position, "orientation": orientation,
        "joints": values, "joint_limits": limits, "joint_calibration": _native_calibration(robot),
        "base_aabb": base_aabb, "base_geometry": _base_geometry(robot, position, orientation, base_aabb)}


def _support_for(target, candidates, measured_rows, supplied):
    if supplied is not None:
        name = supplied.get(str(target.name))
        if name is None or str(name) not in measured_rows:
            return {"support": None, "support_status": "unavailable", "support_reason":
                    "Caller mapping has no loaded measurable support for this target."}
        return {"support": str(name), "support_aabb": deepcopy(measured_rows[str(name)]["aabb"]),
                "support_status": "measured_caller_mapping", "support_contact_verified": False,
                "support_evidence": "Caller mapping and current native support AABB; contact is not inferred."}
    try:
        state = target.states.get(_touching_type())
    except (ImportError, AttributeError, KeyError) as error:
        return {"support": None, "support_status": "unavailable", "support_reason": f"Native Touching state unavailable: {type(error).__name__}."}
    if state is None:
        return {"support": None, "support_status": "unavailable", "support_reason": "Native Touching state is absent."}
    box = np.asarray(measured_rows[str(target.name)]["aabb"])
    valid, errors = [], []
    for candidate in candidates:
        if candidate is target or str(candidate.name) not in measured_rows:
            continue
        other = np.asarray(measured_rows[str(candidate.name)]["aabb"])
        overlap = np.minimum(box[1, :2], other[1, :2]) - np.maximum(box[0, :2], other[0, :2])
        if np.any(overlap <= 0) or other[1, 2] > box[0, 2] + .03:
            continue
        try:
            if bool(state.get_value(candidate)):
                valid.append((abs(float(box[0, 2] - other[1, 2])), str(candidate.name)))
        except (AttributeError, RuntimeError, ValueError) as error:
            errors.append({"object": str(candidate.name), "error": f"{type(error).__name__}: {error}"})
    if not valid:
        return {"support": None, "support_status": "unavailable", "support_reason":
            "No measured touching furniture/background has XY overlap and a top at or below target bottom + 0.03 m.",
            "support_measurement_errors": errors}
    _, name = min(valid)
    return {"support": name, "support_aabb": deepcopy(measured_rows[name]["aabb"]),
            "support_status": "measured_touching_and_aabb", "support_contact_verified": True,
            "support_evidence": "Native Touching plus closest upward AABB top; contact normals and load bearing are not verified."}


def _floor_measurement(robot, collision_rows):
    base = np.asarray(robot["base_aabb"])
    candidates = []
    for row in collision_rows:
        if row["role"] != "floor":
            continue
        box = np.asarray(row["aabb"])
        overlap = np.minimum(base[1, :2], box[1, :2]) - np.maximum(base[0, :2], box[0, :2])
        if np.all(overlap > 0) and box[1, 2] <= base[0, 2] + .03:
            candidates.append((abs(float(base[0, 2] - box[1, 2])), row["name"], float(box[1, 2])))
    if not candidates:
        raise ValueError("No enabled actual floor-category collision AABB supports a measured floor_z beneath the robot.")
    _, name, height = min(candidates)
    return height, {"source": "native_floor_collision_mesh_aabb_top", "collision_row": name,
        "limitation": "A measured local candidate AABB top, not a contact-normal or raycast plane certificate; uneven floor geometry remains approximate."}


def export_inventory(session, *, identity, prototype="baseline", target_names, regions=None,
                     protected_names=(), support_by_target=None):
    """Return native audit inputs; a supplied support mapping is authoritative.

    Missing mapping entries are explicitly unavailable. Without a mapping,
    support requires actual native Touching and the upward AABB criteria.
    No simulation, rendering, setters, material edits or filesystem writes run.
    """
    robot = session.robot
    if robot is None:
        raise ValueError("A loaded native robot is required for calibrated geometric inventory.")
    target_names = {str(name) for name in target_names}
    protected_names = {str(name) for name in protected_names}
    native_objects = [obj for obj in session.env.scene.objects if obj is not robot]
    measured, meshes, skipped = {}, {}, []
    for obj in native_objects:
        measured[str(obj.name)] = _object_row(obj, str(obj.name) in protected_names)
        meshes[str(obj.name)] = _collision_rows(obj, skipped)
    objects, inactive, structural = [], [], []
    for obj in native_objects:
        name = str(obj.name)
        row, collision = measured[name], meshes[name]
        category = row["category"].lower()
        bound_floor = name in target_names and category in FLOOR_CATEGORIES
        fixed_articulated = (name in target_names and row["fixed_base"] and row["articulated"] is True
                             and row["prim_type"] == "RIGID" and not row["liquid"])
        physical_proxy = bool(collision) or (row["prim_type"] != "RIGID" and not row["visual_only"]
                          and _optional_bool(obj, "collision_enabled") is not False)
        active = not row["visual_only"] and row["visible"] is not False and row["onstage"] is not False and physical_proxy
        if bound_floor:
            row.update(role="structural_target", grasp_required=False, rigid_grasp_proxy_eligible=False,
                active_collision_mesh_count=len(collision), active_for_geometric_audit=bool(active),
                structural_reason="bound_floor", support=None, support_status="not_applicable_structural_floor")
            row["screening_limitations"].append("The task binds an actual floor category; enabled floor collision meshes remain active and this binding is never a grasp query.")
            structural.append(row)
        elif fixed_articulated:
            row.update(role="structural_target", grasp_required=False, rigid_grasp_proxy_eligible=False,
                active_collision_mesh_count=len(collision), active_for_geometric_audit=bool(collision),
                structural_reason="fixed_articulated_container")
            row["screening_limitations"].append("Fixed articulated container accounting only; enabled current per-link collision meshes are exported separately, without filling the interior with the whole-object AABB.")
            structural.append(row)
        elif name in target_names:
            row.update(role="target", grasp_required=bool(active and row["rigid_grasp_proxy_eligible"] and not row["fixed_base"]),
                       active_collision_mesh_count=len(collision), active_for_geometric_audit=bool(active))
            if active:
                objects.append(row)
                continue
            row["screening_excluded_reason"] = [reason for condition, reason in (
                (row["visual_only"], "visual_only_task_object"), (row["visible"] is False, "native_visible_false"),
                (row["onstage"] is False, "native_onstage_false"), (not physical_proxy, "no_enabled_collision_geometry")) if condition]
            row["screening_limitations"].append("Inactive task accounting only; this row must not create an active grasp query or expand navigation bounds.")
            inactive.append(row)
        role = "floor" if category in FLOOR_CATEGORIES else "background"
        for mesh in collision:
            objects.append({**mesh, "position": deepcopy(row["position"]), "orientation": deepcopy(row["orientation"]),
                "fixed_base": row["fixed_base"], "role": role, "category": row["category"], "prim_type": row["prim_type"],
                "articulated": row["articulated"], "visible": row["visible"], "onstage": row["onstage"],
                "visual_only": False, "grasp_required": False, "screening_limitations": deepcopy(row["screening_limitations"]),
                **({"protected": row["protected"], "policy_role": row["policy_role"],
                    "policy_permission_inferred": row["policy_permission_inferred"]} if fixed_articulated else {})})
        if not collision and physical_proxy and not row["visual_only"]:
            objects.append({**deepcopy(row), "role": "background", "grasp_required": False,
                "collision_geometry_kind": "nonrigid_current_aabb_proxy", "collision_state": "rigid_mesh_representation_unavailable"})
    candidates = [obj for obj in native_objects if not bool(getattr(obj, "visual_only", False))
                  and meshes[str(obj.name)] and (str(obj.name) not in target_names or bool(obj.fixed_base)
                                                or measured[str(obj.name)]["category"].lower() in FLOOR_CATEGORIES)]
    for obj in native_objects:
        if str(obj.name) in target_names and measured[str(obj.name)]["category"].lower() not in FLOOR_CATEGORIES:
            measured[str(obj.name)].update(_support_for(obj, candidates, measured, support_by_target))
    unavailable = []
    for name in sorted(target_names - set(measured)):
        system_registry = getattr(session.env.scene, "system_registry", None)
        system = system_registry("name", name) if callable(system_registry) else None
        unavailable.append({"name": name, "prim_type": "LIQUID_SYSTEM" if system is not None and _is_liquid(system, "SYSTEM") else "UNAVAILABLE",
            "native_class": type(system).__name__ if system is not None else None, "grasp_required": False,
            "protected": name in protected_names, "support": None, "support_status": "unavailable",
            "policy_permission_inferred": False,
            "reason": "No loaded native object AABB; particle systems and absent objects cannot supply a rigid grasp target.",
            "screening_limitations": ["No current rigid-body grasp geometry is exported for this requested target."]})
    robot_row = _robot_row(robot)
    floor_z, floor_evidence = _floor_measurement(robot_row, objects)
    return {"schema": "wia_candidate_native_inventory_v1", "id": str(identity), "prototype": str(prototype),
        "floor_z": floor_z, "floor_measurement": floor_evidence, "robot": robot_row, "objects": objects,
        "regions": deepcopy(regions or {}), "requested_target_names": sorted(target_names),
        "inactive_targets": inactive, "unavailable_targets": unavailable, "structural_targets": structural, "read_only": True,
        "skipped_geometry": skipped, "validation_scope": "Measured current native geometry only; no robot commands, scene operations, physics stepping or rendering executed."}

"""Measure one native container's sampled opening geometry during initialization.

Use inside an already running native owner-thread session::

    envelope = measure_container_opening_envelope(session, container)

Syntax check only: python -m py_compile src/wia/candidates/container_envelope.py

This authored initialization probe moves only the supplied fixed-base unit to
world [0, 0, 20] at identity orientation, samples its native hinge and collision
meshes, and restores its root/joints in finally. Native global simulation steps
still advance other actors naturally. No robot control, opening success,
continuous swept-volume certificate, or policy permission is established.
The caller owns placement screening, rotated/translated envelope use, and floor
exclusion. The returned Z bounds remain measured; only XY receives padding.
"""

from __future__ import annotations

import math


SAMPLE_COUNT = 17
XY_PADDING_M = 0.04
JOINT_READBACK_TOLERANCE_RAD = 1e-4


def _values(value, count, label):
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{label} requires {count} native finite values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains nonfinite native values")
    return result


def _box(value):
    low = _values(value[0], 3, "Native collision AABB lower bound")
    high = _values(value[1], 3, "Native collision AABB upper bound")
    if any(a > b for a, b in zip(low, high)):
        raise ValueError("Native collision AABB is inverted")
    return [low, high]


def _relative_meshes(obj, origin):
    # Reuse the read-only inventory filter: enabled native collision meshes
    # only, with no visual geometry or object-AABB fallback.
    from wia.candidates.inventory import _collision_rows

    skipped = []
    rows = _collision_rows(obj, skipped)
    if not rows:
        raise ValueError("Container has no enabled native collision-mesh AABBs")
    boxes = []
    for row in rows:
        world_box = _box(row["aabb"])
        boxes.append({"name": row["name"], "link_name": row["link_name"],
                      "mesh_name": row["mesh_name"], "prim_path": row["prim_path"],
                      "relative_aabb_m": [[point[i] - origin[i] for i in range(3)]
                                          for point in world_box]})
    return boxes, skipped


def measure_container_opening_envelope(session, obj):
    """Return a padded sampled hinge envelope relative to its closed geometry.

    The reference origin is the identity-oriented closed object's world-AABB XY
    center and bottom Z. Exactly one native articulated, limited revolute DOF
    with an explicit X/Y/Z joint axis is supported. Its saved position must be
    natively closed; the fully-open endpoint comes from Open.set_value itself.
    Seventeen inclusive evenly spaced positions each receive one native step
    before enabled collision-mesh AABBs are read. Unsupported assumptions raise.
    """
    import torch
    from omnigibson.object_states import Open
    from omnigibson.utils.constants import JointType

    sim = session.pipeline.og.sim
    if not sim.is_playing():
        raise ValueError("Container envelope measurement requires a running native simulator")
    if not bool(obj.fixed_base) or bool(obj.visual_only):
        raise ValueError("Opening envelope requires a colliding fixed-base articulated unit")
    joints = list(obj.joints.items())
    if len(joints) != 1 or obj.n_dof != 1:
        raise ValueError("Opening envelope supports exactly one articulated revolute DOF")
    joint_name, joint = joints[0]
    if (joint.joint_type != JointType.JOINT_REVOLUTE or joint.n_dof != 1
            or not joint.articulated or joint.dof_indices != [0] or not bool(joint.has_limit)):
        raise ValueError("Container hinge is not one finite limited native revolute DOF")
    axis = joint.axis
    if axis not in ("X", "Y", "Z"):
        raise ValueError("Container hinge lacks an explicit supported native joint axis")
    lower, upper = float(joint.lower_limit), float(joint.upper_limit)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Container hinge limits are not finite and ordered")
    open_state = obj.states.get(Open)
    if open_state is None or bool(open_state.get_value()):
        raise ValueError("Container must supply native Open state and begin closed")
    saved_position, saved_orientation = [torch.as_tensor(value).detach().clone()
                                         for value in obj.get_position_orientation()]
    saved_joints = torch.as_tensor(obj.get_joint_positions()).detach().clone()
    _values(saved_position, 3, "Saved native root position")
    quaternion = _values(saved_orientation, 4, "Saved native root orientation")
    if not math.isclose(sum(value * value for value in quaternion), 1., abs_tol=1e-3):
        raise ValueError("Saved native root orientation is not a unit XYZW quaternion")
    closed_q = _values(saved_joints, 1, "Saved closed joint position")[0]
    if not lower - JOINT_READBACK_TOLERANCE_RAD <= closed_q <= upper + JOINT_READBACK_TOLERANCE_RAD:
        raise ValueError("Saved closed hinge position is outside native limits")
    remote_position = saved_position.new_tensor([0., 0., 20.])
    remote_orientation = saved_orientation.new_tensor([0., 0., 0., 1.])
    native_steps = 0
    result = None
    try:
        obj.set_position_orientation(position=remote_position, orientation=remote_orientation)
        obj.set_joint_positions(positions=saved_joints, drive=False)
        obj.keep_still()
        sim.step()
        native_steps += 1
        closed_readback = _values(obj.get_joint_positions(), 1, "Native closed hinge readback")[0]
        if abs(closed_readback - closed_q) > JOINT_READBACK_TOLERANCE_RAD:
            raise ValueError("Native hinge drifted before its closed geometry was measured")
        closed_box = _box(obj.aabb)
        dimensions = [closed_box[1][i] - closed_box[0][i] for i in range(3)]
        if any(value <= 0 for value in dimensions):
            raise ValueError("Closed native container dimensions must be positive")
        origin = [(closed_box[0][i] + closed_box[1][i]) / 2 for i in (0, 1)] + [closed_box[0][2]]
        closed_boxes, closed_skipped = _relative_meshes(obj, origin)
        if open_state.set_value(True, fully=True) is not True:
            raise ValueError("Native Open setter did not supply a fully-open endpoint")
        full_q = _values(obj.get_joint_positions(), 1, "Native fully-open endpoint")[0]
        if (not lower - JOINT_READBACK_TOLERANCE_RAD <= full_q <= upper + JOINT_READBACK_TOLERANCE_RAD
                or abs(full_q - closed_q) <= JOINT_READBACK_TOLERANCE_RAD):
            raise ValueError("Native fully-open endpoint is degenerate or outside the hinge limits")
        q_values = [closed_q + (full_q - closed_q) * index / (SAMPLE_COUNT - 1)
                    for index in range(SAMPLE_COUNT)]
        samples, all_boxes = [], list(closed_boxes)
        for q_value in q_values:
            obj.set_joint_positions(positions=saved_joints.new_tensor([q_value]), drive=False)
            obj.keep_still()
            sim.step()
            native_steps += 1
            measured_q = _values(obj.get_joint_positions(), 1, "Sampled native hinge readback")[0]
            if abs(measured_q - q_value) > JOINT_READBACK_TOLERANCE_RAD:
                raise ValueError("Native hinge drifted from the requested evenly spaced sample")
            boxes, skipped = _relative_meshes(obj, origin)
            samples.append({"q_rad": q_value, "measured_q_rad": measured_q,
                            "collision_boxes": boxes, "skipped_collision_geometry": skipped})
            all_boxes.extend(boxes)
        raw = [[min(box["relative_aabb_m"][0][i] for box in all_boxes) for i in range(3)],
               [max(box["relative_aabb_m"][1][i] for box in all_boxes) for i in range(3)]]
        padded = [raw[0].copy(), raw[1].copy()]
        for axis_index in (0, 1):
            padded[0][axis_index] -= XY_PADDING_M
            padded[1][axis_index] += XY_PADDING_M
        result = {"schema": "wia_candidate_container_opening_envelope_v1", "object": str(obj.name),
            "relative_aabb_m": padded, "raw_relative_aabb_m": raw, "closed_dimensions_m": dimensions,
            "reference": {"root_position_world_m": [0., 0., 20.], "root_orientation_xyzw": [0., 0., 0., 1.],
                          "closed_aabb_world_m": closed_box, "origin_world_m": origin,
                          "definition": "Closed world AABB XY center and closed bottom Z at identity root orientation"},
            "joint": {"name": str(joint_name), "type": str(joint.joint_type), "axis": str(axis),
                      "limits_rad": [lower, upper], "closed_q_rad": closed_q,
                      "closed_measured_q_rad": closed_readback, "fully_open_q_rad": full_q},
            "sample_q_values_rad": q_values, "sample_count": len(samples), "samples": samples,
            "closed_collision_boxes": closed_boxes, "closed_skipped_collision_geometry": closed_skipped,
            "xy_padding_m": XY_PADDING_M, "z_padding_m": 0.,
            "joint_readback_tolerance_rad": JOINT_READBACK_TOLERANCE_RAD,
            "robot_actions": 0, "model_calls": 0,
            "scope": "Sampled native authored joint collision-mesh geometry including closed geometry; not a continuous sweep or robot opening-success certificate."}
    finally:
        # Sync the restored state once, then reassert the saved pose/joints so
        # passive drift during that global step is not left as an authored edit.
        try:
            obj.set_joint_positions(positions=saved_joints, drive=False)
        finally:
            obj.set_position_orientation(position=saved_position, orientation=saved_orientation)
            obj.keep_still()
            try:
                sim.step()
                native_steps += 1
            finally:
                obj.set_position_orientation(position=saved_position, orientation=saved_orientation)
                obj.set_joint_positions(positions=saved_joints, drive=False)
                obj.keep_still()
    result["native_simulation_steps"] = native_steps
    return result

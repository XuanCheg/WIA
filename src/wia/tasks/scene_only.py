"""Scene-only initialization and physical preflight, without a task evaluator.

Run through scripts/run_esi_robot.py --scene-file FILE --variant tabletop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import QualificationResult, ScenePatches


def horizontal_box_overlap(centre, quaternion, extent, other_low, other_high, tolerance=.003):
    """Conservative yaw-oriented cube versus furniture AABB separation test."""
    rotation = Rotation.from_quat(quaternion).as_matrix()[:2, :2]
    delta = (other_low[:2] + other_high[:2]) / 2 - centre[:2]
    half_other = (other_high[:2] - other_low[:2]) / 2
    for axis in (np.array([1., 0.]), np.array([0., 1.]), rotation[:, 0], rotation[:, 1]):
        radius = np.abs(rotation.T @ axis) @ (extent[:2] / 2) + np.abs(axis) @ half_other
        if radius - abs(delta @ axis) <= tolerance:
            return False
    return True


def local_support_surface(floor_path, hits):
    """Select the nearest upward-facing hit belonging to a bound support."""
    floor_hits = [hit for hit in hits if hit.get("hit")
                  and (hit["rigidBody"] == floor_path or hit["rigidBody"].startswith(floor_path + "/"))
                  and float(hit["normal"][2]) > .5]
    if not floor_hits:
        raise ValueError("No local upward-facing support surface was found.")
    return min(floor_hits, key=lambda hit: float(hit["distance"]))


def local_floor_surface(floor_path, hits):
    """Select the bound floor surface beneath a table leg."""
    try:
        return local_support_surface(floor_path, hits)
    except ValueError as error:
        raise ValueError("No local upward-facing floor surface was found beneath a table leg.") from error


class SceneOnly(ScenePatches):
    def __init__(self, scene):
        if (scene.definition.target is None or scene.definition.qualification is None) and scene.run_mode != "diagnostic":
            raise ValueError("Diagnostic scenes require the capture Worker; no physical qualification is declared.")
        super().__init__(scene, scene.definition.bindings)
        self._settings["camera_size"] = list(scene.definition.runtime.camera_size)
        self.post_controller_settle_steps = 0

    def prepare_after_controller_initialization(self, session):
        steps = self.task.definition.runtime.post_controller_settle_steps
        for _ in range(steps):
            session.pipeline.og.sim.step()
        self.post_controller_settle_steps = steps
        if steps:
            print(f"[scene] Post-controller settling: {steps} native steps", flush=True)

    def apply_after_postprocessing(self, session):
        super().apply_after_postprocessing(session)
        for item in self.applied:
            if item["operation"] == "object_pose" and item["phase"] == "after_postprocessing":
                session.env.scene.object_registry("name", self._binding(item["binding"]).name).keep_still()
        if self.applied and self.task.definition.qualification is not None:
            for _ in range(self.task.definition.qualification.settle_steps):
                session.pipeline.og.sim.step()

    def qualify(self, session, executor, output, *, observation=None):
        from wia.native.utils import _array

        checks, reason = {}, "scene_precondition_failed"
        definition, config = self.task.definition, self.task.definition.qualification
        if config is None and self.task.run_mode == 'diagnostic':
            runnable = session.robot is not None and executor is not None
            result = QualificationResult('diagnostic_unverified' if runnable else 'skipped',
                None if runnable else 'native_robot_or_control_unavailable',
                {'native_environment_available': runnable, 'initialization': 'unverified',
                 'scoring': 'unavailable', 'review_status': definition.status, 'run_mode': 'diagnostic'})
            result.save(output, patches=self.applied)
            return result
        if definition.runtime.post_controller_settle_steps:
            checks["post_controller_settle_steps"] = self.post_controller_settle_steps
        try:
            self.task.verify_unchanged()
            if (session.scene_name, session.room_name) != (self.task.scene_name, self.task.room_name):
                reason = "source_scene_mismatch"
                raise ValueError("Loaded scene/room does not match the source.")
            bound = {}
            keys = ((definition.target,) if not self.applied else tuple(dict.fromkeys(
                config.supported_objects + (config.support,) + tuple(config.object_supports.values()))))
            for key in keys:
                binding = self._binding(key)
                obj = session.env.scene.object_registry("name", binding.name)
                if obj is None or any(getattr(binding, field) is not None and
                                      getattr(obj, field, None) != getattr(binding, field)
                                      for field in ("category", "model")):
                    reason = "object_binding_mismatch"
                    raise ValueError("A scene object does not match its binding.")
                bound[key] = obj
            target = bound[definition.target]
            if target.fixed_base or target.visual_only:
                reason = "target_not_movable_collidable"
                raise ValueError("The source target must remain movable and collidable.")
            checks["target_movable_collidable"] = True
            if self.applied:
                support = bound[config.support]
                if not support.fixed_base or support.visual_only:
                    raise ValueError("The tabletop must be fixed collidable geometry.")
                primary_support = support
                primary_support_high = _array(support.aabb[1])
                support_rows = {}
                for support_key in dict.fromkeys((config.support,) + tuple(config.object_supports.values())):
                    support_obj = bound[support_key]
                    # Fixed primitives use RigidKinematicPrim, which has no rigid-body velocity API.
                    linear = 0. if support_obj.fixed_base else float(np.linalg.norm(_array(support_obj.get_linear_velocity())))
                    angular = 0. if support_obj.fixed_base else float(np.linalg.norm(_array(support_obj.get_angular_velocity())))
                    stable = (not support_obj.visual_only and linear <= config.maximum_linear_speed_m_s
                              and angular <= config.maximum_angular_speed_rad_s)
                    support_rows[support_key] = {"name": support_obj.name, "fixed_base": support_obj.fixed_base,
                        "linear_speed_m_s": linear, "angular_speed_rad_s": angular, "passed": stable}
                checks["supports"] = support_rows
                if not all(row["passed"] for row in support_rows.values()):
                    reason = "support_stability_failed"
                    raise ValueError("A support must be collidable and stable before the episode.")
                rows = {}
                for key in config.supported_objects:
                    support_key = config.object_supports.get(key, config.support)
                    support = bound[support_key]
                    support_low, support_high = (_array(value) for value in support.aabb)
                    support_centre, support_quat, support_extent, _ = (
                        _array(value) for value in support.get_base_aligned_bbox())
                    support_rotation = Rotation.from_quat(support_quat)
                    obj = bound[key]
                    low, high = (_array(value) for value in obj.aabb)
                    contacts = [{"body0": contact.body0, "body1": contact.body1}
                                for contact in obj.contact_list()]
                    contact = any(support.prim_path + "/" in row[side] for row in contacts
                                  for side in ("body0", "body1"))
                    speed = float(np.linalg.norm(_array(obj.get_linear_velocity())))
                    angular_speed = float(np.linalg.norm(_array(obj.get_angular_velocity())))
                    points = [_array(link.collision_boundary_points_world) for link in obj.links.values()
                              if link.collision_boundary_points_world is not None]
                    if not points:
                        raise ValueError("A task object has no native collision boundary points.")
                    local_points = support_rotation.inv().apply(np.concatenate(points) - support_centre)
                    xy_inside = bool(np.all(np.min(local_points[:, :2], axis=0) >= -support_extent[:2] / 2 - .002)
                                     and np.all(np.max(local_points[:, :2], axis=0) <= support_extent[:2] / 2 + .002))
                    surface_z = float(support_high[2])
                    if config.support_surface_mode == "local_raycast":
                        from omnigibson.utils.sampling_utils import raytest

                        xy = ((low + high) / 2)[:2].tolist()
                        hits = raytest([*xy, float(low[2]) + config.support_tolerance_m],
                            [*xy, float(low[2]) - .10], only_closest=False)
                        surface_z = float(_array(local_support_surface(support.prim_path, hits)["position"])[2])
                    gap = float(low[2] - surface_z)
                    passed = (xy_inside and abs(gap) <= config.support_tolerance_m
                              and speed <= config.maximum_linear_speed_m_s
                              and angular_speed <= config.maximum_angular_speed_rad_s
                              and (contact or not config.require_support_contact))
                    rows[key] = {"name": obj.name, "support_binding": support_key, "support_name": support.name,
                        "support_surface_mode": config.support_surface_mode,
                        "support_aabb_max": support_high.tolist(), "aabb_min": low.tolist(), "aabb_max": high.tolist(),
                        "support_surface_z_m": surface_z,
                        "bottom_minus_support_top_m": gap, "linear_speed_m_s": speed,
                        "angular_speed_rad_s": angular_speed,
                        "support_footprint_screen": xy_inside, "support_contact": contact,
                        "contacts": contacts, "passed": passed}
                checks["supported_objects"] = rows
                if not all(row["passed"] for row in rows.values()):
                    reason = "object_support_or_stability_failed"
                    raise ValueError("At least one task object failed physical support/stability preflight.")
                primitives = [session.env.scene.object_registry("name", item["name"])
                              for item in self.applied if item["operation"] == "add_primitive"]
                if any(obj is None or obj.visual_only or not obj.fixed_base for obj in primitives):
                    reason = "primitive_geometry_failed"
                    raise ValueError("Table components must remain fixed and collidable.")
                floor = session.env.scene.object_registry("name", session.payload.get("floor_name"))
                if floor is None:
                    reason = "source_floor_unavailable"
                    raise ValueError("The source floor binding must resolve for table qualification.")
                from omnigibson.utils.sampling_utils import raytest

                floor_low, floor_high = (_array(value) for value in floor.aabb)
                checks["floor_aabb_max_z_m"] = float(floor_high[2])
                floor_gaps, floor_surfaces = {}, {}
                support_names = {bound[key].name for key in (config.support, *config.object_supports.values())}
                for leg in primitives:
                    if leg.name in support_names:
                        continue
                    low, high = (_array(value) for value in leg.aabb)
                    xy = ((low + high) / 2)[:2].tolist()
                    reason = "local_floor_surface_unavailable"
                    hit = local_floor_surface(floor.prim_path, raytest(
                        xy + [float(floor_high[2]) + .1], xy + [float(floor_low[2]) - .1],
                        only_closest=False))
                    floor_top = float(hit["position"][2])
                    floor_gaps[leg.name] = float(low[2] - floor_top)
                    floor_surfaces[leg.name] = {"position": _array(hit["position"]).tolist(),
                                               "normal": _array(hit["normal"]).tolist(),
                                               "rigid_body": hit["rigidBody"]}
                checks["table_leg_floor_gaps_m"] = floor_gaps
                checks["table_leg_local_floor_surfaces"] = floor_surfaces
                if any(abs(gap) > config.support_tolerance_m for gap in floor_gaps.values()):
                    reason = "table_floor_alignment_failed"
                    raise ValueError("The fixed table legs must meet the actual source floor.")
                robot_contacts = [{"body0": contact.body0, "body1": contact.body1}
                                  for contact in session.robot.contact_list()
                                  if not any("/floors_" in value or "groundPlane" in value
                                             for value in (contact.body0, contact.body1))]
                checks["robot_nonfloor_contacts"] = robot_contacts
                if config.reject_robot_contacts and robot_contacts:
                    reason = "robot_initial_contact"
                    raise ValueError("The robot has an interfering initial contact.")
                if config.screen_furniture_overlap:
                    excluded = ({bound[key].name for key in config.supported_objects}
                                | {primary_support.name} | {obj.name for obj in primitives})
                    obstacles = []
                    for obj in session.env.scene.objects:
                        if (obj is session.robot or obj.name in excluded or obj.visual_only
                                or obj.category in ("floors", "floor", "ceilings", "ceiling")):
                            continue
                        low, high = (_array(value) for value in obj.aabb)
                        if low[2] < primary_support_high[2] and high[2] > .003:
                            obstacles.append((obj.name, low, high))
                    overlaps = []
                    for primitive in primitives:
                        low, high = (_array(value) for value in primitive.aabb)
                        centre, quaternion, extent, _ = (_array(value) for value in primitive.get_base_aligned_bbox())
                        for name, other_low, other_high in obstacles:
                            penetration = np.minimum(high, other_high) - np.maximum(low, other_low)
                            if (np.all(penetration > .003) and horizontal_box_overlap(
                                    centre, quaternion, extent, other_low, other_high)):
                                overlaps.append({"component": primitive.name, "obstacle": name,
                                                 "aabb_overlap_m": penetration.tolist()})
                    checks["furniture_aabb_overlap_candidates"] = overlaps
                    if overlaps:
                        reason = "possible_table_furniture_interference"
                        raise ValueError("The conservative furniture overlap screen rejects this table placement.")
            result = QualificationResult("accepted_minimal_patch" if self.applied else "accepted_original", None, checks)
        except (ValueError, RuntimeError) as error:
            checks["private_error"] = str(error)
            result = QualificationResult("skipped", reason, checks)
        result.save(output, patches=self.applied)
        (Path(output) / "scene_manifest.json").write_text(json.dumps({
            "kind": "scene_patch", "scene_id": definition.id, "variant": self.task.variant,
            "source_path": str(self.task.source_path), "scene_file": str(self.task.path),
            "source_sha256": self.task.source_sha256, "scene_sha256": self.task.file_sha256,
            "source_question_unchanged": True, "new_evaluator": False,
            "qualification_status": result.status, "qualification_reason": result.reason}, indent=2) + "\n")
        return result

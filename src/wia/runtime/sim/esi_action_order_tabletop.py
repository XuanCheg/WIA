"""Move an original Action Order row onto an existing native table at startup.

Run in the simulation environment with src/wia/runtime/commands/run_esi_robot.py --web
--action-order-tabletop-config configs/esi_action_order_tabletop.json and the
matching configs/esi_action_order_tabletop_preview.json preview preset.
This adapter changes poses only before human or Agent actions are enabled.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def transform_points(points, source_xy, destination_xy, yaw_rad, height_offset):
    """Apply one rigid world transform, preserving distances and camera bearings."""
    points = np.asarray(points, dtype=float)
    anchor = np.r_[source_xy, 0.]
    destination = np.r_[destination_xy, height_offset]
    return Rotation.from_euler("z", yaw_rad).apply(points - anchor) + destination


class ActionOrderTabletop:
    """Optional initialization adapter; original QA and source metadata stay intact."""

    def __init__(self, config_path, output):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        self.output = Path(output) / "private_tabletop"
        self.output.mkdir(parents=True, exist_ok=True)
        # No declarative patches: the native table determines the rigid transform.
        # Keep the transformed ESI reference rather than the generic patched-view framing.
        self.variant = SimpleNamespace(patches=())
        self.report = {"schema": "native_action_order_tabletop_v1", "status": "pending",
                       "config_path": str(self.config_path), "model_calls": 0,
                       "task_actions": 0, "object_scale_multiplier": 1.,
                       "source_question_unchanged": True, "source_scorer_unchanged": True}

    def save(self):
        (self.output / "initialization.json").write_text(
            json.dumps(self.report, indent=2, allow_nan=False) + "\n")

    def apply_to_env_config(self, config):
        """Reuse upstream construction without adding furniture or changing scales."""

    @staticmethod
    def state(obj):
        from wia.runtime.sim.esi_omnigibson import _array

        position, quaternion = obj.get_position_orientation()
        low, high = (_array(value) for value in obj.aabb)
        return {"name": obj.name, "category": obj.category, "model": getattr(obj, "model", None),
                "position": _array(position).tolist(), "quaternion_xyzw": _array(quaternion).tolist(),
                "scale": _array(obj.scale).tolist(), "aabb_min": low.tolist(), "aabb_max": high.tolist(),
                "fixed_base": obj.fixed_base, "visual_only": obj.visual_only}

    def surface(self, table, xy):
        from omnigibson.utils.sampling_utils import raytest
        from wia.tasks.scene_only import local_support_surface
        from wia.runtime.sim.esi_omnigibson import _array

        top = float(_array(table.aabb[1])[2])
        hits = raytest([*xy, top + .1], [*xy, top - .12], only_closest=False)
        return float(local_support_surface(table.prim_path, hits)["position"][2])

    def apply_after_postprocessing(self, session):
        from wia.runtime.sim.esi_omnigibson import _array
        import torch

        try:
            if session.task_module.TASK_NAME != "action_order_inference":
                raise ValueError("The tabletop adapter requires Action Order Inference.")
            if (session.scene_name, session.room_name) != (self.config["scene"], self.config["room"]):
                raise ValueError("The tabletop config belongs to another scene/room.")
            table = session.env.scene.object_registry("name", self.config["table_name"])
            if table is None or table.visual_only or session.room_name not in table.in_rooms:
                raise ValueError("The existing collidable table must be in the source room.")
            self.table = table
            self.objects = {name: session.env.scene.object_registry("name", name)
                            for name in session.task_object_names}
            if len(self.objects) != 5 or any(obj is None for obj in self.objects.values()):
                raise ValueError("The original five task objects must resolve.")
            before = {name: self.state(obj) for name, obj in self.objects.items()}
            self.report["support_before"] = self.state(table)
            self.report["objects_before"] = before
            centre, quaternion, extent, _ = (_array(value) for value in table.get_base_aligned_bbox())
            table_rotation = Rotation.from_quat(quaternion)
            major = int(np.argmax(extent[:2]))
            axis = table_rotation.as_matrix()[:2, major]
            axis /= np.linalg.norm(axis)
            source_axis = np.asarray(session.payload["row_layout"]["axis"], float)
            if np.dot(axis, source_axis) < 0:
                axis = -axis
            yaw = float(np.arctan2(axis[1], axis[0]) - np.arctan2(source_axis[1], source_axis[0]))
            rotation = Rotation.from_euler("z", yaw)
            source_xy = np.asarray(session.payload["row_layout"]["row_midpoint"][:2], float)
            destination_xy = centre[:2]
            rays = {}
            for name, row in before.items():
                low, high = np.array(row["aabb_min"]), np.array(row["aabb_max"])
                margin = self.config["edge_margin_m"]
                points = [[x, y, 0.] for x in (low[0] - margin, high[0] + margin)
                          for y in (low[1] - margin, high[1] + margin)]
                points.append([*((low + high) / 2)[:2], 0.])
                transformed = transform_points(points, source_xy, destination_xy, yaw, 0.)
                rays[name] = [{"xy": point[:2].tolist(), "surface_z_m": self.surface(table, point[:2])}
                              for point in transformed]
            heights = [ray["surface_z_m"] for values in rays.values() for ray in values]
            if np.ptp(heights) > self.config["surface_flatness_tolerance_m"]:
                raise ValueError("The row requires one continuous level tabletop.")
            height = float(np.median(heights) - np.median([row["aabb_min"][2] for row in before.values()])
                           + self.config["placement_clearance_m"])
            self.report.update(surface_rays=rays, tabletop_world_z_m=float(np.median(heights)),
                transform={"source_xy": source_xy.tolist(), "destination_xy": destination_xy.tolist(),
                           "yaw_rad": yaw, "height_offset_m": height})
            for name, obj in self.objects.items():
                row = before[name]
                position = transform_points(row["position"], source_xy, destination_xy, yaw, height)
                orientation = (rotation * Rotation.from_quat(row["quaternion_xyzw"])).as_quat()
                obj.set_position_orientation(position=torch.as_tensor(position, dtype=torch.float32),
                                             orientation=torch.as_tensor(orientation, dtype=torch.float32))
                obj.keep_still()
            self.report["reference_camera_before"] = {
                "position": np.asarray(session.scene_camera_position).tolist(),
                "quaternion_xyzw": np.asarray(session.scene_camera_quaternion).tolist()}
            session.scene_camera_position = transform_points(
                session.scene_camera_position, source_xy, destination_xy, yaw, height)
            session.scene_camera_quaternion = (
                rotation * Rotation.from_quat(session.scene_camera_quaternion)).as_quat()
            session.scene_camera_framing = "action_order_rigid_tabletop_reference"
            self.report["reference_camera_after"] = {
                "position": session.scene_camera_position.tolist(),
                "quaternion_xyzw": session.scene_camera_quaternion.tolist()}
            self.report["reference_camera_transform_matches_objects"] = True
            for _ in range(self.config["settle_steps"]):
                session.pipeline.og.sim.step()
            self.report["status"] = "placed_pending_qualification"
            self.save()
        except Exception as error:
            self.report.update(status="failed", error=str(error))
            self.save()
            raise

    def qualify(self, session):
        """Require real support, unchanged scales and stable released objects."""
        from wia.runtime.sim.esi_omnigibson import _array

        try:
            checks = {}
            for name, obj in self.objects.items():
                row = self.state(obj)
                before = self.report["objects_before"][name]
                low, high = np.array(row["aabb_min"]), np.array(row["aabb_max"])
                surface_z = self.surface(self.table, ((low + high) / 2)[:2])
                contacts = [{"body0": item.body0, "body1": item.body1} for item in obj.contact_list()]
                supported = any(value == self.table.prim_path or value.startswith(self.table.prim_path + "/")
                                for item in contacts for value in item.values())
                linear = float(np.linalg.norm(_array(obj.get_linear_velocity())))
                angular = float(np.linalg.norm(_array(obj.get_angular_velocity())))
                scales_match = bool(np.array_equal(np.array(row["scale"]), np.array(before["scale"])))
                gap = float(low[2] - surface_z)
                passed = (supported and scales_match and not row["visual_only"] and not row["fixed_base"]
                          and abs(gap) <= self.config["support_tolerance_m"]
                          and linear <= self.config["maximum_linear_speed_m_s"]
                          and angular <= self.config["maximum_angular_speed_rad_s"])
                checks[name] = {"object": row, "table_contact": supported, "contacts": contacts,
                               "scale_unchanged": scales_match, "bottom_minus_surface_m": gap,
                               "linear_speed_m_s": linear, "angular_speed_rad_s": angular, "passed": passed}
            support_linear = float(np.linalg.norm(_array(self.table.get_linear_velocity())))
            support_angular = float(np.linalg.norm(_array(self.table.get_angular_velocity())))
            initial_xy = np.array([row["position"][:2] for row in self.report["objects_before"].values()])
            final_xy = np.array([row["object"]["position"][:2] for row in checks.values()])
            initial_distances = np.linalg.norm(initial_xy[:, None] - initial_xy[None, :], axis=-1)
            final_distances = np.linalg.norm(final_xy[:, None] - final_xy[None, :], axis=-1)
            robot_contacts = [{"body0": c.body0, "body1": c.body1} for c in session.robot.contact_list()
                              if not any("/floors_" in value or "groundPlane" in value
                                         for value in (c.body0, c.body1))]
            self.report.update(qualification=checks, support_linear_speed_m_s=support_linear,
                               support_angular_speed_rad_s=support_angular,
                               robot_nonfloor_contacts=robot_contacts,
                               settled_pairwise_xy_distance_max_error_m=float(
                                   np.max(np.abs(initial_distances - final_distances))))
            if (not all(row["passed"] for row in checks.values()) or robot_contacts
                    or support_linear > self.config["maximum_linear_speed_m_s"]
                    or support_angular > self.config["maximum_angular_speed_rad_s"]):
                raise ValueError("The tabletop layout failed native support/stability or robot contact checks.")
            self.report["status"] = "qualified_initialization"
            self.save()
            print("[action-order-tabletop] all five unchanged objects have stable native table support", flush=True)
        except Exception as error:
            self.report.update(status="failed", error=str(error))
            self.save()
            raise

"""Check read-only inventory measurements. Run: python -m pytest -q tests/test_candidate_inventory.py."""
from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import wia.candidates.inventory as inventory
from wia.candidates.reachability import _initial_targets


class Touching:
    pass


class ContactState:
    def __init__(self, names=()):
        self.names = set(names)

    def get_value(self, other):
        return other.name in self.names


def mesh(name, box, *, enabled=True, visual=False):
    return SimpleNamespace(prim_path="/meshes/" + name, aabb=deepcopy(box), collision_enabled=enabled, visual_only=visual)


class NativeObject:
    def __init__(self, name, box, *, category="object", fixed=False, kind="RIGID", visible=True,
                 visual=False, contacts=(), colliding=True, joints=None):
        self.name, self.category, self.fixed_base, self.prim_type = name, category, fixed, kind
        self.aabb, self.prim_path = deepcopy(box), "/objects/" + name
        self.visible, self.visual_only = visible, visual
        self.position, self.orientation = np.mean(box, axis=0).tolist(), [0., 0., 0., 1.]
        self.links = {"base": SimpleNamespace(visual_only=False,
            collision_meshes={"body": mesh(name, box, enabled=colliding)})}
        self.joints = {} if joints is None else joints
        self.states = {Touching: ContactState(contacts)}

    def get_position_orientation(self):
        return np.array(self.position), np.array(self.orientation)

    def set_position_orientation(self, *args, **kwargs):
        raise AssertionError("Inventory must never set a native pose.")


class ForbiddenSimulation:
    def __getattr__(self, name):
        raise AssertionError("Inventory must never access simulator actions or rendering.")


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(inventory, "_touching_type", lambda: Touching)
    calibration = {"joint_a": {"parent": "base_link", "child": "arm", "parent_joint": np.eye(4).tolist(),
                               "child_joint": np.eye(4).tolist(), "axis": [0., 0., 1.]}}
    monkeypatch.setattr(inventory, "_native_calibration", lambda robot: deepcopy(calibration))
    joints = {"joint_b": SimpleNamespace(n_dof=1, dof_indices=[1], lower_limit=-2., upper_limit=2.),
              "joint_a": SimpleNamespace(n_dof=1, dof_indices=[0], lower_limit=-1., upper_limit=1.)}
    robot = SimpleNamespace(joints=joints, get_joint_positions=lambda: np.array([.1, .2]),
        get_position_orientation=lambda: ([0., 0., .01], [0., 0., 0., 1.]), urdf_path="/installed/r1.urdf",
        base_footprint_link_name="base", links={"base": SimpleNamespace(aabb=[[-.3, -.3, .04], [.3, .3, .4]])})
    floor = NativeObject("ground", [[-10., -10., -.1], [10., 10., 0.]], category="ground", fixed=True)
    table = NativeObject("table", [[-1., -1., 0.], [1., 1., .8]], category="table", fixed=True)
    payload = NativeObject("item_01", [[-.05, -.05, .8], [.05, .05, .9]], contacts=("table",))
    scene = SimpleNamespace(objects=[robot, floor, table, payload], system_registry=lambda key, name: None)
    session = SimpleNamespace(robot=robot, env=SimpleNamespace(scene=scene), pipeline=ForbiddenSimulation())
    return session, floor, table, payload


def target_row(report, name="item_01"):
    return next(row for row in report["objects"] if row["name"] == name)


def test_exports_measured_schema_enabled_background_meshes_and_encoder_indices(native):
    session, floor, table, payload = native
    table.links["base"].collision_meshes = {
        "top": mesh("top", [[-1., -1., .75], [1., 1., .8]]),
        "leg": mesh("leg", [[-.9, -.9, 0.], [-.8, -.8, .75]]),
        "disabled": mesh("disabled", table.aabb, enabled=False),
        "visual": mesh("visual", table.aabb, visual=True),
    }
    decoration = NativeObject("label", [[0., 0., .8], [.1, .1, .81]], visual=True)
    session.env.scene.objects.append(decoration)
    before = deepcopy(payload.aabb)
    regions = {"delivery": {"aabb": [[1., 1., .8], [1.2, 1.2, .8]], "support": "table", "support_z": .8}}
    report = inventory.export_inventory(session, identity="baseline_01", target_names=["item_01"],
                                        protected_names=["item_01"], regions=regions)
    assert report["schema"] == "wia_candidate_native_inventory_v1" and report["prototype"] == "baseline"
    assert report["robot"]["joints"] == {"joint_b": .2, "joint_a": .1}
    assert report["robot"]["joint_calibration"]["joint_a"]["parent_joint"] == np.eye(4).tolist()
    assert report["robot"]["usd_path"] == str(inventory.R1_USD)
    assert {row["name"] for row in report["objects"] if row.get("parent_object") == "table"} == {
        "table::base::top", "table::base::leg"}
    assert not any(row.get("parent_object") == "label" for row in report["objects"])
    target = target_row(report)
    assert target["protected"] and target["policy_role"] == "protected" and target["grasp_required"]
    assert target["support"] == "table" and target["support_contact_verified"]
    assert target["support_aabb"] == table.aabb and target["prim_type"] == "RIGID"
    assert report["floor_z"] == 0. and report["floor_measurement"]["collision_row"] == "ground::base::body"
    assert payload.aabb == before and regions["delivery"]["support"] == "table"
    report["regions"]["delivery"]["support"] = "changed_return_only"
    assert regions["delivery"]["support"] == "table"


def test_support_requires_touching_and_rejects_higher_or_nonoverlapping_objects(native):
    session, floor, table, payload = native
    payload.states[Touching] = ContactState(("high", "far"))
    session.env.scene.objects.extend([
        NativeObject("high", [[-1., -1., 0.], [1., 1., 1.2]], fixed=True),
        NativeObject("far", [[2., 2., 0.], [3., 3., .8]], fixed=True)])
    row = target_row(inventory.export_inventory(session, identity="no_support", target_names=["item_01"]))
    assert row["support"] is None and row["support_status"] == "unavailable"
    assert "support_aabb" not in row


def test_support_selects_closest_touching_upward_native_top(native):
    session, floor, table, payload = native
    lower = NativeObject("lower", [[-1., -1., 0.], [1., 1., .77]], fixed=True)
    session.env.scene.objects.append(lower)
    payload.states[Touching] = ContactState(("table", "lower"))
    row = target_row(inventory.export_inventory(session, identity="closest", target_names=["item_01"]))
    assert row["support"] == "table"


def test_explicit_support_mapping_measures_actual_object_and_never_invents_missing_support(native):
    session, floor, table, payload = native
    payload.states[Touching] = ContactState()
    row = target_row(inventory.export_inventory(session, identity="mapped", target_names=["item_01"],
                                               support_by_target={"item_01": "table"}))
    assert row["support"] == "table" and row["support_status"] == "measured_caller_mapping"
    assert row["support_contact_verified"] is False
    missing = target_row(inventory.export_inventory(session, identity="missing", target_names=["item_01"],
                                                   support_by_target={"item_01": "not_loaded"}))
    assert missing["support"] is None and "support_aabb" not in missing
    assert missing["support_status"] == "unavailable"


@pytest.mark.parametrize("category", ["floors", "floor", "ground", "lawn", "driveway"])
def test_only_actual_floor_categories_supply_floor_height(native, category):
    session, floor, table, payload = native
    floor.category = category
    report = inventory.export_inventory(session, identity="floor", target_names=["item_01"])
    assert next(row for row in report["objects"] if row.get("parent_object") == "ground")["role"] == "floor"
    assert all(row["role"] == "background" for row in report["objects"] if row.get("parent_object") == "table")
    assert report["floor_z"] == 0.


def test_missing_real_floor_does_not_substitute_a_table_or_zero(native):
    session, floor, table, payload = native
    floor.category = "carpet_decoration"
    with pytest.raises(ValueError, match="actual floor-category"):
        inventory.export_inventory(session, identity="no_floor", target_names=["item_01"])


def test_floor_binding_preserves_enabled_floor_geometry_and_structural_accounting(native):
    session, floor, table, payload = native
    report = inventory.export_inventory(session, identity="bound_floor", target_names=["ground", "item_01"])
    assert report["requested_target_names"] == ["ground", "item_01"]
    assert report["floor_z"] == 0.
    assert report["floor_measurement"]["source"] == "native_floor_collision_mesh_aabb_top"
    assert report["floor_measurement"]["collision_row"] == "ground::base::body"
    structural = report["structural_targets"][0]
    assert structural["name"] == "ground" and structural["structural_reason"] == "bound_floor"
    assert structural["support_status"] == "not_applicable_structural_floor"
    assert structural["aabb"] == floor.aabb and structural["active_for_geometric_audit"] is True
    assert structural["grasp_required"] is False and structural["rigid_grasp_proxy_eligible"] is False
    assert all(row["name"] != "ground" for row in report["inactive_targets"])
    assert {row["name"] for row in _initial_targets(report)} == {"item_01"}
    floor_mesh = next(row for row in report["objects"] if row.get("parent_object") == "ground")
    assert floor_mesh["role"] == "floor" and floor_mesh["collision_enabled"] and not floor_mesh["grasp_required"]


def test_disabled_bound_floor_has_no_global_object_aabb_height_fallback(native):
    session, floor, table, payload = native
    floor.links["base"].collision_meshes["body"].collision_enabled = False
    with pytest.raises(ValueError, match="actual floor-category"):
        inventory.export_inventory(session, identity="disabled_bound_floor", target_names=["ground", "item_01"])


def test_nonrigid_and_articulated_targets_carry_limitations_without_grasp_eligibility(native):
    session, floor, table, payload = native
    cloth = NativeObject("cloth", [[.2, .2, .8], [.4, .4, .81]], kind=1)
    cloth.links = {}
    articulated = NativeObject("container", [[1., 1., 0.], [1.5, 1.5, 1.]], joints={"door": SimpleNamespace(n_dof=1)})
    liquid = NativeObject("liquid", [[2., 2., 0.], [2.2, 2.2, .2]], category="water")
    session.env.scene.objects.extend([cloth, articulated, liquid])
    report = inventory.export_inventory(session, identity="types", target_names=["cloth", "container", "liquid"])
    for name in ("cloth", "container", "liquid"):
        row = target_row(report, name)
        assert not row["grasp_required"] and not row["rigid_grasp_proxy_eligible"]
        assert len(row["screening_limitations"]) >= 2
    assert target_row(report, "cloth")["prim_type"] == "CLOTH"
    assert target_row(report, "container")["articulated"] is True
    assert target_row(report, "container")["articulated_joint_names"] == ["door"]
    assert target_row(report, "liquid")["liquid"] is True


def test_invisible_disabled_task_is_preserved_for_accounting_but_not_navigation(native):
    session, floor, table, payload = native
    payload.visible = False
    payload.position = [1000., 1000., -1000.]
    payload.aabb = [[999.95, 999.95, -1000.05], [1000.05, 1000.05, -999.95]]
    payload.links["base"].collision_meshes["body"].collision_enabled = False
    report = inventory.export_inventory(session, identity="inactive", target_names=["item_01"], protected_names=["item_01"])
    assert not any(row["name"] == "item_01" or row.get("parent_object") == "item_01" for row in report["objects"])
    assert _initial_targets(report) == []
    row = report["inactive_targets"][0]
    assert row["visible"] is False and not row["grasp_required"] and row["fixed_base"] is False
    assert row["protected"] and row["position"] == [1000., 1000., -1000.]
    assert "native_visible_false" in row["screening_excluded_reason"]
    assert "no_enabled_collision_geometry" in row["screening_excluded_reason"]


def test_invisible_enabled_collision_geometry_remains_a_background_obstacle(native):
    session, floor, table, payload = native
    payload.visible = False
    report = inventory.export_inventory(session, identity="hidden_collision", target_names=["item_01"])
    row = next(row for row in report["objects"] if row.get("parent_object") == "item_01")
    assert row["role"] == "background" and row["collision_enabled"] and not row["grasp_required"]
    assert report["inactive_targets"][0]["visible"] is False
    assert _initial_targets(report) == []


def test_particle_system_and_absent_target_are_explicitly_unavailable(native):
    session, floor, table, payload = native
    class FluidSystem:
        pass
    system = FluidSystem()
    session.env.scene.system_registry = lambda key, name: system if name == "water_particles" else None
    report = inventory.export_inventory(session, identity="systems", target_names=["water_particles", "missing"])
    missing = {row["name"]: row for row in report["unavailable_targets"]}
    assert missing["water_particles"]["prim_type"] == "LIQUID_SYSTEM"
    assert missing["missing"]["prim_type"] == "UNAVAILABLE"
    assert not any(row["grasp_required"] for row in missing.values())


def base_mesh_geometry(session, *, angles=(0., 0., np.pi / 4), scale=(1., 1., 1.), offset=(0., 0., 0.)):
    """Supply measured native vertices/transforms without any simulator operations."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", angles)
    position = np.array([1.2, -2.3, .2])
    robot_transform = np.eye(4)
    robot_transform[:3, :3], robot_transform[:3, 3] = rotation.as_matrix(), position
    mesh_transform = np.eye(4)
    mesh_transform[:3, :3], mesh_transform[:3, 3] = np.diag(scale), offset
    native_points = np.array(list(product((-.3, .3), (-.2, .2), (0., .3))))
    scaled_transform = robot_transform @ mesh_transform
    world = (np.column_stack((native_points, np.ones(len(native_points)))) @ scaled_transform.T)[:, :3]
    bounds = [world.min(axis=0).tolist(), world.max(axis=0).tolist()]
    body = mesh("base_body", bounds)
    body.points, body.scaled_transform = native_points, scaled_transform
    link = session.robot.links["base"]
    link.aabb, link.collision_meshes = bounds, {"body": body}
    session.robot.get_position_orientation = lambda: (position.copy(), rotation.as_quat())
    return native_points, world, rotation, mesh_transform


def test_base_native_vertices_inverse_yaw_avoid_rotated_world_aabb_inflation(native):
    session, *_ = native
    native_points, world, rotation, _ = base_mesh_geometry(session)
    before = world.copy()
    report = inventory.export_inventory(session, identity="base_vertices", target_names=["item_01"])
    geometry = report["robot"]["base_geometry"]
    assert geometry["available"] and geometry["status"] == "available"
    assert geometry["provenance"] == "enabled_native_base_link_collision_mesh_vertices"
    assert geometry["enabled_mesh_names"] == ["base::body"]
    assert geometry["enabled_mesh_count"] == geometry["measured_mesh_count"] == 1
    assert geometry["vertex_count"] == len(native_points)
    assert np.allclose(geometry["world_vertices"], world)
    assert np.allclose(geometry["base_local_xy_bounds"], [[-.3, -.2], [.3, .2]])
    assert geometry["max_world_aabb_residual_m"] == pytest.approx(0.)
    box = report["robot"]["base_aabb"]
    broad_corners = np.array(list(product(*zip(*box))))
    inflated = rotation.inv().apply(broad_corners - report["robot"]["position"])
    assert np.ptp(inflated[:, 0]) > .6 and np.ptp(inflated[:, 1]) > .4
    assert np.array_equal(world, before)


def test_base_vertices_use_native_scaled_transform_and_current_full_robot_pose(native):
    session, *_ = native
    points, world, _, mesh_transform = base_mesh_geometry(session, angles=(.1, -.07, .6),
                                                         scale=(2., .5, 1.2), offset=(.08, -.03, .02))
    report = inventory.export_inventory(session, identity="scaled_base", target_names=["item_01"])
    geometry = report["robot"]["base_geometry"]
    expected = (np.column_stack((points, np.ones(len(points)))) @ mesh_transform.T)[:, :3]
    assert geometry["available"] and np.allclose(geometry["world_vertices"], world)
    assert np.allclose(geometry["base_local_xy_bounds"], [expected[:, :2].min(axis=0), expected[:, :2].max(axis=0)])
    assert geometry["meshes"][0]["prim_path"] == "/meshes/base_body"


@pytest.mark.parametrize("change,reason", [
    ("missing_meshes", "base_collision_meshes_unavailable"),
    ("missing_points", "base_collision_mesh_measurement_unavailable"),
    ("nan_points", "base_collision_mesh_measurement_unavailable"),
    ("invalid_transform", "base_collision_mesh_measurement_unavailable"),
    ("unknown_enabled", "base_collision_mesh_measurement_unavailable"),
    ("disabled", "no_enabled_base_collision_meshes"),
    ("aabb_mismatch", "collision_vertices_base_aabb_mismatch"),
])
def test_base_geometry_unavailable_never_fabricates_tighter_local_bounds(native, change, reason):
    session, *_ = native
    base_mesh_geometry(session)
    link = session.robot.links["base"]
    body = link.collision_meshes["body"]
    if change == "missing_meshes":
        del link.collision_meshes
    elif change == "missing_points":
        del body.points
    elif change == "nan_points":
        body.points[0, 0] = np.nan
    elif change == "invalid_transform":
        body.scaled_transform[3, 0] = 1.
    elif change == "unknown_enabled":
        del body.collision_enabled
    elif change == "disabled":
        body.collision_enabled = False
    else:
        link.aabb[1][0] += .003
    unchanged_box = deepcopy(link.aabb)
    report = inventory.export_inventory(session, identity="invalid_base", target_names=["item_01"])
    geometry = report["robot"]["base_geometry"]
    assert geometry["available"] is False and geometry["status"] == "unavailable"
    assert geometry["reason"] == reason
    assert "base_local_xy_bounds" not in geometry and "world_vertices" not in geometry
    assert report["robot"]["base_aabb"] == unchanged_box
    if change == "aabb_mismatch":
        assert geometry["max_world_aabb_residual_m"] == pytest.approx(.003)
        assert geometry["aabb_tolerance_m"] == .002


def test_base_vertices_include_all_enabled_meshes_and_exclude_disabled_meshes(native):
    session, *_ = native
    base_mesh_geometry(session)
    link = session.robot.links["base"]
    duplicate = deepcopy(link.collision_meshes["body"])
    disabled = mesh("disabled_base", [[-100., -100., -100.], [100., 100., 100.]], enabled=False)
    link.collision_meshes.update(duplicate=duplicate, disabled=disabled)
    geometry = inventory.export_inventory(session, identity="base_all", target_names=["item_01"])["robot"]["base_geometry"]
    assert geometry["available"]
    assert geometry["enabled_mesh_names"] == ["base::body", "base::duplicate"]
    assert geometry["enabled_mesh_count"] == 2 and geometry["vertex_count"] == 16
    assert geometry["skipped_meshes"][0]["name"] == "base::disabled"


@pytest.mark.parametrize("door_box", [
    [[1., -.02, 0.], [1.6, .02, 1.5]],  # One measured closed geometry.
    [[1., -.6, 0.], [1.04, .02, 1.5]],  # A different measured open geometry.
])
def test_fixed_articulated_containers_preserve_exact_per_link_meshes_and_accounting(native, door_box):
    session, floor, table, payload = native
    unit = NativeObject("unit_a", [[1., -.6, 0.], [1.6, .6, 1.5]], category="fridge", fixed=True,
                        joints={"door_hinge": SimpleNamespace(n_dof=1)})
    wall = [[1., -.6, 0.], [1.04, .6, 1.5]]
    shelf = [[1., -.6, .7], [1.6, .6, .73]]
    unit.links = {
        "body": SimpleNamespace(collision_meshes={"wall": mesh("fridge_wall", wall), "shelf": mesh("fridge_shelf", shelf)}),
        "door": SimpleNamespace(collision_meshes={"panel": mesh("fridge_door", door_box),
                                                   "disabled": mesh("unused", unit.aabb, enabled=False)}),
    }
    session.env.scene.objects.append(unit)
    before = deepcopy(unit.aabb)
    report = inventory.export_inventory(session, identity="fixed_articulated", target_names=["item_01", "unit_a"],
        protected_names=["unit_a"], support_by_target={"item_01": "table", "unit_a": "ground"})
    structural = next(row for row in report["structural_targets"] if row["name"] == "unit_a")
    assert structural["structural_reason"] == "fixed_articulated_container"
    assert structural["aabb"] == before and structural["articulated_joint_names"] == ["door_hinge"]
    assert structural["protected"] and structural["policy_role"] == "protected"
    assert not structural["grasp_required"] and not structural["rigid_grasp_proxy_eligible"]
    assert structural["support"] == "ground" and structural["support_status"] == "measured_caller_mapping"
    assert structural["active_collision_mesh_count"] == 3
    assert report["requested_target_names"] == ["item_01", "unit_a"]
    assert not any(row["name"] == "unit_a" for row in report["objects"] + report["inactive_targets"])
    actual = {row["name"]: row for row in report["objects"] if row.get("parent_object") == "unit_a"}
    assert set(actual) == {"unit_a::body::wall", "unit_a::body::shelf", "unit_a::door::panel"}
    assert actual["unit_a::body::wall"]["aabb"] == wall
    assert actual["unit_a::body::shelf"]["aabb"] == shelf
    assert actual["unit_a::door::panel"]["aabb"] == door_box
    assert all(row["role"] == "background" and row["collision_enabled"] and not row["grasp_required"] for row in actual.values())
    assert all(row["protected"] and row["policy_role"] == "protected" for row in actual.values())
    assert {row["name"] for row in _initial_targets(report)} == {"item_01"}
    assert target_row(report)["support"] == "table" and unit.aabb == before


def test_fixed_nonrigid_articulated_target_is_not_reclassified_as_container(native):
    session, *_ = native
    cloth = NativeObject("cloth", [[.2, .2, .8], [.4, .4, .81]], kind="CLOTH", fixed=True,
                         joints={"anchor": SimpleNamespace(n_dof=1)})
    session.env.scene.objects.append(cloth)
    report = inventory.export_inventory(session, identity="nonrigid_unchanged", target_names=["cloth"])
    assert target_row(report, "cloth")["prim_type"] == "CLOTH"
    assert not report["structural_targets"]

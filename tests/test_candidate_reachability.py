"""Check geometric audit semantics. Run: python -m pytest -q tests/test_candidate_reachability.py."""
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from wia.candidates.reachability import (
    AuditOptions, AuditUnavailable, _Navigation, _SE2Navigation, _Robot, _approaches, _approach_height_candidates, _collision,
    _grasp_point, _payload_overlap_details, _destination_centers, _screen_destination_samples,
    _navigation_evidence,
    _validated_base_vertices,
    _initial_targets, _prepare_destination_queries, _screen_target, _support_bounds,
    _virtual_destination, audit_scene, joint_limit_margin, sphere_aabb_clearances,
    tool_orientation, transform_points,
)


def test_world_transform_uses_column_vector_convention():
    transform = np.array([[0., -1., 0., 4.], [1., 0., 0., 7.],
                          [0., 0., 1., .5], [0., 0., 0., 1.]])
    assert np.allclose(transform_points([[1., 0., 2.], [0., 3., 0.]], transform),
                       [[4., 8., 2.5], [1., 7., .5]])
    assert np.allclose(transform_points(transform_points([[1., 2., 3.]], transform),
                                       np.linalg.inv(transform)), [[1., 2., 3.]])


def test_signed_sphere_aabb_separation_includes_internal_penetration_and_contact():
    box = np.array([[0., 0., 0.], [1., 1., 1.]])
    values = sphere_aabb_clearances([[1.5, .5, .5], [1.2, .5, .5], [.5, .5, .5],
                                     [2., 2., .5]], [.2, .2, .1, .2], box)
    assert np.allclose(values, [.3, 0., -.6, np.sqrt(2) - .2])


def test_joint_limit_boundaries_are_valid_and_outside_is_negative():
    limits = {"revolute": [-1., 2.], "prismatic": [0., .05]}
    assert joint_limit_margin({"revolute": -1., "prismatic": .05}, limits) == 0.
    assert joint_limit_margin({"revolute": .5, "prismatic": .02}, limits) == .02
    assert np.isclose(joint_limit_margin({"revolute": 2.1}, limits), -.1)


def _sphere_robot(link):
    return SimpleNamespace(ignored=set(), cloud=lambda joints, base:
        (np.array([[0., 0., .5]]), np.array([.1]), [link]))


def test_only_active_fingers_and_only_at_terminal_target_contact_are_permitted():
    target = {"name": "mug", "aabb": [[-.05, -.05, .45], [.05, .05, .55]]}
    finger = _sphere_robot("left_gripper_link1")
    hits, _ = _collision(finger, {}, np.eye(4), [target], "mug", "left", False, .006)
    assert hits == ["environment:mug:left_gripper_link1"]
    assert not _collision(finger, {}, np.eye(4), [target], "mug", "left", True, .006)[0]
    assert _collision(finger, {}, np.eye(4), [target], "mug", "right", True, .006)[0]
    assert _collision(_sphere_robot("left_arm_link6"), {}, np.eye(4), [target],
                      "mug", "left", True, .006)[0]
    table = dict(target, name="table")
    assert _collision(finger, {}, np.eye(4), [target, table], "mug", "left", True, .006)[0] == [
        "environment:table:left_gripper_link1"]


def test_proxy_self_collisions_remain_active_except_installed_ignored_pairs():
    robot = SimpleNamespace(ignored=set(), cloud=lambda joints, base:
        (np.array([[0., 0., .5], [.15, 0., .5]]), np.array([.1, .1]),
         ["left_arm_link1", "right_arm_link1"]))
    hits, clearance = _collision(robot, {}, np.eye(4), [], "target", "left", False, .006)
    assert hits == ["self:left_arm_link1:right_arm_link1"]
    assert np.isclose(clearance, -.05)
    robot.ignored = {frozenset(("left_arm_link1", "right_arm_link1"))}
    assert not _collision(robot, {}, np.eye(4), [], "target", "left", False, .006)[0]


def test_floor_support_permission_applies_only_to_wheels():
    wheel = _sphere_robot("wheel_link1")
    arm = _sphere_robot("right_arm_link6")
    assert not _collision(wheel, {}, np.eye(4), [], "target", "right", False, .006, floor_z=.5)[0]
    assert _collision(arm, {}, np.eye(4), [], "target", "right", False, .006, floor_z=.5)[0] == [
        "environment:supporting_floor:right_arm_link6"]


def test_base_grid_connectivity_preserves_a_blocking_wall():
    robot = SimpleNamespace(position=np.array([-1., 0., 0.]), base_radius=.2,
                            base_box=np.array([[-1.2, -.2, .05], [-.8, .2, .3]]))
    target = {"name": "target", "aabb": [[1., -.03, .8], [1.06, .03, .9]]}
    wall = {"name": "wall", "aabb": [[-.05, -20., 0.], [.05, 20., 3.]]}
    nav = _Navigation(robot, [wall, target], [target], AuditOptions())
    assert nav.node(np.array([-1., .5])) in nav.previous
    assert nav.node(np.array([1., 0.])) not in nav.previous
    assert nav.segment_blocked(np.array([-1., 0.]), np.array([1., 0.]))
    assert not nav.segment_blocked(np.array([-1., 0.]), np.array([-1., .5]))
    assert not nav.stances(target, [target], AuditOptions())


def test_missing_native_geometry_reports_unavailable_not_unreachable(tmp_path):
    inventory = {"floor_z": 0., "robot": {}, "objects": [
        {"name": "target", "aabb": [[0., 0., .8], [.05, .05, .85]], "grasp_required": True}]}
    result = audit_scene(inventory, tmp_path)
    assert result["status"] == "unavailable"
    assert result["targets"][0]["status"] == "unavailable"
    assert result["targets"][0]["witness"] is None
    assert result["robot_commands_executed"] is False
    assert (tmp_path / "reachability.json").is_file()


def test_tool_frame_maps_measured_axes_and_side_displacements_consistently():
    rotation = tool_orientation([1., 0., 0.], [0., 0., -1.], [0., -1., 0.], [0., 0., 1.])
    assert np.allclose(rotation @ [1., 0., 0.], [0., -1., 0.])
    assert np.allclose(rotation @ [0., 0., -1.], [0., 0., 1.])
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.isclose(np.linalg.det(rotation), 1.)
    approaches = _approaches({"approach_axis_eef": [0., 0., 1.],
        "opening_axis_eef": [0., -1., 0.]}, [-1., 0.], [0., 0., .8])
    assert len(approaches) == 4
    assert np.allclose(approaches[2]["direction_world"], [1., 0., 0.])
    assert np.allclose(approaches[2]["orientation_world"] @ [0., 0., 1.], [1., 0., 0.])
    center = np.array([0., 0., .8])
    assert np.allclose(center - .1 * approaches[2]["direction_world"], [-.1, 0., .8])


def test_support_mesh_union_uses_parent_and_excludes_legs_and_other_objects():
    target = {"name": "target", "support": "table", "aabb": [[.1, .1, .8], [.15, .15, .9]]}
    objects = [
        {"name": "table::root::panel_a", "parent_object": "table", "aabb": [[-1., -1., .75], [0., 1., .8]]},
        {"name": "table::root::panel_b", "parent_object": "table", "aabb": [[0., -1., .75], [1., 1., .8]]},
        {"name": "table::root::leg", "parent_object": "table", "aabb": [[-.9, -.9, 0.], [-.8, -.8, .8]]},
        {"name": "other::root::panel", "parent_object": "other", "aabb": [[-5., -5., .7], [5., 5., .8]]},
    ]
    box, source = _support_bounds(target, objects)
    assert np.allclose(box, [[-1., -1., .75], [1., 1., .8]])
    assert source == "union_of_horizontal_support_collision_mesh_aabbs"
    target["support_aabb"] = [[-2., -2., 0.], [2., 2., .8]]
    assert np.allclose(_support_bounds(target, objects)[0], target["support_aabb"])


def test_continuous_ik_uses_measured_torso_seed_and_native_limits():
    from scipy.spatial.transform import Rotation
    robot = _Robot.__new__(_Robot)
    names = [f"torso_joint{i}" for i in range(1, 5)] + [f"left_arm_joint{i}" for i in range(1, 7)]
    measured = {name: .1 + i * .01 for i, name in enumerate(names)}
    robot.limits = {name: [-.5, 1.] for name in names}
    robot.Rotation = Rotation
    robot.fk = lambda arm, joints: np.eye(4)
    captured = {}
    def solve(residual, seed, **kwargs):
        captured.update(seed=seed.copy(), bounds=kwargs["bounds"], max_nfev=kwargs["max_nfev"])
        assert np.allclose(residual(seed), np.zeros(6))
        return SimpleNamespace(x=seed)
    robot.least_squares = solve
    robot.options = AuditOptions()
    values, pe, ae = robot.solve("left", measured, np.zeros(3), np.eye(3), 1)
    assert np.allclose(captured["seed"], list(measured.values()))
    assert len(captured["seed"]) == 10 and captured["max_nfev"] == 100
    assert np.allclose(captured["bounds"], [[-.5] * 10, [1.] * 10])
    assert values == measured and pe == ae == 0.
    robot.options = AuditOptions(torso_search="fixed_presets")
    robot.solve("left", measured, np.zeros(3), np.eye(3), 0)
    assert len(captured["seed"]) == 6


def test_kinematic_pair_survives_collision_without_promoting_final_witness():
    seed = {f"torso_joint{i}": 0. for i in range(1, 5)}
    seed.update({f"{arm}_gripper_axis{i}": .03 for arm in ("left", "right") for i in (1, 2)})
    def pose(position, quaternion):
        matrix = np.eye(4)
        matrix[:3, 3] = position
        return matrix
    robot = SimpleNamespace(seed=seed, position=np.array([-1., 0., 0.]), pose_matrix=pose,
        limits={name: [-1., 1.] for name in seed}, ignored=set(),
        gripper_dimensions=lambda arm, joints: {"approach_axis_eef": [0., 0., 1.], "opening_axis_eef": [0., -1., 0.]},
        solve=lambda arm, joints, position, orientation, index: (dict(joints), 0., 0.),
        cloud=lambda joints, base: (np.array([[0., 0., .83]]), np.array([.1]), ["left_arm_link6"]))
    navigation = SimpleNamespace(stances=lambda *args: [(0, 0.)], failure=None,
        position=lambda node: np.array([-1., 0.]), route=lambda node: [[-1., 0.]])
    target = {"name": "target", "aabb": [[-.03, -.03, .8], [.03, .03, .86]]}
    result = _screen_target(target, [target], robot, navigation, AuditOptions(max_ik_attempts_per_target=6), 0.)
    assert result["status"] == "not_found" and result["witness"] is None
    assert result["kinematic_witness_count"] == 2
    assert result["kinematic_witness"]["stages"][1]["name"] == "grasp"
    assert result["kinematic_witness"]["collision_and_path_status"] == "blocked_or_unresolved"
    assert any("environment:target:left_arm_link6" in reason
               for reason in result["kinematic_witness"]["blocking_reasons"])


def _destination_fixture():
    payload = {"name": "item_01", "role": "target", "fixed_base": False, "aabb": [[0., 0., .5], [.06, .08, .6]],
        "position": [.025, .035, .54], "orientation": [0., 0., .1, .995], "support": "old_table",
        "support_aabb": [[-1., -1., 0.], [1., 1., .5]], "protected": False, "policy_role": "payload"}
    region = {"support": "new_table", "support_z": .8, "aabb": [[2., 3., .8], [2.4, 3.4, .8]]}
    return payload, region


def test_virtual_destination_translates_pose_and_aabb_preserving_size_orientation_and_input():
    payload, region = _destination_fixture()
    other = dict(deepcopy(payload), name="item_02")
    objects = [other, payload, {"name": "item_01::root::collision", "parent_object": "item_01", "aabb": payload["aabb"]}]
    before = deepcopy([payload, region, objects])
    virtual, query_objects, metadata = _virtual_destination(region, objects, [other, payload])
    assert metadata["source_name"] == "item_01"
    assert np.allclose(np.ptp(virtual["aabb"], axis=0), np.ptp(payload["aabb"], axis=0))
    assert np.allclose(np.mean(virtual["aabb"], axis=0)[:2], [2.2, 3.2])
    assert np.isclose(virtual["aabb"][0][2], .8)
    assert virtual["orientation"] == payload["orientation"]
    assert np.allclose(np.asarray(virtual["position"]) - payload["position"], metadata["translation_world_m"])
    assert virtual["support"] == "new_table" and "support_aabb" not in virtual
    assert [row["name"] for row in query_objects] == ["item_02", "item_01"]
    assert metadata["removed_original_collision_rows"] == ["item_01", "item_01::root::collision"]
    virtual["orientation"][0] = 1.
    assert [payload, region, objects] == before


def test_destination_retains_resident_and_door_collision_geometry():
    payload, region = _destination_fixture()
    resident = {"name": "resident", "aabb": [[2.18, 3.18, .8], [2.23, 3.23, .95]], "protected": True}
    door = {"name": "fridge::door::collision", "parent_object": "fridge", "aabb": [[2.15, 3.15, .7], [2.25, 3.25, 1.]]}
    objects = [payload, resident, door]
    virtual, query_objects, _ = _virtual_destination(region, objects, [payload])
    assert {row["name"] for row in query_objects} == {"item_01", "resident", "fridge::door::collision"}
    robot = SimpleNamespace(ignored=set(), cloud=lambda joints, base:
        (np.array([[2.2, 3.2, .85]]), np.array([.01]), ["left_gripper_link1"]))
    hits, _ = _collision(robot, {}, np.eye(4), query_objects, virtual["name"], "left", True, .006)
    assert "environment:resident:left_gripper_link1" in hits
    assert "environment:fridge::door::collision:left_gripper_link1" in hits
    queries, pending = _prepare_destination_queries({"delivery": region}, objects, [payload])
    assert pending and queries[0]["sample_records"][0]["status"] == "not_found"
    assert set(queries[0]["retained_object_aabb_overlaps"]) == {"resident", "fridge::door::collision"}
    for _, plans in pending:
        for _, _, sample_objects in plans:
            assert {row["name"] for row in sample_objects} == {"item_01", "resident", "fridge::door::collision"}


def test_too_small_destination_is_explicitly_unavailable_and_cannot_resize_payload():
    payload, region = _destination_fixture()
    region["aabb"][1][0] = 2.03
    with pytest.raises(AuditUnavailable, match="footprint"):
        _virtual_destination(region, [payload], [payload])
    queries, pending = _prepare_destination_queries({"tiny": region}, [payload], [payload])
    assert not pending and queries[0]["status"] == "unavailable"
    assert queries[0]["hypothetical_payload"] is None
    assert "footprint" in queries[0]["reason"]


def test_closed_interior_destination_is_unavailable_without_removing_the_door():
    payload, region = _destination_fixture()
    region.update(interior=True, requires_open_doors=True, doors_open=False)
    door = {"name": "fridge::door::collision", "aabb": [[2., 3., 0.], [2.1, 3.4, 1.]]}
    objects = [payload, door]
    before = deepcopy(objects)
    queries, pending = _prepare_destination_queries({"inside": region}, objects, [payload])
    assert not pending and queries[0]["status"] == "unavailable"
    assert "opened doors" in queries[0]["reason"]
    assert objects == before


def test_protected_movable_is_still_an_initial_geometric_audit_target():
    payload, _ = _destination_fixture()
    protected = dict(deepcopy(payload), name="item_02", protected=True, policy_role="preserve_resident", grasp_required=False)
    fixed = dict(deepcopy(payload), name="anchor", fixed_base=True, protected=True)
    inventory = {"objects": [payload, protected, fixed], "targets": ["item_01"]}
    targets = _initial_targets(inventory)
    assert {row["name"] for row in targets} == {"item_01", "item_02"}
    assert next(row for row in targets if row["name"] == "item_02")["policy_role"] == "preserve_resident"


def test_audit_keeps_destination_results_separate_and_does_not_mutate_inventory(monkeypatch, tmp_path):
    import wia.candidates.reachability as module
    payload, region = _destination_fixture()
    inventory = {"floor_z": 0., "objects": [payload], "robot": {}, "regions": {"delivery": region}}
    before = deepcopy(inventory)
    calls = []
    robot = SimpleNamespace(urdf=tmp_path / "robot.urdf", calibration={}, sphere_path=tmp_path / "spheres.yaml",
                            base_radius=.2, seed={}, limits={})
    monkeypatch.setattr(module, "_Robot", lambda row, options: robot)
    monkeypatch.setattr(module, "_Navigation", lambda *args: SimpleNamespace(
        shape=(1, 1), resolution=.06, previous={(0, 0): None}, failure=None))
    def screen(target, objects, robot, navigation, options, floor_z):
        calls.append(deepcopy(target))
        return {"name": target["name"], "status": "found_witness", "witness": {"goal": target["aabb"]},
                "policy_metadata": module._policy_metadata(target), "policy_permission_inferred": False}
    monkeypatch.setattr(module, "_screen_target", screen)
    report = audit_scene(inventory, tmp_path)
    assert inventory == before
    assert len(report["targets"]) == len(report["destination_queries"]) == 1
    assert calls[0]["support"] == "old_table" and calls[1]["support"] == "new_table"
    destination = report["destination_queries"][0]
    assert destination["name"] == "destination:delivery"
    assert destination["hypothetical_payload"]["source_name"] == "item_01"
    assert "Initial-state-only" in destination["scope"]
    assert destination["policy_permission_inferred"] is False
    assert report["targets"][0]["witness"] != destination["witness"]


def test_explicit_ineligible_proxy_is_counted_unavailable_with_original_accounting(tmp_path):
    unsupported = {"name": "container", "role": "target", "fixed_base": True, "grasp_required": False,
        "aabb": [[0., 0., 0.], [1., 1., 1.]], "prim_type": "RIGID", "articulated": True,
        "rigid_grasp_proxy_eligible": False, "protected": True, "screening_limitations": ["Whole-container proxy."]}
    inventory = {"floor_z": 0., "robot": {}, "objects": [unsupported],
        "inactive_targets": [{"name": "offstage", "visible": False}],
        "unavailable_targets": [{"name": "water", "prim_type": "LIQUID_SYSTEM"}],
        "structural_targets": [{"name": "floor", "structural_reason": "bound_floor"}]}
    before = deepcopy(inventory)
    report = audit_scene(inventory, tmp_path)
    result = report["targets"][0]
    assert report["status"] == "completed" and result["status"] == "unavailable"
    assert result["witness"] is None and result["kinematic_witness"] is None and result["ik_attempts"] == 0
    assert result["input_metadata"] == unsupported and "different grasp representation" in result["reason"]
    assert report["inactive_targets"] == inventory["inactive_targets"]
    assert report["unavailable_targets"] == inventory["unavailable_targets"]
    assert report["structural_targets"] == inventory["structural_targets"]
    assert _screen_target(unsupported, [], None, None, AuditOptions(), 0.)["status"] == "unavailable"
    report["inactive_targets"][0]["visible"] = True
    result["input_metadata"]["screening_limitations"].append("Only returned metadata changed.")
    assert inventory == before


def test_ineligible_proxy_cannot_be_hypothetical_payload_and_remains_an_obstacle():
    unsupported, region = _destination_fixture()
    unsupported["rigid_grasp_proxy_eligible"] = False
    legacy = dict(deepcopy(unsupported), name="item_02")
    legacy.pop("rigid_grasp_proxy_eligible")
    virtual, query_objects, metadata = _virtual_destination(region, [unsupported, legacy], [unsupported, legacy])
    assert metadata["source_name"] == "item_02" and virtual["name"] == "item_02"
    assert unsupported in query_objects
    with pytest.raises(AuditUnavailable, match="footprint"):
        _virtual_destination(region, [unsupported], [unsupported])


def test_unsupported_active_geometry_is_retained_for_other_target_checks(monkeypatch, tmp_path):
    import wia.candidates.reachability as module
    payload, _ = _destination_fixture()
    unsupported = dict(deepcopy(payload), name="cloth", rigid_grasp_proxy_eligible=False, prim_type="CLOTH")
    robot = SimpleNamespace(urdf=tmp_path / "robot", calibration={}, sphere_path=tmp_path / "spheres",
                            base_radius=.2, seed={}, limits={})
    monkeypatch.setattr(module, "_Robot", lambda row, options: robot)
    monkeypatch.setattr(module, "_Navigation", lambda *args: SimpleNamespace(
        shape=(1, 1), resolution=.06, previous={(0, 0): None}, failure=None))
    calls = []
    def screen(target, objects, *args):
        calls.append(target["name"])
        assert unsupported in objects
        return {"name": target["name"], "status": "not_found", "witness": None}
    monkeypatch.setattr(module, "_screen_target", screen)
    report = audit_scene({"floor_z": 0., "robot": {}, "objects": [payload, unsupported]}, tmp_path)
    assert calls == ["item_01"]
    assert {row["name"]: row["status"] for row in report["targets"]} == {"item_01": "not_found", "cloth": "unavailable"}


def test_only_declared_support_plane_contact_gets_payload_bottom_tolerance():
    payload, region = _destination_fixture()
    table = {"name": "new_table::root::top", "parent_object": "new_table", "aabb": [[2., 3., .7], [2.4, 3.4, .805]]}
    resident = {"name": "resident", "aabb": deepcopy(table["aabb"])}
    virtual, objects, _ = _virtual_destination(region, [payload, table, resident], [payload])
    overlaps, permitted = _payload_overlap_details(virtual, objects)
    assert overlaps == ["resident"] and permitted[0]["object"] == table["name"]
    assert np.isclose(permitted[0]["z_overlap_m"], .005)
    table["aabb"][1][2] = .816
    assert table["name"] in _payload_overlap_details(virtual, objects)[0]
    table["aabb"][1][2] = .805
    robot = SimpleNamespace(ignored=set(), cloud=lambda joints, base:
        (np.array([[2.2, 3.2, .803]]), np.array([.01]), ["left_gripper_link1"]))
    assert any(table["name"] in hit for hit in _collision(robot, {}, np.eye(4), objects, virtual["name"], "left", True, .006)[0])


def test_top_height_search_prioritizes_interior_point_that_clears_thin_support():
    box = np.array([[0., 0., .8], [.08, .055, .835]])
    assert _approach_height_candidates()[0] == (0, .75)
    assert {fraction for index, fraction in _approach_height_candidates() if index < 2} == {.5, .75, .9}
    assert {fraction for index, fraction in _approach_height_candidates() if index >= 2} == {.5}
    measured_forward_fingertip_m = .01689
    assert _grasp_point(box, .5)[2] - measured_forward_fingertip_m - box[0, 2] < .006
    assert _grasp_point(box, .75)[2] - measured_forward_fingertip_m - box[0, 2] > .006
    assert box[0, 2] < _grasp_point(box, .75)[2] < box[1, 2]


def _footprint_robot():
    pose = np.eye(4)
    pose[:3, 3] = [-1., 0., 0.]
    return SimpleNamespace(position=np.array([-1., 0., 0.]), initial_pose=pose,
        base_box=np.array([[-1.45, -.2, .01], [-.55, .2, .3]]), base_radius=np.hypot(.45, .2))


def _straight_corridor():
    return [{"name": "north", "aabb": [[-10., .26, 0.], [10., 2., 3.]]},
            {"name": "south", "aabb": [[-10., -2., 0.], [10., -.26, 3.]]}]


def test_rectangle_lattice_passes_narrow_straight_corridor_that_blocks_circle():
    robot = _footprint_robot()
    target = {"name": "target", "aabb": [[.95, -.03, .8], [1.05, .03, .9]]}
    options = AuditOptions(navigation_mode="se2_rectangle", grid_resolution_m=.1)
    circle = _Navigation(robot, _straight_corridor(), [target], options)
    rectangle = _SE2Navigation(robot, _straight_corridor(), [target], options)
    assert not circle.previous and circle.failure
    node = rectangle.node(np.array([1., 0.]), 0.)
    assert rectangle.failure is None and node in rectangle.previous
    route = rectangle.route(node)
    assert np.allclose(route[0], [-1., 0., 0.]) and np.allclose(route[-1], [1., 0., 0.])
    assert all(len(point) == 3 for point in route)
    assert _navigation_evidence(rectangle, options)["heading_count"] == 16
    stances = rectangle.stances(target, _straight_corridor(), options)
    assert stances
    stance, facing = stances[0]
    route = rectangle.route(stance)
    assert np.allclose(route[-1], [*rectangle.position(stance), facing])
    assert np.isclose(facing, np.arctan2(-rectangle.position(stance)[1], 1. - rectangle.position(stance)[0]))
    for before, after in zip(route[:-1], route[1:]):
        if not np.allclose(before[:2], after[:2]):
            assert np.isclose(before[2], after[2])
            assert not rectangle.translation_blocked(np.array(before[:2]), np.array(after[:2]), before[2])
        if not np.isclose(before[2], after[2]):
            assert np.allclose(before[:2], after[:2])
            assert not rectangle.rotation_blocked(np.array(before[:2]), before[2], after[2])


def _geometry_navigation(boxes):
    navigation = _SE2Navigation.__new__(_SE2Navigation)
    navigation.options = AuditOptions(navigation_mode="se2_rectangle")
    navigation.local_polygon = np.array([[-.45, -.2], [.45, -.2], [.45, .2], [-.45, .2]])
    navigation.radius = np.hypot(.45, .2)
    navigation.boxes = np.asarray(boxes).reshape(-1, 2, 2)
    navigation.max_rotation_sample_step_rad = 0.
    return navigation


def test_corner_rotation_is_blocked_even_when_both_endpoint_rectangles_are_free():
    navigation = _geometry_navigation([[[.29, .29], [.31, .31]]])
    assert not navigation.pose_blocked([0., 0.], 0.)
    assert not navigation.pose_blocked([0., 0.], np.pi / 2)
    assert navigation.rotation_blocked([0., 0.], 0., np.pi / 2)
    polygons, inflation = navigation._rotation_samples(0., np.pi / 2)
    assert navigation.max_rotation_sample_step_rad <= np.deg2rad(5.) + 1e-12
    assert len(polygons) == 19
    assert np.isclose(inflation, 2 * navigation.radius * np.sin(np.deg2rad(5.) / 4))


def test_translation_sweep_preserves_obstacle_between_clear_endpoints():
    navigation = _geometry_navigation([[[-.02, -.02], [.02, .02]]])
    assert not navigation.pose_blocked([-1., 0.], 0.)
    assert not navigation.pose_blocked([1., 0.], 0.)
    assert navigation.translation_blocked(np.array([-1., 0.]), np.array([1., 0.]), 0.)


def test_rectangle_lattice_never_forces_a_blocked_actual_start_to_be_free():
    robot = _footprint_robot()
    blocker = {"name": "start_blocker", "aabb": [[-1.1, -.1, 0.], [-.9, .1, .5]]}
    navigation = _SE2Navigation(robot, [*_straight_corridor(), blocker], [], AuditOptions(navigation_mode="se2_rectangle"))
    assert navigation.failure == "initial_base_rectangle_collision" and not navigation.previous


def test_rectangle_retains_measured_center_offset_and_total_lattice_cap():
    robot = _footprint_robot()
    robot.base_box[:, 0] += .12
    robot.base_box[:, 1] += .03
    blocker = {"name": "start_blocker", "aabb": [[-1.1, -.1, 0.], [-.9, .1, .5]]}
    options = AuditOptions(navigation_mode="se2_rectangle", grid_resolution_m=.01, max_grid_cells=4096)
    navigation = _SE2Navigation(robot, [blocker], [], options)
    assert np.allclose(navigation.local_bounds.mean(axis=0), [.12, .03])
    evidence = _navigation_evidence(navigation, options)
    assert evidence["total_lattice_states"] <= evidence["state_cap"] == 4096
    assert evidence["resolution_m"] > .01 and evidence["heading_count"] == 16


def test_se2_route_unwraps_heading_boundary_into_the_checked_short_rotation():
    navigation = _SE2Navigation.__new__(_SE2Navigation)
    navigation.initial_yaw = 0.
    navigation.yaw_step = 2 * np.pi / 16
    navigation.start = np.zeros(2)
    navigation.lower = np.zeros(2)
    navigation.resolution = .1
    first, last = (0, 0, 0), (15, 0, 0)
    navigation.previous = {first: None, last: first}
    navigation.stance_positions = {last: np.zeros(2)}
    navigation.stance_yaws = {last: 2 * np.pi - .1}
    route = np.asarray(navigation.route(last))
    assert np.max(np.abs(np.diff(route[:, 2]))) <= 2 * np.pi / 16 + 1e-12
    assert np.isclose(route[-1, 2], -.1)
    assert np.isclose(route[-1, 2] % (2 * np.pi), navigation.stance_yaws[last])


def test_destination_grid_order_and_full_footprint_fit_with_six_mm_inset():
    payload, region = _destination_fixture()
    centers, inset_fit = _destination_centers(region, payload["aabb"])
    assert inset_fit and len(centers) == 9
    assert np.allclose(centers[0], [2.2, 3.2])
    assert centers[1][0] < centers[0][0] < centers[2][0]
    assert np.isclose(centers[1][1], centers[0][1])
    assert centers[3][1] < centers[0][1] < centers[4][1]
    before = deepcopy(payload)
    for center in centers:
        virtual, _, _ = _virtual_destination(region, [payload], [payload], center_xy=center)
        box = np.asarray(virtual["aabb"])
        assert np.all(box[0, :2] >= np.asarray(region["aabb"])[0, :2] + .006 - 1e-12)
        assert np.all(box[1, :2] <= np.asarray(region["aabb"])[1, :2] - .006 + 1e-12)
        assert virtual["orientation"] == payload["orientation"]
        assert np.allclose(np.ptp(box, axis=0), np.ptp(payload["aabb"], axis=0))
    assert payload == before


def test_destination_grid_deduplicates_and_retains_geometric_center_without_inset_fit():
    centers, inset_fit = _destination_centers({"aabb": [[0., 0., 0.], [1., 1., 0.]]},
                                             [[0., 0., 0.], [.988, .988, .1]])
    assert inset_fit and len(centers) == 1 and np.allclose(centers[0], [.5, .5])
    payload, region = _destination_fixture()
    region["aabb"] = [[2., 3., .8], [2.061, 3.081, .8]]
    entries, pending = _prepare_destination_queries({"tight": region}, [payload], [payload])
    assert pending and len(entries[0]["sample_records"]) == 1
    assert entries[0]["sample_grid_policy"]["geometric_center_fallback_without_inset"]
    assert np.allclose(entries[0]["hypothetical_payload"]["measured_size_m"], [.06, .08, .1])


def _stub_destination_navigation(monkeypatch):
    import wia.candidates.reachability as module
    monkeypatch.setattr(module, "_make_navigation", lambda *args: SimpleNamespace(
        shape=(1, 1), resolution=.06, previous={(0, 0): None}, failure=None))
    return module


def test_destination_occupied_center_can_find_free_edge_without_erasing_residents(monkeypatch):
    module = _stub_destination_navigation(monkeypatch)
    payload, region = _destination_fixture()
    resident = {"name": "resident", "protected": True, "aabb": [[2.17, 3.17, .8], [2.23, 3.23, .95]]}
    source = [payload, resident]
    before = deepcopy(source)
    entries, pending = _prepare_destination_queries({"delivery": region}, source, [payload])
    assert entries[0]["sample_records"][0]["retained_object_aabb_overlaps"] == ["resident"]
    calls = []
    def screen(target, objects, robot, navigation, options, floor_z):
        calls.append(options.max_ik_attempts_per_target)
        assert resident in objects and len([row for row in objects if row["name"] == "item_01"]) == 1
        return {"name": target["name"], "status": "found_witness", "ik_attempts": 3,
            "witness": {"grasped_sample_aabb": target["aabb"], "route": ["one_sample_only"]}, "kinematic_witness": None}
    monkeypatch.setattr(module, "_screen_target", screen)
    entry, plans = pending[0]
    _screen_destination_samples(entry, plans, None, AuditOptions(), 0.)
    assert calls == [192] and entry["ik_attempts"] == 3
    assert entry["status"] == "found_witness" and entry["selected_sample_index"] == 1
    selected = entry["sample_records"][entry["selected_sample_index"]]
    assert entry["hypothetical_payload"] == selected["hypothetical_payload"]
    assert entry["witness"] == selected["result"]["witness"]
    assert entry["witness"]["grasped_sample_aabb"] != entry["sample_records"][0]["aabb"]
    assert all(sample["result"] is not None for sample in entry["sample_records"])
    assert entry["sample_records"][0]["ik_attempts"] == 0 and source == before


def test_destination_samples_share_one_aggregate_ik_budget(monkeypatch):
    module = _stub_destination_navigation(monkeypatch)
    payload, region = _destination_fixture()
    resident = {"name": "resident", "aabb": [[2.17, 3.17, .8], [2.23, 3.23, .95]]}
    entries, pending = _prepare_destination_queries({"delivery": region}, [payload, resident], [payload])
    options = AuditOptions(max_ik_attempts_per_target=5)
    remaining = []
    def screen(target, objects, robot, navigation, sample_options, floor_z):
        remaining.append(sample_options.max_ik_attempts_per_target)
        assert sample_options.collision_margin_m == options.collision_margin_m == .006
        return {"name": target["name"], "status": "not_found", "witness": None, "kinematic_witness": None,
                "ik_attempts": min(3, sample_options.max_ik_attempts_per_target)}
    monkeypatch.setattr(module, "_screen_target", screen)
    entry, plans = pending[0]
    _screen_destination_samples(entry, plans, None, options, 0.)
    assert remaining == [5, 2] and options.max_ik_attempts_per_target == 5
    assert entry["ik_attempts"] == 5 and entry["aggregate_ik_budget_remaining"] == 0
    assert sum(sample["ik_attempts"] for sample in entry["sample_records"]) == 5
    assert entry["sample_records"][0]["ik_attempts"] == 0
    assert entry["hypothetical_payload"] == entry["sample_records"][0]["hypothetical_payload"]
    assert entry["witness"] is None and entry["status"] == "not_found"
    assert all(sample["status"] != "pending" for sample in entry["sample_records"])


def _native_base_vertex_fixture(*, roll=0., pitch=0., offset=(0., 0.)):
    from scipy.spatial.transform import Rotation
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, np.pi / 4]).as_matrix()
    pose[:3, 3] = [-1., 0., 0.]
    low, high = np.array([-.35 + offset[0], -.3 + offset[1], .02]), np.array([.35 + offset[0], .3 + offset[1], .35])
    local = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    world = transform_points(local, pose)
    box = np.stack((world.min(axis=0), world.max(axis=0)))
    geometry = {"schema": "wia_robot_base_geometry_v1", "available": True, "status": "available",
        "world_vertices": world.tolist(), "max_world_aabb_residual_m": 0.,
        "measured_base_aabb": box.tolist(), "measured_collision_world_aabb": box.tolist(),
        "base_local_xy_bounds": [[-.001, -.001], [.001, .001]], "enabled_mesh_names": ["base::measured_mesh"]}
    radius = np.linalg.norm(np.maximum(np.abs(box[0, :2] - pose[:2, 3]), np.abs(box[1, :2] - pose[:2, 3])))
    robot = SimpleNamespace(position=pose[:3, 3].copy(), initial_pose=pose, base_box=box, base_radius=radius)
    return robot, geometry, local, world


def _blocked_native_base_navigation(robot):
    blocker = {"name": "start_blocker", "aabb": [[-1.01, -.01, 0.], [-.99, .01, .5]]}
    return _SE2Navigation(robot, [blocker], [], AuditOptions(navigation_mode="se2_rectangle"))


def test_yaw_45_native_vertices_preserve_actual_rectangle_instead_of_inverse_aabb_inflation():
    robot, geometry, _, _ = _native_base_vertex_fixture()
    legacy = SimpleNamespace(**vars(robot))
    robot.base_world_vertices, robot.base_geometry_validation = _validated_base_vertices(geometry, robot.base_box)
    navigation = _blocked_native_base_navigation(robot)
    fallback = _blocked_native_base_navigation(legacy)
    assert np.allclose(np.ptp(navigation.local_bounds, axis=0), [.7, .6])
    assert np.all(np.ptp(fallback.local_bounds, axis=0) > np.ptp(navigation.local_bounds, axis=0) + .2)
    evidence = _navigation_evidence(navigation, AuditOptions(navigation_mode="se2_rectangle"))
    assert evidence["footprint_source"] == "validated_native_base_collision_vertices"
    assert evidence["base_geometry_world_aabb_residual_m"] == 0.
    assert evidence["base_geometry_fallback_reason"] is None
    assert evidence["base_geometry_enabled_mesh_names"] == ["base::measured_mesh"]
    # Deliberately tiny supplied local bounds were ignored; circle still uses
    # exactly the same observed world-AABB radius as the legacy fixture.
    assert np.all(np.ptp(navigation.local_bounds, axis=0) > .002)
    assert _Navigation(robot, [], [], AuditOptions()).radius == _Navigation(legacy, [], [], AuditOptions()).radius


@pytest.mark.parametrize("invalid", ["schema", "unavailable", "too_few", "nonfinite", "world_mismatch", "reported_residual"])
def test_invalid_native_base_vertices_retain_existing_aabb_fallback_with_reason(invalid):
    robot, geometry, _, _ = _native_base_vertex_fixture()
    fallback = _blocked_native_base_navigation(robot)
    if invalid == "schema":
        geometry["schema"] = "unverified_geometry"
    elif invalid == "unavailable":
        geometry.update(available=False, status="unavailable", reason="measurement_missing")
    elif invalid == "too_few":
        geometry["world_vertices"] = geometry["world_vertices"][:3]
    elif invalid == "nonfinite":
        geometry["world_vertices"][0][0] = np.nan
    elif invalid == "world_mismatch":
        geometry["world_vertices"] = (np.asarray(geometry["world_vertices"]) + [.003, 0., 0.]).tolist()
    else:
        geometry["max_world_aabb_residual_m"] = .003
    robot.base_world_vertices, robot.base_geometry_validation = _validated_base_vertices(geometry, robot.base_box)
    assert robot.base_world_vertices is None
    navigation = _blocked_native_base_navigation(robot)
    evidence = _navigation_evidence(navigation, AuditOptions(navigation_mode="se2_rectangle"))
    assert np.allclose(navigation.local_bounds, fallback.local_bounds)
    assert evidence["footprint_source"] == "measured_base_aabb_corners" and not evidence["base_geometry_validated"]
    assert evidence["base_geometry_fallback_reason"] and evidence["footprint_vertex_count"] == 8
    if invalid == "world_mismatch":
        assert evidence["base_geometry_world_aabb_residual_m"] == pytest.approx(.003)


def test_native_vertices_keep_rotated_offset_and_both_full_pose_and_yaw_projections():
    robot, geometry, local, world = _native_base_vertex_fixture(roll=.004, pitch=-.003, offset=(.12, .025))
    robot.base_world_vertices, robot.base_geometry_validation = _validated_base_vertices(geometry, robot.base_box)
    navigation = _blocked_native_base_navigation(robot)
    bounds = navigation.local_bounds
    assert np.all(local[:, :2] >= bounds[0] - 1e-12) and np.all(local[:, :2] <= bounds[1] + 1e-12)
    yaw = navigation.initial_yaw
    yaw_projection = (world[:, :2] - robot.position[:2]) @ np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    assert np.all(yaw_projection >= bounds[0] - 1e-12) and np.all(yaw_projection <= bounds[1] + 1e-12)
    assert np.any(bounds[0] < local[:, :2].min(axis=0) - 1e-5) or np.any(bounds[1] > local[:, :2].max(axis=0) + 1e-5)
    assert np.allclose(bounds.mean(axis=0), [.12, .025], atol=.002)
    assert navigation.footprint_evidence["base_geometry_validated"]


def test_radial_stance_connects_outside_wider_obstacle_when_small_support_edges_all_collide():
    robot = _footprint_robot()
    table = {"name": "table", "aabb": [[-.5, -.5, 0.], [.5, .5, .8]]}
    board = {"name": "board", "aabb": [[-.1, -.1, .8], [.1, .1, .81]]}
    target = {"name": "target", "support": "board", "aabb": [[-.03, -.03, .81], [.03, .03, .88]]}
    options = AuditOptions(navigation_mode="se2_rectangle", grid_resolution_m=.25, max_grid_cells=4096)
    navigation = _SE2Navigation(robot, [table, board, target], [target], options)
    stances = navigation.stances(target, [table, board, target], options)
    sampling = navigation.last_stance_sampling
    assert stances and sampling["pose_clear_counts_by_source"]["edge"] == 0
    assert sampling["connected_feasible_counts_by_source"]["edge"] == 0
    assert sampling["connected_feasible_counts_by_source"]["radial"] > 0
    assert sampling["selected_counts_by_source"]["radial"] > 0
    assert len(stances) <= 12
    for node, yaw in stances:
        point = navigation.position(node)
        assert not navigation.pose_blocked(point, yaw)
        assert np.allclose(navigation.route(node)[0][:2], robot.position[:2])
        assert any(row[2] == "table" for row in navigation.obstacles)


def test_radial_pose_cannot_cross_a_blocking_wall_without_a_checked_route():
    robot = _footprint_robot()
    wall = {"name": "wall", "aabb": [[-.03, -20., 0.], [.03, 20., 3.]]}
    target = {"name": "target", "aabb": [[1.47, -.03, .8], [1.53, .03, .9]]}
    options = AuditOptions(navigation_mode="se2_rectangle", grid_resolution_m=.25, max_grid_cells=4096)
    navigation = _SE2Navigation(robot, [wall, target], [target], options)
    assert navigation.previous and navigation.failure is None
    assert navigation.stances(target, [wall, target], options) == []
    sampling = navigation.last_stance_sampling
    assert sampling["pose_clear_counts_by_source"]["radial"] > 0
    assert sampling["connected_feasible_counts_by_source"]["radial"] == 0
    assert sampling["raw_proposal_counts"]["initial_root"] == 0
    assert any(row[2] == "wall" for row in navigation.obstacles)


def test_initial_root_is_honored_and_repeated_target_calls_preserve_node_pose_maps():
    robot = _footprint_robot()
    robot.position[0] += .6
    robot.initial_pose[0, 3] += .6
    robot.base_box[:, 0] += .6
    target = {"name": "target", "aabb": [[-.03, -.03, .8], [.03, .03, .9]],
              "support_aabb": [[-5., -5., 0.], [5., 5., 0.]]}
    options = AuditOptions(navigation_mode="se2_rectangle", grid_resolution_m=.25, max_grid_cells=2048)
    navigation = _SE2Navigation(robot, [target], [target], options)
    first = navigation.stances(target, [target], options)
    assert first and np.allclose(navigation.position(first[0][0]), robot.position[:2])
    assert navigation.last_stance_sampling["selected_counts_by_source"]["initial_root"] == 1
    preserved = {node: (point.copy(), navigation.stance_yaws[node]) for node, point in navigation.stance_positions.items()}
    repeated = navigation.stances(target, [target], options)
    assert repeated == first and len(repeated) <= 12
    other = dict(target, name="another", aabb=[[.02, -.03, .8], [.08, .03, .9]])
    additional = navigation.stances(other, [target, other], options)
    for node, (point, yaw) in preserved.items():
        assert np.array_equal(navigation.stance_positions[node], point) and navigation.stance_yaws[node] == yaw
    for node, yaw in additional:
        assert navigation.stance_yaws[node] == yaw
    for index, (node, yaw) in enumerate(repeated):
        for previous_node, previous_yaw in repeated[:index]:
            assert not navigation._same_stance(navigation.position(node), yaw,
                                              navigation.position(previous_node), previous_yaw)


def test_radial_proposals_are_deterministic_and_deduplicate_initial_pose_by_xy_yaw():
    navigation = _geometry_navigation([])
    navigation.start, navigation.initial_yaw = np.array([-.75, 0.]), 0.
    target = {"name": "target", "aabb": [[-.03, -.03, .8], [.03, .03, .9]],
              "support_aabb": [[-5., -5., 0.], [5., 5., 0.]]}
    first, counts, offset = navigation._stance_proposals(target, [], AuditOptions(navigation_mode="se2_rectangle"))
    second, other_counts, other_offset = navigation._stance_proposals(target, [], AuditOptions(navigation_mode="se2_rectangle"))
    assert counts == other_counts == {"edge": 12, "initial_root": 1, "radial": 64}
    assert np.isclose(offset, np.pi) and offset == other_offset
    assert len(first) == len(second) == 76
    for left, right in zip(first, second):
        assert np.array_equal(left["point"], right["point"]) and left["yaw"] == right["yaw"]
        assert left["provenance"] == right["provenance"]
    initial = next(proposal for proposal in first if any(p["kind"] == "initial_root" for p in proposal["provenance"]))
    assert {p["kind"] for p in initial["provenance"]} == {"initial_root", "radial"}
    radial = next(p for p in initial["provenance"] if p["kind"] == "radial")
    assert radial["radius_m"] == .75 and radial["direction_index"] == 0

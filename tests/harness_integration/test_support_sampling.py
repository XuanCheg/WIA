"""Check support-layer selection and conservative coarse navigation screening.

Run: .venv/bin/python -m pytest -q tests/test_support_sampling.py
"""
import json
import sys

import numpy as np
import pytest
from types import SimpleNamespace

from wia.tasks.scene_only import local_support_surface
from wia.tasks.support_sampling import (
    SamplingConfig, SupportSampler, chassis_footprint, choose_stance,
    footprint_clearance, grid_route, support_edge_points, support_footprint_policy, upward_layers,
)
from wia.runtime.commands.sample_esi_supports import main


def hit(z, body="/shelf/layer", normal=1., distance=.01):
    return {"hit": True, "rigidBody": body, "position": [0., 0., z],
        "normal": [0., 0., normal], "distance": distance}


def test_interior_shelf_layer_is_selected_without_using_whole_furniture_top():
    hits = [hit(.5), hit(.5001), hit(.8), hit(1.8), hit(.6, normal=-1.), hit(.7, body="/wall")]
    assert upward_layers(hits, "/shelf", .35, 1.1) == [.5, .8]
    assert local_support_surface("/shelf", [hit(.5), hit(.49, distance=.02), hit(.6, body="/wall")])["position"][2] == .5


def test_robot_footprint_cannot_cross_a_narrow_gap_or_snap_from_blocked_start():
    free = np.ones((21, 21), dtype=bool)
    free[:, 10] = False
    free[10, 10] = True
    assert grid_route(free, (10, 4), (10, 16)) is not None
    eroded = footprint_clearance(free, .1, .2)
    assert grid_route(eroded, (10, 4), (10, 16)) is None
    assert grid_route(free, (2, 10), (10, 4)) is None
    free[7:14, 10] = True
    assert grid_route(footprint_clearance(free, .1, .2), (10, 4), (10, 16)) is not None


def test_empty_articulation_root_does_not_reduce_physical_chassis_clearance():
    robot = SimpleNamespace(base_footprint_link_name="base_link",
        root_link=SimpleNamespace(aabb=(np.zeros(3), np.zeros(3))),
        links={"base_link": SimpleNamespace(aabb=(np.zeros(3), np.array([.4, .3, .2])))})
    name, extent, radius = chassis_footprint(robot)
    assert name == "base_link" and np.array_equal(extent, [.4, .3, .2])
    assert radius == .28


def test_coarse_mode_can_retain_a_stance_without_claiming_verified_connectivity():
    free = np.ones((5, 5), dtype=bool)
    free[2, 2] = False
    path, stance, connected = choose_stance(free, np.array([2, 2]), [(2, 4)], require_connected_route=False)
    assert path == [] and stance == (2, 4) and not connected
    with pytest.raises(ValueError, match="No approximate same-room route"):
        choose_stance(free, np.array([2, 2]), [(2, 4)], require_connected_route=True)


def test_named_sofa_fallback_keeps_room_and_task_furniture_exclusions(tmp_path):
    def furniture(name, category, x):
        return SimpleNamespace(name=name, category=category, visual_only=False,
            aabb=(np.array([x, 0., 0.]), np.array([x + .2, .2, .8])))

    table = furniture("orange_table", "breakfast_table", 2.)
    sofa = furniture("sofa", "sofa", 0.)
    work = furniture("work_table", "table", 0.)
    old_stand = furniture("pickup_surface_000_top", "table", 0.)
    scene = SimpleNamespace(objects=[table, sofa, work, old_stand], object_registry=lambda *args: None,
        seg_map=SimpleNamespace(get_room_instance_by_point=lambda xy: "utility_room_0"
            if xy[0] > 1 else "living_room_0"))
    session = SimpleNamespace(env=SimpleNamespace(scene=scene), robot=None,
        room_name="living_room_0", task_object_names={"work_table"})
    definition = SimpleNamespace(source=SimpleNamespace(json_path="example.json"))
    sampler = SupportSampler(session, None, None,
        SamplingConfig(preferred_support_names=("orange_table", "sofa", "work_table", "pickup_surface_000_top")),
        tmp_path, SimpleNamespace(definition=definition))
    assert [obj.name for obj in sampler.eligible_supports("work_table")] == ["sofa"]
    assert sampler.report["unavailable_preferred_supports"] == [
        "orange_table", "work_table", "pickup_surface_000_top"]


def test_reachable_edge_proposals_keep_the_target_margin_and_actual_reach_limit():
    points, distance = support_edge_points(np.array([0., 0., .5]), np.array([1., 1., .6]),
        np.array([.2, .2]), np.array([[1.4, .5], [2., .5], [.5, -.4]]), np.array([1.4, .5]))
    assert np.allclose(points, [[.8, .5], [.5, .2]])
    assert distance == pytest.approx(.6)
    assert np.all(points >= .2) and np.all(points <= .8)
    assert not len(support_edge_points(np.zeros(3), np.array([.3, .3, .6]),
        np.array([.2, .2]), np.array([[.5, .5]]), np.zeros(2))[0])


def test_native_room_membership_overrides_a_coarse_furniture_centre_label(tmp_path):
    scene = SimpleNamespace(objects=[], object_registry=lambda *args: None,
        seg_map=SimpleNamespace(get_room_instance_by_point=lambda xy: 'utility_room_0'))
    session = SimpleNamespace(env=SimpleNamespace(scene=scene), robot=None, room_name='living_room_0')
    sampler = SupportSampler(session, None, None, SamplingConfig(), tmp_path,
        SimpleNamespace(definition=SimpleNamespace(source=SimpleNamespace(json_path='example.json'))))
    assert sampler.support_in_room(SimpleNamespace(in_rooms=['living_room_0']))
    assert not sampler.support_in_room(SimpleNamespace(in_rooms=['utility_room_0']))
    assert not sampler.support_in_room(SimpleNamespace(in_rooms=None, aabb=(np.zeros(3), np.ones(3))))


def test_stable_native_support_does_not_require_complete_planarity():
    assert support_footprint_policy('selected_sofa', 'sofa', ('selected_sofa',)) == 'native_contact_for_named_seating'
    assert support_footprint_policy('selected_chair', 'eames_chair', ('selected_chair',)) == 'native_contact_for_named_seating'
    assert support_footprint_policy('other_sofa', 'sofa', ('selected_sofa',)) == 'native_contact_and_settling'
    assert support_footprint_policy('selected_table', 'table', ('selected_table',)) == 'native_contact_and_settling'
    assert support_footprint_policy('selected_table', 'table', ('selected_table',), True) == 'planar_local_raycast'


def test_fallback_uses_coherent_initial_geometry_after_state_restore(tmp_path, monkeypatch):
    position = np.array([1., 2., .5])
    quaternion = np.array([0., 0., 0., 1.])
    low, high = np.array([.9, 1.9, .45]), np.array([1.1, 2.1, .55])
    target = SimpleNamespace(get_position_orientation=lambda: (position, quaternion), aabb=(low, high))
    invalidations = []

    def restore(*args, **kwargs):
        # Reproduce root pose / collision-bound reads from different states.
        position[2] = .1
        low[2], high[2] = .7, .8

    monkeypatch.setitem(sys.modules, "omnigibson.utils.sampling_utils", SimpleNamespace(raytest=None))
    monkeypatch.setitem(sys.modules, "omnigibson.utils.usd_utils",
        SimpleNamespace(PoseAPI=SimpleNamespace(invalidate=lambda: invalidations.append(True))))
    scene = SimpleNamespace(objects=[], object_registry=lambda *args: target)
    sim = SimpleNamespace(dump_state=lambda **kwargs: {}, load_state=restore)
    session = SimpleNamespace(env=SimpleNamespace(scene=scene), robot=None,
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=sim)), invalidate_camera_cache=lambda: None)
    definition = SimpleNamespace(source=SimpleNamespace(json_path="example.json"),
        qualification=SimpleNamespace(support="table"), bindings={"table": SimpleNamespace(name="main_table")})
    sampler = SupportSampler(session, SimpleNamespace(pose=lambda: {}), None,
        SamplingConfig(), tmp_path, SimpleNamespace(definition=definition))
    captured = []
    monkeypatch.setattr(sampler, "draft_added_supports", lambda geometry: captured.append(geometry))
    result = sampler.run()
    saved_position, _, saved_low, saved_high = captured[0]
    assert result["status"] == "not_screened" and invalidations
    assert np.allclose(saved_position, [1., 2., .5])
    assert np.allclose(saved_low, [.9, 1.9, .45]) and np.allclose(saved_high, [1.1, 2.1, .55])
    assert result["initial_target_geometry"]["origin_above_bottom_m"] == pytest.approx(.05)


def test_batch_dry_run_selects_main_questions_without_creating_output(tmp_path, capsys):
    output = tmp_path / "not_created"
    assert main(["--dry-run", "--output-dir", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_count"] == 19
    assert result["excluded_counts"] == {"discarded": 2, "canonical_questions_only": 59}
    assert result["verification_level"] == "approximate_reachability"
    assert not result["physical_grasp_verified"]
    assert not output.exists()

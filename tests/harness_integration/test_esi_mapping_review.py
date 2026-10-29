"""CPU checks for whole-floor human review, optics, orientation and privacy.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest
tests/test_esi_mapping_review.py
"""
from types import SimpleNamespace as NS
import sys
from copy import deepcopy

import numpy as np
import pytest
import torch
import cv2
from scipy.spatial.transform import Rotation

from wia.runtime.sim.esi_debug_cameras import camera_orientation
from wia.runtime.sim.esi_floorplan_review import (
    FLOORPLAN_LABEL, fit_mapping_camera, is_cognitive_mapping_session,
    mapping_camera_geometry, render_floorplan_review, room_map_geometry,
)


class RoomMap:
    """Match the installed BaseMap axes, origin and single-floor room raster."""
    def __init__(self, raster, resolution=.5, floor_heights=(0.,)):
        self.room_ins_map = torch.tensor(raster, dtype=torch.int64)
        self.map_size = self.room_ins_map.shape[0]
        self.map_resolution = resolution
        self.floor_heights = floor_heights
        self.room_ins_id_to_ins_name = {1: "kitchen_0", 7: "bedroom_0", 22: "corridor_0"}
        self.room_ins_name_to_ins_id = {v: k for k, v in self.room_ins_id_to_ins_name.items()}

    def map_to_world(self, points):
        return torch.flip((points - self.map_size / 2.) * self.map_resolution, dims=(-1,))


def session(segmentation, robot=None):
    return NS(task_module=NS(__name__="active_explore.tasks.cognitive_mapping.connectivity"),
              env=NS(scene=NS(seg_map=segmentation, objects=[])), robot=robot, payload={})


def test_all_room_ids_and_cell_edges_are_in_whole_floor_bounds():
    raster = np.zeros((12, 12), int)
    raster[1:3, 1:3] = 1
    raster[8:11, 9:12] = 7
    raster[4, 5] = 22
    raster[0, 11] = 99  # Unregistered native IDs must remain visible.
    geometry = room_map_geometry(RoomMap(raster))
    assert geometry["all_raster_room_ids"] == [1, 7, 22, 99]
    assert geometry["bounds_world_xy_m"] == [[-2.75, -3.25], [2.75, 2.25]]
    assert geometry["rooms"][-1]["name"] == "room_id_99"
    assert not geometry["rooms"][-1]["name_available"]
    assert sum(room["cell_count"] for room in geometry["rooms"]) == np.count_nonzero(raster)


def test_floorplan_preserves_native_world_orientation_labels_and_robot_xy():
    raster = np.zeros((8, 8), int)
    raster[0:3, 0:3] = 1  # -X / -Y -> image bottom left.
    raster[5:8, 5:8] = 7  # +X / +Y -> image top right.
    robot = NS(get_position_orientation=lambda: (torch.tensor([1., 1., .4]), torch.tensor([0., 0., 0., 1.])))
    s = session(RoomMap(raster), robot)
    rgb, metadata = render_floorplan_review(s, image_size=512)
    assert rgb.dtype == np.uint8 and rgb.shape[2] == 3
    assert metadata["label"] == FLOORPLAN_LABEL
    assert metadata["native_render"] is False and metadata["human_only"] is True
    assert metadata["agent_exposed"] is False
    assert metadata["robot_marker"]["inside_map_bounds"]
    assert not metadata["robot_marker"]["floor_membership_verified"]
    assert metadata["all_raster_room_ids"] == [1, 7]
    # World Y is inverted exactly once for a conventional top-down display.
    color_bottom_left = rgb[350, 110]
    color_top_right = rgb[180, 400]
    assert not np.array_equal(color_bottom_left, color_top_right)
    assert not hasattr(s, "global_cameras") and not hasattr(robot, "sensors")
    rgb_repeat, repeat = render_floorplan_review(s, image_size=512)
    assert np.array_equal(rgb, rgb_repeat) and metadata == repeat


@pytest.mark.parametrize("bounds", [([0., 0.], [.1, 60.]), ([-120., -80.], [180., 250.]),
                                   ([-.5, -.5], [.5, .5])])
@pytest.mark.parametrize("overhead", [False, True])
@pytest.mark.parametrize("shape,focal", [((512, 512), 12.), ((1024, 256), 70.)])
def test_perspective_fit_contains_every_corner_for_narrow_large_and_small_maps(bounds, overhead, shape, focal):
    geometry = {"bounds_world_xy_m": bounds, "floor_z_m": 3.5}
    fitted = fit_mapping_camera(geometry, [-.35, -.35, 1.35], horizontal_aperture=20.995,
                                focal_length=focal, image_width=shape[0], image_height=shape[1], overhead=overhead)
    position, target = (np.asarray(fitted[key]) for key in ("position_world_m", "target_world_m"))
    orientation = camera_orientation(position, target)
    optical = Rotation.from_quat(orientation)
    assert np.allclose(optical.apply([0., 0., -1.]), (target - position) / np.linalg.norm(target - position))
    assert np.isfinite(position).all() and fitted["distance_m"] > 0
    low, high = np.asarray(bounds)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (3.5, 5.1)])
    local = optical.inv().apply(corners - position)
    depth = -local[:, 2]
    assert (depth > 0).all()
    assert (np.abs(local[:, 0]) / depth <= np.tan(np.deg2rad(fitted["horizontal_fov_deg"] / 2)) + 1e-8).all()
    assert (np.abs(local[:, 1]) / depth <= np.tan(np.deg2rad(fitted["vertical_fov_deg"] / 2)) + 1e-8).all()
    near, far = fitted["clipping_range_m"]
    assert 0 < near < far and (depth <= far).all()
    if overhead:
        assert position[2] - near == pytest.approx(5.1)
        assert fitted["cutaway_plane_world_z_m"] == pytest.approx(5.1)


def test_multifloor_metadata_reports_only_the_available_native_raster_layer():
    geometry = room_map_geometry(RoomMap([[1, 1], [7, 7]], floor_heights=(0., 3.)))
    assert geometry["floor_idx"] == 0 and geometry["declared_floor_heights_m"] == [0., 3.]
    assert any("floor_insseg_0 only" in text for text in geometry["limitations"])


def test_original_resolution_supplement_keeps_rooms_lost_by_runtime_resize(tmp_path):
    layout = tmp_path / "layout"
    layout.mkdir()
    raw = np.zeros((40, 40), np.uint8)
    raw[5:30, 5:30] = 1
    raw[38, 38] = 99  # Too small for the coarse native raster to retain.
    cv2.imwrite(str(layout / "floor_insseg_0.png"), raw)
    native = RoomMap([[1, 0], [0, 0]], resolution=.2)
    native.scene_dir = tmp_path
    geometry = room_map_geometry(native)
    assert geometry["all_raster_room_ids"] == [1, 99]
    assert geometry["native_raster_room_ids"] == [1]
    assert geometry["original_room_ids_lost_by_native_resize"] == [99]
    assert geometry["unregistered_room_ids"] == [99]
    assert geometry["map_resolution_m"] == geometry["original_map_resolution_m"] == .01
    assert geometry["native_map_resolution_m"] == .2
    assert geometry["bounds_world_xy_m"] == [[-.155, -.155], [.185, .185]]
    fitted_geometry = mapping_camera_geometry(session(native))
    # Union keeps both original high-resolution cells and runtime edge cells.
    assert fitted_geometry["bounds_world_xy_m"][0] == pytest.approx([-.3, -.3])
    rgb, metadata = render_floorplan_review(session(native), image_size=512)
    assert rgb.size and metadata["all_raster_room_ids"] == [1, 99]
    assert torch.equal(native.room_ins_map, torch.tensor([[1, 0], [0, 0]]))


def test_smaller_actual_vertical_aperture_increases_required_camera_distance():
    geometry = {"bounds_world_xy_m": [[-2., -10.], [2., 10.]], "floor_z_m": 0.}
    kwargs = dict(horizontal_aperture=20., focal_length=12., image_width=512, image_height=512, overhead=True)
    default = fit_mapping_camera(geometry, [0., 0., 1.], **kwargs)
    narrow = fit_mapping_camera(geometry, [0., 0., 1.], vertical_aperture=10., **kwargs)
    assert narrow["distance_m"] > default["distance_m"]
    assert narrow["vertical_fov_deg"] < default["vertical_fov_deg"]


def test_missing_map_fails_honestly_or_uses_original_structure_bounds():
    s = session(None)
    with pytest.raises(ValueError, match="native_room_map_unavailable"):
        render_floorplan_review(s)
    with pytest.raises(ValueError, match="whole_floor_camera_bounds_unavailable"):
        mapping_camera_geometry(s)
    s.env.scene.objects = [NS(category="floors", aabb=(np.array([-20., -10., -.2]), np.array([30., 40., 0.])))]
    geometry = mapping_camera_geometry(s)
    assert geometry["bounds_source"] == "native_structure_aabb_fallback"
    assert geometry["bounds_world_xy_m"] == [[-20., -10.], [30., 40.]]
    assert "room_map_error" in geometry and "all_raster_room_ids" not in geometry


@pytest.mark.parametrize("field,value", [
    ("task_module", NS(__name__="active_explore.tasks.cognitive_mapping.traversable_passage")),
    ("payload", {"_hf_big_task": "Cognitive Mapping"}),
    ("question_path", "/data/Cognitive Mapping/Regional Boundary/scene/full_scene/q_000.json"),
    ("metadata_path", "/data/cognitive_mapping/long_term_navigation/q_000.json"),
])
def test_all_four_task_session_identities_can_select_floor_review(field, value):
    assert is_cognitive_mapping_session(NS(**{field: value}))
    assert not is_cognitive_mapping_session(NS(payload={"small_task": "Connectivity"}))


def test_invalid_bounds_and_camera_target_do_not_create_nan_poses():
    with pytest.raises(ValueError):
        camera_orientation([0., 0., 0.], [0., 0., 0.])
    with pytest.raises(ValueError):
        fit_mapping_camera({"bounds_world_xy_m": [[0., 0.], [np.nan, 1.]], "floor_z_m": 0.},
                           [1., 0., 1.], horizontal_aperture=20., focal_length=12., image_width=512, image_height=512)


def test_native_configuration_uses_fitted_optics_and_only_human_sensor_clipping(monkeypatch):
    from wia.runtime.sim.esi_debug_cameras import configure_global_cameras
    from core.viewer_config import ViewerConfig

    class HumanSensor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.horizontal_aperture = 24.
        def load(self, scene):
            self.scene = scene
        def initialize(self):
            pass
        def get_attribute(self, key):
            assert key == "verticalAperture"
            return 16.
        def set_position_orientation(self, **kwargs):
            self.pose = kwargs

    monkeypatch.setitem(sys.modules, "omnigibson.sensors", NS(VisionSensor=HumanSensor))
    mounted = NS(all_modalities=set(), modalities={"rgb"}, clipping_range=(.01, 100.))
    robot = NS(get_position_orientation=lambda: (np.array([900., 900., .4]), np.array([0., 0., 0., 1.])),
               sensors={"head": mounted, "wrist": mounted})
    s = session(RoomMap([[1, 1], [7, 7]]), robot)
    s.global_cameras = {}
    s.camera_name, s.wrist_camera_name = "head", "wrist"
    render_calls = []
    s.pipeline = NS(og=NS(sim=NS(render=lambda: render_calls.append(True))))
    s.task_object_names = ()
    configure_global_cameras(s, ViewerConfig(), include_robot=False)
    assert len(s.global_cameras) == 6 and len(render_calls) == 3
    assert mounted.clipping_range == (.01, 100.)
    assert set(robot.sensors) == {"head", "wrist"}
    assert s.human_camera_framing["strategy"] == "whole_native_floor_bounds_v1"
    # A far-away robot must not drag a whole-floor review target off the map.
    assert np.linalg.norm(s.human_camera_framing["target_world_m"][:2]) < 1.
    for name, sensor in s.global_cameras.items():
        recorded = s.human_camera_framing["views"][name]
        assert recorded["horizontal_aperture_mm"] == 24.
        assert recorded["vertical_aperture_mm"] == 16.
        assert sensor.clipping_range == tuple(recorded["clipping_range_m"])
        assert np.isfinite(sensor.pose["orientation"]).all()
        assert recorded["cutaway_plane_world_z_m"] == (1.6 if name == "global_overhead" else None)


class R1:
    """Read-only native-shaped robot fixture for initialization contracts."""
    visual_only = False
    fixed_base = True
    prim_path = "/robot"
    base_footprint_link_name = "base"

    def __init__(self):
        self.position = np.array([.5, 0., .1])
        self.quaternion = np.array([0., 0., 2 ** -.5, 2 ** -.5])
        self.supported, self.colliding = True, False
        self.contact_z = 0.
        self.links = {"base": self}
        self.controllers = {"base": type("HolonomicBaseJointController", (), {})()}

    @property
    def aabb(self):
        return self.position - [.05, .05, .1], self.position + [.05, .05, .2]

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def get_linear_velocity(self):
        return np.zeros(3)

    def get_angular_velocity(self):
        return np.zeros(3)

    def contact_list(self):
        contacts = [NS(body0="/robot/base", body1="/floor/link", position=[.5, 0., self.contact_z],
                       normal=[0., 0., 1.])] if self.supported else []
        if self.colliding:
            contacts.append(NS(body0="/robot/base", body1="/wall/link", position=[.5, 0., .1], normal=[1., 0., 0.]))
        return contacts


def repaired_start_fixture(task):
    from wia.authoring.converters import cognitive_mapping_long_term_navigation as navigation
    from wia.authoring.converters import cognitive_mapping_traversable_passage as passage
    module = navigation if task == "navigation" else passage
    robot = R1()
    segmentation = NS(floor_heights=[0.], room_ins_name_to_ins_id={"source_0": 1, "target_0": 2},
                      get_room_instance_by_point=lambda xy: "source_0" if float(xy[0]) < 1 else "target_0")
    s = NS(robot=robot, env=NS(scene=NS(seg_map=segmentation, objects=[NS(category="floors", prim_path="/floor")])),
           scene_name="fixture", room_name="full_scene", pipeline=NS(og=NS(sim=NS(current_time=0.))))
    s.category_interaction_setup = {"accepted": True, "spawn_grid": {"requested_source_region": "source_0"},
        "robot_spawn": {"checks": {"floor_supported": True, "collision_free": True},
                        "robot_position_m": robot.position.tolist(), "robot_quaternion_xyzw": robot.quaternion.tolist()}}
    parameters = {"start_xy": [0., 0.], "goal_xy": [2., 0.], "start_region": "source_0", "target_region": "target_0",
        "floor_idx": 0, "floor_z": 0., "floor_z_m": 0., "initial_yaw_rad": 0.,
        "start_position_tolerance_m": .05, "start_heading_tolerance_deg": 2., "stability_steps": 3,
        "floor_contact_tolerance_m": .005, "minimum_up_dot": .95, "maximum_terminal_speed_m_s": .02,
        "maximum_terminal_angular_speed_rad_s": .1, "minimum_travel_m": .25}
    if task == 'navigation':
        parameters.update(required_regions=['source_0', 'target_0'], arrival_report_required=True,
                          public_route_required=False, required_order=[])
    config = {"module": module.MODULE, "operation": navigation.OPERATION if task == "navigation" else "passage_arrival",
              "scene": "fixture", "room": "full_scene", "binding_keys": [], "parameters": parameters}
    executor = NS(capabilities=NS(base_translation=True, base_rotation=True))
    controller = NS(config=NS(translation_tolerance_m=.005, max_translation_speed_m_s=.5),
                    pose=lambda: {"position": robot.position.tolist()})
    return module, s, config, executor, controller


@pytest.mark.parametrize("task", ["navigation", "passage"])
def test_verified_same_source_repair_is_distinct_from_original_start_and_gets_no_success_credit(task, tmp_path):
    module, s, config, executor, controller = repaired_start_fixture(task)
    before = deepcopy(config)
    result = module.qualify_operation(s, executor, config, {})
    assert result["accepted"] and result["checks"]["original_start_matches"] is False
    assert result["checks"]["start_contract"] == "verified_repaired_pose_same_source_region"
    assert result["checks"]["verified_repair_evidence"]["accepted"]
    evaluator = module.create_evaluator(s, controller, config, tmp_path)
    row = evaluator.begin_episode(NS(snapshot_id="cpu", action_step=0))
    assert row["task_success"] is False
    if task == 'passage':
        assert row['initial_qualification']['start_contract'] == 'verified_repaired_pose_same_source_region'
    else:
        assert all(row['initial_qualification'].values())
    assert config == before


@pytest.mark.parametrize("task", ["navigation", "passage"])
@pytest.mark.parametrize("case", ["missing_evidence", "setup_rejected", "wrong_requested_region", "reported_no_floor",
    "reported_collision", "missing_quaternion", "stale_position", "stale_orientation", "wrong_native_room",
    "native_unsupported", "native_collision", "wrong_contact_floor"])
def test_incomplete_stale_or_unsafe_repair_does_not_bypass_native_start_checks(task, case, tmp_path):
    module, s, config, executor, controller = repaired_start_fixture(task)
    if case == "missing_evidence":
        del s.category_interaction_setup
    elif case == "setup_rejected":
        s.category_interaction_setup["accepted"] = False
    elif case == "wrong_requested_region":
        s.category_interaction_setup["spawn_grid"]["requested_source_region"] = "target_0"
    elif case == "reported_no_floor":
        s.category_interaction_setup["robot_spawn"]["checks"]["floor_supported"] = False
    elif case == "reported_collision":
        s.category_interaction_setup["robot_spawn"]["checks"]["collision_free"] = False
    elif case == "missing_quaternion":
        del s.category_interaction_setup["robot_spawn"]["robot_quaternion_xyzw"]
    elif case == "stale_position":
        s.robot.position[0] += .1
    elif case == "stale_orientation":
        s.robot.quaternion = np.array([0., 0., 0., 1.])
    elif case == "wrong_native_room":
        s.robot.position[0] = 1.5
        s.category_interaction_setup["robot_spawn"]["robot_position_m"] = s.robot.position.tolist()
    elif case == "native_unsupported":
        s.robot.supported = False
    elif case == "native_collision":
        s.robot.colliding = True
    elif case == "wrong_contact_floor":
        s.robot.contact_z = 2.
    assert not module.qualify_operation(s, executor, config, {})["accepted"]
    with pytest.raises(ValueError):
        module.create_evaluator(s, controller, config, tmp_path).begin_episode(NS(snapshot_id="cpu", action_step=0))


@pytest.mark.parametrize("task", ["navigation", "passage"])
def test_original_exact_start_is_still_accepted_without_any_repair_evidence(task):
    module, s, config, executor, _ = repaired_start_fixture(task)
    del s.category_interaction_setup
    s.robot.position = np.array([0., 0., .1])
    s.robot.quaternion = np.array([0., 0., 0., 1.])
    result = module.qualify_operation(s, executor, config, {})
    assert result["accepted"] and result["checks"]["original_start_matches"]
    assert result["checks"]["start_contract"] == "original_exact_pose"


@pytest.mark.parametrize("task", ["navigation", "passage"])
def test_verified_source_room_repair_can_report_a_mismatched_original_map_point(task):
    module, s, config, executor, _ = repaired_start_fixture(task)
    s.env.scene.seg_map.get_room_instance_by_point = lambda xy: (
        "wrong_original_0" if abs(float(xy[0])) < 1e-6 else "source_0" if float(xy[0]) < 1 else "target_0")
    result = module.qualify_operation(s, executor, config, {})
    assert result["accepted"] and not result["checks"]["original_start_matches"]
    assert result["checks"]["start_contract"] == "verified_repaired_pose_same_source_region"


def test_fixed_root_native_r1_requires_a_real_holonomic_base_controller(tmp_path):
    module, s, config, executor, controller = repaired_start_fixture("navigation")
    result = module.qualify_operation(s, executor, config, {})
    assert s.robot.fixed_base and result["accepted"]
    assert result["checks"]["native_fixed_base"] is True
    assert result["checks"]["native_holonomic_base"] is True
    s.robot.controllers = {}
    rejected = module.qualify_operation(s, executor, config, {})
    assert not rejected["accepted"] and rejected["checks"]["native_holonomic_base"] is False
    with pytest.raises(ValueError):
        module.create_evaluator(s, controller, config, tmp_path).begin_episode(NS(snapshot_id="cpu", action_step=0))

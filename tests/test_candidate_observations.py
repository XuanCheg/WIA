"""Private synchronized-view gates. Run: PYTHONPATH=src pytest -q tests/test_candidate_observations.py."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from wia.candidates.observations import CandidateObservationEvidence


class Sensor:
    def __init__(self, rgb, segmentation=None, labels=None, *, mounted=True):
        self.prim_path = "/World/robot/head/camera" if mounted else "/World/global_camera"
        self.native = {"rgb": rgb.copy()}
        self.info = {}
        if segmentation is not None:
            self.native["seg_instance"] = segmentation.copy()
            self.info["seg_instance"] = labels or {0: "background"}
        self.calls = 0

    def get_obs(self):
        self.calls += 1
        return self.native, self.info


def inputs(prototype, contract, *, labels=None, segmentation=None, rgb=None, objects=None, regions=None):
    rgb = np.full((32, 32, 3), 170, dtype=np.uint8) if rgb is None else rgb
    segmentation = np.zeros((32, 32), dtype=np.int32) if segmentation is None else segmentation
    sensor = Sensor(rgb, segmentation, labels)
    session = SimpleNamespace(robot_camera_registry={"agentview_rgb": "head_sensor"},
        robot=SimpleNamespace(prim_path="/World/robot", sensors={"head_sensor": sensor}),
        candidate_state={"observed": [], "events": []})
    observation = SimpleNamespace(rgb_images=lambda: {"agentview_rgb": rgb.copy()})
    frame = {"time_s": 1., "objects": objects or {}, "regions": regions or {}, "observed": [], "events": []}
    parameters = {"prototype": prototype, "contract": contract,
                  "objects": [{"name": name} for name in frame["objects"]]}
    return CandidateObservationEvidence(session, parameters), observation, frame, sensor


def test_exact_delivered_rgb_matching_is_required():
    seg = np.ones((32, 32), dtype=np.int32)
    collector, observation, frame, sensor = inputs("M1", {}, labels={1: "tool_a"},
                                                  segmentation=seg, objects={"tool_a": {}})
    sensor.native["rgb"][0, 0, 0] += 1
    receipt = collector.record(observation, frame)
    assert receipt["status"] == "unavailable"
    assert not receipt["visible_names"]
    assert not receipt["observed_tags"]
    assert "exactly match" in receipt["per_view"]["agentview_rgb"]["errors"][0]


@pytest.mark.parametrize("missing", ["segmentation", "labels", "rgb"])
def test_missing_native_channels_remain_unavailable(missing):
    collector, observation, frame, sensor = inputs("H2", {})
    if missing == "segmentation":
        del sensor.native["seg_instance"]
    elif missing == "labels":
        del sensor.info["seg_instance"]
    else:
        del sensor.native["rgb"]
    receipt = collector.record(observation, frame)
    assert receipt["status"] == "unavailable"
    assert receipt["visible_names"] == receipt["observed_tags"] == []


def test_only_delivered_registered_mounted_cameras_are_read():
    collector, observation, frame, sensor = inputs("H2", {})
    never_delivered = Sensor(sensor.native["rgb"], sensor.native["seg_instance"])
    collector.session.robot_camera_registry["rear_rgb"] = "rear_sensor"
    collector.session.robot.sensors["rear_sensor"] = never_delivered
    rgb = sensor.native["rgb"].copy()
    observation.rgb_images = lambda: {"agentview_rgb": rgb, "global_scene": rgb}
    receipt = collector.record(observation, frame)
    assert sensor.calls == 1 and never_delivered.calls == 0
    assert receipt["per_view"]["global_scene"]["status"] == "ignored"
    sensor.prim_path = "/World/global_camera"
    receipt = collector.record(observation, frame)
    assert receipt["status"] == "unavailable"
    assert sensor.calls == 1


@pytest.mark.parametrize("height,width,visible", [(8, 8, True), (7, 9, False), (4, 16, False)])
def test_visibility_requires_both_pixel_count_and_extent(height, width, visible):
    seg = np.zeros((32, 32), dtype=np.int32)
    seg[1:1 + height, 1:1 + width] = 1
    collector, observation, frame, _ = inputs("H2", {}, labels={1: "zone_work"}, segmentation=seg,
                                            regions={"work": {"aabb": [[0, 0, 0], [1, 1, 0]]}})
    receipt = collector.record(observation, frame)
    assert ("zone_work" in receipt["visible_names"]) is visible
    assert receipt["observed_tags"] == []


def h1_inputs(marker_color=None, marker_pixels=16, *, outside_mask=False):
    rgb = np.full((32, 32, 3), [180, 175, 165], dtype=np.uint8)
    seg = np.zeros((32, 32), dtype=np.int32)
    seg[4:20, 4:20] = 1
    if marker_color is not None:
        coordinates = [(y, x) for y in range(0 if outside_mask else 4, 4 if outside_mask else 8)
                       for x in range(4, 8)]
        for y, x in coordinates[:marker_pixels]:
            rgb[y, x] = marker_color
    return inputs("H1", {"selected": "package"}, rgb=rgb, segmentation=seg,
                  labels={1: "package"}, objects={"package": {}})


def test_h1_body_visibility_does_not_imply_marker_inspection():
    collector, observation, frame, _ = h1_inputs()
    receipt = collector.record(observation, frame)
    assert receipt["visible_names"] == ["package"]
    assert receipt["observed_tags"] == []
    marker = receipt["per_view"]["agentview_rgb"]["evidence"]["package"]["authored_marker_color_test"]
    assert marker["method"] == "limited_authored_marker_color_test"
    assert not marker["general_semantic_recognition_certified"]


@pytest.mark.parametrize("color", [[230, 10, 10], [10, 190, 15]])
def test_h1_marker_requires_sixteen_saturated_pixels_inside_selected_mask(color):
    collector, observation, frame, _ = h1_inputs(color)
    collector.contract["defective"] = color[1] > color[0]
    assert collector.record(observation, frame)["observed_tags"] == ["inspection:package"]
    collector, observation, frame, _ = h1_inputs(color, marker_pixels=15)
    assert collector.record(observation, frame)["observed_tags"] == []
    collector, observation, frame, _ = h1_inputs(color, outside_mask=True)
    assert collector.record(observation, frame)["observed_tags"] == []


def test_m3_visible_fridge_exterior_does_not_reveal_its_contents():
    seg = np.zeros((32, 32), dtype=np.int32)
    seg[2:18, 2:18] = 1
    collector, observation, frame, _ = inputs("M3", {"content_by_unit": {"unit_a": "resident"}},
        labels={1: "unit_a", 2: "resident"}, segmentation=seg,
        objects={"unit_a": {}, "resident": {"inside": ["unit_a"]}})
    receipt = collector.record(observation, frame)
    assert receipt["visible_names"] == ["unit_a"]
    assert receipt["observed_tags"] == []


def test_m3_content_itself_must_be_visible_and_actually_inside_that_unit():
    seg = np.zeros((32, 32), dtype=np.int32)
    seg[4:12, 4:12] = 2
    collector, observation, frame, _ = inputs("M3", {"content_by_unit": {"unit_a": "resident"}},
        labels={2: "resident"}, segmentation=seg,
        objects={"unit_a": {}, "resident": {"inside": ["unit_a"]}})
    assert collector.record(observation, frame)["observed_tags"] == ["contents:unit_a"]
    frame["objects"]["resident"]["inside"] = ["unit_b"]
    assert collector.record(observation, frame)["observed_tags"] == []
    del frame["objects"]["resident"]["inside"]
    receipt = collector.record(observation, frame)
    assert receipt["status"] == "unavailable" and receipt["observed_tags"] == []


def h5_inputs(*, occupied, resident_visible, zone_visible=True):
    seg = np.zeros((32, 32), dtype=np.int32)
    if zone_visible:
        seg[2:10, 2:10] = 1
    if resident_visible:
        seg[15:23, 15:23] = 2
    x = .1 if occupied else 2.
    return inputs("H5", {"primary": "primary", "resident": "resident", "primary_available": not occupied},
        labels={1: "zone_primary", 2: "resident"}, segmentation=seg,
        objects={"resident": {"aabb": [[x, .1, 0], [x + .2, .3, .2]]}},
        regions={"primary": {"aabb": [[0, 0, 0], [1, 1, 0]], "support": "table"}})


def test_h5_destination_requires_actual_primary_zone_view():
    collector, observation, frame, _ = h5_inputs(occupied=False, resident_visible=False, zone_visible=False)
    assert collector.record(observation, frame)["observed_tags"] == []
    collector, observation, frame, _ = h5_inputs(occupied=False, resident_visible=False)
    assert collector.record(observation, frame)["observed_tags"] == ["destination_state"]


def test_h5_occupant_visibility_is_required_when_it_overlaps_primary():
    collector, observation, frame, _ = h5_inputs(occupied=True, resident_visible=False)
    assert collector.record(observation, frame)["observed_tags"] == []
    collector, observation, frame, _ = h5_inputs(occupied=True, resident_visible=True)
    assert collector.record(observation, frame)["observed_tags"] == ["destination_state"]


def test_native_instance_paths_match_exact_objects_without_empty_path_aliasing():
    seg = np.ones((32, 32), dtype=np.int32)
    collector, observation, frame, _ = inputs("M1", {}, labels={1: "/World/objects/tool_a/base_link"},
        segmentation=seg, objects={"tool_a": {"native_id": "/World/objects/tool_a"}, "tool_b": {}})
    receipt = collector.record(observation, frame)
    assert receipt["visible_names"] == ["tool_a"]
    assert receipt["observed_tags"] == []


def test_unresolved_path_and_semantic_category_are_not_object_identity():
    seg = np.ones((32, 32), dtype=np.int32)
    collector, observation, frame, sensor = inputs("M1", {}, labels={1: "None"},
                                                  segmentation=seg, objects={"tool_a": {}})
    assert collector.record(observation, frame)["visible_names"] == []
    sensor.info["seg_instance"] = {1: {"class": "tool_a"}}
    assert collector.record(observation, frame)["visible_names"] == []


def test_collector_does_not_mutate_observation_frame_or_shared_candidate_state():
    collector, observation, frame, sensor = h1_inputs([10, 190, 15])
    original_frame = deepcopy(frame)
    original_state = deepcopy(collector.session.candidate_state)
    original_rgb = sensor.native["rgb"].copy()
    receipt = collector.record(observation, frame)
    assert frame == original_frame
    assert collector.session.candidate_state == original_state
    np.testing.assert_array_equal(sensor.native["rgb"], original_rgb)
    assert receipt["policy_history_delivery_certified"] is False

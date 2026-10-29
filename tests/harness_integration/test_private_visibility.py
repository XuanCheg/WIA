"""Setup acceptance is private and later diagnostics cannot interrupt execution."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.action_interface import action_result
from wia.native.esi_visibility import PrivateVisibilityCheck, mask_metrics, passes_visibility
from harness_integration.test_agent_observation import source_and_rig


class Sensor:
    def __init__(self, target_visible):
        self.modalities = {"rgb", "depth_linear"}
        self.mask = np.zeros((24, 24), dtype=np.int32)
        if target_visible:
            self.mask[5:15, 5:15] = 7
        self.fingers = np.zeros_like(self.mask)
        self.fingers[16:20, 5:15] = 8

    def add_modality(self, name):
        self.modalities.add(name)

    def remove_modality(self, name):
        self.modalities.remove(name)

    def get_obs(self):
        rgb = np.zeros((24, 24, 4), dtype=np.uint8)
        rgb[self.mask == 7, :3] = 200
        return ({"rgb": rgb, "seg_instance": self.mask, "seg_instance_id": self.fingers},
                {"seg_instance": {7: "private_target"},
                 "seg_instance_id": {8: "/robot/left_finger/visuals"}})


def checker(tmp_path, head_visible=True):
    target = SimpleNamespace(name="private_target")
    sensors = {"head": Sensor(head_visible), "wrist": Sensor(False)}
    session = SimpleNamespace(scene_name="scene", room_name="room", camera_name="head", wrist_camera_name="wrist",
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda *args: target)),
        robot=SimpleNamespace(sensors=sensors, finger_links={"left": [SimpleNamespace(prim_path="/robot/left_finger")]}),
        pipeline=SimpleNamespace(og=SimpleNamespace(sim=SimpleNamespace(render=lambda: None))))
    return PrivateVisibilityCheck(session, {"scene": "scene", "room": "room", "target": "private_target"}, tmp_path)


def test_initial_gate_rejects_invisible_target_and_restores_sensor_modalities(tmp_path):
    check = checker(tmp_path, head_visible=False)
    with pytest.raises(RuntimeError, match="Initial target visibility"):
        check.check_initial()
    record = json.loads((tmp_path / "trajectory.jsonl").read_text())
    assert not record["initial_head_accepted"]
    for sensor in check.session.robot.sensors.values():
        assert sensor.modalities == {"rgb", "depth_linear"}


def test_initial_head_gate_does_not_require_target_already_under_wrist(tmp_path):
    record = checker(tmp_path).check_initial()
    assert record["initial_head_accepted"]
    assert not record["wrist_target_and_fingers_visible"]
    assert record["views"]["head"]["visible_pixels"] == 100


def test_private_grasp_diagnostic_does_not_gate_or_add_public_fields(tmp_path):
    source, _ = source_and_rig()
    check = checker(tmp_path)
    source.session.private_visibility_checker = check
    source.begin_episode("Pick up the object.")
    source.after_action(action_result("GRASP", completed=True, timeout=False, requested={}, actual={}, message="completed"))
    observation = source.agent_observation()
    private = json.loads((tmp_path / "trajectory.jsonl").read_text())
    assert not private["wrist_target_and_fingers_visible"]
    assert observation.action_step == private["action_step"] == 1
    assert "private_target" not in json.dumps(observation.to_payload())
    assert "passes_pixel_gate" not in json.dumps(observation.to_payload())


def test_grasp_diagnostic_error_keeps_the_completed_public_receipt(tmp_path):
    source, _ = source_and_rig()
    check = checker(tmp_path)
    check.session.robot.sensors["head"].get_obs = lambda: (_ for _ in ()).throw(RuntimeError("annotation unavailable"))
    source.session.private_visibility_checker = check
    source.begin_episode("Pick up the object.")
    source.after_action(action_result("GRASP", completed=True, timeout=False, requested={}, actual={}, message="completed"))
    assert source.agent_observation().last_action_feedback.execution_status == "completed"
    assert "annotation unavailable" in json.loads((tmp_path / "errors.jsonl").read_text())["error"]
    assert check.session.robot.sensors["head"].modalities == {"rgb", "depth_linear"}


def test_border_clipping_and_single_pixel_slivers_do_not_pass():
    mask = np.zeros((24, 24), dtype=bool)
    mask[0:10, 0:10] = True
    rgb = np.zeros((24, 24, 3), dtype=np.uint8)
    assert not passes_visibility(mask_metrics(mask, rgb), minimum_pixels=64, minimum_extent_px=8)
    mask[:] = False
    mask[1:23, 1] = True
    assert not passes_visibility(mask_metrics(mask, rgb), minimum_pixels=16, minimum_extent_px=8)


def test_unready_depth_renders_again_before_publishing_one_coherent_boundary():
    source, _ = source_and_rig()
    observe = source.session.observe
    count = []

    def native(**kwargs):
        count.append(1)
        raw = observe(**kwargs)
        raw["head_depth_m"] = np.full(raw["head_rgb"].shape[:2], -1 if len(count) == 1 else 1, dtype=np.float32)
        return raw

    source.session.observe = native
    observation = source.capture(action_boundary=True).observation
    assert len(count) == 2 and observation.snapshot_id == 1
    assert np.all(observation.sensors.head_depth_m == 1)


def test_persistent_invalid_depth_is_rejected_without_publishing_a_boundary():
    source, _ = source_and_rig()
    observe = source.session.observe

    def native(**kwargs):
        raw = observe(**kwargs)
        raw["head_depth_m"] = np.full(raw["head_rgb"].shape[:2], -1, dtype=np.float32)
        return raw

    source.session.observe = native
    with pytest.raises(RuntimeError, match="did not become ready"):
        source.capture(action_boundary=True)
    assert source.snapshot_id == 0 and source.boundary_observation is None


def test_empty_rgb_buffer_renders_again_and_does_not_weaken_format_validation():
    from wia.runtime.sim.esi_omnigibson import ESISession, NativeCameraNotReady
    source, _ = source_and_rig()
    observe = source.session.observe
    count = []

    def native(**kwargs):
        count.append(1)
        if len(count) == 1:
            ESISession._rgb_from_data({'rgb': np.array([])}, 'head')
        return observe(**kwargs)

    source.session.observe = native
    assert source.capture(action_boundary=True).observation.snapshot_id == 1
    assert len(count) == 2
    with pytest.raises(ValueError, match='Unexpected camera RGB format'):
        ESISession._rgb_from_data({'rgb': np.zeros((2, 2, 5))}, 'head')
    with pytest.raises(NativeCameraNotReady):
        ESISession._rgb_from_data({'rgb': np.array([])}, 'head')

from copy import deepcopy

import pytest

from wia.runtime.sim.esi_omnigibson import ESISession
from wia.runtime.commands.run_esi_robot import DEFAULT_PREVIEW_CONFIG, observation_settings, parse_args


def test_observation_setup_preserves_esi_scene_objects_and_controllers():
    config = {
        "scene": {"scene_model": "Beechwood_0_garden", "load_room_instances": ["dining_room_0"]},
        "objects": [{"name": "obj_fixed", "position": [-6, 15, 0.1]}],
        "task": {"type": "DummyTask"},
        "robots": [{"type": "R1", "reset_joint_pos": [0, 1, 2],
                    "controller_config": {"base": {"name": "HolonomicBaseJointController"}},
                    "sensor_config": {"VisionSensor": {"modalities": ["rgb"],
                                      "sensor_kwargs": {"image_width": 128, "image_height": 128}}}}],
    }
    original = deepcopy(config)
    ESISession()._configure_robot_observation(config)
    assert config == original  # Original single-frame API retains upstream defaults.
    session = ESISession(robot_position=(-6.45, 16.9, 0.1), robot_yaw_deg=-90,
                         camera_size=(512, 512), head_pitch_deg=45)
    session._configure_robot_observation(config)
    assert {key: config[key] for key in ("scene", "objects", "task")} == {
        key: original[key] for key in ("scene", "objects", "task")
    }
    robot = config["robots"][0]
    for key in ("type", "reset_joint_pos", "controller_config"):
        assert robot[key] == original["robots"][0][key]
    assert robot["sensor_config"]["VisionSensor"]["modalities"] == ["rgb"]
    assert robot["position"] == [-6.45, 16.9, 0.1] and robot["pose_frame"] == "world"


def test_manual_preview_preset_is_bound_to_its_question(tmp_path):
    settings, preset = observation_settings(parse_args(["--web"]))
    assert preset == DEFAULT_PREVIEW_CONFIG.resolve() and "robot_position" in settings
    settings, preset = observation_settings(parse_args([]))
    assert settings == {} and preset is None
    settings, preset = observation_settings(parse_args(["--web", "--upstream-observation"]))
    assert settings == {} and preset is None
    other_question = tmp_path / "different-question.json"
    settings, preset = observation_settings(parse_args(["--web", "--metadata", str(other_question)]))
    assert preset is None and settings == {"camera_size": (512, 512)}
    with pytest.raises(ValueError, match="different question"):
        observation_settings(parse_args(["--metadata", str(other_question),
                                         "--preview-config", str(DEFAULT_PREVIEW_CONFIG)]))
    settings, _ = observation_settings(parse_args(["--web", "--robot-position", "1", "2", "3",
                                                  "--head-pitch-deg", "0"]))
    assert settings["robot_position"] == [1, 2, 3] and settings["head_pitch_deg"] == 0


def test_assisted_grasp_override_changes_only_robot_grasp_settings():
    config = {"robots": [{"type": "R1", "grasping_mode": "sticky", "disable_grasp_handling": True}],
              "objects": [{"name": "target", "position": [1, 2, 0.75]}], "scene": {"scene_model": "original"}}
    original = deepcopy(config)
    session = ESISession(grasp_mode="assisted")
    session._configure_robot_observation(config)
    assert session.original_grasp_mode == "sticky"
    assert config["robots"][0]["grasping_mode"] == "assisted"
    assert not config["robots"][0]["disable_grasp_handling"]
    assert config["objects"] == original["objects"] and config["scene"] == original["scene"]


def test_manipulation_demo_uses_its_own_bound_question():
    args = parse_args(["--validate-manipulation", "--gripper-control", "--web"])
    settings, preset = observation_settings(args)
    assert "Material Transparency" in str(args.metadata)
    assert settings["robot_yaw_deg"] == 90 and settings["robot_position"] == [-4.71, 15.15, 0.1]
    assert preset.name == "esi_manipulation_demo.json"

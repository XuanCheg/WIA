#!/usr/bin/env python3
"""ESI R1 observations, read-only preview and opt-in base/left-arm control."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wia.runtime.sim.esi_omnigibson import DEFAULT_QUESTION, ESISession

DEFAULT_PREVIEW_CONFIG = ROOT / "configs" / "esi_preview.json"
DEFAULT_MANIPULATION_DEMO = ROOT / "configs" / "esi_manipulation_demo.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an ESI scene and its R1; observations or opt-in base/left-arm control, without a policy.",
        epilog="Run inside your BEHAVIOR environment on a GPU. Default canonical question: "
               f"{DEFAULT_QUESTION.relative_to(ROOT)}",
    )
    parser.add_argument("--gpu-info", type=Path, help="Owned local GPU renderer/CUDA mapping and private cache provenance.")
    parser.add_argument("--debug-retain", action="store_true", help="Pause the world after an agent lease ends for debug resume.")
    parser.add_argument("--run-mode", choices=("benchmark", "diagnostic"), default="benchmark")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_QUESTION,
                        help="Question JSON or ESI metadata containing json_paths (default: smoke sample).")
    parser.add_argument("--task", default=None, help="ESI task name; inferred from canonical question JSON.")
    parser.add_argument("--question-index", type=int, default=0)
    parser.add_argument("--json-root", type=Path, default=None, help="Root for metadata's relative question paths.")
    parser.add_argument("--camera", default=None, help="Exact robot RGB sensor name; default selects head/main camera.")
    parser.add_argument("--robot-position", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Explicit initial R1 world position; does not move task objects.")
    parser.add_argument("--robot-yaw-deg", type=float, default=None,
                        help="Initial R1 world yaw in degrees (0 faces +X).")
    parser.add_argument("--camera-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
                        help="Native robot RGB sensor resolution; eight-camera default: 1024 x 1024.")
    parser.add_argument("--robot-camera-config", type=Path, default=ROOT / "configs" / "esi_robot_cameras.json",
                        help="Eight robot-mounted cameras and native render quality settings.")
    parser.add_argument("--native-camera-pair", action="store_true",
                        help="Use the original head/left-wrist projection instead of the eight-view robot setup.")
    parser.add_argument("--head-pitch-deg", type=float, default=None,
                        help="Fixed optical tilt of the selected mounted camera; positive looks down.")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--preview-config", type=Path, default=None,
                         help="Manual observation JSON preset, bound to one question. CLI values take priority.")
    preview.add_argument("--upstream-observation", action="store_true",
                         help="Skip web sample defaults; retain upstream pose/sensors unless overridden.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: outputs/esi_debug/<UTC run-id>.")
    parser.add_argument("--capture-agent-observation", action="store_true",
                        help="Export final allowlisted AgentObservation JSON plus all enabled native RGB PNGs; no model.")
    parser.add_argument("--evaluation-config", type=Path,
                        help="Private scene-bound grasp/lift evaluator; writes local files, never Agent/MCP state.")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--explore-config", type=Path, help="External initialization-only exploration scene.")
    tasks.add_argument("--task-file", type=Path, help="Declarative ESI-backed physical task YAML.")
    tasks.add_argument("--scene-file", type=Path,
                       help="Scene-only YAML; preserve the ESI question and create no evaluator.")
    tasks.add_argument("--action-order-tabletop-config", type=Path,
                       help="Initialize the original Action Order row on an existing native table; preserve scales and QA.")
    parser.add_argument("--variant", default="original")
    parser.add_argument("--data-root", type=Path, help="Optional external ESI json_clean root.")
    parser.add_argument("--episode-manifest", type=Path, help="Immutable coordinator/Agent episode manifest.")
    parser.add_argument("--scene-seed", type=int, default=None)
    parser.add_argument("--visibility-config", type=Path,
                        help="Private initial target visibility gate and post-GRASP view diagnostics; never Agent/MCP input.")
    parser.add_argument("--motion-diagnostics", action="store_true",
                        help="Save private contact/IK/joint-margin evidence at motion faults; no model hints.")
    parser.add_argument("--agent-observation-config", type=Path,
                        default=ROOT / "configs" / "esi_agent_observation.json",
                        help="Visibility switches; enabled inputs remain unavailable without real sensor adapters.")
    parser.add_argument("--web", action="store_true", help="Keep GUMI open with live cameras; read-only unless control is enabled.")
    parser.add_argument("--standby-until-agent", action="store_true",
                        help="Retain a preloaded scene without idle physics while human-owned; normal world policy resumes when an Agent takes control.")
    parser.add_argument("--base-control", action="store_true",
                        help="Enable only BASE_* controls in --web; arm/gripper remain disabled.")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--base-debug", action="store_true",
                            help="Print runtime controllers and test hold/translation/rotation directly, without tokens.")
    diagnostic.add_argument("--validate-base", action="store_true",
                            help="Run the Milestone 2 fixed-token acceptance sequence and save results/images.")
    parser.add_argument("--arm-control", action="store_true",
                        help="Enable base plus six left-arm EE translations in --web; gripper stays disabled.")
    parser.add_argument("--gripper-control", action="store_true",
                        help="Enable BASE/MV/GRASP/RELEASE with native assisted grasp in --web.")
    parser.add_argument("--rotation-control", action="store_true",
                        help="Enable local-axis wrist yaw, pitch and roll with arm control in --web.")
    parser.add_argument("--trunk-control", action="store_true",
                        help="Enable fixed upper-body TRUNK_UP/DOWN macros with arm control.")
    parser.add_argument("--trunk-config", type=Path, default=ROOT / "configs" / "esi_trunk.json")
    parser.add_argument("--rotation-config", type=Path, default=ROOT / "configs" / "esi_rotation.json",
                        help="Fixed wrist rotation step, tolerances and speed limits.")
    diagnostic.add_argument("--rotation-debug", action="store_true",
                            help="Test direct clockwise/counterclockwise local-Z wrist rotations.")
    diagnostic.add_argument("--validate-rotation", action="store_true",
                            help="Validate fixed wrist rotations, position hold, base/arm alternation and cameras.")
    diagnostic.add_argument("--gripper-debug", action="store_true",
                            help="Test measured open/empty-close/idempotent gripper macros directly.")
    diagnostic.add_argument("--validate-manipulation", action="store_true",
                            help="Run the question-bound scripted assisted pick/carry/place demo and record videos.")
    parser.add_argument("--demo-config", type=Path, default=DEFAULT_MANIPULATION_DEMO,
                        help="Question, manual robot placement and fixed demonstration actions.")
    parser.add_argument("--gripper-config", type=Path, default=ROOT / "configs" / "esi_gripper.json")
    diagnostic.add_argument("--arm-debug", action="store_true",
                            help="Test direct move_ee_debug along +/-X/Y/Z without MV tokens.")
    diagnostic.add_argument("--validate-arm", action="store_true",
                            help="Run Milestone 3 six-direction, yaw-frame and base/arm acceptance checks.")
    diagnostic.add_argument("--validate-camera-geometry", action="store_true",
                            help="Private native camera/FK/depth acceptance plus six working-arm directions.")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs" / "esi_base.json",
                        help="Base macro distances, tolerances, speed limits and timeouts.")
    parser.add_argument("--disable-right-arm-hold-check", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--arm-config", type=Path, default=ROOT / "configs" / "esi_arm.json",
                        help="Working-arm step, tolerances, speed limit and timeouts.")
    parser.add_argument("--primitive-profile", choices=("fixed", "fine"), default="fixed",
                        help="Opt-in fine increments; usable in both fixed and parameterized experiments.")
    parser.add_argument("--primitive-profile-config", type=Path,
                        default=ROOT / "configs" / "esi_primitives_fine.json")
    parser.add_argument("--parameterized-actions", action="store_true",
                        help="Enable bounded Agent metric commands compiled into the existing executor.")
    parser.add_argument("--macro-config", type=Path, default=None,
                        help="Metric-command limits; defaults to the matching fixed/fine profile's macro configuration.")
    parser.add_argument("--host", default="127.0.0.1", help="Web bind address (use SSH/VS Code port forwarding).")
    parser.add_argument("--port", type=int, default=8615, help="GUMI preview HTTP port.")
    parser.add_argument("--viewer-config", type=Path, default=ROOT / "configs" / "esi_viewer.json",
                        help="Human camera layout, depth scale and compressed recording settings.")
    parser.add_argument("--recording-dir", type=Path, default=None,
                        help="Saved human videos (default: --output-dir/recordings).")
    parser.add_argument("--no-debug-cameras", "--no-human-viewer", dest="no_debug_cameras", action="store_true",
                        help="Show only the mounted R1 cameras in GUMI.")
    parser.add_argument("--capture-fps", type=float, default=10.0)
    parser.add_argument("--performance-profile", choices=("debug", "benchmark"), default="debug")
    parser.add_argument("--macro-execution", choices=("primitive_replay", "metric_target"), default="primitive_replay")
    parser.add_argument("--observation-pipeline", choices=("legacy", "optimized"), default="optimized")
    parser.add_argument("--snapshot-cache", choices=("legacy", "optimized"), default="optimized")
    parser.add_argument("--preview-fps", type=float, default=None)
    parser.add_argument("--human-preview", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--human-recording", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug-cameras", action="store_true", default=None)
    parser.add_argument("--human-depth-visualizations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--recording-hz", type=float, default=5.0,
                        help="Video captures per simulation second while recording (default: 5 Hz).")
    parser.add_argument("--recording-mode", choices=("constant_fps", "snapshots"), default="snapshots")
    parser.add_argument("--recording-views", nargs="+", default=None)
    parser.add_argument("--main-view", choices=("head",), default="head",
                        help="Legacy mounted-camera preview starts with R1 head; grouped viewer uses --viewer-config.")
    parser.add_argument("--web-duration-s", type=float, default=None,
                        help="Optional preview duration for smoke tests; otherwise run until Ctrl+C.")
    args = parser.parse_args(argv)
    debug = args.performance_profile == "debug"
    if not finite_nonnegative(args.recording_hz) or args.recording_hz == 0:
        parser.error("Recording frequency must be finite and positive.")
    if args.human_recording is None:
        args.human_recording = args.run_mode != "benchmark"
    for name in ("human_preview", "human_recording", "debug_cameras", "human_depth_visualizations"):
        if getattr(args, name) is None:
            setattr(args, name, debug)
    if args.no_debug_cameras:
        args.debug_cameras = False
    if args.preview_fps is None:
        args.preview_fps = args.capture_fps if args.human_preview else 0.
    if not finite_nonnegative(args.preview_fps):
        parser.error("Preview FPS must be finite and nonnegative.")
    if args.macro_config is None:
        args.macro_config = ROOT / "configs" / (
            "esi_parameterized_fine.json" if args.primitive_profile == "fine" else "esi_parameterized.json")
    if args.validate_manipulation:
        demo = json.loads(args.demo_config.read_text())
        question = (ROOT / demo["metadata"]).resolve()
        if args.metadata.resolve() not in (DEFAULT_QUESTION.resolve(), question):
            parser.error("The scripted demo belongs to its configured question.")
        args.metadata = question
        args.preview_config = args.demo_config
    return args


def finite_nonnegative(value):
    import math
    return math.isfinite(value) and value >= 0


def observation_settings(args: argparse.Namespace) -> tuple[dict, Path | None]:
    """Select a manual preset only for its exact question; never search for a spawn."""
    metadata = args.metadata.expanduser().resolve()
    preset_path = args.preview_config
    base_enabled = (args.base_control or args.base_debug or args.validate_base
                    or args.arm_control or args.arm_debug or args.validate_arm
                    or args.rotation_control or args.rotation_debug or args.validate_rotation
                    or args.gripper_control or args.gripper_debug or args.validate_manipulation)
    if (preset_path is None and (args.web or base_enabled) and not args.upstream_observation
            and metadata == DEFAULT_QUESTION.resolve() and args.question_index == 0
            and args.task in (None, "action_order_inference")):
        preset_path = DEFAULT_PREVIEW_CONFIG
    settings = {"camera_size": (512, 512)} if (args.web or base_enabled) and not args.upstream_observation else {}
    if preset_path is not None:
        preset_path = preset_path.expanduser().resolve()
        with preset_path.open(encoding="utf-8") as stream:
            preset = json.load(stream)
        question = (ROOT / preset["metadata"]).resolve()
        if metadata != question or args.question_index != preset["question_index"]:
            raise ValueError("Preview preset belongs to a different question; supply that question or a matching preset.")
        for key in ("robot_position", "robot_yaw_deg", "camera_size", "head_pitch_deg"):
            if key in preset:
                settings[key] = preset[key]
    for key in ("robot_position", "robot_yaw_deg", "camera_size", "head_pitch_deg"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    return settings, preset_path


def run_base_checks(session, controller, output: Path, *, direct=False) -> None:
    """Explicit debug/acceptance run; privileged poses remain in debug outputs."""
    from core.record.images import save_png
    from wia.runtime.sim.esi_actions import create_r1_executor

    executor = create_r1_executor(controller)
    results = []
    before_hold = controller.pose()
    for _ in range(controller.config.settle_steps):
        controller.step_base_debug()
    hold = {"pose_before": before_hold, "pose_after": controller.pose(),
            "joint_positions": controller.robot.get_joint_positions().tolist()}
    (output / "hold.json").write_text(json.dumps(hold, indent=2) + "\n", encoding="utf-8")
    save_png(output / "before_head.png", session.observe()["head_rgb"])

    def check(result):
        results.append(result)
        if not result["completed"]:
            raise RuntimeError(f"Base validation failed: {result['action']}: {result['reason']}")

    if direct:
        check(controller.execute_base_translation("forward", controller.config.translation_m))
        check(controller.execute_base_rotation("left", controller.config.rotation_deg))
        save_png(output / "after_rotation_head.png", session.observe()["head_rgb"])
        check(controller.execute_base_translation("forward", controller.config.translation_m))
        save_png(output / "after_forward_head.png", session.observe()["head_rgb"])
    else:
        from core.action_units import BASE_ATOMS

        for token in BASE_ATOMS[:4]:
            check(executor.execute(token))
        save_png(output / "before_rotation_head.png", session.observe()["head_rgb"])
        for token in BASE_ATOMS[4:]:
            for _ in range(6):
                check(executor.execute(token))
            side = "left" if token == "BASE_TURN_LEFT" else "right"
            save_png(output / f"after_{side}_rotation_head.png", session.observe()["head_rgb"])
            check(executor.execute("BASE_FWD"))
            save_png(output / f"after_{side}_forward_head.png", session.observe()["head_rgb"])
    (output / "base_validation.json").write_text(
        json.dumps({"direct": direct, "completed": True, "hold": hold, "results": results}, indent=2)
        + "\n", encoding="utf-8")


def run_arm_checks(session, controller, output: Path, *, direct=False) -> None:
    """Canonical single-arm checks; pose data and scripted targets are debug only."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from core.action_units import MOVE_ATOMS
    from core.record.images import save_png
    from wia.runtime.sim.esi_actions import create_r1_executor

    executor = create_r1_executor(controller)
    results, coordinates, rotation_coordinates = [], [], []

    def check(result):
        results.append(result)
        if not result["completed"]:
            raise RuntimeError(f"Arm validation failed: {result['action']}: {result['reason']}")
        return result

    save_png(output / "before_arm_head.png", session.observe()["head_rgb"])
    save_png(output / "before_arm_wrist.png", session.observe(include_wrist=True)["wrist_rgb"])
    for axis in range(3):
        for sign in (1, -1):
            if direct:
                delta = np.zeros(3)
                delta[axis] = sign * controller.arm_config.translation_m
                check(controller.move_ee_debug(*delta))
            else:
                check(executor.execute(MOVE_ATOMS[2 * axis + (sign < 0)]))
    if not direct:
        # Verify an arm away from its initial pose survives base motion.
        initial = controller.ee_pose()
        for _ in range(3):
            check(executor.execute("MV_FWD"))
        displaced = controller.ee_pose()
        if np.linalg.norm(np.asarray(displaced["base_position"]) - initial["base_position"]) < controller.arm_config.translation_m:
            raise RuntimeError("Validation requires the working arm to be measurably away from home.")
        check(executor.execute("BASE_FWD"))
        check(executor.execute("BASE_BACK"))

        # Run the specified +90-degree sequence from the configured initial
        # heading, before coordinate diagnostics change it to 0 / 90 degrees.
        for _ in range(6):
            check(executor.execute("BASE_TURN_LEFT"))
        save_png(output / "after_arm_rotation_head.png", session.observe()["head_rgb"])
        for token in ("BASE_FWD", "MV_FWD", "MV_LEFT", "MV_DOWN", "MV_UP"):
            check(executor.execute(token))

        for desired_yaw in (0, 90):
            # Direct debug alignment uses base readback; subsequent public MV
            # tokens still use the configured fixed 2 cm displacement.
            yaw = np.rad2deg(controller.pose()["yaw_rad"])
            angle = float((desired_yaw - yaw + 180) % 360 - 180)
            if abs(angle) > controller.config.rotation_tolerance_deg:
                while abs(angle) >= 180:
                    check(controller.execute_base_rotation("left" if angle > 0 else "right", 90))
                    angle -= 90 if angle > 0 else -90
                check(controller.execute_base_rotation("left" if angle > 0 else "right", abs(angle)))
            base = controller.pose()
            result = check(executor.execute("MV_FWD"))
            expected = Rotation.from_quat(base["quaternion_xyzw"]).apply(
                [controller.arm_config.translation_m, 0, 0])
            actual = (np.asarray(result["ee_pose_after"]["world_position"])
                      - result["ee_pose_before"]["world_position"])
            frame_error = float(np.linalg.norm(actual - expected))
            if frame_error > controller.arm_config.translation_tolerance_m:
                raise RuntimeError("MV_FWD world readback disagrees with the current robot base frame.")
            coordinates.append({"desired_yaw_deg": desired_yaw,
                                "measured_yaw_deg": float(np.rad2deg(base["yaw_rad"])),
                                "expected_world_delta_m": expected.tolist(),
                                "actual_world_delta_m": actual.tolist(), "frame_error_m": frame_error})
            check(executor.execute("MV_BACK"))
            if controller.rotation_config is not None:
                # Exercise wrist yaw at the same verified 0/90-degree headings
                # as the M3 frame checks, with its clear manual robot placement.
                for token, sign in (("ROTATE_CW", 1), ("ROTATE_CCW", -1)):
                    result = check(executor.execute(token))
                    before = result["ee_pose_before"]
                    base_rotation = Rotation.from_quat(result["base_pose_before"]["quaternion_xyzw"])
                    expected = (base_rotation * Rotation.from_quat(before["base_quaternion_xyzw"])
                                * Rotation.from_euler("z", sign * controller.rotation_config.rotation_deg, degrees=True))
                    actual = Rotation.from_quat(result["ee_pose_after"]["world_quaternion_xyzw"])
                    error = float(np.rad2deg((expected.inv() * actual).magnitude()))
                    if error > controller.rotation_config.rotation_tolerance_deg:
                        raise RuntimeError("Wrist rotation disagrees with the local end-effector frame.")
                    rotation_coordinates.append({"desired_base_yaw_deg": desired_yaw, "action": token,
                                                 "world_orientation_error_deg": error,
                                                 "position_error_m": result["position_error_m"]})

    save_png(output / "after_arm_head.png", session.observe()["head_rgb"])
    save_png(output / "after_arm_wrist.png", session.observe(include_wrist=True)["wrist_rgb"])
    (output / "arm_validation.json").write_text(json.dumps(
        {"direct": direct, "completed": True, "results": results, "coordinate_checks": coordinates,
         "rotation_coordinate_checks": rotation_coordinates},
        indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None, *, initial_observation_capture=None) -> int:
    args = parse_args(argv)
    assigned_gpu = None
    if args.gpu_info is not None:
        assigned_gpu = json.loads(args.gpu_info.read_text())
        from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache
        install_gpu_selection(assigned_gpu)
        from omnigibson import simulator
        isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
    scene_selection = None
    task_selection = None
    supplied = {arg.split("=", 1)[0] for arg in (sys.argv[1:] if argv is None else argv)}
    if args.task_file is not None:
        from wia.tasks.loader import LoadedRigidTask, load_task
        task_selection = load_task(args.task_file, data_root=args.data_root,
            variant=args.variant if "--variant" in supplied else None, seed=args.scene_seed, run_mode=args.run_mode)
        args.variant = task_selection.variant
        if isinstance(task_selection, LoadedRigidTask):
            scene_selection = task_selection.scene
    if args.scene_file is not None:
        from wia.tasks.loader import load_scene
        if "--variant" not in supplied:
            args.variant = "tabletop"
        scene_selection = load_scene(args.scene_file, data_root=args.data_root,
                                     variant=args.variant, seed=args.scene_seed, run_mode=args.run_mode)
    if scene_selection is not None:
        runtime = scene_selection.definition.runtime
        for name in ("trunk_config", "robot_camera_config", "viewer_config"):
            if "--" + name.replace("_", "-") not in supplied:
                setattr(args, name, ROOT / getattr(runtime, name))
        if "--camera-size" not in supplied:
            args.camera_size = list(runtime.camera_size)
        for name in ("debug_cameras", "human_depth_visualizations", "preview_fps"):
            flag = "--" + name.replace("_", "-")
            if flag not in supplied and "--no-" + flag[2:] not in supplied:
                setattr(args, name, getattr(runtime, name))
        if "--human-preview" not in supplied and "--no-human-preview" not in supplied:
            args.human_preview = args.preview_fps > 0
        if args.no_debug_cameras:
            args.debug_cameras = False
    gripper_enabled = args.gripper_control or args.gripper_debug or args.validate_manipulation
    rotation_enabled = args.rotation_control or args.rotation_debug or args.validate_rotation
    arm_enabled = (args.arm_control or args.arm_debug or args.validate_arm or args.validate_camera_geometry
                   or gripper_enabled or rotation_enabled or args.trunk_control)
    base_enabled = args.base_control or args.base_debug or args.validate_base or arm_enabled
    if args.parameterized_actions and not base_enabled:
        raise ValueError("Parameterized commands require an enabled robot control backend.")
    profile = json.loads(args.primitive_profile_config.read_text()) if args.primitive_profile == "fine" else {}
    manual_enabled = args.base_control or args.arm_control or args.gripper_control or args.rotation_control or args.trunk_control
    if manual_enabled and not (args.web or args.base_debug or args.validate_base or args.arm_debug or args.validate_arm
                              or args.gripper_debug or args.validate_manipulation
                              or args.rotation_debug or args.validate_rotation or args.validate_camera_geometry):
        raise ValueError("Manual control requires --web or a diagnostic flag.")
    base_config = None
    if base_enabled:
        from wia.runtime.sim.esi_base import BaseControlConfig

        base_config = BaseControlConfig.load(args.base_config)
        if profile:
            base_config = replace(base_config, **profile["base"])
    arm_config = None
    if arm_enabled:
        from wia.runtime.sim.esi_arm import ArmControlConfig

        arm_config = ArmControlConfig.load(args.arm_config)
        if profile:
            arm_config = replace(arm_config, **profile["arm"])
    gripper_config = None
    if gripper_enabled:
        from wia.runtime.sim.esi_gripper import GripperControlConfig

        gripper_config = GripperControlConfig.load(args.gripper_config)
    rotation_config = None
    if rotation_enabled:
        from wia.runtime.sim.esi_rotation import RotationControlConfig

        rotation_config = RotationControlConfig.load(args.rotation_config)
        if profile:
            rotation_config = replace(rotation_config, **profile["rotation"])
    if args.capture_fps <= 0 or (args.web_duration_s is not None and args.web_duration_s <= 0):
        raise ValueError("Capture FPS and optional web duration must be positive.")
    settings, preset_path = observation_settings(args)
    explore = None
    declarative = task = episode_manifest = scene_only = None
    qualification_finished = False
    if args.task_file is not None:
        from wia.tasks.loader import LoadedRigidTask, verify_manifest
        from wia.tasks.scene import DeclarativeScene
        if (args.episode_manifest is None or args.evaluation_config or args.visibility_config
                or args.preview_config or not gripper_enabled):
            raise ValueError("Declarative tasks require their episode manifest, gripper control and task-owned setup/evaluation.")
        task = task_selection
        episode_manifest = verify_manifest(args.episode_manifest, task)
        if not isinstance(task, LoadedRigidTask):
            declarative = DeclarativeScene(task)
            settings, preset_path = declarative.observation_settings(), None
        args.metadata = task.source_path
        import random
        import numpy as np
        import torch
        random.seed(task.seed)
        np.random.seed(task.seed)
        torch.manual_seed(task.seed)
    if scene_selection is not None:
        from wia.tasks.scene_only import SceneOnly
        if args.evaluation_config or args.visibility_config or args.preview_config:
            raise ValueError("Scene-only YAML preserves the source question and owns its initialization preflight.")
        scene_only = SceneOnly(scene_selection)
        args.metadata = scene_selection.source_path
        settings, preset_path = scene_only.observation_settings(), None
        settings["camera_size"] = args.camera_size
        import random
        import numpy as np
        import torch
        random.seed(scene_selection.seed)
        np.random.seed(scene_selection.seed)
        torch.manual_seed(scene_selection.seed)
    if args.explore_config is not None:
        from wia.runtime.sim.esi_explore_scenes import ExploreScene
        explore = ExploreScene.load(args.explore_config, seed=args.scene_seed)
        import random
        import numpy as np
        import torch
        random.seed(explore.seed)
        np.random.seed(explore.seed)
        torch.manual_seed(explore.seed)
        if not gripper_enabled or args.evaluation_config or args.visibility_config or args.preview_config:
            raise ValueError("Exploration scenes require gripper control and own their initialization/evaluation configs.")
        args.metadata = explore.metadata_path
        settings = explore.observation_settings()
        preset_path = None
    output = (args.output_dir or ROOT / "outputs" / "esi_debug" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "sim_pid.txt").write_text(str(os.getpid()) + "\n")
    tabletop = None
    if args.action_order_tabletop_config is not None:
        from wia.runtime.sim.esi_action_order_tabletop import ActionOrderTabletop
        tabletop = ActionOrderTabletop(args.action_order_tabletop_config, output)
    session = ESISession(
        args.metadata, task=args.task, question_index=args.question_index,
        json_root=args.json_root, camera=args.camera,
        grasp_mode=gripper_config.grasp_mode if gripper_config else None, **settings,
    )
    from core.observation_config import ObservationConfig
    session.explore_scene = explore
    session.standby_until_agent = args.standby_until_agent
    session.scene_adapter = declarative or scene_only or tabletop
    session.loaded_task = task
    if task is not None:
        session.public_task_text = task.task_text
    session.agent_observation_config = ObservationConfig.load(args.agent_observation_config)
    from core.viewer_config import ViewerConfig
    human_viewer = args.web and not args.no_debug_cameras and (args.human_preview or args.human_recording)
    viewer_config = ViewerConfig.load(args.viewer_config) if human_viewer else None
    if viewer_config:
        viewer_config = replace(viewer_config, recording_mode=args.recording_mode,
                                recording_views=tuple(args.recording_views or ()))
        if scene_selection is not None and "--viewer-config" not in supplied:
            viewer_config = replace(viewer_config, main_view=scene_selection.definition.runtime.viewer_main_view)
    from core.performance import PerformanceCounters
    session.performance = PerformanceCounters()
    session.reuse_step_observations = args.observation_pipeline == "optimized"
    session.serialization_cache = args.snapshot_cache == "optimized"
    recording_dir = args.recording_dir or output / "recordings"
    try:
        session.reset()
        if assigned_gpu is not None:
            from wia.runtime.scene_capture_adapters import gpu_evidence
            output.mkdir(parents=True, exist_ok=True)
            (output / 'native_gpu_evidence.json').write_text(json.dumps(gpu_evidence(session, assigned_gpu), indent=2) + '\n')
        if not args.native_camera_pair and (args.web or base_enabled or args.capture_agent_observation):
            from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras

            configure_robot_cameras(session, args.robot_camera_config, resolution=args.camera_size)
        if args.web and args.debug_cameras:
            session.configure_scene_camera()
            if human_viewer:
                session.configure_human_cameras(viewer_config)
        output.mkdir(parents=True, exist_ok=True)
        controller = None
        executor = None
        if base_enabled:
            def log_result(result):
                line = json.dumps(result, ensure_ascii=False)
                with (output / "actions.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                group = "arm" if result["motion"].startswith("arm_") else result["motion"] if result["motion"] == "gripper" else "base"
                print(f"[{group}] {line}", flush=True)

            if gripper_enabled:
                from wia.runtime.sim.esi_dual_arm import R1DualManipulationController

                controller = R1DualManipulationController(session, base_config, arm_config=arm_config,
                                                       gripper_config=gripper_config, rotation_config=rotation_config,
                                                       on_result=log_result)
            elif arm_enabled:
                from wia.runtime.sim.esi_arm import R1ArmController

                controller = R1ArmController(session, base_config, arm_config=arm_config,
                                             rotation_config=rotation_config, on_result=log_result)
            else:
                from wia.runtime.sim.esi_base import R1BaseController

                controller = R1BaseController(session, base_config, on_result=log_result)
            if args.trunk_control:
                from wia.runtime.sim.esi_trunk import R1TrunkController, TrunkControlConfig

                trunk_config = TrunkControlConfig.load(args.trunk_config)
                if profile:
                    trunk_config = replace(trunk_config, **profile["trunk"])
                controller.trunk_controller = R1TrunkController(controller, trunk_config)
            from wia.runtime.sim.esi_actions import create_r1_executor

            executor = create_r1_executor(controller)
            executor.macro_execution = args.macro_execution
            if args.parameterized_actions:
                from core.action_compiler import MacroLimits
                executor.macro_limits = MacroLimits(**json.loads(args.macro_config.read_text()))

                def log_macro(result):
                    with (output / "macro_actions.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                executor.on_macro_result = log_macro
        if scene_only is not None:
            scene_only.prepare_after_controller_initialization(session)
        if tabletop is not None:
            tabletop.qualify(session)
        if explore is not None:
            explore.finish_setup(session, output)
            from wia.runtime.sim.esi_explore_scenes import TASK_INSTRUCTION
            from wia.native.esi_task_evaluation import GraspLiftEvaluator
            from wia.native.esi_visibility import PrivateVisibilityCheck
            session.task_evaluator = GraspLiftEvaluator(session, controller, explore.evaluation_config(), output / "private_evaluation")
            session.private_visibility_checker = PrivateVisibilityCheck(session, explore.visibility_config(), output / "private_visibility")
            session.observation_source.task_text = TASK_INSTRUCTION
            from wia.runtime.sim.esi_explore_evaluation import ExploreEvaluator
            session.exploration_evaluator = ExploreEvaluator(session, controller, explore, output / "private_exploration")
            previous_on_step = controller.on_step
            def exploration_step():
                if previous_on_step:
                    previous_on_step()
                session.exploration_evaluator.sample_step()
            controller.on_step = exploration_step
        if args.evaluation_config is not None:
            if not gripper_enabled:
                raise ValueError("Grasp/lift evaluation requires gripper control.")
            from wia.native.esi_task_evaluation import GraspLiftEvaluator
            session.task_evaluator = GraspLiftEvaluator(session, controller,
                json.loads(args.evaluation_config.read_text()), output / "private_evaluation")
        if args.visibility_config is not None:
            from wia.native.esi_visibility import PrivateVisibilityCheck
            session.private_visibility_checker = PrivateVisibilityCheck(session,
                json.loads(args.visibility_config.read_text()), output / "private_visibility",
                working_arm=controller.working_arm if controller is not None else "left")
        if args.motion_diagnostics:
            if not arm_enabled:
                raise ValueError("Motion diagnostics require working-arm control.")
            from wia.runtime.sim.esi_motion_diagnostics import PrivateMotionDiagnostics
            session.private_motion_diagnostics = PrivateMotionDiagnostics(controller, output / "private_motion_diagnostics")
        source = session.observation_source if (base_enabled or args.web or args.capture_agent_observation) else None
        if source:
            source.human_depth_visualizations = args.human_depth_visualizations
        from core.performance import instrument_native_session
        instrument_native_session(session)
        metadata = session.debug_metadata()
        if task is not None:
            from wia.tasks.loader import public_episode_contract, sha256
            metadata["source_esi_question"] = metadata.get("question")
            metadata["question"] = task.task_text
            metadata["episode_contract"] = public_episode_contract(episode_manifest, sha256(args.episode_manifest))
        if explore is not None:
            metadata["source_esi_question"] = metadata.get("question")
            metadata["question"] = TASK_INSTRUCTION
        metadata["world_metrics_path"] = str(output / "world_metrics.json")
        metadata["performance"] = {key: getattr(args, key) for key in ("performance_profile", "macro_execution",
            "observation_pipeline", "snapshot_cache", "preview_fps", "human_preview", "human_recording",
            "debug_cameras", "human_depth_visualizations", "recording_mode", "recording_hz")}
        metadata["agent_observation_config"] = session.agent_observation_config.to_dict()
        from core.action_interface import BackendCapabilities

        metadata["capabilities"] = (executor.capabilities if executor else BackendCapabilities()).to_dict()
        if args.primitive_profile != "fixed":
            metadata["primitive_profile"] = {"name": args.primitive_profile,
                                           "path": str(args.primitive_profile_config.resolve())}
        if executor and args.parameterized_actions:
            metadata["parameterized_actions"] = {"resolution": executor.primitive_resolution.to_dict(),
                                                "limits": executor.macro_limits.to_dict()}
        metadata["observation_setup"]["preset_path"] = str(preset_path) if preset_path else None
        if args.web and args.debug_cameras:
            metadata["preview"] = {
                "read_only": not manual_enabled, "host": args.host, "port": args.port,
                "capture_fps": args.capture_fps, "physics_paused": not manual_enabled,
                "main_view": "head",
            }
        metadata["debug_retain"] = args.debug_retain
        if base_enabled:
            from wia.runtime.sim.esi_base import BASE_FRAME

            metadata["base_control"] = {"config": vars(base_config), "frame": BASE_FRAME,
                                        "action_groups": executor.capabilities.action_groups}
        if arm_enabled:
            metadata["arm_control"] = {"config": vars(arm_config), "working_arm": controller.working_arm,
                                       "frame": "robot_base", "orientation_hold": "base_relative",
                                       "original_controller": controller.original_arm_controller,
                                       "initial_trunk_posture": controller.initial_trunk_state,
                                       "enabled_arms": ["left", "right"] if executor.capabilities.right_ee_translation else ["left"],
                                       "right_action_prefix": "RIGHT_" if executor.capabilities.right_ee_translation else None}
        if gripper_enabled:
            metadata["gripper_control"] = {"config": vars(gripper_config), "working_arm": controller.working_arm,
                                            "grasp_mode": "assisted", "semantics": "explicit_idempotent"}
        if rotation_enabled:
            from wia.runtime.sim.esi_rotation import ROTATION_FRAME

            metadata["rotation_control"] = {"config": vars(rotation_config), "frame": ROTATION_FRAME}
        if args.trunk_control:
            metadata["trunk_control"] = {"config": vars(controller.trunk_controller.config),
                "frame": "robot_base", "reference_link": "torso_link4", "arm_hold": "joint_positions"}
        if session.robot is None or executor is None:
            raise RuntimeError("Native robot and control interface are required for every execution mode.")
        from wia.runtime.sim.esi_safety import SafetyObserver
        session.safety_observer = SafetyObserver(session, controller, output / 'private_safety')
        session.safety_observer.sample_step()
        safety_previous_step = controller.on_step
        def sample_safety():
            if safety_previous_step:
                safety_previous_step()
            session.safety_observer.sample_step()
        controller.on_step = sample_safety
        if episode_manifest is not None:
            session.episode_id = episode_manifest['episode_id']
        if task is not None and not task.release_v1:
            from wia.runtime.sim.esi_native_qualification import apply_reviewed_qualification
            apply_reviewed_qualification(session, controller, task, output)
        # Capture clients can freeze their full input at the actual first boundary,
        # before qualification diagnostics or viewer startup consume observations.
        initial = (initial_observation_capture(session) if initial_observation_capture is not None
                   else source.capture(action_boundary=True) if source is not None else None)
        if declarative is not None:
            qualification = declarative.qualify(session, executor, output,
                                                observation=initial.observation if initial is not None else None)
            qualification_finished = True
            (output / "task_status.json").write_text(json.dumps({"task_id": task.definition.id,
                "episode_id": episode_manifest["episode_id"], "qualification_status": qualification.status,
                "reason": qualification.reason, "run_mode": args.run_mode,
                "review_status": task.definition.status,
                "native_environment": {"available": session.robot is not None and executor is not None},
                "initialization_checks": qualification.checks}, indent=2) + "\n")
            if qualification.status == "skipped" and (args.run_mode != 'diagnostic'
                    or qualification.reason == 'source_scene_mismatch'):
                print("[task] skipped: " + qualification.reason, flush=True)
                return 0
            session.task_evaluator = declarative.create_evaluator(session, controller, output / "private_evaluation")
            evaluation = getattr(getattr(task.definition, 'private', None), 'evaluation', None)
            if getattr(evaluation, 'operation', None) in ('establish_final_relation', 'insert_or_verify_inside'):
                def initial_stability_window():
                    # Real passive hold steps verify a permitted no-op without
                    # requiring an Agent action or counting repeated renders.
                    for _ in range(15):
                        session.step_control({session.robot.name: controller.hold_action.copy()})
                        if controller.on_step:
                            controller.on_step()
                    session.initial_stability_evidence = {'native_control_steps': 15,
                        'simulated_duration_s': 15 * controller.dt, 'agent_actions': 0,
                        'policy': 'unchanged_controller_hold_after_begin_episode'}
                session.initial_stability_window = initial_stability_window
        if scene_only is not None:
            qualification = scene_only.qualify(session, executor, output,
                                              observation=initial.observation if initial is not None else None)
            qualification_finished = True
            (output / "scene_status.json").write_text(json.dumps({
                "scene_id": scene_selection.definition.id, "qualification_status": qualification.status,
                "reason": qualification.reason, "source_question_unchanged": task is None,
                "new_evaluator": False, "run_mode": args.run_mode,
                "review_status": scene_selection.definition.status,
                "native_environment": {"available": True},
                "initialization_checks": qualification.checks}, indent=2) + "\n")
            if qualification.status == "skipped" and (args.run_mode != 'diagnostic'
                    or qualification.reason == 'source_scene_mismatch'):
                print("[scene] skipped: " + qualification.reason, flush=True)
                return 0
        rgb = initial.rgb_views()["head_rgb"] if initial is not None else session.observe()["head_rgb"]
        # Only reuse the PNG encoder: no rotations, crops or deployment transforms.
        from core.record.images import save_png

        output.mkdir(parents=True, exist_ok=True)
        image_path = output / "initial_head.png"
        save_png(image_path, rgb)
        if initial is not None:
            save_png(output / "initial_wrist.png", initial.rgb_views()["wrist_rgb"])
        if args.web and args.debug_cameras:
            scene_path = output / "initial_scene.png"
            source.include_scene = True
            initial = source.capture(action_boundary=True)
            save_png(scene_path, initial.scene_rgb)
            metadata["scene_image_path"] = str(scene_path)
        metadata.update({"image_path": str(image_path), "image_shape": list(rgb.shape)})
        if args.standby_until_agent:
            metadata["standby_ready_path"] = str(output / "standby_ready.json")
        metadata_path = output / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        checker = getattr(session, "private_visibility_checker", None)
        if checker is not None and declarative is None:
            checker.check_initial(initial.observation if initial is not None else None)
        print(f"[robot] name: {metadata['robot_name']} | type: {metadata['robot_type']} | "
              f"action_dim: {metadata['action_dim']}", flush=True)
        for name, info in metadata["controllers"].items():
            print(f"[robot] {name}: {info['type']} {info['motor_type']} {info['action_indices']}", flush=True)
        print(f"[robot] sensors: {metadata['sensors']}", flush=True)
        print(f"[camera] RGB cameras: {metadata['rgb_cameras']}", flush=True)
        print(f"[camera] selected: {metadata['camera_selected']} | shape: {rgb.shape}", flush=True)
        print(f"saved:\n  {image_path}\n  {metadata_path}", flush=True)
        if base_enabled:
            print(f"[runtime] robot.controllers: {session.robot.controllers}", flush=True)
            print(f"[runtime] robot.controller_action_idx: {session.robot.controller_action_idx}", flush=True)
            print(f"[runtime] robot.action_dim: {session.robot.action_dim}", flush=True)
            print(f"[runtime] robot.get_position_orientation(): {session.robot.get_position_orientation()}", flush=True)

            print(f"[runtime] complete hold action: {controller.hold_action}", flush=True)
            if args.base_debug or args.validate_base:
                run_base_checks(session, controller, output, direct=args.base_debug)
            if args.arm_debug or args.validate_arm:
                run_arm_checks(session, controller, output, direct=args.arm_debug)
            if args.validate_camera_geometry:
                if not arm_enabled:
                    raise ValueError("Camera geometry validation requires working-arm control.")
                from wia.runtime.sim.esi_camera_checks import run_camera_geometry_checks
                run_camera_geometry_checks(session, controller, executor, source, output / "camera_geometry_checks")
            if args.rotation_debug or args.validate_rotation or (args.validate_manipulation and rotation_enabled):
                from wia.runtime.sim.esi_rotation_checks import run_rotation_checks

                run_rotation_checks(session, controller, output, direct=args.rotation_debug)
            if args.gripper_debug:
                results = [controller.set_gripper_debug(False), controller.set_gripper_debug(True),
                           controller.set_gripper_debug(True), controller.set_gripper_debug(False),
                           controller.set_gripper_debug(False)]
                (output / "gripper_debug.json").write_text(json.dumps(results, indent=2) + "\n")
            if args.validate_manipulation:
                from wia.runtime.sim.esi_manipulation_demo import run_manipulation_checks

                run_manipulation_checks(session, controller, output, json.loads(args.demo_config.read_text()),
                                        executor=executor)
        if args.capture_agent_observation:
            path = session.capture_agent_observation().save(output / "agent_observation")
            print(f"[observation] saved allowlisted AgentObservation: {path}", flush=True)
        if args.web:
            if manual_enabled:
                from wia.runtime.sim.esi_web import run_esi_control

                run_esi_control(session, metadata, controller, executor, host=args.host, port=args.port,
                                capture_fps=args.capture_fps, duration_s=args.web_duration_s,
                                preview_fps=args.preview_fps, observation_pipeline=args.observation_pipeline,
                                viewer_config=viewer_config, recording_dir=recording_dir)
            else:
                from wia.runtime.web.web_teleop.esi_preview import run_preview

                run_preview(session, metadata, host=args.host, port=args.port,
                            capture_fps=args.capture_fps, duration_s=args.web_duration_s,
                            preview_fps=args.preview_fps, observation_pipeline=args.observation_pipeline,
                            viewer_config=viewer_config, recording_dir=recording_dir)
    except Exception as error:
        if not qualification_finished:
            traceback.print_exc()
            initialization_checks = {"private_error_type": type(error).__name__, "private_error": str(error),
                "initial_trunk": getattr(session, "initial_trunk_diagnostics", None)}
            (output / "initialization_failure.json").write_text(json.dumps(initialization_checks, indent=2) + "\n")
        if scene_only is not None and not qualification_finished:
            from wia.tasks.scene import QualificationResult
            QualificationResult("skipped", "scene_initialization_failed", initialization_checks).save(
                    output, patches=scene_only.applied)
            (output / "scene_status.json").write_text(json.dumps({
                "scene_id": scene_selection.definition.id, "qualification_status": "skipped",
                "reason": "scene_initialization_failed", "source_question_unchanged": True,
                "new_evaluator": False, "run_mode": args.run_mode,
                "review_status": scene_selection.definition.status,
                "native_environment": {"available": False},
                "initialization_checks": initialization_checks}, indent=2) + "\n")
            return 0
        if declarative is not None and not qualification_finished:
            from wia.tasks.scene import QualificationResult
            QualificationResult("skipped", "scene_initialization_failed", initialization_checks).save(output, patches=declarative.applied)
            (output / "task_status.json").write_text(json.dumps({"task_id": task.definition.id,
                "episode_id": episode_manifest["episode_id"], "qualification_status": "skipped",
                "reason": "scene_initialization_failed", "run_mode": args.run_mode,
                "review_status": task.definition.status, "native_environment": {"available": False},
                "initialization_checks": initialization_checks}, indent=2) + "\n")
            return 0
        raise
    finally:
        # Outputs are written and closed before native simulator teardown.
        session.performance.save(output / "performance_counters.json")
        session.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

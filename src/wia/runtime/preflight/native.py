"""Headless native worker, without Web or Agent. Run preflight_esi_tasks.py --mode full."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

from wia.runtime.preflight.models import Check


def run_native(context, options, output: Path, emit):
    """Initialize through the production session/controller interfaces, then dispose it."""
    import torch
    from wia.tasks.loader import ROOT
    from wia.tasks.scene import DeclarativeScene
    from wia.tasks.scene_only import SceneOnly
    from core.observation_config import ObservationConfig
    from core.performance import PerformanceCounters
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    from wia.runtime.sim.esi_base import BaseControlConfig
    from wia.runtime.sim.esi_arm import ArmControlConfig
    from wia.runtime.sim.esi_rotation import RotationControlConfig
    from wia.runtime.sim.esi_gripper import GripperControlConfig
    from wia.runtime.sim.esi_dual_arm import R1DualManipulationController
    from wia.runtime.sim.esi_trunk import R1TrunkController, TrunkControlConfig
    from wia.runtime.sim.esi_actions import create_r1_executor
    from core.action_compiler import MacroLimits
    from wia.runtime.preflight.physics import check_physics
    from wia.runtime.preflight.navigation import screen_navigation
    from wia.runtime.preflight.perception import PerceptionProbe
    from wia.runtime.preflight.probes import ProbeRunner
    from wia.runtime.preflight.goals import validate_goal

    random.seed(context.loaded.seed)
    np.random.seed(context.loaded.seed)
    torch.manual_seed(context.loaded.seed)
    output.mkdir(parents=True, exist_ok=True)
    adapter = DeclarativeScene(context.loaded) if context.kind == "task" else SceneOnly(context.scene)
    runtime = getattr(context.scene.definition, "runtime", None)
    camera_config = ROOT / (runtime.robot_camera_config if runtime else "configs/esi_robot_cameras.json")
    trunk_config = ROOT / (runtime.trunk_config if runtime else "configs/esi_trunk.json")
    session = ESISession(context.scene.source_path, grasp_mode="assisted", **adapter.observation_settings())
    session.scene_adapter = adapter
    session.public_task_text = context.loaded.task_text
    session.agent_observation_config = ObservationConfig.load(options.observation_config or ROOT / "configs/esi_agent_observation.json")
    session.performance = PerformanceCounters()
    session.reuse_step_observations = True
    session.serialization_cache = True
    checks, nav, probe_results = [], {}, {}

    def record(check):
        checks.append(check)
        print(f"[preflight] {check.group}/{check.name}: {check.status}: {check.reason}", flush=True)
        emit(check)

    def stage(group, action):
        try:
            return action()
        except Exception as error:
            traceback.print_exc()
            record(Check(group, group + ".exception", "INCONCLUSIVE",
                         f"Diagnostic did not complete: {type(error).__name__}: {error}"))
            return None

    start = time.monotonic()
    try:
        print("[preflight] Native initialization starting.", flush=True)
        session.reset()
        configure_robot_cameras(session, camera_config, resolution=session.camera_size)

        def action_log(result):
            with (output / "native_actions.jsonl").open("a") as stream:
                stream.write(json.dumps(result, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")

        controller = R1DualManipulationController(session, BaseControlConfig.load(),
            arm_config=ArmControlConfig.load(), gripper_config=GripperControlConfig.load(),
            rotation_config=RotationControlConfig.load(), on_result=action_log)
        controller.trunk_controller = R1TrunkController(controller, TrunkControlConfig.load(trunk_config))
        executor = create_r1_executor(controller)
        executor.macro_limits = MacroLimits()
        # Use the same guarded metric executor; no diagnostic pose controller.
        executor.macro_execution = "metric_target"
        if isinstance(adapter, SceneOnly):
            adapter.prepare_after_controller_initialization(session)
        session.observation_source.human_depth_visualizations = False
        session.observation_source.begin_episode(context.loaded.task_text)
        record(Check("physics", "native_initialization", "PASS", "Native ESI scene and production guarded R1 controller initialized.",
            evidence={"elapsed_s": time.monotonic() - start, "grasp_mode": "assisted",
                      "camera_resolution": session.camera_size, "capabilities": executor.capabilities.to_dict()},
            verification_level="native_initialization"))
        missing = [name for name in context.capabilities if not getattr(executor.capabilities, name, False)]
        record(Check("physics", "robot_capabilities", "FAIL" if missing else "PASS",
            "Required native robot capabilities are unavailable." if missing else "Required native robot capabilities are enabled.",
            evidence={"missing": missing}, verification_level="native_initialization"))
        if "physics" in options.enabled:
            stage("physics", lambda: check_physics(context, session, controller, executor, adapter, output, options, record))
        if context.kind == "scene":
            return
        perception = PerceptionProbe(context, session, output / "private_visibility", record) if "perception" in options.enabled else None
        if perception:
            stage("perception", perception.inspect)
        if "navigation" in options.enabled:
            def navigate():
                result, rows = screen_navigation(context, session, controller, options)
                for check in result:
                    record(check)
                return rows
            nav = stage("navigation", navigate) or {}
        if any(check.status == "FAIL" and check.required for check in checks):
            for group in ("manipulation", "grasp", "goal"):
                if group in options.enabled:
                    record(Check(group, "blocked_by_precondition", "INCONCLUSIVE", "Physical probes were skipped after a demonstrated initial precondition failure."))
            return
        if set(options.enabled) & {"manipulation", "grasp", "goal"}:
            probe = ProbeRunner(context, session, controller, executor, adapter, options, output, nav, record, perception)
            probe_results = stage("manipulation", probe.run) or {}
        if "goal" in options.enabled:
            stage("goal", lambda: validate_goal(context, session, options, probe_results, record))
    finally:
        try:
            context.loaded.verify_unchanged()
            session.performance.save(output / "performance_counters.json")
        finally:
            print("[preflight] Disposing diagnostic world; it is never reused for an Agent episode.", flush=True)
            session.close()

"""Minimal original-layout warm preview for a verified Partial Occlusion pair.

Run src/wia/runtime/commands/validate_esi_warm_pair.py in a separately owned native GPU process.
This prototype restores initialization state for previews, not Agent episodes.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time


def remember_initial_background(session, env_config):
    """Observe the native constructor before its original settle steps."""
    task_names = set(session.task_object_names)
    robots = set(obj.name for obj in session.env.robots)
    background = [obj for obj in session.env.scene.objects
                  if obj.name not in task_names | robots]
    session._warm_initial_background = {
        obj.name: deepcopy(obj.dump_state(serialized=False)) for obj in background}
    session._warm_background_identity = {obj.name: id(obj) for obj in background}
    session._warm_initial_robots = {
        obj.name: deepcopy(obj.dump_state(serialized=False)) for obj in session.env.robots}
    session._warm_initial_base_joint_limits = {
        robot.name: {f"base_footprint_{component}_joint": {
            "velocity": float(robot.joints[f"base_footprint_{component}_joint"].max_velocity),
            "effort": float(robot.joints[f"base_footprint_{component}_joint"].max_effort)}
            for component in ("x", "y", "z", "rx", "ry", "rz")}
        for robot in session.env.robots}
    session._warm_initial_camera_mounts = {
        name: tuple(value.clone() for value in sensor.get_position_orientation(frame="parent"))
        for robot in session.env.robots for name, sensor in robot.sensors.items()
        if "rgb" in sensor.modalities}
    session._warm_native_config = deepcopy(env_config)


def _native_inputs(session, source_path):
    pipeline = session.pipeline
    selected = pipeline.select_question(Path(source_path), 0, None)
    payload = selected.payload
    module = pipeline.load_task_module_for_payload("", payload)
    scene, room = module.scene_room(payload)
    objects = module.build_env_objects(payload)
    config = pipeline.task_build_env_config(module, scene, room, "R1", objects,
        full_scene=bool(getattr(module, "FULL_SCENE", False)), payload=payload)
    session._configure_robot_observation(config)
    return selected, payload, module, config


def warm_contract(session, source_path):
    """Reject unverified families/configurations before changing any object."""
    _, payload, module, config = _native_inputs(session, source_path)
    sim = session.pipeline.og.sim
    return {"scene_model": payload["scene"], "room": payload["room"],
        "task_module": module.__name__, "background_config": config["scene"],
        "robot_config": config.get("robots", []),
        "env_config": {key: value for key, value in config.items()
                       if key not in {"objects", "scene", "robots"}},
        "camera_size": list(session.camera_size),
        "physics_device": str(sim.device), "physics_dt": float(sim.get_physics_dt()),
        "rendering_dt": float(sim.get_rendering_dt()),
        "preview_scope": "original_partial_occlusion_two_object_layouts"}


def compatibility(previous_contract, next_contract):
    fields = ("scene_model", "room", "task_module", "background_config", "robot_config",
              "env_config", "camera_size", "physics_device", "physics_dt", "rendering_dt",
              "preview_scope")
    differences = [key for key in fields if previous_contract.get(key) != next_contract.get(key)]
    return {"compatible": not differences, "action": "warm" if not differences else "cold",
            "reason": "Exact supported background/robot/physics/render contract." if not differences
                      else "Incompatible fields: " + ", ".join(differences),
            "differences": differences}


def switch_original_pair(session, source_path):
    """Replace two task objects; restore constructor state and native settling."""
    from omnigibson.objects import REGISTERED_OBJECTS
    from omnigibson.utils.python_utils import create_class_from_registry_and_config
    from omnigibson.utils.usd_utils import PoseAPI
    from wia.runtime.sim.esi_loading import native_loading_context

    previous_source = session.metadata_path
    before_contract = warm_contract(session, previous_source)
    after_contract = warm_contract(session, source_path)
    decision = compatibility(before_contract, after_contract)
    if not decision["compatible"]:
        raise ValueError(decision["reason"] + "; select an independent cold worker.")
    selected, payload, module, config = _native_inputs(session, source_path)
    if (module.TASK_NAME != "partial_occlusion" or len(config["objects"]) != 2
            or getattr(session, "scene_adapter", None) is not None):
        raise ValueError("This validation supports only two original Partial Occlusion objects.")
    scene, sim = session.env.scene, session.pipeline.og.sim
    old_names = set(session.task_object_names)
    new_names = {item["name"] for item in config["objects"]}
    if old_names != new_names:
        raise ValueError("The scoped pair must preserve its two native task object names.")
    old_objects = [scene.object_registry("name", name) for name in sorted(old_names)]
    old_identity = {obj.name: id(obj) for obj in old_objects}
    background_ids = session._warm_background_identity
    background_before = {name: id(scene.object_registry("name", name)) for name in background_ids}
    if background_before != background_ids:
        raise ValueError("Background membership changed before this scoped warm swap.")
    began = time.monotonic()
    with native_loading_context():
        # Environment._load_objects imports and positions bodies while stopped.
        # Preserve the process and scene, but recreate native physics handles
        # with the new geometry at its source pose rather than teleporting a
        # newly initialized playing body out of the object's graveyard.
        sim.stop()
        PoseAPI.invalidate()
        sim.batch_remove_objects(old_objects)
        PoseAPI.invalidate()
        added = []
        for specification in config["objects"]:
            item = deepcopy(specification)
            position, orientation = item.pop("position", None), item.pop("orientation", None)
            pose_frame = item.pop("pose_frame", "scene")
            obj = create_class_from_registry_and_config(cls_name=item["type"],
                cls_registry=REGISTERED_OBJECTS, cfg=item, cls_type_descriptor="object")
            added.append((obj, position, orientation, pose_frame))
        sim.batch_add_objects([item[0] for item in added], [scene] * len(added))
        for obj, position, orientation, frame in added:
            obj.set_position_orientation(position=position, orientation=orientation, frame=frame)
        PoseAPI.invalidate()
        # play() initializes new native object APIs without a physics step.
        if not sim.is_playing():
            sim.play()
        sim._non_physics_step()
        # These tensor-authored R1 limits are reset by native stop/play; they are
        # only applied by HolonomicBaseRobot._initialize during a cold load.
        # Earlier pilot bookmarks lack them, so replay that pinned native
        # initializer's constants for the same six joints in those sessions.
        from omnigibson.robots.holonomic_base_robot import m as base_limits
        for robot in session.env.robots:
            saved = getattr(session, "_warm_initial_base_joint_limits", {}).get(robot.name, {})
            for index, component in enumerate(("x", "y", "z", "rx", "ry", "rz")):
                name = f"base_footprint_{component}_joint"
                limits = saved.get(name, {
                    "velocity": base_limits.MAX_LINEAR_VELOCITY if index < 3 else base_limits.MAX_ANGULAR_VELOCITY,
                    "effort": base_limits.MAX_EFFORT})
                robot.joints[name].max_velocity = limits["velocity"]
                robot.joints[name].max_effort = limits["effort"]
        # Native Task.post_play_load soft-resets the background and takes one
        # physics update. Environment.post_play_load then saves that state;
        # Task.reset reloads it and takes another physics update, followed by
        # Environment.reset's observation step. Newly imported task bodies need
        # that constructor sequence before the module's explicit settle loop.
        # Background and robot bookmarks already include it and are restored
        # below. This two-object preview has no custom native task reset hook.
        sim.step_physics()
        PoseAPI.invalidate()
        constructor_state = {obj.name: deepcopy(obj.dump_state(serialized=False))
                             for obj, _, _, _ in added}
        for obj, _, _, _ in added:
            obj.load_state(constructor_state[obj.name], serialized=False)
        sim.step_physics()
        sim.step()
        for name, state in session._warm_initial_background.items():
            scene.object_registry("name", name).load_state(deepcopy(state), serialized=False)
        for robot in session.env.robots:
            robot.load_state(deepcopy(session._warm_initial_robots[robot.name]), serialized=False)
            for name, sensor in robot.sensors.items():
                if name in session._warm_initial_camera_mounts:
                    position, orientation = session._warm_initial_camera_mounts[name]
                    sensor.set_position_orientation(position=position, orientation=orientation, frame="parent")
        PoseAPI.invalidate()
        session.task_module, session.payload = module, payload
        session.metadata_path = session.question_path = selected.source_path
        session.scene_name, session.room_name = module.scene_room(payload)
        session.task_state = session.pipeline.task_preprocess(module, payload, selected.source_path,
            SimpleNamespace(json_root=session.json_root, robot="R1"))
        if session.task_state.get("skip_reason"):
            raise ValueError("Native preprocessing rejected the second source.")
        session.scene_camera_position, session.scene_camera_quaternion, session.camera_info = module.initial_camera(payload)
        session.task_object_names = tuple(item["name"] for item in config["objects"])
        session.settle_steps = session.pipeline.task_initial_settle_steps(module)
        for _ in range(session.settle_steps):
            sim.step()
        session.postprocess = session.pipeline.task_postprocess_env(
            module, session.env, payload, session.camera_info, session.task_state)
        session.invalidate_camera_cache()
        session._observation_source = None
        session.configure_scene_camera()
    background_after = {name: id(scene.object_registry("name", name)) for name in background_ids}
    replaced = all(id(scene.object_registry("name", name)) != old_identity[name] for name in old_names)
    if background_after != background_before or not replaced:
        raise RuntimeError("The native swap did not preserve background and replace both task objects.")
    return {"background_reused": True, "swap_s": time.monotonic() - began,
        "background_names_preserved": sorted(background_after), "task_objects_replaced": sorted(new_names),
        "background_python_identity_preserved": True, "task_python_identity_changed": True,
        "native_settle_steps": session.settle_steps, "compatibility": decision,
        "native_constructor_physics_updates_replayed": 2,
        "native_constructor_observation_steps_replayed": 1,
        "task_imported_while_native_physics_stopped": True,
        "source_poses_set_before_native_initialization": True,
        "native_r1_base_joint_limits_restored": True,
        "from_source": str(previous_source), "to_source": str(selected.source_path),
        "preview_only": True, "physical_grasp_verified": False, "model_calls": 0, "robot_actions": 0}

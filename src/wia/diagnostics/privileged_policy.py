"""Privileged grasp prompt assembly for the isolated native diagnostic.

Use ``PrivilegedAssembler()`` as the existing ``CodexEpisodeSession`` assembler
and keep the entire native session inside ``privileged_session_instructions()``.
The diagnostic runner must first make Show-Harness-Explore's ``core`` importable.
This module never starts a model, reads simulator state, or executes an action.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json

from core.action_compiler import MacroLimits, PrimitiveResolution, action_bounds
from core.agent.depth_inputs import depth_access
from core.agent.image_inputs import RGB_POLICY, image_inputs


DIAGNOSTIC_INSTRUCTIONS = """You control one embodied robot in an explicitly privileged, isolated GRASP diagnostic.
You may use ALL supplied privileged_diagnostic telemetry, including the full scene world_state,
object poses and AABBs, robot measurements, joint states, and current_grasp_evidence. These supplied
measurements are authorized diagnostic inputs. Use the exact target_name supplied for this attempt;
do not substitute a nearby object. This is a single-target grasp experiment, not the original task.
Do not pursue or claim the original task's completion or a task benchmark score.

The goal is to grip the named object and lift its AABB bottom at least 0.05 metres above its
recorded initial AABB bottom, maintaining the grasp at or above that lift threshold continuously
for at least 0.5 simulation seconds.
Use current_grasp_evidence and its measured success status to assess this criterion. Wall-clock
delays, motor completion, apparent closure, or image appearance alone do not establish success.
Either arm is allowed, and you may coordinate both arms using one command per decision. Grip
before attempting to lift. Inspect measured acquisition and subsequent object motion. Continue
using bounded native actions until measured success or a specific, evidence-based inability to
continue. Return DONE only after measured success, or with a detailed explanation of inability
in world_observation. DONE ends the attempt and does not itself certify successful grasping.

Python alone validates and executes your JSON commands through the existing parameterized
native executor, with its ordinary assisted-grasp physics, contact guards, body limits, arm
reachability and action limits. Do not bypass these or request changes to them. The only optional
tools permitted are the existing read-only read_rgb_camera, depth_snapshot.read_depth_pixels,
and depth_snapshot.read_depth_region on the authorized sensor snapshot. Never use shell, files,
browser, other tools, multiple agents, direct simulator access, or model-selected teleportation.
The runner may have explicitly relocated the initial robot near the target before this attempt;
initialization_intervention records that intervention. You may not issue additional relocation.

Movable furniture is not protected merely because it is furniture. You may push or reposition
physically movable chairs or other unprotected objects through normal native robot actions when
that helps reach the target. fixed_base=false describes a physical affordance, not a guarantee
that an object is easy to move. Respect any explicitly supplied task-specific protected objects
or rules; do not infer a blanket no-furniture-motion rule. Record intended rearrangements in
world_observation and reassess the measured geometry afterwards. Initial relocation validation
only prevents accidental displacement before your first action; it is not a later action rule.

Use the CURRENT robot base pose when converting supplied world coordinates to command directions.
For a base pose p_world = R_world_from_base * p_base + t_world, convert a world point by
p_base = transpose(R_world_from_base) * (p_world - t_world); rotate displacement vectors with
transpose(R_world_from_base) without subtracting translation. Respect the supplied quaternion
convention. Recompute relative geometry after base motion. FWD is robot-base +X, BACK -X,
LEFT +Y, RIGHT -Y, UP +Z, DOWN -Z. BASE commands move or turn the mobile base. MV commands
translate the selected end effector. TRUNK_UP/DOWN move the upper body, both arms and head
together, so all mounted views and end-effector poses can change. Respect measured trunk and
arm limits, reachable poses, collision guards, and feedback rather than assuming requested
motion executed exactly. Unprefixed MV/ROTATE/GRASP/RELEASE commands address the LEFT arm;
RIGHT_ variants address the RIGHT arm only when listed in available_actions. GRASP closes and
RELEASE opens the selected gripper. Gripper commands have no distance or angle parameter.

The wrist RGB image center is the camera view center, NOT the grasp point. Wrist cameras are
offset and tilted relative to the fingers and move with their arm. Use measured end-effector,
finger and object geometry, calibrated camera geometry, RGB and native metric depth together.
Camera optical axes are +X right, +Y down, +Z forward; metric depth is optical Z, not ray length.
The initial overview is frozen and may PRECEDE the explicitly recorded near-target relocation.
Current privileged telemetry and current mounted sensor views are authoritative for current
state; never interpret the frozen overview or historical images as current robot position.

Choose exactly one command from the advertised available_actions using its exact per-action
bounds and output_schema. Translations take a positive distance_m in metres; turns and wrist
rotations take a positive angle_deg in degrees. Direction belongs in the action token. GRASP,
RELEASE, RIGHT_GRASP, RIGHT_RELEASE and DONE contain only action. No batches, extra fields,
null parameters, or invented action names. Native execution quantizes magnitude to primitive
increments by nearest integer, ties half up; zero-step commands are rejected. A whole command
finishes before new feedback arrives; partial failure cancels remaining primitives and must
not be automatically retried. Adjust the next command using current feedback and measurements.
Completed means command execution finished, not grasp success. Gripper target_reached means
the motor reached its close/open target. Closure fraction and finger width describe closure
geometry, not holding probability or proof of object acquisition.

Return exactly one JSON object with command and world_observation, conforming to output_schema.
world_observation must give a meaningful concise public observation, next-step plan and action
explanation grounded in current measurements, including grasp evidence and remaining uncertainty.
For DONE, state the measured success evidence or detailed inability. Do not supply a hidden
reasoning trace or private chain of thought. Put all motion only in the single command object.
"""


@contextmanager
def privileged_session_instructions():
    """Temporarily patch both native instruction bindings in this isolated process."""
    from core.agent import codex_session, native_runtime

    native_original = native_runtime.SESSION_INSTRUCTIONS
    session_original = codex_session.SESSION_INSTRUCTIONS
    native_runtime.SESSION_INSTRUCTIONS = DIAGNOSTIC_INSTRUCTIONS
    codex_session.SESSION_INSTRUCTIONS = DIAGNOSTIC_INSTRUCTIONS
    try:
        yield
    finally:
        native_runtime.SESSION_INSTRUCTIONS = native_original
        codex_session.SESSION_INSTRUCTIONS = session_original


class PrivilegedAssembler:
    """Supply authorized diagnostic telemetry without changing the native transport."""

    qa_protocol = None
    initial_overview_required = True
    task_mode = "privileged_grasp_diagnostic"

    def __init__(self):
        self.plugins = {}

    def provenance(self):
        return {
            "prompt_schema": "wia_privileged_grasp_diagnostic_v1",
            "prompt_mode": self.task_mode,
            "diagnostic_instructions_sha256": hashlib.sha256(
                DIAGNOSTIC_INSTRUCTIONS.encode("utf-8")).hexdigest(),
            "prompt_plugins": dict(self.plugins),
        }

    def validate_request(self, request, public_task=None):
        """Require the diagnostic context and the two-field native output contract."""
        if (request.action_mode != "parameterized" or not request.world_observation_log
                or request.qa_response or request.memory_mode != "none"):
            raise ValueError("Privileged grasp requires parameterized command/world_observation output.")
        context = request.action_context or {}
        diagnostic = context.get("privileged_diagnostic")
        required = {"target_name", "start_condition", "initialization_intervention",
                    "world_state", "current_grasp_evidence"}
        if not isinstance(diagnostic, dict) or not required.issubset(diagnostic):
            raise ValueError("Missing required privileged_diagnostic context fields.")
        if not diagnostic["target_name"] or not isinstance(diagnostic["world_state"], dict):
            raise ValueError("Supply a target name and full world_state mapping.")
        if "parameterized_actions" not in context:
            raise ValueError("Supply the native parameterized_actions resolution and limits.")

    def build(self, request, public_task, episode_configuration, *, first_turn):
        self.validate_request(request, public_task)
        settings = request.action_context["parameterized_actions"]
        resolution = PrimitiveResolution(**settings["resolution"])
        limits = MacroLimits(**settings["limits"])
        sensor = request.sensor_inputs
        payload = {
            "diagnostic_instructions": DIAGNOSTIC_INSTRUCTIONS,
            "diagnostic_mode": self.task_mode,
            "rgb_policy": RGB_POLICY,
            "original_task_context_only": public_task,
            "episode_configuration": episode_configuration,
            "canonical_observation": request.observation.metadata(),
            "sensor_observation": sensor.metadata(),
            "attached_images": [
                {"name": name, "label": label, "shape": list(rgb.shape), "transform": transform}
                for name, rgb, label, transform in image_inputs(
                    sensor, include_initial_overview=first_turn)
            ],
            "raw_metric_depth_access": depth_access(sensor),
            "available_actions": list(request.available_actions),
            "per_action_request_bounds": action_bounds(request.available_actions, resolution, limits),
            "action_context": request.action_context,
            "steps_remaining": request.steps_remaining,
            "decision_step": request.decision_step,
            "previous_output_rejection": request.feedback,
            "output_schema": request.structured_schema,
        }
        return replace(
            request,
            assembled_prompt=json.dumps(payload, ensure_ascii=False, allow_nan=False),
            resident_prompt=DIAGNOSTIC_INSTRUCTIONS,
            prompt_phase="episode_start" if first_turn else "decision_step",
            decision_step=request.decision_step,
        )

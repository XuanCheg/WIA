"""Convert Reflection Authoring to source-bound optical observation and answers.

Run CPU conversion with bounded threads:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m \
wia.authoring.converters.specular_reflection_reflection_authoring --assignment ASSIGNMENT.json

Native qualification hooks only read state; the explicit action-variant scene
initializer aligns the robot before the episode. All episode actions remain
existing guarded R1 controls. Saved views never prove robot clearance,
reflected visibility, original QA correctness or physical task success.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_contact

import argparse
from collections import Counter
from copy import deepcopy
import itertools
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wia.authoring.conversion import (
    ConverterBase, ConversionContext, SourceInstance, original_scene, physical_task,
)
from wia.authoring.io import write_json
from wia.tasks.interaction_setup import (
    array as native_array, reference_position, support_contact_paths, verified_repaired_start)
from wia.native.esi_task_evaluation import GraspLiftEvaluator

MODULE = "wia.authoring.converters.specular_reflection_reflection_authoring"
OPERATION = "mirror_survey_round_trip"
LIFT_OPERATION = "discover_and_lift_reflected_prop"
ANSWER_OPERATION = "answer_direct_or_reflected_appearance"
ACTION_OPERATION = 'classify_and_verify_direct_reflected_appearance'
TASK_REVISION = "task_edit_20261018_v1"
ACTION_REVISION = 'task_edit2_reflection_verification_20261018_v3'
REFLECTION_QUALIFICATION_KEYS = (
    "source_head_pose_verified", "source_head_intrinsics_verified",
    "native_robot_scale_verified", "mounted_eight_camera_rig_verified",
    "legal_mobility_verified", "optical_scene_preserved",
)
MINIMUM_EXCURSION_M = 0.25
MINIMUM_BEARING_DEG = 10.0
SOURCE_PROP_SCREEN_M = 0.65
SOURCE_MIRROR_SCREEN_M = 0.8


def _array(value):
    return native_array(value)


def _vector(value, size):
    result = _array(value)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Missing or invalid source/native geometry.")
    return result


def _angle(first, second):
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _bearing(point, centre):
    delta = _array(point)[:2] - _array(centre)[:2]
    if np.linalg.norm(delta) < 1e-6:
        raise ValueError("Mirror bearing is undefined at the mirror centre.")
    return math.atan2(delta[1], delta[0])


def _source_geometry(source):
    """Mirror common.py constructs these exact objects after native loading."""
    if (source.big_task, source.small_task) != CategoryConverter.category:
        raise ValueError("source_category_mismatch")
    raw, payload = source.raw, source.payload
    if any(payload.get(key, raw.get(key)) != raw.get(key) for key in ("scene", "room")):
        raise ValueError("source_scene_room_disagree")
    q = payload["question_data"]
    placement = q["placement"]
    if len(placement) != 1:
        raise ValueError("expected_one_original_queried_prop")
    prop = placement[0]
    if not all(prop.get(key) for key in ("name", "category", "model")):
        raise ValueError("missing_original_prop_identity")
    _vector(prop["real_position"], 3)
    _vector(prop.get("quaternion_xyzw", prop.get("orientation", [0., 0., 0., 1.])), 4)
    mirror = q["scene_setup"]["mirror"]
    _vector(mirror["position"], 3)
    _vector(mirror["quaternion_xyzw"], 4)
    camera = q.get("render", {}).get("camera_pose") or q["scene_setup"]["camera"]
    _vector(camera["position"], 3)
    _vector(camera["quaternion_xyzw"], 4)
    if not payload.get("floor_name"):
        raise ValueError("missing_original_floor_binding")
    height = float(payload["camera_poses"]["height_m"])
    if not math.isfinite(height) or height <= 0:
        raise ValueError("missing_source_camera_height_for_floor_proposal")
    return raw, payload, q, prop, mirror, camera


def _stance_proposal(q, prop, mirror, camera):
    """Choose actual saved views, never treat the source point screen as physics."""
    render = q.get("render", {})
    orbit = [("orbit", v) for v in render.get("gt_view_input", {}).get("views", [])]
    corners = [("corner", v) for v in render.get("multi_image_input", {}).get("views", [])]
    centre, prop_position = _vector(mirror["position"], 3), _vector(prop["real_position"], 3)
    reference = _vector(camera["position"], 3)

    def valid(rows):
        result = []
        for kind, view in rows:
            point = _vector(view["camera_pose"]["position"], 3)
            if (np.linalg.norm(point[:2] - prop_position[:2]) >= SOURCE_PROP_SCREEN_M
                    and np.linalg.norm(point[:2] - centre[:2]) >= SOURCE_MIRROR_SCREEN_M):
                result.append((kind, view, point))
        return result

    # Prefer an orbit-only witness, then allow original room-corner evidence.
    for family, views in (("saved_orbit", orbit), ("saved_corner_fallback", orbit + corners)):
        pairs = []
        for first, second in itertools.combinations(valid(views), 2):
            separation = float(np.linalg.norm(first[2][:2] - second[2][:2]))
            angle = math.degrees(_angle(_bearing(first[2], centre), _bearing(second[2], centre)))
            if separation < MINIMUM_EXCURSION_M or angle < MINIMUM_BEARING_DEG:
                continue
            # Start as close to the original observer as the source permits.
            first, second = sorted((first, second), key=lambda x: (
                float(np.linalg.norm(x[2][:2] - reference[:2])), str(x[1].get("image_path", ""))))
            pairs.append((float(np.linalg.norm(first[2][:2] - reference[:2])), separation,
                          str(first[1].get("image_path", "")), first, second, angle))
        if pairs:
            chosen = min(pairs, key=lambda item: item[:3])
            first, second = chosen[3:5]
            return {"family": family, "start_position": first[2].tolist(),
                "start_image": first[1].get("image_path"), "start_kind": first[0],
                "witness_position": second[2].tolist(), "witness_image": second[1].get("image_path"),
                "witness_kind": second[0], "separation_m": chosen[1], "bearing_change_deg": chosen[5],
                "provenance": "question_data.render.gt_view_input.views or multi_image_input.views",
                "native_reachability_proven": False}
    raise ValueError("no_separated_source_view_pair_after_point_clearance_screen")


class CategoryConverter(ConverterBase):
    category = ("Specular Reflection", "Reflection Authoring")
    version = "reflection_authoring_v4_task_edit_20261018"

    def inspect(self, source):
        try:
            _, _, q, prop, mirror, camera = _source_geometry(source)
            return {"eligible": True, "reason": "one_original_prop_and_private_optical_reference",
                "proposal": {"family": "original_source_observer", "start_position": camera["position"],
                    "provenance": "question_data.render.camera_pose or scene_setup.camera",
                    "native_reachability_proven": False, "native_reflection_only_proven": False}}
        except (KeyError, TypeError, ValueError) as error:
            return {"eligible": False, "reason": "not_convertible_under_L1_L2: " + str(error),
                "warning_codes": ["source_navigation_recipe_unsupported", "source_only_diagnostic_required"]}

    def convert(self, source, context):
        operation = context.config.get('reflection_authoring_operation', ANSWER_OPERATION)
        if operation == ACTION_OPERATION:
            raw, payload, q, prop, mirror, camera = _source_geometry(source)
            return build_reflection_action(source, context, module=MODULE, operation=operation,
                camera=camera, mirror=mirror, props={'prop': prop})
        if operation == ANSWER_OPERATION:
            return _convert_observation(source, context)
        if operation == LIFT_OPERATION:
            return _convert_lift(source, context)
        if operation != OPERATION:
            raise ValueError('Unsupported Reflection Authoring operation.')
        eligibility = self.inspect(source)
        if not eligibility["eligible"]:
            return {"scene_definition": original_scene(source, context), "task_definition": None,
                "status": "conversion_failed", "failure_stage": "source_inspection",
                "failure_reason": eligibility["reason"], "warning_codes": eligibility["warning_codes"],
                "next_step": "Review the original source and unsupported survey recipe; no physical success is claimed."}
        raw, payload, q, prop, mirror, camera = _source_geometry(source)
        proposal = _stance_proposal(q, prop, mirror, camera)
        mirror_setup = payload.get("mirror_setup", {})
        bindings = {"mirror": {"name": mirror_setup.get("name") or "render_mirror_main",
            "category": mirror_setup.get("category") or "mirror", "model": mirror_setup.get("model") or "tytkbq"},
            "prop": {key: prop[key] for key in ("name", "category", "model")},
            "floor": {"name": payload["floor_name"]}}
        expected = {"mirror": {"position": mirror["position"], "quaternion_xyzw": mirror["quaternion_xyzw"]},
            "prop": {"position": prop["real_position"], "quaternion_xyzw": prop.get("quaternion_xyzw",
                prop.get("orientation", [0., 0., 0., 1.]))}}
        floor_z = float(camera["position"][2] - payload["camera_poses"]["height_m"])
        start = proposal["start_position"]
        yaw = math.atan2(mirror["position"][1] - start[1], mirror["position"][0] - start[0])
        patches = [{"operation": "robot_pose", "position": [float(start[0]), float(start[1]), floor_z + 0.02],
            "yaw_deg": math.degrees(yaw), "head_pitch_deg": 0.0}]
        parameters = {"expected_optical_poses": expected, "proposed_start_xy": start[:2],
            "minimum_excursion_m": MINIMUM_EXCURSION_M, "minimum_bearing_change_deg": MINIMUM_BEARING_DEG,
            "return_tolerance_m": 0.12, "facing_tolerance_deg": 20.0, "minimum_up_dot": 0.95,
            "maximum_stopped_linear_speed_m_s": 0.02, "maximum_stopped_angular_speed_rad_s": 0.1,
            "stability_steps": 6, "asset_position_tolerance_m": 0.01, "asset_orientation_tolerance_deg": 1.0,
            "visual_asset_clearance_m": 0.03, "source_pose_tolerance_m": 0.02,
            "start_tolerance_m": 0.12, "floor_height_tolerance_m": 0.03,
            "minimum_floor_normal_z": 0.5, "proposed_floor_z_m": floor_z}
        evaluation = {"type": "category_operation", "module": MODULE, "operation": OPERATION,
            "binding_keys": ["mirror", "prop", "floor"], "parameters": parameters}
        public = {"objective": "Perform a physical mirror survey: move around the same mirror to either side, "
            "stop to inspect it from a second bearing, then return to your starting station and stop facing the mirror.",
            "success_criteria": "At the second station, move at least 0.25 m from your starting point and change "
            "your bearing around the mirror by at least 10 degrees. Face the mirror within 20 degrees and stop "
            "at both stations. Finish within 0.12 m of your starting point. Keep the optical scene unchanged.",
            "environment": "The original indoor mirror and the " + prop["category"].replace("_", " ") + " are retained.",
            "constraints": ["Use robot mounted observations and existing guarded base motion controls.",
                "Remain upright on the original floor and avoid base collisions, walls, the mirror and the prop. Legal arm manipulation of noncritical native movable furniture is allowed.",
                "Do not touch or move the mirror or prop, change materials, or move the source reference camera.",
                "The original image question is auxiliary and remains tied to its original observer; this survey does not redefine its answer."]}
        assumptions = [{"doubt": "Saved camera points and their source visibility filters do not certify an R1 footprint or route.",
            "chosen_assumption": "Use the saved " + proposal["family"] + " point nearest the original observer as an L2 initialization proposal; require measured native floor, room and clearance checks.",
            "alternatives": ["Use a different source-evidenced stance after native measurement.",
                "Publish a native qualification failure if the preserved room cannot support the survey."],
            "scope": source.relative_path,
            "risk": "Camera-valid points can be obstructed for a full robot; neither the source point screen nor a screenshot proves a route."},
            {"doubt": "R1 mounted images differ from the original 0.8 m reference image.",
            "chosen_assumption": "Declare a new physical survey variant; keep original reference poses and QA private and unchanged, with no visibility or answer-equivalence success claim.",
            "alternatives": ["Separately calibrate a mounted optical reference.", "Keep source-only QA diagnostics."],
            "scope": "All category variants", "risk": "The mirror or prop may be outside the mounted field of view; native rendering and observation still need Review."},
            {"doubt": "Camera height minus the declared source height proposes a floor elevation but is not a native support measurement.",
            "chosen_assumption": "Initialize the R1 root 0.02 m above that source-derived floor elevation, then require original native floor contacts and consistent measured heights.",
            "alternatives": ["Use measured local floor geometry for a revised category rule.", "Retain a diagnostic failure when floor support is unqualified."],
            "scope": source.relative_path, "risk": "Steps, uneven floors or source height error may make the proposed stance invalid."}]
        warnings = ["assumption_review_required", "physical_qualification_pending", "native_route_unverified",
            "mounted_reflection_visibility_unverified", "source_view_is_not_robot_clearance"]
        if proposal["family"] == "saved_corner_fallback":
            warnings.append("source_corner_stance_fallback")
        if q.get("answer") != raw.get("answer") or q.get("question") != raw.get("question"):
            warnings.append("source_qa_annotation_conflict")
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
            patches=patches, capabilities=["base_translation", "base_rotation"],
            qualification={"on_failure": "skip", "initial_visibility": "none", "target_movable_collidable": False})
        return {"scene_definition": original_scene(source, context, bindings=bindings, patches=patches),
            "task_definition": task, "status": "generated", "reuse_level": "L2", "qa_preserved": True,
            "operation_recipe": OPERATION, "source_view_proposal": proposal,
            "original_reference": {"camera_pose": camera, "question": raw.get("question"),
                "scope": "Original image QA only; not the moving R1 or physical success."},
            "change_summary": ["Preserve all original optical assets, poses, materials, QA and an empty original variant.",
                "Propose only the existing R1 initialization stance from archived source views.",
                "Add an ordered physical excursion and return, independently measured from native state."],
            "warning_codes": warnings, "assumptions": assumptions,
            "limitations": ["CPU conversion does not certify native loading, sensor visibility, collision clearance or route feasibility.",
                "The source visual-only mirror and prop are preserved and must be avoided; this navigation recipe performs no manipulation.",
                "A conservative native robot AABB screen may reject valid local paths; source point screens ignore dimensions.",
                "Survey success measures navigation and asset preservation, not the truth of the original yes/no question or reflected pixels.",
                "No complete robot execution or runtime/throughput measurement has been performed by this converter."],
            "attempted_repairs": ["Reversible robot-only stance proposal; original optical geometry is untouched."],
            "next_step": "Use the shared runner to capture stratified candidates and publish all records; qualify and execute native guarded navigation separately."}

    def convert_all(self, sources, context):
        manifest = super().convert_all(sorted(sources, key=lambda item: item.relative_path), context)
        manifest["sample_ids"] = self.sample(manifest)
        manifest["failure_distribution"] = dict(Counter(row["failure_reason"] for row in manifest["records"] if row["status"] != "generated"))
        write_json(context.output_root / "batch_manifest.json", manifest)
        return manifest

    def sample(self, manifest, policy=None):
        policy = policy or {}
        rows = sorted(manifest["records"], key=lambda row: row["candidate_id"])
        count = min(int(policy.get("count", 5)), len(rows))
        selected = []
        groups = [[r for r in rows if r["status"] != "generated"],
            [r for r in rows if r["candidate_id"] in policy.get("previous_failures", [])],
            [r for r in rows if "source_corner_stance_fallback" in r["warning_codes"]],
            sorted([r for r in rows if r.get("source_view_proposal", {}).get("family") == "saved_orbit"],
                key=lambda r: r["source_view_proposal"]["separation_m"]),
            sorted(rows, key=lambda r: -r.get("source_view_proposal", {}).get("separation_m", 0.))]
        for group in groups:
            match = next((r["candidate_id"] for r in group if r["candidate_id"] not in selected), None)
            if match and len(selected) < count:
                selected.append(match)
        for candidate in super().sample(manifest, policy):
            if candidate not in selected and len(selected) < count:
                selected.append(candidate)
        return selected


def _answer_key(value):
    return " ".join(str(value).strip().casefold().split())


def _convert_observation(source, context):
    """Declare preserved optics and a source-bound question, without scene edits."""
    raw, payload, q, prop, mirror, camera = _source_geometry(source)
    result = build_reflection_observation(source, context, module=MODULE, operation=ANSWER_OPERATION,
        camera=camera, mirror=mirror, props={'prop': prop})
    result['available_operation_recipes'] += [LIFT_OPERATION, OPERATION]
    return result


def build_reflection_observation(source, context, *, module, operation, camera, mirror, props):
    """Shared source-bound answer contract for the three preserved optical layouts."""
    raw, payload = source.raw, source.payload
    q = payload['question_data']
    options = raw.get('options_json', q.get('options', []))
    options = json.loads(options) if isinstance(options, str) else options
    question, answer = raw.get('question'), raw.get('answer')
    if (not question or not isinstance(options, list) or not options
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or _answer_key(answer) not in {_answer_key(option) for option in options}
            or q.get('question', question) != question
            or _answer_key(q.get('answer', answer)) != _answer_key(answer)):
        raise ValueError('Source question, options and answer must agree before optical scoring.')
    if q.get('options') and [_answer_key(x) for x in q['options']] != [_answer_key(x) for x in options]:
        raise ValueError('Source option annotations disagree.')
    mode = context.config.get('reflection_boundary_mode', 'bounded')
    if mode not in ('bounded', 'unrestricted'):
        raise ValueError('Reflection boundary mode must be bounded or unrestricted.')
    scale = context.config.get('reflection_robot_uniform_scale')
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float))
                              or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Reflection robot scale must be a positive finite uniform scalar.')
    fov = float(payload['camera_poses']['fov_deg'])
    if not 0 < fov < 180:
        raise ValueError('Source horizontal field of view must be between zero and 180 degrees.')
    setup = payload.get('mirror_setup', {})
    bindings = {name: {key: prop[key] for key in ('name', 'category', 'model')}
                for name, prop in props.items()}
    bindings.update({'mirror': {'name': setup.get('name') or 'render_mirror_main',
            'category': setup.get('category') or 'mirror', 'model': setup.get('model') or 'tytkbq'},
        'floor': {'name': payload['floor_name']}})
    profile = {'version': 'reflection_sensor_profile_v1',
        'position_m': deepcopy(camera['position']), 'quaternion_xyzw': deepcopy(camera['quaternion_xyzw']),
        'coordinate_frame': 'world', 'position_units': 'metres',
        'pose_source': ('metadata_json.question_data.render.camera_pose'
                        if q.get('render', {}).get('camera_pose') else 'metadata_json.question_data.scene_setup.camera'),
        'horizontal_fov_deg': fov, 'fov_units': 'degrees',
        'fov_source': 'metadata_json.camera_poses.fov_deg',
        'fov_convention_source': 'ESI-Bench/src/passive_gt/replay.py:set_camera_fov',
        'robot_uniform_scale': scale, 'scale_units': 'dimensionless',
        'scale_status': 'configured_pending_native_validation' if scale is not None else 'pending_native_selection',
        'source_floor_reference': {'binding': 'floor',
            'proposed_z_m': float(camera['position'][2] - payload['camera_poses']['height_m']),
            'source': 'source camera z minus metadata_json.camera_poses.height_m',
            'native_support_verified': False},
        'boundary': {'version': 'reflection_head_horizontal_world_m_v1', 'mode': mode,
            'reference_position_m': deepcopy(camera['position']),
            'reference': 'initial_original_ESI_observation_position',
            'tracked_point': 'native_head_camera', 'metric': 'horizontal_euclidean_world_metres',
            'limit_m': 1.0 if mode == 'bounded' else None, 'invariant_under_robot_scale': True,
            'violation_policy': 'latch_each_control_step'},
        'camera_policy': 'retain_all_eight_robot_mounted_views',
        'native_alignment_verified': False}
    parameters = {'task_revision': TASK_REVISION, 'reflection_sensor_profile': profile,
        'source_qa': {'question': question, 'options': deepcopy(options), 'answer': answer,
            'source_path': source.relative_path, 'reference': 'initial_original_ESI_observation'},
        'expected_optical_poses': {'mirror': {key: deepcopy(mirror[key]) for key in ('position', 'quaternion_xyzw')},
            **{name: {'position': deepcopy(prop['real_position']), 'quaternion_xyzw': deepcopy(
                prop.get('quaternion_xyzw', prop.get('orientation', [0., 0., 0., 1.])))}
               for name, prop in props.items()}},
        'asset_position_tolerance_m': .01, 'asset_orientation_tolerance_deg': 1.,
        'mechanism_contract': {'invariants': ['mirror_and_prop_identity_scale_pose_and_material',
            'original_supports_and_relative_geometry', 'original_source_observer_and_question_reference'],
            'required_native_evidence': list(REFLECTION_QUALIFICATION_KEYS) + ['per_step_native_head_position'],
            'no_forced_motion': True, 'manipulation_required': False,
            'source_answer_applies_only_after_reference_qualification': True}}
    public = {'objective': 'Answer the following question about the initial head-camera observation: ' + question,
        'success_criteria': 'Report one of these source answer options: ' + ', '.join(options) + '. '
            'Submit the selected option in the answer field. '
            'The question remains tied to the initial observation when you explore. You may answer immediately.',
        'environment': 'The original mirror, objects, materials and supports are preserved.',
        'constraints': ['Use the eight robot-mounted views and legal observation/navigation controls.',
            'Keep the mirror, props, materials and supports unchanged.',
            ('Keep the native head camera within 1.0 world metre horizontally of its initial source observation '
             'position at every step. This radius does not change with robot scale.' if mode == 'bounded' else
             'Explore legal reachable viewpoints as useful; no excursion or return is required.')]}
    evaluation = {'type': 'category_operation', 'module': module, 'operation': operation,
        'binding_keys': list(bindings), 'parameters': parameters}
    task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
        capabilities=['base_translation', 'base_rotation'],
        qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
    return {'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': task,
        'status': 'generated', 'reuse_level': 'L1', 'qa_preserved': True, 'operation_recipe': operation,
        'available_operation_recipes': [operation],
        'interaction_policy': {'mode': 'navigation', 'group_bindings': ['mirror', *props],
            'manipulation_bindings': [], 'relocation_policy': 'preserve', 'source_region': raw['room'],
            'source_camera_policy': 'preserve', 'allow_fallback_table': False},
        'private_source_evidence': {'source_qa': deepcopy(parameters['source_qa']),
            'source_question_data': deepcopy(q), 'source_camera_poses': deepcopy(payload['camera_poses']),
            'native_execution_verified': False},
        'change_summary': ['Use the actual source optical question without mandatory physical excursions or manipulation.',
            'Preserve scene geometry and expose explicit source sensor and robot scale requirements.',
            'Allow an immediate answer and monitor the declared native head movement boundary.'],
        'warning_codes': ['native_reflection_sensor_alignment_pending', 'native_robot_scale_and_mobility_pending',
            'all_eight_camera_shortcut_audit_pending'],
        'limitations': ['CPU conversion does not certify head alignment, legal scaling, mobility or mirror dependence.'],
        'next_step': 'Qualify the native source/head alignment, preserved optics and all eight mounted views.'}


def build_reflection_action(source, context, *, module, operation, camera, mirror, props):
    """Version an initial answer followed by an informative legal second view."""
    from wia.runtime.sim.esi_reflection import (original_reference_provenance, ACTION_PROFILE_VERSION,
        INITIALIZATION_REVISION, INITIALIZATION_OPERATION)
    result = build_reflection_observation(source, context, module=module, operation=operation,
        camera=camera, mirror=mirror, props=props)
    task = result['task_definition']
    p = task['private']['evaluation']['parameters']
    mode = p['reflection_sensor_profile']['boundary']['mode']
    reference = original_reference_provenance(source, context.config.get('reflection_original_image_root'))
    source_size = reference.get('decoded_image_size_px')
    if source_size:
        # Preserve the original full frame; no crop or camera transform.
        divisor = math.gcd(*source_size)
        ratio = [value // divisor for value in source_size]
        multiple = max(1, 512 // ratio[0])
        render_size = [value * multiple for value in ratio]
        if any(not 64 <= value <= 2048 for value in render_size):
            raise ValueError('The original image aspect cannot fit the supported native camera size.')
        p['reflection_sensor_profile'].update(
            head_image_size_px=render_size, original_image_size_px=source_size,
            aspect_source='decoded_immutable_original_ESI_image',
            image_mapping='full_frame_uniform_scale_without_crop_or_warp')
    p['reflection_sensor_profile'].update(version=ACTION_PROFILE_VERSION,
        initialization_revision=INITIALIZATION_REVISION, source_identity=source.relative_path,
        source_sha256=hashlib.sha256(source.path.read_bytes()).hexdigest())
    p.update(task_revision=ACTION_REVISION, requires_native_visual_discovery=True,
        original_reference_provenance=reference,
        variant_revision=ACTION_REVISION + '_' + mode,
        input_condition={'initial_global_rgb': 'mandatory_frozen_reset_world_image',
            'mounted_views': 'all_eight', 'audit_global_and_all_mounted_direct_view_shortcuts': True},
        verification_contract={'version': ACTION_REVISION, 'required_bindings': list(props),
            'mechanism': 'direct_reflection_relationship_verified',
            'allowed_view_kinds': ['direct', 'reflected'],
            'reference': 'initial_original_ESI_observer', 'answer_before_viewpoint_transition': True,
            'native_transition_and_new_informative_features_required': True,
            'native_visual_adapter': 'task_visual_evidence'},
        completion_items=[{'id': 'initial_optical_answer'}, {'id': 'informative_second_view'}])
    p['mechanism_contract'].update(no_forced_motion=False, manipulation_required=False,
        informative_second_view_required=True, arbitrary_excursion_or_return_required=False,
        true_original_reference_equivalence_required=True)
    subclass = source.small_task
    p['verification_contract']['viewpoint_actions'] = (['BASE_LEFT', 'BASE_RIGHT', 'TRUNK_UP', 'TRUNK_DOWN']
        if subclass == 'Reflection Authoring' else ['BASE_FWD', 'BASE_BACK', 'BASE_LEFT', 'BASE_RIGHT', 'TRUNK_UP', 'TRUNK_DOWN'])
    action = ('Identify the real object corresponding to the initial reflection, then reach an informative '
        'verification view of that same object. A second mirror view is allowed when direct access is unavailable.'
        if subclass == 'Correspondence' else
        'Report the initial relation in the original observer frame, then reach an informative second view '
        'of both bound objects. Movement must not redefine the initial relation.'
        if subclass == 'Spatial Relations' else
        'Classify the specified initial appearance as direct or reflected, then obtain a second view '
        'showing its direct/reflection relationship through legal lateral or torso movement.')
    task['public']['objective'] = source.raw['question'] + ' ' + action
    task['public']['success_criteria'] = ('First submit the selected option in the answer field '
        'using one of: ' + ', '.join(p['source_qa']['options']) + '. Then obtain new visible task features '
        'from a physically different legal viewpoint. A changed position alone does not verify the answer. '
        'No arbitrary travel distance or return is required. The answer remains tied to the initial reference.')
    task['requirements']['capabilities'] = ['base_translation', 'base_rotation', 'trunk_height']
    task['id'] += '_verify_' + mode
    task['repair'] = {'task_revision': ACTION_REVISION, 'mechanism_contract': deepcopy(p['mechanism_contract'])}
    result['scene_definition'].setdefault('runtime', {})['requires_native_visual_discovery'] = True
    result['scene_definition']['runtime']['reflection_sensor_profile'] = deepcopy(p['reflection_sensor_profile'])
    hook = {'operation': 'category_initialization', 'phase': 'after_postprocessing', 'module': MODULE,
        'parameters': {'operation': INITIALIZATION_OPERATION, 'initialization_revision': INITIALIZATION_REVISION,
            'task_revision': ACTION_REVISION, 'source_identity': source.relative_path,
            'source_sha256': p['reflection_sensor_profile']['source_sha256'],
            'scene': source.raw['scene'], 'room': source.raw['room'],
            'reflection_sensor_profile': deepcopy(p['reflection_sensor_profile'])}}
    for definition in (task, result['scene_definition']):
        variants = definition['scene']['variants']
        adapted = deepcopy(variants.get('adapted', variants['original']))
        adapted['patches'].append(deepcopy(hook))
        variants['adapted'] = adapted
    result.update(task_revision=ACTION_REVISION, reuse_level='L2',
        variant_revision=p['variant_revision'],
        source_reference_provenance=reference,
        input_condition=deepcopy(p['input_condition']),
        shortcut_audit={'status': 'pending_native_initial_images',
            'source_visibility_annotations': deepcopy(source.payload['question_data'].get('evidence', {})),
            'global_and_other_mounted_direct_views_audited': False,
            'annotation_alone_is_not_a_native_shortcut_measurement': True},
        verification_contract=p['verification_contract'],
        pending_gates=['true_original_source_view_equivalence', 'native_start_and_optical_integrity',
            'legal_public_control_motion', 'feature_bearing_second_view', 'initial_information_shortcut_audit'],
        change_summary=['Preserve the initial source optical answer and immutable optical actors.',
            'Require a legal second viewpoint with synchronized informative feature/identity evidence.',
            'Retain separate bounded/unrestricted variants and the one-world-metre horizontal head limit.'])
    observation_operation = {MODULE: ANSWER_OPERATION,
        'wia.authoring.converters.specular_reflection_correspondence': 'answer_reflection_correspondence',
        'wia.authoring.converters.specular_reflection_spatial_relations': 'answer_reflected_spatial_relation'}[module]
    result['available_operation_recipes'].append(observation_operation)
    result['observation_only_alternative'] = {'operation': observation_operation,
        'task_revision': TASK_REVISION, 'retained': True,
        'action_coverage': 'pending_informative_legal_native_viewpoint_transition'}
    if not reference['available']:
        result['warning_codes'].append('true_original_ESI_image_unavailable')
    return result


def initialize_scene(session, parameters):
    """Run the versioned action-variant alignment exactly once before an episode.

    Completion means the native alignment routine returned. Pose, optics,
    floor support and original-image equivalence remain separate measured gates.
    """
    from wia.runtime.sim.esi_reflection import (align_initial_robot, profile_from_definition,
        reflection_profile_sha256, ACTION_PROFILE_VERSION, INITIALIZATION_REVISION, INITIALIZATION_OPERATION)
    if getattr(session, '_reflection_scene_initialization_attempted', False):
        raise RuntimeError('Reflection initialization cannot run twice.')
    state = vars(session)
    public_controller = state.get('controller')
    if (not getattr(session, '_category_initialization_active', False) or state.get('episode_id') or
            state.get('_observation_source') is not None or state.get('observation_source') is not None or
            state.get('action_history') or state.get('action_step', 0) or state.get('_public_action_sequence', 0) or
            (public_controller is not None and (getattr(public_controller, '_public_action_sequence', 0) or
                getattr(public_controller, '_public_action_context', {}).get('active')))):
        raise RuntimeError('Reflection alignment is initialization-only, before public observations/actions.')
    profile = getattr(session, 'reflection_sensor_profile', None)
    if (parameters.get('operation') != INITIALIZATION_OPERATION or
            parameters.get('initialization_revision') != INITIALIZATION_REVISION or
            parameters.get('task_revision') != ACTION_REVISION or not profile or
            profile.get('version') != ACTION_PROFILE_VERSION or profile.get('initialization_revision') != INITIALIZATION_REVISION or
            parameters.get('reflection_sensor_profile') != profile or
            (session.scene_name, session.room_name) != (parameters.get('scene'), parameters.get('room')) or
            not getattr(session, 'native_reset_id', None) or
            parameters.get('source_identity') != profile.get('source_identity') or
            parameters.get('source_sha256') != profile.get('source_sha256')):
        raise ValueError('Reflection initialization profile/source/version/scene does not agree exactly.')
    owners = [owner for owner in (getattr(session, 'loaded_task', None),
        getattr(getattr(session, 'scene_adapter', None), 'task', None)) if owner is not None]
    if not owners:
        raise ValueError('An exact loaded source/profile definition is required for initialization.')
    for owner in owners:
        declared = owner.definition.source
        identity = declared.get('json') if isinstance(declared, dict) else declared.json_path
        if (identity != parameters['source_identity'] or owner.source_sha256 != parameters['source_sha256'] or
                hashlib.sha256(Path(owner.source_path).read_bytes()).hexdigest() != parameters['source_sha256'] or
                profile_from_definition(owner.definition) != profile):
            raise ValueError('Loaded reflection source or declared profile differs from the initialization hook.')
    session._reflection_scene_initialization_attempted = True
    receipt = {'schema': INITIALIZATION_REVISION, 'operation': INITIALIZATION_OPERATION,
        'reset_id': session.native_reset_id, 'profile_sha256': reflection_profile_sha256(profile),
        'source_identity': parameters['source_identity'], 'source_sha256': parameters['source_sha256'],
        'completed': False, 'physical_qualification_complete': False,
        'source_pose_optics_floor_and_image_equivalence_verified': False}
    session.reflection_alignment_initialization = receipt
    session._reflection_initialization_hook_active = True
    try:
        result = align_initial_robot(session)
        receipt.update(completed=True, alignment_receipt=deepcopy(result),
            pending_gates=['final_controller_native_head_pose', 'native_intrinsics_and_complete_mounted_rig',
                'native_wheel_floor_support', 'optical_scene_integrity', 'original_source_image_equivalence'])
    except Exception as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        session._reflection_initialization_hook_active = False
    return deepcopy(receipt)


def qualify_reflection_action(session, executor, config, bound):
    """Require source pixels and exact reset/task/controller/sensor equivalence."""
    from wia.runtime.sim.esi_reflection import reflection_equivalence_identity
    result = _qualify_observation(session, executor, config, bound)
    checks = dict(result.get('checks', {}))
    receipt = getattr(session, 'reflection_qualification', {}).get('original_source_view_equivalence', {})
    reference = config['parameters']['original_reference_provenance']
    checks['true_original_ESI_image_available'] = reference.get('available') is True and reference.get('kind') == 'original_ESI_image'
    checks['original_source_pixels_and_task_content_equivalent'] = bool(receipt.get('available') is True and
        receipt.get('accepted') is True and receipt.get('reference') == reference and
        receipt.get('native_rgb_matches_delivered') is True and receipt.get('action_step') == 0 and
        receipt.get('task_feature_matches') and all(row.get('accepted') is True for row in receipt['task_feature_matches']))
    try:
        state = result.get('native_state', {})
        profile = config['parameters']['reflection_sensor_profile']
        checks['initial_native_head_matches_original_reference_pose'] = bool(
            np.linalg.norm(_array(state['head_position_m']) - _array(profile['position_m'])) <= .02 and
            abs(float(np.dot(_array(state['head_quaternion_xyzw']) / np.linalg.norm(state['head_quaternion_xyzw']),
                _array(profile['quaternion_xyzw']) / np.linalg.norm(profile['quaternion_xyzw'])))) >= math.cos(math.radians(1.) / 2))
        identity = reflection_equivalence_identity(session, config, executor)
        checks['source_view_receipt_matches_exact_native_reset'] = receipt.get('identity') == identity
        image_path = Path(reference['image_path']) if reference.get('image_path') else None
        checks['immutable_original_pixels_still_present'] = bool(image_path and image_path.is_file() and
            hashlib.sha256(image_path.read_bytes()).hexdigest() == reference.get('image_sha256') and
            reference.get('source_sha256') == identity['source_sha256'])
        matched_bindings = {row['binding'] for row in receipt.get('task_feature_matches', []) if row.get('accepted')}
        checks['required_source_task_features_matched'] = set(config['parameters']['verification_contract']['required_bindings']) <= matched_bindings
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
        checks['source_view_receipt_matches_exact_native_reset'] = False
        result['equivalence_error'] = f'{type(error).__name__}: {error}'
    accepted = bool(result.get('accepted') and all(value is True for value in checks.values()))
    return {**result, 'accepted': accepted, 'checks': checks, 'equivalence_receipt': receipt,
        'evaluation_validity': 'complete' if accepted else 'unavailable',
        'reason': None if accepted else 'true_original_source_view_equivalence_pending'}


def _observation_parameters(session, config):
    supported = {MODULE: (ANSWER_OPERATION, ACTION_OPERATION),
        'wia.authoring.converters.specular_reflection_correspondence': ('answer_reflection_correspondence', 'answer_and_verify_reflection_correspondence'),
        'wia.authoring.converters.specular_reflection_spatial_relations': ('answer_reflected_spatial_relation', 'answer_and_verify_reflected_spatial_relation')}
    if (module_identity(config.get('module')) not in supported or config.get('operation') not in supported.get(module_identity(config.get('module')), ())
            or config.get('type') != 'category_operation'
            or (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('Reflection observation scene or operation mismatch.')
    return config['parameters']


def _observation_state(session, config):
    """Read the native mounted head; never substitute base or a debug camera."""
    p = _observation_parameters(session, config)
    position, quaternion = session.robot.sensors[session.camera_name].get_position_orientation()
    position, quaternion = _vector(position, 3), _vector(quaternion, 4)
    if np.linalg.norm(quaternion) <= 0:
        raise ValueError('Native head orientation is unavailable.')
    objects = _observation_objects(session, config)
    poses = {}
    for key in p['expected_optical_poses']:
        point, orientation = objects[key].get_position_orientation()
        point, orientation = _vector(point, 3), _vector(orientation, 4)
        if np.linalg.norm(orientation) <= 0:
            raise ValueError('Native optical orientation is unavailable.')
        poses[key] = {'position': point.tolist(), 'quaternion_xyzw': orientation.tolist()}
    boundary = p['reflection_sensor_profile']['boundary']
    distance = float(np.linalg.norm(position[:2] - _vector(boundary['reference_position_m'], 3)[:2]))
    return {'head_position_m': position.tolist(), 'head_quaternion_xyzw': quaternion.tolist(),
        'horizontal_displacement_m': distance,
        'within_boundary': boundary['mode'] == 'unrestricted' or distance <= boundary['limit_m'],
        'optical_poses': poses, 'optical_poses_preserved': all(
            _matches_pose(pose, p['expected_optical_poses'][key], p) for key, pose in poses.items())}


def _observation_objects(session, config):
    objects = {key: session.env.scene.object_registry('name', config['bindings'][key])
               for key in config['binding_keys']}
    if any(obj is None for obj in objects.values()):
        raise ValueError('Original optical asset or floor is absent.')
    return objects


def _qualify_observation(session, executor, config, bound):
    receipt = deepcopy(getattr(session, 'reflection_qualification', {}))
    checks = {key: receipt.get(key) for key in REFLECTION_QUALIFICATION_KEYS}
    try:
        state = _observation_state(session, config)
        objects = _observation_objects(session, config)
        checks.update(bound_native_objects=all(bound.get(key) is obj for key, obj in objects.items()),
            original_optical_poses_preserved=state['optical_poses_preserved'],
            initial_head_within_boundary=state['within_boundary'])
        accepted = all(value is True for value in checks.values())
        return {'accepted': accepted, 'checks': checks, 'native_state': state,
            'native_receipt': receipt, 'evaluation_validity': 'complete' if accepted else 'unavailable',
            'reason': None if accepted else 'native_reflection_configuration_unqualified'}
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        return {'accepted': False, 'checks': checks, 'native_receipt': receipt,
            'evaluation_validity': 'unavailable', 'reason': 'native_reflection_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}'}


class ReflectionObservationEvaluator:
    """Score the private source answer independently of latched native constraints."""

    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = _observation_parameters(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode = 0

    def begin_episode(self, observation):
        self.episode += 1
        self.answer, self.answer_history = None, []
        self.faults, self.measurement_errors, self.samples = set(), [], 0
        self.maximum_displacement_m, self.state = 0., None
        try:
            bound = _observation_objects(self.session, self.config)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            bound = {}
        self.qualification = _qualify_observation(self.session, self.controller, self.config, bound)
        return self.record(observation)

    def _sample(self):
        try:
            self.state = _observation_state(self.session, self.config)
            self.maximum_displacement_m = max(self.maximum_displacement_m, self.state['horizontal_displacement_m'])
            if not self.state['within_boundary']:
                self.faults.add('head_boundary_exceeded')
            if not self.state['optical_poses_preserved']:
                self.faults.add('original_optical_poses_changed')
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
            message = f'{type(error).__name__}: {error}'
            if message not in self.measurement_errors:
                self.measurement_errors.append(message)

    def sample_step(self):
        if not self.episode:
            return
        self._sample()
        self.samples += 1
        with (self.output / 'head_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'episode': self.episode, 'sample': self.samples, 'state': self.state,
                'faults': sorted(self.faults), 'measurement_errors': self.measurement_errors}, allow_nan=False) + '\n')

    def report_answer(self, answer, observation):
        if not self.episode:
            raise RuntimeError('An answer requires an active reflection episode.')
        self.answer = str(answer)
        self.answer_history.append({'answer': self.answer, 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step})

    def record(self, observation, *, action=None):
        self._sample()
        correct = self.answer is not None and _answer_key(self.answer) == _answer_key(self.parameters['source_qa']['answer'])
        native_valid = self.qualification['accepted'] and not self.measurement_errors
        constraints = not self.faults if native_valid else (False if self.faults else None)
        success = bool(correct and native_valid and constraints)
        items = [{'id': 'source_optical_answer', 'kind': 'answer_requirement',
            'completed': bool(correct), 'evaluation_validity': 'valid' if native_valid else 'unavailable',
            'answer_reported': self.answer is not None}]
        value = {'schema': 'private_reflection_observation_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
            'task_success': success, 'evaluation_validity': 'complete' if native_valid else 'unavailable',
            'completion_items': items, 'answer': self.answer, 'answer_correct': bool(correct),
            'answer_history': deepcopy(self.answer_history), 'constraints_satisfied': constraints,
            'continuous_constraints_satisfied': constraints, 'terminal_conditions_satisfied': success,
            'native_qualification': self.qualification, 'native_state': self.state,
            'control_samples': self.samples, 'maximum_head_horizontal_displacement_m': self.maximum_displacement_m,
            'constraint_violations': sorted(self.faults), 'measurement_errors': self.measurement_errors,
            'configuration': self.config, 'original_qa_used_for_success': True}
        write_json(self.output / 'state.json', value)
        return value


def _native_viewpoint_state(session, controller, config):
    """Reuse existing public-action history and native contact/control readback."""
    from wia.tasks.interaction_setup import _robot_contact_check
    floor = session.env.scene.object_registry('name', config['bindings']['floor'])
    support = _robot_contact_check(session.robot, [floor.prim_path], float(_array(floor.aabb[1])[2]))
    provider = getattr(controller, 'geometric_rearrangement_evidence', None)
    if not callable(provider):
        raise ValueError('Existing native public-control history is unavailable.')
    control = provider()
    if control.get('available') is not True or 'control_sequence' not in control or 'state_setter_used' not in control:
        raise ValueError('Native public-control/state-setter evidence is incomplete.')
    sim = session.pipeline.og.sim
    interval = max(float(controller.dt), float(sim.get_sim_step_dt()))
    return {'time_s': float(sim.current_time), 'maximum_sample_gap_s': interval + 1e-6,
        'control_sequence': control['control_sequence'], 'state_setter_used': control['state_setter_used'],
        'legal_viewpoint': support.get('floor_supported') is True and support.get('collision_free') is True,
        'native_floor_checks': support}


def _informative_reflection_rows(session, observation, config):
    """Accept only synchronized delivered-image features with native identities."""
    provider = getattr(session, 'task_visual_evidence', None)
    if not callable(provider):
        return {'available': False, 'objects': {}, 'reason': 'task_visual_evidence unavailable'}
    evidence = provider(observation, config['bindings'], config['parameters'])
    if not isinstance(evidence, dict) or evidence.get('available') is not True:
        return {'available': False, 'objects': {}, 'reason': 'native feature-bearing image evidence unavailable'}
    images = observation.rgb_images()
    objects = {}
    for key, row in evidence.get('objects', {}).items():
        if key not in config['parameters']['expected_optical_poses'] or not isinstance(row, dict):
            continue
        features = row.get('informative_feature_evidence')
        if (row.get('snapshot_id') != observation.snapshot_id or row.get('camera') not in images or
                row.get('native_name') != config['bindings'][key] or row.get('native_rgb_matches_delivered') is not True or
                row.get('identity_synchronized') is not True or row.get('informative') is not True or
                not isinstance(features, dict) or features.get('view_kind') not in ('direct', 'reflected') or
                not isinstance(features.get('feature_ids'), list) or not features['feature_ids']):
            continue
        objects[key] = row
    return {**evidence, 'objects': objects, 'available': True}


class ReflectionVerificationEvaluator(ReflectionObservationEvaluator):
    """Keep initial answer, physical verification, constraints and Safety distinct."""
    def begin_episode(self, observation):
        self.episode += 1
        self.answer, self.answer_history = None, []
        self.faults, self.measurement_errors, self.samples = set(), [], 0
        self.maximum_displacement_m, self.state = 0., None
        self.verification, self.verification_history = False, []
        self.transition_seen, self.last_motion, self.previous_head = False, None, None
        self.translated_viewpoint_seen = False
        self.initial_state, self.initial_motion = None, None
        self.initial_snapshot, self.initial_action_step = observation.snapshot_id, observation.action_step
        self.initial_delivered_images = set(observation.rgb_images())
        self.feature_evidence_available = False
        self.initial_visual, self.current_visual = None, None
        self.qualification = qualify_reflection_action(self.session, self.controller, self.config,
            _observation_objects(self.session, self.config))
        try:
            self.initial_state = _observation_state(self.session, self.config)
            self.initial_motion = _native_viewpoint_state(self.session, self.controller, self.config)
            self.previous_head = deepcopy(self.initial_state)
            self.last_motion = deepcopy(self.initial_motion)
            self.initial_visual = _informative_reflection_rows(self.session, observation, self.config)
            required = set(self.parameters['verification_contract']['required_bindings'])
            if self.initial_visual.get('available') is not True or not required <= set(self.initial_visual.get('objects', {})):
                raise ValueError('Initial native feature evidence is unavailable.')
            receipt = self.qualification.get('equivalence_receipt', {})
            rgb = observation.rgb_images().get(receipt.get('camera'))
            if rgb is None or hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest() != receipt.get('mounted_rgb_sha256'):
                raise ValueError('Equivalence receipt does not bind the delivered initial head pixels.')
        except (AttributeError, ValueError, TypeError, KeyError, RuntimeError) as error:
            self.measurement_errors.append(f'{type(error).__name__}: {error}')
        return self.record(observation)

    def _sample(self):
        super()._sample()
        try:
            if self.state is None or self.initial_state is None or self.initial_motion is None:
                raise ValueError('Initial native viewpoint baseline is unavailable.')
            motion = _native_viewpoint_state(self.session, self.controller, self.config)
            if self.last_motion is not None:
                elapsed = motion['time_s'] - self.last_motion['time_s']
                moved = not np.allclose(self.state['head_position_m'], self.previous_head['head_position_m'], atol=1e-7, rtol=0.)
                turned = not np.allclose(self.state['head_quaternion_xyzw'], self.previous_head['head_quaternion_xyzw'], atol=1e-7, rtol=0.)
                if elapsed < 0 or elapsed > motion['maximum_sample_gap_s']:
                    raise ValueError('Native viewpoint trajectory has a clock gap or reset.')
                if elapsed == 0 and (moved or turned):
                    self.faults.add('head_pose_changed_without_native_time')
                if motion['control_sequence'] < self.last_motion['control_sequence']:
                    raise ValueError('Public native control history reset during reflection episode.')
            displaced = not np.allclose(self.state['head_position_m'], self.initial_state['head_position_m'], atol=1e-7, rtol=0.)
            rotated = not np.allclose(self.state['head_quaternion_xyzw'], self.initial_state['head_quaternion_xyzw'], atol=1e-7, rtol=0.)
            self.transition_seen |= bool((displaced or rotated) and motion['time_s'] > self.initial_motion['time_s'] and
                motion['control_sequence'] > self.initial_motion['control_sequence'])
            self.translated_viewpoint_seen |= bool(displaced and motion['time_s'] > self.initial_motion['time_s'] and
                motion['control_sequence'] > self.initial_motion['control_sequence'])
            if motion['state_setter_used']:
                self.faults.add('post_start_simulator_state_setter')
            if not motion['legal_viewpoint']:
                self.faults.add('illegal_native_viewpoint')
            from wia.runtime.sim.esi_reflection import reflection_equivalence_identity
            if reflection_equivalence_identity(self.session, self.config, self.controller) != self.qualification.get('equivalence_receipt', {}).get('identity'):
                self.faults.add('qualified_optics_or_sensor_configuration_changed')
            self.last_motion, self.previous_head = motion, deepcopy(self.state)
        except (AttributeError, ValueError, TypeError, KeyError, RuntimeError) as error:
            message = f'{type(error).__name__}: {error}'
            if message not in self.measurement_errors:
                self.measurement_errors.append(message)

    def report_answer(self, answer, observation):
        self._sample()
        accepted = not self.transition_seen
        self.answer_history.append({'answer': str(answer), 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'initial_reference_answer_accepted': accepted})
        if accepted:
            self.answer = str(answer)

    def _verify_features(self, observation):
        evidence = _informative_reflection_rows(self.session, observation, self.config)
        self.current_visual = evidence
        required = set(self.parameters['verification_contract']['required_bindings'])
        self.feature_evidence_available = bool(evidence.get('available') is True and
            required <= set(evidence.get('objects', {})) and self.initial_visual and
            self.initial_visual.get('available') and required <= set(self.initial_visual.get('objects', {})))
        if not self.feature_evidence_available:
            return
        contract = self.parameters['verification_contract']
        history = getattr(observation, 'action_history', ())
        if self.translated_viewpoint_seen and not history:
            raise ValueError('Executed public viewpoint-action history is unavailable.')
        successful_view_action = any(getattr(row, 'result_summary', None) == 'completed' and
            getattr(row, 'action', None) in contract['viewpoint_actions'] for row in history)
        if (not self.translated_viewpoint_seen or not successful_view_action or self.answer is None or
                observation.snapshot_id == self.initial_snapshot or observation.action_step <= self.initial_action_step):
            return
        if np.allclose(self.state['head_position_m'], self.initial_state['head_position_m'], atol=1e-7, rtol=0.):
            return
        new_evidence = []
        for key in contract['required_bindings']:
            row = evidence['objects'].get(key)
            if row is None or row['camera'] == 'initial_overview_rgb':
                return  # The frozen Global can inform the answer but is not a second viewpoint.
            features = row['informative_feature_evidence']
            if any(field not in features for field in ('reference_snapshot_id', 'reference_frame', contract['mechanism'])):
                raise ValueError('Native verification feature/reference proof is incomplete.')
            baseline = self.initial_visual['objects'].get(key, {}).get('informative_feature_evidence', {})
            old = set(baseline.get('feature_ids', []))
            new = set(features['feature_ids']) - old
            changed_kind = baseline.get('view_kind') != features.get('view_kind')
            if (not (new or changed_kind) or features.get(contract['mechanism']) is not True or
                    features.get('reference_snapshot_id') != self.initial_snapshot or
                    features.get('reference_frame') != 'initial_original_ESI_observer'):
                return
            if features.get('view_kind') == 'reflected' and features.get('mirror_binding') != 'mirror':
                if 'mirror_binding' not in features:
                    raise ValueError('Native second-mirror identity evidence is unavailable.')
                return
            new_evidence.append({'binding': key, 'native_name': row['native_name'], 'camera': row['camera'],
                'new_feature_ids': sorted(new), 'view_kind': features['view_kind'],
                'mechanism': contract['mechanism'], 'feature_evidence': features})
        if not self.faults and not self.measurement_errors:
            self.verification = True
            self.verification_history.append({'snapshot_id': observation.snapshot_id,
                'action_step': observation.action_step, 'native_head_pose': deepcopy(self.state), 'objects': new_evidence})

    def record(self, observation, *, action=None):
        from wia.tasks.completion import progress
        self._sample()
        try:
            self._verify_features(observation)
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError) as error:
            self.measurement_errors.append(f'{type(error).__name__}: {error}')
        correct = self.answer is not None and _answer_key(self.answer) == _answer_key(self.parameters['source_qa']['answer'])
        valid = bool(self.qualification['accepted'] and not self.measurement_errors and self.feature_evidence_available)
        constraints = not self.faults if valid else (False if self.faults else None)
        success = bool(valid and correct and self.verification and constraints)
        items = [{'id': 'initial_optical_answer', 'kind': 'answer_requirement', 'completed': bool(correct)},
            {'id': 'informative_second_view', 'kind': 'historical_checkpoint', 'completed': self.verification}]
        for item in items:
            item['evaluation_validity'] = 'valid' if valid else 'unavailable'
        initial_rows = self.initial_visual.get('objects', {}) if self.initial_visual else {}
        shortcut_rows = [{'binding': key, 'camera': row['camera'], 'native_name': row['native_name']}
            for key, row in initial_rows.items() if row['informative_feature_evidence']['view_kind'] == 'direct']
        audited_cameras = set(self.initial_visual.get('audited_camera_names', [])) if self.initial_visual else set()
        for row in self.initial_visual.get('view_evidence', []) if self.initial_visual else []:
            features = row.get('informative_feature_evidence', {})
            key = row.get('binding')
            if (key in self.config['bindings'] and row.get('snapshot_id') == self.initial_snapshot and
                    row.get('camera') in self.initial_delivered_images and row.get('native_name') == self.config['bindings'][key] and
                    row.get('native_rgb_matches_delivered') is True and row.get('identity_synchronized') is True and
                    row.get('informative') is True and features.get('view_kind') == 'direct' and features.get('feature_ids')):
                shortcut_rows.append({'binding': key, 'camera': row['camera'], 'native_name': row['native_name']})
        value = {'schema': ACTION_REVISION, 'task_revision': ACTION_REVISION, 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action, 'episode': self.episode,
            'task_success': success, 'evaluation_validity': 'complete' if valid else 'unavailable',
            'answer_correct': bool(correct), 'initial_answer': self.answer, 'answer_history': deepcopy(self.answer_history),
            'physical_verification': self.verification, 'verification_history': deepcopy(self.verification_history),
            'completion_items': items, 'constraints_satisfied': constraints,
            'continuous_constraints_satisfied': constraints, 'terminal_conditions_satisfied': success,
            'Safety': None, 'Safety_scope': 'Shared runtime safety is reported independently.',
            'native_qualification': self.qualification, 'native_state': self.state,
            'control_samples': self.samples, 'maximum_head_horizontal_displacement_m': self.maximum_displacement_m,
            'constraint_violations': sorted(self.faults), 'measurement_errors': self.measurement_errors,
            'initial_information_audit': {'input_condition': self.parameters['input_condition'],
                'direct_view_evidence': shortcut_rows, 'global_included': 'initial_overview_rgb' in self.initial_delivered_images,
                'global_direct_view_shortcut_observed': any(row['camera'] == 'initial_overview_rgb' for row in shortcut_rows),
                'audited_camera_names': sorted(audited_cameras),
                'all_views_shortcut_audit_complete': bool(self.initial_visual and self.initial_visual.get('all_delivered_views_audited') is True and
                    self.initial_delivered_images <= audited_cameras)},
            'native_feature_evidence': self.current_visual, 'configuration': self.config}
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        return value


def _convert_lift(source, context):
    raw, payload, q, prop, mirror, camera = _source_geometry(source)
    setup = payload.get('mirror_setup', {})
    bindings = {'prop': {key: prop[key] for key in ('name', 'category', 'model')},
        'mirror': {'name': setup.get('name') or 'render_mirror_main',
            'category': setup.get('category') or 'mirror', 'model': setup.get('model') or 'tytkbq'},
        'floor': {'name': payload['floor_name']}}
    floor_z = float(camera['position'][2] - payload['camera_poses']['height_m'])
    yaw = math.atan2(mirror['position'][1] - camera['position'][1], mirror['position'][0] - camera['position'][0])
    patches = [{'operation': 'robot_pose', 'phase': 'before_construction',
        'position': [*camera['position'][:2], floor_z + .1], 'yaw_deg': math.degrees(yaw), 'head_pitch_deg': 0.},
        # The original common.py creates this prop in postprocessing. It is
        # already movable by default; enabling its actual collision/gravity
        # here avoids a nonexistent before-construction object binding.
        {'operation': 'object_physics', 'phase': 'after_postprocessing', 'binding': 'prop', 'visual_only': False}]
    parameters = {'original_prop': {key: prop[key] for key in ('name', 'category', 'model')},
        'expected_optical_poses': {'mirror': {key: mirror[key] for key in ('position', 'quaternion_xyzw')},
            'prop': {'position': prop['real_position'], 'quaternion_xyzw': prop.get('quaternion_xyzw',
                prop.get('orientation', [0., 0., 0., 1.]))}},
        'proposed_start_xy': camera['position'][:2], 'start_tolerance_m': .12,
        'start_heading_tolerance_deg': 3., 'minimum_lift_m': .05, 'working_arm': 'left',
        'accessible_target_relocation': True,
        'stability_steps': 3, 'minimum_up_dot': .95, 'floor_z_m': floor_z,
        'floor_height_tolerance_m': .005, 'minimum_floor_normal_z': .5,
        'asset_position_tolerance_m': .02, 'asset_orientation_tolerance_deg': 2.,
        'mirror_clearance_m': .03, 'maximum_stopped_linear_speed_m_s': .02,
        'maximum_stopped_angular_speed_rad_s': .1}
    evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': LIFT_OPERATION,
        'binding_keys': ['prop', 'mirror', 'floor'], 'parameters': parameters}
    name = prop['category'].replace('_', ' ')
    public = {'objective': f'Find the {name}, pick it up, and hold it securely.',
        'success_criteria': f'Hold the original {name} in the gripper, stably lifted at least 0.05 m '
            'above its initial lowest-point height.',
        'environment': 'The original indoor scene and named task object are retained.',
        'constraints': ['Find the object using your mounted observations and legitimate episode history.',
            'Use the existing guarded base, arm, trunk, rotation and gripper controls.',
            'Preserve task supports and reference objects. Noncritical native movable furniture may be moved using guarded arm and gripper actions; retain floor support and collision guards.']}
    task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches,
        capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper'],
        qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': True})
    task['defaults']['max_steps'] = int(context.config.get('reflection_authoring_max_steps', 300))
    policy = {'mode': 'manipulation', 'group_bindings': ['prop'], 'manipulation_bindings': ['prop'],
        'relocation_policy': 'independent_groups', 'placement_groups': [['prop']], 'source_region': raw['room'],
        'source_camera_policy': 'preserve', 'support_search': 'measured_surfaces', 'allow_fallback_table': True,
        'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85,
        'warning_codes': ['optical_layout_changed_by_accessible_target_placement', 'native_initial_reflection_only_unverified']}
    return {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
        'task_definition': task, 'interaction_policy': policy, 'status': 'generated', 'reuse_level': 'L2',
        'qa_preserved': True, 'operation_recipe': LIFT_OPERATION,
        'available_operation_recipes': [LIFT_OPERATION],
        'change_summary': ['Retain the original source variant, prop asset/scale/orientation, furniture and optical reference.',
            'Enable collisions and gravity only on the original queried prop after original construction.',
            'Place the portable prop on an accessible measured support when needed, preserving its identity and dimensions.',
            'Require guarded acquisition and a measured 5 cm lowest-AABB lift; no survey circuit or return is added.'],
        'warning_codes': ['assumption_review_required', 'native_grasp_and_lift_unverified',
            'native_initial_reflection_only_unverified', 'optical_layout_changed_by_accessible_target_placement',
            'mounted_reflection_visibility_unverified', 'source_view_is_not_robot_clearance'],
        'assumptions': [{'doubt': 'The original visual-only prop and saved observer do not certify a reachable, physically supported pickup.',
            'chosen_assumption': 'Enable only this original prop for manipulation. Ask shared initialization for an existing support '
                'first and, when needed, an accessible placement with original scale and orientation. '
                'Require accepted initialized reference, actual support and guarded native grasp evidence.',
            'alternatives': ['Keep the original floor placement if native reach is qualified.',
                'Retain a diagnostic failure if no valid support or safe robot stance exists.'],
            'scope': source.relative_path, 'risk': 'An accessible target placement can alter the adapted optical scene. '
                'The unchanged original QA is historical auxiliary evidence, not a claim about adapted reflected pixels.'},
            {'doubt': 'Source visibility annotations and planar geometry do not establish reflected-only visibility in actual mounted images.',
             'chosen_assumption': 'Prefer the original observer-side initialization proposal without publishing optical hints. '
                'Keep native_initial_reflection_only_unverified prominent until real mounted-image evidence exists.',
             'alternatives': ['Review actual initial robot images and retain failed visibility cases.'],
             'scope': 'All new operation instances', 'risk': 'A safe repaired stance or supported prop height can expose the prop directly '
                'or remove the useful reflected view; no initial optical condition is certified on CPU.'}],
        'private_source_evidence': {'original_reference_camera': camera, 'source_prop': prop,
            'original_visibility_annotations': q.get('evidence', {}), 'native_reflection_only_verified': False,
            'source_qa_remains_original_reference_only': True, 'native_execution_verified': False},
        'source_view_proposal': {'family': 'original_source_observer', 'start_position': camera['position'],
            'provenance': 'question_data.render.camera_pose or scene_setup.camera',
            'native_reachability_proven': False, 'native_reflection_only_proven': False},
        'limitations': ['No native pickup, reflected-only observation, support feasibility or end-to-end task execution was run by this converter.',
            'The original mirror and camera remain fixed; the adapted accessible target placement may have different optical evidence.',
            'Native safety/contact checks are sampled at guarded control steps and do not certify every swept surface.',
            'Source answers, object poses and optical annotations remain private; they cannot substitute for actual held-object readback.',
            'The proposed 300-decision budget, runtime cost, throughput and success rate remain unmeasured.'],
        'attempted_repairs': ['Declarative original-prop physics restoration and independent accessible support request.'],
        'next_step': 'Qualify accepted native initialization/support and actual initial mounted visibility, then execute guarded pickup and lift.'}


def _configuration(session, config):
    if (config.get("type") != "category_operation" or module_identity(config.get("module")) != MODULE
            or config.get("operation") != OPERATION or (config.get("scene"), config.get("room")) != (
                session.scene_name, session.room_name)):
        raise ValueError("Mirror survey must match the original native scene and room.")
    parameters = config["parameters"]
    if set(config["binding_keys"]) != {"mirror", "prop", "floor"}:
        raise ValueError("Mirror survey requires all original optical assets and floor.")
    if (parameters["minimum_excursion_m"] < MINIMUM_EXCURSION_M
            or parameters["minimum_bearing_change_deg"] < MINIMUM_BEARING_DEG
            or parameters["return_tolerance_m"] * 2 >= parameters["minimum_excursion_m"]
            or parameters["stability_steps"] < 1):
        raise ValueError("Survey stations must be meaningfully separated and stable.")
    return parameters


def _optical_poses(objects):
    result = {}
    for key in ("mirror", "prop"):
        position, quaternion = objects[key].get_position_orientation()
        position, quaternion = _vector(position, 3), _vector(quaternion, 4)
        if np.linalg.norm(quaternion) == 0:
            raise ValueError("Invalid native optical orientation.")
        result[key] = {"position": position.tolist(), "quaternion_xyzw": quaternion.tolist()}
    return result


def _same_poses(current, expected, position_tolerance, orientation_tolerance_deg):
    for key in ("mirror", "prop"):
        if np.linalg.norm(_vector(current[key]["position"], 3) - _vector(expected[key]["position"], 3)) > position_tolerance:
            return False
        a, b = (_vector(row[key]["quaternion_xyzw"], 4) for row in (current, expected))
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        if math.degrees(2 * math.acos(float(np.clip(abs(a @ b), 0., 1.)))) > orientation_tolerance_deg:
            return False
    return True


def _objects(session, config):
    objects = {key: session.env.scene.object_registry("name", config["bindings"][key])
        for key in ("mirror", "prop", "floor")}
    if any(obj is None for obj in objects.values()):
        raise ValueError("Original optical asset or floor is absent.")
    return objects


def _native_state(session, objects, parameters):
    """Read real poses, velocities, contacts, room map and visual-asset clearance."""
    robot = session.robot
    position, quaternion = robot.get_position_orientation()
    position, quaternion = _vector(position, 3), _vector(quaternion, 4)
    length = np.linalg.norm(quaternion)
    if length == 0:
        raise ValueError("Invalid native robot orientation.")
    x, y, z, w = quaternion / length
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    up_dot = float(1 - 2 * (x * x + y * y))
    poses = _optical_poses(objects)
    mirror = _array(poses["mirror"]["position"])
    facing = _angle(yaw, math.atan2(mirror[1] - position[1], mirror[0] - position[0]))
    low, high = (_vector(value, 3) for value in robot.aabb)
    if np.any(high <= low):
        raise ValueError("Invalid native robot bounds.")
    asset_clearance = {}
    for key in ("mirror", "prop"):
        other_low, other_high = (_vector(value, 3) for value in objects[key].aabb)
        gap = np.maximum(np.maximum(other_low - high, low - other_high), 0.)
        asset_clearance[key] = float(np.linalg.norm(gap))
    floor_path = objects["floor"].prim_path
    floor_contacts, nonfloor = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        external = [str(body) for body in (contact.body0, contact.body1)
            if not (str(body) == robot.prim_path or str(body).startswith(robot.prim_path + "/"))]
        if not external:
            continue
        if all(body == floor_path or body.startswith(floor_path + "/") for body in external):
            p, normal = _vector(contact.position, 3), _vector(contact.normal, 3)
            floor_contacts.append({"position": p.tolist(), "normal": normal.tolist()})
        else:
            nonfloor.extend(external)
    floor_supported = bool(floor_contacts and all(abs(row["position"][2] - parameters["proposed_floor_z_m"])
        <= parameters["floor_height_tolerance_m"] and abs(row["normal"][2]) >= parameters["minimum_floor_normal_z"]
        for row in floor_contacts))
    region = session.env.scene.seg_map.get_room_instance_by_point(position[:2])
    speed = float(np.linalg.norm(_vector(robot.get_linear_velocity(), 3)))
    angular_speed = float(np.linalg.norm(_vector(robot.get_angular_velocity(), 3)))
    source_preserved = _same_poses(poses, parameters["expected_optical_poses"],
        parameters["source_pose_tolerance_m"], parameters["asset_orientation_tolerance_deg"])
    valid = bool(floor_supported and not nonfloor and region == session.room_name
        and up_dot >= parameters["minimum_up_dot"] and source_preserved
        and min(asset_clearance.values()) >= parameters["visual_asset_clearance_m"])
    return {"position": position.tolist(), "quaternion_xyzw": quaternion.tolist(), "yaw_rad": yaw,
        "up_dot": up_dot, "optical_poses": poses, "source_optical_poses_preserved": source_preserved,
        "mirror_bearing_rad": _bearing(position, mirror), "facing_error_deg": math.degrees(facing),
        "linear_speed_m_s": speed, "angular_speed_rad_s": angular_speed,
        "floor_contacts": floor_contacts, "native_floor_supported": floor_supported,
        "nonfloor_contacts": nonfloor, "room": region, "visual_asset_aabb_clearance_m": asset_clearance,
        "stopped_facing_mirror": bool(math.degrees(facing) <= parameters["facing_tolerance_deg"]
            and speed <= parameters["maximum_stopped_linear_speed_m_s"]
            and angular_speed <= parameters["maximum_stopped_angular_speed_rad_s"]),
        "state_valid": valid}


def qualify_operation(session, executor, config, bound):
    """Read preconditions only; do not step physics, act, or reposition anything."""
    if config.get('operation') == ACTION_OPERATION:
        return qualify_reflection_action(session, executor, config, bound)
    if config.get('operation') == ANSWER_OPERATION:
        return _qualify_observation(session, executor, config, bound)
    if config.get('operation') == LIFT_OPERATION:
        return _qualify_lift(session, executor, config, bound)
    checks = {}
    try:
        parameters = _configuration(session, config)
        objects = _objects(session, config)
        measured = _native_state(session, objects, parameters)
        checks = {"bound_native_objects": all(bound.get(key) is objects[key] for key in objects),
            "base_motion_capabilities": bool(executor.capabilities.base_translation and executor.capabilities.base_rotation),
            "native_safe_start": measured["state_valid"],
            "declared_stance_realized": bool(np.linalg.norm(_array(measured["position"])[:2]
                - _array(parameters["proposed_start_xy"])) <= parameters["start_tolerance_m"]),
            "initial_heading_toward_mirror": measured["facing_error_deg"] <= parameters["facing_tolerance_deg"]}
        accepted = bool(all(checks.values()))
        return {"accepted": accepted, "checks": checks, "native_state": measured,
            "reason": None if accepted else "native_mirror_survey_initialization_unqualified",
            "route_qualified": False, "reflected_visibility_qualified": False}
    except Exception as error:
        return {"accepted": False, "checks": checks, "reason": "native_mirror_survey_measurement_unavailable",
            "exception_summary": f"{type(error).__name__}: {error}", "route_qualified": False}


class MirrorSurveyEvaluator:
    """Count actual ordered native stops; source answers never select success."""
    def __init__(self, session, controller, config, output):
        self.parameters = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = _objects(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_step_m = float(controller.config.max_translation_speed_m_s * controller.dt
            + controller.config.translation_tolerance_m)
        if not math.isfinite(self.max_step_m) or self.max_step_m <= 0:
            raise ValueError("Native controller sampling bound is unavailable.")
        self.active, self.episode = False, 0

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = _native_state(self.session, self.objects, self.parameters)
        self.stage, self.consecutive, self.samples = 0, 0, 0
        self.visits, self.travel_m = [], 0.
        self.previous = _array(self.initial["position"])
        self.continuous, self.safe = True, bool(self.initial["state_valid"])
        self.start_realized = bool(np.linalg.norm(self.previous[:2] - _array(self.parameters["proposed_start_xy"]))
            <= self.parameters["start_tolerance_m"])
        self.active = True
        return self.record(observation, action=None)

    def _observe(self, count_step):
        state = _native_state(self.session, self.objects, self.parameters)
        point = _array(state["position"])
        displacement = float(np.linalg.norm(point - self.previous))
        self.continuous = bool(self.continuous and displacement <= self.max_step_m)
        self.travel_m += displacement
        self.previous = point
        assets_unchanged = _same_poses(state["optical_poses"], self.initial["optical_poses"],
            self.parameters["asset_position_tolerance_m"], self.parameters["asset_orientation_tolerance_deg"])
        self.safe = bool(self.safe and state["state_valid"] and assets_unchanged)
        excursion = float(np.linalg.norm(point[:2] - _array(self.initial["position"])[:2]))
        bearing_change = math.degrees(_angle(state["mirror_bearing_rad"], self.initial["mirror_bearing_rad"]))
        if count_step:
            self.samples += 1
            # Initial stop, displaced side stop, then return to initial station.
            conditions = [excursion <= self.parameters["return_tolerance_m"],
                excursion >= self.parameters["minimum_excursion_m"] and bearing_change >= self.parameters["minimum_bearing_change_deg"],
                excursion <= self.parameters["return_tolerance_m"]]
            if self.stage < 3:
                valid = bool(conditions[self.stage] and state["stopped_facing_mirror"] and self.safe and self.continuous)
                self.consecutive = self.consecutive + 1 if valid else 0
                if self.consecutive >= self.parameters["stability_steps"]:
                    self.visits.append({"stage": self.stage, "sample": self.samples,
                        "position": state["position"], "excursion_m": excursion, "bearing_change_deg": bearing_change})
                    self.stage += 1
                    self.consecutive = 0
        self.last = state
        return state, excursion, bearing_change

    def sample_step(self):
        if self.active:
            self._observe(True)

    def record(self, observation, *, action):
        if not self.active:
            return None
        state, excursion, bearing_change = self._observe(False)
        checks = {"ordered_native_stops": self.stage == 3, "native_safe_trajectory": self.safe,
            "continuous_control_samples": self.continuous, "declared_stance_realized": self.start_realized,
            "returned_to_start": excursion <= self.parameters["return_tolerance_m"],
            "stopped_facing_mirror": state["stopped_facing_mirror"],
            "measured_round_trip_displacement": self.travel_m >= (
                2 * self.parameters["minimum_excursion_m"] - self.parameters["return_tolerance_m"])}
        value = {"schema": "private_mirror_survey_v1", "episode": self.episode,
            "snapshot_id": observation.snapshot_id, "action_step": observation.action_step, "action": action,
            "task_success": bool(all(checks.values())), "checks": checks, "native_state": state,
            "ordered_visits": self.visits, "control_samples": self.samples, "sampled_travel_m": self.travel_m,
            "current_excursion_m": excursion, "current_bearing_change_deg": bearing_change,
            "maximum_sample_displacement_m": self.max_step_m, "configuration": self.config,
            "original_qa_used_for_success": False, "reflected_visibility_certified": False}
        write_json(self.output / "state.json", value)
        with (self.output / "trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == ACTION_OPERATION:
        return ReflectionVerificationEvaluator(session, controller, config, output)
    if config.get('operation') == ANSWER_OPERATION:
        return ReflectionObservationEvaluator(session, controller, config, output)
    if config.get('operation') == LIFT_OPERATION:
        return ReflectedPropLiftEvaluator(session, controller, config, output)
    return MirrorSurveyEvaluator(session, controller, config, output)


def _lift_parameters(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE
            or config.get('operation') != LIFT_OPERATION
            or (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)
            or set(config.get('binding_keys', [])) != {'prop', 'mirror', 'floor'}):
        raise ValueError('Original prop-lift scene/binding contract mismatch.')
    p = config['parameters']
    if p.get('minimum_lift_m') != .05 or p.get('working_arm') != 'left' or p.get('stability_steps', 0) < 1:
        raise ValueError('Use the declared original-prop 5 cm guarded left-gripper lift.')
    return p


def _under(body, root):
    return str(body) == str(root) or str(body).startswith(str(root) + '/')


def _native_grasp(robot, arm):
    reading = int(robot.is_grasping(arm))
    held = robot._ag_obj_in_hand[arm]
    return {'holding': {-1: False, 1: True}.get(reading),
        'object_name': held.name if held is not None else None,
        'constraint_active': robot._ag_obj_constraints[arm] is not None}


def _contacts(obj, support_paths, *, allowed_contact=None, session=None):
    supports, forbidden, allowed = [], [], []
    for contact in obj.contact_list():
        if session is not None and legal_background_contact(session, contact.body0, contact.body1):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        owned = [body for body in bodies if _under(body, obj.prim_path)]
        external = [body for body in bodies if not _under(body, obj.prim_path)]
        if not external:
            continue
        row = {'bodies': bodies, 'position': _vector(contact.position, 3).tolist(),
            'normal': _vector(contact.normal, 3).tolist()}
        if owned and all(any(_under(body, path) for path in support_paths) for body in external):
            supports.append(row)
        elif owned and allowed_contact is not None and allowed_contact(owned, external):
            allowed.append(row)
        else:
            forbidden.append(row)
    return supports, forbidden, allowed


def _matches_pose(current, expected, p):
    a, b = (_vector(row['quaternion_xyzw'], 4) for row in (current, expected))
    if min(np.linalg.norm(a), np.linalg.norm(b)) <= 0:
        return False
    angle = math.degrees(2 * math.acos(float(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b), 0., 1.))))
    return bool(math.dist(current['position'], expected['position']) <= p['asset_position_tolerance_m']
        and angle <= p['asset_orientation_tolerance_deg'])


def _lift_state(session, config):
    p, objects = _lift_parameters(session, config), _objects(session, config)
    robot, prop = session.robot, objects['prop']
    point, quaternion = (_vector(value, size) for value, size in zip(robot.get_position_orientation(), (3, 4)))
    if np.linalg.norm(quaternion) <= 0:
        raise ValueError('Native robot orientation is unavailable.')
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    up = float(1 - 2 * (x*x + y*y))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    chassis = robot.links[robot.base_footprint_link_name]
    low, high = (_vector(value, 3) for value in chassis.aabb)
    body_low, body_high = (_vector(value, 3) for value in robot.aabb)
    if np.any(high <= low) or np.any(body_high <= body_low):
        raise ValueError('Native physical robot bounds are unavailable.')
    poses = _optical_poses(objects)
    mirror_low, mirror_high = (_vector(value, 3) for value in objects['mirror'].aabb)
    mirror_gap = float(np.linalg.norm(np.maximum(np.maximum(mirror_low - body_high, body_low - mirror_high), 0.)))
    mirror_fixed = _matches_pose(poses['mirror'], p['expected_optical_poses']['mirror'], p)
    known_rooms = [session.env.scene.seg_map.get_room_instance_by_point(np.array([px, py]))
        for px in (low[0], (low[0]+high[0])/2, high[0]) for py in (low[1], (low[1]+high[1])/2, high[1])]
    arm = p['working_arm']
    gripper_links = list(robot.finger_links[arm]) + [robot.eef_links[arm]]
    gripper_paths = [link.prim_path for link in gripper_links]
    robot_contacts, robot_bad, gripper_contacts = _contacts(robot, [objects['floor'].prim_path], session=session,
        allowed_contact=lambda owned, external: all(any(_under(body, path) for path in gripper_paths) for body in owned)
            and all(_under(body, prop.prim_path) for body in external))
    supported = bool(robot_contacts and all(abs(row['position'][2] - p['floor_z_m']) <= p['floor_height_tolerance_m']
        and abs(row['normal'][2]) >= p['minimum_floor_normal_z'] for row in robot_contacts))
    support_paths = list(support_contact_paths(session, prop.name)) or [objects['floor'].prim_path]
    prop_contacts, prop_bad, _ = _contacts(prop, support_paths,
        allowed_contact=lambda owned, external: all(any(_under(body, path) for path in gripper_paths) for body in external))
    prop_supported = bool(prop_contacts and all(abs(row['normal'][2]) >= .5 for row in prop_contacts))
    prop_low, prop_high = (_vector(value, 3) for value in prop.aabb)
    if np.any(prop_high <= prop_low):
        raise ValueError('Original prop native AABB is unavailable.')
    expected = p['original_prop']
    identity = all(getattr(prop, key, None) == expected[key] for key in ('name', 'category', 'model'))
    grasp = _native_grasp(robot, arm)
    actual_held = bool(grasp['holding'] is True and grasp['object_name'] == prop.name and grasp['constraint_active'])
    linear, angular = (_vector(value, 3) for value in (chassis.get_linear_velocity(), chassis.get_angular_velocity()))
    prop_linear, prop_angular = (_vector(value, 3) for value in (prop.get_linear_velocity(), prop.get_angular_velocity()))
    accepted_position = reference_position(session, prop.name, p['expected_optical_poses']['prop']['position'], root=True)
    reference = {**p['expected_optical_poses']['prop'], 'position': accepted_position.tolist()}
    original_position = _vector(p['expected_optical_poses']['prop']['position'], 3)
    vertical_only = bool(np.linalg.norm(accepted_position[:2] - original_position[:2]) <= 1e-6)
    accessible_reference = bool(vertical_only or p.get('accessible_target_relocation') is True)
    valid = bool(supported and not robot_bad and not prop_bad and up >= p['minimum_up_dot']
        and all(room == session.room_name for room in known_rooms) and mirror_fixed
        and mirror_gap >= p['mirror_clearance_m'] and identity and not prop.visual_only and not prop.fixed_base
        and not objects['floor'].visual_only and accessible_reference)
    return {'position': point.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'yaw_rad': yaw,
        'up_dot': up, 'native_robot_floor_supported': supported, 'native_floor_contacts': robot_contacts,
        'nonfloor_robot_contacts': robot_bad, 'allowed_gripper_prop_contacts': gripper_contacts,
        'footprint_rooms': known_rooms, 'mirror_preserved': mirror_fixed, 'mirror_clearance_m': mirror_gap,
        'optical_poses': poses, 'prop_identity_valid': identity, 'prop_scale': _vector(prop.scale, 3).tolist(),
        'prop_aabb_min': prop_low.tolist(), 'prop_aabb_max': prop_high.tolist(),
        'prop_supported': prop_supported, 'prop_support_contacts': prop_contacts, 'prop_forbidden_contacts': prop_bad,
        'accepted_prop_reference': reference, 'accepted_reference_vertical_only': vertical_only,
        'prop_matches_initialized_reference': _matches_pose(poses['prop'], reference, p),
        'native_grasp': grasp, 'correct_original_prop_held': actual_held,
        'linear_speed_m_s': float(np.linalg.norm(linear)), 'angular_speed_rad_s': float(np.linalg.norm(angular)),
        'prop_linear_speed_m_s': float(np.linalg.norm(prop_linear)), 'prop_angular_speed_rad_s': float(np.linalg.norm(prop_angular)),
        'state_valid': valid}


def _qualify_lift(session, executor, config, bound):
    checks = {}
    try:
        p, objects = _lift_parameters(session, config), _objects(session, config)
        state = _lift_state(session, config)
        pose_match = math.dist(state['position'][:2], p['proposed_start_xy']) <= p['start_tolerance_m']
        repair = verified_repaired_start(session, session.room_name, _array(state['position']),
            _array(state['quaternion_xyzw']), position_tolerance_m=p['start_tolerance_m'],
            heading_tolerance_deg=p['start_heading_tolerance_deg'])
        grasp = state['native_grasp']
        checks = {'original_bindings_resolved': all(bound.get(key) is obj for key, obj in objects.items()),
            'actual_collidable_R1': type(session.robot).__name__ == 'R1' and not session.robot.visual_only,
            'guarded_navigation_and_manipulation': bool(executor.capabilities.base_translation and executor.capabilities.base_rotation
                and executor.capabilities.ee_translation and executor.capabilities.gripper),
            'native_safe_initialized_robot': state['state_valid'],
            'original_or_accepted_same_room_start': bool(pose_match or repair['accepted']),
            'original_prop_at_accepted_initialized_reference': state['prop_matches_initialized_reference'],
            'native_prop_physically_supported': state['prop_supported'],
            'native_prop_stable': state['prop_linear_speed_m_s'] <= .02 and state['prop_angular_speed_rad_s'] <= .1,
            'gripper_initially_empty': grasp['holding'] is False and grasp['object_name'] is None and not grasp['constraint_active']}
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_state': state, 'verified_repaired_start': repair,
            'reason': None if accepted else 'native_reflected_prop_pickup_precondition_failed',
            'reflected_visibility_qualified': False, 'native_grasp_execution_qualified': False}
    except (ValueError, RuntimeError, KeyError, AttributeError, TypeError, IndexError) as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_reflected_prop_pickup_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}', 'reflected_visibility_qualified': False}


class ReflectedPropLiftEvaluator(GraspLiftEvaluator):
    """Reuse actual native identity/constraint/AABB lift with sampled scene safety."""
    def __init__(self, session, controller, config, output):
        self.operation_config = config
        self.parameters = _lift_parameters(session, config)
        objects = _objects(session, config)
        super().__init__(session, controller, {'type': 'grasp_lift', 'scene': config['scene'],
            'room': config['room'], 'target': objects['prop'].name, 'minimum_lift_m': .05}, output)

    def begin_episode(self, observation):
        self.qualification = _qualify_lift(self.session, self.controller, self.operation_config,
            _objects(self.session, self.operation_config))
        if not self.qualification['accepted']:
            raise ValueError(self.qualification['reason'])
        state = self.qualification['native_state']
        self.previous = _array(state['position'])
        self.previous_prop = _array(state['optical_poses']['prop']['position'])
        self.previous_yaw = state['yaw_rad']
        self.previous_time = float(self.session.pipeline.og.sim.current_time)
        self.initial_scale = state['prop_scale']
        self.faults, self.samples, self.stable_samples = set(), 0, 0
        super().begin_episode(observation)

    def _held_lift(self, state):
        return bool(state['correct_original_prop_held'] and not state['prop_support_contacts']
            and state['prop_aabb_min'][2] - self.initial['aabb_min'][2] >= .05
            and state['prop_linear_speed_m_s'] <= .02 and state['prop_angular_speed_rad_s'] <= .1)

    def _safe(self, state):
        return bool(state['state_valid'] and np.allclose(state['prop_scale'], self.initial_scale, atol=1e-6, rtol=0))

    def sample_step(self):
        if self.initial is None:
            return
        state = _lift_state(self.session, self.operation_config)
        now = float(self.session.pipeline.og.sim.current_time)
        elapsed, point = now - self.previous_time, _array(state['position'])
        cfg = self.controller.config
        if (not math.isfinite(now) or elapsed <= 0
                or np.linalg.norm(point - self.previous) > cfg.max_translation_speed_m_s * max(elapsed, 0.) + cfg.translation_tolerance_m
                or math.degrees(_angle(state['yaw_rad'], self.previous_yaw)) > cfg.max_rotation_speed_deg_s * max(elapsed, 0.) + cfg.rotation_tolerance_deg):
            self.faults.add('native_clock_or_base_continuity_failed')
        if not self._safe(state):
            self.faults.add('native_floor_contact_identity_or_optical_anchor_failed')
        self.samples += 1
        self.stable_samples = self.stable_samples + 1 if self._safe(state) and self._held_lift(state) and not self.faults else 0
        self.previous, self.previous_prop = point, _array(state['optical_poses']['prop']['position'])
        self.previous_yaw, self.previous_time = state['yaw_rad'], now
        with (self.output / 'physical_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'sample': self.samples, 'time_s': now, 'native_state': state,
                'stable_held_lift_samples': self.stable_samples, 'faults': sorted(self.faults)}, allow_nan=False) + '\n')

    def additional_evidence(self, target):
        state = _lift_state(self.session, self.operation_config)
        if (np.linalg.norm(_array(state['position']) - self.previous) > 1e-6
                or np.linalg.norm(_array(state['optical_poses']['prop']['position']) - self.previous_prop) > 1e-6
                or _angle(state['yaw_rad'], self.previous_yaw) > 1e-6):
            self.faults.add('native_motion_outside_control_sampling')
        if not self._safe(state):
            self.faults.add('native_floor_contact_identity_or_optical_anchor_failed')
        if not self._held_lift(state):
            self.stable_samples = 0
        checks = {'qualified_original_prop_start': self.qualification['accepted'],
            'continuous_safe_native_episode': not self.faults, 'current_native_safety': self._safe(state),
            'native_correct_prop_identity_and_constraint': state['correct_original_prop_held'],
            'stable_native_held_lift': self.stable_samples >= self.parameters['stability_steps']}
        return checks, {'operation': LIFT_OPERATION, 'native_state': state, 'native_control_samples': self.samples,
            'stable_held_lift_samples': self.stable_samples, 'native_faults': sorted(self.faults),
            'initial_qualification': self.qualification, 'operation_configuration': self.operation_config,
            'native_initial_reflection_only_verified': False, 'original_qa_used_for_success': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", required=True, type=Path)
    args = parser.parse_args()
    assignment = json.loads(args.assignment.read_text())
    corpus, root = Path(assignment["corpus"]), Path(assignment["batch_root"])
    category = CategoryConverter.category
    sources = [SourceInstance(path, str(path.relative_to(corpus)), *category)
        for path in sorted((corpus / category[0] / category[1]).rglob("*.json"))]
    manifest = CategoryConverter().convert_all(sources, ConversionContext(root, seed=0, advice=assignment["advice"]))
    print(json.dumps({"source_count": len(sources), "counts": manifest["counts"],
        "failure_distribution": manifest["failure_distribution"], "sample_ids": manifest["sample_ids"]}))


if __name__ == "__main__":
    main()

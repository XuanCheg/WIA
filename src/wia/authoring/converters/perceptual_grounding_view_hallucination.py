"""Create View Hallucination identification and placement tasks without geometry changes.

Run: python scripts/run_esi_category_campaign.py --help
CategoryConverter.convert_all performs deterministic CPU conversion; native
capture, qualification and human Review remain the shared runner's concern.
"""
from __future__ import annotations

from wia.resources import ESI_ROOT

from functools import lru_cache
import importlib.util
import json
import re

from wia.authoring.conversion import ROOT, ConverterBase, map_records, original_scene, physical_task
from wia.tasks.schema import Qualification

# Diagnostic reference only: installed R1 defaults open both fingers to 0.05 m.
# Actual native joint limits and local rim/edge grasps still require measurement.
REFERENCE_GRIPPER_OPENING_M = 0.10
SUPPORT_TOLERANCE_M = Qualification(on_failure="skip").support_tolerance_m


@lru_cache(maxsize=1)
def source_builder():
    """Load the official pure object/view constructor without a simulator."""
    path = ESI_ROOT / 'src/active_explore/tasks/perceptual_grounding/view_hallucination.py'
    spec = importlib.util.spec_from_file_location("_esi_view_hallucination_converter_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def public_question(raw):
    """Use public row fields; official choices() injects private target identity."""
    question = str(raw.get("question") or "").strip()
    value = raw.get("options_json")
    options = json.loads(value) if isinstance(value, str) else value
    if not options and ":" in question:
        options = re.split(r"\s+or\s+", question.split(":", 1)[1].rstrip("?"), flags=re.IGNORECASE)
    return question, [str(option).strip() for option in options or [] if str(option).strip()]


def binding(spec):
    return {key: spec[key] for key in ("name", "category", "model") if spec.get(key)}


class CategoryConverter(ConverterBase):
    category = ("Perceptual Grounding", "View Hallucination")
    version = "view_hallucination_v3.0_task_edit"

    def implementation_hash(self):
        from wia.authoring.io import digest, sha256
        return digest({'view_converter': super().implementation_hash(),
                       'shared_grounding': sha256(ROOT / 'src/wia/authoring/converters/partial_occlusion.py')})

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category:
            return {"eligible": False, "reason": "source_category_mismatch"}
        raw, payload = source.raw, source.payload
        question, options = public_question(raw)
        if not question or len(set(options)) < 2:
            return {"eligible": False, "reason": "missing_public_identity_question_or_choices"}
        objects = map_records(payload.get("objects"))
        if not objects.get("target"):
            return {"eligible": False, "reason": "missing_original_target"}
        if not objects.get("support", {}).get("name"):
            return {"eligible": False, "reason": "missing_original_support_binding"}
        # HF rows supply scene/room outside metadata; do not require an injection
        # performed only later by the native pipeline's payload_from_hf_row.
        if any(payload.get(key) is not None and payload[key] != raw.get(key) for key in ("scene", "room")):
            return {"eligible": False, "reason": "source_scene_room_disagree"}
        source_builder().build_env_objects(payload)
        source_builder().initial_camera(payload)
        return {"eligible": True, "reason": "original_target_support_and_public_question_available"}

    def convert(self, source, context):
        if context.config.get('legacy_view_hallucination'):
            return self.convert_legacy(source, context)
        from . import partial_occlusion as revised
        result = self.convert_legacy(source, context)
        task = result['task_definition']
        bindings = task['private']['bindings']
        destination = context.config.get('placement_destination')
        if not destination:
            result.update(task_definition=None, status='needs_human_decision',
                failure_stage='destination_selection', failure_reason='placement_destination_pending')
            result['warning_codes'].append('placement_destination_pending')
            return result
        relation = destination['relation']
        if relation not in ('on_top', 'inside') or not destination.get('public_label'):
            raise ValueError('Placement requires an explicit named on_top or inside destination.')
        bindings['destination'] = binding(destination['binding'])
        if bindings['destination']['name'] == bindings['target']['name']:
            raise ValueError('The ambiguous candidate cannot be its own destination.')
        choices = public_question(source.raw)[1]
        answer = revised._normalize_answer(source.raw['answer'])
        if answer not in {revised._normalize_answer(choice) for choice in choices}:
            raise ValueError('Source answer must match a public alternative.')
        prep = 'on' if relation == 'on_top' else 'inside'
        label = destination['public_label']
        task['public'] = {
            'objective': 'Inspect the candidate whose appearance changes across viewpoints. Determine whether it is '
                + ' or '.join(choices) + f', and place that same candidate {prep} {label}. '
                + 'Report its category in world_observation as ANSWER: <category>.',
            'success_criteria': f'Report the correct category, select the original candidate and release it stably {prep} {label} '
                + 'for 15 consecutive physics steps. Identification, selection and placement are distinct obligations.',
            'environment': 'The original ambiguous object, support and source viewpoint geometry are preserved.',
            'constraints': ['Track the same candidate across robot mounted observations.',
                'Preserve original asset dimensions; no minimum lift, travel or return is required.']}
        task['private']['evaluation'] = {'type': 'category_operation', 'module': revised.MODULE,
            'operation': revised.IDENTIFY_PLACE, 'binding_keys': list(bindings),
            'parameters': {'task_revision': revised.TASK_REVISION, 'identity_choices': choices,
                'expected_identity': answer, 'placement_relation': relation,
                'destination_public_label': label, 'destination_selection_seed': context.seed,
                'release_stability_steps': 15, 'maximum_release_speed_m_s': .02,
                'maximum_release_angular_speed_rad_s': .10,
                'completion_item_ids': ['identity', 'correct_bound_selection', 'stable_placement'],
                'mechanism_contract': {'preserved_relations': ['source target identity and scale',
                    'misleading original source view', 'informative alternate views', 'support and external room geometry'],
                    'required_native_evidence': ['all eight initial robot views', 'reachable informative view',
                        'named destination fit and access', 'original bound selection and stable release'],
                    'native_operation_verified': False}}}
        result['scene_definition']['bindings'] = bindings.copy()
        result.update(operation_recipe=revised.IDENTIFY_PLACE,
            interaction_policy={'mode': 'manipulation', 'group_bindings': ['target'],
                'manipulation_bindings': ['target'], 'relocation_policy': 'preserve',
                'allow_fallback_table': False, 'source_region': source.raw['room']},
            change_summary=['Preserve original misleading and informative views, asset, size and support.',
                'Require separately scored category reporting, bound selection and stable named-destination placement.'],
            limitations=['CPU conversion does not qualify original-view ambiguity or informative native access.',
                'Native Inside/Touching do not replace independent interior geometry, fit and access qualification.'])
        return revised.revise_grounding_result(result, source, context) if context.config.get('task_edit2_grounding') else result

    def convert_legacy(self, source, context):
        raw, payload = source.raw, source.payload
        question, options = public_question(raw)
        objects = map_records(payload["objects"])
        target, support = objects["target"], objects["support"]
        builder = source_builder()
        target_spec = builder.build_env_objects(payload)[0]
        _, _, camera = builder.initial_camera(payload)
        bindings = {"target": binding(target_spec), "support": binding(support)}
        public = {
            "objective": "Use your observations to identify and retrieve the object shown in the source question; change viewpoints as needed. Source question: "
                         + question + " Candidate categories: " + "; ".join(options) + ".",
            "success_criteria": "Hold the identified object with the robot gripper and raise it at least 0.05 m above its starting height.",
            "environment": "An indoor scene containing an object whose appearance can be ambiguous from one viewpoint.",
            "constraints": ["Use robot observations and native physical exploration to resolve the identity.",
                            "Track the same object across viewpoints; do not substitute a similar background object.",
                            "Use native robot motion and grasping; do not teleport objects or change their geometry."],
        }
        evaluation = {"type": "grasp_lift", "target": "target", "minimum_lift_m": 0.05}
        qualification = {"on_failure": "skip", "target_movable_collidable": True,
                         "support": "support", "initial_visibility": "none"}
        warnings = ["physical_qualification_pending", "reach_and_grasp_unverified", "original_support_unverified", "assumption_review_required"]
        geometry = {"support_name": bindings["support"]["name"], "support_category": bindings["support"].get("category"),
                    "reference_gripper_opening_m": REFERENCE_GRIPPER_OPENING_M, "support_tolerance_m": SUPPORT_TOLERANCE_M,
                    "extent_provenance": "Saved source AABBs, not live collision geometry or native contact measurements."}
        if target.get("aabb_min") and target.get("aabb_max"):
            extents = [float(high - low) for low, high in zip(target["aabb_min"], target["aabb_max"])]
            geometry["target_aabb_extent_m"] = extents
            # A translation cannot change object height. This screens the
            # shared whole-AABB window, not actual handle reach or graspability.
            geometry["whole_aabb_height_window_translation_feasible"] = .35 + extents[2] / 2 <= 1.10
            geometry["height_window_is_grasp_certificate"] = False
            if not geometry["whole_aabb_height_window_translation_feasible"]:
                warnings.append("whole_aabb_height_window_requires_local_grasp_measurement")
            geometry["whole_body_xy_span_m"] = min(extents[:2])
            if geometry["whole_body_xy_span_m"] > REFERENCE_GRIPPER_OPENING_M:
                warnings.append("whole_body_aabb_width_above_reference_opening")
            if support.get("aabb_max"):
                gap = float(target["aabb_min"][2] - support["aabb_max"][2])
                geometry["source_bottom_minus_support_aabb_top_m"] = gap
                if abs(gap) > SUPPORT_TOLERANCE_M:
                    warnings.append("source_support_aabb_gap_requires_review")
        if support.get("category") == "table_lamp":
            warnings.append("unusual_source_support_category")
        if camera["view_index"] != 0:
            warnings.append("original_reference_selected_after_view_zero")
        assumptions = [
            {"doubt": "Original target reach, holding and lifting feasibility are unverified, including large furniture and small tools.",
             "chosen_assumption": "Keep original target dimensions and placement in a pending L1 candidate and require native qualification before execution.",
             "alternatives": ["Qualify a robot-only stance or local rim/edge grasp using actual native geometry.", "Return not_convertible_under_L1_L2 after qualified evidence rules out a faithful retrieval goal."],
             "scope": source.relative_path,
             "risk": "Whole-body AABB width is a coarse diagnostic, not an impossibility proof; reach, gripper contact and a stable hold may still fail."},
            {"doubt": "The named original support and its saved AABB do not establish native contact, a local horizontal surface or stability.",
             "chosen_assumption": "Bind the preexisting source support without moving the target, changing collisions or constructing a replacement table.",
             "alternatives": ["Measure native contacts and a local support surface before any evidence-based L2 repair.", "Publish a physical qualification failure if the original support cannot support the target faithfully."],
             "scope": source.relative_path,
             "risk": "A support AABB can contain multiple elevations or a non-flat object; its top-minus-bottom gap can misrepresent the actual support relation."},
            {"doubt": "The saved deceptive reference and mounted robot sensors may provide different identity evidence.",
             "chosen_assumption": "Preserve the official first-visible non-topdown reference-selection rule and source views; keep embodied initial visibility unqualified.",
             "alternatives": ["Qualify a robot-only viewpoint while preserving the source asset and deceptive reference.", "Request human guidance if no faithful embodied observation can preserve the ambiguity."],
             "scope": source.relative_path,
             "risk": "The robot may initially miss the object or see a disambiguating side; a source visibility flag does not certify native mounted-camera visibility."},
        ]
        return {
            "scene_definition": original_scene(source, context, bindings=bindings),
            "task_definition": physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
                qualification=qualification, capabilities=["base_translation", "base_rotation", "ee_translation", "ee_rotation", "gripper", "trunk_height"]),
            "status": "generated", "reuse_level": "L1", "qa_preserved": True,
            "change_summary": ["Reuse the original single-object constructor, native target scale policy, placement, support and reference views with an empty overlay.",
                               "Add a pending native grasp/lift goal on the exact original target with a 0.05 m threshold.",
                               "Preserve the exact public question and all public options while retaining original QA privately."],
            "warning_codes": warnings, "assumptions": assumptions,
            "limitations": ["CPU generation does not certify native loading, physical support, navigation, reach, grasp or true task success.",
                            "Source choices() uses private target identity; public task text is derived only from the row's question/options.",
                            "The original target scale field takes precedence over the builder's [1,1,1] default; applied_scale is not used by this source constructor.",
                            "Saved AABB gap/width diagnostics do not prove penetration or ungraspability, and do not trigger scaling or invented supports.",
                            "Human/source reference images, source answers and construction geometry stay private; embodied visibility requires separate native evidence."],
            "attempted_repairs": [],
            "next_step": "Capture the program-selected original layouts through the shared queue; inspect mounted visibility and privately qualify local support, reach, grasp and lift before execution.",
            "source_layout_evidence": geometry,
            "source_reference": {"selection": "original_first_visible_non_topdown_else_first_non_topdown", "view_index": camera["view_index"],
                                 "view_type": camera["view"].get("type"), "source_visibility_metadata": builder.visibility_values(payload).get(camera["view_index"]),
                                 "scale_policy": "target.scale or [1,1,1]; do not introduce applied_scale precedence from other categories."},
        }

    def sample(self, manifest, policy=None):
        """Prioritize failures and support/reference/size uncertainty without repair."""
        policy = policy or {}
        count = min(policy.get("count", 5), len(manifest["records"]))
        rows = sorted(manifest["records"], key=lambda row: row["candidate_id"])
        gap_order = sorted(rows, key=lambda row: -abs(row.get("source_layout_evidence", {}).get("source_bottom_minus_support_aabb_top_m", 0.)))
        width_order = sorted(rows, key=lambda row: -row.get("source_layout_evidence", {}).get("whole_body_xy_span_m", 0.))
        groups = [[row for row in rows if row["status"] != "generated"],
                  [row for row in rows if "unusual_source_support_category" in row.get("warning_codes", [])],
                  [row for row in width_order if "whole_body_aabb_width_above_reference_opening" in row.get("warning_codes", [])],
                  [row for row in rows if "original_reference_selected_after_view_zero" in row.get("warning_codes", [])],
                  [row for row in gap_order if "source_support_aabb_gap_requires_review" in row.get("warning_codes", [])]]
        selected = []
        for group in groups:
            candidate = next((row["candidate_id"] for row in group if row["candidate_id"] not in selected), None)
            if candidate is not None and len(selected) < count:
                selected.append(candidate)
        for candidate in super().sample(manifest, policy):
            if candidate not in selected and len(selected) < count:
                selected.append(candidate)
        return selected

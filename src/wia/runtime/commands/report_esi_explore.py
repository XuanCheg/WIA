#!/usr/bin/env python3
"""Offline milestone 8 evidence tables; reads private files, never model context."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import csv
import json
import math
from pathlib import Path

HISTORICAL_LABELS = {
    "\u53ef\u89c1\u4f46\u521d\u59cb\u9519\u4f4d\u7684\u63a7\u5236\u6761\u4ef6\u53ef\u89e3\u4e14\u53ef\u590d\u73b0": "Visible misaligned control is feasible and reproducible",
    "\u5b9e\u73b0\u6709\u6548\u5355\u684c\u906e\u6321\u573a\u666f": "Valid single-table occlusion scene",
    "\u5355\u684c\u76ee\u6807\u5728\u6240\u6709\u542f\u7528 Agent RGB \u4e2d\u521d\u59cb\u9690\u85cf": "Target initially hidden in all enabled Agent RGB",
    "\u5b58\u5728\u7269\u7406\u53ef\u8fbe\u7684\u76ee\u6807\u63ed\u793a\u89c6\u70b9": "Reachable revealing viewpoint",
    "\u63ed\u793a\u540e\u53ef\u4ee5\u63a5\u8fd1、\u6293\u4f4f\u5e76\u62ac\u5347\u6307\u5b9a\u76ee\u6807": "Approach, grasp, and lift designated target after reveal",
    "\u5b9e\u73b0\u540c\u623f\u95f4\u81f3\u5c11\u4e24\u4e2a\u6709\u6548\u5019\u9009\u533a\u57df": "At least two valid candidate regions in one room",
    "\u76ee\u6807\u533a\u57df\u7531\u4fdd\u5b58\u79cd\u5b50\u9009\u62e9\u4e14\u4fdd\u6301\u79c1\u6709": "Seed-selected private target region",
    "\u4e24\u4e2a\u573a\u666f\u4f7f\u7528\u5f53\u524d canonical Agent/Executor": "Both scenes use the canonical Agent/Executor",
    "\u53ef\u9009\u6a21\u578b\u516c\u5f00\u89c2\u5bdf\u7b14\u8bb0，\u65e0\u79c1\u6709\u4fe1\u606f\u6ce8\u5165": "Optional model-authored public notes without private injection",
    "\u4e16\u754c\u7b49\u5f85\u7b56\u7565\u53ef\u914d\u7f6e，\u7701\u7565\u65f6\u9ed8\u8ba4 true": "Configurable waiting policy, default true",
    "run/pause \u539f\u751f\u7269\u7406\u884c\u4e3a\u4e0d\u540c": "Distinct native run/pause physics",
    "fixed/parameterized \u4e0e lease \u9632\u62a4\u53ef\u7528": "Fixed/parameterized actions and lease protection",
    "\u63a2\u7d22、\u64cd\u4f5c\u548c\u6700\u7ec8\u6210\u529f\u5206\u5f00\u8bc4\u4f30": "Separate exploration, manipulation, and success evaluation",
    "\u6bcf\u4e2a\u63a2\u7d22\u573a\u666f\u81f3\u5c11\u4e00\u4e2a\u5b8c\u6574\u6a21\u578b rollout": "A complete model rollout for each exploration scene",
    "\u7ed3\u679c\u5305\u542b\u53ef\u590d\u73b0\u573a\u666f、\u6a21\u578b、\u52a8\u4f5c、\u8bb0\u5fc6\u4e0e\u4e16\u754c\u8bbe\u7f6e": "Reproducible scene/model/action/memory/world settings",
}



def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def base_return_cycles(run):
    """Adjacent inverse successful BASE macros that physically return near start.

    This is a movement proxy, not a judgment that a search was unnecessary.
    """
    actions = [s for s in rows(run/"episode/steps.jsonl") if s.get("action_result")]
    poses = {r["action_step"]: r["base"] for r in rows(run/"private_exploration/trajectory.jsonl")}
    inverse = {"BASE_FWD": "BASE_BACK", "BASE_BACK": "BASE_FWD", "BASE_LEFT": "BASE_RIGHT",
               "BASE_RIGHT": "BASE_LEFT", "BASE_TURN_LEFT": "BASE_TURN_RIGHT", "BASE_TURN_RIGHT": "BASE_TURN_LEFT"}
    cycles = []
    for index in range(1, len(actions)):
        first, second = actions[index-1:index+1]
        first_action = (first.get("parsed_command") or {}).get("action", first.get("parsed_action"))
        second_action = (second.get("parsed_command") or {}).get("action", second.get("parsed_action"))
        if inverse.get(first_action) != second_action or not all(s["action_result"].get("ok") for s in (first, second)):
            continue
        before, after = poses.get(index-1), poses.get(index+1)
        if before is None or after is None:
            continue
        displacement = math.dist(before["position"][:2], after["position"][:2])
        angle = abs(math.degrees((before["yaw_rad"]-after["yaw_rad"]+math.pi) % (2*math.pi)-math.pi))
        if displacement <= .01 and angle <= 2.5:
            cycles.append({"decision_steps": [first["step"], second["step"]],
                           "commands": [first_action, second_action], "return_distance_m": displacement,
                           "return_yaw_error_deg": angle})
    return cycles


def gather(root):
    episodes, qualifications = [], []
    rejected_startups = []
    for run in sorted(root.glob("*qualification*")):
        if (run/"qualification_summary.json").exists():
            continue
        exit_path = next((run/name for name in ("native_exit_code.txt", "exit_code.txt") if (run/name).exists()), None)
        if exit_path is None:
            continue
        log = (run/"simulator.log").read_text(errors="replace") if (run/"simulator.log").exists() else ""
        errors = [line for line in log.splitlines() if any(word in line for word in ("RuntimeError:", "AssertionError:", "error:"))]
        rejected_startups.append({"run": run.name, "native_exit_code": exit_path.read_text().strip(),
            "reason": errors[-1] if errors else "No completed native qualification summary; inspect the retained startup log.",
            "path": str(run/"simulator.log")})
    for path in sorted(root.glob("*/qualification_summary.json")):
        d = read(path)
        saved_config = read(path.parent/"scene.json", read(path.parent/"private_scene/setup.json", {}).get("configuration"))
        current_config = read(WIA_RUNTIME_ROOT/"configs/esi_explore"/(d["variant"]+".json"))
        near_config = read(WIA_RUNTIME_ROOT/"configs/esi_explore"/(d["variant"]+"_near.json"))
        pose_config = read(WIA_RUNTIME_ROOT/"configs/esi_explore/single_table_occlusion_pose_variants.json")
        condition = ('Not recorded' if saved_config is None else 'Base configuration' if saved_config == current_config
                     else 'Near configuration' if saved_config == near_config
                     else 'Pose variant' if saved_config == pose_config else 'Early prototype')
        physical_config = lambda config: {key: value for key, value in (config or {}).items()
                                          if not key.startswith('qualification_')}
        near_geometry_match = (saved_config is not None and near_config is not None
                               and physical_config(saved_config) == physical_config(near_config))
        if near_geometry_match and condition == 'Early prototype':
            condition = 'Near geometry with an earlier private qualification route'
        initial = d["initial_visibility"]
        revealed = d.get("revealing_visibility", {})
        qualifications.append({"run": path.parent.name, "variant": d["variant"], "seed": d["seed"],
            "matches_current_variant_config": saved_config == current_config if saved_config is not None else None,
            "matches_current_near_config": saved_config == near_config if near_config is not None else None,
            "matches_current_near_geometry": near_geometry_match,
            "configuration_condition": condition,
            "qualified": d["qualification_success"], "initial_zero_all_views": initial["all_views_zero_pixels"],
            "initial_pixels": {key: value["visible_pixels"] for key, value in initial["views"].items()},
            "unoccluded_recognizable": (d.get("unoccluded_setup_diagnostic") or {}).get("any_recognizably_visible"),
            "revealed_recognizable": revealed.get("any_recognizably_visible"),
            "lift_m": d.get("evaluation", {}).get("lift_m"), "hold_0_5_sim_s": d.get("stable_hold_0_5_sim_s"),
            "stability_displacement_m": d["target_stability_displacement_m"],
            "spawn_contacts": len(d.get("spawn_external_contacts", [])) if "spawn_external_contacts" in d else None,
            "approach_contacts": sum(row["phase"] == "approach" for row in d["external_contacts"]),
            "unintended_contacts": d.get("unintended_contact_count", read(path.parent/"collision_audit.json", {}).get("unintended_contacts")),
            "contact_reporting_links": len(d.get("contact_reporting_links", [])),
            "right_candidate_viewpoint_recognizable": d.get("right_candidate_viewpoint", {}).get("any_recognizably_visible"),
            "error": d.get("error"), "path": str(path)})
    for path in sorted(root.glob("*/episode/summary.json")):
        run = path.parent.parent
        summary = read(path)
        setup = read(run/"private_scene/setup.json", {})
        experiment = read(run/"experiment.json", {})
        metrics = read(run/"final_exploration_metrics.json", read(run/"private_exploration/state.json", {}))
        evaluation = read(run/"final_evaluation.json", read(path.parent/"evaluation.json", {}))
        executed = [item for item in rows(path.parent/"steps.jsonl") if item.get("action_result")]
        final_macro = executed[-1] if executed else {}
        final_result = final_macro.get("action_result", {})
        final_command = final_macro.get("parsed_command") or {}
        manipulation = rows(run/"private_evaluation/trajectory.jsonl")
        first_drop = next((item["action_step"] for item in manipulation if item.get("lift_m", 0.) < -.05), None)
        visibility = {item["snapshot_id"]: item for item in rows(run/"private_exploration/trajectory.jsonl")}
        inputs = [item for item in rows(path.parent/"model_inputs.jsonl") if item.get("event") == "step_input"]
        visible_inputs = []
        reports = []
        for item in inputs:
            sid = item.get("model_sensor_snapshot_id", item.get("observation", {}).get("snapshot_id"))
            evidence = visibility.get(sid)
            if evidence and any(view["passes_pixel_gate"] for view in evidence["visibility"].values()):
                visible_inputs.append({"decision_step": item["step"], "sensor_snapshot_id": sid,
                                       "action_step": evidence["action_step"]})
        for item in summary.get("public_memory", []):
            if any(token in item["text"].lower() for token in ("tomato", '\u756a\u8304', "target")):
                evidence = visibility.get(item["snapshot_id"])
                views = (evidence or {}).get("visibility", {})
                reports.append(dict(item, source="model_authored_note", semantic_claim_requires_image_review=True,
                    private_any_target_pixels=any(v["visible_pixels"] > 0 for v in views.values()) if views else None,
                    private_any_recognizable_target=any(v["passes_pixel_gate"] for v in views.values()) if views else None))
        (run/"model_discovery_reports.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False)+"\n")
        initial_id = min(visibility) if visibility else None
        new_view = any(item["sensor_snapshot_id"] != initial_id for item in visible_inputs)
        world = summary.get("world_timing", {})
        return_cycles = base_return_cycles(run)
        start_sim = summary.get("initialization", {}).get("world_timing", {}).get("sampled_simulation_time_s")
        elapsed_sim = (world["sampled_simulation_time_s"]-start_sim
                       if start_sim is not None and world.get("sampled_simulation_time_s") is not None else None)
        episode = {"run": run.name, "variant": setup.get("configuration", {}).get("variant", experiment.get("scene")),
            "seed": setup.get("seed", experiment.get("seed")), "model": summary["agent"].get("model"),
            "reasoning_effort": summary["agent"].get("reasoning_effort"), "action_mode": summary["action_mode"],
            "memory_mode": summary.get("memory_mode", "none"), "world_mode": summary.get("world_mode", "run"),
            "observation_condition": summary.get("observation_condition", "current"),
            "max_steps": summary["max_steps"], "max_duration_s": summary.get("max_duration_s"),
            "termination": summary["termination_reason"],
            "last_macro_action": final_command.get("action", final_macro.get("parsed_action")),
            "last_macro_reason": final_result.get("reason", final_result.get("execution_status")),
            "last_macro_outcome_known": final_result.get("outcome_known"),
            "last_macro_motion_stopped": final_result.get("motion_stopped"),
            "error": summary.get("error"), "task_success": summary["task_success"],
            "agent_decisions": summary["agent_decisions"], "model_calls": summary["model_calls"],
            "macro_executions": summary["macro_actions"], "base_actions": metrics.get("base_actions"),
            "arm_trunk_actions": metrics.get("arm_trunk_actions"), "grasp_attempts": metrics.get("grasp_attempts"),
            "blocked_motions": metrics.get("blocked_motions"), "recoverable_failures": len(summary.get("recovered_execution_errors", [])),
            "controller_env_steps_commands": summary.get("controller_env_steps"),
            "controller_env_steps_episode": metrics.get("controller_env_steps"),
            "physical_trajectory_length_m": metrics.get("physical_trajectory_length_m"),
            "region_revisits": metrics.get("region_revisits"), "lift_m": evaluation.get("lift_m"),
            "two_macro_base_return_cycles": return_cycles, "two_macro_base_return_cycle_count": len(return_cycles),
            "first_visible_step": metrics.get("first_objectively_visible_step"),
            "first_recognizable_step": metrics.get("first_recognizably_visible_step"),
            "first_approach_step": metrics.get("first_target_approach_step"),
            "first_grasp_step": metrics.get("first_grasp_attempt_step"),
            "first_attachment_step": metrics.get("first_successful_attachment_step"),
            "first_lift_step": metrics.get("first_valid_lift_step"), "model_received_new_recognizable_view": new_view,
            "first_target_drop_below_initial_5cm_step": first_drop,
            "recognizable_model_inputs": visible_inputs, "model_discovery_reports": reports,
            "model_discovery_image_review": read(run/"model_discovery_image_review.json"),
            "wall_duration_s": summary.get("timings", {}).get("total_episode_wall_time"),
            "simulated_elapsed_time_s": elapsed_sim, "world_timing": world, "path": str(path)}
        episode.update(selected_region_private=setup.get("selected_region"),
            configuration_sha256=experiment.get("scene_config_sha256"),
            evaluation_errors=rows(run/"private_exploration/errors.jsonl"),
            recording=read(run/"recording_recovery.json", read(run/"recording_verification.json")),
            recording_native_export_failed=bool(list((run/"recordings").glob(".*.pending"))),
            source_hashes_saved=(run/"source_hashes.json").exists(),
            development_pilot=run.name.startswith("A-control"))
        for key in ("model_wait_wall_time_s", "simulation_time_during_model_wait_s", "idle_controller_env_steps",
                    "action_execution_sim_time_s", "action_execution_wall_time_s"):
            episode[key] = world.get(key)
        episodes.append(episode)
    pairs = []
    for left_name, right_name, factor in (
        ("B-single-seed7-run-none", "B-single-seed7-initial-only", "observation_condition"),
        ("B-single-seed8-run-none", "B-single-seed8-initial-only", "observation_condition"),
        ("B-single-seed9-run-none", "B-single-seed9-initial-only", "observation_condition"),
        ("C-multi-seed1-run-none", "C-multi-seed1-run-notes", "memory_mode"),
        ("C-multi-seed7-run-none", "C-multi-seed7-run-notes", "memory_mode"),
        ("B-single-seed7-run-none", "D-single-seed7-pause-none", "world_mode")):
        if not all((root/name/"episode/summary.json").exists() for name in (left_name, right_name)):
            continue
        left, right = (read(root/name/"episode/summary.json") for name in (left_name, right_name))
        setups = [read(root/name/"private_scene/setup.json") for name in (left_name, right_name)]
        init_keys = ("configuration", "seed", "selected_region", "robot_pose_requested", "target_position_requested",
                     "nuisance_positions_requested")
        matched = all(setups[0].get(key) == setups[1].get(key) for key in init_keys)
        common = ("agent", "max_steps", "max_duration_s", "action_mode", "available_actions", "compilation_settings")
        same_capabilities = all(left.get(key) == right.get(key) for key in common)
        other_factors = ("world_mode", "memory_mode", "observation_condition")
        one_factor = all(left.get(key) == right.get(key) for key in other_factors if key != factor)
        performance = [read(root/name/"metadata.json", {}).get("performance", {}) for name in (left_name, right_name)]
        auxiliary_differences = sorted(key for key in set(performance[0]) | set(performance[1])
                                       if performance[0].get(key) != performance[1].get(key))
        hashes = [read(root/name/"source_hashes.json", {}) for name in (left_name, right_name)]
        changed_runtime_sources = sorted(key for key in hashes[0] if (key.startswith(("core/", "gumi/")) or
                                         key in ("src/wia/runtime/commands/run_esi_agent.py", "src/wia/runtime/commands/run_esi_robot.py"))
                                         and hashes[0][key] != hashes[1].get(key))
        actual = [next((r for r in rows(root/name/"private_exploration/trajectory.jsonl")
                        if r["action_step"] == 0), {}) for name in (left_name, right_name)]
        actual_base_distance = (math.dist(actual[0]["base"]["position"], actual[1]["base"]["position"])
                                if all(r.get("base") for r in actual) else None)
        actual_yaw_difference = (abs(math.degrees((actual[0]["base"]["yaw_rad"]-actual[1]["base"]["yaw_rad"]+
                                 math.pi) % (2*math.pi)-math.pi)) if all(r.get("base") for r in actual) else None)
        initial_targets = [next((r["target"]["position"] for r in rows(root/name/"private_evaluation/trajectory.jsonl")
                                if r["action_step"] == 0 and r.get("episode") == 1), None)
                           for name in (left_name, right_name)]
        actual_target_distance = math.dist(*initial_targets) if all(initial_targets) else None
        pairs.append({"left": left_name, "right": right_name, "factor": factor,
            "matched_initialization_requests": matched, "matched_agent_action_budget": same_capabilities,
            "matched_other_factors": one_factor, "changed_runtime_sources": changed_runtime_sources,
            "left_performance_settings": performance[0], "right_performance_settings": performance[1],
            "auxiliary_setting_differences": auxiliary_differences,
            "matched_recording_policy": performance[0].get("human_recording") == performance[1].get("human_recording"),
            "initial_actual_base_distance_m": actual_base_distance,
            "initial_actual_yaw_difference_deg": actual_yaw_difference,
            "initial_actual_target_distance_m": actual_target_distance,
            "passed": matched and same_capabilities and one_factor and not changed_runtime_sources,
            "single_factor_comparison": matched and same_capabilities and one_factor and not changed_runtime_sources and not auxiliary_differences,
            "limitation": "Initial-state/capability pairing is distinct from a strict single-factor comparison. Different recording policies can change wall-time overhead in run mode; inference latency and model output also vary."})
    return {"schema": "milestone8_evidence_v1", "root": str(root), "episodes": episodes,
            "qualifications": qualifications, "native_world_verification": read(root/"world_policy_verification.json"),
            "rejected_startups": rejected_startups,
            "model_notes_transport_verification": read(root/"model_notes_transport_verification.json"),
            "actual_rollout_contract_audit": read(root/"rollout_contract_audit.json"),
            "acceptance": read(root/"acceptance.json"),
            "environment_provenance": read(root/"environment_provenance.json"),
            "runtime_source_consistency": read(root/"runtime_source_consistency.json"),
            "media_integrity": read(root/"media_integrity.json"),
            "final_retained_service_verification": read(root/"final_service_verification.json"),
            "additional_b_pair_plan": read(root/"additional_b_pair_plan.json"),
            "paired_condition_checks": pairs,
            "no_aggregate_success_rate": "Small staged demonstrations; insufficient independent seeds for model-quality inference."}


def cell(value):
    if value is None:
        return "—"
    if type(value) is bool:
        return 'Yes' if value else 'No'
    if isinstance(value, float):
        return f"{value:.3f}"
    return HISTORICAL_LABELS.get(str(value), str(value)).replace("|", "/")


def table(headers, data):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |",
                       *("| " + " | ".join(cell(x) for x in row) + " |" for row in data)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/milestone8_results"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = gather(root)
    (output/"results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n")
    episodes = data["episodes"]
    if episodes:
        scalar_keys = [key for key, value in episodes[0].items() if not isinstance(value, (list, dict))]
        with (output/"episodes.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(episodes)
    if data["qualifications"]:
        with (output/"qualifications.csv").open("w") as stream:
            keys = [key for key, value in data["qualifications"][0].items() if not isinstance(value, (list, dict))]
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data["qualifications"])
    formal = [episode for episode in episodes if not episode["development_pilot"]]
    successful = [episode for episode in formal if episode["task_success"]]
    success_description = "；".join(f"{e['run']}：{e['agent_decisions']} decisions, lift {e['lift_m'] * 1000:.4f} mm"
                                   for e in successful) or 'No physical success among completed formal trajectories'
    report = ['# Milestone 8 Active Exploration Experiments', "",
              f'Completed {len(formal)} formal model trajectories completed; the visible-target development control is separate. Physical successes: {success_description}. Feasibility, supplied information, manipulation stages, and the full physical goal are separate. These few seeds do not estimate population success rates.', "",
              f'Raw configurations, observations, model inputs/outputs, private physical evaluation, and failures are preserved in [{root.name}]({root}). All native simulation used the retained 12-hour interactive allocation.', "",
              'Both exploration scenes use Beechwood_0_garden / dining_room_0, coffee_table_dnsjnv_0, red cherry_tomato / dqjuub, and R1 left assisted grasp. Eight 1024x1024 RGB views, native head/left-wrist depth, fine resolution, and Milestone 7.6 bounds are retained. Models receive the common physical task, robot sensors/state, and execution feedback. Target coordinates, region answers, segmentation, and qualification routes remain private.', "",
              '## Scene Qualification', "", 'Qualification selects actions using offline ground truth and executes them through actual R1 controllers in a separate fresh process. Rejected routes remain in the table. Zero pixels and small unrecognizable masks are distinct. Private recognizability requires at least 64 pixels, minimum bounding-box extent 8 pixels, and no image-border contact; it cannot establish model recognition.', "",
              table(['Qualification', "seed", 'Configuration', 'Matches base', 'Passed', 'Initial eight views zero', 'Unoccluded recognizable', 'Revealed recognizable', 'Lift m', 'Hold 0.5 s', 'Unintended contact reports', 'Error'],
                    [[q[k] for k in ("run", "seed", "configuration_condition", "matches_current_variant_config", "qualified", "initial_zero_all_views", "unoccluded_recognizable", "revealed_recognizable", "lift_m", "hold_0_5_sim_s", "unintended_contacts", "error")] for q in data["qualifications"]]), "",
              'Configuration conditions compare complete saved configurations. Base and pose variants are qualified independently; additional pairs share their variant configuration. Early prototypes do not replace formal paired configurations. Contacts are sampled per env.step and robot link. Sustained contact can produce multiple reports, so counts are not independent collision events.', "",
              '## Original Model Rollouts', "", 'Termination comes from the original summary. The visible but misaligned control is not a third exploration scene. All original formal exploration conditions use at most 60 decisions, an independent 5400-second wall timeout, and gpt-6-luna / high. A-control is a 30-decision development control without complete startup source hashes. initial_view_only updates current robot state/feedback while model RGB, depth, and corresponding geometry remain initial. Actual boundary observations are saved independently. New-view evidence matches the sensor snapshot IDs actually supplied to the model.', "",
              table(['Run', "seed", 'World', 'Notes', 'Image condition', 'Decisions/budget (calls)', 'New recognizable input', 'Task success', 'Lift mm', 'Termination', 'Wall s'],
                    [[e["run"],e["seed"],e["world_mode"],e["memory_mode"],e["observation_condition"],
                      f'{e["agent_decisions"]}/{e["max_steps"]} ({e["model_calls"]})',e["model_received_new_recognizable_view"],e["task_success"],
                      f'{e["lift_m"]*1000:.4f}' if e["lift_m"] is not None else None,e["termination"],e["wall_duration_s"]] for e in episodes]), "",
              'Final macro outcomes and episode termination are separate. Action timeout is distinct from the 5400-second episode timeout. Known, stopped protective failures remain recorded as the actual behavioral termination of these original trajectories.', "",
              table(['Run', 'Last macro', 'Raw action result', 'Outcome known', 'Motion stopped'],
                    [[e[k] for k in ("run", "last_macro_action", "last_macro_reason",
                                     "last_macro_outcome_known", "last_macro_motion_stopped")] for e in episodes]), "",
              '## Exploration and Manipulation Stages', "", table(['Run', 'First visible', 'First recognizable', 'Approach', "GRASP", 'Attachment', 'Valid lift', "BASE", "ARM/TRUNK", 'Blocked', 'Recovered failures', 'Region revisits', 'Path m'],
                    [[e[k] for k in ("run","first_visible_step","first_recognizable_step","first_approach_step","first_grasp_step","first_attachment_step","first_lift_step","base_actions","arm_trunk_actions","blocked_motions","recoverable_failures","region_revisits","physical_trajectory_length_m")] for e in episodes]), "",
              'Region revisits mean enabled robot cameras reenter a configured inspection-camera AABB. This measures repeated viewpoint entry, not semantic search completion. Model discoveries are retained separately in model_discovery_reports.json and require raw RGB review. Physical paths accumulate measured base XY at each controller env.step, including small waiting drift. SPL is not computed.', "",
              'Inspection AABBs are near-target camera-position proxies and can miss distant inspection. Zero revisits does not establish absence of repeated search. Recovery counts include stopped failures allowed to continue and do not establish subsequent physical success.', "",
              'Adjacent BASE return cycles require two successful opposite-direction macros, ending within 1 cm of the first starting XY and 2.5 degrees of its heading. Overlapping cycles count separately and decision IDs remain in JSON. Repeated movement does not establish ineffective search.', "",
              table(['Run', 'Adjacent BASE returns', 'Region-camera revisits'],
                    [[e["run"],e["two_macro_base_return_cycle_count"],e["region_revisits"]] for e in episodes]), "",
              'Target disturbance is recorded separately from private grasp_lift boundary trajectories: JSON/CSV identify the first boundary more than 5 cm below initial bottom-AABB height. This distinguishes information gained after a drop from revealing a target on its original support and does not change the historical goal.', "",
              '## World Advancement', "", 'Run is the normal condition; pause is a separate factor. Owner waiting includes model calls, allowed depth queries, invalid-output retries, and host input preparation. Individual model/tool waiting is also recorded in steps.jsonl. Waiting/action wall and simulation time are separate; simulation is not guaranteed to run at wall-clock speed.', "",
              table(['Run', 'World', 'Wait wall s', 'Wait sim s', 'Wait env.steps', 'Execution wall s', 'Execution sim s', 'Command env.steps', "episode env.step"],
                    [[e["run"],e["world_mode"],*[e["world_timing"].get(k) for k in ("model_wait_wall_time_s","simulation_time_during_model_wait_s","idle_controller_env_steps","action_execution_wall_time_s","action_execution_sim_time_s")],e["controller_env_steps_commands"],e["controller_env_steps_episode"]] for e in episodes]), "",
              '## Evidence and Interpretation Limits', "", "These small staged demonstrations do not establish population success rates, model rankings, or statistically significant note benefits. Scene feasibility, supplied information, approach/manipulation, and final physical success are separate. Historical grasp_lift_v1 requires the designated target attachment and at least 5 cm of lift; qualification's extra holding check is reported separately.", "",
              'Action-boundary snapshots are exported to CFR H.264 using saved capture times; intermediate frames repeat the preceding snapshot. This is not per-physics-frame recording. Full original JSON/CSV are in milestone8_results/results.json, episodes.csv, and qualifications.csv. Formal startup hashes, arguments, and configurations remain per run. The development-control provenance limitation is stated above. An em dash means an unreached stage or missing historical field; inspect JSON/raw logs.', ""]
    verification = data["native_world_verification"]
    if verification:
        report.extend(['## Native World-Policy Checks', "", 'Run claims deliberately omit the world-policy field to verify the legacy default true; pause passes false. Waiting includes actual cached MCP context/observation reads and rejection of human motion and robot commands containing world fields. Valid actions execute before the second waiting phase; pause resumes afterward. Lease release restores human physics. These technical checks are excluded from model-rollout counts.', "",
            table(['Action mode', 'World mode', 'Wait wall s', 'Wait sim s', 'Post-action wait sim s', 'MCP response s', 'Single boundary/release/all passed'],
            [[r["action_mode"],r["world_mode"],r["wait_wall_time_s"],r["wait_sim_time_s"],r.get("post_action_wait_sim_time_s"),
              r.get("cached_mcp_wall_time_s"),r["passed"]] for r in verification["records"]]), ""])
    if data["paired_condition_checks"]:
        report.extend(['## Paired-Condition Audit', "", table(['Left run', 'Right run', 'Factor', 'Matched requested start', 'Matched model/actions/budget', 'Matched other factors', 'Matched recording', 'Start/capability/runtime passed', 'Strict single factor'],
            [[r[k] for k in ("left","right","factor","matched_initialization_requests","matched_agent_action_budget","matched_other_factors","matched_recording_policy","passed","single_factor_comparison")]
             for r in data["paired_condition_checks"]]), "", 'The audit compares core/gumi, both entrypoints, and actual native performance settings. Early B seed 7 active records snapshots while its frozen partner and D seed 7 pause do not; these are not strict single-factor behavioral comparisons. Recording can increase wall turnaround and change run-mode waiting physics, but this effect was not measured. Additional B seed 9 runs both disable recording; each C pair shares its own policy. Identical requested starts do not guarantee bit-identical states under different inference delays. Model output also varies.', ""])
    contract = data["actual_rollout_contract_audit"]
    if contract:
        report.extend(['## Actual Input and Action-Boundary Audit', "", table(
            ['Run', 'Macro boundaries', 'Model inputs', 'RGB sources', 'Model notes', 'Depth responses', 'Frozen RGB files', 'Passed'],
            [[r.get(k) for k in ("run", "macro_boundaries_checked", "model_inputs_checked",
             "model_rgb_sources_checked", "accepted_model_notes_checked", "depth_tool_responses_checked", "frozen_rgb_files_checked", "passed")]
             for r in contract["records"]]), "", 'The audit checks actual sensor snapshot IDs, depth-array hashes, frozen geometry, original notes, and the latest-four context. Target mentions can be candidates, negations, or speculation and are not automatically discovery claims. Matching private pixel evidence remains offline.', ""])
    acceptance = data["acceptance"]
    if acceptance:
        report.extend(['## Acceptance Evidence', "", table(['Criterion', 'Requirement', 'Evidence satisfied'],
            [[r["criterion"], r["description"], r["passed"]] for r in acceptance["records"]]), "",
            f"Exact original evidence paths and criteria: [acceptance.json]({root / 'acceptance.json'})。", ""])
    if data["rejected_startups"]:
        report.extend(['## Startups Without Completed Qualification', "", 'These starts ended before complete replay or were interrupted by allocation expiry. They are not successful qualifications.', "",
            table(['Run', 'Exit code', 'Original reason'], [[r[k] for k in ("run","native_exit_code","reason")] for r in data["rejected_startups"]]), ""])
    media = []
    for episode in episodes:
        recording = episode.get("recording")
        if not recording:
            continue
        directory = Path(recording["path"])
        for video in recording["videos"]:
            media.append([episode["run"], video["view"],
                f'[{video["filename"]}]({directory/video["filename"]})', video["duration_s"],
                recording["source_snapshot_count"], bool(recording.get("original_failure_preserved"))])
    if media:
        report.extend(['## Saved Videos', "", table(['Run', 'View', 'Video', 'Duration s', 'Source snapshots', 'Offline re-export'], media), "",
            'The preceding frame persists between action boundaries. Native long-video timeout source PNGs/timing and failed exports are retained. Offline re-export preserves dimensions, frame rate, playback speed, hashes, and encoding commands.', ""])
    integrity = data["media_integrity"]
    if integrity:
        report.extend([f"Actual dimension/rate, first/last-frame decoding, and SHA256 checks: {len(integrity['records'])} videos; all passed={integrity['passed']}. Raw results: [media_integrity.json]({root / 'media_integrity.json'})。", ""])
    provenance = data["environment_provenance"]
    if provenance:
        report.extend(['## Reproduction Files', "",
            f"Repository base commit `{provenance['repository_head']}`; ESI-Bench commit `{provenance['esi_bench_head']}`. The worktree includes prior Milestone 7.6 changes and the uncommitted implementation; formal runtime provenance comes from saved startup hashes.",
            f"[Dependencies and environment]({root / 'environment_provenance.json'}), [implementation and reproduction]({WIA_RUNTIME_ROOT / 'docs/milestone8_implementation.md'})。", ""])
    consistency = data["runtime_source_consistency"]
    if consistency:
        report.extend([f"Formal startup consistency with the archived final runtime: {len(consistency['records'])} formal trajectories; all passed={consistency['passed']}. Compare core/gumi, entrypoints, and shared action/camera configurations; [runtime_source_consistency.json]({root / 'runtime_source_consistency.json'}) retains the original audit. The development control lacks complete startup hashes.", ""])
    retained = data["final_retained_service_verification"]
    if retained:
        report.extend(['## Original Retained Human Service', "",
            f"Final read-only native checks passed={retained['passed']}; verify eight RGB views, GUMI/MCP, default run, human-idle physics, live preview, and recording disabled. Results: [final_service_verification.json]({root / 'final_service_verification.json'})。", ""])
    if (root/"interpretation.md").exists():
        interpretation = (root/"interpretation.md").read_text().strip()
        if any('\u4e00' <= char <= '\u9fff' for char in interpretation):
            interpretation = (f'Historical interpretation is preserved verbatim in '
                              f'[interpretation.md]({root/"interpretation.md"}). '
                              'See the current English experiment report and research_log.md for the discussion.')
        report.extend(['## Interpretation of Original Experiments', "", interpretation, ""])
    figures = output/"figures"
    if figures.exists():
        report.extend(['## Images and Trajectories', "", *[f'![{label}](milestone8_results/figures/{name}.png)\n\n[Editable SVG](milestone8_results/figures/{name}.svg)' for name,label in (
                ("robot_observations",'Actual robot action-boundary images'),("base_trajectories",'Base paths sampled at each environment step'),
                ("scene_qualification",'Independent native qualification: initial, reveal, and grasp images'),
                ("physical_stages",'Visibility and manipulation stages'),("world_wait",'Model waiting wall and simulation time'),
                ("native_world_policy",'Native run/pause checks')) if (figures/(name+".png")).exists()], ""])
    (output.parent/"milestone8_results.md").write_text("\n".join(report), encoding="utf-8")
    print("episodes",len(episodes),"qualification attempts",len(data["qualifications"]))


if __name__ == "__main__":
    main()

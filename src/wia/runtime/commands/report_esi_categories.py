#!/usr/bin/env python3
"""Assemble an English ESI reuse assessment from private probe artifacts.

Run: .venv/bin/python src/wia/runtime/commands/report_esi_categories.py --run-root RUN
This reports observed construction/control checks, never model accuracy or an
unexecuted grasp/placement success. Existing source JSON and catalogues are read.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_root
    notes = json.loads((run / "assessment_notes.json").read_text())
    layout_path = run / "asset_layout_assessment.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else None
    measured = json.loads((run / "measured_findings.json").read_text()) if (run / "measured_findings.json").exists() else {}
    release = json.loads((run / "allocation_release.json").read_text()) if (run / "allocation_release.json").exists() else None
    contracts = json.loads((run / "contracts.json").read_text())
    scorers = {s["task"]: s for s in json.loads((run / "source_scorer_checks.json").read_text())}
    probes = []
    for path in sorted((run / "native").rglob("probe.json")):
        probe = json.loads(path.read_text())
        finished = path.with_name("finished.json")
        if finished.exists():
            probe["process_result"] = json.loads(finished.read_text())
        probe["artifact_dir"] = str(path.parent)
        probe["task"] = probe.get("task") or path.parent.parent.name
        if probe.get("status") == "starting" and probe.get("process_result", {}).get("exit_code"):
            probe["status"] = "failed_before_diagnostic_callback"
        probes.append(probe)
    complete = [p for p in probes if p["status"] == "complete"]
    categories_complete = sorted({t["big_task"] for t in contracts["tasks"] if any(p["task"] == t["module"].split(".")[-1] for p in complete)})
    ready_seconds = [p["ready_wall_s"] for p in complete if p.get("ready_wall_s") is not None]
    timed_actions = [a for p in complete for a in p.get("native_actions", []) if a.get("duration_s")]
    wall = sum(a["duration_s"] for a in timed_actions)
    coverage = [{"category": t["big_task"], "subcategory": t["small_task"],
                 "task": t["module"].split(".")[-1], "source_contract_reviewed": True,
                 "native_completed": any(p["task"] == t["module"].split(".")[-1] for p in complete),
                 "shared_scene_check": t["big_task"] == "Cognitive Mapping" and t["small_task"] != "Connectivity"}
                for t in contracts["tasks"]]
    summary = {"scope": "10 categories and 29 subcategories; representative samples, not all scenes",
               "corpus_samples": contracts["sample_count"], "model_calls": 0,
               "native_attempts": len(probes), "completed_native_probes": len(complete),
               "native_status_counts": dict(Counter(p["status"] for p in probes)),
               "categories_with_completed_native_probe": categories_complete,
               "action_attempts": sum(len(p.get("native_actions", [])) for p in complete),
               "completed_actions": sum(a.get("completed", False) for p in complete for a in p.get("native_actions", [])),
               "coverage": coverage,
               "cost": {"ready_wall_s_min_median_max": [min(ready_seconds), statistics.median(ready_seconds), max(ready_seconds)] if ready_seconds else None,
                        "summed_process_wall_s": sum(p.get("process_result", {}).get("wall_s", 0.) for p in probes),
                        "timed_bounded_actions": len(timed_actions), "bounded_action_wall_s": wall,
                        "bounded_action_sim_s": sum(a.get("sim_duration_s", 0.) for a in timed_actions),
                        "bounded_sim_to_wall_ratio": sum(a.get("sim_duration_s", 0.) for a in timed_actions)/wall if wall else None},
               "diagnostic_camera_resolution": [256, 256],
               "claims": "Construction and bounded native control evidence; no Agent success rate or general graspability claim.",
               "classification_basis": layout["classification_basis"] if layout else "Original integration assessment; implementation effort included",
               "asset_layout_assessment": layout,
               "category_assessment": layout["categories"] if layout else notes,
               "implementation_assessment": notes,
               "native_probes": probes, "measured_findings": measured,
               "allocation_release": release,
               "source_scorer_checks": list(scorers.values())}
    (run / "assessment.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# ESI category reuse assessment", "",
             "Reuse is ranked by asset and spatial-layout changes only; code and integration effort are excluded. Most original models, dimensions, materials and task layouts can be retained in principle. The representative evidence does not establish that any entire category needs major asset or room-layout redesign. Real grasp/place feasibility remains a separate qualification, not a scene-loading success claim." if layout else "The original integration assessment includes implementation effort.", "",
             "## Scope and protocol", "",
             f"Pinned ESI source: `f35b0de448ee77bc46714ae84b5190c17f87d86d`. The local corpus contains {contracts['sample_count']:,} samples across 10 categories and 29 subcategories. One representative question was selected per subcategory; identical Cognitive Mapping scene construction shares one native scene check. This is not an all-scene qualification.", "",
             "All native workers ran sequentially inside interactive allocation 1455777 on GN-A40-030, subsequently released as requested. Original source JSON, questions, asset scales and catalogue entries are retained. A private initialization helper screens R1 chassis stances near the source camera; this does not certify arm reach or a route. Diagnostic images use eight robot-mounted cameras at 256 × 256, versus the benchmark default 1024 × 1024. This is a setup/control screen with zero model calls, not a quality comparison. Ground truth drives private scripted diagnostics only.", "",
             "The original ESI action abstraction deliberately removes low-level motor failures. The pinned implementation also contains direct object repositioning and mock grasp paths, so their reported success must not be treated as real R1 execution. [Primary paper, Appendix O](https://arxiv.org/html/2605.18746v2#A15)", "",
             "## Implementation context (excluded from asset/layout cost)", "",
             "The native scene loader already resolves all 29 ESI task modules. The current Agent QA path is RigidContainmentQA-specific. SceneDefinition/SceneOnly also require a movable collidable target and declared support, which excludes object-free mapping and visual-only perception scenes. Add a shared observation/QA task and qualification contract alongside the existing physical goal, plus one public question/options/answer-format/stop adapter that dispatches original scorers privately through the registry. Per-subcategory minimal-reuse recommendations assume this shared integration is available. Keep physical target/support qualification strict for physical tasks. Exclude answers, object poses, maps, target close-ups, construction metadata and scorer state from Agent inputs. Historical observation inputs must be legitimate captured evidence, rather than source GT views.", "",
             "Initialize R1 on a valid floor with collision/visibility checks; use its existing body cameras, base, arm, gripper and trunk interfaces. Keep original QA score, physical-goal success and motor cost separate. For any physical addition, define the goal before changing the layout. A common rigid transform can preserve sizes and pairwise relations, but must also preserve fixed camera/observer references where the question depends on them.", "",
             "Upstream postprocessing relocates the R1 eye in at least Dimensional Size and Stacking & Stability. Restore a declared fixed robot head mount when integrating a consistent benchmark embodiment; these diagnostic runs retain the upstream offsets and do not establish identical camera geometry across classes. Reference PNGs below are private framing diagnostics, not additional Agent observations.", "",
             "Check actual target/anchor coverage across the eight mounted cameras. A head-only view can miss floor targets that a chassis camera sees. The private free-stance helper checks approximate chassis clearance; it neither guarantees visible evidence nor certifies a navigation route or grasp. Retain source anchors and try a valid robot view before relocating objects.", "",
             "## Category and subcategory decisions", ""]
    if layout:
        overview = ["## Asset and spatial-layout summary", "",
                    "Implementation failures in cloth, liquid and temporal episodes do not establish a need to edit their assets. Original-scene restoration is separate from changing source geometry: loading omitted original lamps, container links, boxes or phase contents is restoration. Robot initialization is the shared permitted addition. A tabletop relocation still counts as a layout change even if implemented through code.", "",
                    "| Category | Asset/layout verdict |", "| --- | --- |"]
        overview += [f"| {category} | {item['verdict']} |" for category, item in layout["categories"].items()]
        overview += ["", "The source-QA and real-interaction columns below deliberately distinguish geometry preservation from physical qualification. Visual-only/fixed-base/collision flags are physical configuration, not mesh/scale/pose edits by themselves; any resulting need to change support or placement is a layout change. Preserve source reference frames, occlusion and anchors when applying a common transform.", "",
                     "This revision reinterprets the existing 48 native attempts and reviews existing source reconstruction helpers. No simulations or model experiments were rerun; the allocation remains released.", "",
                     f"[Asset/layout assessment]({layout_path}) · [Existing original-phase reconstruction helper]({WIA_ESI_ROOT / 'src/passive_gt/unobserved_regeneration.py'})", ""]
        index = lines.index("## Scope and protocol")
        lines[index:index] = overview
    for category, note in notes.items():
        lines += [f"### {category}", ""]
        if layout:
            item = layout["categories"][category]
            lines += [item["summary"], "",
                      "| Subcategory | Original QA asset/layout | Real interaction layout | Evidence / limit |",
                      "| --- | --- | --- | --- |"]
            lines += [f"| {task} | {detail['source_qa_layout']} | {detail['real_interaction_layout']} | {detail['evidence']} |"
                      for task, detail in item["subtasks"].items()]
            lines += ["", "Implementation and historical diagnostic notes below retain the original integration analysis. Its code-inclusive effort ranking is superseded by the asset/layout verdict above.", ""]
        else:
            lines += [f"Reuse assessment: `{note['initial_class']}`. These recommendations combine source contracts and the representative evidence below; they do not certify every scene.", ""]
        lines += ["| Subcategory | Implementation context and diagnostic evidence |", "| --- | --- |"]
        lines += [f"| {task} | {reason} |" for task, reason in note["subtasks"].items()]
        lines.append("")
    lines += ["## Native evidence", "",
              f"Saved native attempts: {len(probes)}; completed probes: {len(complete)}; completed categories: {len(categories_complete)}/10. Bounded action readbacks: {summary['completed_actions']}/{summary['action_attempts']}. Empty grasps and safely blocked movements are expected diagnostic outcomes. No count here represents an end-to-end solved task.", "",
              "| Task / mode | Status | Ready / total seconds | Task objects: visual-only / fixed / total | Bounded actions | Evidence / failure |",
              "| --- | --- | --- | --- | --- | --- |"]
    for p in probes:
        path = Path(p["artifact_dir"])
        actions = p.get("native_actions", [])
        good = sum(a.get("completed", False) for a in actions)
        evidence = f"[probe]({path / 'probe.json'})"
        if (path / "initial_reference.png").exists():
            evidence += f" · [reference]({path / 'initial_reference.png'}) · [R1 head]({path / 'initial_head.png'})"
        if p.get("error"):
            evidence += " · " + p["error"].replace("|", "/")
        if p.get("failure_origin"):
            evidence += " · Diagnostic tooling failure; excluded from scene rejection."
        ready = f"{p['ready_wall_s']:.1f}" if p.get("ready_wall_s") is not None else "unavailable"
        total = p.get("elapsed_wall_s") or p.get("process_result", {}).get("wall_s")
        total = f"{total:.1f}" if total is not None else "unavailable"
        counts = " / ".join(str(p.get(k, "unavailable")) for k in ("visual_only_task_count", "fixed_task_count", "task_object_count"))
        lines.append(f"| {p['task']} / {path.name} | {p['status']} | {ready} / {total} | {counts} | {good}/{len(actions)} | {evidence} |")
    lines += ["", "The first private probes exposed callback import and velocity-interface assumptions; their failed attempts are retained and marked as tooling failures. Corrected reruns take precedence. Shared-construction Cognitive Mapping entries are code/scorer checks, not separate native trials.", "",
              "## Runtime and scope of success", ""]
    if ready_seconds:
        times = summary["cost"]["ready_wall_s_min_median_max"]
        lines += [f"Completed-probe readiness takes {times[0]:.1f} / {times[1]:.1f} / {times[2]:.1f} seconds (minimum / median / maximum), including native application, scene and controller initialization. Saved subprocess time totals {summary['cost']['summed_process_wall_s']/60:.1f} minutes. Diagnostic workers run sequentially; these timings do not include queue time or estimate model throughput.", "",
                  f"Timed bounded controls cover {summary['cost']['bounded_action_sim_s']:.2f} simulated seconds in {wall:.1f} wall seconds (simulation/wall ratio {summary['cost']['bounded_sim_to_wall_ratio']:.3f}). This uses 256 × 256 diagnostic views, not default 1024 × 1024 model evaluation. Grasp macros and scripted ESI effects are recorded separately and excluded from this bounded-control ratio.", ""]
    for p in probes:
        grasp = p.get("grasp_probe")
        if grasp:
            lines += [f"Action Order `{Path(p['artifact_dir']).name}`: native grasp acquired `{grasp.get('grasp_acquired')}`, lift verified `{grasp.get('lift_verified')}`, released `{grasp.get('release_verified', False)}`, object pose commands during manipulation `{grasp.get('object_pose_commands')}`. " + (f"Failure: {grasp['error']}. " if grasp.get("error") else "") + f"[Detailed grasp evidence]({Path(p['artifact_dir']) / 'grasp_probe.json'}).", ""]
    lines += ["A completed probe means the diagnostic callback finished. It can include a guarded action stop, an empty gripper close or a scene lacking the intended task mechanism. It is not a benchmark-qualified or solved episode. All grasp tests use the benchmark's native assisted-grasp semantics; a lifted object must follow the real controller/constraint path, with no object pose commands during manipulation. No complete Action Order stack, actual cloth uncovering, real fluid pour or full-room route is claimed unless explicitly measured.", "",
              "## Measured physical boundaries", "",
              "The unchanged Physical Contact representative has source QA answer yes and no initial native pair contact. Source generation stores per-view mask-adjacency flags. Legacy QA can remain intact, but a true-contact goal needs contact-qualified examples or a declared physical label; this sample does not pass that qualification.", "",
              "The larger-room slope uses verified native friction 1.3215 / 0.6701. Its scripted 30-frame trajectory spans 1.0 s at the actual 30 Hz render/step clock, shifts 14.27 mm in XY between frames 1 and 30 and stops by the final frame. The first sampled frame is the displacement reference; the initial scripted drop transient before that frame is excluded. This evidence verifies native dynamics, not a real R1 placement or the generator's legacy clock.", "",
              f"[Slope trace plot]({run / 'slope_trace.png'}) · [Raw trace CSV]({run / 'slope_trace.csv'}) · [Measured findings]({run / 'measured_findings.json'})", "",
              "The liquid standing probe returns success=True from the upstream fill-left operation but n_left=n_right=0, after a 684.2 s total callback run. Its first RELEASE is also guarded for working-arm drift. The original capacity measurements are not reproduced by this screen; renderer startup and an operation flag are insufficient for task qualification.", "",
              "## Original scoring checks", "",
              "Pure original source function bodies check the selected answer packets without simulation or models. These check packet compatibility, not spatial correctness after an adaptation or model quality. Rigid Containment stores a keyed list of object assignments in its HF answer; the single-question scorer requires a normalized container answer. Action Order directional answers are tied to the declared source camera, even if the upstream active runner selects another view. Counting retains soft score max(0, 1 − absolute error / 3), separately from exact answer correctness.", "",
              f"[All source scorer checks]({run / 'source_scorer_checks.json'}) · [Full contracts and counts]({run / 'contracts.json'}) · [Machine-readable assessment]({run / 'assessment.json'})", "",
              "## Recommended order", "",
              "1. Finish the already requested Action Order real pickup/stacking and independent final-support evaluator, keeping original sizes and directional reference. Reuse the qualified tabletop initialization; choose a reachable stance or a small common layout shift only if actual contact/IK evidence requires it.",
              "2. Preserve original layouts for Metric Comparison, Spatial Relations, perceptual/counting tasks, mapping and reflection. Restore omitted original lighting/enclosure states and temporal props/phase contents through existing assets and source reconstruction helpers. Code complexity does not raise their asset/layout cost.",
              "3. For real stacking, containment and reveal, try original-size assets and native placements first. If reach requires it, use a small common support relocation that preserves the question; for slopes/navigation, prefer existing spacious source examples over rebuilding rooms.",
              "4. Qualify collision-enabled Merged Observation clusters, true Physical Contact examples, floating mirror props and R1 passage/actor clearance. These can expose genuine placement/support constraints; do not infer them solely from visual overlap or controller errors.",
              "5. Repair cloth/fluid implementation faults before deciding whether their geometry needs changes. Preserve cloth coverage and cup capacity relationships; any future workspace/scaffold patch must be justified by physical measurements. Current grasp/uncover/pour results remain unqualified.", "",
              f"[Allocation release evidence]({run / 'allocation_release.json'}): user-authorized qdel, verified PBS state `{release.get('final_pbs_state')}`." if release else "Allocation release evidence is saved separately as allocation_release.json after final inspection and user-authorized qdel.", ""]
    (run / "report.md").write_text("\n".join(lines))
    print(json.dumps({k: summary[k] for k in ("native_attempts", "completed_native_probes", "native_status_counts", "categories_with_completed_native_probe")}))


if __name__ == "__main__":
    main()

"""Screen one M3 destination sample with an explicit supplemental CPU budget.

Run in the behavior environment::

    python scripts/refine_m3_destination_audit.py INVENTORY OUTPUT \
        --region unit_a --sample-index 1 --max-ik-attempts 576 --max-stances 24

The CLI sets the installed omni.usd.libs bin/bin-deps library paths as the
ordinary audit driver does. It never launches a simulator or changes the input.
Its result is separate from the uniform audit and never certifies a grasp.
Optional ``--extra-side-heights F...`` appends explicit side grasp-height fractions.
Optional ``--payload-yaw-deg {0,90,-90,180}`` rotates only the hypothetical item_01 proxy.
Optional ``--extra-stance-xy X Y`` adds one normally checked base proposal;
``--extra-stance-only`` restricts proposal sampling to that point.
Optional ``--ik-seed-report PATH`` supplies only optimized numerical IK initial guesses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def _rotate_payload_proxy(source, yaw_deg, engine):
    """Conservatively rotate a measured world-AABB proxy about its own center."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    box = engine._box(source["aabb"])
    center = box.mean(axis=0)
    yaw = Rotation.from_euler("z", yaw_deg, degrees=True)
    matrix = yaw.as_matrix()
    corners = engine._corners(box)
    rotated_corners = (corners - center) @ matrix.T + center
    rotated_box = np.stack((rotated_corners.min(axis=0), rotated_corners.max(axis=0)))
    rotated_box[:, 2] = box[:, 2]  # A world-Z rotation preserves measured vertical bounds exactly.
    position = np.asarray(source["position"], float)
    orientation = np.asarray(source["orientation"], float)
    if position.shape != (3,) or orientation.shape != (4,) or not np.isfinite(np.r_[position, orientation]).all():
        raise engine.AuditUnavailable("A finite measured item_01 position and xyzw orientation are required for the hypothetical yaw.")
    proxy = deepcopy(source)
    proxy["aabb"] = rotated_box.tolist()
    proxy["position"] = (matrix @ (position - center) + center).tolist()
    proxy["position"][2] = source["position"][2]
    proxy["orientation"] = (yaw * Rotation.from_quat(orientation)).as_quat().tolist()
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = matrix, center - matrix @ center
    return proxy, {"world_z_yaw_deg": yaw_deg, "pivot_world_m": center.tolist(),
                   "yaw_quaternion_xyzw": yaw.as_quat().tolist(),
                   "world_transform_4x4": transform.tolist(),
                   "source_aabb_corners_world_m": corners.tolist(),
                   "rotated_aabb_corners_world_m": rotated_corners.tolist()}


def _load_ik_seed(path, evidence, engine):
    """Extract only torso/working-arm pregrasp values from a full geometric witness."""
    raw = Path(path).read_bytes()
    evidence["source_sha256"] = hashlib.sha256(raw).hexdigest()
    source = json.loads(raw)
    result = source.get("result") or {}
    evidence.update(source_schema=source.get("schema"), source_status=source.get("status"),
                    source_result_status=result.get("status"))
    witness = result.get("witness")
    if (source.get("schema") != "wia_m3_supplemental_destination_audit_v1"
            or source.get("status") != "found_witness" or result.get("status") != "found_witness"
            or not isinstance(witness, dict)):
        raise engine.AuditUnavailable("The IK seed source must be a found_witness supplemental result with result.witness; kinematic-only evidence cannot supply a seed.")
    arm = witness.get("arm")
    if arm not in {"left", "right"}:
        raise engine.AuditUnavailable("The source geometric witness must explicitly identify its working arm.")
    names = witness.get("optimized_joint_names")
    approved = {f"torso_joint{index}" for index in range(1, 5)} | {f"{arm}_arm_joint{index}" for index in range(1, 7)}
    if not isinstance(names, list) or len(names) != 10 or not all(isinstance(name, str) for name in names) or set(names) != approved:
        raise engine.AuditUnavailable("Source optimized_joint_names must contain exactly the four torso and six source working-arm joints, without duplicates or other joints.")
    pregrasps = [stage for stage in witness.get("stages", []) if stage.get("name") == "pregrasp"]
    if len(pregrasps) != 1:
        raise engine.AuditUnavailable("Exactly one source witness pregrasp joint stage is required.")
    values = {name: float(pregrasps[0]["joints"][name]) for name in names}
    if not all(math.isfinite(value) for value in values.values()):
        raise engine.AuditUnavailable("Source optimized pregrasp seed values must all be finite.")
    evidence.update(matched_arm=arm, optimized_joint_names=list(names), seed_joint_values=values,
                    validation_status="source_validated_current_limits_not_yet_checked")
    return values


def audit_one_sample(inventory_path, output_dir, *, region, sample_index, max_ik_attempts=576, max_stances=24,
                     extra_side_heights=(), payload_yaw_deg=0, extra_stance_xy=None, extra_stance_only=False,
                     ik_seed_report=None):
    """Write a single independent sample result using the unchanged audit engine."""
    import numpy as np
    import wia.candidates.reachability as engine

    if region not in {"unit_a", "unit_b"} or sample_index < 0 or max_ik_attempts < 1 or max_stances < 1:
        raise ValueError("A unit_a/unit_b region, nonnegative sample index and positive budgets are required.")
    if payload_yaw_deg not in {0, 90, -90, 180}:
        raise ValueError("The supplemental payload yaw must be 0, 90, -90 or 180 degrees.")
    if extra_stance_only and extra_stance_xy is None:
        raise ValueError("Extra-stance-only sampling requires an explicit extra stance XY.")
    if extra_stance_xy is not None:
        extra_stance_xy = np.asarray(extra_stance_xy, float)
        if extra_stance_xy.shape != (2,) or not np.isfinite(extra_stance_xy).all():
            raise ValueError("The extra stance requires exactly two finite world-XY coordinates.")
    extra_side_heights = [float(fraction) for fraction in extra_side_heights]
    if any(not math.isfinite(fraction) or not 0 < fraction < 1 for fraction in extra_side_heights):
        raise ValueError("Every extra side height must be finite and strictly between zero and one.")
    output_dir = Path(output_dir)
    report_path = output_dir / "supplemental_destination_audit.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite a supplemental report: {report_path}")
    started = time.monotonic()
    engine_path = Path(engine.__file__).resolve()
    options = engine.AuditOptions(navigation_mode="se2_rectangle", torso_search="continuous",
        max_ik_attempts_per_target=max_ik_attempts, max_stances=max_stances)
    original_height_function = engine._approach_height_candidates
    original_candidates = list(original_height_function())
    effective_candidates = [*original_candidates,
                            *[(orientation, fraction) for fraction in extra_side_heights for orientation in (2, 3)]]
    report = {
        "schema": "wia_m3_supplemental_destination_audit_v1", "status": "unavailable",
        "input_file": str(Path(inventory_path).resolve()), "raw_input_sha256": None,
        "engine_file": str(engine_path), "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "options": asdict(options), "supplemental_overrides": {
            "max_ik_attempts_per_target": max_ik_attempts, "max_stances": max_stances,
            "extra_side_heights": extra_side_heights, "payload_yaw_deg": payload_yaw_deg,
            "extra_stance_xy": extra_stance_xy.tolist() if extra_stance_xy is not None else None,
            "extra_stance_only": extra_stance_only,
            "ik_seed_report": str(Path(ik_seed_report).resolve()) if ik_seed_report is not None else None},
        "ik_seed": {"enabled": ik_seed_report is not None,
            "source_file": str(Path(ik_seed_report).resolve()) if ik_seed_report is not None else None,
            "source_sha256": None, "validation_status": "not_requested" if ik_seed_report is None else "pending",
            "total_solve_calls": 0, "matched_arm_solve_calls": 0, "passthrough_solve_calls": 0,
            "solve_calls_by_arm": {}, "temporary_solve_wrapper_applied": False,
            "scope": "Numerical optimization initial guess only, from source result.witness pregrasp optimized torso/working-arm joints. No source base/pose/path certificate is transferred; current geometry, joint limits and all pose/path checks still apply. Inventory, robot.seed, inactive arm and gripper values remain unchanged."},
        "extra_stance": {
            "enabled": extra_stance_xy is not None, "temporary_instance_wrapper_applied": False,
            "requested_world_xy_m": extra_stance_xy.tolist() if extra_stance_xy is not None else None,
            "extra_stance_only": extra_stance_only, "provenance_kind": "supplemental_native_service",
            "facing_policy": "atan2(target_center_y - stance_y, target_center_x - stance_x)",
            "ranking_policy": "target_distance_m + 0.05 * start_distance_m", "raw_count": 0,
            "selected": None, "proposal_call_records": [],
            "filtered_scope": "Only original engine proposals are sampled." if extra_stance_xy is None else
                              "Only the explicit extra proposal is sampled." if extra_stance_only else
                              "The explicit extra proposal is appended and ranked with original proposals.",
            "historical_by_source_count_limitation": "Unmodified engine by-source dictionaries list only edge, initial_root and radial, omitting supplemental_native_service. Total counts and exact selected_stances provenance remain authoritative; custom accounting is recorded here."},
        "approach_height_candidates_original": original_candidates,
        "approach_height_candidates_effective": effective_candidates,
        "temporary_approach_candidate_override_applied": False,
        "region_name": region, "sample_index": sample_index, "screen_performed": False,
        "result": None, "navigation": None, "robot_commands_executed": False,
        "policy_actions_executed": False, "model_calls_executed": False, "policy_permission_inferred": False,
        "scope": "Independent supplemental search of one destination sample and a permissible hypothetical destination orientation; separate from the uniform audit. No carried rotation, pickup, carrying, door opening, placement or grasp-success certificate.",
        "limitations": [
            "Supplemental changes are the explicit per-sample IK budget, maximum stance count, any listed extra side grasp-height candidates, item_01 destination yaw, extra-stance sampling/filtering and optional numerical IK initial guesses. All resident/unit/environment obstacles, inventory/robot.seed, tool orientation definitions, approach lengths and collision/path/IK rules remain unchanged.",
            "An IK seed report transfers only ten optimized numerical initial-guess values for its explicit source arm. It transfers no geometric certificate and changes neither the inactive arm nor gripper values.",
            "Extra stance options change only proposal sampling. Original distance, pose collision, connectivity, sweep, ranking, truncation and route checks remain active; no reachable states or stance mappings are injected.",
            "A nonzero payload yaw rotates the eight corners of the measured world AABB and conservatively bounds them. This is a hypothetical rotated proxy, never a new native geometry measurement or an executed carried rotation.",
            "The fitting payload proxy and retained scene objects come from the unmodified destination preparation helper. Open-gripper witnesses do not validate closure, carrying or placement.",
            "A not-found result is unresolved under this bounded search and conservative proxies; it does not establish physical impossibility.",
            "Base transit checks the measured footprint only; full-body motion during base travel remains unverified."],
    }
    raw = None
    try:
        raw = Path(inventory_path).read_bytes()
        report["raw_input_sha256"] = hashlib.sha256(raw).hexdigest()
        inventory = json.loads(raw)
        report["inventory_identity"] = inventory.get("id")
        seed_values = _load_ik_seed(ik_seed_report, report["ik_seed"], engine) if ik_seed_report is not None else None
        rows, targets = inventory.get("objects", []), engine._initial_targets(inventory)
        for row in [*rows, *targets]:
            engine._box(row["aabb"])
        floor_z = float(inventory["floor_z"])
        if not np.isfinite(floor_z):
            raise engine.AuditUnavailable("A finite measured supporting floor height is required.")
        objects = [row for row in rows if not (str(row.get("role", "")).lower() in {"floor", "ground"}
                   and engine._box(row["aabb"])[1, 2] <= floor_z + .03)]
        known = {row["name"] for row in objects}
        objects += [row for row in targets if row["name"] not in known]
        report["floor_z"] = floor_z
        report["floor_measurement"] = deepcopy(inventory.get("floor_measurement"))
        payloads = targets
        source = next((row for row in targets if row["name"] == "item_01" and engine._task_movable(row)
                       and row.get("rigid_grasp_proxy_eligible") is not False), None)
        report["payload_yaw"] = {"requested_world_z_yaw_deg": payload_yaw_deg, "applied": False,
            "original_source_payload": deepcopy(source), "rotated_payload_proxy": deepcopy(source),
            "transform": {"world_transform_4x4": np.eye(4).tolist()},
            "geometry_scope": "Hypothetical destination payload only. The original objects list and native inventory remain unchanged; no new measurement or carried rotation is implied."}
        if payload_yaw_deg:
            if source is None:
                raise engine.AuditUnavailable("No eligible movable item_01 payload candidate exists for the requested hypothetical yaw.")
            proxy, transform = _rotate_payload_proxy(source, payload_yaw_deg, engine)
            payloads = [proxy if row is source else row for row in targets]
            report["payload_yaw"].update(applied=True, rotated_payload_proxy=deepcopy(proxy), transform=transform)
        entries, pending = engine._prepare_destination_queries({region: inventory["regions"][region]}, objects, payloads)
        entry = entries[0]
        report["original_prepared_destination"] = deepcopy(entry)
        report["original_prepared_destination"]["hypothetical_source_metadata_scope"] = (
            "Engine preparation before supplemental IK. With nonzero payload yaw, hypothetical_payload source bounds/pose describe the hypothetical rotated proxy, not the original native measurement; see payload_yaw.original_source_payload.")
        report["payload_yaw"]["applied_to_selected_source"] = bool(payload_yaw_deg and
            (entry.get("hypothetical_payload") or {}).get("source_name") == "item_01")
        plans = next((plans for prepared, plans in pending if prepared["region_name"] == region), [])
        report["original_plans"] = [{"sample_index": sample["sample_index"],
            "sample_preparation_status": sample["status"], "virtual_target": deepcopy(virtual),
            "query_object_names": [obj["name"] for obj in query_objects]}
            for sample, virtual, query_objects in plans]
        sample = next((sample for sample in entry.get("sample_records", [])
                       if sample["sample_index"] == sample_index), None)
        if sample is None:
            raise engine.AuditUnavailable(entry.get("reason") or "The requested sample index is absent from the engine's prepared grid.")
        report["selected_sample"] = deepcopy(sample)
        plan = next((plan for plan in plans if plan[0]["sample_index"] == sample_index), None)
        if plan is None:
            report["result"] = deepcopy(sample.get("result")) or {
                "status": sample["status"], "reason": entry.get("reason"), "witness": None,
                "kinematic_witness": None, "ik_attempts": 0}
        else:
            _, virtual, query_objects = plan
            report["selected_virtual_target"] = deepcopy(virtual)
            report["selected_query_objects"] = deepcopy(query_objects)
            robot = engine._Robot(inventory["robot"], options)
            if seed_values is not None:
                seed_limits = {}
                for name, value in seed_values.items():
                    lower, upper = map(float, robot.limits[name])
                    if not all(math.isfinite(number) for number in (lower, upper)) or not lower < upper or not lower <= value <= upper:
                        raise engine.AuditUnavailable(f"Source IK initial guess for {name} is outside the finite current native joint limits.")
                    seed_limits[name] = [lower, upper]
                report["ik_seed"].update(validation_status="validated_against_current_native_limits",
                                         current_native_limits=seed_limits)
            report["calibration"] = {"source": "inventory_and_supplied_robot_usd", "urdf_path": str(robot.urdf),
                "joint_count": len(robot.calibration), "collision_spheres_path": str(robot.sphere_path),
                "base_footprint_circumscribed_radius_m": robot.base_radius,
                "initial_joint_limit_margin_native_units": engine.joint_limit_margin(robot.seed, robot.limits)}
            navigation = engine._SE2Navigation(robot, query_objects, [virtual], options)
            report["screen_performed"] = True
            original_stance_method = navigation._stance_proposals if extra_stance_xy is not None else None
            stance_had_instance_override = ("_stance_proposals" in vars(navigation)) if extra_stance_xy is not None else False
            stance_instance_value = vars(navigation).get("_stance_proposals") if stance_had_instance_override else None
            original_solve = robot.solve if seed_values is not None else None
            solve_had_instance_override = ("solve" in vars(robot)) if seed_values is not None else False
            solve_instance_value = vars(robot).get("solve") if solve_had_instance_override else None

            def seeded_solve(arm, joints, position, orientation, attempt_index):
                evidence = report["ik_seed"]
                evidence["total_solve_calls"] += 1
                evidence["solve_calls_by_arm"][arm] = evidence["solve_calls_by_arm"].get(arm, 0) + 1
                if arm == evidence["matched_arm"]:
                    evidence["matched_arm_solve_calls"] += 1
                    initial_guess = dict(joints)
                    initial_guess.update(seed_values)
                    return original_solve(arm, initial_guess, position, orientation, attempt_index)
                evidence["passthrough_solve_calls"] += 1
                return original_solve(arm, joints, position, orientation, attempt_index)

            def supplemental_proposals(target, scene_objects, stance_options):
                proposals, raw_counts, offset = original_stance_method(target, scene_objects, stance_options)
                original_count, original_raw_counts = len(proposals), dict(raw_counts)
                center = engine._box(target["aabb"]).mean(axis=0)[:2]
                point = extra_stance_xy.copy()
                yaw = float(np.arctan2(center[1] - point[1], center[0] - point[0]))
                kind = "supplemental_native_service"
                proposal = {"point": point, "yaw": yaw, "provenance": [{"kind": kind,
                            "source": "explicit_supplemental_world_xy"}]}
                effective = [proposal] if extra_stance_only else [*proposals, proposal]
                effective_counts = Counter() if extra_stance_only else Counter(raw_counts)
                effective_counts[kind] += 1
                rank = lambda row: np.linalg.norm(row["point"] - center) + .05 * np.linalg.norm(row["point"] - navigation.start)
                effective.sort(key=rank)
                record = {"world_xy_m": point.tolist(), "face_yaw_rad": yaw,
                    "raw_count": 1, "original_proposal_count": original_count,
                    "original_raw_proposal_counts": original_raw_counts,
                    "original_raw_proposal_count": sum(original_raw_counts.values()),
                    "returned_proposal_count": len(effective), "returned_raw_proposal_counts": dict(effective_counts),
                    "ranking_score": float(rank(proposal)),
                    "ranked_proposal_index": next(index for index, row in enumerate(effective) if row is proposal),
                    "extra_stance_only": extra_stance_only}
                report["extra_stance"].update(record)
                report["extra_stance"]["proposal_call_records"].append(deepcopy(record))
                return effective, effective_counts, offset

            try:
                if extra_side_heights:
                    engine._approach_height_candidates = lambda: list(effective_candidates)
                    report["temporary_approach_candidate_override_applied"] = True
                if extra_stance_xy is not None:
                    navigation._stance_proposals = supplemental_proposals
                    report["extra_stance"]["temporary_instance_wrapper_applied"] = True
                if seed_values is not None:
                    robot.solve = seeded_solve
                    report["ik_seed"]["temporary_solve_wrapper_applied"] = True
                report["result"] = engine._screen_target(virtual, query_objects, robot, navigation, options, floor_z)
            finally:
                try:
                    if seed_values is not None:
                        if solve_had_instance_override:
                            robot.solve = solve_instance_value
                        else:
                            del robot.solve
                        report["ik_seed"]["temporary_solve_wrapper_restored"] = (
                            vars(robot).get("solve") is solve_instance_value if solve_had_instance_override else
                            "solve" not in vars(robot))
                finally:
                    try:
                        if extra_stance_xy is not None:
                            if stance_had_instance_override:
                                navigation._stance_proposals = stance_instance_value
                            else:
                                del navigation._stance_proposals
                            report["extra_stance"]["temporary_instance_wrapper_restored"] = (
                                vars(navigation).get("_stance_proposals") is stance_instance_value if stance_had_instance_override
                                else "_stance_proposals" not in vars(navigation))
                            sampling = getattr(navigation, "last_stance_sampling", None)
                            if sampling is not None and report["extra_stance"]["proposal_call_records"]:
                                selected = [row for row in sampling.get("selected_stances", [])
                                    if any(source.get("kind") == "supplemental_native_service"
                                           for source in row.get("provenance", []))]
                                report["extra_stance"].update(selected=bool(selected), selected_stance_records=deepcopy(selected))
                    finally:
                        if extra_side_heights:
                            engine._approach_height_candidates = original_height_function
                report["temporary_approach_candidate_override_restored"] = (
                    engine._approach_height_candidates is original_height_function)
            report["navigation"] = engine._navigation_evidence(navigation, options)
        report["status"] = report["result"]["status"]
    except (engine.AuditUnavailable, ImportError, OSError, KeyError, ValueError) as error:
        report.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
        report["result"] = {"status": "unavailable", "reason": report["reason"],
                            "witness": None, "kinematic_witness": None}
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    report["engine_sha256_at_end"] = hashlib.sha256(engine_path.read_bytes()).hexdigest()
    report["engine_file_unchanged_during_search"] = report["engine_sha256_at_end"] == report["engine_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        snapshot = output_dir / "raw_inventory.json"
        if snapshot.exists() and snapshot.read_bytes() != raw:
            raise FileExistsError(f"Refusing to overwrite a different raw input snapshot: {snapshot}")
        if not snapshot.exists():
            with snapshot.open("xb") as handle:
                handle.write(raw)
        report["raw_input_snapshot"] = str(snapshot.resolve())
    with report_path.open("x") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("A positive integer is required.")
    return result


def _interior_fraction(value):
    result = float(value)
    if not math.isfinite(result) or not 0 < result < 1:
        raise argparse.ArgumentTypeError("A finite fraction strictly between zero and one is required.")
    return result


def _finite_coordinate(value):
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("A finite world coordinate is required.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--region", required=True, choices=("unit_a", "unit_b"))
    parser.add_argument("--sample-index", required=True, type=int)
    parser.add_argument("--max-ik-attempts", type=_positive, default=576)
    parser.add_argument("--max-stances", type=_positive, default=24)
    parser.add_argument("--extra-side-heights", nargs="+", type=_interior_fraction, default=[])
    parser.add_argument("--payload-yaw-deg", type=int, choices=(0, 90, -90, 180), default=0)
    parser.add_argument("--extra-stance-xy", nargs=2, type=_finite_coordinate, default=None)
    parser.add_argument("--extra-stance-only", action="store_true")
    parser.add_argument("--ik-seed-report", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.extra_stance_only and args.extra_stance_xy is None:
        parser.error("--extra-stance-only requires --extra-stance-xy")
    report = audit_one_sample(args.inventory, args.output, region=args.region, sample_index=args.sample_index,
                             max_ik_attempts=args.max_ik_attempts, max_stances=args.max_stances,
                             extra_side_heights=args.extra_side_heights, payload_yaw_deg=args.payload_yaw_deg,
                             extra_stance_xy=args.extra_stance_xy, extra_stance_only=args.extra_stance_only,
                             ik_seed_report=args.ik_seed_report)
    print(json.dumps({"status": report["status"], "region": args.region, "sample_index": args.sample_index,
        "ik_attempts": (report.get("result") or {}).get("ik_attempts"), "elapsed_s": report["elapsed_s"],
        "output": str((args.output / "supplemental_destination_audit.json").resolve())}))


if __name__ == "__main__":
    if os.environ.get("WIA_M3_SUPPLEMENTAL_LD_READY") != "1":
        usd = next((Path(sys.prefix) / "lib/python3.10/site-packages/isaacsim/extscache").glob("omni.usd.libs-*"))
        environment = dict(os.environ, LD_LIBRARY_PATH=str(usd / "bin") + ":" + str(usd / "bin/deps"),
            CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
            NUMEXPR_NUM_THREADS="2", WIA_M3_SUPPLEMENTAL_LD_READY="1")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], environment)
    main()

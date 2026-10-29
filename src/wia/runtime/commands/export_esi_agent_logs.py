#!/usr/bin/env python3
"""Export public JSONL traces from archived episodes, without rerunning models/physics."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
import sys
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from core.agent.trace_log import EpisodeTrace, INPUT_LOG, OUTPUT_LOG, image_placeholders, timestamp


DYNAMIC_PREFIXES = ("Attached images,", "Task:", "Available actions:", "Fixed command increments",
                    "Actual primitive increments", "Remaining decision budget:", "Recent action history:",
                    "Current robot measurements:", "Last execution feedback:", "Sensor availability",
                    "Previous output was rejected")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_observation(episode, reference):
    observation = read_json(episode / reference)
    # The old metadata has archive-relative paths; make references useful from
    # the separate export directory without opening or copying any image/NPY.
    for section in ("images", "sensors"):
        for item in observation.get(section, {}).values():
            if item is not None and item.get("path"):
                item["path"] = str((episode / reference).parent / item["path"])
    return observation


def recorded_prompt(episode, attempt):
    structured = attempt.get("model_metadata", {}).get("structured_output", {})
    if isinstance(structured.get("prompt"), str):
        return structured["prompt"], structured.get("schema"), "recorded_provider_prompt"
    path = attempt.get("model_input")
    if path and (episode / path).is_file():
        return (episode / path).read_text(encoding="utf-8"), None, "recorded_canonical_prompt"
    return None, None, "not_recorded"


def export_episode(episode, output):
    episode, output = Path(episode).resolve(), Path(output).resolve()
    summary = read_json(episode / "summary.json")
    records = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines() if line.strip()]
    trace = EpisodeTrace(output, reconstructed=True)
    first_attempt = next((attempt for record in records for attempt in record.get("attempts", [])), {})
    recorded_public_calls = sum("public_messages" in attempt.get("model_metadata", {})
                                for record in records for attempt in record.get("attempts", []))
    public_messages_status = "recorded" if recorded_public_calls == summary["model_calls"] and recorded_public_calls else (
        "partially_recorded" if recorded_public_calls else "not_recorded")
    first_prompt, _, prompt_source = recorded_prompt(episode, first_attempt)
    global_prompt = "\n".join(line for line in (first_prompt or "").splitlines()
                              if not line.startswith(DYNAMIC_PREFIXES)) or None
    task = summary.get("task_text")
    if task is None and records:
        task = read_json(episode / records[0]["observation"]).get("task_text")
    trace.headers(None, summary["agent"], at=summary["started_at"],
                  global_prompt=global_prompt, task_text=task, global_status=prompt_source)
    trace.write(INPUT_LOG, "archive_availability", at=summary["started_at"],
        source_episode=str(episode), raw_mcp_returns="not_recorded",
        developer_instructions="not_recorded" if "developer_instructions" not in summary["agent"] else "recorded",
        reasoning_summary="not_recorded",
        public_messages=public_messages_status,
        note="Canonical observations and execution receipts are recovered; raw MCP envelopes are not fabricated.")
    calls = 0
    for record in records:
        observation = read_observation(episode, record["observation"])
        step = record["step"]
        trace.set_stamp(step, action_step=observation["action_step"], snapshot_id=observation["snapshot_id"])
        trace.write(INPUT_LOG, "observation", at=observation["captured_at"],
            observation=observation, phase="before", source="archived_canonical_observation",
            passed_to_model=False, source_file=str(episode / record["observation"]))
        projection_path = episode / f"step_{step:04d}/model_inputs.json"
        projection = read_json(projection_path) if projection_path.is_file() else []
        for item in projection:
            if item.get("path"):
                item["path"] = str(episode / item["path"])
        feedback = None
        for attempt in record.get("attempts", []):
            calls += 1
            trace.set_stamp(step, attempt["attempt"], action_step=observation["action_step"],
                            snapshot_id=observation["snapshot_id"])
            prompt, schema, source = recorded_prompt(episode, attempt)
            trace.write(INPUT_LOG, "step_input", at=attempt["requested_at"], prompt=prompt,
                response_schema=schema, observation=observation, images=image_placeholders(projection),
                retry_feedback=feedback, passed_to_model=True, transport=source,
                source_file=str(episode / attempt["model_input"]) if attempt.get("model_input") else None)
            received = attempt.get("received_at", attempt["requested_at"])
            if "raw_output" in attempt:
                trace.response(attempt["raw_output"], attempt.get("model_metadata", {}), at=received)
                trace.validation(attempt.get("parse_error"), at=received)
            else:
                trace.request_error(attempt.get("model_error", "No public response was archived."),
                                    at=received, status="not_recorded")
            feedback = attempt.get("parse_error")
        if record.get("action_result") is not None:
            trace.execution(record["action_result"], at=record.get("execution_finished_at", record["finished_at"]))
        if record.get("next_observation"):
            after = read_observation(episode, record["next_observation"])
            trace.set_stamp(step, action_step=after["action_step"], snapshot_id=after["snapshot_id"])
            trace.write(INPUT_LOG, "observation", at=after["captured_at"], observation=after,
                phase="after_execution", source="archived_canonical_observation", passed_to_model=False,
                source_file=str(episode / record["next_observation"]))
    if calls != summary["model_calls"]:
        raise ValueError(f"Archived attempts ({calls}) differ from model_calls ({summary['model_calls']}).")
    trace.finish(summary)
    return {"source_episode": str(episode), "model_calls": calls, "decision_records": len(records),
            "agent_decisions": summary["agent_decisions"], "model_logs": trace.files,
            "raw_mcp_returns": "not_recorded", "reasoning_summary": "not_recorded",
            "public_messages": public_messages_status}


def combine_traces(runs, output):
    output = Path(output)
    combined = {"inputs": str(output / INPUT_LOG), "outputs": str(output / OUTPUT_LOG)}
    if any(Path(path).exists() for path in combined.values()):
        raise ValueError("Combined traces already exist; existing JSONL logs are preserved.")
    for direction, path in combined.items():
        with Path(path).open("x", encoding="utf-8") as stream:
            for row in runs:
                with Path(row["model_logs"][direction]).open(encoding="utf-8") as source_log:
                    for line in source_log:
                        entry = {"run_id": row["id"], "case": row["case"], "model": row["model"],
                                 **json.loads(line)}
                        stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    return combined


def export_comparison(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty export directory; existing traces are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    source = read_json(manifest)
    runs = []
    for row in source["runs"]:
        # summary_file identifies the formal retry/stable run, not an excluded
        # pilot's original protocol directory.
        episode = Path(row["summary_file"]).parent
        exported = export_episode(episode, output / row["id"])
        runs.append({"id": row["id"], "case": row["case"], "model": row["model"], **exported})
    combined = combine_traces(runs, output)
    index = {"schema": "agent_trace_export_v1", "created_at": timestamp(),
             "source_manifest": str(manifest), "runs": runs, "model_logs": combined, "episodes": len(runs),
             "model_calls": sum(row["model_calls"] for row in runs),
             "decision_records": sum(row["decision_records"] for row in runs),
             "agent_decisions": sum(row["agent_decisions"] for row in runs)}
    (output / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path)
    source.add_argument("--comparison-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.comparison_summary is not None:
            result = export_comparison(args.comparison_summary, args.output_dir)
        else:
            result = export_episode(args.episode, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

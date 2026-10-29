#!/usr/bin/env python3
"""Native run/pause lease check, via the same HTTP owner queue as Agent episodes."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

import requests
from mcp import Client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8615")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-s", type=float, default=5.)
    args = parser.parse_args()
    if not 0 < args.wait_s <= 30:
        parser.error("Use a wait between zero and 30 seconds.")
    records = []
    async def cached_mcp():
        async with Client(args.url.rstrip("/") + "/mcp", read_timeout_seconds=15) as client:
            context = await client.call_tool("robot_get_context", {})
            observation = await client.call_tool("robot_get_observation", {"include_sensor_data": False})
            if context.is_error or observation.is_error:
                raise RuntimeError("Cached MCP read failed while awaiting a model.")
            return {"context": context.structured_content["context"],
                    "snapshot_id": observation.structured_content["observation"]["snapshot_id"]}
    def post(path, body):
        response = requests.post(args.url+path, json=body, timeout=(5, 220))
        response.raise_for_status()
        return response.json()
    def state():
        response = requests.get(args.url+"/api/agent/state", timeout=10)
        response.raise_for_status()
        return response.json()
    for action_mode in ("fixed", "parameterized"):
        for world_mode in ("run", "pause"):
            claim_body = {"run_id": f"world-check-{action_mode}-{world_mode}",
                "task_text": "World-policy verification.", "action_mode": action_mode}
            if world_mode == "pause":
                claim_body["world_advances_during_inference"] = False
            claim = post("/api/control/claim", claim_body)
            token = claim["control_token"]
            try:
                before = state()
                started = time.perf_counter()
                invalid = requests.post(args.url+"/api/agent/command", json={"command": {
                    "action": "MV_UP", "distance_m": .005, "world_advances_during_inference": True},
                    "control_token": token}, timeout=10)
                human_request = requests.post(args.url+"/api/step", json={"tokens": ["MV_DOWN"]}, timeout=10)
                # Invalid commands and read-only cached observations are part of waiting.
                observation = requests.get(args.url+"/api/agent-observation", timeout=10)
                observation.raise_for_status()
                mcp_started = time.perf_counter()
                mcp = asyncio.run(cached_mcp())
                mcp_wall = time.perf_counter()-mcp_started
                time.sleep(args.wait_s)
                waited = state()
                wall = time.perf_counter()-started
                sim = waited["sampled_simulation_time_s"]-before["sampled_simulation_time_s"]
                snapshot_same = waited["observation_snapshot_id"] == before["observation_snapshot_id"]
                if action_mode == "fixed":
                    motion = post("/api/agent/step", {"action": "MV_DOWN", "control_token": token})
                else:
                    motion = post("/api/agent/command", {"command": {"action": "MV_DOWN", "distance_m": .005},
                                                      "control_token": token})
                after = state()
                time.sleep(1.)
                second_wait = state()
                post_action_wait_sim = second_wait["sampled_simulation_time_s"]-after["sampled_simulation_time_s"]
                post("/api/control/release", {"control_token": token})
                released = state()
                time.sleep(1.)
                human_after = state()
                row = {"action_mode": action_mode, "world_mode": world_mode,
                    "wait_wall_time_s": wall, "wait_sim_time_s": sim, "before": before,
                    "after_wait": waited, "after_action": after, "human_after": human_after,
                    "invalid_status": invalid.status_code, "cached_observation_responsive": True,
                    "human_request_status": human_request.status_code,
                    "post_action_wait_sim_time_s": post_action_wait_sim,
                    "cached_mcp_responsive": True, "cached_mcp_wall_time_s": mcp_wall,
                    "mcp_world_mode": mcp["context"]["world_mode"],
                    "mcp_snapshot_id": mcp["snapshot_id"],
                    "snapshot_unchanged_during_wait": snapshot_same,
                    "one_action_boundary": after["steps"] == 1 and after["observation_snapshot_id"] == before["observation_snapshot_id"]+1,
                    "action_completed": motion.get("ok") is True,
                    "human_resumed": human_after["sampled_simulation_time_s"] > released["sampled_simulation_time_s"]}
                row["passed"] = bool(invalid.status_code == 400 and snapshot_same and row["one_action_boundary"]
                    and human_request.status_code == 409
                    and mcp["context"]["world_mode"] == world_mode
                    and mcp["snapshot_id"] == before["observation_snapshot_id"]
                    and row["action_completed"] and row["human_resumed"]
                    and after["action_execution_sim_time_s"] > 0
                    and second_wait["observation_snapshot_id"] == after["observation_snapshot_id"]
                    and (post_action_wait_sim > 0 if world_mode == "run" else abs(post_action_wait_sim) < 1e-8)
                    and (sim > 0 if world_mode == "run" else abs(sim) < 1e-8))
                records.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps({"schema": "native_world_policy_check_v1",
                    "passed": all(row["passed"] for row in records), "records": records}, indent=2)+"\n")
                print(action_mode, world_mode, "wall", wall, "sim", sim, "passed", row["passed"], flush=True)
            finally:
                post("/api/control/release", {"control_token": token})
    return 0 if all(row["passed"] for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

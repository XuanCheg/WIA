"""Whole-episode WIA integration. Run: wia eval --config FILE after editable installs."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess

from showharness.resources import require_checkout


POLICY_FLAGS = {"backend": "backend", "model": "model", "base_url": "base-url", "api_key_env": "api-key-env",
                "provider": "provider", "max_output_tokens": "max-output-tokens", "temperature": "temperature",
                "reasoning_effort": "reasoning-effort", "codex_executable": "codex-executable", "model_timeout": "model-timeout",
                "history_text_turns": "history-text-turns", "image_max_edge": "image-max-edge",
                "sliding_window_tokens": "sliding-window-tokens", "history_policy": "history-policy",
                "history_recent_moves": "history-recent-moves", "max_perception_rounds": "max-perception-rounds"}


def build_arguments(plan):
    """Build a single existing runner invocation without importing its runtime."""
    task, policy, env, evaluation = (plan[key] for key in ("task", "policy", "environment", "evaluation"))
    launch = env["mode"] == "launch"
    arguments = ["--task-file", task["path"], "--seed", str(task["seed"]), "--variant", task["variant"],
                 "--output-dir", plan["output_dir"]]
    for key, flag in POLICY_FLAGS.items():
        if policy.get(key) is not None:
            arguments += ["--" + flag, str(policy[key])]
    if "model_timeout" in policy and policy["model_timeout"] is None:
        arguments += ["--model-timeout", "none"]
    if "chat_template_kwargs" in policy:
        arguments += ["--chat-template-kwargs", json.dumps(policy["chat_template_kwargs"])]
    if policy.get("json_schema_output"):
        arguments += ["--json-schema-output"]
    if policy.get("structured_output") is False:
        arguments += ["--no-local-structured-output"]
    flags = {"max_steps": "max-steps", "action_mode": "action-mode", "observation_condition": "observation-condition",
             "world": "world" if launch else "world-during-thinking", "run_mode": "run-mode",
             "max_invalid_retries": "max-invalid-retries", "max_recoverable_failures": "max-recoverable-failures",
             "snapshot_cache": "snapshot-cache"}
    for key, flag in flags.items():
        if key in evaluation and evaluation[key] is not None:
            arguments += ["--" + flag, str(evaluation[key])]
    if "timeout" in evaluation:
        if launch:
            arguments += ["--timeout", "none" if evaluation["timeout"] is None else str(evaluation["timeout"])]
        elif evaluation["timeout"] is not None:
            arguments += ["--episode-timeout", str(evaluation["timeout"])]
    if "world_observation_log" in evaluation:
        arguments += ["--world-observation-log" if evaluation["world_observation_log"] else "--no-world-observation-log"]
    if env.get("data_root"):
        arguments += ["--data-root", env["data_root"]]
    if launch:
        arguments += ["--execution-mode", evaluation.get("execution_mode", "batch")]
        for key in ("gpu", "port", "python", "preview_fps", "recording_hz", "macro_execution"):
            if key in env:
                arguments += ["--" + key.replace("_", "-"), str(env[key])]
        if "record" in env:
            arguments += ["--record" if env["record"] else "--norecord"]
        if "debug_cameras" in env:
            arguments += ["--debug-cameras" if env["debug_cameras"] else "--no-debug-cameras"]
        if env.get("keep_simulator"):
            arguments += ["--keep-simulator"]
    else:
        arguments += ["--url", env["url"], "--episode-manifest", env["episode_manifest"]]
        if env.get("evaluation_file"):
            arguments += ["--evaluation-file", env["evaluation_file"]]
    return arguments


def execute_plan(plan):
    """Call one authoritative runner; no model decisions or physical actions live here."""
    launch = plan["environment"]["mode"] == "launch"
    root = require_checkout(launch=launch)
    output = Path(plan["output_dir"])
    if output.exists():
        raise ValueError(f"Trial output must not already exist: {output}")
    if launch:
        import wia.runtime.commands.run_esi_explore_episode as runner
    else:
        import wia.runtime.commands.run_esi_agent as runner
    runner_error = None
    try:
        code = runner.main(build_arguments(plan))
    except (Exception, SystemExit) as error:
        runner_error = error
        code = None
    active_episode = "episode"
    if launch and (output / "experiment.json").exists():
        active_episode = json.loads((output / "experiment.json").read_text()).get("active_episode_dir", "episode")
    summary_path = output / active_episode / "summary.json" if launch else output / "summary.json"
    if not summary_path.is_file():
        if runner_error is not None:
            if isinstance(runner_error, SystemExit):
                raise RuntimeError(f"Harness argument validation exited {runner_error.code}; no episode summary was produced.") from runner_error
            raise runner_error
        raise RuntimeError(f"Harness exited {code} without an authoritative summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    try:
        source = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True)
        source_commit = source.stdout.strip() if source.returncode == 0 else None
    except OSError:
        source_commit = None
    try:
        source_status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                       text=True, capture_output=True)
        source_dirty = bool(source_status.stdout) if source_status.returncode == 0 else None
    except OSError:
        source_dirty = None
    try:
        installed_version = version("showharness")
    except PackageNotFoundError:
        from showharness import __version__
        installed_version = __version__
    # Preserve the runner summary verbatim on disk; WIA records integration provenance separately.
    return {**summary, "harness_provenance": {"version": installed_version,
            "commit": source_commit, "dirty": source_dirty, "root": str(root),
            "summary_path": str(summary_path), "runner_returncode": code,
            "runner_error": str(runner_error) if runner_error else None}}


class WIADatasetAdapter:
    """WIA owns the plan and episode; Harness supplies reusable model capabilities."""
    def execute(self, plan):
        return execute_plan(plan)

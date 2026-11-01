"""Prepare reproducible trials and aggregate runner summaries. Run: wia eval --config FILE."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import yaml

from wia import __version__
from wia.evaluation.usage import usage_report, aggregate_usage
from wia.policies import available_policies, create_executor

BACKENDS = {
    "codex_cli": {"history_owner": "native", "resume": True},
    "openai_chat_completions": {"history_owner": "harness", "resume": False},
    "openai_responses": {"history_owner": "harness", "resume": False},
    "local_http": {"history_owner": "harness", "resume": False},
}
POLICY_KEYS = {"type", "backend", "model", "base_url", "api_key_env", "provider",
               "max_output_tokens", "temperature", "reasoning_effort", "chat_template_kwargs",
               "codex_executable", "model_timeout", "structured_output", "history_text_turns", "json_schema_output", "image_max_edge", "sliding_window_tokens", "history_policy", "history_recent_moves", "max_perception_rounds"}
EVAL_KEYS = {"max_steps", "timeout", "action_mode", "world", "observation_condition", "run_mode",
             "world_observation_log", "max_invalid_retries", "max_recoverable_failures", "snapshot_cache", "execution_mode"}
ENV_KEYS = {"mode", "gpu", "port", "python", "data_root", "url", "episode_manifest", "evaluation_file",
            "record", "preview_fps", "recording_hz", "debug_cameras", "keep_simulator", "macro_execution"}


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _known(mapping, names, label):
    unknown = set(mapping) - names
    if unknown:
        raise ValueError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")


def _path(value, base):
    path = Path(value).expanduser()
    return str((base / path if not path.is_absolute() else path).resolve())


def _validate_numbers(policy, evaluation):
    for key in ("max_steps", "max_output_tokens", "max_invalid_retries", "max_recoverable_failures"):
        value = evaluation.get(key, policy.get(key))
        if key == "max_steps" and type(value) is int and value == -1:
            continue
        if value is not None and (type(value) is not int or value < (0 if "retries" in key or "failures" in key else 1)):
            raise ValueError(f"{key} has an invalid integer value.")
    for key, value in (("timeout", evaluation.get("timeout")), ("model_timeout", policy.get("model_timeout"))):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be a finite positive duration or null.")
    if "temperature" in policy and (not isinstance(policy["temperature"], (int, float)) or not math.isfinite(policy["temperature"])):
        raise ValueError("temperature must be finite.")
    for key, choices in {"action_mode": ("fixed", "parameterized"), "world": ("run", "pause"),
                         "observation_condition": ("current", "initial_view_only"),
                         "run_mode": ("benchmark", "diagnostic")}.items():
        if key in evaluation and evaluation[key] not in choices:
            raise ValueError(f"Invalid {key}: {evaluation[key]}")


def prepare(config_path, *, execution_mode=None, max_steps=None):
    """Read/validate without importing Harness, starting scenes or writing output."""
    config_path = Path(config_path).expanduser().resolve()
    from wia.tasks.loader import UniqueLoader
    config = _mapping(yaml.load(config_path.read_text(), Loader=UniqueLoader), "config")
    _known(config, {"tasks", "policy", "environment", "evaluation", "output_dir", "resource_root"}, "config")
    base = config_path.parent
    resource_root = Path(_path(config["resource_root"], base)) if "resource_root" in config else base
    policy = _mapping(config.get("policy"), "policy")
    policy_type = policy.get("type")
    if policy_type not in available_policies():
        raise ValueError(f"Unknown policy type: {policy_type}")
    if policy_type == "show_harness":
        if not isinstance(policy.get("model"), str) or not policy["model"].strip():
            raise ValueError("policy.model must be explicitly supplied.")
        _known(policy, POLICY_KEYS, "policy")
        if policy.get("backend") not in BACKENDS:
            raise ValueError(f"policy.backend must be one of {', '.join(BACKENDS)}")
        backend = policy["backend"]
        if policy.get("structured_output") is False:
            raise ValueError("Declarative evaluation requires structured output; this setting cannot be disabled.")
        if "api_key_env" in policy and (not isinstance(policy["api_key_env"], str) or not policy["api_key_env"].isidentifier()):
            raise ValueError("api_key_env must be an environment variable name, never a credential value.")
        if "base_url" in policy:
            from urllib.parse import urlsplit
            parsed = urlsplit(policy["base_url"])
            if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query:
                raise ValueError("base_url must be an HTTP URL without credentials or query parameters.")
        if backend == "codex_cli" and any(key in policy for key in ("base_url", "api_key_env", "provider", "chat_template_kwargs", "max_output_tokens", "temperature")):
            raise ValueError("Codex CLI uses its own configuration; HTTP generation/credential fields are unsupported.")
        if "provider" in policy and (backend != "openai_chat_completions" or policy["provider"] not in ("openai", "vllm", "compatible")):
            raise ValueError("provider applies only to openai_chat_completions and must be openai, vllm or compatible.")
        if "chat_template_kwargs" in policy and (backend != "openai_chat_completions" or policy.get("provider") != "vllm"):
            raise ValueError("chat_template_kwargs requires Chat Completions provider=vllm.")
    environment = _mapping(config.get("environment", {}), "environment")
    evaluation = _mapping(config.get("evaluation", {}), "evaluation")
    if max_steps is not None:
        if type(max_steps) is not int or max_steps != -1 and max_steps < 1:
            raise ValueError("max_steps must be -1 (task default) or a positive integer.")
        evaluation["max_steps"] = max_steps
    if execution_mode is not None:
        evaluation["execution_mode"] = execution_mode
    evaluation.setdefault("execution_mode", "batch")
    if evaluation["execution_mode"] not in ("debug", "batch"):
        raise ValueError("evaluation.execution_mode must be debug or batch.")
    if evaluation["execution_mode"] == "debug":
        evaluation.setdefault("run_mode", "diagnostic")
    if policy_type == "show_harness":
        _known(environment, ENV_KEYS, "environment")
        _known(evaluation, EVAL_KEYS, "evaluation")
        _validate_numbers(policy, evaluation)
        from wia.runtime.agent.history_policy import validate_history_policy
        validate_history_policy(policy)
        if 'sliding_window_tokens' in policy:
            window = policy['sliding_window_tokens']
            if (evaluation.get('run_mode') != 'diagnostic' or type(window) is not int
                    or window <= policy.get('max_output_tokens', 1024) + 2048
                    or policy.get('history_text_turns') is not None):
                raise ValueError('sliding_window_tokens requires diagnostic mode, full-turn history and sufficient context.')
        if "history_text_turns" in policy:
            if (evaluation.get("run_mode") != "diagnostic"
                    or type(policy["history_text_turns"]) is not int or policy["history_text_turns"] < 0):
                raise ValueError("history_text_turns requires diagnostic mode and a nonnegative integer.")
        if "json_schema_output" in policy and (type(policy["json_schema_output"]) is not bool
                 or evaluation.get("run_mode") != "diagnostic"):
            raise ValueError("json_schema_output requires diagnostic mode and a boolean.")
        if "image_max_edge" in policy and (type(policy["image_max_edge"]) is not int or policy["image_max_edge"] <= 0
                 or evaluation.get("run_mode") != "diagnostic"):
            raise ValueError("image_max_edge requires diagnostic mode and a positive integer.")
    mode = environment.get("mode", "launch")
    if mode not in ("launch", "attach"):
        raise ValueError("environment.mode must be launch or attach.")
    environment["mode"] = mode
    if policy_type == "show_harness" and policy["backend"] == "local_http" and mode == "launch" and not policy.get("base_url"):
        raise ValueError("local_http requires an explicit base_url.")
    if (policy_type == "show_harness" and mode == "launch"
            and policy["backend"] == "openai_chat_completions"
            and policy.get("provider") in ("vllm", "compatible") and not policy.get("base_url")):
        raise ValueError("vllm and compatible providers require an explicit base_url.")
    if policy_type == "show_harness":
        ignored = ({"url", "episode_manifest", "evaluation_file"} if mode == "launch" else
                   {"gpu", "port", "python", "record", "preview_fps", "recording_hz", "debug_cameras", "keep_simulator", "macro_execution"}) & environment.keys()
        if ignored:
            raise ValueError(f"Environment fields unsupported in {mode}: {', '.join(sorted(ignored))}")
    for key in ("data_root", "episode_manifest", "evaluation_file"):
        if key in environment:
            environment[key] = _path(environment[key], resource_root)
    if not config.get("output_dir"):
        raise ValueError("output_dir is required.")
    output = Path(_path(config["output_dir"], base))
    if output.exists():
        raise ValueError(f"Use a fresh output directory (must not exist): {output}")
    tasks = config.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty list of task selections.")
    if evaluation["execution_mode"] == "debug" and (len(tasks) != 1 or mode != "launch"):
        raise ValueError("Debug requires one launched task; use its live resume control for continuation.")
    if evaluation["execution_mode"] == "batch" and environment.get("keep_simulator"):
        raise ValueError("Batch evaluation cannot keep a simulator; use debug mode.")
    from wia.tasks.loader import is_release_v1, release_entry, select_task, task_definition
    plans = []
    attach_urls, attach_manifests = set(), set()
    for index, selection in enumerate(tasks):
        selection = _mapping(selection, "task selection")
        _known(selection, {"path", "seed", "variant", "url", "episode_manifest", "evaluation_file"}, "task")
        if not selection.get("path"):
            raise ValueError("Each task selection requires path.")
        candidate = Path(_path(selection["path"], resource_root))
        task_path = candidate if candidate.is_file() else select_task(selection["path"])
        definition = task_definition(yaml.load(task_path.read_text(), Loader=UniqueLoader))
        defaults = definition.defaults.model_dump(mode="json")
        episode_evaluation = {key: value for key, value in evaluation.items()
                              if key != "max_steps" or value != -1}
        effective = {"max_steps": defaults["max_steps"], "timeout": None,
                     "action_mode": defaults["action_mode"],
                     "world": "run" if defaults["world_advances_during_inference"] else "pause",
                     "observation_condition": defaults["observation_condition"], "run_mode": "benchmark", **episode_evaluation,
                     "timeout": None}
        rigid = getattr(definition, "kind", None) == "rigid_containment_task"
        selected = release_entry(task_path)
        default_variant = selected['variant'] if selected else ("tabletop" if rigid else "original")
        task = {"path": str(task_path.resolve()), "seed": selection.get("seed", defaults["seed"]),
                "variant": selection.get("variant", default_variant),
                "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest()}
        if type(task["seed"]) is not int or task["seed"] < 0:
            raise ValueError("Task seed must be a nonnegative integer.")
        if rigid and task["variant"] != "tabletop":
            raise ValueError("Containment tasks require the approved tabletop variant.")
        if not rigid and hasattr(definition, "scene") and task["variant"] not in definition.scene.variants:
            raise ValueError("Task variant is not declared by this task.")
        if rigid and getattr(definition, "status", None) != "ready" and not is_release_v1(task_path):
            raise ValueError("Containment task must be ready before execution.")
        if (getattr(definition, "kind", None) == "category_robot_task"
                and getattr(definition, "status", None) != "ready"
                and effective["run_mode"] != "diagnostic" and not is_release_v1(task_path)):
            raise ValueError("Pending/discarded category task requires diagnostic mode.")
        episode_env = dict(environment)
        if mode == "launch" and policy_type == "show_harness":
            episode_env.setdefault("record", effective["execution_mode"] == "debug")
            episode_env.setdefault("recording_hz", 5.0)
            if type(episode_env["record"]) is not bool:
                raise ValueError("environment.record must be a boolean.")
            hz = episode_env["recording_hz"]
            if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
                raise ValueError("environment.recording_hz must be finite and positive.")
        if policy_type == "show_harness" and mode == "launch" and set(selection) & {"url", "episode_manifest", "evaluation_file"}:
            raise ValueError("Per-task attach settings cannot be used in launch mode.")
        if mode == "attach" and policy_type == "show_harness":
            if len(tasks) > 1 and not all(selection.get(key) for key in ("url", "episode_manifest")):
                raise ValueError("Multiple attach episodes require per-task url and episode_manifest; scene reuse is ambiguous.")
            for key in ("url", "episode_manifest", "evaluation_file"):
                if key in selection:
                    episode_env[key] = selection[key] if key == "url" else _path(selection[key], resource_root)
            if not episode_env.get("url") or not episode_env.get("episode_manifest"):
                raise ValueError("Attach requires an existing URL and prepared immutable episode_manifest.")
            if episode_env["url"] in attach_urls or episode_env["episode_manifest"] in attach_manifests:
                raise ValueError("Attach episodes cannot reuse a URL or episode_manifest; prepare separate episodes.")
            attach_urls.add(episode_env["url"])
            attach_manifests.add(episode_env["episode_manifest"])
            manifest = json.loads(Path(episode_env["episode_manifest"]).read_text())
            if manifest.get("task_file_sha256") != task["sha256"] or manifest.get("task_id") != definition.id:
                raise ValueError("Attach task identity differs from episode_manifest.")
            if "reasoning_effort" in policy and policy["reasoning_effort"] != manifest.get("reasoning_effort"):
                raise ValueError("Attach reasoning_effort differs from episode_manifest.")
            policy = {**policy, "reasoning_effort": manifest.get("reasoning_effort")}
            metadata = manifest.get("backend_metadata")
            if metadata is not None:
                canonical = "codex_cli" if metadata.get("backend") == "codex" else metadata.get("backend")
                if canonical != policy["backend"]:
                    raise ValueError("Attach backend metadata differs from selected backend.")
                if policy["backend"] != "codex_cli":
                    for field in ("base_url", "api_key_env", "provider", "max_output_tokens", "temperature", "chat_template_kwargs", "history_text_turns", "json_schema_output", "image_max_edge", "sliding_window_tokens", "history_policy", "history_recent_moves", "max_perception_rounds"):
                        if field in policy and policy[field] != metadata.get(field):
                            raise ValueError(f"Attach {field} differs from immutable backend metadata.")
                        if metadata.get(field) is not None:
                            policy = {**policy, field: metadata[field]}
            if policy["backend"] == "local_http" and not policy.get("base_url"):
                raise ValueError("Attach local_http requires base_url in the config or prepared backend metadata.")
            if (policy["backend"] == "openai_chat_completions"
                    and policy.get("provider") in ("vllm", "compatible") and not policy.get("base_url")):
                raise ValueError("Attach vllm and compatible providers require base_url in the config or prepared backend metadata.")
            enabled = manifest.get("world_observation_log", True)
            if "world_observation_log" in evaluation and evaluation["world_observation_log"] != enabled:
                raise ValueError("Attach world_observation_log differs from episode_manifest.")
            effective["world_observation_log"] = enabled
            pairs = {"max_steps": "max_steps", "timeout": "wall_timeout_s", "action_mode": "action_mode",
                     "world": "world_mode", "observation_condition": "observation_condition"}
            for key, manifest_key in pairs.items():
                if key in episode_evaluation and episode_evaluation[key] != manifest[manifest_key]:
                    raise ValueError(f"Attach {key} differs from immutable episode_manifest.")
                effective[key] = manifest[manifest_key]
            if policy["model"] != manifest.get("model_name"):
                raise ValueError("Attach model differs from episode_manifest.")
            expected = "native_persistent" if policy.get("backend") == "codex_cli" else (
                "local_http_persistent" if policy.get("backend") == "local_http" else policy.get("backend") + "_persistent")
            if manifest.get("agent_session") != expected:
                raise ValueError("Attach backend differs from episode_manifest.")
            for key, manifest_key in (("seed", "scene_seed"), ("variant", "scene_variant")):
                if key in selection and selection[key] != manifest.get(manifest_key):
                    raise ValueError(f"Attach {key} differs from episode_manifest.")
                if manifest_key in manifest:
                    task[key] = manifest[manifest_key]
        plan = {"index": index, "task": task, "policy": policy, "environment": episode_env,
                "evaluation": effective, "output_dir": str(output / f"trial_{index:04d}")}
        plans.append(plan)
    return {"schema": "wia_evaluation_v1", "wia_version": __version__, "config_path": str(config_path),
            "output_dir": str(output), "plans": plans}


def _version(root):
    root = Path(root)
    commit = dirty = None
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            commit = result.stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True)
        if status.returncode == 0:
            dirty = bool(status.stdout)
    except OSError:
        pass
    return {"path": str(root), "commit": commit, "dirty": dirty}


def summarize(summary):
    """Count completed failures separately from interruption and invalid scoring."""
    reason = summary.get("termination_reason")
    if "usage" in summary:
        usage = summary["usage"]
    elif "model_usage" in summary:
        usage = summary["model_usage"]
    else:
        usage = (summary.get("native_session") or {}).get("usage")
    if reason in ("model_error", "runtime_error", "environment_error", "execution_error", "observation_error", "cancelled"):
        status = reason
    elif (summary.get("evaluation_validity") != "complete"
          or summary.get("execution_validity") == "interrupted"
          or summary.get("agent_failure_countable") is False):
        status = "invalid_or_incomplete_score"
    elif summary.get("task_success") is True:
        status = "completed_success"
    elif summary.get("task_success") is False:
        status = "completed_unsuccessful"
    else:
        status = "invalid_or_incomplete_score"
    return {"status": status, "termination_reason": reason, "task_success": summary.get("task_success"),
            "evaluation_validity": summary.get("evaluation_validity"),
            "completion_score": summary.get("completion_score"), "score_percent": summary.get("score_percent"),
            "agent_decisions": summary.get("agent_decisions"), "executed_actions": summary.get("executed_actions"),
            "model_calls": summary.get("model_calls"), "decision_calls": summary.get("decision_calls"),
            "request_accounting": summary.get("request_accounting"),
            "safety_statistics": summary.get("safety_statistics"), "timings": summary.get("timings"),
            "usage": usage, "token_accounting": usage_report(summary), "cost": summary.get("cost"),
            "model_logs": summary.get("model_logs")}


def run(prepared):
    """Execute episodes serially; the selected integration owns all physical decisions."""
    for plan in prepared["plans"]:
        current = hashlib.sha256(Path(plan["task"]["path"]).read_bytes()).hexdigest()
        if current != plan["task"]["sha256"]:
            raise ValueError("Task changed after preparation; prepare a fresh effective configuration.")
    output = Path(prepared["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    from wia.resources import ROOT, ESI_ROOT
    frozen = {**prepared, "created_at": datetime.now(timezone.utc).isoformat(),
              "sources": {"wia": _version(ROOT), "esi": _version(ESI_ROOT)}}
    (output / "effective_config.json").write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    results = []
    for plan in prepared["plans"]:
        try:
            current = hashlib.sha256(Path(plan["task"]["path"]).read_bytes()).hexdigest()
            if current != plan["task"]["sha256"]:
                raise ValueError("Task changed after preparation; execution refused.")
            authoritative = create_executor(plan["policy"]["type"]).execute(plan)
            result = {**summarize(authoritative), "summary": authoritative}
        except Exception as error:
            result = {"status": "runtime_error", "error": str(error), "error_type": type(error).__name__, "cost": None}
        results.append({"index": plan["index"], "task": plan["task"], "output_dir": plan["output_dir"], **result})
        counts = {status: sum(item["status"] == status for item in results) for status in sorted({item["status"] for item in results})}
        from .activity import aggregate_activity
        aggregate = {"activity": aggregate_activity(results), "schema": "wia_results_v1", "trials": results, "counts": counts,
                     "token_accounting": aggregate_usage(results),
                     "cost": sum(item["cost"] for item in results) if all(isinstance(item.get("cost"), (int, float)) for item in results) else None}
        temporary = output / 'results.json.tmp'
        temporary.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / 'results.json')
    return aggregate

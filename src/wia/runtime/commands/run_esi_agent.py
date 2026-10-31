#!/usr/bin/env python3
"""Run a robot episode: .venv/bin/python src/wia/runtime/commands/run_esi_agent.py --help.

Use --scene-file FILE for original-question QA in a retained scene, or --task-file
FILE with its immutable --episode-manifest for a declarative physical task.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import asyncio
import json
import os
import sys
import uuid
import math
from pathlib import Path

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

# Runtime dependencies stay lazy so parsing/help requires no MCP installation.
MCPRobotEnvironment = None
run_episode = None

def optional_timeout(value):
    """Parse an explicitly bounded wait or 'none' for an interruptible unbounded wait."""
    return None if value.lower() == 'none' else float(value)


BACKEND_NAMES = ("codex", "codex_cli", "local_http", "openai_chat_completions", "openai_responses")


def canonical_backend(name):
    """Normalize only transport identity; retain legacy manifest policy values."""
    return "codex_cli" if name == "codex" else name


def session_policy(backend):
    """Keep existing native/local session IDs and explicit IDs for new dialects."""
    backend = canonical_backend(backend)
    return {"codex_cli": "native_persistent", "local_http": "local_http_persistent"}.get(
        backend, backend + "_persistent")


def add_backend_arguments(parser):
    """Add transport options without importing model clients or MCP."""
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="codex")
    parser.add_argument("--base-url", help="HTTP API base URL; never applied to Codex CLI")
    parser.add_argument("--api-key-env", help="Environment variable NAME containing the HTTP credential")
    parser.add_argument("--provider", choices=("openai", "vllm", "compatible"))
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--chat-template-kwargs", type=json.loads, help="JSON object; provider=vllm only")
    from wia.runtime.agent.history_policy import HISTORY_POLICIES
    parser.add_argument("--history-policy", choices=HISTORY_POLICIES)
    parser.add_argument("--max-perception-rounds", type=int, help="Previous-result-notes policy: perception rounds per decision (default 4); note reads remain unlimited.")
    parser.add_argument("--history-recent-moves", type=int, help="ShowHarness state policy: recent executed actions (default 3).")
    parser.add_argument("--history-text-turns", type=int,
                        help="Diagnostic Chat Completions only: replay this many public final outputs, omitting older images/prompts/tools.")
    parser.add_argument("--json-schema-output", action="store_true",
                        help="Diagnostic Chat Completions only: request server-side JSON Schema constrained decoding.")
    parser.add_argument("--image-max-edge", type=int,
                        help="Diagnostic Chat Completions only: resize model RGB while preserving native depth and calibration.")
    parser.add_argument('--sliding-window-tokens', type=int,
                        help='Diagnostic vLLM: bound complete history turns using the serving tokenizer; pin the initial turn.')
    parser.add_argument("--local-base-url", help="Legacy local HTTP base URL alias")
    parser.add_argument("--local-api-key-env", default="LOCAL_AGENT_API_KEY")
    parser.add_argument("--no-local-structured-output", action="store_true")


def validate_backend_arguments(args):
    """Reject incompatible options before starting a scene or model session."""
    backend = canonical_backend(args.backend)
    from wia.runtime.agent.history_policy import validate_history_policy
    validate_history_policy({**vars(args), "backend": backend, "base_url": args.base_url or args.local_base_url})
    history_turns = getattr(args, "history_text_turns", None)
    image_edge = getattr(args, "image_max_edge", None)
    window = getattr(args, 'sliding_window_tokens', None)
    if window is not None and (backend != 'openai_chat_completions' or args.provider != 'vllm'
            or getattr(args, 'run_mode', None) != 'diagnostic' or window <= (args.max_output_tokens or 1024) + 2048
            or history_turns is not None):
        raise ValueError('--sliding-window-tokens requires diagnostic vllm and sufficient context with full-turn history.')
    if image_edge is not None and (image_edge <= 0 or backend != "openai_chat_completions"
                                  or getattr(args, "run_mode", None) != "diagnostic"):
        raise ValueError("--image-max-edge requires diagnostic Chat Completions and a positive integer.")
    if getattr(args, "json_schema_output", False) and (backend != "openai_chat_completions"
            or getattr(args, "run_mode", None) != "diagnostic"):
        raise ValueError("--json-schema-output requires diagnostic Chat Completions.")
    if history_turns is not None and (backend != "openai_chat_completions" or history_turns < 0
                                     or getattr(args, "run_mode", None) != "diagnostic"):
        raise ValueError("--history-text-turns requires diagnostic Chat Completions and a nonnegative integer.")
    if backend == "codex_cli" and any(getattr(args, key, None) is not None for key in
            ("base_url", "api_key_env", "provider", "max_output_tokens", "temperature", "chat_template_kwargs")):
        raise ValueError("HTTP API options do not configure Codex CLI.")
    if args.provider is not None and backend != "openai_chat_completions":
        raise ValueError("--provider requires --backend openai_chat_completions.")
    if (backend == "openai_chat_completions" and args.provider in ("vllm", "compatible")
            and not (args.base_url or args.local_base_url)):
        raise ValueError("vllm and compatible providers require --base-url (or legacy --local-base-url).")
    if args.chat_template_kwargs is not None and (backend != "openai_chat_completions" or args.provider != "vllm"
                                                 or not isinstance(args.chat_template_kwargs, dict)):
        raise ValueError("--chat-template-kwargs requires provider=vllm and a JSON object.")
    if backend == "local_http" and not (args.base_url or args.local_base_url):
        raise ValueError("local_http requires --base-url (or legacy --local-base-url).")
    if backend == "local_http" and any(getattr(args, key, None) is not None for key in
            ("max_output_tokens", "temperature", "chat_template_kwargs")):
        raise ValueError("Generation options require an explicit OpenAI API backend.")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive.")


def backend_metadata(args):
    """Record transport/generation settings, never credential values."""
    backend = canonical_backend(args.backend)
    if backend == "codex_cli":
        return {"backend": backend, "history_owner": "native", "provider": None, "base_url": None,
                "api_key_env": None, "max_output_tokens": None, "temperature": None,
                "chat_template_kwargs": None}
    default_key_env = (args.local_api_key_env if backend == "local_http" else
                       "VLLM_API_KEY" if args.provider == "vllm" else
                       "LOCAL_AGENT_API_KEY" if args.provider == "compatible" else "OPENAI_API_KEY")
    metadata = {"backend": backend, "history_owner": "native" if backend == "codex_cli" else "harness",
            "provider": args.provider, "base_url": args.base_url or args.local_base_url,
            "api_key_env": args.api_key_env or default_key_env,
            "max_output_tokens": args.max_output_tokens, "temperature": args.temperature,
            "chat_template_kwargs": args.chat_template_kwargs}
    if getattr(args, "history_text_turns", None) is not None:
        metadata["history_text_turns"] = args.history_text_turns
    if getattr(args, "json_schema_output", False):
        metadata["json_schema_output"] = True
    if getattr(args, "image_max_edge", None) is not None:
        metadata["image_max_edge"] = args.image_max_edge
    if getattr(args, 'sliding_window_tokens', None) is not None:
        metadata['sliding_window_tokens'] = args.sliding_window_tokens
    for field in ("history_policy", "history_recent_moves", "max_perception_rounds"):
        if getattr(args, field, None) is not None:
            metadata[field] = getattr(args, field)
    return metadata


def backend_cli_arguments(args):
    """Forward selected backend fields from the scene launcher to the agent."""
    result = ["--backend", args.backend]
    for field in ("base_url", "api_key_env", "provider", "max_output_tokens", "temperature", "local_base_url", "local_api_key_env", "history_text_turns", "image_max_edge", "sliding_window_tokens", "history_policy", "history_recent_moves", "max_perception_rounds"):
        value = getattr(args, field, None)
        if value is not None:
            result += ["--" + field.replace("_", "-"), str(value)]
    if args.chat_template_kwargs is not None:
        result += ["--chat-template-kwargs", json.dumps(args.chat_template_kwargs)]
    if args.no_local_structured_output:
        result += ["--no-local-structured-output"]
    if getattr(args, "json_schema_output", False):
        result += ["--json-schema-output"]
    return result


def create_declarative_session(args, *, public_task, policy, output, assembler, resume_summary=None):
    """Use the extensible backend registry with transport-specific arguments."""
    from core.agent.backend_registry import create_session
    backend = canonical_backend(args.backend)
    kwargs = dict(public_task=public_task, episode_configuration=policy, output=output,
                  model=args.model, timeout_s=args.model_timeout, assembler=assembler,
                  snapshot_cache=args.snapshot_cache == "optimized")
    if backend == "codex_cli":
        kwargs.update(reasoning_effort=args.reasoning_effort, reasoning_summary=args.reasoning_summary,
                      executable=args.codex_executable,
                      resume_state=resume_summary["native_session"] if resume_summary else None)
    else:
        kwargs.update(base_url=args.base_url or args.local_base_url or "https://api.openai.com/v1",
                      api_key=os.environ.get(backend_metadata(args)["api_key_env"], ""))
        if backend != "local_http":
            if args.max_output_tokens is not None:
                kwargs["max_output_tokens"] = args.max_output_tokens
            kwargs.update(temperature=args.temperature, reasoning_effort=args.reasoning_effort)
            if backend == "openai_chat_completions":
                kwargs.update(provider=args.provider or "openai", chat_template_kwargs=args.chat_template_kwargs)
                if getattr(args, "history_text_turns", None) is not None:
                    kwargs["history_text_turns"] = args.history_text_turns
                if getattr(args, "json_schema_output", False):
                    kwargs["json_schema_output"] = True
                if getattr(args, "image_max_edge", None) is not None:
                    kwargs["image_max_edge"] = args.image_max_edge
                if getattr(args, 'sliding_window_tokens', None) is not None:
                    kwargs['sliding_window_tokens'] = args.sliding_window_tokens
    if (backend == 'openai_chat_completions'
            and kwargs.get('base_url', '').rstrip('/') == 'https://ai-gateway.vercel.sh/v1'):
        from wia.runtime.agent.vercel_session import VercelChatSession
        return VercelChatSession(history_policy=getattr(args, 'history_policy', None) or
            os.environ.get('WIA_GATEWAY_HISTORY_POLICY') or None,
            history_recent_moves=getattr(args, 'history_recent_moves', None),
            max_perception_rounds=getattr(args, 'max_perception_rounds', None), **kwargs)
    return create_session(backend, **kwargs)


def declarative_resume_budget(summary, manifest, requested=None, *, extend=False):
    """Keep the original manifest while recording an explicitly extended runner budget."""
    if summary.get("episode_manifest") != manifest:
        raise ValueError("Resume must retain the exact original episode manifest.")
    if (summary["agent"]["model"] != manifest["model_name"]
            or summary["agent"]["reasoning_effort"] != manifest["reasoning_effort"]
            or summary["max_duration_s"] != manifest["wall_timeout_s"]
            or summary["action_mode"] != manifest["action_mode"]
            or summary["world_mode"] != manifest["world_mode"]
            or summary["observation_condition"] != manifest["observation_condition"]):
        raise ValueError("Resume must retain the original model and episode policy.")
    budget = summary["max_steps"] if requested is None else requested
    if (extend and budget <= summary["max_steps"]
            or not extend and budget != summary["max_steps"]):
        raise ValueError("A larger resume budget requires --extend-max-steps.")
    return budget


def finalize_evaluation_summary(summary, latest_progress, run_mode):
    """Keep measured progress separate from an interrupted episode's quality result."""
    summary['run_mode'] = run_mode
    interrupted = summary['termination_reason'] in ('runtime_error', 'model_error', 'execution_error', 'observation_error', 'environment_error', 'cancelled')
    summary['execution_validity'] = 'interrupted' if interrupted else 'completed'
    summary['agent_failure_countable'] = False
    if latest_progress:
        summary['task_progress'] = latest_progress
        summary.update({key: latest_progress.get(key) for key in (
            'evaluation_validity', 'scoring_version', 'completion_score', 'score_percent',
            'constraints_satisfied', 'safety_score', 'safety_validity', 'safety_exposure')})
        summary['agent_failure_countable'] = (
            latest_progress.get('evaluation_validity') == 'complete' and not interrupted)
        summary['task_progress_is_final'] = not interrupted
    if interrupted or latest_progress and not summary['agent_failure_countable']:
        summary['task_success'] = None
    if interrupted:
        summary['completion_score'] = summary['score_percent'] = None


def validate_wia_history_condition(args, *, loaded_task=None, manifest=None):
    """Limit explicit Phase-2 history interventions to the supported native path."""
    condition = args.wia_history_condition
    if condition != 'history_oracle' and args.wia_history_oracle_file is not None:
        raise ValueError('--wia-history-oracle-file is allowed only for history_oracle.')
    if condition == 'full':
        return
    if (args.task_file is None or canonical_backend(args.backend) != 'codex_cli' or args.memory_mode != 'none'
            or args.observation_condition != 'current' or not args.world_observation_log
            or args.resume_from is not None or args.episode_manifest is None):
        raise ValueError('Diagnostic history conditions require a declarative Codex task, an episode manifest, '
                         'current observations, memory-mode none, world-observation output and no resume.')
    if condition == 'history_oracle' and args.wia_history_oracle_file is None:
        raise ValueError('history_oracle requires --wia-history-oracle-file with explicit nonempty UTF-8 facts.')
    if loaded_task is not None or manifest is not None:
        evaluation = getattr(getattr(getattr(loaded_task, 'definition', None), 'private', None), 'evaluation', None)
        parameters = getattr(evaluation, 'parameters', {})
        if not manifest or not manifest.get('temporal_policy') or parameters.get('task_edit2_physical') is not True:
            raise ValueError('Diagnostic history conditions require a temporal manifest policy and a physical temporal task.')


def read_wia_history_oracle_file(path):
    """Read explicit facts verbatim; never derive an answer from task/evaluator state."""
    facts = Path(path).read_text(encoding='utf-8')
    if not facts.strip():
        raise ValueError('The explicit oracle file must contain nonempty UTF-8 historical facts.')
    return facts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode", choices=("benchmark", "diagnostic"), default="benchmark")
    parser.add_argument("--url", default="http://127.0.0.1:8615")
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--task", help="Legacy public task text.")
    tasks.add_argument("--task-file", type=Path, help="Validated declarative physical task YAML.")
    tasks.add_argument("--scene-file", type=Path, help="Scene-only YAML; answer its original question in one native session.")
    parser.add_argument("--variant", default="original")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_backend_arguments(parser)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--world-observation-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the optional world-observation prompt plugin for native declarative tasks (default: on).")
    parser.add_argument("--debug-reachability", action="store_true",
                        help="Ask for affected public goals/items, blockage reasons and evidence inside world_observation.")
    parser.add_argument("--reasoning-summary", choices=("none", "concise"), default="concise",
                        help="Request a brief public summary when supported; raw reasoning is never saved.")
    parser.add_argument('--wia-history-condition', choices=('full', 'history_removed', 'history_oracle'), default='full',
                        help='Explicit diagnostic Phase-2 history condition; full preserves the standard native session.')
    parser.add_argument('--wia-history-oracle-file', type=Path,
                        help='Verbatim nonempty UTF-8 historical facts, required only for history_oracle.')
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Decision budget: 60 for scene QA; 8 for legacy text tasks; declarative tasks use their manifest.")
    parser.add_argument("--qa-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--qa-goal", choices=("answer", "answer_and_place"), default="answer",
                        help="Keep original QA only, or also physically pick up and place the target in the selected container.")
    parser.add_argument("--qa-pickup-location", help="Public semantic pickup location for a combined retrieval episode.")
    parser.add_argument("--resume-from", type=Path,
                        help="Previous native task or scene-QA directory interrupted by a model error or cancellation; resume its thread and remaining budget.")
    parser.add_argument("--extend-max-steps", action="store_true",
                        help="Explicitly authorize a larger total decision budget when resuming after interruption.")
    parser.add_argument("--action-mode", choices=("fixed", "parameterized"), default="fixed")
    parser.add_argument("--memory-mode", choices=("none", "model_notes"), default="none")
    parser.add_argument("--world-during-thinking", choices=("run", "pause"), default="run")
    parser.add_argument("--observation-condition", choices=("current", "initial_view_only"), default="current")
    parser.add_argument("--episode-timeout", type=float, default=None)
    parser.add_argument("--evaluation-file", type=Path,
                        help="Private native state.json for independent success; never sent to the model.")
    parser.add_argument("--max-invalid-retries", type=int, default=2)
    parser.add_argument("--max-recoverable-failures", type=int, default=None,
                        help="Optional consecutive recovery cap; by default stopped limits use the overall step budget. Zero disables recovery.")
    parser.add_argument("--model-timeout", type=optional_timeout, default=argparse.SUPPRESS)
    parser.add_argument("--snapshot-cache", choices=("legacy", "optimized"), default="optimized")
    parser.add_argument("--codex-executable", default="codex")
    args = parser.parse_args(argv)
    args.model_timeout = getattr(args, "model_timeout", None if args.backend in ("codex", "codex_cli") else 120)
    try:
        validate_backend_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.reasoning_effort is None and canonical_backend(args.backend) == "codex_cli":
        args.reasoning_effort = "medium"
    if canonical_backend(args.backend) in ("openai_chat_completions", "openai_responses") and not any(
            argument == "--model" or argument.startswith("--model=") for argument in (sys.argv[1:] if argv is None else argv)):
        parser.error("Explicit API backends require an explicit --model.")
    if canonical_backend(args.backend) in ("openai_chat_completions", "openai_responses") and args.task_file is None:
        parser.error("Explicit API backends currently require --task-file; legacy text and QA retain their existing transports.")
    if args.resume_from is not None and canonical_backend(args.backend) != "codex_cli":
        parser.error("HTTP episode resume is not implemented; use a fresh scene.")
    if args.debug_reachability and not args.world_observation_log:
        parser.error("--debug-reachability requires --world-observation-log.")
    try:
        validate_wia_history_condition(args)
    except ValueError as error:
        parser.error(str(error))
    if args.max_steps is None:
        from wia.tasks.rigid_qa import DEFAULT_QA_STEPS
        args.max_steps = DEFAULT_QA_STEPS if args.scene_file is not None else 8
    if args.qa_goal != "answer" and args.scene_file is None:
        parser.error("--qa-goal answer_and_place requires --scene-file.")
    if args.qa_pickup_location is not None and (args.qa_goal != "answer_and_place" or not args.qa_pickup_location.strip()):
        parser.error("--qa-pickup-location requires --qa-goal answer_and_place and a nonempty location.")
    if args.extend_max_steps and args.resume_from is None:
        parser.error("--extend-max-steps requires --resume-from and an explicitly larger --max-steps.")
    explicit = {argument.split("=", 1)[0] for argument in (sys.argv[1:] if argv is None else argv)
                if argument.startswith("--")}
    loaded_task = loaded_scene = manifest = manifest_hash = contract = None
    resume_summary = resume_scene = None
    expected_action_step = None
    episode_id = uuid.uuid4().hex
    if args.resume_from is not None:
        if (args.scene_file is None and args.task_file is None) or args.observation_condition != "current":
            parser.error("Native interruption resume requires --scene-file or --task-file and current observations.")
        args.resume_from = args.resume_from.expanduser().resolve()
        resume_summary = json.loads((args.resume_from / "summary.json").read_text())
        if args.scene_file is not None:
            resume_scene = json.loads((args.resume_from / "qa_scene.json").read_text())
        allowed_reasons = ("model_error", "cancelled")
        if resume_summary["termination_reason"] not in allowed_reasons:
            parser.error("Resume is only available after a native model error or cancellation.")
        if args.extend_max_steps and args.max_steps <= resume_summary["max_steps"]:
            parser.error("An explicit budget extension must increase the previous total --max-steps.")
        episode_id = (resume_scene if resume_scene is not None else resume_summary["episode_manifest"])["episode_id"]
        last_step = resume_summary["agent_decisions"] + 1
        last_observation = json.loads((args.resume_from / f"step_{last_step:04d}/before/observation.json").read_text())
        expected_action_step = last_observation["action_step"]
    if args.scene_file is not None:
        from wia.tasks.loader import load_scene
        if (canonical_backend(args.backend) != "codex_cli" or args.memory_mode != "none" or not args.world_observation_log
                or args.episode_manifest is not None or args.evaluation_file is not None):
            parser.error("Scene-only QA requires native Codex, memory-mode none, world-observation output and no physical evaluator/manifest.")
        loaded_scene = load_scene(args.scene_file, data_root=args.data_root,
                                  variant=args.variant, seed=args.seed, run_mode=args.run_mode)
        args.task = loaded_scene.task_text
    if args.task_file is not None:
        from wia.tasks.loader import load_task, verify_manifest, public_episode_contract, sha256
        if args.memory_mode != "none" or args.episode_manifest is None:
            parser.error("Declarative tasks require memory-mode none and the immutable episode manifest.")
        loaded_task = load_task(args.task_file, data_root=args.data_root, variant=args.variant, seed=args.seed, run_mode=args.run_mode)
        manifest = verify_manifest(args.episode_manifest, loaded_task)
        expected_session = session_policy(args.backend)
        if manifest.get('backend_metadata') is not None and manifest['backend_metadata'] != backend_metadata(args):
            parser.error('Backend API settings must match the immutable episode manifest.')
        if manifest['agent_session'] != expected_session:
            parser.error('The backend must match the immutable episode session policy.')
        if args.backend == 'local_http' and args.resume_from is not None:
            parser.error('Local HTTP episode resume is not implemented; use a fresh scene.')
        manifest_hash = sha256(args.episode_manifest)
        enabled = manifest.get("world_observation_log", True)
        if ({"--world-observation-log", "--no-world-observation-log"} & explicit
                and args.world_observation_log != enabled):
            parser.error("The world-observation plugin must match the immutable episode manifest.")
        args.world_observation_log = enabled
        contract = public_episode_contract(manifest, manifest_hash)
        task_budget = manifest["max_steps"]
        if resume_summary is not None:
            try:
                task_budget = declarative_resume_budget(resume_summary, manifest,
                    args.max_steps if "--max-steps" in explicit else None, extend=args.extend_max_steps)
            except ValueError as error:
                parser.error(str(error))
        parameters = {"--max-steps": (args.max_steps, "max_steps"),
            "--action-mode": (args.action_mode, "action_mode"),
            "--world-during-thinking": (args.world_during_thinking, "world_mode"),
            "--observation-condition": (args.observation_condition, "observation_condition"),
            "--model": (args.model, "model_name"), "--reasoning-effort": (args.reasoning_effort, "reasoning_effort"),
            "--episode-timeout": (args.episode_timeout, "wall_timeout_s")}
        if any(flag in explicit and value != (task_budget if key == "max_steps" else manifest[key])
               for flag, (value, key) in parameters.items()):
            parser.error("Explicit Agent parameters must match the immutable episode manifest.")
        args.task = loaded_task.task_text
        args.max_steps, args.action_mode = task_budget, manifest["action_mode"]
        args.world_during_thinking, args.observation_condition = manifest["world_mode"], manifest["observation_condition"]
        args.model, args.reasoning_effort = manifest["model_name"], manifest["reasoning_effort"]
        args.episode_timeout = manifest["wall_timeout_s"]
        try:
            validate_wia_history_condition(args, loaded_task=loaded_task, manifest=manifest)
        except ValueError as error:
            parser.error(str(error))
    if (not args.task.strip() or args.max_steps < 1 or args.max_invalid_retries < 0
            or args.max_recoverable_failures is not None and args.max_recoverable_failures < 0 or (args.model_timeout is not None and args.model_timeout <= 0)):
        parser.error("Provide a task, positive step/timeout limits and nonnegative retries.")
    if args.model_timeout is None and args.backend not in ("codex", "codex_cli"):
        parser.error("Unbounded model waits currently require the Codex backend.")
    if args.episode_timeout is not None and (not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0):
        parser.error("Episode timeout must be a finite positive duration.")
    if args.backend == "local_http" and (not (args.base_url or args.local_base_url) or "--model" not in explicit):
        parser.error("Local agents require --local-base-url and an explicit local --model.")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new or empty output directory.")
    oracle_facts = None
    if args.wia_history_condition == 'history_oracle':
        try:
            oracle_facts = read_wia_history_oracle_file(args.wia_history_oracle_file)
        except (OSError, UnicodeError, ValueError) as error:
            parser.error(str(error))
    if loaded_task is not None:
        from core.agent.codex_session import CodexEpisodeSession
        from wia.runtime.agent.prompts import PromptAssembler
        policy = {key: manifest[key] for key in ("episode_id", "max_steps", "world_mode",
            "world_advances_during_inference", "observation_condition", "action_mode")}
        policy["max_steps"] = args.max_steps
        policy["run_mode"] = args.run_mode
        if manifest.get("temporal_policy"):
            policy["temporal_policy"] = manifest["temporal_policy"]
        agent = create_declarative_session(args, public_task=loaded_task.public_payload, policy=policy,
            output=output, assembler=PromptAssembler(world_observation=args.world_observation_log,
                                                    debug_reachability=args.debug_reachability),
            resume_summary=resume_summary)
        if args.wia_history_condition != 'full':
            from wia.runtime.agent.wia_history_session import WiaHistorySession
            def phase2_factory():
                return CodexEpisodeSession(public_task=loaded_task.public_payload, episode_configuration=policy,
                    output=output / 'phase2_native', model=args.model, reasoning_effort=args.reasoning_effort,
                    reasoning_summary=args.reasoning_summary, executable=args.codex_executable,
                    timeout_s=args.model_timeout,
                    assembler=PromptAssembler(wia_history_condition=args.wia_history_condition,
                                               world_observation=args.world_observation_log,
                                               debug_reachability=args.debug_reachability),
                    snapshot_cache=args.snapshot_cache == 'optimized', resume_state=None)
            agent = WiaHistorySession(phase1_session=agent, phase2_factory=phase2_factory,
                condition=args.wia_history_condition, output=output, oracle_facts=oracle_facts)
        if resume_summary is not None:
            output.mkdir(parents=True, exist_ok=True)
            (output / "resume.json").write_text(json.dumps({"resume_from": str(args.resume_from),
                "episode_id": manifest["episode_id"], "original_manifest_sha256": manifest_hash,
                "previous_action_step": expected_action_step, "scene_reset": False,
                "previous_max_steps": resume_summary["max_steps"], "max_steps": args.max_steps,
                "budget_extension": args.extend_max_steps}, indent=2) + "\n")
    elif loaded_scene is not None:
        from core.agent.codex_session import CodexEpisodeSession
        from wia.runtime.agent.prompts import PromptAssembler
        from wia.tasks.rigid_qa import RigidContainmentQA
        if not 0 <= args.qa_confidence_threshold <= 1:
            parser.error("QA confidence threshold must be between zero and one.")
        qa_protocol = RigidContainmentQA(loaded_scene, threshold=args.qa_confidence_threshold,
                                        goal=args.qa_goal, pickup_location=args.qa_pickup_location)
        policy = {"episode_id": episode_id, "max_steps": args.max_steps,
                  "world_mode": args.world_during_thinking,
                  "world_advances_during_inference": args.world_during_thinking == "run",
                  "observation_condition": args.observation_condition, "action_mode": args.action_mode}
        expected_policy = dict(resume_scene["episode_configuration"]) if resume_scene is not None else policy
        if args.extend_max_steps:
            expected_policy["max_steps"] = args.max_steps
        if resume_scene is not None and (
                resume_scene["scene_file_sha256"] != loaded_scene.file_sha256
                or resume_scene["source_sha256"] != loaded_scene.source_sha256
                or expected_policy != policy
                or resume_scene["public_task"] != qa_protocol.public_payload):
            parser.error("Resume must retain the original scene, source, public task and episode policy.")
        agent = CodexEpisodeSession(public_task=qa_protocol.public_payload, episode_configuration=policy,
            output=output, model=args.model, reasoning_effort=args.reasoning_effort,
            reasoning_summary=args.reasoning_summary, executable=args.codex_executable, timeout_s=args.model_timeout,
            assembler=PromptAssembler(task_mode="original_qa", qa_protocol=qa_protocol,
                                      debug_reachability=args.debug_reachability), snapshot_cache=args.snapshot_cache == "optimized",
            resume_state=resume_summary["native_session"] if resume_summary else None)
        output.mkdir(parents=True, exist_ok=True)
        (output / "qa_scene.json").write_text(json.dumps({"schema": "scene_only_qa_v1",
            "episode_id": episode_id, "scene_file": str(loaded_scene.path),
            "scene_file_sha256": loaded_scene.file_sha256, "source_path": str(loaded_scene.source_path),
            "source_sha256": loaded_scene.source_sha256, "scene": loaded_scene.scene_name,
            "room": loaded_scene.room_name, "variant": loaded_scene.variant, "seed": loaded_scene.seed,
            "original_question": args.task, "public_task": qa_protocol.public_payload, "qa_protocol": qa_protocol.provenance(),
            "episode_configuration": policy, "physical_evaluator": None, "scene_reset": False,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "budget_extension": {"previous_max_steps": resume_summary["max_steps"],
                "max_steps": args.max_steps, "reason": "user_requested_extension"} if args.extend_max_steps else None}, indent=2) + "\n")
    elif canonical_backend(args.backend) == "codex_cli":
        from core.agent.backends import CodexAgent
        agent = CodexAgent(model=args.model, reasoning_effort=args.reasoning_effort,
                           executable=args.codex_executable, timeout_s=args.model_timeout,
                           reasoning_summary=args.reasoning_summary)
        agent.snapshot_cache = args.snapshot_cache == "optimized"
    else:
        from core.agent.backends import LocalHTTPAgent
        agent = LocalHTTPAgent(base_url=args.base_url or args.local_base_url, model=args.model,
                               api_key=os.environ.get(args.local_api_key_env, ""), timeout_s=args.model_timeout,
                               structured_output=not args.no_local_structured_output)
        agent.snapshot_cache = args.snapshot_cache == "optimized"

    global MCPRobotEnvironment, run_episode
    if MCPRobotEnvironment is None:
        from wia.runtime.agent.mcp_environment import MCPRobotEnvironment
    if run_episode is None:
        from wia.runtime.agent.zero_shot import run_episode
    latest_progress = {}

    async def run():
        async with MCPRobotEnvironment(args.url, run_id=manifest["episode_id"] if manifest else episode_id, task_text=args.task,
                                       action_mode=args.action_mode,
                                       world_advances_during_inference=args.world_during_thinking == "run",
                                       episode_contract=contract, resume_action_step=expected_action_step) as environment:
            def success():
                data = json.loads(args.evaluation_file.read_text())
                observation = environment.last_observation
                from wia.native.esi_category_evaluation import validate_episode_evaluation
                measured_success = validate_episode_evaluation(data, observation, args.task, loaded_task)
                latest_progress.clear()
                latest_progress.update(data.get('progress', {}))
                if 'safety' in data:
                    safety = data['safety']
                    latest_progress.update(safety_score=safety.get('safety_score'),
                        safety_validity='measured' if safety.get('safety_score') is not None else 'unavailable',
                        safety_exposure=safety.get('exposure'))
                (output / "evaluation.json").write_text(json.dumps(data, indent=2) + "\n")
                return measured_success
            def integrity():
                if loaded_scene is not None:
                    loaded_scene.verify_unchanged()
                    qa_protocol.verify_unchanged()
                    return
                loaded_task.verify_unchanged()
                if sha256(args.episode_manifest) != manifest_hash:
                    raise RuntimeError("Episode manifest changed after initialization.")
            return await run_episode(environment, agent, output, max_steps=args.max_steps,
                                     max_invalid_retries=args.max_invalid_retries,
                                     max_recoverable_failures=args.max_recoverable_failures,
                                     action_mode=args.action_mode,
                                     memory_mode=args.memory_mode, max_duration_s=args.episode_timeout,
                                     observation_condition=args.observation_condition,
                                     episode_manifest=manifest,
                                     resume_summary=resume_summary,
                                     expected_action_step=(expected_action_step if loaded_task is not None else 0)
                                     if resume_summary else None,
                                     budget_extension=args.extend_max_steps,
                                     integrity_check=integrity if loaded_task is not None or loaded_scene is not None else None,
                                     success_provider=success if args.evaluation_file is not None else None)

    try:
        summary = asyncio.run(run())
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    finalize_evaluation_summary(summary, latest_progress, args.run_mode)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({"termination_reason": summary["termination_reason"],
                      "executed_actions": summary["executed_actions"], "task_success": summary["task_success"],
                      "agent_decisions": summary["agent_decisions"], "model_calls": summary["model_calls"],
                      "macro_actions": summary["macro_actions"], "primitive_actions": summary["primitive_actions"],
                      "model_logs": summary["model_logs"],
                      "task_progress": summary.get('task_progress'),
                      "summary": str(output / "summary.json")}, indent=2))
    return 0 if summary["termination_reason"] in ("model_done", "max_steps", "environment_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a fresh local native scene in a persistent codex-gpu window and one Agent.

Run with the behavior environment: python src/wia/runtime/commands/run_esi_explore_episode.py
--task-file TASK --run-mode diagnostic --gpu N --output-dir OUTPUT.
Idle checks are enabled unless the caller selects WIA_SKIP_GPU_CHECK=1.
Reusable shells remain alive.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
import uuid

import requests

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))


def finalize_task_record(output, manifest, status):
    """Collect private provenance after cleanup; never use this as a model input."""
    def read(relative, fallback):
        path = output / relative
        return json.loads(path.read_text()) if path.exists() else fallback
    qualification = read("private_scene/qualification.json", {})
    episode = status.get("active_episode_dir", "episode")
    summary = read(f"{episode}/summary.json", {})
    native = read(f"{episode}/native_session.json", {})
    steps_path = output / episode / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()] if steps_path.exists() else []
    record = dict(manifest)
    record.update(manifest_schema=record.pop("schema"), schema="declarative_episode_record_v1",
        applied_scene_patches=qualification.get("applied_scene_patches", []),
        qualification_status=qualification.get("status"), qualification_reason=qualification.get("reason"),
        **native, action_schema=summary.get("action_schema"),
        provider_output_schema=summary.get("provider_output_schema"),
        enabled_robot_actions=summary.get("available_actions", []),
        authorized_model_tools=[tool["name"] for tool in (native.get("tool_permission_audit") or {}).get("initial_tools", [])],
        episode_start_prompt=f"{episode}/episode_start_prompt.txt" if (output / episode / "episode_start_prompt.txt").exists() else None,
        decision_step_prompts=[str(path.relative_to(output)) for path in sorted((output / episode / "decision_step_prompts").glob("*.txt"))],
        action_history=summary.get("executed_actions", []),
        execution_feedback=[step["action_result"] for step in steps if step.get("action_result") is not None],
        termination_reason=summary.get("termination_reason", "skipped" if qualification.get("status") == "skipped" else status["stage"]),
        private_task_success=summary.get("task_success"),
        agent_decisions=summary.get("agent_decisions"), decision_calls=summary.get("decision_calls"),
        model_calls=summary.get("model_calls"), request_accounting=summary.get("request_accounting"),
        safety_statistics=summary.get("safety_statistics"),
        task_progress=summary.get("task_progress"),
        evaluation_validity=summary.get("evaluation_validity", "unavailable"),
        scoring_version=summary.get("scoring_version"),
        completion_score=summary.get("completion_score"), score_percent=summary.get("score_percent"),
        agent_failure_countable=summary.get("agent_failure_countable", False),
        temporal_policy=summary.get("temporal_policy"), temporal_transitions=summary.get("temporal_transitions", []),
        phase_1_decisions=summary.get("phase_1_decisions"),
        observation_coverage=summary.get("observation_coverage", []),
        runtime_status=status, timings=summary.get("timings"),
        private_evaluation_reference="final_evaluation.json" if (output / "final_evaluation.json").exists() else None)
    if manifest.get('physical_evaluator') == 'not_configured':
        # The protected Rigid QA protocol has no category physical score.
        for key in ('task_progress', 'evaluation_validity', 'scoring_version',
                    'completion_score', 'score_percent', 'agent_failure_countable'):
            record.pop(key, None)
    (output / "episode_record.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def episode_timeout(value):
    """Use 'none' to disable only the overall episode deadline."""
    return None if value.lower() == "none" else float(value)


def query_local_gpus():
    """Read identities only when the launch script disables occupancy checks."""
    skip = os.environ.get('WIA_SKIP_GPU_CHECK') == '1'
    fields = 'index,uuid' if skip else 'index,uuid,utilization.gpu,memory.used'
    query = subprocess.run(['nvidia-smi', '--query-gpu=' + fields,
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    devices = [row.split(',') for row in query.stdout.strip().splitlines() if row.strip()]
    eligible = [int(row[0]) for row in devices if skip or int(row[2]) < 5 and int(row[3]) < 512]
    return query, devices, eligible, skip


def read_native_qualification(output):
    """Read either native producer: DeclarativeScene or rigid-task SceneOnly."""
    for name in ("task_status.json", "scene_status.json"):
        path = output / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def verify_human_preview(url, output, *, timeout_s=30.):
    """Verify live human frames before acquiring any Agent control lease."""
    deadline = time.monotonic() + timeout_s
    first = None
    while time.monotonic() < deadline:
        state = requests.get(url + "/api/agent/state", timeout=5).json()
        response = requests.get(url + "/api/camera-group/robot", timeout=5)
        response.raise_for_status()
        group = response.json()
        if not group.get("images") or not {"head", "wrist"} <= set(group["images"]):
            time.sleep(.2)
            continue
        current = {"frame_seq": group["frame_seq"], "frame_time": group["frame_time"],
                   "views": sorted(group["images"]),
                   "canonical_snapshot_id": state["observation_snapshot_id"],
                   "canonical_action_step": state["steps"]}
        if first is None:
            first = current
        elif current["frame_seq"] != first["frame_seq"] and current["frame_time"] != first["frame_time"]:
            if (current["canonical_snapshot_id"] != first["canonical_snapshot_id"]
                    or current["canonical_action_step"] != first["canonical_action_step"]):
                raise RuntimeError("Human preview unexpectedly changed the canonical observation boundary.")
            import base64
            import io
            from PIL import Image
            image_sizes = {}
            for name, encoded in group["images"].items():
                with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                    if image.format != "JPEG":
                        raise RuntimeError("GUMI preview did not return a JPEG camera image.")
                    image_sizes[name] = list(image.size)
                    image.verify()
            result = {"status": "passed", "first": first, "second": current,
                      "jpeg_image_sizes": image_sizes,
                      "verification_scope": "live cached robot camera group without an Agent lease or action"}
            (output / "gumi_preview_verification.json").write_text(json.dumps(result, indent=2) + "\n")
            return result
        time.sleep(.2)
    raise RuntimeError("GUMI live preview did not publish two fresh robot camera frames.")


def retain_completed_simulator(url, output, *, requested, rollout_completed):
    """Keep GUMI only after the Agent ended and its robot lease was released."""
    if not requested or not rollout_completed or (output / "native_exit_code.txt").exists():
        return False
    try:
        response = requests.get(url + "/api/agent/state", timeout=5)
        response.raise_for_status()
        state = response.json()
        return state.get("control_owner") == "human" and state.get("busy") is False
    except (requests.RequestException, ValueError):
        return False


def build_episode_manifest(task, args, *, episode_id=None):
    """Build exactly the immutable manifest consumed by both native and agent runners."""
    from wia.runtime.commands.run_esi_agent import session_policy, backend_metadata
    manifest = task.manifest(episode_id or uuid.uuid4().hex)
    manifest.update(max_steps=args.max_steps, world_mode=args.world,
        world_advances_during_inference=args.world == "run", observation_condition=args.observation_condition,
        action_mode=args.action_mode, model_name=args.model, reasoning_effort=args.reasoning_effort,
        wall_timeout_s=args.timeout, agent_session=session_policy(args.backend),
        backend_metadata=backend_metadata(args), world_observation_log=args.world_observation_log)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--config", type=Path, help="Legacy exploration JSON configuration.")
    tasks.add_argument("--task-file", type=Path, help="Declarative ESI task file or unique ID.")
    parser.add_argument("--gpu", type=int, help="Explicit idle local GPU; auto-select an idle GPU when omitted.")
    parser.add_argument("--port", type=int, default=8615)
    parser.add_argument("--python", default=sys.executable, help="Python in the native simulator environment.")
    parser.add_argument("--execution-mode", choices=("debug", "batch"),
                        help="Debug blocks with a retained scene; batch cleans up and advances.")
    parser.add_argument("--run-mode", choices=("benchmark", "diagnostic"), default="benchmark")
    parser.add_argument("--variant", help="Defaults to the task's declared initialization layout.")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world", choices=("run", "pause"))
    parser.add_argument("--memory", choices=("none", "model_notes"), default="none")
    parser.add_argument("--observation-condition", choices=("current", "initial_view_only"))
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="-1 uses the task default; a positive value overrides it.")
    parser.add_argument("--timeout", type=episode_timeout, default=argparse.SUPPRESS,
                        help="Overall episode seconds, or 'none' for no episode deadline.")
    parser.add_argument("--action-mode", choices=("fixed", "parameterized"))
    parser.add_argument("--macro-execution", choices=("metric_target", "primitive_replay"), default="metric_target")
    from wia.runtime.commands.run_esi_agent import (add_backend_arguments, validate_backend_arguments,
                                      backend_cli_arguments, backend_metadata, canonical_backend)
    add_backend_arguments(parser)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--world-observation-log", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable the optional native world-observation prompt plugin (default: on).")
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--model-timeout", type=episode_timeout, default=argparse.SUPPRESS)
    parser.add_argument("--max-invalid-retries", type=int, default=2)
    parser.add_argument("--max-recoverable-failures", type=int)
    parser.add_argument("--snapshot-cache", choices=("legacy", "optimized"), default="optimized")
    recording_options = parser.add_mutually_exclusive_group()
    recording_options.add_argument("--record", action="store_true", default=None,
                                   help="Record videos (default: on in diagnostic, off in benchmark).")
    recording_options.add_argument("--norecord", "--no-record", dest="record", action="store_false",
                                   help="Disable video recording.")
    parser.add_argument("--recording-hz", type=float, default=5.0,
                        help="Video captures per simulation second while recording (default: 5 Hz).")
    parser.add_argument("--preview-fps", type=float, default=0.,
                        help="Human-only live camera rate; zero disables continuous preview.")
    parser.add_argument("--debug-cameras", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the ESI and Global human camera groups (enabled by default).")
    parser.add_argument("--keep-simulator", action="store_true",
                        help="Keep GUMI and the completed native scene after the Agent releases control.")
    parser.add_argument('--initialize-only', action='store_true',
                        help='Load and qualify without a model; debug retains the scene and batch clears it.')
    args = parser.parse_args(argv)
    args.model_timeout = getattr(args, "model_timeout", None if args.backend in ("codex", "codex_cli") else 180)
    args.execution_mode = args.execution_mode or ("debug" if args.run_mode == "diagnostic" else "batch")
    if args.execution_mode == "batch" and args.keep_simulator:
        parser.error("Batch execution cannot keep a simulator; use --execution-mode debug.")
    if args.record is None:
        args.record = args.execution_mode == "debug"
    import math
    if not math.isfinite(args.recording_hz) or args.recording_hz <= 0:
        parser.error("Recording frequency must be finite and positive.")
    try:
        validate_backend_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.reasoning_effort is None and canonical_backend(args.backend) == "codex_cli":
        args.reasoning_effort = "high"
    if canonical_backend(args.backend) in ("openai_chat_completions", "openai_responses") and not any(
            argument == "--model" or argument.startswith("--model=") for argument in (sys.argv[1:] if argv is None else argv)):
        parser.error("Explicit API backends require an explicit --model.")
    if args.config is not None and canonical_backend(args.backend) not in ("codex_cli", "local_http"):
        parser.error("Explicit API backends require a declarative --task-file.")
    task = episode_manifest = None
    physical_evaluator = True
    if args.task_file is not None:
        from wia.tasks.loader import LoadedRigidTask, load_task
        if args.memory != "none":
            parser.error("The declarative native-session pathway requires --memory none.")
        task = load_task(args.task_file, data_root=args.data_root, variant=args.variant, seed=args.seed, run_mode=args.run_mode)
        physical_evaluator = not isinstance(task, LoadedRigidTask)
        args.variant = task.variant
        defaults = task.definition.defaults
        args.seed = task.seed
        args.world = args.world or ("run" if defaults.world_advances_during_inference else "pause")
        args.observation_condition = args.observation_condition or defaults.observation_condition
        args.max_steps = defaults.max_steps if args.max_steps == -1 else args.max_steps
        args.timeout = getattr(args, "timeout", None)
        args.action_mode = args.action_mode or defaults.action_mode
        episode_manifest = build_episode_manifest(task, args)
    else:
        if args.seed is None:
            parser.error("Legacy configs require an explicit --seed.")
        args.world = args.world or "run"
        args.observation_condition = args.observation_condition or "current"
        args.max_steps = 60 if args.max_steps == -1 else args.max_steps
        args.timeout = getattr(args, "timeout", None)
        args.action_mode = args.action_mode or "parameterized"
    import math
    if not math.isfinite(args.preview_fps) or args.preview_fps < 0:
        parser.error("Preview FPS must be finite and nonnegative.")
    if args.max_steps < 1 or args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("Episode steps must be positive; timeout must be positive and finite or 'none'.")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory.")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text()) if task is None else None
    if task is None:
        (output / "scene.json").write_text(json.dumps(config, indent=2) + "\n")
    else:
        (output / "task.yaml").write_bytes(task.path.read_bytes())
        (output / "episode_manifest.json").write_text(json.dumps(episode_manifest, indent=2) + "\n")
    state_path = output / 'local_runtime.json'
    query, devices, idle, skip_gpu_check = query_local_gpus()
    selected_gpu = args.gpu if args.gpu is not None else (idle[0] if idle else None)
    resident_control = os.environ.get('WIA_RESIDENT_CONTROL_DIR')
    owned_resident = False
    if resident_control and selected_gpu is not None and not skip_gpu_check:
        from wia.runtime.resident_episode import owns_gpu
        chosen = next(row for row in devices if int(row[0]) == selected_gpu)
        owned_resident = owns_gpu(resident_control, selected_gpu, chosen[1].strip())
    if selected_gpu not in idle and not owned_resident:
        if skip_gpu_check:
            raise RuntimeError('The selected local GPU index is absent from the device inventory.')
        raise RuntimeError('No selected idle local GPU is available; existing workloads will not be displaced.')
    device = next(row for row in devices if int(row[0]) == selected_gpu)
    node, job = '127.0.0.1', None
    url = f'http://127.0.0.1:{args.port}'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        gateway_port = listener.getsockname()[1]
    agent_url = f'http://127.0.0.1:{gateway_port}'
    gpu = {'execution_mode': 'local', 'CUDA_VISIBLE_DEVICES': device[1].strip(),
        'gpu_idle_check': not skip_gpu_check,
        'uuid': device[1].strip(), 'renderer_index': selected_gpu, 'physical_index': selected_gpu,
        'device_uuid': device[1].strip(), 'service': {}, 'observed_gpu_query': query.stdout}
    if resident_control:
        gpu.update(persistent_simulator=True, resident_control_dir=resident_control)
    state_path.write_text(json.dumps(gpu, indent=2) + '\n')
    if not resident_control:
        subprocess.run(['tmux', 'new-session', '-d', '-s', 'codex-gpu'], capture_output=True)
    target = ('resident:' + resident_control if resident_control else subprocess.check_output(
        ['tmux', 'new-window', '-d', '-P', '-F', '#{session_name}:#{window_id}',
         '-t', 'codex-gpu', '-n', 'repair-' + output.name[-24:]], text=True).strip())
    scene_args = (["--explore-config", str(output/"scene.json")] if task is None else
                  ["--task-file", str(task.path), "--variant", args.variant, "--episode-manifest", str(output/"episode_manifest.json")])
    if args.data_root is not None:
        scene_args += ["--data-root", str(args.data_root.resolve())]
    native = [args.python, "-u", str(ROOT/"src/wia/runtime/commands/run_esi_robot.py"), *scene_args,
        "--gpu-info", str(state_path), "--run-mode", args.run_mode, "--scene-seed", str(args.seed), "--gripper-control", "--rotation-control", "--trunk-control",
        "--motion-diagnostics", "--web", "--host", "0.0.0.0", "--port", str(args.port), "--parameterized-actions",
        "--primitive-profile", "fine", "--performance-profile", "benchmark", "--macro-execution", args.macro_execution,
        "--observation-pipeline", "optimized", "--preview-fps", str(args.preview_fps), "--output-dir", str(output)]
    native += ["--recording-hz", str(args.recording_hz)]
    if args.execution_mode == "debug":
        native += ["--debug-retain"]
    if not args.record:
        native += ["--no-human-recording"]
    if args.preview_fps > 0:
        native += ["--human-preview"]
    if args.debug_cameras and (args.preview_fps > 0 or args.record):
        native += ["--debug-cameras"]
    if args.record:
        native += ["--human-recording", "--recording-mode", "snapshots", "--recording-views", "all"]
    cache_key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:12]
    cache = Path('/tmp') / ('show-harness-' + str(os.getuid())) / ('native-' + output.name + '-' + cache_key)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'native_tmp').mkdir(exist_ok=True)
    environment = {key: os.environ[key] for key in ('LD_LIBRARY_PATH', 'OMNIGIBSON_DATA_PATH',
        'VK_ICD_FILENAMES') if key in os.environ}
    from wia.runtime.resources import execution_resource_environment
    environment.update(execution_resource_environment())
    environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', OMNIGIBSON_GPU_ID='0', OMNIGIBSON_HEADLESS='1',
        OMNI_KIT_ACCEPT_EULA='YES', OMNIGIBSON_APPDATA_PATH=str(cache),
        XDG_CACHE_HOME=str(cache / 'xdg_cache'), CUDA_CACHE_PATH=str(cache / 'cuda_cache'),
        __GL_SHADER_DISK_CACHE_PATH=str(cache / 'nvidia_shader_cache'),
        TMPDIR=str(cache / 'native_tmp'), TMP=str(cache / 'native_tmp'), TEMP=str(cache / 'native_tmp'))
    exports = ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in environment.items())
    script = (exports + f'export CUDA_VISIBLE_DEVICES={device[1].strip()}\n'
        'export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\n'
        f'cd {shlex.quote(str(ROOT))} || return 1\n' + shlex.join(native) +
        f' >{shlex.quote(str(output/"simulator.log"))} 2>&1\n'
        f'printf "%s\\n" "$?" >{shlex.quote(str(output/"native_exit_code.txt"))}\n')
    (output/"start.sh").write_text(script)
    from wia.runtime.sim.esi_explore_scenes import TASK_INSTRUCTION
    agent_task = (["--task", TASK_INSTRUCTION] if task is None else ["--task-file", str(task.path),
        "--variant", args.variant, "--seed", str(args.seed), "--episode-manifest", str(output/"episode_manifest.json")])
    if args.data_root is not None:
        agent_task += ["--data-root", str(args.data_root.resolve())]
    agent = [args.python, "-u", str(ROOT/"src/wia/runtime/commands/run_esi_agent.py"),
        "--run-mode", args.run_mode, "--url", agent_url, *agent_task, "--output-dir", str(output/"episode"),
        "--model", args.model, "--model-timeout", str(args.model_timeout),
        "--max-steps", str(args.max_steps),
        "--action-mode", args.action_mode, "--memory-mode", args.memory, "--world-during-thinking", args.world,
        "--observation-condition", args.observation_condition,
        "--codex-executable", args.codex_executable]
    agent += backend_cli_arguments(args)
    if args.reasoning_effort is not None:
        agent += ["--reasoning-effort", args.reasoning_effort]
    agent += ["--max-invalid-retries", str(args.max_invalid_retries), "--snapshot-cache", args.snapshot_cache]
    if args.max_recoverable_failures is not None:
        agent += ["--max-recoverable-failures", str(args.max_recoverable_failures)]
    if physical_evaluator:
        agent += ["--evaluation-file", str(output/"private_evaluation/state.json")]
    agent += ["--world-observation-log" if args.world_observation_log else "--no-world-observation-log"]
    if args.timeout is not None:
        agent += ["--episode-timeout", str(args.timeout)]
    manifest = {"stage": "initializing", "native_command": native, "agent_command": agent, "seed": args.seed,
        "scene_config": str(output/("scene.json" if task is None else "task.yaml")),
        "scene_config_sha256": hashlib.sha256((output/("scene.json" if task is None else "task.yaml")).read_bytes()).hexdigest(),
        "model": args.model, "reasoning_effort": args.reasoning_effort, "backend_metadata": backend_metadata(args), "world_mode": args.world,
        "world_observation_log": args.world_observation_log,
        "memory_mode": args.memory, "observation_condition": args.observation_condition,
        "execution_mode": "local", "gpu_mapping": gpu, "tmux_target": target, "node": node, "started_at": datetime.now(timezone.utc).isoformat(),
        "human_preview": args.preview_fps > 0, "preview_fps": args.preview_fps,
        "keep_simulator_requested": args.keep_simulator, "recording_requested": args.record,
        "execution_profile": args.execution_mode, "active_episode_dir": "episode"}
    hashes = {}
    for folder in ("core", "gumi", "scripts", "configs"):
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and path.suffix in (".py", ".json", ".html"):
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/"source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    def save_status(stage, **fields):
        manifest.update(stage=stage, **fields)
        (output/"experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if stage in ("debug_waiting", "agent_resuming") and state_path.exists():
            runtime = json.loads(state_path.read_text())
            service = runtime.get("service", {})
            if service.get("output_dir") == str(output):
                service.update(status="running", simulator_retained_after_episode=True,
                               control_owner="human" if stage == "debug_waiting" else "agent")
                service.setdefault("agent", {}).update(status=stage,
                    output_dir=str(output / manifest.get("active_episode_dir", "episode")))
                state_path.write_text(json.dumps(runtime, indent=2) + "\n")
        print("[experiment]", output.name, stage, flush=True)
    # The caller must finish/stop its previous scene before launching a fresh one.
    try:
        existing = requests.get(url+"/api/state", timeout=2)
        if existing.ok:
            raise RuntimeError("Stop the previous native service before starting a fresh scene.")
    except requests.RequestException:
        pass
    save_status("initializing")
    resident_client = None
    from wia.runtime.lifecycle_log import emit
    scene_started = time.monotonic()
    lifecycle = {'task': task.definition.id if task is not None else output.name,
                 'output_dir': str(output)}
    from wia.runtime.scene_capture_resident import RendererLogMonitor
    renderer_monitor = RendererLogMonitor(output / 'simulator.log')
    if resident_control:
        from wia.runtime.resident_episode import ResidentEpisodeClient, read_service
        resident_client = ResidentEpisodeClient(resident_control, gpu, environment, args.python)
        resident_client.ensure()
        first_world = read_service(resident_control).get('completed_scenes', 0) == 0
        scene_started = time.monotonic()
        emit('scene_loading', **lifecycle)
        resident_client.submit(native[3:], output)
        first_world = not resident_client.started_reused
        save_status('initializing', persistent_simulator=True, resident_control_dir=resident_control)
    else:
        emit('simulator_initializing', **lifecycle)
        emit('scene_loading', **lifecycle)
        subprocess.run(["tmux", "send-keys", "-t", target, "-l", "source " + shlex.quote(str(output/"start.sh"))], check=True)
        subprocess.run(["tmux", "send-keys", "-t", target, "Enter"], check=True)
    recording = False
    rollout_completed = False
    native_ready = False
    gateway = None
    gateway_log = None
    try:
        # Native web controls expose HTTP, while the existing CPU gateway owns MCP.
        gateway_log = (output / 'gateway.log').open('w')
        gateway = subprocess.Popen([args.python, '-u', str(ROOT / 'src/wia/runtime/commands/run_gumi_gateway.py'),
            '--backend-url', url, '--host', '127.0.0.1', '--port', str(gateway_port)],
            cwd=ROOT, stdout=gateway_log, stderr=subprocess.STDOUT)
        save_status('initializing', gateway_url=agent_url, gateway_pid=gateway.pid)
        # Asset loading depends on hardware and scene size, not episode budgets.
        # Wait for readiness or an actual native exit; the user can interrupt.
        while True:
            if resident_client is not None:
                from wia.runtime.resident_episode import read_service
                fatal = renderer_monitor.failure()
                service = read_service(resident_control)
                failure = fatal
                if service is None or service.get('state') == 'FAILED':
                    failure = failure or 'Owned native service failed during scene initialization.'
                if (output/'native_exit_code.txt').exists() or (output/'initialization_failure.json').exists():
                    failure = failure or 'Native initialization failed; inspect simulator.log.'
                if time.monotonic() - scene_started >= 1200:
                    failure = failure or 'Native initialization exceeded the 1200-second deadline.'
                if failure:
                    if resident_client.retry_initialization(failure, agent_started=False):
                        renderer_monitor = RendererLogMonitor(output/'simulator.log')
                        renderer_monitor.offset = 0
                        first_world = True
                        scene_started = time.monotonic()
                        save_status('initializing', cold_fallback=True, agent_started=False)
                        continue
                    from wia.runtime.resident_episode import stop_service
                    stop_service(resident_control)
                    raise RuntimeError(failure + ' Cold initialization failed; no further startup retry.')
            if gateway.poll() is not None:
                raise RuntimeError('Owned MCP gateway exited; inspect gateway.log.')
            qualification = read_native_qualification(output)
            if qualification is not None:
                if qualification["qualification_status"] == "skipped" and args.run_mode != "diagnostic":
                    save_status("skipped", qualification_status="skipped", skip_reason=qualification["reason"],
                                agent_started=False, task_success=None)
                    return 0
            if (output/"native_exit_code.txt").exists():
                raise RuntimeError("Native initialization failed; inspect simulator.log.")
            try:
                response = requests.get(agent_url+"/api/agent/state", timeout=3)
                if response.ok and (output/"metadata.json").exists():
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        native_ready = True
        if resident_control and first_world:
            emit('simulator_initialized', **lifecycle, elapsed_s=round(time.monotonic()-scene_started, 2))
        emit('scene_ready', **lifecycle, elapsed_s=round(time.monotonic()-scene_started, 2),
             persistent_simulator=bool(resident_control))
        if args.preview_fps > 0:
            verify_human_preview(url, output)
        if args.record:
            response = requests.post(url+"/api/start", json={}, timeout=10)
            response.raise_for_status()
            recording = response.json().get("ok") is True
        current_gpu = json.loads(state_path.read_text())
        current_gpu["service"].update(status="running", milestone=8.5 if task is not None else 8., output_dir=str(output),
            pid=int((output/"sim_pid.txt").read_text()),
            log_file=str(output/"simulator.log"), start_script=str(output/"start.sh"),
            validation_output_dir=str(output), current_scene=config["scene"] if config else task.scene_name,
            current_room=config["room"] if config else task.room_name,
            scene_config=manifest["scene_config"], scene_seed=args.seed,
            recording_dir=str(output/"recordings"), evaluation_dir=str(output/"private_evaluation"),
            forward_target=url, observation_pipeline="optimized", macro_execution=args.macro_execution)
        current_gpu["service"].update(performance_profile="benchmark", preview_fps=args.preview_fps,
                                      human_preview=args.preview_fps > 0,
                                      human_recording=args.record,
                                      recording=recording, control_owner="agent",
                                      scene_variant=config["variant"] if config else args.variant)
        current_gpu["service"].setdefault("agent", {}).update(status="running", control_owner="agent",
            model=args.model, reasoning_effort=args.reasoning_effort, output_dir=str(output / "episode"))
        current_gpu["service"].setdefault("gumi", {}).update(status="running", control_owner="agent",
            recording_dir=str(output / "recordings"))
        current_gpu["service"].setdefault("mcp", {}).update(status="running",
            server_pid=gateway.pid, url=agent_url + '/mcp', backend_url=url)
        if args.preview_fps > 0:
            current_gpu["service"].update(verified_at_utc=datetime.now(timezone.utc).isoformat(),
                verification_file=str(output / "gumi_preview_verification.json"))
        if task is not None:
            save_status("qualified", qualification_status=read_native_qualification(output)["qualification_status"],
                        episode_manifest=str(output/"episode_manifest.json"))
        state_path.write_text(json.dumps(current_gpu, indent=2) + "\n")
        if args.initialize_only:
            save_status('initialization_only_completed', agent_started=False, model_calls=0)
            return 0
        minimum_context = int(os.environ.get('WIA_MIN_MODEL_CONTEXT', '0'))
        if minimum_context and args.backend == 'openai_chat_completions':
            endpoint = (args.base_url or args.local_base_url).rstrip('/') + '/models'
            credential = os.environ.get(backend_metadata(args)['api_key_env'], '')
            headers = {'Authorization': 'Bearer ' + credential} if credential else {}
            current_gpu['service']['agent']['status'] = 'waiting_model_context'
            current_gpu['service']['control_owner'] = 'human'
            current_gpu['service']['gumi']['control_owner'] = 'human'
            state_path.write_text(json.dumps(current_gpu, indent=2) + '\n')
            save_status('waiting_model_context', minimum_model_context=minimum_context)
            while True:
                if resident_client is not None:
                    from wia.runtime.resident_episode import read_service
                    fatal = renderer_monitor.failure()
                    if fatal:
                        resident_client.dispose_failed(output, fatal)
                        raise RuntimeError('Native renderer failed while waiting for model readiness. ' + fatal)
                    if read_service(resident_control) is None:
                        raise RuntimeError('The owned resident service exited while waiting for model readiness.')
                readiness = {'minimum_context': minimum_context, 'model': args.model,
                             'checked_at_utc': datetime.now(timezone.utc).isoformat()}
                try:
                    response = requests.get(endpoint, headers=headers, timeout=5)
                    response.raise_for_status()
                    models = response.json().get('data', [])
                    selected = next((model for model in models if model.get('id') == args.model), {})
                    observed = selected.get('max_model_len')
                    readiness['observed_context'] = observed
                    ready = isinstance(observed, int) and observed >= minimum_context
                except (requests.RequestException, ValueError) as error:
                    readiness['error_type'] = type(error).__name__
                    ready = False
                readiness['ready'] = ready
                (output / 'model_readiness.json').write_text(json.dumps(readiness, indent=2) + '\n')
                if ready:
                    break
                time.sleep(5)
            current_gpu['service']['agent']['status'] = 'running'
            current_gpu['service']['control_owner'] = 'agent'
            current_gpu['service']['gumi']['control_owner'] = 'agent'
            state_path.write_text(json.dumps(current_gpu, indent=2) + '\n')
        save_status("agent_running")
        emit('agent_running', **lifecycle)
        with (output/"agent.log").open("w") as stream:
            completed = subprocess.run(agent, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        finished_deadline = time.monotonic()+5
        while (output/"private_exploration/state.json").exists() and time.monotonic() < finished_deadline:
            if json.loads((output/"private_exploration/state.json").read_text()).get("episode_finished"):
                break
            time.sleep(.1)
        summary = json.loads((output/"episode/summary.json").read_text()) if (output/"episode/summary.json").exists() else {}
        for name, source in (("final_exploration_metrics.json", "private_exploration/state.json"),
                             ("final_evaluation.json", "private_evaluation/state.json")):
            if (output/source).exists():
                (output/name).write_bytes((output/source).read_bytes())
        save_status("rollout_completed", agent_exit_code=completed.returncode,
                    termination_reason=summary.get("termination_reason", "driver_initialization_failed"),
                    task_success=summary.get("task_success"), run_mode=args.run_mode,
                    evaluation_validity=summary.get('evaluation_validity'),
                    completion_score=summary.get('completion_score'), score_percent=summary.get('score_percent'))
        rollout_completed = True
        emit('agent_completed', **lifecycle, termination_reason=summary.get('termination_reason'))
    except BaseException as error:
        emit('scene_failed', **lifecycle, error=str(error))
        save_status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        if recording:
            try:
                response = requests.post(url+"/api/stop", json={}, timeout=600)
                (output/"recording_result.json").write_text(json.dumps(response.json(), indent=2)+"\n")
                export_deadline = time.monotonic()+600
                while time.monotonic() < export_deadline:
                    state = requests.get(url+"/api/state", timeout=10).json()
                    if state.get("recording_status") not in ("recording", "finalizing"):
                        (output/"recording_verification.json").write_text(json.dumps(state.get("last_recording"), indent=2)+"\n")
                        if state.get("recording_error"):
                            (output/"recording_failure_state.json").write_text(json.dumps({key: state.get(key) for key in
                                ("recording_status", "recording_error", "recording_views", "last_recording")}, indent=2)+"\n")
                        break
                    time.sleep(2)
            except requests.RequestException as error:
                (output/"recording_error.txt").write_text(str(error)+"\n")
        debug_retained = (args.execution_mode == "debug" and native_ready
                          and not (output / "native_exit_code.txt").exists())
        retained = debug_retained or retain_completed_simulator(url, output, requested=args.keep_simulator,
                                                               rollout_completed=rollout_completed)
        if debug_retained:
            # The native lease release has already paused physics. Keep both
            # processes and block this trial rather than advancing the suite.
            from wia.runtime.debug_lifecycle import wait_for_debug_control
            retained = wait_for_debug_control(output, url=url, gateway_url=agent_url,
                agent=agent, cwd=ROOT, record=args.record, save_status=save_status)
            active = output / manifest.get("active_episode_dir", "episode")
            if (active / "summary.json").exists():
                latest = json.loads((active / "summary.json").read_text())
                manifest.update(termination_reason=latest.get("termination_reason"),
                    task_success=latest.get("task_success"), evaluation_validity=latest.get("evaluation_validity"),
                    completion_score=latest.get("completion_score"), score_percent=latest.get("score_percent"))
            for name, source in (("final_exploration_metrics.json", "private_exploration/state.json"),
                                 ("final_evaluation.json", "private_evaluation/state.json")):
                if (output / source).exists():
                    (output / name).write_bytes((output / source).read_bytes())
        if gateway is not None and not retained and gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait()
        if gateway_log is not None:
            gateway_log.close()
        if not retained and not (output/"native_exit_code.txt").exists():
            emit('scene_clearing', **lifecycle)
            if resident_client is not None:
                resident_client.stop_scene()
            else:
                subprocess.run(["tmux", "send-keys", "-t", target, "C-c"], check=True)
        deadline = time.monotonic()+120
        while not retained and not (output/"native_exit_code.txt").exists() and time.monotonic() < deadline:
            time.sleep(1)
        if not retained and resident_client is not None:
            from wia.runtime.resident_episode import read_service
            while time.monotonic() < deadline:
                service = read_service(resident_control)
                if service is None or service.get('state') != 'SCENE_RUNNING':
                    break
                time.sleep(.2)
        if not retained:
            emit('scene_cleared' if (output/"native_exit_code.txt").exists() else 'scene_clear_timeout', **lifecycle)
        save_status("finished", simulator_retained=retained,
                    native_exit_code=(output/"native_exit_code.txt").read_text().strip()
                    if (output/"native_exit_code.txt").exists() else None)
        current_gpu = json.loads(state_path.read_text())
        if current_gpu.get("service", {}).get("output_dir") == str(output):
            current_gpu["service"].update(status="running" if retained else
                "stopped" if manifest["native_exit_code"] is not None else "stopping",
                recording=False, control_owner="human", last_completed_experiment=str(output/"experiment.json"),
                simulator_retained_after_episode=retained)
            current_gpu["service"].setdefault("agent", {}).update(status="idle", control_owner="human",
                last_completed_episode=str(output / manifest.get("active_episode_dir", "episode") / "summary.json"))
            current_gpu["service"].setdefault("gumi", {}).update(
                status="running" if retained else "gateway_running_backend_stopped", control_owner="human")
            current_gpu["service"].setdefault("mcp", {}).update(
                status="running" if retained else "gateway_running_backend_stopped",
                server_pid=gateway.pid if retained and gateway is not None else None)
            state_path.write_text(json.dumps(current_gpu, indent=2) + "\n")
        # Long snapshot recordings may exceed the historical native encoder's
        # per-view timeout. The native source PNGs and timings remain complete.
        # Export them with a CPU-only worker after physics has stopped, allowing
        # the next fresh scene to use the retained interactive allocation.
        if recording and manifest["native_exit_code"] is not None and not (output/"recording_recovery.json").exists():
            pending = list((output/"recordings").glob(".*.pending"))
            if len(pending) == 1 and (pending[0]/"samples.jsonl").exists():
                with (output/"offline_video_export.log").open("w") as stream:
                    recovery = subprocess.Popen([sys.executable, "-u", str(ROOT/"src/wia/runtime/commands/recover_esi_snapshot_video.py"),
                        str(output)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                (output/"recording_recovery_job.json").write_text(json.dumps({"pid": recovery.pid,
                    "status": "running", "source": "complete_native_snapshot_recording", "cpu_only": True,
                    "started_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2)+"\n")
        if episode_manifest is not None:
            finalize_task_record(output, episode_manifest, manifest)
    return manifest.get("agent_exit_code", 0) if episode_manifest is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())

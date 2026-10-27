# WIA: World Interactive Agency

WIA owns benchmark tasks, scene preparation contracts, scoring, trial configuration and result aggregation. It has no model runtime dependency. ShowHarness is the active optional system integration and executes each full episode through its existing native runner.

The pinned `ESI` submodule contains upstream ESI-Bench source code and datasets. Separately licensed native simulator assets are external prerequisites and are not included in that submodule.

This migration supports **editable source installs**. Keep the complete WIA and ShowHarness checkouts and WIA's pinned `ESI` submodule; stripped wheels do not contain all task, prompt and native script resources. Set `WIA_ROOT` or `WIA_ESI_ROOT` only when relocating benchmark resources; install the complete Harness checkout with the same Python interpreter. Missing Harness resources produce an actionable error before execution.

```bash
python -m pip install -e /absolute/path/to/WIA
wia --help
wia tasks
wia backends
wia eval --config /absolute/path/to/WIA/configs/codex_cli.yaml --dry-run
```

Base installation requires NumPy, PyYAML and Pydantic. The commands above and `import wia`, `wia.contracts`, and `wia.environments` work without ShowHarness, MCP, CUDA or a model client. To run the active integration:

```bash
python -m pip install -e /absolute/path/to/Show-Harness-Explore[harness]
python -m pip install -e /absolute/path/to/WIA[harness]
wia eval --config /absolute/path/to/config.yaml
```

Use an environment with the existing ESI simulator dependencies for launch mode; installing WIA's `simulator` extra supplies lightweight client dependencies, not NVIDIA drivers, separately licensed native simulator assets or OmniGibson. Launch retains the native runner's idle-GPU checks, persistent `codex-gpu` tmux window, explicit device mapping, control gateway, recording and teardown. Existing workloads are never displaced by WIA. No GPU or live model was used for the integration tests.

A configuration is a YAML mapping:

```yaml
tasks:
  - path: ../explore_to_achieve/wia_v1/temporal/tasks/history_a_v1.yaml
    # seed and variant are optional: use the task's defaults.
policy:
  type: show_harness
  backend: codex_cli
  model: YOUR_EXPLICIT_MODEL_NAME
environment:
  mode: launch
  # python: /absolute/path/to/simulator/python
  # gpu: 0
evaluation:
  run_mode: diagnostic
  # Omitted budgets, world timing, observations and action mode retain task defaults.
output_dir: ../results/my_fresh_run
```

The sample task is pending, so its configs explicitly use diagnostic mode. This does not qualify it for benchmark scoring. Choose a ready task for benchmark mode. Models must be explicit; replace `YOUR_EXPLICIT_MODEL_NAME`. Relative task, data, manifest and evaluator paths resolve against the config directory, or against `resource_root` when set. `output_dir` and `resource_root` themselves resolve against the config directory. Unique task IDs also use the canonical task catalogue; ambiguous IDs fail. Task YAML and config duplicate keys fail. Seeds are nonnegative integers. Ordinary task variants default to `original`; approved rigid containment tasks default to `tabletop`.

`policy.backend` accepts `codex_cli`, `openai_chat_completions`, `openai_responses` or legacy `local_http`. The HTTP dialect is always explicit. Chat Completions supports `provider: openai|vllm|compatible`, `max_output_tokens`, optional `temperature` and `reasoning_effort`; `chat_template_kwargs` is accepted only for `provider: vllm`. Responses supports token limits, optional temperature and reasoning effort. Codex CLI uses its own configuration and does not accept HTTP URL/credential/generation settings. `local_http` preserves its existing adapter; it requires `base_url` and does not accept new API generation options. Declarative tasks require structured output. API dialects currently support declarative task episodes; existing Harness text/scene-QA paths keep their original transports.

The [vLLM example](configs/vllm_chat_completions.yaml) supplies the required explicit endpoint for `vllm` and `compatible` providers; their default credential variables are `VLLM_API_KEY` and `LOCAL_AGENT_API_KEY`, while OpenAI uses `OPENAI_API_KEY`, legacy local HTTP uses `LOCAL_AGENT_API_KEY`, and explicit `api_key_env` takes precedence.

HTTP credentials are read from the variable named by `api_key_env`, never from credential values in YAML. Defaults are `OPENAI_API_KEY` for canonical API backends and `LOCAL_AGENT_API_KEY` for legacy `local_http`. An explicitly named empty/unset variable permits a server that needs no key. Base URLs cannot include embedded credentials or query parameters. Output contains credential variable names only; provider keys stay in the process environment.

Evaluation options include `max_steps`, `timeout` (seconds or null), `action_mode`, `world`, `observation_condition`, `run_mode`, `world_observation_log`, `max_invalid_retries`, `max_recoverable_failures` and `snapshot_cache`. Launch environment options include `gpu`, `port`, `python`, `data_root`, `record`, `preview_fps`, `debug_cameras`, `keep_simulator` and `macro_execution`. Unknown fields fail instead of being silently ignored.

Attach uses an existing prepared episode without a scene reset:

```yaml
tasks:
  - path: /absolute/path/to/task.yaml
    url: http://127.0.0.1:8615
    episode_manifest: /absolute/path/to/prepared/episode_manifest.json
    evaluation_file: /absolute/path/to/private_evaluation/state.json
policy:
  type: show_harness
  backend: openai_responses
  model: YOUR_EXPLICIT_MODEL_NAME
environment:
  mode: attach
output_dir: /absolute/path/to/fresh_results
```

A single attach episode may place the URL, manifest and private evaluator reference in `environment`. Multiple episodes require explicit per-task URL/manifest pairs. WIA rejects repeated attach URLs or manifests because their reuse would ambiguously resume the same scene. Task identity, source hashes, seed, variant, model, reasoning effort, backend and episode settings must agree with the immutable manifest. Omitted episode settings inherit the manifest. Private evaluator references are never policy observations. Without a configured private evaluator, a completed execution can have unavailable scoring and is not counted as a scored success/failure.

Dry-run validates and prints frozen execution plans. It imports no Harness code, touches no GPU, starts no simulator, calls no model or network, and writes no output. Existing output directories are rejected even when empty. Execution writes `effective_config.json` and `results.json` in a fresh directory. Each trial retains its authoritative Harness `summary.json`, prompts, model traces, timing, usage and evaluation records. WIA reports completed successes, completed unsuccessful episodes, model errors, environment/observation/execution errors, runtime errors and invalid/incomplete scores separately. Unknown cost is JSON null. It never adds a second decision loop or adds action counts to the runner's counts. Source commits/versions are recorded when available.

External integrations register a trial executor via `wia.policies.register_policy(name, factory)` or an installed `wia.policy_systems` entry point returning a zero-argument executor factory. An executor implements `execute(plan) -> authoritative summary`; it owns its policy/environment lifecycle and action loop. External policies are not required to name a model or backend. `PolicySystem` defines the public `initialize`, `start_episode`, `decide`, `end_episode`, `close` lifecycle for future policy, WorldModel, WAM or Claude integrations. These are extension interfaces, not implemented model systems. See [architecture](/local_home2/zhouxinyang/models/WIA/docs/architecture.md).

For old ShowHarness source-checkout imports, the migration retains explicit compatibility modules and a relative `wia` source bridge. That bridge supports running older scripts without restarting an already active process; it is excluded from Harness distribution discovery. Ordinary Harness package/backend-registry use does not require WIA. Benchmark compatibility modules that now alias WIA require installing WIA or retaining the source bridge.

Run CPU-only integration checks:

```bash
python -m pytest tests/test_evaluation_entry.py tests/test_optional_harness.py
```

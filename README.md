# WIA: World Interactive Agency

WIA owns benchmark tasks, scene preparation contracts, scoring, trial configuration and result aggregation. It has no model runtime dependency. WIA owns the native runner under `wia.runtime`. ShowHarness is an optional provider of model sessions, actions, observations and recording; its generic dataset interface loads `wia.integrations.show_harness:WIADatasetAdapter` only when selected.

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

Use an environment with the existing ESI simulator dependencies for launch mode; installing WIA's `simulator` extra supplies lightweight client dependencies, not NVIDIA drivers, separately licensed native simulator assets or OmniGibson. Launch retains the native runner's idle-GPU checks, persistent `codex-gpu` tmux window, explicit device mapping, control gateway, recording and teardown. Existing workloads are never displaced by WIA. The extraction was checked with CPU integration tests and five native smoke tasks; no live model was used in those smoke tests.

A configuration is a YAML mapping:

```yaml
tasks:
  - path: ../explore_to_achieve/wia_v1/release/tasks/t40.yaml
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

The sample configs explicitly use diagnostic mode. Selected release tasks may also run in benchmark mode under the frozen-load policy; neither mode certifies physical success. Pending tasks outside the selected catalogue still require qualification or diagnostic mode. Models must be explicit; replace `YOUR_EXPLICIT_MODEL_NAME`. Relative task, data, manifest and evaluator paths resolve against the config directory, or against `resource_root` when set. `output_dir` and `resource_root` themselves resolve against the config directory. Unique task IDs also use the canonical task catalogue; ambiguous IDs fail. Task YAML and config duplicate keys fail. Seeds are nonnegative integers. Selected release tasks default to their catalogue variant. Other ordinary task variants default to `original`; approved rigid containment tasks default to `tabletop`.

`policy.backend` accepts `codex_cli`, `openai_chat_completions`, `openai_responses` or legacy `local_http`. The HTTP dialect is always explicit. Chat Completions supports `provider: openai|vllm|compatible`, `max_output_tokens`, optional `temperature` and `reasoning_effort`; `chat_template_kwargs` is accepted only for `provider: vllm`. Responses supports token limits, optional temperature and reasoning effort. Codex CLI uses its own configuration and does not accept HTTP URL/credential/generation settings. `local_http` preserves its existing adapter; it requires `base_url` and does not accept new API generation options. Declarative tasks require structured output. API dialects currently support declarative task episodes; existing Harness text/scene-QA paths keep their original transports.

The [vLLM example](configs/vllm_chat_completions.yaml) supplies the required explicit endpoint for `vllm` and `compatible` providers; their default credential variables are `VLLM_API_KEY` and `LOCAL_AGENT_API_KEY`, while OpenAI uses `OPENAI_API_KEY`, legacy local HTTP uses `LOCAL_AGENT_API_KEY`, and explicit `api_key_env` takes precedence.

The [Hopper deployment preset](/local_home2/zhouxinyang/models/WIA/configs/hopper_qwen38_27b.yaml) selects `Qwen/Qwen3.8-27B` at `http://127.0.0.1:8000/v1`. Keep the forwarding connections active while using it. Current user defaults explicitly enable thinking, allow 50,000 output tokens, use a 262,144-token sliding conversation window and a 1,800-second model/tool timeout. Native task defaults remain intact. The selected temporal task is pending and uses diagnostic mode; API checks do not qualify a native benchmark result. Historical integration runs retain their original 1,024-token/full-history settings in their frozen artifacts. With the editable packages installed in the active environment, validate this preset with:

```bash
python -m wia eval --config /local_home2/zhouxinyang/models/WIA/configs/hopper_qwen38_27b.yaml --dry-run
```

The separate [bounded-history Hopper diagnostic](/local_home2/zhouxinyang/models/WIA/configs/hopper_qwen38_27b_history_diagnostic.yaml) sets `history_text_turns: 2`, disables thinking and permits 4,096 output tokens. It keeps every current native camera/depth input and calibration unchanged, while replaying only the last two verbatim public final outputs from earlier decisions. Historical prompts, images and depth-tool exchanges are omitted. The prompt and session provenance explicitly identify this intervention; `history_text_turns` is accepted only for Chat Completions in diagnostic mode. Its temporal/memory outcomes must not be presented as results under the standard full-history protocol.

The [schema-constrained full-input diagnostic](/local_home2/zhouxinyang/models/WIA/configs/hopper_qwen38_27b_schema_diagnostic.yaml) preserves original RGB resolution and the complete conversation. `json_schema_output: true` registers a typed `submit_decision` output function alongside the existing read-only depth functions and requests schema-constrained function selection. The output function returns exact model-authored arguments to the ordinary validator; robot execution remains in the runner. This is an explicitly different diagnostic transport and is not the default free-output condition. Native launch can optionally wait for the matching served model to publish a minimum context length with the operational environment variable `WIA_MIN_MODEL_CONTEXT`; this does not change task budgets. The October 27 full-input campaign requests 240,000 tokens of published capacity before dispatch.

For persistent native episodes, set `WIA_RESIDENT_CONTROL_DIR` to a dedicated per-GPU service directory and `WIA_RESIDENT_TASK_PROFILES` to the campaign's native task profile map. The WIA runtime retains one native process and Kit application per GPU, dispatching each task through the original native runner. At each scene boundary the existing capture lifecycle clears the entire stage and sensor graphs; the next task receives fresh controllers, evaluator and episode state. Per-scene `persistent_scene_receipt.json` records process/application identity and cleanup. Failed native initialization makes a service unusable. The campaign deadline ends model work while leaving healthy simulator services available for reuse.

The current [thinking/sliding preset](/local_home2/zhouxinyang/models/WIA/configs/hopper_qwen38_27b_sliding_diagnostic.yaml) retains original RGB, raw depth and schema-function decisions. `sliding_window_tokens` uses the serving tokenizer to reserve the configured output budget plus 2,048 safety tokens. It pins the entire initial task/overview turn, preserves the current observation and in-progress tool chain, and removes the oldest complete intermediate turns as needed. It neither summarizes history nor resizes images. An input that cannot fit without discarding pinned/current content fails explicitly. Eviction receipts record the boundary changes. Results are separate from earlier nonthinking and thinking/full-history conditions; their latency numbers must not be pooled across these settings.

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

WIA v0.2 uses the active catalogue at `explore_to_achieve/wia_v1/release/catalogue.json` (60 selected definitions across six families, including AP-4 recovery mechanisms). `wia tasks` and campaign preparation use this catalogue. Task statuses, physical settings and scoring remain unchanged. Historical task/review artifacts are retained but are not supported runtime inputs: old `core.esi_converters.*` identities and Harness import aliases have been removed. Install WIA explicitly; no Harness source bridge is provided.

Episode archives now use content-addressed exact model PNGs, 512-pixel JPEG previews (quality 80), and lossless gzipped NPY evidence. Read archive paths from JSON rather than assuming filenames. Live model images and numeric depth inputs remain unchanged. New human videos default to 384-pixel tiles and CRF 32. Existing outputs are not rewritten.

Results include `token_accounting`: completed-decision/provider-native usage, observed API-request usage, and missing-usage coverage. Aggregate fields expose known sums and missing-trial counts; absent usage never becomes zero. Reasoning tokens are a subset of output tokens. Billing completeness remains unknown.

Run CPU-only integration checks:

```bash
python -m pytest tests/test_evaluation_entry.py tests/test_optional_harness.py
```

## Fixed v1 loading and runtime ownership

The release catalogue at `explore_to_achieve/wia_v1/release/catalogue.json` contains 60 selected task/scene pairs. Its files load in benchmark mode without physical requalification or a pending-review gate. Loading records `loaded_release_v1`, `physical_checks_run: false` and `physical_safety_inferred: false`. Schema/resource checks, scene construction, runtime action interlocks and evaluation remain active; successful loading is not a calibrated Safety result.

Launch native tools with `python -m wia.runtime.commands.run_esi_robot` and the episode runner with `python -m wia.runtime.commands.run_esi_explore_episode`. Dataset-specific tools and their integration tests moved from Harness to `src/wia/runtime` and `tests/harness_integration`; old Harness module paths have no wrappers. Native prompt templates live in `explore_to_achieve/prompts`. The dataset API stays in [ShowHarness](/local_home2/zhouxinyang/models/Show-Harness-Explore/showharness/datasets.py).

A bounded native check can be run in the behavior environment with `python scripts/smoke_release_native.py --task t01 --gpu 4 --output /absolute/new/output --actions`. The script records its explicit GPU mapping and configuration, rechecks idle capacity, uses this server's installed Vulkan/assets configuration, and makes no model call. Choose a currently idle GPU and a fresh output directory.

### Video recording and intermediate rendering

`wia debug` (or `evaluation.execution_mode: debug`) records by default;
`wia eval` defaults to batch execution and disables recording by default.
The lifecycle setting is independent of scientific `evaluation.run_mode`. Explicit `environment.record: true` or
`false` overrides either default. `environment.recording_hz` defaults to `5.0`:
while a recording is active, actions acquire human camera frames every 0.2
seconds of simulation time. Physics stepping and the model's action-boundary
observation contract are unchanged. Recording adds rendering and encoding cost.

```yaml
environment:
  mode: launch
  record: true
  recording_hz: 5.0
evaluation:
  run_mode: diagnostic
```

The episode CLI accepts `--record`, `--no-record`, and `--recording-hz 5`.
The native scene CLI uses `--human-recording` / `--no-human-recording` and the
same `--recording-hz` option. Capture frequency is distinct from encoded FPS
and playback speed: 5 Hz supplies five source frames per simulation second.
Existing wall-clock export and playback-speed settings still apply; accelerated
playback can discard intermediate frames. This is not per-physics-step video.

For native Codex trials, model waits and the overall episode have no deadline by
default. Use `policy.model_timeout: null` and `evaluation.timeout: null` to record
this explicitly. Control-lease initialization also waits without a time limit.
Manual interruption remains available; single-action controller timeouts and
step-count budgets are unchanged. Other model backends retain their existing
model-timeout support and defaults.

Agent video recording excludes the `awaiting_model` phase from its playback timeline. This affects video timing only: physics, model inputs, action execution and episode wall-time metrics remain unchanged. Recording metadata preserves `wall_duration_s`, adds `recorded_duration_s` and `omitted_model_wait_s`, and marks `skip_model_wait: true`. Snapshot timing records preserve wall timestamps and add `recorded_elapsed_s`. The 5 Hz action-render cadence and configured playback speed remain independent settings.

Default robot action deadlines are 120 seconds of simulation time for base motion, arm translation/wrist rotation, trunk motion and gripper control. `timeout_wall_s` is `null`: video rendering and machine load do not consume this action deadline. No-progress checks, IK/joint guards, settling requirements and explicit cancellation remain active. Native motion receipt waits (including MCP HTTP reads) have no wall deadline. Stop/recovery steps may follow the deadline to halt the robot. Custom diagnostic configs can still specify a finite wall deadline explicitly.

### Debug scene retention and batch evaluation

Use `wia debug --config /local_home2/zhouxinyang/models/WIA/configs/t07_sol_high_debug.yaml`
for one interactive episode. Debug defaults to 5 Hz video recording and diagnostic
qualification. After the model exits, including a capacity error, the coordinator
blocks, keeps the simulator and MCP gateway alive, and pauses physics at the
released action boundary. It never advances to another task automatically.
The existing model-wait omission policy still applies to video.

From another terminal, resume the saved native model thread and remaining budget:

```bash
wia debug-control --trial /absolute/output/trial_0000 --action resume
```

Each resume writes a separate `resume_0001`, `resume_0002`, etc. directory while
keeping the same live physical episode. `debug_state.json` reports the current
URLs, model progress and resume availability. Resume currently requires the
Codex backend and a model-error/cancelled episode with remaining decisions.
Other backends still retain the scene for inspection. A coordinator interruption
also leaves the scene alive; its saved resume command can be used directly,
but the detached coordinator cannot consume new control requests.

Explicitly close the retained scene and let the owner finish cleanup with:

```bash
wia debug-control --trial /absolute/output/trial_0000 --action close
```

Use `wia eval --config /local_home2/zhouxinyang/models/WIA/configs/t07_sol_high_batch.yaml`
for batch execution. It defaults to no recording, records per-trial failures,
cleans up each owned scene and continues to the next configured task.
`environment.record` and `environment.recording_hz` remain explicit overrides.
Batch mode rejects `keep_simulator`; debug accepts exactly one launched task.
Always choose a fresh output directory before launching either preset. Existing
resident processes must load the updated native code before using debug retention.

### Context policies

Set `policy.history_policy` in evaluation YAML or use `--history-policy`.
Explicit configuration overrides `WIA_HISTORY_POLICY` (with `WIA_GATEWAY_HISTORY_POLICY` as a legacy alias); omitted settings
preserve existing legacy behavior.

| Policy | History sent with current observations |
| --- | --- |
| `full` | Complete conversation, including historical images/tools |
| `current_sensors_full_text` | All historical text and returned reasoning; old images/depth omitted; historical RGB recall available |
| `current_sensors_last_result_notes` | Only the immediately previous final public action/output; persistent notes and historical RGB recall available |
| `recent_text` | Last N public outputs (`history_text_turns: N`, diagnostic mode) |
| `sliding_window` | Token-budgeted complete turns, initial/current turns pinned (`sliding_window_tokens: N`, diagnostic mode) |
| `showharness_state` | Fresh task/rules/current state and bounded executed actions (`history_recent_moves: 3` by default) |

These policies belong to the shared context engine, not to a model vendor, Gateway,
or API dialect. Direct Chat Completions, Responses, and local HTTP sessions use the
same policy and tool budgets; adapters only encode requests and routing options.
Native CLI sessions own their history internally and cannot yet apply client-side
projections; unsupported history ownership is rejected instead of silently ignoring
the policy. Sliding windows use the serving tokenizer when available (vLLM), otherwise
a logged UTF-8/image-patch estimate, not an exact provider token count.

Last-result/notes and fresh-state requests place resident instructions, output schemas,
task/policy and current action definitions before dynamic observations. Recovery guidance
stays with current measurements. Prefixes are rebuilt every decision so changed limits
are never frozen. Tool catalogs remain stable after the perception cap; prohibited calls
are rejected locally before dispatch. Request logs include prefix/catalog fingerprints.
These structural checks do not establish cache hit rates or task quality on a provider.
`current_sensors_recent2_notes` is accepted as a compatibility alias for
`current_sensors_last_result_notes`; it no longer replays two complete turns.
Historical experiment artifacts retain their original recorded semantics.

```yaml
policy:
  history_policy: current_sensors_last_result_notes
  max_perception_rounds: 4
```

The previous-result policy keeps a fixed global task/rules/tool prompt, then the
previous final public output and the current observation. Prior observation prompts,
intermediate tool chains and returned reasoning are not replayed. A rejected tool
decision contributes a rejection result rather than an older action. Current measured
execution feedback remains authoritative; the current action-history list contains at
most one executed action. Original logs and archived images remain on disk.

Within the current decision, sensor/tool results and returned reasoning remain available.
Each response containing depth/current RGB/historical RGB calls spends one perception
round; batch compatible calls. At the configured positive limit (default four),
perception tools are removed and a reminder requests final action JSON. Calling a
removed tool consumes an invalid decision with feedback; no action is inferred by the
harness. The allowance resets next decision. Global instructions explain that reads
use an immutable snapshot and do not refresh the world.

`episode_note({"operation":"append","text":"..."})` appends episode-local memory.
Successful writes total at most 128 Unicode code points per decision, including whitespace;
oversized writes are truncated to the remaining allowance and return a successful
result that tells the Agent how much text was saved. A zero allowance saves no text.
The cumulative 128-character allowance resets at each decision. `episode_note({"operation":"read"})`
returns all notes without size/count limits. Notes are not injected automatically.
Note-only calls do not consume perception rounds; reads remain available after the
perception limit. Thus this is not a hard cap on all API rounds or context size.
Writes persist in `episode_notes.jsonl`; receipts in `note_tool_calls.jsonl`.
The tools are documented in the global prompt. Logs do not add API episode resume support.

Current depth/RGB tools remain available in other policies as before. Historical RGB
recall is enabled in the two current-sensor policy families, not ShowHarness-state.
The state policy adapts official ShowLab commit
`6ad77dbc6180e973dff8148ffcca6c085580cea2` context management to WIA, preserving WIA's
sensors, actions and scoring. Initial images follow WIA's first-decision protocol.

Remove incompatible history-text/sliding-window fields when changing policies.
Model/provider restrictions, reasoning effort, output caps and decision budget are
separate settings. Fixed global prefixes allow reuse, but changing history means a
higher cache-hit percentage is not guaranteed. Explicit note reads can enlarge input.

### Extra fields in model action outputs

WIA drops unsupported JSON keys from a command envelope and from a recognized action's
command object before running the existing validation. A valid command executes normally;
extra fields do not spend an invalid decision. The next decision receives an explicit
accepted-with-ignored-fields notice. Original text, normalized text and ignored key paths
are saved in the attempt record; the previous-result policy replays the normalized public
result. Required fields, types, action names, magnitude limits and duplicate-key checks
remain enforced. No values are coerced, filled or changed. New code takes effect in newly
started agent processes; running Python agents are not hot-patched.

### v0.2 release suite

The [three-part release guide](/local_home2/zhouxinyang/models/WIA/docs/releases/v0.2/README.md) describes the independent dataset, cleaned evaluation code and ShowHarnessExplore snapshot. Set `WIA_DATASET_ROOT` for an external dataset package; code resources remain under `WIA_ROOT`. Default discovery follows the selected catalogue, so archived YAML files are not added to the active list. Build reproducible local archives with `python scripts/build_release_v02.py --output-dir dist/v0.2` using a fresh output directory.

### Parallel bench entry and terminal billing

Use the same entry point for any configured model and supported HTTP transport:

```bash
python -m wia bench --config configs/bench_deepseek_60.yaml --gpus 0,1,2,3 --output outputs/my_fresh_bench
python -m wia bench --config configs/bench_deepseek_60.yaml --workers 4 --gpus auto --dry-run
```

The supervisor checks model capabilities and official routing on Gateway, selects only
idle GPUs, checks independent ports, and starts one serial worker per GPU. Each native
scene launch rechecks the selected device. Tasks keep their original variants, seeds,
limits and scoring. Existing jobs are never displaced. Auto-GPU dry runs use placeholder
indices and do not contact a provider, inspect GPUs or start a simulator.

Bench defaults to one persistent simulator process per GPU worker. Each task
rebuilds its world, controllers, sensors and evaluator while retaining the Kit
application. Services are closed when the suite finishes or is interrupted.
Resident services run as worker-owned subprocesses and create no tmux sessions
or windows. Use `--no-persistent-simulator` to select fresh simulators instead.

The deep59 convenience script loads the key from `~/apikey_vercel.key` (or
`WIA_API_KEY_FILE`) without printing it, configures the existing simulator environment,
and runs in the current terminal. It sets `WIA_SKIP_GPU_CHECK=1`: the supervisor and
native episode launcher skip utilization, memory, process-occupancy and idle
rechecks. GPU index/UUID queries remain solely for explicit device mapping. With
`--gpus auto`, the first four visible devices are selected without checking whether
they are busy; explicit `--gpus` indices are used directly. Other entry points
retain their ordinary checks unless the same opt-out is set. No existing process
is terminated by this mode. Create or attach your own tmux session before invoking it:

```bash
bash scripts/bench_deepseek_4gpu.sh --dry-run
bash scripts/bench_deepseek_4gpu.sh
# Optional explicit physical device selection:
bash scripts/bench_deepseek_4gpu.sh --gpus 0,1,2,3
```

Defaults are all 60 tasks in the release catalogue, seed 0, per-task decision caps, diagnostic
batch execution, no recording, current sensors with the previous result and notes,
exact `deepseek/deepseek-v4.1-flash-1017`, official DeepSeek only, automatic caching,
and user-approved `low` reasoning for this debug bench (the model does not declare medium),
and a 384,000-token output reservation. `WIA_PYTHON`, `WIA_RUNTIME_LIBS`,
`OMNIGIBSON_DATA_PATH`, `VK_ICD_FILENAMES` and `WIA_GPUS` override server defaults.
Use a fresh output directory; existing results are not overwritten.

The parent terminal uses a single tqdm progress bar, refreshed every 30 seconds
by default (`--log-interval 30`). Redirected output receives plain progress lines
at the same interval. Simulator initialization, service readiness, scene loading
and readiness with elapsed time, Agent start/completion, and scene clearing are
reported immediately. These events do not add waits or pause physics. Full live
results are saved to disk; the terminal prints their path after completion.
The bar includes aggregate input, cached, output,
reasoning tokens and receipt-reported USD cost. Worker logs include every successful
API response's latency and billing receipt, including responses later rejected as
invalid decisions. Cached input is a subset of input; reasoning is a subset of output.
Missing counts/costs remain explicitly unknown; prices are not inferred from a model
name or a promotion. HTTP failures with no usage receipt cannot establish a final bill.

Artifacts include `bench_plan.json` (GPU UUID mapping and preflight metadata),
`worker_NN.log`, `worker_NN_events.jsonl` (timestamped lifecycle events),
`resident_NN/service.log`, per-worker effective configs and trial results, per-episode
`api_usage.jsonl`, and the combined `results.json`. A worker crash creates explicit
missing-trial failures and a nonzero exit status. No paid probe or GPU execution is
performed by `--dry-run`.

For the current launch audit, protocol risks and a complete archived T07 input/output walkthrough, see [formal evaluation readiness](/local_home2/zhouxinyang/models/WIA/docs/FORMAL_EVALUATION_READINESS_20261031.md).

To inspect all 60 scenes serially without a model, run the fixed native diagnostic
in an existing persistent terminal:

```bash
python scripts/check_all_scenes.py --gpu 4 --output outputs/my_fresh_scene_sweep
```

The [fixed sequence](/local_home2/zhouxinyang/models/WIA/configs/native_scene_validation.json)
runs after each scene loads: base left/right 5-degree turns, forward/back 0.05 m
translations, working-arm up/down 0.02 m moves, GRASP and RELEASE. Initial and
per-action checks read RGB, native depth pixels/regions and private scores. This
does not solve tasks or call model APIs. A blocked motion is retained as an action
outcome rather than treated as a scorer failure. The sweep reuses one simulator,
continues after failed scenes, and writes incremental `results.json`, per-scene
`scorer_validation.json`, simulator logs and lifecycle receipts. `--dry-run`
prints the plan; `--tasks t01 t02` selects a subset; `--scene-timeout` sets the
diagnostic load/sequence watchdog (default: 1200 seconds).

### Private activity statistics and bounded API attempts

New episode summaries separate `agent_decisions` (consumed decision steps),
`decision_calls` (invocations of the decision interface, including unsuccessful
invocations and forced selections), `model_calls` (actual HTTP generation
attempts, including tool rounds and failed attempts), and physical actions.
Historical summaries keep their original field meanings. Native Codex supplier
request counts are unavailable and are `null`; visible dynamic tools are counted.
`request_accounting` reports API successes, failures, timeouts and retries, plus
requested/executed/failed/rejected tool counts by name. Failed tool executions are
a subset of executed calls. Output-function submissions are named separately.
Private `api_usage.jsonl` contains one row per HTTP attempt; `tool_calls.jsonl`
contains local request/dispatch events. Missing usage or cost remains unknown.

HTTP model requests use at most three non-timeout failed attempts or ten timeout
attempts for the same unchanged request. The applicable counter reaching its limit
ends the episode with a model error. The counters reset after each successful
request; mixed failures retain both counters until that request succeeds or a
limit is reached. HTTP 408/504 and transport timeouts use the timeout counter.
Retries do not execute or replay a robot command, consume another decision step,
or reset the world. Each retry receives the same remaining model time allowance;
an explicit overall episode deadline can still end the episode sooner.
Tokenization calls are recorded separately from model generations. Codex CLI owns
its internal provider transport; these HTTP attempt limits do not claim control
over hidden CLI provider retries.

Safety uses `observational_safety_statistics_v3` and is statistics only. The
private native ledger records contact intervals/categories/objects, furniture
and structural contacts, controller-interlock events and drop/tipping/disturbance
candidates. Initialization events are separate. Contact is not automatically
classified as realized harm. Counts retain unavailable-measurement/coverage
information, stay outside task progress and model observations, and cannot modify
completion, success or task validity. The compatibility `safety_score` is always
`null`. Task-specific physical success predicates and controller interlocks remain
unchanged. `results.json` includes private `activity` aggregates and explicit
missing-trial counts; they do not enter quality scoring.

`world=run` is an intentional protocol choice and remains intact. No task budget
or ramp definition was changed by this statistics/retry update. Budget calibration
will optimize measured quality against API cost before setting a cap around
`ceil(1.3 * optimal_steps)`; the earlier large engineering recommendations have
not been applied.

### Per-task decision caps

The accepted geometric budgets are stored in each selected task's
`defaults.max_steps`. T52 has an explicit 10-decision cap. T42–T44 retain their
previous task defaults. The estimator and its ordinary-grasp/tray-placement
assumptions are reproducible with `python scripts/estimate_decision_budgets.py
--output /absolute/output`; T52's user override is included in that output.

The bench CLI and `scripts/bench_deepseek_4gpu.sh` default to `--max-steps -1`:
each trial resolves its own task budget before sharding and native launch.
A positive CLI value overrides every selected task, including T52. Zero and values
below -1 are rejected. For example:

```bash
bash scripts/bench_deepseek_4gpu.sh --dry-run
bash scripts/bench_deepseek_4gpu.sh --dry-run --max-steps 64
```

`wia eval` and `wia debug` also accept `--max-steps -1|N`; if omitted, they retain
the evaluation config's existing override behavior. YAML `evaluation.max_steps:
-1` inherits task defaults, while a positive YAML value remains a global override
unless the CLI overrides it. The bench CLI's default -1 explicitly selects task
budgets even if an older config contains a positive global value. The native
episode launcher likewise defaults to -1 and resolves it before manifest creation.
Every effective plan/manifest contains the resolved positive integer. Existing
attach episodes retain their immutable prepared manifest budget.

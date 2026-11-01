# Official simulator lifecycle audit

## Scope and evidence

This is a source/documentation audit, not a new native experiment. The installed official BEHAVIOR-1K checkout is clean at commit `79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429`; OmniGibson reports version 3.7.1. Local source inspection is pinned to this commit. Current online documentation and later fixes are identified separately.

## Official behavior

OmniGibson supports clearing and loading scenes while retaining the Kit application. The official [simulator guide](https://behavior.stanford.edu/omnigibson/simulator.html) explicitly documents `og.clear()` for clearing all scenes. The installed [environment source](https://github.com/StanfordVL/BEHAVIOR-1K/blob/79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429/OmniGibson/omnigibson/envs/env_base.py) also exposes `reload(configs)` and `reload_model(scene_model)`; these are not equivalent to a full stage clear and should not be treated as complete cleanup on their own.

| Purpose | Official operation | Lifetime |
| --- | --- | --- |
| Another episode in the existing environment | `env.reset()` | Keeps the loaded environment and simulator |
| Clear scenes and construct a new environment | `og.clear(...)`, then `og.Environment(configs=...)` | Rebuilds Simulator and stage; retains Kit application |
| Reload configuration/model through the environment API | `env.reload(...)` / `env.reload_model(...)` | Uses the environment loading path; requires version-specific lifecycle review |
| End the application | `og.shutdown()` | Closes the application |

The installed [official clear implementation](https://github.com/StanfordVL/BEHAVIOR-1K/blob/79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429/OmniGibson/omnigibson/__init__.py) performs:

1. Save current simulator settings unless explicit replacements are supplied.
2. Call `sim._partial_clear()`: stop physics; clear each scene using its real `scene.clear()`; remove global cameras, floor and skybox; clear vision sensors, object-state globals, materials, pose and naming caches.
3. Close the USD stage and clear the SimulationContext singleton.
4. Launch a new Simulator using the saved or explicit settings while preserving `og.app`.

The installed [official environment tests](https://github.com/StanfordVL/BEHAVIOR-1K/blob/79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429/OmniGibson/tests/test_envs.py) construct an environment, reset/step it, and call `og.clear()` between test cases. This supports the lifecycle's intended use; it does not establish that every scene/camera combination is free of native bugs. `Environment.close()` in this version is a no-op and does not replace `og.clear()`.

By default, `og.clear()` preserves the previous physics/render/action timestep. A new task with different required frequencies must supply its own settings. This agrees with the existing WIA timing reconciliation repair.

## Differences in WIA before the repair

The per-GPU request service, request ownership, task-specific cleanup integration and periodic process retirement are WIA additions. At the audited baseline (`d7a42d1`), `run_esi_capture_service.py` temporarily replaced each scene's `clear()` with a shortened cleanup that skipped native object removal and its callbacks. It also reset SyntheticData graph handles before official sensor cleanup. Consequently, calling `og.clear()` there did not mean WIA followed the complete official cleanup sequence.

These differences are concrete candidates for lifecycle investigation, not proven causes of the observed GPU crashes, camera-render hangs or placement hangs. The earlier explanation that same-process cross-scene reuse itself was a WIA invention was too broad and is corrected by this audit.

## Relevant official repair

Official [PR #2114](https://github.com/StanfordVL/BEHAVIOR-1K/pull/2114), merged on 2026-04-17 and listed in the [v3.9.0 release notes](https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.0), adds three explicit subscription cancellations immediately after stopping physics in `_partial_clear()`:

```python
self._pre_physics_step_callback.unsubscribe()
self._post_physics_step_callback.unsubscribe()
self._simulation_event_callback.unsubscribe()
```

Its stated purpose is to prevent callbacks belonging to old simulator instances from surviving `og.clear()` cycles. The GitHub API confirmed the merged patch. All three subscription attributes exist in installed 3.7.1, but its `_partial_clear()` lacks these calls. This makes a narrow compatibility-checked backport worth testing; it does not prove that the patch resolves our failures. A wholesale upgrade would change additional scientific/runtime behavior and is outside this audit.

NVIDIA's [current SimulationApp API](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.simulation_app/docs/api.html) likewise distinguishes application lifetime from stage lifetime and documents reapplying render settings when opening a new stage. This is corroborating current documentation, not a claim of exact API parity with installed Isaac Sim 4.5.

## Consequence for the pending repair

First compare the standard official cleanup path with WIA's shortened cleanup and validate the narrow official subscription fix against this installation. Retain the requested per-GPU persistent application architecture. Use fresh-process runs as an explicit isolation control or fallback, rather than describing them as the official requirement. No code patch, GPU rerun, scorer change or successful repair is claimed by this audit.

## Implemented correction

Commit `f6c4fbd` replaces the shortened cleanup and premature SyntheticData reset with `native_stage_cleanup` in the episode service, capture service, native editor and empty-stage timing/profile rebuild paths. Official object removal, sensor teardown, cache reset and stage reconstruction are retained. Kit application reuse and the existing bounded process lifetime are preserved.

For OmniGibson 3.7.1 only, the compatibility shim temporarily wraps the current Simulator's `_partial_clear`. It registers a native stop callback during that method so the exact three upstream subscriptions are cancelled after existing stop callbacks and before scene removal. This avoids a second `stop()` invocation, which would otherwise run existing stop callbacks twice. The temporary method and callback are restored even on failure. Other SDK versions use their native implementation. An unsupported subscription API or a cleanup exception fails visibly and prevents reuse.

The first 77 focused CPU checks passed; an additional passing regression verifies propagation and restoration if native unsubscribe fails. Native verification uses eight scenes on GPUs 4–7, two sequential scenes per resident process: T01/T36, T06/T49, T22/T50, T28/iw_m3_01. T01 is a healthy control. Commands, task definitions, seeds, physics profiles and scorers are unchanged. This selection changes shard order relative to the preceding full sweep, so runtime and pass-count comparisons are not controlled single-variable experiments. Native outcomes are recorded separately after completion.

## Follow-up corrections and fallback policy

The first native check exposed expensive regex compilation inside Isaac's real prim-deletion callbacks and a source cleanup hook swallowing an initial SDK clear failure. The follow-up retains every native object removal/callback, caches compiled deletion patterns within a bounded teardown-only cache, and releases extra non-robot/non-viewer cameras through their SDK API before parent scene deletion. The first clear exception is retained; a partially failed clear cannot be attempted a second time.

Different native launch macro settings now require a new owned Kit process. The client and service resolve the same defaults and declared overrides, including GPU dynamics and flatcache. Matching profiles retain the application; the service refuses hot macro changes after application launch. This is a conservative compatibility boundary, not proof that macro changes caused every observed renderer failure.

The user explicitly requested kill/restart fallback and recording failed cold scenes before proceeding. Default behavior is:

1. An actual reused-scene initialization failure retires the verified owner, archives the failed attempt, and tries the identical task/seed/arguments once in a fresh process. Elapsed real time does not trigger fallback.
2. If initialization was already cold, or the cold retry fails, record the failure and continue the diagnostic task list. No repeated cold attempts are made.
3. Once Agent execution starts, automatic replay is prohibited. A completed diagnostic sequence is also not replayed. Scorer unavailability alone does not trigger an initialization retry.
4. An actual completed-episode cleanup failure retires the old owner before the next task. Slow cleanup waits without a deadline. Active Agent/debug worlds are protected.
5. Before a new owner starts, unanswered old requests receive explicit retirement records so they cannot be executed again. These records do not invent native exit codes or certify success.

Failed warm artifacts remain under `TRIAL/initialization_attempts/REQUEST_ID/`; `TRIAL/simulator_fallback.json` records the cause and archive location. Cold failure evidence remains in the trial's native logs, initialization failure record and diagnostic results. Lifecycle events identify fallback and process recycling. No model/API calls or Agent decisions are replayed during this recovery.

CPU coverage includes matching/changed launch profiles, failed-owner disposal, original artifact preservation, bounded cold attempts, a real two-process CPU service demonstrating owner replacement without request replay, and a diagnostic coordinator demonstrating continuation after a failed cold scene. Native evidence and unresolved scene/scorer failures are documented in the accompanying validation report.

## Current real-time policy and completed cold evidence

The user subsequently removed simulator real-time limits and explicitly retained simulated-time convergence/blocked-motion rules. Startup, loading, switching, cleanup, capture, local simulator HTTP/MCP waits, video finalization and macro execution have no wall-clock deadline. Legacy motion wall-time settings and macro duration fields are ignored. Task decision/primitive/distance/angle budgets, physical convergence/blocked-motion predicates and model API timeout/retry rules remain intact. Polling intervals are status sampling, not failure deadlines. Genuine native failures and explicit user stops can still dispose the verified owner.

To apply this to the two already running cold checks, replaced only their three CPU watchdog processes. The native PIDs 2315416 (T49/GPU 4) and 2315411 (T50/GPU 6) continued unchanged under [resume_scene_checks.py](/local_home2/zhouxinyang/models/WIA/scripts/resume_scene_checks.py). Both eventually completed all eight actions, returned native exit code 0 and completed official full cleanup. T49 took approximately 35.9 minutes overall (10.27-second cleanup); T50 took 29.5 minutes (9.00-second cleanup). These observations show why a 20-minute wall deadline cannot establish that these scene initializations are broken. Historical watchdog outcomes remain preserved and were not rescored.

Both cold runs still report unavailable grading at their ten observation boundaries; this is an unresolved scorer issue, not a load or teardown failure. All owned native processes exited and GPUs 4–7 returned to 19–20 MiB. No model API call was made.

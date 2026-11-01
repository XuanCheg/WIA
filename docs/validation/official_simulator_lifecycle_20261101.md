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

## Differences in WIA

The per-GPU request service, request ownership, task-specific cleanup integration and periodic process retirement are WIA additions. More significantly, [run_esi_capture_service.py](/local_home2/zhouxinyang/models/WIA/src/wia/runtime/commands/run_esi_capture_service.py:44) temporarily replaces each scene's `clear()` with a shortened cleanup that skips native object removal and its callbacks. It also resets SyntheticData graph handles before official sensor cleanup. Consequently, calling `og.clear()` here does not mean WIA follows the complete official cleanup sequence.

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

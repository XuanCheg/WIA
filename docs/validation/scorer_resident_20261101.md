# Scoring, formal launch and loaded-world retention

## Scope and plan

Repair the observed T49/T50 and M3 unavailable results, remove prechecks from formal runs, and retain a completed world on each GPU. Preserve task goals, real scoring measurements, task action limits, private Safety statistics, API retry accounting, and simulated-time action convergence. No simulator wall-time deadline or paid model call is introduced.

## Scoring repairs

T49/T50 use an explicitly authored native cloth redrape layout. The old interaction initializer still attempted a rigid table relocation, eventually rejected it, and restored the native floor layout. The scorer subsequently required a successful table-relocation receipt, although actual original cover measurements passed. The cloth initializer now owns this interaction layout, without a table search. Diagnostic qualification verifies the exact task/source hash, inventory and live mesh/visual cover instead of requiring relocation. Formal scoring directly samples its native baseline. Cover movement, native contacts, grasp, retrieval and delivery predicates remain measured.

M3 previously returned missing predicates when a fully measured payload was not placed in any storage unit. That is an incomplete goal, so those predicates now return false. A missing native placement or containment measurement still returns unavailable. The first finite clock sample is a valid history baseline; it does not establish dwell or successful placement.

The first native rerun caught a structured-binding interface error in the redrape hook. It was corrected, and both rejected attempts remain in their original output directories. They are not successful scores.

## Formal runs

Formal `benchmark` runs omit gateway/model catalogue checks, proactive credential checks, GPU occupancy checks, port probes, preview verification, model-context readiness polling, scene qualification, visibility gates, and redundant evaluator initial qualifications. Device UUID lookup, configuration/ownership validation, scene construction, initial observations and scoring baselines remain necessary execution steps. A skipped qualification is recorded as `not_run`, with `accepted: null` where applicable. It is never recorded as a measured native pass.

Debug qualifications remain enabled. Actual missing scoring measurements still produce unavailable; omitting prechecks does not create substitute geometry, contacts or containment measurements.

## Resident lifecycle

1. Load the selected native scene, cameras and controllers.
2. Save the initial world through native `dump_state(serialized=False)` before fixed-sequence or Agent actions.
3. Execute the diagnostic or episode and persist its scoring/action outputs.
4. Restore that initial state with native `load_state`, restore source task-state counters/containers while preserving native handles, pause physics, and retain the world and cameras on the same GPU.
5. Publish `SCENE_RETAINED` before publishing the completed request response. This ordering prevents the parent monitor from mistaking completion for an active request and stopping it.
6. Leave successful resident owners alive when benchmark/debug workers finish. A request for another world performs the complete official cleanup before loading it. A genuine native failure still permits the previously implemented verified-owner cold fallback.

`native_exit_code.txt` remains the legacy completion receipt for the scene invocation. A zero there does not mean the resident OS process exited; `service.json` and the verified PID describe its lifetime. The existing bounded renderer lifetime still applies when another scene is requested, and can be disabled with `WIA_RESIDENT_MAX_SCENES=0`.

An early retention test exposed the response/state publication race; its passing T36/M3 diagnostics were subsequently cleared by the old monitor. A new native run validates the corrected order. State JSON alone is insufficient evidence of residency: verify the owner PID and actual GPU memory as well.

For supported fixed-sequence diagnostic worlds, repeat validation on the same owner without loading a new scene:

```bash
cd /local_home2/zhouxinyang/models/WIA
PYTHONPATH=src:/local_home2/zhouxinyang/models/Show-Harness-Explore \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python \
  scripts/validate_retained_scene.py \
  --control-dir /absolute/path/to/resident_000 \
  --output /absolute/path/to/a/new/result_directory
```

The owner advertises `retained_validation: true` when this operation is supported. A repeated check gets fresh observation/evaluator/Safety histories, keeps the same authored world, and restores the initial native state again afterward. Temporal phase worlds currently require ordinary scene initialization. This diagnostic interface does not silently attach a new formal Agent episode to an old episode contract.

## Evidence

The exact eight commands are left turn 5 degrees, right turn 5 degrees, forward 0.05 m, backward 0.05 m, working-arm up 0.02 m, down 0.02 m, GRASP, RELEASE, followed by DONE. Each check records the initial boundary, eight action boundaries and the final boundary. These actions test runtime/scorer availability; they do not solve the task.

Native artifacts are under:

- [First diagnostic run](/local_home2/zhouxinyang/models/WIA/outputs/scorer_resident_20261101)
- [Corrected T36/T49/T50 run](/local_home2/zhouxinyang/models/WIA/outputs/scorer_resident_fixed_20261101)
- [Corrected M3 retention run](/local_home2/zhouxinyang/models/WIA/outputs/scorer_resident_m3_fixed_20261101)

The final combined CPU group passed 556 checks, with 18 historical mirror definition/builder cases deselected after reproducing their failures with unchanged HEAD adapters. A separate lifecycle/ownership group passed 21 checks; these groups overlap. The existing failures concern generated mirror revision/module interfaces and are outside this request. No new failure remains in the targeted policy, scoring, lifecycle and ownership checks.

## Final native results

| Task | Physical GPU | Resident PID | Initial load + diagnostic | Repeat on loaded world | Scoring availability |
| --- | ---: | ---: | ---: | ---: | --- |
| T36 | 4 | 2472637 | 248.87 s | 32.93 s | complete at every boundary |
| T49 | 5 | 2472632 | 379.58 s | 89.15 s | complete at every boundary |
| T50 | 6 | 2472641 | 409.58 s | 135.79 s | complete at every boundary |
| iw_m3_01 | 7 | 2474225 | 306.45 s | 43.79 s | complete at every boundary |

All eight native checks completed the eight actions and ten observation/scoring boundaries, with no reported errors: 64 actions and 80 complete scoring boundaries overall. Repeated checks report `world_rebuilt: false` and preserve their original PIDs. All four owners were independently verified alive afterward in `SCENE_RETAINED`; GPU utilization was 0% with roughly 10.4–11.4 GiB allocated per loaded world. No model/API calls occurred. These are diagnostic timings with warmed caches, not controlled policy throughput measurements or task-solving results.

The old T49/T50 cold checks took 2154.19 s and 1768.96 s and returned unavailable scores. The new initial checks take 379.58 s and 409.58 s and have complete scoring. Both the interaction initialization path and post-check lifecycle changed, so these numbers are operational evidence, not an isolated speedup experiment.

The retained native owners in this report were started before the optional `--skip-prechecks` repeat-policy selector and source-counter container copying were added. They support ordinary repeat validation; changing precheck policy requires a newly started owner advertising that capability. Formal precheck omission and source-container restoration were validated with CPU checks; no paid formal Agent episode was launched.

Machine-readable timings, PID/control paths and artifact locations are in [the validation summary](/local_home2/zhouxinyang/models/WIA/docs/validation/scorer_resident_20261101.json). Every initial and repeated check retains its full native log, fixed action inputs/results, observation boundaries and private evaluator state in the linked output directories.

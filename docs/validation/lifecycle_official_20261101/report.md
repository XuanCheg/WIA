# Official lifecycle repair and cold fallback validation

## Behavior retained

Use the official full scene/object/sensor/global cleanup and preserve Kit for matching native launch macro settings. Backport the three PhysX subscription cancellations only for OmniGibson 3.7.1. Remove extra scene cameras through SDK APIs before parent deletion. Preserve the first cleanup exception. Cache only native deletion regex compilation; all deletion callbacks still execute.

Changes in native launch macros require an owned process restart. Failed reused initialization receives one fresh-process attempt with the same task, seed and native arguments. A failed cold attempt is recorded and the diagnostic proceeds to the next scene. Failed warm artifacts are archived; unanswered retired requests cannot replay. Agent execution and completed diagnostic sequences are never automatically replayed. Completed-episode cleanup failures retire the owner for the next task. Simulator real-time limits are disabled. Simulated-time convergence/blocked-motion predicates and decision/primitive/distance/angle budgets are retained; model API timeouts and retries are unchanged.

## Protocol and limits

Installed vendor checkout: OmniGibson 3.7.1, commit 79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429; no vendor files modified. Fixed sequence: turn left/right 5 degrees, move forward/back 0.05 m, move up/down 0.02 m, GRASP, RELEASE, followed by the existing DONE protocol. Phase transition uses the existing public protocol where applicable. Task definitions, sources, seeds, physics frequencies, score thresholds, safety isolation and world=run semantics were not edited. All checks use zero model calls.

The initial native run uses f6c4fbd; the follow-up uses b2da31d; the cold comparison uses f607239. Later request-retirement, continuation, local transport and macro wall-policy changes are covered by CPU fixtures. During the cold comparison, only the CPU watchdogs were replaced with an unbounded monitor; the two native processes and worlds were preserved. The current fallback was not injected into already running older services. This is a lifecycle investigation, not a controlled quality/speed comparison or a full 60-task certification. A diagnostic pass means commands and score availability were checked; it does not mean the fixed sequence solved the task.

## Follow-up resident results

| Task | GPU | Diagnostic status | Actions | Reused Kit | Cleanup (s) |
| --- | --- | --- | --- | --- | --- |
| t01 | 4 | passed | 8/8 | false | 12.45 |
| t06 | 5 | passed | 8/8 | false | 61.77 |
| t22 | 6 | passed | 8/8 | false | 8.17 |
| t28 | 7 | passed | 8/8 | false | 12.54 |
| t36 | 4 | passed | 8/8 | true | 9.11 |
| t49 | 5 | runtime_error | 0/0 | - | - |
| t50 | 6 | runtime_error | 0/0 | - | - |
| iw_m3_01 | 7 | issues_found | 8/8 | true | 12.56 |

T01 -> T36 and T28 -> iw_m3_01 retain the same process/application. T06 completes full cleanup in 61.77 seconds; the earlier attempt was interrupted after over five minutes inside native prim-deletion callbacks. Several cleanup changes and cache states differ, so no isolated speedup ratio is claimed. T22/T28/T36 no longer show their earlier camera-render timeouts in this subset. iw_m3_01 completes its commands and cleanup but still has independent candidate scorer unavailability. T50 encounters a Vulkan GPU page fault during a mixed-profile transition; T49 exceeds the initialization deadline. Macro-boundary recycling is a conservative compatibility fix, not proof of the cause of all GPU faults.

## Fresh-process T49/T50 comparison

Completed.

| Task | GPU | Diagnostic status | Actions | Reused Kit | Cleanup (s) |
| --- | --- | --- | --- | --- | --- |
| t49 | 4 | issues_found | 8/8 | false | 10.27 |
| t50 | 6 | issues_found | 8/8 | false | 9.00 |

Both tasks had a 600-second main-thread stack inside interaction_setup._place_robot -> Simulator.step before the sequence, but subsequently completed after wall-time watchdogs were removed. The original native PIDs were preserved. T49 completed eight actions and cleanup with native exit 0 in approximately 35.9 minutes; T50 did so in 29.5 minutes. Cleanup took 10.27 and 9.00 seconds. The earlier timeout alone cannot establish that initialization was broken. Both still have ten score_unavailable records; neither is classified as a load/teardown failure in the cold comparison. GPUs 4–7 were verified idle after completion.

## Software validation

454 focused CPU tests pass, including native cleanup ordering, subscription errors, original error preservation, camera ownership, matching/changed macro profiles, bounded cold attempts, a real two-process CPU service proving owner replacement without request replay, and a full diagnostic coordinator proving continuation after a failed cold scene. Slow-clock tests cover unbounded startup, cleanup, primitive replay and metric-target macros; native simulated-time timeout feedback and explicit cancellation remain tested. One existing trimesh degenerate-mesh warning remains. Native automatic warm-failure retry with the final request-retirement implementation is not claimed as exercised by these older native runs.

## Evidence

- initial: [lifecycle_official_20261101](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_official_20261101/results.json), completed=False, counts={"passed": 4, "issues_found": 2}
- followup: [lifecycle_official_followup_20261101](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_official_followup_20261101/results.json), completed=True, counts={"passed": 5, "runtime_error": 2, "issues_found": 1}
- cold: [lifecycle_cloth_cold_20261101](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_cloth_cold_20261101/results.json), completed=True, counts={"issues_found": 2}

Detailed machine-readable inventory: [summary.json](/local_home2/zhouxinyang/models/WIA/docs/validation/lifecycle_official_20261101/summary.json). Official-source audit and operational policy: [official_simulator_lifecycle_20261101.md](/local_home2/zhouxinyang/models/WIA/docs/validation/official_simulator_lifecycle_20261101.md).

## Concrete input/output example: T49

This is an actual fixed-sequence run, not a model-generated trajectory. The Agent is not started: model_calls=0, Agent decisions=0, native command executions=8. The diagnostic script supplies each command directly; no HTTP tool invocation or model output is invented.

Before starting, use the installed behavior environment and BEHAVIOR assets, select idle GPUs, and run from an existing persistent terminal. The diagnostic script prepares native/Vulkan/device settings. Model credentials are needed only for a separate formal Agent benchmark. Task YAML step budgets and explicit positive global overrides remain unchanged; no simulator wall-time option is needed.

```bash
cd /local_home2/zhouxinyang/models/WIA
/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python scripts/check_all_scenes.py --gpus 4,5,6,7 --output outputs/my_new_scene_check --tasks t01 t06 t22 t28 t36 t49 t50 iw_m3_01
```

The simulator constructs the world and robot, applies the declared initialization, binds cameras, and creates the initial observation. The scorer receives native measurements at each observation boundary and writes private evaluation separately. Safety evidence remains private statistics and is never used as a score or an Agent hint.

| Input | Simulator output / action receipt | Observation | Scorer output |
| --- | --- | --- | --- |
| Begin episode | Initial native camera/depth observation | snapshot 3, action step 0 | unavailable |
| `{"action":"BASE_TURN_LEFT","angle_deg":5}` | completed=true, timeout=false, reason=completed | snapshot 4, action step 1 | unavailable |
| `{"action":"BASE_TURN_RIGHT","angle_deg":5}` | completed=true, timeout=false, reason=completed | snapshot 5, action step 2 | unavailable |
| `{"action":"BASE_FWD","distance_m":0.05}` | completed=true, timeout=false, reason=completed | snapshot 6, action step 3 | unavailable |
| `{"action":"BASE_BACK","distance_m":0.05}` | completed=true, timeout=false, reason=completed | snapshot 7, action step 4 | unavailable |
| `{"action":"MV_UP","distance_m":0.02}` | completed=true, timeout=false, reason=completed | snapshot 8, action step 5 | unavailable |
| `{"action":"MV_DOWN","distance_m":0.02}` | completed=true, timeout=false, reason=completed | snapshot 9, action step 6 | unavailable |
| `{"action":"GRASP"}` | completed=true, timeout=false, reason=completed | snapshot 10, action step 7 | unavailable |
| `{"action":"RELEASE"}` | completed=true, timeout=false, reason=completed | snapshot 11, action step 8 | unavailable |
| DONE / final record | Terminal owner report; no additional motion | snapshot 11, action step 8 | unavailable |

The unavailable scorer output is an unresolved grading issue; it is not converted to a numeric zero or hidden by the successful simulator run. After all actions, the service cleans up SDK objects/sensors/subscriptions, writes its native receipt, and exits when the diagnostic owner closes it. No timer terminated this world.

For a formal Agent run, the model receives public task text, the initial overview and mounted observations; its chosen command is dispatched through the same owner-controlled execution path. The returned public observation becomes the next model input. The scorer remains private. agent.log then records public world_observation; model API attempts, Agent decisions and tool counts are separate. Those model inputs/outputs do not exist in this no-model example.

- [episode_manifest.json](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_cloth_cold_20261101/worker_00/00_t49/episode_manifest.json)
- [metadata.json](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_cloth_cold_20261101/worker_00/00_t49/metadata.json)
- [scorer_validation.json](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_cloth_cold_20261101/worker_00/00_t49/scorer_validation.json)
- [persistent_scene_receipt.json](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_cloth_cold_20261101/worker_00/00_t49/persistent_scene_receipt.json)
- [simulator.log](/local_home2/zhouxinyang/models/WIA/outputs/lifecycle_cloth_cold_20261101/worker_00/00_t49/simulator.log)

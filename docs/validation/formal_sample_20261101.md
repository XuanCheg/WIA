# Representative formal-path native sample and cold-start comparison

## Outcome

Twenty selected tasks finished: **15 passed**, **4 have observation/scoring problems**, and **1 has incomplete coverage**. The sample executed **152 fixed commands** and collected **191 explicit observation/scoring boundaries**. No model/API call occurred. These are runtime/scorer diagnostics, not task-solving results or model-quality scores.

The final boundary-validity distribution is 137 complete, 30 unavailable, 15 partial and 9 pending_done. The five Phase-1 partial boundaries in T40 and nine pre-DONE boundaries in GF2-1 are expected intermediate states; both tasks finalize complete. T38 remains partial. The 30 unavailable boundaries belong to T10, H1 and M4; M1 fails before the fixed sequence begins.

Four reused-world initializations failed. All four recovered initialization and fixed-sequence execution through the existing one-cold-attempt fallback. T38 still ends with partial scoring coverage; recovery from a renderer failure does not certify full evaluator coverage.

## Protocol and provenance

- Experiment code: commit `8de56f7`.
- Physical GPUs 4–7, five serial tasks per GPU, in existing `codex-gpu` terminals.
- Formal initialization (`run_mode=benchmark`), fixed native sequence, no policy/model. All 20 task-status records say `qualification_status=not_run`, and every validation report records `prechecks_enabled=false`.
- Original task files, source files, variants, seed 0 and per-task step caps are preserved. Final task/source hashes match the input receipt.
- Commands: left turn 5 degrees, right turn 5 degrees, forward 0.05 m, backward 0.05 m, working-arm up 0.02 m, down 0.02 m, GRASP, RELEASE; then DONE and an explicit public phase transition when supported.
- No simulator wall-time deadline. Existing virtual-time convergence/blocked-motion predicates remain. No videos are recorded.
- Keep the native profile restart boundary, default four-scene renderer lifetime, full SDK scene cleanup and one cold retry. A failed cold start would be recorded before continuing; no Agent work is replayed.
- The user requested no Vulkan/driver/renderer/scorer repairs, asked to try cold starts, and authorized continued use after other GPU workloads appeared. No such repair or quality reduction was applied.

The formal-mode selector was added to the existing diagnostic sweep and checked with 23 focused CPU cases before launch. The sample deliberately includes old problem cases. Formal-mode metadata is not a guarantee that every internal execution guard is removed: M1's authored-use-preview consistency assertion still aborts begin_episode in this version.

## Per-task results

Times include loading, any archived failed attempt/cold retry, fixed commands, observations and native initial-state restoration. Shared GPU workloads and cache differences make these operational timings, not an isolated performance comparison.

| Task | GPU | Final initialization | Diagnostic status | Commands | Final validity | Elapsed s |
| --- | ---: | --- | --- | ---: | --- | ---: |
| t01 | 4 | cold | passed | 8 | complete | 359.0 |
| t04 | 5 | cold | passed | 8 | complete | 289.9 |
| t06 | 6 | cold | passed | 8 | complete | 532.8 |
| t07 | 7 | cold | passed | 8 | complete | 395.6 |
| t10 | 4 | reused Kit | issues_found | 8 | unavailable | 256.4 |
| t15 | 5 | reused Kit | passed | 8 | complete | 225.3 |
| t18 | 6 | reused Kit | passed | 8 | complete | 315.5 |
| t22 | 7 | reused Kit | passed | 8 | complete | 216.3 |
| t28 | 4 | cold retry | passed | 8 | complete | 360.6 |
| t38 | 5 | cold retry | incomplete_coverage | 8 | partial | 858.9 |
| t40 | 6 | cold retry | passed | 8 | complete | 977.9 |
| t49 | 7 | cold | passed | 8 | complete | 369.8 |
| r-c1 | 4 | reused Kit | passed | 8 | complete | 190.3 |
| r-s1 | 5 | reused Kit | passed | 8 | complete | 184.3 |
| r-a1 | 6 | cold retry | passed | 8 | complete | 562.3 |
| iw_h1_02 | 7 | cold | issues_found | 8 | unavailable | 349.4 |
| iw_m1_01 | 4 | reused Kit | issues_found | 0 | none | 171.3 |
| iw_m4_02 | 5 | reused Kit | issues_found | 8 | unavailable | 207.3 |
| gf2-1 | 6 | reused Kit | passed | 8 | complete | 280.4 |
| gf3-1 | 7 | reused Kit | passed | 8 | complete | 207.3 |

## Remaining issues, retained without repair

| Task | Observation |
| --- | --- |
| T10 | All eight commands execute, but the revised mapping scorer cannot validate its observed-anchor receipt. The native source-region relation check is also false. Ten scoring boundaries are unavailable. |
| iw_m1_01 | begin_episode raises `RuntimeError: An unowned object moved during the authored use preview`. No fixed command executes and no diagnostic scoring boundary is recorded. |
| iw_h1_02 | All eight commands execute. `inspection_before_delivery` remains unknown; ten scoring boundaries are unavailable. This task starts in a fresh process after the cloth-profile restart, so this failure is also observed under cold initialization. |
| iw_m4_02 | All eight commands execute. Six `load_transport`/`delivery_transport` checks for item_01–item_03 remain unknown; ten scoring boundaries are unavailable. |
| T38 | The cold retry executes all eight commands and DONE without reported sequence/scoring errors, but final validity is partial. Mark incomplete coverage rather than complete success. |

Earlier full-debug artifacts already show nonpassing results for these aliases, although T10 previously stopped at an initial-qualification gate. This sample exposes its remaining observation-path failure after that gate is skipped. No missing observation is turned into a fabricated zero or successful measurement.

## Cold-start evidence

| Task | Reused-world failure | Independent cold result |
| --- | --- | --- |
| T28 | Native assertion: bookcase emitter local transform is not orthogonal. | 8 commands; complete final scoring; passed. |
| T38 | Vulkan device-memory allocation failures, followed by GPU page fault/crash. | 8 commands; no repeat of the recorded Vulkan crash; partial final scoring coverage. |
| T40 | Native owner exits before returning its response. | 8 commands, explicit phase transition and DONE; complete final scoring; passed. |
| R-A1 | Native owner exits before returning its response. | 8 commands; complete final scoring; passed. |

The decisive T38 log contains 48 `ERROR_OUT_OF_DEVICE_MEMORY` entries starting at 03:42:23 UTC, then a GPU page fault and crash at 03:42:54 UTC. Kernel logs independently record `NV_ERR_NO_MEMORY` at 11:42:22 Singapore time and Xid109 for PID2518909 at11:42:53. The two timestamp formats refer to the same incident. [The first Vulkan allocation failure](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/worker_01/02_t38/initialization_attempts/d09f88c6335041aaa29e3f2a16400b22/simulator.log:168) precedes the recorded crash.

NVIDIA identifies Xid109 as context-switch timeout; that label does not establish the underlying cause. [NVIDIA Xid catalog](https://docs.nvidia.com/deploy/xid-errors/archive/). The observed allocation failures establish device-memory pressure. Retained renderer resources, allocator fragmentation and the scene's own peak requirement have not been isolated; the fresh T38 process also approached20 GiB. Do not conclude that every failure is a memory leak or hardware defect.

Unrelated root-owned OpenPI processes appeared on GPUs4–7 after this sample started and consumed additional memory. No unrelated process was stopped. This resource competition limits conclusions about warm-versus-cold memory peaks and timing. It does not by itself prove those processes caused the GPU5 crash. The user explicitly authorized continuing with shared GPUs.

## Retained owners and artifacts

Final scene owners remain paused in `SCENE_RETAINED`, with their native initial state restored:

- GPU 4: iw_m1_01, PID 2564796, [control directory](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/worker_00/resident_000).
- GPU 5: iw_m4_02, PID 2575336, [control directory](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/worker_01/resident_000).
- GPU 6: gf2-1, PID 2626943, [control directory](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/worker_02/resident_000).
- GPU 7: gf3-1, PID 2579647, [control directory](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/worker_03/resident_000).

The completed monitor was PID2518663 in `codex-gpu:1.0`; the terminal returned to its shell. Resident GPU processes remain independent of that completed monitor. The final M1/M4 worlds are retained despite their diagnostic scoring problems, so residency must not be confused with a passing evaluator.

- [Machine-readable summary](/local_home2/zhouxinyang/models/WIA/docs/validation/formal_sample_20261101.json)
- [Selected task/shard plan](/local_home2/zhouxinyang/models/WIA/docs/validation/formal_sample_plan_20261101.json)
- [Original input identities and native profiles](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/input_receipt.json)
- [Full results and native artifacts](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/results.json)
- [Selected Vulkan/kernel crash evidence](/local_home2/zhouxinyang/models/WIA/outputs/formal_sample_20261101/gpu_crash_evidence.json)

All initialization_attempts directories retain their failed native evidence, while the main task directory contains the final cold attempt. Native GPU dumps remain in the output tree; no binary artifact is added to the documentation commit.

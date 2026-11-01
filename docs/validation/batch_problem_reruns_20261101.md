# Targeted batch repairs and T52 confirmation

The three outstanding cases are resolved. Combining the original 57 passing checks with the T06, R-C3 and amended T52 reruns gives passing fixed-action pipeline evidence for all 60 release tasks. This is cumulative coverage, not a new uninterrupted sixty-task sweep or a model-quality result. The original failures remain preserved. The [resolution receipt](/local_home2/zhouxinyang/models/WIA/outputs/batch_mcp_problem_reruns_20261101/resolution.json) maps every task to its accepted trial.

## Native results

Each trial uses the production HTTP/MCP runner, eight fixed commands followed by active DONE, seed 0, parameterized actions and an advancing world. All four checks completed without validation errors or model API calls.

| Trial | Physical GPU | Total wall time, including initialization | Final task progress | Evidence |
| --- | --- | --- | --- | --- |
| T06 cold retry | 7 | 517.6 s | Complete, 0% | [Trial](/local_home2/zhouxinyang/models/WIA/outputs/batch_mcp_problem_reruns_20261101/worker_00/00_t06) |
| R-C3 cold retry | 5 | 425.6 s | Complete, 0% | [Trial](/local_home2/zhouxinyang/models/WIA/outputs/batch_mcp_problem_reruns_20261101/worker_01/00_r-c3) |
| T52: inside declaration, later DONE | 5 | 416.9 s | Complete, 100% | [Trial](/local_home2/zhouxinyang/models/WIA/outputs/t52_confirmation_positive_20261101/00_t52) |
| T52: no declaration, active DONE | 7 | 362.8 s | Complete, 0% | [Trial](/local_home2/zhouxinyang/models/WIA/outputs/t52_confirmation_negative_20261101/00_t52) |

Zero task progress for T06/R-C3 is expected from this generic diagnostic sequence; it does not mean their pipeline checks failed. The previous native OOM failures did not recur in these cold trials. Resource contention remains a possible contributor, not an established sole cause. Other workloads shared the selected devices, so these timings are operational measurements rather than controlled latency comparisons.

## T52-only scoring contract

Per the user's instruction, T52 version 2 requires both an explicit standalone `ANSWER: inside` in `world_observation` and an active model-issued `DONE`, alongside measured stable, released containment. The declaration may precede DONE in a different decision of the same episode. Physical state alone, an absent or wrong declaration, or automatic finalization without active DONE cannot earn credit. Missing native evidence remains unavailable rather than becoming a measured zero. New episodes clear declarations; an interrupted segment can resume the same episode, but active DONE cannot be reopened.

The positive native trial declares inside at decision 1/action step 0 and issues DONE at decision 9/action step 8. All nine pre-DONE observation boundaries remain `pending_done`, without a numerical score. The terminal score is 100%. The negative trial has the same physically satisfied containment and active DONE but no inside declaration; its final score is 0%. Both retain sealed terminal records and separate Safety artifacts. Safety remains observational statistics with unverified coverage, not a calibrated safety pass.

The [task](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/release/tasks/t52.yaml), source-preserving builder and parameter-gated containment evaluator implement this contract. Source assets, geometry, seed and budget are unchanged. T15–T17 retain their contracts. An already-contained T52 does not require extraction or manipulation. The positive [sequence](/local_home2/zhouxinyang/models/WIA/configs/t52_confirmation_validation_20261101.json) supplies a fixed public declaration to exercise the protocol; it does not establish visual recognition.

## Validation and provenance

The final CPU scope passes 1297 checks, with 57 explicitly documented baseline failures excluded. Coverage includes separate-decision confirmation, no active DONE, wrong answers, failed or missing physical evidence, valid no-manipulation confirmation, episode reset and interrupted resume. Only the two active-DONE cases above were verified in the native simulator; resume was CPU-tested. See the [command and exclusions](/local_home2/zhouxinyang/models/WIA/outputs/t52_confirmation_positive_20261101/cpu_command.json) and [CPU log](/local_home2/zhouxinyang/models/WIA/outputs/t52_confirmation_positive_20261101/cpu.log).

R-C3 loaded a provisional diagnostic `stop_on_success=False` option before the user clarified the T52-specific contract. None of its measured boundaries was successful, so that option did not alter its executed trajectory or termination. The option has been removed from the final implementation; T06 and both amended T52 trials use normal termination. The [comparison receipt](/local_home2/zhouxinyang/models/WIA/outputs/batch_mcp_problem_reruns_20261101/comparison.json) records this difference and unchanged task/source hashes and evaluation settings for T06/R-C3.

The old-contract T52 retry was intentionally cancelled and retained as superseded. Its coordinator exit is not a failure of either amended-contract trial. A subsequent exclusive GPU 4 preflight was rejected when another job arrived before launch; no episode began. The positive and negative runs used authorized sharing on GPUs 5 and 7. Rebuildable texture caches totaling 46.23 GiB were removed only from verified stopped owners; raw experiment evidence remains. See the [cleanup receipt](/local_home2/zhouxinyang/models/WIA/outputs/batch_mcp_problem_reruns_20261101/cache_cleanup.json).

Positive task-solving trajectories, especially M4 grasp transport, and a real model benchmark remain separate work in [TODO](/local_home2/zhouxinyang/models/WIA/TODO.md). No commit or push was made.

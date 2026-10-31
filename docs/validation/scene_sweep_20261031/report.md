# Four-GPU fixed-sequence scene sweep

Source run: [scene_sweep_4gpu_20261031_212148](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/results.json). Code revision: `626a4e0`.
The agent launched this sweep in the preceding iteration and inspected its completed native artifacts. No new simulation or model/API call was launched for this status analysis.

All 60 catalogue tasks were attempted on physical GPUs 4, 5, 6 and 7, with 15 serial scenes per worker. Started 2026-10-31 21:21:52 Singapore time; finished 22:52:29; elapsed 90.61 minutes. All workers and owned native services have exited. Model calls: 0.

## Observed totals

| Measure | Count |
|---|---:|
| Raw passed | 19 |
| Raw issues_found | 37 |
| Raw runtime_error | 4 |
| Scenes completing all eight command attempts | 52 |
| Scenes not reaching the command sequence | 8 |
| Command attempts / completed / controller_error | 416 / 396 / 20 |
| Receipts reporting engine reuse | 42 |

Raw status is an infrastructure diagnostic label, not task success or a quality score. The script flags every non-complete score state, including legitimate phase-1 partial and pending_done states; therefore the 41 raw non-passes must not be reported as 41 broken tasks.

## Confirmed failures before sequence execution

| Task | Finding |
|---|---|
| t23 | Native preprocess rejects the source variant: storage_active_single_only_skips_variant_json. |
| t30 | Vulkan ERROR_OUT_OF_DEVICE_MEMORY followed by a segmentation fault. |
| t38 | Vulkan ERROR_OUT_OF_DEVICE_MEMORY followed by a segmentation fault. |
| t40 | Vulkan ERROR_OUT_OF_DEVICE_MEMORY followed by a segmentation fault. |
| t49 | Physics frequency mismatch: expected dt=1/240 s, observed dt=1/120 s. |
| t50 | Physics frequency mismatch: expected dt=1/240 s, observed dt=1/120 s. |
| r-c2 | 1200-second load/sequence watchdog expired before the sequence began; root cause unresolved. |
| iw_m1_01 | begin_episode raised: An unowned object moved during the authored use preview. |

The three Vulkan crashes occurred in resident services. This run alone does not distinguish accumulated renderer state from a scene that would also exceed memory in a fresh process. Likewise, the frequency mismatch identifies an incompatible timestep, not yet the precise lifecycle defect. Cold-process comparisons are needed before assigning those causes.

## Score availability and coverage

- T01, T02, T05 and T06 now pass all eight commands and have complete scores at all ten observation boundaries. This is new native confirmation of the prior scorer repairs.
- Fourteen tasks reject initial hand-release-related evidence: t22, t24, t25, t26, t28, t32, t33, r-c1, r-c3, r-s1, r-s2, r-s3, r-a1, r-a2. This shared pattern warrants checking the same detachment/aperture distinction fixed for T02. T22/T24/T25 also reject an explicit revision check; these must not be reduced to a single cause without review.
- T09/T11 have accepted Mapping initial conditions and no measurement_errors, but missing observed-anchor evidence makes scoring unavailable. T10 rejects initial native visit conditions; T21 rejects initial pose preservation.
- Six candidate tasks mark absent future event evidence unavailable: iw_h1_02, iw_h2_02, iw_h4_02, iw_h5_02, iw_m3_01, iw_m4_02. The sequence does not solve them; review whether not-yet-attempted conditions should be false rather than unavailable.
- iw_h3_03 and iw_m2_01 recover to complete scoring after the first action. Their raw non-pass reflects the initial record.
- T41 and T45–T48 remain in phase 1. Their 20 rejected manipulation attempts execute zero control steps; the fixed sequence does not exercise the temporal phase transition. T45–T48 remain partial. These are diagnostic protocol coverage limits, not proof of a controller defect.
- gf2-1/gf2-2 report pending_done. The diagnostic omitted DONE; their terminal scores have not been tested. This is a limitation of the diagnostic sequence.

## All task records

| Task | GPU | Raw status | Attempts | Final validity | Wall seconds | Finding |
|---|---:|---|---:|---|---:|---|
| [t01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/00_t01) | 4 | passed | 8 | complete | 403.7 | No issue found by this fixed-sequence diagnostic. |
| [t02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/00_t02) | 5 | passed | 8 | complete | 644.0 | No issue found by this fixed-sequence diagnostic. |
| [t04](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/00_t04) | 6 | passed | 8 | complete | 297.5 | No issue found by this fixed-sequence diagnostic. |
| [t05](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/00_t05) | 7 | passed | 8 | complete | 366.6 | No issue found by this fixed-sequence diagnostic. |
| [t06](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/01_t06) | 4 | passed | 8 | complete | 606.0 | No issue found by this fixed-sequence diagnostic. |
| [t07](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/01_t07) | 5 | passed | 8 | complete | 452.3 | No issue found by this fixed-sequence diagnostic. |
| [t08](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/01_t08) | 6 | passed | 8 | complete | 668.0 | No issue found by this fixed-sequence diagnostic. |
| [t09](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/01_t09) | 7 | issues_found | 8 | unavailable | 345.0 | Unobserved anchor completion evidence is unavailable despite accepted initial/native Mapping qualification; scoring semantics need review. |
| [t10](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/02_t10) | 4 | issues_found | 8 | unavailable | 436.9 | Evaluator begin_episode rejected native visit start; qualification details need further diagnosis. |
| [t11](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/02_t11) | 5 | issues_found | 8 | unavailable | 495.7 | Unobserved anchor completion evidence is unavailable despite accepted initial/native Mapping qualification; scoring semantics need review. |
| [t15](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/02_t15) | 6 | passed | 8 | complete | 304.7 | No issue found by this fixed-sequence diagnostic. |
| [t16](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/02_t16) | 7 | passed | 8 | complete | 347.8 | No issue found by this fixed-sequence diagnostic. |
| [t17](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/03_t17) | 4 | passed | 8 | complete | 294.7 | No issue found by this fixed-sequence diagnostic. |
| [t18](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/03_t18) | 5 | passed | 8 | complete | 234.9 | No issue found by this fixed-sequence diagnostic. |
| [t21](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/03_t21) | 6 | issues_found | 8 | unavailable | 457.7 | Initial native preconditions reject object_2_initial_pose_preserved. |
| [t22](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/03_t22) | 7 | issues_found | 8 | unavailable | 384.0 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t23](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/04_t23) | 4 | issues_found | 0 | not reached | 3.6 | Native preprocess rejects the source variant: storage_active_single_only_skips_variant_json. |
| [t24](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/04_t24) | 5 | issues_found | 8 | unavailable | 218.7 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t25](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/04_t25) | 6 | issues_found | 8 | unavailable | 202.3 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t26](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/04_t26) | 7 | issues_found | 8 | unavailable | 293.0 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t28](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/05_t28) | 4 | issues_found | 8 | unavailable | 304.9 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t30](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/05_t30) | 5 | runtime_error | 0 | not reached | 110.0 | Vulkan ERROR_OUT_OF_DEVICE_MEMORY followed by a segmentation fault. |
| [t32](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/05_t32) | 6 | issues_found | 8 | unavailable | 120.6 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t33](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/05_t33) | 7 | issues_found | 8 | unavailable | 382.8 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [t35](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/06_t35) | 4 | passed | 8 | complete | 197.3 | No issue found by this fixed-sequence diagnostic. |
| [t36](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/06_t36) | 5 | passed | 8 | complete | 291.4 | No issue found by this fixed-sequence diagnostic. |
| [t38](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/06_t38) | 6 | runtime_error | 0 | not reached | 118.8 | Vulkan ERROR_OUT_OF_DEVICE_MEMORY followed by a segmentation fault. |
| [t40](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/06_t40) | 7 | runtime_error | 0 | not reached | 299.7 | Vulkan ERROR_OUT_OF_DEVICE_MEMORY followed by a segmentation fault. |
| [t41](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/07_t41) | 4 | issues_found | 8 | unavailable | 213.3 | Temporal task remains in phase 1 with unavailable comparison/answer evidence; phase transition was not exercised. |
| [t42](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/07_t42) | 5 | passed | 8 | complete | 399.8 | No issue found by this fixed-sequence diagnostic. |
| [t43](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/07_t43) | 6 | passed | 8 | complete | 227.4 | No issue found by this fixed-sequence diagnostic. |
| [t44](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/07_t44) | 7 | passed | 8 | complete | 227.8 | No issue found by this fixed-sequence diagnostic. |
| [t45](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/08_t45) | 4 | issues_found | 8 | partial | 214.7 | Phase-1 partial score; four manipulation commands return controller_error with zero controller steps. Phase transition/DONE were not exercised; policy rejection is not a native controller crash. |
| [t46](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/08_t46) | 5 | issues_found | 8 | partial | 281.0 | Phase-1 partial score; four manipulation commands return controller_error with zero controller steps. Phase transition/DONE were not exercised; policy rejection is not a native controller crash. |
| [t47](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/08_t47) | 6 | issues_found | 8 | partial | 381.8 | Phase-1 partial score; four manipulation commands return controller_error with zero controller steps. Phase transition/DONE were not exercised; policy rejection is not a native controller crash. |
| [t48](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/08_t48) | 7 | issues_found | 8 | partial | 334.5 | Phase-1 partial score; four manipulation commands return controller_error with zero controller steps. Phase transition/DONE were not exercised; policy rejection is not a native controller crash. |
| [t49](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/09_t49) | 4 | issues_found | 0 | not reached | 6.6 | Physics frequency mismatch: expected dt=1/240 s, observed dt=1/120 s. |
| [t50](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/09_t50) | 5 | issues_found | 0 | not reached | 6.6 | Physics frequency mismatch: expected dt=1/240 s, observed dt=1/120 s. |
| [r-c1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/09_r-c1) | 6 | issues_found | 8 | unavailable | 256.8 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [r-c2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/09_r-c2) | 7 | runtime_error | 0 | not reached | 1231.6 | 1200-second load/sequence watchdog expired before the sequence began; root cause unresolved. |
| [r-c3](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/10_r-c3) | 4 | issues_found | 8 | unavailable | 451.6 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [r-s1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/10_r-s1) | 5 | issues_found | 8 | unavailable | 275.9 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [r-s2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/10_r-s2) | 6 | issues_found | 8 | unavailable | 247.8 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [r-s3](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/10_r-s3) | 7 | issues_found | 8 | unavailable | 409.6 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [r-a1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/11_r-a1) | 4 | issues_found | 8 | unavailable | 302.6 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [r-a2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/11_r-a2) | 5 | issues_found | 8 | unavailable | 308.1 | Initial hand-release qualification rejected; resembles the earlier aperture-versus-detachment issue. Other failed preconditions may also apply. |
| [iw_h1_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/11_iw_h1_02) | 6 | issues_found | 8 | unavailable | 308.5 | Untriggered delivery/inspection/transport event evidence is unavailable. Must distinguish an unfinished task from an unavailable measurement before formal scoring. |
| [iw_h2_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/11_iw_h2_02) | 7 | issues_found | 8 | unavailable | 222.8 | Untriggered delivery/inspection/transport event evidence is unavailable. Must distinguish an unfinished task from an unavailable measurement before formal scoring. |
| [iw_h3_03](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/12_iw_h3_03) | 4 | issues_found | 8 | complete | 235.9 | Initial score unavailable, but all subsequent nine records are complete; transient startup classification needs review. |
| [iw_h4_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/12_iw_h4_02) | 5 | issues_found | 8 | unavailable | 313.9 | Untriggered delivery/inspection/transport event evidence is unavailable. Must distinguish an unfinished task from an unavailable measurement before formal scoring. |
| [iw_h5_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/12_iw_h5_02) | 6 | issues_found | 8 | unavailable | 290.0 | Untriggered delivery/inspection/transport event evidence is unavailable. Must distinguish an unfinished task from an unavailable measurement before formal scoring. |
| [iw_m1_01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/12_iw_m1_01) | 7 | issues_found | 0 | not reached | 153.3 | begin_episode raised: An unowned object moved during the authored use preview. |
| [iw_m2_01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/13_iw_m2_01) | 4 | issues_found | 8 | complete | 258.4 | Initial score unavailable, but all subsequent nine records are complete; transient startup classification needs review. |
| [iw_m3_01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/13_iw_m3_01) | 5 | issues_found | 8 | unavailable | 260.8 | Untriggered delivery/inspection/transport event evidence is unavailable. Must distinguish an unfinished task from an unavailable measurement before formal scoring. |
| [iw_m4_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/13_iw_m4_02) | 6 | issues_found | 8 | unavailable | 296.4 | Untriggered delivery/inspection/transport event evidence is unavailable. Must distinguish an unfinished task from an unavailable measurement before formal scoring. |
| [gf2-1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/13_gf2-1) | 7 | issues_found | 8 | pending_done | 173.5 | pending_done at every boundary; this diagnostic never submitted DONE. Terminal scoring is untested, not a confirmed task failure. |
| [gf2-2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_00/14_gf2-2) | 4 | issues_found | 8 | pending_done | 247.7 | pending_done at every boundary; this diagnostic never submitted DONE. Terminal scoring is untested, not a confirmed task failure. |
| [gf3-1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_01/14_gf3-1) | 5 | passed | 8 | complete | 258.4 | No issue found by this fixed-sequence diagnostic. |
| [gf3-2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_02/14_gf3-2) | 6 | passed | 8 | complete | 263.4 | No issue found by this fixed-sequence diagnostic. |
| [t52](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/worker_03/14_t52) | 7 | passed | 8 | complete | 256.2 | No issue found by this fixed-sequence diagnostic. |

## Operational state and next checks

The output tree occupies approximately 249 GiB, primarily simulation caches; the local filesystem had approximately 235 GiB free at inspection. GPU 4–7 each retain only 19–20 MiB with no sweep compute owners. Original logs, initialization failures and receipts are preserved.

Prioritize: repair/retest the shared initial hand-release gates; compare T30/T38/T40 and T49/T50 in fresh versus reused services; resolve T23 source preprocessing and the r-c2 stall; then add valid terminal/temporal protocol coverage and distinguish unattempted conditions from measurement failures. No task/scorer code or original result was changed during this analysis.

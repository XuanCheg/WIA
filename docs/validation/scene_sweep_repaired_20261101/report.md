# Repaired four-GPU scene sweep: completed results

Current run: [scene_sweep_4gpu_repaired_20261031_235326](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/results.json). Baseline: [scene_sweep_4gpu_20261031_212148](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_20261031_212148/results.json). Current code revision: `b14daea`.
This agent launched the experiment in the prior iteration and inspected its completed native artifacts in this iteration. No new GPU workload, model/API request, code repair, rescore or deletion was performed for this analysis.

## Completion and comparison

All 60 tasks were attempted on GPUs 4–7, 15 per worker. Started October 31 at 23:53:31 and finished November 1 at 01:52:09 Singapore time; elapsed 118.64 minutes. All workers and native services exited. GPU 4–7 memory at inspection: 19–20 MiB; no remaining sweep owners. Model calls: 0.

| Raw diagnostic status | Baseline | Repeat |
|---|---:|---:|
| passed | 19 | 40 |
| issues_found | 37 | 13 |
| runtime_error | 4 | 6 |
| incomplete_coverage | 0 | 1 |

The raw pass count changed from 19 to 40, but the repeat also added phase transitions/DONE and distinguished pending coverage. This is not an isolated quality or speed ablation. Passed means that this infrastructure sequence and score-availability checks passed; it does not mean that the Agent solved the task.

- Fifty-two scenes completed all eight commands; all 416 commands completed successfully. The baseline had 396 completed and 20 phase-1 controller rejections.
- Final scoring: 42 complete, 9 unavailable, 1 partial, and 8 scenes never reached the command sequence. Two of the 42 complete final scores have an initially unavailable record and therefore remain raw issues_found.
- Twenty-three baseline non-passes now pass: t23, t24, t25, t26, t30, t32, t33, t40, t41, t45, t46, t47, t48, r-c1, r-c2, r-c3, r-s1, r-s2, r-s3, r-a1, r-a2, gf2-1, gf2-2.
- Two baseline passes regressed: t06, t36. They must not be omitted from the outcome.

## Repairs supported by new native evidence

- T23 now loads its declared variant and completes the sequence with complete scoring.
- T24/T25, T26/T32/T33, and the seven previously hand-gated reflection tasks pass the repaired initial-hand predicates. T22/T28 stop during camera setup, so their scoring changes remain unverified in this run.
- T41 and T45–T48 execute the public phase transition, all eight commands and DONE, with complete final scoring. gf2-1/gf2-2 now complete terminal scoring after DONE.
- T30, T40 and r-c2 pass; T38 reaches and completes its sequence with a partial score. This does not prove renderer stability: other tasks still crash or hang.
- Logs for T49/T50 confirm a preserved application with the requested 1/240 s physics and 1/60 s rendering/action timesteps. The previous frequency mismatch disappeared, but both tasks still fail later.

## Remaining native execution failures

| Task | Observed failure |
|---|---|
| t06 | GPU crash during reused native scene loading; passed the baseline. No successful command sequence in this run. |
| t22 | 20-minute watchdog; 10-minute stack is blocked in configure_robot_cameras -> simulator.render. Initial-hand/revision repair was not reached. |
| t28 | 20-minute watchdog; 10-minute stack is blocked in configure_robot_cameras -> simulator.render. Scoring repair was not reached. |
| t36 | 20-minute watchdog; 10-minute stack is blocked in configure_robot_cameras -> simulator.render. Passed the baseline. |
| t49 | Native timing was reconciled to 240 Hz physics / 60 Hz rendering and action. Initialization then failed with a max() reduction on an empty tensor. |
| t50 | Native timing was reconciled to 240/60/60 Hz. Watchdog stack is blocked in interaction_setup._place_robot -> simulator.step. |
| iw_m1_01 | Authored-use preview still rejects movement of an unowned object before the sequence. |
| iw_m3_01 | GPU crash during loading; no sequence executed. |

The three camera timeouts have the same main-thread location: `isaacsim SimulationContext.render` -> `omnigibson Simulator.render` -> `esi_robot_cameras.configure_robot_cameras:336`. T50 instead stalls inside the native physics step during robot placement. These are observed blocking locations, not yet established underlying causes.
The four-scene retirement bound was exercised (recycles per GPU 4/5/6/7: 1/1/3/2), but did not eliminate GPU crashes or hangs. T06 failed on an early reused scene. Do not describe periodic recycling as a complete renderer fix or assert that every crash is out-of-memory; these two logs report GPU crash detection rather than the baseline explicit out-of-device-memory failure.

## Remaining scoring/coverage observations

- T09/T10/T11/T21 retain their prior anchor/start/initial-pose evidence problems.
- iw_h1_02/iw_h2_02/iw_h4_02/iw_h5_02/iw_m4_02 retain unavailable future-event evidence. The sequence does not solve these tasks; absent task events must be distinguished from missing measurement capabilities before changing the rubric.
- iw_h3_03 and iw_m2_01 are initially unavailable but recover by subsequent boundaries, including the final one.
- T38 finishes its commands with partial scoring; this is tracked separately as incomplete_coverage.

## All task records

| Task | GPU | Baseline | Repeat | Commands | Final validity | Seconds | Finding |
|---|---:|---|---|---:|---|---:|---|
| [t01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/00_t01) | 4 | passed | passed | 8 | complete | 362.0 | No issue found by the expanded fixed-sequence diagnostic. |
| [t02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/00_t02) | 5 | passed | passed | 8 | complete | 770.2 | No issue found by the expanded fixed-sequence diagnostic. |
| [t04](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/00_t04) | 6 | passed | passed | 8 | complete | 289.0 | No issue found by the expanded fixed-sequence diagnostic. |
| [t05](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/00_t05) | 7 | passed | passed | 8 | complete | 361.6 | No issue found by the expanded fixed-sequence diagnostic. |
| [t06](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/01_t06) | 4 | passed | runtime_error | 0 | not reached | 208.5 | GPU crash during reused native scene loading; passed the baseline. No successful command sequence in this run. |
| [t07](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/01_t07) | 5 | passed | passed | 8 | complete | 494.4 | No issue found by the expanded fixed-sequence diagnostic. |
| [t08](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/01_t08) | 6 | passed | passed | 8 | complete | 718.4 | No issue found by the expanded fixed-sequence diagnostic. |
| [t09](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/01_t09) | 7 | issues_found | issues_found | 8 | unavailable | 376.8 | Observed-anchor evidence remains unavailable; full sequence and DONE completed. |
| [t10](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/02_t10) | 4 | issues_found | issues_found | 8 | unavailable | 441.7 | native_visit_episode_start_rejected remains; full sequence and DONE completed. |
| [t11](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/02_t11) | 5 | issues_found | issues_found | 8 | unavailable | 578.1 | Observed-anchor evidence remains unavailable; full sequence and DONE completed. |
| [t15](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/02_t15) | 6 | passed | passed | 8 | complete | 282.4 | No issue found by the expanded fixed-sequence diagnostic. |
| [t16](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/02_t16) | 7 | passed | passed | 8 | complete | 335.7 | No issue found by the expanded fixed-sequence diagnostic. |
| [t17](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/03_t17) | 4 | passed | passed | 8 | complete | 277.4 | No issue found by the expanded fixed-sequence diagnostic. |
| [t18](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/03_t18) | 5 | passed | passed | 8 | complete | 269.8 | No issue found by the expanded fixed-sequence diagnostic. |
| [t21](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/03_t21) | 6 | issues_found | issues_found | 8 | unavailable | 596.3 | Initial native pose-preservation qualification remains unavailable; full sequence and DONE completed. |
| [t22](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/03_t22) | 7 | issues_found | runtime_error | 0 | not reached | 1231.2 | 20-minute watchdog; 10-minute stack is blocked in configure_robot_cameras -> simulator.render. Initial-hand/revision repair was not reached. |
| [t23](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/04_t23) | 4 | issues_found | passed | 8 | complete | 421.6 | No issue found by the expanded fixed-sequence diagnostic. |
| [t24](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/04_t24) | 5 | issues_found | passed | 8 | complete | 439.3 | No issue found by the expanded fixed-sequence diagnostic. |
| [t25](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/04_t25) | 6 | issues_found | passed | 8 | complete | 251.5 | No issue found by the expanded fixed-sequence diagnostic. |
| [t26](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/04_t26) | 7 | issues_found | passed | 8 | complete | 322.5 | No issue found by the expanded fixed-sequence diagnostic. |
| [t28](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/05_t28) | 4 | issues_found | runtime_error | 0 | not reached | 1231.1 | 20-minute watchdog; 10-minute stack is blocked in configure_robot_cameras -> simulator.render. Scoring repair was not reached. |
| [t30](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/05_t30) | 5 | runtime_error | passed | 8 | complete | 142.2 | No issue found by the expanded fixed-sequence diagnostic. |
| [t32](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/05_t32) | 6 | issues_found | passed | 8 | complete | 107.2 | No issue found by the expanded fixed-sequence diagnostic. |
| [t33](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/05_t33) | 7 | issues_found | passed | 8 | complete | 404.8 | No issue found by the expanded fixed-sequence diagnostic. |
| [t35](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/06_t35) | 4 | passed | passed | 8 | complete | 303.5 | No issue found by the expanded fixed-sequence diagnostic. |
| [t36](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/06_t36) | 5 | passed | runtime_error | 0 | not reached | 1231.2 | 20-minute watchdog; 10-minute stack is blocked in configure_robot_cameras -> simulator.render. Passed the baseline. |
| [t38](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/06_t38) | 6 | runtime_error | incomplete_coverage | 8 | partial | 407.6 | No renderer crash in this run; all eight commands and DONE completed. Final scoring is partial, so diagnostic coverage is incomplete. |
| [t40](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/06_t40) | 7 | runtime_error | passed | 8 | complete | 412.6 | No issue found by the expanded fixed-sequence diagnostic. |
| [t41](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/07_t41) | 4 | issues_found | passed | 8 | complete | 291.8 | No issue found by the expanded fixed-sequence diagnostic. |
| [t42](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/07_t42) | 5 | passed | passed | 8 | complete | 455.7 | No issue found by the expanded fixed-sequence diagnostic. |
| [t43](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/07_t43) | 6 | passed | passed | 8 | complete | 153.2 | No issue found by the expanded fixed-sequence diagnostic. |
| [t44](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/07_t44) | 7 | passed | passed | 8 | complete | 212.3 | No issue found by the expanded fixed-sequence diagnostic. |
| [t45](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/08_t45) | 4 | issues_found | passed | 8 | complete | 270.4 | No issue found by the expanded fixed-sequence diagnostic. |
| [t46](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/08_t46) | 5 | issues_found | passed | 8 | complete | 217.3 | No issue found by the expanded fixed-sequence diagnostic. |
| [t47](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/08_t47) | 6 | issues_found | passed | 8 | complete | 431.9 | No issue found by the expanded fixed-sequence diagnostic. |
| [t48](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/08_t48) | 7 | issues_found | passed | 8 | complete | 451.6 | No issue found by the expanded fixed-sequence diagnostic. |
| [t49](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/09_t49) | 4 | issues_found | issues_found | 0 | not reached | 362.7 | Native timing was reconciled to 240 Hz physics / 60 Hz rendering and action. Initialization then failed with a max() reduction on an empty tensor. |
| [t50](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/09_t50) | 5 | issues_found | runtime_error | 0 | not reached | 1203.5 | Native timing was reconciled to 240/60/60 Hz. Watchdog stack is blocked in interaction_setup._place_robot -> simulator.step. |
| [r-c1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/09_r-c1) | 6 | issues_found | passed | 8 | complete | 184.3 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-c2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/09_r-c2) | 7 | runtime_error | passed | 8 | complete | 192.9 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-c3](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/10_r-c3) | 4 | issues_found | passed | 8 | complete | 454.7 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-s1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/10_r-s1) | 5 | issues_found | passed | 8 | complete | 253.9 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-s2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/10_r-s2) | 6 | issues_found | passed | 8 | complete | 155.6 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-s3](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/10_r-s3) | 7 | issues_found | passed | 8 | complete | 327.7 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-a1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/11_r-a1) | 4 | issues_found | passed | 8 | complete | 360.5 | No issue found by the expanded fixed-sequence diagnostic. |
| [r-a2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/11_r-a2) | 5 | issues_found | passed | 8 | complete | 275.6 | No issue found by the expanded fixed-sequence diagnostic. |
| [iw_h1_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/11_iw_h1_02) | 6 | issues_found | issues_found | 8 | unavailable | 247.4 | Future inspection/delivery event evidence remains unavailable despite the completed control sequence. |
| [iw_h2_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/11_iw_h2_02) | 7 | issues_found | issues_found | 8 | unavailable | 289.4 | Future inspection/delivery event evidence remains unavailable despite the completed control sequence. |
| [iw_h3_03](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/12_iw_h3_03) | 4 | issues_found | issues_found | 8 | complete | 284.4 | Only the initial record is unavailable; subsequent records and final scoring are complete. |
| [iw_h4_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/12_iw_h4_02) | 5 | issues_found | issues_found | 8 | unavailable | 263.6 | Approach/withdrawal event evidence remains unavailable despite the completed control sequence. |
| [iw_h5_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/12_iw_h5_02) | 6 | issues_found | issues_found | 8 | unavailable | 260.9 | Destination-observation-before-release evidence remains unavailable despite the completed control sequence. |
| [iw_m1_01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/12_iw_m1_01) | 7 | issues_found | issues_found | 0 | not reached | 222.0 | Authored-use preview still rejects movement of an unowned object before the sequence. |
| [iw_m2_01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/13_iw_m2_01) | 4 | issues_found | issues_found | 8 | complete | 334.5 | Only the initial record is unavailable; subsequent records and final scoring are complete. |
| [iw_m3_01](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/13_iw_m3_01) | 5 | issues_found | runtime_error | 0 | not reached | 220.4 | GPU crash during loading; no sequence executed. |
| [iw_m4_02](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/13_iw_m4_02) | 6 | issues_found | issues_found | 8 | unavailable | 207.5 | Transport-event evidence remains unavailable despite the completed control sequence. |
| [gf2-1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/13_gf2-1) | 7 | issues_found | passed | 8 | complete | 288.4 | No issue found by the expanded fixed-sequence diagnostic. |
| [gf2-2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_00/14_gf2-2) | 4 | issues_found | passed | 8 | complete | 323.0 | No issue found by the expanded fixed-sequence diagnostic. |
| [gf3-1](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_01/14_gf3-1) | 5 | passed | passed | 8 | complete | 295.5 | No issue found by the expanded fixed-sequence diagnostic. |
| [gf3-2](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_02/14_gf3-2) | 6 | passed | passed | 8 | complete | 259.8 | No issue found by the expanded fixed-sequence diagnostic. |
| [t52](/local_home2/zhouxinyang/models/WIA/outputs/scene_sweep_4gpu_repaired_20261031_235326/worker_03/14_t52) | 7 | passed | passed | 8 | complete | 311.5 | No issue found by the expanded fixed-sequence diagnostic. |

## Recommended next isolation

Prioritize a matched fresh-process versus reused-process comparison for T06/T22/T28/T36/iw_m3_01 using the same scenes, seeds, source configuration and commands. Retain the camera/physics stack dumps and GPU crash artifacts. Diagnose T49 empty tensor geometry independently of timestep selection, and T50 placement stepping independently of camera initialization. These tests have not been launched by the present inspection.

The current output tree is approximately 139 GiB; local free space is approximately 320 GiB. The prior sweep logs/observations/scores remain retained; only its inactive texture caches were removed in the earlier cleanup. No additional cleanup occurred in this inspection.

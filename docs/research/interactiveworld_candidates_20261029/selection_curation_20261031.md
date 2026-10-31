# Unified selection and variant reduction — 2026-10-31

The subsequent balance revision contains **59 active definitions**: 46 retained original tasks, nine interactive-world mechanism representatives, and four household geometry tasks. Ten original instances moved to a reversible extension set; their definitions, screenshots and reviews remain preserved. Family counts are AP 11, NT 8, GF 8, SR 9, GG 15 and TM 8. All four household tasks passed native initialization, visual review, complete-shape fit checks and bounded static access screening. Inclusion does not certify physical completion. See the [balance profile](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/balance_20261031/balance_profile.json), [per-task matrix](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/balance_20261031/task_matrix.csv), and [extension restoration manifest](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/balance_20261031/extension_tasks.json). Difficulty levels are authored estimates; no empirical calibration or runtime sampling change is claimed.

The four additions use ordinary objects with different geometric decisions, without extra cosmetic variants:

| Alias | Subclass | Daily-life task | Geometric distinction |
|---|---|---|---|
| GF2-1 | GF-2 | Lay a juice carton into a low pantry cubby | The 105 mm upright carton exceeds the 87 mm opening; pitching it makes the full body fit. |
| GF2-2 | GF-2 | Turn a cutting board into a narrow drying slot | Rotate an upright broadside board to align its thin dimension with the mouth. |
| GF3-1 | GF-3 | Match cucumber lengths to produce compartments | Compare 120/160/200 mm objects and assign all three to different usable lengths. |
| GF3-2 | GF-3 | Match spice-jar heights to shelf compartments | Match 90/135 mm jars to low/high shelves while keeping them upright. |

All seven targets have initial pickup witnesses and full-shape fit proposals using current native measurements. Two GF-2 queries additionally check aligned handling poses wholly before the mouth; five GF-3 queries check legal hypothetical final poses. The screen includes connected base stances, four torso joints, six working-arm joints, joint limits, native furniture proxies and sampled collision checks. The larger original carton was reduced after a gripper-clearance failure; both size assignments start near incorrect destinations to avoid a proximity shortcut. For the tall spice jar, a legal front placement and an outside-shelf arm waypoint resolve a direct-interpolation roof collision. No obstacle or scoring rule was relaxed.

Final validation: 59 selected definitions loaded, zero definition errors, and 59 mapped captures. The live review API has the exact same 59 aliases, with all four new scenes rendered and unreviewed. Existing review data is byte-identical to the pre-balance snapshot, including notes. Obsolete recaptured PNGs were removed only after their native process exited; historical measurements remain. Core household/dimensional checks passed 193 tests, diagnostics passed 55, shared-GPU launch/rechecks passed 27, and the final household capture-integration run passed 20. A broader legacy mirror suite still has 18 independently reproduced module-identity fixture failures, recorded separately; no all-suite pass is claimed.

The [native/static evidence](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/balance_20261031/household_validation.json) and [final definition/capture scan](/local_home2/zhouxinyang/models/WIA/outputs/dataset_balance_20261031/scan_final/report.json) have different scopes. No policy rollout, grasp closure, carrying, insertion, release or complete physical reference execution was performed for these four tasks. The new static checks do not requalify all older tasks: T22/T46 access uncertainty and the other validity concerns in the balance profile remain explicit. The final carton world is retained on physical GPU 7, with sharing authorized only after a fresh 14,336 MiB free-memory check; other users' processes were not terminated.

The remainder records the preceding 65-task selection and its validation. That selection contained 56 original selections and nine new mechanism representatives. Five D-series tasks were temporarily excluded at the user's request and remain excluded. Ten redundant new variants remain archived for controlled comparisons; they are not additional active tasks.

| Mechanism | Active alias | Class | Retention reason |
|---|---|---|---|
| H1 | iw_h1_02 | GG-4 | Keep the rejection branch; archive the acceptable-original control. |
| H2 | iw_h2_02 | AP-4 | Keep the disturbance branch; archive the no-disturbance control. |
| H3 | iw_h3_03 | AP-4 | Keep reversed progress; archive the static-deficit and already-complete controls. |
| H4 | iw_h4_02 | SR-1 | Keep the central-blocker layout; archive the alternative blocker layout. |
| H5 | iw_h5_02 | GG-4 | Keep the unavailable-primary branch; archive the available-primary control. |
| M1 | iw_m1_01 | TM-1 | Keep one tool identity; archive the matched identity-swap control. |
| M2 | iw_m2_01 | AP-5 | Keep orders 4 and 3: largest-first allocation to order A can fail; archive the 5-and-3 quantity change. |
| M3 | iw_m3_01 | GG-4 | Keep one content assignment; archive the matched contents-swap control. |
| M4 | iw_m4_02 | AP-5 | Keep the 24-second station-B job; archive the 8-second duration control. |

AP-5 (Resource-constrained Planning) is a new subclass inside the existing AP family. Coupled allocation and exclusive-resource scheduling require choosing a plan; neither is merely executing a prescribed order. H2/H3 provide AP-4 recovery coverage. H4 is geometric placement/clearing; H1/H5/M3 use observed item or destination evidence; M1 depends on observed use history. Primary assignments are provisional and do not establish construct validity.

The nine retained variants cannot alone demonstrate counterfactual decision sensitivity: an always-replace, always-fallback, or fixed-identity policy may exploit the reduced selection. Re-enable the archived controls for paired scientific evaluation. This reduction changes task selection, not the nine evaluator contracts, sensor inputs, timers, or physical definitions. Existing native qualification limitations remain.

The unified catalogue is authoritative for active scans; the candidate catalogue is explicitly a historical 19-variant pool with an active_representatives list. Original paths and aliases stay stable so captures and review notes remain traceable. Review defaults to all active tasks and groups both origins under the same taxonomy. Source provenance can still be filtered.

- [Active catalogue](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/release/catalogue.json)
- [Archived selections and restoration metadata](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/curation_20261031/archived_selections.json)
- [Pre-change inputs](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/curation_20261031/parents.json)
- [Read-only 65-task definition scan](/local_home2/zhouxinyang/models/WIA/outputs/review_repairs_20261031/unified_scan/report.json)

Validation: all 65 selected definitions loaded, with zero definition errors. The local review API returned 65 total / 56 original / 9 new, with the exact same alias set as the active catalogue. 103 focused publisher/capture integration checks passed. Review JSON was byte-identical before and after selection publication; separate T22 recapture subsequently reset only that decision while preserving its note. This validation does not certify grasp or task success.

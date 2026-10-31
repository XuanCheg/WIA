# Candidate implementation plan — 2026-10-29

Baseline: WIA b7101de; InteractiveWorld 40957c7243aa38ea8550d929dd3f10e99d8733d6.

## Scope and decisions

Implement nine candidate mechanisms outside the frozen release. User update supersedes the original validation requirement: native scene loading, screenshots and geometric reachability auditing are authorized; no policy or reference robot rollout in this iteration. A geometric witness is not grasp success. No new commit or push.

Main agent read the design package, critical TaskPack world/goal/interaction definitions, symbolic pour/cook implementation, native and released scoring reduction, WIA layout/loader/evaluation/capture paths. Source Cook declares a predicate; pouring declares an inside relation; late water-film collision is disabled. These do not validate thermal, composition or cleanup physics.

Approved decisions: H1 rigid packages with hidden acceptance mark; M1 retain cleaning tools and actual visible use context (supersedes the earlier container suggestion); M4 simulation time advances with physical execution and pauses during model waits. Read current state in H3, use minimum stock of two without gratuitous additions. Keep source scene models, native background furniture and source object models where compatible; record explicit compact placement changes. No taxonomy assignment or release admission.

| ID | Source / scene | Preserved and changed | Evidence and decision | Physical endpoints / paired factor | Dependencies and risks |
|---|---|---|---|---|---|
| H1 | GUIDE-87017 / restaurant_hotel | Preparation, supply, service structure; approved rigid-package replacement | Backside acceptance mark; retain vs replace | Delivery and quarantine; defect absent/present | Mark visibility and attached visual geometry |
| H2 | ORCHESTRATE-112022 / restaurant_hotel | Workspace and delivery; rigid obstruction replaces spill | Completed readiness becomes false after delivery | Clear region, delivery preserved, obstruction stored; disturbance/no disturbance | Event ordering and post-event observation |
| H3 | ASK-99600 / restaurant_asian | Matching mugs and stock/return supports; compact minimum-two inventory | Current count after retained/reversed progress | Stock and preservation; deficit/complete/undone | No arbitrary replacement IDs or action-attempt counting |
| H4 | NARRATE-12802 / school_chemistry | Scale, bottles, fixed anchors, bounded work region | Current footprint interference; selective clearance | Full target footprint, legal stored blockers; blocker position changes | Reachability and approach/withdrawal remain distinct from on-table |
| H5 | ASK-30000 / grocery_store_cafe | Cat-food stock, primary and fallback receiving stations | Public procedural no-resident rule | Preferred eligible placement and resident preservation; occupancy changes | Do not claim geometric infeasibility for procedural exclusion |
| M1 | WATCH-28002 / hotel_suite_large | Matching cleaning tools, desk, clean/used destinations | Observed cleaning contact changes eligibility | Physical segregation; used identity reverses with matched final layout | Event delivery and identity continuity; no contamination claim |
| M2 | GUIDE-24003 / grocery_store_asian | Sealed units, quantity evidence, receiving destinations | Coupled indivisible allocation | Quantities 3,2,2,1; orders (4,3) vs (5,3), all feasible subsets | Enumerate all assignments; labelled units are not package geometry |
| M3 | GUIDE-14018 / restaurant_asian | Matching fridge models, milk/chicken/cheese | Contents determine suitable storage | Actual inside placement, closed doors, preservation; contents swap | Door access and interior support; no cooling claim |
| M4 | ORCHESTRATE-44001 / restaurant_diner | Preparation, processing and delivery context | Public durations and station exclusivity | Timed supported occupancy then delivery; one duration changes | Simulation clock, robot handling, enumeration; no cooking claim |

## Execution and verification

1. Finish one native candidate loading/capture path before broadening layouts. Reuse WIA category initializers and native capture queue; author explicit candidate sources/tasks/scenes.
2. Implement pure physical/process reducers with targeted counterexample tests. Three authorized GPT-6.1-SOL/xhigh workers have disjoint evaluator and geometric-audit ownership. Main owns scene decisions, native integration, review and launches.
3. Capture every candidate variant through the installed behavior environment, inspect images and initial support/geometry, correct problems, then audit all manipulative objects and destination approaches using measured inventory.
4. Reachability uses native calibrated R1 kinematics, limits, orientation-aware IK, conservative collision and approach/retract checks, and base access. Report unresolved checks without declaring infeasibility or successful grasp.
5. Append immutable candidate captures after existing jobs in the original queue. Original 61/61 captures are complete and workers already exited at inspection; no suspension has occurred. Use physical GPUs 0–3 while idle, avoid other owners on 4/5/7, recheck before launches. Temporary extra-GPU processes must be terminated after use.
6. Report separate CPU contracts, native load/render, geometric witnesses, and deferred robot execution. Keep all limitations explicit.

## User-expanded scope and audit protocol

The later user instruction authorizes repairing the working baseline definitions as well as adding candidates. The combined review scope is 61 baseline selections and 19 candidate variants representing nine mechanisms; it is not 80 independent mechanisms. Preserve baseline commit b7101de and per-change provenance. Candidate admission remains pending. No model, policy, or reference-controller manipulation runs are authorized in this iteration.

AP-4 means Closed-loop Recovery & Adaptation. H2 is the primary proposed coverage: an explicitly authored reversible post-delivery deviation, an opportunity to observe it, and physical restoration while preserving delivery. H3 supplies net-progress revision. Do not add redundant tasks merely to increase category count. Future controlled diagnosis should compare feedback-enabled and feedback-limited conditions with matched geometry, event, action/time budget and policy; this iteration implements and reviews the task conditions, not that agent experiment.

Full review proceeds through definition/source consistency, current native captures, all bound-object accounting, both phases of temporal scenes, calibrated static arm/base screening, and issue-specific simulator rechecks. Floor bindings are structural geometry, hidden/inactive actors remain accounted separately, and cloth/articulated bodies do not receive rigid grasp certificates. Static screening uses continuous four-joint torso and six-joint working-arm IK, both arms, joint limits, multiple top/side orientations and grasp heights, open gripper geometry, 6 mm obstacle margin, sampled approach/retract joint paths, and a 16-heading swept rectangular base lattice. A found witness does not certify closure, carrying, controller tracking, or real grasp success. A bounded search failure is unresolved rather than an impossibility result.

After repairs and review, retain exact manifests/logs, delete superseded screenshot images only for modified tasks, and publish replacement jobs plus all new variants at the end of the original queue. Restore GPU 0–3 screenshot workers; temporary processes on 4–7 must finish or be explicitly cleaned up. Before every native launch, recheck actual utilization and process occupancy. The old queue was already complete and its workers had exited; none was suspended.

# InteractiveWorld task candidates for WIA

Research design review, 2026-10-29. Integration into V1 or a later release remains undecided. This package adds documents only and does not change WIA v0.1.

## Current approach

Start with the scientific question, redesign the suitable InteractiveWorld scene for robot action, and decide classification afterward. Existing WIA is an overlap baseline, not a framework that every candidate must fit.

## Read first

- [Scene redesign plan and candidate table](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/scene_redesign_plan.md): what to preserve, what to modify, physical outcomes and counterfactuals. This is the current organizing plan.

- [Main-reviewer decisions](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/reviewer_decisions.md): independent judgments on 27 deep candidates, duplicates and research priorities.
- [Eight robot task prototypes](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/robot_task_prototypes.md): goals, knowledge, decisions, physical outcomes, counterfactuals, WIA differences and difficulty.
- [Source and validity audit](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/source_audit.md): paper/code relationship, scoring layers and limits of environment reuse.

## Complete coverage

- [All 120 tasks, readable table](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/screening_all_120.md)
- [All 120 tasks, CSV](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/screening_all_120.csv)
- [Accessibility and prevention: nine deep reviews](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/deep_accessibility_prevention.md)
- [Guidance and planning: nine deep reviews](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/deep_guidance_planning.md)
- [Decision and finding: nine deep reviews](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/deep_decision_finding.md)

Initial screening yields 12 priority, 56 conditional, 35 duplicate and 17 weak source tasks. These are triage counts, not counts of independent additions. Main-reviewer decisions supersede provisional labels when the deeper evidence or required redesign changes the judgment. In particular, the conspicuous shoe-in-fridge task is not independently prioritized, and the fixed gap-clearing task becomes valuable only after an adaptive redesign.

## Main recommendation

Design compact probes for evidence-triggered replacement, recovery after a completed condition is invalidated, state-based resumption, adaptive space clearing and conditional destination revision. These can make world knowledge change the robot's plan and produce a measurable physical difference. Consolidate identical-object history tasks into one functional-identity family. Develop quantitative allocation as a shared-resource problem, and keep continuous-time scheduling conditional on a real clock and resource model.

Do not import ordinary remembered-location search, prescribed housekeeping orders, or arithmetic followed by moving an answer card as new headline capabilities. Do not interpret a symbolic cooked state or disappearing spill texture as verified material dynamics. Existing WIA already has history-conditioned manipulation and controlled physical trials; novelty must be shown against those concrete tasks.

- [Proposed diagnostic evaluation](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/evaluation_design.md): paired conditions, physical scoring, failure attribution and runtime cost.

## Provenance and status

- [Plan](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/research_plan.md)
- [Research log](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/research_log.md)
- [120 source goal and effective completion contracts](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/source_contract_inventory.json), including catalog fallbacks and source hashes.
- [61 current WIA task identities and hashes](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/wia_baseline_tasks.json)
- [Pre-existing workspace status](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/baseline_worktree_status.txt)

No simulation or model experiment was run. All feasibility estimates are static design estimates. The source results were not reproduced by this review. The user explicitly permits knowledge-guided real pick-and-place, with priority determined by how strongly knowledge is coupled to the physical task. Simple semantic selection is a lower-priority diagnostic, not a new capability by itself. Candidate scenes may be redesigned directly without assigning them to an existing family. See the scene redesign plan above.

# Independent Safety evaluator — 2026-11-01

## Result and scope

Task grading and Safety now have separate evaluators and separate outputs. Safety is a generic private statistics logger; its verdict, telemetry availability and logging failures do not participate in task completion, success, validity or quality aggregation. The Agent receives neither Safety results nor Safety logs.

This review covers **all 60 release tasks and 29 evaluator operations**, including T42–T44. The previous audit excluded those three because of the user's earlier explicit instruction; the current scope has no exclusions. T42–T44 share the controlled-ramp comparison evaluator. The [complete inventory](/local_home2/zhouxinyang/models/WIA/docs/validation/safety_separation_20261101_inventory.json) records every alias and entry operation.

This is code and CPU/native-fixture validation, not a fresh replay of all scenes. Existing task definitions, scenes, seeds, budgets, model/API accounting, `world=run`, action interlocks and the no-DONE terminal-scoring rule are retained. Generic Safety predicates have been removed from task grading; actual placement, release, withdrawal, identity, source-event and controlled-trial requirements remain.

## Evaluators and outputs

The generic evaluator is [SafetyEvaluator](/local_home2/zhouxinyang/models/WIA/src/wia/evaluation/safety.py), with schema `observational_safety_statistics_v4`. The old simulator import remains a compatibility shim. Runtime hooks sample native physics, mark action attribution and record real observation boundaries. The task evaluator never calls the Safety evaluator or reads its result.

| Output | Purpose |
| --- | --- |
| `private_evaluation/state.json` | Task evaluation only. |
| `final_evaluation.json` | Frozen task evaluation copy. |
| `episode/summary.json` | Task quality, termination and execution/API statistics; no Safety result. |
| `results.json` | Task outcomes and API/tool activity; no Safety result or Safety aggregate. |
| `private_safety/safety.json` | Current standalone Safety statistics, telemetry availability and exposure. |
| `private_safety/events.jsonl` | Individual private incidents/candidates with action attribution and measured evidence. |
| `private_safety/boundaries.jsonl` | Private Safety summaries stamped with episode, snapshot and action-step identifiers. |
| `final_safety.json` | Separate final Safety log copy. |
| `safety_results.json` | Batch aggregation of standalone Safety logs, with explicit missing-trial counts. |

The Safety log retains contact intervals, counterpart objects/categories, furniture/structure contacts, controller-interlock observations and drop/tipping/disturbance candidates. It also logs robot floor-support-loss/tilt candidates and authored apparatus-boundary exit candidates. Initialization events remain separate. These are observations, not automatically confirmed harm. The compatibility `safety_score` remains null; Safety does not supply a task-quality number.

Safety hook and file errors are contained within the logging path. A failed or malformed Safety log is reported as missing/unavailable Safety telemetry rather than a failed task. The final Safety copy and batch aggregation are also independent of task finalization. They cannot reuse an unrelated log when a trial has no output directory.

## Task-path changes

| Affected path | Removed Safety dependency | Task evidence retained |
| --- | --- | --- |
| H4 | Approach collision history and combined collision/withdrawal flags. | Actual end-effector separation from the destination and released target; target placement, blocker storage and anchors. |
| Shared temporal placement: T38, T40–T48, T50 | `robot_terminal_safe` and its floor/collision/upright measurement availability. | Real target/destination, dynamic collidable target, native gripper release, hand withdrawal, speeds and dwell. |
| Temporal source controller/navigation | Robot floor support, clearance and upright flags in aggregate source validity; contact-history fields in task results. | Source object/event integrity, read-only action contract, actual navigation continuity and task-specific protected inventory. |
| Mapping visits and composed mapping worlds | Generic contacts/support/upright conditions in occupancy and legacy Safety constraints in the parent. | Full chassis room footprint, segmentation layer, stopped occupancy, real visits/reports, source anchors and explicit task constraints. |
| Room collection | Generic robot initialization floor/upright/collision flags in the task initialization receipt. | Native tray grasp, inventory support/room identity, route/acquisition order and final supported contents. |
| Shared active layouts and dimensional qualification | Robot Safety measurements and verdicts in item-support acceptance; upright availability as a task gate. | Item support, fixed-mount geometry, native identities, dimensions, aperture, release and physical placement. |
| Containment and stack qualification | Generic robot contact/floor predicates. | Native context and actual object support/relations, identity and physics. |
| Cloth retrieval | Incidental contacts by other robot links and the generic cloth floor-height Safety bound. | Working-hand reveal, original geometry/physics, reveal-before-grasp and sustained real target lift. |
| T42–T44 controlled ramp trials | The old generic `safe_bounds_m` verdict in trial validity. Boundary exits now belong to Safety logging. | Measured release mark/orientation, unassisted trial physics, real observation duration, receiver outcome, matched unused spare selection and delivery. |
| Shared task envelope, Rigid grasp/lift, summary and batch aggregation | Embedded Safety evaluation calls and `safety`/`safety_score`/`safety_statistics` results. | The task's own evidence, completion formulas and execution accounting. |

Safety independence is tested with fixed task evidence. A collision can physically move a goal object out of its destination, in which case the changed physical goal state still affects task completion. That is different from a generic collision flag deducting points, rejecting an otherwise complete goal, or making task evidence unavailable.

The ramp's no-interference requirement remains a controlled-experiment condition: an externally pushed/assisted object does not establish the requested unassisted trial. The generic authored boundary envelope is no longer a task penalty. Task-specific requested preservation, valid native measurements and actual release/support are also retained.

## Separate example results

The following are **synthetic CPU fixtures**, not a native experiment or an Agent benchmark. The example uses a completed H4 goal. Changing only legacy collision flags leaves its task result unchanged; an independent furniture contact is logged separately.

- [Task result](/local_home2/zhouxinyang/models/WIA/docs/validation/safety_separation_example_20261101/task.json): complete validity, task success true, completion score 1.0 / 100%.
- [Safety log](/local_home2/zhouxinyang/models/WIA/docs/validation/safety_separation_example_20261101/safety.json): one furniture-contact interval, `mode=statistics_only`, `affects_scoring=false`, `passed_to_model=false`, no calibrated Safety score.

The task example does not contain the Safety log. The Safety example does not determine task success. Both artifacts explicitly record their synthetic provenance.

## Validation

[Safety independence regressions](/local_home2/zhouxinyang/models/WIA/tests/test_safety_independence.py) exercise:

- H4 collision/unknown-Safety perturbations with unchanged task results, plus an actual unwithdrawn-hand negative control.
- Shared temporal placement with true/false/missing Safety data, preserving release/withdrawal and missing-task-measurement negative controls.
- Actual room-footprint occupancy with unsupported/tilted robot states, and temporal source validity with a broken contact reader.
- Owner observation capture and a real R1 adapter action continuing after Safety hook/log errors, without adding Safety to Agent observations.
- Independent event/boundary/final/batch logs, malformed/missing Safety data and logging filesystem failure, with unchanged task results.
- Controlled-ramp completion independent of legacy Safety flags, while external interference still invalidates an unassisted trial.
- A valid working-hand cloth reveal and sustained grasp despite incidental forearm contact, unavailable incidental-contact data or the old Safety floor bound.

Expanded validation also covers existing native-shaped adapters, initial supports, composite task progression, mapping answers, containment/stack behavior, DONE/budget finalization, retained-boundary resume, HTTP/MCP transport, Agent observation privacy, evaluation entry and batch orchestration. The final selection reports **835 passed, 6 deselected in 37.72 s**, logged at `/tmp/wia_safety_final_20261101.log`.

Two source-generation assertions, for dimensional-size and metric-distance candidate generation, still expect an unconditional generated result where the converter returns `needs_human_decision`. Both failures reproduce against an isolated archive of the original `7f71ef7` source/tests. They are unrelated to this Safety change and remain excluded from the final regression selection. Four previously documented fake-session API-accounting assertions are also excluded; no API-accounting implementation was changed here.

No new GPU simulator run or model API call occurred. Existing resident simulator owners were not restarted; they must load the updated code on their next start to use this implementation. Earlier native results and audit counterexamples are preserved. Other non-Safety findings from the prior scorer audit are not claimed repaired by this iteration.

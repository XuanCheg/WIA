# Release scorer audit — 2026-11-01

Follow-up: the [independent Safety implementation](/local_home2/zhouxinyang/models/WIA/docs/validation/safety_separation_20261101.md) addresses F02 and includes all 60 release tasks, including T42–T44. The original audit results and counterexample artifacts below retain their historical scope and observations.

## Outcome and scope

Reviewed **57 configured release tasks, 16 entry modules and 28 distinct evaluator operations**, including nested composite members. T42–T44 are excluded as requested. The nested members add no new operation beyond those 28. Coverage includes task contracts, pure reducers, native evidence adapters, shared completion formulas, observation producers and terminal transport. This is a source/CPU audit, not 57 new simulator executions or a task-solving benchmark.

The scorers are **not all ready**. The main findings are measured incomplete actions becoming unavailable, indirect Safety gates, missing anchor observation production, an authored preview abort, and terminal-state handling gaps. Nine finding groups are detailed below. **F06 is repaired in this iteration following the user's explicit policy: without DONE, score actual terminal completion.** The other groups remain findings; passing existing tests does not resolve them.

The previous representative native sample is supporting evidence from an earlier iteration, not a new experiment by this audit: [20-task sample](/local_home2/zhouxinyang/models/WIA/docs/validation/formal_sample_20261101.md). Its statement that the user prohibited scorer repairs was incorrect and has been corrected: the no-repair instruction concerned Vulkan/driver/rendering changes.

- [All task and operation entries](/local_home2/zhouxinyang/models/WIA/docs/validation/scorer_audit_20261101_inventory.json)
- [CPU reproduction program](/local_home2/zhouxinyang/models/WIA/docs/validation/scorer_audit_20261101_repro.py)
- [Initial counterexamples before the terminal repair](/local_home2/zhouxinyang/models/WIA/docs/validation/scorer_audit_20261101_before.json)
- [Current counterexamples and terminal repair result](/local_home2/zhouxinyang/models/WIA/docs/validation/scorer_audit_20261101_cases.json)

## Findings

### F01 — Known unfinished obligations become unknown

Confirmed in source and CPU counterexamples; H1/M4 also reproduce in the previous native sample. With complete native object/clock/event channels and no required future action yet:

| Operation | Current unknown value | Required distinction |
| --- | --- | --- |
| H1 | `inspection_before_delivery` when no delivery exists | No delivery is an unfinished obligation. Missing observation-channel evidence is a different case. |
| H2 | Four post-delivery/event checks when the event list is empty | An observed empty event list establishes that the event has not happened. |
| H3 | `inventory_measurements` when the known initial dwell is shorter than the threshold | Insufficient accumulated dwell differs from a gap in the measurement channel. |
| H5 | `destination_observed_before_release` before any release | No release has occurred. |
| M1 | `observed_use` before a cleaning-use event | No use has occurred; do not infer that the event stream is missing. |
| M4 | Every `load_transport` and `delivery_transport` before the first episode | No transport has occurred. |
| H4 | Native approach/withdrawal process evidence before the first grasp | Initial incompleteness and generic collision statistics need separate treatment. |

Sources: [HIGH reducers](/local_home2/zhouxinyang/models/WIA/src/wia/candidates/high.py:173), [M1](/local_home2/zhouxinyang/models/WIA/src/wia/candidates/medium.py:230), [M4](/local_home2/zhouxinyang/models/WIA/src/wia/candidates/medium.py:476), [H4 producer](/local_home2/zhouxinyang/models/WIA/src/wia/candidates/native.py:54).

These values propagate to `evaluation_validity=unavailable`, including at termination. F06 cannot make them numeric by itself. A repair must distinguish an intact channel containing no event from absent/invalid native measurements; globally replacing `None` with `False` would hide real data failures.

The HIGH `_stable` helper returns `None` after exhausting a fully measured but short history, whereas MEDIUM's corresponding helper returns `False`. The H3 initial-frame case exposes this inconsistency. Actual intersample gaps must still remain unavailable.

### F02 — Safety is still coupled to task results indirectly

The shared envelope keeps the private Safety observer separate from the task score. Several underlying evaluators still use generic safety evidence to gate task success or measurement validity, contrary to the user's statistics-only policy:

- **H4:** `approach_clear`/`withdrawal_clear` include generic non-target arm collisions and participate in the success conjunction. A CPU case changes only `approach_clear`: success changes from true to false. Preserve the public goal that hands withdraw from the placed target; generic contact history must be private statistics.
- **Shared temporal placement:** `robot_terminal_safe` includes robot floor contact, collision clearance and upright state. Changing only this flag changes placement from true to false with the target's support, release, speed and withdrawal unchanged. This path is used by T38, T50 and physical temporal operations. [Placement measurement](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/temporal_understanding_agent_observation.py:1033).
- **Two-phase temporal tasks:** the controller's `valid` flag includes `robot_floor_support`, `robot_clear` and `robot_upright`; the physical scorer requires that aggregate validity. Generic contact can therefore make task evidence unavailable. [Measurement checks](/local_home2/zhouxinyang/models/WIA/src/wia/native/esi_temporal_phases.py:292), [physical source validity](/local_home2/zhouxinyang/models/WIA/src/wia/native/esi_temporal_phases.py:836).
- **Room collection:** the initial receipt aggregates floor support, upright and collision clearance. The result requires its aggregate `accepted` flag even with formal qualification disabled. [Initialization](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/wia_room_collection.py:484), [score dependency](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/wia_room_collection.py:565).
- **Composite mapping:** the parent recomputes `safe` from raw motion faults and legacy constraint flags. This can override a leaf that already declares Safety independent of task success. A CPU case changes only a non-floor contact and legacy Safety flags: parent success changes to false while the child's task-specific constraints remain true. [Composite aggregation](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/composite_task.py:206).

T49 also permanently invalidates its working-hand cloth sequence after other robot/cloth contacts. This requires a narrower contract review: distinguish a required manipulation method from incidental body-contact statistics. It is not evidence that every cloth contact condition should be deleted.

No direct private numerical-score or Safety payload leak was found in the reviewed Agent transport allowlists. That statement is limited to this code review and the terminal privacy tests, not a full adversarial security evaluation.

### F03 — T09–T11 consume an observation receipt with no production writer

The observed-anchor scorer reads `session.mapping_anchor_observations[source_path]`, validates episode/binding/public-label/RGB/snapshot evidence, and cannot establish observed-anchor relations without it. Repository searches find fixture writers but no production writer. The previous T10 native sample reaches all commands while all ten evaluation boundaries remain unavailable. [Receipt consumer](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/cognitive_mapping_visit_reports.py:666).

There is an existing private synchronized RGB/segmentation collector, but its output is not this receipt. It also requires real camera modalities and identity binding; mask presence alone is not proof of semantic recognition. A repair must connect actual delivered mounted-camera evidence to the required receipt and enable the needed private modality before capture. Do not insert an empty list, synthetic observation or unconditional acceptance. [Existing collector](/local_home2/zhouxinyang/models/WIA/src/wia/runtime/sim/esi_visual_evidence.py:303).

### F04 — M1 can abort before the Agent begins

The authored use preview calls `_preserved` for unowned objects. Both `False` and `None` raise the same message, `An unowned object moved during the authored use preview`. Consequently, the message alone does not establish whether an object moved or its measurement was missing. The previous native sample has zero executed commands and no usable evaluation boundary for M1. [Guard](/local_home2/zhouxinyang/models/WIA/src/wia/candidates/events.py:290).

This preview supplies required task history and differs from optional scene qualification. Preserve the physical demonstration and measured observation history. The next diagnosis needs object names, measured translation/rotation and availability, followed by an appropriate preview trajectory/settling fix. Freezing unowned objects or forcing the preservation result true would alter the task.

### F05 — T38 has no defined complete negative terminal result before its delayed query

Before the survey unlocks the delayed query, `_update` returns without updating delivery evidence. Survey/query/identity obligations can be known false while delivery remains unavailable, so the aggregate is partial. T38 has no native DONE-specific reducer that converts this known pre-query terminal state into an evaluated negative result. The previous eight-command diagnostic sequence never satisfies the survey condition; its partial final result is therefore explainable but not proof of a broken native placement sensor. [History update](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/wia_history_collection.py:162).

The terminal contract should evaluate the actually reached stage, with an explicit interpretation of unopened delayed goals. Do not invent the delayed query, claim a hidden answer was delivered, or mark genuinely missing placement evidence measured. The shared terminal transport repair does not resolve this task-specific policy gap.

### F06 — No-DONE terminal scoring: repaired

Previously, nine InteractiveWorld tasks and two GF2 tasks could remain pending when the Agent exhausted its budget without DONE. The canonical validator returned `None`; the summary then excluded the result. The user's policy is now explicit: **score actual completion whether or not DONE is declared**.

Implemented a runner-owned `/api/agent/finalize` transaction on the simulator owner thread. At termination it captures the actual stopped boundary without a robot action or physics step, evaluates native evidence, seals the result, and returns only a public acknowledgement and boundary identifiers. It does not report model DONE, choose an answer or trigger a temporal phase transition.

- `model_declared_done` remains false; `termination_reason` remains the actual reason, such as `max_steps`.
- Completed, partially completed and unfinished measured goals produce their actual numerical scores. Missing native measurements remain unavailable.
- GF2 uses the existing selection/passage/placement reduction on a copy of the pure state. The live simulator and resumable history are retained.
- Interrupted segments can reopen their non-DONE terminal cache on the existing resume path; a genuine DONE remains terminal.
- A successfully measured interrupted terminal state keeps its measured score, while `execution_validity=interrupted` and the existing exclusion from Agent-failure attribution remain separate statistics.
- If terminal acquisition fails, stale preterminal values cannot become the final score. No simulator wall-time deadline is introduced.

The updated transport must be loaded by a newly started simulator owner. Already resident owners retain their old Python code; this audit has not restarted them or modified active GPU workloads.

### F07 — Room collection can retain a stale perfect progress score after clock regression

If native time moves backward, `sample_step` appends `native_clock_regressed` and returns without clearing the old inventory. `record` still labels completion evidence complete based on the initial receipt. A CPU case first completes the task, then regresses the clock: completion remains **1.0 / complete** while task success becomes false. [Clock branch](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/wia_room_collection.py:579).

The score should no longer claim valid current evidence after the regression. This is a specific stale-evidence path, not a claim that all native measurement exceptions are silently swallowed.

### F08 — Some initial eligibility gates remain outside formal qualification

Ordered stacking, spatial trays, mirror tasks and dimensional reducers retain initial errors/eligibility gates, including initially incomplete stacks, targets initially in destinations, initially satisfied mirror spatial goals, and starting-side/geometry conditions. Disabling formal qualification does not disable these reducers' internal checks.

These are confirmed residual gates, but they require a contract decision per check. Required identity/geometry/baseline measurements must remain; debug-only difficulty eligibility should not silently make a formal result unavailable. This finding is not authorization to make all initially satisfied scenes pass or to delete all initial validation.

### F09 — M4's public timing description conflicts with `world=run`

M4 describes model waits as pausing the world, while the configured benchmark intentionally advances it during inference. The reducer uses native simulation time, so waiting can consume task time. This is a prompt/protocol mismatch, not evidence that the clock formula should change. Preserve `world=run` and the current thresholds; align the public description when this issue is addressed.

## Complete operation review matrix

“No additional defect identified” means the reviewed contracts/formulas do not add a finding here. It is not a certificate of native task solvability.

| Entry module / operation | Release aliases | Review conclusion |
| --- | --- | --- |
| action_order / ordered_grasp_lift_stack | T01, T02 | Final placement/order and released stability reviewed; F08. |
| cognitive_mapping_connectivity / answer_direct_adjacency_with_exploration | T04 | Visit evidence, answer normalization, unique-question denominator and physical obligations reviewed; no additional defect identified. |
| task_edit2_mapping_world / canonical_mapping_joint_world | T05, T06 | Nested visit/connectivity, answer-sheet aggregation and world identity reviewed; F02 at composite parent. |
| wia_room_collection / visit_rooms_and_collect_on_right_hand_tray | T07, T08 | Route order, pickup identity, supported inventory and dwell reviewed; F02, F07. |
| cognitive_mapping_regional_boundary / answer_observed_anchor_region_relation | T09, T10, T11 | Same-region/boundary truth and exposure evidence reviewed; F03. |
| perceptual_grounding_material_transparency / insert_or_verify_inside | T15, T16, T17, T52 | Exact target, native containment, release and stability reviewed; no additional defect identified. |
| physical_dynamics_stacking_stability / build_released_three_object_stack | T18, T21 | Support/contact, footprint, speeds and consecutive dwell reviewed; no additional defect identified. |
| rigid_partial_insertion / pick_release_partial_insert | T22–T25 | Native Inside/geometry and persistent sequential verification reviewed; historical credit is explicitly allowed by the public tasks. |
| wia_spatial_tray / retrieve_arrange_in_tray | T26, T28, T30, T32, T33 | Equilateral, isosceles, collinear, separated and touching branches reviewed; F08. |
| wia_room_collection / search_and_collect_on_right_hand_tray | T35, T36 | Discovery/acquisition and final supported inventory reviewed; F02, F07. |
| temporal_understanding_agent_observation / history_multi_retrieval_v1 | T38 | Survey, delayed query, identity and delivery progression reviewed; F02, F05. |
| temporal_understanding_two_phase / compare_change_and_restore_or_collect | T40, T41, T45–T48 | Removed, unchanged, added, moved, replaced and multiple-change operations reviewed; F02. |
| physical_structure_deformable / sweep_cloth_retrieve_item | T49 | Reveal/manipulation history, support and untouched objects reviewed; contact-policy issue noted under F02. |
| deformable_redrape / uncover_select_and_deliver | T50 | Reveal/select, stable release and actual target placement reviewed; shared F02. |
| wia_mirror / mirror_select_deliver | R-C1–R-C3 | Frozen optical selection and original-object placement reviewed; F08. |
| wia_mirror / mirror_reconfigure | R-S1–R-S3 | Farther, nearer and reversed initial distance-order inequalities reviewed; F08. |
| wia_mirror / classify_appearance_deliver | R-A1, R-A2 | Reflected/direct appearance report and original-object placement reviewed; F08. |
| interactiveworld_candidates / H1 | IW_H1_02 | Inspect-before-delivery and spares reviewed; F01, F06. |
| interactiveworld_candidates / H2 | IW_H2_02 | Delivery-linked check/disturbance and recovery reviewed; F01, F06. |
| interactiveworld_candidates / H3 | IW_H3_03 | Minimum inventory, authored removal, preservation and unnecessary additions reviewed; F01, F06. |
| interactiveworld_candidates / H4 | IW_H4_02 | Blocker storage, anchor preservation, placement and hand withdrawal reviewed; F01, F02, F06. |
| interactiveworld_candidates / H5 | IW_H5_02 | Observation/placement order and destination evidence reviewed; F01, F06. |
| interactiveworld_candidates / M1 | IW_M1_01 | Real use/contact, demonstration observation and delivery reviewed; F01, F04, F06. |
| interactiveworld_candidates / M2 | IW_M2_01 | Exact sealed-unit allocation, interchangeable valid subsets, stock and supported placement reviewed; F06. |
| interactiveworld_candidates / M3 | IW_M3_01 | Actual resident contents, eligible unit, native containment, equipment closure and payload placement reviewed; F06. |
| interactiveworld_candidates / M4 | IW_M4_02 | Per-job load/process/transport/delivery and native deadlines reviewed; F01, F06, F09. |
| wia_household_fit / select_pass_aperture | GF2-1, GF2-2 | Native mouth frame, real sampled passage, identity and stable compartment placement reviewed; F06, F08. |
| wia_household_fit / sort_intrinsic_dimension | GF3-1, GF3-2 | Native intrinsic dimensions, unique assignment and released compartment placement reviewed; F08. |

## Formulas and behavior intentionally retained

Shared completion uses declared obligation identities and their full denominator. Complete measured evidence permits a numerical completion fraction; unavailable items remain unavailable rather than being removed from the denominator. Constraint validity, physical success and a primary QA metric are distinct fields.

Semantic QA accuracy may be 100% while a physical exploration obligation is unfinished; the envelope preserves the declared semantic primary metric instead of replacing it with a mixed diagnostic fraction. Likewise, a completion fraction and task success need not be identical when an explicit task constraint fails. This does not justify the forbidden Safety dependencies in F02.

T22–T25 explicitly permit sequential verification and removal to reuse a container. Persistent verified-item credit is therefore intentional. Current final-state revisions elsewhere still check their actual terminal spatial relations and dwell; they do not silently add a historical grasp/lift requirement. Mirror farther/nearer inequalities have the appropriate opposite signs and reverse-order uses the frozen initial ordering.

No task thresholds, prompts, scenes, seeds, max-step fields, API retry policies, model-call accounting or `world=run` settings were changed. No Vulkan, driver or renderer repair was made.

## Validation and limits

All current audit validation is CPU-based, using synthetic native measurements and the real owner queue/HTTP/MCP paths where specified. No model API call or new simulator launch occurred.

1. Initial broad scorer baseline: **814 passed, 4 failed, 5 setup errors** across the selected reducer/adapter/integration modules. Three failures expect legacy `model_calls == decisions` from fake sessions; one depends on a historical gallery JSON deleted under the user's cleanup authorization. Five errors are legacy history-variant fixture/template module-schema incompatibilities. These are separate from the concrete scorer counterexamples.
2. The reproduction program initially records ten counterexamples. After F06, nine existing counterexamples remain, the additional H3 short-dwell case also reproduces, and the final case records a valid score without DONE. The before JSON is retained for comparison.
3. Terminal-focused validation: **256 passed, 4 deselected** in 17.30 s across DONE transport, dimensional adapter, household fit, runner, temporal phase transport and the evaluation entry. It covers 0/50/100% terminal progress, distinct DONE metadata, an actual MCP/HTTP owner transaction, privacy, idempotence, missing measurements, native/transport finalizer failures, and the same GF2 partial/full score with and without DONE. DONE and final-progress flags are also retained in episode records and evaluation summaries.
4. The four deselected runner/temporal tests were also run without deselection and fail because fake model sessions expose no API-attempt accounting. Loading the original `HEAD` runner implementation reproduces all four failures (**4 failed**, same `None` versus decision-count mismatch). They are not counted as passing checks or new terminal-scoring regressions.
5. The existing retained-boundary resume integration selection also passes: **3 passed, 9 deselected** in 2.56 s. No physical scene reset or action replay is introduced by reopening a non-DONE terminal cache.

Reproduce the counterexamples:

```bash
cd /local_home2/zhouxinyang/models/WIA
PYTHONPATH=src:tests:/local_home2/zhouxinyang/models/Show-Harness-Explore \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python \
  docs/validation/scorer_audit_20261101_repro.py
```

Temporary execution logs are `/tmp/wia_scorer_audit_baseline_20261101.log`, `/tmp/wia_terminal_scoring_20261101.log`, `/tmp/wia_terminal_baseline_compare_20261101.log`, `/tmp/wia_terminal_scoring_final_20261101.log` and `/tmp/wia_terminal_resume_20261101.log`. The checked-in JSONs and this report preserve the findings independently of those temporary files.

The next repair priority is F01/F02/F03, followed by the M1 preview and T38 terminal contract. F07 needs a small stale-state correction. F08 requires separating true measurement requirements from difficulty qualification; F09 is a public-description correction. Fresh native checks should target repaired paths rather than treating this static/CPU review as a successful full release replay.

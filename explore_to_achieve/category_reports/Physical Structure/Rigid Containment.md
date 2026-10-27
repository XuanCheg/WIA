# Rigid Containment

## Supplementary assessment comparison

Read-only review: 2026-10-14 21:15:07 UTC. Fingerprint: 94f19e59cb152a62763c267147e78f28ecb4c8beaeecf27f2e3023b045562cd8.

Review plan: compare the implementation spec and supplementary assessment with the current source builder, legacy conversion/task code, all available current and previous definitions, CPU contracts and CaptureQueue diagnostics. Separate original QA, initialization qualification, physical terminal evidence and historical review metadata. This reviewer ran CPU reads and contract checks only; no simulation, model episode or robot experiment was run.

**Conclusion:** the existing tasks request meaningful actual operations, but independent terminal certification is missing. Preserve previous work and continue Review and screenshots. Recommendations require Main's implementation review and create no capture or publication gate.

### Available work and missing campaign artifacts

[Implementation spec](</local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md>) permits preserved existing work and deferred incremental coverage. The category remains DEFERRED, revision 0. The requested core.esi_converters.physical_structure_rigid_containment module and assigned category directory are absent. Therefore no BatchManifest, CPU validation artifact or sample_evidence.json exists at /home/zhouxinyang/esi_category_conversion/20261014_v2/categories/bf79a025cc985aff3873; review.batch_file is null. No convert_all was run or inferred from other categories.

Current CPU inspection found 80 source JSON files: 20 base records and 60 derivatives. All 80 scene-tree JSON copies are byte-identical to the corpus; all 60 derivative metadata_json values match their bases. The [scene catalogue](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/scenes/catalog.json>) retains 92 entries covering all 80 source identities: 27 ready, 63 pending and two discarded. Conversion labels are 17 converted_and_native_validated, ten multi_object_manual_review, 59 not_started, two conversion_failed, two retained_for_review and two discarded_invalid_source. These are inherited states, not new execution measurements.

All 27 assigned existing_task_files match their provided SHA-256 values. They comprise 17 single-target, seven two-target and three three-target tasks. Every public objective matches its own source question verbatim. Current CPU load_task checks pass for all 27 with 60-step budgets. The 27 ready scene descriptors load; 63 pending and two discarded descriptors are rejected by the execution loader as intended. Every task target appears in declared support qualification. Execution eligibility must remain separate from diagnostic Review visibility.

### Historical assessment versus current semantics

The [assessment](</local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md>) has the supplied SHA-256 5bfdf6f3fb408bc7bd8aedd9873d6ddc5639dbff18237d20312ecc689275dd07. Its claims that the existing path is reusable and that actual insertion remains separate from construction are supported. Its recommendation to preserve dimensions is justified, but the historical outputs do not uniformly preserve them.

The [legacy converter](</local_home2/zhouxinyang/models/Show-Harness-Explore/scripts/convert_esi_rigid_catalog.py>) calls [size_target](</local_home2/zhouxinyang/models/Show-Harness-Explore/core/tasks/pickup_policy.py>), whose default screen shrinks the primary target uniformly to a minimum global extent of 0.08 m unless keep-target-scale is requested. It records source_answer_revalidated=false and source_answer_geometry_preserved=false when scaling changes. Six base layouts shrink their target; inherited multi-target tasks increase the affected formal-task count to ten:

| Base scene / room / sample | Source target | Multiplier |
| --- | --- | --- |
| Beechwood_1_int / playroom_0 / storage2_2 | snow globe | 0.883311 |
| Beechwood_1_int / television_room_0 / storage2_1 | granola box | 0.615608 |
| Merom_0_int / living_room_0 / storage2_2 | bowling ball | 0.364473 |
| grocery_store_half_stocked / grocery_store_0 / storage2_1 | gift box | 0.533333 |
| house_double_floor_lower / living_room_0 / storage2_0 | gift box | 0.427296 |
| restaurant_hotel / lobby_0 / storage2_1 | doll | 0.227594 |

Five of 13 extra targets are also smaller than their source asset scales; the other eight match exactly:

| Derivative task | Extra binding | Multiplier |
| --- | --- | --- |
| grocery_store_half_stocked_scene_dict_storage2_2_revised__criterion_2 | multi_extra1 | 0.517836 |
| house_double_floor_lower_scene_dict_storage2_0_revised__criterion_4 | multi_extra1 | 0.683795 |
| house_double_floor_lower_scene_dict_storage2_0_revised__criterion_4 | multi_extra2 | 0.609243 |
| restaurant_hotel_scene_dict_storage2_1_revised__criterion_4 | multi_extra1 | 0.353918 |
| restaurant_hotel_scene_dict_storage2_1_revised__criterion_4 | multi_extra2 | 0.347585 |

These are preserved historical recipes, not reviewer changes. They may change the smallest suitable container. No native fit-label reversal or physical insertion has been measured here. Main should mark the semantic difference and consider separate source-size revisions while preserving all previous tasks and review records.

The [tabletop builder](</local_home2/zhouxinyang/models/Show-Harness-Explore/core/tasks/tabletop.py>) independently repacks objects, adds support primitives and moves pickup targets. All 27 formal layouts change at least one source target/container pair distance by more than 0.01 m. Per-layout maximum changes range from 0.630088 to 11.932429 m, comparing source initial_poses with declared final object_pose patches. This is static pose evidence, not a new native measurement. These are L2 adaptations, not a common rigid transform preserving pairwise relations. Private source camera records remain intact but do not automatically depict the relocated scene.

The assessment references ESI commit f35b0de448ee77bc46714ae84b5190c17f87d86d; the local checkout and local capture receipts use 9f6f15ed43e81654a97de9e4db2a1fedadca2bc1. Current AST comparison verifies that all 15 shared pure QA functions match the official inline source exactly. This supports protocol reuse, not physical qualification.

### Meaningful operations and terminal predicates

The [source builder](</local_home2/zhouxinyang/models/Show-Harness-Explore/ESI-Bench/src/active_explore/tasks/physical_structure/rigid_containment.py>) creates only the primary containee and containers, restores initial poses and offsets the robot eye. It skips source filenames containing __. Its pick-up action directly moves the target, enables visual_only and creates a mock attachment. Put-inside samples onTop, checks a heuristic and repositions objects during retries. Source success flags therefore do not certify real R1 manipulation. Canonical construction with a separate derivative question is useful, but the primary-containee active scorer does not score every extra object.

The source _is_inside test uses container outer XY bounds with 0.05 m tolerance and target minimum Z below container maximum Z minus 0.05 m. It has no lower-Z bound, native cavity test, release requirement or velocity check. The original public rule allows stable partial insertion and rejects resting entirely on top. Native Inside may impose stricter geometry. Preserve original QA and document a separate native partial-insertion predicate; raw AABB overlap or an uncalibrated native Inside substitution is insufficient.

All 27 formal tasks declare answer_and_place, but [LoadedRigidTask](</local_home2/zhouxinyang/models/Show-Harness-Explore/core/tasks/loader.py>) reports physical_evaluator=not_configured. The [episode runner](</local_home2/zhouxinyang/models/Show-Harness-Explore/scripts/run_esi_explore_episode.py>) disables their physical evaluation feed, and the native runner omits declarative evaluator creation for this task class. The [placement preflight](</local_home2/zhouxinyang/models/Show-Harness-Explore/core/preflight/goals.py>) correctly reports INCONCLUSIVE and complete_goal_verified=false even when released-state probes exist. The gap affects single-target as well as multi-target formal tasks.

A future read-only evaluator should record initially unheld/outside state, native target-specific attachment and lift, guarded transport when required, actual destination, release without a constraint, calibrated partial cavity insertion, appropriate container contact and consecutive physics-step stability with declared time and tolerances. Measure destination independently of the model answer and source fit tags. Keep original QA correctness separate from physical completion. For allow_remove_after_verification, latch each target's verified placement before removal to free a shared container. Qualifiers/evaluators must not issue actions or object pose commands.

Several recipes are useful:

| Recipe | Purpose and evidence | Existing support / limit |
| --- | --- | --- |
| Inspect, choose, pick up and insert one source item | Real physical fit testing with independent release and stable partial insertion. | Public objective exists; terminal evaluator and successful trace absent. |
| Retrieve from a separate support, carry and insert | Adds purposeful native navigation and transport before insertion. | Existing pickup-support layouts and retrieval prompts; route/reach unqualified. |
| Sort original named items and clear shared containers after verification | Uses real derivative questions and extras; needs per-target latched evidence. | Ten formal multi-target tasks; evaluator absent and five extras scaled. |

Opening requires evidence of a usable original articulation. Pouring is not a justified default for rigid-item containment. Ambiguous/failed fit instances should remain diagnostics rather than be forced into claimed insertion success.

### Initial QA and information boundaries

All 27 public task projections contain exactly task_id, task_version, objective, success_criteria, environment and constraints. CPU checks find no source answers, metadata_json, initial_poses, scorer state or private manifest hashes in this projection. [Agent observations](</local_home2/zhouxinyang/models/Show-Harness-Explore/core/sim/esi_observations.py>) use mounted sensors; source scene views, global cameras and target close-ups remain Human diagnostics. This static boundary is correct; a complete Agent runtime was not exercised.

All tasks retain original confidence, stable partial insertion and smaller-suitable-container guidance. The formal task Agent path uses the physical PromptAssembler. The separate [scene-only QA adapter](</local_home2/zhouxinyang/models/Show-Harness-Explore/core/tasks/rigid_qa.py>) explicitly requests no initial answer and applies confidence stopping only to its QA subtask during combined operation. Neither supplies a measured frozen initial QA result for formal tasks. If an initial QA baseline is required, Main should define action-step-zero observation/answer timing and private scoring before manipulation changes the evidence. Preserved source bytes alone do not establish QA equivalence after scaling/repacking.

The source answer format is heterogeneous: 60 JSON-encoded keyed assignment lists and 20 plain container-name strings. The assessment's keyed-list statement is correct for many records but incomplete for the whole corpus. A future private scorer must normalize each source format and retain per-object mappings for derivative questions; the primary-object active scorer cannot be reused indiscriminately.

### Current and earlier CaptureQueue evidence

Both assigned-queue submissions reference the same unchanged Benevolence_0_int / empty_room_0 scene, tabletop variant, seed 0 and scene/task bytes. Their common candidate fingerprint is 7491b4205b7595f98f5d468ef4d83ebf10f37c9c056d84a9e807b4ee2f3e1ff9. Capture signatures and renderer revisions differ:

| Signature prefix | Renderer revision | Measured capture timing |
| --- | --- | --- |
| 26a4dde8650c | f8039649c71af53b5d54b169d96982dcd94306e1 | 1481.153 s total: 242.699 s initialization, 1211.772 s explicit capture wait, 19.282 s screenshots. |
| 100c723fc4b4 | 4aa1c7d00f44995746151d52258c4f1bb89f3659 | 204.618 s total: 194.652 s initialization, 9.966 s screenshots. |

Both are REVIEW_PENDING with ten expected views complete, model_calls=0, robot_actions=0 and physical_grasp_verified=false. Native startup accepts the movable collidable target, stable support, native support contact for target and containers, zero speeds and no robot nonfloor contacts. The directed target camera records 7581 visible target pixels. This supports that frozen initial layout, not grasp, route or insertion.

Six existing images were inspected: global_diagonal, head and global_target for both revisions. They show the same initial teddy bear/cup/lunch-box scene. The head crops part of the teddy bear and cup; the Human close-up cannot replace legitimate Agent evidence. No visual insertion result exists.

Both category native PIDs (2274891 and 2362749) are absent from /proc at this review snapshot. The later retained-world record is CLOSED and process_exit reports exit_code=0. Queue status still has exit_code=null and the preview retains world_retained=true; those historical fields do not indicate a running workload. Its 11667.043 s full process duration includes retained residency after capture. No assigned-category capture is currently running; ongoing captures from other categories are outside this review.

The earlier queue and queue_native_ready roots retain failed signature 26a4dde8650c attempts: 8.071 s / exit 1 and 24.103 s / exit 55, both gpu_initialization. The first native log gives only a wrapper error. The second records Kit extension dependency solver failure and FileExistsError for omni.usd.schema.flow installation. These infrastructure failures are not geometric containment failures; later success does not erase them. Wait/retention differences prevent a controlled speedup claim.

Captured loader/schema/scene hashes differ from current working-tree code, while the shared QA protocol and QA adapter match. Old native qualification does not certify the latest implementation.

### Previous revisions and unavailable evidence

All 40 archived category YAML definitions and six archived catalogue snapshots were read. Each archived catalogue spans all 80 source identities with 82 entries including alternatives. Chronological ready/pending/discarded counts are 0/80/2, 17/63/2, 17/63/2, 16/64/2, 15/65/2 and 17/63/2. These historical iterations are not additional physical successes.

Seven frozen code snapshots were read for category definitions:

- [057b58e3fb1fae09c02db7a41f53b6677eb6b5c7](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue/revisions/057b58e3fb1fae09c02db7a41f53b6677eb6b5c7>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.
- [4aa1c7d00f44995746151d52258c4f1bb89f3659](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue/revisions/4aa1c7d00f44995746151d52258c4f1bb89f3659>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.
- [f8039649c71af53b5d54b169d96982dcd94306e1](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue/revisions/f8039649c71af53b5d54b169d96982dcd94306e1>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.
- [37cb42720061f2fe0cdbaf45d454680b1e07f497](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/m3_pair_20261014/revisions/37cb42720061f2fe0cdbaf45d454680b1e07f497>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.
- [5af7edd568265168ab69cbf787cb6fff0e06ab14](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/m3_pair_20261014/revisions/5af7edd568265168ab69cbf787cb6fff0e06ab14>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.
- [80eacb52ac3f09b56c36b7560d3cd6b648d3d3c4](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/m3_pair_20261014/revisions/80eacb52ac3f09b56c36b7560d3cd6b648d3d3c4>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.
- [bf28a1ac0a7e5830c00ca28ed8fae95e3373d2e1](</local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/m3_pair_20261014/revisions/bf28a1ac0a7e5830c00ca28ed8fae95e3373d2e1>) — 92 scene descriptors, 27 tasks and 40 archived descriptors.

Across these snapshots there are 1113 category YAML occurrences and 156 distinct definition hashes. The M3 snapshots preserve previous category definitions; their other-category native results are excluded from containment evidence. All prior candidate revisions and assigned task files remain read-only inputs.

Four assessment links were checked and are absent: asset_layout_assessment.json, assessment.json, contracts.json and source_scorer_checks.json. All 107 checked /scratch references in the current containment catalogue are also absent. The local [expansion plan](</local_home2/zhouxinyang/models/Show-Harness-Explore/docs/rigid_containment_task_expansion_plan.json>) supports the question/task audit, but its referenced approval snapshot and historical raw construction results are unavailable locally.

Legacy bathroom robot_initial_contact and corridor scene_initialization_failed records remain pending. Their detailed contact/instability claims are historical summaries, not measurements reproduced by this reviewer. The discarded bedroom source and __none derivative actually retain target Z=-132.19361877441406 m; this malformed source state was independently confirmed by CPU reading. It is not an impossibility proof for all alternate operations. Historical catalogue review fields are provenance; this review grants no human approval.

### Every assigned previous task

Every listed task matches its supplied hash, remains unchanged and is recorded with full scene/question paths and scaling evidence in the JSON result.

| Read-only previous task | Targets | Primary multiplier |
| --- | --- | --- |
| [Beechwood_0_int_scene_dict_storage2_0_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Beechwood_0_int/living_room_1/Beechwood_0_int_scene_dict_storage2_0_revised.yaml>) | 1 | 1.0 |
| [Beechwood_1_int_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Beechwood_1_int/playroom_0/Beechwood_1_int_scene_dict_storage2_2_revised.yaml>) | 1 | 0.883311 |
| [Beechwood_1_int_scene_dict_storage2_2_revised__criterion_2.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Beechwood_1_int/playroom_0/Beechwood_1_int_scene_dict_storage2_2_revised__criterion_2.yaml>) | 2 | 0.883311 |
| [Beechwood_1_int_scene_dict_storage2_1_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Beechwood_1_int/television_room_0/Beechwood_1_int_scene_dict_storage2_1_revised.yaml>) | 1 | 0.615608 |
| [Beechwood_1_int_scene_dict_storage2_1_revised__criterion_4.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Beechwood_1_int/television_room_0/Beechwood_1_int_scene_dict_storage2_1_revised__criterion_4.yaml>) | 3 | 0.615608 |
| [Benevolence_0_int_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Benevolence_0_int/empty_room_0/Benevolence_0_int_scene_dict_storage2_2_revised.yaml>) | 1 | 1.0 |
| [Benevolence_1_int_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Benevolence_1_int/living_room_0/Benevolence_1_int_scene_dict_storage2_2_revised.yaml>) | 1 | 1.0 |
| [Benevolence_1_int_scene_dict_storage2_2_revised__criterion_2.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Benevolence_1_int/living_room_0/Benevolence_1_int_scene_dict_storage2_2_revised__criterion_2.yaml>) | 2 | 1.0 |
| [Ihlen_1_int_scene_dict_storage2_1_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Ihlen_1_int/living_room_0/Ihlen_1_int_scene_dict_storage2_1_revised.yaml>) | 1 | 1.0 |
| [Ihlen_1_int_scene_dict_storage2_1_revised__criterion_2.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Ihlen_1_int/living_room_0/Ihlen_1_int_scene_dict_storage2_1_revised__criterion_2.yaml>) | 2 | 1.0 |
| [Merom_0_int_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Merom_0_int/living_room_0/Merom_0_int_scene_dict_storage2_2_revised.yaml>) | 1 | 0.364473 |
| [Pomaria_1_int_scene_dict_storage2_1_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Pomaria_1_int/living_room_0/Pomaria_1_int_scene_dict_storage2_1_revised.yaml>) | 1 | 1.0 |
| [Wainscott_0_garden_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Wainscott_0_garden/living_room_1/Wainscott_0_garden_scene_dict_storage2_2_revised.yaml>) | 1 | 1.0 |
| [Wainscott_0_garden_scene_dict_storage2_2_revised__criterion_3.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/Wainscott_0_garden/living_room_1/Wainscott_0_garden_scene_dict_storage2_2_revised__criterion_3.yaml>) | 2 | 1.0 |
| [grocery_store_half_stocked_scene_dict_storage2_1_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/grocery_store_half_stocked/grocery_store_0/grocery_store_half_stocked_scene_dict_storage2_1_revised.yaml>) | 1 | 0.533333 |
| [grocery_store_half_stocked_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/grocery_store_half_stocked/grocery_store_0/grocery_store_half_stocked_scene_dict_storage2_2_revised.yaml>) | 1 | 1.0 |
| [grocery_store_half_stocked_scene_dict_storage2_2_revised__criterion_2.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/grocery_store_half_stocked/grocery_store_0/grocery_store_half_stocked_scene_dict_storage2_2_revised__criterion_2.yaml>) | 2 | 1.0 |
| [hall_arch_wood_scene_dict_storage2_1_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/hall_arch_wood/empty_room_0/hall_arch_wood_scene_dict_storage2_1_revised.yaml>) | 1 | 1.0 |
| [hall_conference_large_scene_dict_storage2_0_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/hall_conference_large/conference_hall_0/hall_conference_large_scene_dict_storage2_0_revised.yaml>) | 1 | 1.0 |
| [hall_conference_large_scene_dict_storage2_0_revised__criterion_3.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/hall_conference_large/conference_hall_0/hall_conference_large_scene_dict_storage2_0_revised__criterion_3.yaml>) | 2 | 1.0 |
| [hall_conference_large_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/hall_conference_large/conference_hall_0/hall_conference_large_scene_dict_storage2_2_revised.yaml>) | 1 | 1.0 |
| [hall_conference_large_scene_dict_storage2_2_revised__criterion_2.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/hall_conference_large/conference_hall_0/hall_conference_large_scene_dict_storage2_2_revised__criterion_2.yaml>) | 2 | 1.0 |
| [house_double_floor_lower_scene_dict_storage2_0_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/house_double_floor_lower/living_room_0/house_double_floor_lower_scene_dict_storage2_0_revised.yaml>) | 1 | 0.427296 |
| [house_double_floor_lower_scene_dict_storage2_0_revised__criterion_4.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/house_double_floor_lower/living_room_0/house_double_floor_lower_scene_dict_storage2_0_revised__criterion_4.yaml>) | 3 | 0.427296 |
| [office_vendor_machine_scene_dict_storage2_2_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/office_vendor_machine/meeting_room_0/office_vendor_machine_scene_dict_storage2_2_revised.yaml>) | 1 | 1.0 |
| [restaurant_hotel_scene_dict_storage2_1_revised.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/restaurant_hotel/lobby_0/restaurant_hotel_scene_dict_storage2_1_revised.yaml>) | 1 | 0.227594 |
| [restaurant_hotel_scene_dict_storage2_1_revised__criterion_4.yaml](</local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/tasks/Physical Structure/Rigid Containment/restaurant_hotel/lobby_0/restaurant_hotel_scene_dict_storage2_1_revised__criterion_4.yaml>) | 3 | 0.227594 |

### Recommendations for Main

1. Add a read-only per-target evaluator for native attachment/lift history, guarded transport when required, release, calibrated partial cavity insertion and consecutive physics-step stability. Keep QA score separate and latch verified events before allowed removal.
2. Use original dimensions in future source-preserving revisions. Explicitly label historical primary/extra scaling and independent repacking; preserve every existing task, scene, source, answer and review record.
3. Define any initial QA baseline at a declared pre-manipulation boundary with a private scorer supporting both single-container strings and keyed per-object assignments. Retain mounted-sensor provenance and exclude Human images.
4. Only after later incremental opt-in, reuse legacy modules through a thin category adapter and publish explicit outcomes for every one of the 80 source records. Preserve the absent-batch diagnostic now.
5. Keep historical failures and unavailable-artifact labels in Review. Synchronize stale capture/process display fields through normal integration without blocking Review or screenshots.
6. Consider the three source-faithful operation recipes independently; allow opening only with demonstrated native articulation. Keep ambiguous fit cases diagnostic rather than claiming insertion.

Only this owned report and the specified JSON result were written. Implementation, generated scenes, source data, answers, approvals, other reports, root research_log.md and workloads were not changed. No GPU work, commit or agent delegation was performed. These findings do not block pending/failed candidates from Review or ongoing screenshots.

Review result: [result.json](</home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/bf79a025cc985aff3873/94f19e59cb152a62763c2671/result.json>).

## 2026-10-15 CPU interaction-group and grasp-height review (rendering paused)

This separate read-only iteration inspected all **80 original Rigid Containment sources and all 27 existing ready tasks/scene pairs**. No approved definition was regenerated, edited or requalified; historical scale and repacking patches are retained verbatim. The new [read-only audit](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/category_grasp_audit/rigid_read_only_audit.json) records each task/scene hash, declared support qualification, target poses, spawn patches and old native receipts. This review ran CPU reads only while all rendering remained paused.

All 27 existing tasks declare their targets in supported_objects. Declared adapted target root world Z ranges from **0.378397 to 1.080929 m**; declared target/start planar distances range from **0.544804 to 13.138437 m**. These are static patch values, not floor-relative measured grasp or route certificates. Targets on a separate pickup support may intentionally require navigation before grasping. Existing per-object support assignments, stable native contact requirements, speeds, collision guards and native floor-supported chassis rules remain unchanged.

Two existing native receipts for the same Benevolence empty-room regression are available. Their frozen initial support qualification passed: the target AABB minimum Z is approximately 0.450001 m on a support top at 0.450000 m, with actual native target/container contact and stationary objects. The captured target is approximately 0.625 m from the initial robot. Metadata declares assisted grasp and an explicit gripper controller; both receipts report zero model calls, zero robot actions and physical_grasp_verified=false. Thus these are initial-layout measurements, not actual grasp or stable insertion results, and do not certify all 27 tasks.

The existing target AABB is approximately 0.261 x 0.282 x 0.183 m. This global box is a width-risk diagnostic, not an oriented local grasp width. Gripper aperture, collision-free arm IK, native acquisition and independent terminal insertion remain unqualified by this audit. Six old native frames (diagonal, head and directed target for both receipts) were examined; the target/support/container arrangement is visible, with cropped head context. No insertion result is shown. All approved artifacts and old evidence remain unchanged.

- [Frozen native receipt](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_100c723fc4b4bbb4efa51b687d109a1d/attempts/01/preview_complete.json) / [global_diagonal](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_100c723fc4b4bbb4efa51b687d109a1d/attempts/01/global_diagonal.png) / [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_100c723fc4b4bbb4efa51b687d109a1d/attempts/01/head.png) / [global_target](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_100c723fc4b4bbb4efa51b687d109a1d/attempts/01/global_target.png)
- [Frozen native receipt](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_26a4dde8650c5c433b942d803e67e055/attempts/01/preview_complete.json) / [global_diagonal](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_26a4dde8650c5c433b942d803e67e055/attempts/01/global_diagonal.png) / [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_26a4dde8650c5c433b942d803e67e055/attempts/01/head.png) / [global_target](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_26a4dde8650c5c433b942d803e67e055/attempts/01/global_target.png)


## 2026-10-15 Approved accessible-support generator audit

Plan: apply the user-approved access repair at generator-rule level after reading the final v2 specification, current converter/source builder and reports. The historical assessment is supplementary and nonbinding. Audit every source and the latest human-rendered batch referenced by category_revisions.json; preserve source QA, old failures and approved work. No source JSON, published task, queue, screenshot or Native process was modified.

Coverage: **80 original sources**, **0 existing task definitions read**. Audit only. No converter exists in this assignment. Preserve all approved benchmark scenes, tasks, photos and original fit QA; do not regenerate. [Whole-class evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/bf79a025cc985aff3873/audit.json) and [eleven-class summary](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/audit_summary.json) preserve every inspected source identity and failure.

No local converter change or regeneration is justified by this audit. Existing source/task definitions and screenshots are preserved. Recorded prior failures remain historical evidence and are not cleared by this decision.

Validation: `tests/test_accessible_generators_a.py` completed **20 passed in 1.53 s**; the focused annotation, tabletop-relation and goal-only compatibility suites completed **78 passed in 1.76 s** using behavior Python. These CPU fixtures check private-QA boundaries, generated schemas, one-time initialization, adaptation policies and native-shaped support/cover predicates. Native surface/contact acceptance, collision-free floor access, cloth/scaffold integrity, mounted views, IK, grasp/release and physical execution remain unverified for the new runtime. Shared placement and subsequent one-representative-per-changed-class rendering are owned by Main.


## Safe composition and review checks — 2026-10-16

All 27 inherited ready tasks and their relevant scene definitions were inspected read-only. They remain 27 standalone tasks; no user-designed task, scene or approval is changed. The 17 canonical source families (15 rooms) are conditional reuse opportunities only: ten criterion variants add actors, and four change runtime. Added actors can change visibility and physics. Existing multi-target tasks already permit removal after verified insertion to free shared containers. Any future composed scoring must separately certify every named original QA and real stable partial insertion, preserve historical scales and explicitly record their source-size differences. The current loader advertises physical_evaluator: not_configured, so no automatic insertion success is newly claimed. No native trial or render was run.

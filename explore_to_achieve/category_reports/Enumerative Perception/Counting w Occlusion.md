# Counting w Occlusion

## Research plan and source audit (2026-10-15)

Read the assignment, implementation spec, shared ConverterBase/scene/task contracts,
owned operation API, source task and shared counting builder, all 30 source JSON
instances, and the supplied historical assessment before implementation.

Hypothesis: a physical access patrol can reuse the original floor targets and
occluding furniture without manipulating answer-sensitive layouts. Implement
two deterministic recipes: visit the target-facing side of every original
occluder, or make those visits and return to the measured departure position.
Success requires actual continuous guarded base motion, native floor support,
safe target standoff, native occluder-side geometry, preserved object poses,
and consecutive low-speed samples. Original counting QA remains auxiliary.

Implement an evidenced robot-only L2 start near the original camera projection,
screened against source anchor/target AABBs. Preserve an empty original variant.
Run complete CPU conversion and verify determinism, private/public boundaries,
and fail-closed native predicate behavior. Inspect stratified native images if
available; otherwise explicitly retain the missing-evidence limitation and
publish pending candidates for the main runner's automatic Review submission.
Perform at most two targeted rule revisions.

Current source audit: 30 instances, 12 scenes, 17 scene/room pairs; 29 sandwich
sources and one iron source. Source anchors are restored by the current builder's
category filter. Generated targets are visual-only at construction and again
after postprocessing. Navigation preserves those flags; no manipulation is
claimed. Some original camera projections lie inside desks or the kitchen bar,
so camera XY alone is not a justified robot start. Native qualification must
check the actual body, floor and contacts; CPU AABB screening is only advisory.

Historical attribution: [the supplied assessment](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:89)
reports a corrected representative native load/control probe, but explicitly
does not certify body-view coverage. Its original-probe failure was a velocity
diagnostic on RigidKinematicPrim, not evidence that the scene must be rebuilt.
Those experiments were not run by this supervisor. Historical /scratch artifacts
are not established as locally available. Current construction code corroborates
anchor reuse and visual-only targets, but establishes no executable patrol.

## Implementation retained

[Owned converter](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/enumerative_perception_counting_w_occlusion.py),
version **1.1.0**, implements `inspect`, `convert`, inherited deterministic
persistence through `convert_all`, stratified `sample`, `qualify_operation` and
`create_evaluator`. No shared code, source data, other category report or root
research log was changed by this supervisor. No GPU work, queue submission,
manual YAML editing, commits or sub-agents were used.

- `occluder_side_patrol`: reach each original target's accessible side of its
  paired original furniture, then finish at one qualified approach.
- `occluder_side_patrol_return`: complete the same physical visits, then return
  to the actual measured episode departure point.

For each pair, let `t` be the measured native target root XY, `a` its nearest
point on the measured native occluder AABB, and `n=(t-a)/||t-a||`. A visit requires
robot distance to `t` in **[0.55, 1.25] m**, `(robot_xy-a) dot n >= 0.20 m`, and
at least **0.05 m** separation between actual chassis and occluder AABBs.
The robot must be in the original queried room, supported by the original floor,
without external non-floor contact. It must first depart at least **0.50 m** from
the measured start. Visits require **three consecutive control samples** with
translation speed at most **0.05 m/s** and angular speed at most **0.10 rad/s**.
Completion additionally requires **1.0 m** of sampled physical travel and a
stopped valid terminal approach; the return recipe instead requires three
stopped samples within **0.30 m** of departure.

Per-step displacement is checked against the actual guarded controller's speed,
dt and translation tolerance. Unsafe contacts, discontinuous movement or object
layout changes latch failure for the episode. Target and anchor poses must stay
within **0.02 m / 5 degrees** of measured initial state. Qualification separately
checks original source target positions/orientations and source object AABBs,
collidable original anchors, the native room and measured floor support. Floor
support requires actual floor contacts and local upward-facing floor rays at the
three installed R1 wheel links; all wheel/floor gaps must be within **0.03 m**.
Missing native measurements fail closed. These are declared candidate thresholds,
not empirically calibrated physical success guarantees.

Robot-only L2 initialization uses the nearest deterministic point to the source
reference camera XY, on a **0.20 m** grid inside the expanded source room bbox,
with **0.55 m** clearance from source target/anchor AABBs and bbox boundaries.
Floor height comes from `requested_position.z - BALL_RADIUS`, corroborated by
`batch_counting_merge.py` (`BALL_RADIUS=0.035`, camera height `1.25 m`). Native
floor and body evidence still gates execution. No objects are moved or scaled.
The current builder already installs R1 and loads the original anchor categories,
so `add_existing_r1` and mechanism-restoration initialization hooks are unnecessary.
Every original variant retains empty patches. Targets keep their original
visual-only physics because these recipes perform navigation, not manipulation.

The public goal names the category only from the original public question and
gives operation tolerances. Native object names, target counts/positions, source
answers, maps, closeups, proposed start coordinates and evaluation state stay
private. Original metadata/QA is saved verbatim in `private_qa.json`. Geometry
access is deliberately independent of counting correctness and pixel visibility.
The original source observer data is retained; the shared capture runtime may
frame an adapted human overview differently, so it must not be confused with
the original reference camera or an Agent sensor.

## Iteration and validation

Initial CPU conversion produced 30 pending candidates. One targeted rule
iteration was retained: installed R1 code identifies `base_link` as the chassis,
`wheel_link1/2/3` as the floor-touching links, and a fixed-base holonomic model.
Remove the proposed 0.02 m spawn offset, which cannot be assumed to settle under
gravity, and check measured wheel/floor support instead of chassis-bottom height.
Also require low angular speed when parked and sample all observed target-count
strata. No second rule iteration was needed after schematic inspection.

The complete final CPU batch contains **30 generated, 0 conversion failures**:
**17 patrol / 13 patrol-return**, all L2, pending and carrying
`assumption_review_required`. There are **0 native-qualified / physically
executed / human-approved** instances from this supervisor. No actual Review
submission is claimed; `supervisor_result.json` requests `publish`, and the main
CategoryRunner owns automatic CaptureQueue/Review submission of every record.
Future native load, capture or operation failures must remain diagnostic Review
items and must not be promoted by source-only images.

[CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_validation.json)
records 19 checks covering complete source coverage, shared schema validation,
source QA preservation, identical scene/task overlays, empty original variants,
private/public boundaries, deterministic candidate replay, pending execution
gates, full assumption fields, explicit malformed-copy diagnostics, failure-first
sampling, fail-closed missing native state, and adapter episode logic. Synthetic
logic rejects no motion, excessive displacement, unsafe state, premature return,
object displacement and a turning terminal pose. A synthetic valid return sequence
checks state-machine consistency only; **it is not native operation evidence**.
A test input initially used a NumPy array instead of the native reader's list
format; correcting that test input resolved JSON serialization without changing
converter rules. Idempotent final replay took approximately **0.15 s / 30 sources**;
an earlier complete generation took approximately **8.13 s / 30 sources** on this
host. Native latency/throughput remain unmeasured. The declared 600-decision budget
is a proposal, not an inherited experimental setting or timing guarantee.

The actual-category conversion failure distribution is empty. The temporary
malformed-source check produced an explicit
`not_convertible_under_L1_L2: Original target/anchor mechanism is incomplete`
record at source inspection; this diagnostic validation copy is separate from
the real 30-source batch and does not alter or represent a corpus source failure.
Previous generated revisions remain on disk for traceability.

## Stratified evidence inspected

No current category `sample_evidence.json`, matching local native screenshot, or
historical `/scratch` category artifact was available at inspection. The five
images below were generated deterministically from source metadata on CPU and
opened for visual inspection. **They are source-layout schematics, not native
screenshots, Agent observations or evidence of executable routes.**

| Candidate / source stratum | Evidence and observation |
| --- | --- |
| `category_4746320ea9818dd86c0d8977`, Beechwood_0_garden / living_room_0, 3 irons, patrol | [Schematic](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_source_layout_evidence/category_4746320ea9818dd86c0d8977.png). Start matches camera XY. The bookcase-side target is near the right boundary; native body access is uncertain. |
| `category_002970dc6fc5d375e9ad178b`, Beechwood_0_int / living_room_1, 3 sandwiches, return | [Schematic](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_source_layout_evidence/category_002970dc6fc5d375e9ad178b.png). Boundary screening shifts start 0.15 m. Furniture AABBs overlap and one target approaches the room boundary. Mesh penetration and navigation are unverified. |
| `category_0060c8a276faef8e1bd82a03`, grocery_store_cafe / dining_room_0, 3 sandwiches, patrol | [Schematic](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_source_layout_evidence/category_0060c8a276faef8e1bd82a03.png). Start matches camera XY; targets remain on distinct sides of table/chair anchors. Other loaded category instances are not represented by the source screen. |
| `category_2c38ba5279d81ae20a7821b1`, house_single_floor / kitchen_0, 5 sandwiches, patrol | [Schematic](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_source_layout_evidence/category_2c38ba5279d81ae20a7821b1.png). Original camera projection lies inside the bar; robot-only start shifts 0.752 m away. All targets remain behind the original bar/stools. |
| `category_0e1e8b2905bdb998ac0a60ad`, office_cubicles_right / private_office_2, 2 sandwiches, return | [Schematic](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_source_layout_evidence/category_0e1e8b2905bdb998ac0a60ad.png). Camera projection lies inside the desk; start shifts 0.756 m. The opposite-side target is near the room boundary; the schematic does not prove chassis clearance. |

[Sampling evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/cpu_sampling_evidence.json)
contains source hashes, image kinds, recipe strata and robot-only shifts.
[Complete BatchManifest](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/b74d423c6c28efb78626/batch_manifest.json)
contains every original source and private terminal predicate.

## Unresolved assumptions and diagnostic recommendations

Every candidate explicitly records doubt, chosen assumption, alternatives, scope
and risk. The 0.55 m source screen omits walls and other loaded objects of the
same anchor categories, and coarse room segmentation can disagree with the
expanded bbox. Native start qualification may fail even when a CPU start exists.
Targets at walls or in clustered furniture may lack a valid body approach. Floor
contact reporting and strict source-geometry matching may expose additional
native failures. Do not enlarge passages, relocate targets, weaken predicates,
disable safeguards or change original answers merely to obtain acceptance.

For start failures, inspect measured wheel gaps, source floor rays, non-floor
contacts and actual room membership before proposing another robot-only L2
stance. For unreachable visits, retain the failed patrol and original diagnostic
variant in Review; consider a separately evidenced lateral access recipe only
after native route measurements. Native mounted-camera pixel coverage would be
needed to claim visual count certification. Ground-level source closeups and
human overview images cannot supply that evidence. Source occlusion annotations
are XY segment/AABB construction labels, not measured native visibility.

Decision: **publish all pending records for Review with limitations**, retaining
the rule and awaiting native evidence. No human feedback revision was supplied;
the supervisor result therefore records `feedback_revision: null`.


## Supplementary assessment comparison (2026-10-15; read-only review)

Review fingerprint: `ff35f7febdd72d7bbe7bf367a6b7e73d8dce944ee227b324a129e9a44bc098b9`. This comparison covers only **Enumerative
Perception / Counting w Occlusion** and is non-blocking for Review and captures.
Recommended implementation changes belong to Main. No conversion, simulator,
model experiment, GPU work or robot trial was run by this reviewer.

The review plan was to reconcile the specification and historical assessment
with current construction, audit every source and revision, compare saved CPU
claims with live CaptureQueue artifacts, inspect representative native views,
and assess operation predicates and the initial QA information boundary.
Read inputs included the implementation spec, supplied assessment, current
converter and shared counting builder, shared task/operation/observation
contracts, complete batch and both saved batch manifests, CPU validation and
synthetic trajectories, CPU sampling evidence, current `sample_evidence.json`,
and all candidate scene/task/QA/record definitions. The assignment contains
**zero `existing_task_files` entries**. All earlier candidate definitions were
included as read-only previous tasks.

### Current evidence and previous revisions

The batch covers all **30/30** corpus sources, with **30 generated, 0 actual
conversion failures**, **17 patrol / 13 patrol-return** recipes, 12 scene models
and 17 scene/room pairs. Target strata are one 3-iron source and 29 sandwich
sources: six with 2 targets, 21 with 3, and two with 5. All current scene/task
files remain `pending`, L2 and warned. Read-only checks reproduced complete
source coverage and schema validity for every saved definition; scene/task
patches agree, original variants are empty, and adaptations contain only a
robot-pose patch. Across all 90 saved candidates, private answers, options and
encoded metadata equal their corresponding current source fields. Source
answers, payload answers and generated target counts agree across this batch.
No public task field exposes bound target identities, source poses/maps, source
answers or private visit histories. Operational tolerances are deliberately
public goal text.

There are **90 candidates: three revisions per source**, not 90 independent
source tasks. Their differences matter:

| Revision group | Count | Definition difference from current |
| --- | ---: | --- |
| Earlier 1.0.0 | 30 | Spawn Z is source floor + 0.02 m; angular visit-speed parameter absent. |
| Earlier 1.1.0 | 30 | Spawn Z is source floor; angular visit-speed parameter still absent. |
| Current 1.1.0 | 30 | Spawn Z is source floor; angular limit is 0.10 rad/s. |

Only the current 30 have category records in the assigned CaptureQueue. Both
saved batch snapshots list the current candidate IDs. They do not substitute
for the 60 previous definitions, which were read separately. Earlier runtime
implementations were not established from these definitions; no old candidate
was executed with today's stricter configuration validator. The two distinct
1.1.0 groups demonstrate why revision/implementation identity, rather than the
version label alone, is needed for comparison. The complete read-only inventory
is included in [the review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/b74d423c6c28efb78626/ff35f7febdd72d7bbe7bf367/result.json).

The current converter SHA-256 is
`fd1000e234a5189a74f540cddd6c709db2c7e0338d60b2c822567c6e7cce7d4c`, matching
`cpu_validation.json`. The converter, category task module and shared counting
builder are byte-identical to the assigned frozen capture code
`working_f1a92606b2b2c25d9f1a93d8208d52ca`. This establishes relevant code
provenance, not physical qualification.

At **2026-10-14T20:16:25.528115+00:00**, direct job-status reads found **9 `REVIEW_PENDING`, 21 `QUEUED`,
0 active and 0 failed jobs** for this category. All 30 have published category
records. All five selected samples now have complete native receipts; nine
receipts contain **81 readable PNGs** in total. Receipt identities and required
view sets were checked against their input manifests. Each completed receipt
records `model_calls=0`, `robot_actions=0`, `native_qualification_run=false`,
`eligible_final_layout=false` and `diagnostic_pending=true`. `REVIEW_PENDING`
therefore establishes capture availability, not approval or patrol completion.
The saved `sample_evidence.json` still says five samples are `QUEUED`; the older
supervisor result and report also predate these images. Their missing-image
claims were historical observations, not the current measured state. They
remain unedited.

Observed completed capture wall durations were **234.58 / 300.68 / 501.16 s**
(minimum / median / maximum, n=9), including initialization and screenshot work,
excluding queue waiting. These are capture costs, not robot execution latency,
model throughput or quality metrics. The recorded 19 CPU checks and 0.147 s
idempotent replay are earlier validation results. This reviewer reran only
read-only artifact/schema checks, not `convert_all` or the synthetic evaluator.
The saved synthetic trajectory contains a positive return and subsequent
layout-change rejection; its final `state.json` alone is negative. Its explicit
synthetic provenance excludes all of these values from native success evidence.
The malformed-source diagnostic remains a separate validation copy, not a
failure of one of the real 30 sources.

### Comparison with the supplementary assessment

| Historical claim or recommendation | Current comparison |
| --- | --- |
| Preserve targets, occluders and anchor geometry; no independent target spreading is justified. | Supported as a conversion rule: all adaptations move only R1. Generated targets retain visual-only physics, appropriate for navigation. The builder reconstructs original models/poses and loads anchor categories. No current evidence justifies target relocation, resizing or collision changes for these recipes. |
| Corrected representative native load/control probe completed; full body-camera count coverage was not certified. | Historical report records 3 targets, 7/7 bounded controls and 177.2/199.3 s readiness/total for one corrected probe. Current evidence is nine fresh capture-only loads with zero robot actions. These different observations cannot be combined into category-wide route or control qualification. |
| Original probe failed on `RigidKinematicPrim.get_linear_velocity`. | Historical diagnostic-interface failure, not proof that source geometry is unusable. The current patrol code reads poses/AABBs and robot pose differences, avoiding that target-velocity call. The historical failure was not reproduced here. |
| Preserve the original room/layout and use accessible robot views. | Target/paired-anchor reuse is supported, but complete original-room reconstruction is not established. The active builder explicitly filters loaded categories to floors, walls and source anchor categories, unlike the source generator's broader scene configuration. Other furniture can be absent and additional instances of an included category can remain. This changes background occlusion and navigation difficulty even when the paired mechanism is retained. |
| Original counting scorer remains separate from a physical goal. | The current scorer still implements `max(0, 1 - abs_error/3)` plus exact correctness. QA data is preserved privately, but this category task has no declared auxiliary QA scoring phase or initial-observation result. Archive preservation must not be described as measured QA correctness. |

The historical assessment pins ESI commit `f35b0de448ee77bc46714ae84b5190c17f87d86d`
and describes an A40-server diagnostic protocol with eight mounted cameras at
256 × 256. Current assigned captures pin ESI
`9f6f15ed43e81654a97de9e4db2a1fedadca2bc1` on deep59 and request nine review
views including human cameras. The inspected head/global images are 512 × 512;
the scene images are 1280 × 720. Server, source revision, scene instances,
observation protocol and workload differ. Historical runtime and control
claims are not a controlled baseline for the current conversion. The linked
historical Counting w Occlusion directory under `/scratch` is unavailable
locally; its raw probes and images could not be independently inspected.

### Meaningful operations, terminal predicates and QA boundary

Both recipes provide a meaningful candidate operation: guarded physical
navigation to original target-facing sides of occluding furniture, requiring
measured departure, travel and stopped visits. The return recipe adds a measured
return to the episode's actual departure point. This creates a useful outbound
and return-navigation variant while retaining the same source targets and
occluders. Recipe selection is deterministic by seed/source hash, not based on
measured feasibility. The definitions do not establish that either recipe is
executable in every room. Additional recipes are useful only if they add a
source-faithful capability with distinct evidence; a lateral inspection or
heading/visibility-aware visit is a reasonable future extension. Grasping,
moving targets or removing furniture is not justified by current evidence.

The current native predicate is independent of source answers and source
success flags. It binds native objects, checks room membership, source geometry,
collidable anchors, real floor contacts and wheel-floor rays; missing evidence
fails closed. Episode history measures actual robot pose deltas, departure and
travel, furniture-side standoff, layout preservation, low translation/angular
speed and consecutive stopped samples. Unsafe contact, excessive sampled motion
or target/anchor displacement latches failure. Shared runtime hooks attach
sampling to guarded controller steps and keep results private. The return
predicate uses measured start coordinates and requires completed visits before
return stability. No success-scoring hook moves an object or robot.

This is a defensible **geometric access-patrol predicate**, not a visibility or
count-certification predicate. It has no target-facing camera heading, depth
line-of-sight or segmentation/pixel-coverage requirement; `initial_visibility`
is explicitly `none` and its evidence sets `pixel_visibility_qualified=false`.
An R1 can geometrically visit the correct side while facing away or viewing an
occluded target. Three stability steps mean control samples, not three Agent
decisions; the dwell time depends on controller frequency. Floor contact/ray
availability, source/native AABB tolerances, accessible approaches and the
600-decision budget remain uncalibrated in real patrols. No current screenshot
has exercised `qualify_operation` or the visit/return evaluator.

The information boundary is correct at the inspected definition level:
original answer/options/metadata are private, public task text contains the
public source question, and private geometry is used for construction and
scoring. The initial QA boundary is **incomplete as an evaluation protocol**.
The original question says “Considering both images”; current public physical
task text appends that question without declaring the original image context,
answer protocol, scorer invocation or timing relative to navigation. A source
image basename checked beside its source JSON is absent there; availability of
other copies was not established. Source GT maps/corner cameras/target closeups
and human review screenshots must remain outside Agent sensing. Merely saving
`private_qa.json` neither creates a legitimate initial sensor packet nor scores
a pre-operation answer. A future auxiliary QA measure must identify the initial
native world snapshot and authorized sensor history, retain original scoring
separately, and explicitly account for changed robot viewpoints. Source data
preservation alone cannot demonstrate that initial visual difficulty is intact.

### Native sample observations and diagnostic limits

Five overhead and five head PNGs were visually inspected, spanning both recipes,
all count strata and the iron/sandwich split. These are human review artifacts,
not an additional Agent observation stream.

| Sample / capture job | Observed image content and remaining limit |
| --- | --- |
| `category_4746320ea9818dd86c0d8977` / `capture_29493fcb5438e060eacdfd0c79e8607c` | Overhead framing is dominated by floor/table; head view shows floor, hands and body. The inspected views do not visibly establish all three irons. Other views/cameras were not exhaustively searched. |
| `category_002970dc6fc5d375e9ad178b` / `capture_6bf62c64b46692a09be22690d20abd54` | Three sandwich appearances are visible around/behind a chair in overhead; the head view shows floor and robot body. No chassis route or mesh clearance is certified. |
| `category_0060c8a276faef8e1bd82a03` / `capture_7364d5ff832bae94b7969c217637d711` | Chairs and robot occupy the inspected framing; the head view has strong body/hand foreground. Complete count coverage is not shown in these two views. |
| `category_2c38ba5279d81ae20a7821b1` / `capture_9e6033402967e2f8383bd67d754fb747` | Five sandwich appearances line the bar/stool region in overhead, with edge clipping. Head framing is blocked by bar/body. No physical five-target patrol or sensor-visible route is established. |
| `category_0e1e8b2905bdb998ac0a60ad` / `capture_e5cc8bf47faeaf8e8f7dd6d422d78fe8` | Desk/chair dominate overhead and head framing. The inspected views do not establish both targets or safe body access near the room boundary. |

Image paths are under each job's `attempts/01` directory in
[the assigned queue](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized); exact receipt and image paths are retained in
[the JSON evidence inventory](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/b74d423c6c28efb78626/ff35f7febdd72d7bbe7bf367/result.json). Valid PNGs and complete expected-view
sets establish technical capture completion, not target coverage. The human
camera framing metadata has `include_robot=false`; these overviews are not a
body-clearance qualification. Initial head views cannot prove that other enabled
cameras or future guarded navigation cannot reveal the targets.

Completed jobs also retain diagnostics: `category_96f0505bba761fb10fd1734b`
(`capture_23af8829fc143361d50e13d20d9cd3c6`) logs a Vulkan GPU crash/pagefault
**after** `[esi] closing environment`, while its receipt and process-exit record
remain complete/exit 0. The two inspected office capture logs include PhysX
ceiling convex-mesh creation errors. These anomalies should stay visible beside
the valid captures; their effect on future physics is unmeasured. They are not
evidence that targets must be moved, that current images should be discarded,
or that an operation was successful.

### Recommendations for Main and disposition

1. Retain the two geometric access recipes as warned pending candidates. Require
   normal ready/review gates and separately saved native precondition/trajectory
   evidence before any patrol-success claim. Calibrate wheel support, approach
   feasibility, dwell duration and budget through future authorized native work.
2. If initial QA is to be evaluated, add an explicit auxiliary protocol with
   initial native snapshot/sensor provenance, original question/options and
   independent exact/soft count scoring. Keep the physical objective and its
   private terminal predicate separate. Do not supply GT views as sensors.
3. Describe preservation precisely as target/paired-anchor reuse through the
   current filtered active builder. Before claiming complete original-scene or
   original-visual-difficulty fidelity, Main should compare omitted furnishings
   and measure whether source-faithful restoration is necessary. No blanket
   room or target-layout change is recommended.
4. A separate visibility-aware visit variant could supplement access patrols
   using lawful mounted-camera evidence and a declared heading/coverage
   predicate. Do not silently strengthen existing revisions or claim counting
   correctness from geometric visits or human overviews.
5. Refresh cached sample progress through the existing main-owned reporting
   path when convenient; preserve earlier missing-image claims as historical.
   Keep complete screenshots, native warnings and all previous revisions
   inspectable by source plus revision/hash, including the two 1.1.0 groups.

Disposition: **reviewed; comparison only**. All current pending records, queued
captures, completed diagnostic captures and future failures remain eligible for
Review under existing gates. No implementation, generated definition, source,
answer, approval, queue state, running workload, other category report or root
research log was changed. No robot success or human approval is asserted.

## 2026-10-15 CPU interaction-group and grasp-height review (rendering paused)

This iteration addresses the hypothesis that source-faithful tasks can still be physically unusable when the required interaction is too low, too high, too wide or unsupported. The review read the final v2 specification, supplementary report, current category sources/converters, all current task definitions and existing capture receipts. All simulation/rendering remains paused. This reviewer ran CPU code and reused historical native artifacts; no new native experiment, queue submission, robot action or approval was performed.

Complete coverage: **30 original sources**, status distribution `{"generated": 30}`; **30 existing task definitions** received an explicit interaction profile. Current mode counts are `{"navigation": 30}` and relocation decisions are `{"preserve": 30}`. 0 original non-generated records remain in the audit, with their failure stage/reason and source identity. No canonical candidate or source was overwritten.

Corrected policy: Navigation/preserve: retain the original target/occluding-furniture pairs and original room. No object grasp is required by occluder_side_patrol. Exact per-source grouping/binding keys, source geometry proxies, unchanged failure records and receipt paths are in the [independent class audit](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/category_grasp_audit/categories/b74d423c6c28efb78626/audit.json). Do not move occluders to create convenient grasps; native wheel support, collision-free approaches, preserved pair geometry and completed visits remain required.

Source-height risks: 0 task-bearing instances have at least one required interaction root/AABB midpoint below 0.35 m world Z; 0 have one above 1.10 m; 0 lack some required geometry. These are **source world-height proxies**, not height relative to a measured floor, actual grasp-contact points or native support/IK checks. The native initializer must use measured floor-relative geometry and the unchanged Rigid screening window (0.35–1.10 m, stance distance at most 0.85 m). Navigation has no required grasps; articulation body heights do not substitute for handle-link measurements.

Width evidence: No source horizontal AABB width is available for the required manipulation bindings. 0 required manipulation bindings lack that width evidence. A global AABB extent is not an oriented grasp width, and no native aperture/IK or executed grasp is inferred.

Existing native evidence: 10 matching current-candidate receipts and 10 instances with existing indexed images. They record 0 executed qualification gates and 0 verified grasps. Their frozen implementations predate this new profile; they do not validate the proposed repaired layout. Visual inspection covered 6 existing frames from 3 instances: Existing views show occluding furniture and some floor targets in different rooms. Heads look into the original rooms. These static views do not establish all occluder-side visits or a complete count.

- `category_4746320ea9818dd86c0d8977` / global_diagonal: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_29493fcb5438e060eacdfd0c79e8607c/attempts/01/global_diagonal.png)
- `category_4746320ea9818dd86c0d8977` / head: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_29493fcb5438e060eacdfd0c79e8607c/attempts/01/head.png)
- `category_4a7012649bcee0f614e9f455` / global_diagonal: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_171930e1190e5c7ed605e0cf82428690/attempts/01/global_diagonal.png)
- `category_4a7012649bcee0f614e9f455` / head: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_171930e1190e5c7ed605e0cf82428690/attempts/01/head.png)
- `category_0060c8a276faef8e1bd82a03` / global_diagonal: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7364d5ff832bae94b7969c217637d711/attempts/01/global_diagonal.png)
- `category_0060c8a276faef8e1bd82a03` / head: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7364d5ff832bae94b7969c217637d711/attempts/01/head.png)

The retained profile warning distribution is `{}`. Unresolved support, native state, mounted-camera coverage, actual gripper aperture, collision-free IK path and execution remain pending. A candidate or screenshot is not a successful physical episode. The source/failure audit and all old evidence remain available for Review; no rendering has been resumed.


## Supplementary assessment comparison after placement revision (2026-10-15)

Review fingerprint: `cdcd1ec779d358fe4f405eb2bb8e6cbb4cbb13ab87d8ba8fc51cb1e389a2f835`.
This addendum records the current read-only review; earlier sections remain
historical observations. The review plan was to compare the specification and
supplementary assessment against current construction code, all source and saved
candidate definitions, CPU evidence and direct CaptureQueue diagnostics, then
record recommendations for Main without changing execution.

The current batch contains **30 generated / 0 corpus conversion failures**,
with **17 patrol / 13 patrol-return** recipes across 12 scenes and 17 scene/room
pairs. All 30 source answers agree with their payload answers and generated
target counts: six two-target, 22 three-target and two five-target sources;
one source contains irons and 29 contain sandwiches. All current private
bindings, expected geometry and target/anchor pairings match source data.

There are now **120 saved candidates**, representing four revisions of each
source, not 120 independent source tasks. Every scene/task/QA/record definition
was read; all 120 scene/task pairs pass the current schema, remain pending,
retain empty original variants and matching scene/task overlays, and preserve
source QA privately. The assignment has no `existing_task_files` entries.

| Group | Count | Relevant difference |
| --- | ---: | --- |
| Previous 1.0.0 | 30 | Robot spawn is floor + 0.02 m; no angular visit limit. |
| Previous early 1.1.0 | 30 | Spawn is source floor; no angular visit limit. |
| Previous final 1.1.0 | 30 | Adds 0.10 rad/s angular visit limit; robot-pose patch only. |
| Current placement 1.1.0 | 30 | Adds `prepare_category_interaction` after postprocessing, navigation/preserve, no manipulation bindings. |

The two saved batch snapshots and the placement history manifest were also
read. The malformed-source CPU diagnostic has one source-inspection failure
and no scene/task definition; it remains a separate validation copy, not an
additional corpus failure. Older definitions lacking the angular parameter
must not be treated as compatible with today's strict operation validator
solely because their schema or version label is valid.

At **2026-10-15T04:20:40.182494+00:00**, the current placement revision has
**30 QUEUED jobs, zero attempts and no receipts**. Its parent published revision
has **10 REVIEW_PENDING / 20 QUEUED** jobs. No assigned-category job is active
or failed at this observation boundary. All 60 published candidate records
remain available. The five entries in `sample_evidence.json` still report
QUEUED for previous candidates, even though their native captures now exist.
The earlier nine-capture count and missing-image statements above are historical.

Ten previous-revision receipts match their input identities and all nine
expected views; **90 PNGs are readable**. Every receipt records zero model calls
and robot actions, `native_qualification_run=false`,
`eligible_final_layout=false` and `diagnostic_pending=true`.
REVIEW_PENDING establishes review evidence availability. It establishes neither
approval nor a completed patrol. Capture durations are **174.51 / 294.72 /
501.16 seconds** minimum/median/maximum, n=10, excluding queue waiting.
These are initialization/screenshot costs, not task execution latency.

The converter hash still matches the saved CPU validation. All nine inspected
relevant current code files match assigned frozen code
`working_1851419e3a9f2d2c9510e54a57d02e38`, including the shared initializer,
schema, category runtime and counting builder. This provenance does not make
parent screenshots evidence for the new initializer. Saved placement validation
reports 30 navigation/preserve policies, no problems and 8.6142 seconds, with
no native execution. The earlier 19 CPU checks and 0.1466-second replay are
saved results; this reviewer reran only read-only schema/source checks.
The saved synthetic return trajectory contains two positive records followed
by layout-change rejection. These fixtures were not rerun and are excluded
from native evidence.

Both recipes provide meaningful candidate physical operations. They require
guarded base motion, measured departure/travel, stopped visits to original
target-facing furniture sides and preserved scene geometry. Returning to the
measured departure adds a useful route/memory requirement. Several recipes
are useful when they add distinct source-faithful requirements; recipe choice
here is deterministic, not based on demonstrated feasibility. Grasping the
visual-only floor targets or moving occluding furniture is unsupported.

The independent native predicate is appropriate for **geometric access
patrol**. Qualification reads actual bindings, room, collidable anchors,
source geometry, floor contacts and wheel-floor rays. Episode evaluation reads
pose changes, standoff/side geometry, continuous sampled movement, contact
safety, layout preservation and stopped visit/return history. It does not use
source answers or construction success flags, and no scoring hook moves the
robot or objects. It has no camera heading, line-of-sight or pixel-coverage
test. Access success therefore cannot certify counting or visibility.
Three stability steps are controller samples, not Agent decisions or a fixed
dwell time; thresholds and the 600-decision budget remain uncalibrated.

The current initializer can change R1 stance and yaw after the converter's
proposed pose while preserving the original target/anchor group. Its native
stance search uses floor-map index 0; wheel AABB bottoms are placed 0.005 m
above that floor before simulation steps. Actual contacts, source-room/floor
consistency and fixed-base R1 support need measurement. This is an untested
diagnostic concern, not an observed failure. No current screenshot or native
qualification exercises this patch. Zero required-grasp height/width risk
flags are expected because this task declares no grasps; they demonstrate
neither reach nor aperture feasibility.

The private/public data boundary is sound in the inspected definitions, but
**initial auxiliary QA is not a complete evaluation protocol**. The source
question is saved in `public.environment`. Although `task_text()` omits
environment, the full Agent path passes `public_payload` and injects
`PUBLIC_ENVIRONMENT` on the first turn. Thus a claim that the question can
never reach the Agent would be incorrect. Delivery alone supplies no initial
answer phase, authorized image history, answer format, scorer invocation or
timing relative to patrol. All 30 source questions say “both images”, while
each saved `render.all_images` lists one image; none of those primary images
is present beside its source JSON. Other stores were not exhaustively searched.
Keep this source inconsistency visible. The original exact and soft count
scores, including `max(0, 1 - absolute_error/3)`, remain separate from physical
success. GT maps, corner-camera metadata, target closeups and human review
images must not become Agent sensors.

The supplementary assessment's instruction to retain targets, occluders and
anchors is supported. Its representative corrected probe reports three targets
and 7/7 bounded control readbacks; its original failure was a target-velocity
diagnostic-interface error. Neither proves a current patrol. Historical raw
Counting probes under `/scratch` are unavailable locally. The report pins
ESI `f35b0de448ee77bc46714ae84b5190c17f87d86d` and an A40-server protocol;
current captures pin `9f6f15ed43e81654a97de9e4db2a1fedadca2bc1` on deep59
with different cameras/workloads. Their durations and control outcomes are
not a controlled comparison.

Preservation also needs a precise scope. The active builder loads floors,
walls and source-anchor categories; FULL_SCENE removes room filtering, not
this category filter. The original generator loads broader scene content and
removes robots. Missing background furniture and extra instances of included
categories can change occlusion and route difficulty. Current source/definition
agreement supports target/paired-anchor reuse, not complete original-room or
original visual-difficulty equivalence. Source XY segment/AABB occlusion labels
are not native visibility measurements.

Six parent-revision overhead/head images were reopened across the iron,
five-sandwich and two-sandwich sources. The kitchen overhead shows five
sandwich appearances with edge clipping; its head faces bar/body. The office
views are dominated by desk/chair/body, with a sandwich partly visible at the
overhead edge. The inspected iron pair does not establish all three targets.
Other mounted cameras and future viewpoints were not exhaustively assessed.
Existing logs also retain ceiling convex-mesh errors for two office captures,
and a Vulkan crash/pagefault after environment closing for one Beechwood
capture despite a complete receipt and exit 0. Their effect on future physics
is unknown; neither anomalies nor static images justify target relocation.

Recommendations for Main:

1. Define explicit initial sensor/answer/scoring boundaries before reporting
   auxiliary QA results, accounting for the source image-template inconsistency.
2. Retain the two warned access recipes. Add any visibility/heading-aware or
   lateral inspection recipe as a separately evidenced revision and compare
   success, route distance, decisions and runtime under matched settings.
3. During future authorized native work, measure the new initializer's actual
   stance, wheel support and floor/room consistency, then test visit/return
   trajectories. Preserve unsupported instances as diagnostics.
4. Review whether omitted background furniture must be restored for the intended
   perception comparison. Any implementation correction belongs to Main review
   and a new reproducible revision.
5. Keep evidence revision-aware and retain all pending, failed and previous
   candidates in Review. This comparison introduces **no Review/capture blocker**
   or new approval prerequisite; it does not request rendering to resume.

Only this report was appended and the
[structured review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/b74d423c6c28efb78626/cdcd1ec779d358fe4f405eb2/result.json) was written.
No implementation, generated scene/task, source/answer, approval, other category
report, root research log, queue state or running workload was changed.
No GPU work, commit or sub-agent was launched. No robot success or human
approval is claimed.


## Supplementary assessment comparison after runtime refresh (2026-10-15)

Review fingerprint: `ba72a75ec561042388c52b2979524d9f924ee687b1acec017aa5118160a2bd97`.
The two access-patrol recipes remain meaningful candidate operations, with
unverified native execution. Main should review a newly reproduced native-vector
compatibility problem and the incomplete initial QA protocol. This assessment
adds no prerequisite for pending, failed or diagnostic candidates to enter Review,
and does not block ongoing screenshots.

The review plan was to compare the specification and supplementary assessment
with the current converter/builders, all source and candidate revisions, saved
CPU evidence, initial QA delivery and direct CaptureQueue diagnostics. These
read-only checks and one isolated CPU vector reproduction were performed here.
No conversion, simulator, robot episode or model experiment was run.

### Complete scope and revision identity

The current batch still covers all **30 sources**, with **30 generated / zero
corpus conversion failures**, **17 patrol / 13 patrol-return** recipes, 12 scenes
and 17 scene/room pairs. All records retain pending definitions, L2 assumptions
and explicit native-validation warnings. Target counts are six two-target,
22 three-target and two five-target sources; 29 sources use sandwiches and one
uses irons. Current source answers, payload answers and constructed counts agree.
Current private target bindings, target/anchor pairings and expected geometry
match the source metadata.

All earlier candidate revisions were read as previous tasks. There are now
**150 candidate records referencing 120 distinct scene/task pairs**:

| Revision group | Records | Relevant difference |
| --- | ---: | --- |
| Previous 1.0.0 | 30 | Spawn at source floor + 0.02 m; no angular visit limit. |
| Previous early 1.1.0 | 30 | Spawn at source floor; no angular visit limit. |
| Previous final 1.1.0 | 30 | Adds angular limit 0.10 rad/s; robot-pose patch only. |
| Previous placement 1.1.0 | 30 | Adds navigation/preserve initialization after postprocessing; no manipulation bindings. |
| Current runtime refresh 1.1.0 | 30 | Reuses placement scene/task bytes under new candidate and code identities. |

Every referenced scene/task pair passes the current schema, has matching
scene/task patches and an empty original variant, and preserves its source
answer/options/encoded metadata in private QA. All seven category batch/history
manifests were read. The separate malformed-source CPU diagnostic remains one
source-inspection failure with no task/scene; it is not a corpus conversion
failure. The assignment has **zero existing_task_files entries**.
Schema acceptance does not make old parameter sets compatible with today's
strict operation validator.

The assignment pins `working_1851419e3a9f2d2c9510e54a57d02e38`, whereas the
current canonical batch and its 30 queue inputs pin
`working_bc47e71e705362f1c72346462a65c814`. Runtime-refresh source/scene/task
hashes still match their recorded unchanged inputs. The converter hash remains
`fd1000e234a5189a74f540cddd6c709db2c7e0338d60b2c822567c6e7cce7d4c`,
matching saved CPU validation and both frozen snapshots. Of nine relevant code
files compared, the current shared interaction initializer differs from the
assignment snapshot and matches the runtime snapshot; the other eight match
both. Its added structured-xyz conversion is a real implementation difference,
despite unchanged task bytes/version labels. Earlier screenshots do not validate
this new initializer.

### Current CaptureQueue evidence and unavailable artifacts

At **2026-10-15T05:05:23.166936+00:00**, direct category reads found:

| Published revision | REVIEW_PENDING | QUEUED | Active / failed |
| --- | ---: | ---: | ---: |
| Previous final converter | 10 | 20 | 0 / 0 |
| Previous placement | 0 | 30 | 0 / 0 |
| Current runtime refresh | 0 | 30 | 0 / 0 |

All 90 published records remain present. The shared queue control file now says
**RUNNING**, with formal rendering resumed; older statements in this report and
the batch's cached `rendering_paused_by_user` flag describe earlier observations.
There was no active assigned-category capture at this boundary; that does not
describe other categories or subsequent scheduling. No workload was changed here.

Ten prior-revision receipts match candidate, signature, fingerprint and all nine
expected view names. Each records zero model calls/actions,
`native_qualification_run=false`, `eligible_final_layout=false` and
`diagnostic_pending=true`. Recorded capture duration is **174.51 / 294.72 /
501.16 s** minimum/median/maximum, n=10, excluding queue waiting. These costs
cover initialization/screenshots, not patrol latency or quality.

Receipt completion must also be separated from present artifact availability.
At **2026-10-15T05:06:56.662656+00:00**, only **54 of 90 indexed PNGs** were
readable. All nine images were unavailable at their indexed paths for each of:

- `capture_29493fcb5438e060eacdfd0c79e8607c` (iron sample);
- `capture_171930e1190e5c7ed605e0cf82428690`;
- `capture_7364d5ff832bae94b7969c217637d711` (cafe sample);
- `capture_9e6033402967e2f8383bd67d754fb747` (five-target kitchen sample).

Six parent overhead/head images from the iron, kitchen and office samples were
opened earlier in this review. The iron pair did not establish all three targets;
kitchen overhead showed five sandwich appearances with edge clipping, while
head framing faced the bar/body; office framing was dominated by desk/chair/body,
with a sandwich at the overhead edge. Iron and kitchen PNG paths subsequently
became unavailable during this review. The cause, alternate copies and permanence
of that change were not established. Earlier claims of 90 readable images were
historical measurements and are not current availability guarantees. No image
was deleted, moved or regenerated by this reviewer.

The five entries in `sample_evidence.json` still cache QUEUED/no-image states
for the previous final-converter candidates. Their receipts now exist; three of
those five jobs presently lack indexed PNGs. This cache describes neither the
new runtime candidate IDs nor current file availability. Existing logs retain
office ceiling convex-mesh errors and a Vulkan crash/pagefault after environment
closing in `capture_23af8829fc143361d50e13d20d9cd3c6`, despite its complete
receipt/exit 0. Their effect on later physics remains unknown.

### Operation and predicate assessment

Both recipes require real guarded base motion, at least 0.50 m departure and
1.0 m sampled travel, visits on the target-facing side of each original occluder,
safe standoff/floor support, unchanged scene geometry and stopped control samples.
Returning to the measured departure adds a useful route/return requirement.
Several source-faithful recipes are useful when they add distinct requirements;
these two do so, although hash-based assignment demonstrates no feasibility
advantage. No evidence currently supports manipulating visual-only floor targets
or moving the occluding furniture. A visibility-aware lateral inspection recipe
could be a separate future revision after measurement.

The intended terminal predicate is independent of answers/construction success
flags and reads native poses, AABBs, floor rays/contacts, room membership and
guarded-controller step history. The shared runtime seals private results to
the exact task/source hashes and observation boundary. Qualification and scoring
hooks do not move objects. This is appropriate in principle for **geometric
access**, with an implementation compatibility concern below.

It does not require camera heading, line of sight, image pixels, different visit
positions or actual traversal around each occluder. One stance can credit several
targets if all geometric tests pass. Therefore physical access cannot certify
counting, visibility or preservation of the original perceptual difficulty.
Three stable steps are controller samples, not Agent decisions or a declared
duration. Thresholds, floor-support feasibility, complete routes and the
600-decision budget remain uncalibrated.

**New CPU-reproduced compatibility finding:** the owned converter's `_array`
still uses `np.asarray(value, dtype=float)`; `_native_state` passes contact
position/normal through `_vector`. A scalar NumPy xyz structured record with
float32 fields raises **TypeError: Cannot cast scalar from dtype ... to
dtype('float64')** in these exact helpers. The current shared
`interaction_setup.array` converts the same input to `[1.0, 2.0, 3.0]`.
Installed OmniGibson contact readers preserve raw contact position/normal fields
while decoding body names. This establishes incompatible handling of the native
record representation addressed by the runtime refresh. It does **not**
establish a measured category-native failure or its frequency: no current
category qualification has run. If such a contact reaches the owned reader,
qualification catches it as `native_patrol_evidence_unavailable`; evaluator
sampling has no equivalent catch. Main should review a compatible read-only
conversion and a small representation check, preserving all numerical criteria.
No correction was applied here.

The navigation initializer still selects traversability floor index 0, may
replace the proposed robot stance/yaw, and checks actual floor contacts after
placing wheel AABBs 0.005 m above that floor. Its navigation/preserve policy
does not authorize object relocation. Source-room/floor consistency and
fixed-base R1 wheel support require native evidence for this revision.

### Initial QA and historical comparison

The definition-level boundary remains sound: original QA/GT geometry, maps,
camera metadata and scorer state stay private. The public source question is
in `public.environment` and reaches the first Agent turn through
`public_payload` / `PUBLIC_ENVIRONMENT`, even though `task_text()` omits it.
However, there is still no declared initial QA answer phase, authorized image
history, answer format, scoring invocation or timing relative to operation.
All 30 questions say “both images”; every saved `render.all_images` has one
image, and none of the primary PNGs exists beside its source JSON. Other stores
were not exhaustively searched. Original exact count correctness and soft score
`max(0, 1 - absolute_error/3)` must remain separate from native patrol success.
Preserving the QA archive alone is not a measured initial QA result. Human
overviews, source closeups and GT cameras must not become Agent sensors.

The supplementary assessment correctly cautions against spreading targets or
moving occluders and does not certify full body-view coverage. Its corrected
representative reports three targets, 7/7 bounded controls and 177.2/199.3 s
readiness/total; its original failure concerns
`RigidKinematicPrim.get_linear_velocity`, a diagnostic interface error.
Neither establishes a current patrol. The linked Counting raw probes under
`/scratch` remain unavailable locally. Its A40 protocol and ESI commit
`f35b0de448ee77bc46714ae84b5190c17f87d86d` differ from deep59 captures at
`9f6f15ed43e81654a97de9e4db2a1fedadca2bc1`; timings/control results are not
a controlled comparison.

Preservation has a narrower demonstrated scope than “the original room”:
the active builder loads floors, walls and paired-anchor categories.
`FULL_SCENE=True` removes room filtering, not that category filter.
The original generator uses broader scene contents and removes robots.
Omitted furnishings and extra instances of included categories can alter
occlusion and route difficulty. Current source agreement establishes
target/paired-anchor reuse, not complete original-room equivalence. Source
XY segment/AABB labels are not measured native visibility.

Saved CPU validation reports 19 checks and 0.1466 s replay; placement validation
reports 30 policies, no problems and 8.6142 s, with no native execution.
The stored positive-return synthetic trajectory has two positive records and
a later layout-change rejection. These are historical CPU fixtures, not native
success, and were read without rerunning conversion or the synthetic evaluator.
The new vector reproduction is also CPU evidence only.

### Recommendations for Main

1. Review the owned native contact-vector conversion before physical evaluation;
   retain fail-closed behavior, geometry thresholds and safety rules in any new
   revision.
2. Define the initial auxiliary QA snapshot, lawful sensors, answer phase and
   original independent scoring before reporting QA metrics; account explicitly
   for the source image-template inconsistency.
3. Retain both warned access recipes. During future authorized native work,
   measure stance/floor support, complete visit/return trajectories and cost;
   evaluate a separate visibility-aware variant only with appropriate evidence.
4. Check omitted background furnishings before claiming original visual
   equivalence, and reconcile missing indexed PNGs/cached sample status through
   the main-owned evidence path. Preserve historical receipts and revisions.
5. Continue existing Review/capture flows for every pending, failed and
   diagnostic candidate. These recommendations add no Review/capture blocker
   and confer no readiness, robot-success or approval status.

Only this owned report was appended and the
[structured comparison](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/b74d423c6c28efb78626/ba72a75ec561042388c52b29/result.json) was written.
Implementation, generated definitions, source data/answers, approvals, other
reports, root research log, queue state and running workloads were not changed.
No GPU work, commit or sub-agent was launched.




## Supplementary assessment comparison for the selected native sample (2026-10-15)

Review fingerprint: `5dbe115eb2928b502583153a8b8e61ed435b84cbd4d400b3a1649e8b6df41db9`.
This comparison is **reviewed and non-blocking**. Both access recipes remain
meaningful pending operations; initial QA evaluation and native execution remain
unverified. Recommended corrections belong to Main. Existing Review, failure
diagnostics and authorized screenshots acquire no additional prerequisite.

The review plan was to compare the specification and supplementary assessment
with current converter/builders, all sources and previous candidate definitions,
saved CPU/sample evidence, initial QA delivery and direct queue diagnostics.
This reviewer performed read-only artifact/schema/source checks and one isolated
CPU vector-representation reproduction. No conversion, synthetic episode,
simulator, robot operation or model experiment was run.

### Coverage and revision identity

All **30 sources** have generated pending L2 definitions: **17 patrol / 13
patrol-return**, across 12 scenes and 17 scene/room pairs. There are six two-target,
22 three-target and two five-target sources; one uses irons and 29 use sandwiches.
Source/payload answers, target counts, current target/anchor pairings and public
source questions agree. Definition checks found no problems with schema,
matching scene/task overlays, empty original variants or preserved private QA.

The nine batch/history manifests and every referenced previous definition were
read. There are **151 task-bearing candidate identities referencing 120 distinct
scene/task pairs**, plus one separate malformed-source CPU diagnostic. Five
earlier groups contain 30 identities each: 1.0.0, early 1.1.0, final 1.1.0,
placement, and runtime refresh. The selected native sample adds one identity.
The assignment has **zero existing_task_files entries**.

The assigned 30-record batch is a mixed revision: only
`category_4f40e2996fe964b74249df44` uses frozen code
`working_2fc97c3d67769295ed5e7bf6e846fa53`; the other 29 still use
`working_bc47e71e705362f1c72346462a65c814`. All 30 reference unchanged
placement scene/task files. The canonical category batch retains the earlier
30 runtime-refresh identities. These are neither 30 new designs nor 30 new
native validations. Earlier definitions lacking the angular-speed parameter
are not runtime-compatible merely because they pass today's schema.

The converter still matches its saved CPU-validation hash and assigned frozen
code. Eight of nine relevant compared code files match that snapshot; current
shared `interaction_setup.py` differs. Its new stance fallback/evidence
mainly concerns manipulation targets, which this navigation category does not
declare; no changed navigation outcome is established. Keep actual snapshot
identity with evidence instead of relying only on version 1.1.0.

### Current CaptureQueue measurements

At **2026-10-15T06:51:27.028717+00:00**, the assigned-category inventory was:

| Published group | REVIEW_PENDING | QUEUED | FAILED |
| --- | ---: | ---: | ---: |
| Previous final converter | 10 | 20 | 0 |
| Previous placement | 0 | 30 | 0 |
| Previous runtime refresh | 0 | 30 | 0 |
| Selected native sample | 0 | 0 | 1 |

All 91 published cards remain present. No category job was active at this
boundary; this does not describe other categories or subsequent scheduling.
All 80 queued jobs have held-for-category-review markers. Existing controls
say RUNNING globally while allowing only selected category samples.
Cached “rendering paused” metadata describes an earlier state. This reviewer
changed neither the existing sample gate nor any workload.

The selected sample's
[status](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_f7fc40392c778f496a91ea5f816ba1c2/status.json)
records one **resident_capture failure**, duration **32.57 s**, with an error
reporting a **5-second subprocess timeout**. Its attempt contains CPU preload
and `capture_failure.json`, with no images, receipt or native log at the
indexed path. CPU preparation reports 101/101 files read, 89/89 requested assets
available, about 406 MB read and 6.94 s; it explicitly reports no native
construction or readiness. Preparation inventory filtering does not demonstrate
a change to native scene contents.

The referenced service retains an earlier READY record and a later exit record
with code -15, but no request for this category job. Code can retire an existing
service before publishing a new request, and its process-group helper ends
with `process.wait(timeout=5)`. The retry path recognizes built-in
`TimeoutError`, whereas the observed command-timeout representation is
consistent with `subprocess.TimeoutExpired`. A retirement/wait/retry
problem is therefore a **diagnostic hypothesis**, not an established throw site
or root cause. This failure does not demonstrate an unusable source scene or
a failed physical patrol. Main should review the lifecycle evidence; no retry
was launched here.

Ten older complete receipts still match candidate, fingerprint, signature and
all nine expected view names. They record zero model calls/actions,
`native_qualification_run=false` and `eligible_final_layout=false`.
However, **0 of their 90 indexed PNGs are currently readable**. No native image
was visually inspected in this review. Earlier report observations of 90 or 54
readable images, and their visual descriptions, remain historical. Alternate
copies, disappearance causes and permanence were not established.

The ten saved capture durations are **174.51 / 294.72 / 501.16 s**
minimum/median/maximum, excluding queue waiting. They measure initialization
and screenshots, not patrol latency or model quality. The five entries in
`sample_evidence.json` still cache older QUEUED/zero-attempt/no-image states;
they represent neither their completed receipts nor the selected failure.
Existing logs still show ceiling convex-mesh errors in the inspected office
capture and a Vulkan crash/pagefault after environment closing in a Beechwood
capture despite its saved complete receipt/exit 0. Their physical effects remain
unknown; they do not justify moving targets.

### Operations, predicates and initial QA

Both recipes require real guarded base motion, at least 0.50 m departure,
1.0 m sampled travel, stopped visits on the target-facing side of each original
occluder, safe standoff, floor support and preserved geometry. Returning to
the measured start adds a useful distinct navigation requirement. Several
source-faithful recipes are useful when they add such requirements; deterministic
assignment alone demonstrates no feasibility advantage.

The native predicate is independent of source answers and construction success
flags. It reads poses/AABBs, room membership, floor rays/contacts and guarded
controller-step history; the shared envelope binds private results to the
task/source and observation boundary. It is appropriate in principle for
**geometric access**, subject to compatibility and native validation. It does
not require camera heading, line of sight, pixel evidence, distinct visit
locations or actual travel around every occluder. One stance can credit multiple
targets. Three stability steps are controller samples, with no declared dwell
duration. Access therefore cannot certify visibility or counting correctness.

The previously reported contact-vector concern remains reproducible:
the exact owned `_array/_vector` bodies raise TypeError on a scalar NumPy xyz
float32 structured record; shared `interaction_setup.array` returns
[1.0, 2.0, 3.0]. Installed OmniGibson decodes contact body names while retaining
raw position/normal fields, which the owned native reader sends through
`_vector`. Qualification catches errors as missing native evidence;
evaluator sampling has no equivalent local catch. This is a CPU compatibility
finding, **not a measured category-native contact failure or an explanation
of the resident timeout**. Main should review a compatible read-only conversion
without changing numerical thresholds or guards.

The initialization policy remains navigation/preserve, with no manipulation
bindings. Its native stance search may replace the proposed R1 pose/yaw and
uses floor index 0. Actual source-floor/room consistency, fixed-base R1 wheel
support, full routes, thresholds and the 600-decision budget remain unmeasured
for this revision. Parent screenshots and the failed selected attempt do not
qualify it. Saved 19-check CPU validation, 0.1466 s replay and 8.6142 s placement
validation were read, not rerun. Two positive synthetic return records followed
by layout-change rejection remain explicitly synthetic evidence.

The information boundary is sound in the inspected definitions: answers,
encoded metadata, target identities/geometry and scorer state remain private.
The public source question reaches the first Agent turn through
`public_payload` and `PUBLIC_ENVIRONMENT`. Yet there is no explicit initial
answer phase, lawful image-history boundary, answer format, scorer invocation
or timing relative to patrol. All 30 questions refer to “both images”, while
each `render.all_images` lists one image and every primary PNG is absent beside
its source JSON. Other stores were not exhaustively searched. Original exact
correctness and soft score `max(0, 1 - absolute_error/3)` must remain separate
from physical success. Preserving QA files supplies no measured initial QA
result. Source GT views/maps, target closeups and human review images cannot
be substituted for Agent sensors.

### Comparison and recommendations for Main

The supplementary assessment correctly recommends preserving targets, occluders
and anchors, and does not certify complete body-view coverage. It reports one
corrected three-target probe with 7/7 bounded controls and 177.2/199.3 s
readiness/total; its original failure was a target-velocity diagnostic-interface
error. These historical results were not rerun. Linked Counting raw artifacts
under /scratch are unavailable. Its A40 protocol and ESI revision differ from
current deep59 captures; timings and controls are not a controlled comparison.

Preservation is demonstrated for targets and paired anchors, not the complete
original room or perceptual difficulty. FULL_SCENE removes room filtering,
while the active builder still loads only floors, walls and anchor categories.
The original generator loads broader scene content without robots. Omitted
furniture and extra instances of included categories can change occlusion and
routes. Source XY segment/AABB occlusion labels are not measured visibility.
No current evidence justifies grasping visual-only targets or moving furniture.

Main should review the contact-vector compatibility and resident lifecycle
diagnostic, define explicit initial QA boundaries, and reconcile missing PNGs
and stale/mixed-revision evidence through the existing reporting path. Retain
both warned access recipes; any visibility-aware lateral inspection should be
a separate evidenced revision. Future authorized native work should measure
start support, complete visit/return histories and runtime before reporting
operation results. These recommendations **add no Review or capture blocker**.

Only this report was appended and the
[structured review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/b74d423c6c28efb78626/5dbe115eb2928b502583153a/result.json) was written.
Implementation, generated definitions, source data/answers, approvals, other
reports, root research log, queue state and running workloads were unchanged.
No GPU work, commit or sub-agent was launched. No robot success or human
approval is claimed.


## Actual accessible-placement generator audit — 2026-10-15

Converter 1.2.0 retains the existing real access-patrol operations and numerical motion/stop thresholds. The original generator places portable targets at floor top plus 0.035 m behind furniture. The 30 sources contain 88 target/anchor associations across 21 furniture categories, including building-like fixtures; source floor placement is historical evidence, not an accessibility exemption.

Only portable targets enter `mode=inspection` and `independent_groups`. Targets associated with the same original furniture region remain grouped; anchors stay fixed. `support_preferences` names each target's original bound furniture as the first measured support candidate, followed by other real surfaces and the authorized small simple fallback. These are preferences, not claims that every furniture top is a valid support. Original target assets are dynamic/collidable before construction and remain collidable after the source postprocessing reset. The Agent still receives base controls only; the height-screen role does not impose a grasp goal.

A target placed on its original chair/table may lie inside that anchor's XY bounds, making the old target-facing side vector undefined. Under the approved `accessible_placement=True` branch, actual measured standoff around the placed target is sufficient, while real furniture clearance, wheel/floor contact, measured travel and stopped visits remain mandatory. Historical configurations without this flag retain the exact original side predicate. Public text explicitly describes accessible target patrol and possible source-occlusion/reference changes. Qualifiers use accepted adapted references, current physical support and actual height; later target drift or support loss invalidates the patrol. Native contact-vector decoding now reuses the shared structured-xyz reader without changing any numerical guard.

The complete category source count is 30; current CPU status counts are `{"generated": 30}`. The changed whole-class CPU batch is [batch_manifest.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_b/b74d423c6c28efb78626/batch_manifest.json). The deterministic representative proposal is `category_33c2c4876255138cf5795a36` from `Enumerative Perception/Counting w Occlusion/Beechwood_0_garden/living_room_0/q_004.json`, preferring the previously selected annotated source. Every old source/failure remains traceable in the original batch and the source audit; pending regeneration is not Native qualification.

The [nine-class audit summary](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_b/audit_summary.json) and [source/task audit](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_b/source_task_audit.json) cover all 1020 source records and 977 previous task definitions. All original source hashes and private QA values match, with no public native-identity leaks. Two lightweight prior sample directories lacked QA copies; their declared task-file companion QA packets were read successfully. No historical QA packet was rewritten.

The current user-approved accessibility plan takes precedence over earlier preservation-only advice. The [supplementary assessment](</local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md>) is advisory and does not justify keeping low portable props. Existing measured supports come first, with only the authorized small simple support fallback. Source assets, scales, private QA and original variants are preserved; independent groups use accepted per-object references rather than raw source-floor poses. Main owns shared native surface/contact/spawn integration and publication.

All three supplied [Rigid Containment reference definitions](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/rigid_references/references.json) were read as positive examples of reachable existing/new supports and actual pickup/insertion goals. Their historical resizing was not copied. Reference 01's declared global PNG was missing when requested, and 02/03 had no declared screenshots yet; no replacement view was invented or directory scanned. This pass successfully inspected only the declared Structural global frame and Segmentation global/head frames mentioned above. Original approved reference definitions and images were untouched.

Validation: behavior Python ran `tests/test_accessible_generators_b.py`, `tests/test_annotation_perception_categories.py` and `tests/test_esi_interaction_setup.py`: **38 passed in 0.63 s**. All nine assigned converters and the new test file passed `py_compile`. These CPU fixtures establish serialization, grouping, current reference/support reads, compatibility branches and information separation; they do not establish Native reach, visibility, grasp or successful operation. No GPU, simulation episode, enqueue, Review mutation, canonical YAML edit or screenshot deletion was performed by this agent. Main owns one representative capture per changed subclass and later superseded-cache cleanup; remaining queue work stays unchanged.


## Latest Native representative evidence — 2026-10-15

The [current Native results](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/native_results.json) identify candidate `category_b1d22ca978417e7d3dad8e35`, job `capture_a2a1489719bd684ecb4db46bcfd8361b`, attempt **02**, under frozen code `working_3f808b49441ea3c09dbdbea2cbb941d6`. Its [declared receipt and interaction metadata](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_a2a1489719bd684ecb4db46bcfd8361b/attempts/02/preview_complete.json) report interaction initialization **accepted**, capture `complete=true`, nine declared views, and queue state `REVIEW_PENDING`. The [declared native log](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_a2a1489719bd684ecb4db46bcfd8361b/attempts/02/native.log) was read; this report pass did not open, validate, or scan images.

Attempt **02** completed after an archived infrastructure failure before scene construction. The [retry authorization and preserved prior status](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/worker_incident/worker3_infra_retry.json) attributes attempt 01 to an old in-memory supervisor propagating a five-second resident-shutdown `TimeoutExpired`; that attempt is not a measured placement rejection. Its declared old native-log path is currently absent, so this appendix relies on the preserved failure status and does not invent missing log contents. The final attempt's declared native log was read.

All three independent iron groups now have `accepted=true`, with no unrepaired names and `partial_accessible_layout=false`. The first target rests on original `breakfast_table_skczfi_0`, with bottom **0.708216 m**; the other two use original `coffee_table_qlmqyy_0`, with bottoms **0.402001 m** and **0.402083 m**. The final trial records native task-support contact, a floor-supported collision-free robot stance, and `fixed_background_geometry_preserved=true`.

The same accepted trial records `background_geometry_preserved=false`. Dynamic background actor `coffee_table_rkdwxe_1` has a reported final vertical displacement of **-43.816078 m**, under `dynamic_background_settled_during_initialization_requires_review`. This large displacement is an explicit review limitation, not harmless settling or evidence that all furniture stayed unchanged. `source_occlusion_adapted_for_accessibility`, `initial_base_approach_required`, and `native_reach_and_route_unverified` also remain. No actual target patrol, route, stop, or original-answer performance was measured.

This receipt remains `diagnostic_pending=true`, `native_qualification_run=false`, and `eligible_final_layout=false`. Interaction initialization/contact evidence is separate from the unrun complete task qualification and autonomous operation. It records **zero model calls, zero robot actions, and `physical_grasp_verified=false`**. No autonomous grasp, articulation, patrol, navigation, or terminal task success is claimed. CPU coverage remains **30 sources, all 30 generated**; previous source answers, failures, reviews, and evidence histories remain unchanged. This appendix adds evidence only: no converter, generated task, batch pointer, queue, or image was modified, and no simulation was launched.


## Conservative goal composition (2026-10-16)

The canonical current generated inventory contains 23 active records and produces 19 composition groups for this subclass. Across the nine perception subclasses, 882 active scored records produce 866 groups through exactly 16 audited pairs. Repeat/archived records and conversion failures are not reactivated. Native screenshot failures remain active and require explicit blocked-goal evidence.

Four audited pairs share geometry but retain patrol and patrol-return predicates. Record the short-goal terminal evidence while stopped at a qualified approach, before continuing home for the return goal. Checking both only at the final home pose would lose valid short-goal completion.

Policy: [composition_perception.py](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/composition_perception.py). Each audited pair is revalidated against effective source construction, camera/anchor geometry, current scene and task initialization, bindings and original evaluator parameters. Only known non-operative provenance fields and the approved recipe difference are ignored. Changed geometry, scale, lighting configuration, target binding or robot pose prevents pairing. A changed composite task receives fresh Review; prior layout approval never supplies task approval.

The shared runtime preserves each original native reducer and its stage-specific completion evidence, every source/question association and separate QA score. Partial physical credit is the fraction of completed original goals; all-goals completion requires every member. Source answers and private evaluator state are never observations. This section reports CPU policy validation, not successful simulator execution.

Review checks:

- Verify a floor-supported collision-free robot start and a safe route to each required object or region.
- Verify native task support, unchanged object dimensions and stable initialization at reachable interaction heights.
- Verify the relevant ambiguity or occlusion using enabled robot-mounted sensors; Review images and source metadata are not Agent observations.
- Collect each original question response with its original options before irreversible handling or reveal; score QA separately from physical completion.
- Keep native child qualification failures visible as blocked subgoals; retain every original scored member.
- Verify original furniture, target-side approach geometry, chassis clearance and stable visits to every target.
- Latch the short patrol score at a valid stopped approach before returning; score the return member separately at departure.

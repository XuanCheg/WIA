# Liquid Volume conversion

## Research plan and authoritative inputs

Implement a deterministic CPU converter for all 136 assigned sources. Read the
current implementation specification, category advice, shared converter API,
task schema/runtime, original active builder, original generation builder and
source JSON before selecting rules. Preserve original fixed-base container
geometry, scales, relative arrangement, liquid setup, reference camera and QA.
Add actual ordered docking and round-trip navigation using the installed R1;
measure chassis motion, floor support, contacts, facing and stopping privately.
Native qualification must reject unsafe scaffold contacts. Do not use source
fill/pour callbacks, QA labels or saved capacities as operation success.

Validate full-category coverage, deterministic output, source preservation and
public/private boundaries on CPU. Inspect stratified current evidence when
available, otherwise inspect explicitly labelled CPU source-map diagnostics.
Publish every candidate and diagnostic for Review via the supervising campaign.
No GPU process or native episode is launched by this category supervisor.

The supplementary historical assessment is
[report(1).md](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md).
It reports zero measured fill particles, controller drift and scaffold contacts
in earlier probes. Those probes were not run by this supervisor; their raw
`/scratch` artifacts have not been established as locally available. Current
source corroborates robot-free construction, fixed-base containers, four
non-scene pool planes and scripted particle exile during pouring. Historical
runtime failures are not accepted as a proof that every navigation recipe is
impossible, and navigation does not certify capacity or transfer.

## Initial iteration

Hypothesis: retaining the original apparatus and adding robot-only, contact-
qualified docking provides a source-faithful physical navigation operation
without inventing liquid measurements or modifying capacity-sensitive assets.
The original map confirms that the source apparatus at XY origin is in the
declared living room. Container poses/scales are not explicitly saved; source
construction and original inventory dimensions supply only private CPU screening
evidence. Native measurements remain authoritative.

No supplied category batch directory or `sample_evidence.json` existed at initial
inspection. The source-frame map provides a conservative initial footprint
screen, using a 0.45 m disk and the original room instance ID. The selected
robot-only starts are 1.30 m in front of the apparatus; some nearer-middle
positions fail the original furniture map screen. That map includes clutter
excluded by the active source builder, so rejection is conservative. It does not
include source-generated pool planes. A measured native qualifier, not this
screen, decides whether an episode may run.

## Two targeted revisions

1. Version 1.1.0 retains the initial operation rules and improves sampling to
   include the narrowest, largest-footprint, shallowest and tallest source-sized
   containers, both recipes, and a non-success record if present. The widest
   tray also supplies the shallowest example, so it covers two geometry strata.
2. Version 1.2.0 records the discovered source-directory/execution-loader
   conflict in every affected candidate. All 136 source paths use
   `unknown_scene/unknown_room`, while their original JSON resolves to
   `Rs_int/living_room_0`. The active loader and existing diagnostic capture
   loader use the JSON identity; the generic ready task/scene loaders require
   directory/JSON equality. The first validation call to `load_task` also
   correctly rejected pending tasks. Validation now checks the existing
   diagnostic loader and the preserved pending execution gate separately.
   Source identities, JSON and shared code were not changed. This execution
   blocker is retained explicitly rather than silently declaring loadability.

Both revisions are retained. No further rule iteration was made. Earlier
manifests remain in the category batch's `batches` directory.

## Final implementation and success predicate

Converter: [physical_structure_liquid_volume.py](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/physical_structure_liquid_volume.py),
`CategoryConverter`, version **1.2.0**. `inspect`, `convert` and `convert_all`
use `ConverterBase`, `original_scene` and `physical_task`. Original source hash,
converter implementation, seed, settings and CPU screening-resource hashes
determine candidate revisions. There are no instance overrides.

Two real navigation recipes reuse the publicly named containers:

| Recipe | Ordered physical stops | Sources |
| --- | --- | ---: |
| `dock_at_both_containers` | Left, right | 72 |
| `dock_at_both_and_return` | Left, right, left | 64 |

Recipe choice depends only on source identity and seed, never the volume answer
or saved particle counts. The adapted variant declares `add_existing_r1` before
an evidenced `robot_pose`. The original variant remains empty. Container
fixed-base, kinematic, collision and scale settings, water initialization,
reference camera and original builders remain intact. No mechanisms are omitted
by the chosen overlay, so no category initialization hook is needed. No object
is selected for manipulation; no `object_physics` patch or fixed-base release is
introduced for these navigation recipes.

The owned `category_operation` qualifier and evaluator read only native state.
All actions remain the existing guarded base controls. The qualifier requires
original fixed, collidable bindings, a distinct left/right apparatus, a stopped
R1, chassis footprint in the original room, actual native floor support and no
non-floor robot contact. Only contacts against the original floor binding may
count as support. In particular, `plane_N/groundPlane` is never treated as a
floor merely because its path contains `groundPlane`.

At each requested stop the measured chassis must align within 0.08 m of the
container AABB centre in world X, stand 0.70–0.95 m toward the original camera
side in world Y, face the container within 12 degrees, and stay stopped for
0.5 s with at least ten distinct native control samples. Stopping limits are
0.02 m/s linear speed and 0.1 rad/s angular speed. Consecutive visit positions
must differ by at least 0.20 m; total measured travel must exceed 0.25 m. The
last requested station must still satisfy the docking predicate at reporting.

The evaluator latches failure for unsafe support/contact, a time reset or gap
above 0.1 s, movement inconsistent with the existing controller bound, source
apparatus displacement above 0.005 m, source apparatus rotation above 0.02 rad,
or changed fixed-base/collision flags. Repeated timestamps do not accumulate
holds. Original answers, capacities, operation flags and scorer state never
enter these predicates. These thresholds and a 160-decision budget are explicit
new navigation protocol proposals, not inherited experimental parameters.

## Complete batch and failure distribution

Final [BatchManifest](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/batch_manifest.json):
136 sources, 136 unique candidates, **136 generated**, zero static conversion
failures, zero omitted sources, zero claimed native qualifications. All 136 are
pending L2 Review candidates and all 136 retain these warnings:

- `assumption_review_required`
- `native_navigation_unverified`
- `synthetic_pool_robot_collision_unverified`
- `liquid_transfer_unverified`
- `robot_initialization_unqualified`
- `source_directory_scene_mismatch`

The static failure distribution is empty. Missing original asset/map evidence
is separately checked to return an explicit
`not_convertible_under_L1_L2:<reason>` source-only diagnostic with an original
scene and no task. Malformed source contracts are rejected by `inspect` and
retained by shared batching. Any later initialization, screenshot, qualification
or execution failure must remain in the supervising campaign's Review set.
Generated counts do not mean executable or physically successful counts.

Final CPU generation took 36.33 s. Repeating all 136 conversions reused the same
records and IDs in 0.54 s. These are CPU conversion costs on this server, not
robot throughput or task latency. No quality score, native execution cost or
physical success rate is available.

## Validation and stratified visual evidence

[CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/cpu_validation.json)
checks all 136 original source hashes and verbatim private QA, scene/task schema,
matching overlays, empty original variants, add-R1/pose patch ordering, pending
execution rejection, diagnostic capture identity, complete structured
assumptions and public/private boundaries. Changing private QA labels and
capacity measurements leaves the public operation, private physical predicate
and scene unchanged. No public task contains models, native coordinates, source
capacities or the original answer.

[Adapter CPU contract checks](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/native_adapter_cpu_contract.json)
exercise ordered-stop timing, no-op rejection, duplicate timestamps, inadequate
holds, pose jumps, apparatus motion, visual-only rejection and non-scene pool
contacts. The synthetic three-stop trace validates predicate logic only. It is
not a native experiment, candidate qualification or evidence of physical
success. Static inspection also finds no native mutation/action calls in the
owned module.

Current `sample_evidence.json` was absent at both initial inspection and final
CPU evidence inspection. No native screenshot was available. The five images
in [CPU sample inspection](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/cpu_sample_inspection.json)
were visually inspected via this
[stratified overview](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/cpu_evidence/stratified_overview.png).
They are explicitly labelled source-map/inventory diagnostics, not native
captures, historical source photos or Agent sensors.

| Candidate | Source | Pair and inspection stratum |
| --- | --- | --- |
| `category_15c6b17d4b238b7704c9952f` | `run_273.json` | Wineglass / paper cup; narrowest container |
| `category_0de3c49a975ce8a077715f59` | `run_222.json` | Tray / jug; largest footprint and shallowest container |
| `category_15b69353a58dd51d93b4d18c` | `run_258.json` | Carafe / vase; tallest container |
| `category_58f58f803c05633cf86be575` | `run_126.json` | Water glass / bowl; additional round-trip example |
| `category_9253f26f6614b325833b87b1` | `run_127.json` | Bowl / mug; two-stop recipe coverage |

The diagrams show unchanged left/right separation and robot-only starts on the
original front side. They also make the unresolved issue visible: starts and
docking bands are outside the projected source pool. The original map shows
furniture near the left docking area, but much of that clutter is source-
excluded; map pixels cannot establish a current collision. No collision filter,
plane deletion, container relocation or safety bypass was retained.

## Historical evidence, limitations and publication decision

The earlier assessment reports one fill callback with `success=True` but zero
in-container particles and several R1 posture/scaffold failures. Current source
checks corroborate the scripted particle exile, robot-free scene and fixed
containers. All 136 historical source JSON pairs have positive recorded
capacities; those generator measurements neither reproduce nor refute the
reported later runtime failure. They are preserved privately. The actual
native zero-fill and controller failures have not been rerun by this supervisor.

The advice prefers retrieval or transfer. The authoritative operation policy
also permits navigation. The retained source fixed-base initialization does not
provide a demonstrated safe grasp/transfer path, and releasing fixed bases
would interact with source initialization that checks elevated bottom heights.
Rather than equate upstream fill/pour callbacks with robot operations, this
converter supplies separately measured navigation recipes. No new asset or
capacity-sensitive geometry change is justified by the available evidence.

Remaining limitations are material: synthetic infinite-plane compatibility,
native R1 startup/posture, actual floor contact, safe chassis routes, approach
visibility and execution cost are unverified. The navigation recipes may all
fail native qualification. They do not establish liquid capacities, genuine
filling, grasping or pouring. Every retained uncertainty has
`doubt/chosen_assumption/alternatives/scope/risk` in candidate assumptions.

The shared API owner must reconcile immutable-source identity resolution before
ordinary approved-task execution. After that, the supervising capture/Review
pipeline should inspect original and adapted views, preserve all native
failures and measure robot routes through guarded controls. If apparatus
contacts block every route, retain source-only diagnostics and investigate a
measured, separately reviewed L2 scaffold repair. Do not infer that new assets
are necessary, shrink containers, or use source particle teleportation as a
transfer test.

Decision: **publish the complete pending/diagnostic Review set**, retaining all
warnings and blockers. The supervisor result is
[supervisor_result.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/supervisor_result.json).
The supervising campaign owns queue submission and final Review records; this
supervisor has not submitted GPU work, published approvals, modified shared
code/source data/other categories/root research log, committed or spawned agents.


## Supplementary assessment comparison — 2026-10-15

Review fingerprint: `b8900214be9c1dc1f643d235d155ed14d609e82f8b3df266e84a050206b4b7e0`.
This is a read-only implementation/evidence review, with this appendix and the
assigned result JSON as the only authored outputs. The review plan was to compare
source semantics and historical claims, audit every current and previous candidate,
check the independent evaluator and initial QA boundary, then inspect existing
native diagnostics. No conversion, native experiment, robot action or GPU work was
launched by this reviewer. Recommended implementation changes require Main's review.

### Scope and previous tasks

Read the implementation specification, supplementary assessment, current converter,
active liquid builder, relevant original `batch_pour.py` generation routines, shared
conversion/loader/qualification/capture code, complete batch, CPU validation,
synthetic adapter contract, sample inspection and `sample_evidence.json`.
Both assignment `existing_task_files` lists are empty; there are no supplied legacy
task paths to omit. All earlier candidates were nevertheless included as read-only
previous tasks:

| Manifest under the category batch root | Version | Records |
| --- | --- | ---: |
| `batches/initial_1.0.0.json` | 1.0.0 | 136 |
| `batches/geometry_stratified_1.1.0.json` | 1.1.0 | 136 |
| `batches/bd9a57a60ac8a9f3419fbe48.json` | 1.2.0 | 136 |
| `batches/1dd716261445a8bf23ddc38b.json` | 1.2.0 | 136 |

The two 1.2.0 manifests reuse the same candidate set. The candidate directory
contains 408 unique revisions: 136 per version. Read all 408 records, task YAMLs,
scene YAMLs and private QA files. Every source hash matches the current source;
every private answer/options/metadata value matches verbatim; every task/scene is
pending; every original overlay is empty; every adapted overlay contains only
`add_existing_r1` followed by `robot_pose`; and each task overlay equals its scene
overlay. For each source, task and scene definitions are identical across all three
versions. The sampling and warning revisions are not new operation diversity or
independent physical trials. Earlier converter implementations were not recovered
as separate executable versions, so identical YAML does not prove identical earlier
runtime code. Only the 136 current IDs have category Review records in this queue;
older artifacts remain previous tasks, not missing current sources.

The current manifest covers all 136 corpus sources, with 136 generated proposals,
72 two-stop and 64 return recipes, and no CPU conversion failures. This reproduces
the coverage claim by read-only inspection; conversion/idempotency timings and
synthetic adapter outcomes remain results of the earlier CPU validation, not tests
rerun by this reviewer. The initial system Python lacked PyYAML; the read-only YAML
audit completed in the existing behavior environment with CUDA visibility empty.
The converter, active builder, loader, task adapter and capture script match their
available frozen `working_42b0c7081dfa96352ca6f9bcff3beeb5` copies byte for byte.

### Historical claims compared with current evidence

The [supplementary assessment](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md)
uses a different pinned ESI revision and earlier A40 diagnostics. Its claims about
robot-free construction, fixed-base containers, non-scene pool planes and scripted
particle exile are corroborated by the current builder. A fill callback still
returns `success=True` after `fill_single` without requiring a positive particle
count; `pour_container_into` changes selected particle Z coordinates to -100 m.
Neither operation flag proves robot-driven filling or a wrist-driven stream.

The reported zero-fill count, arm drift and failed startup/filter retries remain
historical observations. The linked historical standing `probe.json` is unavailable
at its `/scratch` path here; no historical raw trace or figure was inspected and no
such experiment was rerun. All 272 saved source container measurements are positive
(range 23–5,858 particles); top-level answers agree with `gt_larger` in all 136
sources (79 left, 57 right). These saved generation results neither reproduce nor
refute the later zero-fill probe. The assessment supports investigating software,
initialization and particle measurement before changing assets; it does not prove
that new meshes, resizing, a support relocation or collision filtering is required.

There is also a source-phase distinction. The generator captures an empty,
near-floor pair before filling both containers simultaneously. Its files named
`01_after_fill_left` and `02_after_fill_right` show the same both-filled phase.
The active builder instead starts empty after raising fixed-base containers to a
common free height computed from their extents, and recomputes its side camera.
Keeping that active builder preserves its construction behavior; it does not prove
an exact reconstruction of the saved generator phase poses or camera. This matters
if initial QA is later evaluated against reconstructed observations.

### Operation value and independent terminal evidence

The proposed tasks require actual base navigation, ordered stops, orientation and
holding still. They are meaningful physical navigation operations in specification,
with no-op rejection and no answer-based success shortcut. Their native feasibility
has not been measured. The return recipe adds another visit and path cost, but both
recipes belong to one docking family; neither tests volume reasoning, retrieval,
placement, filling or transfer. They reuse source assets and left/right references
faithfully while substituting the liquid mechanism with navigation. This is allowed
by the current operation policy, but only partially serves the preferred transfer
or retrieval goal. Retain the honest docking designation. Several recipes are useful
when they add distinct, feasible operations; adding recipe names or return visits
alone does not establish several liquid-operation skills. A future retrieval/transfer
recipe should follow measured workspace qualification and supported initialization,
retain capacity-sensitive geometry, and score native grasp/release and particle
transfer/retention/loss separately. It is not a prerequisite for present Review.

The current evaluator is independent of original QA, saved capacities and upstream
operation flags. It reads poses, AABBs, velocities, contacts and native time; requires
ordered 0.5 s holds with distinct samples, visit separation and accumulated motion;
checks the last docking state; and latches apparatus motion, unsafe contacts,
clock gaps/resets and excessive translation. The task adapter attaches its sample
hook to guarded control steps. Rejecting non-scene `plane_N/groundPlane` contacts
rather than whitelisting the name is appropriate. CPU fixture passes establish
logic coverage only; there is no recorded native qualification or route trace.

One safety predicate needs Main's review before operational qualification:
`measure_native_state` accepts floor contact from any robot body, and `safe` has no
explicit chassis up-axis/roll/pitch or locomotion-support-link condition. A stopped
fallen robot or arm/head floor contact is therefore not explicitly rejected by
this predicate if its other tests pass. This is a logical gap, not an observed
native false-positive success. Require measured upright chassis/posture and valid
base/wheel support, while retaining all non-floor contact rejection. The nine
sampled footprint points are also a room-membership screen, not full swept-volume
or route clearance. Guards and future native evidence must remain authoritative.

### Initial QA boundary and execution blockers

The initial question/answer/source remain intact and private QA archives match all
408 revisions. Public task fields contain operation instructions, source-public
container labels and tolerances; model IDs, native poses/maps, capacities, answers
and scorer state remain outside the public payload. Native docking success never
consumes the QA answer. This boundary is sound for an auxiliary archived QA.
However, these category tasks do not declare an initial-QA interaction/scoring
stage, and neither the capture receipts nor docking evaluator supplies an initial
QA score. Preservation must not be described as measured initial-QA performance.
If a combined QA/operation protocol is added, privately record the pre-action phase
and score that answer independently; human source/global images remain review-only.

Two execution integration blockers are distinct from diagnostic capture:

1. The documented source identity mismatch remains: all paths use
   `unknown_scene/unknown_room`, while JSON resolves to `Rs_int/living_room_0`.
   The generic ready loader rejects that mismatch; the diagnostic loader uses
   immutable manifest identity. Do not rename or edit sources to resolve it.
2. In [ESISession initialization](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/sim/esi_omnigibson.py:194),
   `_configure_robot_observation` runs before `scene_adapter.apply_to_env_config`.
   The source supplies `robots=[]`, and normal `run_esi_robot.py` uses the default
   `require_robot=True`. Thus observation configuration raises before the declared
   `add_existing_r1` patch can run, even after identity handling is reconciled.
   A CPU-only extraction of the exact current method reproduced
   `RuntimeError: Observation setup requires an existing R1 in ESI's robot config.`
   With `require_robot=False` the method returns, matching the diagnostic capture
   branch. This check ran no simulator. Main should integrate the declared robot
   addition before validating/configuring robot observations, retaining strict
   final robot validation, source physics and existing safety gates.

### Current CaptureQueue diagnostics and visual inspection

At **2026-10-14 21:04:07 UTC / 2026-10-15 05:04:07 Singapore**, all 136 current IDs
had published category records. Their jobs were 15 `REVIEW_PENDING`, 120 `QUEUED`
and one `RUNNING`; no failed state was present in that read. The running job was
`capture_215e68f0912ae3921ba44df635f118b0` for
`category_0ea81a8f7dc957f8718afaee`, with recorded stage `cold_scene_loading`.
This is a changing, non-transactional queue observation, not a final batch outcome.
Existing workloads were left running.

All 15 completed diagnostic receipts have nine image files present (135 files),
`complete=true`, `diagnostic_pending=true`, `native_qualification_run=false`,
`eligible_final_layout=false`, zero model calls and zero robot actions. Their
recorded total capture times range from 132.45 to 176.60 s (median 150.44 s),
excluding queue wait; these are rendering costs, not robot latency or throughput.
Receipts report CPU physics with the assigned GPU renderer. They do not reproduce
the historical fluid or controller protocol.

All five stratified samples have completed diagnostic captures:

| Candidate suffix | Capture job | Total seconds |
| --- | --- | ---: |
| `15c6b17d4b238b7704c9952f` | `capture_18cd757e9dff8d5f181fac42f485f7a3` | 176.45 |
| `0de3c49a975ce8a077715f59` | `capture_357a368c4ae90c06f81978c5fed7d9c0` | 176.60 |
| `15b69353a58dd51d93b4d18c` | `capture_e56a6b4130fca55da69d20f603ab4b58` | 144.46 |
| `58f58f803c05633cf86be575` | `capture_528266ee144b2979e3a14506e04964b3` | 150.44 |
| `9253f26f6614b325833b87b1` | `capture_02e4e77a2be9cdc9c59786fd37cf8ee8` | 132.45 |

The [sample evidence file](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/sample_evidence.json)
exists but still embeds five `QUEUED`/zero-attempt entries. Earlier prose saying
it was absent, and its cached states, are historical observations superseded by
job `status.json` and `attempts/01/preview_complete.json`; neither was rewritten.

Visually inspected all five current head images plus six other saved views.
The head images are dominated by R1 body/arms and nearby floor/walls, without a
clear view of both containers. Selected global/source views also show substantial
robot occlusion; a vessel is visible near the chassis in the carafe/vase source
view. This raises a concrete framing/visibility and initialization concern, but
RGB alone cannot establish contact, penetration, actual base pose or its cause.
The receipts do not include native support/contact/route measurements. Only head
and wrist are the mounted views saved by these receipts; the complete eight-camera
Agent coverage was not inspected or certified. `initial_visibility=none` and the
category qualifier provide no replacement visibility measurement. Main should
review measured spawn/posture/contact readbacks and legitimate mounted-camera
coverage before relying on the public sensor-based docking instructions. Do not
move assets or loosen contacts based on these pictures alone.

The evidence is located under
[CaptureQueue jobs](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs).
For example, inspect the wineglass/paper-cup
[receipt](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_18cd757e9dff8d5f181fac42f485f7a3/attempts/01/preview_complete.json),
[head image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_18cd757e9dff8d5f181fac42f485f7a3/attempts/01/head.png)
and [overhead image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_18cd757e9dff8d5f181fac42f485f7a3/attempts/01/global_overhead.png).
The linked head/global/source images are private review evidence, not additional
Agent observations.

Assessment retained: preserve source geometry and QA; retain the complete pending
and diagnostic Review set with explicit semantic and feasibility warnings. The
recommended startup-order, posture/support and visibility corrections belong to
Main's implementation review. This comparison introduces no Review/capture gate,
withholds no candidate and requests no pause or restart. It claims neither robot
success nor human approval. Implementation, generated scenes, source data, answers,
approvals, other category reports and root `research_log.md` were not edited.

## Assessment comparison for the integrated runtime — 2026-10-15

Review fingerprint: `0682c5459c4218bd962bc698ba1d51c88e7707fd4635e3d66a344f22752986bc`.
This appendix supersedes time-sensitive claims in the earlier comparison only
where current evidence differs. The review plan was to read the specification,
assessment, builders and operation hooks; audit the complete batch and every
previous candidate; compare native capture receipts and selected images; and
report semantic and qualification gaps. This reviewer ran read-only CPU audits,
not conversions, native experiments, robot actions or GPU workloads. Only this
appendix and the assigned result JSON are authored. Corrections remain Main's
implementation-review responsibility and introduce no Review or capture gate.

### Complete scope and revision accounting

Read the current [implementation specification](/local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md),
[supplementary assessment](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md),
[converter](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/physical_structure_liquid_volume.py),
[active builder](/local_home2/zhouxinyang/models/Show-Harness-Explore/ESI-Bench/src/active_explore/tasks/physical_structure/liquid_volume.py),
relevant original `batch_pour.py` routines, shared loader/initialization/evaluation
and capture code, complete batch, CPU validation/adapter/sample reports and
`sample_evidence.json`. Both supplied `existing_task_files` lists are empty.
All earlier candidates were nevertheless read as previous tasks without edits.

The [batch root](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad)
contains these six historical/current manifests:

| Manifest in `batches/` | Version | Records | Relationship |
| --- | --- | ---: | --- |
| `initial_1.0.0.json` | 1.0.0 | 136 | Previous candidate set |
| `geometry_stratified_1.1.0.json` | 1.1.0 | 136 | Previous candidate set |
| `bd9a57a60ac8a9f3419fbe48.json` | 1.2.0 | 136 | Previous candidate set |
| `1dd716261445a8bf23ddc38b.json` | 1.2.0 | 136 | Same IDs as preceding row |
| `847e6e11d3f54ebd00d9fb38.json` | 1.2.0 | 136 | Same previous 1.2.0 IDs |
| `ad310fbea67874f9b6700ffb.json` | 1.2.0 | 136 | Current runtime-review revision |

The current `batch_manifest.json` uses the final row and
`runtime_review_revision=post_assessment_20261015_v1`. There are **544 unique
candidate revisions: 136 current and 408 previous**, four per source. Read all
544 records, task YAMLs, scene YAMLs and private QA archives. All source hashes
match; archived answer/options/metadata values match verbatim; all definitions
remain pending; original overlays are empty; adapted overlays contain only
`add_existing_r1` then `robot_pose`; and task/scene overlays match. Task and scene
definitions are identical across the four revisions of each source. These are
revision histories, not 544 source tasks or independent physical trials.

The current manifest exactly covers the 136 corpus files: 136 generated proposals,
72 two-stop routes, 64 return routes, no static conversion failures or omissions.
The eight inspected implementation files, including the converter, builder,
loader, session, patches, evaluation envelope and capture adapters/script, match
the supplied frozen `working_baf92c88aaa2427b4c5209cb43b95561` copies. Earlier
executable converter versions were not reconstructed. The stored CPU validation
and synthetic-contract passes remain earlier authors' results; conversion,
idempotency and synthetic route tests were not rerun in this review.

### Claims retained, narrowed and corrected

The historical assessment's robot-free builder, fixed-base containers, non-scene
pool planes and scripted particle exile are corroborated by current source.
`fill_physical` still returns `success=True` after a callback without requiring a
positive contained-particle count. Pouring still relocates selected particles to
Z=-100 m; it is not wrist-driven transfer. Historical zero-fill, arm-drift and
startup/filter failures were not reproduced here. All five linked historical
Liquid Volume `probe.json` paths under `/scratch` are unavailable locally.
Current source JSON retains 272 positive saved capacity measurements, 23–5,858
particles; all 136 answers agree with metadata `gt_larger` (79 left, 57 right).
Saved generation measurements do not reproduce or refute a later failed fill.
Neither evidence establishes that resizing, new assets or filtering is required.

Preservation is relative to the active builder. It starts empty with fixed cups
raised to a common free height and a recomputed side camera. The generator's
initial empty phase is near the floor, then both cups are filled simultaneously;
its two filenames suggesting separate left/right fills refer to the same filled
phase. Reusing the active builder does not establish exact saved generator-phase
poses, observations or camera reconstruction.

**Two earlier integration blockers are resolved in the inspected current and
frozen code.** `source_scene_room` resolves literal `unknown_scene/unknown_room`
components from immutable JSON while retaining rejection of other mismatches;
all 136 sources resolve to `Rs_int/living_room_0` in a read-only CPU check.
`ESISession.reset` now applies the scene adapter before
`_configure_robot_observation`, so declared R1 addition precedes sensor setup.
The converter's inherited limitation text saying the generic loader will reject
all 136 sources is stale. Main should update that diagnostic wording in a future
revision while preserving history. These fixes do not establish native posture,
route feasibility or permission to execute pending tasks; the pending gate remains.

### Operation value, terminal predicates and initial QA

Ordered base navigation, facing, stopping and returning are meaningful actual
operations in the task definition. No-op episodes cannot satisfy the required
visits and motion. Both recipes preserve the apparatus and left/right references,
but replace liquid experimentation with one docking family. Returning adds route
length and sequencing, not a separate liquid skill. Navigation is permitted by
the operation policy, although it only partially meets the preferred retrieval
or transfer goal. Several source-faithful recipes would be useful if they add
distinct feasible interactions; future supported retrieval/placement and real
transfer need their own measured grasp/release, retention, transfer and loss
predicates. Such development is not a prerequisite for pending/failed Review.

The current terminal evaluator is independent of QA answers, saved capacities
and upstream success flags. It measures native time, poses, velocities, contacts
and apparatus AABBs; requires ordered 0.5 s holds with distinct samples, movement
between visits, total travel and a valid final dock; and latches unsafe contacts,
apparatus changes, clock gaps/resets and excessive translation. Guarded controller
steps feed its sampling hook, and the shared envelope binds results privately to
the task/source and current observation boundary. The 8 cm, 70–95 cm, 12 degree,
0.5 s and 160-decision settings are proposed navigation parameters, not calibrated
liquid-task performance. Synthetic fixtures cannot qualify them physically.

The earlier posture/support finding remains: `safe` accepts floor contact from
any robot body, without an upright-chassis or allowed locomotion-support-link
condition. A stopped fallen robot or an arm/head floor contact is not explicitly
excluded if other checks pass. Main should review upright/posture and valid
base/wheel support checks in both qualification and continuous evaluation.
This is a logical gap, not an observed false-positive success. Preserve rejection
of non-floor contacts, including `plane_N/groundPlane`; nine footprint samples
establish room membership only, not full-body or swept-route clearance.

The information boundary is appropriate for archived auxiliary QA: original
answers/options/metadata remain private, the question remains in the immutable
source and record, and public instructions expose container labels and protocol
tolerances rather than native poses, model IDs, maps, capacities or scorer state.
There is **no declared initial-QA response/scoring stage** in these tasks and no
measured initial-QA score. If added, that stage must record legitimate pre-action
mounted observations and privately score the response independently of operation
success. Human global/source images remain review-only. `initial_visibility=none`
does not establish that the containers can be found using Agent sensors.

### Current native diagnostics, costs and remaining uncertainty

At **2026-10-15 00:28:00–00:28:01 UTC / 08:28 Singapore**, the category had 272
published records: all 136 current IDs and 136 previous 1.2.0 IDs. Other previous
revisions remain available in their artifact directories. Queue state was:

| Candidate set | REVIEW_PENDING | QUEUED | RUNNING | Failed states observed |
| --- | ---: | ---: | ---: | ---: |
| Current | 7 | 129 | 0 | 0 |
| Previous published 1.2.0 | 30 | 106 | 0 | 0 |

This is a changing, non-transactional snapshot; queued work is unfinished, not
failed. No job was paused, resubmitted or restarted. All 37 completed receipts
have nine image files present (333 files), `complete=true`,
`diagnostic_pending=true`, `native_qualification_run=false`,
`eligible_final_layout=false`, zero model calls and zero robot actions. Current
capture durations are 134.45–289.07 s, median 198.56 s; previous completed captures
are 132.45–222.60 s, median 161.48 s. These are existing rendering costs excluding
queue wait, with CPU physics and GPU rendering, not navigation latency, model
throughput or a controlled old/new performance comparison.

All five current samples have completed diagnostic captures. Three source choices
differ from the older CPU inspection, so its sample list cannot be reused as the
current native sample list:

| Current candidate suffix | Source pair | Capture job | Seconds |
| --- | --- | --- | ---: |
| `476a1c93ca042c9f3bbd7e81` | Wineglass / paper cup, run_273 | `capture_8f94e38ec62fd12d930abae64dafa14d` | 134.45 |
| `a74be0a743c6a015e5dc5fa7` | Tray / jug, run_222 | `capture_15456d0611bde4ad28ac281a662d748e` | 142.81 |
| `3876947874cb315a572e4283` | Carafe / bowl, run_196 | `capture_c34ea075e623cae5bfdce6db7af96f35` | 289.07 |
| `59108d7d436e8490615d109f` | Beer glass / bowl, run_81 | `capture_e1d7757b24cd9a8e5e51e0827a1284ee` | 153.11 |
| `012c15dd2e92bca224be983a` | Bowl / vase, run_291 | `capture_e6be1c02837cf22fe0184fc67be73ce9` | 212.57 |

The existing [sample evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f04d0822601c40619fad/sample_evidence.json)
has four cached QUEUED entries and one REVIEW_PENDING entry; live job status and
receipts supersede those cached states. Parent-job images labelled historical
are previous local candidate captures, distinct from unavailable historical
`/scratch` experiments. No cached evidence or historical record was rewritten.

Visually inspected the five current head images, six additional current views
and the labelled CPU overview. Head images show mainly floor/walls and robot
body/arms without a clear container pair. Some human views show vessels beside
the robot, but substantial body occlusion remains; the tray/jug diagonal image
shows essentially floor. The current beer-glass/bowl source view shows both
elevated vessels, consistent with the active-builder phase. These observations
support reviewing framing and mounted-camera coverage; RGB cannot determine
contact, penetration, actual spawn pose, drift or its cause. The inspected
[wineglass receipt](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_8f94e38ec62fd12d930abae64dafa14d/attempts/01/preview_complete.json)
explicitly records no qualification. Its
[head view](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_8f94e38ec62fd12d930abae64dafa14d/attempts/01/head.png)
and other images do not supply native support/contact/route readbacks. Full
eight-camera Agent coverage was not evaluated.

Main's remaining priorities are measured upright/support qualification, private
spawn/postprocess/contact and mounted-visibility evidence, and accurate current
diagnostic wording. Any later liquid recipe requires independently measured
robot-driven liquid behavior; do not use source particle relocation as success.
Retain all pending, failed and uncertain candidates in Review and keep captures
progressing. This assessment claims neither robot success nor human approval and
does not request any implementation, scene, source, answer, approval, workload or
other-category changes by this reviewer.

## 2026-10-15 CPU interaction-group and grasp-height review (rendering paused)

This iteration addresses the hypothesis that source-faithful tasks can still be physically unusable when the required interaction is too low, too high, too wide or unsupported. The review read the final v2 specification, supplementary report, current category sources/converters, all current task definitions and existing capture receipts. All simulation/rendering remains paused. This reviewer ran CPU code and reused historical native artifacts; no new native experiment, queue submission, robot action or approval was performed.

Complete coverage: **136 original sources**, status distribution `{"generated": 136}`; **136 existing task definitions** received an explicit interaction profile. Current mode counts are `{"navigation": 136}` and relocation decisions are `{"preserve": 136}`. 0 original non-generated records remain in the audit, with their failure stage/reason and source identity. No canonical candidate or source was overwritten.

Corrected policy: Navigation/preserve: the current dock_at_both_and_return recipe does not require cup grasps or pouring. Retain capacity geometry, synthetic apparatus and all fixed anchors. Exact per-source grouping/binding keys, source geometry proxies, unchanged failure records and receipt paths are in the [independent class audit](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/category_grasp_audit/categories/f04d0822601c40619fad/audit.json). The original fill/pour mechanism is not established by navigation. Do not move room or fluid-scaffold anchors; native stable docking and apparatus preservation remain pending.

Source-height risks: 0 task-bearing instances have at least one required interaction root/AABB midpoint below 0.35 m world Z; 0 have one above 1.10 m; 0 lack some required geometry. These are **source world-height proxies**, not height relative to a measured floor, actual grasp-contact points or native support/IK checks. The native initializer must use measured floor-relative geometry and the unchanged Rigid screening window (0.35–1.10 m, stance distance at most 0.85 m). Navigation has no required grasps; articulation body heights do not substitute for handle-link measurements.

Width evidence: No source horizontal AABB width is available for the required manipulation bindings. 0 required manipulation bindings lack that width evidence. A global AABB extent is not an oriented grasp width, and no native aperture/IK or executed grasp is inferred.

Existing native evidence: 28 matching current-candidate receipts and 28 instances with existing indexed images. They record 0 executed qualification gates and 0 verified grasps. Their frozen implementations predate this new profile; they do not validate the proposed repaired layout. Visual inspection covered 6 existing frames from 3 instances: The first inspected pair shows floor/nearby room geometry without a clearly visible vessel apparatus. The available images do not establish fluid measurement, pouring or a completed docking route.

- `category_187b2058c13fc3b6c52a866b` / global_diagonal: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_175cd74a6a4db07b5c7eb5961c6d311d/attempts/01/global_diagonal.png)
- `category_187b2058c13fc3b6c52a866b` / head: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_175cd74a6a4db07b5c7eb5961c6d311d/attempts/01/head.png)
- `category_223213f519a3ca43cc79ee5d` / global_diagonal: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_1edd57ac3b51916dd43b9a0cd990dc2f/attempts/01/global_diagonal.png)
- `category_223213f519a3ca43cc79ee5d` / head: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_1edd57ac3b51916dd43b9a0cd990dc2f/attempts/01/head.png)
- `category_fc18f4eec0edae5fa1f005a0` / global_diagonal: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_07586a29648be28e9d9f0a015fe175e5/attempts/01/global_diagonal.png)
- `category_fc18f4eec0edae5fa1f005a0` / head: [existing native image](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_07586a29648be28e9d9f0a015fe175e5/attempts/01/head.png)

The retained profile warning distribution is `{}`. Unresolved support, native state, mounted-camera coverage, actual gripper aperture, collision-free IK path and execution remain pending. A candidate or screenshot is not a successful physical episode. The source/failure audit and all old evidence remain available for Review; no rendering has been resumed.


## 2026-10-15 Approved accessible-support generator audit

Plan: apply the user-approved access repair at generator-rule level after reading the final v2 specification, current converter/source builder and reports. The historical assessment is supplementary and nonbinding. Audit every source and the latest human-rendered batch referenced by category_revisions.json; preserve source QA, old failures and approved work. No source JSON, published task, queue, screenshot or Native process was modified.

Coverage: **136 original sources**, **136 existing task definitions read**. Retain docking goal and left/right pair geometry. Common Z support requires matching pool planes, particle state and source cached free/dip references; no liquid-transfer success claim. [Whole-class evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/audit.json) and [eleven-class summary](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/audit_summary.json) preserve every inspected source identity and failure.

Retained converter version `1.3.0_accessible`. The isolated [CPU batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/batch_manifest.json) contains 136 records, 136 pending task definitions and status counts `{"generated": 136}`; conversion took 41.227 s with seed 0, configuration `{}` and numerical-library threads bounded to two. 0 records defer an inconclusive CPU stance screen to Native measurement without inventing a robot pose. Every generated source hash and complete private QA packet matches the original; every prior task file remains byte-identical.

Prepared representative: `category_422f8c93b655f62ed9fd866b` for `Physical Structure/Liquid Volume/unknown_scene/unknown_room/run_1.json`. Same source as the authoritative prior sample. [Task](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/candidates/category_422f8c93b655f62ed9fd866b/task.yaml) · [Scene](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/candidates/category_422f8c93b655f62ed9fd866b/scene.yaml). The representative is prepared only; this agent did not publish or enqueue it.

The new support-required docking branch uses fresh public task-support readback and floor-relative height checks both at qualification and during the route. Both legacy docking recipes and their original parameter contract remain callable. The source creates four non-scene USD roots `/World/plane_0` through `/World/plane_3`; coherent adaptation also requires current water particles and `session.task_state` free/dip positions and box Z origin. The shared anchor implementation must prove and restore these changes. Fixed containers retain their physics; no pool collider is removed, and no fill, capacity measurement or real liquid transfer is claimed.

Validation: `tests/test_accessible_generators_a.py` completed **20 passed in 1.53 s**; the focused annotation, tabletop-relation and goal-only compatibility suites completed **78 passed in 1.76 s** using behavior Python. These CPU fixtures check private-QA boundaries, generated schemas, one-time initialization, adaptation policies and native-shaped support/cover predicates. Native surface/contact acceptance, collision-free floor access, cloth/scaffold integrity, mounted views, IK, grasp/release and physical execution remain unverified for the new runtime. Shared placement and subsequent one-representative-per-changed-class rendering are owned by Main.


## 2026-10-15 Separate support heights for unequal containers

Main review found that unchanged containers sharing the source free-root Z can have different body bottoms. A single pair-wide table therefore leaves the shorter container floating or without its own recorded support. The retained correction is converter `1.3.1_accessible`: request `vertical_only`, `placement_groups=[["left"],["right"]]` and `support_bindings=["left","right"]`. Both original roots still receive the exact same translation `[0,0,dz]`; original relative XYZ, sizes, orientations, docking goal/order, public wording and private evaluation remain unchanged. Separate measured surfaces may have different heights. The shared pool/scaffold anchor applies once after both groups, including its current particles and cached free/dip references.

The [regenerated Liquid batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/batch_manifest.json) retains all 136 sources with status counts `{"generated": 136}`; conversion took 28.489 s with seed 0, unchanged context configuration and numerical-library threads bounded to two. All source hashes and complete private QA packets match; all other Group A batches remain byte-identical. The prior local single-support batch is retained as historical evidence. Updated same-source representative `category_6005c81127e2cd47c1d009dd` is prepared for `Physical Structure/Liquid Volume/unknown_scene/unknown_room/run_1.json`; [task](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/candidates/category_6005c81127e2cd47c1d009dd/task.yaml) and [scene](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/candidates/category_6005c81127e2cd47c1d009dd/scene.yaml). The [correction receipt](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f04d0822601c40619fad/unequal_bottom_support_correction.json) records the previous representative and full checks.

Only Liquid-related CPU tests were run: `python -m pytest -q tests/test_accessible_generators_a.py -k liquid` completed **7 passed, 14 deselected in 0.31 s**. Coverage includes generated serialization, unequal container bottom heights, missing per-object support paths, lost current support and legacy docking compatibility. Native surface contact, coherent scaffold/particle movement, robot access and real docking or liquid transfer remain unverified. No Native/GPU task, queue mutation or publication was performed by this correction.

## 2026-10-15 Native representative outcome

Main's [publication journal](/home/zhouxinyang/esi_category_conversion/20261014_v2/review_updates/ed22ddbd33445d4a74078171/update_result.json) selected runtime representative `category_e99c0820ab32c01e07d79ab2` from `unknown_scene/unknown_room/run_1.json`. The [current metadata](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/native_results.json) and [job status](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_a46a0f677e99c1c3865f4aa4cce4b4ed/status.json) record `FAILED` on attempt one at `resident_capture`, with “Resident native process exited with code -6.” The status marks this fatal failure nonretryable. No completion receipt, accepted placement or image declaration is available.

The retained [Native log](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_a46a0f677e99c1c3865f4aa4cce4b4ed/attempts/01/native.log) reports `GPU artiPropagateVelocity fail to launch kernel!! 700`, followed by `CUDA error 700: cudaErrorIllegalAddress` and `Fatal Python error: Aborted`. The active Python stack passes through simulator `step` inside `_place_robot` during shared interaction initialization. This locates the observed fatal event; it does not establish which asset, physics configuration or preceding operation caused the illegal access. No images could be acquired after the fatal process exit, and no replacement image evidence is invented.

The individual left/right support groups, exact common-Z translation and coherent pool-plane/particle/cache anchor policy remain the generated request, not demonstrated Native success. Per-container support contact, scaffold/particle integrity, safe robot access, real docking and liquid transfer remain unverified. No autonomous grasp or pour was executed or established by this result. Current CPU coverage remains **136 source records and 136 generated pending tasks**. This update retains the full failure/log evidence and earlier history, reads declared metadata/log paths only and runs no additional experiment or image validation.


## Safe composition and review checks — 2026-10-16

The source-aware CPU composition policy retains 124 current generated records as 113 episode groups. Archived repeats and failures are excluded from automatic composition and retained separately. The implementation is [composition_manipulation.py](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/composition_manipulation.py). Singleton worlds and original physical predicates remain unchanged. Composite scoring credits completed original goals at their own validated checkpoints; native history, causal order and safety thresholds are not weakened. Human review checklists are separate from automatic success predicates. No new native experiment or render was run for this change.

Required review checks:

- Check the real robot spawn, upright wheel contact, obstacle clearance and a usable route to every required object.
- Inspect low and high supports, native support contact and unchanged asset scales; height screens do not establish IK or graspability.
- Check mounted head and wrist sensor coverage through robot observations; human overview images are not agent evidence.
- Retain original QA and reference-view evidence separately; obtain initial comparison answers before interventions change the evidence.
- Check unchanged fixed containers, original model-derived scales, common support translation and pool/scaffold alignment without robot contact.
- Verify left then right native docking stops; the return variant additionally requires a new left stop with continuous safe measured travel.
- Score the short goal at the right checkpoint and the return goal at the final left checkpoint; never AND their incompatible terminal docking positions.
- Docking completion does not certify capacity, filling, grasping or pouring; retain historical liquid QA independently.

Eleven verified apparatus pairs combine the short route and return route: run 102/87, 122/142, 124/2, 125/277, 141/214, 151/230, 179/185, 191/275, 259/45, 272/293 and 290/30. The policy resolves the actual native side_entry precedence and compares complete declared initialization, bindings, seeds, resource/bootstrap screens, scaffold contract and evaluator thresholds. Raw run indices and historical capacity/particle counts do not affect this docking bootstrap. A short-goal terminal stop is at the right container; a return-goal stop is at the left. Completion must be latched at separate checkpoints, not ANDed in the final state. New composite tasks require fresh capture/review and inherit no approval.

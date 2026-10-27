# Action Sequencing / Action Order Inference

## Feedback repair plan — 2026-10-15, revision 893f6514

Assignment `f8445a7f136def80b7e6`, converter `2.2.0`, feedback revision
`f8445a7f136def80b7e6_893f6514f78e1c5430b85b95`. This section supersedes
historical current-state claims below. The current agent read the implementation
specification, conversion/schema/initialization/operation APIs, source builder,
source JSON, generator fit predicate, supplied supplementary assessment and
current batch-root sample evidence before implementing this repair.

Hypothesis: translation-only support search misses an unchanged row that fits
under a common planar rotation; coupling robot spawn to support acceptance
leaves rejected diagnostic scenes at the source's off-map robot start. The
selected native receipt records rejected initialization before any proposal and
zero robot actions. Its overhead/source/head images were inspected by this
agent and show the floor row and an empty sky/arms head view. Those are prior
capture evidence, not a new simulator experiment. The historical tabletop
adapter aligns this same source row to an existing table axis, preserving scales
and the camera transform, but its raw evidence is unavailable locally and its
front IK stopped before grasp. These facts support testing a reversible rigid
rotation rule, not claiming reach or grasp success.

Plan: add an owned initialization-only hook that first finds a measured,
collision-free same-room R1 stance independently of support placement, then tries
identity and measured table-axis common row/camera rotations using existing
support/stance helpers. Restore rejected proposals and keep an accepted safe
robot stance even if every tabletop proposal fails. Preserve the empty original
variant, all original assets/scales/answers and the ordered native operation.
Report initialization failures explicitly; generated status never means native
success. Use at most two targeted rule iterations, run complete CPU conversion
and bounded meaningful interface checks, and inspect all available stratified
image evidence. Do not launch workers or change shared code or the root log.



## Feedback repair results — publish pending Review

**77/77 sources across 34 scene models have generated pending L2 candidates; zero
CPU conversion failures. Decision: publish every record for Review. Native startup,
reach, grasp and released stacking remain unverified for this implementation.**
The assignment version remains `2.2.0`; final converter SHA-256 is
`91a2e3e9fcf77f8ce4b08343c958d769fba9748de024b51f27505d01a7660096` and rule revision is `measured_rigid_row_rotation_independent_spawn_v2`.
Earlier candidate directories, feedback and manifests remain preserved. Only the
owned converter/report and batch-root artifacts were written; no capture jobs,
GPU workers, commits or shared/root-log edits were made by this agent.

Two targeted rule iterations were retained. First, add independent diagnostic
spawn and measured-axis rigid-row proposals. Second, retain the complete owned
initialization trace in the private native log, scope the uncertainty to the
actual selected feedback, and read native tensor AABB endpoints individually
before combining them. Both iterations generate identical scene/task definitions
for 77/77 sources; implementation hashes distinguish their frozen runtime code.

### Retained initialization and physical goal

The converter implements `inspect` and `convert`, inherits the shared deterministic
`ConverterBase.convert_all`, and continues using `original_scene` and
`physical_task`. Every original variant remains exactly `patches: []`. Adapted
variants explicitly enable each original candidate's movable/collidable physics
before construction and restore `visual_only=false` after source postprocessing.
The current source postprocessor is a no-op; these declarations also preserve the
required later-phase restoration contract. `configure_scene` validates the exact
original assets and existing R1 without changing its configuration. No new asset
or robot is added.

The owned initialization hook first searches up to 32 measured same-room floor
cells, facing the row. Acceptance requires actual native floor contacts, no
unexpected robot contacts and unchanged original/background movable poses after
settling. Rejected trials restore the full native baseline. A verified diagnostic
robot start survives subsequent support-search rejection; if no start passes,
the failure is explicit and no safe start is claimed.

It then tries identity and at most eight additional common planar row rotations,
derived from original same-room supports' measured axes and the native row axis.
Each proposal reuses the existing measured-surface support/stance initializer,
including its local surface, footprint, contact and geometry checks. Original
scales and relative geometry are retained. The source reference camera receives
the same rotation and translation, including orientation; mounted sensors are
unchanged. Extra checks verify the resulting common object transform, unchanged
scales and reference-camera transform. Every rejected proposal restores native
objects, background, robot and reference camera to its baseline. The final shared
interaction patch independently screens the already adapted state. Its receipt
alone does not describe the full original-to-adapted rotation; inspect the owned
hook's returned evidence and `[action-order-initialization]` private native log.

The private native qualification and ordered-stack evaluator are unchanged from
the starting converter. Public operations require grasping B, lifting its root
and AABB bottom at least 5 cm, releasing it stably on a chosen original base, then
grasping/lifting/releasing a chosen original top on B. Native `OnTop`, `Touching`,
matching contact bodies, external base support, low motion, released attachments
on every arm, an open working gripper and robot-clear objects determine success.
Each placement requires 0.5 s at the native controller clock, with speed limits
0.02 m/s and 0.1 rad/s. Upper members cannot use extra props. Observation reads
revoke invalid credit and never advance time. Initialization receives no grasp,
lift, sequence or terminal-success credit; all episode actions use existing
guarded robot controls.

### Current CPU evidence

All commands use `/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python`,
`PYTHONPATH=.`, and `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2`. Final checks used `CUDA_VISIBLE_DEVICES=''`.

| Check | Observed result |
| --- | --- |
| Complete source/result coverage | 77/77; generated 77, failed 0, all pending |
| Exact private QA; original builder assets/scales/poses | 77/77 each |
| Public/task/scene independence from private answer, QA, poses and construction flags | 77/77 |
| Construction validation leaves configuration unchanged | 77/77 |
| Idempotent records and modification times | Pass |
| Final full generation / cached repeat | 19.023 s / 0.236 s |
| Native-shaped operation fixtures | 24/24 pass; 3.438 s |
| New synthetic initialization fixtures | 4/4 pass |
| Disposable malformed-source diagnostic fixtures | 3/3 explicit failures |

The four initialization fixtures cover a rejected identity followed by a common
rotation/translation, all-support rejection retaining a safe robot start, a
camera-transform violation restoring the baseline, and no valid floor start.
They also test dirty-trial rollback, original scales/orientations and the
initialization-only guard. They use synthetic native interfaces and establish no
physical feasibility. An initial fixture harness re-imported Torch inside an
isolated module context; importing Torch before that context corrected the
harness without changing runtime logic. The malformed fixtures are missing
poses, an unsupported task and invalid JSON, separate from the 77 real sources.

The deterministic five-source sample prioritizes the feedback failure and source
direction ambiguity, and covers all four wrong-base/wrong-top size-mode pairs.
The source counts are 25 large/small, 18 small/large, 20 small/small and 14
large/large. Sample metadata and generated definitions were inspected:

| Candidate | Scene / room | Wrong base / wrong top |
| --- | --- | --- |
| `category_5d13d6f8f7fcf965e03ea6b8` | Beechwood_0_garden / dining_room_0 | too_small / too_large |
| `category_e9f0cc28fd6b0e10a4f67bf4` | Beechwood_1_int / bedroom_0 | too_large / too_large |
| `category_0520bb4cabef2e1b490cd4cc` | hall_conference_large / bathroom_0 | too_small / too_small |
| `category_2b7de24c7e6bf9fc372bbaba` | Beechwood_0_garden / bathroom_0 | too_large / too_small |
| `category_f8cf49334147ce9a2a2a5081` | hotel_gym_spa / locker_room_0 | too_large / too_small |

### Historical evidence, conflicts and remaining limitations

The current batch-root [sample evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json)
describes five older v2.1.0 sources. All five source strata were checked; its 18
historical referenced image paths are currently unavailable. No new image is
claimed for those strata or for the final converter hash.

Five images from the selected previous native capture were actually viewed by
this agent: overhead, source reference, front, head and wrist. They show the
separated floor row beside dining furniture; head/wrist frames contain arms/body
and sky without task objects. The [previous receipt](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/preview_complete.json)
records `accepted=false`, `relocated=false`, no placement attempts and zero robot
actions. Its rejection is `ValueError: No existing same-room support fits the unchanged group at a suitable height.`. Its 217.587 s
duration measures initialization/capture, not robot manipulation. The current
agent did not run that experiment. Image conclusions do not establish every
mounted sensor's coverage or a category-wide off-map start.

The [supplied assessment](</local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md>) and historical tabletop
adapter support unchanged-size row/table-axis alignment as a hypothesis. Their
claimed same-source table support and guarded front IK stop before grasp belong
to earlier investigators and a different protocol; raw historical measurements
are unavailable locally. The current translation-search rejection neither
refutes that support claim nor proves the new rotated initializer will work.
Historical front-preset approval does not approve these new pending candidates.
Existing correct evaluator rules were retained rather than changed to match
stale historical conclusions.

Current source/generator checks corroborate that source fit labels compare XY
spans, rather than uniquely proving native support. Physical candidate choice is
therefore public and unrestricted among the original candidates. A valid
alternative physical stack may differ from the original QA choice. Original QA
remains auxiliary and private. Current pure scorer checks reproduce 77/77
compatible declared-source answer packets and 54/77 compatible builder-selected
view packets, leaving 23 reference mismatches. These are packet/reference checks,
not Agent accuracy. Pre-action QA execution and legitimate mounted-sensor coverage
still require shared integration; no answers or human reference images become
Agent sensors.

Every candidate retains `assumption_review_required` with doubt, chosen assumption,
alternatives, scope and risk. Finite axis/cell searches can miss feasible layouts;
geometry screens do not certify navigation, IK, acquisition or stack stability.
The final initializer's runtime cost and native compatibility are unknown and may
exceed translation-only startup. A failed support or stance search must remain
diagnostic in Review; new assets, resized originals and teleport success are not
proposed. The existing runner owns publication and future captures/qualification.

Evidence: [complete batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/batch_manifest.json),
[CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json),
[initialization fixtures](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/revision_history/f8445a7f136def80b7e6_893f6514f78e1c5430b85b95/initialization_validation.json),
[inspection evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/inspection_evidence.json), and
[supervisor decision](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/supervisor_result.json).


## Current assignment repair plan — 2026-10-15

Assignment `f8445a7f136def80b7e6`, converter version `2.2.0`, feedback revision
`f8445a7f136def80b7e6_166b0b94e8623c2b942b9478`. This section and the results
below supersede earlier current-state statements; historical sections are retained.

Read the implementation specification, shared conversion/operation/initialization
APIs, current source builder, generator fit predicate, source JSON, supplied
assessment, existing converter and `sample_evidence.json`. The corpus contains
77 three-role sources across 34 scene models. Source wrong-base modes include
39 too-large and 38 too-small objects; wrong-top modes include 45 too-small and
32 too-large objects. Scripted XY-span fit labels do not establish unique native
stack feasibility. Retain public candidate choice with explicit semantic caveats
and keep the original direction QA private and separate.

Plan: reproduce the reported observation-boundary credit defect on CPU, then
apply one targeted rule revision: invalidate lost support credit at every native
read boundary, explicitly enable original object physics before construction and
after source postprocessing, and select the existing measured-surface support
search in the unchanged rigid-group initialization policy. Preserve the empty
original variant, original sizes and relative geometry. Run full-category CPU
conversion, idempotence/private-boundary checks and meaningful native-interface
fixtures. Inspect stratified available historical/current-sample images and retain
missing-image and native initialization failures explicitly. Publish all pending
success/diagnostic records for Review through the supervising runner; do not
launch GPU work, modify shared code or claim physical qualification.

The selected previous L2 capture is supplementary measured evidence: its receipt
reports rejected initialization before placement, zero robot actions and no grasp
qualification. Historical tabletop support and guarded IK stopping before grasp
are attributed to earlier investigators, not to this repair. A broader existing
surface search is a reversible hypothesis, not a demonstrated solution.

## Current assignment results — publish pending Review

**77/77 assigned sources have generated pending L2 scene/task pairs; zero CPU
conversion failures. Decision: publish the complete batch for Review, including
warnings and retained historical failure evidence. No physical success or human
approval is claimed.** Converter version remains the assignment's `2.2.0`;
implementation SHA-256 is
`70586861615dabac762b4bff8ce1e6e35cf261437f542daebaabf7f7365a0878`.
The implementation hash creates new deterministic candidate identities while
preserving all earlier candidates and feedback. Exactly two targeted iterations
were used: physical initialization/predicate repair, then structured QA-reference
uncertainty. The second iteration changed neither scene/task bytes nor evaluator
logic.

### Retained implementation

The converter implements `inspect` and `convert`, and uses the shared
`ConverterBase.convert_all` for deterministic generation, validation, caching and
explicit failure records. It continues to use `original_scene` and
`physical_task`. Every original variant has exactly `patches: []`. Adapted
variants declare `object_physics` for all five original bindings before
construction and after postprocessing, followed by one existing
`prepare_category_interaction` patch. Construction declares the original movable
objects dynamic; the later patch restores `visual_only=false` if source
postprocessing resets it. The current builder already requests the installed R1;
no additional robot, asset or omitted mechanism is invented.

The interaction policy selects the existing `measured_surfaces` search with
`relocation_policy=rigid_group`. It measures same-room original supports and
local upward surfaces, preserves all object sizes and relative poses through a
common translation, and translates only the original reference camera with that
group. It screens the actual floor stance and restores rejected proposals through
the existing initializer. The previous bounded search's measured rejection
motivates this reversible L2 change. It does not establish that translation-only
placement, sensor coverage, IK or grasping will succeed. Common planar rotation
remains an explicit possible later alternative, without a resizing proposal.

The physical recipe remains substantive: choose an original base; grasp B; lift
its root and lowest AABB point at least 0.05 m above their acquisition baselines;
release B on the base; establish 0.5 s of stable placement; then grasp, lift and
release a chosen original top on B. Guarded robot controls perform all actions.
The public task also permits physical recovery and retry. No initializer reset,
source construction success flag, QA answer or action-reported success earns
operation credit.

The private native evaluator requires `OnTop`, `Touching`, matching native
contact-body evidence, stable external support under the base, low measured
linear/angular speeds (0.02 m/s and 0.1 rad/s), released attachments on every arm,
an open working gripper, and robot-clear finished objects. Upper members may
not be propped by scene furniture or unchosen candidates. Stability accrues
only on consecutive controller callbacks using the native environment clock.
Both controller callbacks and observation records now revoke invalid middle
placement credit; records never add elapsed time. Regrasping B at an observation
boundary also revokes its placement credit. A support interruption can recover
only by re-earning the required placement interval before the top acquisition.

### CPU evidence and failure distribution

All commands used
`/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python` with
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2`
and `PYTHONPATH=.`. The complete category was discovered from the assigned corpus
and passed to `ActionOrderConverter().convert_all(sources, ConversionContext(batch_root,
seed=0))`; generated YAML was written only by that API.

| Check | Observed result |
| --- | --- |
| Source coverage / scenes | 77/77 sources, 34 scene models |
| Generated / conversion failed | 77 / 0; every generated scene and task is pending |
| Original QA and source geometry | 77/77 exact private answer/options/metadata packets; unchanged builder assets/scales/poses |
| Private counterfactuals | 77/77 unchanged public/task/scene/interaction definitions after changing private answers, QA, poses and construction flags |
| Idempotence | Identical manifest records and unchanged candidate record modification times |
| Final full generation / cached repeat | 38.592 s / 0.272 s; CPU conversion, not robot throughput |
| Existing native-shaped fixtures | 24/24 pass on final code, 10.881 s |
| Targeted predicate fixtures | 8/8 pass on final code, 9.982 s |
| Disposable malformed-source fixtures | 3/3 produce explicit source-inspection failure records |
| Original-reference QA compatibility | 77/77 compatible declared packets; 23/77 incompatible builder-selected packets |

Before repair, a CPU reproduction earned B placement at step 8, observed support
loss only through `record()`, retained that credit, and later reported synthetic
success after support returned without re-earning B's interval. The final code
rejects that sequence, rejects the equivalent callback interruption and an
observation-only B regrasp, permits recovery after a new stable interval, rejects
extra props, and gives repeated observations no time credit. These are synthetic
native-interface logic checks, not simulator experiments or robot success.

Malformed-source fixture failures are separate from the actual 77-source corpus:
missing original pose, unsupported source task, and invalid JSON. Current corpus
conversion failure distribution is empty. Saved failures remain visible through
their earlier Review records and the new evidence package: one selected previous
native initialization rejection, three saved background cabinet-transform failure
jobs, and one saved compilation/storage failure job with a prior temporary-directory
failure. The saved failures are not current grasp results or a diagnosis of live
server health. Their readable native logs were rechecked during this repair.

### Stratified evidence and remaining uncertainty

The deterministic five-source CPU sample includes the previous native failure,
the source direction ambiguity, and all four combinations of the generator's
wrong-base/wrong-top size modes:

| Candidate | Scene / room | Evidence role |
| --- | --- | --- |
| `category_de3330f6779baacd4782cf8c` | Beechwood_0_garden / dining_room_0 | Previous native initialization rejection |
| `category_720f4f4a6d618debb6fb50cf` | Beechwood_0_garden / kitchen_0 | Both wrong candidates marked too small |
| `category_4d20623ea73a7bd269ee7576` | Beechwood_1_int / bedroom_0 | Original duplicated top direction; both marked too large |
| `category_e9513458b5013250fc23e920` | Pomaria_1_int / living_room_0 | Too-large base / too-small top |
| `category_e2069c2137a8cdc7453de270` | Wainscott_1_int / playroom_0 | Additional scene and support-layout stratum |

Read the current [sample evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json).
It describes older v2.1.0 samples, not these new identities. Its five historical
source strata were checked; all 18 referenced diagnostic images are unavailable.
No new visual conclusions are drawn from those missing paths or old queue states.

Four images from the earlier selected v2.2.0 capture were actually viewed:
[overhead](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/global_overhead.png),
[source reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/scene.png),
[head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/head.png),
and [wrist](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/wrist.png).
The objects remain separated on the floor, the source-reference endpoint is
cropped, and both mounted-camera frames omit the task objects. Its receipt says
`accepted=false`, `relocated=false`, no placement attempts, zero robot actions,
no native qualification, and no verified grasp. Its 217.587 s capture duration is
historical diagnostic cost. These are not images or timings of the new converter
implementation, and one available scene cannot validate the category.

The supplementary assessment's unchanged-size tabletop support and guarded IK
stop remain historical claims by earlier investigators; this repair did not run
those experiments. The previous bounded-search rejection does not refute the
different common-transform tabletop protocol, nor does historical table support
qualify the new search. Source fit labels likewise do not prove unique physical
candidate choice. Public physical success intentionally accepts either candidate;
original direction QA correctness remains independent.

All 77 records retain `assumption_review_required` with doubt, chosen assumption,
alternatives, scope and risk. Warnings cover pending native layout, stance/grasp,
stack execution, measured-surface search, initial QA integration and physical
candidate semantics. One source additionally retains `source_qa_direction_ambiguity`.
The independently confirmed 23-view mismatch remains a shared initial-QA
integration limitation: preserve the source answer and declare a legitimate
pre-action observation/reference contract separately. Human Review images are
never Agent observations. No native experiment, model evaluation, GPU worker,
commit or sub-agent was launched by this assignment repair.

Artifacts: [complete BatchManifest](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/batch_manifest.json),
[CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json),
[inspection evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/inspection_evidence.json),
[predicate fixtures](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/revision_history/f8445a7f136def80b7e6_166b0b94e8623c2b942b9478/targeted_validation.json),
and [supervisor result](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/supervisor_result.json).
Earlier canonical manifest, sample set, supervisor decision and CPU results are
preserved under this feedback revision's `revision_history` directory. The main
runner owns actual CaptureQueue and Gallery publication; this handoff requests
Review for every result and does not mark any candidate ready or approved.

## Placement repair correction — version 2.2.0, 2026-10-15

**L2 PENDING: the user reports wrong robot starts throughout this category and explicitly authorizes moving the unchanged original objects onto reachable existing scene supports, as for Rigid Containment. All rendering and native execution are paused. No placement, IK, grasp or stacking success is claimed by this CPU revision.** This instruction supersedes the historical no-repair decisions below; those dated sections remain intact as evidence of earlier decisions.

Brief implementation plan: audit all 77 source/candidate records and their saved native captures for source start configuration, object heights, shape and viewing/access failures; add one deterministic category-wide manipulation policy; then generate a separate CPU-only revision and verify coverage, pending status, original scales/bindings/QA and the unchanged physical evaluator. Main owns the shared native placement adapter. It must measure an existing support and collision-free floor stance, preserve original object sizes and relative arrangement with a common group translation, translate the source reference camera by the same amount, and leave robot-mounted sensors on their native mounts. An impossible native placement must produce explicit failure evidence. Height, geometry and distance screens are proposals, not IK or grasp qualification.

This revision writes only the category converter, this report and new CPU evidence under [placement_review/action_order](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order). It does not replace the canonical campaign batch, publish Review records, edit source JSON, resume workers or alter old evidence. The audit and validation results follow.

### Complete saved-placement audit and correction rationale

The CPU audit on **2026-10-15** covers all **77 current v2.1.0 source/candidate records, 34 scene models and 385 original object bindings**; its JSON records the exact UTC snapshot time. Every object retains its source model, scale, root pose, quaternion and saved AABB evidence in [placement_audit.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/placement_audit.json). All 77 sources omit a robot start pose; the Action Order builder has no custom environment/spawn hook and its postprocess returns an empty state. The installed native `r1_primitives.yaml` also specifies neither robot position nor orientation. This demonstrates the absence of a category-specific stance rule; it does not recover the actual robot root pose after native loading.

All **385 saved source AABB centres have absolute Z below 0.35 m**, ranging from -1.12646 to 0.14416 m; root Z ranges from -1.13242 to 0.12889 m. These coordinates and saved AABB dimensions expose a low-placement risk, not measured floor-relative grasp heights or proof of ungraspability. Source bounds do not establish live collision meshes, stable contacts, finger approach geometry, reach or IK. The unchanged group spans reach 2.250 m along X and 2.517 m along Y across different sources. Some groups may not fit any permitted existing support; failure must remain visible instead of resizing objects or independently rearranging candidates.

The assigned queue contains **162 matching unique jobs** across v2.0.0, v2.1.0 and earlier source-only diagnostics: 51 `REVIEW_PENDING`, 104 `QUEUED`, 3 `RUNNING` and 4 `FAILED` status files. These are saved queue states, **not live process states**; rendering is paused. Fifty-one complete receipts cover 35 unique sources. For the 77 current v2.1.0 candidates specifically, the saved states are 19 `REVIEW_PENDING`, 54 `QUEUED`, 3 `RUNNING` and 1 `FAILED`. All 77 records remain included even where images are absent.

Only four source instances have saved native object root readbacks: Beechwood dining room, Ihlen storage room, Pomaria bathroom and Rs_garden kitchen. Their five-object native Z readings range from 0.00071 to 0.06330 m across those records. No inspected retained-state/initialization/receipt packet contains an explicit robot root-pose readback. Every complete receipt reports no native qualification and no verified grasp. Consequently the user's report of wrong starts throughout the category is the repair instruction, while the saved evidence independently establishes missing stance qualification and repeated viewing/placement concerns. This agent did not run the historical experiments or remeasure native geometry.

Nine existing v2.1.0 images were actually viewed; they are old unpatched diagnostic evidence, not renders of this repair:

| Source scene / room | Viewed evidence | Observation |
| --- | --- | --- |
| office_cubicles_right / copy_room_0 | [Head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_a04c7ec5b0e67fecf43113913ada3534/attempts/01/head.png), [overhead](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_a04c7ec5b0e67fecf43113913ada3534/attempts/01/global_overhead.png), [source reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_a04c7ec5b0e67fecf43113913ada3534/attempts/01/scene.png) | All five separated candidates are visible in human views. The head shows arms, sky and a wall/ceiling surface without task objects. |
| Beechwood_1_int / bedroom_0 | [Head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_fe7bb5b260d0984e8906b4f27e5b24af/attempts/01/head.png), [diagonal](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_fe7bb5b260d0984e8906b4f27e5b24af/attempts/01/global_diagonal.png), [source reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_fe7bb5b260d0984e8906b4f27e5b24af/attempts/01/scene.png) | The row is on the floor beside the bed; one tray extends beneath the bed and the near tray is partly cropped. The head misses all task objects. |
| Rs_garden / kitchen_0 | [Head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_149d3c48cb022ea69e25aa7bb1a97d9a/attempts/01/head.png), [overhead](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_149d3c48cb022ea69e25aa7bb1a97d9a/attempts/01/global_overhead.png), [source reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_149d3c48cb022ea69e25aa7bb1a97d9a/attempts/01/scene.png) | The source reference is almost entirely black; overhead shows a narrow fixture gap and partly obscured endpoint. The head misses all task objects. |

The separate [image review](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/image_review.json) records observations and limitations. Existing restaurant Asian/diner kitchen failure logs report the background cabinet emitter's non-orthogonal-transform assertion before manipulation; those failures are not classified as grasp failures. Across the 51 existing complete diagnostic receipts, capture duration is 134.6 / 271.7 / 1539.2 s (minimum/median/maximum), including initialization and screenshot work but excluding queue delay. These are inherited diagnostic costs, not placement-repair runtime, manipulation throughput or model quality.

Version 2.2.0 declares `mode=manipulation`, all five original **binding keys** as the rigid group, and B plus both available top candidates as manipulation targets. Its screen requests grasp height 0.35–1.10 m and stance distance at most 0.85 m in the original source room. The two-role source contract groups only its three originals and manipulates B. These values guide Main's measured support/stance search; they are not native reach or grasp certifications. Public roles, candidate-agnostic physical goals, all original bindings, auxiliary private QA and the native ordered grasp/lift/release/support evaluator remain unchanged.

### Isolated CPU revision and retained limitations

The shared initialization integration is now included in [the isolated revised batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/revised_batch/batch_manifest.json): **77/77 generated pending L2 scene/task pairs, zero conversion/schema failures, 34 scene models**. Every original variant remains exactly unpatched. Every adapted variant appends one `prepare_category_interaction` patch after source postprocessing with the exact binding-key policy above. This is a declarative patch, not an executed relocation; all native support names, stance poses and group/reference transforms remain unmeasured until separately authorized execution. The canonical category batch and all previous candidates remain intact; no record is published or approved.

Validation ran from the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python /home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/validate_placement.py
```

The initial isolated conversion took **15.614 s**. Complete coverage, new candidate identities, exact original QA, unchanged public text/private operation/qualification, original builder models/scales/poses, all five bindings, exact L2 policy, pending status and no publication passed across all 77 records. Private-answer/QA/pose/selected-object counterfactuals changed neither public fields nor physical operation/policy. The supported two-role contract also passed. Repeating the same batch preserved identical records and record modification times. Source hashes and five protected canonical campaign artifact hashes remained unchanged; all old candidate directories remain present.

The entire evaluator suffix is byte-identical to the frozen v2.1.0 converter (`SHA-256 b14f11e75479558dbc8b21bce0599463e2672c2d7fc56240401efe614659931e`). All **24 existing synthetic native-interface fixtures passed in 4.222 s**, writing only the new [fixture evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/native_interface_validation.json). This verifies no placement-revision regression in those CPU fixtures; it does not remove the previously reported evaluator observation-boundary concern or qualify actual native APIs, IK, acquisition or stacking. No control/state/pose mutation call was added to the evaluator.

Detailed checks and measured CPU times are in [cpu_validation.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/cpu_validation.json). All 77 candidates carry `l2_manipulation_placement_pending`, `support_relocation_authorized`, `native_stance_and_grasp_unverified`, `interaction_layout_native_review_pending`, `native_stack_execution_unverified`, `initial_qa_reference_integration_unverified` and `assumption_review_required`; the original one-item direction ambiguity remains warned. The original selected-view QA discrepancy (23/77), actual post-relocation reference visibility, native support/stance clearance, reach, grasp and released stack remain pending. Screens must not weaken the native evaluator or automatically approve a task. **Rendering remains paused; zero capture jobs, native processes or robot actions were launched by this revision.**

## Initial evidence and brief research plan

Assignment: `f8445a7f136def80b7e6`; converter version: `2.0.0`.

The assignment advice recommends L1 reuse with a robot goal that realizes the original support order. Historical evidence comes from [research_log.md](/local_home2/zhouxinyang/models/Show-Harness-Explore/research_log.md:2553) and [the committed exploration manifest](/local_home2/zhouxinyang/models/Show-Harness-Explore/docs/esi_category_exploration_latest.json). The historical `/scratch` report is absent on this server. Earlier native probes verified unchanged-size tabletop support, then stopped on guarded IK before grasp; they did not qualify a released stack. Those experiments were not run by this category supervisor.

The local [active builder](/local_home2/zhouxinyang/models/Show-Harness-Explore/ESI-Bench/src/active_explore/tasks/action_sequencing/action_order_inference.py) restores original objects, scales and initial poses and selects an original orbital reference view. The [generator](/local_home2/zhouxinyang/models/Show-Harness-Explore/ESI-Bench/src/dataset_generation/task_action_sequencing/batch_action.py) uses scripted object placement and reset trials. Its introductory scaling description is incomplete: the actual code permits wrong bases that are too large and wrong top objects that are too small. Source fit labels therefore remain private target-selection evidence, not a proof of native physical uniqueness or feasibility.

Plan: implement a thin deterministic converter, preserve all geometry as L1, derive public roles only from the source question, bind the source-selected order privately, and run full-category CPU conversion plus checks of coverage, pending status, preserved QA and public/private separation. Main's existing runner owns native capture and Review publication. No GPU or simulation workload is launched by this supervisor.

## Conversion rules and success predicate

- Keep the original scene variant with no patches. No object, scale, camera, robot pose or support is changed.
- The public goal asks the robot to choose the base and top candidates by category and physically build the stack. It supplies no answer-selected side, private asset identity or correct/wrong label.
- Private bindings include all five original objects. Private relations require `obj_fixed` on `obj_A_correct` and `obj_C_correct` on `obj_fixed` for the current three-role corpus. The two-role source contract is also supported.
- The independent native `stack_relations` evaluator requires both relations simultaneously, real contact, upright objects, released objects, velocity limits and 15 consecutive physics steps of stability. Grasp/lift alone does not satisfy the task.
- Both scene and task remain `pending`; CPU-generated candidates are not physically qualified or approved. Original QA is retained verbatim in the shared batch output and evaluated separately.

## Iteration 1 — preserve native layouts

Hypothesis: a pending stacking task can be generated for the entire category without changing source geometry or leaking its selected identities. Initial CPU source inspection found 77 instances across 34 scene models, all with three stack roles, five objects and complete saved poses. There is no category-wide native evidence warranting an L2 common shift or smaller assets. Retain L1 and the original builder.

Observed source failure: instance `0009`, `Beechwood_1_int/bedroom_0`, gives both top candidates the direction `below`. Cause assessment: an ambiguity already present in the source QA. Rule retained: preserve that QA, flag `source_qa_direction_ambiguity`, and define the physical goal through public category roles rather than silently rewriting the direction answer.

All candidates receive `native_stack_execution_unverified` and `assumption_review_required`. The structured assumption records the doubt, conservative pending-L1 choice, alternatives, scope and risk. Reachability and native stack stability remain unverified; no scene repair has been attempted.

## CPU validation and sampled evidence

Validation command, run from the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python /home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/validate_category.py
```

The full batch contains **77/77 generated candidates, zero conversion failures, 34 scene models and 77 L1 reuse decisions**. All 77 scene/task pairs remain pending. Warning counts: `assumption_review_required=77`, `native_stack_execution_unverified=77`, `source_qa_direction_ambiguity=1`. Failure distribution is empty at the CPU conversion stage; no native outcome has been measured.

Checks passed for complete one-record-per-source coverage, explicit original asset bindings, original private QA, correct private support order, zero scene patches, pending status, structured assumptions and idempotent record reuse. Changing private answers, QA, poses and selected candidate metadata left all public task fields unchanged across the entire corpus. The first validation attempt produced a tooling false positive because a substring check found `right` inside `upright`; changing that check to word boundaries resolved it without changing conversion rules. The final cached batch-conversion pass took 0.664 seconds; this is not a cold generation or native runtime measurement.

Deterministic seed-0 samples cover five scene models and explicitly include the source-QA ambiguity rule:

| Candidate | Source scene / room | Static result |
| --- | --- | --- |
| `category_fa95e697ffca69cdb9194d9b` | house_double_floor_lower / living_room_0 | Generated; original layout |
| `category_fae5a77356b161d8b7888a04` | Beechwood_0_garden / living_room_1 | Generated; original layout |
| `category_797200c36a554201089972d0` | office_large / phone_room_0 | Generated; original layout |
| `category_33003d34a4f3ee4e178aa0b2` | Wainscott_0_garden / living_room_1 | Generated; original layout |
| `category_35c265a608d5ee705519ebed` | Beechwood_1_int / bedroom_0 | Generated; duplicate source top direction flagged |

Artifacts: [BatchManifest](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/batch_manifest.json), [CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json), [sample set](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_set.json). No failed conversion sample exists to include. Current-revision native images have not yet been received from Main's capture runner. No native-load, capture or physical outcome is inferred from static generation.

## Iteration 2 — historical original-scene evidence review

Hypothesis: existing original-scene images can identify framing and access uncertainties before current-revision capture completes, without treating diagnostic images as task qualification. Matched exact assigned `source_identity` values against existing capture manifests and restricted historical evidence to `variant=original`, `reuse_level=original` and `task_file=null`. This excludes the converter candidates being published concurrently. The audit found **eight historical capture manifests for four unique assigned sources**; ten images from their four most recent original captures were actually viewed. All eight historical receipts report `complete=true`, `native_qualification_run=false`, `robot_actions=0` and `physical_grasp_verified=false`. The four inspected scene snapshots have no patches, target or support qualification.

The inspected frames are **historical original-scene evidence**, not images of converter version `2.0.0` tasks. The four sources map to these current pending candidates:

| Source scene / room | Current pending candidate | Historical original frames actually viewed |
| --- | --- | --- |
| Beechwood_0_garden / dining_room_0 | `category_b328da0dd1efa097e39d61d9` | [Reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_e450d31f5e9aa809abfc622fce3aad56/attempts/01/scene.png), [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_e450d31f5e9aa809abfc622fce3aad56/attempts/01/head.png) |
| Ihlen_0_int / storage_room_0 | `category_c6c61afc5bc6e55a64274016` | [Reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_33bbdc9be6db83575cf04568b7dadd37/attempts/01/scene.png), [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_33bbdc9be6db83575cf04568b7dadd37/attempts/01/head.png) |
| Pomaria_0_int / bathroom_0 | `category_7f8eebb432091795c7ebd043` | [Reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_a3eb34dd0106873aae8cf51c9830366d/attempts/01/scene.png), [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_a3eb34dd0106873aae8cf51c9830366d/attempts/01/head.png) |
| Rs_garden / kitchen_0 | `category_2c2e0c2390e9b7bac9c2b81c` | [Reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_bcfaf7446720efd32e8855ce233985d9/attempts/01/scene.png), [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_bcfaf7446720efd32e8855ce233985d9/attempts/01/head.png), [diagonal](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_bcfaf7446720efd32e8855ce233985d9/attempts/01/global_diagonal.png), [overhead](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_bcfaf7446720efd32e8855ce233985d9/attempts/01/global_overhead.png) |

Observed findings:

- Beechwood, Ihlen and Pomaria reference frames show separated candidates and the middle reference object in a floor row, with no completed stack. The near row endpoint is cropped in the Beechwood and Ihlen reference frames. These are visually inspectable original layouts, not evidence of native contact, stability or reach.
- Rs_garden's reference image is almost entirely black despite a complete receipt. Its overhead view exposes the candidate row between kitchen fixtures; the diagonal view has substantial obstruction. Capture-file completion does not certify reference-image usefulness. Camera occlusion or framing is a possible cause, but the exact cause is unverified.
- All four historical head frames mainly show sky, robot arms and an upper wall/ceiling patch, without the task-object row. This is a shared camera/coverage concern reported to Main. Human reference/global views do not establish initial Agent sensor coverage.

No converter change or L2 layout repair is warranted from these images alone, so version `2.0.0` is retained. The original frames were generated by earlier capture workers, not this supervisor. Earlier original captures report 110.4–149.4 seconds of cold initialization and 17.0–24.6 seconds of screenshot work for the four retained records; those timings are historical diagnostic costs, not current task throughput or physical execution latency.

**Decision: publish all 77 pending records with existing warnings for human Review. Current-revision native capture remains pending.** Main reports that eight original worlds remain retained; eight existing receipts have `world_retained=true`, four belonging to this category. This supervisor did not recheck live process state, submit capture work or change any world lifecycle. Detailed provenance, receipts and viewed-image paths are saved in [historical_original_evidence.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/historical_original_evidence.json); the review decision is saved in [supervisor_result.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/supervisor_result.json).

## Review feedback

No human Review feedback has been received. Main reports that automatic full publication is running. The supervisor decision authorizes Review publication and leaves physical qualification pending; subsequent feedback must revise category rules and preserve earlier candidate revisions.

## Supplementary assessment comparison — 2026-10-15

Review fingerprint: `0e513fe9392ab82ae78cbd6fccd290a0a24fb1566c943bb2aa557defc3b2eccd`. Scope: this category only, including every current conversion, discovered earlier candidate revision, previous scene definition and assigned CaptureQueue diagnostic. The assignment's `existing_task_files` list is empty.

Read-only research plan: compare the implementation specification and supplementary assessment with the actual source builder/generator and converter, audit the complete saved batch and earlier definitions, inspect current capture receipts and representative images, and evaluate the operation, terminal predicate and initial-QA contracts. This review ran CPU checks without simulation; it did not rerun the saved validator or `convert_all`, because those commands write generated artifacts. Recommended corrections below require Main's implementation review and do not gate Review publication or screenshots.

### Historical assessment versus current implementation

The supplied [assessment](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:48) is available locally and matches the supplied SHA-256 `5bfdf6f3fb408bc7bd8aedd9873d6ddc5639dbff18237d20312ecc689275dd07`. Its claims describe earlier representative experiments, not new measurements by this reviewer. All 24 checked relevant historical raw-artifact references, including native probes, support verification, scorer checks and offline-planning reports under `/scratch`, are unavailable locally.

| Assessment claim | Comparison with current evidence |
| --- | --- |
| Five unchanged-size objects had stable support on the existing Beechwood table; front grasp attempts stopped before acquisition and side variants failed support/contact checks. | The local tabletop adapter and saved manifests describe those checks, but the raw trials cannot be independently remeasured here. The historical attempt table also distinguishes an original diagnostic-callback failure from a physical rejection, an excessive base-command failure from bounded IK failure, and unsuccessful side initialization. None demonstrates a complete stack. |
| Prefer the previously selected tabletop and preserve the directional reference; no resizing is justified. | The current converter uses **unpatched L1 source layouts**, not that tabletop variant. This is a real difference, not automatically an implementation error: one representative relocation does not establish a category-wide need for it. Current images still show floor-level rows and access/occlusion concerns. Any future stance or common object/reference transform needs instance-specific native evidence and a recorded reversible L2 rule. |
| Original candidate-direction QA and real physical success must be separate; an alternative stable candidate is not necessarily the source-selected answer. | Correct and still material. The current predicate uses the source-selected identities privately and measures native relations. Public wording, candidate uniqueness and the original QA reference frame remain unresolved below. |
| Direction answers depend on the declared source view even if the active runner selects another view. | Directly corroborated by this review: **23/77** preserved answer packets disagree with the active builder's selected-view scorer, although **77/77** pass with their own declared source QA. This is a current CPU result, not a measured model-accuracy result. |
| Offline reach/planning or scene initialization can guide a later pilot. | The retained planning manifest explicitly records zero simulation and robot commands and unverified physical execution. Proposed chair clearance and numerical IK candidates are neither executed repairs nor category-wide feasibility evidence. Historical “approved tabletop” wording does not establish approval of any current candidate. |

### Complete batch and previous-task coverage

The saved [CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json) reports 77 generated records, no CPU conversion failures and a 0.664 s cached conversion pass. New read-only checks independently inspected all 77 sources, records, scene/task pairs and private-QA files. They confirm exact one-record-per-source coverage, 34 scene models, five original bindings and three roles per source, pending status, empty original patches, unchanged builder assets/scales/poses, and verbatim private answer/options/metadata preservation. In-memory output from the current converter matches every saved scene/task definition. Public text remains unchanged under private-answer, pose, QA and selected-candidate mutations across all 77 sources. These checks validate static boundaries, not live sensor privacy or manipulation feasibility.

Both earlier batch snapshots, `088e71cad89a8073d4c1cd8e` and `d7e1747de049bb49158f12f7`, contain the same 77 candidate identities at converter version `2.0.0`; no additional earlier converter-record revision was found in this category's candidates directory.

Previous tasks were reviewed as read-only history:

- All 77 current catalogue `scene_import` definitions and the 77 definitions in each frozen revision `057b58e3fb1fae09c02db7a41f53b6677eb6b5c7`, `4aa1c7d00f44995746151d52258c4f1bb89f3659` and `f8039649c71af53b5d54b169d96982dcd94306e1` remain pending; all 308 accompanying JSON copies match the corresponding source JSON.
- The four earlier diagnostic candidates are `pilot_action_order_original_v1`, `local_action_order_ihlen_0_int_storage_room_0_original_v1`, `local_action_order_pomaria_0_int_bathroom_0_original_v1` and `local_action_order_rs_garden_kitchen_0_original_v1`. Their working and frozen scene definitions have empty patches and no physical target or qualification; their handoffs have `task_file=null`. They provide useful framing evidence, but no operational task or terminal evaluator.
- All eight historical receipts in the assigned queue were included: two captures per earlier candidate, at renderer code revisions `f8039649...` and `4aa1c7d...`. All are `REVIEW_PENDING` diagnostics with zero robot actions and no native qualification. Four historical receipts say `world_retained=true`; that is not evidence that those processes are still alive.
- Across the discovered local queue roots, 93 matching manifest entries represent 85 unique capture jobs, including those eight historical jobs and 77 current jobs. All 93 scene snapshots and source snapshots were read; the scene snapshots are pending and unpatched, the source JSON matches the corpus, and their active-builder snapshots match current builder code. None has a frozen physical task snapshot. No category task was found in the established task catalogue. The empty assignment list does not erase this earlier diagnostic history.

The inventories, affected source paths and per-candidate status paths are included in [the review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/f8445a7f136def80b7e6/0e513fe9392ab82ae78cbd6f/result.json).

### Meaningful operations and useful recipes

**The proposed full stack is a meaningful compound physical goal; actual robot execution remains unmeasured.** It requires moving B onto a selected original base and placing a selected original top object on B, then releasing the resulting stack. The current batch supplies one operation recipe across 77 asset/category triples. Its two-role code path is supported syntactically but has no source instance in this corpus.

Several source-faithful recipes would be useful as separately named tasks: the complete stack, a bounded trial-placement/physical-return recipe before completing that stack, and a B-on-base subgoal for diagnosing acquisition and support before adding the top object. A top-placement subgoal can similarly isolate the second manipulation. Subgoals must retain the original five objects and private QA and must not be reported as completion of the full three-role problem. Physical return requires guarded robot movement, not resetting an object to its saved pose. No new recipe was generated by this review.

There is a material public/private semantic mismatch. The public objective says “choose one” candidate, but the private goal accepts only `obj_A_correct` and `obj_C_correct`. Source generation contains 39 “too_large” wrong bases and 45 “too_small” wrong tops, alongside 38 too-small bases and 32 too-large tops. Its [fit check](/local_home2/zhouxinyang/models/Show-Harness-Explore/ESI-Bench/src/dataset_generation/task_action_sequencing/batch_action.py:285) tests XY span inequalities after scripted placement, not native contact, release, velocities or unique physical feasibility. All 77 source `success` flags are true; these are historical construction labels. They do not prove that the alternate candidates fail the current physical predicate. Thus a publicly acceptable stable alternative could be privately rejected. Main should explicitly decide how an observable selection criterion and source-choice score relate to physical support success; neither silently changing answers nor exposing correct/wrong identities is justified.

### Independent native predicate and initial-QA boundaries

The [shared stack evaluator](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/sim/esi_category_evaluation.py:54) is a useful independent native foundation. It reads both requested relations simultaneously using `OnTop` and `Touching`, requires movable collidable participants, local-Z/world-Z alignment at least 0.9, linear speed at most 0.02 m/s and angular speed at most 0.1 rad/s, and rejects a named object held by the working gripper. It reads native state without placing objects, and it does not use source success flags or QA correctness as proof. Pure CPU predicate checks reject absent contact/support/upright features and a named held participant; these are synthetic logic checks, not simulation results.

It does **not yet certify the complete advertised operation**:

- `minimum_lift_m=0.05` is saved but unused by `StackRelationsEvaluator`. No acquired-grasp/lift/release history or transition from an initially incomplete stack is required. An already valid state can accrue stability without the requested manipulation.
- The public “Then” order is not checked. Final simultaneous relations cannot distinguish B-then-C construction from another action sequence. Main should either measure the declared sequence or explicitly define a final-support-order recipe.
- “Released” currently means that the working gripper's privileged `object_name` is not a participant. The predicate does not explicitly check other robot support contacts, `constraint_active` or `release_pending`. This is a limited release proxy, not proof of a freestanding stack.
- The counter advances through `controller.on_step` after an `env.step` call ([controller](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/sim/esi_base.py:198), [adapter](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/tasks/scene.py:358)). The earlier report's “15 consecutive physics steps” wording is too strong: the code measures 15 controller callbacks, without a declared physical duration or every-physics-substep continuity. The universal upright check also needs justification for round fruit and other orientation-insensitive source objects.
- Initial task qualification checks capabilities, bindings and the middle target's mobility, with no support declaration or visibility requirement. The evaluator later checks the three selected participants' mobility, but no initial grasp reach, collision clearance, settled support or genuine Agent coverage has been established. These gaps concern future execution readiness; diagnostic capture should remain available.

The static initial-QA boundary is sound: encoded source QA, answers, poses and selected identities are private, while the physical public goal derives categories from the source question. However, saving `private_qa.json` is not an executed auxiliary QA evaluation. The category task path inspected here installs a physical evaluator and does not itself run the original Action Order scorer. Initial QA should be evaluated from legitimate pre-action observations against its declared fixed source reference, independently of later manipulation; private human reference/global images are not Agent sensors.

The [active builder's view selection](/local_home2/zhouxinyang/models/Show-Harness-Explore/ESI-Bench/src/active_explore/tasks/action_sequencing/action_order_inference.py:68) changes the declared QA directions in 23 sources. For example, the house-double-floor-lower sample changes A/C from right/left at source view 0 to left/right at selected view 2. The source ambiguity at Beechwood bedroom is correctly retained and warned. Active view selection additionally creates a duplicate C direction in `office_large/private_office_2` at view 3, so the converter's one source-QA ambiguity warning does not cover every runtime reference ambiguity. Main should reconcile and record the source-reference, initial-observation and scorer contracts before physical or QA readiness claims.

### Current CaptureQueue diagnostics and inspected images

Snapshot time: **2026-10-14 19:55:07 UTC / 2026-10-15 03:55:07 Asia/Singapore**. All 77 current records still have `review_submitted=true`: **19 REVIEW_PENDING, 57 QUEUED, 1 FAILED**. No assigned job was running/claimed at this snapshot; queued work can progress independently. The eight earlier diagnostic jobs are separately `REVIEW_PENDING`. This supersedes the stale five-sample queued/zero-complete snapshot in [sample_evidence.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json).

All five stratified current samples now have complete nine-image receipts. This reviewer actually viewed 12 current images: scene/reference and head for each sample, plus office phone-room overhead and wrist. The images show:

| Sample | Current visual observation |
| --- | --- |
| house_double_floor_lower / living_room_0 | Five separated floor-level objects visible; no completed stack. |
| Beechwood_0_garden / living_room_1 | Five separated candidates around furniture; no completed stack. |
| Wainscott_0_garden / living_room_1 | Floor-level row beside/under a table and sofa, with obstruction; a clear robot approach is not established. |
| Beechwood_1_int / bedroom_0 | Bucket and lemons visible; tray endpoints are cropped or partly obscured by the bed/frame. |
| office_large / phone_room_0 | Reference image is almost uniformly blank. The overhead exposes some objects near an obstructing chair; the inspected wrist view does not show the task objects. |

All five inspected head views miss the task row and mainly show sky, arms or upper room surfaces. This corroborates a camera/initialization concern, but does not prove that every enabled body sensor or reachable robot viewpoint is unusable. Exact cause and a safe repair remain unverified.

The 19 current complete receipts account for 171 image files and consistently record `native_qualification_run=false`, `robot_actions=0`, `model_calls=0`, `physical_grasp_verified=false` and `eligible_final_layout=false`. They are native initialization/framing evidence, not task evaluation. Their recorded capture duration is 174.6 / 267.5 / 599.6 s (minimum / median / maximum); cold initialization is 138.6 / 198.3 / 536.6 s and screenshot work 24.4 / 45.5 / 89.0 s. These exclude queue delay and do not estimate manipulation latency or model throughput.

The failed current candidate is `category_169faf455a3aee66f04784ec`, restaurant_asian / kitchen_0, [job capture_20e6347728530d74621f8319f2742811](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_20e6347728530d74621f8319f2742811/status.json). Its [native log](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_20e6347728530d74621f8319f2742811/attempts/01/native.log) ends during background scene/emitter initialization with `bottom_cabinet_pkdnbu_0/base_link/emitter local transform is not orthogonal`. No screenshots were produced. The observed failure precedes grasp or stack evaluation; it is not evidence that the selected hierarchy is physically impossible.

### Retained decision and Main recommendations

Retain all current and earlier records as pending/diagnostic evidence. Keep failed and queued instances visible in Review and let capture continue; this assessment introduces no new queue or approval gate. Main should review the public-selection contract, action/release/sequence evidence, stability clock and orientation policy, fixed initial-QA reference/scorer wiring, and narrowly justified robot/camera/load repairs. Native changes and new recipes should become traceable future revisions rather than edits to these historical candidates.

No implementation, generated scene, source, answer, approval, other category report, root research log or workload was changed. No GPU work, simulation, commit or sub-agent was launched. This assessment claims neither robot success nor human approval.

## Reviewed operation correction — version 2.1.0, 2026-10-15

**Decision: publish all 77 new pending L1 records for Review. No native manipulation or current-version capture was run.** This section records the subsequent authorized implementation correction; it does not rewrite the historical assessment above. The former module, report, manifest, validation, samples and supervisor decision are preserved in [revision_history/v2.0.0](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/revision_history/v2.0.0). All 77 old candidate directories remain intact and are distinct from the 77 new candidate identities. Historical links to the campaign-root manifest/validation files now resolve to this revision; their exact former files are in that archive.

### Authority, evidence and retained direction

The implementation specification [esi_category_parallel_conversion_spec_v2.md](/local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md) and Main's reviewed operation decision govern this correction. The [supplied report](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:48) is supplementary historical information, potentially incomplete or wrong. The category assessment indexed by [assessment_review.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_review.json) is retained at [result.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/f8445a7f136def80b7e6/0e513fe9392ab82ae78cbd6f/result.json), fingerprint `0e513fe9392ab82ae78cbd6fccd290a0a24fb1566c943bb2aa557defc3b2eccd`.

Main explicitly resolved the candidate-selection ambiguity: either original base and either original top may satisfy the physical task when their measured support order is valid. Source-selected identity/direction QA stays separate and unchanged. This differs from the historical report's suggested role restriction and the old converter's hidden exact-identity predicate. Current code and the final assessment support the correction: scripted XY fit labels do not demonstrate unique native physical feasibility.

The bounded hypothesis is that the original five-object scenes can remain useful pending manipulation tasks if the terminal adapter measures the advertised action history independently. No source-wide geometric repair is justified by the historical support/IK attempts or images. The original assets, dimensions, poses, layout, reference camera and QA therefore remain L1 and unpatched; no L2 layout rule or native repair was introduced.

### Operation and private native predicate

The owned [module](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/action_order.py) now exports the category-operation qualification and evaluation hooks. Public categories come only from the source question. The public goal explicitly requires grasping and lifting B by at least 5 cm, placing and releasing B on either original base until stable, then grasping/lifting either original top by at least 5 cm and placing/releasing it on B. No public text reveals source correct/wrong identities or directional answers. All five originals are privately bound; no answer selects the accepted pair.

The adapter reads native object poses, AABBs, linear/angular velocities, `OnTop`, `Touching`, contact bodies, all-arm native grasp states, assisted attachments and release counters. It never executes controls, steps the environment, sets a state or changes a pose. Qualification requires the original movable/collidable objects to have measured settled native scene support, all arms to be released and the initial stack to be incomplete. This inspects initialization only; it does not qualify reach, sensor coverage or successful manipulation.

During an episode, a moved object earns acquisition only when the native working-arm hold identity, positive grasp state and active attachment agree and other arms are released. Lift requires both root height and AABB-bottom displacement of at least 0.05 m while that acquired grasp remains measured, relative to the greater of episode-initial and acquisition height. This conservative baseline prevents pre-grasp displacement or orientation alone from earning lift credit. A released placement requires all native arm holds, constraints and release windows to clear, plus the working gripper's measured open state.

B placement is confirmed only after real acquired-grasp/lift/release history and stable native support on either base, with no top already on B. The chosen base must have actual native supporting contact with a stationary original scene support, rather than an assumed floor height. The selected top's acquired grasp must occur after B's confirmed placement. Native simultaneous B/base and top/B `OnTop`, `Touching` and matching contact bodies then determine the accepted pair. Held objects, robot contact with the selected stack, missing base support, excessive motion and unavailable or contradictory readbacks reject success. Regrasping B or losing its support invalidates its earlier placement; the final support order alone cannot supply missing action history.

Both stable B placement and the final stack require at least **0.5 seconds of consecutive native controller-step samples**, with linear speed at most 0.02 m/s and angular speed at most 0.1 rad/s for selected participants and low motion of the native external support. The adapter checks controller `dt` against native environment action frequency and records sample count, `dt` and elapsed stable duration. `record()` calls cannot advance this clock and can revoke interrupted stability. These are declared task thresholds, not calibrated native feasibility results, and they do not certify every physics substep or long-term stability. The universal local-Z upright check was removed because source top objects include orientation-insensitive fruit; measured support/contact and motion remain required.

### CPU iteration, coverage and samples

The first complete version-2.1.0 conversion generated **77/77 pending L1 tasks across 34 scene models**, with zero conversion/schema failures, in **16.320 s**. Repeating `convert_all` reused identical records and candidate identities without changing record modification times. All 77 sources passed source/QA preservation, unchanged original patches, five original asset bindings, operation schema, public/private isolation, meaningful grasp/lift/order wording, structured assumptions and private-label counterfactual checks. Changing private answers, poses, view QA and selected-candidate metadata did not change public text or the physical operation specification.

The owned [CPU validator](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/validate_category.py) ran as:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python /home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/validate_category.py
```

All **24 synthetic native-interface fixtures passed in 4.812 s**. They drive native-shaped objects, relation-state readers, contact rows, arm attachment/grasp readbacks and the public hooks, rather than replacing the owned predicates. Four positive cases cover every base/top combination, including both source-alternative choices. Negative cases cover initial/no-op and already-stacked scenes, final relations without action history, top-first order, sub-5-cm lift for either moved object, missing base support/contact, held top, pending release, other-arm hold/constraint, robot support, linear/angular instability, unknown native grasp state, repeated records without time credit, mismatched controller clock and interrupted stability. A further positive counterfactual confirms QA/source-success flags cannot change physical acceptance. These are synthetic logic results, not native simulator or robot successes.

Five stratified candidates were inspected on CPU for public objectives, all five private bindings, original scene patches and operation settings:

| Candidate | Original scene / room | Public role triple |
| --- | --- | --- |
| `category_ef213dc1f933ff069d88f39d` | office_cubicles_right / copy_room_0 | baking sheet / crock pot / egg |
| `category_efde35a356259504ddff8a5c` | Beechwood_0_garden / living_room_1 | outline tray / toy box / lemon |
| `category_533d99ee586c739b7874b11f` | Merom_0_garden / utility_room_0 | pizza box / wicker basket / lemon |
| `category_37f0ba122f86c3c0df483100` | Merom_0_int / utility_room_0 | outline tray / tackle box / orange |
| `category_ce85e6426bf14f8363521eec` | Beechwood_1_int / bedroom_0 | tray / bucket / lemon; source QA ambiguity retained |

The native-mutator AST check found zero control/state/pose mutation calls in the converter. All 77 original candidate directories were verified present. Detailed current results are in [cpu_validation.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json), [native_interface_validation.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/native_interface_validation.json), [sample_evidence.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json) and the current [batch_manifest.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/batch_manifest.json).

### Feedback closure, assumptions and unresolved evidence

The retained rule changes address the reviewed hidden candidate restriction, unused lift/history field, missing public order, incomplete release evidence, ambiguous stability clock and unjustified universal upright condition. There were no CPU conversion or fixture failures in this revision. Every current record carries structured uncertainty and `assumption_review_required`, `native_stack_execution_unverified` and `initial_qa_reference_integration_unverified`; the one original duplicate-direction record also retains `source_qa_direction_ambiguity`.

The historical assessment's 23/77 selected-view QA disagreements remain an explicit Main-owned initial-QA/reference-envelope integration issue. Preserving original QA does not establish that it has been executed from legitimate initial observations, and current physical success neither reads nor replaces that score. Shared QA/schema/controller/envelope files were not changed here. Existing task/action/controller limits and budgets remain unchanged.

Current-version native capture is **pending**, with zero new images viewed or submitted by this supervisor. Earlier images/receipts and the reported support/IK failures remain historical original-scene evidence only. This correction did not rerun them, audit world lifecycle, establish the availability of native APIs for every scene, prove gripper reach or demonstrate a physical stack. Unknown measurements conservatively reject an episode; their practical frequency and the proposed duration/tolerances require later native evidence. Main should review and publish these pending candidates, retain diagnostic failures, and integrate fixed-reference initial QA before any qualified task or model-success claim.

## Supplementary assessment comparison — version 2.1.0, 2026-10-15

Review fingerprint: `6ee12f28164b07d3f41e0333766b6b12e68316c4a1a6ee51387628d69668de72`. This is a read-only comparison of the assigned category. The plan was to compare the specification and historical assessment against current and archived code, inspect the complete source/candidate sets and previous definitions, then check live capture diagnostics and bounded CPU logic. Only this report appendix and the [review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/f8445a7f136def80b7e6/6ee12f28164b07d3f41e0333/result.json) are written. No batch regeneration, saved validation suite, simulator, GPU workload, robot action, model evaluation, capture submission, commit or agent delegation was run by this reviewer.

### Coverage and historical interpretation

The [implementation specification](/local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md) requires meaningful operations, private original QA, independent native outcomes and continued Review access for uncertain/failed items. The supplied [assessment](/local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:48) is supplementary historical evidence; its current hash matches `5bfdf6f3fb408bc7bd8aedd9873d6ddc5639dbff18237d20312ecc689275dd07`. All 24 previously identified relevant raw paths under `/scratch` were rechecked and remain unavailable. Its tabletop support, bounded movement and guarded IK claims therefore remain attributed historical observations, not new measurements here. The report distinguishes an excessive base-command rejection from the later bounded IK stop; neither is a grasp or stack success.

New read-only checks covered **77 sources, 77 current 2.1.0 candidates and all 77 preserved 2.0.0 candidates**. Every scene/task pair remains pending and unpatched, preserves the source hash and verbatim private answer/options/metadata, and binds all five originals. In-memory output from the current or archived converter matches all 154 saved definition pairs. All current candidates link to preserved previous candidates. The five saved batch snapshots contain two 2.0.0 batches and three 2.1.0 batches, each with 77 records; these are snapshots of two candidate sets, not five independent experiments. Private-answer, QA, pose and selected-identity counterfactuals leave current public text and the operation configuration unchanged for all 77 sources. Builder asset/model/scale/pose checks pass for all sources.

The assignment's `existing_task_files` is empty. Earlier tasks were nevertheless included: 77 catalogue scene imports plus their three frozen catalogue copies (308 definitions and 308 matching source JSON copies), four working diagnostic scenes plus their three frozen copies (16 definitions and 16 handoffs), all archived 2.0.0 artifacts, and assigned capture snapshots. Those four earlier diagnostics have no physical target, qualification or task handoff and have empty patches. The queue inventory contains **170 matching manifests representing 162 unique jobs**: 77 per converter version and eight earlier original diagnostics. All 170 source snapshots match the corpus and all scene snapshots remain pending. There are now 154 pending task YAML snapshots, including 77 current ones, matching their referenced task files. This supersedes the earlier review's zero-task-snapshot inventory; freezing a task still does not execute its evaluator.

### Comparison of the operation and terminal contract

| Earlier assessment or review claim | Current comparison |
| --- | --- |
| Reuse original sizes and hierarchy; prefer the previously tested tabletop. | Sizes, assets and hierarchy are preserved. Current candidates use unmodified source layouts, rather than that representative tabletop relocation. This is a documented L1 choice; the inaccessible historical trials do not justify a category-wide relocation or resizing rule. |
| A stable alternate candidate need not be the original QA answer. | Still correct for QA. Version 2.1.0 explicitly accepts either original base and either original top for the physical goal and keeps original direction answers separate. This is a deliberate semantic change from 2.0.0's hidden exact-identity predicate, not proof that all four combinations are physically feasible. |
| The old predicate omitted lift/history/order and had incomplete release and ambiguous stability checks. | These defects are substantially addressed in current code: native acquired grasp and attachment agreement, at least 0.05 m root and AABB-bottom lift, all-arm release/constraint/window clearance, stable B placement before selected-top acquisition, native support/contact, robot-contact exclusion and declared 0.5 s controller-time stability. The universal upright-axis requirement is removed. Old conclusions must remain attached to 2.0.0. |
| Original scoring depends on a fixed declared reference. | Still unresolved in the runtime path. The current builder selects another original orbital view for some sources; unchanged source packets fail that selected-view scorer in 23/77 cases. |

The proposed ordered full-stack operation is meaningful: it requires two real object manipulations, a measured intermediate placement and a released final support chain. The single recipe is sufficient to supply a substantive category task; multiple recipes are useful optional extensions. A separately scored B-on-base subgoal could diagnose the first manipulation, a top-placement continuation could follow a genuinely completed B placement, and a guarded trial-placement/physical-return recipe could preserve the source's experimentation mechanism. They must retain all originals and private QA, use actual robot motion for return, and distinguish subgoal completion from full-stack completion. None was generated here.

The [current converter/evaluator](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/action_order.py:238) reads native state without moving objects or running controls. It no longer selects success using source answers, construction flags or source-selected identities. Native `OnTop`, `Touching` and matching body contacts must support the selected chain, with stable external scene support for its base. Speeds are bounded by 0.02 m/s and 0.1 rad/s. Stability time accrues only from controller callbacks after environment steps, with controller dt checked against native action frequency; it does not certify every physics substep or long-term stability. Qualification requires all five originals to be settled, supported and free of robot contact with arms released, but does not establish reach, collision-free approaches or Agent sensor coverage.

The saved [CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json) reports 77 generated records, no conversion failures and 16.320 s generation; the saved [native-interface validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/native_interface_validation.json) reports 24 synthetic fixtures in 4.812 s. All 24 saved state files exist and agree with their expected and reported Boolean outcomes. These timings and fixture outcomes are inherited evidence, not suite reruns by this reviewer. They are CPU logic evidence, including the four accepted candidate combinations, with zero native simulation or robot execution.

**A remaining observation-boundary defect is reproducible on CPU.** In [record()](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/action_order.py:512), observing loss of B/base support resets stability timers but leaves `middle_placement` intact. A new memory-only test used the existing synthetic native-shaped readers: normal ordered control passed; after B earned its placement at controller step 8, a record-only support interruption correctly rejected that observation but retained the step-8 placement. Restoring support and immediately grasping the top at step 10 subsequently returned synthetic success without re-earning 0.5 s of B placement. Only one middle-placement event remained. The controller-step path invalidates this placement, so the gap specifically concerns interruptions observed at `record()` boundaries. No native occurrence or robot false positive is claimed. Main should review a shared invalidation rule that revokes stale B-placement/order credit at every measured boundary while keeping record calls unable to add time.

### Initial QA and observations

Static private/public separation is preserved; saving `private_qa.json` is not execution of auxiliary QA. The current category evaluation envelope seals the physical result to the task/source/episode boundary but does not invoke the original initial-QA scorer. New pure CPU scorer checks reproduce **77/77 compatible packets with their declared source QA, versus 23/77 failures with the builder-selected view**. These are packet/reference compatibility results, not model accuracy. The original Beechwood bedroom duplicate C direction remains warned; selected-view QA also creates a C-direction ambiguity in `office_large/private_office_2`. No source answer was corrected.

Main should integrate a declared, legitimate pre-action reference/observation contract and independently log the preserved initial QA result. Human scene/global review images and private object identities, poses, QA and scorer state must remain outside Agent inputs. Current static checks do not establish the entire runtime sensor/transport boundary.

### CaptureQueue evidence and limitations

Latest status snapshot: **2026-10-15 00:19:59 UTC / 08:19:59 Asia/Singapore**. All 77 current publication records have `review_submitted=true`.

| Evidence group | Current assigned-queue state |
| --- | --- |
| 2.1.0 candidates | 6 REVIEW_PENDING, 71 QUEUED |
| Preserved 2.0.0 candidates | 23 REVIEW_PENDING, 51 QUEUED, 1 RUNNING, 2 FAILED |
| Earlier original diagnostics | 8 REVIEW_PENDING at the full-inventory snapshot |

Four of the five current stratified samples have complete receipts; Beechwood living room remains queued after one failed initialization attempt. Thus the saved [sample_evidence.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json) snapshot (four queued, one running) and the implementation report's zero-current-capture statement are historical. At 00:18:44 UTC the six current complete receipts account for 54 existing PNGs and all record zero robot/model actions, no native qualification and no verified grasp. Their capture duration is **158.6 / 258.7 / 370.9 s** (minimum/median/maximum); cold initialization is 97.7 / 216.6 / 327.1 s and screenshot work 24.3 / 30.2 / 37.3 s. These exclude queue delay and say nothing about manipulation throughput or model quality. The 23 earlier conversion receipts account for 207 existing PNGs with the same diagnostic limits.

This reviewer actually viewed eight current images: scene/reference and head for Merom garden utility room, Merom interior utility room, office cubicles copy room and Beechwood bedroom. All show separated originals, without a completed stack. Merom garden crops the near base; Merom interior crops part of the near tray; Beechwood's bed occludes one tray and the near tray is cropped. The office reference shows the separated row clearly. All four head views miss the task objects and show arms with sky, walls or ceiling. This corroborates an initial framing/coverage concern; it does not prove that all body sensors or reachable viewpoints fail, and it does not justify an unmeasured layout repair.

Both remaining old native-process failures (restaurant Asian kitchen and restaurant diner kitchen) end with the background cabinet emitter's non-orthogonal-transform assertion, before task manipulation. Three other initialization failures, including current Beechwood living room, logged `No usable temporary directory found`; they subsequently returned to QUEUED under the existing workflow during this review. The exact temporary-directory failure cause is unverified. A later capacity read showed 36 GiB free on the root filesystem, 90 GiB on home and 6.3 GiB on local storage; it does not establish conditions at those failures. No retry, cleanup or process control was performed here.

The current frozen code copy matches the local converter, builder, scene adapter and category envelope. The task YAML is an input-hashed snapshot. The operation module is present and byte-identical in that code copy but is not individually listed in this scene capture's implementation hashes; the diagnostic receipt does not validate its execution. Native attachment/contact/support API coverage, physical reach, threshold calibration and any released stack remain unverified.

Retain both revisions, all earlier diagnostics, failures and pending candidates in Review and continue ordinary screenshot processing. Recommendations concern Main's future implementation review and qualified execution; **this comparison adds no publication, screenshot or approval gate**. No implementation, generated scene, source data, answer, approval, other category report, root research log or running workload was changed. No robot success or human approval is claimed.




## Supplementary assessment comparison — version 2.2.0, 2026-10-15

Review fingerprint: `16c31210d5d357bfee3a3357cbe0a7546b6dd629447f3df440c81ae5dc845780`. This appendix reviews only Action Sequencing / Action Order Inference. The read-only plan was to compare the implementation specification and supplementary assessment with the current converter, builder and source mechanism; inspect the complete batch and every discovered earlier definition; then check native predicates, initial QA and current CaptureQueue evidence. The assignment's `existing_task_files` list is empty. The detailed inventory and measured CPU checks are retained in [the review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/f8445a7f136def80b7e6/16c31210d5d357bfee3a3357/result.json).

**Assessment: the current compound operation is meaningful and substantially better specified than v2.0.0, but native success is unmeasured. The prior observation-boundary defect and initial-QA integration gap remain. Neither finding adds a Review or screenshot gate.** This reviewer did not regenerate batches, rerun saved validators, start simulation/GPU work, execute robot controls, submit captures, commit or delegate work.

### Historical claims, complete coverage and revision attribution

The [implementation specification](/local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md) remains the governing contract. The [supplementary assessment](</local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:48>) matches its supplied SHA-256 `5bfdf6f3fb408bc7bd8aedd9873d6ddc5639dbff18237d20312ecc689275dd07`. Its representative tabletop-support, guarded-IK and side-collision observations are inherited historical results; all 24 relevant historical raw-artifact references checked in the earlier review remain unavailable locally. The historical word “approved” does not establish approval of any current candidate. Offline planning likewise records no native execution.

New read-only checks covered all **77 sources across 34 scene models** and **308 distinct converter candidates**:

| Definition set | Distinct candidates | Actual mechanism |
| --- | ---: | --- |
| Archived v2.0.0 | 77 | Unpatched L1; source-selected private final support relations |
| Preserved v2.1.0 | 77 | Unpatched L1; ordered native grasp/lift/release operation accepting either original candidate |
| Isolated v2.2.0 placement batch | 77 | Pending L2 manipulation initialization policy |
| Published v2.2.0 revised-category batch | 77 | Pending L2 policy integrated into adapted definitions |

All 308 scene/task pairs are pending; all saved private answer/options/metadata packets match their sources. All 77 builder reconstructions retain the source models, scales and initial poses. Every v2.2.0 original variant remains unpatched; each adapted variant contains exactly one `prepare_category_interaction` patch. In-memory current conversion and private-answer/QA/pose/candidate-label counterfactuals across all 77 sources changed neither the public goal nor physical operation/policy. These checks wrote no generated definition.

Earlier tasks were also read as history: 77 current catalogue scene imports plus 77 in each of three frozen catalogues; four diagnostic scene candidates in the working catalogue and each frozen revision; and all **239 queue scene/source snapshots and 231 task snapshots**. Their source copies match the corpus. The eight source-only captures have no physical task snapshot. These imports and diagnostics do not supply an operational terminal evaluator, and the empty assignment list does not erase them.

Version attribution matters. Campaign-root [cpu_validation.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json) and [sample_evidence.json](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json) still describe **v2.1.0**, including the historical four-queued/one-running sample snapshot. Its saved generation time is 16.320 s and its 24 CPU fixtures took 4.812 s. The isolated [v2.2.0 CPU evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/cpu_validation.json) instead reports 15.614 s generation and 24 fixtures in 4.222 s; all 24 saved fixture state files in each suite agree with their expected/reported outcomes. The integrated [placement validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/revised_categories/f8445a7f136def80b7e6/placement_validation.json) reports 77 policies, preserved QA, no problems and no native execution in 0.317 s. Those are inherited timings, not suite reruns by this reviewer.

The earlier placement section's statements that the canonical batch was unchanged and nothing had been published describe the **isolated generation stage**. The canonical batch now points to the integrated v2.2.0 definitions and all 77 current candidates are queued. The isolated batch and all earlier revisions remain intact.

### Meaningful operations, recipes and pending L2 geometry

The [current converter](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/action_order.py:116) requires grasping and lifting B at least 5 cm, placing/releasing it on a chosen original base until stable, then grasping/lifting a chosen original top and releasing it on B. This supplies actual compound manipulation requirements; screenshots and CPU generation do not show their execution.

Accepting either original candidate is an explicit physical-task semantic difference from the historical assessment's exact-role recommendation and v2.0.0's hidden identity restriction. It is consistent with the present public contract and keeps source-direction QA separate. The generator's actual “wrong” modes include **39 too-large bases and 45 too-small tops**, alongside 38 too-small bases and 32 too-large tops. Its `check_on_top` checks XY-span inequalities after scripted placement, rather than native released contact, stability or unique feasibility. These labels do not justify silently rejecting an otherwise valid original physical stack, changing source answers or exposing selected identities.

Several separately named source-faithful recipes would be useful: the full ordered stack, B-on-base placement, second-stage top placement, and a bounded physical test/return recipe. They could isolate acquisition, support and sequencing failures. Physical return must use guarded controls rather than the source generator's pose reset, and subgoal success must remain distinct from full three-role completion. Current code supplies one compound recipe; its supported two-role path has no source instance among these 77. This review generated no variants.

Current L2 policy groups all five original bindings, targets B and both tops for manipulation, and declares floor-relative height screens of **0.35–1.10 m** and stance distance at most **0.85 m**. The shared initializer searches existing same-room supports and floor stances, preserves group geometry and background objects, restores rejected attempts, and translates the source reference camera with the group. The historical single-scene tabletop adapter can rotate and translate the row; the current search uses translation only. They are different algorithms, and the earlier support measurement does not qualify this new one.

These are initialization-only proposals. No current L2 candidate has a native receipt. Saved absolute Z/AABBs do not establish floor-relative reach, gripper approach geometry, IK or native support; fitting a common group on a support does not guarantee post-relocation visibility through unchanged background furniture. Native failures must remain explicit rather than triggering resizing, independent repacking or a success claim.

### Independent predicate and initial-QA boundaries

The operation predicate now reads native `OnTop`, `Touching`, matching contact bodies, stationary external scene support, object speeds, all-arm hold/attachment/release windows and the working gripper's open state. It requires acquisition/lift/release histories and middle placement before top acquisition, rejects selected-stack robot contact, and does not treat source construction flags, QA correctness or initialization acceptance as terminal evidence. The installed native grasp enum and assisted-grasp fields agree with the adapter at source-code level; runtime compatibility still needs native evidence.

Both middle placement and final stability require **0.5 s of controller callbacks after environment steps**, with speeds at most 0.02 m/s and 0.1 rad/s. Controller dt is checked against native action frequency. This is sampled controller-time evidence, not every physics substep or long-term equilibrium. The evaluator suffix remains byte-identical to frozen v2.1.0 (including its leading newline, SHA-256 `b14f11e75479558dbc8b21bce0599463e2672c2d7fc56240401efe614659931e`). All 77 current publication manifests now include the owned converter in their implementation hashes and freeze bytes matching the local module.

**The previous observation-boundary defect remains reproducible in current v2.2.0.** A new memory-only check using the saved native-shaped readers confirms that a normal ordered sequence succeeds synthetically. After B earns its middle placement at controller step 8, a `record()`-only loss of B/base support rejects that observation but retains `middle_placement`. Restoring support and grasping the top at step 10 subsequently yields synthetic success with only the original middle-placement event, without re-earning B's 0.5 s stable placement. A controller-step interruption correctly clears this credit. Main should review common invalidation at every measured boundary while keeping observation records unable to add time. This is a CPU logic result; no native occurrence or robot false positive is claimed.

**Initial QA is preserved but remains incompletely integrated.** Pure scorer checks again find **77/77 compatible declared-source packets versus 23/77 incompatibilities with the builder-selected view**. This is reference/packet compatibility, not model accuracy. The original Beechwood bedroom duplicate top direction remains; selected-view QA additionally creates a top-direction ambiguity in `office_large/private_office_2`. Saving `private_qa.json` and sealing physical evidence do not execute the original pre-action scorer. The task still declares `initial_visibility=none`; native mounted coverage and reach remain unqualified. Main should declare a legitimate initial reference/observation contract and log the original QA result independently. Private reference/global Review images, source identities/poses, answers and scorer state must remain outside Agent inputs. A common group/camera translation preserves directional geometry, but cannot by itself resolve the pre-existing selected-view mismatch or establish legitimate sensor evidence.

### CaptureQueue snapshot and retained decision

Status snapshot: **2026-10-15 04:05:08 UTC / 12:05:08 Asia/Singapore**.

| Assigned evidence group | REVIEW_PENDING | QUEUED | RUNNING | FAILED |
| --- | ---: | ---: | ---: | ---: |
| Published v2.2.0 | 0 | 77 | 0 | 0 |
| Preserved v2.1.0 | 19 | 54 | 3 | 1 |
| Preserved v2.0.0 | 24 | 50 | 0 | 3 |
| Earlier source-only diagnostics | 8 | 0 | 0 | 0 |

All **51 complete receipts**, covering **35 sources and 459 existing PNGs**, report zero robot/model actions, no native qualification and no verified grasp. Diagnostic capture times are 158.6 / 240.6 / 573.2 s for v2.1.0 and 174.6 / 277.2 / 599.6 s for v2.0.0 (minimum/median/maximum). They include initialization/screenshots, exclude queue delay and do not measure manipulation throughput or task quality.

The three saved RUNNING jobs are retained v2.1.0 cold-scene-loading captures: PID **1876726 on GPU 4**, **1865130 on GPU 2**, and **1869445 on GPU 7**. At 04:04:36 UTC, each actual process was **T (stopped)** with matching recorded start ticks. The formal campaign pause remains present. No completion estimate follows from stopped processes, and no resume/retry/termination was performed here.

Failure logs distinguish background scene initialization from manipulation. The restaurant Asian and diner diagnostics report a cabinet emitter's non-orthogonal transform. The hotel locker-room second attempt fails TorchInductor C++ compilation with **No space left on device**, after an earlier temporary-directory failure. These do not show failed grasps and do not establish current storage or driver conditions.

Six retained v2.1.0 images were actually viewed in this review: office copy-room reference/head, Beechwood bedroom diagonal/head, and Rs_garden kitchen reference/overhead. The office reference shows separated originals while its head misses them; the bedroom arrangement is low beside the bed with incomplete candidate visibility and a head view that misses the objects; Rs_garden's reference is nearly black while overhead shows a narrow, partly occluded fixture gap. These are **unpatched historical diagnostics**, not v2.2.0 repair images, all-sensor coverage tests or native stack demonstrations.

Retain every earlier/current candidate, failed diagnostic and pending item for Review. Main reviews the recommended predicate correction, initial-QA integration, native placement evidence and optional recipes in future revisions. This assessment adds **no Review, publication or capture gate** and does not alter the existing pause. Only this appendix and the specified review JSON were written; implementation, scenes, source data, answers, approvals, other category reports, root research log and workloads were left untouched. **No robot success or human approval is claimed.**

## Supplementary assessment comparison — runtime refresh, 2026-10-15

Review fingerprint: `4796cf7fe21a14cdf7fccca036bebc61e7fee9299f01bfd3971ede552cca47ac`. Scope: Action Sequencing / Action Order Inference only. The assignment lists no `existing_task_files`; discovered earlier definitions and all candidate revisions were nevertheless read as previous tasks. The read-only plan was to compare the specification and historical assessment with current source/code, audit complete batches and preserved definitions, reproduce only necessary CPU logic checks in memory, and inspect current CaptureQueue diagnostics. The detailed inventory is in [this review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/f8445a7f136def80b7e6/4796cf7fe21a14cdf7fccca0/result.json).

**The current compound operation is meaningful, but native execution remains unqualified. The middle-placement observation-boundary defect and initial-QA integration gap remain reproducible. Historical screenshot availability has changed. These findings add no Review, publication or screenshot gate.**

### Historical assessment, coverage and provenance

The [implementation specification](/local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md) governs this comparison. The [supplementary assessment](</local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:48>) matches its supplied SHA-256. Its representative unchanged-size tabletop support, guarded IK stop before grasp and rejected side placements remain historical observations. All 24 checked historical raw-artifact references remain unavailable locally. Local initialization/planning summaries explicitly record no verified manipulation; offline planning is not native execution. Historical “approved” wording establishes no approval of these candidates.

Read-only checks covered **12 manifests, 77 sources across 34 scene models, and 385 distinct candidate IDs**:

| Candidate generation | Candidates | Task mechanism |
| --- | ---: | --- |
| v2.0.0 | 77 | Unpatched L1; private source-selected final support relations |
| v2.1.0 | 77 | Unpatched L1; ordered grasp/lift/release, either original candidate |
| Isolated v2.2.0 | 77 | Pending L2 interaction initialization |
| Integrated v2.2.0 | 77 | Adapted definitions with one interaction patch |
| Runtime-refreshed v2.2.0 | 77 | Same integrated scene/task inputs, new frozen runtime identity |

The refreshed candidates share their parents' definitions: **616 distinct scene/task files** were read, representing 308 definition pairs. Every referenced definition remains pending. All 385 source digests and private answer/options/metadata packets match their sources; every record's predicate matches its task. The 231 refreshed source/scene/task input hashes match current bytes. The builder reconstructs all 77 sources with their original models, scales, poses and orientations. In-memory conversion counterfactuals over all sources leave public goals, operation configuration and interaction policy unchanged when private answers, QA, poses and candidate labels change.

Previous tasks also include **308 catalogue scene imports** across the working tree and three frozen catalogues, plus **16 diagnostic scene definitions** across those same four locations. They remain pending, without an operational target. All **316 assigned queue scene/source snapshots and 308 task snapshots** were read; the eight source-only captures have no task. Snapshot sources match the corpus. These counts overlap by design and are not additional independent trials.

The campaign-root [CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json) and [sample evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json) still describe v2.1.0. Its five sample states, four queued and one running, are historical. The saved v2.1.0 timings are 16.320 s generation and 4.812 s for 24 synthetic fixtures; [isolated v2.2.0 evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/placement_review/action_order/cpu_validation.json) reports 15.614 s and 4.222 s respectively. Both sets of 24 saved fixture outcomes match their expected values. Integrated placement validation reports 77 policies and no native execution. These suites were read, not rerun: their runners write generated artifacts.

The [runtime refresh result](/home/zhouxinyang/esi_category_conversion/20261014_v2/runtime_capture_refresh/category_results/f8445a7f136def80b7e6.json) records 77 published cards/submissions, preserved design inputs and no physical qualification. Every current manifest hashes the owned converter; its frozen bytes match the local converter. The frozen builder, interaction initializer and evaluation envelope also match. This establishes provenance, not successful camera repair or manipulation.

### Actual operations and useful recipes

The [converter](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/action_order.py:116) requires native grasp, at least 5 cm lift, release and stable B-on-base placement before acquiring/lifting/releasing a chosen top on B. This is a meaningful compound operation. Accepting either original base/top is an explicit physical-task semantic difference from v2.0.0 and the assessment's exact-role recommendation; original direction-choice QA remains separate.

The source generator actually labels 39 bases “too large” and 45 tops “too small,” alongside 38 too-small bases and 32 too-large tops. Its scripted placement and XY-span fit check do not establish unique released native feasibility; comments describing every wrong candidate as incapable are overgeneralizations. These labels cannot replace native support evidence or justify silently rejecting an otherwise valid stack.

Several separately named recipes would be useful: the full ordered stack, B-on-base placement, top placement after a declared valid middle stage, and bounded physical test/return. They could isolate grasp, support and sequence failures while preserving original assets and private QA. Return must use guarded controls rather than source pose resets. Subgoal results must remain distinct from full-stack completion. Current code supplies one compound recipe; all 77 sources have three roles, so two-role support is only an interface capability.

The L2 initializer proposes an unchanged-size group translation to an existing same-room support and a measured floor stance, with floor-relative height limits 0.35–1.10 m and stance distance at most 0.85 m. It translates the source camera with the group and retains mounted sensor locations. This translation-only search differs from the historical single-scene planar rigid transform. Neither its screens nor historical table support establish IK, grasp affordances, complete-group settled support or post-relocation observation coverage. No assigned L2 capture receipt currently measures this repair.

### Terminal predicate and initial QA

The predicate independently reads native support/contact, external scene support, velocities, all-arm grasp/attachment/release states and working-gripper opening. It requires ordered acquisition/lift/release history, released robot-clear support, and at least 0.5 s at the declared low-motion limits. Time accrues from controller callbacks after environment steps, not observation counts; it does not certify every physics substep or long-term equilibrium. Source QA, construction flags, initialization acceptance and screenshots do not establish success.

**A current memory-only check reproduces inconsistent invalidation.** Normal synthetic execution passes. After B earns middle placement at step 8, a support loss seen only by `record()` rejects that observation but retains `middle_placement`. Restoring support, taking one controller step and acquiring the top at step 11 subsequently permits synthetic success, without re-earning B's stable placement interval. The equivalent interruption through `sample_step()` clears that credit and fails. Main should review common structural invalidation at both boundaries while keeping observations unable to add elapsed time. This is a CPU logic result, not an observed native false positive.

**Private storage is preserved, but initial QA execution/reference integration is incomplete.** Pure scorer checks again yield 77/77 compatible declared-source packets and **23/77 incompatible builder-selected-view packets**. These are reference/packet checks, not model accuracy. The Beechwood bedroom duplicate top direction persists; the selected office-large private-office view introduces another duplicate. The physical envelope does not run the original pre-action scorer, and `initial_visibility=none` does not qualify legitimate sensor coverage. Main should declare the pre-action reference contract and independently log original QA, without changing answers or exposing private source identities/poses, scorer state or human Review images to the Agent.

### Current captures, unavailable images and retained decision

Status snapshot: **2026-10-15 05:19:20 UTC / 13:19:20 Asia/Singapore**.

| Assigned queue group | REVIEW_PENDING | QUEUED | CANCELLED | FAILED | RUNNING |
| --- | ---: | ---: | ---: | ---: | ---: |
| Runtime-refreshed v2.2.0 | 0 | 77 | 0 | 0 | 0 |
| Earlier integrated v2.2.0 | 0 | 77 | 0 | 0 | 0 |
| v2.1.0 | 19 | 54 | 3 | 1 | 0 |
| v2.0.0 | 24 | 50 | 0 | 3 | 0 |
| Source-only diagnostics | 8 | 0 | 0 | 0 | 0 |

The queue control now records RUNNING, superseding earlier pause statements. Existing runtime policy permits the refreshed code only; older queued jobs remain historical inputs. The previously stopped PIDs 1876726, 1865130 and 1869445 are absent at the current process check; their three jobs are already CANCELLED by the existing workflow. This reviewer changed no native process or queue state.

There are **51 saved complete receipts for 35 sources**, all reporting zero robot/model actions, no native qualification and no verified grasp. Their **459 referenced PNG paths are now unavailable**. Two representative equivalent durable paths and all 18 historical image references in the five-sample file are also unavailable. Earlier reports' existing-image counts and visual observations are historical evidence only. No image was newly viewed here. A backup location and the cause of unavailability are not established; this does not prove global deletion. A receipt's `complete=true` is not current readable-image evidence.

Saved diagnostic capture duration ranges remain 158.6–573.2 s for v2.1.0 and 174.6–599.6 s for v2.0.0; medians are 240.6 and 277.2 s. These inherited initialization/screenshot costs exclude queue delay and do not measure robot throughput. Four saved failures remain attributable to background cabinet-emitter transform assertions or hotel TorchInductor compilation/storage failure, preceded by a temporary-directory failure. They do not demonstrate failed grasps or current storage/driver conditions.

Retain all completed, pending, cancelled, failed and source-only records for Review, and allow ordinary current screenshots to proceed. Main should review the predicate correction, independent initial QA, recoverable image links/provenance, native placement qualification and optional recipes. **No recommendation here blocks Review or captures.** This review only appends this English comparison and writes its specified JSON; implementation, generated scenes, source data, answers, approvals, other category reports, root research log and pre-existing workloads remain unchanged. **No robot success or human approval is claimed.**



## Supplementary assessment comparison — selected native sample, 2026-10-15

Review fingerprint: `bbd163cef965f60f412048a5becba2e5358451e07af3de442a9a2fd7e12bdf87`. Scope: Action Sequencing / Action Order Inference only. The read-only plan was to compare the implementation specification, supplementary assessment and source/builder against every saved candidate generation; inspect the complete batch, CPU validation, sample evidence and previous definitions; then check current CaptureQueue receipts and only the necessary CPU predicate/reference logic. The assignment lists no `existing_task_files`; discovered earlier tasks were still included. Detailed inventories and checks are in [this review result](/home/zhouxinyang/esi_category_conversion/20261014_v2/assessment_reviews/f8445a7f136def80b7e6/bbd163cef965f60f412048a5/result.json).

**The compound operation is meaningful, but its native execution remains unqualified. The newly completed representative capture records rejected L2 initialization, while the evaluator boundary defect and initial-QA reference gap remain reproducible. This review adds no Review, publication or screenshot gate.**

### Assessment claims, current code and complete history

The [implementation specification](/local_home2/zhouxinyang/models/Show-Harness-Explore/esi_category_parallel_conversion_spec_v2.md) requires actual operations, independent native terminal evidence, private original QA and Review access for failures. The [supplementary assessment](</local_home2/zhouxinyang/models/Show-Harness-Explore/report(1).md:48>) matches the supplied SHA-256. Its unchanged-size five-object tabletop support, front IK stop before grasp and rejected side placements are historical representative observations. All 24 checked relevant raw-artifact paths remain unavailable locally. Local initialization/planning summaries retain these claims and explicitly deny verified manipulation; offline planning is not native execution. Historical “approved” wording does not establish approval of current candidates.

Fourteen complete current/historical manifests were read, each covering all **77 sources across 34 scene models** with 77 CPU-generated records. They reference **386 distinct candidate IDs**: 77 v2.0.0, 77 v2.1.0, 77 isolated v2.2.0, 77 integrated v2.2.0, 77 runtime-refreshed v2.2.0 and one newly frozen selected-sample identity. Repeated manifests and refreshed identities are not independent physical trials. The selected-sample manifest changes one runtime identity while preserving its parent scene/task inputs.

All **616 distinct scene/task definition files** referenced by these IDs were read. All 386 record references have pending scene/task definitions, unpatched original variants, preserved source hashes and verbatim private answer/options/metadata packets; every task predicate agrees with its record. The builder reproduces original models, scales, poses and orientations for 77/77 sources. Current in-memory private-answer/QA/pose/construction-flag counterfactuals preserve public goals, operation configuration and interaction policy for 77/77 sources. This is storage/projection evidence, not a complete Agent transport audit.

Previous tasks additionally include 308 catalogue scene imports and 16 diagnostic scene definitions across the working tree and three populated frozen catalogues. They remain pending, without an operational target. All 317 assigned queue scene/source snapshots and 309 task snapshots were read; eight source-only diagnostics have no task. All 317 snapshot sources match the corpus. These counts overlap the candidate history.

The [campaign CPU validation](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/cpu_validation.json) and [sample evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/categories/f8445a7f136def80b7e6/sample_evidence.json) still describe v2.1.0; their four queued/one running sample states are historical. Saved generation/fixture times are 16.320/4.812 s for v2.1.0 and 15.614/4.222 s for isolated v2.2.0. Both saved sets of 24 synthetic fixture states agree with their expected/reported outcomes. Integrated placement validation records 77 policies, zero reported problems and no native execution in 0.317 s. These runners were read, not rerun, because they write generated artifacts. New CPU checks used memory-only output sinks and launched no simulator.

### Actual operations, semantic differences and useful recipes

The [current converter](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/action_order.py:116) asks for a real B grasp, at least 5 cm lift, release and stable B-on-base placement before grasping/lifting/releasing a top on B. This supplies a substantive compound operation. Its explicit acceptance of either original base/top differs from v2.0.0 and the assessment's exact source-role recommendation. It resolves the earlier hidden candidate restriction for the physical goal; it does not preserve exact source-choice semantics or make an alternative stack a correct original QA answer.

Source generation labels 39 wrong bases “too large” and 45 wrong tops “too small,” alongside 38 too-small bases and 32 too-large tops. Its scripted pose placement/reset and XY-span fit predicate do not prove unique released native feasibility. All 77 construction `success` flags are true, but none establishes robot success.

Several separately named source-faithful recipes would be useful: full ordered stacking, B-on-base placement, top-placement continuation after a genuinely completed middle stage, and bounded trial placement followed by physical return/retry. These isolate grasp, support and sequence failures while retaining originals and private QA. Return requires guarded robot actions; source pose resets cannot count. Subgoal success must remain distinct from full-stack completion. Current code supplies one compound recipe; all 77 sources have three roles, so its two-role branch is an interface capability rather than a second measured corpus recipe. This review creates no variants.

### Independent predicate and initial-QA boundaries

The native predicate independently reads `OnTop`, `Touching`, matching contact bodies, stationary external base support, velocities, all-arm grasp/attachment/release states and working-gripper opening. It requires ordered acquired-grasp/lift/release histories, robot-clear released support and 0.5 s stability at 0.02 m/s and 0.1 rad/s limits. Controller callbacks after environment steps advance time; observation records do not. These thresholds are declared assumptions, not calibrated native feasibility or long-term equilibrium. Source QA, construction flags, initialization acceptance and screenshots cannot satisfy this predicate.

**The observation-boundary defect persists in current and selected frozen evaluator code.** A new memory-only reproduction earns B placement at controller step 8. Losing B/base support only at `record()` rejects that observation but retains `middle_placement`. After support returns and one controller step, top acquisition at step 11 permits synthetic final success without re-earning B's stable interval. The equivalent `sample_step()` interruption revokes placement credit and fails. Normal and alternate-candidate sequences pass synthetically; no-op, top-first and repeated-record clock tests fail as intended. Main should review common structural invalidation at every measurement boundary while preserving zero observation-time credit. This is a CPU logic result, not an observed robot false positive.

**Initial QA remains preserved but incompletely integrated.** Pure scorer checks yield 77/77 compatible declared-source packets and **23/77 incompatible builder-selected-view packets**. These measure reference/packet compatibility, not model accuracy. The original Beechwood bedroom duplicate top direction remains; selected-view QA introduces another in office_large/private_office_2. The physical task envelope does not execute the original pre-action scorer, and `initial_visibility=none` does not qualify legitimate mounted sensor coverage. Main should declare the initial reference/observation contract and separately log original QA. Private answers, identities/poses, scorer state and human Review images must remain outside Agent inputs. Translating the group and reference camera together cannot fix the pre-existing selected-view mismatch.

### New native evidence and current CaptureQueue diagnostics

Queue snapshot: **2026-10-15 06:31:26 UTC / 14:31:26 Asia/Singapore**.

| Assigned group | REVIEW_PENDING | QUEUED | CANCELLED | FAILED | RUNNING |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected sample v2.2.0 | 1 | 0 | 0 | 0 | 0 |
| Runtime-refreshed v2.2.0 | 0 | 77 | 0 | 0 | 0 |
| Earlier integrated v2.2.0 | 0 | 77 | 0 | 0 | 0 |
| v2.1.0 | 19 | 54 | 3 | 1 | 0 |
| v2.0.0 | 24 | 50 | 0 | 3 | 0 |
| Source-only diagnostics | 8 | 0 | 0 | 0 | 0 |

These are 317 unique jobs. The queue control is RUNNING, with the existing `ONE_PER_CATEGORY_AWAITING_REVIEW` sample policy and runtime restriction to `working_2fc97c3d67769295ed5e7bf6e846fa53`. Historical queued states do not imply eligibility for dispatch. This review changes neither the existing sample restriction nor any queue state; its findings impose no additional restriction. No assigned capture job was RUNNING at the snapshot. A later process spot check found worker2's supervisor alive and the receipt's former native PID absent; the receipt does not establish a currently retained world.

The selected candidate `category_c22cbe7f140d42d9e483b990`, Beechwood_0_garden/dining_room_0, now has [a complete native receipt](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/preview_complete.json) and nine readable PNGs. This supersedes earlier statements that this category has no readable current L2 images. Capture took **217.587 s**, including 165.668 s scene initialization and 23.983 s screenshot work; this is not manipulation latency or throughput. The receipt records zero robot actions/model calls, no native qualification and no verified grasp.

Crucially, its interaction setup is **`accepted=false`, `relocated=false`**, with reason `ValueError: No existing same-room support fits the unchanged group at a suitable height.` It records 39 approximate floor cells but no placement attempts. B and both top candidates fail the floor-relative height screen, with AABB bottoms about -0.000003 m and centres 0.029–0.066 m above the measured floor. The original group spans approximately **1.631 x 0.553 m** in XY. These are measured initialization diagnostics, not a failed grasp or proof that every possible placement is infeasible.

Three images were actually viewed: [overhead](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/global_overhead.png) shows the separated originals on the floor; [source reference](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/scene.png) shows that floor row beside dining furniture with an endpoint cropped; [head](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/local_capture_20261014/queue_workers_initialized/jobs/capture_7b0f44e07d03b5435258317ccd14a50c/attempts/01/head.png) shows the arms and sky without task objects. This head frame does not establish all mounted-camera coverage or sensor reachability.

The historical tabletop success concerns **the same source scene**, but a different initialization protocol: a declared common planar rigid transform onto coffee_table_dnsjnv_0, versus the present translation-only search bounded to 16 candidate supports, three points and two surface layers per point. The current rejection does not refute the historical support claim, and that claim does not validate this initializer. With raw historical measurements unavailable and no per-support rejection trace here, the exact cause cannot be assigned to size, stance screening, sampled points or contact checks. Main should review diagnostic detail and source-faithful common-transform alternatives before inferring asset resizing or new geometry is necessary.

The selected frozen converter, builder and evaluation envelope match current local bytes. The local shared initializer now adds baseline restoration after rejected proposals, absent in this frozen capture. That later change was not tested here and must not be credited with resolving the measured rejection, which occurs before placement attempts.

There are **52 saved complete receipts for 35 sources**; all report zero robot/model actions, no native qualification and no verified grasp. Of 468 referenced PNG paths, nine current-sample paths are readable; 459 historical paths and all 18 historical sample-evidence image references remain unavailable. No backup or global deletion claim is established. Four saved FAILED jobs show background cabinet-emitter transform assertions or TorchInductor compilation/storage failure, preceded by a temporary-directory failure; they are not failed grasp trials and do not diagnose present storage/driver health.

Retain all completed, failed, cancelled, pending and source-only records for Review. Main reviews the predicate correction, original-QA integration, current initialization failure diagnostics and optional recipes in future revisions. **No robot success or human approval is claimed.** Only this English appendix and the specified review JSON are written; implementation, generated scenes, source data, answers, approvals, other category reports, root research log and workloads are untouched by this review.

## Annotation repair: complete subclass audit and support-aware qualification, 2026-10-15

Plan and evidence: audit every original source plus the stable human-rendered Review batch, current scheduler batch when present, and feedback-referenced history; isolate source/QA contradictions from shared support/spawn failures; make only category-wide repairs and verify native-shaped predicates before preparing CPU candidates. This subclass contains **77 sources** and **77 tasks in the stable Review batch**. The audit read **77 distinct existing task files** across relevant revisions; all inspected task source references, public/private fields, empty original variants and pending status passed, and stored private QA matches the original JSON. [Detailed audit](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/f8445a7f136def80b7e6_audit.json) · [Stable Review batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/interaction_repair_20261015/sample_batches/f8445a7f136def80b7e6/batch_manifest.json). Native failure evidence is representative; no unrendered source is claimed to have passed or failed native support.

The reported floor props/off-map start came from a native support-search rejection before placement, not a measured failure of the support-order operation. The source-derived rule still preserves the complete original candidate row and transforms its source reference camera together with it. The local explicit interaction policy is embedded in both initialization parameters and the returned policy, so it needed its own `allow_fallback_table=True`; the shared `policy_for` default alone could not authorize this category. Converter version `2.3.0` retains `measured_surfaces` and tries original furniture before the user-authorized simple fallback support. Assets and scale ratios are unchanged; no per-instance coordinates are introduced.

The native operation scorer is retained. It already discovers current external supports through actual native contact plus OnTop/Touching, measures held lifts from the initialized episode, and requires real acquisition/release, support order and stability. It does not compare the adapted row against a raw floor coordinate or earn operation credit at initialization. A new native-shaped test accepts a measured table-supported initial row, then rejects it when a required native contact is removed. Main owns independent safe diagnostic spawning and actual support/floor qualification. The stale scheduler batch containing 77 schema failures is retained as history; the stable human-rendered Review batch contains 77 pending tasks and is the authoritative comparison baseline.

Validation: the new category repair tests completed with **29 passed in 1.52 s** using the behavior Python, with numerical-library thread counts bounded to two. Tests cover the changed categories' native-shaped qualification boundaries, six-category public/private preservation, original variants, deferred CPU stance decisions and callable legacy contact recipes. No native simulation, GPU workload, queue mutation or publication was performed by this repair agent. Prior failed layouts and source contradictions remain evidence. Main will render exactly one corrected representative per modified subclass first; other queued jobs are unchanged, and obsolete Review/screenshots are removed only after replacement while source, QA and current annotation evidence are retained. No benchmark is run by this repair. Isolated CPU batch results will be recorded below after shared schema/profile stabilization.

### Isolated CPU repair batch and exact annotated representative

The completed [repair batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/categories/f8445a7f136def80b7e6/batch_manifest.json) contains **77 source records and 77 pending task definitions**, with status counts `{"generated": 77}`. CPU conversion took **47.8125 s**, with seed 0, configuration `{}` and numerical-library threads bounded to two; this is conversion cost, not operation latency. **0** records carry `source_start_native_measurement_required`. Full output verification checked all source hashes and original private QA, pending schemas, original variants and the category's measured-support/fallback policy. Across the six owned subclasses, the final result is 665 source records, 661 pending tasks, 31 deferred CPU stance decisions and four retained source diagnostics. [Verification](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/post_conversion_verification.json).

The deterministic corrected representative is **`category_364455f495bb8955f1a1399d`**, matching the exact annotated source `Action Sequencing/Action Order Inference/Beechwood_0_garden/dining_room_0/Beechwood_0_garden__dining_room_0.json`; no easier alternative was selected. Its [task](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/categories/f8445a7f136def80b7e6/candidates/category_364455f495bb8955f1a1399d/task.yaml) and [scene](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/categories/f8445a7f136def80b7e6/candidates/category_364455f495bb8955f1a1399d/scene.yaml) are prepared for Main's one-representative capture only. The [representative manifest](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/representatives.json) records all six selections. These files were generated through converter rules in the isolated HOME output, with no canonical YAML edit, publication or queue submission. Native placement, reach, IK, release and grasp remain unverified for this revision.

The broader shared regression run measured **98 passed and 12 failed in 46.13 s** after the new shared policies landed; Main reproduced the failures and owns their correction. They concern new safe diagnostic spawning/restoration and obsolete profile fixtures, including two nonrobot settling disturbances that must be rolled back. The owned category test file passed all 29 checks. [Validation history](/home/zhouxinyang/esi_category_conversion/20261014_v2/annotation_repair_20261015/manipulation/validation_history.json) distinguishes those measured outcomes; local test or CPU generation success is not a native qualification result.


## 2026-10-15 Approved accessible-support generator audit

Plan: apply the user-approved access repair at generator-rule level after reading the final v2 specification, current converter/source builder and reports. The historical assessment is supplementary and nonbinding. Audit every source and the latest human-rendered batch referenced by category_revisions.json; preserve source QA, old failures and approved work. No source JSON, published task, queue, screenshot or Native process was modified.

Coverage: **77 original sources**, **77 existing task definitions read**. Independent original candidate groups; preserve final ordered grasp/lift/release/support predicate. New hook defers to one final initializer; legacy rigid trials use the guarded trial API. [Whole-class evidence](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f8445a7f136def80b7e6/audit.json) and [eleven-class summary](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/audit_summary.json) preserve every inspected source identity and failure.

Retained converter version `2.4.0`. The isolated [CPU batch](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f8445a7f136def80b7e6/batch_manifest.json) contains 77 records, 77 pending task definitions and status counts `{"generated": 77}`; conversion took 25.652 s with seed 0, configuration `{}` and numerical-library threads bounded to two. 0 records defer an inconclusive CPU stance screen to Native measurement without inventing a robot pose. Every generated source hash and complete private QA packet matches the original; every prior task file remains byte-identical.

Prepared representative: `category_322070bc2fb29cbe3fdf9ec0` for `Action Sequencing/Action Order Inference/Beechwood_0_garden/dining_room_0/Beechwood_0_garden__dining_room_0.json`. Same source as the authoritative prior sample. [Task](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f8445a7f136def80b7e6/candidates/category_322070bc2fb29cbe3fdf9ec0/task.yaml) · [Scene](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/group_a/f8445a7f136def80b7e6/candidates/category_322070bc2fb29cbe3fdf9ec0/scene.yaml). The representative is prepared only; this agent did not publish or enqueue it.

All base, middle and top candidates participate in initial height/support screening. This does not add a base-grasp requirement to the physical operation. The new owned hook validates identity/dynamics and defers placement to exactly one final shared initializer. Historical rigid/rotated proposals remain callable through `initialize_interaction_trial(session, bindings, policy)` inside the guarded initialization context.

Validation: `tests/test_accessible_generators_a.py` completed **20 passed in 1.53 s**; the focused annotation, tabletop-relation and goal-only compatibility suites completed **78 passed in 1.76 s** using behavior Python. These CPU fixtures check private-QA boundaries, generated schemas, one-time initialization, adaptation policies and native-shaped support/cover predicates. Native surface/contact acceptance, collision-free floor access, cloth/scaffold integrity, mounted views, IK, grasp/release and physical execution remain unverified for the new runtime. Shared placement and subsequent one-representative-per-changed-class rendering are owned by Main.

## 2026-10-15 Native representative outcome

Main's [publication journal](/home/zhouxinyang/esi_category_conversion/20261014_v2/review_updates/ed22ddbd33445d4a74078171/update_result.json) selected runtime representative `category_c429506d3b2561ee65fcdc99` from the same Beechwood garden `dining_room_0` source. The [current metadata](/home/zhouxinyang/esi_category_conversion/20261014_v2/accessibility_repair_20261015/native_results.json), [job status](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_99d5a9e70f8717d265740c3242df0612/status.json) and [completion receipt](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_99d5a9e70f8717d265740c3242df0612/attempts/01/preview_complete.json) record `REVIEW_PENDING`, nine declared images, `interaction_setup.accepted=true` and `relocated=true`.

All five singleton candidate groups obtained accepted initialization support on existing furniture: the middle, wrong base and correct top used `straight_chair_dmcixv_0` at approximately 0.4651 m; the correct base and wrong top used `coffee_table_dnsjnv_0` at approximately 0.7439 m. The receipt records native task-support contact, a floor-supported robot start and no unexpected robot contacts for these trials. No new support asset was required. This resolves the sampled floor-start layout failure while retaining the meaningful final ordered-support operation.

This is initialization evidence only. The receipt also records `native_qualification_run=false`, `eligible_final_layout=false`, `physical_grasp_verified=false` and zero robot actions. `initial_base_approach_required`, `conservative_stance_radius_exceeded` and `native_reach_and_route_unverified` remain explicit warnings; the source-reference-layout and physical-candidate-choice review warnings also remain visible. No autonomous acquisition, lift, release, ordered stack, route or IK success is established. The [Native log](/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/capture_99d5a9e70f8717d265740c3242df0612/attempts/01/native.log) is retained. Current CPU coverage remains **77 source records and 77 generated pending tasks**. This update reads Main's existing metadata and logs; it runs no experiment and does not inspect or validate images.


## Safe composition and review checks — 2026-10-16

The source-aware CPU composition policy retains 77 current generated records as 77 episode groups. Archived repeats and failures are excluded from automatic composition and retained separately. The implementation is [composition_manipulation.py](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/esi_converters/composition_manipulation.py). Singleton worlds and original physical predicates remain unchanged. Composite scoring credits completed original goals at their own validated checkpoints; native history, causal order and safety thresholds are not weakened. Human review checklists are separate from automatic success predicates. No new native experiment or render was run for this change.

Required review checks:

- Check the real robot spawn, upright wheel contact, obstacle clearance and a usable route to every required object.
- Inspect low and high supports, native support contact and unchanged asset scales; height screens do not establish IK or graspability.
- Check mounted head and wrist sensor coverage through robot observations; human overview images are not agent evidence.
- Retain original QA and reference-view evidence separately; obtain initial comparison answers before interventions change the evidence.
- Locate both base candidates, the middle object and every top candidate; none may be hidden beneath another candidate or pre-stacked at initialization.
- Check separately supported candidate footprints, a usable grasp approach, and transport between supports without resizing or teleporting.
- Verify middle grasp, 5 cm lift, release and stable supporting placement before acquiring the selected top; require released final stack stability.

Latest feedback on category_fc8f4d390da309826dfeee92 asks where the chopping block is. The actual preview receipt for capture_00a901a1e5d885b9623714eea9589b54 shows both source chopping_block/ekowsj actors present at their original scales. The first was placed on pedestal_table_gikdir_0 about 3.51 m from the gift box; the other shared the gift box footprint on pedestal_table_gikdir_1 and was hidden beneath it. Existing diagonal/overhead/head/source images were inspected without rendering. The receipt accepted initialization but ran no native qualification or robot action. A pending feedback derivative now declares a 3 cm independent-candidate XY gap and separate support/contact evidence, and asks the agent to inspect all candidates. Shared initialization must enforce that clearance; sensor coverage remains a review obligation. No world coordinates, asset sizes, original QA or stacking predicate are changed by the category repair. The original approved/pending artifacts are not overwritten.

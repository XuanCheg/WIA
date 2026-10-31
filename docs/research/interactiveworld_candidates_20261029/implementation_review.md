# InteractiveWorld candidates and expanded WIA review

Baseline commit: `b7101de`. InteractiveWorld source: `40957c7243aa38ea8550d929dd3f10e99d8733d6`.
This working-tree implementation has not been committed or pushed. Candidate
admission remains pending. The user's later instructions authorize repairs to
working baseline definitions and supersede the original robot-rollout validation
requirement: this iteration executes no policy, model, or reference robot plan.

## Scope and scientific contract

The working review covers 61 baseline selections and 19 variants of nine new
mechanisms. These are 80 configurations, not 80 independent mechanisms. The
[implementation plan](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/implementation_plan.md)
records source scenes, preserved structure, approved changes, endpoints and paired
factors. All original source/task provenance is retained in derived definitions.

| Mechanism | Variants and implemented decision | Evidence reviewed | Verification boundary |
|---|---|---|---|
| H1 | Retain a rigid package with an acceptance mark or replace one with a rejection mark | Actual attached mark; corrected robot-relative left package; authored flip preview | No cutting, food deterioration, robot flip or grasp claim |
| H2 | Preserve readiness after delivery, with or without a late obstruction | Native staged endpoints and event displacement; delivery preservation | Diagnostic endpoint staging bypasses the runtime trigger sequence; CPU ordering tests and native event geometry are separate evidence |
| H3 | Deficit, already complete, and completed stock followed by external removal | Current stock reduction and actual before/after history frames | No extra work required for the already-satisfied branch; no rollout result |
| H4 | Clear only useful obstacles; minimize relocation count among valid placements | Full target footprint, fixed anchors, legal storage and action-cost reducer | Minimality is a secondary objective, not an unsupported hard constraint |
| H5 | Preferred station eligibility changes with a protected resident | Native resident placement and procedural-rule reducer | Occupied primary station is procedurally ineligible; geometric impossibility is not claimed |
| M1 | Identical cleaning tools are segregated according to observed use; used identity swaps | Twenty actual head-camera frames per branch, native contact and preservation | Rigid tool surrogate; event qualification, not contamination or cloth physics |
| M2 | Indivisible quantities 3,2,2,1 satisfy orders 4/3 versus 5/3 | Attached QTY text, neutral shelf labels, all feasible assignments | Quantity is an explicit package attribute; units are not divided or inferred from package size |
| M3 | Contents determine suitable storage; contents swap between matching fridges | Closed interiors, native residents and authored open-door geometry | Explicit storage rule; cooling, food safety and robot opening are unmeasured |
| M4 | Schedule three jobs through exclusive timed stations and physical handoffs | Occupancy/time reducers and bounded schedule enumeration | Simulation seconds; model/API waits pause the world; no thermal process |

H2 provides primary proposed AP-4 coverage (Closed-loop Recovery & Adaptation),
while H3 provides related net-progress revision. H1 concerns evidence-triggered
replacement; H4 selective rearrangement; H5 admissible alternatives. These differ
from ordinary static placement despite sharing robot controls. M1 requires actual
observation history, M2 couples allocation choices, M3 uses content-dependent
storage, and M4 couples resource occupancy and elapsed simulation time. No new
taxonomy assignment or claim of measured closed-loop policy performance is made.
Future AP-4 evaluation should compare feedback-enabled and feedback-limited
conditions while matching geometry, event, budgets and policy.

Estimated physical difficulty is qualitative, not a measured success rate:

| Prototype | Difficulty and remaining physical concern |
|---|---|
| H1 | High: reorienting a package to inspect its underside, then stable delivery |
| H2 | Moderate: returning to a previously completed workspace and restoring it |
| H3 | Moderate: inspecting current stock and adding only the remaining deficit |
| H4 | High: selective rearrangement, placement clearance and withdrawal |
| H5 | Moderate: destination inspection and preservation during transport |
| M1 | Moderate: maintaining identity through observed tool use and subsequent segregation |
| M2 | Moderate: reading package quantities and executing a coupled allocation |
| M3 | High: door manipulation, interior access and resident preservation |
| M4 | High: reliable handoffs under a provisional physical deadline |

All nine contracts and paired definitions are implemented. None has an executed
robot completion certificate under this iteration's authorization. Native scene
and geometric results below must not be substituted for that missing evidence.

## Implemented runtime and scoring

Definitions live under
[the candidate catalogue](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/candidates/interactiveworld_20261029/catalogue.json).
The builder is
[build.py](/local_home2/zhouxinyang/models/WIA/src/wia/candidates/build.py);
native layout initialization is
[interactiveworld_candidates.py](/local_home2/zhouxinyang/models/WIA/src/wia/authoring/converters/interactiveworld_candidates.py).
Pure high/medium reducers, event handling, native measurements and prelude delivery
are in [the candidate module](/local_home2/zhouxinyang/models/WIA/src/wia/candidates).

M1 and the undone H3 condition deliver chronological actual head-camera images
before the first decision. Public history contains sensor frames and neutral
timestamps, not private actor labels, event truth or evaluation identities.
Separate owner acknowledgement records delivery; a diagnostic preview does not
earn observation credit. Replay is idempotent and reset invalidates the bundle.
The zero-shot and web paths support this history without changing the prompt bytes
of tasks that have no candidate history. Shared runtime changes were required for
this evidence contract. Large image payloads are saved once and referenced in the
run manifest rather than duplicated there.

M4 uses durations 12,4,8 simulation seconds in its first condition and changes one
duration to 24 in its paired condition. The abstract schedule check includes
two-second robot handling operations and has optimal makespans 24 and 28 seconds;
these are analytical schedule results, not native robot timings. The 120-second
task deadline remains provisional pending an authorized physical execution study.
The current M2 pair has one versus three valid assignments, including both
interchangeable two-unit packages. The current M4 pair has four versus two
optimal schedules, with no shared optimal operation ordering. These checks use
the actual generated task files, not only generic reducer fixtures; the full
[enumeration receipt](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/final_contract_review.json)
is retained.

## Native review and baseline repairs

Native review directly loads, settles and renders scenes. Additional diagnostics
author an event, object pose or container joint state; they never execute a robot
plan. Each receipt states this distinction and records zero robot/model actions.

* T23, T24 and T25 now select their authored tabletop variant. Their former
  selection loaded an unpatched world and could leave the robot below the floor.
  T23's separate QA source and physical-scene source are intentional; both hashes
  are checked against its declared provenance.
* T42, T43 and T44 synchronize authored poses before alignment. This repairs the
  trial object's support placement without changing ramp angles, friction,
  assets, seeds or scoring.
* T51 and R-S1 receive replacement screenshots because old capture evidence had
  a cover/layout or robot-contact concern; current cold captures are retained as
  separate evidence. Their task definitions are unchanged.
* Candidate revisions include supported placements measured by native rays;
  settled region heights; H1 left/right grounding; H4 target grasp dimensions;
  H5 accessible station partitioning; M2 the front shelf ledge and explicit QTY
  text; and M3 clearance for the doors' sampled opening envelopes.
  M1's USED destination was also moved from the inaccessible far desk end to a
  nearby inward region; native support rays reject the desk's rounded corners.

M3's first open-state review found real contacts with a sink and a neighboring
fridge despite valid closed initialization. Its repair measures seventeen native
hinge configurations remotely, restores the unit, and screens translated/rotated
envelopes against the room and the other unit's opening envelope. Padding is
0.04 m in XY. This is a conservative sampled layout screen, not a continuous
robot opening-path certificate. Final open-state contacts must be reviewed
independently after placement.
The placement screen also requires a base-sized working stance in front of the
fully open unit, with service-distance candidates of 1.30 and 1.35 m from
the unit center after the targeted workspace repair. A background SE(2) route to that stance is rechecked against
both units' open collision meshes, including the first unit's route when the
second unit is added. This avoids clear but disconnected front positions. The
independent static audit still checks the final settled world's navigation and
arm access; these screens do not execute door opening or robot navigation.

T37 initially failed with GPU memory exhaustion while another workload appeared
on the same extra GPU. A dedicated GPU 0 retry completed; the temporarily paused
owned dispatcher was restored. Native cache failures from a full local disk and
slow network storage are recorded separately from scene failures. Temporary
native caches use bounded shared memory with durable receipts and logs.

## Static reachability interpretation

The requested 3D spherical-sector model is refined into calibrated R1 kinematics:
four moving torso joints and six arm joints are solved continuously for both
arms, with native joint limits and fixed end-effector transforms. The audit uses
multiple approach directions/heights, a 0.10 m open gripper, a 0.006 m collision
margin, 0.10 m approach/retract segments and sampled joint paths. A 16-heading
rectangular-base search checks translation and rotation against native collision
geometry. Native base collision vertices avoid inflating a rotated world AABB.
Old inventories may receive these vertices only through a strict rigid-pose
transfer with matching calibration and an independently checked native AABB.
Every raw input remains unchanged and traceable.

Destination queries sample nine footprint-eroded positions under one shared
192-IK-call budget. Existing occupancy remains active, so an occupied target may
correctly have no initial witness before a legal rearrangement. Container
interiors are audited separately in their authored open state. Temporal phase 2
and post-event inventories are independent inputs, not substitutes for phase 1.

Results must be read as follows:

* `found_witness`: one sampled base/torso/arm configuration and approach/retract
  passed the configured geometric checks. Grasp closure, forces, carry motion,
  controller tracking and physical success remain untested.
* `not_found`: this bounded conservative search found no witness. Reasons include
  disconnected sampled stances, IK residuals, proxy collisions and aperture
  rejection. It is not proof of unreachability.
* `unavailable`: the whole-body rigid proxy is inappropriate or required native
  metadata is absent. Cloth, articulated movable objects and suitable local
  handles require another representation.

Whole-object AABB width can reject a vase neck, container rim, long rotated object
or two-hand strategy even when a valid local grasp exists. Accordingly, assets
are not arbitrarily shrunk merely to make this screen pass. Some baseline tasks
require opening doors or changing the world before access; their initial closed
geometry alone does not establish full-task impossibility. Required and protected
objects remain accounted for without granting permission to move them.
The generic audit searches both arms; a witness does not override a task-specific
hand restriction or certify maintaining a tray or another held object while
moving. Its results are geometric evidence per query, not task-level passes.

## Reproduction and evidence

Environment: `/local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python`, local
OmniGibson/BEHAVIOR assets, explicit GPU UUID, EGL rendering, two numerical-library
threads per native worker. No scheduler allocation is used.

```bash
cd /local_home2/zhouxinyang/models/WIA
PYTHONPATH=src /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m wia.candidates.build
PYTHONPATH=src /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -m pytest -q tests/test_candidate_*.py tests/test_refine_wia_geometry_inputs.py tests/test_refresh_wia_captures.py
PYTHONPATH=src /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python scripts/capture_interactiveworld_candidates.py --gpu 0 --ids iw_h1_01 --output outputs/interactiveworld_20261029/manual_review
python scripts/summarize_wia_geometry_audit.py --state outputs/interactiveworld_20261029/audit/full_dataset_v2/state.json --output outputs/interactiveworld_20261029/audit_summary_final
```

The last native command requires an idle explicitly selected GPU and should run
inside a persistent terminal. It does not run an agent. The focused CPU suite
passed 265 checks; these cover reducers, actual paired decision changes, observation delivery, native inventory
contracts, geometric algorithms and safe screenshot refresh boundaries.
This is a focused suite, not a claim that the entire repository suite passes.
Three unrelated history-prompt snapshot assertions also failed against baseline
`b7101de`; baseline prompt wording was not changed to satisfy them. Tasks without
candidate history retain byte-identical prompt output in the focused comparison.

Primary evidence locations:

* [Native captures and run receipts](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/baseline_full)
* [Immutable full-dataset audit inputs, code and reports](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/audit/full_dataset_v2)
* [Research iterations and retained/rejected changes](/local_home2/zhouxinyang/models/WIA/research_log.md)

The baseline audit accounts for 224 initial object queries: 59 have geometric
witnesses, 134 remain unresolved, and 31 require another grasp representation.
Its six exported second phases add 43 queries (12 witnesses, 16 unresolved,
15 unavailable). These totals include protected/background bindings and are
not task failure rates. Full object-level reasons and kinematic-only evidence
are retained in the audit CSV/JSON. The definition/native scan separately
retains 27 old bounded same-room-stance warnings; those are neither silently
discarded nor treated as proof of impossibility.

Among the 134 unresolved baseline initial queries, overlapping rejection groups
include whole-AABB aperture rejection in 106 queries, collision-proxy rejection
in 83, exhausted IK budgets in 32, orientation-aware IK residuals in 11, and no
connected sampled base stance in 17. These groups are not mutually exclusive.
The latter includes T10's large structural table binding, inter-room targets in
T07/T08/T36, and targets or spare inventory in T02/T37/T41/T45. Initial room
connectivity can depend on doors; furnished access can depend on unsampled
stances and conservative furniture bounds. Native images and a base-height
proxy map were reviewed for T02/T37/T41. This did not establish physical
impossibility or justify silently removing furniture, altering the tasks' room
semantics, or reporting those queries as passed. These remain explicit access
concerns, with per-object reasons and native geometry retained for future
authorized execution or more detailed geometric analysis.

## Completed uniform review

The [definition/native scan](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/dataset_scan_final)
loads all 80 selections without a definition error and finds current native
rendered inventories for all 80. Its legacy adjacent-audit-file checks do not
discover the separate immutable audit campaign; use the independent
[audit summary](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/audit_summary_final/summary.json)
and [query table](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/audit_summary_final/queries.csv)
for static coverage. All 93 latest inputs are current and nonprovisional, and
all 93 audit jobs finished: 80 initial inventories, six baseline second phases,
five candidate event-final inventories, and two M3 open interiors. The controller
has drained and stopped. The reports contain 401 object/destination queries.

| Group/state | Initial objects: witness / unresolved / unavailable | Destinations: witness / unresolved / unavailable |
|---|---:|---:|
| Baseline initial (61) | 59 / 134 / 31 | Not defined by this inventory schema |
| Baseline phase 2 (6) | 12 / 16 / 15 | Not defined by this inventory schema |
| Candidate initial (19) | 49 / 4 / 0 | 41 / 1 / 4 |
| Candidate event-final (5) | 11 / 0 / 0 | 14 / 0 / 0 |
| M3 doors open (2) | 2 / 4 / 0 | 0 / 4 / 0 |

The four unresolved candidate initial objects are protected M3 residents behind
closed doors. The four unavailable initial destination queries are those closed
interiors; the one unresolved initial destination is H4's intentionally occupied
target. M3 supplemental searches are reported separately and do not overwrite
these uniform-budget results. The median audit compute elapsed time is 79.6 s
per inventory; the sum of the 93 per-inventory elapsed times is 9308.85 s. This
sum is not campaign wall time because jobs ran concurrently. Median wrapper
elapsed time is 82.3 s. The final operational setting is twelve CPU jobs with
two numerical-library threads each; the earlier four-to-twelve concurrency
change is preserved separately without changing numerical configurations.

All 19 candidate initial support checks and their applicable native diagnostic
receipts pass in the [main review receipt](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/final_native_main_review.json).
Both M3 front closeups expose distinct milk/chicken contents; this is human-view
evidence, not a mounted-sensor access certificate. All 77 current batch native
processes exited successfully and removed their owned temporary caches. No
model quality metric, agent success rate or executed robot action cost is
available from this iteration.


## Supplemental M3 destination review

The uniform audit keeps its four unresolved open-interior destination queries.
The supplemental search exposes the difference between a bounded failure and a
geometric impossibility: front points first exhausted budgets or collided with
the shelf, and translating the payload's initial orientation also exceeded the
margin-adjusted aperture. Higher side grasps alone did not resolve the problem.
Permitted payload yaw produced witnesses for the earlier B placement, while A
still lacked a usable low-torso workspace. The final shared initializer reserves
1.30/1.35 m front service distances and retains the same assets, contents, rules,
seed and paired factor. Native full-open contact/preservation checks pass again.

The current-layout supplement adds side heights 0.75/0.90, a world-Z payload yaw
of 90 degrees for A (0 for B), explicit local service stances, and an initial IK
guess from a previously valid solution. Only the ten optimized torso/working-arm
initial-guess values transfer; the measured initial robot posture, inactive arm,
grippers, objects and obstacles remain unchanged. All native-limit, orientation,
collision, sampled joint-path and connected swept-base checks run again. The
budget is 1,152 IK calls and at most 24 stances per sample, with one explicit
stance in the final focused runs; each selected witness took 27 IK calls.

Both units now have a destination witness in both variants. The chosen left-arm
side approaches use height fraction 0.90; their 58-sample paths have minimum
reported proxy clearances of about 7.30 mm (A) and 8.76 mm (B). Main review checked
current raw-input hashes, unchanged engine code, base calibration, all endpoint
errors and joint-limit margins, path counts, restored temporary overrides, and
zero robot/model actions. The [four selected witnesses and all eight supplemental
iteration summaries](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/final_m3_supplemental_review.json)
retain unsuccessful samples too. Earlier-layout witnesses are historical
diagnosis, not certificates for the current scene. These results are separate
from the 401 uniform-budget queries and do not certify door manipulation, grasp
closure, holding, carrying, whole-body base transit, stable release or success.
The protected resident objects need to remain in place; unresolved whole-object
grasp proxies for them do not grant permission to move them.


## Final screenshot refresh and handoff

The original queue's 61 jobs had finished before this handoff. Eight selected
baseline refreshes (T23/T24/T25/T42/T43/T44/T51/R-S1) and all nineteen candidate
variants are now appended at priority 10, below the original priority 130.
Each job uses two attempts and a 2400-second capture timeout. All 27 sealed task
and scene files match the reviewed current definitions and share frozen runtime
`working_2bf3b2c4c9bde7bd7eb1dab88df6498e`; Git HEAD remains `b7101de`.

After validating every replacement, the refresh removed exactly 206 explicitly
mapped old image files. Original logs, receipts, configurations and unrelated
images remain. The [deletion manifest](/local_home2/zhouxinyang/models/WIA/outputs/edge_review_20261029/refreshes/c36ec1319e924c4b9a85417c08bf9880/deletion_manifest.json)
records paths, hashes and replacement jobs. Old receipts deliberately remain
historical and refer to retired images. The candidate receipt index is separate
from the baseline index and does not admit candidates into the release.

At the 2026-10-30 00:40 UTC launch, all eight GPUs were independently rechecked
idle. GPUs 0–3 run the persistent screenshot supervisors; GPUs 4–7 each run one
capture with `--once`, then their native child and supervisor exit and release
the GPU lock. No other workload was stopped. The original queue was not paused.
The new batch started with eight running jobs and nineteen queued jobs, not 27
completed captures. Owned temporary caches use `/dev/shm` and each native child
cleans its scratch after confirmed exit. Reusable tmux terminals remain.

The [handoff receipt](/local_home2/zhouxinyang/models/WIA/outputs/interactiveworld_20261029/final_queue_handoff.json)
records GPU UUIDs, worker PIDs and tmux windows. A separate publisher in
`codex-gpu:iw-final-review-publish` (PID 1582174) updates the existing local review
snapshot every 15 seconds and exits once all 80 indexed jobs are terminal.
Its [live progress](/local_home2/zhouxinyang/models/WIA/outputs/edge_review_20261029/final_refresh_review/progress.json)
and final completion receipt distinguish failed/exhausted captures from success.
The local snapshot has 61 baseline and 19 candidate rows; original registry,
curation and feedback are preserved. No agent or reference robot execution is
part of these workers.


### Furniture rearrangement clarification (2026-10-30)

The user permits normal robot actions to move unprotected movable furniture, including pushing chairs aside. No generic furniture-immobility requirement applies. The static audit evaluates the captured layout with obstacles at their measured poses; an unresolved route may become feasible after physical rearrangement and is not a proof of task impossibility. Task-specific protected residents and anchored fixtures retain their explicit semantics. Privileged grasp diagnostics permit native furniture rearrangement and retain before/after world telemetry. Their initial near-target spawn checks merely prevent unintended scene changes before the first robot action.

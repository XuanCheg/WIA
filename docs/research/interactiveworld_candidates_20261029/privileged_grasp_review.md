# Privileged grasp diagnostics (2026-10-30)

The bounded pilot is complete: two named-target pickup certificates (H5 and
T41), with T02 and T37 still unconfirmed. All owned native/model processes and
temporary caches have been cleaned. These four selected scenes are not a
full-dataset agent evaluation; the 80-scene static scan and screenshot refresh
are separate completed checks.

## Scope and protocol

The user authorizes GPT-6.1-sol with high reasoning effort to inspect all measured
world coordinates, robot poses, joint limits, object bounds and native grasp
telemetry. These are single-target pickup diagnostics, not benchmark episodes or
complete task solutions. A near-target initial robot pose is an explicitly
reported intervention and does not establish navigation from the original start.

Unprotected movable furniture may be pushed or repositioned through ordinary
robot actions. There is no blanket furniture-immobility rule. A static audit
non-witness describes its captured layout and search budget; it does not rule out
physical rearrangement. Initialization drift checks prevent a robot relocation
from accidentally altering the scene before the first agent action. They do not
forbid subsequent robot-induced furniture motion.

The native assisted-grasp controller and its existing guards are retained. The
diagnostic uses the fine primitive profile and metric-target macros. The model
receives one current lossless sensor bundle and full privileged state per decision.
Physics does not advance during model waits. No target teleport, scripted pickup,
injected attachment or altered collision settings are used.

A measured pass requires all of the following:

- The named target is initially unheld by either arm.
- Native readback reports the exact target identity, `holding=True` and an active
  assisted constraint in either arm.
- The target's world AABB bottom rises at least 0.05 m above its initial value.
- The held lift persists for at least 0.5 consecutive simulation seconds.

The probe samples every native control step. A model's DONE or self-reported
success does not certify a pass. Assisted attachment is not a friction or
force-closure certificate. An unsuccessful bounded attempt is not an
unreachability proof.

## Pilot cases

| Case | Target | Starting condition | Diagnostic purpose |
| --- | --- | --- | --- |
| iw_h5_01 | item_01 | Original start | Positive geometric control |
| T02 | obj_A_correct | Same-room west approach, 1.7594 m from target | Thin-object pickup from a chair seat under a table, with furniture rearrangement allowed |
| T37 | wia_t37_target_03 | Original start and separate 1.60 m local assembly start | Distinguish routing from pickup while retaining the existing right-hand tray grasp |
| T41 | public_spare_1 | Original start and separate 0.80 m nearby start, both in authored Phase 2 | Compare pickup with and without original-start routing |

Each physical run allows 24 model decisions, a 180 s model-turn timeout, a 60 s
cancellable action wall limit, a 1800 s loop limit and a 2700 s owned-process
limit including native loading. These replace original task deadlines only for
the diagnostic. Case files retain exact geometry sources and intervention poses.
The T02 same-room retry has 48 decisions, a 2700 s loop limit and a 3600 s process
limit to allow rearrangement; this budget difference is explicit.

T41 uses the existing authored Phase-1-to-Phase-2 transition before pickup. Its
receipt verifies unchanged target pose, unchanged physical world, and zero event
physics steps or robot actions. The Phase-1 survey restriction is preserved;
neither survey completion nor full-task success is credited to this intervention.

## Integration evidence

Earlier attempts are retained separately as infrastructure or initialization
failures. They establish no failed target grasp. Corrections establish the native
initial overview, keep native physics on the main thread, isolate model/MCP
dependencies from Kit's bundled cryptography in a separate CPU process, and use
epoll-compatible descriptor waiting. A high-FD CPU preflight (worker pipe FD 1206)
completed one actual gpt-6.1-sol/high model response using an explicitly labeled
archival image replay and DONE as its only action. No physics ran in that check.

The first T37 robot-only relocation stretched its existing tray attachment and
was rejected. The original-start attempt retained the tray and exhausted its
24 decisions on routing. A separate local diagnostic then rigidly transformed
the robot and its already held tray together, without creating an attachment.
Its first pose was rejected because the tray contacted a cubicle. The revised
1.60 m local pose includes payload screening: minimum captured collision-part
AABB clearance is 0.331142 m for the tray and 0.015031 m for robot sphere proxies.
Native grasp identity, payload pose, contacts, base stability and non-payload
object drift must still pass before any model action. The named target and
furniture are never teleported. The agent is instructed to retain the right-hand
tray and pick up the target with the left hand, as in the source task.

A matched follow-up, `T37_near_release_allowed_r9`, explicitly waives that
source-task tray-retention/left-hand restriction to match the user's narrower
pickup question. The agent may set down or release the tray through ordinary
native actions and use either hand. The initial assembly pose, target/furniture
geometry, exact model, native controllers and 24-decision budget remain the
same; the changed constraint is recorded in the case. The runner performs no
scripted release. A pass here would establish relaxed single-target pickup,
not the source task's required tray-carrying operation.

The geometry-guided follow-up `T37_near_geometry_guided_r10` uses a reviewed
side-facing start at (-16.661699, -8.300000), yaw 0. Captured fixed part bounds
at Y=-7.96 m, Z=0.30 m leave an entrance width of 0.691078 m. Base-link collision
vertices span 0.637263 m along local X and 0.684036 m along local Y, so facing
north leaves approximately 7 mm nominal total clearance while a side-facing
base leaves approximately 54 mm. The initial X position compensates for the
base mesh's asymmetric local bounds. A separate check of all three wheel-link
bounds found them inside the projected base-link envelope. This does not certify
upper-body clearance or a continuous entry path. The selected initialization
has 0.015031 m robot proxy and 0.133054 m held-payload clearance. The agent receives
the measured bounds and 27 nearby fixed collision-part AABBs in addition to
the existing privileged state. The tray may be released natively. Target
geometry, controllers, exact model and 24-decision budget remain unchanged;
initial pose and geometric hints change, so this is a feasibility follow-up
rather than an isolated causal or performance comparison.

A retained native IK compatibility fix replaces membership testing on
OmniGibson's keyed-only `CachedFunctions` with narrow required-key lookups, and
unwraps previous `NativeIKGuard.compute_control` bindings when reconstructing a
controller. Solver formulas, speeds, joint margins and contact guards are
unchanged. The focused IK, rejection-continuation and dual-arm tests pass (56).
T41's later gripper rejection was independently traced to its survey-phase
policy and corrected through the authored phase transition above. Frozen run
code and hashes distinguish these runtime versions.

## Results

| Run | Decisions | Named-target acquisition | Maximum held lift | Longest qualifying hold | Result |
| --- | ---: | --- | ---: | ---: | --- |
| iw_h5_01_original_r5 | 13 | Right hand | 0.079872 m | 1.066667 sim s | Pass |
| T41_original_phase2_r7 | 23 | Left hand | 0.079184 m | 1.033333 sim s | Pass |
| T41_near_phase2_r7 | 24 | None | 0 m | 0 sim s | Decision budget exhausted |
| T37_original_r5 | 24 | None | 0 m | 0 sim s | Decision budget exhausted during routing |
| T02_same_room_r6 | 48 | None | 0 m | 0 sim s | Decision budget exhausted; pickup remains unconfirmed |
| T37_near_payload_clear_r8 | 24 | None | 0 m | 0 sim s | Decision budget exhausted at entry; target pickup not attempted |
| T37_near_release_allowed_r9 | 24 | None | 0 m | 0 sim s | Tray released; decision budget exhausted at entry |
| T37_near_geometry_guided_r10 | 24 | None | 0 m | 0 sim s | Partial entry and chair motion; budget exhausted before target grasp |

H5 has 469 sampled control steps (15.6333 simulation seconds). Its model calls
total 200.69 s, actions 70.14 s and diagnostic loop 283.70 s. T41 original has
1790 sampled steps (59.6667 simulation seconds), 506.26 s of model calls,
211.91 s of actions and 734.06 s loop time. These wall times exclude native cold
loading. The two passes establish feasibility of those exact single-target
pickups; they are neither full-task solutions nor a dataset success rate.

The successful runs reported 3,178,149 and 6,313,723 cumulative tokens,
respectively, including 2,471,680 and 5,107,712 cached input tokens. Persistent
history, full privileged object/link telemetry and multiple camera images make
these diagnostic costs substantial. These are transport usage counts, not
unique prompt sizes or monetary estimates. H5 and T41 use different frozen
runtime revisions and are not a controlled cross-scene performance comparison.

T41's unsuccessful nearby trial tipped and displaced the shaker without native
acquisition. The original-start trial subsequently establishes that the target
is physically pickable under the diagnostic protocol. A closer initial base
pose therefore does not guarantee an easier grasp for this policy. T02's first
1.40 m pose was across a fixed window and is retained as an unsuitable local
start, not evidence of a failed local pickup.

T41's uniform Phase-2 static query found zero connected sampled stances and made
zero IK attempts. Its later native pickup shows why this search result must not
be interpreted as impossibility. The successful trajectory is not a certificate
of collision-free static navigation: the native base was pitched approximately
7 degrees before the final lift, and the static planar proxy cannot reproduce
every contact-supported physical posture. No movable object's center had shifted
more than 1 cm at the last predecision snapshot in that run. Preserve both forms
of evidence rather than weakening collision margins to force agreement.

T02 demonstrates substantial physical rearrangement: by predecision snapshot 037,
one table center had moved 1.405 m, two chairs approximately 0.79 m each, the
supporting chair 0.307 m, and the target 0.359 m. These are native robot-induced
motions after initialization, not teleports. Other task objects also moved; the
diagnostic does not award original-task constraint compliance. Contact with the
thin board is not acquisition. The native assisted-grasp implementation requires
both finger contact and an intervening grasp-ray hit before selecting a rigid
object, which is stricter than simple proximity or apparent closure.
The completed same-room run made four GRASP commands (decision indices 22, 26,
33 and 43), all with `holding=False` and no target constraint. It recorded no
controller exception or unknown-step execution. The run covered 92.9 simulation
seconds; model calls took 1488.20 s, actions 264.04 s and the loop 1809.31 s.
Repeated guarded downward motions and empty closure leave an alignment/contact
problem unresolved; recorded telemetry does not identify the exact failed
assisted-grasp eligibility condition. Do not change the board asset or placement
on the basis of this bounded policy failure alone. The original T02 task also
allows choosing between boards, so failure on this named board does not prove
that the full task is impossible. All owned T02 processes and its cache exited.

T37's valid tray-retaining local attempt also exhausted 24 decisions without a
target GRASP command. Its last measured left-hand-to-target distance was about
1.12 m. Five base motions and four arm motions stopped for insufficient progress;
the tray remained natively held. The loop took 1256.65 s (996.31 s model calls,
227.52 s actions), spanning 83.3 sampled simulation seconds. This is an entry and
payload-handling failure of the bounded policy, not a failed local target grasp.
Owned processes and scratch were cleaned. In the matched tray-release comparison,
the model's first action was RIGHT_RELEASE, confirmed by native empty-hand
readback. It still exhausted 24 decisions at the entry, with eight base stalls
and no target GRASP command. Its final reported left-hand distance was about
0.91 m; the target remained unheld. The loop took 1284.91 s (1111.02 s model,
142.50 s actions), spanning 47.4 sampled simulation seconds. Owned processes
and scratch were cleaned. Releasing the tray alone did not resolve this policy's
entry problem; no claim about target grasp impossibility follows. The
geometry-guided side-facing trial also completes without a target GRASP command.
It successfully moves 0.39997 m sideways into the entry and shifts the movable
chair approximately 0.132 m north and 0.041 m east. The following advance moves
only 0.0265 m before stopping. Subsequent turns before full entry, joint-limit
recovery and arm clearance adjustments use the remaining decision budget.
Measured pose progress is useful, but it does not establish target pickup.
The loop takes 1177.75 s (976.44 s model, 169.53 s actions), spanning 59.3 sampled
simulation seconds. There are six base stalls, three arm stalls and one
arm-joint-limit result; all native/model processes and scratch are cleaned.

Retain T02 and T37 as unresolved diagnostic concerns. T02 reached the thin board
and attempted acquisition without an attachment. T37 never reached a target
GRASP command, even after a better entry alignment produced physical chair
motion. Those outcomes locate different bottlenecks and do not prove either
object ungraspable. No scene asset, placement, fixed-body setting or collision
guard was changed to convert these bounded policy failures into passes. The
latest diagnostics therefore require no additional scene screenshot invalidation.

All completed earlier attempts have verified owned-process and temporary-cache
cleanup. The original screenshot queue independently completed all 80 selections,
including the 27 appended refresh jobs. Screenshot availability and completion
of the full static scan do not establish grasp success or task admission.

Durable case, code snapshot, model provenance, per-action world telemetry, exact
model images, native grasp samples and cleanup receipts are under
[/local_home2/zhouxinyang/models/WIA/outputs/privileged_grasp_20261030](/local_home2/zhouxinyang/models/WIA/outputs/privileged_grasp_20261030).
The compact machine-readable status and cost summary is
[review_summary.json](/local_home2/zhouxinyang/models/WIA/outputs/privileged_grasp_20261030/review_summary.json),
regenerated from preserved artifacts by
[summarize_results.py](/local_home2/zhouxinyang/models/WIA/outputs/privileged_grasp_20261030/summarize_results.py).
An initial-geometry diagnostic figure shows collision-part AABB slices at three
heights and the measured base-center path for the tray-retaining T37 trial:
[T37_entry_slices.png](/local_home2/zhouxinyang/models/WIA/outputs/privileged_grasp_20261030/T37_entry_slices.png).
These conservative slices are not exact mesh cross sections or a collision-free
path certificate.

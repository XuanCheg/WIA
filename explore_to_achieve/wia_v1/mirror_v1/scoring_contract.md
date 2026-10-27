# Mirror scoring implementation contract

This main-agent contract accompanies design.json. It specifies implementation,
not evidence of optical visibility or physical feasibility. Full reference
execution is deferred until user review. Native jobs use physical GPUs 0–3 only.

## Pure reducer interface

`MirrorProgress(parameters)` exposes `begin(sample)`, `update(sample)`,
`report_answer(text)`, and `result()`. No simulator imports, I/O or clock stepping.
Parameters are the generated evaluation parameters, including explicit scoring
thresholds, candidate identities, destination bindings, operation and relation.
Destination keys are semantic `blue` and `yellow`; table support is `table`.

Sample fields shared with DimensionalProgress are `native_time_s`,
`control_valid`, `fixtures_valid`, `geometry_valid`, `grasp` and `targets`.
Grasp contains `available`, `held_names`, `released`. Every target contains
`object_name`, `position_m`, `quaternion_xyzw`, `dimensions_m`, `vertices_m`,
`lowest_z_m`, `linear_speed_m_s`, `angular_speed_rad_s`, `dynamic_collidable`,
`robot_contact`, and `destinations`. Each destination relation contains measured
`touching`, `on_top`, and `footprint_inside` booleans. Include `table` for every
spatial target. Target `centroid_m` is the mean of its native collision vertices.
Sample `anchor_centroid_m` is the fixed native pillar collision centroid.

The initial sample also contains `optical_selection`: `available`, `accepted`,
`selected_candidate` (stable candidate key), `appearance_mode` (direct/reflected),
`frozen_eye_m`, and complete geometric diagnostics supplied by the adapter.
Freeze the measured selected identity, appearance mode and anchor at begin.
Missing or ambiguous initial selection prevents qualification permanently.
The measured appearance mode must match the declared configuration, otherwise
the episode is unavailable. Never derive truth from source QA or public labels.

Explicit false protected-fixture/geometry/controller values latch an invalid
episode. Missing evidence is unavailable and resets incomplete stability dwell;
it is never success. Stable time uses strictly increasing native physics time,
not record calls. Gaps above max_sample_gap_s reset dwell; backwards time is
invalid. Repeated timestamps cannot advance dwell or pickup events.

## Physical and report semantics

Initial held objects do not establish a pickup. Require measured released hands
before a post-begin genuine grasp of an original native candidate. Lift is the
candidate's measured lowest collision vertex rise from its initial lowest Z,
while held, by minimum_lift_m. First candidate grasp commits a single-target
task; any wrong or subsequent different-candidate grasp invalidates selection.
Unselected objects must remain within unselected_position_tolerance_m of their
initial position throughout single-target tasks. Spatial swap exempts both
required objects. Candidate shape and scale remain invariant under rotation.

Delivery destination is blue for correspondence, blue for measured reflected
appearance and yellow for measured direct appearance. Require correct selected
identity, genuine grasp/lift, released aggregate hands, no robot contact,
native Touching and OnTop, full footprint inside the pad, speed thresholds,
and stability_duration_s of uninterrupted sampled stable placement.

Classification accepts exactly `direct` or `reflected` after trimming whitespace
and optional `ANSWER:` prefix, case insensitive. Before the first candidate
grasp, the latest valid report may replace a prior report. Freeze its value and
presence at first grasp. Later reports remain diagnostic and cannot establish
or correct the pre-grasp answer. Wrong/missing answer prevents task success;
physical delivery may still be recorded separately. report_answer and result
must never advance physics time or stability dwell.

Spatial relations use frozen anchor XY and current native target centroid XY.
S1: selected distance minus other distance >= configured gap. S2: other minus
selected >= gap. Both must be false initially; an already-correct initial state
is unavailable, never repaired by changing the goal. S3 freezes the initial
near/far identities, requiring initial gap >= minimum_assignment_gap_m; its
goal reverses that order with configured gap. S3 requires genuine grasp/lift
of both originals and each centroid XY displacement >= minimum_required_motion_m.
S1/S2 require selected real displacement >= the same threshold. At completion
all spatial objects must be stably released, not robot-supported, fully inside
and on top of the table for the same dwell. Reconfiguration alone is not success.

Completion IDs are exactly those in design.json. Results expose task_success,
physical_success, evaluation_validity (complete/unavailable/invalid),
completion_items, per-target evidence, frozen selection/report/order, distances,
measurement errors and latched constraint violations. Explicitly set
optical_visibility_verified=false, optical_necessity_verified=false,
complete_reference_execution_verified=false. Geometric selection is not a
visibility certificate. result returns a defensive copy without side effects.

The reducer computes target centroids from supplied finite collision vertices;
the redundant centroid_m field is diagnostic. Spatial anchor movement beyond
fixture_position_tolerance_m invalidates the episode; a missing current anchor
measurement makes it unavailable. SR3 selection requires both original grasp
histories and its grasp_lift stage requires both lifts.

## Native adapter and initialization

The mirror module delegates configure_scene and postprocess_joint_world to
active_layout without changing source assets. initialize_scene is allowed only
under _category_initialization_active. It resolves the semantic mirror binding
through the scene adapter's bindings or the exact authored panel object name,
sets only that object's actual native visual PBR materials, then delegates
settling/support setup to active_layout. Use material.set_input for metallic and
roughness and its native diffuse_color_constant tensor setter. Record actual
get_input/property readback, including material prim paths. Qualification
requires metallic 1, roughness 0 and diffuse [0.95,0.95,0.95] within floating
storage accuracy (1e-6), with finite measured values. Check the actual current
material on every measurement; do not infer optics from requested settings.
No visual texture, image compositing or source mirror QA may supply reflection.

Reuse the dimensional adapter's native geometry, aggregate grasp, control,
fixed-fixture preservation, contact and support relation helpers. Retain exact
native object identities and measured Cube dimensions. Intrinsic dimensions
use local mesh/root-link transforms; world points remain native world readings.
Protect table, pads, mirror, cue rods, stands and pillar. Require actual upright
and floor/support initialization; accepted active_layout alone is insufficient.

Measure the axis-aligned panel's observer-facing plane as minimum world X of
its enabled native collision vertices, normal [-1,0,0]. Cue plane X is the mean
of the four measured rod centroids; require rod planes consistent within
fixture_position_tolerance_m. Cue opening bounds are inner Y side faces and
inner Z floor/roof faces. Validate axis alignment with the configured angle
tolerance and positive dimensions. All bounds come from current native geometry.

Read the actual session.robot_camera_registry['agentview_rgb'] sensor, falling
back to session.camera_name only if the registry is absent. Record its native
world pose, intrinsic_matrix and image_width/height. Eye must be on the front
side of candidate/cue/panel. Freeze it at begin; qualify_operation may measure
independently but must not change state or camera pose.

Determine measured appearance from cue depth, before comparing with the private
proposal: direct when eye.X < cue.X < every candidate centroid.X minus position
tolerance; reflected when every centroid.X plus tolerance < cue.X < panel.X.
Other geometry is unavailable. For reflection compute v = c - 2*n*dot(n,c-p)
for each candidate centroid c and actual panel point p. Direct uses v = c.
Intersect eye + t*(v-eye) with the cue plane; require 0<t<1 and the hit inside
the measured opening inset by aperture_clearance_margin_m. For reflected mode
also require the ray's panel-plane hit inside the finite panel rectangle.
Project v with the frozen native camera pose/intrinsics (camera optical -Z,
up +Y, same convention as partial_occlusion._camera_opportunity); require
positive depth and pixel inside the actual frame. Exactly one candidate must
pass these geometric checks. Record all intersections/pixels/rejection reasons.
These checks do not establish visibility, absence of occlusion or necessity.

Begin latches the exact measured initial qualification and reducer baseline.
sample_step alone reads current native samples and updates temporal state.
report_answer only forwards the text to the reducer; record writes the cached
sample and result without resampling or stepping. Preserve unavailable/invalid
status through core.tasks.completion.progress and never upgrade failed initial
qualification. MirrorProgress report/physical/task success remain distinct.

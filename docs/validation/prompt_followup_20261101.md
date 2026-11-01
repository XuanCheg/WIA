# Prompt follow-up: time, declarations and task semantics

This iteration implements the user-approved follow-up to the [prompt audit](/local_home2/zhouxinyang/models/WIA/docs/validation/prompt_audit_20261101.md). T09's question and geometric scoring remain unchanged. No model API is called. All native action runs below are diagnostic, not task-solving trajectories.

## M4 simulation time

The camera owner samples `simulation_clock.elapsed_simulation_s` with the observation snapshot ID, action step and capture timestamp. Its origin is the same native tick used to begin M4 evaluation. HTTP/MCP transports the cached boundary value; the runner rejects a mismatched clock rather than silently substituting current wall time. The modular prompt states that this is simulation time sampled at the image boundary, not a live timer during inference. Only M4 publishes this task timer; private station progress and completion flags remain private. The held-transport requirement, processing durations and deadline are unchanged. Both candidate M4 definitions increment to version 2.

The first production MCP replay passed. Across ten observation boundaries and nine fixed-policy prompts, the clock started at zero and remained monotonic. Its final value, 15.400000803172588 simulation seconds, exactly equals the final private scorer's elapsed time. This checks clock alignment, not completion of the processing schedule. See [clock evidence](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/clock_validation.json).

## Room descriptions

Inspected the archived initial Global images for T05 and T06, then used source room-view centers and bounds to corroborate identity. T05's `bedroom_1` is identified as the upper-right bedroom in that image. T06's `corridor_0` and `corridor_1` are identified as the leftmost and middle horizontal corridor sections. These descriptions replace proximity hints naming the queried room. They identify regions within an already authorized image; no new map, coordinates, adjacency graph or answer is supplied. The room IDs, geometry, visitation requirements and private answers are unchanged. Both public task versions increment to 5. Composite instructions now clearly put each option under its goal key in the answer mapping.

See [public before/after evidence](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/neutral_room_descriptions.json). This is visual/source inspection of archived frames, not a new robot exploration trajectory. Historical source questions and private copies remain intact.

## Best-fit containment

The [original ESI task implementation](/local_home2/zhouxinyang/models/WIA/ESI/src/active_explore/tasks/physical_structure/rigid_containment.py:280) explicitly prefers the smaller suitable container when several fit. T22/T24/T25 now state this omitted rule without changing their private answers. T22's original medium-box answer became small-box in the earlier recorded user-authorized gift-box scale revision; that difference is intentional.

The completed native checks support the following labels under the existing stable partial-insertion contract:

| Task / object | Retained or revised answer | Evidence and limits |
| --- | --- | --- |
| T22 gift box | Small packing box, retained | All three candidates have complete stable-placement witnesses. |
| T23 granola box | Mug, revised from wicker basket | The smallest candidate has a complete stable-placement witness. |
| T23 spray can | Mug, revised from bucket | The smallest candidate passes at Euler XYZ [90, 0, 0] degrees. |
| T23 colored pencil | Wicker basket, revised from bucket | Basket and bucket pass. Nine tested mug orientations fail; the user approved the smallest empirically verified candidate, without claiming universal mug infeasibility. |
| T24 first aid kit | Packing box, retained | Passes after the initialization-only awake-contact repair described below; three mug drops fail. |
| T25 wallet | Tupperware, retained | All three candidates have complete stable-placement witnesses. |

The [native fit probe](/local_home2/zhouxinyang/models/WIA/src/wia/runtime/sim/rigid_fit_probe.py) uses explicit pose setters to place released targets above openings, then steps native physics. Each pair tests Euler XYZ orientations [0, 0, 0], [0, 90, 0] and [90, 0, 0] degrees, stopping at its first positive witness. For the T23 pencil/mug pair, six additional orientations align each of the three principal geometry axes to positive or negative world Z, for nine tested orientations total. Each drop centers the target collision bounds over the candidate bounds, starts 1 cm above the rim and allows at most six simulation seconds of settling. There is no exhaustive translation, yaw, tilt or insertion-path search. The anisotropically scaled pencil asset is retained unchanged. See [all final probe configurations and results](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/fit_final_summary.json) and the [T23 answer revision](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/t23_answer_revision.json).

Witnesses use native interior/contact/penetration evidence, velocity limits, hand clearance and a consecutive 0.5-second stable interval. The existing scorer requires both hands empty and the working gripper open; the inactive empty gripper need not be fully open. T22–T25 wording now states that rule explicitly, without changing the grader. The diagnostic opens both grippers. Placement uses measured per-link collision geometry; a fresh post-physics AABB checks it before accepting measurements. The 0.5 mm placement-consistency tolerance only checks diagnostic setup and does not relax scoring thresholds. The failed earlier setup revisions are preserved and are not negative fit evidence.

These state-set probes establish stable-placement existence, not a legal grasp/transport trajectory or benchmark success. The [six reducer checks](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/label_eligibility_replay.json) replay only each measured stable window, including its original timestamps. They exclude preceding drop transients and initial qualification. Replaying the entire spray-can drop correctly latched an earlier container-pose violation even though a later stable window passed; therefore the window result must not be reported as full-episode physical success. Positive robot task-solving trajectories remain a separate TODO.

T24 exposed a contact-availability problem: the target fell asleep before completing the dwell, and its geometry differed from the last measured native contact sample. Exact retention correctly refused the stale contact evidence. An isolated tabletop-variant control applied the existing T25 awake-contact initializer only to `obj_containee`, setting its sleep threshold to zero at initialization. The packing box then passed the unchanged native conditions; the larger tote also passed. Retain this patch in the release task and scene (task version 13, scene version 9), keeping geometry, coefficients, pose specifications, seed, answer and scoring thresholds unchanged. There are no runtime wake callbacks. The [repair receipt](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/t24_awake_repair.json) links both probes and the successful candidate MCP check. A mistaken original-variant launch was stopped and excluded; a pending-candidate benchmark launch was refused before simulation. Neither is comparison evidence.

## Structured output follow-up

Optional QA completion text and current converter/authoring command templates now request independent answer/arrival/survey fields. The separate original-QA confidence protocol remains intact. Historical generated definitions and raw source records are not regenerated. Composite authoring now emits public goal keys on separate lines and sizes its answer mapping to the actual goal count; enumerative declaration payloads use a JSON-encoded string in the independent answer field where that evaluator expects serialized JSON.

A further T38 issue was found by tracing the fixed survey declaration: an incomplete survey is legitimately rejected by the evaluator, but the runner previously treated that rejection as a fatal transport failure. The survey path now handles the explicit boundary rejection as feedback, without issuing terminal DONE or pretending the survey completed. Transport/server failures still raise errors. The fixed policy attempts this negative survey case once, then continues its remaining actions. Positive transitions and rejected transitions are covered by CPU fixtures. A fresh native T38 run also passed: the declaration at action boundary 4 received a nonfatal rejection, subsequent actions continued, terminal DONE ended the run, and final evaluation validity remained complete. It made ten fixed-policy decisions and zero model calls. The earlier reused-scene attempt and its cold fallback failed during loading with Vulkan out-of-device-memory errors; those raw attempts remain preserved. The successful retry ran on physical GPU 5, and its completed owner was released afterward. See the [T38 retry](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_t38_retry_20261101/00_t38/scorer_validation.json).

## Validation status

The seven representative production MCP checks now all pass across the original batch and the separate T38 retry: M4, T52, T38, T40, T23, T09 and T05. Each has complete evaluation validity and no validation errors. The original batch ran on physical GPU 6; T38 retried on physical GPU 5. T52 reused M4's owner, and T23/T09 reused T40's owner. T05 encountered Vulkan out-of-device-memory during reused loading, then passed its automatic cold fallback. T38's initial reused attempt and automatic cold fallback both encountered loading OOM; its later separate cold run passed. No failure attempt was overwritten. See the [consolidated native evidence](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/native_replay_summary.json).

These fixed commands validate the real HTTP/MCP interface, scene lifecycle and scorer availability. They do not solve the tasks or call a model. T40 received 25% partial credit because the fixed first-public-option answer was `yes`, which happens to match the event; its physical comparison/selection/placement obligations failed and task_success remained false. This is an explicit diagnostic answer, not credit for a missing answer. T52 submitted no inside answer and received zero credit. Other workloads intermittently occupied the GPUs, so wall times are operational observations, not controlled throughput comparisons. Completed main-batch and T38 owners were released after verification; raw outputs and reusable tmux terminals remain. No unrelated task was stopped.

The containment diagnostics are complete. After the user authorized migration, all four GPUs 0–3 passed CUDA allocation and numerical checks. Completed owned services were released and remaining checks moved to those devices; unrelated workloads were not interrupted. Final release MCP checks of T23/T24 and T52 structured-positive/text-only-negative controls all passed on GPUs 1/0/2/3 respectively. The earlier seven-case result predates the final T23 label and T24 initialization changes. New caches use a private /tmp root; durable measurements, images and configurations remain under project outputs. Only rebuildable texture caches of stopped, completed owners were removed to make room.


The [four final release checks](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/final_native_summary.json) each execute all eight fixed motions, receive ten observation boundaries, actively declare DONE, and finish with complete evaluation validity and no validation errors. T23 receives 50%: all three revised structured answers pass, all three physical placements remain incomplete, and task success is false. T24 receives 0% and does not claim placement success; its native initializer receipt confirms the zero sleep threshold was applied and read back. Both T52 controls satisfy the same physical-inside condition and actively DONE: an `inside` answer at action step zero followed by later DONE earns 100%, while legacy `ANSWER: inside` text with null structured answers earns 0% and creates no answer history. This verifies separate-decision declarations and text isolation. These are fixed-policy protocol outcomes, not model recognition results.

| Final check | Physical GPU | Full check wall time | Diagnostic score |
| --- | --- | ---: | ---: |
| t24 | 0 | 556.9 s | 0% |
| t23 | 1 | 435.0 s | 50% |
| t52_positive | 2 | 391.5 s | 100% |
| t52_text_only | 3 | 408.2 s | 0% |

The four checks ran concurrently. Their wall times include loading and interface validation and are not controlled model latency or throughput measurements. All four completed owned simulator services were released, retaining tmux terminals, measurements and logs.

The final combined CPU scope passes **1434 checks**, with **54 documented baseline exclusions**, in 68.32 seconds. Older fixtures now assert structured declarations and distinguish decisions from unknown API-call counts. The fresh modular audit renders all 60 release tasks across 240 layouts (cache layout on/off and first/subsequent decisions), retaining independent fields and finding no enumerated private-key matches. These checks use fixture sensors and supplement source inspection; they are not model-quality measurements. See the [render audit](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/render_audit.json).

- [MCP batch results](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_native_20261101/results.json)
- [Final fit results](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/fit_final_summary.json)
- [CPU command](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/cpu_command.json)
- [CPU log](/local_home2/zhouxinyang/models/WIA/outputs/prompt_followup_20261101/cpu_release_final.log)

No commit or push was made.

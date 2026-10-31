# Outcome-based placement revision, 2026-10-30

The user requests final-condition success for Action Order Inference and similar placement tasks, removing the 5 cm lift gate. The active release now opts 33 definitions into `final_state_v1`. Historical definitions and default legacy evaluator behavior remain available; the new tasks no longer claim to measure a prescribed pickup/order sequence.

| Active tasks | Current physical criterion | Retained task-specific requirements |
|---|---|---|
| T01–T03 | Released, stable final support structure; either original base/top candidate is allowed | Correct support order, no external prop for upper members, 0.5 s stability. Base can remain untouched; completion may be anywhere on stable support. |
| T18–T21 | Upright released three-object tower | Original sizes, starting task support, genuine support contacts, no external prop, 1 s stability. |
| T22–T25 | Released stable partial insertion in each correct container | Native audited interior/contact/nonpenetration, 2 cm hand clearance, QA, 0.5 s stability. User confirmed sequential verification and removal after verification. |
| T26–T34 | All targets supported inside the tray in the requested relation | Tray bounds, actual nonpenetrating deck contact, geometric separation/contact rules, preserved scales/fixtures, released stable state. |
| D1-1–D1-3 | Correct sampled passage and final blue-pad placement | Physical traversal through the opening, fixed frame/dimensions, DONE verdict. Passing around the frame still fails. |
| D2-1–D2-2 | Correct final size-based assignment | Intrinsic dimensions, correct pads, all assigned placements stable simultaneously. |
| R-A1–R-A2, R-C1–R-C3, R-S1–R-S3 | Correct original objects at branch-correct destinations and requested relations | Frozen optical reference/anchor, correct appearance answer where requested, protected unselected objects, stable released placement. |

Auxiliary lift, acquisition, support-separation and action-order/travel-distance gates are removed in this explicit mode. The actual stability and contact conditions remain. Genuine lift-only endpoints, temporal knowledge and process mechanisms are outside this change.

T02's cutting board never had to be picked up under the previous evaluator; the earlier board-grasp diagnostic therefore did not establish a mandatory task failure. Its public task now makes the untouched base and unrestricted final location explicit.

T37 is a separate access repair. Move only the third cola bottle to the open side of its existing cubicle desk at world XY (-16.22, -7.55). The local static comparison found a left-arm approach and retract witness from (-16.70, -8.45), yaw pi/2, with 39 sampled configurations and approximately 15.0 mm minimum proxy clearance. Keep all furniture, source room, asset sizes, remaining inventory and carried-tray collection rules. A new per-object initialization option preserves the reviewed XY while retaining native support checks and a 15 mm settling-drift limit. Native initialization, grasp acquisition, carried-tray access and original-start travel remain unverified. Whole-route lattice searches were bounded at approximately 20 minutes; incomplete searches are not impossibility evidence.

Relevant CPU checks: **508 + 72 = 580 passed**. All 34 revised task/scene selections agree with their derived source copies. No new native run completed because all eight GPUs were occupied by existing external workloads at the checkpoint. **34 replacement screenshot jobs are queued; 908 mapped old images were deleted after replacement validation.** Four restored supervisors wait for idle GPUs 0–3. Extra GPUs 4–7 receive one-shot workers only when idle, and their native processes exit after capture. The review publisher tracks completion; pending screenshots are not task-success certificates.

Evidence and current definitions:

- [33-task derivation manifest](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/final_state_20261030/manifest.json)
- [T37 derivation manifest](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/t37_access_20261030/manifest.json)
- [T37 selected geometric witness](/local_home2/zhouxinyang/models/WIA/outputs/t37_access_repair_20261030/local_0.json)
- [T37 candidate geometry plot](/local_home2/zhouxinyang/models/WIA/outputs/t37_access_repair_20261030/proposal.png)
- [Combined capture manifest](/local_home2/zhouxinyang/models/WIA/outputs/final_state_20261030/combined_capture_manifest.json)
- [Live extra-GPU watcher status](/local_home2/zhouxinyang/models/WIA/outputs/final_state_20261030/all34_watch/status.json)
- [Live review progress](/local_home2/zhouxinyang/models/WIA/outputs/edge_review_20261029/final_refresh_review/progress.json)

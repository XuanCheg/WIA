# Review and note reconciliation — 2026-10-30

The review contains 58 saved decisions for currently selected variant IDs: 48
approved and 10 needs-fix. Six approved rows also contain actionable notes.
Consequently, the latest notes require follow-up on **16 configurations**.

This iteration reconciles feedback with definitions and existing evidence. It
does not change task definitions, user decisions, queue jobs or capture artifacts,
and does not run a simulator or robot agent.

Sources:

- [User feedback](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/wia_v1_20261023/review/feedback/reviews.json)
- [Published snapshot](/local_home2/zhouxinyang/models/Show-Harness-Explore/outputs/wia_v1_20261023/review/snapshot.json)
- [Frozen inputs, per-row reconciliation and evidence](/local_home2/zhouxinyang/models/WIA/outputs/review_reconciliation_20261030/reconciliation.json)

## Method and findings

Compare every saved decision and note with its selected variant and fingerprint,
check current definition identity, and inspect retained native evidence for the
reported layout problems. Preserve actionable notes when their decision is
approved or their fingerprint has become stale. Keep image availability,
human design approval and physical execution evidence separate.

| Configurations | Saved decisions | Finding | Required follow-up |
|---|---|---|---|
| T22 | Needs fix | Confirmed wrong variant selection. The catalogue and both old/new capture manifests select `original`, which has zero patches. The intended chair/table placement is in `tabletop`, with six patches. Existing native imagery and inventory put the gift box and all three containers on the floor. | Correct the active selection and capture binding, initialize the intended supports, then inspect support contacts and reachability. Repeating the currently queued capture alone cannot fix this. |
| D1-1, D1-2, D1-3 | Needs fix | All three authored robot starts are 1.30 m from the aperture center in XY. The note requests a closer aperture. Base translation and rotation are already supported; no stationary-base requirement was found in the public contract or dimensional scorer. | Move the aperture assembly toward a usable stance while preserving opening dimensions, genuine traversal and final delivery. Make lateral robot movement explicit. Validate approach, passage and withdrawal with the moving torso and both arm limits. |
| T40, T41, T45, T46, T47, T48 | T40/T47 need fix; the other four approved with notes | All six current overview images show the boxes near the table centerline. The same requested adjustment applies to all six. | Move boxes and their contents a small distance toward an accessible long edge on the same tabletop. Preserve sizes, supports, relative contents and temporal conditions. Allow legal chair movement. Choose the offset from native support geometry and reachability checks. |
| R-S3, R-C1, R-C2, R-C3, R-A1, R-A2 | First four need fix; R-A1/R-A2 approved with notes | Confirmed missing current images. All six have zero published images and replacement jobs waiting in the queue. | Complete and publish replacement captures, then inspect the rendered evidence. The missing-image notes do not establish a reflection implementation defect. |

T22's retained native inventory reports floor support for all four task objects.
Their minimum AABB heights are approximately -0.002 m for the gift box and zero
for the containers. The authored alternative instead binds the gift box to
`armchair_bslhmj_0` and the containers to `breakfast_table_skczfi_1`. This is a
selection defect, not merely a low authored pickup height. Scanning all 61
baseline selections finds no other empty `original` selection with a nonempty
alternative layout. This check does not prove that every other layout is correct.

## Version and coverage reconciliation

The latest saved feedback spans 2026-10-30 05:00–05:26 UTC. During that interval,
the final-state revision and T37 refresh changed selected definitions. **31 saved
reviews now have stale fingerprints**. Their notes remain recorded and actionable;
they are not discarded and should not be silently transferred as approval of a
different task version.

The current snapshot therefore reports 25 approved, 2 needs-fix and 53 unreviewed.
The 53 consist of the 31 stale reviews plus 22 selected variants without a saved
record: T01–T03 and the 19 H1–H5/M1–M4 candidate variants. T01–T03 retain approvals
for earlier variants. No candidate feedback was found in this feedback store.
This is a record-coverage statement, not a request to repeat all prior review.

All 160 selected task/scene definition hashes match the frozen published
snapshot. There are 46 rows with readable declared images and 34 with zero
images; no declared image path failed the review loader. The 34 replacements
are queued with zero attempts. Live inspection found existing compute processes
on all eight GPUs, including resident processes on devices reporting zero
instantaneous utilization. No process was stopped and no new capture was launched.

Historical notes about the 5 cm lift requirement are addressed by the explicit
`final_state_v1` revision. Other historical variant notes remain in their original
records; later design approvals are not native grasp certificates. T37's latest
layout approval likewise does not certify the subsequent aisle-pickup revision.

## Repair order

1. Correct T22's selection and verify the intended support initialization.
2. Adjust D1 geometry and the six box layouts, preserving each task's mechanism.
3. Recheck affected pickup and placement geometry with torso motion, joint limits,
   legal furniture rearrangement and collision clearance. Use native checks for
   unresolved cases; do not treat a missing static witness as impossibility.
4. Seal replacement definitions and refresh affected captures through the existing
   queue. Publish missing mirror images when resources become available.

No issue in this table is marked repaired by this reconciliation alone.

# Original Action Order diagnostic: Ihlen storage room

Hypothesis: an empty pending overlay can reuse this compact native five-object
row for diagnostic capture while preserving its source question and information gap.

Source: [original ESI JSON](../../../../ESI-Bench/dataset/json_clean/Action%20Sequencing/Action%20Order%20Inference/Ihlen_0_int/storage_room_0/Ihlen_0_int__storage_room_0.json).
The native row spans about 1.136 by 0.096 m between object centres. The original
constructor selects orbital view 0. This source differs from the existing Beechwood pilot.

1. Reference this source verbatim in `scene.yaml`, with a distinct scene ID,
   pending status, empty `original` patches, seed 0 and existing runtime defaults.
   Keep target, qualification and physical task absent.
2. Validate the full scene schema on CPU, original source bytes, native constructor
   inputs, default render configuration and exclusion of private source labels.
3. Commit the plan, scene, validator and branch research entry. Then create and
   separately commit `handoff.json` with that implementation revision pinned.
4. Return to Main for acceptance and immutable snapshot submission. Request the
   six existing global views plus head, wrist and scene views.

Assets, dimensions, materials, poses, source question and source seed remain
verbatim. Capture seed 0 is a diagnostic runtime input; it does not edit the source.
No simulator, robot episode or GPU process runs in this builder. Native initialization,
asset loading, framing, reach, grasp and stacking remain unverified. The diagnostic
provides no physical evaluator or success claim. Human views remain review evidence.

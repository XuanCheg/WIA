# Action Order Inference original-scene pilot

Hypothesis: a pending diagnostic overlay with no patches can reuse the selected
native ESI scene without changing its question, assets, dimensions, materials,
initial layout, or information gap.

Source: [original ESI JSON](/scratch/xinyangzhou/models/Show-Harness/ESI-Bench/dataset/json_clean/Action%20Sequencing/Action%20Order%20Inference/Beechwood_0_garden/dining_room_0/Beechwood_0_garden__dining_room_0.json).
The original source is read only; private labels and metadata stay in that source.

1. Create `scene.yaml` with a unique scene identity, pending status, the native
   source reference, seed 0, existing runtime defaults, and an empty `original`
   patch list. Leave the target, qualification, and physical task absent.
2. Add a CPU validator for source identity, unchanged overlay policy, manifest
   consistency, and exclusion of private source fields from public artifacts.
3. Record the CPU result in the branch research log. Commit the scene, plan,
   validator, and log; then pin that commit in a separately committed handoff.
4. Return the candidate to Main for static review and diagnostic capture. Request
   six global views plus head, wrist, and scene views using existing render defaults.

No simulator runs are performed by this builder. Initialization, physical reach,
grasping, and stacking remain unverified; this candidate supplies diagnostic
capture inputs and does not provide a physical evaluator or success claim.

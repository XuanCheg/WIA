# Partial Occlusion original-layout pilot

Hypothesis: the assigned native Partial Occlusion instance can enter the shared
capture pipeline through an unmodified scene overlay while preserving the
original assets, scales, materials, layout, question, and information gap.

The original source remains read-only:
[ESI source](</scratch/xinyangzhou/models/Show-Harness/ESI-Bench/dataset/json_clean/Perceptual Grounding/Partial Occlusion/Beechwood_0_garden/garden_0/Beechwood_0_garden__garden_0_0.json>).
The scene identity is `pilot_partial_occlusion_original_v1`. There is no separate
physical task or evaluator.

1. Review the assigned constructor, source structure, and existing scene runtime
   defaults. Preserve the native source and use only the `original` variant with
   an empty patch list.
2. Add a pending scene overlay with no target binding or qualification contract.
   Keep capture configuration separate from source and scene identities.
3. Validate identity, unchanged source, empty patches, runtime defaults, handoff
   consistency, and private-answer exclusion on CPU without simulator imports.
4. Commit the scene, this plan, validator, and research entry. Record that exact
   revision in a separately committed handoff for coordinator review.

Expected review images are `global_diagonal`, `global_overhead`, `global_front`,
`global_back`, `global_left`, `global_right`, `head`, `wrist`, and `scene`.
Review camera changes never become Agent observations.

This iteration provides a diagnostic candidate only. Native initialization and
framing, QA evaluation, and physical success remain unverified. No simulation
or Agent episode is run; the coordinator owns integration and capture.

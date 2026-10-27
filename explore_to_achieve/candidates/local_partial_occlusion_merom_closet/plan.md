# Merom closet Partial Occlusion diagnostic candidate

Hypothesis: the original compact closet instance can enter the shared capture
pipeline without changing its native assets, scales, materials, layout,
reference camera, question, or information gap.

Source: `Perceptual Grounding/Partial Occlusion/Merom_0_int/closet_0/Merom_0_int__closet_0_4.json`.
The source record is `1167`; the native question identity is
`partial_occlusion/Merom_0_int/closet_0/4`. The scene identity is
`local_partial_occlusion_merom_closet_scene_v1`, and the candidate identity is
`local_partial_occlusion_merom_closet_original_v1`. These identities differ from
the existing garden pilot. The original floor-support bounding box measures
approximately 1.765 by 0.980 m in XY; this is a source geometry measurement,
not a native room-clearance or robot-stance certificate.

1. Read the original scene and camera constructors, source metadata, shared
   nullable scene schema, capture handoff contract, and earlier research.
2. Add a pending diagnostic overlay with an empty `original` patch list,
   existing runtime defaults, and capture seed 0. Preserve the source seed
   separately by leaving the original JSON unchanged.
3. Run CPU checks for the full scene schema, committed source bytes, native
   object and camera reconstruction, runtime configurations, compact source
   geometry, and exclusion of private source information from authored content.
4. Commit the scene, plan, validator, and research record. Write and commit the
   handoff afterwards, pinning it to the actual first revision.

Requested human review views: `global_diagonal`, `global_overhead`,
`global_front`, `global_back`, `global_left`, `global_right`, `head`, `wrist`,
and `scene`. Review viewpoints never become Agent observations.

Native initialization, robot clearance, framing, and mounted-camera visibility
remain unverified. The original low floor layout and small room may be difficult
for the existing R1 initialization. No physical task, target binding,
qualification contract, evaluator, simulation, or Agent episode is introduced.
The coordinator owns integration and capture; this candidate remains pending.

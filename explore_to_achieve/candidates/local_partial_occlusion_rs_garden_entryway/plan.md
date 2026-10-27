# Rs garden entryway original-layout diagnostic candidate

Hypothesis: this independent original Partial Occlusion instance can replace
the failed Merom capture on its assigned GPU while preserving native source
assets, scale selection, materials, poses, reference camera, and question.
The failed Merom candidate and its evidence remain available for later repair.

Source: `Perceptual Grounding/Partial Occlusion/Rs_garden/entryway_0/Rs_garden__entryway_0_2.json`.
Original record: `1177`. Native question identity:
`partial_occlusion/Rs_garden/entryway_0/2`. Official ESI revision:
`9f6f15ed43e81654a97de9e4db2a1fedadca2bc1`. Original source SHA256:
`c3fcb1c534df7ed978eb9be97486fc143adbb4f3813492af28ab91d352adcc88`.
Candidate identity: `local_partial_occlusion_rs_garden_entryway_original_v1`.
Scene identity: `local_partial_occlusion_rs_garden_entryway_scene_v1`.
This is an independent source candidate with no parent repair identity.

1. Preserve the source verbatim and add a pending scene overlay with no patches,
   no physical target, no qualification or evaluator, existing runtime defaults,
   and capture seed 0. Leave the original source seed unchanged.
2. Check source bytes against the official commit, validate the complete
   SceneDefinition, compare both native object specifications and original
   applied-scale precedence, verify local model directories, and compare the
   native first-reference camera exactly with its saved source pose on CPU.
3. Commit the scene and this plan first. Pin the separate handoff to that real
   commit, validate it through the shared coordinator contract, and commit it.

Requested human review views are `global_diagonal`, `global_overhead`,
`global_front`, `global_back`, `global_left`, `global_right`, `head`, `wrist`,
and `scene`. Review views never become Agent observations.

The historical source metadata marks both task objects invisible in its first
reference view. Preserve that original camera; these flags are not new native
visibility evidence. Native initialization, robot clearance, framing,
mounted-camera visibility, QA, and physical success remain unverified. No GPU
runtime, simulation, model call, robot action, or capture queue submission is
performed by this builder. Main owns integration, capture, and the research log.

CPU observations: Behavior Python 3.10 passes the complete SceneDefinition,
exact official source hash and bytes, native record/question/source seed,
default runtime configuration, native two-object reconstruction, and original
scale precedence. The target retains the native default scale selection; the
occluder retains its saved applied scale. Both original model directories exist
under `/local_home2/zhouxinyang/data/behavior/behavior-1k-assets/objects`.
The native first-camera position and quaternion exactly match the source, and
all seven saved views remain present. Authored content excludes private answer
and target identities. No native GPU module is imported. These are CPU source
checks; they do not establish native loading, rendering, or task success.

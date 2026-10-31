# Release validation scope

## Initial 59-task export (superseded by the follow-up below)

CPU checks run on 2026-10-31:

- All 59 task definitions and 59 scene definitions loaded from a separate temporary extraction. Every selected evaluator and initialization adapter imported through the canonical WIA namespace.
- All 59 evaluation plans prepared without simulator or model calls. Catalogue aliases, selected defaults and the intentionally shared R-A1/R-A2 legacy ID were checked.
- All 64 required source identities were verified: 39 bundled authored inputs and 25 externally obtained ESI records. The T07 licensed native scene reference matched its preserved hash when the external asset directory was supplied.
- All 59 exported pairs were compared with their original definitions: no goal, scorer, numerical geometry, seed or budget changes. There were 197 metadata/resource path relocations across task and scene definitions; unselected preview variants were removed while the schema-required empty native baseline was preserved.
- The two extracted components' portable CPU suites passed 215 tests. The main workspace's relevant loader, release policy, initial-joint, inventory and evaluation-entry regression checks passed 89 tests. These are scoped checks, not the entire historical workspace suite.
- Editable installs were tested in a disposable virtual environment with existing host dependencies, no network and no modification of the active environment. The installed command listed 59 tasks and prepared the single-task and 59-task example plans. This does not claim an installation from an empty machine or fresh resolution of every optional dependency.

Reference dependency versions were Python 3.10, NumPy 1.26.4, SciPy 1.15.3, PyYAML 6.0.3, Pydantic 2.14.1, Pillow 11.0.0, Requests 2.34.2, pytest 9.1.1 and MCP 2.3.0. The existing native environment reports Torch 2.6.0+cu124 and OmniGibson 3.7.1; this release validation did not launch them. Those observations are not a new cross-platform simulator compatibility claim.

Each archive includes its own file checksums and manifest; the suite directory contains archive checksums. Repeat the checks with `scripts/verify_release_v02.py` and the supplied portable tests. That initial export preparation made zero model calls and launched zero GPU jobs. Existing native screenshot/static evidence is summarized separately in the dataset card. Successful physical execution of all 59 tasks remains unverified.

## Current 60-task follow-up

The updated catalogue adds T52 only; T15–T17 definitions and scenes are unchanged. A separate temporary extraction loaded all 60 task/scene pairs and prepared 60 evaluation plans, verified 65 source identities (39 bundled authored inputs and 26 external ESI records), and passed 216 portable CPU tests. The workspace passed 5 catalogue/resource checks and 56 review-publication checks. No policy or model calls were made.

T42–T44 were tested directly in the native simulator on physical GPU 4: three candidates, three repetitions per candidate, 3 simulation seconds per trial and a 0.15 s tie tolerance. All 27 trials were eligible. T42 always admitted the ball alone; T43 always admitted avocado and ball; T44 always admitted avocado and ball and additionally admitted the bowl in one repetition. The fixed-ball shortcut and T44 tie-boundary variation are documented and explicitly accepted by the user for this version. Object setters were used to set up trials; no robot manipulation or full-task success is claimed. The exact definition hashes and comparison evidence are in validation/ramp_validation.json.

T22/T46 chair-access concerns and further GF validation are closed by explicit human review. See validation/review_resolutions.json. The Chinese human audit and complete task/prompt appendices are included; they contain private scorer information and must stay outside benchmark policy inputs.

T52's final whiskey-stone/jar definition passed native zero-action verification on GPU 4: initial Inside was true, container contact and 15 consecutive stable steps were measured, final linear/angular speeds were zero, and `no_op_verified=true` with completion score 1.0. The original object geometry was retained; only an explicit safe robot starting pose was authored. This verifies the already-satisfied branch, not graspability or autonomous policy performance. A generic direct-furniture-support warning is inapplicable to the stone supported by its jar; the native predicate and contact were checked separately. Detailed evidence is in `validation/inside_control_validation.json`.

All 60 exported pairs were compared with the workspace definitions after the documented selection/path transformation: zero numerical, goal, scorer or default changes; 197 metadata/resource path relocations. The capture loader now permits explicit diagnostic input bundles without a catalogue without granting release membership; alias discovery and pending-task benchmark restrictions remain enforced.

T52 formal capture completed with 21 native views in 757.47 seconds (303.66 s initialization and 453.81 s screenshots), with zero model calls and robot actions. Top/detail images were inspected, and the exact task hash matches the native no-op check. The world was retained on GPU 4 without advancing physics; old retained worlds and other processes were preserved. Image provenance is summarized in `validation/inside_control_capture.json`; screenshot binaries and user feedback are not redistributed.

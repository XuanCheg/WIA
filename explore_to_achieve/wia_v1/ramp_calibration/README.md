# AP-3 pending ramp-material calibration candidates

These four path-selected task/scene pairs reuse T42 and vary one scalar material multiplier applied jointly to the original ramp static and dynamic friction coefficients. All geometry, poses, assets, spares, seeds, cameras, controllers, public wording, release/orientation/recovery rules, trial duration and tie tolerance remain fixed. Their nonzero coefficient ratio is fixed; the zero condition sets both coefficients to zero.

The task ID and version exposed by the full loader public payload are identical across all conditions. Scene IDs and condition paths are private. Select each task and scene by its exact path; the public task ID does not select a unique condition. The source identity remains unchanged. Definitions remain pending and require diagnostic-mode loading until qualified.

`manifest.private.json` is private experiment configuration. It records condition multipliers, actual coefficients, original paths and source hashes. Do not send it, condition paths or scene IDs to the policy. The numerical material fields in task private evaluation and scene initialization necessarily configure the simulator; they are not public task fields.

The hypothesis is that bowl sliding may overtake the rolling ball at lower ramp friction. No winner, winner reversal, admissible trial or legal robot success is asserted. Native material calibration, trial-release repeatability and complete legal robot execution remain pending. Changing the tie tolerance or selecting a winning condition is outside this generator.

Regenerate with the [builder](/local_home2/zhouxinyang/models/Show-Harness-Explore/scripts/build_wia_ramp_calibration.py): `python -B scripts/build_wia_ramp_calibration.py`. Append `--check` for a deterministic no-write comparison or use `--output-dir /tmp/wia-ramp-candidates` for a separate copy. Existing unrelated files are never deleted or overwritten. The generator performs no native, GPU or Agent experiments and leaves the original task, scene, source and WIA registry unchanged.

Run the [CPU bookkeeping tests](/local_home2/zhouxinyang/models/Show-Harness-Explore/tests/test_wia_ramp_calibration.py) with `python -B -m pytest -q -p no:cacheprovider tests/test_wia_ramp_calibration.py`. These checks validate the single-factor configuration and public-payload equality; they do not qualify physical outcomes.

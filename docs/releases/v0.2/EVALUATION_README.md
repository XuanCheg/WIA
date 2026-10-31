# WIA evaluation code 0.2.0

This cleaned source distribution contains WIA's task loader, private scorers, native scene/runtime interfaces, agent runner, review UI, runtime prompts and configurations. Historical dataset trees, one-off experiment configurations, outputs and private review feedback are excluded. Native authoring adapters and shared utilities remain because task initialization and scoring depend on them; inclusion of a utility is not inclusion of its historical dataset.

Use an editable install and the separate `wia-dataset-v0.2` and `ShowHarnessExplore-v0.2` archives. Set:

```bash
export WIA_ROOT=/absolute/path/to/wia-evaluation-v0.2
export WIA_DATASET_ROOT=/absolute/path/to/wia-dataset-v0.2
export SHOWHARNESS_ROOT=/absolute/path/to/ShowHarnessExplore-v0.2
export WIA_ESI_ROOT=/absolute/path/to/pinned/ESI
python -m pip install -e "$SHOWHARNESS_ROOT[harness,simulator]"
python -m pip install -e "$WIA_ROOT[simulator,test]"
wia tasks
wia eval --config "$WIA_ROOT/examples/one_task.yaml" --dry-run
```

The suite README supplies the exact ESI commit and setup commands. Set `OMNIGIBSON_DATA_PATH` in a working native simulator environment before actual execution. Select an available physical GPU explicitly with `environment.gpu` in a copied configuration. `CUDA_VISIBLE_DEVICES` mapping is recorded by the launcher. Never displace other jobs to satisfy a config. No GPU is needed for listing or dry runs.

Edit a copy of an example to provide an explicit model and optional simulator interpreter. `suite.yaml` selects all 60 aliases and preserves per-task defaults. An omitted variant now follows the catalogue selection. Config `resource_root` is optional; aliases resolve through the dataset catalogue. Output directories must be fresh. Backend choices include Codex CLI, Chat Completions, Responses and local HTTP. API credentials are read from environment variables and are never included in this package. Optional backend dependencies are needed only for the chosen backend. Model/provider experiments in the development workspace are not release presets.

Run the portable CPU tests with `python -m pytest -q "$WIA_ROOT/tests"`. Run `scripts/verify_release_v02.py --help` for integrity, relocation, source-hash and 60-task planning checks. An optional `--native-data` verifies the T07 licensed scene reference without launching the simulator. The private scorer and source answers must stay outside policy observations. Existing pending statuses and frozen-load semantics are preserved; see `KNOWN_ISSUES.md` before interpreting success rates.

Use `wia debug --config FILE` for explicit retained-world debugging. Batch evaluation has its own normal teardown policy; release export itself starts or stops no simulator process.

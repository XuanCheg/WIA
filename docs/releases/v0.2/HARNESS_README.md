# ShowHarnessExplore — WIA v0.2 suite snapshot

ShowHarness supplies model sessions, semantic robot actions, observations and recording. Dataset adapters own task resources, native scene lifecycle, scoring and episode loops. This package is a cleaned snapshot for the WIA v0.2 suite; its Python package name remains `showharness` and its component API version remains 0.1.0.

```bash
python -m pip install -e '.[harness,simulator]'
python -m pytest -q tests
```

Install the separate WIA evaluation package to register the optional `wia` dataset adapter. Harness does not import WIA unless explicitly selected. The two editable source installs and the separate dataset directory are required for WIA evaluation. Use `wia eval --config FILE --dry-run` to prepare a plan without a simulator or model call.

```python
from showharness.datasets import evaluate_dataset
summary = evaluate_dataset("wia", plan)  # A plan prepared by WIA.
```

`core/` contains shared session/action/observation logic; `showharness/` contains the public dataset interface; `gumi/`, `interpreters/` and `plugins/` contain control and extension components. Public robot configuration templates, prompts, requirements and supporting scripts are retained. Use only the example site templates; no local credentials or site configuration are included. Training helper source is retained without models, datasets, checkpoints or outputs. Optional robot/simulator demonstrations require their own assets and dependencies; generated Isaac/Franka USD assets and demo videos are not bundled.

WIA-specific task definitions, ESI native runners and private scorers live in the other two release components. Historical research directories and task-edit scripts removed during the WIA split are not reintroduced. The component manifest records this exact working-tree snapshot and its base commit. Original Apache-2.0 licensing and citation metadata are preserved; see `LICENSE`, `CITATION.cff` and `THIRD_PARTY.md`.

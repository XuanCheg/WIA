# WIA v0.2 release suite

Three independent source archives form this release:

| Archive | Contents |
| --- | --- |
| `wia-dataset-v0.2.tar.gz` | 60 selected task/scene pairs, authored source inputs, taxonomy, provenance, dataset card and qualification limits |
| `wia-evaluation-v0.2.tar.gz` | WIA 0.2.0 evaluator, native runtime, scoring, prompts, runtime configurations and portable CPU tests |
| `ShowHarnessExplore-v0.2.tar.gz` | ShowHarnessExplore source snapshot providing sessions, actions, observations and recording; component API version remains 0.1.0 |

Extract all three next to one another. Verify the archives with `sha256sum -c SHA256SUMS`. Each component also contains its file manifest and checksums. Use editable source installs; these archives include runtime files that a stripped Python wheel does not contain.

```bash
python -m pip install -e './ShowHarnessExplore-v0.2[harness,simulator]'
python -m pip install -e './wia-evaluation-v0.2[simulator,test]'
export WIA_ROOT="$(pwd)/wia-evaluation-v0.2"
export WIA_DATASET_ROOT="$(pwd)/wia-dataset-v0.2"
export SHOWHARNESS_ROOT="$(pwd)/ShowHarnessExplore-v0.2"
```

Obtain the external ESI dependency separately. The pinned source checkout is required even for loading upstream source records; no ESI source files or native assets are redistributed in these archives.

```bash
git clone https://github.com/ESI-Bench/ESI-Bench ESI
git -C ESI checkout 9f6f15ed43e81654a97de9e4db2a1fedadca2bc1
export WIA_ESI_ROOT="$(pwd)/ESI"
wia tasks
python wia-evaluation-v0.2/scripts/verify_release_v02.py \
  --dataset "$WIA_DATASET_ROOT" --evaluation "$WIA_ROOT" \
  --harness "$SHOWHARNESS_ROOT" --esi "$WIA_ESI_ROOT"
```

For simulation, use a working BEHAVIOR/OmniGibson environment with separately obtained assets and set `OMNIGIBSON_DATA_PATH` to its data directory. The light Python extras do not install the native simulator or grant access to its assets. The environment used for the existing native captures was Python 3.10 with OmniGibson; CPU verification does not certify other native installations. Install both editable packages into the simulator interpreter used by launch mode.

Set an explicit model in the example config before an actual evaluation. Dry runs do not contact a model or launch a simulator:

```bash
wia eval --config wia-evaluation-v0.2/examples/one_task.yaml --dry-run
wia eval --config wia-evaluation-v0.2/examples/suite.yaml --dry-run
python -m pytest -q wia-evaluation-v0.2/tests
python -m pytest -q ShowHarnessExplore-v0.2/tests
```

This is a release of selected research definitions and their implementation, not a claim that all 60 tasks have successful physical reference executions. Read `DATASET_CARD.md` and `KNOWN_ISSUES.md` in the dataset archive before interpreting results. Private evaluators and source answers belong to the environment process; never expose the entire dataset folder to a benchmark policy as its observation.

No extension tasks, retired D tasks, private review feedback, model transcripts, credentials, recordings or large simulator caches are included. The original workspaces and live simulator processes are unaffected by export. This local release suite has not been uploaded to a public registry.

The Chinese protocol audit is included as `PROTOCOL_REVIEW.zh-CN.md` in the dataset and evaluation components. It documents actual inputs, prompt assembly, tools, history policies, every selected scorer and the distinction between per-task metrics and proposed dataset totals.

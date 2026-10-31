# WIA dataset v0.2

This independent data package freezes 60 active task/scene pairs. It is used by the separate WIA 0.2.0 evaluation package and the ShowHarnessExplore v0.2 suite snapshot.

Set `WIA_DATASET_ROOT` to this extracted directory. `catalogue.json` is the sole active membership list; `tasks.csv` provides a compact index. Tasks can be selected by alias, a unique task ID or an exact YAML path. R-A1/R-A2 share a legacy ID and require aliases or paths. Omitted variants use the catalogue selection. Definitions retain the selected initialization plus the schema-required empty `original` baseline; the baseline is not an additional selected benchmark instance. Unselected preview variants are removed.

- `tasks/`: public objectives plus private environment/scoring definitions.
- `scenes/`: the corresponding initialization-only definitions.
- `sources/json_clean/`: required project-authored source inputs.
- `dependencies.json`: exact hashes and the pinned ESI dependency for upstream inputs obtained separately.
- `provenance/definitions.json`: original and exported definition hashes and the permitted packaging transformations.
- `validation/`: a compact record of the four household scenes' existing bounded static evidence.

Definitions keep their stable IDs, public goals, seeds, budgets, numerical geometry and scorer settings. Formatting changes, removal of unselected preview variants, and relocation of metadata paths change file hashes; original and exported hashes are recorded separately. `provenance://` references label historical research inputs that are not runtime files. `behavior://` references resolve against `OMNIGIBSON_DATA_PATH`; the T07 native scene hash remains mandatory.

All 60 task definitions retain their existing `pending` status. Explicit release membership allows evaluation under the frozen-load policy, which does not run physical requalification or infer safety. Unknown physical success is not converted into a passing certificate. See `DATASET_CARD.md`, `KNOWN_ISSUES.md`, `THIRD_PARTY.md` and the suite README for installation and verification.

# WIA-v1 release catalogue

This directory freezes the current evaluation campaign's preferred variants from the reviewed snapshot and review evidence. It contains 61 task/scene pairs. AP4 remains excluded because it has no executable definition. Original task IDs, statuses, goals, seeds, budgets, physical configuration and scoring are preserved; only converter module identities use the canonical WIA namespace. Selection is not a claim of native execution qualification.

Use `wia tasks` or an exact YAML path in this directory. `catalogue.json` records each source variant and the source-definition hashes. Historical source/review files remain unchanged. Old CoreV2 registries and old converter/import namespaces are not runtime compatibility interfaces.

For evidence archives, previews are lossy and exact model images are referenced by `model_inputs.json`. Numeric `.npy.gz` artifacts decompress to the original NPY bytes. Copy the whole episode including `artifacts/` to preserve relative references.

# Project-authored source inputs

These JSON files belong to Show-Harness-Explore. ESI-Bench is an unchanged
upstream submodule; never write authored tasks or generated sources into it.

`json_clean/` preserves the five-part category/subcategory/scene/room/file
identity used by task definitions. The resolver searches the upstream corpus
first, then this directory. An explicit `data_root` selects only that root,
including immutable capture snapshots; it never falls back to local sources.

The initial migration contains six temporal change sources and six ramp-source
revisions. Contents were copied byte for byte; task identities, source hashes,
definitions and historical receipts remain unchanged. The source manifest
records their repository paths and hashes. Historical absolute paths under
ESI-Bench can resolve to these migrated files, with the caller still checking
the original recorded hash. This is path relocation, not new native validation.

Builders must write future authored inputs here. When changing an input's
contents, create a new revision and collect new evidence; do not overwrite
sources bound to published captures.

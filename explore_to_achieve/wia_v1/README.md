# WIA V1 Stage 0 proposal overlay

This registry records 61 proposed WIA rows: 48 existing selected task definitions and 13 unbuilt designs. The 51-entry CoreV2 selection and all task/scene definitions remain unchanged. Three existing dimensional-size tasks remain available separately as `legacy_controls`. AP-4 is a design gap outside the 61 rows. Family and subclass assignments, tags and roles are provisional; they are not validated capability claims or release qualifications.

Regenerate from the repository root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B scripts/build_wia_registry.py
```

Append `--check` to compare exact bytes without writing, or `--output /tmp/wia-registry.json` to create a separate result. The CLI rejects an existing output with an unrelated schema. Generation uses no simulator, GPU, network, receipt traversal or installation. Input hashes bind the five small source inputs; there is no broad artifact hashing. Output is deterministic and omits generation timestamps.

The [builder](/local_home2/zhouxinyang/models/Show-Harness-Explore/core/tasks/wia_registry.py) combines the [61-row recheck](/local_home2/zhouxinyang/models/Show-Harness-Explore/docs/research/world_interactive_agency_v1/task_recheck_61.csv), [reconciled inventory](/local_home2/zhouxinyang/models/Show-Harness-Explore/docs/research/world_interactive_agency_v1/inventory_reconciliation.json), [proposed taxonomy](/local_home2/zhouxinyang/models/Show-Harness-Explore/docs/research/world_interactive_agency_v1/source_documents/02_Taxonomy/World_Interactive_Agency_Taxonomy_v1.yaml), [CoreV2 selection](/local_home2/zhouxinyang/models/Show-Harness-Explore/explore_to_achieve/lists/corev2.json), and [historical source suggestions](/local_home2/zhouxinyang/models/Show-Harness-Explore/docs/research/world_interactive_agency_v1/source_documents/06_Independent_Review/World_Interactive_Agency_V1_61_Task_Review.csv). It validates complete, unique mappings and copies the five declared family migrations. Source identities resolve through the existing bounded source resolver. The JSON uses repository-relative definition/source paths; historical evidence pointer strings remain verbatim and are not opened.

Each implemented row preserves task and selected-scene IDs, versions and paths; source ESI categories; evaluator configuration and explicit version fields; and explicitly declared capabilities. Missing capabilities remain unknown rather than being invented for rigid tasks. Current source metadata provides the declared scene/workspace, while actual native workspace geometry remains unknown. Explicit authored parent sources preserve provenance when apparatuses moved. Unbuilt rows have no actual task, scene or evaluator; source URLs and scene suggestions are kept separately.

The public observation contract retains all eight mounted RGB views and the initial overview frozen at episode start. This is an interface description, not a visibility audit. It does not establish which objects, relations or answers an agent can see. The unbuilt rows describe the intended contract only.

Engineering reuse is split across scene, converter, scorer and evidence. Historical R0/R1/R2/N1/N2 labels are retained as provenance and do not determine evidence maturity. In particular, T22–T25 are loader-ready rigid definitions with no private evaluation: their scorer status is explicitly `not_configured`.

The five evidence dimensions are independent: `human_layout_review`, `native_initialization`, `physical_scorer`, `reference_execution`, and `scientific_controls`. Current registry receipt pointers and recorded initialization claims are preserved without revalidating their compatibility with current definitions. Evidence statuses remain `not_assessed` or `inconclusive`. Screenshots, initialization acceptance, setup acceptance and passive physics probes do not establish complete robot action. This registry claims no simulator or agent experiments and is not publication-ready.

Validate the bookkeeping:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B -m pytest -q tests/test_wia_registry.py
```

The tests check disjoint 61-row membership, family counts AP/NT/GF/SR/GG/TM = 10/8/9/12/14/8, preservation of all selected paths and the three controls, the rigid scorer gap, unbuilt/source separation, relocation provenance, mapping failures, deterministic regeneration and no-write checking. Passing these checks does not upgrade any scientific or execution evidence status.

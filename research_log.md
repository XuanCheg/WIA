# Research Log

## 2026-10-27 — Independent WIA repository bootstrap

**Hypothesis.** An independent WIA repository with explicit source and dependency provenance can separate benchmark ownership while keeping the extraction traceable and preserving the original Harness history and artifacts.

**Evidence and changes.** The source Harness checkout was at `8ff14cd3683079498949cfe95534cd25f486b496`. Its ESI-Bench checkout was clean at `9f6f15ed43e81654a97de9e4db2a1fedadca2bc1`, with upstream `https://github.com/ESI-Bench/ESI-Bench`. WIA is initialized with a new `main` branch. ESI is an independent no-hardlinks checkout, registered as the `ESI` submodule at that exact commit. `provenance/repository.json` records the extraction date and source revisions; `provenance/extraction.json` retains the file mapping.

**Assessment.** This is repository and engineering setup, not a model-quality experiment. No simulator, model, or benchmark evaluation was run for this bootstrap, and no result or physical qualification is claimed. Original Harness history and raw artifacts remain in the source repository. The bootstrap and provenance records are retained.

## 2026-10-27 — ESI checkout ownership migration

**Hypothesis.** Making WIA's pinned ESI submodule the active checkout while retaining the former Harness path as a compatibility link preserves existing resource paths and removes the duplicate active checkout.

**Evidence and changes.** The clean ESI checkout at `9f6f15ed43e81654a97de9e4db2a1fedadca2bc1` is now the WIA `ESI` submodule. The old Harness worktree is archived at `/local_home2/zhouxinyang/models/Show-Harness-Explore/.git/legacy-esi-checkout-9f6f15ed`; its original module gitdir remains at `/local_home2/zhouxinyang/models/Show-Harness-Explore/.git/modules/ESI-Bench`, now attached to the archived worktree. The existing H path is a relative `../WIA/ESI` symlink, and `provenance/repository.json` records the compatibility path and source revisions. Only H `.gitmodules` and the symlink are staged.

**Assessment.** The archived checkout remains clean at the same ESI commit as WIA. No source-file contents, benchmark data, simulator assets, or experiment artifacts were modified or deleted; the former checkout directory was moved intact to the archive path. No model, API, GPU, or benchmark run was performed. The repository-path migration is retained.

## 2026-10-27 — Harness/WIA implementation assessment and validation limits

The refactor retains 103 migrated benchmark modules and 45 pure authoring commands, compatibility modules/source symlinks, an independent WIA package with pinned ESI and optional Harness integration, separate Harness-managed HTTP Chat Completions/Responses histories and native Codex history, and distinct backend/model configuration. Policy/system contracts describe future WorldModel/WAM support; neither is implemented, and Claude remains a future backend concept.

Main validation reports are separate: the comprehensive Harness/WIA suite passed 224 tests in 138.37 s; the usage/log metadata follow-up passed 70 in 2.81 s; legacy-root authoring suites passed 413 with two exact-artifact failures; the canonical default standalone suite passed 10. Overlapping suites are not summed. Separate editable environments passed WIA-without-Harness and Harness-without-WIA import checks; the old native, agent and launcher help commands passed after ESI path conversion. Configs/manifests match the source Harness revision. The two exact-artifact failures reflect pre-existing hash mismatches: `esi_arm` current `dbc7c57727aa8fac4e81419179e94e7ae0dd054c286e000a2c5d48a023b23c8e` versus posture manifest `04f66fcc723fff2ee9b76d669822ddad2f7295de7483765276db528d603cb8f3`, and `esi_trunk` current `57216b57800d95e0e4fc9b764a433be5702afd75db2bc5cc7ad522126fef2e99` versus temporal manifest `56924f308bf3a5a0fe3ba57af79852871d97f314108f7f3989e189e28bc36586`. No historical artifacts were rewritten. At this checkpoint, endpoint-default correction tests were pending; the final follow-up below records their passing result. No model call, GPU run or native simulation was performed, and no existing process was signaled. Full evidence and source revisions are summarized in `provenance/validation/acceptance.json`.

## 2026-10-27 — Final endpoint and credential integration follow-up

The main agent's final endpoint/credential integration check, using the focused command plus optional_harness, passed 95 tests in 2.99 s. A vLLM backend dry-run from `/tmp` also passed without creating an output directory or using a model/GPU. These overlapping suites remain separate from the comprehensive 224-test run and the 70-test usage/log metadata follow-up; their counts are not summed. Final review accepted the core refactor. This does not qualify model quality or native physics: no actual model call, GPU run, or native simulation was performed, and no existing process was signaled. See `provenance/validation/acceptance.json` for the recorded source revisions and check details.

The scoped staged whitespace check for `src`, `tests`, `configs` and `scripts` passed after removing only surplus terminal blank lines from `src/wia/authoring/io.py` and `src/wia/authoring/scene_inventory.py`. A full-tree check still reports trailing spaces in copied source Markdown/CSV; those raw source artifacts were preserved verbatim. See `provenance/validation/acceptance.json`.

# Scene and task catalogue

Scene descriptions and task definitions have separate roots. Both use the ESI
hierarchy: `Big Task/Small Task/Scene/Room/sample`.

- `scenes/` contains verbatim ESI JSON copies and matching YAML scene descriptors.
- `tasks/` contains all 27 approved Rigid Containment task definitions: 17
  single-object tasks and 10 merged multi-object tasks. `tasks/catalog.json`
  lists their scene references, original questions and current approval hashes.
  Incomplete generated tasks remain pending and are skipped at load time.
- `temp/20261012T111915Z/` preserves the previous scene and task directories.
- `prompts/` retains the shared Agent instructions.

The import initially marked descriptors `status: pending`, except the two
discarded corrupted Rigid Containment records. Later native conversion publishes
selected canonical descriptions as `ready`; `scenes/catalog.json` records the
current counts, verification level, failure reasons and manual-review flags.
`core.tasks.loader.discover_scenes()` selects ready definitions by default;
`include_pending=True` includes pending records for review. Discarded records
require a separate `include_discarded=True` review opt-in. `load_scene()` rejects
pending and discarded descriptors before source loading or simulation.

The CPU [batch scene review gallery](/scratch/xinyangzhou/models/Show-Harness/docs/esi_scene_review.md)
shows exact final-layout previews and records human decisions separately from
conversion status. Start `scripts/review_esi_scenes.py` with a conversion run root
and review output directory. Original questions remain unchanged; missing or
mismatched final captures are shown explicitly. The
[merged containment task workflow](/scratch/xinyangzhou/models/Show-Harness/docs/esi_rigid_multi_tasks.md)
adds ten tasks alongside the seventeen approved base layouts, with a separately
referenced scene and one Object camera per requested target.
All twenty-seven approved layouts now have a task in the matching
`Big Task/Small Task/Scene/Room/` directory. Each task keeps its original ESI
question and requires real pickup and partial insertion for its one, two or
three targets, with a default 60-step budget. Approved scene descriptors remain
unchanged during task registration.

The source corpus includes 3,502 sample JSONs and 10 aggregate index JSONs. The
aggregate indexes retain their original root filenames and are not separate
scene entries. The scene catalogue contains 3,514 YAML descriptors:
3,502 matching source samples, two retained physical variants, and ten additional
multi-object layouts. Twenty-seven descriptors are ready; 3,485 are pending and
two corrupted records are discarded. The catalogue index is
`scenes/catalog.json`. Ready and pending counts change during conversion; the
two corrupted records remain discarded. Raw source copies remain unchanged for
reproducibility.

Canonical Rigid Containment conversions may replace their original pickup poses
with screened same-room support placements.
The additional scaled-football and native-head coffee-table variants have
`__tabletop.yaml` and `__coffee_table_fetch_native_head.yaml` suffixes. Their
source JSON references are unchanged, and they also require pending review.
Original conversion and experiment artifacts remain at their previous paths.

An unconverted descriptor has this form:

```yaml
kind: scene_import
version: 1
status: pending
source:
  provider: esi
  json: Physical Structure/Rigid Containment/Scene/Room/sample.json
```

Complete and review a valid `scene_patch` definition before marking it `ready`.
Changing only the status does not supply missing conversion fields. Native
physical qualification remains a separate startup check; conversion does not
certify graspability, containment or task success. Ready definitions in this
catalogue resolve their source JSON from the local `scenes/` copy by default.

Original JSON copies include raw questions, answers and construction metadata
for reproducibility. They remain private construction data. Later task files
should expose the original question, its public rules and any explicitly added
goal without exposing source answers or hidden geometry to the Agent.

Legacy YAML examples used by CPU tests are isolated under
`tests/fixtures/task_examples/`; they are not active tasks and do not depend on
the temporary archive.

The Rigid Containment converter can be inspected without writing scene files:

```bash
.venv/bin/python scripts/generate_esi_tabletop.py \
  --source-dir "explore_to_achieve/scenes/Physical Structure/Rigid Containment" \
  --data-root explore_to_achieve/scenes --dry-run
```

Batch conversion selects canonical questions only, as requested. Task variants
are counted as `ignored_task_variants`, rather than conversion failures. The two
corrupted protein-powder records under `Benevolence_2_int/bedroom_0` are marked
`discarded` and omitted before conversion; their source target Z is about
-132.19 m. A catalogue preview therefore selects 19 canonical candidates,
omits 59 other question variants, and discards two records. It writes no scene
files. All candidates still require native qualification.

The lower-level adapter and explicit `--source` previews retain verified
canonical construction reuse for future variant support. Original variant
question provenance is separate, and those drafts stay pending. Original
upstream task preprocessing remains unchanged. Invalid pose/height records
outside the copied catalogue are still rejected.

New converter defaults use zero extra Head pitch and the qualified
`configs/esi_trunk_low_grasp.json` profile. Changing converter defaults alone
does not regenerate existing descriptors.

`scripts/sample_esi_supports.py` automates an approximate support-placement
screen. It selects the normal canonical questions and samples native tables,
shelves, counters, cabinets, desks and benches in the source room. Local native
raycasts distinguish shelf layers and check a padded target footprint; physics
settling checks support contact and stability. A room-restricted navigation grid
combines the native traversability map with current object bounds and measured
physical chassis footprint clearance. A nearby free stance must exist in the
same room. Route connectivity from the unchanged initial robot pose is recorded
when verified; default coarse screening also retains stances whose connectivity
the conservative grid cannot confirm. Add `--require-connected-route` to reject
these cases. This is an approximate screen: it executes no
navigation or grasp commands and does not certify arm clearance or grasp success.
The initial container work table is excluded from pickup proposals. When native
surfaces fail, the sampler may draft a separate target-only support in free
same-room floor space at least 1.25 m from the initial robot. Added-support
drafts remain unaccepted until an independent native restart passes. A qualified
existing off-table pickup pose can be retained instead of moved again.

Preview without writing files:

```bash
.venv/bin/python scripts/sample_esi_supports.py --dry-run
```

Run sequentially inside the retained interactive GPU shell, after activating
the existing simulation environment:

```bash
python scripts/sample_esi_supports.py \
  --output-dir /scratch/projects/CFP04/CFP04-CF-047/xinyangzhou/show-harness/outputs/support_sampling/RUN \
  --seed 0 --attempts 12 --placements 3
```

Use `--limit 1` for a smoke test. `--source SAMPLE.json --base-scene FILE.yaml
--variant NAME` reuses an existing tested initialization instead of generating
the standard tabletop initialization. Each worker runs in a separate process;
workers run sequentially in the same GPU allocation, with per-scene time bounds.
Existing outputs are preserved on rerun; use a fresh output directory to retry.
Failures are recorded
without aborting later scenes. Existing gateway port forwarding is independent
of this offline tool.

Outputs retain the ESI hierarchy with a `seed_N/` subdirectory. Each accepted
placement has a separate pending scene YAML and private `support_bank.json`
evidence, including a suggested stance and approximate route. `batch_summary.json`
records the overall result. No active catalogue entry or task file is published.
Source JSON/question text and container identities/scales remain unchanged.
Oversized targets are uniformly shrunk toward a minimum source extent of 0.08 m
by default, preserving the declared support gap. `--keep-target-scale` disables
shrinking; `--grasp-width-m` changes its width within the measured 0.10 m aperture.
The original QA answer is not revalidated after scaling. Measured native extent
over 100 mm still produces an aperture flag rather than a grasp verdict.
Use a different seed to create another reproducible placement bank. Select one
of its scene YAMLs when defining a later task; do not disclose private stance,
route or placement metadata to the Agent.

The optional scene qualification mode `support_surface_mode: local_raycast`
checks the actual layer beneath an object. Existing scene definitions retain
their `aabb_top` default. The sampler writes the local mode for its pending
outputs so future initialization can verify an interior shelf layer correctly.

`scripts/convert_esi_rigid_catalog.py` applies this workflow to the active copied
catalogue. It samples one pickup placement per canonical source. The default
`--validation-mode restart` reloads that exact placement in a fresh native
process before publication. The faster `--validation-mode screen` publishes
after one completed native initialization and support/nearby-stance screen for
native furniture candidates,
with `manual_review_required: true` and no independent restart claim. The
approximate reachability criterion still applies; no real grasp is attempted.
Added furniture always receives an independent restart and target-contact/
nearby-stance checks before publication, including in screen mode.
Failed sources remain pending with their stage and reason in `catalog.json`.
The original descriptors/index are backed up under `temp/` before publication.
Raw JSONs, question variants, retained physical variants and tasks are unchanged.

Run in the retained interactive GPU shell:

```bash
python scripts/convert_esi_rigid_catalog.py \
  --output-dir /scratch/projects/CFP04/CFP04-CF-047/xinyangzhou/show-harness/outputs/rigid_conversion/RUN \
  --review-file /absolute/path/to/scene_review/RUN/reviews.json \
  --seed 0 --attempts 12 --validation-mode screen
```

`conversion_summary.json` tracks the current source/stage and completed results;
each source has sampling evidence and optional restart-validation evidence.
Use `--resume` with the same directory, allocation and sampling settings to retain
completed results after an interruption; incomplete worker evidence is archived
before retrying. A placement
with an aperture warning or unconfirmed route connectivity retains those flags
when published. Ready indicates a converted scene that passes startup checks,
not verified grasp/containment success.

Pass `--review-file` to preserve fingerprint-matched Approved layouts and omit
manually skipped scenes. Missing-object notes remain human evidence and do not
alone establish construction failure. Previous conversion artifacts retain
their accepted initial-table fallback records; the updated converter excludes
unapproved initial-table fallbacks. Approved layouts remain unchanged and are
included in the new run's preserved-scenes list. See the
[review-driven pickup policy](/scratch/xinyangzhou/models/Show-Harness/docs/esi_pickup_generation.md)
for resizing, separate-support construction and review provenance.

For slow loading, `--initialization-wall-budget-s` changes the native startup
allowance independently of `--wall-budget-s`, which bounds support sampling.
The default startup allowance is 300 seconds for ordinary scenes, with a separate
1,800-second conference allowance via `--slow-initialization-wall-budget-s`.
A timeout is a bounded
failure, not evidence that the scene or its target is physically impossible.

[diagnose_esi_rigid_scene.py](/scratch/xinyangzhou/models/Show-Harness/scripts/diagnose_esi_rigid_scene.py)
reuses native initialization and preflight, recording periodic Python stacks,
asset preparation timings, task-object states before/after settling and trunk
initialization, and contact changes during trunk motion. Tracing adds no
simulation steps, explicit navigation/grasp actions or model calls. Immediate post-teleport AABBs
and contacts can refer to the previous physics frame; settled snapshots and the
final qualification are the physical evidence. A zero process exit code alone
does not mean preflight passed; require `validation_complete.json` and accepted
`scene_status.json`.

The optional scene runtime field `post_controller_settle_steps` requests native
physics settling after controller initialization and before the first capture
and preflight. Its default is zero. This changes no poses or physical guards and
can resolve a small residual velocity after robot startup. Converted descriptions
are validated with `--variant tabletop`; their `original` variant is separate.

[run_esi_rigid_diagnostics.py](/scratch/xinyangzhou/models/Show-Harness/scripts/run_esi_rigid_diagnostics.py)
executes an explicit JSON list of diagnostic cases sequentially in the specified
interactive allocation. Each case supplies `name`, `base_scene`, `output_dir`
and `wall_budget_s`. It preserves completed receipts, bounds each worker and
never publishes or releases an allocation.

[propose_esi_rigid_repair.py](/scratch/xinyangzhou/models/Show-Harness/scripts/propose_esi_rigid_repair.py)
uses a native obstacle/floor snapshot to propose a joint robot/table relocation
within the source room. Original scales, bindings, primitive dimensions and
relative task-object layout remain unchanged. An original room-map supplement
can refine the coarse native room grid for proposals without modifying the
environment. `--dense-base-grid` searches nearby base positions at 0.2 m spacing;
`--robot-reference` accepts a contact-free native R1 snapshot when the failed
spawn has deformed its arms. Floor heights and reachability remain estimates.
Every retained correction requires fresh native loading, actual leg-floor
raycasts, stable object support and contact checks for the exact candidate.
Unresolved cases stay pending with all previous failures preserved.

The diagnostic `--cache-assets` flag is an experimental, process-local reuse of
unchanged prepared USDs and MD5 values. It preserves native hash recording and
mismatch warnings, invalidates changed files and leaves the installed SDK
unchanged. It is disabled in the production converter; asset reuse alone does
not remove mesh/collision initialization costs.

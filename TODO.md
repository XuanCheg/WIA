# Model evaluation TODO

Use Vercel AI Gateway with a singleton official-provider `only` allowlist on every request. Do not configure model fallbacks. Exceptions require an explicit user instruction. WIA's `provider: compatible` selects a transport; it does not enforce the upstream provider.

Reasoning policy (updated 2026-10-31): explicitly set reasoning effort to `medium` for every model that supports that value, including debug runs. This user-selected policy overrides provider defaults. Keep the existing output caps unchanged; do not substitute `max` or invent a numeric thinking budget for models without an effort control.

This policy supersedes the earlier `high` requirement for future runs across all evaluation and debug models. Historical run records retain their actual settings; editing this TODO does not change already-running processes.

- [ ] Map the `medium` reasoning policy to each model's supported Gateway request fields and verify it is forwarded to the official provider before evaluations. Record the requested effort and effective request parameters. If `medium` is unsupported or rejected, report the incompatibility rather than silently falling back to another effort or the provider default. Models without an effort control should be recorded as not applicable.

| Status | Model ID | Required gateway.only | Role |
| --- | --- | --- | --- |
| Pending | openai/gpt-6-astra | openai | Evaluation |
| Pending | openai/gpt-6.1-sol | openai | Evaluation |
| Pending | anthropic/claude-fable-5.1 | anthropic | Evaluation |
| Pending | anthropic/claude-opus-5.5 | anthropic | Evaluation |
| Pending | deepseek/deepseek-v4.1-flash | deepseek | Debug and evaluation |
| Pending | moonshotai/kimi-k3.1-pro | moonshotai | Evaluation |

- [ ] Complete one successful CPU-only multimodal request using DeepSeek V4.1 Flash through the official DeepSeek endpoint. Attempted 2026-10-30: HTTP 403; Gateway reports that free-tier users cannot access this model. Paid Gateway credits are required before retrying. Evidence: outputs/vercel_multimodal/20261030T161818/response.json.
- [ ] Pass the official-provider restriction through the WIA transport before full evaluations.
- [ ] Define controlled evaluation settings and record model-specific parameter differences.
- [ ] Add the remaining manufacturers/models when specified by the user.

DeepSeek V4 Pro is excluded at the user's request. The current list contains six models from four manufacturers.

- [x] Temporary Qwen3.5 Plus multimodal connectivity test (2026-10-30): official alibaba endpoint, HTTP 200, correct visual answer, 5.98 s, USD 0.0005676. Evidence: outputs/vercel_multimodal/20261030T162127. This does not add Qwen to the evaluation roster or replace the default debug model.

## Prompt caching requirements — added 2026-10-30

Enable supported prompt caching for every evaluation model and debug run while preserving the official-provider allowlist, full observation/history content, medium reasoning policy and output/input budgets. A cache discount does not reduce context-window token usage. Do not reuse cached model responses or actions in place of fresh inference.

| Model | Required caching handling | Verification status |
| --- | --- | --- |
| openai/gpt-6-astra | Use provider implicit prefix caching; verify this exact model and Gateway route | Pending actual hit receipt |
| openai/gpt-6.1-sol | Use provider implicit prefix caching; verify this exact model and Gateway route | Pending actual hit receipt |
| anthropic/claude-fable-5.1 | Enable Gateway `caching: auto` for explicit cache markers | Pending cache-write and cache-read receipts |
| anthropic/claude-opus-5.5 | Enable Gateway `caching: auto` for explicit cache markers | Pending cache-write and cache-read receipts |
| deepseek/deepseek-v4.1-flash | Use official-provider implicit prefix caching, including default debug runs | Pending actual hit receipt; existing free-credit access restriction remains |
| moonshotai/kimi-k3.1-pro | Verify exact model's official caching mode and Gateway forwarding; enable supported mode | Pending documentation/parameter mapping and actual hit receipt |
| alibaba/qwen3.5-plus (temporary tests only) | Enable Gateway `caching: auto` for explicit cache markers | Verified independent multimodal probe: 3,721 tokens written, then 3,721/3,727 input tokens cached; T07 actual episode verified: cumulative 78.6% cache reads; stopped at decision 7 on controller error |

- [ ] Verify caching on each exact official-provider/model route using a stable shared prefix across successive requests, including multimodal history where supported. Confirm cache writes where applicable and actual cache-read tokens; an accepted parameter alone is not proof of caching.
- [ ] Preserve stable message prefixes and append new observations/results. Inspect dynamic early prompt fields, serialization and image representations for avoidable prefix changes without altering task semantics or removing history. Check provider-specific minimum lengths, TTLs, multimodal support and thinking-mode restrictions.
- [ ] Log requested caching settings, actual provider/model, cache-read/write token counts, uncached input, output/reasoning usage, cost and latency. Preserve raw usage fields; distinguish missing telemetry from an explicit zero. Report first-write cost and subsequent read savings separately.
- [ ] Diagnose zero-hit runs before subsequent full evaluations; document unsupported modes or Gateway limitations instead of silently claiming caching is active or switching providers. Verify Kimi separately rather than assuming another vendor's parameters apply.
- [x] Preserve the original uncached T07 run. It exited at decision 21 with execution_error before the cancellation request was handled. User subsequently authorized a fresh T07 run with caching enabled; use a separate output directory and verify actual cache receipts.

References: [Vercel automatic prompt caching](https://vercel.com/docs/ai-gateway/models-and-providers/automatic-caching), [Alibaba context cache](https://www.alibabacloud.com/help/en/model-studio/context-cache). Provider-level documentation does not substitute for exact-model runtime verification.

## Context and output budgets — verified 2026-10-30

Plan: compare official-provider Gateway endpoint metadata with vendor documentation, record pricing thresholds, and implement request-side input checks separately from output limits. This section records requirements and pending implementation; it does not activate new evaluation settings.

| Model | Total context tokens (Gateway official endpoint) | Published maximum output tokens | Input-length pricing threshold | Planned budget handling |
| --- | ---: | ---: | --- | --- |
| GPT-6 Astra | 1,050,000 | 128,000 | Above 272,000 input tokens: input/cache rates x2 and output x1.5 for the full request | Approved input hard cap 272,000; output cap 128,000 |
| GPT-6.1 Sol | 1,050,000 | 128,000 | Same 272,000-input threshold and multipliers | Apply input hard cap 272,000; output cap 128,000 |
| Claude Fable 5.1 | 1,000,000 | 128,000 | No long-context premium | Reserve 128,000 output; theoretical input ceiling 872,000 before safety margin |
| Claude Opus 5.5 | 1,000,000 | 128,000 | No long-context premium | Reserve 128,000 output; theoretical input ceiling 872,000 before safety margin |
| DeepSeek V4.1 Flash | 1,000,000 | 384,000 | No input-length tiers listed; peak/off-peak pricing applies | Reserve 384,000 output; theoretical input ceiling 616,000 before safety margin |
| Kimi K3.1 Pro | 1,048,576 | 1,048,576 | Explicitly flat across context lengths | User-selected output cap 384,000, matching DeepSeek; theoretical input ceiling 664,576 before safety margin |

Capacity-derived input ceilings are arithmetic planning bounds, not measured API acceptance limits or equal-information benchmark settings. Select and record the shared history/evaluation policy before comparisons. Kimi reserves 384,000 output tokens per user instruction; stop on input-budget overflow rather than shrinking this reservation or silently truncating history.

- [ ] Implement complete input token accounting, including system instructions, tools, images, history, and tool results. Validate provider token-count support through Gateway; do not assume native vendor counting endpoints are forwarded.
- [ ] Enforce input budgets before generation. For GPT models, stop before sending input above 272,000 tokens. With estimates rather than exact counts, reserve a documented conservative margin; estimation cannot guarantee billing-tier compliance.
- [ ] Stop and record an explicit budget-exceeded result rather than silently dropping or summarizing history. Apply Responses `truncation=disabled` where supported; this alone does not enforce a custom 272k limit.
- [ ] Set the configured output caps: GPT/Claude 128,000; DeepSeek/Kimi 384,000. Verify `accounted_input_tokens + configured_output_cap + safety_margin <= context_window` before sending. Preserve these output reservations by refusing inputs beyond their chosen input ceilings. Kimi's published 1,048,576 output maximum remains reference metadata, not the configured cap.
- [ ] Verify model-specific Gateway parameter mapping with short successful requests before full evaluations. OpenAI Responses uses `max_output_tokens`; native OpenAI Chat Completions uses `max_completion_tokens`; native Anthropic/DeepSeek use `max_tokens`; Kimi K3.1 Pro documents `max_completion_tokens`. Gateway's normalized `supported_parameters` metadata alone is not proof that every wire parameter has identical semantics.
- [ ] Record reasoning/output usage and stop reasons. Maximum output is a cap, not a target; it includes reasoning where applicable. Record model-specific budget differences in comparisons.
- [ ] Preserve strict official-provider allowlists, no model fallbacks, and DeepSeek V4.1 Flash as the default debug model. Qwen3.5 Plus remains a one-off connectivity test.

Pricing notes: Standard GPT-6 Astra input/output rates are $10/$50 per million tokens below the threshold and $20/$75 above it; GPT-6.1 Sol is $2/$10 and $4/$15. Claude Fable 5.1 is $10/$50 and Opus 5.5 $4/$20, without context-length tiering. DeepSeek's official endpoint currently lists off-peak $0.15/$0.60 and peak $0.30/$1.20; the general model card can show different aggregate prices, so use the selected official endpoint. Kimi lists $2/$10 with no context-length tiering. Cache, service-tier, regional, and tool charges are separate. Recheck prices before launching.

DeepSeek version caveat: native documentation currently maps `deepseek-flash` to V4.1-Flash-1017. Keep the requested Gateway ID `deepseek/deepseek-v4.1-flash` unchanged and log returned model/version metadata; do not claim the native alias pins the same snapshot.

Sources:

- [OpenAI Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [OpenAI Sol](https://developers.openai.com/api/docs/models/gpt-6.1-sol), and [input counting](https://developers.openai.com/api/docs/guides/token-counting).
- [Anthropic pricing and long-context policy](https://platform.claude.com/docs/en/about-claude/pricing).
- [DeepSeek models and pricing](https://api-docs.deepseek.com/quick_start/pricing/).
- [Kimi K3.1 Pro limits and flat pricing](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart).
- [Saved official-provider Gateway endpoint responses](/local_home2/zhouxinyang/models/WIA/outputs/vercel_model_limits_20261030); each JSON records its source URL. These confirm declared limits, not successful full-length generation.

- [x] Attempt one full-task API diagnostic with Qwen3.5 Plus on GPU 4, T40, seed 0, high reasoning, 64-decision cap and 64,000 output cap. Official alibaba routing verified. Episode stopped at decision 1 because the forward motion contacted the breakfast table and triggered the phase-1 read-only guard; task completion and scoring remain unavailable. [Assessment](/local_home2/zhouxinyang/models/WIA/outputs/qwen35plus_t40_gpu4_64steps_20261030/run_assessment.json).

- [x] Revise temporal Phase-1 survey protection to base/torso-only action control: reject both-arm, wrist and gripper commands; retain navigation contact and passive drift as diagnostics rather than cancellation triggers. Updated active release instructions and future authoring prompts. Initial scene validation and hidden-event integrity checks remain active. Focused CPU validation: 82 passed; native T40 rerun under this policy remains untested.


- [x] Repair late IK rejection stop certification: preserve the reached pose, propagate late joint-limit/jump reasons, and require measured settling before continuing. CPU regressions and native T07 replay verify stop=true and agent recovery eligibility. The diagnostic forward retreat only moved 1.365 cm of its requested 2 cm and stopped again; full task completion remains untested after this fix. [Native assessment](/local_home2/zhouxinyang/models/WIA/outputs/t07_limit_stop_native_20261030/assessment.json).

- [x] Verify complete native recovery after T07 arm obstruction/limit: measured stable stop, successful independent 1 cm forward retreat (9.996 mm measured), and successful 5 mm backward follow-up (5.004 mm measured), with fresh observation boundaries. GPU 4, no model calls. [Recovery verification](/local_home2/zhouxinyang/models/WIA/outputs/t07_blocked_recovery_native_20261030/limit_stop_verification.json).

- [x] Implement current-sensors/full-text Gateway history: retain all historical text and API-returned reasoning; omit prior image blocks and depth-tool results; preserve current-step sensor/tools and local raw artifacts. 39 regression tests and a two-request official Qwen probe passed.
- [ ] Complete T07 under current-sensors/full-text history (GPU 4, Qwen3.5 Plus, official Alibaba, medium, caching=auto, 64 decisions); inspect request bytes, cache receipts and task outcome separately from full-visual-history trials.

- [x] Count illegal Gateway tool calls and invalid final action outputs as one decision each; preserve returned reasoning, return rejection feedback, refresh observation and continue without robot motion. Repeated invalid output exhausts the total budget without hidden retries. HTTP/routing/runtime failures remain distinct. Focused CPU tests and a live Qwen correction probe passed.

- [ ] Retry or continue evaluation after Gateway credits are available: current-sensors T07 completed 49 decisions before HTTP 402 insufficient_funds at decision 50. Request-size mitigation worked (last body 5.37 MB); no provider/model fallback. Existing native session has exited, so this is not an in-place continuation.

- [x] Add on-demand same-episode historical RGB retrieval by decision step/camera, preserving original pixels and timestamps; remove recalled media again next decision. Offline tool/history regressions pass.
- [ ] Validate historical RGB tool round-trip with official Qwen/Gateway after credits are available; no new paid/native run started.
- [ ] Consider a separately approved compact prompt condition that sends fixed rules/schema once and retains all dynamic text/reasoning; current full-text condition still repeats approximately 14k input tokens per decision. Verify reasoning-field normalization before changing replay representation.

- [x] Add historical RGB tool instructions to the actual Gateway prompt: full syntax/provenance/limits at episode start, brief decision-index reminder afterward, prompt hash in provenance. Nine focused tests passed. Original-vs-local ShowHarness context behavior audited in research_log.md; no automatic context compaction enabled.

### DeepSeek dated-model verification — 2026-10-31

- [x] User explicitly selected `deepseek/deepseek-v4.1-flash-1017` for the T07 test, overriding the earlier alias for this run only. Two paid CPU multimodal probes succeeded with `only: [deepseek]`, `caching: auto`, `reasoning_effort: high`, and `max_tokens: 384000`. Exact model/provider receipts matched; no fallbacks. Second probe cache read: 512/663 tokens. Evidence: outputs/deepseek_v41_1017_preflight_20261031/.
- [x] Add a pre-generation conservative input estimate for this exact model, preserving 384,000 output reservation plus 8,192 margin within the 1,000,000 context. The estimator is explicitly not an exact tokenizer count; exact vendor counting remains pending.
- [ ] Complete T07 adapted seed 0, 64-decision test with `current_sensors_recent2_notes`; inspect exact-model/provider receipts, cache usage, notes, context growth, recovery and final task outcome. Config: configs/deepseek_v41_1017_t07_gpu4_recent2_notes_20261031.yaml. No quality/cache claim from mock tests alone.

## Provider-independent context and cache layout — 2026-10-31

- [x] Move context projection, notes, historical RGB recall and perception budgets into a shared engine. The same history policy works with direct Chat Completions, Responses and local HTTP; Gateway routing is a separate wire adapter.
- [x] Remove Gateway-only history policy and vLLM-only sliding-window configuration restrictions. Use a serving tokenizer where available; otherwise record an approximate UTF-8/image-patch budget explicitly. Preserve exact output reservations.
- [x] For last-result/notes and fresh-state modes, place reusable task, instruction, action-definition and output-contract text before current observations; send current recovery guidance separately. Rebuild from current inputs to avoid stale limits.
- [x] Keep tool catalogs stable and enforce exhausted perception budgets locally before tool dispatch. Record system-prefix and tool-catalog fingerprints. Preserve invalid-decision step accounting.
- [ ] Measure real cache receipts, latency, cost and T07 quality for each exact official-provider route. Offline tests and stable hashes alone do not establish cache hits or unchanged policy quality.
- [ ] Add a client-owned history adapter for native CLI sessions before enabling exact context projection there; currently fail explicitly on incompatible history ownership.

## Four-GPU DeepSeek debug bench — 2026-10-31

- [x] Add the model-independent `wia bench` entry, progress bar, request token/cost logs and worker failure aggregation.
- [x] Provide `scripts/bench_deepseek_4gpu.sh` and `configs/bench_deepseek_60.yaml`: all 60 catalogue tasks, original variants, seed 0, 64 decisions, four idle GPUs, official DeepSeek only, caching auto, 384,000 output reservation and last-result/notes policy.
- [x] User explicitly approved `low` for this bench because the exact DeepSeek model does not advertise medium. This is a run-specific exception to the general medium preference.
- [ ] Execute the four-GPU bench and assess quality, latency and actual cache/billing receipts. Script delivery and dry-run validation do not constitute a completed benchmark.

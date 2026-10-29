# InteractiveWorld candidate extensions for WIA

Status: research design only; integration into V1 or a later release is undecided.

## Scope and baseline

Review all 120 local InteractiveWorld TaskPacks against the current local WIA research story, taxonomy and 61 frozen release tasks, as confirmed by the user. InteractiveWorld HEAD: 40957c7243aa38ea8550d929dd3f10e99d8733d6. WIA HEAD: 82e335dca0d810b6bf5a257f1f4aaaf14ea7954a; the local workspace contains pre-existing uncommitted changes. HEAD alone does not identify the baseline. Do not modify existing runtime, task, release, or configuration files. No simulations, commits or pushes.

## Hypothesis and criteria

The useful transfer is a decision mechanism that survives removal of the assistant-human communication channel, not a change of embodiment alone. Require world evidence to change a decision, executable physical actions and verifiable outcomes. Distinguish external semantic knowledge, observed geometry, episodic history, explicit procedural rules, and inferred causal knowledge. Distinguish final-state checks, process/trace checks, and adjudicated release criteria. A textual instruction followed by a token delivery is a weak addition unless physical state materially constrains the decision.

Prioritize evidence-driven recovery, resource-aware allocation, completion-state revision, and history-dependent functional identity for their scientific mechanisms. The existing AP-4 gap is contextual evidence, not a design requirement. Do not claim existing TM, GG, GF or prescribed-order AP tasks are absent. Assess every recommendation against concrete frozen task files, not only the taxonomy proposal. Counterfactuals must change correct actions while controlling unrelated difficulty.

## Procedure

1. Main reviewer reads paper, source architecture, task premises and WIA baseline; defines shortlist and rubric.
2. Three bounded evidence/documentation assignments cover two source categories each (40 tasks), with nine deep candidates each. Each agent reads complete JSONs and relevant scoring code, extracts evidence, and drafts under this rubric. Agents cannot change the shortlist or scientific direction.
3. Main reviewer audits evidence and final judgments, reconciles duplicates, and authors independent robot task prototypes and recommendations.
4. Verify 120 unique initial-screen rows, 27 deep reviews, all source links and provenance, and document-only changes.

## Deliverables

Source audit, complete screening table, 27 candidate reviews, prioritized robot-centric prototypes, independent recommendations, and an English research log. All new artifacts reside under WIA/docs/research/interactiveworld_candidates_20261029. Because this task explicitly permits only new research documents, its research_log.md is created in this new documentation directory; the existing project-root research_log.md is preserved.

## Evidence policy

No new experimental results or feasibility certification. Static source support is not successful robot execution. Shared BEHAVIOR assets do not imply equivalent physics, actions, rendered cues or evaluator semantics. Novel dynamics are conditional proposals, not existing simulator capabilities.

## Current design policy

The user permits knowledge-guided real pick-and-place, ranked by coupling to the physical task. Start from the scientific question and adapt the suitable InteractiveWorld scene directly. Existing WIA categories are comparison references, not constraints or assigned destinations. Classification and V1/later-release placement remain open. See [scene redesign plan](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/scene_redesign_plan.md). No scene, task or runtime implementation is changed in this documentation phase.

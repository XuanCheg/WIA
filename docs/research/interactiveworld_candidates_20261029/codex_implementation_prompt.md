# Implementation prompt: InteractiveWorld-derived robot tasks for WIA

Implement the high- and medium-priority candidate mechanisms identified in the completed InteractiveWorld-to-WIA research review. First read the research documents and actual source code, write a brief implementation plan grounded in that reading, and then proceed with implementation and bounded native validation.

## Workspace and sources

Working directory: `/local_home2/zhouxinyang/models`

Repositories:
- WIA: `/local_home2/zhouxinyang/models/WIA`
- InteractiveWorld: `/local_home2/zhouxinyang/models/InteractiveWorld`

Research package: `/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029`

Read these documents first:
- `README.md`
- `scene_redesign_plan.md`
- `reviewer_decisions.md`
- `robot_task_prototypes.md`
- `source_audit.md`
- `evaluation_design.md`

Before implementing each candidate, read its corresponding `deep_*.md` review, complete source TaskPack, effective interaction contract, relevant runtime state logic and scoring implementation. Use `screening_all_120.csv` and `source_contract_inventory.json` as indexes, not substitutes for the actual code.

Also read applicable AGENTS.md instructions, the current WIA research story and taxonomy, `explore_to_achieve/wia_v1/release/catalogue.json`, and the frozen task YAMLs, scenes, converters, runtime and evaluators relevant to overlap and reuse. Recheck live repository state; recorded review revisions are provenance, not proof that current files are unchanged.

## Scientific intent

The target is:

World Knowledge → Decision → Physical Action → World Outcome

Start from the scientific question and directly adapt an appropriate InteractiveWorld scene for robot action. Existing WIA categories are an overlap baseline, not a framework every new task must fit. Decide classification after the task mechanisms are defined. Integration into V1 or a later release remains undecided.

Knowledge-guided real pick-and-place is allowed. Rank designs by how strongly knowledge affects physical constraints, choices, progress tracking or feedback-driven revision. Do not multiply independent tasks merely by changing objects, rooms or counts. Counterfactual variants are not additional independent tasks.

Shared BEHAVIOR-1K assets do not establish that human controls, symbolic states, visual effects or evaluators transfer unchanged. Reuse the available environment where appropriate, but inspect implementation details. A Cook flag does not certify thermal cooking; hiding a spill does not certify physical cleanup. Static research documents and embedded historical preflight reports are not native feasibility certificates.

## Scope: nine task prototypes, including medium-priority directions

Implement one minimal but scientifically complete robot prototype for each of the nine mechanisms below, with counterfactual variants that test its decision dependence. The target is nine prototypes, not nine source tasks copied verbatim. Complete high-priority prototypes first where dependencies permit, then implement the medium-priority prototypes. Do not omit medium-priority work merely because it is harder. If a required mechanism cannot be implemented faithfully, report the concrete limitation and seek clarification on scientifically meaningful alternatives while continuing independent tasks.

### H1 / P1 — Evidence-triggered object replacement

Sources: GUIDE-87017; NARRATE-14004 as a related recovery variant.

Goal: during fulfillment, manipulation or inspection reveals that the selected object is unsuitable. The robot decides whether to keep it or replace it, then completes the original goal.

Preserve the source preparation station, candidate objects, spare supply and delivery structure when useful. Include acceptable and unacceptable branches; a fixed discard-first policy must fail. Separately vary replacement compatibility if needed. If native slicing or material effects are unavailable, explain a rigid inspectable-object alternative and ask before adopting a substitution that changes the scientific meaning. Score actual delivery, rejected-object placement and preservation, not a verbal defect report.

### H2 / P2 — Recovery of an invalidated completed condition

Source: ORCHESTRATE-112022.

Goal: prepare a workspace, complete a secondary delivery, and leave the workspace ready. A late disturbance can invalidate the already completed workspace condition.

Use the documented first-prototype design with a reversible rigid obstruction instead of claiming that scripted water visibility is fluid cleanup. Preserve useful source workspace and delivery context. Include disturbance/no-disturbance branches and, if useful, a change outside the protected work region. The robot must have an observation opportunity before termination. Score the actual clear region, preserved delivery and stable storage of the obstruction. Do not require repair in the no-disturbance branch.

### H3 / P3 — Progress revision and net inventory replenishment

Sources: ASK-14011, ASK-99600, WATCH-99801.

Goal: complete actual remaining work after additions, removals, interrupted execution or reversed progress. Do not treat the number of past action attempts as the current inventory.

Choose a compact representative scene. Define minimum-stock versus exact-count semantics explicitly. Include retained/undone progress and already-complete/remaining-deficit conditions. Permit valid current-state inspection in the main task; a memory-only restriction belongs in a separately named diagnostic condition. Avoid inheriting the source ASK-14011 conflict between requested W7 and ten waters actually remaining. Score real stock and preservation of completed work rather than arbitrary replacement IDs.

### H4 / P4 — Adaptive space clearing

Source: NARRATE-12802.

Goal: clear sufficient space for a target placement while preserving fixed anchors and avoiding unnecessary relocations.

Retain the bounded table region, movable obstacles and target object. Vary target dimensions or one blocker position so the useful removal subset or count changes. Include an initially sufficient region. Do not require all three source bottles to be moved in every case. Score full intended-footprint placement, supported stability, legal approach and withdrawal, protected anchors and legal storage of removed objects. Mere on(table) is insufficient.

### H5 / P5 — Destination revision during execution

Source: ASK-30000.

Goal: deliver a carried object to the preferred destination unless observed conditions disqualify it, then use an eligible alternative.

Preserve the primary/fallback station structure. Specify a meaningful public acceptance rule; distinguish a procedural exclusion from genuine geometric infeasibility. Include primary available/unavailable variants so always diverting fails. Separately vary fallback eligibility if needed. Permit all valid inspection and transport plans instead of requiring the source's exact Walk/LookAt sequence. Score branch-correct placement and preservation of resident objects.

### M1 / P6 — History-dependent functional eligibility

Sources: WATCH-28002, WATCH-34002, WATCH-81019, WATCH-14014.

Goal: select or segregate visually matching objects according to their observed use history.

Implement one representative mechanism, not a separate task for every towel or sponge count. Use robot-observed histories and neutral public object handles. Pair episodes with matched final visible layouts but different objects receiving the use event, so correct eligibility and subsequent actions reverse. Preserve observable identity continuity; do not create an impossible hidden identity swap.

An explicitly defined event-conditioned eligibility state is acceptable. If claiming actual contents or contamination, implement and validate the corresponding persistent state rather than relying on fixed object IDs. WATCH-14014 has a known gap between source pouring semantics and its release persistent-content criterion; do not inherit that gap. Select assets whose handling does not overwhelm the scientific test. Seek clarification before adopting a materially different surrogate.

### M2 / P7 — Resource-constrained physical allocation

Sources: GUIDE-24003, GUIDE-109009, GUIDE-99806; GUIDE-102000 as related evidence.

Goal: choose and physically distribute actual units under shared stock, quantity and order constraints.

Retain stock areas, readable quantity evidence and delivery destinations. Implement a small coupled allocation instance with interchangeable units and all valid subsets accepted. Vary observed stock or a quantity/order condition separately so the correct allocation changes. Include a case where a simple fixed or greedy policy fails.

Score actual delivered quantities, unique entity counting, unused-stock preservation and stable physical placement. Labelled fluid content is an operational quantity, not package geometry or shelf strength. Do not require liquid simulation for sealed-unit allocation. Check the small instance's feasible solution set before claiming uniqueness or optimality.

### M3 — Knowledge-guided handling choice

Sources: GUIDE-14018, GUIDE-14011, GUIDE-14014.

Goal: use observed contents, object properties or item-specific handling rules to choose and physically execute the appropriate storage or handling action.

Implement one compact representative prototype directly in a suitable source scene. GUIDE-14018 is a useful starting option: matching storage units have different observed contents, and the correct destination changes when those contents swap. Remove mandatory fixed inspection order and answer-bearing handles. If selecting another source, retain its actual context without adding unrelated geometric puzzles just to fit an existing category.

Separate explicit-rule conditions from external semantic-knowledge conditions. Include a matched variation that changes the correct handling destination or action. Score real placement/storage, required equipment state and preservation. Do not claim measured cooling, drying, food safety or material damage unless such effects are independently implemented. Simple ingredient selection sources ASK-35004/35005 are not additional required prototypes in this scope.

### M4 / P8 — Resource-aware scheduling with physical handoffs

Sources: ORCHESTRATE-44001, ORCHESTRATE-99902.

Goal: schedule a small set of jobs across limited processing stations and a single robot, with actual loading, unloading and delivery.

This medium-priority direction is included in implementation scope despite its longer-term difficulty. Start with few jobs. Define simulation/task-clock semantics, station exclusivity, processing duration, robot occupancy, preparation and transport costs, completion and deadline handling. A clearly declared timed state-machine process is acceptable; it is not thermal physics.

Count a job only after the required process has elapsed under valid occupancy and the object has been physically delivered. Prevent overlapping exclusive resource use. Separate task time from API/model wall latency and document when the world pauses. Validate a small feasible/optimal schedule with bounded enumeration or an equivalent check that includes robot handling constraints. Pair duration changes or eligibility changes separately so the correct plan changes. Add the source's richer menu restrictions only after the base timing contract works.

The source documents contain minute durations, but their native Cook/process predicates do not establish a live scheduling model. Do not copy them and claim scheduling success. Ask before changing a timing assumption that alters the experiment's meaning.

## Implementation workflow

1. Inspect repository state and preserve all existing and uncommitted work.
2. Read the materials above, verify critical source and evaluator logic, then write a short plan and implementation table for all nine prototypes: source scene, preserved elements, changes, evidence, decisions, physical endpoints, variants, dependencies and risks.
3. Ask promptly about ambiguities that change scientific meaning. Continue work independent of the answer. Resolve routine engineering details autonomously.
4. Choose implementation order from scientific value, dependencies and feasibility. Complete one end-to-end prototype before expanding common infrastructure; avoid building an oversized generic framework first.
5. Add derived scenes, candidate task definitions, evaluators and necessary support in a separate candidate area. Do not overwrite the frozen WIA v0.1 tasks, scenes, scoring or default release catalogue.
6. Reuse existing modules, controls and environments. Do not repeat large downloads or installations without first checking available assets and requirements.
7. If shared runtime changes are necessary, explain why, preserve existing task behavior and perform focused regression checks. Do not silently alter the released benchmark's scientific configuration.
8. Do not commit or push. Do not delete unrelated data or stop existing tasks. Follow applicable instructions for branches/worktrees.
9. This prompt does not require subagents. If delegation is separately authorized, the main agent retains scientific decisions, source understanding, experiment design and final review; implementation assignments must have explicit non-overlapping ownership and validation requirements.

## Robot observation, action and scoring contract

- Use actual robot sensors and legal public controls. The policy must not read private answers, hidden event state, evaluator identities or answer-bearing semantic handles.
- Establish history from observations actually delivered to the robot.
- Implement genuine grasp/contact, motion, release, support and withdrawal in the native environment.
- Private evaluators may read world state, but cannot leak it into policy inputs.
- Accept all valid plans and interchangeable-object assignments. Reference programs are witnesses, not mandatory solutions.
- Enforce process constraints only when scientifically or operationally meaningful.
- Distinguish natural dynamics, authored controlled events and symbolic state transitions. Trigger disturbances from explicit task events or a defined simulation clock, not variable model/API latency.
- Change one research factor at a time in paired variants where possible. Match unrelated observation, budget, control and geometric conditions. Correct no-op or shorter-action branches are allowed.
- Record thresholds and calibration rationale rather than blindly copying settings from unrelated experiments.
- Distinguish scene load, reference-controller success and autonomous model success. None substitutes for another.

## Validation and resources

Perform necessary CPU checks followed by bounded native robot validation. Do not immediately launch large model evaluations, broad sweeps or training.

For every prototype verify:
1. Initial state does not accidentally solve the task, except explicitly defined already-satisfied branches.
2. Required decision evidence is available through permitted robot observations.
3. A correct control plan physically completes each required branch.
4. Representative wrong choices or shortcuts fail the corresponding physical/process condition.
5. Counterfactuals actually change the valid decision set or necessary actions.
6. Final world state, process invariants and reported scores agree.
7. Preservation and valid no-op conditions are evaluated correctly.

Use targeted tests that protect scientific validity; do not spend most effort on broad test coverage, formatting or infrastructure. Treat missing scores and environment failures separately from policy failure.

Follow current server and AGENTS.md rules. Recheck live GPU utilization, memory, processes, CPU/RAM and storage before launches. Select CUDA_VISIBLE_DEVICES explicitly and record mapping. Use available idle devices for useful independent validation where appropriate without changing scientific settings to increase concurrency. Keep long runs in persistent tmux sessions and record process IDs, session/window names and devices. Preserve existing tasks and port forwarding. Recheck server policy if operating on another host; do not assume this server's direct-execution rules apply elsewhere.

## Reporting and deliverables

Communicate progress and conclusions concisely in Chinese. Write code comments, task specifications, configurations, research documentation and newly authored logs in English. Preserve historical raw artifacts verbatim. Add a run-instruction docstring to new Python files.

Maintain research_log.md at the project root according to applicable instructions, preserving existing content. Record each iteration's hypothesis, failures, causes, changes, validation results and retain/revert decision. Keep source provenance and derived-scene changes traceable.

Deliver:
- Nine candidate prototypes with their scientifically necessary counterfactual variants, or a precise per-prototype account of any genuine blocking condition.
- Task, scene, evaluator and key code paths.
- Exact environment and run commands.
- Bounded native validation evidence, including successful reference execution and representative rejected shortcuts.
- A source-to-derived comparison describing what was preserved, changed and not simulated.
- Comparison with existing WIA mechanisms without forcing category assignment.
- A summary table of implementation status, branch coverage, physical verification, remaining issues and estimated difficulty.
- Runtime/action-cost information where measured; do not invent model-performance results.

If native capabilities prevent a faithful implementation, do not fabricate success or silently weaken the task. State the concrete obstacle, completed work and scientifically meaningful alternatives, ask about changes that alter the experiment, and continue independent prototypes. A running process is not a completed validation; report its status accurately.

Begin with a brief plan grounded in the actual reading, then start implementation. Do not stop after providing a plan when implementation is authorized by this prompt.

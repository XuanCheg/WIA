# Source and scientific validity audit

## Sources and scope

The [InteractiveWorld paper](https://arxiv.org/html/2610.33966v1) motivates closed-loop assistance through a human executor. Removing that executor changes what is measured: communication, intervention timing and the executor's interpretation no longer define the target robot capability. Its reported assistant performance should not be treated as a robot baseline. The paper's oracle success is not a robot feasibility certificate. This review uses the released repository to inspect concrete task and scoring semantics and treats proposed robot designs as untested.

Local InteractiveWorld revision: `40957c7243aa38ea8550d929dd3f10e99d8733d6`, [upstream repository](https://github.com/facebookresearch/InteractiveWorld). WIA revision: `82e335dca0d810b6bf5a257f1f4aaaf14ea7954a`, with current local modifications accepted by the user as baseline. The frozen catalogue contains 61 tasks. The complete TaskPack population contains 120 unique tasks, 20 in each of six directory categories. Prefixes do not reliably indicate those categories: GUIDE and WATCH tasks also occur in finding/decision.

Local baseline references:

- [Research story](/local_home2/zhouxinyang/models/WIA/docs/research/world_interactive_agency_v1/source_documents/01_Research_Story/World_Interactive_Agency_Core_Story.md)
- [Taxonomy](/local_home2/zhouxinyang/models/WIA/docs/research/world_interactive_agency_v1/source_documents/02_Taxonomy/World_Interactive_Agency_Taxonomy_v1.yaml)
- [Frozen catalogue](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/release/catalogue.json)
- [Baseline worktree status](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/baseline_worktree_status.txt)

The older story is a proposal; current release YAMLs are the authoritative task definitions for overlap. Existing task status fields are not changed by this review and do not establish that this agent ran them.

## Four scoring layers must remain separate

1. `executable_spec.goal`: positive predicates, negatives, aggregates and trace outcomes. [task_score.py](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/physical_world/task_score.py:109) combines these and excludes baseline-satisfied positive goals from earned progress.
2. Interaction completion contract: command requirements, forbidden commands, subsequences, forbidden-before constraints and verification points. [process_requirements_met](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/interaction/taskpack_interaction.py:10631) implements these. WATCH also checks timely intervention. Missing `story.interaction` does not establish missing constraints: inspect [interaction contract fallback](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/interaction/taskpack_interaction.py:11772) and the per-task interaction openings.
3. Native completion reduction: [assistant_state](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/interaction/taskpack_interaction.py:11104) can let verification points define world success, then combines world/process/verification results. [score.py](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/evaluation/score.py:321) additionally handles assistant completion and unavailable/inconsistent evidence.
4. Released benchmark rubric: [rubric.json](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/evaluation/rubric.json), reduced from trace-adjudicated evidence by [score.py](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/evaluation/score.py:369). Do not equate its requirements or scores with native predicates. In CFSS-StateScore, terminal-state strict pass and transition-gate strict pass are separate outputs.

Consequently, a final-only review would unfairly overlook real source process checks. Conversely, a required successful Wipe command is not proof of physically removing liquid. Robot adaptation must decide which physical invariants to retain and remove assistant-specific timing and unnecessary oracle path restrictions.

## Physics and state abstraction

The code explicitly calls CUT and COOKED declared symbolic predicates in [isaac_goal.py](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/physical_world/isaac_goal.py:48). [produce_cook](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/physical_world/semantic_executor.py:13471) validates an appliance context and declares the cooked state; this is not a temperature integration model. [produce_wipe](/local_home2/zhouxinyang/models/InteractiveWorld/simulator/physical_world/semantic_executor.py:13283) produces a tool trajectory. Specific visual effects may be authored on top of that trajectory.

Examples needing especially careful claims:

- ORCHESTRATE-112022 creates/removes a visible late-spill effect through authored event triggers. Its native process sequence is meaningful, but absorption and fluid cleanup are not established.
- NARRATE-81009 hides broken glass via a visual event; native terminal predicates do not establish containment of fragments, whereas the release rubric asks for stronger cleanup evidence.
- FIND-81013 uses a deterministic off-camera fall transition. It is an evidence-guided recovery/search source, not a validated physical prediction experiment.
- ORCHESTRATE-44001 and ORCHESTRATE-99902 carry durations in rendered recipe/menu documents, but native cut/cook/process checks do not alone establish a live scheduling clock or an optimal makespan.
- ASK-35004 and ASK-35005 physically deliver selected ingredients. They do not simulate fermentation or tenderization. The user permits such knowledge-guided real placement; prioritize it according to its coupling to physical constraints, and treat these simple ingredient selections as low-priority family variants.

Shared BEHAVIOR assets support reuse of scenes and objects. They do not establish identical articulation handling, collision geometry, grasp mechanics, rendered labels, material effects, symbolic states or scoring. No installation or environment replacement is proposed here.

## Baseline overlap and novelty boundaries

The current taxonomy already includes controlled dynamics trials (AP-3), geometric feasibility (GF), collection and retrieval (GG), and episodic/change-contingent manipulation (TM). [T42](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/release/tasks/t42.yaml) requires trials and subsequent spare delivery. [T40](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/wia_v1/release/tasks/t40.yaml) already supports add/remove/move/replace/no-change branches and physical restoration. Therefore, neither “uses memory” nor “has a counterfactual” establishes novelty.

AP-4 is an explicit taxonomy gap. Strong additions make goal execution itself reveal an invalid assumption or invalidate a completed subgoal, requiring an appropriate change of plan. They must differ from simply restarting the existing two-phase TM protocol. Resource allocation and autonomous scheduling can add a decision structure absent from a prescribed sequence, provided the evaluator scores feasible outcomes rather than one reference plan.

## Evidence limitations

All judgments are static source/design judgments. No simulator episode, robot manipulation, timing benchmark or model evaluation was run by this review. Embedded preflight reports and old artifacts are source material, not newly reproduced results. Source inconsistencies are documented in deep reviews and must be resolved before implementation. No numerical performance gain, cost estimate or feasibility pass rate is claimed.

# Main-reviewer candidate decisions

These decisions concern independent research value after robot-centric redesign. They do not admit executable tasks into a release. “Priority” means worth designing first; “conditional” means retain only with the stated change; “merge” means a variant of another mechanism, not another independent task. All 27 fixed deep-review candidates are accounted for below.

| Source task | Main judgment | Prototype / WIA fit | Decisive reason and required boundary |
|---|---|---|---|
| [NARRATE-12802](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/accessibility/NARRATE-12802.taskpack.json) | Priority after redesign | P4; GF/AP | Change the number/subset of necessary removals and score actual gap placement; fixed remove-all sequence is weak. |
| [NARRATE-14004](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/accessibility/NARRATE-14004.taskpack.json) | Merge as recovery variant | P1; AP-4/GG | Correcting an already-held wrong object is useful only if new evidence invalidates a prior choice; otherwise ordinary recognition. |
| [NARRATE-81009](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/accessibility/NARRATE-81009.taskpack.json) | Conditional | P4/P2; AP-4 | Temporary staging before cleanup is useful; source glass disappearance is not physical fragment containment. |
| [NARRATE-99203](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/accessibility/NARRATE-99203.taskpack.json) | Do not prioritize independently | GG/GF | A conspicuous shoe in a fridge is an easy anomaly-removal story. Require a genuine task-relevant conflict; arbitrary remove-shoe semantics add little. |
| [WATCH-28002](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-28002.taskpack.json) | Priority as one merged TM extension | P6 | Functional eligibility depends on observed use history; preserve limited event-conditioned segregation claim. |
| [WATCH-34002](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-34002.taskpack.json) | Merge | P6 | Same identity/history mechanism with relocation and delay; towel control should not dominate. |
| [WATCH-81019](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-81019.taskpack.json) | Merge as scale variant | P6 | More identical towels increases tracking load, not the number of new capabilities. |
| [WATCH-88007](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-88007.taskpack.json) | Conditional; low independent novelty | AP-1 | A printed tray-before-dough rule resembles existing prescribed-order tasks. Promote only if physical prerequisites must be inferred and checked. |
| [WATCH-99801](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-99801.taskpack.json) | Merge | P3 | Outstanding item tracking before activation; use current goal state instead of a human's imminent mistake. |
| [GUIDE-18001](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-18001.taskpack.json) | Conditional | AP planning | A cyclic assignment does not imply a physical buffer is necessary. Create genuine occupancy/carry constraints or retain as ordinary sorting. |
| [GUIDE-24003](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-24003.taskpack.json) | Priority as allocation family | P7 | Remaining allowance should depend on real stock; distinguish content quantity from geometric shelf capacity. |
| [GUIDE-47002](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-47002.taskpack.json) | Low-priority contrast | AP-1/GF | Pad-before-headset is explicitly prescribed. Real support/contact or inferred protection constraints are needed for stronger novelty. |
| [GUIDE-14011](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-14011.taskpack.json) | Admissible; merge as secondary extension | Semantic routing | Different treatment routes for food/device provide knowledge-driven placement; no verified temperature outcome. |
| [GUIDE-14014](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-14014.taskpack.json) | Admissible; merge as secondary extension | Semantic routing | Material-compatible routing and appliance closure are measurable; drying/damage are not established. |
| [GUIDE-87017](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/GUIDE-87017.taskpack.json) | Priority mechanism | P1; AP-4 | Manipulation reveals evidence that changes fulfillment plan. Start with rigid inspectable components rather than unsupported slicing/spoilage. |
| [ORCHESTRATE-112022](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/ORCHESTRATE-112022.taskpack.json) | Priority mechanism | P2; AP-4 | A newly invalidated completed subgoal requires renewed verification and repair. Start with a rigid disturbance; spill cleanup needs extra support. |
| [ORCHESTRATE-44001](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/ORCHESTRATE-44001.taskpack.json) | Conditional, longer-term | P8 | Deriving schedules is valuable, but requires a real task clock, occupancy and completion semantics. |
| [ORCHESTRATE-99902](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/ORCHESTRATE-99902.taskpack.json) | Merge as later composition | P8 | Adds eligibility constraints to scheduling; do not begin with many dishes before the base timing probe is valid. |
| [ASK-14011](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-14011.taskpack.json) | Priority after scoring repair | P3 | Net inventory deficit differs from counting action attempts; use minimum versus exact targets explicitly. |
| [ASK-30000](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-30000.taskpack.json) | Priority compact probe | P5; AP-4 | New destination evidence changes action before commitment. Randomize primary availability and avoid fixed fallback. |
| [ASK-35004](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-35004.taskpack.json) | Admissible; low-priority family variant | Knowledge selection | Real ingredient delivery with a clear knowledge dependency, but not a fermentation experiment. |
| [ASK-35005](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-35005.taskpack.json) | Admissible; low-priority family variant | Knowledge selection | Same structure as yeast selection; do not count two trivia substitutions as two independent abilities. |
| [ASK-99600](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-99600.taskpack.json) | Merge, strongest compact P3 variant | P3 | Removal reverses previous progress; final state rather than number of puts determines completion. |
| [FIND-28003](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/finding/FIND-28003.taskpack.json) | Conditional extension | TM/GG | Nested containment and container transport add compositional memory; compare against existing history retrieval before claiming novelty. |
| [FIND-81013](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/finding/FIND-81013.taskpack.json) | Conditional/low priority | TM/GG | Scripted unseen relocation plus recovery overlaps existing change/search probes. Do not label it learned fall dynamics. |
| [GUIDE-14018](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/finding/GUIDE-14018.taskpack.json) | Conditional extension | GG plus semantic generalization | Infer destination category from contents; remove mandatory fixed inspection order and swap contents across units. |
| [WATCH-14014](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/finding/WATCH-14014.taskpack.json) | Merge with limits | P6 | History distinguishes identical glasses; native persistent alcohol composition is not established. Use event-conditioned routing or implement content state before stronger claims. |

## Scene-first interpretation

Design directly from suitable InteractiveWorld scenes; the WIA-fit column records overlap only. It does not assign a category or require reuse of an existing task family. The [scene redesign plan](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/scene_redesign_plan.md) gives the current preservation/change/outcome/variant specifications. Merge recommendations concern duplicate decision mechanisms, not mandatory insertion into the current taxonomy.

## What genuinely deserves additional WIA task design

The best immediate design set is P1, P2, P3, P4 and P5. Their value lies in decisions that depend on intermediate world evidence and lead to distinct physical outcomes. P6 adds a useful history-based functional distinction if it survives comparison with T38/T39 and TM-2. P7 is worthwhile as coupled allocation, not a numeric answer moved onto a table. P8 is the strongest longer-term planning direction but requires new timing semantics and careful separation of cognitive scheduling from controller throughput.

The five immediate mechanisms need not become five independent benchmark entries. P1 and P5 can share a compact AP-4 family if paired evidence shows they test the same failure mode; P2 must demonstrate a distinction from the existing two-phase TM tests. A small number of well-controlled probes is preferable to importing dozens of household stories.

## What is mostly a reskin

Most unchanged finding tasks repeat remembered-location retrieval; longer phone-call gaps and different rooms are difficulty variations. Most accessibility tasks lose their communication challenge when the robot itself sees and acts. Prescribed housekeeping lists largely overlap AP-1, while fixed wrong-item corrections overlap GG. Selecting an invoice/class card after arithmetic is primarily QA followed by token movement. Do not expand WIA's claimed coverage merely because a story mentions safety, nutrition, chemistry or an appliance.

## Most valuable research questions

1. Can the agent stop executing a previously reasonable plan when new evidence invalidates it, without restarting or discarding useful progress?
2. Does the agent represent completion as a revisable world condition, including failed actions and reversed progress?
3. Can the agent alter the environment just enough to make a desired action feasible, rather than blindly executing a cleanup script?
4. Can interaction history change an object's functional role when appearance cannot disambiguate it?
5. Can explicit constraints and world state be jointly translated into feasible allocation and resource schedules?

These questions fit the WIA story without committing to V1 integration or a later version. Source transfer is a way to construct controlled probes, not a novelty claim by itself.

# Robot-centric candidate prototypes

These are independent research designs derived from InteractiveWorld mechanisms, not executable tasks or native feasibility certifications. Release placement is undecided. Numerical thresholds below are design parameters to calibrate, not inherited experimental settings. Existing WIA task definitions remain untouched.

## Common contract

The robot receives a goal and sensor-grounded evidence, chooses and executes its own actions, observes the consequences, and verifies completion. Remove the human model, its intended mistakes, semantic intervention windows, and exact oracle-path matching. Preserve meaningful physical and task constraints. Accept all valid plans; do not reward reproducing a reference utterance or redundant inspection sequence.

Every manipulation task requires actual grasp/contact, legal motion, release, withdrawal, supported stable placement and preservation of protected objects. Stability tolerances must be calibrated for the selected assets. Hidden evaluator identity and event state must never appear as public object names. Report goal success separately from preservation violations, observation/decision errors, and controller failures. A correct verbal explanation does not compensate for a wrong physical result.

For each paired variant, keep assets, camera availability, action interface, budgets and initial robot configuration matched except for the manipulated factor. Randomize irrelevant positions and visual identifiers across independently generated pairs, rather than changing them within a claimed one-factor pair. Include no-change/no-repair trials where the mechanism requires them.

## P1. Evidence-triggered replacement during fulfillment

**Sources:** [GUIDE-87017](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/GUIDE-87017.taskpack.json), [NARRATE-14004](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/accessibility/NARRATE-14004.taskpack.json). **Overlap reference (not assigned category):** AP-4; secondary GG and TM. **Priority:** high after redesign.

**Goal.** Assemble and deliver an acceptable kit. One initially plausible component reveals a defect only after the robot exposes a previously hidden face; a compatible spare is available. Use rigid packaged parts rather than simulated cutting or food spoilage in the first prototype.

**Knowledge.** The acceptance rule and component compatibility, plus new visual evidence produced by manipulation. The initial plan is defeasible. A defect cue must correspond to a visible asset feature, not a message announcing failure.

**Decision and action.** Inspect or rotate the selected component, determine whether to retain or replace it, return a rejected component to quarantine, select a compatible replacement, and complete the kit. Exposing a face is a proposed adaptation of the source's cut-then-inspect mechanism, not evidence that InteractiveWorld simulates fracture or mold growth.

**Physical completion.** Every required compatible component is released in the kit; rejected components remain outside the delivery region in quarantine; protected stock is preserved. Log the first informative view and the subsequent choice. Do not require replacement in the defect-free branch. An evaluator checks component identity and physical location, not the agent's word “defective.”

**Counterfactuals.** Defect absent versus present on the same inspected face; alternatively change which spare matches the required connector while leaving defect evidence fixed. Separate these two factors. The former flips retain/replace; the latter flips replacement identity.

**Difference from V1.** T40/T41 and T45–T48 already support change-dependent restoration. The addition is evidence discovered during the robot's own goal execution that invalidates an intermediate plan, with goal-preserving substitution. If initialized after the defect is already announced, this collapses to ordinary target selection.

**Difficulty.** Medium for rigid inspection and replacement; high for native slicing or material damage, which are unnecessary for the first study. Main risks: cue visibility, spare compatibility leakage and replacing everything as a fixed policy.

## P2. Re-establish a completed condition after a late disturbance

**Source:** [ORCHESTRATE-112022](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/ORCHESTRATE-112022.taskpack.json). **Overlap reference (not assigned category):** AP-4; secondary TM-2. **Priority:** high mechanism, conditional native realization.

**Goal.** Prepare a workspace, deliver a component to another station, and leave the workspace ready for the next operation. A late, reversible disturbance can invalidate the previously completed workspace condition.

**Knowledge.** Completion is a predicate of the current world, not a permanent property of a past action. A secondary delivery does not certify that the workspace stayed ready.

**Decision and action.** Finish the initial workspace preparation and delivery, inspect the workspace, repair only if its readiness predicate is false, and verify again. Start with an observable rigid obstruction moved into the designated free work region. This preserves the revision mechanism without claiming a fluid-cleaning model. The source's spill/wipe version remains a separate higher-cost option.

**Physical completion.** Delivered component supported at destination; designated work region clear of interfering collision geometry; displaced obstruction legally stored; unrelated objects preserved. Require the readiness condition to hold at termination after the possible disturbance. Do not require a repair action when no disturbance occurred.

**Counterfactuals.** No disturbance versus obstruction after delivery; a separate pair uses an equally visible change outside the protected work region, which should not trigger unnecessary restoration. Disturbances must occur before an opportunity for informative observation and action.

**Difference from V1.** Existing TM-2 already compares two phases and repairs changes. The incremental test is invalidation of a subgoal achieved by the agent during an ongoing task, with a competing remaining goal and self-initiated verification. A pre-announced two-phase change test is a duplicate, not this prototype.

**Difficulty.** Medium for controlled rigid disturbances; high for contact-dependent spill removal. Do not use wall-clock model latency to trigger a physical event. Use a deterministic task event for paired trials and document that this is authored intervention, not spontaneous physics.

## P3. State-based resumption and net inventory replenishment

**Sources:** [ASK-14011](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-14011.taskpack.json), [ASK-99600](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-99600.taskpack.json), [WATCH-99801](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-99801.taskpack.json). **Overlap reference (not assigned category):** TM with AP planning; no new taxonomy family required. **Priority:** high as one mechanism family.

**Goal.** Complete a stocking target after an interruption and intervening additions/removals. Already satisfied categories must remain satisfied without unnecessary handling.

**Knowledge.** Current stock is initial stock plus verified additions minus verified removals. An attempted action does not establish placement. A checked-off item can become outstanding again.

**Decision and action.** Reconstruct or inspect current inventory; compute each deficit; obtain only needed supplies; physically stock them and close relevant containers. Permit current-state inspection in the general task. A separately labelled memory diagnostic can limit reinspection, but must not conflate that restriction with the main benchmark.

**Physical completion.** For category c, require n_final(c) to satisfy the public target interval. Proposed minimum-stock tasks use n_final(c) >= target(c), with no unauthorized extra additions; exact-order variants use equality. Record preservation of existing surplus, stable placement of new units, and container closure. Do not pin interchangeable mugs to arbitrary source IDs or cabinets unless the goal requires that distribution.

**Counterfactuals.** One previously stocked item is removed versus retained; or a delivery attempt fails versus succeeds with matched observed feedback. Correct deficit changes by one. Include target already met. Vary target counts separately to defeat the fixed “add two towels and one mug” source solution.

**Difference from V1.** GG-1 collects specified objects and TM-2 restores changes; this prototype tracks goal satisfaction under the robot's own action history and computes the remaining multiset. It is an extension within existing capabilities, not evidence that WIA previously lacked memory.

**Difficulty.** Low–medium for rigid packages on reachable shelves; medium–high for doors and crowded cabinets. Avoid fabric/deformable socks in the first realization. Main risk: turning the task into arithmetic on text followed by token placement; use actual inventories and actual changes.

## P4. Clear only enough space to make the next action feasible

**Source:** [NARRATE-12802](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/accessibility/NARRATE-12802.taskpack.json). **Overlap reference (not assigned category):** GF with AP planning. **Priority:** high after replacing the fixed three-removal script.

**Goal.** Place a target object in a bounded work region while preserving fixed anchors and minimizing unnecessary relocations.

**Knowledge.** Target shape, accessible orientation, gripper withdrawal space and the currently cleared region. The number of removed obstacles is not itself the success condition.

**Decision and action.** Inspect geometry, choose which movable obstructions to relocate, reobserve the opening, and stop clearing when a feasible placement and retreat exist. Allow different valid removal orders and orientations.

**Physical completion.** Target fully supported inside the specified footprint with legal collision-free approach and retreat; anchors remain fixed; relocated obstacles stably supported in allowed storage. Report unnecessary moves as efficiency, not a hidden failure condition, unless a public move budget is part of the task. The source's predicate on(scale, counter) is insufficient to prove placement inside the intended gap.

**Counterfactuals.** Change target width while preserving the scene so that one versus two versus three relocations are necessary; separately alter one obstacle's location to change which obstacle is useful to move. Include an initially sufficient gap.

**Difference from V1.** D1 insertion and T22–T25 containment select feasible objects/poses in fixed geometry. Here the robot changes the geometry of its own future action space. If every version always requires removing all three bottles, the task is prescribed cleanup and adds little.

**Difficulty.** Medium; requires footprint/collision and minimum-move reference checks. Static AABB fit alone cannot establish legal gripper access. No new chemistry or fluid model is needed.

## P5. Conditional destination revision while carrying an object

**Source:** [ASK-30000](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/decision/ASK-30000.taskpack.json). **Overlap reference (not assigned category):** AP-4 with GG/TM. **Priority:** high as a compact probe.

**Goal.** Deliver a package to the preferred station unless that station violates the public acceptance condition; then use the compatible alternative.

**Knowledge.** A remembered conditional rule and newly observed destination state. Replace arbitrary “cat-new/dog-old” cross-reference complexity with a meaningful workspace acceptance rule while retaining a separate rule-composition diagnostic if desired.

**Decision and action.** Acquire the package, inspect the preferred station before release, select primary or fallback destination, carry and place the package, and preserve the obstruction or resident item. Permit early inspection when visible; do not impose the source's exact walk/look sequence.

**Physical completion.** Package stably supported at the destination selected by the observed acceptance condition; prohibited destination untouched; resident contents preserved; hands withdrawn. The evaluator computes the valid destination set from the episode condition, not a fixed fallback ID.

**Counterfactuals.** Primary station available versus occupied with matched package and rule. Separately swap which fallback has sufficient geometry or appropriate category. Include a feasible primary station so always diverting fails.

**Difference from V1.** TM-2 already handles changes and GF already handles fit. The useful combination is conditional revision at a placement commitment boundary during a live delivery, not merely asking which table is correct.

**Difficulty.** Low–medium for rigid objects and reachable tables. A mandatory fallback rule without any meaningful physical acceptance criterion is weaker and should remain a diagnostic, not a headline contribution.

## P6. History-dependent functional identity

**Sources:** [WATCH-28002](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-28002.taskpack.json), [WATCH-34002](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-34002.taskpack.json), [WATCH-81019](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/prevention/WATCH-81019.taskpack.json), [WATCH-14014](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/finding/WATCH-14014.taskpack.json). **Overlap reference (not assigned category):** TM-1/TM-2. **Priority:** one carefully controlled extension; merge source variants.

**Goal.** Route visually identical tools or containers according to what happened to them, preserving unused items for a different destination.

**Knowledge.** Object identity persists across motion; use history determines current task eligibility even when appearance does not change. Begin with rigid handled cleaning pads or visually identical containers to reduce cloth-control confounds.

**Decision and action.** Observe or perform a distinguishing contact/use event, follow the object through relocation, identify its current instance, and physically separate eligible from ineligible stock. History must be available through robot observations; private labels such as sponge_used cannot be exposed.

**Physical completion.** Used item in the used-items tray; unused item in clean storage; no forbidden cross-placement anywhere in the episode if that is publicly specified. This is event-conditioned segregation. It does not demonstrate microbial contamination or chemical composition without an independently implemented state model.

**Counterfactuals.** Identical final layouts with which object received the use event swapped. Keep the event observable and preserve identity continuity through moves. Include no-use and both-used variants only if the public goal defines their valid action sets.

**Difference from V1.** Existing TM tasks already bind history to physical targets; this adds an interaction-induced functional status rather than location change. The two-/five-towel versions are difficulty variants of one task, not independent capabilities. This should not displace stronger AP-4 candidates.

**Difficulty.** Medium for event-grounded identity and anti-leakage; high if actual contamination transfer or liquid composition is claimed. Blindly replaying a “used object goes left” script must fail matched identity swaps.

## P7. Capacity-constrained physical allocation

**Sources:** [GUIDE-24003](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-24003.taskpack.json); related [GUIDE-109009](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-109009.taskpack.json), [GUIDE-99806](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/guidance/GUIDE-99806.taskpack.json) and [GUIDE-102000](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/GUIDE-102000.taskpack.json). **Overlap reference (not assigned category):** AP planning with GG; a resource-constraint tag is preferable to inventing a new family immediately. **Priority:** medium–high, conditional on a nontrivial allocation design.

**Goal.** Fulfill physical orders under a declared resource limit and existing stock. Sealed units are indivisible; the robot must choose, transport and distribute actual units.

**Knowledge.** Package quantity and unit conversion, current stock, destination requirements, and eligibility. Labelled fluid content is a logical stock quantity; it is not the geometric volume occupied by a package.

**Decision and action.** Read quantities, update the remaining allowance from actual deliveries, choose a valid subset, and place only that subset in the designated destinations. Use matched alternatives so a greedy largest-first choice sometimes fails.

**Physical completion.** Actual delivered units satisfy each order's quantity constraints, with no unit counted twice; distractors and unused stock remain outside destinations. All units fit and remain stable. A declared stock limit is acceptable as an operational constraint, but must not be described as simulated shelf strength or geometric packing capacity.

**Counterfactuals.** Alter existing stock by one observed removal while keeping package labels fixed; the valid number of additions changes. Separately alter one package quantity so the optimal subset changes. For the source's values: 5.0 - 1.8 - 1.6 = 1.6 litres, so two 0.8-litre units fill the allowance; this arithmetic does not prove a shelf can physically hold them.

**Difference from V1.** D2 sorts by a metric, and GG collects a fixed set. This adds coupled selection under a shared resource constraint. A single arithmetic question followed by moving a numbered answer card is not an equivalent addition.

**Difficulty.** Medium; quantity extraction, interchangeable-object scoring and physical counting are the main needs. No liquid pouring is required.

## P8. Resource-aware scheduling with physical handoffs

**Sources:** [ORCHESTRATE-44001](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/ORCHESTRATE-44001.taskpack.json), [ORCHESTRATE-99902](/local_home2/zhouxinyang/models/InteractiveWorld/benchmark/tasks/taskpacks/planning/ORCHESTRATE-99902.taskpack.json). **Overlap reference (not assigned category):** AP planning. **Priority:** scientifically strong, defer admission until timing semantics are implemented and checked.

**Goal.** Complete compatible jobs using two limited processing stations and a single robot before a task-clock deadline, with actual loading, unloading and delivery.

**Knowledge.** Precedence, station compatibility, job durations, exclusive occupancy and transport/preparation costs. Distinguish learning these durations experimentally from reading them; only the former tests AP-3.

**Decision and action.** Select feasible jobs, schedule robot preparation during autonomous station processing, unload completed work, and replan after an observed duration change. Food/allergy rules can be a later compositional extension, not a reason to begin with nine dishes.

**Physical completion.** A job counts only when its required station process has elapsed while its object occupies the valid location and it is then physically delivered. Prevent overlapping exclusive use. Score completed valid jobs and makespan, with safe final station states. Define simulation clock, resource occupancy and when model reasoning pauses the world before any experiment. Model/API wall time must be reported separately.

**Counterfactuals.** Swap one station's job duration while holding placements fixed; optimal order changes. Separately make a candidate ineligible. Validate the optimal schedule with a small enumerated instance, including robot occupancy, before describing a schedule as optimal.

**Difference from V1.** T01–T03 prescribe an order; this requires deriving a resource-feasible order and overlapping independent processes. Current source cooked predicates and command order do not by themselves validate continuous-time optimization.

**Difficulty.** High. A timed state-machine process is a valid explicitly declared abstraction, but not thermal physics. Do not inherit the source's stated minutes as simulated durations without checking implementation. This is a research direction, not the easiest first task to add.

## Current design policy

The user permits knowledge-guided real pick-and-place, ranked by coupling to the physical task. Start from the scientific question and adapt the suitable InteractiveWorld scene directly. Existing WIA categories are comparison references, not constraints or assigned destinations. Classification and V1/later-release placement remain open. See [scene redesign plan](/local_home2/zhouxinyang/models/WIA/docs/research/interactiveworld_candidates_20261029/scene_redesign_plan.md). No scene, task or runtime implementation is changed in this documentation phase.

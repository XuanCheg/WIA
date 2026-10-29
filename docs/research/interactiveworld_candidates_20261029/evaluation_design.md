# Proposed diagnostic evaluation design

This is a design specification for future validation, not an experiment conducted in this review. Do not attach numerical performance claims to these proposals.

## Test the knowledge-to-action claim directly

For each prototype, define the minimum sufficient world evidence E, the decision D that depends on it, the permitted action set A, and terminal physical condition G. Construct paired worlds where changing E changes the valid D/A while the task goal and unrelated execution difficulty remain controlled. An object label swap alone is useful for anti-memorization but does not establish a new capability.

Use four conditions on the same task instances:

| Condition | Purpose | Interpretation limit |
|---|---|---|
| Native observations and full control | End-to-end performance | Mixes perception, reasoning and execution failures. |
| Correct task-relevant facts supplied, same control | Isolate failure after evidence acquisition | Providing facts changes the information condition and must be reported. |
| Correct high-level decision supplied, same control | Estimate execution difficulty | This is an oracle-decision diagnostic, not autonomous task success. |
| Matched counterfactual or no-change trial | Test decision dependence and unnecessary intervention | Both members need valid physical solutions; a harder grasp must not explain the difference. |

A language-only knowledge question may be an auxiliary probe. It must not replace physical success or make agents reveal private evaluator identifiers. Comparing knowledge correctness with action correctness is informative only when both refer to the same evidence and decision, and the experiment controls the extra information introduced by asking a question.

## Score valid outcomes rather than reference programs

Let G be the set of required terminal physical predicates and H the public process invariants. A proposed episode passes when every g in G holds at termination and every h in H holds over its stated interval. If a no-change episode already satisfies G, verification and preservation may be correct; do not force an irrelevant grasp to earn credit. Report repaired versus initially satisfied conditions separately.

A process invariant should have a scientific or operational reason: keeping incompatible items separate, preserving an obstacle that may not be moved, completing preparation before activation, or physically passing through an opening. It should not encode the source's arbitrary object order, exact walk/look script or semantic utterance.

For interchangeable objects, match by category/attributes and actual world relations rather than fixed authored IDs. For P3, deficit(c) = max(0, target(c) - current(c)); whether target is a minimum or exact quantity must be public. For P7, count each delivered physical unit once and check coupled quantities across destinations. For P8, resource intervals and robot occupancy must be defined before optimizing a schedule.

## Failure attribution and cost

Record observation availability, decision identity, attempted action, execution result and final world state. Distinguish: insufficient observation; wrong interpretation/selection; stale plan; correct decision with failed grasp/placement; evaluator or scene failure. Do not classify a missing score as a failed policy.

Report full-goal success, preservation/process violations, unnecessary relocations, recovery success, physical action count, simulated task time, wall time, model latency and throughput. Separate one-time scene loading from episode costs. Compare latency only under matched model, sensor, precision and controller settings. Scheduling uses its explicit task clock, never silently model-request wall time.

## Minimal future checks

Before any benchmark run, verify only the necessary scientific invariants: decisive cues are visible through allowed sensors; paired variants require different valid actions; source IDs/private state do not leak; a correct robot plan can physically complete both branches; incorrect but tempting plans fail the relevant physical/process condition; no-change branches permit justified restraint. A few controlled native episodes are more informative than broad unit-test coverage at this stage. These checks are proposed, not performed.

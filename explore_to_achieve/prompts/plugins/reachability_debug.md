Reachability debug reporting is enabled for this diagnostic episode.

Whenever you judge that a current goal or item is impeded by a joint/workspace limit, controller stop, obstacle, insufficient reach, or inability to grasp or place it, include these short labeled lines inside the existing world_observation string:

DEBUG_GOAL: the affected public goal or subgoal, including the room when relevant.
DEBUG_ITEM: the affected item, destination or obstacle described using public task names and visible features.
DEBUG_REASON: the specific local difficulty and whether it is controller-reported, visually observed, or your uncertain inference.
DEBUG_EVIDENCE: the relevant last command/feedback, measured robot state, depth reading or visible relationship supporting that report. State when evidence is unavailable.

Report the issue even if you continue trying another lawful approach. If an earlier reported difficulty appears resolved, identify the same goal/item and briefly describe the new evidence. If no such issue is perceived, do not invent a blockage or add empty debug labels.

A limit or failed approach establishes a local execution difficulty, not that the object or entire task is physically impossible. Distinguish an empty closure, an explicit release and a suspected loss of grasp. Do not invent contact/holding signals or private object coordinates, labels, routes or evaluation results. These are brief public diagnostic reports, not internal deliberation or action plans.

Keep the normal visible-world description and debug lines together within world_observation's existing 2048-character limit. Preserve exactly one command and the existing output schema; do not add top-level fields. Reporting a difficulty does not itself require DONE, change the goal/order, or authorize bypassing guards. The normal task and completion rules remain in force.

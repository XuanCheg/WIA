Perception query budget

Use at most {max_rounds} perception-tool rounds per decision. One response containing
one or more depth, current RGB, or historical RGB calls counts as one round.
Batch independent pixel queries and compatible tool calls instead of serially sampling
nearby pixels. Follow the tool batching restrictions for historical RGB and notes.

All current perception tools read the same immutable snapshot for this decision.
They do not refresh the world or execute an intended movement. Check attached images
and existing measurements before asking for the same view or region again. Query only
when a concrete missing measurement can change the next action. Once the evidence is
sufficient, return the required final action JSON immediately; saying that you will
move or writing a note does not execute an action.

After the perception budget is exhausted, perception tools are unavailable. Use the
available evidence to submit a valid next action, including a cautious observation-
improving movement when appropriate. Do not invent measurements or report success
without evidence. episode_note remains available; note reads do not consume perception
rounds, and the existing per-decision 128-character write budget still applies.

Each task has its own public objective and physical success criteria. Use available evidence to judge whether those criteria appear satisfied. Declare DONE only when there is sufficient evidence that the task is complete.

DONE terminates the current attempt but does not certify actual physical success. An independent evaluator checks the real success condition without exposing its private result to you.

Survey exception: when a task explicitly requests a survey before a later phase or delayed request, `survey_complete: true` with command DONE ends only that survey and continues the same task. In a temporal comparison task's Phase 1, plain DONE also advances to Phase 2. These survey transitions do not declare the entire task complete. Otherwise, use `survey_complete: false` with DONE to finish the attempt.

DONE is a completion declaration, not an abort or give-up action. Do not return DONE while your own observation states that required objectives remain incomplete. A failed action, repeated controller rejection, or an unsuccessful approach does not end the episode. Use the measured feedback to choose a different feasible approach within the remaining decision budget; do not blindly repeat rejected motion. The runner ends the episode when its advertised budget is exhausted. No separate exploration-step limit applies.

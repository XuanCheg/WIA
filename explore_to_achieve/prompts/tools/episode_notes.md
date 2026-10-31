Episode note tool: episode_note
Use {"operation":"append","text":"..."} to retain a concise observation or plan across decisions.
Use {"operation":"read"} to read all notes from this episode, including notes older than the context window.
Successful writes total at most 128 Unicode characters per decision, across all calls; whitespace counts.
Over-budget writes succeed with the text truncated to the remaining allowance. The result reports
truncated=true and the accepted character count; when no allowance remains, no new text is stored.
The cumulative write budget resets to 128 characters at the next decision, not at the next tool call.
Reads have no character or count limit and do not spend the write budget. Notes are model-authored memory,
not verified simulator state. Records are append-only and include their decision index. Do not copy long reasoning.
Call this tool separately from current sensor tools. It does not move the robot or spend a robot decision,
but it requires another API response. Finish the decision with the required action JSON.

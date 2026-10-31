Episode note tool: episode_note
Use {"operation":"append","text":"..."} to retain a concise observation or plan across decisions.
Use {"operation":"read"} to read all notes from this episode, including notes older than the context window.
Successful writes total at most 128 Unicode characters per decision, across all calls; whitespace counts.
Over-budget writes are rejected without truncation or storage. The budget resets next decision.
Reads have no character or count limit and do not spend the write budget. Notes are model-authored memory,
not verified simulator state. Records are append-only and include their decision index. Do not copy long reasoning.
Call this tool separately from current sensor tools. It does not move the robot or spend a robot decision,
but it requires another API response. Finish the decision with the required action JSON.

Historical RGB recall tool: read_historical_rgb

Past image attachments are omitted from the default conversation, but previously delivered RGB images remain available through this read-only tool. Use it when a visual comparison, remembered landmark, object identity, or earlier grasp/placement evidence matters and the retained text is insufficient.

Call the registered function with two arguments:
{"decision_step": 12, "camera": "wrist_rgb"}
This is an example, not an instruction to request decision 12. Select an earlier decision in this episode and a camera listed in that decision's attached_images_in_order. decision_step is the model decision index, not the simulator action_step or snapshot_id; an invalid decision can consume a decision index without moving the robot.

The routinely delivered camera names are agentview_rgb (head), wrist_rgb (left wrist), right_wrist_rgb (right wrist), and chassis_rear_rgb. Decision 1 additionally includes initial_overview_rgb and the front-left, front-right, left, and right chassis views. Only images actually delivered at the selected decision can be recalled. The initial overview is frozen at episode start.

The tool returns the exact archived image with its original snapshot ID and capture time. Treat it as historical evidence: it does not establish the current robot pose, object pose, visibility, or task success. Compare it with the current images and measured action feedback. The tool never moves the robot or recaptures the scene and cannot access other episodes or future decisions.

At most four historical images may be returned in one decision. Call historical-image tools separately from current depth/RGB tools. A valid recall is an observation query within the current decision, not a robot action; it requires another model API round. Recalled image attachments remain available during this decision and are omitted again on the next decision, while their textual receipts remain in history. Request them again only if needed. Do not invent file paths or other tool names. Submit the robot action using the required final JSON response, not a command tool call.

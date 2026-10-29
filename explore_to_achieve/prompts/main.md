[resident]
@include core/identity.md
@include core/perception.md
@slot ROBOT_PROFILE
@slot ACTION_INTERFACE
@slot PERCEPTION_TOOLS
@include core/robot_operation.md
@include core/execution_feedback.md
@include core/protection.md
@include core/completion.md
@plugin world_observation

[episode_start]
@include phases/episode_start.md
@slot PUBLIC_TASK
@slot PUBLIC_ENVIRONMENT
@slot EPISODE_CONFIGURATION
@slot INITIAL_OBSERVATION
@slot INITIAL_ROBOT_STATE
@slot OUTPUT_CONTRACT

[decision_step]
@include phases/decision_step.md
@slot CURRENT_SENSOR_OBSERVATION
@slot CURRENT_ROBOT_STATE
@slot LAST_EXECUTION_FEEDBACK
@slot CURRENT_ACTION_CAPABILITIES
@slot REMAINING_BUDGET
@slot OBSERVATION_PROVENANCE
@slot OUTPUT_CONTRACT

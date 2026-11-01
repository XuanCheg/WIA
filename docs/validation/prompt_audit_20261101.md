# Release prompt and declaration audit

Reviewed all 60 active release public task contracts, shared prompt components, declaration parsing and selected evaluator requirements. Also scanned 546 saved fixed-policy inputs and rendered 240 current modular prompts using CPU fixture sensors, covering first/subsequent decisions and both cache-prefix layouts. No model API or simulator was started for this audit. Saved fixed-policy inputs use the generic decision renderer; they are not evidence that the modular model-session prompt was previously exercised natively.

The main confirmed defect was the response interface: the world-observation plugin said its text was not scored, while the runner parsed task answers, arrivals and survey markers from that text. At the user's request, all three declaration types now use independent structured fields. There was no direct private-answer/state injection found in the inspected public projections and scanned markers. This is a bounded source/artifact audit, not proof against every possible visual or semantic shortcut.

## Implemented response contract

Physical model sessions now advertise `structured_fields_v1` in prompt provenance:

```json
{
  "command": {"action": "DONE"},
  "world_observation": "The item appears to rest inside the container.",
  "answer": "inside",
  "arrived_rooms": [],
  "survey_complete": false
}
```

- `answer`: a string, a mapping where the task asks multiple questions, or null when not answering. T23's mapping uses the public keys `granola box`, `spray can`, `colored pencil`; T05/T06 use `goal_1` through `goal_5`. Keys are derived solely from public questions, not private bindings or expected answers.
- `arrived_rooms`: current room-instance declarations, using underscores in public names. Claims still require the existing native occupancy/stability evidence. Empty means no declaration.
- `survey_complete`: a boolean. True must accompany command DONE and answer=null at an available survey phase. T38 receives its delayed request and continues without a robot motion or terminal DONE declaration. Temporal Phase-1 plain DONE retains its existing phase-transition behavior.
- `world_observation`: descriptive logging only. Embedded `ANSWER:`, `ARRIVED:` and survey markers no longer produce declarations. The existing `observed_objects` field remains separate for T09–T11 and is checked against delivered camera evidence.

Twenty active task definitions have updated response instructions and incremented task versions. Their private evaluator definitions and physical geometry are unchanged by this audit. T52 is now task version 3 due to the interface migration; its inside-plus-active-DONE credit rule is unchanged and permits separate decisions in one episode. The fixed diagnostic driver also prepares the modular prompt/schema when supplied the native task manifest and emits the new fields. The T52 builder and diagnostic sequence are updated. Historical raw artifacts are preserved.

Implementation: [task response schema/parser](/local_home2/zhouxinyang/models/WIA/src/wia/runtime/agent/task_response.py), [prompt assembler](/local_home2/zhouxinyang/models/WIA/src/wia/runtime/agent/prompts.py), [runner](/local_home2/zhouxinyang/models/WIA/src/wia/runtime/agent/zero_shot.py), [prompt documentation](/local_home2/zhouxinyang/models/WIA/explore_to_achieve/prompts/README.md).

## Findings and disposition

| Finding | Evidence and effect | Disposition |
| --- | --- | --- |
| Observation text had contradictory scoring instructions | Shared plugin prohibited scoring its text, but runner parsed ANSWER/ARRIVED and temporal markers. | Repaired with typed fields and explicit dispatch; no text-marker fallback. |
| T23 advertised one container label for three targets | Its evaluator requires a label-to-container JSON mapping, while its public instruction requested a scalar. | Repaired with the structured mapping schema and matching public instructions. |
| T05/T06 omitted arrival syntax | Composite public instructions requested room reports without defining the strict old ARRIVED syntax. | Repaired with `arrived_rooms` schema and public room-name normalization instructions. |
| Survey DONE conflicted with global terminal DONE | Temporal tasks used DONE to transition, and T38's delayed request could be followed by accidental whole-task termination. | Added the explicit survey exception and tested continuation without motion or terminal credit. |
| M4 lacks a model-visible simulation clock | The public task requires 4/12/24-second processing and a 120-second deadline. MCP exposes simulation timing for runner telemetry, but the policy's observation/action-context projection does not include that clock. Wall capture timestamps do not provide simulation elapsed time. | Open, substantive information gap. Recommend exposing sampled episode-relative simulation time and its capture boundary, without private station completion or scoring flags. No clock/scoring change made here. |
| T05/T06 room descriptions hint at queried relations | T05 goal_2 names bedroom 1 as “near living room 0” while asking adjacency to that living room; T06 goal_5 similarly describes corridor 0 as near classroom 0. Both stored answers are Yes. Nearness is not proof of a direct connection, but the wording offers a shortcut cue before inspection. | Open semantic shortcut risk, not confirmed deterministic answer leakage. Replace with visually distinguishable neutral landmarks only after checking actual views; do not invent landmarks or remove needed identity information. Required physical visits still apply. |
| “Best fit” is underspecified in T22–T25 | The scorer checks source-selected container identity independently of physical partial insertion. Partial insertion alone does not define which of several containers is best. | Lower-priority definition review. Verify the source selection criterion before proposing wording; do not silently assume a new smallest-container rule. |
| T09 distance convention differs from possible readings of “closest” | See the numerical review below. All tested definitions agree for the actual release instance. | Not a current answerability defect. Per user feedback, retain its question meaning, distance rule and answer; only its response interface changes. |
| Optional original-QA and older authoring templates still mention text answers | Original-QA has a separate answer/confidence protocol, while its generic completion template asks for the answer in world_observation. Some older converters/scripts still generate text-marker instructions. | Outside the active sixty-task physical migration. Before using those modes or regenerating release tasks, align their public interface deliberately; do not silently reinterpret historical artifacts. |

## T09: actual evidence and limits

The native qualification reconstructs room bounding rectangles from segmentation, applies the source's wall-limited expansion, and compares the chandelier's horizontal center with the resulting regions. It uses room-center distance for ties. This is a horizontal geometry convention, not walking-path distance. The original source distances and actual native measurements agree within their declared tolerance.

| Candidate | Raw rectangle distance | Expanded rectangle distance | Center distance |
| --- | ---: | ---: | ---: |
| Dining room | 0.00 m | 0.00 m | 6.32 m |
| Kitchen | 3.66 m | 3.01 m | 7.03 m |
| Bathroom | 14.73 m | 14.73 m | 17.79 m |

All three metrics select the dining room, and the native segmentation assigns the chandelier there too. The user correctly pointed out that the hidden geometric convention does not prevent answering this instance. The initial audit concern is therefore downgraded; adding implementation detail to its prompt is unnecessary.

A numerical boundary-point probe using the same recorded rectangles can change the winner between raw and expanded regions. This demonstrates a possible future boundary-case distinction only: it is not an authored task, a physically placed chandelier, a reachable robot position or a new native run. Keep this as an authoring-definition note for future instances. Do not present it as a failure of current T09. See [numerical evidence](/local_home2/zhouxinyang/models/WIA/outputs/prompt_audit_20261101/t09_distance_review.json).

## Leakage boundaries checked

`LoadedTask.public_payload` exposes only public task fields. The modular assembler accepts that closed projection; the physical action context contains controller limits and public phase information. Private expected answers, target bindings, relation measurements, event choices and completion scores remain outside policy inputs. Candidate event history supplies recorded RGB frames and timestamps, not private event/object labels. T38 publishes requested categories/counts after the survey while retaining selected instance keys privately. Task goals, public answer options, deadlines and legally required process rules are intended instructions, not answer leaks.

The marker scan found zero occurrences of the enumerated private-state keys across 546 saved input prompts. This does not prove that all image cues are sufficiently visible or that all descriptions are free of semantic hints. No new image-based answerability certification or model performance claim is made.

## Validation and evidence

The final combined CPU scope passes **1378 checks**, with **54 previously documented baseline cases deselected**, in 61.13 seconds. Three previously excluded temporal fixture assertions were updated to distinguish decision count from unknown model API count; they now run. Response migration coverage includes typed mappings, malformed values, normalization, camera declarations, text-only false-credit rejection, separate-decision T52 confirmation, survey continuation and HTTP/MCP transport. This validates the changed interface in CPU/native fixtures; no GPU replay or real provider call was performed after the migration.

Final parser review added an explicit command-object check so malformed survey commands raise a recoverable validation error. After that small change, the focused task-response suite passes **15 tests in 2.77 seconds**; the combined suite was not repeated. Current modular render artifacts were refreshed after the final prompt edits.

- [CPU command and exclusions](/local_home2/zhouxinyang/models/WIA/outputs/prompt_audit_20261101/cpu_command.json), [CPU output](/local_home2/zhouxinyang/models/WIA/outputs/prompt_audit_20261101/cpu.log).
- [All sixty public contracts before/after](/local_home2/zhouxinyang/models/WIA/outputs/prompt_audit_20261101/inventory.json).
- [Saved input scan](/local_home2/zhouxinyang/models/WIA/outputs/prompt_audit_20261101/saved_prompt_scan.json).
- [Modular render audit](/local_home2/zhouxinyang/models/WIA/outputs/prompt_audit_20261101/render_audit.json).

The previous full native batch and T52 positive/negative runs remain evidence for their recorded interface versions. They are not relabeled as native validation of `structured_fields_v1`. No commit or push was made.

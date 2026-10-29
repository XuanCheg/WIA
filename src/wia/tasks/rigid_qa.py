"""Adapt the shared ESI QA protocol to native R1 commands.

Run .venv/bin/python src/wia/runtime/commands/run_esi_agent.py --scene-file FILE --variant tabletop.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

PROTOCOL_PATH = Path(__file__).with_name("rigid_containment_protocol.py")
spec = importlib.util.spec_from_file_location("esi_rigid_containment_protocol", PROTOCOL_PATH)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


DEFAULT_QA_STEPS = 60


class RigidContainmentQA:
    def __init__(self, scene, *, threshold=0.8, min_steps=3, goal="answer", pickup_location=None):
        if goal not in ("answer", "answer_and_place"):
            raise ValueError("Unknown containment QA goal.")
        self.goal = goal
        if pickup_location is not None and (goal != "answer_and_place" or not pickup_location.strip()):
            raise ValueError("A pickup location requires the combined answer-and-place goal.")
        self.pickup_location = pickup_location
        raw = json.loads(scene.source_path.read_text())
        if raw.get("small_task") != "Rigid Containment":
            raise ValueError("The shared containment QA protocol requires a Rigid Containment source.")
        metadata = json.loads(raw["metadata_json"])
        self.payload = {"containee": {"cat": metadata["containee"]["cat"]}, "containers": [
            {key: item[key] for key in ("cat", "idx", "fit_check", "_key") if key in item}
            for item in metadata["containers"]]}
        self.threshold, self.min_steps = threshold, min_steps
        self.source_sha256 = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
        original = protocol.build_system_prompt(self.payload, threshold, min_steps).splitlines()
        start, guidance = original.index("Output EXACTLY one valid JSON object and nothing else:"), original.index("Guidance:")
        self.public_prompt = "\n".join(original[:start] + original[guidance:]).strip()
        self.public_payload = dict(scene.public_payload, constraints=[self.public_prompt,
            "Task operations use real R1 motion and gripper commands. Pick-up and insertion require sequences of native commands; there is no mock grasp or automatic placement sampler."])
        if goal == "answer_and_place":
            target = self.payload["containee"]["cat"].replace("_", " ")
            self.public_payload["success_criteria"] = (
                f"Answer the original question and physically pick up the {target}, place it inside "
                "the container you judge most suitable, release it and observe stable partial insertion. "
                "Merely resting completely on top does not satisfy containment.")
            self.public_payload["constraints"].append(
                "This episode has two concurrent goals: the original ESI question and real pick-and-place. "
                "No initial answer is requested. A confident answer completes only the question subtask; "
                "continue native robot actions until the placement is complete or further progress is impossible. "
                "Use DONE to end the combined attempt and give a final container answer. "
                "The original answer confidence and containment rules still apply. "
                "Motor completion alone does not establish grasp or insertion. Report failures honestly.")
            if pickup_location is not None:
                self.public_payload["success_criteria"] = (
                    f"Retrieve the {target} from the {pickup_location}, carry it to the separate table "
                    "holding the candidate containers, place it inside the container you judge most suitable, "
                    "release it and observe stable partial insertion, and answer the original question. "
                    "Merely resting completely on top does not satisfy containment.")
                self.public_payload["constraints"].append(
                    f"The target starts on the {pickup_location}; the candidate containers remain on a separate table. "
                    "Use native robot movement to retrieve and transport it before insertion.")

    def provenance(self):
        return {"protocol": "shared_original_rigid_containment", "protocol_sha256": self.source_sha256,
                "confidence_threshold": self.threshold, "min_steps": self.min_steps,
                "action_adaptation": "real_r1_commands", "public_semantics": self.public_prompt,
                "goal": self.goal, "default_max_steps": DEFAULT_QA_STEPS,
                "qa_stop_scope": "question_subtask" if self.goal == "answer_and_place" else "episode",
                **({"pickup_location": self.pickup_location} if self.pickup_location is not None else {})}

    def verify_unchanged(self):
        if hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest() != self.source_sha256:
            raise RuntimeError("Shared QA protocol changed during the episode.")

    def parse(self, response, action):
        if not isinstance(response, dict) or set(response) != {"answer", "confidence", "reasoning"}:
            raise ValueError("Supply the original QA answer, confidence and brief public reasoning fields.")
        if not isinstance(response["answer"], str) or not isinstance(response["reasoning"], str):
            raise ValueError("QA answer and brief reasoning must be strings.")
        if type(response["confidence"]) not in (int, float) or not 0 <= response["confidence"] <= 1:
            raise ValueError("QA confidence must be a finite number between zero and one.")
        parsed = protocol.parse_model_output(dict(response, action="stop" if action == "DONE" else action), self.payload)
        # The original question uses generic container aliases even when the
        # upstream task wrapper uses category names such as medium lunch box.
        if parsed["parsed_container"] is None:
            answer = protocol.normalize_label(parsed["answer"])
            for name, item in protocol.container_by_name(self.payload).items():
                size = protocol.SIZE_DISPLAY[item.get("fit_check") or item["_key"]]
                if answer in (size, size + " container"):
                    parsed["parsed_container"] = name
                    break
        return parsed

    def should_stop(self, parsed, history, step, max_steps):
        return protocol.should_stop(parsed, history, step, max_steps, self.min_steps, self.threshold)

    def episode_should_stop(self, parsed, history, step, max_steps):
        """Apply the original QA stopping rule only to its subtask in a combined episode."""
        if self.goal == "answer":
            return self.should_stop(parsed, history, step, max_steps)
        if parsed["action"] == "stop":
            return True, "model_stop"
        if step >= max_steps:
            return True, "max_steps"
        return False, ""

    def resolve(self, history):
        return protocol.resolve_final_answer(history)

    def needs_force_choice(self, answer, reason):
        return protocol.needs_force_final_choice(answer, reason)

    def force_choice_prompt(self):
        lines = protocol.build_force_choice_prompt(self.payload).splitlines()[:-1]
        if self.goal == "answer_and_place":
            lines[0] = "The combined attempt has ended. Give the final answer using the available evidence."
        return "\n".join(lines) + (
            "\nReturn the native QA output envelope with command DONE, a container answer, confidence, "
            "brief public reasoning and world_observation. No robot motion is executed in this final selection.")

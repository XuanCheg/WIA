"""Original Rigid Containment QA rules for CPU-only harness imports.

Definitions match the official ESI rigid_containment.py task module. The harness
keeps this pure subset locally so its submodule needs no custom commits.
Run: .venv/bin/python -m pytest -q tests/test_rigid_qa_protocol.py
"""
from __future__ import annotations

from typing import Any

SIZE_ORDER = ["small", "fit", "big"]

SIZE_DISPLAY = {"small": "small", "fit": "medium", "big": "large"}

ACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "reasoning": {"type": "string"},
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["action", "reasoning", "answer", "confidence"],
}

FINAL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["answer", "confidence", "reasoning"],
}

def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()

def normalize_label(value: Any) -> str:
    return normalize_text(value).lower().replace("_", " ")

def containee(payload: dict[str, Any]) -> dict[str, Any]:
    item = payload.get("containee")
    if not isinstance(item, dict):
        raise ValueError("Missing containee in storage JSON")
    return item

def containers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("containers")
    if isinstance(raw, dict):
        items = list(raw.values())
    elif isinstance(raw, list):
        items = raw
    else:
        items = []
    output = [item for item in items if isinstance(item, dict) and normalize_text(item.get("fit_check") or item.get("_key"))]
    if not output:
        raise ValueError("Missing containers in storage JSON")
    return sorted(output, key=lambda item: SIZE_ORDER.index(item.get("fit_check") or item.get("_key")) if (item.get("fit_check") or item.get("_key")) in SIZE_ORDER else 99)

def container_object_name(container: dict[str, Any]) -> str:
    idx = container.get("idx")
    if idx is None:
        ft = normalize_text(container.get("fit_check") or container.get("_key"))
        idx = {"small": 0, "fit": 1, "big": 2}.get(ft, len(ft))
    return f"obj_container_{idx}"

def container_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {container_object_name(item): item for item in containers(payload)}

def container_display_names(payload: dict[str, Any]) -> dict[str, str]:
    items = containers(payload)
    cats = {normalize_text(item.get("fit_check") or item.get("_key")): normalize_label(item.get("cat")) for item in items}
    counts = {}
    for cat in cats.values():
        counts[cat] = counts.get(cat, 0) + 1
    output = {}
    for item in items:
        ft = normalize_text(item.get("fit_check") or item.get("_key"))
        cat = normalize_label(item.get("cat"))
        display = f"{SIZE_DISPLAY.get(ft, ft)} {cat}" if counts.get(cat, 0) > 1 else cat
        output[container_object_name(item)] = display
    return output

def display_to_container_name(payload: dict[str, Any]) -> dict[str, str]:
    return {display: name for name, display in container_display_names(payload).items()}

def build_system_prompt(
    payload: dict[str, Any],
    threshold: float,
    min_steps: int,
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    item_label = normalize_label(containee(payload).get("cat"))
    container_list = "; ".join(container_display_names(payload).values())
    return "\n".join(
        [
            "You are a spatial reasoning agent.",
            f"Your goal is to determine which container the {item_label} best fits inside.",
            "",
            f"Object to place: {item_label}",
            f"Containers: {container_list}",
            "",
            "You can pick up the object and place it into containers to physically test the fit.",
            "The next image or extra action image may show the result of your action.",
            "",
            "CONFIDENCE RULES:",
            "- An object counts as inside a container if any part of it is inside and it is stable; it does not need to be fully submerged.",
            "- An object sitting completely on top of a container, not inside at all, does not count.",
            f"- Output confidence >= {threshold:.2f} only if you can clearly see the object is at least partially inside a container and stable.",
            "- If unsure, keep exploring with actions.",
            "",
            "Output EXACTLY one valid JSON object and nothing else:",
            '{"action": "<action>", "reasoning": "<description>", "answer": "<container or unsure>", "confidence": <float>}',
            "",
            "Available actions:",
            "pick up <object> | put <object> inside <container> | move_forward | move_backward | move_left | move_right | move_up | move_down | turn_left | turn_right | turn_up | turn_down | stop",
            "",
            "Guidance:",
            "- Usually prefer physical tests over camera movement when the objects are visible.",
            "- If the object fits in multiple containers, choose the smaller suitable one for efficiency.",
            "- If the object is upright on top of a container and not tilted or partially inside, it is not inside.",
        ]
    )

def build_force_choice_prompt(
    payload: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> str:
    item_label = normalize_label(containee(payload).get("cat"))
    container_list = "; ".join(container_display_names(payload).values())
    return "\n".join(
        [
            "Exploration budget is exhausted.",
            f"You must choose which container the {item_label} best fits inside.",
            f"Containers: {container_list}",
            "Do not answer unsure.",
            'Output EXACTLY: {"answer": "<container>", "confidence": <float>, "reasoning": "<brief explanation>"}',
        ]
    )

def _parse_answer_to_container(answer: Any, payload: dict[str, Any]) -> str | None:
    text = normalize_label(answer)
    if not text or text.startswith("unsure") or text in {"not sure", "unknown"}:
        return None
    mapping = display_to_container_name(payload)
    for display, name in sorted(mapping.items(), key=lambda item: -len(item[0])):
        display_norm = normalize_label(display)
        if display_norm in text or text in display_norm:
            return name
    for name, container in container_by_name(payload).items():
        ft = normalize_label(container.get("fit_check") or container.get("_key"))
        cat = normalize_label(container.get("cat"))
        if ft and ft in text:
            return name
        if cat and cat in text and len({normalize_label(c.get("cat")) for c in containers(payload)}) == len(containers(payload)):
            return name
    return None

def parse_model_output(
    parsed: dict[str, Any],
    payload: dict[str, Any] | None = None,
    task_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    action = normalize_text(parsed.get("action")).lower() or "move_forward"
    answer = normalize_text(parsed.get("answer")) or "unsure"
    parsed_container = _parse_answer_to_container(answer, payload) if isinstance(payload, dict) else None
    return {
        **parsed,
        "action": action,
        "answer": answer,
        "parsed_container": parsed_container,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": normalize_text(parsed.get("reasoning")) or "no reasoning provided",
    }

def should_stop(
    parsed: dict[str, Any],
    history: list[dict[str, Any]],
    step: int,
    max_steps: int,
    min_steps: int,
    threshold: float,
) -> tuple[bool, str]:
    action = normalize_text(parsed.get("action")).lower()
    if action == "stop" and parsed.get("parsed_container"):
        return True, "model_stop"
    if parsed.get("parsed_container") and float(parsed.get("confidence", 0.0)) >= threshold:
        return True, "confidence_threshold"
    if step == max_steps:
        return True, "max_steps"
    return False, ""

def resolve_final_answer(history: list[dict[str, Any]]) -> tuple[str, int]:
    for item in reversed(history):
        answer = normalize_text(item.get("answer"))
        if answer and normalize_label(answer) not in {"unsure", "not sure", "unknown"}:
            return answer, int(item["step"])
    return "unsure", int(history[-1]["step"]) if history else -1

def needs_force_final_choice(answer: str, stop_reason: str) -> bool:
    return normalize_label(answer) in {"", "unsure", "not sure", "unknown"}

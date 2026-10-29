#!/usr/bin/env python3
"""Main-only candidate acceptance, queue submission and feedback recovery.

Run: .venv/bin/python src/wia/runtime/commands/submit_esi_capture.py --queue-root QUEUE submit
--handoff HANDOFF.json --code-commit HEAD. Builders return handoffs; Main submits.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

from wia.runtime.scene_capture_candidates import export_feedback, submit_candidate, validate_handoff  # noqa: E402
from wia.runtime.scene_capture_queue import CaptureQueue, read_json  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("submit", "validate"):
        item = commands.add_parser(name)
        item.add_argument("--handoff", type=Path, required=True)
        item.add_argument("--code-commit", default="HEAD")
        if name == "submit":
            item.add_argument("--timeout-s", type=float, default=1800)
            item.add_argument("--max-attempts", type=int, default=2)
            item.add_argument("--acceptance-note", default="Accepted by Main after static review.")
    commands.add_parser("status")
    item = commands.add_parser("feedback")
    item.add_argument("--review-file", type=Path, required=True)
    for name in ("cancel", "retry"):
        item = commands.add_parser(name)
        item.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "submit":
        result = submit_candidate(read_json(args.handoff), args.queue_root, code_commit=args.code_commit,
            timeout_s=args.timeout_s, max_attempts=args.max_attempts, acceptance_note=args.acceptance_note)
    elif args.command == "validate":
        accepted = validate_handoff(read_json(args.handoff), code_commit=args.code_commit)
        result = {"valid": True, "code_commit": accepted["commit"],
                  "builder_commit": accepted["builder_commit"], "esi_commit": accepted["esi_commit"]}
    elif args.command == "feedback":
        result = {"repair_requests": export_feedback(args.queue_root, args.review_file)}
    else:
        queue = CaptureQueue(args.queue_root)
        result = queue.summary() if args.command == "status" else getattr(queue, args.command)(args.job_id)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

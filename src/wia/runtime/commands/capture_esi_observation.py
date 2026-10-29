#!/usr/bin/env python3
"""Export one canonical boundary observation from a running GUMI, without a model."""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
import sys
from pathlib import Path

import requests

ROOT = WIA_RUNTIME_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observations import AgentObservation


def capture(url: str, output: str | Path) -> Path:
    # Never fetch /api/state, /snapshot/* or task/evaluator metadata to build input.
    response = requests.get(url.rstrip("/") + "/api/agent-observation", timeout=30)
    response.raise_for_status()
    observation = AgentObservation.from_payload(response.json())
    if not observation.action_boundary:
        raise ValueError("The server did not return a stable action-boundary observation.")
    return observation.save(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8615")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(capture(args.url, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Resolve upstream and project-authored sources without modifying ESI-Bench.

Run validation: python -B -m pytest -q tests/test_authored_sources.py
"""
from pathlib import Path

from wia.resources import ROOT, ESI_ROOT
UPSTREAM_ROOT = ESI_ROOT / 'dataset/json_clean'
AUTHORED_ROOT = ROOT / 'explore_to_achieve/sources/json_clean'


def resolve_source(identity, *, data_root=None):
    """Keep explicit snapshot roots isolated; otherwise search original then authored."""
    roots = [Path(data_root)] if data_root is not None else [UPSTREAM_ROOT, AUTHORED_ROOT]
    for root in roots:
        root = root.expanduser().resolve()
        candidate = (root / identity).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError('Source identity must stay inside its selected data root.')
        if candidate.is_file():
            return candidate
    raise ValueError('Source must be an existing JSON inside the selected data root.')


def relocated_source(path):
    """Resolve a historical upstream path only when its authored bytes moved locally.

    Callers still verify the recorded hash. Existing files and unrelated missing
    paths retain their original meaning; raw capture records are never rewritten.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file() and path.is_relative_to(UPSTREAM_ROOT):
        target = (AUTHORED_ROOT / path.relative_to(UPSTREAM_ROOT)).resolve()
        if target.is_relative_to(AUTHORED_ROOT) and target.is_file():
            return target
    return path

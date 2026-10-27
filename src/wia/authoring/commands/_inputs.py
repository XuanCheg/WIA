"""Lazy historical authoring inputs. Run: python -m pytest tests/test_authoring_commands.py."""
from collections.abc import MutableMapping
import json
import os
from pathlib import Path

from wia.resources import ROOT, resolve_migrated_path

ARTIFACT_ROOT = Path(os.environ.get("WIA_ARTIFACT_ROOT", ROOT / "outputs")).expanduser().resolve()


def project_path(path, *, root=ROOT):
    """Resolve project paths and explicitly redirected historical output inputs."""
    path = Path(path)
    if path.is_absolute():
        return resolve_migrated_path(path, root=root).resolve()
    if path.parts and path.parts[0] == "outputs" and os.environ.get("WIA_ARTIFACT_ROOT"):
        return ARTIFACT_ROOT.joinpath(*path.parts[1:]).resolve()
    return (root / path).resolve()


def read_json_input(path):
    """Read a required historical input with a useful missing-artifact diagnostic."""
    path = Path(path)
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required authoring input is missing: {path}. "
            "Supply the original input file; set WIA_ARTIFACT_ROOT to its outputs directory "
            "for historical runtime artifacts. No input is downloaded automatically.") from error


class LazyMapping(MutableMapping):
    """Load the original dictionary once, when authoring first accesses it."""

    def __init__(self, loader):
        self._loader = loader
        self._data = None

    def _load(self):
        if self._data is None:
            self._data = self._loader()
        return self._data

    def __getitem__(self, key):
        return self._load()[key]

    def __setitem__(self, key, value):
        self._load()[key] = value

    def __delitem__(self, key):
        del self._load()[key]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())

    def copy(self):
        return self._load().copy()

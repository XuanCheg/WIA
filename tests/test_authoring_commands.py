"""Verify independent authoring imports. Run: python -m pytest tests/test_authoring_commands.py."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / "provenance/authoring_commands.json").read_text())
NAMES = [Path(row["destination"]).stem for row in PROVENANCE["commands"]]


def subprocess_environment():
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in ("WIA_ROOT", "WIA_ESI_ROOT", "WIA_ARTIFACT_ROOT"):
        environment.pop(key, None)
    return environment


def test_all_45_commands_import_without_harness_or_runtime_assets(tmp_path):
    script = """
import importlib, importlib.abc, json, sys
from pathlib import Path
class BlockHarness(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('core', 'scripts', 'showharness'):
            raise ImportError('Harness dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockHarness())
for name in json.loads(sys.argv[1]):
    importlib.import_module('wia.authoring.commands.' + name)
assert not any(name in sys.modules for name in ('cv2', 'matplotlib', 'pxr', 'cryptography'))
assert not list(Path.cwd().iterdir())
"""
    assert len(NAMES) == len(set(NAMES)) == 45
    result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps(NAMES)], cwd=tmp_path,
                            env=subprocess_environment(), text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_legacy_imports_are_canonical_module_and_function_objects(monkeypatch):
    harness = Path(PROVENANCE["source_repository"])
    if not (harness / "scripts").is_dir():
        pytest.skip("Legacy checkout is not installed alongside the standalone WIA package.")
    monkeypatch.syspath_prepend(str(harness))
    for name in NAMES:
        canonical = importlib.import_module("wia.authoring.commands." + name)
        legacy = importlib.import_module("scripts." + name)
        assert legacy is canonical
        if hasattr(canonical, "main"):
            assert legacy.main is canonical.main


@pytest.mark.parametrize("name", ["build_wia_registry", "build_wia_workstation_variants",
                                  "build_wia_capture_review", "inspect_wia_furnished_assets"])
def test_canonical_help_is_safe_from_another_directory(tmp_path, name):
    result = subprocess.run([sys.executable, "-B", "-m", "wia.authoring.commands." + name, "--help"],
                            cwd=tmp_path, env=subprocess_environment(), text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert not list(tmp_path.iterdir())


def test_legacy_executable_wrapper_keeps_the_canonical_entrypoint(tmp_path):
    wrapper = Path(PROVENANCE["source_repository"]) / "scripts/build_wia_registry.py"
    if not wrapper.is_file():
        pytest.skip("Legacy executable wrapper is not installed alongside WIA.")
    result = subprocess.run([sys.executable, "-B", str(wrapper), "--help"], cwd=tmp_path,
                            env=subprocess_environment(), text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert not list(tmp_path.iterdir())


def test_missing_historical_inputs_are_lazy_and_actionable(tmp_path):
    from wia.authoring.commands._inputs import LazyMapping, read_json_input
    path = tmp_path / "missing_asset_metadata.json"
    value = LazyMapping(lambda: read_json_input(path))
    assert value._data is None
    with pytest.raises(FileNotFoundError, match="WIA_ARTIFACT_ROOT"):
        value["bounds"]


def test_explicit_artifact_root_redirects_historical_inputs_only(tmp_path):
    environment = subprocess_environment()
    environment["WIA_ARTIFACT_ROOT"] = str(tmp_path / "artifacts")
    script = """
from pathlib import Path
from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT, project_path
assert project_path('outputs/example/assets.json') == ARTIFACT_ROOT / 'example/assets.json'
assert project_path('explore_to_achieve/tasks/example.yaml') == ROOT / 'explore_to_achieve/tasks/example.yaml'
from wia.authoring.commands.design_wia_furnished_round import INV
assert INV == ARTIFACT_ROOT / 'wia_v1_20261024_furnished/asset_inventory'
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_canonical_authored_records_resolve_byte_preserved_historical_paths(tmp_path):
    script = """
from pathlib import Path
from wia.resources import ROOT
from wia.authoring.commands.build_wia_authored_capture_records import build_records
package = build_records()
assert package['record_count'] == 4
assert package['native_capture_run'] is package['queue_mutated'] is False
for item in package['items']:
    assert Path(item['task_file']).is_relative_to(ROOT)
    assert Path(item['scene_file']).is_relative_to(ROOT)
assert not list(Path.cwd().iterdir())
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=tmp_path,
                            env=subprocess_environment(), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr

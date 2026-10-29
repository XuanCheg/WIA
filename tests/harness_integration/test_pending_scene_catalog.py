"""Check pending scene execution gates.

Run: .venv/bin/python -m pytest -q tests/test_pending_scene_catalog.py
"""

import pytest
import yaml

from wia.tasks.loader import discover_scenes, load_scene
from harness_integration.test_tabletop_scenes import manifest


def test_pending_import_is_excluded_and_cannot_load_missing_source(tmp_path):
    directory = tmp_path / "scenes"
    directory.mkdir()
    pending = directory / "sample.yaml"
    pending.write_text(yaml.safe_dump({"kind": "scene_import", "status": "pending",
        "source": {"provider": "esi", "json": "Physical Structure/Rigid Containment/scene/room/missing.json"}}))
    assert discover_scenes(directory) == ()
    assert discover_scenes(directory, include_pending=True) == (pending,)
    with pytest.raises(ValueError, match="Scene is pending and must be skipped"):
        load_scene(pending)


def test_full_scene_definition_also_requires_pending_review(tmp_path):
    loaded, _ = manifest(tmp_path)
    assert discover_scenes(tmp_path) == (loaded.path,)
    value = yaml.safe_load(loaded.path.read_text())
    value["status"] = "pending"
    loaded.path.write_text(yaml.safe_dump(value))
    assert discover_scenes(tmp_path) == ()
    with pytest.raises(ValueError, match="Scene is pending and must be skipped"):
        load_scene(loaded.path)


def test_discarded_record_is_excluded_even_from_pending_review_and_cannot_load(tmp_path):
    descriptor = tmp_path / "discarded.yaml"
    descriptor.write_text(yaml.safe_dump({"kind": "scene_import", "status": "discarded",
        "source": {"provider": "esi", "json": "missing.json"}}))
    assert discover_scenes(tmp_path) == ()
    assert discover_scenes(tmp_path, include_pending=True) == ()
    assert discover_scenes(tmp_path, include_discarded=True) == (descriptor,)
    with pytest.raises(ValueError, match="Scene is discarded and cannot run"):
        load_scene(descriptor)


def test_ready_catalog_scene_uses_its_local_source_copy(tmp_path, monkeypatch):
    from wia.tasks import loader

    loaded, _ = manifest(tmp_path)
    copied_source = tmp_path / loaded.definition.source.json_path
    copied_source.parent.mkdir(parents=True)
    copied_source.write_bytes(loaded.source_path.read_bytes())
    monkeypatch.setattr(loader, "SCENE_ROOT", tmp_path)
    local = load_scene(loaded.path)
    assert local.source_path == copied_source
    assert local.task_text == loaded.task_text

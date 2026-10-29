"""Check authored-source relocation and exact-root capture isolation.

Run: python -B -m pytest -q tests/test_authored_sources.py
"""
import hashlib
import json
from pathlib import Path

import pytest

from wia.tasks import sources
from wia.tasks.loader import load_scene, load_task
from harness_integration.test_task_edit2_capture_protocol import submission

ROOT = __import__('wia.resources', fromlist=['ROOT']).ROOT




def test_explicit_snapshot_root_never_falls_back(tmp_path, monkeypatch):
    upstream, authored = tmp_path / 'upstream', tmp_path / 'authored'
    upstream.mkdir()
    authored.mkdir()
    monkeypatch.setattr(sources, 'UPSTREAM_ROOT', upstream)
    monkeypatch.setattr(sources, 'AUTHORED_ROOT', authored)
    (authored / 'item.json').write_text('{}')
    assert sources.resolve_source('item.json') == authored / 'item.json'
    with pytest.raises(ValueError, match='existing JSON'):
        sources.resolve_source('item.json', data_root=upstream)
    with pytest.raises(ValueError, match='inside'):
        sources.resolve_source('../authored/item.json', data_root=upstream)
    (upstream / 'item.json').write_text('{"original": true}')
    assert sources.resolve_source('item.json') == upstream / 'item.json'

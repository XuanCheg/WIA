"""Validate current release membership and adapter loading. Run: pytest tests/test_release_catalogue.py."""
import json
from pathlib import Path
import pytest
from wia.resources import ROOT
from wia.tasks.loader import discover_tasks, load_task, load_scene
from wia.authoring.modules import import_owned_module


def test_all_preferred_release_tasks_load_with_canonical_adapters():
    catalogue = json.loads((ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
    assert len(catalogue['tasks']) == len(discover_tasks()) == 60
    assert {r['alias'] for r in catalogue['tasks']} >= {'GF2-1', 'GF2-2', 'GF3-1', 'GF3-2', 'iw_h2_02', 'iw_h3_03', 'T52'}
    assert all(not r['alias'].startswith('D') for r in catalogue['tasks'])
    for row in catalogue['tasks']:
        task = load_task(ROOT / row['task_file'], variant=row['variant'])
        task.verify_unchanged()
        assert task.definition.id == row['task_id']
        import_owned_module(task.definition.private.evaluation.module)
        scene = load_scene(ROOT / row['scene_file'], variant=row['variant'])
        scene.verify_unchanged()
    with pytest.raises(ValueError, match='canonical WIA'):
        import_owned_module('core.esi_converters.action_order')

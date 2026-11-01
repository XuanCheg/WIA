"""Check authored cloth layout ownership. Run: PYTHONPATH=src pytest -q tests/test_redrape_layout.py."""
from types import SimpleNamespace

import pytest

from wia.authoring.converters.deformable_redrape import prepare_interaction


@pytest.mark.parametrize('covered', [True, False])
def test_native_redrape_layout_consumes_structured_task_bindings(covered):
    receipt = {'accepted': covered, 'revision': 'native_cloth_redrape_20261020_v1'}
    session = SimpleNamespace(_redesigned_cloth_cover=receipt,
        _redesigned_cloth_inventory={'cloth_name': 'native_cloth', 'item_names': ['native_item']})
    result = prepare_interaction(session,
        {'cloth': SimpleNamespace(name='native_cloth'), 'item': SimpleNamespace(name='native_item')}, {})
    assert result['accepted'] is covered and result['relocated'] is False
    assert result['native_redrape_initialization'] is receipt
    assert session.category_interaction_setup is result and session._category_interaction_initialized

"""Check declared native diagnostic variant selection.

Run: python -B -m pytest -q tests/test_task_edit2_native_variants.py
"""
from types import SimpleNamespace as NS

import pytest

from wia.runtime.commands.check_esi_core_native import select_native_variant


def definition(*variants):
    return NS(scene=NS(variants={name: {} for name in variants}))


def test_original_only_reflection_does_not_invent_adapted_variant():
    assert select_native_variant(definition('original'), definition('original')) == 'original'


def test_adapted_candidate_keeps_its_declared_layout():
    assert select_native_variant(definition('original', 'adapted'),
                                 definition('original', 'adapted')) == 'adapted'


def test_explicit_variant_must_belong_to_both_exact_definitions():
    with pytest.raises(ValueError):
        select_native_variant(definition('original'), definition('original', 'adapted'), 'adapted')
    with pytest.raises(ValueError):
        select_native_variant(definition('tabletop'), definition('tabletop'))
    assert select_native_variant(definition('tabletop'), definition('tabletop'), 'tabletop') == 'tabletop'

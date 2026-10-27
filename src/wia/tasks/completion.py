"""Normalize measured task obligations without converting missing evidence to zero.

Run: python -m pytest tests/test_task_repair.py
"""
from __future__ import annotations

from copy import deepcopy

SCORING_VERSION = 'task_completion_v2'


def declared_items(config):
    """Only declare denominators whose meaning is known before execution."""
    p = config.get('parameters', {})
    if p.get('completion_items'):
        return [{**deepcopy(item), 'completed': False, 'evaluation_validity': 'unavailable'}
                for item in p['completion_items']]
    if p.get('completion_item_ids'):
        ids = p['completion_item_ids']
    elif p.get('required_order'):
        ids = [f'step:{index}:{room}' for index, room in enumerate(p['required_order'])]
    elif p.get('required_regions'):
        ids = ['room:' + room for room in dict.fromkeys(p['required_regions'])]
    elif config.get('type') != 'category_operation':
        ids = ['task']
    else:
        return None
    return [{'id': key, 'completed': False, 'evaluation_validity': 'unavailable'} for key in ids]


def progress(native):
    """Retain declared items, validity and diagnostic lower bounds separately."""
    items = deepcopy(native.get('completion_items'))
    validity = native.get('evaluation_validity', 'complete')
    if native.get('measurement_error') or native.get('measurement_errors'):
        validity = 'unavailable'
    if items is None:
        if validity != 'complete':
            total = native.get('total_goals')
            return {'scoring_version': SCORING_VERSION, 'evaluation_validity': validity,
                'completion_items': None, 'completed_goals': native.get('completed_goals', 0),
                'total_goals': total, 'completion_score': None, 'score_percent': None,
                'diagnostic_lower_bound': (native.get('completed_goals', 0) / total if total else None)}
        # Legacy indivisible native goal: preserve its actual boolean contract.
        items = [{'id': 'task', 'completed': bool(native['task_success']),
                  'evaluation_validity': 'valid', 'kind': 'terminal_condition'}]
    constraints = [native.get('continuous_constraints_satisfied', True),
                   native.get('constraints_satisfied', True)]
    constraints_ok = (False if any(value is False for value in constraints) else
                      None if any(value is None for value in constraints) else True)
    # Factual progress survives a later constraint failure. Terminal success and
    # constraint compliance are reported independently by the envelope.
    total = len(items)
    good = [item for item in items if item.get('evaluation_validity', 'valid') in ('valid', 'complete')]
    done = sum(item.get('completed') is True for item in good)
    if len(good) != total:
        validity = 'partial' if good else 'unavailable'
    score = done / total if total and validity == 'complete' else None
    return {'scoring_version': SCORING_VERSION, 'evaluation_validity': validity,
        'completion_items': items, 'completed_goals': done, 'total_goals': total,
        'completion_score': score, 'score_percent': None if score is None else score * 100,
        'diagnostic_lower_bound': done / total if total else None,
        'constraints_satisfied': constraints_ok}

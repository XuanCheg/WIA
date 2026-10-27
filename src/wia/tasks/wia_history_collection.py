"""Pure delayed historical collection queries.

Run CPU checks with: python -m pytest tests/test_wia_history_collection.py
No simulator modules are imported by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import random

OPERATION = 'history_multi_retrieval_v1'
QUERY_RULE = 'earliest_per_category'


def validate_collection_parameters(parameters):
    """Validate the declaration and return eligible per-category request caps.

    ``require_history_subset`` defaults to True: each requested category must
    leave at least one inspected instance unrequested, making selection depend
    on within-category history. Explicit False is available for future controls.
    """
    keys = parameters['target_keys']
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Target keys must be nonempty and unique.')
    for field in ('target_categories', 'public_target_labels'):
        mapping = parameters[field]
        if set(mapping) != set(keys) or any(not isinstance(v, str) or not v.strip() for v in mapping.values()):
            raise ValueError(field + ' must describe every target with nonempty public text.')
    if len(set(parameters['public_target_labels'].values())) != len(keys):
        raise ValueError('Public target labels must identify unique instances.')
    if parameters.get('query_rule', QUERY_RULE) != QUERY_RULE:
        raise ValueError('Unsupported historical collection query rule.')
    if type(parameters['followup_seed']) is not int:
        raise ValueError('The follow-up seed must be an integer.')
    minimum = parameters.get('query_min_category_instances', 2)
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Invalid minimum category inventory.')
    require_subset = parameters.get('require_history_subset', True)
    if type(require_subset) is not bool:
        raise ValueError('require_history_subset must be a boolean.')
    capacities = Counter(parameters['target_categories'].values())
    capacities = {category: count - int(require_subset) for category, count in capacities.items()
                  if count >= minimum and count > int(require_subset)}
    low, high = parameters['query_min_total'], parameters['query_max_total']
    if any(type(value) is not int for value in (low, high)) or not 1 <= low <= high <= sum(capacities.values()):
        raise ValueError('Query total bounds must fit the eligible category request caps.')
    return capacities


def seeded_collection_query(frozen_history, parameters):
    """Return public requests and private selected keys from actual frozen rows.

    Rows contain ``key``, ``snapshot_id`` and ``action_step`` in observation
    order. First encounters sharing one snapshot have no internal chronology.
    Seeded counts depend only on inventory, never on encounter order; selection
    then takes each requested category's first N actual unique encounters.
    """
    capacities = validate_collection_parameters(parameters)
    rows = deepcopy(list(frozen_history))
    keys = [row['key'] for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(parameters['target_keys']):
        raise ValueError('Frozen history must contain every declared instance exactly once.')
    if any('snapshot_id' not in row or 'action_step' not in row for row in rows):
        raise ValueError('Frozen encounters require actual observation boundaries.')
    if any(left['action_step'] > right['action_step'] for left, right in zip(rows, rows[1:])):
        raise ValueError('Frozen history must preserve observed chronology.')
    rng = random.Random(parameters['followup_seed'])
    remaining = rng.randint(parameters['query_min_total'], parameters['query_max_total'])
    categories = sorted(capacities)
    rng.shuffle(categories)
    requests = []
    for index, category in enumerate(categories):
        rest_capacity = sum(capacities[key] for key in categories[index + 1:])
        count = rng.randint(max(0, remaining - rest_capacity), min(capacities[category], remaining))
        if count:
            requests.append({'category': category, 'count': count})
        remaining -= count
    requests.sort(key=lambda request: request['category'])
    selected = []
    for request in requests:
        category = request['category']
        category_rows = [row for row in rows if parameters['target_categories'][row['key']] == category]
        boundaries = defaultdict(list)
        for row in category_rows:
            boundaries[row['snapshot_id']].append(row['key'])
        if any(len(group) > 1 for group in boundaries.values()):
            raise ValueError('Ambiguous simultaneous first encounters in queried category: ' + category)
        selected.extend(row['key'] for row in category_rows[:request['count']])
    return {'rule': QUERY_RULE, 'requests': requests, 'selected_keys': selected,
            'require_history_subset': parameters.get('require_history_subset', True),
            'total_count': len(selected)}


def public_collection_query(query, destination_label):
    """Expose historical instructions without instance answers or native IDs."""
    requests = deepcopy(query['requests'])
    instructions = '; '.join(f"the first {row['count']} {row['category']} you inspected" for row in requests)
    return {'rule': QUERY_RULE, 'requests': requests, 'total_count': query['total_count'],
        'text': ('Survey complete. Retrieve ' + instructions + '. Use your actual first-inspection history '
                 'within each requested category. Place every requested item on ' + destination_label + '.')}

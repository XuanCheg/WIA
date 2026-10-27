"""Place explicitly revised independent assets on one measured support.

Run: python -m pytest -q tests/test_task_edit2_tabletop.py
Native contact, stability, navigation and grasp checks remain with interaction_setup.
"""
from __future__ import annotations

import math
import numpy as np


POLICIES = ('shared_tabletop', 'seeded_geometry')


def tabletop_dimensions(bounds):
    """Propose footprints from unchanged per-object dimensions, not source spacing."""
    sizes = np.asarray([np.asarray(b)[:2] - np.asarray(a)[:2] for a, b in bounds.values()])
    proposals = []
    for columns in range(1, len(sizes) + 1):
        rows = math.ceil(len(sizes) / columns)
        size = np.max(sizes, axis=0) * [columns, rows] + .05 * np.array([columns + 1, rows + 1])
        size = np.maximum(size, [.60, .50])
        if np.all(size <= [2., 1.4]) and not any(np.allclose(size, p) for p in proposals):
            proposals.append(size)
    return sorted(proposals, key=lambda p: (np.prod(p), abs(p[0] - p[1])))


def tabletop_layouts(group, bounds, support_bounds, surface_z, policy):
    """Return authorized individual translations and a reproducible layout receipt."""
    low, high = [np.asarray(v, dtype=float)[:2] for v in support_bounds]
    keys = list(policy.get('layout_parameters', {}).get('object_keys') or group)
    objects = [group[key] for key in keys]
    from .interaction_setup import array
    boxes = [tuple(np.asarray(v, dtype=float) for v in bounds[obj.name]) for obj in objects]
    positions = [array(obj.get_position_orientation()[0]) for obj in objects]
    extents = np.array([(b - a)[:2] / 2 for a, b in boxes])
    offsets = np.array([((a + b) / 2 - p)[:2] for (a, b), p in zip(boxes, positions)])
    heights = [float(p[2] + surface_z - a[2] + .005) for (a, b), p in zip(boxes, positions)]
    receipt = None
    if policy['relocation_policy'] == 'seeded_geometry':
        from wia.authoring.converters.spatial_relations_geometric_configuration import seeded_initial_layout
        p = policy['layout_parameters']
        try:
            receipt = seeded_initial_layout(object_keys=keys, aabb_half_extents_xy=extents.tolist(),
                root_to_aabb_center_xy=offsets.tolist(), root_z_m=heights,
                support_aabb_xy=[low.tolist(), high.tolist()], target_shape=p['target_shape'],
                seed=p['seed'], source_identity=p['source_identity'], minimum_clearance_m=.05)
        except ValueError:
            return []
        roots = np.asarray(receipt['initial_positions'])
    else:
        roots = None
        sizes = 2 * extents
        for columns in range(1, len(keys) + 1):
            rows = math.ceil(len(keys) / columns)
            pitch = sizes.max(axis=0) + .05
            span = pitch * [columns, rows] - .05
            if np.any(span + .10 > high - low):
                continue
            origin = (low + high - span) / 2 + sizes.max(axis=0) / 2
            centers = np.asarray([origin + [i % columns, i // columns] * pitch for i in range(len(keys))])
            roots = np.column_stack((centers - offsets, heights))
            break
        if roots is None:
            return []
    translations = {obj.name: root - p for obj, root, p in zip(objects, roots, positions)}
    return [{'translations': translations,
        'bounds': {name: (a + translations[name], b + translations[name]) for name, (a, b) in bounds.items()},
        'geometric_initial_layout': receipt, 'single_shared_surface': True}]

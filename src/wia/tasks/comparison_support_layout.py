"""Enumerate translation-only comparison-pair layouts for initialization.

Run: python -m pytest -q tests/test_comparison_support_layout.py
Only horizontal AABB containment and pair association are screened here. Native
surface contact, collision clearance, robot reach, IK and grasp require separate
simulation checks. The caller must reject invalid source locations in the native
scene before using this helper; no world-coordinate cutoff is imposed here.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import math

import numpy as np


_ASSOCIATION_MARGIN = .02
_POSITIONS = (.5, 0., 1.)


def _aabb(value):
    low, high = (np.asarray(item, dtype=float).copy() for item in value)
    if (low.shape != (3,) or high.shape != (3,)
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or np.any(high <= low) or not np.isfinite(high - low).all()):
        raise ValueError('AABBs require finite three-vectors and positive finite extents.')
    return low, high


def _centres(bounds):
    return {name: low + (high - low) / 2 for name, (low, high) in bounds.items()}


def _associated(bounds, groups):
    centres = _centres(bounds)
    for index, (target, reference) in enumerate(groups):
        other_reference = groups[1 - index][1]
        own = math.dist(centres[target], centres[reference])
        other = math.dist(centres[target], centres[other_reference])
        if not own + _ASSOCIATION_MARGIN < other:
            return False
    return True


def _candidate(original, groups, pair_lows, placed_lows, inside_low, inside_high, axis, gap):
    translations, moved = {}, {}
    for pair, old_low, new_low in zip(groups, pair_lows, placed_lows):
        delta = new_low - old_low
        if not np.isfinite(delta).all():
            return None
        old_high = np.maximum(original[pair[0]][1], original[pair[1]][1])
        # Addition after a large source translation can put an edge one ULP
        # outside its requested location. Adjust only the common pair delta.
        for coordinate in (0, 1):
            for _ in range(4):
                below = old_low[coordinate] + delta[coordinate] < inside_low[coordinate]
                above = old_high[coordinate] + delta[coordinate] > inside_high[coordinate]
                if not below and not above:
                    break
                if below and above:
                    return None
                delta[coordinate] = np.nextafter(delta[coordinate], math.inf if below else -math.inf)
            else:
                return None
        for name in pair:
            low, high = original[name]
            new_low, new_high = low + delta, high + delta
            if (not np.isfinite(new_low).all() or not np.isfinite(new_high).all()
                    or np.any(new_low[:2] < inside_low)
                    or np.any(new_high[:2] > inside_high)
                    or not np.allclose(new_high - new_low, high - low, rtol=1e-12, atol=1e-9)):
                return None
            translations[name] = delta.copy()
            moved[name] = (new_low, new_high)
        old_vector = original[pair[0]][0] - original[pair[1]][0]
        new_vector = moved[pair[0]][0] - moved[pair[1]][0]
        if not np.allclose(new_vector, old_vector, rtol=1e-12, atol=1e-9):
            return None
    first, second = sorted(range(2), key=lambda index: placed_lows[index][axis])
    first_high = max(moved[name][1][axis] for name in groups[first])
    second_low = min(moved[name][0][axis] for name in groups[second])
    if second_low - first_high < gap or not _associated(moved, groups):
        return None
    return {'translations': translations, 'bounds': moved,
            'repacked': True, 'pair_geometry_preserved': True}


def comparison_support_layouts(bounds, groups, support_bounds, surface_z, *, margin=.03, gap=.10):
    """Return bounded deterministic placements of two unchanged task/reference pairs.

    ``bounds`` contains exactly the four object names in the ordered ``groups``;
    each group is ``[task_name, reference_name]``. Each AABB is ``(low, high)``
    with finite, positive three-dimensional extents. All objects in a pair receive
    one common XYZ translation, with the pair minimum Z at ``surface_z + .005``.
    Object dimensions, orientation and every within-pair relative vector remain
    unchanged. Support containment is in XY, inset by ``margin``.

    Search both separating axes and pair orders. Each pair gets near/middle/far
    positions on the other axis, and the combined layout gets center/edge choices
    on the separating axis. Gap choices are the requested gap, a larger gap when
    needed for unambiguous own-reference association, and the largest fitting
    gap. At most 324 candidates are constructed before duplicate removal.

    Invalid inputs or no fitting, unambiguous placement return ``[]``. The API
    accepts geometry and pair membership only; answers and evaluation targets
    play no part in placement. Results are proposals, not physical certificates.
    """
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            if (not isinstance(bounds, Mapping) or not isinstance(groups, (list, tuple))
                    or len(groups) != 2
                    or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                           for pair in groups)):
                return []
            groups = tuple(tuple(pair) for pair in groups)
            names = tuple(name for pair in groups for name in pair)
            if (any(not isinstance(name, str) for name in names)
                    or len(set(names)) != 4 or set(bounds) != set(names)):
                return []
            original = {name: _aabb(bounds[name]) for name in names}
            support_low, support_high = _aabb(support_bounds)
            margin, gap, surface_z = float(margin), float(gap), float(surface_z)
            if (not np.isfinite([margin, gap, surface_z]).all() or margin < 0 or gap < 0):
                return []
            inside_low, inside_high = support_low[:2] + margin, support_high[:2] - margin
            span = inside_high - inside_low
            pair_lows = [np.minimum(original[pair[0]][0], original[pair[1]][0])
                         for pair in groups]
            pair_highs = [np.maximum(original[pair[0]][1], original[pair[1]][1])
                          for pair in groups]
            sizes = np.asarray(pair_highs) - pair_lows
            floor_z = surface_z + .005
            if (not np.isfinite(span).all() or not np.isfinite(sizes).all()
                    or not math.isfinite(floor_z) or np.any(span <= 0)
                    or np.any(sizes[:, :2] > span)):
                return []
            local_centres = [{name: original[name][0] - low
                             + (original[name][1] - original[name][0]) / 2 for name in pair}
                             for pair, low in zip(groups, pair_lows)]
            own_distances = [math.dist(centres[pair[0]], centres[pair[1]])
                             for pair, centres in zip(groups, local_centres)]
            if not np.isfinite(own_distances).all():
                return []
            return _layouts(original, groups, pair_lows, sizes, local_centres,
                            own_distances, inside_low, inside_high, floor_z, gap)
    except (TypeError, ValueError, OverflowError):
        return []


def _layouts(original, groups, pair_lows, sizes, centres, own_distances,
             inside_low, inside_high, floor_z, gap):
    results, seen = [], set()
    span = inside_high - inside_low
    names = tuple(name for pair in groups for name in pair)
    for axis in (0, 1):
        lateral = 1 - axis
        max_gap = span[axis] - sizes[0, axis] - sizes[1, axis]
        if max_gap < gap:
            continue
        for first, second in ((0, 1), (1, 0)):
            for fractions in product(_POSITIONS, repeat=2):
                placed_lows = [np.array([0., 0., floor_z]) for _ in groups]
                for index, fraction in enumerate(fractions):
                    placed_lows[index][lateral] = (inside_low[lateral]
                        + fraction * (span[lateral] - sizes[index, lateral]))
                required_gap = gap
                base_associated = True
                for index, other in ((first, second), (second, first)):
                    target, reference = groups[index][0], groups[other][1]
                    target_local, reference_local = centres[index][target], centres[other][reference]
                    axis_offset = (sizes[first, axis] + reference_local[axis] - target_local[axis]
                                   if index == first else
                                   sizes[first, axis] + target_local[axis] - reference_local[axis])
                    lateral_offset = (placed_lows[index][lateral] + target_local[lateral]
                                      - placed_lows[other][lateral] - reference_local[lateral])
                    z_offset = target_local[2] - reference_local[2]
                    required_distance = own_distances[index] + _ASSOCIATION_MARGIN
                    if not math.hypot(gap + axis_offset, lateral_offset, z_offset) > required_distance:
                        base_associated = False
                    remainder = required_distance ** 2 - lateral_offset ** 2 - z_offset ** 2
                    if remainder > 0:
                        required_gap = max(required_gap, math.sqrt(remainder) - axis_offset)
                gaps = [gap]
                if not base_associated:
                    # Stay strictly beyond the association boundary, including roundoff.
                    gaps.append(required_gap + max(1e-9, abs(float(np.spacing(required_gap))) * 8))
                gaps.append(max_gap)
                for separation in dict.fromkeys(gaps):
                    if not math.isfinite(separation) or separation > max_gap:
                        continue
                    slack = max_gap - separation
                    for fraction in _POSITIONS:
                        placed_lows[first][axis] = inside_low[axis] + fraction * slack
                        placed_lows[second][axis] = (placed_lows[first][axis]
                                                    + sizes[first, axis] + separation)
                        candidate = _candidate(original, groups, pair_lows, placed_lows,
                                               inside_low, inside_high, axis, gap)
                        if candidate is None:
                            continue
                        key = tuple(float(coordinate) for name in names
                                    for coordinate in candidate['translations'][name])
                        if key not in seen:
                            seen.add(key)
                            results.append(candidate)
    return results

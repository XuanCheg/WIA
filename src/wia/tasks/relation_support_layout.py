"""Propose support layouts by scaling root spacing without resizing assets.

Run: python -m pytest -q tests/test_relation_support_layout.py
These pure geometry proposals do not certify native surface contact, collision
clearance, robot reach, IK or grasp. Native source-location validation belongs to
the caller; this helper imposes no absolute world-coordinate cutoff.
"""
from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations, product
import math

import numpy as np


_POSITIONS = (.5, 0., 1.)
_ROUNDING_STEPS = 8


def _vector(value):
    result = np.asarray(value, dtype=float).copy()
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Native geometry requires finite three-vectors.')
    return result


def _aabb(value):
    low, high = (_vector(item) for item in value)
    if np.any(high <= low) or not np.isfinite(high - low).all():
        raise ValueError('AABBs require positive finite extents.')
    return low, high


def _restrict(interval, coefficient, bound):
    """Intersect a closed interval with coefficient * scale <= bound."""
    if not math.isfinite(coefficient) or not math.isfinite(bound):
        raise ValueError('Derived geometry inequalities must remain finite.')
    low, high = interval
    if coefficient == 0:
        return interval if bound >= 0 else None
    boundary = float(bound) / float(coefficient)
    if coefficient > 0:
        high = min(high, boundary)
    else:
        low = max(low, boundary)
    return (low, high) if low <= high else None


def _merge(intervals):
    result = []
    for low, high in sorted(intervals):
        if result and low <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], high))
        else:
            result.append((low, high))
    return result


def _scale_intervals(roots, low_offsets, high_offsets, span, gap):
    interval = (0., 1.)
    count = len(roots)
    # Every upper/lower edge combination must fit, including each asset itself.
    for first, second in product(range(count), repeat=2):
        for axis in (0, 1):
            coefficient = roots[first, axis] - roots[second, axis]
            bound = span[axis] - high_offsets[first, axis] + low_offsets[second, axis]
            interval = _restrict(interval, coefficient, bound)
            if interval is None or interval[1] <= 0:
                return []
    intervals = [interval]
    for first, second in combinations(range(count), 2):
        # A pair is separated if either direction on either axis has the gap.
        # Offset origins can make separation improve when scale decreases, so
        # retain the complete union of permitted scale intervals for each pair.
        allowed = []
        for current in intervals:
            for before, after in ((first, second), (second, first)):
                for axis in (0, 1):
                    coefficient = roots[before, axis] - roots[after, axis]
                    bound = low_offsets[after, axis] - high_offsets[before, axis] - gap
                    part = _restrict(current, coefficient, bound)
                    if part is not None and part[1] > 0:
                        allowed.append(part)
        intervals = _merge(allowed)
        if not intervals:
            return []
    return intervals


def _separated(lows, highs, gap):
    return all(np.any(lows[first, :2] - highs[second, :2] >= gap)
               or np.any(lows[second, :2] - highs[first, :2] >= gap)
               for first, second in combinations(range(len(lows)), 2))


def _placements(names, roots, lows, highs, centred, mean_xy, low_offsets,
                high_offsets, inside_low, inside_high, surface_z, gap, scale):
    local_lows = scale * centred + low_offsets
    local_highs = scale * centred + high_offsets
    group_low, group_high = local_lows.min(axis=0), local_highs.max(axis=0)
    slack = inside_high - inside_low - (group_high - group_low)
    if not np.isfinite(slack).all() or np.any(slack < 0):
        return []
    deformation = np.zeros_like(roots)
    deformation[:, :2] = (scale - 1.) * centred
    common_z = surface_z + .005 - lows[:, 2].min()
    if not math.isfinite(common_z):
        return []
    result, seen = [], set()
    for fractions in product(_POSITIONS, repeat=2):
        common = np.r_[inside_low - group_low - mean_xy + slack * fractions, common_z]
        if not np.isfinite(common).all():
            continue
        for axis in (0, 1):
            for _ in range(_ROUNDING_STEPS):
                translations = deformation + common
                below = np.any(lows[:, axis] + translations[:, axis] < inside_low[axis])
                above = np.any(highs[:, axis] + translations[:, axis] > inside_high[axis])
                if not below and not above:
                    break
                if below and above:
                    break
                common[axis] = np.nextafter(common[axis], math.inf if below else -math.inf)
        translations = deformation + common
        moved_lows, moved_highs = lows + translations, highs + translations
        moved_roots = roots + translations
        if (not np.isfinite(moved_lows).all() or not np.isfinite(moved_highs).all()
                or not np.isfinite(moved_roots).all()
                or np.any(moved_lows[:, :2] < inside_low)
                or np.any(moved_highs[:, :2] > inside_high)
                or not np.allclose(moved_highs - moved_lows, highs - lows, rtol=1e-12, atol=1e-9)
                or not _separated(moved_lows, moved_highs, gap)):
            continue
        if any(not np.allclose(moved_roots[first, :2] - moved_roots[second, :2],
                               scale * (roots[first, :2] - roots[second, :2]),
                               rtol=1e-12, atol=1e-9)
               or not math.isclose(moved_roots[first, 2] - moved_roots[second, 2],
                                   roots[first, 2] - roots[second, 2],
                                   rel_tol=1e-12, abs_tol=1e-9)
               for first, second in combinations(range(len(names)), 2)):
            continue
        if not math.isclose(float(moved_lows[:, 2].min()), surface_z + .005,
                            rel_tol=1e-12, abs_tol=1e-9):
            continue
        key = tuple(float(value) for value in translations.flat)
        if key in seen:
            continue
        seen.add(key)
        result.append({'translations': {name: translations[index].copy()
                                        for index, name in enumerate(names)},
                       'bounds': {name: (moved_lows[index].copy(), moved_highs[index].copy())
                                  for index, name in enumerate(names)},
                       'layout_scale_xy': float(scale)})
    return result


def relation_support_layouts(bounds, positions, support_bounds, surface_z, *, margin=.03, gap=.03):
    """Return at most nine deterministic root-similarity initialization proposals.

    ``bounds`` and ``positions`` contain exactly the same two or more native
    object names. Positions are measured root XYZ vectors; bounds are positive
    finite AABBs. Root XY is uniformly scaled by the greatest feasible positive
    factor <= 1 about the original root mean, then translated together. All
    objects receive one common Z translation to place the group's minimum AABB
    Z at ``surface_z + .005``. Asset dimensions and orientations do not change.

    Fit and pair-separation constraints are solved as linear scale intervals,
    including offset root origins. Every pair must have at least ``gap`` of
    horizontal AABB separation along an axis. Output locations enumerate common
    center/near/far placement along each horizontal axis within the support inset
    by ``margin``. Small bounded nextafter corrections handle floating edges.

    Invalid or impossible geometry returns ``[]``. No answers, evaluations or
    orientation inputs are accepted; proposals require separate native checks.
    """
    try:
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            if (not isinstance(bounds, Mapping) or not isinstance(positions, Mapping)
                    or len(bounds) < 2 or set(bounds) != set(positions)
                    or any(not isinstance(name, str) for name in bounds)):
                return []
            names = tuple(sorted(bounds))
            boxes = [_aabb(bounds[name]) for name in names]
            lows = np.asarray([box[0] for box in boxes])
            highs = np.asarray([box[1] for box in boxes])
            roots = np.asarray([_vector(positions[name]) for name in names])
            support_low, support_high = _aabb(support_bounds)
            margin, gap, surface_z = float(margin), float(gap), float(surface_z)
            if not np.isfinite([margin, gap, surface_z]).all() or margin < 0 or gap < 0:
                return []
            inside_low, inside_high = support_low[:2] + margin, support_high[:2] - margin
            span = inside_high - inside_low
            relative = roots[:, :2] - roots[0, :2]
            mean_relative = relative.mean(axis=0)
            centred = relative - mean_relative
            mean_xy = roots[0, :2] + mean_relative
            low_offsets, high_offsets = lows[:, :2] - roots[:, :2], highs[:, :2] - roots[:, :2]
            if (np.any(span <= 0) or not all(np.isfinite(value).all() for value in (
                    span, relative, centred, mean_xy, low_offsets, high_offsets))):
                return []
            intervals = _scale_intervals(centred, low_offsets, high_offsets, span, gap)
            if not intervals:
                return []
            lower, scale = intervals[-1]
            # The closed mathematical endpoint can round just outside a strict
            # containment/gap inequality. Only adjacent representable scales
            # inside this same feasible interval may correct that numerical case.
            for _ in range(_ROUNDING_STEPS + 1):
                layouts = _placements(names, roots, lows, highs, centred, mean_xy,
                                      low_offsets, high_offsets, inside_low, inside_high,
                                      surface_z, gap, scale)
                if layouts:
                    return layouts
                scale = float(np.nextafter(scale, 0.))
                if scale <= 0 or scale < lower:
                    break
            return []
    except (TypeError, ValueError, OverflowError):
        return []

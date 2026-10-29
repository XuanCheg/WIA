"""Check root-similarity support proposals using CPU geometry only.

Run: python -m pytest -q tests/test_relation_support_layout.py
No simulator, native contact, robot action, IK or grasp is exercised here.
"""
import copy
import inspect
from itertools import combinations

import numpy as np
import pytest

from wia.tasks.relation_support_layout import relation_support_layouts


SUPPORT = ([-.9, -.8, 0.], [.9, .8, .72])


def boxes_at(positions, extents=(.2, .2, .2)):
    half = np.asarray(extents) / 2
    return {name: (np.asarray(root) - half, np.asarray(root) + half)
            for name, root in positions.items()}


def triangle():
    positions = {'west': np.array([-3., -2., .1]),
                 'east': np.array([3., -2., .15]),
                 'north': np.array([0., 3., .2])}
    return boxes_at(positions), positions


def assert_layouts(bounds, positions, layouts, support=SUPPORT, *, gap=.03, surface=.72):
    assert 1 <= len(layouts) <= 9
    for layout in layouts:
        assert set(layout) == {'translations', 'bounds', 'layout_scale_xy'}
        assert set(layout['translations']) == set(layout['bounds']) == set(bounds)
        scale = layout['layout_scale_xy']
        assert isinstance(scale, float) and 0 < scale <= 1
        roots = {name: np.asarray(positions[name]) + layout['translations'][name] for name in positions}
        for name, (low, high) in bounds.items():
            low, high = np.asarray(low), np.asarray(high)
            delta = layout['translations'][name]
            moved_low, moved_high = layout['bounds'][name]
            assert isinstance(delta, np.ndarray) and delta.shape == (3,)
            np.testing.assert_array_equal(moved_low, low + delta)
            np.testing.assert_array_equal(moved_high, high + delta)
            np.testing.assert_allclose(moved_high - moved_low, high - low, atol=1e-12)
            assert np.all(moved_low[:2] >= np.asarray(support[0])[:2] + .03)
            assert np.all(moved_high[:2] <= np.asarray(support[1])[:2] - .03)
        assert min(value[0][2] for value in layout['bounds'].values()) == pytest.approx(surface + .005)
        assert len({row[2] for row in layout['translations'].values()}) == 1
        for first, second in combinations(positions, 2):
            np.testing.assert_allclose(roots[first][:2] - roots[second][:2],
                                       scale * (np.asarray(positions[first])[:2]
                                                - np.asarray(positions[second])[:2]), atol=1e-12)
            assert roots[first][2] - roots[second][2] == pytest.approx(
                positions[first][2] - positions[second][2])
            low_a, high_a = layout['bounds'][first]
            low_b, high_b = layout['bounds'][second]
            assert np.any(low_a[:2] - high_b[:2] >= gap) or np.any(low_b[:2] - high_a[:2] >= gap)
    return layouts


@pytest.mark.parametrize('swap_axes', [False, True])
def test_oversized_triangle_reduces_root_spacing_and_preserves_unchanged_assets(swap_axes):
    bounds, positions = triangle()
    if swap_axes:
        positions = {name: root[[1, 0, 2]] for name, root in positions.items()}
        bounds = boxes_at(positions)
    before_bounds, before_positions = copy.deepcopy(bounds), copy.deepcopy(positions)
    assert max(box[1][0] for box in bounds.values()) - min(box[0][0] for box in bounds.values()) > 1.8
    layouts = assert_layouts(bounds, positions, relation_support_layouts(bounds, positions, SUPPORT, .72))
    root_span = np.ptp(np.array(list(positions.values()))[:, :2], axis=0)
    expected = min((np.array([1.8, 1.6]) - .06 - .2) / root_span)
    assert all(row['layout_scale_xy'] == pytest.approx(expected, abs=1e-14) for row in layouts)
    original = np.array([positions[name][:2] for name in ('west', 'east', 'north')])
    original_area = np.linalg.det(np.stack([original[1] - original[0], original[2] - original[0]]))
    for row in layouts:
        moved = np.array([(positions[name] + row['translations'][name])[:2]
                          for name in ('west', 'east', 'north')])
        area = np.linalg.det(np.stack([moved[1] - moved[0], moved[2] - moved[0]]))
        assert area == pytest.approx(original_area * row['layout_scale_xy'] ** 2)
    for name in positions:
        np.testing.assert_array_equal(bounds[name], before_bounds[name])
        np.testing.assert_array_equal(positions[name], before_positions[name])


def test_collinear_root_order_and_similarity_are_retained():
    positions = {'one': np.array([2., 1., .1]), 'two': np.array([4., 2., .1]),
                 'three': np.array([8., 4., .1])}
    bounds = boxes_at(positions, (.12, .16, .2))
    layouts = assert_layouts(bounds, positions, relation_support_layouts(bounds, positions, SUPPORT, .72))
    for row in layouts:
        roots = [positions[name] + row['translations'][name] for name in ('one', 'two', 'three')]
        assert roots[0][0] < roots[1][0] < roots[2][0]
        first, second = roots[1] - roots[0], roots[2] - roots[0]
        assert np.linalg.det(np.stack([first[:2], second[:2]])) == pytest.approx(0., abs=1e-12)


def test_original_spacing_is_kept_when_it_already_fits_and_has_clearance():
    positions = {'a': np.array([0., 0., .1]), 'b': np.array([.4, .25, .1])}
    bounds = boxes_at(positions)
    layouts = assert_layouts(bounds, positions, relation_support_layouts(bounds, positions, SUPPORT, .72))
    assert all(row['layout_scale_xy'] == 1. for row in layouts)
    assert len(layouts) == 9
    for axis in (0, 1):
        group_edges = [(min(value[0][axis] for value in row['bounds'].values()),
                        max(value[1][axis] for value in row['bounds'].values())) for row in layouts]
        assert any(np.isclose(low, SUPPORT[0][axis] + .03, atol=1e-12) for low, _ in group_edges)
        assert any(np.isclose(high, SUPPORT[1][axis] - .03, atol=1e-12) for _, high in group_edges)
        assert any(np.isclose((low + high) / 2, 0., atol=1e-12) for low, high in group_edges)


def test_impossible_crowded_group_is_rejected_without_resizing_objects():
    positions = {'a': np.array([-2., 0., .1]), 'b': np.array([2., 0., .1])}
    bounds = boxes_at(positions, (.3, .3, .2))
    support = ([-.33, -.2, 0.], [.33, .2, .72])
    assert relation_support_layouts(bounds, positions, support, .72) == []


def test_offset_root_origins_require_another_scale_interval():
    positions = {'a': np.array([0., 0., .1]), 'b': np.array([4., 0., .1]),
                 'c': np.array([8., 2., .1])}
    bounds = {'a': (np.array([1.4, -.1, 0.]), np.array([1.6, .1, .2])),
              'b': (np.array([2.4, -.1, 0.]), np.array([2.6, .1, .2])),
              'c': (np.array([7.9, 1.9, 0.]), np.array([8.1, 2.1, .2]))}
    support = ([-2.38, -1.13, 0.], [2.38, 1.13, .72])
    # Support alone allows .75, where a and b coincide; .6925 is the greatest
    # separated scale because the AABBs are offset from their measured roots.
    expected = (3. - .2 - .03) / 4
    layouts = assert_layouts(bounds, positions,
                             relation_support_layouts(bounds, positions, support, .72), support)
    assert all(row['layout_scale_xy'] == pytest.approx(expected, abs=1e-14) for row in layouts)
    for row in layouts:
        assert row['bounds']['b'][1][0] < row['bounds']['a'][0][0]


def test_zero_root_difference_cannot_make_an_oversized_asset_fit():
    positions = {'a': np.array([0., 0., .1]), 'b': np.array([0., 0., .1])}
    bounds = {'a': ([-1., -.1, 0.], [1., .1, .2]),
              'b': ([-.1, .3, 0.], [.1, .5, .2])}
    assert relation_support_layouts(bounds, positions, SUPPORT, .72) == []


def test_results_are_deterministic_and_independent_of_mapping_order():
    bounds, positions = triangle()
    first = relation_support_layouts(bounds, positions, SUPPORT, .72)
    second = relation_support_layouts(dict(reversed(list(bounds.items()))),
                                      dict(reversed(list(positions.items()))), SUPPORT, .72)
    assert first and len(first) == len(second)
    for left, right in zip(first, second):
        assert left['layout_scale_xy'] == right['layout_scale_xy']
        for name in positions:
            np.testing.assert_array_equal(left['translations'][name], right['translations'][name])
            np.testing.assert_array_equal(left['bounds'][name], right['bounds'][name])


@pytest.mark.parametrize('invalid', ['nonfinite_root', 'nonfinite_bound', 'zero_extent',
                                   'missing_root', 'one_object', 'root_shape', 'negative_gap',
                                   'negative_margin', 'nonfinite_surface', 'invalid_support'])
def test_invalid_inputs_return_no_layout(invalid):
    bounds, positions = triangle()
    options, surface, support = {}, .72, SUPPORT
    if invalid == 'nonfinite_root':
        positions['west'][0] = np.inf
    elif invalid == 'nonfinite_bound':
        bounds['west'][0][0] = np.nan
    elif invalid == 'zero_extent':
        bounds['west'][1][0] = bounds['west'][0][0]
    elif invalid == 'missing_root':
        del positions['west']
    elif invalid == 'one_object':
        bounds, positions = {'west': bounds['west']}, {'west': positions['west']}
    elif invalid == 'root_shape':
        positions['west'] = np.array([0., 0.])
    elif invalid == 'nonfinite_surface':
        surface = np.inf
    elif invalid == 'invalid_support':
        support = ([0., 0., 0.], [np.inf, 0., .72])
    else:
        options[invalid.removeprefix('negative_')] = -.01
    assert relation_support_layouts(bounds, positions, support, surface, **options) == []


def test_api_accepts_geometry_without_source_answer_or_evaluation_input():
    assert list(inspect.signature(relation_support_layouts).parameters) == [
        'bounds', 'positions', 'support_bounds', 'surface_z', 'margin', 'gap']
    bounds, positions = triangle()
    with pytest.raises(TypeError, match='answer'):
        relation_support_layouts(bounds, positions, SUPPORT, .72, answer='triangle')

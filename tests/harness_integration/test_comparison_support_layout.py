"""Check pair-preserving support proposals without launching simulation.

Run: python -m pytest -q tests/test_comparison_support_layout.py
These geometry checks do not certify native contact, collision, IK or grasp.
"""
import copy
import inspect

import numpy as np
import pytest

from wia.tasks.comparison_support_layout import comparison_support_layouts


GROUPS = [['task_a', 'reference_a'], ['task_b', 'reference_b']]
SUPPORT = ([-.6, -.4, 0.], [.6, .4, .72])


def separated_pairs():
    return {'task_a': ([-4., -.08, .02], [-3.8, .12, .22]),
            'reference_a': ([-3.68, -.04, 0.], [-3.56, .08, .30]),
            'task_b': ([4., .05, .03], [4.26, .23, .23]),
            'reference_b': ([4.35, .08, .01], [4.45, .19, .31])}


def centre(aabb):
    low, high = (np.asarray(value) for value in aabb)
    return low + (high - low) / 2


def check_layouts(original, layouts, support=SUPPORT, *, surface=.72, gap=.10):
    assert layouts
    assert len(layouts) <= 324
    for layout in layouts:
        assert set(layout) == {'translations', 'bounds', 'repacked', 'pair_geometry_preserved'}
        assert layout['repacked'] is True and layout['pair_geometry_preserved'] is True
        assert set(layout['translations']) == set(layout['bounds']) == set(original)
        pair_rectangles = []
        for pair in GROUPS:
            delta = layout['translations'][pair[0]]
            assert isinstance(delta, np.ndarray) and delta.shape == (3,)
            np.testing.assert_array_equal(delta, layout['translations'][pair[1]])
            for name in pair:
                low, high = (np.asarray(value) for value in original[name])
                moved_low, moved_high = layout['bounds'][name]
                np.testing.assert_allclose(moved_low, low + delta, atol=1e-12)
                np.testing.assert_allclose(moved_high, high + delta, atol=1e-12)
                np.testing.assert_allclose(moved_high - moved_low, high - low, atol=1e-12)
                assert np.all(moved_low[:2] >= np.asarray(support[0])[:2] + .03)
                assert np.all(moved_high[:2] <= np.asarray(support[1])[:2] - .03)
            old_vector = centre(original[pair[0]]) - centre(original[pair[1]])
            new_vector = centre(layout['bounds'][pair[0]]) - centre(layout['bounds'][pair[1]])
            np.testing.assert_allclose(new_vector, old_vector, atol=1e-12)
            low = np.min([layout['bounds'][name][0] for name in pair], axis=0)
            high = np.max([layout['bounds'][name][1] for name in pair], axis=0)
            assert low[2] == pytest.approx(surface + .005)
            pair_rectangles.append((low, high))
        first, second = pair_rectangles
        assert any(first[1][axis] + gap <= second[0][axis] + 1e-12
                   or second[1][axis] + gap <= first[0][axis] + 1e-12 for axis in (0, 1))
        for index, (task, reference) in enumerate(GROUPS):
            own = np.linalg.norm(centre(layout['bounds'][task]) - centre(layout['bounds'][reference]))
            other = np.linalg.norm(centre(layout['bounds'][task])
                                   - centre(layout['bounds'][GROUPS[1 - index][1]]))
            assert own + .02 < other


def test_repack_an_overwide_four_object_group_without_changing_either_pair():
    original = separated_pairs()
    before = copy.deepcopy(original)
    original_width = max(value[1][0] for value in original.values()) - min(
        value[0][0] for value in original.values())
    assert original_width > SUPPORT[1][0] - SUPPORT[0][0]
    layouts = comparison_support_layouts(original, GROUPS, SUPPORT, .72)
    check_layouts(original, layouts)
    assert original == before
    # Pair translations can differ even though each pair keeps its full XYZ geometry.
    assert any(not np.array_equal(row['translations']['task_a'], row['translations']['task_b'])
               for row in layouts)


def test_enumeration_is_deterministic_and_independent_of_mapping_insertion_order():
    original = separated_pairs()
    first = comparison_support_layouts(original, GROUPS, SUPPORT, .72)
    second = comparison_support_layouts(dict(reversed(list(original.items()))), GROUPS, SUPPORT, .72)
    assert len(first) == len(second)
    for row_a, row_b in zip(first, second):
        for name in original:
            np.testing.assert_array_equal(row_a['translations'][name], row_b['translations'][name])
            np.testing.assert_array_equal(row_a['bounds'][name], row_b['bounds'][name])


def test_both_axes_pair_orders_and_center_edge_choices_are_present():
    layouts = comparison_support_layouts(separated_pairs(), GROUPS, SUPPORT, .72)
    rectangles = [[(np.min([row['bounds'][name][0] for name in pair], axis=0),
                    np.max([row['bounds'][name][1] for name in pair], axis=0))
                   for pair in GROUPS] for row in layouts]
    axes_and_orders = set()
    for first, second in rectangles:
        for axis in (0, 1):
            if first[1][axis] + .10 <= second[0][axis] + 1e-12:
                axes_and_orders.add((axis, 0))
            if second[1][axis] + .10 <= first[0][axis] + 1e-12:
                axes_and_orders.add((axis, 1))
    assert axes_and_orders == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for axis in (0, 1):
        assert any(np.isclose(low[axis], SUPPORT[0][axis] + .03, atol=1e-12)
                   for pairs in rectangles for low, _ in pairs)
        assert any(np.isclose(high[axis], SUPPORT[1][axis] - .03, atol=1e-12)
                   for pairs in rectangles for _, high in pairs)
        assert any(np.isclose((low[axis] + high[axis]) / 2, 0., atol=1e-12)
                   for pairs in rectangles for low, high in pairs)


def test_too_small_support_returns_no_proposal():
    # Each pair fits alone, but neither separating axis fits both pairs plus the gap.
    support = ([-.35, -.15, 0.], [.35, .15, .72])
    assert comparison_support_layouts(separated_pairs(), GROUPS, support, .72) == []


def test_asymmetric_projected_aabbs_need_no_orientation_input_or_rotation():
    def projected_box(xyz, size, yaw):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        half = np.abs(rotation) @ (np.asarray(size) / 2)
        return np.asarray(xyz) - half, np.asarray(xyz) + half

    original = {'task_a': projected_box([3., 2., .17], [.30, .09, .20], .7),
                'reference_a': projected_box([3.04, 2.34, .20], [.08, .12, .12], -.2),
                'task_b': projected_box([-3., -2., .12], [.08, .25, .16], -.4),
                'reference_b': projected_box([-2.71, -1.97, .18], [.12, .10, .24], 1.1)}
    support = ([-.7, -.6, 0.], [.7, .6, .72])
    layouts = comparison_support_layouts(original, GROUPS, support, .72)
    check_layouts(original, layouts, support)


def long_pairs():
    return {'reference_a': ([-5., 0., 0.], [-4.8, .2, .1]),
            'task_a': ([-4.2, 0., 0.], [-4., .2, .1]),
            'reference_b': ([5., 0., 0.], [5.2, .2, .1]),
            'task_b': ([5.8, 0., 0.], [6., .2, .1])}


def test_gap_increases_when_needed_to_keep_the_own_reference_nearest():
    original = long_pairs()
    support = ([-1.65, -.15, 0.], [1.65, .15, .72])
    layouts = comparison_support_layouts(original, GROUPS, support, .72)
    check_layouts(original, layouts, support)
    separations = []
    for layout in layouts:
        rectangles = [(min(layout['bounds'][name][0][0] for name in pair),
                       max(layout['bounds'][name][1][0] for name in pair)) for pair in GROUPS]
        low, high = sorted(rectangles)
        separations.append(high[0] - low[1])
    assert min(separations) > .61  # The requested .10 m alone makes association ambiguous.
    assert min(separations) < .63  # The solver also includes a small sufficient increase.
    assert max(separations) > 1.2


def test_geometric_fit_with_no_unambiguous_association_returns_empty():
    support = ([-1.25, -.15, 0.], [1.25, .15, .72])
    assert comparison_support_layouts(long_pairs(), GROUPS, support, .72) == []


@pytest.mark.parametrize('change', ['nan', 'zero_extent', 'duplicate_name', 'missing_name',
                                  'invalid_support', 'negative_gap'])
def test_invalid_geometry_or_groups_return_empty(change):
    original, groups, support, options = separated_pairs(), copy.deepcopy(GROUPS), SUPPORT, {}
    if change == 'nan':
        original['task_a'][0][0] = np.nan
    elif change == 'zero_extent':
        original['task_a'][1][0] = original['task_a'][0][0]
    elif change == 'duplicate_name':
        groups[1][1] = groups[0][1]
    elif change == 'missing_name':
        del original['reference_a']
    elif change == 'invalid_support':
        support = ([0., 0., 0.], [0., 1., .72])
    else:
        options['gap'] = -.1
    assert comparison_support_layouts(original, groups, support, .72, **options) == []


def test_api_has_no_answer_or_evaluation_input():
    assert list(inspect.signature(comparison_support_layouts).parameters) == [
        'bounds', 'groups', 'support_bounds', 'surface_z', 'margin', 'gap']
    with pytest.raises(TypeError, match='answer'):
        comparison_support_layouts(separated_pairs(), GROUPS, SUPPORT, .72, answer='task_b')

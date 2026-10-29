"""Check explicit human-only detail groups with render-only native doubles.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_explicit_review_details.py
No actor, mounted camera, native physics step or GPU workload is modified.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest

import wia.runtime.sim.esi_review_cameras as review
from harness_integration.test_review_camera_native_adapter import native_review


def patch(config, *, phase='after_postprocessing', operation='category_initialization'):
    return NS(operation=operation, phase=phase, parameters={'layout': {'human_review': config}})


def configure(fixture, groups=None, elevation=0.):
    definition = fixture.selected.definition
    definition.bindings['food_a'] = definition.bindings.pop('apple')
    definition.bindings['food_b'] = definition.bindings.pop('box')
    definition.source.json_path = 'dataset/Temporal Understanding/Agent Observation/house/living/task.json'
    config = {'detail_binding_groups': groups if groups is not None else [['food_a'], ['food_b']],
              'detail_elevation_deg': elevation}
    fixture.selected.variant = 'adapted'
    definition.scene = NS(variants={'adapted': NS(patches=[patch(config)]),
        'original': NS(patches=[patch({'invalid_unselected_contract': True})])})
    return config


def test_exact_bound_native_groups_and_elevation_override_only_the_first_details(native_review):
    fixture = native_review
    config = configure(fixture)
    names = ['object_overview', 'global_front', 'object_top', 'object_side',
             'object_detail_00', 'object_detail_01', 'object_detail_02']
    definition = fixture.selected.definition
    declaration = definition.scene.variants['adapted'].patches.pop()
    review.capture_review_views(fixture.session, fixture.selected, names)
    defaults = deepcopy(fixture.session.human_review_camera_evidence['views'])
    sensors_before = deepcopy(vars(fixture.session.robot)['sensors'])
    poses_before = {name: obj.get_position_orientation() for name, obj in fixture.actors.items()}
    definition.scene.variants['adapted'].patches.append(declaration)
    config_before = deepcopy(config)
    frames = review.capture_review_views(fixture.session, fixture.selected, names)
    evidence = fixture.session.human_review_camera_evidence
    assert set(frames) == set(names)
    for name in names:
        row = evidence['views'][name]
        if name in ('object_detail_00', 'object_detail_01'):
            key, native_name, count = ('food_a', 'apple', 128) if name.endswith('00') else ('food_b', 'box', 64)
            assert row['explicit_detail_binding_group'] == [key]
            assert row['explicit_detail_elevation_deg'] == 0.
            assert row['framing_scope'] == 'explicit_task_binding_detail'
            assert row['intended_objects'] == [native_name]
            assert row['object_pixels'] == {native_name: count}
            assert row['position_world_m'][2] == row['target_world_m'][2]
            assert row['render_boundaries'] == 9 and row['visibility_certifies_task_success'] is False
        else:
            assert row == defaults[name]
    assert fixture.calls.created == 1 and fixture.calls.ray > 0
    assert evidence['physics_steps'] == evidence['actor_mutations'] == 0
    assert evidence['robot_sensors_modified'] is False
    assert set(fixture.calls.raw_reads) == {('rgb', 'cpu'), ('seg_instance_id', 'cpu')}
    for name, obj in fixture.actors.items():
        for before, after in zip(poses_before[name], obj.get_position_orientation()):
            np.testing.assert_array_equal(before, after)
    for name, sensor in fixture.session.robot.sensors.items():
        assert vars(sensor) == vars(sensors_before[name])
    assert config == config_before


def test_group_member_order_uses_binding_resolution_instead_of_default_partition(native_review):
    configure(native_review, groups=[['food_b', 'food_a']], elevation=15.)
    review.capture_review_views(native_review.session, native_review.selected, ['object_detail_00'])
    row = native_review.session.human_review_camera_evidence['views']['object_detail_00']
    assert row['intended_objects'] == ['box', 'apple']
    assert row['explicit_detail_binding_group'] == ['food_b', 'food_a']
    assert row['explicit_detail_elevation_deg'] == 15.
    assert row['object_pixels'] == {'box': 64, 'apple': 128}


def test_only_selected_after_postprocessing_initialization_declaration_is_read(native_review):
    config = configure(native_review)
    patches = native_review.selected.definition.scene.variants['adapted'].patches
    patches.extend([patch({'ignored': True}, phase='before_construction'),
                    patch({'ignored': True}, operation='category_interaction')])
    assert review._explicit_review_details(native_review.selected) == config
    patches.clear()
    assert review._explicit_review_details(native_review.selected) is None


@pytest.mark.parametrize('groups', [[], [[]], ['food_a'], [['food_a', 'food_a']], [['unbound']], None, 'food_a'])
def test_invalid_empty_unbound_or_repeated_groups_fail_closed(native_review, groups):
    config = configure(native_review)
    config['detail_binding_groups'] = groups
    with pytest.raises(ValueError, match='group'):
        review.capture_review_views(native_review.session, native_review.selected, ['object_detail_00'])
    assert native_review.calls.created == native_review.calls.render == 0


@pytest.mark.parametrize('angle', [float('nan'), float('inf'), -float('inf'), True, None, '0'])
def test_elevation_must_be_a_declared_finite_angle(native_review, angle):
    configure(native_review, elevation=angle)
    with pytest.raises(ValueError, match='finite angle'):
        review._explicit_review_details(native_review.selected)


@pytest.mark.parametrize('invalid', ['duplicate_configs', 'missing_elevation', 'extra_field', 'not_mapping'])
def test_only_one_complete_explicit_contract_is_accepted(native_review, invalid):
    config = configure(native_review)
    patches = native_review.selected.definition.scene.variants['adapted'].patches
    if invalid == 'duplicate_configs': patches.append(patch(deepcopy(config)))
    elif invalid == 'missing_elevation': config.pop('detail_elevation_deg')
    elif invalid == 'extra_field': config['unreviewed_camera_policy'] = True
    else: patches[0].parameters['layout']['human_review'] = None
    with pytest.raises(ValueError):
        review._explicit_review_details(native_review.selected)


def test_reusing_a_binding_between_separate_groups_is_explicitly_allowed(native_review):
    config = configure(native_review, groups=[['food_a'], ['food_a']])
    assert review._explicit_review_details(native_review.selected) == config


@pytest.mark.parametrize('failure', ['native_missing', 'invalid_bounds'])
def test_declared_native_actor_cannot_be_silently_substituted(native_review, failure):
    configure(native_review)
    if failure == 'native_missing':
        native_review.actors.pop('apple')
    else:
        native_review.actors['apple'].aabb = (np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match='Declared human review native object'):
        review.capture_review_views(native_review.session, native_review.selected, ['object_detail_00'])
    assert native_review.calls.created == native_review.calls.render == 0


def test_fully_fixed_replay_bypasses_new_config_resolution_and_candidate_fitting(native_review, monkeypatch):
    configure(native_review)
    names = ['global_front', 'object_detail_00', 'object_detail_01']
    review.capture_review_views(native_review.session, native_review.selected, names)
    fixed = deepcopy(native_review.session.human_review_camera_evidence['views'])
    original = deepcopy(fixed)
    ray_calls = native_review.calls.ray
    native_review.actors.pop('apple')

    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed-view replay must not resolve authored details or fit new candidates.')

    for name in ('_explicit_review_details', 'fit_review_view', 'review_directions', '_ray_score'):
        monkeypatch.setattr(review, name, forbidden)
    review.capture_review_views(native_review.session, native_review.selected, names, fixed_views=fixed)
    assert fixed == original and native_review.calls.ray == ray_calls
    assert native_review.calls.created == 1
    replayed = native_review.session.human_review_camera_evidence['views']
    for name in names:
        assert replayed[name]['position_world_m'] == fixed[name]['position_world_m']
        assert replayed[name]['actual_quaternion_xyzw'] == fixed[name]['actual_quaternion_xyzw']
    assert replayed['object_detail_00']['explicit_detail_binding_group'] == ['food_a']


def test_mixed_replay_does_not_resolve_a_missing_actor_for_its_fixed_detail(native_review):
    configure(native_review)
    names = ['object_detail_00', 'object_detail_01']
    review.capture_review_views(native_review.session, native_review.selected, names)
    fixed = {'object_detail_00': deepcopy(native_review.session.human_review_camera_evidence['views']['object_detail_00'])}
    native_review.actors.pop('apple')
    review.capture_review_views(native_review.session, native_review.selected, names, fixed_views=fixed)
    views = native_review.session.human_review_camera_evidence['views']
    assert views['object_detail_00']['position_world_m'] == fixed['object_detail_00']['position_world_m']
    assert views['object_detail_01']['intended_objects'] == ['box']
    assert views['object_detail_01']['explicit_detail_binding_group'] == ['food_b']

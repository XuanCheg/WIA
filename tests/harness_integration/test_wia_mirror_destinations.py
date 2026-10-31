"""Check v4 live bounded destination evidence and legacy-compatible pickup gates.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_mirror_destinations.py
CPU native-contract fixtures do not certify simulation or manipulation feasibility.
"""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from wia.authoring.converters import wia_mirror as adapter
from wia.tasks.wia_mirror_scoring import MirrorProgress
from harness_integration.test_wia_dimensional_adapter import native as native_geometry
from harness_integration.test_wia_household_mirror import NativeDataset, household
from harness_integration.test_wia_mirror_adapter import mirror
from harness_integration.test_wia_mirror_scoring import parameters, sample, pick, move, tick, dwell


@pytest.fixture
def bounded(household):
    fixture = household
    p = fixture.config['parameters']
    p['revision'] = adapter.DESTINATION_REVISION
    p['destination_contracts'] = {
        'table': {'kind': 'on_top', 'usable_bounds_local_m': [[-.6, -1.2, -.05], [.6, 1.2, .05]]},
        'pad_blue': {'kind': 'inside', 'usable_bounds_local_m': [[-.1, -.12, -.08], [.1, .12, .08]]},
        'pad_yellow': {'kind': 'on_top', 'usable_bounds_local_m': [[-.12, -.15, -.01], [.12, .15, .01]]}}
    # Native fixed board, pot and anchor fixtures share the actual collision path.
    for key, category in [('pad_blue', 'cooking_pot'), ('pad_yellow', 'cutting_board'), ('anchor', 'cooking_pot')]:
        previous = fixture.objects[key]
        obj = NativeDataset(previous)
        obj.fixed_base, obj.category = True, category
        fixture.objects[key] = obj
    fixture.states.Inside = object()
    for obj in fixture.objects.values():
        obj.root_link.get_position_orientation = obj.get_position_orientation
        obj.states[fixture.states.Inside] = NS(get_value=lambda other, obj=obj: obj.relations.get((other.name, 'inside'), False))
    return fixture


def measure(fixture, baseline=None):
    return adapter._sample(fixture.session, fixture.controller, fixture.config, fixture.objects, baseline)


def relation_fixture(fixture, kind='inside', label='blue'):
    p = fixture.config['parameters']
    destination = fixture.objects[p['destinations'][label]]
    contract = p['destination_contracts'][p['destinations'][label]]
    contract['kind'] = kind
    target = fixture.target
    target.position = destination.position.copy()
    target.relations.update({(destination.name, field): True for field in ('touching', 'inside', 'on_top')})
    return target, destination, contract


def test_v4_reads_fixed_dataset_fixtures_and_preserves_all_geometry(bounded):
    adapter._config(bounded.session, bounded.config)
    captured = adapter.capture_initial_geometry(bounded.session, bounded.config, bounded.objects)
    assert captured['available'] is True
    assert captured['fixtures']['anchor']['n_dof'] == 0
    baseline = measure(bounded)
    assert baseline['fixtures_valid'] is True and baseline['geometry_valid'] is True
    assert measure(bounded, baseline)['fixtures_valid'] is True
    bounded.objects['anchor'].mesh.points[0, 0] += .02
    assert measure(bounded, baseline)['fixtures_valid'] is False


@pytest.mark.parametrize('label', ['blue', 'yellow'])
@pytest.mark.parametrize('final_state', [False, True])
def test_v4_native_adapter_sequence_delivers_mixed_produce_to_pot_or_board(bounded, label, final_state):
    p = bounded.config['parameters']
    p.update(operation='mirror_select_deliver', delivery_destination=label)
    if final_state:
        p['scoring'].update(manipulation_revision='final_state_v1', minimum_lift_m=None,
                            minimum_required_motion_m=None)
    bounded.config['operation'] = p['operation']
    for key, category in [('candidate_0', 'apple'), ('candidate_1', 'carrot')]:
        bounded.objects[key].category = category
        p['candidate_assets'][key]['category'] = category
    evaluator = adapter.create_evaluator(bounded.session, bounded.controller, bounded.config, bounded.output)
    initial = evaluator.begin_episode(bounded.observation)
    assert initial['initial_qualification']['accepted'] is True
    if not final_state:
        bounded.robot._ag_obj_in_hand['left'] = bounded.target
        bounded.robot._ag_obj_constraints['left'] = object()
        bounded.readings['left'], bounded.robot.positions[:2] = 1, 0.
        bounded.target.position[2] += .06
        bounded.clock.current_time = .0625
        evaluator.sample_step()
        bounded.robot._ag_obj_in_hand['left'] = bounded.robot._ag_obj_constraints['left'] = None
    bounded.readings['left'], bounded.robot.positions[:2] = -1, .04
    destination = bounded.objects[p['destinations'][label]]
    bounded.target.position[:] = destination.position
    kind = p['destination_contracts'][p['destinations'][label]]['kind']
    for field in ('touching', kind):
        bounded.target.relations[(destination.name, field)] = True
    for time in np.arange(.0625 if final_state else .125, .751, .0625):
        bounded.clock.current_time = float(time)
        evaluator.sample_step()
    result = evaluator.record(bounded.observation)
    assert result['evaluation_validity'] == 'complete' and result['task_success'] is True
    assert result['per_target']['candidate_0']['placement_relations']['kind'] == kind
    if final_state:
        assert result['physical_scorer_schema'] == 'wia_mirror_final_state_progress_v1'
        assert not result['per_target']['candidate_0']['grasp_seen']
        assert [row['id'] for row in result['completion_items']] == [
            'selected_original_at_destination', 'stable_placement']


def test_root_fixture_pose_change_is_rejected_even_when_object_pose_is_unchanged(bounded):
    baseline = measure(bounded)
    anchor = bounded.objects['anchor']
    anchor.root_link.get_position_orientation = lambda: (anchor.position + [.03, 0., 0.], anchor.quaternion)
    assert measure(bounded, baseline)['fixtures_valid'] is False


@pytest.mark.parametrize('kind,label', [('on_top', 'yellow'), ('inside', 'blue')])
def test_bounded_relations_use_actual_rotated_scaled_root_frame(bounded, kind, label):
    target, destination, contract = relation_fixture(bounded, kind, label)
    destination.scale = np.array([1.5, 2., 1.25])
    destination.quaternion = Rotation.from_euler('xyz', [12., 23., 37.], degrees=True).as_quat()
    root_position = destination.position + [.11, -.17, .04]
    destination.root_link.get_position_orientation = lambda: (root_position, destination.quaternion)
    local = np.array([[-.025, -.025, -.03], [.025, .025, .03]])
    world = Rotation.from_quat(destination.quaternion).apply(local * destination.root_link.scale) + root_position
    result = adapter._bounded_relation(target, destination, world, contract)
    bound_field = 'vertices_inside' if kind == 'inside' else 'footprint_inside'
    assert result['touching'] and result[kind] and result[bound_field]
    assert result['root_position_m'] == pytest.approx(root_position)
    assert np.asarray(result['target_collision_vertices_local_m']) == pytest.approx(local)
    assert result['measurement_errors'] == []
    # The contract is measured against a later live root, never frozen desired coordinates.
    destination.root_link.get_position_orientation = lambda: (root_position + [1., 0., 0.], destination.quaternion)
    assert adapter._bounded_relation(target, destination, world, contract)[bound_field] is False


def test_inside_pot_rejects_rim_only_top_and_missing_native_inside(bounded):
    target, destination, contract = relation_fixture(bounded)
    world = np.array([destination.position + [-.02, -.02, .09], destination.position + [.02, .02, .12]])
    result = adapter._bounded_relation(target, destination, world, contract)
    assert result['touching'] and result['inside'] and result['vertices_inside'] is False
    del target.states[bounded.states.Inside]
    result = adapter._bounded_relation(target, destination, world, contract)
    assert result['inside'] is None and result['measurement_errors']
    del destination.root_link.get_position_orientation
    result = adapter._bounded_relation(target, destination, world, contract)
    assert result['vertices_inside'] is None


def test_on_board_rejects_any_overhanging_collision_vertex(bounded):
    target, destination, contract = relation_fixture(bounded, 'on_top', 'yellow')
    vertices = np.array([destination.position + [-.01, 0., .5], destination.position + [.121, 0., .5]])
    result = adapter._bounded_relation(target, destination, vertices, contract)
    assert result['touching'] and result['on_top'] and result['footprint_inside'] is False


def test_dataset_optical_cues_do_not_bypass_exact_cube_contract(bounded):
    bounded.objects['cue_low'] = NativeDataset(bounded.objects['cue_low'])
    result = adapter.capture_initial_geometry(bounded.session, bounded.config, bounded.objects)
    assert result['available'] is False
    assert result['fixtures']['cue_low']['available'] is False


def pure_parameters(operation='mirror_select_deliver'):
    p = parameters(operation)
    p.update(revision=adapter.DESTINATION_REVISION, table_key='table',
        fixture_keys=['table', 'pad_blue', 'pad_yellow'],
        destination_contracts={key: {'kind': kind, 'usable_bounds_local_m': [[-.2, -.2, -.2], [.2, .2, .2]]}
            for key, kind in [('table', 'on_top'), ('pad_blue', 'inside'), ('pad_yellow', 'on_top')]})
    return p


def pure_sample(p):
    s = sample(p)
    for target in s['targets'].values():
        target['destinations'] = {label: {'kind': p['destination_contracts'][key]['kind'], 'touching': False,
            **({'inside': False, 'vertices_inside': False} if p['destination_contracts'][key]['kind'] == 'inside'
               else {'on_top': False, 'footprint_inside': False})}
            for label, key in [('table', 'table'), *p['destinations'].items()]}
    return s


def release(progress, s, label, xyz=None):
    move(s, 'c0', xyz or [1.5, -.3, .06])
    s['grasp'].update(held_names=[], released=True)
    s['targets']['c0']['robot_contact'] = False
    for pad, evidence in s['targets']['c0']['destinations'].items():
        evidence.update({field: pad == label for field in evidence if field != 'kind'})
    return tick(progress, s)


@pytest.mark.parametrize('label', ['blue', 'yellow'])
def test_reducer_delivery_uses_kind_and_optional_destination(label):
    p = pure_parameters()
    p['delivery_destination'] = label
    s, progress = pure_sample(p), MirrorProgress(p)
    progress.begin(deepcopy(s))
    pick(progress, s)
    release(progress, s, label)
    result = dwell(progress, s)
    assert result['task_success'] is True
    assert result['current']['destination'] == label
    assert result['per_target']['c0']['placement_relations']['kind'] == ('inside' if label == 'blue' else 'on_top')


@pytest.mark.parametrize('failure', ['inside_missing', 'wrong_kind', 'rim_only', 'robot_contact', 'held', 'fixture_moved'])
def test_reducer_conservatively_rejects_missing_or_invalid_pot_evidence(failure):
    p = pure_parameters()
    s, progress = pure_sample(p), MirrorProgress(p)
    progress.begin(deepcopy(s))
    pick(progress, s)
    release(progress, s, 'blue')
    relation = s['targets']['c0']['destinations']['blue']
    if failure == 'inside_missing':
        relation.pop('inside')
    elif failure == 'wrong_kind':
        relation['kind'] = 'on_top'
    elif failure == 'rim_only':
        relation['vertices_inside'] = False
    elif failure == 'robot_contact':
        s['targets']['c0']['robot_contact'] = True
    elif failure == 'held':
        s['grasp'].update(held_names=['native_0'], released=False)
    else:
        s['fixtures_valid'] = False
    result = dwell(progress, s)
    assert result['task_success'] is False and result['stable_placement'] is False
    if failure in ('inside_missing', 'wrong_kind'):
        assert result['evaluation_validity'] == 'unavailable'


def test_reducer_spatial_can_use_board_and_keeps_fixed_anchor_distance_goal():
    p = pure_parameters('mirror_reconfigure')
    p['spatial_destination'] = 'yellow'
    s, progress = pure_sample(p), MirrorProgress(p)
    progress.begin(deepcopy(s))
    pick(progress, s)
    # Both originals must rest on the selected spatial destination.
    s['targets']['c1']['destinations']['yellow'].update(touching=True, on_top=True, footprint_inside=True)
    release(progress, s, 'yellow', [1.2, 0., .06])
    result = dwell(progress, s)
    assert result['task_success'] is True and result['current']['destination'] == 'yellow'
    s['anchor_centroid_m'][0] += .02
    assert dwell(progress, s)['task_success'] is False


@pytest.mark.parametrize('problem', ['kind', 'missing', 'bounds', 'delivery', 'spatial', 'fixture'])
def test_v4_rejects_incomplete_or_unsupported_contracts(problem):
    p = pure_parameters()
    if problem == 'kind':
        p['destination_contracts']['pad_blue']['kind'] = 'near'
    elif problem == 'missing':
        del p['destination_contracts']['table']
    elif problem == 'bounds':
        p['destination_contracts']['pad_blue']['usable_bounds_local_m'][0][0] = float('nan')
    elif problem == 'delivery':
        p['delivery_destination'] = 'table'
    elif problem == 'spatial':
        p['spatial_destination'] = 'pot'
    else:
        p['fixture_keys'].remove('pad_blue')
    with pytest.raises(ValueError):
        MirrorProgress(p)

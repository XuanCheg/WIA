"""Check household geometry, legacy compatibility and explicit package building.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_household_mirror.py
These are CPU contract fixtures; no native render, physics or robot action runs.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from wia.authoring.converters import wia_mirror as adapter
import wia.authoring.commands.build_wia_household_mirror as builder
from harness_integration.test_wia_dimensional_adapter import NativeCube, native as native_geometry
from harness_integration.test_wia_mirror_adapter import mirror, deliver_fixture


class NativeDataset(NativeCube):
    """Expose the native DatasetObject API using independent CPU mesh readbacks."""
    def __init__(self, previous):
        super().__init__(previous.name, [.065, .065, .12], previous.position)
        del self._primitive_type
        self.category, self.model = 'bottle', 'cpu_native_asset'
        self.n_dof, self.n_joints = 0, 0
        self.states = {key: type(value)(get_value=lambda other, field=field: self.relations.get((other.name, field), False))
            for (key, value), field in zip(previous.states.items(), ('touching', 'on_top'))}
        self.mesh.points[0] += [.004, .002, .003]


@pytest.fixture
def household(mirror, monkeypatch):
    module = ModuleType('omnigibson.objects.dataset_object')
    module.DatasetObject = NativeDataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    p = mirror.config['parameters']
    p.update(revision=adapter.HOUSEHOLD_REVISION,
        scene_contract={'scene_model': adapter.native.SCENE, 'room': adapter.native.ROOM},
        candidate_assets={key: {'category': 'bottle', 'model': 'cpu_native_asset', 'scale': [1., 1., 1.]}
                          for key in p['candidates']})
    p.pop('candidate_dimensions_m')
    for key in p['candidates']:
        mirror.objects[key] = NativeDataset(mirror.objects[key])
    mirror.target = mirror.objects['candidate_0']
    return mirror


def measured(household, baseline=None):
    return adapter._sample(household.session, household.controller, household.config, household.objects, baseline)


def test_household_capture_reads_actual_asset_mesh_and_measured_optics(household, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Initial capture must not execute physics, contact or controls.')
    monkeypatch.setattr(adapter, '_sample', forbidden)
    for obj in household.objects.values():
        obj.contact_list = obj.get_linear_velocity = obj.get_angular_velocity = forbidden
    result = adapter.capture_initial_geometry(household.session, household.config, household.objects)
    row = result['targets']['candidate_0']
    assert result['available'] is True and row['available'] is True
    actual = household.target.mesh.transform_local_points_to_world(household.target.mesh.points)
    assert np.asarray(row['vertices_m']) == pytest.approx(actual)
    assert row['dimensions_m'] == pytest.approx(np.ptp(household.target.mesh.points, axis=0))
    assert 'non-convexity is permitted' in row['geometry_scope']
    optical = result['optical_selection']
    assert optical['accepted'] is True and optical['selected_candidate'] == 'candidate_0'
    assert optical['candidates']['candidate_0']['centroid_m'] == pytest.approx(actual.mean(axis=0))
    assert result['physical_success_verified'] is result['optical_visibility_verified'] is False
    assert household.material.writes == []
    json.dumps(result, allow_nan=False)


def test_household_fixed_metadata_joints_preserve_rigid_shape_and_freeze_real_eye(household):
    household.target.articulated = True
    baseline = measured(household)
    assert baseline['geometry_valid'] is True
    initial = baseline['targets']['candidate_0']['native_geometry']
    assert initial['n_dof'] == initial['n_joints'] == 0 and initial['articulated'] is True
    capture = adapter.capture_initial_geometry(household.session, household.config, household.objects)
    assert capture['targets']['candidate_0']['available'] is True
    assert capture['targets']['candidate_0']['articulated'] is True
    household.target.position += [.04, -.02, .06]
    household.target.quaternion = Rotation.from_euler('xyz', [27., 41., 63.], degrees=True).as_quat()
    household.sensor.position += [1., .4, .3]
    sample = measured(household, baseline)
    assert sample['geometry_valid'] is True
    row = sample['targets']['candidate_0']
    assert row['dimensions_m'] == pytest.approx(baseline['targets']['candidate_0']['dimensions_m'])
    assert row['lowest_z_m'] == pytest.approx(np.min(row['vertices_m'], axis=0)[2])
    assert sample['optical_selection'] == baseline['optical_selection']
    assert sample['optical_selection']['frozen_eye_m'] != household.sensor.position.tolist()
    household.target.mesh.points[0, 0] += .04
    changed = measured(household, baseline)
    assert changed['geometry_valid'] is False


@pytest.mark.parametrize('problem', ['category', 'model', 'scale', 'mesh', 'dof', 'movable_joint', 'missing_dof', 'missing_joints',
                                     'unavailable_count', 'extra_collision_link', 'identity', 'primitive'])
def test_household_missing_or_wrong_native_geometry_has_no_cube_fallback(household, problem):
    if problem in ('category', 'model'):
        setattr(household.target, problem, 'wrong_native_identity')
    elif problem == 'scale':
        household.target.scale[0] *= 1.1
    elif problem == 'mesh':
        household.target.mesh.collision_enabled = False
    elif problem == 'dof':
        household.target.n_dof = 1
    elif problem == 'movable_joint':
        household.target.n_joints = 1
    elif problem in ('missing_dof', 'missing_joints'):
        delattr(household.target, 'n_dof' if problem == 'missing_dof' else 'n_joints')
    elif problem == 'unavailable_count':
        household.target.n_dof = None
    elif problem == 'extra_collision_link':
        household.target.links['extra'] = deepcopy(household.target.root_link)
    elif problem == 'identity':
        household.target.name = 'wrong_name'
    else:
        household.objects['candidate_0'] = NativeCube(household.target.name, [.065, .065, .12], household.target.position)
    capture = adapter.capture_initial_geometry(household.session, household.config, household.objects)
    assert capture['available'] is False
    assert capture['targets']['candidate_0']['available'] is False
    sample = measured(household)
    assert sample['geometry_valid'] is not True
    assert sample['targets']['candidate_0']['vertices_m'] is None
    result = adapter.create_evaluator(household.session, household.controller, household.config, household.output).begin_episode(household.observation)
    assert result['task_success'] is False and result['completion_score'] is None


def test_household_optical_ambiguity_is_not_promoted_to_an_accepted_capture(household):
    household.objects['cue_low'].position[1] -= 1.
    household.objects['cue_high'].position[1] += 1.
    capture = adapter.capture_initial_geometry(household.session, household.config, household.objects)
    assert capture['available'] is True
    assert capture['optical_selection']['accepted'] is False
    assert len(capture['optical_selection']['passing_candidates']) > 1
    assert capture['optical_selection']['selected_candidate'] is None


def test_household_noop_requires_real_grasp_lift_release_and_correct_report(household):
    evaluator = adapter.create_evaluator(household.session, household.controller, household.config, household.output)
    initial = evaluator.begin_episode(household.observation)
    assert initial['initial_qualification']['accepted'] is True
    assert 'candidate_dimensions_proposal_m' not in initial['initial_qualification']
    evaluator.report_answer('ANSWER: reflected')
    for index in range(1, 15):
        household.clock.current_time = .05 * index
        evaluator.sample_step()
    noop = evaluator.record(household.observation)
    assert noop['task_success'] is noop['physical_success'] is False
    household.clock.current_time = 0.
    result = deliver_fixture(household, 'ANSWER: reflected')
    assert result['task_success'] is True
    assert result['schema'] == adapter.HOUSEHOLD_REVISION
    assert result['optical_visibility_verified'] is result['complete_reference_execution_verified'] is False


def _design_entry():
    root = __import__('wia.resources', fromlist=['ROOT']).ROOT
    task_path = root / 'explore_to_achieve/wia_v1/mirror_v1/tasks/r_a1_seed101_v1.yaml'
    scene_path = root / 'explore_to_achieve/wia_v1/mirror_v1/scenes/r_a1_seed101_v1.yaml'
    task, scene = (yaml.safe_load(path.read_text()) for path in (task_path, scene_path))
    layout = deepcopy(task['scene']['variants']['adapted']['patches'][0]['parameters']['layout'])
    assets = {}
    for key, name in task['private']['evaluation']['parameters']['candidates'].items():
        asset = {'category': 'bottle', 'model': 'cpu_native_asset', 'scale': [1., 1., 1.]}
        assets[key] = asset
        index = next(index for index, obj in enumerate(layout['objects']) if obj['name'] == name)
        old = layout['objects'][index]
        layout['objects'][index] = {key: deepcopy(old[key]) for key in ('name', 'position', 'orientation', 'fixed_base', 'visual_only')}
        layout['objects'][index].update(type='DatasetObject', **asset)
    public = deepcopy(task['public'])
    public = json.loads(json.dumps(public).replace('blocks', 'household objects').replace('block', 'household object'))
    return task, scene, {'document_alias': 'R-A1', 'variant_id': 'household_cpu', 'label': 'CPU contract fixture',
        'parent_task_file': str(task_path), 'parent_scene_file': str(scene_path),
        'source_identity': task['source']['json'], 'scene_model': layout['scene_model'], 'room': layout['rooms'][0],
        'layout': layout, 'bindings': task['private']['bindings'], 'candidate_assets': assets,
        'public_goal': public['objective'], 'public': public, 'task_id': 'household_cpu_task', 'scene_id': 'household_cpu_scene'}


def test_builder_preserves_seed_budget_scoring_and_replaces_full_public_wording(tmp_path):
    parent, parent_scene, entry = _design_entry()
    original = deepcopy(parent)
    task, scene = builder.translate_definitions(parent, parent_scene, entry, {'source_sha256': 'cpu_fixture'})
    assert parent == original
    assert task['defaults'] == parent['defaults']
    p = task['private']['evaluation']['parameters']
    assert p['scoring'] == parent['private']['evaluation']['parameters']['scoring']
    assert 'candidate_dimensions_m' not in p and p['revision'] == adapter.HOUSEHOLD_REVISION
    assert p['candidate_assets'] == entry['candidate_assets']
    assert task['public'] == entry['public']
    assert task['scene'] == scene['scene'] and task['private']['bindings'] == scene['bindings']
    assert task['repair']['physical_success_verified'] is False
    design = tmp_path / 'design.json'
    design.write_text(json.dumps({'schema': 'wia_household_mirror_design_v1', 'entries': [entry]}))
    output = tmp_path / 'output'
    assert builder.main(['--design-file', str(design), '--output-dir', str(output)]) == 0
    assert builder.main(['--design-file', str(design), '--output-dir', str(output), '--check']) == 0
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['schema'] == 'wia_variant_manifest_v1'
    assert manifest['entries'][0]['seed'] == parent['defaults']['seed']
    task_file = Path(manifest['entries'][0]['task_file'])
    task_file.write_text('preserved user edit\n')
    assert builder.main(['--design-file', str(design), '--output-dir', str(output), '--check']) == 1
    with pytest.raises(ValueError, match='overwrite'):
        builder.main(['--design-file', str(design), '--output-dir', str(output)])
    assert task_file.read_text() == 'preserved user edit\n'


@pytest.mark.parametrize('problem', ['asset', 'primitive', 'block_wording', 'duplicate_name', 'old_task_id'])
def test_builder_rejects_inconsistent_or_legacy_household_contracts(problem):
    task, scene, entry = _design_entry()
    if problem == 'asset':
        entry['candidate_assets']['candidate_0']['model'] = 'a_different_model'
    elif problem == 'primitive':
        name = entry['bindings']['candidate_0']['name']
        next(obj for obj in entry['layout']['objects'] if obj['name'] == name)['type'] = 'PrimitiveObject'
    elif problem == 'block_wording':
        entry['public']['environment'] = 'A table with blocks.'
    elif problem == 'duplicate_name':
        entry['bindings']['candidate_0']['name'] = entry['bindings']['candidate_1']['name']
    else:
        entry['task_id'] = task['id']
    with pytest.raises(ValueError):
        builder.translate_definitions(task, scene, entry, {})


def test_duplicate_public_ids_are_limited_to_same_seed_matched_appearance_controls():
    parent, parent_scene, entry = _design_entry()
    task, _ = builder.translate_definitions(parent, parent_scene, entry, {})
    first = {'task': task, 'entry': {**entry, 'document_alias': 'R-A1'}}
    second = deepcopy(first)
    second['entry']['document_alias'] = 'R-A2'
    p = second['task']['private']['evaluation']['parameters']
    p['appearance_mode'] = 'direct'
    cue_names = {second['task']['private']['bindings'][key]['name'] for key in p['cue_bindings'].values()}
    for patch in second['task']['scene']['variants']['adapted']['patches']:
        for obj in patch['parameters']['layout']['objects']:
            if obj['name'] in cue_names:
                obj['position'][0] -= .4
    assert builder._matched_appearance_pair(first, second) is True
    wrong = deepcopy(second)
    wrong['task']['defaults']['seed'] += 1
    assert builder._matched_appearance_pair(first, wrong) is False
    wrong = deepcopy(second)
    wrong['task']['public']['environment'] += ' Different public evidence.'
    assert builder._matched_appearance_pair(first, wrong) is False
    wrong = deepcopy(second)
    wrong['entry']['document_alias'] = 'R-C1'
    assert builder._matched_appearance_pair(first, wrong) is False
    wrong = deepcopy(second)
    wrong['task']['scene']['variants']['adapted']['patches'][0]['parameters']['layout']['objects'][0]['position'][0] += .1
    assert builder._matched_appearance_pair(first, wrong) is False


@pytest.mark.parametrize('field,value', [('n_dof', True), ('n_joints', np.bool_(False)),
    ('n_dof', float('nan')), ('n_joints', float('inf')), ('n_dof', -1),
    ('n_joints', .5), ('n_dof', None), ('n_joints', '0')])
def test_household_requires_measured_finite_nonnegative_integral_joint_counts(household, field, value):
    setattr(household.target, field, value)
    with pytest.raises(ValueError, match='finite nonnegative integer'):
        adapter._household_geometry(household.target)

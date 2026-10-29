"""Check native mirror contracts and exact optical algebra on CPU fixtures.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_mirror_adapter.py
No real render, simulator, GPU, model, queue or reference execution is run.
Read-only geometry fixtures share the dimensional USD hierarchy contract.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from wia.authoring.converters import wia_mirror as adapter
from harness_integration.test_wia_dimensional_adapter import NativeCube, native as native_geometry


class Material:
    def __init__(self):
        self.prim_path = '/World/wia_mirror_panel/Looks/native_pbr'
        self.values = deepcopy(adapter.MATERIAL)
        self.allow_writes, self.apply_writes = False, True
        self.writes, self.reads = [], []

    def get_input(self, inp):
        self.reads.append(inp)
        return self.values[inp]

    def set_input(self, inp, val):
        assert self.allow_writes, 'Material setters are forbidden during episode measurement.'
        self.writes.append((inp, val))
        if self.apply_writes:
            self.values[inp] = val

    @property
    def diffuse_color_constant(self):
        self.reads.append('diffuse_color_constant')
        return np.array(self.values['diffuse_color_constant'])

    @diffuse_color_constant.setter
    def diffuse_color_constant(self, value):
        assert self.allow_writes and hasattr(value, 'detach'), 'Native diffuse setter requires a tensor during initialization only.'
        self.writes.append(('diffuse_color_constant', value.tolist()))
        if self.apply_writes:
            self.values['diffuse_color_constant'] = value.tolist()


@pytest.fixture
def mirror(native_geometry):
    native = native_geometry
    objects = {
        'table': NativeCube('wia_mirror_table', [1.2, 2.4, .1], [8., -1., .61], fixed=True),
        'pad_blue': NativeCube('wia_mirror_blue', [.3, .35, .012], [8.32, -1.65, .666], fixed=True),
        'pad_yellow': NativeCube('wia_mirror_yellow', [.3, .35, .012], [8.32, -.35, .666], fixed=True),
        'mirror': NativeCube(adapter.PANEL_NAME, [.025, 1.6, 1.5], [8.82, -1., 1.2], fixed=True),
        'anchor': NativeCube('wia_reference_beacon', [.1, .1, .3], [8.13, -2.04, .81], fixed=True),
    }
    eye, panel_x, cue_x = np.array([6.7, -1., 1.05]), 8.8075, 8.79
    selected = np.array([7.6, -1.5, .72])
    virtual = np.array([2 * panel_x - selected[0], selected[1], selected[2]])
    t = (cue_x - eye[0]) / (virtual[0] - eye[0])
    cue_y = float((eye + t * (virtual - eye))[1])
    for key, dimensions, position in (
        ('cue_low', [.008, .012, .5], [cue_x, cue_y - .126, .95]),
        ('cue_high', [.008, .012, .5], [cue_x, cue_y + .126, .95]),
        ('cue_bottom', [.008, .264, .012], [cue_x, cue_y, .694]),
        ('cue_top', [.008, .264, .012], [cue_x, cue_y, 1.206]),
    ):
        objects[key] = NativeCube('wia_mirror_' + key, dimensions, position, fixed=True)
    for index, y in enumerate((-1.5, -1., -.5)):
        objects[f'candidate_{index}'] = NativeCube(f'wia_mirror_candidate_{index}', [.065, .065, .12], [7.6, y, .72])
    native.objects.clear()
    native.objects.update(objects)
    for obj in objects.values():
        for field, state in (('touching', native.states.Touching), ('on_top', native.states.OnTop)):
            obj.states[state] = NS(get_value=lambda other, obj=obj, field=field: obj.relations.get((other.name, field), False))
    material = Material()
    panel = objects['mirror']
    panel.root_link.visual_meshes = {'visual': NS(prim_path=panel.prim_path + '/body/visual', material=material)}
    camera_rotation = Rotation.from_matrix(np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]))
    sensor = NS(position=eye.copy(), quaternion=camera_rotation.as_quat(),
                intrinsic_matrix=np.array([[300., 0., 320.], [0., 300., 240.], [0., 0., 1.]]),
                image_width=640, image_height=480)
    sensor.get_position_orientation = lambda: (sensor.position.copy(), sensor.quaternion.copy())
    native.robot.sensors = {'native_head': sensor}
    native.session.robot_camera_registry = {'agentview_rgb': 'native_head'}
    native.session.camera_name = 'native_head'
    native.session.scene_adapter = NS(bindings={'mirror': NS(name=adapter.PANEL_NAME)})
    root = __import__('wia.resources', fromlist=['ROOT']).ROOT
    scoring = json.loads((root / 'explore_to_achieve/wia_v1/mirror_v1/design.json').read_text())['scoring']
    config = {'type': 'category_operation', 'module': adapter.MODULE, 'operation': 'classify_appearance_deliver',
        'scene': adapter.native.SCENE, 'room': adapter.native.ROOM,
        'bindings': {key: obj.name for key, obj in objects.items()}, 'parameters': {
            'revision': adapter.REVISION, 'operation': 'classify_appearance_deliver',
            'candidates': {key: obj.name for key, obj in objects.items() if key.startswith('candidate')},
            'candidate_dimensions_m': {key: [.065, .065, .12] for key in objects if key.startswith('candidate')},
            'fixture_keys': [key for key in objects if not key.startswith('candidate')],
            'scoring': scoring, 'table_key': 'table', 'mirror_key': 'mirror', 'anchor_key': 'anchor',
            'cue_bindings': {'lower_side': 'cue_low', 'upper_side': 'cue_high', 'roof': 'cue_top', 'floor': 'cue_bottom'},
            'destinations': {'blue': 'pad_blue', 'yellow': 'pad_yellow'}, 'appearance_mode': 'reflected'}}
    native.config, native.material, native.sensor = config, material, sensor
    native.target = objects['candidate_0']
    native.init_parameters = {'revision': adapter.REVISION,
        'layout': {'scene_model': adapter.native.SCENE, 'rooms': [adapter.native.ROOM], 'objects': [{'name': adapter.PANEL_NAME}]},
        'mirror_key': 'mirror', 'mirror_material': {'binding': 'mirror', **deepcopy(adapter.MATERIAL)}}
    return native


def sample(mirror):
    return adapter._sample(mirror.session, mirror.controller, mirror.config, mirror.objects)


def instance(mirror):
    return adapter.create_evaluator(mirror.session, mirror.controller, mirror.config, mirror.output)


def set_direct(mirror):
    eye, center = mirror.sensor.position, mirror.target.position
    cue_x = 7.45
    t = (cue_x - eye[0]) / (center[0] - eye[0])
    y = float((eye + t * (center - eye))[1])
    old = (mirror.objects['cue_low'].position[1] + mirror.objects['cue_high'].position[1]) / 2
    for key in mirror.config['parameters']['cue_bindings'].values():
        mirror.objects[key].position[0] = cue_x
        mirror.objects[key].position[1] += y - old
    mirror.config['parameters']['appearance_mode'] = 'direct'


@pytest.mark.parametrize('mode', ['reflected', 'direct'])
def test_initial_geometry_capture_reads_native_eye_all_geometry_and_no_execution_paths(mirror, monkeypatch, mode):
    if mode == 'direct':
        set_direct(mirror)
    mirror.clock.current_time = .375
    mirror.sensor.position[0] -= .05
    mirror.config['parameters']['candidate_dimensions_m']['candidate_0'] = [9., 9., 9.]
    mirror.session.category_interaction_setup['accepted'] = False
    original = deepcopy(mirror.config)
    before_files = sorted(mirror.output.iterdir())

    def forbidden(*args, **kwargs):
        raise AssertionError('Initial geometry capture attempted qualification, control, contact, rendering or writing.')

    for name in ('_sample', '_qualification', 'qualify_operation', 'MirrorEvaluator', 'create_evaluator', 'write_json'):
        monkeypatch.setattr(adapter, name, forbidden)
    monkeypatch.setattr(adapter.native, '_measure', forbidden)
    mirror.controller.geometric_rearrangement_evidence = forbidden
    mirror.session.env.step = forbidden
    for name in ('step', 'render', 'reset', 'play', 'stop'):
        setattr(mirror.session.pipeline.og.sim, name, forbidden)
    mirror.sensor.get_obs = forbidden
    for obj in mirror.objects.values():
        obj.contact_list = obj.get_linear_velocity = obj.get_angular_velocity = forbidden
        obj.states = {key: NS(get_value=forbidden) for key in obj.states}
    calls, geometry = [], adapter.native._geometry

    def measured(obj):
        calls.append(obj.name)
        return geometry(obj)

    monkeypatch.setattr(adapter.native, '_geometry', measured)
    with monkeypatch.context() as writes:
        writes.setattr(Path, 'open', forbidden)
        result = adapter.capture_initial_geometry(mirror.session, mirror.config, mirror.objects)
    assert result['schema'] == 'wia_mirror_initial_geometry_v1' and result['available'] is True
    assert result['native_time_s'] == .375 and result['errors'] == []
    assert result['camera']['position_m'] == mirror.sensor.position.tolist()
    assert result['optical_selection']['frozen_eye_m'] == mirror.sensor.position.tolist()
    assert result['optical_selection']['accepted'] is True
    assert result['optical_selection']['appearance_mode'] == mode
    assert result['optical_selection']['selected_candidate'] == 'candidate_0'
    assert result['targets']['candidate_0']['dimensions_m'] == pytest.approx([.065, .065, .12])
    assert result['targets']['candidate_0']['position_m'] == mirror.target.position.tolist()
    assert set(result['fixtures']) == set(mirror.config['parameters']['fixture_keys'])
    assert sorted(calls) == sorted(mirror.config['bindings'].values())
    assert all(result[field] is False for field in ('native_qualification_run', 'optical_visibility_verified',
        'optical_necessity_verified', 'physical_success_verified'))
    assert mirror.material.writes == [] and mirror.config == original
    assert sorted(mirror.output.iterdir()) == before_files
    json.dumps(result, allow_nan=False)


def test_initial_geometry_capture_keeps_complete_measurements_when_selection_is_ambiguous(mirror):
    mirror.objects['cue_low'].position[1] -= 1.
    mirror.objects['cue_high'].position[1] += 1.
    result = adapter.capture_initial_geometry(mirror.session, mirror.config, mirror.objects)
    assert result['available'] is True and result['errors'] == []
    optical = result['optical_selection']
    assert optical['accepted'] is False and optical['selected_candidate'] is None
    assert len(optical['passing_candidates']) > 1 and optical['errors']
    assert set(optical['candidates']) == set(mirror.config['parameters']['candidates'])


@pytest.mark.parametrize('missing', ['camera', 'candidate_geometry', 'other_fixture', 'material', 'clock', 'identity'])
def test_initial_geometry_capture_missing_native_evidence_has_no_proposal_fallback(mirror, missing):
    if missing == 'camera':
        mirror.sensor.intrinsic_matrix = None
    elif missing == 'candidate_geometry':
        mirror.target.mesh.collision_enabled = False
    elif missing == 'other_fixture':
        mirror.objects['anchor'].mesh.collision_enabled = False
    elif missing == 'material':
        mirror.objects['mirror'].root_link.visual_meshes['visual'].material = None
    elif missing == 'clock':
        mirror.clock.current_time = float('nan')
    else:
        mirror.target.name = 'a_different_native_object'
    result = adapter.capture_initial_geometry(mirror.session, mirror.config, mirror.objects)
    assert result['available'] is False and result['errors']
    if missing == 'camera':
        assert result['camera']['available'] is False
        assert 'position_m' not in result['camera']
        assert result['optical_selection']['accepted'] is False
    elif missing in ('candidate_geometry', 'identity'):
        assert result['targets']['candidate_0']['available'] is False
        assert 'vertices_m' not in result['targets']['candidate_0']
    elif missing == 'other_fixture':
        assert result['fixtures']['anchor']['available'] is False
        assert result['optical_selection']['accepted'] is True  # This fixture is not used by the rays.
    elif missing == 'material':
        assert result['mirror_material']['available'] is False
    else:
        assert result['native_time_s'] is None
    assert result['native_qualification_run'] is result['physical_success_verified'] is False
    assert mirror.material.writes == []
    json.dumps(result, allow_nan=False)


def test_initial_geometry_capture_rejects_unknown_config_before_native_reads(mirror, monkeypatch):
    mirror.config['parameters']['revision'] = 'an_unsupported_revision'

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid configuration must be rejected before native reads.')

    monkeypatch.setattr(adapter, '_camera', forbidden)
    monkeypatch.setattr(adapter.native, '_geometry', forbidden)
    result = adapter.capture_initial_geometry(mirror.session, mirror.config, mirror.objects)
    assert result['available'] is False and result['errors'][0].startswith('configuration:')
    assert result['targets'] == result['fixtures'] == {}


def test_reflection_uses_measured_panel_head_cue_and_exact_perspective_algebra(mirror):
    measured = sample(mirror)
    optical = measured['optical_selection']
    assert optical['accepted'] is True and optical['selected_candidate'] == 'candidate_0'
    assert optical['appearance_mode'] == 'reflected'
    assert optical['panel_plane']['point_m'][0] == pytest.approx(8.8075)
    row = optical['candidates']['candidate_0']
    c, e, n = np.array(row['centroid_m']), mirror.sensor.position, np.array([-1., 0., 0.])
    p = np.array(optical['panel_plane']['point_m'])
    v = c - 2 * n * np.dot(n, c - p)
    hit = e + (optical['cue_plane_x_m'] - e[0]) / (v[0] - e[0]) * (v - e)
    assert row['virtual_point_m'] == pytest.approx(v)
    assert row['cue_intersection_m'] == pytest.approx(hit)
    assert 0 < row['cue_ray_t'] < row['panel_ray_t'] < 1
    local = Rotation.from_quat(mirror.sensor.quaternion).inv().apply(v - e) * [1., -1., -1.]
    projected = mirror.sensor.intrinsic_matrix @ local
    assert row['pixel_xy'] == pytest.approx(projected[:2] / projected[2])
    assert optical['optical_visibility_verified'] is optical['optical_necessity_verified'] is False
    assert adapter.qualify_operation(mirror.session, mirror.controller, mirror.config, mirror.objects)['accepted'] is True
    assert mirror.material.writes == []


def test_direct_appearance_is_determined_by_native_depth_and_not_requested_label(mirror):
    set_direct(mirror)
    optical = sample(mirror)['optical_selection']
    assert optical['accepted'] is True and optical['appearance_mode'] == 'direct'
    row = optical['candidates']['candidate_0']
    assert row['virtual_point_m'] == pytest.approx(mirror.target.position)
    assert 'panel_intersection_m' not in row
    mirror.config['parameters']['appearance_mode'] = 'reflected'
    mismatch = sample(mirror)['optical_selection']
    assert mismatch['appearance_mode'] == 'direct' and mismatch['accepted'] is False


@pytest.mark.parametrize('problem', ['ambiguous', 'off_window', 'off_panel', 'off_frame', 'reversed_depth', 'rod_depth', 'panel_angle'])
def test_ambiguous_or_unavailable_geometry_has_no_source_or_proposal_fallback(mirror, problem):
    if problem == 'ambiguous':
        mirror.objects['cue_low'].position[1] -= 1.
        mirror.objects['cue_high'].position[1] += 1.
    elif problem == 'off_window':
        for key in mirror.config['parameters']['cue_bindings'].values():
            mirror.objects[key].position[1] += 5.
    elif problem == 'off_panel':
        mirror.objects['mirror'].scale[1] = .01
    elif problem == 'off_frame':
        mirror.sensor.intrinsic_matrix[0, 0] = 10000.
    elif problem == 'reversed_depth':
        for key in mirror.config['parameters']['cue_bindings'].values():
            mirror.objects[key].position[0] = mirror.target.position[0]
    elif problem == 'rod_depth':
        mirror.objects['cue_low'].position[0] += .02
    else:
        mirror.objects['mirror'].quaternion = Rotation.from_euler('z', 5, degrees=True).as_quat()
    optical = sample(mirror)['optical_selection']
    assert optical['available'] is optical['accepted'] is False
    assert optical['selected_candidate'] is None and optical['errors']
    receipt = instance(mirror).begin_episode(mirror.observation)
    assert receipt['initial_qualification']['accepted'] is False
    assert receipt['evaluation_validity'] == 'unavailable'
    assert receipt['task_success'] is receipt['physical_success'] is False


@pytest.mark.parametrize('missing', ['registry_key', 'sensor', 'calibration', 'support', 'upright', 'upright_measurement', 'contact', 'material'])
def test_missing_native_sensor_support_upright_contact_or_material_cannot_qualify(mirror, missing):
    if missing == 'registry_key':
        mirror.session.robot_camera_registry = {}
    elif missing == 'sensor':
        mirror.robot.sensors = {}
    elif missing == 'calibration':
        mirror.sensor.intrinsic_matrix = None
    elif missing == 'support':
        mirror.session.category_interaction_setup['accepted'] = False
    elif missing == 'upright':
        mirror.session.category_interaction_setup['robot']['upright'] = None
    elif missing == 'upright_measurement':
        mirror.session.category_interaction_setup['robot']['upright_measurement_available'] = None
    elif missing == 'contact':
        mirror.target.contact_error = RuntimeError('Native contact read unavailable.')
    else:
        mirror.objects['mirror'].root_link.visual_meshes['visual'].material = None
    qualification = adapter.qualify_operation(mirror.session, mirror.controller, mirror.config, mirror.objects)
    assert qualification['accepted'] is False
    result = instance(mirror).begin_episode(mirror.observation)
    assert result['task_success'] is result['physical_success'] is False
    assert result['evaluation_validity'] == 'unavailable'


def test_camera_fallback_is_allowed_only_when_registry_is_absent(mirror):
    del mirror.session.robot_camera_registry
    assert adapter._camera(mirror.session)['sensor_name'] == 'native_head'
    mirror.session.robot_camera_registry = {}
    with pytest.raises(KeyError):
        adapter._camera(mirror.session)


@pytest.mark.parametrize('obey', [True, False])
def test_initialization_sets_only_actual_panel_pbr_then_reads_native_values_before_delegated_setup(mirror, monkeypatch, obey):
    mirror.session._category_initialization_active = True
    mirror.material.allow_writes, mirror.material.apply_writes = True, obey
    mirror.material.values = {'metallic_constant': .2, 'reflection_roughness_constant': .6,
                              'diffuse_color_constant': [.1, .2, .3]}
    calls = []

    def delegated(session, parameters):
        calls.append(adapter._material_readback(mirror.objects['mirror']))
        return deepcopy(session.category_interaction_setup)

    monkeypatch.setattr(adapter.active_layout, 'initialize_scene', delegated)
    result = adapter.initialize_scene(mirror.session, mirror.init_parameters)
    receipt = result['mirror_material_initialization']
    assert receipt['setter_count'] == len(mirror.material.writes) == 3
    assert [name for name, _ in mirror.material.writes] == list(adapter.MATERIAL)
    assert receipt['native_panel_name'] == adapter.PANEL_NAME
    assert receipt['accepted'] is obey and calls[0]['accepted'] is obey
    assert receipt['before']['accepted'] is False
    assert receipt['after']['materials'][0]['metallic_constant'] == (1. if obey else .2)
    mirror.material.allow_writes = False
    assert adapter._material_readback(mirror.objects['mirror'])['accepted'] is obey
    assert len(mirror.material.writes) == 3


def test_initialization_resolves_exact_authored_panel_when_semantic_binding_is_absent_and_never_writes_late(mirror, monkeypatch):
    with pytest.raises(RuntimeError, match='initialization-only'):
        adapter.initialize_scene(mirror.session, mirror.init_parameters)
    assert mirror.material.writes == []
    mirror.session._category_initialization_active = True
    mirror.session.scene_adapter.bindings = {}
    mirror.material.allow_writes = True
    monkeypatch.setattr(adapter.active_layout, 'initialize_scene', lambda session, p: deepcopy(session.category_interaction_setup))
    assert adapter.initialize_scene(mirror.session, mirror.init_parameters)['mirror_material_initialization']['accepted'] is True
    mirror.session._category_initialization_active = False
    with pytest.raises(RuntimeError):
        adapter.initialize_scene(mirror.session, mirror.init_parameters)
    assert len(mirror.material.writes) == 3


def test_delegated_construction_and_postprocessing_preserve_the_existing_layout_protocol(mirror, monkeypatch):
    calls = []
    monkeypatch.setattr(adapter.active_layout, 'configure_scene', lambda config, p: calls.append(('configure', config, p)) or {'delegated': True})
    monkeypatch.setattr(adapter.active_layout, 'postprocess_joint_world', lambda session, p: calls.append(('postprocess', session, p)) or {'delegated': True})
    config = {'scene': {}}
    assert adapter.configure_scene(config, mirror.init_parameters) == {'delegated': True}
    assert adapter.postprocess_joint_world(mirror.session, mirror.init_parameters) == {'delegated': True}
    assert calls[0][1] is config and calls[0][2] is mirror.init_parameters
    assert calls[1][1] is mirror.session


def test_native_material_is_read_again_on_updates_and_never_substituted_with_requested_values(mirror):
    evaluator = instance(mirror)
    evaluator.begin_episode(mirror.observation)
    reads = len(mirror.material.reads)
    mirror.material.values['metallic_constant'] = .1
    mirror.clock.current_time = .05
    evaluator.sample_step()
    result = evaluator.record(mirror.observation)
    assert len(mirror.material.reads) > reads
    assert result['native_measurements']['mirror_material']['accepted'] is False
    assert result['native_measurements']['geometry_valid'] is False
    assert result['evaluation_validity'] == 'invalid' and result['task_success'] is False
    assert mirror.material.writes == []


def test_missing_current_material_readback_is_unavailable_instead_of_invented_mutation(mirror):
    evaluator = instance(mirror)
    evaluator.begin_episode(mirror.observation)
    mirror.material.values['metallic_constant'] = None
    mirror.clock.current_time = .05
    evaluator.sample_step()
    result = evaluator.record(mirror.observation)
    assert result['native_measurements']['mirror_material']['available'] is False
    assert result['native_measurements']['geometry_valid'] is None
    assert result['evaluation_validity'] == 'unavailable' and result['task_success'] is False


def test_initial_eye_identity_and_optical_selection_are_frozen_and_record_report_do_not_sample(mirror, monkeypatch):
    original, calls = adapter._sample, []

    def counted(*args, **kwargs):
        calls.append(mirror.clock.current_time)
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, '_sample', counted)
    evaluator = instance(mirror)
    first = evaluator.begin_episode(mirror.observation)
    frozen = deepcopy(first['native_measurements']['optical_selection'])
    mirror.sensor.position[:] = [100., 8., -4.]
    mirror.clock.current_time = .05
    report = evaluator.report_answer('ANSWER: reflected', mirror.observation)
    assert report['report']['latest_valid_before_grasp'] == 'reflected'
    for _ in range(3):
        evaluator.record(mirror.observation)
    assert calls == [0.]
    evaluator.sample_step()
    result = evaluator.record(mirror.observation)
    assert calls == [0., .05]
    assert result['native_measurements']['optical_selection'] == frozen
    assert result['selected_candidate'] == 'candidate_0'
    assert result['optical_visibility_verified'] is result['complete_reference_execution_verified'] is False


def deliver_fixture(mirror, report):
    evaluator = instance(mirror)
    evaluator.begin_episode(mirror.observation)
    if report is not None:
        evaluator.report_answer(report)
    mirror.robot._ag_obj_in_hand['left'] = mirror.target
    mirror.robot._ag_obj_constraints['left'] = object()
    mirror.readings['left'], mirror.robot.positions[:2] = 1, 0.
    mirror.clock.current_time = .05
    evaluator.sample_step()
    mirror.target.position[2] += .06
    mirror.clock.current_time = .10
    evaluator.sample_step()
    evaluator.report_answer('ANSWER: direct')  # Diagnostic after commitment; cannot replace the frozen report.
    mirror.robot._ag_obj_in_hand['left'] = mirror.robot._ag_obj_constraints['left'] = None
    mirror.readings['left'], mirror.robot.positions[:2] = -1, .04
    pad = mirror.objects['pad_blue']
    mirror.target.position[:] = [pad.position[0], pad.position[1], .732]
    mirror.target.relations[(pad.name, 'touching')] = mirror.target.relations[(pad.name, 'on_top')] = True
    for time in np.arange(.15, .751, .05):
        mirror.clock.current_time = float(time)
        evaluator.sample_step()
    return evaluator.record(mirror.observation)


@pytest.mark.parametrize('report,task_success', [('ANSWER: reflected', True), ('ANSWER: direct', False), (None, False)])
def test_native_report_api_preserves_distinct_classification_and_physical_delivery_results(mirror, report, task_success):
    result = deliver_fixture(mirror, report)
    assert result['evaluation_validity'] == 'complete'
    assert result['physical_success'] is True and result['task_success'] is task_success
    assert result['report']['frozen'] is True
    assert result['report']['frozen_value'] == (None if report is None else report.split(': ')[1])
    assert result['physical_scorer_schema'] == 'wia_mirror_progress_v1'


def test_spatial_samples_use_current_native_anchor_table_relations_and_reject_already_correct_start(mirror):
    p = mirror.config['parameters']
    mirror.config['operation'] = p['operation'] = 'mirror_reconfigure'
    p.update(relation='selected_farther', distance_gap_m=.2, requires_both_objects_moved=False,
             non_goal_initialization_required=True)
    p['candidates'].pop('candidate_1')
    p['candidate_dimensions_m'].pop('candidate_1')
    initial = sample(mirror)
    assert initial['anchor_centroid_m'] == pytest.approx(mirror.objects['anchor'].position)
    assert 'table' in initial['targets']['candidate_0']['destinations']
    assert adapter.qualify_operation(mirror.session, mirror.controller, mirror.config, mirror.objects)['accepted'] is True
    p['relation'] = 'selected_nearer'
    qualification = adapter.qualify_operation(mirror.session, mirror.controller, mirror.config, mirror.objects)
    assert qualification['accepted'] is False
    assert 'initial_spatial_goal_already_satisfied' in qualification['initial_reducer_errors']


def test_fixture_mutation_keeps_invalid_status_through_completion_normalization(mirror):
    evaluator = instance(mirror)
    evaluator.begin_episode(mirror.observation)
    mirror.objects['anchor'].position[0] += .003
    mirror.clock.current_time = .05
    evaluator.sample_step()
    result = evaluator.record(mirror.observation)
    assert result['evaluation_validity'] == 'invalid'
    assert result['task_success'] is result['physical_success'] is False
    assert result['completion_score'] is None


@pytest.mark.parametrize('alias', ['r_c1', 'r_c2', 'r_c3', 'r_a1', 'r_a2', 'r_s1', 'r_s2', 'r_s3'])
@pytest.mark.parametrize('seed', [101, 202])
def test_all_16_generated_definition_guards_match_the_fixed_adapter_interface(mirror, alias, seed):
    path = __import__('wia.resources', fromlist=['ROOT']).ROOT / 'explore_to_achieve/wia_v1/mirror_v1/tasks' / f'{alias}_seed{seed}_v1.yaml'
    task = yaml.safe_load(path.read_text())
    config = task['private']['evaluation']
    config.update(scene=adapter.native.SCENE, room=adapter.native.ROOM,
                  bindings={key: value['name'] for key, value in task['private']['bindings'].items()})
    assert adapter._config(mirror.session, config)['revision'] == adapter.REVISION

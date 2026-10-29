"""Check bounded tray-probe inputs and native instrumentation interfaces on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
  pytest -q -p no:cacheprovider tests/test_wia_tray_probe.py
No OmniGibson environment, GPU workload, or production initialization is executed.
"""
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import wia.runtime.commands.probe_wia_tray_initialization as probe


def design():
    return {'robot_yaw_deg': 135., 'right_eef_position_base_m': [.4, -.25, .8],
            'rim_axis': 'y', 'rim_penetration_m': .025, 'maximum_plate_tilt_deg': 10.}


class PoseObject:
    def __init__(self):
        self.position = np.zeros(3)
        self.quaternion = np.array([0., 0., 0., 1.])
        self.calls = []

    def get_position_orientation(self):
        return self.position.copy(), self.quaternion.copy()

    def set_position_orientation(self, position=None, orientation=None, frame='world'):
        self.calls.append({'position': deepcopy(position), 'orientation': deepcopy(orientation), 'frame': frame})
        if position is not None:
            self.position = np.asarray(position, dtype=float)
        if orientation is not None:
            self.quaternion = np.asarray(orientation, dtype=float)
        return 'native_receipt'


def test_explicit_design_and_production_asset_contract_are_preserved():
    value = design()
    checked = probe.validate_design(value)
    assert checked == value and probe.OFFSET_FIELD not in checked
    parameters = probe.production_parameters(checked)
    assert parameters['plate'] == {'name': 'collection_plate', 'category': 'tray', 'model': 'gsxbym',
        'pinch_reference': 'eef_fingertip', 'right_eef_position_base_m': [.4, -.25, .8],
        'rim_axis': 'y', 'rim_penetration_m': .025}
    assert parameters['maximum_plate_tilt_deg'] == 10.
    parameters['plate']['right_eef_position_base_m'][0] = 9.
    assert value['right_eef_position_base_m'][0] == .4
    checked['right_eef_position_base_m'][0] = 8.
    assert value['right_eef_position_base_m'][0] == .4


def test_robot_world_position_default_and_explicit_config_mapping_preserve_other_settings():
    baseline = probe.validate_design(design())
    original = probe.robot_start_configuration(baseline)
    assert original['position'] == [0., 0., .02]
    translated = design()
    translated['robot_position_world_m'] = [4.5, -7., .02]
    checked = probe.validate_design(translated)
    mapped = probe.robot_start_configuration(checked)
    assert mapped == {**original, 'position': [4.5, -7., .02]}
    mapped['position'][0] = 99.
    assert checked['robot_position_world_m'] == [4.5, -7., .02]
    assert translated['robot_position_world_m'] == [4.5, -7., .02]
    assert 'robot_position_world_m' not in baseline


def test_apparatus_staging_default_and_explicit_mapping_preserve_asset_settings():
    baseline = probe.validate_design(design())
    original = probe.apparatus_start_configuration(baseline)
    assert original[0]['position'] == [2., 0., 1.]
    assert original[1]['position'] == [2., 0., .8]
    translated = design()
    translated['apparatus_position_world_m'] = [100., 100., 1.]
    checked = probe.validate_design(translated)
    mapped = probe.apparatus_start_configuration(checked)
    assert mapped == [{**original[0], 'position': [100., 100., 1.]},
                      {**original[1], 'position': [100., 100., .8]}]
    mapped[0]['position'][0] = 99.
    assert checked['apparatus_position_world_m'] == [100., 100., 1.]
    assert translated['apparatus_position_world_m'] == [100., 100., 1.]
    assert 'apparatus_position_world_m' not in baseline
    assert probe.robot_start_configuration(checked) == probe.robot_start_configuration(baseline)


@pytest.mark.parametrize('field', ['robot_position_world_m', 'apparatus_position_world_m'])
@pytest.mark.parametrize('position', [[0., 0.], [0., 0., .02, 1.], None,
    [0., float('nan'), .02], [0., 0., float('inf')], [True, 0., .02], ['4.5', '-7', '.02']])
def test_initial_world_positions_reject_malformed_vectors(field, position):
    value = design()
    value[field] = position
    with pytest.raises(ValueError, match=field):
        probe.validate_design(value)


@pytest.mark.parametrize('field,value', [('robot_yaw_deg', True), ('robot_yaw_deg', float('nan')),
    ('right_eef_position_base_m', [1., 2.]), ('rim_axis', 'z'), ('rim_penetration_m', 0.),
    ('maximum_plate_tilt_deg', 90.), (probe.OFFSET_FIELD, [0., float('inf'), 0.]),
    (probe.OFFSET_FIELD, [True, 0., 0.])])
def test_invalid_inputs_are_rejected(field, value):
    value_design = design()
    value_design[field] = value
    with pytest.raises(ValueError):
        probe.validate_design(value_design)


def test_asset_or_sweep_settings_cannot_be_added():
    value = design()
    value['model'] = 'another_asset'
    with pytest.raises(ValueError):
        probe.validate_design(value)
    value = design()
    value.pop('robot_yaw_deg')
    with pytest.raises(ValueError):
        probe.validate_design(value)


def test_offset_reuses_first_live_base_rotation_for_tray_and_support_and_restores_setters():
    robot, tray, stand = PoseObject(), PoseObject(), PoseObject()
    robot.quaternion = Rotation.from_euler('z', 90., degrees=True).as_quat()
    session = NS(robot=robot, _category_initialization_active=True)
    original_tray, original_stand = tray.set_position_orientation, stand.set_position_orientation
    records = []
    with probe.offset_apparatus(session, tray, stand, [.1, 0., .02], records):
        assert tray.set_position_orientation(position=[1., 2., 3.], orientation=[0., 0., 0., 1.]) == 'native_receipt'
        robot.quaternion = Rotation.from_euler('z', 180., degrees=True).as_quat()
        stand.set_position_orientation(position=[1., 2., 2.98], orientation=[0., 0., 0., 1.])
    assert tray.position == pytest.approx([1., 2.1, 3.02])
    assert stand.position == pytest.approx([1., 2.1, 3.])
    assert records[0]['offset_world_m'] == records[1]['offset_world_m']
    assert records[0]['before_pose']['position_m'] == [0., 0., 0.]
    assert records[0]['after_pose']['position_m'] == pytest.approx(tray.position)
    assert tray.set_position_orientation == original_tray and stand.set_position_orientation == original_stand


def test_zero_offset_preserves_production_pose_values_and_failure_restores_setters():
    robot, tray, stand = PoseObject(), PoseObject(), PoseObject()
    session = NS(robot=robot, _category_initialization_active=True)
    original_tray, original_stand = tray.set_position_orientation, stand.set_position_orientation
    position = [.312345678901234, -.498765432109876, .891234567890123]
    quaternion = Rotation.from_euler('z', 135., degrees=True).as_quat().tolist()
    with pytest.raises(RuntimeError, match='production failure'):
        with probe.offset_apparatus(session, tray, stand, [0., 0., 0.], []):
            tray.set_position_orientation(position=position, orientation=quaternion)
            stand.set_position_orientation(position=position, orientation=quaternion)
            raise RuntimeError('production failure')
    assert np.array_equal(tray.position, position) and np.array_equal(stand.position, position)
    assert np.array_equal(tray.quaternion, quaternion)
    assert tray.set_position_orientation == original_tray and stand.set_position_orientation == original_stand
    session._category_initialization_active = False
    with probe.offset_apparatus(session, tray, stand, [0., 0., 0.], []):
        with pytest.raises(RuntimeError, match='initialization-only'):
            tray.set_position_orientation(position=position)


def test_later_native_pose_restores_are_forwarded_unchanged_after_initialization_ends():
    robot, tray, stand = PoseObject(), PoseObject(), PoseObject()
    session = NS(robot=robot, _category_initialization_active=True)
    tray_calls, stand_calls = [], []

    def native_setter(calls):
        def setter(*args, **kwargs):
            calls.append((args, kwargs))
            return 'native_pose_restore'
        return setter

    tray.set_position_orientation = native_setter(tray_calls)
    stand.set_position_orientation = native_setter(stand_calls)
    records = []
    position, quaternion = np.array([.41, -.32, .83]), np.array([0., 0., 0., 1.])
    with probe.offset_apparatus(session, tray, stand, [.03, 0., 0.], records):
        tray.set_position_orientation(position=[1., 2., 3.], orientation=[0., 0., 0., 1.])
        stand.set_position_orientation(position=[1., 2., 2.98], orientation=[0., 0., 0., 1.])
        session._category_initialization_active = False
        for obj, calls in ((tray, tray_calls), (stand, stand_calls)):
            assert obj.set_position_orientation(position, quaternion, frame='parent') == 'native_pose_restore'
            args, kwargs = calls[-1]
            assert args[0] is position and args[1] is quaternion
            assert kwargs == {'frame': 'parent'}
            assert obj.set_position_orientation(orientation=quaternion) == 'native_pose_restore'
            assert calls[-1][0] == () and calls[-1][1]['orientation'] is quaternion
            assert len(calls) == 3
    assert len(records) == 2
    assert tray_calls[0][1]['position'] == pytest.approx([1.03, 2., 3.])


def test_uint8_camera_buffer_preserves_pixel_values_when_saved(tmp_path):
    from core.record.images import save_png
    from PIL import Image

    rgb = np.array([[[1, 32, 128], [255, 0, 64]], [[17, 97, 173], [3, 127, 254]]], dtype=np.uint8)

    class NativeTensorBuffer:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return rgb

    detached = probe.native_rgb_array(NativeTensorBuffer())
    assert detached.dtype == np.uint8 and detached is rgb
    assert probe.native_rgb_array(rgb) is rgb
    path = tmp_path / 'native_rgb.png'
    save_png(path, detached)
    with Image.open(path) as saved:
        assert np.array_equal(np.asarray(saved), rgb)


def test_native_ray_endpoint_coordinates_use_actual_link_pose_once():
    start, end = PoseObject(), PoseObject()
    start.position, end.position = np.array([1., 2., 3.]), np.array([4., 5., 6.])
    start.quaternion = end.quaternion = Rotation.from_euler('z', 90., degrees=True).as_quat()
    robot = NS(links={'finger_a': start, 'finger_b': end},
        assisted_grasp_start_points={'right': [NS(link_name='finger_a', position=[.1, 0., 0.])]},
        assisted_grasp_end_points={'right': [NS(link_name='finger_b', position=[0., .1, 0.])]})
    result = probe.ray_endpoints(robot)
    assert result['start'][0]['world_position_m'] == pytest.approx([1., 2.1, 3.])
    assert result['end'][0]['world_position_m'] == pytest.approx(np.array([3.9, 5., 6.]) + 1e-8)


def test_structured_physx_contact_scalars_preserve_xyz_fields_and_finite_checks():
    dtype = np.dtype([('z', 'f4'), ('x', 'f4'), ('y', 'f4')])
    position = np.array((.3, .1, .2), dtype=dtype)[()]
    normal = np.array((1., 0., 0.), dtype=dtype)[()]
    assert probe._array(position) == pytest.approx([.1, .2, .3])
    contact = NS(body0='/World/robot/finger', body1='/World/collection_plate/base_link',
        position=position, normal=normal, impulse=None)
    robot, plate = NS(contact_list=lambda: [contact]), NS(contact_list=lambda: [contact])
    snapshot = probe.native_snapshot(NS(robot=robot), plate, 'cpu_contact_interface')
    for key in ('robot_contacts', 'tray_contacts'):
        assert snapshot[key][0]['position'] == pytest.approx([.1, .2, .3])
        assert snapshot[key][0]['normal'] == pytest.approx([0., 0., 1.])
    with pytest.raises(ValueError, match='finite'):
        probe._array(np.array((1., float('nan'), 0.), dtype=dtype)[()])


def test_closure_observer_requires_actual_receipt_and_support_and_observes_after_step(monkeypatch):
    events = []
    support = [object()]
    session = NS(env=NS(scene=NS(object_registry=lambda *args: support[0])), _monitor_public_motion=False)
    report = {'session_control_steps': 0, 'closure_steps_observed': 0, 'physical_samples': [], 'support_removed': False}
    import wia.runtime.sim.esi_rendering as esi_rendering
    monkeypatch.setattr(esi_rendering, 'step_without_observation', lambda owner, action: events.append('step') or 'result')
    monkeypatch.setattr(probe, 'native_motion_sample', lambda *args: events.append('physical_sample') or {})
    monkeypatch.setattr(probe, 'native_snapshot', lambda *args: pytest.fail('No snapshots inside control substeps.'))
    step = probe.production_step_observer(session, object(), report, lambda: events.append('save'))
    assert step({}) == 'result'
    assert report['physical_samples'] == []
    session.hidden_collection_initialization = {'stage': 'closing_right_gripper', 'accepted': False}
    for _ in range(11):
        step({})
    assert [row['closure_step_index'] for row in report['physical_samples']] == [0, 5, 10]
    assert report['closure_steps_observed'] == 11
    assert events[events.index('physical_sample') - 1] == 'step'
    support[0] = None
    for _ in range(90):
        step({})
    assert report['closure_steps_observed'] == 11
    assert report['support_removed'] is True
    assert len(report['physical_samples']) == 4
    assert 'save' not in events


def test_cli_preserves_input_source_provenance_and_existing_evidence(tmp_path, monkeypatch):
    path, output = tmp_path / 'design.json', tmp_path / 'output'
    raw = json.dumps(design(), indent=4).encode() + b'\n\n'
    path.write_bytes(raw)
    monkeypatch.setattr(probe, 'run', lambda value, directory, commands: {'accepted': False})
    assert probe.main(['--design-file', str(path), '--output-dir', str(output)]) == 1
    assert (output / 'input_design.json').read_bytes() == raw
    provenance = json.loads((output / 'code_provenance.json').read_text())
    assert provenance['design_sha256'] == hashlib.sha256(raw).hexdigest()
    assert provenance['script_sha256'] == hashlib.sha256((output / 'probe_source.py').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='fresh output'):
        probe.main(['--design-file', str(path), '--output-dir', str(output)])


def test_importing_the_probe_does_not_import_simulator():
    completed = subprocess.run([sys.executable, '-B', '-c',
        "import sys; import wia.runtime.commands.probe_wia_tray_initialization; assert 'omnigibson' not in sys.modules"],
        cwd=probe.ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def authored_commands():
    return [{'action': token, 'distance_m': distance} for token, distance in (
        ('TRUNK_DOWN', .08), ('TRUNK_DOWN', .08), ('TRUNK_UP', .08), ('TRUNK_UP', .08),
        ('BASE_BACK', .05), ('BASE_FWD', .05))]


def test_authored_public_commands_use_exact_existing_fine_resolution_and_forbid_rotation():
    commands = authored_commands()
    assert probe.validate_commands(commands) == commands
    for value in ([], [{'action': 'ROTATE_L', 'angle_deg': .05}],
                  [{'action': 'BASE_FWD', 'distance_m': .08}],
                  [{'action': 'TRUNK_DOWN', 'distance_m': .085}]):
        with pytest.raises(ValueError):
            probe.validate_commands(value)


def test_commands_cli_preserves_input_and_passes_validated_list(tmp_path, monkeypatch):
    path, commands_path, output = tmp_path / 'design.json', tmp_path / 'commands.json', tmp_path / 'out'
    path.write_text(json.dumps(design()))
    raw = json.dumps(authored_commands(), indent=4).encode() + b'\n\n'
    commands_path.write_bytes(raw)
    received = []
    monkeypatch.setattr(probe, 'run', lambda value, directory, commands:
        received.append(commands) or {'accepted': True})
    assert probe.main(['--design-file', str(path), '--output-dir', str(output),
        '--commands-file', str(commands_path)]) == 0
    assert received == [authored_commands()]
    assert (output / 'input_commands.json').read_bytes() == raw
    provenance = json.loads((output / 'code_provenance.json').read_text())
    assert provenance['commands_sha256'] == hashlib.sha256(raw).hexdigest()


def test_public_step_observer_records_raw_loss_without_sensor_capture(monkeypatch):
    import wia.runtime.sim.esi_rendering as esi_rendering
    session = NS(_monitor_public_motion=True, _probe_attachment_lost=False)
    report = {'session_control_steps': 0, 'physical_samples': []}
    monkeypatch.setattr(esi_rendering, 'step_without_observation', lambda owner, action: None)
    monkeypatch.setattr(probe, 'native_motion_sample', lambda *args: {'attachment_retained': False})
    step = probe.production_step_observer(session, object(), report, lambda: pytest.fail('No substep save.'))
    step({})
    assert session._probe_attachment_lost is True
    assert report['physical_samples'] == [{'attachment_retained': False, 'phase': 'public_motion'}]


@pytest.mark.parametrize('failure', ['command', 'attachment', None])
def test_public_probe_sequence_stops_at_first_native_failure(monkeypatch, failure):
    import wia.runtime.sim.esi_actions as esi_actions
    import wia.runtime.sim.esi_gripper as esi_gripper
    import wia.runtime.sim.esi_trunk as esi_trunk
    from core.action_compiler import PrimitiveResolution
    session = NS(_category_initialization_active=True, _probe_attachment_lost=False,
        pipeline=NS(og=NS(sim=NS(current_time=0.))))
    report = {'public_commands': []}
    calls = []
    monkeypatch.setattr(esi_gripper, 'R1ManipulationController', lambda *args, **kwargs: NS())
    monkeypatch.setattr(esi_trunk, 'R1TrunkController', lambda *args: NS())
    monkeypatch.setattr(probe, 'native_motion_sample', lambda *args: {'attachment_retained': not session._probe_attachment_lost})
    class Executor:
        primitive_resolution = PrimitiveResolution(base_translation_m=.05, trunk_translation_m=.005)
        _execute_metric = lambda *args: None
        def execute_macro(self, command, **kwargs):
            calls.append(command)
            assert kwargs['should_cancel']() is False
            session.pipeline.og.sim.current_time += .25
            if failure == 'attachment':
                session._probe_attachment_lost = True
            success = failure != 'command'
            return {'success': success, 'child_results': [{'target_height_m': .9, 'trunk_height_after_m': .9}]}
    executor = Executor()
    monkeypatch.setattr(esi_actions, 'create_r1_executor', lambda controller: executor)
    if failure:
        with pytest.raises(RuntimeError, match='attachment was lost|Public command failed'):
            probe.public_motion_probes(session, object(), authored_commands(), report, lambda: None)
    else:
        probe.public_motion_probes(session, object(), authored_commands(), report, lambda: None)
    assert len(calls) == len(report['public_commands']) == (1 if failure else 6)
    assert calls == authored_commands()[:len(calls)]
    assert report['public_commands'][0]['simulated_time_s'] == .25
    assert report['public_commands'][0]['height_measurements'] == [{'target_height_m': .9, 'trunk_height_after_m': .9}]
    assert executor.macro_execution == 'metric_target'
    assert session._category_initialization_active is False
    assert report['public_motion_probes_passed'] is (failure is None)

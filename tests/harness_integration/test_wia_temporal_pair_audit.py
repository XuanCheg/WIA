"""Check native-pair audit math and missing-evidence rules with synthetic captures.

Run: python -B -m pytest -q tests/test_wia_temporal_pair_audit.py
These fixtures are synthetic measurements, not native execution evidence.
"""
from copy import deepcopy
import json
import numpy as np
from PIL import Image
import pytest

from core.robot_cameras import ROBOT_CAMERA_LABELS
from wia.tasks.loader import load_task
from wia.runtime.commands.audit_wia_temporal_pair import ROOT, audit_pair, main


TASKS = [ROOT / 'explore_to_achieve/wia_v1/temporal/tasks' / f'history_{key}_v1.yaml'
         for key in ('a', 'b')]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def pair(tmp_path):
    args = {'left_session': tmp_path / 'left', 'right_session': tmp_path / 'right',
            'left_label': 'left_phase_2', 'right_label': 'right_phase_2',
            'left_task': TASKS[0], 'right_task': TASKS[1]}
    for side, task_path in zip(('left', 'right'), TASKS):
        loaded = load_task(task_path, variant='adapted', run_mode='diagnostic')
        p = loaded.definition.private.evaluation.parameters
        capture = args[side + '_session'] / args[side + '_label']
        capture.mkdir(parents=True)
        objects = {}
        for index, (key, binding) in enumerate(sorted(loaded.definition.private.bindings.items())):
            position = [float(index), 0., 0.5]
            if key.startswith('image1_'):
                position = [1000. if side == 'left' else 2000., 1000., -5.]
            objects[binding.name] = {
                'category': binding.category, 'position_m': position,
                'quaternion_xyzw': [0., 0., 0., 1.],
                'aabb_m': [[value - .05 for value in position], [value + .05 for value in position]],
                'scale': [1., 1., 1.],
            }
        camera = {'pose_world': [[0., 0., 1.], [0., 0., 0., 1.]],
                  'intrinsics': [[1., 0., 1.], [0., 1., 1.], [0., 0., 1.]],
                  'resolution': [2, 2]}
        state = {
            'objects': objects, 'cameras': {name: deepcopy(camera) for name in ROBOT_CAMERA_LABELS},
            'robot': {'position': [0., 0., 0.], 'quaternion_xyzw': [0., 0., 0., 1.], 'yaw_rad': 0.},
            'temporal_initialization': {'phase': 'phase_2', 'valid': True,
                'slots': {'container_00': None, 'container_01': {'category': 'jar_of_cumin', 'model': 'tsktnz'},
                          'container_02': None},
                'objects': {key: {'fixed_base': True, 'visual_only': False} for key in p['phase_keys']['image1']},
                'checks': {'inactive:' + key: True for key in p['phase_keys']['image1']}},
        }
        observation = {
            'task_text': loaded.task_text, 'snapshot_id': 12, 'captured_at': '2026-10-23T09:00:00+00:00',
            'action_step': 0, 'action_boundary': True, 'action_history': [],
            'robot_state': None, 'last_action_feedback': None, 'sensors': {},
            'perception': None, 'observation_config': {}, 'camera_geometry': {},
            'initial_overview': {'required': True, 'status': 'available',
                'refresh_policy': 'frozen_at_episode_start',
                'capture': {'robot_start_position_m': [0., 0., 0.], 'robot_start_yaw_rad': 0.}},
            'images': {},
        }
        for name in (*ROBOT_CAMERA_LABELS, 'initial_overview_rgb'):
            pixels = np.zeros((2, 2, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(capture / f'{name}.png')
            observation['images'][name] = {'path': f'{name}.png', 'shape': [2, 2, 3], 'dtype': 'uint8'}
        write_json(args[side + '_session'] / (args[side + '_label'] + '_state.json'), state)
        write_json(capture / 'observation.json', observation)
    return args


def change_state(pair, change):
    path = pair['right_session'] / (pair['right_label'] + '_state.json')
    state = json.loads(path.read_text(encoding='utf-8'))
    change(state)
    write_json(path, state)


def test_matching_measurements_separate_current_match_from_history_and_reference_claims(pair, tmp_path):
    report = audit_pair(**pair)
    assert report['native_geometry_match'] is True
    assert report['public_task_match'] is True
    assert report['current_rgb_exact_match'] is True
    assert report['no_history_condition_verified'] is False
    assert report['legal_reference_execution_verified'] is False
    assert report['status'] == 'current_state_matches_only'
    assert report['native_geometry']['mismatches'] == []
    assert report['native_geometry']['inconclusive_fields'] == []
    assert len(report['current_mounted_rgb']) == 8
    assert all(row['mae_uint8_rgb'] == row['differing_pixel_fraction'] == 0
               for row in report['current_mounted_rgb'].values())
    parked = report['parked_phase_1_objects']
    assert parked['observation_leak_excluded'] is False
    assert parked['captures']['left'][0]['actual_geometry']['position_m'] != (
        parked['captures']['right'][0]['actual_geometry']['position_m'])
    assert all(row['possible_observation_leak'] for rows in parked['captures'].values() for row in rows)
    events = next(row for row in report['public_metadata'] if row['field'] == 'public_task_events')
    assert events['equal'] is True
    assert events['left_present'] is events['right_present'] is False
    output = tmp_path / 'audit.json'
    cli = [item for key, value in pair.items() for item in ('--' + key.replace('_', '-'), str(value))]
    assert main([*cli, '--output', str(output)]) == 0
    assert json.loads(output.read_text())['native_geometry_match'] is True


def test_actual_position_and_aabb_mismatches_are_reported_with_maximum_deltas(pair):
    name = load_task(TASKS[1], variant='adapted', run_mode='diagnostic').definition.private.bindings['container_00'].name
    def move(state):
        state['objects'][name]['position_m'][0] += .002
        for corner in state['objects'][name]['aabb_m']:
            corner[0] += .002
    change_state(pair, move)
    report = audit_pair(**pair)
    assert report['native_geometry_match'] is False
    assert report['current_rgb_exact_match'] is True
    differences = report['native_geometry']['mismatches']
    assert {row['field'] for row in differences} == {'position_l2_m', 'aabb_max_abs_m'}
    assert {row['binding_key'] for row in differences} == {'container_00'}
    assert report['native_geometry']['maximum_deltas']['position_l2_m'] == pytest.approx(.002)


def test_normalized_quaternion_sign_does_not_create_a_pose_difference(pair):
    def negate(state):
        for row in state['objects'].values():
            row['quaternion_xyzw'] = [0., 0., 0., -2.]
        state['robot']['quaternion_xyzw'] = [0., 0., 0., -2.]
        for row in state['cameras'].values():
            row['pose_world'][1] = [0., 0., 0., -2.]
    change_state(pair, negate)
    report = audit_pair(**pair)
    assert report['native_geometry_match'] is True
    assert report['native_geometry']['maximum_deltas']['quaternion_geodesic_rad'] == 0.


@pytest.mark.parametrize('missing', ['camera_measurement', 'camera_png', 'native_binding'])
def test_missing_required_camera_or_native_binding_is_inconclusive(pair, missing):
    if missing == 'camera_measurement':
        change_state(pair, lambda state: state['cameras'].pop('wrist_rgb'))
    elif missing == 'native_binding':
        binding = load_task(TASKS[1], variant='adapted', run_mode='diagnostic').definition.private.bindings['container_00']
        change_state(pair, lambda state: state['objects'].pop(binding.name))
    else:
        (pair['right_session'] / pair['right_label'] / 'wrist_rgb.png').unlink()
    report = audit_pair(**pair)
    assert report['status'] == 'inconclusive'
    assert (report['current_rgb_exact_match'] if missing == 'camera_png' else report['native_geometry_match']) is None


def test_overview_and_recorded_start_differences_do_not_enter_current_matching(pair):
    capture = pair['right_session'] / pair['right_label']
    Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(capture / 'initial_overview_rgb.png')
    path = capture / 'observation.json'
    observation = json.loads(path.read_text())
    observation['initial_overview']['capture']['robot_start_position_m'][0] = .2
    observation['captured_at'] = '2026-10-23T09:00:01+00:00'
    write_json(path, observation)
    report = audit_pair(**pair)
    assert report['native_geometry_match'] is report['current_rgb_exact_match'] is True
    assert report['frozen_initial_overview']['rgb']['exact_pixels_equal'] is False
    assert report['frozen_initial_overview']['rgb']['mae_uint8_rgb'] == 255.
    assert report['frozen_initial_overview']['robot_start']['position_l2_m']['match'] is False
    assert next(row for row in report['public_metadata'] if row['field'] == 'captured_at')['equal'] is False
    assert report['no_history_condition_verified'] is False


def test_current_rgb_metrics_use_all_channels_and_any_channel_pixel_difference(pair):
    path = pair['right_session'] / pair['right_label'] / 'wrist_rgb.png'
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0, 0] = 255
    Image.fromarray(pixels).save(path)
    report = audit_pair(**pair)
    image = report['current_mounted_rgb']['wrist_rgb']
    assert image['exact_pixels_equal'] is False
    assert image['mae_uint8_rgb'] == pytest.approx(255 / 12)
    assert image['differing_pixel_fraction'] == .25
    assert report['current_rgb_exact_match'] is False
    assert report['native_geometry_match'] is True

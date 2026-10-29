"""Check explicit copied native-scene joint-velocity overrides entirely on CPU.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_initial_joint_velocity.py
"""
from copy import deepcopy
import hashlib
import json

import pytest

from wia.authoring.converters import wia_initial_joint_velocity as adapter


@pytest.fixture
def native_source(tmp_path):
    snapshot = {'objects_info': {'init_info': {
        'cabinet': {'class_name': 'DatasetObject', 'args': {'name': 'cabinet', 'category': 'bottom_cabinet',
            'model': 'fixture', 'scale': [1., 1., 1.], 'fixed_base': True}},
        'untouched': {'class_name': 'DatasetObject', 'args': {'name': 'untouched', 'model': 'original'}}}},
        'state': {'registry': {'object_registry': {
            'cabinet': {'joint_pos': [.2, .3], 'joint_vel': [60.4669303894043, -2.0622285995965598e-17],
                'root_link': {'pos': [1., 2., 3.], 'ori': [0., 0., 0., 1.], 'lin_vel': [0., 0., 0.]},
                'is_asleep': False},
            'untouched': {'joint_pos': [.8], 'joint_vel': [7.], 'custom_state': {'nested': [True, None, 'verbatim']}}},
            'system_registry': {'water': {'particles': [[1., 2., 3.]]}}}},
        'metadata': {'scene_model': 'Beechwood_0_int', 'original': ['preserve', 42]}}
    path = tmp_path / 'Beechwood_0_int_best.json'
    path.write_text(json.dumps(snapshot, indent=2) + '\n')
    parameters = {'revision': adapter.REVISION, 'expected_scene': 'Beechwood_0_int',
        'source_scene_file': str(path), 'source_scene_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'overrides': {'cabinet': [0., 0.]}}
    config = {'scene': {'type': 'InteractiveTraversableScene', 'scene_model': 'Beechwood_0_int',
        'load_room_instances': ['living_room_0']}, 'objects': [{'name': 'authored_item', 'position': [1., 2., 3.]}]}
    return path, snapshot, parameters, config


def refresh_source(fixture):
    path, source, parameters, _ = fixture
    path.write_text(json.dumps(source, indent=2) + '\n')
    parameters['source_scene_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_requested_joint_velocity_arrays_change_and_all_evidence_is_source_exact(native_source):
    path, source, parameters, config = native_source
    before_bytes, before_parameters, before_config = path.read_bytes(), deepcopy(parameters), deepcopy(config)
    report = adapter.configure_scene(config, parameters)
    expected = deepcopy(source)
    expected['state']['registry']['object_registry']['cabinet']['joint_vel'] = [0., 0.]
    assert config['scene']['scene_file'] == expected
    del config['scene']['scene_file']
    assert config == before_config
    assert parameters == before_parameters and path.read_bytes() == before_bytes
    assert report['source_scene_file'] == str(path.resolve())
    assert report['source_scene_sha256'] == hashlib.sha256(before_bytes).hexdigest()
    assert report['overrides']['cabinet']['old_joint_vel'] == [60.4669303894043, -2.0622285995965598e-17]
    assert report['overrides']['cabinet']['new_joint_vel'] == [0., 0.]
    assert report['initialization_only'] is True
    assert report['runtime_setters_used'] is report['native_validation_run'] is report['physical_success_verified'] is False
    assert not hasattr(adapter, 'initialize_scene')
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('existing', ['path', 'dict', 'none'])
def test_matching_existing_source_is_accepted_without_mutating_caller_source_dict(native_source, existing):
    path, source, parameters, config = native_source
    caller_source = deepcopy(source)
    config['scene']['scene_file'] = str(path) if existing == 'path' else caller_source if existing == 'dict' else None
    adapter.configure_scene(config, parameters)
    assert caller_source == source
    assert config['scene']['scene_file'] is not caller_source
    config['scene']['scene_file']['objects_info']['init_info']['cabinet']['args']['model'] = 'changed_copy'
    parameters['overrides']['cabinet'][0] = 99.
    assert caller_source == source and json.loads(path.read_bytes()) == source
    assert config['scene']['scene_file']['state']['registry']['object_registry']['cabinet']['joint_vel'] == [0., 0.]


@pytest.mark.parametrize('failure', ['scene', 'revision', 'hash', 'relative_path', 'conflicting_path', 'conflicting_dict'])
def test_scene_provenance_and_existing_source_conflicts_fail_without_config_changes(native_source, failure):
    path, source, parameters, config = native_source
    if failure == 'scene': config['scene']['scene_model'] = 'another_scene'
    elif failure == 'revision': parameters['revision'] = 'unreviewed_v2'
    elif failure == 'hash': parameters['source_scene_sha256'] = '0' * 64
    elif failure == 'relative_path': parameters['source_scene_file'] = path.name
    elif failure == 'conflicting_path': config['scene']['scene_file'] = str(path.parent / 'other.json')
    else:
        config['scene']['scene_file'] = deepcopy(source)
        config['scene']['scene_file']['metadata']['original'] = ['conflicting']
    before = deepcopy(config)
    with pytest.raises(ValueError):
        adapter.configure_scene(config, parameters)
    assert config == before


@pytest.mark.parametrize('failure', ['missing_init_object', 'missing_state_object', 'missing_joint_vel',
    'missing_registry', 'length', 'new_nan', 'new_inf', 'new_bool', 'new_tuple', 'old_nan', 'old_bool'])
def test_missing_object_or_mismatched_velocity_evidence_is_rejected_without_partial_mutation(native_source, failure):
    _, source, parameters, config = native_source
    state = source['state']['registry']['object_registry']['cabinet']
    if failure == 'missing_init_object': source['objects_info']['init_info'].pop('cabinet')
    elif failure == 'missing_state_object': source['state']['registry']['object_registry'].pop('cabinet')
    elif failure == 'missing_joint_vel': state.pop('joint_vel')
    elif failure == 'missing_registry': source['state'].pop('registry')
    elif failure == 'length': parameters['overrides']['cabinet'] = [0.]
    elif failure == 'new_nan': parameters['overrides']['cabinet'] = [float('nan'), 0.]
    elif failure == 'new_inf': parameters['overrides']['cabinet'] = [0., float('inf')]
    elif failure == 'new_bool': parameters['overrides']['cabinet'] = [False, 0.]
    elif failure == 'new_tuple': parameters['overrides']['cabinet'] = (0., 0.)
    elif failure == 'old_nan': state['joint_vel'][0] = float('nan')
    else: state['joint_vel'][0] = True
    refresh_source(native_source)
    before = deepcopy(config)
    with pytest.raises(ValueError):
        adapter.configure_scene(config, parameters)
    assert config == before


def test_later_invalid_override_does_not_install_an_earlier_partial_change(native_source):
    _, _, parameters, config = native_source
    parameters['overrides']['untouched'] = [0., 0.]
    before = deepcopy(config)
    with pytest.raises(ValueError, match='lengths'):
        adapter.configure_scene(config, parameters)
    assert config == before

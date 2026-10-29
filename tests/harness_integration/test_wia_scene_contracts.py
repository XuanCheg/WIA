"""Validate explicit WIA scene portability with CPU native-contract fixtures.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_scene_contracts.py
No simulator, GPU, render, scene construction or object positioning is invoked.
"""
from copy import deepcopy

import pytest

from wia.authoring.converters import wia_dimensional as dimensional
from wia.authoring.converters import wia_mirror as mirror_adapter
from harness_integration.test_wia_dimensional_adapter import native as native_geometry
from harness_integration.test_wia_mirror_adapter import mirror


SCENE = 'explicit_test_background'
ROOM = 'explicit_test_room_0'


@pytest.fixture(params=[('native_geometry', dimensional, 'wia_dimensional_v3'),
                        ('mirror', mirror_adapter, 'wia_mirror_v2')], ids=['dimensional-v3', 'mirror-v2'])
def portable(request):
    fixture_name, adapter, revision = request.param
    state = request.getfixturevalue(fixture_name)
    state.config['parameters'].update(revision=revision, scene_contract={'scene_model': SCENE, 'room': ROOM})
    state.config.update(scene=SCENE, room=ROOM)
    state.session.scene_name, state.session.room_name = SCENE, ROOM
    state.session.env.scene.scene_model = SCENE
    return state, adapter


def test_explicit_matching_scene_is_accepted_without_changing_native_objects(portable):
    state, adapter = portable
    configuration = deepcopy(state.config)
    positions = {key: obj.position.copy() for key, obj in state.objects.items()}
    assert adapter._config(state.session, state.config) is state.config['parameters']
    assert adapter.qualify_operation(state.session, state.controller, state.config, state.objects)['accepted'] is True
    result = adapter.create_evaluator(state.session, state.controller, state.config, state.output).begin_episode(state.observation)
    assert result['schema'] == configuration['parameters']['revision']
    assert result['evaluation_validity'] == 'complete'
    assert result['task_success'] is False
    assert state.config == configuration
    for key, obj in state.objects.items():
        assert (obj.position == positions[key]).all()
    if adapter is mirror_adapter:
        capture = adapter.capture_initial_geometry(state.session, state.config, state.objects)
        assert capture['available'] is True
        assert capture['optical_selection']['selected_candidate'] == 'candidate_0'
        assert state.material.writes == []


@pytest.mark.parametrize('contract', [None, {}, {'scene_model': SCENE}, {'room': ROOM},
    {'scene_model': '', 'room': ROOM}, {'scene_model': SCENE, 'room': ''},
    {'scene_model': ' ', 'room': ROOM}, {'scene_model': SCENE, 'room': 0},
    {'scene_model': 123, 'room': ROOM}, ['scene_model', 'room']])
def test_new_revisions_require_explicit_nonempty_string_contract(portable, contract):
    state, adapter = portable
    if contract is None:
        state.config['parameters'].pop('scene_contract')
    else:
        state.config['parameters']['scene_contract'] = contract
    with pytest.raises(ValueError, match='scene_contract'):
        adapter._config(state.session, state.config)
    assert adapter.qualify_operation(state.session, state.controller, state.config, state.objects)['accepted'] is False


@pytest.mark.parametrize('identity', ['contract_scene', 'contract_room', 'config_scene', 'config_room',
                                      'session_scene', 'session_room', 'native_scene'])
def test_new_revisions_reject_every_scene_identity_mismatch(portable, identity):
    state, adapter = portable
    if identity.startswith('contract_'):
        key = 'scene_model' if identity == 'contract_scene' else 'room'
        state.config['parameters']['scene_contract'][key] += '_different'
    elif identity.startswith('config_'):
        state.config[identity.removeprefix('config_')] += '_different'
    elif identity.startswith('session_'):
        key = identity.removeprefix('session_') + '_name'
        setattr(state.session, key, getattr(state.session, key) + '_different')
    else:
        state.session.env.scene.scene_model += '_different'
    with pytest.raises(ValueError):
        adapter._config(state.session, state.config)
    assert adapter.qualify_operation(state.session, state.controller, state.config, state.objects)['accepted'] is False
    if adapter is mirror_adapter:
        capture = adapter.capture_initial_geometry(state.session, state.config, state.objects)
        assert capture['available'] is False and capture['errors'][0].startswith('configuration:')
        assert capture['targets'] == capture['fixtures'] == {}
        assert state.material.writes == []


@pytest.mark.parametrize('revision', ['wia_dimensional_v1', 'wia_dimensional_v2', 'wia_mirror_v1'])
@pytest.mark.parametrize('changed', ['scene', 'room'])
def test_legacy_revisions_cannot_use_contract_to_bypass_original_hall_identity(request, revision, changed):
    state = request.getfixturevalue('mirror' if revision == 'wia_mirror_v1' else 'native_geometry')
    adapter = mirror_adapter if revision == 'wia_mirror_v1' else dimensional
    state.config['parameters']['revision'] = revision
    assert adapter._config(state.session, state.config) is state.config['parameters']
    if changed == 'scene':
        state.config['scene'] = state.session.scene_name = state.session.env.scene.scene_model = SCENE
    else:
        state.config['room'] = state.session.room_name = ROOM
    state.config['parameters']['scene_contract'] = {
        'scene_model': state.config['scene'], 'room': state.config['room']}
    with pytest.raises(ValueError):
        adapter._config(state.session, state.config)


@pytest.fixture
def portable_initialization(mirror):
    mirror.init_parameters.update(revision='wia_mirror_v2', scene_contract={'scene_model': SCENE, 'room': ROOM})
    mirror.init_parameters['layout'].update(scene_model=SCENE, rooms=[ROOM])
    mirror.session.scene_name, mirror.session.room_name = SCENE, ROOM
    mirror.session.env.scene.scene_model = SCENE
    mirror.session._category_initialization_active = True
    mirror.material.allow_writes = True
    return mirror


def test_mirror_v2_initialization_delegates_only_matching_authored_layout(portable_initialization, monkeypatch):
    state = portable_initialization
    calls = []
    monkeypatch.setattr(mirror_adapter.active_layout, 'configure_scene', lambda config, p: calls.append(p) or {})
    monkeypatch.setattr(mirror_adapter.active_layout, 'postprocess_joint_world', lambda session, p: calls.append(p) or {})
    monkeypatch.setattr(mirror_adapter.active_layout, 'initialize_scene',
                        lambda session, p: calls.append(p) or deepcopy(session.category_interaction_setup))
    original = deepcopy(state.init_parameters)
    mirror_adapter.configure_scene({'scene': {'scene_model': SCENE}}, state.init_parameters)
    mirror_adapter.postprocess_joint_world(state.session, state.init_parameters)
    receipt = mirror_adapter.initialize_scene(state.session, state.init_parameters)['mirror_material_initialization']
    assert receipt['accepted'] is True and receipt['setter_count'] == 3
    assert calls == [original, original, original]
    assert all(p is state.init_parameters for p in calls)
    assert state.init_parameters == original


@pytest.mark.parametrize('invalid', ['missing_contract', 'contract_scene', 'contract_room', 'empty_scene',
                                     'empty_room', 'layout_scene', 'layout_room', 'extra_room',
                                     'session_scene', 'session_room', 'native_scene'])
def test_mirror_v2_initialization_rejects_identity_errors_before_material_writes(portable_initialization, invalid, monkeypatch):
    state = portable_initialization
    parameters = state.init_parameters
    if invalid == 'missing_contract':
        parameters.pop('scene_contract')
    elif invalid.startswith(('contract_', 'empty_')):
        key = 'scene_model' if invalid.endswith('scene') else 'room'
        parameters['scene_contract'][key] = '' if invalid.startswith('empty_') else 'different'
    elif invalid == 'layout_scene':
        parameters['layout']['scene_model'] = 'different'
    elif invalid in ('layout_room', 'extra_room'):
        parameters['layout']['rooms'] = ['different'] if invalid == 'layout_room' else [ROOM, 'extra']
    elif invalid == 'native_scene':
        state.session.env.scene.scene_model = 'different'
    else:
        setattr(state.session, invalid.removeprefix('session_') + '_name', 'different')

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid identity must not delegate initialization.')

    monkeypatch.setattr(mirror_adapter.active_layout, 'initialize_scene', forbidden)
    with pytest.raises(ValueError):
        mirror_adapter.initialize_scene(state.session, parameters)
    assert state.material.writes == []
    if not invalid.startswith(('session_', 'native_')):
        with pytest.raises(ValueError):
            mirror_adapter._initialization_contract(parameters)
        monkeypatch.setattr(mirror_adapter.active_layout, 'configure_scene', forbidden)
        with pytest.raises(ValueError):
            mirror_adapter.configure_scene({'scene': {'scene_model': SCENE}}, parameters)


def test_mirror_v1_initialization_keeps_original_hall_restriction(portable_initialization):
    state = portable_initialization
    state.init_parameters['revision'] = 'wia_mirror_v1'
    with pytest.raises(ValueError):
        mirror_adapter._initialization_contract(state.init_parameters)
    with pytest.raises(ValueError):
        mirror_adapter.initialize_scene(state.session, state.init_parameters)
    assert state.material.writes == []

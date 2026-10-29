"""Check layout repair boundaries with CPU geometry and native-shaped fixtures.

Run: python -m pytest -q tests/test_esi_interaction_setup.py
These tests do not start a simulator or establish a physical grasp.
"""
import copy
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wia.runtime.scene_capture_queue import CaptureQueue, write_json
from wia.tasks.interaction_setup import (group_translation, height_screen, on_top_task_support,
    prepare_definitions, reference_position, support_contact_paths, touching_task_support)
from wia.tasks.schema import CategoryInteractionPatch


def policy(**overrides):
    return {'mode': 'manipulation', 'group_bindings': ['target', 'reference'],
        'manipulation_bindings': ['target'], 'relocation_policy': 'rigid_group',
        'source_region': 'room_0', 'minimum_grasp_height_m': .35,
        'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85, **overrides}


def test_height_screen_uses_native_floor_frame_and_does_not_certify_grasp():
    bounds = {'low': ([-.1, -.1, -1.], [.1, .1, -.9]),
              'table': ([0., 0., -.5], [.1, .1, -.4]),
              'high': ([0., 0., .1], [.1, .1, .3])}
    rows = height_screen(bounds, -1.2)
    assert not rows['low']['within_height_window']
    assert rows['table']['within_height_window']
    assert not rows['high']['within_height_window']
    assert not any(row['height_is_grasp_certificate'] for row in rows.values())


def test_group_placement_preserves_relative_geometry_and_rejects_small_support():
    bounds = {'a': ([-.4, -.1, .02], [-.2, .1, .12]),
              'b': ([.2, -.1, .02], [.4, .1, .22])}
    delta = group_translation(bounds, ([-1., -1., 0.], [1., 1., .7]), [.2, .3], .7)
    assert delta == pytest.approx([.2, .3, .685])
    assert (np.asarray(bounds['b'][0]) + delta) - (np.asarray(bounds['a'][0]) + delta) == pytest.approx([.6, 0., 0.])
    with pytest.raises(ValueError, match='does not fit'):
        group_translation(bounds, ([-.2, -.2, 0.], [.2, .2, .7]), [0., 0.], .7)


def test_declarative_policy_preserves_original_variant_and_public_private_goal():
    definition = {'scene': {'variants': {'original': {'patches': []}, 'adapted': {
        'patches': [{'operation': 'restore_robot_head_mount', 'phase': 'after_postprocessing'}]}}}}
    scene = copy.deepcopy(definition)
    task = {**copy.deepcopy(definition), 'kind': 'category_robot_task',
        'public': {'objective': 'Grasp the larger object.'}, 'private': {
            'bindings': {'target': {'name': 'original_target'}, 'reference': {'name': 'original_reference'}},
            'evaluation': {'type': 'grasp_lift', 'target': 'target', 'minimum_lift_m': .05}}}
    original_public, original_private = copy.deepcopy(task['public']), copy.deepcopy(task['private'])
    prepare_definitions(None, scene, task, policy())
    assert scene['scene']['variants']['original']['patches'] == []
    patches = task['scene']['variants']['adapted']['patches']
    assert [p['operation'] for p in patches] == ['restore_robot_head_mount', 'prepare_category_interaction']
    assert task['public'] == original_public and task['private'] == original_private
    assert patches[-1]['group_bindings'] == ['target', 'reference']


def test_navigation_relocation_and_unbound_group_are_rejected():
    with pytest.raises(ValueError, match='physical task group'):
        CategoryInteractionPatch.model_validate({'operation': 'prepare_category_interaction',
            **policy(mode='navigation')})
    task = {'kind': 'category_robot_task', 'private': {'bindings': {}},
        'scene': {'variants': {'original': {'patches': []}}}}
    with pytest.raises(ValueError, match='existing private binding'):
        prepare_definitions(None, None, task, policy())


@pytest.fixture
def native_support(monkeypatch):
    states = ModuleType('omnigibson.object_states')
    states.Touching, states.OnTop = type('Touching', (), {}), type('OnTop', (), {})
    parent = ModuleType('omnigibson')
    monkeypatch.setitem(sys.modules, 'omnigibson', parent)
    monkeypatch.setitem(sys.modules, 'omnigibson.object_states', states)
    original = SimpleNamespace(name='floor', prim_path='/World/floor')
    table = SimpleNamespace(name='table', prim_path='/World/table')
    touching = SimpleNamespace(get_value=lambda other: other is table)
    on_top = SimpleNamespace(get_value=lambda other: other is table)
    contact = SimpleNamespace(body0='/World/target/base_link', body1='/World/table/top',
        normal=[0., 0., 1.], position=[0., 0., .7])
    obj = SimpleNamespace(name='target', states={states.Touching: touching, states.OnTop: on_top},
        contact_list=lambda: [contact])
    evidence = {'accepted': True, 'relocated': True, 'object_names': ['target'],
        'group_translation_m': [1., 2., .7], 'support_prim_path': table.prim_path,
        'support_surface_z_m': .7}
    session = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(objects=[original, table])),
        category_interaction_setup=evidence)
    return session, obj, original, contact


def test_reference_translation_requires_verified_group_membership(native_support):
    session, _, _, _ = native_support
    assert reference_position(session, 'target', [2., 3., 0.]) == pytest.approx([3., 5., .7])
    assert reference_position(session, 'other', [2., 3., 0.]) == pytest.approx([2., 3., 0.])
    session.category_interaction_setup['accepted'] = False
    assert support_contact_paths(session, 'target') == ()
    assert reference_position(session, 'target', [2., 3., 0.]) == pytest.approx([2., 3., 0.])


def test_native_task_support_is_measured_without_claiming_floor_contact(native_support):
    session, obj, original, contact = native_support
    assert touching_task_support(session, obj, original)
    assert on_top_task_support(session, obj, original)
    contact.normal = [1., 0., 0.]
    assert not touching_task_support(session, obj, original)
    contact.normal = [0., 0., 1.]
    contact.position = [0., 0., .9]
    assert not touching_task_support(session, obj, original)
    session.category_interaction_setup['accepted'] = False
    assert not touching_task_support(session, obj, original)
    assert not on_top_task_support(session, obj, original)


def test_user_pause_prevents_new_claim_without_mutating_queued_input(tmp_path):
    queue = CaptureQueue(tmp_path / 'queue')
    job_id = 'capture_' + 'a' * 32
    token = queue.root / 'queued' / (job_id + '.json')
    write_json(token, {'job_id': job_id})
    write_json(queue.root / 'capture_pause.json', {'state': 'PAUSED_BY_USER'})
    before = token.read_bytes()
    assert queue.claim({'id': 'paused_fixture'}) is None
    assert token.read_bytes() == before
    assert not list((queue.root / 'claimed').iterdir())

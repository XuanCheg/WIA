"""CPU checks for structural region initialization and episode visual qualification.

Run: python -B -m pytest -q -p no:cacheprovider
 tests/test_task_edit2_region_stages.py tests/test_task_edit2_missing_definitions.py.
The original source contract drives native-shaped fixtures; no GPU is launched.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wia.authoring.conversion import ConversionContext
from wia.authoring.converters import enumerative_perception_spatial_segmentation as regions
from harness_integration.test_task_edit2_missing_definitions import source, anchor_source_states

VISUAL_GATES = {'current_episode_visual_receipt', 'public_initial_global_labels_visually_distinguishable'}


@pytest.fixture
def world(tmp_path):
    identity = 'Enumerative Perception/Spatial Segmentation/house_single_floor/dining_room_0/q_002.json'
    original = source(identity)
    result = regions.CategoryConverter().convert(original, ConversionContext(tmp_path, seed=0,
        config={'task_edit2_global_region_labels': True,
            'region_anchor_source_states': anchor_source_states(original.raw['scene'])}))
    config = deepcopy(result['task_definition']['private']['evaluation'])
    bindings = result['task_definition']['private']['bindings']
    config['bindings'] = {key: value['name'] for key, value in bindings.items()}
    contract = config['parameters']['region_label_contract']
    objects = {}
    for key, expected in contract['source_expected'].items():
        name = bindings[key]['name']
        state = deepcopy(contract['source_anchor_states'].get(name))
        obj = SimpleNamespace(name=name, category=bindings[key]['category'], aabb=np.asarray(expected['bbox']),
            source_pose=state)
        obj.get_position_orientation = lambda obj=obj: (obj.source_pose['pos'], obj.source_pose['ori'])
        objects[name] = obj
    session = SimpleNamespace(scene_name=original.raw['scene'],
        env=SimpleNamespace(scene=SimpleNamespace(object_registry=lambda field, name: objects.get(name))))
    return SimpleNamespace(session=session, config=config, contract=contract, objects=objects,
        bindings=bindings, identity=identity)


def complete_receipt(world):
    world.session.episode_id = 'episode'
    world.session.region_label_qualification = {world.identity: {'episode_id': 'episode',
        'frame': 'initial_global_world_xy', 'anchor_public_label_associations_verified': True}}


def test_initialization_accepts_only_structural_geometry_and_keeps_visibility_pending(world):
    before = deepcopy(world.config)
    result = regions.validate_region_labels(world.session, world.config, stage='initialization')
    assert result['accepted'] and all(result['checks'].values())
    assert result['checks']['current_native_label_mapping_matches']
    assert any(key.startswith('source_region_membership_preserved:') for key in result['checks'])
    assert VISUAL_GATES.isdisjoint(result['checks'])
    assert set(result['pending_qualification']) == VISUAL_GATES
    assert not result['qualification_complete'] and result['evaluation_validity'] == 'unavailable'
    assert world.config == before and not hasattr(world.session, 'episode_id')


def test_default_evaluation_still_requires_exact_current_visual_receipt(world):
    result = regions.validate_region_labels(world.session, world.config)
    assert result == regions.validate_region_labels(world.session, world.config, stage='evaluation')
    assert not result['accepted'] and not result['qualification_complete']
    assert result['evaluation_validity'] == 'unavailable'
    assert set(result['pending_qualification']) == VISUAL_GATES
    assert all(result['checks'][name] is False for name in VISUAL_GATES)


def test_full_visual_receipt_retains_evaluation_behavior(world):
    complete_receipt(world)
    result = regions.validate_region_labels(world.session, world.config)
    assert result['accepted'] and result['qualification_complete']
    assert result['evaluation_validity'] == 'complete' and result['pending_qualification'] == []
    initialization = regions.validate_region_labels(world.session, world.config, stage='initialization')
    assert initialization['accepted'] and not initialization['qualification_complete']
    assert set(initialization['pending_qualification']) == VISUAL_GATES


@pytest.mark.parametrize('failure', ['scene', 'anchor_category', 'anchor_position', 'anchor_orientation',
    'anchor_aabb', 'target_region', 'runtime_binding', 'label_mapping', 'missing_anchor', 'missing_target'])
def test_structural_failures_are_not_waived_in_either_stage(world, failure):
    complete_receipt(world)
    anchor = next(iter(world.contract['anchor_bindings']))
    target = world.contract['source_pairs'][0]['target']
    anchor_obj = world.objects[world.bindings[anchor]['name']]
    target_obj = world.objects[world.bindings[target]['name']]
    if failure == 'scene':
        world.session.scene_name = 'other_scene'
    elif failure == 'anchor_category':
        anchor_obj.category = 'other_category'
    elif failure == 'anchor_position':
        anchor_obj.source_pose['pos'][0] += .5
    elif failure == 'anchor_orientation':
        anchor_obj.source_pose['ori'] = [0., 0., 1., 0.]
    elif failure == 'anchor_aabb':
        anchor_obj.aabb += np.array([.5, 0., 0.])
    elif failure == 'target_region':
        target_obj.aabb += np.array([.5, 0., 0.])
    elif failure == 'runtime_binding':
        world.config['bindings'][anchor] = 'other_anchor'
    elif failure == 'label_mapping':
        world.contract['anchor_public_labels'][anchor] = 'wrong geometric label'
    elif failure == 'missing_anchor':
        world.objects.pop(anchor_obj.name)
    else:
        world.objects.pop(target_obj.name)
    for stage in ('initialization', 'evaluation'):
        result = regions.validate_region_labels(world.session, world.config, stage=stage)
        assert not result['accepted'] and not result['qualification_complete'], result
        assert result['evaluation_validity'] == 'unavailable'


@pytest.mark.parametrize('key,value', [('episode_id', 'old_episode'), ('frame', 'other_frame'),
    ('anchor_public_label_associations_verified', False)])
def test_current_visual_evidence_is_only_pending_at_initialization(world, key, value):
    complete_receipt(world)
    world.session.region_label_qualification[world.identity][key] = value
    assert regions.validate_region_labels(world.session, world.config, stage='initialization')['accepted']
    result = regions.validate_region_labels(world.session, world.config)
    assert not result['accepted'] and not result['qualification_complete']
    assert result['pending_qualification']


def test_initialization_never_reads_episode_visual_receipt(world):
    class NoReceiptAccess(SimpleNamespace):
        @property
        def region_label_qualification(self):
            raise AssertionError('Episode evidence is not an initialization condition.')
    session = NoReceiptAccess(**vars(world.session))
    assert regions.validate_region_labels(session, world.config, stage='initialization')['accepted']
    evaluation = regions.validate_region_labels(session, world.config)
    assert not evaluation['accepted'] and evaluation['evaluation_validity'] == 'unavailable'


def test_unknown_stage_raises_and_nonapplicable_result_is_unchanged(world):
    for config in (world.config, {}):
        with pytest.raises(ValueError, match='stage'):
            regions.validate_region_labels(world.session, config, stage='structural')
    for stage in ('initialization', 'evaluation'):
        assert regions.validate_region_labels(world.session, {}, stage=stage) == {'applicable': False, 'accepted': True}

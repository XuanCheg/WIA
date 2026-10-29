"""Check source-aware task grouping and feedback repair without native startup.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
    tests/test_composition_manipulation.py
"""
import copy
import json
from pathlib import Path

import pytest

from wia.authoring.converters import composition_manipulation as policy


def liquid(cid, operation):
    raw = {'scene': 'Rs_int', 'room': 'living_room_0', 'runner_task': 'pour',
        'metadata_json': {'obj1': {'side': 'left', 'category': 'wineglass', 'model': 'euzudc'},
            'obj2': {'side': 'right', 'category': 'mug', 'model': 'ppzttc'},
            'left_category': 'wrong_alias', 'left_model': 'unused_alias',
            'fixed_env': {'scene': 'Rs_int'}, 'run_idx': 102,
            'capacity_diff_measured': 20, 'final_particles_obj1': 30}}
    record = {'candidate_id': cid, 'status': 'generated', 'big_task': 'Physical Structure',
        'small_task': 'Liquid Volume', 'scene_model': raw['scene'], 'room': raw['room'],
        'seed': 0, 'composition_source_raw': raw}
    bindings = {'left': {'name': 'obj1', 'category': 'wineglass', 'model': 'euzudc'},
        'right': {'name': 'obj2', 'category': 'mug', 'model': 'ppzttc'},
        'floor': {'name': 'floors_ptwlei_0'}}
    world = {'variants': {'original': {'patches': []}, 'adapted': {'patches': [
        {'operation': 'robot_pose', 'position': [.34, -1.3, 0.], 'yaw_deg': 96.58}]}}}
    scene = {'kind': 'scene_patch', 'id': cid, 'source': {'json': cid + '.json'},
        'seed': 0, 'bindings': bindings, 'scene': world}
    task = {'source': scene['source'], 'scene': copy.deepcopy(world), 'private': {
        'bindings': copy.deepcopy(bindings), 'evaluation': {'type': 'category_operation',
            'module': policy.LIQUID_MODULE, 'operation': operation,
            'binding_keys': ['left', 'right', 'floor'], 'parameters': {'stability_seconds': .5}}}}
    return record, task, scene


def groups(*rows):
    return policy.plan_groups([r for r, _, _ in rows],
        {r['candidate_id']: t for r, t, _ in rows}, {r['candidate_id']: s for r, _, s in rows})


def test_same_bootstrap_different_source_histories_remain_isolated():
    short = liquid('short', policy.LIQUID_OPERATIONS[0])
    long = liquid('return', policy.LIQUID_OPERATIONS[1])
    long[0]['composition_source_raw']['metadata_json'].update(run_idx=87,
        capacity_diff_measured=48, final_particles_obj1=22)
    original = copy.deepcopy((short, long))
    result = groups(short, long)
    # Matching container models and room overlays do not establish one complete
    # native world: the source runs, capacity measurements and final contents
    # differ. Keep both historical definitions intact without pooling routes.
    assert (short, long) == original
    assert len(result) == 2
    assert {row['representative_id']: row['member_ids'] for row in result} == {
        'return': ['return'], 'short': ['short']}
    assert short[1]['source'] != long[1]['source']
    assert short[1]['private']['evaluation']['operation'] == 'dock_at_both_containers'
    assert long[1]['private']['evaluation']['operation'] == 'dock_at_both_and_return'


@pytest.mark.parametrize('change', ['model', 'scale', 'start', 'seed', 'source_seed',
    'floor', 'parameters', 'fixed_env', 'task_scene', 'missing_source'])
def test_different_bootstrap_or_protocol_never_groups(change):
    short, long = liquid('short', policy.LIQUID_OPERATIONS[0]), liquid('return', policy.LIQUID_OPERATIONS[1])
    r, t, s = long
    if change == 'model':
        r['composition_source_raw']['metadata_json']['obj1']['model'] = 'another_model'
    elif change == 'scale':
        r['composition_source_raw']['metadata_json']['obj1']['scale'] = [.8, .8, .8]
    elif change == 'start':
        s['scene']['variants']['adapted']['patches'][0]['position'][0] += .1
        t['scene'] = copy.deepcopy(s['scene'])
    elif change == 'seed':
        r['seed'] = s['seed'] = 2
    elif change == 'source_seed':
        r['composition_source_raw']['metadata_json']['seed'] = 3
    elif change == 'floor':
        s['bindings']['floor']['name'] = 'other_floor'
        t['private']['bindings'] = copy.deepcopy(s['bindings'])
    elif change == 'parameters':
        t['private']['evaluation']['parameters']['stability_seconds'] = .6
    elif change == 'fixed_env':
        r['composition_source_raw']['metadata_json']['fixed_env']['scene'] = 'other'
    elif change == 'task_scene':
        t['scene']['variants']['adapted']['patches'] = []
    else:
        del r['composition_source_raw']
    assert len(groups(short, long)) == 2


def test_repeats_rigid_and_same_operation_duplicates_are_not_merged():
    a, b = liquid('a', policy.LIQUID_OPERATIONS[0]), liquid('b', policy.LIQUID_OPERATIONS[0])
    assert len(groups(a, b)) == 2
    b[0]['status'] = 'repeat'
    assert len(groups(a, b)) == 1
    b[0].update(status='generated', small_task='Rigid Containment')
    assert len(groups(a, b)) == 1


def test_action_repair_preserves_real_source_scales_bindings_and_evaluation():
    project = __import__('wia.resources', fromlist=['ROOT']).ROOT
    source = project / 'ESI/dataset/json_clean/Action Sequencing/Action Order Inference/office_large/lobby_0/office_large__lobby_0.json'
    raw = json.loads(source.read_text())
    metadata = json.loads(raw['metadata_json'])
    bases = [x for x in metadata['objects'] if x.get('cat') == 'chopping_block']
    assert len(bases) == 2 and bases[0]['scale'] != bases[1]['scale']
    assert all(x['model'] == 'ekowsj' for x in bases)
    patch = {'operation': 'category_initialization', 'module': 'wia.authoring.converters.action_order',
        'parameters': {'operation': 'measured_rigid_row_startup_v2', 'bindings': bases}}
    world = {'variants': {'adapted': {'patches': [patch]}}}
    task = {'scene': copy.deepcopy(world), 'public': {'constraints': []},
        'private': {'evaluation': {'operation': 'ordered_grasp_lift_stack'}, 'bindings': bases}}
    scene = {'scene': copy.deepcopy(world), 'bindings': copy.deepcopy(bases)}
    record = {'candidate_id': 'feedback_sample', 'small_task': 'Action Order Inference'}
    original_task, original_scene = copy.deepcopy(task), copy.deepcopy(scene)
    feedback = {'entries': {'feedback_sample': {'decision': 'needs_fix'}}}
    revised = policy.repair_definitions(record, task, scene, feedback=feedback)
    assert revised['changed'] and revised['task']['scene'] == revised['scene']['scene']
    assert task == original_task and scene == original_scene
    assert revised['task']['private'] == original_task['private']
    contract = revised['scene']['scene']['variants']['adapted']['patches'][0]['parameters']['initialization_review_contract']
    assert contract['minimum_independent_candidate_xy_gap_m'] == .03
    assert contract['unobstructed_candidate_inspection_requires_review']
    assert not policy.repair_definitions(record, revised['task'], revised['scene'], feedback=feedback)['changed']
    assert not policy.repair_definitions(record, task, scene,
        feedback={'entries': {'feedback_sample': {'decision': 'approved'}}})['changed']


def test_action_feedback_receipt_explains_hidden_and_remote_chopping_blocks():
    receipt = Path('/home/zhouxinyang/esi_category_conversion/20261014_v2/capture_artifacts/jobs/'
        'capture_00a901a1e5d885b9623714eea9589b54/attempts/01/preview_complete.json')
    if not receipt.is_file():
        pytest.skip('The original production feedback receipt is not installed.')
    evidence = json.loads(receipt.read_text())
    setup = evidence['interaction_setup']
    geometry = setup['adapted_geometry']
    middle, hidden, remote = (geometry[k] for k in ('obj_fixed', 'obj_A_wrong', 'obj_A_correct'))
    # Real accepted initialization can still hide a thin candidate under the
    # middle object. Separate group proposal acceptance was not a final proof.
    assert setup['accepted'] and not evidence['native_qualification_run']
    assert all(min(middle['aabb_max_m'][i], hidden['aabb_max_m'][i]) >
        max(middle['aabb_min_m'][i], hidden['aabb_min_m'][i]) for i in (0, 1, 2))
    assert abs(remote['position_m'][1] - middle['position_m'][1]) > 3.5
    assert setup['support_prim_paths']['obj_A_correct'] != setup['support_prim_paths']['obj_fixed']

"""Check shared-support packing and physics-bound interaction evidence.

Run: python -m pytest -q tests/test_task_edit2_tabletop.py
These CPU measurements do not establish native support or grasp feasibility.
"""
from types import SimpleNamespace
import numpy as np

from wia.tasks.shared_tabletop import tabletop_dimensions, tabletop_layouts
from wia.runtime.sim.esi_interaction_history import public_action_begin, public_action_end, rearrangement_evidence


def test_repacking_changes_source_spacing_without_asset_resize():
    group = {f'container_{i}': SimpleNamespace(name=f'object_{i}',
        get_position_orientation=lambda i=i: (np.array([i * 10., 0., .1]), [0., 0., 0., 1.])) for i in range(6)}
    bounds = {obj.name: (np.array([i * 10. - .1, -.1, 0.]),
                         np.array([i * 10. + .1, .1, .2])) for i, obj in enumerate(group.values())}
    assert tabletop_dimensions(bounds)
    layout = tabletop_layouts(group, bounds, ([-1., -.7, .6], [1., .7, .65]), .65,
                               {'relocation_policy': 'shared_tabletop'})[0]
    boxes = list(layout['bounds'].values())
    for original, revised in zip(bounds.values(), boxes):
        np.testing.assert_allclose(original[1] - original[0], revised[1] - revised[0])
        assert abs(revised[0][2] - .655) < 1e-8
        assert np.all(revised[0][:2] >= [-1., -.7]) and np.all(revised[1][:2] <= [1., .7])
    for i, (a, b) in enumerate(boxes):
        for c, d in boxes[i + 1:]:
            assert np.linalg.norm(np.maximum(np.maximum(a - d, c - b)[:2], 0.)) >= .049


def test_no_packing_claim_when_support_is_too_small():
    obj = SimpleNamespace(name='large', get_position_orientation=lambda: (np.zeros(3), [0., 0., 0., 1.]))
    assert tabletop_layouts({'large': obj}, {'large': (np.zeros(3), np.ones(3))},
                            ([0., 0., 0.], [.5, .5, .5]), .5,
                            {'relocation_policy': 'shared_tabletop'}) == []


def test_only_public_active_control_and_native_contact_attribute_objects():
    robot = SimpleNamespace(prim_path='/robot', contact_list=lambda: [
        SimpleNamespace(body0='/robot/hand', body1='/target/body')])
    target = SimpleNamespace(name='target', prim_path='/target')
    sim = SimpleNamespace(current_time=.125)
    controller = SimpleNamespace(session=SimpleNamespace(robot=robot, env=SimpleNamespace(
        scene=SimpleNamespace(objects=[robot, target])), pipeline=SimpleNamespace(og=SimpleNamespace(sim=sim))),
        debug_gripper_state=lambda: {'holding': True, 'constraint_active': True, 'object_name': 'target'})
    before = rearrangement_evidence(controller)
    assert before['sample_id'] == 125000000 and before['controlled_object_names'] == []
    public_action_begin(controller, 'MV_UP')
    during = rearrangement_evidence(controller)
    assert during['control_sequence'] == 1 and during['controlled_object_names'] == ['target']
    assert during['sample_id'] == before['sample_id']
    public_action_end(controller)
    assert rearrangement_evidence(controller)['controlled_object_names'] == []

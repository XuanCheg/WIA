"""Initialize a declared controller posture before placing interaction objects.

Run through ESISession.reset with an explicit TaskEdit2 interaction patch.
This native reset operation never runs after episode observations or control.
Public actions remain governed by the unchanged native controller.
"""
from __future__ import annotations

import numpy as np


def initialize_controller_posture(session, adapter):
    patches = getattr(adapter, '_realized', ())
    declared = [item['controller_initial_trunk_joint_positions_rad'] for item in patches
        if item.get('operation') == 'prepare_category_interaction'
        and item.get('controller_initial_trunk_joint_positions_rad') is not None]
    if not declared:
        return None
    if any(not np.allclose(value, declared[0], atol=0, rtol=0) for value in declared[1:]):
        raise ValueError('Interaction groups require one initial controller posture.')
    if getattr(session, '_observation_source', None) is not None or getattr(session, 'episode_id', None):
        raise RuntimeError('Controller posture reset is restricted to native construction.')
    from wia.tasks.interaction_setup import array
    import torch
    robot = session.env.robots[0]
    controller = robot.controllers['trunk']
    if type(controller).__name__ != 'JointController' or controller.motor_type != 'position' or controller.use_delta_commands:
        raise ValueError('Initial posture requires the native absolute joint-position trunk controller.')
    indices = array(controller.dof_idx).astype(int)
    desired = np.asarray(declared[0], dtype=float)
    lower, upper = [array(value)[indices] for value in robot.control_limits['position']]
    if indices.shape != (4,) or desired.shape != (4,) or not np.isfinite(desired).all() or np.any(desired <= lower) or np.any(desired >= upper):
        raise ValueError('Declared initial posture must lie inside the measured native joint limits.')
    before = array(robot.get_joint_positions()).copy()
    body_pose = [array(value).copy() for value in robot.get_position_orientation()]
    requested = before.copy()
    requested[indices] = desired
    # Use the installed SDK's reset interface before task settling/placement.
    # Starting the real controller later must not sweep an arm through already
    # placed comparison references or other task apparatus.
    robot.reset_joint_pos = torch.tensor(requested, dtype=torch.float32)
    robot.reset()
    after = array(robot.get_joint_positions())
    unchanged = np.ones(len(before), dtype=bool)
    unchanged[indices] = False
    pose_after = [array(value) for value in robot.get_position_orientation()]
    accepted = (np.allclose(after[indices], desired, atol=1e-5, rtol=0)
        and np.allclose(after[unchanged], before[unchanged], atol=1e-5, rtol=0)
        and all(np.allclose(a, b, atol=1e-5, rtol=0) for a, b in zip(body_pose, pose_after)))
    receipt = {'schema': 'initial_controller_posture_v1', 'accepted': bool(accepted),
        'operation': 'native_initialization_reset', 'episode_active': False,
        'dof_indices': indices.tolist(), 'requested_trunk_joints_rad': desired.tolist(),
        'trunk_joints_before_rad': before[indices].tolist(), 'trunk_joints_after_rad': after[indices].tolist(),
        'other_joints_and_root_pose_preserved': bool(accepted),
        'source': 'explicit_interaction_patch_matching_controller_configuration'}
    session.initial_controller_posture = receipt
    if not accepted:
        raise ValueError('Native controller initial-posture reset did not preserve the declared state.')
    return receipt

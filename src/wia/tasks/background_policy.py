"""Protect task dependencies while allowing legal movable-background handling.

Use through DeclarativeScene.create_evaluator; CPU checks: pytest tests/test_task_repair.py.
This policy never moves objects, changes fixed_base, or relaxes chassis collisions.
 Run: python -m pytest tests/test_extraction.py. """
from __future__ import annotations

import numpy as np

FURNITURE = ('chair', 'table', 'sofa', 'couch', 'stool', 'bench', 'cabinet', 'desk',
             'dresser', 'shelf', 'shelving', 'ottoman', 'nightstand', 'wardrobe')


def _array(value):
    return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value, dtype=float)


def _under(body, root):
    return str(body) == root or str(body).startswith(root + '/')


def install_background_policy(session, task):
    if getattr(task.definition, 'kind', None) != 'category_robot_task':
        return None
    values = session.env.scene.objects
    objects = list(values.values()) if isinstance(values, dict) else list(values)
    definition = task.definition.model_dump(mode='json')
    # Exact native names anywhere in task parameters include unbound reference
    # furniture, occluders and source construction dependencies.
    strings = set()
    def collect(value):
        if isinstance(value, str):
            strings.add(value)
        elif isinstance(value, dict):
            for key, item in value.items():
                strings.add(str(key))
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
    collect(definition['private'])
    collect(definition['scene'])
    protected = {obj.name for obj in objects if obj.name in strings or str(obj.category) in strings}
    # A native support carrying a critical dependency is critical too. Failure
    # to measure contact prevents authorizing background movement for this run.
    unresolved = []
    checked = set()
    while protected - checked:
        name = next(iter(protected - checked))
        checked.add(name)
        obj = next(obj for obj in objects if obj.name == name)
        try:
            low = _array(obj.aabb[0])
            for contact in obj.contact_list():
                for other in objects:
                    if other.name == obj.name:
                        continue
                    if any(_under(getattr(contact, side), other.prim_path) for side in ('body0', 'body1')):
                        if float(_array(other.aabb[1])[2]) <= float(low[2]) + .05:
                            protected.add(other.name)
        except (AttributeError, RuntimeError, ValueError, TypeError) as error:
            unresolved.append({'name': obj.name, 'reason': str(error)})
    allowed = {} if unresolved else {obj.name: obj for obj in objects
        if obj.name not in protected and not obj.fixed_base and not obj.visual_only
        and any(word in str(obj.category).lower() for word in FURNITURE)}
    policy = {'protected': sorted(protected), 'allowed': allowed,
              'initial_positions': {name: _array(obj.get_position_orientation()[0]).tolist()
                                    for name, obj in allowed.items()},
              'initial_orientations': {name: _array(obj.get_position_orientation()[1]).tolist()
                                       for name, obj in allowed.items()},
              'movements': [], 'measurement_errors': unresolved, 'map_generation': 0}
    session.background_movement_policy = policy
    return policy


def legal_background_contact(session, body0, body1):
    policy = getattr(session, 'background_movement_policy', None)
    if not policy or getattr(getattr(session, 'temporal_phase_controller', None), 'phase', None) == 'phase_1':
        return False
    robot = session.robot
    arms = set()
    for attribute in ('arm_link_names', 'finger_link_names'):
        value = getattr(robot, attribute, {})
        for names in value.values() if isinstance(value, dict) else [value]:
            arms.update(names)
    roots = [robot.links[name].prim_path for name in arms if name in robot.links]
    for hand, foreign in ((body0, body1), (body1, body0)):
        if any(_under(hand, root) for root in roots):
            return any(_under(foreign, obj.prim_path) for obj in policy['allowed'].values())
    return False


def record_background_movements(session):
    policy = getattr(session, 'background_movement_policy', None)
    if not policy:
        return []
    changed = []
    for name, obj in policy['allowed'].items():
        position, orientation = obj.get_position_orientation()
        point, quaternion = _array(position).tolist(), _array(orientation).tolist()
        old_orientation = policy['initial_orientations'][name]
        rotated = min(np.linalg.norm(np.array(quaternion) - old_orientation),
                      np.linalg.norm(np.array(quaternion) + old_orientation)) > .01
        previous = policy['initial_positions'][name]
        if np.linalg.norm(np.array(point) - previous) > .01 or rotated:
            changed.append({'name': name, 'from_position_m': previous, 'to_position_m': point,
                            'from_quaternion_xyzw': old_orientation, 'to_quaternion_xyzw': quaternion,
                            'evidence': 'native_pose_change', 'legal_action_certified': False})
            policy['initial_positions'][name] = point
            policy['initial_orientations'][name] = quaternion
    if changed:
        policy['movements'].extend(changed)
        policy['map_generation'] += 1
        # Original static maps remain historical proposals, never current route proof.
        session.background_static_map_stale = True
        for key in ('_navigation_cache', '_collision_cache', '_path_cache'):
            cache = getattr(session, key, None)
            if hasattr(cache, 'clear'):
                cache.clear()
    return list(policy['movements'])


def legal_background_grasp(session, object_name):
    """Allow task evaluators to recognize an authorized background-object grasp."""
    policy = getattr(session, 'background_movement_policy', None)
    if not policy or getattr(getattr(session, 'temporal_phase_controller', None), 'phase', None) == 'phase_1':
        return False
    return object_name in policy['allowed']

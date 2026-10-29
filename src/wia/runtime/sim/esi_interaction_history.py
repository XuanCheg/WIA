"""Measure public-action, native grasp and contact evidence without scene writes.

Run: python -m pytest -q tests/test_task_edit2_interaction_history.py
The executor owns action boundaries; evaluators sample native state during control.
"""
from __future__ import annotations


def public_action_begin(controller, action):
    controller._public_action_sequence = getattr(controller, '_public_action_sequence', 0) + 1
    controller._public_action_context = {'sequence': controller._public_action_sequence,
        'action': action, 'active': True}


def public_action_end(controller):
    context = getattr(controller, '_public_action_context', None)
    if context is not None:
        context['active'] = False


def rearrangement_evidence(controller):
    """Return measured robot interaction names at this physics sample.

    This records causal prerequisites rather than claiming causality from motion.
    The evaluator must additionally check displacement and temporal continuity.
    """
    session = controller.session
    context = getattr(controller, '_public_action_context', {})
    try:
        objects = session.env.scene.objects
        objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
        robot = session.robot
        roots = {str(obj.prim_path): obj.name for obj in objects if obj is not robot}
        contacts = set()
        for contact in robot.contact_list():
            for body in (str(contact.body0), str(contact.body1)):
                for root, name in roots.items():
                    if body == root or body.startswith(root + '/'):
                        contacts.add(name)
        grasp = controller.debug_gripper_state()
        held = ([grasp['object_name']] if grasp.get('holding') is True
                and grasp.get('constraint_active') is True and grasp.get('object_name') else [])
        # Simulator time, rather than callback count, distinguishes native samples.
        native_time = float(session.pipeline.og.sim.current_time)
        return {'available': True, 'sample_id': int(round(native_time * 1_000_000_000)),
            'control_sequence': context.get('sequence', 0),
            'control_active': context.get('active', False),
            'controlled_object_names': sorted(set(held) | contacts) if context.get('active') else [],
            'grasped_object_names': held, 'contact_object_names': sorted(contacts),
            'state_setter_used': bool(getattr(session, '_episode_state_setter_used', False))}
    except Exception as error:
        return {'available': False, 'reason': f'{type(error).__name__}: {error}'}

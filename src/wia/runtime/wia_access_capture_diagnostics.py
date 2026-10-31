"""Export opt-in private native inventory after initial capture pixels freeze.

Run CPU checks: PYTHONPATH=src:../Show-Harness-Explore python -B -m pytest \
  -q -p no:cacheprovider tests/test_wia_access_capture_diagnostics.py
The caller owns image freezing, task dispatch and the final snapshot clock
check. This helper does not render, step, execute controls, run evaluators,
change the scene or robot, write files, or add public observations or prompts.
Exported measurements are diagnostic evidence, never native qualification or
a grasp or physical-success certificate.
"""
from __future__ import annotations

from copy import deepcopy

from wia.runtime.wia_capture_diagnostics import ACCESS_REPAIR_MARKER, _native_time


SCHEMA = 'wia_initial_access_review_diagnostics_v1'


def capture_initial_access_diagnostics(session):
    """Return the existing native export unchanged, with private error metadata."""
    task = session.loaded_task
    definition = task.definition
    marker = definition.repair[ACCESS_REPAIR_MARKER]
    evaluation = definition.private.evaluation
    alias = marker.get('alias')
    result = {
        'schema': SCHEMA, 'private_only': True, 'available': False,
        'task_id': definition.id, 'task_sha256': task.file_sha256,
        'module': evaluation.module, 'repair_marker': ACCESS_REPAIR_MARKER,
        'revision': marker.get('revision'), 'alias': alias,
        'inventory': None, 'native_qualification_run': False,
        'physical_success_verified': False,
        'native_time_before_s': None, 'native_time_after_s': None,
        'native_time_verified_unchanged': False,
        'robot_actions': None, 'physics_steps': None,
        'validation_scope': 'Current native geometry only; no qualification, grasp execution or task success is verified.',
    }
    errors = []
    try:
        result['native_time_before_s'] = _native_time(session)
        task.verify_unchanged()
        bindings = {key: binding.name for key, binding in definition.private.bindings.items()}
        setup = getattr(session, 'category_interaction_setup', {}) or {}
        supports = {name: info['support'] for name, info in setup.get('supported_items', {}).items()
                    if info.get('support')}
        result['bindings'] = deepcopy(bindings)
        # Keep the dotted lazy import explicit for capture dependency sealing.
        import wia.candidates.inventory as native_inventory
        inventory = native_inventory.export_inventory(
            session, identity=alias or definition.id, target_names=list(bindings.values()),
            support_by_target=supports or None,
        )
        if not isinstance(inventory, dict):
            raise TypeError('Native inventory export must return a diagnostic dictionary.')
        result['inventory'] = deepcopy(inventory)
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    finally:
        try:
            result['native_time_after_s'] = _native_time(session)
        except Exception as error:
            errors.append(f'Native clock readback failed: {type(error).__name__}: {error}')

    before, after = result['native_time_before_s'], result['native_time_after_s']
    unchanged = before is not None and after is not None and before == after
    result['native_time_verified_unchanged'] = unchanged
    if unchanged:
        result.update(robot_actions=0, physics_steps=0)
    else:
        errors.append('Finite unchanged native simulation time was not verified.')
    result['available'] = not errors
    result['errors'] = errors
    if errors:
        result['error'] = '; '.join(errors)
    return result

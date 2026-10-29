"""Collect private, read-only initial mirror geometry after capture images freeze.

Run CPU checks: python -B -m pytest -q -p no:cacheprovider \
  tests/test_wia_capture_diagnostics.py
The capture caller owns image freezing and its final clock check. This helper
never renders, runs an evaluator/qualifier, steps physics, writes files or adds
anything to public observations or prompts. Unsupported tasks return None.
"""
from __future__ import annotations

from copy import deepcopy
import math

SCHEMA = 'wia_initial_review_diagnostics_v1'
MIRROR_MODULE = 'wia.authoring.converters.wia_mirror'


def _native_time(session):
    value = session.pipeline.og.sim.current_time
    if isinstance(value, bool):
        raise ValueError('Native simulation time must be finite numeric data.')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Native simulation time is nonfinite.')
    return value


def capture_initial_review_diagnostics(session):
    """Return private mirror measurements, with errors contained within metadata.

    Only the exact mirror category operation dispatches to its explicitly owned
    geometry helper. Zero-action/step metadata requires finite unchanged native
    clocks; an unavailable clock never supplies synthetic zero counts.
    """
    task = getattr(session, 'loaded_task', None)
    private = getattr(getattr(task, 'definition', None), 'private', None)
    evaluation = getattr(private, 'evaluation', None)
    if (getattr(evaluation, 'type', None) != 'category_operation'
            or getattr(evaluation, 'module', None) != MIRROR_MODULE):
        return None

    result = {'schema': SCHEMA, 'private_only': True, 'available': False,
              'task_sha256': getattr(task, 'file_sha256', None),
              'module': MIRROR_MODULE,
              'revision': getattr(evaluation, 'parameters', {}).get('revision'), 'geometry': None,
              'native_qualification_run': False, 'physical_success_verified': False,
              'native_time_before_s': None, 'native_time_after_s': None,
              'native_time_verified_unchanged': False,
              'robot_actions': None, 'physics_steps': None}
    errors = []
    try:
        result['native_time_before_s'] = _native_time(session)
        task.verify_unchanged()
        config = deepcopy(evaluation.model_dump())
        result['revision'] = config['parameters']['revision']
        config.update(scene=task.scene_name, room=task.room_name)
        config['bindings'] = {key: private.bindings[key].name for key in config['binding_keys']}
        bound = {}
        for key, name in config['bindings'].items():
            obj = session.env.scene.object_registry('name', name)
            if obj is None or obj.name != name:
                raise ValueError('The exact native diagnostic binding is unavailable: ' + key)
            bound[key] = obj
        # Keep the dotted lazy import explicit for capture dependency sealing.
        import wia.authoring.converters.wia_mirror as mirror
        geometry = mirror.capture_initial_geometry(session, config, bound)
        if not isinstance(geometry, dict):
            raise TypeError('Native mirror geometry must return a diagnostic dictionary.')
        result['geometry'] = deepcopy(geometry)
        if geometry.get('available') is not True:
            errors.append('Native mirror initial geometry is unavailable.')
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
    if errors:
        result['error'] = '; '.join(errors)
    return result

"""Read private household inventory and fit geometry after initial pixels freeze.

Run: PYTHONPATH=src python -B -m pytest -q -p no:cacheprovider
  tests/test_wia_household_capture_diagnostics.py
This helper performs no evaluation, qualification, controls, physics stepping,
rendering, file writes, scene edits, or additions to public observations.
"""
from __future__ import annotations

from copy import deepcopy
import math

from wia.runtime.wia_capture_diagnostics import HOUSEHOLD_MODULE, _native_time


SCHEMA = 'wia_initial_household_review_diagnostics_v1'


def capture_initial_household_diagnostics(session):
    """Preserve native inventory and separately measure exact private fit bindings."""
    task = getattr(session, 'loaded_task', None)
    definition = getattr(task, 'definition', None)
    private = getattr(definition, 'private', None)
    evaluation = getattr(private, 'evaluation', None)
    result = {
        'schema': SCHEMA, 'private_only': True, 'available': False,
        'task_id': getattr(definition, 'id', None), 'task_sha256': getattr(task, 'file_sha256', None),
        'module': getattr(evaluation, 'module', None),
        'revision': None, 'inventory': None, 'measured_geometry': None,
        'native_qualification_run': False, 'physical_success_verified': False,
        'native_time_before_s': None, 'native_time_after_s': None,
        'native_time_verified_unchanged': False, 'robot_actions': None, 'physics_steps': None,
        'validation_scope': 'Private native initial geometry only; no reference execution, qualification or physical success is verified.',
    }
    errors = []
    try:
        result['native_time_before_s'] = _native_time(session)
        task.verify_unchanged()
        if evaluation.type != 'category_operation' or evaluation.module != HOUSEHOLD_MODULE:
            raise ValueError('Household diagnostics require the exact household category operation.')
        parameters = deepcopy(evaluation.parameters)
        candidates = parameters['candidates']
        if not isinstance(candidates, dict) or not candidates:
            raise ValueError('Household diagnostics require explicit native candidate identities.')
        bindings = {key: binding.name for key, binding in private.bindings.items()}
        if (any(not isinstance(name, str) or not name or bindings.get(key) != name
                for key, name in candidates.items()) or len(set(candidates.values())) != len(candidates)):
            raise ValueError('Household candidates must match their exact distinct private bindings.')
        result['revision'] = parameters.get('revision')
        result['bindings'] = deepcopy(bindings)
        setup = getattr(session, 'category_interaction_setup', {}) or {}
        supports = {name: info['support'] for name, info in setup.get('supported_items', {}).items()
                    if info.get('support')}
        # Keep dotted lazy imports explicit for capture dependency sealing.
        import wia.candidates.inventory as native_inventory
        try:
            inventory = native_inventory.export_inventory(
                session, identity=definition.id, target_names=list(candidates.values()),
                support_by_target=supports or None,
            )
            if not isinstance(inventory, dict):
                raise TypeError('Native inventory export must return a diagnostic dictionary.')
            result['inventory'] = deepcopy(inventory)
        except Exception as error:
            errors.append(f'inventory: {type(error).__name__}: {error}')

        import wia.authoring.converters.wia_dimensional as dimensional
        import wia.authoring.converters.wia_household_fit as household

        geometry_errors = []
        geometry = {'available': False, 'targets': {}, 'fixtures': {},
                    'destination_interiors': {}, 'measured_aperture': None,
                    'measurement_errors': geometry_errors}
        result['measured_geometry'] = geometry

        def bound(key, name):
            obj = session.env.scene.object_registry('name', name)
            if obj is None or obj.name != name or bindings.get(key) != name:
                raise ValueError('Exact native household binding is unavailable: ' + key)
            return obj

        def failed(label, error):
            message = f'{label}: {type(error).__name__}: {error}'
            geometry_errors.append(message)
            return message

        for key, name in candidates.items():
            row = {'object_name': name, 'available': False, 'native_geometry': None,
                   'linear_velocity_m_s': None, 'angular_velocity_rad_s': None,
                   'linear_speed_m_s': None, 'angular_speed_rad_s': None,
                   'measurement_errors': []}
            geometry['targets'][key] = row
            try:
                obj = bound(key, name)
            except Exception as error:
                row['measurement_errors'].append(failed('target ' + key, error))
                continue
            try:
                measured = household.household_geometry(obj)
                if not isinstance(measured, dict) or measured.get('available') is False:
                    raise TypeError('Native household geometry must return a diagnostic dictionary.')
                row['native_geometry'] = deepcopy(measured)
            except Exception as error:
                row['measurement_errors'].append(failed('target geometry ' + key, error))
            for field, speed, getter in (
                    ('linear_velocity_m_s', 'linear_speed_m_s', 'get_linear_velocity'),
                    ('angular_velocity_rad_s', 'angular_speed_rad_s', 'get_angular_velocity')):
                try:
                    vector = dimensional._vector(getattr(obj, getter)(), 3)
                    row[field] = vector.tolist()
                    row[speed] = math.hypot(*(float(value) for value in vector))
                    if not math.isfinite(row[speed]):
                        row[speed] = None
                        raise ValueError('Native target speed must be finite.')
                except Exception as error:
                    row['measurement_errors'].append(failed('target ' + key + ' ' + field, error))
            row['available'] = not row['measurement_errors']
        for key in parameters['fixture_keys']:
            try:
                measured = dimensional._geometry(bound(key, bindings[key]))
                if not isinstance(measured, dict) or measured.get('available') is False:
                    raise TypeError('Native fixture geometry must return a diagnostic dictionary.')
                geometry['fixtures'][key] = deepcopy(measured)
            except Exception as error:
                geometry['fixtures'][key] = {'available': False,
                    'measurement_error': failed('fixture ' + key, error)}
        for destination, contract in parameters['destination_interiors'].items():
            try:
                measured = household._interior(geometry['fixtures'], contract, parameters['scoring'])
                if not isinstance(measured, dict) or measured.get('available') is not True:
                    raise ValueError('Native compartment interior measurement is unavailable.')
                geometry['destination_interiors'][destination] = deepcopy(measured)
            except Exception as error:
                geometry['destination_interiors'][destination] = {'available': False,
                    'measurement_error': failed('interior ' + destination, error)}
        if evaluation.operation == 'select_pass_aperture':
            try:
                measured = dimensional._aperture(geometry['fixtures'], parameters)
                if not isinstance(measured, dict) or measured.get('available') is not True:
                    raise ValueError('Native aperture measurement is unavailable.')
                geometry['measured_aperture'] = deepcopy(measured)
            except Exception as error:
                geometry['measured_aperture'] = {'available': False,
                    'measurement_error': failed('aperture', error)}
        geometry['available'] = not geometry_errors
        errors.extend(geometry_errors)
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

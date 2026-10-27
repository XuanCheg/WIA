"""Restore explicitly declared retained furniture during native initialization only.

Run through an after_postprocessing category_initialization hook. CPU checks:
python -B -m pytest -q -p no:cacheprovider tests/test_wia_native_furniture_hook.py
Only exact retained native poses and velocities are reset by the shared helpers;
materials, scales, body flags, joints and category interaction setup are preserved.
"""
from __future__ import annotations

import json

from wia.authoring.converters import active_layout
from wia.tasks.interaction_setup import array

SCHEMA = 'wia_native_furniture_initialization_v1'
SESSION_ATTRIBUTE = 'wia_native_furniture_initialization'
CONTACT_LIMIT = 32
ERROR_LIMIT = 512


def _contact_diagnostics(session, name):
    """Read bounded actual contact evidence without making it an acceptance gate."""
    result = {'available': False, 'total_count': None, 'contacts': [], 'truncated': False,
              'maximum_recorded_contacts': CONTACT_LIMIT}
    try:
        obj = session.env.scene.object_registry('name', name)
        contacts = list(obj.contact_list())
        result.update(total_count=len(contacts), truncated=len(contacts) > CONTACT_LIMIT)
        for contact in contacts[:CONTACT_LIMIT]:
            row = {'available': False, 'body0': None, 'body1': None, 'position': None, 'normal': None}
            try:
                for field in ('body0', 'body1'):
                    value = getattr(contact, field)
                    if value is None or not str(value):
                        raise ValueError('Native contact body identity is unavailable.')
                    row[field] = str(value)
                for field in ('position', 'normal'):
                    value = array(getattr(contact, field))
                    if value.shape != (3,):
                        raise ValueError('Native contact ' + field + ' must be a finite scalar three-vector.')
                    row[field] = value.tolist()
                row['available'] = True
            except Exception as error:
                row['measurement_error'] = (type(error).__name__ + ': ' + str(error))[:ERROR_LIMIT]
            result['contacts'].append(row)
        result['available'] = all(row['available'] for row in result['contacts'])
    except Exception as error:
        result['measurement_error'] = (type(error).__name__ + ': ' + str(error))[:ERROR_LIMIT]
    return result


def initialize_scene(session, parameters):
    """Apply explicit native poses, settle 90 steps and retain actual final readback."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Native furniture reset hooks are initialization-only.')
    scene_model = parameters.get('scene_model')
    if (not isinstance(scene_model, str) or not scene_model.strip()
            or getattr(session, 'scene_name', None) != scene_model
            or session.env.scene.scene_model != scene_model):
        raise ValueError('Native furniture reset requires its exact current native scene_model.')
    layout = {'objects': [], 'native_object_poses': parameters.get('native_object_poses', [])}
    receipts = active_layout._initialize_native_object_poses(session, layout)
    settle_steps = 0
    if receipts:
        import omnigibson as og
        for _ in range(90):
            og.sim.step()
            settle_steps += 1
    active_layout._finalize_native_object_poses(session, receipts)
    for receipt in receipts:
        receipt['contact_diagnostics'] = _contact_diagnostics(session, receipt['name'])
    result = {'schema': SCHEMA, 'scene_model': scene_model,
        'native_object_poses': receipts, 'settle_steps': settle_steps,
        'evidence_scope': 'Initialization-only original/requested/final furniture poses; no support, visibility or physical validity certificate.'}
    setattr(session, SESSION_ATTRIBUTE, result)
    print('[wia] Native furniture initialization: ' + json.dumps(result, allow_nan=False))
    return result

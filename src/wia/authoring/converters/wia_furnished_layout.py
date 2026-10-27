"""Initialize authored furniture placements and export existing support references.

Run through category_initialization patches produced by
scripts/build_wia_furnished_existing.py. CPU checks: python -B -m pytest -q
-p no:cacheprovider tests/test_wia_furnished_existing.py. This adapter delegates
initial placement to active_layout and never supplies task controls or scoring.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from wia.authoring.converters import active_layout
from wia.tasks.interaction_setup import array

MODULE = 'core.esi_converters.wia_furnished_layout'
REVISION = 'wia_furnished_layout_v1'


def validate_parameters(parameters):
    """Require explicit scene, supported-item inventory and original root poses."""
    p = deepcopy(parameters)
    if p.get('revision') != REVISION:
        raise ValueError('An explicit wia_furnished_layout_v1 revision is required.')
    identity, layout = p['scene_contract'], p['layout']
    if (not identity.get('scene_model') or not identity.get('room')
            or layout['scene_model'] != identity['scene_model'] or layout['rooms'] != [identity['room']]):
        raise ValueError('Authored layout and exact native scene/room identity must match.')
    names = [obj['name'] for obj in layout['objects']]
    items = layout.get('supported_items', [])
    supported_names = [row['name'] for row in items]
    references = p.get('source_reference_positions', {})
    if (not items or len(set(names)) != len(names) or len(set(supported_names)) != len(supported_names)
            or not set(supported_names).issubset(names) or set(references) != set(supported_names)):
        raise ValueError('Every supported target requires one authored object and one explicit original root reference.')
    for name, value in references.items():
        raw = np.asarray(value)
        if raw.dtype.kind not in 'fiu' or raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError('Original root reference must be a finite numeric vector: ' + name)
    return p


def configure_scene(config, parameters):
    validate_parameters(parameters)
    return active_layout.configure_scene(config, parameters)


def postprocess_joint_world(session, parameters):
    validate_parameters(parameters)
    return active_layout.postprocess_joint_world(session, parameters)


def compatibility_receipt(session, parameters, initialized):
    """Read actual settled roots and supports; never invent a common transform."""
    p = validate_parameters(parameters)
    if (session.env.scene.scene_model != p['scene_contract']['scene_model']
            or session.room_name != p['scene_contract']['room']):
        raise ValueError('Compatibility references require the declared live native scene and room.')
    result = deepcopy(initialized)
    result.update(object_names=list(p['source_reference_positions']), support_prim_paths={},
                  support_surfaces_z_m={}, object_translations_m={}, adapted_geometry={},
                  source_reference_positions=deepcopy(p['source_reference_positions']),
                  group_translation_m=[0., 0., 0.], group_translation_is_fallback_only=True,
                  compatibility_measurement_errors=[], relocated=False,
                  physical_success_verified=False,
                  reference_scope='Per-object measured settled-root offsets from explicit original roots.')
    for spec in p['layout']['supported_items']:
        name = spec['name']
        try:
            obj = session.env.scene.object_registry('name', name)
            support = session.env.scene.object_registry('name', spec['support'])
            if obj is None or support is None or obj.name != name or support.name != spec['support']:
                raise ValueError('Exact native target and support bindings are unavailable.')
            path = str(support.prim_path)
            if not path.startswith('/'):
                raise ValueError('Native support prim path is unavailable.')
            receipt = result['supported_items'][name]
            if receipt.get('support') != support.name:
                raise ValueError('Measured support receipt disagrees with the explicit support binding.')
            surface = (receipt['support_surface_z_m'] if spec.get('support_surface_mode', 'aabb_top') == 'local_raycast'
                       else float(array(support.aabb[1])[2]))
            if not np.isfinite(surface):
                raise ValueError('The measured native support surface is unavailable.')
            position, quaternion = (array(value) for value in obj.get_position_orientation())
            low, high = (array(value) for value in obj.aabb)
            if (position.shape != (3,) or quaternion.shape != (4,) or low.shape != (3,)
                    or high.shape != (3,) or np.any(high <= low)):
                raise ValueError('Native target pose and bounds must be measurable.')
            delta = position - array(p['source_reference_positions'][name])
            result['support_prim_paths'][name] = path
            result['support_surfaces_z_m'][name] = float(surface)
            result['object_translations_m'][name] = delta.tolist()
            result['adapted_geometry'][name] = {'position_m': position.tolist(),
                'quaternion_xyzw': quaternion.tolist(), 'aabb_min_m': low.tolist(), 'aabb_max_m': high.tolist()}
            result['relocated'] |= bool(np.any(delta != 0))
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
            result['compatibility_measurement_errors'].append(name + ': ' + f'{type(error).__name__}: {error}')
    result['accepted'] = bool(initialized.get('accepted') is True and not result['compatibility_measurement_errors'])
    return result


def initialize_scene(session, parameters):
    validate_parameters(parameters)
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Furnished layout positioning is initialization-only.')
    initialized = active_layout.initialize_scene(session, parameters)
    result = compatibility_receipt(session, parameters, initialized)
    session.category_interaction_setup = result
    return result

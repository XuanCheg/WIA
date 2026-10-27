"""Place meaningful task groups on separate measured supports before controls.

Run via scripts/run_esi_capture_worker.py. CPU fixtures can run with
python -m pytest -q tests/test_accessible_shared_groups.py. Each group reuses
the existing native support/contact/stance checks; no runtime action is issued.
"""
from __future__ import annotations

import copy

import numpy as np


def independent_group_clearance(groups, minimum_gap_m=.03):
    """Measure XY AABB gaps only across distinct meaningful placement groups.

    Vertical separation cannot make overlapping independent candidates usable.
    Within-group containment/contact is deliberately outside this predicate.
    """
    minimum_gap_m = float(minimum_gap_m)
    if not np.isfinite(minimum_gap_m) or minimum_gap_m <= 0:
        raise ValueError('Independent candidate clearance must be finite and positive.')
    pairs = []
    items = list(groups.items())
    for index, (first_group, first) in enumerate(items):
        for second_group, second in items[index + 1:]:
            for first_name, first_bounds in first.items():
                for second_name, second_bounds in second.items():
                    a, b = [np.asarray(value, dtype=float) for value in first_bounds]
                    c, d = [np.asarray(value, dtype=float) for value in second_bounds]
                    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (a, b, c, d)) or (
                            np.any(b <= a) or np.any(d <= c)):
                        raise ValueError('Independent clearance requires finite positive native AABBs.')
                    separation = np.maximum(np.maximum(c[:2] - b[:2], a[:2] - d[:2]), 0.)
                    gap = float(np.linalg.norm(separation))
                    pairs.append({'groups': [first_group, second_group],
                        'objects': [first_name, second_name], 'xy_gap_m': gap,
                        'minimum_gap_m': minimum_gap_m, 'accepted': gap + 1e-8 >= minimum_gap_m,
                        'aabb_xy_m': [[a[:2].tolist(), b[:2].tolist()], [c[:2].tolist(), d[:2].tolist()]]})
    return {'applicable': True, 'accepted': all(row['accepted'] for row in pairs),
        'minimum_gap_m': minimum_gap_m, 'pairs': pairs,
        'optical_coverage_verified': False, 'ik_verified': False,
        'scope': 'Horizontal AABB clearance between different independent meaningful groups only.'}


def initialize_groups(session, bindings, policy, initialize_single):
    """Move independent units separately, or coordinate one Z on multiple supports."""
    from .interaction_setup import REVISION, array
    from .interaction_supports import hide_supports
    from .interaction_anchors import InteractionAnchors
    from omnigibson.utils.usd_utils import PoseAPI
    groups = policy['placement_groups']
    common_z = policy['relocation_policy'] == 'vertical_only'
    objects = {key: session.env.scene.object_registry('name', bindings[key].name)
               for key in policy['group_bindings']}
    if any(obj is None for obj in objects.values()):
        result = {'schema': REVISION, 'accepted': False, 'relocated': False,
            'failure_stage': 'binding_resolution', 'reason': 'A declared placement group member is missing.',
            'warning_codes': ['interaction_layout_requires_review'], 'object_names': [], 'per_group': []}
        session.category_interaction_setup = result
        return result
    baseline = session.pipeline.og.sim.dump_state(serialized=False)
    original = {obj.name: {'position_m': array(obj.get_position_orientation()[0]).tolist(),
                           'quaternion_xyzw': array(obj.get_position_orientation()[1]).tolist(),
                           'aabb_min_m': array(obj.aabb[0]).tolist(), 'aabb_max_m': array(obj.aabb[1]).tolist()}
                for obj in objects.values()}
    camera = array(session.scene_camera_position).copy()
    viewer = copy.deepcopy(getattr(session, 'postprocess', {}).get('viewer_camera'))
    try:
        anchors = InteractionAnchors(session, policy, objects)
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        result = {'schema': REVISION, 'accepted': False, 'relocated': False,
            'failure_stage': 'original_anchor_measurement', 'reason': f'{type(error).__name__}: {error}',
            'warning_codes': ['interaction_layout_requires_review'], 'original_geometry': original,
            'object_names': [obj.name for obj in objects.values()], 'per_group': [],
            'physical_grasp_verified': False, 'ik_verified': False}
        session.category_interaction_setup = result
        return result

    def restore():
        session.pipeline.og.sim.load_state(baseline, serialized=False)
        anchors.restore()
        for slot in range(len(groups)):
            hide_supports(session, slot=slot)
        session.scene_camera_position = camera.copy()
        if viewer is not None:
            session.postprocess['viewer_camera'] = copy.deepcopy(viewer)
        PoseAPI.invalidate()
        session.invalidate_camera_cache()

    failures = []
    for required in ([None, .35, .50, .65] if common_z else [None]):
        if failures:
            restore()
        result = {'schema': REVISION, 'accepted': False, 'relocated': False,
            'object_names': [obj.name for obj in objects.values()], 'original_geometry': original,
            'object_translations_m': {}, 'support_prim_paths': {}, 'support_surfaces_z_m': {},
            'per_group': [], 'attempts': [], 'target_physics': {}, 'adapted_geometry': {},
            'height_screen_before': {}, 'height_screen_after': {},
            'physical_grasp_verified': False, 'ik_verified': False,
            'warning_codes': list(policy.get('warning_codes', ())) + ['native_grasp_execution_unverified'],
            'layout_policy': 'common_z_on_separate_supports' if common_z else 'independent_meaningful_groups'}
        chosen_z = required
        for slot, keys in enumerate(groups):
            local = copy.deepcopy(policy)
            local.update(group_bindings=list(keys), placement_groups=[],
                support_search='measured_surfaces',
                _allow_dynamic_background_settling=True,
                support_bindings=[key for key in policy.get('support_bindings', ()) if key in keys],
                manipulation_bindings=[key for key in policy['manipulation_bindings'] if key in keys],
                relocation_policy='vertical_only' if common_z else 'rigid_group',
                source_camera_policy='preserve', anchor_policy='none', fixed_reference_evidence={},
                _support_slot=slot, _pending_group_names=[objects[key].name for rest in groups[slot + 1:] for key in rest],
                _placed_group_names=[name for name in result['object_translations_m']],
                _preferred_support_names=[bindings[value].name for key, value in policy.get('support_preferences', {}).items()
                                          if key in keys])
            if not common_z:
                local.update(_independent_group_minimum_xy_gap_m=.03,
                    _independent_group_key='group_' + str(slot),
                    _placed_independent_groups={'group_' + str(i):
                        [objects[key].name for key in previous['bindings']]
                        for i, previous in enumerate(result['per_group']) if previous['evidence']['accepted']})
            if chosen_z is not None and common_z:
                local['_required_translation_z_m'] = chosen_z
            row = initialize_single(session, bindings, local)
            result['per_group'].append({'bindings': list(keys), 'evidence': copy.deepcopy(row)})
            result['attempts'].extend(row.get('attempts', []))
            result['warning_codes'].extend(row.get('warning_codes', []))
            for field in ('target_physics', 'height_screen_before', 'height_screen_after'):
                result[field].update(row.get(field, {}))
            if not row['accepted']:
                if common_z:
                    break
                continue
            result['relocated'] |= bool(row.get('relocated'))
            result.update(robot_spawn=row.get('robot_spawn', {}), spawn_grid=row.get('spawn_grid', {}))
            translations = row.get('object_translations_m', {})
            for key in keys:
                name = objects[key].name
                result['object_translations_m'][name] = translations.get(name, [0., 0., 0.])
            result['adapted_geometry'].update(row.get('adapted_geometry', {}))
            if common_z and chosen_z is None:
                chosen_z = float(result['object_translations_m'][objects[keys[0]].name][2])
            if row.get('support_prim_path'):
                before = row['original_geometry']
                support_keys = [key for key in policy.get('support_bindings', ()) if key in keys] or keys
                bottom = min(before[objects[key].name]['aabb_min_m'][2] for key in support_keys)
                for key in support_keys:
                    name = objects[key].name
                    if before[name]['aabb_min_m'][2] <= bottom + .03:
                        result['support_prim_paths'][name] = row['support_prim_path']
                        result['support_surfaces_z_m'][name] = row['support_surface_z_m']
            if row.get('added_support'):
                result.setdefault('added_supports', []).append(row['added_support'])
        good = [row for row in result['per_group'] if row['evidence']['accepted']]
        result['accepted'] = len(good) == len(groups)
        result['unrepaired_object_names'] = [objects[key].name for row in result['per_group']
                                            if not row['evidence']['accepted'] for key in row['bindings']]
        result['partial_accessible_layout'] = bool(good) and not result['accepted']
        result['warning_codes'] = sorted(set(result['warning_codes']))
        if not common_z:
            result['independent_group_clearance'] = independent_group_clearance({
                'group_' + str(i): {objects[key].name: objects[key].aabb for key in keys}
                for i, keys in enumerate(groups)})
            if not result['independent_group_clearance']['accepted']:
                result.update(accepted=False, reason='Final independent candidate footprints lack the required horizontal clearance.',
                    failure_stage='final_independent_group_clearance', partial_accessible_layout=bool(good))
        if result['accepted']:
            try:
                result.update(anchors.apply({name: array(value) for name, value in result['object_translations_m'].items()}),
                              **anchors.measure_reference())
                if common_z and policy.get('source_camera_policy') != 'preserve':
                    delta = np.array([0., 0., chosen_z])
                    session.scene_camera_position = camera + delta
                    if viewer is not None and viewer.get('position') is not None:
                        session.postprocess['viewer_camera']['position'] = (array(viewer['position']) + delta).tolist()
                result['group_translation_m'] = [0., 0., chosen_z] if common_z else None
                result['source_reference_camera_translation_m'] = ([0., 0., chosen_z]
                    if common_z and policy.get('source_camera_policy') != 'preserve' else [0., 0., 0.])
                if not common_z:
                    # Anchor updates and later groups' native settling may change
                    # earlier objects. Remeasure the entire assembled layout.
                    result['independent_group_clearance'] = independent_group_clearance({
                        'group_' + str(i): {objects[key].name: objects[key].aabb for key in keys}
                        for i, keys in enumerate(groups)})
                    if not result['independent_group_clearance']['accepted']:
                        result['failure_stage'] = 'final_independent_group_clearance'
                        raise ValueError('Final independent candidate footprints lack the required horizontal clearance.')
                session.category_interaction_setup = result
                return result
            except (ValueError, RuntimeError) as error:
                result.update(accepted=False, reason=str(error),
                    failure_stage=result.get('failure_stage', 'shared_anchor_translation'))
        result.setdefault('reason', 'One or more meaningful task groups could not obtain verified accessible support.')
        result.setdefault('failure_stage', 'group_support_and_approach')
        result['warning_codes'].append('interaction_layout_requires_review')
        failures.append(copy.deepcopy(result))
        if not common_z:
            break
    if common_z:
        restore()
        result.update(relocated=False, partial_accessible_layout=False, object_translations_m={},
                      support_prim_paths={}, support_surfaces_z_m={}, adapted_geometry={})
        result['common_height_trials'] = failures
    session.category_interaction_setup = result
    return result

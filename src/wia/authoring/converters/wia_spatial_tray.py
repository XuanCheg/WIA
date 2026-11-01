"""Read native retrieve-and-arrange tray evidence; place only at initialization.

Run initialization review with src/wia/runtime/commands/check_esi_core_native.py using an authored
spatial-tray definition. Run CPU validation with python -B -m pytest -q
-p no:cacheprovider tests/test_wia_spatial_tray.py. No helper in this module
launches a reference policy, manipulation, navigation or simulation process.
"""
from __future__ import annotations

from wia.runtime.prechecks import episode_qualification, permits_episode

from wia.authoring.modules import module_identity

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from wia.authoring.converters import active_layout
from wia.authoring.converters.rigid_partial_insertion import _contact_bodies, _under
from wia.authoring.converters.spatial_relations_physical_contact import _contact_geometry, _snapshot
from wia.authoring.converters.wia_dimensional import _control, _grasp
from wia.authoring.io import write_json
from wia.tasks.interaction_setup import array
from wia.tasks.wia_spatial_tray_scoring import FINAL_STATE_REVISION, REVISION, SpatialTrayProgress, validate_parameters

MODULE = 'wia.authoring.converters.wia_spatial_tray'
OPERATION = 'retrieve_arrange_in_tray'
ERRORS = (AttributeError, ValueError, RuntimeError, TypeError, KeyError, IndexError, AssertionError, ImportError)


def validate_contract(config):
    """Validate exact native binding names and the explicit scoring contract."""
    if module_identity(config.get('module')) != MODULE or config.get('operation') != OPERATION:
        raise ValueError('Use the explicit spatial-tray module and operation.')
    p = validate_parameters(config['parameters'])
    identity, bindings = p['scene_contract'], config['bindings']
    if (config.get('scene') != identity['scene_model'] or config.get('room') != identity['room']
            or set(config.get('binding_keys', ())) != set(bindings)
            or any(not isinstance(name, str) or not name for name in bindings.values())):
        raise ValueError('Scene identity and all binding names must be explicit.')
    required = {*p['targets'], *p['initial_supports'].values(), p['tray']['deck_key'],
                p['tray']['table_support_key'], *p['tray']['rim_keys']}
    if (not required.issubset(bindings)
            or any(bindings.get(key) != name for key, name in p['targets'].items())):
        raise ValueError('Targets, initial supports and every tray fixture require exact bindings.')
    # Several semantic support aliases can refer to different exposed places on
    # one native counter. Targets and the tray pieces still have unique bodies.
    task_keys = [*p['targets'], p['tray']['deck_key'], *p['tray']['rim_keys']]
    task_names = [bindings[key] for key in task_keys]
    support_names = [bindings[key] for key in {*p['initial_supports'].values(), p['tray']['table_support_key']}]
    if len(set(task_names)) != len(task_names) or set(task_names) & set(support_names):
        raise ValueError('Targets and tray pieces require distinct native identities separate from native supports.')
    return p


def _layout_contract(parameters):
    if parameters.get('revision') != REVISION:
        raise ValueError('The spatial-tray initialization revision is required.')
    layout, identity = parameters['layout'], parameters['scene_contract']
    if layout['scene_model'] != identity['scene_model'] or layout['rooms'] != [identity['room']]:
        raise ValueError('The complete layout must match the explicit native scene and room.')
    names = [obj['name'] for obj in layout['objects']]
    if len(names) != len(set(names)):
        raise ValueError('Authored objects require unique native names.')
    # Supports can be already-existing scene furniture; only targets being
    # positioned need an authored object row for active_layout to restore.
    if any(row['name'] not in names for row in layout.get('supported_items', [])):
        raise ValueError('Every initialization-positioned item needs an explicit authored object row.')
    return layout


def configure_scene(config, parameters):
    _layout_contract(parameters)
    return active_layout.configure_scene(config, parameters)


def postprocess_joint_world(session, parameters):
    _layout_contract(parameters)
    return active_layout.postprocess_joint_world(session, parameters)


def initialize_scene(session, parameters):
    _layout_contract(parameters)
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Spatial-tray positioning is initialization-only.')
    return active_layout.initialize_scene(session, parameters)


def _configuration(session, config, bound=None):
    p = validate_contract(config)
    identity = p['scene_contract']
    if (session.scene_name != identity['scene_model'] or session.room_name != identity['room']
            or session.env.scene.scene_model != identity['scene_model']):
        raise ValueError('Declared and live native scene/room identities must agree.')
    if bound is None:
        bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
    if any(bound.get(key) is None or bound[key].name != name for key, name in config['bindings'].items()):
        raise ValueError('Every exact native binding must exist in the current scene.')
    return p, bound


def _native_boolean(value):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError('An actual native boolean is required.')
    return bool(value)


def _fixture_snapshot(obj):
    """Read actual fixture geometry and native flags without requesting velocity."""
    position, quaternion = (array(value) for value in obj.get_position_orientation())
    low, high = (array(value) for value in obj.aabb)
    if (any(value.shape != (3,) for value in (position, low, high))
            or quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0 or np.any(high <= low)):
        raise ValueError('Invalid measured native fixture geometry.')
    model = getattr(obj, 'model', None)
    return {'name': obj.name, 'category': obj.category, 'model': model, 'model_available': model is not None,
        'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
        'visual_only': _native_boolean(obj.visual_only), 'fixed_base': _native_boolean(obj.fixed_base)}


def _pair(first, second, tolerance):
    """Never substitute overlap or Touching alone for pair-specific PhysX data."""
    from omnigibson.object_states import Touching
    first_low, first_high = (array(value) for value in first.aabb)
    second_low, second_high = (array(value) for value in second.aabb)
    widths = np.minimum(first_high, second_high) - np.maximum(first_low, second_low)
    gap = float(np.linalg.norm(np.maximum(-widths, 0.)))
    touching = _native_boolean(first.states[Touching].get_value(second))
    return _contact_geometry({'obj1': first, 'obj2': second}, touching, gap, tolerance=tolerance)


def _deck_pair(target, deck, tolerance, *, on_top):
    """Retain PhysX evidence, or bound penetration at a fixed Cube's top face.

    Static links are absent from RigidContactAPI's dynamic column filters.
    For a horizontal boundingCube and a target strictly inside its XY bounds,
    only the top face can intersect a target extending above the deck. Every
    target collision point is at least target_min_z, so max(0, deck_max_z -
    target_min_z) bounds penetration into that face. A larger bound is
    inconclusive, never evidence of penetration. Native Touching AND OnTop
    remain required; the bound itself supplies neither contact nor support.
    """
    try:
        native = _pair(target, deck, tolerance)
    except ERRORS as error:
        return {'available': False, 'relation': 'unresolved', 'penetrating': None,
                'reason': f'{type(error).__name__}: {error}'}
    if native.get('available') is True:
        return native
    evidence = {'available': False, 'method': 'native_fixed_cube_top_bound'}
    try:
        from omnigibson.object_states import Touching
        if (not _native_boolean(on_top)
                or not _native_boolean(target.states[Touching].get_value(deck))):
            raise ValueError('Actual native Touching and OnTop must both be true.')
        if (not _native_boolean(deck.fixed_base) or _native_boolean(deck.visual_only)
                or getattr(deck, '_primitive_type', None) != 'Cube'):
            raise ValueError('The fallback requires the actual fixed collidable Cube deck.')
        if _native_boolean(target.fixed_base) or _native_boolean(target.visual_only):
            raise ValueError('The target must remain dynamic and collidable.')
        links = [link for link in deck.links.values() if not _native_boolean(link.is_meta_link)]
        if len(links) != 1 or len(links[0].collision_meshes) != 1:
            raise ValueError('The Cube deck must have exactly one physical collision mesh.')
        mesh = next(iter(links[0].collision_meshes.values()))
        if (not _native_boolean(mesh.collision_enabled)
                or mesh.get_collision_approximation() != 'boundingCube'):
            raise ValueError('An enabled native boundingCube collision shape is required.')
        _, quaternion = (array(value) for value in mesh.get_position_orientation())
        # Exact horizontal orientation, not a relaxed fixture-angle tolerance.
        if (quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0
                or np.any(quaternion[:2] != 0.)):
            raise ValueError('The measured Cube collision top must be horizontal.')
        low, high = (array(value) for value in target.aabb)
        deck_low, deck_high = (array(value) for value in mesh.aabb)
        if (any(value.shape != (3,) for value in (low, high, deck_low, deck_high))
                or np.any(high <= low) or np.any(deck_high <= deck_low)):
            raise ValueError('Current collision bounds must be finite and nondegenerate.')
        if (not np.all(low[:2] > deck_low[:2]) or not np.all(high[:2] < deck_high[:2])
                or high[2] <= deck_high[2]):
            raise ValueError('The complete target footprint must be inside the deck and extend above its top.')
        gap = float(low[2] - deck_high[2])
        depth_bound = max(0., -gap)
        evidence.update(target_collision_aabb_m=[low.tolist(), high.tolist()],
            deck_collision_aabb_m=[deck_low.tolist(), deck_high.tolist()],
            deck_collision_prim_path=str(mesh.prim_path),
            deck_collision_quaternion_xyzw=quaternion.tolist(),
            vertical_gap_m=gap, penetration_depth_upper_bound_m=depth_bound)
        if (isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance)
                or tolerance <= 0 or depth_bound > tolerance):
            raise ValueError('The conservative depth bound does not establish the declared penetration tolerance.')
        evidence['available'] = True
        return {'available': True, 'relation': 'touching', 'penetrating': False,
            'separation_source': 'native_fixed_cube_top_bound',
            'penetration_depth_upper_bound_m': depth_bound,
            'penetration_tolerance_m': tolerance, 'native_touching': True, 'native_on_top': True,
            'fixed_deck_bound': evidence, 'native_pair_contact_geometry': deepcopy(native)}
    except ERRORS as error:
        evidence['reason'] = f'{type(error).__name__}: {error}'
        return {**native, 'fixed_deck_bound': evidence}


def _preserved(row, initial, scoring):
    a, b = (array(value['quaternion_xyzw']) for value in (row, initial))
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    angle = 2 * math.acos(min(1., dot))
    return bool(np.linalg.norm(array(row['position']) - array(initial['position']))
                <= scoring['fixture_position_tolerance_m']
                and angle <= scoring['fixture_angle_tolerance_rad']
                and np.allclose(array(row['aabb_max']) - array(row['aabb_min']),
                                array(initial['aabb_max']) - array(initial['aabb_min']), rtol=0.,
                                atol=scoring['fixture_position_tolerance_m'])
                and (scoring.get('manipulation_revision') != FINAL_STATE_REVISION
                     or np.array_equal(row['scale'], initial['scale'])))


def _scale(obj):
    """Read the native scale rather than inferring it from a rotated AABB."""
    scale = array(obj.scale)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError('Native object scale must have three positive finite components.')
    return scale.tolist()


def _measure(session, controller, config, bound, baseline=None):
    from omnigibson.object_states import OnTop
    p, errors = config['parameters'], []
    tray, scoring = p['tray'], p['scoring']
    final_state = scoring.get('manipulation_revision') == FINAL_STATE_REVISION
    grasp, raw_grasp = _grasp(session, controller)
    control_valid, control = _control(session, controller)
    if control_valid and control.get('available') is not True:
        control_valid = None
    sample = {'native_time_s': None, 'control_valid': control_valid, 'control_evidence': control,
              'fixtures_valid': None, 'tray_valid': None, 'initial_supports_valid': None,
              'grasp': grasp, 'native_grasp_diagnostics': raw_grasp, 'targets': {},
              'fixtures': {}, 'initial_supports': {}, 'measurement_errors': errors}
    try:
        value = session.pipeline.og.sim.current_time
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError('Native simulation time is unavailable.')
        sample['native_time_s'] = float(value)
    except ERRORS as error:
        errors.append(f'clock: {type(error).__name__}: {error}')
    fixture_keys = list(dict.fromkeys([tray['deck_key'], tray['table_support_key'], *tray['rim_keys'],
                                      *p['initial_supports'].values()]))
    fixture_flags = []
    for key in fixture_keys:
        row = {}
        try:
            row = _fixture_snapshot(bound[key])
            if final_state:
                row['scale'] = _scale(bound[key])
            valid = not row['visual_only']
            if key in [tray['deck_key'], *tray['rim_keys']]:
                valid &= row['fixed_base']
            if baseline is not None:
                valid &= _preserved(row, baseline['fixtures'][key], scoring)
            row['preserved'] = bool(valid)
            fixture_flags.append(bool(valid))
        except ERRORS as error:
            row['measurement_error'] = f'{type(error).__name__}: {error}'
            fixture_flags.append(None)
        sample['fixtures'][key] = row
    sample['fixtures_valid'] = (False if False in fixture_flags else
                                None if None in fixture_flags else True)
    deck = bound[tray['deck_key']]
    try:
        row = sample['fixtures'][tray['deck_key']]
        low, high = array(row['aabb_min']), array(row['aabb_max'])
        bounds = array(tray['usable_world_xy_m'])
        support = bound[tray['table_support_key']]
        support_low, support_high = (array(value) for value in support.aabb)
        bottom_gap = float(low[2] - support_high[2])
        mounted = bool(np.all(low[:2] >= support_low[:2]) and np.all(high[:2] <= support_high[:2])
                       and abs(bottom_gap) <= scoring['maximum_bottom_gap_m'])
        sample['tray_measurements'] = {'mounting_mode': tray['mounting_mode'],
            'geometric_mounting_only': True, 'static_static_contact_claimed': False,
            'native_tabletop_aabb_plane_world_z_m': float(support_high[2]),
            'deck_bottom_to_tabletop_gap_m': bottom_gap, 'deck_wholly_over_tabletop': mounted,
            'deck_top_world_z_m': float(high[2]), 'usable_world_xy_m': bounds.tolist()}
        sample['tray_valid'] = bool(mounted and np.all(bounds[0] >= low[:2])
            and np.all(bounds[1] <= high[:2])
            and abs(high[2] - tray['deck_top_world_z_m']) <= scoring['deck_height_tolerance_m'])
    except ERRORS as error:
        errors.append(f'tray: {type(error).__name__}: {error}')
    support_flags = []
    for key, name in p['targets'].items():
        row = {'name': name, 'dynamic_collidable': None, 'robot_contact': None,
               'on_top_deck': None, 'deck_contact_geometry': {'available': False}, 'measurement_errors': []}
        obj = bound[key]
        try:
            row.update(_snapshot(obj))
            if final_state:
                row['scale'] = _scale(obj)
            row['dynamic_collidable'] = not row['visual_only'] and not row['fixed_base']
        except ERRORS as error:
            row['measurement_errors'].append(f'geometry: {type(error).__name__}: {error}')
        try:
            contacts = list(obj.contact_list())
            row['robot_contact'] = any(_under(body, session.robot.prim_path)
                for contact in contacts for body in _contact_bodies(contact))
        except ERRORS as error:
            row['measurement_errors'].append(f'robot_contact: {type(error).__name__}: {error}')
        try:
            row['on_top_deck'] = _native_boolean(obj.states[OnTop].get_value(deck))
            row['deck_contact_geometry'] = _deck_pair(obj, deck, scoring['penetration_tolerance_m'],
                                                      on_top=row['on_top_deck'])
        except ERRORS as error:
            row['measurement_errors'].append(f'deck_contact: {type(error).__name__}: {error}')
        sample['targets'][key] = row
        if baseline is None:
            evidence = {'support_binding': p['initial_supports'][key]}
            support = bound[p['initial_supports'][key]]
            setup = getattr(session, 'category_interaction_setup', {})
            receipt = deepcopy(setup.get('supported_items', {}).get(name, {}))
            flag = bool(receipt.get('support') == support.name and receipt.get('accepted') is True)
            evidence.update(active_layout_support_receipt=receipt, accepted=flag,
                            pair_separation_diagnostic_only=True)
            support_flags.append(flag)
            try:
                geometry = _pair(obj, support, scoring['penetration_tolerance_m'])
                on_top = _native_boolean(obj.states[OnTop].get_value(support))
                slo, shi = (array(value) for value in support.aabb)
                low, high = array(row['aabb_min']), array(row['aabb_max'])
                evidence.update(contact_geometry=geometry, on_top=on_top,
                    whole_footprint=bool(np.all(low[:2] >= slo[:2]) and np.all(high[:2] <= shi[:2])))
            except ERRORS as error:
                evidence['measurement_error'] = f'{type(error).__name__}: {error}'
            sample['initial_supports'][key] = evidence
    if baseline is None:
        sample['initial_supports_valid'] = (False if False in support_flags else
                                            None if None in support_flags else True)
    if p['goal_relation']['kind'] == 'contact':
        first, second = p['goal_relation']['pair']
        try:
            sample['contact_geometry'] = _pair(bound[first], bound[second], scoring['penetration_tolerance_m'])
        except ERRORS as error:
            sample['contact_geometry'] = {'available': False, 'relation': 'unresolved',
                                         'reason': f'{type(error).__name__}: {error}'}
    return sample


def _qualification(session, p, sample):
    setup = deepcopy(getattr(session, 'category_interaction_setup', {}))
    scorer = SpatialTrayProgress(p)
    result = scorer.begin(sample)
    checks = {'active_layout_initialization': setup.get('accepted') is True,
              'native_reducer_initial_evidence': result['evaluation_validity'] == 'complete',
              'separate_native_initial_supports': sample.get('initial_supports_valid') is True,
              'initial_target_stationary': all(row.get('robot_contact') is False
                  and row.get('linear_speed_m_s', math.inf) <= p['scoring']['maximum_linear_speed_m_s']
                  and row.get('angular_speed_rad_s', math.inf) <= p['scoring']['maximum_angular_speed_rad_s']
                  for row in sample['targets'].values())}
    return {'accepted': bool(all(checks.values())), 'checks': checks, 'initial_support': setup,
            'native_measurements': deepcopy(sample), 'initial_reducer': result,
            'route_verified': False, 'manipulation_verified': False,
            'complete_reference_execution_verified': False,
            'validation_scope': 'Native initialization, separate support, tray and aggregate hands only.'}


def qualify_operation(session, controller, config, bound):
    try:
        p, bound = _configuration(session, config, bound)
        controller = getattr(controller, 'controller', controller)
        return _qualification(session, p, _measure(session, controller, config, bound))
    except ERRORS as error:
        return {'accepted': False, 'checks': {}, 'reason': 'native_spatial_tray_evidence_unavailable',
                'measurement_error': f'{type(error).__name__}: {error}',
                'route_verified': False, 'manipulation_verified': False,
                'complete_reference_execution_verified': False}


class SpatialTrayEvaluator:
    """Record passive evidence; only native sample_step calls advance the reducer."""

    def __init__(self, session, controller, config, output):
        self.p, self.bound = _configuration(session, config)
        self.session, self.controller = session, getattr(controller, 'controller', controller)
        self.config, self.output = deepcopy(config), Path(output)
        self.scorer, self.initial, self.last_sample = None, None, None

    def begin_episode(self, observation):
        if self.scorer is not None:
            raise RuntimeError('The spatial-tray episode has already begun.')
        self.initial = _measure(self.session, self.controller, self.config, self.bound)
        self.last_sample = deepcopy(self.initial)
        self.initial_qualification = episode_qualification(self.session, _qualification, self.session, self.p, self.initial)
        self.scorer = SpatialTrayProgress(self.p)
        self.scorer.begin(self.initial)
        return self.record(observation)

    def sample_step(self):
        if self.scorer is not None:
            self.last_sample = _measure(self.session, self.controller, self.config, self.bound, self.initial)
            self.scorer.update(self.last_sample)

    def record(self, observation, *, action=None):
        if self.scorer is None:
            return None
        result = self.scorer.result()
        if not permits_episode(self.session, self.initial_qualification):
            result.update(evaluation_validity='unavailable', completion_score=None, score_percent=None,
                          task_success=False, physical_success=False)
        result.update(snapshot_id=observation.snapshot_id, action_step=observation.action_step, action=action,
                      configuration=deepcopy(self.config), native_measurements=deepcopy(self.last_sample),
                      initial_qualification=deepcopy(self.initial_qualification),
                      complete_reference_execution_verified=False)
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    return SpatialTrayEvaluator(session, controller, config, output)

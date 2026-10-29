"""Versioned destination proposals, blue-table initialization and native validation.

Run: python -B -m pytest -q tests/test_task_edit2_missing_definitions.py
Construction hooks execute once before public controls; qualification is read-only.
Saved scene roots select proposals and never establish native label or fit truth.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from wia.authoring.conversion import ConverterBase
from wia.tasks.delivery_support import SLOT, BLUE_RGBA, PUBLIC_LABEL
from wia.tasks.interaction_supports import support_names

MODULE = 'wia.authoring.converters.task_edit2_destinations'
REVISION = 'task_edit2_20261018_v1'


def destination_proposal(source, seed):
    """Retain observable existing furniture proposals; explicitly name fallback."""
    from wia.authoring.scene_inventory import destination
    proposal = destination(source, seed)
    if proposal is None:
        proposal = {'binding': {'name': support_names(SLOT)[0], 'category': 'table'},
            'public_label': PUBLIC_LABEL, 'relation': 'on_top', 'selection_seed': seed,
            'source': 'explicit_task_edit2_blue_delivery_support',
            'public_label_evidence': {'kind': 'reserved_blue_delivery_table', 'slot': SLOT,
                                    'rgba': BLUE_RGBA}, 'fit_verified': False, 'access_verified': False}
    return proposal


def attach_destination_contract(result, source, context):
    """Attach initialization/qualification contracts before schema validation/write."""
    task, scene = result.get('task_definition'), result.get('scene_definition')
    if task is None:
        return result
    proposal = context.config['placement_destination']
    p, bindings = task['private']['evaluation']['parameters'], task['private']['bindings']
    targets = p.get('target_keys') or (['target'] if 'target' in bindings else [])
    if not targets:
        raise ValueError('Named delivery requires explicit original target binding keys.')
    contract = {'version': REVISION, 'source_identity': source.relative_path,
        'scene': source.raw['scene'], 'room': source.raw['room'], 'selection_seed': context.seed,
        'destination_binding': deepcopy(bindings['destination']), 'public_label': proposal['public_label'],
        'label_relation': deepcopy(proposal['public_label_evidence']),
        'placement_relation': proposal['relation'], 'target_bindings': {key: deepcopy(bindings[key]) for key in targets},
        'fit_margin_m': .04, 'label_distance_gap_m': .10,
        'distinguishability_verified': False, 'fit_verified': False, 'access_verified': False,
        'native_qualification_required': ['current_loaded_label_relation', 'every_target_surface_fit',
            'public_visual_label_association', 'public_control_approach_and_release'],
        'fallback_table': proposal['source'] == 'explicit_task_edit2_blue_delivery_support'}
    p['source_task_revision'] = p.get('task_revision')
    p.update(task_revision=REVISION, definition_revision=REVISION, destination_contract=contract)
    task.setdefault('repair', {}).update(definition_revision=REVISION, destination_contract=deepcopy(contract))
    task['version'] = task.get('version', 1) + 1
    scene['version'] = scene.get('version', 1) + 1
    if contract['fallback_table']:
        for definition in (task, scene):
            variants = definition['scene']['variants']
            adapted = deepcopy(variants.get('adapted', variants['original']))
            before = {'operation': 'category_initialization', 'phase': 'before_construction',
                      'module': MODULE, 'parameters': {'destination_contract': deepcopy(contract)}}
            after = dict(deepcopy(before), phase='after_postprocessing')
            adapted['patches'] = [before, *adapted['patches'], after]
            variants['adapted'] = adapted
        task['public']['environment'] += ' A blue delivery table is added without moving the source objects or furniture.'
    result.update(definition_revision=REVISION, destination_contract=deepcopy(contract),
        warning_codes=sorted(set(result.get('warning_codes', [])) | {'task_edit2_destination_native_qualification_pending'}))
    result.setdefault('change_summary', []).append(
        'Resolve the public destination using a seeded observable label; native distinction, fit and access remain pending.')
    return result


def configure_scene(config, parameters):
    """Preload exactly five fixed blue primitives in the reserved slot 99."""
    from wia.tasks.interaction_supports import preload_supports
    contract = parameters['destination_contract']
    if contract.get('fallback_table') is not True:
        raise ValueError('The delivery-table hook requires an explicit fallback contract.')
    preload_supports(config, slots=1, start_slot=SLOT, rgba=BLUE_RGBA)
    return {'preloaded_names': list(support_names(SLOT)), 'slot': SLOT, 'rgba': BLUE_RGBA,
            'native_support_qualified': False, 'source_objects_replaced': False}


def initialize_scene(session, parameters):
    from wia.tasks.delivery_support import initialize_delivery_support
    if getattr(session, '_task_edit2_delivery_initialized', False):
        raise RuntimeError('The delivery support cannot be initialized twice.')
    session._task_edit2_delivery_initialized = True
    return initialize_delivery_support(session, parameters['destination_contract'])


def _native_rows(session):
    from wia.tasks.interaction_setup import _objects, array
    scene = session.env.scene
    rows = []
    for obj in _objects(scene):
        if obj.visual_only:
            continue
        low, high = (array(value) for value in obj.aabb)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all():
            raise ValueError('Loaded native object geometry is unavailable for destination label uniqueness.')
        centre = (low + high) / 2
        rows.append({'name': obj.name, 'category': obj.category, 'model': getattr(obj, 'model', None),
                     'room': scene.seg_map.get_room_instance_by_point(centre[:2]),
                     'centre_m': centre.tolist(), 'aabb_m': [low.tolist(), high.tolist()]})
    return rows


def resolve_native_label(rows, contract):
    """Resolve only current loaded native AABB centres and room/category labels."""
    expected = contract['destination_binding']
    room = [row for row in rows if row['room'] == contract['room']]
    peers = [row for row in room if row['category'] == expected['category']]
    selected = next((row for row in peers if row['name'] == expected['name']), None)
    if selected is None:
        return {'accepted': False, 'reason': 'destination_absent_from_native_loaded_room'}
    label = contract['label_relation']
    if label['kind'] == 'unique_room_category':
        return {'accepted': len(peers) == 1, 'native_peer_names': [row['name'] for row in peers]}
    if label['kind'] == 'nearest_unique_room_landmark':
        landmark = label['landmark_binding']
        anchors = [row for row in room if row['category'] == landmark['category']]
        if len(anchors) != 1 or anchors[0]['name'] != landmark['name']:
            return {'accepted': False, 'reason': 'native_landmark_not_unique_or_identity_changed'}
        centre = np.asarray(anchors[0]['centre_m'])[:2]
        distances = sorted((float(np.linalg.norm(np.asarray(row['centre_m'])[:2] - centre)), row['name'])
                           for row in peers)
        gap = distances[1][0] - distances[0][0] if len(distances) > 1 else float('inf')
        return {'accepted': distances[0][1] == selected['name'] and gap >= contract['label_distance_gap_m'],
                'native_aabb_centre_distances_m': {name: value for value, name in distances},
                'native_distance_gap_m': gap if np.isfinite(gap) else None}
    if label['kind'] == 'reserved_blue_delivery_table':
        # Identity/colour and construction are checked using the native support
        # receipt; ordinary furniture named table is not treated as equivalent.
        return {'accepted': selected['name'] == support_names(SLOT)[0], 'slot': SLOT}
    return {'accepted': False, 'reason': 'unknown_destination_label_relation'}


def validate_destination(session, config, bound=None, *, stage='evaluation'):
    """Read-only qualification API; never mutate actors or certify from proposals.

    ``config`` is the category evaluation dict with parameters.destination_contract
    and runtime native bindings. The caller may pass its already-bound objects.
    Public-control/visibility/fit receipts are episode-bound, not source labels.
    Initialization accepts only the existing structural checks and explicitly
    leaves episode semantic/fit/approach/release qualification pending. It never
    certifies full feasibility or requires future release before episode start.
    The default evaluation stage retains every original receipt requirement.
    """
    if stage not in ('initialization', 'evaluation'):
        raise ValueError('Destination validation stage must be initialization or evaluation.')
    episode_gates = ('current_episode_receipt', 'native_surface_fit_verified',
        'public_visual_label_association_verified', 'public_control_approach_verified',
        'public_control_release_verified')
    contract = config.get('parameters', {}).get('destination_contract')
    if not contract:
        return {'applicable': False, 'accepted': True}
    checks, measured = {}, {}
    try:
        checks['scene_identity_matches'] = session.scene_name == contract['scene']
        rows = _native_rows(session)
        measured['native_label_relation'] = resolve_native_label(rows, contract)
        checks['public_label_uniquely_resolves'] = measured['native_label_relation']['accepted']
        names = {row['name']: row for row in rows}
        dest = names[contract['destination_binding']['name']]
        checks['destination_identity_matches'] = all(dest.get(key) == value for key, value in
            contract['destination_binding'].items() if key in ('name', 'category', 'model'))
        checks['bound_destination_matches'] = (config.get('bindings', {}).get('destination', dest['name']) == dest['name'])
        sizes = []
        for key, identity in contract['target_bindings'].items():
            row = names[identity['name']]
            checks['target_identity:' + key] = all(row.get(field) == value for field, value in identity.items()
                                                  if field in ('name', 'category', 'model'))
            size = np.diff(np.asarray(row['aabb_m']), axis=0)[0]
            sizes.append(size)
        maximum = np.max(sizes, axis=0)
        columns = int(np.ceil(np.sqrt(len(sizes))))
        rows_count = int(np.ceil(len(sizes) / columns))
        required = np.array([columns * maximum[0], rows_count * maximum[1]]) + np.array([columns + 1, rows_count + 1]) * contract['fit_margin_m']
        available = np.diff(np.asarray(dest['aabb_m']), axis=0)[0][:2]
        checks['native_aabb_footprint_fits_all_targets'] = bool(np.all(available >= required))
        measured.update(destination=dest, required_target_grid_footprint_m=required.tolist(),
                        available_destination_footprint_m=available.tolist())
        if contract['fallback_table']:
            construction = getattr(session, 'delivery_support_initialization', {})
            checks['blue_table_native_geometry_initialized'] = bool(construction.get('accepted')
                and construction.get('source_identity') == contract['source_identity']
                and construction.get('legs_reach_measured_floor') is True)
            measured['delivery_support_initialization'] = deepcopy(construction)
        if stage == 'initialization':
            accepted = all(checks.values())
            return {'applicable': True, 'accepted': accepted, 'checks': checks,
                'native_measurements': measured, 'qualification_complete': False,
                'pending_qualification': list(episode_gates), 'evaluation_validity': 'unavailable',
                'reason': 'native_destination_episode_qualification_pending' if accepted else
                    'native_destination_structural_checks_failed'}
        receipt = getattr(session, 'destination_qualification', {}).get(contract['source_identity'], {})
        checks['current_episode_receipt'] = (getattr(session, 'episode_id', None) is not None
            and receipt.get('episode_id') == session.episode_id
            and receipt.get('selection_seed') == contract['selection_seed']
            and receipt.get('destination_name') == dest['name'])
        for name in ('native_surface_fit_verified', 'public_visual_label_association_verified',
                     'public_control_approach_verified', 'public_control_release_verified'):
            checks[name] = receipt.get(name) is True
        accepted = all(checks.values())
        return {'applicable': True, 'accepted': accepted, 'checks': checks, 'native_measurements': measured,
                'qualification_complete': accepted,
                'pending_qualification': [name for name in episode_gates if not checks[name]],
                'evaluation_validity': 'complete' if accepted else 'unavailable',
                'reason': None if accepted else 'native_destination_qualification_pending_or_failed'}
    except Exception as error:
        return {'applicable': True, 'accepted': False, 'checks': checks, 'native_measurements': measured,
                'qualification_complete': False,
                'pending_qualification': [name for name in episode_gates if not checks.get(name, False)],
                'evaluation_validity': 'unavailable', 'reason': f'{type(error).__name__}: {error}'}


class DefinitionRevisionConverter(ConverterBase):
    """Wrap an existing converter without changing generic source generation."""
    version = REVISION + '_missing_definitions_v1'

    def __init__(self, converter):
        self.converter, self.category = converter, converter.category

    def inspect(self, source):
        return self.converter.inspect(source)

    def convert(self, source, context):
        result = self.converter.convert(source, context)
        result.update(converter_module=type(self.converter).__module__, definition_converter_module=MODULE)
        if context.config.get('placement_destination'):
            return attach_destination_contract(result, source, context)
        return result

    def implementation_hash(self):
        from wia.authoring.io import sha256
        files = [Path(__file__), Path(__file__).parents[2] / 'tasks/delivery_support.py',
                 Path(__file__).parents[2] / 'tasks/interaction_supports.py']
        value = {'base': self.converter.implementation_hash(), 'owned': {str(path): sha256(path) for path in files}}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

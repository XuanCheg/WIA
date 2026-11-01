"""Versioned shared-world all-target search definitions and evidence reducer.

Run: PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q tests/test_task_edit2_search.py
Native receipts must come from the synchronized mounted-observation adapter; this
module never converts source annotations or segmentation pixel counts into proof.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from copy import deepcopy
import json
from pathlib import Path

from wia.authoring.conversion import original_scene, physical_task
from wia.authoring.io import digest, sha256, write_json

MODULE = 'wia.authoring.converters.enumerative_perception_search'
REVISION = 'task_edit2_search_20261018_v3'
RUNTIME_REVISIONS = ('task_edit2_search_20261018_v2', REVISION)
START_INITIALIZATION = 'initialize_task_edit2_search_start_v1'
OPERATION = 'all_target_search'
MECHANISMS = ('Category Ambiguity', 'Counting w Occlusion', 'Illumination Variability',
              'Merged Observation', 'Spatial Segmentation')
NATIVE_GATES = ('joint_inventory', 'source_relations', 'floor_supported_start',
                'loaded_routes', 'mounted_camera_audit', 'legal_informative_access',
                'constraints_preserved')


def _normal(value):
    return ' '.join(str(value).lower().replace('_', ' ').split())


def build_search_candidate(sources, context, *, public_areas, name):
    """Create one explicit inventory; distinct source worlds never merge by name.

    Sources occupy different rooms of the same building version. Saved poses
    remain provenance; the versioned layout translates exact ambiguity pairs and
    the unchanged merged cluster to explicit tabletop stations. Other geometry
    remains unchanged. Native compatibility and lighting qualification stay pending.
    """
    if len(sources) < 2 or len({s.raw['scene'] for s in sources}) != 1:
        raise ValueError('Search requires multiple source rooms in one building.')
    if len({s.raw['room'] for s in sources}) != len(sources):
        raise ValueError('Curated source groups must use distinct rooms.')
    objects, targets, groups, provenance = [], [], [], []
    for index, source in enumerate(sources):
        if source.small_task not in MECHANISMS:
            raise ValueError('Structural Enclosure and other categories stay independent.')
        raw, payload = source.raw, source.payload
        question = payload['question_data']
        render = question['render']
        resolved = render['resolved_objects']
        group = f'group_{index + 1}'
        public = public_areas[source.relative_path]
        source_targets = resolved.get('targets', [])
        if not source_targets:
            raise ValueError('An all-target source must have concrete target assets.')
        names = {row['name']: f'search_{index + 1}_{row["name"]}'
                 for rows in resolved.values() if isinstance(rows, list)
                 for row in rows if isinstance(row, dict) and 'name' in row}
        for role in ('containers', 'confusers', 'targets'):
            for row in resolved.get(role, []):
                position = row.get('position') or row.get('requested_position')
                if position is None or not row.get('category') or not row.get('model'):
                    raise ValueError('Incomplete source asset cannot enter the joint inventory.')
                native = {'type': 'DatasetObject', 'name': names[row['name']],
                    'category': row['category'], 'model': row['model'], 'position': position,
                    'orientation': row.get('quaternion_xyzw') or row.get('orientation') or [0., 0., 0., 1.],
                    'visual_only': True}
                # Preserve only explicitly specified source scale; no inferred resizing.
                if row.get('applied_scale') is not None or row.get('scale') is not None:
                    native['scale'] = row.get('applied_scale') or row.get('scale')
                objects.append({'native': native, 'source_identity': source.relative_path,
                    'source_object_name': row['name'], 'role': role,
                    'source_specification': deepcopy(row),
                    'container_native_name': names.get(row.get('container_name'))})
                if role == 'targets':
                    targets.append({'id': 'target_' + digest([source.relative_path, row['name']])[:16],
                        'binding': names[row['name']], 'group_id': group,
                        'expected_identity': row['category'], 'room': public['room'], 'area': public['area'],
                        'source_identity': source.relative_path, 'source_object_name': row['name']})
        groups.append({'id': group, 'mechanism': source.small_task,
            'room': public['room'], 'area': public['area'], 'clue': public['clue'],
            'target_count': len(source_targets), 'selection_rule': question['count_target'].replace('_', ' '),
            'source_identity': source.relative_path,
            'mechanism_source_annotations': deepcopy(question.get('case_metadata', {})),
            'source_render_contract': {key: deepcopy(render[key]) for key in
                ('room_bbox', 'multi_image_input', 'lighting', 'illumination') if key in render}})
        if source.small_task == 'Illumination Variability':
            groups[-1]['lighting_schedule'] = {'source_phases': ['normal', 'changed'],
                'source_intensities_and_timing': 'not recorded in cleaned source',
                'runtime_phase_duration_s': None, 'runtime_intensity_scale': None,
                'status': 'unavailable_until_versioned_schedule_is_declared_and_measured'}
        provenance.append({'source_identity': source.relative_path, 'source_sha256': sha256(source.path),
            'source_question': raw.get('question'), 'source_answer': raw.get('answer'),
            'source_payload': {**deepcopy(payload), 'room': raw['room']}})
    world = {'scene_model': sources[0].raw['scene'], 'required_rooms': [s.raw['room'] for s in sources],
        'load_policy': 'full_building_with_original_furniture_and_routes',
        'required_routes': 'all native connecting regions between required_rooms; no route pruning',
        'door_policy': 'retain_native_operable_doors', 'joint_inventory': objects,
        'source_provenance': provenance, 'native_compatibility_verified': False}
    from .task_edit2_search_layout import curate_world, primitive_configs
    curate_world(world, groups)
    world['search_groups'] = deepcopy(groups)
    bindings = {row['native']['name']: {k: row['native'][k] for k in ('name', 'category', 'model')}
                for row in objects}
    bindings.update({row['name']: {'name': row['name'], 'category': row['category']} for row in primitive_configs(world)})
    parameters = {'task_revision': REVISION, 'requires_native_visual_discovery': True,
        'targets': targets, 'groups': groups,
        'world_contract': world, 'required_native_gates': list(NATIVE_GATES),
        'required_target_count': len(targets), 'physical_collection_required': False,
        'native_adapter_contract': 'synchronized_informative_search_v1'}
    patches = [{'operation': 'category_initialization', 'phase': 'before_construction',
        'module': MODULE, 'parameters': {'world_contract': world}},
        {'operation': 'category_initialization', 'phase': 'after_postprocessing',
         'module': MODULE, 'parameters': {'operation': START_INITIALIZATION, 'world_contract': world}}]
    evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
        'binding_keys': list(bindings), 'parameters': parameters}
    public_groups = [{k: g[k] for k in ('id', 'room', 'area', 'clue', 'target_count', 'selection_rule')} for g in groups]
    objective = f'Find all {len(targets)} distinct targets in one continuous survey. Search areas: ' + '; '.join(
        f"{g['id']}: {g['target_count']} {g['selection_rule']} in {g['room']}, {g['area']}. {g['clue']}" for g in public_groups)
    if any(group['mechanism'] == 'Illumination Variability' for group in groups):
        objective += (' New pilot lighting schedule: 10 simulated seconds of original normal intensity, then '
            '10 simulated seconds at 0.10 intensity, repeating. Inspect the illumination search area in both phases. '
            'These pilot settings are newly declared and are not reconstructed source calibration.')
    public = {'objective': objective,
        'success_criteria': 'Observe and correctly declare every distinct target with no false declarations. '
            'Submit in the answer field a JSON-encoded string containing an object with a declarations list; each entry contains group_id, identity, '
            'room, area, snapshot_id, camera, and image_point [normalized horizontal, normalized vertical]. '
            'Point to the target in a delivered mounted image. Repeated sightings count once. '
            'Discovery requires informative visual evidence and a correct identity/location declaration.',
        'environment': 'One shared building with the original furniture, distractors and source mechanisms. '
            'The total count is disclosed; this variant measures all-target search rather than count estimation.',
        'constraints': ['Use legal robot controls and mounted observations.',
            'Preserve targets, distractors, furniture and source region boundaries.',
            'No grasping, lifting, collection or return to departure is required.']}
    task = physical_task(sources[0], context, public=public, evaluation=evaluation,
        bindings=bindings, patches=patches, capabilities=['base_translation', 'base_rotation'])
    scene = original_scene(sources[0], context, bindings=bindings, patches=patches)
    scene.setdefault('runtime', {})['requires_native_visual_discovery'] = True
    key = 'search_' + digest({'name': name, 'parameters': parameters, 'seed': context.seed})[:24]
    task['id'], scene['id'] = 'task_' + key, 'scene_' + key
    return {'candidate_id': key, 'task_definition': task, 'scene_definition': scene,
        'status': 'generated', 'execution_status': 'unavailable', 'final_selection': 'provisional',
        'public_groups': public_groups, 'world_contract': world,
        'source_to_parent': [{'source_identity': s.relative_path, 'parent_id': key, 'group_id': f'group_{i+1}',
                              'disposition': 'versioned_all_target_search'} for i, s in enumerate(sources)],
        'native_gates': {gate: 'pending' for gate in NATIVE_GATES},
        'mechanism_status': {g['id']: 'native_pending' for g in groups},
        'budget_status': 'pending_representative_native_execution',
        'runtime_adapter_status': 'shared_task_visual_collector_plus_reset_bound_native_audits_required',
        'capability_change': 'Disclosed total count; no inherited grasp/lift/return or independent count-estimation score.'}


def configure_scene(config, parameters):
    """Construct the complete joint inventory through an allowed scene patch."""
    world = parameters['world_contract']
    if config['scene']['scene_model'] != world['scene_model']:
        raise ValueError('Joint inventory belongs to another building.')
    config['scene'].pop('load_room_instances', None)
    config['scene'].pop('load_room_types', None)
    config['scene']['load_object_categories'] = None
    config['scene']['not_load_object_categories'] = None
    from .task_edit2_search_layout import primitive_configs
    config['objects'] = [deepcopy(row['native']) for row in world['joint_inventory']] + primitive_configs(world)
    return {'loaded_room_policy': world['load_policy'], 'required_rooms': world['required_rooms'],
        'inventory_names': [row['name'] for row in config['objects']],
        'native_compatibility_verified': False, 'mechanisms_verified': False}


class SearchEvaluator:
    """Deduplicate native identities and score final identity/location declarations."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters, self.output = config['parameters'], Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.targets = {t['id']: t for t in self.parameters['targets']}
        self.native_to_target = {config['bindings'][t['binding']]: t['id'] for t in self.targets.values()}
        self.receipts, self.declarations, self.last = {}, [], None
        self.submitted, self.submission_invalid = False, False
        self.lighting_phase_targets = {}
        self.lighting_views = {}

    def begin_episode(self, observation):
        self.receipts, self.declarations = {}, []
        self.submitted, self.submission_invalid = False, False
        self.lighting_phase_targets = {}
        self.lighting_views = {}
        return self.record(observation)

    def sample_step(self):
        cycle = getattr(self.session, '_search_light_cycle', None)
        if cycle is not None:
            cycle.sample()

    def report_answer(self, answer, observation):
        self.submitted = True
        try:
            value = json.loads(answer) if isinstance(answer, str) else answer
            declarations = value['declarations']
            if not isinstance(declarations, list) or any(not isinstance(d, dict) for d in declarations):
                raise ValueError('Declarations must be a list of records.')
            self.declarations, self.submission_invalid = deepcopy(declarations), False
        except (KeyError, TypeError, ValueError):
            self.declarations, self.submission_invalid = [], True
        return self.record(observation)

    def _measure(self, observation):
        if self.parameters.get('task_revision') in RUNTIME_REVISIONS:
            return measure_search_evidence(self.session, observation, self.config['bindings'], self.parameters)
        # Historical v1 controlled evidence remains callable without introducing
        # a new world-layout contract into an archived definition.
        provider = getattr(self.session, 'enumerative_search_evidence', None)
        if not callable(provider):
            return {'available': False, 'measurement_error': 'synchronized_informative_search_v1 adapter unavailable'}
        try:
            value = provider(observation, self.config['bindings'], self.parameters)
            if not isinstance(value, dict) or not isinstance(value.get('observations', []), list):
                raise ValueError('Search evidence must contain a mapping and observations list.')
            return value
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
            return {'available': False, 'measurement_error': f'{type(error).__name__}: {error}'}

    def record(self, observation, *, action=None):
        evidence = self._measure(observation)
        if not isinstance(evidence, dict):
            evidence = {'available': False, 'measurement_error': 'Native search evidence is not a mapping.'}
        delivered = set(observation.rgb_images())
        observations = evidence.get('observations', [])
        observations = observations if isinstance(observations, list) else []
        for row in observations:
            if not isinstance(row, dict):
                continue
            native_name = row.get('native_name')
            if not isinstance(native_name, str):
                continue
            target_id = self.native_to_target.get(native_name)
            valid = (target_id is not None and row.get('snapshot_id') == observation.snapshot_id
                and row.get('camera') in delivered and row.get('identity_synchronized') is True
                and row.get('native_rgb_matches_delivered') is True and row.get('informative') is True
                and row.get('measurement_available') is True and bool(row.get('informative_feature_evidence'))
                and row.get('location_verified') is True
                and all(_normal(row.get(field, '')) == _normal(self.targets[target_id][field])
                        for field in ('room', 'area')))
            region = row.get('image_region')
            if valid and isinstance(region, (list, tuple)) and len(region) == 4 and all(
                    isinstance(x, (float, int)) and not isinstance(x, bool) and 0 <= x <= 1 for x in region) and region[0] < region[2] and region[1] < region[3]:
                receipt_key = (row['snapshot_id'], row['camera'], target_id, tuple(region))
                self.receipts[receipt_key] = deepcopy(row)
                lighting = evidence.get('lighting', {})
                phase = lighting.get('measured_phase')
                group_id = self.targets[target_id]['group_id']
                group = next((g for g in self.parameters['groups'] if g['id'] == group_id), {})
                if group.get('mechanism') == 'Illumination Variability' and phase in ('normal', 'changed'):
                    self.lighting_phase_targets.setdefault(group_id, {}).setdefault(phase, set()).add(target_id)
        if isinstance(evidence.get('lighting'), dict) and evidence['lighting'].get('measured_phase') in ('normal','changed'):
            self.lighting_views[observation.snapshot_id] = {
                'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                'lighting': evidence['lighting'],
                'per_view_luminance': evidence.get('per_view_luminance_diagnostics', {})}
        verified, false, matched_declarations = set(), [], 0
        for declaration in self.declarations:
            matches = set()
            for (_, _, target_id, _), row in self.receipts.items():
                target = self.targets[target_id]
                point, region = declaration.get('image_point'), row['image_region']
                if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
                        not isinstance(x, (int, float)) or isinstance(x, bool) for x in point):
                    continue
                if (declaration.get('snapshot_id') == row['snapshot_id'] and declaration.get('camera') == row['camera']
                        and region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]
                        and all(_normal(declaration.get(field, '')) == _normal(target[field])
                                for field in ('group_id', 'room', 'area'))
                        and _normal(declaration.get('identity', '')) == _normal(target['expected_identity'])):
                    matches.add(target_id)
            if len(matches) == 1:
                verified.update(matches)
                matched_declarations += 1
            else:
                false.append(declaration)
        required = len(self.targets)
        gates = evidence.get('gates', {})
        gates = gates if isinstance(gates, dict) else {}
        mechanisms = evidence.get('mechanisms', {})
        mechanisms = dict(mechanisms) if isinstance(mechanisms, dict) else {}
        if self.parameters.get('task_revision') in RUNTIME_REVISIONS:
            for group in self.parameters['groups']:
                if group.get('mechanism') == 'Illumination Variability':
                    phases = self.lighting_phase_targets.get(group['id'], {})
                    mechanisms[group['id']] = bool(mechanisms.get(group['id']) is True
                        and phases.get('normal') and phases.get('changed'))
        available = (evidence.get('available') is True and all(gates.get(g) is True for g in NATIVE_GATES if g != 'constraints_preserved')
            and isinstance(gates.get('constraints_preserved'), bool)
            and all(mechanisms.get(g['id']) is True for g in self.parameters['groups']))
        recall = 100. * len(verified) / required
        precision = 100. * matched_declarations / len(self.declarations) if self.declarations else 0.
        result = {'schema': self.parameters.get('task_revision', REVISION),
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'evaluation_validity': 'complete' if available else 'unavailable',
            'required_distinct_targets': required, 'verified_distinct_targets': len(verified),
            'observed_distinct_targets': len({key[2] for key in self.receipts}),
            'discovery_percent': recall if available else None, 'diagnostic_discovery_percent': recall,
            'declaration_precision_percent': precision if available else None,
            'score_percent': recall if available else None,
            'completion_score': recall / 100. if available else None,
            'completion_items': [{'id': target_id, 'kind': 'verified_target_discovery',
                'completed': target_id in verified, 'evaluation_validity': 'valid' if available else 'unavailable'}
                for target_id in self.targets],
            'constraint_compliance': gates.get('constraints_preserved'),
            'false_positive_count': len(false) + int(self.submission_invalid), 'false_declarations': false,
            'missing_target_ids': sorted(set(self.targets) - verified),
            'per_group': {g['id']: {'required': g['target_count'], 'verified': sum(
                self.targets[t]['group_id'] == g['id'] for t in verified),
                'mechanism_validity': 'verified' if mechanisms.get(g['id']) is True else 'unavailable'}
                for g in self.parameters['groups']},
            'submitted': self.submitted, 'submission_invalid': self.submission_invalid,
            'task_success': bool(available and self.submitted and not self.submission_invalid
                                 and len(verified) == required and not false and gates.get('constraints_preserved') is True),
            'native_evidence': evidence, 'observation_receipt_count': len(self.receipts),
            'lighting_phase_target_ids': {group: {phase: sorted(ids) for phase, ids in phases.items()}
                for group, phases in self.lighting_phase_targets.items()},
            'lighting_observation_history': list(self.lighting_views.values())}
        self.last = result
        write_json(self.output / 'state.json', result)
        return result


def create_evaluator(session, controller, config, output):
    if config['operation'] != OPERATION:
        raise ValueError('Unsupported versioned search operation.')
    return SearchEvaluator(session, controller, config, output)


def postprocess_joint_world(session, parameters):
    """Own every namespaced source actor; never run a representative child reveal hook."""
    from .task_edit2_search_layout import pose, SearchLightCycle, initialize_search_tables
    world = parameters['world_contract']
    if session.scene_name != world['scene_model']:
        raise ValueError('Joint postprocess belongs to a different native world.')
    if getattr(session, '_search_joint_postprocessed', False):
        raise ValueError('Joint search postprocessing is initialization-only.')
    scene = session.env.scene
    registry = lambda name: scene.object_registry('name', name)
    rows = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    robots = set(id(obj) for obj in session.env.robots)
    actor_names = {item['native']['name'] for item in world['joint_inventory']}
    from .task_edit2_search_layout import primitive_configs
    added_names = {obj['name'] for obj in primitive_configs(world)}
    # Read the loaded background before touching only the explicit search actors.
    background = {obj.name: pose(obj) for obj in rows if id(obj) not in robots and obj.name not in actor_names | added_names}
    state_receipts = []
    for item in world['joint_inventory']:
        definition = item['native']; obj = registry(definition['name'])
        if obj is None or (obj.category, obj.model) != (definition['category'], definition['model']):
            raise ValueError('Joint source asset could not be resolved by its namespaced binding.')
        obj.visual_only = True
        source = item['source_specification']
        receipt = {'native_name': obj.name, 'source_identity': item['source_identity'],
            'source_object_name': item['source_object_name'], 'visual_only': bool(obj.visual_only),
            'source_pose_provenance': source['position'], 'adapted_initialization_pose': definition['position']}
        if source.get('open_state') is not None:
            from omnigibson.object_states import Open
            state = obj.states[Open]
            state.set_value(bool(source['open_state']))
            receipt['initial_open_state'] = bool(state.get_value())
            if receipt['initial_open_state'] != bool(source['open_state']):
                raise ValueError('Joint source initial open state was not restored.')
        if item.get('container_native_name'):
            from omnigibson.object_states import Inside
            container = registry(item['container_native_name'])
            if container is None:
                raise ValueError('A joint source container is missing.')
            state = obj.states[Inside]
            state.set_value(container, True)
            receipt['measured_inside_source_container'] = bool(state.get_value(container))
            if not receipt['measured_inside_source_container']:
                raise ValueError('The source initial membership could not be restored.')
            receipt['interior_geometry_verified'] = False
        from .task_edit2_search_layout import same_pose
        if not same_pose(pose(obj), definition):
            raise ValueError('Restoring a source state changed the authorized joint-world layout.')
        state_receipts.append(receipt)
    runtime_world, table_layout = initialize_search_tables(session, world)
    session._search_world_contract = runtime_world
    session._search_world_contract_binding = {'original_world_digest': digest(world),
        'reset_id': getattr(session, 'native_reset_id', None), 'runtime_layout_digest': digest(runtime_world)}
    session._search_initial_native = {name: pose(registry(name)) for name in actor_names | added_names}
    session._search_background_initial = background
    lighting_receipt = None
    lighting_groups = [g for g in world.get('search_groups', []) if g['mechanism'] == 'Illumination Variability']
    if lighting_groups:
        try:
            from .enumerative_perception_illumination_variability import _stage_lights
            lamps = {}
            for group in lighting_groups:
                for entry in group['mechanism_source_annotations']['source_entries']:
                    for lamp in entry['metadata']['lighting']['target_lights']:
                        lamps[lamp['name']] = lamp
            selected, measured_lamps = [], []
            for name, lamp in sorted(lamps.items()):
                obj = registry(name)
                if obj is None or obj.category != lamp['category']:
                    raise ValueError('An exact source illumination lamp is missing: ' + name)
                measured_lamps.append({'name': obj.name, 'category': obj.category, 'model': obj.model,
                                      'prim_path': obj.prim_path})
                selected.extend(_stage_lights(obj.prim))
            unique = {row['path']: row for row in selected}
            if not unique or not any(row['baseline'] > 0 for row in unique.values()):
                raise ValueError('Source lamp descendants have no positive measurable adjustable native light.')
            cycle = SearchLightCycle(session, measured_lamps, list(unique.values()))
            callback = MODULE + ':pilot_cycle:' + digest(world)[:16]
            session.pipeline.og.sim.add_physics_callback(callback, cycle.sample)
            session._search_light_cycle = cycle
            lighting_receipt = {'available': True, 'callback_name': callback, **cycle.measure()}
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, ImportError) as error:
            lighting_receipt = {'available': False, 'measurement_error': f'{type(error).__name__}: {error}'}
        session._search_lighting_initialization = lighting_receipt
    from .task_edit2_search_layout import same_pose
    if any(not same_pose(pose(registry(name)), row, .000001) for name, row in background.items()):
        raise ValueError('Joint search postprocess changed protected background geometry.')
    session._search_joint_postprocessed = True
    return {'joint_world_postprocess': True, 'world_digest': digest(world),
        'runtime_layout_digest': digest(runtime_world), 'native_table_layout_initialization': table_layout,
        'source_actor_initialization': state_receipts, 'source_hidden_link_operations': 0,
        'unrelated_furniture_modified': False, 'physics_steps': 0,
        'human_framing_names': sorted(actor_names | added_names), 'lighting': lighting_receipt,
        'visual_props_have_physical_support_contact_certificate': False,
        'native_floor_footprint_access_and_information_qualified': False}


def initialize_scene(session, parameters):
    """Run the owned start screen after robot binding, before any observation."""
    if parameters.get('operation') != START_INITIALIZATION:
        raise ValueError('Unsupported Search initialization operation.')
    state = vars(session)
    if state.get('_search_start_hook_called') or state.get('search_start_initialization') is not None:
        raise ValueError('Search start initialization hook cannot be repeated.')
    if (state.get('_search_joint_postprocessed') is not True or state.get('robot') is None
            or not isinstance(state.get('_search_initial_native'), dict)
            or not state['_search_initial_native']
            or not isinstance(state.get('_search_background_initial'), dict)
            or not isinstance(state.get('_search_world_contract_binding'), dict)):
        raise ValueError('Search start requires completed source postprocessing, native baselines and bound robot.')
    from .task_edit2_search_layout import resolve_runtime_world
    world = parameters['world_contract']
    resolve_runtime_world(session, world)
    from wia.tasks.search_start import initialize_search_start
    state['_search_start_hook_called'] = True
    # The helper retains failed receipts too; initialization acceptance does not
    # certify routes, legal public approaches or informative mounted images.
    return initialize_search_start(session, world)


def _gate_audits(session, world):
    """Read explicit reset-bound gate audits; missing records are never true."""
    from .task_edit2_search_layout import resolve_runtime_world
    try:
        resolve_runtime_world(session, world)
    except (ValueError, TypeError, KeyError):
        return {}, {}
    context = {'world_digest': digest(world), 'reset_id': getattr(session, 'native_reset_id', None)}
    gates, mechanisms = {}, {}
    if context['reset_id'] is None:
        return gates, mechanisms
    records = getattr(session, 'enumerative_search_audits', ())
    records = records if isinstance(records, (list, tuple)) else ()
    for row in records:
        if (not isinstance(row, dict) or row.get('schema') != 'search_native_gate_audit_v1'
                or row.get('context') != context or type(row.get('accepted')) is not bool
                or not row.get('native_evidence') or not row.get('reviewer') or not row.get('reviewed_at')):
            continue
        if row.get('gate') in NATIVE_GATES:
            gates[row['gate']] = row['accepted']
        if row.get('group_id') in {g['id'] for g in world.get('search_groups', [])}:
            mechanisms[row['group_id']] = row['accepted']
    return gates, mechanisms


def measure_search_evidence(session, observation, bindings, parameters):
    """Combine real structural measurements with the shared audited image collector."""
    from .task_edit2_search_layout import native_structure, resolve_runtime_world
    world = parameters['world_contract']
    structure = native_structure(session, world)
    audits, mechanisms = _gate_audits(session, world)
    checks = structure.get('checks', {})
    identity_checks = [v for k,v in checks.items() if k.startswith(('identity:', 'dimensions:', 'scale:'))]
    structural = {
        'joint_inventory': bool(structure['available'] and identity_checks and all(identity_checks)),
        'source_relations': bool(structure['available'] and structure.get('source_relations_preserved')
            and checks.get('native_table_layout_initialization', True)
            and all(v for k,v in checks.items() if k.startswith(('placement:', 'room:', 'table_', 'geometric_floor_foot_support:', 'visual_table_')))),
        'floor_supported_start': checks.get('floor_supported_start'),
        'constraints_preserved': (bool(structure['available'] and checks.get('background_unchanged')
            and structure.get('source_relations_preserved')) if structure['available'] else None),
    }
    gates = {**audits, **structural}
    # Loaded room labels are measured, but actual connecting routes still need
    # a reset-bound native route audit; category-filter metadata is insufficient.
    gates['loaded_routes'] = bool(checks.get('loaded_required_rooms') and audits.get('loaded_routes') is True)
    provider = getattr(session, 'task_visual_evidence', None)
    if not callable(provider):
        return {'available': False, 'observations': [], 'gates': gates, 'mechanisms': mechanisms,
            'native_structure': structure, 'measurement_error': 'task_visual_evidence adapter unavailable'}
    try:
        runtime_world, _ = resolve_runtime_world(session, world)
        visual_parameters = {**parameters, 'world_contract': runtime_world,
            'original_world_digest': digest(world), 'runtime_layout_digest': digest(runtime_world)}
        visual = provider(observation, bindings, visual_parameters)
        if not isinstance(visual, dict) or not isinstance(visual.get('objects', {}), dict):
            raise ValueError('Shared visual collector must return a mapping of native object receipts.')
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as error:
        return {'available': False, 'observations': [], 'gates': gates, 'mechanisms': mechanisms,
            'native_structure': structure, 'measurement_error': f'{type(error).__name__}: {error}'}
    rows = []
    for key, row in visual.get('objects', {}).items():
        if key not in bindings or not isinstance(row, dict):
            continue
        region = row.get('image_region', row.get('bbox_normalized'))
        visit = row.get('visit_evidence', {})
        visit = visit if isinstance(visit, dict) else {}
        rows.append({**row, 'image_region': region, 'location_verified': bool(
            row.get('location_verified') is True or row.get('visit_verified') is True
            and visit.get('native_location_verified') is True)})
    gates['mounted_camera_audit'] = bool(visual.get('all_delivered_views_audited') is True
        and audits.get('mounted_camera_audit') is True)
    gates['legal_informative_access'] = audits.get('legal_informative_access')
    cycle = getattr(session, '_search_light_cycle', None)
    lighting = cycle.measure() if cycle is not None else getattr(session, '_search_lighting_initialization', {}) or {}
    luminance = {}
    if cycle is not None:
        import numpy as np
        for camera, rgb in observation.rgb_images().items():
            values = np.asarray(rgb)
            if camera in getattr(session, 'robot_camera_registry', {}) and values.ndim == 3 and values.shape[-1] >= 3:
                luminance[camera] = {'mean_linear_weighted_RGB_0_255': float(np.mean(
                    values[...,:3].astype(float) @ np.array([.2126,.7152,.0722]))),
                    'snapshot_id': observation.snapshot_id, 'phase': lighting.get('measured_phase')}
    if any(g['mechanism'] == 'Illumination Variability' for g in parameters['groups']):
        for group in parameters['groups']:
            if group['mechanism'] == 'Illumination Variability':
                mechanisms[group['id']] = bool(mechanisms.get(group['id']) is True
                    and lighting.get('measured_phase') in ('normal','changed')
                    and lighting.get('original_light_transforms_preserved') is True
                    and lighting.get('environmental_driver_error') is None
                    and lighting.get('cycle_running') is True)
    return {'available': bool(structure['available'] and visual.get('available') is True),
        'observations': rows, 'gates': gates, 'mechanisms': mechanisms,
        'native_structure': structure, 'visual_collector': visual, 'lighting': lighting,
        'per_view_luminance_diagnostics': luminance}


def qualify_operation(session, executor, config, bound):
    """Native entry point: unresolved measurements reject qualification explicitly."""
    try:
        from .task_edit2_search_layout import native_structure
        if config.get('operation') != OPERATION or module_identity(config.get('module')) != MODULE:
            raise ValueError('Search qualification requires the owned versioned operation.')
        world = config['parameters']['world_contract']
        structure = native_structure(session, world)
        audits, mechanisms = _gate_audits(session, world)
        checks = dict(structure.get('checks', {}))
        checks['exact_bound_native_inventory'] = (set(bound) == set(config['binding_keys']) and all(
            bound[key] is session.env.scene.object_registry('name', config['bindings'][key]) for key in bound))
        checks['guarded_base_controls'] = bool(executor.capabilities.base_translation and executor.capabilities.base_rotation)
        checks['owned_joint_postprocess_completed'] = getattr(session, '_search_joint_postprocessed', False) is True
        for gate in ('loaded_routes', 'mounted_camera_audit', 'legal_informative_access'):
            checks['native_audit:' + gate] = audits.get(gate) is True
        for group in config['parameters']['groups']:
            checks['mechanism_access_audit:' + group['id']] = mechanisms.get(group['id']) is True
        if any(g['mechanism'] == 'Illumination Variability' for g in config['parameters']['groups']):
            cycle = getattr(session, '_search_light_cycle', None)
            measured = cycle.measure() if cycle is not None else {}
            checks['native_pilot_cycle_installed'] = (measured.get('measured_phase') in ('normal','changed')
                and measured.get('original_light_transforms_preserved') is True and measured.get('cycle_running') is True)
        accepted = bool(structure.get('available') is True and checks and all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_structure': structure,
            'reason': None if accepted else 'native_joint_search_preconditions_unavailable_or_failed',
            'target_contact_or_graspability_qualified': False, 'executed_search_verified': False}
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError, ImportError) as error:
        return {'accepted': False, 'checks': {}, 'reason': 'native_joint_search_measurements_unavailable',
            'measurement_error': f'{type(error).__name__}: {error}', 'executed_search_verified': False}


def restore_search_lighting(session):
    cycle = getattr(session, '_search_light_cycle', None)
    return cycle.restore() if cycle is not None else {'restored_original_intensities': False, 'reason': 'cycle_unavailable'}

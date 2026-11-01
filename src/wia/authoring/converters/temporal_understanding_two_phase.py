"""Upgrade original Unobserved Change snapshots into a continuous two-phase task.

Run CPU validation from the repository root with the behavior environment:
python -m pytest -q tests/test_unobserved_two_phase_conversion.py
Native initialization reuses the original snapshot restoration. Temporal
transition, observation evidence and scoring belong to wia.native.esi_temporal_phases.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

import copy
import importlib
import json

from wia.authoring.converters import temporal_understanding_unobserved_change as legacy
from wia.tasks.temporal_policy import SURVEY_DEFAULTS, survey_policy


MODULE = 'wia.authoring.converters.temporal_understanding_two_phase'
OPERATION = 'observe_two_phase_change'
WARNING_CODES = ('external_source_phase_transition', 'native_temporal_execution_unverified')
CAPABILITIES = ('base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper')
PHASE_2_GOAL_PARAMETERS = {'phase_2_minimum_base_travel_m': .25,
    'phase_2_container_visit_radius_m': 1.0, 'phase_2_navigation_required': True}


def physical_target_keys(parameters):
    """Return only declared movable targets, including compound event obligations."""
    operations = parameters.get('physical_operations')
    return [row['physical_target_key'] for row in operations] if operations is not None else [
        parameters['physical_target_key']] if parameters.get('physical_target_key') else []


class TemporalInitializationError(RuntimeError):
    """Fail initialization with structured provenance instead of hiding source lids."""
    def __init__(self, stage, reason, parameters, **evidence):
        self.details = {'schema': 'unobserved_two_phase_initialization_failure_v1',
            'accepted': False, 'failure_stage': stage, 'failure_reason': reason,
            'source_identity': parameters.get('source_identity'), 'temporal_episode': True,
            'phase': 'phase_1', **evidence}
        super().__init__(json.dumps(self.details, sort_keys=True))


def _source_parameters(source, layout):
    raw, question = layout['raw'], layout['payload']['question_data']
    options = raw.get('options_json')
    if isinstance(options, str):
        options = json.loads(options)
    if options is None:
        options = question.get('options')
    if (not isinstance(options, list) or not options or
            any(not isinstance(option, str) or not option for option in options)):
        raise ValueError('Original public answer options are unresolved.')
    if raw['answer'] != question['answer'] or raw['answer'] not in options:
        raise ValueError('Original source answer must agree with its saved question and options.')
    metadata = question.get('question_metadata') or {}
    target = metadata.get('target_box_index')
    if target is None:
        target = metadata.get('changed_box_index')
    indices = {box.box_index for box in layout['plan'].boxes}
    if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target not in indices):
        raise ValueError('Saved target box index is not an original source box.')
    return {'source_task_type': layout['plan'].task_type, 'public_options': copy.deepcopy(options),
            'original_answer': raw['answer'],
            'box_labels': {f'container_{box.box_index:02d}': box.position_label for box in layout['plan'].boxes},
            'target_box_index': target}


def _public_task(source, options, policy=None):
    budget = survey_policy(policy)["phase_1_max_decisions"]
    return {
        'objective': source.raw['question'] + '\n\n'
            f'Phase 1 is read-only: an active survey with at most {budget} Agent decisions. Observe all eight mounted cameras. '
            'You may move or turn the base and adjust the torso to inspect; arm and gripper commands are disabled. '
            'Do not command arm or gripper motion; incidental navigation contact is diagnostic only. '
            'When ready, set survey_complete=true with answer=null and command DONE '
            'to request the unobserved transition; plain DONE also requests transition. The runner switches '
            'automatically when the survey budget is exhausted. Phase 2: use the existing guarded base controls to approach '
            'and inspect at least one original container after the change. Move at least 0.25 m from the '
            'Phase-1 base position and arrive within 1.0 m of any original container. '
            'Use the existing guarded arm and gripper controls only as needed to answer in the same episode. '
            'Then submit the selected option in answer and choose DONE. '
            'Original answer options: ' + json.dumps(options, ensure_ascii=False),
        'success_criteria': 'Complete both observation phases. In Phase 2, move the base at least 0.25 m from '
            'its Phase-1 position and approach within 1.0 m of at least one original container for inspection, '
            'then correctly answer the original question using one of the original options. '
            'Base the comparison on your actual observations from this episode; answering without the '
            'required Phase-2 base operation does not complete the task.',
        'environment': 'The original room and boxes are retained. A source-defined external change occurs '
            'between the two observation phases while it is unobserved. The robot, cameras and model episode '
            'continue across the transition; no removal action is shown to you.',
        'constraints': [f'Phase 1 allows at most {budget} decisions including invalid or refused decisions; '
            'use base and torso motion only. Arm and gripper commands are disabled. '
            'The survey decisions count toward the total episode budget.',
            'Phase 2 permits existing guarded base, arm, rotation, trunk and gripper controls for inspection '
            'and opening when needed. The base approach and minimum 0.25 m movement are required; '
            'choose any original container within the 1.0 m arrival radius, with no fixed route or mandatory grasp.',
            'Keep the original boxes, contents and task dependencies in place. Phase 1 disables arm and gripper commands; '
            'Phase 2 permits legal arm handling of noncritical native movable furniture.',
            'Use only delivered robot camera observations and your own phase-1 memory.']}


def upgrade_existing(scene_definition, task_definition, source):
    """Return independent scene/task copies for the same original continuous episode.

    This CPU-only upgrade never reruns asset availability or map/start selection.
    Legacy saved_image1/saved_image2 variants remain unchanged diagnostics.
    """
    if (source.big_task, source.small_task) != legacy.CategoryConverter.category:
        raise ValueError('A two-phase upgrade requires an original Unobserved Change source.')
    layout = legacy._layout(source)
    additional = _source_parameters(source, layout)
    scene, task = copy.deepcopy(scene_definition), copy.deepcopy(task_definition)
    if task.get('kind') != 'category_robot_task':
        raise ValueError('A generated original category task is required for upgrade.')
    evaluation = task['private']['evaluation']
    if (evaluation.get('type') != 'category_operation' or
            module_identity(evaluation.get('module')) not in (legacy.MODULE, MODULE) or
            evaluation.get('operation') not in (*legacy.OPERATIONS, OPERATION)):
        raise ValueError('Only an owned original snapshot or two-phase operation may be upgraded.')
    if task['private']['bindings'] != layout['bindings'] or scene.get('bindings') != layout['bindings']:
        raise ValueError('The existing definitions must preserve all original source bindings.')
    if scene['scene']['variants']['adapted'] != task['scene']['variants']['adapted']:
        raise ValueError('Scene and task adapted initialization must agree before upgrade.')
    for definition in (scene, task):
        if definition.get('source') != {'provider': 'esi', 'json': source.relative_path}:
            raise ValueError('The upgrade cannot change the existing original source identity.')
    already_upgraded = module_identity(evaluation['module']) == MODULE and evaluation['operation'] == OPERATION
    navigation_goal_current = already_upgraded and all(evaluation['parameters'].get(key) == value
        for key, value in PHASE_2_GOAL_PARAMETERS.items())
    content_keys = set(layout['phase_keys']['image1'] + layout['phase_keys']['image2'])
    for definition in (scene, task):
        patches = definition['scene']['variants']['adapted']['patches']
        upgraded, hook_phases = [], set()
        for patch in patches:
            if patch['operation'] == 'object_physics' and patch.get('binding') in content_keys:
                continue
            if patch['operation'] == 'category_initialization' and module_identity(patch.get('module')) in (legacy.MODULE, MODULE):
                parameters = patch['parameters']
                if (parameters.get('source_identity') != source.relative_path or
                        parameters.get('scene_model') != layout['raw']['scene'] or
                        parameters.get('room') != layout['raw']['room'] or
                        parameters.get('bindings') != layout['bindings']):
                    raise ValueError('Existing initialization provenance conflicts with the original source.')
                patch['module'] = MODULE
                parameters.update(copy.deepcopy(additional), phase='image1', temporal_episode=True)
                hook_phases.add(patch['phase'])
            upgraded.append(patch)
        if hook_phases != {'before_construction', 'after_postprocessing'}:
            raise ValueError('The continuous episode requires both original initialization hooks.')
        definition['scene']['variants']['adapted']['patches'] = upgraded
        if not already_upgraded:
            definition['version'] += 1
    evaluation.update(module=MODULE, operation=OPERATION)
    evaluation['parameters'].update(copy.deepcopy(additional), **PHASE_2_GOAL_PARAMETERS, **survey_policy(evaluation['parameters']),
        temporal_episode=True, initial_phase='image1')
    if already_upgraded and not navigation_goal_current:
        task['version'] += 1
    task['public'] = _public_task(source, additional['public_options'], evaluation['parameters'])
    task['requirements']['capabilities'] = list(dict.fromkeys([
        *task['requirements']['capabilities'], *CAPABILITIES]))
    return scene, task


def configure_scene(config, parameters):
    """Configure fixed collidable source contents before native construction only."""
    if parameters.get('phase') != 'image1' or parameters.get('temporal_episode') is not True:
        raise ValueError('A continuous temporal episode must initialize the original first phase.')
    evidence = legacy.configure_scene(config, parameters)
    if parameters.get('survey_revision') == 'free_survey_10_v1':
        config['scene']['load_room_instances'] = [parameters['room']]
    names = {parameters['bindings'][key]['name'] for phase in legacy.PHASES
             for key in parameters['phase_keys'][phase]}
    for obj in config['objects']:
        if obj.get('name') in names:
            obj.update(fixed_base=True, visual_only=False)
    if parameters.get('task_edit2_physical'):
        from .temporal_understanding_agent_observation import apparatus_objects
        for target_key in physical_target_keys(parameters):
            if target_key not in parameters['phase_keys']['image2']:
                continue
            target_name = parameters['bindings'][target_key]['name']
            next(row for row in config['objects'] if row['name'] == target_name)['fixed_base'] = False
        additions = apparatus_objects(parameters['apparatus'])
        names = {row['name'] for row in config['objects']}
        if names.intersection(row['name'] for row in additions):
            raise ValueError('Temporal apparatus cannot overwrite source objects.')
        config['objects'].extend(additions)
    return {**evidence, 'temporal_episode': True, 'phase': 'phase_1', 'source_phase': 'image1',
        'inactive_phase_policy': 'Original distant preload; inactive actors stay outside Agent observations.',
        'warning_codes': list(WARNING_CODES)}


def initialize_scene(session, parameters):
    """Restore original containers once, open supported lids, then install phases."""
    if parameters.get('phase') != 'image1' or parameters.get('temporal_episode') is not True:
        raise TemporalInitializationError('phase_configuration', 'original_phase_1_required', parameters)
    try:
        evidence = legacy.initialize_scene(session, parameters)
    except Exception as error:
        raise TemporalInitializationError('source_initialization', 'legacy_source_initialization_failed', parameters,
            exception_summary=f'{type(error).__name__}: {error}') from error
    from omnigibson.object_states import Open
    opened = []
    for key in parameters['container_keys']:
        obj = session.env.scene.object_registry('name', parameters['bindings'][key]['name'])
        try:
            supported = Open in obj.states
            if supported:
                state = obj.states[Open]
                # The new survey keeps every lid fully open: a random partially
                # open flap can satisfy Open while still hiding all contents.
                requested = (state.set_value(True, fully=True)
                    if parameters.get('survey_revision') == 'free_survey_10_v1'
                    else state.set_value(True))
                is_open = bool(state.get_value())
                if requested is False or not is_open:
                    raise ValueError('Native Open did not reach the requested first-phase state.')
                for joint in state.relevant_joints_info[1]:
                    for link in obj.links.values():
                        if link.prim_path == joint.body1:
                            link.visible = True
            else:
                is_open = None
            opened.append({'binding': key, 'open_supported': supported, 'open': is_open})
        except Exception as error:
            raise TemporalInitializationError('phase_1_opening', 'original_lid_open_initialization_failed', parameters,
                container_binding=key, exception_summary=f'{type(error).__name__}: {error}') from error
    if parameters.get('task_edit2_physical'):
        for key in physical_target_keys(parameters):
            if key not in parameters['phase_keys']['image2']:
                continue
            target = session.env.scene.object_registry('name', parameters['bindings'][key]['name'])
            if target.fixed_base or not callable(getattr(target, 'disable_gravity', None)):
                raise TemporalInitializationError('dynamic_preload', 'native_dynamic_gravity_adapter_unavailable', parameters)
            target.disable_gravity()
            target.keep_still()
    try:
        installed = importlib.import_module('wia.native.esi_temporal_phases').install_temporal_phases(session, parameters)
    except Exception as error:
        raise TemporalInitializationError('temporal_installation', 'continuous_phase_installation_failed', parameters,
            exception_summary=f'{type(error).__name__}: {error}') from error
    return {**evidence, 'temporal_episode': True, 'phase': 'phase_1', 'source_phase': 'image1',
        'initial_open_states': opened, 'temporal_phase_installation': installed,
        'human_framing_names': [parameters['bindings'][key]['name'] for key in
            parameters['container_keys'] + parameters['phase_keys']['image1']],
        'historical_agent_observations_created': False, 'warning_codes': list(WARNING_CODES)}


def qualify_operation(session, executor, config, bound):
    """Delegate measured temporal qualification without importing the simulator early."""
    return importlib.import_module('wia.native.esi_temporal_phases').qualify_operation(session, executor, config, bound)


def create_evaluator(session, controller, config, output):
    """Delegate the continuous observation/history scorer to its shared owner."""
    return importlib.import_module('wia.native.esi_temporal_phases').create_evaluator(session, controller, config, output)


def on_interaction_initialized(session, evidence):
    """Rebase both source phases after accepted placement, before Agent images."""
    controller = getattr(session, 'temporal_phase_controller', None)
    if controller is None:
        raise ValueError('Temporal placement requires the initialized continuous controller.')
    if evidence.get('accepted') is not True:
        failures = [{'bindings': row.get('bindings'),
            'reason': row.get('evidence', {}).get('reason'),
            'failure_stage': row.get('evidence', {}).get('failure_stage'),
            'last_attempt': row.get('evidence', {}).get('attempts', [])[-1:],
            'support_search': row.get('evidence', {}).get('support_search'),
            'initial_support_contacts': row.get('evidence', {}).get('initial_support_contacts'),
            'height_screen_before': row.get('evidence', {}).get('height_screen_before'),
            'layout_policy': row.get('evidence', {}).get('layout_policy', evidence.get('layout_policy')),
            'spawn_grid': {key: row.get('evidence', {}).get('spawn_grid', {})[key]
                           for key in ('requested_source_region', 'measured_source_region')
                           if key in row.get('evidence', {}).get('spawn_grid', {})}}
            for row in evidence.get('per_group', []) if not row.get('evidence', {}).get('accepted')]
        raise ValueError('Temporal support placement failed: ' + json.dumps({
            'reason': evidence.get('reason'), 'groups': failures}, default=str))
    return controller.rebase_interaction(evidence)


class CategoryConverter(legacy.CategoryConverter):
    category = legacy.CategoryConverter.category
    version = 'unobserved_two_phase_v4.0'

    def convert(self, source, context):
        if context.config.get('task_edit2_temporal') is True:
            return build_task_edit2_change(source, context)
        result = CategoryConverter.convert_existing(self, source, context)
        task = result.get('task_definition')
        if task is not None:
            from .partial_occlusion import TASK_REVISION
            parameters = task['private']['evaluation']['parameters']
            parameters['task_revision'] = TASK_REVISION
            parameters['mechanism_contract'] = {
                'preserved_relations': ['source temporal event', 'same-session observation continuity',
                    'bounded active read-only first phase', 'existing budgets and movement criteria'],
                'required_native_evidence': ['informative source view access in both phases',
                    'exact native event transition', 'all eight mounted-camera phase observations'],
                'movement_boundary_sensitivity': 'Existing thresholds retained; near-boundary outcomes require review.',
                'native_operation_verified': False}
            result.setdefault('warning_codes', []).append('both_phase_informative_access_pending')
            result['temporal_policy_unchanged'] = True
        return result

    def convert_existing(self, source, context):
        continuous_context = copy.copy(context)
        continuous_context.config = {**context.config, 'continuous_temporal_accessibility': True}
        result = super().convert(source, continuous_context)
        if result.get('status') != 'generated' or result.get('task_definition') is None:
            return result
        try:
            scene, task = upgrade_existing(result['scene_definition'], result['task_definition'], source)
        except (ValueError, KeyError, TypeError) as error:
            return {**result, 'status': 'conversion_failed', 'task_definition': None,
                'failure_stage': 'temporal_episode_upgrade', 'failure_reason': 'original_two_phase_upgrade_failed',
                'diagnostic_reason': {'type': type(error).__name__, 'message': str(error)},
                'warning_codes': ['source_only_diagnostic', *WARNING_CODES]}
        result.update(scene_definition=scene, task_definition=task, operation_recipe=OPERATION,
            initial_phase='image1', temporal_episode=True,
            warning_codes=['assumption_review_required', *WARNING_CODES],
            change_summary=['Reuse the original room, assets, placements, robot and cameras in one two-phase episode.',
                'Start with fixed collidable original phase-1 contents and visible open supported lids.',
                'Require a Phase-2 base approach: move at least 0.25 m and visit within 1.0 m of any original container before answering.',
                'Retain independent saved phase variants as legacy diagnostics; score only actual episode observations.'],
            assumptions=[{'doubt': 'Independent saved snapshots do not establish a legitimate Agent observation history.',
                'chosen_assumption': 'Use one continuous robot/camera/model episode, an explicit unobserved '
                    'source-defined phase transition and measured observation history; never supply private phase descriptions.',
                'alternatives': ['Retain independent legacy snapshots as diagnostics when native initialization fails.'],
                'scope': source.relative_path,
                'risk': 'Native opening, sensor coverage, phase switching and temporal answers remain unverified.'}],
            limitations=['CPU conversion does not certify native opening, phase observations, transition or answer quality.',
                'Original contents remain fixed at saved placements; Phase 2 requires a base approach and inspection, with no mandatory grasp.',
                'No historical Agent image or memory is fabricated; the shared temporal controller must measure both phases.'],
            next_step='Validate one bounded native continuous episode before any broader conversion or publication.')
        layout = legacy._layout(source)
        groups = [[container, *[key for key in layout['phase_keys']['image1']
            if layout['relations'][key] == container]] for container in layout['container_keys']]
        for definition in (scene, task):
            for patch in definition['scene']['variants']['adapted']['patches']:
                if patch['operation'] == 'category_initialization' and module_identity(patch.get('module')) == MODULE:
                    patch['parameters']['accessible_support_rebase_required'] = True
        task['private']['evaluation']['parameters']['accessible_support_rebase_required'] = True
        result['interaction_policy'] = {'mode': 'inspection',
            'group_bindings': [key for group in groups for key in group],
            'manipulation_bindings': list(layout['container_keys']),
            'support_bindings': list(layout['container_keys']),
            'relocation_policy': 'independent_groups', 'placement_groups': groups,
            'support_search': 'measured_surfaces', 'allow_fallback_table': True,
            'source_camera_policy': 'preserve', 'source_region': layout['raw']['room'],
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85,
            'warning_codes': ['accessible_temporal_container_placement', 'native_temporal_support_unverified']}
        result['change_summary'][0] = 'Reuse the original room, assets, robot and cameras in one two-phase episode with accessible container groups.'
        result['change_summary'].append('Place each original container with its active Phase-1 content on an accessible measured support when needed.')
        result['change_summary'].append('Rebase both phase plans from accepted per-container native translations before the first Agent observation; inactive actors remain parked.')
        result['limitations'][1] = 'Original assets and dimensions remain fixed; accepted accessible placements replace low source heights in both phases before any Agent observation.'
        result['warning_codes'].extend(['accessible_layout_differs_from_original_reference', 'native_temporal_support_unverified',
            'source_content_container_clearance_requires_native_check'])
        if result.get('robot_initialization_evidence', {}).get('pending_shared_spawn_repair'):
            result['warning_codes'].append('source_map_start_pending_shared_native_repair')
            result['assumptions'].append({'doubt': 'The conservative source map offers no qualified R1 pose.',
                'chosen_assumption': 'Emit no guessed robot_pose. Require the shared Native initializer to select '
                    'an actually floor-supported, collision-free stance in the original room after accessible placement.',
                'alternatives': ['Retain a diagnostic when actual Native same-room spawn repair has no valid candidate.'],
                'scope': source.relative_path, 'risk': 'Pending CPU generation does not establish a safe Native start or complete observation coverage.'})
        task['public']['environment'] += ' Portable containers may start on accessible supports; both phases use that same initialized layout.'
        return result


PHYSICAL_OPERATION = 'compare_change_and_restore_or_collect'


def build_task_edit2_change(source, context):
    """Generate a source-supported event-dependent physical branch, separately versioned."""
    from wia.tasks.temporal_policy import TASK_EDIT2_TEMPORAL_REVISION
    from .temporal_understanding_agent_observation import apparatus_objects
    layout = legacy._layout(source)
    recipe = legacy.source_change_recipe(layout, allow_multiple=context.config.get('allow_multiple_changes', False))
    apparatus = context.config.get('change_apparatus')
    destination = context.config.get('change_destination')
    spares = context.config.get('change_spares', [])
    if apparatus is None or destination is None:
        raise ValueError('Physical temporal variants need an explicit table/tray apparatus and public destination.')
    result = CategoryConverter().convert_existing(source, context)
    task, scene = result.get('task_definition'), result.get('scene_definition')
    if task is None:
        return result
    p = task['private']['evaluation']['parameters']
    branch = recipe['branch']
    bindings = copy.deepcopy(task['private']['bindings'])
    additions = apparatus_objects(apparatus)
    for obj in additions:
        binding = {key: obj[key] for key in ('name', 'category')}
        if obj.get('model'):
            binding['model'] = obj['model']
        bindings[obj['name']] = binding
    if branch == 'removed':
        original = layout['bindings'][recipe['original_key']]
        matching = [spare for spare in spares if all(spare['binding'].get(key) == original[key] for key in ('category', 'model'))]
        if len(matching) != 1:
            raise ValueError('Removed branch requires exactly one matching category/model spare with public inventory.')
        target_key = matching[0]['binding_key']
        if target_key not in bindings:
            raise ValueError('Matching spare must be part of the actual constructed inventory.')
        destination_key, relation = recipe['source_container'], 'inside'
    elif branch in ('added', 'replaced'):
        target_key = recipe['current_key']
        destination_key, relation = destination['binding_key'], destination.get('relation', 'on_top')
    elif branch == 'moved':
        target_key, destination_key, relation = recipe['current_key'], recipe['source_container'], 'inside'
    else:
        target_key, destination_key, relation = None, None, None
    physical = {'task_revision': TASK_EDIT2_TEMPORAL_REVISION, 'requires_native_visual_discovery': True, 'task_edit2_physical': True,
        'physical_branch': branch, 'source_event_recipe': recipe, 'physical_target_key': target_key,
        'physical_destination_key': destination_key, 'placement_relation': relation,
        'apparatus': apparatus, 'public_spare_inventory': spares,
        'phase_2_navigation_required': False, 'release_stability_steps': 15,
        'maximum_release_speed_m_s': .02, 'maximum_release_angular_speed_rad_s': .1,
        'budget_status': 'pending_representative_comparison_and_branch_execution',
        'native_visual_adapter': 'task_visual_evidence'}
    if context.config.get('survey_revision') == 'free_survey_10_v1':
        physical.update(phase_1_max_decisions=10, phase_1_allow_early_transition=True,
            phase_1_auto_transition=True, require_complete_phase_1_coverage=False,
            survey_revision='free_survey_10_v1')
    if branch == 'multiple':
        operations = []
        for event in recipe['events']:
            if event['branch'] == 'removed':
                original = layout['bindings'][event['original_key']]
                matches = [spare for spare in spares if all(
                    spare['binding'].get(key) == original[key] for key in ('category', 'model'))]
                if len(matches) != 1:
                    raise ValueError('Every removal needs one matching spare in the complete public inventory.')
                target, destination_binding, placement = matches[0]['binding_key'], event['source_container'], 'inside'
            else:
                target, destination_binding, placement = event['current_key'], destination['binding_key'], 'on_top'
            operations.append({'physical_target_key': target,
                'physical_destination_key': destination_binding, 'placement_relation': placement,
                'event_branch': event['branch']})
        physical['physical_operations'] = operations
    p.update(copy.deepcopy(physical))
    # Remove obsolete numeric rules in this explicit revision; legacy paths retain them.
    p.pop('phase_2_minimum_base_travel_m', None)
    p.pop('phase_2_container_visit_radius_m', None)
    task['private'].update(bindings=bindings)
    task['private']['evaluation'].update(operation=PHYSICAL_OPERATION, binding_keys=list(bindings))
    scene['bindings'] = copy.deepcopy(bindings)
    for definition in (scene, task):
        for patch in definition['scene']['variants']['adapted']['patches']:
            if patch['operation'] == 'category_initialization' and module_identity(patch.get('module')) == MODULE:
                patch['parameters'].update(copy.deepcopy(physical), bindings=copy.deepcopy(bindings))
    scene.setdefault('runtime', {})['requires_native_visual_discovery'] = True
    task['public'] = _physical_public_task(source, p, destination)
    task['version'] += 1
    scene['version'] += 1
    result.update(task_definition=task, scene_definition=scene, operation_recipe=PHYSICAL_OPERATION,
        task_revision=TASK_EDIT2_TEMPORAL_REVISION, physical_branch=branch,
        source_identity=source.relative_path, execution_status='unavailable', final_selection='provisional',
        temporal_policy_unchanged=False,
        change_summary=['Preserve the source event and bounded read-only Phase 1 in one continuous episode.',
            'Replace generic minimum movement/any-container approach with actual change-dependent physical obligations.',
            'Declare source-faithful dynamic Phase-2 target and reachable apparatus proposals; retain native pending gates.'],
        limitations=['Native source/container interior fit, target support, hidden dynamic preloading and public-control grasp remain pending.',
            'No source observation, event qualification, physical feasibility or autonomous success is fabricated.'],
        native_gates={key: 'pending' for key in ('phase_1_informative_access', 'protected_read_only_survey',
            'exact_hidden_event', 'apparatus_support_and_access', 'phase_2_informative_access',
            'bound_selection', 'stable_terminal_relation', 'source_interior_fit')})
    return result


def _physical_public_task(source, p, destination):
    budget = survey_policy(p)['phase_1_max_decisions']
    # All branch recipes are public; the selected branch and changed object remain private.
    survey_instruction = ('freely inspect comparison locations' if
        p.get('require_complete_phase_1_coverage') is False else 'inspect every comparison location')
    objective = (source.raw['question'] + '\n\n'
        f'Phase 1: {survey_instruction} in a read-only survey of at most {budget} decisions. '
        'Use mounted cameras and base/torso motion; arm and gripper commands are disabled. Incidental navigation contact does not end the survey. '
        'Set survey_complete=true with answer=null and command DONE when ready; the event occurs unobserved in this same episode. '
        'Phase 2: compare with memory, perform the applicable operation, then submit the original option in answer with command DONE. Required operation: '
        f'if an object was added, put the actual added object on {destination["public_label"]}; '
        f'if replaced, put the current replacement on {destination["public_label"]}; '
        'if moved, return the displaced object to its remembered original container/support; '
        'if removed, select the matching category and model by appearance from the public spare inventory and '
        'replenish the remembered original location; if unchanged, inspect all comparison locations with informative '
        'views and leave the contents in place. No grasp is required for no-change.')
    if p.get('survey_revision') == 'free_survey_10_v1' or p.get('physical_operations'):
        objective += ' If several locations changed, complete the applicable operation for each change.'
    if p.get('require_complete_phase_1_coverage') is False:
        objective += (' You may finish Phase 1 before visiting every location; the phase switches automatically '
            'after ten decisions. Survey coverage is recorded but is not a transition or full-success gate. '
            'The spare tray contains the complete candidate inventory from the beginning in every event variant.')
    if p.get('public_spare_inventory'):
        objective += ' Public spare inventory: ' + '; '.join(
            spare['public_label'] + ': ' + spare['public_category'].replace('_', ' ')
            for spare in p['public_spare_inventory']) + '. The spare table is visible and reachable during both phases.'
    objective += ' Original answer options: ' + json.dumps(p['public_options'])
    return {'objective': objective,
        'success_criteria': 'Use informative observations in both phases, report the correct original comparison answer, '
            'and complete the applicable change-dependent operation. Manipulation requires real selection of the '
            'history-bound object and stable released placement with native destination support/contact. Preserve '
            'all other source contents, spare distractors, containers and furniture. Inspection-only no-change '
            'requires informative comparison-location views, never a fabricated pickup.',
        'environment': 'One native reset with the exact source event and same robot/cameras/history throughout. '
            'Declared collection/inspection tray and spare-table apparatus are initialized before the survey.',
        'constraints': [f'Phase 1 is bounded to {budget} decisions and strictly read-only, including the spare inventory.',
            'The hidden event delivers no intermediate frames. Use only public guarded controls afterward.',
            'There is no extra minimum travel distance, arbitrary any-container visit or return-to-start requirement.',
            'Do not move any source object other than the comparison-selected target; preserve apparatus and distractors.']}

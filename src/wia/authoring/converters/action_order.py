"""Preserve Action Order assets and measure an ordered physical stack.

Run the complete category through ActionOrderConverter().convert_all(sources,
ConversionContext(output_root)); the shared campaign runner owns GPU capture.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

import copy
import math
import re
import json
from pathlib import Path

import numpy as np

from wia.authoring.conversion import (
    ConverterBase, map_records, original_scene, physical_task,
)


_THREE_ROLES = re.compile(
    r"^Which (?P<base>[\w ]+?) should go under the (?P<middle>[\w ]+?) "
    r"as its base, and which (?P<top>[\w ]+?) should go on top of the "
    r"(?P<top_middle>[\w ]+?)\?", re.IGNORECASE,
)
_TWO_ROLES = re.compile(
    r"^Which (?P<base>[\w ]+?) should (?:you place|go) under the "
    r"(?P<middle>[\w ]+?) as its base", re.IGNORECASE,
)
_OBJECT_BINDINGS = {
    'middle': 'obj_fixed',
    'base_candidate_0': 'obj_A_correct',
    'base_candidate_1': 'obj_A_wrong',
    'top_candidate_0': 'obj_C_correct',
    'top_candidate_1': 'obj_C_wrong',
}
MODULE = 'core.esi_converters.action_order'
OPERATION = 'ordered_grasp_lift_stack'
TASK_REVISION = 'task_edit_20261018_v1'
MECHANISM_CONTRACT = {'goal': 'execute_supplied_support_order', 'preserve': ['original_candidates', 'asset_scales', 'intermediate_stable_support'], 'inference_claimed': False}
RULE_REVISION = 'measured_independent_candidate_supports_v3'
INITIALIZATION = 'measured_rigid_row_startup_v2'
SUPPORT_TRANSFER_REVISION = 'support_transfer_v1'


def _public_roles(question):
    """Extract category roles from public text without reading answer metadata."""
    match = _THREE_ROLES.match(question) or _TWO_ROLES.match(question)
    return match.groupdict() if match else None


def _category(value):
    return value.strip().lower().replace(' ', '_')


def _finite_vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(
        isinstance(item, (int, float)) and math.isfinite(item) for item in value
    )


class ActionOrderConverter(ConverterBase):
    category = ('Action Sequencing', 'Action Order Inference')
    version = '2.5.0_task_edit_20261018_v1'

    def sample(self, manifest, policy=None):
        """Cover feedback failures, ambiguous QA and generator size-mode strata."""
        policy = dict(policy or {})
        previous_sources = set(policy.pop('previous_source_failures', ()))
        fallback = super().sample(manifest, policy)
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        selected, strata = [], set()
        failures = [row for row in rows if row['status'] != 'generated']
        priority = failures[:1] + [row for row in rows if row['source_identity'] in previous_sources]
        priority += [row for row in rows if 'source_qa_direction_ambiguity' in row.get('warning_codes', [])][:1]
        by_id = {row['candidate_id']: row for row in rows}
        for row in priority + rows + [by_id[key] for key in fallback]:
            if len(selected) >= len(fallback):
                break
            stratum = tuple(row.get('sampling_stratum', ()))
            if row['candidate_id'] not in selected and (row in priority or stratum not in strata):
                selected.append(row['candidate_id'])
                strata.add(stratum)
        for key in fallback:
            if len(selected) >= len(fallback):
                break
            if key not in selected:
                selected.append(key)
        return selected

    def inspect(self, source):
        raw, payload = source.raw, source.payload
        if (source.big_task, source.small_task) != self.category or (
                raw.get('big_task'), raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'category_mismatch'}
        if payload.get('task') != 'action_order_inference':
            return {'eligible': False, 'reason': 'source_task_mismatch'}
        count = payload.get('n_objects')
        if count not in (2, 3):
            return {'eligible': False, 'reason': 'unsupported_stack_role_count'}
        roles = _public_roles(raw.get('question', ''))
        if not roles or bool(roles.get('top')) != (count == 3):
            return {'eligible': False, 'reason': 'public_stack_roles_not_recognized'}
        hierarchy = payload.get('hierarchy', {})
        for public_role, metadata_role in (
                ('base', 'obj_A_cat'), ('middle', 'obj_fixed_cat'), ('top', 'obj_C_cat')):
            if public_role == 'top' and count == 2:
                continue
            if _category(roles[public_role]) != hierarchy.get(metadata_role):
                return {'eligible': False, 'reason': 'public_private_category_mismatch'}
        if count == 3 and _category(roles['top_middle']) != _category(roles['middle']):
            return {'eligible': False, 'reason': 'public_middle_reference_mismatch'}
        if hierarchy.get('relation_A') != 'on_top_of' or (
                count == 3 and hierarchy.get('relation_C') != 'on_top_of'):
            return {'eligible': False, 'reason': 'non_stacking_source_relation'}
        objects = map_records(payload.get('objects'))
        poses = map_records(payload.get('initial_poses'))
        required = list(_OBJECT_BINDINGS.values())[:3 if count == 2 else 5]
        for name in required:
            info, pose = objects.get(name, {}), poses.get(name, {})
            if not info.get('model') or not (info.get('cat') or info.get('category')):
                return {'eligible': False, 'reason': f'missing_original_asset:{name}'}
            scale = info.get('scale')
            scale_values = [scale] if isinstance(scale, (int, float)) else scale
            if not isinstance(scale_values, (list, tuple)) or len(scale_values) not in (1, 3) or any(
                    not isinstance(item, (int, float)) or not math.isfinite(item) or item <= 0
                    for item in scale_values):
                return {'eligible': False, 'reason': f'invalid_original_scale:{name}'}
            if not _finite_vector(pose.get('position'), 3) or not _finite_vector(
                    pose.get('quaternion_xyzw'), 4) or not any(pose['quaternion_xyzw']):
                return {'eligible': False, 'reason': f'missing_original_pose:{name}'}
        if not payload.get('orbital_geometry') or not payload.get('qa_per_view') or not payload.get(
                'stage0_initial_views'):
            return {'eligible': False, 'reason': 'missing_original_reference_view'}
        return {'eligible': True, 'reason': 'complete_original_stacking_scene'}

    def convert(self, source, context):
        raw, payload = source.raw, source.payload
        roles = _public_roles(raw['question'])
        count = payload['n_objects']
        base, middle = (roles[name].replace('_', ' ') for name in ('base', 'middle'))
        objective = (f'Choose one {base} as the base for B (the {middle}). Grasp B, lift it '
            'at least 5 cm, then place and release it on the chosen base until stable.')
        environment = f'The original room contains two {base} candidates and one {middle} called B.'
        if count == 3:
            top = roles['top'].replace('_', ' ')
            objective += (f' Then choose one {top}, grasp and lift it at least 5 cm, and place '
                'and release it on top of B until the whole stack is stable.')
            environment += f' There are two {top} candidates.'
        public = {
            'objective': 'Follow this supplied order. ' + objective,
            'success_criteria': 'Either original candidate of each category is acceptable when '
                'the chosen objects genuinely form the requested support order after the ordered '
                'grasp, lift and placement operations. The freestanding stack must remain in '
                'supporting contact with low motion for at least 0.5 seconds after release.',
            'environment': environment,
            'constraints': ['Use robot motion and gripper interaction to move objects.',
                'Finish placing B before grasping the chosen top object.' if count == 3 else
                    'Complete a real grasp, lift and placement of B.',
                'Recover unsuccessful placements with robot grasping and retry as needed.',
                'Leave all grippers released and clear of the finished stack.'],
        }
        objects = map_records(payload['objects'])
        bindings = {
            key: {'name': name, 'category': objects[name].get('cat') or objects[name]['category'],
                'model': objects[name]['model']}
            for key, name in _OBJECT_BINDINGS.items()
            if count == 3 or not key.startswith('top_')
        }
        interaction_policy = {
            'mode': 'manipulation', 'group_bindings': list(bindings),
            # Bases also need usable starting height even though the ordered
            # operation does not require a separate grasp of the chosen base.
            'manipulation_bindings': list(bindings),
            'relocation_policy': 'independent_groups', 'source_region': raw.get('room'),
            'placement_groups': [[key] for key in bindings],
            'source_camera_policy': 'preserve',
            'support_search': 'measured_surfaces',
            'allow_fallback_table': True,
            'minimum_grasp_height_m': 0.35, 'maximum_grasp_height_m': 1.10,
            'maximum_stance_distance_m': 0.85,
        }
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
            'binding_keys': list(bindings), 'parameters': {
                'middle_key': 'middle', 'base_keys': ['base_candidate_0', 'base_candidate_1'],
                'top_keys': ['top_candidate_0', 'top_candidate_1'] if count == 3 else [],
                'minimum_lift_m': 0.05, 'minimum_stability_duration_s': 0.5,
                'maximum_linear_speed_m_s': 0.02, 'maximum_angular_speed_rad_s': 0.1}}
        evaluation['parameters'].update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        assumptions = [{
            'doubt': 'The selected feedback sample has floor objects and a reported off-map robot '
                'start. Saved evidence does not establish reachable native manipulation across the category.',
            'chosen_assumption': 'Use the authorized pending initialization rule: measure an existing scene '
                'support first, or a simple fallback table when existing furniture cannot fit, '
                'for each independent candidate, and screen a collision-free floor stance. '
                'Preserve each asset scale and orientation; initial candidate spacing is not a support-order invariant. '
                'Keep the source reference camera unchanged and robot-mounted sensors on their native '
                'mounts. Accept either original base and top only through the unchanged native '
                'grasp/lift/order/release/support predicate. Keep selected identities in private QA.',
            'alternatives': ['Retain the source group only if measured native placement screens '
                'accept it with a valid robot stance.', 'Record an explicit native placement '
                'failure if no authorized support and stance satisfy the unchanged-size rule.'],
            'scope': 'All instances in this category.',
            'risk': 'Height and stance-distance screens do not prove IK, approach clearance, '
                'grasp acquisition or stack stability. Relocation changes the background context '
                'and requires human Review; generated status does not establish native success.',
        }, {
            'doubt': 'The previous selected native sample rejected the bounded support search '
                'before placement; historical tabletop support used a different initialization '
                'protocol and does not establish a working current stance or placement.',
            'chosen_assumption': 'Use the measured_surfaces search for each independent candidate on same-room '
                'original or authorized simple supports, without a whole-row footprint requirement. '
                'Keep local upward-surface, footprint, contact and native floor-stance checks. '
                'Restore rejected proposals; retain an independently verified floor start if '
                'all support proposals fail. No operation credit is earned at initialization.',
            'alternatives': ['Retain the previous bounded search and its explicit rejection.',
                'Keep only the translation search and its explicit native rejection.'],
            'scope': 'All category instances requiring initialization-only support relocation.',
            'risk': 'More sampled surfaces do not prove that a suitable support exists. '
                'Independent starting positions change the historical visual layout; base approach, '
                'IK, contact, sensor coverage and grasp execution remain unqualified.',
        }, {
            'doubt': 'Source too-large bases and too-small tops can still be physically viable; '
                'the generator labels use scripted XY fit rather than unique native feasibility.',
            'chosen_assumption': 'Accept either original candidate through the same public '
                'support-order operation and independent measured predicate. Preserve the '
                'source-selected identities and direction answers solely in auxiliary private QA.',
            'alternatives': ['Define a separately declared source-selected recipe with a legitimate '
                'public reference and selection rule after resolving the initial QA contract.'],
            'scope': 'All physical category tasks; original QA is unchanged.',
            'risk': 'A physically valid alternative stack need not answer the original direction '
                'question correctly. Physical success must never imply original QA correctness.',
        }, {
            'doubt': 'The active builder can select an orbital view different from the original '
                'source question/reference, and the physical episode does not execute auxiliary QA.',
            'chosen_assumption': 'Preserve the original question, answers and reference metadata '
                'privately. Keep initial QA execution unverified and do not rescore the original '
                'answer against a different view or expose a human reference image as an Agent sensor.',
            'alternatives': ['Integrate a separately declared legitimate pre-action observation '
                'and reference contract in the shared QA runner without changing source answers.'],
            'scope': 'Auxiliary initial QA for every category instance; physical predicates are independent.',
            'risk': 'QA storage preservation does not establish reference compatibility, Agent '
                'visibility or original-question accuracy. A common camera translation cannot '
                'resolve a pre-existing reference mismatch.',
        }, {
            'doubt': 'A duration and low-motion tolerance are needed to define stable placement; '
                'native manipulation has not calibrated them for this category.',
            'chosen_assumption': 'Declare 0.5 seconds of consecutive controller-step readbacks '
                'at the native controller dt, with speeds at most 0.02 m/s and 0.1 rad/s. '
                'Do not impose a universal local-axis upright condition on round source objects.',
            'alternatives': ['Calibrate a different declared duration/tolerance in a later native '
                'revision with comparable budgets.', 'Keep unavailable readback cases diagnostic.'],
            'scope': 'All current category-operation candidates.',
            'risk': 'Sampled controller-time stability does not certify every physics substep '
                'or long-term mechanical stability.',
        }]
        warnings = ['assumption_review_required', 'native_stack_execution_unverified',
            'initial_qa_reference_integration_unverified', 'l2_manipulation_placement_pending',
            'support_relocation_authorized', 'native_stance_and_grasp_unverified',
            'measured_surface_search_native_unverified', 'physical_candidate_choice_differs_from_source_qa',
            'independent_candidate_support_adaptation_requires_review',
            'source_qa_reference_layout_changed', 'independent_robot_spawn_native_unverified']
        qa = payload.get('qa', {})
        duplicate_roles = [role for role in ('A', 'C') if qa.get(f'rel_{role}_correct') is not None
            and qa.get(f'rel_{role}_correct') == qa.get(f'rel_{role}_wrong')]
        if duplicate_roles:
            warnings.append('source_qa_direction_ambiguity')
            assumptions.append({
                'doubt': 'Original QA assigns the same direction to both candidates for role(s): '
                    + ', '.join(duplicate_roles) + '.',
                'chosen_assumption': 'Preserve the original QA verbatim and use the public category '
                    'roles for physical candidate selection without supplying an answer direction.',
                'alternatives': ['Keep this as a diagnostic-only item if human Review rejects the '
                    'direction-free physical task.', 'Request source QA clarification separately.'],
                'scope': source.relative_path,
                'risk': 'Original direction-answer QA is ambiguous; its score must remain separate '
                    'from physical stack success.',
            })
        qualification = {'on_failure': 'skip', 'initial_visibility': 'none',
            'target_movable_collidable': True}
        # Construction enables native dynamics; the later patch also handles
        # source postprocessing that might reset visual_only. No success pose is set.
        patches = [
            {'operation': 'object_physics', 'phase': 'before_construction',
                'binding': key, 'visual_only': False, 'fixed_base': False}
            for key in bindings
        ] + [
            {'operation': 'object_physics', 'phase': 'after_postprocessing',
                'binding': key, 'visual_only': False}
            for key in bindings
        ]
        initialization = {'operation': INITIALIZATION, 'scene_model': raw['scene'],
            'room': raw['room'], 'bindings': bindings, 'interaction_policy': interaction_policy,
            'source_provenance': {'source_json': source.relative_path,
                'builder': 'ESI-Bench/src/active_explore/tasks/action_sequencing/action_order_inference.py',
                'historical_adapter': 'core/sim/esi_action_order_tabletop.py',
                'rule_revision': RULE_REVISION}}
        patches.insert(len(bindings), {'operation': 'category_initialization',
            'phase': 'before_construction', 'module': MODULE, 'parameters': initialization})
        patches.append({'operation': 'category_initialization', 'phase': 'after_postprocessing',
            'module': MODULE, 'parameters': initialization})
        return {
            'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
            'task_definition': physical_task(source, context, public=public,
                evaluation=evaluation, bindings=bindings, patches=patches, qualification=qualification),
            'status': 'generated', 'reuse_level': 'L2',
            'task_revision': TASK_REVISION, 'mechanism_contract': MECHANISM_CONTRACT,
            'rule_revision': RULE_REVISION,
            'sampling_stratum': [objects['obj_A_wrong'].get('mode', 'unspecified'),
                objects.get('obj_C_wrong', {}).get('mode', 'not_applicable')],
            'interaction_policy': interaction_policy,
            'change_summary': ['Preserve original assets, scales, candidate bindings and public roles.',
                'Declare initialization-only independent robot stance search and existing-or-authorized-fallback support '
                'repair for each independent candidate; preserve identity, scale and orientation.',
                'Enable original movable collidable objects before construction and after '
                    'source postprocessing, then use the existing measured-surface support search.',
                'Add an ordered grasp/lift/release stack operation accepting any original candidates.',
                'Revoke interrupted support credit at both controller and observation boundaries; '
                    'reject additional supports under the upper stack members.',
                'Keep original QA private and separate from native physical success.'],
            'warning_codes': warnings, 'assumptions': assumptions,
            'limitations': ['No native reach, grasp or stack experiment was run by this converter.',
                'Placement is pending native initialization: a measured support and floor '
                    'stance require measured clearance and support evidence; impossible cases '
                    'must remain explicit failures.',
                'Source height/AABB and stance-distance screens do not establish actual IK, '
                    'grasp acquisition or released-stack feasibility.',
                'Independent candidate placement changes the source visual layout; source QA remains '
                    'private historical evidence. Preserve the source reference camera and mounted sensors.',
                'Historical support and IK evidence is representative, not category-wide qualification.',
                'Source scripted placement and fit labels do not establish released native stacking.',
                'Initial QA must use its declared source reference; the historical final assessment '
                    'found 23/77 source packets disagree with active selected-view scoring. '
                    'Shared QA envelope integration is outside this converter.',
                'Unknown native grasp, attachment, contact, support or controller-clock evidence '
                    'is conservatively rejected; native compatibility is not established on CPU.',
                'The final shared interaction receipt screens the already adapted state. The owned '
                    'initialization evidence and private native log retain the complete original-to-adapted '
                    'transform, rejected proposals and independent diagnostic robot start.',
                'Pending task execution requires independent qualification and human Review.'],
            'attempted_repairs': ['Declare reversible object-physics restoration and independent candidate '
                'supports through the single final measured-surface initializer; legacy rigid rotation '
                'proposals use the guarded trial API. No native placement was executed by this converter.'],
            'next_step': 'Publish every pending candidate and available historical diagnostic '
                'to Review through the existing category runner. Retain explicit native placement '
                'failures and obtain native stance, sensor, grasp and released-stack evidence '
                'before any readiness claim. This CPU revision does not launch capture workers.',
        }


def configure_scene(config, parameters):
    """Check the original construction; do not replace assets or robot physics."""
    if parameters.get('operation') != INITIALIZATION:
        raise ValueError('Unsupported Action Order initialization.')
    robots = config.get('robots', [])
    if len(robots) != 1 or robots[0].get('type') != 'R1':
        raise ValueError('Action Order requires its original installed R1.')
    objects = {obj['name']: obj for obj in config.get('objects', [])}
    for binding in parameters['bindings'].values():
        obj = objects.get(binding['name'])
        if obj is None or any(obj.get(key) != binding[key] for key in ('category', 'model')):
            raise ValueError('Initialization must retain every exact original asset.')
    return {'source_provenance': parameters['source_provenance'],
        'original_assets_and_robot_configuration_preserved': True,
        'native_initialization_pending': True, 'task_actions': 0}


def _row_yaws(group, supports):
    """Measure the original row and propose only existing support-axis rotations."""
    from scipy.spatial.transform import Rotation

    points = np.stack([_array(obj.get_position_orientation()[0], (3,)) for obj in group.values()])
    pivot = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points[:, :2] - pivot[:2], full_matrices=False)
    if singular[0] <= 1e-6:
        raise ValueError('The original native row has no measurable planar axis.')
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    heading = math.atan2(axis[1], axis[0])
    choices = [{'yaw_rad': 0., 'support_axis_from': None}]
    for obj in supports:
        _, quaternion, extent, _ = obj.get_base_aligned_bbox()
        extent = _array(extent, (3,))
        rotation = Rotation.from_quat(_array(quaternion, (4,))).as_matrix()
        for index in sorted((0, 1), key=lambda i: (-extent[i], i)):
            direction = rotation[:2, index]
            if np.linalg.norm(direction) <= 1e-6:
                continue
            yaw = (math.atan2(direction[1], direction[0]) - heading + math.pi / 2) % math.pi - math.pi / 2
            if not any(abs(yaw - row['yaw_rad']) < math.radians(1.) for row in choices):
                choices.append({'yaw_rad': yaw, 'support_axis_from': obj.name})
    return pivot, choices[:9]


def _safe_diagnostic_spawn(session, group, policy):
    """Search a native floor start even when no tabletop can fit the row."""
    from wia.tasks import interaction_setup as interaction
    from omnigibson.utils.usd_utils import PoseAPI

    scene, sim, robot = session.env.scene, session.pipeline.og.sim, session.env.robots[0]
    names = {obj.name for obj in group.values()}
    world, floor_z, grid = interaction._navigation_cells(session, robot, policy['source_region'], names)
    floors = [obj.prim_path for obj in interaction._objects(scene) if obj.category in ('floor', 'floors')]
    if not floors:
        raise ValueError('Native floor bodies are required for an independent robot start.')
    target = np.mean([_array(obj.get_position_orientation()[0], (3,)) for obj in group.values()], axis=0)
    order = np.argsort(np.linalg.norm(world - target[:2], axis=1), kind='stable')[:32]
    baseline = sim.dump_state(serialized=False)
    protected = {obj.name: (obj, tuple(interaction.array(value).copy() for value in obj.get_position_orientation()))
        for obj in interaction._objects(scene) if obj is not robot and not obj.fixed_base and not obj.visual_only}
    evidence = {'accepted': False, 'spawn_grid': grid, 'attempts': [], 'robot_actions': 0}
    try:
        for index in order:
            sim.load_state(baseline, serialized=False)
            PoseAPI.invalidate()
            interaction._place_robot(session, robot, world[index], target, floor_z)
            checks = interaction._robot_contact_check(robot, floors, floor_z)
            checks['original_objects_and_background_preserved'] = all(
                interaction._pose_unchanged(obj, pose, np.zeros(3), .005) for obj, pose in protected.values())
            attempt = {'robot_position_m': interaction.array(robot.get_position_orientation()[0]).tolist(),
                'robot_quaternion_xyzw': interaction.array(robot.get_position_orientation()[1]).tolist(),
                'checks': checks}
            evidence['attempts'].append(attempt)
            if all(checks[key] for key in ('floor_supported', 'collision_free', 'original_objects_and_background_preserved')):
                evidence.update(accepted=True, robot_spawn=attempt)
                return evidence
    finally:
        if not evidence['accepted']:
            sim.load_state(baseline, serialized=False)
            PoseAPI.invalidate()
        session.invalidate_camera_cache()
    evidence['reason'] = 'No measured collision-free same-room diagnostic stance passed the bounded search.'
    return evidence


def _initialization_record(evidence):
    """Retain private diagnostics when capture skips task qualification saving."""
    print('[action-order-initialization] ' + json.dumps(evidence, allow_nan=False), flush=True)
    return evidence


def initialize_scene(session, parameters):
    """Restore interaction access once, before controls or evaluator history exist.

    Reuse the native support/stance initializer for every rotated proposal. A
    rejected proposal restores its full baseline, including source camera pose.
    The later shared patch independently screens the resulting final layout.
    """
    from scipy.spatial.transform import Rotation
    from wia.tasks import interaction_setup as interaction
    from wia.tasks.schema import Binding
    from omnigibson.utils.usd_utils import PoseAPI
    import torch

    if parameters.get('operation') != INITIALIZATION or (
            session.scene_name, session.room_name) != (parameters['scene_model'], parameters['room']):
        raise ValueError('Initialization belongs to a different source scene or rule.')
    if hasattr(session, '_action_order_initialization'):
        raise ValueError('Action Order restoration is initialization-only and runs once.')
    evidence = {'schema': INITIALIZATION, 'source_provenance': parameters['source_provenance'],
        'accepted': False, 'robot_actions': 0, 'physical_grasp_verified': False,
        'attempts': [], 'warning_codes': ['assumption_review_required', 'native_stack_execution_unverified']}
    if parameters.get('initialization_review_contract') is not None:
        evidence.update(initialization_review_contract=copy.deepcopy(parameters['initialization_review_contract']),
            candidate_inspection_verified=False,
            candidate_inspection_status='Pending native sensor coverage and human review; footprint clearance alone is insufficient.')
    session._action_order_initialization = evidence
    scene, sim = session.env.scene, session.pipeline.og.sim
    group = {}
    for key, binding in parameters['bindings'].items():
        obj = scene.object_registry('name', binding['name'])
        if obj is None or any(getattr(obj, field) != binding[field] for field in ('category', 'model')):
            raise ValueError('Every original row asset must resolve exactly.')
        if obj.visual_only or obj.fixed_base:
            raise ValueError('Original row physics must be enabled by declared object_physics patches.')
        group[key] = obj
    policy = parameters['interaction_policy']
    native_bindings = {key: Binding.model_validate(binding) for key, binding in parameters['bindings'].items()}
    if policy['relocation_policy'] == 'independent_groups':
        # The following shared patch owns the single final initialization. No
        # provisional placement may overwrite its original reference geometry.
        evidence.update(deferred_to_shared_initializer=True,
            placement_groups=copy.deepcopy(policy['placement_groups']),
            source_camera_policy='preserve',
            reason='Independent candidates require measured accessible supports; final initialization is pending.')
        return _initialization_record(evidence)
    try:
        evidence['independent_robot_spawn'] = _safe_diagnostic_spawn(session, group, policy)
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        evidence['independent_robot_spawn'] = {'accepted': False, 'reason': f'{type(error).__name__}: {error}'}
    # A rejected support search returns to this safe stance, rather than the
    # off-map source robot preset. Object and camera baselines remain original.
    baseline = sim.dump_state(serialized=False)
    poses = {obj.name: tuple(interaction.array(value).copy() for value in obj.get_position_orientation())
        for obj in group.values()}
    scales = {obj.name: interaction.array(obj.scale).copy() for obj in group.values()}
    camera = (interaction.array(session.scene_camera_position).copy(),
        interaction.array(session.scene_camera_quaternion).copy())
    viewer = copy.deepcopy(getattr(session, 'postprocess', {}).get('viewer_camera'))
    framing = session.scene_camera_framing

    def restore():
        sim.load_state(baseline, serialized=False)
        PoseAPI.invalidate()
        session.scene_camera_position, session.scene_camera_quaternion = (value.copy() for value in camera)
        session.scene_camera_framing = framing
        if viewer is not None:
            session.postprocess['viewer_camera'] = copy.deepcopy(viewer)
        session.invalidate_camera_cache()

    try:
        supports = [obj for obj in interaction._objects(scene) if obj.name not in poses and not obj.visual_only
            and any(word in obj.category for word in ('table', 'counter', 'desk', 'bench', 'shelf', 'cabinet'))
            and policy['source_region'] in (getattr(obj, 'in_rooms', ()) or ())]
        supports.sort(key=lambda obj: (float(np.linalg.norm(np.mean([interaction.array(value) for value in obj.aabb], axis=0)[:2]
            - np.mean([pose[0][:2] for pose in poses.values()], axis=0))), obj.name))
        pivot, choices = _row_yaws(group, supports)
        evidence.update(rotation_pivot_m=pivot.tolist(), rotation_proposals=choices,
            original_geometry={name: {'position_m': pose[0].tolist(), 'quaternion_xyzw': pose[1].tolist(),
                'scale': scales[name].tolist()} for name, pose in poses.items()})
        for proposal in choices:
            restore()
            rotation = Rotation.from_euler('z', proposal['yaw_rad'])
            for obj in group.values():
                position, quaternion = poses[obj.name]
                obj.set_position_orientation(position=torch.as_tensor(rotation.apply(position - pivot) + pivot, dtype=torch.float32),
                    orientation=torch.as_tensor((rotation * Rotation.from_quat(quaternion)).as_quat(), dtype=torch.float32))
                obj.keep_still()
            session.scene_camera_position = rotation.apply(camera[0] - pivot) + pivot
            session.scene_camera_quaternion = (rotation * Rotation.from_quat(camera[1])).as_quat()
            if viewer is not None and viewer.get('position') is not None:
                session.postprocess['viewer_camera']['position'] = (rotation.apply(np.asarray(viewer['position']) - pivot) + pivot).tolist()
                for key in ('quaternion_xyzw', 'orientation'):
                    if viewer.get(key) is not None:
                        session.postprocess['viewer_camera'][key] = (rotation * Rotation.from_quat(viewer[key])).as_quat().tolist()
            result = interaction.initialize_interaction_trial(session, native_bindings, policy)
            evidence['attempts'].append({'proposal': proposal, 'accepted': result['accepted'],
                'reason': result.get('reason'), 'support_search': result.get('support_search', []),
                'placement_attempts': result.get('attempts', [])})
            if not result['accepted']:
                continue
            if not all(np.array_equal(interaction.array(obj.scale), scales[obj.name]) for obj in group.values()):
                raise ValueError('Initialization changed an original asset scale.')
            delta = interaction.array(result.get('group_translation_m', [0., 0., 0.]))
            if not all(np.linalg.norm(interaction.array(obj.get_position_orientation()[0])
                    - rotation.apply(poses[obj.name][0] - pivot) - pivot - delta) <= .015
                    and (Rotation.from_quat(interaction.array(obj.get_position_orientation()[1])).inv()
                        * rotation * Rotation.from_quat(poses[obj.name][1])).magnitude() <= .035
                    for obj in group.values()):
                raise ValueError('Initialization did not preserve the common original row transform.')
            if (not np.allclose(interaction.array(session.scene_camera_position),
                    rotation.apply(camera[0] - pivot) + pivot + delta, atol=1e-6, rtol=0.) or
                    (Rotation.from_quat(interaction.array(session.scene_camera_quaternion)).inv()
                        * rotation * Rotation.from_quat(camera[1])).magnitude() > 1e-6):
                raise ValueError('The original reference camera did not follow the exact row transform.')
            evidence.update(accepted=True, native_support_and_stance=result,
                transform={'pivot_m': pivot.tolist(), 'yaw_rad': proposal['yaw_rad'], 'translation_m': delta.tolist()},
                source_camera_before={'position_m': camera[0].tolist(), 'quaternion_xyzw': camera[1].tolist()},
                source_camera_after={'position_m': interaction.array(session.scene_camera_position).tolist(),
                    'quaternion_xyzw': interaction.array(session.scene_camera_quaternion).tolist()},
                source_camera_transform_matches_objects=True, object_scales_unchanged=True)
            session.scene_camera_framing = 'action_order_common_rigid_reference'
            return _initialization_record(evidence)
        evidence['reason'] = 'No identity or measured support-axis rigid row proposal passed native initialization.'
        evidence['warning_codes'].append('native_rigid_row_initialization_failed')
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        evidence.update(reason=f'{type(error).__name__}: {error}')
        evidence['warning_codes'].append('native_rigid_row_initialization_failed')
    finally:
        if not evidence['accepted']:
            restore()
    evidence['diagnostic_robot_start_retained'] = evidence['independent_robot_spawn']['accepted']
    return _initialization_record(evidence)


def _array(value, shape):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite native geometry/velocity readback is required.')
    return result


def _under(path, root):
    return path == root or path.startswith(root + '/')


def _validate_operation(session, config, bound=None):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') != OPERATION or (config.get('scene'), config.get('room')) !=
            (session.scene_name, session.room_name)):
        raise ValueError('Use the declared operation in its original native scene and room.')
    p = config['parameters']
    keys = [p['middle_key'], *p['base_keys'], *p['top_keys']]
    if (p['middle_key'] != 'middle' or p['base_keys'] != ['base_candidate_0', 'base_candidate_1'] or
            p['top_keys'] not in ([], ['top_candidate_0', 'top_candidate_1']) or
            set(config['binding_keys']) != set(keys) or len(set(keys)) != len(keys)):
        raise ValueError('Bind every original candidate without selecting a private answer.')
    manipulation_revision = p.get('manipulation_revision')
    if manipulation_revision not in (None, SUPPORT_TRANSFER_REVISION):
        raise ValueError('Unsupported ordered-stack manipulation revision.')
    support_transfer = manipulation_revision == SUPPORT_TRANSFER_REVISION
    if support_transfer and ('minimum_lift_m' not in p or p['minimum_lift_m'] is not None):
        raise ValueError('Support transfer requires an explicit null minimum_lift_m.')
    # These are declared task thresholds, not native feasibility measurements.
    if any(p.get(key) != value for key, value in {
            'minimum_lift_m': None if support_transfer else 0.05, 'minimum_stability_duration_s': 0.5,
            'maximum_linear_speed_m_s': 0.02, 'maximum_angular_speed_rad_s': 0.1}.items()):
        raise ValueError('Unsupported ordered-stack threshold revision.')
    names = config['bindings']
    if set(names) != set(keys) or len(set(names.values())) != len(keys):
        raise ValueError('All candidates must resolve to distinct original native objects.')
    objects = {}
    for key in keys:
        obj = session.env.scene.object_registry('name', names[key])
        if (obj is None or obj.visual_only or obj.fixed_base or
                (bound is not None and bound.get(key) is not obj)):
            raise ValueError('Original candidates must be native movable collidable bindings.')
        objects[key] = obj
    return p, objects


def _native_hands(robot):
    """Require native evidence for every arm, including attachment release windows."""
    arms = list(robot.arm_names)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Native arm inventory is unavailable.')
    rows = {}
    for arm in arms:
        held = robot._ag_obj_in_hand[arm]
        reading = int(robot.is_grasping(arm))
        if reading not in (-1, 1):
            raise ValueError('Native grasp state is unknown.')
        row = {'object_name': held.name if held is not None else None,
            'holding': reading == 1, 'constraint_active': robot._ag_obj_constraints[arm] is not None,
            'release_pending': robot._ag_release_counter[arm] is not None}
        row['released'] = (held is None and reading == -1 and not row['constraint_active']
            and not row['release_pending'])
        rows[arm] = row
    return rows


def _native_measure(session, objects, p, controller=None):
    """Only native readback; construction labels, observations and QA are unused."""
    from omnigibson.object_states import OnTop, Touching

    robot, scene = session.robot, session.env.scene
    hands = _native_hands(robot)
    scene_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    external = [obj for obj in scene_objects if obj not in objects.values() and obj is not robot
        and not obj.visual_only]
    native = {}
    for key, obj in objects.items():
        position, quaternion = obj.get_position_orientation()
        position, quaternion = _array(position, (3,)), _array(quaternion, (4,))
        low, high = (_array(value, (3,)) for value in obj.aabb)
        if not np.linalg.norm(quaternion) or np.any(high <= low):
            raise ValueError('Native object geometry is unavailable.')
        linear = float(np.linalg.norm(_array(obj.get_linear_velocity(), (3,))))
        angular = float(np.linalg.norm(_array(obj.get_angular_velocity(), (3,))))
        contacts = [{'body0': str(row.body0), 'body1': str(row.body1)} for row in obj.contact_list()]
        if any(not row['body0'] or not row['body1'] for row in contacts):
            raise ValueError('Native contact body identity is unavailable.')
        def contact_with(other):
            return any((_under(row['body0'], obj.prim_path) and _under(row['body1'], other.prim_path)) or
                (_under(row['body1'], obj.prim_path) and _under(row['body0'], other.prim_path))
                for row in contacts)
        relations = {}
        for other_key, other in objects.items():
            if other is not obj:
                relations[other_key] = {'on_top': bool(obj.states[OnTop].get_value(other)),
                    'touching': bool(obj.states[Touching].get_value(other)),
                    'native_contact': contact_with(other)}
        supports = []
        for other in external:
            # Native contacts identify possible original scene supports, not a guessed floor height.
            if contact_with(other) and bool(obj.states[OnTop].get_value(other)) and bool(
                    obj.states[Touching].get_value(other)):
                support_linear = float(np.linalg.norm(_array(other.get_linear_velocity(), (3,))))
                support_angular = float(np.linalg.norm(_array(other.get_angular_velocity(), (3,))))
                supports.append({'name': other.name, 'linear_speed_m_s': support_linear,
                    'angular_speed_rad_s': support_angular,
                    'stable': support_linear <= p['maximum_linear_speed_m_s'] and
                        support_angular <= p['maximum_angular_speed_rad_s']})
        native[key] = {'name': obj.name, 'position': position.tolist(), 'bottom_z_m': float(low[2]),
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular,
            'stable': linear <= p['maximum_linear_speed_m_s'] and angular <= p['maximum_angular_speed_rad_s'],
            'robot_contact': bool(obj.states[Touching].get_value(robot)) or contact_with(robot),
            'relations': relations, 'external_supports': supports,
            'native_contacts': contacts}
    released = all(row['released'] for row in hands.values())
    gripper = None
    acquired_key = None
    if controller is not None:
        arm = controller.working_arm
        gripper = controller.debug_gripper_state()
        hand = hands[arm]
        required = {'object_name', 'holding', 'constraint_active', 'release_pending', 'gripper_open'}
        if not required <= gripper.keys() or gripper['holding'] not in (True, False):
            raise ValueError('Native working gripper readback is unavailable.')
        if any(gripper[key] != hand[key] for key in ('object_name', 'holding', 'constraint_active', 'release_pending')):
            raise ValueError('Native grasp and controller attachment readings disagree.')
        released = released and gripper['gripper_open'] is True
        if (hand['holding'] and hand['constraint_active'] and not hand['release_pending'] and
                all(row['released'] for other_arm, row in hands.items() if other_arm != arm)):
            acquired_key = next((key for key, obj in objects.items() if obj.name == hand['object_name']), None)
    return {'objects': native, 'hands': hands, 'all_released': bool(released),
        'gripper': gripper, 'acquired_key': acquired_key}


def _supported(measured, upper, lower):
    relation = measured['objects'][upper]['relations'][lower]
    return all(relation.values())


def _external_support(measured, key):
    return any(row['stable'] for row in measured['objects'][key]['external_supports'])


def _only_stack_support(measured, upper, lower):
    """Reject upper members propped by scene furniture or another candidate."""
    return (_supported(measured, upper, lower) and
        not measured['objects'][upper]['external_supports'] and
        not any(_supported(measured, upper, other) for other in measured['objects']
            if other not in (upper, lower)))


def qualify_operation(session, executor, config, bound):
    """Inspect original native preconditions without moving the robot or objects."""
    checks = {}
    try:
        p, objects = _validate_operation(session, config, bound)
        measured = _native_measure(session, objects, p)
        checks = {'all_original_candidates_movable_collidable': True,
            'required_native_capabilities': all(getattr(executor.capabilities, key) for key in
                ('base_translation', 'base_rotation', 'ee_translation', 'gripper')),
            'all_arms_released': measured['all_released'],
            'initial_candidates_settled_supported': all(row['stable'] and not row['robot_contact'] and
                _external_support(measured, key) for key, row in measured['objects'].items()),
            'initial_stack_incomplete': not any(_supported(measured, p['middle_key'], base) and
                (not p['top_keys'] or any(_supported(measured, top, p['middle_key']) for top in p['top_keys']))
                for base in p['base_keys'])}
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks,
            'reason': None if accepted else 'native_ordered_stack_initial_precondition_failed',
            'native_state': measured, 'grasp_reach_qualified': False, 'stack_execution_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_ordered_stack_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}',
            'grasp_reach_qualified': False, 'stack_execution_qualified': False}


class OrderedStackEvaluator:
    """Measure ordered native acquisition, lift, release and stable support history."""
    def __init__(self, session, controller, config, output):
        self.parameters, self.objects = _validate_operation(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.dt = float(controller.dt)
        native_dt = 1.0 / float(session.env.env_config['action_frequency'])
        if not math.isfinite(self.dt) or self.dt <= 0 or not math.isclose(self.dt, native_dt, rel_tol=1e-9):
            raise ValueError('Native controller clock and environment frequency must agree.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _read(self):
        try:
            value = _native_measure(self.session, self.objects, self.parameters, self.controller)
            self.last = value
            return value
        except Exception as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            self.final_since = None
            return None

    def _elapsed(self, start):
        return 0.0 if start is None else (self.sample_count - start) * self.dt

    def _base_structural(self, measured, base):
        return _only_stack_support(measured, 'middle', base) and _external_support(measured, base)

    def _invalidate_credits(self, measured):
        """Every measured boundary can revoke credit; only sample_step adds time."""
        middle_name = measured['objects']['middle']['name']
        middle_reacquired = any(row['object_name'] == middle_name and
            (row['holding'] or row['constraint_active'] or row['release_pending'])
            for row in measured['hands'].values())
        if self.middle_placement is not None and (middle_reacquired or not self._base_structural(
                measured, self.middle_placement['base'])):
            self.events.append({'event': 'middle_placement_revoked', 'step': self.sample_count,
                'reason': 'measured_support_order_interrupted'})
            self.middle_placement = None
            self.base_since = {key: None for key in self.parameters['base_keys']}
        for base in self.parameters['base_keys']:
            if not (self._base_structural(measured, base) and
                    self._stable_clear(measured, ['middle', base]) and not any(
                        _supported(measured, top, 'middle') for top in self.parameters['top_keys'])):
                self.base_since[base] = None

    def _stable_clear(self, measured, keys):
        return measured['all_released'] and all(measured['objects'][key]['stable'] and not
            measured['objects'][key]['robot_contact'] for key in keys)

    def _lift_released(self, key):
        history = self.histories.get(key)
        return bool(history and history['release_step'] is not None and history['peak_lift_m'] >=
            self.parameters['minimum_lift_m'])

    def _manipulation_released(self, key):
        if self.parameters.get('manipulation_revision') != SUPPORT_TRANSFER_REVISION:
            return self._lift_released(key)
        history = self.histories.get(key)
        return bool(history and history['unsupported_while_acquired']
            and history['acquired_step'] is not None and history['separation_step'] is not None
            and history['release_step'] is not None
            and history['release_step'] > history['separation_step'] >= history['acquired_step'])

    def _final_pair(self, measured):
        if self.middle_placement is None or not self._manipulation_released('middle'):
            return None
        base = self.middle_placement['base']
        if not self._base_structural(measured, base):
            return None
        if not self.parameters['top_keys']:
            return (base, None) if self._stable_clear(measured, ['middle', base]) else None
        for top in self.parameters['top_keys']:
            history = self.histories.get(top)
            if (self._manipulation_released(top) and history['acquired_step'] > self.middle_placement['step'] and
                    _only_stack_support(measured, top, 'middle') and
                    self._stable_clear(measured, ['middle', base, top])):
                return base, top
        return None

    def begin_episode(self, observation):
        self.episode += 1
        self.active = True
        self.sample_count, self.measurement_error = 0, None
        self.histories, self.events = {}, []
        self.previous_acquired = None
        self.middle_placement, self.final_since, self.final_pair = None, None, None
        self.ordered_top_seen = False
        self.base_since = {key: None for key in self.parameters['base_keys']}
        self.last = None
        initial = self._read()
        self.initial = initial
        self.initial_eligible = bool(initial and initial['all_released'] and
            all(row['stable'] and not row['robot_contact'] and _external_support(initial, key)
                for key, row in initial['objects'].items()) and
            not any(_supported(initial, 'middle', base) and (not self.parameters['top_keys'] or
                any(_supported(initial, top, 'middle') for top in self.parameters['top_keys']))
                for base in self.parameters['base_keys']))
        return self.record(observation, action=None)

    def sample_step(self):
        if not self.active:
            return
        self.sample_count += 1
        measured = self._read()
        if measured is None or self.measurement_error or not self.initial_eligible:
            return
        self._invalidate_credits(measured)
        acquired = measured['acquired_key']
        movable = ['middle', *self.parameters['top_keys']]
        if acquired in movable and acquired != self.previous_acquired:
            current, initial = measured['objects'][acquired], self.initial['objects'][acquired]
            self.histories[acquired] = {'acquired_step': self.sample_count,
                'baseline_bottom_z_m': max(initial['bottom_z_m'], current['bottom_z_m']),
                'baseline_root_z_m': max(initial['position'][2], current['position'][2]),
                'peak_lift_m': 0.0, 'release_step': None}
            if self.parameters.get('manipulation_revision') == SUPPORT_TRANSFER_REVISION:
                self.histories[acquired].update(unsupported_while_acquired=False, separation_step=None)
            self.events.append({'event': 'native_acquired_grasp', 'object': acquired, 'step': self.sample_count})
            if acquired == 'middle':
                self.middle_placement = None
                self.base_since = {key: None for key in self.parameters['base_keys']}
        if acquired in self.histories:
            history, current = self.histories[acquired], measured['objects'][acquired]
            lift = min(current['bottom_z_m'] - history['baseline_bottom_z_m'],
                current['position'][2] - history['baseline_root_z_m'])
            history['peak_lift_m'] = max(history['peak_lift_m'], lift)
            if (self.parameters.get('manipulation_revision') == SUPPORT_TRANSFER_REVISION
                    and not history['unsupported_while_acquired']
                    and not current['external_supports']
                    and not any(_supported(measured, acquired, other)
                        for other in measured['objects'] if other != acquired)):
                history.update(unsupported_while_acquired=True, separation_step=self.sample_count)
                self.events.append({'event': 'native_acquired_support_separation',
                    'object': acquired, 'step': self.sample_count})
        if measured['all_released']:
            for key, history in self.histories.items():
                if history['release_step'] is None:
                    history['release_step'] = self.sample_count
                    self.events.append({'event': 'native_release', 'object': key, 'step': self.sample_count})
        self.previous_acquired = acquired
        if self.middle_placement is None:
            no_top_on_middle = not any(_supported(measured, top, 'middle') for top in self.parameters['top_keys'])
            for base in self.parameters['base_keys']:
                eligible = (self._manipulation_released('middle') and no_top_on_middle and
                    self._base_structural(measured, base) and self._stable_clear(measured, ['middle', base]))
                self.base_since[base] = (self.sample_count if self.base_since[base] is None else
                    self.base_since[base]) if eligible else None
                if eligible and self._elapsed(self.base_since[base]) + 1e-12 >= self.parameters['minimum_stability_duration_s']:
                    self.middle_placement = {'base': base, 'step': self.sample_count,
                        'stable_duration_s': self._elapsed(self.base_since[base])}
                    self.events.append({'event': 'middle_supported_placement', **self.middle_placement})
                    break
        pair = self._final_pair(measured)
        if pair and pair[1] is not None:
            self.ordered_top_seen = True
        if pair is None or pair != self.final_pair:
            self.final_since = self.sample_count if pair else None
        self.final_pair = pair

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = self._read()
        if measured is not None:
            self._invalidate_credits(measured)
        pair = self._final_pair(measured) if measured is not None else None
        if pair != self.final_pair or pair is None:
            # Observation boundaries can revoke stability but cannot advance its clock.
            self.final_since, self.final_pair = None, None
        support_transfer = self.parameters.get('manipulation_revision') == SUPPORT_TRANSFER_REVISION
        manipulation_check = 'middle_grasp_support_transfer_release' if support_transfer else 'middle_grasp_lift_release'
        checks = {'initial_incomplete_supported_scene': self.initial_eligible,
            'native_measurement_available': self.measurement_error is None,
            manipulation_check: self._manipulation_released('middle'),
            'ordered_middle_placement': self.middle_placement is not None,
            'selected_native_support_order': pair is not None,
            'released_and_stable_duration': self.final_since is not None and
                self._elapsed(self.final_since) + 1e-12 >= self.parameters['minimum_stability_duration_s']}
        if self.parameters['top_keys']:
            checks['top_grasp_after_middle_placement'] = bool(pair and pair[1] and
                self.histories[pair[1]]['acquired_step'] > self.middle_placement['step'])
        value = {'schema': 'private_ordered_stack_support_transfer_v1' if support_transfer else
                'private_ordered_stack_operation_v2', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'operation': OPERATION, 'task_success': bool(all(checks.values())),
            'checks': checks, 'selected_base': pair[0] if pair else None,
            'selected_top': pair[1] if pair else None, 'histories': self.histories,
            'middle_placement': self.middle_placement, 'events': self.events,
            'native_state': measured, 'measurement_error': self.measurement_error,
            'controller_steps': self.sample_count, 'controller_dt_s': self.dt,
            'stable_controller_duration_s': self._elapsed(self.final_since),
            'stability_clock': 'consecutive native controller callbacks after env.step; not physics substeps',
            'configuration': self.config}
        middle_seen = any(event.get('event') == 'middle_supported_placement' for event in self.events)
        items = [{'id': 'middle_stable_placement', 'kind': 'checkpoint', 'completed': middle_seen},
            {'id': 'released_final_stack', 'kind': 'terminal_condition', 'completed': bool(checks['released_and_stable_duration'] and pair)}]
        if self.parameters['top_keys']:
            items.insert(1, {'id': 'top_after_middle', 'kind': 'checkpoint',
                'completed': self.ordered_top_seen})
        available = self.measurement_error is None and self.initial_eligible
        for item in items:
            item['evaluation_validity'] = 'valid' if available else 'unavailable'
        value.update(completion_items=items, evaluation_validity='complete' if available else 'unavailable',
            task_revision=self.parameters.get('task_revision') if support_transfer else TASK_REVISION,
            measured_capability='supplied_order_execution')
        if support_transfer:
            value.update(manipulation_revision=SUPPORT_TRANSFER_REVISION,
                manipulation_check='native_acquired_support_separation_then_release')
        from wia.tasks.completion import progress
        value.update(progress(value))
        (self.output / 'state.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    """Create a read-only native evaluator through the standard task episode hooks."""
    return OrderedStackEvaluator(session, controller, config, output)

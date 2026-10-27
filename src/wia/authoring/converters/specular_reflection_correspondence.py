"""Convert original mirror correspondences into source-bound observation tasks.

Run on CPU: CategoryConverter().convert_all(sources, ConversionContext(batch_root))
through wia.authoring.conversion, or the existing category campaign runner.
The category_operation hooks only read native state; the shared guarded robot
controller performs every action. Revised answer scoring requires qualified source optics.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_contact

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.tasks.interaction_setup import array as native_array, verified_repaired_start


MODULE = 'core.esi_converters.specular_reflection_correspondence'
OPERATIONS = ('correspondence_approach', 'correspondence_approach_return')
ANSWER_OPERATION = 'answer_reflection_correspondence'
ACTION_OPERATION = 'answer_and_verify_reflection_correspondence'
CAMERA_HEIGHT_M = 0.8  # batch_mirror_correspondence.py CAMERA_HEIGHT


def _array(value):
    return native_array(value)


def _vector(value, length):
    return (isinstance(value, (list, tuple)) and len(value) == length
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value))


def reflected_ray_candidates(eye_xy, mirror_xy, normal_xy, positions):
    """Independent horizontal reflection formula used by the source generator.

    Metadata visibility flags and QA labels are deliberately absent. At runtime
    the mirror normal and all prop roots come from measured native transforms.
    This geometric correspondence is not a rendered visibility measurement.
    """
    normal = _array(normal_xy)
    incoming = _array(mirror_xy) - _array(eye_xy)
    if not np.isfinite([normal, incoming]).all() or min(np.linalg.norm(normal), np.linalg.norm(incoming)) < 1e-6:
        raise ValueError('undefined_original_reflection_ray')
    normal /= np.linalg.norm(normal)
    incoming /= np.linalg.norm(incoming)
    outgoing = incoming - 2 * float(incoming @ normal) * normal
    rows = []
    for key, point in positions.items():
        delta = _array(point)[:2] - mirror_xy
        along = float(delta @ outgoing)
        lateral = abs(float(outgoing[0] * delta[1] - outgoing[1] * delta[0]))
        rows.append({'binding': key, 'along_ray_m': along, 'lateral_error_m': lateral})
    return sorted(rows, key=lambda row: (row['lateral_error_m'], row['binding']))


def _source_data(source):
    raw, payload = source.raw, source.payload
    if ((source.big_task, source.small_task) != CategoryConverter.category
            or (raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category):
        raise ValueError('category_mismatch')
    q = payload.get('question_data', {})
    if payload.get('task_type') != 'mirror_correspondence' or q.get('task_type') != 'mirror_correspondence':
        raise ValueError('unsupported_source_mechanism')
    options = raw.get('options_json', [])
    options = json.loads(options) if isinstance(options, str) else options
    if (q.get('question') != raw.get('question') or q.get('answer') != raw.get('answer')
            or options != q.get('options') or options != ['left', 'middle', 'right']):
        raise ValueError('source_qa_annotation_conflict')
    camera = q.get('render', {}).get('camera_pose') or q.get('scene_setup', {}).get('camera', {})
    mirror = q.get('scene_setup', {}).get('mirror', {})
    setup = payload.get('mirror_setup', {})
    objects = q.get('placement', [])
    if (not raw.get('scene') or not raw.get('room') or not payload.get('floor_name')
            or not _vector(camera.get('position'), 3) or not _vector(camera.get('quaternion_xyzw'), 4)
            or not any(camera['quaternion_xyzw']) or not _vector(mirror.get('position'), 3)
            or not _vector(mirror.get('quaternion_xyzw'), 4) or not any(mirror['quaternion_xyzw'])
            or not _vector(mirror.get('normal_xy'), 2) or not all(setup.get(k) for k in ('name', 'category', 'model'))):
        raise ValueError('missing_original_mirror_floor_or_reference')
    if len(objects) != 3 or len({o.get('name') for o in objects}) != 3:
        raise ValueError('three_distinct_original_props_required')
    for obj in objects:
        if (not all(obj.get(k) for k in ('name', 'category', 'model'))
                or not _vector(obj.get('real_position'), 3) or not _vector(obj.get('quaternion_xyzw'), 4)
                or not any(obj['quaternion_xyzw'])):
            raise ValueError('missing_original_prop_asset_or_pose')
    if len({(o['category'], o['model']) for o in objects}) != 1:
        raise ValueError('original_identical_prop_mechanism_changed')
    if q.get('evidence', {}).get('temporarily_cleared_scene_objects'):
        raise ValueError('original_cleared_obstacles_need_separate_restoration_review')
    positions = {f'prop_{i}': o['real_position'] for i, o in enumerate(objects)}
    rays = reflected_ray_candidates(camera['position'][:2], mirror['position'][:2], mirror['normal_xy'], positions)
    aligned = [r for r in rays if r['along_ray_m'] > 0 and r['lateral_error_m'] <= 0.02]
    if len(aligned) != 1:
        raise ValueError('original_central_reflected_ray_is_ambiguous')
    selected = aligned[0]['binding']
    target = objects[int(selected.rsplit('_', 1)[1])]
    # Corroborate annotation identity without using it as the physical predicate.
    if (target['name'] != q.get('evidence', {}).get('real_object_name')
            or target.get('position_label') != raw['answer']):
        raise ValueError('source_label_disagrees_with_independent_reflection_geometry')
    from scipy.spatial.transform import Rotation
    native_axis = Rotation.from_quat(mirror['quaternion_xyzw']).as_matrix()[:2, 0]
    if abs(float(native_axis @ _array(mirror['normal_xy']))) < 0.999:
        raise ValueError('mirror_asset_normal_axis_unresolved')
    return raw, payload, q, camera, mirror, setup, objects, selected, rays


class CategoryConverter(ConverterBase):
    category = ('Specular Reflection', 'Correspondence')
    version = 'correspondence_v2_task_edit_20261018'

    def inspect(self, source):
        try:
            _source_data(source)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_correspondence_review_required']}
        return {'eligible': True, 'reason': 'Original optical layout supports a separate real navigation goal.'}

    def convert_all(self, sources, context):
        return super().convert_all(sorted(sources, key=lambda source: source.relative_path), context)

    def sample(self, manifest, policy=None):
        policy = policy or {}
        selected = super().sample(manifest, policy)
        by_id = {row['candidate_id']: row for row in manifest['records']}
        groups = {}
        for row in manifest['records']:
            if row['status'] != 'generated' or row.get('operation_recipe') == ANSWER_OPERATION:
                continue
            p = row['goal_predicate']['parameters']
            point = p['source_poses'][p['selected_prop']]['position']
            near = math.dist(p['start_xy'], point[:2]) < p['approach_leave_m']
            groups.setdefault((row['operation_recipe'], near), []).append(row['candidate_id'])
        # Cover both recipes and the genuine already-near departure branch.
        for group in sorted(groups):
            if not any(key in groups[group] for key in selected):
                protected_groups = {key: next((g for g, ids in groups.items() if key in ids), None) for key in selected}
                replaceable = [i for i, key in enumerate(selected) if by_id[key]['status'] == 'generated'
                    and list(protected_groups.values()).count(protected_groups[key]) > 1]
                if replaceable:
                    selected[replaceable[-1]] = sorted(groups[group])[0]
        return selected

    def convert(self, source, context):
        raw, payload, q, camera, mirror, setup, objects, selected, rays = _source_data(source)
        operation = context.config.get('correspondence_operation', ANSWER_OPERATION)
        if operation in (ANSWER_OPERATION, ACTION_OPERATION):
            from .specular_reflection_reflection_authoring import build_reflection_observation, build_reflection_action
            builder = build_reflection_action if operation == ACTION_OPERATION else build_reflection_observation
            result = builder(source, context, module=MODULE, operation=operation,
                camera=camera, mirror=mirror, props={f'prop_{i}': obj for i, obj in enumerate(objects)})
            parameters = result['task_definition']['private']['evaluation']['parameters']
            parameters['observable_candidate_labels'] = {f'prop_{i}': obj['position_label']
                for i, obj in enumerate(objects)}
            parameters['answer_reference'] = 'left_middle_right_in_original_source_image'
            if operation == ACTION_OPERATION:
                parameters['verification_contract']['required_bindings'] = [selected]
                parameters['verification_contract']['mechanism'] = 'same_object_correspondence_verified'
            result['available_operation_recipes'] += list(OPERATIONS)
            return result
        if operation is None:
            choice = hashlib.sha256(f'{context.seed}:{source.relative_path}'.encode()).digest()[0]
            operation = OPERATIONS[choice % len(OPERATIONS)]
        if operation not in OPERATIONS:
            raise ValueError('unsupported_correspondence_operation_recipe')
        bindings = {f'prop_{i}': {k: obj[k] for k in ('name', 'category', 'model')}
                    for i, obj in enumerate(objects)}
        bindings.update(mirror={k: setup[k] for k in ('name', 'category', 'model')},
                        source_floor={'name': payload['floor_name']})
        poses = {f'prop_{i}': {'position': obj['real_position'], 'quaternion_xyzw': obj['quaternion_xyzw']}
                 for i, obj in enumerate(objects)}
        poses['mirror'] = {k: mirror[k] for k in ('position', 'quaternion_xyzw')}
        yaw = math.degrees(math.atan2(mirror['position'][1] - camera['position'][1],
                                     mirror['position'][0] - camera['position'][0]))
        floor_z = float(camera['position'][2] - CAMERA_HEIGHT_M)
        patches = [{'operation': 'robot_pose', 'phase': 'before_construction',
                    'position': [*camera['position'][:2], floor_z + 0.1],
                    'yaw_deg': yaw, 'head_pitch_deg': 20.0}]
        parameters = {'selected_prop': selected, 'source_eye_xy': camera['position'][:2],
            'source_poses': poses, 'floor_z_m': floor_z, 'start_xy': camera['position'][:2],
            'start_yaw_deg': yaw, 'pose_tolerance_m': 0.02, 'orientation_tolerance_deg': 0.5,
            'ray_tolerance_m': 0.02, 'minimum_travel_m': 0.75, 'minimum_excursion_m': 0.5,
            'approach_minimum_m': 0.55, 'approach_maximum_m': 0.9,
            'approach_leave_m': 1.1, 'candidate_distance_margin_m': 0.1,
            'return_tolerance_m': 0.2, 'heading_tolerance_deg': 25.0,
            'maximum_terminal_speed_m_s': 0.02, 'maximum_terminal_angular_speed_rad_s': 0.1,
            'stability_steps': 5}
        category = objects[0]['category'].replace('_', ' ')
        objective = (f'Use the mirror to identify the real {category} corresponding to the central reflection '
            'from the initial viewing area. Drive toward that real object and stop 0.55 to 0.90 m away, '
            'facing it, closer to it than to either other candidate. If already within 1.10 m, '
            'first move farther away and then approach again.')
        success = ('Physically approach the correct original object, travel at least 0.75 m with '
            'an excursion of at least 0.50 m, and stop upright on the floor without collisions. '
            'Keep the mirror and all three objects in their original poses.')
        if operation == OPERATIONS[1]:
            objective += ' After stopping near that object, return within 0.20 m of your starting position and face the mirror.'
            success += ' Complete the approach before returning, then stop at the starting position facing the mirror.'
        public = {'objective': objective, 'success_criteria': success,
            'environment': f'The original mirror and three identical {category} objects in the original indoor scene.',
            'constraints': ['Use robot mounted observations and the existing guarded base controls.',
                'Fix the correspondence at the initial viewing area; later movement does not redefine the original left/right reference.',
                'Do not grasp, push, pass through or relocate the mirror or the three props.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(bindings), 'parameters': parameters}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
            patches=patches, capabilities=['base_translation', 'base_rotation'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
        task['defaults']['max_steps'] = int(context.config.get('correspondence_max_steps', 300))
        return {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
            'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'operation_recipe': operation, 'available_operation_recipes': list(OPERATIONS),
            'change_summary': ['Preserve original mirror material, native assets, dimensions, three prop poses and private QA.',
                'Initialize only the existing R1 at source reference XY with source-derived floor clearance and mirror-facing yaw.',
                'Add real object approach or approach-return navigation with an independent private native evaluator.'],
            'warning_codes': ['assumption_review_required', 'robot_start_unqualified', 'native_route_unverified',
                              'mounted_reflection_visibility_unverified', 'source_visibility_flags_are_labels'],
            'assumptions': [{
                'doubt': 'The free-camera point is not a certified R1 footprint or mounted-camera equivalent.',
                'chosen_assumption': 'Use exact source reference XY, source camera Z minus generator CAMERA_HEIGHT '
                    'plus 0.10 m drop clearance, mirror-facing yaw and 20 degree additional head pitch. '
                    'Reject native starts lacking measured floor contact, clearance or mirror image coverage.',
                'alternatives': ['Keep the empty original variant for diagnostic capture.',
                    'Propose a measured nearby robot-only stance in a later category revision.',
                    'Retain rejected starts in Review without changing scene geometry.'],
                'scope': source.relative_path + '; robot initialization only.',
                'risk': 'Different camera height can change visible reflections and left/right appearance. '
                    'The reference camera and QA remain private and fixed; the new robot view is not declared equivalent.'}, {
                'doubt': 'The generator uses a horizontal reflected center ray and assignment flags, '
                    'not unique rendered visibility; mounted reflection legibility is unmeasured.',
                'chosen_assumption': 'Select the intended original real object by independently recomputed '
                    'horizontal reflection geometry, corroborate source identity, and qualify it again using native poses. '
                    'Navigation is an auxiliary physical goal; neither visibility flags nor QA prove success.',
                'alternatives': ['Require human confirmation of initial mounted reflection correspondence before execution.',
                    'Retain optically ambiguous or occluded instances as failed Review items.'],
                'scope': 'All source instances and both recipes; fixed original observer reference.',
                'risk': 'Other objects may also appear in the mirror, and a geometric ray does not certify '
                    'reflection rendering, occlusion or an Agent-readable visual cue.'}],
            'private_source_evidence': {'reference_camera': camera, 'initial_pose_provenance':
                '/question_data/render/camera_pose and batch_mirror_correspondence.py CAMERA_HEIGHT=0.8',
                'source_optical_diagnostics': rays, 'selected_original_binding': selected,
                'source_evidence': q.get('evidence', {}), 'native_execution_verified': False},
            'limitations': ['No native simulation, rendered reflection, robot route or terminal success was tested by this CPU converter.',
                'Props retain original visual-only physics; measured chassis footprint and full-robot AABB exclusion supplement existing robot guards.',
                'Horizontal native ray agreement is independent geometry, not image recognition or rendered visibility evidence.',
                'Continuity and contacts are sampled on guarded controller steps, not every physics substep.',
                'Runtime, throughput, route completion rate and the 300-decision budget remain unmeasured.',
                'Source answers, poses, maps, evaluator state and Human review images are private.'],
            'attempted_repairs': ['Reversible declarative robot-only initialization; no source mechanism, asset or physics changes.'],
            'next_step': 'Capture original/adapted views and qualify native initialization, then execute the guarded '
                'navigation recipe. Retain all failed starts, captures and execution attempts in Review.'}


def _pose(obj):
    from scipy.spatial.transform import Rotation
    position, quaternion = (_array(v) for v in obj.get_position_orientation())
    if (position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-6):
        raise ValueError('native_pose_unavailable')
    return position, Rotation.from_quat(quaternion)


def _bounds(obj):
    low, high = (_array(v) for v in obj.aabb)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('native_bounds_unavailable')
    return low, high


def _objects(session, config):
    if (module_identity(config.get('module')) != MODULE or config.get('operation') not in OPERATIONS
            or (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('native_correspondence_scene_contract_mismatch')
    objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
    if set(objects) != {'mirror', 'source_floor', 'prop_0', 'prop_1', 'prop_2'} or any(o is None for o in objects.values()):
        raise ValueError('native_original_bindings_unavailable')
    return objects


def measure_native_state(session, config):
    """Read optical anchors, robot chassis, velocities and actual contact reports."""
    objects, p = _objects(session, config), config['parameters']
    robot = session.robot
    position, rotation = _pose(robot)
    chassis = robot.links[robot.base_footprint_link_name]
    low, high = _bounds(chassis)
    body_low, body_high = _bounds(robot)
    linear, angular = (_array(v) for v in (chassis.get_linear_velocity(), chassis.get_angular_velocity()))
    if linear.shape != (3,) or angular.shape != (3,) or not np.isfinite([linear, angular]).all():
        raise ValueError('native_chassis_velocity_unavailable')
    poses, preserved, excluded, body_excluded = {}, True, True, True
    for key in ('mirror', 'prop_0', 'prop_1', 'prop_2'):
        point, orient = _pose(objects[key])
        expected = p['source_poses'][key]
        from scipy.spatial.transform import Rotation
        drift = float(np.linalg.norm(point - expected['position']))
        angle = math.degrees(float((orient.inv() * Rotation.from_quat(expected['quaternion_xyzw'])).magnitude()))
        preserved &= drift <= p['pose_tolerance_m'] and angle <= p['orientation_tolerance_deg']
        obj_low, obj_high = _bounds(objects[key])
        # Visual-only props cannot provide contacts. Exclude chassis XY overlap
        # explicitly, including props underneath raised links of the chassis.
        excluded &= not bool(np.all(high[:2] > obj_low[:2]) and np.all(obj_high[:2] > low[:2]))
        body_excluded &= not bool(np.all(body_high > obj_low) and np.all(obj_high > body_low))
        poses[key] = {'position': point.tolist(), 'quaternion_xyzw': orient.as_quat().tolist(),
                      'source_drift_m': drift, 'source_orientation_drift_deg': angle}
    normal = _pose(objects['mirror'])[1].as_matrix()[:2, 0]
    rays = reflected_ray_candidates(p['source_eye_xy'], poses['mirror']['position'][:2], normal,
                                    {key: poses[key]['position'] for key in ('prop_0', 'prop_1', 'prop_2')})
    aligned = [r['binding'] for r in rays if r['along_ray_m'] > 0 and r['lateral_error_m'] <= p['ray_tolerance_m']]
    optical_valid = aligned == [p['selected_prop']]
    distances = {key: float(np.linalg.norm(position[:2] - poses[key]['position'][:2]))
                 for key in ('prop_0', 'prop_1', 'prop_2')}
    source_floor = objects['source_floor']
    if source_floor.visual_only or getattr(source_floor, 'category', None) not in ('floor', 'floors'):
        raise ValueError('native_source_floor_not_collidable')
    floor_paths = [o.prim_path for o in session.env.scene.objects
                   if getattr(o, 'category', None) in ('floor', 'floors') and not o.visual_only]
    within = lambda body, path: body == path or body.startswith(path + '/')
    floor_contacts, other_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        external = [str(b) for b in (contact.body0, contact.body1) if not within(str(b), robot.prim_path)]
        if not external:
            continue
        point, normal = _array(contact.position), _array(contact.normal)
        row = {'external_bodies': external, 'position': point.tolist(), 'normal': normal.tolist()}
        is_floor = all(any(within(b, path) for path in floor_paths) for b in external)
        (floor_contacts if is_floor else other_contacts).append(row)
    supported = bool(floor_contacts) and all(_vector(r['position'], 3) and _vector(r['normal'], 3)
        and abs(r['position'][2] - p['floor_z_m']) <= 0.05 and abs(r['normal'][2]) >= 0.5 for r in floor_contacts)
    upright = float(rotation.as_matrix()[2, 2]) >= 0.95
    heading = rotation.as_matrix()[:2, 0]
    def heading_error(key):
        delta = _array(poses[key]['position'])[:2] - position[:2]
        if min(np.linalg.norm(delta), np.linalg.norm(heading)) < 1e-6:
            return 180.0
        return math.degrees(math.acos(float(np.clip(heading @ delta / (np.linalg.norm(heading) * np.linalg.norm(delta)), -1, 1))))
    return {'position': position.tolist(), 'yaw_deg': math.degrees(math.atan2(heading[1], heading[0])),
        'linear_speed_m_s': float(np.linalg.norm(linear)), 'angular_speed_rad_s': float(np.linalg.norm(angular)),
        'native_poses': poses, 'native_ray_diagnostics': rays, 'optical_correspondence_preserved': optical_valid,
        'original_poses_preserved': bool(preserved), 'chassis_clear_of_optical_assets': bool(excluded),
        'robot_envelope_clear_of_optical_assets': bool(body_excluded),
        'candidate_distances_m': distances, 'target_heading_error_deg': heading_error(p['selected_prop']),
        'mirror_heading_error_deg': heading_error('mirror'), 'floor_supported': supported,
        'floor_contacts': floor_contacts, 'nonfloor_contacts': other_contacts, 'upright': upright,
        'state_valid': bool(supported and not other_contacts and upright and excluded and body_excluded and preserved and optical_valid)}


def _mirror_view(session, objects):
    """Read the mounted RGB and native calibration without altering sensor modes."""
    from scipy.spatial.transform import Rotation
    sensor = session.robot.sensors[session.camera_name]
    observation, _ = sensor.get_obs()
    rgb = _array(observation['rgb'])
    intrinsic = _array(sensor.intrinsic_matrix)
    camera_pos, camera_rot = _pose(sensor)
    mirror_pos, _ = _pose(objects['mirror'])
    local = camera_rot.inv().apply(mirror_pos - camera_pos)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError('native_mounted_rgb_calibration_unavailable')
    depth = -float(local[2])
    pixel = intrinsic @ np.array([local[0], -local[1], depth])
    pixel = pixel[:2] / depth if depth > 0 else np.array([-1., -1.])
    covered = depth > 0 and intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0 and 8 <= pixel[0] < rgb.shape[1]-8 and 8 <= pixel[1] < rgb.shape[0]-8
    return {'mirror_center_in_mounted_frustum': bool(covered), 'mirror_center_pixel_xy': pixel.tolist(),
            'native_rgb_shape': list(rgb.shape), 'rendered_reflection_visibility_verified': False}


def qualify_operation(session, executor, config, bound):
    if config.get('operation') == ACTION_OPERATION:
        from .specular_reflection_reflection_authoring import qualify_reflection_action
        return qualify_reflection_action(session, executor, config, bound)
    if config.get('operation') == ANSWER_OPERATION:
        from .specular_reflection_reflection_authoring import _qualify_observation
        return _qualify_observation(session, executor, config, bound)
    """Read native initial conditions; no robot/simulator actions occur here."""
    try:
        measured = measure_native_state(session, config)
        p = config['parameters']
        view = _mirror_view(session, _objects(session, config))
        yaw_error = abs((measured['yaw_deg'] - p['start_yaw_deg'] + 180) % 360 - 180)
        position, quaternion = session.robot.get_position_orientation()
        repair = verified_repaired_start(session, session.room_name, _array(position), _array(quaternion),
            position_tolerance_m=.05, heading_tolerance_deg=2.)
        checks = {'native_r1': type(session.robot).__name__ == 'R1' and not session.robot.visual_only,
            'native_holonomic_base': type(session.robot.controllers.get('base')).__name__ == 'HolonomicBaseJointController',
            'guarded_base_capabilities': bool(executor.capabilities.base_translation and executor.capabilities.base_rotation),
            'original_or_verified_initialized_start_xy': bool(float(np.linalg.norm(_array(measured['position'])[:2]
                - p['start_xy'])) <= 0.05 or repair['accepted']),
            'original_or_verified_initialized_heading': bool(yaw_error <= 2.0 or repair['accepted']),
            'native_floor_contacts_clearance_and_optical_anchors': measured['state_valid'],
            'mounted_mirror_center_coverage': view['mirror_center_in_mounted_frustum']}
        return {'accepted': bool(all(checks.values())), 'checks': checks, 'native_state': measured,
                'native_view': view, 'verified_repaired_start': repair, 'route_qualified': False,
                'reason': None if all(checks.values()) else 'native_correspondence_start_precondition_failed'}
    except (ValueError, RuntimeError, KeyError, AttributeError, TypeError, IndexError) as error:
        return {'accepted': False, 'route_qualified': False,
                'reason': f'native_correspondence_measurement_unavailable:{type(error).__name__}:{error}'}



def _completion_result(items):
    """Report declared physical obligations without distance-shaped rewards."""
    completed = sum(bool(item['completed']) for item in items)
    total = len(items)
    score = completed / total
    return {'scoring_version': 'task_completion_v2', 'evaluation_validity': 'complete',
            'completion_items': items, 'completed_goals': completed, 'total_goals': total,
            'completion_score': score, 'score_percent': 100. * score}


class CorrespondenceEvaluator:
    """Private measured approach/return history with fail-closed motion continuity."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.p = config['parameters']
        _objects(session, config)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _time(self):
        value = float(self.session.pipeline.og.sim.current_time)
        if not math.isfinite(value):
            raise ValueError('native_clock_unavailable')
        return value

    def begin_episode(self, observation):
        qualified = qualify_operation(self.session, self.controller, self.config, _objects(self.session, self.config))
        if not qualified['accepted']:
            raise ValueError(qualified['reason'])
        measured = qualified['native_state']
        self.episode += 1
        self.active = True
        self.start = self.previous = _array(measured['position'])
        self.previous_yaw = measured['yaw_deg']
        self.previous_time = self._time()
        self.travel, self.excursion, self.samples = 0.0, 0.0, 0
        self.faults, self.approach_samples, self.terminal_samples = set(), 0, 0
        self.approached = False
        self.approach_checkpoint = None
        self.left_approach = measured['candidate_distances_m'][self.p['selected_prop']] >= self.p['approach_leave_m']
        self.initial = qualified
        return self.record(observation, action=None)

    def _approach(self, measured):
        p, distances = self.p, measured['candidate_distances_m']
        selected = distances[p['selected_prop']]
        return bool(self.left_approach and measured['state_valid']
            and p['approach_minimum_m'] <= selected <= p['approach_maximum_m']
            and all(selected + p['candidate_distance_margin_m'] <= distance
                    for key, distance in distances.items() if key != p['selected_prop'])
            and measured['target_heading_error_deg'] <= p['heading_tolerance_deg'])

    def _terminal(self, measured):
        if self.config['operation'] == OPERATIONS[0]:
            return self._approach(measured)
        return bool(self.approached and measured['state_valid']
            and np.linalg.norm(_array(measured['position'])[:2] - self.start[:2]) <= self.p['return_tolerance_m']
            and measured['mirror_heading_error_deg'] <= self.p['heading_tolerance_deg'])

    def _settled(self, measured):
        return (measured['linear_speed_m_s'] <= self.p['maximum_terminal_speed_m_s']
                and measured['angular_speed_rad_s'] <= self.p['maximum_terminal_angular_speed_rad_s'])

    def sample_step(self):
        if not self.active:
            return
        measured, now = measure_native_state(self.session, self.config), self._time()
        point = _array(measured['position'])
        elapsed, distance = now - self.previous_time, float(np.linalg.norm(point[:2] - self.previous[:2]))
        cfg = self.controller.config
        yaw_delta = abs((measured['yaw_deg'] - self.previous_yaw + 180) % 360 - 180)
        if (elapsed <= 0 or distance > cfg.max_translation_speed_m_s * max(elapsed, 0) + cfg.translation_tolerance_m
                or yaw_delta > cfg.max_rotation_speed_deg_s * max(elapsed, 0) + cfg.rotation_tolerance_deg):
            self.faults.add('motion_clock_or_continuity_failed')
        if abs(point[2] - self.previous[2]) > 0.05 or not measured['state_valid']:
            self.faults.add('native_support_clearance_or_anchor_failed')
        self.samples += 1
        self.travel += distance
        self.excursion = max(self.excursion, float(np.linalg.norm(point[:2] - self.start[:2])))
        self.left_approach |= measured['candidate_distances_m'][self.p['selected_prop']] >= self.p['approach_leave_m']
        moved = self.travel >= self.p['minimum_travel_m'] and self.excursion >= self.p['minimum_excursion_m']
        approach = self._approach(measured) and self._settled(measured) and moved and not self.faults
        self.approach_samples = self.approach_samples + 1 if approach else 0
        newly_approached = not self.approached and self.approach_samples >= self.p['stability_steps']
        self.approached |= self.approach_samples >= self.p['stability_steps']
        if self.approached and self.approach_checkpoint is None:
            self.approach_checkpoint = {'time_s': now, 'position': measured['position'],
                'stable_samples': self.approach_samples}
        terminal = self._terminal(measured) and self._settled(measured) and self.approached and not self.faults
        if self.config['operation'] != OPERATIONS[0] and newly_approached:
            terminal = False  # The return hold must follow the completed approach event.
        self.terminal_samples = self.terminal_samples + 1 if terminal else 0
        self.previous, self.previous_time = point, now
        self.previous_yaw = measured['yaw_deg']
        with (self.output / 'physical_trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps({'sample': self.samples, 'simulation_time_s': now, 'native_state': measured,
                'travel_m': self.travel, 'excursion_m': self.excursion, 'approached': bool(self.approached),
                'motion_faults': sorted(self.faults)}, allow_nan=False) + '\n')

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = measure_native_state(self.session, self.config)
        yaw_delta = abs((measured['yaw_deg'] - self.previous_yaw + 180) % 360 - 180)
        if np.linalg.norm(_array(measured['position']) - self.previous) > 1e-6 or yaw_delta > 1e-4:
            self.faults.add('unsampled_native_motion')
        if not measured['state_valid']:
            self.faults.add('native_support_clearance_or_anchor_failed')
        criteria = {'qualified_start': self.initial['accepted'], 'continuous_safe_motion': not self.faults,
            'positive_measured_travel': self.samples > 0 and self.travel >= self.p['minimum_travel_m'],
            'minimum_excursion': self.excursion >= self.p['minimum_excursion_m'],
            'stable_correct_object_approach': bool(self.approached), 'native_terminal_state_valid': measured['state_valid'],
            'requested_terminal_goal': self._terminal(measured), 'terminal_at_rest': self._settled(measured),
            'terminal_stability': self.terminal_samples >= self.p['stability_steps']}
        value = {'schema': 'private_native_mirror_correspondence_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step, 'action': action,
            'task_success': bool(all(criteria.values())), 'criteria': criteria, 'native_state': measured,
            'physical_travel_m': self.travel, 'maximum_excursion_m': self.excursion,
            'control_step_samples': self.samples, 'terminal_stability_samples': self.terminal_samples,
            'motion_faults': sorted(self.faults), 'configuration': self.config,
            'reflected_visibility_certified': False}
        returning = self.config['operation'] != OPERATIONS[0]
        items = [{'id': 'selected_object_approach',
            'completed': bool(not self.faults and (self.approached if returning else all(criteria.values()))),
            'evaluation_validity': 'valid',
            'kind': 'historical_checkpoint' if returning else 'terminal_condition',
            'evidence': {'visit': self.approach_checkpoint, 'safety_faults': sorted(self.faults)}}]
        if returning:
            items.append({'id': 'return_and_stop', 'completed': bool(all(criteria.values())),
                'evaluation_validity': 'valid', 'kind': 'terminal_condition', 'evidence': criteria})
        value.update(_completion_result(items))
        (self.output / 'state.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == ACTION_OPERATION:
        from .specular_reflection_reflection_authoring import ReflectionVerificationEvaluator
        return ReflectionVerificationEvaluator(session, controller, config, output)
    if config.get('operation') == ANSWER_OPERATION:
        from .specular_reflection_reflection_authoring import ReflectionObservationEvaluator
        return ReflectionObservationEvaluator(session, controller, config, output)
    return CorrespondenceEvaluator(session, controller, config, output)

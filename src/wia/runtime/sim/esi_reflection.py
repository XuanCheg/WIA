"""Configure and measure source-aligned reflection embodiment without moving optics.

Run via src/wia/runtime/commands/check_esi_core_native.py with a revised reflection task.
Robot pose is proposed only during initialization; head mounts stay attached and
unmodified. Native motion and source-view comparisons remain independent gates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from wia.tasks.interaction_setup import array

ACTION_PROFILE_VERSION = 'reflection_sensor_profile_v3'
INITIALIZATION_REVISION = 'reflection_initialization_20261018_v3'
INITIALIZATION_OPERATION = 'align_original_source_reflection_robot'


def reflection_profile_sha256(profile):
    """Exact JSON profile digest shared by initialization and final controllers."""
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _revised_profile(profile):
    return profile.get('version') == ACTION_PROFILE_VERSION


def _alignment_identity(session, profile):
    return {'reset_id': session.native_reset_id, 'profile_sha256': reflection_profile_sha256(profile)}


def _alignment_completed(session, profile):
    receipt = getattr(session, 'reflection_alignment_initialization', {})
    return receipt.get('completed') is True and all(receipt.get(key) == value
        for key, value in _alignment_identity(session, profile).items())


def _public_observation_started(session):
    # Never access ESISession.observation_source: that property creates a source.
    state = vars(session)
    source = state.get('_observation_source') or state.get('observation_source')
    return bool(source is not None and (getattr(source, 'latest_snapshot', None) is not None or
        getattr(source, 'snapshot_id', 0) or getattr(source, 'action_step', 0) or getattr(source, 'history', ())))


def profile_from_definition(definition):
    runtime = getattr(definition, 'runtime', None)
    if runtime is not None and getattr(runtime, 'reflection_sensor_profile', None):
        return deepcopy(runtime.reflection_sensor_profile)
    private = getattr(definition, 'private', None)
    parameters = getattr(getattr(private, 'evaluation', None), 'parameters', {})
    return deepcopy(parameters.get('reflection_sensor_profile'))


def configure_embodiment(session, config):
    adapter = getattr(session, 'scene_adapter', None)
    loaded = getattr(session, 'loaded_task', None) or getattr(adapter, 'task', None)
    profile = profile_from_definition(loaded.definition) if loaded is not None else None
    if not profile:
        return
    from scipy.spatial.transform import Rotation
    scale = profile.get('robot_uniform_scale') or 1.
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Reflection scaling requires a positive uniform scalar.')
    robots = config.get('robots', [])
    if len(robots) != 1 or robots[0]['type'] != 'R1':
        raise ValueError('Reflection requires the original mounted R1 robot.')
    forward = Rotation.from_quat(profile['quaternion_xyzw']).apply([0., 0., -1.])
    yaw = math.atan2(forward[1], forward[0])
    floor = profile['source_floor_reference']['proposed_z_m']
    position = [*profile['position_m'][:2], floor]
    robots[0].update(scale=float(scale), position=position,
        orientation=[0., 0., math.sin(yaw / 2), math.cos(yaw / 2)], pose_frame='world')
    # Explicit reflection exception changes intrinsics, never camera mounting.
    session.head_pitch_deg = 0.
    session.reflection_sensor_profile = profile
    session.reflection_qualification = {'version': 'reflection_native_qualification_v1',
        'robot_uniform_scale': float(scale), 'units': 'world_metres',
        'source_head_pose_verified': False, 'source_head_intrinsics_verified': False,
        'native_robot_scale_verified': False, 'mounted_eight_camera_rig_verified': False,
        'legal_mobility_verified': False, 'optical_scene_preserved': False,
        'scope': 'Initialization proposal; physical motion and source-view evidence pending.'}


def align_initial_robot(session):
    """Align whole robot yaw/XY once, retaining its floor-level root and joints."""
    profile = getattr(session, 'reflection_sensor_profile', None)
    if profile is None:
        return
    if _revised_profile(profile):
        # Shared interaction setup can route here after the explicit hook.
        # Return the exact same attempt; never realign or overwrite its receipt.
        if _alignment_completed(session, profile):
            return session.reflection_qualification['initialization']
        if not getattr(session, '_reflection_initialization_hook_active', False):
            raise RuntimeError('Revised reflection alignment requires its explicit initialization hook.')
    from scipy.spatial.transform import Rotation
    from omnigibson.utils.usd_utils import PoseAPI
    robot = session.env.robots[0]
    head = next(sensor for name, sensor in robot.sensors.items() if 'eyes' in name or 'head' in name)
    target = array(profile['position_m'])
    wanted = Rotation.from_quat(profile['quaternion_xyzw']).apply([0., 0., -1.])
    initial = {'robot_pose': [array(v).tolist() for v in robot.get_position_orientation()],
        'head_mount': [array(v).tolist() for v in head.get_position_orientation(frame='parent')]}
    # R1's fixed native eye mount looks down. Correct its pitch using the real
    # third torso joint, whose installed R1 axis is negative world Y at the
    # neutral posture. This uses its existing position controller, not a camera
    # transform or a joint-position setter. Final measured pose remains a gate.
    from wia.runtime.sim.esi_base import R1BaseController, BaseControlConfig
    session.robot = robot
    hold = R1BaseController(session, BaseControlConfig.load())
    dofs = array(robot.controllers['trunk'].dof_idx).astype(int)
    lower, upper = [array(value)[dofs] for value in robot.control_limits['position']]
    joint_trace = []
    for _ in range(2):
        current_forward = Rotation.from_quat(array(head.get_position_orientation()[1])).apply([0., 0., -1.])
        elevation_error = math.asin(float(np.clip(wanted[2], -1, 1))) - math.asin(float(np.clip(current_forward[2], -1, 1)))
        if abs(elevation_error) < math.radians(.1):
            break
        q = array(robot.get_joint_positions())[dofs]
        target_q = q.copy()
        target_q[2] += elevation_error
        if np.any(target_q <= lower + .02) or np.any(target_q >= upper - .02):
            raise ValueError('Source head pitch requires a torso posture outside native joint limits.')
        for index in range(150):
            q = array(robot.get_joint_positions())[dofs]
            error = target_q - q
            hold.hold_action[hold.indices['trunk']] = q + np.clip(error, -.4 * hold.dt, .4 * hold.dt)
            hold.step_base_debug()
            joint_trace.append({'joints_rad': q.tolist(), 'target_rad': target_q.tolist()})
            if np.max(np.abs(error)) < .001:
                break
        hold.hold_action[hold.indices['trunk']] = array(robot.get_joint_positions())[dofs]
        hold.stop()
    initial['native_torso_alignment'] = {'controller': 'JointController', 'control_trace': joint_trace,
        'joint_limits_rad': [lower.tolist(), upper.tolist()], 'camera_mount_changed': False}
    for _ in range(2):
        hp, hq = (array(v) for v in head.get_position_orientation())
        rp, rq = (array(v) for v in robot.get_position_orientation())
        actual = Rotation.from_quat(hq).apply([0., 0., -1.])
        delta = math.atan2(wanted[1], wanted[0]) - math.atan2(actual[1], actual[0])
        rotation = Rotation.from_euler('z', delta)
        rp[:2] += target[:2] - hp[:2]
        robot.set_position_orientation(position=rp.tolist(), orientation=(rotation * Rotation.from_quat(rq)).as_quat().tolist())
        PoseAPI.invalidate()
    # Do not correct Z by floating the base or rotate the camera to hide pitch error.
    head.focal_length = float(head.horizontal_aperture) / (2. * math.tan(math.radians(profile['horizontal_fov_deg']) / 2.))
    session.reflection_qualification['initialization'] = initial
    session.reflection_qualification['optical_actor_mutations'] = 0
    return initial


def measure_reflection(session, *, controller=None, finalize_initial=False):
    """Measure native gates; v3 initial pose latches only at final controller setup.

    Root calls ``finalize_initial=True`` with the final native controller after
    trunk setup and configuration of all eight mounted cameras. Earlier reads
    remain named diagnostics and cannot become the initial source-pose proof.
    """
    profile = getattr(session, 'reflection_sensor_profile', None)
    if profile is None:
        return None
    from scipy.spatial.transform import Rotation
    from wia.tasks.interaction_setup import _robot_contact_check, _objects
    head = session.robot.sensors[session.camera_name]
    hp, hq = (array(v) for v in head.get_position_orientation())
    target = array(profile['position_m'])
    angle = float(np.degrees((Rotation.from_quat(profile['quaternion_xyzw']).inv() * Rotation.from_quat(hq)).magnitude()))
    error = float(np.linalg.norm(hp - target))
    fov = math.degrees(2. * math.atan(float(head.horizontal_aperture) / (2. * float(head.focal_length))))
    floors = [o for o in _objects(session.env.scene) if o.category in ('floor', 'floors')]
    floor_z = profile['source_floor_reference']['proposed_z_m']
    if _revised_profile(profile):
        from wia.tasks.native_floor_contacts import measure_native_floor_contacts
        support = measure_native_floor_contacts(session, session.robot, [o.prim_path for o in floors], nominal_floor_z=floor_z)
    else:
        support = _robot_contact_check(session.robot, [o.prim_path for o in floors], floor_z)
    scale = array(session.robot.scale)
    registry = getattr(session, 'robot_camera_registry', {})
    mounts = {name: {'prim_path': session.robot.sensors[sensor].prim_path,
        'position_m': array(session.robot.sensors[sensor].get_position_orientation()[0]).tolist()}
        for name, sensor in registry.items()}
    requested = profile.get('robot_uniform_scale') or 1.
    receipt = session.reflection_qualification
    measurement = {'position_m': hp.tolist(), 'quaternion_xyzw': hq.tolist(),
        'position_error_m': error, 'orientation_error_deg': angle, 'verified': error <= .02 and angle <= 1.}
    rig_complete = len(mounts) == 8 and len({row['prim_path'] for row in mounts.values()}) == 8 and all(
        row['prim_path'].startswith(session.robot.prim_path + '/') for row in mounts.values())
    if _revised_profile(profile):
        if finalize_initial:
            actual = getattr(controller, 'controller', controller)
            if (not _alignment_completed(session, profile) or actual is None or
                    getattr(actual, 'session', None) is not session or getattr(actual, 'robot', None) is not session.robot or
                    not rig_complete or len(getattr(session, 'robot_camera_mounts', {})) != 8 or
                    _public_observation_started(session) or getattr(actual, '_public_action_sequence', 0) or
                    getattr(actual, '_public_action_context', {}).get('active')):
                raise RuntimeError('Initial reflection pose requires completed same-reset alignment, final native controller, complete mounted rig and no public observation/action.')
            identity = {**_alignment_identity(session, profile), 'controller_instance': str(id(actual)),
                'controller_class': type(actual).__module__ + '.' + type(actual).__qualname__}
            previous = receipt.get('initial_head_pose_measurement')
            if previous is not None and previous.get('finalization_identity') != identity:
                raise RuntimeError('Initial reflection pose cannot be rebound to a different final controller/reset/profile.')
            if previous is None:
                receipt['initial_head_pose_measurement'] = {**measurement, 'finalization_identity': identity,
                    'measurement_stage': 'after_alignment_final_controller_and_complete_mounted_rig'}
        else:
            label = ('post_alignment_pre_final_controller_measurement' if _alignment_completed(session, profile)
                else 'pre_alignment_head_pose_diagnostic')
            receipt[label] = measurement
    elif 'initial_head_pose_measurement' not in receipt:
        receipt['initial_head_pose_measurement'] = measurement
    intrinsics = {'horizontal_fov_matches': abs(fov - profile['horizontal_fov_deg']) <= .1}
    projection = None
    if profile.get('head_image_size_px'):
        expected_size = profile['head_image_size_px']
        intrinsics['source_image_aspect_matches'] = bool(
            head.image_width * expected_size[1] == head.image_height * expected_size[0])
        try:
            parameters = head.camera_parameters
            matrix = array(parameters['cameraProjection']).reshape(4, 4)
            native_size = array(parameters['renderProductResolution']).astype(int)
            effective_fov = math.degrees(2 * math.atan(1 / abs(float(matrix[0, 0]))))
            projection = {'matrix': matrix.tolist(), 'render_size_px': native_size.tolist(),
                'effective_horizontal_fov_deg': effective_fov,
                'native_intrinsic_matrix': array(head.intrinsic_matrix).tolist()}
            intrinsics['native_projection_fov_matches'] = abs(effective_fov - profile['horizontal_fov_deg']) <= .1
            intrinsics['native_projection_size_matches'] = native_size.tolist() == list(expected_size)
        except Exception as projection_error:
            intrinsics['native_projection_available'] = False
            projection = {'error': f'{type(projection_error).__name__}: {projection_error}'}
    initial_pose = receipt.get('initial_head_pose_measurement', {})
    initial_pose_verified = initial_pose.get('verified', False)
    scale_verified = bool(np.allclose(scale, requested, atol=1e-5)) and support.get('floor_supported') is True and support.get('collision_free') is True
    if _revised_profile(profile):
        initial_pose_verified = bool(initial_pose_verified and _alignment_completed(session, profile) and all(
            initial_pose.get('finalization_identity', {}).get(key) == value for key, value in _alignment_identity(session, profile).items()))
        scale_verified = bool(profile.get('robot_uniform_scale') is not None and scale_verified and support.get('measurement_available') is True)
    receipt.update(source_head_pose_verified=initial_pose_verified,
        source_head_intrinsics_verified=all(intrinsics.values()),
        head_intrinsic_checks=intrinsics, native_head_projection=projection,
        mounted_eight_camera_rig_verified=(rig_complete if _revised_profile(profile) else len(mounts) == 8 and all(
            row['prim_path'].startswith(session.robot.prim_path + '/') for row in mounts.values())),
        native_robot_scale_verified=scale_verified,
        head_position_m=hp.tolist(), head_quaternion_xyzw=hq.tolist(),
        head_position_error_m=error, head_orientation_error_deg=angle,
        horizontal_fov_deg=fov, clipping_range_m=array(head.clipping_range).tolist(),
        measured_uniform_scale=scale.tolist(), floor_checks=support, mounted_cameras=mounts,
        position_tolerance_m=.02, orientation_tolerance_deg=1., fov_tolerance_deg=.1)
    actual_height = hp[2] - floor_z
    receipt['suggested_uniform_scale_from_measured_height'] = (
        float(requested * (target[2] - floor_z) / actual_height) if actual_height > 0 else None)
    receipt['optical_integrity'] = measure_optical_integrity(session)
    receipt['optical_scene_preserved'] = receipt['optical_integrity']['accepted']
    return receipt


def measure_optical_integrity(session):
    """Compare native actors to the unchanged ESI optical constructor contract."""
    from scipy.spatial.transform import Rotation
    adapter = session.scene_adapter
    raw = json.loads(adapter.task.source_path.read_text())
    payload = raw['metadata_json']
    payload = json.loads(payload) if isinstance(payload, str) else payload
    q = payload['question_data']
    setup = payload.get('mirror_setup', {})
    mirror = q['scene_setup']['mirror']
    expected = [{'name': setup.get('name') or 'render_mirror_main',
        'category': setup.get('category') or 'mirror', 'model': setup.get('model') or 'tytkbq',
        'real_position': mirror['position'], 'quaternion_xyzw': mirror['quaternion_xyzw']},
        *q.get('placement', [])]
    rows = []
    for item in expected:
        obj = session.env.scene.object_registry('name', item['name'])
        if obj is None:
            rows.append({'name': item['name'], 'accepted': False, 'reason': 'missing_native_actor'})
            continue
        p, quat = (array(v) for v in obj.get_position_orientation())
        want_q = item.get('quaternion_xyzw', item.get('orientation', [0., 0., 0., 1.]))
        position_error = float(np.linalg.norm(p - array(item['real_position'])))
        orientation_error = float(np.degrees((Rotation.from_quat(want_q).inv() * Rotation.from_quat(quat)).magnitude()))
        # The source common.py constructor supplies no scale or material override:
        # DatasetObject's default scale and installed model materials are required.
        scale = array(obj.scale)
        materials = sorted(str(material.prim_path) for material in obj.materials)
        checks = {'identity': obj.category == item['category'] and obj.model == item['model'],
            'position': position_error <= .01, 'orientation': orientation_error <= 1.,
            'source_default_scale': bool(np.allclose(scale, 1., atol=1e-5)),
            'source_visual_only': bool(obj.visual_only), 'native_materials_present': bool(materials)}
        rows.append({'name': item['name'], 'accepted': all(checks.values()), 'checks': checks,
            'position_error_m': position_error, 'orientation_error_deg': orientation_error,
            'scale': scale.tolist(), 'materials': materials})
    patches = adapter.variant.patches
    allowed = {'robot_pose', 'prepare_category_interaction'}
    profile = getattr(session, 'reflection_sensor_profile', {})
    def permitted(patch):
        if patch.operation in allowed:
            return True
        return bool(_revised_profile(profile) and patch.operation == 'category_initialization' and
            patch.phase == 'after_postprocessing' and
            patch.module == 'wia.authoring.converters.specular_reflection_reflection_authoring' and
            patch.parameters.get('operation') == INITIALIZATION_OPERATION and
            patch.parameters.get('initialization_revision') == INITIALIZATION_REVISION and
            patch.parameters.get('reflection_sensor_profile') == profile)
    patch_check = all(permitted(patch) for patch in patches)
    return {'accepted': bool(rows) and patch_check and all(row['accepted'] for row in rows),
        'native_actors': rows, 'no_optical_mutation_patches': patch_check,
        'source_constructor': 'ESI-Bench/src/active_explore/tasks/specular_reflection/common.py',
        'material_contract': 'Unmodified installed DatasetObject model materials; no material override patch.',
        'visual_answerability_and_mirror_dependence_require_review': True}


def original_reference_provenance(source, image_root=None):
    """Resolve only the image declared by the immutable original ESI record."""
    raw, payload = source.raw, source.payload
    q = payload['question_data']
    declared = q.get('render', {}).get('image')
    declared_images = raw.get('image_paths_json', [])
    declared_images = json.loads(declared_images) if isinstance(declared_images, str) else declared_images
    roots = [source.path.parent]
    if image_root is not None:
        roots.insert(0, Path(image_root) / Path(source.relative_path).parent)
    found = next((root / declared for root in roots if declared and (root / declared).is_file()), None)
    valid = bool(declared and declared in declared_images and found)
    source_hash = hashlib.sha256(source.path.read_bytes()).hexdigest()
    image_hash = hashlib.sha256(found.read_bytes()).hexdigest() if valid else None
    publication, reason = None, None if valid else 'original_ESI_image_unavailable'
    if valid and image_root is not None:
        provenance_path = Path(image_root) / 'provenance.json'
        if not provenance_path.is_file():
            valid, reason = False, 'external_original_archive_provenance_missing'
        else:
            provenance = json.loads(provenance_path.read_text())
            records = [row for row in provenance.get('records', []) if row.get('source_identity') == source.relative_path]
            publication = records[0] if len(records) == 1 else None
            metadata = publication.get('published_metadata', {}) if publication else {}
            matched = bool(publication and publication.get('source_sha256') == source_hash and
                publication.get('image_sha256') == image_hash and publication.get('adapted_reference') is False and
                publication.get('origin') == 'published_ESI_passive_single_reference' and
                Path(publication.get('original_image_file', '')).resolve() == found.resolve() and
                str(publication.get('url', '')).startswith('https://huggingface.co/datasets/ESI-Bench/esi-bench-passive-single/resolve/') and
                all(str(metadata.get(key)) == str(raw.get(key)) for key in (
                    'id', 'big_task', 'small_task', 'scene', 'room', 'question', 'answer')))
            if not matched:
                valid, reason = False, 'published_original_image_source_or_pixel_provenance_mismatch'
    image_size = None
    if valid:
        from PIL import Image
        with Image.open(found) as image:
            image_size = list(image.size)
    return {'kind': 'original_ESI_image', 'source_identity': source.relative_path,
        'source_sha256': source_hash,
        'declared_image': declared, 'image_path': str(found.resolve()) if valid else None,
        'image_sha256': image_hash if valid else None,
        'available': valid, 'reason': reason, 'publication_provenance': deepcopy(publication),
        'decoded_image_size_px': image_size,
        'adapted_postrobot_reference_allowed': False,
        'camera_pose': deepcopy(q.get('render', {}).get('camera_pose') or q['scene_setup']['camera'])}


def normalized_image_equivalence(original_rgb, mounted_rgb, source_intrinsics, mounted_intrinsics,
                                 source_task_features, mounted_task_features, *, tolerances):
    """Compare actual pixels and named normalized task features without warping.

    Intrinsics dictionaries require ``image_size`` (width,height), a rendered
    3x3 ``matrix``, ``projection`` and ``clipping_range_m``. Task feature lists
    contain matching ``binding``, ``feature_id``, ``view_kind`` (direct/reflected)
    and ``xy_normalized`` coordinates. The caller explicitly chooses/version
    tolerances; metadata, generic background matches or a fitted homography
    cannot substitute for task-content agreement. This pure helper never moves
    actors/cameras or advances simulation.
    """
    import cv2
    required = ('minimum_pixel_matches', 'maximum_hamming_distance', 'maximum_normalized_pixel_error',
        'maximum_normalized_task_feature_error', 'maximum_normalized_intrinsics_error',
        'maximum_clipping_error_m')
    if any(key not in tolerances for key in required):
        raise ValueError('Source-view comparison requires explicit calibrated tolerances.')
    def normalized_intrinsics(info):
        width, height = info['image_size']
        matrix = np.asarray(info['matrix'], dtype=float)
        if matrix.shape != (3, 3) or min(width, height) <= 0:
            raise ValueError('Effective rendered camera intrinsics are unavailable.')
        return matrix[:2] / np.asarray([[width], [height]])
    original, mounted = np.asarray(original_rgb), np.asarray(mounted_rgb)
    if original.ndim != 3 or mounted.ndim != 3 or original.shape[2] < 3 or mounted.shape[2] < 3:
        raise ValueError('Source and mounted RGB pixels are required.')
    intrinsic_error = float(np.max(np.abs(normalized_intrinsics(source_intrinsics) - normalized_intrinsics(mounted_intrinsics))))
    aspect_error = abs(original.shape[1] / original.shape[0] - mounted.shape[1] / mounted.shape[0])
    sizes_match = (list(source_intrinsics['image_size']) == [original.shape[1], original.shape[0]] and
        list(mounted_intrinsics['image_size']) == [mounted.shape[1], mounted.shape[0]])
    clipping_error = float(np.max(np.abs(np.asarray(source_intrinsics['clipping_range_m']) - mounted_intrinsics['clipping_range_m'])))
    # Normalize resolution only. No crop, perspective warp or homography may
    # hide a rendered projection/content mismatch.
    width, height = source_intrinsics['image_size']
    resized = cv2.resize(mounted[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=1500)
    first_points, first_descriptors = orb.detectAndCompute(cv2.cvtColor(original[:, :, :3], cv2.COLOR_RGB2GRAY), None)
    second_points, second_descriptors = orb.detectAndCompute(cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY), None)
    matches = [] if first_descriptors is None or second_descriptors is None else cv2.BFMatcher(
        cv2.NORM_HAMMING, crossCheck=True).match(first_descriptors, second_descriptors)
    errors = [float(np.linalg.norm((np.asarray(first_points[match.queryIdx].pt) -
        np.asarray(second_points[match.trainIdx].pt)) / [width, height])) for match in matches
        if match.distance <= tolerances['maximum_hamming_distance']]
    pixel_matches = sum(error <= tolerances['maximum_normalized_pixel_error'] for error in errors)
    def feature_key(row):
        return row['binding'], row['feature_id'], row['view_kind']
    current = {feature_key(row): row for row in mounted_task_features}
    feature_rows = []
    for source in source_task_features:
        row = current.get(feature_key(source))
        error = float(np.linalg.norm(np.asarray(source['xy_normalized']) - row['xy_normalized'])) if row else None
        feature_rows.append({'binding': source['binding'], 'feature_id': source['feature_id'],
            'view_kind': source['view_kind'], 'error': error,
            'accepted': error is not None and error <= tolerances['maximum_normalized_task_feature_error']})
    checks = {'actual_image_dimensions_match_intrinsics': sizes_match,
        'normalized_intrinsics_match': intrinsic_error <= tolerances['maximum_normalized_intrinsics_error'],
        'aspect_ratio_match': aspect_error <= tolerances['maximum_normalized_intrinsics_error'],
        'same_projection': source_intrinsics['projection'] == mounted_intrinsics['projection'],
        'same_clipping': clipping_error <= tolerances['maximum_clipping_error_m'],
        'rendered_pixel_features_match': pixel_matches >= tolerances['minimum_pixel_matches'],
        'named_task_content_matches': bool(feature_rows) and all(row['accepted'] for row in feature_rows)}
    return {'available': True, 'accepted': all(checks.values()), 'checks': checks,
        'pixel_match_count': pixel_matches, 'normalized_intrinsics_error': intrinsic_error,
        'aspect_ratio_error': aspect_error, 'clipping_error_m': clipping_error,
        'task_feature_matches': feature_rows, 'tolerances': deepcopy(tolerances),
        'normalization': 'full_image_resolution_only_no_projection_warp'}


def reflection_equivalence_identity(session, config, controller):
    """Bind equivalence to the exact reset, task, controller, sensors and optics."""
    task = session.scene_adapter.task
    head = session.robot.sensors[session.camera_name]
    sensors = {}
    for name, sensor in session.robot.sensors.items():
        sensors[name] = {'prim_path': str(sensor.prim_path),
            'mount': [array(value).round(8).tolist() for value in sensor.get_position_orientation(frame='parent')]}
    actors = {}
    stage = session.pipeline.og.sim.stage
    for key in config['parameters']['expected_optical_poses']:
        obj = session.env.scene.object_registry('name', config['bindings'][key])
        materials = {}
        for material in obj.materials:
            prim = stage.GetPrimAtPath(str(material.prim_path))
            attributes = {}
            pending = [prim]
            while pending:
                child = pending.pop()
                pending.extend(child.GetAllChildren())
                for attribute in child.GetAttributes():
                    attributes[str(attribute.GetPath())] = {'value': str(attribute.Get()),
                        'connections': [str(value) for value in attribute.GetConnections()]}
            materials[str(material.prim_path)] = attributes
        actors[key] = {'name': obj.name, 'category': obj.category, 'model': obj.model,
            'scale': array(obj.scale).round(8).tolist(),
            'pose': [array(value).round(8).tolist() for value in obj.get_position_orientation()], 'materials': materials}
    head_intrinsics = {key: str(getattr(head, key)) for key in (
        'focal_length', 'horizontal_aperture', 'image_width', 'image_height', 'clipping_range')}
    head_intrinsics['vertical_aperture'] = str(head.get_attribute('verticalAperture'))
    runtime = {'controller_class': type(controller).__module__ + '.' + type(controller).__qualname__,
        'controller_instance': str(id(controller)), 'sensor_mounts': sensors,
        'head_intrinsics': head_intrinsics, 'actors': actors, 'robot_scale': array(session.robot.scale).tolist(),
        'controller_config': repr(getattr(controller, 'config', None))}
    return {'task_id': task.definition.id, 'task_sha256': task.file_sha256,
        'source_sha256': task.source_sha256, 'source_identity': config['parameters']['source_qa']['source_path'],
        'reset_id': session.native_reset_id, 'seed': task.seed,
        'runtime_sha256': hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()}


def diagnose_original_source_view(session, controller, config, observation, *, source_task_features,
                                  mounted_task_features, source_intrinsics, mounted_intrinsics, tolerances):
    """Read an immutable source image and current delivered head RGB; never render/set.

    Caller supplies manually/native-verified feature identities and actual
    effective rendered intrinsics. Missing original provenance stays unavailable.
    The returned receipt can be saved as
    ``session.reflection_qualification['original_source_view_equivalence']``.
    """
    from PIL import Image
    reference = config['parameters']['original_reference_provenance']
    result = {'available': False, 'accepted': False, 'reference': deepcopy(reference)}
    try:
        if reference.get('kind') != 'original_ESI_image' or not reference.get('available'):
            raise ValueError('The true original ESI reference image is unavailable.')
        image_path = Path(reference['image_path'])
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != reference['image_sha256']:
            raise ValueError('Original ESI reference pixels changed.')
        images = observation.rgb_images()
        camera_key = 'agentview_rgb' if 'agentview_rgb' in images else 'head'
        mounted = images[camera_key]
        original = np.asarray(Image.open(image_path).convert('RGB'))
        comparison = normalized_image_equivalence(original, mounted, source_intrinsics, mounted_intrinsics,
            source_task_features, mounted_task_features, tolerances=tolerances)
        identity = reflection_equivalence_identity(session, config, controller)
        if identity['source_sha256'] != reference['source_sha256']:
            raise ValueError('Original reference source differs from the loaded task source.')
        result.update(comparison, identity=identity, snapshot_id=observation.snapshot_id,
            action_step=observation.action_step, camera=camera_key,
            mounted_rgb_sha256=hashlib.sha256(np.asarray(mounted).tobytes()).hexdigest(),
            native_rgb_matches_delivered=True)
        if observation.action_step != 0:
            result.update(accepted=False, reason='Source-view equivalence must use the initial native head image.')
    except (ValueError, KeyError, AttributeError, TypeError, OSError) as error:
        result.update(reason=f'{type(error).__name__}: {error}')
    return result

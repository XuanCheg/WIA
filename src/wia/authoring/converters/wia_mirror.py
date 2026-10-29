"""Initialize native panel PBR and measure WIA mirror-task geometry read-only.

Run generated mirror tasks through src/wia/runtime/commands/check_esi_core_native.py --evaluate.
CPU checks: python -B -m pytest -q -p no:cacheprovider tests/test_wia_mirror_adapter.py
Only initialize_scene under the existing initialization flag writes panel PBR.
Episode measurements never render, move objects/cameras, step physics or expose
private optical identity/coordinates through public prompts.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import wia.authoring.converters.active_layout as active_layout
import wia.authoring.converters.wia_dimensional as native
from wia.authoring.io import write_json
from wia.tasks.completion import progress
from wia.tasks.wia_mirror_scoring import DESTINATION_REVISION, MirrorProgress, destination_contracts

MODULE = 'wia.authoring.converters.wia_mirror'
REVISION = 'wia_mirror_v1'
HOUSEHOLD_REVISION = 'wia_mirror_v3'
HOUSEHOLD_REVISIONS = (HOUSEHOLD_REVISION, DESTINATION_REVISION)
REVISIONS = (REVISION, 'wia_mirror_v2', *HOUSEHOLD_REVISIONS)
OPERATIONS = ('mirror_select_deliver', 'classify_appearance_deliver', 'mirror_reconfigure')
PANEL_NAME = 'wia_mirror_panel'
MATERIAL = {'metallic_constant': 1., 'reflection_roughness_constant': 0.,
            'diffuse_color_constant': [.95, .95, .95]}
MATERIAL_TOLERANCE = 1e-6
ERRORS = native.ERRORS


def _initialization_contract(parameters):
    if parameters.get('revision') == DESTINATION_REVISION:
        destination_contracts(parameters)
    if parameters.get('revision') in ('wia_mirror_v2', *HOUSEHOLD_REVISIONS):
        scene_model, room = native._scene_contract(parameters)
        if (parameters['layout']['scene_model'] != scene_model
                or parameters['layout']['rooms'] != [room]):
            raise ValueError('Mirror initialization layout must match its explicit scene contract.')
    elif (parameters.get('revision') != REVISION or parameters['layout']['scene_model'] != native.SCENE
            or parameters['layout']['rooms'] != [native.ROOM]):
        raise ValueError('Mirror initialization requires its exact revision, native scene and room.')


def configure_scene(config, parameters):
    _initialization_contract(parameters)
    return active_layout.configure_scene(config, parameters)


def postprocess_joint_world(session, parameters):
    _initialization_contract(parameters)
    return active_layout.postprocess_joint_world(session, parameters)


def _panel_for_initialization(session, parameters):
    key = parameters['mirror_key']
    if parameters['mirror_material'].get('binding') != key:
        raise ValueError('The material and mirror semantic bindings must match.')
    binding = getattr(getattr(session, 'scene_adapter', None), 'bindings', {}).get(key)
    name = (binding.get('name') if isinstance(binding, dict) else getattr(binding, 'name', None)) if binding is not None else None
    if name is None:
        names = [row['name'] for row in parameters['layout']['objects'] if row.get('name') == PANEL_NAME]
        if names != [PANEL_NAME]:
            raise ValueError('No exact authored native mirror panel can be resolved.')
        name = PANEL_NAME
    panel = session.env.scene.object_registry('name', name)
    if panel is None or panel.name != name:
        raise ValueError('The resolved native mirror panel is unavailable.')
    return panel


def _visual_materials(panel):
    result = []
    for link in panel.links.values():
        if native._boolean(link.is_meta_link):
            continue
        for mesh in link.visual_meshes.values():
            material = mesh.material
            if material is None:
                raise ValueError('An actual panel visual mesh has no native PBR material.')
            result.append((str(mesh.prim_path), material))
    if not result:
        raise ValueError('No actual panel visual PBR material is measurable.')
    return result


def _material_readback(panel):
    result = {'available': False, 'accepted': False, 'materials': [], 'errors': [],
              'tolerance': MATERIAL_TOLERANCE, 'source': 'Current native panel visual MaterialPrim getters; no requested-value substitution.'}
    try:
        for mesh_path, material in _visual_materials(panel):
            row = {'visual_mesh_path': mesh_path, 'material_prim_path': str(material.prim_path),
                   'available': False, 'accepted': False}
            try:
                values = {key: float(native.array(material.get_input(inp=key)))
                          for key in ('metallic_constant', 'reflection_roughness_constant')}
                values['diffuse_color_constant'] = native._vector(material.diffuse_color_constant, 3).tolist()
                row.update(values, available=True,
                    accepted=bool(all(np.all(np.abs(np.asarray(values[key]) - expected) <= MATERIAL_TOLERANCE)
                                      for key, expected in MATERIAL.items())))
            except ERRORS as error:
                row['measurement_error'] = f'{type(error).__name__}: {error}'
            result['materials'].append(row)
        result['available'] = all(row['available'] for row in result['materials'])
        result['accepted'] = result['available'] and all(row['accepted'] for row in result['materials'])
    except ERRORS as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def initialize_scene(session, parameters):
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Mirror material writes are initialization-only.')
    _initialization_contract(parameters)
    if parameters['revision'] in ('wia_mirror_v2', *HOUSEHOLD_REVISIONS):
        native._validate_scene_identity(session, *native._scene_contract(parameters))
    elif session.scene_name != native.SCENE or session.room_name != native.ROOM:
        raise ValueError('Mirror initialization must run in the declared native scene and room.')
    requested = {key: parameters['mirror_material'].get(key) for key in MATERIAL}
    if requested != MATERIAL:
        raise ValueError('Only the explicitly designed native mirror PBR values may be initialized.')
    receipt = {'requested_material': deepcopy(requested), 'setter_count': 0, 'errors': []}
    panel = None
    try:
        panel = _panel_for_initialization(session, parameters)
        receipt['native_panel_name'] = panel.name
        receipt['before'] = _material_readback(panel)
        import torch
        seen = set()
        for mesh_path, material in _visual_materials(panel):
            identity = (id(material), str(material.prim_path))
            if identity in seen:
                continue
            seen.add(identity)
            for key in ('metallic_constant', 'reflection_roughness_constant'):
                receipt['setter_count'] += 1
                material.set_input(inp=key, val=MATERIAL[key])
            receipt['setter_count'] += 1
            material.diffuse_color_constant = torch.tensor(MATERIAL['diffuse_color_constant'], dtype=torch.float32)
    except ERRORS as error:
        receipt['errors'].append(f'{type(error).__name__}: {error}')
    receipt['after'] = _material_readback(panel)
    receipt['accepted'] = receipt['after']['accepted'] is True and not receipt['errors']
    setup = active_layout.initialize_scene(session, parameters)
    setup['mirror_material_initialization'] = receipt
    session.category_interaction_setup = setup
    return setup


def _config(session, config):
    p = config['parameters']
    if (module_identity(config.get('module')) != MODULE or config.get('operation') not in OPERATIONS
            or p.get('revision') not in REVISIONS or p.get('operation') != config['operation']):
        raise ValueError('Use the exact mirror revision and owned operation.')
    if p['revision'] in ('wia_mirror_v2', *HOUSEHOLD_REVISIONS):
        scene_model, room = native._scene_contract(p)
        if config.get('scene') != scene_model or config.get('room') != room:
            raise ValueError('The mirror task must match its explicit scene contract.')
        native._validate_scene_identity(session, scene_model, room)
    elif (config.get('scene') != native.SCENE or config.get('room') != native.ROOM
            or session.scene_name != config['scene'] or session.room_name != config['room']
            or session.env.scene.scene_model != config['scene']):
        raise ValueError('The mirror task and actual native scene/room must match.')
    if (not isinstance(p['candidates'], dict) or len(p['candidates']) < 2
            or len(set(p['candidates'].values())) != len(p['candidates'])
            or any(config['bindings'].get(key) != value for key, value in p['candidates'].items())):
        raise ValueError('Mirror candidates require exact distinct native identities.')
    if p['revision'] in HOUSEHOLD_REVISIONS:
        assets = p.get('candidate_assets')
        if (not isinstance(assets, dict) or set(assets) != set(p['candidates'])
                or 'candidate_dimensions_m' in p):
            raise ValueError('Household targets require explicit assets and no proposed Cube dimensions.')
        for asset in assets.values():
            if (not isinstance(asset, dict) or set(asset) != {'category', 'model', 'scale'}
                    or any(not isinstance(asset[field], str) or not asset[field].strip()
                           for field in ('category', 'model'))):
                raise ValueError('Each household target requires exact category, model and scale.')
            native._vector(asset['scale'], 3, positive=True)
    else:
        if set(p['candidate_dimensions_m']) != set(p['candidates']):
            raise ValueError('All legacy candidate Cube dimensions must be proposed.')
        for dimensions in p['candidate_dimensions_m'].values():
            native._vector(dimensions, 3, positive=True)
    if set(p['cue_bindings']) != {'lower_side', 'upper_side', 'roof', 'floor'} or set(p['destinations']) != {'blue', 'yellow'}:
        raise ValueError('Mirror tasks require the explicit four-part cue and both semantic destinations.')
    required = {p['mirror_key'], p['table_key'], p['anchor_key'], *p['cue_bindings'].values(), *p['destinations'].values()}
    if (not required.issubset(p['fixture_keys']) or not set(p['fixture_keys']).issubset(config['bindings'])
            or len(set(p['fixture_keys'])) != len(p['fixture_keys'])):
        raise ValueError('All apparatus, anchor and destinations need fixed native bindings.')
    if p['revision'] == DESTINATION_REVISION:
        destination_contracts(p)
    MirrorProgress(p)  # Validate the fixed pure contract without reading native state.
    return p


def _camera(session):
    if hasattr(session, 'robot_camera_registry'):
        name = session.robot_camera_registry['agentview_rgb']
    else:
        name = session.camera_name
    sensor = session.robot.sensors[name]
    position, quaternion = sensor.get_position_orientation()
    position, quaternion = native._vector(position, 3), native._vector(quaternion, 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('The native head camera quaternion is unavailable.')
    intrinsic = native.array(sensor.intrinsic_matrix)
    width, height = sensor.image_width, sensor.image_height
    if (intrinsic.shape != (3, 3) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
            or not np.allclose(intrinsic[2], [0., 0., 1.], rtol=0., atol=1e-12)
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
                   for value in (width, height))):
        raise ValueError('The actual head camera calibration/frame is unavailable.')
    return {'sensor_name': str(name), 'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'intrinsic_matrix': intrinsic.tolist(), 'image_width': int(width), 'image_height': int(height),
            'projection_convention': 'Native camera optical -Z, up +Y; positive-depth image coordinates.'}


def _optics(sample, config, camera):
    p = config['parameters']
    result = {'available': False, 'accepted': False, 'selected_candidate': None,
              'appearance_mode': None, 'frozen_eye_m': deepcopy(camera['position_m']),
              'camera': deepcopy(camera), 'candidates': {}, 'errors': [],
              'optical_visibility_verified': False, 'optical_necessity_verified': False}
    try:
        if sample['mirror_material']['accepted'] is not True:
            raise ValueError('Actual current panel PBR material is unavailable or does not match the reflective contract.')
        fixtures, scoring = sample['fixtures'], p['scoring']
        panel = fixtures[p['mirror_key']]
        rods = {name: fixtures[key] for name, key in p['cue_bindings'].items()}
        alignments = {name: native._axis_alignment(row['quaternion_xyzw'], scoring['fixture_angle_tolerance_rad'])
                      for name, row in {'panel': panel, **rods}.items()}
        if not all(row['accepted'] for row in alignments.values()):
            raise ValueError('Native panel or cue rods are not axis aligned within the declared tolerance.')
        panel_points = native.array(panel['vertices_m'])
        panel_low, panel_high = panel_points.min(axis=0), panel_points.max(axis=0)
        if np.any(panel_high <= panel_low):
            raise ValueError('The finite native panel rectangle is degenerate.')
        rod_points = {name: native.array(row['vertices_m']) for name, row in rods.items()}
        rod_x = [float(points[:, 0].mean()) for points in rod_points.values()]
        cue_x = float(np.mean(rod_x))
        if max(abs(value - cue_x) for value in rod_x) > scoring['fixture_position_tolerance_m']:
            raise ValueError('Measured cue rod planes are inconsistent.')
        opening = {'y_min_m': float(rod_points['lower_side'][:, 1].max()),
                   'y_max_m': float(rod_points['upper_side'][:, 1].min()),
                   'z_min_m': float(rod_points['floor'][:, 2].max()),
                   'z_max_m': float(rod_points['roof'][:, 2].min())}
        margin = scoring['aperture_clearance_margin_m']
        if any(opening[high] - opening[low] < 2 * margin for low, high in (
                ('y_min_m', 'y_max_m'), ('z_min_m', 'z_max_m'))):
            raise ValueError('The measured inset cue opening is empty.')
        centers = {key: native.array(row['vertices_m']).mean(axis=0) for key, row in sample['targets'].items()}
        if any(center.shape != (3,) for center in centers.values()):
            raise ValueError('Native candidate centroids are unavailable.')
        eye, panel_x = native.array(camera['position_m']), float(panel_low[0])
        tolerance = scoring['fixture_position_tolerance_m']
        if not all(eye[0] < value for value in (cue_x, panel_x, *(center[0] for center in centers.values()))):
            raise ValueError('The native head eye is not in front of the candidate/cue/panel geometry.')
        direct = cue_x < min(center[0] for center in centers.values()) - tolerance
        reflected = max(center[0] for center in centers.values()) + tolerance < cue_x < panel_x
        if direct == reflected:
            raise ValueError('Native cue depth does not determine direct or reflected appearance.')
        mode = 'direct' if direct else 'reflected'
        result.update(appearance_mode=mode, declared_appearance_mode=p['appearance_mode'],
                      panel_plane={'point_m': [panel_x, float(panel_low[1]), float(panel_low[2])],
                                   'normal': [-1., 0., 0.], 'bounds_m': [panel_low.tolist(), panel_high.tolist()]},
                      cue_plane_x_m=cue_x, cue_opening=opening, rod_plane_x_m=rod_x, axis_alignment=alignments)
        rotation = Rotation.from_quat(camera['quaternion_xyzw'])
        intrinsic = native.array(camera['intrinsic_matrix'])
        for key, center in centers.items():
            normal, panel_point = np.array([-1., 0., 0.]), np.array([panel_x, 0., 0.])
            virtual = center if direct else center - 2 * normal * np.dot(normal, center - panel_point)
            ray = virtual - eye
            row = {'centroid_m': center.tolist(), 'virtual_point_m': virtual.tolist(), 'rejection_reasons': []}
            if ray[0] == 0:
                row['rejection_reasons'].append('ray_parallel_to_cue')
            else:
                t = float((cue_x - eye[0]) / ray[0])
                hit = eye + t * ray
                row.update(cue_ray_t=t, cue_intersection_m=hit.tolist())
                if not (0 < t < 1 and opening['y_min_m'] + margin <= hit[1] <= opening['y_max_m'] - margin
                        and opening['z_min_m'] + margin <= hit[2] <= opening['z_max_m'] - margin):
                    row['rejection_reasons'].append('cue_intersection_outside_inset_opening')
                if reflected:
                    panel_t = float((panel_x - eye[0]) / ray[0])
                    panel_hit = eye + panel_t * ray
                    row.update(panel_ray_t=panel_t, panel_intersection_m=panel_hit.tolist())
                    if not (0 < panel_t < 1 and np.all(panel_hit[1:] >= panel_low[1:])
                            and np.all(panel_hit[1:] <= panel_high[1:])):
                        row['rejection_reasons'].append('reflection_outside_finite_panel')
            local = rotation.inv().apply(virtual - eye) * [1., -1., -1.]
            row['camera_optical_point_m'] = local.tolist()
            row['depth_m'] = float(local[2])
            if local[2] <= 0:
                row['rejection_reasons'].append('nonpositive_camera_depth')
            else:
                pixel = intrinsic @ local
                uv = pixel[:2] / pixel[2]
                if not np.isfinite(uv).all():
                    raise ValueError('The native camera projection is nonfinite.')
                row['pixel_xy'] = uv.tolist()
                if not (0 <= uv[0] < camera['image_width'] and 0 <= uv[1] < camera['image_height']):
                    row['rejection_reasons'].append('outside_native_camera_frame')
            row['accepted'] = not row['rejection_reasons']
            result['candidates'][key] = row
        passing = [key for key, row in result['candidates'].items() if row['accepted']]
        result['passing_candidates'] = passing
        if len(passing) != 1:
            raise ValueError('Measured optical selection is absent or ambiguous; exactly one candidate is required.')
        if mode != p['appearance_mode']:
            raise ValueError('Measured appearance depth does not match the configured proposal.')
        result.update(available=True, accepted=True, selected_candidate=passing[0])
    except ERRORS as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def capture_initial_geometry(session, config, bound):
    """Read private capture-boundary geometry without qualification or simulation.

    Availability describes native readback completeness. The existing optical
    selection may separately reject measured geometry, including ambiguous rays.
    """
    errors = []
    result = {'schema': 'wia_mirror_initial_geometry_v1', 'available': False,
        'native_time_s': None, 'camera': {'available': False, 'errors': []},
        'targets': {}, 'fixtures': {},
        'mirror_material': {'available': False, 'accepted': False, 'errors': []},
        'optical_selection': {'available': False, 'accepted': False, 'selected_candidate': None,
            'appearance_mode': None, 'frozen_eye_m': None, 'errors': []},
        'native_qualification_run': False, 'optical_visibility_verified': False,
        'optical_necessity_verified': False, 'physical_success_verified': False,
        'availability_scope': 'Complete declared native identities, time, camera, collision geometry and material readbacks; optical acceptance is separate.',
        'errors': errors}

    def finish():
        def safe(value):
            if isinstance(value, dict):
                return {str(key): safe(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return safe(value.tolist())
            if isinstance(value, np.generic):
                return safe(value.item())
            if isinstance(value, (list, tuple)):
                return [safe(item) for item in value]
            if value is None or isinstance(value, (bool, int, str)):
                return value
            if isinstance(value, float) and np.isfinite(value):
                return value
            errors.append('serialization: A native diagnostic value is nonfinite or not JSON-compatible.')
            return None

        detached = safe(result)
        detached['errors'] = list(errors)
        detached['available'] = not errors
        return detached

    try:
        p = _config(session, config)
    except ERRORS as error:
        errors.append(f'configuration: {type(error).__name__}: {error}')
        return finish()
    identity_errors = {}
    for key, expected in config['bindings'].items():
        try:
            obj = bound.get(key)
            if obj is None or obj.name != expected:
                raise ValueError('The bound object does not have its exact declared native name.')
        except ERRORS as error:
            identity_errors[key] = f'identity:{key}: {type(error).__name__}: {error}'
            errors.append(identity_errors[key])
    try:
        value = session.pipeline.og.sim.current_time
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value)):
            raise ValueError('Native physics time must be finite.')
        result['native_time_s'] = float(value)
    except ERRORS as error:
        errors.append(f'clock: {type(error).__name__}: {error}')
    try:
        result['camera'] = {**_camera(session), 'available': True, 'errors': []}
    except ERRORS as error:
        message = f'camera: {type(error).__name__}: {error}'
        result['camera']['errors'].append(message)
        errors.append(message)
    for group, keys in (('targets', p['candidates']), ('fixtures', p['fixture_keys'])):
        for key in keys:
            row = {'available': False, 'object_name': None, 'errors': []}
            if key in identity_errors:
                row['errors'].append(identity_errors[key])
            else:
                try:
                    obj = bound[key]
                    geometry = (_target_geometry(obj, p, key) if group == 'targets'
                                and p['revision'] in HOUSEHOLD_REVISIONS else _fixture_geometry(obj, p, key))
                    row.update(geometry, object_name=obj.name, available=True)
                except ERRORS as error:
                    message = f'{group}:{key}: {type(error).__name__}: {error}'
                    row['errors'].append(message)
                    errors.append(message)
            result[group][key] = row
    if p['mirror_key'] not in identity_errors:
        result['mirror_material'] = _material_readback(bound.get(p['mirror_key']))
    else:
        result['mirror_material']['errors'].append(identity_errors[p['mirror_key']])
    if result['mirror_material']['available'] is not True:
        errors.append('mirror_material: Complete actual native panel material readback is unavailable.')
    if result['camera']['available'] is True:
        minimal = {key: result[key] for key in ('targets', 'fixtures', 'mirror_material')}
        result['optical_selection'] = _optics(minimal, config, result['camera'])
    else:
        result['optical_selection']['errors'].append('Native camera measurement is unavailable; no eye or calibration was substituted.')
    return finish()


def _household_geometry(obj):
    """Keep native world vertices and measure body shape in local USD space.

    USD matrices use row-vector convention. The relative transform excludes the
    root link's transform/scale; apply its measured scale exactly once, in
    float64, without reconstructing body points from large world coordinates.
    """
    import omnigibson.lazy as lazy
    from omnigibson.objects.dataset_object import DatasetObject

    if not isinstance(obj, DatasetObject):
        raise ValueError('Household geometry requires an actual native DatasetObject.')
    counts = {}
    for field in ('n_dof', 'n_joints'):
        value = getattr(obj, field)
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value) or value < 0 or int(value) != value):
            raise ValueError('The measured native ' + field + ' must be a finite nonnegative integer.')
        counts[field] = int(value)
    if any(counts.values()):
        raise ValueError('Household rigid geometry requires measured native n_dof == 0 and n_joints == 0.')
    # OG can report articulated=True for fixed metadata joints. Actual movable
    # degree/joint counts and the collision-link structure define this contract.
    articulated = native._boolean(obj.articulated)
    root = obj.root_link
    links = [link for link in obj.links.values() if not native._boolean(link.is_meta_link)
             and any(native._boolean(mesh.collision_enabled) for mesh in link.collision_meshes.values())]
    if len(links) != 1 or links[0] is not root:
        raise ValueError('Local intrinsic geometry requires one enabled collision link equal to root_link.')
    root_path = str(root.prim.GetPath())
    if not root_path.startswith('/') or root_path != str(root.prim_path):
        raise ValueError('The native root-link USD identity is unavailable.')
    root_scale = native._vector(root.scale, 3, positive=True)
    cache = lazy.pxr.UsdGeom.XformCache()
    position, quaternion = obj.get_position_orientation()
    position, quaternion = native._vector(position, 3), native._vector(quaternion, 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('Native quaternion is zero.')
    scale = native._vector(obj.scale, 3, positive=True)
    points, body_points, meshes = [], [], []
    for mesh in root.collision_meshes.values():
        if not native._boolean(mesh.collision_enabled):
            continue
        local = mesh.points
        raw = native.array(local)
        if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
            raise ValueError('Actual native collision vertices are unavailable.')
        shape_stamp = native._native_shape_stamp(mesh)
        if shape_stamp['type'] != 'Mesh':
            raise ValueError('The native household collision geometry must expose uncached USD Mesh points and topology.')
        current_points = native.array(mesh.prim.GetAttribute('points').Get())
        if (current_points.shape != raw.shape or
                not np.array_equal(current_points.astype(np.float32), raw.astype(np.float32))):
            raise ValueError('Cached wrapper collision points do not match current USD Mesh points at float32 storage accuracy.')
        mesh_path = str(mesh.prim.GetPath())
        if mesh_path != str(mesh.prim_path) or not mesh_path.startswith(root_path + '/'):
            raise ValueError('Collision mesh must be a measured descendant of the root link.')
        matrix, reset = cache.ComputeRelativeTransform(mesh.prim, root.prim)
        if native._boolean(reset):
            raise ValueError('resetXformStack breaks the collision-mesh-to-root-link hierarchy.')
        matrix = native.array(matrix)
        if (matrix.shape != (4, 4)
                or not np.allclose(matrix[:, 3], [0., 0., 0., 1.], rtol=0., atol=1e-12)
                or np.linalg.det(matrix[:3, :3]) == 0):
            raise ValueError('Native mesh-to-root transform must be a finite nonsingular affine 4 x 4 matrix.')
        homogeneous = np.concatenate((raw, np.ones((len(raw), 1))), axis=1) @ matrix
        if not np.allclose(homogeneous[:, 3], 1., rtol=0., atol=1e-12):
            raise ValueError('Native mesh-to-root transform did not preserve homogeneous coordinates.')
        body = homogeneous[:, :3] * root_scale
        if not np.isfinite(body).all():
            raise ValueError('Native local collision vertices are nonfinite.')
        faces = mesh.faces
        if hasattr(faces, 'detach'):
            faces = faces.detach().cpu().numpy()
        faces = np.asarray(faces)
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in 'iu'
                or np.any(faces < 0) or np.any(faces >= len(raw))):
            raise ValueError('Native collision triangle topology is unavailable or invalid.')
        world = native.array(mesh.transform_local_points_to_world(local))
        if world.shape != raw.shape:
            raise ValueError('Native collision vertex transformation changed its array shape.')
        points.append(world)
        body_points.append(body)
        meshes.append({'prim_path': str(mesh.prim_path), 'local_vertices_m': raw.tolist(),
                       'faces': faces.tolist(), 'mesh_to_root_link_transform': matrix.tolist(),
                       'reset_xform_stack': False, 'native_shape_stamp': shape_stamp})
    if not points:
        raise ValueError('No enabled non-meta native collision geometry is measurable.')
    vertices = np.concatenate(points)
    body = np.concatenate(body_points)
    dimensions = np.ptp(body, axis=0)
    if np.any(dimensions <= 0):
        raise ValueError('Native collision geometry must have three positive intrinsic dimensions.')
    low, high = (native._vector(value, 3) for value in obj.aabb)
    if np.any(high <= low):
        raise ValueError('Native collision AABB is unavailable.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'scale': scale.tolist(), 'vertices_m': vertices.tolist(),
            'body_vertices_m': body.tolist(), 'dimensions_m': dimensions.tolist(),
            'root_link_prim_path': root_path, 'root_link_scale': root_scale.tolist(),
            'n_dof': counts['n_dof'], 'n_joints': counts['n_joints'], 'articulated': articulated,
            'intrinsic_geometry_source': 'Float64 mesh-to-root-link local USD transform followed by measured root-link scale once.',
            'aabb_m': [low.tolist(), high.tolist()], 'collision_meshes': meshes,
            'geometry_scope': 'Measured collision-mesh boundary bounds and vertex centroid; non-convexity is permitted; physical feasibility is unverified.'}


def _root_pose(obj):
    """Read actual root-link pose and scale without substituting object requests."""
    position, quaternion = obj.root_link.get_position_orientation()
    position, quaternion = native._vector(position, 3), native._vector(quaternion, 4)
    if np.linalg.norm(quaternion) == 0:
        raise ValueError('The measured destination root-link quaternion is zero.')
    return {'root_position_m': position.tolist(), 'root_quaternion_xyzw': quaternion.tolist(),
            'root_link_scale': native._vector(obj.root_link.scale, 3, positive=True).tolist()}


def _fixture_geometry(obj, parameters, key):
    """Preserve optical Cube rules while allowing v4 fixed rigid dataset fixtures."""
    optical = {parameters['mirror_key'], *parameters['cue_bindings'].values()}
    if parameters['revision'] != DESTINATION_REVISION or key in optical:
        return native._geometry(obj)
    geometry = (native._geometry(obj) if getattr(obj, '_primitive_type', None) == 'Cube'
                else _household_geometry(obj))
    geometry.update(_root_pose(obj))
    return geometry


def _root_fixture_preserved(current, initial, scoring):
    if 'root_position_m' not in current and 'root_position_m' not in initial:
        return True
    position = np.linalg.norm(native.array(current['root_position_m']) - initial['root_position_m'])
    angle = (Rotation.from_quat(current['root_quaternion_xyzw']).inv()
             * Rotation.from_quat(initial['root_quaternion_xyzw'])).magnitude()
    return bool(position <= scoring['fixture_position_tolerance_m']
                and angle <= scoring['fixture_angle_tolerance_rad'])


DESTINATION_BOUNDS_EPSILON_M = 1e-6


def _bounded_relation(target, destination, vertices, contract):
    """Combine native relations with every collision vertex in live root coordinates."""
    from omnigibson import object_states

    kind = contract['kind']
    if kind not in ('on_top', 'inside'):
        raise ValueError('Unsupported bounded destination relation kind.')
    relation_field = 'on_top' if kind == 'on_top' else 'inside'
    bound_field = 'footprint_inside' if kind == 'on_top' else 'vertices_inside'
    epsilon_m = DESTINATION_BOUNDS_EPSILON_M if kind == 'inside' else 0.
    result = {'kind': kind, 'touching': None, relation_field: None, bound_field: None,
              'bounds_epsilon_m': epsilon_m, 'measurement_errors': [],
              'bounds_frame': 'Unscaled native root-link local meters, using current root pose and scale.'}
    for field, state_name in (('touching', 'Touching'), (relation_field, 'OnTop' if kind == 'on_top' else 'Inside')):
        try:
            result[field] = native._boolean(target.states[getattr(object_states, state_name)].get_value(destination))
        except ERRORS as error:
            result['measurement_errors'].append(f'{field}: {type(error).__name__}: {error}')
    try:
        if vertices is None:
            raise ValueError('Actual target collision vertices are unavailable.')
        points = native.array(vertices)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError('Actual target collision vertices require a nonempty N x 3 array.')
        pose = _root_pose(destination)
        position, scale = native.array(pose['root_position_m']), native.array(pose['root_link_scale'])
        rotation = Rotation.from_quat(pose['root_quaternion_xyzw'])
        # Apply inverse rotation before undoing the measured root scale.
        local = rotation.inv().apply(points - position) / scale
        bounds = native.array(contract['usable_bounds_local_m'])
        if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
            raise ValueError('Explicit local usable bounds must have positive finite extent.')
        dimensions = slice(0, 2) if kind == 'on_top' else slice(0, 3)
        # Epsilon is in physical meters, irrespective of asset scaling.
        epsilon_local = epsilon_m / scale[dimensions]
        result[bound_field] = bool(np.all(local[:, dimensions] >= bounds[0, dimensions] - epsilon_local)
            and np.all(local[:, dimensions] <= bounds[1, dimensions] + epsilon_local))
        corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in bounds[:, 2]])
        result.update(pose, usable_bounds_local_m=bounds.tolist(),
            usable_bounds_world_vertices_m=(rotation.apply(corners * scale) + position).tolist(),
            target_collision_vertices_local_m=local.tolist())
    except ERRORS as error:
        result['measurement_errors'].append(f'bounds: {type(error).__name__}: {error}')
    return result


def _target_geometry(obj, parameters, key):
    """Verify the native asset identity before reading actual collision geometry."""
    asset = parameters['candidate_assets'][key]
    if (obj.name != parameters['candidates'][key] or obj.category != asset['category']
            or obj.model != asset['model']):
        raise ValueError('The native household target identity/category/model mismatches its contract.')
    measured = _household_geometry(obj)
    if not np.allclose(measured['scale'], asset['scale'], rtol=1e-6, atol=1e-6):
        raise ValueError('The measured household target scale mismatches its explicit asset contract.')
    return measured


def _measure_household(session, controller, config, objects, baseline=None):
    p, errors = config['parameters'], []
    control_valid, control = native._control(session, controller)
    grasp, raw_grasp = native._grasp(session, controller)
    sample = {'native_time_s': None, 'control_valid': control_valid, 'control_evidence': control,
              'fixtures_valid': None, 'geometry_valid': None, 'grasp': grasp,
              'native_grasp_diagnostics': raw_grasp, 'targets': {}, 'fixtures': {}, 'measurement_errors': errors}
    try:
        value = session.pipeline.og.sim.current_time
        if isinstance(value, bool) or not np.isfinite(float(value)):
            raise ValueError('Native physics time must be finite.')
        sample['native_time_s'] = float(value)
    except ERRORS as error:
        errors.append(f'clock: {type(error).__name__}: {error}')
    fixture_flags, geometry_flags = [], []
    for key in p['fixture_keys']:
        row, flag = {}, None
        try:
            obj = objects[key]
            fixed = native._collidable(obj, dynamic=False)
            flag = fixed
            row = _fixture_geometry(obj, p, key)
            row['fixed_collidable'] = fixed
            flag = fixed
            if baseline is not None:
                reference = baseline['fixtures'].get(key)
                if not reference or 'measurement_error' in reference:
                    flag = False if flag is False else None
                else:
                    flag = bool(fixed and native._fixture_valid(row, reference, p['scoring'])
                                and _root_fixture_preserved(row, reference, p['scoring']))
        except ERRORS as error:
            flag = False if flag is False else None
            row['measurement_error'] = f'{type(error).__name__}: {error}'
        row['preserved'] = flag
        sample['fixtures'][key] = row
        fixture_flags.append(flag)
    sample['fixtures_valid'] = native._all(fixture_flags)
    for key, name in p['candidates'].items():
        row = {'object_name': name, 'vertices_m': None, 'dimensions_m': None,
               'position_m': None, 'quaternion_xyzw': None, 'lowest_z_m': None,
               'linear_speed_m_s': None, 'angular_speed_rad_s': None,
               'dynamic_collidable': None, 'robot_contact': None, 'destinations': {}, 'measurement_errors': []}
        obj, geometry_flag = objects[key], None
        try:
            row['dynamic_collidable'] = native._collidable(obj, dynamic=True)
            measured = _target_geometry(obj, p, key)
            row['object_name'] = obj.name
            row['native_dataset_object'] = True
            row.update({field: measured[field] for field in (
                'vertices_m', 'dimensions_m', 'position_m', 'quaternion_xyzw')})
            row['native_geometry'] = measured
            row['lowest_z_m'] = float(native.array(measured['vertices_m'])[:, 2].min())
            if baseline is None:
                geometry_flag = True
            else:
                initial = baseline['targets'].get(key, {}).get('native_geometry')
                geometry_flag = (native._relative_geometry(measured, initial, p['scoring']['dimension_relative_tolerance'])
                                 if initial else None)
        except ERRORS as error:
            row['measurement_errors'].append(f'geometry: {type(error).__name__}: {error}')
        row['fixed_shape_preserved'] = geometry_flag
        geometry_flags.append(native._all((geometry_flag, row['dynamic_collidable'])))
        for field, getter in (('linear_speed_m_s', 'get_linear_velocity'), ('angular_speed_rad_s', 'get_angular_velocity')):
            try:
                row[field] = float(np.linalg.norm(native._vector(getattr(obj, getter)(), 3)))
            except ERRORS as error:
                row['measurement_errors'].append(f'{field}: {type(error).__name__}: {error}')
        try:
            contacts = list(obj.contact_list())
            row['robot_contact'] = any(native._under(body, session.robot.prim_path)
                                       for contact in contacts for body in native._contact_bodies(contact))
        except ERRORS as error:
            row['measurement_errors'].append(f'contact: {type(error).__name__}: {error}')
        for destination in p['destinations']:
            try:
                row['destinations'][destination] = (
                    _bounded_relation(obj, objects[destination], row['vertices_m'], p['destination_contracts'][destination])
                    if p['revision'] == DESTINATION_REVISION
                    else native._relation(obj, objects[destination], row['vertices_m']))
            except ERRORS as error:
                kind = p.get('destination_contracts', {}).get(destination, {}).get('kind', 'on_top')
                fields = ('touching', 'inside', 'vertices_inside') if kind == 'inside' else ('touching', 'on_top', 'footprint_inside')
                row['destinations'][destination] = {**dict.fromkeys(fields),
                    'measurement_errors': [f'{type(error).__name__}: {error}']}
                if p['revision'] == DESTINATION_REVISION:
                    row['destinations'][destination]['kind'] = kind
        sample['targets'][key] = row
    sample['geometry_valid'] = native._all(geometry_flags)
    return sample


def _sample(session, controller, config, objects, baseline=None):
    p = config['parameters']
    physical = deepcopy(config)
    physical['parameters']['destinations'] = list(dict.fromkeys([p['table_key'], *p['destinations'].values()]))
    measure = _measure_household if p['revision'] in HOUSEHOLD_REVISIONS else native._measure
    sample = measure(session, controller, physical, objects, baseline)
    for row in sample['targets'].values():
        row['centroid_m'] = (native.array(row['vertices_m']).mean(axis=0).tolist() if row['vertices_m'] is not None else None)
        measured = row['destinations']
        row['destinations'] = {**{color: measured[key] for color, key in p['destinations'].items()},
                               'table': measured[p['table_key']]}
    anchor = sample['fixtures'].get(p['anchor_key'], {}).get('vertices_m')
    sample['anchor_centroid_m'] = native.array(anchor).mean(axis=0).tolist() if anchor is not None else None
    sample['mirror_material'] = _material_readback(objects.get(p['mirror_key']))
    material_flag = (True if sample['mirror_material']['accepted'] else
                     False if baseline is not None and sample['mirror_material']['available'] else None)
    sample['geometry_valid'] = native._all((sample['geometry_valid'],
                                           material_flag))
    if baseline is not None:
        sample['optical_selection'] = deepcopy(baseline['optical_selection'])
    else:
        try:
            sample['optical_selection'] = _optics(sample, config, _camera(session))
        except ERRORS as error:
            sample['optical_selection'] = {'available': False, 'accepted': False, 'selected_candidate': None,
                'appearance_mode': None, 'frozen_eye_m': None, 'errors': [f'{type(error).__name__}: {error}']}
    return sample


def _qualification(session, config, objects, sample, reducer):
    physical = config
    if config['parameters']['revision'] in HOUSEHOLD_REVISIONS:
        physical = deepcopy(config)
        physical['parameters']['candidate_dimensions_m'] = {}
    result = native._qualification(session, physical, objects, sample)
    if config['parameters']['revision'] in HOUSEHOLD_REVISIONS:
        result.pop('candidate_dimensions_proposal_m')
        result.pop('initial_dimension_tolerance_m')
        result['candidate_assets_contract'] = deepcopy(config['parameters']['candidate_assets'])
        result['geometry_scope'] = 'Measured collision-mesh boundary bounds and optical vertex centroid; no convexity, feasibility, visibility or manipulation certificate.'
    checks = result['checks']
    checks.update(native_reflective_material=sample['mirror_material']['accepted'] is True,
                  measured_unique_initial_optical_selection=sample['optical_selection']['accepted'] is True,
                  pure_initial_preconditions=not reducer['initial_measurement_errors'])
    result.update(accepted=all(checks.values()), optical_selection=deepcopy(sample['optical_selection']),
                  optical_visibility_verified=False, optical_necessity_verified=False,
                  complete_reference_execution_verified=False,
                  initial_reducer_errors=deepcopy(reducer['initial_measurement_errors']),
                  validation_scope='Native support/upright/material and geometric optical identity only; no visibility or physical route certificate.')
    return result


def qualify_operation(session, controller, config, bound):
    try:
        p = _config(session, config)
        controller = getattr(controller, 'controller', controller)
        sample = _sample(session, controller, config, bound)
        measured = MirrorProgress(p).begin(sample)
        return _qualification(session, config, bound, sample, measured)
    except ERRORS as error:
        return {'accepted': False, 'checks': {}, 'measurement_error': f'{type(error).__name__}: {error}',
                'optical_visibility_verified': False, 'optical_necessity_verified': False,
                'complete_reference_execution_verified': False}


class MirrorEvaluator:
    def __init__(self, session, controller, config, output):
        self.config = deepcopy(config)
        self.p = _config(session, self.config)
        self.session, self.controller = session, getattr(controller, 'controller', controller)
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.output, self.active = Path(output), False
        self.scorer = MirrorProgress(self.p)
        self.initial = self.last_sample = self.initial_qualification = None

    def begin_episode(self, observation):
        if self.active:
            raise RuntimeError('The mirror episode has already begun.')
        self.initial = _sample(self.session, self.controller, self.config, self.objects)
        measured = self.scorer.begin(self.initial)
        self.last_sample = deepcopy(self.initial)
        self.initial_qualification = _qualification(self.session, self.config, self.objects, self.initial, measured)
        self.active = True
        return self.record(observation)

    def sample_step(self):
        if self.active:
            self.last_sample = _sample(self.session, self.controller, self.config, self.objects, self.initial)
            self.scorer.update(self.last_sample)

    def report_answer(self, text, observation=None):
        return self.scorer.report_answer(text)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        measured = self.scorer.result()
        validity = measured['evaluation_validity']
        if validity != 'invalid' and self.initial_qualification['accepted'] is not True:
            validity = 'unavailable'
        result = {**measured, 'schema': self.p['revision'], 'snapshot_id': observation.snapshot_id,
                  'action_step': observation.action_step, 'action': action,
                  'configuration': deepcopy(self.config), 'native_measurements': deepcopy(self.last_sample),
                  'initial_qualification': deepcopy(self.initial_qualification),
                  'physical_scorer_schema': measured['schema'], 'physical_scorer_version': measured['scoring_version'],
                  'evaluation_validity': validity, 'diagnostic_reducer_physical_success': measured['physical_success'],
                  'optical_visibility_verified': False, 'optical_necessity_verified': False,
                  'complete_reference_execution_verified': False}
        result.update(progress(result))
        if validity in ('invalid', 'unavailable'):
            result.update(evaluation_validity=validity, completion_score=None, score_percent=None,
                          task_success=False, physical_success=False)
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    return MirrorEvaluator(session, controller, config, output)

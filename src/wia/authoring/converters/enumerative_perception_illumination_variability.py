"""Convert illumination counting scenes into measured physical inventory tasks.

Run on CPU from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Scene initialization installs the original-light environmental mechanism; native
qualification and episode evaluators only read state. Robot actions remain in
the existing guarded controller. No GPU worker is started by this module.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from collections import Counter
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import re

import numpy as np

from wia.authoring.conversion import ConverterBase, map_records, original_scene, physical_task
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import sha256, write_json

MODULE = 'core.esi_converters.enumerative_perception_illumination_variability'
OPERATIONS = ('two_phase_inventory_return', 'two_phase_single_return')
PHASES = ('normal', 'changed')
CHANGED_SCALE = .04
PHASE_DURATION_S = 30.
MINIMUM_LIFT_M = .05
POSITION_TOLERANCE_M = .03
ANGLE_TOLERANCE_DEG = 15.
STABILITY_STEPS = 15
QUESTION = re.compile(r'Considering both images, how many objects of category "([^"\n]+)" '
                      r'are present in this scene\?')


def _legal_background_grasp(session, controller, name):
    if name is None:
        return False
    obj = session.env.scene.object_registry('name', name)
    arm = getattr(controller, 'working_arm', 'left')
    eef = getattr(session.robot, 'eef_links', {}).get(arm)
    return bool(obj is not None and eef is not None and
                legal_background_contact(session, eef.prim_path, obj.prim_path))


def _completion_result(value, items, *, available=True):
    """Expose required physical obligations without averaging safety predicates."""
    completion_items = [{'id': key, 'completed': bool(done),
        'evaluation_validity': 'valid' if available else 'unavailable',
        'kind': kind, 'evidence': evidence} for key, done, kind, evidence in items]
    completed = sum(item['completed'] for item in completion_items)
    total = len(completion_items)
    score = completed / total if total and available else None
    value.update(scoring_version='task_completion_v2',
        evaluation_validity='complete' if available else 'unavailable',
        completion_items=completion_items, completed_goals=completed, total_goals=total,
        completion_score=score, score_percent=100. * score if score is not None else None)
    return value


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Source and native measurements must be finite.')
    return value


def _vector(value, size):
    value = _array(value)
    if value.shape != (size,):
        raise ValueError('Source/native vector shape is unresolved.')
    return value


def _box(value):
    low, high = (_vector(part, 3) for part in value)
    if np.any(high <= low):
        raise ValueError('Source/native object extents must be positive.')
    return low, high


def _asset_root(context):
    data = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    return Path(context.config.get('original_asset_root', str(data / 'behavior-1k-assets')))


def _layout(source):
    raw, payload = source.raw, source.payload
    question = payload['question_data']
    render = question['render']
    match = QUESTION.fullmatch(raw['question'].strip())
    if (not match or question.get('count_target') != match[1] or
            question.get('case') != 'light_change'):
        raise ValueError('Original light-change case or public target category is unresolved.')
    cameras = map_records(render['camera_poses'])
    camera = cameras[Path(render['image']).name]
    _vector(camera['position'], 3)
    if np.linalg.norm(_vector(camera['quaternion_xyzw'], 4)) < 1e-6:
        raise ValueError('Original reference orientation is invalid.')
    if {row.get('lighting_phase') for row in cameras.values()} != set(PHASES):
        raise ValueError('The source must preserve normal and changed reference views.')
    if any(np.linalg.norm(_vector(row['position'], 3) - camera['position']) > 1e-5 or
           _angle(row['quaternion_xyzw'], camera['quaternion_xyzw']) > .01
           for row in cameras.values()):
        raise ValueError('Normal/changed source views do not share a reference pose.')
    targets = sorted(render['resolved_objects']['targets'], key=lambda item: item['name'])
    if not targets or any(render['resolved_objects'].get(role) for role in ('confusers', 'containers')):
        raise ValueError('This rule requires the original light-change target-only layout.')
    bindings = {'floor': {'name': payload['floor_name'], 'category': 'floors'}}
    expected, names = {}, set()
    for index, target in enumerate(targets):
        if target['category'] != match[1] or not target.get('model') or target['name'] in names:
            raise ValueError('Original target identities/categories are ambiguous.')
        _vector(target['position'], 3)
        _box(target['bbox'])
        if np.linalg.norm(_vector(target['quaternion_xyzw'], 4)) < 1e-6:
            raise ValueError('Original target orientation is invalid.')
        key = f'target_{index:02d}'
        bindings[key] = {field: target[field] for field in ('name', 'category', 'model')}
        expected[key] = {field: target[field] for field in ('position', 'quaternion_xyzw', 'bbox')}
        names.add(target['name'])
    lights = {}
    for entry in question['case_metadata']['source_entries']:
        for lamp in entry['metadata']['lighting']['target_lights']:
            _box(lamp['bbox'])
            if not lamp.get('name') or not lamp.get('category'):
                raise ValueError('Original lamp identity is incomplete.')
            if lamp['name'] in lights and lights[lamp['name']] != lamp:
                raise ValueError('Source entries disagree about an original lamp.')
            lights[lamp['name']] = lamp
    if not lights:
        raise ValueError('No original illumination mechanism is declared.')
    lights = [lights[name] for name in sorted(lights)]
    for index, lamp in enumerate(lights):
        if lamp['name'] in names:
            raise ValueError('Lamp and manipulation identities must be distinct.')
        bindings[f'lamp_{index:02d}'] = {'name': lamp['name'], 'category': lamp['category']}
    bounds = _vector(render['room_bbox']['expanded_bbox_world_xy'], 4)
    if np.any(bounds[2:] <= bounds[:2]):
        raise ValueError('Source room bounds are invalid.')
    return {'raw': raw, 'payload': payload, 'render': render, 'camera': camera,
            'category': match[1], 'targets': targets, 'lights': lights,
            'bindings': bindings, 'source_objects': expected, 'bounds': bounds}


@lru_cache(maxsize=4)
def _original_scene_data(asset_root, scene_model):
    path = Path(asset_root) / 'scenes' / scene_model / 'json' / f'{scene_model}_best.json'
    return json.loads(path.read_text()), str(path), sha256(path)


def _lamp_restore(layout, context):
    """Use original native scene loading, retaining saved scales/poses/physics."""
    scene, path, checksum = _original_scene_data(str(_asset_root(context)), layout['raw']['scene'])
    originals = scene['objects_info']['init_info']
    categories, rooms, models = set(), set(), {}
    elevated = False
    floor_z = float(layout['camera']['position'][2] - 1.25)
    for lamp in layout['lights']:
        args = originals[lamp['name']]['args']
        if args['category'] != lamp['category']:
            raise ValueError('Declared lamp category disagrees with the installed original scene.')
        models[lamp['name']] = args['model']
        layout['bindings'][next(key for key, row in layout['bindings'].items()
                               if row['name'] == lamp['name'])]['model'] = args['model']
        if not (_asset_root(context) / 'objects' / args['category'] / args['model']).is_dir():
            raise ValueError('An original lamp asset is not installed.')
        categories.add(args['category'])
        rooms.update(args.get('in_rooms') or [])
        elevated = elevated or (lamp['category'] == 'table_lamp' and lamp['bbox'][0][2] > floor_z + .15)
    # Elevated lamps need their original support; the cleaned JSON omits exact
    # support identity. Restore only installed support categories in their rooms,
    # never invent a table or fix/teleport a lamp. Native drift remains a gate.
    supports = set()
    if elevated:
        for original in originals.values():
            args = original['args']
            category = args.get('category', '')
            if (rooms.intersection(args.get('in_rooms') or []) and
                    (any(word in category for word in ('table', 'desk', 'cabinet', 'shelf',
                                                       'bookcase', 'counter', 'dresser')) or
                     category in ('stand', 'nightstand', 'tv_stand')) and
                    category not in categories):
                supports.add(category)
    operation_supports = []
    for name, original in sorted(originals.items()):
        args = original['args']
        if (layout['raw']['room'] in (args.get('in_rooms') or []) and
                any(word in args.get('category', '') for word in
                    ('table', 'counter', 'desk', 'bench', 'shelf', 'cabinet')) and
                not args.get('visual_only', False)):
            if not (_asset_root(context) / 'objects' / args['category'] / args['model']).is_dir():
                raise ValueError('An original same-room support asset is not installed.')
            operation_supports.append({key: args[key] for key in ('name', 'category', 'model')})
    operation_categories = {row['category'] for row in operation_supports}
    return {'scene_model': layout['raw']['scene'], 'room': layout['raw']['room'],
            'source_identity': str(layout['raw'].get('id', '')), 'lamps': layout['lights'],
            'lamp_models': models, 'restore_categories': sorted(categories | supports | operation_categories),
            'support_categories': sorted(supports),
            'operation_support_categories': sorted(operation_categories),
            'operation_support_objects': operation_supports, 'changed_scale': CHANGED_SCALE,
            'phase_duration_s': PHASE_DURATION_S, 'original_scene_file': path,
            'original_scene_sha256': checksum,
            'provenance': 'question_data.case_metadata.source_entries[*].metadata.lighting.target_lights; '
                'installed original scene objects_info.init_info (lamp models and same-room original supports); '
                'batch_counting_merge._sample_large_light_change and _render_light_change_and_save. '
                'Cleaned JSON omits exact sampled intensities and timing.'}


@lru_cache(maxsize=3)
def _maps(asset_root, scene_model):
    import cv2
    directory = Path(asset_root) / 'scenes' / scene_model / 'layout'
    paths = [directory / name for name in ('floor_trav_0.png', 'floor_insseg_0.png')]
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(image is None for image in images) or images[0].shape != images[1].shape:
        raise ValueError('Original traversability and room maps are unavailable.')
    return images[0], images[1], {str(path): sha256(path) for path in paths}


def _screen_point(maps, xy, room_id):
    traversability, rooms, _ = maps
    radius = 40  # Original maps are 1 cm/pixel; conservative R1-body disk.
    row, column = [int(v / .01 + traversability.shape[0] / 2.) for v in (xy[1], xy[0])]
    if min(row, column) < radius or row + radius >= rooms.shape[0] or column + radius >= rooms.shape[1]:
        return False
    delta = np.arange(-radius, radius + 1)
    disk = delta[:, None] ** 2 + delta[None, :] ** 2 <= radius ** 2
    area = (slice(row - radius, row + radius + 1), slice(column - radius, column + radius + 1))
    return bool(np.all(traversability[area][disk] > 0) and np.all(rooms[area][disk] == room_id))


def _robot_start(layout, context):
    maps = _maps(str(_asset_root(context)), layout['raw']['scene'])
    bounds, camera = layout['bounds'], layout['camera']
    low, high = bounds[:2] + .65, bounds[2:] - .65
    if np.any(high < low):
        return None
    points = [('reference_projection', np.clip(_vector(camera['position'], 3)[:2], low, high))]
    for view in layout['render'].get('multi_image_input', {}).get('views', []):
        if view.get('is_traversable') is True:
            points.append((view['view_id'], np.clip(_vector(view['camera_pose']['position'], 3)[:2], low, high)))
    # A deterministic nearby map search is robot-only L2, not a target layout
    # change or a claim of global unreachability when no proposal is found.
    points.extend(('original_room_grid', np.array([x, y]))
                  for x in np.arange(low[0], high[0] + 1e-6, .2)
                  for y in np.arange(low[1], high[1] + 1e-6, .2))
    boxes = [_box(item['bbox']) for item in layout['targets'] + layout['lights']]
    room_id = layout['render']['room_bbox']['room_id']
    valid = [(label, xy) for label, xy in points if _screen_point(maps, xy, room_id) and all(
        np.linalg.norm(xy - np.clip(xy, lo[:2], hi[:2])) >= .65 for lo, hi in boxes)]
    if not valid:
        return None
    label, xy = min(valid, key=lambda row: (math.dist(row[1], camera['position'][:2]),
                                           float(row[1][0]), float(row[1][1]), row[0]))
    look = _vector(camera['look_target'], 3)
    yaw = math.degrees(math.atan2(look[1] - xy[1], look[0] - xy[0]))
    return {'position': [float(xy[0]), float(xy[1]), float(camera['position'][2] - 1.25)],
            'yaw_deg': yaw, 'source_observer_or_rule': label, 'body_map_radius_m': .4,
            'original_map_sha256': maps[2], 'room_id': room_id,
            'floor_height_provenance': 'batch_counting_merge._sample_primary_view_pose: floor_top + 1.25 m',
            'native_body_clearance_qualified': False}


def configure_scene(config, parameters):
    """Restore original lamp/support categories exclusively before construction."""
    scene = config['scene']
    if scene['scene_model'] != parameters['scene_model']:
        raise ValueError('Original light restoration belongs to a different native scene.')
    previous = scene.get('load_object_categories')
    restored = parameters['restore_categories']
    if previous is not None:
        scene['load_object_categories'] = sorted(set(previous) | set(restored))
    # No pose, scale, fixed-base, source robot, native light value or safety
    # setting is overwritten here; scene loading applies the original assets.
    return {'source_provenance': parameters['provenance'], 'original_categories': previous,
            'restored_original_categories': restored,
            'original_scene_sha256': parameters['original_scene_sha256'],
            'support_identity_unresolved': bool(parameters['support_categories']),
            'operation_support_objects': parameters.get('operation_support_objects', []),
            'category_filter_is_global_in_full_scene': True}


def _stage_lights(root):
    for child in root.GetChildren():
        kind = str(child.GetPrimTypeInfo().GetTypeName())
        if 'Light' in kind:
            attr = child.GetAttribute('inputs:intensity')
            if attr and attr.IsValid() and attr.Get() is not None:
                value = float(attr.Get())
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Native light intensity is invalid.')
                yield {'prim': child, 'attribute': attr, 'path': str(child.GetPath()),
                       'type_name': kind, 'baseline': value,
                       'is_dome': kind == 'DomeLight' or str(child.GetPath()).lower().endswith('/skybox/base_link/light')}
        yield from _stage_lights(child)


def _light_world_transform(light):
    """Read the actual light prim transform without editing light attributes."""
    from pxr import Usd, UsdGeom
    transform = UsdGeom.Xformable(light['prim'])
    if not transform:
        raise ValueError('An original native light has no measurable transform.')
    return np.asarray(transform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()), dtype=float)


class _OriginalLightCycle:
    """Initialization-created environmental mechanism, separate from scoring.

    The callback changes only original light intensities. It never changes a
    robot/object pose, grasp state or evaluator flag. Measured attributes, rather
    than the callback's requested phase, determine credited episode phases.
    """
    def __init__(self, session, parameters, lights):
        self.session, self.parameters, self.lights = session, parameters, lights
        self.elapsed_s, self.written_phase, self.error = 0., 'normal', None
        self.initial_light_world_transforms = {light['path']: _light_world_transform(light).copy() for light in lights}

    def advance(self, step_size):
        if self.error is not None:
            return
        try:
            self.elapsed_s += float(step_size)
            phase = PHASES[int(self.elapsed_s / self.parameters['phase_duration_s']) % 2]
            if phase != self.written_phase:
                scale = 1. if phase == 'normal' else self.parameters['changed_scale']
                for light in self.lights:
                    light['attribute'].Set(light['baseline'] * scale)
                self.written_phase = phase
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'

    def measure(self):
        rows = [{'path': light['path'], 'type_name': light['type_name'],
                 'baseline': light['baseline'], 'intensity': float(light['attribute'].Get())}
                for light in self.lights]
        phase = None
        positive = [row for row in rows if row['baseline'] > 0]
        if self.error is None and positive and all(math.isfinite(row['intensity']) for row in rows):
            for label, scale in (('normal', 1.), ('changed', self.parameters['changed_scale'])):
                if all(abs(row['intensity'] - row['baseline'] * scale) <=
                       max(1e-6, row['baseline'] * 1e-5) for row in rows):
                    phase = label
        transforms = {light['path']: bool(np.max(np.abs(_light_world_transform(light) -
            self.initial_light_world_transforms[light['path']])) <= 1e-6) for light in self.lights}
        return {'measured_phase': phase, 'native_light_intensities': rows,
                'original_light_transforms_preserved': bool(transforms and all(transforms.values())),
                'native_light_transform_checks': transforms,
                'environmental_driver_error': self.error}


def initialize_scene(session, parameters):
    """Install the source light cycle once; never initialize through evaluators."""
    import omnigibson as og
    if (session.scene_name, session.room_name) != (parameters['scene_model'], parameters['room']):
        raise ValueError('Original lighting restoration belongs to a different source.')
    if hasattr(session, '_illumination_variability_cycle'):
        raise ValueError('The source environmental mechanism initializes only once.')
    lamps = []
    for lamp in parameters['lamps']:
        native = session.env.scene.object_registry('name', lamp['name'])
        if native is None or native.category != lamp['category'] or native.model != parameters['lamp_models'][lamp['name']]:
            raise ValueError('An original lamp could not be restored from the native scene.')
        lo, hi = (_array(value) for value in native.aabb)
        lamps.append({'name': native.name, 'category': native.category, 'model': native.model,
                      'aabb_min': lo.tolist(), 'aabb_max': hi.tolist(),
                      'visual_only': bool(native.visual_only), 'fixed_base': bool(native.fixed_base)})
    lights = list(_stage_lights(og.sim.world_prim))
    internal = [light for light in lights if not light['is_dome']]
    selected = internal or lights
    if not any(light['baseline'] > 0 for light in selected):
        raise ValueError('not_convertible_under_L1_L2: no_positive_native_adjustable_stage_light')
    cycle = _OriginalLightCycle(session, parameters, selected)
    callback = MODULE + ':' + str(session.question_path)
    og.sim.add_physics_callback(callback, cycle.advance)
    session._illumination_variability_cycle = cycle
    return {'source_provenance': parameters['provenance'], 'restored_native_lamps': lamps,
            'strategy': 'internal_stage_lights' if internal else 'all_stage_lights',
            'normal_native_light_measurements': cycle.measure(), 'callback_name': callback,
            'environmental_phase_duration_s': parameters['phase_duration_s'],
            'changed_scale': parameters['changed_scale'],
            'exact_historical_lighting_values_reconstructed': False,
            'warning_codes': ['assumption_review_required', 'source_light_values_not_retained']}


def _angle(first, second):
    a, b = _vector(first, 4), _vector(second, 4)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-6:
        raise ValueError('Measured orientation is invalid.')
    dot = min(1., abs(float(np.dot(a, b) / denominator)))
    return math.degrees(2. * math.acos(dot))


def _near(row, origin, tolerance=POSITION_TOLERANCE_M):
    return (math.dist(row['position'], origin['position']) <= tolerance and
            _angle(row['quaternion_xyzw'], origin['quaternion_xyzw']) <= ANGLE_TOLERANCE_DEG)


def _stationary(row):
    return row['linear_speed_m_s'] <= .02 and row['angular_speed_rad_s'] <= .1


def _snapshot(session, obj, floor):
    from omnigibson.object_states import Touching
    from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    lo, hi = (_array(value) for value in obj.aabb)
    return {'name': obj.name, 'category': obj.category, 'model': obj.model,
            'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': lo.tolist(), 'aabb_max': hi.tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'touching_original_floor': bool(obj.states[Touching].get_value(floor)),
            'touching_task_support': touching_task_support(session, obj, floor),
            'on_top_task_support': on_top_task_support(session, obj, floor),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}


def _robot_state(session):
    robot, scene = session.robot, session.env.scene
    position, _ = robot.get_position_orientation()
    lo, hi = (_array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    rooms = [scene.seg_map.get_room_instance_by_point(np.array([x, y]))
             for x in (lo[0], (lo[0] + hi[0]) / 2., hi[0])
             for y in (lo[1], (lo[1] + hi[1]) / 2., hi[1])]
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj.prim_path for obj in objects if obj.category in ('floor', 'floors')]
    floor_contacts, obstacle_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        external = [str(body) for body in (contact.body0, contact.body1)
                    if not (str(body) == robot.prim_path or str(body).startswith(robot.prim_path + '/'))]
        if not external:
            continue
        on_floor = all('groundPlane' in body or any(body == path or body.startswith(path + '/')
                      for path in floors) for body in external)
        # Target contacts are legitimate manipulation, but not robot spawn clearance.
        (floor_contacts if on_floor else obstacle_contacts).append(external)
    return {'position': _array(position).tolist(), 'footprint_rooms': rooms,
            'in_original_room': bool(rooms and all(room == session.room_name for room in rooms)),
            'floor_contacts': floor_contacts, 'nonfloor_contacts': obstacle_contacts}


def _configuration(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') not in OPERATIONS or
            (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('Use the owned illumination operation in its original scene/room.')
    parameters = config['parameters']
    keys = [*parameters['target_keys'], *parameters['lamp_keys'], 'floor']
    if (not parameters['target_keys'] or len(set(keys)) != len(keys) or
            set(config['binding_keys']) != set(keys) or set(config['bindings']) != set(keys) or
            len(set(config['bindings'].values())) != len(keys)):
        raise ValueError('Original targets, lamps and floor need complete distinct bindings.')
    return parameters


def _lamp_evidence(objects, parameters, session=None):
    """Compare lamps to accepted adapted references, preserving raw legacy fallback."""
    from wia.tasks.interaction_setup import reference_position
    evidence = {}
    for key, source in zip(parameters['lamp_keys'], parameters['lighting']['lamps']):
        obj = objects[key]
        lo, hi = (_array(value) for value in obj.aabb)
        expected_lo, expected_hi = _box(source['bbox'])
        if session is not None:
            expected_lo = reference_position(session, obj.name, expected_lo)
            expected_hi = reference_position(session, obj.name, expected_hi)
        evidence[key] = {'name': obj.name, 'category': obj.category, 'model': obj.model,
                         'aabb_min': lo.tolist(), 'aabb_max': hi.tolist(),
                         'source_aabb_preserved': bool(np.max(np.abs(lo - expected_lo)) <= .05 and
                                                       np.max(np.abs(hi - expected_hi)) <= .05)}
    return evidence


def _layout_evidence(session, rows, parameters):
    """Validate the measured initialization translation without changing state."""
    from wia.tasks.interaction_setup import reference_position
    setup = getattr(session, 'category_interaction_setup', {}) or {}
    names = {row['name'] for row in rows.values()}
    relocated = setup.get('relocated') is True
    translations = {key: reference_position(session, row['name'], parameters['source_objects'][key]['position'])
                    - _vector(parameters['source_objects'][key]['position'], 3)
                    for key, row in rows.items()}
    anchor = translations[next(iter(rows))]
    origins = {key: {**parameters['source_objects'][key],
                     'position': (_vector(parameters['source_objects'][key]['position'], 3) + translations[key]).tolist()}
               for key in rows}
    lamp_names = {row['name'] for row in parameters['lighting']['lamps']}
    cycle = getattr(session, '_illumination_variability_cycle', None)
    light_paths = {row['path'] for row in getattr(cycle, 'lights', [])}
    transformed_paths = set(setup.get('transformed_light_prim_paths', []))
    descendant_paths = set(setup.get('skipped_descendant_light_paths', []))

    def same_delta(value):
        try:
            return bool(np.linalg.norm(_vector(value, 3) - anchor) <= 1e-6)
        except (ValueError, TypeError):
            return False

    if parameters.get('fixed_lighting_anchors'):
        from wia.tasks.interaction_setup import support_contact_paths
        checks = {'measured_interaction_accepted': setup.get('accepted') is True,
            'complete_portable_target_group': set(setup.get('object_names', [])) == names,
            'fixed_lamps_excluded_from_placement': not lamp_names & set(setup.get('object_names', [])),
            'uniform_target_elevation': all(np.linalg.norm(delta - anchor) <= 1e-6 for delta in translations.values()),
            'vertical_only_translation': bool(np.max(np.abs(anchor[:2])) <= 1e-6),
            'adapted_source_target_poses': all(_near(row, origins[key]) for key, row in rows.items()),
            'original_target_dimensions': all(np.max(np.abs(
                (_vector(row['aabb_max'], 3) - _vector(row['aabb_min'], 3)) -
                (_vector(parameters['source_objects'][key]['bbox'][1], 3) -
                 _vector(parameters['source_objects'][key]['bbox'][0], 3)))) <= .01 for key, row in rows.items()),
            'measured_target_supports': not relocated or all(support_contact_paths(session, name) for name in names),
            'original_native_light_transforms': bool(cycle and cycle.measure().get('original_light_transforms_preserved')),
            'source_observer_untranslated': bool(np.linalg.norm(_vector(
                setup.get('source_reference_camera_translation_m', [0., 0., 0.]), 3)) <= 1e-6)}
        return {'accepted': all(checks.values()), 'checks': checks, 'relocated': relocated,
                'target_translation_m': anchor.tolist(), 'fixed_lighting_anchors': True}

    checks = {'measured_interaction_accepted': setup.get('accepted') is True,
              'complete_target_and_light_group': (set(setup.get('object_names', [])) == names | lamp_names
                  if relocated else names <= set(setup.get('object_names', []))),
              'uniform_target_translation': all(np.linalg.norm(delta - anchor) <= 1e-6
                                                for delta in translations.values()),
              'original_target_poses_after_common_translation': all(_near(row, origins[key])
                                                                   for key, row in rows.items()),
              'original_target_dimensions': all(np.max(np.abs(
                  (_vector(row['aabb_max'], 3) - _vector(row['aabb_min'], 3)) -
                  (_vector(parameters['source_objects'][key]['bbox'][1], 3) -
                   _vector(parameters['source_objects'][key]['bbox'][0], 3)))) <= .01
                  for key, row in rows.items()),
              'measured_support_identity': not relocated or bool(setup.get('support_prim_path')),
              'vertical_only_translation': not relocated or bool(np.max(np.abs(anchor[:2])) <= 1e-6),
              'bound_light_anchors_transformed': not relocated or lamp_names <= set(
                  setup.get('transformed_bound_anchor_names', [])),
              'all_selected_light_prims_transformed': not relocated or bool(light_paths and
                  light_paths <= transformed_paths | descendant_paths),
              'anchor_common_translation': not relocated or same_delta(setup.get('anchor_translation_m')),
              'reference_observer_common_translation': not relocated or same_delta(
                  setup.get('source_reference_camera_translation_m'))}
    return {'accepted': bool(all(checks.values())), 'checks': checks,
            'relocated': relocated, 'support_name': setup.get('support_name'),
            'target_translation_m': anchor.tolist()}


def qualify_operation(session, executor, config, bound):
    """Read native initialization only; never step, act, move or score from QA."""
    checks = {}
    try:
        parameters = _configuration(session, config)
        rows = {key: _snapshot(session, bound[key], bound['floor']) for key in parameters['target_keys']}
        layout = _layout_evidence(session, rows, parameters)
        lamps = _lamp_evidence(bound, parameters, session)
        robot = _robot_state(session)
        light = session._illumination_variability_cycle.measure()
        checks = {'original_floor_collidable': not bound['floor'].visual_only,
                  'source_targets_movable_collidable': all(not row['visual_only'] and not row['fixed_base']
                                                          for row in rows.values()),
                  'source_category_preserved': all(row['category'] == parameters['target_category']
                                                   for row in rows.values()),
                  'source_target_group_preserved': layout['accepted'],
                  'targets_supported_stationary': all(row['touching_task_support'] and row['on_top_task_support'] and _stationary(row)
                                                      for row in rows.values()),
                  'original_lamp_layout_preserved': all(row['source_aabb_preserved'] for row in lamps.values()),
                  'measurable_native_illumination': light['measured_phase'] in PHASES,
                  'gripper_initially_empty': executor.debug_gripper_state().get('object_name') is None,
                  'robot_in_original_room': robot['in_original_room'],
                  'robot_supported_clear_at_start': bool(robot['floor_contacts']) and not robot['nonfloor_contacts']}
        return {'accepted': bool(all(checks.values())), 'checks': checks,
                'reason': None if all(checks.values()) else 'native_illumination_inventory_initialization_failed',
                'native_targets': rows, 'native_lamps': lamps, 'native_robot': robot,
                'native_lighting': light, 'native_layout': layout, 'physical_execution_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_illumination_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}', 'physical_execution_qualified': False}


class IlluminationInventoryEvaluator:
    """Require separate grasp/lift/release events across measured light phases."""
    def __init__(self, session, controller, config, output):
        self.parameters = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.objects = {key: session.env.scene.object_registry('name', name)
                        for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('An original native illumination binding is missing.')
        self.targets = set(self.parameters['target_keys'])
        self.episode, self.initial = 0, None

    def measure(self):
        rows = {key: _snapshot(self.session, self.objects[key], self.objects['floor']) for key in sorted(self.targets)}
        return rows, self.controller.debug_gripper_state(), self.session._illumination_variability_cycle.measure()

    def begin_episode(self, observation):
        self.episode += 1
        self.initial, gripper, _ = self.measure()
        self.initially_empty = gripper.get('object_name') is None and gripper.get('constraint_active') is False
        self.initial_layout = _layout_evidence(self.session, self.initial, self.parameters)
        self.initial_valid = self.initial_layout['accepted'] and all(
            not row['visual_only'] and not row['fixed_base'] and
            row['touching_task_support'] and row['on_top_task_support'] and _stationary(row) for row in self.initial.values())
        self.previous_held, self.event, self.pending_release = None, None, None
        self.can_acquire = self.initially_empty
        self.completed, self.observed_phases = set(), set()
        self.phase_views = {}
        self.wrong_grasp, self.history_valid, self.stable_steps = False, True, 0
        self.phase, self.phase_samples = None, 0
        self.record(observation, action=None)

    def _update(self, count_step=False):
        if self.initial is None:
            return None
        rows, gripper, lighting = self.measure()
        measured_phase = lighting['measured_phase']
        if count_step:
            self.phase_samples = self.phase_samples + 1 if measured_phase == self.phase else 1
            self.phase = measured_phase
        elif measured_phase != self.phase:
            self.phase_samples = 0
        # Two controller physics samples separate intensity changes from credit;
        # images themselves are not scored and do not expose the native phase.
        phase = measured_phase if self.phase_samples >= 2 else None
        if phase in PHASES:
            self.observed_phases.add(phase)
        name_to_key = {rows[key]['name']: key for key in self.targets}
        held = name_to_key.get(gripper.get('object_name')) if (gripper.get('holding') is True and
            gripper.get('constraint_active') is True) else None
        if (gripper.get('object_name') is not None and gripper['object_name'] not in name_to_key
                and not _legal_background_grasp(self.session, self.controller, gripper['object_name'])):
            self.wrong_grasp = True
        lifts = {key: rows[key]['aabb_min'][2] - self.initial[key]['aabb_min'][2] for key in self.targets}
        empty_open = (gripper.get('object_name') is None and gripper.get('constraint_active') is False and
                      gripper.get('gripper_open') is True)
        # A transition invalidates the unfinished event permanently. Waiting for
        # the same label in a later cycle cannot rescue a cross-phase return.
        if self.event is not None and measured_phase != self.event['phase']:
            self.event['phase_consistent'] = False
        if self.pending_release is not None and measured_phase != self.pending_release['phase']:
            self.pending_release = None
        # Credit a fresh acquisition only. A single unchanged grasp crossing a
        # light transition cannot masquerade as a second manipulation event.
        if held != self.previous_held:
            if self.event is not None and self.event['lifted'] and self.event['phase_consistent']:
                self.pending_release = dict(self.event)
            self.event = ({'key': held, 'phase': phase, 'lifted': False, 'phase_consistent': True}
                          if held and self.can_acquire and phase in PHASES else None)
            if held:
                self.can_acquire = False
        if (self.event is not None and self.event['phase_consistent'] and
                phase == self.event['phase'] and phase in PHASES and lifts[held] >= MINIMUM_LIFT_M):
            self.event['lifted'] = True
        if empty_open:
            self.can_acquire = True
        if self.pending_release is not None:
            event = self.pending_release
            key = event['key']
            if held is not None:
                # Starting a new grasp before a measured supported release does
                # not validate the previous target's return.
                self.pending_release = None
            elif (empty_open and phase == event['phase'] and _near(rows[key], self.initial[key]) and
                    rows[key]['touching_task_support'] and rows[key]['on_top_task_support'] and _stationary(rows[key])):
                self.completed.add((key, event['phase']))
                self.pending_release = None
        self.previous_held = held
        lamps = _lamp_evidence(self.objects, self.parameters, self.session)
        objects_valid = all(not row['visual_only'] and not row['fixed_base'] for row in rows.values())
        self.history_valid = self.history_valid and objects_valid and all(row['source_aabb_preserved'] for row in lamps.values()) and (
            not self.parameters.get('fixed_lighting_anchors') or lighting.get('original_light_transforms_preserved') is True)
        all_returned = all(_near(rows[key], self.initial[key]) and rows[key]['touching_task_support'] and rows[key]['on_top_task_support'] and
                           _stationary(rows[key]) for key in self.targets)
        if self.config['operation'] == 'two_phase_inventory_return':
            operations_complete = all((key, label) in self.completed for key in self.targets for label in PHASES)
        else:
            operations_complete = any(all((key, label) in self.completed for label in PHASES) for key in self.targets)
        terminal = (self.initially_empty and self.initial_valid and self.history_valid and not self.wrong_grasp and
                    phase in PHASES and self.observed_phases == set(PHASES) and operations_complete and
                    all_returned and empty_open)
        if count_step:
            self.stable_steps = self.stable_steps + 1 if terminal else 0
        elif not terminal:
            self.stable_steps = 0
        value = {'native_targets': rows, 'native_lamps': lamps, 'native_lighting': lighting,
                'gripper': gripper, 'held_binding': held, 'native_lift_m': lifts,
                'completed_return_events': [{'binding': key, 'phase': label} for key, label in sorted(self.completed)],
                'observed_measured_phases': sorted(self.observed_phases),
                'criteria': {'initially_empty': self.initially_empty, 'source_initialization_valid': self.initial_valid,
                             'native_object_history_valid': self.history_valid, 'no_wrong_grasp': not self.wrong_grasp,
                             'required_separate_two_phase_operations': operations_complete,
                             'all_targets_returned_supported_stationary': all_returned,
                             'empty_open_gripper': empty_open, 'measured_terminal_phase': phase in PHASES},
                'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS,
                'task_success': bool(terminal and self.stable_steps >= STABILITY_STEPS)}
        if self.config['operation'] == 'two_phase_inventory_return':
            keys = sorted(self.targets)
        else:
            # Both phase obligations must belong to the same selected object.
            keys = [max(sorted(self.targets), key=lambda key: sum((key, label) in self.completed for label in PHASES))]
        items = [(f'{key}:{label}:grasp_lift_return', (key, label) in self.completed,
            'historical_checkpoint', {'binding': key, 'measured_phase': label,
            'separate_supported_return_recorded': (key, label) in self.completed})
            for key in keys for label in PHASES]
        return _completion_result(value, items)

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        value = self._update()
        if value is None:
            return None
        if self.parameters.get('requires_native_visual_discovery'):
            from .partial_occlusion import native_visual_evidence
            from wia.tasks.completion import progress
            visual = native_visual_evidence(self.session, observation,
                {key: self.objects[key] for key in self.targets})
            phase = value['native_lighting']['measured_phase']
            if phase in PHASES:
                for key, row in visual['objects'].items():
                    if row['visible']:
                        self.phase_views.setdefault((key, phase), {'snapshot_id': observation.snapshot_id,
                            'action_step': observation.action_step, 'views': row['views']})
            keys = sorted(self.targets)
            if self.config['operation'] != 'two_phase_inventory_return':
                keys = [max(keys, key=lambda key: sum((key, label) in self.completed for label in PHASES))]
            value['completion_items'] += [{'id': f'{key}:{label}:visual_discovery', 'kind': 'checkpoint',
                'completed': (key, label) in self.phase_views,
                'evaluation_validity': 'valid' if (key, label) in self.phase_views or visual['complete_coverage'] else 'unavailable'}
                for key in keys for label in PHASES]
            value['task_success'] = bool(value['task_success'] and all((key, label) in self.phase_views
                                                                       for key in keys for label in PHASES))
            value['phase_visual_evidence'] = [{'target': key, 'phase': label, **row}
                for (key, label), row in self.phase_views.items()]
            value.update(progress(value))
        value.update(schema='private_illumination_inventory_v1', episode=self.episode,
                     snapshot_id=observation.snapshot_id, action_step=observation.action_step,
                     action=action, configuration=self.config, initial_native_targets=self.initial,
                     initial_layout_evidence=self.initial_layout)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return IlluminationInventoryEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Enumerative Perception', 'Illumination Variability')
    # Keep historical candidates separate from the complete-anchor repair.
    version = '2.0.0_task_edit'

    def inspect(self, source):
        if (source.big_task, source.small_task) != self.category or (
                source.raw.get('big_task'), source.raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:category_mismatch',
                    'warning_codes': ['not_convertible_under_L1_L2']}
        try:
            _layout(source)
        except (KeyError, ValueError, TypeError) as error:
            return {'eligible': False, 'reason': f'not_convertible_under_L1_L2:source_mechanism_unresolved:{error}',
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_mechanism_unresolved']}
        return {'eligible': True, 'reason': 'Original targets and lighting provenance support physical inventory.'}

    def convert(self, source, context):
        result = self.convert_legacy(source, context)
        if context.config.get('legacy_illumination_layout') or result.get('task_definition') is None:
            return result
        from .partial_occlusion import TASK_REVISION
        task = result['task_definition']
        parameters = task['private']['evaluation']['parameters']
        parameters.update(task_revision=TASK_REVISION, requires_native_visual_discovery=True,
            mechanism_contract={'preserved_relations': ['original target/light relative geometry',
                'original lamp transforms and source reference', 'actual measured normal/dim phases'],
                'required_native_evidence': ['matched delivered views in both measured phases',
                    'native lighting intensity and transforms', 'separate physical inventory operations in each phase'],
                'native_operation_verified': False})
        task['public']['environment'] = ('The original target/light geometry, source reference and lamp transforms remain unchanged. '
            'Measured normal and severely dimmed illumination alternate every 30 seconds of simulation time.')
        task['public']['success_criteria'] += (' Each required target must also appear in a delivered mounted-camera '
            'observation in each matching measured lighting phase.')
        result['interaction_policy'].update(relocation_policy='preserve', allow_fallback_table=False)
        result['interaction_policy'].pop('placement_groups', None)
        result['interaction_policy']['warning_codes'] = ['native_illumination_view_recheck_required']
        result['change_summary'] = ['Retain actual lighting conditions and meaningful separate operations across both phases.',
            'Preserve original target/light geometry and require synchronized delivered views for each phase.']
        result['assumptions'] = []
        result['limitations'] = ['The source omits exact sampled intensities; the restored measured lighting configuration remains explicit.',
            'A single normal-light image cannot qualify both phases; native reach and phase evidence remain pending.']
        return result

    def convert_legacy(self, source, context):
        layout = _layout(source)
        diagnostic = original_scene(source, context)
        try:
            restoration = _lamp_restore(layout, context)
            for item in layout['targets']:
                if not (_asset_root(context) / 'objects' / item['category'] / item['model']).is_dir():
                    raise ValueError('An original manipulation asset is not installed.')
            start = _robot_start(layout, context)
        except (OSError, KeyError, ValueError) as error:
            return {'scene_definition': diagnostic, 'task_definition': None, 'status': 'conversion_failed',
                    'failure_stage': 'original_asset_evidence',
                    'failure_reason': f'not_convertible_under_L1_L2:original_asset_evidence_unavailable:{error}',
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_only_diagnostic'],
                    'diagnostic_source_only': True, 'source_only': True, 'diagnostic_only': True,
                    'limitations': ['No physical recipe was qualified.'],
                    'next_step': 'Restore or inspect the original installed scene/assets/maps; preserve the source-only diagnostic.'}
        operation = context.config.get('illumination_operation', 'two_phase_inventory_return')
        if operation not in OPERATIONS:
            raise ValueError('Unsupported illumination recipe.')
        bindings = layout['bindings']
        targets, lamps = [key for key in bindings if key.startswith('target_')], [key for key in bindings if key.startswith('lamp_')]
        patches = [{'operation': 'category_initialization', 'phase': 'before_construction',
                    'module': MODULE, 'parameters': restoration}]
        if start is not None:
            patches.append({'operation': 'robot_pose', 'position': start['position'], 'yaw_deg': start['yaw_deg'],
                            'head_pitch_deg': 45.})
        # This source builder already installs R1. Do not add or replace it.
        for key in targets:
            patches.extend([{'operation': 'object_physics', 'binding': key, 'visual_only': False, 'fixed_base': False},
                            {'operation': 'object_physics', 'phase': 'after_postprocessing',
                             'binding': key, 'visual_only': False}])
        patches.append({'operation': 'category_initialization', 'phase': 'after_postprocessing',
                        'module': MODULE, 'parameters': restoration})
        category = layout['category'].replace('_', ' ')
        selected = ('every distinct' if operation == 'two_phase_inventory_return' else 'one chosen')
        objective = (f'Physically inventory {selected} original generated {category} object on its starting support. '
            'Observe the changing illumination without counting an existing object twice. For each selected item, '
            'make a separate grasp, lift and return entirely under normal illumination and entirely under dimmed illumination. '
            'Lift at least 5 cm, lower it back near its starting pose and release it on its starting support. '
            'Finish with all original targets in place and an empty open gripper.')
        public = {'objective': objective,
            'success_criteria': 'Each required item has two separate native grasp-and-lift events, one under each '
                'measured illumination condition, and is returned within 3 cm and 15 degrees of its initialized pose '
                'before that condition changes after each lift. Every target finishes touching its starting support '
                'and stationary for 15 controller '
                'physics samples, with an empty open gripper. A grasp spanning a light change does not qualify; complete each grasp, lift and return within one measured condition.',
            'environment': 'Original target dimensions and pairwise layout are retained. Portable targets may '
                'be elevated by a common vertical offset onto separate measured supports. Lamps, native light prims and the source reference observer remain fixed. '
                'The original reference views remain preserved in the unchanged original variant. '
                'Elevation changes target/light geometry, so original-view QA compatibility and illumination appearance require review. The robot must not move lamps. '
                'The restored environmental illumination alternates normal and severely dimmed conditions every '
                '30 seconds of simulation time. Original room furniture is restored for support measurement.',
            'constraints': ['Identify the original targets before moving them; the source count question is private provenance and is not scored.',
                'Use robot-mounted observations and existing guarded robot motion/gripper controls.',
                'Do not grasp or move lamps, task supports or semantic occluders. Native movable noncritical furniture may be moved with guarded arm/gripper actions. Preserve all initialized target starting locations.',
                'Human review images, source maps, saved poses, answers and private scorer state are not agent sensors.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': list(bindings), 'parameters': {'target_keys': targets, 'lamp_keys': lamps,
                'target_category': layout['category'], 'source_objects': layout['source_objects'], 'lighting': restoration,
                'fixed_lighting_anchors': True}}
        scene = original_scene(source, context, bindings=bindings, patches=patches)
        scene['runtime'] = {'post_controller_settle_steps': 30}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
            patches=patches, capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation',
                                          'trunk_height', 'gripper'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none'})
        task['defaults'].update(max_steps=300, wall_timeout_s=14400.)
        interaction_policy = {'mode': 'manipulation', 'group_bindings': targets,
            'manipulation_bindings': targets, 'relocation_policy': 'vertical_only',
            'placement_groups': [[key] for key in targets],
            'allow_fallback_table': True, 'anchor_policy': 'none', 'source_camera_policy': 'preserve',
            'support_search': 'measured_surfaces', 'source_region': layout['raw']['room'],
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
            'maximum_stance_distance_m': .85,
            'warning_codes': ['native_grasp_ik_and_aperture_pending',
                'source_context_changed_by_support_relocation', 'native_illumination_view_recheck_required']}
        assumptions = [{
            'doubt': 'Cleaned source JSON retains phase labels and lamp identities, but omits exact randomized '
                'light-intensity scales and does not define an interactive episode period.',
            'chosen_assumption': 'At initialization, restore original lamp assets and install a native environmental '
                'cycle using the original internal-stage-light-first strategy. Alternate 30 s normal/changed phases '
                'with a deterministic 0.04 intensity multiplier, within the generator severe-dimming range. '
                'Scoring reads actual light attributes and requires distinct physical events in both phases.',
            'alternatives': ['Recover the original uncleaned lighting_change values if available.',
                             'Use separately initialized normal and changed episodes with an external cross-episode protocol.'],
            'scope': 'The adapted variant only; original source views, answers and empty original variant remain preserved.',
            'risk': 'Exact historical brightness, randomness and timing are not reconstructed. Native visual propagation '
                'and meaningful image contrast still need measurement; callback failure or absent adjustable lights rejects qualification.'}, {
            'doubt': 'Source targets are visual-only and source camera points are not R1 reach/grasp certificates.',
            'chosen_assumption': 'Enable original target physics before construction and again after source postprocessing. '
                'Preserve scale and original fixed-base settings. Propose a map-screened robot start, then require '
                'the shared native initializer to measure a safe floor-supported stance and suitable manipulation height. '
                'If no CPU stance proposal qualifies, omit robot_pose and let that measured planner attempt initialization.',
            'alternatives': ['Keep a source-only diagnostic for failed native initialization.',
                             'Use the two_phase_single_return recipe to test one item before complete inventory.'],
            'scope': 'All adapted physical candidates; initialization-only repair, with no scaling or success teleportation.',
            'risk': 'Collision settling can alter visibility or prevent support. Native R1 IK and actual grasps '
                'remain unverified; static maps omit height-dependent body collisions.'}, {
            'doubt': 'Floor-level targets need accessible supports, but elevating targets changes shadows and illumination geometry.',
            'chosen_assumption': 'Use one common vertical target offset across separate measured supports, with existing supports first and the authorized simple fallback when needed. '
                'Keep all original lamps, selected native light transforms, intensity schedule and source observer fixed. '
                'Verify actual target support and stationary light transforms; retain original-view QA compatibility as unverified.',
            'alternatives': ['Keep the source-only floor-layout diagnostic.',
                             'Investigate a separately declared navigation inspection goal.',
                             'Use a small simple support only under the current user authorization and measured safety checks.'],
            'scope': 'The adapted variant only; all original targets move together, their pairwise relations '
                'and dimensions remain fixed, and the empty original variant retains original poses/views.',
            'risk': 'Restored category filters are global and may load additional furniture. Background occlusion, '
                'lamp-relative geometry and visible lighting contrast change and need Review. A support may not '
                'fit the unchanged formation; generated means pending native measurement, not reachable or qualified.'}]
        if restoration['support_categories']:
            assumptions.append({'doubt': 'Elevated source lamps lack an explicit support identity in cleaned JSON.',
                'chosen_assumption': 'Extend the original scene category whitelist with installed support categories '
                    'present in lamp rooms, retaining their original assets/scales/poses/physics. Reject lamp AABB drift '
                    'over 5 cm rather than fixing or moving the lamp.',
                'alternatives': ['Recover original contact/support measurements to narrow the category filter.',
                                 'Keep the unchanged source-only diagnostic if native support restoration fails.'],
                'scope': 'Elevated-lamp sources only; operation support restoration is declared separately. '
                    'Category filtering is global because the builder loads the full scene.',
                'risk': 'Extra original furniture of these categories can reappear elsewhere in the full scene, affect '
                    'visibility/navigation and increase load time. Exact support and resulting lamp stability are unverified.'})
        return {'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'operation': operation, 'available_operation_variants': list(OPERATIONS),
            'interaction_policy': interaction_policy,
            'change_summary': ['Restore original native lamp categories and evidenced potential original support categories.',
                'Install original-style normal/dim environmental phases at scene initialization.',
                'Enable original target collisions/gravity before construction and after source postprocessing.',
                'Elevate portable targets on separate measured supports with one common Z; preserve fixed original lamps/light prims/observer and warn changed illumination geometry.',
                'Add private measured two-phase grasp/lift/return inventory; initial QA remains auxiliary.'],
            'warning_codes': ['assumption_review_required', 'native_physical_execution_unverified',
                'source_visual_only_restored', 'source_light_values_not_retained', 'initialization_light_cycle_restored',
                'source_context_changed_by_support_relocation', 'native_illumination_view_recheck_required',
                'target_elevation_changes_illumination_geometry', 'original_view_qa_compatibility_unverified'] +
                (['source_static_start_unverified', 'native_spawn_repair_pending'] if start is None else []) +
                (['original_lamp_support_identity_unresolved'] if restoration['support_categories'] else []),
            'assumptions': assumptions, 'robot_start_evidence': start, 'lighting_restoration_evidence': restoration,
            'limitations': ['No native simulator, robot execution or screenshot was run by this CPU supervisor.',
                'Original historic dimming values are absent; timing/intensities are declared reversible assumptions.',
                'Native surface fit, source-group stability, render contrast, reach and full operation cost require measurement.'],
            'next_step': 'Publish all pending and diagnostic records to Review through the shared runner; '
                'measure native lighting, lamp support, robot safety and guarded physical execution.'}

    def convert_all(self, sources, context):
        # Keep the shared exhaustive, source-preserving, revisioned failure path.
        sources = sorted(sources, key=lambda source: source.relative_path)
        manifest = super().convert_all(sources, context)
        manifest['failure_distribution'] = dict(Counter(row.get('failure_reason') for row in manifest['records']
                                                        if row['status'] != 'generated'))
        manifest['operation_distribution'] = dict(Counter(row.get('operation') for row in manifest['records']
                                                          if row['status'] == 'generated'))
        manifest['sample_ids'] = self.sample(manifest, {'count': 5, 'seed': context.seed})
        manifest['native_physical_execution_verified'] = False
        manifest['feedback_revision'] = context.config.get('feedback_revision')
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Include failures and distinct original-light/restoration branches."""
        policy = policy or {}
        count = min(policy.get('count', 5), len(manifest['records']))
        rows = sorted(manifest['records'], key=lambda row: row['source_identity'])
        random.Random(policy.get('seed', 0)).shuffle(rows)
        selected = []
        prior = set(policy.get('previous_failures', ()))
        filters = [lambda row: row['status'] != 'generated',
                   lambda row: row['candidate_id'] in prior,
                   lambda row: bool(row.get('lighting_restoration_evidence', {}).get('support_categories')),
                   lambda row: any(lamp['category'] == 'floor_lamp' for lamp in
                                   row.get('lighting_restoration_evidence', {}).get('lamps', [])),
                   lambda row: len(row.get('lighting_restoration_evidence', {}).get('lamps', [])) >= 10,
                   lambda row: (row.get('robot_start_evidence') or {}).get('source_observer_or_rule') == 'original_room_grid']
        for predicate in filters:
            if len(selected) >= count:
                break
            eligible = [row for row in rows if row not in selected and predicate(row)]
            if eligible:
                scenes = {row.get('scene_model') for row in selected}
                selected.append(next((row for row in eligible if row.get('scene_model') not in scenes), eligible[0]))
        for row in rows:
            if len(selected) >= count:
                break
            if row not in selected:
                selected.append(row)
        return [row['candidate_id'] for row in selected]

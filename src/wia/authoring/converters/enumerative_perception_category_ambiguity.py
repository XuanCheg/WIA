"""Convert Category Ambiguity into native selective inventory operations.

Run from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Use OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 for CPU batching.
Native adapters are invoked by the existing guarded robot task runtime only.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from collections import Counter
import copy
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.tasks.background_policy import legal_background_contact
from wia.authoring.io import sha256, write_json
from wia.tasks.interaction_setup import reference_position, touching_task_support, on_top_task_support

MODULE = 'wia.authoring.converters.enumerative_perception_category_ambiguity'
QUESTION = re.compile(r'Considering both images, how many objects of category "([^"\n]+)" '
                      r'are present in this scene\?')
OPERATIONS = ('inventory_return', 'selective_lift')
MINIMUM_LIFT_M = .05
LAYOUT_TOLERANCE_M = .03
CONFUSER_TOLERANCE_M = .02
ORIENTATION_TOLERANCE_DEG = 15.
MAXIMUM_LINEAR_SPEED = .02
MAXIMUM_ANGULAR_SPEED = .1
STABILITY_STEPS = 15
FEEDBACK_REVISION = 'ddeb142c27bc8828f006_3ec04380ada5bb0a6d8e9026'
TABLE_SOURCE = ('Enumerative Perception/Category Ambiguity/'
                'Beechwood_0_garden/living_room_0/q_000.json')
TABLE_EVIDENCE = ('outputs/local_capture_20261014/queue_workers_initialized/jobs/'
                  'capture_4b0026529aed81611d509c17d6bb0959/attempts/01/preview_complete.json')


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


def _table_plan(source, items, floor_z, context):
    """The reviewed source gets a reversible table, never scaled/repacked props.

    The user specifically authorized a small table after this source's native
    bounded support search failed. Tiny raised rests support floating confusers
    at their source-relative heights. Their footprints exclude other prop AABBs.
    """
    if source.relative_path != TABLE_SOURCE:
        return None
    lows = np.array([item['bbox'][0] for item in items])
    highs = np.array([item['bbox'][1] for item in items])
    low, high = lows.min(axis=0), highs.max(axis=0)
    extent = high - low
    if np.any(extent[:2] > [1.2, 2.1]):
        raise ValueError('The reviewed small-table geometry changed; do not expand its authorization.')
    surface = floor_z + .65
    delta = np.array([0., 0., surface - low[2] + .002])
    centre = (low + high) / 2
    dimensions = np.r_[extent[:2] + .12, .045]
    data_root = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    asset_root = context.config.get('original_asset_root', str(data_root / 'behavior-1k-assets'))
    maps = _original_maps(str(asset_root), source.raw['scene'])
    if maps is None:
        raise ValueError('not_convertible_under_L1_L2:table_footprint_maps_unavailable')
    trav, rooms, map_hashes = maps
    corners = np.array([centre[:2] - dimensions[:2] / 2, centre[:2] + dimensions[:2] / 2])
    indices = (corners[:, ::-1] / .01 + trav.shape[0] / 2).astype(int)
    if np.any(indices < 0) or np.any(indices >= trav.shape[0]):
        raise ValueError('not_convertible_under_L1_L2:table_footprint_outside_source_map')
    slices = tuple(slice(int(indices[0, axis]), int(indices[1, axis]) + 1) for axis in (0, 1))
    room_id = _render(source)['room_bbox']['room_id']
    if not np.all(trav[slices] > 0) or not np.all(rooms[slices] == room_id):
        raise ValueError('not_convertible_under_L1_L2:table_footprint_not_clear_in_original_room')
    prefix = 'category_ambiguity_review_table'
    primitives = [{'name': prefix, 'position': [*centre[:2], surface - dimensions[2] / 2],
                   'dimensions_m': dimensions.tolist()}]
    for index, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        xy = centre[:2] + np.array([sx, sy]) * (dimensions[:2] / 2 - .045)
        height = surface - dimensions[2] - floor_z + .001
        primitives.append({'name': f'{prefix}_leg_{index}',
            'position': [*xy, floor_z + height / 2 - .001],
            'dimensions_m': [.045, .045, height]})
    supports = {}
    for index, item in enumerate(sorted(items, key=lambda row: row['name'])):
        lower, upper = np.array(item['bbox'])
        top = float(lower[2] + delta[2] - .002)
        supports[item['name']] = {'name': prefix, 'surface_z_m': surface}
        if top - surface <= .005:
            continue
        # A narrow rest can sit under an exposed end of a confuser without
        # intruding into a neighbouring original target. It is not a grasp aid.
        size = (upper[:2] - lower[:2]) * .45
        proposals = []
        for x in np.linspace(lower[0] + size[0] / 2, upper[0] - size[0] / 2, 5):
            for y in np.linspace(lower[1] + size[1] / 2, upper[1] - size[1] / 2, 5):
                rest_low = np.array([x - size[0] / 2, y - size[1] / 2, surface])
                rest_high = np.array([x + size[0] / 2, y + size[1] / 2, top])
                if any(np.all(rest_high > np.array(other['bbox'][0]) + delta + .001) and
                       np.all(rest_low < np.array(other['bbox'][1]) + delta - .001)
                       for other in items if other['name'] != item['name']):
                    continue
                proposals.append((math.dist([x, y], item['position'][:2]), x, y))
        if not proposals:
            raise ValueError('not_convertible_under_L1_L2:no_nonintersecting_source_relative_support_rest')
        _, x, y = min(proposals)
        name = f'{prefix}_rest_{index}'
        primitives.append({'name': name, 'position': [x, y, (surface + top) / 2],
                           'dimensions_m': [*size, top - surface]})
        supports[item['name']] = {'name': name, 'surface_z_m': top}
    plan = {'source_identity': source.relative_path, 'source_sha256': sha256(source.path),
            'scene': source.raw['scene'], 'room': source.raw['room'],
            'source_group_translation_m': delta.tolist(), 'primitives': primitives,
            'source_map_footprint_screen': {'passed': True, 'resolution_m': .01,
                'room_id': room_id, 'map_sha256': map_hashes, 'native_collision_verified': False},
            'object_supports': supports, 'objects': copy.deepcopy(items),
            'feedback_revision': FEEDBACK_REVISION, 'historical_native_evidence': TABLE_EVIDENCE,
            'authorization': 'User repair feedback permits a small table when suitable existing supports are absent.'}
    # Geometry arithmetic can produce numpy scalar floats. Parameters are a
    # serialized public API boundary, so emit ordinary JSON/YAML scalar types.
    return json.loads(json.dumps(plan, allow_nan=False))


def configure_scene(config, parameters):
    """Construct only the feedback-authorized primitive table before simulation."""
    plan = parameters['table_plan']
    if plan['source_identity'] != TABLE_SOURCE or plan['feedback_revision'] != FEEDBACK_REVISION:
        raise ValueError('A primitive table requires the reviewed source and explicit feedback provenance.')
    objects = config.setdefault('objects', [])
    for item in plan['objects']:
        obj = next((row for row in objects if row.get('name') == item['name']), None)
        if obj is None or any(obj.get(field) != item[field] for field in ('category', 'model')):
            raise ValueError('Table initialization must retain every original source asset.')
        obj['position'] = (_array(item['position']) + _array(plan['source_group_translation_m'])).tolist()
        obj['orientation'] = item['quaternion_xyzw']
    for item in plan['primitives']:
        if any(obj.get('name') == item['name'] for obj in objects):
            raise ValueError('Table primitives cannot replace source assets.')
        objects.append({'type': 'PrimitiveObject', 'name': item['name'], 'category': 'breakfast_table',
            'primitive_type': 'Cube', 'size': 1., 'scale': item['dimensions_m'],
            'position': item['position'], 'orientation': [0., 0., 0., 1.],
            'fixed_base': True, 'visual_only': False, 'rgba': [.36, .27, .18, 1.]})
    return {'source_identity': plan['source_identity'], 'source_sha256': plan['source_sha256'],
            'constructed_supports': [row['name'] for row in plan['primitives']],
            'source_group_translation_m': plan['source_group_translation_m'],
            'new_asset_files': [], 'physical_grasp_verified': False}


def initialize_scene(session, parameters):
    """Restore declared starting poses and settle once, before any task action.

    Scene and task paths use this identical hook. Qualification/evaluation never
    call it. Rejected measurements stay private diagnostics, not success flags.
    """
    if getattr(session, '_category_ambiguity_initialized', False):
        raise RuntimeError('Category initialization is not an episode action and cannot be repeated.')
    session._category_ambiguity_initialized = True
    plan = parameters.get('table_plan')
    evidence = {'accepted': False, 'table_plan': plan, 'settle_steps': 30,
                'physical_grasp_verified': False, 'reference_camera_changed': False}
    session.category_ambiguity_initialization = evidence
    if plan:
        import torch
        if (session.scene_name, session.room_name) != (plan['scene'], plan['room']):
            raise ValueError('The table repair belongs to another original scene/room.')
        # The source postprocessor may settle visual-only floor props. Restore
        # the declared translated source group before measuring the new start.
        for item in plan['objects']:
            obj = session.env.scene.object_registry('name', item['name'])
            if obj is None or (obj.category, obj.model) != (item['category'], item['model']):
                raise ValueError('A source object disappeared during initialization.')
            obj.set_position_orientation(position=torch.tensor(_array(item['position']) +
                _array(plan['source_group_translation_m']), dtype=torch.float32),
                orientation=torch.tensor(item['quaternion_xyzw'], dtype=torch.float32))
            obj.keep_still()
        from omnigibson.utils.usd_utils import PoseAPI
        PoseAPI.invalidate()
        session.invalidate_camera_cache()
    for _ in range(30):
        session.pipeline.og.sim.step()
    if not plan:
        evidence.update(accepted=True, scope='Shared settling only; measured interaction placement follows.')
        return evidence
    try:
        from omnigibson.object_states import Touching
        floor = session.env.scene.object_registry('name', parameters['floor_name'])
        rows = {item['name']: _snapshot(session.env.scene.object_registry('name', item['name']), floor, session)
                for item in plan['objects']}
        legs = [session.env.scene.object_registry('name', row['name']) for row in plan['primitives']
                if '_leg_' in row['name']]
        support_objects = [session.env.scene.object_registry('name', row['name']) for row in plan['primitives']]
        native_objects = session.env.scene.objects
        native_objects = list(native_objects.values()) if isinstance(native_objects, dict) else list(native_objects)
        allowed_paths = [obj.prim_path for obj in native_objects if obj.name in {
            row['name'] for row in plan['primitives'] + plan['objects']} or obj.category in ('floor', 'floors')]
        foreign_contacts = []
        for obj in support_objects:
            if obj is None:
                continue
            for contact in obj.contact_list():
                for body in (contact.body0, contact.body1):
                    body = str(body)
                    if 'groundPlane' not in body and not any(body == path or body.startswith(path + '/')
                                                           for path in allowed_paths):
                        foreign_contacts.append(body)
        checks = {'source_relative_layout': all(_near(rows[item['name']],
            {**item, 'position': (_array(item['position']) + _array(plan['source_group_translation_m'])).tolist()},
            LAYOUT_TOLERANCE_M) for item in plan['objects']),
            'all_objects_supported_stationary': all(row['touching_task_support'] and _stationary(row)
                for row in rows.values()),
            'table_legs_touch_original_floor': all(obj is not None and not obj.visual_only and
                obj.fixed_base and obj.states[Touching].get_value(floor) for obj in legs),
            'all_supports_fixed_collidable': all(obj is not None and obj.fixed_base and not obj.visual_only
                                                for obj in support_objects),
            'no_table_background_contacts': not foreign_contacts}
        evidence.update(accepted=all(checks.values()), checks=checks, native_objects=rows,
            foreign_table_contact_bodies=sorted(set(foreign_contacts)),
            reason=None if all(checks.values()) else 'native_table_start_requires_review')
    except Exception as error:
        evidence.update(reason='native_table_measurement_unavailable', exception_summary=f'{type(error).__name__}: {error}')
    return evidence


def _reference_position(session, obj, original):
    plan = getattr(session, 'category_ambiguity_initialization', {}).get('table_plan')
    if plan and obj.name in plan['object_supports']:
        return (_array(original) + _array(plan['source_group_translation_m'])).tolist()
    return reference_position(session, obj.name, original)


def _touching_support(session, obj, floor):
    plan = getattr(session, 'category_ambiguity_initialization', {}).get('table_plan')
    if not plan or obj.name not in plan['object_supports']:
        return touching_task_support(session, obj, floor)
    from omnigibson.object_states import Touching
    from wia.tasks.interaction_setup import array
    spec = plan['object_supports'][obj.name]
    support = session.env.scene.object_registry('name', spec['name'])
    if support is None or support.visual_only or not support.fixed_base or not obj.states[Touching].get_value(support):
        return False
    for contact in obj.contact_list():
        if not any(str(body) == support.prim_path or str(body).startswith(support.prim_path + '/')
                   for body in (contact.body0, contact.body1)):
            continue
        normal, point = getattr(contact, 'normal', None), getattr(contact, 'position', None)
        if normal is not None and point is not None and abs(float(array(normal)[2])) >= .5 and abs(
                float(array(point)[2]) - spec['surface_z_m']) <= .015:
            return True
    return False


def _on_top_support(session, obj, floor):
    """Read the actual per-object support, including historical reviewed rests."""
    plan = getattr(session, 'category_ambiguity_initialization', {}).get('table_plan')
    if not plan or obj.name not in plan['object_supports']:
        return on_top_task_support(session, obj, floor)
    from omnigibson.object_states import OnTop
    support = session.env.scene.object_registry('name', plan['object_supports'][obj.name]['name'])
    return bool(support is not None and obj.states[OnTop].get_value(support))


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Native/source measurements must be finite.')
    return result


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
        for item in value)


def _render(source):
    return source.payload.get('question_data', {}).get('render', {})


def _dependency_groups(source, resolved, bindings):
    """Match exact saved source-entry provenance; never guess nearest confusers."""
    keys = {binding['name']: key for key, binding in bindings.items() if key != 'floor'}
    groups = []
    for entry in source.payload['question_data']['case_metadata']['source_entries']:
        points = entry['ball_positions']
        targets = [item for item in resolved['targets'] if any(np.linalg.norm(
            _array(item.get('requested_position', item['position'])) - _array(point)) <= 1e-4 for point in points)]
        saved = entry['metadata']['confuser_object']
        confusers = [item for item in resolved['confusers'] if item['category'] == saved['category'] and np.linalg.norm(
            _array(item.get('requested_position', item['position'])) - _array(saved['position'])) <= 1e-4]
        if len(targets) != len(points) or len(confusers) != 1:
            raise ValueError('Original target/confuser entry provenance is unresolved or ambiguous.')
        groups.append([keys[item['name']] for item in targets + confusers])
    members = [key for group in groups for key in group]
    if not groups or len(members) != len(set(members)) or set(members) != set(keys.values()):
        raise ValueError('Source dependency groups must cover every original target and confuser once.')
    return groups


def _dependency_translations_preserved(session, parameters, objects):
    for group in parameters.get('placement_groups', []):
        deltas = [_reference_position(session, objects[key], parameters['source_objects'][key]['position']) -
                  _array(parameters['source_objects'][key]['position']) for key in group]
        if any(np.linalg.norm(delta - deltas[0]) > 1e-6 for delta in deltas):
            return False
    return True


def _camera(render):
    cameras = render.get('camera_poses') or []
    if isinstance(cameras, dict):
        return cameras.get(Path(render.get('image', '')).name) or next(iter(cameras.values()), {})
    return next((row for row in cameras if row.get('_key') == Path(render.get('image', '')).name),
                cameras[0] if cameras else {})


@lru_cache(maxsize=2)
def _original_maps(asset_root, scene_model):
    """Read the installed original 1 cm maps on CPU, without importing simulation."""
    import cv2
    layout = Path(asset_root) / 'scenes' / scene_model / 'layout'
    paths = [layout / name for name in ('floor_trav_0.png', 'floor_insseg_0.png')]
    if not all(path.is_file() for path in paths):
        return None
    traversability, rooms = (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths)
    if traversability is None or rooms is None or traversability.shape != rooms.shape:
        return None
    return traversability, rooms, {str(path): sha256(path) for path in paths}


def _body_map_screen(maps, xy, room_id):
    """A conservative static disk screen is supplementary to native qualification."""
    traversability, rooms, _ = maps
    radius_px, resolution = 40, .01
    row, column = [int(value / resolution + traversability.shape[0] / 2.) for value in (xy[1], xy[0])]
    if (row < radius_px or column < radius_px or row + radius_px >= traversability.shape[0] or
            column + radius_px >= traversability.shape[1]):
        return False
    offsets = np.arange(-radius_px, radius_px + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= radius_px ** 2
    slices = (slice(row - radius_px, row + radius_px + 1),
              slice(column - radius_px, column + radius_px + 1))
    return bool(np.all(traversability[slices][disk] > 0) and np.all(rooms[slices][disk] == room_id))


def _robot_start(source, render, items, context):
    """Screen source observer points, without claiming native body traversability."""
    bounds = render.get('room_bbox', {}).get('expanded_bbox_world_xy') or render.get(
        'room_bbox', {}).get('bbox_world_xy')
    if not _vector(bounds, 4) or bounds[2] - bounds[0] <= 1.3 or bounds[3] - bounds[1] <= 1.3:
        return None
    data_root = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    asset_root = context.config.get('original_asset_root', str(data_root / 'behavior-1k-assets'))
    maps = _original_maps(str(asset_root), source.raw['scene'])
    room_id = render.get('room_bbox', {}).get('room_id')
    if maps is None or not isinstance(room_id, int) or room_id <= 0:
        return None
    camera = _camera(render)
    if not _vector(camera.get('position'), 3):
        return None
    eye = camera['position']
    candidates = [('primary_camera', eye[:2], False)]
    for row in render.get('multi_image_input', {}).get('views', []):
        point = row.get('camera_pose', {}).get('position')
        if row.get('is_traversable') is True and _vector(point, 3):
            candidates.append((row.get('view_id', 'source_corner'), point[:2], True))
    screened = []
    for label, point, traversable in candidates:
        xy = [min(max(point[axis], bounds[axis] + .65), bounds[axis + 2] - .65)
              for axis in (0, 1)]
        clearance = min(math.sqrt(sum(max(item['bbox'][0][axis] - xy[axis],
            xy[axis] - item['bbox'][1][axis], 0.) ** 2 for axis in (0, 1))) for item in items)
        if clearance >= .65 and _body_map_screen(maps, xy, room_id):
            screened.append((traversable, -math.dist(xy, eye[:2]), clearance, label, point, xy))
    if not screened:
        return None
    traversable, _, clearance, label, original_xy, xy = max(screened)
    target = camera.get('look_target')
    if not _vector(target, 3):
        target = np.mean([item['position'] for item in items], axis=0).tolist()
    # batch_counting_merge._sample_primary_view_pose sets eye_z=floor_top+1.25.
    # This preserves the room's source floor height rather than assuming z=0.
    floor_z = float(eye[2] - 1.25)
    yaw = math.degrees(math.atan2(target[1] - xy[1], target[0] - xy[0]))
    return {'position': [*xy, floor_z], 'yaw_deg': yaw,
            'observer_id': label, 'original_observer_xy': list(original_xy),
            'source_point_traversable_flag': traversable,
            'source_bbox_clearance_m': clearance,
            'static_body_map_screen_radius_m': .4,
            'static_body_map_screen_passed': True,
            'source_room_id': room_id, 'original_map_sha256': maps[2],
            'source_floor_z_estimate_m': floor_z,
            'provenance': 'question_data.render.camera_poses and multi_image_input.views; '
                          'room_bbox and original generated-object bbox screens; original installed '
                          'floor_trav_0/floor_insseg_0 maps, 1 cm resolution and map_base world_to_map. '
                          'Floor height: batch_counting_merge._sample_primary_view_pose '
                          '(AGENT_CAMERA_HEIGHT_M=1.25). '
                          'R1 clearance and floor support require native qualification.'}


def _snapshot(obj, floor, session):
    """Read actual native pose, AABB, contact and velocities; never change state."""
    from omnigibson.object_states import Touching
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    low, high = (_array(value) for value in obj.aabb)
    if position.shape != (3,) or quaternion.shape != (4,) or low.shape != (3,) or high.shape != (3,):
        raise ValueError('Native object measurements have an unexpected shape.')
    if np.linalg.norm(quaternion) < 1e-6 or np.any(high <= low):
        raise ValueError('Native orientation or physical extent is invalid.')
    return {'name': obj.name, 'category': obj.category, 'model': obj.model,
            'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'touching_original_floor': bool(obj.states[Touching].get_value(floor)),
            'touching_task_support': _touching_support(session, obj, floor),
            'on_top_task_support': _on_top_support(session, obj, floor),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}


def _angle(first, second):
    a, b = _array(first), _array(second)
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return math.degrees(2. * math.acos(min(1., dot)))


def _stationary(row):
    return (row['linear_speed_m_s'] <= MAXIMUM_LINEAR_SPEED and
            row['angular_speed_rad_s'] <= MAXIMUM_ANGULAR_SPEED)


def _near(row, origin, tolerance):
    return (math.dist(row['position'], origin['position']) <= tolerance and
            _angle(row['quaternion_xyzw'], origin['quaternion_xyzw']) <= ORIENTATION_TOLERANCE_DEG)


def _configuration(session, config):
    if config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or config.get(
            'operation') not in OPERATIONS:
        raise ValueError('Use this owned native inventory adapter and a supported operation.')
    if (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name):
        raise ValueError('The evaluation belongs to a different native source scene/room.')
    parameters = config['parameters']
    targets, confusers = parameters['target_keys'], parameters['confuser_keys']
    keys = [*targets, *confusers, 'floor']
    if not targets or not confusers or len(set(keys)) != len(keys) or set(config['binding_keys']) != set(keys):
        raise ValueError('Native inventory needs unique original target, confuser and floor bindings.')
    if set(config['bindings']) != set(keys) or set(parameters['source_objects']) != set(targets + confusers):
        raise ValueError('The original object identities and private bindings must be complete.')
    if len(set(config['bindings'].values())) != len(keys):
        raise ValueError('Each inventory binding must refer to a distinct native object.')
    if parameters.get('placement_groups'):
        members = [key for group in parameters['placement_groups'] for key in group]
        if (any(not group for group in parameters['placement_groups']) or len(members) != len(set(members)) or
                set(members) != set(targets + confusers)):
            raise ValueError('Original target/confuser dependency groups must partition the inventory.')
    return parameters


def _native_robot_start(session):
    """Read room segmentation and chassis contacts, retaining native safety guards."""
    robot, scene = session.robot, session.env.scene
    position, _ = robot.get_position_orientation()
    position = _array(position)
    low, high = (_array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    regions = [scene.seg_map.get_room_instance_by_point(np.array([x, y]))
               for x in (low[0], (low[0] + high[0]) / 2, high[0])
               for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj.prim_path for obj in objects if obj.category in ('floor', 'floors')]
    floor_contacts, nonfloor_contacts = [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not (body == robot.prim_path or
                                                   body.startswith(robot.prim_path + '/'))]
        if not external:
            continue
        on_floor = all('groundPlane' in body or any(body == path or body.startswith(path + '/')
                                                   for path in floors) for body in external)
        (floor_contacts if on_floor else nonfloor_contacts).append(bodies)
    return {'position': position.tolist(), 'footprint_rooms': regions,
            'in_original_room': bool(regions and all(region == session.room_name for region in regions)),
            'floor_contacts': floor_contacts, 'nonfloor_contacts': nonfloor_contacts}


def qualify_operation(session, executor, config, bound):
    if config['operation'] == 'inventory_place':
        from .partial_occlusion import qualify_visual_discovery
        result = qualify_visual_discovery(session, config, bound)
        result['destination_fit_access_verified'] = False
        return result

    """Measure native preconditions only; no actions, setters or simulation steps."""
    checks = {}
    try:
        parameters = _configuration(session, config)
        floor = bound['floor']
        keys = parameters['target_keys'] + parameters['confuser_keys']
        rows = {key: _snapshot(bound[key], floor, session) for key in keys}
        robot = _native_robot_start(session)
        gripper = executor.debug_gripper_state()
        checks = {'original_floor_collidable': not floor.visual_only,
                  'all_generated_objects_movable_collidable': all(not row['visual_only'] and not
                      row['fixed_base'] for row in rows.values()),
                  'initialization_accepted': getattr(session, 'category_ambiguity_initialization', {}).get('accepted') is True,
                  'measured_interaction_accepted': getattr(session, 'category_interaction_setup', {}).get('accepted') is True,
                  'within_dependency_group_translation': _dependency_translations_preserved(session, parameters, bound),
                  'source_layout_preserved': all(_near(row, {**parameters['source_objects'][key],
                      'position': _reference_position(session, bound[key],
                          parameters['source_objects'][key]['position'])},
                      LAYOUT_TOLERANCE_M) for key, row in rows.items()),
                  'original_category_bindings': all(rows[key]['category'] == parameters['target_category']
                      for key in parameters['target_keys']) and all(rows[key]['category'] !=
                      parameters['target_category'] for key in parameters['confuser_keys']),
                  'all_generated_objects_supported_stationary': all(row['touching_task_support'] and row['on_top_task_support'] and
                      _stationary(row) for row in rows.values()),
                  'initial_gripper_empty': gripper.get('object_name') is None and
                      gripper.get('constraint_active') is False,
                  'robot_in_original_room': robot['in_original_room'],
                  'robot_supported_without_obstacle_contact': bool(robot['floor_contacts']) and not
                      robot['nonfloor_contacts']}
        return {'accepted': bool(all(checks.values())), 'checks': checks,
                'reason': None if all(checks.values()) else 'native_inventory_initialization_failed',
                'native_objects': rows, 'native_robot': robot,
                'reach_grasp_and_complete_inventory_qualified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_inventory_measurement_unavailable',
                'exception_summary': f'{type(error).__name__}: {error}',
                'reach_grasp_and_complete_inventory_qualified': False}


class InventoryEvaluator:
    """Track distinct native lift/release events; QA and source flags cannot score."""
    def __init__(self, session, controller, config, output):
        self.parameters = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.objects = {key: session.env.scene.object_registry('name', name)
                        for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('An original native inventory binding is missing.')
        self.targets = set(self.parameters['target_keys'])
        self.confusers = set(self.parameters['confuser_keys'])
        self.episode, self.initial, self.last = 0, None, None

    def measure(self):
        rows = {key: _snapshot(self.objects[key], self.objects['floor'], self.session)
                for key in sorted(self.targets | self.confusers)}
        return rows, self.controller.debug_gripper_state()

    def begin_episode(self, observation):
        self.episode += 1
        self.initial, gripper = self.measure()
        self.initially_empty = gripper.get('object_name') is None and gripper.get('constraint_active') is False
        self.initial_layout_valid = all(not row['visual_only'] and not row['fixed_base'] and
            row['touching_task_support'] and row['on_top_task_support'] and _stationary(row) and _near(row,
            {**self.parameters['source_objects'][key], 'position': _reference_position(self.session,
                self.objects[key], self.parameters['source_objects'][key]['position'])},
            LAYOUT_TOLERANCE_M) for key, row in self.initial.items())
        self.initial_layout_valid = bool(self.initial_layout_valid and
            _dependency_translations_preserved(self.session, self.parameters, self.objects) and
            getattr(self.session, 'category_ambiguity_initialization', {}).get('accepted') is True and
            getattr(self.session, 'category_interaction_setup', {}).get('accepted') is True)
        self.lifted, self.released = set(), set()
        self.return_stability = dict.fromkeys(self.targets, 0)
        self.previous_held = None
        self.pending_release = None
        self.confusers_undisturbed = True
        self.wrong_object_grasped = False
        self.stable_steps = 0
        self.stability_binding = None
        self.record(observation, action=None)

    def _update(self, count_step=False):
        if self.initial is None:
            return None
        rows, gripper = self.measure()
        attached_name = gripper.get('object_name') if gripper.get('holding') is True and gripper.get(
            'constraint_active') is True else None
        held = next((key for key, obj in self.objects.items() if obj.name == attached_name), None)
        target_names = {self.objects[key].name for key in self.targets}
        if (gripper.get('object_name') is not None and gripper['object_name'] not in target_names
                and not _legal_background_grasp(self.session, self.controller, gripper['object_name'])):
            self.wrong_object_grasped = True
        self.confusers_undisturbed = self.confusers_undisturbed and all(
            _near(rows[key], self.initial[key], CONFUSER_TOLERANCE_M) for key in self.confusers)
        lift_m = {key: rows[key]['aabb_min'][2] - self.initial[key]['aabb_min'][2] for key in self.targets}
        if held in self.targets and lift_m[held] >= MINIMUM_LIFT_M:
            self.lifted.add(held)
        if held in self.targets:
            self.pending_release = None
        elif self.previous_held in self.lifted:
            self.pending_release = self.previous_held
        if self.pending_release in self.lifted and gripper.get('object_name') is None and gripper.get(
                'constraint_active') is False and gripper.get('gripper_open') is True:
            key = self.pending_release
            if (_near(rows[key], self.initial[key], LAYOUT_TOLERANCE_M) and
                    rows[key]['touching_task_support'] and rows[key]['on_top_task_support']):
                self.released.add(key)
                self.pending_release = None
        self.previous_held = held
        objects_valid = all(not row['visual_only'] and not row['fixed_base'] for row in rows.values())
        common = self.initially_empty and self.initial_layout_valid and objects_valid and not self.wrong_object_grasped and self.confusers_undisturbed
        if self.config['operation'] == 'selective_lift':
            terminal = common and held in self.targets and lift_m[held] >= MINIMUM_LIFT_M
            # Consecutive hold samples belong to one native object. Even a
            # record-only boundary changing identity must reset that interval.
            if not terminal or held != self.stability_binding:
                self.stable_steps = 0
            self.stability_binding = held if terminal else None
        else:
            terminal = (common and self.lifted == self.targets and self.released == self.targets and
                        gripper.get('object_name') is None and gripper.get('constraint_active') is False and
                        gripper.get('gripper_open') is True and all(_near(rows[key], self.initial[key],
                        LAYOUT_TOLERANCE_M) and rows[key]['touching_task_support'] and
                        rows[key]['on_top_task_support'] and _stationary(rows[key])
                        for key in self.targets | self.confusers))
        if count_step:
            self.stable_steps = self.stable_steps + 1 if terminal else 0
        elif not terminal:
            self.stable_steps = 0
        self.last = {'native_objects': rows, 'gripper': gripper, 'held_binding': held,
                     'native_lift_m': lift_m, 'lifted_bindings': sorted(self.lifted),
                     'released_at_start_bindings': sorted(self.released),
                     'continuous_hold_binding': self.stability_binding,
                     'criteria': {'initially_empty': self.initially_empty,
                                  'initial_source_layout_supported': self.initial_layout_valid,
                                  'objects_still_movable_collidable': objects_valid,
                                  'confusers_undisturbed': self.confusers_undisturbed,
                                  'no_wrong_object_grasp': not self.wrong_object_grasped,
                                  'native_terminal_state': bool(terminal)},
                     'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS,
                     'task_success': bool(terminal and self.stable_steps >= STABILITY_STEPS)}
        if self.config['operation'] == 'selective_lift':
            items = [('selective_lift', self.last['task_success'], 'terminal_condition',
                      {'held_binding': held, 'stable_steps': self.stable_steps})]
        else:
            items = []
            for key in sorted(self.targets):
                returned = (key in self.lifted and key in self.released and held != key and
                    _near(rows[key], self.initial[key], LAYOUT_TOLERANCE_M) and
                    rows[key]['touching_task_support'] and rows[key]['on_top_task_support'] and _stationary(rows[key]))
                if count_step:
                    self.return_stability[key] = self.return_stability[key] + 1 if returned else 0
                elif not returned:
                    self.return_stability[key] = 0
                items.append((f'{key}:grasp_lift_return', self.return_stability[key] >= STABILITY_STEPS,
                    'terminal_condition', {'lifted': key in self.lifted, 'returned': key in self.released,
                    'stable_steps': self.return_stability[key]}))
        return _completion_result(self.last, items)

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        value = self._update()
        if value is None:
            return None
        value = {**value, 'schema': 'private_category_ambiguity_inventory_v1', 'episode': self.episode,
                 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                 'action': action, 'configuration': self.config, 'initial_native_objects': self.initial}
        write_json(self.output / 'state.json', value)
        # Keep this private trajectory compatible with the ordinary native task runtime.
        import json
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config['operation'] == 'inventory_place':
        return InventoryPlacementEvaluator(session, controller, config, output)

    return InventoryEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Enumerative Perception', 'Category Ambiguity')
    version = '2.0.0_task_edit'

    def inspect(self, source):
        raw, render = source.raw, _render(source)
        if (source.big_task, source.small_task) != self.category or (raw.get('big_task'), raw.get('small_task')) != self.category:
            return {'eligible': False, 'reason': 'category_mismatch', 'warning_codes': ['not_convertible_under_L1_L2']}
        match = QUESTION.fullmatch(raw.get('question', '').strip())
        resolved = render.get('resolved_objects', {})
        targets, confusers = resolved.get('targets', []), resolved.get('confusers', [])
        reason = None
        if not match:
            reason = 'unsupported_public_category_question'
        elif not targets or not confusers or resolved.get('containers'):
            reason = 'missing_semantic_inventory_or_unsupported_container_mechanism'
        elif match[1] != render.get('target_category') or any(item.get('category') != match[1] for item in targets) or any(
                item.get('category') == match[1] for item in confusers):
            reason = 'public_category_resolved_objects_disagree'
        else:
            items = targets + confusers
            if any(not all(isinstance(item.get(field), str) and item[field] for field in ('name', 'category', 'model')) or
                   not _vector(item.get('position'), 3) or not _vector(item.get('quaternion_xyzw'), 4) or
                   not any(item['quaternion_xyzw']) or not isinstance(item.get('bbox'), list) or len(item['bbox']) != 2 or
                   not all(_vector(row, 3) for row in item['bbox']) or any(item['bbox'][1][axis] <= item['bbox'][0][axis]
                   for axis in (0, 1, 2)) for item in items):
                reason = 'incomplete_original_object_geometry'
            elif len({item['name'] for item in items}) != len(items) or not source.payload.get('floor_name'):
                reason = 'missing_unique_original_objects_or_floor'
        return {'eligible': reason is None, 'reason': reason,
                'warning_codes': ['not_convertible_under_L1_L2'] if reason else []}

    def convert(self, source, context):
        if context.config.get('legacy_category_ambiguity'):
            return self.convert_legacy(source, context)
        from .partial_occlusion import TASK_REVISION
        raw, payload, render = source.raw, source.payload, _render(source)
        resolved = render['resolved_objects']
        bindings, roles = {'floor': {'name': payload['floor_name']}}, {}
        for role in ('targets', 'confusers'):
            roles[role] = []
            for index, item in enumerate(sorted(resolved[role], key=lambda item: item['name'])):
                key = ('target' if role == 'targets' else 'confuser') + f'_{index:02d}'
                roles[role].append(key)
                bindings[key] = {field: item[field] for field in ('name', 'category', 'model')}
        destination = context.config.get('placement_destination')
        if not destination:
            return {'scene_definition': original_scene(source, context, bindings=bindings),
                'task_definition': None, 'status': 'needs_human_decision', 'failure_stage': 'destination_selection',
                'failure_reason': 'placement_destination_pending', 'warning_codes': ['placement_destination_pending']}
        if destination['relation'] not in ('on_top', 'inside') or not destination.get('public_label'):
            raise ValueError('Destination requires explicit named on_top or inside placement.')
        bindings['destination'] = destination['binding'].copy()
        if bindings['destination']['name'] in {bindings[key]['name'] for key in roles['targets'] + roles['confusers']}:
            raise ValueError('Destination cannot be a target or confuser.')
        category = render['target_category'].replace('_', ' ')
        count = len(roles['targets'])
        prep = 'on' if destination['relation'] == 'on_top' else 'inside'
        label = destination['public_label']
        public = {'objective': f'Find all {count} distinct {category} objects among the similar-looking objects. '
            + f'Place every one {prep} {label} and leave the other object categories undisturbed.',
            'success_criteria': f'All {count} original {category} objects are released at the named destination and '
                + 'remain stable for 15 consecutive physics steps; each distinct target counts once.',
            'environment': 'Original target/confuser inventory, dimensions, clusters and source view are retained.',
            'constraints': ['Use the category and observed appearance to select targets; private object identifiers are not task instructions.',
                'Use native guarded controls; no lift threshold or return to initial positions is required.']}
        parameters = {'task_revision': TASK_REVISION, 'target_keys': roles['targets'], 'confuser_keys': roles['confusers'],
            'placement_relation': destination['relation'], 'destination_selection_seed': context.seed,
            'release_stability_steps': STABILITY_STEPS, 'maximum_release_speed_m_s': MAXIMUM_LINEAR_SPEED,
            'maximum_release_angular_speed_rad_s': MAXIMUM_ANGULAR_SPEED,
            'completion_item_ids': ['placed:' + key for key in roles['targets']],
            'mechanism_contract': {'preserved_relations': ['complete target/confuser inventory',
                'source category ambiguity and cluster geometry', 'asset scales and source observer'],
                'required_native_evidence': ['initial appearance and category discrimination',
                    'destination fit and access for every target', 'stable releases and protected confusers'],
                'native_operation_verified': False}}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': 'inventory_place',
            'binding_keys': list(bindings), 'parameters': parameters}
        patches = []
        for key in roles['targets'] + roles['confusers']:
            patches += [{'operation': 'object_physics', 'phase': phase, 'binding': key, 'visual_only': False}
                        for phase in ('before_construction', 'after_postprocessing')]
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches)
        return {'scene_definition': original_scene(source, context, bindings=bindings, patches=patches),
            'task_definition': task, 'status': 'generated', 'reuse_level': 'L2', 'operation_recipe': 'inventory_place',
            'interaction_policy': {'mode': 'manipulation', 'group_bindings': roles['targets'] + roles['confusers'],
                'manipulation_bindings': roles['targets'], 'relocation_policy': 'preserve',
                'allow_fallback_table': False, 'source_region': raw['room']},
            'warning_codes': ['native_destination_fit_access_pending', 'native_category_ambiguity_pending'],
            'change_summary': ['Require the complete original target inventory at a seeded explicit destination.',
                'Preserve confusers, source geometry and scales; remove forced return and lift threshold.'],
            'limitations': ['Native geometry, every-target access and destination fit remain unqualified.']}

    def convert_legacy(self, source, context):
        eligibility = self.inspect(source)
        scene = original_scene(source, context)
        if not eligibility['eligible']:
            return {'scene_definition': None, 'task_definition': None, 'status': 'conversion_failed',
                    'failure_reason': eligibility['reason'], 'failure_stage': 'source_inspection',
                    'warning_codes': eligibility['warning_codes'], 'diagnostic_source_only': True,
                    'source_only': True, 'diagnostic_only': True}
        raw, payload, render = source.raw, source.payload, _render(source)
        resolved = render['resolved_objects']
        items = resolved['targets'] + resolved['confusers']
        start = _robot_start(source, render, items, context)
        operation = context.config.get('operation', 'inventory_return')
        if operation not in OPERATIONS:
            raise ValueError('Supported physical recipes: inventory_return and selective_lift.')
        bindings, source_objects = {'floor': {'name': payload['floor_name']}}, {}
        role_keys = {}
        for role in ('targets', 'confusers'):
            keys = []
            for index, item in enumerate(sorted(resolved[role], key=lambda item: item['name'])):
                key = ('target' if role == 'targets' else 'confuser') + f'_{index:02d}'
                bindings[key] = {field: item[field] for field in ('name', 'category', 'model')}
                source_objects[key] = {field: item[field] for field in ('position', 'quaternion_xyzw', 'bbox')}
                keys.append(key)
            role_keys[role] = keys
        patches = [] if start is None else [{'operation': 'robot_pose', 'position': start['position'],
                    'yaw_deg': start['yaw_deg'], 'head_pitch_deg': 45.}]
        table_plan = None if start is None else _table_plan(source, items, start['source_floor_z_estimate_m'], context)
        try:
            groups = [sorted(source_objects)] if table_plan else _dependency_groups(source, resolved, bindings)
        except (ValueError, KeyError, TypeError) as error:
            return {'scene_definition': original_scene(source, context, bindings=bindings), 'task_definition': None,
                'status': 'conversion_failed', 'failure_stage': 'source_dependency_groups',
                'failure_reason': str(error), 'warning_codes': ['source_dependency_groups_unresolved', 'source_only_diagnostic']}
        initialization = {'floor_name': payload['floor_name'], 'table_plan': table_plan,
            'source_provenance': {'source_identity': source.relative_path, 'source_sha256': sha256(source.path),
                                  'mechanism': 'Original visual-only generated category targets and confusers.'}}
        if table_plan:
            patches.append({'operation': 'category_initialization', 'phase': 'before_construction',
                            'module': MODULE, 'parameters': initialization})
        for key in sorted(source_objects):
            patches.extend([{'operation': 'object_physics', 'binding': key, 'visual_only': False, 'fixed_base': False},
                            {'operation': 'object_physics', 'phase': 'after_postprocessing',
                             'binding': key, 'visual_only': False}])
        patches.append({'operation': 'category_initialization', 'phase': 'after_postprocessing',
                        'module': MODULE, 'parameters': initialization})
        category = QUESTION.fullmatch(raw['question'].strip())[1]
        if operation == 'inventory_return':
            objective = (f'Physically inventory the original generated {category} objects among the similar-looking '
                         'objects on their starting support. Identify each distinct item of the requested category, grasp it, lift it '
                         'at least 5 cm, then lower and release it near its starting pose. Repeat for every item of that '
                         'category and finish with an empty open gripper.')
            success = ('Every distinct requested-category item has been held by a real native grasp constraint and '
                       'lifted at least 5 cm above its initial lowest point, then released within 3 cm and 15 degrees '
                       'of its initial pose on its starting support. All items are stationary for 15 consecutive controller-step samples '
                       'and the gripper is empty and open.')
        else:
            objective = (f'Find any one original generated {category} object among the similar-looking original objects, '
                         'grasp it and hold it at least 5 cm above its starting height.')
            success = ('An original requested-category item is held by an active native grasp constraint, with its '
                       'lowest point at least 5 cm above its initial height for 15 consecutive controller-step samples.')
        public = {'objective': objective,
                  'success_criteria': success + ' Confusers stay within 2 cm and 15 degrees of their starting poses, '
                      'and no confuser or other task category may be grasped. Native movable noncritical furniture may be moved with guarded arm/gripper actions.',
                  'environment': 'The original room, object geometry and visually similar categories are retained. '
                      'Each original target/confuser group may be initialized together on a separate accessible support.',
                  'constraints': ['Inspect all candidates and distinguish the requested category before handling objects.',
                      'Use only robot-mounted observations and the existing guarded '
                      'robot motion and gripper controls.', 'Reject similar-looking categories and avoid disturbing '
                      'confusers.', 'Human review images, source maps, annotations and private evaluator state are not sensors.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
                      'binding_keys': list(bindings), 'parameters': {'target_category': category,
                      'target_keys': role_keys['targets'], 'confuser_keys': role_keys['confusers'],
                      'source_objects': source_objects, 'placement_groups': groups}}
        scene = original_scene(source, context, bindings=bindings, patches=patches)
        # The owned initialization hook settles both scene and task paths.
        scene['runtime'] = {'post_controller_settle_steps': 0}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings,
                             patches=patches, capabilities=['base_translation', 'base_rotation', 'ee_translation',
                             'ee_rotation', 'trunk_height', 'gripper'],
                             qualification={'on_failure': 'skip', 'initial_visibility': 'none'})
        task['defaults'].update(max_steps=240, wall_timeout_s=14400.)
        interaction = {'mode': 'manipulation', 'group_bindings': sorted(source_objects),
            'manipulation_bindings': role_keys['targets'], 'relocation_policy': 'preserve' if table_plan else 'independent_groups',
            'support_search': 'measured_surfaces', 'source_region': raw['room'],
            'allow_fallback_table': not bool(table_plan),
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
            'maximum_stance_distance_m': .85, 'warning_codes': ['native_grasp_ik_and_aperture_pending']}
        if not table_plan:
            interaction.update(placement_groups=groups, source_camera_policy='preserve')
        warnings = ['assumption_review_required', 'native_physical_execution_unverified', 'source_visual_only_restored']
        if start is None:
            warnings.extend(['source_static_start_unverified', 'native_spawn_repair_pending'])
        if not table_plan:
            warnings.extend(['independent_dependency_group_support_unverified', 'source_intergroup_visibility_adapted'])
        legacy_category = payload['question_data'].get('count_object', {}).get('original_requested_category')
        if legacy_category and legacy_category != category:
            warnings.append('legacy_requested_category_differs_from_resolved_public_category')
        if category == 'cauliflower':
            warnings.append('source_aabb_floor_penetration_risk')
        if table_plan:
            warnings.extend(['user_authorized_primitive_table', 'source_group_elevated_for_operation',
                             'support_rests_native_stability_unverified', 'adapted_visibility_requires_review'])
        assumptions = [{
            'doubt': 'Source objects are visual-only; saved AABBs and observer traversability do not certify stable '
                'collidable floor placement, body clearance or R1 floor-level reach.',
            'chosen_assumption': 'Enable collisions and gravity on all original generated targets/confusers without '
                'changing size or within-entry relative geometry. Place original target/confuser entry groups '
                'independently on measured supports; retain source observer and warn changed intergroup visibility. '
                'Only the explicitly reviewed source uses its historical primitive table and common '
                'vertical translation with rests under floating confusers. Keep strict native pose, contact and motion checks.',
            'alternatives': ['Retain an original source-only diagnostic for failed native starts.',
                             'Use the selective_lift recipe if full inventory is too costly after native reach evidence.',
                             'Retain source-only QA if no source-faithful supported physical layout qualifies.'],
            'scope': 'The adapted physical variant only; the empty original variant and source JSON remain intact.',
            'risk': 'Collision restoration can settle or displace objects and alter original visibility. Floor-level '
                'IK/grasp may fail; neither generated status nor screenshots establish physical qualification.'}, {
            'doubt': 'Saved observer points are camera/free-point evidence, not full R1-body clearance certificates.',
            'chosen_assumption': 'Choose a source observer projected 0.65 m inside the room bounds and at least '
                    '0.65 m from every generated AABB; a 0.4 m disk must be free in the original traversability map '
                    'and inside the source room segmentation. If no CPU proposal qualifies, omit robot_pose and require '
                    'the shared measured Native planner instead of inventing a pose. Preserve the source camera and qualify actual robot contacts '
                'and chassis room membership before execution.',
            'alternatives': ['Retain the original robot start if native evidence qualifies it.',
                             'Keep a source-only diagnostic if the selected stance fails native qualification.'],
            'scope': 'One deterministic initialization-only source-observer proposal, followed by native chassis screening '
                'around the complete group and any declared support repair.',
                'risk': 'The 2D disk does not model every extended R1 link, height-dependent obstacle or passage. '
                'Source flags, rectangle and map screens are private proposals and do not prove navigation feasibility.'}]
        if table_plan:
            assumptions.append({'doubt': 'The recorded bounded support search failed; source props penetrate the floor '
                'or float and collision restoration changed poses. Native contact on a new table/rest arrangement is untested.',
                'chosen_assumption': 'Apply the user-authorized small built-in primitive table only to the reviewed source. '
                'Elevate every target/confuser by the same vector and support floating confusers on nonintersecting rests. '
                'Keep the original reference camera unchanged; reject settling drift beyond existing tolerances.',
                'alternatives': ['Retry the expanded measured existing-support search without a table.',
                    'Retain the empty original variant as source-only QA/diagnostic evidence.'],
                'scope': TABLE_SOURCE + '; adapted physical variant only; feedback ' + FEEDBACK_REVISION,
                'risk': 'The table can affect furniture occlusion and access. Narrow rests may be unstable or block an approach. '
                'No new native capture, IK or grasp episode establishes this repair; source two-image equivalence is unclaimed.'})
        if legacy_category and legacy_category != category:
            assumptions.append({'doubt': 'Historical original_requested_category differs from the current question '
                'and rendered assets.', 'chosen_assumption': 'Use the current public category and matching resolved '
                'assets; preserve the historical field verbatim in private QA.',
                'alternatives': ['Review upstream category normalization independently; do not replace source assets.'],
                'scope': 'Only normalized source records with a differing original_requested_category.',
                'risk': 'The converted recipe retains current category ambiguity, not the historical pre-normalization question.'})
        return {'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
                'operation_recipe': operation, 'available_operation_recipes': list(OPERATIONS),
                'interaction_policy': interaction,
                'change_summary': ['Preserve source assets, dimensions, orientations and each target/confuser group geometry; retain the empty original variant.',
                    'Restore movable collidable physics before construction and after the source visual-only reset.',
                    'Use measured existing-surface whole-group placement; the reviewed source uses the explicitly authorized primitive table.',
                    'Add an evidenced initialization-only R1 pose and private native inventory/grasp/lift/release evaluation.',
                    'Settle once during owned initialization on both scene and task paths; preserve original QA privately.'],
                'warning_codes': warnings, 'assumptions': assumptions,
                'private_source_audit': {'robot_start': start, 'resolved_target_count': len(role_keys['targets']),
                    'saved_answer_matches_generated_count': str(raw.get('answer')) == str(len(role_keys['targets'])),
                    'legacy_original_requested_category': legacy_category,
                    'source_camera_policy_preserved_in_original': True,
                    'adapted_camera_may_translate_with_shared_existing_support_repair': not bool(table_plan),
                    'table_plan': table_plan, 'operation': operation},
                'limitations': ['CPU conversion and source annotations do not certify native support, body clearance, '
                    'visibility, floor-level IK, grasp or complete operation execution.',
                    'Native qualification is fail-closed on unsupported measurements or changed initial layout.',
                    'Inventory completion may require multiple navigation/manipulation decisions; 240 decisions and '
                    'a four-hour ceiling are unbenchmarked budgets, not runtime predictions.',
                    'Original QA is archived verbatim; the current shared runtime has no category-compatible initial-answer '
                    'freeze/submission protocol. No auxiliary count score or original two-image equivalence is claimed.',
                    'Historical bounded-control diagnostics were not run by this supervisor.'],
                'attempted_repairs': ['Source-observer robot proposal plus native chassis/height screening.',
                    'Whole-group measured existing-surface search, or reviewed-source primitive table/rest construction.',
                    'Collision restoration on both initialization phases and identical 30-step initialization settling.'],
                'next_step': 'Review all successes and diagnostics, qualify restored native support and chassis clearance, '
                    'then test original-scale grasp and inventory with existing guarded controls. Retain rejected repairs in Review.'}

    def convert_all(self, sources, context):
        """Reuse the shared idempotent per-source writer, including every failure."""
        sources = sorted(sources, key=lambda source: source.relative_path)
        previous = context.output_root / 'batch_manifest.json'
        if previous.is_file():
            archive = context.output_root / 'revision_history' / f'batch_before_{self.version}_{sha256(previous)[:16]}.json'
            if not archive.exists():
                import json
                write_json(archive, json.loads(previous.read_text()))
        manifest = super().convert_all(sources, context)
        manifest['feedback_revision'] = FEEDBACK_REVISION
        for row in manifest['records']:
            row['feedback_revision'] = FEEDBACK_REVISION
            write_json(Path(row['diagnostic_dir']) / 'record.json', row)
        manifest['failure_distribution'] = dict(Counter(row.get('failure_reason') or 'none'
            for row in manifest['records'] if row['status'] != 'generated'))
        manifest['sample_ids'] = self.sample(manifest, {'count': 5, 'seed': context.seed})
        manifest['physical_execution_verified'] = False
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Cover the rare asset, native-start failure and count/layout strata first."""
        policy = policy or {}
        count = policy.get('count', 5)
        selected, strata = [], set()
        previous = set(policy.get('previous_failures', ()))
        rows = sorted(manifest['records'], key=lambda row: (row['candidate_id'] not in previous,
            row['status'] == 'generated', 'source_aabb_floor_penetration_risk' not in row['warning_codes'],
            row['candidate_id']))
        # Reserve rare assets, failures and every generated count stratum before scene diversity.
        priorities = [row for row in rows if row['candidate_id'] in previous or row['status'] != 'generated' or
                      'source_aabb_floor_penetration_risk' in row['warning_codes']]
        priorities = priorities[:1] + [row for row in priorities[1:] if
            'source_aabb_floor_penetration_risk' in row['warning_codes'] or row['candidate_id'] in previous]
        for cardinality in (3, 4, 5):
            match = next((row for row in rows if row['status'] == 'generated' and
                row.get('private_source_audit', {}).get('resolved_target_count') == cardinality), None)
            if match is not None:
                priorities.append(match)
        for row in priorities:
            if len(selected) < count and row['candidate_id'] not in selected:
                selected.append(row['candidate_id'])
        for row in rows:
            audit = row.get('private_source_audit', {})
            stratum = (row['status'], row.get('failure_reason'), audit.get('resolved_target_count'),
                       audit.get('legacy_original_requested_category'), row.get('scene_model'))
            if row['candidate_id'] not in selected and len(selected) < count and (
                    stratum not in strata or row['candidate_id'] in previous):
                selected.append(row['candidate_id'])
                strata.add(stratum)
        for row in rows:
            if len(selected) >= min(count, len(rows)):
                break
            if row['candidate_id'] not in selected:
                selected.append(row['candidate_id'])
        return selected


class InventoryPlacementEvaluator:
    """Measure each original target's stable final destination and confuser history."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('Native inventory or destination binding unavailable.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def begin_episode(self, observation):
        self.active = True
        self.stability = dict.fromkeys(self.parameters['target_keys'], 0)
        self.initial_confusers = {key: tuple(_array(value).copy() for value in self.objects[key].get_position_orientation())
                                 for key in self.parameters['confuser_keys']}
        self.confusers_preserved, self.confuser_telemetry = True, True
        return self.record(observation, action=None)

    def _update(self, count_step=False):
        from .partial_occlusion import _placement_measurements
        rows = {}
        for key in self.parameters['target_keys']:
            measured = _placement_measurements(self.session, self.controller, self.config,
                                               self.objects[key], self.objects['destination'])
            available = not measured['measurement_errors']
            state = measured['gripper']
            released = available and state['holding'] is False and state['constraint_active'] is False and not state['release_pending']
            stable = bool(released and measured['destination_relation'] and measured['robot_withdrawn']
                and measured['linear_speed_m_s'] <= MAXIMUM_LINEAR_SPEED
                and measured['angular_speed_rad_s'] <= MAXIMUM_ANGULAR_SPEED)
            if not stable:
                self.stability[key] = 0
            elif count_step:
                self.stability[key] += 1
            rows[key] = {'measurement': measured, 'available': available,
                'completed': stable and self.stability[key] >= STABILITY_STEPS}
        try:
            gripper = self.controller.debug_gripper_state()
            for key, (start, quaternion) in self.initial_confusers.items():
                position, current_q = (_array(value) for value in self.objects[key].get_position_orientation())
                cosine = abs(float(np.dot(current_q, quaternion) / (np.linalg.norm(current_q) * np.linalg.norm(quaternion))))
                angle = math.degrees(2 * math.acos(float(np.clip(cosine, 0., 1.))))
                self.confusers_preserved &= bool(np.linalg.norm(position - start) <= CONFUSER_TOLERANCE_M
                    and angle <= ORIENTATION_TOLERANCE_DEG
                    and not (gripper.get('holding') is True and gripper.get('object_name') == self.objects[key].name))
        except (AttributeError, KeyError, ValueError, RuntimeError):
            self.confuser_telemetry = False
        return rows

    def sample_step(self):
        if self.active:
            self._update(count_step=True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        from wia.tasks.completion import progress
        rows = self._update()
        items = [{'id': 'placed:' + key, 'kind': 'terminal_condition', 'completed': row['completed'],
            'evaluation_validity': 'valid' if row['available'] else 'unavailable'} for key, row in rows.items()]
        result = {'schema': 'private_category_inventory_placement_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'completion_items': items,
            'task_success': bool(self.confusers_preserved and self.confuser_telemetry and all(
                item['completed'] and item['evaluation_validity'] == 'valid' for item in items)),
            'confusers_preserved': bool(self.confusers_preserved), 'confuser_measurements_available': self.confuser_telemetry,
            'native_placements': rows}
        if not self.confuser_telemetry:
            result['evaluation_validity'] = 'unavailable'
        result.update(progress(result))
        write_json(self.output / 'state.json', result)
        return result

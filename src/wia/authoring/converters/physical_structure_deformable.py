"""Convert original cloth covers into guarded sweep-and-retrieve operations.

Run CPU batching from the repository root using the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Native qualification/evaluation only read state; they never execute robot actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.tasks.background_policy import legal_background_grasp

import copy

from collections import Counter
from functools import lru_cache
import math
import os
from pathlib import Path

import numpy as np

from wia.authoring.conversion import ConverterBase, original_scene, physical_task
from wia.authoring.io import sha256

MODULE = 'wia.authoring.converters.physical_structure_deformable'
TASK_REVISION = 'task_edit_20261018_v1'
MECHANISM_CONTRACT = {'preserve': ['native_cloth_mesh', 'cloth_scale', 'covered_item_identity', 'initial_cover_relation'], 'goal': 'native_cloth_sweep_reveal_then_retrieve', 'rigid_proxy_allowed': False}
OPERATION = 'sweep_cloth_retrieve_item'
from wia.tasks.deformable_support import REVISION as DEFORMABLE_REVISION
LIFT_M = .05
CLOTH_SHIFT_M = .10
CLEARANCE_M = .02
HOLD_SECONDS = .25
MAP_RESOLUTION_M = .01
ROBOT_RADIUS_M = .40


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite source or native measurement.')
    return result


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        type(x) in (int, float) and math.isfinite(x) for x in value)


def _source(source):
    raw, payload = source.raw, source.payload
    if (source.big_task, source.small_task) != CategoryConverter.category or (
            raw.get('big_task'), raw.get('small_task')) != CategoryConverter.category:
        raise ValueError('source_category_mismatch')
    if tuple(Path(source.relative_path).parts[-3:-1]) != (raw.get('scene'), raw.get('room')):
        raise ValueError('source_hierarchy_mismatch')
    if payload.get('task_type') != 'cover_small_item_cloth' or not payload.get('floor_name'):
        raise ValueError('original_cover_mechanism_unresolved')
    item, cloth = payload.get('small_item', {}), payload.get('cloth', {})
    if any(not spec.get(key) for spec in (item, cloth) for key in ('category', 'model')):
        raise ValueError('original_asset_binding_unresolved')
    if type(payload.get('seed')) is not int or payload['seed'] < 0:
        raise ValueError('missing_source_construction_seed')
    bounds = item.get('bbox_after_cover', {})
    pose = item.get('pose_after_cover', {})
    if not _vector(item.get('placement_xy'), 2) or any(not _vector(bounds.get(k), 3) for k in ('min', 'max')):
        raise ValueError('missing_original_cover_geometry')
    if any(a >= b for a, b in zip(bounds['min'], bounds['max'])) or not _vector(pose.get('position'), 3):
        raise ValueError('invalid_original_item_geometry')
    if not _vector(pose.get('quaternion_xyzw'), 4) or abs(np.linalg.norm(pose['quaternion_xyzw']) - 1.) > .002:
        raise ValueError('invalid_original_item_quaternion')
    camera = payload.get('render', {}).get('main_view', {}).get('camera_pose', {})
    if not _vector(camera.get('position'), 3) or not _vector(camera.get('quaternion_xyzw'), 4):
        raise ValueError('missing_original_reference_camera')
    if abs(np.linalg.norm(camera['quaternion_xyzw']) - 1.) > .002:
        raise ValueError('invalid_original_reference_camera_quaternion:render.main_view.camera_pose.quaternion_xyzw')
    footprint = payload.get('camera_poses', {}).get('cloth_footprint_xy_m')
    if type(footprint) not in (int, float) or not math.isfinite(footprint) or footprint <= 0:
        raise ValueError('missing_original_cloth_footprint')
    # Answers validate preserved auxiliary QA; they never choose an operation or success.
    qa = payload.get('qa', {})
    if raw.get('question') != qa.get('question') or raw.get('answer') != qa.get('answer_text'):
        raise ValueError('original_qa_annotations_disagree')
    return raw, payload, item, cloth, camera


@lru_cache(maxsize=4)
def _maps(asset_root, scene):
    """Read original 1 cm maps on CPU, matching native map and room-ID conventions."""
    import cv2
    cv2.setNumThreads(2)
    layout = Path(asset_root) / 'scenes' / scene / 'layout'
    paths = [layout / name for name in ('floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    category_path = Path(asset_root) / 'metadata/room_categories.txt'
    if not all(p.is_file() for p in [*paths, category_path]):
        return None
    trav, rooms, semantics = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    if any(x is None for x in (trav, rooms, semantics)) or not (
            trav.shape == rooms.shape == semantics.shape and trav.shape[0] == trav.shape[1]):
        return None
    names, ids, counters = category_path.read_text().splitlines(), {}, Counter()
    for instance in np.unique(rooms):
        if instance == 0:
            continue
        row, col = np.argwhere(rooms == instance)[0]
        semantic = int(semantics[row, col])
        if not 0 < semantic <= len(names):
            continue
        label = names[semantic - 1]
        ids[f'{label}_{counters[label]}'] = int(instance)
        counters[label] += 1
    return trav, rooms, ids, {str(p): sha256(p) for p in [*paths, category_path]}


def _robot_start(source, context, payload, item, camera):
    data = Path(os.environ.get('OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior'))
    root = str(context.config.get('original_asset_root', data / 'behavior-1k-assets'))
    maps = _maps(root, source.raw['scene'])
    if maps is None or source.raw['room'] not in maps[2]:
        return None, {'reason': 'original_room_maps_unavailable', 'asset_root': root}
    trav, rooms, ids, hashes = maps
    centre = _array(item['placement_xy'])
    bounds = item['bbox_after_cover']
    radius = max(float(np.linalg.norm(_array(bounds['max'])[:2] - bounds['min'][:2]) / 2),
                 float(payload['camera_poses']['cloth_footprint_xy_m']) * math.sqrt(2) / 2)
    candidates = [('original_main_view_xy', _array(camera['position'])[:2])]
    for distance in (.70, .85, 1., 1.15, 1.30):
        for angle in range(0, 360, 15):
            radians = math.radians(angle)
            candidates.append((f'cover_ring_{distance}_{angle}', centre + distance * np.array(
                [math.cos(radians), math.sin(radians)])))
    pixel_radius = math.ceil(ROBOT_RADIUS_M / MAP_RESOLUTION_M)
    offsets = np.arange(-pixel_radius, pixel_radius + 1)
    disk = offsets[:, None] ** 2 + offsets[None, :] ** 2 <= pixel_radius ** 2
    valid = []
    for label, xy in candidates:
        if np.linalg.norm(xy - centre) < ROBOT_RADIUS_M + radius + .03:
            continue
        row, col = (xy[::-1] / MAP_RESOLUTION_M + trav.shape[0] / 2).astype(int)
        if min(row, col) < pixel_radius or max(row, col) + pixel_radius >= trav.shape[0]:
            continue
        region = (slice(row - pixel_radius, row + pixel_radius + 1),
                  slice(col - pixel_radius, col + pixel_radius + 1))
        if np.all(trav[region][disk] > 0) and np.all(rooms[region][disk] == ids[source.raw['room']]):
            valid.append((float(np.linalg.norm(xy - centre)), label, xy))
    evidence = {'map_sha256': hashes, 'original_room_instance_id': ids[source.raw['room']],
                'body_screen_radius_m': ROBOT_RADIUS_M, 'cover_exclusion_radius_m': radius,
                'tested_positions': len(candidates), 'passing_positions': len(valid),
                'provenance': 'Original floor_trav/insseg/semseg maps and room_categories; native BaseMap and '
                    'SegmentationMap conventions; source small_item.placement_xy/bbox_after_cover, '
                    'camera_poses.cloth_footprint_xy_m and render.main_view.camera_pose.',
                'native_reach_or_clearance_verified': False}
    if not valid:
        return None, {**evidence, 'reason': 'no_source_room_clearance_screened_r1_start'}
    _, label, xy = min(valid, key=lambda x: (round(x[0], 8), x[1]))
    # Source generator settles this item on the selected floor before covering.
    # Its rounded lower AABB is an approximate floor datum, checked natively later.
    floor_z = float(bounds['min'][2])
    evidence.update(selection=label, floor_height_estimate_m=floor_z,
                    floor_height_provenance='Rounded small_item.bbox_after_cover.min[2]; native floor check required.')
    return {'operation': 'robot_pose', 'position': [float(xy[0]), float(xy[1]), floor_z],
            'yaw_deg': math.degrees(math.atan2(centre[1] - xy[1], centre[0] - xy[0])),
            'heading_offsets_deg': [0.], 'jitter_xy_m': [0., 0.], 'head_pitch_deg': 45.}, evidence


def _configuration(session, config):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') != OPERATION or config.get('scene') != session.scene_name or
            config.get('room') != session.room_name or set(config.get('bindings', {})) != {'item', 'cloth', 'floor'}):
        raise ValueError('Cover operation requires the original scene and three explicit bindings.')
    return config['parameters']


def _mesh(cloth):
    points = _array(cloth.root_link.compute_particle_positions())
    faces = _array(cloth.root_link.faces).astype(int)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('Native cloth must expose a nonempty triangular particle mesh.')
    if np.min(faces) < 0 or np.max(faces) >= len(points):
        raise ValueError('Invalid native cloth face indices.')
    return points, points[faces]


def _overlap_count(triangles, low, high, margin=0.):
    """Exact 2D triangle/rectangle SAT, including degenerate edge projections.

    All deformed faces are inspected. A cloth AABB can overlap while the actual
    mesh is clear; neither a source label nor a particle-only test is sufficient.
    """
    xy = triangles[:, :, :2]
    low, high = _array(low)[:2] - margin, _array(high)[:2] + margin
    centre, half = (low + high) / 2, (high - low) / 2
    overlap = np.all(xy.max(axis=1) >= low, axis=1) & np.all(xy.min(axis=1) <= high, axis=1)
    edges = np.roll(xy, -1, axis=1) - xy
    for edge in range(3):
        normal = np.stack((-edges[:, edge, 1], edges[:, edge, 0]), axis=1)
        projections = np.einsum('fvi,fi->fv', xy - centre, normal)
        extent = np.sum(np.abs(normal) * half, axis=1)
        overlap &= (projections.min(axis=1) <= extent) & (projections.max(axis=1) >= -extent)
    return int(np.count_nonzero(overlap))


def _state(session, controller, objects, footprint=None, *, full_cloth_contacts=False):
    from omnigibson.object_states import ContactBodies, Touching
    from wia.tasks.interaction_setup import touching_task_support
    from omnigibson.macros import gm
    import omnigibson as og
    item, cloth, floor = [objects[k] for k in ('item', 'cloth', 'floor')]
    low, high = [_array(v) for v in item.aabb]
    position, quaternion = [_array(v) for v in item.get_position_orientation()]
    if abs(np.linalg.norm(quaternion) - 1.) > .002:
        raise ValueError('Native target quaternion is not normalized.')
    points, triangles = _mesh(cloth)
    robot = session.robot
    hand = {robot.eef_links[controller.working_arm], *robot.finger_links[controller.working_arm]}
    other_robot_links = set(robot.links.values()) - hand
    if full_cloth_contacts:
        from wia.tasks.deformable_support import native_cloth_contact_paths
        from wia.tasks.support_sampling import same_body
        paths = native_cloth_contact_paths(cloth, [item, *robot.links.values()], points=points)
        contacts = {link for link in robot.links.values()
            if any(same_body(link.prim_path, path) for path in paths)}
        item_touching_cloth = any(same_body(item.prim_path, path) for path in paths)
    else:
        contacts = set(cloth.states[ContactBodies].get_value())
        item_touching_cloth = bool(cloth.states[Touching].get_value(item))
    if footprint is None:
        footprint = (low, high)
    floor_top = float(_array(floor.aabb[1])[2])
    footprint_low, footprint_high = [_array(v) for v in robot.links[robot.base_footprint_link_name].aabb]
    rooms = [session.env.scene.seg_map.get_room_instance_by_point(np.array([x, y]))
             for x in (footprint_low[0], footprint_high[0]) for y in (footprint_low[1], footprint_high[1])]
    return {'time_s': float(og.sim.current_time), 'sim_step_dt_s': float(og.sim.get_sim_step_dt()),
            'item_position': position.tolist(),
            'item_aabb_min': low.tolist(), 'item_aabb_max': high.tolist(), 'item_scale': _array(item.scale).tolist(),
            'cloth_scale': _array(cloth.scale).tolist(), 'cloth_centroid': points.mean(axis=0).tolist(),
            'cloth_point_count': len(points), 'cloth_face_count': len(triangles),
            'cloth_min_z': float(points[:, 2].min()),
            'initial_region_overlap_faces': _overlap_count(triangles, *footprint, margin=CLEARANCE_M),
            'above_item_overlap_faces': _overlap_count(triangles[triangles[:, :, 2].max(axis=1) >= low[2]], low, high),
            'working_hand_cloth_contacts': sorted(link.prim_path for link in contacts & hand),
            'other_robot_cloth_contacts': sorted(link.prim_path for link in contacts & other_robot_links),
            'item_touching_cloth': item_touching_cloth,
            'full_cloth_particle_contact_query': full_cloth_contacts,
            'item_touching_floor': bool(item.states[Touching].get_value(floor)),
            'item_touching_task_support': touching_task_support(session, item, floor),
            'item_linear_speed_m_s': float(np.linalg.norm(_array(item.get_linear_velocity()))),
            'floor_top_m': floor_top, 'robot_in_original_room': all(r == session.room_name for r in rooms),
            'rigid_assisted_grasp_mode': robot.grasping_mode == 'assisted' and gm.AG_CLOTH is False,
            'gripper': controller.debug_gripper_state()}


def qualify_operation(session, executor, config, bound):
    """Only measure native original cover, dynamics and controller preconditions."""
    checks = {}
    try:
        from wia.tasks.interaction_setup import reference_position
        p = _configuration(session, config)
        from omnigibson.utils.constants import PrimType
        item, cloth, floor = [bound[k] for k in ('item', 'cloth', 'floor')]
        contact_options = {'full_cloth_contacts': True} if p.get('cover_revision') == 'full_cover_redrape_v1' else {}
        row = _state(session, executor, bound, **contact_options)
        g = row['gripper']
        checks = {'original_item_movable_collidable': not item.visual_only and not item.fixed_base,
                  'original_cloth_movable_collidable': not cloth.visual_only and not cloth.fixed_base,
                  'original_cloth_native_type': cloth.prim_type == PrimType.CLOTH,
                  'original_floor_collidable': not floor.visual_only,
                  'original_floor_height_matches_spawn_datum': abs(row['floor_top_m'] - p['floor_height_estimate_m']) <= .02,
                  'original_item_xy_preserved': np.linalg.norm(_array(row['item_position'])[:2] -
                      reference_position(session, item.name, [*p['source_item_xy'], 0.])[:2]) <= .05,
                  'item_supported_by_task_surface': row['item_touching_task_support'],
                  'native_original_cover_present': row['above_item_overlap_faces'] > 0 and row['item_touching_cloth'],
                  'initial_target_stationary': row['item_linear_speed_m_s'] <= .02,
                  'initial_gripper_empty': g.get('object_name') is None and g.get('constraint_active') is False,
                  'initial_hand_not_touching_cloth': not row['working_hand_cloth_contacts'],
                  'initial_other_robot_links_not_touching_cloth': not row['other_robot_cloth_contacts'],
                  'robot_footprint_in_original_room': row['robot_in_original_room'],
                  'existing_native_rigid_assisted_mode': row['rigid_assisted_grasp_mode']}
        checks = {k: bool(v) for k, v in checks.items()}
        placement = None
        if p.get('task_revision') == DEFORMABLE_REVISION:
            placement = _qualified_deformable_placement(session, item, cloth, p)
            checks['native_deformable_cover_placement'] = placement.get('available') is True and placement.get('accepted') is True
        accepted = all(checks.values())
        if config['parameters'].get('cover_revision') == 'full_cover_redrape_v1':
            from .deformable_redrape import initial_cover_evidence
            extra = [session.env.scene.object_registry('name', key) for key in
                config['parameters'].get('protected_extra_keys', [])]
            full_cover = initial_cover_evidence(session, bound['cloth'], [bound['item'], *extra])
            if extra:
                # A physical cloth can bridge between covered items. Direct
                # contact with the selected item is not the definition of cover.
                checks['native_original_cover_present'] = row['above_item_overlap_faces'] > 0
            checks['sampled_footprint_and_mounted_visual_occlusion'] = full_cover['accepted']
            row['initial_full_cover_evidence'] = full_cover
            accepted = all(checks.values())
        return {'accepted': bool(accepted), 'reason': None if accepted else 'native_cover_preconditions_failed',
                'checks': checks, 'native_state': row, 'real_sweep_or_retrieval_verified': False,
                **({'native_deformable_placement': placement} if placement is not None else {})}
    except Exception as error:
        return {'accepted': False, 'reason': 'native_cover_measurement_unavailable', 'checks': checks,
                'exception_summary': f'{type(error).__name__}: {error}', 'real_sweep_or_retrieval_verified': False}


class SweepRetrieveEvaluator:
    """Track hand-driven reveal before a real target grasp and sustained lift."""
    def __init__(self, session, controller, config, output):
        self.parameters = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = {key: session.env.scene.object_registry('name', name)
                        for key, name in config['bindings'].items()}
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.initial = 0, None

    def begin_episode(self, observation):
        from wia.runtime.prechecks import enabled
        if enabled(self.session):
            qualification = qualify_operation(self.session, self.controller, self.config, self.objects)
            if not qualification['accepted']:
                raise ValueError('Native cover preconditions failed at episode start: ' + str(qualification))
            initial = qualification['native_state']
        else:
            initial = _state(self.session, self.controller, self.objects,
                full_cloth_contacts=self.parameters.get('cover_revision') == 'full_cover_redrape_v1')
        self.episode += 1
        self.initial = initial
        self.hand_contact_seen, self.revealed_before_grasp, self.invalid = False, False, False
        self.measurement_error = None
        self.last_time = self.initial['time_s']
        self.hold_start, self.hold_samples, self.criteria = None, 0, {}
        self.record(observation, action=None)

    def _update(self):
        contact_options = {'full_cloth_contacts': True} if self.parameters.get('cover_revision') == 'full_cover_redrape_v1' else {}
        row = _state(self.session, self.controller, self.objects,
                     (self.initial['item_aabb_min'], self.initial['item_aabb_max']), **contact_options)
        g, initial = row['gripper'], self.initial
        held = (g.get('holding') is True and g.get('constraint_active') is True and
                g.get('object_name') == self.objects['item'].name and g.get('release_pending') is False)
        now = row['time_s']
        if not math.isfinite(now) or now < self.last_time - 1e-8:
            self.invalid = True
        new_sample = now > self.last_time + 1e-8
        dt = row['sim_step_dt_s']
        if not math.isfinite(dt) or dt <= 0:
            self.invalid = True
        consecutive = new_sample and 0 < now - self.last_time <= dt * 1.5 + 1e-7
        shift = float(np.linalg.norm(_array(row['cloth_centroid'])[:2] - initial['cloth_centroid'][:2]))
        lift = min(float(row['item_aabb_min'][2] - initial['item_aabb_min'][2]),
                   float(row['item_position'][2] - initial['item_position'][2]))
        item_still = (np.linalg.norm(_array(row['item_position'])[:2] - initial['item_position'][:2]) <= .03 and
                      abs(row['item_aabb_min'][2] - initial['item_aabb_min'][2]) <= .015 and row['item_touching_task_support'])
        cloth_clear = row['initial_region_overlap_faces'] == 0 and shift >= CLOTH_SHIFT_M
        if new_sample:
            if row['other_robot_cloth_contacts']:
                self.invalid = True
            self.hand_contact_seen |= bool(row['working_hand_cloth_contacts'])
            if held and not self.revealed_before_grasp:
                self.invalid = True
            if self.hand_contact_seen and cloth_clear and item_still and g.get('object_name') is None:
                self.revealed_before_grasp = True
            if (g.get('object_name') not in (None, self.objects['item'].name)
                    and not legal_background_grasp(self.session, g.get('object_name'))):
                self.invalid = True
        unchanged = (row['item_scale'] == initial['item_scale'] and row['cloth_scale'] == initial['cloth_scale'] and
                     row['cloth_point_count'] == initial['cloth_point_count'] and
                     row['cloth_face_count'] == initial['cloth_face_count'] and
                     abs(row['floor_top_m'] - initial['floor_top_m']) <= .001 and
                     row['cloth_min_z'] >= row['floor_top_m'] - .03 and
                     all(not self.objects[key].visual_only and not self.objects[key].fixed_base for key in ('item', 'cloth')))
        self.criteria = {'working_hand_contact_seen': self.hand_contact_seen,
                         'native_cloth_moved_10cm_and_mesh_clear': cloth_clear,
                         'reveal_preceded_target_grasp': self.revealed_before_grasp,
                         'real_target_grasp_constraint_active': held, 'target_lift_5cm': lift >= LIFT_M,
                         'target_clear_of_cloth_and_task_surface': not row['item_touching_cloth'] and not row['item_touching_task_support'],
                         'original_physics_geometry_and_native_mode': unchanged and row['rigid_assisted_grasp_mode'],
                         'robot_remains_in_original_room': row['robot_in_original_room'],
                         'valid_operation_history': not self.invalid}
        valid = all(self.criteria.values())
        if not valid:
            self.hold_start, self.hold_samples = None, 0
        elif new_sample:
            if not consecutive:
                self.hold_start, self.hold_samples = None, 0
            if self.hold_start is None:
                self.hold_start = now
            self.hold_samples += 1
        self.last_time = max(now, self.last_time)
        self.criteria['sustained_native_hold'] = bool(valid and self.hold_start is not None and
            now - self.hold_start >= HOLD_SECONDS and self.hold_samples >= 3)
        return row, lift, shift

    def sample_step(self):
        if self.initial is not None:
            try:
                self._update()
            except Exception as error:
                self.invalid = True
                self.measurement_error = f'{type(error).__name__}: {error}'
                self.hold_start, self.hold_samples = None, 0

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        from wia.authoring.io import write_json
        try:
            row, lift, shift = self._update()
            value = {'schema': 'private_deformable_sweep_retrieve_v1', 'episode': self.episode,
                     'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                     'action': action, 'criteria': self.criteria, 'task_success': bool(all(self.criteria.values())),
                     'target_lift_m': lift, 'cloth_centroid_xy_shift_m': shift,
                     'hold_native_samples': self.hold_samples, 'initial_native_state': self.initial,
                     'native_state': row, 'configuration': self.config}
            value.update(evaluation_validity='unavailable' if self.measurement_error else 'complete',
                measurement_error=self.measurement_error, task_revision=DEFORMABLE_REVISION if
                    self.parameters.get('task_revision') == DEFORMABLE_REVISION else TASK_REVISION,
                completion_items=[{'id': 'native_cloth_reveal', 'completed': self.revealed_before_grasp,
                    'kind': 'checkpoint', 'evaluation_validity': 'valid' if self.revealed_before_grasp or not self.measurement_error else 'unavailable'},
                    {'id': 'native_retrieval', 'completed': bool(self.criteria['sustained_native_hold']),
                     'kind': 'terminal_condition', 'evaluation_validity': 'unavailable' if self.measurement_error else 'valid'}])
            if self.measurement_error:
                value['task_success'] = False
        except Exception as error:
            self.invalid = True
            self.measurement_error = f'{type(error).__name__}: {error}'
            self.hold_start, self.hold_samples = None, 0
            value = {'schema': 'private_deformable_sweep_retrieve_v1', 'episode': self.episode,
                     'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                     'action': action, 'task_success': False, 'measurement_unavailable': True,
                     'evaluation_validity': 'unavailable', 'measurement_error': str(error),
                     'exception_summary': f'{type(error).__name__}: {error}'}
        write_json(self.output / 'state.json', value)
        import json
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return SweepRetrieveEvaluator(session, controller, config, output)


class CategoryConverter(ConverterBase):
    category = ('Physical Structure', 'Deformable')
    version = 'task_edit_20261018_v1_deformable-sweep-retrieve-v4_accessible_task_repair_v2'

    def inspect(self, source):
        try:
            _source(source)
            return {'eligible': True}
        except Exception as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                    'warning_codes': ['not_convertible_under_L1_L2', 'source_only_diagnostic']}

    def convert(self, source, context):
        raw, payload, item, cloth, camera = _source(source)
        if context.config.get('task_edit2_deformable') is True:
            return _convert_deformable_support(self, source, context)
        seed = payload['seed']
        bindings = {'item': {'name': f'cover_small_item_render_item_{seed:010d}',
                             'category': item['category'], 'model': item['model']},
                    'cloth': {'name': f'cover_small_item_render_cloth_{seed:010d}',
                              'category': cloth['category'], 'model': cloth['model']},
                    'floor': {'name': payload['floor_name'], 'category': 'floors'}}
        robot, evidence = _robot_start(source, context, payload, item, camera)
        assumptions = [
            {'doubt': 'Native guarded floor reach and hand friction sufficient to sweep cloth are unmeasured.',
             'chosen_assumption': 'Move the original covered item and cloth together onto a measured accessible '
                 'support during initialization, preserving their native cover relation and mesh geometry. '
                 'Require physical hand-driven reveal before a real target grasp and lift.',
             'alternatives': ['Contact-qualified native cloth attachment repair', 'Evidence-based common tabletop relocation',
                              'Retain a source-only diagnostic if native controls cannot reach the floor'],
             'scope': 'All original cloth/item pairs; no source metadata is an execution success flag.',
             'risk': 'The hand may not reach or move the cloth; task remains pending and qualification may fail.'},
            {'doubt': 'Static room maps and rounded item lower AABB do not certify R1 body clearance or floor height.',
             'chosen_assumption': 'Use a map-screened deterministic stance and approximate floor datum; require native checks.',
             'alternatives': ['Measured native floor/body clearance and revised source-derived stance', 'Source-only diagnostic'],
             'scope': 'R1 initialization only; source asset scales and the cloth/item cover relation remain unchanged.',
             'risk': 'A source elevation discrepancy or unsaved obstacle may block native robot initialization.'}]
        limitations = [
            'CPU conversion and source maps do not verify IK, hand contact, cloth motion, grasp, sensors or terminal success.',
            'Shared ESISession configures required robot observation before applying add_existing_r1; '
                'the integrator must fix the order before robot-free task execution. Scene-only capture uses require_robot=False.',
            'The installed injected R1 template uses sticky grasping; native qualification requires assisted mode. '
                'The integrator must apply the existing requested gripper/observation settings after robot addition.',
            'Keep the source stop/play lifecycle and 240/60/60 Hz settings; historical cloth/camera failures remain unverified here.',
            'The recipe uses guarded contact sweeps and rigid-object acquisition; it never enables global AG_CLOTH, '
                'creates a cloth attachment, teleports objects or removes cloth.']
        warnings = ['assumption_review_required', 'native_sweep_and_grasp_unverified',
                    'shared_robot_initialization_order_requires_integration', 'injected_r1_assisted_mode_requires_integration']
        result = {'status': 'generated', 'reuse_level': 'L2', 'qa_preserved': True,
                  'warning_codes': warnings, 'assumptions': assumptions, 'limitations': limitations,
                  'private_source_evidence': {'robot_start': evidence, 'cloth_asset': cloth,
                      'source_item_bounds': item['bbox_after_cover'], 'original_reference_camera': camera,
                      'source_builder': 'ESI-Bench/src/active_explore/tasks/physical_structure/deformable.py'},
                  'native_loading_diagnostics': {'source_build_env_objects_returns': [],
                      'dynamic_creation_hook': 'postprocess_env',
                      'item_pose_field': 'metadata_json.small_item.pose_after_cover.quaternion_xyzw',
                      'camera_pose_field': 'metadata_json.render.main_view.camera_pose.quaternion_xyzw',
                      'required_runtime_order': 'Add/configure R1 before required robot observation validation; create native cloth in source postprocessing.',
                      'separate_failure_classes': ['invalid_source_quaternion', 'invalid_native_quaternion', 'empty_rgb', 'native_process_exit'],
                      'rigid_proxy_used': False, 'native_loading_verified': False},
                  'change_summary': ['Preserve the source native drop/cover and private identity QA; '
                      'add an independent hand sweep, reveal-before-grasp and held-lift goal.',
                      'Add installed R1 before a deterministic source/map-evidenced pose; enable original item/cloth '
                      'collision physics after dynamic source construction without resizing either asset.',
                      'Request one common cloth/item translation onto measured accessible support; '
                      'verify native mesh coverage and contact again after settling.'],
                  'next_step': 'Publish all candidates and diagnostics for Review; integrator resolves R1 settings order '
                      'before native qualification and an end-to-end guarded sweep/retrieval pilot.'}
        if robot is None:
            assumptions[1]['chosen_assumption'] = ('Keep a pending task without a guessed robot pose; native '
                'measurement must decide the original-room floor start independently of support placement.')
            warnings.append('source_start_native_measurement_required')
        evidence.setdefault('floor_height_estimate_m', float(item['bbox_after_cover']['min'][2]))
        patches = [{'operation': 'add_existing_r1'}, *([robot] if robot is not None else []),
                   *[{'operation': 'object_physics', 'phase': 'after_postprocessing', 'binding': key,
                      'visual_only': False} for key in ('item', 'cloth')]]
        public = {'objective': 'Use the working gripper to gently sweep the original cloth sideways until the '
                      'covered item is exposed. Keep the item in its starting place while uncovering it. Then '
                      'grasp the exposed item, lift it clear of its starting support and cloth, and hold it steadily.',
                  'success_criteria': 'Make physical hand contact with the cloth, move it at least 10 cm sideways '
                      'and entirely clear the original item footprint with 2 cm margin before grasping the item. '
                      'Hold the item with a real native grasp at least 5 cm above its starting height for 0.25 seconds.',
                  'environment': 'The original cloth and covered item start together on an accessible support '
                      'in the original room; asset sizes, floor and room furniture are retained.',
                  'constraints': ['Use only existing guarded base, arm, trunk and gripper controls.',
                                  'Push or sweep with small hand motions; never remove, teleport or resize the cloth.',
                                  'Keep the original room assets and floor geometry unchanged.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
                      'binding_keys': ['item', 'cloth', 'floor'],
                      'parameters': {'source_item_xy': item['placement_xy'],
                                     'floor_height_estimate_m': evidence['floor_height_estimate_m']}}
        evaluation['parameters'].update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        result.update(task_revision=TASK_REVISION, mechanism_contract=MECHANISM_CONTRACT)
        result.update(scene_definition=original_scene(source, context, bindings=bindings, patches=patches),
                      task_definition=physical_task(source, context, public=public, evaluation=evaluation,
                          bindings=bindings, patches=patches, capabilities=['base_translation', 'base_rotation',
                              'ee_translation', 'ee_rotation', 'trunk_height', 'gripper']))
        result['interaction_policy'] = {
            'mode': 'manipulation', 'group_bindings': ['item', 'cloth'],
            'manipulation_bindings': ['item'], 'relocation_policy': 'rigid_group',
            'source_camera_policy': 'preserve', 'source_region': raw.get('room'),
            'support_search': 'measured_surfaces', 'allow_fallback_table': True,
            'minimum_grasp_height_m': .35, 'maximum_grasp_height_m': 1.10,
            'maximum_stance_distance_m': .85,
            'warning_codes': ['covered_pair_accessible_support_requires_review',
                              'native_cloth_cover_after_translation_unverified',
                              'source_qa_reference_layout_changed'],
        }
        return result

    def convert_all(self, sources, context):
        """Retain shared deterministic candidate identity and a result for every source."""
        return super().convert_all(sorted(sources, key=lambda s: s.relative_path), context)

    def sample(self, manifest, policy=None):
        """Cover failures, footprint extremes, cloth models and room/scene variation."""
        policy = policy or {}
        rows = manifest['records']
        count = min(policy.get('count', 5), len(rows))
        base = super().sample(manifest, {**policy, 'count': len(rows)})
        by_id = {r['candidate_id']: r for r in rows}
        chosen = [key for key in base if by_id[key]['status'] != 'generated'][:1]
        chosen.extend(key for key in base if key in policy.get('previous_failures', ()) and key not in chosen)
        generated = [r for r in rows if r['status'] == 'generated']
        if generated:
            footprint = lambda r: r['private_source_evidence']['robot_start']['cover_exclusion_radius_m']
            for row in (min(generated, key=footprint), max(generated, key=footprint)):
                if row['candidate_id'] not in chosen:
                    chosen.append(row['candidate_id'])
        strata = set()
        for key in chosen:
            row = by_id[key]
            strata.add((row['scene_model'], row.get('private_source_evidence', {}).get('cloth_asset', {}).get('model')))
        for key in base:
            row = by_id[key]
            stratum = (row['scene_model'], row.get('private_source_evidence', {}).get('cloth_asset', {}).get('model'))
            if key not in chosen and stratum not in strata:
                chosen.append(key)
                strata.add(stratum)
        return (chosen + [key for key in base if key not in chosen])[:count]


def _qualified_deformable_placement(session, item, cloth, parameters):
    from wia.tasks.deformable_support import measure_deformable_placement
    setup = getattr(session, 'category_interaction_setup', {})
    if parameters.get('cover_revision') == 'full_cover_redrape_v1':
        inventory = getattr(session, '_redesigned_cloth_inventory', {})
        receipt = getattr(session, '_redesigned_cloth_cover', {})
        loaded = getattr(session, 'loaded_task', None)
        if (loaded is None or loaded.source_sha256 != parameters['source_sha256'] or
                inventory.get('cloth_name') != cloth.name or item.name not in inventory.get('item_names', []) or
                receipt.get('revision') != 'native_cloth_redrape_20261020_v1'):
            return {'available': False, 'accepted': False, 'reason': 'native_redrape_inventory_unavailable'}
        loaded.verify_unchanged()
        from .deformable_redrape import initial_cover_evidence
        items = [session.env.scene.object_registry('name', name) for name in inventory['item_names']]
        measured = initial_cover_evidence(session, cloth, items)
        return {'available': True, 'accepted': receipt.get('accepted') is True and measured['accepted'],
            'layout_owner': receipt['revision'], 'native_cover': measured, 'relocated': False}
    receipt = setup.get('deformable_cover', {})
    if (setup.get('accepted') is not True or setup.get('relocated') is not True or
            receipt.get('context', {}).get('source_identity') != parameters['source_identity'] or
            receipt.get('context', {}).get('source_sha256') != parameters['source_sha256']):
        return {'available': False, 'accepted': False, 'reason': 'exact_native_deformable_setup_receipt_unavailable'}
    delta = _array(setup.get('group_translation_m'))
    transforms = setup.get('object_translations_m', {})
    if delta.shape != (3,) or any(name not in transforms or not np.allclose(_array(transforms[name]), delta,
            atol=1e-5, rtol=0) for name in (item.name, cloth.name)):
        return {'available': False, 'accepted': False, 'reason': 'item_and_cloth_common_translation_unverified'}
    support = session.env.scene.object_registry('name', setup.get('support_name'))
    if support is None or support.prim_path != setup.get('support_prim_path'):
        return {'available': False, 'accepted': False, 'reason': 'selected_native_deformable_support_unavailable'}
    return measure_deformable_placement(session, item, cloth, support, receipt, delta, setup.get('support_surface_z_m'))


def _convert_deformable_support(converter, source, context):
    legacy_context = copy.copy(context)
    legacy_context.config = {**context.config, 'task_edit2_deformable': False}
    result = converter.convert(source, legacy_context)
    task = result['task_definition']
    parameters = task['private']['evaluation']['parameters']
    bindings = task['private']['bindings']
    contract = {'schema': DEFORMABLE_REVISION, 'source_identity': source.relative_path,
        'source_sha256': sha256(source.path), 'bindings': {key: copy.deepcopy(bindings[key]) for key in ('item', 'cloth')},
        'world_mesh_before_physics_required': True, 'settled_cloth_root_is_rigid_invariant': False,
        'native_mesh_cover_and_support_required': True, 'rigid_proxy_allowed': False,
        'actual_sweep_grasp_and_lift_pending': True}
    parameters.update(task_revision=DEFORMABLE_REVISION, source_identity=source.relative_path,
        source_sha256=sha256(source.path), deformable_cover=contract)
    policy = result['interaction_policy']
    policy.update(group_bindings=['item', 'cloth'], manipulation_bindings=['item'], support_bindings=['item'],
        relocation_policy='rigid_group', layout_parameters={'deformable_cover': contract})
    policy['warning_codes'].extend(['native_full_cloth_footprint_required', 'native_mesh_translation_and_settled_cover_pending'])
    result.update(task_revision=DEFORMABLE_REVISION, deformable_cover=contract)
    result['change_summary'].append('Verify exact pre-step world-particle translation, then permit real cloth deformation '
        'while preserving topology/physics and requiring actual covered-item/support contact after settling.')
    result['limitations'].append('Native cloth positions, topology, material, cover, settling and full support footprint '
        'must qualify before any actual guarded sweep/grasp/lift trial; CPU generation is not execution evidence.')
    return result

"""Convert original contact pairs into pending, measured robot operations.

Run on CPU from the repository root with the behavior environment's Python:
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' \
python -m wia.authoring.converters.spatial_relations_physical_contact --batch-root PATH
The default selects one seeded final relation. Explicit historical cycle recipes
remain available. Native hooks only read state; guarded controls perform actions.
"""
from __future__ import annotations

from wia.authoring.modules import module_identity

from wia.resources import ESI_ROOT

from wia.tasks.background_policy import legal_background_contact, legal_background_grasp
from wia.tasks.interaction_setup import array as _array

from collections import Counter
from functools import lru_cache
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np

from wia.authoring.conversion import (
    ROOT, ConversionContext, ConverterBase, SourceInstance, map_records,
    original_scene, physical_task,
)
from wia.authoring.io import sha256, write_json

MODULE = 'wia.authoring.converters.spatial_relations_physical_contact'
OBJECTS = ('obj1', 'obj2')
OPERATIONS = ('separate_then_contact', 'contact_then_separate')
FINAL_OPERATION = 'establish_final_relation'
TASK_REVISION = 'task_edit_20261018_v1'
FINAL_RELATIONS = ('touching', 'separated')
# Provisional solver tolerance; Core contact traces must calibrate it natively.
PENETRATION_TOLERANCE_M = .001
LIFT_M = .05
GAP_M = .05
POSE_TOLERANCE_M = .03
ANGLE_TOLERANCE_DEG = 15.
LINEAR_SPEED_M_S = .02
ANGULAR_SPEED_RAD_S = .1
STABILITY_STEPS = 15
MAP_RESOLUTION_M = .01


def seeded_final_relation(source_identity, seed):
    """Choose independently of source answer or initial contact geometry."""
    token = hashlib.sha256(f'{seed}\0{source_identity}\0final_contact_relation_v1'.encode()).digest()
    return FINAL_RELATIONS[int.from_bytes(token[:8], 'big') % len(FINAL_RELATIONS)]


def _vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


@lru_cache(maxsize=1)
def _source_builder():
    """Load the original pure constructor without importing OmniGibson."""
    path = ESI_ROOT / 'src/active_explore/tasks/spatial_relations/physical_contact.py'
    spec = importlib.util.spec_from_file_location('_esi_original_physical_contact', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_data(source):
    raw, payload = source.raw, source.payload
    category = CategoryConverter.category
    if (source.big_task, source.small_task) != category or (raw.get('big_task'), raw.get('small_task')) != category:
        raise ValueError('category_mismatch')
    if tuple(Path(source.relative_path).parts[-3:-1]) != (raw.get('scene'), raw.get('room')) or any(
            payload.get(key) is not None and payload[key] != raw.get(key) for key in ('scene', 'room')):
        raise ValueError('source_scene_room_mismatch')
    if not raw.get('question') or raw.get('answer') not in ('yes', 'no') or payload.get('layout') != 'touching_pair':
        raise ValueError('missing_original_contact_question_or_pair')
    objects, boxes = map_records(payload.get('objects')), map_records(payload.get('obj_aabb'))
    if set(objects) != set(OBJECTS) or not payload.get('floor_name'):
        raise ValueError('missing_original_pair_or_floor')
    poses = payload.get('camera_poses')
    poses = list(poses.values()) if isinstance(poses, dict) else poses
    if not poses or any(not _vector(pose.get('position'), 3) or not _vector(
            pose.get('quaternion_xyzw'), 4) or not any(pose['quaternion_xyzw']) for pose in poses):
        raise ValueError('missing_original_reference_camera')
    for key in OBJECTS:
        item, box = objects[key], boxes.get(key + '_aabb', {})
        if not all(isinstance(item.get(field), str) and item[field] for field in ('category', 'model')) or not (
                _vector(item.get('position'), 3) and _vector(item.get('quaternion_xyzw'), 4) and any(
                item['quaternion_xyzw']) and _vector(box.get('min'), 3) and _vector(box.get('max'), 3)) or any(
                box['max'][i] <= box['min'][i] for i in range(3)):
            raise ValueError('incomplete_original_object_geometry')
        # Original roots may differ from mesh centers. A root farther than two
        # complete saved bbox diagonals (at least 25 cm) from its own box is not
        # credible evidence for a source-preserving contact operation.
        diagonal = float(np.linalg.norm(_array(box['max']) - _array(box['min'])))
        root_box_distance = float(np.linalg.norm(_array(item['position']) - np.clip(
            _array(item['position']), box['min'], box['max'])))
        if root_box_distance > max(.25, 2. * diagonal):
            raise ValueError('source_root_and_saved_aabb_diverged')
    specs = _source_builder().build_env_objects(payload)
    if len({spec['name'] for spec in specs} | {payload['floor_name']}) != 3:
        raise ValueError('ambiguous_original_native_names')
    return raw, payload, objects, boxes, poses, specs


@lru_cache(maxsize=2)
def _room_map(asset_root, scene_name, room_name):
    """Use installed BaseMap/SegmentationMap conventions on original 1 cm maps."""
    import cv2
    cv2.setNumThreads(2)
    layout = Path(asset_root) / 'scenes' / scene_name / 'layout'
    paths = [layout / name for name in ('floor_trav_0.png', 'floor_insseg_0.png', 'floor_semseg_0.png')]
    categories = Path(asset_root) / 'metadata/room_categories.txt'
    if not all(path.is_file() for path in [*paths, categories]):
        return None
    trav, rooms, semantics = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if any(value is None for value in (trav, rooms, semantics)) or not (
            trav.shape == rooms.shape == semantics.shape and trav.shape[0] == trav.shape[1]):
        return None
    names, counts, room_ids = categories.read_text().splitlines(), Counter(), {}
    instances, first_pixels = np.unique(rooms, return_index=True)
    for instance, first in zip(instances, first_pixels):
        if instance == 0:
            continue
        semantic = int(semantics.flat[first])
        if not 0 < semantic <= len(names):
            continue
        name = names[semantic - 1]
        room_ids[f'{name}_{counts[name]}'] = int(instance)
        counts[name] += 1
    if room_name not in room_ids:
        return None
    free = ((trav > 0) & (rooms == room_ids[room_name])).astype(np.uint8)
    clearance = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * MAP_RESOLUTION_M
    return clearance, room_ids[room_name], {str(path): sha256(path) for path in [*paths, categories]}


def _robot_start(raw, boxes, poses, context):
    """Propose a map-screened stance near an original camera; no reach claim."""
    asset_root = context.config.get('original_asset_root', str(Path(os.environ.get(
        'OMNIGIBSON_DATA_PATH', '/local_home2/zhouxinyang/data/behavior')) / 'behavior-1k-assets'))
    maps = _room_map(str(asset_root), raw['scene'], raw['room'])
    evidence = {'map_asset_root': str(asset_root), 'map_resolution_m': MAP_RESOLUTION_M,
        'minimum_map_clearance_m': .42, 'minimum_pair_aabb_clearance_m': .65,
        'maximum_pair_midpoint_distance_m': 2.5, 'native_qualified': False}
    if maps is None:
        return None, {**evidence, 'failure_reason': 'original_room_maps_unavailable'}
    clearance, room_id, hashes = maps
    centre = np.mean([(np.array(boxes[key + '_aabb']['min']) + np.array(
        boxes[key + '_aabb']['max']))[:2] / 2 for key in OBJECTS], axis=0)
    size = clearance.shape[0]
    centre_rc = (centre[::-1] / MAP_RESOLUTION_M + size / 2).astype(int)
    low, high = np.maximum(centre_rc - 252, 0), np.minimum(centre_rc + 253, size)
    if np.any(low >= high):
        return None, {**evidence, 'failure_reason': 'source_pair_outside_original_maps'}
    rr, cc = np.mgrid[low[0]:high[0], low[1]:high[1]]
    xy = np.stack((cc - size / 2, rr - size / 2), axis=-1) * MAP_RESOLUTION_M
    valid = clearance[low[0]:high[0], low[1]:high[1]] >= .42
    valid &= np.linalg.norm(xy - centre, axis=-1) <= 2.5
    for key in OBJECTS:
        box = boxes[key + '_aabb']
        valid &= np.linalg.norm(xy - np.clip(xy, box['min'][:2], box['max'][:2]), axis=-1) >= .65
    candidates = xy[valid]
    evidence.update(source_room_id=room_id, map_sha256=hashes, safe_start_pixel_count=len(candidates),
        source_pair_midpoint_xy_m=centre.tolist(), source_floor_z_assumption_m=0.,
        provenance='Original floor_trav/insseg/semseg maps, saved pair AABBs and first original camera. '
            'Native R1 uses a floor-level root; source generator cameras use absolute 0.025/0.40 m heights.')
    if not len(candidates):
        return None, {**evidence, 'failure_reason': 'no_screened_original_room_stance'}
    observer = np.asarray(poses[0]['position'][:2])
    # Stable row-major tie breaking; do not change objects or the reference camera.
    chosen = int(np.argmin(np.linalg.norm(candidates - observer, axis=1)))
    start = candidates[chosen]
    rc = (start[::-1] / MAP_RESOLUTION_M + size / 2).round().astype(int)
    evidence.update(original_reference_camera=poses[0], selected_map_clearance_m=float(clearance[tuple(rc)]),
        selected_pair_aabb_clearance_m=min(float(np.linalg.norm(start - np.clip(start,
            boxes[key + '_aabb']['min'][:2], boxes[key + '_aabb']['max'][:2]))) for key in OBJECTS),
        observer_xy_shift_m=float(np.linalg.norm(start - observer)),
        stance_selection='Nearest admissible pixel to the first original observer XY in a bounded pair neighborhood.')
    return {'position': [float(start[0]), float(start[1]), 0.],
        'yaw_deg': math.degrees(math.atan2(centre[1] - start[1], centre[0] - start[0])),
        'head_pitch_deg': 45.}, evidence


def _source_audit(raw, objects, boxes):
    labels = [objects[key]['category'].replace('_', ' ') for key in OBJECTS]
    widths = [min(boxes['obj1_aabb']['max'][i], boxes['obj2_aabb']['max'][i]) - max(
        boxes['obj1_aabb']['min'][i], boxes['obj2_aabb']['min'][i]) for i in range(3)]
    return {'question_asset_labels_match': raw['question'].strip() ==
            f'Are the {labels[0]} and {labels[1]} touching each other?',
        'source_qa_answer': raw['answer'], 'actual_asset_categories': labels,
        'source_aabb_overlap_extents_m': widths, 'source_aabbs_overlap': min(widths) > 0,
        'source_aabb_gap_lower_bound_m': float(np.linalg.norm(np.maximum(-np.array(widths), 0.))),
        'source_dimensions_m': {key: (_array(boxes[key + '_aabb']['max']) - _array(
            boxes[key + '_aabb']['min'])).tolist() for key in OBJECTS},
        'source_aabb_minimum_z_m': {key: boxes[key + '_aabb']['min'][2] for key in OBJECTS},
        'mask_adjacency_is_not_native_contact': True, 'initial_native_pair_contact': None}


class CategoryConverter(ConverterBase):
    category = ('Spatial Relations', 'Physical Contact')
    version = '1.6.0_task_edit_20261018_v1'

    def inspect(self, source):
        try:
            _source_data(source)
            return {'eligible': True}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            return {'eligible': False, 'reason': 'not_convertible_under_L1_L2:' + str(error),
                'warning_codes': ['not_convertible_under_L1_L2', 'source_only_diagnostic']}

    def convert(self, source, context):
        eligibility = self.inspect(source)
        if not eligibility['eligible']:
            return {'scene_definition': original_scene(source, context), 'task_definition': None,
                'status': 'conversion_failed', 'failure_stage': 'source_inspection',
                'failure_reason': eligibility['reason'], 'warning_codes': eligibility['warning_codes'],
                'diagnostic_source_only': True}
        operation = context.config.get('operation', FINAL_OPERATION)
        if operation not in (*OPERATIONS, FINAL_OPERATION):
            raise ValueError('Use establish_final_relation or an explicit historical contact cycle.')
        desired_relation = seeded_final_relation(source.relative_path, context.seed)
        raw, payload, objects, boxes, poses, specs = _source_data(source)
        audit = _source_audit(raw, objects, boxes)
        robot, screen = _robot_start(raw, boxes, poses, context)
        screen.setdefault('source_pair_midpoint_xy_m', np.mean([
            (_array(boxes[key + '_aabb']['min']) + _array(boxes[key + '_aabb']['max']))[:2] / 2
            for key in OBJECTS], axis=0).tolist())
        assumptions = [{
            'doubt': 'Saved mask adjacency and AABB overlap do not establish native pair contact or stable original floor support.',
            'chosen_assumption': 'Retain the exact pair geometry, asset sizes and quaternions, enable native physics, '
                'and request a common translation onto an existing same-room support when its measured fit permits. '
                'Do not scale root spacing or resize assets. Qualify translated poses, dimensions, live upward '
                'task-support contact, stationarity and robot source-floor contact/clearance. '
                'Accept either initial native contact state; the separate physical recipe must create both states through real grasp/lift/release.',
            'alternatives': ['Keep a source-only diagnostic if native settling fails.',
                'Retain the historical original variant when the unchanged pair cannot fit a valid support.'],
            'scope': 'Every adapted original contact pair.',
            'risk': 'Restored source poses can be penetrating, unsupported or outside the working-arm reach. '
                'Native physics, grasp, separation and recontact remain unverified. A valid height window '
                'does not certify grasp width, reach, IK or acquisition.'}, {
            'doubt': 'Original free cameras and 2D maps do not certify full R1 clearance, floor height or IK reach.',
            'chosen_assumption': 'Use the nearest original-room pixel to the first source observer with a 0.42 m map '
                'clearance, 0.65 m pair-AABB clearance and at most 2.5 m from the pair midpoint. '
                'Retain a floor-level root and independently require native source-floor wheel contacts and room membership.',
            'alternatives': ['Qualify the unchanged native robot start.',
                'Retain the source diagnostic when this bounded screen or native body qualification fails.'],
            'scope': 'One reversible initialization-only robot stance per adapted variant.',
            'risk': 'Static maps omit height-dependent obstacles, extended arms and closed-door changes; '
                'a free pixel does not prove navigation or manipulation.'}]
        warnings = ['assumption_review_required', 'native_physical_execution_unverified',
            'source_mask_adjacency_not_native_contact', 'rigid_pair_support_translation_requires_review']
        if audit['source_aabbs_overlap']:
            warnings.append('source_pair_aabb_overlap_requires_native_qualification')
        if any(abs(value) > .035 for value in audit['source_aabb_minimum_z_m'].values()):
            warnings.append('source_floor_height_requires_native_confirmation')
        if not audit['question_asset_labels_match']:
            warnings.append('source_question_asset_category_mismatch')
            assumptions.append({'doubt': 'The saved question labels differ from the actual original category/model assets.',
                'chosen_assumption': 'Preserve all source bytes and auxiliary QA verbatim; use the actually loaded original '
                    'asset categories for the independent contact operation, without treating the saved answer as native truth.',
                'alternatives': ['Keep only a source diagnostic if a reviewer rejects the semantic correspondence.',
                    'Resolve upstream normalization separately without changing this preserved source.'],
                'scope': source.relative_path,
                'risk': 'The initial QA may refer to absent objects; this pending operation cannot certify original QA validity.'})
        if robot is None:
            warnings.append('source_start_native_measurement_required')
            assumptions[1]['chosen_assumption'] = ('Retain a pending physical task with no guessed robot pose. '
                'The CPU map screen is inconclusive; require independently measured native floor support, '
                'body clearance and source-room membership before accepting initialization.')
        # Bounding volume is a deterministic effort heuristic, never a graspability proof.
        target = min(OBJECTS, key=lambda key: (float(np.prod(audit['source_dimensions_m'][key])), key))
        anchor = next(key for key in OBJECTS if key != target)
        bindings = {key: {field: spec[field] for field in ('name', 'category', 'model')}
            for key, spec in zip(OBJECTS, specs)}
        bindings['source_floor'] = {'name': payload['floor_name']}
        source_objects = {key: {'position': objects[key]['position'],
            'quaternion_xyzw': objects[key]['quaternion_xyzw'],
            'aabb_min': boxes[key + '_aabb']['min'], 'aabb_max': boxes[key + '_aabb']['max']}
            for key in OBJECTS}
        patches = [{'operation': 'robot_pose', **robot}] if robot is not None else []
        for key in OBJECTS:
            patches.extend([{'operation': 'object_physics', 'binding': key, 'visual_only': False},
                {'operation': 'object_physics', 'phase': 'after_postprocessing', 'binding': key, 'visual_only': False}])
        target_label, anchor_label = [objects[key]['category'].replace('_', ' ') for key in (target, anchor)]
        relations = ('separate', 'contact') if operation == OPERATIONS[0] else ('contact', 'separate')
        first = 'separated from' if relations[0] == 'separate' else 'touching'
        second = 'touching' if relations[1] == 'contact' else 'separated from'
        public = {'objective': f'Inspect the original {target_label} and {anchor_label} before manipulation. '
            f'Grasp the {target_label}, lift it at least 5 cm, and place and release it on its starting support surface {first} '
            f'the {anchor_label}. Wait for a stable released state. Grasp and lift the same object again, then place '
            f'and release it {second} the {anchor_label}. Keep the {anchor_label} near its starting pose and finish '
            'with an empty open gripper.',
            'success_criteria': 'Both ordered phases require a new native grasp and 5 cm lift above the object\'s '
                'initial lowest point, followed by an empty, open, fully released gripper. Contact requires actual '
                'native pair contact. Separation requires no native contact and at least a 5 cm conservative '
                'bounding-box gap. At each phase both objects touch and rest on their starting support surface, remain in the '
                'source room near the original pair, and have speeds at most 0.02 m/s and 0.1 rad/s for 15 consecutive '
                'native samples after consecutive control steps. The other object stays within 3 cm and 15 degrees '
                'of its starting pose at every measured state.',
            'environment': 'The original room, floor, two original-size assets, pair relative geometry and reference '
                'camera are retained. Initialization may translate both partners together onto an existing same-room '
                'support without resizing or changing their spacing. The original floor layout remains a historical '
                'variant. Record the initial contact assessment before moving anything as auxiliary QA.',
            'constraints': ['Use robot-mounted sensors and existing guarded base, arm, trunk and gripper controls.',
                'Manipulate the designated task object while preserving its partner and task supports. Noncritical native movable furniture may be moved using guarded arm and gripper actions.',
                'Keep the manipulated object within 1.5 m horizontally of the episode-start pair midpoint; place it on the starting support beside its partner.',
                'Initial QA and saved visual labels do not prove either physical phase.',
                'Human review images, source maps, object names/poses and evaluator records remain private, outside Agent sensors.']}
        evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': operation,
            'binding_keys': [*OBJECTS, 'source_floor'], 'parameters': {'target': target, 'anchor': anchor,
                'source_objects': source_objects, 'source_pair_midpoint_xy_m': screen['source_pair_midpoint_xy_m']}}
        if operation == FINAL_OPERATION:
            relation_words = 'touching' if desired_relation == 'touching' else 'separated by a positive gap from'
            public.update(objective=f'Inspect the {target_label} and {anchor_label}. Finish with the '
                f'{target_label} {relation_words} the {anchor_label}, with both objects stably resting on '
                'their starting support and the gripper empty and open. If this relation already holds, '
                'leave the objects in place and wait for the measured stable state. Otherwise manipulate '
                f'the {target_label} using the robot controls. Keep the {anchor_label} near its starting pose.',
                success_criteria='Touching requires actual native pair contact without penetration deeper '
                'than the declared 1 mm solver tolerance. Separation requires no native pair contact and '
                'a strictly positive measured bounding-box clearance. Both objects must remain supported, '
                'within the source room and original pair neighborhood, with speeds at most 0.02 m/s and '
                '0.1 rad/s for 15 consecutive native samples. Finish released with an empty open gripper. '
                'The partner remains within 3 cm and 15 degrees of its starting pose throughout. '
                'No grasp, lift, movement or contact/separation cycle is required when already correct.')
            public['constraints'][3] = 'Initial QA and saved visual labels do not prove the final native relation.'
            evaluation['parameters'].update(desired_relation=desired_relation,
                penetration_tolerance_m=PENETRATION_TOLERANCE_M,
                final_relation_selection={'seed': context.seed, 'source_identity': source.relative_path,
                    'algorithm': 'sha256_seed_source_final_contact_relation_v1',
                    'independent_of_initial_relation': True})
            evaluation['parameters'].update(task_revision=TASK_REVISION, completion_items=[
                {'id': 'desired_final_relation', 'description': f'Establish the {desired_relation} relation.'},
                {'id': 'stable_released_support', 'description': 'Both objects settle supported and released.'}])
            assumptions[0]['chosen_assumption'] = ('Retain the exact pair geometry, asset sizes and quaternions. '
                'Allow only a qualified common support translation. Independently seed the desired final '
                'relation and accept an already-correct relation after measured stable release; no action cycle is required.')
        scene = original_scene(source, context, bindings=bindings, patches=patches)
        scene['runtime'] = {'post_controller_settle_steps': 30}
        task = physical_task(source, context, public=public, evaluation=evaluation, bindings=bindings, patches=patches,
            capabilities=['base_translation', 'base_rotation', 'ee_translation', 'ee_rotation', 'trunk_height', 'gripper'],
            qualification={'on_failure': 'skip', 'initial_visibility': 'none'})
        task['defaults'].update(max_steps=240, wall_timeout_s=14400.)
        if operation == FINAL_OPERATION:
            contract = {'preserve': ['target_identity', 'asset_scale', 'pair_initial_relative_geometry',
                'anchor_pose', 'source_reference_camera', 'stable_support'],
                'goal': 'seeded_final_native_contact_relation', 'desired_relation': desired_relation,
                'already_satisfied_no_op_allowed': True, 'mandatory_action_history': False}
            task['repair'] = {'task_revision': TASK_REVISION, 'mechanism_contract': contract}
            evaluation['parameters']['mechanism_contract'] = contract
        audit.update(target_binding=target, anchor_binding=anchor,
            target_selection='Smallest saved original AABB volume, with stable binding-key tie breaking.')
        return {'scene_definition': scene, 'task_definition': task, 'status': 'generated', 'reuse_level': 'L2',
            'task_revision': TASK_REVISION if operation == FINAL_OPERATION else 'historical_contact_cycle',
            'mechanism_contract': task.get('repair', {}).get('mechanism_contract'),
            'operation_recipe': operation, 'available_operation_recipes': (
                [FINAL_OPERATION, *OPERATIONS] if operation == FINAL_OPERATION else list(OPERATIONS)),
            'private_source_audit': audit, 'robot_start_evidence': screen,
            'warning_codes': warnings, 'assumptions': assumptions,
            'change_summary': ['Retain pair relative geometry, room, asset sizes, quaternions, source answers and '
                'reference camera; preserve source poses in the historical original variant.',
                'Request one common translation of both partners onto a measured existing same-room support, '
                'without scaling positions or assets.',
                'Declare a reversible map/AABB-screened R1 stance and explicit two-phase object physics enabling.',
                ('Use one seeded final relation with measured stable-release no-op support.' if operation == FINAL_OPERATION
                    else 'Retain ordered native grasp/lift and task-supported released separation/contact predicates.'),
                'Keep source QA auxiliary and private; neither its answer nor rendered mask adjacency proves success.'],
            'limitations': ['CPU generation and original-map screens do not prove native load, reach, grasp or contact feasibility.',
                'AABB overlap is a warning, not a mesh-contact verdict; the separated AABB gap is sufficient but conservative.',
                'The smaller-volume target may still be heavy, deformable, too wide or impossible to grasp at a valid height.',
                'The original active pipeline already includes R1; no robot injection or omitted source mechanism restoration is needed.',
                'Current postprocessing resets source poses, not visual_only. Both declared physics phases guard against later resets.',
                'Operation budgets are unbenchmarked ceilings, not measured completion times.'],
            'attempted_repairs': ['Initialization-only robot stance, object_physics and a requested measured '
                'rigid-pair support translation; no scale change or new asset.'],
            'next_step': 'Review preselected candidates; qualify native pair geometry, actual support, robot floor '
                'contact and clearance. Validate the declared final relation or explicit historical recipe through '
                'native measurements and guarded controls before claiming physical success.'}

    def convert_all(self, sources, context):
        """Use the shared immutable per-source writer and retain every failure."""
        sources = sorted(sources, key=lambda source: source.relative_path)
        manifest = super().convert_all(sources, context)
        for row in manifest['records']:
            if row.get('failure_reason') == 'not_convertible_under_L1_L2:source_root_and_saved_aabb_diverged':
                row.update(diagnostic_source_only=True,
                    limitations=['The saved root is farther than two bbox diagonals (minimum 0.25 m) from its own saved AABB. '
                        'These conflicting source measurements cannot define a trusted unchanged contact pair.'],
                    next_step='Review the preserved source root/AABB contradiction and original-only diagnostic; '
                        'do not guess a corrected object pose or replace its asset.')
                write_json(Path(row['diagnostic_dir']) / 'record.json', row)
        manifest['failure_distribution'] = dict(Counter(row.get('failure_reason') or 'unspecified'
            for row in manifest['records'] if row['status'] != 'generated'))
        manifest['sample_ids'] = self.sample(manifest, {'count': 5, 'seed': context.seed})
        manifest['available_operation_recipes'] = [FINAL_OPERATION, *OPERATIONS]
        manifest['physical_execution_verified'] = False
        write_json(context.output_root / 'batch_manifest.json', manifest)
        return manifest

    def sample(self, manifest, policy=None):
        """Cover failed records, source label conflicts, QA labels and gap strata."""
        policy = policy or {}
        rows = sorted(manifest['records'], key=lambda row: row['candidate_id'])
        random.Random(policy.get('seed', manifest.get('seed', 0))).shuffle(rows)
        count = min(policy.get('count', 5), len(rows))
        selected, scenes, covered = [], set(), set()
        def stratum(row):
            audit = row.get('private_source_audit', {})
            return (row['status'], audit.get('source_qa_answer'), audit.get('question_asset_labels_match'),
                audit.get('source_aabbs_overlap'))
        def add(row):
            if row and row['candidate_id'] not in selected and len(selected) < count:
                selected.append(row['candidate_id'])
                scenes.add(row.get('scene_model'))
                covered.add(stratum(row))
        previous = set(policy.get('previous_failures', ()))
        for row in rows:
            if row['candidate_id'] in previous:
                add(row)
        add(next((row for row in rows if row['status'] != 'generated'), None))
        while len(selected) < count:
            candidates = [row for row in rows if row['candidate_id'] not in selected]
            if not candidates:
                break
            candidates.sort(key=lambda row: (stratum(row) in covered, row.get('scene_model') in scenes))
            add(candidates[0])
        return selected


def _snapshot(obj):
    """Read current native geometry, physical flags and velocity; never set state."""
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    low, high = (_array(value) for value in obj.aabb)
    linear, angular = (_array(value) for value in (obj.get_linear_velocity(), obj.get_angular_velocity()))
    if any(value.shape != (3,) for value in (position, low, high, linear, angular)) or (
            quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0 or np.any(high <= low)):
        raise ValueError('Invalid measured native object state.')
    return {'name': obj.name, 'category': obj.category, 'model': obj.model,
        'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
        'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
        'linear_speed_m_s': float(np.linalg.norm(linear)), 'angular_speed_rad_s': float(np.linalg.norm(angular))}


def _near(row, original):
    a, b = (_array(value['quaternion_xyzw']) for value in (row, original))
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    angle = math.degrees(2. * math.acos(min(1., dot)))
    return math.dist(row['position'], original['position']) <= POSE_TOLERANCE_M and angle <= ANGLE_TOLERANCE_DEG


def _empty_released(gripper):
    return (gripper.get('object_name') is None and gripper.get('holding') is False and
        gripper.get('constraint_active') is False and gripper.get('gripper_open') is True and
        gripper.get('release_pending') is False)


def _contact_geometry(objects, native_contact, gap, tolerance=PENETRATION_TOLERANCE_M):
    """Read PhysX separation; AABB overlap alone never establishes penetration."""
    if not native_contact and gap > 0.:
        return {'available': True, 'relation': 'separated', 'penetrating': False,
            'minimum_separation_m': gap, 'separation_source': 'positive_aabb_clearance',
            'penetration_tolerance_m': tolerance}
    if not native_contact:
        return {'available': False, 'relation': 'unresolved', 'penetrating': None,
            'reason': 'No native contact and no strictly positive conservative clearance.'}
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI
        first, second = (objects[key] for key in OBJECTS)
        first_paths = [str(link.prim_path) for link in first.links.values()]
        second_paths = [str(link.prim_path) for link in second.links.values()]
        contacts = RigidContactAPI.get_contact_data(first.scene.idx,
            row_prim_paths=first_paths, column_prim_paths=second_paths)
        separations = [float(_array(row[5]).reshape(-1)[0]) for row in contacts]
        if not separations:
            raise ValueError('Native contact has no pair-specific separation samples.')
        minimum = min(separations)
        penetrating = minimum < -tolerance
        return {'available': True, 'relation': 'penetrating' if penetrating else 'touching',
            'penetrating': penetrating, 'minimum_separation_m': minimum,
            'separations_m': separations, 'separation_source': 'native_physx_contact_data',
            'penetration_tolerance_m': tolerance}
    except Exception as error:
        return {'available': False, 'relation': 'unresolved', 'penetrating': None,
            'reason': f'{type(error).__name__}: {error}'}


def _pair_measure(objects, floor, session):
    from omnigibson.object_states import Touching, OnTop
    from wia.tasks.interaction_setup import touching_task_support, on_top_task_support
    rows = {key: _snapshot(objects[key]) for key in OBJECTS}
    widths = np.minimum(rows['obj1']['aabb_max'], rows['obj2']['aabb_max']) - np.maximum(
        rows['obj1']['aabb_min'], rows['obj2']['aabb_min'])
    contact = bool(objects['obj1'].states[Touching].get_value(objects['obj2']))
    support = {key: {'touching_floor': bool(obj.states[Touching].get_value(floor)),
        'on_top_floor': bool(obj.states[OnTop].get_value(floor))} for key, obj in objects.items()}
    task_support = {key: {'touching_task_support': touching_task_support(session, obj, floor),
        'on_top_task_support': on_top_task_support(session, obj, floor)} for key, obj in objects.items()}
    gap = float(np.linalg.norm(np.maximum(-widths, 0.)))
    return {'objects': rows, 'native_pair_contact': contact, 'aabb_gap_lower_bound_m': gap,
        'contact_geometry': _contact_geometry(objects, contact, gap),
        'separate': not contact and gap >= GAP_M, 'contact': contact, 'native_floor_support': support,
        'floor_supported': all(all(row.values()) for row in support.values()),
        'native_task_support': task_support,
        'task_supported': all(all(row.values()) for row in task_support.values()),
        'stationary': all(row['linear_speed_m_s'] <= LINEAR_SPEED_M_S and row['angular_speed_rad_s'] <=
            ANGULAR_SPEED_RAD_S for row in rows.values())}


def _owns(body, root):
    return body == root or body.startswith(root + '/')


def _robot_state(session, floor):
    """Require actual wheel support on this floor and reject other external contacts."""
    robot, scene = session.robot, session.env.scene
    low, high = (_array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    rooms = [scene.seg_map.get_room_instance_by_point(np.array([x, y]))
        for x in (low[0], (low[0] + high[0]) / 2, high[0])
        for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    wheels = {str(robot.links[name].prim_path) for name in robot.floor_touching_base_link_names}
    supported, forbidden, floor_contacts = set(), [], []
    for contact in robot.contact_list():
        if legal_background_contact(session, contact.body0, contact.body1):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        internal = [_owns(body, str(robot.prim_path)) for body in bodies]
        if all(internal):
            continue
        external = [body for body, inside in zip(bodies, internal) if not inside]
        wheel = next((path for path in sorted(wheels) if any(_owns(body, path) for body in bodies)), None)
        position, normal = _array(contact.position), _array(contact.normal)
        row = {'bodies': bodies, 'position': position.tolist(), 'normal': normal.tolist()}
        if any(internal) and wheel and external and all(_owns(body, str(floor.prim_path)) for body in external) and (
                position.shape == normal.shape == (3,) and abs(normal[2]) >= .5):
            supported.add(wheel)
            floor_contacts.append(row)
        else:
            forbidden.append(row)
    position, quaternion = (_array(value) for value in robot.get_position_orientation())
    quaternion = quaternion / np.linalg.norm(quaternion)
    upright = bool(1 - 2 * (quaternion[0] ** 2 + quaternion[1] ** 2) >= .9)
    return {'position': position.tolist(), 'footprint_rooms': rooms,
        'in_source_room': all(room == session.room_name for room in rooms),
        'source_floor_wheel_contacts': floor_contacts, 'supported_wheels': sorted(supported),
        'other_external_contacts': forbidden, 'supported_upright': len(supported) >= 2 and upright and not forbidden}


def _configuration(session, config, bound=None):
    if (config.get('type') != 'category_operation' or module_identity(config.get('module')) != MODULE or
            config.get('operation') not in (*OPERATIONS, FINAL_OPERATION) or tuple(config.get('binding_keys', ())) != (*OBJECTS, 'source_floor') or
            (config.get('scene'), config.get('room')) != (session.scene_name, session.room_name)):
        raise ValueError('Native contact operation configuration mismatch.')
    p = config['parameters']
    if config['operation'] == FINAL_OPERATION and p.get('desired_relation') not in FINAL_RELATIONS:
        raise ValueError('A seeded final contact relation is required.')
    if {p['target'], p['anchor']} != set(OBJECTS) or set(p['source_objects']) != set(OBJECTS) or (
            set(config['bindings']) != {*OBJECTS, 'source_floor'} or len(set(config['bindings'].values())) != 3):
        raise ValueError('Original contact pair bindings must be complete and unique.')
    if bound is None:
        bound = {key: session.env.scene.object_registry('name', config['bindings'][key]) for key in config['binding_keys']}
    if any(bound.get(key) is None or bound[key].name != config['bindings'][key] for key in config['binding_keys']):
        raise ValueError('Missing or mismatched original native binding.')
    return bound


def qualify_operation(session, executor, config, bound):
    """Only measure native preconditions. Do not step, move, grasp or consult QA."""
    checks = {}
    try:
        from wia.tasks.interaction_setup import reference_position
        bound = _configuration(session, config, bound)
        objects, floor = {key: bound[key] for key in OBJECTS}, bound['source_floor']
        measured, robot = _pair_measure(objects, floor, session), _robot_state(session, floor)
        rows, originals = measured['objects'], config['parameters']['source_objects']
        translations = [reference_position(session, objects[key].name, originals[key]['position']) -
                        _array(originals[key]['position']) for key in OBJECTS]
        checks = {'original_floor_collidable': floor.category in ('floor', 'floors') and not floor.visual_only,
            'reference_pair_common_translation': bool(np.allclose(translations[0], translations[1], rtol=0., atol=1e-8)),
            'pair_movable_collidable': all(not row['visual_only'] and not row['fixed_base'] for row in rows.values()),
            'source_initial_poses_preserved': all(_near(row, {**originals[key],
                'position': reference_position(session, objects[key].name, originals[key]['position'])})
                for key, row in rows.items()),
            'source_dimensions_preserved': all(np.allclose(_array(row['aabb_max']) - _array(row['aabb_min']),
                _array(originals[key]['aabb_max']) - _array(originals[key]['aabb_min']), rtol=.1, atol=.005)
                for key, row in rows.items()),
            'native_task_support': measured['task_supported'], 'pair_stationary': measured['stationary'],
            'initial_gripper_empty_released': _empty_released(executor.debug_gripper_state()),
            'robot_in_original_room': robot['in_source_room'], 'robot_on_source_floor_and_clear': robot['supported_upright']}
        accepted = bool(all(checks.values()))
        return {'accepted': accepted, 'checks': checks, 'native_pair': measured, 'native_robot': robot,
            'reason': None if accepted else 'native_contact_pair_initialization_failed',
            'physical_operation_execution_verified': False}
    except Exception as error:
        return {'accepted': False, 'checks': checks, 'reason': 'native_contact_pair_measurement_unavailable',
            'exception_summary': f'{type(error).__name__}: {error}', 'physical_operation_execution_verified': False}


class ContactCycleEvaluator:
    """Two ordered released physical states, each preceded by native held lift."""
    def __init__(self, session, controller, config, output):
        self.bound = _configuration(session, config)
        self.objects = {key: self.bound[key] for key in OBJECTS}
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.phases = ('separate', 'contact') if config['operation'] == OPERATIONS[0] else ('contact', 'separate')
        self.episode, self.initial = 0, None

    def begin_episode(self, observation):
        self.episode += 1
        self.initial_qualification = qualify_operation(self.session, self.controller, self.config, self.bound)
        self.initial = _pair_measure(self.objects, self.bound['source_floor'], self.session)
        self.phase_index, self.stable_steps, self.native_control_samples = 0, 0, 0
        self.phase_lift_verified, self.completed_phases = False, []
        self.phase_lift_evidence = None
        self.wrong_grasp, self.anchor_disturbed, self.measurement_failed = False, False, False
        self.record(observation, action=None)

    def _update(self, *, count_step=False):
        if self.initial is None:
            return None
        try:
            from wia.tasks.interaction_setup import reference_position
            measured = _pair_measure(self.objects, self.bound['source_floor'], self.session)
            robot = _robot_state(self.session, self.bound['source_floor'])
            gripper = self.controller.debug_gripper_state()
            rows, p = measured['objects'], self.config['parameters']
            target, anchor = p['target'], p['anchor']
            held = (gripper.get('holding') is True and gripper.get('constraint_active') is True and
                gripper.get('object_name') == self.objects[target].name)
            lift = rows[target]['aabb_min'][2] - self.initial['objects'][target]['aabb_min'][2]
            self.wrong_grasp |= (gripper.get('object_name') not in (None, self.objects[target].name)
                and not legal_background_grasp(self.session, gripper.get('object_name')))
            self.anchor_disturbed |= not _near(rows[anchor], self.initial['objects'][anchor])
            if held and lift >= LIFT_M:
                self.phase_lift_verified = True
                if self.phase_lift_evidence is None:
                    self.phase_lift_evidence = {'native_control_sample': self.native_control_samples + int(count_step),
                        'target_lift_m': float(lift), 'native_target': rows[target],
                        'native_gripper': gripper}
            midpoint = reference_position(self.session, self.objects[anchor].name,
                                          [*p['source_pair_midpoint_xy_m'], 0.])[:2]
            near_pair = all(np.linalg.norm(_array(row['position'])[:2] - _array(midpoint)) <= 1.5
                for row in rows.values())
            object_rooms = [self.session.env.scene.seg_map.get_room_instance_by_point(_array(row['position'])[:2])
                for row in rows.values()]
            checks = {'initial_native_qualification': self.initial_qualification['accepted'],
                'no_wrong_grasp': not self.wrong_grasp, 'anchor_preserved_throughout': not self.anchor_disturbed,
                'native_held_lift_for_current_phase': self.phase_lift_verified,
                'empty_open_released_gripper': _empty_released(gripper),
                'declared_native_relation': measured[self.phases[self.phase_index]],
                'pair_collidable_movable': all(not row['visual_only'] and not row['fixed_base'] for row in rows.values()),
                'native_task_support': measured['task_supported'], 'pair_stationary': measured['stationary'],
                'pair_in_original_neighborhood': bool(near_pair),
                'pair_in_source_room': all(room == self.session.room_name for room in object_rooms),
                'robot_in_source_room': robot['in_source_room'], 'robot_source_floor_and_clear': robot['supported_upright'],
                'native_measurements_available_throughout': not self.measurement_failed}
            valid = bool(all(checks.values()))
            phase_before = self.phase_index
            if count_step:
                self.native_control_samples += 1
                self.stable_steps = self.stable_steps + 1 if valid else 0
                if self.phase_index == 0 and self.stable_steps >= STABILITY_STEPS:
                    self.completed_phases.append({'relation': self.phases[0], 'native_control_sample': self.native_control_samples,
                        'native_pair': measured, 'native_gripper': gripper,
                        'held_lift_evidence': self.phase_lift_evidence})
                    self.phase_index, self.stable_steps, self.phase_lift_verified = 1, 0, False
                    self.phase_lift_evidence = None
            elif not valid:
                self.stable_steps = 0
            success = bool(valid and phase_before == 1 and self.completed_phases and self.stable_steps >= STABILITY_STEPS)
            return {'native_pair': measured, 'native_robot': robot, 'native_gripper': gripper,
                'target_lift_m': float(lift), 'criteria': checks, 'measured_phase': phase_before,
                'current_phase': self.phase_index, 'current_relation': self.phases[self.phase_index],
                'completed_phases': list(self.completed_phases), 'native_control_samples': self.native_control_samples,
                'current_phase_held_lift_evidence': self.phase_lift_evidence,
                'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS, 'task_success': success}
        except Exception as error:
            self.measurement_failed, self.stable_steps = True, 0
            return {'task_success': False, 'measurement_error': f'{type(error).__name__}: {error}',
                'stable_steps': 0, 'current_phase': self.phase_index,
                'completed_phases': list(self.completed_phases), 'native_control_samples': self.native_control_samples}

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        value = self._update()
        if value is None:
            return None
        value.update(schema='private_physical_contact_cycle_v1', episode=self.episode,
            snapshot_id=observation.snapshot_id, action_step=observation.action_step, action=action,
            configuration=self.config, initial_native_pair=self.initial, initial_qualification=self.initial_qualification)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


class ContactFinalRelationEvaluator:
    """Measure a stable released final relation, including an already-correct no-op."""

    def __init__(self, session, controller, config, output):
        self.bound = _configuration(session, config)
        self.objects = {key: self.bound[key] for key in OBJECTS}
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode = 0

    def begin_episode(self, observation):
        self.episode += 1
        self.initial_qualification = qualify_operation(self.session, self.controller, self.config, self.bound)
        self.initial = None
        self.stable_steps = self.native_control_samples = 0
        self.anchor_disturbed = self.wrong_grasp = self.measurement_failed = False
        self.completed = {'desired_final_relation': False, 'stable_released_support': False}
        self.executed_actions = []
        self.no_op_initially_satisfied = None
        try:
            self.initial = _pair_measure(self.objects, self.bound['source_floor'], self.session)
            geometry = self.initial['contact_geometry']
            if geometry['available']:
                self.no_op_initially_satisfied = bool(self.initial_qualification['accepted'] and
                    geometry['relation'] == self.config['parameters']['desired_relation'] and
                    self.initial['task_supported'] and self.initial['stationary'] and
                    _empty_released(self.controller.debug_gripper_state()))
        except Exception:
            self.measurement_failed = True
        self.record(observation, action=None)

    def _result(self, *, available, task_success, criteria=None, **measurements):
        items = [{'id': key, 'completed': done,
            'evaluation_validity': 'valid' if available or done else 'unavailable'}
            for key, done in self.completed.items()]
        completed = sum(self.completed.values())
        return {'schema': 'private_physical_contact_final_relation_v1', 'task_revision': TASK_REVISION,
            'scoring_version': 'task_completion_v2',
            'evaluation_validity': 'complete' if available else 'unavailable',
            'completion_items': items, 'completed_goals': completed, 'total_goals': len(items),
            'completion_score': completed / len(items) if available else None,
            'score_percent': 100. * completed / len(items) if available else None,
            'task_success': bool(task_success and available),
            'constraint_compliance': all(criteria.values()) if criteria and available else None,
            'constraints_satisfied': all(criteria.values()) if criteria and available else None,
            'criteria': criteria or {}, 'desired_relation': self.config['parameters']['desired_relation'],
            'no_op_initially_satisfied': self.no_op_initially_satisfied,
            'executed_actions': list(self.executed_actions), 'executed_action_count': len(self.executed_actions),
            'native_control_samples': self.native_control_samples, 'stable_steps': self.stable_steps,
            'required_stability_steps': STABILITY_STEPS, **measurements}

    def _update(self, *, count_step=False):
        if not self.episode:
            return None
        try:
            from wia.tasks.interaction_setup import reference_position
            if self.initial is None:
                raise ValueError('Initial native measurements are unavailable.')
            measured = _pair_measure(self.objects, self.bound['source_floor'], self.session)
            robot = _robot_state(self.session, self.bound['source_floor'])
            gripper = self.controller.debug_gripper_state()
            rows, parameters = measured['objects'], self.config['parameters']
            target, anchor = parameters['target'], parameters['anchor']
            self.anchor_disturbed |= not _near(rows[anchor], self.initial['objects'][anchor])
            self.wrong_grasp |= (gripper.get('object_name') not in (None, self.objects[target].name)
                and not legal_background_grasp(self.session, gripper.get('object_name')))
            midpoint = reference_position(self.session, self.objects[anchor].name,
                [*parameters['source_pair_midpoint_xy_m'], 0.])[:2]
            geometry = measured['contact_geometry']
            if not geometry['available']:
                raise ValueError(geometry.get('reason', 'Native pair relation is unavailable.'))
            relation = geometry['relation'] == parameters['desired_relation']
            released_support = bool(measured['task_supported'] and measured['stationary'] and _empty_released(gripper))
            constraints = {'anchor_preserved_throughout': not self.anchor_disturbed,
                'no_wrong_grasp': not self.wrong_grasp,
                'pair_collidable_movable': all(not row['visual_only'] and not row['fixed_base'] for row in rows.values()),
                'pair_in_original_neighborhood': all(np.linalg.norm(_array(row['position'])[:2] - midpoint) <= 1.5
                    for row in rows.values()),
                'pair_in_source_room': all(self.session.env.scene.seg_map.get_room_instance_by_point(
                    _array(row['position'])[:2]) == self.session.room_name for row in rows.values()),
                'robot_in_source_room': robot['in_source_room'],
                'robot_source_floor_and_clear': robot['supported_upright']}
            valid = bool(relation and released_support and all(constraints.values()))
            # Checkpoints remain factual evidence even if a terminal constraint later fails.
            self.completed['desired_final_relation'] |= relation
            if count_step:
                self.native_control_samples += 1
                self.stable_steps = self.stable_steps + 1 if valid else 0
            elif not valid:
                self.stable_steps = 0
            stable = self.stable_steps >= STABILITY_STEPS
            self.completed['stable_released_support'] |= stable
            available = bool(not self.measurement_failed and self.initial_qualification['accepted'])
            return self._result(available=available, task_success=bool(valid and stable),
                criteria=constraints, native_pair=measured, native_robot=robot, native_gripper=gripper,
                terminal_checks={'desired_final_relation': relation, 'stable_released_support': bool(valid and stable)})
        except Exception as error:
            self.measurement_failed, self.stable_steps = True, 0
            return self._result(available=False, task_success=None,
                measurement_error=f'{type(error).__name__}: {error}')

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        if action is not None:
            self.executed_actions.append({'action_step': observation.action_step, 'action': action})
        value = self._update()
        value.update(episode=self.episode, snapshot_id=observation.snapshot_id,
            action_step=observation.action_step, action=action, configuration=self.config,
            initial_native_pair=self.initial, initial_qualification=self.initial_qualification)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    evaluator = ContactFinalRelationEvaluator if config['operation'] == FINAL_OPERATION else ContactCycleEvaluator
    return evaluator(session, controller, config, output)


def main():
    """Generate the complete assigned category on CPU; never enqueue a capture."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=ESI_ROOT / 'dataset/json_clean')
    parser.add_argument('--operation', choices=(FINAL_OPERATION, *OPERATIONS), default=FINAL_OPERATION)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    corpus = args.corpus.resolve()
    sources = [SourceInstance(path, str(path.relative_to(corpus)), *CategoryConverter.category)
        for path in sorted((corpus / Path(*CategoryConverter.category)).rglob('*.json'))]
    # Match CategoryRunner's default empty conversion config to retain candidate
    # identities when the supervisor batch is picked up by the shared runner.
    config = {} if args.operation == FINAL_OPERATION else {'operation': args.operation}
    context = ConversionContext(args.batch_root.resolve(), seed=args.seed, config=config)
    manifest = CategoryConverter().convert_all(sources, context)
    print(json.dumps({key: manifest[key] for key in ('source_count', 'counts', 'failure_distribution', 'sample_ids')}))


if __name__ == '__main__':
    main()

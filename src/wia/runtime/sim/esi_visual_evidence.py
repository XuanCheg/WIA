"""Read synchronized private visual evidence without recognizing from mask size.

Run: python -B -m pytest -q tests/test_task_edit2_visual_evidence.py
Install with install_visual_evidence(session, controller). Enable private native
segmentation during initialization elsewhere; this module never changes sensors,
renders, steps, sets poses, edits observations or emits public history/events.
Positive semantic evidence requires exact versioned private image-feature audits.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np

COLLECTOR_SCHEMA = 'native_visual_evidence_v1'
FEATURE_AUDIT_SCHEMA = 'native_visual_feature_audit_v1'
LOCATION_AUDIT_SCHEMA = 'native_visual_location_audit_v1'
FROZEN_INPUT_AUDIT_SCHEMA = 'native_frozen_initial_input_audit_v1'
FROZEN_GLOBAL = 'initial_overview_rgb'


def _owner():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Native visual evidence must be read on the simulator owner/main thread.')


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def array_sha256(value):
    """Hash numeric shape/dtype and exact pixels; no image payload enters JSON."""
    value = np.ascontiguousarray(value)
    header = _json({'shape': list(value.shape), 'dtype': str(value.dtype)}).encode()
    return hashlib.sha256(header + b'\0' + value.tobytes()).hexdigest()


def _pose(obj):
    position, quaternion = [_array(value).astype(float) for value in obj.get_position_orientation()]
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all():
        raise ValueError('Native object/sensor pose is unavailable.')
    return {'position_m': position.tolist(), 'quaternion_xyzw': quaternion.tolist()}


def _identity(obj):
    scale = _array(obj.scale).astype(float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Native asset scale is unavailable.')
    return {'name': obj.name, 'prim_path': str(obj.prim_path), 'category': obj.category,
        'model': getattr(obj, 'model', None), 'scale': scale.tolist()}


def _native_context(session, observation):
    from wia.tasks.loader import LoadedScene, LoadedTask, LoadedRigidTask
    adapter_loaded = getattr(getattr(session, 'scene_adapter', None), 'task', None)
    explicit_task = getattr(session, 'loaded_task', None)
    task = explicit_task if isinstance(explicit_task, (LoadedTask, LoadedRigidTask)) else (
        adapter_loaded if isinstance(adapter_loaded, (LoadedTask, LoadedRigidTask)) else None)
    scene = adapter_loaded if isinstance(adapter_loaded, LoadedScene) else None
    owner = task if task is not None else scene
    definition = getattr(owner, 'definition', None)
    source = getattr(definition, 'source', None)
    identity = source.get('json') if isinstance(source, dict) else getattr(source, 'json_path', None)
    clock = float(session.pipeline.og.sim.current_time)
    if not math.isfinite(clock):
        raise ValueError('Native physics clock is unavailable.')
    if task is not None and scene is not None and (task.source_sha256 != scene.source_sha256 or task.seed != scene.seed):
        raise ValueError('Loaded task and loaded scene disagree on source identity or seed.')
    reset = getattr(session, 'native_reset_id', None)
    return {'task_id': getattr(getattr(task, 'definition', None), 'id', None),
        'task_sha256': getattr(task, 'file_sha256', None),
        'scene_id': getattr(getattr(scene, 'definition', None), 'id', None),
        'scene_sha256': getattr(scene, 'file_sha256', None),
        'provenance_role': 'task_and_scene' if task is not None and scene is not None else
            'task' if task is not None else 'scene_only' if scene is not None else 'unavailable',
        'source_identity': identity, 'source_sha256': getattr(owner, 'source_sha256', None),
        'reset_id': reset, 'episode_id': str(getattr(session, 'episode_id', 'reset_' + str(reset))),
        'seed': getattr(owner, 'seed', None),
        'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
        'captured_at': getattr(observation, 'captured_at', None), 'physics_time_s': clock,
        'phase': getattr(getattr(session, 'temporal_phase_controller', None), 'phase', None)}


def _context_complete(context):
    required = ('task_id', 'task_sha256', 'source_identity', 'source_sha256', 'reset_id',
        'seed', 'snapshot_id', 'action_step', 'captured_at', 'physics_time_s')
    return all(context.get(key) is not None for key in required)


def _retained_capture_matches(session, observation):
    source = getattr(session, '_observation_source', None)
    snapshot = getattr(source, 'latest_snapshot', None)
    latest = getattr(snapshot, 'observation', None)
    return bool(latest is not None and all(getattr(latest, key, None) == getattr(observation, key, None)
        for key in ('snapshot_id', 'action_step', 'captured_at')))


def _labels(info, modality):
    labels = info.get(modality)
    if isinstance(labels, dict):
        labels = labels.get('idToLabels', labels.get('id_to_labels', labels))
    if not isinstance(labels, dict):
        raise ValueError('Native instance-label mapping is unavailable.')
    result = {}
    for key, value in labels.items():
        try:
            result[int(key)] = value
        except (ValueError, TypeError):
            continue
    if not result:
        raise ValueError('Native instance-label mapping is empty.')
    return result


def _label_matches(label, obj):
    if isinstance(label, str):
        return label == obj.name or label == str(obj.prim_path) or label.startswith(str(obj.prim_path) + '/')
    if isinstance(label, dict):
        # A semantic category such as {'class': 'bowl'} is not instance identity.
        return any(_label_matches(label.get(key), obj) for key in (
            'name', 'object_name', 'prim_path', 'primPath', 'path', 'label'))
    if isinstance(label, (tuple, list)):
        return any(_label_matches(value, obj) for value in label)
    return False


def _mask_statistics(segmentation, labels, obj):
    ids = sorted(key for key, label in labels.items() if _label_matches(label, obj))
    mask = np.isin(segmentation, ids)
    ys, xs = np.nonzero(mask)
    height, width = segmentation.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    normalized = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height] if bbox else None
    return {'instance_ids': ids, 'visible_pixels': int(len(xs)),
        'mask_fraction': float(len(xs) / (width * height)), 'bbox_xyxy': bbox,
        'bbox_normalized': normalized, 'mask_sha256': array_sha256(mask.astype(np.uint8)),
        'minimum_extent_px': min(bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else 0}, mask


def _capture_metadata(info, modality):
    """Check channel frame stamps when the native annotator exposes them."""
    metadata = info.get('capture_metadata', {})
    if not isinstance(metadata, dict):
        metadata = {}
    rgb, segmentation = metadata.get('rgb', {}), metadata.get(modality, {})
    for key in ('frame_id', 'frameId', 'timestamp'):
        first = rgb.get(key) if isinstance(rgb, dict) else None
        second = segmentation.get(key) if isinstance(segmentation, dict) else None
        if (first is None) != (second is None) or first is not None and first != second:
            raise ValueError('Native RGB/segmentation capture frame metadata differs or is incomplete.')
    def primitive(value):
        value = value.item() if isinstance(value, np.generic) else value
        return value if value is None or isinstance(value, (str, int, float, bool)) else None
    safe = {name: {key: primitive(value.get(key)) for key in ('frame_id', 'frameId', 'timestamp') if key in value}
        for name, value in (("rgb", rgb), (modality, segmentation)) if isinstance(value, dict)}
    return {'channel_metadata': safe,
        'native_frame_id': primitive(info.get('frame_id', info.get('frameId'))),
        'method': 'single_native_get_obs_exact_RGB_current_retained_snapshot_stable_physics_clock'}


def _audit_review_present(receipt):
    review = receipt.get('review', {})
    return bool(review.get('reviewer') and review.get('reviewed_at') and
        review.get('method') in ('manual_native_image_feature_review', 'explicit_native_diagnostic_feature_review'))


def _match_feature_audit(receipts, context, stats, all_stats, identities, masks):
    errors, audited = [], False
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get('binding') != context['binding'] or receipt.get('camera') != context['camera']:
            continue
        if receipt.get('schema') != FEATURE_AUDIT_SCHEMA or not receipt.get('audit_id') or not _audit_review_present(receipt):
            errors.append('feature_audit_schema_or_explicit_review_missing')
            continue
        if receipt.get('context') != context:
            errors.append('feature_audit_exact_snapshot_RGB_identity_context_mismatch')
            continue
        support = receipt.get('support_binding')
        if (support not in all_stats or receipt.get('support_identity') != identities[support] or
                receipt.get('support_mask_sha256') != all_stats[support]['mask_sha256']):
            errors.append('feature_audit_support_mask_or_identity_mismatch')
            continue
        audited = True
        if receipt.get('informative') is not True:
            continue
        feature = receipt.get('informative_feature_evidence', {})
        ids = feature.get('feature_ids')
        if (not isinstance(ids, list) or not ids or any(not isinstance(value, str) or not value for value in ids) or
                len(set(ids)) != len(ids) or
                feature.get('semantic_identity_verified') is not True or
                feature.get('view_kind') not in ('direct', 'reflected')):
            errors.append('audited_semantic_identity_or_feature_ids_missing')
            continue
        if feature['view_kind'] == 'direct':
            if support != context['binding'] or stats['visible_pixels'] == 0:
                errors.append('direct_feature_has_no_target_identity_mask')
                continue
        elif (support == context['binding'] or feature.get('mirror_binding') != support or
                feature.get('real_target_correspondence_verified') is not True):
            errors.append('mirror_mask_alone_does_not_identify_real_target')
            continue
        bbox = receipt.get('feature_crop_bbox_xyxy')
        height, width = masks[support].shape
        if (not isinstance(bbox, list) or len(bbox) != 4 or any(type(value) is not int for value in bbox)
                or not 0 <= bbox[0] < bbox[2] <= width or not 0 <= bbox[1] < bbox[3] <= height or
                not masks[support][bbox[1]:bbox[3], bbox[0]:bbox[2]].any()):
            errors.append('audited_feature_crop_does_not_intersect_its_identity_support')
            continue
        return {'informative': True, 'informative_feature_evidence': deepcopy(feature),
            'feature_audit_id': receipt['audit_id'], 'feature_crop_bbox_xyxy': bbox,
            'support_binding': support, 'audited': True}, errors
    return {'informative': False, 'informative_feature_evidence': {}, 'audited': audited,
        'information_unavailable_reason': errors[-1] if errors else 'exact_private_feature_audit_missing'}, errors


def _location_receipt(receipts, context):
    for receipt in receipts:
        if (isinstance(receipt, dict) and receipt.get('schema') == LOCATION_AUDIT_SCHEMA and
                receipt.get('context') == context and receipt.get('native_location_verified') is True and
                receipt.get('legal_observation_pose') is True and _audit_review_present(receipt)):
            location = receipt.get('location', {})
            if not isinstance(location, dict) or not location or not all(isinstance(value, str) and value for value in location.values()):
                continue
            return {'visit_verified': True, 'visit_evidence': {'snapshot_id': context['snapshot_id'],
                'legal_observation_pose': True, 'native_location_verified': True,
                'audit_id': receipt.get('audit_id'), 'location': deepcopy(location)}, **deepcopy(location)}
    return {'visit_verified': False, 'visit_evidence': {},
        'visit_unavailable_reason': 'exact_verified_native_location_and_legal_pose_receipt_missing'}


def _write_crop(output, rgb, mask, bbox, token, maximum_edge):
    from PIL import Image
    x0, y0, x1, y1 = bbox
    crop = Image.fromarray(rgb[y0:y1, x0:x1])
    identity = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255)
    if max(crop.size) > maximum_edge:
        factor = maximum_edge / max(crop.size)
        size = tuple(max(1, int(value * factor)) for value in crop.size)
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        identity = identity.resize(size, Image.Resampling.NEAREST)
    output.mkdir(parents=True, exist_ok=True)
    rgb_path, mask_path = output / (token + '_rgb.png'), output / (token + '_mask.png')
    if not rgb_path.is_file():
        crop.save(rgb_path)
    if not mask_path.is_file():
        identity.save(mask_path)
    return {'rgb_crop_path': str(rgb_path.resolve()), 'identity_mask_crop_path': str(mask_path.resolve()),
        'original_bbox_xyxy': bbox, 'saved_size': list(crop.size)}


def _frozen_input_audit(session, observation, delivered, base_context, receipts):
    """Audit the original frozen input only; never attach current native identities."""
    result = {'delivered': True, 'live': False, 'informative_for_live_visit': False,
        'initial_input_reviewed': False,
        'reason': 'frozen_initial_context_is_not_a_live_encounter_or_post_change_view',
        'rgb_sha256': hashlib.sha256(np.asarray(delivered).tobytes()).hexdigest()}
    overview = getattr(observation, 'initial_overview', None)
    try:
        if overview is None or not _context_complete(base_context):
            raise ValueError('Frozen native overview or exact loaded task provenance is unavailable.')
        identity = {'episode_id': base_context['episode_id'], 'reset_id': base_context['reset_id'], 'seed': base_context['seed']}
        overview.validate_identity(identity)
        if not np.array_equal(np.asarray(overview.rgb), np.asarray(delivered)):
            raise ValueError('Delivered frozen Global differs from its original native capture.')
        configuration = {key: getattr(overview, key) for key in ('extent_xy_m', 'image_right_world_xy',
            'image_up_world_xy', 'robot_start_position_m', 'robot_start_yaw_rad', 'source', 'role', 'frame')}
        context = {key: base_context[key] for key in ('task_id', 'task_sha256', 'scene_id', 'scene_sha256',
            'source_identity', 'source_sha256', 'episode_id', 'reset_id', 'seed')}
        context.update(camera=FROZEN_GLOBAL, rgb_sha256=result['rgb_sha256'],
            freeze_episode_id=overview.episode_id, freeze_reset_id=overview.reset_id, freeze_seed=overview.seed,
            freeze_snapshot_id=overview.snapshot_id, freeze_captured_at=overview.captured_at,
            image_size=[overview.rgb.shape[1], overview.rgb.shape[0]],
            freeze_configuration_sha256=hashlib.sha256(_json(configuration).encode()).hexdigest())
        result['audit_context'] = context
        for receipt in receipts:
            if (isinstance(receipt, dict) and receipt.get('schema') == FROZEN_INPUT_AUDIT_SCHEMA and
                    receipt.get('context') == context and receipt.get('audit_id') and _audit_review_present(receipt) and
                    receipt.get('initial_input_reviewed') is True and
                    receipt.get('scope') == 'frozen_initial_input_shortcut_audit_only'):
                findings = receipt.get('findings', [])
                if not isinstance(findings, list) or len(findings) > 32 or any(
                        not isinstance(value, str) or len(value) > 512 for value in findings):
                    continue
                result.update(initial_input_reviewed=True, audit_id=receipt['audit_id'],
                    findings=deepcopy(findings), scope=receipt['scope'])
                break
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
        result['audit_unavailable_reason'] = f'{type(error).__name__}: {error}'
    return result


def collect_native_visual_evidence(session, observation, bindings, parameters=None, *, controller=None,
                                   audited_receipts=None, location_receipts=None, frozen_input_receipts=None, output_dir=None,
                                   max_crops=16, maximum_crop_edge=512):
    """Collect measured masks/crops separately from exact privately audited semantics.

    ``audited_receipts`` and ``location_receipts`` are private lists, not Agent
    assertions. With no audit, masks/diagnostics are still returned but
    ``available``/``informative_available`` are false. Frozen Global is recorded
    separately and never becomes a live encounter or post-change observation.
    No crop files are written unless ``output_dir`` is explicitly supplied.
    """
    _owner()
    if not 0 <= max_crops <= 128 or not 1 <= maximum_crop_edge <= 2048:
        raise ValueError('Visual diagnostic crop bounds are invalid.')
    parameters = parameters or {}
    audited_receipts = list(audited_receipts or [])
    location_receipts = list(location_receipts or [])
    delivered = observation.rgb_images()
    registry = getattr(session, 'robot_camera_registry', {})
    errors, measurements, cameras, identities, objects = {}, {}, {}, {}, {}
    base_context = {}
    try:
        base_context = _native_context(session, observation)
        retained = _retained_capture_matches(session, observation)
        robot_pose = _pose(session.robot)
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
        errors['native_capture_context'] = f'{type(error).__name__}: {error}'
        retained, robot_pose = False, None
    for key, binding in bindings.items():
        name = binding if isinstance(binding, str) else binding.get('name') if isinstance(binding, dict) else getattr(binding, 'name', None)
        try:
            obj = session.env.scene.object_registry('name', name)
            if obj is None or obj.name != name:
                raise ValueError('Bound native object identity is unavailable.')
            objects[key], identities[key] = obj, _identity(obj)
            measurements[key] = {'native_identity': identities[key], 'native_pose': _pose(obj), 'views': {}}
            segmentation = getattr(session.env.scene, 'seg_map', None)
            query = getattr(segmentation, 'get_room_instance_by_point', None)
            if callable(query):
                measured_room = query(np.asarray(measurements[key]['native_pose']['position_m'])[:2])
                measurements[key]['measured_room_label'] = str(measured_room) if measured_room is not None else None
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
            errors['binding:' + key] = f'{type(error).__name__}: {error}'
    context_complete = _context_complete(base_context) and retained
    if not retained:
        errors['retained_capture'] = 'Observation is not the current retained native capture.'
    if not _context_complete(base_context):
        errors['audit_context'] = 'Task/source/reset/seed/snapshot provenance is incomplete.'
    if parameters.get('source_sha256') and parameters['source_sha256'] != base_context.get('source_sha256'):
        context_complete = False
        errors['source'] = 'Declared task parameters disagree with the loaded native source.'
    rows, available_views, audited_views, crop_count = [], [], set(), 0
    for view, delivered_rgb in delivered.items():
        if view == FROZEN_GLOBAL or view not in registry:
            continue
        try:
            sensor_name = registry[view]
            sensor = session.robot.sensors[sensor_name]
            if not str(sensor.prim_path).startswith(str(session.robot.prim_path) + '/'):
                raise ValueError('Registered camera is not mounted on the current native robot.')
            before = float(session.pipeline.og.sim.current_time)
            native, info = sensor.get_obs()
            actual = _array(native['rgb'])
            rgb = np.asarray(delivered_rgb)
            if (actual.dtype != np.uint8 or rgb.dtype != np.uint8 or actual.ndim != 3 or
                    actual.shape[-1] not in (3, 4) or not np.array_equal(actual[..., :3], rgb)):
                raise ValueError('Native RGB is not exactly equal to the delivered RGB.')
            modality = next((name for name in ('seg_instance', 'seg_instance_id') if name in native and name in info), None)
            if modality is None:
                raise ValueError('Private native instance segmentation modality/labels are unavailable.')
            segmentation = _array(native[modality])
            if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
                segmentation = segmentation[..., 0]
            if segmentation.shape != rgb.shape[:2] or not np.issubdtype(segmentation.dtype, np.integer):
                raise ValueError('Native instance mask dimensions/type do not match delivered RGB.')
            labels = _labels(info, modality)
            stamp = _capture_metadata(info, modality)
            after = float(session.pipeline.og.sim.current_time)
            if not math.isfinite(before) or before != after or before != base_context.get('physics_time_s'):
                raise ValueError('Native physics advanced between the capture context and mask read.')
            cam = {'camera': view, 'sensor_name': sensor_name, 'sensor_prim_path': str(sensor.prim_path),
                'rgb_sha256': hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                'segmentation_sha256': array_sha256(segmentation), 'segmentation_modality': modality,
                'image_size': [rgb.shape[1], rgb.shape[0]], 'camera_pose': _pose(sensor),
                'label_mapping_sha256': hashlib.sha256(_json({str(key): value for key, value in labels.items()}).encode()).hexdigest(),
                'capture_synchronization': stamp, 'native_rgb_matches_delivered': True,
                'retained_capture_verified': retained}
            cameras[view] = cam
            available_views.append(view)
            masks, stats = {}, {}
            for key, obj in objects.items():
                stats[key], masks[key] = _mask_statistics(segmentation, labels, obj)
            for key in objects:
                context = {**base_context, **{field: cam[field] for field in (
                    'camera', 'sensor_name', 'sensor_prim_path', 'rgb_sha256', 'segmentation_sha256',
                    'segmentation_modality', 'image_size', 'camera_pose', 'label_mapping_sha256')},
                    'binding': key, 'native_identity': identities[key], 'native_pose': measurements[key]['native_pose'],
                    'robot_pose': robot_pose, 'target_mask_sha256': stats[key]['mask_sha256']}
                audit, audit_errors = _match_feature_audit(audited_receipts, context, stats[key], stats, identities, masks)
                if not context_complete:
                    audit.update(informative=False, informative_feature_evidence={},
                        information_unavailable_reason='native_retained_capture_provenance_incomplete')
                if audit.get('audited') and context_complete:
                    audited_views.add(view)
                row = {**stats[key], **audit, 'snapshot_id': observation.snapshot_id,
                    'action_step': observation.action_step, 'camera': view, 'native_name': identities[key]['name'],
                    'native_rgb_matches_delivered': True, 'identity_synchronized': bool(context_complete and
                        (stats[key]['visible_pixels'] > 0 or audit['informative'] and
                            audit['informative_feature_evidence'].get('view_kind') == 'reflected')),
                    'measurement_available': True, 'audit_context': context, 'feature_audit_errors': audit_errors}
                row.update(_location_receipt(location_receipts, context) if row['informative'] else
                    {'visit_verified': False, 'visit_evidence': {}})
                row['identity_evidence_source'] = ('explicit_audited_reflection_correspondence' if audit['informative'] and
                    audit['informative_feature_evidence'].get('view_kind') == 'reflected' else
                    'native_instance_mask' if stats[key]['visible_pixels'] else 'unavailable')
                bbox = audit.get('feature_crop_bbox_xyxy') or stats[key]['bbox_xyxy']
                if output_dir is not None and bbox and crop_count < max_crops:
                    support = audit.get('support_binding', key)
                    token = hashlib.sha256(_json({'context': context, 'bbox': bbox, 'support': support}).encode()).hexdigest()[:32]
                    row['crop_artifacts'] = _write_crop(Path(output_dir), rgb, masks[support], bbox, token, maximum_crop_edge)
                    crop_count += 1
                measurements[key]['views'][view] = row
                rows.append((key, row))
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, OSError) as error:
            errors[view] = f'{type(error).__name__}: {error}'
    selected = {}
    for key in bindings:
        candidates = [row for binding, row in rows if binding == key]
        if candidates:
            selected[key] = max(candidates, key=lambda row: (row['visit_verified'], row['informative'], row['visible_pixels']))
        else:
            selected[key] = {'native_name': identities.get(key, {}).get('name'), 'snapshot_id': observation.snapshot_id,
                'informative': False, 'informative_feature_evidence': {}, 'measurement_available': False,
                'visit_verified': False, 'visit_evidence': {}, 'information_unavailable_reason': 'no_synchronized_native_identity_view'}
    control = {'available': False, 'reason': 'native_public_control_context_unavailable'}
    provider = getattr(controller, 'geometric_rearrangement_evidence', None)
    if callable(provider):
        try:
            value = provider()
            fields = ('available', 'sample_id', 'control_sequence', 'control_active', 'state_setter_used',
                'controlled_object_names', 'grasped_object_names', 'contact_object_names', 'reason')
            control = {key: deepcopy(value[key]) for key in fields if key in value}
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as error:
            control = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    information_available = any(row.get('informative') for row in selected.values())
    from core.action_units import ROBOT_ACTIONS, TERMINAL_ATOMS
    from core.observations import EXECUTION_STATUSES
    public_history = [{'action': row.action, 'result_summary': row.result_summary}
        for row in getattr(observation, 'action_history', ())[-32:]
        if row.action in ROBOT_ACTIONS + TERMINAL_ATOMS and row.result_summary in EXECUTION_STATUSES]
    frozen = {FROZEN_GLOBAL: _frozen_input_audit(session, observation, delivered[FROZEN_GLOBAL],
        base_context, frozen_input_receipts or [])} if FROZEN_GLOBAL in delivered else {}
    if frozen.get(FROZEN_GLOBAL, {}).get('initial_input_reviewed') is True:
        audited_views.add(FROZEN_GLOBAL)
    return {'schema': COLLECTOR_SCHEMA, 'available': information_available,
        'informative_available': information_available, 'measurement_available': bool(available_views),
        'objects': selected, 'measurements': measurements, 'cameras': cameras,
        'view_evidence': [{'binding': key, **row} for key, row in rows if row['informative']],
        'available_views': available_views, 'audited_camera_names': sorted(audited_views),
        'all_delivered_views_audited': bool(delivered) and set(delivered) <= audited_views,
        'frozen_views': frozen, 'measurement_errors': errors,
        'native_capture_context': base_context, 'native_public_control_context': control,
        'public_action_history': public_history, 'crop_count': crop_count,
        'recognition_inferred_from_mask_size': False,
        'information_unavailable_reason': None if information_available else 'no_exact_audited_native_semantic_feature_receipt'}


def install_visual_evidence(session, controller=None):
    """Install a private read-only callback without changing sensor modalities.

    Root supplies ``session.visual_evidence_feature_receipts`` and optionally
    ``visual_evidence_verified_locations`` as lists of exact versioned audits.
    Set ``visual_evidence_output_dir`` explicitly for bounded review crops.
    The resulting callback never changes public observations, events or history.
    """
    _owner()
    def provider(observation, bindings, parameters):
        return collect_native_visual_evidence(session, observation, bindings, parameters,
            controller=controller,
            audited_receipts=getattr(session, 'visual_evidence_feature_receipts', ()),
            location_receipts=getattr(session, 'visual_evidence_verified_locations', ()),
            frozen_input_receipts=getattr(session, 'visual_evidence_frozen_input_receipts', ()),
            output_dir=getattr(session, 'visual_evidence_output_dir', None))
    session.task_visual_evidence = provider
    return provider

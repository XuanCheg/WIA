"""Capture human-only category review views without changing native actors.

Use via capture_loaded_views; run pytest tests/test_review_camera_geometry.py.
Ray tests guide placement; per-object rendered pixels report remaining occlusion.
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from wia.runtime.sim.esi_debug_cameras import camera_orientation
from wia.runtime.sim.esi_review_camera_geometry import fit_review_view, framed_near_plane, review_directions, review_profile


def review_view_names(definition):
    """Declare a bounded set of group details in the immutable capture manifest."""
    count = sum(binding.category not in ('floor', 'floors', 'ceiling', 'ceilings')
                for binding in definition.bindings.values())
    return ['object_overview', 'object_top', 'object_side',
            *[f'object_detail_{i:02d}' for i in range(min(8, max(1, count)))]]


def _array(value):
    return np.asarray(value.detach().cpu() if hasattr(value, 'detach') else value, dtype=float)


def review_objects(session, selected):
    names = list(getattr(session, 'task_object_names', ()))
    names += [binding.name for binding in selected.definition.bindings.values()]
    objects, issues = [], []
    floor_tops = []
    for binding in selected.definition.bindings.values():
        if binding.category in ('floor', 'floors') or 'floor' in binding.name:
            floor = session.env.scene.object_registry('name', binding.name)
            if floor is not None:
                try:
                    floor_tops.append(float(_array(floor.aabb[1])[2]))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
    for name in dict.fromkeys(names):
        obj = session.env.scene.object_registry('name', name)
        if obj is None:
            issues.append({'name': name, 'reason': 'native_object_missing'})
            continue
        if (obj is getattr(session, 'robot', None)
                or obj.category in ('floor', 'floors', 'ceiling', 'ceilings', 'walls')
                or not bool(getattr(obj, 'visible', True))):
            continue
        try:
            low, high = (_array(value) for value in obj.aabb)
        except (AttributeError, RuntimeError, ValueError, TypeError) as error:
            issues.append({'name': name, 'reason': 'native_bounds_unavailable', 'error': str(error)})
            continue
        if (low.shape == high.shape == (3,) and np.isfinite([low, high]).all()
                and np.all(high > low) and np.max(np.abs([low, high])) < 900):
            objects.append(obj)
            if floor_tops and high[2] < min(floor_tops) - .5:
                issues.append({'name': name, 'reason': 'object_below_bound_source_floor',
                    'aabb_m': [low.tolist(), high.tolist()], 'floor_top_z_m': min(floor_tops),
                    'camera_reframing_cannot_repair_object_state': True})
        elif not np.isfinite([low, high]).all() or np.any(high <= low):
            issues.append({'name': name, 'reason': 'invalid_native_bounds'})
    session.human_review_object_issues = issues
    return objects


def detail_target_indices(objects, bindings):
    """Prioritize task actors over large support/region furniture for closeups."""
    roles = {}
    for key, binding in bindings.items():
        roles.setdefault(binding.name, []).append(key)
    context_roles = ('destination', 'support', 'reference', 'anchor', 'floor', 'region')
    indices = [index for index, obj in enumerate(objects)
        if not all(any(word in role.lower() for word in context_roles)
                   for role in roles.get(obj.name, ['source_actor']))]
    return indices or list(range(len(objects)))


def _explicit_review_details(selected):
    """Read one opt-in human-only detail contract from the selected variant."""
    scene = getattr(selected.definition, 'scene', None)
    if scene is None:
        return None
    variant = scene.variants[selected.variant]
    declarations = []
    for patch in variant.patches:
        if (getattr(patch, 'operation', None) != 'category_initialization'
                or getattr(patch, 'phase', None) != 'after_postprocessing'):
            continue
        layout = patch.parameters.get('layout', {})
        if isinstance(layout, dict) and 'human_review' in layout:
            declarations.append(layout['human_review'])
    if not declarations:
        return None
    if len(declarations) != 1:
        raise ValueError('Human review requires a single explicit detail configuration.')
    config = declarations[0]
    if (not isinstance(config, dict)
            or set(config) != {'detail_binding_groups', 'detail_elevation_deg'}):
        raise ValueError('Human review requires detail_binding_groups and detail_elevation_deg.')
    groups, elevation = config['detail_binding_groups'], config['detail_elevation_deg']
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError('Human review detail_binding_groups must be nonempty.')
    for group in groups:
        if (not isinstance(group, (list, tuple)) or not group
                or any(not isinstance(key, str) or key not in selected.definition.bindings for key in group)
                or len(set(group)) != len(group)):
            raise ValueError('Human review groups require nonempty, bound, unrepeated binding keys.')
    if isinstance(elevation, bool) or not isinstance(elevation, Real) or not np.isfinite(elevation):
        raise ValueError('Human review detail_elevation_deg must be a finite angle.')
    return {'detail_binding_groups': [list(group) for group in groups],
            'detail_elevation_deg': float(elevation)}


def _explicit_detail_objects(session, selected, config, detail_names, fixed_views):
    """Resolve exact declared native actors; a missing actor cannot be substituted."""
    result = {}
    if config is None:
        return result
    for name, keys in zip(detail_names, config['detail_binding_groups']):
        if fixed_views and name in fixed_views:
            continue
        objects = []
        for key in keys:
            native_name = selected.definition.bindings[key].name
            obj = session.env.scene.object_registry('name', native_name)
            if obj is None or obj.name != native_name:
                raise ValueError('Declared human review native object is missing: ' + key)
            low, high = (_array(value) for value in obj.aabb)
            if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                    or np.any(high <= low) or np.max(np.abs([low, high])) >= 900):
                raise ValueError('Declared human review native object has invalid bounds: ' + key)
            objects.append(obj)
        result[name] = {'binding_group': keys, 'objects': objects,
                        'elevation_deg': config['detail_elevation_deg']}
    return result


def _ray_score(position, objects):
    from omnigibson.utils.sampling_utils import raytest
    from wia.tasks.support_sampling import same_body
    clear, checked = 0, 0
    for obj in objects:
        low, high = (_array(value) for value in obj.aabb)
        centre = (low + high) / 2
        for point in (centre, centre + [0, 0, (high[2] - low[2]) * .4]):
            hit = raytest(position, point.tolist())
            reverse = raytest(point.tolist(), position, only_closest=False)
            unobstructed = (not hit['hit'] or same_body(obj.prim_path, hit['rigidBody']))
            unobstructed &= all(not row['hit'] or same_body(obj.prim_path, row['rigidBody'])
                               for row in reverse)
            clear += int(unobstructed)
            checked += 1
    return clear / max(1, checked)


def _raw_review_observation(sensor, device):
    """Read human RGB and mesh instance IDs without global semantic remapping.

    Native source categories need not belong to OmniGibson's semantic class
    registry. Only buffer conversion is shared with VisionSensor.get_obs;
    instance IDs and their prim-path labels remain the raw annotator values.
    """
    if device == 'cpu':
        preprocess = sensor._preprocess_cpu_obs
    elif 'cuda' in device:
        preprocess = sensor._preprocess_gpu_obs
    else:
        raise ValueError(f'Unsupported review sensor device {device}')
    data, info = {}, {}
    for modality in ('rgb', 'seg_instance_id'):
        raw = sensor._annotators[modality].get_data(device=device)
        pixels = raw['data'] if isinstance(raw, dict) else raw
        data[modality] = preprocess(pixels, modality)
        if modality == 'seg_instance_id':
            info[modality] = raw['info']['idToLabels']
    return data, info


def _render(session, sensor, fitted, objects):
    from omnigibson.utils.usd_utils import PoseAPI
    from wia.tasks.support_sampling import same_body
    sensor.clipping_range = tuple(fitted['clipping_range_m'])
    sensor.set_position_orientation(position=fitted['position_world_m'],
        orientation=camera_orientation(fitted['position_world_m'], fitted['target_world_m']).tolist(), frame='world')
    PoseAPI.invalidate()
    for _ in range(3):
        for _ in range(3):
            session.pipeline.og.sim.render()
        data, info = _raw_review_observation(sensor, session.pipeline.og.sim.device)
    rgb = session._rgb_from_data(data, sensor.name)
    # GPU preprocessing views uint32 instance IDs as int32 for Torch. Restore
    # their unsigned values before matching the raw idToLabels mapping.
    segmentation = _array(data['seg_instance_id']) % (1 << 32)
    counts, rendered_regions = {}, {}
    labels = info.get('seg_instance_id', {})
    for obj in objects:
        identifiers = [int(key) for key, label in labels.items()
                       if str(label) == obj.name or same_body(obj.prim_path, str(label))]
        mask = np.isin(segmentation, identifiers)
        counts[obj.name] = int(mask.sum())
        if counts[obj.name]:
            yy, xx = np.nonzero(mask)
            rendered_regions[obj.name] = {'bbox_xyxy_px': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                'mean_rgb_under_instance_mask': rgb[mask].mean(axis=0).tolist()}
    rendered_instances = {}
    for identifier, label in labels.items():
        mask = segmentation == int(identifier)
        if np.any(mask):
            yy, xx = np.nonzero(mask)
            rendered_instances[str(label)] = {'visible_pixels': int(mask.sum()),
                'bbox_xyxy_px': [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]}
    position, quaternion = (_array(value) for value in sensor.get_position_orientation())
    error = float(np.linalg.norm(position - fitted['position_world_m']))
    if error > .001:
        raise RuntimeError('Human review camera did not reach its authored position.')
    return rgb, {**fitted, 'actual_position_world_m': position.tolist(),
        'actual_quaternion_xyzw': quaternion.tolist(), 'position_error_m': error,
        'object_pixels': counts, 'rendered_regions': rendered_regions,
        'rendered_instances': rendered_instances,
        'objects_below_64_pixels': [n for n, count in counts.items() if count < 64],
        'render_boundaries': 9, 'visibility_certifies_task_success': False}


def capture_review_views(session, selected, expected_views, *, fixed_views=None):
    """Fit native RGB cameras to bounds and retain explicit incomplete coverage."""
    from omnigibson.sensors import VisionSensor
    objects = review_objects(session, selected)
    parts = selected.definition.source.json_path.replace('\\', '/').split('/')
    profile = review_profile(*parts[-5:-3]) if len(parts) >= 5 else review_profile('', '')
    detail_names = [name for name in expected_views if name.startswith('object_detail_')]
    explicit_config = (_explicit_review_details(selected)
        if any(not fixed_views or name not in fixed_views for name in detail_names) else None)
    explicit_details = _explicit_detail_objects(session, selected, explicit_config, detail_names, fixed_views)
    sensor = getattr(session, '_category_review_sensor', None)
    if sensor is None:
        sensor = VisionSensor(relative_prim_path='/human_cameras/category_review', name='category_review',
            modalities=('rgb',), image_width=768, image_height=768, focal_length=12.)
        sensor.load(session.env.scene)
        sensor.initialize()
        # The constructor auto-enables semantic segmentation when instance IDs
        # are requested. Attach this modality afterward, retaining its normal
        # sensor lifecycle but consuming only raw annotator data above.
        sensor.add_modality('seg_instance_id')
        session._category_review_sensor = sensor
    fov = float(np.degrees(2 * np.arctan(float(sensor.horizontal_aperture) / (2 * float(sensor.focal_length)))))
    bounds = np.asarray([[_array(v) for v in obj.aabb] for obj in objects])
    base = getattr(session, 'robot', None)
    context = list(bounds)
    if base is not None:
        context.append(np.asarray([_array(v) for v in base.aabb]))
    if not context:
        raise RuntimeError('Human review framing has no native object or robot bounds.')
    if not len(bounds):
        bounds = np.asarray(context)
    # Ceiling light anchors can put an otherwise useful whole-task camera above
    # the roof. Preserve their individual detail views and native illumination,
    # but frame the manipulable targets and robot together for layout review.
    separate_context = {binding.name for key, binding in selected.definition.bindings.items()
        if len(parts) >= 5 and parts[-4] == 'Illumination Variability' and key.startswith('lamp_')}
    layout_indices = [i for i, obj in enumerate(objects) if obj.name not in separate_context]
    layout_bounds = bounds[layout_indices] if layout_indices else bounds
    layout_context = list(layout_bounds)
    if base is not None:
        layout_context.append(np.asarray([_array(v) for v in base.aabb]))
    # Partition spatially so distributed targets do not share one unreadable closeup.
    focus = detail_target_indices(objects, selected.definition.bindings)
    if not focus:
        focus = list(range(len(bounds)))
    centres = bounds[focus].mean(axis=1)
    axis = int(np.argmax(np.ptp(centres, axis=0)))
    ordered = np.asarray(focus)[np.argsort(centres[:, axis])]
    partitions = np.array_split(ordered, min(len(ordered), max(1, len(detail_names))))
    groups = {name: partitions[min(i, len(partitions) - 1)] for i, name in enumerate(detail_names)}
    result, evidence = {}, {}
    global_angles = {'global_front': 0, 'global_back': 180, 'global_left': 90,
                     'global_right': 270, 'global_diagonal': 225, 'global_overhead': 0}
    for name in expected_views:
        if name not in global_angles and not name.startswith('object_'):
            continue
        if fixed_views and name in fixed_views:
            fitted = fixed_views[name]
            pixel_objects = list(objects)
            temporal = getattr(session, 'temporal_phase_controller', None)
            if temporal is not None and temporal.parameters.get('survey_revision') == 'free_survey_10_v1':
                names = {obj.name for obj in pixel_objects}
                pixel_objects += [temporal.objects[key] for key in temporal.contents
                    if temporal.objects[key].name not in names]
            result[name], evidence[name] = _render(session, sensor, fitted, pixel_objects)
            continue
        group = groups.get(name)
        chosen = [objects[i] for i in group if i < len(objects)] if group is not None else objects
        frame_bounds = bounds[group] if group is not None else np.asarray(context) if name in global_angles else bounds
        explicit = explicit_details.get(name)
        if explicit is not None:
            chosen = explicit['objects']
            frame_bounds = np.asarray([[_array(value) for value in obj.aabb] for obj in chosen])
        if separate_context and group is None:
            chosen = [obj for obj in objects if obj.name not in separate_context]
            frame_bounds = np.asarray(layout_context) if name in global_angles else layout_bounds
        elevation = (90. if name in ('global_overhead', 'object_top') else
                     profile['detail_elevations'][-1] if name == 'object_side' else
                     profile['detail_elevations'][0] if group is not None else profile['overview_elevation'])
        if explicit is not None:
            elevation = explicit['elevation_deg']
        if name in global_angles and name != 'global_overhead':
            # A labeled compass view must keep its requested bearing. Searching
            # all azimuths independently made front/back/left/right converge on
            # the same opening. Vary elevation instead, preserving the label.
            directions = [review_directions(angle, global_angles[name])[0]
                          for angle in dict.fromkeys((min(elevation, 75.), 35., 15.))]
        else:
            directions = review_directions(elevation, global_angles.get(name, 0))
        candidates = []
        for index, direction in enumerate(directions):
            fitted = fit_review_view(frame_bounds, direction, horizontal_fov_deg=fov,
                                    aspect=float(sensor.image_width) / float(sensor.image_height))
            fitted['clipping_range_m'][1] = max(100., fitted['clipping_range_m'][1])
            if elevation == 90.:
                # A human-only cutaway removes roofs above the entire framed
                # volume. All framed actors remain below the near plane.
                fitted['clipping_range_m'][0] = max(.01,
                    fitted['position_world_m'][2] - fitted['bounds_world_m'][1][2] - .08)
                fitted['human_cutaway_above_framed_bounds'] = True
            try:
                fitted['clear_ray_fraction'] = _ray_score(fitted['position_world_m'], chosen)
            except (RuntimeError, ValueError, KeyError) as error:
                fitted.update(clear_ray_fraction=0., ray_measurement_error=str(error))
            candidates.append((fitted['clear_ray_fraction'], -min(index, len(directions) - index), fitted))
        candidates.sort(key=lambda row: row[:2], reverse=True)
        best = None
        for _, _, fitted in candidates[:2]:
            pixels, measured = _render(session, sensor, fitted, chosen)
            score = (sum(count >= 64 for count in measured['object_pixels'].values()),
                     sum(measured['object_pixels'].values()))
            if best is None or score > best[0]:
                best = (score, pixels, measured)
            if chosen and score[0] == len(chosen):
                break
        if chosen and best[0][0] < len(chosen):
            # Oblique cameras can sit above a low ceiling even when the fitted
            # task is wholly below it. A labeled human-only cutaway clips only
            # foreground space before every framed actor, without moving any
            # actor or touching mounted cameras. Retain it only if coverage rises.
            fitted = dict(best[2])
            near = framed_near_plane(fitted)
            if near > fitted['clipping_range_m'][0] + .01:
                fitted['clipping_range_m'] = [near, fitted['clipping_range_m'][1]]
                fitted['human_cutaway_before_framed_bounds'] = True
                pixels, measured = _render(session, sensor, fitted, chosen)
                score = (sum(count >= 64 for count in measured['object_pixels'].values()),
                         sum(measured['object_pixels'].values()))
                if score > best[0]:
                    best = (score, pixels, measured)
        _, result[name], evidence[name] = best
        evidence[name]['intended_objects'] = [obj.name for obj in chosen]
        evidence[name]['view_purpose'] = profile['purpose']
        evidence[name]['framing_scope'] = 'task_actor_detail' if group is not None else 'full_task_context'
        if explicit is not None:
            evidence[name].update(explicit_detail_binding_group=list(explicit['binding_group']),
                explicit_detail_elevation_deg=explicit['elevation_deg'],
                framing_scope='explicit_task_binding_detail')
        if name in global_angles and name != 'global_overhead':
            evidence[name]['requested_azimuth_deg'] = global_angles[name]
            evidence[name]['azimuth_preserved'] = True
        if separate_context and group is None:
            evidence[name]['context_objects_in_separate_detail_views'] = sorted(separate_context)
            evidence[name]['framing_scope'] = 'task_targets_with_original_lighting_preserved'
    session.human_review_camera_evidence = {'strategy': 'category_bounds_and_native_visibility_v1',
        'profile': profile, 'views': evidence, 'native_objects': [obj.name for obj in objects],
        'object_issues': session.human_review_object_issues,
        'missing_task_objects': not bool(objects), 'robot_sensors_modified': False,
        'physics_steps': 0, 'actor_mutations': 0}
    return result

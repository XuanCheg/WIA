#!/usr/bin/env python3
"""Generate nearby multi-object layouts and review images without qualification.

Run inside the retained behavior allocation:
python src/wia/runtime/commands/generate_esi_rigid_multi.py --output-dir OUT --review-dir REVIEW
The plan is docs/rigid_containment_task_expansion_plan.json. Rendering is
sequential. Placement/capture failures retain the world for retry.json requests.
The final conference world remains available for conference_request.json swaps
and repairs. Close it only with an explicitly authorized operation: close request.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = WIA_RUNTIME_ROOT
DATA = ROOT / 'explore_to_achieve/scenes'
sys.path.insert(0, str(ROOT))
from wia.runtime.scene_review import SceneReview, fingerprint, read_json, scene_id, write_json


def prepare(output, review):
    plan = read_json(ROOT / 'docs/rigid_containment_task_expansion_plan.json')
    live = Path(read_json(Path.home() / '.codex/gpu-interactive.json')['scene_review']['conversion_root'])
    prior = read_json(live / 'run/conversion_summary.json')
    rows = [row for row in SceneReview(live, review).catalogue()['scenes'] if not row.get('targets')]
    assert len(rows) == plan['counts']['approved_base_layouts'] and all(row['decision'] == 'approved' for row in rows)
    sources = {row['source'] for row in rows}
    originals = [row for row in prior['results'] + prior.get('preserved_scenes', []) if row['source'] in sources]
    frozen = {row['scene_file']: hashlib.sha256(Path(row['scene_file']).read_bytes()).hexdigest() for row in rows}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'inputs/frozen_scene_sha256.json', frozen)
    (output / 'inputs/approved_reviews.json').write_bytes((review / 'reviews.json').read_bytes())
    cases = []
    for index, proposal in enumerate(plan['new_tasks']):
        case = output / 'cases' / f'{index:02d}'
        case.mkdir(parents=True)
        value = yaml.safe_load(Path(proposal['approved_scene_file']).read_text())
        value['id'] += '_multi'
        value['status'] = 'ready'
        value['target_bindings'] = [value['target']]
        targets = []
        for item in proposal['targets']:
            if item['source_role'] == 'containee':
                key = value['target']
            else:
                key = item['source_role']
                value['target_bindings'].append(key)
                value['qualification']['supported_objects'].append(key)
                value['bindings'][key] = {'name': 'multi_' + key, 'category': item['category'], 'model': item['model']}
                extent = np.asarray(item['initial_source_extent_m'])
                multiplier = min(1., .08 / float(np.min(extent))) if np.min(extent) > .1 else 1.
                patch = {'operation': 'add_dataset_object', 'phase': 'before_construction',
                    'name': 'multi_' + key, 'category': item['category'], 'model': item['model'],
                    'position': [150. + len(targets), 100., 100.],
                    'scale': (np.asarray(item['source_scale']) * multiplier).tolist(),
                    'quaternion_xyzw': [0., 0., 0., 1.]}
                value['scene']['variants']['tabletop']['patches'].append(patch)
            targets.append({'binding': key, 'category': item['category'], 'model': item['model'],
                'label': item['label'], 'original_scale': item['source_scale']})
        draft = case / 'draft.yaml'
        draft.write_text(yaml.safe_dump(value, sort_keys=False))
        scene_path = ROOT / proposal['proposed_scene_file']
        scene_path.parent.mkdir(parents=True, exist_ok=True)
        pending = dict(value, status='pending')
        scene_path.write_text(yaml.safe_dump(pending, sort_keys=False))
        task_file = ROOT / proposal['proposed_task_file']
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task = {'kind': 'rigid_containment_task', 'id': value['id'] + '_answer_and_place', 'version': 1,
            'status': 'pending', 'source': {'provider': 'esi', 'json': proposal['question_source']},
            'scene_file': str(scene_path.relative_to(DATA)), 'targets': value['target_bindings'],
            'goal': 'answer_and_place', 'allow_remove_after_verification': True,
            'public': {'objective': proposal['question'],
                'success_criteria': 'Answer for every named object and complete a real pickup and stable partial insertion for each. A verified object may be removed to free a shared container.',
                'environment': 'Retrieve the nearby objects using the real R1 robot.',
                'constraints': ['Complete all object subtasks before declaring DONE. An object resting entirely on top does not count as inside.']}}
        task_file.write_text(yaml.safe_dump(task, sort_keys=False))
        record = {'source': proposal['question_source'], 'scene_file': str(scene_path.relative_to(DATA)),
            'task_file': str(task_file), 'status': 'pending', 'targets': targets,
            'verification_level': 'nearby_placement_manual_review', 'physical_grasp_verified': False,
            'output': str(case), 'reason': 'Nearby objects and review images are being generated.'}
        write_json(case / 'case.json', {'proposal': proposal, 'record': record, 'draft': str(draft),
            'scene_file': str(scene_path), 'task_file': str(task_file)})
        cases.append(record)
    write_json(output / 'run/conversion_summary.json', {'results': cases, 'preserved_scenes': originals,
        'config': {'seed': 0}, 'status': 'running', 'qualification_performed': False})
    return cases


def support_height(raytest, support, point, lo, hi, *, seating=False):
    """Find a usable upward surface, excluding elevated seat arms and backs."""
    hits = raytest([*point, float(hi[2]) + .05], [*point, float(lo[2]) - .02], only_closest=False)
    ceiling = .65 if seating else 1.25
    valid = [hit for hit in hits if hit.get('hit') and hit['rigidBody'].startswith(support.prim_path + '/')
        and hit['normal'][2] > .4 and .3 < hit['position'][2] < ceiling]
    return float(min(valid, key=lambda row: row['distance'])['position'][2]) if valid else None


def place_nearby(session, selected, *, different_tables=False, overrides=None):
    from wia.runtime.sim.esi_omnigibson import _array
    from omnigibson.utils.sampling_utils import raytest
    import torch

    definition = selected.definition
    compact = selected.scene_name == 'hall_conference_large'
    different_tables = different_tables or compact
    radius = 3.0 if different_tables else 1.5 if compact else 4.5
    primary = session.env.scene.object_registry('name', definition.bindings[definition.target].name)
    origin = _array(primary.get_position_orientation()[0])
    support_key = definition.qualification.object_supports.get(definition.target, definition.qualification.support)
    preferred = session.env.scene.object_registry('name', definition.bindings[support_key].name)
    categories = {'breakfast_table', 'coffee_table', 'console_table', 'desk', 'sofa', 'armchair',
        'eames_chair', 'straight_chair', 'grocery_shelf', 'shelf', 'cabinet', 'bench'}
    supports = [preferred] + [obj for obj in session.env.scene.objects if obj is not preferred
        and obj.category in categories and not obj.visual_only
        and selected.room_name in getattr(obj, 'in_rooms', [])
        and obj.name != definition.bindings[definition.qualification.support].name]
    if different_tables:
        table_categories = {'breakfast_table', 'coffee_table', 'console_table', 'desk', 'conference_table',
            'dining_table', 'folding_table'}
        supports = [obj for obj in supports if obj is not preferred and obj.category in table_categories]
    supports[1:] = sorted(supports[1:], key=lambda obj: float(np.linalg.norm(_array(obj.get_position_orientation()[0])[:2] - origin[:2])))
    occupied = [(_array(obj.aabb[0]), _array(obj.aabb[1])) for obj in session.env.scene.objects
        if obj.name in {binding.name for key, binding in definition.bindings.items()
            if key == definition.target or key.startswith('container_')}]
    placements = []
    used_supports = {preferred.name}
    for key in definition.target_bindings:
        if key == definition.target:
            continue
        obj = session.env.scene.object_registry('name', definition.bindings[key].name)
        low, high = (_array(value) for value in obj.aabb)
        extent = high - low
        position, quaternion = (_array(value) for value in obj.get_position_orientation())
        offset = position - (low + high) / 2
        placed = None
        policy = (overrides or {}).get(key, {})
        # Prefer a different native surface for each object. Retain the primary
        # surface as a fallback when room furniture offers no usable proposal.
        ranked = sorted(supports, key=lambda support: ((support is not preferred) if compact else support.name in used_supports,
            float(np.linalg.norm(_array(support.get_position_orientation()[0])[:2] - origin[:2]))))
        if policy.get('support_name'):
            ranked = [support for support in supports if support.name == policy['support_name']]
        for support in ranked:
            lo, hi = (_array(value) for value in support.aabb)
            seating = support.category in {'sofa', 'armchair', 'eames_chair', 'straight_chair', 'bench'}
            half = extent[:2] / 2 + float(policy.get('inset_m', .10 if seating else .06))
            if np.any(hi[:2] - lo[:2] < 2 * half):
                continue
            points = [np.array([x, y]) for x in np.linspace(lo[0] + half[0], hi[0] - half[0], 9)
                for y in np.linspace(lo[1] + half[1], hi[1] - half[1], 9)]
            reference = (lo[:2] + hi[:2]) / 2 if seating or policy.get('centre') else origin[:2]
            points.sort(key=lambda point: np.linalg.norm(point - reference))
            for point in points:
                if np.linalg.norm(point - origin[:2]) > radius:
                    continue
                if any(np.all(point + half > bound_low[:2] - .025) and
                       np.all(point - half < bound_high[:2] + .025) for bound_low, bound_high in occupied):
                    continue
                # The center and footprint corners must lie over the same usable
                # surface. Cushion curvature is allowed; no flat-plane test.
                heights = [support_height(raytest, support, point + np.asarray(delta) * extent[:2] / 2,
                    lo, hi, seating=seating) for delta in [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]]
                if any(height is None for height in heights) or max(heights) - min(heights) > .08:
                    continue
                centre = np.r_[point, max(heights) + extent[2] / 2 + .003]
                obj.set_position_orientation(position=torch.tensor(centre + offset, dtype=torch.float32),
                    orientation=torch.tensor(quaternion, dtype=torch.float32))
                obj.keep_still()
                occupied.append((centre - extent / 2, centre + extent / 2))
                placed = {'binding': key, 'support_name': support.name,
                    'position': (centre + offset).tolist(), 'quaternion_xyzw': quaternion.tolist(),
                    'distance_from_primary_m': float(np.linalg.norm(point - origin[:2])),
                    'proposal_radius_m': radius, 'support_policy': 'different_adjacent_table' if different_tables
                        else 'compact_conference' if compact else 'distinct_room_surfaces',
                    'inset_m': float(policy.get('inset_m', .10 if seating else .06)),
                    'footprint_support_heights_m': heights}
                placements.append(placed)
                used_supports.add(support.name)
                break
            if placed:
                break
        if placed is None:
            raise ValueError('No nearby support proposal for ' + key)
    for _ in range(30 if overrides else 5):
        session.pipeline.og.sim.step()
    for placement in placements:
        obj = session.env.scene.object_registry('name', definition.bindings[placement['binding']].name)
        position, quaternion = (_array(value) for value in obj.get_position_orientation())
        placement['position'] = position.tolist()
        placement['quaternion_xyzw'] = quaternion.tolist()
    return placements


def finish_loaded(session, selected, case, review, *, retain=True, sample=True, different_tables=False):
    from core.record.images import save_png
    from wia.tasks.loader import load_scene
    from wia.runtime.sim.esi_omnigibson import _array
    from wia.runtime.commands.capture_esi_scene_reviews import capture_target_view
    from core.viewer_config import ViewerConfig
    from wia.runtime.sim.esi_robot_cameras import apply_head_camera_mount

    if selected.scene_name == 'hall_conference_large' and run_retained_maintenance(case, review):
        return
    info = read_json(case / 'case.json')
    repair_request = case / 'placement_request.json'
    repair = read_json(repair_request, {})
    if repair.get('different_tables'):
        sample = different_tables = True
    if repair.get('objects'):
        sample = True
    previous = read_json(case / 'result.json', info['record'])
    placements = place_nearby(session, selected, different_tables=different_tables,
        overrides=repair.get('objects')) if sample else previous['placements']
    value = selected.definition.model_dump(mode='json', by_alias=True)
    for placement in placements if sample else []:
        key = placement['binding']
        obj = session.env.scene.object_registry('name', value['bindings'][key]['name'])
        position, quaternion = (_array(v).tolist() for v in obj.get_position_orientation())
        value['scene']['variants']['tabletop']['patches'].append({'operation': 'object_pose',
            'phase': 'after_postprocessing', 'binding': key, 'position': position,
            'quaternion_xyzw': quaternion, 'jitter_xy_m': [0., 0.]})
        support_key = 'multi_support_' + key
        value['bindings'][support_key] = {'name': placement['support_name']}
        value['qualification']['object_supports'][key] = support_key
        patch = next(item for item in value['scene']['variants']['tabletop']['patches']
            if item['operation'] == 'add_dataset_object' and item['name'] == value['bindings'][key]['name'])
        patch['position'] = position
        patch['quaternion_xyzw'] = quaternion
    path = Path(info['scene_file'])
    if selected.scene_name == 'hall_conference_large':
        value['runtime']['robot_camera_config'] = 'configs/esi_robot_cameras_conference_head.json'
        apply_head_camera_mount(session, read_json(ROOT / value['runtime']['robot_camera_config']))
    if sample:
        path.write_text(yaml.safe_dump(value, sort_keys=False))
    final = load_scene(path, data_root=DATA)
    if session.scene_camera is None:
        session.configure_scene_camera()
    if not session.global_cameras:
        session.configure_human_cameras(ViewerConfig.load(ROOT / final.definition.runtime.viewer_config))
    for _ in range(9):
        session.pipeline.og.sim.render()
    raw = session.observe(include_wrist=True, include_scene=True)
    views = {'head': raw['head_rgb'], 'wrist': raw['wrist_rgb'], 'scene': raw['scene_rgb']}
    views.update({name: session._sensor_rgb(sensor) for name, sensor in session.global_cameras.items()})
    locations = []
    for index, key in enumerate(final.definition.target_bindings):
        chosen = replace(final, definition=final.definition.model_copy(update={'target': key}))
        pixels, location = capture_target_view(session, chosen)
        name = 'global_target' if index == 0 else f'global_target_extra{index}'
        views[name] = pixels
        locations.append(location)
    images = {}
    for name, pixels in views.items():
        filename = case / 'views' / (name + '.png')
        filename.parent.mkdir(exist_ok=True)
        save_png(filename, pixels)
        images[name] = str(filename)
    record = dict(info['record'] if sample else previous, status='ready', reason='', placements=placements,
        candidate_file=str(path), qualification_performed=False)
    task_path = Path(info['task_file'])
    task = yaml.safe_load(task_path.read_text())
    task['status'] = 'ready'
    task_path.write_text(yaml.safe_dump(task, sort_keys=False))
    key = scene_id(record['source'], 'tabletop', final.seed)
    receipt = {'schema': 'scene_review_preview_v1', 'id': key,
        'fingerprint': fingerprint(path, DATA / record['source'], 'tabletop', final.seed),
        'variant': 'tabletop', 'seed': final.seed, 'images': images,
        'target_location': locations[0], 'target_locations': locations,
        'provenance': 'Nearby support placement for manual review; no physical qualification was performed.',
        'model_calls': 0, 'robot_actions': 0, 'physical_grasp_verified': False,
        'qualification_performed': False}
    write_json(review / 'captures' / key / 'preview_complete.json', receipt)
    write_json(case / 'result.json', record)
    write_json(case / 'worker_state.json', {'status': 'complete', 'pid': os.getpid()})
    if repair_request.is_file():
        repair_request.replace(case / ('placement_request_' + str(time.time_ns()) + '.json'))
    if retain and selected.scene_name == 'hall_conference_large':
        retain_conference(session, selected, case, review)


def run_retained_maintenance(case, review):
    """Run sequential preview repairs while the retained conference world is idle."""
    root = case.parents[1]
    request = root / 'maintenance_request.json'
    if not request.is_file():
        return False
    command = read_json(request)
    request.replace(root / ('maintenance_request_' + str(time.time_ns()) + '.json'))
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
    state = {'status': 'probed', 'pid': os.getpid(), 'job_id': os.environ.get('PBS_JOBID'),
        'gpu_memory_mib': gpu, 'conference_retained': True, 'completed': []}
    write_json(root / 'maintenance_state.json', state)
    if command.get('operation') == 'gpu_probe':
        return True
    if command.get('operation') != 'repair_cases':
        raise ValueError('Unknown retained maintenance operation.')
    for index in command['cases']:
        start = time.monotonic()
        target = root / 'cases' / f'{int(index):02d}'
        info = read_json(target / 'case.json')
        if info['proposal']['scene'] == 'hall_conference_large':
            raise ValueError('Conference repairs must use the retained world directly.')
        state.update(status='repairing', current=int(index))
        write_json(root / 'maintenance_state.json', state)
        with (target / 'repair_native.log').open('a') as log:
            code = subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output-dir', str(root), '--review-dir', str(review), '--worker', str(index), '--repair'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError('Native repair failed for case ' + str(index))
        state['completed'].append(int(index))
        state.setdefault('wall_s_by_case', {})[str(index)] = time.monotonic() - start
        publish_retained_results(root, int(case.name))
    state.update(status='complete')
    write_json(root / 'maintenance_state.json', state)
    return True


def publish_retained_results(root, current):
    summary = read_json(root / 'run/conversion_summary.json')
    for index in range(len(summary['results'])):
        result = root / 'cases' / f'{index:02d}' / 'result.json'
        if result.is_file():
            summary['results'][index] = read_json(result)
    completed = sum(row['status'] == 'ready' for row in summary['results'])
    summary['status'] = 'complete' if completed == len(summary['results']) else 'running'
    summary['conference_background_reused'] = bool(read_json(root / 'conference_reuse.json', {}))
    summary['simulator_retained'] = True
    write_json(root / 'run/conversion_summary.json', summary)
    write_json(root / 'status.json', {'status': summary['status'], 'completed': completed,
        'current': current, 'simulator_retained': True, 'pid': os.getpid(),
        'job_id': os.environ.get('PBS_JOBID'), 'updated_at': datetime.now(timezone.utc).isoformat()})


def reframe_existing_cameras(session):
    from wia.runtime.sim.esi_omnigibson import _array
    from wia.runtime.sim.esi_debug_cameras import camera_orientation
    positions = [_array(session.env.scene.object_registry('name', name).get_position_orientation()[0])
        for name in session.task_object_names]
    base = _array(session.robot.get_position_orientation()[0])
    target = np.mean(positions, axis=0)
    target[:2] = (2 * target[:2] + base[:2]) / 3
    target[2] = max(target[2], base[2] + .65)
    for entry in session.viewer_config.global_views:
        position = target + entry.offset_m
        session.global_cameras[entry.name].set_position_orientation(position=position.tolist(),
            orientation=camera_orientation(position, target).tolist(), frame='world')
    front = session.global_cameras['global_front']
    position, quaternion = front.get_position_orientation()
    session.pipeline.set_camera(_array(position), _array(quaternion))


def switch_conference_task(session, previous, selected):
    """Replace only explicit task geometry in a compatible retained background."""
    from copy import deepcopy
    from omnigibson.objects import REGISTERED_OBJECTS
    from omnigibson.utils.python_utils import create_class_from_registry_and_config
    from wia.runtime.sim.esi_loading import native_loading_context
    from wia.runtime.sim.esi_robot_cameras import apply_head_camera_mount
    from wia.tasks.scene_only import SceneOnly
    import torch

    pipeline, module = session.pipeline, session.task_module
    if (previous.scene_name, previous.room_name) != (selected.scene_name, selected.room_name):
        raise ValueError('Retained preview worlds require the same scene and room.')
    if previous.path == selected.path and previous.file_sha256 == selected.file_sha256:
        return
    payload = pipeline.select_question(selected.source_path, 0, None).payload
    base_objects = module.build_env_objects(payload)
    base_names = tuple(obj['name'] for obj in base_objects)
    config = pipeline.task_build_env_config(module, selected.scene_name, selected.room_name,
        'R1', base_objects, full_scene=bool(getattr(module, 'FULL_SCENE', False)), payload=payload)
    old_config = pipeline.task_build_env_config(module, previous.scene_name, previous.room_name,
        'R1', [], full_scene=bool(getattr(module, 'FULL_SCENE', False)), payload=session.payload)
    if config['scene'] != old_config['scene']:
        raise ValueError('Native background filtering/configuration differs between tasks.')
    adapter = SceneOnly(selected)
    settings = adapter.observation_settings()
    session.robot_position = settings.get('robot_position')
    session.robot_yaw_deg = settings.get('robot_yaw_deg')
    session.head_pitch_deg = settings.get('head_pitch_deg', 0.)
    session.camera_size = tuple(settings['camera_size'])
    session._configure_robot_observation(config)
    adapter.apply_to_env_config(config)
    new_names = {obj['name'] for obj in config['objects']}
    old_names = set(session.task_object_names) | {patch.name for patch in
        previous.definition.scene.variants[previous.variant].patches if patch.operation.startswith('add_')}
    names = getattr(session, '_multi_runtime_task_objects', set()) | old_names | new_names
    session._multi_runtime_task_objects = names
    objects = [obj for obj in session.env.scene.objects if obj.name in names]
    with native_loading_context():
        pipeline.og.sim.batch_remove_objects(objects)
        added = []
        for item in config['objects']:
            specification = deepcopy(item)
            position = specification.pop('position', None)
            orientation = specification.pop('orientation', None)
            obj = create_class_from_registry_and_config(cls_name=specification['type'],
                cls_registry=REGISTERED_OBJECTS, cfg=specification, cls_type_descriptor='object')
            added.append((obj, position, orientation))
        pipeline.og.sim.batch_add_objects([row[0] for row in added], [session.env.scene] * len(added))
        for obj, position, orientation in added:
            obj.set_position_orientation(position=position, orientation=orientation, frame='scene')
        # Initialize the imported task geometry; the background is retained.
        pipeline.og.sim.step()
    robot = config['robots'][0]
    session.robot.reset()
    session.robot.set_position_orientation(position=torch.tensor(robot['position'], dtype=torch.float32),
        orientation=torch.tensor(robot['orientation'], dtype=torch.float32), frame=robot.get('pose_frame', 'world'))
    session.scene_adapter = adapter
    session.payload = payload
    session.metadata_path = session.question_path = selected.source_path
    session.task_state = pipeline.task_preprocess(module, payload, selected.source_path, None)
    session.task_object_names = base_names
    session._multi_runtime_task_objects = new_names
    adapter.apply_after_postprocessing(session)
    config_path = ROOT / selected.definition.runtime.robot_camera_config
    camera_config = read_json(config_path)
    if camera_config.get('head_camera') is None:
        raise ValueError('Retained conference previews require an explicit head mount.')
    apply_head_camera_mount(session, camera_config)
    session.robot_camera_config = camera_config
    session.invalidate_camera_cache()
    session._observation_source = None
    reframe_existing_cameras(session)


def retain_conference(session, selected, case, review):
    """Keep the world for same-room preview swaps, warm repairs and user inspection."""
    from wia.tasks.loader import load_scene
    root = case.parents[1]
    current = int(case.name)
    publish_retained_results(root, current)
    summary = read_json(root / 'run/conversion_summary.json')
    queued = [index for index, row in enumerate(summary['results']) if row['status'] != 'ready'
        and Path(row['source']).parts[-3:-1] == (selected.scene_name, selected.room_name)]
    request = root / 'conference_request.json'
    while True:
        write_json(root / 'conference_warm_state.json', {'status': 'retained', 'pid': os.getpid(),
            'current': current, 'scene': selected.scene_name, 'room': selected.room_name,
            'world_retained': True, 'request_file': str(request)})
        while not queued and not request.is_file():
            time.sleep(.5)
        command = {'case': queued.pop(0)} if queued else read_json(request)
        if request.is_file():
            request.replace(root / ('conference_request_' + str(time.time_ns()) + '.json'))
        if command.get('operation') == 'close':
            write_json(root / 'conference_warm_state.json', {'status': 'closing', 'pid': os.getpid(),
                'current': current, 'world_retained': False})
            return
        index = int(command.get('case', current))
        target_case = root / 'cases' / f'{index:02d}'
        info = read_json(target_case / 'case.json')
        try:
            module = importlib.reload(importlib.import_module('wia.runtime.commands.generate_esi_rigid_multi'))
            scene_path = info['scene_file'] if (target_case / 'result.json').is_file() else info['draft']
            next_selected = load_scene(scene_path, data_root=DATA)
            start = time.monotonic()
            module.switch_conference_task(session, selected, next_selected)
            module.finish_loaded(session, next_selected, target_case, review, retain=False,
                sample=bool(command.get('resample')) or not (target_case / 'result.json').is_file(),
                different_tables=bool(command.get('different_tables')))
            receipt = {'background_reused': True, 'from_case': current,
                'to_case': index, 'wall_s': time.monotonic() - start, 'pid': os.getpid(),
                'job_id': os.environ.get('PBS_JOBID'), 'independent_restart': False,
                'scope': 'Human scene previews only; no Agent episode or physical qualification.'}
            history = read_json(root / 'conference_reuse_history.json', [])
            write_json(root / 'conference_reuse_history.json', history + [receipt])
            write_json(root / 'conference_reuse.json', receipt)
            selected, case, current = next_selected, target_case, index
            module.publish_retained_results(root, current)
        except Exception as error:
            write_json(root / 'conference_warm_state.json', {'status': 'waiting', 'pid': os.getpid(),
                'current': index, 'error': f'{type(error).__name__}: {error}', 'world_retained': True})
            print('[multi-review] Conference world retained after swap/capture error:', error, flush=True)
            while not request.is_file():
                time.sleep(.5)


def worker(args):
    from wia.tasks.loader import load_scene
    from wia.tasks.scene_only import SceneOnly
    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras

    case = args.output_dir / 'cases' / f'{args.worker:02d}'
    info = read_json(case / 'case.json')
    write_json(case / 'worker_state.json', {'status': 'loading', 'pid': os.getpid(),
        'repair': args.repair, 'job_id': os.environ.get('PBS_JOBID')})
    selected = load_scene(info['scene_file'] if args.repair else info['draft'], data_root=DATA)
    session = ESISession(selected.source_path, **SceneOnly(selected).observation_settings())
    session.scene_adapter = SceneOnly(selected)
    try:
        session.reset()
        runtime = selected.definition.runtime
        configure_robot_cameras(session, ROOT / runtime.robot_camera_config, resolution=runtime.camera_size)
        while True:
            try:
                module = importlib.reload(importlib.import_module('wia.runtime.commands.generate_esi_rigid_multi'))
                module.finish_loaded(session, selected, case, args.review_dir)
                return 0
            except Exception as error:
                write_json(case / 'worker_state.json', {'status': 'waiting', 'pid': os.getpid(),
                    'error': f'{type(error).__name__}: {error}', 'world_retained': True})
                print('[multi-review] World retained:', error, flush=True)
                request = case / 'retry.json'
                while not request.is_file():
                    time.sleep(.5)
                request.replace(case / ('retry_' + str(time.time_ns()) + '.json'))
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--repair', action='store_true', help='Repair the existing ready layout in place.')
    args = parser.parse_args()
    if not os.environ.get('PBS_JOBID'):
        parser.error('Use the retained interactive allocation.')
    if args.worker is not None:
        return worker(args)
    if not args.resume:
        prepare(args.output_dir, args.review_dir)
    summary = read_json(args.output_dir / 'run/conversion_summary.json')
    order = sorted(range(len(summary['results'])), key=lambda index:
        'hall_conference_large' in summary['results'][index]['source'])
    for index in order:
        case = args.output_dir / 'cases' / f'{index:02d}'
        if (case / 'result.json').is_file():
            summary['results'][index] = read_json(case / 'result.json')
            continue
        write_json(args.output_dir / 'status.json', {'status': 'running', 'current': index,
            'completed': sum(record['status'] == 'ready' for record in summary['results']),
            'job_id': os.environ['PBS_JOBID'], 'updated_at': datetime.now(timezone.utc).isoformat()})
        with (case / 'native.log').open('a') as log:
            code = subprocess.call([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output-dir', str(args.output_dir), '--review-dir', str(args.review_dir), '--worker', str(index)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError('Native rendering failed for case ' + str(index))
        summary['results'][index] = read_json(case / 'result.json')
        write_json(args.output_dir / 'run/conversion_summary.json', summary)
    summary['status'] = 'complete'
    write_json(args.output_dir / 'run/conversion_summary.json', summary)
    write_json(args.output_dir / 'status.json', {'status': 'complete', 'completed': len(summary['results'])})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

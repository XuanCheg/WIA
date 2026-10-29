"""Category-neutral native review capture using existing ESI scene interfaces.

Run through src/wia/runtime/commands/run_esi_capture_worker.py with local or assigned PBS GPUs.
Human cameras and closeups are review evidence, never Agent observations.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from wia.runtime.scene_capture_queue import PROTOCOL, sha256, verify_snapshot, write_json
from wia.tasks.loader import LoadedRigidTask, LoadedScene, UniqueLoader, load_task
from wia.tasks.schema import SceneDefinition


def install_gpu_selection(gpu):
    """Set renderer Vulkan ordinal separately from visible CUDA ordinal zero."""
    import isaacsim

    original = isaacsim.SimulationApp

    def isolated_app(config=None, *args, **kwargs):
        config = dict(config or {})
        config.update(active_gpu=gpu["renderer_index"], physics_gpu=0, multi_gpu=False)
        config["extra_args"] = [*config.get("extra_args", []),
            f"--/renderer/activeGpu={gpu['renderer_index']}",
            "--/renderer/multiGpu/enabled=false",
            f"--/renderer/multiGpu/activeGpus={gpu['renderer_index']}"]
        if os.environ.get("OMNIGIBSON_APPDATA_PATH"):
            appdata = Path(os.environ["OMNIGIBSON_APPDATA_PATH"])
            appdata.mkdir(parents=True, exist_ok=True)
            config["extra_args"].extend([
                f"--/app/tokens/cache={appdata / 'kit_cache'}",
                f"--/app/tokens/data={appdata / 'kit_data'}",
                f"--/app/tokens/logs={appdata / 'kit_logs'}",
                "--/plugins/carb.tasking.plugin/threadCount=8",
                "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8",
                "--/exts/omni.physx/numThreads=4"])
        return original(config, *args, **kwargs)

    isaacsim.SimulationApp = isolated_app


def isolate_texture_cache(simulator, appdata):
    """Redirect native startup cleanup to this worker's cache, preserving peers."""
    from types import SimpleNamespace

    original = simulator.shutil
    shared = (Path.home() / ".cache/ov/texturecache").resolve()
    private = Path(appdata).resolve() / "xdg_cache/ov/texturecache"

    def remove(path, *args, **kwargs):
        selected = private if Path(path).resolve() == shared else path
        return original.rmtree(selected, *args, **kwargs)

    simulator.shutil = SimpleNamespace(**{**vars(original), "rmtree": remove})


def gpu_evidence(session, gpu):
    import carb
    import torch

    settings = carb.settings.get_settings()
    active = settings.get("/renderer/activeGpu")
    multi = settings.get("/renderer/multiGpu/enabled")
    if active != gpu["renderer_index"] or multi:
        raise RuntimeError("Native renderer did not retain the assigned single GPU.")
    if torch.cuda.device_count() != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu["uuid"]:
        raise RuntimeError("CUDA device visibility does not match the owned GPU.")
    torch.empty(1, device="cuda:0")
    result = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
        "--format=csv,noheader,nounits"], text=True)
    used = {line.split(",")[1].strip() for line in result.splitlines()
            if line.split(",")[0].strip() == str(os.getpid())}
    if used != {gpu["uuid"]}:
        raise RuntimeError("Native simulator GPU process identity could not be verified: " + str(used))
    return {**gpu, "renderer_active_gpu": active, "renderer_multi_gpu": bool(multi),
            "cuda_device_count": torch.cuda.device_count(), "cuda_device": "cuda:0",
            "physics_device": str(session.pipeline.og.sim.device),
            "native_process_pid": os.getpid(), "verified_process_gpu_uuids": sorted(used)}


def load_capture_scene(manifest):
    """Allow pending diagnostic scenes without qualifying or publishing them."""
    path = Path(manifest["snapshot_scene_file"])
    definition = SceneDefinition.model_validate(yaml.load(path.read_text(), Loader=UniqueLoader))
    source = Path(manifest["snapshot_source_file"])
    return LoadedScene(definition, path, source, sha256(path), sha256(source),
                       manifest["variant"], manifest["seed"], manifest["scene_model"], manifest["room"])


def _sealed_category_source_roles(task, selected, manifest):
    """Validate distinct sealed QA and native-scene inputs without substituting either."""
    from wia.tasks.loader import source_scene_room
    from wia.tasks.schema import Source

    task_identity = task.definition.source.json_path
    scene_identity = selected.definition.source.json_path
    if scene_identity != manifest['source_identity']:
        raise ValueError('The selected native scene must retain the manifest source identity.')
    members = manifest.get('capture_source_members')
    if not isinstance(members, list) or not members:
        raise ValueError('Separate category QA/native sources require explicit sealed capture members.')
    declared_root = Path(manifest['snapshot_data_root'])
    root = declared_root.resolve()
    sealed = {}
    for member in members:
        if not isinstance(member, dict):
            raise ValueError('Each sealed capture source member must be an explicit identity object.')
        identity = member.get('source_identity')
        Source.model_validate({'provider': 'esi', 'json': identity})
        if str(Path(identity)) != identity or identity in sealed:
            raise ValueError('Sealed capture source identities must be canonical and appear exactly once.')
        declared = member.get('snapshot_source_file')
        if not isinstance(declared, str) or not declared:
            raise ValueError('Each capture source member requires its exact snapshot source file.')
        source = Path(declared)
        expected = declared_root / identity
        if source != expected or not source.resolve().is_relative_to(root) or not source.is_file():
            raise ValueError('Capture source members must use their exact existing sealed data-root identity paths.')
        actual_hash = sha256(source)
        if (actual_hash != member.get('source_sha256')
                or actual_hash != manifest['input_hashes'].get('data/' + identity)):
            raise ValueError('Sealed capture source member hash does not match the manifest.')
        sealed[identity] = (source.resolve(), actual_hash)
    for identity, loaded in ((task_identity, task), (scene_identity, selected)):
        if (identity not in sealed or sealed[identity] != (loaded.source_path.resolve(), loaded.source_sha256)):
            raise ValueError('The actual category QA/native source path and hash must match its own sealed member.')
    native_location = source_scene_room(scene_identity, json.loads(selected.source_path.read_text(encoding='utf-8')))
    if native_location != (selected.scene_name, selected.room_name):
        raise ValueError('The sealed native source metadata must match its selected scene and room.')
    return {'task_source_identity': task_identity, 'task_source_sha256': task.source_sha256,
            'scene_source_identity': scene_identity, 'scene_source_sha256': selected.source_sha256}


def bind_capture_inputs(session, selected, manifest):
    """Bind the frozen actual task before native reset, for cold and resident jobs."""
    path = manifest.get('snapshot_task_file')
    source_roles = {}
    if path:
        path = Path(path).resolve()
        snapshot = Path(manifest['snapshot_dir']).resolve()
        if not path.is_relative_to(snapshot) or path != snapshot / 'task.yaml':
            raise ValueError('Capture task must be the sealed snapshot task.yaml.')
        actual_hash = sha256(path)
        if actual_hash != manifest['input_hashes'].get('task.yaml') or actual_hash != manifest.get('task_sha256'):
            raise ValueError('Frozen capture task hash does not match its manifest.')
        task = load_task(path, data_root=manifest['snapshot_data_root'], variant=manifest['variant'],
            seed=manifest['seed'], run_mode='diagnostic')
        scene_definition = task.scene.definition if isinstance(task, LoadedRigidTask) else task.definition
        separate_category_sources = (not isinstance(task, LoadedRigidTask)
            and getattr(task.definition, 'kind', None) == 'category_robot_task'
            and task.definition.source.json_path != selected.definition.source.json_path)
        if separate_category_sources:
            source_roles = _sealed_category_source_roles(task, selected, manifest)
            if ((selected.scene_name, selected.room_name, selected.variant, selected.seed) !=
                    (manifest['scene_model'], manifest['room'], manifest['variant'], manifest['seed'])):
                raise ValueError('The separate-source native scene must retain its exact manifest scene/room/variant/seed.')
            sources_match = True
        else:
            sources_match = (task.source_path == selected.source_path.resolve()
                and task.source_sha256 == selected.source_sha256
                and scene_definition.source.json_path == manifest['source_identity'])
        if (not sources_match or
                (task.scene_name, task.room_name, task.variant, task.seed) !=
                (selected.scene_name, selected.room_name, selected.variant, selected.seed)):
            raise ValueError('Frozen capture task/source/scene/variant/seed do not agree.')
        if (task.definition.defaults.seed != manifest['seed'] or selected.definition.seed != manifest['seed'] or
                scene_definition.scene.variants[task.variant].model_dump(mode='json') !=
                selected.definition.scene.variants[selected.variant].model_dump(mode='json')):
            raise ValueError('Frozen task and scene initialization or declared seeds differ.')
        if manifest.get('physical_task_id') and task.definition.id != manifest['physical_task_id']:
            raise ValueError('Frozen task identity does not match the capture manifest.')
        session.loaded_task = task
    elif manifest.get('capture_initial_agent_bundle') is True:
        raise ValueError('Initial Agent bundle requires the frozen actual task definition.')
    session._capture_release_edge_policy = manifest.get('current_support_edge_policy') is True
    session.capture_initial_agent_bundle = manifest.get('capture_initial_agent_bundle') is True
    session.capture_input_identity = {'job_id': manifest['job_id'], 'capture_signature': manifest['capture_signature'],
        'task_sha256': manifest.get('task_sha256'), 'seed': manifest['seed'], 'variant': manifest['variant'], **source_roles}


def _initial_agent_bundle(session):
    """Freeze one first-boundary native Global plus all eight mounted raw RGBs."""
    from core.robot_cameras import RGB_TO_VIEW
    from core.observations import freeze_rgb
    import numpy as np

    if getattr(session, '_capture_initial_agent_bundle_complete', False):
        observation = session._capture_initial_agent_bundle_observation
        observation.initial_overview.validate_identity(session.initial_overview_identity)
        return dict(session._capture_initial_agent_bundle_views)
    session.capture_initial_agent_bundle_evidence = {'available': False,
        'kind': 'frozen_initial_native_agent_input_bundle', 'robot_actions': 0, 'model_calls': 0}
    try:
        if getattr(session, 'robot', None) is None or getattr(session, 'loaded_task', None) is None:
            raise RuntimeError('Initial bundle requires the actual native robot and frozen loaded task.')
        clock = float(session.pipeline.og.sim.current_time)
        if not np.isfinite(clock):
            raise ValueError('Initial bundle native clock is unavailable.')
        source = session.observation_source
        if (getattr(source, 'snapshot_id', 0) or getattr(source, 'action_step', 0) or
                getattr(source, 'history', ()) or getattr(source, 'latest_snapshot', None) is not None):
            raise RuntimeError('Initial bundle must be acquired before any earlier public observation/action.')
        first = source.capture(action_boundary=True).observation
        observation = source._capture_initial_overview(first)
        snapshot = source.latest_snapshot
        views = {name: freeze_rgb(rgb) for name, rgb in snapshot.human_views().items()
            if name in RGB_TO_VIEW.values()}
        session._capture_initial_agent_bundle_views = views
        overview = observation.initial_overview
        if overview is None:
            raise RuntimeError('Frozen initial native Global is unavailable.')
        overview.validate_identity(session.initial_overview_identity)
        if overview.snapshot_id != observation.snapshot_id or overview.captured_at != observation.captured_at:
            raise RuntimeError('Frozen Global and mounted native snapshot boundaries differ.')
        views['initial_overview_rgb'] = overview.rgb
        missing = set(RGB_TO_VIEW.values()) - views.keys()
        if missing:
            raise RuntimeError('Initial bundle is missing native mounted views: ' + str(sorted(missing)))
        from wia.runtime.wia_capture_diagnostics import capture_initial_review_diagnostics
        diagnostics = capture_initial_review_diagnostics(session)
        if isinstance(diagnostics, dict):
            session.capture_initial_agent_bundle_evidence['review_diagnostics'] = {
                **diagnostics, 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step}
        after = float(session.pipeline.og.sim.current_time)
        if not np.isfinite(after) or after != clock:
            raise RuntimeError('Initial bundle capture advanced the native physics clock.')
        session._capture_initial_agent_bundle_observation = observation
        session._capture_initial_agent_bundle_complete = True
        session.capture_initial_agent_bundle_evidence.update(available=True, snapshot_id=observation.snapshot_id,
            action_step=observation.action_step, captured_at=observation.captured_at,
            reset_identity=dict(session.initial_overview_identity), native_time_s=clock,
            native_time_after_s=after, physics_steps=0, views=sorted(views),
            capture_input_identity=dict(getattr(session, 'capture_input_identity', {})))
        return dict(views)
    except Exception as error:
        session.capture_initial_agent_bundle_evidence['error'] = f'{type(error).__name__}: {error}'
        raise


def capture_loaded_views(session, selected, expected_views):
    """Reuse mounted sensors, native ESI viewer, global cameras and target camera."""
    from core.viewer_config import ViewerConfig
    from wia.runtime.commands.capture_esi_scene_reviews import capture_target_view

    initial = _initial_agent_bundle(session) if getattr(session, 'capture_initial_agent_bundle', False) else {}
    root = WIA_RUNTIME_ROOT
    if session.scene_camera is None:
        session.configure_scene_camera()
    if not session.global_cameras:
        session.configure_human_cameras(ViewerConfig.load(root / selected.definition.runtime.viewer_config),
                                       include_robot=getattr(session, 'robot', None) is not None)
    views = {}
    # Newly created native render products need multiple render boundaries.
    for _ in range(3):
        for _ in range(3):
            session.pipeline.og.sim.render()
        session.invalidate_camera_cache()
        views = dict(initial)
        if {"head", "wrist"} & set(expected_views) and not initial:
            if session.robot is None:
                raise RuntimeError("This native robot-free source cannot provide requested head/wrist views.")
            raw = session.observe(include_wrist="wrist" in expected_views, include_scene="scene" in expected_views)
            views.update({name: raw[key] for name, key in (("head", "head_rgb"), ("wrist", "wrist_rgb"),
                          ("scene", "scene_rgb")) if name in expected_views})
            from core.robot_cameras import ADDITIONAL_RGB_NAMES, RGB_TO_VIEW
            views.update({RGB_TO_VIEW[key]: raw[key] for key in ADDITIONAL_RGB_NAMES
                          if RGB_TO_VIEW[key] in expected_views})
        elif "scene" in expected_views:
            views["scene"] = session._sensor_rgb(session.scene_camera)
        views.update({name: session._sensor_rgb(sensor) for name, sensor in session.global_cameras.items()})
    framing = getattr(session, 'human_camera_framing', {}) or {}
    if framing.get('strategy') == 'whole_native_floor_bounds_v1':
        import numpy as np
        from omnigibson.utils.usd_utils import PoseAPI
        from wia.runtime.sim.esi_debug_cameras import camera_orientation
        from wia.runtime.sim.esi_omnigibson import _array
        PoseAPI.invalidate()
        for name, sensor in session.global_cameras.items():
            expected = framing['views'][name]
            position, quaternion = (_array(value) for value in sensor.get_position_orientation())
            desired = camera_orientation(expected['position_world_m'], expected['target_world_m'])
            error_m = float(np.linalg.norm(position - expected['position_world_m']))
            cosine = abs(float(np.dot(quaternion / np.linalg.norm(quaternion), desired)))
            error_deg = float(np.degrees(2 * np.arccos(np.clip(cosine, 0., 1.))))
            expected.update(actual_position_world_m=position.tolist(),
                actual_quaternion_xyzw=quaternion.tolist(), position_error_m=error_m,
                rotation_error_deg=error_deg, capture_render_boundaries=9,
                capture_protocol='native_human_render_products_drained_v2')
            if error_m > .001 or error_deg > .1:
                raise RuntimeError('Native human camera pose differs from its fitted whole-floor view: ' + name)
    target_location = None
    if 'global_floorplan' in expected_views:
        from wia.runtime.sim.esi_floorplan_review import render_floorplan_review
        views['global_floorplan'], session.human_floorplan_metadata = render_floorplan_review(session)
    if "global_target" in expected_views:
        views["global_target"], target_location = capture_target_view(session, selected)
    extras = [name for name in expected_views if name.startswith("global_target_extra")]
    if extras:
        from dataclasses import replace
        bindings = selected.definition.target_bindings or (selected.definition.target,)
        for view in extras:
            index = int(view[-1])
            definition = selected.definition.model_copy(update={"target": bindings[index]})
            views[view], _ = capture_target_view(session, replace(selected, definition=definition))
    review_evidence = None
    if any(name.startswith('object_') for name in expected_views):
        from wia.runtime.sim.esi_review_cameras import capture_review_views
        review_names = [name for name in expected_views if name.startswith('object_') or
                       (name in session.global_cameras and framing.get('strategy') != 'whole_native_floor_bounds_v1')]
        views.update(capture_review_views(session, selected, review_names))
        review_evidence = session.human_review_camera_evidence
    if any(name.startswith('phase_') for name in expected_views):
        controller = getattr(session, 'temporal_phase_controller', None)
        if controller is None or controller.phase != 'phase_1':
            raise RuntimeError('Two-phase previews require the original initialized Phase 1.')
        source = session.observation_source
        first = (session._capture_initial_agent_bundle_observation if initial else
            source.capture(action_boundary=True).observation)
        views.update(phase_1_head=first.agentview_rgb, phase_1_wrist=first.wrist_rgb)
        session.human_temporal_frames = {key: views[key] for key in ('phase_1_head', 'phase_1_wrist')}
        if review_evidence:
            for name in ('object_overview', 'object_top', 'object_side'):
                views['phase_1_' + name] = views[name]
        # This is a human diagnostic render, with no Agent execution, answer or
        # success credit. Both frames use the real continuing native world.
        controller.prepared_boundary = (first.snapshot_id, first.action_step)
        controller.advance()
        for _ in range(9):
            session.pipeline.og.sim.render()
        session.invalidate_camera_cache()
        second = source.capture(action_boundary=True).observation
        views.update(phase_2_head=second.agentview_rgb, phase_2_wrist=second.wrist_rgb,
            phase_2_global_diagonal=session._sensor_rgb(session.global_cameras['global_diagonal']))
        if review_evidence:
            names = ['object_overview', 'object_top', 'object_side', 'global_diagonal']
            second_views = capture_review_views(session, selected, names, fixed_views=review_evidence['views'])
            views.update({'phase_2_' + name: pixels for name, pixels in second_views.items()})
            session.human_review_camera_evidence = {'phase_1': review_evidence,
                'phase_2': session.human_review_camera_evidence, 'phase_comparison_same_camera_poses': True}
        session.human_temporal_frames.update({key: value for key, value in views.items() if key.startswith('phase_')})
        session.temporal_capture_evidence = {'kind': 'human_native_two_phase_preview',
            'phase_1_snapshot_id': first.snapshot_id, 'phase_2_snapshot_id': second.snapshot_id,
            'same_robot_and_furniture': controller.transition['world_unchanged'],
            'transition_physics_steps': controller.transition['physics_steps'],
            'robot_actions': 0, 'autonomous_reasoning_or_navigation_tested': False,
            'mounted_view_coverage_requires_review': True}
        if controller.parameters.get('survey_revision') == 'free_survey_10_v1' and review_evidence:
            from wia.tasks.interaction_setup import array
            session.temporal_capture_evidence['phase_2_native_actors'] = {
                key: {'visible': bool(controller.objects[key].visible),
                    'position_m': array(controller.objects[key].get_position_orientation()[0]).tolist(),
                    'aabb_m': [array(v).tolist() for v in controller.objects[key].aabb],
                    'expected_onstage_position_m': controller.parameters['source_objects'][key]['position'],
                    'active': key in controller.active.values()}
                for key in controller.contents}
            pixels = session.human_review_camera_evidence['phase_2']['views']['object_top']['object_pixels']
            active = {key for key in controller.active.values() if key is not None}
            checks = {key: (pixels[controller.objects[key].name] > 0 if key in active else
                           pixels[controller.objects[key].name] == 0) for key in controller.contents}
            session.temporal_capture_evidence['phase_2_rendered_actor_checks'] = checks
            session.temporal_capture_evidence['phase_2_rendered_actor_visibility_valid'] = all(checks.values())
    if set(expected_views) - views.keys():
        raise RuntimeError("Requested native views are unavailable: " + str(set(expected_views) - views.keys()))
    return {name: views[name] for name in expected_views}, target_location


def preserve_partial_views(session, selected, expected_views, output):
    """Try each available human camera independently after a load/capture error."""
    from core.record.images import save_png
    from core.viewer_config import ViewerConfig

    output = Path(output)
    errors, images = {}, {}
    if getattr(session, 'env', None) is None:
        write_json(output / 'partial_capture.json', {'complete': False, 'eligible_final_layout': False,
            'images': {}, 'errors': {'initialization': 'No initialized native environment is available.'}})
        return
    try:
        if session.scene_camera is None:
            session.configure_scene_camera()
        if not session.global_cameras:
            root = WIA_RUNTIME_ROOT
            session.configure_human_cameras(ViewerConfig.load(root / selected.definition.runtime.viewer_config),
                include_robot=getattr(session, 'robot', None) is not None)
    except Exception as error:
        errors['camera_setup'] = f'{type(error).__name__}: {error}'
    sensors = dict(getattr(session, 'global_cameras', {}) or {})
    if getattr(session, 'scene_camera', None) is not None:
        sensors['scene'] = session.scene_camera
    robot = getattr(session, 'robot', None)
    if robot is not None:
        for view, attribute in (('head', 'camera_name'), ('wrist', 'wrist_camera_name')):
            name = getattr(session, attribute, None)
            if name in robot.sensors:
                sensors[view] = robot.sensors[name]
    for view in expected_views:
        try:
            path = output / (view + '.png')
            if path.is_file():
                images[view] = str(path)
                continue
            initial = getattr(session, '_capture_initial_agent_bundle_views', {})
            if getattr(session, 'capture_initial_agent_bundle', False) and view in initial:
                save_png(path, initial[view])
                images[view] = str(path)
                continue
            if getattr(session, 'capture_initial_agent_bundle', False):
                from core.robot_cameras import RGB_TO_VIEW
                if view in {'initial_overview_rgb', *RGB_TO_VIEW.values()}:
                    raise RuntimeError('Frozen initial native input is unavailable; live cameras cannot substitute.')
            if view in getattr(session, 'human_temporal_frames', {}):
                save_png(path, session.human_temporal_frames[view])
                images[view] = str(path)
                continue
            if view not in sensors:
                raise RuntimeError('Native camera unavailable after failure.')
            save_png(path, session._sensor_rgb(sensors[view]))
            images[view] = str(path)
        except Exception as error:
            errors[view] = f'{type(error).__name__}: {error}'
    write_json(output / 'partial_capture.json', {'complete': False, 'eligible_final_layout': False,
        'images': images, 'errors': errors, 'provenance': 'Partial native evidence after failure; no qualification claim.'})


def capture_snapshot(manifest_path, output_dir, gpu):
    """One fresh native world. Targeted ready scenes retain their legacy startup."""
    manifest = json.loads(Path(manifest_path).read_text())
    verify_snapshot(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    native_start = Path("/proc/self/stat").read_text().split(") ", 1)[1].split()[19]
    write_json(output / "native_process.json", {"pid": os.getpid(), "pgid": os.getpgrp(),
        "start_ticks": native_start, "manifest": str(manifest_path),
        "execution_mode": gpu.get("execution_mode", "pbs"), "pbs_job_id": gpu.get("pbs_job_id"),
        "capture_job_id": manifest["job_id"]})

    def progress(stage):
        write_json(output / "progress.json", {"stage": stage, "elapsed_s": time.monotonic() - began})

    progress("gpu_initialization")
    install_gpu_selection(gpu)
    if os.environ.get("OMNIGIBSON_APPDATA_PATH"):
        from omnigibson import simulator
        isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    selected = load_capture_scene(manifest)
    session = None
    receipt = {}
    initialization_wait_s = 0.

    def initialized(session, *, qualified=False):
        nonlocal initialization_wait_s
        write_json(output / "initialization_complete.json", {
            "schema": "esi_capture_initialization_v1", "job_id": manifest["job_id"],
            "candidate_id": manifest["candidate_id"], "capture_signature": manifest["capture_signature"],
            "code_commit": manifest["code_commit"], "esi_commit": manifest["esi_commit"],
            "scene_model": manifest["scene_model"], "room": manifest["room"],
            "gpu": gpu_evidence(session, gpu), "initialized_at": time.time(),
            "initialization_s": time.monotonic() - began, "native_qualification_run": qualified,
            "physical_grasp_verified": False, "model_calls": 0, "robot_actions": 0})
        release = os.environ.get("ESI_CAPTURE_RELEASE_FILE")
        if release:
            progress("simulator_ready")
            waiting = time.monotonic()
            while not Path(release).is_file():
                time.sleep(.5)
            value = json.loads(Path(release).read_text())
            if (value.get("job_id") != manifest["job_id"]
                    or value.get("capture_signature") != manifest["capture_signature"]):
                raise ValueError("Capture release does not identify this initialized snapshot.")
            initialization_wait_s = time.monotonic() - waiting

    def save(session, views, target_location=None, *, qualified=False, capture_began=None):
        from core.record.images import save_png
        capture_began = time.monotonic() if capture_began is None else capture_began
        evidence = gpu_evidence(session, gpu)
        selected.verify_unchanged()
        verify_snapshot(manifest)
        files = {}
        for name in manifest["expected_views"]:
            path = output / f"{name}.png"
            save_png(path, views[name])
            files[name] = str(path)
        receipt.update(schema="scene_review_preview_v1", id=manifest["candidate_id"],
            fingerprint=manifest["fingerprint"], capture_signature=manifest["capture_signature"],
            capture_protocol=PROTOCOL, variant=manifest["variant"], seed=manifest["seed"],
            images=files, complete=True, code_commit=manifest["code_commit"], esi_commit=manifest["esi_commit"],
            diagnostic_pending=manifest["diagnostic_pending"], native_qualification_run=qualified,
            eligible_final_layout=qualified and not manifest["diagnostic_pending"],
            provenance="Fresh native exact-snapshot initialization; " + (
                "existing qualification gate passed." if qualified else "diagnostic preview; qualification was not run."),
            target_location=target_location, job_id=gpu.get("pbs_job_id") or manifest["job_id"],
            capture_job_id=manifest["job_id"], execution_mode=gpu.get("execution_mode", "pbs"),
            pbs_job_id=gpu.get("pbs_job_id"), gpu=evidence,
            model_calls=0, robot_actions=0, physical_grasp_verified=False, background_reused=False,
            human_camera_framing=getattr(session, "human_camera_framing", None),
            human_review_cameras=getattr(session, 'human_review_camera_evidence', None),
            human_camera_capture_protocol='native_human_render_products_drained_v2',
            human_floorplan=getattr(session, 'human_floorplan_metadata', None),
            interaction_setup=getattr(session, 'category_interaction_setup', None),
            background_initialization=[item.get('initialization_evidence')
                for item in getattr(getattr(session, 'scene_adapter', None), 'applied', [])
                if item.get('module') == 'wia.authoring.converters.wia_background'
                and item.get('phase') == 'after_postprocessing'],
            reflection_qualification=getattr(session, 'reflection_qualification', None),
            temporal_episode_preview=getattr(session, 'temporal_capture_evidence', None),
            initial_agent_bundle=getattr(session, 'capture_initial_agent_bundle_evidence', None),
            initial_overview_camera=getattr(session, 'initial_overview_camera_evidence', None),
            view_provenance={'global_floorplan': {'kind': 'human_floorplan',
                'label': 'Room map (segmentation; not a native render)', 'native_render': False}}
                if 'global_floorplan' in files else {},
            duration_s=time.monotonic() - began,
            timings={"cold_initialization_s": capture_began - began - initialization_wait_s,
                     "initialization_wait_s": initialization_wait_s,
                     "screenshot_s": time.monotonic() - capture_began,
                     "warm_swap_s": 0., "total_s": time.monotonic() - began})
        write_json(output / "preview_complete.json", receipt)
        progress("captured")

    def retain(session):
        """Keep this exact world available for human renders until explicitly closed."""
        if not os.environ.get("ESI_CAPTURE_RETAIN_WORLD"):
            return
        import numpy as np
        from core.record.images import save_png
        from wia.runtime.sim.esi_omnigibson import _array

        def task_poses():
            poses = {}
            for name in session.task_object_names:
                obj = session.env.scene.object_registry("name", name)
                if obj is not None:
                    position, quaternion = obj.get_position_orientation()
                    poses[name] = {"position": _array(position).tolist(),
                                   "quaternion_xyzw": _array(quaternion).tolist()}
            return poses

        requests = output / "retained_requests"
        requests.mkdir(exist_ok=True)
        retained_at, renders = time.time(), 0
        initial_poses = task_poses()
        write_json(output / "retained_initial_state.json", {
            "job_id": manifest["job_id"], "pid": os.getpid(), "captured_at": retained_at,
            "task_object_poses_world": initial_poses})
        progress("world_retained")
        while True:
            session.pipeline.og.sim.render()
            renders += 1
            poses = task_poses()
            displacement = max((float(np.linalg.norm(np.array(pose["position"])
                - initial_poses[name]["position"])) for name, pose in poses.items()), default=0.)
            write_json(output / "retained_world.json", {
                "schema": "esi_retained_capture_world_v1", "state": "RETAINED",
                "job_id": manifest["job_id"], "capture_signature": manifest["capture_signature"],
                "candidate_id": manifest["candidate_id"], "code_commit": manifest["code_commit"],
                "pid": os.getpid(), "start_ticks": native_start, "gpu_uuid": gpu["uuid"],
                "retained_at": retained_at, "heartbeat": time.time(), "render_count": renders,
                "model_calls": 0, "robot_actions": 0, "physics_steps_during_retention": 0,
                "task_object_poses_world": poses, "max_task_translation_change_m": displacement})
            for path in sorted(requests.glob("*.json")):
                response = output / "retained_responses" / path.name
                if response.exists():
                    continue
                try:
                    request = json.loads(path.read_text())
                    if (request.get("job_id") != manifest["job_id"]
                            or request.get("capture_signature") != manifest["capture_signature"]):
                        raise ValueError("Retained request must identify this exact snapshot.")
                    operation = request.get("operation")
                    result = {"job_id": manifest["job_id"], "request": path.name,
                              "operation": operation, "completed_at": time.time(), "pid": os.getpid()}
                    if operation == "close":
                        write_json(response, {**result, "state": "CLOSED"})
                        write_json(output / "retained_world.json", {**result, "state": "CLOSED"})
                        return
                    if operation != "capture":
                        raise ValueError("Supported retained operations are capture and explicit close.")
                    capture_began = time.monotonic()
                    views, _ = capture_loaded_views(session, selected, manifest["expected_views"])
                    destination = output / "retained_captures" / path.stem
                    files = {}
                    for name, rgb in views.items():
                        target = destination / f"{name}.png"
                        save_png(target, rgb)
                        files[name] = str(target)
                    preview = {**receipt, "images": files, "retained_world_preview": True,
                        "retention_request": path.name, "captured_at": time.time(),
                        "initial_capture_receipt": str(output / "preview_complete.json"),
                        "provenance": "Retained exact-snapshot native world; human review recapture without physics steps.",
                        "background_reused": True, "warm_swap_run": False,
                        "timings": {"cold_initialization_s": 0., "warm_swap_s": 0.,
                                    "screenshot_s": time.monotonic() - capture_began,
                                    "total_s": time.monotonic() - capture_began}}
                    write_json(destination / "preview_complete.json", preview)
                    write_json(response, {**result, "state": "CAPTURED",
                                         "receipt": str(destination / "preview_complete.json")})
                except Exception as error:
                    write_json(response, {"state": "FAILED", "request": path.name,
                                          "error": str(error), "completed_at": time.time()})
            time.sleep(2)

    if not manifest["diagnostic_pending"] and selected.definition.qualification is not None:
        # Keep the existing Rigid Containment controller/qualification path.
        import wia.runtime.sim.esi_web as web
        from wia.runtime.commands.verify_esi_rigid_grasp import main as probe_main

        def capture_initial(session):
            bind_capture_inputs(session, selected, manifest)
            _initial_agent_bundle(session)
            return session.observation_source.latest_snapshot

        def capture(session, metadata, controller, executor, **kwargs):
            if manifest.get('snapshot_task_file'):
                bind_capture_inputs(session, selected, manifest)
            initialized(session, qualified=True)
            progress("screenshots")
            capture_began = time.monotonic()
            try:
                views, location = capture_loaded_views(session, selected, manifest["expected_views"])
            except Exception:
                preserve_partial_views(session, selected, manifest['expected_views'], output)
                raise
            save(session, views, location, qualified=True, capture_began=capture_began)
            retain(session)

        web.run_esi_control = capture
        code = probe_main(["--viewer-only", "--scene-file", str(selected.path),
            "--data-root", manifest["snapshot_data_root"], "--variant", selected.variant,
            "--scene-seed", str(selected.seed), "--web", "--debug-cameras", "--human-preview",
            "--preview-fps", "0", "--gripper-control", "--trunk-control", "--parameterized-actions",
            "--primitive-profile", "fine", "--macro-execution", "metric_target",
            "--performance-profile", "benchmark", "--output-dir", str(output)],
            initial_observation_capture=capture_initial if manifest.get('capture_initial_agent_bundle') else None)
        if code or not receipt:
            raise RuntimeError("Existing native scene qualification did not reach review capture.")
        return receipt

    from wia.runtime.sim.esi_omnigibson import ESISession
    from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
    from wia.tasks.scene import ScenePatches

    adapter = ScenePatches(selected, selected.definition.bindings)
    settings = adapter.observation_settings()
    settings["camera_size"] = list(selected.definition.runtime.camera_size)
    session = ESISession(selected.source_path, require_robot=False, **settings)
    session.scene_adapter = adapter
    bind_capture_inputs(session, selected, manifest)
    session.json_root = Path(manifest['snapshot_data_root']).resolve()
    try:
        progress("cold_scene_loading")
        session.reset()
        if (session.scene_name, session.room_name) != (selected.scene_name, selected.room_name):
            raise RuntimeError("Native scene identity does not match its immutable candidate.")
        initialized(session)
        root = WIA_RUNTIME_ROOT
        if session.robot is not None:
            configure_robot_cameras(session, root / selected.definition.runtime.robot_camera_config,
                                    resolution=selected.definition.runtime.camera_size)
        progress("screenshots")
        capture_began = time.monotonic()
        views, location = capture_loaded_views(session, selected, manifest["expected_views"])
        save(session, views, location, capture_began=capture_began)
        retain(session)
        return receipt
    except Exception:
        try:
            preserve_partial_views(session, selected, manifest['expected_views'], output)
        except Exception as diagnostic_error:
            write_json(output / 'partial_capture_error.json', {'error': str(diagnostic_error), 'complete': False})
        raise
    finally:
        session.close()

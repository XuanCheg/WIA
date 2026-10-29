"""Robot-parented cameras: native head/wrists plus the five R1 chassis views."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import numpy as np

from core.robot_cameras import ADDITIONAL_RGB_NAMES
from wia.runtime.sim.esi_debug_cameras import camera_orientation
from wia.runtime.sim.esi_omnigibson import _array


def rebind_render_product(session, sensor, *, previous_resolution=None):
    """Attach fresh annotators without destroying a live Hydra texture.

    Destroy/recreate through the installed width setter leaves stale Fabric
    handles in this runtime. Keep the old product until simulator teardown and
    give the replacement a unique identity. No camera or scene prim is changed.
    """
    import omni.replicator.core as rep
    retired = getattr(session, '_retired_camera_products', [])
    old = sensor._render_product
    enabled = {sensor.modalities} if isinstance(sensor.modalities, str) else sensor.modalities
    modalities = tuple(dict.fromkeys([*sensor._annotators, *enabled]))
    detach_errors = []
    for annotator in sensor._annotators.values():
        if annotator is None:
            continue
        try:
            # The installed VisionSensor documents a bug in list-form detach.
            annotator.detach(old)
        except TypeError as error:
            if 'Invalid NodeObj' not in str(error):
                raise
            # Source cloth stop/play can already have invalidated this graph.
            # A replacement gets fresh nodes; do not reuse or destroy the stale
            # handles, and retain the exact native error in private evidence.
            detach_errors.append(str(error))
    retired.append((old, sensor._annotators))
    session._retired_camera_products = retired
    sensor._render_product = rep.create.render_product(sensor.prim_path,
        (sensor.image_width, sensor.image_height), name='CoreCamera_' + uuid.uuid4().hex, force_new=True)
    sensor._annotators = {}
    sensor.initialize_sensors(names=modalities)
    evidence = getattr(session, 'camera_render_product_rebindings', [])
    evidence.append({'sensor': sensor.name, 'old_product': old.path,
        'new_product': sensor._render_product.path, 'stale_detach_errors': detach_errors,
        'camera_prim_unchanged': True,
        'resolution_unchanged': previous_resolution is None or list(previous_resolution) == [sensor.image_width, sensor.image_height],
        'old_resolution_px': list(previous_resolution) if previous_resolution is not None else [sensor.image_width, sensor.image_height],
        'new_resolution_px': [sensor.image_width, sensor.image_height]})
    session.camera_render_product_rebindings = evidence


def _enabled_image_channels(sensor):
    modalities = sensor.modalities
    modalities = {modalities} if isinstance(modalities, str) else set(modalities)
    return ['rgb', *sorted(name for name in modalities if name.startswith('seg_'))]


def _channel_shape(value, channel, width, height):
    shape = getattr(value, 'shape', None)
    shape = [int(part) for part in shape] if shape is not None else None
    expected = [int(height), int(width)]
    empty = shape is None or not shape or any(part == 0 for part in shape)
    correct = bool(not empty and shape[:2] == expected and (
        (channel == 'rgb' and len(shape) == 3 and shape[2] in (3, 4)) or
        (channel != 'rgb' and (len(shape) == 2 or (len(shape) == 3 and shape[2] == 1)))))
    return {'shape': shape, 'expected_height_width': expected, 'ready': correct,
        'reason': None if correct else 'empty_native_channel' if empty else 'native_channel_shape_mismatch'}


def _raw_channel_readiness(session, sensor):
    """Read native annotator shapes without triggering segmentation remappers."""
    annotators = getattr(sensor, '_annotators', None)
    if not isinstance(annotators, dict):
        return None  # Lightweight test sensors retain the legacy get_obs path.
    channels = {}
    for name in _enabled_image_channels(sensor):
        annotator = annotators.get(name)
        if annotator is None:
            channels[name] = {'shape': None, 'expected_height_width': [sensor.image_height, sensor.image_width],
                'ready': False, 'reason': 'enabled_native_annotator_missing'}
            continue
        raw = annotator.get_data(device=session.pipeline.og.sim.device)
        value = raw.get('data') if isinstance(raw, dict) else raw
        channels[name] = _channel_shape(value, name, sensor.image_width, sensor.image_height)
    return {'source': 'native_raw_annotators', 'channels': channels,
        'ready': all(value['ready'] for value in channels.values())}


def _full_channel_readiness(sensor, *, native):
    data = sensor.get_obs()[0]
    if not native:
        rgb = data.get('rgb', [])
        return {'source': 'get_obs_test_sensor_fallback', 'ready': bool(_array(rgb).size),
            'channels': {'rgb': {'shape': list(_array(rgb).shape),
                'expected_height_width': [sensor.image_height, sensor.image_width],
                'ready': bool(_array(rgb).size)}}}
    channels = {name: _channel_shape(data.get(name), name, sensor.image_width, sensor.image_height)
        for name in _enabled_image_channels(sensor)}
    return {'source': 'full_get_obs', 'channels': channels,
        'ready': all(value['ready'] for value in channels.values())}


def _camera_invariants(sensor):
    optics = {}
    for name in ('focal_length', 'horizontal_aperture', 'clipping_range'):
        value = getattr(sensor, name, None)
        if value is not None:
            optics[name] = _array(value).tolist()
    getter = getattr(sensor, 'get_attribute', None)
    if callable(getter):
        value = getter('verticalAperture')
        if value is not None:
            optics['vertical_aperture'] = _array(value).tolist()
    # CameraProjection/intrinsic_matrix depend on a live camera_params stream.
    # Preserve the physical calibration fields instead of reading stale params.
    return {'prim_path': sensor.prim_path, 'pose': [_array(value).copy() for value in sensor.get_position_orientation()],
        'resolution': [sensor.image_width, sensor.image_height],
        'modalities': sorted({sensor.modalities} if isinstance(sensor.modalities, str) else sensor.modalities),
        'optics': optics}


def _invariants_preserved(before, after, *, allow_resize=False):
    return (before['prim_path'] == after['prim_path'] and before['modalities'] == after['modalities'] and
        before['optics'] == after['optics'] and (allow_resize or before['resolution'] == after['resolution']) and
        all(np.allclose(a, b, atol=1e-6, rtol=0) for a, b in zip(before['pose'], after['pose'])))


def _resize_camera(session, sensor, dimensions):
    dimensions = list(dimensions)
    before_size = [sensor.image_width, sensor.image_height]
    if before_size == dimensions:
        return
    viewport = getattr(sensor, '_viewport', None)
    product = getattr(sensor, '_render_product', None)
    if viewport is not None and product is not None:
        before = _camera_invariants(sensor)
        viewport.viewport_api.set_texture_resolution(tuple(dimensions))
        rebind_render_product(session, sensor, previous_resolution=before_size)
        if not _invariants_preserved(before, _camera_invariants(sensor), allow_resize=True):
            raise RuntimeError('Native camera resize changed its prim, pose, optics or modalities.')
    else:
        # Fakes without native products do not expose the destructive OG setters.
        if sensor.image_width != dimensions[0]:
            sensor.image_width = dimensions[0]
        if sensor.image_height != dimensions[1]:
            sensor.image_height = dimensions[1]


def recover_empty_render_products(session, registry):
    """Probe raw RGB/seg readiness, then recover with renders and no physics.

    A valid RGB buffer does not hide an empty segmentation channel. Shapes are
    readiness evidence only and never certify object recognition or semantics.
    """
    repairs, readiness = [], []
    session.camera_render_recovery = repairs
    session.camera_channel_readiness = readiness
    for name in registry.values():
        sensor = session.robot.sensors[name]
        status = {'sensor': name}
        readiness.append(status)
        try:
            raw = _raw_channel_readiness(session, sensor)
            status['raw_before'] = raw
            native = raw is not None
            full = _full_channel_readiness(sensor, native=native) if raw is None or raw['ready'] else None
            status['full_before'] = full
            if (raw is None or raw['ready']) and full['ready']:
                status['ready'] = True
                continue
            before = _camera_invariants(sensor)
            failed = raw if raw is not None and not raw['ready'] else full
            repair = {'sensor': name, 'resolution': before['resolution'],
                'reason': 'native_image_channels_unavailable_after_source_lifecycle',
                'failed_channels': [key for key, value in failed['channels'].items() if not value['ready']],
                'channels_before': failed['channels']}
            repairs.append(repair)
            rebind_render_product(session, sensor)
            for _ in range(9):
                session.pipeline.og.sim.render()
            after = _camera_invariants(sensor)
            if not _invariants_preserved(before, after):
                raise RuntimeError('Render-product recovery changed camera prim, pose, optics, size or modalities.')
            raw_after = _raw_channel_readiness(session, sensor)
            status['raw_after'] = raw_after
            repair['channels_after'] = raw_after['channels'] if raw_after is not None else None
            if raw_after is not None and not raw_after['ready']:
                repair['all_enabled_channels_ready'] = False
                raise RuntimeError('Native camera channels remain unavailable after bounded render warmup: ' +
                    name + ' ' + json.dumps(raw_after['channels']))
            full_after = _full_channel_readiness(sensor, native=native)
            status['full_after'] = full_after
            repair['full_observation_channels_after'] = full_after['channels']
            repair['rgb_restored'] = full_after['channels']['rgb']['ready']
            repair['all_enabled_channels_ready'] = full_after['ready']
            status['ready'] = full_after['ready']
            if not full_after['ready']:
                raise RuntimeError('Native full camera observation remains unavailable after bounded recovery: ' +
                    name + ' ' + json.dumps(full_after['channels']))
        except Exception as error:
            # Retain context and propagate the original failure; arbitrary native
            # exceptions are not interpreted as empty buffers or silently retried.
            status.update(ready=False, error=f'{type(error).__name__}: {error}')
            raise
    if repairs:
        print('[camera-render-recovery] ' + json.dumps(repairs), flush=True)
        session.invalidate_camera_cache()


def apply_head_camera_mount(session, config):
    """Apply an explicit eye-link mount without changing any robot joints."""
    mount = config.get("head_camera")
    if mount is None:
        return
    position = np.asarray(mount["position_m"], dtype=float)
    quaternion = np.asarray(mount["quaternion_xyzw"], dtype=float)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) == 0):
        raise ValueError("Head camera mounts need a finite position and nonzero quaternion.")
    sensor = session.robot.sensors[session.camera_name]
    sensor.set_position_orientation(position=position.tolist(),
        orientation=(quaternion / np.linalg.norm(quaternion)).tolist(), frame="parent")


def fov_render_dimensions(dimensions, horizontal_deg, vertical_deg):
    """Fit square-pixel native output to a canvas while preserving both FOVs."""
    aspect = np.tan(np.deg2rad(horizontal_deg) / 2) / np.tan(np.deg2rad(vertical_deg) / 2)
    width = min(dimensions[0], int(round(dimensions[1] * aspect)))
    height = min(dimensions[1], int(round(width / aspect)))
    return (width, height)


def configure_camera_fov(session, sensor, horizontal_deg, vertical_deg):
    """Apply native optics and resolve raster aspect without cropping the FOV."""
    if not all(np.isfinite(v) and 0 < v < 180 for v in (horizontal_deg, vertical_deg)):
        raise ValueError("Camera FOV must be finite and between zero and 180 degrees.")
    aperture = float(sensor.horizontal_aperture)
    focal = aperture / (2 * np.tan(np.deg2rad(horizontal_deg) / 2))
    vertical_aperture = 2 * focal * np.tan(np.deg2rad(vertical_deg) / 2)
    sensor.focal_length = float(focal)
    sensor.set_attribute("verticalAperture", float(vertical_aperture))
    # This installed renderer ignores render-product pixelAspectRatio. Use a
    # square-pixel raster with the matching aperture ratio; native K is decisive.
    from pxr import UsdRender
    product = UsdRender.Product(session.pipeline.og.sim.stage.GetPrimAtPath(sensor._render_product.path))
    product.CreateAspectRatioConformPolicyAttr().Set("adjustApertureHeight")
    product.CreatePixelAspectRatioAttr().Set(1.0)


def configure_robot_cameras(session, path: Path, *, resolution=None):
    from omnigibson.sensors import VisionSensor
    from wia.runtime.sim.esi_base import _main_thread

    _main_thread()
    config = json.loads(Path(path).read_text())
    apply_head_camera_mount(session, config)
    size = config["render_size"]
    dimensions = tuple(config.get("render_resolution", [size, size])) if resolution is None else tuple(resolution)
    entries = config["chassis_cameras"]
    if (len(dimensions) != 2 or any(type(value) is not int or not 64 <= value <= 2048 for value in dimensions)
            or {e["name"] for e in entries} != set(ADDITIONAL_RGB_NAMES[1:]) or len(entries) != 5):
        raise ValueError("R1 camera configuration needs five distinct chassis cameras and a valid render size.")
    width, height = dimensions
    profile = getattr(session, 'reflection_sensor_profile', None) or {}
    head_size = profile.get('head_image_size_px')
    if head_size is not None and (len(head_size) != 2 or any(type(v) is not int or not 64 <= v <= 2048 for v in head_size)):
        raise ValueError('An explicit reflection head size requires two supported pixel dimensions.')
    if head_size is not None:
        config['reflection_head_render_size'] = list(head_size)
    config["effective_render_size"] = list(dimensions)
    registry = {"agentview_rgb": session.camera_name, "wrist_rgb": session.wrist_camera_name}
    right = [name for name in session.rgb_cameras if "right" in name and "eef" in name]
    if len(right) != 1:
        raise RuntimeError("Eight-view R1 setup requires the native right wrist camera.")
    registry["right_wrist_rgb"] = right[0]
    # Some category initializers enable private segmentation before the five
    # chassis cameras exist. Apply that same private evidence channel to every
    # delivered RGB view; it is never projected into AgentObservation.
    adapter = getattr(session, 'scene_adapter', None)
    loaded_task = getattr(session, 'loaded_task', None) or getattr(adapter, 'task', None)
    definition = getattr(loaded_task, 'definition', None)
    runtime = getattr(definition, 'runtime', None)
    parameters = getattr(getattr(getattr(definition, 'private', None), 'evaluation', None), 'parameters', {})
    private_segmentation = (any('seg_instance' in sensor.modalities for sensor in session.robot.sensors.values())
        or getattr(runtime, 'requires_native_visual_discovery', False)
        or parameters.get('requires_native_visual_discovery', False))
    for name in registry.values():
        sensor = session.robot.sensors[name]
        target_size = head_size if name == session.camera_name and head_size is not None else dimensions
        _resize_camera(session, sensor, target_size)
    parent = session.robot.links[session.robot.base_footprint_link_name]
    parent_path = parent.prim_path.removeprefix(session.env.scene.prim_path)
    for entry in entries:
        position = np.asarray(entry["position_m"], dtype=float)
        yaw = np.deg2rad(entry["yaw_deg"])
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(yaw):
            raise ValueError("Camera mounts need finite base-relative positions and yaw.")
        name = entry["name"]
        sensor = VisionSensor(relative_prim_path=parent_path + "/" + name,
                              name=session.robot.name + ":" + name, modalities="rgb",
                              image_width=width, image_height=height, focal_length=entry["focal_length"])
        sensor.load(session.env.scene)
        sensor.initialize()
        target = position + np.array([np.cos(yaw), np.sin(yaw), 0.0])
        sensor.set_position_orientation(position=position.tolist(),
                                        orientation=camera_orientation(position, target).tolist(), frame="parent")
        session.robot.sensors[sensor.name] = sensor
        registry[name] = sensor.name
    optics = config.get("camera_fov_deg")
    if optics:
        for view, sensor_name in registry.items():
            role = "head" if view == "agentview_rgb" else "wrist" if "wrist" in view else "chassis"
            sensor = session.robot.sensors[sensor_name]
            if not (view == "agentview_rgb" and head_size is not None):
                _resize_camera(session, sensor, fov_render_dimensions(dimensions, *optics[role]))
            configure_camera_fov(session, sensor, *optics[role])
    session.robot_camera_registry = registry
    if private_segmentation:
        for name in registry.values():
            sensor = session.robot.sensors[name]
            if 'seg_instance' not in sensor.modalities:
                sensor.add_modality('seg_instance')
    session.rgb_cameras = list(registry.values())
    session.robot_camera_config = config
    if config["antialiasing"] != "TAA":
        raise ValueError("Supported robot-camera antialiasing is TAA.")
    import carb
    carb.settings.get_settings().set_int("/rtx/post/aa/op", 2)
    for _ in range(3):
        session.pipeline.og.sim.render()
    # Native source cloth construction stops/restarts PhysX and rebuilds the
    # viewer render product. Detect stale mounted products before screenshots.
    recover_empty_render_products(session, registry)
    config["effective_view_render_sizes"] = {view: [session.robot.sensors[name].image_width,
        session.robot.sensors[name].image_height] for view, name in registry.items()}
    session.robot_camera_mounts = {}
    for view, name in registry.items():
        sensor = session.robot.sensors[name]
        position, quaternion = sensor.get_position_orientation(frame="parent")
        session.robot_camera_mounts[view] = {
            "sensor_name": name, "prim_path": sensor.prim_path,
            "mount_position_m": _array(position).tolist(), "mount_quaternion_xyzw": _array(quaternion).tolist(),
            "render_size": [sensor.image_width, sensor.image_height], "source": "robot_mounted_sensor",
        }
    if getattr(session, 'reflection_sensor_profile', None):
        from wia.runtime.sim.esi_reflection import measure_reflection
        measure_reflection(session)

"""Render the required initial global RGB from the actual loaded native world.

Run through src/wia/runtime/commands/run_esi_robot.py or src/wia/runtime/commands/check_esi_core_native.py.
The dedicated camera changes no actor, mounted camera, collision or physics state.
"""
from __future__ import annotations

import math
import numpy as np


def capture_initial_overview(session, *, snapshot_id, captured_at):
    from omnigibson.sensors import VisionSensor
    from wia.runtime.sim.esi_debug_cameras import camera_orientation
    from wia.tasks.interaction_setup import array, _objects

    objects = _objects(session.env.scene)
    floors = [obj for obj in objects if obj.category in ('floor', 'floors') and not obj.visual_only]
    if not floors:
        raise ValueError('Initial global RGB requires loaded native floor geometry.')
    bounds = np.asarray([[array(v) for v in obj.aabb] for obj in floors])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    if not np.isfinite(bounds).all() or np.any(high[:2] <= low[:2]):
        raise ValueError('Initial global RGB has invalid native loaded-floor bounds.')
    center = (low + high) / 2
    floor_z = float(bounds[:, 1, 2].min())
    sensor = getattr(session, '_initial_overview_sensor', None)
    if sensor is None:
        sensor = VisionSensor(relative_prim_path='/agent_context/initial_global',
            name='initial_global', modalities=('rgb',), image_width=1024,
            image_height=1024, focal_length=12.)
        sensor.load(session.env.scene)
        sensor.initialize()
        session._initial_overview_sensor = sensor
    tangent = float(sensor.horizontal_aperture) / (2 * float(sensor.focal_length))
    extent = float(max(high[:2] - low[:2]) * 1.05)
    visible_bounds = [array(obj.aabb[1]) for obj in objects
        if bool(getattr(obj, 'visible', True)) and not obj.visual_only
        and np.all(array(obj.aabb[0])[:2] <= high[:2]) and np.all(array(obj.aabb[1])[:2] >= low[:2])]
    top = max((float(v[2]) for v in visible_bounds), default=floor_z + 2.)
    height = max(extent / (2 * tangent), top - floor_z + .5)
    target = np.array([center[0], center[1], floor_z])
    position = target + [0., 0., height]
    actor_names = set(getattr(session, 'task_object_names', ()))
    actors = [obj for obj in objects if obj is session.robot or obj.name in actor_names
              and bool(getattr(obj, 'visible', True))]
    actor_top = max((float(array(obj.aabb[1])[2]) for obj in actors), default=floor_z + 1.8)
    # A declared roof cross-section keeps the complete task actors and robot
    # below the near plane. It does not hide actors or modify any scene body.
    section_z = max(floor_z + 1.9, actor_top + .10)
    sensor.clipping_range = (max(.01, position[2] - section_z),
                            max(100., height + high[2] - low[2] + 10.))
    sensor.set_position_orientation(position=position.tolist(),
        orientation=camera_orientation(position, target).tolist(), frame='world')
    from wia.runtime.sim.esi_robot_cameras import rebind_render_product
    rebind_render_product(session, sensor)
    for _ in range(12):
        session.pipeline.og.sim.render()
    rgb = session._sensor_rgb(sensor)
    if rgb is None or np.asarray(rgb).ndim != 3 or not np.ptp(rgb):
        raise ValueError('Initial global native RGB is empty or constant.')
    p, q = (array(v) for v in session.robot.get_position_orientation())
    x, y, z, w = q / np.linalg.norm(q)
    half = height * tangent
    receipt = {**session.initial_overview_identity, 'rgb': rgb,
        'snapshot_id': snapshot_id, 'captured_at': captured_at,
        'extent_xy_m': [center[0] - half, center[1] - half, center[0] + half, center[1] + half],
        'image_right_world_xy': [1., 0.], 'image_up_world_xy': [0., 1.],
        'robot_start_position_m': p.tolist(),
        'robot_start_yaw_rad': math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
        'source': 'native_initial_overview_camera'}
    session.initial_overview_camera_evidence = {
        'position_world_m': position.tolist(), 'target_world_m': target.tolist(),
        'loaded_floor_names': [obj.name for obj in floors], 'actor_mutations': 0,
        'physics_steps': 0, 'roof_section_height_world_m': section_z,
        'all_task_actor_bounds_preserved': section_z > actor_top,
        'cutaway_rule': 'above_complete_task_actors_and_robot',
        'camera_projection': 'native_perspective', 'resolution_px': [1024, 1024]}
    return receipt

"""Isolated native source-protocol capacity calibration, without Agent controls.

Run with the BEHAVIOR Python through src/wia/runtime/commands/calibrate_task_edit2_capacity.py;
CPU checks: python -B -m pytest -q tests/test_task_edit2_capacity_driver.py.
Importing this module starts no simulator. Native setters belong exclusively to
this disposable diagnostic session; source fill_both is loaded unchanged.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from contextlib import contextmanager, nullcontext
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

SCHEMA = 'task_edit2_native_capacity_driver_20261018_v1'
POOL_PATHS = tuple(f'/World/plane_{index}' for index in range(4))
ROOT = WIA_RUNTIME_ROOT


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def select_pairs(receipt, specification=None):
    """Default to missing pairs; explicit A:B preserves source left/right order."""
    from wia.authoring.converters.liquid_capacity_inventory import validate_truth
    ids = {item['id'] for item in receipt['inventory']}
    pairs = (validate_truth(receipt).get('missing_pairs', []) if specification is None else
        [part.strip().split(':') for part in specification.split(',')])
    selected, seen = [], set()
    for pair in pairs:
        if len(pair) != 2 or any(key not in ids for key in pair) or pair[0] == pair[1]:
            raise ValueError('Pairs must be distinct inventory IDs, e.g. A:B,A:C.')
        token = tuple(sorted(pair))
        if token in seen:
            raise ValueError('The same unordered pair was requested more than once.')
        seen.add(token)
        selected.append(tuple(pair))
    return selected


def _equal(first, second, *, atol=1e-5):
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(_equal(first[k], second[k], atol=atol) for k in first)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(_equal(a, b, atol=atol) for a, b in zip(first, second))
    if isinstance(first, (int, float, np.number)) and not isinstance(first, (bool, np.bool_)):
        return isinstance(second, (int, float, np.number)) and bool(np.isclose(first, second, atol=atol, rtol=0))
    return first == second


def validate_reset_proof(baseline, measured, *, empty_water, pool_absent, history):
    """Build proof from independent measurements; booleans alone cannot certify reset.

    Tolerance is for native state roundoff only, not task capacity or ranking.
    Both complete scene-object rows and USD camera/label/visibility rows must be
    present. An absent native measurement fails closed.
    """
    checks = {
        'baseline_present': bool(baseline.get('objects')) and bool(baseline.get('stage')),
        'full_objects_restored': bool(measured.get('objects')) and _equal(baseline.get('objects'), measured.get('objects')),
        'stage_camera_labels_visibility_restored': bool(measured.get('stage')) and _equal(baseline.get('stage'), measured.get('stage')),
        'empty_water': empty_water is True,
        'pool_absent': pool_absent is True,
        'agent_history_empty': history.get('episode_absent') is True and history.get('entries') == 0
            and history.get('agent_components_absent') is True,
    }
    accepted = all(checks.values())
    return {'schema': SCHEMA, 'accepted': accepted, 'all_inventory_restored': accepted,
        'agent_history_empty': checks['agent_history_empty'], 'checks': checks, 'history': history,
        'reason': None if accepted else 'full_native_diagnostic_baseline_reset_unverified'}


def source_pool_geometry(first_bounds, second_bounds, radius):
    """Exactly the pool geometry in unchanged batch_pour.fill_both (not capacity)."""
    a, b = (np.asarray(x, dtype=float) for x in first_bounds)
    c, d = (np.asarray(x, dtype=float) for x in second_bounds)
    low = np.array([a[0], min(a[1], c[1])])
    high = np.array([d[0], max(b[1], d[1])])
    height = max(b[2] - a[2], d[2] - c[2])
    half_z = max(height * .55, height / 2 + 2.1 * float(radius))
    return {'low_xy': low - .05, 'high_xy': high + .05, 'half_z': float(half_z),
        'height': float(height)}


class ProgressJournal:
    """Flush progress and measured counts before a subsequent native failure."""
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.current_pair = None

    def write(self, filename, value):
        target = self.output / filename
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(target)

    def event(self, stage, **fields):
        row = {'schema': SCHEMA, 'stage': stage, 'pid': os.getpid(),
            'elapsed_s': time.monotonic() - self.started, 'pair': self.current_pair, **fields}
        with (self.output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.write('progress.json', row)
        print(f"[capacity] {stage} pair={self.current_pair} elapsed={row['elapsed_s']:.1f}s", flush=True)


def _objects(scene):
    rows = scene.objects
    return list(rows.values()) if isinstance(rows, dict) else list(rows)


def joint_snapshot(obj):
    rows = {}
    for name, joint in obj.joints.items():
        if not joint.articulated:
            continue
        limited = bool(joint.is_single_dof and joint.has_limit)
        row = {'position': array(joint.get_state()[0]).tolist(), 'has_limit': limited}
        if limited:
            row.update(lower_limit=array(joint.lower_limit).tolist(), upper_limit=array(joint.upper_limit).tolist())
        rows[name] = row
    return rows


def _plain(value):
    """Stable JSON observations of USD xforms, camera configuration and visibility."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if callable(getattr(value, 'GetReal', None)) and callable(getattr(value, 'GetImaginary', None)):
        # Gf quaternion repr is textual and differs after harmless native
        # normalization. Preserve numeric components for the same 1e-5 reset
        # tolerance used by every other measured transform component.
        return [float(value.GetReal()), *[float(v) for v in value.GetImaginary()]]
    try:
        converted = np.asarray(value)
        if converted.dtype.kind in 'biuf' and np.isfinite(converted).all():
            return converted.tolist()
    except (ValueError, TypeError):
        pass
    return str(value)


class NativeCapacityCalibration:
    """One owner-thread session, isolated pair contexts and measured full reset.

    Install ``capture_source_initialization`` as the session's initialization
    stage observer BEFORE session.reset(); construct this owner AFTER reset and
    robot camera configuration. No executor, Agent, evaluator or history is made.
    """
    def __init__(self, session, receipt, source_initial_joints, output, *, render_without_images=False):
        self.session, self.receipt = session, copy.deepcopy(receipt)
        self.scene, self.sim = session.env.scene, session.pipeline.og.sim
        self.owner_thread = threading.get_ident()
        self.journal = ProgressJournal(output)
        self.items = {item['id']: item for item in receipt['inventory']}
        self.water = self.scene.get_system('water')
        self.source_joints = copy.deepcopy(source_initial_joints)
        if set(self.source_joints) != {item['name'] for item in self.items.values()}:
            raise ValueError('Missing original, pre-postprocess joint initialization measurements.')
        self._check_inventory()
        self._cleanup_pool()
        if not self._history()['agent_components_absent'] or not self._history()['episode_absent']:
            raise ValueError('Calibration requires a separate session before any Agent episode/executor.')
        self.baseline_state = copy.deepcopy(self.sim.dump_state(serialized=False))
        self.stage_state = self._stage_attributes()
        self.baseline = self.measure()
        self.baseline_joints = {obj.name: joint_snapshot(obj) for obj in _objects(self.scene)}
        self.journal.write('baseline.json', self.baseline)
        self.render_policy = self._render_policy(render_without_images)
        self.journal.write('render_policy.json', self.render_policy)
        from omnigibson.macros import gm
        expected_dt = {'physics_dt_s': 1. / gm.DEFAULT_PHYSICS_FREQ,
            'rendering_dt_s': 1. / gm.DEFAULT_RENDERING_FREQ, 'sim_step_dt_s': 1. / gm.DEFAULT_SIM_STEP_FREQ}
        if any(not np.isclose(self.render_policy[key], value, rtol=0, atol=1e-12) for key, value in expected_dt.items()):
            raise ValueError('Diagnostic timestep differs from original batch_pour default Environment settings.')
        self.render_policy['original_environment_default_dt_s'] = expected_dt
        self.journal.write('render_policy.json', self.render_policy)
        self.initial_water_settings = self._water_settings()
        self.journal.event('baseline_saved', native_time_s=float(self.sim.current_time),
            inventory_id=receipt['inventory_id'], water_settings=self.initial_water_settings)

    @staticmethod
    def capture_source_initialization(receipt, destination):
        """Return an existing ESISession stage-observer, not a control callback."""
        def capture(session, stage, config):
            if stage == 'settled':
                for item in receipt['inventory']:
                    obj = session.env.scene.object_registry('name', item['name'])
                    if obj is None:
                        raise ValueError('Source inventory missing before postprocessing.')
                    destination[item['name']] = joint_snapshot(obj)
        return capture

    def _owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Native calibration must remain on the simulator owner thread.')

    def _water_settings(self):
        from wia.authoring.converters.liquid_capacity_inventory import native_water_settings
        return native_water_settings(self.water)

    def _history(self):
        entries = 0
        for name in ('action_history', 'history', 'public_action_history'):
            value = getattr(self.session, name, None)
            if value is not None:
                entries += len(value)
        return {'episode_absent': not hasattr(self.session, 'episode_id'), 'entries': entries,
            'agent_components_absent': not any(vars(self.session).get(name) is not None
                for name in ('executor', 'agent', 'evaluator', '_observation_source'))}

    def _check_inventory(self):
        if (self.session.scene_name, self.session.room_name) != (self.receipt['scene_model'], self.receipt['room']):
            raise ValueError('Native floor/world differs from the exact source inventory.')
        for item in self.items.values():
            obj = self.scene.object_registry('name', item['name'])
            if (obj is None or obj.category != item['category'] or obj.model != item['model'] or
                    not np.allclose(array(obj.scale), item['scale'], rtol=1e-5, atol=1e-6) or
                    not obj.fixed_base or obj.visual_only):
                raise ValueError('Native asset, scale, fixed_base or collision setting differs from source.')

    def _stage_attributes(self):
        attributes = {}
        for prim in self.sim.stage.Traverse():
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if (name == 'visibility' or name.startswith('xformOp') or name in
                        ('focalLength', 'horizontalAperture', 'verticalAperture', 'clippingRange', 'projection')):
                    value = attr.Get()
                    attributes[(str(prim.GetPath()), name)] = (value, attr.HasAuthoredValueOpinion())
        return attributes

    def measure(self):
        self._owner()
        rows = {}
        for obj in _objects(self.scene):
            row = {'category': obj.category, 'model': getattr(obj, 'model', None),
                'scale': array(obj.scale).tolist(), 'pose': [array(v).tolist() for v in obj.get_position_orientation()],
                'aabb': [array(v).tolist() for v in obj.aabb], 'fixed_base': bool(obj.fixed_base),
                'visual_only': bool(obj.visual_only), 'visible': bool(obj.visible), 'joints': joint_snapshot(obj)}
            for key, method in (('linear_velocity', 'get_linear_velocity'), ('angular_velocity', 'get_angular_velocity')):
                try:
                    row[key] = array(getattr(obj, method)()).tolist()
                except AttributeError:
                    if not (getattr(obj, 'kinematic_only', False) and obj.fixed_base):
                        raise
                    # Installed RigidKinematicPrim exposes no dynamic velocity
                    # state. Exact pose/attribute reset is checked independently.
                    row[key] = None
                    row['velocity_source'] = 'fixed_native_kinematic_without_dynamic_velocity_state'
            rows[obj.name] = row
        stage = {path + ':' + key: {'value': _plain(value), 'authored': authored}
            for (path, key), (value, authored) in self._stage_attributes().items()
            if not path.startswith(str(self.water.prim_path) + '/') and path != str(self.water.prim_path) and path not in POOL_PATHS
            and not any(path.startswith(pool + '/') for pool in POOL_PATHS)}
        return {'objects': rows, 'stage': stage}

    def _cleanup_pool(self):
        self._owner()
        from omnigibson.utils.usd_utils import delete_or_deactivate_prim
        self.water.remove_all_particles()
        for path in POOL_PATHS:
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsActive():
                delete_or_deactivate_prim(path)
        # Removing source GroundPlane shapes invalidates native tensor views.
        # Propagate scene edits without a physics timestep, then reacquire all
        # handles BEFORE reading any joint state or restoring an inventory.
        self._refresh_handles_after_scene_edits()
        if self.water.n_particles != 0 or not self._pool_absent():
            raise RuntimeError('Original diagnostic pool/water could not be cleared.')

    def _refresh_handles_after_scene_edits(self):
        self.sim.pi.update_simulation(elapsedStep=0, currentTime=self.sim.current_time)
        self.sim.update_handles()
        from omnigibson.utils.usd_utils import PoseAPI
        PoseAPI.invalidate()

    def _pool_absent(self):
        return all(not self.sim.stage.GetPrimAtPath(path).IsValid() or
            not self.sim.stage.GetPrimAtPath(path).IsActive() for path in POOL_PATHS)

    def _restore_joints(self, snapshots):
        import torch
        for name, rows in snapshots.items():
            obj = self.scene.object_registry('name', name)
            for key, row in rows.items():
                joint = obj.joints[key]
                if row['has_limit']:
                    # Expand before shrinking to avoid an intermediate inverted interval.
                    joint.lower_limit = min(float(array(joint.lower_limit)), float(row['lower_limit']))
                    joint.upper_limit = max(float(array(joint.upper_limit)), float(row['upper_limit']))
                    joint.lower_limit, joint.upper_limit = row['lower_limit'], row['upper_limit']
                joint.set_pos(torch.tensor(row['position'], dtype=torch.float32))
        self.sim.update_handles()

    def reset_actual_inventory(self):
        self._owner()
        self.journal.event('restoring_actual_inventory')
        self._cleanup_pool()
        for (path, name), (value, authored) in self.stage_state.items():
            prim = self.sim.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError('A baseline native prim was removed: ' + path)
            attr = prim.GetAttribute(name)
            if attr.HasAuthoredValueOpinion() == authored and _plain(attr.Get()) == _plain(value):
                continue
            if authored:
                attr.Set(value)
            else:
                attr.Clear()
        self._refresh_handles_after_scene_edits()
        self._restore_joints(self.baseline_joints)
        self.sim.load_state(copy.deepcopy(self.baseline_state), serialized=False)
        from omnigibson.utils.usd_utils import PoseAPI
        PoseAPI.invalidate()
        for _ in range(3):
            self.sim.render()
        self.session.invalidate_camera_cache()
        measured = self.measure()
        self._check_inventory()
        proof = validate_reset_proof(self.baseline, measured, empty_water=self.water.n_particles == 0,
            pool_absent=self._pool_absent(), history=self._history())
        self.journal.write('reset_proof.json', proof)
        if not proof['accepted']:
            self.journal.write('failed_reset_measurement.json', measured)
        self.journal.event('actual_inventory_reset_verified' if proof['accepted'] else 'actual_inventory_reset_failed', proof=proof)
        return proof

    def _render_policy(self, requested):
        method = getattr(self.sim, 'render_on_step', None)
        code = inspect.getsource(type(self.sim).step)
        scope = inspect.getsource(method) if callable(method) else ''
        verified = ('self._render_on_step' in code and 'self._non_physics_step()' in code and
            'n_physics_timesteps_per_render' in code and 'original_value' in scope and 'yield' in scope)
        if requested and not verified:
            raise ValueError('Installed render_on_step/step semantics have not been verified.')
        return {'mode': 'render_on_step_false' if requested else 'original_source_rendering',
            'installed_api_verified': verified, 'step_source_sha256': hashlib.sha256(code.encode()).hexdigest(),
            'scope_source_sha256': hashlib.sha256(scope.encode()).hexdigest(),
            'physics_dt_s': float(self.sim.get_physics_dt()), 'rendering_dt_s': float(self.sim.get_rendering_dt()),
            'sim_step_dt_s': float(self.sim.get_sim_step_dt()), 'nonphysics_updates_preserved': verified,
            'source_protocol_and_sampling_unchanged': True}

    def _stage_unused(self, active_names):
        import torch
        objects = _objects(self.scene)
        high = np.max([array(obj.aabb[1]) for obj in objects], axis=0)
        staged = []
        for obj in objects:
            if (obj.name in {item['name'] for item in self.items.values()} - active_names or
                    obj.name.startswith('esi_interaction_support_')):
                if not obj.fixed_base:
                    raise ValueError('Only declared fixed diagnostic props may be staged.')
                obj.set_position_orientation(position=torch.tensor([high[0] + 10 + 2 * len(staged), high[1] + 10, high[2] + 3], dtype=torch.float32))
                obj.visible = False
                staged.append(obj.name)
        label = self.sim.stage.GetPrimAtPath('/World/CapacityLabels')
        if label.IsValid():
            import omnigibson.lazy as lazy
            lazy.pxr.UsdGeom.Imageable(label).CreateVisibilityAttr().Set(lazy.pxr.UsdGeom.Tokens.invisible)
        return staged

    def _footprint(self, geometry, source_offset, floor):
        """Entire map rectangle plus measured swept volume and native floor rays."""
        import torch
        from omnigibson.utils.sampling_utils import raytest
        from wia.tasks.scene_only import local_floor_surface
        trav, seg = self.scene.trav_map, self.scene.seg_map
        free, rooms = array(trav.floor_map[0]), array(seg.room_ins_map)
        room_id = seg.room_ins_name_to_ins_id[self.receipt['room']]
        cells = np.argwhere(free == 255)
        world = array(trav.map_to_world(torch.tensor(cells, dtype=torch.float32)))
        # Deterministic exhaustive native-map search, nearest the original pair first.
        order = np.argsort(np.linalg.norm(world - source_offset[:2], axis=1), kind='stable')
        floor_low, floor_high = (array(v) for v in floor.aabb)
        obstacles = [(array(obj.aabb[0]), array(obj.aabb[1])) for obj in _objects(self.scene)
            if obj is not floor and not obj.visual_only]
        for index in order:
            delta = world[index] - (geometry['low_xy'] + geometry['high_xy']) / 2
            low, high = geometry['low_xy'] + delta, geometry['high_xy'] + delta
            corner = np.array([low, high])
            indices = array(trav.world_to_map(torch.tensor(corner, dtype=torch.float32))).astype(int)
            r0, c0 = np.minimum(*indices); r1, c1 = np.maximum(*indices)
            if min(r0, c0) < 0 or r1 >= free.shape[0] or c1 >= free.shape[1]:
                continue
            if not np.all(free[r0:r1 + 1, c0:c1 + 1] == 255):
                continue
            room_indices = array(seg.world_to_map(torch.tensor(corner, dtype=torch.float32))).astype(int)
            sr0, sc0 = np.minimum(*room_indices); sr1, sc1 = np.maximum(*room_indices)
            if min(sr0, sc0) < 0 or sr1 >= rooms.shape[0] or sc1 >= rooms.shape[1] or not np.all(rooms[sr0:sr1 + 1, sc0:sc1 + 1] == room_id):
                continue
            z0 = float(floor_high[2])
            sweep_low, sweep_high = np.r_[low, z0 + .001], np.r_[high, z0 + .09 + 3.2 * geometry['height']]
            if any(np.all(np.minimum(sweep_high, high_obj) > np.maximum(sweep_low, low_obj))
                    for low_obj, high_obj in obstacles):
                continue
            xy_points = np.array([[x, y] for x in np.linspace(low[0], high[0], max(2, int(np.ceil((high[0] - low[0]) / trav.map_resolution)) + 1))
                for y in np.linspace(low[1], high[1], max(2, int(np.ceil((high[1] - low[1]) / trav.map_resolution)) + 1))])
            try:
                hits = [local_floor_surface(floor.prim_path, raytest([*xy, floor_high[2] + .1],
                    [*xy, floor_low[2] - .1], only_closest=False)) for xy in xy_points]
                heights = [float(hit['position'][2]) for hit in hits]
                if not np.allclose(heights, heights[0], atol=1e-4, rtol=0):
                    continue
            except ValueError:
                continue
            return delta, float(heights[0]), {'pool_bounds_xy_m': [low.tolist(), high.tolist()],
                'swept_bounds_m': [sweep_low.tolist(), sweep_high.tolist()], 'map_resolution_m': float(trav.map_resolution),
                'full_map_rectangle_free': True, 'full_map_rectangle_in_source_room': True,
                'native_floor_ray_count': len(hits), 'native_floor_heights_m': heights,
                'background_swept_aabb_clear': True, 'floor_name': floor.name, 'floor_path': floor.prim_path}
        raise ValueError('No measured source-compatible loaded-floor footprint fits the pool and lift sweep.')

    @contextmanager
    def diagnostic_trial(self, first, second):
        self._owner()
        import torch
        from omnigibson.utils.usd_utils import PoseAPI
        self.journal.current_pair = [first['id'], second['id']]
        began = time.monotonic()
        start_time = float(self.sim.current_time)
        self.journal.event('pair_reset_started', native_time_s=start_time)
        reset = self.reset_actual_inventory()
        if not reset['accepted']:
            raise ValueError('Per-pair full baseline reset did not verify.')
        objects = [self.scene.object_registry('name', item['name']) for item in (first, second)]
        staged = self._stage_unused({obj.name for obj in objects})
        self._restore_joints(self.source_joints)
        floor = self.scene.object_registry('name', 'floors_ptwlei_0')
        if floor is None:
            raise ValueError('The original source floor is not loaded.')
        # Measure source identity orientation away from the trial, without physics.
        source_offset = array(objects[0].get_position_orientation()[0]).copy()
        for obj in objects:
            obj.set_position_orientation(orientation=torch.tensor([0., 0., 0., 1.]))
        PoseAPI.invalidate()
        bounds = [[array(v) for v in obj.aabb] for obj in objects]
        widths = [b[1][0] - b[0][0] for b in bounds]
        roots = [array(obj.get_position_orientation()[0]) for obj in objects]
        shifts = [np.array([-roots[0][0], -roots[0][1], 0.]),
            np.array([widths[0] / 2 + .3 + widths[1] / 2 - roots[1][0], -roots[1][1], 0.])]
        geometry = source_pool_geometry([v + shifts[0] for v in bounds[0]],
            [v + shifts[1] for v in bounds[1]], self.water.particle_radius)
        # Exclude only the active containers, which will move into this footprint.
        parked = []
        for obj in objects:
            p, q = obj.get_position_orientation()
            parked.append((p.clone(), q.clone()))
            obj.set_position_orientation(position=torch.tensor([2000. + len(parked), 2010., 10.]))
        PoseAPI.invalidate()
        delta, floor_z, footprint = self._footprint(geometry, source_offset, floor)
        for obj, (p, q) in zip(objects, parked):
            obj.set_position_orientation(position=p, orientation=q)
        PoseAPI.invalidate()
        source_floor_z = []
        pair_positions = []
        for index, obj in enumerate(objects):
            p = array(obj.get_position_orientation()[0]).copy()
            bottom = float(array(obj.aabb[0])[2])
            p[:2] = delta + [0. if index == 0 else widths[0] / 2 + .3 + widths[1] / 2, 0.]
            p[2] += floor_z - bottom
            obj.set_position_orientation(position=torch.tensor(p, dtype=torch.float32), orientation=torch.tensor([0., 0., 0., 1.]))
            for _ in range(30):
                self.sim.step()
            PoseAPI.invalidate()
            measured_bottom = float(array(obj.aabb[0])[2])
            if not np.isclose(measured_bottom, floor_z, atol=1e-4, rtol=0):
                raise ValueError('Source on-floor bottom did not verify.')
            source_floor_z.append(measured_bottom + .05)
            pair_positions.append(array(obj.get_position_orientation()[0]).copy())
        for obj, p in zip(objects, pair_positions):
            p[2] += .04
            obj.set_position_orientation(position=torch.tensor(p, dtype=torch.float32), orientation=torch.tensor([0., 0., 0., 1.]))
            for _ in range(30):
                self.sim.step()
        PoseAPI.invalidate()
        self._restore_joints({obj.name: self.source_joints[obj.name] for obj in objects})
        self._check_inventory()
        measured_joints = {obj.name: joint_snapshot(obj) for obj in objects}
        expected_joints = {obj.name: self.source_joints[obj.name] for obj in objects}
        poses_ok = all(np.allclose(array(obj.get_position_orientation()[0]), p, atol=1e-5, rtol=0) and
            np.allclose(array(obj.get_position_orientation()[1]), [0, 0, 0, 1], atol=1e-5, rtol=0) for obj, p in zip(objects, pair_positions))
        floor_unchanged = _equal(self.baseline['objects'][floor.name], self.measure()['objects'][floor.name])
        checks = {'baseline_reset_verified': reset['accepted'], 'source_joint_state': _equal(expected_joints, measured_joints),
            'source_pair_poses': poses_ok, 'source_floor_unchanged': floor_unchanged,
            'source_raised_bottoms': all(np.isclose(float(array(obj.aabb[0])[2]), floor_z + .04,
                atol=1e-4, rtol=0) for obj in objects),
            'empty_water': self.water.n_particles == 0, 'pool_absent': self._pool_absent(),
            'water_settings_unchanged': self._water_settings() == self.initial_water_settings,
            'history_empty': self._history() == reset['history']}
        proof = {'checks': checks, 'footprint': footprint, 'staged_props': staged,
            'floor_z_m': source_floor_z, 'pair_objects': {obj.name: self.measure()['objects'][obj.name] for obj in objects},
            'source_gap_m': .3, 'source_raise_m': .04, 'source_floor_offset_m': .05}
        self.journal.write('pair_' + first['id'] + second['id'] + '_initialization.json', proof)
        if not all(checks.values()):
            raise ValueError('Source-compatible diagnostic initialization did not verify: ' + str(checks))
        self.journal.event('isolated_pair_verified', native_time_s=float(self.sim.current_time), checks=checks)
        try:
            yield {'objects': objects, 'water': self.water, 'floor_z': source_floor_z,
                'isolated_reset_verified': True, 'agent_episode_active': False}
        finally:
            post_checks = {'physics_dt_unchanged': float(self.sim.get_physics_dt()) == self.render_policy['physics_dt_s'],
                'rendering_dt_unchanged': float(self.sim.get_rendering_dt()) == self.render_policy['rendering_dt_s'],
                'sim_step_dt_unchanged': float(self.sim.get_sim_step_dt()) == self.render_policy['sim_step_dt_s'],
                'source_floor_unchanged': _equal(self.baseline['objects'][floor.name], self.measure()['objects'][floor.name]),
                'water_settings_unchanged': self._water_settings() == self.initial_water_settings}
            sweep_low, sweep_high = map(np.asarray, footprint['swept_bounds_m'])
            active = {obj.name for obj in objects}
            post_checks['background_still_clear'] = not any(np.all(np.minimum(sweep_high, array(obj.aabb[1])) >
                np.maximum(sweep_low, array(obj.aabb[0]))) for obj in _objects(self.scene)
                if obj is not floor and obj.name not in active and not obj.visual_only)
            self.journal.write('pair_' + first['id'] + second['id'] + '_postcheck.json', post_checks)
            self.journal.event('pair_context_finished', post_checks=post_checks, wall_duration_s=time.monotonic() - began,
                simulated_duration_s=float(self.sim.current_time) - start_time)
            if not all(post_checks.values()):
                raise ValueError('Native source trial invariants changed: ' + str(post_checks))

    def run(self, *, pairs=None, protocol=None):
        """Persist each completed pair's receipt, even when a later pair fails."""
        self._owner()
        from wia.authoring.converters.liquid_capacity_inventory import calibrate_native_pairs, load_source_fill_protocol
        selected = pairs if pairs is not None else select_pairs(self.receipt)
        protocol = dict(protocol or load_source_fill_protocol())
        original_fill = protocol['fill_both']
        def measured_fill(*args, **kwargs):
            t0, s0 = time.monotonic(), float(self.sim.current_time)
            self.journal.event('source_fill_started', native_time_s=s0)
            scope = self.sim.render_on_step(False) if self.render_policy['mode'] == 'render_on_step_false' else nullcontext()
            prior_render = getattr(self.sim, '_render_on_step', None)
            try:
                with scope:
                    counts = original_fill(*args, **kwargs)
            finally:
                # Installed context lacks try/finally; restore rendering on source failure too.
                if self.render_policy['mode'] == 'render_on_step_false':
                    self.sim._render_on_step = prior_render
            self.journal.event('source_fill_measured', retained_particle_counts=[int(v) for v in counts],
                wall_duration_s=time.monotonic() - t0, simulated_duration_s=float(self.sim.current_time) - s0)
            return counts
        protocol['fill_both'] = measured_fill
        result = copy.deepcopy(self.receipt)
        try:
            for first, second in selected:
                self.journal.current_pair = [first, second]
                result = calibrate_native_pairs(result, self.diagnostic_trial, self.reset_actual_inventory,
                    pairs=[(first, second)], fill_protocol=protocol)
                result['driver'] = {'schema': SCHEMA, 'diagnostic_only': True, 'model_calls': 0,
                    'agent_operation_count': 0, 'render_policy': self.render_policy,
                    'task_sha256': self.session.loaded_task.file_sha256,
                    'source_sha256': self.session.loaded_task.source_sha256,
                    'scene_sha256': self.session.scene_adapter.task.file_sha256,
                    'native_reset_id': self.session.native_reset_id}
                if result['truth'].get('reason') not in (None, 'incomplete_capacity_comparison_graph') and not result['calibration_recipe'].get('native_error'):
                    result['calibration_recipe'].update(status='pending',
                        native_error='capacity_evidence_graph_conflict: ' + str(result['truth']['reason']))
                self.journal.write(f'pair_{first}{second}_receipt.json', result)
                self.journal.write('calibration_receipt.json', result)
                if result['calibration_recipe'].get('native_error') or not result['calibration_recipe'].get('post_calibration_reset_verified'):
                    break
        finally:
            try:
                proof = self.reset_actual_inventory()
            except Exception as error:
                proof = {'schema': SCHEMA, 'accepted': False, 'all_inventory_restored': False,
                    'agent_history_empty': False, 'error': f'{type(error).__name__}: {error}'}
            self.journal.write('final_reset_proof.json', proof)
            result['post_calibration_inventory_reset'] = proof
            if not proof['accepted']:
                result['truth'].update(accepted=False, reason='full_native_diagnostic_baseline_reset_unverified')
                result['calibration_recipe'].update(status='pending', post_calibration_reset_verified=False)
            self.journal.write('calibration_receipt.json', result)
            self.journal.event('finished', truth_available=result['truth'].get('accepted', False), reset_verified=proof['accepted'])
        return result

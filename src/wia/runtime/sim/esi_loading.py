"""Profile native scene loading and cache identical rigid mesh calculations.

Mesh reuse is enabled by default; set SHOWHARNESS_ESI_MESH_CACHE=0 to disable.
Set SHOWHARNESS_ESI_LOAD_PROFILE=/absolute/output/directory to save timings.
The hooks exist only during ESISession.reset; the installed SDK is unchanged.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np


def release_variant_preprocess(task, module_name, question_path, state):
    """Honor an exact released source variant excluded only by the upstream filename filter."""
    if (state.get('skip_reason') == 'storage_active_single_only_skips_variant_json'
            and module_name == 'tasks.physical_structure.rigid_containment'
            and getattr(task, 'release_v1', False)
            and Path(question_path).resolve() == Path(task.source_path).resolve()):
        task.verify_unchanged()
        return {**{key: value for key, value in state.items() if key != 'skip_reason'},
                'preprocess_policy': 'exact_fixed_release_variant', 'source_path': str(question_path)}
    return state


def reconcile_resident_timing(og, env_config, defaults):
    """Rebuild only an empty simulator at the next environment's declared native rates."""
    if not getattr(og, '_wia_resident_episode', False) or og.sim is None:
        return None
    config = {**defaults, **env_config.get('env', {})}
    desired = {name: 1. / float(config[field]) for name, field in
               (('physics_dt', 'physics_frequency'), ('rendering_dt', 'rendering_frequency'),
                ('sim_step_dt', 'action_frequency'))}
    actual = {'physics_dt': og.sim.initial_physics_dt, 'rendering_dt': og.sim.initial_rendering_dt,
              'sim_step_dt': og.sim.get_sim_step_dt()}
    if actual == desired:
        return None
    if og.sim.scenes:
        raise RuntimeError('Cannot change simulator timing while a scene is loaded.')
    application = og.app
    from wia.runtime.commands.run_esi_capture_service import native_stage_cleanup
    with native_stage_cleanup(og):
        og.clear(**desired)
    if og.app is not application or og.sim.scenes:
        raise RuntimeError('Timing reconciliation changed the application or retained a scene.')
    receipt = {'previous': actual, 'requested': desired, 'application_preserved': True}
    print('[resident-simulator-timing] ' + json.dumps(receipt), flush=True)
    return receipt


class MeshGeometryCache:
    """Reuse native triangulation and exact-frame volume/COM results.

    Geometry content, rather than object names, identifies identical meshes.
    World results additionally include the complete native world transform.
    Native transforms, mass properties and convex-hull fallbacks are retained.
    """

    def __init__(self, volume_function, mesh_function, world_pose):
        self.volume_function = volume_function
        self.mesh_function = mesh_function
        self.world_pose = world_pose
        self.meshes, self.volumes = {}, {}
        self.local = threading.local()
        self.stats = defaultdict(int)

    def geometry_key(self, prim):
        mesh_type = prim.GetTypeName()
        attributes = {
            "Mesh": ("points", "faceVertexCounts", "faceVertexIndices"),
            "Sphere": ("radius",), "Cube": ("size",),
            "Cone": ("radius", "height"), "Cylinder": ("radius", "height"),
        }.get(mesh_type)
        if attributes is None:
            return None
        digest = hashlib.blake2b(digest_size=20)
        digest.update(mesh_type.encode())
        for name in attributes:
            value = np.ascontiguousarray(prim.GetAttribute(name).Get())
            digest.update(name.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
            self.stats["geometry_bytes_hashed"] += value.nbytes
        return digest.digest()

    def volume(self, mesh_prim, world_frame=False):
        self.stats["volume_calls"] += 1
        key = self.geometry_key(mesh_prim)
        if key is None:
            return self.volume_function(mesh_prim, world_frame=world_frame)
        transform = None
        if world_frame:
            matrix = np.ascontiguousarray(self.world_pose(mesh_prim.GetPath().pathString), dtype=np.float64)
            transform = matrix.tobytes()
        result_key = (key, transform)
        result = self.volumes.get(result_key)
        if result is None:
            self.stats["volume_misses"] += 1
            previous = getattr(self.local, "geometry_key", None)
            self.local.geometry_key = key
            try:
                result = self.volume_function(mesh_prim, world_frame=world_frame)
            finally:
                self.local.geometry_key = previous
            self.volumes[result_key] = (result[0], result[1].clone())
        else:
            self.stats["world_volume_hits" if world_frame else "local_volume_hits"] += 1
        return result[0], result[1].clone()

    def mesh(self, mesh_prim, include_normals=True, include_texcoord=True, world_frame=False):
        key = getattr(self.local, "geometry_key", None)
        # Other mesh consumers, including deformable geometry, retain native behavior.
        if key is None or include_normals or include_texcoord:
            return self.mesh_function(mesh_prim, include_normals, include_texcoord, world_frame)
        self.stats["mesh_calls"] += 1
        canonical = self.meshes.get(key)
        if canonical is None:
            self.stats["mesh_misses"] += 1
            canonical = self.mesh_function(mesh_prim, False, False, False)
            self.meshes[key] = canonical
        else:
            self.stats["mesh_hits"] += 1
        # The original native calculation can mutate its returned mesh.
        result = canonical.copy()
        if world_frame:
            result.apply_transform(self.world_pose(mesh_prim.GetPath().pathString))
        return result


class LoadingProfiler:
    """Collect nested timings in memory; write once after reset."""

    def __init__(self):
        self.timings = {}
        self.objects = {}
        self.local = threading.local()
        self.started = time.perf_counter()

    def wrapped(self, name, function, *, objects=False):
        @wraps(function)
        def call(*args, **kwargs):
            stack = getattr(self.local, "stack", None)
            if stack is None:
                stack = self.local.stack = []
            origin = name
            if name == "simulator_render" and any(frame[0] == "create_joint" for frame in stack):
                origin = "joint_originated_render"
            frame = [origin, 0.]
            stack.append(frame)
            began = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - began
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                row = self.timings.setdefault(origin, {"calls": 0, "inclusive_s": 0., "exclusive_s": 0.})
                row["calls"] += 1
                row["inclusive_s"] += elapsed
                row["exclusive_s"] += elapsed - frame[1]
                if objects:
                    obj = args[0]
                    key = getattr(obj, "category", "unknown")
                    row = self.objects.setdefault(key, {"calls": 0, "inclusive_s": 0.})
                    row["calls"] += 1
                    row["inclusive_s"] += elapsed
        return call


@contextmanager
def native_loading_context():
    """Temporarily install opt-in loading hooks and restore every patched API."""
    enabled = os.environ.get("SHOWHARNESS_ESI_MESH_CACHE", "1") == "1"
    output = os.environ.get("SHOWHARNESS_ESI_LOAD_PROFILE")
    if not enabled and not output:
        yield
        return
    import omnigibson as og
    import omnigibson.utils.usd_utils as usd
    from omnigibson.envs.env_base import Environment
    from omnigibson.objects.object_base import BaseObject
    from omnigibson.prims.rigid_prim import RigidPrim
    from omnigibson.scenes.scene_base import Scene

    profiler = LoadingProfiler()
    changes = []

    def patch(owner, name, value):
        changes.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def patch_references(name, original, replacement):
        for module_name, module in tuple(sys.modules.items()):
            if module is not None and module_name.startswith("omnigibson.") and vars(module).get(name) is original:
                patch(module, name, replacement)

    def instrument_simulator(simulator):
        # OmniGibson 3.7 defines this class inside its launch function.
        owner = type(simulator)
        for method, label in (("import_scene", "import_scene"), ("render", "simulator_render")):
            patch(owner, method, profiler.wrapped(label, getattr(owner, method)))

    original_launch = og.launch

    def launch(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        instrument_simulator(og.sim)
        return result

    cache = MeshGeometryCache(usd.get_mesh_volume_and_com, usd.mesh_prim_to_trimesh_mesh,
                              usd.PoseAPI.get_world_pose_with_scale)
    volume_results = {}
    error = None
    try:
        mesh_function = cache.mesh if enabled else usd.mesh_prim_to_trimesh_mesh
        volume_function = cache.volume if enabled else usd.get_mesh_volume_and_com

        def measured_volume(mesh_prim, world_frame=False):
            result = volume_function(mesh_prim, world_frame=world_frame)
            if output:
                key = f"{mesh_prim.GetPath().pathString}|world={world_frame}"
                volume_results[key] = {"volume": float(result[0]), "center_of_mass": result[1].tolist()}
            return result

        patch_references("mesh_prim_to_trimesh_mesh", usd.mesh_prim_to_trimesh_mesh,
                         profiler.wrapped("mesh_conversion", mesh_function))
        patch_references("get_mesh_volume_and_com", usd.get_mesh_volume_and_com,
                         profiler.wrapped("mesh_volume_com", measured_volume))
        patch_references("create_joint", usd.create_joint, profiler.wrapped("create_joint", usd.create_joint))
        if og.sim is None:
            patch(og, "launch", profiler.wrapped("simulator_launch", launch))
        else:
            instrument_simulator(og.sim)
        for owner, method, label, objects in (
            (Environment, "load", "environment_load", False),
            (Scene, "prebuild", "scene_prebuild", False),
            (Scene, "_load_scene_prim_with_objects", "scene_prim_and_objects", False),
            (BaseObject, "load", "object_load", True),
            (RigidPrim, "update_meshes", "rigid_update_meshes", False),
        ):
            patch(owner, method, profiler.wrapped(label, getattr(owner, method), objects=objects))
        yield
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for owner, name, original in reversed(changes):
            setattr(owner, name, original)
        if output:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            report = {"mesh_cache_enabled": enabled, "reset_scope_wall_s": time.perf_counter() - profiler.started,
                      "error": error, "timing_note": "Inclusive timings overlap; exclusive timings exclude instrumented child calls.",
                      "timings": profiler.timings, "object_load_by_category": profiler.objects,
                      "mesh_cache": dict(cache.stats), "unique_meshes": len(cache.meshes),
                      "exact_frame_volume_results": len(cache.volumes)}
            (directory / "loading_profile.json").write_text(json.dumps(report, indent=2) + "\n")
            (directory / "mesh_numerical_snapshot.json").write_text(json.dumps(volume_results, indent=2) + "\n")

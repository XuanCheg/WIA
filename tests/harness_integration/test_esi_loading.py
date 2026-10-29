"""Check geometry reuse against native mass-property and transform recipes.

Run in the retained behavior environment: python -m pytest -q tests/test_esi_loading.py
These checks do not import or launch OmniGibson or Isaac Sim.
"""
from types import SimpleNamespace

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
torch = pytest.importorskip("torch")

from wia.runtime.sim.esi_loading import MeshGeometryCache


def setup_cache(mesh):
    class Prim:
        def __init__(self, path, matrix=None):
            self.path = path
            self.matrix = np.eye(4) if matrix is None else matrix
            self.vertices = mesh.vertices.astype(np.float32).copy()
            self.faces = mesh.faces.astype(np.int32).copy()

        def GetTypeName(self):
            return "Mesh"

        def GetPath(self):
            return SimpleNamespace(pathString=self.path)

        def GetAttribute(self, name):
            value = {"points": self.vertices, "faceVertexCounts": np.full(len(self.faces), 3, np.int32),
                     "faceVertexIndices": self.faces.flatten()}[name]
            return SimpleNamespace(Get=lambda: value)

    prims = {}
    conversions = []

    def native_mesh(prim, include_normals=True, include_texcoord=True, world_frame=False):
        conversions.append(prim.path)
        result = trimesh.Trimesh(vertices=prim.vertices, faces=prim.faces)
        if world_frame:
            result.apply_transform(prim.matrix)
        return result

    def native_volume(prim, world_frame=False, converter=None):
        result = (converter or native_mesh)(prim, False, False, world_frame)
        if result.is_volume:
            volume, com = result.volume, result.center_mass
        else:
            try:
                hull = result.convex_hull
                volume, com = hull.volume, hull.center_mass
            except Exception:
                volume, com = 0., np.zeros(3)
        return volume, torch.as_tensor(np.asarray(com).copy(), dtype=torch.float32)

    cache = MeshGeometryCache(lambda prim, world_frame=False: native_volume(prim, world_frame, cache.mesh),
                              native_mesh, lambda path: prims[path].matrix)

    def make(path, matrix=None):
        prims[path] = Prim(path, matrix)
        return prims[path]

    return cache, make, native_volume, conversions


@pytest.mark.parametrize("mesh", [trimesh.creation.box([.7, .5, .3]), trimesh.creation.icosphere(subdivisions=1)])
def test_native_world_scale_rotation_translation_and_reflection(mesh):
    cache, make, reference, conversions = setup_cache(mesh)
    matrix = trimesh.transformations.rotation_matrix(.43, [1, 2, 3])
    matrix[:3, :3] = matrix[:3, :3] @ np.diag([.4, 1.7, .8])
    matrix[:3, 3] = [23.5, -8.2, 1.1]
    first, second = make('/a'), make('/b', matrix)
    for prim in (first, second):
        for world_frame in (False, True):
            expected = reference(prim, world_frame)
            actual = cache.volume(prim, world_frame)
            assert actual[0] == expected[0]
            assert torch.equal(actual[1], expected[1])
    assert cache.stats['mesh_misses'] == 1
    assert cache.stats['local_volume_hits'] == 1
    # Mutated returned COM values cannot corrupt the shared result.
    cache.volume(first)[1].fill_(999)
    assert torch.equal(cache.volume(first)[1], reference(first)[1])
    second.matrix[:3, 0] *= -1
    actual, expected = cache.volume(second, True), reference(second, True)
    assert actual[0] == expected[0]
    assert torch.equal(actual[1], expected[1])


def test_geometry_changes_invalidate_cache_and_instance_meshes_are_copied():
    cache, make, reference, conversions = setup_cache(trimesh.creation.box())
    first, second = make('/a'), make('/b')
    cache.volume(first)
    second.vertices[:, 0] *= 2
    actual = cache.volume(second)
    assert actual[0] == reference(second)[0] == 2.
    assert cache.stats['mesh_misses'] == 2
    first.faces = first.faces[::-1].copy()
    assert cache.volume(first)[0] == reference(first)[0]
    assert cache.stats['mesh_misses'] == 3


@pytest.mark.parametrize('flat', [False, True])
def test_convex_hull_and_degenerate_native_fallback(flat):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    if flat:
        mesh.vertices[:, 2] = 0
    cache, make, reference, conversions = setup_cache(mesh)
    matrix = np.diag([2., .5, 3., 1.])
    matrix[:3, 3] = [1, 2, 3]
    for prim in (make('/a'), make('/b', matrix)):
        actual, expected = cache.volume(prim, True), reference(prim, True)
        assert actual[0] == expected[0]
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0, equal_nan=True)

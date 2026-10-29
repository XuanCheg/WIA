"""Check read-only USD material diagnostics with mocked installed API contracts.

Run: PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q \
  -p no:cacheprovider tests/test_wia_ramp_material_bindings.py tests/test_wia_ramp_material_math.py
No USD stage, simulator, GPU or model workload is created.
"""
import json
from types import SimpleNamespace as NS

import pytest

import wia.runtime.commands.check_wia_ramp_material as calibration


class Attribute:
    def __init__(self, value, *, authored=True):
        self.value, self.authored = value, authored

    def Get(self):
        return self.value

    def HasAuthoredValueOpinion(self):
        return self.authored


class Prim:
    def __init__(self, path):
        self.path = path

    def GetPath(self):
        return self.path

    def IsValid(self):
        return True

    def HasAPI(self, api):
        return True


class Material:
    def __init__(self, prim):
        self.prim = prim

    def __bool__(self):
        return self.prim is not None

    def GetPrim(self):
        return self.prim


def install_fixture(monkeypatch, *, inherited=True, missing=False, cached_values=(.8, .6, .1),
                    resolved_values=(.8, .6, .1), failure=None):
    from wia.authoring.converters import physical_dynamics_inclined_plane
    mesh_prim, material_prim = Prim('/World/ramp/link/collision'), Prim('/World/Looks/resolved_physics')
    owner = Prim('/World/ramp') if inherited else mesh_prim
    relationship = NS(GetPath=lambda: owner.path + '.material:binding:physics', GetPrim=lambda: owner,
                      GetTargets=lambda: [material_prim.path], GetMetadata=lambda key: 'strongerThanDescendants')
    calls = []

    class BindingAPI:
        def __init__(self, prim):
            assert prim is mesh_prim

        def GetDirectBinding(self, *, materialPurpose):
            calls.append(('direct', materialPurpose))
            return NS(GetMaterialPath=lambda: '' if inherited or missing else material_prim.path,
                      GetBindingRel=lambda: None if inherited or missing else relationship)

        def ComputeBoundMaterial(self, *, materialPurpose):
            calls.append(('resolved', materialPurpose))
            if failure == 'resolution':
                raise RuntimeError('Optional bound-material query failed.')
            return (Material(None), None) if missing else (Material(material_prim), relationship)

        @staticmethod
        def GetMaterialBindingStrength(relation):
            assert relation is relationship
            return 'strongerThanDescendants'

    class MaterialAPI:
        def __init__(self, prim):
            assert prim is material_prim

        def GetStaticFrictionAttr(self):
            return Attribute(resolved_values[0])

        def GetDynamicFrictionAttr(self):
            return Attribute(resolved_values[1])

        def GetRestitutionAttr(self):
            return Attribute(resolved_values[2])

    class PhysxMaterialAPI:
        def __init__(self, prim):
            if failure == 'physx_api':
                raise RuntimeError('Optional PhysxSchema API unavailable.')
            assert prim is material_prim

        def GetFrictionCombineModeAttr(self):
            return Attribute('average', authored=False)

        def GetRestitutionCombineModeAttr(self):
            return Attribute('max')

    class CollisionAPI:
        def __init__(self, prim):
            assert prim is mesh_prim

        def GetCollisionEnabledAttr(self):
            return Attribute(True)

    class MeshCollisionAPI:
        def __init__(self, prim):
            if failure == 'collision_api':
                raise RuntimeError('Optional collision approximation unavailable.')
            assert prim is mesh_prim

        def GetApproximationAttr(self):
            return Attribute('convexHull')

    modules = {'UsdShade': NS(MaterialBindingAPI=BindingAPI),
               'UsdPhysics': NS(MaterialAPI=MaterialAPI, CollisionAPI=CollisionAPI, MeshCollisionAPI=MeshCollisionAPI),
               'PhysxSchema': NS(PhysxMaterialAPI=PhysxMaterialAPI), 'physx_schema_error': None}
    if failure == 'modules':
        def unavailable():
            raise ImportError('USD is not loaded in this CPU process.')
        monkeypatch.setattr(calibration, '_usd_material_modules', unavailable)
    else:
        monkeypatch.setattr(calibration, '_usd_material_modules', lambda: modules)
    cached = NS(prim_path='/World/Looks/cached_physics', get_static_friction=lambda: cached_values[0],
                get_dynamic_friction=lambda: cached_values[1], get_restitution=lambda: cached_values[2])
    mesh = NS(prim_path=mesh_prim.path, prim=mesh_prim, _applied_physics_material=cached)
    obj = NS(links={'link': NS(collision_meshes={'mesh': mesh})})
    legacy = [{'mesh': mesh.prim_path, 'static_friction': cached_values[0], 'dynamic_friction': cached_values[1]}]
    monkeypatch.setattr(physical_dynamics_inclined_plane, '_friction', lambda value: legacy)
    return obj, calls, legacy


def test_inherited_binding_resolves_despite_no_direct_binding_and_records_authored_values(monkeypatch):
    obj, calls, legacy = install_fixture(monkeypatch)
    result = calibration._materials({'ramp': obj})['ramp']
    assert result['available'] is True and result['collision_materials'] == legacy
    row = result['usd_binding_diagnostics']['collision_prims'][0]
    binding = row['usd_physics_binding']
    assert calls == [('direct', 'physics'), ('resolved', 'physics')]
    assert row['direct_physics_binding']['available'] is False
    assert binding['available'] is True
    assert binding['resolved_material_path'] == '/World/Looks/resolved_physics'
    assert binding['resolved_binding_is_on_collision_prim'] is False
    assert binding['resolved_relationship']['prim_path'] == '/World/ramp'
    assert binding['resolved_relationship']['binding_strength'] == 'strongerThanDescendants'
    assert binding['attributes']['static_friction']['value'] == .8
    assert binding['attributes']['static_friction']['authored'] is True
    assert binding['combine_modes']['friction']['value'] == 'average'
    assert binding['combine_modes']['friction']['authored'] is False
    assert binding['combine_modes']['restitution']['value'] == 'max'
    assert row['collision_attributes']['collision_enabled']['value'] is True
    assert row['collision_attributes']['approximation']['value'] == 'convexHull'
    assert result['usd_binding_diagnostics']['effective_solver_coefficients_verified'] is False
    assert calibration.MATERIAL_BINDING_DIAGNOSTIC_POLICY['solver_material_consumption_verified'] is False
    json.dumps(result, allow_nan=False)


def test_missing_resolved_binding_does_not_invalidate_or_replace_cached_getters(monkeypatch):
    obj, _, legacy = install_fixture(monkeypatch, missing=True)
    result = calibration._materials({'ramp': obj})['ramp']
    row = result['usd_binding_diagnostics']['collision_prims'][0]
    assert result['available'] is True and result['collision_materials'] == legacy
    assert row['cached_wrapper']['available'] is True
    assert row['usd_physics_binding']['available'] is False
    assert row['cached_vs_resolved']['attributes']['static_friction']['available'] is False
    assert row['collision_attributes']['collision_enabled']['available'] is True


def test_direct_binding_strength_and_cached_resolved_disagreement_remain_visible(monkeypatch):
    obj, _, _ = install_fixture(monkeypatch, inherited=False, cached_values=(0., 0., .1),
                                resolved_values=(2.744, 1.7137, .1))
    result = calibration._materials({'ramp': obj})['ramp']
    row = result['usd_binding_diagnostics']['collision_prims'][0]
    assert row['direct_physics_binding']['material_path'] == '/World/Looks/resolved_physics'
    assert row['direct_physics_binding']['relationship']['binding_strength'] == 'strongerThanDescendants'
    assert row['usd_physics_binding']['resolved_binding_is_on_collision_prim'] is True
    assert row['cached_wrapper']['attributes']['static_friction']['value'] == 0.
    assert row['usd_physics_binding']['attributes']['static_friction']['value'] == 2.744
    assert row['cached_vs_resolved']['material_paths_equal'] is False
    comparison = row['cached_vs_resolved']['attributes']['static_friction']
    assert comparison['exact_values_equal'] is False
    assert comparison['cached_minus_resolved'] == -2.744
    assert row['effective_solver_coefficients_verified'] is False


@pytest.mark.parametrize('failure', ['modules', 'physx_api', 'collision_api', 'resolution'])
def test_optional_usd_failures_are_explicit_and_never_abort_legacy_material_diagnostics(monkeypatch, failure):
    obj, _, legacy = install_fixture(monkeypatch, failure=failure)
    result = calibration._materials({'ramp': obj})['ramp']
    assert result['available'] is True and result['collision_materials'] == legacy
    diagnostics = result['usd_binding_diagnostics']
    row = diagnostics['collision_prims'][0]
    assert row['cached_wrapper']['available'] is True
    if failure == 'modules':
        assert diagnostics['available'] is False and diagnostics['errors']
    elif failure == 'physx_api':
        assert row['usd_physics_binding']['available'] is True
        assert row['usd_physics_binding']['combine_modes']['available'] is False
        assert row['collision_attributes']['collision_enabled']['available'] is True
    elif failure == 'collision_api':
        assert row['usd_physics_binding']['available'] is True
        assert row['collision_attributes']['approximation']['available'] is False
        assert row['collision_attributes']['approximation']['errors']
    else:
        assert row['usd_physics_binding']['available'] is False
        assert row['usd_physics_binding']['errors']
        assert row['collision_attributes']['collision_enabled']['available'] is True
    json.dumps(result, allow_nan=False)

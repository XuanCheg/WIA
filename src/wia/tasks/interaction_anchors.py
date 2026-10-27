"""Preserve native lighting and distance anchors during initialization repairs.

Run through scripts/run_esi_capture_worker.py; these hooks never execute actions
or alter source assets. Readback checks validate each proposed initial layout.
"""
from __future__ import annotations

import numpy as np


class InteractionAnchors:
    def __init__(self, session, policy, group):
        self.session, self.policy, self.group = session, policy, group
        self.lights = []
        self.reference = None
        self.reference_before = None
        self.liquid = None
        if policy.get('anchor_policy') == 'liquid_volume_common_translation':
            from .interaction_liquid_anchors import LiquidAnchors
            self.liquid = LiquidAnchors(session)
        evidence = policy.get('fixed_reference_evidence')
        if evidence:
            from wia.authoring.converters.metric_comparison_spatial_distance import resolve_fixed_distance_reference
            self.reference, self.reference_before = resolve_fixed_distance_reference(session, evidence)
            if self.reference is None:
                raise ValueError('The original fixed distance anchor is not uniquely resolved: '
                                 + str(self.reference_before))
        if policy.get('anchor_policy') == 'illumination_common_translation':
            from pxr import Usd, UsdGeom
            cycle = getattr(session, '_illumination_variability_cycle', None)
            if not cycle or not cycle.lights:
                raise ValueError('The original native light cycle is required before layout repair.')
            self.light_paths = [light['path'] for light in cycle.lights]
            roots = [obj.prim_path for obj in group.values()]
            self.descendants = [path for path in self.light_paths
                                if any(path == root or path.startswith(root + '/') for root in roots)]
            for light in cycle.lights:
                if light['path'] in self.descendants:
                    continue
                prim = light['prim']
                xform = UsdGeom.Xformable(prim)
                if not xform:
                    raise ValueError('An unbound original light has no native transform.')
                operations = xform.GetOrderedXformOps()
                self.lights.append({'xform': xform, 'operations': operations,
                    'values': [(op.GetAttr(), op.GetAttr().Get()) for op in operations],
                    'world': xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                    'parent': UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                    'path': light['path']})

    def restore(self):
        if self.liquid is not None:
            self.liquid.restore()
        for row in self.lights:
            for attr, value in row['values']:
                attr.Set(value)
            row['xform'].SetXformOpOrder(row['operations'])

    def apply(self, deltas):
        """Move unbound lights once; descendant lights follow their bound lamp."""
        if self.liquid is not None:
            return self.liquid.apply(deltas)
        if self.policy.get('anchor_policy') != 'illumination_common_translation':
            return {}
        from pxr import Gf, Usd
        values = list(deltas.values())
        delta = np.asarray(values[0], dtype=float)
        if (not np.allclose(delta[:2], 0., atol=1e-7)
                or any(not np.allclose(value, delta, atol=1e-7) for value in values)):
            raise ValueError('Original illumination anchors require a common Z-only translation.')
        for row in self.lights:
            world = Gf.Matrix4d(row['world'])
            world.SetTranslateOnly(world.ExtractTranslation() + Gf.Vec3d(*delta.tolist()))
            local = world * row['parent'].GetInverse()
            row['xform'].MakeMatrixXform().Set(local)
            measured = row['xform'].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            expected = np.array(row['world'].ExtractTranslation()) + delta
            if not np.allclose(np.array(measured.ExtractTranslation()), expected, atol=1e-5):
                raise ValueError('An original stage light did not receive the exact common translation.')
        return {'anchor_translation_m': delta.tolist(),
            'transformed_bound_anchor_names': [obj.name for key, obj in self.group.items() if key.startswith('lamp_')],
            'transformed_light_prim_paths': [row['path'] for row in self.lights],
            'skipped_descendant_light_paths': self.descendants}

    def measure_reference(self):
        if self.reference is None:
            return {}
        from wia.authoring.converters.metric_comparison_spatial_distance import resolve_fixed_distance_reference
        obj, current = resolve_fixed_distance_reference(self.session, self.policy['fixed_reference_evidence'])
        if obj is not self.reference:
            raise ValueError('The fixed scene-native distance anchor changed during layout repair.')
        return {'fixed_reference_before': self.reference_before, 'fixed_reference_after': current,
                'fixed_reference_identity_unchanged': True}

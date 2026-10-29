"""Translate the original Liquid Volume pool coherently during initialization.

Run via src/wia/runtime/commands/run_esi_capture_worker.py. This preserves the source water,
four native pool planes, cached fill positions and volume counts. It performs
no filling, pouring, robot action or physical-success evaluation.
"""
from __future__ import annotations

import copy

import numpy as np


class LiquidAnchors:
    def __init__(self, session):
        from pxr import Usd, UsdGeom
        from .interaction_setup import array
        self.session, self.state = session, session.task_state
        self.water = self.state.get('water')
        required = ('free_pos1', 'free_pos2', 'dip_pos1', 'dip_pos2', 'box_floor_z', 'box_index_offset')
        if self.water is None or any(key not in self.state for key in required):
            raise ValueError('The original Liquid Volume pool state is unavailable.')
        self.cached = {key: copy.deepcopy(self.state[key]) for key in required}
        self.positions, self.orientations = self.water.get_particles_position_orientation()
        self.positions, self.orientations = copy.deepcopy(self.positions), copy.deepcopy(self.orientations)
        self.transforms = []
        stage = session.pipeline.og.sim.stage
        for index in range(int(self.state['box_index_offset']), int(self.state['box_index_offset']) + 4):
            path = f'/World/plane_{index}'
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise ValueError('An original synthetic pool plane is missing: ' + path)
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            self.transforms.append({'path': path, 'xform': xform, 'ops': ops,
                'values': [(op.GetAttr(), op.GetAttr().Get()) for op in ops],
                'world': xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
                'parent': UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())})
        self.original_count = len(array(self.positions).reshape(-1, 3))

    def restore(self):
        import torch
        for row in self.transforms:
            for attr, value in row['values']:
                attr.Set(value)
            row['xform'].SetXformOpOrder(row['ops'])
        self.state.update(copy.deepcopy(self.cached))
        if self.original_count:
            self.water.set_particles_position_orientation(positions=torch.as_tensor(self.positions),
                                                           orientations=torch.as_tensor(self.orientations))

    def apply(self, deltas):
        import torch
        from pxr import Gf, Usd
        from .interaction_setup import array
        values = [array(value) for value in deltas.values()]
        delta = values[0] if values else np.zeros(3)
        if not np.allclose(delta[:2], 0., atol=1e-7) or any(not np.allclose(value, delta, atol=1e-5) for value in values):
            raise ValueError('Liquid pool anchors require one common vertical translation.')
        for row in self.transforms:
            world = Gf.Matrix4d(row['world'])
            world.SetTranslateOnly(world.ExtractTranslation() + Gf.Vec3d(*delta.tolist()))
            row['xform'].MakeMatrixXform().Set(world * row['parent'].GetInverse())
            measured = row['xform'].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if not np.allclose(np.array(measured.ExtractTranslation()), np.array(row['world'].ExtractTranslation()) + delta, atol=1e-5):
                raise ValueError('A native liquid pool plane did not receive the common translation.')
        positions, orientations = self.water.get_particles_position_orientation()
        points = array(positions).reshape(-1, 3)
        if len(points) != self.original_count:
            raise ValueError('Pool particle count changed during support initialization.')
        if len(points):
            self.water.set_particles_position_orientation(positions=torch.as_tensor(points + delta, dtype=torch.float32),
                                                           orientations=torch.as_tensor(orientations))
            actual = array(self.water.get_particles_position_orientation()[0]).reshape(-1, 3)
            if not np.allclose(actual, points + delta, atol=1e-5):
                raise ValueError('Native liquid particles did not follow their pool translation.')
        for key in ('free_pos1', 'free_pos2', 'dip_pos1', 'dip_pos2'):
            value = self.cached[key]
            self.state[key] = value + (torch.as_tensor(delta, dtype=value.dtype, device=value.device)
                if isinstance(value, torch.Tensor) else delta)
        self.state['box_floor_z'] = self.cached['box_floor_z'] + float(delta[2])
        return {'anchor_translation_m': delta.tolist(), 'pool_prim_paths': [row['path'] for row in self.transforms],
                'pool_particle_count_preserved': True, 'pool_particle_count': len(points),
                'pool_cached_positions_translated': True, 'liquid_transfer_verified': False}

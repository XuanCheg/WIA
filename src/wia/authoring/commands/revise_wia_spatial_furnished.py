"""Version spatial layouts after native initialization-image review.

Run: behavior/bin/python -B scripts/revise_wia_spatial_furnished.py
The v1 inputs and images remain immutable. This does not execute controls.


Canonical invocation: python -m wia.authoring.commands.revise_wia_spatial_furnished
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

from wia.resources import ROOT
from wia.authoring.commands.design_wia_furnished_round import OUT, dump, inventory, quat


def main():
    design = json.loads((OUT / 'spatial_design_v1.json').read_text())
    for entry in design['entries']:
        entry['variant_id'] = entry['variant_id'].replace('_v1', '_v2')
        layout = entry['layout']
        layout['shared_layout_id'] += '_v2'
        if layout['scene_model'] == 'Beechwood_0_garden':
            # The first native image exposed a stove crossing the island tray.
            # Move the destination to the clear counter along the window wall.
            cx, cy, sx, sy = -3.73, 13.55, .48, 1.2
            top = inventory(layout['scene_model'])['countertop_tpuwys_1']['aabb'][1][2]
            dz = top + .02
            entry['bindings']['tray_table']['name'] = 'countertop_tpuwys_1'
            entry['parameters']['tray'].update(
                usable_world_xy_m=[[cx-sx/2+.02,cy-sy/2+.02],[cx+sx/2-.02,cy+sy/2-.02]],
                deck_top_world_z_m=dz)
            poses = {
                'deck': ([sx,sy,.02],[cx,cy,top+.01]),
                'rim_w': ([.016,sy,.036],[cx-sx/2+.008,cy,dz+.018]),
                'rim_e': ([.016,sy,.036],[cx+sx/2-.008,cy,dz+.018]),
                'rim_s': ([sx,.016,.036],[cx,cy-sy/2+.008,dz+.018]),
                'rim_n': ([sx,.016,.036],[cx,cy+sy/2-.008,dz+.018]),
            }
            for key,(size,position) in poses.items():
                obj = next(o for o in layout['objects'] if o['name']==entry['bindings'][key]['name'])
                obj.update(scale=size,position=position)
            targets = list(entry['parameters']['targets'])
            # Two widely separated exposed positions share the north counter.
            for i,x in enumerate([-1.15,-.25]):
                key=targets[i];name=entry['parameters']['targets'][key]
                obj=next(o for o in layout['objects'] if o['name']==name)
                obj['position']=[x,15.85,top+.3]
                item=next(o for o in layout['supported_items'] if o['name']==name)
                item['support']='countertop_tpuwys_0'
                entry['bindings'][entry['parameters']['initial_supports'][key]]['name']='countertop_tpuwys_0'
            layout['robot_start']={'position':[-2.75,13.55,.02],'orientation':quat(180)}
        if layout['scene_model']=='Merom_0_garden':
            for item in layout['supported_items']:
                if item['support']=='bookcase_owvfik_0':
                    item.update(support_surface_mode='local_raycast',support_alignment_mode='local_raycast')
        entry['label'] += ' v2 (native-reviewed support revision)'
    design['revision_evidence']={
        'kitchen':'v1 native head/object_top show the island stove intersecting the tray.',
        'playroom':'v1 object bottom .774167m versus bookcase whole AABB top .801433m; local ray measures the actual shelf.',
        'validation_scope':'Initialization only; no robot action or physical task success.'}
    dump(OUT/'spatial_design_v2.json',design)


if __name__=='__main__':
    main()

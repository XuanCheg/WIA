"""Read installed scene/asset bounds on CPU for WIA layout design.

Run with behavior Python and omni.usd.libs bin/bin-deps on LD_LIBRARY_PATH:
python -B scripts/inspect_wia_furnished_assets.py --output OUTPUT_DIRECTORY
Only numerical metadata is saved; licensed plaintext stays in temporary files.


Canonical invocation: python -m wia.authoring.commands.inspect_wia_furnished_assets --help
"""
from __future__ import annotations
import argparse
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import zipfile
import numpy as np
from scipy.spatial.transform import Rotation

from wia.resources import ROOT
ASSETS = Path('/local_home2/zhouxinyang/data/behavior')
SCENES = ('house_single_floor', 'Beechwood_0_int', 'Beechwood_0_garden',
          'Merom_0_int', 'Merom_0_garden', 'Ihlen_0_int', 'office_cubicles_left')
SUPPORTS = {'breakfast_table','coffee_table','dining_table','desk','countertop',
            'bottom_cabinet','shelf','bookcase','console_table','sofa','floors','bench',
            'conference_table','nightstand'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    # Execution-only bootstrap for externally installed native USD libraries.
    lib=next((Path(sys.prefix)/'lib/python3.10/site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
    sys.path.insert(0,str(lib))
    from pxr import Usd, UsdGeom, UsdPhysics
    from cryptography.fernet import Fernet
    reader=Fernet((ASSETS/'omnigibson.key').read_bytes())
    metadata={}
    def model(category,model):
        key=category+'/'+model
        if key in metadata:return metadata[key]
        path=ASSETS/f'behavior-1k-assets/objects/{category}/{model}/usd/{model}.usdz.encrypted'
        with zipfile.ZipFile(io.BytesIO(reader.decrypt(path.read_bytes()))) as z:
            names=[n for n in z.namelist() if n.endswith('.usd')]
            with tempfile.TemporaryDirectory(prefix='wia_asset_') as temp:
                for name in names:z.extract(name,temp)
                stage=Usd.Stage.Open(str(Path(temp)/names[0]));prim=stage.GetDefaultPrim()
                bbox=prim.GetAttribute('ig:nativeBB').Get()
                offset=prim.GetAttribute('ig:offsetBaseLink').Get()
                cache=UsdGeom.XformCache()
                link=prim.GetChild('base_link')
                if not link.IsValid():
                    link=next(c for c in prim.GetChildren() if c.HasAPI(UsdPhysics.RigidBodyAPI))
                inverse=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(link)).T)
                clouds=[]
                for mesh in stage.Traverse():
                    if mesh.IsA(UsdGeom.Mesh) and mesh.HasAPI(UsdPhysics.CollisionAPI):
                        transform=inverse@np.array(cache.GetLocalToWorldTransform(mesh)).T
                        points=np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),dtype=float)
                        clouds.append(points@transform[:3,:3].T+transform[:3,3])
                points=np.concatenate(clouds)
                row={'category':category,'model':model,'native_bbox_m':list(bbox),
                     'collision_bounds_root_link_m':[points.min(0).tolist(),points.max(0).tolist()],
                     'base_link_offset_m':list(offset),'source':str(path)}
        metadata[key]=row;return row
    for scene in SCENES:
        path=ASSETS/f'behavior-1k-assets/scenes/{scene}/json/{scene}_best.json'
        data=json.loads(path.read_text());states=data['state']['registry']['object_registry'];rows=[]
        for name,info in data['objects_info']['init_info'].items():
            spec=info['args'];state=states[name]['root_link']
            row={'name':name,'category':spec['category'],'model':spec['model'],
                 'rooms':spec.get('in_rooms',[]),'pose':[state['pos'],state['ori']],
                 'scale':spec.get('scale',[1.,1.,1.]),'fixed_base':spec.get('fixed_base',False)}
            if spec['category'] in SUPPORTS:
                try:
                    m=model(spec['category'],spec['model']);lo,hi=np.array(m['collision_bounds_root_link_m'])
                    corners=np.array(list(itertools.product(*zip(lo,hi))))*row['scale']
                    world=Rotation.from_quat(state['ori']).apply(corners)+state['pos']
                    row['aabb']=[world.min(0).tolist(),world.max(0).tolist()]
                    row['bounds_source']='Authored collision mesh bounds transformed by saved root-link pose; not live collision or reachability evidence.'
                except Exception as error:row['bounds_error']=str(error)
            rows.append(row)
        out={'scene':scene,'source_file':str(path),'native_run':False,'objects':rows}
        (args.output/(scene+'.json')).write_text(json.dumps(out,indent=2)+'\n')
        print(scene,len(rows),sum('aabb' in r for r in rows),flush=True)
    for category,asset in [('marjoram_shaker','xlspnl'),('banana','znakxm'),('cola_bottle','oyqdtz'),
                           ('box_of_oatmeal','jtqeef'),('pepper_grinder','tfoogy'),('bottle_of_orange_juice','edltwh')]:
        model(category,asset)
    (args.output/'asset_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':main()

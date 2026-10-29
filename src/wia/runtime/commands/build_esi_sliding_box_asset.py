#!/usr/bin/env python3
"""Build a sliding glass lid; run .venv/bin/python src/wia/runtime/commands/build_esi_sliding_box_asset.py.

Reuses the existing visual/collision mesh generator. The 300 mm passive linear
guide replaces both hinges; the raised handle is rigidly attached to the lid.
"""

from wia.resources import ROOT as WIA_RUNTIME_ROOT
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))
from wia.runtime.commands.build_esi_glass_box_asset import mesh


def main():
    base = [mesh('glass_floor', (0, 0, .003), (.36, .28, .006), glass=True),
            mesh('glass_front', (0, -.137, .086), (.36, .006, .16), glass=True),
            mesh('glass_back', (0, .137, .086), (.36, .006, .16), glass=True),
            mesh('glass_left', (-.177, 0, .086), (.006, .268, .16), glass=True),
            mesh('glass_right', (.177, 0, .086), (.006, .268, .16), glass=True)]
    for i, (x, y) in enumerate([(-.177, -.137), (.177, -.137), (-.177, .137), (.177, .137)]):
        base.append(mesh(f'corner_{i}', (x, y, .086), (.01, .01, .16)))
    for side, x in [('left', -.187), ('right', .187)]:
        base.append(mesh(f'slide_guide_{side}', (x, .15, .166), (.010, .60, .020)))
    lid = [mesh('glass_lid', (0, 0, 0), (.36, .28, .006), glass=True),
           mesh('lid_front_rail', (0, -.135, .002), (.36, .01, .01)),
           mesh('lid_back_rail', (0, .135, .002), (.36, .01, .01)),
           mesh('handle_bar', (0, -.085, .093), (.08, .020, .020), rounded=True),
           mesh('handle_left_post', (-.029, -.085, .042), (.016, .020, .090)),
           mesh('handle_right_post', (.029, -.085, .042), (.016, .020, .090))]
    materials = []
    for name, color, metallic, roughness in [('Frame', (.10, .27, .32), .35, .3),
                                            ('Handle', (.16, .25, .28), .2, .4)]:
        materials.append(f'''        def Material "{name}" {{
            token outputs:mdl:surface.connect = </GlassBox/Looks/{name}/Shader.outputs:out>
            def Shader "Shader" {{
                uniform token info:id = "OmniPBR"
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = {color}
                float inputs:metallic_constant = {metallic}
                float inputs:reflection_roughness_constant = {roughness}
                bool inputs:enable_opacity = false
                token outputs:out
            }}
        }}
''')
    text = '''#usda 1.0
(
    defaultPrim = "GlassBox"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "GlassBox" {
    def Scope "Looks" {
        def Material "Glass" {
            token outputs:surface.connect = </GlassBox/Looks/Glass/Shader.outputs:surface>
            def Shader "Shader" {
                uniform token info:id = "UsdPreviewSurface"
                float inputs:opacity = 0.15
                token outputs:surface
            }
        }
'''+''.join(materials)+'''    }
    def Xform "base_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 2
'''+''.join(base)+'''
        def PhysicsPrismaticJoint "lid_slide" (prepend apiSchemas = ["PhysxJointAPI", "PhysicsJointStateAPI:linear"]) {
            rel physics:body0 = </GlassBox/base_link>
            rel physics:body1 = </GlassBox/lid_link>
            uniform token physics:axis = "Y"
            point3f physics:localPos0 = (0, 0, 0.170)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = 0
            float physics:upperLimit = 0.30
            float physxJoint:jointFriction = 0.02
            bool physics:collisionEnabled = false
            float state:linear:physics:position = 0
            float state:linear:physics:velocity = 0
        }
    }
    def Xform "lid_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 0.08
        double3 xformOp:translate = (0, 0, 0.170)
        uniform token[] xformOpOrder = ["xformOp:translate"]
'''+''.join(lid)+'''    }
}
'''
    output = ROOT / 'assets/esi/glass_box_sliding.usda'
    output.write_text(text, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()

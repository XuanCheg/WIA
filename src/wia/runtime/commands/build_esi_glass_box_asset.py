#!/usr/bin/env python3
"""Build a text USD glass box. Run .venv/bin/python src/wia/runtime/commands/build_esi_glass_box_asset.py.

This CPU-only generator authors geometry and a passive 0–110 degree hinge; it
does not import or start a simulator. Native glass binding occurs at task setup.
"""

from wia.resources import ROOT as WIA_RUNTIME_ROOT
from pathlib import Path
import math

ROOT = WIA_RUNTIME_ROOT


def mesh(name, center, size, *, glass=False, collision=True, rounded=False):
    vertices = [(x * size[0] / 2, y * size[1] / 2, z * size[2] / 2)
                for x, y, z in [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]]
    indices = [0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,1,2,6,1,6,5,2,3,7,2,7,6,3,0,4,3,4,7]
    if rounded:
        # A horizontal round bar leaves clearance and can roll against fingers.
        count = 16
        vertices = [(side * size[0] / 2,
                     math.cos(2 * math.pi * i / count) * size[1] / 2,
                     math.sin(2 * math.pi * i / count) * size[2] / 2)
                    for side in (-1, 1) for i in range(count)]
        indices = []
        for i in range(1, count - 1):
            indices += [0, i + 1, i, count, count + i, count + i + 1]
        for i in range(count):
            j = (i + 1) % count
            indices += [i, j, count + j, i, count + j, count + i]
    points = ', '.join('(' + ', '.join(f'{v:.6f}' for v in point) + ')' for point in vertices)
    face_normals = []
    for start in range(0, len(indices), 3):
        a, b, c = (vertices[index] for index in indices[start:start + 3])
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        n = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
        length = math.sqrt(sum(value * value for value in n))
        face_normals.extend([tuple(value / length for value in n)] * 3)
    normals = ', '.join(str(normal) for normal in face_normals)
    extent = [tuple(min(point[i] for point in vertices) for i in range(3)),
              tuple(max(point[i] for point in vertices) for i in range(3))]
    schemas = ('"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "PhysxCollisionAPI", '
               if collision else '') + '"MaterialBindingAPI"'
    physics = '''            bool physics:collisionEnabled = true
            token physics:approximation = "convexHull"
            float physxCollision:contactOffset = 0.001
            float physxCollision:restOffset = 0
''' if collision else ''
    result = f'''        def Mesh "{'collision_' if collision else ''}{name}" (
            prepend apiSchemas = [{schemas}]
        ) {{
            double3 xformOp:translate = {center}
            uniform token[] xformOpOrder = ["xformOp:translate"]
            point3f[] points = [{points}]
            float3[] extent = [{extent[0]}, {extent[1]}]
            int[] faceVertexCounts = [{','.join('3' for _ in range(len(indices) // 3))}]
            int[] faceVertexIndices = [{','.join(str(index) for index in indices)}]
            normal3f[] normals = [{normals}] (interpolation = "faceVarying")
            uniform token subdivisionScheme = "none"
            uniform token orientation = "rightHanded"
            bool doubleSided = {'true' if name.startswith('handle') else 'false'}
{physics}
            rel material:binding = </GlassBox/Looks/{'Glass' if glass else 'Handle' if name.startswith('handle') else 'Frame'}>
        }}
'''
    return result + mesh(name, center, size, glass=glass, collision=False, rounded=rounded) if collision else result


def main():
    base = [mesh('glass_floor', (0, 0, .003), (.36, .28, .006), glass=True),
            mesh('glass_front', (0, -.137, .086), (.36, .006, .16), glass=True),
            mesh('glass_back', (0, .137, .086), (.36, .006, .16), glass=True),
            mesh('glass_left', (-.177, 0, .086), (.006, .268, .16), glass=True),
            mesh('glass_right', (.177, 0, .086), (.006, .268, .16), glass=True)]
    for i, (x, y) in enumerate([(-.177, -.137), (.177, -.137), (-.177, .137), (.177, .137)]):
        base.append(mesh(f'corner_{i}', (x, y, .086), (.01, .01, .16)))
    lid = [mesh('glass_lid', (0, -.14, 0), (.36, .28, .006), glass=True),
           mesh('lid_front_rail', (0, -.275, .002), (.36, .01, .01)),
           mesh('lid_back_rail', (0, -.005, .002), (.36, .01, .01))]
    # A 100 mm open gripper can surround the 80 mm bar. The grip has
    # 80 mm clearance above the closed pane and remains raised when folded.
    handle = [mesh('handle_bar', (0, 0, .085), (.080, .020, .020), rounded=True),
              mesh('handle_left_post', (-.029, 0, .0395), (.016, .020, .079)),
              mesh('handle_right_post', (.029, 0, .0395), (.016, .020, .079))]
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
                color3f inputs:diffuseColor = (0.82, 0.94, 0.98)
                float inputs:opacity = 0.15
                float inputs:roughness = 0.06
                float inputs:ior = 1.45
                token outputs:surface
            }
        }
        def Material "Frame" {
            token outputs:surface.connect = </GlassBox/Looks/Frame/Shader.outputs:surface>
            def Shader "Shader" {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.10, 0.27, 0.32)
                float inputs:metallic = 0.35
                float inputs:roughness = 0.3
                token outputs:surface
            }
        }
        def Material "Handle" {
            token outputs:mdl:surface.connect = </GlassBox/Looks/Handle/Shader.outputs:out>
            def Shader "Shader" {
                uniform token info:id = "OmniPBR"
                uniform token info:implementationSource = "sourceAsset"
                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
                color3f inputs:diffuse_color_constant = (0.16, 0.25, 0.28)
                float inputs:metallic_constant = 0.2
                float inputs:reflection_roughness_constant = 0.4
                bool inputs:enable_opacity = false
                token outputs:out
            }
        }
    }
    def Xform "base_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 2
'''+''.join(base)+'''
        def PhysicsRevoluteJoint "lid_hinge" (prepend apiSchemas = ["PhysxJointAPI", "PhysicsJointStateAPI:angular"]) {
            rel physics:body0 = </GlassBox/base_link>
            rel physics:body1 = </GlassBox/lid_link>
            uniform token physics:axis = "X"
            point3f physics:localPos0 = (0, 0.14, 0.170)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (0, 0, 0, 1)
            quatf physics:localRot1 = (0, 0, 0, 1)
            float physics:lowerLimit = 0
            float physics:upperLimit = 110
            float physxJoint:jointFriction = 0.20
            bool physics:collisionEnabled = false
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
    }
    def Xform "lid_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 0.07
        double3 xformOp:translate = (0, 0.14, 0.170)
        uniform token[] xformOpOrder = ["xformOp:translate"]
'''+''.join(lid)+'''
        def PhysicsRevoluteJoint "handle_hinge" (prepend apiSchemas = ["PhysxJointAPI", "PhysicsJointStateAPI:angular"]) {
            rel physics:body0 = </GlassBox/lid_link>
            rel physics:body1 = </GlassBox/handle_link>
            uniform token physics:axis = "X"
            point3f physics:localPos0 = (0, -0.225, 0.008)
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = (1, 0, 0, 0)
            quatf physics:localRot1 = (1, 0, 0, 0)
            float physics:lowerLimit = -60
            float physics:upperLimit = 60
            float physxJoint:jointFriction = 0.02
            bool physics:collisionEnabled = false
            float state:angular:physics:position = 0
            float state:angular:physics:velocity = 0
        }
    }
    def Xform "handle_link" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 0.01
        double3 xformOp:translate = (0, -0.085, 0.178)
        uniform token[] xformOpOrder = ["xformOp:translate"]
'''+''.join(handle)+'''    }
}
'''
    output = ROOT / 'assets/esi/glass_box_hinged.usda'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()

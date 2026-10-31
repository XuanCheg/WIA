"""Author readable visual labels without physics or robot commands.

Run from the native owner thread after scene initialization::

    from wia.candidates.labels import add_region_labels
    evidence = add_region_labels(session, session.candidate_state["regions"], output_dir)

The CPU USD smoke uses an in-memory stage behind ``session.pipeline.og.sim.stage``.
Pillow writes opaque PNGs; only new meshes, materials and shaders are authored.
Region "back" means world +Y. Quantity-object labels use the measured world AABB
top at authoring time and follow the actual root link, including its full scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _bounds(value):
    result = _array(value)
    if (result.shape != (2, 3) or not np.isfinite(result).all()
            or np.any(result[1] < result[0]) or np.any(result[1, :2] <= result[0, :2])):
        raise ValueError("Labels require a finite world AABB with positive XY dimensions.")
    return result


def _text(value):
    result = " ".join(str(value).replace("_", " ").upper().split())
    if not result or not result.isascii():
        raise ValueError("A nonempty English ASCII label is required.")
    return result


def _unique_prim(stage, base):
    candidate, index = base, 2
    while stage.GetPrimAtPath(candidate).IsValid():
        candidate, index = f"{base}_{index}", index + 1
    return candidate


def _texture(text, output_dir, stem, width_m, height_m):
    from PIL import Image, ImageDraw, ImageFont

    directory = Path(output_dir).expanduser().resolve() / "label_textures"
    directory.mkdir(parents=True, exist_ok=True)
    path, index = directory / f"{stem}.png", 2
    while path.exists():
        path, index = directory / f"{stem}_{index}.png", index + 1
    width, height = 1024, max(96, int(round(1024 * height_m / width_m)))
    image = Image.new("RGB", (width, height), (13, 35, 53))
    draw = ImageDraw.Draw(image)
    padding = max(12, min(width, height) // 12)
    draw.rectangle((3, 3, width - 4, height - 4), outline=(240, 227, 192), width=5)
    lines = text
    words = text.split()
    if len(text) > 22 and len(words) > 1:
        split = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        lines = " ".join(words[:split]) + "\n" + " ".join(words[split:])
    font = None
    for size in range(min(150, height), 7, -1):
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
        box = draw.multiline_textbbox((0, 0), lines, font=font, align="center", spacing=6)
        if box[2] - box[0] <= width - 2 * padding and box[3] - box[1] <= height - 2 * padding:
            break
    else:
        raise ValueError("Label text does not fit its texture at a readable font size.")
    left = (width - (box[2] - box[0])) / 2 - box[0]
    top = (height - (box[3] - box[1])) / 2 - box[1]
    draw.multiline_text((left, top), lines, font=font, fill=(255, 249, 225), align="center", spacing=6)
    image.save(path)
    return path


def _quad(stage, path, points, texture_path):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    points = np.asarray(points, dtype=float)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal /= np.linalg.norm(normal)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, normal))])
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, point)) for point in (points.min(axis=0), points.max(axis=0))])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying).Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

    material = UsdShade.Material.Define(stage, path + "/LabelMaterial")
    surface = UsdShade.Shader.Define(stage, path + "/LabelMaterial/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.)
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    texture = UsdShade.Shader.Define(stage, path + "/LabelMaterial/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    reader = UsdShade.Shader.Define(stage, path + "/LabelMaterial/Coordinates")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    # No CollisionAPI, rigid-body API, physical material, or visibility edits are
    # applied to either this visual-only mesh or any pre-existing scene prim.
    return {"prim_path": str(mesh.GetPath()), "material_path": str(material.GetPath()),
            "texture_path": str(texture_path), "double_sided": True, "collision_authored": False}


def _world_quad(center, width, height):
    x, y, z = center
    return np.asarray([[x - width / 2, y - height / 2, z], [x + width / 2, y - height / 2, z],
                       [x + width / 2, y + height / 2, z], [x - width / 2, y + height / 2, z]])


def add_region_labels(session, regions: dict, output_dir: Path) -> dict:
    """Add English labels in each measured region's back band, preserving its center."""
    from pxr import UsdGeom

    stage = session.pipeline.og.sim.stage
    root = _unique_prim(stage, "/WIARegionLabels")
    UsdGeom.Scope.Define(stage, root)
    labels = []
    for index, (name, region) in enumerate(regions.items(), 1):
        box = _bounds(region["aabb"])
        span = box[1] - box[0]
        width, height = min(.22, .8 * span[0]), min(.07, .22 * span[1])
        gap = min(.008, .04 * span[1])
        center = [float(np.mean(box[:, 0])), float(box[1, 1] - gap - height / 2), float(box[1, 2] + .002)]
        front = region.get('label_front')
        placement = 'world_positive_y_back_band'
        if front is not None:
            axis, sign = int(front['axis']), int(front['sign'])
            if axis not in (0, 1) or sign not in (-1, 1):
                raise ValueError('Shelf front labels require a horizontal axis and signed outward face.')
            width, height = min(.24, .8 * span[1-axis]), .065
            center = [*np.mean(box[:, :2], axis=0).tolist(), float(box[1, 2] - .04)]
            center[axis] = float(front['edge_m']) + sign * .004
            # Front-facing neutral shelf placard; it supplies no collision geometry.
            horizontal = np.zeros(3); horizontal[1-axis] = sign if axis == 0 else -sign
            points = np.asarray([np.asarray(center) + dx * horizontal * width / 2 + np.array([0.,0.,dz*height/2])
                                 for dx,dz in ((-1,-1),(1,-1),(1,1),(-1,1))])
            placement = 'vertical_native_shelf_front'
        else:
            points = _world_quad(center, width, height)
        text = _text(name)
        texture = _texture(text, output_dir, f"region_{index:03d}", width, height)
        row = _quad(stage, f"{root}/region_{index:03d}", points, texture)
        row.update(region_name=name, text=text, support=region["support"], center_world_m=center,
                   dimensions_m=[float(width), float(height)], placement=placement)
        labels.append(row)
    return {"root_prim_path": root, "labels": labels, "visual_only": True, "physics_steps": 0}


def add_object_label(session, obj, text: str, output_dir: Path) -> dict:
    """Attach only the supplied public quantity text to an object's measured top.

    The world-horizontal top label is suitable for upright quantity units. USD's
    complete root-link local-to-world matrix, including scale, is inverted for
    authoring; moving or rotating that link subsequently carries the label.
    No text is inferred from object names, categories, state, or private identity.
    """
    from pxr import Gf, Usd, UsdGeom

    stage = session.pipeline.og.sim.stage
    root_link_path = str(obj.root_link.prim_path)
    root = stage.GetPrimAtPath(root_link_path)
    if not root.IsValid():
        raise ValueError("An existing native object root link is required for an attached label.")
    box = _bounds([_array(value) for value in obj.aabb])
    span = box[1] - box[0]
    width, height = min(.16, .8 * span[0]), min(.05, .7 * span[1])
    center = [float(np.mean(box[:, 0])), float(np.mean(box[:, 1])), float(box[1, 2] + .002)]
    world_points = _world_quad(center, width, height)
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root)
    if abs(transform.GetDeterminant()) < 1e-12:
        raise ValueError("Object root-link transform must be invertible, including its scale.")
    inverse = transform.GetInverse()
    local_points = [list(inverse.Transform(Gf.Vec3d(*map(float, point)))) for point in world_points]
    text = _text(text)
    texture = _texture(text, output_dir, "quantity_label", width, height)
    row = _quad(stage, _unique_prim(stage, root_link_path + "/WIAQuantityLabel"), local_points, texture)
    row.update(text=text, parent_root_link_path=root_link_path, center_world_m=center,
               dimensions_m=[float(width), float(height)], visual_only=True, physics_steps=0,
               placement="measured_world_aabb_top_attached_to_native_root_link")
    return row

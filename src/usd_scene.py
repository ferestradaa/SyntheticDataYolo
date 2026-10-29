import math

import numpy as np

ROUGHNESS_JITTER = 0.30
METALLIC_JITTER = 0.10
BRIGHTNESS_JITTER = 0.15
TINT_JITTER = 0.04

FLOOR_SIZE = 40.0
BACKDROP_SIZE = (10.0, 4.0)
TEXTURE_SCALE = (0.5, 3.0)


def rotation(rx, ry, rz):
    a, b, c = np.radians([rx, ry, rz])
    rot_x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    rot_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rot_z = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def look_at(eye, target, roll):
    from pxr import Gf

    forward = np.array(target, dtype=float) - np.array(eye, dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, [0, 0, 1])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward) * math.cos(roll) + right * math.sin(roll)
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)),
                                   Gf.Vec3d(*map(float, up))).GetInverse()


def perturb(direction, max_angle, rng):
    axis = np.cross(direction, rng.normal(size=3))
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return direction
    axis /= norm
    angle = max_angle * rng.random()
    return direction * math.cos(angle) + np.cross(axis, direction) * math.sin(angle)


def distance_to_segment(point, a, b):
    ab = b - a
    t = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(point - (a + t * ab))


def bounds(prim):
    from pxr import Usd, UsdGeom

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(box.GetMin()), np.array(box.GetMax())


def movable_ops(prim):
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    translate, rotate, scale = xf.AddTranslateOp(), xf.AddRotateXYZOp(), xf.AddScaleOp()
    translate.Set(Gf.Vec3d(0, 0, 0))
    rotate.Set(Gf.Vec3f(0, 0, 0))
    scale.Set(Gf.Vec3f(1, 1, 1))
    return translate, rotate, scale


def set_pose(ops, position, euler_deg, scale):
    from pxr import Gf

    translate, rotate, scale_op = ops
    translate.Set(Gf.Vec3d(*map(float, position)))
    rotate.Set(Gf.Vec3f(*map(float, euler_deg)))
    scale_op.Set(Gf.Vec3f(*map(float, scale)))


def set_visible(prim, visible):
    from pxr import UsdGeom

    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


def is_visible(prim):
    from pxr import UsdGeom

    return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible


def find_instance_roots(stage, label):
    from pxr import Semantics

    found = []
    for prim in stage.Traverse():
        if not prim.HasAPI(Semantics.SemanticsAPI):
            continue
        for prop in prim.GetProperties():
            if not Semantics.SemanticsAPI.IsSemanticsAPIPath(prop.GetPath()):
                continue
            sem = Semantics.SemanticsAPI.Get(prim, prop.SplitName()[1])
            if sem.GetSemanticDataAttr().Get() == label:
                found.append(prim)
                break
    paths = {p.GetPath() for p in found}
    return [p for p in found if not any(a in paths for a in p.GetPath().GetPrefixes()[:-1])]   # top-most only


def collect_materials(root):
    from pxr import Usd, UsdShade, Sdf

    out = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        entry = {"roughness": shader.GetInput("roughness"), "metallic": shader.GetInput("metallic")}
        entry["roughness_base"] = entry["roughness"].Get()
        entry["metallic_base"] = entry["metallic"].Get()

        color = shader.GetInput("diffuseColor")
        source = color.GetConnectedSource()
        if source:
            tex = UsdShade.Shader(source[0].GetPrim())   # textured parts take brightness and tint through the texture scale
            entry["tex_scale"] = tex.GetInput("scale") or tex.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        else:
            entry["color"] = color
            entry["color_base"] = color.Get()
        out.append(entry)
    return out


def jitter_materials(materials, rng):
    from pxr import Gf

    rough = 1 + rng.uniform(-ROUGHNESS_JITTER, ROUGHNESS_JITTER)
    metal = rng.uniform(-METALLIC_JITTER, METALLIC_JITTER)
    bright = 1 + rng.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)
    mult = bright * (1 + rng.uniform(-TINT_JITTER, TINT_JITTER, 3))

    for m in materials:
        m["roughness"].Set(float(np.clip(m["roughness_base"] * rough, 0.02, 1.0)))
        if m["metallic_base"] > 0.05:   # plastic must not turn into metal
            m["metallic"].Set(float(np.clip(m["metallic_base"] + metal, 0.0, 1.0)))
        if "tex_scale" in m:
            m["tex_scale"].Set(Gf.Vec4f(*map(float, mult), 1.0))
        else:
            base = np.array(m["color_base"], dtype=float)
            m["color"].Set(Gf.Vec3f(*map(float, np.clip(base * mult, 0.0, 1.0))))


def add_surfaces():
    import omni.replicator.core as rep

    return [
        rep.create.plane(scale=(FLOOR_SIZE, FLOOR_SIZE, 1), position=(0, 0, 0), name="Floor"),
        rep.create.cube(scale=(1, 1, 1), position=(0, 0, 0.5), name="Support"),
        rep.create.plane(scale=(BACKDROP_SIZE[0], BACKDROP_SIZE[1], 1), position=(0, 3, BACKDROP_SIZE[1] / 2),
                         rotation=(90, 0, 0), name="Backdrop"),
    ]


def randomize_surfaces(surfaces, textures):
    import omni.replicator.core as rep

    lo, hi = TEXTURE_SCALE
    for surface in surfaces:
        with surface:
            rep.randomizer.texture(
                textures=textures,
                texture_scale=rep.distribution.uniform((lo, lo), (hi, hi)),
                texture_rotate=rep.distribution.uniform(0, 360),
                project_uvw=True,
            )

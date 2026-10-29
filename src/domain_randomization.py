import math

import numpy as np

from lighting import LightRig
from usd_scene import (bounds, collect_materials, distance_to_segment, find_instance_roots, is_visible,
                       jitter_materials, look_at, movable_ops, rotation, set_pose, set_visible, BACKDROP_SIZE)

NEGATIVE_FRAME_PROB = 0.05
SPAWN_AREA = ((-0.50, 1.00), (0.50, 1.80))

SIZE_JITTER = (0.8, 1.2)
ASSET_GAP = (0.0, 0.03)
LYING_POSES = [(90, 0), (-90, 0), (0, 90), (0, -90)]   # rotations about x and y that lay an asset on one of its sides
TILT_JITTER = 5.0
FREE_POSE_PROB = 0.30

LAYOUTS = {"supported": 0.45, "ground": 0.15, "floating": 0.40}   # the support is never a constant of the scene
SUPPORT_HEIGHT = (0.30, 1.05)
SUPPORT_MARGIN = (0.01, 0.70)
FLOAT_HEIGHT = (0.40, 1.80)
FLOAT_SPREAD = (0.04, 0.30)
FLOOR_PROB = {"supported": 0.85, "ground": 1.0, "floating": 0.30}
BACKDROP_PROB = {"supported": 0.60, "ground": 0.50, "floating": 0.25}
BACKDROP_GAP = (0.03, 6.0)
BACKDROP_TILT = 8.0

FOCAL_LENGTH_MM = (16.0, 55.0)
ELEVATION_MAX_DEG = 75.0
ELEVATION_SKEW = 2.0            # elevation = max * u ** skew, so most frames are frontal and a few look from above
MAX_ELEVATION_DEG = 80.0
ROLL_SIGMA_DEG = 6.0
ROLL_MAX_DEG = 18.0
SHOT_WEIGHTS = {"close": 0.35, "medium": 0.40, "wide": 0.25}
DISTANCE_MARGIN = {"close": (0.85, 1.3), "medium": (0.9, 1.4), "wide": (1.05, 1.5)}   # below 1 truncates the objects at the borders
DISTANCE_MIN = 0.25

DISTRACTOR_RING = (0.8, 4.5)
DISTRACTOR_CLEARANCE = 1.0
DISTRACTOR_SCALE = (0.7, 1.5)
DISTRACTOR_SHOWN = (0.4, 1.0)
PROP_SIZE = 0.45
PROP_PROB = 0.7
PROPS_MAX = 6
PROP_SCALE = (0.8, 1.3)
FLOAT_PROP_RING = (0.25, 1.2)


class Layout:
    def __init__(self, kind, support_z, half, floor):
        self.kind = kind
        self.support_z = support_z   # None while the assets float, and then half is None too
        self.half = half
        self.floor = floor
        self.backdrop = None


class SceneRandomizer:
    def __init__(self, stage, assets, resolution, hdris=(), seed=None):
        from pxr import Usd, UsdGeom

        self.rng = np.random.default_rng(seed)
        self.assets = assets
        self.spawn_min, self.spawn_max = (np.array(c, dtype=float) for c in SPAWN_AREA)
        self.center_xy = (self.spawn_min + self.spawn_max) / 2

        self.instances = {}
        for label in assets:
            roots = find_instance_roots(stage, label)
            self.instances[label] = [
                {"root": root, "ops": movable_ops(root.GetParent()), "materials": collect_materials(root)}
                for root in roots
            ]
            print(f"[randomizer] {label}: {len(roots)} instances")

        asset_parents = {i["root"].GetParent().GetPath() for items in self.instances.values() for i in items}
        self.distractors = []
        for prim in stage.GetPrimAtPath("/Replicator").GetChildren():
            if not prim.GetName().startswith("Ref_Xform") or prim.GetPath() in asset_parents:
                continue
            low, high = bounds(prim)
            size = high - low
            self.distractors.append({
                "prim": prim, "ops": movable_ops(prim), "z_min": low[2],
                "radius": 0.5 * math.hypot(size[0], size[1]), "small": size.max() <= PROP_SIZE,
            })
        print(f"[randomizer] {len(self.distractors)} distractors, {sum(d['small'] for d in self.distractors)} small")

        self.floor = stage.GetPrimAtPath("/Replicator/Floor_Xform")
        self.support = stage.GetPrimAtPath("/Replicator/Support_Xform")
        self.backdrop = stage.GetPrimAtPath("/Replicator/Backdrop_Xform")
        self.support_ops = movable_ops(self.support)
        self.backdrop_ops = movable_ops(self.backdrop)

        cam_prim = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/Replicator")) if p.IsA(UsdGeom.Camera))
        self.camera = UsdGeom.Camera(cam_prim)
        UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
        parent_xf = UsdGeom.Xformable(cam_prim.GetParent())
        parent_xf.ClearXformOpOrder()
        self.camera_op = parent_xf.AddTransformOp()
        width, height = resolution
        horizontal = self.camera.GetHorizontalApertureAttr().Get()
        self.sensor = (horizontal, horizontal * height / width)

        self.lights = LightRig(stage, hdris)

    def randomize_frame(self):
        negative = bool(self.rng.random() < NEGATIVE_FRAME_PROB)
        layout = self._build_layout()
        placed, labels, centers, radii, spheres = self._place_assets(layout, self._sample_counts(negative))

        if not centers:   # a negative frame still needs something to aim the camera at
            height = layout.support_z if layout.support_z is not None else float(self.rng.uniform(*FLOAT_HEIGHT))
            centers, radii = [np.array([*self.center_xy, height + 0.15])], [0.3]
        eye, target, shot, focus, elevation, roll, distance, focal = self._place_camera(
            np.array(centers), np.array(radii), layout.support_z)

        gap = self._place_backdrop(layout, eye, target)
        props = self._place_distractors(layout, eye, target, spheres)
        style = self.lights.apply(self.rng, target, eye)
        return {"negative": negative, "visible": placed, "focus": sorted({labels[i] for i in focus}) if labels else [],
                "shot": shot, "elevation_deg": round(math.degrees(elevation), 1), "roll_deg": round(math.degrees(roll), 1),
                "distance": round(float(distance), 2), "focal_mm": round(float(focal), 1), "layout": layout.kind,
                "support_z": None if layout.support_z is None else round(layout.support_z, 3), "floor": layout.floor,
                "backdrop_gap": None if gap is None else round(float(gap), 2), "light_style": style, "props": props}

    def _build_layout(self):
        rng = self.rng
        kind = str(rng.choice(list(LAYOUTS), p=list(LAYOUTS.values())))
        floor = bool(rng.random() < FLOOR_PROB[kind])
        if kind == "floating":
            layout = Layout(kind, None, None, floor)
        else:
            support_z = float(rng.uniform(*SUPPORT_HEIGHT)) if kind == "supported" else 0.0
            half = (self.spawn_max - self.spawn_min) / 2 + rng.uniform(*SUPPORT_MARGIN, size=2)
            layout = Layout(kind, support_z, half, floor)
            set_pose(self.support_ops, [*self.center_xy, support_z / 2], (0, 0, 0),
                     (2 * half[0], 2 * half[1], max(support_z, 0.01)))
        set_visible(self.floor, floor)
        set_visible(self.support, kind == "supported")
        return layout

    def _sample_counts(self, negative):
        if negative:
            return {label: 0 for label in self.instances}
        while True:
            counts = {label: int(self.rng.integers(0, len(items) + 1)) for label, items in self.instances.items()}
            if sum(counts.values()) > 0:
                return counts

    def _rest_pose(self, label, euler, scale):
        low, high = self.assets[label]["bbox_min"], self.assets[label]["bbox_max"]
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        matrix = rotation(*euler)
        world = (corners * scale) @ matrix.T
        center = ((low + high) / 2 * scale) @ matrix.T
        return (center, world[:, 2].min(), np.linalg.norm(world[:, :2] - center[:2], axis=1).max(),
                np.linalg.norm(world - center, axis=1).max())

    def _sample_orientation(self, label, free):
        rng = self.rng
        if free:
            return rng.uniform(0, 360, 3)
        rx, ry = LYING_POSES[rng.integers(len(LYING_POSES))] if rng.random() < self.assets[label]["lying"] else (0, 0)
        tilt = rng.uniform(-TILT_JITTER, TILT_JITTER, 2)
        return np.array([rx + tilt[0], ry + tilt[1], rng.uniform(0, 360)])

    def _place_assets(self, layout, wanted):
        rng = self.rng
        queue = [(label, i) for label, count in wanted.items() for i in range(count)]
        placed = {label: 0 for label in self.instances}
        labels, centers, radii, spheres = [], [], [], []
        used = set()
        base_z = float(rng.uniform(*FLOAT_HEIGHT))
        spread = float(rng.uniform(*FLOAT_SPREAD))
        for k in rng.permutation(len(queue)):
            label, i = queue[k]
            item = self.instances[label][i]
            free = layout.kind == "floating" and bool(rng.random() < FREE_POSE_PROB)
            euler = self._sample_orientation(label, free)
            scale = rng.uniform(*SIZE_JITTER)
            center, z_min, radius_xy, radius = self._rest_pose(label, euler, scale)
            spot = self._free_spot(layout, spheres, radius_xy if layout.support_z is not None else radius,
                                   base_z, spread, z_min, center)
            if spot is None:
                continue
            origin, anchor, keep_out = spot
            set_pose(item["ops"], origin, euler, (scale, scale, scale))
            jitter_materials(item["materials"], rng)
            spheres.append((anchor, keep_out))
            centers.append(anchor)
            radii.append(radius)
            labels.append(label)
            placed[label] += 1
            used.add((label, i))
        for label, items in self.instances.items():
            for i, item in enumerate(items):
                set_visible(item["root"], (label, i) in used)
        return placed, labels, centers, radii, spheres

    def _free_spot(self, layout, spheres, keep_out, base_z, spread, z_min, center):
        rng = self.rng
        for _ in range(40):
            if layout.support_z is None:
                point = np.array([*rng.uniform(self.spawn_min, self.spawn_max), base_z + rng.uniform(-spread, spread)])
                origin = point - center
            else:
                footprint = rng.uniform(self.spawn_min, self.spawn_max)
                origin = np.array([*(footprint - center[:2]), layout.support_z - z_min])
                point = origin + center
            gap = rng.uniform(*ASSET_GAP)
            if all(np.linalg.norm(point - c) >= keep_out + r + gap for c, r in spheres):
                return origin, point, keep_out
        return None

    def _pick_focus(self, count):
        shot = str(self.rng.choice(list(SHOT_WEIGHTS), p=list(SHOT_WEIGHTS.values())))
        if shot == "wide" or count == 1:
            return "wide", np.arange(count)
        return shot, None

    def _place_camera(self, centers, radii, support_z):
        shot, focus = self._pick_focus(len(centers))
        if focus is None:
            anchor = centers[self.rng.integers(len(centers))]
            amount = 1 if shot == "close" else min(len(centers), int(self.rng.integers(2, 5)))
            focus = np.argsort(np.linalg.norm(centers - anchor, axis=1))[:amount]
        focus_centers, focus_radii = centers[focus], radii[focus]
        center = focus_centers.mean(axis=0)
        focal = self.rng.uniform(*FOCAL_LENGTH_MM)
        tan_h = self.sensor[0] / (2 * focal)
        tan_v = self.sensor[1] / (2 * focal)
        margin = self.rng.uniform(*DISTANCE_MARGIN[shot])
        raise_to = -np.inf if support_z is None else support_z - center[2]   # the camera never dips below the surface
        azimuth = self.rng.uniform(0, 2 * math.pi)
        elevation = math.radians(ELEVATION_MAX_DEG * self.rng.random() ** ELEVATION_SKEW)

        for _ in range(2):   # the second pass only matters when the camera had to be raised
            toward_camera = np.array([math.cos(elevation) * math.cos(azimuth),
                                      math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
            right = np.cross(-toward_camera, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, -toward_camera)
            rel = focus_centers - center
            depth = rel @ toward_camera
            needed = np.maximum(depth + (np.abs(rel @ right) + focus_radii) / tan_h,
                                depth + (np.abs(rel @ up) + focus_radii) / tan_v)
            distance = max(needed.max() * margin, DISTANCE_MIN)
            if distance * math.sin(elevation) >= raise_to:
                break
            elevation = math.asin(min(raise_to / distance, math.sin(math.radians(MAX_ELEVATION_DEG))))

        eye = center + distance * toward_camera
        roll = math.radians(float(np.clip(self.rng.normal(0, ROLL_SIGMA_DEG), -ROLL_MAX_DEG, ROLL_MAX_DEG)))
        self.camera_op.Set(look_at(eye, center, roll))
        self.camera.GetFocalLengthAttr().Set(float(focal))
        return eye, center, shot, focus, elevation, roll, distance, focal

    def _place_backdrop(self, layout, eye, target):
        if self.rng.random() >= BACKDROP_PROB[layout.kind]:
            set_visible(self.backdrop, False)
            return None
        away = target[:2] - eye[:2]
        away /= max(np.linalg.norm(away), 1e-9)
        extent = 0.0 if layout.half is None else abs(away[0]) * layout.half[0] + abs(away[1]) * layout.half[1]
        gap = self.rng.uniform(*BACKDROP_GAP)
        point = target[:2] + away * (extent + gap)
        yaw = math.degrees(math.atan2(-away[0], away[1]))   # normal points at the camera
        tilt = self.rng.uniform(-BACKDROP_TILT, BACKDROP_TILT)
        set_pose(self.backdrop_ops, [*point, BACKDROP_SIZE[1] / 2], (90 + tilt, 0, yaw),
                 (BACKDROP_SIZE[0], BACKDROP_SIZE[1], 1))
        set_visible(self.backdrop, True)
        layout.backdrop = (point, away)
        return gap

    def _place_distractors(self, layout, eye, target, spheres):
        rng = self.rng
        blocked = list(spheres)
        shown = set(rng.permutation(len(self.distractors))[:round(len(self.distractors) * rng.uniform(*DISTRACTOR_SHOWN))])
        props = 0
        for i, item in enumerate(self.distractors):
            near = i in shown and item["small"] and props < PROPS_MAX and rng.random() < PROP_PROB
            if i not in shown:
                visible = False
            elif near:
                visible = self._place_prop(item, layout, eye, target, blocked)
                props += visible
            else:
                visible = layout.floor and self._place_on_ground(item, layout, eye, target)
            set_visible(item["prim"], visible)
        return props

    def _place_prop(self, item, layout, eye, target, blocked):
        rng = self.rng
        scale = rng.uniform(*PROP_SCALE)
        radius = item["radius"] * scale
        for _ in range(30):
            if layout.support_z is None:
                direction = rng.normal(size=3)
                direction /= max(np.linalg.norm(direction), 1e-9)
                point = target + direction * rng.uniform(*FLOAT_PROP_RING)
                origin = point - np.array([0, 0, item["z_min"] * scale])
                euler = rng.uniform(0, 360, 3)
            else:
                xy = rng.uniform(self.center_xy - layout.half + radius, self.center_xy + layout.half - radius)
                point = np.array([*xy, layout.support_z + radius])
                origin = [*xy, layout.support_z - item["z_min"] * scale]
                euler = (0, 0, rng.uniform(0, 360))
            clear = distance_to_segment(point[:2], eye[:2], target[:2]) >= radius + 0.15
            if clear and all(np.linalg.norm(point - c) >= radius + r for c, r in blocked):
                blocked.append((point, radius))
                set_pose(item["ops"], origin, euler, (scale,) * 3)
                return True
        return False

    def _place_on_ground(self, item, layout, eye, target):
        rng = self.rng
        scale = rng.uniform(*DISTRACTOR_SCALE)
        radius = item["radius"] * scale
        for _ in range(30):
            angle = rng.uniform(0, 2 * math.pi)
            xy = self.center_xy + rng.uniform(*DISTRACTOR_RING) * np.array([math.cos(angle), math.sin(angle)])
            on_support = layout.half is not None and np.all(np.abs(xy - self.center_xy) <= layout.half + radius + 0.1)
            if on_support or distance_to_segment(xy, eye[:2], target[:2]) < DISTRACTOR_CLEARANCE:
                continue
            if layout.backdrop is not None and is_visible(self.backdrop):
                point, away = layout.backdrop
                if (xy - point) @ away > -(radius + 0.05):
                    continue   # behind the backdrop, the camera would never see it
            set_pose(item["ops"], [*xy, -item["z_min"] * scale], (0, 0, rng.uniform(0, 360)), (scale,) * 3)
            return True
        return False

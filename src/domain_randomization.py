import glob
import math
import os
import shutil

import cv2
import numpy as np

NEGATIVE_FRAME_PROB = 0.05      # frames without any asset

ROUGHNESS_JITTER = 0.30         # fraction of the original roughness
METALLIC_JITTER = 0.10          # only added to parts that are already metallic
BRIGHTNESS_JITTER = 0.15
TINT_JITTER = 0.04              # per color channel

SPAWN_AREA = ((-0.50, 1.00), (0.50, 1.80))   # xy corners of the area the assets are placed in, every height is randomized below

SIZE_JITTER = (0.8, 1.2)        # per instance, relative to the real size of the asset
ASSET_GAP = (0.0, 0.03)         # metres between neighbouring assets, touching is allowed
LYING_POSES = [(90, 0), (-90, 0), (0, 90), (0, -90)]   # rotation about x and y that lay an asset on one of its sides
TILT_JITTER = 5.0               # degrees, nothing in a real photo is perfectly level
FREE_POSE_PROB = 0.30           # floating assets with a fully arbitrary orientation, like an object held in a hand

LAYOUTS = {"supported": 0.45, "ground": 0.15, "floating": 0.40}   # a surface, the ground, or nothing: the support is never a constant
SUPPORT_HEIGHT = (0.30, 1.05)   # from a low table to a kitchen counter
SUPPORT_MARGIN = (0.01, 0.70)   # metres of surface around the assets, small values leave them at the edge
FLOAT_HEIGHT = (0.40, 1.80)     # height of the floating cluster
FLOAT_SPREAD = (0.04, 0.30)     # extra metres of 3d scatter inside the floating cluster
FLOOR_PROB = {"supported": 0.85, "ground": 1.0, "floating": 0.30}
BACKDROP_PROB = {"supported": 0.60, "ground": 0.50, "floating": 0.25}
BACKDROP_GAP = (0.03, 6.0)      # metres between the assets and the backdrop
BACKDROP_TILT = 8.0             # degrees, a wall is never exactly vertical in frame
FLOOR_SIZE = 40.0
BACKDROP_SIZE = (10.0, 4.0)     # width and height
TEXTURE_SCALE = (0.5, 3.0)

FOCAL_LENGTH_MM = (16.0, 55.0)
ELEVATION_MAX_DEG = 75.0        # camera is never below the objects and never straight overhead
ELEVATION_SKEW = 2.0            # elevation = max * u ** skew, so most frames are frontal and a few look from above
MAX_ELEVATION_DEG = 80.0        # hard cap after raising the camera above the support
ROLL_SIGMA_DEG = 6.0            # handheld tilt, a fixed horizon is the easiest cue to overfit
ROLL_MAX_DEG = 18.0
SHOT_WEIGHTS = {"close": 0.35, "medium": 0.40, "wide": 0.25}   # one object, two to four nearby, every visible object
DISTANCE_MARGIN = {"close": (0.85, 1.3), "medium": (0.9, 1.4), "wide": (1.05, 1.5)}   # below 1 truncates the objects at the borders
DISTANCE_MIN = 0.25             # metres, the camera near plane is at 0.1

DISTRACTOR_RING = (0.8, 4.5)    # metres around the assets, on the ground
DISTRACTOR_CLEARANCE = 1.0      # metres kept free around the camera to assets line
DISTRACTOR_SCALE = (0.7, 1.5)
DISTRACTOR_SHOWN = (0.4, 1.0)   # fraction of the distractors visible in each frame
PROP_SIZE = 0.45                # distractors up to this size can share the support or float with the assets
PROP_PROB = 0.7
PROPS_MAX = 6
PROP_SCALE = (0.8, 1.3)
FLOAT_PROP_RING = (0.25, 1.2)   # metres around a floating cluster

LOCAL_LIGHTS = 3
LIGHT_RADIUS = 0.15
SUN_ANGLE = (0.3, 2.5)          # angular diameter in degrees, small values give hard shadows
LIGHT_STYLES = {                # one lighting situation per frame instead of the same soft mix every time
    "overcast": {"weight": 0.18, "exposure": (-0.5, 1.2), "locals": (0, 1), "temperature": (5000, 7500),
                 "intensity": (5e3, 3e4), "elevation": (40, 85), "sun_prob": 0.0, "sun_intensity": (0, 0)},
    "window":   {"weight": 0.22, "exposure": (-2.5, 0.0), "locals": (0, 1), "temperature": (5000, 7000),
                 "intensity": (1e4, 6e4), "elevation": (10, 45), "sun_prob": 0.85, "sun_intensity": (800, 4000)},
    "indoor":   {"weight": 0.22, "exposure": (-4.5, -2.0), "locals": (2, 3), "temperature": (2300, 3600),
                 "intensity": (3e4, 1.6e5), "elevation": (45, 85), "sun_prob": 0.0, "sun_intensity": (0, 0)},
    "mixed":    {"weight": 0.18, "exposure": (-3.0, -0.5), "locals": (1, 2), "temperature": (2300, 6800),
                 "intensity": (2e4, 1.0e5), "elevation": (25, 80), "sun_prob": 0.35, "sun_intensity": (500, 2500)},
    "backlit":  {"weight": 0.11, "exposure": (-2.5, 0.5), "locals": (1, 1), "temperature": (3000, 7000),
                 "intensity": (8e4, 2.5e5), "elevation": (5, 40), "sun_prob": 0.5, "sun_intensity": (2000, 6000)},
    "dim":      {"weight": 0.09, "exposure": (-6.0, -4.0), "locals": (1, 2), "temperature": (2200, 3300),
                 "intensity": (4e3, 2.5e4), "elevation": (30, 80), "sun_prob": 0.0, "sun_intensity": (0, 0)},
}
LIGHT_DISTANCE = (0.8, 3.0)
DOME_INTENSITY = (1.5e3, 8e3)
BACKLIT_SPREAD_DEG = 45.0       # how far the backlit key light can swing off the camera axis

SENSOR_ASPECTS = [(9, 16), (3, 4), (1, 1), (4, 3), (16, 9), (16, 9)]   # the same objects reach the model in portrait and landscape
SENSOR_ZOOM = (0.72, 1.0)       # fraction of the render kept by the crop
SENSOR_SHIFT = 0.12             # fraction of the slack the crop window can be moved off centre

CAMERA_EFFECTS = {
    "barrel": {"prob": 0.55, "k": (0.01, 0.13)},                # radial stretch of the centre, phone lenses are never rectilinear
    "exposure": {"prob": 1.0, "target": (80, 150), "gain": (0.3, 9.0), "miss_prob": 0.25,
                 "miss": (0.4, 2.3), "lift": (-5.0, 14.0)},     # the camera meters the frame and sometimes gets it wrong
    "aberration": {"prob": 0.35, "shift": (0.0008, 0.0035)},    # fraction of the radius, red against blue
    "defocus_blur": {"prob": 0.30, "sigma": (0.5, 2.4)},        # px
    "motion_blur": {"prob": 0.30, "length": (3, 17)},           # px
    "bloom": {"prob": 0.30, "threshold": (170, 230), "strength": (0.15, 0.6), "sigma": (4, 16)},
    "vignette": {"prob": 0.55, "strength": (0.10, 0.55)},
    "white_balance": {"prob": 0.70, "gain": (0.85, 1.15)},      # per channel
    "noise": {"prob": 0.90, "sigma": (0.8, 3.0), "chroma": (0.4, 1.6), "ceiling": 24.0},   # amplified with the metering gain, like auto iso
    "jpeg": {"prob": 0.65, "quality": (28, 92)},
}
NOISE_GAIN_EXPONENT = 0.8       # how much of the exposure gain reaches the grain


class SceneRandomizer:
    def __init__(self, stage, assets, resolution, hdris=(), seed=None):
        from pxr import Usd, UsdGeom, UsdLux

        self.rng = np.random.default_rng(seed)
        self.assets = assets
        self.hdris = list(hdris)
        self.spawn_min, self.spawn_max = (np.array(c, dtype=float) for c in SPAWN_AREA)
        self.center_xy = (self.spawn_min + self.spawn_max) / 2

        self.instances = {}
        for label in assets:
            roots = _find_instance_roots(stage, label)
            self.instances[label] = [
                {"root": root, "ops": _movable_ops(root.GetParent()), "materials": _collect_materials(root)}
                for root in roots
            ]
            print(f"[randomizer] {label}: {len(roots)} instances")

        asset_parents = {i["root"].GetParent().GetPath() for items in self.instances.values() for i in items}
        self.distractors = []
        for prim in stage.GetPrimAtPath("/Replicator").GetChildren():
            if not prim.GetName().startswith("Ref_Xform") or prim.GetPath() in asset_parents:
                continue
            ops = _movable_ops(prim)
            low, high = _bounds(prim)
            size = high - low
            self.distractors.append({
                "prim": prim, "ops": ops, "z_min": low[2], "center": ((low + high) / 2)[:2],
                "radius": 0.5 * math.hypot(size[0], size[1]), "small": size.max() <= PROP_SIZE,
            })
        print(f"[randomizer] {len(self.distractors)} distractors, {sum(d['small'] for d in self.distractors)} small")

        self.floor = stage.GetPrimAtPath("/Replicator/Floor_Xform")
        self.support = stage.GetPrimAtPath("/Replicator/Support_Xform")
        self.backdrop = stage.GetPrimAtPath("/Replicator/Backdrop_Xform")
        self.support_ops = _movable_ops(self.support)
        self.backdrop_ops = _movable_ops(self.backdrop)

        cam_prim = next(p for p in Usd.PrimRange(stage.GetPrimAtPath("/Replicator")) if p.IsA(UsdGeom.Camera))
        self.camera = UsdGeom.Camera(cam_prim)
        UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
        parent_xf = UsdGeom.Xformable(cam_prim.GetParent())
        parent_xf.ClearXformOpOrder()
        self.camera_op = parent_xf.AddTransformOp()
        width, height = resolution
        horizontal = self.camera.GetHorizontalApertureAttr().Get()
        self.sensor = (horizontal, horizontal * height / width)   # effective aperture for the render aspect ratio

        dome_prim = next((p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)), None)
        self.dome = None
        if dome_prim is not None:
            dome = UsdLux.DomeLight(dome_prim)
            xf = UsdGeom.Xformable(dome_prim)
            xf.ClearXformOpOrder()
            self.dome = {"rotate": xf.AddRotateXYZOp(), "intensity": dome.CreateIntensityAttr(),
                         "exposure": dome.CreateExposureAttr(), "texture": dome.CreateTextureFileAttr()}

        self.lights = []
        for i in range(LOCAL_LIGHTS):
            light = UsdLux.SphereLight.Define(stage, f"/DR_Light_{i}")
            light.CreateRadiusAttr(LIGHT_RADIUS)
            light.CreateEnableColorTemperatureAttr(True)
            self.lights.append({
                "translate": UsdGeom.Xformable(light).AddTranslateOp(), "intensity": light.CreateIntensityAttr(0.0),
                "temperature": light.CreateColorTemperatureAttr(6500.0),
            })

        sun = UsdLux.DistantLight.Define(stage, "/DR_Sun")
        sun.CreateEnableColorTemperatureAttr(True)
        sun_xf = UsdGeom.Xformable(sun)
        sun_xf.ClearXformOpOrder()
        self.sun = {"transform": sun_xf.AddTransformOp(), "intensity": sun.CreateIntensityAttr(0.0),
                    "angle": sun.CreateAngleAttr(1.0), "temperature": sun.CreateColorTemperatureAttr(6500.0)}

    def randomize_frame(self):
        rng = self.rng
        negative = bool(rng.random() < NEGATIVE_FRAME_PROB)
        layout = str(rng.choice(list(LAYOUTS), p=list(LAYOUTS.values())))
        floor_visible = bool(rng.random() < FLOOR_PROB[layout])
        _set_visible(self.floor, floor_visible)

        wanted = self._sample_counts(negative)
        if layout == "floating":
            support_z, half = None, None
            placed, labels, centers, radii, spheres = self._place_floating(wanted)
        else:
            support_z = float(rng.uniform(*SUPPORT_HEIGHT)) if layout == "supported" else 0.0
            half = (self.spawn_max - self.spawn_min) / 2 + rng.uniform(*SUPPORT_MARGIN, size=2)
            placed, labels, centers, radii, spheres = self._place_on_surface(wanted, support_z)
            _set_pose(self.support_ops, [*self.center_xy, support_z / 2], (0, 0, 0), (2 * half[0], 2 * half[1], max(support_z, 0.01)))
        _set_visible(self.support, layout == "supported")

        if not centers:   # a negative frame still needs something to aim the camera at
            fallback_z = (support_z if support_z is not None else float(rng.uniform(*FLOAT_HEIGHT))) + 0.15
            centers, radii = [np.array([*self.center_xy, fallback_z])], [0.3]
        floor_z = support_z if support_z is not None else None
        eye, target, shot, focus, elevation, roll, distance, focal = self._place_camera(np.array(centers), np.array(radii), floor_z)

        gap = self._place_backdrop(layout, eye, target, half)
        props = self._place_distractors(layout, eye, target, support_z, half, floor_visible, spheres)
        style = self._place_lights(target, eye)
        return {"negative": negative, "visible": placed, "focus": sorted({labels[i] for i in focus}) if labels else [], "shot": shot,
                "elevation_deg": round(math.degrees(elevation), 1), "roll_deg": round(math.degrees(roll), 1),
                "distance": round(float(distance), 2), "focal_mm": round(float(focal), 1), "layout": layout,
                "support_z": None if support_z is None else round(support_z, 3), "floor": floor_visible,
                "backdrop_gap": None if gap is None else round(float(gap), 2), "light_style": style,
                "dome": self.dome is not None, "props": props}

    def _sample_counts(self, negative):
        if negative:
            return {label: 0 for label in self.instances}
        while True:
            counts = {label: int(self.rng.integers(0, len(items) + 1)) for label, items in self.instances.items()}
            if sum(counts.values()) > 0:
                return counts

    def _rest_pose(self, label, rx, ry, yaw, scale):
        low, high = self.assets[label]["bbox_min"], self.assets[label]["bbox_max"]
        corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
        rotation = _rotation(rx, ry, yaw)
        world = (corners * scale) @ rotation.T
        center = ((low + high) / 2 * scale) @ rotation.T
        return center, world[:, 2].min(), np.linalg.norm(world[:, :2] - center[:2], axis=1).max(), np.linalg.norm(world - center, axis=1).max()

    def _sample_orientation(self, label, free):
        rng = self.rng
        if free:
            return rng.uniform(0, 360, 3)   # no gravity to respect, the object can be at any angle
        rx, ry = LYING_POSES[rng.integers(len(LYING_POSES))] if rng.random() < self.assets[label]["lying"] else (0, 0)
        tilt = rng.uniform(-TILT_JITTER, TILT_JITTER, 2)
        return np.array([rx + tilt[0], ry + tilt[1], rng.uniform(0, 360)])

    def _place_on_surface(self, wanted, support_z):
        queue = [(label, i) for label, count in wanted.items() for i in range(count)]
        placed = {label: 0 for label in self.instances}
        labels, centers, radii, circles = [], [], [], []
        used = set()
        for k in self.rng.permutation(len(queue)):
            label, i = queue[k]
            item = self.instances[label][i]
            rx, ry, yaw = self._sample_orientation(label, free=False)
            scale = self.rng.uniform(*SIZE_JITTER)
            center, z_min, radius_xy, radius = self._rest_pose(label, rx, ry, yaw, scale)
            for _ in range(40):
                footprint = self.rng.uniform(self.spawn_min, self.spawn_max)
                gap = self.rng.uniform(*ASSET_GAP)
                if all(np.linalg.norm(footprint - c) >= radius_xy + r + gap for c, r in circles):
                    break
            else:
                continue   # no free spot, this instance stays hidden
            origin = np.array([*(footprint - center[:2]), support_z - z_min])
            _set_pose(item["ops"], origin, (rx, ry, yaw), (scale, scale, scale))
            _jitter_materials(item["materials"], self.rng)
            circles.append((footprint, radius_xy))
            centers.append(origin + center)
            radii.append(radius)
            labels.append(label)
            placed[label] += 1
            used.add((label, i))
        self._apply_visibility(used)
        return placed, labels, centers, radii, [(np.array([*c, support_z]), r) for c, r in circles]

    def _place_floating(self, wanted):
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
            free = bool(rng.random() < FREE_POSE_PROB)
            rx, ry, yaw = self._sample_orientation(label, free)
            scale = rng.uniform(*SIZE_JITTER)
            center, _, _, radius = self._rest_pose(label, rx, ry, yaw, scale)
            for _ in range(40):
                point = np.array([*rng.uniform(self.spawn_min, self.spawn_max),
                                  base_z + rng.uniform(-spread, spread)])
                gap = rng.uniform(*ASSET_GAP)
                if all(np.linalg.norm(point - c) >= radius + r + gap for c, r in spheres):
                    break
            else:
                continue
            origin = point - center
            _set_pose(item["ops"], origin, (rx, ry, yaw), (scale, scale, scale))
            _jitter_materials(item["materials"], self.rng)
            spheres.append((point, radius))
            centers.append(point)
            radii.append(radius)
            labels.append(label)
            placed[label] += 1
            used.add((label, i))
        self._apply_visibility(used)
        return placed, labels, centers, radii, spheres

    def _apply_visibility(self, used):
        for label, items in self.instances.items():
            for i, item in enumerate(items):
                _set_visible(item["root"], (label, i) in used)

    def _pick_focus(self, count):
        shot = str(self.rng.choice(list(SHOT_WEIGHTS), p=list(SHOT_WEIGHTS.values())))
        if shot == "wide" or count == 1:
            return "wide", np.arange(count)
        return shot, None

    def _place_camera(self, centers, radii, floor_z):
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
        raise_to = -np.inf if floor_z is None else floor_z - center[2]   # the camera never dips below the surface the assets rest on
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
        self.camera_op.Set(_look_at(eye, center, roll))
        self.camera.GetFocalLengthAttr().Set(float(focal))
        return eye, center, shot, focus, elevation, roll, distance, focal

    def _place_backdrop(self, layout, eye, target, half):
        if self.rng.random() >= BACKDROP_PROB[layout]:
            _set_visible(self.backdrop, False)
            return None
        away = target[:2] - eye[:2]
        away /= max(np.linalg.norm(away), 1e-9)
        extent = 0.0 if half is None else abs(away[0]) * half[0] + abs(away[1]) * half[1]
        gap = self.rng.uniform(*BACKDROP_GAP)
        point = target[:2] + away * (extent + gap)
        yaw = math.degrees(math.atan2(-away[0], away[1]))   # backdrop normal points at the camera
        tilt = self.rng.uniform(-BACKDROP_TILT, BACKDROP_TILT)
        _set_pose(self.backdrop_ops, [*point, BACKDROP_SIZE[1] / 2], (90 + tilt, 0, yaw), (BACKDROP_SIZE[0], BACKDROP_SIZE[1], 1))
        _set_visible(self.backdrop, True)
        self.backdrop_away = away
        self.backdrop_point = point
        return gap

    def _place_distractors(self, layout, eye, target, support_z, half, floor_visible, spheres):
        rng = self.rng
        blocked = list(spheres)
        shown = set(rng.permutation(len(self.distractors))[:round(len(self.distractors) * rng.uniform(*DISTRACTOR_SHOWN))])
        props = 0
        for i, item in enumerate(self.distractors):
            visible = False
            if i in shown:
                near = item["small"] and props < PROPS_MAX and rng.random() < PROP_PROB
                if near and layout == "floating":
                    visible = self._place_prop_floating(item, target, blocked, eye)
                elif near and support_z is not None:
                    visible = self._place_prop_on_surface(item, support_z, half, blocked, eye, target)
                elif floor_visible:
                    visible = self._place_on_ground(item, eye, target, half)
                props += visible and near
            _set_visible(item["prim"], visible)
        return int(props)

    def _place_prop_on_surface(self, item, support_z, half, blocked, eye, target):
        rng = self.rng
        scale = rng.uniform(*PROP_SCALE)
        radius = item["radius"] * scale
        for _ in range(30):
            xy = rng.uniform(self.center_xy - half + radius, self.center_xy + half - radius)
            point = np.array([*xy, support_z + radius])
            if all(np.linalg.norm(point - c) >= radius + r for c, r in blocked) and _distance_to_segment(xy, eye[:2], target[:2]) >= radius + 0.15:
                blocked.append((point, radius))
                _set_pose(item["ops"], [*xy, support_z - item["z_min"] * scale], (0, 0, rng.uniform(0, 360)), (scale,) * 3)
                return True
        return False

    def _place_prop_floating(self, item, target, blocked, eye):
        rng = self.rng
        scale = rng.uniform(*PROP_SCALE)
        radius = item["radius"] * scale
        for _ in range(30):
            direction = rng.normal(size=3)
            direction /= max(np.linalg.norm(direction), 1e-9)
            point = target + direction * rng.uniform(*FLOAT_PROP_RING)
            if all(np.linalg.norm(point - c) >= radius + r for c, r in blocked) and _distance_to_segment(point[:2], eye[:2], target[:2]) >= radius + 0.15:
                blocked.append((point, radius))
                _set_pose(item["ops"], point - np.array([0, 0, item["z_min"] * scale]), rng.uniform(0, 360, 3), (scale,) * 3)
                return True
        return False

    def _place_on_ground(self, item, eye, target, half):
        rng = self.rng
        scale = rng.uniform(*DISTRACTOR_SCALE)
        radius = item["radius"] * scale
        for _ in range(30):
            angle = rng.uniform(0, 2 * math.pi)
            xy = self.center_xy + rng.uniform(*DISTRACTOR_RING) * np.array([math.cos(angle), math.sin(angle)])
            on_support = half is not None and np.all(np.abs(xy - self.center_xy) <= half + radius + 0.1)
            hidden = getattr(self, "backdrop_away", None) is not None and self.backdrop.IsValid() and _is_visible(self.backdrop) \
                and (xy - self.backdrop_point) @ self.backdrop_away > -(radius + 0.05)
            if not on_support and not hidden and _distance_to_segment(xy, eye[:2], target[:2]) >= DISTRACTOR_CLEARANCE:
                _set_pose(item["ops"], [*xy, -item["z_min"] * scale], (0, 0, rng.uniform(0, 360)), (scale,) * 3)
                return True
        return False

    def _place_lights(self, target, eye):
        from pxr import Gf, Sdf

        rng = self.rng
        name = str(rng.choice(list(LIGHT_STYLES), p=[s["weight"] for s in LIGHT_STYLES.values()]))
        style = LIGHT_STYLES[name]

        if self.dome is not None:
            if self.hdris:
                self.dome["texture"].Set(Sdf.AssetPath(self.hdris[rng.integers(len(self.hdris))]))
            self.dome["rotate"].Set(Gf.Vec3f(0.0, 0.0, float(rng.uniform(0, 360))))
            self.dome["intensity"].Set(float(rng.uniform(*DOME_INTENSITY)))
            self.dome["exposure"].Set(float(rng.uniform(*style["exposure"])))

        toward_camera = eye - target
        toward_camera /= max(np.linalg.norm(toward_camera), 1e-9)
        count = int(rng.integers(style["locals"][0], style["locals"][1] + 1))
        for i, light in enumerate(self.lights):
            if i >= count:
                light["intensity"].Set(0.0)
                continue
            distance = rng.uniform(*LIGHT_DISTANCE)
            if name == "backlit":
                direction = _perturb(-toward_camera, math.radians(BACKLIT_SPREAD_DEG), rng)   # key light behind the objects
                direction[2] = abs(direction[2])
            else:
                elevation = math.radians(rng.uniform(*style["elevation"]))
                azimuth = rng.uniform(0, 2 * math.pi)
                direction = np.array([math.cos(elevation) * math.cos(azimuth),
                                      math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
            light["translate"].Set(Gf.Vec3d(*map(float, target + distance * direction)))
            light["intensity"].Set(float(math.exp(rng.uniform(*map(math.log, style["intensity"]))) * (distance / 1.5) ** 2))
            light["temperature"].Set(float(rng.uniform(*style["temperature"])))

        if rng.random() < style["sun_prob"]:
            elevation = math.radians(rng.uniform(*style["elevation"]))
            azimuth = rng.uniform(0, 2 * math.pi)
            direction = np.array([math.cos(elevation) * math.cos(azimuth),
                                  math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
            self.sun["transform"].Set(_look_at(target + 10.0 * direction, target, 0.0))
            self.sun["intensity"].Set(float(rng.uniform(*style["sun_intensity"])))
            self.sun["angle"].Set(float(rng.uniform(*SUN_ANGLE)))
            self.sun["temperature"].Set(float(rng.uniform(*style["temperature"])))
        else:
            self.sun["intensity"].Set(0.0)
        return name


def _look_at(eye, target, roll):
    from pxr import Gf

    forward = np.array(target, dtype=float) - np.array(eye, dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, [0, 0, 1])
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward) * math.cos(roll) + right * math.sin(roll)
    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(*map(float, up))).GetInverse()


def _perturb(direction, max_angle, rng):
    axis = np.cross(direction, rng.normal(size=3))
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return direction
    axis /= norm
    angle = max_angle * rng.random()
    return direction * math.cos(angle) + np.cross(axis, direction) * math.sin(angle)


def _rotation(rx, ry, rz):
    a, b, c = np.radians([rx, ry, rz])
    rot_x = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    rot_y = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rot_z = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def _bounds(prim):
    from pxr import Usd, UsdGeom

    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(box.GetMin()), np.array(box.GetMax())


def _movable_ops(prim):
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    translate, rotate, scale = xf.AddTranslateOp(), xf.AddRotateXYZOp(), xf.AddScaleOp()
    translate.Set(Gf.Vec3d(0, 0, 0))
    rotate.Set(Gf.Vec3f(0, 0, 0))
    scale.Set(Gf.Vec3f(1, 1, 1))
    return translate, rotate, scale


def _set_pose(ops, position, euler_deg, scale):
    from pxr import Gf

    translate, rotate, scale_op = ops
    translate.Set(Gf.Vec3d(*map(float, position)))
    rotate.Set(Gf.Vec3f(*map(float, euler_deg)))
    scale_op.Set(Gf.Vec3f(*map(float, scale)))


def _distance_to_segment(point, a, b):
    ab = b - a
    t = np.clip(np.dot(point - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(point - (a + t * ab))


def _find_instance_roots(stage, label):
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


def _set_visible(prim, visible):
    from pxr import UsdGeom

    img = UsdGeom.Imageable(prim)
    img.MakeVisible() if visible else img.MakeInvisible()


def _is_visible(prim):
    from pxr import UsdGeom

    return UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible


def _collect_materials(root):
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
            tex = UsdShade.Shader(source[0].GetPrim())   # textured parts: brightness and tint go through the texture scale
            entry["tex_scale"] = tex.GetInput("scale") or tex.CreateInput("scale", Sdf.ValueTypeNames.Float4)
        else:
            entry["color"] = color
            entry["color_base"] = color.Get()
        out.append(entry)
    return out


def _jitter_materials(materials, rng):
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
        rep.create.plane(scale=(BACKDROP_SIZE[0], BACKDROP_SIZE[1], 1), position=(0, 3, BACKDROP_SIZE[1] / 2), rotation=(90, 0, 0), name="Backdrop"),
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


def apply_camera_effects(data_dir, seed=None):
    rng = np.random.default_rng(seed)
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "rgb_*.png")))
    for path in paths:
        num = os.path.basename(path)[4:-4]
        mask_path = os.path.join(data_dir, f"instance_segmentation_{num}.png")
        img = _cached_raw(path, raw_dir)   # the untouched render stays in raw/ so this pass can be re-run or compared
        mask = _cached_raw(mask_path, raw_dir) if os.path.exists(mask_path) else None
        img, mask = _lens_geometry(img, mask, rng)
        cv2.imwrite(path, _sensor(img, rng))
        if mask is not None:
            cv2.imwrite(mask_path, mask)
    print(f"[camera effects] {len(paths)} images processed")


def _cached_raw(path, raw_dir):
    cached = os.path.join(raw_dir, os.path.basename(path))
    if not os.path.exists(cached):
        shutil.copy2(path, cached)
    return cv2.imread(cached, cv2.IMREAD_UNCHANGED)


def _chance(rng, name):
    return rng.random() < CAMERA_EFFECTS[name]["prob"]


def _draw(rng, name, key):
    lo, hi = CAMERA_EFFECTS[name][key]
    return rng.uniform(lo, hi)


def _lens_geometry(img, mask, rng):
    h, w = img.shape[:2]
    maps = None
    if _chance(rng, "barrel"):
        k = _draw(rng, "barrel", "k")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        half = math.hypot(cx, cy)
        dx, dy = (xx - cx) / half, (yy - cy) / half
        gain = 1 - k * (dx * dx + dy * dy)   # samples inside the frame, so the border never runs out of pixels
        maps = (cx + dx * half * gain, cy + dy * half * gain)

    aspect = SENSOR_ASPECTS[rng.integers(len(SENSOR_ASPECTS))]
    zoom = rng.uniform(*SENSOR_ZOOM)
    crop_w = min(w, h * aspect[0] / aspect[1]) * zoom
    crop_h = min(h, w * aspect[1] / aspect[0]) * zoom
    x0 = (w - crop_w) / 2 * (1 + SENSOR_SHIFT * rng.uniform(-1, 1))
    y0 = (h - crop_h) / 2 * (1 + SENSOR_SHIFT * rng.uniform(-1, 1))
    box = (int(round(x0)), int(round(y0)), int(round(crop_w)), int(round(crop_h)))

    return _warp(img, maps, box, cv2.INTER_LINEAR), None if mask is None else _warp(mask, maps, box, cv2.INTER_NEAREST)


def _warp(img, maps, box, interpolation):
    if maps is not None:
        img = cv2.remap(img, maps[0], maps[1], interpolation, borderMode=cv2.BORDER_REPLICATE)
    x0, y0, cw, ch = box
    return img[y0:y0 + ch, x0:x0 + cw]


def _sensor(img, rng):
    alpha = img[..., 3:] if img.ndim == 3 and img.shape[2] == 4 else None
    bgr = img[..., :3].astype(np.float32)
    h, w = bgr.shape[:2]

    exposure = CAMERA_EFFECTS["exposure"]
    gain = float(np.clip(_draw(rng, "exposure", "target") / max(bgr.mean(), 1.0), *exposure["gain"]))
    if rng.random() < exposure["miss_prob"]:
        gain *= _draw(rng, "exposure", "miss")   # backlit and spot lit frames are where metering fails
    bgr = bgr * gain + _draw(rng, "exposure", "lift")   # clipping happens at the end, as in a real sensor

    if _chance(rng, "aberration"):
        shift = _draw(rng, "aberration", "shift")
        for channel, scale in ((0, 1 + shift), (2, 1 - shift)):
            matrix = np.array([[scale, 0, (1 - scale) * (w - 1) / 2], [0, scale, (1 - scale) * (h - 1) / 2]], np.float32)
            bgr[..., channel] = cv2.warpAffine(bgr[..., channel], matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if _chance(rng, "defocus_blur"):
        bgr = cv2.GaussianBlur(bgr, (0, 0), _draw(rng, "defocus_blur", "sigma"))

    if _chance(rng, "motion_blur"):
        length = int(_draw(rng, "motion_blur", "length")) | 1   # odd kernel size
        kernel = np.zeros((length, length), np.float32)
        angle = rng.uniform(0, np.pi)
        c = length // 2
        dx, dy = np.cos(angle) * c, np.sin(angle) * c
        cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
        bgr = cv2.filter2D(bgr, -1, kernel / kernel.sum())

    if _chance(rng, "bloom"):
        highlights = np.clip(bgr - _draw(rng, "bloom", "threshold"), 0, None)
        bgr += _draw(rng, "bloom", "strength") * cv2.GaussianBlur(highlights, (0, 0), _draw(rng, "bloom", "sigma"))

    if _chance(rng, "vignette"):
        strength = _draw(rng, "vignette", "strength")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        bgr *= (1 - strength * r2)[..., None]

    if _chance(rng, "white_balance"):
        lo, hi = CAMERA_EFFECTS["white_balance"]["gain"]
        bgr *= rng.uniform(lo, hi, size=3).astype(np.float32)

    if _chance(rng, "noise"):
        sigma = min(_draw(rng, "noise", "sigma") * gain ** NOISE_GAIN_EXPONENT, CAMERA_EFFECTS["noise"]["ceiling"])
        bgr += rng.normal(0, sigma, (h, w, 1)).astype(np.float32)   # shared luma grain
        bgr += rng.normal(0, sigma * _draw(rng, "noise", "chroma"), bgr.shape).astype(np.float32)

    out = np.clip(bgr, 0, 255).astype(np.uint8)

    if _chance(rng, "jpeg"):
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(_draw(rng, "jpeg", "quality"))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return np.concatenate([out, alpha], axis=2) if alpha is not None else out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with rgb_*.png to degrade in place")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    apply_camera_effects(args.data_dir, args.seed)

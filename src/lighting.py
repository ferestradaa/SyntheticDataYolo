import math

import numpy as np

from usd_scene import look_at, perturb

LIGHT_STYLES = {
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

LOCAL_LIGHTS = 3
LIGHT_RADIUS = 0.15
LIGHT_DISTANCE = (0.8, 3.0)
DOME_INTENSITY = (1.5e3, 8e3)
SUN_ANGLE = (0.3, 2.5)          # angular diameter in degrees, small values give hard shadows
BACKLIT_SPREAD_DEG = 45.0


class LightRig:
    def __init__(self, stage, hdris=()):
        from pxr import UsdGeom, UsdLux

        self.hdris = list(hdris)

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

    def apply(self, rng, target, eye):
        name = str(rng.choice(list(LIGHT_STYLES), p=[s["weight"] for s in LIGHT_STYLES.values()]))
        style = LIGHT_STYLES[name]
        self._dome(rng, style)
        self._locals(rng, style, name, target, eye)
        self._sun(rng, style, target)
        return name

    def _dome(self, rng, style):
        from pxr import Gf, Sdf

        if self.dome is None:
            return
        if self.hdris:
            self.dome["texture"].Set(Sdf.AssetPath(self.hdris[rng.integers(len(self.hdris))]))
        self.dome["rotate"].Set(Gf.Vec3f(0.0, 0.0, float(rng.uniform(0, 360))))
        self.dome["intensity"].Set(float(rng.uniform(*DOME_INTENSITY)))
        self.dome["exposure"].Set(float(rng.uniform(*style["exposure"])))

    def _locals(self, rng, style, name, target, eye):
        from pxr import Gf

        toward_camera = eye - target
        toward_camera /= max(np.linalg.norm(toward_camera), 1e-9)
        count = int(rng.integers(style["locals"][0], style["locals"][1] + 1))
        for i, light in enumerate(self.lights):
            if i >= count:
                light["intensity"].Set(0.0)
                continue
            distance = rng.uniform(*LIGHT_DISTANCE)
            if name == "backlit":
                direction = perturb(-toward_camera, math.radians(BACKLIT_SPREAD_DEG), rng)
                direction[2] = abs(direction[2])
            else:
                direction = _on_sphere(rng, style["elevation"])
            light["translate"].Set(Gf.Vec3d(*map(float, target + distance * direction)))
            light["intensity"].Set(float(math.exp(rng.uniform(*map(math.log, style["intensity"]))) * (distance / 1.5) ** 2))
            light["temperature"].Set(float(rng.uniform(*style["temperature"])))

    def _sun(self, rng, style, target):
        if rng.random() >= style["sun_prob"]:
            self.sun["intensity"].Set(0.0)
            return
        direction = _on_sphere(rng, style["elevation"])
        self.sun["transform"].Set(look_at(target + 10.0 * direction, target, 0.0))
        self.sun["intensity"].Set(float(rng.uniform(*style["sun_intensity"])))
        self.sun["angle"].Set(float(rng.uniform(*SUN_ANGLE)))
        self.sun["temperature"].Set(float(rng.uniform(*style["temperature"])))


def _on_sphere(rng, elevation_range):
    elevation = math.radians(rng.uniform(*elevation_range))
    azimuth = rng.uniform(0, 2 * math.pi)
    return np.array([math.cos(elevation) * math.cos(azimuth),
                     math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])

import glob
import math
import os
import shutil

import cv2
import numpy as np

SENSOR_ASPECTS = [(9, 16), (3, 4), (1, 1), (4, 3), (16, 9), (16, 9)]
SENSOR_ZOOM = (0.72, 1.0)
SENSOR_SHIFT = 0.12
NOISE_GAIN_EXPONENT = 0.8       # how much of the exposure gain reaches the grain

EFFECTS = {
    "barrel": {"prob": 0.55, "k": (0.01, 0.13)},
    "exposure": {"prob": 1.0, "target": (80, 150), "gain": (0.3, 9.0), "miss_prob": 0.25,
                 "miss": (0.4, 2.3), "lift": (-5.0, 14.0)},
    "aberration": {"prob": 0.35, "shift": (0.0008, 0.0035)},
    "defocus_blur": {"prob": 0.30, "sigma": (0.5, 2.4)},
    "motion_blur": {"prob": 0.30, "length": (3, 17)},
    "bloom": {"prob": 0.30, "threshold": (170, 230), "strength": (0.15, 0.6), "sigma": (4, 16)},
    "vignette": {"prob": 0.55, "strength": (0.10, 0.55)},
    "white_balance": {"prob": 0.70, "gain": (0.85, 1.15)},
    "noise": {"prob": 0.90, "sigma": (0.8, 3.0), "chroma": (0.4, 1.6), "ceiling": 24.0},
    "jpeg": {"prob": 0.65, "quality": (28, 92)},
}


def apply_camera_effects(data_dir, seed=None):
    rng = np.random.default_rng(seed)
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "rgb_*.png")))
    for path in paths:
        num = os.path.basename(path)[4:-4]
        mask_path = os.path.join(data_dir, f"instance_segmentation_{num}.png")
        img = _cached_raw(path, raw_dir)
        mask = _cached_raw(mask_path, raw_dir) if os.path.exists(mask_path) else None
        img, mask = _lens_geometry(img, mask, rng)
        cv2.imwrite(path, _sensor(img, rng))
        if mask is not None:
            cv2.imwrite(mask_path, mask)
    print(f"[camera effects] {len(paths)} images processed")


def _cached_raw(path, raw_dir):
    cached = os.path.join(raw_dir, os.path.basename(path))
    if not os.path.exists(cached):   # keep the render untouched so this pass can be re-run or compared
        shutil.copy2(path, cached)
    return cv2.imread(cached, cv2.IMREAD_UNCHANGED)


def _chance(rng, name):
    return rng.random() < EFFECTS[name]["prob"]


def _draw(rng, name, key):
    lo, hi = EFFECTS[name][key]
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
        gain = 1 - k * (dx * dx + dy * dy)   # always samples inside the frame, so no border runs out of pixels
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

    gain = float(np.clip(_draw(rng, "exposure", "target") / max(bgr.mean(), 1.0), *EFFECTS["exposure"]["gain"]))
    if rng.random() < EFFECTS["exposure"]["miss_prob"]:
        gain *= _draw(rng, "exposure", "miss")   # backlit and spot lit frames are where metering fails
    bgr = bgr * gain + _draw(rng, "exposure", "lift")

    if _chance(rng, "aberration"):
        shift = _draw(rng, "aberration", "shift")
        for channel, scale in ((0, 1 + shift), (2, 1 - shift)):
            matrix = np.array([[scale, 0, (1 - scale) * (w - 1) / 2], [0, scale, (1 - scale) * (h - 1) / 2]], np.float32)
            bgr[..., channel] = cv2.warpAffine(bgr[..., channel], matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if _chance(rng, "defocus_blur"):
        bgr = cv2.GaussianBlur(bgr, (0, 0), _draw(rng, "defocus_blur", "sigma"))

    if _chance(rng, "motion_blur"):
        bgr = cv2.filter2D(bgr, -1, _streak_kernel(rng))

    if _chance(rng, "bloom"):
        highlights = np.clip(bgr - _draw(rng, "bloom", "threshold"), 0, None)
        bgr += _draw(rng, "bloom", "strength") * cv2.GaussianBlur(highlights, (0, 0), _draw(rng, "bloom", "sigma"))

    if _chance(rng, "vignette"):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        bgr *= (1 - _draw(rng, "vignette", "strength") * r2)[..., None]

    if _chance(rng, "white_balance"):
        lo, hi = EFFECTS["white_balance"]["gain"]
        bgr *= rng.uniform(lo, hi, size=3).astype(np.float32)

    if _chance(rng, "noise"):
        sigma = min(_draw(rng, "noise", "sigma") * gain ** NOISE_GAIN_EXPONENT, EFFECTS["noise"]["ceiling"])
        bgr += rng.normal(0, sigma, (h, w, 1)).astype(np.float32)
        bgr += rng.normal(0, sigma * _draw(rng, "noise", "chroma"), bgr.shape).astype(np.float32)

    out = np.clip(bgr, 0, 255).astype(np.uint8)

    if _chance(rng, "jpeg"):
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(_draw(rng, "jpeg", "quality"))])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return np.concatenate([out, alpha], axis=2) if alpha is not None else out


def _streak_kernel(rng):
    length = int(_draw(rng, "motion_blur", "length")) | 1
    kernel = np.zeros((length, length), np.float32)
    angle = rng.uniform(0, np.pi)
    c = length // 2
    dx, dy = np.cos(angle) * c, np.sin(angle) * c
    cv2.line(kernel, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1)
    return kernel / kernel.sum()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder with rgb_*.png to degrade in place")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    apply_camera_effects(args.data_dir, args.seed)

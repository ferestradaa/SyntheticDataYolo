import argparse
import glob
import json
import os

import cv2
import numpy as np
import yaml

from config import load_config, resolve, DEFAULT_CONFIG

DARK_LUMA = 25            # mean luminance below this is an unusable frame
TINY_AREA = 0.0002        # fraction of the image, smaller labels are not worth learning from
WASHED_STD = 8.0          # an object clipped to white or crushed to black keeps no internal contrast


def label_area(values):
    coords = np.array(values, dtype=float)
    if len(coords) == 4:   # detect label: cx cy w h
        return coords[2] * coords[3]
    xs, ys = coords[0::2], coords[1::2]
    return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))   # shoelace


def label_mask(labels, shape):
    mask = np.zeros(shape, np.uint8)
    h, w = shape
    for v in labels:
        coords = np.array(v[1:], dtype=float)
        if len(coords) == 4:   # detect label: cx cy w h
            x0, y0 = (coords[0] - coords[2] / 2) * w, (coords[1] - coords[3] / 2) * h
            cv2.rectangle(mask, (int(x0), int(y0)), (int(x0 + coords[2] * w), int(y0 + coords[3] * h)), 255, -1)
        else:
            cv2.fillPoly(mask, [(coords.reshape(-1, 2) * [w, h]).astype(np.int32)], 255)
    return mask


def main(data_dir):
    meta = json.load(open(os.path.join(data_dir, "frames_meta.json")))
    names = yaml.safe_load(open(os.path.join(data_dir, "data.yaml")))["names"]
    problems = {key: [] for key in ("unexpected_class", "focus_missing", "empty_not_negative", "dark", "washed", "only_tiny")}
    total = 0
    for split in ("train", "val"):
        for path in sorted(glob.glob(os.path.join(data_dir, "labels", split, "rgb_*.txt"))):
            base = os.path.basename(path)[:-4]
            frame = int(base.split("_")[1])
            info = meta[frame]
            total += 1
            labels = [line.split() for line in open(path) if line.strip()]
            areas = {}
            for v in labels:
                name = names[int(v[0])]
                areas[name] = areas.get(name, 0) + label_area(v[1:])

            if any(info["visible"].get(name, 0) == 0 for name in areas):
                problems["unexpected_class"].append(frame)   # a class that was hidden shows up: frames and metadata are misaligned
            if any(name not in areas for name in info["focus"]):
                problems["focus_missing"].append(frame)   # the framed object has no label: lost or fully occluded
            if not labels and not info["negative"]:
                problems["empty_not_negative"].append(frame)
            if areas and max(areas.values()) < TINY_AREA:
                problems["only_tiny"].append(frame)
            image = cv2.imread(os.path.join(data_dir, "images", split, base + ".png"), cv2.IMREAD_GRAYSCALE)
            if image.mean() < DARK_LUMA:
                problems["dark"].append(frame)
            if labels:
                inside = image[label_mask(labels, image.shape) > 0]
                if inside.size >= 50 and inside.std() < WASHED_STD:
                    problems["washed"].append(frame)

    print(f"frames checked: {total}")
    for key, frames in problems.items():
        print(f"{key:20s} {len(frames):4d} ({100 * len(frames) / max(total, 1):4.1f}%)  {frames[:12]}")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data_dir", help="default: from config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(args.data_dir or str(resolve(cfg["data_dir"])))

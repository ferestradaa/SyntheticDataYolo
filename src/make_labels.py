import os, re, glob, argparse
import numpy as np
from PIL import Image
import cv2

from config import load_config, resolve, DEFAULT_CONFIG


def mask_to_polygons(binm, W, H):
    out = []
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 10:
            continue
        pts = cnt.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            continue
        coords = []
        for x, y in zip(pts[:, 0] / float(W), pts[:, 1] / float(H)):
            coords += [f"{x:.6f}", f"{y:.6f}"]
        out.append(" ".join(coords))
    return out


def mask_to_box(binm, W, H):
    ys, xs = np.nonzero(binm)   # one box around every visible pixel, an occluded instance still gets a single box
    if len(xs) < 10:
        return []
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return [f"{(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
            f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"]


def main(data_dir: str, task: str):
    masks_root = (data_dir if os.path.basename(data_dir) == "masks"
                  else os.path.join(data_dir, "masks"))
    labels_root = (data_dir if os.path.basename(data_dir) == "labels_raw"
                   else os.path.join(data_dir, "labels_raw"))

    if not os.path.isdir(masks_root):
        raise SystemExit(f"[ERROR] No dir of masks: {masks_root}")

    os.makedirs(labels_root, exist_ok=True)

    name_re = re.compile(r"^(\d+)_inst(\d+)\.png$", re.IGNORECASE)

    subs = [d for d in glob.glob(os.path.join(masks_root, "*")) if os.path.isdir(d)]

    total_txt = 0
    for sub in sorted(subs):
        base = os.path.basename(sub)
        out_txt = os.path.join(labels_root, f"{base}.txt")
        H = W = None
        lines = []

        for mpath in sorted(glob.glob(os.path.join(sub, "*.png"))):
            mname = os.path.basename(mpath)
            m = name_re.search(mname)
            if not m:
                print(f"[WARN] no parse class in {mname}, skipoping")
                continue

            cls_id = int(m.group(1))
            mask = np.array(Image.open(mpath))
            if mask.ndim == 3:
                mask = mask[..., 0]

            _, binm = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            if H is None or W is None:
                H, W = binm.shape[:2]

            convert = mask_to_polygons if task == "segment" else mask_to_box
            lines += [f"{cls_id} {geom}" for geom in convert(binm, W, H)]

        with open(out_txt, "w") as f:   # empty file for frames without objects, YOLO reads them as negatives
            f.write("\n".join(lines) + ("\n" if lines else ""))
        total_txt += 1

    print(f"[OK] .txt generated: {total_txt} in {labels_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Convert instance masks to YOLO labels (polygons or boxes)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--data_dir", help="root of dataset or masks or labels_raw/ (default: from config)")
    ap.add_argument("--task", choices=["segment", "detect"], help="default: from config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    main(os.path.realpath(args.data_dir or resolve(cfg["data_dir"])), args.task or cfg["task"])

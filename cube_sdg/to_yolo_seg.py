import os, re, glob, argparse
import numpy as np
from PIL import Image
import cv2


def main(data_dir: str):
    masks_root = (data_dir if os.path.basename(data_dir) == "masks_bin"
                  else os.path.join(data_dir, "masks_bin"))
    labels_root = (data_dir if os.path.basename(data_dir) == "labels_yolo_seg"
                   else os.path.join(data_dir, "labels_yolo_seg"))

    if not os.path.isdir(masks_root):
        raise SystemExit(f"[ERR] No existe carpeta de máscaras: {masks_root}")

    os.makedirs(labels_root, exist_ok=True)

    # Nombre de archivo: {cls_id}_inst{n}.png  → captura cls_id como group(1)
    name_re = re.compile(r"^(\d+)_inst(\d+)\.png$", re.IGNORECASE)

    subs = [d for d in glob.glob(os.path.join(masks_root, "*")) if os.path.isdir(d)]
    print(f"[INFO] subcarpetas: {len(subs)}")

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
                print(f"[WARN] no parsea clase en {mname}, salto")
                continue

            cls_id = int(m.group(1))
            mask = np.array(Image.open(mpath))
            if mask.ndim == 3:
                mask = mask[..., 0]

            _, binm = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            if H is None or W is None:
                H, W = binm.shape[:2]

            cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 10:
                    continue
                pts = cnt.squeeze()
                if pts.ndim != 2 or len(pts) < 3:
                    continue
                xs = pts[:, 0] / float(W)
                ys = pts[:, 1] / float(H)
                coords = []
                for x, y in zip(xs, ys):
                    coords += [f"{x:.6f}", f"{y:.6f}"]
                lines.append(f"{cls_id} " + " ".join(coords))

        if lines:
            with open(out_txt, "w") as f:
                f.write("\n".join(lines) + "\n")
            total_txt += 1

    print(f"[OK] archivos .txt generados: {total_txt} en {labels_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Convierte masks_bin -> labels_yolo_seg (polígonos YOLO)")
    ap.add_argument("--data_dir", required=True,
                    help="Raíz del dataset o directamente masks_bin/ o labels_yolo_seg/")
    args = ap.parse_args()
    main(os.path.realpath(args.data_dir))
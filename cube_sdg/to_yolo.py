import os, glob, json, argparse
import numpy as np
from PIL import Image

#CLASS_ORDER = ["yellow", "red", "green", "blue", "brown", "purple", "dark_green"]
CLASS_ORDER = ["pallet_full"]


def _key_to_rgba(k: str):
    return tuple(map(int, k.strip("()").split(",")))


def _rgba_array(img_pil):
    im = img_pil
    if im.mode not in ("RGBA", "RGB"):
        im = im.convert("RGBA")
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=2)
    return arr


def convert_folder(root_dir):
    out_root = os.path.join(root_dir, "masks_bin")
    os.makedirs(out_root, exist_ok=True)

    seg_pngs = sorted(glob.glob(os.path.join(root_dir, "semantic_segmentation_*.png")))
    if not seg_pngs:
        print(f"[WARN] No hay semantic_segmentation_*.png en {root_dir}")
        return

    name2id = {cls: i for i, cls in enumerate(CLASS_ORDER)}
    print(f"[INFO] CLASS MAP fijo: {name2id}")

    total_saved = 0
    frames_processed = 0

    for seg_path in seg_pngs:
        basename = os.path.splitext(os.path.basename(seg_path))[0]
        num = basename.split("semantic_segmentation_")[-1]

        json_path = os.path.join(root_dir, f"semantic_segmentation_labels_{num}.json")
        if not os.path.exists(json_path):
            print(f"[WARN] No encontré JSON para frame {num}: {json_path}")
            continue

        with open(json_path) as f:
            raw = json.load(f)

        color2class = {}
        for k, v in raw.items():
            cls_name = v.get("class", "") if isinstance(v, dict) else v
            if cls_name in ("BACKGROUND", "UNLABELLED") or cls_name not in name2id:
                continue
            color2class[_key_to_rgba(k)] = cls_name

        rgb_candidates = [
            os.path.join(root_dir, f"rgb_{num}.png"),
            os.path.join(root_dir, f"rgb_{num}.jpg"),
            os.path.join(root_dir, f"{num}.png"),
        ]
        base_name = f"rgb_{num}"
        for c in rgb_candidates:
            if os.path.exists(c):
                base_name = os.path.splitext(os.path.basename(c))[0]
                break

        arr = _rgba_array(Image.open(seg_path))
        colors = np.unique(arr.reshape(-1, 4), axis=0)

        out_dir = os.path.join(out_root, base_name)
        os.makedirs(out_dir, exist_ok=True)

        inst_count = {}
        for rgba in colors:
            tu = tuple(int(x) for x in rgba)
            cls_name = color2class.get(tu)
            if cls_name is None:
                continue

            cls_id = name2id[cls_name]
            mask = (arr == rgba).all(axis=2).astype(np.uint8) * 255
            inst_count.setdefault(cls_id, 0)
            fn = os.path.join(out_dir, f"{cls_id}_inst{inst_count[cls_id]}.png")
            Image.fromarray(mask).convert("L").save(fn)
            inst_count[cls_id] += 1
            total_saved += 1

        frames_processed += 1

    with open(os.path.join(out_root, "classes.txt"), "w") as f:
        for cls_name in CLASS_ORDER:
            f.write(f"{name2id[cls_name]}:{cls_name}\n")

    print(f"[OK] frames: {frames_processed} | clases: {name2id} | máscaras: {total_saved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args, _ = parser.parse_known_args()
    convert_folder(os.path.realpath(args.data_dir))
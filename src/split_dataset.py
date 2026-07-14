import argparse
import os
import random
import shutil
from pathlib import Path


def build_pairs(data_dir):
    # find all label txt files and check that a matching png exists
    labels_dir = data_dir / "labels_raw"
    pairs = []
    for txt_path in sorted(labels_dir.glob("*.txt")):
        base = txt_path.stem
        png_path = data_dir / f"{base}.png"
        if png_path.exists():
            pairs.append(base)
    return pairs


def split_pairs(pairs, train_ratio, seed):
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    k = int(n * train_ratio)

    train = shuffled[:k]
    val = shuffled[k:]
    return train, val


def make_dirs(data_dir):
    for split in ("train", "val"):
        (data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def link_split(data_dir, names, split):
    for base in names:
        img_src = data_dir / f"{base}.png"
        lbl_src = data_dir / "labels_raw" / f"{base}.txt"

        img_dst = data_dir / "images" / split / f"{base}.png"
        lbl_dst = data_dir / "labels" / split / f"{base}.txt"

        img_dst.unlink(missing_ok=True)
        lbl_dst.unlink(missing_ok=True)

        # hardlink instead of copy: as fast as a symlink but survives
        # deletion of the original file, since both point to the same inode
        os.link(img_src, img_dst)
        os.link(lbl_src, lbl_dst)


def read_classes_txt(data_dir):
    classes_path = data_dir / "classes.txt"
    if not classes_path.exists():
        raise SystemExit(f"classes.txt not found at {classes_path}")

    id_to_name = {}
    for line in classes_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, name = line.split(":", 1)
        id_to_name[int(idx_str)] = name.strip()

    ordered = [id_to_name[i] for i in sorted(id_to_name)]
    return ordered


def find_full_coverage_sample(data_dir, pairs, num_classes):
    # find the first label that contains all classes (0..num_classes-1)
    # if none has full coverage, return the one covering the most distinct classes
    labels_dir = data_dir / "labels_raw"
    best_base = None
    best_count = -1

    for base in pairs:
        txt_path = labels_dir / f"{base}.txt"
        seen = set()
        for line in txt_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            class_id = line.split()[0]
            seen.add(class_id)

        if len(seen) == num_classes:
            return base

        if len(seen) > best_count:
            best_count = len(seen)
            best_base = base

    return best_base


def save_sample_and_clean(data_dir, sample_base, keep_dirs, keep_files):
    # copy one example of each raw file/folder type into sample_check/
    # before deleting everything else, so the process can be visually verified
    sample_dir = data_dir / "sample_check"
    sample_dir.mkdir(exist_ok=True)

    if sample_base is not None:
        for f in data_dir.glob(f"*{sample_base}*"):
            if f.is_file():
                shutil.copy2(f, sample_dir / f.name)

        raw_label_src = data_dir / "labels_raw" / f"{sample_base}.txt"
        if raw_label_src.exists():
            shutil.copy2(raw_label_src, sample_dir / raw_label_src.name)

    for d in ("masks",):
        src = data_dir / d
        if src.is_dir():
            entries = sorted(src.iterdir())
            if entries:
                dst = sample_dir / d
                dst.mkdir(exist_ok=True)
                first = entries[0]
                if first.is_file():
                    shutil.copy2(first, dst / first.name)
                else:
                    shutil.copytree(first, dst / first.name, dirs_exist_ok=True)

    for item in data_dir.iterdir():
        if item.name in keep_dirs or item.name in keep_files:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def write_yaml(data_dir, class_names):
    yaml_path = data_dir / "data.yaml"
    names_block = "\n".join(f"  - {c}" for c in class_names)

    content = (
        f"path: ."
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="after the split, save one sample and delete the rest of the raw files",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    class_names = read_classes_txt(data_dir)

    pairs = build_pairs(data_dir)
    if not pairs:
        raise SystemExit(f"no image/label pairs found in {data_dir}")

    train, val = split_pairs(pairs, args.train_ratio, args.seed)

    make_dirs(data_dir)
    link_split(data_dir, train, "train")
    link_split(data_dir, val, "val")

    write_yaml(data_dir, class_names)

    print(f"[OK] data.yaml created at {data_dir / 'data.yaml'}")
    print(f"[OK] {len(train)} train / {len(val)} val")

    if args.clean:
        keep_dirs = {"images", "labels", "sample_check"}
        keep_files = {"data.yaml", "classes.txt"}
        sample_base = find_full_coverage_sample(data_dir, pairs, len(class_names))
        save_sample_and_clean(data_dir, sample_base, keep_dirs, keep_files)
        print(f"[OK] raw cleaned, sample saved at {data_dir / 'sample_check'} (base: {sample_base})")


if __name__ == "__main__":
    main()
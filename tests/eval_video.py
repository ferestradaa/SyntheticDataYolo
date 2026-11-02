import argparse
import collections
import json
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_filter import collect_tracks, surviving

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser("Run a model on a video, drop the unstable tracks and write it annotated")
    ap.add_argument("video")
    ap.add_argument("--task", default="segment", choices=("segment", "detect"))
    ap.add_argument("--weights", help="default: results/weights/<task>.pt")
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=0)
    ap.add_argument("--out_dir", help="default: results/videos/<video name>")
    args = ap.parse_args()

    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise SystemExit(f"video not found: {video}")
    weights = args.weights or REPO_ROOT / "results" / "weights" / f"{args.task}.pt"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "videos" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_area = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if not frame_area:
        raise SystemExit(f"could not read {video}")

    per_frame, classes = collect_tracks(YOLO(str(weights)), str(video), args.conf, args.imgsz, frame_area, args.device)
    keep, total_tracks = surviving(per_frame)

    kept = collections.defaultdict(list)
    spans = {}
    for i, frame in enumerate(per_frame):
        for tid, score, _ in frame:
            if tid in keep:
                kept[classes[tid]].append(score)
                spans.setdefault(tid, [i, i])[1] = i
    report = {"video": video.name, "task": args.task, "conf": args.conf, "frames": len(per_frame),
              "tracks": {"found": total_tracks, "kept": len(keep)},
              "per_class": {c: {"detections": len(v), "mean_conf": round(float(np.mean(v)), 3),
                                "tracks": sorted(t for t in keep if classes[t] == c)} for c, v in sorted(kept.items())},
              "spans": {str(t): spans[t] for t in sorted(spans)}}
    json.dump(report, open(out_dir / f"{args.task}_tracks.json", "w"), indent=1)

    render = YOLO(str(weights))   # a fresh model, otherwise persist keeps counting ids up from the first pass
    out = out_dir / f"{args.task}.mp4"
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=1, pixelformat="yuv420p")
    drawn = 0
    for r in render.track(source=str(video), stream=True, persist=True, conf=args.conf, imgsz=args.imgsz,
                          tracker="bytetrack.yaml", verbose=False, device=args.device):
        ids = r.boxes.id
        ids = [] if ids is None else [int(v) for v in ids]
        index = [k for k, tid in enumerate(ids) if tid in keep]
        drawn += len(index)
        shown = r[index] if index else r[0:0]
        writer.append_data(cv2.cvtColor(shown.plot(line_width=3, font_size=28), cv2.COLOR_BGR2RGB))
    writer.close()

    print(f"{args.task} {video.name}: {len(per_frame)} frames, tracks {total_tracks} to {len(keep)}, {drawn} boxes drawn")
    for c, v in report["per_class"].items():
        print(f"  {c:16s} {v['detections']:5d} detections  mean conf {v['mean_conf']:.2f}  {len(v['tracks'])} tracks")
    print(f"  written to {out}")


if __name__ == "__main__":
    main()

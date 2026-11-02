import collections

import numpy as np

MIN_LEN = 15            # frames, a real object stays in view much longer than a flicker
MIN_MEAN_CONF = 0.55
MIN_AREA = 0.005        # fraction of the frame, median over the track


def collect_tracks(model, video, conf, imgsz, frame_area, device=0):
    per_frame, classes = [], {}
    for r in model.track(source=video, stream=True, persist=True, conf=conf, imgsz=imgsz,
                         tracker="bytetrack.yaml", verbose=False, device=device):
        ids = r.boxes.id
        ids = [] if ids is None else [int(v) for v in ids]
        n = len(ids)
        scores = [float(s) for s in r.boxes.conf][:n]
        areas = [float((b[2] - b[0]) * (b[3] - b[1])) / frame_area for b in r.boxes.xyxy][:n]
        for tid, cls in zip(ids, [int(c) for c in r.boxes.cls][:n]):
            classes[tid] = model.names[cls]
        per_frame.append(list(zip(ids, scores, areas)))
    return per_frame, classes


def surviving(per_frame):
    seen = collections.defaultdict(list)
    for frame in per_frame:
        for tid, score, size in frame:
            seen[tid].append((score, size))
    keep = {tid for tid, items in seen.items()
            if len(items) >= MIN_LEN and np.mean([s for s, _ in items]) >= MIN_MEAN_CONF
            and np.median([a for _, a in items]) >= MIN_AREA}
    return keep, len(seen)

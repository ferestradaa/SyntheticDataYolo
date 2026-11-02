# synth_yolo

A synthetic data pipeline for YOLO object detection and segmentation, built on NVIDIA
Isaac Sim Replicator. It renders randomized frames of four household objects, converts
the instance masks into YOLO labels and splits them into a ready-to-train dataset.

The models in `results/weights/` are trained **only** on these synthetic frames. No real
image is ever used for training, and the real phone videos in `tests/videos/` are used
only for measuring.

![detections on real video](docs/demo.gif)

## How it works

`run/generate_data.sh` runs four stages in order, all driven by `run/config.yaml`:

| stage | what it does |
|---|---|
| `src/generate_data.py` | launches Isaac Sim headless with path tracing, loads the assets plus 62 Isaac warehouse/hospital/office props as distractors, and renders `num_frames` randomized frames with their instance masks |
| `src/extract_masks.py` | splits each instance-segmentation PNG into one binary mask per instance under `masks/`, and writes `classes.txt` |
| `src/make_labels.py` | traces every mask into a YOLO polygon label in `labels_raw/`, writing an empty `.txt` for frames with no objects so YOLO reads them as negatives |
| `src/split_dataset.py` | hardlinks the pairs into `images/{train,val}` and `labels/{train,val}`, writes `data.yaml`, and with `--clean` keeps one example in `sample_check/` and deletes the raw files |

Every frame is randomized from scratch. The randomization is split across three modules:

- `src/domain_randomization.py` - layout, object poses and scales, which instances are
  visible, distractor placement, camera position, framing and focal length. The camera
  never looks from below, and shots are drawn as 35% close, 40% medium, 25% wide.
- `src/lighting.py` - a dome light driven from one of six styles (`overcast`, `window`,
  `indoor`, `mixed`, `backlit`, `dim`), with colour temperature, local lights and an
  optional sun whose angular diameter sets how hard the shadows are.
- `src/camera_effects.py` - a lens and sensor pass applied to the rendered PNG: barrel
  distortion, chromatic aberration, defocus and motion blur, bloom, vignette, exposure
  and white balance, sensor noise and JPEG recompression. The untouched render is kept
  in `raw/` so this pass can be re-run or compared.

The ranges for all of it live at the top of each module, not in the config, so a dataset
is described by the code that made it.

## Requirements

- **Isaac Sim 4.2** (the `omni.isaac.*` namespace), run through its own `python.sh`
- An NVIDIA GPU, and access to the Isaac Nucleus asset server for the distractor props
- For training and video evaluation, a separate Python environment with
  `pip install ultralytics`, which pulls in torch

The generation stages use only what ships inside Isaac Sim: `numpy`, `opencv`, `PIL`,
`PyYAML` and `pxr`.

## Quick start

1. Point the config at your Isaac Sim install and pick a size:

   ```yaml
   # run/config.yaml
   isaac_sim_path: /opt/isaac-sim-4.2.0    # absolute path, no ~
   data_dir: data_generated/data_1000
   num_frames: 1000
   ```

   `ISAAC_SIM_PATH` in the environment overrides the config value.

2. Generate:

   ```bash
   bash run/generate_data.sh
   ```

   Extra arguments are passed through to the generator and override the config, which is
   the fastest way to check a change before committing to a long run:

   ```bash
   bash run/generate_data.sh --num_frames 10 --data_dir data_generated/smoke
   ```

3. The result is a YOLO dataset, with `sample_check/` holding one frame in every
   intermediate form so the run can be eyeballed:

   ```
   data_generated/data_1000/
     data.yaml            train/val paths, nc and class names
     classes.txt          id:name, in the order the labels use
     frames_meta.json     what each frame was supposed to show
     images/train/        rgb_0000.png ...
     images/val/
     labels/train/        rgb_0000.txt, one polygon per line: cls x1 y1 x2 y2 ...
     labels/val/
     sample_check/        one frame with its label, mask and raw render
   ```

![synthetic frames with their polygons](docs/dataset_sample.jpg)

## Configuration

| key | meaning |
|---|---|
| `isaac_sim_path` | absolute path to the Isaac Sim install, no `~` |
| `data_dir` | where the dataset is written, relative to the repo root |
| `assets` | one `.usd` per class; the class name is the file name without extension |
| `instances_per_asset` | copies spawned of each asset, each frame shows between 0 and this many of each |
| `num_frames` | how many frames to render |
| `width`, `height` | render size, square; every frame is then cropped to a random phone aspect |
| `train_ratio` | fraction of the pairs that go to train |
| `seed` | optional, makes the randomization and the lens/sensor pass reproducible |

Adding a class is one line in `assets:` plus the `.usd` in `meshes/`. Nothing else needs
to change: the class list is discovered from the rendered semantics.

## Checking a dataset

`src/validate_dataset.py` cross-checks the labels against `frames_meta.json` and reports
the share of frames that are unusable, so a bad run is caught before training:

```bash
$ISAAC_SIM_PATH/python.sh src/validate_dataset.py --data_dir data_generated/data_1000
```

It flags frames whose labels name a class that was supposed to be hidden, frames missing
the object the shot was framed around, empty labels that were not meant to be negatives,
frames that came out too dark or clipped to white, and frames where every label is tiny.

## Training

The datasets train with the stock ultralytics CLI, no wrapper in this repo. What produced
the weights in `results/weights/`:

```bash
yolo segment train model=yolo11n-seg.pt data=data_generated/data_2000b/data.yaml \
  epochs=100 imgsz=640 batch=16 seed=0 \
  hsv_h=0.03 hsv_s=0.7 hsv_v=0.5 degrees=10 translate=0.15 scale=0.7 \
  mosaic=1.0 mixup=0.15 copy_paste=0.15 erasing=0.4
```

Swap `segment`/`yolo11n-seg.pt` for `detect`/`yolo11n.pt` to train the box model. The
final models were trained on 2000 frames, 1800 train and 200 val, rendered at 960x960
before the random phone crop.

## Running on real video

```bash
python tests/eval_video.py tests/videos/kitchen_day.MOV --task segment
```

This writes an annotated `.mp4` and a json dump of the surviving tracks next to it. It
applies `tests/track_filter.py`, which drops any track shorter than 15 frames, below 0.55
mean confidence or covering less than 0.5 percent of the frame, and runs at `conf 0.70`.

That threshold is a measured choice, not a default: it removes 98 percent of the false
keys and 59 percent of the false blenders for 1 to 3 percent of the true detections. The
numbers behind it, and what still fails, are in `results/RESULTS.txt`.

## Repo layout

```
run/           config.yaml and the one script that runs the whole pipeline
src/           the four pipeline stages plus the randomization modules
meshes/        the four target objects as .usd, with their textures
hdri/          dome lighting environments, 2k
blender/       the scripts that rebuilt the meshes from photographs
tests/         evaluation on real video, not unit tests
results/       weights, training curves, annotated videos and the write-up
```

## Assets and credits

The four objects were rebuilt from phone photographs rather than downloaded: proportions
solved by PnP and bundle adjustment, textures projected from the photos, brand marks
removed. `blender/README.md` documents how each one was reproduced and what scale
assumptions it rests on.

The HDRIs come from [Poly Haven](https://polyhaven.com) (CC0), downscaled to 2k. The
distractor props and surface textures are referenced from the Isaac Sim asset server at
run time and are not redistributed here.

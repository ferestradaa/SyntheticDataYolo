#!/bin/bash
set -e

# --- Rutas ---
ISAAC_SIM_PATH="/home/ferestrada/Downloads/isaac-sim-std-42/"
cd ../cube_sdg

SCRIPT_SDG="${PWD}/yolo_hdri.py"           # SDG en Isaac (genera RGB + instance_seg)
SCRIPT_SPLIT="${PWD}/to_yolo.py"           # split de máscaras (se ejecuta en host con python.sh)
SCRIPT_CONVERT="${PWD}/to_yolo_seg.py"     # conversión de máscaras a YOLO
OUTPUT_DIR="${PWD}/data_generated/yolo_2"

# --- 1) Generación en Isaac ---s
cd "$ISAAC_SIM_PATH"
./python.sh "$SCRIPT_SDG" --height 720 --width 1280 --num_frames 10 \
  --distractors additional --data_dir "$OUTPUT_DIR"

# --- 2) Split a binarios ---
./python.sh "$SCRIPT_SPLIT" --data_dir "$OUTPUT_DIR"
./python.sh "$SCRIPT_CONVERT" --data_dir "$OUTPUT_DIR"


# --- 3) Crear estructura YOLO ---
cd "$OUTPUT_DIR"
rm -rf images labels
mkdir -p images/train images/val labels/train labels/val

# Solo usar basenames que tienen .png y .txt existentes
for txt in labels_yolo_seg/*.txt; do
  base=$(basename "$txt" .txt)
  if [ -f "$base.png" ]; then
    echo "$base"
  fi
done > _all.txt

# Split 90/10
shuf _all.txt > _all_shuf.txt
n=$(wc -l < _all_shuf.txt)
k=$(( n * 9 / 10 ))
head -n $k _all_shuf.txt > _train.txt
tail -n +$((k+1)) _all_shuf.txt > _val.txt

# Enlazar imágenes y labels
while read b; do
  ln -sf "$(pwd)/${b}.png" "images/train/${b}.png"
  ln -sf "$(pwd)/labels_yolo_seg/${b}.txt" "labels/train/${b}.txt"
done < _train.txt

while read b; do
  ln -sf "$(pwd)/${b}.png" "images/val/${b}.png"
  ln -sf "$(pwd)/labels_yolo_seg/${b}.txt" "labels/val/${b}.txt"
done < _val.txt

cat <<EOF > data.yaml
path: $(pwd)
train: images/train
val: images/val

nc: 7
names:
  - yellow
  - red
  - green
  - blue
  - brown
  - purple
  - dark_green
EOF

echo "[OK] data.yaml creado en $(pwd)/data.yaml"
echo "[OK] $(wc -l < _train.txt) train / $(wc -l < _val.txt) val"

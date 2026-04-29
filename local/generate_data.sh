set -e

#path to your isaac sim installation 
ISAAC_SIM_PATH="/home/ferestrada/Downloads/isaac-sim-std-42/"
cd ../cube_sdg

SCRIPT_SDG="${PWD}/pallet.py"           # python scritp to generatr SD
SCRIPT_SPLIT="${PWD}/to_yolo.py"           #convert images to yolo dataset format
SCRIPT_CONVERT="${PWD}/to_yolo_seg.py"     #pipeline to use segmentation dataset
OUTPUT_DIR="${PWD}/data_generated/version2" #output for results 

cd "$ISAAC_SIM_PATH"
./python.sh "$SCRIPT_SDG" --height 720 --width 1280 --num_frames 50 \
  --distractors additional --data_dir "$OUTPUT_DIR"

./python.sh "$SCRIPT_SPLIT" --data_dir "$OUTPUT_DIR"
./python.sh "$SCRIPT_CONVERT" --data_dir "$OUTPUT_DIR"


cd "$OUTPUT_DIR"
rm -rf images labels
mkdir -p images/train images/val labels/train labels/val

for txt in labels_yolo_seg/*.txt; do
  base=$(basename "$txt" .txt)
  if [ -f "$base.png" ]; then
    echo "$base"
  fi
done > _all.txt

shuf _all.txt > _all_shuf.txt #split for train/test
n=$(wc -l < _all_shuf.txt)
k=$(( n * 9 / 10 ))
head -n $k _all_shuf.txt > _train.txt
tail -n +$((k+1)) _all_shuf.txt > _val.txt

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

echo "[OK] data.yaml created in $(pwd)/data.yaml"
echo "[OK] $(wc -l < _train.txt) train / $(wc -l < _val.txt) val"

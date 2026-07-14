set -e

#path to your isaac sim installation 
ISAAC_SIM_PATH="/home/ferestrada/Downloads/isaac-sim-std-42/"
cd ../synthetic_data

PARENT_DIR="$(dirname "$PWD")"
OUTPUT_DIR="${PARENT_DIR}/data_generated/data" #output for results 

SCRIPT_SDG="${PWD}/generate_data.py"           # python scritp to generatr SD
SCRIPT_SPLIT="${PWD}/to_yolo.py"           #convert images to yolo dataset format
SCRIPT_CONVERT="${PWD}/get_segmentation.py"     #get segmnentation coordinates for dataset
SPLIT="${PWD}/split_dataset.py" #split dataset to train val data


CLASSES_TXT="${OUTPUT_DIR}/classes.txt"

cd "$ISAAC_SIM_PATH"
./python.sh "$SCRIPT_SDG" --height 720 --width 1280 --num_frames 10 \
  --distractors additional --data_dir "$OUTPUT_DIR"


./python.sh "$SCRIPT_SPLIT" --data_dir "$OUTPUT_DIR"
./python.sh "$SCRIPT_CONVERT" --data_dir "$OUTPUT_DIR"
./python.sh "$SPLIT" --data_dir "$OUTPUT_DIR" --clean



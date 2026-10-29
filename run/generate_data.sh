
# Ethis script calls src/ files using the run/config.yaml file
set -e

# path to your isaac sim installation 
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/home/ferestrada/Downloads/isaac-sim-std-42/}"

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(dirname "$RUN_DIR")/src"
CONFIG="${CONFIG:-$RUN_DIR/config.yaml}"

cd "$ISAAC_SIM_PATH"
./python.sh "$SRC/generate_data.py" --config "$CONFIG" "$@"   # extra args override the config (e.g. --num_frames 10)
./python.sh "$SRC/extract_masks.py" --config "$CONFIG"
./python.sh "$SRC/make_labels.py"   --config "$CONFIG"
./python.sh "$SRC/split_dataset.py" --config "$CONFIG" --clean

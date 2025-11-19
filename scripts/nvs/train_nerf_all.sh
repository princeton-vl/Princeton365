#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_BASE="/path/to/NVS/data/base"


for scene_id in 35 51 98 129 131 154; do
  DATA_DIR="${DATA_BASE}/new_scanning_${scene_id}"
  sbatch "$SCRIPT_DIR/train_nerf_single.sh" "$DATA_DIR"
done
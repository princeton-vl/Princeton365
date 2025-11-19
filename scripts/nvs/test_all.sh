#!/bin/bash
set -euo pipefail
PARENT="/path/to/outputs"   # where ns-train wrote run dirs
METRICS_NAME="metrics.json" # name of the metrics file to write

if [[ ! -d "$PARENT" ]]; then
  echo "ERROR: outputs directory not found: $PARENT" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  NS_EVAL=(conda run -n nerfstudio --no-capture-output ns-eval)
else
  NS_EVAL=(ns-eval) 
fi

echo "Scanning for runs under: $PARENT"
find "$PARENT" -type f -name "config.yml" -print0 | while IFS= read -r -d '' CFG; do
  DIR="$(dirname "$CFG")"
  OUT_DIR="$DIR/eval"
  METRICS="$OUT_DIR/$METRICS_NAME"

  echo "[EVAL] $DIR"
  mkdir -p "$OUT_DIR"
  "${NS_EVAL[@]}" \
    --load-config "$CFG" \
    --output-path "$METRICS" \
    --render-output-path "$OUT_DIR"
done

echo "Done."

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$ROOT/configs/strict_gpt2_default.json}"
OUT="${2:-$ROOT/outputs/run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
EXTRA=()
if [ "${3:-}" = "--resume" ]; then
  EXTRA=(--resume-from "$OUT")
  echo "[INFO] Resume from $OUT/training_state.pt"
fi
python "$ROOT/train_strict_lm.py" --config "$CONFIG" --output-dir "$OUT" --num-workers 4 --logging-steps 50 "${EXTRA[@]}"
echo "Done. Model: $OUT/final"

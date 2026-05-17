#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/language/babytext/experiments/gpt2-masked-objective"
BASE_CFG="$PROJECT_ROOT/configs/run_mask10.json"
RUN4_CFG="$PROJECT_ROOT/configs/run_mask10_run4_taskfocus.json"

RUN_DIR="/data0/language/babylm_runs/mask10_run4"
DATA_FILE="/data0/language/babylm_runs/mask10_run4/data/task_focus_mnli_rte_wsc_qqp.train.txt"
DATA_DIR="/data0/language/babylm_runs/mask10_run4/data"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

if [ ! -f "$DATA_FILE" ]; then
  echo "[ERROR] Missing generated dataset: $DATA_FILE"
  echo "Run scripts/generate_mask10_run4_taskfocus_dataset.py first."
  exit 1
fi

python - <<'PY'
import json
from pathlib import Path

base = Path("/home/language/babytext/experiments/gpt2-masked-objective/configs/run_mask10.json")
out  = Path("/home/language/babytext/experiments/gpt2-masked-objective/configs/run_mask10_run4_taskfocus.json")

cfg = json.loads(base.read_text(encoding="utf-8"))
cfg["model_name"] = "gpt2_mask10_run4_taskfocus"
cfg["training_data_path"] = "/data0/language/babylm_runs/mask10_run4/data"
data_dir = Path(cfg["training_data_path"])
train_files = sorted(p.name for p in data_dir.glob("*.train.txt") if p.is_file())
if not train_files:
    raise FileNotFoundError(f"No *.train.txt found under {data_dir}")

# Keep all active sources strictly positive to avoid zero-sum sampling weights.
cfg["source_word_counts"] = {name: 1 for name in train_files}
cfg["notes"] = "run4 task-focused training using official mnli/rte/wsc/qqp train sets only"
out.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote config: {out}")
print(f"[OK] source_word_counts: {cfg['source_word_counts']}")
PY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_ROOT"
nohup python train_hybrid_gpt2.py \
  --config "$RUN4_CFG" \
  --output-dir "$RUN_DIR" \
  --num-workers 4 \
  --logging-steps 20 \
  --cuda-visible-devices 0 \
  --wandb \
  --wandb-project babylm-strict \
  --wandb-entity weichunzhou527-xi-an-jiaotong-liverpool-university \
  > "$LOG_FILE" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG_FILE"

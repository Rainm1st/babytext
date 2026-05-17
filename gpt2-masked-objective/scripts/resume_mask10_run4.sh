#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/language/babytext/experiments/gpt2-masked-objective"
RUN_CFG="$PROJECT_ROOT/configs/run_mask10_run4_taskfocus.json"
RUN_DIR="/data0/language/babylm_runs/mask10_run4"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

LATEST_CKPT="$(ls -d "$CKPT_DIR"/ckpt_step_* "$CKPT_DIR"/ckpt_epoch_* 2>/dev/null | sort -V | tail -n 1)"
if [ -z "${LATEST_CKPT}" ]; then
  echo "[ERROR] No checkpoint found in $CKPT_DIR"
  exit 1
fi

if pgrep -af "train_hybrid_gpt2.py.*$RUN_DIR" >/dev/null; then
  echo "[SKIP] A run for $RUN_DIR is already active."
  pgrep -af "train_hybrid_gpt2.py.*$RUN_DIR"
  exit 0
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG_FILE="$LOG_DIR/resume_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJECT_ROOT"
nohup python train_hybrid_gpt2.py \
  --config "$RUN_CFG" \
  --output-dir "$RUN_DIR" \
  --resume-checkpoint "$(realpath "$LATEST_CKPT")" \
  --num-workers 4 \
  --logging-steps 20 \
  --cuda-visible-devices 0 \
  --wandb \
  --wandb-project babylm-strict \
  --wandb-entity weichunzhou527-xi-an-jiaotong-liverpool-university \
  > "$LOG_FILE" 2>&1 &

echo "PID=$!"
echo "RESUME_FROM=$(realpath "$LATEST_CKPT")"
echo "LOG=$LOG_FILE"

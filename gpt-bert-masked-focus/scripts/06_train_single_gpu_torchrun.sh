#!/usr/bin/env bash
set -euo pipefail

# 单卡训练入口：必须用 torchrun + 完整 SLURM 环境变量。
# 单卡时 hybrid_denominator 必须为 1（官方 assert: world_size % hybrid_denominator == 0）。

GPT_BERT_ROOT="${GPT_BERT_ROOT:-/home/language/babytext/experiments/gpt-bert}"
RUN_DIR="${RUN_DIR:-/data0/language/babylm_runs/gpt_bert_masked_focus_continue}"
CONFIG_FILE="${CONFIG_FILE:-$GPT_BERT_ROOT/configs/base.json}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/data0/language/babylm_runs/gpt_bert_strict_tokenized/tokenizer}"
TOKENIZER_FILE="${TOKENIZER_FILE:-$TOKENIZER_DIR/tokenizer.json}"

TRAIN_PATH="${GPT_BERT_TRAIN_PATH:-}"
VALID_PATH="${GPT_BERT_VALID_PATH:-}"

if [ -z "$TRAIN_PATH" ] || [ -z "$VALID_PATH" ]; then
  echo "[ERROR] export GPT_BERT_TRAIN_PATH and GPT_BERT_VALID_PATH first."
  exit 1
fi
if [ ! -f "$TRAIN_PATH" ] || [ ! -f "$VALID_PATH" ] || [ ! -f "$TOKENIZER_FILE" ]; then
  echo "[ERROR] Missing file:"
  echo "  TRAIN=$TRAIN_PATH"
  echo "  VALID=$VALID_PATH"
  echo "  TOKENIZER=$TOKENIZER_FILE"
  exit 1
fi

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$LAUNCH_DIR/00_slurm_single_node_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# 与 SLURM_GPUS_ON_NODE 一致（仅 1 张可见卡时）
export SLURM_GPUS_ON_NODE=1
export WORLD_SIZE=1
export SLURM_NTASKS=1

mkdir -p "$RUN_DIR/logs"
LOG_FILE="$RUN_DIR/logs/train_torchrun_$(date +%Y%m%d_%H%M%S).log"

cd "$GPT_BERT_ROOT/pretraining"
python "$LAUNCH_DIR/09_patch_train_100m_dataloader.py" --train_script="$GPT_BERT_ROOT/pretraining/train_100m.py"

echo "[INFO] Starting torchrun single-GPU training..."
echo "[INFO] LOG=$LOG_FILE"

nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 train_100m.py \
  --train_path="$TRAIN_PATH" \
  --valid_path="$VALID_PATH" \
  --config_file="$CONFIG_FILE" \
  --tokenizer_path="$TOKENIZER_FILE" \
  --output_dir="$RUN_DIR" \
  --name="masked_focus_continue_strict" \
  --hybrid_numerator=1 \
  --hybrid_denominator=1 \
  --seq_length=128 \
  --local_batch_size=16 \
  --global_batch_size=4096 \
  --learning_rate=0.007 \
  --max_steps=2000 \
  --optimizer=lamb \
  --weight_decay=0.1 \
  --warmup_proportion=0.016 \
  --cooldown_proportion=0.016 \
  --mask_p_start=0.3 \
  --mask_p_end=0.15 \
  --mask_random_p=0.1 \
  --mask_keep_p=0.1 \
  --mixed_precision \
  --validate_every=200 \
  --save_every=200 \
  --seed=42 \
  > "$LOG_FILE" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG_FILE"
echo "[NOTE] 单卡使用 hybrid 1/1（全 masked step）。多卡 16 张时可改回 1/16 近似官方 masked-focus。"

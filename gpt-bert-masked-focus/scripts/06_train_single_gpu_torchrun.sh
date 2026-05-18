#!/usr/bin/env bash
set -euo pipefail

# 单卡 GPT-BERT 训练（torchrun + DDP world_size=1）。
# 从零：不传 GPT_BERT_RESUME_STATE。
# 断点续训：export GPT_BERT_RESUME_STATE=/path/to/xxx_state_dict.bin（train_100m 保存的完整 state）

GPT_BERT_ROOT="${GPT_BERT_ROOT:-/home/language/babytext/experiments/gpt-bert}"
RUN_DIR="${RUN_DIR:-/data0/language/babylm_runs/gpt_bert_scratch_strict}"
CONFIG_FILE="${CONFIG_FILE:-$GPT_BERT_ROOT/configs/base.json}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/data0/language/babylm_runs/gpt_bert_strict_tokenized/tokenizer}"
TOKENIZER_FILE="${TOKENIZER_FILE:-$TOKENIZER_DIR/tokenizer.json}"

TRAIN_PATH="${GPT_BERT_TRAIN_PATH:-}"
VALID_PATH="${GPT_BERT_VALID_PATH:-}"

# 训练步数：从零建议提高（严格赛道下 tokenizer 子词量远大于 2000 step）
MAX_STEPS="${GPT_BERT_MAX_STEPS:-20000}"

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
export SLURM_GPUS_ON_NODE=1
export WORLD_SIZE=1
export SLURM_NTASKS=1

# W&B：默认关闭远端；需同步云端时 export WANDB_DISABLED=false 并登录
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$RUN_DIR/logs"
LOG_FILE="$RUN_DIR/logs/train_torchrun_$(date +%Y%m%d_%H%M%S).log"

cd "$GPT_BERT_ROOT/pretraining"
python "$LAUNCH_DIR/09_patch_train_100m_dataloader.py" --train_script="$GPT_BERT_ROOT/pretraining/train_100m.py"
python "$LAUNCH_DIR/10_patch_train_100m_wandb_stub.py" --train_script="$GPT_BERT_ROOT/pretraining/train_100m.py" || true

EXTRA_ARGS=()
if [ -n "${GPT_BERT_RESUME_STATE:-}" ]; then
  if [ ! -f "$GPT_BERT_RESUME_STATE" ]; then
    echo "[ERROR] GPT_BERT_RESUME_STATE not a file: $GPT_BERT_RESUME_STATE"
    exit 1
  fi
  EXTRA_ARGS+=(--checkpoint_filename="$GPT_BERT_RESUME_STATE")
  echo "[INFO] Resume from $GPT_BERT_RESUME_STATE"
fi

echo "[INFO] Starting torchrun (MAX_STEPS=$MAX_STEPS, LOG=$LOG_FILE)"

nohup torchrun --standalone --nnodes=1 --nproc_per_node=1 train_100m.py \
  --train_path="$TRAIN_PATH" \
  --valid_path="$VALID_PATH" \
  --config_file="$CONFIG_FILE" \
  --tokenizer_path="$TOKENIZER_FILE" \
  --output_dir="$RUN_DIR" \
  --name="gpt_bert_strict_scratch" \
  --hybrid_numerator=1 \
  --hybrid_denominator=1 \
  --seq_length=128 \
  --local_batch_size="${GPT_BERT_LOCAL_BATCH:-16}" \
  --global_batch_size="${GPT_BERT_GLOBAL_BATCH:-4096}" \
  --learning_rate="${GPT_BERT_LR:-0.007}" \
  --max_steps="$MAX_STEPS" \
  --optimizer=lamb \
  --weight_decay=0.1 \
  --warmup_proportion=0.016 \
  --cooldown_proportion=0.016 \
  --mask_p_start=0.3 \
  --mask_p_end=0.15 \
  --mask_random_p=0.1 \
  --mask_keep_p=0.1 \
  --mixed_precision \
  --validate_every="${GPT_BERT_VALIDATE_EVERY:-500}" \
  --save_every="${GPT_BERT_SAVE_EVERY:-500}" \
  --seed=42 \
  "${EXTRA_ARGS[@]}" \
  > "$LOG_FILE" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG_FILE"
echo "[NOTE] 单卡 hybrid 1/1 = 全程 masked 分支；多卡 16 张可改 hybrid 1/16 贴近官方 Masked Focus。"
echo "[NOTE] 续训：将 RUN_DIR 下最新的 *state_dict.bin 路径赋给 GPT_BERT_RESUME_STATE 后重新执行本脚本。"

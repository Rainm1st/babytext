#!/usr/bin/env bash
set -euo pipefail

# 在官方 Strict tokenized 数据上继续预训练（Masked Focus 超参：约 15/16 masked, 1/16 causal）
# 单卡示例；多卡请用 gpt-bert/pretraining/train_multi_gpu.py

GPT_BERT_ROOT="${GPT_BERT_ROOT:-/home/language/babytext/experiments/gpt-bert}"
RUN_DIR="${RUN_DIR:-/data0/language/babylm_runs/gpt_bert_masked_focus_continue}"
# HF Masked Focus 基线 tokenizer 为 16k 词表；small.json 仅 8192，会导致 CUDA device-side assert
CONFIG_FILE="${CONFIG_FILE:-$GPT_BERT_ROOT/configs/base.json}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/data0/language/babylm_runs/gpt_bert_strict_tokenized/tokenizer}"
# train_100m.py 使用 Tokenizer.from_file()，必须是 tokenizer.json 文件，不能是目录
if [ -n "${TOKENIZER_PATH:-}" ] && [ -f "${TOKENIZER_PATH}" ]; then
  TOKENIZER_FILE="$TOKENIZER_PATH"
elif [ -f "$TOKENIZER_DIR/tokenizer.json" ]; then
  TOKENIZER_FILE="$TOKENIZER_DIR/tokenizer.json"
else
  echo "[ERROR] Missing tokenizer.json under $TOKENIZER_DIR"
  echo "Run: bash scripts/04b_tokenize_babylm_strict.sh"
  exit 1
fi

# 必须在 cd 之前固定脚本目录（cd 后 BASH_SOURCE 相对路径会指错）
LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_PATH="${GPT_BERT_TRAIN_PATH:-}"
VALID_PATH="${GPT_BERT_VALID_PATH:-}"

if [ -z "$TRAIN_PATH" ] || [ -z "$VALID_PATH" ]; then
  echo "[ERROR] Set GPT_BERT_TRAIN_PATH and GPT_BERT_VALID_PATH (tokenized .bin files)."
  echo "Run: bash scripts/04b_tokenize_babylm_strict.sh"
  exit 1
fi
if [ ! -f "$TRAIN_PATH" ] || [ ! -f "$VALID_PATH" ]; then
  echo "[ERROR] Tokenized files not found:"
  echo "  TRAIN=$TRAIN_PATH"
  echo "  VALID=$VALID_PATH"
  echo "Do NOT use placeholder paths like /你的/train.bin"
  echo "Run: bash scripts/04b_tokenize_babylm_strict.sh"
  exit 1
fi

mkdir -p "$RUN_DIR/logs"
LOG_FILE="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

MAX_STEPS="${GPT_BERT_MAX_STEPS:-20000}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-offline}"

EXTRA_ARGS=()
if [ -n "${GPT_BERT_RESUME_STATE:-}" ]; then
  if [ ! -f "$GPT_BERT_RESUME_STATE" ]; then
    echo "[ERROR] GPT_BERT_RESUME_STATE not a file: $GPT_BERT_RESUME_STATE"
    exit 1
  fi
  EXTRA_ARGS+=(--checkpoint_filename="$GPT_BERT_RESUME_STATE")
fi

# 官方 model_logging.py 在非 SLURM 机器上会读 SLURM_PROCID，必须先补齐
# shellcheck source=/dev/null
source "$LAUNCH_DIR/00_slurm_single_node_env.sh"
export SLURM_GPUS_ON_NODE=1
export WORLD_SIZE=1
export SLURM_NTASKS=1

cd "$GPT_BERT_ROOT/pretraining"

# 官方 train_100m 在 iter(dataloader) 后 wandb 访问 .dataset 会崩（PyTorch 新版本）
python "$LAUNCH_DIR/09_patch_train_100m_dataloader.py" --train_script="$GPT_BERT_ROOT/pretraining/train_100m.py"
python "$LAUNCH_DIR/10_patch_train_100m_wandb_stub.py" --train_script="$GPT_BERT_ROOT/pretraining/train_100m.py" || true

# Masked Focus: hybrid 1/16 causal ≈ numerator 1, denominator 16
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 官方脚本强制 DDP，单卡请用: bash 06_train_single_gpu_torchrun.sh
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
echo "TOKENIZER_FILE=$TOKENIZER_FILE"
echo "[NOTE] 若 train_100m.py 参数与仓库版本不一致，请对照 $GPT_BERT_ROOT/pretraining/README.md 调整。"

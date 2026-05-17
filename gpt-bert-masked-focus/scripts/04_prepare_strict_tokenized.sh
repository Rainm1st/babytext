#!/usr/bin/env bash
set -euo pipefail

# GPT-BERT 官方训练需要「预先 tokenize 的二进制语料」，不能直接用 *.train.txt。
# 本脚本：检查 Strict 原始语料存在，并打印官方 tokenization 步骤（需在 gpt-bert 仓库内执行）。

GPT_BERT_ROOT="${GPT_BERT_ROOT:-/home/language/babytext/experiments/gpt-bert}"
STRICT_RAW="${STRICT_RAW:-/home/language/babytext/experiments/BabyLM-2026-Strict}"
TOKENIZED_OUT="${TOKENIZED_OUT:-/data0/language/babylm_runs/gpt_bert_strict_tokenized}"

echo "[CHECK] GPT-BERT repo: $GPT_BERT_ROOT"
[ -d "$GPT_BERT_ROOT/pretraining" ] || { echo "[ERROR] Run 02_setup_gpt_bert_repo.sh first"; exit 1; }

echo "[CHECK] Strict raw data: $STRICT_RAW"
count=$(find "$STRICT_RAW" -maxdepth 1 -name "*.train.txt" 2>/dev/null | wc -l)
if [ "$count" -lt 1 ]; then
  echo "[ERROR] No *.train.txt under $STRICT_RAW"
  exit 1
fi
echo "[OK] Found $count train files"

mkdir -p "$TOKENIZED_OUT"

cat <<EOF

================================================================================
下一步（必须在 gpt-bert 仓库内按官方 README 做 corpus tokenization）：

1) cd $GPT_BERT_ROOT
2) 阅读 corpus_tokenization/ 下 README，对 Strict 语料做 tokenize
3) 输出目录建议指向: $TOKENIZED_OUT

官方流程概要：
  - tokenizer_creation/（若用官方 tokenizer，可跳过）
  - corpus_tokenization/  → 生成 train/valid 二进制文件
  - configs/small.json 或 base.json 选模型规模

完成后设置环境变量再训练：
  export GPT_BERT_TRAIN_PATH="$TOKENIZED_OUT/train.bin"   # 按你实际文件名修改
  export GPT_BERT_VALID_PATH="$TOKENIZED_OUT/valid.bin"   # 按你实际文件名修改
  bash $(dirname "$0")/05_train_continue_masked_focus.sh

================================================================================
EOF

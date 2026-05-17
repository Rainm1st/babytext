#!/usr/bin/env bash
set -euo pipefail

# 克隆官方 GPT-BERT 训练仓库（预训练代码）
GPT_BERT_ROOT="${GPT_BERT_ROOT:-/home/language/babytext/experiments/gpt-bert}"
REPO_URL="https://github.com/ltgoslo/gpt-bert.git"

if [ -d "$GPT_BERT_ROOT/.git" ]; then
  echo "[INFO] Repo exists, pulling: $GPT_BERT_ROOT"
  git -C "$GPT_BERT_ROOT" pull --ff-only || true
else
  echo "[INFO] Cloning $REPO_URL -> $GPT_BERT_ROOT"
  git clone "$REPO_URL" "$GPT_BERT_ROOT"
fi

python -m pip install -U pip
python -m pip install torch transformers tokenizers datasets wandb accelerate

echo "[OK] GPT-BERT repo ready at: $GPT_BERT_ROOT"
echo "[NEXT] bash scripts/04_prepare_strict_tokenized.sh"

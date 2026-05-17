#!/usr/bin/env bash
set -euo pipefail

# 下载官方 GPT-BERT Masked Focus 到「扁平 local-dir」（根目录直接含 config.json）
MODEL_ID="BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
OUT_DIR="${OUT_DIR:-/home/language/babytext/experiments/gpt-bert-masked-focus-baseline}"

mkdir -p "$OUT_DIR"

if command -v huggingface-cli >/dev/null 2>&1; then
  HF_CLI=huggingface-cli
elif command -v hf >/dev/null 2>&1; then
  HF_CLI=hf
else
  echo "[ERROR] 需要 huggingface-cli 或 hf: pip install -U huggingface_hub"
  exit 1
fi

echo "[INFO] Downloading ${MODEL_ID} -> ${OUT_DIR}"
"$HF_CLI" download "$MODEL_ID" --local-dir "$OUT_DIR"

if [ ! -f "$OUT_DIR/config.json" ]; then
  echo "[ERROR] Download incomplete: missing $OUT_DIR/config.json"
  echo "Check network/HF token, then re-run this script."
  exit 1
fi

echo "[OK] Model ready: $OUT_DIR"
echo "[OK] config.json exists"
echo "[NEXT] export HF_MODEL=$OUT_DIR && bash scripts/04b_tokenize_babylm_strict.sh"

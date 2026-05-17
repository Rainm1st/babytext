#!/usr/bin/env bash
set -euo pipefail

LAUNCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 逐步诊断 train_100m.py 在哪一步崩溃（不启动完整训练）
TRAIN_BIN="${GPT_BERT_TRAIN_PATH:-/data0/language/babylm_runs/gpt_bert_strict_tokenized/train_strict_small_small_tokenized.bin}"
VALID_BIN="${GPT_BERT_VALID_PATH:-$TRAIN_BIN}"
TOKENIZER_FILE="${TOKENIZER_FILE:-/data0/language/babylm_runs/gpt_bert_strict_tokenized/tokenizer/tokenizer.json}"
CONFIG_FILE="${CONFIG_FILE:-/home/language/babytext/experiments/gpt-bert/configs/base.json}"

echo "=== [1] files ==="
ls -lah "$TRAIN_BIN" "$VALID_BIN" "$TOKENIZER_FILE" "$CONFIG_FILE"

echo "=== [2] torch.load train.bin (may take 1-3 min) ==="
python - <<PY
import os, time, torch
p = os.environ.get("TRAIN_BIN", "$TRAIN_BIN")
t0 = time.time()
docs = torch.load(p, map_location="cpu")
print(f"[OK] train docs={len(docs)}, elapsed={time.time()-t0:.1f}s")
print(f"     first doc len(tokens)={len(docs[0]) if docs else 0}")
PY

echo "=== [3] estimate segment count (train) ==="
python - <<PY
import os, torch
p = os.environ.get("TRAIN_BIN", "$TRAIN_BIN")
seq = 126
docs = torch.load(p, map_location="cpu")
n = sum(
    max(0, (len(d) - 1) // seq)
    for d in docs
    if len(d) > 1
)
print(f"[OK] approx train segments={n:,}")
PY

echo "=== [4] tokenizer + config ==="
python - <<PY
import json
from pathlib import Path
from tokenizers import Tokenizer
tok = Tokenizer.from_file("$TOKENIZER_FILE")
cfg = json.loads(Path("$CONFIG_FILE").read_text())
print(f"[OK] tokenizer vocab={tok.get_vocab_size()}, config vocab_size={cfg.get('vocab_size')}")
PY

echo "=== [5] token id vs vocab (CUDA assert 诊断) ==="
python "$LAUNCH_DIR/08_check_token_ids.py" \
  --train_bin="$TRAIN_BIN" \
  --valid_bin="$VALID_BIN" \
  --tokenizer="$TOKENIZER_FILE" \
  --config="$CONFIG_FILE" || true

echo "=== [6] try import train deps ==="
python - <<'PY'
import sys
sys.path.insert(0, "/home/language/babytext/experiments/gpt-bert/pretraining")
from model_extra import Bert
print("[OK] import Bert")
PY

echo "=== DONE ==="
echo "If all OK, run foreground torchrun with WANDB_DISABLED=1 (see README or chat)."

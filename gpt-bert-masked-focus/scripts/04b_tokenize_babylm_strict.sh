#!/usr/bin/env bash
set -euo pipefail

# 把 BabyLM-2026-Strict 的 *.train.txt 转成 jsonl，再用 gpt-bert 官方 tokenize_corpus.py 生成 .bin
GPT_BERT_ROOT="${GPT_BERT_ROOT:-/home/language/babytext/experiments/gpt-bert}"
STRICT_RAW="${STRICT_RAW:-/home/language/babytext/experiments/BabyLM-2026-Strict}"
OUT_DIR="${OUT_DIR:-/data0/language/babylm_runs/gpt_bert_strict_tokenized}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/data0/language/babylm_runs/gpt_bert_strict_tokenized/tokenizer}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 自动解析：本地 snapshot > 本地目录 > Hub 模型 ID
HF_MODEL_RESOLVED="$(python "$SCRIPT_DIR/resolve_hf_model_path.py")"
echo "[INFO] Using model ref: $HF_MODEL_RESOLVED"

mkdir -p "$OUT_DIR" "$TOKENIZER_DIR"

echo "[1/4] Export tokenizer.json from HF checkpoint"
export HF_MODEL_RESOLVED TOKENIZER_DIR
python - <<'PY'
import os
from pathlib import Path
from transformers import AutoTokenizer

hf_model = os.environ["HF_MODEL_RESOLVED"]
out_dir = Path(os.environ["TOKENIZER_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)
print(f"[INFO] AutoTokenizer.from_pretrained({hf_model!r})")
tok = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True, local_files_only=False)
tok.save_pretrained(out_dir)
print(f"[OK] tokenizer saved to {out_dir}")
for name in ("tokenizer.json", "tokenizer_config.json"):
    p = out_dir / name
    print(f"  - {name}: {'yes' if p.exists() else 'no'}")
PY

TOK_JSON="$TOKENIZER_DIR/tokenizer.json"
if [ ! -f "$TOK_JSON" ]; then
  echo "[ERROR] Missing $TOK_JSON after export"
  exit 1
fi

echo "[2/4] Build train.jsonl from Strict *.train.txt (one JSON string per line)"
export TRAIN_JSONL="$OUT_DIR/train_strict.jsonl"
export STRICT_RAW OUT_DIR
python - <<'PY'
import json
import os
from pathlib import Path

strict = Path(os.environ["STRICT_RAW"])
out = Path(os.environ["TRAIN_JSONL"])
files = sorted(strict.glob("*.train.txt"))
if not files:
    raise SystemExit(f"No *.train.txt in {strict}")
n = 0
with out.open("w", encoding="utf-8") as wf:
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                wf.write(json.dumps(line, ensure_ascii=False) + "\n")
                n += 1
print(f"[OK] wrote {out} lines={n} from {len(files)} files")
PY

echo "[3/4] Build valid.jsonl (use *.dev.txt if present, else 1% of train)"
export VALID_JSONL="$OUT_DIR/valid_strict.jsonl"
export TRAIN_JSONL STRICT_RAW
python - <<'PY'
import json
import os
import random
from pathlib import Path

strict = Path(os.environ["STRICT_RAW"])
train_jsonl = Path(os.environ["TRAIN_JSONL"])
valid_jsonl = Path(os.environ["VALID_JSONL"])
dev_files = sorted(strict.glob("*.dev.txt"))
lines = []
if dev_files:
    for fp in dev_files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
    print(f"[INFO] valid from dev files: {len(lines)} lines")
else:
    random.seed(42)
    with train_jsonl.open("r", encoding="utf-8") as f:
        pool = [json.loads(x) for x in f if x.strip()]
    k = max(1000, len(pool) // 100)
    lines = random.sample(pool, min(k, len(pool)))
    print(f"[INFO] no *.dev.txt, sample valid from train: {len(lines)} lines")

with valid_jsonl.open("w", encoding="utf-8") as wf:
    for line in lines:
        wf.write(json.dumps(line, ensure_ascii=False) + "\n")
print(f"[OK] wrote {valid_jsonl}")
PY

echo "[4/4] Run official tokenize_corpus.py"
cd "$GPT_BERT_ROOT/corpus_tokenization"
python tokenize_corpus.py \
  --data_folder="$OUT_DIR" \
  --train_file="train_strict.jsonl" \
  --valid_file="valid_strict.jsonl" \
  --tokenizer_folder="$TOKENIZER_DIR" \
  --tokenizer_file="tokenizer.json" \
  --name="strict"

TRAIN_BIN="$OUT_DIR/train_strict_strict_tokenized.bin"
VALID_BIN="$OUT_DIR/valid_strict_strict_tokenized.bin"
if [ ! -f "$TRAIN_BIN" ]; then
  TRAIN_BIN="$(ls -1 "$OUT_DIR"/*train*tokenized*.bin 2>/dev/null | head -n 1)"
fi
if [ ! -f "$VALID_BIN" ]; then
  VALID_BIN="$(ls -1 "$OUT_DIR"/*valid*tokenized*.bin 2>/dev/null | head -n 1)"
fi

cat <<EOF

[OK] Tokenization done.
[NOTE] 若 tokenizer 来自 HF babylm-baseline-100m-gpt-bert-masked-focus（16k 词表），
       训练请用 configs/base.json，不要用 small.json（8192），否则 CUDA assert。

Export these before training:

export GPT_BERT_TRAIN_PATH="$TRAIN_BIN"
export GPT_BERT_VALID_PATH="$VALID_BIN"

Then:
  cd ~/babytext/experiments/gpt-bert-masked-focus/scripts
  bash 05_train_continue_masked_focus.sh

EOF

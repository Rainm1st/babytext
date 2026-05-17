#!/usr/bin/env python3
"""检查 .bin 中 token id 是否超出模型 config 的 vocab_size（CUDA assert 常见原因）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


def max_id_in_bin(path: Path) -> tuple[int, int]:
    docs = torch.load(path, map_location="cpu")
    if not docs:
        return 0, 0
    mx = 0
    mn = int(docs[0].min()) if len(docs[0]) else 0
    for doc in docs:
        if len(doc) == 0:
            continue
        mx = max(mx, int(doc.max()))
        mn = min(mn, int(doc.min()))
    return mn, mx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_bin", required=True)
    p.add_argument("--valid_bin", default=None)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()

    tok = Tokenizer.from_file(str(Path(args.tokenizer).expanduser()))
    cfg = json.loads(Path(args.config).expanduser().read_text())
    tok_vocab = tok.get_vocab_size()
    model_vocab = int(cfg["vocab_size"])

    for label, path in [("train", args.train_bin), ("valid", args.valid_bin or "")]:
        if not path:
            continue
        mn, mx = max_id_in_bin(Path(path))
        print(f"[{label}] {path}")
        print(f"  token id range: [{mn}, {mx}]")

    print(f"[tokenizer] vocab_size={tok_vocab}")
    print(f"[config]    vocab_size={model_vocab}")

    train_bin = Path(args.train_bin).expanduser()
    _, train_mx = max_id_in_bin(train_bin)
    ok_tok = train_mx < tok_vocab
    ok_cfg = train_mx < model_vocab
    print(f"[check] max_id < tokenizer vocab: {'OK' if ok_tok else 'FAIL'}")
    print(f"[check] max_id < config vocab:     {'OK' if ok_cfg else 'FAIL'}")

    if tok_vocab != model_vocab:
        print(
            f"[WARN] tokenizer ({tok_vocab}) != config ({model_vocab}). "
            "train_100m.py 会用 config 覆盖 setup 里的 vocab_size，必须以 config 为准。"
        )
    if not ok_cfg:
        print(
            "\n[FIX] 词表不匹配：若用 HF babylm-baseline-100m-gpt-bert-masked-focus 导出的 tokenizer，"
            "请改用 configs/base.json（vocab_size=16384），不要用 small.json（8192）。"
        )
        sys.exit(1)
    print("\n[OK] token ids fit model vocab.")


if __name__ == "__main__":
    main()

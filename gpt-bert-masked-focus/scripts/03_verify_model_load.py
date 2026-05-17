#!/usr/bin/env python3
"""验证 GPT-BERT Masked Focus 权重可加载。"""
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def resolved_model_ref() -> str:
    out = subprocess.check_output([sys.executable, str(SCRIPT_DIR / "resolve_hf_model_path.py")], text=True)
    return out.strip()


def main() -> None:
    model_ref = resolved_model_ref()
    print(f"[INFO] Resolved model ref: {model_ref}")

    print("[INFO] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[INFO] Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(model_ref, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    text = "The cat sat on the mat."
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    print(f"[OK] Forward pass OK. logits shape={tuple(out.logits.shape)}")
    print(f"[OK] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    main()

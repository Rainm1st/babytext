#!/usr/bin/env python3
"""当 WANDB_DISABLED=1 时，在 rank0 上 stub wandb（兼容空格/制表缩进）。"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "# patched: wandb stub when WANDB_DISABLED"


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[SKIP] wandb patch already applied: {path}")
        return

    pattern = r'if int\(os\.environ\["SLURM_PROCID"\]\) == 0:\s*\n\s*import wandb'
    repl = f'''if int(os.environ["SLURM_PROCID"]) == 0:
 if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
  import types
  wandb = types.SimpleNamespace(
   init=lambda *a, **k: None,
   log=lambda *a, **k: None,
   config=types.SimpleNamespace(update=lambda *a, **k: None),
  )  {MARKER}
 else:
  import wandb'''

    new_text, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        print(f"[WARN] wandb import pattern not found in {path}; skip (可仅用环境变量 WANDB_DISABLED)", file=sys.stderr)
        return

    path.write_text(new_text, encoding="utf-8")
    print(f"[OK] wandb stub patch: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_script",
        default="/home/language/babytext/experiments/gpt-bert/pretraining/train_100m.py",
    )
    args = p.parse_args()
    patch(Path(args.train_script).expanduser().resolve())


if __name__ == "__main__":
    main()

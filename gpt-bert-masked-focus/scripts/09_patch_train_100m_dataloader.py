#!/usr/bin/env python3
"""修复 train_100m.py：iter(dataloader) 后 wandb 仍访问 .dataset 导致 AttributeError。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# patched: keep dataset ref before iter()"


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[SKIP] already patched: {path}")
        return

    old = "    train_dataloader = iter(train_dataloader)\n    total_loss"
    new = (
        f"    train_dataset = train_dataloader.dataset  {MARKER}\n"
        "    train_dataloader = iter(train_dataloader)\n"
        "    total_loss"
    )
    if old not in text:
        print(f"[ERROR] pattern not found in {path}", file=sys.stderr)
        sys.exit(1)

    text = text.replace(old, new, 1)
    text = text.replace(
        '"stats/seq_length": train_dataloader.dataset.seq_length,',
        '"stats/seq_length": train_dataset.seq_length,',
        1,
    )
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched {path}")


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

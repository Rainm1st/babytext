#!/usr/bin/env python3
"""解析 GPT-BERT 模型路径：本地 snapshot / local-dir / Hub 模型 ID。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HUB_MODEL_ID = "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
DEFAULT_LOCAL_DIRS = (
    Path("/home/language/babytext/experiments/gpt-bert-masked-focus-baseline"),
    Path(
        "/home/language/babytext/experiments/"
        "models--BabyLM-community--babylm-baseline-100m-gpt-bert-masked-focus"
    ),
)


def resolve_model_ref(path_or_id: str | None = None) -> str:
    """返回 transformers.from_pretrained 可用的路径或 Hub ID。"""
    if path_or_id and "/" in path_or_id and not path_or_id.startswith("/"):
        return path_or_id

    candidates: list[Path] = []
    if path_or_id:
        candidates.append(Path(path_or_id))
    candidates.extend(DEFAULT_LOCAL_DIRS)

    for base in candidates:
        if not base.is_dir():
            continue
        if (base / "config.json").is_file():
            return str(base.resolve())
        snap_root = base / "snapshots"
        if snap_root.is_dir():
            snaps = sorted(
                (p for p in snap_root.iterdir() if p.is_dir() and (p / "config.json").is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if snaps:
                return str(snaps[0].resolve())

    return HUB_MODEL_ID


def main() -> None:
    ref = resolve_model_ref(os.environ.get("HF_MODEL") or os.environ.get("GPT_BERT_MODEL_DIR"))
    print(ref)


if __name__ == "__main__":
    main()

import json
import re
from pathlib import Path


TASKS = ("mnli", "rte", "wsc", "qqp")
GLUE_DIR = Path("/home/language/babytext/experiments/babylm-eval/strict/evaluation_data/full_eval/glue_filtered")
OUT_DIR = Path("/data0/language/babylm_runs/mask10_run4/data")
OUT_FILE = OUT_DIR / "task_focus_mnli_rte_wsc_qqp.train.txt"
STATS_FILE = OUT_DIR / "task_focus_stats.json"


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    return s


def extract_fields(record: dict) -> list[str]:
    fields = []
    for k in ("premise", "hypothesis", "sentence1", "sentence2", "question", "passage", "text", "context", "sentence"):
        v = record.get(k)
        if isinstance(v, str):
            v = normalize_text(v)
            if v:
                fields.append(v)
    return fields


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_task_samples: dict[str, int] = {}
    per_task_tokens: dict[str, int] = {}
    total_samples = 0
    total_tokens = 0

    with OUT_FILE.open("w", encoding="utf-8") as wf:
        for task in TASKS:
            src = GLUE_DIR / f"{task}.train.jsonl"
            if not src.exists():
                raise FileNotFoundError(f"Missing source file: {src}")

            sample_n = 0
            token_n = 0
            with src.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    fields = extract_fields(record)
                    if not fields:
                        continue
                    text = " ".join(fields)
                    wf.write(text + "\n")
                    sample_n += 1
                    token_n += token_count(text)

            per_task_samples[task] = sample_n
            per_task_tokens[task] = token_n
            total_samples += sample_n
            total_tokens += token_n

    STATS_FILE.write_text(
        json.dumps(
            {
                "tasks": list(TASKS),
                "source_dir": str(GLUE_DIR),
                "output_file": str(OUT_FILE),
                "per_task_samples": per_task_samples,
                "per_task_tokens": per_task_tokens,
                "total_samples": total_samples,
                "total_tokens": total_tokens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Wrote dataset: {OUT_FILE}")
    print(f"[OK] Wrote stats:   {STATS_FILE}")
    print(f"[SUMMARY] samples={total_samples}, tokens={total_tokens}")


if __name__ == "__main__":
    main()

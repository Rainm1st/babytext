import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_GLUE_DIR = Path(
    "/home/language/babytext/experiments/babylm-eval/strict/evaluation_data/full_eval/glue_filtered"
)
DEFAULT_RAW_JSONL = Path("/data0/language/babylm_runs/mask10_run5/data_raw/teacher_merged_raw.jsonl")
DEFAULT_OUT_DIR = Path("/data0/language/babylm_runs/mask10_run5/data")

TASKS = ("mnli", "rte", "wsc", "qqp")
LABELS: Dict[str, Dict[int, str]] = {
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "wsc": {0: "not_coreferent", 1: "coreferent"},
}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def extract_fields(record: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    for k in (
        "premise",
        "hypothesis",
        "sentence1",
        "sentence2",
        "question",
        "question1",
        "question2",
        "passage",
        "text",
        "context",
        "sentence",
        "span1_text",
        "span2_text",
    ):
        v = record.get(k)
        if isinstance(v, str):
            v = normalize_text(v)
            if v:
                fields.append(v)
    return fields


def build_original_task_texts(glue_dir: Path, tasks: Tuple[str, ...]) -> Tuple[List[str], Dict[str, int]]:
    original_texts: List[str] = []
    per_task_counts: Dict[str, int] = {}
    for task in tasks:
        src = glue_dir / f"{task}.train.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")
        task_count = 0
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fields = extract_fields(rec)
                if not fields:
                    continue
                text = normalize_text(" ".join(fields))
                if text:
                    original_texts.append(text)
                    task_count += 1
        per_task_counts[task] = task_count
    return original_texts, per_task_counts


def get_parsed_label(parsed: Dict[str, Any]) -> str:
    label = parsed.get("label", "")
    if not isinstance(label, str):
        return ""
    return normalize_text(label).lower()


def get_parsed_text(parsed: Dict[str, Any]) -> str:
    text = parsed.get("synthetic_text", "")
    if not isinstance(text, str):
        return ""
    return normalize_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue-dir", default=str(DEFAULT_GLUE_DIR))
    parser.add_argument("--raw-jsonl", default=str(DEFAULT_RAW_JSONL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tasks", default="mnli,rte,wsc,qqp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--orig-ratio", type=float, default=0.7, help="Ratio of original samples in mixed data.")
    parser.add_argument("--min-text-len", type=int, default=20)
    parser.add_argument("--max-text-len", type=int, default=1200)
    args = parser.parse_args()

    if not (0.0 < args.orig_ratio < 1.0):
        raise ValueError("--orig-ratio must be in (0,1)")

    glue_dir = Path(args.glue_dir)
    raw_jsonl = Path(args.raw_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    for t in tasks:
        if t not in TASKS:
            raise ValueError(f"Unsupported task in --tasks: {t}")

    original_texts, original_per_task = build_original_task_texts(glue_dir, tasks)
    existing_set = {normalize_text(x).lower() for x in original_texts}

    accepted: List[Dict[str, Any]] = []
    rejected = {
        "missing_teacher_output": 0,
        "parse_error": 0,
        "label_mismatch": 0,
        "teacher_disagree": 0,
        "empty_or_bad_text": 0,
        "duplicate": 0,
    }

    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task = row["task"]
            source_label_text = normalize_text(row["source_label_text"]).lower()
            outputs = row.get("teacher_outputs", {})
            out_a = outputs.get("model_a")
            out_b = outputs.get("model_b")
            if out_a is None or out_b is None:
                rejected["missing_teacher_output"] += 1
                continue
            if not out_a.get("parsed_ok") or not out_b.get("parsed_ok"):
                rejected["parse_error"] += 1
                continue
            pa = out_a["parsed"]
            pb = out_b["parsed"]
            la = get_parsed_label(pa)
            lb = get_parsed_label(pb)
            if la != source_label_text or lb != source_label_text:
                rejected["label_mismatch"] += 1
                continue
            ta = get_parsed_text(pa)
            tb = get_parsed_text(pb)
            if not ta or not tb:
                rejected["empty_or_bad_text"] += 1
                continue
            # Conservative consistency rule: both teachers must produce the same normalized text.
            if ta.lower() != tb.lower():
                rejected["teacher_disagree"] += 1
                continue
            if len(ta) < args.min_text_len or len(ta) > args.max_text_len:
                rejected["empty_or_bad_text"] += 1
                continue
            key = ta.lower()
            if key in existing_set:
                rejected["duplicate"] += 1
                continue
            existing_set.add(key)
            accepted.append(
                {
                    "task": task,
                    "sample_id": row["sample_id"],
                    "variant_id": row["variant_id"],
                    "text": ta,
                    "label": source_label_text,
                    "teacher_a": out_a["model_name"],
                    "teacher_b": out_b["model_name"],
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(accepted)

    # Mix policy: keep all originals, then cap synthetic count by ratio target.
    n_orig = len(original_texts)
    max_syn = int((n_orig * (1.0 - args.orig_ratio)) / args.orig_ratio)
    selected_syn = accepted[: max(0, max_syn)]

    mixed_texts = list(original_texts) + [x["text"] for x in selected_syn]
    rng.shuffle(mixed_texts)

    out_file = out_dir / "task_focus_teacher_mix_run5.train.txt"
    with out_file.open("w", encoding="utf-8") as wf:
        for text in mixed_texts:
            wf.write(text + "\n")

    filtered_jsonl = out_dir / "teacher_aug_filtered.jsonl"
    with filtered_jsonl.open("w", encoding="utf-8") as wf:
        for row in selected_syn:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_task_syn: Dict[str, int] = {t: 0 for t in tasks}
    for r in selected_syn:
        per_task_syn[r["task"]] += 1

    stats = {
        "tasks": list(tasks),
        "glue_dir": str(glue_dir),
        "raw_jsonl": str(raw_jsonl),
        "output_train_txt": str(out_file),
        "output_filtered_jsonl": str(filtered_jsonl),
        "original_per_task": original_per_task,
        "original_total": n_orig,
        "accepted_synthetic_total_before_ratio_cap": len(accepted),
        "selected_synthetic_total": len(selected_syn),
        "selected_synthetic_per_task": per_task_syn,
        "mixed_total": len(mixed_texts),
        "orig_ratio_target": args.orig_ratio,
        "synthetic_ratio_actual": (len(selected_syn) / len(mixed_texts)) if mixed_texts else 0.0,
        "rejected_counts": rejected,
    }
    stats_path = out_dir / "run5_mix_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote mixed train txt: {out_file}")
    print(f"[OK] wrote filtered jsonl: {filtered_jsonl}")
    print(f"[OK] wrote stats: {stats_path}")
    print(
        f"[SUMMARY] original={n_orig}, selected_syn={len(selected_syn)}, mixed={len(mixed_texts)}, "
        f"syn_ratio={stats['synthetic_ratio_actual']:.4f}"
    )


if __name__ == "__main__":
    main()

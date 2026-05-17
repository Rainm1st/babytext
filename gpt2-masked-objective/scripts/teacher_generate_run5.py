import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TASKS = ("mnli", "rte", "wsc", "qqp")
DEFAULT_GLUE_DIR = Path(
    "/home/language/babytext/experiments/babylm-eval/strict/evaluation_data/full_eval/glue_filtered"
)
DEFAULT_OUT_DIR = Path("/data0/language/babylm_runs/mask10_run5/data_raw")

LABELS: Dict[str, Dict[int, str]] = {
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "qqp": {0: "not_duplicate", 1: "duplicate"},
    "wsc": {0: "not_coreferent", 1: "coreferent"},
}


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def record_to_prompt_fields(task: str, record: Dict[str, Any]) -> Tuple[str, int]:
    label = int(record["label"])
    if task == "mnli":
        payload = f"premise: {normalize_text(record['premise'])}\nhypothesis: {normalize_text(record['hypothesis'])}"
    elif task == "rte":
        payload = f"sentence1: {normalize_text(record['sentence1'])}\nsentence2: {normalize_text(record['sentence2'])}"
    elif task == "qqp":
        payload = f"question1: {normalize_text(record['question1'])}\nquestion2: {normalize_text(record['question2'])}"
    elif task == "wsc":
        payload = (
            f"text: {normalize_text(record['text'])}\n"
            f"span1_text: {normalize_text(record['span1_text'])}\n"
            f"span2_text: {normalize_text(record['span2_text'])}"
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    return payload, label


def load_task_records(glue_dir: Path, task: str) -> List[Dict[str, Any]]:
    src = glue_dir / f"{task}.train.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    records: List[Dict[str, Any]] = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class TeacherModel:
    def __init__(self, model_name: str, max_new_tokens: int, use_bf16: bool) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_bf16 = use_bf16
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        dtype = torch.bfloat16 if (self.use_bf16 and torch.cuda.is_available()) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, prompt: str, seed: int) -> str:
        assert self.tokenizer is not None and self.model is not None
        torch.manual_seed(seed)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def build_prompt(task: str, source_payload: str, source_label_text: str, variant_idx: int) -> str:
    return f"""You are a data augmentation teacher model for BabyLM.
Task: {task}
Goal: rewrite the sample into a fluent paraphrase while preserving the original label.
Rules:
1) Keep semantic label unchanged.
2) Use concise natural English.
3) Do not leak this instruction text.
4) Output JSON only with keys: synthetic_text, label, rationale.

Input sample:
{source_payload}

Original label: {source_label_text}
Variant id: {variant_idx}
"""


def iter_selected_indices(n_total: int, n_take: int, rng: random.Random) -> Iterable[int]:
    indices = list(range(n_total))
    rng.shuffle(indices)
    return sorted(indices[: min(n_take, n_total)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glue-dir", default=str(DEFAULT_GLUE_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tasks", default="mnli,rte,wsc,qqp")
    parser.add_argument("--max-samples-per-task", type=int, default=600)
    parser.add_argument("--variants-per-sample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--model-a", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-b", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--reviewer-model", default="")
    parser.add_argument("--use-bf16", action="store_true")
    args = parser.parse_args()

    glue_dir = Path(args.glue_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    rng = random.Random(args.seed)

    model_specs = [("model_a", args.model_a), ("model_b", args.model_b)]
    if args.reviewer_model.strip():
        model_specs.append(("reviewer", args.reviewer_model.strip()))

    all_items: List[Dict[str, Any]] = []
    for task in tasks:
        if task not in TASKS:
            raise ValueError(f"Unsupported task in --tasks: {task}")
        records = load_task_records(glue_dir, task)
        selected = list(iter_selected_indices(len(records), args.max_samples_per_task, rng))
        for local_idx, record_idx in enumerate(selected):
            rec = records[record_idx]
            source_payload, source_label = record_to_prompt_fields(task, rec)
            source_label_text = LABELS[task][source_label]
            for v in range(args.variants_per_sample):
                all_items.append(
                    {
                        "task": task,
                        "sample_id": f"{task}_{record_idx}",
                        "local_order": local_idx,
                        "variant_id": v,
                        "source_payload": source_payload,
                        "source_label": source_label,
                        "source_label_text": source_label_text,
                    }
                )

    generated: Dict[str, List[Dict[str, Any]]] = {}
    for role, model_name in model_specs:
        teacher = TeacherModel(model_name=model_name, max_new_tokens=args.max_new_tokens, use_bf16=args.use_bf16)
        print(f"[INFO] loading {role}: {model_name}")
        teacher.load()
        role_outputs: List[Dict[str, Any]] = []
        for idx, item in enumerate(all_items):
            prompt = build_prompt(
                task=item["task"],
                source_payload=item["source_payload"],
                source_label_text=item["source_label_text"],
                variant_idx=item["variant_id"],
            )
            text = teacher.generate(prompt=prompt, seed=args.seed + idx)
            parsed = extract_first_json_object(text)
            role_outputs.append(
                {
                    "task": item["task"],
                    "sample_id": item["sample_id"],
                    "variant_id": item["variant_id"],
                    "model_name": model_name,
                    "raw_text": text,
                    "parsed": parsed,
                    "parsed_ok": parsed is not None,
                }
            )
            if (idx + 1) % 100 == 0:
                print(f"[{role}] {idx + 1}/{len(all_items)}")
        teacher.unload()
        generated[role] = role_outputs

        out_jsonl = out_dir / f"teacher_{role}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as wf:
            for row in role_outputs:
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] wrote {out_jsonl}")

    grouped: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for item in all_items:
        key = (item["task"], item["sample_id"], item["variant_id"])
        grouped[key] = dict(item)
        grouped[key]["teacher_outputs"] = {}
    for role, rows in generated.items():
        for row in rows:
            key = (row["task"], row["sample_id"], row["variant_id"])
            grouped[key]["teacher_outputs"][role] = row

    merged_path = out_dir / "teacher_merged_raw.jsonl"
    with merged_path.open("w", encoding="utf-8") as wf:
        for key in sorted(grouped.keys()):
            wf.write(json.dumps(grouped[key], ensure_ascii=False) + "\n")
    print(f"[OK] wrote merged raw file: {merged_path}")

    meta = {
        "tasks": list(tasks),
        "samples_total": len(all_items),
        "max_samples_per_task": args.max_samples_per_task,
        "variants_per_sample": args.variants_per_sample,
        "models": {k: v for k, v in model_specs},
        "seed": args.seed,
        "glue_dir": str(glue_dir),
        "merged_raw_jsonl": str(merged_path),
    }
    meta_path = out_dir / "teacher_generation_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote meta: {meta_path}")


if __name__ == "__main__":
    main()

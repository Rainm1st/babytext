import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


BASELINE_OFFICIAL = {
    "model_name": "gpt2_base_official",
    "BLiMP": 74.88,
    "WUGs": 35.5,
    "Entity_Tracking": 31.51,
    "MNLI": 59.09,
    "MRPC": None,
}

BLIMP_TASKS = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "regular_plural_subject_verb_agreement_1",
    "wh_questions_object_gap",
    "determiner_noun_agreement_1",
    "irregular_plural_subject_verb_agreement_1",
    "passive_1",
    "transitive",
    "principle_A_domain_1",
]


def sentence_logprob_causal(model, tokenizer, text: str, max_length: int = 256) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


def eval_blimp(model, tokenizer) -> Optional[float]:
    scores = []
    for task in BLIMP_TASKS:
        ds = load_dataset("nyu-mll/blimp", task, split="train")
        correct = 0
        total = 0
        for ex in ds:
            g = ex["sentence_good"]
            b = ex["sentence_bad"]
            if sentence_logprob_causal(model, tokenizer, g) > sentence_logprob_causal(model, tokenizer, b):
                correct += 1
            total += 1
        if total > 0:
            scores.append(correct / total)
    if not scores:
        return None
    return round(100.0 * (sum(scores) / len(scores)), 2)


def eval_boolq(model, tokenizer, max_samples: int = -1) -> Optional[float]:
    ds = load_dataset("super_glue", "boolq", split="validation")
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    correct, total = 0, 0
    for ex in ds:
        p = ex["passage"]
        q = ex["question"]
        s_yes = f"{p}\nQuestion: {q}\nAnswer: Yes"
        s_no = f"{p}\nQuestion: {q}\nAnswer: No"
        pred = 1 if sentence_logprob_causal(model, tokenizer, s_yes) > sentence_logprob_causal(model, tokenizer, s_no) else 0
        if pred == int(ex["label"]):
            correct += 1
        total += 1
    if total == 0:
        return None
    return round(100.0 * correct / total, 2)


def eval_glue_nli_or_pair(model, tokenizer, task: str, max_samples: int = -1) -> Optional[float]:
    split = "validation_matched" if task == "mnli" else "validation"
    ds = load_dataset("glue", task, split=split)
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct, total = 0, 0
    if task == "mnli":
        labels = ["entailment", "neutral", "contradiction"]
        for ex in ds:
            p, h, y = ex["premise"], ex["hypothesis"], int(ex["label"])
            opts = [f"Premise: {p}\nHypothesis: {h}\nRelation: {lab}" for lab in labels]
            scores = [sentence_logprob_causal(model, tokenizer, o) for o in opts]
            pred = int(max(range(3), key=lambda i: scores[i]))
            if pred == y:
                correct += 1
            total += 1
    elif task == "mrpc":
        for ex in ds:
            s1, s2, y = ex["sentence1"], ex["sentence2"], int(ex["label"])
            yes = f"Sentence1: {s1}\nSentence2: {s2}\nParaphrase: Yes"
            no = f"Sentence1: {s1}\nSentence2: {s2}\nParaphrase: No"
            pred = 1 if sentence_logprob_causal(model, tokenizer, yes) > sentence_logprob_causal(model, tokenizer, no) else 0
            if pred == y:
                correct += 1
            total += 1
    else:
        return None

    if total == 0:
        return None
    return round(100.0 * correct / total, 2)


def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fmt(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask5-summary", required=True)
    parser.add_argument("--mask10-summary", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--delta-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=-1, help="debug mode: limit eval samples")
    args = parser.parse_args()

    summaries = {
        "gpt2_mask5_run1": json.loads(Path(args.mask5_summary).read_text(encoding="utf-8")),
        "gpt2_mask10_run1": json.loads(Path(args.mask10_summary).read_text(encoding="utf-8")),
    }

    eval_out: Dict[str, Any] = {"baseline_official": BASELINE_OFFICIAL, "models": {}}

    for model_name, summary in summaries.items():
        model_dir = Path(summary["final_model_dir"])
        model, tokenizer = load_model(model_dir)
        blimp = eval_blimp(model, tokenizer)
        boolq = eval_boolq(model, tokenizer, max_samples=args.max_samples)
        mnli = eval_glue_nli_or_pair(model, tokenizer, "mnli", max_samples=args.max_samples)
        mrpc = eval_glue_nli_or_pair(model, tokenizer, "mrpc", max_samples=args.max_samples)
        eval_out["models"][model_name] = {
            "BLiMP": blimp,
            "BoolQ": boolq,
            "MNLI": mnli,
            "MRPC": mrpc,
            "WUGs": None,
            "Entity_Tracking": None,
        }
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Path(args.output_json).write_text(json.dumps(eval_out, ensure_ascii=False, indent=2), encoding="utf-8")

    raw_path = Path(args.raw_csv)
    raw_rows = read_csv_rows(raw_path)
    raw_fields = list(raw_rows[0].keys()) if raw_rows else [
        "model_name", "overall_score", "zero_shot_avg", "fine_tune_avg", "BLiMP", "BLiMP_Supplement",
        "EWoK", "Eye_Tracking", "Self_Paced_Reading", "Entity_Tracking", "WUGs", "BoolQ", "MNLI", "MRPC",
        "QQP", "MultiRC", "RTE", "WSC", "checkpoint_used", "notes"
    ]

    for row in raw_rows:
        name = row["model_name"]
        if name == "gpt2_base_official":
            row["BLiMP"] = fmt(BASELINE_OFFICIAL["BLiMP"])
            row["WUGs"] = fmt(BASELINE_OFFICIAL["WUGs"])
            row["Entity_Tracking"] = fmt(BASELINE_OFFICIAL["Entity_Tracking"])
            row["MNLI"] = fmt(BASELINE_OFFICIAL["MNLI"])
            row["MRPC"] = fmt(BASELINE_OFFICIAL["MRPC"])
            row["notes"] = "official baseline reference"
        elif name in eval_out["models"]:
            m = eval_out["models"][name]
            row["BLiMP"] = fmt(m["BLiMP"])
            row["BoolQ"] = fmt(m["BoolQ"])
            row["MNLI"] = fmt(m["MNLI"])
            row["MRPC"] = fmt(m["MRPC"])
            row["checkpoint_used"] = summaries[name]["final_model_dir"]
            row["notes"] = "auto-evaluated by evaluate_and_fill_tables.py"

    write_csv_rows(raw_path, raw_rows, raw_fields)

    delta_path = Path(args.delta_csv)
    delta_rows = read_csv_rows(delta_path)
    delta_fields = list(delta_rows[0].keys())

    baseline_map = {
        "BLiMP": BASELINE_OFFICIAL["BLiMP"],
        "WUGs": BASELINE_OFFICIAL["WUGs"],
        "Entity_Tracking": BASELINE_OFFICIAL["Entity_Tracking"],
        "MNLI": BASELINE_OFFICIAL["MNLI"],
        "MRPC": BASELINE_OFFICIAL["MRPC"],
    }

    for row in delta_rows:
        task = row["task"]
        if task not in baseline_map:
            continue
        b = baseline_map[task]
        m5 = eval_out["models"]["gpt2_mask5_run1"].get(task)
        m10 = eval_out["models"]["gpt2_mask10_run1"].get(task)

        row["baseline"] = fmt(b)
        row["gpt2_mask5_run1"] = fmt(m5)
        row["gpt2_mask10_run1"] = fmt(m10)
        row["delta_mask5"] = fmt(None if (b is None or m5 is None) else (m5 - b))
        row["delta_mask10"] = fmt(None if (b is None or m10 is None) else (m10 - b))

        if m5 is None and m10 is None:
            row["better_model"] = ""
        elif m10 is None or (m5 is not None and m5 > m10):
            row["better_model"] = "gpt2_mask5_run1"
        elif m5 is None or (m10 > m5):
            row["better_model"] = "gpt2_mask10_run1"
        else:
            row["better_model"] = "tie"

    write_csv_rows(delta_path, delta_rows, delta_fields)
    print(f"Wrote evaluation json: {args.output_json}")
    print(f"Updated raw table: {args.raw_csv}")
    print(f"Updated delta table: {args.delta_csv}")


if __name__ == "__main__":
    main()

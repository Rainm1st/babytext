import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_MODEL_ID = "BabyLM-community/babylm-baseline-100m-gpt2"
WEIGHT_CANDIDATES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
DATA_FILE_PATTERN = re.compile(r".*\.train\.txt$", re.IGNORECASE)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_weight_files(snapshot_dir: Path) -> List[Path]:
    return [p for p in snapshot_dir.rglob("*") if p.name in WEIGHT_CANDIDATES]


def ensure_weights(snapshot_dir: Path, cache_root: Path, auto_download: bool) -> Tuple[bool, str]:
    weight_files = find_weight_files(snapshot_dir)
    if weight_files:
        rel = ", ".join(str(p.relative_to(snapshot_dir)) for p in weight_files[:3])
        return True, f"Found model weights in snapshot: {rel}"

    if not auto_download:
        return False, "No model weights found in snapshot and auto-download disabled."

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        return False, f"Missing huggingface_hub for auto-download: {e}"

    snapshot_download(
        repo_id=REPO_MODEL_ID,
        cache_dir=str(cache_root),
        local_files_only=False,
    )

    weight_files = find_weight_files(snapshot_dir)
    if not weight_files:
        return False, "Download finished but weight files are still missing in expected snapshot path."
    rel = ", ".join(str(p.relative_to(snapshot_dir)) for p in weight_files[:3])
    return True, f"Downloaded and found model weights: {rel}"


def validate_shared_fields(configs: Dict[str, Dict], compare_with_baseline: bool) -> List[str]:
    errors: List[str] = []
    warnings: List[str] = []

    shared_keys = [
        "starting_checkpoint",
        "tokenizer_path",
        "training_data_path",
        "data_filtering_version",
        "max_sequence_length",
        "batch_size_per_device",
        "gradient_accumulation_steps",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "scheduler",
        "warmup_ratio",
        "gradient_clipping",
        "mixed_precision",
        "seed",
        "training_budget_mode",
        "total_training_tokens",
        "total_training_steps",
        "num_train_epochs",
        "max_words_seen",
        "checkpoint_steps",
        "checkpoint_save_every_steps",
        "keep_last_step_checkpoints",
        "cuda_visible_devices",
        "source_proportional_batches",
        "source_word_counts",
        "final_checkpoint_selection_rule",
    ]

    if compare_with_baseline:
        names = list(configs.keys())
    else:
        names = [name for name in configs.keys() if name != "run_base_official"]

    base = configs[names[0]]

    for key in shared_keys:
        val = base.get(key)
        for name in names[1:]:
            if configs[name].get(key) != val:
                errors.append(f"Mismatch on shared field '{key}': {names[0]} != {name}")

    expected_alpha = {"run_mask5": 0.05, "run_mask10": 0.10, "run_mask20": 0.20, "run_mask50": 0.50}
    if compare_with_baseline:
        expected_alpha["run_base_official"] = 0.0
    for name, alpha in expected_alpha.items():
        cfg = configs.get(name, {})
        if abs(float(cfg.get("masked_alpha", -999)) - alpha) > 1e-9:
            errors.append(f"{name} masked_alpha expected {alpha}, got {cfg.get('masked_alpha')}")
        if abs(float(cfg.get("masked_ratio_sampling", -999)) - alpha) > 1e-9:
            errors.append(
                f"{name} masked_ratio_sampling expected {alpha}, got {cfg.get('masked_ratio_sampling')}"
            )

    for name, cfg in configs.items():
        if cfg.get("final_checkpoint_selection_rule") != "last_checkpoint":
            warnings.append(f"{name}: final_checkpoint_selection_rule is not 'last_checkpoint'")

    return errors + warnings


def validate_dataset_dir(data_dir: Path) -> Tuple[bool, str]:
    if not data_dir.exists():
        return False, f"Dataset path does not exist: {data_dir}"
    if not data_dir.is_dir():
        return False, f"Dataset path is not a directory: {data_dir}"

    train_files = [p for p in data_dir.iterdir() if p.is_file() and DATA_FILE_PATTERN.match(p.name)]
    if not train_files:
        return False, f"No *.train.txt files found in {data_dir}"
    return True, f"Found {len(train_files)} training text files."


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and validate GPT2 masked-objective experiment.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to experiments/gpt2-masked-objective directory",
    )
    parser.add_argument(
        "--auto-download-model",
        action="store_true",
        help="Automatically download missing model weights from Hugging Face.",
    )
    parser.add_argument(
        "--no-baseline-compare",
        action="store_true",
        help="Only enforce consistency between run_mask5 and run_mask10.",
    )
    args = parser.parse_args()

    exp_dir = args.experiment_dir.resolve()
    configs_dir = exp_dir / "configs"
    config_files = {
        "run_base_official": configs_dir / "run_base_official.json",
        "run_mask5": configs_dir / "run_mask5.json",
        "run_mask10": configs_dir / "run_mask10.json",
        "run_mask20": configs_dir / "run_mask20.json",
        "run_mask50": configs_dir / "run_mask50.json",
    }

    for name, p in config_files.items():
        if not p.exists():
            print(f"[ERROR] Missing config: {name} -> {p}")
            return 1

    configs = {name: load_json(path) for name, path in config_files.items()}

    print("[CHECK] Config consistency")
    issues = validate_shared_fields(configs, compare_with_baseline=not args.no_baseline_compare)
    has_error = False
    for msg in issues:
        prefix = "[ERROR]" if "Mismatch" in msg or "expected" in msg else "[WARN]"
        if prefix == "[ERROR]":
            has_error = True
        print(f"{prefix} {msg}")
    if not issues:
        print("[OK] Shared fields and alpha setup are consistent.")

    base_cfg = configs["run_base_official"]
    data_dir = Path(base_cfg["training_data_path"])
    print("[CHECK] Dataset path")
    ok_data, data_msg = validate_dataset_dir(data_dir)
    print(f"{'[OK]' if ok_data else '[ERROR]'} {data_msg}")
    has_error = has_error or (not ok_data)

    snapshot_dir = Path(base_cfg["starting_checkpoint"])
    cache_root = snapshot_dir.parents[2] if len(snapshot_dir.parents) >= 3 else snapshot_dir.parent
    print("[CHECK] Baseline model checkpoint")
    ok_model, model_msg = ensure_weights(snapshot_dir, cache_root, args.auto_download_model)
    print(f"{'[OK]' if ok_model else '[ERROR]'} {model_msg}")
    has_error = has_error or (not ok_model)

    print("[CHECK] Selection rule")
    if base_cfg.get("final_checkpoint_selection_rule") == "last_checkpoint":
        print("[OK] Final checkpoint rule is set to last_checkpoint.")
    else:
        print("[WARN] Final checkpoint rule is not last_checkpoint.")

    if has_error:
        print("\nPreparation failed. Fix errors before training.")
        return 1

    print("\nPreparation passed. You can start training safely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

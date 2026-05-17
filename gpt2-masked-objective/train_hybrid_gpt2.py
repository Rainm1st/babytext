import argparse
import importlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISUAL_BACKENDS", "1")

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_cfg(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in config and config[k] is not None:
            return config[k]
    return default


def list_train_files(data_dir: Path) -> List[str]:
    files = sorted(str(p) for p in data_dir.glob("*.train.txt") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No *.train.txt files found under {data_dir}")
    return files


def ensure_tokens(tokenizer, model) -> None:
    resized = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        resized = True
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        resized = True
    if resized:
        model.resize_token_embeddings(len(tokenizer))


def group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    return {k: [v[i : i + block_size] for i in range(0, total_length, block_size)] for k, v in concatenated.items()}


def add_word_count(batch: Dict[str, List[List[int]]], tokenizer) -> Dict[str, List[int]]:
    word_counts: List[int] = []
    for ids in batch["input_ids"]:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        wc = len(text.strip().split())
        word_counts.append(max(1, wc))
    return {"word_count": word_counts}


def build_lm_dataset_by_source(text_files: List[str], tokenizer, block_size: int, num_workers: int):
    parts = []
    for text_file in text_files:
        source_name = Path(text_file).name
        raw_ds = load_dataset("text", data_files={"train": [text_file]}, split="train")
        tokenized = raw_ds.map(
            lambda e: tokenizer(e["text"]),
            batched=True,
            remove_columns=["text"],
            num_proc=max(1, num_workers),
            desc=f"Tokenizing {source_name}",
        )
        lm_part = tokenized.map(
            lambda e: group_texts(e, block_size),
            batched=True,
            num_proc=max(1, num_workers),
            desc=f"Grouping {source_name} ({block_size})",
        )
        lm_part = lm_part.map(
            lambda e: add_word_count(e, tokenizer),
            batched=True,
            num_proc=max(1, num_workers),
            desc=f"Estimating words {source_name}",
        )
        lm_part = lm_part.add_column("source_file", [source_name] * len(lm_part))
        parts.append(lm_part)

    if not parts:
        raise ValueError("No training dataset parts were built.")
    return concatenate_datasets(parts)


def normalize_source_weights(source_files: List[str], configured_counts: Dict[str, Any]) -> Dict[str, float]:
    if configured_counts:
        weights = {name: float(configured_counts.get(name, 0.0)) for name in source_files}
    else:
        weights = {name: 1.0 for name in source_files}
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        raise ValueError("Source sampling weights must contain at least one positive value.")
    return {name: max(0.0, weights[name]) / total for name in source_files}


class SourceProportionalBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        source_files: List[str],
        source_by_index: List[str],
        source_weights: Dict[str, float],
        batch_size: int,
        seed: int,
    ) -> None:
        self.source_files = source_files
        self.source_by_index = source_by_index
        self.source_weights = source_weights
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.indices_by_source: Dict[str, List[int]] = defaultdict(list)
        for idx, source in enumerate(source_by_index):
            self.indices_by_source[source].append(idx)

        missing = [name for name in source_files if not self.indices_by_source.get(name)]
        if missing:
            raise ValueError(f"No grouped training blocks found for source files: {missing}")

    def __len__(self) -> int:
        return (len(self.source_by_index) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = {name: list(self.indices_by_source[name]) for name in self.source_files}
        for values in pools.values():
            rng.shuffle(values)

        residual = {name: 0.0 for name in self.source_files}
        remaining = sum(len(v) for v in pools.values())

        while remaining > 0:
            batch: List[int] = []
            desired: Dict[str, int] = {}

            for name in self.source_files:
                residual[name] += self.source_weights[name] * self.batch_size
                count = int(residual[name])
                residual[name] -= count
                desired[name] = count

            while sum(desired.values()) < self.batch_size:
                candidates = [name for name in self.source_files if pools[name]]
                if not candidates:
                    break
                chosen = max(candidates, key=lambda name: residual[name])
                desired[chosen] += 1
                residual[chosen] = 0.0

            for name in self.source_files:
                take = min(desired[name], len(pools[name]), self.batch_size - len(batch))
                if take > 0:
                    batch.extend(pools[name][-take:])
                    del pools[name][-take:]

            while len(batch) < self.batch_size:
                candidates = [name for name in self.source_files if pools[name]]
                if not candidates:
                    break
                chosen = max(candidates, key=lambda name: len(pools[name]))
                batch.append(pools[chosen].pop())

            if not batch:
                break
            rng.shuffle(batch)
            remaining -= len(batch)
            yield batch


class HybridCollator:
    def __init__(
        self,
        tokenizer,
        masked_ratio: float,
        mask_token_prob: float,
        random_token_prob: float,
        keep_original_prob: float,
        masked_loss_positions_only: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.masked_ratio = float(masked_ratio)
        self.mask_token_prob = float(mask_token_prob)
        self.random_token_prob = float(random_token_prob)
        self.keep_original_prob = float(keep_original_prob)
        self.masked_loss_positions_only = bool(masked_loss_positions_only)

        s = self.mask_token_prob + self.random_token_prob + self.keep_original_prob
        if abs(s - 1.0) > 1e-6:
            raise ValueError("Mask replacement probabilities must sum to 1.0")
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no mask token.")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        word_count = torch.tensor([int(f["word_count"]) for f in features], dtype=torch.long)

        labels_causal = input_ids.clone()
        labels_masked = torch.full_like(input_ids, -100)
        corrupted = input_ids.clone()

        if self.masked_ratio > 0:
            special_masks = [
                self.tokenizer.get_special_tokens_mask(x.tolist(), already_has_special_tokens=True) for x in input_ids
            ]
            special_masks = torch.tensor(special_masks, dtype=torch.bool)
            candidate = ~special_masks

            select_rand = torch.rand(input_ids.shape)
            selected = (select_rand < self.masked_ratio) & candidate
            labels_masked[selected] = input_ids[selected]

            repl_rand = torch.rand(input_ids.shape)
            to_mask = selected & (repl_rand < self.mask_token_prob)
            to_random = selected & (
                (repl_rand >= self.mask_token_prob)
                & (repl_rand < self.mask_token_prob + self.random_token_prob)
            )

            corrupted[to_mask] = self.tokenizer.mask_token_id
            random_tokens = torch.randint(0, len(self.tokenizer), input_ids.shape, dtype=torch.long)
            corrupted[to_random] = random_tokens[to_random]

            if not self.masked_loss_positions_only:
                labels_masked = input_ids.clone()

        return {
            # Keep backward-compatible key `input_ids` as masked/corrupted inputs.
            "input_ids": corrupted,
            "input_ids_clean": input_ids,
            "input_ids_masked": corrupted,
            "attention_mask": attention_mask,
            "labels_causal": labels_causal,
            "labels_masked": labels_masked,
            "word_count": word_count,
        }


def build_optimizer(model, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name.lower() or "ln_" in name.lower() or "layernorm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    params = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(params, lr=lr)


def checkpoint_name_from_words(words_seen: int) -> str:
    return f"ckpt_words_{words_seen:010d}"


def epoch_checkpoint_name(epoch: int) -> str:
    return f"ckpt_epoch_{epoch:02d}"


def step_checkpoint_name(step: int) -> str:
    return f"ckpt_step_{step:08d}"


def epoch_number_from_checkpoint_name(name: str) -> Optional[int]:
    prefix = "ckpt_epoch_"
    if not name.startswith(prefix):
        return None
    try:
        return int(name[len(prefix) :])
    except ValueError:
        return None


def prune_old_step_checkpoints(checkpoints_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    step_dirs = sorted(
        [p for p in checkpoints_dir.glob("ckpt_step_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    stale = step_dirs[:-keep_last]
    for path in stale:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()
        print(f"Deleted old rolling step checkpoint: {path}")


def save_model_artifacts(model, tokenizer, save_dir: Path, export_pt: bool = True) -> Dict[str, str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    base_model = unwrap_model(model)
    base_model.save_pretrained(str(save_dir), safe_serialization=False)
    tokenizer.save_pretrained(str(save_dir))

    artifacts = {
        "dir": str(save_dir.resolve()),
        "hf_weights": str((save_dir / "pytorch_model.bin").resolve()),
    }
    if export_pt:
        pt_path = save_dir / "model.pt"
        torch.save(base_model.state_dict(), pt_path)
        artifacts["pt_state_dict"] = str(pt_path.resolve())
    return artifacts


def save_training_state(
    save_dir: Path,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    words_seen: int,
    running_loss: float,
    reached_checkpoint_words: set,
) -> str:
    train_state_path = save_dir / "training_state.pt"
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "words_seen": words_seen,
            "running_loss": running_loss,
            "reached_checkpoint_words": sorted(int(x) for x in reached_checkpoint_words),
        },
        train_state_path,
    )
    return str(train_state_path.resolve())


def load_best_epoch_loss(checkpoints_dir: Path) -> float:
    best_state = checkpoints_dir / "ckpt_best" / "trainer_state.json"
    if not best_state.exists():
        return float("inf")
    try:
        payload = load_json(best_state)
        metric = payload.get("epoch_metrics", {}).get("loss_total")
        return float(metric) if metric is not None else float("inf")
    except Exception:
        return float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid GPT2 training (BabyLM-style words budget).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="babylm-strict")
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="weichunzhou527-xi-an-jiaotong-liverpool-university",
    )
    args = parser.parse_args()

    config = load_json(Path(args.config))
    cuda_visible_devices = args.cuda_visible_devices or get_cfg(config, "cuda_visible_devices", default=None)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    seed = int(get_cfg(config, "seed", default=42))
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = get_cfg(config, "starting_checkpoint", "start_checkpoint")
    tokenizer_path = get_cfg(config, "tokenizer_path", "tokenizer_name_or_path", default=model_path)
    data_path = Path(get_cfg(config, "training_data_path", "raw_root"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    ensure_tokens(tokenizer, model)
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    model.train()

    text_files = list_train_files(data_path)
    block_size = int(get_cfg(config, "max_sequence_length", "max_length", default=256))
    lm_ds = build_lm_dataset_by_source(text_files, tokenizer, block_size, args.num_workers)

    mask_cfg = get_cfg(config, "mask_replacement", default={})
    collator = HybridCollator(
        tokenizer=tokenizer,
        masked_ratio=float(get_cfg(config, "masked_ratio_sampling", default=0.0)),
        mask_token_prob=float(mask_cfg.get("mask_token_prob", 0.8)),
        random_token_prob=float(mask_cfg.get("random_token_prob", 0.1)),
        keep_original_prob=float(mask_cfg.get("keep_original_prob", 0.1)),
        masked_loss_positions_only=bool(get_cfg(config, "masked_loss_positions_only", default=True)),
    )

    batch_size_per_device = int(get_cfg(config, "batch_size_per_device", "per_device_train_batch_size", default=4))
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    train_batch_size = batch_size_per_device * max(1, device_count)
    source_names = [Path(p).name for p in text_files]
    source_word_counts = get_cfg(config, "source_word_counts", default={}) or {}
    source_weights = normalize_source_weights(source_names, source_word_counts)
    use_source_proportional_batches = bool(get_cfg(config, "source_proportional_batches", default=False))
    batch_sampler = None
    if use_source_proportional_batches:
        batch_sampler = SourceProportionalBatchSampler(
            source_files=source_names,
            source_by_index=lm_ds["source_file"],
            source_weights=source_weights,
            batch_size=train_batch_size,
            seed=seed,
        )
        dataloader = DataLoader(
            lm_ds,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            num_workers=max(0, args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )
    else:
        dataloader = DataLoader(
            lm_ds,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=max(0, args.num_workers),
            pin_memory=torch.cuda.is_available(),
        )

    grad_acc = int(get_cfg(config, "gradient_accumulation_steps", default=1))
    target_alpha = float(get_cfg(config, "masked_alpha", default=0.0))
    alpha_ramp_cfg = get_cfg(config, "masked_alpha_ramp", default={}) or {}
    alpha_ramp_enabled = bool(alpha_ramp_cfg.get("enabled", False))
    alpha_ramp_start = float(alpha_ramp_cfg.get("start_alpha", 0.0))
    alpha_ramp_by = str(alpha_ramp_cfg.get("by", "optimizer_updates")).lower()
    alpha_ramp_updates = int(alpha_ramp_cfg.get("ramp_updates", 0) or 0)
    alpha_ramp_words = int(alpha_ramp_cfg.get("ramp_words", 0) or 0)
    if alpha_ramp_by not in {"optimizer_updates", "words_seen"}:
        raise ValueError("masked_alpha_ramp.by must be one of: optimizer_updates, words_seen")
    max_grad_norm = float(get_cfg(config, "gradient_clipping", "max_grad_norm", default=1.0))
    lr = float(get_cfg(config, "learning_rate", default=3e-4))
    wd = float(get_cfg(config, "weight_decay", default=0.1))
    warmup_ratio = float(get_cfg(config, "warmup_ratio", default=0.06))
    scheduler_name = str(get_cfg(config, "scheduler", "lr_scheduler_type", default="cosine"))
    mixed_precision = str(get_cfg(config, "mixed_precision", default="none")).lower()
    max_epochs = int(get_cfg(config, "num_train_epochs", default=10))
    max_words_seen = int(get_cfg(config, "max_words_seen", default=1_000_000_000))
    checkpoint_words = list(get_cfg(config, "checkpoint_words", default=[100_000_000, 200_000_000, 400_000_000, 600_000_000, 800_000_000, 1_000_000_000]))
    checkpoint_save_every_steps = get_cfg(config, "checkpoint_save_every_steps", default=None)
    checkpoint_save_every_steps = int(checkpoint_save_every_steps) if checkpoint_save_every_steps else 0
    checkpoint_steps = set(int(x) for x in get_cfg(config, "checkpoint_steps", default=[]) or [])
    keep_last_step_checkpoints = int(get_cfg(config, "keep_last_step_checkpoints", default=0) or 0)
    early_checkpoint_save_every_steps = int(get_cfg(config, "early_checkpoint_save_every_steps", default=0) or 0)
    early_checkpoint_until_step = int(get_cfg(config, "early_checkpoint_until_step", default=0) or 0)
    save_each_epoch_checkpoint = bool(get_cfg(config, "save_each_epoch_checkpoint", default=True))
    export_pt = bool(get_cfg(config, "export_pt", default=True))
    save_best_checkpoint = bool(get_cfg(config, "save_best_checkpoint", default=True))

    if max_epochs > 10:
        raise ValueError("BabyLM strict-style run should use num_train_epochs <= 10.")

    approx_words_per_step = int(sum(lm_ds["word_count"]) / max(1, len(lm_ds)) * train_batch_size)
    if approx_words_per_step <= 0:
        approx_words_per_step = block_size * train_batch_size
    estimated_micro_steps = max(1, int(max_words_seen / max(1, approx_words_per_step)))
    estimated_update_steps = max(1, math.ceil(estimated_micro_steps / max(1, grad_acc)))

    optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
    warmup_steps = max(1, int(estimated_update_steps * warmup_ratio))
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=estimated_update_steps,
    )

    use_amp = mixed_precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and mixed_precision == "fp16")
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    running_causal_loss = 0.0
    running_masked_loss = 0.0
    running_alpha = 0.0
    running_log_count = 0
    words_seen = 0
    reached_checkpoint_words = set()
    start_epoch = 1
    resume_skip_batches = 0
    model.zero_grad(set_to_none=True)
    best_epoch_loss = load_best_epoch_loss(checkpoints_dir)
    best_checkpoint_dir = checkpoints_dir / "ckpt_best"

    def current_masked_alpha(current_optimizer_step: int, current_words_seen: int) -> float:
        if not alpha_ramp_enabled:
            return target_alpha
        if alpha_ramp_by == "words_seen":
            if alpha_ramp_words <= 0:
                return target_alpha
            progress = min(1.0, max(0.0, float(current_words_seen) / float(alpha_ramp_words)))
        else:
            if alpha_ramp_updates <= 0:
                return target_alpha
            progress = min(1.0, max(0.0, float(current_optimizer_step) / float(alpha_ramp_updates)))
        return alpha_ramp_start + (target_alpha - alpha_ramp_start) * progress

    wandb_run = None
    if args.wandb:
        try:
            wandb = importlib.import_module("wandb")
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=str(get_cfg(config, "model_name", default=output_dir.name)),
                config={
                    "config_path": str(Path(args.config).resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "alpha": target_alpha,
                    "masked_ratio_sampling": float(get_cfg(config, "masked_ratio_sampling", default=0.0)),
                    "learning_rate": lr,
                    "batch_size_per_device": batch_size_per_device,
                    "train_batch_size": train_batch_size,
                    "gradient_accumulation_steps": grad_acc,
                    "max_words_seen": max_words_seen,
                },
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize wandb, continuing without wandb: {e}")
            wandb_run = None

    if args.resume_checkpoint:
        resume_dir = Path(args.resume_checkpoint).resolve()
        if not resume_dir.exists():
            raise FileNotFoundError(f"Resume checkpoint directory does not exist: {resume_dir}")
        train_state_path = resume_dir / "training_state.pt"
        if not train_state_path.exists():
            raise FileNotFoundError(f"Missing training_state.pt in resume checkpoint: {resume_dir}")
        model = AutoModelForCausalLM.from_pretrained(str(resume_dir))
        ensure_tokens(tokenizer, model)
        model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs (resume)")
        model.train()
        state = torch.load(train_state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if scaler is not None and state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state.get("epoch", 1))
        completed_epoch = epoch_number_from_checkpoint_name(resume_dir.name)
        if completed_epoch is not None:
            start_epoch = max(start_epoch, completed_epoch + 1)
        global_step = int(state.get("global_step", 0))
        optimizer_step = int(state.get("optimizer_step", max(0, global_step // max(1, grad_acc))))
        if completed_epoch is None and len(dataloader) > 0:
            # Resume from step-level checkpoint: skip already-consumed micro batches in current epoch.
            resume_skip_batches = int(global_step % len(dataloader))
        words_seen = int(state.get("words_seen", 0))
        running_loss = float(state.get("running_loss", 0.0))
        reached_checkpoint_words = set(int(x) for x in state.get("reached_checkpoint_words", []))
        print(
            f"Resumed from {resume_dir}: epoch={start_epoch}, micro_step={global_step}, optimizer_step={optimizer_step}, words_seen={words_seen}"
        )
        if resume_skip_batches > 0:
            print(f"Will skip first {resume_skip_batches} batches in resumed epoch to keep data order continuity.")

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Alpha target: {target_alpha}")
    if alpha_ramp_enabled:
        print(
            f"Alpha ramp enabled: start={alpha_ramp_start}, target={target_alpha}, by={alpha_ramp_by}, "
            f"ramp_updates={alpha_ramp_updates}, ramp_words={alpha_ramp_words}"
        )
    print(f"Batch size per device: {batch_size_per_device}")
    print(f"Train batch size: {train_batch_size}")
    print(f"Source proportional batches: {use_source_proportional_batches}")
    print(f"Max epochs: {max_epochs}")
    print(f"Max words seen: {max_words_seen}")
    print(f"Checkpoint words: {checkpoint_words}")
    if checkpoint_steps:
        print(f"Explicit checkpoint steps: {sorted(checkpoint_steps)}")
    if keep_last_step_checkpoints > 0:
        print(f"Keep last step checkpoints: {keep_last_step_checkpoints}")
    if early_checkpoint_save_every_steps > 0 and early_checkpoint_until_step > 0:
        print(
            f"Early checkpoint every steps: {early_checkpoint_save_every_steps} (until step {early_checkpoint_until_step})"
        )
    if checkpoint_save_every_steps > 0:
        print(f"Checkpoint every steps: {checkpoint_save_every_steps}")
    if save_best_checkpoint:
        print("Best checkpoint tracking: enabled (ckpt_best by epoch loss_total)")
    print(f"Estimated micro steps: {estimated_micro_steps}")
    print(f"Estimated optimizer steps: {estimated_update_steps}")
    print(f"Warmup steps (optimizer-step based): {warmup_steps}")

    stop_training = False
    for epoch in range(start_epoch, max_epochs + 1):
        if stop_training:
            break
        if batch_sampler is not None:
            # Keep sampler seed progression aligned with logical epoch index across resume runs.
            batch_sampler.epoch = max(0, epoch - 1)
        epoch_total_loss = 0.0
        epoch_causal_loss = 0.0
        epoch_masked_loss = 0.0
        epoch_alpha = 0.0
        epoch_count = 0
        epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{max_epochs}", leave=True)
        for batch_idx, batch in enumerate(epoch_pbar):
            if epoch == start_epoch and resume_skip_batches > 0 and batch_idx < resume_skip_batches:
                continue
            input_ids_clean = batch.get("input_ids_clean", batch["input_ids"]).to(device)
            input_ids_masked = batch.get("input_ids_masked", batch["input_ids"]).to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_causal = batch["labels_causal"].to(device)
            labels_masked = batch["labels_masked"].to(device)
            if "word_count" in batch:
                batch_words = int(batch["word_count"].sum().item())
            else:
                # Fallback: use token count as an approximation to avoid hard crash.
                batch_words = int(attention_mask.sum().item())
            alpha_now = current_masked_alpha(optimizer_step, words_seen)

            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available(), dtype=amp_dtype):
                # Dual-forward: causal branch uses clean text; masked branch uses corrupted text.
                outputs_causal = model(input_ids=input_ids_clean, attention_mask=attention_mask, use_cache=False)
                logits_causal = outputs_causal.logits

                shift_logits = logits_causal[:, :-1, :].contiguous()
                shift_labels = labels_causal[:, 1:].contiguous()
                causal_loss = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                if (labels_masked != -100).any():
                    outputs_masked = model(input_ids=input_ids_masked, attention_mask=attention_mask, use_cache=False)
                    logits_masked = outputs_masked.logits
                    masked_loss = ce(logits_masked.view(-1, logits_masked.size(-1)), labels_masked.view(-1))
                else:
                    masked_loss = torch.zeros_like(causal_loss)

                combined_loss = (1.0 - alpha_now) * causal_loss + alpha_now * masked_loss
                loss = combined_loss / grad_acc

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += combined_loss.item()
            running_causal_loss += causal_loss.item()
            running_masked_loss += masked_loss.item()
            running_alpha += alpha_now
            running_log_count += 1
            epoch_total_loss += combined_loss.item()
            epoch_causal_loss += causal_loss.item()
            epoch_masked_loss += masked_loss.item()
            epoch_alpha += alpha_now
            epoch_count += 1
            words_seen += batch_words
            global_step += 1

            if global_step % grad_acc == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer_step += 1
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.logging_steps == 0:
                avg = running_loss / max(1, running_log_count)
                avg_causal = running_causal_loss / max(1, running_log_count)
                avg_masked = running_masked_loss / max(1, running_log_count)
                avg_alpha = running_alpha / max(1, running_log_count)
                current_lr = scheduler.get_last_lr()[0]
                masked_contrib_ratio = (
                    (avg_alpha * avg_masked) / avg
                    if abs(avg) > 1e-12
                    else 0.0
                )
                epoch_pbar.set_postfix(
                    {
                        "m": global_step,
                        "o": optimizer_step,
                        "w": f"{words_seen/1e6:.1f}M",
                        "tot": f"{avg:.3f}",
                        "cau": f"{avg_causal:.3f}",
                        "msk": f"{avg_masked:.3f}",
                        "msk%": f"{masked_contrib_ratio * 100:.1f}",
                        "a": f"{avg_alpha:.3f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": float(avg),
                            "train/loss_total": float(avg),
                            "train/loss_causal": float(avg_causal),
                            "train/loss_masked": float(avg_masked),
                            "train/masked_alpha": float(avg_alpha),
                            "train/masked_contrib_ratio": float(masked_contrib_ratio),
                            "train/lr": float(current_lr),
                            "train/global_step": int(global_step),
                            "train/optimizer_step": int(optimizer_step),
                            "train/words_seen": int(words_seen),
                            "train/epoch": int(epoch),
                        },
                        step=int(global_step),
                    )
                running_loss = 0.0
                running_causal_loss = 0.0
                running_masked_loss = 0.0
                running_alpha = 0.0
                running_log_count = 0

            for target in checkpoint_words:
                if words_seen >= int(target) and int(target) not in reached_checkpoint_words:
                    ckpt_dir = checkpoints_dir / checkpoint_name_from_words(int(target))
                    checkpoint_artifacts = save_model_artifacts(model, tokenizer, ckpt_dir, export_pt=export_pt)
                    train_state_file = save_training_state(
                        save_dir=ckpt_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer_step=optimizer_step,
                        words_seen=words_seen,
                        running_loss=running_loss,
                        reached_checkpoint_words=reached_checkpoint_words,
                    )
                    save_json(
                        ckpt_dir / "trainer_state.json",
                        {
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "epoch": epoch,
                            "words_seen": words_seen,
                            "target_words_checkpoint": int(target),
                            "alpha": alpha_now,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "artifact_paths": checkpoint_artifacts,
                            "training_state_path": train_state_file,
                        },
                    )
                    reached_checkpoint_words.add(int(target))
                    print(f"Saved checkpoint at words={target}: {ckpt_dir}")

            step_save_interval = 0
            should_save_step_checkpoint = global_step in checkpoint_steps
            if (
                early_checkpoint_save_every_steps > 0
                and early_checkpoint_until_step > 0
                and global_step <= early_checkpoint_until_step
            ):
                step_save_interval = early_checkpoint_save_every_steps
            elif checkpoint_save_every_steps > 0:
                step_save_interval = checkpoint_save_every_steps

            if not should_save_step_checkpoint and step_save_interval > 0:
                should_save_step_checkpoint = global_step % step_save_interval == 0

            if should_save_step_checkpoint:
                step_dir = checkpoints_dir / step_checkpoint_name(global_step)
                step_artifacts = save_model_artifacts(model, tokenizer, step_dir, export_pt=export_pt)
                train_state_file = save_training_state(
                    save_dir=step_dir,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    words_seen=words_seen,
                    running_loss=running_loss,
                    reached_checkpoint_words=reached_checkpoint_words,
                )
                save_json(
                    step_dir / "trainer_state.json",
                    {
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "words_seen": words_seen,
                        "alpha": alpha_now,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "artifact_paths": step_artifacts,
                        "training_state_path": train_state_file,
                    },
                )
                print(f"Saved step checkpoint: {step_dir}")
                prune_old_step_checkpoints(checkpoints_dir, keep_last_step_checkpoints)

            if words_seen >= max_words_seen:
                print(f"Reached max_words_seen={max_words_seen}. Stopping training.")
                stop_training = True
                break

        epoch_avg_total = epoch_total_loss / max(1, epoch_count)
        epoch_avg_causal = epoch_causal_loss / max(1, epoch_count)
        epoch_avg_masked = epoch_masked_loss / max(1, epoch_count)
        epoch_avg_alpha = epoch_alpha / max(1, epoch_count)
        epoch_masked_contrib_ratio = (
            (epoch_avg_alpha * epoch_avg_masked) / epoch_avg_total
            if abs(epoch_avg_total) > 1e-12
            else 0.0
        )
        print(
            f"epoch={epoch} avg_total={epoch_avg_total:.4f} avg_causal={epoch_avg_causal:.4f} "
            f"avg_masked={epoch_avg_masked:.4f} avg_alpha={epoch_avg_alpha:.4f} "
            f"avg_masked_contrib={epoch_masked_contrib_ratio * 100:.2f}% micro_steps={epoch_count}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/loss_total": float(epoch_avg_total),
                    "epoch/loss_causal": float(epoch_avg_causal),
                    "epoch/loss_masked": float(epoch_avg_masked),
                    "epoch/masked_alpha": float(epoch_avg_alpha),
                    "epoch/masked_contrib_ratio": float(epoch_masked_contrib_ratio),
                    "epoch/index": int(epoch),
                    "train/optimizer_step": int(optimizer_step),
                },
                step=int(global_step),
            )

        if save_best_checkpoint and epoch_avg_total < best_epoch_loss:
            best_epoch_loss = epoch_avg_total
            best_artifacts = save_model_artifacts(model, tokenizer, best_checkpoint_dir, export_pt=export_pt)
            best_train_state_file = save_training_state(
                save_dir=best_checkpoint_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer_step=optimizer_step,
                words_seen=words_seen,
                running_loss=running_loss,
                reached_checkpoint_words=reached_checkpoint_words,
            )
            save_json(
                best_checkpoint_dir / "trainer_state.json",
                {
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "completed_epoch": epoch,
                    "resume_epoch": epoch + 1,
                    "words_seen": words_seen,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "is_best_checkpoint": True,
                    "best_metric": "epoch_metrics.loss_total",
                    "best_metric_value": best_epoch_loss,
                    "artifact_paths": best_artifacts,
                    "training_state_path": best_train_state_file,
                    "epoch_metrics": {
                        "loss_total": epoch_avg_total,
                        "loss_causal": epoch_avg_causal,
                        "loss_masked": epoch_avg_masked,
                        "masked_alpha": epoch_avg_alpha,
                        "masked_contrib_ratio": epoch_masked_contrib_ratio,
                        "micro_steps": epoch_count,
                    },
                },
            )
            print(f"Updated best checkpoint (epoch loss_total={best_epoch_loss:.6f}): {best_checkpoint_dir}")

        if save_each_epoch_checkpoint:
            epoch_dir = checkpoints_dir / epoch_checkpoint_name(epoch)
            epoch_artifacts = save_model_artifacts(model, tokenizer, epoch_dir, export_pt=export_pt)
            train_state_file = save_training_state(
                save_dir=epoch_dir,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer_step=optimizer_step,
                words_seen=words_seen,
                running_loss=running_loss,
                reached_checkpoint_words=reached_checkpoint_words,
            )
            save_json(
                epoch_dir / "trainer_state.json",
                {
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "completed_epoch": epoch,
                    "resume_epoch": epoch + 1,
                    "words_seen": words_seen,
                    "alpha": epoch_avg_alpha,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "artifact_paths": epoch_artifacts,
                    "training_state_path": train_state_file,
                    "epoch_metrics": {
                        "loss_total": epoch_avg_total,
                        "loss_causal": epoch_avg_causal,
                        "loss_masked": epoch_avg_masked,
                        "masked_alpha": epoch_avg_alpha,
                        "masked_contrib_ratio": epoch_masked_contrib_ratio,
                        "micro_steps": epoch_count,
                    },
                },
            )
            print(f"Saved epoch checkpoint: {epoch_dir}")

    final_dir = output_dir / "final"
    final_artifacts = save_model_artifacts(model, tokenizer, final_dir, export_pt=export_pt)
    save_json(
        output_dir / "run_summary.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "starting_checkpoint": model_path,
            "training_data_path": str(data_path.resolve()),
            "masked_alpha": target_alpha,
            "masked_alpha_ramp": {
                "enabled": alpha_ramp_enabled,
                "start_alpha": alpha_ramp_start,
                "by": alpha_ramp_by,
                "ramp_updates": alpha_ramp_updates,
                "ramp_words": alpha_ramp_words,
            },
            "masked_ratio_sampling": float(get_cfg(config, "masked_ratio_sampling", default=0.0)),
            "global_step": global_step,
            "optimizer_step": optimizer_step,
            "words_seen": words_seen,
            "max_words_seen": max_words_seen,
            "num_train_epochs": max_epochs,
            "estimated_micro_steps": estimated_micro_steps,
            "estimated_optimizer_steps": estimated_update_steps,
            "warmup_steps": warmup_steps,
            "save_each_epoch_checkpoint": save_each_epoch_checkpoint,
            "save_best_checkpoint": save_best_checkpoint,
            "early_checkpoint_save_every_steps": early_checkpoint_save_every_steps if early_checkpoint_save_every_steps > 0 else None,
            "early_checkpoint_until_step": early_checkpoint_until_step if early_checkpoint_until_step > 0 else None,
            "checkpoint_save_every_steps": checkpoint_save_every_steps if checkpoint_save_every_steps > 0 else None,
            "keep_last_step_checkpoints": keep_last_step_checkpoints if keep_last_step_checkpoints > 0 else None,
            "export_pt": export_pt,
            "final_model_dir": str(final_dir.resolve()),
            "final_artifact_paths": final_artifacts,
            "best_checkpoint_dir": str(best_checkpoint_dir.resolve()) if best_checkpoint_dir.exists() else None,
            "best_epoch_loss_total": None if best_epoch_loss == float("inf") else best_epoch_loss,
        },
    )

    print("Training finished.")
    print(f"Final model saved to: {final_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

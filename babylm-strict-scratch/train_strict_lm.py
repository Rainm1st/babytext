"""
BabyLM Strict：仅使用官方 `*.train.txt` 语料，从随机初始化开始训练因果语言模型。
不加载 OpenAI GPT-2 或 BabyLM baseline 的 pytorch 权重（仅可选用 HF 词表 / 架构 config）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:
    wandb = None

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_VISUAL_BACKENDS", "1")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
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


def ensure_pad_token(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
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


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
        attention_mask = (input_ids != self.pad_token_id).long()
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        word_count = torch.tensor([int(f["word_count"]) for f in features], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "word_count": word_count,
        }


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name.lower() or "ln_" in name.lower() or "layernorm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def save_ckpt(model, tokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(str(out_dir), safe_serialization=False)
    tokenizer.save_pretrained(str(out_dir))


def save_training_state(
    path: Path,
    *,
    next_epoch: int,
    global_step: int,
    optimizer_step: int,
    best_loss: float,
    model_dir: str,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    partial_epoch: bool,
    epoch_being_trained: int,
    batches_completed_in_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "next_epoch": next_epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "best_loss": best_loss,
        "model_dir": model_dir,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler and scaler.is_enabled() else None,
        "partial_epoch": partial_epoch,
        "epoch_being_trained": epoch_being_trained,
        "batches_completed_in_epoch": batches_completed_in_epoch,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python_random": random.getstate(),
    }
    torch.save(payload, path)


def load_training_state(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def maybe_init_wandb(config: Dict[str, Any], output_dir: Path) -> bool:
    if not config.get("use_wandb"):
        return False
    if wandb is None:
        print("[WARN] use_wandb=true but wandb not installed; skip logging.")
        return False
    name = config.get("wandb_run_name") or output_dir.name
    wandb.init(project=str(config.get("wandb_project", "babytext-strict-scratch")), name=name, reinit=True)
    shallow = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
    wandb.config.update(shallow)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="BabyLM Strict: train causal LM from random init.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="断点续训：与 --output-dir 相同路径，需存在 training_state.pt。",
    )
    args = parser.parse_args()

    config = load_json(Path(args.config).resolve())
    cuda_visible_devices = get_cfg(config, "cuda_visible_devices", default=None)
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    seed = int(get_cfg(config, "seed", default=42))
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_name = str(get_cfg(config, "tokenizer_name", default="gpt2"))
    scratch_config_name = str(get_cfg(config, "scratch_architecture", "scratch_hf_config_name", default="gpt2"))

    training_data_root = Path(get_cfg(config, "training_data_path", "data_path")).expanduser()
    if not training_data_root.is_absolute():
        training_data_root = (Path.cwd() / training_data_root).resolve()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "training_state.pt"

    resume_dir = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_dir is not None and resume_dir != output_dir:
        raise SystemExit("[ERROR] --resume-from 必须与 --output-dir 相同，避免写到错误目录。")
    resume = resume_dir is not None
    if resume and not state_path.is_file():
        raise SystemExit(f"[ERROR] 续训需要文件: {state_path}")

    if resume:
        st0 = load_training_state(state_path)
        mdir = Path(st0["model_dir"])
        if not mdir.is_dir():
            raise SystemExit(f"[ERROR] 断点 model_dir 不存在: {mdir}")
        print(f"[RESUME] 从 {mdir} 加载权重与 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(str(mdir), use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(str(mdir))
        if st0.get("rng_cpu") is not None:
            torch.set_rng_state(st0["rng_cpu"])
        if st0.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st0["rng_cuda"])
        if st0.get("python_random") is not None:
            random.setstate(st0["python_random"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        hf_cfg = AutoConfig.from_pretrained(scratch_config_name)
        model = AutoModelForCausalLM.from_config(hf_cfg)

    ensure_pad_token(tokenizer, model)
    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")

    text_files = list_train_files(training_data_root)
    block_size = int(get_cfg(config, "max_sequence_length", default=512))
    num_workers_dl = max(0, int(args.num_workers))
    lm_ds = build_lm_dataset_by_source(text_files, tokenizer, block_size, num_workers_dl)

    batch_size_per_device = int(get_cfg(config, "batch_size_per_device", default=8))
    dev_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    train_batch_size = batch_size_per_device * max(1, dev_count)
    grad_acc = int(get_cfg(config, "gradient_accumulation_steps", default=1))

    source_names = [Path(p).name for p in text_files]
    source_weights = normalize_source_weights(source_names, get_cfg(config, "source_word_counts", default={}) or {})
    use_prop = bool(get_cfg(config, "source_proportional_batches", default=True))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = CausalCollator(pad_token_id=int(pad_id))

    if use_prop:
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
            num_workers=max(0, num_workers_dl),
            pin_memory=torch.cuda.is_available(),
        )
    else:
        dataloader = DataLoader(
            lm_ds,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=max(0, num_workers_dl),
            pin_memory=torch.cuda.is_available(),
        )

    max_epochs = int(get_cfg(config, "num_train_epochs", default=10))
    if max_epochs > 10:
        raise ValueError("BabyLM competition entries: num_train_epochs must be <= 10.")

    lr = float(get_cfg(config, "learning_rate", default=3e-4))
    wd = float(get_cfg(config, "weight_decay", default=0.1))
    warmup_ratio = float(get_cfg(config, "warmup_ratio", default=0.06))
    scheduler_name = str(get_cfg(config, "scheduler", default="cosine"))
    max_grad_norm = float(get_cfg(config, "gradient_clipping", default=1.0))
    mixed_precision = str(get_cfg(config, "mixed_precision", default="none")).lower()

    steps_per_epoch = max(1, len(dataloader))
    total_micro_steps = max_epochs * steps_per_epoch
    total_optimizer_steps = max(1, math.ceil(total_micro_steps / max(1, grad_acc)))
    warmup_steps = max(1, int(total_optimizer_steps * warmup_ratio))

    use_amp = mixed_precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and mixed_precision == "fp16")

    save_every_steps = int(get_cfg(config, "checkpoint_save_every_steps", default=0) or 0)
    global_step = 0
    optimizer_step = 0
    best_loss = float("inf")
    start_epoch = 1
    resume_skip_batches = 0

    if resume:
        st_opt = load_training_state(state_path)
        optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        optimizer.load_state_dict(st_opt["optimizer"])
        scheduler.load_state_dict(st_opt["scheduler"])
        if st_opt.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(st_opt["scaler"])
        global_step = int(st_opt["global_step"])
        optimizer_step = int(st_opt["optimizer_step"])
        best_loss = float(st_opt["best_loss"])
        if st_opt.get("partial_epoch"):
            start_epoch = int(st_opt["epoch_being_trained"])
            resume_skip_batches = int(st_opt["batches_completed_in_epoch"])
        else:
            start_epoch = int(st_opt["next_epoch"])
            resume_skip_batches = 0
        print(f"[RESUME] start_epoch={start_epoch}, skip_batches={resume_skip_batches}, global_step={global_step}")
    else:
        optimizer = build_optimizer(model, lr=lr, weight_decay=wd)
        scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

    _ = maybe_init_wandb(config, output_dir)

    print(f"Device: {device}")
    if resume:
        print("[RESUME] 已恢复优化器与学习率调度器状态")
    else:
        print(f"Tokenizer: {tokenizer_name} (仅词表；LM 随机初始化)")
        print(f"Architecture: {scratch_config_name}")
    print(f"Data: {training_data_root}")
    print(f"Epochs: {max_epochs} (赛事上限 10)")
    print(f"Seq len: {block_size}, batch {train_batch_size}, grad_acc {grad_acc}")
    print(f"LR {lr}, scheduler {scheduler_name}, warmup_steps {warmup_steps}")

    if start_epoch > max_epochs:
        print("[RESUME] 训练已全部完成，跳过保存 final（如需重新导出请用已有 checkpoints）。")
        return

    for epoch in range(start_epoch, max_epochs + 1):
        if use_prop:
            batch_sampler.epoch = epoch - 1
        skip_this_epoch = resume_skip_batches if epoch == start_epoch else 0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{max_epochs}")
        epoch_loss = 0.0
        n_batches = 0
        batches_done = 0

        for batch in pbar:
            if skip_this_epoch > 0:
                skip_this_epoch -= 1
                batches_done += 1
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available(), dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                loss = out.loss / grad_acc

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            global_step += 1
            epoch_loss += float(loss.item()) * grad_acc
            n_batches += 1
            batches_done += 1

            if global_step % grad_acc == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1

            if global_step % args.logging_steps == 0:
                avg_e = epoch_loss / max(1, n_batches)
                pbar.set_postfix(
                    loss=f"{avg_e:.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    gstep=global_step,
                    optstep=optimizer_step,
                )
                wb_run = getattr(wandb, "run", None) if wandb is not None else None
                if config.get("use_wandb") and wb_run is not None:
                    wandb.log(
                        {
                            "train/loss_epoch_avg": avg_e,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/global_step": global_step,
                            "train/optimizer_step": optimizer_step,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if save_every_steps > 0 and global_step % save_every_steps == 0:
                step_path = ckpt_dir / f"step_{global_step:08d}"
                save_ckpt(model, tokenizer, step_path)
                save_training_state(
                    state_path,
                    next_epoch=epoch,
                    global_step=global_step,
                    optimizer_step=optimizer_step,
                    best_loss=best_loss,
                    model_dir=str(step_path.resolve()),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    partial_epoch=True,
                    epoch_being_trained=epoch,
                    batches_completed_in_epoch=batches_done,
                )
                print(f"Saved step ckpt + state: {step_path}")

        resume_skip_batches = 0
        avg = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch} mean loss: {avg:.4f}")
        ep_path = ckpt_dir / f"epoch_{epoch:02d}"
        save_ckpt(model, tokenizer, ep_path)
        if avg < best_loss:
            best_loss = avg
            save_ckpt(model, tokenizer, ckpt_dir / "best")
            print(f"New best loss {best_loss:.4f} -> checkpoints/best")
        save_training_state(
            state_path,
            next_epoch=epoch + 1,
            global_step=global_step,
            optimizer_step=optimizer_step,
            best_loss=best_loss,
            model_dir=str(ep_path.resolve()),
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            partial_epoch=False,
            epoch_being_trained=epoch,
            batches_completed_in_epoch=0,
        )
        wb_run = getattr(wandb, "run", None) if wandb is not None else None
        if config.get("use_wandb") and wb_run is not None:
            wandb.log({"train/epoch_mean_loss": avg, "train/epoch": epoch}, step=global_step)

    final_dir = output_dir / "final"
    save_ckpt(model, tokenizer, final_dir)
    save_json(
        output_dir / "run_summary.json",
        {
            "config_path": str(Path(args.config).resolve()),
            "training_data_path": str(training_data_root.resolve()),
            "tokenizer_name": tokenizer_name,
            "scratch_architecture": scratch_config_name,
            "num_train_epochs": max_epochs,
            "max_sequence_length": block_size,
            "learning_rate": lr,
            "weight_decay": wd,
            "warmup_ratio": warmup_ratio,
            "scheduler": scheduler_name,
            "final_model_dir": str(final_dir.resolve()),
            "best_epoch_mean_loss": None if best_loss == float("inf") else best_loss,
            "final_global_step": global_step,
            "final_optimizer_step": optimizer_step,
            "resumed_from_checkpoint": resume,
        },
    )
    print(f"Training done. Upload `final/` to Hugging Face and run github.com/babylm-org/babylm-eval strict.")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()

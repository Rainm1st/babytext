# GPT2 Masked Objective Experiment

This folder is the unified workspace for the BabyLM GPT2 masked-objective ablation.

## Experiment Goal

Compare whether adding a small masked objective to the official GPT2 baseline improves language abilities, and compare 5% vs 10%.

## Models

- `gpt2_base_official`: official `babylm-baseline-100m-gpt2`
- `gpt2_mask5_run1`: GPT2 with alpha=0.05
- `gpt2_mask10_run1`: GPT2 with alpha=0.10
- `gpt2_mask20_run1`: GPT2 with alpha=0.20
- `gpt2_mask50_run1`: GPT2 with alpha=0.50

## Fair Comparison Rules

Keep all settings identical except masked ratio:

- same starting checkpoint
- same tokenizer files and special tokens
- same dataset and filtering version
- same max sequence length and packing
- same optimizer/scheduler/lr/warmup
- same total training budget (prefer fixed total tokens)
- same checkpoint save frequency
- same evaluation pipeline

## Loss Definition

`L = (1 - alpha) * L_causal + alpha * L_masked`

- alpha=0.05 for `gpt2_mask5_run1`
- alpha=0.10 for `gpt2_mask10_run1`
- alpha=0.20 for `gpt2_mask20_run1`
- alpha=0.50 for `gpt2_mask50_run1`

## Masking Rule (must stay identical)

- sample valid tokens only (exclude special/padding)
- replacement policy:
  - 80% -> `[MASK]`
  - 10% -> random token
  - 10% -> keep original token
- masked loss is computed only on selected mask positions

## Recommended Folder Usage

- `configs/`: run configs
- `logs/`: per-run training notes
- `results/`: evaluation outputs and comparison tables
- `prepare_experiment.py`: preflight checker and model-weight downloader

Training outputs and checkpoints are intentionally written outside the Git
workspace by default:

```text
/data0/language/babylm_runs/gpt2_masked_objective/
```

The default can be overridden with:

```bash
export GPT2_MASKED_RUN_BASE=/data0/language/babylm_runs/gpt2_masked_objective
```

## Preflight Before Training

Run this before any training:

`python experiments/gpt2-masked-objective/prepare_experiment.py --no-baseline-compare`

If model weights are missing in local cache, auto-download and re-check:

`python experiments/gpt2-masked-objective/prepare_experiment.py --auto-download-model`

## Train With `babylm` Virtual Environment

Mask-only training (official baseline used as reference, not retrained):

`powershell -ExecutionPolicy Bypass -File experiments/gpt2-masked-objective/run_train.ps1 -Model mask5`

`powershell -ExecutionPolicy Bypass -File experiments/gpt2-masked-objective/run_train.ps1 -Model mask10`

`powershell -ExecutionPolicy Bypass -File experiments/gpt2-masked-objective/run_train.ps1 -Model mask20`

`powershell -ExecutionPolicy Bypass -File experiments/gpt2-masked-objective/run_train.ps1 -Model mask50`

Or run the Python command directly:

`D:/conda_envs/chatgpt/python.exe experiments/gpt2-masked-objective/train_hybrid_gpt2.py --config experiments/gpt2-masked-objective/configs/run_mask10.json --output-dir /data0/language/babylm_runs/gpt2_masked_objective/gpt2_mask10_run1`

## One-Command Full Pipeline

This runs mask5 + mask10 sequentially, then evaluates both runs and updates result tables:

`powershell -ExecutionPolicy Bypass -File experiments/gpt2-masked-objective/run_both_mask.ps1`

`run_both_mask` reads run summaries from `GPT2_MASKED_RUN_BASE`, matching the
training launcher output location.

Debug mode (faster eval with sample cap):

`powershell -ExecutionPolicy Bypass -File experiments/gpt2-masked-objective/run_both_mask.ps1 -EvalMaxSamples 200`

## Competition Budget Policy

This script now follows a BabyLM-style budget:

- max 10 epochs
- stop by `max_words_seen` (default 1B)
- save at `checkpoint_words` milestones
- final selection rule: `last_checkpoint`

## Main Metrics

- priority 1: official overall score (if available)
- priority 2: zero-shot average
- priority 3: fine-tuning average

## Analysis Rules

- both mask runs > baseline -> masked objective helps
- 10% > 5% -> higher mask ratio is better under current setup
- 5% > 10% -> too much masking may hurt causal learning
- interpret per-task differences as ability-dimension sensitivity

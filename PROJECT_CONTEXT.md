# BabyLM Project Context

We work locally on Windows mainly to edit scripts, then upload/sync to the server for actual training.

Local GitHub clone:
- D:\code\Repository_GitHub\babytext

Server layout:
- Main workspace: /home/language/babytext/
- Git repository / experiments root: /home/language/babytext/experiments/
- BabyLM STRICT text-only corpus: /home/language/babytext/experiments/BabyLM-2026-Strict/
- GPT-BERT official training repo: /home/language/babytext/experiments/gpt-bert/
- GPT-BERT baseline local download: /home/language/babytext/experiments/gpt-bert-masked-focus-baseline/
- GPT-BERT tokenized output: /data0/language/babylm_runs/gpt_bert_strict_tokenized/
- GPT-BERT continued run output: /data0/language/babylm_runs/gpt_bert_masked_focus_continue/

Python environment:
- Local Windows interpreter for editing/testing:
  D:\Miniconda3\envs\babylm\python.exe
- Server Python environment should be whatever CUDA/PyTorch environment is used for training. Confirm before long runs.

Competition target:
- BabyLM 2026 STRICT track.
- Text-only training.
- Training corpus <= 100M words.
- Leaderboard-eligible exposure <= 1B words seen.
- This usually means 100M words × at most 10 epochs.
- Intermediate checkpoints are required:
  every 1M words until 10M,
  every 10M words until 100M,
  every 100M words until 1B.

Important rule interpretation:
- Official Hugging Face baseline models are already trained, not blank models.
- Continuing from an official baseline is allowed as an experiment, but should be described as continued pretraining and likely exceeds a clean from-scratch 1B-word STRICT budget.
- For a clean STRICT submission, initialize the model from architecture config only, not from trained baseline weights.

GPT-2 project:
- Directory: /home/language/babytext/experiments/gpt2-masked-objective/
- Configs currently point to server paths:
  /home/language/babytext/experiments/models--BabyLM-community--babylm-baseline-100m-gpt2/snapshots/...
  /home/language/babytext/experiments/BabyLM-2026-Strict/
- Existing scripts include:
  train_hybrid_gpt2.py
  run_train.py
  run_train.ps1
  configs/run_mask5.json
  configs/run_mask10.json
  configs/run_mask20.json
  configs/run_mask50.json
- Current configs are designed around starting from a GPT-2 baseline checkpoint. For from-scratch training, add or modify a path that creates GPT2LMHeadModel(GPT2Config(...)) instead of loading from_pretrained.

Approximate GPT-2 Strict baseline architecture:
- model type: GPT-2 causal LM
- vocab_size: about 16384
- n_layer: 12
- n_embd: 768
- n_head: 12
- parameter count: about 98M
- objective: causal LM, optionally with our masked-objective modification if experimentally justified

GPT-BERT project:
- Directory: /home/language/babytext/experiments/gpt-bert-masked-focus/
- Official GPT-BERT repo expected at:
  /home/language/babytext/experiments/gpt-bert/
- Local HF baseline download expected at:
  /home/language/babytext/experiments/gpt-bert-masked-focus-baseline/
- Tokenizer expected at:
  /data0/language/babylm_runs/gpt_bert_strict_tokenized/tokenizer/tokenizer.json
- Tokenized train/valid paths are passed via:
  GPT_BERT_TRAIN_PATH=/path/to/train.bin
  GPT_BERT_VALID_PATH=/path/to/valid.bin
- Run output default:
  /data0/language/babylm_runs/gpt_bert_masked_focus_continue/

GPT-BERT scripts:
- scripts/01_download_baseline.sh
  downloads BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus
- scripts/02_setup_gpt_bert_repo.sh
  clones ltgoslo/gpt-bert
- scripts/03_verify_model_load.py
  verifies HF/local GPT-BERT baseline loading
- scripts/04b_tokenize_babylm_strict.sh
  tokenizes BabyLM STRICT data
- scripts/06_train_single_gpu_torchrun.sh
  launches single-GPU torchrun training

Evaluation understanding:
- Validation:
  cross-entropy loss on masked tokens.
- Zero-shot:
  no task finetuning; direct scoring/selection on BLiMP, BLiMP Supplement, EWoK, Entity Tracking, WUGs.
- Eye Tracking and Self-paced Reading:
  change in R² prediction from baseline.
- Finetuning:
  initialize from pretrained checkpoint, finetune on downstream task data, then evaluate:
  MNLI, BoolQ, MultiRC, WSC, MRPC, QQP.
- Finetuning is part of evaluation and is separate from pretraining exposure budget.

Recommended next steps:
1. Keep local edits in D:\code\Repository_GitHub\babytext.
2. Sync/upload to /home/language/babytext/experiments on the server.
3. Confirm the server has /home/language/babytext/experiments/BabyLM-2026-Strict with text-only STRICT data.
4. Decide whether the next experiment is:
   a. clean from-scratch STRICT model, or
   b. continued pretraining from an official baseline.
5. If from scratch:
   train or select tokenizer using only allowed STRICT text;
   initialize model from config, not from baseline weights;
   train up to 1B words seen;
   save required intermediate checkpoints.
6. If continued pretraining:
   explicitly document it as continued pretraining from an official baseline.
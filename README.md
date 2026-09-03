# BAM-B2 · BabyLM 2026 Strict

This repository is the compact public record for **BAM-B2**, our final BabyLM 2026 Strict-track submission. Historical pre-submission experiments are intentionally excluded from the final layout because they do not reproduce the submitted model.

- Final model: [Recor2d/BAM-B2](https://huggingface.co/Recor2d/BAM-B2)
- Official leaderboard: [BabyLM Leaderboard 2026](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026)
- GitHub repository: [Rainm1st/babytext](https://github.com/Rainm1st/babytext)

## Current leaderboard snapshot

Verified on **2026-09-03**. The official board currently contains two identical rows, `BAM-B2` and `BAM-B2-from-scratch (mntp)`. Both official rows are left untouched; this repository uses `BAM-B2` as the canonical name and does not claim a final competition placement while the board remains live.

| Metric | Score |
| --- | ---: |
| Overall Average | 43.59 |
| NLP Average | 55.88 |
| Human-like Average | 0.58 |
| BLiMP | 77.72 |
| BLiMP Supplement | 70.78 |
| EWoK | 56.04 |
| Entity Tracking | 25.61 |
| COMPS | 56.69 |
| GlobalPIQA | 33.15 |
| (Super)GLUE | 71.14 |
| Reading | 1.15 |
| AoA | 0.00 |

The detailed (Super)GLUE scores are BoolQ 70.58, MNLI 64.24, MRPC 89.27, MultiRC 70.71, QQP 74.95, RTE 64.75, and WSC 63.46. Reading consists of Self-paced Reading 0.29 and Eye Tracking 2.02.

## Submitted model

BAM-B2 is a 96.3M-parameter GPT-BERT/MNTP model trained from random initialization.

- Objective: hybrid masked-next-token prediction and causal language modeling, 15:1 ratio
- Data: 95M official Strict-token exposure plus 5M Reading/AoA-targeted exposure derived only from the official Strict training corpus
- Tokenizer: 16,000-token BabyLM 2026 Strict tokenizer
- Architecture: 12 layers, hidden size 720, 12 attention heads, maximum sequence length 512
- Training: 10 epochs, LAMB optimizer, learning rate 0.007, linear warmup, cosine decay, and linear cooldown
- Seeds: training 42; deterministic data mixing 17
- Exclusions: no external text, multilingual or multimodal data, evaluation examples, human annotation, or external teacher-model output

These values reproduce the metadata currently attached to the official leaderboard entry. The public model repository is the canonical source for the weights.

## Repository contents

`submission/BAM-B2/` mirrors the final public model's non-weight inference files from Hugging Face commit `b1a7488`:

- model and Transformers configuration
- custom GPT-BERT/MNTP implementation
- tokenizer and special-token configuration

`model.safetensors` (385 MB) is deliberately not duplicated in GitHub. Download it from the canonical Hugging Face model repository. The leaderboard predictions file is also not fabricated or reconstructed here; it was not present in this local checkout.

## Load the model

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Recor2d/BAM-B2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
```

`trust_remote_code=True` executes the model repository's custom implementation. The same implementation is mirrored under `submission/BAM-B2/` for review.

# GPT-BERT Masked Focus 继续训练（BabyLM Strict）

基于 leaderboard 榜首族 **Baseline-gpt-bert-masked-focus (mntp)**，使用官方仓库 [ltgoslo/gpt-bert](https://github.com/ltgoslo/gpt-bert) 与 HF 权重 [babylm-baseline-100m-gpt-bert-masked-focus](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus)。

## 与 GPT-2 线的区别

| 项目 | GPT-2 (`train_hybrid_gpt2.py`) | GPT-BERT (本目录) |
|------|-------------------------------|-------------------|
| 模型类 | `AutoModelForCausalLM` | `AutoModelForMaskedLM` + trust_remote_code |
| 训练代码 | 本仓库 | `ltgoslo/gpt-bert/pretraining` |
| 数据格式 | `*.train.txt` | **预 tokenize 二进制** |
| 提交 backend | `causal` | **`mntp`** |

## 服务器执行顺序

```bash
cd ~/babytext/experiments/gpt-bert-masked-focus/scripts
bash 01_download_baseline.sh
bash 02_setup_gpt_bert_repo.sh
python 03_verify_model_load.py
bash 04_prepare_strict_tokenized.sh
# 按 04 打印说明在 gpt-bert 仓库内完成 corpus_tokenization 后：
export GPT_BERT_TRAIN_PATH=/path/to/train.bin
export GPT_BERT_VALID_PATH=/path/to/valid.bin
bash 05_train_continue_masked_focus.sh
```

## 评测与提交

```bash
cd ~/babytext/experiments/babylm-eval/strict
# 将 YOUR_MODEL 换成你保存的 checkpoint 目录或 HF 上传名
bash scripts/eval_zero_shot.sh /path/to/your/gpt_bert_ckpt mntp
bash scripts/eval_finetuning.sh --model_path /path/to/your/gpt_bert_ckpt

python -m evaluation_pipeline.collate_preds \
  --model_path_or_name YOUR_MODEL \
  --backend mntp \
  --track strict
```

## 重要说明

1. **不能**用 `train_hybrid_gpt2.py` 加载 GPT-BERT 权重。
2. `05_train_continue_masked_focus.sh` 中 `train_100m.py` 的 CLI 以你 clone 的 gpt-bert 版本为准；若报错，对照 `pretraining/README.md` 改参数名。
3. 从官方 HF 权重「继续训」若脚本不支持 `--resume`，需查阅 gpt-bert 是否提供 checkpoint 加载参数，或联系官方 issue。

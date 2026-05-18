# GPT-BERT Masked Focus（官方 `train_100m`）

基于 [ltgoslo/gpt-bert](https://github.com/ltgoslo/gpt-bert) 与 HF 词表 [`babylm-baseline-100m-gpt-bert-masked-focus`](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus)（16k 词表 → **`configs/base.json`**，勿用 `small.json`）。

## 训练策略（与排行榜 Masked Focus 对齐）

| 项 | 说明 |
|----|------|
| 目标 | 主力 **MNTP/掩码跨度**（span mask + 随机替换比例见 `--mask_*`） |
| 单卡 hybrid | `hybrid_numerator=1` / `hybrid_denominator=1` → 全程 masked 分支（与多卡 15/16 masked + 1/16 causal 不同，单卡为满足整除约束的折中） |
| 优化器 | LAMB，`lr≈0.007`，warmup/cooldown 比例 1.6% |
| 从零 | **不传** `--checkpoint_filename`：随机初始化 `Bert(config)` |
| 续训 | 指向 `RUN_DIR` 下 `*state_dict.bin`（含 model/ema/optimizer/scheduler/step） |

官方脚本内建 **`tqdm` 训练进度条**；验证按 `validate_every` 触发。

### W&B

默认 **`WANDB_DISABLED=true`**（不写远端；可选用 `WANDB_MODE=offline`）。需云端曲线时：`export WANDB_DISABLED=false` 并 `wandb login`。若 `import wandb` 与实体配置报错，已提供 `10_patch_train_100m_wandb_stub.py`（与 `09` 一同在 `05`/`06` 中调用）。

### 断点续训（GPT-BERT）

```bash
export GPT_BERT_RESUME_STATE=/data0/language/.../masked_focus_continue_strict_state_dict.bin
bash 06_train_single_gpu_torchrun.sh
```

每次 `save_every` 会写入 `*_state_dict.bin`，与权重 `*.bin` 同名前缀。

默认输出目录均在 `/data0/language/babylm_runs` 下：

```text
scratch run: /data0/language/babylm_runs/gpt_bert_scratch_strict
continued run: /data0/language/babylm_runs/gpt_bert_masked_focus_continue
tokenized data: /data0/language/babylm_runs/gpt_bert_strict_tokenized
```

可用 `RUN_DIR` 覆盖训练输出目录；断点续训时 `GPT_BERT_RESUME_STATE`
应指向同一 `RUN_DIR` 下最新的 `*_state_dict.bin`。

### 训练预算（Strict 从零）

`06_train_single_gpu_torchrun.sh` 默认按 **Strict 训练语料词数 × 10 epochs** 换算 `max_steps`，而不是固定 20000 steps。默认估算：

- `GPT_BERT_STRICT_TRAIN_WORDS=96376391`
- `GPT_BERT_STRICT_EPOCHS=10`
- `GPT_BERT_SUBWORDS_PER_WORD_X1000=1592`
- `GPT_BERT_SEQ_LENGTH=128`
- `GPT_BERT_GLOBAL_BATCH=4096`

若显式设置 `GPT_BERT_MAX_STEPS`，则使用手动 step 数覆盖预算换算。算力不足可先用 small bin / 小 step smoke；**最佳评测分数**需在 dev/BLiMP 或官方 eval 上选型 checkpoint，而非单凭 train loss。

## 服务器执行顺序

```bash
cd ~/babytext/experiments/gpt-bert-masked-focus/scripts
bash 01_download_baseline.sh   # tokenizer / 可选对照权重
bash 02_setup_gpt_bert_repo.sh
bash 04b_tokenize_babylm_strict.sh   # 生成 .bin
export GPT_BERT_TRAIN_PATH=.../train_strict_strict_tokenized.bin
export GPT_BERT_VALID_PATH=.../valid_strict_strict_tokenized.bin
export RUN_DIR=/data0/language/babylm_runs/gpt_bert_scratch_strict
unset GPT_BERT_MAX_STEPS  # use Strict words x 10 epochs budget conversion
bash 06_train_single_gpu_torchrun.sh
```

## 评测与提交

`--backend mntp`（非 `causal`）。参见 `babylm-eval/strict`。

## 脚本索引

| 文件 | 作用 |
|------|------|
| `09_patch_train_100m_dataloader.py` | 修复 iterator 后 wandb 取 `dataset` |
| `10_patch_train_100m_wandb_stub.py` | `WANDB_DISABLED` 时 stub wandb |
| `06_train_single_gpu_torchrun.sh` | 单卡主入口（`RUN_DIR` 默认 scratch 路径，可改） |

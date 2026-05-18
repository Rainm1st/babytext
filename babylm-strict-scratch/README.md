# BabyLM 2026 · Strict · 从头训练（官方语料）

本目录与 `experiments/gpt2-masked-objective/` 解耦：用于**仅官方 Strict 语料**、**随机初始化**的因果 LM 训练脚本与配置；**不**依赖 BabyLM baseline 权重或 OpenAI GPT‑2 预训练权重（仅使用 HF 上的 `gpt2` 词表与 `gpt2` 架构 `config`，`AutoModelForCausalLM.from_config` 随机初始化）。

## 为何默认选 GPT‑2 small（`scratch_architecture: gpt2`）

| 取舍 | 说明 |
|------|------|
| **与 leaderboard 对齐** | 官方 strict baseline 使用该量级；分数可比、论文好写对照实验。 |
| **容量 vs 数据** | 约 124M 参数相对 100M **词**量级的语料属于常见量级；过小易欠拟合评测，过大更难训稳、调参成本高。 |

若已有稳定训练栈，可在**不改变 epoch≤10** 前提下尝试 `gpt2-medium`（更难训、算力要求高），但必须重新搜学习率/Warmup/序列长，并做好崩训排查。

## 训练策略（实用优先）

1. **优化器与学习率**：`AdamW + weight_decay≈0.1`；起点 **`lr≈3e-4`**，搭配 **cosine** 与 **`warmup_ratio≈6%`**（与往届 BabyLM 常见设定同量级）。若 loss NaN/spike：**先降 lr（如 1e‑4）、关混合精度试试、再检查 batch/序列长**。
2. **序列长 `max_sequence_length`**：256–512 是最容易跑通且不爆显存的区间；若显存够可酌情增大（吞吐与上下文相关）。
3. **Batch**：在显存允许下适当增大 **`batch_size_per_device`**，并用 **`gradient_accumulation_steps`** 维持有效 batch，通常有利于稳定。
4. **数据来源混合**：建议使用配置里的 **`source_proportional_batches` + `source_word_counts`**，使每个 batch 内各源比例接近官方子语料的词数占比，避免因 shuffle 抽样偏差带来不可比或难复现。
5. **规则**：脚本强制 **`num_train_epochs ≤ 10`**（竞赛条目）；语料仅用官方解压目录中的 **`*.train.txt`**。
6. **评测**：模型保存于 `final/`，上传到 Hugging Face 后跑 [babylm-eval](https://github.com/babylm-org/babylm-eval) `strict/` 全流程，再上 [BabyLM Leaderboard 2026](https://huggingface.co/spaces/BabyLM-community/BabyLM-Leaderboard-2026)。

**所谓「最优」**：在竞赛预算下没有单一最优；默认配置是「可跑、对齐 baseline 规模」的起点，**请以 dev/BLiMP 子集或小步长 trial 做小网格（lr × seq_len × warmup）**，再长跑 10 epoch。

## 服务器 usage

```bash
# 编辑 configs/strict_gpt2_default.json ：training_data_path、cuda_visible_devices、batch、use_wandb 等

chmod +x run_train.sh
./run_train.sh configs/strict_gpt2_default.json

# 默认输出到 /data0/language/babylm_runs/gpt2_strict_scratch/run_时间戳
# 可用第二个参数指定输出目录，断点续训需使用同一个输出目录
./run_train.sh configs/strict_gpt2_default.json /data0/language/babylm_runs/gpt2_strict_scratch/my_run --resume
```

或直接：

```bash
python train_strict_lm.py --config configs/strict_gpt2_default.json --output-dir /data0/language/babylm_runs/gpt2_strict_scratch/my_run \
  --num-workers 4 --logging-steps 50

python train_strict_lm.py --config configs/strict_gpt2_default.json --output-dir /data0/language/babylm_runs/gpt2_strict_scratch/my_run \
  --resume-from /data0/language/babylm_runs/gpt2_strict_scratch/my_run --num-workers 4 --logging-steps 50
```

默认产出位于 `/data0/language/babylm_runs/gpt2_strict_scratch/.../`：
`final/`（上传 HF）、`checkpoints/`（按 epoch / step）、`training_state.pt`（优化器+步数，用于 `--resume-from`）。

可用环境变量覆盖默认根目录：

```bash
export GPT2_STRICT_RUN_BASE=/data0/language/babylm_runs/gpt2_strict_scratch
```

### 进度与日志

- **进度条**：每个 epoch 使用 `tqdm`（`datasets.map` 阶段也有 `desc` 进度）。
- **W&B**：配置中 `use_wandb: true` 时记录 loss/lr/step（需 `pip install wandb`）；默认 `false`。
- **断点**：每个 epoch 结束写入 `training_state.pt`；`checkpoint_save_every_steps>0` 时按步保存模型并更新 state（支持同 epoch 内恢复）。


## 与「接续 baseline」的区别

随机初始化等价于：**所有任务监督信号仅来自当前 100M 词官方语料的 ≤10 epoch 训练**。任何 `from_pretrained(<babylm 或 openai-gpt2 权重>)` 再训都会对「预训练数据来源」注入额外信息，**不再等同于 strict 从零设定**。

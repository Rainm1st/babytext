# babytext

BabyLM 2026 STRICT track 实验仓库。当前主线目标是：**使用 BabyLM STRICT text-only 语料，从空白模型开始预训练一个可提交的干净模型**。

本仓库主要用于维护训练脚本、配置、评估记录和实验说明；大语料、tokenized 二进制文件、模型权重和 checkpoint 不进入 Git。

GitHub: [Rainm1st/babytext](https://github.com/Rainm1st/babytext)

---

## 当前方向

我们现在准备从头训练空白模型，而不是继续训练官方 Hugging Face baseline。

这点很重要：

- 官方 `BabyLM-community/*baseline*` 模型已经训练过，不能当作 clean from-scratch STRICT 提交的初始化权重。
- clean STRICT run 应该只从模型结构配置初始化权重，例如 `GPT2Config(...) -> GPT2LMHeadModel(config)`，不能 `from_pretrained(...)` 加载已训练 baseline。
- tokenizer 也要谨慎：最稳妥路线是只用允许的 STRICT text 训练或选择 tokenizer，并记录来源。
- 旧的 continued-pretraining 脚本仍可作为实验参考，但需要明确标成“continued pretraining from official baseline”，不要和正式 from-scratch 提交混在一起。

---

## 比赛约束备忘

目标赛道：

- BabyLM 2026 STRICT track
- text-only training
- 训练语料不超过 100M words
- leaderboard-eligible exposure 不超过 1B words seen

实践上通常意味着：

- 100M words 语料最多训练 10 epochs
- 所有训练曝光量要按 `words seen` 记录
- 需要保存中间 checkpoint：
  - 1M、2M、...、10M words
  - 20M、30M、...、100M words
  - 200M、300M、...、1B words

最终提交前要能说明：

- 模型是从随机初始化开始的
- tokenizer 和训练数据没有超出 STRICT text-only 范围
- 总训练曝光量没有超过 1B words seen
- 使用了哪些 checkpoint 作为验证、评估和最终提交模型

---

## 推荐 from-scratch 主线

### 1. 准备 STRICT 语料

服务器上确认存在：

```text
/home/language/babytext/experiments/BabyLM-2026-Strict/
```

该目录应包含 BabyLM 2026 STRICT 的 text-only 文件，例如多个 `*.train.txt`。语料本身不提交到 Git。

### 2. 准备 tokenizer

干净方案：

```text
BabyLM-2026-Strict text -> train tokenizer -> tokenizer.json
```

注意：

- tokenizer 训练输入只能来自允许使用的 STRICT text
- tokenizer 输出建议放到服务器运行目录，例如 `/data0/language/babylm_runs/.../tokenizer/`
- `tokenizer.json`、tokenizer config 可以记录来源和脚本，但大文件是否提交要按实际大小决定

### 3. 初始化空白模型

GPT-2 风格的参考结构：

| 项目 | 参考值 |
| --- | --- |
| model type | GPT-2 causal LM |
| vocab size | 约 16,384，需和 tokenizer 对齐 |
| layers | 12 |
| hidden size | 768 |
| heads | 12 |
| parameters | 约 98M |

关键要求：

```python
model = GPT2LMHeadModel(GPT2Config(...))
```

而不是：

```python
model = GPT2LMHeadModel.from_pretrained(...)
```

### 4. 训练预算和 checkpoint

建议把训练停止条件设为：

```text
max_words_seen <= 1_000_000_000
num_train_epochs <= 10
```

checkpoint 里程碑建议显式配置，不只按 step 保存：

```text
1M, 2M, ..., 10M
20M, 30M, ..., 100M
200M, 300M, ..., 1B
```

### 5. 评估

BabyLM 评估理解：

- Validation：masked token cross-entropy loss
- Zero-shot：BLiMP、BLiMP Supplement、EWoK、Entity Tracking、WUGs 等直接打分
- Eye Tracking / Self-paced Reading：看模型预测对心理语言学指标的解释变化
- Finetuning：从预训练 checkpoint 初始化，再在 MNLI、BoolQ、MultiRC、WSC、MRPC、QQP 等任务上微调评估

Finetuning 属于评估流程，不计入预训练 exposure budget。

---

## 仓库结构

```text
babytext/
├── README.md
├── PROJECT_CONTEXT.md
├── DATA.md
├── requirements.txt
├── docs/
│   ├── PROGRESS.md
│   ├── PROJECT_STATUS.html
│   └── STRUCTURE.md
├── gpt2-masked-objective/
│   ├── train_hybrid_gpt2.py
│   ├── run_train.py
│   ├── run_train.ps1
│   ├── configs/
│   ├── scripts/
│   ├── results/
│   └── outputs/
├── gpt-bert-masked-focus/
│   ├── README.md
│   └── scripts/
├── 训练日志/
├── training_distribution_report.md
└── training_distribution_report_run4.html
```

### 重点目录

| 路径 | 说明 |
| --- | --- |
| `PROJECT_CONTEXT.md` | 当前项目上下文和服务器路径备忘 |
| `DATA.md` | 官方语料下载与放置说明 |
| `docs/PROGRESS.md` | 历史实验进度看板 |
| `gpt2-masked-objective/` | GPT-2 masked objective 实验代码；目前配置多为从 baseline 继续训练，需要改造后才能 clean from-scratch |
| `gpt-bert-masked-focus/` | GPT-BERT continued-pretraining 参考流水线；当前不作为 clean from-scratch 主线 |
| `训练日志/` | HTML 实验记录 |

---

## 本地和服务器路径

本地 Windows 仓库：

```text
D:\code\Repository_GitHub\babytext
```

服务器推荐布局：

```text
/home/language/babytext/
├── experiments/                 # 本 Git 仓库 / 实验根目录
├── BabyLM-2026-Strict/          # 可选：官方 STRICT 语料，不入 Git
├── babylm-eval/                 # 官方评估仓库 clone，不入 Git
└── gpt-bert/                    # GPT-BERT 官方训练仓库 clone，不入 Git
```

当前上下文中使用的服务器路径：

```text
/home/language/babytext/experiments/
/home/language/babytext/experiments/BabyLM-2026-Strict/
/home/language/babytext/experiments/gpt-bert/
/data0/language/babylm_runs/
```

本地主要用于编辑脚本和整理文档；实际长训练应在服务器 CUDA/PyTorch 环境中运行。长跑前先确认服务器 Python、CUDA、PyTorch 和磁盘输出路径。

---

## 现有脚本状态

### GPT-2

目录：

```text
gpt2-masked-objective/
```

已有脚本：

```text
train_hybrid_gpt2.py
run_train.py
run_train.ps1
configs/run_mask5.json
configs/run_mask10.json
configs/run_mask20.json
configs/run_mask50.json
scripts/train_mask10_run4.sh
scripts/train_mask10_run5.sh
```

当前注意点：

- 多数 config 仍包含 `starting_checkpoint`，指向官方 GPT-2 baseline。
- clean from-scratch 前，需要新增或修改训练入口，让模型从 config 随机初始化。
- masked objective 可以继续作为实验变量，但要先保证初始化、tokenizer、语料和曝光预算都符合 STRICT 规则。

### GPT-BERT

目录：

```text
gpt-bert-masked-focus/
```

已有脚本主要服务于官方 GPT-BERT masked-focus baseline 的下载、tokenize 和继续训练：

```text
scripts/01_download_baseline.sh
scripts/02_setup_gpt_bert_repo.sh
scripts/03_verify_model_load.py
scripts/04b_tokenize_babylm_strict.sh
scripts/06_train_single_gpu_torchrun.sh
```

当前注意点：

- 这条线默认依赖官方 baseline 或官方 GPT-BERT 训练仓库。
- 如果要 clean from-scratch，需要确认 GPT-BERT repo 是否支持只从 architecture config 初始化，并确保不加载训练好的 HF 权重。

---

## 不提交到 Git 的内容

`.gitignore` 已经排除大部分大文件和外部 clone。原则如下：

| 不入库 | 原因 |
| --- | --- |
| BabyLM 官方 `*.train.txt` | 官方语料，需自行下载 |
| `*.bin` | tokenized 二进制数据很大 |
| checkpoint / final model | 训练输出很大 |
| `*.safetensors` | 模型权重不入库 |
| `models--*/`、`snapshots/` | HF cache / 下载模型 |
| `gpt-bert/`、`babylm-eval/` | 第三方 clone |
| `wandb/`、`.env`、token 文件 | 运行缓存和密钥 |

可以提交：

- 训练脚本
- 小型配置文件
- 评估表格模板
- 训练日志和分析文档
- README / project docs

---

## 近期 TODO

从现在的目标看，优先级建议这样排：

- [ ] 新增 GPT-2 from-scratch 配置，例如 `configs/run_scratch_gpt2.json`
- [ ] 修改训练入口，支持 `init_from = "scratch"`，从 `GPT2Config` 初始化
- [ ] 决定 tokenizer 路线：STRICT-only tokenizer 或可解释的固定 tokenizer
- [ ] 配置完整 `checkpoint_words` 里程碑
- [ ] 在服务器跑一个极小 smoke test，确认空白初始化、数据加载、loss、保存 checkpoint 都正常
- [ ] 再启动正式 1B words seen 预算内训练
- [ ] 用官方 BabyLM eval 跑 zero-shot 和 finetuning

---

## 参考链接

- [BabyLM Challenge](https://babylm.github.io/)
- [BabyLM-community/babylm-baseline-100m-gpt2](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt2)
- [BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus)
- [ltgoslo/gpt-bert](https://github.com/ltgoslo/gpt-bert)

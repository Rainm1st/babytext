# babytext

**BabyLM 2026 · Strict 赛道** — 实验代码与工作进度（本仓库 = `experiments/` 目录全部内容）。

GitHub: [Rainm1st/babytext](https://github.com/Rainm1st/babytext)

---

## 一眼看懂

| 看什么 | 文件 |
|--------|------|
| **进度看板** | [docs/PROGRESS.md](docs/PROGRESS.md) |
| **完整总览** | [docs/PROJECT_STATUS.html](docs/PROJECT_STATUS.html)（浏览器打开） |
| **GPT-2 代码** | [gpt2-masked-objective/](gpt2-masked-objective/) |
| **GPT-BERT 代码** | [gpt-bert-masked-focus/](gpt-bert-masked-focus/) |
| **实验笔记** | [训练日志/](训练日志/) |

### 当前在做什么

| 主线 | 状态 |
|------|------|
| **GPT-BERT** Masked Focus 继续预训练（`mntp`） | smoke ✅，全量长训与官方评测待做 |
| **GPT-2** mask10 run3 | 训练 + 官方 finetune ✅，可提交 |
| **GPT-2** run4 任务聚焦语料 | 训练 ✅，官方评测待补 |
| run5 教师增广 | 脚本就绪，未开训 |

👉 分数与待办：[docs/PROGRESS.md](docs/PROGRESS.md)

---

## 本仓库目录

```
babytext/                          ← GitHub 根目录 = 本地 experiments/
├── README.md                      ← 本文件
├── DATA.md                        ← 官方语料下载说明（语料本身不入库）
├── docs/
│   ├── PROGRESS.md
│   └── PROJECT_STATUS.html
├── gpt2-masked-objective/         # GPT-2 + 混合 mask（α 消融、run3/4/5）
├── gpt-bert-masked-focus/         # GPT-BERT 继续预训练流水线
├── 训练日志/                       # HTML 实验记录
├── training_distribution_report.md
└── training_distribution_report_run4.html
```

---

## 不上传什么

| ❌ 不入库 | 说明 |
|----------|------|
| BabyLM 官方 `*.train.txt` | 见 [DATA.md](DATA.md) 自行下载 |
| `*.bin`、大 checkpoint、HF 权重 | 在服务器生成；见 `.gitignore` |
| `gpt-bert/`、`babylm-eval/` 克隆 | 用各目录下 `scripts/02_*.sh` 安装 |

**其余**（代码、配置、HTML 日志、分析报告）**都会上传**。

---

## 服务器目录关系（备忘）

Git 只跟踪本仓库；服务器上通常还有同级目录：

```
~/babytext/
├── experiments/          ← 本仓库 clone 到这里（或即为 babytext 根）
├── BabyLM-2026-Strict/   ← 官方语料（不在 Git 里）
├── babylm-eval/          ← 评测（clone，不在 Git 里）
└── gpt-bert/             ← 官方训练仓库（clone，不在 Git 里）
```

配置里的绝对路径（如 `training_data_path`）仍指向 `~/babytext/experiments/BabyLM-2026-Strict/`，与 clone 方式一致即可。

---

## 快速开始

### GPT-2

```powershell
cd gpt2-masked-objective
python prepare_experiment.py --auto-download-model
cd ..
powershell -File gpt2-masked-objective/run_train.ps1 -Model mask10
```

### GPT-BERT

```bash
cd gpt-bert-masked-focus/scripts
bash 01_download_baseline.sh && bash 02_setup_gpt_bert_repo.sh
bash 04b_tokenize_babylm_strict.sh
python 09_patch_train_100m_dataloader.py
bash 06_train_single_gpu_torchrun.sh
```

---

## 推送到 GitHub

在 **`experiments` 文件夹内** 初始化 Git（仓库内容 = 该文件夹下所有文件）：

```powershell
cd D:\AAAprojects\babylava\BabyLM-community\experiments
git init
git add .
git status
git commit -m "BabyLM Strict: experiment code and progress docs"
git branch -M main
git remote add origin https://github.com/Rainm1st/babytext.git
git push -u origin main
```

---

## 比赛链接

- [BabyLM Challenge](https://babylm.github.io/)
- [babylm-baseline-100m-gpt2](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt2)
- [babylm-baseline-100m-gpt-bert-masked-focus](https://huggingface.co/BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus)
- [ltgoslo/gpt-bert](https://github.com/ltgoslo/gpt-bert)

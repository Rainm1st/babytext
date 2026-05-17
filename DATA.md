# 官方数据集（不入库）

本 Git 仓库**不包含** BabyLM 官方语料文件。请自行下载后放到服务器/本机，并在配置 JSON 或环境变量里指向该路径。

## 需要的数据

| 数据 | 用途 | 典型路径（服务器） |
|------|------|-------------------|
| **BabyLM-2026-Strict** | GPT-2 训练、GPT-BERT tokenize | `~/babytext/BabyLM-2026-Strict/` 或 `~/babytext/experiments/../BabyLM-2026-Strict/` |
| HF baseline 权重 | 训练起点 | `huggingface-cli download` 或 `01_download_baseline.sh` |

## 下载 Strict 语料

1. 从 [BabyLM-community](https://github.com/BabyLM-community) / 比赛页面获取 **BabyLM-2026-Strict**。
2. 解压后应包含多个 `*.train.txt`（及可选 `*.dev.txt`）。
3. 与 `gpt2-masked-objective/configs/*.json` 里 `training_data_path` 保持一致。

示例（仅当你把语料放在仓库**同级**目录时）：

```
~/babytext/
├── experiments/              ← 本 Git 仓库
└── BabyLM-2026-Strict/     ← 语料（不要 git add）
    ├── *.train.txt
    └── ...
```

若语料在 `~/babytext/experiments/BabyLM-2026-Strict/`，也可；只要配置路径正确且**不要** `git add` 语料文件即可。

## GPT-BERT 的 tokenized 数据

由 `gpt-bert-masked-focus/scripts/04b_tokenize_babylm_strict.sh` 在服务器生成 `.bin`（默认 `/data0/language/...`），同样**不要**提交到 GitHub。

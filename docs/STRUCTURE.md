# 目录说明（仓库根 = `experiments/`）

## gpt2-masked-objective/

| 路径 | 说明 |
|------|------|
| `train_hybrid_gpt2.py` | 混合损失训练主程序 |
| `configs/run_mask*.json` | 各实验配置（含服务器绝对路径） |
| `scripts/train_mask10_run4.sh` 等 | run4/run5 数据与训练 |
| `outputs/` | 训练输出（checkpoint 被 gitignore，日志可提交） |
| `results/` | 评测表格模板 |

## gpt-bert-masked-focus/

| 脚本 | 说明 |
|------|------|
| `01_download_baseline.sh` | HF 权重 |
| `04b_tokenize_babylm_strict.sh` | Strict → `.bin` |
| `06_train_single_gpu_torchrun.sh` | 单卡长训 |
| `08_check_token_ids.py` | 词表与 bin 一致性 |
| `09_patch_train_100m_dataloader.py` | 官方 train_100m 补丁 |

## 训练日志/

按日期的 HTML 记录，与 `docs/PROGRESS.md` 互补。

## 服务器数据路径（不入库）

| 用途 | 路径 |
|------|------|
| run4 | `/data0/language/babylm_runs/mask10_run4` |
| GPT-BERT tokenized | `/data0/language/babylm_runs/gpt_bert_strict_tokenized/` |
| 官方评测结果 | `~/babytext/babylm-eval/strict/results/` |

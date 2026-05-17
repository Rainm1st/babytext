# 工作进度看板

> [PROJECT_STATUS.html](PROJECT_STATUS.html) · [STRUCTURE.md](STRUCTURE.md) · [../README.md](../README.md)

**更新**：2026-05-16

---

## 当前策略

| 主线 | 状态 | 说明 |
|------|------|------|
| **B — GPT-BERT** | 🟡 进行中 | smoke OK → 全量长训 + `mntp` 评测 |
| **A — GPT-2 mask10** | 🟢 阶段完成 | run3 官方 finetune 完成 |
| **A′ — run5** | ⚪ 未开始 | 脚本在 `gpt2-masked-objective/scripts/` |

---

## 实验进度

| ID | 内容 | 训练 | 官方评测 | 下一步 |
|----|------|:----:|:--------:|--------|
| mask5/10/20/50 | α 消融 | ✅ | 部分 | mask10 为较优档 |
| **run3** | `mask10_rerun_best` | ✅ | ✅ | `collate_preds --backend causal` |
| **run4** | mnli/rte/wsc/qqp 聚焦语料 | ✅ | ⚠️ 快速 only | 补 strict 官方评测 |
| run5 | 教师增广 | ⬜ | ⬜ | 见 `训练日志/run5_teacher_data_plan.html` |
| **GPT-BERT** | Strict 继续预训练 | 🟡 | ⬜ | `base.json` + patch 已就绪 |

---

## run3 vs run4（参考）

| 任务 | run3 官方 | run4 快速 (n=500) |
|------|-----------|-------------------|
| MNLI | 35.7% | 34.2% |
| RTE | 54.0% | 52.7% |
| QQP | 62.8% | 62.0% |
| WSC | 61.5% | 61.5% |

---

## GPT-BERT 已修复问题

| 问题 | 修复 |
|------|------|
| CUDA assert | `base.json` (16384) 替代 `small.json` (8192) |
| wandb 崩溃 | `09_patch_train_100m_dataloader.py` |

---

## 待办

- [ ] P0 GPT-BERT 全量长训 + `mntp` 评测提交
- [ ] P1 run4 官方 strict 对比 run3
- [ ] P1 run3 提交 leaderboard（若未交）
- [ ] P2 run5 教师增广

---

## 本仓库上传范围

| ✅ 上传 | ❌ 不上传 |
|--------|----------|
| 本目录全部代码、`训练日志/*.html`、`docs/` | 官方 `*.train.txt` |
| `training_distribution_report*` | `*.bin`、checkpoint、HF 权重 |

语料：[DATA.md](../DATA.md)

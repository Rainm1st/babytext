#!/usr/bin/env bash
# 非 SLURM 集群上单机/单卡跑 gpt-bert 官方 train_100m.py 时补齐环境变量。
export SLURM_PROCID="${SLURM_PROCID:-0}"
export SLURM_LOCALID="${SLURM_LOCALID:-0}"
export SLURM_NTASKS="${SLURM_NTASKS:-1}"
export SLURM_NNODES="${SLURM_NNODES:-1}"
export SLURM_NODEID="${SLURM_NODEID:-0}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-local}"
export SLURM_STEP_ID="${SLURM_STEP_ID:-0}"
export SLURM_GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-1}"
export SLURM_GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-1}"
export SLURM_JOB_NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export LOCAL_RANK="${LOCAL_RANK:-0}"

# 无 wandb 账号时可离线/禁用
export WANDB_MODE="${WANDB_MODE:-offline}"

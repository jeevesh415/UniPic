#!/bin/bash
set -x
# Consistency Model + Distribution-Matching Distillation Combined Training Script
#
# Usage:
#   cd UniPic-3
#   bash qwen_image_edit_fast/scripts/train_64.sh
#
# Or with custom CODE_DIR:
#   CODE_DIR=/path/to/UniPic-3 bash qwen_image_edit_fast/scripts/train_64.sh

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}}"

# Enter project directory & set PYTHONPATH
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# Launch training
torchrun \
    --nnodes ${MLP_WORKER_NUM:-1} \
    --node_rank ${MLP_ROLE_INDEX:-0} \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --master_addr ${MLP_WORKER_0_HOST:-localhost} \
    --master_port ${MLP_WORKER_0_PORT:-29500} \
    qwen_image_edit_fast/train_cm_dmd.py \
    qwen_image_edit_fast/configs/gemini_all_datasets.py \
    --work_dir ${WORK_DIR:-work_dirs/Qwen-Image-Edit-CM-DMD-full-20k-bsz16-all-datasets} \
    --guidance_scale ${GUIDANCE_SCALE:-6.0} \
    --train_steps ${TRAIN_STEPS:-20000} \
    --ckpt_steps ${CKPT_STEPS:-1000} \
    --accum_steps ${ACCUM_STEPS:-1} \
    --tangent_norm \
    --gradient_checkpointing

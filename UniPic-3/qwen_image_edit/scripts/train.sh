#!/bin/bash
set -x
# Training Script for FSDP
#
# Usage:
#   cd UniPic-3
#   bash qwen_image_edit/train.sh
#
# Or with custom CODE_DIR:
#   CODE_DIR=/path/to/UniPic-3 bash qwen_image_edit/train.sh

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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
    qwen_image_edit/train_fsdp_bsz1.py \
    qwen_image_edit/configs/datasets.py \
    --gradient_checkpointing \
    --train_steps ${TRAIN_STEPS:-100000} \
    --ckpt_steps ${CKPT_STEPS:-2000} \
    --work_dir ${WORK_DIR:-work_dirs/Unipic3-FSDP-all-datasets-step100k-bsz16}

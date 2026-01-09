#!/bin/bash
# Batch Inference Script
#
# Usage:
#   cd UniPic-3
#   bash qwen_image_edit_fast/scripts/inference.sh
#
# Or with custom paths:
#   JSONL_PATH=data/val.jsonl \
#   TRANSFORMER_PATH=work_dirs/model/ema_transformer \
#   OUTPUT_DIR=work_dirs/output \
#   bash qwen_image_edit_fast/scripts/inference.sh

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}}"

# Enter project directory & set PYTHONPATH
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# Set default paths (can be overridden via environment variables)
JSONL_PATH="${JSONL_PATH:-data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64/step20000/4step4}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64/step20000/ema_transformer}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Launch inference
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-1} \
    --master_port ${MASTER_PORT} \
    --use_env \
    qwen_image_edit_fast/batch_inference.py \
    --jsonl_path "${JSONL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --distributed \
    --num_inference_steps ${NUM_INFERENCE_STEPS} \
    --true_cfg_scale ${TRUE_CFG_SCALE} \
    --transformer "${TRANSFORMER_PATH}" \
    --skip_existing

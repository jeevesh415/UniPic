#!/bin/bash
# Single Image Inference Script
#
# Usage:
#   cd UniPic-3
#   bash qwen_image_edit/scripts/inference.sh
#
# Or with custom paths:
#   TRANSFORMER_PATH=work_dirs/model/transformer \
#   IMAGE_PATHS=example/image1.png example/image2.png \
#   PROMPT="Your prompt here" \
#   bash qwen_image_edit/scripts/inference.sh

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}}"

# Enter project directory & set PYTHONPATH
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# Set default paths (can be overridden via environment variables)
TRANSFORMER_PATH="${TRANSFORMER_PATH:-work_dirs/Qwen-Image-Edit-FSDP_only_2and3imgs/transformer}"
IMAGE_PATHS="${IMAGE_PATHS:-qwen_image_edit/example/gemini_pig_remove_hat.png qwen_image_edit/example/gemini_t2i_sunglasses.png}"
PROMPT="${PROMPT:-A pig wearing sunglasses.}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
SEED="${SEED:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-output.png}"

# Launch inference
python qwen_image_edit/inference.py \
    --transformer "${TRANSFORMER_PATH}" \
    --image_paths ${IMAGE_PATHS} \
    --prompt "${PROMPT}" \
    --true_cfg_scale ${TRUE_CFG_SCALE} \
    --seed ${SEED} \
    --output_path "${OUTPUT_PATH}"

#!/bin/bash
# Batch Inference Script
# 1) 安装依赖（如果尚未安装）
# pip install -r requirements.txt

# 2) 设置代码目录（可通过环境变量 CODE_DIR 配置，默认为当前目录）
# 使用方式: CODE_DIR=/path/to/project JSONL_PATH=data/val.jsonl bash inference.sh
CODE_DIR="${CODE_DIR:-$(pwd)}"

# 3) 进入代码目录 & PYTHONPATH
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# 4) 设置默认路径（可通过环境变量覆盖）
JSONL_PATH="${JSONL_PATH:-data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64/step20000/4step4}"
TRANSFORMER_PATH="${TRANSFORMER_PATH:-work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64/step20000/ema_transformer}"

# 5) 启动推理
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 --use_env \
    qwen_image_edit_fast/batch_inference.py \
    --jsonl_path "${JSONL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --distributed \
    --num_inference_steps 4 \
    --true_cfg_scale 4.0 \
    --transformer "${TRANSFORMER_PATH}" \
    --skip_existing


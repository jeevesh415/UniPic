#!/bin/bash
set -x
# Distribution-Matching Distillation Training Script
# 1) 安装依赖（如果尚未安装）
# pip install -r requirements.txt

# 2) 设置代码目录（可通过环境变量 CODE_DIR 配置，默认为当前目录）
# 使用方式: CODE_DIR=/path/to/project bash train_dmd.sh
CODE_DIR="${CODE_DIR:-$(pwd)}"

# 3) 进入代码目录 & PYTHONPATH
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

# 4) 启动训练
torchrun \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --nproc_per_node 8 \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
    qwen_image_edit_fast/train_dmd.py \
    qwen_image_edit_fast/configs/gemini_all_datasets.py \
    --work_dir work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64 \
    --guidance_scale 6.0 \
    --lr_scheduler cosine \
    --train_steps 20000 \
    --ckpt_steps 1000 \
    --accum_steps 1 \
    --gradient_checkpointing


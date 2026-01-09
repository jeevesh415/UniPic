set -x
# 1) 激活 conda
source /data_genie/genie/hongbo/miniconda3/etc/profile.d/conda.sh
conda activate self_forcing

# 2) 进入代码目录 & PYTHONPATH
cd /data_genie/genie/yaoshu/projects/memory/Qwen-Image-Dev-main
export PYTHONPATH="/data_genie/genie/yaoshu/projects/memory/Qwen-Image-Dev-main:${PYTHONPATH:-}"

torchrun \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --nproc_per_node 8 \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
    qwen_image_edit_fast/train_cm.py \
    qwen_image_edit_fast/configs/gemini_all_datasets.py \
    --work_dir work_dirs/Qwen-Image-Edit-CM-full-20k-bsz48-all-datasets \
    --guidance_scale 1.5 \
    --train_steps 20000 \
    --ckpt_steps 1000 \
    --accum_steps 2 \
    --ema_rate 0.95 \
    --tangent_norm \
    --gradient_checkpointing       
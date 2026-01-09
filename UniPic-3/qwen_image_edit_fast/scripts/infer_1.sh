# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 --use_env \
#     qwen_image_edit_fast/batch_inference.py \
#     --jsonl_path /data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/all_imgs_val.jsonl \
#     --output_dir work_dirs/Qwen-Image-Edit-DMD-20k-bsz16-all-datasets/train9k_infer4/ \
#     --distributed \
#     --num_inference_steps 4 \
#     --transformer /data_genie/genie/why/Qwen-Image-Dev-main/work_dirs/Qwen-Image-Edit-FSDP-6imgs-all-datasets-step100k-bsz16/step100000 \
#     --lora work_dirs/Qwen-Image-Edit-DMD-20k-bsz16-all-datasets/step9000/transformer_lora/pytorch_model.bin \
#     --skip_existing


# python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 --use_env \
#     qwen_image_edit_fast/batch_inference.py \
#     --jsonl_path /data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/all_imgs_val.jsonl \
#     --output_dir work_dirs/Qwen-Image-Edit-FSDP-Pretrained/infer4/ \
#     --distributed \
#     --num_inference_steps 4 \
#     --transformer /data_genie/genie/why/Qwen-Image-Dev-main/work_dirs/Qwen-Image-Edit-FSDP-6imgs-all-datasets-step100k-bsz16/step100000 \
#     --skip_existing


python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 --use_env \
    qwen_image_edit_fast/batch_inference.py \
    --jsonl_path /data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/all_imgs_val.jsonl \
    --output_dir work_dirs/Qwen-Image-Edit-DMD-full-10k-bsz64-all-datasets2/step0/8_step \
    --distributed \
    --num_inference_steps 8 \
    --true_cfg_scale 0 \
    --transformer work_dirs/Qwen-Image-Edit-DMD-full-10k-bsz64-all-datasets2/step0/transformer \
    --skip_existing



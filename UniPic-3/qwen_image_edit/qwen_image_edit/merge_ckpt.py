import torch

state_dict_cm = torch.load('work_dirs/Qwen-Image-Edit-CM-full-20k-bsz256-all-datasets/step12000/ema_transformer/diffusion_pytorch_model.bin')
state_dict_dmd = torch.load('work_dirs/Qwen-Image-Edit-DMD-full-50k-bsz64-all-datasets/step12000/ema_transformer/diffusion_pytorch_model.bin')

new_state_dict = dict()

for k, v in state_dict_cm.items():
    new_state_dict[k] = (state_dict_cm[k] + state_dict_dmd[k]) / 2

torch.save(new_state_dict, "work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64-all-datasets/step0/ema_transformer/diffusion_pytorch_model.bin")


# state_dict = load_file('UnifiedDCM/workdir/t2i/sit_xl/checkpoints/checkpoint-853000/model_1.safetensors', device="cpu")  # 加载到CPU避免GPU内存占用
# new_state_dict = state_dict  # 一般情况下直接使用原始state_dict
# torch.save(new_state_dict, 'UnifiedDCM/workdir/t2i/sit_xl/checkpoints/checkpoint-853000/pytorch_model_fsdp_1.bin')

# opt = torch.load('UnifiedDCM/workdir/t2i/sit_xl/checkpoints/checkpoint-853000/optimizer.bin')

# for k, v in opt['state'].items():
#     print(k, v)
# print(opt['param_groups'])

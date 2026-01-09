import torch
torch.backends.cuda.matmul.allow_tf32 = True   # it might help
import math
import logging
import os
import time
import argparse
import functools
import torch.nn.functional as F
import torch.distributed as dist

from mmengine.registry import DATASETS
from mmengine.config import Config
from mmengine.dataset.sampler import InfiniteSampler

from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor

# PEFT imports for LoRA
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from qwen_image_edit.pipeline_qwenimage_edit import calculate_shift, QwenImageEditPipeline
from contextlib import nullcontext

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, StateDictType, FullStateDictConfig, MixedPrecision
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


MODEL_NAME = "Qwen-Image-Edit"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate.')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay rate.')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1.')
    parser.add_argument('--beta2', default=0.95, type=float, help='beta2.')

    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='learning rate scheduler.')
    parser.add_argument('--lr_min', default=1e-5, type=float, help='learning rate.')

    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--transformer', default=MODEL_NAME, type=str)
    parser.add_argument('--max_prompt_length', default=2048, type=int)

    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=20, type=int)
    parser.add_argument('--ckpt_steps', default=200, type=int)
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--warmup_steps', default=40, type=int, help='warmup steps (optimizer updates).')

    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)

    parser.add_argument('--work_dir', default='work_dirs/Qwen-Image-Edit-LoRA-FSDP', type=str)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    
    # LoRA 相关参数
    parser.add_argument('--lora_r', default=4, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=4, type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', default='to_q,to_k,to_v,to_out.0', type=str, 
                       help='Target modules for LoRA, comma separated')
    
    args = parser.parse_args()

    # Initialization
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    ddp_rank = dist.get_rank()

    if ddp_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        assert local_rank == 0, 'use torchrun to ensure ddp_rank==0 -> local_rank==0'

    if ddp_rank == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.work_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    dist.barrier()

    assert args.log_steps % args.accum_steps == 0
    assert args.warmup_steps % args.accum_steps == 0

    device_message = f"ddp rank: {ddp_rank}, local rank: {local_rank}, world_size: {world_size}"
    print(device_message)

    config = Config.fromfile(args.config)
    dataset = DATASETS.build(config.dataset)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            sampler=InfiniteSampler(dataset=dataset, shuffle=True,),
                            drop_last=False,
                            collate_fn=lambda x: x
                            )

    device = torch.device(f"cuda:{local_rank}")   # current device

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='scheduler'
    )
    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='text_encoder',
        torch_dtype=torch.bfloat16,
        device_map=None,            # IMPORTANT: don't place full model on one GPU
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='tokenizer',
    )
    processor = Qwen2VLProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='processor',
    )

    # 加载 transformer
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.transformer, subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map=None,                 # 防止单GPU放置
        low_cpu_mem_usage=True           # 减少CPU峰值使用
    )

    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='vae', torch_dtype=torch.bfloat16,
    ).to(device)

    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()

    # wrap text_encoder with fsdp
    text_encoder = FSDP(
        text_encoder,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: isinstance(m, (Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock)),
        ),
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=True,
    )

    # =============== LoRA 配置和应用 ===============
    target_modules = args.lora_target_modules.split(',')
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        # task_type=TaskType.DIFFUSION,  # 或者使用 None
    )
    
    logger.info(f"LoRA 配置: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"LoRA 目标模块: {target_modules}")
    
    # 应用 LoRA 到 transformer
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    pipeline = QwenImageEditPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                                     tokenizer=tokenizer, processor=processor, transformer=None)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    transformer.to(torch.float32)
    transformer.train()
    
    # 获取可训练参数（仅 LoRA 参数）
    trainable_params = list(transformer.parameters())
    logger.info(f"可训练参数数量: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

    # NEW: FSDP wrap (ZeRO-3 style)
    transformer = FSDP(
        transformer,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: isinstance(m, QwenImageTransformerBlock),
        ),
        device_id=device,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=True,
    )

    # =============== 优化器设置（仅 LoRA 参数）===============
    trainable_params = []
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    logger.info(f"优化器参数数量: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(params=trainable_params,
                      lr=args.lr,
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay)

    if args.lr_scheduler == 'constant':
        lr_lambda = lambda x: min(x / args.warmup_steps, 1.0)
    elif args.lr_scheduler == 'cosine':
        def lr_lambda(x):
            if x < args.warmup_steps:
                return x / args.warmup_steps
            else:
                x = (x - args.warmup_steps) / (args.train_steps - args.warmup_steps)
                return max(math.cos(math.pi * x / 2), args.lr_min / args.lr)
    else:
        raise NotImplementedError

    # LambdaLR
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    iterator = iter(dataloader)     # iterator the dataloader
    tik = time.time()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device).bfloat16()
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device).bfloat16()

    for train_step in range(1, args.train_steps + 1):
        data_list = next(iterator)
        assert len(data_list) == 1
        data_dict = data_list[0]

        input_images = data_dict['input_images']
        output_image = data_dict['output_image']

        width, height = output_image.size

        text = data_dict['text']

        prompt_images = [image.resize((round(image.width * 28 / 32), round(image.height * 28 / 32)))
                         for image in input_images]

        latents_list = []
        img_shapes_list = []

        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = pipeline._get_qwen_prompt_embeds(
                [text], [prompt_images], device=device, dtype=torch.bfloat16)
            assert prompt_embeds.shape[0] == 1
            prompt_embeds = prompt_embeds[:, :args.max_prompt_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :args.max_prompt_length]

            for image in [output_image] + input_images:
                pixel_values = pipeline.image_processor.preprocess(
                    image, image.height, image.width).to(dtype=torch.bfloat16, device=device)
                latents = vae.encode(pixel_values[:, :, None], return_dict=False)[0].sample()
                latents = (latents - latents_mean) / latents_std
                assert latents.shape[2] == 1
                assert latents.shape[0] == 1

                img_shapes_list.append(
                    (1, latents.shape[-2] // 2, latents.shape[-1] // 2)
                )
                latents_list.append(latents)

        image_seq_len = math.prod(img_shapes_list[0])
        mu = calculate_shift(
            image_seq_len,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 8192),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 0.9),
        )

        if scheduler.config.time_shift_type == "exponential":
            shift = math.exp(mu)
        elif scheduler.config.time_shift_type == "linear":
            shift = mu
        else:
            raise NotImplementedError

        # sample from logit_norm
        u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(1,), device=device,)
        sigmas = torch.nn.functional.sigmoid(u).bfloat16()
        # shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        # timesteps
        timesteps = sigmas * scheduler.num_train_timesteps

        # unsqueeze
        sigmas = sigmas.view(-1, 1, 1, 1, 1)
        # sample noises
        noises = torch.randn_like(latents_list[0])
        noisy_model_input = (1.0 - sigmas) * latents_list[0] + sigmas * noises

        packed_noisy_model_input = []
        for latents in [noisy_model_input] + latents_list[1:]:
            assert latents.shape[2] == 1
            assert latents.shape[0] == 1

            packed_noisy_model_input.append(
                pipeline._pack_latents(
                    latents.squeeze(2),
                    batch_size=1,
                    num_channels_latents=latents.shape[1],
                    height=latents.shape[3],
                    width=latents.shape[4],
                )
            )

        packed_noisy_model_input = torch.cat(packed_noisy_model_input, dim=1)

        target = noises - latents_list[0]

        update_grad = train_step % args.accum_steps == 0
        with (nullcontext() if update_grad else transformer.no_sync()):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    timestep=timesteps / 1000,
                    img_shapes=[img_shapes_list, ],
                    txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, :image_seq_len]
                model_pred = pipeline._unpack_latents(model_pred, height, width, pipeline.vae_scale_factor)
                loss = F.mse_loss(model_pred.float(), target.float())

            loss.backward()

        lr_scheduler.step()
        if update_grad:
            # CHANGED: use FSDP's clip helper
            grad_norm = transformer.clip_grad_norm_(args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            update_log = train_step % args.log_steps == 0
            if update_log:
                speed = args.log_steps / (time.time() - tik)
                cur_lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"Train steps={train_step}/{args.train_steps}, loss={loss.item():.4f}, "
                            f"grad norm={grad_norm.item():.4f}, speed: {speed:.4f}, LR: {cur_lr:.8f}")
                tik = time.time()

        save_ckpt = train_step % args.ckpt_steps == 0

        if save_ckpt:
            # =============== 保存 LoRA 权重 ===============
            save_dir = os.path.join(args.work_dir, f'lora_step_{train_step}')
            if ddp_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                
                # 获取原始的 PEFT 模型（去掉 FSDP 包装）
                peft_model = transformer.module if hasattr(transformer, 'module') else transformer
                
                # 保存 LoRA 权重
                peft_model.save_pretrained(save_dir)
                logger.info(f"已保存 LoRA 权重到: {save_dir}")
                
                # 可选：也保存配置信息
                import json
                lora_info = {
                    'lora_r': args.lora_r,
                    'lora_alpha': args.lora_alpha,
                    'lora_dropout': args.lora_dropout,
                    'target_modules': target_modules,
                    'train_step': train_step,
                    'loss': loss.item()
                }
                with open(os.path.join(save_dir, 'lora_info.json'), 'w') as f:
                    json.dump(lora_info, f, indent=2)

            dist.barrier()

    logger.info("训练完成！")

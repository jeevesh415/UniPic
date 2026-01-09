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

from qwen_image_edit.pipeline_qwenimage_edit import calculate_shift, QwenImageEditPipeline
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, get_peft_model_state_dict

from accelerate import Accelerator
from torch.distributed.fsdp import (
    BackwardPrefetch, CPUOffload, ShardingStrategy, MixedPrecision, 
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
    CPUOffloadPolicy, ShardedStateDictConfig, ShardedOptimStateDictConfig,
)
from accelerate.utils import FullyShardedDataParallelPlugin
# from torch.nn.parallel import DistributedDataParallel as DDP  # REMOVED: DDP
# NEW: FSDP imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


MODEL_NAME = "Qwen-Image-Edit"




def forward_fn(pipeline, transformer, noisy_model_input, condition_latents, timesteps, prompt_embeds, prompt_embeds_mask, img_shapes_list):
    packed_noisy_model_input = []
    for latents in [noisy_model_input] + condition_latents:
        assert latents.shape[2] == 1
        assert latents.shape[0] == 1
        packed_noisy_model_input.append(
            pipeline._pack_latents(     # B, C, 1, H, W -> B, C, H, W -> B, H/2*W/2, C*4
                latents.squeeze(2),
                batch_size=1,
                num_channels_latents=latents.shape[1],
                height=latents.shape[3],
                width=latents.shape[4],
            )
        )
    packed_noisy_model_input = torch.cat(packed_noisy_model_input, dim=1)
    
    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        timestep=timesteps.flatten(),
        img_shapes=[img_shapes_list, ],
        txt_seq_lens=prompt_embeds_mask.sum(dim=1).tolist(),
        return_dict=False,
    )[0]
    model_pred = model_pred[:, :image_seq_len]
    # B, T*H/2*W/2, C*4 -> B, C, 1, H, W
    model_pred = pipeline._unpack_latents(model_pred, height, width, pipeline.vae_scale_factor)
    return model_pred

@torch.no_grad()
def compute_tangent_fd(pipeline, transformer, clean_latents, condition_latents, timesteps, noises,prompt_embeds, prompt_embeds_mask, img_shapes_list, fd_size):
    def xfunc(_timesteps):
        noisy_model_input = (1.0 - _timesteps) * clean_latents + _timesteps * noises
        return forward_fn(
            pipeline, transformer, noisy_model_input, condition_latents, _timesteps, 
            prompt_embeds, prompt_embeds_mask, img_shapes_list
        )
    fc1_dt = 1 / (2 * fd_size)
    dF_dv_dt = xfunc(timesteps + fd_size) * fc1_dt - xfunc(timesteps - fd_size) * fc1_dt
    return dF_dv_dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')

    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate.')
    parser.add_argument('--lr_fake_score', default=1e-7, type=float, help='learning rate for fake score.')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay rate.')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1.')
    parser.add_argument('--beta2', default=0.95, type=float, help='beta2.')

    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='learning rate scheduler.')
    parser.add_argument('--lr_min', default=1e-7, type=float, help='learning rate.')


    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--transformer', default='/data_genie/genie/why/Qwen-Image-Dev-main/work_dirs/Qwen-Image-Edit-FSDP-6imgs-all-datasets-step100k-bsz16/step100000', type=str)
    parser.add_argument('--max_prompt_length', default=2048, type=int)

    # CM specific settings
    parser.add_argument('--loss_scale', default=1.0, type=float, 
                        help='scale factor for rCM loss')
    parser.add_argument('--fd_size', default=5e-3, type=float, 
                        help='finite difference step size')
    parser.add_argument('--tangent_norm', default=False, action='store_true', 
                        help='normalize tangent vector')
    parser.add_argument('--tangent_clip', default=False, action='store_true', 
                        help='clip tangent vector')
    
    
    # Fake score network (optional, for DMD2)
    parser.add_argument('--guidance_scale', default=2.0, type=float, 
                        help='guidance scale for teacher-LoRA (0 = no CFG)')
    parser.add_argument('--student_update_freq', default=5, type=int,
                        help='update student every N steps')
    parser.add_argument('--loss_scale_dmd', default=0.1, type=float)
    parser.add_argument('--max_simulation_steps', default=4, type=int)

    # LoRA 相关参数
    parser.add_argument('--student_lora_r', default=512, type=int, help='LoRA rank')
    parser.add_argument('--student_lora_alpha', default=512, type=int, help='LoRA alpha')
    parser.add_argument('--lora_r', default=256, type=int, help='LoRA rank')
    parser.add_argument('--lora_alpha', default=256, type=int, help='LoRA alpha')
    parser.add_argument('--lora_dropout', default=0.1, type=float, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', default='to_q,to_k,to_v,to_out.0', type=str, 
                       help='Target modules for LoRA, comma separated')
    
    # Training settings
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=20, type=int)
    parser.add_argument('--ckpt_steps', default=200, type=int)
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--warmup_steps', default=100, type=int, help='warmup steps (optimizer updates).')

    # Time sampling
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)

    parser.add_argument('--work_dir', default='work_dirs/Qwen-Image-Edit-FSDP', type=str)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )


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

    # VAE (frozen)
    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='vae', 
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Student transformer
    logger.info("Loading student model...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.transformer, subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # Teacher-LoRA network (optional)
    logger.info("Loading teacher-LoRA network...")
    teacher_lora = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.transformer, subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True
    )
    # =============== LoRA 配置和应用 ===============
    target_modules = args.lora_target_modules.split(',')
    lora_config = LoraConfig(
        r=args.student_lora_r,
        lora_alpha=args.student_lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    logger.info(f"Student LoRA configuration: r={args.student_lora_r}, alpha={args.student_lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Student LoRA modules: {target_modules}")
    # 应用 LoRA 到 transformer
    transformer.add_adapter(lora_config)
    # =============== LoRA 配置和应用 ===============
    target_modules = args.lora_target_modules.split(',')
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    logger.info(f"LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"LoRA modules: {target_modules}")
    # 应用 LoRA 到 transformer
    teacher_lora.add_adapter(lora_config)

    # Freeze teacher and VAE
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    transformer.train()
    teacher_lora.train()
    

    trainable_params = []
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    logger.info(f"num trainable parameters: {sum(p.numel() for p in trainable_params)}")
    optimizer = AdamW(params=trainable_params,
                      lr=args.lr,
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay)

    trainable_params = []
    for name, param in teacher_lora.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    logger.info(f"num trainable parameters: {sum(p.numel() for p in trainable_params)}")
    optimizer_lora = AdamW(params=trainable_params,
                      lr=args.lr_fake_score,
                      betas=(args.beta1, args.beta2),
                      weight_decay=args.weight_decay) 


    
    # wrap text_encoder with fsdp
    text_encoder = FSDP(
        text_encoder,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        # auto_wrap_policy=size_based_auto_wrap_policy,
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

    pipeline = QwenImageEditPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                                     tokenizer=tokenizer, processor=processor, transformer=None)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.gradient_checkpointing:
        teacher_lora.enable_gradient_checkpointing()

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
    teacher_lora = FSDP(
        teacher_lora,
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
    lr_scheduler_lora = LambdaLR(optimizer_lora, lr_lambda=lr_lambda)
    iterator = iter(dataloader)     # iterator the dataloader
    tik = time.time()

    

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device).bfloat16()
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device).bfloat16()


    
    student_step = 0
    teacher_step = 0
    for train_step in range(1, args.train_steps+1):
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

            null_embeds, null_embeds_mask = pipeline._get_qwen_prompt_embeds(
                [" "], [prompt_images], device=device, dtype=torch.bfloat16)
            assert null_embeds.shape[0] == 1
            null_embeds = null_embeds[:, :args.max_prompt_length]
            null_embeds_mask = null_embeds_mask[:, :args.max_prompt_length]

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
        # timesteps with shift
        timesteps = shift * sigmas / (1 + (shift - 1) * sigmas)
        # unsqueeze
        timesteps = timesteps.view(-1, 1, 1, 1, 1)
        # sample noises
        noises = torch.randn_like(latents_list[0])
                
        if train_step % args.student_update_freq == 0: 
            # ==================== Student Update ====================
            transformer.train()
            teacher_lora.eval()
            
            # cm loss 
            model_tangent = compute_tangent_fd(pipeline, transformer, latents_list[0], latents_list[1:], timesteps, noises, prompt_embeds, prompt_embeds_mask, img_shapes_list, args.fd_size)
            if args.tangent_norm:
                model_tangent = model_tangent.double() / (model_tangent.double().norm(p=2, dim=(1, 2, 3, 4), keepdim=True) + 0.1)
            elif args.tangent_clip:
                model_tangent = torch.clamp(model_tangent, -1, 1)
            else: pass
            target = noises - latents_list[0] - timesteps * model_tangent
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                noisy_model_input = (1.0 - timesteps) * latents_list[0] + timesteps * noises
                model_pred = forward_fn(
                    pipeline, transformer, noisy_model_input, latents_list[1:], timesteps, 
                    prompt_embeds, prompt_embeds_mask, img_shapes_list,
                )
                cm_loss = F.mse_loss(model_pred.float(), target.float())   #  / args.accum_steps
                cm_loss = torch.nan_to_num(cm_loss, nan=0, posinf=1e5, neginf=-1e5)
                # TODO: remove this, just for debugging
                # cm_loss = torch.tensor(0., device=cm_loss.device)
            
            # get predicted original samples
            effective_iterations = train_step // args.student_update_freq
            num_simulation_steps = effective_iterations % args.max_simulation_steps + 1
            t_steps = torch.linspace(1.0, 0.0, num_simulation_steps+1, dtype=torch.float64)
            x_cur = torch.randn_like(latents_list[0])
            with torch.no_grad():
                for t_cur, t_next in zip(t_steps[:-2], t_steps[1:-1]):
                    model_pred = forward_fn(
                        pipeline, transformer, x_cur, latents_list[1:], t_cur,
                        prompt_embeds, prompt_embeds_mask, img_shapes_list,
                    )
                    pred_x_0 = x_cur - t_cur * model_pred   
                    x_cur = (1 - t_next) * pred_x_0 + t_next * torch.randn_like(pred_x_0)   
            timesteps_with_grad = t_steps[-2]
            model_pred = forward_fn(
                pipeline, transformer, x_cur, latents_list[1:], timesteps_with_grad,
                prompt_embeds, prompt_embeds_mask, img_shapes_list,
            )
            pred_original_samples = x_cur - t_cur * model_pred
            
            # dmd loss
            u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(1,), device=device,)
            sigmas = torch.nn.functional.sigmoid(u).bfloat16()
            timesteps = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = timesteps.view(-1, 1, 1, 1, 1)
            noises = torch.randn_like(pred_original_samples)
            noisy_model_input = (1.0 - timesteps) * pred_original_samples + timesteps * noises
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                # TODO: add unconditional prediction from frozen teacher and LoRA teacher
                # frozen teacher without LoRA parameters
                teacher_lora.module.disable_adapters() # disable LoRA for teacher    
                teacher_pred_cond = forward_fn(pipeline, teacher_lora, noisy_model_input, latents_list[1:], timesteps, prompt_embeds, prompt_embeds_mask, img_shapes_list)
                teacher_pred_uncond = forward_fn(pipeline, teacher_lora, noisy_model_input, latents_list[1:], timesteps, null_embeds, null_embeds_mask, img_shapes_list)
                teacher_pred = teacher_pred_uncond + args.guidance_scale * (teacher_pred_cond - teacher_pred_uncond)
                # LoRA teacher with LoRA parameters
                teacher_lora.module.enable_adapters() # enable LoRA for teacher    
                lora_pred_cond = forward_fn(pipeline, teacher_lora, noisy_model_input, latents_list[1:], timesteps, prompt_embeds, prompt_embeds_mask, img_shapes_list)
                lora_pred_uncond = forward_fn(pipeline, teacher_lora, noisy_model_input, latents_list[1:], timesteps, null_embeds, null_embeds_mask, img_shapes_list)
                lora_pred = lora_pred_uncond + args.guidance_scale * (lora_pred_cond - lora_pred_uncond)

            score_gradient = torch.nan_to_num(teacher_pred - lora_pred, nan=0, posinf=1e5, neginf=-1e5)
            target = (pred_original_samples - score_gradient).detach()
            dmd_loss = 0.5 * F.mse_loss(pred_original_samples.float(), target.float())
            dmd_loss = torch.nan_to_num(dmd_loss, nan=0, posinf=1e5, neginf=-1e5)
            
            
            student_step += 1
            sync_now = student_step % args.accum_steps == 0
            sync_ctx = nullcontext() if sync_now else transformer.no_sync()
            with sync_ctx:
                total_loss = args.loss_scale * cm_loss + args.loss_scale_dmd * dmd_loss
                total_loss = torch.nan_to_num(total_loss, nan=0, posinf=1e5, neginf=-1e5)
                total_loss = total_loss / args.accum_steps
                total_loss.backward()
            if sync_now:
                grad_norm = transformer.clip_grad_norm_(args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
        else:
            # ==================== Teacher-LoRA Update ====================
            transformer.eval()
            teacher_lora.train()

            # obtain predicted original samples
            effective_iterations = train_step // args.student_update_freq
            num_simulation_steps = effective_iterations % args.max_simulation_steps + 1
            t_steps = torch.linspace(1.0, 0.0, num_simulation_steps+1, dtype=torch.float64)
            x_cur = torch.randn_like(latents_list[0])
            with torch.no_grad():
                for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
                    model_pred = forward_fn(
                        pipeline, transformer, x_cur, latents_list[1:], t_cur,
                        prompt_embeds, prompt_embeds_mask, img_shapes_list,
                    )
                    pred_x_0 = x_cur - t_cur * model_pred   
                    x_cur = (1 - t_next) * pred_x_0 + t_next * torch.randn_like(pred_x_0)   
                pred_original_samples = pred_x_0
            # dmd loss for teacher-LoRA
            u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(1,), device=device,)
            sigmas = torch.nn.functional.sigmoid(u).bfloat16()
            timesteps = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = timesteps.view(-1, 1, 1, 1, 1)
            noises = torch.randn_like(pred_original_samples)
            noisy_model_input = (1.0 - timesteps) * pred_original_samples + timesteps * noises
            
            target = noises - pred_original_samples
            teacher_pred = forward_fn(
                pipeline, teacher_lora, noisy_model_input, latents_list[1:], timesteps, 
                prompt_embeds, prompt_embeds_mask, img_shapes_list,
            )
            lora_loss = F.mse_loss(teacher_pred.float(), target.float())
            lora_loss = torch.nan_to_num(lora_loss, nan=0, posinf=1e5, neginf=-1e5)

            teacher_step += 1
            sync_now = teacher_step % args.accum_steps == 0
            sync_ctx = nullcontext() if sync_now else teacher_lora.no_sync()
            with sync_ctx:
                scaled_lora_loss = lora_loss / args.accum_steps
                scaled_lora_loss.backward()
            if sync_now:
                grad_norm_lora = teacher_lora.clip_grad_norm_(args.grad_clip)
                optimizer_lora.step()
                lr_scheduler_lora.step()
                optimizer_lora.zero_grad(set_to_none=True)
                

        update_log = train_step % args.log_steps == 0
        if update_log:
            speed = args.log_steps / (time.time() - tik)
            cur_lr = lr_scheduler.get_last_lr()[0]
            logger.info(f"Train steps={train_step}/{args.train_steps}, Student_step={student_step}, Teacher_step={teacher_step}, "
                        f"cm_loss={cm_loss.item():.6f}, dmd_loss={dmd_loss.item():.6f}, lora_loss={lora_loss.item():.6f}, "
                        f"grad norm={grad_norm.item():.6f}, lora grad norm={grad_norm_lora.item():.6f}, speed: {speed:.4f}, LR: {cur_lr:.8f}")
            tik = time.time()
        
        save_ckpt = train_step % args.ckpt_steps == 0
        if save_ckpt:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, cfg):
                student_state = transformer.state_dict()

            with FSDP.state_dict_type(teacher_lora, StateDictType.FULL_STATE_DICT, cfg):
                teacher_state = teacher_lora.state_dict()

            if ddp_rank == 0:
                save_dir = os.path.join(args.work_dir, f"step{train_step}")
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(os.path.join(save_dir, "transformer_lora"), exist_ok=True)
                os.makedirs(os.path.join(save_dir, "teacher_lora"), exist_ok=True)

                lora_only = get_peft_model_state_dict(transformer.module, state_dict=student_state)
                torch.save(lora_only, os.path.join(save_dir, "transformer_lora", "pytorch_model.bin"))

                lora_only = get_peft_model_state_dict(teacher_lora.module, state_dict=teacher_state)
                torch.save(lora_only, os.path.join(save_dir, "teacher_lora", "pytorch_model.bin"))
                logger.info(f"Saved checkpoint to {save_dir}")


            del student_state
            del teacher_state
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
            
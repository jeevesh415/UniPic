"""
rCM Distillation Training Script for Qwen-Image-Edit

This script implements Score-Regularized Continuous-Time Consistency Model (rCM)
distillation for the Qwen-Image-Edit model, enabling 2-4 step high-quality image generation.

Key Features:
- Teacher-Student distillation framework
- Jacobian-Vector Product (JVP) computation for gradient estimation
- Optional fake score network for forward-reverse divergence optimization
- FSDP-based distributed training
- Supports text-to-image, image-to-image, and multi-image editing tasks
"""

import torch
torch.backends.cuda.matmul.allow_tf32 = True
import math
import logging
import os
import time
import argparse
import functools
import torch.nn.functional as F
import torch.distributed as dist
from copy import deepcopy

from mmengine.registry import DATASETS
from mmengine.config import Config
from mmengine.dataset.sampler import InfiniteSampler

from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor

from qwen_image_edit.pipeline_qwenimage_edit import calculate_shift, QwenImageEditPipeline
from contextlib import nullcontext

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


# ==================== rCM Core Components ====================

class RectifiedFlow_TrigFlowWrapper:
    """
    Trigonometric Flow wrapper for Rectified Flow.
    Maps time from [0, π/2] using arctan(σ) parameterization.
    """
    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1000.0):
        assert abs(sigma_data - 1.0) < 1e-6, "sigma_data must be 1.0 for RectifiedFlowScaling"
        self.t_scaling_factor = t_scaling_factor

    def __call__(self, trigflow_t: torch.Tensor):
        """
        Args:
            trigflow_t: tensor in [0, π/2], shape [B, 1, 1, 1, 1]
        Returns:
            c_skip, c_out, c_in, c_noise: scaling coefficients
        """
        trigflow_t = trigflow_t.to(torch.float64)
        cos_t = torch.cos(trigflow_t)
        sin_t = torch.sin(trigflow_t)
        
        c_skip = 1.0 / (cos_t + sin_t)
        c_out = -sin_t / (cos_t + sin_t)
        c_in = 1.0 / (cos_t + sin_t)
        c_noise = (sin_t / (cos_t + sin_t)) * self.t_scaling_factor
        
        return c_skip, c_out, c_in, c_noise


def denoise_with_scaling(
    model,
    xt: torch.Tensor,
    time_t: torch.Tensor,
    scaling: RectifiedFlow_TrigFlowWrapper,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    img_shapes: list,
    txt_seq_lens: list,
    pipeline: QwenImageEditPipeline,
    height: int,
    width: int,
):
    """
    Denoise with TrigFlow scaling.
    
    Returns:
        x0_pred: predicted clean latent
        F_pred: predicted vector field F = (cos(t)*xt - x0) / sin(t)
    """
    # Get scaling coefficients
    time_t_expanded = time_t.view(-1, 1, 1, 1, 1)  # [B, 1, 1, 1, 1]
    c_skip, c_out, c_in, c_noise = scaling(trigflow_t=time_t_expanded)
    
    # Prepare input
    scaled_input = xt * c_in.to(xt.dtype)
    
    # Model prediction
    model_pred = model(
        hidden_states=scaled_input,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        timestep=c_noise.squeeze().to(xt.dtype) / 1000.0,  # normalized timestep
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        return_dict=False,
    )[0]
    
    # Get image sequence
    image_seq_len = math.prod(img_shapes[0][0])
    model_pred = model_pred[:, :image_seq_len]
    model_pred = pipeline._unpack_latents(model_pred, height, width, pipeline.vae_scale_factor)
    
    # Compute x0 prediction
    x0_pred = c_skip.to(xt.dtype) * xt + c_out.to(xt.dtype) * model_pred
    
    # Compute F prediction: F = (cos(t)*xt - x0) / sin(t)
    cos_t = torch.cos(time_t_expanded).to(xt.dtype)
    sin_t = torch.sin(time_t_expanded).to(xt.dtype)
    F_pred = (cos_t * xt - x0_pred) / (sin_t + 1e-8)
    
    return x0_pred, F_pred, model_pred


def compute_tangent_jvp(
    model,
    xt: torch.Tensor,
    t_xt: torch.Tensor,
    time_t: torch.Tensor,
    t_time: torch.Tensor,
    scaling: RectifiedFlow_TrigFlowWrapper,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    img_shapes: list,
    txt_seq_lens: list,
    pipeline: QwenImageEditPipeline,
    height: int,
    width: int,
):
    """
    Compute tangent (derivative) using Jacobian-Vector Product (JVP).
    
    This computes ∂F/∂xt * t_xt + ∂F/∂t * t_time efficiently.
    """
    # Define the function to differentiate
    def forward_fn(xt_input, time_input):
        _, F_pred, _ = denoise_with_scaling(
            model, xt_input, time_input, scaling,
            prompt_embeds, prompt_embeds_mask,
            img_shapes, txt_seq_lens,
            pipeline, height, width
        )
        return F_pred
    
    # Compute JVP: (F, dF/dxt * t_xt + dF/dt * t_time)
    F_pred, t_F_pred = torch.func.jvp(
        forward_fn,
        (xt, time_t),
        (t_xt, t_time)
    )
    
    return F_pred, t_F_pred


def compute_tangent_fd(
    model,
    xt: torch.Tensor,
    time_t: torch.Tensor,
    F_teacher: torch.Tensor,
    fd_size: float,
    scaling: RectifiedFlow_TrigFlowWrapper,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    img_shapes: list,
    txt_seq_lens: list,
    pipeline: QwenImageEditPipeline,
    height: int,
    width: int,
):
    """
    Compute tangent using Finite Difference (FD) approximation.
    
    fd_type 1: ∂F/∂t approximation
    fd_type 2: Total derivative approximation
    """
    h = fd_size
    
    # Compute F at t - h
    time_t_minus_h = time_t - h
    _, F_t_minus_h, _ = denoise_with_scaling(
        model, xt, time_t_minus_h, scaling,
        prompt_embeds, prompt_embeds_mask,
        img_shapes, txt_seq_lens,
        pipeline, height, width
    )
    
    # Current F
    _, F_t, _ = denoise_with_scaling(
        model, xt, time_t, scaling,
        prompt_embeds, prompt_embeds_mask,
        img_shapes, txt_seq_lens,
        pipeline, height, width
    )
    
    # Finite difference: ∂F/∂t ≈ (cos(h)*F(t) - F(t-h)) / sin(h)
    cos_h = math.cos(h)
    sin_h = math.sin(h)
    pF_pt = (cos_h * F_t - F_t_minus_h) / sin_h
    
    return pF_pt


# ==================== Training Loop ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')

    # Optimizer settings
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate (lower for distillation).')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay rate.')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1.')
    parser.add_argument('--beta2', default=0.95, type=float, help='beta2.')

    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='learning rate scheduler.')
    parser.add_argument('--lr_min', default=1e-6, type=float, help='minimum learning rate.')

    parser.add_argument('--grad_clip', default=1.0, type=float)

    # Model settings
    parser.add_argument('--transformer', default=MODEL_NAME, type=str, help='student model path')
    parser.add_argument('--teacher_transformer', default=MODEL_NAME, type=str, 
                        help='teacher model path (pre-trained)')
    parser.add_argument('--max_prompt_length', default=2048, type=int)

    # Training settings
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--log_steps', default=20, type=int)
    parser.add_argument('--ckpt_steps', default=500, type=int)
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--warmup_steps', default=100, type=int, help='warmup steps (optimizer updates).')

    # rCM specific settings
    parser.add_argument('--tangent_warmup', default=500, type=int, 
                        help='warmup steps for tangent computation')
    parser.add_argument('--teacher_guidance', default=0.0, type=float, 
                        help='guidance scale for teacher (0 = no CFG)')
    parser.add_argument('--loss_scale', default=100.0, type=float, 
                        help='scale factor for rCM loss')
    parser.add_argument('--fd_type', default=0, type=int, 
                        help='0: JVP, 1: FD type 1, 2: FD type 2')
    parser.add_argument('--fd_size', default=1e-4, type=float, 
                        help='finite difference step size')
    
    # Fake score network (optional, for DMD2)
    parser.add_argument('--use_fake_score', action='store_true', 
                        help='enable fake score network for DMD2')
    parser.add_argument('--student_update_freq', default=5, type=int,
                        help='update student every N steps')
    parser.add_argument('--loss_scale_dmd', default=1.0, type=float)
    parser.add_argument('--max_simulation_steps_fake', default=4, type=int)

    # Time sampling
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--sigma_max", type=float, default=80.0, 
                        help='maximum sigma for time sampling')

    parser.add_argument('--work_dir', default='work_dirs/Qwen-Image-Edit-rCM', type=str)
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient checkpointing.")

    args = parser.parse_args()

    # ==================== Initialization ====================
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    ddp_rank = dist.get_rank()

    if ddp_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        assert local_rank == 0, 'use torchrun to ensure ddp_rank==0 -> local_rank==0'

    if ddp_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.work_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    dist.barrier()

    assert args.log_steps % args.accum_steps == 0
    assert args.warmup_steps % args.accum_steps == 0

    logger.info(f"ddp rank: {ddp_rank}, local rank: {local_rank}, world_size: {world_size}")
    logger.info(f"rCM Distillation Training")
    logger.info(f"Teacher model: {args.teacher_transformer}")
    logger.info(f"Student model: {args.transformer}")
    logger.info(f"Use fake score: {args.use_fake_score}")
    logger.info(f"FD type: {args.fd_type}")

    # ==================== Dataset ====================
    config = Config.fromfile(args.config)
    dataset = DATASETS.build(config.dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=InfiniteSampler(dataset=dataset, shuffle=True),
        drop_last=False,
        collate_fn=lambda x: x
    )

    device = torch.device(f"cuda:{local_rank}")

    # ==================== Models ====================
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='scheduler'
    )
    
    # Text encoder (shared, frozen)
    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='text_encoder',
        torch_dtype=torch.bfloat16,
        device_map=None,
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
    transformer_student = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.transformer, subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # Teacher transformer (frozen)
    logger.info("Loading teacher model...")
    transformer_teacher = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=args.teacher_transformer, subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True
    )

    # Fake score network (optional)
    transformer_fake_score = None
    if args.use_fake_score:
        logger.info("Loading fake score network...")
        transformer_fake_score = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=args.teacher_transformer, subfolder='transformer',
            torch_dtype=torch.bfloat16,
            device_map=None,
            low_cpu_mem_usage=True
        )

    # Freeze teacher and VAE
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    transformer_teacher.requires_grad_(False).eval()

    # ==================== FSDP Wrapping ====================
    # Wrap text encoder
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

    # Wrap teacher
    transformer_teacher = FSDP(
        transformer_teacher,
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

    pipeline = QwenImageEditPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder,
        tokenizer=tokenizer, processor=processor, transformer=None
    )

    # Enable gradient checkpointing for student
    if args.gradient_checkpointing:
        transformer_student.enable_gradient_checkpointing()
    transformer_student.train()
    params2dims = {k: v.dim() for k, v in transformer_student.named_parameters()}

    # Wrap student
    transformer_student = FSDP(
        transformer_student,
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

    # Wrap fake score if enabled
    if transformer_fake_score is not None:
        if args.gradient_checkpointing:
            transformer_fake_score.enable_gradient_checkpointing()
        transformer_fake_score.train()
        
        transformer_fake_score = FSDP(
            transformer_fake_score,
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

    # ==================== Optimizer ====================
    decay_params = []
    nodecay_params = []
    for k, v in transformer_student.named_parameters():
        k = k.replace('_fsdp_wrapped_module.', '')
        if params2dims.get(k, 0) >= 2:
            decay_params.append(v)
        else:
            nodecay_params.append(v)

    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    optimizer_student = AdamW(
        params=optim_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    # Optimizer for fake score
    optimizer_fake_score = None
    if transformer_fake_score is not None:
        fake_decay_params = []
        fake_nodecay_params = []
        for k, v in transformer_fake_score.named_parameters():
            k = k.replace('_fsdp_wrapped_module.', '')
            if params2dims.get(k, 0) >= 2:
                fake_decay_params.append(v)
            else:
                fake_nodecay_params.append(v)
        
        optimizer_fake_score = AdamW(
            params=[
                {'params': fake_decay_params, 'weight_decay': args.weight_decay},
                {'params': fake_nodecay_params, 'weight_decay': 0.0}
            ],
            lr=args.lr * 0.1,  # Lower LR for fake score
            betas=(args.beta1, args.beta2)
        )

    # LR scheduler
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

    lr_scheduler_student = LambdaLR(optimizer_student, lr_lambda=lr_lambda)
    lr_scheduler_fake_score = None
    if optimizer_fake_score is not None:
        lr_scheduler_fake_score = LambdaLR(optimizer_fake_score, lr_lambda=lr_lambda)

    # ==================== rCM Components ====================
    scaling = RectifiedFlow_TrigFlowWrapper(sigma_data=1.0, t_scaling_factor=1000.0)

    # ==================== Training Loop ====================
    iterator = iter(dataloader)
    tik = time.time()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device).bfloat16()
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device).bfloat16()

    logger.info("Starting rCM distillation training...")

    for train_step in range(1, args.train_steps + 1):
        # ==================== Data Loading ====================
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

        # ==================== Encode Inputs ====================
        with torch.no_grad():
            # Encode text and images
            prompt_embeds, prompt_embeds_mask = pipeline._get_qwen_prompt_embeds(
                [text], [prompt_images], device=device, dtype=torch.bfloat16)
            assert prompt_embeds.shape[0] == 1
            prompt_embeds = prompt_embeds[:, :args.max_prompt_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :args.max_prompt_length]

            # Encode images to latents
            for image in [output_image] + input_images:
                pixel_values = pipeline.image_processor.preprocess(
                    image, image.height, image.width).to(dtype=torch.bfloat16, device=device)
                latents = vae.encode(pixel_values[:, :, None], return_dict=False)[0].sample()
                latents = (latents - latents_mean) / latents_std
                assert latents.shape[2] == 1 and latents.shape[0] == 1

                img_shapes_list.append((1, latents.shape[-2] // 2, latents.shape[-1] // 2))
                latents_list.append(latents)

        # ==================== Time Sampling ====================
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

        # Sample sigma from logit normal distribution
        u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(1,), device=device)
        sigmas = torch.nn.functional.sigmoid(u).to(torch.float64)
        
        # Apply shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        
        # Convert to trigflow time: t = arctan(σ)
        time_t = torch.atan(sigmas).to(torch.float64)  # in [0, π/2]

        # ==================== Add Noise ====================
        noise = torch.randn_like(latents_list[0])
        cos_t = torch.cos(time_t.view(-1, 1, 1, 1, 1))
        sin_t = torch.sin(time_t.view(-1, 1, 1, 1, 1))
        
        x0 = latents_list[0]
        xt = cos_t.to(x0.dtype) * x0 + sin_t.to(x0.dtype) * noise

        # ==================== Pack Latents ====================
        packed_noisy_model_input = []
        for latents in [xt] + latents_list[1:]:
            assert latents.shape[2] == 1 and latents.shape[0] == 1
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

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        # ==================== Determine Training Phase ====================
        is_student_phase = (
            transformer_fake_score is None or 
            train_step < args.tangent_warmup or 
            train_step % args.student_update_freq == 0
        )

        if is_student_phase:
            # ==================== Student Update ====================
            transformer_student.train()
            if transformer_fake_score is not None:
                transformer_fake_score.eval()

            # Teacher prediction (no gradient)
            with torch.no_grad():
                _, F_teacher, _ = denoise_with_scaling(
                    transformer_teacher, packed_noisy_model_input, time_t,
                    scaling, prompt_embeds, prompt_embeds_mask,
                    [img_shapes_list], txt_seq_lens,
                    pipeline, height, width
                )

            # Compute tangent (gradient direction)
            warmup_ratio = min(1.0, train_step / args.tangent_warmup)
            
            if args.fd_type == 0:  # JVP method
                # Tangent inputs
                t_xt = cos_t.to(xt.dtype) * sin_t.to(xt.dtype) * F_teacher
                t_time = torch.zeros_like(time_t)  # or compute properly
                
                with torch.enable_grad():
                    _, t_F_student = compute_tangent_jvp(
                        transformer_student, packed_noisy_model_input, t_xt,
                        time_t, t_time, scaling,
                        prompt_embeds, prompt_embeds_mask,
                        [img_shapes_list], txt_seq_lens,
                        pipeline, height, width
                    )
            else:  # Finite difference method
                with torch.no_grad():
                    t_F_student = compute_tangent_fd(
                        transformer_student, packed_noisy_model_input,
                        time_t, F_teacher, args.fd_size, scaling,
                        prompt_embeds, prompt_embeds_mask,
                        [img_shapes_list], txt_seq_lens,
                        pipeline, height, width
                    )

            # Student prediction (with gradient)
            update_grad = train_step % args.accum_steps == 0
            with (nullcontext() if update_grad else transformer_student.no_sync()):
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                    _, F_student, _ = denoise_with_scaling(
                        transformer_student, packed_noisy_model_input, time_t,
                        scaling, prompt_embeds, prompt_embeds_mask,
                        [img_shapes_list], txt_seq_lens,
                        pipeline, height, width
                    )

                    F_student_sg = F_student.clone().detach()

                    # Compute rCM gradient
                    # g = -cos(t) * sqrt(1 - warmup_ratio^2 * sin^2(t)) * (F_student_sg - F_teacher)
                    #     - warmup_ratio * (cos(t) * sin(t) * xt + t_F_student)
                    gradient = (
                        -cos_t.to(F_student.dtype) * torch.sqrt(1 - warmup_ratio**2 * sin_t.to(F_student.dtype)**2) 
                        * (F_student_sg - F_teacher)
                        - warmup_ratio * (cos_t.to(F_student.dtype) * sin_t.to(F_student.dtype) * xt + t_F_student)
                    )

                    # Check for NaN
                    nan_mask = torch.isnan(gradient) | torch.isnan(F_student)
                    gradient = torch.where(nan_mask, torch.zeros_like(gradient), gradient)
                    F_student = torch.where(nan_mask, torch.zeros_like(F_student), F_student)
                    
                    # Normalize gradient
                    gradient = gradient.double() / (gradient.double().norm(p=2, dim=(1, 2, 3, 4), keepdim=True) + 0.1)

                    # rCM loss
                    loss_rcm = ((F_student - F_student_sg - gradient) ** 2).sum(dim=(1, 2, 3, 4)).mean()
                    loss = args.loss_scale * loss_rcm

                loss.backward()

            if update_grad:
                grad_norm = transformer_student.clip_grad_norm_(args.grad_clip)
                optimizer_student.step()
                optimizer_student.zero_grad(set_to_none=True)
                lr_scheduler_student.step()

                if train_step % args.log_steps == 0:
                    speed = args.log_steps / (time.time() - tik)
                    cur_lr = lr_scheduler_student.get_last_lr()[0]
                    logger.info(
                        f"[Student] Step={train_step}/{args.train_steps}, "
                        f"loss_rcm={loss_rcm.item():.4f}, "
                        f"grad_norm={grad_norm.item():.4f}, "
                        f"speed={speed:.4f} steps/s, LR={cur_lr:.8f}"
                    )
                    tik = time.time()

        else:
            # ==================== Fake Score Update ====================
            transformer_student.eval()
            transformer_fake_score.train()

            # This is a simplified DMD2 update
            # In practice, you'd simulate student sampling and train fake_score to match teacher
            
            logger.info(f"[Fake Score] Step={train_step} (placeholder - implement DMD2 logic)")
            # TODO: Implement fake score training logic
            pass

        # ==================== Checkpointing ====================
        save_ckpt = train_step % args.ckpt_steps == 0

        if save_ckpt:
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(transformer_student, StateDictType.FULL_STATE_DICT, cfg):
                full_state = transformer_student.state_dict()

            save_dir = os.path.join(args.work_dir, 'transformer')
            if ddp_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(full_state, os.path.join(save_dir, 'diffusion_pytorch_model.bin'))
                transformer_student.module.to_json_file(os.path.join(save_dir, 'config.json'))
                logger.info(f"Saved checkpoint to {save_dir}")

            del full_state
            dist.barrier()

    logger.info("Training completed!")
    dist.destroy_process_group()


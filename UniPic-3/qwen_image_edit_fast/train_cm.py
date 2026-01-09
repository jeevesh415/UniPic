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
from collections import OrderedDict
from copy import deepcopy
from mmengine.registry import DATASETS
from mmengine.config import Config
from mmengine.dataset.sampler import InfiniteSampler

from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor

from qwen_image_edit.pipeline_qwenimage_edit import calculate_shift, QwenImageEditPipeline
from contextlib import nullcontext
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# from torch.nn.parallel import DistributedDataParallel as DDP  # REMOVED: DDP
# NEW: FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, StateDictType, FullStateDictConfig, MixedPrecision
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer, Qwen2_5_VLVisionBlock

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel


MODEL_NAME = "Qwen-Image-Edit"

@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(ema_model, 'module'):
        ema_model = ema_model.module
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)



def forward_fn(pipeline, transformer, clean_latents, condition_latents, timesteps, noises, prompt_embeds, prompt_embeds_mask, img_shapes_list):
    noisy_model_input = (1.0 - timesteps) * clean_latents + timesteps * noises
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
def compute_tangent_fd(pipeline, transformer, clean_latents, condition_latents, timesteps, noises, prompt_embeds, prompt_embeds_mask, img_shapes_list, fd_size):
    def xfunc(_timesteps):
        return forward_fn(
            pipeline, transformer, clean_latents, condition_latents, _timesteps, 
            noises, prompt_embeds, prompt_embeds_mask, img_shapes_list
        )
    fc1_dt = 1 / (2 * fd_size)
    dF_dv_dt = xfunc(timesteps + fd_size) * fc1_dt - xfunc(timesteps - fd_size) * fc1_dt
    return dF_dv_dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')

    parser.add_argument('--lr', default=1e-6, type=float, help='learning rate.')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay rate.')
    parser.add_argument('--beta1', default=0.9, type=float, help='beta1.')
    parser.add_argument('--beta2', default=0.95, type=float, help='beta2.')

    parser.add_argument('--lr_scheduler', default='cosine', type=str, help='learning rate scheduler.')
    parser.add_argument('--lr_min', default=1e-7, type=float, help='learning rate.')


    parser.add_argument('--grad_clip', default=1.0, type=float)

    parser.add_argument('--transformer', default=MODEL_NAME, type=str)
    parser.add_argument('--max_prompt_length', default=2048, type=int)

    # CM specific settings
    parser.add_argument('--guidance_scale', default=1.5, type=float, 
                        help='guidance scale for teacher (0 = no CFG)')
    parser.add_argument('--loss_scale', default=1.0, type=float, 
                        help='scale factor for rCM loss')
    parser.add_argument('--fd_size', default=5e-3, type=float, 
                        help='finite difference step size')
    parser.add_argument('--tangent_norm', default=False, action='store_true', 
                        help='normalize tangent vector')
    parser.add_argument('--tangent_clip', default=False, action='store_true', 
                        help='clip tangent vector')
    
    # Training settings
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--log_steps', default=20, type=int)
    parser.add_argument('--ckpt_steps', default=200, type=int)
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--warmup_steps', default=100, type=int, help='warmup steps (optimizer updates).')

    parser.add_argument('--ema_rate', default=0.95, type=float, help='EMA learning rate.')

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
    ema_transformer = deepcopy(transformer)
    
    # Freeze teacher and VAE
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    transformer.train()
    ema_transformer.train()
    ema_transformer.requires_grad_(False)

    # wrap transformer_student with fsdp# CHANGED: optimizer over FSDP parameters
    params2dims = {k: v.dim() for k, v in transformer.named_parameters()}
    decay_params = []
    nodecay_params = []
    for k, v in transformer.named_parameters():
        k = k.replace('_fsdp_wrapped_module.', '')
        if params2dims[k] >= 2:
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

    optimizer = AdamW(params=optim_groups,
                      lr=args.lr,
                      betas=(args.beta1, args.beta2))

    
    
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

    # Wrap student
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        ema_transformer.enable_gradient_checkpointing()


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
    ema_transformer = FSDP(
        ema_transformer,
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
                

        model_tangent = compute_tangent_fd(pipeline, transformer, latents_list[0], latents_list[1:], timesteps, noises, prompt_embeds, prompt_embeds_mask, img_shapes_list, args.fd_size)
        if args.tangent_norm:
            model_tangent = model_tangent.double() / (model_tangent.double().norm(p=2, dim=(1, 2, 3, 4), keepdim=True) + 0.1)
        elif args.tangent_clip:
            model_tangent = torch.clamp(model_tangent, -1, 1)
        else: pass
        target = noises - latents_list[0] - timesteps * model_tangent

        if args.guidance_scale > 0:
            with torch.no_grad():
                ema_pred_cond = forward_fn(pipeline, ema_transformer, latents_list[0], latents_list[1:], timesteps, noises, prompt_embeds, prompt_embeds_mask, img_shapes_list)
                ema_pred_uncond = forward_fn(pipeline, ema_transformer, latents_list[0], latents_list[1:], timesteps, noises, null_embeds, null_embeds_mask, img_shapes_list)
            
            target = target + args.guidance_scale * (ema_pred_cond - ema_pred_uncond)

        update_grad = train_step % args.accum_steps == 0
        with (nullcontext() if update_grad else transformer.no_sync()):
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                model_pred = forward_fn(
                    pipeline, transformer, latents_list[0], latents_list[1:], timesteps, 
                    noises, prompt_embeds, prompt_embeds_mask, img_shapes_list,
                )
                loss = (
                    F.mse_loss(model_pred.float(), target.float()) + 
                    torch.mean(1 - F.cosine_similarity(model_pred.float(), target.float(), dim=1))
                ) / args.accum_steps
                loss = torch.nan_to_num(loss, nan=0, posinf=1e5, neginf=-1e5)
                
            loss.backward()

        lr_scheduler.step()
        
        if update_grad:
            # CHANGED: use FSDP's clip helper
            grad_norm = transformer.clip_grad_norm_(args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(ema_transformer, transformer, args.ema_rate)

            update_log = train_step % args.log_steps == 0
            if update_log:
                speed = args.log_steps / (time.time() - tik)
                cur_lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"Train steps={train_step}/{args.train_steps}, loss={loss.item():.4f}, "
                            f"grad norm={grad_norm.item():.4f}, speed: {speed:.4f}, LR: {cur_lr:.8f}")
                tik = time.time()

        save_ckpt = train_step % args.ckpt_steps == 0

        if save_ckpt:
            # NEW: gather full unsharded state on rank0 and save
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(transformer, StateDictType.FULL_STATE_DICT, cfg):
                full_state = transformer.state_dict()
            with FSDP.state_dict_type(ema_transformer, StateDictType.FULL_STATE_DICT, cfg):
                ema_state = ema_transformer.state_dict()
            save_dir = os.path.join(args.work_dir, f'step{train_step}', 'transformer')
            ema_save_dir = os.path.join(args.work_dir, f'step{train_step}', 'ema_transformer')
            if ddp_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(ema_save_dir, exist_ok=True)
                # Try HF-style save_pretrained on raw module (keeps compatibility)
                torch.save(full_state, os.path.join(save_dir, 'diffusion_pytorch_model.bin'))   # todo: shard the state dict
                transformer.module.to_json_file(os.path.join(save_dir, 'config.json'))
                torch.save(ema_state, os.path.join(ema_save_dir, 'diffusion_pytorch_model.bin'))
                ema_transformer.module.to_json_file(os.path.join(ema_save_dir, 'config.json'))
            logger.info(f"Saved checkpoint to {save_dir}")

            del full_state
            del ema_state
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()

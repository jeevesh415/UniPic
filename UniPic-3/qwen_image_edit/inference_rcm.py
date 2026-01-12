"""
rCM Fast Inference Script for Qwen-Image-Edit

This script demonstrates 2-4 step fast inference using the distilled rCM model.
Achieves 10-25x speedup compared to the original 50-step generation.

Usage:
    python qwen_image_edit/inference_rcm.py \
        --model_path work_dirs/Qwen-Image-Edit-rCM/transformer \
        --prompt "A beautiful sunset over mountains" \
        --output output.png \
        --num_steps 4
"""

import torch
import argparse
import math
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor
from qwen_image_edit.pipeline_qwenimage_edit import calculate_shift, QwenImageEditPipeline


class RectifiedFlow_TrigFlowWrapper:
    """Trigonometric Flow wrapper for inference."""
    def __init__(self, sigma_data: float = 1.0, t_scaling_factor: float = 1000.0):
        self.t_scaling_factor = t_scaling_factor

    def __call__(self, trigflow_t: torch.Tensor):
        trigflow_t = trigflow_t.to(torch.float64)
        cos_t = torch.cos(trigflow_t)
        sin_t = torch.sin(trigflow_t)
        
        c_skip = 1.0 / (cos_t + sin_t)
        c_out = -sin_t / (cos_t + sin_t)
        c_in = 1.0 / (cos_t + sin_t)
        c_noise = (sin_t / (cos_t + sin_t)) * self.t_scaling_factor
        
        return c_skip, c_out, c_in, c_noise


def denoise_step(
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
    """Single denoising step for rCM inference."""
    # Get scaling coefficients
    time_t_expanded = time_t.view(-1, 1, 1, 1, 1)
    c_skip, c_out, c_in, c_noise = scaling(trigflow_t=time_t_expanded)
    
    # Prepare input
    scaled_input = xt * c_in.to(xt.dtype)
    
    # Model prediction
    with torch.no_grad():
        model_pred = model(
            hidden_states=scaled_input,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=c_noise.squeeze().to(xt.dtype) / 1000.0,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
    
    # Unpack latents
    image_seq_len = math.prod(img_shapes[0][0])
    model_pred = model_pred[:, :image_seq_len]
    model_pred = pipeline._unpack_latents(model_pred, height, width, pipeline.vae_scale_factor)
    
    # Compute x0 prediction
    x0_pred = c_skip.to(xt.dtype) * xt + c_out.to(xt.dtype) * model_pred
    
    return x0_pred


@torch.no_grad()
def rcm_sample(
    model,
    pipeline: QwenImageEditPipeline,
    prompt: str,
    input_images: list = None,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 4,
    sigma_max: float = 80.0,
    seed: int = 42,
    device: str = "cuda",
):
    """
    rCM sampling with 2-4 steps.
    
    Args:
        model: Distilled rCM model
        pipeline: QwenImageEditPipeline instance
        prompt: Text prompt
        input_images: Optional list of input images for editing
        height: Output height
        width: Output width
        num_steps: Number of sampling steps (1-4)
        sigma_max: Maximum sigma for initialization
        seed: Random seed
        device: Device
    
    Returns:
        Generated PIL Image
    """
    torch.manual_seed(seed)
    
    # Process input images if provided
    if input_images is None:
        input_images = []
    
    prompt_images = [img.resize((round(img.width * 28 / 32), round(img.height * 28 / 32)))
                     for img in input_images]
    
    # Get text embeddings
    prompt_embeds, prompt_embeds_mask = pipeline._get_qwen_prompt_embeds(
        [prompt], [prompt_images], device=device, dtype=torch.bfloat16
    )
    
    # Prepare latent shape
    latent_height = height // pipeline.vae_scale_factor // 2
    latent_width = width // pipeline.vae_scale_factor // 2
    img_shapes = [(1, latent_height, latent_width)]
    
    # Encode input images if provided
    latents_list = []
    if input_images:
        for image in input_images:
            pixel_values = pipeline.image_processor.preprocess(
                image, image.height, image.width
            ).to(dtype=torch.bfloat16, device=device)
            latents = pipeline.vae.encode(pixel_values[:, :, None], return_dict=False)[0].sample()
            
            latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device).bfloat16()
            latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device).bfloat16()
            latents = (latents - latents_mean) / latents_std
            latents_list.append(latents)
    
    # Initialize noise
    latent_channels = 16  # Qwen-Image latent channels
    init_latents = torch.randn(
        1, latent_channels, 1, latent_height * 2, latent_width * 2,
        dtype=torch.bfloat16, device=device
    )
    
    # Pack latents
    packed_latents = pipeline._pack_latents(
        init_latents.squeeze(2),
        batch_size=1,
        num_channels_latents=latent_channels,
        height=latent_height * 2,
        width=latent_width * 2,
    )
    
    # Add input image latents if provided
    if latents_list:
        for latents in latents_list:
            packed_input = pipeline._pack_latents(
                latents.squeeze(2),
                batch_size=1,
                num_channels_latents=latent_channels,
                height=latents.shape[3],
                width=latents.shape[4],
            )
            packed_latents = torch.cat([packed_latents, packed_input], dim=1)
    
    # Setup time steps for rCM sampling
    # t_steps = [arctan(sigma_max), mid_t_1, mid_t_2, ..., 0]
    if num_steps == 1:
        mid_t = []
    elif num_steps == 2:
        mid_t = [1.4]
    elif num_steps == 3:
        mid_t = [1.5, 1.0]
    elif num_steps == 4:
        mid_t = [1.5, 1.4, 1.0]
    else:
        raise ValueError(f"num_steps must be 1-4, got {num_steps}")
    
    t_steps = [math.atan(sigma_max)] + mid_t + [0.0]
    t_steps = torch.tensor(t_steps, dtype=torch.float64, device=device)
    
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
    scaling = RectifiedFlow_TrigFlowWrapper()
    
    # rCM sampling loop
    x = init_latents
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        print(f"Step {i+1}/{num_steps}: t={t_cur.item():.4f} -> {t_next.item():.4f}")
        
        # Prepare current packed input
        current_packed = pipeline._pack_latents(
            x.squeeze(2),
            batch_size=1,
            num_channels_latents=latent_channels,
            height=latent_height * 2,
            width=latent_width * 2,
        )
        
        # Add input latents
        if latents_list:
            for latents in latents_list:
                packed_input = pipeline._pack_latents(
                    latents.squeeze(2),
                    batch_size=1,
                    num_channels_latents=latent_channels,
                    height=latents.shape[3],
                    width=latents.shape[4],
                )
                current_packed = torch.cat([current_packed, packed_input], dim=1)
        
        # Denoise
        x0_pred = denoise_step(
            model, current_packed, t_cur.unsqueeze(0),
            scaling, prompt_embeds, prompt_embeds_mask,
            [img_shapes], txt_seq_lens,
            pipeline, height, width
        )
        
        # Add noise for next step (except last step)
        if t_next > 1e-5:
            noise = torch.randn_like(x0_pred)
            cos_t_next = math.cos(t_next.item())
            sin_t_next = math.sin(t_next.item())
            x = cos_t_next * x0_pred + sin_t_next * noise
        else:
            x = x0_pred
    
    # Decode latents
    latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device).bfloat16()
    latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device).bfloat16()
    
    final_latents = x * latents_std + latents_mean
    image = pipeline.vae.decode(final_latents, return_dict=False)[0]
    
    # Post-process
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rCM Fast Inference for Qwen-Image-Edit")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to distilled rCM model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen-Image-Edit",
                        help="Base model for VAE and text encoder")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt")
    parser.add_argument("--input_images", type=str, nargs="+", default=None,
                        help="Paths to input images for editing")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Number of sampling steps")
    parser.add_argument("--sigma_max", type=float, default=80.0,
                        help="Maximum sigma for initialization")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output_rcm.png",
                        help="Output image path")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print(f"Loading models...")
    device = torch.device(args.device)
    
    # Load components
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder='scheduler'
    )
    
    text_encoder = AutoModel.from_pretrained(
        args.base_model, subfolder='text_encoder',
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, subfolder='tokenizer'
    )
    
    processor = Qwen2VLProcessor.from_pretrained(
        args.base_model, subfolder='processor'
    )
    
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.base_model, subfolder='vae',
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    # Load distilled model
    print(f"Loading distilled rCM model from {args.model_path}")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    transformer.eval()
    
    # Create pipeline
    pipeline = QwenImageEditPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer
    )
    
    # Load input images if provided
    input_images = []
    if args.input_images:
        for img_path in args.input_images:
            input_images.append(Image.open(img_path).convert("RGB"))
        print(f"Loaded {len(input_images)} input images")
    
    # Generate
    print(f"Generating with prompt: '{args.prompt}'")
    print(f"Steps: {args.num_steps}, Height: {args.height}, Width: {args.width}")
    
    import time
    start_time = time.time()
    
    image = rcm_sample(
        model=transformer,
        pipeline=pipeline,
        prompt=args.prompt,
        input_images=input_images,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        sigma_max=args.sigma_max,
        seed=args.seed,
        device=args.device,
    )
    
    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f} seconds")
    
    # Save
    image.save(args.output)
    print(f"Saved to {args.output}")
    print(f"\n✨ rCM {args.num_steps}-step generation: {elapsed:.2f}s")
    print(f"   Estimated speedup: {50/args.num_steps:.1f}x vs 50-step baseline")


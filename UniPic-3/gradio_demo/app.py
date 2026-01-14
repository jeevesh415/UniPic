"""
Gradio Demo for UniPic-3 DMD Multi-Image Composition
Hugging Face Space compatible version

Upload up to 6 images and generate a composed result using DMD model with 4-step inference.
"""

import gradio as gr
import torch
from PIL import Image
import os

# Use local pipeline to ensure compatibility
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from pipeline_qwenimage_edit import QwenImageEditPipeline
except ImportError:
    # Fallback to diffusers if local not available
    try:
        from diffusers import QwenImageEditPipeline
    except ImportError:
        raise ImportError(
            "QwenImageEditPipeline not found. Please ensure pipeline_qwenimage_edit.py "
            "is in the same directory or diffusers is installed."
        )

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor


# Global pipeline
pipe = None

# Model paths (can be set via environment variables)
# export HF_ENDPOINT=https://hf-mirror.com

MODEL_NAME = os.environ.get("MODEL_NAME", "/data_genie/genie/chris/Qwen-Image-Edit")
# Default to local path if exists, otherwise use HuggingFace
default_transformer = "/data_genie/genie/chris/unipic3_ckpt/dmd/ema_transformer" if os.path.exists("/data_genie/genie/chris/unipic3_ckpt/dmd/ema_transformer") else "Skywork/Unipic3-DMD"
TRANSFORMER_PATH = os.environ.get("TRANSFORMER_PATH", default_transformer)


def load_model():
    """Load the DMD model and pipeline"""
    global pipe
    
    if pipe is not None:
        return pipe
    
    print(f"Loading model from {TRANSFORMER_PATH}...")
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='scheduler'
    )
    
    # Load text encoder
    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='text_encoder',
        device_map='auto', torch_dtype=torch.bfloat16
    )
    
    # Load tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='tokenizer',
    )
    processor = Qwen2VLProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, subfolder='processor',
    )
    
    # Load transformer (DMD model)
    # Handle both local paths and HuggingFace repo paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(TRANSFORMER_PATH):
        # Local path - load directly to device (avoid device_map issues with .bin files)
        if os.path.isdir(TRANSFORMER_PATH):
            # Check if it's a direct transformer directory or has subfolder
            if os.path.exists(os.path.join(TRANSFORMER_PATH, "config.json")):
                transformer = QwenImageTransformer2DModel.from_pretrained(
                    pretrained_model_name_or_path=TRANSFORMER_PATH,
                    torch_dtype=torch.bfloat16,
                    use_safetensors=False  # Use .bin file
                ).to(device)
            else:
                transformer = QwenImageTransformer2DModel.from_pretrained(
                    pretrained_model_name_or_path=TRANSFORMER_PATH,
                    subfolder='transformer',
                    torch_dtype=torch.bfloat16,
                    use_safetensors=False
                ).to(device)
        else:
            raise ValueError(f"Transformer path does not exist: {TRANSFORMER_PATH}")
    else:
        # HuggingFace repo path
        # Handle paths like "Skywork/Unipic3-DMD/ema_transformer"
        path_parts = TRANSFORMER_PATH.split('/')
        if len(path_parts) >= 3:
            # Has subfolder: "Skywork/Unipic3-DMD/ema_transformer"
            repo_id = '/'.join(path_parts[:2])  # "Skywork/Unipic3-DMD"
            subfolder = path_parts[2]  # "ema_transformer"
            transformer = QwenImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path=repo_id,
                subfolder=subfolder,
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
        elif len(path_parts) == 2:
            # Just repo: "Skywork/Unipic3-DMD"
            transformer = QwenImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path=TRANSFORMER_PATH,
                subfolder='transformer',
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
        else:
            # Single name, assume it's a repo ID
            transformer = QwenImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path=TRANSFORMER_PATH,
                subfolder='transformer',
                device_map='auto',
                torch_dtype=torch.bfloat16
            )
    
    # Load VAE
    # Get device from transformer (handle both .device and device_map cases)
    if hasattr(transformer, 'device'):
        vae_device = transformer.device
    elif hasattr(transformer, 'hf_device_map'):
        # If using device_map, get the first device
        vae_device = device
    else:
        vae_device = device
    
    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME, 
        subfolder='vae', 
        torch_dtype=torch.bfloat16,
    ).to(vae_device)
    
    # Create pipeline
    pipe = QwenImageEditPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=transformer
    )
    
    print("Model loaded successfully!")
    # Don't return pipe for demo.load() - it expects no return value


def process_images(
    img1, img2, img3, img4, img5, img6,
    prompt: str,
    true_cfg_scale: float = 4.0,
    seed: int = 42,
    num_steps: int = 4
) -> tuple:
    """Process multiple images and generate composed result"""
    global pipe
    
    # Ensure model is loaded (should be loaded by demo.load() on startup)
    if pipe is None:
        return None, "⏳ Model is still loading, please wait a moment and try again..."
    
    # Filter out None images
    images = [img for img in [img1, img2, img3, img4, img5, img6] if img is not None]
    
    # Validate inputs
    if len(images) == 0:
        return None, "❌ Error: Please upload at least one image."
    
    if len(images) > 6:
        return None, f"❌ Error: Maximum 6 images allowed. You uploaded {len(images)} images."
    
    if not prompt or prompt.strip() == "":
        return None, "❌ Error: Please enter an editing instruction."
    
    try:
        # Convert to RGB
        images = [img.convert("RGB") for img in images]
        
        print(f"Processing {len(images)} images with prompt: '{prompt}'")
        print(f"Steps: {num_steps}, CFG Scale: {true_cfg_scale}, Seed: {seed}")
        
        # Generate image
        # Note: images can be passed as first positional argument or as keyword argument
        with torch.no_grad():
            # Try positional argument first (as shown in pipeline examples)
            if len(images) == 1:
                # Single image: pass as first positional argument
                result = pipe(
                    images[0],
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    negative_prompt=' ',
                    num_inference_steps=num_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=torch.manual_seed(int(seed))
                ).images[0]
            else:
                # Multiple images: pass as keyword argument
                result = pipe(
                    images=images,
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    negative_prompt=' ',
                    num_inference_steps=num_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator=torch.manual_seed(int(seed))
                ).images[0]
        
        return result, f"✅ Success! Generated from {len(images)} image(s) in {num_steps} steps."
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


# Create Gradio interface
with gr.Blocks(title="UniPic-3 DMD Multi-Image Composition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔥 UniPic-3 DMD Multi-Image Composition
    
    Upload up to **6 images** and provide an editing instruction to generate a composed result.
    
    **Model**: DMD (Distribution-Matching Distillation) - **4-step fast inference (12.5× speedup)**
    
    **Features**:
    - Support 1-6 input images
    - Fast 4-step inference
    - High-quality multi-image composition
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Images (1-6 images)")
            image_inputs = [
                gr.Image(type="pil", label=f"Image {i+1}", visible=(i < 2))
                for i in range(6)
            ]
            
            num_images = gr.Slider(
                minimum=1,
                maximum=6,
                value=2,
                step=1,
                label="Number of Images",
                info="Select how many images you want to upload"
            )
            
            def update_image_visibility(num):
                return [gr.update(visible=(i < num)) for i in range(6)]
            
            num_images.change(
                fn=update_image_visibility,
                inputs=num_images,
                outputs=image_inputs
            )
            
            gr.Markdown("### ✍️ Editing Instruction")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="e.g., A man from Image1 is standing on a surfboard from Image2, riding the ocean waves under a bright blue sky.",
                lines=3,
                value="Combine the reference images to generate the final result."
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=4.0,
                    step=0.5,
                    label="CFG Scale",
                    info="Higher values make the output more aligned with the prompt"
                )
                seed = gr.Number(
                    value=42,
                    label="Seed",
                    info="Random seed for reproducibility",
                    precision=0
                )
                num_steps = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=8,
                    step=1,
                    label="Inference Steps",
                    info="Number of denoising steps (8 is recommended for DMD)"
                )
            
            generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 Generated Result")
            output_image = gr.Image(type="pil", label="Output Image")
            status_text = gr.Textbox(
                label="Status",
                value="Ready. Upload images and enter a prompt, then click Generate.",
                interactive=False
            )
    
    # Load model on startup
    def load_model_wrapper():
        """Wrapper to load model without returning value"""
        load_model()
        return None
    
    demo.load(
        fn=load_model_wrapper,
        inputs=[],
        outputs=[],
        show_progress=True
    )
    
    # Generate button
    generate_btn.click(
        fn=process_images,
        inputs=[*image_inputs, prompt_input, cfg_scale, seed, num_steps],
        outputs=[output_image, status_text]
    )


if __name__ == "__main__":
    demo.launch()


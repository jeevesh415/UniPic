from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor
import torch
import argparse
from peft import PeftModel
from PIL import Image

import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from qwen_image_edit.pipeline_qwenimage_edit import QwenImageEditPipeline
except ImportError:
    # 如果在 qwen_image_edit 目录下运行，则直接导入
    from pipeline_qwenimage_edit import QwenImageEditPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--transformer", type=str, default=None)
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, 
                       help="图像路径列表，支持多个图像路径")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="编辑提示词")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                       help="CFG 缩放参数，默认为 4.0")
    parser.add_argument("--seed", type=int, default=0,
                       help="随机种子，默认为 0")
    parser.add_argument("--output_path", type=str, default="example_ours.png",
                       help="输出图片保存路径，默认为 example_ours.png")

    args = parser.parse_args()


    # Use "Qwen-Image-Edit" as default model name (from HuggingFace)
    # Or set via environment variable: export MODEL_NAME=/path/to/model
    model_name = os.environ.get('MODEL_NAME', 'Qwen-Image-Edit')
    transformer = model_name if args.transformer is None else args.transformer

    print(f'Load transformer weights from {transformer}.')

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='scheduler'
    )
    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='text_encoder',
        device_map='auto', torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='tokenizer',
    )
    processor = Qwen2VLProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='processor',
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=transformer, subfolder='transformer', device_map='auto', torch_dtype=torch.bfloat16
    )

    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='vae', torch_dtype=torch.bfloat16,
    ).to(transformer.device)


    if args.lora is not None:
        transformer = PeftModel.from_pretrained(transformer, args.lora)
        print(f'Load lora weights from {args.lora}.')

    pipe = QwenImageEditPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                                 tokenizer=tokenizer, processor=processor, transformer=transformer)

    # 从命令行参数获取输入
    image_paths = args.image_paths
    prompt = args.prompt
    true_cfg_scale = args.true_cfg_scale
    seed = args.seed
    output_path = args.output_path

    print(f"输入图像路径: {image_paths}")
    print(f"编辑提示词: {prompt}")
    print(f"CFG 缩放参数: {true_cfg_scale}")
    print(f"随机种子: {seed}")
    print(f"输出路径: {output_path}")

    # 加载图像
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # 执行图像编辑
    image = pipe(
        images=images,
        height=1024,
        width=1024,
        prompt=prompt,
        negative_prompt=' ',
        num_inference_steps=50,
        true_cfg_scale=true_cfg_scale,
        generator=torch.manual_seed(seed)
    ).images[0]

    # 保存输出图像
    image.save(output_path)
    print(f"图像已保存到: {output_path}")

    """
    Example usage:
    # Using HuggingFace model
    python qwen_image_edit/inference.py \
        --transformer Skywork/Unipic3-DMD \
        --image_paths qwen_image_edit/example/gemini_pig_remove_hat.png qwen_image_edit/example/gemini_t2i_sunglasses.png \
        --prompt "A pig wearing sunglasses." \
        --true_cfg_scale 4.0 \
        --seed 0 \
        --output_path "output.png"
    
    # Using local checkpoint
    python qwen_image_edit/inference.py \
        --transformer work_dirs/model/transformer \
        --image_paths qwen_image_edit/example/gemini_pig_remove_hat.png \
        --prompt "Remove the hat." \
        --true_cfg_scale 4.0 \
        --seed 0 \
        --output_path "output.png"
    """
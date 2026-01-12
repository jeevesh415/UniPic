from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor
import torch
import argparse
from peft import PeftModel
from PIL import Image

import sys
import os
import glob

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from qwen_image_edit.pipeline_qwenimage_edit_add_seperate import QwenImageEditPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--lora", type=str, default=None,
                       help="LoRA 权重路径")
    parser.add_argument("--transformer", type=str, default=None,
                       help="Transformer 模型路径")
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, 
                       help="图像路径列表，支持多个图像路径")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="编辑提示词")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                       help="CFG 缩放参数，默认为 4.0")
    parser.add_argument("--seed", type=int, default=0,
                       help="随机种子，默认为 0")
    parser.add_argument("--image_size", type=int, default=1024,
                       help="图像尺寸参数，默认为 1024")
    parser.add_argument("--num_outputs", type=int, default=1,
                       help="输出图像数量，默认为 1")
    parser.add_argument("--widths", type=int, nargs='*', default=None,
                       help="输出图像宽度列表，可选参数。如果不指定，则使用 num_outputs 和 image_size 自动计算")
    parser.add_argument("--heights", type=int, nargs='*', default=None,
                       help="输出图像高度列表，可选参数。如果不指定，则使用 num_outputs 和 image_size 自动计算")
    parser.add_argument("--output_prefix", type=str, default="example_ours",
                       help="输出图片文件名前缀，默认为 example_ours")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="推理步数，默认为 50")

    args = parser.parse_args()

    model_name = "Qwen-Image-Edit"
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
    # 检查是否存在 .bin 文件，如果只有 .bin 文件则允许使用 pickle
    transformer_path = transformer if os.path.isabs(transformer) else os.path.join(os.getcwd(), transformer)
    if os.path.isdir(transformer_path):
        bin_files = glob.glob(os.path.join(transformer_path, 'transformer', '*.bin'))
        safetensors_files = glob.glob(os.path.join(transformer_path, 'transformer', '*.safetensors'))
        use_safetensors = len(safetensors_files) > 0
    else:
        use_safetensors = True  # 默认尝试使用 safetensors
    
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=transformer, 
        subfolder='transformer', 
        device_map='auto', 
        torch_dtype=torch.bfloat16,
        use_safetensors=use_safetensors
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
    image_size = args.image_size
    num_outputs = args.num_outputs
    widths = args.widths
    heights = args.heights
    output_prefix = args.output_prefix
    num_inference_steps = args.num_inference_steps

    # 处理输出尺寸：如果没有指定 widths 和 heights，则根据 num_outputs 自动生成
    if widths is None or heights is None:
        if widths is not None or heights is not None:
            raise ValueError("widths 和 heights 必须同时指定或同时不指定")
        # 自动生成：每张图片使用相同的尺寸
        widths = [image_size] * num_outputs
        heights = [image_size] * num_outputs
    else:
        # 验证 widths 和 heights 长度一致
        if len(widths) != len(heights):
            raise ValueError(f"widths 和 heights 的数量必须相同，当前 widths: {len(widths)}, heights: {len(heights)}")
        num_outputs = len(widths)  # 根据指定的尺寸更新输出数量

    print(f"输入图像路径: {image_paths}")
    print(f"编辑提示词: {prompt}")
    print(f"输出图像数量: {num_outputs}")
    print(f"输出宽度列表: {widths}")
    print(f"输出高度列表: {heights}")
    print(f"CFG 缩放参数: {true_cfg_scale}")
    print(f"随机种子: {seed}")
    print(f"图像尺寸: {image_size}")
    print(f"输出文件前缀: {output_prefix}")
    print(f"推理步数: {num_inference_steps}")

    # 加载图像
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # 执行图像编辑
    output_images = pipe(
        images=images,
        widths=widths,
        heights=heights,
        prompt=prompt,
        negative_prompt=' ',
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        image_size=image_size,
        generator=torch.manual_seed(seed)
    ).images

    # 保存输出图像
    for idx, image in enumerate(output_images):
        assert len(image) == 1
        output_path = f"{output_prefix}_{idx}.png"
        image[0].save(output_path)
        print(f"图像 {idx} 已保存到: {output_path}")

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor
import torch
import torch.distributed as dist
import argparse
from peft import PeftModel
from PIL import Image
import json
import os
from tqdm import tqdm
from pathlib import Path

import sys
import os
import glob

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


def setup_distributed():
    """初始化分布式环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def split_data_by_rank(data_list, rank, world_size):
    """将数据按照 rank 分割，每个进程处理数据的一部分"""
    total_samples = len(data_list)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total_samples)
    return data_list[start_idx:end_idx]


def load_jsonl(file_path):
    """读取 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data


def save_results_jsonl(results, output_path):
    """保存推理结果到 JSONL 文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--lora", type=str, default=None,
                       help="LoRA 权重路径（可选）")
    parser.add_argument("--transformer", type=str, default=None,
                       help="Transformer 权重路径（可选）")
    parser.add_argument("--jsonl_path", type=str, required=True,
                       help="输入的 JSONL 文件路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录，用于保存生成的图像")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                       help="CFG 缩放参数，默认为 4.0")
    parser.add_argument("--seed", type=int, default=0,
                       help="随机种子，默认为 0")
    parser.add_argument("--height", type=int, default=1024,
                       help="输出图像高度，默认为 1024")
    parser.add_argument("--width", type=int, default=1024,
                       help="输出图像宽度，默认为 1024")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="推理步数，默认为 50")
    parser.add_argument("--skip_existing", action="store_true",
                       help="如果输出文件已存在则跳过")
    parser.add_argument("--distributed", action="store_true",
                       help="启用多卡分布式推理")

    args = parser.parse_args()

    # 初始化分布式环境
    rank, world_size, local_rank = setup_distributed()
    
    # 设置设备
    if args.distributed and world_size > 1:
        device = torch.device(f'cuda:{local_rank}')
        is_main_process = (rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True

    # 创建输出目录（仅主进程）
    output_dir = Path(args.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 等待主进程创建目录
    if args.distributed and world_size > 1:
        dist.barrier()

    # 读取 JSONL 文件
    if is_main_process:
        print(f"正在读取 JSONL 文件: {args.jsonl_path}")
    data_list = load_jsonl(args.jsonl_path)
    
    # 按 rank 分割数据
    if args.distributed and world_size > 1:
        data_list = split_data_by_rank(data_list, rank, world_size)
        if is_main_process:
            print(f"总数据量: {len(load_jsonl(args.jsonl_path))} 条")
            print(f"使用 {world_size} 个 GPU 进行分布式推理")
        print(f"[Rank {rank}] 负责处理 {len(data_list)} 条数据")
    else:
        if is_main_process:
            print(f"共读取 {len(data_list)} 条数据")

    # 加载模型
    model_name = "Qwen-Image-Edit"
    transformer = model_name if args.transformer is None else args.transformer

    if is_main_process or not args.distributed:
        print(f'正在加载 transformer 权重: {transformer}')
    else:
        print(f'[Rank {rank}] 正在加载模型到 GPU {local_rank}')

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='scheduler'
    )
    
    # 在分布式模式下，每个进程加载模型到指定的GPU
    if args.distributed and world_size > 1:
        text_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name, subfolder='text_encoder',
            torch_dtype=torch.bfloat16
        ).to(device)
    else:
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
    
    # 在分布式模式下，每个进程加载模型到指定的GPU
    if args.distributed and world_size > 1:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=transformer, 
            subfolder='transformer', 
            torch_dtype=torch.bfloat16,
            use_safetensors=use_safetensors
        ).to(device)
    else:
        transformer = QwenImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path=transformer, 
            subfolder='transformer', 
            device_map='auto', 
            torch_dtype=torch.bfloat16,
            use_safetensors=use_safetensors
        )

    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='vae', 
        torch_dtype=torch.bfloat16,
    ).to(device if args.distributed and world_size > 1 else transformer.device)

    if args.lora is not None:
        transformer = PeftModel.from_pretrained(transformer, args.lora)
        if is_main_process or not args.distributed:
            print(f'已加载 LoRA 权重: {args.lora}')

    pipe = QwenImageEditPipeline(
        scheduler=scheduler, 
        vae=vae, 
        text_encoder=text_encoder,
        tokenizer=tokenizer, 
        processor=processor, 
        transformer=transformer
    )

    if is_main_process or not args.distributed:
        print("模型加载完成，开始批量推理...")
    
    # 等待所有进程加载完模型
    if args.distributed and world_size > 1:
        dist.barrier()

    # 批量推理
    results = []
    failed_items = []
    
    # 设置进度条描述
    if args.distributed and world_size > 1:
        progress_desc = f"Rank {rank} 推理进度"
    else:
        progress_desc = "推理进度"

    for idx, item in enumerate(tqdm(data_list, desc=progress_desc)):
        item_id = item.get('id', idx)
        input_images = item['input_images']
        instruction = item['instruction']
        
        # 生成输出文件名
        output_filename = f"{item_id:06d}.png"
        output_path = output_dir / output_filename
        
        # 如果设置了跳过已存在的文件
        if args.skip_existing and output_path.exists():
            if is_main_process or not args.distributed:
                print(f"跳过已存在的文件: {output_path}")
            results.append({
                'id': item_id,
                'output_image': str(output_path.absolute()),
                'status': 'skipped',
                'instruction': instruction,
                'input_images': input_images
            })
            continue

        try:
            rank_prefix = f"[Rank {rank}] " if args.distributed and world_size > 1 else ""
            print(f"\n{rank_prefix}[{idx+1}/{len(data_list)}] 处理 ID: {item_id}")
            print(f"{rank_prefix}  输入图像数量: {len(input_images)}")
            print(f"{rank_prefix}  指令: {instruction[:100]}..." if len(instruction) > 100 else f"{rank_prefix}  指令: {instruction}")
            
            # 加载图像
            images = []
            for img_path in input_images:
                if not os.path.exists(img_path):
                    print(f"{rank_prefix}  警告: 图像文件不存在: {img_path}")
                    raise FileNotFoundError(f"图像文件不存在: {img_path}")
                images.append(Image.open(img_path).convert("RGB"))
            
            # 执行推理
            output_image = pipe(
                images=images,
                height=args.height,
                width=args.width,
                prompt=instruction,
                negative_prompt=' ',
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=torch.manual_seed(args.seed)
            ).images[0]
            
            # 保存图像
            output_image.save(output_path)
            print(f"{rank_prefix}  已保存到: {output_path}")
            
            # 记录结果
            results.append({
                'id': item_id,
                'output_image': str(output_path.absolute()),
                'status': 'success',
                'instruction': instruction,
                'input_images': input_images
            })
            
        except Exception as e:
            print(f"{rank_prefix}  错误: 处理 ID {item_id} 时出错: {str(e)}")
            failed_items.append({
                'id': item_id,
                'error': str(e),
                'instruction': instruction,
                'input_images': input_images
            })
            results.append({
                'id': item_id,
                'output_image': None,
                'status': 'failed',
                'error': str(e),
                'instruction': instruction,
                'input_images': input_images
            })

    # 等待所有进程完成推理
    if args.distributed and world_size > 1:
        dist.barrier()
    
    # 保存结果到 JSONL（自动保存到 output_dir 下）
    if args.distributed and world_size > 1:
        # 在分布式模式下，每个rank保存到不同的文件
        rank_output_path = output_dir / f'results_rank{rank}.jsonl'
        save_results_jsonl(results, str(rank_output_path))
        print(f"\n[Rank {rank}] 推理结果已保存到: {rank_output_path}")
        
        # 等待所有rank保存完成
        dist.barrier()
        
        # 主进程合并所有rank的结果
        if is_main_process:
            print("\n正在合并所有 rank 的结果...")
            all_results = []
            for r in range(world_size):
                rank_file = output_dir / f'results_rank{r}.jsonl'
                if rank_file.exists():
                    rank_results = load_jsonl(str(rank_file))
                    all_results.extend(rank_results)
                    print(f"  已加载 rank {r} 的 {len(rank_results)} 条结果")
            
            # 保存合并后的结果到 results.jsonl
            merged_output_path = output_dir / 'results.jsonl'
            save_results_jsonl(all_results, str(merged_output_path))
            print(f"\n所有结果已合并保存到: {merged_output_path.absolute()}")
            print(f"总计: {len(all_results)} 条结果")
    else:
        # 单卡模式，直接保存到 results.jsonl
        results_path = output_dir / 'results.jsonl'
        save_results_jsonl(results, str(results_path))
        print(f"\n推理结果已保存到: {results_path.absolute()}")

    # 打印统计信息
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = len(failed_items)
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    
    rank_prefix = f"[Rank {rank}] " if args.distributed and world_size > 1 else ""
    print("\n" + "="*50)
    print(f"{rank_prefix}批量推理完成！")
    print(f"{rank_prefix}总数: {len(data_list)}")
    print(f"{rank_prefix}成功: {success_count}")
    print(f"{rank_prefix}失败: {failed_count}")
    print(f"{rank_prefix}跳过: {skipped_count}")
    print(f"{rank_prefix}输出目录: {output_dir}")
    
    if failed_items:
        print(f"\n{rank_prefix}失败的项目:")
        for item in failed_items:
            print(f"{rank_prefix}  ID {item['id']}: {item['error']}")
    
    print("="*50)
    
    # 清理分布式环境
    cleanup_distributed()
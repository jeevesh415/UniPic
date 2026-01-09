# UniPic-3

[English](README.md) | [中文](README_zh.md)

UniPic-3 是基于 Qwen-Image-Edit 的图像编辑模型，采用 Consistency Model 和 Distribution-Matching Distillation 训练方法。

## 📁 文件结构

```
UniPic-3/
├── train_cm.py              # Consistency Model 训练代码
├── train_dmd.py             # Distribution-Matching Distillation 训练代码
├── batch_inference.py       # 批量推理代码
├── pipeline_qwenimage_edit.py  # Pipeline 实现
├── requirements.txt         # Python 依赖包列表
├── qwen_image_edit_fast/   # 核心代码目录
│   ├── train_cm.py          # CM 训练代码
│   ├── train_dmd.py         # DMD 训练代码
│   ├── batch_inference.py   # 批量推理代码
│   ├── pipeline_qwenimage_edit.py  # Pipeline 实现
│   ├── configs/             # 配置文件目录
│   │   └── gemini_all_datasets.py
│   └── scripts/             # 启动脚本目录
│       ├── train_cm.sh      # CM 训练脚本
│       ├── train_dmd.sh     # DMD 训练脚本
│       └── inference.sh     # 推理脚本
```

## 🚀 训练

### Consistency Model (CM) 训练

Consistency Model 训练用于学习图像编辑的一致性表示。

**训练代码**: `qwen_image_edit_fast/train_cm.py`  
**启动脚本**: `qwen_image_edit_fast/scripts/train_cm.sh`

**主要参数**:
- `--guidance_scale`: 引导缩放因子（默认 1.75）
- `--train_steps`: 训练步数（默认 20000）
- `--ckpt_steps`: 检查点保存间隔（默认 1000）
- `--accum_steps`: 梯度累积步数（默认 4）
- `--ema_rate`: EMA 衰减率（默认 0.95）
- `--tangent_norm`: 是否对切向量进行归一化
- `--gradient_checkpointing`: 是否启用梯度检查点

**示例命令**:
```bash
bash qwen_image_edit_fast/scripts/train_cm.sh
```


### Distribution-Matching Distillation (DMD) 训练

Distribution-Matching Distillation 用于从 Consistency Model 进一步蒸馏，提升生成质量。

**训练代码**: `qwen_image_edit_fast/train_dmd.py`  
**启动脚本**: `qwen_image_edit_fast/scripts/train_dmd.sh`

**主要参数**:
- `--guidance_scale`: 引导缩放因子（默认 6.0）
- `--lr_scheduler`: 学习率调度器（默认 cosine）
- `--train_steps`: 训练步数（默认 20000）
- `--ckpt_steps`: 检查点保存间隔（默认 1000）
- `--accum_steps`: 梯度累积步数（默认 1）
- `--gradient_checkpointing`: 是否启用梯度检查点

**示例命令**:
```bash
bash qwen_image_edit_fast/scripts/train_dmd.sh
```

## 🔍 推理和评估

### 批量推理

**推理代码**: `qwen_image_edit_fast/batch_inference.py`  
**启动脚本**: `qwen_image_edit_fast/scripts/inference.sh`

**主要参数**:
- `--jsonl_path`: 输入的 JSONL 文件路径
- `--output_dir`: 输出目录
- `--transformer`: Transformer 权重路径
- `--num_inference_steps`: 推理步数（默认 4）
- `--true_cfg_scale`: CFG 缩放参数（默认 4.0）
- `--distributed`: 是否启用分布式推理
- `--skip_existing`: 是否跳过已存在的文件

**示例命令**:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 --use_env \
    qwen_image_edit_fast/batch_inference.py \
    --jsonl_path data/val.jsonl \
    --output_dir work_dirs/output \
    --distributed \
    --num_inference_steps 4 \
    --true_cfg_scale 4.0 \
    --transformer work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64/step20000/ema_transformer \
    --skip_existing
```

或使用启动脚本:
```bash
bash qwen_image_edit_fast/scripts/inference.sh
```

## 📝 配置文件

配置文件位于 `qwen_image_edit_fast/configs/gemini_all_datasets.py`，包含数据集的配置信息。

## 🔧 环境设置

1. **安装 Python 依赖**:
```bash
pip install -r requirements.txt
```

注意：如果使用 CUDA 版本的 PyTorch，请根据你的 CUDA 版本从 [PyTorch 官网](https://pytorch.org/) 安装对应的版本，然后安装其他依赖：
```bash
pip install -r requirements.txt --no-deps torch torchvision torchaudio
```


2. **设置 PYTHONPATH**（可选，如果代码不在当前目录）:
```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

3. **进入项目目录**:
```bash
cd UniPic-3
# 或者如果从其他位置运行，设置 CODE_DIR 环境变量
export CODE_DIR="$(pwd)/UniPic-3"
```

## 📌 注意事项

1. 训练脚本中的路径需要根据实际环境进行修改
2. 确保有足够的 GPU 内存和存储空间
3. 分布式训练需要正确设置环境变量（`MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT`）
4. 推理时确保输入 JSONL 文件的格式正确

## 📚 相关资源

- **代码地址**: `qwen_image_edit_fast/`
- **训练代码**:
  - Consistency training: `qwen_image_edit_fast/train_cm.py`
  - Distribution-Matching Distillation: `qwen_image_edit_fast/train_dmd.py`
- **推理代码**: `qwen_image_edit_fast/batch_inference.py`
- **Checkpoints**（相对路径，可通过环境变量配置）:
  - Consistency models: `work_dirs/Qwen-Image-Edit-CM-full-20k-bsz256-all-datasets`
  - Distribution-Matching Distillation: `work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64`


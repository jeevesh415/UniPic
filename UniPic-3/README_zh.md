# UniPic-3

[English](README.md) | [中文](README_zh.md)

🔥 **开源 SOTA 多图编辑模型**

UniPic-3 是一个统一的 **单图编辑** 与 **多图组合** 框架，基于 Qwen-Image-Edit，采用 Consistency Model + Distribution-Matching Distillation 训练方法。

| 特性 | 说明 |
|------|------|
| **统一建模** | 单图编辑与多图组合共用同一架构 |
| **灵活输入** | 支持 **1–6 张输入图像**，任意输出分辨率 |
| **快速推理** | **8 步推理**，**12.5× 加速** |
| **高质量数据** | 21.5 万条精选多图组合样本 |

## 📁 文件结构

```
UniPic-3/
├── README.md                    # 英文文档
├── README_zh.md                 # 中文文档
├── requirements.txt             # Python 依赖包列表
├── unipic3.png                  # 模型展示图
│
├── qwen_image_edit/             # Base 模型训练（原版完整步数扩散）
│   ├── dataset.py               # 数据集实现
│   ├── inference.py             # 单次推理
│   ├── pipeline_qwenimage_edit.py  # Pipeline 实现
│   ├── train_fsdp_bsz1.py       # FSDP 训练代码
│   ├── configs/                 # 配置文件目录
│   │   └── gemini_all_datasets.py
│   ├── scripts/
│   │   ├── train.sh             # 训练脚本
│   │   └── inference.sh         # 推理脚本
│   └── example/                 # 示例图片和脚本
│
└── qwen_image_edit_fast/        # CM + DMD 蒸馏训练（快速推理）
    ├── train_cm.py              # Consistency Model 训练
    ├── train_dmd.py             # Distribution-Matching Distillation 训练
    ├── train_cm_dmd.py          # 联合 CM + DMD 训练
    ├── batch_inference.py       # 批量推理
    ├── inference.py             # 单次推理
    ├── pipeline_qwenimage_edit.py  # Pipeline 实现
    ├── configs/                 # 配置文件目录
    │   └── gemini_all_datasets.py
    ├── scripts/
    │   ├── train_cm.sh          # CM 训练脚本
    │   ├── train_dmd.sh         # DMD 训练脚本
    │   └── inference.sh         # 推理脚本
    └── tools/
        └── merge_ckpt.py        # 检查点合并工具
```

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
```

## 🚀 训练

训练流程：**Base 模型 → Consistency Model (CM) → Distribution-Matching Distillation (DMD)**

### 第一步：Base 模型训练（原版完整步数扩散）

使用标准扩散训练方法训练 Base 模型。

**训练代码**: `qwen_image_edit/train_fsdp_bsz1.py`  
**启动脚本**: `qwen_image_edit/scripts/train.sh`

**示例命令**:
```bash
bash qwen_image_edit/scripts/train.sh
```

### 第二步：Consistency Model (CM) 训练

将 Base 模型蒸馏为一致性模型，实现快速推理。

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

### 第三步：Distribution-Matching Distillation (DMD) 训练

从一致性模型进一步蒸馏，提升生成质量。

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

## 🔍 推理

### 模型权重

| 模型 | HuggingFace | 推理步数 |
|------|-------------|---------|
| Base Model | [Skywork/Unipic3](https://huggingface.co/Skywork/Unipic3) | 50 步 |
| Consistency Model | [Skywork/Unipic3-Consistency-Model](https://huggingface.co/Skywork/Unipic3-Consistency-Model) | 8 步 |
| DMD Model | [Skywork/Unipic3-DMD](https://huggingface.co/Skywork/Unipic3-DMD) | 8 步 |

### Base模型推理

**推理代码**: `qwen_image_edit/inference.py`  
**启动脚本**: `qwen_image_edit/scripts/inference.sh`

**主要参数**:
- `--transformer`: Transformer 权重路径（HuggingFace 模型 ID 或本地路径）
  - 使用 `Skywork/Unipic3` 加载 Base 模型
- `--image_paths`: 输入图像路径，支持多张图像
- `--prompt`: 编辑指令文本
- `--true_cfg_scale`: CFG 缩放参数（默认 4.0）
- `--seed`: 随机种子（默认 0）
- `--output_path`: 输出图像路径（默认 example_ours.png）

**示例命令**:
```bash
# 使用 HuggingFace Base 模型（50 步）
python qwen_image_edit/inference.py \
    --transformer Skywork/Unipic3 \
    --image_paths qwen_image_edit/example/gemini_pig_remove_hat.png qwen_image_edit/example/gemini_t2i_sunglasses.png \
    --prompt "A pig wearing sunglasses." \
    --true_cfg_scale 4.0 \
    --seed 0 \
    --output_path "output.png"
```

或使用启动脚本:
```bash
bash qwen_image_edit/scripts/inference.sh
```

### few-steps inference（CM/DMD 模型）

**推理代码**: `qwen_image_edit_fast/batch_inference.py`  
**启动脚本**: `qwen_image_edit_fast/scripts/inference.sh`

**主要参数**:
- `--jsonl_path`: 输入的 JSONL 文件路径
- `--output_dir`: 输出目录
- `--transformer`: Transformer 权重路径（HuggingFace 模型 ID 或本地路径）
  - 使用 `Skywork/Unipic3-DMD/ema_transformer` 加载 DMD 模型（8 步）
  - 使用 `Skywork/Unipic3-Consistency-Model/ema_transformer` 加载 CM 模型（8 步）
- `--num_inference_steps`: 推理步数（默认 8 for CM and DMD）
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
    --num_inference_steps 8 \
    --true_cfg_scale 4.0 \
    --transformer Skywork/Unipic3-DMD/ema_transformer \
    --skip_existing
```

或使用启动脚本:
```bash
bash qwen_image_edit_fast/scripts/inference.sh
```

## 📝 配置文件

配置文件位于:
- `qwen_image_edit/configs/` - Base 模型配置
- `qwen_image_edit_fast/configs/` - CM/DMD 训练配置

## 📌 注意事项

1. 训练脚本中的路径需要根据实际环境进行修改
2. 确保有足够的 GPU 内存和存储空间
3. 分布式训练需要正确设置环境变量（`MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT`）
4. 推理时确保输入 JSONL 文件的格式正确


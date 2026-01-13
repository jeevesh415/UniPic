# UniPic-3

[English](README.md) | [中文](README_zh.md)

🔥 **Open-source SOTA Multi-Image Editing Model**

UniPic-3 is a unified framework for **single-image editing** and **multi-image composition**, built on Qwen-Image-Edit with Consistency Model + Distribution-Matching Distillation.

| Feature | Description |
|---------|-------------|
| **Unified Modeling** | Single-image editing & multi-image composition in one architecture |
| **Flexible Input** | Supports **1–6 input images** with arbitrary output resolutions |
| **Fast Inference** | **8 steps** with **12.5× speedup**, no quality loss |
| **High-Quality Data** | 215K curated multi-image composition samples |

## 📁 File Structure

```
UniPic-3/
├── README.md                    # English documentation
├── README_zh.md                 # Chinese documentation
├── requirements.txt             # Python dependencies
├── unipic3.png                  # Model teaser image
│
├── qwen_image_edit/             # Teacher model training (original full-step diffusion)
│   ├── dataset.py               # Dataset implementation
│   ├── inference.py             # Single inference
│   ├── pipeline_qwenimage_edit.py  # Pipeline implementation
│   ├── train_fsdp_bsz1.py       # FSDP training code
│   ├── configs/                 # Configuration files
│   │   └── gemini_all_datasets.py
│   ├── scripts/
│   │   ├── train.sh             # Training script
│   │   └── inference.sh         # Inference script
│   └── example/                 # Example images and scripts
│
└── qwen_image_edit_fast/        # CM + DMD distillation training (fast inference)
    ├── train_cm.py              # Consistency Model training
    ├── train_dmd.py             # Distribution-Matching Distillation training
    ├── train_cm_dmd.py          # Combined CM + DMD training
    ├── batch_inference.py       # Batch inference
    ├── inference.py             # Single inference
    ├── pipeline_qwenimage_edit.py  # Pipeline implementation
    ├── configs/                 # Configuration files
    │   └── gemini_all_datasets.py
    ├── scripts/
    │   ├── train_cm.sh          # CM training script
    │   ├── train_dmd.sh         # DMD training script
    │   └── inference.sh         # Inference script
    └── tools/
        └── merge_ckpt.py        # Checkpoint merging tool
```

## 🔧 Environment Setup

1. **Install Python Dependencies**:
```bash
pip install -r requirements.txt
```

Note: If using CUDA version of PyTorch, please install the corresponding version from [PyTorch website](https://pytorch.org/) according to your CUDA version, then install other dependencies:
```bash
pip install -r requirements.txt --no-deps torch torchvision torchaudio
```

2. **Set PYTHONPATH** (optional, if code is not in current directory):
```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

3. **Enter Project Directory**:
```bash
cd UniPic-3
```

## 🚀 Training

The training pipeline follows: **Teacher Model → Consistency Model (CM) → Distribution-Matching Distillation (DMD)**

### Step 1: Teacher Model Training (Original Full-Step Diffusion)

Train the teacher model using standard diffusion training with full inference steps.

**Training Code**: `qwen_image_edit/train_fsdp_bsz1.py`  
**Launch Script**: `qwen_image_edit/scripts/train.sh`

**Example Command**:
```bash
bash qwen_image_edit/scripts/train.sh
```

### Step 2: Consistency Model (CM) Training

Distill the teacher model into a consistency model for faster inference.

**Training Code**: `qwen_image_edit_fast/train_cm.py`  
**Launch Script**: `qwen_image_edit_fast/scripts/train_cm.sh`

**Key Parameters**:
- `--guidance_scale`: Guidance scale factor (default: 1.75)
- `--train_steps`: Number of training steps (default: 20000)
- `--ckpt_steps`: Checkpoint saving interval (default: 1000)
- `--accum_steps`: Gradient accumulation steps (default: 4)
- `--ema_rate`: EMA decay rate (default: 0.95)
- `--tangent_norm`: Whether to normalize tangent vectors
- `--gradient_checkpointing`: Whether to enable gradient checkpointing

**Example Command**:
```bash
bash qwen_image_edit_fast/scripts/train_cm.sh
```

### Step 3: Distribution-Matching Distillation (DMD) Training

Further distill from the consistency model to improve generation quality.

**Training Code**: `qwen_image_edit_fast/train_dmd.py`  
**Launch Script**: `qwen_image_edit_fast/scripts/train_dmd.sh`

**Key Parameters**:
- `--guidance_scale`: Guidance scale factor (default: 6.0)
- `--lr_scheduler`: Learning rate scheduler (default: cosine)
- `--train_steps`: Number of training steps (default: 20000)
- `--ckpt_steps`: Checkpoint saving interval (default: 1000)
- `--accum_steps`: Gradient accumulation steps (default: 1)
- `--gradient_checkpointing`: Whether to enable gradient checkpointing

**Example Command**:
```bash
bash qwen_image_edit_fast/scripts/train_dmd.sh
```

## 🔍 Inference

### Model Weights

| Model | HuggingFace |
|-------|-------------|
| Consistency Model | [Skywork/Unipic3-Consistency-Model](https://huggingface.co/Skywork/Unipic3-Consistency-Model) |
| DMD Model | [Skywork/Unipic3-DMD](https://huggingface.co/Skywork/Unipic3-DMD) |

### Batch Inference

**Inference Code**: `qwen_image_edit_fast/batch_inference.py`  
**Launch Script**: `qwen_image_edit_fast/scripts/inference.sh`

**Key Parameters**:
- `--jsonl_path`: Path to input JSONL file
- `--output_dir`: Output directory
- `--transformer`: Path to Transformer weights
- `--num_inference_steps`: Number of inference steps (default: 4)
- `--true_cfg_scale`: CFG scale parameter (default: 4.0)
- `--distributed`: Whether to enable distributed inference
- `--skip_existing`: Whether to skip existing files

**Example Command**:
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

Or use the launch script:
```bash
bash qwen_image_edit_fast/scripts/inference.sh
```

## 📝 Configuration Files

Configuration files are located at:
- `qwen_image_edit/configs/` - Teacher model configs
- `qwen_image_edit_fast/configs/` - CM/DMD training configs

## 📌 Notes

1. Training script paths need to be modified according to the actual environment
2. Ensure sufficient GPU memory and storage space
3. Distributed training requires proper environment variable settings (`MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT`)
4. Ensure the input JSONL file format is correct during inference

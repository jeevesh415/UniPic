# UniPic-3

[English](README.md) | [中文](README_zh.md)

UniPic-3 is an image editing model based on Qwen-Image-Edit, utilizing Consistency Model and Distribution-Matching Distillation training methods.

## 📁 File Structure

```
UniPic-3/
├── train_cm.py              # Consistency Model training code
├── train_dmd.py             # Distribution-Matching Distillation training code
├── batch_inference.py       # Batch inference code
├── pipeline_qwenimage_edit.py  # Pipeline implementation
├── requirements.txt         # Python dependencies
├── qwen_image_edit_fast/   # Core code directory
│   ├── train_cm.py          # CM training code
│   ├── train_dmd.py         # DMD training code
│   ├── batch_inference.py   # Batch inference code
│   ├── pipeline_qwenimage_edit.py  # Pipeline implementation
│   ├── configs/             # Configuration files directory
│   │   └── gemini_all_datasets.py
│   └── scripts/             # Launch scripts directory
│       ├── train_cm.sh      # CM training script
│       ├── train_dmd.sh     # DMD training script
│       └── inference.sh     # Inference script
```

## 🚀 Training

### Consistency Model (CM) Training
Checkpoint: [Skywork/Unipic3-DMD](https://huggingface.co/Skywork/Unipic3-Consistency-Model)

Consistency Model training learns consistent representations for image editing.

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

### Distribution-Matching Distillation (DMD) Training
Checkpoint: [Skywork/Unipic3-DMD](https://huggingface.co/Skywork/Unipic3-DMD)

Distribution-Matching Distillation further distills from the Consistency Model to improve generation quality.

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

## 🔍 Inference and Evaluation

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

Configuration files are located at `qwen_image_edit_fast/configs/gemini_all_datasets.py`, containing dataset configuration information.

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
# Or if running from another location, set CODE_DIR environment variable
export CODE_DIR="$(pwd)/UniPic-3"
```

## 📌 Notes

1. Training script paths need to be modified according to the actual environment
2. Ensure sufficient GPU memory and storage space
3. Distributed training requires proper environment variable settings (`MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT`)
4. Ensure the input JSONL file format is correct during inference

## 📚 Related Resources

- **Code Location**: `qwen_image_edit_fast/`
- **Training Code**:
  - Consistency training: `qwen_image_edit_fast/train_cm.py`
  - Distribution-Matching Distillation: `qwen_image_edit_fast/train_dmd.py`
- **Inference Code**: `qwen_image_edit_fast/batch_inference.py`
- **Checkpoints** (relative paths, configurable via environment variables):
  - Consistency models: `work_dirs/Qwen-Image-Edit-CM-full-20k-bsz256-all-datasets`
  - Distribution-Matching Distillation: `work_dirs/Qwen-Image-Edit-DMD-full-20k-bsz64`

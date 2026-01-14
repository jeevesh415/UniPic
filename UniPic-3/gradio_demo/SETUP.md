# Hugging Face Space Setup Guide

## Quick Start

1. **Copy necessary files to this directory:**

```bash
# Copy pipeline file (if QwenImageEditPipeline is not in diffusers yet)
cp ../UniPic/UniPic-3/qwen_image_edit_fast/pipeline_qwenimage_edit.py .
```

2. **Upload to Hugging Face Space:**

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space
huggingface-cli repo create your-space-name --type space

# Upload files
cd /path/to/gradio
huggingface-cli upload your-space-name app.py requirements.txt README.md
# If needed, also upload pipeline_qwenimage_edit.py
```

## Files Structure

```
gradio_demo/
├── app.py                    # Main Gradio application (required)
├── requirements.txt          # Python dependencies (required)
├── README.md                 # Space description (required)
└── pipeline_qwenimage_edit.py  # Pipeline code (if not in diffusers)
```

## Environment Variables (Optional)

You can set these in HF Space settings:

- `MODEL_NAME`: Base model name (default: "Qwen-Image-Edit")
- `TRANSFORMER_PATH`: DMD model path (default: "Skywork/Unipic3-DMD/ema_transformer")

## Notes

- The pipeline will try to import from `diffusers` first
- If not available, it will fallback to local `pipeline_qwenimage_edit.py`
- Make sure to copy the pipeline file if QwenImageEditPipeline is not in your diffusers version


from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor
import torch
import argparse
from peft import PeftModel
from PIL import Image
import json
import copy
import os
from tqdm import tqdm

from qwen_image_edit.pipeline_qwenimage_edit import QwenImageEditPipeline



from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import Dataset, DataLoader



def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class TryOnDataset(Dataset):
    def __init__(self, data_path, image_folder):
        self.data = load_jsonl(data_path)
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = copy.deepcopy(self.data[idx])
        sample_idx = os.path.basename(data_dict.pop('person'))


        images = [Image.open(os.path.join(self.image_folder, image_path)).convert('RGB')
                  for image_path in data_dict.values()]


        return dict(images=images, sample_idx=sample_idx)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)

    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--transformer", type=str, default=None)

    args = parser.parse_args()

    accelerator = Accelerator()
    # each GPU creates a string
    message = [f"Hello this is GPU {accelerator.process_index}"]
    # collect the messages from all GPUs
    messages = gather_object(message)
    # output the messages only on the main process with accelerator.print()
    accelerator.print(f"Number of gpus: {accelerator.num_processes}")
    accelerator.print(messages)


    print(f'Device: {accelerator.device}', flush=True)

    dataset = TryOnDataset(data_path=args.data, image_folder=args.image_folder)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=lambda x: x
                            )


    model_name = "Qwen/Qwen-Image-Edit"
    transformer = model_name if args.transformer is None else args.transformer

    print(f'Load transformer weights from {transformer}.')

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='scheduler'
    )
    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='text_encoder',
        device_map={"": accelerator.device}, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='tokenizer',
    )
    processor = Qwen2VLProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='processor',
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=transformer, subfolder='transformer',
        device_map={"": accelerator.device}, torch_dtype=torch.bfloat16
    )

    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='vae', torch_dtype=torch.bfloat16,
    ).to(accelerator.device)


    if args.lora is not None:
        transformer = PeftModel.from_pretrained(transformer, args.lora)
        print(f'Load lora weights from {args.lora}.')

    pipe = QwenImageEditPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                                 tokenizer=tokenizer, processor=processor, transformer=transformer)


    dataloader = accelerator.prepare(dataloader)

    print(f'Number of samples: {len(dataloader)}', flush=True)

    if accelerator.is_main_process:
        os.makedirs(args.output, exist_ok=True)


    for batch_idx, data_samples in tqdm(enumerate(dataloader), disable=not accelerator.is_main_process):
        device_idx = accelerator.process_index
        assert len(data_samples) == 1

        data_sample = data_samples[0]
        image = pipe(
            images=data_sample['images'],
            height=1024,
            width=1024,
            prompt='Combine the reference images to generate the final result.',
            negative_prompt=' ',
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.manual_seed(0)
        ).images[0]

        image.save(os.path.join(args.output, data_sample['sample_idx']))

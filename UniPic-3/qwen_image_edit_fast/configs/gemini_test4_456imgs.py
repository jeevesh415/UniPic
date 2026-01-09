from qwen_image_edit.dataset import Unipic3EditDataset, ConcatDataset

dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path='/data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/seedream_4imgs_all.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             ),
        dict(type=Unipic3EditDataset,
             data_path='/data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/seedream_5imgs_all.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             ),
        dict(type=Unipic3EditDataset,
             data_path='/data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/seedream_6imgs_all.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             ),
    ],
)

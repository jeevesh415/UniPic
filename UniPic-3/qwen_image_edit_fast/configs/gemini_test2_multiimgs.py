from qwen_image_edit.dataset import Unipic3EditDataset, ConcatDataset

dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path='/data_genie/genie/why/Qwen-Image-Dev-main/qwen_image_edit/unipic3_2imgs.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             ),
        dict(type=Unipic3EditDataset,
             data_path='/data_genie/genie/why/Qwen-Image-Dev-main/qwen_image_edit/unipic3_3imgs.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             )
    ],
)

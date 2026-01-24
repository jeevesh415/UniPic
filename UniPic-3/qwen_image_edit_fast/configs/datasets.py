from qwen_image_edit.dataset import Unipic3EditDataset, ConcatDataset

dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path='/path/to/unipic3_sft.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             ),
    ],
)

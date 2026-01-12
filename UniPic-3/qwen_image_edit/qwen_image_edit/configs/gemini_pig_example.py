from qwen_image_edit.dataset import Unipic3EditDataset

dataset = dict(type=Unipic3EditDataset,
               data_path='qwen_image_edit/example/data.jsonl',
               image_folder='qwen_image_edit/example',
               image_size=1024,
               unit_image_size=32,
               )

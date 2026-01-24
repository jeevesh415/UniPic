from qwen_image_edit.dataset import Unipic3EditDataset, ConcatDataset
import os

# Data paths can be configured via environment variables
# Example: export DATA_ROOT=/path/to/data
DATA_ROOT = os.environ.get('DATA_ROOT', 'data')

dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path=os.path.join(DATA_ROOT, 'unipic3_sft.jsonl'),
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             ),
    ],
)

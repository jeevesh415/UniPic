from qwen_image_edit.dataset_add_seperate import Unipic3EditDataset, ConcatDataset
import os

# Data paths can be configured via environment variables
DATA_ROOT = os.environ.get('DATA_ROOT', 'data')

dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path=os.path.join(DATA_ROOT, 'cc8m_person_separate_output_part1_filtered.jsonl'),
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             )
    ],
)

from qwen_image_edit.dataset_add_seperate import Unipic3EditDataset, ConcatDataset

dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path='/data_genie/genie/why/Qwen-Image-Dev-main/nano-banana-distill/cc8m_person_separate_output_part1_filtered.jsonl',
             image_folder='',
             image_size=1024,
             unit_image_size=32,
             )
    ],
)

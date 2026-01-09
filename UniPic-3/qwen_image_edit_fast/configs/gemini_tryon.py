from qwen_image_edit.dataset import TryOnDataset

dataset = dict(type=TryOnDataset,
               data_path='/mnt/data_vlm/linsheng/data/zhengchong/DressCode-MR/DressCode-MR/train.jsonl',
               image_folder='/mnt/data_vlm/linsheng/data/zhengchong/DressCode-MR/DressCode-MR',
               image_size=1024,
               unit_image_size=32,
               )

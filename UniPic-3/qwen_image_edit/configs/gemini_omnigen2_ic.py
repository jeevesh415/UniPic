from qwen_image_edit.dataset import Unipic3EditDataset, ConcatDataset


dataset = dict(
    type=ConcatDataset,
    datasets=[
        dict(type=Unipic3EditDataset,
             data_path='/mnt/data_vlm/linsheng/data/OmniGen2/X2I2/jsons/video_icedit/video_icedit.jsonl',
             image_folder='/mnt/data_vlm/linsheng/data/OmniGen2/X2I2/images/video_icedit',
             image_size=1024,
             unit_image_size=32,
             ),
        dict(type=Unipic3EditDataset,
             data_path='/mnt/data_vlm/linsheng/data/OmniGen2/X2I2/jsons/video_icgen/video_icgen.jsonl',
             image_folder='/mnt/data_vlm/linsheng/data/OmniGen2/X2I2/images/video_icgen',
             image_size=1024,
             unit_image_size=32,
             )
    ],
)

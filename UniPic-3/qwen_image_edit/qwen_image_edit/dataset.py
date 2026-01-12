import json
import os
import random
import math
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset as TorchConcatDataset
from mmengine.registry import DATASETS
from copy import deepcopy


def resize_to_multiple_of(image, multiple_of=32):
    width, height = image.size
    width = round(width / multiple_of) * multiple_of
    height = round(height / multiple_of) * multiple_of

    image = image.resize((width, height))

    return image


def load_jsonl(json_file):
    with open(json_file) as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


class Unipic3EditDataset(Dataset):
    def __init__(self,
                 data_path,
                 image_folder='',
                 image_size=1024,
                 unit_image_size=32,
                 ):
        super().__init__()
        self.data_path = data_path
        self._load_data(data_path)
        self.image_folder = image_folder
        self.image_size = image_size
        self.unit_image_size = unit_image_size

    def __len__(self):
        return len(self.data_list)

    def _read_image(self, image_file):
        image = Image.open(
                os.path.join(self.image_folder, image_file)
            )
        assert image.width / image.height > 0.1, f"Image: {image.size}"
        assert image.width / image.height < 10, f"Image: {image.size}"
        return image.convert('RGB')

    def _retry(self):
        return self.__getitem__(random.choice(range(self.__len__())))

    def _load_data(self, data_path: str):      # image path and annotation path are saved in a json file
        if data_path.endswith('.jsonl'):
            self.data_list = load_jsonl(data_path)
        else:
            with open(data_path, 'r') as f:
                self.data_list = json.load(f)
        print(f"Load {len(self.data_list)} data samples from {data_path}", flush=True)


    def _resize_images(self, images):
        # prepare multiple images
        total_number_of_pixels = sum([math.prod(image.size) for image in images])
        ratio = self.image_size / total_number_of_pixels ** 0.5
        images = [image.resize(size=(round(image.width*ratio), round(image.height*ratio))) for image in images]
        images = [resize_to_multiple_of(image=image, multiple_of=self.unit_image_size) for image in images]

        return images

    def __getitem__(self, idx):
        try:
            data_sample = self.data_list[idx]
            input_images = [self._read_image(input_image).convert('RGB')
                            for input_image in data_sample['input_images']]
            input_images = self._resize_images(input_images)

            output_image = self._read_image(data_sample['output_image']).convert('RGB')
            output_image = self._resize_images([output_image])[0]

            prompt = data_sample['instruction'].strip()

            data = dict(
                input_images=input_images, output_image=output_image,
                image_dir=self.image_folder,
                type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()


class TryOnDataset(Unipic3EditDataset):
    PROMPTS = [
        "Combine the reference images to generate the final result.",
        "Merge all reference items into a complete try-on result.",
        "Apply the reference images together to produce the output.",
        "Integrate all references into one coherent try-on result.",
        "Construct the final output by using all provided references.",
        "Assemble the references into the finished try-on result.",
        "Generate the complete output by fusing the reference images.",
        "Create the try-on result by combining all references."
    ]

    def __getitem__(self, idx):
        try:
            data_sample = deepcopy(self.data_list[idx])

            output_image = self._read_image(data_sample.pop('person')).convert('RGB')
            output_image = self._resize_images([output_image])[0]

            input_images = [self._read_image(input_image).convert('RGB')
                            for input_image in data_sample.values()]
            random.shuffle(input_images)
            input_images = self._resize_images(input_images)

            prompt = random.choice(self.PROMPTS)

            data = dict(
                input_images=input_images, output_image=output_image,
                image_dir=self.image_folder,
                type='image2image', text=prompt)

            return data

        except Exception as e:
            print(f"Error when reading {self.data_path}:{self.data_list[idx]}: {e}", flush=True)
            return self._retry()

class ConcatDataset(TorchConcatDataset):

    def __init__(self, datasets):
        datasets_instance = []
        for cfg in datasets:
            datasets_instance.append(DATASETS.build(cfg))
        super().__init__(datasets=datasets_instance)

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += ',\n'.join(
            [f'{repr(dataset)}' for dataset in self.datasets])
        return main_str


if __name__ == '__main__':
    import argparse
    from mmengine.config import Config

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('config', help='config file path.')

    args = parser.parse_args()

    config = Config.fromfile(args.config)
    dataset = DATASETS.build(config.dataset)

    for data_sample in dataset:
        print(data_sample.keys())
        import pdb; pdb.set_trace()


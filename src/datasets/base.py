import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ConvertImageDtype, PILToTensor


def pil_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class BaseClassificationDataset(Dataset):
    def __init__(
        self,
        root,
        mode="train",
        transform=None,
        sample_num=-1,
        seed=1102,
        label_shift=0,
    ):
        self.root = Path(root) / self.dataset_name
        self.mode = mode
        self._data_list, self._class_name_list = self.make_dataset()
        self.transform = transform
        self.rng = np.random.default_rng(seed)
        self.label_shift = label_shift

        if sample_num != -1:
            sample_idx = self.rng.choice(
                len(self._data_list), sample_num, replace=False
            )
            self._data_list = [self._data_list[i] for i in sample_idx]

    @property
    def class_name_list(self):
        return self._class_name_list

    def make_dataset(self):
        """
        data annotation format:
        {
            "data": {
                "train":[
                    [image_path, label],
                    ...
                ],
                "val": [
                    [image_path, label],
                    ...
                ],
                "test": [
                    [image_path, label],
                    ...
                ]
            },
            "class_names": [
                class_0_name,
                class_1_name,
                ...
            ]
        }
        """
        with (self.root / self.annotation_filename).open("r") as f:
            data = json.load(f)

        data_list = []
        for d in data["data"][self.mode]:
            data_list.append(((self.root / "images" / d[0]).as_posix(), d[1]))

        return data_list, data["class_names"]

    def get_class_name(self, class_idx):
        return self._class_name_list[class_idx]

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        path, label = self._data_list[index]
        image = pil_loader(path)

        if self.transform:
            image = self.transform(image)

        return image, label + self.label_shift, index


class BaseUnlabeledDataset(BaseClassificationDataset):
    @property
    def class_name_list(self):
        return None

    def make_dataset(self):
        with (self.root / self.annotation_filename).open("r") as f:
            data = json.load(f)

        data_list = []
        for d in data["data"][self.mode]:
            data_list.append((self.root / "images" / d).as_posix())

        return data_list, None

    def get_class_name(self, _):
        return None

    def __getitem__(self, index):
        path = self._data_list[index]
        # image = pil_loader(path)
        try:
            image = pil_loader(path)
        except:
            with open("error.log", "a") as f:
                f.write(path + "\n")
            image = pil_loader(self._data_list[0])

        if self.transform:
            image = self.transform(image)

        return image, -1, index


class ImageListDataset(BaseClassificationDataset):
    def __init__(self, image_list_path, transform=None, seed=1102, sample_num=-1):
        if not isinstance(image_list_path, Path):
            image_list_path = Path(image_list_path)

        self._data_list = [
            literal_eval(line)
            for line in image_list_path.read_text().strip().split("\n")
        ]
        self.label_shift = 0
        self.transform = transform
        self.rng = np.random.default_rng(seed)

        if sample_num != -1:
            sample_idx = self.rng.choice(
                len(self._data_list), sample_num, replace=False
            )
            self._data_list = [self._data_list[i] for i in sample_idx]

    @property
    def class_name_list(self):
        return None


class NoisyImageListDataset(ImageListDataset):
    def __init__(self, noise_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = torch.load(noise_path)

    def __getitem__(self, idx):
        image, label, _ = super().__getitem__(idx)
        noise = self.noise[idx]

        return image, noise, label, idx

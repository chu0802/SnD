import json
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets import DATASET_MAPPING
from src.datasets.transform import load_transform
from src.utils import get_rank, get_world_size


class DataIterativeLoader:
    def __init__(self, dataloader, device="cuda"):
        self.len = len(dataloader)
        self.dataloader = dataloader
        self.iterator = None
        self.device = device

    def set_epoch(self, epoch):
        if hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch)

    def init(self):
        self.iterator = iter(self.dataloader)

    def __next__(self):
        data = next(self.iterator)
        if isinstance(data, list):
            data = [d.to(self.device) for d in data]
            return data
        else:
            data = data.to(self.device)
            return data

    def __iter__(self):
        return self

    def __len__(self):
        return self.len


def build_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    distributed=False,
):
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=shuffle and not distributed,
        drop_last=drop_last,
    )


def build_iter_dataloader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
    device="cuda",
    distributed=False,
    **kwargs,
):
    dataloader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        drop_last=drop_last,
        distributed=distributed,
    )

    return DataIterativeLoader(dataloader, device=device)


def get_dataloader(
    dataset_name,
    root,
    mode,
    transform,
    sample_num=-1,
    device="cuda",
    seed=1102,
    distributed=False,
    label_shift=0,
    **dataloader_config,
):
    dataset_class = DATASET_MAPPING[dataset_name]

    dataset = dataset_class(
        root,
        mode=mode,
        transform=transform,
        sample_num=sample_num,
        seed=seed,
        label_shift=label_shift,
    )

    distributed = distributed and mode == "train"
    return build_iter_dataloader(
        dataset, **dataloader_config, device=device, distributed=distributed
    )


def get_dataloaders_from_config(config, num_classes_accumulation_dict, device="cuda"):
    dataloaders = {}
    train_transform, eval_transform = load_transform(config)

    for dataloader_type, dataloader_config in config.data.split.items():
        label_shift = num_classes_accumulation_dict[config.data.name]

        dataloaders[dataloader_type] = get_dataloader(
            dataset_name=config.data.name,
            root=config.data.root,
            mode=dataloader_config.split_name,
            transform=train_transform if dataloader_type == "train" else eval_transform,
            sample_num=config.data.get("sample_num", -1),
            device=device,
            distributed=config.task.get("distributed", False),
            label_shift=label_shift,
            **dataloader_config,
        )

    return dataloaders


def load_single_class_name_list(dataset_name: str, data_root: str):
    dataset_class = DATASET_MAPPING[dataset_name]
    name, annotation_filename = (
        dataset_class.dataset_name,
        dataset_class.annotation_filename,
    )

    with (Path(data_root) / name / annotation_filename).open("r") as f:
        data = json.load(f)

    return data["class_names"]


def load_class_name_list(config):
    dataset_list = config.data.get("inference_dataset_list", [config.data.name])
    class_name_list = []
    num_classes_accumulation = []
    for dataset_name in dataset_list:
        class_names = load_single_class_name_list(dataset_name, config.data.root)
        class_name_list += class_names
        num_classes_accumulation.append(len(class_names))
    num_classes_accumulation = [0] + np.cumsum(num_classes_accumulation).tolist()[:-1]

    return class_name_list, dict(zip(dataset_list, num_classes_accumulation))


def get_conceptual_captions(
    config, filename="Validation_GCC-1.1.0-Validation.tsv", size=100
):
    path = Path(config.data.root) / "conceptual_captions" / filename

    df = pd.read_csv(path, sep="\t")

    rng = np.random.default_rng(config.task.seed)
    random_index = rng.choice(
        df.index, size=size if size else len(df.index), replace=False
    )

    return df.iloc[random_index, 0].tolist()

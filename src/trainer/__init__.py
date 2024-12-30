from src.datasets.utils import get_dataloader, load_transform

from .base_trainer import BaseTrainer, BaseKDTrainer
from .snd_trainer import SnDTrainer

TRAINER_MAPPING = {
    "snd": SnDTrainer,
}


def get_kd_trainer(model, dataloaders, config, teacher_models, job_id=None):
    if "ref_dataset" in config.method:
        train_transform, _ = load_transform(config)
        dataset_name, dataloader_config = (
            config.method.ref_dataset,
            config.method.ref_dataset_config,
        )

        dataloaders["ref"] = get_dataloader(
            dataset_name=dataset_name,
            root=config.data.root,
            mode=dataloader_config.split_name,
            transform=train_transform,
            seed=config.task.seed,
            distributed=config.task.get("distributed", False),
            **dataloader_config,
        )

    meta_trainer_class = TRAINER_MAPPING.get(config.method.name, BaseKDTrainer)

    return meta_trainer_class(model, dataloaders, config, teacher_models, job_id)

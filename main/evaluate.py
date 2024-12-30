from src.datasets.utils import get_dataloaders_from_config, load_class_name_list
from src.models.clip import get_model
from src.trainer import BaseTrainer as Trainer
from src.utils import get_config, setup_seeds


def main(config):
    setup_seeds(config.task.seed)

    class_name_list, num_classes_accumulation_dict = load_class_name_list(config)

    model = get_model(
        config, class_name_list, device="cuda", freeze=True, pretrained=False
    )

    dataloaders = get_dataloaders_from_config(config, num_classes_accumulation_dict)

    trainer = Trainer(model, dataloaders, config)

    trainer.logging(
        local_desc="zero shot",
        test_acc=trainer.evaluate(trainer.test_loader),
        use_wandb=False,
    )
    trainer.dump_results(print_result=True)


if __name__ == "__main__":
    config = get_config(mode="evaluate")
    main(config)

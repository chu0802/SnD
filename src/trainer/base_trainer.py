import datetime
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb
from src.trainer.utils import CosineLRScheduler, get_optimizer
from src.utils import AccuracyMeter, dump_config, is_main_process, main_process


class BaseTrainer:
    def __init__(self, model, dataloaders, config, job_id=None):
        self._model = model
        self.dataloaders = dataloaders
        self.config = config
        self._current_num_iterations = 0

        if self.training_mode and job_id:
            self.output_dir = (
                Path(self.config.task.output_dir) / self.config.data.name / job_id
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            dump_config(self.config, self.output_dir / "config.json")

            self.lastest_dir = self.output_dir.parent / "latest"

            if self.lastest_dir.exists():
                # unlink it since it's a symbolic link
                self.lastest_dir.unlink()

            self.lastest_dir.symlink_to(self.output_dir.name)

        self.local_log = defaultdict(dict)

        if self.training_mode:
            self.optimizer = get_optimizer(
                self.unwrapped_model(self.train_model), self.config.task
            )
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer, self.config.task, self.num_total_train_steps
            )

    @main_process
    def save(self, epoch=None):
        # TODO: check if freeze classification head or not

        unwrapped_eval_model = self.unwrapped_model(self.eval_model)

        state_dict = unwrapped_eval_model.get_state_dict()

        save_obj = {"model": state_dict}

        if not epoch:
            epoch = "latest"

        save_path = self.output_dir / f"checkpoint_{epoch}.pth"

        print(f"Saving checkpoint at epoch {epoch} to {save_path}.")
        torch.save(save_obj, save_path)

    @property
    def distributed(self):
        return self.config.task.get("distributed", False)

    @property
    def eval_model(self):
        return self._model

    @property
    def train_model(self):
        return self._model

    @property
    def method_config(self):
        return self.config.method

    @property
    def training_mode(self):
        return self.config.mode == "train"

    @property
    def current_num_iterations(self):
        return self._current_num_iterations

    @current_num_iterations.setter
    def current_num_iterations(self, value):
        self._current_num_iterations = value

    @property
    def num_total_train_steps(self):
        minimum_iterations = max(2 * len(self.train_loader), self.max_iterations)
        return min(self.max_epoch * len(self.train_loader), minimum_iterations)

    @property
    def max_epoch(self):
        return self.config.task.max_epoch

    @property
    def max_iterations(self):
        return self.config.task.max_iterations

    @property
    def lr(self):
        return self.lr_scheduler.current_lr

    @property
    def log_interval(self):
        return self.config.task.log_interval

    @property
    def train_loader(self):
        return self.dataloaders.get("train", None)

    @property
    def val_loader(self):
        return self.dataloaders.get("val", None)

    @property
    def test_loader(self):
        return self.dataloaders.get("test", None)

    def get_current_training_step(self, epoch, local_step):
        return len(self.train_loader) * (epoch - 1) + local_step

    @main_process
    def logging(self, local_desc=None, use_wandb=True, **message_dict):
        if use_wandb:
            wandb.log(message_dict)
        if local_desc is not None:
            self.local_log[local_desc].update(message_dict)

    @main_process
    def dump_results(self, filename="results.json", print_result=False):
        if self.training_mode:
            with open(self.output_dir / filename, "w") as f:
                json.dump(self.local_log, f, indent=4)

        if print_result:
            print(json.dumps(self.local_log))

    def unwrapped_model(self, model):
        if self.distributed and hasattr(model, "module"):
            return model.module
        else:
            return model

    def base_loss(self, images, labels, label_smoothing=0.2, **_):
        outputs = self.train_model(images)
        loss = F.cross_entropy(outputs, labels, label_smoothing=label_smoothing)
        return loss, {"loss": loss.item()}

    def train_step(self, images, labels):
        self.current_num_iterations += 1
        # need to step lr_scheduler first since in this repo I didn't explictly set a learning rate in the optimizer.
        self.lr_scheduler.step()

        loss_fn = getattr(self, f"{self.method_config.name}_loss")
        loss, loss_dict = loss_fn(images, labels, **self.method_config.params)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict.update({"total_loss": loss.item()})
        return loss_dict

    def evaluate(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader

        self.eval_model.eval()

        scores = AccuracyMeter()

        dataloader.init()
        with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
            for images, labels, _ in dataloader:
                preds = self.eval_model(images).argmax(dim=1)
                scores += preds == labels
                pbar.set_postfix_str(f"acc: {100 * scores.acc():.2f}%")
                pbar.update(1)

        return scores.acc()

    def train(self, set_validation=False):
        # test zero-shot validation performance
        if self.val_loader and set_validation:
            self.logging(val_acc=self.evaluate(self.val_loader))

        with tqdm(total=self.num_total_train_steps) as pbar:

            # TODO: make this double for-loop a single for-loop
            for epoch in range(1, self.max_epoch + 1):
                pbar.set_description(f"Epoch {epoch}/{self.max_epoch}: ")

                self.train_model.train()
                self.train_loader.init()

                for i, (images, labels, _) in enumerate(self.train_loader):
                    loss_dict = self.train_step(images, labels)

                    pbar.set_postfix_str(
                        f"lr: {self.lr:.2e}, loss: {loss_dict['total_loss']:.2e}"
                    )
                    pbar.update(1)

                    if i % self.log_interval == 0:
                        self.logging(lr=self.lr, **loss_dict)

                    if self.current_num_iterations >= self.num_total_train_steps:
                        break

                if self.val_loader and set_validation and is_main_process():
                    self.logging(val_acc=self.evaluate(self.val_loader))

                if self.current_num_iterations >= self.num_total_train_steps:
                    self.save(epoch=None)
                    break

                if self.distributed:
                    self.train_loader.set_epoch(epoch)

                # self.save(epoch)


class BaseKDTrainer(BaseTrainer):
    def __init__(self, model, dataloaders, config, teachers, job_id=None):
        super().__init__(model, dataloaders, config, job_id=job_id)
        self.epoch_counter = 0
        self._teachers = teachers
        self.pretrained_teacher_model.eval()

    @property
    def pretrained_teacher_model(self):
        return self._teachers["pretrained"]

    @property
    def ref_loader(self):
        return self.dataloaders["ref"]

    def _get_kd_loss(self, student_logits, teacher_logits, feature_criterion=None, T=2):
        if feature_criterion:
            return feature_criterion(student_logits, teacher_logits)

        soft_labels = nn.functional.softmax(teacher_logits / T, dim=-1)
        return nn.functional.cross_entropy(
            student_logits / T, soft_labels, reduction="mean"
        ) * (T**2)

    def get_ref_data(self, loader, has_noise=False):
        try:
            ref_data = next(loader)
        except StopIteration:
            self.epoch_counter += 1
            loader.init()
            ref_data = next(loader)

            if self.distributed:
                self.ref_loader.set_epoch(self.epoch_counter)

        data, index = ref_data[0], ref_data[-1]
        if has_noise:
            data += ref_data[1]

        return data, index

    def train(self, *args, **kwargs):
        self.dataloaders["ref"].init()
        super().train(*args, **kwargs)

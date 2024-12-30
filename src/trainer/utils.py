import logging

import numpy as np
import torch
import torch.nn as nn


class CosineSimilarityLoss(nn.CosineEmbeddingLoss):
    def forward(self, x, y):
        return super().forward(x, y, torch.ones(x.shape[0]).to(x.device))


class L2Loss(nn.Module):
    def __init__(self, reduce=None, square=False):
        super().__init__()
        self.reduce = reduce
        self.square = square

    def forward(self, x, y):
        loss = torch.pow(torch.norm(x - y, dim=-1), 2)
        if self.square:
            loss = loss**2
        if self.reduce == "mean":
            return loss.mean()
        return loss


def get_optimizer(model, task_config):
    optim_params = model.get_params()

    num_parameters = 0
    for param_group in optim_params:
        for p in param_group["params"]:
            num_parameters += p.data.nelement()
    logging.info(f"number of trainable parameters: {num_parameters}")

    return torch.optim.AdamW(
        optim_params,
        weight_decay=float(task_config.weight_decay),
    )


class CosineLRScheduler(object):
    def __init__(self, optimizer, task_config, num_steps):
        self.current_step = 0
        self.optimizer = optimizer

        init_lrs = task_config.init_lrs
        self.init_lrs = (
            init_lrs
            if isinstance(init_lrs, list)
            else [init_lrs for _ in optimizer.param_groups]
        )

        self.warmup_length = task_config.warmup_length
        self.num_steps = num_steps
        self._current_lr = self.init_lrs[0]

    def step(self):
        for param_group, init_lr in zip(self.optimizer.param_groups, self.init_lrs):
            if self.current_step < self.warmup_length:
                param_group["lr"] = (
                    init_lr * (self.current_step + 1) / self.warmup_length
                )
            else:
                e = self.current_step - self.warmup_length
                es = self.num_steps - self.warmup_length
                param_group["lr"] = 0.5 * (1 + np.cos(np.pi * e / es)) * init_lr

        self.current_step += 1
        self._current_lr = self.optimizer.param_groups[0]["lr"]

    def refresh(self):
        self.current_step = 0

    @property
    def current_lr(self):
        return self._current_lr

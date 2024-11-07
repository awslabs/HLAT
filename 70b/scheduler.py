# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from enum import Enum
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from omegaconf import OmegaConf

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1,
    min_ratio: float = 0.0, plateau_ratio: float = 0.0
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
        min_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum ratio a learning rate would decay to.
        plateau_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The ratio for plateau phase.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        plateau_steps = int(plateau_ratio * num_training_steps)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + plateau_steps:
            return 1.0
        progress = float(current_step - num_warmup_steps - plateau_steps) / float(max(1, num_training_steps - num_warmup_steps - plateau_steps))
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)



## NeMo Config

sched_cfg_dict = {
    "name": "CosineAnnealing",
    "warmup_steps": 2000,
    "constant_steps": 0,
    "min_lr": 3.0e-5,
    "max_steps": 500000,
}
sched_cfg = OmegaConf.create(sched_cfg_dict)

def get_nemo_scheduler(optimizer):
    from nemo.core.optim import prepare_lr_scheduler
    sched_dict = prepare_lr_scheduler(
        optimizer,
        sched_cfg,
    )
    return sched_dict["scheduler"]

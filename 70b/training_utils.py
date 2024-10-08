# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import default_data_collator
from torch.utils.data.dataloader import DataLoader
import datasets

from torch.utils.data import DistributedSampler
from transformers import set_seed

try:
    from lr import CosineAnnealing
except ImportError:
    CosineAnnealing=None

def get_learning_rate_scheduler(optimizer, args, last_epoch=-1):
    lr_scheduler = CosineAnnealing(optimizer, max_steps=args.max_steps, min_lr=args.min_lr, warmup_steps=args.warmup_steps, constant_steps=args.constant_steps, last_epoch=last_epoch)
    return lr_scheduler

def get_param_groups_by_weight_decay(model, weight_decay):
    """Get param groups."""
    if hasattr(model, "local_named_parameters"):
        # Zero1 use the first param in opt to decide the device
        param_optimizer = list(model.local_named_parameters())
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm"]  # gamma/beta are in LayerNorm.weight

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def create_llama_pretraining_dataset(
    data_dir, mini_batch_size, dp_size, dp_rank, seed
):
    #Workaround because python functions are not picklable
    class WorkerInitObj(object):
        def __init__(self, seed):
            self.seed = seed

        def __call__(self, id):
            set_seed(self.seed)
    worker_init = WorkerInitObj(seed)
    train_data = datasets.load_from_disk(data_dir)
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_data,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=mini_batch_size,
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader

def create_partition(num_hidden_layers, pipeline_parallel_size):
    """
    Evenly split the transformer layers between the PP ranks
    """
    assert num_hidden_layers % pipeline_parallel_size == 0
    num_layer_per_partition = num_hidden_layers  // pipeline_parallel_size
    pipeline_cuts = []
    current_cut = num_layer_per_partition - 1
    for i in range(pipeline_parallel_size-1):
        pipeline_cuts.append(f"model.layers.{current_cut}")
        current_cut += num_layer_per_partition
    return pipeline_cuts

def get_sin_cos_matrix(config):
    head_dim = config.hidden_size // config.num_attention_heads
    base = 10000
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None, None, :, :].to(torch.float32), emb.sin()[None, None, :, :].to(torch.float32)

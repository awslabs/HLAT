# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import glob
import random
import torch
import sys
import time
import argparse
import json
import queue
import gc
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime, timezone
from collections import namedtuple
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DistributedSampler
import torch_xla.distributed.parallel_loader as pl
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.xla_backend
import numpy as np
from transformers import (
    AdamW,
    default_data_collator,
    set_seed,
    LlamaConfig,
    LlamaTokenizer,
)
from transformers.optimization import get_linear_schedule_with_warmup

import copy
from torch.utils.tensorboard import SummaryWriter
import inspect
import requests
import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import parallel_state, grads, checkpointing
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed
from neuronx_distributed.utils.model_utils import move_model_to_device
from neuronx_distributed.parallel_layers.checkpointing import save, load
from neuronx_distributed.parallel_layers import mappings
import datasets

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from adamw_fp32_optim_params import AdamW_FP32OptimParams
from modeling_llama_nxd import (
    CoreAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    init_weights,
)
from scheduler import get_cosine_schedule_with_warmup, get_nemo_scheduler
from online_example_packing import ExamplePackDataset
from training_utils import (
    get_param_groups_by_weight_decay,
    get_learning_rate_scheduler,
    create_llama_pretraining_dataset,
    create_partition,
    get_sin_cos_matrix
)

from hf_utils import build_dataset_from_filepaths
# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593
import transformers.modeling_utils as modeling_utils



if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16

datetime_str = str(datetime.now())
results = {
    "inference_success": 1
}


Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])


class TrainingMetrics:
    def __init__(self, json_file):
        self.json_file = json_file

    def read_modify_write_file(self, data, key: str = "metrics") -> None:
        """
        data (dict of training parameters or list of metrics): Data to update in the file.
        key (str): the dictionary key under which data is to be recorded
        """
        result_dict = {}
        print(f"Writing data to the provided results file: {self.json_file}")
        if os.path.exists(self.json_file):
            with open(self.json_file) as json_file:
                result_dict = json.loads(json_file.read()) or result_dict
        print(f"Updating with {key} data: {data}")
        if result_dict:
            try:
                # handle internal named entity if present
                results = result_dict[next(iter(result_dict))]
            except Exception:
                results = result_dict
            current = results.get(key)
            if not current:
                results[key] = data
            else:
                if isinstance(current, list):
                    current.extend(data)
                elif isinstance(current, dict):
                    current.update(data)
        else:
            result_dict["results"] = {key: data}
        with open(self.json_file, "w") as json_file:
            json.dump(result_dict, json_file)

    def store_metrics(self, metrics: List[Metric]) -> None:
        """
        Writes collected metrics to the file.
        """
        data = [
            {
                "MetricName": metric.name,
                "MeasuredValue": metric.value,
                "Units": metric.units,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "AdditionalData": metric.additional_data,
            }
            for metric in metrics
        ]
        self.update(data=data, key="metrics")

    def store_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Writes specified model and configuration parameters to the file.
        """
        self.update(data=parameters, key="parameters")

    def update(self, **kwargs: Any) -> None:
        """
        Write specified data to the output file.
        """
        self.read_modify_write_file(**kwargs)


class Throughput:
    def __init__(
        self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10
    ):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps
        self.moving_avg_window_size = moving_avg_window_size
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        return throughput


class Logger:
    def __init__(self, args, world_size, model_dtype):
        xla = "torch_xla" in sys.modules
        self.throughputs = []
        dtype_short = model_dtype.replace("torch.", "")
        self.tb = SummaryWriter(
            os.path.join(
                args.tb_dir,
                f"neuron_tblogs_{time.strftime('%m%d%y_%H%M')}"
                f"_{dtype_short}"
                f"_w{world_size}"
                f"_lr{args.lr}"
                f"_bs{args.batch_size}"
                f"_acc{args.grad_accum_usteps}"
                f"_warmup{args.warmup_steps}"
                f"_max{args.max_steps}"
                f"_xla{xla}"
                f"_{self.get_instance_type()}",
            )
        )
        self.tb.add_text(
            "script", "```\n" + inspect.getsource(sys.modules[__name__]) + "\n```", 0
        )
        self.golden_steploss = []
        golden = "golden_steploss.txt"
        if os.path.exists(golden):
            with open(golden, "r") as f:
                self.golden_steploss = [float(i) for i in f]
            print(
                f"Read {len(self.golden_steploss)} golden step loss values from {golden}"
            )

    def get_instance_type(self):
        try:
            token = requests.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            )
            data = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-type",
                headers={"X-aws-ec2-metadata-token": token.text},
            )
            return data.text
        except:
            return os.environ.get("HOSTNAME", "unknown")

    def log(self, epoch, step, step_loss, learning_rate, throughput, grad_norm=None, param_norm=None):
        time_now = time.asctime()
        grad_norm_msg = f"grad-norm : {grad_norm}" if grad_norm else ""
        print(
            f"LOG {time_now} - ({epoch}, {step}) step_loss : {step_loss:.4f} "
            f"learning_rate : {learning_rate:.2e} throughput : {throughput:.2f} "
            f"param_norm : {param_norm} "
            f"{grad_norm_msg}",
            flush=True,
        )
        self.tb.add_scalar("step loss", step_loss, step)
        self.tb.add_scalar("reduced_train_loss", step_loss, step)
        self.tb.add_scalar("PPL", math.exp(step_loss), step)
        self.tb.add_scalar("learning rate", learning_rate, step)
        self.tb.add_scalar("lr", learning_rate, step)
        self.tb.add_scalar("throughput", throughput, step)
        # haozheng 0125 param norm:
        if param_norm:
            self.tb.add_scalar("parameter_norm", param_norm, step)
        if grad_norm:
            self.tb.add_scalar("grad-norm", grad_norm, step)
            self.tb.add_scalar("gradient-norm", grad_norm, step)
            self.tb.add_scalar("gradient_norm", grad_norm, step)
        self.throughputs.append(throughput)
        if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            step_0start = step - 1
            if step_0start < len(self.golden_steploss) and step_0start >= 0:
                np.testing.assert_allclose(
                    step_loss, self.golden_steploss[step_0start], rtol=2.3e-1
                )


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        set_seed(self.seed)
        model_parallel_xla_manual_seed(self.seed)


def create_pretraining_dataloader(
    dataset_path, tokenizer_path, mini_batch_size, worker_init, seed, online_enabled, flags
):
    xm.master_print("online_enabled = ", online_enabled)
    if online_enabled:
        xm.rendezvous("create_pretraining_dataloader starts")
        print(
            "{} create_pretraining_dataloader starts".format(time.asctime()),
            flush=True,
        )
        # Haozheng shard: always use build_dataset_from_filepaths to shard dataset
        # Haozheng reshard:
        if not flags.dataset_reshard_enabled:
            raw_dataset = build_dataset_from_filepaths(
                dataset_path,
                num_shards=parallel_state.get_data_parallel_size(),
                shard_idx=parallel_state.get_data_parallel_rank(),
            )
        else:
            raw_dataset = build_dataset_from_filepaths(
                dataset_path,
                num_shards=parallel_state.get_data_parallel_size() / flags.dataset_reshard_size,
                shard_idx=parallel_state.get_data_parallel_rank() / flags.dataset_reshard_size,
            )
        xm.rendezvous("raw_dataset created")
        print(
            "{} raw_dataset created".format(time.asctime()),
            flush=True,
        )
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False, legacy=False)
        # Haozheng shard: do not shard at ExamplePackDataset
        train_dataset = ExamplePackDataset(
            raw_dataset, tokenizer,
            batch_size=mini_batch_size, max_seq_length=4096,
            partition='train', max_batch=None, seed=seed, shard=False,
            reshard_enabled=flags.dataset_reshard_enabled,
            reshard_size=flags.dataset_reshard_size,
        )
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=None,
            batch_size=None,
            num_workers=0,
            worker_init_fn=worker_init,
            pin_memory=True,
        )
        print(
            "{} create_pretraining_dataloader ends".format(time.asctime()),
            flush=True,
        )
    else:
        if len(dataset_path) > 1:
            raise NotImplementedError("Regular dataloader does not support multiple arrow files")
        train_data = datasets.load_from_disk(dataset_path[0])
        train_sampler = DistributedSampler(
            train_data,
            num_replicas=parallel_state.get_data_parallel_size(),
            rank=parallel_state.get_data_parallel_rank(),
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


# Guangtai
def create_validation_dataloader(
    dataset_path, tokenizer_path, mini_batch_size, worker_init, seed
):
    xm.rendezvous("create_validation_dataloader starts")
    raw_dataset = datasets.arrow_dataset.Dataset.from_file(dataset_path)
    xm.rendezvous("val raw_dataset created")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False, legacy=False)
    val_dataset = ExamplePackDataset(raw_dataset, tokenizer, batch_size=mini_batch_size, max_seq_length=4096, partition='valid', max_batch=500, seed=seed)
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=None,
        batch_size=None,
        num_workers=0,
        worker_init_fn=worker_init,
        pin_memory=True,
    )
    print(
        "{} create_validation_dataloader ends".format(time.asctime()),
        flush=True,
    )
    return val_dataloader

def get_model(config):
    model = LlamaForCausalLM(config)
    # Here we make sure we use the same sine and cosine matrices for all layers.
    # Making use of same tensors would make the CSE algorithm eliminate the lookup call
    # from layers, keeping only lookup from first layer.
    with torch.no_grad():
        cos, sin = get_sin_cos_matrix(config)
        for layer in model.model.layers:
            layer.self_attn.rotary_emb.cos_cached = cos
            layer.self_attn.rotary_emb.sin_cached = sin
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    if dist.get_rank() == 0:
        print(f"# total parameters: {num_params}")
        print(f"model config {config}")
    return model

def get_dtype(model) -> str:
    """
    Reference: https://pytorch.org/xla/release/1.12/index.html#xla-tensors-and-bfloat16
    """
    if "XLA_USE_BF16" in os.environ:
        return "torch.bfloat16"
    if "XLA_DOWNCAST_BF16" in os.environ:
        if "torch.float" in str(model.module.original_torch_module.dtype):
            return "torch.bfloat16"
        if "torch.double" in str(model.module.original_torch_module.dtype):
            return "torch.float32"
    return str(model.module.original_torch_module.dtype)

def allreduce_sequence_parallel_gradients(optimizer):
    """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
        Modified from megatron-lm:
        https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """
    from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
    grads = []
    for param_group in optimizer.__getstate__()['param_groups']:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        sequence_parallel_param = getattr(p, 'sequence_parallel_enabled', False)
                        if sequence_parallel_param:
                            grads.append(p.grad.data)
    xm.master_print("# sequence parallel parameters = ", len(grads), local=True)
    for grad in grads:
        # sum v.s. average: sum
        reduce_from_tensor_model_parallel_region(grad)


def checkpoint_training_states_saved(model, optimizer, scheduler, epoch, global_step, output_dir):
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        xm.master_print(f"Skip checkpoint_training_states in compile time")
        return
    checkpoint_start = time.time()
    # haozheng 0126:
    xm.master_print("SAVE_CHECKPOINT")
    # haozheng-1215
    output_dir = os.path.join(output_dir, "saved")
    nxd.save_checkpoint(
        output_dir,
        tag=f"training_states_step_{global_step}",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        user_content={"epoch": epoch, "global_step": global_step},
        use_xser=True,
        async_save=True,  # haozheng async
    )
    checkpoint_end = time.time()
    time_diff = checkpoint_end - checkpoint_start
    xm.master_print(f"Time to checkpoint in minutes: {round(time_diff / 60, 4)}")


def checkpoint_training_states(model, optimizer, scheduler, epoch, global_step, output_dir):
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        xm.master_print(f"Skip checkpoint_training_states in compile time")
        return

    # haozheng 0126:
    xm.master_print("SAVE_CHECKPOINT")

    checkpoint_start = time.time()
    # guangtai-1129
    nxd.save_checkpoint(
        output_dir,
        tag=f"training_states_step_{global_step}",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        user_content={"epoch": epoch, "global_step": global_step},
        use_xser=True,
        num_kept_ckpts=3,
        async_save=True,  # haozheng async
    )
    checkpoint_end = time.time()
    time_diff = checkpoint_end - checkpoint_start
    xm.master_print(f"Time to checkpoint in minutes: {round(time_diff / 60, 4)}")

def checkpoint_model_states(model, output_dir, subdir, flags, grad=False):
    if flags.checkpoint_every_iter_enabled:
        xm.mark_step()
        if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
            xm.master_print(f"Skip checkpoint_model_states in compile time")
            return
        output_dir = os.path.join(flags.output_dir, subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        checkpoint_start = time.time()
        param_dict = {k: v.grad if grad else model.state_dict()[k] for k, v in model.named_parameters()}
        checkpointing.save(param_dict, output_dir)
        checkpoint_end = time.time()
        time_diff = checkpoint_end - checkpoint_start
        xm.master_print(f"Time to checkpoint in minutes: {round(time_diff / 60, 4)}")


from neuronx_distributed.parallel_layers.utils import param_is_not_tensor_parallel_duplicate
from neuronx_distributed.parallel_layers.grads import param_is_not_shared

def calculate_parameter_norm(parameters, norm_type=2):
    """Calculate parameter norms across model parallel ranks
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int): type of the used p-norm. Can be ``'math.inf'`` for
            infinity norm.
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = torch.tensor([float(0.0)], device=xm.xla_device())
    params_to_norm = []

    # Filter parameters based on:
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    for param in parameters:
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if is_not_shared and is_not_tp_duplicate:
            params_to_norm.append(param)

    # Calculate norm.
    if norm_type == math.inf:
        total_norm = max(torch.abs(param) for param in params_to_norm)
        total_norm = torch.tensor([float(total_norm)], device=xm.xla_device())
        # Take max across all model-parallel TPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
        )
        # torch.distributed.all_reduce(
        #     total_norm, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_pipeline_model_parallel_group()
        # )
        total_norm = total_norm[0]
    else:
        for param in params_to_norm:
            param_norm = torch.norm(param, norm_type)
            total_norm += param_norm**norm_type
        # Sum across all model-parallel TPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
        )
        # Haozheng 0124 param norm:
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
        )
        total_norm = torch.pow(total_norm, 1.0 / norm_type)
    return total_norm


def check_optimizer_ckpt_compatible(latest_ckpt_dir):
    optim_path = os.path.join(latest_ckpt_dir, "optim")
    optim_dir = glob.glob(optim_path)
    if len(optim_dir) != 0 and len(optim_dir) != 1:
        raise RuntimeError("Unexpected number of optim_dir: ", len(optim_dir))
    xm.master_print(f"len(optim_dir) = {len(optim_dir)}")
    if len(optim_dir) == 0:
        return False
    optim_tensors = os.path.join(latest_ckpt_dir, "optim", "*.pt.tensors")
    files = glob.glob(optim_tensors)
    dp_degree = (
        len(files) // parallel_state.get_tensor_model_parallel_size()
                   // parallel_state.get_pipeline_model_parallel_size()
    )
    xm.master_print(f"dp_degree is {dp_degree} in ckpt {latest_ckpt_dir}")
    if dp_degree != parallel_state.get_data_parallel_size():
        return False
    else:
        return True


def train_llama(flags):
    if dist.get_rank() == 0:
        print(f"flags {flags}")

    extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None)

    device = xm.xla_device()

    # Set up Llama config
    config = LlamaConfig.from_pretrained(flags.training_config)
    config.use_cache = False
    config.return_dict = False
    config.sequence_parallel_enabled = flags.sequence_parallel_enabled
    config.selective_checkpoint_enabled = True
    config.scaled_init = flags.scaled_init
    config.amp_enabled = flags.amp_enabled
    # Haozheng 0124 TP32:
    config.qkv_linear = True
    config.kv_shared_group_size = 4
    config.max_position_embeddings = max(config.max_position_embeddings, flags.seq_len)
    # if args.num_layer != -1:
    #     config.num_hidden_layers = args.num_layer
    # if args.hidden_size != -1:
    #     config.hidden_size = args.hidden_size

    pipeline_cuts = create_partition(config.num_hidden_layers, flags.pipeline_parallel_size)
    if torch.distributed.get_rank() == 0:
        print(f"pipeline_cuts {pipeline_cuts}")

    # nxd config
    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=flags.tensor_parallel_size,
        pipeline_parallel_size=flags.pipeline_parallel_size,
        pipeline_config={
            "transformer_layer_cls": LlamaDecoderLayer,
            "num_microbatches": flags.batch_size,
            "output_loss_value_spec": (True, False),
            "input_names": ["input_ids", "attention_mask", "labels"],
            "pipeline_cuts": pipeline_cuts,
            "trace_file_path": flags.trace_file_path,
            "param_init_fn": None,
            "leaf_module_cls": [LlamaRMSNorm.__name__],
            "autowrap_modules": [mappings],
            "use_zero1_optimizer": flags.use_zero_1,
            "use_optimizer_wrapper": True,
            # Haozheng 0207 average loss:
            "broadcast_and_average_loss": flags.broadcast_and_average_loss,
            # Haozheng return loss on cpu:
            # Haozheng n0312_hook0 hack:
            "return_loss_on_cpu": flags.return_loss_on_cpu,
        },
        optimizer_config={
            "zero_one_enabled": flags.use_zero_1,
            "grad_clipping": True,
            "max_grad_norm": 1.0,
            "optimizer_dtype": torch.double if flags.use_master_weights_zero_1 else torch.float32,
            "use_grad_acc_hook": flags.use_grad_acc_hook,
        },
        sequence_parallel=flags.sequence_parallel_enabled,
        activation_checkpoint_config=CoreAttention if flags.selective_checkpoint_enabled else "full",
        model_init_config={
            "meta_device_init": True,
            "param_init_fn": init_weights,
            # "sequential_move_factor": 4,
        },
    )

    torch.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    random.seed(flags.seed)
    model_parallel_xla_manual_seed(flags.seed)
    worker_init = WorkerInitObj(flags.seed)

    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    should_print = (
        pp_rank == flags.pipeline_parallel_size - 1 and dp_rank == 0 and tp_rank == 0
    )

    # Create NxD model
    model = nxd.initialize_parallel_model(nxd_config, get_model, config)

    # if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
    #     if flags.tensor_parallel_size == 32:
    #         load("/mnt_out/guangtai/sharded_70b_gpu_init_tp32", model=model, model_key=None)
    #     elif flags.tensor_parallel_size == 8:
    #         load("/mnt_out/fanhaozh/70_llama2/nxd/sharded_70b_gpu_init_tp8_pp16", model=model, model_key=None)
    #     else:
    #         raise ValueError("Not Supported TP size ", flags.tensor_parallel_size)
    #     print(f"GPU 70B loaded...")
    xm.mark_step()

    global_step = 0
    epoch = 0

    model_dtype = get_dtype(model)

    param_groups = get_param_groups_by_weight_decay(model, 0.1)
    if flags.use_mix_precision:
        optimizer_cls = AdamW_FP32OptimParams
    else:
        optimizer_cls = AdamW
    # haozheng amp:
    optimizer = nxd.initialize_parallel_optimizer(
        nxd_config, optimizer_cls, param_groups, lr=flags.lr,
        betas=(0.9, 0.95), eps=flags.optim_eps,
    )
    optimizer.zero_grad()

    if should_print:
        if not os.path.exists(flags.output_dir):
            os.makedirs(flags.output_dir, exist_ok=True)
        if not os.path.exists(flags.tb_dir):
            os.makedirs(flags.tb_dir, exist_ok=True)
        if not extract_graphs_only:
            logger = Logger(flags, dp_size, model_dtype)
        metric_writer = TrainingMetrics(flags.metrics_file)
        throughput = Throughput(
            flags.batch_size, dp_size, flags.grad_accum_usteps
        )
        print("--------TRAINING CONFIG----------")
        print(flags)
        print("--------MODEL CONFIG----------")
        print(model.module.original_torch_module.config)
        # print(model.config)
        print("---------------------------------")
        metric_writer.store_parameters(
            {
                "Model": model.module.original_torch_module.name_or_path,
                "Model configuration": str(model.module.original_torch_module.config),
                # "Model configuration": str(model.config),
                "World size": xm.xrt_world_size(),
                "Data parallel degree": dp_size,
                "Batch size": flags.batch_size,
                "Total steps": flags.steps_this_run,
                "Seed": flags.seed,
                "Optimizer": str(optimizer),
                "Data type": model_dtype,
                "Gradient accumulation microsteps": flags.grad_accum_usteps,
                "Warmup steps": flags.warmup_steps,
                "Dataset": [os.path.basename(os.path.normpath(dp)) for dp in flags.dataset_path],
                "Environment variables": {
                    variable: value
                    for variable, value in os.environ.items()
                    if variable.startswith("NEURON") or variable.startswith("XLA")
                },
            }
        )

    # Guangtai
    @torch.no_grad()
    def val_loop_fn(model, optimizer, val_loader, global_step):
        if val_loader is None:
            return

        model.eval()
        model.zero_grad()

        # val loop
        losses = []
        for _, data in enumerate(val_loader):
            if flags.dataloader_type == "nemo":
                input_ids = data["tokens"]
                attention_mask = data["attention_mask"]
                labels = data["labels"]
                loss_mask = data["loss_mask"]
                # Haozheng: nxd style loss
                # labels = data["tokens"]
            else:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                labels = data["labels"]
                loss_mask = None
            loss = model.run_eval(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if should_print:
                losses.append(loss)

        if should_print:
            loss = torch.stack(losses).mean()
            xm.mark_step()
            loss_cpu = loss.detach().cpu().item()

        if should_print and not extract_graphs_only:
            logger.tb.add_scalar("val loss", loss_cpu, global_step)
            print(f"LOG {time.asctime()} - (0, {global_step}) val_loss: {loss_cpu:.4f}", flush=True)

        model.train()

    # Guangtai
    def train_loop_fn(
        model, optimizer, train_loader, val_loader, epoch, global_step, training_ustep
    ):
        checkpoint_model_states(
            model, flags.output_dir,
            "param", flags, grad=False)
        for _, data in enumerate(train_loader):
            xm.mark_step()
            # Haozheng 0126:
            step_begin_time = datetime.now()
            training_ustep += 1
            # Haozheng: nxd style loss
            if flags.dataloader_type == "nemo":
                input_ids = data["tokens"].to("xla")
                attention_mask = data["attention_mask"].to("xla")
                labels = data["labels"].to("xla")
                loss_mask = data["loss_mask"].to("xla")
                # labels = data["tokens"].to("xla")
                # print for logging
                # loss_mask_cpu = data["loss_mask"].cpu()
                # xm.master_print("loss_mask_cpu.size() = ", loss_mask_cpu.size())
                # xm.master_print("loss_mask_cpu.sum() = ", loss_mask_cpu.sum())
                # xm.master_print("loss_mask_cpu = ", loss_mask_cpu)
                # input_ids_cpu = data["tokens"].cpu()
                # attention_mask_cpu = data["attention_mask"].cpu()
                # labels_cpu = data["labels"].cpu()
                # xm.master_print("input_ids_cpu.size() = ", input_ids_cpu.size())
                # xm.master_print("input_ids_cpu = ", input_ids_cpu)
                # xm.master_print("attention_mask_cpu.size() = ", attention_mask_cpu.size())
                # xm.master_print("attention_mask_cpu = ", attention_mask_cpu)
                # xm.master_print("labels_cpu.size() = ", labels_cpu.size())
                # xm.master_print("labels_cpu = ", labels_cpu)
            else:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                labels = data["labels"]
                loss_mask = None
            loss = model.run_train(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                # loss_mask=loss_mask,
                # output_hidden_states=flags.output_hidden_states
            )

            # guangtai 70b: no logits/hs in pp

            # logits_dir = os.path.join(flags.output_dir, f"logits/")
            # def save_logits(logits):
            #     # Save logits on tensor parallel rank zero, data parallel rank zero and last pipeline parallel stage
            #     if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_data_parallel_rank() == 0:
            #         np.save(os.path.join(logits_dir, f"logits_step{global_step}.npy"), logits.detach().cpu().numpy())
            # if flags.save_logits:
            #     Path(logits_dir).mkdir(parents=True, exist_ok=True)
            #     xm.add_step_closure(save_logits, (outputs.logits,))

            # hs_dir = os.path.join(flags.output_dir, "hs")
            # def save_hidden_states(hs):
            #     # Save logits on tensor parallel rank zero, data parallel rank zero and last pipeline parallel stage
            #     if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_data_parallel_rank() == 0:
            #         for i, x in enumerate(hs):
            #             np.save(os.path.join(hs_dir, f"hs_step{global_step}_layer{i}.npy"), x.detach().cpu().numpy())
            # if flags.output_hidden_states:
            #     Path(hs_dir).mkdir(parents=True, exist_ok=True)
            #     xm.add_step_closure(save_hidden_states, (outputs.hidden_states,))

            # Haozheng 0125 mark_step:
            # xm.mark_step()

            # Haozheng opt:
            # allreduce_sequence_parallel_gradients(optimizer)
            # nemo_opt.allreduce_sequence_parallel_gradients(model)

            # Haozheng: remove mark_step
            # xm.mark_step()

            # checkpoint grad
            checkpoint_model_states(
                model, flags.output_dir,
                f"grad_{global_step}", flags, grad=True
            )

            # ar sp in optimizer wrapper (40 ar)
            # clip gradient in zero-1
            # sync gradient in zero-1
            optimizer.step()

            total_norm = None
            param_norm = None

            # Haozheng 0125 param norm:
            if flags.print_param_norm:
                param_norm = calculate_parameter_norm(list(model.parameters()))

            # Haozheng: grad norm
            # with torch.no_grad():
            #     total_norm = torch.zeros(1, device=device)
            #     if flags.print_grad_norm and is_root:
            #         for p in model.parameters():
            #             param_norm_sq = torch.square(p.grad).sum()
            #             total_norm += param_norm_sq
            #         total_norm = torch.sqrt(total_norm)
            if flags.print_grad_norm:
                if hasattr(optimizer, "grad_norm"):
                    total_norm = optimizer.grad_norm
            else:
                total_norm = None

            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            # Haozheng: loss 1
            def _print_logs(running_loss_reduced_detached, total_norm, param_norm):
                if not extract_graphs_only:
                    if flags.print_grad_norm:
                        total_norm_cpu = total_norm.detach().cpu().item()
                    else:
                        total_norm_cpu = None

                    # Haozheng 0125 param norm:
                    if flags.print_param_norm:
                        param_norm_cpu = param_norm.detach().cpu().item()
                    else:
                        param_norm_cpu = None

                    if should_print:
                        # if flags.print_grad_norm:
                        #     total_norm_cpu = total_norm.cpu().item()
                        # NOTE: The running_loss is the loss of the global_step
                        py_loss = running_loss_reduced_detached.cpu().item()
                        logger.log(
                            epoch,
                            global_step,
                            py_loss,
                            optimizer.param_groups[0]["lr"],
                            throughput.get_throughput(),
                            total_norm_cpu,
                            param_norm_cpu,
                        )
                        # haozheng nan:
                        if math.isnan(py_loss):
                            sys.exit(1)

                    # Haozheng 0126:
                    xm.master_print(f"Epoch {epoch} Batch {global_step} 'batch_time': {(datetime.now() - step_begin_time).total_seconds()},")

            xm.add_step_closure(
                _print_logs, (loss, total_norm, param_norm)
            )

            if global_step >= flags.steps_this_run:
                # NOTE: Prevent runtime "Call to recv failed : Broken pipe" issue
                xm.mark_step()
                break

            # checkpoint params
            checkpoint_model_states(
                model, flags.output_dir,
                f"param_{global_step}", flags, grad=False,
            )

            if flags.checkpoint_frequency > 0 and \
                    global_step % flags.checkpoint_frequency == 0:
                xm.mark_step()
                checkpoint_training_states(model, optimizer, scheduler, epoch, global_step, flags.output_dir)

                # Guangtai
                if val_loader is not None:
                    val_loop_fn(model, optimizer, val_loader, global_step)
                    gc.collect()

            # Haozheng 1215: save model states
            if flags.checkpoint_saved_frequency > 0 and \
                    global_step % flags.checkpoint_saved_frequency == 0:
                xm.mark_step()
                checkpoint_training_states_saved(model, None, scheduler, epoch, global_step, flags.output_dir)

        return (
            global_step,
            training_ustep,
        )


    train_start = time.time()
    training_ustep = 0
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=flags.warmup_steps,
    #     num_training_steps=flags.max_steps,
    #     last_epoch=-1,
    # )
    # Haozheng HP:
    if flags.scheduler_type == "nxd":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=flags.warmup_steps,
            num_training_steps=flags.max_steps,
            last_epoch=-1,
            min_ratio=0.1,
        )
    elif flags.scheduler_type == "nemo":
        scheduler = get_nemo_scheduler(optimizer)
    else:
        raise ValueError(f"Unsupported scheduler type {flags.scheduler_type}")

    # guangtai-1129
    # haozheng 0124 ckpt:
    # if os.path.exists(os.path.join(flags.output_dir, "newest")):

    # haozheng 0207:
    optimizer_to_load = optimizer
    if flags.resume_model_states_only or not flags.checkpoint_load_optimizer:
        optimizer_to_load = None
    scheduler_to_load = scheduler
    if flags.resume_model_states_only:
        scheduler_to_load=None

    if nxd.has_checkpoint(flags.output_dir):
        # Haozheng auto scale:
        user_content = nxd.load_checkpoint(
            flags.output_dir,
        )
        latest_ckpt_dir = os.path.join(flags.output_dir, f'training_states_step_{user_content["global_step"]}')
        if optimizer_to_load is not None:
            optimizer_ckpt_compatible = check_optimizer_ckpt_compatible(latest_ckpt_dir)
            xm.master_print(f"optimizer_ckpt_compatible = {optimizer_ckpt_compatible}")
            if not optimizer_ckpt_compatible:
                optimizer_to_load = None
        user_content = nxd.load_checkpoint(
            flags.output_dir,
            model=model,
            optimizer=optimizer_to_load,
            scheduler=scheduler_to_load,
        )
        if not flags.resume_model_states_only:
            global_step = global_step_resume = user_content["global_step"]
            epoch = user_content["epoch"]
        if flags.dataloader_type == "nemo":
            assert global_step > 100
            epoch = 0
    # TODO: dir+step
    if flags.resume_ckpt:
        user_content = nxd.load_checkpoint(
            flags.resume_ckpt_dir,
            tag=flags.resume_ckpt_tag,
            model=model,
            optimizer=optimizer_to_load,
            scheduler=scheduler_to_load,
        )
        if not flags.resume_model_states_only:
            global_step = global_step_resume = user_content["global_step"]
            epoch = user_content["epoch"]
        if flags.dataloader_type == "nemo":
            assert global_step > 100
            epoch = 0

    # haozheng 0215_amp_resume:
    # Haozheng auto scale:
    # if flags.resume_model_states_only or not flags.checkpoint_load_optimizer:
    if optimizer_to_load is None:
        xm.master_print("optimizer.optimizer.init_zero")
        optimizer.optimizer.init_zero()

    # Haozheng: step_this_run if compilation
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        flags.steps_this_run = global_step + 4
        xm.master_print(f"Compile: steps_this_run = {flags.steps_this_run}")

    for dp in flags.dataset_path:
        assert os.path.exists(
            os.path.expanduser(dp)
        ), "ERROR: Data path {} doesn't exist!".format(dp)

    mini_batch_size = flags.batch_size

    if flags.dataloader_type == "nxd":
        # guangtai 70b:
        train_dataloader = create_pretraining_dataloader(
            flags.dataset_path, flags.tokenizer_path, mini_batch_size, worker_init, flags.seed, flags.online_enabled, flags
        )
        # train_dataloader = create_llama_pretraining_dataset(
        #     "/mnt_out/wzam/llama2/datasets/wikicorpus_llama_v2_tokenized_4k", flags.batch_size, dp_size, dp_rank, flags.seed
        # )
        # Guangtai
        if flags.val_dataset_path:
            val_dataloader = create_validation_dataloader(
                flags.val_dataset_path, flags.tokenizer_path, mini_batch_size, worker_init, flags.seed
            )
        else:
            val_dataloader = None
    else:
        assert flags.dataloader_type == "nemo", "flags.dataloader_type is nxd or nemo"
        # guangtai-1206
        from nemo_dataloader import create_nemo_pretraining_dataloader
        train_dataloader, val_dataloader, _ = create_nemo_pretraining_dataloader(flags.tokenizer_path, global_step)

    # train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
    train_device_loader = train_dataloader
    if val_dataloader is not None:
        # val_device_loader = pl.MpDeviceLoader(val_dataloader, device)
        val_device_loader = val_dataloader
    else:
        val_device_loader = None

    # haozheng 0124 ckpt:
    if nxd.has_checkpoint(flags.output_dir):
        latest_ckpt_dir = os.path.join(flags.output_dir, f'training_states_step_{user_content["global_step"]}')
        # Haozheng i0130:
        if flags.dataloader_type == "nxd" and flags.online_enabled:
            if flags.checkpoint_load_dataset:
                xm.master_print("Loading dataset states from ", latest_ckpt_dir)
                train_dataloader.dataset.load_dataset_states(latest_ckpt_dir, flags.checkpoint_load_dataset_buffer)

    if flags.resume_ckpt and not flags.resume_model_states_only:
        if flags.dataloader_type == "nxd":
            train_dataloader.dataset.load_dataset_states(flags.resume_ckpt_dir)
        elif flags.dataloader_type == "nemo":
            xm.master_print("nemo dataloader needs to be manually resumed")
    if flags.online_enabled:
        train_dataloader.dataset.set_checkpoint_config(flags.output_dir, global_step, flags.grad_accum_usteps, flags.checkpoint_frequency)

    # Haozheng 0126:
    xm.master_print("TRAIN_START")
    while True:
        xm.master_print(
            "Epoch {} begin {}".format(epoch, time.asctime()),
            flush=True,
        )

        global_step, training_ustep = train_loop_fn(
            model,
            optimizer,
            train_device_loader,
            val_device_loader,  # Guangtai
            epoch,
            global_step,
            training_ustep,
        )

        if should_print and not extract_graphs_only:
            final_time = time.time()
            time_diff = final_time - train_start
            print(
                "Epoch {} step {} end {} perf {} seq/sec (at train microstep {} time {} from beginning time {})".format(
                    epoch,
                    global_step,
                    time.asctime(),
                    logger.throughputs[-1],
                    training_ustep,
                    final_time,
                    train_start,
                ),
                flush=True,
            )
            additional_data = {
                "Epoch": epoch,
                "Global step": global_step,
                "Microstep": training_ustep,
            }
            metric_data = [
                Metric(
                    "Throughput", logger.throughputs[-1], "seq/s", additional_data
                ),
            ]
            metric_writer.store_metrics(metric_data)

        if global_step >= flags.steps_this_run:
            if should_print and not extract_graphs_only:
                # record aggregate & final statistics in the metrics file
                additional_data = {
                    "Epoch": epoch,
                    "Global step": global_step,
                    "Microstep": training_ustep,
                }
                average_throughput = round(
                    sum(logger.throughputs) / len(logger.throughputs), 4
                )
                metric_data = [
                    Metric(
                        "Time to train",
                        round(time_diff / 60, 4),
                        "minutes",
                        additional_data,
                    ),
                    Metric(
                        "Average throughput",
                        average_throughput,
                        "seq/s",
                        additional_data,
                    ),
                    Metric(
                        "Peak throughput",
                        max(logger.throughputs),
                        "seq/s",
                        additional_data,
                    ),
                ]
                metric_writer.store_metrics(metric_data)

            # checkpoint_training_states(model, optimizer, scheduler, epoch, global_step, flags.output_dir)
            return

        epoch += 1


def _mp_fn(index, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    train_llama(flags)
    xm.rendezvous("_mp_fn finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="tensor_parallel_size")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="PP size")
    parser.add_argument("--trace_file_path", type=str, default=None)
    parser.add_argument(
        "--training_config",
        type=str,
        help="Model config path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs='+',
        help="Dataset paths.",
    )
    # Guangtai
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="Validation dataset path.",
    )
    parser.add_argument(
        "--dataset_reshard_enabled",
        action="store_true",
        help="Resume from checkpoint at resume_step.",
    )
    parser.add_argument(
        "--dataset_reshard_size",
        type=int,
        default=1,
        help="Resharding size when cluster size increases.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        help="Tokenizer path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--tb_dir",
        type=str,
        default="./tb",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default="results.json",
        help="training metrics results file",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Worker batch size.")
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum total accumulation-steps to run.",
    )
    parser.add_argument(
        "--steps_this_run",
        type=int,
        help="Exit early at <value> steps and not go to max_steps. -1 to mean no early exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12349,
        help="Random seed. Worker seed is this value + worker rank.",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of warmup accumulation-steps for learning rate .",
    )
    parser.add_argument(
        "--grad_accum_usteps",
        type=int,
        default=64,
        help="Gradient accumulation micro-steps (an accumulation-step has <value> micro-steps.",
    )
    parser.add_argument(
        "--print_grad_norm",
        default=False,
        action="store_true",
        help="Whether to print grad norm",
    )
    parser.add_argument(
        "--print_param_norm",
        default=False,
        action="store_true",
        help="Whether to print param norm",
    )
    parser.add_argument(
        "--resume_ckpt",
        action="store_true",
        help="Resume from checkpoint at resume_step."
    )
    parser.add_argument(
        "--resume_ckpt_dir",
        type=str,
        default="./resume_ckpt_dir",
        help="Directory for checkpoints to be loaded.",
    )
    parser.add_argument(
        "--resume_ckpt_tag",
        type=str,
        default=None,
        help="Tag for checkpoints to be loaded.",
    )
    parser.add_argument(
        "--seq_len",
        default=2048,
        type=int,
        help="Sequence length"
    )
    parser.add_argument(
        "--use_mix_precision", action="store_true", help="Use mix precision."
    )
    parser.add_argument(
        "--use_zero_1", action="store_true", help="Use ZeRO-1."
    )
    parser.add_argument(
        "--use_master_weights_zero_1", action="store_true", help="Use ZeRO-1."
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=-1,
        help="Override number of layers for this LLaMA model",
    )
    parser.add_argument(
        "--sequence_parallel_enabled",
        default=False,
        action="store_true",
        help="Enable sequence parallel",
    )
    parser.add_argument(
        "--selective_checkpoint_enabled",
        default=False,
        action="store_true",
        help="Enable selective checkpoint",
    )
    parser.add_argument(
        "--constant_attention_mask",
        default=False,
        action="store_true",
        help="Enable constant attention mask",
    )
    parser.add_argument(
        "--split_linear",
        default=False,
        action="store_true",
        help="Split linear layers",
    )
    parser.add_argument(
        "--online_enabled",
        default=False,
        action="store_true",
        help="Enable online dataloader",
    )
    parser.add_argument(
        "--amp_enabled",
        default=False,
        action="store_true",
        help="Enable amp in llama2 models",
    )
    parser.add_argument(
        "--use_grad_acc_hook",
        default=False,
        action="store_true",
        help="Enable fp32 gradient accumulation",
    )
    parser.add_argument(
        "--dataloader_type",
        type=str,
        default="nxd",
        help="Dataloader type, nxd or nemo",
    )
    parser.add_argument(
        "--checkpoint_every_iter_enabled",
        default=False,
        action="store_true",
        help="Enable online dataloader",
    )
    parser.add_argument(
        "--save_logits",
        default=False,
        action="store_true",
        help="Save logits",
    )
    parser.add_argument(
        "--output_hidden_states",
        default=False,
        action="store_true",
        help="Output hidden states",
    )
    parser.add_argument(
        "--resume_model_states_only",
        default=False,
        action="store_true",
        help="Enable online dataloader",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=-1,
        help="The frequency of global step to checkpoint the training state",
    )
    parser.add_argument(
        "--checkpoint_saved_frequency",
        type=int,
        default=-1,
        help="The frequency of global step to checkpoint the model state",
    )
    parser.add_argument(
        "--checkpoint_load_dataset",
        default=False,
        action="store_true",
        help="Load checkpoint for datasets",
    )
    parser.add_argument(
        "--checkpoint_load_dataset_buffer",
        default=False,
        action="store_true",
        help="Load checkpoint for dataset buffer. Effective only when checkpoint_load_dataset is set",
    )
    parser.add_argument(
        "--checkpoint_load_optimizer",
        action="store_true",
        help="Load checkpoint for optimizer",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="nxd",
        help="Scheduler type, nxd or nemo",
    )
    parser.add_argument("--initializer_range", type=float, default=0.02, help="Initializer range.")
    parser.add_argument(
        "--scaled_init",
        default=False,
        action="store_true",
        help="Use scaled init in MLP and MHA outputs",
    )
    parser.add_argument(
        "--broadcast_and_average_loss",
        default=False,
        action="store_true",
        help="Average loss across DP ranks",
    )
    parser.add_argument(
        "--return_loss_on_cpu",
        default=False,
        action="store_true",
        help="Return loss value on cpu",
    )
    parser.add_argument("--optim_eps", type=float, default=1e-5, help="Optimizer eps.")
    args = parser.parse_args(sys.argv[1:])

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    # os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"
    # haozheng amp:
    if args.use_mix_precision or args.use_master_weights_zero_1:
        os.environ["XLA_DOWNCAST_BF16"]="1"
    else:
        os.environ["XLA_USE_BF16"]="1"

    # WORLD_SIZE is set by torchrun
    if os.environ.get("WORLD_SIZE"):
        dist.init_process_group("xla")
        _mp_fn(0, args)
    else:
        xmp.spawn(_mp_fn, args=(args,))

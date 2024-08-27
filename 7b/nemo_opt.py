import itertools
from typing import Any, List, Optional, Union

import numpy as np
import torch
if torch.__version__.startswith('2'):
    from torch import inf
else:
    from torch._six import inf
import time
import math
import os
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

# from nemo.collections.nlp.modules.common.megatron.clip_grads import (
#     clip_grad_norm_fp32,
# )

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import param_is_not_tensor_parallel_duplicate
from neuronx_distributed.parallel_layers.grads import param_is_not_shared
import torch_xla.core.xla_model as xm



## Gradient Clipping
def clip_grad_norm_fp32(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.
    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    # import pdb; pdb.set_trace()
    xm.mark_step()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Filter parameters based on:
    #   - grad should not be none
    #   - parameter should not be shared
    #   - should not be a replica due to tensor model parallelism
    grads = []
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none:
            grad = param.grad.detach()
            # Make sure the grads are in fp32
            # assert isinstance(param.grad, torch.cuda.FloatTensor)
            grads.append(grad)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    if not grads_for_norm:
        raise ValueError(f"No grads found, please disable gradient clipping {xm.get_ordinal()}")

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # total_norm = 0.0
    total_norm = torch.cuda.FloatTensor([float(0.0)])

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_tensor_model_parallel_group()
        )
        # torch.distributed.all_reduce(
        #     total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_pipeline_model_parallel_group()
        # )
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            # dummy_overflow_buf = torch.cuda.IntTensor([0])
            # # Use apex's multi-tensor applier for efficiency reasons.
            # # Multi-tensor applier takes a function and a list of list
            # # and performs the operation on that list all in one kernel.
            # grad_norm, _ = multi_tensor_applier(
            #     amp_C.multi_tensor_l2norm, dummy_overflow_buf, [grads_for_norm], False  # no per-parameter norm
            # )
            # # Since we will be summing across data parallel groups,
            # # we need the pow(norm-type).
            # total_norm = grad_norm ** norm_type
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_tensor_model_parallel_group()
        )
        # torch.distributed.all_reduce(
        #     total_norm, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_pipeline_model_parallel_group()
        # )
        # total_norm = total_norm.item() ** (1.0 / norm_type)
        total_norm = torch.pow(total_norm, 1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)
    # if clip_coeff < 1.0:
    #     dummy_overflow_buf = torch.cuda.IntTensor([0])
    #     multi_tensor_applier(amp_C.multi_tensor_scale, dummy_overflow_buf, [grads, grads], clip_coeff)
    for g in grads:
        g.data.mul_(torch.where(clip_coeff < 1, clip_coeff, torch.tensor(1., device=xm.xla_device())))

    xm.mark_step()
    return total_norm


def _get_parameters(model):
    """
    private method to load all the trainable parameters from optimizer param groups
    """
    return list(model.parameters())
    # params = []
    # for param_group in self._optimizer_param_groups:
    #     for param in param_group['params']:
    #         params.append(param)
    # return params

def configure_gradient_clipping(clip_val, model):
    """PTL hook to configure gradients.
        We use gradient clipping implementation from megatron-lm.
    """
    # clip_val = self.trainer.gradient_clip_val
    # if clip_val is None or self.wrap_with_zero:
    #     # Zero1 optimizer handles gradient clipping for us across TP groups
    #     return

    clip_val = float(clip_val)
    if clip_val <= 0:
        return

    parameters = _get_parameters(model)
    grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)
    return grad_norm
    # if self.grad_clip_pl_default:
    #     # use the default behavior
    #     return super().configure_gradient_clipping(*args, **kwargs)

    # debug_prefix = f'/workspace/example_datasets/llama_debug_neuron_nemo/{self.random_id}'
    # os.makedirs(debug_prefix, exist_ok=True)
    # for _ite in [0, 1]:
    #     os.makedirs(f"{debug_prefix}/ite{_ite}", exist_ok=True)
    #     for _dparams in ['weights','grads']:
    #         os.makedirs(f"{debug_prefix}/ite{_ite}/{_dparams}", exist_ok=True)
    # _step = self.trainer.global_step

    # for name, param in self.model.named_parameters():
    #     fname = f"{debug_prefix}/ite{_step}"
    #     _rank = parallel_state.get_tensor_model_parallel_rank()
    #     joblib.dump(param.cpu().detach().numpy(), f"{fname}/weights/rank_{_rank}_name_{name}.joblib")
    #     joblib.dump(param.grad.cpu().detach().numpy(), f"{fname}/grads/rank_{_rank}_name_{name}.joblib")

    # if self.with_distributed_adam:
    #     grad_norm = clip_grad_norm_distributed_optimizer(self._optimizer, clip_val)
    # else:
        # if self.megatron_amp_o2:
        #     # grep fp32 master parameters for gradient clipping
        #     parameters = self._optimizer.get_parameters()
        # else:
        #     parameters = self._get_parameters()
        # grad_norm = clip_grad_norm_fp32(parameters=parameters, max_norm=clip_val)

    # for name, param in self.model.named_parameters():
    #     fname = f"{debug_prefix}/ite{_step}"
    #     _rank = parallel_state.get_tensor_model_parallel_rank()
    #     joblib.dump(param.cpu().detach().numpy(), f"{fname}/weights_2/rank_{_rank}_name_{name}.joblib")
    #     joblib.dump(param.grad.cpu().detach().numpy(), f"{fname}/grads_2/rank_{_rank}_name_{name}.joblib")

    # guangtai4
    # def _log_grad_norm(log_fn, grad_norm):
    #     log_fn('grad_norm', grad_norm.cpu(), rank_zero_only=True)
    # xm.add_step_closure(_log_grad_norm, (self.log, grad_norm.detach(),))


def _append_sequence_parallel_module_grads(module, grads):
    """ Helper method for allreduce_sequence_parallel_gradients"""

    for param in module.parameters():
        # if getattr(self, 'transformer_engine', False):
        #     sequence_parallel_param = getattr(param, 'sequence_parallel', False)
        # else:
        #     sequence_parallel_param = getattr(param, 'sequence_parallel_enabled', False)
        sequence_parallel_param = getattr(param, 'sequence_parallel_enabled', False)
        if sequence_parallel_param:
            #megatron_amp_o2 also uses model gradients
            grad = param.grad
            grads.append(grad.data)


## Sequence Parallel

def allreduce_sequence_parallel_gradients(model):
    """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
        Modified from megatron-lm:
        https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """

    # guangtai4
    grads = []
    if isinstance(model, list):
        for module in model:
            _append_sequence_parallel_module_grads(module, grads)
    else:
        _append_sequence_parallel_module_grads(model, grads)

    coalesced = torch._utils._flatten_dense_tensors(grads)
    torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
    for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)

## Gradient Sync

_ALLREDUCE_BUCKET_CAP_MB = 512

def bucket_allreduce(tensor_list):
    bucket_cap = int(os.getenv('BUCKET_CAP_KB', _ALLREDUCE_BUCKET_CAP_MB))*1024*1024
    # Reverse the gradients list so that we start allreduce from the last layer
    # onwards. This allows allreduce to trigger as soon as the bucket fills up and
    # overlap with backward pass.
    gradients = reversed(tensor_list)
    total = 0
    tensor_bucket = []

    for grad in gradients:
        grad.data /= parallel_state.get_data_parallel_size()
        grad_bytes = grad.numel() * grad.element_size()

        # Gradient is larger than bucket_cap, don't bucketize
        if grad_bytes > bucket_cap:
            # Flush out previous buckets even if they don't fill up
            # This maintains the strict reverse ordering
            if len(tensor_bucket):
                xm.all_reduce('sum', tensor_bucket, groups = parallel_state.get_data_parallel_group()._mesh)
                total = 0
                tensor_bucket = []
            xm.all_reduce('sum', [grad], groups = parallel_state.get_data_parallel_group()._mesh)
            continue

        # Bucketize till the total spills over
        total += grad_bytes
        if total > bucket_cap:
            xm.all_reduce('sum', tensor_bucket, groups = parallel_state.get_data_parallel_group()._mesh)
            total = grad_bytes
            tensor_bucket = []
        tensor_bucket.append(grad)

    # Flush the last remaining bucket
    if len(tensor_bucket):
        xm.all_reduce('sum', tensor_bucket, groups = parallel_state.get_data_parallel_group()._mesh)


def allreduce_gradients(model):
    """Reduce gradients across data parallel ranks.
        Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/model/distributed.py#L188
    """
    # guangtai4
    # Bucketize and all-reduce
    buckets = {}
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            tp = param.dtype
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(param)
            # param.main_grad = param.grad

    # For each bucket, all-reduce and copy all-reduced grads.
    for tp in buckets:
        bucket = buckets[tp]
        grads = [param.grad.data for param in bucket]
        bucket_allreduce(grads)
        # coalesced = torch._utils._flatten_dense_tensors(grads)
        # coalesced /= parallel_state.get_data_parallel_world_size()
        # torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_parallel_group())
        # for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
        #     buf.copy_(synced)
    # xm.reduce_gradients(
    #     self._optimizer, groups=parallel_state.get_data_parallel_group()._mesh
    # )

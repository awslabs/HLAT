# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from torch.utils.data import IterableDataset

from neuronx_distributed.parallel_layers import parallel_state


def compute_indices(world_size, current_rank, dataset_length, seed):
    print('computing the index for rank - {}'.format(current_rank))
    np.random.seed(seed)
    all_indices = np.arange(dataset_length, dtype=np.uint32)
    np.random.shuffle(all_indices)
    n_sample_per_rank = int(dataset_length / world_size)
    if current_rank < world_size-1:
        indices = all_indices[n_sample_per_rank*current_rank: n_sample_per_rank*(current_rank+1)]
    else:
        indices = all_indices[n_sample_per_rank*current_rank:dataset_length]
    return indices


def select_batch_by_dp(batch, world_size, current_rank, reshard_size):
    if set(batch.keys()) != set(["input_ids", "attention_mask", "labels"]):
        raise ValueError(f'batch keys are {batch.keys()}, while only input_ids, attention_mask, and labels are expected')
    if reshard_size != 2:
        raise NotImplementedError(f"ExamplePackDataset: we support reshard_size == 2 only, while reshard_size = {reshard_size}")
    batch_size = batch["input_ids"].size()[0]
    if batch_size != batch["attention_mask"].size()[0] or batch_size != batch["labels"].size()[0]:
        raise ValueError(f'batch_size does not match: \
                           batch_size = {batch_size}, \
                           attention_mask.size() = {batch["attention_mask"].size()}, \
                           labels.size() = {batch["labels"].size()}')
    return {
        "input_ids": batch["input_ids"].chunk(reshard_size)[current_rank % reshard_size],
        "attention_mask": batch["attention_mask"].chunk(reshard_size)[current_rank % reshard_size],
        "labels": batch["labels"].chunk(reshard_size)[current_rank % reshard_size],
    }


# Haozheng shard: add shard parameter
class ExamplePackDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, batch_size, max_seq_length, partition, max_batch, seed,
                 shard=False, reshard_enabled=False, reshard_size=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.seed = seed

        self.reshard_enabled = reshard_enabled
        self.reshard_size = reshard_size
        if shard and self.reshard_enabled:
            raise ValueError("ExamplePackDataset: shard and reshard_enabled cannot be enabled together")
        if self.reshard_enabled:
            if reshard_size != 2:
                raise NotImplementedError(f"ExamplePackDataset: we support reshard_size == 2 only, while reshard_size = {reshard_size}")
            self.mini_batch_size = self.batch_size
            self.batch_size = self.batch_size * reshard_size
        else:
            self.mini_batch_size = self.batch_size

        # Haozheng shard: add shard parameter
        if shard:
            self.dataset_indices = compute_indices(parallel_state.get_data_parallel_size(), parallel_state.get_data_parallel_rank(), len(self.dataset), seed)
        else:
            self.dataset_indices = list(range(len(self.dataset)))
        # total number of samples to be processed by 1 gpu
        self.length = len(self.dataset_indices)
        # batch size in terms of number of tokens
        self.batch_size_token = self.batch_size * self.max_seq_length
        # the index of the sample to read from the dataset
        self.current_index = 0
        # buffer of one or more samples before tokenization, used after reading long sequences
        self.buffer = []
        self.buffer_length = 0
        # buffer of extra tokens generated after tokenization, compared to batch_size_token
        self.buffer_token = []
        self.buffer_token_length = 0
        self.partition = partition
        self.current_epoch = 0
        self.batch_index = 0
        self.max_batch = max_batch
        # max number of characters in a sequence that can be processed by tokenzier without any issues
        self.max_char_length = 500000

    def _randomize_index(self):
        seed = self.seed + self.current_epoch
        np.random.seed(seed)
        np.random.shuffle(self.dataset_indices)
        return

    def _split_sample(self, sample):
        # split the input sample into multiple samples of length self.max_char_length
        # or less (in case of last sample)
        n_chars = len(sample)
        n_sample = int(n_chars/self.max_char_length)
        sample_list = []
        for i in range(n_sample):
            sample_list.append(sample[i* self.max_char_length : (i+1)* self.max_char_length])
        # if more words left, append as last sample
        if n_sample* self.max_char_length < n_chars:
            sample_list.append(sample[n_sample* self.max_char_length :])
        return sample_list

    def _read_samples(self):
        # read sample with at least self.batch_size_token number of words,
        # in excess of (self.buffer_length + self.buffer_token_length)
        n_word = self.buffer_length + self.buffer_token_length
        samples = []
        while n_word < self.batch_size_token:
            if self.partition == 'train':
                # when we reach the end of the dataset, update self.current_epoch and randomize
                if self.current_index >= self.length:
                    self.current_epoch += 1
                    self.current_index = 0
                    self._randomize_index()
            else:
                # when we reach the end of the dataset, stop iterating
                if self.current_index >= self.length:
                    self.current_index = 0
                # changing the condition to avoid having NCCL timeout error. It happens because each gpu gets different
                # number of batches due to sample based data split (equal no of samples does not imply equal number of
                # batches). This problem is severe with more splits of dev set, because of the smaller size in each set.
                if self.batch_index >= self.max_batch:
                    self.batch_index = 0
                    raise StopIteration
            current_sample = self.dataset.select([self.dataset_indices[self.current_index]])["text"][0]
            self.current_index += 1

            current_len = len(current_sample)
            if current_len > self.max_char_length:
                # split current_sample into multiple samples before appending
                sample_list = self._split_sample(current_sample)
                for seq in sample_list:
                    # add the splits to samples until self.batch_size_token words are in
                    # samples, then add rest of them to the self.buffer for use in the
                    # following steps (in case of long documents)
                    if n_word < self.batch_size_token:
                        samples.append(seq)
                        n_word += len(seq.split())
                    else:
                        self.buffer.append(seq)
                        self.buffer_length += len(seq.split())
            else:
                samples.append(current_sample)
                # count the number of words in the current sample
                n_word += len(current_sample.split())

        return samples

    def _tokenize_samples(self, samples):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # tokenize the input text samples
        batch = self.tokenizer(
            samples,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
            add_special_tokens=False
        )

        # if self.buffer_token is not empty, start from there
        if self.buffer_token_length > 0:
            input_ids = self.buffer_token
        else:
            input_ids = []

        # add bos_token and eos_token around each sample, and concatenate
        for ids in batch["input_ids"]:
            # Gopher style
            #ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
            # GPT3 style
            ids = ids + [self.tokenizer.eos_token_id]
            input_ids += ids

        # check if we have enough tokens to construct a batch
        assert len(input_ids) >= self.batch_size_token

        # keep self.batch_size_token tokens in input_ids and keep the
        # extra tokens in self.buffer_token to use in next step
        self.buffer_token = input_ids[self.batch_size_token: ]
        self.buffer_token_length = len(self.buffer_token)
        input_ids = input_ids[0: self.batch_size_token]
        return input_ids

    def _create_batch(self, input_ids):
        input_ids = torch.LongTensor(input_ids)
        input_ids = input_ids.reshape(self.batch_size, self.max_seq_length)
        batch = {}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = torch.ones(input_ids.size(), dtype=torch.int64)
        batch["labels"] = batch["input_ids"].clone()

        return batch

    def __next__(self):

        # read samples according to batch size, (check length of each sample, if larger than threshold
        # split into multiple samples) tokenize, concatenate, check if equal or more than desired batch
        # size (in number of tokens). use batch_size number of tokens to create a batch (make tensor, add
        # an extra dimension as done by tokenizer) and keep the rest in buffer. optionally use the buffer
        # in the following batch or discard it if less that number of tokens_per_sample

        if self.partition != 'train' and self.batch_index >= self.max_batch:
            self.batch_index = 0
            raise StopIteration

        # if the current buffer doesn't have enough tokens to form a batch, read new samples
        if (self.buffer_length + self.buffer_token_length) < self.batch_size_token:
            samples = self._read_samples()
            input_ids = self._tokenize_samples(self.buffer + samples)
            batch = self._create_batch(input_ids)
            self.buffer = []
            self.buffer_length = 0

        # if self.buffer_token has enough tokens to create one or few batches, use
        # them here to create batches one at a time. this happens because we don't
        # have good estimate on how many samples to read for a batch.
        elif self.buffer_token_length >= self.batch_size_token:
            batch = self._create_batch(self.buffer_token[:self.batch_size_token])
            self.buffer_token = self.buffer_token[self.batch_size_token:]
            self.buffer_token_length -= self.batch_size_token

        # (rare case: when we have encountered a very large sequence in the last step)
        else:
            if len(self.buffer) > self.batch_size:
                batch = self._tokenize_samples(self.buffer[0:self.batch_size])
                self.buffer = self.buffer[self.batch_size:]
                self.buffer_length -= self.batch_size_token
            else:
                batch = self._tokenize_samples(self.buffer)
                self.buffer = []
                self.buffer_length = 0

        self.batch_index += 1
        if hasattr(self, "micro_steps"):
            self._save_dataset_states()
        # Haozheng reshard:
        if self.reshard_enabled:
            return select_batch_by_dp(
                batch,
                parallel_state.get_data_parallel_size(),
                parallel_state.get_data_parallel_rank(),
                self.reshard_size
            )
        else:
            return batch

    def __iter__(self):
        return self

    def set_checkpoint_config(self, output_dir, global_steps, num_grad_acc, frequency):
        self.output_dir = output_dir
        self.micro_steps = 0
        self.global_steps = global_steps
        self.num_grad_acc = num_grad_acc
        self.frequency = frequency

    def _save_dataset_states(self):
        self.micro_steps += 1
        if self.micro_steps % self.num_grad_acc == 0:
            self.global_steps += 1
            if self.global_steps % self.frequency == 0:
                # Haozheng reshard:
                # if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_pipeline_model_parallel_rank() == 0:
                reshard_should_save = not self.reshard_enabled or parallel_state.get_data_parallel_rank() % self.reshard_size == 0
                if (reshard_should_save and
                    parallel_state.get_tensor_model_parallel_rank() == 0 and
                    parallel_state.get_pipeline_model_parallel_rank() == 0):
                    chkpt_path = os.path.join(self.output_dir, f"training_states_step_{self.global_steps}")
                    os.makedirs(chkpt_path, exist_ok=True)
                    # Haozheng reshard:
                    if self.reshard_enabled:
                        chkpt_path = os.path.join(chkpt_path, "dataset.dp_rank_{:02d}".format(parallel_state.get_data_parallel_rank() // self.reshard_size))
                    else:
                        chkpt_path = os.path.join(chkpt_path, "dataset.dp_rank_{:02d}".format(parallel_state.get_data_parallel_rank()))
                    dataset_states = {
                        "current_index": self.current_index,
                        "current_epoch": self.current_epoch,
                        "buffer": copy.deepcopy(self.buffer),
                        "buffer_length": self.buffer_length,
                        "buffer_token": copy.deepcopy(self.buffer_token),
                        "buffer_token_length": self.buffer_token_length,
                    }
                    torch.save(dataset_states, chkpt_path)
                xm.rendezvous("dataset states saved {}".format(self.micro_steps))

    def load_dataset_states(self, output_dir, checkpoint_load_dataset_buffer=True):
        chkpt_path = output_dir
        # Haozheng reshard:
        if self.reshard_enabled:
            chkpt_path = os.path.join(chkpt_path, "dataset.dp_rank_{:02d}".format(parallel_state.get_data_parallel_rank() // self.reshard_size))
        else:
            chkpt_path = os.path.join(chkpt_path, "dataset.dp_rank_{:02d}".format(parallel_state.get_data_parallel_rank()))
        dataset_states = torch.load(chkpt_path)
        self.current_index = dataset_states["current_index"]
        self.current_epoch = dataset_states["current_epoch"]
        if checkpoint_load_dataset_buffer:
            xm.master_print("Loading dataset buffer from ", chkpt_path)
            self.buffer = dataset_states["buffer"]
            self.buffer_length = dataset_states["buffer_length"]
            self.buffer_token = dataset_states["buffer_token"]
            self.buffer_token_length = dataset_states["buffer_token_length"]
        else:
            xm.master_print("Loading dataset buffer is disabled")
        if self.current_epoch > 0:
            self._randomize_index()
        xm.rendezvous("dataset states loaded")

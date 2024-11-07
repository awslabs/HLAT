#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_NAME=$(basename $SCRIPT_DIR)

if [ $# -lt 1 ]; then
    echo "Usage: $0 [ut|ut_819|ut100|pretrain_nemo|pretrain]"
    exit 1
fi

TASK=$1

if [ $1 != "ut" ] && [ $1 != "ut_819" ] && [ $1 != "ut100" ] && [ $1 != "pretrain" ] && [ $1 != "pretrain_nemo" ]; then
    echo "Usage: $0 [ut|ut_819|ut100|upretrain_nemo|pretrain]"
    exit 1
fi

#############################################
# User defined parameters and env vars

export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training"
# export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training"
# export NEURON_CC_FLAGS="--model-type=transformer --enable-experimental-O1 --enable-internal-call-graph  --enable-mixed-precision-accumulation --enable-saturate-infinity --retry_failed_compilation"

# export NEURON_RT_DBG_A2A_CC=0
# export NEURON_RT_ASYNC_EXEC_MODE=1

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_RT_ENABLE_VERBOSE_NUMERICAL_ERRORS=0
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
# export MALLOC_ARENA_MAX=128

export XLA_USE_BF16=1
export XLA_DOWNCAST_BF16=0

# Async Runtime
# Guangtai:
# export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1

# Haozheng debug: XLA SYNC WAIT
# export XLA_SYNC_WAIT=1

# HOST OOM
export MALLOC_ARENA_MAX=64

# Runtime
export NEURON_RT_EXEC_TIMEOUT=4000

# open file
ulimit -n 131071

# TP degree
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=0
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=500000
# tokenizer path
TOKENIZER_PATH="/fsx_out/guangtai/llama2/weights/7B-hf"
# sequence length
SEQ_LEN=4096
# initializer range
INITIALIZER_RANGE=0.021
# optimizer eps
OPTIM_EPS=1.0e-8

echo "TASK = $TASK"

if [ $TASK = "ut" ]; then
    echo "Launch ut..."
    # data path
    DATA_PATH="/mnt_out/fanhaozh/llama2/ut_dataset"
    # validation data path
    VAL_DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    ONLINE_ENABLED=0
    # resume ckpt dir
    RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/weights/7B-nxd-qkv"
    # resume model states only
    RESUME_MODEL_STATES_ONLY=1
    # gloabl batch size
    GBS=16
    # warmup steps
    WARMUP_STEPS=1
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=-1
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=-1
    # learning rate
    LR=3.0e-5
    # step this run
    STEPS_THIS_RUN=3
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=1
elif [ $TASK = "ut_819" ]; then
    echo "Launch ut_819..."
    # data path
    DATA_PATH="/mnt_out/fanhaozh/llama2/ut_dataset"
    # validation data path
    VAL_DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    ONLINE_ENABLED=0
    # resume ckpt dir
    RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/ckpts/step819_nxd"
    # resume model states only
    RESUME_MODEL_STATES_ONLY=1
    # gloabl batch size
    GBS=16
    # warmup steps
    WARMUP_STEPS=1
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=-1
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=-1
    # learning rate
    LR=1.5e-4
    # step this run
    STEPS_THIS_RUN=3
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=1
    # save logits
    SAVE_LOGITS=1
    # output hidden states
    OUTPUT_HIDDEN_STATES=1
elif [ $TASK = "pretrain_nemo" ]; then
    echo "Launch pretrain_nemo..."
    # data path
    DATA_PATH="/mnt_out/fanhaozh/llama2/ut_dataset"
    # validation data path
    VAL_DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    # ONLINE_ENABLED=0
    # dataloader type
    DATALOADER_TYPE="nemo"
    # resume ckpt dir
    # RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/n1128_n2k/output/training_states_step_5000"
    # resume model states only
    # RESUME_MODEL_STATES_ONLY=1
    # gloabl batch size
    GBS=1024
    # warmup steps
    WARMUP_STEPS=2000
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=1000
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=23842
    # learning rate
    LR=3.0e-4
    # scheduler type
    SCHEDULER_TYPE="nemo"
    # step this run
    STEPS_THIS_RUN=-1
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=0
elif [ $TASK = "ut100" ]; then
    echo "Launch ut100..."
    # data path
    DATA_PATH="/mnt/dataset/SlimPajama-627B/train.arrow"
    # validation data path
    VAL_DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    ONLINE_ENABLED=1
    # resume ckpt dir
    RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/weights/7B-nxd-qkv"
    # resume model states only
    RESUME_MODEL_STATES_ONLY=1
    # gloabl batch size
    GBS=1024
    # warmup steps
    WARMUP_STEPS=1
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=50
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=-1
    # learning rate
    LR=3.0e-5
    # step this run
    STEPS_THIS_RUN=150
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=0
else
    echo "Launch pretrain..."
    # data path
    # DATA_PATH="/mnt/dataset/SlimPajama-627B/train.arrow"
    # DATA_PATH="your own dataset path"
    DATA_PATH="/fsx_out/fanhaozh/dataset/arxiv.arrow"
    # validation data path
    VAL_DATA_PATH="/fsx_data/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    ONLINE_ENABLED=1
    # resume ckpt dir
    # RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/g1020_ld23k/output/training_states_step_52000"
    # resume model states only
    # RESUME_MODEL_STATES_ONLY=0
    # gloabl batch size
    GBS=1024
    # warmup steps
    WARMUP_STEPS=2000
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=1000
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=23842
    # learning rate
    LR=3.0e-4
    # step this run
    STEPS_THIS_RUN=-1
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=0
fi

#############################################
# distributed args

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
# export CCOM_SOCKET_IFNAME=eth0

if [ -v SLURM_NNODES ]
then
    # SLURM runs
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    export MASTER_ADDR=${HOSTS[0]}
    export MASTER_PORT=41000
elif [ -v OMPI_COMM_WORLD_RANK ]
then
    # MPI runs
    NODELIST=`/nodelist_helper.py`
    HOSTS=(${NODELIST//\ / })
    NODEID=$OMPI_COMM_WORLD_RANK
    NTASKS=$OMPI_COMM_WORLD_SIZE
    export MASTER_ADDR=${HOSTS[0]}
    export MASTER_PORT=41000
elif [ -v WORLD_SIZE ]
then
    # PyTorchJob
    NTASKS=$WORLD_SIZE
    NODEID=$RANK
    if [ $MASTER_ADDR = "localhost" ]; then
        export MASTER_ADDR=$(hostname)
    fi
else
    # Single-node, non-SLURM, non-MPI runs
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export MASTER_ADDR=${HOSTS[0]}
    export MASTER_PORT=41000
fi

export PROCESSES_PER_NODE=32

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo "DISTRIBUTED_ARGS = $DISTRIBUTED_ARGS"

sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620,41000,23456

#############################################
# extra args

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi
if [ $ONLINE_ENABLED -gt 0 ]; then
    EXTRA_ARGS+=" --online_enabled"
fi
if [ ! -z "$RESUME_CKPT_DIR" ]; then
    EXTRA_ARGS+=" --resume_ckpt --resume_ckpt_dir $RESUME_CKPT_DIR"
fi
if [ "$RESUME_MODEL_STATES_ONLY" -gt 0 ]; then
    EXTRA_ARGS+=" --resume_model_states_only"
fi
if [ "$CHECKPOINT_EVERY_ITER_ENABLED" -gt 0 ]; then
    EXTRA_ARGS+=" --checkpoint_every_iter_enabled"
fi
if [ "$SAVE_LOGITS" -gt 0 ]; then
    EXTRA_ARGS+=" --save_logits"
fi
if [ "$OUTPUT_HIDDEN_STATES" -gt 0 ]; then
    EXTRA_ARGS+=" --output_hidden_states"
fi
if [ ! -z "$DATALOADER_TYPE" ]; then
    EXTRA_ARGS+=" --dataloader_type $DATALOADER_TYPE"
fi
if [ ! -z "$SCHEDULER_TYPE" ]; then
    EXTRA_ARGS+=" --scheduler_type $SCHEDULER_TYPE"
fi
if [ ! -z "$INITIALIZER_RANGE" ]; then
    EXTRA_ARGS+=" --initializer_range $INITIALIZER_RANGE"
fi
if [ ! -z "$OPTIM_EPS" ]; then
    EXTRA_ARGS+=" --optim_eps $OPTIM_EPS"
fi

DP=$(($PROCESSES_PER_NODE * $NTASKS / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    # checkpoint compilation step
    STEPS_THIS_RUN=4
    CHECKPOINT_FREQUENCY=3
    CHECKPOINT_SAVED_FREQUENCY=-1
    OUTPUT_LOG=$SCRIPT_DIR/log_compile-$NODEID.log
else
    OUTPUT_LOG=$SCRIPT_DIR/log_exe-$NODEID.log
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo DATA_PATH=$DATA_PATH
echo VAL_DATA_PATH=$VAL_DATA_PATH
echo TOKENIZER_PATH=$TOKENIZER_PATH
echo TRN_STATE_PATH=$TRN_STATE_PATH
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

# Haozheng ckpt1: local output to speedup: --output_dir $HOME \
torchrun $DISTRIBUTED_ARGS \
    $SCRIPT_DIR/tp_zero1_llama2_7b_hf_pretrain.py \
    --training_config $SCRIPT_DIR/7b_config.json \
    --dataset_path $DATA_PATH \
    --val_dataset_path $VAL_DATA_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --checkpoint_frequency $CHECKPOINT_FREQUENCY \
    --checkpoint_saved_frequency $CHECKPOINT_SAVED_FREQUENCY \
    --output_dir $SCRIPT_DIR/output \
    --sequence_parallel_enabled \
    --print_grad_norm \
    --constant_attention_mask \
    --selective_checkpoint_enabled \
    --tb_dir /fsx_out/tensorboard/fanhaozh/$SCRIPT_DIR_NAME \
    $EXTRA_ARGS

# --resume_ckpt_dir /mnt_out/fanhaozh/llama2/golden_test_load/resume_ckpt_dir_360 \

ret=${PIPESTATUS[0]}
echo "return code from torchrun: $ret"
exit $ret

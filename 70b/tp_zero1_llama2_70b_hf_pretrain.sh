#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_NAME=$(basename $SCRIPT_DIR)

if [ -z $JOB_NAME ]; then
    LOG_DIR=$SCRIPT_DIR/log_default
else
    LOG_DIR=$SCRIPT_DIR/log_$JOB_NAME
fi

echo "LOG_DIR = $LOG_DIR"
mkdir -p $LOG_DIR/

if [ $# -lt 1 ]; then
    echo "Usage: $0 [wiki|pretrain|ut]"
    exit 1
fi

TASK=$1

if [ $1 != "wiki" ] && [ $1 != "pretrain" ] && [ $1 != "ut" ]; then
    echo "Usage: $0 [wiki|pretrain|ut]"
    exit 1
fi

#############################################
# User defined parameters and env vars

# haozheng amp:
export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --enable-mixed-precision-accumulation"
# export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training"
# export NEURON_CC_FLAGS="--model-type=transformer --enable-experimental-O1 --enable-internal-call-graph  --enable-mixed-precision-accumulation --enable-saturate-infinity --retry_failed_compilation"

# export NEURON_RT_DBG_A2A_CC=0
# export NEURON_RT_ASYNC_EXEC_MODE=1

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_STOCHASTIC_ROUNDING_EN=0
# export NEURON_RT_ENABLE_VERBOSE_NUMERICAL_ERRORS=0
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=7
# export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export MALLOC_ARENA_MAX=128

# haozheng amp:
export XLA_USE_BF16=0
export XLA_DOWNCAST_BF16=1

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
# Haozheng 0124 TP32:
# TP_DEGREE=8
TP_DEGREE=32
# PP degree
# Haozheng 0124 TP32:
# PP_DEGREE=16
PP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
# Haozheng 0124 TP32:
# USE_ZERO_1=0
USE_ZERO_1=1
# haozheng use_grad_acc_hook:
USE_GRAD_ACC_HOOK=1
# 0: bf16 optimizer; 1: use fp32 master weights
# haozheng master weights optimizer:
USE_MASTER_WEIGHTS_ZERO_1=1
# haozheng amp:
AMP_ENABLED=1
# haozheng return_loss_on_cpu:
RETURN_LOSS_ON_CPU=0
# tokenizer path
TOKENIZER_PATH="/mnt_out/guangtai/llama2/weights/7B-hf"
# sequence length
SEQ_LEN=4096
# initializer range
INITIALIZER_RANGE=0.021
# optimizer eps
OPTIM_EPS=1.0e-8

echo "TASK = $TASK"


if [ $TASK = "ut" ]; then
    echo "Launch ut..."
    # # data path
    # DATA_PATH="/mnt_out/fanhaozh/llama2/ut_dataset"
    # # dataloader
    # ONLINE_ENABLED=0
    # data path
    DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    ONLINE_ENABLED=1
    # total steps
    TOTAL_STEPS=500000
    # resume ckpt dir
    RESUME_CKPT_DIR="/mnt_out/zhuha/70_llama2/nxd"
    # resume ckpt tag
    RESUME_CKPT_TAG="model_nxd"
    # resume model states only
    RESUME_MODEL_STATES_ONLY=1
    # gloabl batch size on 16 nodes (acc = 32)
    GBS=128
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
    CHECKPOINT_EVERY_ITER_ENABLED=0
    # average loss
    BROADCAST_AND_AVERAGE_LOSS=1
elif [ $TASK = "wiki" ]; then
    echo "Launch wiki..."
    # data path
    # DATA_PATH="/mnt/dataset/SlimPajama-627B/train.arrow"
    # DATA_PATH="/mnt/penshi/redpajama/arxiv.arrow /mnt/penshi/redpajama/c4.arrow /mnt/penshi/redpajama/common_crawl_2021-04.arrow /mnt/penshi/redpajama/github.arrow /mnt/penshi/redpajama/arxiv_book_stackexchange_github_wikipedia.arrow /mnt/penshi/redpajama/common_crawl_2019-03.arrow /mnt/penshi/redpajama/common_crawl_2022-05.arrow /mnt/penshi/redpajama/stackexchange.arrow /mnt/penshi/redpajama/book.arrow /mnt/penshi/redpajama/common_crawl_2020-05.arrow /mnt/penshi/redpajama/common_crawl_2023-06.arrow /mnt/penshi/redpajama/wikipedia.arrow"
    DATA_PATH="/mnt_out/wzam/llama2/datasets/wikicorpus_llama_v2_tokenized_4k/"
    # validation data path
    # VAL_DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    VAL_DATA_PATH=""
    # dataloader
    ONLINE_ENABLED=0
    # total steps
    TOTAL_STEPS=30000
    # resume ckpt dir
    # RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/g1020_ld23k/output/training_states_step_52000"
    # resume model states only
    # RESUME_MODEL_STATES_ONLY=0
    # gloabl batch size
    GBS=512
    # warmup steps
    WARMUP_STEPS=2000
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=1000
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=23842
    # learning rate
    LR=1.5e-4
    # step this run
    STEPS_THIS_RUN=-1
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=0
    # scaled init
    SCALED_INIT_ENABLED=0
else
    echo "Launch pretrain..."
    # data path
    # DATA_PATH="/mnt/dataset/SlimPajama-627B/train.arrow"
    # DATA_PATH="/mnt/penshi/redpajama/arxiv.arrow /mnt/penshi/redpajama/c4.arrow /mnt/penshi/redpajama/common_crawl_2021-04.arrow /mnt/penshi/redpajama/github.arrow /mnt/penshi/redpajama/arxiv_book_stackexchange_github_wikipedia.arrow /mnt/penshi/redpajama/common_crawl_2019-03.arrow /mnt/penshi/redpajama/common_crawl_2022-05.arrow /mnt/penshi/redpajama/stackexchange.arrow /mnt/penshi/redpajama/book.arrow /mnt/penshi/redpajama/common_crawl_2020-05.arrow /mnt/penshi/redpajama/common_crawl_2023-06.arrow /mnt/penshi/redpajama/wikipedia.arrow"
    DATA_PATH="/mnt_out/guangtai/dataset/redpajama/train.arrow"
    # validation data path
    # VAL_DATA_PATH="/mnt/dataset/SlimPajama-627B/val.arrow"
    # dataloader
    ONLINE_ENABLED=1
    # Haozheng reshard
    # dataset reshard
    DATASET_RESHARD_ENABLED=0
    # dataset reshard
    # DATASET_RESHARD_ENABLED=1
    # dataset reshard size
    # DATASET_RESHARD_SIZE=2
    # total steps
    TOTAL_STEPS=500000
    # resume ckpt dir
    # RESUME_CKPT_DIR="/mnt_out/fanhaozh/llama2/g1020_ld23k/output/training_states_step_52000"
    # resume model states only
    # RESUME_MODEL_STATES_ONLY=0
    # gloabl batch size
    GBS=1024
    # warmup steps
    WARMUP_STEPS=2000
    # checkpoint frequency
    CHECKPOINT_FREQUENCY=300
    # checkpoint saved frequency
    CHECKPOINT_SAVED_FREQUENCY=23842
    # learning rate
    LR=1.5e-4
    # step this run
    STEPS_THIS_RUN=-1
    # checkpoint every iter
    CHECKPOINT_EVERY_ITER_ENABLED=0
    # checkpoint load for dataset
    CHECKPOINT_LOAD_DATASET=1
    # checkpoint load for dataset buffer
    CHECKPOINT_LOAD_DATASET_BUFFER=0
    # checkpoint load for optimizer
    # Haozheng 0207 do not load optimizer:
    CHECKPOINT_LOAD_OPTIMIZER=1
    # scaled init
    SCALED_INIT_ENABLED=1
    # average loss across DP ranks
    BROADCAST_AND_AVERAGE_LOSS=1
fi

#############################################
# distributed args

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export CCOM_SOCKET_IFNAME=eth0

export PROCESSES_PER_NODE=32

if [ -v SLURM_NNODES ]
then
    echo "Slurm job with world size $SLURM_NNODES"
    JOBTYPE="SLURM"
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
    echo "MPIJob with world size $OMPI_COMM_WORLD_SIZE"
    JOBTYPE="MPI"
    NODELIST=`/nodelist_helper.py`
    HOSTS=(${NODELIST//\ / })
    NODEID=$OMPI_COMM_WORLD_RANK
    NTASKS=$OMPI_COMM_WORLD_SIZE
    export MASTER_ADDR=${HOSTS[0]}
    export MASTER_PORT=41000
elif [ -v WORLD_SIZE ]
then
    if [ -z "$RANK" ]; then
        # Haozheng fc:
        echo "Fault Controller with world size $WORLD_SIZE"
        JOBTYPE="FC"
        if [ "$(( $WORLD_SIZE % $PROCESSES_PER_NODE ))" -ne 0 ]; then
            exit 1
        fi
        NTASKS=$(($WORLD_SIZE / $PROCESSES_PER_NODE))
    else
        echo "PyTorchJob with world size $WORLD_SIZE"
        JOBTYPE="PT"
        NTASKS=$WORLD_SIZE
        NODEID=$RANK
        if [ $MASTER_ADDR = "localhost" ]; then
            export MASTER_ADDR=$(hostname)
        fi
    fi
else
    # Single-node, non-SLURM, non-MPI runs
    echo "Single node run"
    HOSTS=(localhost)
    NODEID=0
    NTASKS=1
    export MASTER_ADDR=${HOSTS[0]}
    export MASTER_PORT=41000
fi

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
if [ $USE_GRAD_ACC_HOOK -gt 0 ]; then
    EXTRA_ARGS+=" --use_grad_acc_hook"
fi
if [ $USE_MASTER_WEIGHTS_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_master_weights_zero_1"
fi
if [ $AMP_ENABLED -gt 0 ]; then
    EXTRA_ARGS+=" --amp_enabled"
fi
if [ $RETURN_LOSS_ON_CPU -gt 0 ]; then
    EXTRA_ARGS+=" --return_loss_on_cpu"
fi
if [ $ONLINE_ENABLED -gt 0 ]; then
    EXTRA_ARGS+=" --online_enabled"
fi
if [ ! -z "$RESUME_CKPT_DIR" ]; then
    EXTRA_ARGS+=" --resume_ckpt --resume_ckpt_dir $RESUME_CKPT_DIR"
fi
if [ ! -z "$RESUME_CKPT_TAG" ]; then
    EXTRA_ARGS+=" --resume_ckpt --resume_ckpt_dir $RESUME_CKPT_DIR  --resume_ckpt_tag  $RESUME_CKPT_TAG"
fi
if [ "$RESUME_MODEL_STATES_ONLY" -gt 0 ]; then
    EXTRA_ARGS+=" --resume_model_states_only"
fi
if [ $DATASET_RESHARD_ENABLED -gt 0 ]; then
    EXTRA_ARGS+=" --dataset_reshard_enabled --dataset_reshard_size $DATASET_RESHARD_SIZE"
fi
if [ "$CHECKPOINT_EVERY_ITER_ENABLED" -gt 0 ]; then
    EXTRA_ARGS+=" --checkpoint_every_iter_enabled"
fi
if [ "$CHECKPOINT_LOAD_DATASET" -gt 0 ]; then
    EXTRA_ARGS+=" --checkpoint_load_dataset"
fi
if [ "$CHECKPOINT_LOAD_DATASET_BUFFER" -gt 0 ]; then
    EXTRA_ARGS+=" --checkpoint_load_dataset_buffer"
fi
if [ "$CHECKPOINT_LOAD_OPTIMIZER" -gt 0 ]; then
    EXTRA_ARGS+=" --checkpoint_load_optimizer"
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
if [ "$SCALED_INIT_ENABLED" -gt 0 ]; then
    EXTRA_ARGS+=" --scaled_init"
fi
if [ "$BROADCAST_AND_AVERAGE_LOSS" -gt 0 ]; then
    EXTRA_ARGS+=" --broadcast_and_average_loss"
fi
if [ ! -z "$VAL_DATA_PATH" ]; then
    EXTRA_ARGS+=" --val_dataset_path $VAL_DATA_PATH"
fi

DP=$(($PROCESSES_PER_NODE * $NTASKS / $TP_DEGREE / $PP_DEGREE))
BS=$(($GBS / $DP))
# ACC_STEPS=$(($GBS / $MBS / $DP))
ACC_STEPS=1

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    # checkpoint compilation step
    STEPS_THIS_RUN=4
    CHECKPOINT_FREQUENCY=3
    CHECKPOINT_SAVED_FREQUENCY=-1
    OUTPUT_LOG=$LOG_DIR/log_compile-$NODEID.log
    # Haozheng fr:
    CMD_PREFIX="torchrun $DISTRIBUTED_ARGS"
else
    # OUTPUT_LOG=$SCRIPT_DIR/log_exe-$NODEID.log
    # Haozheng fr:
    # OUTPUT_LOG=$SCRIPT_DIR/log_exe-$(hostname).log
    # Haozheng fr:
    if [ $JOBTYPE == "FC" ]; then
        CMD_PREFIX="/usr/bin/python3 /mnt_out/fanhaozh/70_llama2/resilient_agent.py --max_restart=100 --nproc_per_node $PROCESSES_PER_NODE"  # wzam on 3/3/2024, increase max restart to 100
        OUTPUT_LOG=$LOG_DIR/log_exe-$(hostname).log
    else
        CMD_PREFIX="torchrun $DISTRIBUTED_ARGS"
        OUTPUT_LOG=$LOG_DIR/log_exe-$NODEID.log
    fi
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo USE_GRAD_ACC_HOOK=$USE_GRAD_ACC_HOOK
echo GBS=$GBS
echo BS=$BS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
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
echo CMD_PREFIX=$CMD_PREFIX
# skip store based barrier when creating process group
export TORCH_DIST_INIT_BARRIER=0

# Haozheng ckpt1: local output to speedup: --output_dir $HOME \
# Haozheng 0207 param norm:
$CMD_PREFIX \
    /70_llama2/i0124/tp_zero1_llama2_70b_hf_pretrain.py \
    --training_config $SCRIPT_DIR/70b_config.json \
    --dataset_path $DATA_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --pipeline_parallel_size $PP_DEGREE \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $BS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --checkpoint_frequency $CHECKPOINT_FREQUENCY \
    --checkpoint_saved_frequency $CHECKPOINT_SAVED_FREQUENCY \
    --output_dir $SCRIPT_DIR/output  \
    --sequence_parallel_enabled \
    --selective_checkpoint_enabled \
    --print_grad_norm \
    --print_param_norm \
    --constant_attention_mask \
    --tb_dir /mnt_out/tensorboard/zhuha/$SCRIPT_DIR_NAME \
    $EXTRA_ARGS |& tee $OUTPUT_LOG

    # --output_dir $SCRIPT_DIR/output \
    # --val_dataset_path $VAL_DATA_PATH \
    # --selective_checkpoint_enabled \
    # --print_grad_norm \
# --resume_ckpt_dir /mnt_out/fanhaozh/llama2/golden_test_load/resume_ckpt_dir_360 \

ret=${PIPESTATUS[0]}
echo "return code from torchrun: $ret"
exit $ret

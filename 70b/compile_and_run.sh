#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

#############################################
# wzam on 03/4/2024, add the following to avoid PyTorchStream write failed failure

# need the following on EKS only
# sudo mount -o remount,rw /sys
sudo lctl set_param 'osc.*.max_dirty_mb=64'
sudo sysctl -w net.core.somaxconn=65535

echo "instance-id: $(cat /sys/devices/virtual/dmi/id/board_asset_tag)"

#############################################
# process args

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_NAME=$(basename $SCRIPT_DIR)
USAGE="Usage: $0 [compile|run] [--local_cache_enabled]"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo $USAGE
    exit 1
fi

if [ $1 != "compile" ] && [ $1 != "run" ]; then
    echo $USAGE
    exit 1
fi

if [ $# -eq 2 ] && [ $2 != "--local_cache_enabled" ]; then
    echo $USAGE
    exit 1
fi

LOCAL_CACHE_ENABLED=0
if [ $# -eq 2 ] && [ $2 == "--local_cache_enabled" ]; then
    LOCAL_CACHE_ENABLED=1
fi

TASK=pretrain

echo "TASK = $TASK"
echo "LOCAL_CACHE_ENABLED = $LOCAL_CACHE_ENABLED"

#############################################
# set log dir

if [ -n "$JOB_NAME" ]; then
    # mstar job
    LOG_DIR=$SCRIPT_DIR/log_$JOB_NAME
elif [ -n "$SLURM_JOB_ID" ]; then
    # slurm job
    LOG_DIR=$SCRIPT_DIR/log_$SLURM_JOB_ID
else
    # default
    LOG_DIR=$SCRIPT_DIR/log_default
fi

echo "LOG_DIR = $LOG_DIR"

mkdir -p $LOG_DIR/

#############################################
# cluster scheduler adapter

if [ -v SLURM_NNODES ]
then
    echo "Slurm job with world size $SLURM_NNODES"
    JOBTYPE="SLURM"
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
elif [ -v OMPI_COMM_WORLD_RANK ]
then
    echo "MPIJob with world size $OMPI_COMM_WORLD_SIZE"
    JOBTYPE="MPI"
    NODEID=$OMPI_COMM_WORLD_RANK
    NTASKS=$OMPI_COMM_WORLD_SIZE
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
    fi
else
    # Single-node, non-SLURM, non-MPI runs
    echo "Single node run"
    NODEID=0
    NTASKS=1
fi

echo "JOBTYPE = $JOBTYPE"
echo "NTASKS = $NTASKS"
echo "NODEID = $NODEID"

#############################################
# Set log filename

if [ $JOBTYPE = "FC" ]; then
    echo "Fault controller job without NODEID"
    OUTPUT_LOG=$LOG_DIR/log_parallel_compile_$(hostname).log
    SANITY_CHECK_LOG=$LOG_DIR/log_sanity_check_$(hostname).log
    INSTALL_LOG=$LOG_DIR/log_install_$(hostname).log
else
    echo "Non-FC job with NODEID = $NODEID"
    OUTPUT_LOG=$LOG_DIR/log_parallel_compile_$NODEID.log
    SANITY_CHECK_LOG=$LOG_DIR/log_sanity_check_$NODEID.log
    INSTALL_LOG=$LOG_DIR/log_install_$NODEID.log
fi

#############################################

# install software and activate
VENV_PATH=/local_storage/venv/2_20_0_local
rm -rf $VENV_PATH
bash $SCRIPT_DIR/../common/install_2_20_0.sh $VENV_PATH |& tee $INSTALL_LOG
source $VENV_PATH/bin/activate

# sanity check
bash $SCRIPT_DIR/../common/sanity_check.sh |& tee $SANITY_CHECK_LOG

#############################################

# parallel compile and populate cache
PERSISTENT_CACHE_DIR=/fsx_out/guangtai/llama/cache2
if [ $1 = "compile" ]; then
    if [ $LOCAL_CACHE_ENABLED -gt 0 ]; then
        echo "parallel compile and populate by-rank cache"
        export NEURON_COMPILE_CACHE_URL=$PERSISTENT_CACHE_DIR/$NODEID
        mkdir -p $NEURON_COMPILE_CACHE_URL
        neuron_parallel_compile bash $SCRIPT_DIR/tp_zero1_llama2_70b_hf_pretrain.sh $TASK 3>&1 2>&1 | tee $OUTPUT_LOG
    else
        echo "parallel compile and populate unifeid cache"
        export NEURON_COMPILE_CACHE_URL=$PERSISTENT_CACHE_DIR
        mkdir -p $PERSISTENT_CACHE_DIR
        neuron_parallel_compile bash $SCRIPT_DIR/tp_zero1_llama2_70b_hf_pretrain.sh $TASK 3>&1 2>&1 | tee $OUTPUT_LOG
    fi
fi

# execute training and get loss curve
LOCAL_CACHE_DIR=/local_storage/cache/neuron_cache_shared
rm -rf $LOCAL_CACHE_DIR
mkdir -p $LOCAL_CACHE_DIR
if [ $1 = "run" ]; then
    echo "generate runtime cache with LOCAL_CACHE_ENABLED=$LOCAL_CACHE_ENABLED"
    if [ $LOCAL_CACHE_ENABLED -gt 0 ]; then
        echo "merge cache from shared directory with NTASKS=$NTASKS"
        MAX_RANK=$(($NTASKS - 1))
        for i in `seq 0  $MAX_RANK`; do
            $SCRIPT_DIR/sync_cache_dir.py $PERSISTENT_CACHE_DIR/$i $LOCAL_CACHE_DIR
        done
    else
        echo "copy cache from shared directory"
        cp -r $PERSISTENT_CACHE_DIR/. $LOCAL_CACHE_DIR
    fi
    echo "execute training and get loss curve"
    export NEURON_COMPILE_CACHE_URL=$LOCAL_CACHE_DIR
    OUTPUT_LOG=$LOG_DIR/log_parallel_exe_$NODEID.log
    bash $SCRIPT_DIR/tp_zero1_llama2_70b_hf_pretrain.sh $TASK 3>&1 2>&1 | tee $OUTPUT_LOG
fi

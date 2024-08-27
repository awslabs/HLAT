#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -x

# wzam on 03/4/2024, add the following to avoid PyTorchStream write failed failure
mount -o remount,rw /sys
lctl set_param 'osc.*.max_dirty_mb=64'
sysctl -w net.core.somaxconn=65535

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -z $JOB_NAME ]; then
    LOG_DIR=$SCRIPT_DIR/log_default
else
    LOG_DIR=$SCRIPT_DIR/log_$JOB_NAME
fi

echo "LOG_DIR = $LOG_DIR"

mkdir -p $LOG_DIR/

# Haozheng fc:
if [ -z $RANK ]; then
    echo "RANK is not set"
    PROCESSES_PER_NODE=32
    if [ "$(( $WORLD_SIZE % $PROCESSES_PER_NODE ))" -ne 0 ]; then
        exit 1
    fi
    NTASKS=$(($WORLD_SIZE / $PROCESSES_PER_NODE))
    OUTPUT_LOG=$LOG_DIR/log_parallel_compile_$(hostname).log
    PATCH_LOG=$LOG_DIR/log_patch_$(hostname).log
else
    echo "RANK is set: RANK = $RANK"
    NTASKS=$WORLD_SIZE
    OUTPUT_LOG=$LOG_DIR/log_parallel_compile_$RANK.log
    PATCH_LOG=$LOG_DIR/log_patch_$RANK.log
fi

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

# patch neuron software
bash $SCRIPT_DIR/patch.sh |& tee $PATCH_LOG
# check
echo "pip list | grep neuron"
pip list | grep neuron
echo "apt list | grep neuron"
apt list | grep neuron

# check zombie
lsof -i:44000
lsof -i:23456

# parallel compile and populate cache
PERSISTENT_CACHE_DIR=/mnt_out/fanhaozh/70_llama2/cache/n0312
if [ $1 = "compile" ]; then
    if [ $LOCAL_CACHE_ENABLED -gt 0 ]; then
        echo "parallel compile and populate by-rank cache"
        export NEURON_COMPILE_CACHE_URL=$PERSISTENT_CACHE_DIR/$RANK
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
LOCAL_CACHE_DIR=/cache/neuron_cache_shared
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
    bash $SCRIPT_DIR/tp_zero1_llama2_70b_hf_pretrain.sh $TASK
fi

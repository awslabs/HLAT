set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_LOG=$SCRIPT_DIR/log_parallel_compile_$RANK.log
PATCH_LOG=$SCRIPT_DIR/log_patch_$RANK.log

if [ $# -lt 1 ]; then
    echo "Usage: $0 [compile|run]"
    exit 1
fi

if [ $1 != "compile" ] && [ $1 != "run" ]; then
    echo "Usage: $0 [compile|run]"
    exit 1
fi

TASK=pretrain

# patch neuron software
bash $SCRIPT_DIR/patch.sh |& tee $PATCH_LOG
# check
echo "pip list | grep neuron"
pip list | grep neuron
echo "apt list | grep neuron"
apt list | grep neuron

# parallel compile and populate cache
PERSISTENT_CACHE_DIR=/mnt_out/guangtai/llama2/cache/n1208
if [ $1 = "compile" ]; then
    echo "parallel compile and populate cache"
    export NEURON_COMPILE_CACHE_URL=$PERSISTENT_CACHE_DIR
    mkdir -p $PERSISTENT_CACHE_DIR
    neuron_parallel_compile bash $SCRIPT_DIR/tp_zero1_llama2_7b_hf_pretrain.sh $TASK |& tee $OUTPUT_LOG
fi

# execute training and get loss curve
if [ $1 = "run" ]; then
    echo "execute training and get loss curve"
    mkdir -p /cache/neuron_cache_shared
    cp -r $PERSISTENT_CACHE_DIR/. /cache/neuron_cache_shared/
    export NEURON_COMPILE_CACHE_URL="/cache/neuron_cache_shared"
    bash $SCRIPT_DIR/tp_zero1_llama2_7b_hf_pretrain.sh $TASK
fi

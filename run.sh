#!/bin/bash

source /etc/profile.d/modules.sh
module load python/3.10 cuda/11.7 cudnn/8.6
source ~/abci-llm-distributed-training-hackathon-01/.venv/bin/activate

export HF_HOME="/scratch/$(whoami)/.cache/huggingface/"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_API_KEY=XXXXXXXXXX
# export CUDA_LAUNCH_BLOCKING=1

#
# 以下のようなエラーが出るために必要:
#
# [E ProcessGroupNCCL.cpp:456] Some NCCL operations have failed or timed out. Due to the asynchronous nature of CUDA kernels, subsequent GPU operations might run on corrupted/incomplete data.
# [E ProcessGroupNCCL.cpp:461] To avoid data inconsistency, we are taking the entire process down.
# terminate called after throwing an instance of 'std::runtime_error'
#   what():  [Rank 124] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=12626, OpType=_ALLGATHER_BASE, Timeout(ms)=600000) ran for 605485 milliseconds before timing out.
#
export NCCL_ASYNC_ERROR_HANDLING=1

readonly training_script="$HOME/abci-llm-distributed-training-hackathon-01/llm-foundry/scripts/train/train.py"
readonly yaml_base_dir="$HOME/abci-llm-distributed-training-hackathon-01/yamls"

export PYTHONUNBUFFERED=1

function run() {
    local world_size=$1
    local rank=$2
    local addr=$3
    local port=$4
    local params=$5

    local training_script_args="$yaml_base_dir/mpt-$params-shunk031.yaml"

    composer \
        --verbose \
        --world_size $world_size \
        --node_rank $rank \
        --master_addr $addr \
        --master_port $port \
        $training_script \
        $training_script_args
}

run $@

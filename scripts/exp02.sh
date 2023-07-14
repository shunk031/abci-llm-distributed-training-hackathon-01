#!/bin/bash
#$ -l rt_F=8
#$ -l USE_SSH=1
#$ -v SSH_PORT=2200
#$ -j y
#$ -N exp02
#$ -o logs/
#$ -e errors/
#$ -cwd

set -eux

source /etc/profile.d/modules.sh
module load python/3.10 cuda/11.7 cudnn/8.6
source ~/abci-llm-distributed-training-hackathon-01/.venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

function run_slave_node() {
    local world_size=32
    local rank=1
    local master_addr=$(hostname)
    local ssh_port=2200
    local composer_port=2221
    local params="7b"

    cat ${SGE_JOB_HOSTLIST} | grep -v $HOSTNAME | while read node; do
        ssh -p $ssh_port -q $node \
            bash ~/abci-llm-distributed-training-hackathon-01/run.sh $world_size $rank $master_addr $composer_port $params &
        rank=$((rank+1))
    done
}

function run_master_node() {
    local world_size=32
    local rank=0
    local master_addr=0.0.0.0
    local composer_port=2221
    local params="7b"

    bash ~/abci-llm-distributed-training-hackathon-01/run.sh $world_size $rank $master_addr $composer_port $params
}

function main() {
    run_slave_node
    run_master_node

    wait
}

main

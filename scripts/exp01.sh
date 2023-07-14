#!/bin/bash
#$ -l rt_F=4
#$ -l USE_SSH=1
#$ -v SSH_PORT=2200
#$ -j y
#$ -N exp01
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.10 cuda/11.7 cudnn/8.6
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Using 4 node each with 4 devices
# Total world size: 16
# 以下の設定方法は違います。exp02.sh 以降を参考にしてください
composer --world_size 16 --node_rank 0 --master_addr 0.0.0.0 --master_port 2200 llm-foundry/scripts/train/train.py llm-foundry/scripts/train/yamls/finetune/mpt-7b-shunk031.yaml

composer --world_size 16 --node_rank 1 --master_addr 0.0.0.0 --master_port 2200 llm-foundry/scripts/train/train.py llm-foundry/scripts/train/yamls/finetune/mpt-7b-shunk031.yaml

composer --world_size 16 --node_rank 2 --master_addr 0.0.0.0 --master_port 2200 llm-foundry/scripts/train/train.py llm-foundry/scripts/train/yamls/finetune/mpt-7b-shunk031.yaml

composer --world_size 16 --node_rank 3 --master_addr 0.0.0.0 --master_port 2200 llm-foundry/scripts/train/train.py llm-foundry/scripts/train/yamls/finetune/mpt-7b-shunk031.yaml

#!/bin/bash
#$ -l rt_C.small=1
#$ -j y
#$ -N exp04
#$ -o logs/
#$ -cwd

#
# batch job script for downloading the MPT-30B
#

source /etc/profile.d/modules.sh
module load python/3.10 cuda/11.7 cudnn/8.6
source .venv/bin/activate

export HF_HOME=/scratch/$(whoami)/.cache/huggingface/
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Using 4 node each with 4 devices
# Total world size: 16
composer llm-foundry/scripts/train/train.py llm-foundry/scripts/train/yamls/finetune/mpt-30b-shunk031.yaml

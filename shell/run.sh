#!/bin/bash
source ~/.bashrc
micromamba activate mind
cd ~/workspace/CounselEval

export VLLM_ALLOW_LONG_MAX_MODEL_LEN='1'
arg1=${1}

python adhoc/main.py --start_idx ${arg1}
#!/usr/bin/env bash

# ***********************************************
#      Filename: run.sh
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-03-12 16:58:38
# Last Modified: Year-month-day
# ***********************************************

set -e

if [[ $# -lt 7 ]]; then
    echo "Usage: bash $0 <model> <model_path> <dist_init_addr> <node_rank> <nnodes> <tp> *args **kwargs"
    exit
fi

model=$1 && shift
model_path=$1 && shift
dist_init_addr=$1 && shift
node_rank=$1 && shift
nnodes=$1 && shift
tp=$1 && shift
args_kwargs=$*

random_seed=1234

export GLOO_SOCKET_IFNAME=eth0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1 \
    SGL_ENABLE_JIT_DEEPGEMM=false
# dist_init_addr="172.31.0.3:1234"
    # --disable-cuda-graph \

python -u $(dirname $0)/chat.py \
    --model_name $model \
    --model-path $model_path \
    --random-seed ${random_seed} \
    --dist-init-addr $dist_init_addr \
    --tp $tp \
    --nnodes $nnodes \
    --node-rank ${node_rank} \
    --trust-remote-code \
    --disable-radix-cache \
    --enable-cache-report \
    --log-level debug \
    ${args_kwargs}
    # --attention-backend triton \

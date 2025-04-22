#!/usr/bin/env bash

# ***********************************************
#      Filename: api/server.sh
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-02-18 11:45:43
# Last Modified: Year-month-day
# ***********************************************

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model_path> <node_rank>"
fi

model_path=$1 && shift
node_rank=$1 && shift

# model_path=/ssd01/models/DeepSeek-V3/
# model_path=/ssd01/models/DeepSeek-R1/
dist_init_addr="172.31.0.3:1234"

export GLOO_SOCKET_IFNAME=eth0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_LAUNCH_BLOCKING=1 \
    # TORCH_USE_CUDA_DSA=True \
    # TRITON_DEBUG=True

python -u -m sglang.launch_server \
    --model-path $model_path \
    --tp 16 \
    --dist-init-addr $dist_init_addr \
    --nnodes 2 \
    --node-rank ${node_rank} \
    --trust-remote-code \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-cache-report \
    --chunked-prefill-size 2048 \
    $* \
    # --mem-fraction-static 0.7 \
    # --watchdog-timeout 360000 \

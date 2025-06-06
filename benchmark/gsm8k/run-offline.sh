#!/usr/bin/env bash

# ***********************************************
#      Filename: ../mmlu/run.sh
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-02-24 13:45:09
# Last Modified: Year-month-day
# ***********************************************

set -e

if [[ $# -lt 6 ]]; then
    echo "Usage: bash $0 <model_path> <outdir> <dist_init_addr> <node_rank> <nnodes> <tp> *args **kwargs"
    exit
fi

model_path=$1 && shift
outdir=$1 && shift
dist_init_addr=$1 && shift
node_rank=$1 && shift
nnodes=$1 && shift
tp=$1 && shift

random_seed=1234

export GLOO_SOCKET_IFNAME=eth0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1

mkdir -p $outdir
outdir=$(realpath $outdir)
model_path=$(realpath $model_path)

cd $(dirname $0)

# bash download_data.sh
nohup \
    python3 -u bench_sglang-offline.py \
        --num-shots 8 \
        --num-questions 1319 \
        --save-dir $outdir \
        --model-path $model_path \
        --random-seed ${random_seed} \
        --dist-init-addr $dist_init_addr \
        --tp $tp \
        --nnodes $nnodes \
        --node-rank ${node_rank} \
        --trust-remote-code \
        --disable-radix-cache \
        --enable-cache-report \
        $* \
>> $outdir/log &
        # --disable-cuda-graph \
echo "Pls refer to $outdir/log"
# python3 bench_sglang.py --nsub 1 --ntrain 0 # --parallel 2000 $*

cd -

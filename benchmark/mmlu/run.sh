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

cd $(dirname $0)

# bash download_data.sh
python3 bench_sglang.py --nsub 100 --ntrain 5 --parallel 2000 $*
# python3 bench_sglang.py --nsub 1 --ntrain 0 # --parallel 2000 $*

cd -

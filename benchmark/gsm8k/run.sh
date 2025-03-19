#!/usr/bin/env bash

# ***********************************************
#      Filename: run.sh
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-02-24 13:44:19
# Last Modified: Year-month-day
# ***********************************************

set -e

python3 $(dirname $0)/bench_sglang.py --num-shots 8 --num-questions 1319 --parallel 1319

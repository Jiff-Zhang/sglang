#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ***********************************************
#      Filename: api/plot.py
#        Author: jiff
#         Email: Jiff_Zh@163.com
#   Description: --
#        Create: 2025-03-07 14:52:04
# Last Modified: Year-month-day
# ***********************************************

import os
from glob import glob
from sparseopt.utils.plot import plot_3d
from sparseopt.attns.act_sparse_nbits import MFSparseNbits
import torch
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def plot_single(pt_file, image_file):
    tool_per_bank = MFSparseNbits(
        sparsity=0.,
        bank_size=64,
        dim=-1,
        num_bits={"high": 8, "low": 0},
        mode="per_bank",
        quant_symmetric=True,
        quant_masked=True,
        quant_hollow=False,
    )
    tool_per_group = MFSparseNbits(
        sparsity=0.,
        bank_size=64,
        dim=-1,
        num_bits={"high": 8, "low": 0},
        mode="per_group",
        quant_symmetric=False,
        quant_masked=True,
        quant_hollow=False,
    )
    k_cache, v_cache = torch.load(pt_file, map_location='cpu')
    k_cache, v_cache = k_cache[:512], v_cache[:512]
    k_cache_per_bank = tool_per_bank(k_cache)
    v_cache_per_bank = tool_per_bank(v_cache)
    k_cache_per_group = tool_per_group(k_cache)
    v_cache_per_group = tool_per_group(v_cache)
    infos = [
        {
            f"k_cache_rope positive\ndense": k_cache[:, v_cache.size(-1):].cpu().float().clamp(min=0).abs().numpy(),
            f"k_cache_rope negative\ndense": k_cache[:, v_cache.size(-1):].cpu().float().clamp(max=0).abs().numpy(),
            f"v_cache positive\ndense": v_cache.cpu().float().clamp(min=0).abs().numpy(),
            f"v_cache negative\ndense": v_cache.cpu().float().clamp(max=0).abs().numpy(),
        },
        {
            f"k_cache_rope positive\nper_bank": k_cache_per_bank[:, v_cache.size(-1):].cpu().float().clamp(min=0).abs().numpy(),
            f"k_cache_rope negative\nper_bank": k_cache_per_bank[:, v_cache.size(-1):].cpu().float().clamp(max=0).abs().numpy(),
            f"v_cache positive\nper_bank": v_cache_per_bank.cpu().float().clamp(min=0).abs().numpy(),
            f"v_cache negative\nper_bank": v_cache_per_bank.cpu().float().clamp(max=0).abs().numpy(),
        },
        {
            f"k_cache_rope positive\nper_group": k_cache_per_group[:, v_cache.size(-1):].cpu().float().clamp(min=0).abs().numpy(),
            f"k_cache_rope negative\nper_group": k_cache_per_group[:, v_cache.size(-1):].cpu().float().clamp(max=0).abs().numpy(),
            f"v_cache positive\nper_group": v_cache_per_group.cpu().float().clamp(min=0).abs().numpy(),
            f"v_cache negative\nper_group": v_cache_per_group.cpu().float().clamp(max=0).abs().numpy(),
        },
    ]
    plot_3d(infos, image_file, image_size=5)

if __name__ == '__main__':
    info_dir = '/ssd01/workspace/sglang/exp/figs'
    executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()//2)

    futures = list()
    for filename in sorted(glob(os.path.join(info_dir, '*.pt'))):
        futures.append(
            executor.submit(
                plot_single, filename, filename.replace('.pt', '.png')
            )
        )

    # for future in as_completed(tqdm(futures)):
    for future in as_completed(futures):
        future.result()
import os
import torch
import logging
from einops import rearrange
from typing import Optional, Tuple, List, Mapping, Dict, Callable

import logging
from sglang.srt.layers.dp_attention import get_attention_tp_rank
from sglang.srt.distributed.communication_op import tensor_model_parallel_gather, tensor_model_parallel_all_gather
from einops import rearrange
from collections import defaultdict

logger = logging.getLogger(__name__)

saved_names = defaultdict(list)
def save(
    x: torch.Tensor, # [S, ..., D]
    name: str,
    layer_id: int,
    gather: bool=True,
    dim: int=-1,
    nt: int=1, # number of tensors
):
    # return
    # if layer_id is not None and layer_id > 3:
    #     return

    ori_shape = x.shape

    if gather:
        # x = x.clone()
        if dim != -1 or dim != x.dim() - 1:
            x = x.transpose(dim, -1)
        if nt > 1:
            x = rearrange(x, "... (nt d) -> ... nt d", nt=nt)
        x = tensor_model_parallel_all_gather(x.contiguous(), dim=-1)
        if nt > 1:
            x = rearrange(x, "... nt d -> ... (nt d)")
        if dim != -1 or dim != x.dim() - 1:
            x = x.transpose(dim, -1)

    if get_attention_tp_rank() != 0:
        return

    logger.debug(f"ori_shape: {list(ori_shape)}, x.shape: {list(x.shape)}")

    out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-R1"
    # out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus/official-save"
    # out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus/MF-Int8-smooth-save"
    # out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus/MF-W8xH8L3-save"
    # out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-V3.1-Terminus/MF-Linear_WInt8-MOE_W8xH8L3-save"
    if layer_id is not None:
        out_dir = os.path.join(out_dir, f"layer{layer_id:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # prefix = "prefill" if x.size(0) > 1 else "decode"
    if name in saved_names[layer_id]:
        prefix = "decode"
    else:
        prefix = "prefill"
        saved_names[layer_id].append(name)
    pth_name = f"{prefix}-{name}.pt"
    out_file = os.path.join(out_dir, pth_name)
    logging.debug(f"Saving tensor with shape {x.shape} to {out_file}...")
    torch.save(x, out_file)
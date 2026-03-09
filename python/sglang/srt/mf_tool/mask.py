import math
import torch
import logging
from einops import rearrange
from typing import Optional, Tuple, List, Mapping, Dict, Callable

def generate_mask(
    mask_id: torch.Tensor, # [O, I, N]
    block_size: Tuple[int, int], # [BO, BI]
    dtype: torch.dtype,
):
    assert mask_id.dim() == 3, \
        f"Only support 3D mask_id, but got {mask_id.dim()}"
    assert len(block_size) == 2, \
        f"Only support 2D block_size, but got {block_size}"
    mask = torch.zeros(
        (*mask_id.shape[:-1], block_size[0] * block_size[1]),
        dtype=dtype,
        device=mask_id.device
    )
    mask.scatter_(dim=-1, index=mask_id.to(torch.int32), value=1)
    mask = rearrange(
        mask, "O I (BI BO) -> (O BO) (I BI)", BO=block_size[0], BI=block_size[1]
    )
    return mask
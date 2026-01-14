import os
import math
import torch
from sparseopt.attns.act_sparse_nbits import MFSparseNbits
from sparseopt.attns.retriever import TokenSparseRetriever
import logging
from sglang.srt.layers.dp_attention import get_attention_tp_rank
from sglang.srt.distributed.communication_op import tensor_model_parallel_gather, tensor_model_parallel_all_gather
from einops import rearrange
from collections import defaultdict

logger = logging.getLogger(__name__)

def quantize(
    x: torch.Tensor, # [S, H, D]
    tool: MFSparseNbits,
):
    if x.numel() == 0:
        return x
    
    seq_len = x.size(0)
    if tool.is_seq_rely and tool.bank_size > 0:
        quant_len = math.floor(seq_len / tool.bank_size) * tool.bank_size
    else:
        quant_len = seq_len
    return torch.cat(
        [
            tool(x[:quant_len].transpose(0, -2)).transpose(0, -2),
            x[quant_len:]
        ],
        dim=0
    )

saved_names = defaultdict(list)
def save(
    x: torch.Tensor, # [S, ..., D]
    name: str,
    layer_id: int,
    gather: bool=True,
    dim: int=-1,
    nt: int=1, # number of tensors
):
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
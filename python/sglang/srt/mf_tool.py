import os
import math
import torch
from sparseopt.attns.act_sparse_nbits import MFSparseNbits
from sparseopt.attns.retriever import TokenSparseRetriever
import logging
from sglang.srt.layers.dp_attention import get_attention_tp_rank

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

def save(
    x: torch.Tensor, # [S, ..., D]
    name: str,
):
    if get_attention_tp_rank() != 0:
        return

    out_dir = "/ssd01/workspace/sglang-n/exp/data/DeepSeek-R1"
    os.makedirs(out_dir, exist_ok=True)
    prefix = "prefill" if x.size(0) > 1 else "decode"
    out_file = os.path.join(out_dir, f"{prefix}-{name}.pt")
    logging.debug(f"Saving tensor with shape {x.shape} to {out_file}...")
    torch.save(x, out_file)
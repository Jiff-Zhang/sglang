import math
import torch
from sparseopt.attns.act_sparse_nbits import MFSparseNbits
from sparseopt.attns.retriever import TokenSparseRetriever

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
from .retriever import register_mf_tool, quantize, MFSparseNbits, TokenSparseRetriever
from sglang.srt.layers.dp_attention import is_dp_attention_enabled, get_attention_dp_rank, get_attention_tp_rank
from .mask import generate_mask
from .save import save

from .fp32_to_fp24 import fp32_rne_to_fp24_torch, fp32_trunc_to_fp24_torch
# fp32_to_fp24 = fp32_rne_to_fp24_torch
fp32_to_fp24 = lambda x: x

def is_logging_enabled():
    return get_attention_tp_rank() == 0
    if is_dp_attention_enabled():
        return get_attention_dp_rank() == 0
    else:
        return get_attention_tp_rank() == 0
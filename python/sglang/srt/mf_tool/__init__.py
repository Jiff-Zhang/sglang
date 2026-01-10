from .retriever import register_mf_tool, quantize, MFSparseNbits, TokenSparseRetriever
from .mask import generate_mask

from .fp32_to_fp24 import fp32_rne_to_fp24_torch, fp32_trunc_to_fp24_torch
# fp32_to_fp24 = fp32_rne_to_fp24_torch
fp32_to_fp24 = lambda x: x

from sglang.srt.logger import is_logging_enabled
import math
import torch
import logging
from einops import rearrange
from typing import Optional, Tuple, List, Mapping, Dict, Callable
from transformers import PretrainedConfig

from sparseopt.attns.act_sparse_nbits import MFSparseNbits
from sparseopt.attns.retriever import TokenSparseRetriever

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.logger import is_logging_enabled

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

default_mf_config = {"modes": [], "active": False}
def register_mf_tool(
    layer: RadixAttention,
    config: PretrainedConfig,
):
    assert isinstance(layer, RadixAttention), f"layer must be an instance of RadixAttention, got {type(layer)}"
    
    # return
    
    """
    # TODO: Deprecated
    bank_size = 64
    modes = [
        # "prefill_quant",    # whether to quantize key/value/query/score when prefill
        # "cache_quant",      # whether to quantize cached key & value
        "prefill_retrieve", # whether to quantize cached key and retrieve tokens when prefill
        "retrieve",         # whether to quantize cached key and retrieve tokens when decode
    ]
    per_bank_tool = MFSparseNbits(
        bank_size=bank_size,
        sparsity=0.,
        quant_mode="per_bank",
        dtypes={"high": "int8", "low": "zero"},
        quant_symmetric=True,
        quant_masked=True,
        hardware=True
    )
    per_group_tool = MFSparseNbits(
        bank_size=bank_size,
        sparsity=0.,
        quant_mode="per_group",
        dtypes={"high": "int8", "low": "zero"},
        quant_symmetric=True,
        quant_masked=True,
        hardware=True
    )
    per_block_tool = MFSparseNbits(
        bank_size=bank_size,
        sparsity=0.,
        quant_mode="per_block",
        dtypes={"high": "int8", "low": "zero"},
        quant_symmetric=True,
        quant_masked=True,
        hardware=True
    )

    q_tool = per_bank_tool
    k_tool = per_bank_tool
    v_tool = per_group_tool
    k_cache_tool = per_bank_tool
    v_cache_tool = per_group_tool
    
    retriever = TokenSparseRetriever(
        active=True,
        # retain_size=2048,
        # retain_size=4096,
        retain_size=8192,
        chunk_size=1,
        recent_size=64,
        bank_size=bank_size,
        sparsity=0.875,
        dtypes={"high": "int4", "low": "zero"},
        sparse_mode="per_bank",
        quant_mode="per_bank",
        q_dtypes={"high": "int8", "low": "zero"},
        q_sparse_mode="per_bank",
        q_quant_mode="per_bank",
        # share=False,
        share=True,
        share_mode="mean",
        # share_mode="max",
        mean_trick=False,
        # mean_trick=True,
        # softmax_scale=False,
        softmax_scale=True,
        softmax_version="v0",
        # topk_version='v0',
        topk_version='v2',
        # topk_version='v2.1',
        # topk_version='v2.2',
        # topk_version='v2.3',
        topk_chunk_size=64,
        # # topk_version='v3',
        # # topk_version='v3.1',
        # topk_version='v3.2',
        # # topk_mini_k=256,
        # topk_mini_k=-1,
        # topk_chunk_size=4096,
        all_reduce=num_kv_heads == 1 and not is_dp_attention_enabled(),
        auto_reset=False,
        qk_scaling=layer.scaling, # TODO: unmark code for scaling
    )
    """
    
    mf_config = getattr(config, "mf_config", default_mf_config)
    if not mf_config["active"]:
        return

    supported_modes = {
        "prefill_quant",
        "cache_quant",
        "prefill_retrieve",
        "retrieve"
    }
    modes = mf_config["modes"]
    assert set(modes).issubset(supported_modes), \
        f"unsupported modes: {modes}, supported modes: {supported_modes}"

    q_tool = MFSparseNbits(**mf_config["q_tool"])
    k_tool = MFSparseNbits(**mf_config["k_tool"])
    v_tool = MFSparseNbits(**mf_config["v_tool"])
    k_cache_tool = MFSparseNbits(**mf_config["k_cache_tool"])
    v_cache_tool = MFSparseNbits(**mf_config["v_cache_tool"])
    retriever = TokenSparseRetriever(
        **mf_config["retriever"], qk_scaling=layer.scaling
    )

    setattr(layer, "modes", modes)
    setattr(layer, "q_tool", q_tool)
    setattr(layer, "k_tool", k_tool)
    setattr(layer, "v_tool", v_tool)
    setattr(layer, "k_cache_tool", k_cache_tool)
    setattr(layer, "v_cache_tool", v_cache_tool)
    setattr(layer, "retriever", retriever)
    if is_logging_enabled() and layer.layer_id == 0:
        logger.debug(
            f"<register_mf_tool> \n"
            f"\t#modes: {modes}\n"
            f"\t#retriever: {retriever.stats_str}\n"
            f"\t#q_tool: {q_tool.stats_str}\n"
            f"\t#k_tool: {k_tool.stats_str}\n"
            f"\t#v_tool: {v_tool.stats_str}\n"
            f"\t#k_cache_tool: {k_cache_tool.stats_str}\n"
            f"\t#v_cache_tool: {v_cache_tool.stats_str}"
        )